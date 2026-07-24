"""
core/dispatcher.py
──────────────────
Platform-agnostic message routing extracted from main.py.

Covers:
  • handle_unregistered_user  — send registration link to unknown chat users
  • _run_example_validation   — background: validate KB SQL against live DB
  • _run_log_harvest          — background: harvest query log into examples
  • dispatch()                — route a PlatformEvent to handle_query or commands
"""

from __future__ import annotations

import asyncio
import logging
import re

import store
from gateway import PlatformEvent
from fastapi import BackgroundTasks
from core.pipeline_context import get_state, get_client_db, get_portal_base
from core.pipeline_helpers import _looks_like_new_query
from core.clarification import (
    get_pending, clear_pending, combine_with_clarification,
    resolve_option_text, was_recently_expired, acknowledge_recently_expired,
)
from core.llm import is_ddl_attempt, _DDL_USER_MESSAGE, llm_complete, resolve_provider

log = logging.getLogger("querybot")

_HELP = (
    "*QueryBot* 🤖\n\nAsk any data question:\n"
    "  • _What is my total revenue this month?_\n"
    "  • _Show top 10 customers by value_\n"
    "  • _How many records were created last week?_\n\n"
    "*Commands:* `help` · `status` · `whoami`"
)

_ABOUT = (
    "*QueryBot* — Your AI-powered data assistant\n\n"
    "I connect directly to your business database and answer plain-English "
    "questions with live data. No dashboards, no filters — just ask.\n\n"
    "*What I can do:*\n"
    "  • Answer questions about revenue, sales, customers, inventory, and more\n"
    "  • Show trends over any time period — today, this month, last quarter\n"
    "  • Compare values across products, regions, customers, or teams\n"
    "  • Calculate KPIs and business metrics (margin %, days to pay, etc.)\n"
    "  • Rank and filter — top 10 customers, lowest performing products\n"
    "  • Produce charts automatically for visual results\n\n"
    "*Example questions:*\n"
    "  • _What is our gross margin percentage this year?_\n"
    "  • _Show revenue and COGS by customer_\n"
    "  • _Which products had the most sales last month?_\n"
    "  • _What is the average days to pay for each customer?_\n\n"
    "Just type your question in plain English and I'll query the database for you.\n"
    "Type `help` for commands or `status` to check your connection."
)

_OFF_TOPIC_REPLY = (
    "I'm QueryBot — I only answer questions about your business data.\n\n"
    "That question is outside my scope. Try asking me something like:\n"
    "  • _What is our revenue this month?_\n"
    "  • _Show top customers by sales_\n"
    "  • _What is our gross margin percentage this year?_\n"
    "  • _Which products had the highest COGS last quarter?_\n\n"
    "Type `help` to see more examples."
)

# Patterns that signal the user is asking about the bot itself rather than data
_ABOUT_RE = re.compile(
    r"\b("
    r"who are you|what are you|tell me about (your)?self|about you(rself)?"
    r"|what (can|do) you do|what('s| is) your (purpose|function|role|job)"
    r"|what (are )?your capabilities|what (can|could) (i|we) ask (you)?"
    r"|how (can|do) you help|how do(es)? (this|querybot|the bot) work"
    r"|what is querybot|what('s| is) querybot|querybot (features?|capabilities?)"
    r"|are you (a )?bot|are you (an )?ai|what kind of (bot|assistant) are you"
    r"|what questions? (can|should) (i|we) ask"
    r")\b",
    re.IGNORECASE,
)

_DATA_REQUEST_ACTION_RE = re.compile(
    r"^\s*(?:please\s+)?(?:show|list|find|retrieve|display|give|count|calculate|"
    r"compute|compare|analy[sz]e|summari[sz]e|rank|identify)\b",
    re.IGNORECASE,
)
_DATA_REQUEST_SHAPE_RE = re.compile(
    r"\b(?:all|top|bottom|total|count|number|average|trend|records?|rows?|"
    r"fields?|columns?|include|with|where|whose|who\s+have|by|per|for\s+each|each)\b",
    re.IGNORECASE,
)


def _looks_like_data_request(text: str) -> bool:
    """Recognize explicit record retrieval without making policy decisions."""
    value = (text or "").strip()
    return bool(
        _DATA_REQUEST_ACTION_RE.search(value)
        and _DATA_REQUEST_SHAPE_RE.search(value)
    )


# ── Off-topic classifier (LLM-based, dynamic) ────────────────────────────────

def _build_analyst_context(account_id: str, client_row: dict) -> str:
    """
    Assemble a compact metadata string for the conversational analyst:
    business description, industry, table summary, and metric list.
    Never includes real data rows — safe for regulated tenants.
    """
    parts: list[str] = []

    # Business description (fixed key: business_desc, not business_description)
    biz = str((client_row or {}).get("business_desc") or "").strip()
    if biz:
        parts.append(f"Business: {biz[:600]}")

    # Industry from compliance profile
    try:
        profile = store.get_compliance_profile(account_id)
        industry = str(profile.get("industry") or "").strip()
        if industry:
            parts.append(f"Industry: {industry}")
    except Exception:
        pass

    # Table summary — schema → count from KB state_data known tables
    try:
        from core.pipeline_context import get_state
        state = get_state(account_id)
        known_tables = state.get("known_tables") or []
        if known_tables:
            schemas: dict[str, int] = {}
            for fqn in known_tables:
                p = str(fqn).split(".")
                schema = p[-2] if len(p) >= 2 else "DEFAULT"
                schemas[schema] = schemas.get(schema, 0) + 1
            table_summary = "; ".join(
                f"{s} ({c} table{'s' if c != 1 else ''})"
                for s, c in sorted(schemas.items())
            )
            parts.append(f"Available schemas: {table_summary}")
    except Exception:
        pass

    # Metric list — names + short descriptions, capped at 15
    try:
        metric_lines: list[str] = []
        for metric in store.list_metrics(account_id):
            if not metric.get("is_active", 1):
                continue
            name = str(metric.get("name") or "").strip()
            desc = str(metric.get("description") or "").strip()
            if name:
                metric_lines.append(f"{name}" + (f" — {desc[:80]}" if desc else ""))
            if len(metric_lines) >= 15:
                break
        if metric_lines:
            parts.append("Defined metrics: " + ", ".join(metric_lines[:15]))
    except Exception:
        pass

    return "\n".join(parts)


# Sentinel returned by the LLM when the message is a genuine data query
_PROCEED_TO_QUERY = "PROCEED_TO_QUERY"


async def _generate_analyst_reply(text: str, account_id: str, client_row: dict) -> str | None:
    """
    Dynamic conversational analyst — replaces the static _ABOUT / _OFF_TOPIC_REPLY blocks.

    Returns:
      None          — message is a genuine data request; fall through to SQL pipeline.
      reply string  — a capability/meta/off-topic answer; send this directly.

    Fails open (returns None) on any error so a misconfigured LLM never blocks queries.
    Wraps the LLM call in llm_audit_scope so audit rows are written (previously missing).
    Only reasons over metadata (business_desc, industry, table/metric names) — never over
    real data rows, so it is safe for regulated tenants.
    """
    # Fast-path: obvious data requests skip the LLM call entirely
    if _looks_like_data_request(text):
        return None

    try:
        from core.llm_audit import llm_audit_scope
        provider, model, api_key, extra = resolve_provider(client_row, purpose="query")
        context = _build_analyst_context(account_id, client_row)
        context_block = f"\n\nWorkspace context:\n{context}" if context else ""

        system = (
            "You are QueryBot's conversational analyst. You answer questions about "
            "QueryBot's capabilities and what data this workspace can provide — using "
            "ONLY the workspace context supplied below. Never invent tables, metrics, or "
            "data values. Never claim to show real numbers.\n"
            "If the message is clearly a specific data retrieval request (asking for "
            "actual figures, records, trends, or comparisons from the database), "
            f"reply with exactly: {_PROCEED_TO_QUERY}\n"
            "Otherwise reply in 2-4 sentences: what QueryBot can help with in this "
            "workspace, referencing the real metrics and schemas listed below."
            f"{context_block}"
        )
        with llm_audit_scope(
            account_id=account_id,
            component="conversational_analyst",
        ):
            reply, _, _ = await llm_complete(
                system, f'User message: "{text}"',
                provider, model, api_key,
                max_tokens=200, temperature=0.2, **extra,
            )
        reply = reply.strip()
        if reply.upper().startswith(_PROCEED_TO_QUERY):
            return None  # genuine data query — fall through to pipeline
        return reply or None
    except Exception as e:
        log.debug("Analyst reply error, falling through to pipeline: %s", e)
        return None  # fail open


# ── Background query task (typing + classification + pipeline) ────────────────

async def _run_query_with_guard(
    account_id, event, adapter, text, portal_user, client_row, *, is_clarification=False
):
    """
    Background task — three stages:
      1. Immediate typing indicator so the user sees '...' before any LLM work.
         Running this in the background (not in dispatch) keeps the HTTP 200
         response to Teams instant, avoiding the 5-second webhook timeout.
      2. Off-topic classification — only for fresh queries; skipped for
         clarification replies which have already been through the pipeline.
      3. Query pipeline wrapped in a persistent typing loop for adapters like
         Teams that drop the indicator after ~4 s (re-sent every 2.5 s).
    """
    import asyncio
    from core.query_pipeline import handle_query as _hq

    _send = getattr(adapter, "send_status", None)

    # Stage 1 — immediate typing so the user has instant feedback
    if _send:
        try:
            await _send(event, "processing", "Working on it")
        except Exception:
            pass

    # Regulated tenants: scrub user-typed PII from the question BEFORE the
    # off-topic classifier below sends the raw text to the LLM. handle_query
    # re-scrubs (idempotent) for callers that bypass this dispatcher — the
    # user-facing "identifiers were removed" notice is sent from there, not
    # here, so it isn't duplicated.
    try:
        _profile = store.get_compliance_profile(account_id)
        if _profile.get("mode") == "regulated":
            from core.masking import scrub_question_pii
            text, _ = scrub_question_pii(text, _profile.get("industry", ""))
    except Exception as e:
        log.debug("question PII scrub skipped: %s", e)

    # Stage 2 — dynamic analyst gate (skipped for clarification replies)
    # Replaces the old _classify_is_data_question + static _OFF_TOPIC_REPLY:
    # _generate_analyst_reply returns None for genuine data requests (fall through)
    # or a tailored capability/off-topic answer to send directly.
    if not is_clarification:
        from core.result_cache import result_cache
        from core.result_commands import parse_result_command

        _session_id = getattr(adapter, "session_id", "") or ""
        _is_cached_result_command = bool(
            _session_id
            and result_cache.has_result(_session_id)
            and parse_result_command(text) is not None
        )
        if not _is_cached_result_command:
            _analyst_reply = await _generate_analyst_reply(text, account_id, client_row)
            if _analyst_reply is not None:
                await adapter.send_message(event, _analyst_reply)
                return

    # Stage 3 — run query pipeline; Teams needs re-sent typing every 2.5 s
    if getattr(adapter, "persistent_typing", False) and _send:
        stop = asyncio.Event()

        async def _loop():
            while not stop.is_set():
                try:
                    await _send(event, "processing", "Working on it")
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(asyncio.shield(stop.wait()), timeout=2.5)
                except asyncio.TimeoutError:
                    pass

        task = asyncio.create_task(_loop())
        try:
            await _hq(account_id, event, adapter, text, portal_user,
                      is_clarification=is_clarification)
        finally:
            stop.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    else:
        await _hq(account_id, event, adapter, text, portal_user,
                  is_clarification=is_clarification)


def _enqueue_query(bg, account_id, event, adapter, text, portal_user, client_row, *, is_clarification=False):
    """Schedule the query pipeline as a background task."""
    bg.add_task(
        _run_query_with_guard, account_id, event, adapter, text, portal_user, client_row,
        is_clarification=is_clarification,
    )


# ── Access request (admin-approval flow, no registration link) ────────────────

async def handle_unregistered_user(account_id, zoom_user_id, event, adapter):
    """
    Create a pending_platform_user row and notify the user once.
    If the user is already pending/rejected, stay silent so they aren't spammed.
    Approved users should never reach this path (they have a portal_user row).
    """
    platform_type = getattr(event, "platform", "teams") or "teams"

    # Extract display name from the raw webhook payload — structure varies per platform
    raw = getattr(event, "raw", None) or {}
    if platform_type == "teams":
        display_name = (raw.get("from") or {}).get("name", "") or ""
    elif platform_type == "zoom":
        display_name = (raw.get("payload") or {}).get("userName", "") or ""
    else:
        # Slack and others don't carry the display name in the event payload
        display_name = ""

    is_new, _pending = store.upsert_pending_user(
        account_id=account_id,
        platform_type=platform_type,
        platform_user_id=zoom_user_id,
        display_name=display_name,
        conversation_ref=event.channel_id or "{}",
    )

    if is_new:
        await adapter.send_message(event,
            "👋 *Welcome to QueryBot!*\n\n"
            "Your access request has been sent to your administrator.\n"
            "You'll receive a message here once your access is approved.\n\n"
            "_You don't need to do anything — your admin will be in touch._"
        )
        log.info(
            "Pending access request created: platform=%s user=%s account=%s name=%r",
            platform_type, zoom_user_id, account_id, display_name,
        )
    else:
        status = _pending.get("status", "pending")
        if status == "rejected":
            await adapter.send_message(event,
                "Your access request was not approved. "
                "Please contact your administrator for assistance."
            )
        # If still pending: stay silent — they already got the "request sent" message


# ── Background tasks ──────────────────────────────────────────────────────────

async def _run_example_validation(
    account_id: str, kb_dir: str, chroma_dir: str, db_cfg: dict
) -> None:
    """Step 2 — Validate Stage 2 SQL patterns against real DB in background.

    validate_and_store_examples() is synchronous and runs up to ~200 blocking
    DB calls sequentially on one connection. Run it in the default executor
    (thread pool), not directly on the event loop — otherwise a slow pattern
    freezes every other request the whole app is serving, not just this KB
    build. core/examples.py sets a per-query driver timeout so a single bad
    pattern can't stall the batch itself; this wait_for is an outer ceiling
    in case that somehow doesn't fire (e.g. a hung network read).
    """
    try:
        from core.examples import validate_and_store_examples
        loop = asyncio.get_running_loop()
        count = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                validate_and_store_examples,
                account_id, kb_dir, db_cfg["credentials"], db_cfg["db_type"], chroma_dir,
            ),
            timeout=1200.0,  # 20 min ceiling for the whole batch
        )
        log.info("Example validation complete: %d validated examples for %s",
                 count, account_id)
    except asyncio.TimeoutError:
        log.error("Example validation timed out after 20 min for %s", account_id)
    except Exception as e:
        log.error("Example validation failed for %s: %s", account_id, e)


async def _run_log_harvest(account_id: str, chroma_dir: str) -> None:
    """Step 4 — Harvest successful query log entries into validated examples.

    Disabled when enable_feedback_collection = 1: the governed learning loop
    handles example creation via quality scoring + admin review, so legacy
    auto-harvesting would bypass that governance gate.
    """
    _cli = store.get_client(account_id) or {}
    if _cli.get("enable_feedback_collection"):
        log.info(
            "_run_log_harvest skipped for %s — governed learning loop is active "
            "(enable_feedback_collection=1). Use the admin Learning Queue instead.",
            account_id,
        )
        return
    try:
        from core.examples import harvest_and_embed
        added = harvest_and_embed(account_id, chroma_dir)
        if added > 0:
            log.info("Harvested %d new examples from query log for %s", added, account_id)
    except Exception as e:
        log.error("Log harvest failed for %s: %s", account_id, e)


# ── Message dispatcher ────────────────────────────────────────────────────────

async def dispatch(
    account_id,
    event: PlatformEvent,
    adapter,
    bg: BackgroundTasks,
    portal_user: dict | None = None,   # pre-authenticated for web portal sessions
):

    text = event.text.strip()

    # Onboarding is admin-only. Unknown workspaces get a clear error — we do NOT
    # auto-create a client row here anymore.
    client_row = store.get_client(account_id)
    if not client_row:
        await adapter.send_message(event,
            "⚠️ This workspace is not registered with QueryBot.\n"
            "Ask your administrator to register it in the admin panel "
            "before sending queries.")
        return

    if text.lower() == "help":
        await adapter.send_message(event, _HELP); return

    # _ABOUT_RE questions ("what can you do", "what is querybot") are now
    # handled dynamically by _generate_analyst_reply inside the READY branch
    # so they get a real business-context-aware answer instead of _ABOUT.
    # We keep a fast-path for when state is not yet READY (NEW/KB_BUILDING):
    # in those states client_row exists but the pipeline can't answer data
    # questions anyway, so the static _ABOUT response is still appropriate.
    if _ABOUT_RE.search(text) and get_state(account_id).get("state") not in ("READY",):
        await adapter.send_message(event, _ABOUT); return

    state = get_state(account_id).get("state", "NEW")

    if state in ("NEW", "SCHEMA_READY"):
        await adapter.send_message(event,
            "⚠️ This workspace isn't set up yet.\n\n"
            "Ask your administrator to finish the *Schema & Knowledge Base Setup* "
            "in the QueryBot admin panel before sending queries.")
        return
    if state == "KB_BUILDING":
        await adapter.send_message(event,
            "⏳ Knowledge Base is still being built by the admin — "
            "try again in a few minutes.")
        return

    # For web portal sessions the user is already authenticated via
    # the signed cookie — portal_user is passed in directly. For webhook
    # sessions (Zoom/Teams/Slack) resolve the linked external identity
    # inside this account so roles and table access cannot cross tenants.
    if portal_user is None and event.user_id:
        portal_user = store.get_user_by_platform_id(account_id, event.user_id)
        if not portal_user:
            await handle_unregistered_user(account_id, event.user_id, event, adapter)
            return

    # ── Behavioral front door (deterministic, no LLM) ─────────────────────────
    # Greetings/thanks/goodbye/frustration used to fall through every guard
    # into the SQL pipeline — "thanks" got answered with "I couldn't find the
    # right tables or columns". Handle them here, state-independent. The
    # data-aware kinds (data_inventory / opinion / vague) are handled later,
    # inside the READY branch, after the pending-clarification check — a
    # clarification reply must never be hijacked by this classifier.
    from core.conversational import build_reply, build_reply_split, detect_conversational
    _conv_kind = detect_conversational(text)
    if _conv_kind in ("greeting", "thanks", "goodbye", "frustration"):
        _send_sq = getattr(adapter, "send_suggested_questions", None)
        if _conv_kind == "greeting" and callable(_send_sq):
            _intro, _qs = build_reply_split(_conv_kind, account_id, portal_user)
            await adapter.send_message(event, _intro)
            if _qs:
                await _send_sq(event, "Here are some questions to get you started:", _qs)
        else:
            await adapter.send_message(event, build_reply(_conv_kind, account_id, portal_user))
        return

    if text.lower() == "whoami":
        pu = portal_user or (store.get_user_by_platform_id(account_id, event.user_id) if event.user_id else None)
        if pu:
            t = store.get_allowed_tables(pu)
            tlist = ", ".join(sorted(t)) if t else "All tables (admin)"
            await adapter.send_message(event,
                f"*{pu['name']}* | {pu['role']} | Group: {pu['group_name'] or 'none'}\n"
                f"Tables: {tlist}")
        else:
            await adapter.send_message(event, "Not registered yet — send any message for your registration link.")
        return

    if text.lower() == "status":
        client = store.get_client(account_id) or {}
        db_cfg = get_client_db(account_id)
        used   = store.get_monthly_query_count(account_id)
        limit  = client.get("query_limit_monthly", 500)
        pu     = portal_user or (store.get_user_by_platform_id(account_id, event.user_id) if event.user_id else None)
        await adapter.send_message(event,
            f"*State:* {get_state(account_id)['state']}\n"
            f"*Database:* {db_cfg['name'] if db_cfg else 'not configured'}\n"
            f"*Queries this month:* {used}/{limit}\n"
            f"*User:* {pu['name'] if pu else 'not registered'}")
        return

    if state == "READY":

        # ── Session greeting — fires once per new session, before the query ──
        # touch_user_activity returns True when last_active_at is NULL (first-ever
        # message) or when the gap since last activity exceeds 30 minutes.
        # Sending the greeting here (after all early-return commands, and after
        # the greeting/thanks/goodbye/frustration conversational checks at the top
        # of dispatch) means it is an *extra* message prepended to the user's
        # actual answer — the real question is still processed normally below.
        # Skipped for clarification replies and DDL — those handle their own flow.
        if portal_user and not _conv_kind:
            try:
                _is_new_session = store.touch_user_activity(portal_user["id"])
                if _is_new_session:
                    _first_name = (portal_user.get("name") or "").split()[0] or (portal_user.get("name") or "")
                    _was_never_active = not portal_user.get("last_active_at")
                    if _was_never_active:
                        _greet_msg = f"👋 Welcome, {_first_name}! I'm QueryBot — ask me anything about your business data."
                    else:
                        _greet_msg = f"👋 Welcome back, {_first_name}!"
                    await adapter.send_message(event, _greet_msg)
            except Exception as _greet_exc:
                log.debug("Session greeting skipped: %s", _greet_exc)

        # ── Clarification reply check — before DDL and before normal routing ──
        if event.user_id:
            pending = get_pending(account_id, event.user_id)
            if pending:
                cmeta = pending.get("clarification_meta") or {}
                opts = cmeta.get("options") or []
                selected_text = text
                matched_option_id: str | None = None
                # Compound-split replies run the chosen half as a standalone
                # question — combining it with the original (bundled) text
                # would re-create the exact multi-intent problem we split.
                if cmeta.get("source") == "compound_split" and opts:
                    match = resolve_option_text(opts, text)
                    clear_pending(account_id, event.user_id)
                    chosen_q = str((match or {}).get("value") or text).strip()
                    log.info("Compound split resolved: running %r", chosen_q[:80])
                    _enqueue_query(bg, account_id, event, adapter,
                                   chosen_q, portal_user, client_row,
                                   is_clarification=True)
                    return
                if opts:
                    match = resolve_option_text(opts, text)
                    if not match:
                        if _looks_like_new_query(text, pending["original_q"]):
                            clear_pending(account_id, event.user_id)
                            log.info(
                                "Cleared stale clarification for '%s' because a new query arrived: '%s'",
                                pending["original_q"][:80],
                                text[:80],
                            )
                            _enqueue_query(bg, account_id, event, adapter, text, portal_user, client_row)
                            return
                        send_prompt = getattr(adapter, "send_clarification_prompt", None)
                        if callable(send_prompt):
                            await send_prompt(event, cmeta.get("question") or "Please choose one of the available options.", opts)
                        else:
                            await adapter.send_message(event, "Please reply using one of the clarification options so I can continue.")
                        return
                    selected_text = str(match.get("value") or match.get("label") or text).strip()
                    matched_option_id = str(match.get("id") or "") or None
                else:
                    # Continue the original request with the user's free-text
                    # clarification instead of resetting the conversation.
                    combined, term_hint = combine_with_clarification(
                        pending["original_q"],
                        text,
                        cmeta,
                    )
                    if term_hint:
                        combined = f"{combined}\n\n{term_hint}"
                    clear_pending(account_id, event.user_id)
                    log.info(
                        "Free-text clarification received for '%s' — combined: '%s'",
                        pending["original_q"][:80],
                        combined[:120],
                    )
                    _enqueue_query(bg, account_id, event, adapter,
                                   combined, portal_user, client_row, is_clarification=True)
                    return
                combined, term_hint = combine_with_clarification(
                    pending["original_q"],
                    selected_text,
                    cmeta,
                    selected_option_id=matched_option_id,
                )
                if term_hint:
                    combined = f"{combined}\n\n{term_hint}"
                clear_pending(account_id, event.user_id)
                log.info("Clarification received for '%s' — combined: '%s' (term_hint=%s)",
                         pending["original_q"][:50], combined[:120], bool(term_hint))
                _enqueue_query(bg, account_id, event, adapter,
                               combined, portal_user, client_row, is_clarification=True)
                return

            # Fix #7 — their clarification lapsed in the 5-min TTL. Tell them
            # instead of silently processing the reply as a fresh query.
            if was_recently_expired(account_id, event.user_id):
                acknowledge_recently_expired(account_id, event.user_id)
                # Only surface the hint if the reply looks like a short answer
                # to a clarification, not a brand-new question.
                if len(text.split()) <= 6 and not is_ddl_attempt(text):
                    await adapter.send_message(event,
                        "⏱️ Your previous clarification request timed out. "
                        "Please ask your original question again and I'll pick "
                        "it up from there."
                    )
                    return

        # ── Behavioral front door, data-aware kinds ────────────────────────
        # Placed after the clarification checks (a pending clarification
        # reply always wins) and before the DDL/LLM path. These have real
        # deterministic answers — sending them into SQL generation only
        # produces a confusing failure.
        if _conv_kind in ("data_inventory", "opinion", "vague"):
            _send_sq = getattr(adapter, "send_suggested_questions", None)
            if _conv_kind in ("data_inventory", "vague") and callable(_send_sq):
                _intro, _qs = build_reply_split(_conv_kind, account_id, portal_user)
                await adapter.send_message(event, _intro)
                if _qs:
                    await _send_sq(event, "Try one of these:", _qs)
            else:
                await adapter.send_message(event, build_reply(_conv_kind, account_id, portal_user))
            return

        # DDL check on raw user message before any LLM call
        if is_ddl_attempt(text):
            await adapter.send_message(event, _DDL_USER_MESSAGE)
            return

        # ── Compound question — offer a guided split ────────────────────────
        # Two independent asks in one message ("revenue by region and also
        # top 10 customers") make one SQL attempt answer half or fail. Offer
        # to run them one at a time; the user picks which goes first. Never
        # auto-fan-out into two silent queries.
        from core.conversational import detect_compound_question
        _split = detect_compound_question(text)
        if _split and event.user_id:
            from core.clarification import save_pending
            _q1, _q2 = _split
            _split_opts = [
                {"id": "part1", "label": _q1[:80], "value": _q1},
                {"id": "part2", "label": _q2[:80], "value": _q2},
            ]
            save_pending(
                account_id, event.user_id, text,
                clarification_meta={"source": "compound_split",
                                    "question": "Which should I answer first?",
                                    "options": _split_opts},
            )
            _prompt = (
                "That looks like two questions in one — I answer them best "
                "one at a time. Which should I run first?"
            )
            send_prompt = getattr(adapter, "send_clarification_prompt", None)
            if callable(send_prompt):
                await send_prompt(event, _prompt, _split_opts)
            else:
                await adapter.send_message(
                    event,
                    f"{_prompt}\n  1. {_q1}\n  2. {_q2}\n\n"
                    "Reply with the one you want (or ask it directly).")
            return

        _enqueue_query(bg, account_id, event, adapter, text, portal_user, client_row)
