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
from core.llm import is_ddl_attempt, _DDL_USER_MESSAGE

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
    """Step 2 — Validate Stage 2 SQL patterns against real DB in background."""
    try:
        from core.examples import validate_and_store_examples
        count = validate_and_store_examples(
            account_id  = account_id,
            queries_dir = kb_dir,
            credentials = db_cfg["credentials"],
            db_type     = db_cfg["db_type"],
            chroma_dir  = chroma_dir,
        )
        log.info("Example validation complete: %d validated examples for %s",
                 count, account_id)
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
    from core.query_pipeline import handle_query

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

    if _ABOUT_RE.search(text):
        await adapter.send_message(event, _ABOUT); return

    if text.lower() == "whoami":
        pu = portal_user or (store.get_user_by_zoom_id(event.user_id) if event.user_id else None)
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
        pu     = portal_user or (store.get_user_by_zoom_id(event.user_id) if event.user_id else None)
        await adapter.send_message(event,
            f"*State:* {get_state(account_id)['state']}\n"
            f"*Database:* {db_cfg['name'] if db_cfg else 'not configured'}\n"
            f"*Queries this month:* {used}/{limit}\n"
            f"*User:* {pu['name'] if pu else 'not registered'}")
        return

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
    if state == "READY":
        # For web portal sessions the user is already authenticated via
        # the signed cookie — portal_user is passed in directly. For webhook
        # sessions (Zoom/Teams/Slack) we look up by zoom_user_id as before.
        if portal_user is None and event.user_id:
            portal_user = store.get_user_by_zoom_id(event.user_id)
            if not portal_user:
                await handle_unregistered_user(account_id, event.user_id, event, adapter)
                return

        # ── Clarification reply check — before DDL and before normal routing ──
        if event.user_id:
            pending = get_pending(account_id, event.user_id)
            if pending:
                cmeta = pending.get("clarification_meta") or {}
                opts = cmeta.get("options") or []
                selected_text = text
                matched_option_id: str | None = None
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
                            bg.add_task(handle_query, account_id, event, adapter, text, portal_user)
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
                    bg.add_task(handle_query, account_id, event, adapter,
                                combined, portal_user, is_clarification=True)
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
                bg.add_task(handle_query, account_id, event, adapter,
                            combined, portal_user, is_clarification=True)
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

        # DDL check on raw user message before any LLM call
        if is_ddl_attempt(text):
            await adapter.send_message(event, _DDL_USER_MESSAGE)
            return
        bg.add_task(handle_query, account_id, event, adapter, text, portal_user)
