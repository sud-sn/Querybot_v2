"""
QueryBot v2 — main entry point

Per-user access control flow:
  1. User messages bot → Zoom sends accountId + userId
  2. Bot looks up portal_user by zoom_user_id
  3. Unknown user → one-time registration link sent
  4. Registered user → group tables loaded → enforced in RAG + validator
  5. Admin role → unrestricted all-table access
"""

import json
import asyncio
import logging
import time
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse

import store
from store.db import init_db
from gateway import get_adapter, PlatformEvent
from core.llm import llm_complete, build_sql_system_prompt, resolve_provider, is_ddl_attempt, _DDL_USER_MESSAGE
from core.examples import retrieve_similar_examples, format_examples_for_prompt
from core.clarification import (
    check_ambiguity_glossary_first, save_pending, get_pending,
    clear_pending, combine_with_clarification, resolve_option_text,
    build_schema_grounded_clarification_hint,
    was_recently_expired, acknowledge_recently_expired,
)
from core.webhook_dedup import is_duplicate_event, remember_event
from core.schema import run_query, load_known_tables
from core.knowledge import load_retriever
from core.validator import validate_sql
from core.chart import detect_chart_type, build_chart_payload
from core.query_semantics import build_generic_query_hints
from core.response_builder import build_assistant_response
from core.insight import generate_followup_suggestions
from core.graph_resolver import resolve_for_question as _graph_resolve
from core.llm_audit import llm_audit_scope, make_llm_audit_request_id
from core.result_cache import result_cache
from core.query_router import should_route_to_result_cache, build_duckdb_system_prompt
from core.duckdb_sql_validator import validate_duckdb_result_sql
from admin import router as admin_router
from portal import router as portal_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("querybot")

app = FastAPI(title="QueryBot", version="2.0.0")
app.include_router(admin_router)
app.include_router(portal_router)


@app.on_event("startup")
async def startup() -> None:
    init_db()
    # Fix #6: Warn if session secrets are using insecure defaults
    import os
    if not os.getenv("SESSION_SECRET") and not os.getenv("PORTAL_SESSION_SECRET"):
        log.warning(
            "⚠️  SESSION_SECRET / PORTAL_SESSION_SECRET not set — "
            "using insecure default. Set these environment variables "
            "before deploying to production."
        )
    if not os.getenv("ADMIN_SESSION_SECRET") and not os.getenv("SESSION_SECRET"):
        log.warning(
            "⚠️  ADMIN_SESSION_SECRET not set — "
            "admin sessions use an insecure default."
        )

    # LLM audit log retention. Default 30 days; override via LLM_AUDIT_RETENTION_DAYS.
    try:
        retention = int(os.getenv("LLM_AUDIT_RETENTION_DAYS", "30"))
        deleted = store.purge_old_llm_calls(retention)
        if deleted:
            log.info("Purged %d llm_call_log rows older than %d days", deleted, retention)
    except Exception as e:
        log.warning("LLM audit purge failed at startup: %s", e)

    try:
        from core.log_export import scheduled_log_export_loop
        app.state.log_export_task = asyncio.create_task(scheduled_log_export_loop())
        log.info("External log export scheduler started")
    except Exception as e:
        log.warning("External log export scheduler failed to start: %s", e)

    log.info("QueryBot v2 started — database ready")


# ── Helpers ───────────────────────────────────────────────────────────────────

@app.on_event("shutdown")
async def shutdown() -> None:
    task = getattr(app.state, "log_export_task", None)
    if task:
        task.cancel()


def client_dir(account_id: str) -> Path:
    p = Path("clients") / account_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_state(account_id: str) -> dict:
    client = store.get_client(account_id)
    if not client:
        return {"state": "NEW"}
    state_data = json.loads(client.get("state_data") or "{}")
    return {"state": client["state"], **state_data}


def save_state(account_id, state, state_data=None, business_desc=None):
    store.update_client_state(account_id, state, state_data or {}, business_desc)


def get_client_db(account_id: str) -> dict | None:
    client = store.get_client(account_id)
    if not client:
        return None

    db_config_id = client.get("db_config_id")
    if not db_config_id:
        return None

    return store.get_db_config(db_config_id)


def check_query_limit(account_id: str) -> tuple[bool, int, int]:
    client = store.get_client(account_id)
    limit  = (client or {}).get("query_limit_monthly") or 500
    used   = store.get_monthly_query_count(account_id)
    return used < limit, used, limit


def check_token_limit(account_id: str) -> tuple[bool, int, int]:
    client = store.get_client(account_id) or {}
    limit = int(client.get("token_limit_monthly") or 0)
    usage = store.get_monthly_token_status(account_id)
    used = int(usage.get("total_tokens") or 0)
    if limit <= 0:
        return True, used, 0
    return used < limit, used, limit


def get_portal_base() -> str:
    import os
    return os.getenv("PORTAL_BASE_URL", "http://localhost:8000").rstrip("/")


def _looks_like_new_query(text: str, original_q: str = "") -> bool:
    """
    Return True only when the user's message is clearly a brand-new question
    unrelated to any pending clarification.

    Key fix: a message ending with "?" that shares significant vocabulary
    with the original question is a REFINEMENT, not a new query.
    Example: original="find employees by department", reply="by late status?"
    → shares "by", "department"-adjacent context → treat as refinement.
    """
    msg = (text or "").strip().lower()
    if not msg:
        return False

    # If we have an original question to compare against, use word-overlap
    # to detect refinements before checking for new-query signals.
    if original_q:
        orig_words = {
            w for w in original_q.lower().split()
            if len(w) >= 4  # ignore short words like "is", "by", "the"
        }
        reply_words = {w for w in msg.split() if len(w) >= 4}
        overlap = orig_words & reply_words
        # If reply shares 2+ meaningful words with the original question,
        # it is almost certainly a refinement — not a new query
        if len(overlap) >= 2:
            return False

    # Short one/two word clarification answers are never new queries
    if len(msg.split()) <= 2:
        return False

    starters = (
        "show", "what", "which", "how", "compare", "list", "give",
        "break down", "breakdown", "analyze", "analyse", "explain",
        "why", "trend", "count", "total", "find", "get",
    )
    if any(msg.startswith(prefix) for prefix in starters):
        return True

    # Only treat "?" as a new-query signal when the message is long
    # (short messages ending in "?" are often just emphasis: "attrition status?")
    if "?" in msg and len(msg.split()) >= 5:
        return True

    return len(msg.split()) >= 8  # long messages with no starter are new queries




def _extract_kb_synonym_injection(context: str) -> str:
    """
    Scan retrieved KB text for ## Business Synonyms and ## Key Metrics sections
    and build a compact 'Plain-English term → exact column' injection block.

    This fires at every query — even follow-ups — because it works directly from
    the already-retrieved KB chunks without needing the glossary DB to be populated.
    It is the last-resort guard against the LLM inventing CamelCase column names.
    """
    import re as _re

    synonym_rows: list[tuple[str, str]] = []   # (plain-english terms, column)
    metric_rows:  list[tuple[str, str]] = []   # (metric name, column)

    in_synonyms = False
    in_metrics  = False

    for line in context.splitlines():
        stripped = line.strip()

        # KB chunk separator — reset section state
        if stripped == "---":
            in_synonyms = False
            in_metrics  = False
            continue

        # Section detection
        if stripped.startswith("## ") or stripped.startswith("### "):
            header = stripped.lstrip("#").strip().lower()
            in_synonyms = header.startswith("business synonym")
            in_metrics  = header.startswith("key metric")
            continue

        # Business Synonyms table rows: | Plain English | Column | Notes |
        if in_synonyms and stripped.startswith("|") and "---" not in stripped:
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if len(cells) >= 2:
                eng  = cells[0].strip("`").strip()
                col  = cells[1].strip("`").strip()
                if col and eng and eng.lower() not in ("plain english", "column", ""):
                    synonym_rows.append((eng, col))

        # Key Metrics lines: - **Metric name**: `COLUMN_NAME` — ...
        if in_metrics and stripped.startswith("-"):
            m = _re.match(
                r"-\s*\*\*([^*]+)\*\*\s*:?\s*`?([A-Za-z_][A-Za-z0-9_]*)`?",
                stripped,
            )
            if m:
                metric_name = m.group(1).strip()
                col_name    = m.group(2).strip()
                if metric_name and col_name:
                    metric_rows.append((metric_name, col_name))

    if not synonym_rows and not metric_rows:
        return ""

    lines = [
        "COLUMN SYNONYM MAP (authoritative — use EXACT column names shown here):",
        "When the user's question mentions any of the plain-English terms below, "
        "use the exact column name on the right. Do NOT invent CamelCase variants.",
    ]
    seen_cols: set[str] = set()
    for eng, col in synonym_rows[:25]:
        if col.upper() not in seen_cols:
            lines.append(f"  • '{eng}' → exact column: {col}")
            seen_cols.add(col.upper())
    for metric, col in metric_rows[:15]:
        if col.upper() not in seen_cols:
            lines.append(f"  • '{metric}' → exact column: {col}")
            seen_cols.add(col.upper())

    return "\n".join(lines) + "\n"


async def _send_live_stage(adapter, event, stage: str, label: str, detail: str = "") -> None:
    sender = getattr(adapter, "send_status", None)
    if callable(sender):
        try:
            await sender(event, stage, label, detail)
        except Exception as e:
            log.debug("Live status send failed: %s", e)


def _format_metric_formula_context(metrics: list[dict]) -> str:
    if not metrics:
        return ""

    blocks = [
        "APPROVED METRIC FORMULAS:",
        "Use these admin-approved metric definitions whenever the user asks for the metric name or a synonym.",
        "For formula expressions, use the expression exactly in the SELECT list and combine it with the requested GROUP BY dimensions.",
        "For percentage/rate formulas, do not average row-level percentage columns unless the formula explicitly uses AVG().",
        "If required columns are not present in the retrieved KB context, reply CANNOT_GENERATE instead of inventing columns.",
    ]
    for idx, metric in enumerate(metrics, start=1):
        formula_type = (metric.get("formula_type") or "query").lower()
        kind = "formula expression" if formula_type == "expression" else "trusted SQL query/template"
        lines = [
            f"{idx}. Metric: {metric.get('name', '')}",
            f"   Type: {kind}",
            f"   Result format: {metric.get('result_format') or 'number'}",
            f"   Synonyms: {metric.get('synonyms') or '(none)'}",
        ]
        if metric.get("description"):
            lines.append(f"   Business meaning: {metric.get('description')}")
        if metric.get("required_columns"):
            lines.append(f"   Required columns: {metric.get('required_columns')}")
        if metric.get("allowed_dimensions"):
            lines.append(f"   Safe dimensions: {metric.get('allowed_dimensions')}")
        if metric.get("grain"):
            lines.append(f"   Grain: {metric.get('grain')}")
        if metric.get("example_questions"):
            lines.append(f"   Example questions: {metric.get('example_questions')}")
        if metric.get("default_time_column"):
            lines.append(f"   Default time column: {metric.get('default_time_column')} — use this column when grouping by date/period")
        lines.append(f"   Approved calculation: {metric.get('sql_template', '').strip()}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)

# ══════════════════════════════════════════════════════════════════════════════
# Registration
# ══════════════════════════════════════════════════════════════════════════════

async def handle_unregistered_user(account_id, zoom_user_id, event, adapter):
    token   = store.create_registration_token(account_id, zoom_user_id)
    reg_url = f"{get_portal_base()}/portal/register?token={token}"
    await adapter.send_message(event,
        "👋 *Welcome to QueryBot!*\n\n"
        "You need to register before you can query data.\n\n"
        f"*Register here (link valid 48 hours):*\n{reg_url}\n\n"
        "_Your administrator will assign you to a group after registration._"
    )
    log.info("Registration link sent to zoom_user_id=%s", zoom_user_id)


# ══════════════════════════════════════════════════════════════════════════════
# Onboarding is handled exclusively in the admin panel.
# See admin/routes.py: /admin/clients/new, /admin/clients/{id}/setup
# Chat platforms only accept queries for clients already in the READY state.
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# DuckDB helper — generate SQL for in-memory result cache queries
# ══════════════════════════════════════════════════════════════════════════════

async def _generate_duckdb_sql(question: str, system_prompt: str, client: dict) -> str:
    """
    Use the LLM to generate a DuckDB SELECT query for the virtual `result` table.

    Returns the raw SQL string, or "CANNOT_GENERATE" if the LLM signals it
    cannot answer the question from the cached data.
    """
    try:
        provider, model, api_key, az_kwargs = resolve_provider(client, purpose="query")
        sql, _, _ = await llm_complete(
            system=system_prompt,
            user=question,
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=0.0,
            max_tokens=512,
            **az_kwargs,
        )
        return (sql or "").strip()
    except Exception as exc:
        log.warning("_generate_duckdb_sql failed: %s", exc)
        return "CANNOT_GENERATE"


# ══════════════════════════════════════════════════════════════════════════════
# Query pipeline — table-aware
# ══════════════════════════════════════════════════════════════════════════════

async def handle_query(account_id, event, adapter, question, portal_user, is_clarification=False):
    start_ms = int(time.time() * 1000)
    state    = get_state(account_id)
    db_cfg   = get_client_db(account_id)
    client   = store.get_client(account_id) or {}
    audit_enabled = bool(client.get("enable_llm_audit"))
    audit_request_id = make_llm_audit_request_id()

    # Identity passed through every query-log row for audit + billing.
    pu_id  = portal_user.get("id") if portal_user else None
    zid    = event.user_id or ""
    trace_id = _trace_create(
        account_id=account_id,
        question_id=audit_request_id,
        question=question,
        portal_user_id=pu_id,
        platform_user_id=zid,
        session_id=getattr(adapter, "session_id", "") or "",
        request_source=getattr(event, "platform", "") or "",
    )
    _trace_step(trace_id, "receive_question", output_summary={"question_id": audit_request_id})

    if not db_cfg:
        _trace_finish(trace_id, status="error", answer_type="error", error_message="No database assigned")
        await adapter.send_message(event, "⚠️ No database assigned. Contact your administrator.")
        return

    await _send_live_stage(adapter, event, "authorization", "Checking access", "Verifying workspace access and available data.")

    within_limit, used, limit = check_query_limit(account_id)
    if not within_limit:
        _trace_finish(trace_id, status="error", answer_type="error", error_message="Monthly query limit reached")
        await adapter.send_message(event, f"❌ Monthly query limit reached ({used}/{limit}).")
        return
    if used >= int(limit * 0.8):
        await adapter.send_message(event, f"⚠️ {used}/{limit} queries used this month.")

    within_token_limit, tokens_used, token_limit = check_token_limit(account_id)
    if not within_token_limit:
        _trace_finish(trace_id, status="error", answer_type="error", error_message="Monthly token limit reached")
        await adapter.send_message(event, f"❌ Monthly token limit reached ({tokens_used}/{token_limit}).")
        return
    if token_limit and tokens_used >= int(token_limit * 0.8):
        await adapter.send_message(event, f"⚠️ {tokens_used}/{token_limit} tokens used this month.")

    try:
        provider, model, api_key, az_kwargs = resolve_provider(client, purpose="query")
        _trace_update(trace_id, llm_provider=provider, llm_model=model, db_type=db_cfg.get("db_type", ""))
    except RuntimeError as e:
        _trace_finish(trace_id, status="error", answer_type="error", error_message=str(e))
        await adapter.send_message(event, f"⚠️ Config error: {e}")
        return

    # Determine this user's allowed tables
    # all_known      : every base table in the connected DB (authoritative).
    # allowed_tables : user's permitted subset (uppercase). None = admin.
    # effective      : intersection — what this user can actually see.
    allowed_tables = store.get_allowed_tables(portal_user) if portal_user else None
    all_known      = load_known_tables(state.get("schema_dir", ""))

    if allowed_tables is None:
        effective = all_known  # admin — unrestricted
    else:
        allowed_tables = {t.upper() for t in allowed_tables}
        effective = {t for t in all_known if t in allowed_tables}

    # ── Schema scoping — applied when user selects a specific schema tab ──────
    # schema_hint comes from the portal chat schema selector (e.g. "HR").
    # We narrow effective and allowed_tables to only tables in that schema.
    # This scopes RAG retrieval, SQL generation, and validation to that schema.
    schema_hint = (getattr(event, "schema_hint", "") or "").upper().strip()
    if schema_hint:
        def _in_schema(fqn: str) -> bool:
            """True if the FQN's schema part matches the selected schema."""
            parts = fqn.upper().split(".")
            # DB.SCHEMA.TABLE → parts[-2] is schema
            # SCHEMA.TABLE    → parts[-2] is schema
            # TABLE           → no schema, keep it (can't filter)
            if len(parts) >= 2:
                return parts[-2] == schema_hint
            return True  # bare name — keep

        effective = {t for t in effective if _in_schema(t)}
        if allowed_tables is not None:
            allowed_tables = {t for t in allowed_tables if _in_schema(t)}

        if not effective:
            await adapter.send_message(event,
                f"⚠️ No tables from the **{schema_hint}** schema are available to you. "
                f"Switch to a different schema or ask your administrator to grant access.")
            return

    _trace_update(
        trace_id,
        selected_schema=schema_hint,
        allowed_tables_snapshot=sorted(effective),
    )
    _trace_step(
        trace_id,
        "resolve_user_permissions",
        output_summary={"tables_available": len(effective), "schema": schema_hint or ""},
    )

    if portal_user and allowed_tables is not None and not effective:
        _trace_finish(trace_id, status="error", answer_type="error", error_message="No table access assigned")
        await adapter.send_message(event,
            "🔒 *No table access assigned.*\n\n"
            "Your account is not yet linked to any tables in this workspace. "
            "Please contact your administrator to request access before you "
            "can ask data questions.")
        return

    if not effective:
        _trace_finish(trace_id, status="error", answer_type="error", error_message="No tables available")
        await adapter.send_message(event,
            "⚠️ No tables are available to query. Contact your administrator.")
        return

    # ── Step 2.5: Tier-2 DuckDB routing — answer from cached result set ──────
    # When the user's question clearly refers to the already-returned data
    # ("who is below average?", "rank these", "show outliers"), run the query
    # against the in-memory DuckDB result cache instead of hitting the
    # production database.  Fast, private, supports full analytic SQL.
    _session_id = getattr(adapter, "session_id", None)
    _cached_cols = [s["name"] for s in result_cache.get_schema(_session_id)] if _session_id else []
    if _session_id and should_route_to_result_cache(question, result_cache.has_result(_session_id), cached_col_names=_cached_cols):
        _trace_update(trace_id, route="duckdb_cache")
        _trace_step(trace_id, "route", output_summary="duckdb_cache")
        log.info("Routing to DuckDB result cache for session %s", _session_id[:16])
        await _send_live_stage(adapter, event, "retrieving_context", "Analysing results", "Running analytics on the previously returned data.")
        try:
            _duck_schema = result_cache.get_schema(_session_id)
            _duck_stats  = result_cache.get_stats(_session_id)
            _duck_sys_prompt = build_duckdb_system_prompt(
                _duck_schema,
                db_type    = db_cfg.get("db_type", "azure_sql"),
                data_stats = _duck_stats,
            )
            _duck_sql = await _generate_duckdb_sql(question, _duck_sys_prompt, client)
            if _duck_sql and _duck_sql.strip().upper() != "CANNOT_GENERATE":
                _duck_verdict = validate_duckdb_result_sql(_duck_sql)
                _trace_update(
                    trace_id,
                    generated_sql=_duck_sql,
                    sql_validation_status="pass" if _duck_verdict.ok else "fail",
                    sql_validation_error="" if _duck_verdict.ok else _duck_verdict.reason,
                )
                _trace_step(
                    trace_id,
                    "validate_sql",
                    input_summary=_duck_sql,
                    output_summary=_duck_verdict.reason,
                    status="success" if _duck_verdict.ok else "error",
                    metadata={"code": _duck_verdict.code},
                )
                if not _duck_verdict.ok:
                    raise ValueError(f"DuckDB SQL rejected: {_duck_verdict.reason}")
                await _send_live_stage(adapter, event, "executing_query", "Running query", "Querying in-memory result set.")
                _duck_rows = result_cache.query(_session_id, _duck_sql)
                if _duck_rows is not None:
                    duration_ms = int(time.time() * 1000) - start_ms
                    _log_q(account_id, question, _duck_sql, len(_duck_rows), True, "",
                           "duckdb_cache", "duckdb", 0, 0, duration_ms,
                           portal_user_id=pu_id, zoom_user_id=zid,
                           question_id=audit_request_id)
                    _add_history = getattr(adapter, "add_to_history", None)
                    if callable(_add_history) and _duck_rows:
                        _add_history(
                            question=question,
                            sql=_duck_sql,
                            columns=list(_duck_rows[0].keys()) if _duck_rows else [],
                            row_count=len(_duck_rows),
                        )
                    await _send_results(event, adapter, question, _duck_rows, _duck_sql,
                                        duration_ms, portal_user, account_id, db_cfg,
                                        question_id=audit_request_id)
                    _trace_finish(
                        trace_id,
                        status="success",
                        answer_type="table",
                        row_count=len(_duck_rows),
                        duration_ms=duration_ms,
                        final_answer_summary="Answered from in-memory DuckDB result cache",
                    )
                    return
        except Exception as _duck_exc:
            log.warning("DuckDB cache route failed, falling through to normal pipeline: %s", _duck_exc)
        # Fall through to normal pipeline if DuckDB routing fails

    # ── Step 3: Metric registry — deterministic SQL for known metrics ────────
    # If the question matches a defined metric, assemble SQL without the LLM.
    # Downstream query scope. For unrestricted admins this remains None unless
    # they explicitly select a schema tab; then the selected schema must also
    # constrain retrieval, prompt grounding, validation, and repair.
    query_scope_tables = effective if (allowed_tables is not None or schema_hint) else None

    matched_metric = store.match_metric(account_id, question)
    if matched_metric:
        _trace_update(trace_id, route="metric_registry", generated_sql=matched_metric["sql_template"].strip())
        _trace_step(trace_id, "route", output_summary={"route": "metric_registry", "metric": matched_metric.get("name", "")})
        await _send_live_stage(adapter, event, "metric_registry", "Using known metric", "Found a trusted metric definition for this question.")
        sql_from_metric = matched_metric["sql_template"].strip()
        log.info("Metric registry hit: %s → %s", matched_metric["name"], sql_from_metric[:60])
        try:
            await _send_live_stage(adapter, event, "executing_query", "Running query", "Executing the trusted metric query against your database.")
            rows = run_query(db_cfg["credentials"], db_cfg["db_type"], sql_from_metric)
            duration_ms = int(time.time()*1000) - start_ms
            _log_q(account_id, question, sql_from_metric, len(rows), True, "",
                   "metric_registry", "deterministic", 0, 0, duration_ms,
                   portal_user_id=pu_id, zoom_user_id=zid,
                   question_id=audit_request_id)
            _trace_update(
                trace_id,
                sql_validation_status="trusted_metric",
                query_row_count=len(rows),
                query_duration_ms=duration_ms,
            )
            _trace_step(trace_id, "execute_sql", input_summary=sql_from_metric, output_summary={"rows": len(rows)}, duration_ms=duration_ms)
            # Record metric-registry hits in conversation history too so
            # follow-up questions ("filter to top 5", "break that down by region")
            # can reference the returned columns and SQL shape.
            _add_history = getattr(adapter, "add_to_history", None)
            if callable(_add_history) and rows:
                _add_history(
                    question=question,
                    sql=sql_from_metric,
                    columns=list(rows[0].keys()) if rows else [],
                    row_count=len(rows),
                )
            await _send_results(event, adapter, question, rows, sql_from_metric,
                                duration_ms, portal_user, account_id, db_cfg,
                                question_id=audit_request_id)
            _trace_finish(trace_id, status="success", answer_type="table", row_count=len(rows), duration_ms=duration_ms, final_answer_summary="Answered by metric registry")
            return
        except Exception as e:
            _trace_step(trace_id, "execute_sql", input_summary=sql_from_metric, output_summary=str(e), status="error")
            log.warning("Metric registry SQL failed, falling through to LLM: %s", e)
            # Fall through to normal LLM pipeline

    # ── RAG — scoped to effective tables ─────────────────────────────────────
    # The retriever filters by doc_id (table name), not by substring match,
    # so disallowed tables never leak into the LLM prompt. allowed_tables
    # is passed through explicitly; None means admin/unrestricted.
    try:
        await _send_live_stage(adapter, event, "retrieving_context", "Understanding your data", "Retrieving the most relevant schema, examples, and business context.")
        import re as _re
        retriever    = load_retriever(account_id)   # Qdrant — no filesystem path needed

        _grouping = bool(_re.search(
            r"\b(by|per|grouped by|breakdown|split by|each|for each)\s+\w",
            question.lower()
        ))
        _n = 10 if _grouping else 8

        rag_filter = query_scope_tables
        relevant_kbs = retriever.retrieve(question, n=_n, allowed_tables=rag_filter)

        pinned    = [d for d in relevant_kbs if retriever._is_global(d)]
        table_kbs = [d for d in relevant_kbs if not retriever._is_global(d)]

        if _grouping:
            fact_patterns = retriever.retrieve_fact_patterns(
                question, n=2, allowed_tables=rag_filter,
            )
            for fp in fact_patterns:
                if fp not in (pinned + table_kbs):
                    table_kbs.insert(0, fp)

        relevant_kbs = (pinned + table_kbs)[:7]
        context = "\n\n---\n\n".join(relevant_kbs)
        _trace_update(
            trace_id,
            route="normal_sql",
            retrieved_kb_chunk_ids=store.kb_chunk_refs(relevant_kbs),
        )
        _trace_step(trace_id, "retrieve_kb", output_summary={"chunks": len(relevant_kbs)})

        # ── Step 2: Retrieve validated SQL examples — few-shot grounding ─────
        # Examples now live in Qdrant alongside KB docs — no chroma_dir needed
        examples = retrieve_similar_examples(
            question, account_id, n=3, allowed_tables=rag_filter,
        )
        if examples:
            context = format_examples_for_prompt(examples) + "\n\n---\n\n" + context
            log.info("Injected %d validated examples into prompt", len(examples))
            _trace_step(trace_id, "retrieve_examples", output_summary={"examples": len(examples)})

    except Exception as e:
        log.error("RAG retrieval failed: %s", e)
        _trace_finish(trace_id, status="error", answer_type="error", error_message=f"RAG retrieval failed: {e}")
        await adapter.send_message(event, "⚠️ Knowledge Base not ready.")
        return

    # SQL generation — inject any matched business-glossary terms as grounding hints
    term_injection = store.build_term_injection(account_id, question, query_scope_tables)
    schema_grounded_hint = build_schema_grounded_clarification_hint(
        account_id,
        question,
        context,
        allowed_tables=query_scope_tables,
    )
    generic_hints = build_generic_query_hints(question)
    metric_formula_context = _format_metric_formula_context(
        store.list_metric_formula_context(account_id, question, limit=6)
    )

    # If the question came from a suggested-question click, inject the FQN hint
    # so the LLM uses the correct table name format for this DB type.
    table_hint_str = getattr(event, "table_hint", "") or ""
    if table_hint_str:
        parts = table_hint_str.upper().split(".")
        db_type_hint = db_cfg.get("db_type", "")
        if db_type_hint == "azure_sql":
            # Azure SQL only supports 2-part: [SCHEMA].[TABLE]
            sql_name = f"[{parts[-2]}].[{parts[-1]}]" if len(parts) >= 2 else f"[{parts[-1]}]"
        elif db_type_hint == "oracle":
            # Oracle uses OWNER.TABLE
            sql_name = f"{parts[-2]}.{parts[-1]}" if len(parts) >= 2 else parts[-1]
        else:
            # Snowflake supports DATABASE.SCHEMA.TABLE
            sql_name = ".".join(parts)
        table_hint_injection = (
            f"SCHEMA HINT: This question is about the table {table_hint_str}. "
            f"Use exactly {sql_name} in the SQL if relevant."
        )
    else:
        table_hint_injection = ""

    selected_schema_injection = ""
    if schema_hint:
        selected_schema_injection = (
            f"ACTIVE SCHEMA: The user selected the {schema_hint} schema in the chat UI. "
            "Use only tables from this schema. If a table is referenced, qualify it "
            f"with the {schema_hint} schema in the generated SQL."
        )

    # Scan already-retrieved KB for Business Synonyms / Key Metrics → compact map.
    # This runs even when the glossary DB is empty and guards against the LLM
    # inventing CamelCase column names for well-known business terms.
    kb_synonym_injection = _extract_kb_synonym_injection(context)

    context_parts = [
        part for part in (
            selected_schema_injection,
            table_hint_injection,
            term_injection,
            kb_synonym_injection,
            schema_grounded_hint,
            generic_hints,
            metric_formula_context,
            context,
        )
        if part
    ]
    context_with_terms = "\n\n".join(context_parts)
    # Multi-turn memory: inject conversation history for web portal sessions
    _conv_history = []
    _history_fn = getattr(adapter, "get_history", None)
    if callable(_history_fn):
        _conv_history = _history_fn()

    # Entity graph — deterministic JOIN skeleton resolution
    # Loads the client's entity graph once and resolves JOINs from the question
    # before the LLM is called so the LLM never guesses table relationships.
    _graph_ctx: dict = {}
    try:
        _full_graph = store.get_full_graph(account_id)
        # When the user selected a specific schema, restrict the graph to
        # entities that belong to that schema so the resolver never proposes
        # JOINs to tables from a different schema (which validation would reject).
        if schema_hint and _full_graph.get("entities"):
            _sh = schema_hint.upper()
            _filtered_entities = [
                e for e in _full_graph["entities"]
                if not e.get("schema_name") or e.get("schema_name", "").upper() == _sh
            ]
            _in_schema_names = {e["entity_name"] for e in _filtered_entities}
            _full_graph = {
                "entities": _filtered_entities,
                "relationships": [
                    r for r in _full_graph.get("relationships", [])
                    if r["from_entity"] in _in_schema_names
                    and r["to_entity"] in _in_schema_names
                ],
                "properties": _full_graph.get("properties", []),
            }
        if _full_graph.get("entities"):
            _graph_ctx = _graph_resolve(
                question=question,
                account_id=account_id,
                db_type=db_cfg.get("db_type", "azure_sql"),
                graph=_full_graph,
            )
            if _graph_ctx.get("enabled"):
                log.info(
                    "Graph resolved for %s: entities=%s anchor=%s schema_filter=%s",
                    account_id, _graph_ctx.get("detected"), _graph_ctx.get("anchor"),
                    schema_hint or "none",
                )
    except Exception as _gex:
        log.debug("Graph resolution skipped: %s", _gex)

    # ── Table coverage guarantee ──────────────────────────────────────────────
    # After graph resolution we know which tables are required (from detected
    # entities) and which are already covered by the retrieved KB docs.
    # For any gap — a table the graph needs but RAG missed — we do a direct
    # Qdrant filter fetch (not a semantic search) and append the KB doc to
    # context_with_terms so the LLM sees every table's column definitions.
    #
    # Why this matters: dense + BM25 retrieval ranks by similarity to the
    # *question*.  Secondary JOIN tables (e.g. a patient dim that's never
    # mentioned by name) often score below the cutoff and are dropped.  The
    # LLM then hallucinates column names for those tables → CANNOT_GENERATE.
    #
    # Capped at 3 gap-fill docs; all failures are swallowed so this never
    # blocks SQL generation.
    if _graph_ctx.get("enabled"):
        try:
            from core.table_coverage import build_required_fqns, guarantee_table_coverage
            _required_fqns = build_required_fqns(_graph_ctx, _full_graph)
            if _required_fqns:
                _gap_docs = guarantee_table_coverage(
                    account_id    = account_id,
                    required_fqns = _required_fqns,
                    retrieved_docs = relevant_kbs,   # what actually went into context
                    rag_filter    = rag_filter,       # ACL scope (None = admin)
                    max_fill      = 3,
                )
                if _gap_docs:
                    context_with_terms = (
                        context_with_terms
                        + "\n\n---\n\n"
                        + "\n\n---\n\n".join(_gap_docs)
                    )
                    log.info(
                        "Table coverage: injected %d gap-fill doc(s) into prompt "
                        "for %s — missing tables now covered",
                        len(_gap_docs), sorted(_required_fqns),
                    )
        except Exception as _cov_exc:
            log.debug("Table coverage guarantee skipped: %s", _cov_exc)

    system = build_sql_system_prompt(
        db_cfg["db_type"], context_with_terms,
        conversation_history=_conv_history or None,
        graph_context=_graph_ctx or None,
    )
    try:
        await _send_live_stage(adapter, event, "generating_sql", "Generating query", "Translating the business question into SQL.")
        with llm_audit_scope(
            account_id=account_id,
            question=question,
            enabled=audit_enabled,
            request_id=audit_request_id,
            question_id=audit_request_id,
            component="sql_generation",
        ):
            sql, tok_in, tok_out = await llm_complete(
                system, question, provider, model, api_key, max_tokens=512, **az_kwargs)
            _trace_update(
                trace_id,
                generated_sql=sql,
                prompt_tokens=tok_in,
                completion_tokens=tok_out,
            )
            _trace_step(
                trace_id,
                "llm_generate_sql",
                output_summary={"tokens_in": tok_in, "tokens_out": tok_out},
            )
    except Exception as e:
        _log_q(account_id, question, "", 0, False, str(e), provider, model, 0, 0,
               int(time.time()*1000)-start_ms,
               portal_user_id=pu_id, zoom_user_id=zid,
               question_id=audit_request_id)
        _trace_finish(trace_id, status="error", answer_type="error", error_message=f"AI error: {e}")
        await adapter.send_message(event, f"⚠️ AI error: {e}")
        return

    if sql.startswith("```"):
        sql = "\n".join(sql.split("\n")[1:]).rsplit("```", 1)[0].strip()

    # ── Safety net: inject SELECT DISTINCT for list-entity questions ──────────
    # Fires only when the LLM forgot DISTINCT on a non-aggregate list query.
    # Silently skipped for aggregate / GROUP BY / already-DISTINCT queries.
    sql = _inject_distinct_if_needed(sql, question)

    # CANNOT_GENERATE — try clarification before giving up (Approach B)
    if "CANNOT_GENERATE" in sql.upper():
        _trace_update(trace_id, generated_sql=sql, sql_validation_status="cannot_generate")
        _log_q(account_id, question, "", 0, False, "CANNOT_GENERATE",
               provider, model, tok_in, tok_out, int(time.time()*1000)-start_ms,
               portal_user_id=pu_id, zoom_user_id=zid,
               question_id=audit_request_id)

        # Skip ambiguity check when this IS a clarification reply — prevents infinite loop
        if not is_clarification:
            with llm_audit_scope(
                account_id=account_id,
                question=question,
                enabled=audit_enabled,
                request_id=audit_request_id,
            question_id=audit_request_id,
                component="clarification",
            ):
                is_ambiguous, clarifying_q, cmeta = await check_ambiguity_glossary_first(
                    account_id, question, context, provider, model, api_key, az_kwargs,
                    allowed_tables=query_scope_tables,
                )
            if is_ambiguous and clarifying_q and event.user_id:
                opts = (cmeta or {}).get("options") or []
                if opts:
                    save_pending(account_id, event.user_id, question, context, clarification_meta=cmeta)
                    send_prompt = getattr(adapter, "send_clarification_prompt", None)
                    if callable(send_prompt):
                        await send_prompt(event, clarifying_q or "I need a bit more context to answer that.", opts)
                    else:
                        # Plain-text fallback: list the options so the user can
                        # reply with one of them. Without this, they see only
                        # the clarifying question and have to guess what to say.
                        option_lines = []
                        for o in opts[:5]:
                            lbl = (o.get("label") or o.get("value") or "").strip()
                            if lbl:
                                option_lines.append(f"  • {lbl}")
                        options_text = "\n".join(option_lines)
                        await adapter.send_message(event,
                            f"❓ I need a bit more context to answer that.\n\n"
                            f"{clarifying_q}\n\n"
                            f"{options_text}\n\n"
                            f"_Reply with one of the options above (or type your own)._"
                        )
                else:
                    save_pending(account_id, event.user_id, question, context, clarification_meta=cmeta)
                    await adapter.send_message(event,
                        f"❓ I need a bit more context to answer that.\n\n"
                        f"{clarifying_q}\n\n"
                        f"_Reply in plain language and I'll continue with your original question._"
                    )
                return
        await adapter.send_message(event,
                "❓ I couldn't find the right tables or columns to answer that.\n\n"
                "Try rephrasing — for example:\n"
                "  • Be more specific about the metric you want\n"
                "  • Include a time range (last month, this year)\n"
                "  • Mention the specific column or category name\n\n"
                "If this is a business concept not in the data, ask your "
                "administrator to add it to the Metric Registry."
            )
        return

    # ── Validate + Execute with ONE unified retry on failure ────────────────
    # Retry fires on:
    #   * validator failure (unknown_table / parse)   — LLM picks a valid table
    #   * execution failure                           — LLM fixes the SQL
    # Retry does NOT fire on:
    #   * access_denied — the user genuinely lacks permission; we need the
    #                     admin to intervene, not a different SQL.
    #   * ddl / cannot_generate — already terminal.
    # Only the currently-failing query is retried — a single attempt, never a loop.
    rows        = None
    exec_error  = None
    last_reason = ""
    last_code   = ""

    await _send_live_stage(adapter, event, "validating_sql", "Checking query safety", "Verifying table access, structure, and execution safety.")
    ok, reason, code = validate_sql(sql, all_known, db_cfg["db_type"], query_scope_tables)
    _trace_update(
        trace_id,
        generated_sql=sql,
        sql_validation_status="pass" if ok else "fail",
        sql_validation_error="" if ok else reason,
    )
    _trace_step(
        trace_id,
        "validate_sql",
        input_summary=sql,
        output_summary=reason,
        status="success" if ok else "error",
        metadata={"code": code},
    )

    if ok:
        try:
            await _send_live_stage(adapter, event, "executing_query", "Running query", "Executing the SQL against your connected data source.")
            rows = run_query(db_cfg["credentials"], db_cfg["db_type"], sql)
            _trace_step(trace_id, "execute_sql", input_summary=sql, output_summary={"rows": len(rows)})
        except Exception as first_error:
            exec_error = str(first_error)
            _trace_step(trace_id, "execute_sql", input_summary=sql, output_summary=exec_error, status="error")
            log.warning("First execution failed for %s: %s",
                        account_id, exec_error[:100])
    else:
        last_reason, last_code = reason, code

    retryable = (not ok and code in ("unknown_table", "parse")) or (exec_error is not None)

    if retryable:
        if exec_error is not None:
            import re as _re_retry
            # Extract column name(s) flagged as invalid by the DB engine.
            # SQL Server / Azure SQL format: "Invalid column name 'XYZ'"
            bad_cols = _re_retry.findall(
                r"Invalid column name '([^']+)'", exec_error, _re_retry.IGNORECASE
            )
            col_fix_note = ""
            if bad_cols:
                cols_list = ", ".join(f"'{c}'" for c in bad_cols)
                col_fix_note = (
                    f"\n⚠️  COLUMN NAME ERROR: The column(s) {cols_list} do NOT exist in the database.\n"
                    f"These column names were invented — they are NOT in the schema.\n"
                    f"MANDATORY: Look at the 'COLUMN SYNONYM MAP' and 'BUSINESS TERM DEFINITIONS' "
                    f"sections in the system prompt to find the EXACT column name for each concept.\n"
                    f"Also check the 'Session context' section — if the previous turn returned a column "
                    f"that represents the same concept, reuse that EXACT column name verbatim.\n"
                    f"NEVER guess, never use CamelCase variants of column names.\n"
                )
            retry_user = (
                f"The following SQL failed with this error:\n"
                f"SQL: {sql}\n"
                f"Error: {exec_error}\n"
                f"{col_fix_note}\n"
                f"The original question was: {question}\n\n"
                f"Rewrite the SQL to fix the error. Use ONLY column names that appear "
                f"verbatim in the Knowledge Base (system prompt). "
                f"Return only the corrected SQL, no explanation."
            )
        else:
            retry_user = (
                f"The following SQL failed validation with: {last_reason}\n"
                f"SQL: {sql}\n\n"
                f"The original question was: {question}\n\n"
                f"Rewrite the SQL using only tables and columns that appear in "
                f"the provided knowledge base context. Return only the corrected "
                f"SQL, no explanation."
            )

        try:
            await _send_live_stage(adapter, event, "repairing_query", "Repairing query", "Fixing a validation or execution issue before retrying.")
            with llm_audit_scope(
                account_id=account_id,
                question=question,
                enabled=audit_enabled,
                request_id=audit_request_id,
            question_id=audit_request_id,
                component="sql_repair",
            ):
                sql_retry, _, _ = await llm_complete(
                    build_sql_system_prompt(db_cfg["db_type"], context_with_terms),
                    retry_user, provider, model, api_key,
                    max_tokens=512, **az_kwargs,
                )
            if sql_retry.startswith("```"):
                sql_retry = "\n".join(sql_retry.split("\n")[1:]).rsplit("```", 1)[0].strip()

            sql_retry = _inject_distinct_if_needed(sql_retry, question)

            if "CANNOT_GENERATE" not in sql_retry.upper() and len(sql_retry) > 10:
                ok2, reason2, code2 = validate_sql(
                    sql_retry, all_known, db_cfg["db_type"], query_scope_tables)
                if ok2:
                    try:
                        await _send_live_stage(adapter, event, "executing_query", "Retrying query", "Running the corrected query against your data.")
                        rows = run_query(
                            db_cfg["credentials"], db_cfg["db_type"], sql_retry)
                        sql         = sql_retry
                        exec_error  = None
                        ok, last_reason, last_code = True, "OK", "ok"
                        log.info("Retry succeeded for %s", account_id)
                    except Exception as retry_exec_err:
                        exec_error = str(retry_exec_err)
                        log.warning("Retry execution failed for %s: %s",
                                    account_id, exec_error[:100])
                else:
                    last_reason, last_code = reason2, code2
                    log.warning("Retry still invalid for %s: %s",
                                account_id, reason2[:120])
        except Exception as retry_err:
            log.warning("Retry LLM call failed for %s: %s",
                        account_id, str(retry_err)[:100])

    # ── Terminal failure handling ────────────────────────────────────────────
    if not ok:
        _log_q(account_id, question, sql, 0, False, last_reason, provider, model,
               tok_in, tok_out, int(time.time()*1000) - start_ms,
               portal_user_id=pu_id, zoom_user_id=zid,
               question_id=audit_request_id)
        _trace_finish(trace_id, status="error", answer_type="error", error_message=last_reason)
        await adapter.send_message(event, f"❌ {last_reason}")
        return

    if exec_error is not None or rows is None:
        _log_q(account_id, question, sql, 0, False,
               exec_error or "Unknown error",
               provider, model, tok_in, tok_out,
               int(time.time()*1000) - start_ms,
               portal_user_id=pu_id, zoom_user_id=zid,
               question_id=audit_request_id)
        _trace_finish(trace_id, status="error", answer_type="error", error_message=exec_error or "Unknown error")
        sql_preview = sql[:200] + "..." if len(sql) > 200 else sql
        await adapter.send_message(event,
            f"❌ Database error — could not execute after retry.\n"
            f"Error: {exec_error}\n\n"
            f"SQL tried:\n{sql_preview}"
        )
        return

    duration_ms = int(time.time()*1000) - start_ms
    _log_q(account_id, question, sql, len(rows), True, "", provider, model,
           tok_in, tok_out, duration_ms,
           portal_user_id=pu_id, zoom_user_id=zid,
           question_id=audit_request_id)

    # Zero rows — try clarification only if NOT already a clarification reply
    # AND there is some ambiguity signal (Fix #3). Blind LLM ambiguity checks
    # on every empty result set waste tokens and confuse users whose filter
    # was legitimately correct but the data genuinely has no rows.
    if len(rows) == 0 and event.user_id and not is_clarification:
        _zr_matches = store.match_terms_in_question(account_id, question, query_scope_tables)
        _zr_has_required = any(
            m.get("requires_clarification") and m.get("clarification_options")
            for m in _zr_matches
        )
        _zr_has_multi_metric = len([m for m in _zr_matches if m.get("kind") == "metric"]) >= 2

        if not (_zr_has_required or _zr_has_multi_metric):
            # No ambiguity signal — just tell the user there's no data.
            _trace_finish(trace_id, status="success", answer_type="empty", row_count=0, duration_ms=duration_ms, final_answer_summary="Query returned no rows")
            await adapter.send_message(event,
                "The query ran successfully but returned *no rows*.\n\n"
                "Try broadening the filter (e.g. a wider date range) or "
                "checking the category values used in the question."
            )
            return

        with llm_audit_scope(
            account_id=account_id,
            question=question,
            enabled=audit_enabled,
            request_id=audit_request_id,
            question_id=audit_request_id,
            component="clarification",
        ):
            is_ambiguous, clarifying_q, cmeta = await check_ambiguity_glossary_first(
                account_id, question, context, provider, model, api_key, az_kwargs,
                allowed_tables=query_scope_tables,
            )
        if is_ambiguous and clarifying_q:
            opts = (cmeta or {}).get("options") or []
            if opts:
                save_pending(account_id, event.user_id, question, context, clarification_meta=cmeta)
                send_prompt = getattr(adapter, "send_clarification_prompt", None)
                if callable(send_prompt):
                    await send_prompt(event, clarifying_q or "I need a bit more context to answer that.", opts)
                else:
                    await adapter.send_message(event,
                        f"The query ran successfully but returned *no results*.\n\n"
                        f"❓ {clarifying_q}\n\n"
                        f"_Reply with one of the listed clarification options and I'll rerun the query._"
                    )
            else:
                save_pending(account_id, event.user_id, question, context, clarification_meta=cmeta)
                await adapter.send_message(event,
                    f"The query ran successfully but returned *no results*.\n\n"
                    f"❓ {clarifying_q}\n\n"
                    f"_Reply in plain language and I'll continue with your original question._"
                )
            return

    await _send_live_stage(adapter, event, "formatting_results", "Preparing results", "Formatting the answer and any chart for display.")
    # Record this turn in conversation history (web portal only)
    _add_history = getattr(adapter, "add_to_history", None)
    if callable(_add_history) and rows:
        _add_history(
            question=question,
            sql=sql,
            columns=list(rows[0].keys()) if rows else [],
            row_count=len(rows),
        )
    await _send_results(event, adapter, question, rows, sql, duration_ms,
                        portal_user, account_id, db_cfg,
                        rag_context=context, question_id=audit_request_id)
    _trace_finish(trace_id, status="success", answer_type="table", row_count=len(rows), duration_ms=duration_ms, final_answer_summary="Answered from database query")


async def _send_results(event, adapter, question, rows, sql, duration_ms,
                        portal_user, account_id, db_cfg,
                        rag_context: str = "", question_id: str | None = None):
    """Send formatted results to the chat platform. Shared by LLM and metric registry paths."""
    # Cache result on adapter for insight follow-ups (WebSocket sessions).
    # Pass question_id so drilldowns can link back to this original question.
    cache_fn = getattr(adapter, "cache_result", None)
    if callable(cache_fn):
        cache_fn(rows, question, sql, db_cfg, rag_context, question_id=question_id)

    table_text = _rows_to_table(rows)
    row_word   = "row" if len(rows) == 1 else "rows"
    dur_label  = f"{duration_ms}ms" if duration_ms < 1000 else f"{duration_ms/1000:.1f}s"

    chart_type = detect_chart_type(rows, question=question)
    pin_token = None
    chart_payload = None
    if chart_type and portal_user:
        pin_token = _create_pin_token(
            user_id=portal_user["id"],
            account_id=account_id,
            question=question,
            sql_query=sql,
            chart_type=chart_type,
            db_config_id=db_cfg["id"],
        )
    if chart_type:
        chart_payload = build_chart_payload(rows, chart_type, title=question)
        if chart_payload:
            if pin_token:
                chart_payload["pin_token"] = pin_token
            chart_payload["question"] = question
            chart_payload["duration_label"] = dur_label
            chart_payload["row_count"] = len(rows)

    rich_sender = getattr(adapter, "send_assistant_response", None)
    if callable(rich_sender):
        if chart_payload:
            await _send_live_stage(adapter, event, "chart_ready", "Building chart", "Rendering an interactive chart for this answer.")
        response_payload = build_assistant_response(
            question=question,
            rows=rows,
            sql=sql,
            duration_ms=duration_ms,
            chart=chart_payload,
            data_source=str(db_cfg.get("db_type", "")),
        )
        # ── Result-aware follow-up suggestions (web portal only) ──────────
        # Generate suggestions from the statistical brief already computed
        # inside build_assistant_response — no extra DB call or raw row exposure.
        # Uses a lightweight 160-token LLM call; failures are silent.
        if portal_user and rows and len(rows[0]) >= 2 if rows else False:
            brief         = response_payload.get("data_brief") or {}
            result_scope  = response_payload.get("result_scope") or {}
            audit_rid     = response_payload.get("analysis_contract", {}).get("request_id", "") or question_id or ""
            audit_enabled = bool((db_cfg or {}).get("account_id"))
            try:
                suggestions = await generate_followup_suggestions(
                    brief=brief,
                    question=question,
                    result_scope=result_scope,
                    db_cfg=db_cfg,
                    account_id=account_id,
                    rows=rows,           # statistical signal computation — no raw values sent to LLM
                    audit_enabled=True,
                    audit_request_id=str(audit_rid),
                )
                if suggestions:
                    response_payload["follow_up_suggestions"] = suggestions
            except Exception as _fex:
                log.debug("Follow-up suggestions skipped: %s", _fex)
        await rich_sender(event, response_payload)
        return

    if len(rows) == 1 and len(rows[0]) == 1:
        col_name = list(rows[0].keys())[0]
        value    = _format_value(rows[0][col_name])
        greeting = f"*{portal_user['name']}* — " if portal_user else ""
        reply = (
            f"{greeting}*{question}*\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"  {col_name}\n"
            f"  *{value}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"_{dur_label} · {col_name.replace('_', ' ')}_"
        )
    else:
        greeting = f"*{portal_user['name']}*\n" if portal_user else ""
        reply = (
            f"{greeting}*{question}*\n\n"
            f"*{len(rows)} {row_word}*\n"
            f"{'─' * 40}\n"
            f"{table_text}\n"
            f"{'─' * 40}\n"
            f"_{dur_label}_"
        )

    await adapter.send_message(event, reply)

    chart_sender = getattr(adapter, "send_chart", None)
    if chart_payload and callable(chart_sender):
        await _send_live_stage(adapter, event, "chart_ready", "Building chart", "Rendering an interactive chart for this answer.")
        await chart_sender(event, chart_payload)
        return

    if chart_payload and portal_user:
        pin_url = f"{get_portal_base()}/portal/pin-confirm?token={pin_token}"
        await adapter.send_message(event, f"Chart ready — [Pin to my dashboard]({pin_url})")


def _create_pin_token(
    user_id: int, account_id: str, question: str,
    sql_query: str, chart_type: str, db_config_id: int,
) -> str:
    """
    Store pending pin data server-side and return a short token.
    The token is passed in the URL — SQL never goes through Zoom markdown.
    Token expires after 30 minutes (user must click the pin link promptly).
    """
    import secrets
    from store.db import get_db
    token = secrets.token_urlsafe(16)
    with get_db() as conn:
        # Create table if not exists (lightweight, no migration needed)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pin_token (
                token        TEXT PRIMARY KEY,
                user_id      INTEGER NOT NULL,
                account_id   TEXT NOT NULL,
                question     TEXT NOT NULL,
                sql_query    TEXT NOT NULL,
                chart_type   TEXT NOT NULL,
                db_config_id INTEGER NOT NULL,
                expires_at   TEXT NOT NULL,
                created_at   TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            INSERT INTO pin_token
                (token, user_id, account_id, question, sql_query,
                 chart_type, db_config_id, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', '+30 minutes'))
        """, (token, user_id, account_id, question, sql_query,
              chart_type, db_config_id))
    return token


def _log_q(account_id, question, sql, rows, success, error,
           provider, model, tok_in, tok_out, dur_ms,
           portal_user_id=None, zoom_user_id="",
           question_id="", parent_question_id=""):
    store.log_query(
        account_id=account_id, question=question, sql_generated=sql,
        row_count=rows, success=success, error_msg=error,
        llm_provider=provider, llm_model=model,
        tokens_in=tok_in, tokens_out=tok_out, duration_ms=dur_ms,
        portal_user_id=portal_user_id, zoom_user_id=zoom_user_id or "",
        question_id=question_id or "", parent_question_id=parent_question_id or "",
    )


def _trace_create(**kwargs) -> int | None:
    try:
        if "question" in kwargs and "question_text" not in kwargs:
            kwargs["question_text"] = kwargs.pop("question")
        return store.create_answer_trace(**kwargs)
    except Exception as exc:
        log.debug("answer trace create failed: %s", exc)
        return None


def _trace_update(trace_id: int | None, **fields) -> None:
    try:
        store.update_answer_trace(trace_id, **fields)
    except Exception as exc:
        log.debug("answer trace update failed: %s", exc)


def _trace_step(trace_id: int | None, step_name: str, **kwargs) -> None:
    try:
        store.log_answer_trace_step(trace_id, step_name=step_name, **kwargs)
    except Exception as exc:
        log.debug("answer trace step failed: %s", exc)


def _trace_finish(trace_id: int | None, **kwargs) -> None:
    try:
        store.finish_answer_trace(trace_id, **kwargs)
    except Exception as exc:
        log.debug("answer trace finish failed: %s", exc)


def _format_value(val) -> str:
    """
    Format a single cell value for clean display.
    - None          → —
    - Whole floats  → 792 (not 792.000000)
    - Decimal floats→ 538.42 (not 538.420000), with comma separators
    - Large ints    → 1,973,331
    - Strings       → as-is
    """
    if val is None:
        return "—"
    if isinstance(val, float):
        # Strip trailing zeros: 792.000000 → 792, 538.420000 → 538.42
        if val == int(val):
            return f"{int(val):,}"
        # Round to 4dp then strip trailing zeros to avoid float precision noise
        rounded = round(val, 4)
        # Format with commas, then strip trailing zeros
        parts = f"{rounded:,.4f}".split(".")
        decimal = parts[1].rstrip("0")
        return f"{parts[0]}.{decimal}" if decimal else parts[0]
    if isinstance(val, int) and val > 999:
        return f"{val:,}"
    return str(val)


async def _generate_result_narration(
    question: str,
    rows: list[dict],
    currency_cols: list[str],
    client: dict,
) -> str:
    """
    Generate a 1-2 sentence plain-English insight from a drill-down result.
    Uses a minimal 150-token LLM call.  Returns "" on any failure so callers
    can silently skip it.
    """
    if not rows:
        return ""
    try:
        # Build a compact data summary — top 5 rows only, no raw dumps
        col_names = list(rows[0].keys())
        preview_rows = rows[:5]
        row_lines = []
        for r in preview_rows:
            parts = []
            for k, v in r.items():
                if k in currency_cols and v is not None:
                    try:
                        parts.append(f"{k}=${float(v):,.2f}")
                    except (ValueError, TypeError):
                        parts.append(f"{k}={v}")
                else:
                    parts.append(f"{k}={v}")
            row_lines.append("  " + ", ".join(parts))
        total = len(rows)
        summary = (
            f"Question: {question}\n"
            f"Result ({total} row{'s' if total != 1 else ''}):\n"
            + "\n".join(row_lines)
            + (f"\n  ...and {total - 5} more rows" if total > 5 else "")
        )
        provider, model, api_key, az_kwargs = resolve_provider(client, purpose="query")
        narration, _, _ = await llm_complete(
            system=(
                "You are a data analyst. Given a question and a query result, "
                "write exactly ONE concise sentence (max 30 words) summarising "
                "the key insight or direct answer. Be specific — include the top "
                "value or most notable number. No padding, no 'The data shows'."
            ),
            user=summary,
            provider=provider,
            model=model,
            api_key=api_key,
            max_tokens=80,
            temperature=0.2,
            **az_kwargs,
        )
        return (narration or "").strip()
    except Exception as _exc:
        log.debug("result_chat narration skipped: %s", _exc)
        return ""


def _build_cannot_generate_hint(
    schema: list[dict],
    stats: dict,
    prev_rows: list[dict] | None = None,
    prev_question: str = "",
) -> str:
    """
    Build a column-aware suggestion message when DuckDB AND the production
    DB fallback both fail to answer.  Lists what the user CAN ask based on
    the available columns, and adds a rephrasing tip with explicit values
    when the question likely references entities from the previous result.
    """
    if not schema:
        return "Try asking a fresh question in the main chat."

    numeric_cols = [
        c["name"] for c in stats.get("columns", [])
        if c.get("min") is not None
    ]
    text_cols = [
        c["name"] for c in stats.get("columns", [])
        if c.get("sample_values")   # truthy: empty list [] (numeric cols) is falsy
    ]
    currency_cols = [
        c["name"] for c in stats.get("columns", [])
        if c.get("is_currency")
    ]

    suggestions = []
    if numeric_cols:
        col = numeric_cols[0]
        prefix = "$" if col in currency_cols else ""
        suggestions += [
            f"'what is the average {col.lower().replace('_', ' ')}'",
            f"'show rows where {col.lower().replace('_', ' ')} is above average'",
            f"'rank by {col.lower().replace('_', ' ')}'",
        ]
    if text_cols:
        col = text_cols[0]
        label = col.lower().replace("_", " ")
        suggestions.append(f"'filter by {label}'")

    col_summary = ", ".join(f"`{c['name']}`" for c in schema)

    # ── Special case: pure aggregate result ────────────────────────────────────
    # When every column is numeric (count/sum/avg) and there are no identifier
    # columns the user cannot drill into it — the result is just a total.
    # Give a direct redirect instead of analytcis suggestions that make no sense.
    if not text_cols and numeric_cols:
        num_col = numeric_cols[0].lower().replace("_", " ")
        # For single-row aggregate results like "Total Prescriptions: 16"
        if prev_rows and len(prev_rows) == 1:
            val = list(prev_rows[0].values())[0] if prev_rows[0] else ""
            return (
                f"This result shows a summary total ({col_summary} = **{val}**). "
                f"There are no patient or record identifiers here to drill into.\n\n"
                f"To list the actual records, ask a **fresh question in the main chat** — for example:\n"
                f"  • *'List all prescriptions with their patient details'*\n"
                f"  • *'Show me the prescriptions that make up this total'*\n"
                f"  • *'List patients with prescription counts'*"
            )
        # Multi-row aggregate (e.g. count per doctor)
        return (
            f"The current result only has summary columns: {col_summary}.\n"
            "Questions you can ask here:\n"
            + "\n".join(f"  • {s}" for s in suggestions[:3])
            + "\n\nTo see record-level details, ask a fresh question in the main chat."
        )

    # ── Normal result with identifier columns ─────────────────────────────────
    hint = (
        f"The current result only has these columns: {col_summary}.\n"
        "Questions you can ask:\n"
        + "\n".join(f"  • {s}" for s in suggestions[:4])
    )

    # If we have the actual result values, suggest a rephrased question
    # that names them explicitly so the main chat can answer it directly.
    if prev_rows and text_cols:
        key_col = text_cols[0]
        values  = list(dict.fromkeys(
            str(r[key_col]) for r in prev_rows if r.get(key_col) is not None
        ))[:5]
        if values:
            names = ", ".join(values)
            hint += (
                f"\n\nFor anything else, ask a fresh question in the main chat — "
                f"for example, name them explicitly:\n"
                f"  *'... for {names}'*"
            )
    else:
        hint += "\n\nFor anything else, ask a fresh question in the main chat."

    return hint


def _result_has_identifiers(schema: list[dict]) -> bool:
    """
    Return True if the cached result contains at least one text/categorical
    column that could serve as an entity identifier for a drill-down query.

    A pure-aggregate result (only numeric columns like TOTAL_PRESCRIPTIONS,
    AVG_CHARGES, etc.) has no identifier to filter on — drilling into it
    via the production DB is structurally impossible.
    """
    _TEXT_TYPES = {
        "TEXT", "VARCHAR", "STRING", "NVARCHAR", "CHAR",
        "CHARACTER VARYING", "NCHAR",
    }
    return any(
        s.get("type", "TEXT").upper().split("(")[0].strip() in _TEXT_TYPES
        for s in (schema or [])
    )


def _build_aggregate_drilldown_context(
    original_sql: str,
    original_question: str,
    follow_up_question: str,
) -> str:
    """
    Build a context block for the production-DB fallback when the previous
    result was a pure aggregate (COUNT/SUM/AVG — no identifier columns).

    Strategy: give the LLM the ORIGINAL aggregate SQL and ask it to rewrite
    as a detail query.  The WHERE conditions and FROM clause are preserved
    exactly; only the SELECT list changes to return the columns the user wants.

    This is far more reliable than reverse-engineering filter values from the
    result rows (which don't exist for aggregates).
    """
    if not original_sql:
        return ""

    lines = [
        "AGGREGATE DRILL-DOWN REQUEST:",
        "The user previously ran an aggregate query (COUNT/SUM/AVG) and now",
        "wants to see the underlying detail records that make up that total.",
        "",
        f'The original question was: "{original_question}"',
        "",
        "The original aggregate SQL was:",
        original_sql.strip(),
        "",
        "YOUR TASK:",
        "Rewrite that SQL as a detail query that:",
        "  1. Keeps the exact same FROM clause and WHERE conditions (if any).",
        "  2. Removes the aggregate function (COUNT, SUM, AVG, etc.) from SELECT.",
        "  3. Selects the specific columns the user is asking for (patient name,",
        "     prescription details, etc.) based on the follow-up question and the",
        "     Knowledge Base schema context.",
        "  4. Adds a row limit (TOP 20 / LIMIT 20) unless the user specified a number.",
        "",
        "CRITICAL: Use ONLY column names that appear verbatim in the Knowledge Base.",
        "Do NOT use the aggregate alias (e.g. TOTAL_PRESCRIPTIONS) as a column name —",
        "that was a SELECT alias, not a real column in the database.",
        "Do NOT invent column names. If unsure, return CANNOT_GENERATE.",
    ]
    return "\n".join(lines)


def _build_drilldown_context(
    prev_question: str,
    prev_rows: list[dict],
    schema: list[dict],
) -> str:
    """
    Build a context block injected into the production-DB fallback prompt
    so the LLM can answer follow-up questions that reference entities from
    the previous result ("these top 5 prescribers", "the ones shown", etc.).

    Extracts the distinct values for each TEXT/categorical column and emits
    them as filter-value hints.  IMPORTANT: only the VALUES are provided —
    the LLM must use its KB schema context to identify the correct production
    table and column name to filter on.  The result column names are aliases
    that may differ from the underlying DB column names.
    """
    if not prev_rows or not schema:
        return ""

    parts: list[str] = []
    if prev_question:
        parts.append(
            "The user is following up on a previous database result.\n"
            f'The previous question was: "{prev_question}"'
        )
    else:
        parts.append("The user is following up on a previous database result.")

    col_names = ", ".join(s["name"] for s in schema)
    parts.append(
        f"That result returned {len(prev_rows)} row(s) with columns: {col_names}."
    )

    # Collect VALUES (not column aliases) for up to 3 TEXT/categorical columns.
    # We deliberately do NOT tell the LLM to use the result column name directly,
    # because result aliases (e.g. "Formula_Name") may not exist in production tables.
    # The LLM should use these values in an IN-list while finding the right
    # table.column from the KB schema context.
    value_groups: list[tuple[str, list[str]]] = []
    for col_info in schema:
        col   = col_info["name"]
        dtype = col_info.get("type", "TEXT").upper()
        if dtype not in ("TEXT", "VARCHAR", "STRING", "NVARCHAR", "CHAR"):
            continue
        values = list(dict.fromkeys(
            str(r[col]) for r in prev_rows if r.get(col) is not None
        ))[:15]
        if values:
            value_groups.append((col, values))
        if len(value_groups) >= 3:
            break

    if value_groups:
        hint_lines: list[str] = []
        for col, values in value_groups:
            in_list = ", ".join(f"'{v}'" for v in values)
            hint_lines.append(
                f"  - The result column '{col}' contained these specific values: {in_list}"
            )
        parts.append(
            "The follow-up question references items from that result.\n"
            "Use the values below to build a WHERE ... IN (...) filter, but identify "
            "the correct production table and column name from the KB schema context — "
            "do NOT use the result column name directly as a database column:\n"
            + "\n".join(hint_lines)
        )

    return "\n".join(parts)


def _sanitize_rows(rows: list[dict]) -> list[dict]:
    """
    Convert non-JSON-serializable cell values to safe Python primitives.

    Covers the types that database drivers and DuckDB commonly return:
      decimal.Decimal  → float  (DB numeric/money columns)
      datetime / date  → ISO string
      bytes            → hex string
      anything else    → str()

    Safe to call on rows that are already clean — passes through
    int/float/str/None unchanged.
    """
    import decimal as _decimal
    import datetime as _datetime

    def _safe(v):
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        if isinstance(v, _decimal.Decimal):
            return float(v)
        if isinstance(v, (_datetime.datetime, _datetime.date)):
            return v.isoformat()
        if isinstance(v, bytes):
            return v.hex()
        return str(v)

    return [{k: _safe(val) for k, val in row.items()} for row in (rows or [])]


def _inject_distinct_if_needed(sql: str, question: str) -> str:
    """
    Safety net: if the LLM forgot SELECT DISTINCT on a list-entity query, add it.

    Fires only when ALL of:
      1. The question is a "list entities" question (list/show/who/which/find/get/what)
      2. The SQL has no aggregate functions (SUM/COUNT/AVG/MIN/MAX) in the SELECT clause
      3. The SQL has no GROUP BY clause
      4. SELECT DISTINCT is not already present

    This prevents duplicate rows when a dimension table is joined to a fact table
    without deduplication (e.g. prescribers × fill records = many rows per prescriber).
    """
    import re as _re

    # Only fire on list/entity questions
    _LIST_Q_RE = _re.compile(
        r"^\s*(list|show|who|which|find|get|what\s+(are|is)\s+the|"
        r"give\s+me|display|return|fetch|identify)",
        _re.IGNORECASE,
    )
    if not _LIST_Q_RE.match(question.strip()):
        return sql

    sql_upper = sql.upper()

    # Don't touch if already DISTINCT
    if "SELECT DISTINCT" in sql_upper:
        return sql

    # Don't touch aggregates — they deduplicate via GROUP BY
    _AGG_RE = _re.compile(
        r"\b(SUM|COUNT|AVG|MIN|MAX|STDEV|VARIANCE|PERCENTILE)\s*\(",
        _re.IGNORECASE,
    )
    if _AGG_RE.search(sql):
        return sql

    # Don't touch GROUP BY queries
    if "GROUP BY" in sql_upper:
        return sql

    # Don't touch subqueries that are already wrapped in an outer SELECT DISTINCT
    # Only patch the outermost SELECT
    patched = _re.sub(r"(?i)^(\s*SELECT\s+)", r"\1DISTINCT ", sql, count=1)
    if patched != sql:
        log.info("DISTINCT injected into list-entity SQL for question: %r", question[:80])
    return patched


def _rows_to_table(rows) -> str:
    """Format query results as a clean text table with smart value formatting."""
    if not rows:
        return "(no results)"
    headers = list(rows[0].keys())
    # Format all values first
    formatted = [{h: _format_value(r.get(h)) for h in headers} for r in rows]
    widths = {
        h: max(len(str(h)), max(len(f[h]) for f in formatted))
        for h in headers
    }
    head = " | ".join(str(h).ljust(widths[h]) for h in headers)
    sep  = "-+-".join("-" * widths[h] for h in headers)
    body = "\n".join(
        " | ".join(f[h].ljust(widths[h]) for h in headers)
        for f in formatted
    )
    return f"{head}\n{sep}\n{body}"


# ══════════════════════════════════════════════════════════════════════════════
# Dispatch
# ══════════════════════════════════════════════════════════════════════════════

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
    """Step 4 — Harvest successful query log entries into validated examples."""
    try:
        from core.examples import harvest_and_embed
        added = harvest_and_embed(account_id, chroma_dir)
        if added > 0:
            log.info("Harvested %d new examples from query log for %s", added, account_id)
    except Exception as e:
        log.error("Log harvest failed for %s: %s", account_id, e)


_HELP = (
    "*QueryBot* 🤖\n\nAsk any data question:\n"
    "  • _What is my total revenue this month?_\n"
    "  • _Show top 10 customers by value_\n"
    "  • _How many records were created last week?_\n\n"
    "*Commands:* `help` · `status` · `whoami`"
)


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
                matched_option_id: str | None = None    # Fix #2
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
                    matched_option_id = str(match.get("id") or "") or None   # Fix #2
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
                    selected_option_id=matched_option_id,    # Fix #2
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


# ══════════════════════════════════════════════════════════════════════════════
# Webhooks
# ══════════════════════════════════════════════════════════════════════════════

def _load_adapter(platform_type):
    platforms = store.list_platforms(platform_type)
    active    = [p for p in platforms if p.get("is_active")]
    if not active:
        raise HTTPException(503, detail=f"No active {platform_type} platform configured.")
    return get_adapter(platform_type, active[0]["credentials"])


@app.post("/webhook/zoom")
async def webhook_zoom(request: Request, bg: BackgroundTasks):
    body = await request.body()
    headers = dict(request.headers)
    try:
        adapter = _load_adapter("zoom")
    except HTTPException:
        return {"status": "not_configured"}
    try:
        payload = json.loads(body)
        if payload.get("event") == "endpoint.url_validation":
            challenge = adapter.handle_challenge(body)
            return JSONResponse(challenge) if challenge else {"status": "ok"}
    except Exception:
        pass
    if not await adapter.verify_request(body, headers):
        raise HTTPException(401, detail="Invalid Zoom signature")
    event = adapter.parse_event(body, headers)
    if not event:
        return {"status": "ignored"}
    # Fix #8 — webhook idempotency (Zoom retries at-least-once)
    if is_duplicate_event(event):
        return {"status": "duplicate"}
    remember_event(event)
    await dispatch(event.account_id, event, adapter, bg)
    return {"status": "ok"}


@app.post("/webhook/teams")
async def webhook_teams(request: Request, bg: BackgroundTasks):
    body = await request.body()
    headers = dict(request.headers)
    adapter = _load_adapter("teams")
    if not await adapter.verify_request(body, headers):
        raise HTTPException(401, detail="Invalid Teams auth")
    event = adapter.parse_event(body, headers)
    if not event:
        return {"status": "ignored"}
    # Fix #8 — webhook idempotency
    if is_duplicate_event(event):
        return {"type": "message", "text": ""}
    remember_event(event)
    await dispatch(event.account_id, event, adapter, bg)
    return {"type": "message", "text": ""}


@app.post("/webhook/slack")
async def webhook_slack(request: Request, bg: BackgroundTasks):
    body = await request.body()
    headers = dict(request.headers)
    adapter = _load_adapter("slack")
    try:
        payload = json.loads(body)
        if payload.get("type") == "url_verification":
            challenge = adapter.handle_challenge(body)
            return JSONResponse(challenge) if challenge else {"status": "ok"}
    except Exception:
        pass
    if not await adapter.verify_request(body, headers):
        raise HTTPException(401, detail="Invalid Slack signature")
    event = adapter.parse_event(body, headers)
    if not event:
        return {"status": "ignored"}
    # Fix #8 — webhook idempotency (Slack delivers event_id on retries)
    if is_duplicate_event(event):
        return {"status": "duplicate"}
    remember_event(event)
    await dispatch(event.account_id, event, adapter, bg)
    return {"status": "ok"}


@app.websocket("/ws/chat/{account_id}")
async def ws_chat(websocket: WebSocket, account_id: str):
    """
    WebSocket endpoint for the internal chat UI.
    Auth: portal session cookie must be present and valid.
    Messages route through the same dispatch() as Zoom/Teams/Slack.
    """
    from gateway.web_adapter import WebAdapter
    from fastapi import BackgroundTasks

    # Verify portal session from signed portal cookie
    from portal.routes import _read_session_value

    cookie = websocket.cookies.get("qb_portal_session")
    user_id = _read_session_value(cookie) if cookie else None

    if not user_id:
        await websocket.close(code=4001)
        return

    portal_user = store.get_user(user_id)
    if not portal_user or portal_user.get("account_id") != account_id:
        await websocket.close(code=4003)
        return

    # Check client has chat UI enabled
    client = store.get_client(account_id) or {}
    if not client.get("chat_ui_enabled"):
        await websocket.close(code=4004)
        return

    await websocket.accept()

    zoom_user_id = portal_user.get("zoom_user_id") or f"web_{user_id}"
    adapter      = WebAdapter(websocket, account_id, zoom_user_id)

    # Load known tables once for drill-down validation in insight calls
    _ws_state = get_state(account_id)
    _ws_known_tables = load_known_tables(_ws_state.get("schema_dir", ""))

    # Per-result-card conversation history for result_chat multi-turn memory.
    # Keyed by result_id; each value is a list of {question, sql, row_count}.
    # Cleared automatically when a new main query replaces the result card.
    _result_chat_histories: dict[str, list[dict]] = {}

    # Send welcome message with user name
    await websocket.send_json({
        "type":    "system",
        "content": f"Connected as {portal_user['name']}. Ask me anything about your data.",
    })

    log.info("WebSocket chat connected: user=%d account=%s", user_id, account_id)
    # History is NOT cleared on reconnect — the browser will send a
    # `history_sync` message immediately after opening the socket with
    # turns persisted in localStorage so multi-turn memory survives refreshes.

    try:
        while True:
            data = await websocket.receive_json()
            msg_type   = (data.get("type") or "").strip()
            action     = (data.get("action") or "").strip()
            context    = data.get("context") or {}
            text       = (data.get("text") or "").strip()
            # table_hint: FQN hint from clicked suggested question
            table_hint = (data.get("table_hint") or "").strip()
            # schema_hint: schema name selected by the user in the portal UI
            # e.g. "HR" or "PHARMACY" — filters allowed_tables to that schema only
            schema_hint = (data.get("schema_hint") or "").strip().upper()

            # ── history_sync: browser restores prior-session history ──────────
            if msg_type == "history_sync":
                incoming = data.get("history")
                if isinstance(incoming, list):
                    load_fn = getattr(adapter, "load_history", None)
                    if callable(load_fn):
                        load_fn(incoming)
                        log.debug("history_sync: loaded %d turn(s) from client", len(incoming))
                continue

            # ── result_chat: inline card chat — always DuckDB, no routing ────
            # Sent by the inline mini-chat panel inside each result card.
            # result_id tags the response so the browser renders it inside
            # the correct card rather than the main thread.
            if msg_type == "result_chat":
                rc_question  = (data.get("question") or "").strip()
                rc_result_id = (data.get("result_id") or "").strip()
                if not rc_question:
                    await websocket.send_json({
                        "type": "result_chat_error",
                        "result_id": rc_result_id,
                        "content": "Please type a question.",
                    })
                    continue
                await websocket.send_json({
                    "type": "result_chat_typing",
                    "result_id": rc_result_id,
                    "active": True,
                })
                _rc_start_ms    = int(time.time() * 1000)
                _rc_question_id = make_llm_audit_request_id()
                _rc_parent_qid  = getattr(adapter, "last_question_id", "") or ""
                _rc_pu_id       = portal_user.get("id") if portal_user else None
                _rc_trace_id = _trace_create(
                    account_id=account_id,
                    question_id=_rc_question_id,
                    parent_question_id=_rc_parent_qid,
                    question=rc_question,
                    portal_user_id=_rc_pu_id,
                    platform_user_id=zoom_user_id or "",
                    session_id=getattr(adapter, "session_id", "") or "",
                    request_source="portal",
                    route="result_chat",
                )
                _trace_step(_rc_trace_id, "receive_question", output_summary={"result_id": rc_result_id})
                try:
                    _sid = getattr(adapter, "session_id", None)
                    if not _sid or not result_cache.has_result(_sid):
                        _trace_finish(_rc_trace_id, status="error", answer_type="error", error_message="No cached result found")
                        await websocket.send_json({
                            "type": "result_chat_error",
                            "result_id": rc_result_id,
                            "content": "No cached result found. Please run a query first.",
                        })
                        continue

                    _rc_schema      = result_cache.get_schema(_sid)
                    _rc_stats       = result_cache.get_stats(_sid)
                    _rc_currency    = result_cache.get_currency_columns(_sid)
                    _rc_db_cfg      = get_client_db(account_id) or {}
                    _rc_history     = _result_chat_histories.get(rc_result_id, [])

                    _rc_sys = build_duckdb_system_prompt(
                        _rc_schema,
                        db_type   = _rc_db_cfg.get("db_type", "azure_sql"),
                        data_stats= _rc_stats,
                        history   = _rc_history,
                    )
                    _rc_sql = await _generate_duckdb_sql(rc_question, _rc_sys, client)
                    _trace_update(_rc_trace_id, generated_sql=_rc_sql, db_type="duckdb")
                    _trace_step(_rc_trace_id, "llm_generate_sql", output_summary={"sql": (_rc_sql or "")[:300]})

                    # ── DuckDB CANNOT_GENERATE → fallback to production DB ────
                    if not _rc_sql or _rc_sql.strip().upper() == "CANNOT_GENERATE":
                        _trace_update(_rc_trace_id, route="result_chat_db_fallback", sql_validation_status="cannot_generate")
                        log.info(
                            "result_chat CANNOT_GENERATE for %r — attempting production DB fallback "
                            "(strategy: %s)",
                            rc_question[:60],
                            "value-drill-down" if _result_has_identifiers(_rc_schema)
                            else "aggregate-rewrite",
                        )
                        _log_q(account_id, rc_question, "", 0, False,
                               "CANNOT_GENERATE→DB_FALLBACK", "result_chat", "duckdb", 0, 0,
                               int(time.time() * 1000) - _rc_start_ms,
                               portal_user_id=_rc_pu_id, zoom_user_id=zoom_user_id,
                               question_id=_rc_question_id,
                               parent_question_id=_rc_parent_qid)

                        # Try to answer from the production database using the
                        # KB context stored with the last result card.
                        # Skip entirely when the cached result is a pure aggregate
                        # (no text/identifier columns) — there are no entity values
                        # to build a WHERE filter from, so the LLM will always
                        # return CANNOT_GENERATE, wasting a round-trip.
                        _fb_rows = None
                        _fb_sql  = None
                        _fb_err  = None
                        _fb_has_ids = _result_has_identifiers(_rc_schema)
                        try:
                            _cached_result = getattr(adapter, "last_result", None) or {}
                            _fb_rag_ctx    = _cached_result.get("rag_context", "")
                            _fb_db_cfg     = _cached_result.get("db_cfg") or _rc_db_cfg
                            if _fb_db_cfg:
                                # Re-retrieve KB for the follow-up question so the LLM
                                # has the right schema context (e.g. prescriber/patient tables)
                                # rather than just the original question's KB context.
                                # Merge: follow-up context first (most relevant) + original.
                                try:
                                    _fb_retriever  = load_retriever(account_id)
                                    _fb_fresh_docs = _fb_retriever.retrieve(rc_question, n=6)
                                    _fb_fresh_ctx  = "\n\n---\n\n".join(_fb_fresh_docs)
                                    _fb_rag_ctx = (
                                        _fb_fresh_ctx + "\n\n---\n\n" + _fb_rag_ctx
                                        if _fb_rag_ctx else _fb_fresh_ctx
                                    )
                                except Exception as _ret_exc:
                                    log.debug("result_chat fallback KB re-retrieval failed: %s", _ret_exc)

                            if _fb_db_cfg and _fb_rag_ctx:
                                await websocket.send_json({
                                    "type": "result_chat_typing",
                                    "result_id": rc_result_id,
                                    "active": True,
                                    "message": "Querying your database for a complete answer…",
                                })
                                _fb_prov, _fb_model, _fb_key, _fb_az = resolve_provider(
                                    client, purpose="query"
                                )
                                _fb_system = build_sql_system_prompt(
                                    _fb_db_cfg.get("db_type", "azure_sql"),
                                    _fb_rag_ctx,
                                )
                                # Inject drill-down context.
                                # For aggregate results (no identifier columns): use the
                                # original SQL so the LLM can rewrite it as a detail query.
                                # For normal results (has identifier columns): inject the
                                # specific values from the result rows as a WHERE IN hint.
                                _prev_rows = _cached_result.get("rows") or []
                                if not _fb_has_ids:
                                    _drill_ctx = _build_aggregate_drilldown_context(
                                        original_sql=_cached_result.get("sql", ""),
                                        original_question=_cached_result.get("question", ""),
                                        follow_up_question=rc_question,
                                    )
                                else:
                                    _drill_ctx = _build_drilldown_context(
                                        _cached_result.get("question", ""),
                                        _prev_rows,
                                        _rc_schema,
                                    )
                                if _drill_ctx:
                                    _fb_system = _fb_system + "\n\n---\n\n" + _drill_ctx
                                _fb_sql_raw, _, _ = await llm_complete(
                                    _fb_system, rc_question,
                                    _fb_prov, _fb_model, _fb_key,
                                    max_tokens=512, **_fb_az,
                                )
                                log.info(
                                    "result_chat DB fallback generated SQL: %s",
                                    (_fb_sql_raw or "None")[:300],
                                )
                                if _fb_sql_raw and _fb_sql_raw.startswith("```"):
                                    _fb_sql_raw = "\n".join(
                                        _fb_sql_raw.split("\n")[1:]
                                    ).rsplit("```", 1)[0].strip()
                                _fb_sql_raw = _inject_distinct_if_needed(
                                    _fb_sql_raw or "", rc_question
                                )
                                if _fb_sql_raw and "CANNOT_GENERATE" not in _fb_sql_raw.upper():
                                    _fb_ok, _, _ = validate_sql(
                                        _fb_sql_raw, _ws_known_tables,
                                        _fb_db_cfg.get("db_type", "azure_sql"),
                                        None,
                                    )
                                    if _fb_ok:
                                        _fb_rows = run_query(
                                            _fb_db_cfg["credentials"],
                                            _fb_db_cfg["db_type"],
                                            _fb_sql_raw,
                                        )
                                        _fb_sql = _fb_sql_raw
                        except Exception as _fb_exc:
                            _fb_err = str(_fb_exc)
                            log.warning("result_chat DB fallback failed: %s", _fb_exc)

                        if _fb_rows is not None and _fb_sql:
                            _fb_dur = int(time.time() * 1000) - _rc_start_ms
                            _fb_rows = _sanitize_rows(_fb_rows)
                            _log_q(account_id, rc_question, _fb_sql, len(_fb_rows), True, "",
                                   "result_chat_db_fallback",
                                   _rc_db_cfg.get("db_type", "unknown"),
                                   0, 0, _fb_dur,
                                   portal_user_id=_rc_pu_id, zoom_user_id=zoom_user_id,
                                   question_id=_rc_question_id,
                                   parent_question_id=_rc_parent_qid)
                            # Store this fallback turn in history so follow-up questions
                            # have context even when DuckDB couldn't answer.
                            _rc_history.append({
                                "question":  rc_question,
                                "sql":       _fb_sql,
                                "row_count": len(_fb_rows),
                            })
                            _result_chat_histories[rc_result_id] = _rc_history[-5:]
                            await websocket.send_json({
                                "type":             "result_chat_response",
                                "result_id":        rc_result_id,
                                "question":         rc_question,
                                "sql":              _fb_sql,
                                "rows":             _fb_rows,
                                "row_count":        len(_fb_rows),
                                "source":           "database",
                                "source_note":      "Answer required a full database query.",
                                "currency_columns": _rc_currency,
                            })
                            _trace_update(
                                _rc_trace_id,
                                generated_sql=_fb_sql,
                                sql_validation_status="pass",
                                query_row_count=len(_fb_rows),
                                query_duration_ms=_fb_dur,
                            )
                            _trace_finish(_rc_trace_id, status="success", answer_type="table", row_count=len(_fb_rows), duration_ms=_fb_dur, final_answer_summary="Result-chat answered by production DB fallback")
                            log.info(
                                "result_chat DB fallback succeeded: %r → %d rows",
                                rc_question[:60], len(_fb_rows),
                            )
                        else:
                            # Both DuckDB and DB fallback failed — give a column-aware hint
                            # with a rephrasing tip that includes the actual values
                            _prev_rows_hint = (_cached_result.get("rows") or []) if _cached_result else []
                            _hint = _build_cannot_generate_hint(
                                _rc_schema, _rc_stats,
                                prev_rows=_prev_rows_hint,
                                prev_question=_cached_result.get("question", "") if _cached_result else "",
                            )
                            _trace_finish(_rc_trace_id, status="error", answer_type="error", error_message=_hint)
                            await websocket.send_json({
                                "type":      "result_chat_error",
                                "result_id": rc_result_id,
                                "content":   _hint,
                            })
                        continue

                    # ── DuckDB answered — run query ───────────────────────────
                    _rc_verdict = validate_duckdb_result_sql(_rc_sql)
                    _trace_update(
                        _rc_trace_id,
                        sql_validation_status="pass" if _rc_verdict.ok else "fail",
                        sql_validation_error="" if _rc_verdict.ok else _rc_verdict.reason,
                    )
                    _trace_step(
                        _rc_trace_id,
                        "validate_sql",
                        input_summary=_rc_sql,
                        output_summary=_rc_verdict.reason,
                        status="success" if _rc_verdict.ok else "error",
                        metadata={"code": _rc_verdict.code},
                    )
                    if not _rc_verdict.ok:
                        raise ValueError(f"DuckDB SQL rejected: {_rc_verdict.reason}")
                    _rc_rows   = _sanitize_rows(result_cache.query(_sid, _rc_sql))
                    _rc_dur_ms = int(time.time() * 1000) - _rc_start_ms
                    _trace_step(_rc_trace_id, "execute_sql", input_summary=_rc_sql, output_summary={"rows": len(_rc_rows)}, duration_ms=_rc_dur_ms)

                    _log_q(account_id, rc_question, _rc_sql, len(_rc_rows), True, "",
                           "result_chat", "duckdb", 0, 0, _rc_dur_ms,
                           portal_user_id=_rc_pu_id, zoom_user_id=zoom_user_id,
                           question_id=_rc_question_id,
                           parent_question_id=_rc_parent_qid)

                    # Update multi-turn history for this result card
                    _rc_history.append({
                        "question":  rc_question,
                        "sql":       _rc_sql,
                        "row_count": len(_rc_rows),
                    })
                    _result_chat_histories[rc_result_id] = _rc_history[-5:]

                    # Auto-chart detection
                    _rc_chart = None
                    try:
                        _rc_chart_type = detect_chart_type(_rc_rows, question=rc_question)
                        if _rc_chart_type:
                            _rc_chart = build_chart_payload(
                                _rc_rows, _rc_chart_type, title=rc_question
                            )
                    except Exception:
                        pass

                    # 1-sentence narration (lightweight LLM call, silent on failure)
                    _rc_narration = await _generate_result_narration(
                        rc_question, _rc_rows, _rc_currency, client
                    )

                    await websocket.send_json({
                        "type":             "result_chat_response",
                        "result_id":        rc_result_id,
                        "question":         rc_question,
                        "sql":              _rc_sql,
                        "rows":             _rc_rows,
                        "row_count":        len(_rc_rows),
                        "source":           "cache",
                        "currency_columns": _rc_currency,
                        "chart":            _rc_chart,
                        "narration":        _rc_narration or None,
                    })
                    _trace_finish(_rc_trace_id, status="success", answer_type="table", row_count=len(_rc_rows), duration_ms=_rc_dur_ms, final_answer_summary="Result-chat answered from DuckDB cache")
                    log.info(
                        "result_chat answered %r → %d rows via DuckDB (parent=%s, "
                        "chart=%s, narration=%s)",
                        rc_question[:60], len(_rc_rows), _rc_parent_qid[:16] or "none",
                        bool(_rc_chart), bool(_rc_narration),
                    )

                except Exception as _rce:
                    log.warning("result_chat error: %s", _rce)
                    _trace_finish(_rc_trace_id, status="error", answer_type="error", error_message=str(_rce))
                    _log_q(account_id, rc_question, "", 0, False, str(_rce),
                           "result_chat", "duckdb", 0, 0,
                           int(time.time() * 1000) - _rc_start_ms,
                           portal_user_id=_rc_pu_id, zoom_user_id=zoom_user_id,
                           question_id=_rc_question_id,
                           parent_question_id=_rc_parent_qid)
                    await websocket.send_json({
                        "type": "result_chat_error",
                        "result_id": rc_result_id,
                        "content": "Something went wrong. Please try again.",
                    })
                finally:
                    await websocket.send_json({
                        "type": "result_chat_typing",
                        "result_id": rc_result_id,
                        "active": False,
                    })
                continue

            if msg_type == "clarification_response":
                pending = get_pending(account_id, zoom_user_id)
                if not pending:
                    await websocket.send_json({"type": "error", "content": "That clarification is no longer active. Please ask the question again."})
                    continue
                cmeta = pending.get("clarification_meta") or {}
                opts = cmeta.get("options") or []
                selected_id = str(data.get("option_id") or "").strip()
                free_text = (data.get("text") or "").strip()

                # Fix #9 — if the pending has no options (pure free-text
                # clarification, e.g. from the plain LLM classifier fallback),
                # accept text. If it does have options, require an option_id
                # OR tolerant text match against the options.
                if opts:
                    selected = next(
                        (o for o in opts if str(o.get("id") or "") == selected_id),
                        None,
                    )
                    if not selected and free_text:
                        selected = resolve_option_text(opts, free_text)
                    if not selected:
                        send_prompt = getattr(adapter, "send_clarification_prompt", None)
                        if callable(send_prompt):
                            await send_prompt(
                                adapter.make_event(pending["original_q"]),
                                cmeta.get("question") or "Please choose one option.",
                                opts,
                            )
                        else:
                            await websocket.send_json({"type": "error", "content": "Please choose one of the available clarification options."})
                        continue
                    selected_text = str(selected.get("value") or selected.get("label") or "").strip()
                    selected_opt_id = str(selected.get("id") or "") or None   # Fix #2
                    combined, term_hint = combine_with_clarification(
                        pending["original_q"],
                        selected_text,
                        cmeta,
                        selected_option_id=selected_opt_id,                   # Fix #2
                    )
                    log_label = selected_text
                else:
                    # Free-text clarification (Fix #9)
                    if not free_text:
                        await websocket.send_json({"type": "error", "content": "Please type your clarification."})
                        continue
                    combined, term_hint = combine_with_clarification(
                        pending["original_q"],
                        free_text,
                        cmeta,
                    )
                    log_label = free_text

                await websocket.send_json({"type": "typing", "active": True})
                if term_hint:
                    combined = f"{combined}\n\n{term_hint}"
                clear_pending(account_id, zoom_user_id)
                log.info(
                    "WS clarification resolved for '%s' with reply '%s'",
                    pending["original_q"][:80], log_label[:80],
                )
                try:
                    await handle_query(
                        account_id,
                        adapter.make_event(combined),
                        adapter,
                        combined,
                        portal_user,
                        is_clarification=True,
                    )
                except Exception as e:
                    log.error("WS clarification handle_query error: %s", e)
                    await websocket.send_json({"type": "error", "content": "I hit an error while applying that clarification. Please try again."})
                finally:
                    await websocket.send_json({"type": "typing", "active": False})
                continue

            await websocket.send_json({"type": "typing", "active": True})

            if action:
                try:
                    # Prefer async LLM-powered insight if we have cached result
                    cached = adapter.last_result
                    if cached and cached.get("rows") is not None:
                        try:
                            provider, model, api_key, az_kwargs = resolve_provider(client, purpose="query")
                            from core.response_builder import generate_analysis_response
                            with llm_audit_scope(
                                account_id=account_id,
                                question=f"{action}: {cached.get('question', '')}".strip(),
                                enabled=bool(client.get("enable_llm_audit")),
                                request_id=make_llm_audit_request_id(),
                                question_id=getattr(adapter, "last_question_id", None) or "",
                                component="analysis",
                            ):
                                insight = await generate_analysis_response(
                                    action=action,
                                    rows=cached["rows"],
                                    question=cached.get("question", ""),
                                    provider=provider,
                                    model=model,
                                    api_key=api_key,
                                    original_sql=cached.get("sql", ""),
                                    db_cfg=cached.get("db_cfg"),
                                    context=cached.get("rag_context", ""),
                                    known_tables=_ws_known_tables,
                                    **az_kwargs,
                                )
                            await websocket.send_json(insight)
                        except Exception as insight_err:
                            log.warning("LLM insight failed, using static fallback: %s", insight_err)
                            from core.response_builder import build_analysis_response
                            await websocket.send_json(build_analysis_response(action, context))
                    else:
                        from core.response_builder import build_analysis_response
                        await websocket.send_json(build_analysis_response(action, context))
                finally:
                    await websocket.send_json({"type": "typing", "active": False})
                continue

            if not text:
                await websocket.send_json({"type": "typing", "active": False})
                continue

            # Detect "why" follow-up questions about the last result
            from core.insight import is_insight_question
            cached = adapter.last_result
            if is_insight_question(text) and cached and cached.get("rows"):
                try:
                    provider, model, api_key, az_kwargs = resolve_provider(client, purpose="query")
                    from core.response_builder import generate_analysis_response
                    with llm_audit_scope(
                        account_id=account_id,
                        question=text,
                        enabled=bool(client.get("enable_llm_audit")),
                        request_id=make_llm_audit_request_id(),
                        question_id=getattr(adapter, "last_question_id", None) or "",
                        component="analysis",
                    ):
                        insight = await generate_analysis_response(
                            action="why",
                            rows=cached["rows"],
                            question=cached.get("question", ""),
                            provider=provider,
                            model=model,
                            api_key=api_key,
                            follow_up=text,
                            original_sql=cached.get("sql", ""),
                            db_cfg=cached.get("db_cfg"),
                            context=cached.get("rag_context", ""),
                            known_tables=_ws_known_tables,
                            **az_kwargs,
                        )
                    await websocket.send_json(insight)
                    await websocket.send_json({"type": "typing", "active": False})
                    continue
                except Exception as e:
                    log.warning("Why-insight failed, falling through to normal query: %s", e)

            # Frontend renders the user message locally before send.
            # Only send processing / assistant events back over the socket.
            bg = BackgroundTasks()
            event = adapter.make_event(text)
            if table_hint:
                event.table_hint = table_hint
            if schema_hint:
                event.schema_hint = schema_hint
            await dispatch(account_id, event, adapter, bg, portal_user=portal_user)

            # Run any background tasks synchronously in WebSocket context
            for task in bg.tasks:
                try:
                    await task.func(*task.args, **task.kwargs)
                except Exception as e:
                    log.error("WS bg task error: %s", e)

            await websocket.send_json({"type": "typing", "active": False})

    except WebSocketDisconnect:
        log.info("WebSocket chat disconnected: user=%d account=%s", user_id, account_id)
    except Exception as e:
        log.error("WebSocket error for user %d: %s", user_id, e)
        try:
            await websocket.send_json({
                "type":    "error",
                "content": "Connection error. Please refresh and try again.",
            })
        except Exception:
            pass


@app.get("/health")
async def health():
    clients = store.list_clients()
    return {"status": "ok", "version": "2.0.0",
            "clients": len(clients),
            "ready": sum(1 for c in clients if c["state"] == "READY")}


@app.get("/")
async def root():
    return RedirectResponse("/admin")
