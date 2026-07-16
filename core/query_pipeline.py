"""
core/query_pipeline.py
──────────────────────
Main query pipeline extracted from main.py.

Covers:
  • _generate_duckdb_sql  — LLM-backed DuckDB SELECT for result-cache queries
  • handle_query()        — full query pipeline (~1,100 lines)
"""

from __future__ import annotations

import asyncio
import logging
import time

import store
from gateway import PlatformEvent
from core.llm import llm_complete, build_sql_system_prompt, resolve_provider
from core.examples import retrieve_similar_examples, format_examples_for_prompt
from core.clarification import (
    check_ambiguity_glossary_first, save_pending,
    build_schema_grounded_clarification_hint, extract_original_question,
)
from core.schema import run_query, load_known_tables, load_schema_columns
from core.knowledge import load_retriever
from core.validator import normalize_generated_sql, validate_sql
from core.query_semantics import analyze_query_intent, build_generic_query_hints
from core.graph_resolver import resolve_for_question as _graph_resolve
from core.llm_audit import llm_audit_scope, make_llm_audit_request_id
from core.result_cache import result_cache
from core.query_router import should_route_to_result_cache, build_duckdb_system_prompt
from core.duckdb_sql_validator import validate_duckdb_result_sql
from core.semantic_planner import build_semantic_field_plan
from core.semantic_model import (
    build_runtime_semantic_context, build_runtime_semantic_plan,
    build_field_plan_repair_note,
)
from core.metric_scope import metric_source_tables, resolve_metric_scope
from core.answer_rca import extract_sql_tables
from core.pipeline_context import (
    get_state, get_client_db, _merge_semantic_plans,
    check_query_limit, check_token_limit,
)
from core.pipeline_helpers import (
    _extract_kb_synonym_injection, _send_live_stage, _sql_preview,
    _count_tables_for_zero_row, _build_zero_row_message,
    _format_metric_formula_context, _extract_metric_formula_tables,
    _build_row_metric_join_sql, attempt_field_plan_repair,
    _clamp_kb_doc, _clamp_prompt_context,
)
from core.pipeline_trace import (
    _log_q, _trace_create, _trace_update, _trace_step, _trace_finish,
    _create_learning_candidate,
)
from core.result_renderer import (
    _send_results, _inject_distinct_if_needed,
)
from core.compliance.governed_query import (
    PolicyDeniedError, execute_governed_query,
)
from core.compliance.models import ResourceRef
from core.compliance.policy_engine import evaluate as evaluate_policy, resolve_context

log = logging.getLogger("querybot")


def _table_matches_policy_scope(table: str, scope: set[str]) -> bool:
    table = table.upper()
    return any(
        table == candidate
        or table.endswith("." + candidate)
        or candidate.endswith("." + table)
        for candidate in scope
    )

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
# Why/causal insight — channel-agnostic follow-through
# ══════════════════════════════════════════════════════════════════════════════
# "Why did revenue drop last month?" used to get a real causal analysis only on
# the portal, and only when a previous result was still cached — on Teams/Zoom/
# Slack (or with no cached result) it became a plain SQL attempt that cannot
# answer causality. Now: the factual query runs first as usual, then the same
# analysis engine that powers the portal's "why" follow-ups runs on the fresh
# rows and its narrative is sent as a second message on ANY channel.

def _format_insight_markdown(insight: dict) -> str:
    """Render an assistant_analysis payload as chat-channel markdown.
    The portal renders the raw payload natively (send_analysis_response);
    this is the fallback for adapters that only speak text."""
    parts: list[str] = []
    headline = (insight.get("headline") or "").strip()
    if headline:
        parts.append(f"*{headline}*")
    body = (insight.get("body") or "").strip()
    if body:
        parts.append(body)
    bullets = [str(b).strip() for b in (insight.get("bullets") or []) if str(b).strip()]
    if bullets:
        parts.append("\n".join(f"  • {b}" for b in bullets))
    next_step = (insight.get("next_step") or "").strip()
    if next_step:
        parts.append(f"_Next step: {next_step}_")
    return "\n\n".join(parts).strip()


async def _send_why_insight(
    adapter, event, *,
    question: str,
    rows: list,
    sql: str,
    client: dict,
    account_id: str,
    db_cfg: dict,
    rag_context: str = "",
    known_tables: set | None = None,
    query_executor=None,
    question_id: str = "",
) -> None:
    """Generate and send a causal analysis of `rows` after the factual answer.
    Best-effort: the factual answer is already on the wire, so any failure
    here is logged and swallowed — never surfaced as a user-facing error."""
    from core.compliance.policy_engine import result_llm_features_allowed
    if not result_llm_features_allowed(account_id):
        # Silent skip (matches this function's own failure-swallowing
        # contract) — but still leave a proof-of-refusal audit row, since
        # this is an auto-triggered second message the user never asked
        # for, not an action button click that deserves a visible reply.
        with llm_audit_scope(
            account_id=account_id,
            question=f"why: {question}"[:500],
            enabled=bool(client.get("enable_llm_audit")),
            request_id=make_llm_audit_request_id(),
            question_id=question_id,
            component="analysis",
        ):
            from core.llm_audit import record_llm_blocked
            record_llm_blocked(
                "analysis",
                "why-insight blocked — regulated tenant, LLM never received result rows.",
            )
        return
    try:
        from core.response_builder import generate_analysis_response
        provider, model, api_key, az_kwargs = resolve_provider(client, purpose="query")
        with llm_audit_scope(
            account_id=account_id,
            question=f"why: {question}"[:500],
            enabled=bool(client.get("enable_llm_audit")),
            request_id=make_llm_audit_request_id(),
            question_id=question_id,
            component="analysis",
        ):
            insight = await generate_analysis_response(
                action="why",
                rows=rows,
                question=question,
                provider=provider,
                model=model,
                api_key=api_key,
                account_id=account_id,
                follow_up=question,
                original_sql=sql,
                db_cfg=db_cfg,
                context=rag_context,
                known_tables=known_tables,
                query_executor=query_executor,
                **az_kwargs,
            )
        _send_analysis = getattr(adapter, "send_analysis_response", None)
        if callable(_send_analysis):
            await _send_analysis(event, insight)
            return
        text = _format_insight_markdown(insight)
        if text:
            await adapter.send_message(event, text)
    except Exception as exc:
        log.warning("Why-insight after factual answer failed (answer already sent): %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# Query pipeline — table-aware
# ══════════════════════════════════════════════════════════════════════════════

async def handle_query(account_id, event, adapter, question, portal_user, is_clarification=False):
    start_ms = int(time.time() * 1000)
    state    = get_state(account_id)
    db_cfg   = get_client_db(account_id)
    client   = store.get_client(account_id) or {}

    # Activate the client's terminology packs for this request. Default (no
    # packs) equals the builtin vocabulary, so behavior is unchanged for
    # clients that have not enabled any pack. ContextVars do not cross
    # run_in_executor threads — anything vocab-dependent called through an
    # executor must take vocab= explicitly.
    from core.vocab_packs import vocab_for_account, activate_vocab
    _vocab = vocab_for_account(account_id)
    activate_vocab(_vocab)
    audit_enabled = bool(client.get("enable_llm_audit"))
    audit_request_id = make_llm_audit_request_id()

    # Channel-agnostic why-route: explicitly causal questions get the factual
    # answer first (normal pipeline below), then a causal analysis of the
    # fresh rows as a second message. Clarification replies are exempt — the
    # causal wording there belongs to the original question, already handled.
    from core.insight import is_causal_question
    _why_mode = bool(not is_clarification and is_causal_question(question))

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
    all_columns    = load_schema_columns(state.get("schema_dir", ""))

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
    compliance_profile = store.get_compliance_profile(account_id)
    compliance_context = resolve_context(
        account_id,
        portal_user,
        action="query_execution",
        channel=getattr(event, "platform", "") or "portal",
        purpose_id=getattr(event, "purpose_id", "") or "",
        provider=provider,
        break_glass_grant_id=getattr(event, "break_glass_grant_id", None),
    )

    def _execute_with_policy(candidate_sql: str, semantic: dict | None = None):
        context = resolve_context(
            account_id,
            portal_user,
            action="query_execution",
            channel=getattr(event, "platform", "") or "portal",
            purpose_id=compliance_context.purpose_id,
            provider=provider,
            break_glass_grant_id=compliance_context.break_glass_grant_id,
        )
        return execute_governed_query(
            db_cfg["credentials"],
            db_cfg["db_type"],
            candidate_sql,
            context=context,
            known_tables=all_known,
            table_columns=all_columns,
            allowed_tables=effective,
            semantic_context=semantic,
        )

    if compliance_profile.get("mode") == "regulated":
        classification_map = store.get_classification_map(account_id)
        scoped_resources = []
        for key in classification_map:
            table, _, column = key.rpartition(".")
            if _table_matches_policy_scope(table, effective):
                scoped_resources.append(ResourceRef(table=table, column=column))
        llm_context = resolve_context(
            account_id,
            portal_user,
            action="llm_context",
            channel=getattr(event, "platform", "") or "portal",
            purpose_id=compliance_context.purpose_id,
            provider=provider,
            break_glass_grant_id=compliance_context.break_glass_grant_id,
        )
        llm_decision = evaluate_policy(llm_context, scoped_resources)
        _trace_step(
            trace_id,
            "regulated_llm_context",
            output_summary={
                "allowed": llm_decision.effective_allowed,
                "reason": llm_decision.reason_code,
                "policy_version": llm_decision.policy_version,
            },
            status="success" if llm_decision.effective_allowed else "error",
        )
        if not llm_decision.effective_allowed:
            _trace_finish(
                trace_id,
                status="error",
                answer_type="policy_denied",
                error_message=llm_decision.explanation,
            )
            await adapter.send_message(
                event,
                "This request is blocked by the workspace data policy. "
                f"Reason: {llm_decision.explanation}",
            )
            return

    _session_id = getattr(adapter, "session_id", None)
    _cached_cols = [s["name"] for s in result_cache.get_schema(_session_id)] if _session_id else []
    if _session_id and result_cache.has_result(_session_id):
        cache_context = resolve_context(
            account_id,
            portal_user,
            action="cache_read",
            channel=getattr(event, "platform", "") or "portal",
            purpose_id=compliance_context.purpose_id,
            provider=provider,
        )
        cache_decision = evaluate_policy(cache_context, [])
        if not cache_decision.effective_allowed:
            result_cache.clear(_session_id)
        elif compliance_profile.get("mode") == "regulated":
            _trace_step(
                trace_id,
                "regulated_cache_read",
                output_summary={"reason": cache_decision.reason_code},
            )
    if _session_id and should_route_to_result_cache(question, result_cache.has_result(_session_id), cached_col_names=_cached_cols):
        _trace_update(trace_id, route="duckdb_cache")
        _trace_step(trace_id, "route", output_summary="duckdb_cache")
        log.info("Routing to DuckDB result cache for session %s", _session_id[:16])

        # Regulated tenants: the cache_read check above only governs whether
        # the cache may be used at all (its default rule always allows it) —
        # it does not verify a BAA for whatever's actually IN the cached
        # table before building a new prompt that embeds sample values from
        # it. Re-derive the resources from the ORIGINAL cached SQL (not just
        # the user's broader effective-table scope, which the front-door
        # llm_context check above already covers) and re-run the same
        # BAA/prohibited-data check specifically scoped to this cache entry.
        if compliance_profile.get("mode") == "regulated":
            _cached_sql = result_cache.get_sql(_session_id)
            if _cached_sql:
                from core.compliance.sql_guard import analyze_sql

                _cache_analysis = analyze_sql(_cached_sql, db_cfg.get("db_type", "azure_sql"))
                _cache_llm_context = resolve_context(
                    account_id, portal_user, action="llm_context",
                    channel=getattr(event, "platform", "") or "portal",
                    purpose_id=compliance_context.purpose_id, provider=provider,
                    break_glass_grant_id=compliance_context.break_glass_grant_id,
                )
                _cache_llm_decision = evaluate_policy(_cache_llm_context, _cache_analysis.resources)
                _trace_step(
                    trace_id, "duckdb_cache_llm_context",
                    output_summary={
                        "allowed": _cache_llm_decision.effective_allowed,
                        "reason": _cache_llm_decision.reason_code,
                    },
                    status="success" if _cache_llm_decision.effective_allowed else "error",
                )
                if not _cache_llm_decision.effective_allowed:
                    with llm_audit_scope(
                        account_id=account_id, question=question,
                        enabled=bool(client.get("enable_llm_audit")),
                        request_id=make_llm_audit_request_id(),
                        question_id=audit_request_id, component="duckdb_cache",
                    ):
                        from core.llm_audit import record_llm_blocked
                        record_llm_blocked(
                            "duckdb_cache",
                            f"DuckDB follow-up blocked — {_cache_llm_decision.explanation}",
                        )
                    _trace_finish(
                        trace_id, status="error", answer_type="policy_denied",
                        error_message=_cache_llm_decision.explanation,
                    )
                    await adapter.send_message(
                        event,
                        "This request is blocked by the workspace data policy. "
                        f"Reason: {_cache_llm_decision.explanation}",
                    )
                    return

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
                                        question_id=audit_request_id,
                                        explicit_column_formats=(
                                            result_cache.get_column_formats(_session_id)
                                        ))
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
        # Warn when user asks for a dimensional breakdown of a query-type metric
        import re as _re_grp
        if matched_metric.get("formula_type") == "query" and _re_grp.search(r'\b(by|per|for each|grouped by|split by|breakdown)\b', question, _re_grp.IGNORECASE):
            await adapter.send_message(
                event,
                f"ℹ️ **{matched_metric.get('label') or matched_metric.get('name', 'This metric')}** is a fixed SQL query — "
                "it returns an overall value and cannot be broken down by individual dimensions. "
                "Showing the overall result:",
            )
        log.info("Metric registry hit: %s → %s", matched_metric["name"], sql_from_metric[:60])
        _metric_exec_t0 = time.time()
        try:
            await _send_live_stage(adapter, event, "executing_query", "Running query", "Executing the trusted metric query against your database.")
            _loop = asyncio.get_running_loop()
            try:
                governed = await asyncio.wait_for(
                    _loop.run_in_executor(None, _execute_with_policy, sql_from_metric),
                    timeout=180.0,
                )
            except asyncio.TimeoutError:
                await adapter.send_message(
                    event,
                    "⏱ Query timed out after 3 minutes. Try adding a filter (e.g. date range or specific customer) to narrow the result.",
                )
                _trace_finish(trace_id, status="error", answer_type="timeout", error_message="Metric query timed out after 60s")
                return
            rows = governed.rows
            sql_from_metric = governed.sql
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
                                question_id=audit_request_id,
                                display_context={
                                    "format_scope": "metric_registry",
                                    "metrics": [matched_metric],
                                })
            if _why_mode and rows:
                await _send_why_insight(
                    adapter, event,
                    question=question, rows=rows, sql=sql_from_metric,
                    client=client, account_id=account_id, db_cfg=db_cfg,
                    known_tables=all_known,
                    query_executor=lambda _cfg, _s: _execute_with_policy(_s),
                    question_id=audit_request_id,
                )
            _trace_finish(trace_id, status="success", answer_type="table", row_count=len(rows), duration_ms=duration_ms, final_answer_summary="Answered by metric registry")
            return
        except PolicyDeniedError as policy_error:
            _trace_finish(
                trace_id,
                status="error",
                answer_type="policy_denied",
                error_message=policy_error.decision.explanation,
            )
            await adapter.send_message(
                event,
                "This metric is blocked by the workspace data policy. "
                f"Reason: {policy_error.decision.explanation}",
            )
            return
        except Exception as e:
            _trace_step(trace_id, "execute_sql", input_summary=sql_from_metric, output_summary=str(e), status="error", duration_ms=int((time.time() - _metric_exec_t0) * 1000))
            log.warning("Metric registry SQL failed, falling through to LLM: %s", e)
            # Fall through to normal LLM pipeline

    # ── RAG — scoped to effective tables ─────────────────────────────────────
    # The retriever filters by doc_id (table name), not by substring match,
    # so disallowed tables never leak into the LLM prompt. allowed_tables
    # is passed through explicitly; None means admin/unrestricted.
    _kb_phase_t0 = time.time()
    _weak_retrieval = False
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
        _weak_retrieval = bool(getattr(retriever, "last_retrieval_weak", False))

        pinned    = [d for d in relevant_kbs if retriever._is_global(d)]
        table_kbs = [d for d in relevant_kbs if not retriever._is_global(d)]

        if _grouping:
            fact_patterns = retriever.retrieve_fact_patterns(
                question, n=2, allowed_tables=rag_filter,
            )
            for fp in fact_patterns:
                if fp not in (pinned + table_kbs):
                    table_kbs.insert(0, fp)

        # ── Multi-schema coherence (no schema_hint = "All" mode) ─────────────
        # When the user is in "All" mode across multiple schemas, the semantic
        # search can return KB docs from different schemas. If one schema
        # dominates the top results (≥60%, ≥2 docs), do a focused re-retrieval
        # scoped only to that schema's tables so the LLM gets clean, single-
        # schema context instead of a mix.
        if not schema_hint and table_kbs:
            _schema_votes: dict[str, int] = {}
            for _doc in table_kbs:
                _first_line = _doc.splitlines()[0].strip().lstrip("#").strip()
                _parts = _first_line.upper().split(".")
                if len(_parts) >= 2:
                    _sch = _parts[-2].strip("[]")
                    if _sch and _sch not in {"DBO", "SYS", "INFORMATION_SCHEMA", "GUEST"}:
                        _schema_votes[_sch] = _schema_votes.get(_sch, 0) + 1
            if _schema_votes:
                _dominant_sch = max(_schema_votes, key=_schema_votes.get)
                _total_votes  = sum(_schema_votes.values())
                _dom_ratio    = _schema_votes[_dominant_sch] / _total_votes
                if _dom_ratio >= 0.6 and _total_votes >= 2 and len(_schema_votes) > 1:
                    # Build a focused filter for just the dominant schema
                    _base_pool = effective if effective else all_known
                    _focused = {
                        t for t in _base_pool
                        if len(t.split(".")) >= 2
                        and t.upper().split(".")[-2].strip("[]") == _dominant_sch
                    }
                    if _focused:
                        _focused_kbs = retriever.retrieve(
                            question, n=_n, allowed_tables=_focused
                        )
                        _focused_table_kbs = [
                            d for d in _focused_kbs if not retriever._is_global(d)
                        ]
                        if len(_focused_table_kbs) >= 2:
                            table_kbs = _focused_table_kbs
                            log.info(
                                "Multi-schema: re-retrieved focused on %s "
                                "(ratio=%.0f%%, schemas_seen=%d)",
                                _dominant_sch, _dom_ratio * 100, len(_schema_votes),
                            )

        relevant_kbs = [_clamp_kb_doc(d) for d in (pinned + table_kbs)[:7]]
        context = "\n\n---\n\n".join(relevant_kbs)
        _trace_update(
            trace_id,
            route="normal_sql",
            retrieved_kb_chunk_ids=store.kb_chunk_refs(relevant_kbs),
        )
        _trace_step(
            trace_id, "retrieve_kb",
            output_summary={"chunks": len(relevant_kbs)},
            duration_ms=int((time.time() - _kb_phase_t0) * 1000),
        )

        # ── Step 2: Retrieve validated SQL examples — few-shot grounding ─────
        # Examples now live in Qdrant alongside KB docs — no chroma_dir needed
        _examples_t0 = time.time()
        examples = retrieve_similar_examples(
            question, account_id, n=3,
            allowed_tables=rag_filter,
            schema_scope=schema_hint,
            kb_dir=state.get("kb_dir", ""),
        )
        if examples:
            context = format_examples_for_prompt(examples) + "\n\n---\n\n" + context
            log.info("Injected %d validated examples into prompt", len(examples))
            _trace_step(
                trace_id, "retrieve_examples",
                output_summary={"examples": len(examples)},
                duration_ms=int((time.time() - _examples_t0) * 1000),
            )

    except Exception as e:
        log.error("RAG retrieval failed: %s", e)
        _trace_finish(trace_id, status="error", answer_type="error", error_message=f"RAG retrieval failed: {e}")
        await adapter.send_message(event, "⚠️ Knowledge Base not ready.")
        return

    # ── Compiled semantic contract — the single runtime truth source ─────────
    # All approved semantics (model, metrics, graph, terms) come from ONE
    # versioned artifact, recompiled on every admin approval. The consumers
    # below take the contract's sections instead of re-reading each store, so
    # the contract_version stamped on this answer is exactly what was used.
    # Every consumer falls back to its own store read when the contract is
    # absent (accounts that predate the first compile).
    from core.semantic_contract import load_contract
    _contract = load_contract(state.get("kb_dir", ""))
    _contract_version = (_contract.get("meta") or {}).get("contract_version", "")
    if _contract_version:
        _trace_update(trace_id, contract_version=_contract_version)
    _contract_model = _contract.get("model") if _contract else None
    _contract_metrics = _contract.get("metrics") if _contract else None
    _contract_terms = _contract.get("terms") if _contract else None

    # SQL generation — inject any matched business-glossary terms as grounding hints
    term_injection = store.build_term_injection(
        account_id, question, query_scope_tables, terms=_contract_terms,
    )
    schema_grounded_hint = build_schema_grounded_clarification_hint(
        account_id,
        question,
        context,
        allowed_tables=query_scope_tables,
    )
    # Value grounding: resolve user-typed filter literals ("emco corp") to
    # exact database values ("EMCO Corporation") via the per-client value
    # index, so the LLM writes WHERE literals that actually exist. The index
    # covers high-cardinality display columns that schema discovery's
    # 30-distinct-value cap excludes from the KB entirely.
    verified_values_hint = ""
    _value_clarify: list[dict] = []
    try:
        from core.value_index import value_index_enabled
        from core.value_resolver import (
            resolve_literals, build_verified_values_injection,
            build_known_terms,
        )
        if value_index_enabled(state):
            _known_terms = build_known_terms(account_id, all_columns)
            _resolved_values = resolve_literals(
                account_id, question, allowed_tables=query_scope_tables,
                known_terms=_known_terms,
            )
            verified_values_hint = build_verified_values_injection(_resolved_values)
            _value_clarify = _resolved_values.get("clarify") or []
            if verified_values_hint or _value_clarify:
                _trace_step(
                    trace_id,
                    "value_resolution",
                    output_summary={
                        "verified": len(_resolved_values.get("verified") or []),
                        "in_lists": len(_resolved_values.get("in_lists") or []),
                        "clarify": len(_value_clarify),
                    },
                )
    except Exception as _vr_exc:
        log.debug("Value resolution skipped: %s", _vr_exc)
    generic_hints = build_generic_query_hints(question)
    query_intent = analyze_query_intent(question)
    # Candidate metrics are account-wide. We delay injecting/enforcing them
    # until graph + semantic planning has inferred the question's schema/domain.
    _metric_candidates = store.list_metric_formula_context(
        account_id, question, limit=10, metrics=_contract_metrics,
    )
    _matched_metrics: list[dict] = []
    metric_formula_context = ""
    _metric_formula_tables: set[str] = set()

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
            f"ACTIVE SCHEMA LOCK — {schema_hint}:\n"
            f"The user has explicitly locked the query to the {schema_hint} schema.\n"
            f"MANDATORY RULES:\n"
            f"1. Use ONLY tables and columns that belong to the {schema_hint} schema.\n"
            f"2. NEVER use column names from any other schema (e.g. if a column name appears "
            f"in a different schema's documents it is FORBIDDEN here — do not copy it).\n"
            f"3. Every column name you write MUST appear verbatim in the Knowledge Base "
            f"documents provided in this prompt.\n"
            f"4. If the user asks for a concept (e.g. 'revenue') and you cannot find its "
            f"exact column name in the {schema_hint} KB documents, return CANNOT_GENERATE "
            f"and explain which column is missing — NEVER borrow a name from another schema."
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
            verified_values_hint,
            generic_hints,
            context,
        )
        if part
    ]
    context_with_terms = "\n\n".join(context_parts)

    # Value-resolution ambiguity across DIFFERENT columns ("Emco" matches a
    # customer name AND an item description) can't be settled deterministically
    # or by the LLM — ask the user, mirroring the metric-scope clarification.
    if _value_clarify and not is_clarification:
        _vc = _value_clarify[0]
        _vc_options = [
            {
                "label": f"{opt['value']} ({opt.get('business_name') or opt['column']})",
                "value": opt["value"],
            }
            for opt in (_vc.get("options") or [])[:5]
        ]
        if _vc_options:
            clarifying_q = (
                f"'{_vc['phrase']}' matches more than one thing in your data. "
                f"Which one did you mean?"
            )
            if event.user_id:
                save_pending(
                    account_id,
                    event.user_id,
                    question,
                    context_with_terms,
                    clarification_meta={
                        "term": _vc["phrase"],
                        "options": _vc_options,
                        "source": "value_resolver",
                    },
                )
            send_prompt = getattr(adapter, "send_clarification_prompt", None)
            if callable(send_prompt):
                await send_prompt(event, clarifying_q, _vc_options)
            else:
                option_lines = "\n".join(f"  • {o['label']}" for o in _vc_options)
                await adapter.send_message(
                    event,
                    f"❓ {clarifying_q}\n\n{option_lines}\n\n"
                    "_Reply with one of the options above._",
                )
            return

    try:
        _semantic_model_context = build_runtime_semantic_context(
            state.get("kb_dir", ""),
            question=question,
            selected_schema=schema_hint,
            model=_contract_model,
        )
        if _semantic_model_context:
            context_with_terms = _semantic_model_context + "\n\n" + context_with_terms
            _trace_step(
                trace_id,
                "semantic_model_context",
                output_summary={"enabled": True, "schema": schema_hint or "ALL"},
            )
    except Exception as _sm_exc:
        log.debug("Structured semantic model context skipped: %s", _sm_exc)

    # Multi-turn memory: inject conversation history for web portal sessions
    _conv_history = []
    _history_fn = getattr(adapter, "get_history", None)
    if callable(_history_fn):
        _conv_history = _history_fn()

    # Entity graph — deterministic JOIN skeleton resolution
    # Loads the client's entity graph once and resolves JOINs from the question
    # before the LLM is called so the LLM never guesses table relationships.
    _graph_ctx: dict = {}
    _full_graph: dict = {}
    try:
        _full_graph = (
            dict(_contract.get("graph") or {}) if _contract.get("graph")
            else store.get_full_graph(account_id)
        )
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
                intent=query_intent,
                metric_formula_tables=set(),
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

    # Deterministic field-plan builders scan the question text for literal
    # aliases/business terms. On a clarification retry, `question` is the
    # original question PLUS the raw clarification wrapper (e.g. a chip
    # label like "Synonyms: customer id, customer key") — that wrapper is
    # UI metadata, not natural language, and can spuriously match column
    # aliases (e.g. "key" against a *_DMS_KEY column). Strip it back out
    # for field-plan purposes only; the LLM-facing prompt still gets the
    # full `question` with clarification context further below.
    _semantic_plan_question = extract_original_question(question)

    _semantic_plan = {}
    try:
        _semantic_plan = build_semantic_field_plan(
            _semantic_plan_question,
            all_columns,
            query_scope_tables,
            selected_schema=schema_hint,
            vocab=_vocab,
        )
        if _semantic_plan.get("enabled"):
            _trace_step(
                trace_id,
                "semantic_field_plan",
                output_summary={
                    "fields": [
                        f"{f.get('term')}={f.get('table')}.{f.get('column')}"
                        for f in _semantic_plan.get("fields", [])
                    ],
                    "joins": len(_semantic_plan.get("joins") or []),
                },
            )
            log.info(
                "Semantic field plan for %s: fields=%s joins=%d",
                account_id,
                [
                    f"{f.get('term')}={f.get('table')}.{f.get('column')}"
                    for f in _semantic_plan.get("fields", [])
                ],
                len(_semantic_plan.get("joins") or []),
            )
    except Exception as _sp_exc:
        _semantic_plan = {}
        log.debug("Semantic field planning skipped: %s", _sp_exc)

    _semantic_model_plan = {}
    try:
        _semantic_model_plan = build_runtime_semantic_plan(
            state.get("kb_dir", ""),
            question=_semantic_plan_question,
            selected_schema=schema_hint,
            model=_contract_model,
        )
        if _semantic_model_plan.get("enabled"):
            _trace_step(
                trace_id,
                "semantic_model_plan",
                output_summary={
                    "fields": [
                        f"{f.get('term')}={f.get('table')}.{f.get('column')}"
                        for f in _semantic_model_plan.get("fields", [])
                    ],
                    "joins": len(_semantic_model_plan.get("joins") or []),
                },
            )
    except Exception as _smp_exc:
        _semantic_model_plan = {}
        log.warning("Structured semantic model planning failed — approved field "
                    "enforcement inactive for this question: %s", _smp_exc)

    _semantic_plan = _merge_semantic_plans(_semantic_plan, _semantic_model_plan)

    # ── Table-coverage fallback when the entity graph didn't resolve ──────────
    # The graph-driven gap-fill above is gated on _graph_ctx["enabled"] — an
    # entity with no table_name, or a disabled/empty graph, silently killed
    # the ONLY structural recovery for retrieval misses. The merged semantic
    # plan is a second independent source of "tables this answer definitely
    # needs": approved-field and planner mappings name exact tables. When the
    # graph produced nothing, gap-fill against those instead.
    if not _graph_ctx.get("enabled"):
        try:
            from core.table_coverage import guarantee_table_coverage
            _plan_fqns = {
                str(f.get("table") or "").upper()
                for f in (_semantic_plan or {}).get("fields") or []
                if f.get("table") and "." in str(f.get("table"))
            }
            if _plan_fqns:
                _plan_gap_docs = guarantee_table_coverage(
                    account_id     = account_id,
                    required_fqns  = _plan_fqns,
                    retrieved_docs = relevant_kbs,
                    rag_filter     = rag_filter,
                    max_fill       = 3,
                )
                if _plan_gap_docs:
                    context_with_terms = (
                        context_with_terms
                        + "\n\n---\n\n"
                        + "\n\n---\n\n".join(_plan_gap_docs)
                    )
                    log.info(
                        "Table coverage (semantic-plan fallback): injected %d "
                        "gap-fill doc(s) — graph was unavailable",
                        len(_plan_gap_docs),
                    )
        except Exception as _plan_cov_exc:
            log.debug("Semantic-plan coverage fallback skipped: %s", _plan_cov_exc)

    # Build entity→schema lookup from the (possibly schema-filtered) full graph.
    # Metrics have a base_entity field; this map lets metric_source_schemas() use
    # the entity graph's schema_name directly — more reliable than parsing bare
    # base_table names or matching required_columns against _schema.json.
    _entity_schema_map: dict[str, str] = {
        e["entity_name"]: (e.get("schema_name") or "").upper().strip()
        for e in (_full_graph.get("entities") or [])
        if (e.get("schema_name") or "").strip()
    }

    # Metric scoping happens after graph + semantic planning so "revenue by
    # prescriber" can choose the Pharmacy revenue metric, while "revenue by
    # warehouse" can choose the Profitability metric.  In All-schema mode, a
    # bare "revenue" question remains ambiguous and should ask the user.
    _metric_scope = resolve_metric_scope(
        _metric_candidates,
        question,
        all_columns,
        selected_schema=schema_hint,
        graph_context=_graph_ctx,
        graph=_full_graph,
        semantic_plan=_semantic_plan,
        entity_schema_map=_entity_schema_map or None,
        limit=6,
    )
    if _metric_scope.ambiguous and not is_clarification:
        options = _metric_scope.options or []
        clarifying_q = (
            "I found more than one revenue definition. Which one should I use?"
        )
        if event.user_id and options:
            save_pending(
                account_id,
                event.user_id,
                question,
                context_with_terms,
                clarification_meta={
                    "term": "revenue",
                    "options": [{"label": opt, "value": opt} for opt in options],
                    "source": "metric_scope",
                },
            )
        send_prompt = getattr(adapter, "send_clarification_prompt", None)
        if callable(send_prompt) and options:
            await send_prompt(
                event,
                clarifying_q,
                [{"label": opt, "value": opt} for opt in options],
            )
        else:
            option_lines = "\n".join(f"  • {opt}" for opt in options[:5])
            await adapter.send_message(
                event,
                f"❓ {clarifying_q}\n\n{option_lines}\n\n"
                "_Reply with one of the options above._",
            )
        return

    _matched_metrics = _metric_scope.metrics
    if _matched_metrics:
        try:
            store.increment_metric_usage(account_id, [m.get("name") for m in _matched_metrics if m.get("name")])
        except Exception as _usage_exc:
            log.debug("Metric usage increment skipped: %s", _usage_exc)
    metric_formula_context = _format_metric_formula_context(_matched_metrics, account_id=account_id)
    _metric_formula_tables = set()
    for _metric in _matched_metrics:
        _metric_formula_tables.update(metric_source_tables(_metric, all_columns))
    if metric_formula_context:
        # Prepend metric formulas BEFORE the KB context so the LLM reads them
        # first and they take precedence over any similar-column documentation
        # in the 10,000+ chars of KB content that follows.
        context_with_terms = metric_formula_context + "\n\n" + context_with_terms

    # Fetch KB docs for every table referenced by the selected metric formulas.
    # This is deliberately after metric scoping; otherwise a generic metric from
    # another schema can pollute an All-schema question.
    if _metric_formula_tables:
        try:
            from core.table_coverage import guarantee_table_coverage
            _mf_gap_docs = guarantee_table_coverage(
                account_id    = account_id,
                required_fqns = _metric_formula_tables,
                retrieved_docs = relevant_kbs,
                rag_filter    = None,   # metric tables are admin-approved; bypass per-user ACL
                max_fill      = 4,
            )
            if _mf_gap_docs:
                context_with_terms = (
                    context_with_terms
                    + "\n\n---\n\n"
                    + "\n\n---\n\n".join(_mf_gap_docs)
                )
                log.info(
                    "Metric formula coverage: injected %d doc(s) for scoped metric tables %s",
                    len(_mf_gap_docs), sorted(_metric_formula_tables),
                )
        except Exception as _mf_exc:
            log.debug("Scoped metric formula table coverage skipped: %s", _mf_exc)

    # Row-calculated metric joins → promote from text hints to deterministic SQL.
    # After metric scoping we know which metrics are active. If any are
    # row-calculated and the graph resolver has produced a join skeleton, append
    # the required joins so the LLM treats them as hard constraints (part of the
    # "MUST use this exact structure" block) rather than optional instructions.
    if _graph_ctx.get("enabled") and _matched_metrics:
        try:
            _row_joins = _build_row_metric_join_sql(
                _matched_metrics,
                db_cfg.get("db_type", "azure_sql"),
                _graph_ctx.get("join_skeleton", ""),
            )
            if _row_joins:
                _graph_ctx = {
                    **_graph_ctx,
                    "join_skeleton": _graph_ctx["join_skeleton"] + "\n" + _row_joins,
                }
                log.info(
                    "Row-metric joins appended to graph skeleton for %s", account_id
                )
        except Exception as _rmj_exc:
            log.debug("Row-metric join injection skipped: %s", _rmj_exc)

    # ── Analytical intent detection — inject SQL hints into system prompt ──────
    # Detects window functions, anomaly, contribution, relative date patterns
    # from the question and appends precise SQL construction hints so the LLM
    # emits the correct syntax without needing general training on window funcs.
    _analytic_hints: list[str] = []
    try:
        from core.insight import detect_analytical_intents
        from core.window_analytics import build_window_sql_hint
        from core.anomaly_detection import build_anomaly_sql_hint
        from core.contribution_analysis import build_contribution_sql_hint

        _intents = detect_analytical_intents(question)

        if _intents.get("window"):
            _analytic_hints.append(
                build_window_sql_hint(_intents["window"], db_cfg.get("db_type", "azure_sql"))
            )
            log.info("analytic_intent: window=%s", _intents["window"].type)

        if _intents.get("anomaly"):
            _analytic_hints.append(build_anomaly_sql_hint(db_cfg.get("db_type", "azure_sql")))
            log.info("analytic_intent: anomaly=True")

        if _intents.get("contribution"):
            _analytic_hints.append(build_contribution_sql_hint())
            log.info("analytic_intent: contribution=True")

        if _intents.get("relative_date"):
            ri = _intents["relative_date"]
            _analytic_hints.append(
                f"RELATIVE DATE HINT: The user is asking about a rolling window of "
                f"{ri.n} {ri.unit}(s). Use dynamic date arithmetic rather than hardcoded dates. "
                f"For SQL Server/Azure SQL use DATEADD; for Snowflake use DATEADD or interval syntax; "
                f"for Oracle use INTERVAL or ADD_MONTHS. "
                f"{'Also compute the prior window for comparison.' if ri.compare else ''}"
            )
            log.info("analytic_intent: relative_date=%s", ri.unit)

        # ── Tier 2 SQL hints ──────────────────────────────────────────────────
        if _intents.get("budget_vs_actual"):
            from core.budget_vs_actual import build_bva_sql_hint
            _analytic_hints.append(build_bva_sql_hint())
            log.info("analytic_intent: budget_vs_actual=True")

        if _intents.get("cohort"):
            from core.cohort_analysis import build_cohort_sql_hint
            _analytic_hints.append(build_cohort_sql_hint())
            log.info("analytic_intent: cohort=True")

        if _intents.get("correlation"):
            from core.correlation_analysis import build_correlation_sql_hint
            _analytic_hints.append(build_correlation_sql_hint())
            log.info("analytic_intent: correlation=True")

        if _intents.get("pivot"):
            from core.pivot_table import build_pivot_sql_hint
            _analytic_hints.append(build_pivot_sql_hint())
            log.info("analytic_intent: pivot=True")

        # ── Tier 3 SQL hints ──────────────────────────────────────────────────
        if _intents.get("funnel"):
            from core.funnel_analysis import build_funnel_sql_hint
            _analytic_hints.append(build_funnel_sql_hint())
            log.info("analytic_intent: funnel=True")

        if _intents.get("forecast"):
            from core.forecast import build_forecast_sql_hint, extract_forecast_periods
            _n_fc = extract_forecast_periods(question)
            _analytic_hints.append(build_forecast_sql_hint(_n_fc))
            log.info("analytic_intent: forecast=True periods=%d", _n_fc)

        if _intents.get("fiscal"):
            from core.fiscal_calendar import build_fiscal_sql_hint
            _fiscal_month = db_cfg.get("fiscal_year_start_month", 1)
            _analytic_hints.append(
                build_fiscal_sql_hint(question, _fiscal_month, db_cfg.get("db_type", "azure_sql"))
            )
            log.info("analytic_intent: fiscal=True start_month=%d", _fiscal_month)

        if _intents.get("histogram"):
            from core.distribution_analysis import build_histogram_sql_hint
            _analytic_hints.append(build_histogram_sql_hint())
            log.info("analytic_intent: histogram=True")

        if _intents.get("boxplot"):
            from core.distribution_analysis import build_boxplot_sql_hint
            _analytic_hints.append(build_boxplot_sql_hint())
            log.info("analytic_intent: boxplot=True")

        if _intents.get("whatif"):
            from core.whatif import parse_whatif_params, build_whatif_sql_hint
            _wi_params = parse_whatif_params(question)
            _analytic_hints.append(build_whatif_sql_hint(_wi_params))
            log.info("analytic_intent: whatif=True col_hint=%s", _wi_params.col_hint)
            if hasattr(event, '__dict__'):
                event.__dict__['_whatif_params'] = _wi_params

        # Store intents on event so _send_results can post-process the result
        if hasattr(event, '__dict__'):
            event.__dict__['_analytic_intents'] = _intents
    except Exception as _ai_exc:
        log.debug("Analytical intent detection skipped: %s", _ai_exc)
        _intents = {}

    if _analytic_hints:
        context_with_terms = (
            context_with_terms
            + "\n\n---\n\n"
            + "\n\n".join(_analytic_hints)
        )

    # Final safety cap after every prepend/append is done. Priority blocks
    # (metric formulas, semantic model context) sit at the HEAD of the string,
    # so tail truncation only ever sacrifices the lowest-priority material.
    context_with_terms = _clamp_prompt_context(context_with_terms)

    system = build_sql_system_prompt(
        db_cfg["db_type"], context_with_terms,
        conversation_history=_conv_history or None,
        graph_context=_graph_ctx or None,
        semantic_plan=_semantic_plan or None,
    )
    try:
        await _send_live_stage(adapter, event, "generating_sql", "Generating query", "Translating the business question into SQL.")
        _llm_gen_t0 = time.time()
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
                duration_ms=int((time.time() - _llm_gen_t0) * 1000),
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
    sql = normalize_generated_sql(sql, db_cfg["db_type"])

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
        from core.failure_messages import suggest_closest_terms
        _closest = suggest_closest_terms(question, account_id, state.get("kb_dir", ""))
        _closest_line = (
            f"Closest known terms in your data: {', '.join(_closest)}.\n\n" if _closest else ""
        )
        await adapter.send_message(event,
                "❓ I couldn't find the right tables or columns to answer that.\n\n"
                + _closest_line +
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
    retry_count = 0
    semantic_context = {
        "intent": query_intent,
        "question": question,
        "graph_context": _graph_ctx,
        "semantic_plan": _semantic_plan,
        "metric_formulas": _matched_metrics,
    }
    _validate_t0 = time.time()
    ok, reason, code = validate_sql(
        sql, all_known, db_cfg["db_type"], query_scope_tables, all_columns, semantic_context
    )
    _validate_ms = int((time.time() - _validate_t0) * 1000)

    # Display-field plan mismatches are mechanically fixable from the plan
    # itself (add the dimension join, swap key → display column). Try that
    # before burning an LLM retry — and before surfacing a validator error.
    if not ok and code == "field_plan_mismatch":
        _repair_t0 = time.time()
        try:
            _repaired_sql = attempt_field_plan_repair(
                sql, db_cfg["db_type"], all_known, query_scope_tables,
                all_columns, semantic_context,
            )
        except Exception as _rep_exc:
            _repaired_sql = ""
            log.debug("Field-plan repair skipped: %s", _rep_exc)
        if _repaired_sql:
            _trace_step(
                trace_id,
                "field_plan_repair",
                input_summary=sql,
                output_summary=_repaired_sql,
                metadata={"mode": "deterministic"},
                duration_ms=int((time.time() - _repair_t0) * 1000),
            )
            sql = _repaired_sql
            ok, reason, code = True, "OK", "ok"

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
        duration_ms=_validate_ms,
    )

    if ok:
        _exec_t0 = time.time()
        try:
            await _send_live_stage(adapter, event, "executing_query", "Running query", "Executing the SQL against your connected data source.")
            _loop = asyncio.get_running_loop()
            governed = await asyncio.wait_for(
                _loop.run_in_executor(None, _execute_with_policy, sql, semantic_context),
                timeout=180.0,
            )
            rows = governed.rows
            sql = governed.sql
            _trace_step(trace_id, "execute_sql", input_summary=sql, output_summary={"rows": len(rows)}, duration_ms=int((time.time() - _exec_t0) * 1000))
        except asyncio.TimeoutError:
            exec_error = "Query timed out after 3 minutes. Try narrowing your question with a filter (e.g. date range or specific customer)."
            _trace_step(trace_id, "execute_sql", input_summary=sql, output_summary=exec_error, status="error", duration_ms=int((time.time() - _exec_t0) * 1000))
            log.warning("Query timed out for %s", account_id)
        except PolicyDeniedError as policy_error:
            rows = None
            exec_error = None
            ok = False
            last_reason = policy_error.decision.explanation or "Blocked by regulated data policy."
            last_code = policy_error.decision.reason_code
            _trace_step(
                trace_id,
                "policy_enforcement",
                input_summary=sql,
                output_summary={
                    "reason": last_code,
                    "audit_id": policy_error.decision.audit_id,
                },
                status="error",
                duration_ms=int((time.time() - _exec_t0) * 1000),
            )
        except Exception as first_error:
            exec_error = str(first_error)
            _trace_step(trace_id, "execute_sql", input_summary=sql, output_summary=exec_error, status="error", duration_ms=int((time.time() - _exec_t0) * 1000))
            log.warning("First execution failed for %s: %s",
                        account_id, exec_error[:100])
    else:
        last_reason, last_code = reason, code

    retryable = (not ok and code in ("unknown_table", "unknown_column", "date_key_format", "anti_join_shape", "field_plan_mismatch", "metric_formula_mismatch", "null_aggregate_diagnostic", "parse")) or (exec_error is not None)

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
            validation_repair_note = ""
            if last_code == "unknown_column":
                # When a schema is locked, "switch to the table where the column exists" is
                # wrong if that table lives in a different schema — detect this and override
                # the repair instruction so the LLM stays in the selected schema.
                _cross_schema = (
                    schema_hint
                    and "Exact column exists on" in last_reason
                    and schema_hint.upper() not in last_reason.upper().split(
                        "Exact column exists on"
                    )[1][:120]   # check the first 120 chars of the "exists on" suffix
                )
                if _cross_schema:
                    validation_repair_note = (
                        f"\nSCHEMA-LOCKED COLUMN REPAIR RULE:\n"
                        f"- The column you used does NOT exist in the {schema_hint} schema.\n"
                        f"- The validator found it in a DIFFERENT schema — you are FORBIDDEN "
                        f"from using that table or column.\n"
                        f"- Search the {schema_hint} Knowledge Base documents in this prompt "
                        f"for the correct column that represents the same concept.\n"
                        f"- If no matching column exists in {schema_hint}, return CANNOT_GENERATE "
                        f"and state exactly which column is missing from the {schema_hint} schema.\n"
                        f"- Do NOT copy column names from other schemas under any circumstance.\n"
                    )
                else:
                    validation_repair_note = (
                        "\nUNKNOWN COLUMN REPAIR RULE:\n"
                        "- The SQL used a column on a table where that column does not exist.\n"
                        "- If the validator lists 'Exact column exists on', switch the source table "
                        "or add the required JOIN to that table.\n"
                        "- Do not keep the same table alias and do not retry the same invalid column/table pair.\n"
                        "- If no exact column exists anywhere in the KB context, use the closest exact "
                        "business synonym/column from the KB or return CANNOT_GENERATE.\n"
                    )
            elif last_code == "anti_join_shape":
                validation_repair_note = (
                    "\nANTI-JOIN REPAIR RULE:\n"
                    "- The question asks for missing records/data, so use LEFT JOIN ... WHERE right_key IS NULL.\n"
                    "- The FROM table must be the source/parent table containing the records to list.\n"
                    "- Do not answer with a single-table WHERE measure IS NULL query.\n"
                )
            elif last_code == "date_key_format":
                validation_repair_note = (
                    "\nDATE-KEY REPAIR RULE:\n"
                    "- Convert integer YYYYMMDD date keys before FORMAT/YEAR/MONTH/DATEPART.\n"
                    "- For Azure SQL use TRY_CONVERT(date, CONVERT(varchar(8), alias.DATE_KEY_COL), 112).\n"
                )
            elif last_code == "field_plan_mismatch":
                validation_repair_note = build_field_plan_repair_note(_semantic_plan or {})
            elif last_code == "metric_formula_mismatch":
                # Inject the EXACT approved formula(s) verbatim — do not rely on
                # the LLM finding them in the KB context, which can be overridden.
                _metric_formulas_inline = ""
                for _mf in _matched_metrics:
                    if ((_mf.get("formula_type") or "query").lower() == "expression"
                            and _mf.get("sql_template")):
                        _metric_formulas_inline += (
                            f"\n  Metric name : {_mf.get('name', 'metric')}\n"
                            f"  EXACT formula: {_mf['sql_template'].strip()}\n"
                            f"  Required columns (must appear in SELECT): "
                            f"{_mf.get('required_columns', 'see formula above')}\n"
                        )
                validation_repair_note = (
                    "\nAPPROVED METRIC FORMULA REPAIR RULE:\n"
                    "- Your SQL used a column that is NOT the approved metric formula.\n"
                    "- You MUST replace your aggregate expression with the EXACT formula below:\n"
                    f"{_metric_formulas_inline if _metric_formulas_inline else '  See the APPROVED METRIC FORMULAS section in the prompt.'}\n"
                    "- The formula MUST appear in the SELECT clause — not just in a WHERE filter.\n"
                    "- Do NOT use CUS_IVC_LIN_AMT or any other similar column as a substitute "
                    "for the approved metric, even if the Knowledge Base suggests it.\n"
                    "- Keep all other query structure (GROUP BY, JOINs, date filters) unchanged.\n"
                    "- Only replace the aggregate expression itself.\n"
                )
            elif last_code == "null_aggregate_diagnostic":
                validation_repair_note = (
                    "\nNULL-AWARE AGGREGATE REPAIR RULE:\n"
                    "- This is a filtered single-row aggregate such as revenue for one customer/key.\n"
                    "- Include COUNT_BIG(*) AS [MatchedRows] for Azure SQL, or COUNT(*) AS MatchedRows for other DBs.\n"
                    "- Include COUNT(metric_column) AS [NonNullMetricRows] for every SUM metric.\n"
                    "- Wrap every SUM metric with COALESCE(SUM(metric_column), 0) so missing values render as 0.\n"
                    "- Keep the user's filter value unchanged.\n"
                )
            retry_user = (
                f"The following SQL failed validation with: {last_reason}\n"
                f"SQL: {sql}\n\n"
                f"The original question was: {question}\n\n"
                f"{validation_repair_note}\n"
                f"Rewrite the SQL using only tables and columns that appear in "
                f"the provided knowledge base context. Return only the corrected "
                f"SQL, no explanation."
            )

        # Keep semantic-plan guardrails during repair. Otherwise the retry can
        # pass validation while still using a raw dimension key as a business
        # label instead of the planned display field.
        _retry_plan = _semantic_plan or None
        _retry_semantic_context = dict(semantic_context)

        try:
            await _send_live_stage(adapter, event, "repairing_query", "Repairing query", "Fixing a validation or execution issue before retrying.")
            _retry_llm_t0 = time.time()
            with llm_audit_scope(
                account_id=account_id,
                question=question,
                enabled=audit_enabled,
                request_id=audit_request_id,
            question_id=audit_request_id,
                component="sql_repair",
            ):
                sql_retry, _, _ = await llm_complete(
                    build_sql_system_prompt(
                        db_cfg["db_type"],
                        context_with_terms,
                        graph_context=_graph_ctx or None,
                        semantic_plan=_retry_plan,
                    ),
                    retry_user, provider, model, api_key,
                    max_tokens=512, **az_kwargs,
                )
            # Retry timings accumulate onto the same buckets as the first
            # attempt (bucket aggregation sums by step_name), not separate rows.
            _trace_step(trace_id, "llm_generate_sql", output_summary={"retry": True}, duration_ms=int((time.time() - _retry_llm_t0) * 1000))
            if sql_retry.startswith("```"):
                sql_retry = "\n".join(sql_retry.split("\n")[1:]).rsplit("```", 1)[0].strip()

            sql_retry = _inject_distinct_if_needed(sql_retry, question)
            sql_retry = normalize_generated_sql(sql_retry, db_cfg["db_type"])

            if "CANNOT_GENERATE" not in sql_retry.upper() and len(sql_retry) > 10:
                _retry_validate_t0 = time.time()
                ok2, reason2, code2 = validate_sql(
                    sql_retry, all_known, db_cfg["db_type"], query_scope_tables, all_columns, _retry_semantic_context)
                _trace_step(
                    trace_id, "validate_sql", input_summary=sql_retry, output_summary=reason2,
                    status="success" if ok2 else "error", metadata={"code": code2, "retry": True},
                    duration_ms=int((time.time() - _retry_validate_t0) * 1000),
                )
                if ok2:
                    _retry_exec_t0 = time.time()
                    try:
                        await _send_live_stage(adapter, event, "executing_query", "Retrying query", "Running the corrected query against your data.")
                        _loop = asyncio.get_running_loop()
                        governed = await asyncio.wait_for(
                            _loop.run_in_executor(None, _execute_with_policy, sql_retry, _retry_semantic_context),
                            timeout=180.0,
                        )
                        rows = governed.rows
                        sql         = governed.sql
                        exec_error  = None
                        ok, last_reason, last_code = True, "OK", "ok"
                        retry_count += 1
                        log.info("Retry succeeded for %s", account_id)
                        _trace_step(trace_id, "execute_sql", input_summary=sql, output_summary={"rows": len(rows), "retry": True}, duration_ms=int((time.time() - _retry_exec_t0) * 1000))
                    except asyncio.TimeoutError:
                        exec_error = "Retry query timed out after 3 minutes."
                        log.warning("Retry query timed out for %s", account_id)
                        _trace_step(trace_id, "execute_sql", input_summary=sql_retry, output_summary=exec_error, status="error", duration_ms=int((time.time() - _retry_exec_t0) * 1000))
                    except PolicyDeniedError as policy_error:
                        exec_error = None
                        ok = False
                        last_reason = policy_error.decision.explanation or "Blocked by regulated data policy."
                        last_code = policy_error.decision.reason_code
                        log.warning(
                            "Retry denied by policy for %s: %s",
                            account_id,
                            last_code,
                        )
                        _trace_step(trace_id, "policy_enforcement", input_summary=sql_retry, output_summary={"reason": last_code, "retry": True}, status="error", duration_ms=int((time.time() - _retry_exec_t0) * 1000))
                    except Exception as retry_exec_err:
                        exec_error = str(retry_exec_err)
                        log.warning("Retry execution failed for %s: %s",
                                    account_id, exec_error[:100])
                        _trace_step(trace_id, "execute_sql", input_summary=sql_retry, output_summary=exec_error, status="error", duration_ms=int((time.time() - _retry_exec_t0) * 1000))
                else:
                    last_reason, last_code = reason2, code2
                    log.warning("Retry still invalid for %s: %s",
                                account_id, reason2[:120])
        except Exception as retry_err:
            log.warning("Retry LLM call failed for %s: %s",
                        account_id, str(retry_err)[:100])

    # ── Terminal failure handling ────────────────────────────────────────────
    # Raw reasons/errors stay in query_log + answer_trace (audit unchanged);
    # only the chat message is translated to business language with a next step.
    if not ok:
        _log_q(account_id, question, sql, 0, False, last_reason, provider, model,
               tok_in, tok_out, int(time.time()*1000) - start_ms,
               portal_user_id=pu_id, zoom_user_id=zid,
               question_id=audit_request_id)
        _trace_finish(trace_id, status="error", answer_type="error", error_message=last_reason)
        from core.failure_messages import translate_failure, suggest_closest_terms, _VALIDATION_REASONS
        if (last_code or "").lower() in _VALIDATION_REASONS:
            _suggest = (
                suggest_closest_terms(question, account_id, state.get("kb_dir", ""))
                if (last_code or "").lower() in {"unknown_column", "cannot_generate", "field_plan_mismatch"}
                else []
            )
            _rca = translate_failure(
                kind="validation", code=last_code, reason=last_reason,
                sql=sql, question=question,
                suggestions=_suggest,
            )
            from core.answer_formatter import format_failure_business_response
            await adapter.send_message(event, format_failure_business_response(
                rca=_rca, sql=sql, sql_preview_fn=_sql_preview,
            ))
        else:
            # Policy denials and other non-validator codes already carry a
            # business-written explanation — pass through untouched.
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
        from core.failure_messages import translate_failure
        from core.answer_formatter import format_failure_business_response
        _rca = translate_failure(
            kind="execution", exception_text=exec_error or "Unknown error",
            sql=sql, question=question,
        )
        await adapter.send_message(event, format_failure_business_response(
            rca=_rca, sql=sql, sql_preview_fn=_sql_preview,
        ))
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
            # No ambiguity signal: return business-readable RCA for the empty result.
            _zr_tables = extract_sql_tables(sql, db_cfg.get("db_type", "azure_sql"))
            _zr_counts = await asyncio.to_thread(_count_tables_for_zero_row, db_cfg, _zr_tables)
            _zr_empty_tables = [table for table, count in _zr_counts.items() if count == 0]
            _trace_finish(trace_id, status="success", answer_type="empty", row_count=0, duration_ms=duration_ms, final_answer_summary="Query returned no rows")
            await adapter.send_message(event, _build_zero_row_message(
                question,
                sql,
                _graph_ctx,
                last_code or code or "ok",
                retry_count,
                tables_used=_zr_tables,
                empty_tables=_zr_empty_tables,
                semantic_plan=_semantic_plan,
                account_id=account_id,
            ))
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
    from core.metric_semantics import detect_derived_metric_gap
    _confidence_context = {
        "validation_code": last_code or code or "ok",
        "retry_count": retry_count,
        "has_semantic_plan": bool((_semantic_plan or {}).get("enabled")),
        "has_graph_context": bool((_graph_ctx or {}).get("enabled") or (_graph_ctx or {}).get("detected")),
        "tables_used": extract_sql_tables(sql, db_cfg.get("db_type", "azure_sql")),
        # "open order quantity" with no approved formula anywhere = the LLM
        # could only SUM a raw column; surface that as a confidence warning
        # instead of presenting a business-wrong number with full confidence.
        "derived_metric_gap": detect_derived_metric_gap(
            question,
            has_metric_formula=bool(metric_formula_context),
            has_term_expression=bool(term_injection),
        ),
        # Everything above the relevance floor was dropped — the KB context
        # the SQL was built on matched this question only weakly.
        "weak_retrieval": _weak_retrieval,
        # Carried through to cache_result so compare_prior can read them.
        "semantic_plan": _semantic_plan or {},
    }

    # ── Post-processing: apply contribution / anomaly analytics ──────────────
    # Run after DB execution, before _send_results.  Augments rows in-place
    # with computed columns so the frontend can render them directly.
    _post_intents = getattr(event, '_analytic_intents', None) or _intents if '_intents' in dir() else {}
    if rows and _post_intents:
        try:
            if _post_intents.get("contribution") and not any("contribution_pct" in r for r in rows[:1]):
                from core.contribution_analysis import compute_contribution, infer_numeric_col as _inc
                _val_col = _inc(rows)
                if _val_col:
                    rows = compute_contribution(rows, _val_col)
                    log.info("post_process: contribution_pct added for col=%s", _val_col)

            if _post_intents.get("anomaly") and not any("anomaly_flag" in r for r in rows[:1]):
                from core.anomaly_detection import detect_anomalies, infer_value_col as _ivc
                _val_col = _ivc(rows)
                if _val_col:
                    _anom_result = detect_anomalies(rows, _val_col)
                    rows = _anom_result.rows
                    log.info(
                        "post_process: anomaly detection complete col=%s flagged=%d/%d",
                        _val_col, _anom_result.flagged_rows, _anom_result.total_rows,
                    )

            # ── Tier 2 post-processing ────────────────────────────────────────
            if _post_intents.get("budget_vs_actual") and not any("variance" in r for r in rows[:1]):
                from core.budget_vs_actual import infer_bva_cols, compute_bva
                _a_col, _b_col = infer_bva_cols(rows)
                if _a_col and _b_col:
                    rows = compute_bva(rows, _a_col, _b_col)
                    log.info("post_process: bva variance added actual=%s budget=%s", _a_col, _b_col)

            if _post_intents.get("cohort") and not any("cohort" in str(list(r.keys())) for r in rows[:1]):
                from core.cohort_analysis import infer_cohort_cols, compute_cohort_matrix
                _cohort_col, _period_col, _value_col = infer_cohort_cols(rows)
                if _cohort_col and _period_col and _value_col:
                    rows = compute_cohort_matrix(rows, _cohort_col, _period_col, _value_col)
                    log.info("post_process: cohort matrix built cohort=%s period=%s", _cohort_col, _period_col)

            if _post_intents.get("correlation") and not any("__corr_r" in r for r in rows[:1]):
                from core.correlation_analysis import infer_corr_cols, compute_correlation, annotate_rows_with_correlation
                _x_col, _y_col = infer_corr_cols(rows, question)
                if _x_col and _y_col:
                    _corr = compute_correlation(rows, _x_col, _y_col)
                    rows = annotate_rows_with_correlation(rows, _corr)
                    log.info("post_process: correlation r=%.4f (%s) n=%d", _corr.pearson_r or 0, _corr.interpretation, _corr.n)

            if _post_intents.get("pivot") and not any("TOTAL" in r for r in rows[:1]):
                from core.pivot_table import infer_pivot_cols, compute_pivot_table
                _rk, _ck, _vk = infer_pivot_cols(rows)
                if _rk and _ck and _vk:
                    rows = compute_pivot_table(rows, _rk, _ck, _vk)
                    log.info("post_process: pivot table built row=%s col=%s value=%s", _rk, _ck, _vk)

            # ── Tier 3 post-processing ────────────────────────────────────────
            if _post_intents.get("funnel") and not any("funnel_pct" in r for r in rows[:1]):
                from core.funnel_analysis import infer_funnel_cols, compute_funnel
                _s_col, _c_col = infer_funnel_cols(rows)
                if _s_col and _c_col:
                    rows = compute_funnel(rows, _s_col, _c_col)
                    log.info("post_process: funnel computed stage=%s count=%s", _s_col, _c_col)

            if _post_intents.get("forecast") and not any("is_forecast" in r for r in rows[:1]):
                from core.forecast import infer_forecast_cols, compute_forecast, extract_forecast_periods
                _p_col, _v_col = infer_forecast_cols(rows)
                if _p_col and _v_col:
                    _n_fc = extract_forecast_periods(question)
                    rows = compute_forecast(rows, _p_col, _v_col, _n_fc)
                    log.info("post_process: forecast appended periods=%d period=%s value=%s", _n_fc, _p_col, _v_col)

            if _post_intents.get("histogram") and not any("bin_label" in r for r in rows[:1]):
                from core.distribution_analysis import infer_histogram_col, compute_histogram
                _h_col = infer_histogram_col(rows)
                if _h_col:
                    rows = compute_histogram(rows, _h_col)
                    log.info("post_process: histogram binned col=%s bins=%d", _h_col, len(rows))

            if _post_intents.get("boxplot") and not any("bp_data" in r for r in rows[:1]):
                from core.distribution_analysis import infer_boxplot_cols, compute_boxplot
                _g_col, _v_col = infer_boxplot_cols(rows)
                if _v_col:
                    rows = compute_boxplot(rows, _v_col, _g_col)
                    log.info("post_process: boxplot computed group=%s value=%s", _g_col, _v_col)

            if _post_intents.get("whatif") and not any(k.startswith("scenario_") for k in (rows[0] if rows else {})):
                from core.whatif import compute_whatif
                _wi_p = getattr(event, '_whatif_params', None)
                if _wi_p:
                    rows = compute_whatif(rows, _wi_p)
                    log.info("post_process: what-if scenario applied delta_pct=%s", _wi_p.delta_pct)

        except Exception as _pp_exc:
            log.debug("Post-processing analytics skipped: %s", _pp_exc)

    await _send_results(event, adapter, question, rows, sql, duration_ms,
                        portal_user, account_id, db_cfg,
                        rag_context=context, question_id=audit_request_id,
                        confidence_context=_confidence_context,
                        display_context={
                            "format_scope": "metric_context",
                            "metrics": _matched_metrics,
                        })
    if _why_mode and rows:
        await _send_why_insight(
            adapter, event,
            question=question, rows=rows, sql=sql,
            client=client, account_id=account_id, db_cfg=db_cfg,
            rag_context=context,
            known_tables=all_known,
            query_executor=lambda _cfg, _s: _execute_with_policy(_s),
            question_id=audit_request_id,
        )
    _trace_finish(trace_id, status="success", answer_type="table", row_count=len(rows), duration_ms=duration_ms, final_answer_summary="Answered from database query")

    # ── Learning loop: persist quality candidate ──────────────────────────────
    # Runs AFTER the response is already on the wire — zero user-facing latency.
    # Gated by enable_feedback_collection so it's a no-op until the pilot is on.
    if client.get("enable_feedback_collection"):
        _create_learning_candidate(
            account_id        = account_id,
            question_id       = audit_request_id,
            question          = question,
            sql               = sql,
            validation_passed = ok,
            had_repair        = retry_count > 0,
            repair_succeeded  = retry_count > 0,   # if we're here after retries, repair worked
            row_count         = len(rows),
            confidence_ctx    = _confidence_context,
            schema_scope      = schema_hint,
            kb_dir            = state.get("kb_dir", ""),
            schema_dir        = state.get("schema_dir", ""),
        )


