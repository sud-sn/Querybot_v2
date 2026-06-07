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
    build_schema_grounded_clarification_hint,
)
from core.schema import run_query, load_known_tables, load_schema_columns
from core.knowledge import load_retriever
from core.validator import validate_sql
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
    _extract_kb_synonym_injection, _send_live_stage,
    _count_tables_for_zero_row, _build_zero_row_message,
    _format_metric_formula_context, _extract_metric_formula_tables,
)
from core.pipeline_trace import (
    _log_q, _trace_create, _trace_update, _trace_step, _trace_finish,
    _create_learning_candidate,
)
from core.result_renderer import (
    _send_results, _inject_distinct_if_needed,
)

log = logging.getLogger("querybot")

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
                                question_id=audit_request_id,
                                display_context={
                                    "format_scope": "metric_registry",
                                    "metrics": [matched_metric],
                                })
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
            question, account_id, n=3,
            allowed_tables=rag_filter,
            schema_scope=schema_hint,
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
    query_intent = analyze_query_intent(question)
    # Candidate metrics are account-wide. We delay injecting/enforcing them
    # until graph + semantic planning has inferred the question's schema/domain.
    _metric_candidates = store.list_metric_formula_context(account_id, question, limit=10)
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
            generic_hints,
            context,
        )
        if part
    ]
    context_with_terms = "\n\n".join(context_parts)
    try:
        _semantic_model_context = build_runtime_semantic_context(
            state.get("kb_dir", ""),
            question=question,
            selected_schema=schema_hint,
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

    _semantic_plan = {}
    try:
        _semantic_plan = build_semantic_field_plan(
            question,
            all_columns,
            query_scope_tables,
            selected_schema=schema_hint,
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
            question=question,
            selected_schema=schema_hint,
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
        log.debug("Structured semantic model planning skipped: %s", _smp_exc)

    _semantic_plan = _merge_semantic_plans(_semantic_plan, _semantic_model_plan)

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
    metric_formula_context = _format_metric_formula_context(_matched_metrics)
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

    system = build_sql_system_prompt(
        db_cfg["db_type"], context_with_terms,
        conversation_history=_conv_history or None,
        graph_context=_graph_ctx or None,
        semantic_plan=_semantic_plan or None,
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
    retry_count = 0
    semantic_context = {
        "intent": query_intent,
        "question": question,
        "graph_context": _graph_ctx,
        "semantic_plan": _semantic_plan,
        "metric_formulas": _matched_metrics,
    }
    ok, reason, code = validate_sql(
        sql, all_known, db_cfg["db_type"], query_scope_tables, all_columns, semantic_context
    )
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
            if sql_retry.startswith("```"):
                sql_retry = "\n".join(sql_retry.split("\n")[1:]).rsplit("```", 1)[0].strip()

            sql_retry = _inject_distinct_if_needed(sql_retry, question)

            if "CANNOT_GENERATE" not in sql_retry.upper() and len(sql_retry) > 10:
                ok2, reason2, code2 = validate_sql(
                    sql_retry, all_known, db_cfg["db_type"], query_scope_tables, all_columns, _retry_semantic_context)
                if ok2:
                    try:
                        await _send_live_stage(adapter, event, "executing_query", "Retrying query", "Running the corrected query against your data.")
                        rows = run_query(
                            db_cfg["credentials"], db_cfg["db_type"], sql_retry)
                        sql         = sql_retry
                        exec_error  = None
                        ok, last_reason, last_code = True, "OK", "ok"
                        retry_count += 1
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
            f"SQL tried:\n```sql\n{sql_preview}\n```"
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
    _confidence_context = {
        "validation_code": last_code or code or "ok",
        "retry_count": retry_count,
        "has_semantic_plan": bool((_semantic_plan or {}).get("enabled")),
        "has_graph_context": bool((_graph_ctx or {}).get("enabled") or (_graph_ctx or {}).get("detected")),
        "tables_used": extract_sql_tables(sql, db_cfg.get("db_type", "azure_sql")),
        # Carried through to cache_result so compare_prior can read them.
        "semantic_plan": _semantic_plan or {},
    }
    await _send_results(event, adapter, question, rows, sql, duration_ms,
                        portal_user, account_id, db_cfg,
                        rag_context=context, question_id=audit_request_id,
                        confidence_context=_confidence_context,
                        display_context={
                            "format_scope": "metric_context",
                            "metrics": _matched_metrics,
                        })
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
        )


