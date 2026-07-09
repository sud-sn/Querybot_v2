"""
gateway/webhooks.py
───────────────────
Platform webhook endpoints and the WebSocket chat handler, extracted from main.py.

Routes registered on an APIRouter that main.py mounts at startup.
"""

from __future__ import annotations

import json
import logging
import os
import time

import store
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

from gateway import get_adapter, PlatformEvent
from core.webhook_dedup import is_duplicate_event, remember_event
from core.dispatcher import dispatch
from core.query_pipeline import handle_query, _generate_duckdb_sql
from core.pipeline_context import get_state, get_client_db
from core.pipeline_trace import (
    _log_q, _trace_create, _trace_update, _trace_step, _trace_finish,
)
from core.pipeline_helpers import _extract_kb_synonym_injection
from core.result_renderer import (
    _sanitize_rows, _generate_result_narration, _result_has_identifiers,
    _build_followup_sql_context, _build_cannot_generate_hint,
    _inject_distinct_if_needed,
)
from core.result_cache import result_cache
from core.schema import run_query, load_known_tables, load_schema_columns
from core.knowledge import load_retriever
from core.validator import validate_sql
from core.llm import llm_complete, build_sql_system_prompt, resolve_provider
from core.llm_audit import llm_audit_scope, make_llm_audit_request_id
from core.query_router import build_duckdb_system_prompt
from core.duckdb_sql_validator import validate_duckdb_result_sql
from core.query_semantics import analyze_query_intent, build_generic_query_hints
from core.graph_resolver import resolve_for_question as _graph_resolve
from core.chart import detect_chart_type, build_chart_payload
from core.examples import retrieve_similar_examples, format_examples_for_prompt
from core.clarification import (
    get_pending, clear_pending, combine_with_clarification, resolve_option_text,
)

log = logging.getLogger("querybot")

router = APIRouter()


# ══════════════════════════════════════════════════════════════════════════════
# REST API adapter — used by /api/ask (Copilot Studio, Power Automate, testing)
# ══════════════════════════════════════════════════════════════════════════════

class _CaptureAdapter:
    """
    In-memory adapter that buffers pipeline output instead of posting to a
    chat platform.  Used by POST /api/ask so callers get a synchronous text
    response without needing Azure Bot / Slack / Zoom credentials.

    Charts and live-status events are silently dropped — they make no sense
    over a plain HTTP response.  Clarification prompts are captured as
    structured data AND as plain text so the 'answer' field is always set.
    """

    platform_type = "api"

    def __init__(self):
        self.messages: list[str] = []
        self.clarification: dict | None = None

    # Required abstract methods
    async def verify_request(self, body: bytes, headers: dict) -> bool:
        return True

    def parse_event(self, body: bytes, headers: dict):
        return None

    async def send_message(self, event, text: str) -> None:
        if text and text.strip():
            self.messages.append(text.strip())

    async def upload_file(self, event, file_bytes: bytes, filename: str, mime_type: str = "image/png") -> None:
        pass  # charts not returned over REST

    # Optional methods called by the pipeline
    async def send_status(self, event, stage: str, label: str, detail: str = "") -> None:
        pass  # progress indicators not needed for sync API

    async def send_clarification_prompt(self, event, question: str, options: list, pending_id=None) -> None:
        self.clarification = {
            "question": question,
            "options": [
                {"id": o.get("id") or o.get("_term_id") or "", "label": o.get("label") or o.get("value") or ""}
                for o in (options or [])
            ],
        }
        # Also populate messages so 'answer' is never empty
        opts_text = "\n".join(f"- {o.get('label') or o.get('value', '')}" for o in (options or []))
        self.messages.append(f"{question}\n\n{opts_text}")

    async def send_chart(self, event, chart: dict) -> None:
        pass  # charts not returned over REST


# ══════════════════════════════════════════════════════════════════════════════
# POST /api/ask — synchronous REST endpoint
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/api/ask")
async def api_ask(request: Request):
    """
    Synchronous question-answering endpoint for Copilot Studio, Power Automate,
    and any HTTP caller that wants a plain JSON response.

    Request body (JSON):
        {
            "question":   "What is my total revenue this month?",
            "account_id": "Emco_Poc",
            "api_key":    "your-secret-key"   ← required if QUERYBOT_API_KEY is set
        }

    Response:
        {
            "answer":        "Your total revenue is $1.2 M ...",
            "clarification": null | { "question": "...", "options": [...] }
        }

    Security: set QUERYBOT_API_KEY environment variable to restrict access.
    If the env var is not set the endpoint is open (suitable for local dev/demo only).
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, detail="Request body must be valid JSON.")

    question   = (body.get("question") or "").strip()
    account_id = (body.get("account_id") or "").strip()
    api_key    = (body.get("api_key") or "").strip()

    if not question:
        raise HTTPException(400, detail="'question' is required.")
    if not account_id:
        raise HTTPException(400, detail="'account_id' is required.")

    # API key guard — only enforced when QUERYBOT_API_KEY is set
    expected_key = os.getenv("QUERYBOT_API_KEY", "")
    if expected_key and api_key != expected_key:
        raise HTTPException(401, detail="Invalid or missing api_key.")

    # Confirm the client exists before spinning up the pipeline
    client = store.get_client(account_id)
    if not client:
        raise HTTPException(404, detail=f"No client found for account_id '{account_id}'.")

    event = PlatformEvent(
        account_id = account_id,
        user_id    = "api",
        channel_id = "api",
        text       = question,
        platform   = "api",
    )
    adapter = _CaptureAdapter()

    # Run the full pipeline synchronously — portal_user=None gives admin-level
    # table access (no per-user restrictions), which is correct for system callers.
    await handle_query(account_id, event, adapter, question, portal_user=None)

    answer = "\n\n".join(adapter.messages) if adapter.messages else "No answer generated."
    return {
        "answer":        answer,
        "clarification": adapter.clarification,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Webhooks
# ══════════════════════════════════════════════════════════════════════════════

def _load_adapter(platform_type):
    platforms = store.list_platforms(platform_type)
    active    = [p for p in platforms if p.get("is_active")]
    if not active:
        raise HTTPException(503, detail=f"No active {platform_type} platform configured.")
    return get_adapter(platform_type, active[0]["credentials"])


@router.post("/webhook/zoom")
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


@router.post("/webhook/teams")
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
        return Response(status_code=200)
    remember_event(event)

    # The Teams tenant ID rarely matches a QueryBot account_id directly.
    # If no client is found by tenant ID, look for the single configured client
    # (one with a db_config assigned) and route there automatically.
    # This works for the common single-tenant deployment with no hardcoding.
    if not store.get_client(event.account_id):
        all_clients = store.list_clients()
        configured  = [c for c in all_clients if c.get("db_config_id")]
        if len(configured) == 1:
            tenant_id = event.account_id
            event.account_id = configured[0]["account_id"]
            log.info("Teams: auto-mapped tenant %s → client %s", tenant_id, event.account_id)
        else:
            log.warning(
                "Teams: tenant %s not registered; %d configured clients found — "
                "cannot auto-map (need exactly 1)",
                event.account_id, len(configured),
            )

    send_status = getattr(adapter, "send_status", None)
    if callable(send_status):
        try:
            await send_status(event, "accepted", "Working on it")
        except Exception as exc:
            log.debug("Teams initial typing indicator failed: %s", exc)

    await dispatch(event.account_id, event, adapter, bg)
    return Response(status_code=200)


@router.post("/webhook/slack")
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


@router.websocket("/ws/chat/{account_id}")
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
    _ws_table_columns = load_schema_columns(_ws_state.get("schema_dir", ""))

    def _ws_execute_governed(db_cfg: dict, sql: str, semantic_context: dict | None = None):
        from core.compliance.governed_query import execute_governed_query
        from core.compliance.policy_engine import resolve_context

        policy_context = resolve_context(
            account_id,
            portal_user,
            action="query_execution",
            channel="portal",
        )
        return execute_governed_query(
            db_cfg["credentials"],
            db_cfg["db_type"],
            sql,
            context=policy_context,
            known_tables=_ws_known_tables,
            table_columns=_ws_table_columns,
            allowed_tables=store.get_allowed_tables(portal_user),
            semantic_context=semantic_context,
        )

    # Per-result-card conversation history for result_chat multi-turn memory.
    # Keyed by result_id; each value is a list of {question, sql, row_count}.
    # Cleared automatically when a new main query replaces the result card.
    _result_chat_histories: dict[str, list[dict]] = {}

    # Send welcome message with user name
    await websocket.send_json({
        "type":    "system",
        "content": f"Connected as {portal_user.get('name', portal_user.get('id', 'user'))}. Ask me anything about your data.",
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
                    _rc_formats     = result_cache.get_column_formats(_sid)
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
                        _fb_rows     = None
                        _fb_sql      = None
                        _fb_err      = None
                        _fb_full_ctx = ""   # populated below; guard for missing db_cfg
                        _fb_graph_ctx: dict = {}
                        _fb_has_ids  = _result_has_identifiers(_rc_schema)
                        try:
                            _cached_result = getattr(adapter, "last_result", None) or {}
                            _fb_db_cfg     = _cached_result.get("db_cfg") or _rc_db_cfg

                            if _fb_db_cfg:
                                # ── Full main-pipeline context assembly ────────────────
                                # The fallback runs the same context-building steps as
                                # handle_query() so the LLM has complete schema knowledge
                                # — entity graph, business terms, synonyms, examples, table
                                # coverage — not just the original question's RAG context.

                                import re as _fb_re

                                # 1. RAG retrieval — same n as main pipeline; detect grouping
                                _fb_rag_ctx  = _cached_result.get("rag_context", "")
                                _fb_grouping = bool(_fb_re.search(
                                    r"\b(by|per|grouped by|breakdown|split by|each|for each)\s+\w",
                                    rc_question.lower()
                                ))
                                _fb_n = 10 if _fb_grouping else 8
                                try:
                                    _fb_retriever  = load_retriever(account_id)
                                    _fb_fresh_docs = _fb_retriever.retrieve(rc_question, n=_fb_n)
                                    _fb_pinned     = [d for d in _fb_fresh_docs if _fb_retriever._is_global(d)]
                                    _fb_table_docs = [d for d in _fb_fresh_docs if not _fb_retriever._is_global(d)]
                                    if _fb_grouping:
                                        _fb_fact_pats = _fb_retriever.retrieve_fact_patterns(rc_question, n=2)
                                        for _fp in _fb_fact_pats:
                                            if _fp not in (_fb_pinned + _fb_table_docs):
                                                _fb_table_docs.insert(0, _fp)
                                    _fb_fresh_docs = (_fb_pinned + _fb_table_docs)[:7]
                                    _fb_fresh_ctx  = "\n\n---\n\n".join(_fb_fresh_docs)
                                    # Merge: follow-up context first (highest relevance), then
                                    # the original query's context for background schema.
                                    _fb_rag_ctx = (
                                        _fb_fresh_ctx + "\n\n---\n\n" + _fb_rag_ctx
                                        if _fb_rag_ctx else _fb_fresh_ctx
                                    )
                                except Exception as _ret_exc:
                                    log.debug("result_chat fallback KB retrieval failed: %s", _ret_exc)

                                # 2. Few-shot validated examples
                                try:
                                    _fb_examples = retrieve_similar_examples(rc_question, account_id, n=3)
                                    if _fb_examples:
                                        _fb_rag_ctx = (
                                            format_examples_for_prompt(_fb_examples)
                                            + "\n\n---\n\n" + _fb_rag_ctx
                                        )
                                except Exception:
                                    pass

                                # 3. Business term injection (glossary)
                                _fb_term_inj = store.build_term_injection(account_id, rc_question, None)

                                # 4. KB synonym map (from ## Business Synonyms sections)
                                _fb_synonym_inj = _extract_kb_synonym_injection(_fb_rag_ctx)

                                # 5. Generic query hints (date anchoring, aggregation rules)
                                _fb_generic_hints = build_generic_query_hints(rc_question)

                                # 6. Entity graph — deterministic JOIN path resolution
                                # SCOPE to the schemas used in the original SQL so the
                                # resolver never pulls in unrelated tables (e.g. PROFITABILITY
                                # tables when the original query was against PHARMACY).
                                _fb_graph_ctx: dict = {}
                                try:
                                    _fb_full_graph = store.get_full_graph(account_id)
                                    if _fb_full_graph.get("entities"):
                                        # Extract schema names present in the original SQL
                                        # e.g. [PHARMACY].[TABLE] → {"PHARMACY"}
                                        import re as _fb_schema_re
                                        _orig_schemas = {
                                            m.upper()
                                            for m in _fb_schema_re.findall(
                                                r'\[([A-Za-z_][A-Za-z0-9_]*)\]\.\[',
                                                _cached_result.get("sql", ""),
                                            )
                                        }
                                        if _orig_schemas:
                                            # Filter graph to only entities from original schemas
                                            _fb_ents = [
                                                e for e in _fb_full_graph["entities"]
                                                if not e.get("schema_name")
                                                or e.get("schema_name", "").upper() in _orig_schemas
                                            ]
                                            if _fb_ents:
                                                _fb_ent_names = {e["entity_name"] for e in _fb_ents}
                                                _fb_full_graph = {
                                                    "entities": _fb_ents,
                                                    "relationships": [
                                                        r for r in _fb_full_graph.get("relationships", [])
                                                        if r["from_entity"] in _fb_ent_names
                                                        and r["to_entity"]   in _fb_ent_names
                                                    ],
                                                    "properties": _fb_full_graph.get("properties", []),
                                                }
                                                log.debug(
                                                    "result_chat fallback graph scoped to schemas %s",
                                                    _orig_schemas,
                                                )
                                        _fb_graph_ctx = _graph_resolve(
                                            question   = rc_question,
                                            account_id = account_id,
                                            db_type    = _fb_db_cfg.get("db_type", "azure_sql"),
                                            graph      = _fb_full_graph,
                                        )
                                except Exception as _gex:
                                    log.debug("result_chat fallback graph resolution skipped: %s", _gex)

                                # 7. Table coverage guarantee — fill any JOIN gaps RAG missed
                                if _fb_graph_ctx.get("enabled"):
                                    try:
                                        from core.table_coverage import (
                                            build_required_fqns,
                                            guarantee_table_coverage,
                                        )
                                        _fb_required = build_required_fqns(_fb_graph_ctx, _fb_full_graph)
                                        if _fb_required:
                                            _fb_gap_docs = guarantee_table_coverage(
                                                account_id     = account_id,
                                                required_fqns  = _fb_required,
                                                retrieved_docs = _fb_fresh_docs,
                                                rag_filter     = None,
                                                max_fill       = 3,
                                            )
                                            if _fb_gap_docs:
                                                _fb_rag_ctx += "\n\n---\n\n" + "\n\n---\n\n".join(_fb_gap_docs)
                                    except Exception:
                                        pass

                                # 8. Assemble full context (same priority order as main pipeline)
                                _fb_context_parts = [
                                    p for p in (
                                        _fb_term_inj,
                                        _fb_synonym_inj,
                                        _fb_generic_hints,
                                        _fb_rag_ctx,
                                    ) if p
                                ]
                                _fb_full_ctx = "\n\n".join(_fb_context_parts)

                            if _fb_db_cfg and _fb_full_ctx:
                                await websocket.send_json({
                                    "type": "result_chat_typing",
                                    "result_id": rc_result_id,
                                    "active": True,
                                    "message": "Querying your database for a complete answer…",
                                })
                                _fb_prov, _fb_model, _fb_key, _fb_az = resolve_provider(
                                    client, purpose="query"
                                )
                                # Full system prompt — same as main pipeline, with graph context
                                _fb_system = build_sql_system_prompt(
                                    _fb_db_cfg.get("db_type", "azure_sql"),
                                    _fb_full_ctx,
                                    graph_context=_fb_graph_ctx or None,
                                )
                                # Conversation history for this result card (last 5 turns)
                                if _rc_history:
                                    _fb_hist_lines = ["Session context (recent result-chat turns):"]
                                    for _ht in _rc_history[-3:]:
                                        _fb_hist_lines.append(
                                            f"  Q: {_ht.get('question','')[:80]}"
                                        )
                                        if _ht.get("sql"):
                                            _fb_hist_lines.append(
                                                f"  SQL: {_ht['sql'][:120]}"
                                            )
                                    _fb_system = _fb_system + "\n\n" + "\n".join(_fb_hist_lines)

                                # Original SQL anchor — unified for both aggregate and
                                # identifier results. The LLM can use it as a subquery,
                                # CTE, or just keep the same WHERE conditions.
                                _prev_rows = _cached_result.get("rows") or []
                                _orig_sql  = _cached_result.get("sql", "")
                                _orig_q    = _cached_result.get("question", "")
                                _drill_ctx = _build_followup_sql_context(
                                    original_sql      = _orig_sql,
                                    original_question = _orig_q,
                                    follow_up_question= rc_question,
                                    prev_rows         = _prev_rows,
                                    schema            = _rc_schema,
                                    has_identifiers   = _fb_has_ids,
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
                                    _fb_semantic_context = {
                                        "intent": analyze_query_intent(rc_question),
                                        "question": rc_question,
                                        "graph_context": _fb_graph_ctx,
                                    }
                                    _fb_ok, _fb_reason, _fb_code = validate_sql(
                                        _fb_sql_raw, _ws_known_tables,
                                        _fb_db_cfg.get("db_type", "azure_sql"),
                                        None,
                                        _ws_table_columns,
                                        _fb_semantic_context,
                                    )
                                    # One repair attempt when validation fails
                                    # (unknown_table or parse error — same as main pipeline)
                                    if not _fb_ok and _fb_code in ("unknown_table", "unknown_column", "date_key_format", "anti_join_shape", "parse"):
                                        log.info(
                                            "result_chat fallback SQL failed validation (%s): %s — retrying",
                                            _fb_code, _fb_reason,
                                        )
                                        _fb_retry_user = (
                                            f"The following SQL failed validation: {_fb_reason}\n"
                                            f"SQL: {_fb_sql_raw}\n\n"
                                            f"The original question was: {rc_question}\n\n"
                                            "If the error says a column exists on another table, "
                                            "switch the source table or add the required JOIN to that table. "
                                            "Do not retry the same invalid column/table pair. "
                                            "For missing-record questions, use LEFT JOIN ... WHERE right_key IS NULL. "
                                            "Rewrite the SQL using ONLY table and column names that "
                                            "appear verbatim in the Knowledge Base. "
                                            "If unsure of column names, use SELECT TOP 20 * from the "
                                            "same table. Return only the corrected SQL."
                                        )
                                        _fb_retry_raw, _, _ = await llm_complete(
                                            _fb_system, _fb_retry_user,
                                            _fb_prov, _fb_model, _fb_key,
                                            max_tokens=512, **_fb_az,
                                        )
                                        if _fb_retry_raw and _fb_retry_raw.startswith("```"):
                                            _fb_retry_raw = "\n".join(
                                                _fb_retry_raw.split("\n")[1:]
                                            ).rsplit("```", 1)[0].strip()
                                        if _fb_retry_raw and "CANNOT_GENERATE" not in _fb_retry_raw.upper():
                                            _fb_ok2, _, _ = validate_sql(
                                                _fb_retry_raw, _ws_known_tables,
                                                _fb_db_cfg.get("db_type", "azure_sql"),
                                                None,
                                                _ws_table_columns,
                                                _fb_semantic_context,
                                            )
                                            if _fb_ok2:
                                                _fb_sql_raw = _fb_retry_raw
                                                _fb_ok = True
                                                log.info("result_chat fallback SQL repair succeeded")
                                    if _fb_ok:
                                        try:
                                            _fb_governed = _ws_execute_governed(
                                                _fb_db_cfg,
                                                _fb_sql_raw,
                                                _fb_semantic_context,
                                            )
                                            _fb_rows = _fb_governed.rows
                                            _fb_sql = _fb_governed.sql
                                        except Exception as _exec_exc:
                                            # Execution failed (e.g. Invalid column name).
                                            # One repair attempt — same pattern as main pipeline.
                                            import re as _re_exec
                                            _exec_err = str(_exec_exc)
                                            log.info(
                                                "result_chat fallback execution failed: %s — retrying",
                                                _exec_err[:120],
                                            )
                                            _bad_cols = _re_exec.findall(
                                                r"Invalid column name '([^']+)'",
                                                _exec_err, _re_exec.IGNORECASE,
                                            )
                                            _col_note = ""
                                            if _bad_cols:
                                                _cols_str = ", ".join(f"'{c}'" for c in _bad_cols)
                                                _col_note = (
                                                    f"\n⚠️  COLUMN NAME ERROR: The column(s) {_cols_str} "
                                                    f"do NOT exist in the database.\n"
                                                    f"Find the EXACT column names in the Knowledge Base "
                                                    f"(system prompt). NEVER guess or use CamelCase variants.\n"
                                                )
                                            _exec_retry_user = (
                                                f"The following SQL failed with this error:\n"
                                                f"SQL: {_fb_sql_raw}\n"
                                                f"Error: {_exec_err}\n"
                                                f"{_col_note}\n"
                                                f"The original question was: {rc_question}\n\n"
                                                "Rewrite the SQL to fix the error. Use ONLY column names "
                                                "that appear verbatim in the Knowledge Base. "
                                                "Return only the corrected SQL."
                                            )
                                            _exec_retry_raw, _, _ = await llm_complete(
                                                _fb_system, _exec_retry_user,
                                                _fb_prov, _fb_model, _fb_key,
                                                max_tokens=512, **_fb_az,
                                            )
                                            if _exec_retry_raw and _exec_retry_raw.startswith("```"):
                                                _exec_retry_raw = "\n".join(
                                                    _exec_retry_raw.split("\n")[1:]
                                                ).rsplit("```", 1)[0].strip()
                                            if _exec_retry_raw and "CANNOT_GENERATE" not in _exec_retry_raw.upper():
                                                _exec_ok, _, _ = validate_sql(
                                                    _exec_retry_raw, _ws_known_tables,
                                                    _fb_db_cfg.get("db_type", "azure_sql"),
                                                    None,
                                                    _ws_table_columns,
                                                    _fb_semantic_context,
                                                )
                                                if _exec_ok:
                                                    _fb_governed = _ws_execute_governed(
                                                        _fb_db_cfg,
                                                        _exec_retry_raw,
                                                        _fb_semantic_context,
                                                    )
                                                    _fb_rows = _fb_governed.rows
                                                    _fb_sql = _fb_governed.sql
                                                    log.info("result_chat fallback execution repair succeeded")
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
                                "column_formats":   _rc_formats,
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
                        _rc_chart_type = detect_chart_type(
                            _rc_rows,
                            question=rc_question,
                            column_formats=_rc_formats,
                        )
                        if _rc_chart_type:
                            _rc_chart = build_chart_payload(
                                _rc_rows,
                                _rc_chart_type,
                                title=rc_question,
                                question=rc_question,
                                column_formats=_rc_formats,
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
                        "column_formats":   _rc_formats,
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
                    cached = adapter.last_result

                    # Numeric-value check that works for Python int/float AND
                    # decimal.Decimal returned by Azure SQL / pyodbc.
                    def _to_float(v):
                        try:
                            f = float(str(v).replace(",", "").replace("$", "").replace("%", ""))
                            return None if f != f else f
                        except (TypeError, ValueError):
                            return None

                    # ── drill_dim: add a dimension to the result ─────────────
                    # action format: "drill_dim:{DimensionName}"
                    if action.startswith("drill_dim:") and cached and cached.get("rows"):
                        _dim_name = action[len("drill_dim:"):]
                        try:
                            from core.drill_dimension import generate_drill_by_dimension
                            provider, model, api_key, az_kwargs = resolve_provider(client, purpose="query")
                            _dd_plan = cached.get("semantic_plan") or {}
                            with llm_audit_scope(
                                account_id=account_id,
                                question=f"drill_dim:{_dim_name}: {cached.get('question', '')}".strip(),
                                enabled=bool(client.get("enable_llm_audit")),
                                request_id=make_llm_audit_request_id(),
                                question_id=getattr(adapter, "last_question_id", None) or "",
                                component="drill_dim",
                            ):
                                _dd_result = await generate_drill_by_dimension(
                                    dim_name=_dim_name,
                                    rows=cached["rows"],
                                    question=cached.get("question", ""),
                                    original_sql=cached.get("sql", ""),
                                    semantic_plan=_dd_plan,
                                    db_cfg=cached.get("db_cfg") or {},
                                    known_tables=_ws_known_tables,
                                    provider=provider,
                                    model=model,
                                    api_key=api_key,
                                    query_executor=_ws_execute_governed,
                                    **az_kwargs,
                                )
                            await websocket.send_json(_dd_result)
                            # Cache the drill result so subsequent actions apply to it
                            if _dd_result.get("type") == "assistant_response":
                                _dd_cache_fn = getattr(adapter, "cache_result", None)
                                if callable(_dd_cache_fn):
                                    _dd_cache_fn(
                                        _dd_result.get("data", {}).get("rows") or [],
                                        _dd_result.get("question", ""),
                                        (_dd_result.get("trust") or {}).get("sql", ""),
                                        cached.get("db_cfg"),
                                        cached.get("rag_context", ""),
                                        semantic_plan=_dd_plan,
                                        data_brief=_dd_result.get("data_brief") or {},
                                    )
                        except Exception as _dd_err:
                            log.warning("drill_dim failed: %s", _dd_err)
                            await websocket.send_json({
                                "type": "assistant_error",
                                "action": "drill_dim",
                                "content": "Could not complete the drill-down.",
                                "suggestion": f"Try asking: \"Break down by {_dim_name}\" directly.",
                            })
                        finally:
                            await websocket.send_json({"type": "typing", "active": False})
                        continue

                    # ── compare_prior: fetch prior period from DB ─────────────
                    # This is the only action that requires a live DB call —
                    # all other actions work purely from the cached result rows.
                    if action == "compare_prior" and cached and cached.get("rows"):
                        try:
                            from core.period_comparison import generate_period_comparison
                            provider, model, api_key, az_kwargs = resolve_provider(client, purpose="query")
                            _cp_brief = cached.get("data_brief") or {}
                            if not _cp_brief:
                                # brief not cached on older results — recompute
                                from core.insight import compute_data_brief
                                _cp_brief = compute_data_brief(
                                    cached["rows"], cached.get("question", "")
                                )
                            with llm_audit_scope(
                                account_id=account_id,
                                question=f"compare_prior: {cached.get('question', '')}".strip(),
                                enabled=bool(client.get("enable_llm_audit")),
                                request_id=make_llm_audit_request_id(),
                                question_id=getattr(adapter, "last_question_id", None) or "",
                                component="compare_prior",
                            ):
                                _cp_result = await generate_period_comparison(
                                    rows=cached["rows"],
                                    question=cached.get("question", ""),
                                    original_sql=cached.get("sql", ""),
                                    data_brief=_cp_brief,
                                    db_cfg=cached.get("db_cfg") or {},
                                    known_tables=_ws_known_tables,
                                    provider=provider,
                                    model=model,
                                    api_key=api_key,
                                    query_executor=_ws_execute_governed,
                                    business_context=cached.get("rag_context", ""),
                                    semantic_plan=cached.get("semantic_plan"),
                                    **az_kwargs,
                                )
                            await websocket.send_json(_cp_result)
                        except Exception as _cp_err:
                            log.warning("compare_prior failed: %s", _cp_err)
                            await websocket.send_json({
                                "type": "assistant_analysis",
                                "action": "compare_prior",
                                "title": "Prior period comparison",
                                "headline": "Could not complete the prior period comparison.",
                                "body": (
                                    "An unexpected error occurred while preparing the prior period. "
                                    "Try asking the comparison directly in your question."
                                ),
                                "bullets": [],
                                "next_step": "Ask: \"Show [metric] for [period A] vs [period B]\"",
                            })
                        finally:
                            await websocket.send_json({"type": "typing", "active": False})
                        continue

                    # ── contribution: append % share column to cached rows ────
                    # Pure Python transform — no LLM, no DB call.
                    if action == "contribution" and cached and cached.get("rows"):
                        try:
                            from core.result_transforms import (
                                add_contribution_pct, describe_contribution_sql,
                            )
                            from core.response_builder import build_assistant_response
                            _ct_rows  = cached["rows"]
                            _ct_ctx   = cached.get("analysis_context") or {}
                            _ct_mcol  = _ct_ctx.get("value_col") or (
                                # fallback: first numeric col in the first row.
                                # Use _to_float so decimal.Decimal (returned by
                                # Azure SQL / pyodbc) is recognised as numeric,
                                # not just Python int/float.
                                next((k for k, v in (_ct_rows[0] if _ct_rows else {}).items()
                                      if _to_float(v) is not None), "")
                            )
                            _ct_result, _ct_stats = add_contribution_pct(_ct_rows, _ct_mcol)
                            if not _ct_stats.get("ok"):
                                await websocket.send_json({
                                    "type": "assistant_error",
                                    "action": "contribution",
                                    "content": "Could not compute contribution share.",
                                    "detail": _ct_stats.get("reason", ""),
                                })
                            else:
                                _ct_sql = describe_contribution_sql(
                                    _ct_mcol, _ct_stats["total"]
                                )
                                _ct_resp = build_assistant_response(
                                    question=f"{cached.get('question', '')} (% contribution)",
                                    rows=_ct_result,
                                    sql=_ct_sql,
                                    duration_ms=0,
                                    data_source=str((cached.get("db_cfg") or {}).get("db_type", "")),
                                    semantic_plan=cached.get("semantic_plan"),
                                )
                                _ct_resp["contribution_stats"] = _ct_stats
                                await websocket.send_json(_ct_resp)
                        except Exception as _ct_err:
                            log.warning("contribution transform failed: %s", _ct_err)
                            await websocket.send_json({
                                "type": "assistant_error",
                                "action": "contribution",
                                "content": "Could not compute the % share breakdown.",
                            })
                        finally:
                            await websocket.send_json({"type": "typing", "active": False})
                        continue

                    # ── outliers: filter cached rows to exceptional values ─────
                    # Pure Python transform — no LLM, no DB call.
                    if action == "outliers" and cached and cached.get("rows"):
                        try:
                            from core.result_transforms import (
                                filter_outliers, describe_outlier_sql,
                            )
                            from core.response_builder import build_assistant_response
                            _ol_rows = cached["rows"]
                            _ol_ctx  = cached.get("analysis_context") or {}
                            _ol_mcol = _ol_ctx.get("value_col") or (
                                next((k for k, v in (_ol_rows[0] if _ol_rows else {}).items()
                                      if _to_float(v) is not None), "")
                            )
                            _ol_result, _ol_stats = filter_outliers(_ol_rows, _ol_mcol)
                            if not _ol_stats.get("ok"):
                                await websocket.send_json({
                                    "type": "assistant_error",
                                    "action": "outliers",
                                    "content": (
                                        _ol_stats.get("detail")
                                        or "No outliers found in this result."
                                    ),
                                })
                            else:
                                _ol_sql = describe_outlier_sql(_ol_mcol, _ol_stats)
                                _ol_resp = build_assistant_response(
                                    question=f"{cached.get('question', '')} (exceptions only)",
                                    rows=_ol_result,
                                    sql=_ol_sql,
                                    duration_ms=0,
                                    data_source=str((cached.get("db_cfg") or {}).get("db_type", "")),
                                    semantic_plan=cached.get("semantic_plan"),
                                )
                                _ol_resp["outlier_stats"] = _ol_stats
                                await websocket.send_json(_ol_resp)
                        except Exception as _ol_err:
                            log.warning("outlier filter failed: %s", _ol_err)
                            await websocket.send_json({
                                "type": "assistant_error",
                                "action": "outliers",
                                "content": "Could not filter outliers from this result.",
                            })
                        finally:
                            await websocket.send_json({"type": "typing", "active": False})
                        continue

                    # ── download_csv: generate CSV from cached rows ──────────
                    # Pure Python — no LLM, no DB call.
                    if action == "download_csv" and cached and cached.get("rows"):
                        try:
                            from core.compliance.policy_engine import evaluate, resolve_context
                            from core.compliance.sql_guard import analyze_sql
                            from core.export import rows_to_csv, build_csv_filename
                            _csv_analysis = analyze_sql(
                                cached.get("sql", ""),
                                (cached.get("db_cfg") or {}).get("db_type", "azure_sql"),
                            )
                            _csv_context = resolve_context(
                                account_id,
                                portal_user,
                                action="export",
                                channel="portal",
                            )
                            _csv_decision = evaluate(
                                _csv_context, _csv_analysis.resources
                            )
                            if (
                                not _csv_decision.effective_allowed
                                or not _csv_decision.export_allowed
                            ):
                                await websocket.send_json({
                                    "type": "assistant_error",
                                    "action": "download_csv",
                                    "content": (
                                        _csv_decision.explanation
                                        or "Export is blocked by the workspace data policy."
                                    ),
                                })
                                continue
                            _csv_rows     = cached["rows"]
                            _csv_col_fmts = cached.get("column_formats") or {}
                            _csv_content  = rows_to_csv(
                                _csv_rows, column_formats=_csv_col_fmts
                            )
                            _csv_filename = build_csv_filename(
                                cached.get("question", "")
                            )
                            await websocket.send_json({
                                "type":      "assistant_export",
                                "action":    "download_csv",
                                "format":    "csv",
                                "filename":  _csv_filename,
                                "content":   _csv_content,
                                "row_count": len(_csv_rows),
                            })
                        except Exception as _csv_err:
                            log.warning("download_csv failed: %s", _csv_err)
                            await websocket.send_json({
                                "type":    "assistant_error",
                                "action":  "download_csv",
                                "content": "Could not generate CSV from this result.",
                            })
                        finally:
                            await websocket.send_json({"type": "typing", "active": False})
                        continue

                    # ── set_alert: define a change-monitoring alert ──────────
                    # Creates a persisted alert definition; no DB read at
                    # creation time — baseline_value comes from the cached rows.
                    if action == "set_alert" and cached and cached.get("rows"):
                        try:
                            from core.compliance.policy_engine import evaluate, resolve_context
                            from core.compliance.sql_guard import analyze_sql
                            from core.alert_engine import create_alert
                            _alert_analysis = analyze_sql(
                                cached.get("sql", ""),
                                (cached.get("db_cfg") or {}).get("db_type", "azure_sql"),
                            )
                            _alert_context = resolve_context(
                                account_id, portal_user, action="alert", channel="portal"
                            )
                            _alert_decision = evaluate(
                                _alert_context, _alert_analysis.resources
                            )
                            if not _alert_decision.effective_allowed:
                                await websocket.send_json({
                                    "type": "assistant_error",
                                    "action": "set_alert",
                                    "content": (
                                        _alert_decision.explanation
                                        or "Alerts are blocked by the workspace data policy."
                                    ),
                                })
                                continue
                            _al_rows = cached["rows"]
                            _al_ctx  = cached.get("analysis_context") or {}
                            # Prefer the value_col the response_builder identified;
                            # fall back to the first numeric key in the first row.
                            _al_mcol = _al_ctx.get("value_col") or (
                                next(
                                    (k for k, v in (_al_rows[0] if _al_rows else {}).items()
                                     if _to_float(v) is not None),
                                    "",
                                )
                            )
                            _al_raw = (
                                _al_rows[0].get(_al_mcol)
                                if _al_rows and _al_mcol else None
                            )
                            if _al_raw is None or not _al_mcol:
                                await websocket.send_json({
                                    "type":    "assistant_error",
                                    "action":  "set_alert",
                                    "content": (
                                        "Could not identify a numeric metric to monitor. "
                                        "Ask for a specific KPI result first."
                                    ),
                                })
                            else:
                                try:
                                    _al_baseline = float(str(_al_raw).replace(",", ""))
                                except (TypeError, ValueError):
                                    _al_baseline = 0.0
                                _al_def = create_alert(
                                    question  = cached.get("question", ""),
                                    sql       = cached.get("sql", ""),
                                    metric_col= _al_mcol,
                                    baseline_value = _al_baseline,
                                    condition  = "change_pct",
                                    threshold  = 10.0,
                                    db_cfg     = cached.get("db_cfg") or {},
                                    account_id = account_id,
                                    user_id    = str(portal_user.get("id") or ""),
                                    purpose_id = _alert_context.purpose_id,
                                )
                                await websocket.send_json({
                                    "type":      "assistant_analysis",
                                    "action":    "set_alert",
                                    "title":     "Alert created",
                                    "body": (
                                        f"I'll monitor **{_al_mcol}** (baseline: {_al_raw}) "
                                        f"and flag it when the value changes by more than 10%."
                                    ),
                                    "secondary": (
                                        f"Alert ID: {_al_def['id']} — "
                                        "use this ID to check the current value against "
                                        "the baseline at any time."
                                    ),
                                    "bullets": [
                                        f"Metric: {_al_mcol}",
                                        f"Baseline: {_al_raw}",
                                        "Trigger: change > 10%",
                                        "Condition: change_pct",
                                    ],
                                    "next_step": (
                                        "Ask \"Check alert " + _al_def["id"] + "\" "
                                        "to compare the current value to this baseline."
                                    ),
                                    "alert_id": _al_def["id"],
                                    "alert":    _al_def,
                                })
                        except Exception as _al_err:
                            log.warning("set_alert failed: %s", _al_err)
                            await websocket.send_json({
                                "type":    "assistant_error",
                                "action":  "set_alert",
                                "content": "Could not create the alert.",
                            })
                        finally:
                            await websocket.send_json({"type": "typing", "active": False})
                        continue

                    # ── diagnose: root-cause analysis for significant drops/rises ─
                    # Runs the full drilldown pipeline: LLM generates breakdown
                    # SQL queries, executes them, then synthesises a narrative
                    # explaining what dimension/segment drove the change.
                    if action == "diagnose" and cached and cached.get("rows"):
                        try:
                            provider, model, api_key, az_kwargs = resolve_provider(client, purpose="query")
                            from core.response_builder import generate_analysis_response
                            _dx_brief = cached.get("data_brief") or {}
                            _dx_ts    = _dx_brief.get("time_series") or {}
                            _dx_pct   = _dx_ts.get("overall_pct_change") or 0.0
                            _dx_dir   = _dx_ts.get("direction") or "stable"
                            _dx_sign  = "dropped" if _dx_pct < 0 else "rose"
                            _dx_follow_up = (
                                f"The metric {_dx_sign} by {abs(_dx_pct):.1f}% ({_dx_dir}). "
                                "Break it down by the available dimensions to identify which "
                                "segment or category drove this change the most. "
                                "In 2-3 sentences, explain the primary cause."
                            )
                            with llm_audit_scope(
                                account_id=account_id,
                                question=f"diagnose: {cached.get('question', '')}".strip(),
                                enabled=bool(client.get("enable_llm_audit")),
                                request_id=make_llm_audit_request_id(),
                                question_id=getattr(adapter, "last_question_id", None) or "",
                                component="diagnose",
                            ):
                                insight = await generate_analysis_response(
                                    action="why",
                                    rows=cached["rows"],
                                    question=cached.get("question", ""),
                                    provider=provider,
                                    model=model,
                                    api_key=api_key,
                                    follow_up=_dx_follow_up,
                                    original_sql=cached.get("sql", ""),
                                    db_cfg=cached.get("db_cfg"),
                                    context=cached.get("rag_context", ""),
                                    known_tables=_ws_known_tables,
                                    query_executor=_ws_execute_governed,
                                    **az_kwargs,
                                )
                            # Override the title so the card is clearly labelled
                            if isinstance(insight, dict):
                                insight["action"] = "diagnose"
                                if not insight.get("title"):
                                    insight["title"] = "Root cause analysis"
                            await websocket.send_json(insight)
                        except Exception as _dx_err:
                            log.warning("diagnose action failed: %s", _dx_err)
                            await websocket.send_json({
                                "type": "assistant_analysis",
                                "action": "diagnose",
                                "title": "Root cause analysis",
                                "body": (
                                    "I could not run the breakdown automatically. "
                                    "Try asking directly: \"Why did this value change?\" "
                                    "or \"Break it down by [dimension]\"."
                                ),
                                "bullets": [],
                            })
                        finally:
                            await websocket.send_json({"type": "typing", "active": False})
                        continue

                    # ── standard action buttons (explain, analyze, compare …) ─
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
                                    query_executor=_ws_execute_governed,
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
                            query_executor=_ws_execute_governed,
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
                    # Defect class: a crash while generating/sending the
                    # answer used to end in total silence — the query ran,
                    # the server log had the error, the user saw nothing.
                    # Degrade to a visible generic error instead.
                    try:
                        await websocket.send_json({
                            "type": "assistant_error",
                            "role": "assistant",
                            "content": (
                                "Something went wrong while preparing your answer — "
                                "please try asking again."
                            ),
                        })
                    except Exception:
                        pass

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

