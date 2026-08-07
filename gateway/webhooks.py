"""
gateway/webhooks.py
───────────────────
Platform webhook endpoints and the WebSocket chat handler, extracted from main.py.

Routes registered on an APIRouter that main.py mounts at startup.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time

import store
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

from gateway import get_adapter, PlatformEvent
from core.webhook_dedup import is_duplicate_event, remember_event, get_user_serialization_lock
from core.dispatcher import dispatch
from core.query_pipeline import handle_query
from core.pipeline_context import get_state, get_client_db
from core.pipeline_trace import (
    _log_q, _trace_create, _trace_update, _trace_step, _trace_finish,
)
from core.pipeline_helpers import _extract_kb_synonym_injection
from core.result_renderer import (
    _sanitize_rows, _result_has_identifiers, _build_metadata_followup_context,
    _build_cannot_generate_hint,
    _inject_distinct_if_needed,
)
from core.result_cache import result_cache
from core.result_commands import (
    compile_confirmed_result_presentation,
    execute_result_command,
    needs_result_reference_confirmation,
    parse_result_command,
)
from core.governed_result_followup import adopt_cached_snapshot, run_governed_result_followup
from core.result_planner import (
    is_metadata_result_question,
    strip_result_context,
)
from core.query_router import should_route_to_result_cache
from core.schema import run_query, load_known_tables, load_schema_columns
from core.knowledge import load_retriever
from core.validator import validate_sql
from core.llm import llm_complete, build_sql_system_prompt, resolve_provider
from core.llm_audit import llm_audit_scope, make_llm_audit_request_id
from core.query_semantics import analyze_query_intent, build_generic_query_hints
from core.graph_resolver import resolve_for_question as _graph_resolve
from core.chart import detect_chart_type, build_chart_payload
from core.examples import retrieve_similar_examples, format_examples_for_prompt
from core.clarification import (
    get_pending, save_pending, clear_pending, combine_with_clarification,
    resolve_option_text, resolve_date_option_text, attach_clarification_resolution,
    prepare_clarification_meta,
)
from core.plan_preview import build_plan_preview, pending_plan_previews
from core.agent_runtime import AgentRunSession, activate_agent_run

log = logging.getLogger("querybot")

router = APIRouter()

# Conversational report/playbook builder intent -- deliberately simple
# keyword/regex fast path (mirrors core/result_commands.py's style), not an
# LLM classification, so it never adds a round-trip to ordinary questions.
# Available to every portal user, not gated by role.
_REPORT_BUILDER_INTENT_RE = re.compile(
    r"\b(?:build|create|make|set\s*up|start|schedule)\s+(?:me\s+)?(?:a|an|my)?\s*report\b",
    re.IGNORECASE,
)

# Conversational dashboard artifacts. These routes never generate or execute
# SQL themselves: they attach only an already-governed result, or queue the
# normal governed question pipeline and materialize its successful answer.
_DASHBOARD_CREATE_INTENT_RE = re.compile(
    r"\b(?:build|create|make|start|set\s*up)\s+(?:me\s+)?(?:a|an|my)?\s*dashboard\b",
    re.IGNORECASE,
)
_DASHBOARD_ADD_INTENT_RE = re.compile(
    r"\b(?:add|put|save|place)\s+"
    r"(?:it|this|that|(?:(?:this|that|the|my)\s+)?(?:result|chart|visual|table|kpi))\s+"
    r"(?:to|on|in)\s+(?:(?:this|the|my)\s+)?dashboard\b",
    re.IGNORECASE,
)
_DASHBOARD_ADD_QUERY_RE = re.compile(
    r"\badd\s+(?:a\s+)?(?:chart|visual|table|kpi|view)\s+"
    r"(?:showing|for|of|with)?\s*(?P<question>.+?)\s+to\s+"
    r"(?:this|the|my)?\s*dashboard\b",
    re.IGNORECASE,
)
_DASHBOARD_CHART_TYPE_RE = re.compile(
    r"\b(?:change|switch|make|set)\s+(?:this|the|my)?\s*dashboard(?:'s)?\s+"
    r"(?:chart|visual)(?:\s+type)?\s+(?:to|as)\s+"
    r"(?P<chart_type>bar|line|area|pie|donut|scatter)\b",
    re.IGNORECASE,
)
_DASHBOARD_RENAME_INTENT_RE = re.compile(
    r"\brename\s+(?:this|the|my)?\s*dashboard\s+to\s+(?P<name>.+)$",
    re.IGNORECASE,
)
_DASHBOARD_PUBLISH_INTENT_RE = re.compile(
    r"\bpublish\s+(?:this|the|my)?\s*dashboard\b",
    re.IGNORECASE,
)
_DASHBOARD_FILTER_INTENT_RE = re.compile(
    r"\badd\s+(?:a\s+)?filter\s+(?:for|on|by)\s+(?P<field>[A-Za-z0-9_. -]{1,120}?)"
    r"(?:\s+to\s+(?:this|the|my)?\s*dashboard)?\s*$",
    re.IGNORECASE,
)
_DASHBOARD_TAB_INTENT_RE = re.compile(
    r"\badd\s+(?:a\s+)?tab(?:\s+(?:called|named|for))?\s+(?P<tab>[A-Za-z0-9 _.-]{1,60}?)"
    r"(?:\s+to\s+(?:this|the|my)?\s*dashboard)?\s*$",
    re.IGNORECASE,
)
_DASHBOARD_SCHEDULE_INTENT_RE = re.compile(
    r"\b(?:refresh|schedule|update)\s+(?:this|the|my)?\s*dashboard\s+"
    r"(?P<schedule>manually|manual|hourly|daily|weekly)\b",
    re.IGNORECASE,
)
_DASHBOARD_SHARE_INTENT_RE = re.compile(
    r"\b(?:share|publish)\s+(?:this|the|my)?\s*dashboard\s+(?:with|to)\s+(?:the\s+)?team\b",
    re.IGNORECASE,
)
_DASHBOARD_ROLLBACK_INTENT_RE = re.compile(
    r"\b(?:rollback|roll\s+back|restore)\s+(?:this|the|my)?\s*dashboard\s+"
    r"(?:to\s+)?version\s+(?P<version>\d+)\b",
    re.IGNORECASE,
)


def _dashboard_request_tail(text: str) -> str:
    match = _DASHBOARD_CREATE_INTENT_RE.search(text or "")
    if not match:
        return ""
    tail = (text[match.end():] or "").strip(" .,:;-")
    tail = re.sub(
        r"^(?:that\s+is\s+)?(?:showing|show|with|for|of|tracking|about)\s+",
        "",
        tail,
        flags=re.IGNORECASE,
    ).strip()
    if re.fullmatch(
        r"(?:from\s+)?(?:this|that)(?:\s+(?:result|answer|chart))?|"
        r"(?:from\s+)?the\s+(?:result|answer|chart)",
        tail,
        re.I,
    ):
        return ""
    return tail


def _dashboard_name(question: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(question or "")).strip(" .")
    if not cleaned:
        return "Analysis dashboard"
    return (cleaned[:72].rstrip() + " dashboard")[:120]

# Opt-in "explain your plan first" preview -- explicitly user-invoked, never
# a default gate on every question. Matches a trigger phrase optionally
# followed by the actual question after a colon/dash/comma, e.g.
# "explain your plan: what was net revenue for last 7 days".
_PLAN_PREVIEW_INTENT_RE = re.compile(
    r"^\s*(?:explain\s+(?:your\s+)?plan|show\s+(?:me\s+)?(?:your\s+)?plan|"
    r"tell\s+me\s+(?:your\s+plan|how\s+you(?:'d| would)\s+(?:answer|do)\s+this))"
    r"(?:\s+(?:before\s+running|before\s+you\s+run|first))?"
    r"\s*[:\-,]?\s*(?P<question>.*)$",
    re.IGNORECASE,
)
_PLAN_PREVIEW_CONFIRM_RE = re.compile(
    r"^\s*(?:go\s+ahead|yes|yep|yeah|sure|ok(?:ay)?|proceed|continue|"
    r"run\s+it|do\s+it|sounds\s+good|confirm(?:ed)?)\s*[.!]?\s*$",
    re.IGNORECASE,
)

# Reconcile-against-a-known-number intent -- "the real number is X, why is
# yours Y" style phrasing against the active cached result. QueryBot cannot
# see the methodology behind an externally-known number, so this is a
# guided-comparison tool (restate the exact definition + offer testable
# hypotheses), never a promise to "explain the gap".
_RECONCILE_INTENT_RE = re.compile(
    r"\b(?:real|actual|correct|right)\s+number\s+is\b|"
    r"\b(?:my|our)\s+(?:dashboard|report|spreadsheet|system)\s+shows\b|"
    r"\bdoesn'?t\s+match\b|"
    r"\byours?\s+(?:says?|shows?|gives?)\b.{0,20}\bbut\b|"
    r"\bshould\s+be\s+\S.{0,40}\bnot\b|"
    r"\bwalk\s+me\s+through\s+how\s+you\s+calculated\b",
    re.IGNORECASE,
)

# Explicit deep-work follow-ups over an already governed result.  This route
# never queries the database. Prebuilt operations and administrator-enabled
# governed Python both receive only bounded copies of released result rows.
_ANALYSIS_WORK_INTENT_RE = re.compile(
    r"\b(?:analyse|analyze|profile|inspect)\s+"
    r"(?:(?:this|these)\s+|(?:(?:my|the)\s+)?"
    r"(?:(?:latest|last|previous|most\s+recent)\s+)?)"
    r"(?:successful\s+)?"
    r"(?:result|results|data|answer|rows)(?:\s+(?:deeply|thoroughly|in\s+depth))?\b|"
    r"\b(?:find|show|check|identify)\s+(?:the\s+)?(?:correlations?|relationships?|"
    r"outliers?|anomalies|trends?)\s+(?:in|within|from)\s+(?:this|the|these)?\s*"
    r"(?:result|results|data|answer|rows)\b|"
    r"\b(?:use|run)\s+python\s+(?:to\s+)?(?:analyse|analyze|profile|inspect)\b",
    re.IGNORECASE,
)
_CUSTOM_PYTHON_INTENT_RE = re.compile(
    r"```(?:python|py)?\s*\n|"
    r"\b(?:run|execute)\s+(?:this\s+)?python\s*:|"
    r"\b(?:use|using|with|in)\s+python\s+(?:to\s+)?(?:calculate|compute|derive|"
    r"transform|rank|normalize|analyse|analyze|build|create|find|show)|"
    r"\bpython\s+(?:analysis|calculation|transform)\b",
    re.IGNORECASE,
)


def _analysis_artifact_answer(
    operation: str,
    summary: str,
    derived_row_count: int,
    input_row_count: int,
) -> dict:
    """Build an honest headline for a worker-produced analysis table.

    Profile/outlier tables contain administrative columns such as ``count``
    and ``mean``. Passing them through the generic ranking summarizer can emit
    nonsense like "TOTAL_ORDERS leads at 1". The isolated worker's summary is
    already deterministic and audited, so it is the authoritative narrative.
    """
    label = {
        "profile": "profile row",
        "outliers": "potential outlier",
        "correlation": "correlation pair",
        "trend": "trend row",
        "python": "derived row",
    }.get(operation, "analysis row")
    plural = "" if derived_row_count == 1 else "s"
    return {
        "headline": summary or f"{operation.title()} analysis completed.",
        "short_value": f"{derived_row_count} {label}{plural}",
        "comparison": f"Based on {input_row_count} returned row{'s' if input_row_count != 1 else ''}",
        "scope_badge": "Returned result only",
        "scope_note": "Calculated in an isolated worker without a new database query.",
    }


def _ws_text_value(value, *preferred_keys: str) -> str:
    """Return a safe text value from a WebSocket payload field.

    Autocomplete and suggestion features may carry structured metadata. A
    hint object must never tear down live chat because plain text was expected.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, dict):
        for key in preferred_keys:
            candidate = value.get(key)
            if isinstance(candidate, str):
                candidate = candidate.strip()
                if candidate:
                    return candidate
            elif isinstance(candidate, (int, float, bool)):
                return str(candidate).strip()
    return ""


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


def _teams_tenant_from_body(body: bytes) -> str:
    try:
        payload = json.loads(body)
    except Exception:
        return ""
    channel_data = payload.get("channelData") or {}
    conversation = payload.get("conversation") or {}
    return str(
        (channel_data.get("tenant") or {}).get("id")
        or conversation.get("tenantId")
        or ""
    ).strip()


def _load_teams_adapter(body: bytes):
    """Load the Teams credentials assigned to the inbound tenant's client."""
    tenant_id = _teams_tenant_from_body(body)
    client = store.get_client_by_teams_tenant_id(tenant_id) if tenant_id else None
    platform_id = (client or {}).get("platform_config_id")
    if platform_id:
        platform = store.get_platform(int(platform_id))
        if (
            platform
            and platform.get("platform_type") == "teams"
            and platform.get("is_active")
        ):
            log.info(
                "Teams: selected assigned platform_config_id=%s for client %s",
                platform_id,
                client.get("account_id"),
            )
            return get_adapter("teams", platform["credentials"])
        log.warning(
            "Teams: client %s references an inactive or invalid Teams platform %s",
            client.get("account_id"),
            platform_id,
        )
    return _load_adapter("teams")


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
    adapter = _load_teams_adapter(body)
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
    # Prefer the explicit per-client mapping (client.teams_tenant_id, set on
    # the client's Settings tab) — it's the only path that still works once
    # more than one client is configured. Fall back to the single-configured-
    # client heuristic for deployments that haven't set the mapping yet.
    if not store.get_client(event.account_id):
        tenant_id = event.account_id
        mapped = store.get_client_by_teams_tenant_id(tenant_id)
        if mapped:
            event.account_id = mapped["account_id"]
            log.info("Teams: mapped tenant %s → client %s (explicit mapping)", tenant_id, event.account_id)
        else:
            all_clients = store.list_clients()
            configured  = [c for c in all_clients if c.get("db_config_id")]
            if len(configured) == 1:
                event.account_id = configured[0]["account_id"]
                log.info("Teams: auto-mapped tenant %s → client %s", tenant_id, event.account_id)
            else:
                log.warning(
                    "Teams: tenant %s not registered; %d configured clients found — "
                    "cannot auto-map (need exactly 1, or set teams_tenant_id on the client)",
                    tenant_id, len(configured),
                )

    bind_session = getattr(adapter, "bind_session", None)
    if callable(bind_session):
        try:
            bind_session(event.account_id, event.user_id)
        except ValueError as exc:
            log.warning("Teams: could not establish governed session: %s", exc)
            raise HTTPException(400, detail="Teams activity is missing tenant or user identity")

    send_status = getattr(adapter, "send_status", None)
    if callable(send_status):
        try:
            await send_status(event, "accepted", "Working on it")
        except Exception as exc:
            log.debug("Teams initial typing indicator failed: %s", exc)

    # Serialize dispatch() per (platform, account, user) -- Teams has no
    # other ordering guarantee between two near-simultaneous messages from
    # the same user (each webhook POST is its own independent request
    # coroutine). See core/webhook_dedup.py::get_user_serialization_lock
    # and core/dispatcher.py::_run_query_with_guard (the same lock also
    # serializes the backgrounded answer-sending work this call enqueues).
    async with get_user_serialization_lock(event):
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
    # Serialize dispatch() per (platform, account, user) -- see the
    # matching comment in webhook_teams above.
    async with get_user_serialization_lock(event):
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
    if (
        not portal_user
        or portal_user.get("account_id") != account_id
        or not portal_user.get("is_active")
    ):
        # is_active re-check: an admin's "Temporarily Stop Access" toggle
        # only blocks fresh logins (store.get_user_by_email filters
        # is_active=1) -- a cookie issued before the toggle would otherwise
        # keep this chat socket open indefinitely.
        await websocket.close(code=4003)
        return

    # Check client has chat UI enabled
    client = store.get_client(account_id) or {}
    if not client.get("chat_ui_enabled"):
        await websocket.close(code=4004)
        return

    await websocket.accept()

    zoom_user_id = portal_user.get("zoom_user_id") or f"web_{user_id}"
    thread_id = _ws_text_value(
        websocket.query_params.get("thread_id"), "thread_id", "value"
    )
    adapter = WebAdapter(
        websocket,
        account_id,
        zoom_user_id,
        thread_id=thread_id,
        portal_user_id=int(user_id),
    )
    selected_dashboard_id = 0
    selected_dashboard_read_only = False
    raw_dashboard_id = str(websocket.query_params.get("dashboard_id") or "")
    if raw_dashboard_id.isdigit():
        selected = store.get_dashboard_for_view(
            int(raw_dashboard_id), int(user_id), account_id
        )
        if selected:
            selected_dashboard_id = int(selected["id"])
            selected_dashboard_read_only = not bool(selected.get("can_edit"))

    # Evaluate session liveness before restoring the selected thread.
    _is_new_portal_session = False
    try:
        _is_new_portal_session = store.touch_user_activity(user_id)
    except Exception as _touch_exc:
        log.debug("Portal session touch skipped: %s", _touch_exc)

    if _is_new_portal_session:
        from core.conversational import build_reply
        await websocket.send_json({
            "type":    "message",
            "role":    "assistant",
            "content": build_reply("greeting", account_id, portal_user),
        })
        try:
            from core.dispatcher import _offer_login_report_prompt
            await _offer_login_report_prompt(account_id, portal_user, adapter.make_event(""), adapter)
        except Exception as _report_prompt_exc:
            log.debug("Login report prompt skipped: %s", _report_prompt_exc)
    else:
        await websocket.send_json({
            "type":    "system",
            "content": f"Connected as {portal_user.get('name', portal_user.get('id', 'user'))}. Ask me anything about your data.",
        })

    # Restore structural multi-turn context from governed server traces. Raw
    # SQL no longer needs to be persisted in browser localStorage.
    try:
        recent_turns = list(reversed(store.list_answer_traces(
            account_id,
            limit=3,
            portal_user_id=int(portal_user["id"]),
            session_id=adapter.session_id,
        )))
        history = []
        for trace in recent_turns:
            raw_rows = trace.get("result_rows") or "[]"
            if isinstance(raw_rows, str):
                try:
                    raw_rows = json.loads(raw_rows)
                except (TypeError, ValueError):
                    raw_rows = []
            columns = list(raw_rows[0]) if isinstance(raw_rows, list) and raw_rows else []
            history.append({
                "question": trace.get("question_text_sanitized") or "",
                "sql": trace.get("generated_sql") or "",
                "columns": columns,
                "row_count": int(trace.get("query_row_count") or 0),
            })
        adapter.load_history(history)
    except Exception as history_exc:
        log.debug("Server thread history restore skipped: %s", history_exc)

    # The currently in-flight main-question task, if any — lets the receive
    # loop below cancel it when the user clicks Stop. Question handling runs
    # as a background task (not awaited inline) specifically so this loop
    # stays free to receive a {"type":"cancel"} message while a query runs.
    current_query_task: asyncio.Task | None = None

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

    log.info("WebSocket chat connected: user=%d account=%s", user_id, account_id)
    # History is restored from this user's server-side thread traces above.

    async def _run_main_question(text: str, table_hint: str, schema_hint: str) -> None:
        """Answers one question. Runs as a background task (see the send
        loop below) so the receive loop stays free to see a "cancel"
        message mid-flight. Wrapped end-to-end in its own error handling
        since nothing awaits this inline anymore — an unhandled exception
        here would otherwise only surface as an asyncio "exception was
        never retrieved" warning, with the user seeing nothing (previously
        such an error would propagate to the outer handler and silently
        end the whole connection; this is a strict improvement, not just
        a refactor).
        """
        bg = BackgroundTasks()
        event = adapter.make_event(text)
        if table_hint:
            event.table_hint = table_hint
        if schema_hint:
            event.schema_hint = schema_hint
        agent_run = None
        try:
            agent_run = AgentRunSession.start(
                account_id=account_id,
                portal_user=portal_user,
                external_thread_id=adapter.thread_id,
                objective=text,
                selected_schema=schema_hint,
                purpose_id=str(getattr(event, "purpose_id", "") or ""),
            )
            await adapter.send_agent_event(agent_run.event("agent_run_started"))
        except Exception as exc:
            # Runtime observability fails open; the established governed query
            # path remains available if audit persistence is unavailable.
            log.warning("Agent run could not be started; continuing query: %s", exc)
        agent_context = activate_agent_run(agent_run)
        agent_context.__enter__()
        background_failure = None
        try:
            await dispatch(account_id, event, adapter, bg, portal_user=portal_user)

            # Run any background tasks synchronously in WebSocket context
            for task in bg.tasks:
                try:
                    await task.func(*task.args, **task.kwargs)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    background_failure = e
                    log.error("WS bg task error: %s", e)
                    # Defect class: a crash while generating/sending the
                    # answer used to end in total silence — the query ran,
                    # the server log had the error, the user saw nothing.
                    # Degrade to a visible generic error instead.
                    try:
                        async with adapter.send_lock:
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

            if agent_run:
                try:
                    if background_failure:
                        agent_run.fail(background_failure)
                    else:
                        agent_run.complete_if_running()
                    await adapter.send_agent_event(agent_run.event("agent_run_finished"))
                except Exception as exc:
                    log.debug("Agent run finalization failed: %s", exc)
            async with adapter.send_lock:
                await websocket.send_json({"type": "typing", "active": False})
        except asyncio.CancelledError:
            # User clicked Stop — the "cancel" branch below already sent a
            # user-facing message, nothing more to do here.
            if agent_run:
                try:
                    agent_run.cancel()
                    await adapter.send_agent_event(agent_run.event("agent_run_finished"))
                except Exception as exc:
                    log.debug("Agent cancellation audit failed: %s", exc)
            raise
        except Exception as e:
            log.error("Main question task failed: %s", e)
            if agent_run:
                try:
                    agent_run.fail(e)
                    await adapter.send_agent_event(agent_run.event("agent_run_finished"))
                except Exception as exc:
                    log.debug("Agent failure audit failed: %s", exc)
            try:
                async with adapter.send_lock:
                    await websocket.send_json({
                        "type": "assistant_error",
                        "role": "assistant",
                        "content": (
                            "Something went wrong while preparing your answer — "
                            "please try asking again."
                        ),
                    })
                    await websocket.send_json({"type": "typing", "active": False})
            except Exception:
                pass
        finally:
            agent_context.__exit__(None, None, None)

    async def _run_local_result_command(
        text: str, command, table_hint: str = "", schema_hint: str = "",
        planner_evidence: dict | None = None,
        precomputed_outcome=None,
    ) -> None:
        """Apply a conservative, deterministic command to the latest result.

        The user's command may contain a regulated value. It is therefore not
        copied into traces or LLM audit text. Execution has no route to either
        the LLM or source database. ``planner_evidence`` records a preceding
        metadata-only planning call; it never contains rows or literal values.
        """
        planner_evidence = dict(planner_evidence or {})
        planner_used = bool(planner_evidence)
        start_ms = int(time.time() * 1000)
        question_id = make_llm_audit_request_id()
        parent_question_id = getattr(adapter, "last_question_id", "") or ""
        session_id = getattr(adapter, "session_id", "") or ""
        trace_id = _trace_create(
            account_id=account_id,
            question_id=question_id,
            parent_question_id=parent_question_id,
            question="Transform the prior result locally",
            portal_user_id=portal_user.get("id") if portal_user else None,
            platform_user_id=zoom_user_id or "",
            session_id=session_id,
            request_source="portal",
            route="deterministic_result_command",
        )
        _trace_step(
            trace_id,
            "receive_result_command",
            output_summary={
                "operation": getattr(command, "action", ""),
                "raw_values_logged": 0,
                "llm_invoked": planner_used,
                "llm_input_mode": "metadata_only" if planner_used else "none",
                "rows_sent_to_llm": 0,
                "sample_values_sent_to_llm": 0,
                "database_queried": False,
            },
        )
        try:
            source_result_id = getattr(adapter, "last_result_id", None)
            outcome = precomputed_outcome or execute_result_command(
                session_id,
                command,
                source_result_id=source_result_id,
            )
            if outcome.clarification_required:
                options = list(outcome.clarification_options or [])
                prompt = outcome.clarification_prompt or outcome.message
                clarification_meta = {
                    "source": "local_result_command",
                    "question": prompt,
                    "options": options,
                }
                save_pending(
                    account_id,
                    zoom_user_id,
                    text,
                    clarification_meta=prepare_clarification_meta(
                        adapter.make_event(text),
                        clarification_meta,
                        source="local_result_command",
                    ),
                    session_id=adapter.session_id,
                )
                _trace_finish(
                    trace_id,
                    status="waiting_for_user",
                    answer_type="clarification",
                    final_answer_summary="Waiting for a governed display-format clarification",
                )
                await adapter.send_clarification_prompt(
                    adapter.make_event(text), prompt, options,
                )
                async with adapter.send_lock:
                    await websocket.send_json({"type": "typing", "active": False})
                return
            if not outcome.ok:
                if not outcome.handled and getattr(command, "fallback_allowed", False):
                    _trace_finish(
                        trace_id,
                        status="success",
                        answer_type="governed_fallback",
                        final_answer_summary=(
                            "Cached result lacked required columns; routed to governed query pipeline"
                        ),
                    )
                    await _run_main_question(
                        outcome.retry_question or text, table_hint, schema_hint,
                    )
                    return
                _trace_finish(
                    trace_id,
                    status="error",
                    answer_type="local_result_command",
                    error_message="Local result command could not be resolved",
                )
                async with adapter.send_lock:
                    await websocket.send_json({
                        "type": "assistant_error",
                        "role": "assistant",
                        "content": outcome.message,
                        "detail": "No LLM or database query was used.",
                    })
                    await websocket.send_json({"type": "typing", "active": False})
                return

            snapshot = outcome.snapshot
            rows = list(snapshot.get("rows") or [])
            safe_rows = _sanitize_rows(rows)
            column_formats = dict(snapshot.get("column_formats") or {})
            display_formats = dict(
                (snapshot.get("metadata") or {}).get("display_formats") or {}
            )
            source_question = str(snapshot.get("question") or "Result")
            duration_ms = int(time.time() * 1000) - start_ms
            operation = str(outcome.operation or snapshot.get("operation") or "")

            render_question = source_question
            if operation == "keep_top":
                limit = int((snapshot.get("metadata") or {}).get("limit") or len(rows))
                order_column = str((snapshot.get("metadata") or {}).get("order_column") or "")
                suffix = f" by {order_column.replace('_', ' ').lower()}" if order_column else ""
                render_question = f"Top {limit} rows{suffix} from the prior result"
            elif operation == "sort":
                render_question = "Ranked rows from the prior result"
            elif operation == "exclude":
                render_question = "Filtered rows from the prior result"
            elif operation == "contribution":
                render_question = "Percentage contribution within the prior result"
            elif operation == "presentation":
                _presentation_override = str(
                    (snapshot.get("metadata") or {}).get("chart_type_override") or "auto"
                ).lower()
                render_question = (
                    "Best-fit chart from the prior result"
                    if _presentation_override == "auto"
                    else f"Prior result shown as {_presentation_override}"
                )

            chart_payload = None
            try:
                _presentation_override = str(
                    (snapshot.get("metadata") or {}).get("chart_type_override") or ""
                ).lower()
                if operation == "presentation" and _presentation_override == "table":
                    chart_type = None
                elif operation == "presentation" and _presentation_override in {
                    "area", "bar", "line", "pie", "donut", "scatter",
                }:
                    chart_type = _presentation_override
                else:
                    chart_type = (
                        "bar" if operation in {"keep_top", "sort", "contribution"}
                        else detect_chart_type(
                            safe_rows,
                            question=render_question,
                            column_formats=column_formats,
                        )
                    )
                if chart_type:
                    chart_payload = build_chart_payload(
                        safe_rows,
                        chart_type,
                        title="Updated result",
                        question=render_question,
                        column_formats=column_formats,
                    )
            except Exception as chart_exc:
                log.debug("Local result chart generation skipped: %s", chart_exc)

            from core.response_builder import build_assistant_response

            response = build_assistant_response(
                question=render_question,
                rows=safe_rows,
                sql=str(snapshot.get("sql") or ""),
                duration_ms=duration_ms,
                chart=chart_payload,
                data_source="governed session cache",
                display_context={"result_operation": operation},
                column_formats=column_formats,
                display_formats=display_formats,
                question_id=question_id,
            )
            response["result_command"] = {
                "operation": outcome.operation,
                "source_result_id": outcome.source_result_id,
                "derived_result_id": outcome.derived_result_id,
                "affected_count": outcome.affected_count,
                "rows_before": outcome.rows_before,
                "rows_after": outcome.rows_after,
                "llm_invoked": planner_used,
                "llm_input_mode": "metadata_only" if planner_used else "none",
                "rows_sent_to_llm": 0,
                "sample_values_sent_to_llm": 0,
                "database_queried": False,
                "execution_engine": "duckdb",
                "audit_request_id": question_id,
            }
            response.setdefault("trust", {}).update({
                "question_id": question_id,
                "parent_question_id": parent_question_id,
                "result_id": outcome.derived_result_id,
                "operation": (
                    "Metadata-planned cached-result transform"
                    if planner_used else "Deterministic cached-result transform"
                ),
                "llm_invoked": planner_used,
                "llm_input_mode": "metadata_only" if planner_used else "none",
                "rows_sent_to_llm": 0,
                "sample_values_sent_to_llm": 0,
                "source_sql_sent_to_llm": False,
                "database_queried": False,
                "execution_engine": "DuckDB (session-local)",
            })
            if planner_used:
                response["result_command"]["planner"] = planner_evidence
                response["trust"]["planner"] = planner_evidence

            adopt_cached_snapshot(adapter, snapshot, question_id=question_id)

            with llm_audit_scope(
                account_id=account_id,
                question="Transform the prior result locally",
                enabled=bool(client.get("enable_llm_audit")),
                request_id=question_id,
                question_id=question_id,
                component="result_command",
            ):
                from core.llm_audit import record_llm_blocked

                record_llm_blocked(
                    "result_command",
                    "Deterministic session-cache transform; "
                    f"operation={outcome.operation}; affected={outcome.affected_count}; "
                    f"rows_before={outcome.rows_before}; rows_after={outcome.rows_after}; "
                    "raw_values_logged=0; rows_sent_to_llm=0; sample_values_sent_to_llm=0; "
                    f"planner_used={str(planner_used).lower()}; database_queried=false.",
                )

            _trace_step(
                trace_id,
                "transform_cached_result",
                output_summary={
                    "operation": outcome.operation,
                    "affected_count": outcome.affected_count,
                    "rows_before": outcome.rows_before,
                    "rows_after": outcome.rows_after,
                    "raw_values_logged": 0,
                    "llm_invoked": planner_used,
                    "llm_input_mode": "metadata_only" if planner_used else "none",
                    "rows_sent_to_llm": 0,
                    "sample_values_sent_to_llm": 0,
                    "database_queried": False,
                },
            )
            _trace_finish(
                trace_id,
                status="success",
                answer_type="table",
                row_count=outcome.rows_after,
                duration_ms=duration_ms,
                final_answer_summary=(
                    "Cached result transformed locally after metadata-only planning"
                    if planner_used
                    else "Cached result transformed locally without an LLM or database call"
                ),
            )
            await adapter.send_assistant_response(
                adapter.make_event("Transform the prior result locally"),
                response,
            )
            async with adapter.send_lock:
                await websocket.send_json({"type": "typing", "active": False})
        except Exception as exc:
            log.exception("Deterministic result command failed: %s", exc)
            _trace_finish(
                trace_id,
                status="error",
                answer_type="local_result_command",
                error_message="Deterministic result transform failed",
            )
            async with adapter.send_lock:
                await websocket.send_json({
                    "type": "assistant_error",
                    "role": "assistant",
                    "content": "I could not update the cached result. Please run the business question again.",
                    "detail": "No result values were sent to an LLM.",
                })
                await websocket.send_json({"type": "typing", "active": False})

    async def _run_metadata_result_planner(
        text: str, table_hint: str = "", schema_hint: str = "",
    ) -> None:
        """Plan a cached-result transform from metadata, then execute locally."""
        session_id = getattr(adapter, "session_id", "") or ""
        source_result_id = getattr(adapter, "last_result_id", None)
        snapshot = result_cache.get_snapshot(session_id, source_result_id)
        if not snapshot:
            await _run_main_question(strip_result_context(text), table_hint, schema_hint)
            return

        provider, model, api_key, az_kwargs = resolve_provider(client, purpose="query")

        async def _complete_metadata_plan(**kwargs):
            return await llm_complete(
                provider=provider,
                model=model,
                api_key=api_key,
                **kwargs,
                **az_kwargs,
            )

        planner_request_id = make_llm_audit_request_id()
        with llm_audit_scope(
            account_id=account_id,
            question="Plan a cached-result analysis from metadata",
            enabled=bool(client.get("enable_llm_audit")),
            request_id=planner_request_id,
            question_id=getattr(adapter, "last_question_id", None) or "",
            component="result_metadata_planner",
        ):
            followup = await run_governed_result_followup(
                text,
                session_id,
                complete=_complete_metadata_plan,
                source_result_id=source_result_id,
            )

        if followup.executed and followup.command is not None:
            evidence = dict(followup.evidence)
            evidence["planner_request_id"] = planner_request_id
            await _run_local_result_command(
                text,
                followup.command,
                table_hint,
                schema_hint,
                planner_evidence=evidence,
                precomputed_outcome=followup.outcome,
            )
            return

        # A locally bound literal may be regulated. If planning cannot safely
        # compile it, do not let the original wording fall into another LLM.
        if followup.status == "blocked":
            async with adapter.send_lock:
                await websocket.send_json({
                    "type": "assistant_error",
                    "role": "assistant",
                    "content": (
                        "I could not safely apply that operation to the cached result. "
                        "Use an exact result column name or a row number."
                    ),
                    "detail": (
                        "The request was stopped locally. No cached rows, sample values, "
                        "or bound literals were sent to the LLM or source database."
                    ),
                })
                await websocket.send_json({"type": "typing", "active": False})
            return

        # An uncertain interpretation (low planner confidence, or an
        # ambiguous "which cached result" reference) must be confirmed, not
        # silently executed or silently discarded into an unrelated fresh
        # query. Reuses the same clarification_prompt message shape and
        # adapter method the metric-ambiguity flow already sends over this
        # socket (gateway/web_adapter.py::send_clarification_prompt) rather
        # than hand-building a new JSON shape the frontend may not recognize.
        if followup.status == "clarification" and followup.outcome is not None:
            _cf_options = list(followup.outcome.clarification_options or [])
            _cf_prompt = followup.outcome.clarification_prompt or "Which result value did you mean?"
            if _cf_options:
                _cf_event = adapter.make_event(text)
                await adapter.send_clarification_prompt(_cf_event, _cf_prompt, _cf_options)
            else:
                async with adapter.send_lock:
                    await websocket.send_json({"type": "message", "content": _cf_prompt})
            async with adapter.send_lock:
                await websocket.send_json({"type": "typing", "active": False})
            return

        # Non-value-bearing analytical requests may use the existing governed
        # source-query pipeline when the cached schema cannot answer them.
        # Symmetric with _run_local_result_command's fallback above: if the
        # planned command's outcome carries a retry_question (e.g. a
        # presentation/chart request that needs a fresh breakdown), use
        # that instead of the literal follow-up text.
        _retry_question = (followup.outcome.retry_question if followup.outcome else "") or ""
        await _run_main_question(
            _retry_question or strip_result_context(text), table_hint, schema_hint,
        )

    async def _run_report_builder_chat(text: str) -> None:
        """Turn a plain-language report request ("build me a report with net
        revenue and top customers, scheduled every Monday at 9am") into a
        real report + subscription, using the exact same store calls the
        checkbox report-creation form already uses (report_store.
        create_report/add_metric_to_report/create_subscription). Available
        to every portal user, not gated by role. The LLM sees only this
        user's own ACL-filtered metric names/descriptions
        (core/report_planner.py), never row data.
        """
        await websocket.send_json({"type": "typing", "active": True})
        try:
            all_metrics = store.list_metrics(account_id)
            allowed = store.get_allowed_tables(portal_user)
            if allowed is not None:
                all_metrics = [
                    m for m in all_metrics
                    if not m.get("base_table") or m["base_table"].upper() in allowed
                ]
            if not all_metrics:
                await websocket.send_json({
                    "type": "assistant_error",
                    "action": "define_report",
                    "content": (
                        "There are no metrics available to you yet -- ask your "
                        "admin to add some to the metric registry first."
                    ),
                })
                return

            provider, model, api_key, az_kwargs = resolve_provider(client, purpose="query")

            async def _complete_report_plan(**kwargs):
                return await llm_complete(
                    provider=provider, model=model, api_key=api_key, **kwargs, **az_kwargs,
                )

            from core.report_planner import parse_report_plan

            request_id = make_llm_audit_request_id()
            with llm_audit_scope(
                account_id=account_id,
                question="Plan a report from chat",
                enabled=bool(client.get("enable_llm_audit")),
                request_id=request_id,
                question_id=getattr(adapter, "last_question_id", None) or "",
                component="report_builder_chat",
            ):
                plan, plan_error = await parse_report_plan(text, all_metrics, _complete_report_plan)

            if plan is None:
                await websocket.send_json({
                    "type": "assistant_error",
                    "action": "define_report",
                    "content": (
                        plan_error
                        or "Could not build a report from that -- try naming the metrics explicitly."
                    ),
                })
                return

            from store import report_store

            report = report_store.create_report(
                account_id, plan.name, created_by_user_id=portal_user.get("id"),
            )
            metrics_by_id = {m["id"]: m for m in all_metrics}
            metric_names: list[str] = []
            for sort_order, metric_id in enumerate(plan.metric_ids):
                report_store.add_metric_to_report(report["id"], metric_id, sort_order)
                metric = metrics_by_id.get(metric_id)
                metric_names.append((metric or {}).get("name") or str(metric_id))

            bullets = [f"Metrics: {', '.join(metric_names)}"]
            schedule_line = ""
            if plan.cadence:
                report_store.create_subscription(
                    account_id, portal_user.get("id"), report["id"],
                    cadence=plan.cadence, day_of_week=plan.day_of_week, hour=plan.hour,
                )
                if plan.cadence == "weekly":
                    day_name = (
                        "Monday Tuesday Wednesday Thursday Friday Saturday Sunday"
                    ).split()[plan.day_of_week]
                    schedule_line = f"Delivered every {day_name} at {plan.hour:02d}:00."
                else:
                    schedule_line = f"Delivered daily at {plan.hour:02d}:00."
                bullets.append(schedule_line)

            metric_word = "metric" if len(metric_names) == 1 else "metrics"
            await websocket.send_json({
                "type": "assistant_analysis",
                "action": "define_report",
                "title": f'Report "{plan.name}" created',
                "body": (
                    f"I've created **{plan.name}** with {len(metric_names)} {metric_word}."
                    + (f" {schedule_line}" if schedule_line
                       else " No schedule was requested -- ask any time to add one.")
                ),
                "bullets": bullets,
                "report_id": report["id"],
            })
        except Exception as exc:
            log.warning("report builder chat failed for %s: %s", account_id, exc)
            await websocket.send_json({
                "type": "assistant_error",
                "action": "define_report",
                "content": "Could not build that report right now.",
            })
        finally:
            await websocket.send_json({"type": "typing", "active": False})

    async def _run_dashboard_chat(text: str) -> None:
        """Create and refine user-owned dashboards from governed results."""
        await websocket.send_json({"type": "typing", "active": True})
        try:
            user_id = int(portal_user.get("id") or 0)
            latest = (
                store.get_dashboard_for_view(selected_dashboard_id, user_id, account_id)
                if selected_dashboard_id
                else store.latest_dashboard_for_thread(
                    account_id, user_id, getattr(adapter, "thread_id", "default")
                )
            )
            if latest and selected_dashboard_read_only:
                await websocket.send_json({
                    "type": "assistant_error",
                    "action": "dashboard",
                    "content": (
                        "This is a published team dashboard owned by another user, so it is read-only. "
                        "You can ask questions about the data, but only the owner can change the artifact."
                    ),
                })
                return

            rollback_match = _DASHBOARD_ROLLBACK_INTENT_RE.search(text)
            if rollback_match:
                if not latest:
                    await websocket.send_json({"type": "assistant_error", "action": "dashboard", "content": "There is no dashboard here to restore yet."})
                    return
                target_version = int(rollback_match.group("version"))
                updated = store.rollback_dashboard(latest["id"], user_id, account_id, target_version)
                if not updated:
                    await websocket.send_json({"type": "assistant_error", "action": "dashboard", "content": f"Version {target_version} is not available for this dashboard."})
                    return
                await websocket.send_json({
                    "type": "assistant_dashboard", "action": "restored",
                    "title": f'Restored "{updated["name"]}" from version {target_version}',
                    "body": "The restore created a new draft checkpoint, so the full history is still available.",
                    "dashboard": {"id": updated["id"], "name": updated["name"], "status": updated["status"], "version": updated["version"], "url": f'/portal/dashboard?dashboard_id={updated["id"]}'},
                })
                return

            schedule_match = _DASHBOARD_SCHEDULE_INTENT_RE.search(text)
            if schedule_match:
                if not latest:
                    await websocket.send_json({"type": "assistant_error", "action": "dashboard", "content": "Create a dashboard before setting its refresh schedule."})
                    return
                schedule = schedule_match.group("schedule").lower().replace("manually", "manual")
                updated = store.update_dashboard_controls(
                    latest["id"], user_id, account_id,
                    refresh_schedule=schedule,
                    change_summary=f"Refresh set to {schedule}",
                )
                await websocket.send_json({
                    "type": "assistant_dashboard", "action": "schedule_changed",
                    "title": f'{updated["name"]} will refresh {schedule}',
                    "body": "Scheduled refreshes run as the dashboard owner through current ACL, semantic, validation, and compliance controls. Released rows are encrypted and expire at the policy cache TTL.",
                    "dashboard": {"id": updated["id"], "name": updated["name"], "status": updated["status"], "version": updated["version"], "url": f'/portal/dashboard?dashboard_id={updated["id"]}'},
                })
                return

            filter_match = _DASHBOARD_FILTER_INTENT_RE.search(text)
            if filter_match:
                if not latest:
                    await websocket.send_json({"type": "assistant_error", "action": "dashboard", "content": "Create a dashboard before adding filters."})
                    return
                field = filter_match.group("field").strip(" .")
                try:
                    filters = json.loads(latest.get("filters_json") or "[]")
                except (TypeError, ValueError):
                    filters = []
                if not any(str(item.get("field") or "").lower() == field.lower() for item in filters if isinstance(item, dict)):
                    filters.append({"field": field, "label": field, "type": "text"})
                updated = store.update_dashboard_controls(
                    latest["id"], user_id, account_id, filters=filters,
                    change_summary=f"Filter added: {field}",
                )
                await websocket.send_json({
                    "type": "assistant_dashboard", "action": "filter_added",
                    "title": f'Added a {field} filter to "{updated["name"]}"',
                    "body": "The control is applied only to dashboard sources that return a matching field.",
                    "dashboard": {"id": updated["id"], "name": updated["name"], "status": updated["status"], "version": updated["version"], "url": f'/portal/dashboard?dashboard_id={updated["id"]}'},
                })
                return

            tab_match = _DASHBOARD_TAB_INTENT_RE.search(text)
            if tab_match:
                if not latest:
                    await websocket.send_json({"type": "assistant_error", "action": "dashboard", "content": "Create a dashboard before adding tabs."})
                    return
                tab = tab_match.group("tab").strip(" .")
                try:
                    tabs = json.loads(latest.get("tabs_json") or '["Overview"]')
                except (TypeError, ValueError):
                    tabs = ["Overview"]
                if tab.lower() not in {str(value).lower() for value in tabs}:
                    tabs.append(tab)
                updated = store.update_dashboard_controls(
                    latest["id"], user_id, account_id, tabs=tabs,
                    change_summary=f"Tab added: {tab}",
                )
                await websocket.send_json({
                    "type": "assistant_dashboard", "action": "tab_added",
                    "title": f'Added the {tab} tab to "{updated["name"]}"',
                    "body": "Ask me to add a new visual to this dashboard and name the tab to place it there.",
                    "dashboard": {"id": updated["id"], "name": updated["name"], "status": updated["status"], "version": updated["version"], "url": f'/portal/dashboard?dashboard_id={updated["id"]}'},
                })
                return

            if _DASHBOARD_SHARE_INTENT_RE.search(text):
                if not latest:
                    await websocket.send_json({"type": "assistant_error", "action": "dashboard", "content": "Create a dashboard before sharing it."})
                    return
                updated = store.update_dashboard_controls(
                    latest["id"], user_id, account_id, visibility="team",
                    change_summary="Shared with team",
                )
                updated = store.publish_dashboard(updated["id"], user_id, account_id) or updated
                await websocket.send_json({
                    "type": "assistant_dashboard", "action": "shared",
                    "title": f'Published "{updated["name"]}" to your workspace team',
                    "body": "Workspace users can view and filter it under their own current data access. Only the owner can edit or restore the artifact.",
                    "dashboard": {"id": updated["id"], "name": updated["name"], "status": updated["status"], "version": updated["version"], "url": f'/portal/dashboard?dashboard_id={updated["id"]}'},
                })
                return

            rename_match = _DASHBOARD_RENAME_INTENT_RE.search(text)
            if rename_match:
                if not latest:
                    await websocket.send_json({
                        "type": "assistant_error",
                        "action": "dashboard",
                        "content": "There is no dashboard in this thread to rename yet.",
                    })
                    return
                updated = store.rename_dashboard(
                    latest["id"], user_id, account_id, rename_match.group("name")
                )
                await websocket.send_json({
                    "type": "assistant_dashboard",
                    "action": "renamed",
                    "title": f'Dashboard renamed to "{updated["name"]}"',
                    "dashboard": {
                        "id": updated["id"], "name": updated["name"],
                        "status": updated["status"], "version": updated["version"],
                        "url": f'/portal/dashboard?dashboard_id={updated["id"]}',
                    },
                })
                return

            if _DASHBOARD_PUBLISH_INTENT_RE.search(text):
                if not latest:
                    await websocket.send_json({
                        "type": "assistant_error",
                        "action": "dashboard",
                        "content": "There is no dashboard in this thread to publish yet.",
                    })
                    return
                updated = store.publish_dashboard(latest["id"], user_id, account_id)
                await websocket.send_json({
                    "type": "assistant_dashboard",
                    "action": "published",
                    "title": f'Dashboard "{updated["name"]}" published',
                    "dashboard": {
                        "id": updated["id"], "name": updated["name"],
                        "status": updated["status"], "version": updated["version"],
                        "url": f'/portal/dashboard?dashboard_id={updated["id"]}',
                    },
                })
                return

            chart_type_match = _DASHBOARD_CHART_TYPE_RE.search(text)
            if chart_type_match:
                if not latest:
                    await websocket.send_json({
                        "type": "assistant_error", "action": "dashboard",
                        "content": "There is no dashboard in this thread to update yet.",
                    })
                    return
                charts = store.list_dashboard_charts(latest["id"], user_id)
                if not charts:
                    await websocket.send_json({
                        "type": "assistant_error", "action": "dashboard",
                        "content": "That dashboard does not have a visual to update yet.",
                    })
                    return
                chart = charts[-1]
                chart_type = chart_type_match.group("chart_type").lower()
                store.update_dashboard_chart(
                    latest["id"], chart["id"], user_id, account_id,
                    chart_type=chart_type,
                )
                latest = store.get_dashboard(
                    latest["id"], user_id, account_id
                ) or latest
                await websocket.send_json({
                    "type": "assistant_dashboard", "action": "chart_type_changed",
                    "title": f'Changed the latest visual in "{latest["name"]}" to {chart_type}',
                    "dashboard": {
                        "id": latest["id"], "name": latest["name"],
                        "status": "draft", "version": latest["version"],
                        "url": f'/portal/dashboard?dashboard_id={latest["id"]}',
                    },
                })
                return

            add_query_match = _DASHBOARD_ADD_QUERY_RE.search(text)
            if add_query_match:
                if not latest:
                    await websocket.send_json({
                        "type": "assistant_error", "action": "dashboard",
                        "content": "Create a dashboard first, then I can add new governed visuals to it.",
                    })
                    return
                question = add_query_match.group("question").strip(" .,:;-")
                adapter.queue_dashboard(latest["name"], dashboard_id=int(latest["id"]))
                await websocket.send_json({"type": "typing", "active": False})
                await _run_main_question(question, "", "")
                return

            if _DASHBOARD_ADD_INTENT_RE.search(text):
                if not selected_dashboard_id:
                    response = getattr(adapter, "last_response_payload", None) or {}
                    chart = response.get("chart") if isinstance(response.get("chart"), dict) else {}
                    token = str(response.get("pin_token") or chart.get("pin_token") or "")
                    if not token:
                        await websocket.send_json({
                            "type": "assistant_error",
                            "action": "dashboard",
                            "content": "Run the result again so I can open the dashboard chooser for it.",
                        })
                        return
                    kpi = response.get("kpi") if isinstance(response.get("kpi"), dict) else {}
                    await websocket.send_json({
                        "type": "dashboard_selection_required",
                        "pin_token": token,
                        "title": str(
                            chart.get("title") or kpi.get("label")
                            or response.get("question") or "Dashboard visual"
                        )[:120],
                        "chart_type": str(
                            chart.get("chart_type")
                            or response.get("dashboard_item_type")
                            or ("kpi" if kpi else "table")
                        ),
                        "color_palette": str(chart.get("color_palette") or "default"),
                    })
                    return
                if not latest:
                    await websocket.send_json({
                        "type": "assistant_error",
                        "action": "dashboard",
                        "content": (
                            "There is no dashboard in this thread yet. Say "
                            '"create a dashboard from this result" first.'
                        ),
                    })
                    return
                artifact = adapter.materialize_dashboard(
                    name=latest["name"], dashboard_id=int(latest["id"])
                )
                await websocket.send_json({
                    "type": "assistant_dashboard",
                    "action": "item_added",
                    "title": f'Added this result to "{artifact["name"]}"',
                    "dashboard": artifact,
                })
                return

            tail = _dashboard_request_tail(text)
            has_result = bool(
                getattr(adapter, "last_result", None)
                and (adapter.last_result or {}).get("rows")
            )
            name = _dashboard_name(tail or (adapter.last_result or {}).get("question", ""))
            if tail:
                from core.dashboard_planner import looks_like_multi_widget_request

                if looks_like_multi_widget_request(tail):
                    all_metrics = store.list_metrics(account_id)
                    allowed = store.get_allowed_tables(portal_user)
                    if allowed is not None:
                        all_metrics = [
                            metric for metric in all_metrics
                            if not metric.get("base_table") or str(metric["base_table"]).upper() in allowed
                        ]
                    provider, model, api_key, az_kwargs = resolve_provider(client, purpose="query")

                    async def _complete_dashboard_plan(**kwargs):
                        return await llm_complete(
                            provider=provider, model=model, api_key=api_key,
                            **kwargs, **az_kwargs,
                        )

                    from core.dashboard_planner import parse_dashboard_plan

                    request_id = make_llm_audit_request_id()
                    with llm_audit_scope(
                        account_id=account_id,
                        question="Plan dashboard work from chat",
                        enabled=bool(client.get("enable_llm_audit")),
                        request_id=request_id,
                        question_id=getattr(adapter, "last_question_id", None) or "",
                        component="dashboard_builder_chat",
                    ):
                        plan, plan_error = await parse_dashboard_plan(text, all_metrics, _complete_dashboard_plan)
                    if plan is None or plan.confidence < 0.60:
                        await adapter.send_clarification_prompt(
                            adapter.make_event(text),
                            plan_error or "I can build that dashboard, but I need the exact visuals, time range, and grouping you want before I run several queries.",
                            [],
                        )
                        return
                    dashboard = store.create_dashboard(
                        account_id, user_id, adapter.thread_id, plan.name,
                        description="Created by QueryBot's governed dashboard work planner.",
                    )
                    dashboard = store.update_dashboard_controls(
                        dashboard["id"], user_id, account_id,
                        visibility=plan.visibility,
                        refresh_schedule=plan.refresh_schedule,
                        tabs=list(plan.tabs),
                        change_summary="Dashboard work plan created",
                    ) or dashboard
                    await websocket.send_json({
                        "type": "assistant_analysis", "action": "dashboard_plan",
                        "title": f'Building "{dashboard["name"]}"',
                        "body": f"I’ll run {len(plan.widgets)} governed data tasks and assemble the successful results in the artifact pane.",
                        "bullets": [f"{index + 1}. {widget.title or widget.question}" for index, widget in enumerate(plan.widgets)],
                    })
                    completed = 0
                    for index, widget in enumerate(plan.widgets):
                        await websocket.send_json({
                            "type": "status", "stage": "dashboard_work",
                            "label": f"Building visual {index + 1} of {len(plan.widgets)}",
                            "detail": widget.title or widget.question,
                        })
                        adapter.queue_dashboard(
                            dashboard["name"], dashboard_id=int(dashboard["id"]),
                            tab=widget.tab, chart_type=widget.chart_type, title=widget.title,
                        )
                        await _run_main_question(widget.question, "", "")
                        if adapter.pending_dashboard is None:
                            completed += 1
                        else:
                            adapter.pending_dashboard = None
                    final_dashboard = store.get_dashboard(dashboard["id"], user_id, account_id) or dashboard
                    await websocket.send_json({
                        "type": "assistant_dashboard", "action": "created",
                        "title": f'Built {completed} of {len(plan.widgets)} visuals for "{final_dashboard["name"]}"',
                        "body": "Open the artifact to review the live charts, data sources, controls, and revision history.",
                        "dashboard": {"id": final_dashboard["id"], "name": final_dashboard["name"], "status": final_dashboard["status"], "version": final_dashboard["version"], "url": f'/portal/dashboard?dashboard_id={final_dashboard["id"]}'},
                    })
                    return
                # The request includes a business question. Run it through the
                # standard guarded pipeline, then materialize only its success.
                adapter.queue_dashboard(name)
                await websocket.send_json({"type": "typing", "active": False})
                await _run_main_question(tail, "", "")
                return
            if not has_result:
                await websocket.send_json({
                    "type": "message",
                    "content": (
                        "What should the dashboard track? For example, say "
                        '"create a dashboard showing monthly revenue by region".'
                    ),
                })
                return
            artifact = adapter.materialize_dashboard(name=name)
            await websocket.send_json({
                "type": "assistant_dashboard",
                "action": "created",
                "title": f'Dashboard "{artifact["name"]}" created',
                "dashboard": artifact,
            })
        except ValueError as exc:
            await websocket.send_json({
                "type": "assistant_error",
                "action": "dashboard",
                "content": str(exc),
            })
        except Exception as exc:
            log.exception("dashboard chat failed for %s: %s", account_id, exc)
            await websocket.send_json({
                "type": "assistant_error",
                "action": "dashboard",
                "content": "I could not update that dashboard right now.",
            })
        finally:
            await websocket.send_json({"type": "typing", "active": False})

    async def _run_reconcile_chat(snapshot: dict) -> None:
        """"the real number is X, why is yours Y" -- QueryBot cannot see the
        methodology behind an externally-known number, so this must NOT
        promise to explain the gap. It restates its own answer's exact
        definition (the real SQL that ran -- not vague language) and offers
        1-2 concrete, clickable hypotheses that re-run via the normal
        pipeline, exactly like every other correction/fallback path in this
        file. A guided-comparison tool, not automated root-cause magic.
        """
        question = str(snapshot.get("question") or "this result")
        sql = str(snapshot.get("sql") or "")
        rows = snapshot.get("rows") or []

        bullets = [f"Question asked: {question}", f"Rows returned: {len(rows)}"]
        if len(rows) == 1 and len(rows[0]) == 1:
            from core.response_builder import _safe_cell
            value = next(iter(rows[0].values()))
            bullets.insert(0, f"My value: {_safe_cell(value)}")

        await websocket.send_json({
            "type": "assistant_analysis",
            "action": "reconcile",
            "title": "Here's exactly what I ran",
            "body": (
                "I can't see how your number was calculated, so I can't explain the "
                "gap directly -- but here's my exact definition. Try one of these to "
                "see if it closes the difference:"
            ),
            "bullets": bullets,
            "secondary": sql,
            "follow_up_suggestions": [
                f"{question}, excluding internal or administrative records",
                f"{question}, using last calendar month instead",
            ],
        })
        await websocket.send_json({"type": "typing", "active": False})

    async def _run_analysis_work(text: str) -> None:
        """Run auditable prebuilt or governed-Python work over released rows."""
        cached = getattr(adapter, "last_result", None) or {}
        rows = list(cached.get("rows") or [])
        if not rows:
            await websocket.send_json({
                "type": "assistant_error",
                "action": "analyze_result",
                "content": "Run a data question first, then ask me to analyze that result.",
            })
            await websocket.send_json({"type": "typing", "active": False})
            return

        from core.analysis_code_planner import (
            extract_user_python, parse_python_analysis_plan,
        )
        from core.analysis_sandbox import (
            plan_analysis_operations, run_governed_python_analysis,
            run_isolated_analysis, validate_python_analysis,
        )
        from core.response_builder import build_assistant_response

        user_code = extract_user_python(text)
        custom_python = bool(user_code or _CUSTOM_PYTHON_INTENT_RE.search(text))
        python_code = ""
        code_source = ""
        analysis_title = "Analysis work"
        plan_explanation = ""
        planner_used = False
        validation = None

        if custom_python:
            if not int(client.get("enable_python_analysis") or 0):
                await websocket.send_json({
                    "type": "assistant_error",
                    "action": "python_analysis",
                    "content": (
                        "Governed Python analysis is disabled for this workspace. "
                        "An administrator can enable it in Client settings → Agent Analysis."
                    ),
                })
                await websocket.send_json({"type": "typing", "active": False})
                return
            if user_code:
                if not int(client.get("allow_user_python") or 0):
                    await websocket.send_json({
                        "type": "assistant_error",
                        "action": "python_analysis",
                        "content": (
                            "This workspace allows governed Python plans but not pasted source. "
                            "Ask for the calculation in plain English, or have an administrator "
                            "enable user-submitted Python."
                        ),
                    })
                    await websocket.send_json({"type": "typing", "active": False})
                    return
                python_code = user_code
                code_source = "user_submitted"
                analysis_title = "Custom Python analysis"
            else:
                provider, model, api_key, az_kwargs = resolve_provider(client, purpose="query")

                async def _complete_python_plan(**kwargs):
                    return await llm_complete(
                        provider=provider, model=model, api_key=api_key,
                        **kwargs, **az_kwargs,
                    )

                with llm_audit_scope(
                    account_id=account_id,
                    question="Plan governed Python analysis from result metadata",
                    enabled=bool(client.get("enable_llm_audit")),
                    request_id=make_llm_audit_request_id(),
                    question_id=getattr(adapter, "last_question_id", None) or "",
                    component="analysis_code_planner",
                ):
                    plan, plan_error = await parse_python_analysis_plan(
                        text, rows, _complete_python_plan,
                    )
                if plan is None or plan.confidence < 0.65:
                    await websocket.send_json({
                        "type": "assistant_error",
                        "action": "python_analysis",
                        "content": plan_error or (
                            "I need a more precise calculation and output shape before I run Python."
                        ),
                    })
                    await websocket.send_json({"type": "typing", "active": False})
                    return
                python_code = plan.code
                code_source = "metadata_only_planner"
                analysis_title = plan.title
                plan_explanation = plan.explanation
                planner_used = True
            validation = validate_python_analysis(python_code)
            operations = ["python"]
        else:
            operations = plan_analysis_operations(text, rows)

        agent_run = None
        subtasks: list[dict] = []
        results: list[tuple[str, object]] = []
        try:
            agent_run = AgentRunSession.start(
                account_id=account_id,
                portal_user=portal_user,
                external_thread_id=adapter.thread_id,
                objective=text,
                max_tool_calls=1,
                initial_tool="analyze_result",
                initial_label="Analyzing the governed result",
                initial_detail=(
                    f"Running {len(operations)} bounded child task(s) in isolated workers"
                ),
                initial_metadata={
                    "database_queried": False,
                    "rows_sent_to_llm": 0,
                    "custom_python": custom_python,
                    "code_source": code_source,
                    "code_hash": validation.code_hash if validation else "",
                    "source_result_id": getattr(adapter, "last_result_id", None) or "",
                    "operations": operations,
                },
            )
            await adapter.send_agent_event(agent_run.event("agent_run_started"))
            for operation in operations:
                subtasks.append(store.create_agent_subtask(
                    parent_run_id=agent_run.run_id,
                    account_id=account_id,
                    portal_user_id=int(portal_user["id"]),
                    objective_sanitized=f"{operation.title()} analysis of released result",
                    tool_name=f"analysis_{operation}",
                    metadata={
                        "source_result_id": getattr(adapter, "last_result_id", None) or "",
                        "input_row_count": min(len(rows), 5000),
                        "database_queried": False,
                        "rows_sent_to_llm": 0,
                        "code_source": code_source,
                        "code_hash": validation.code_hash if validation else "",
                        "ast_nodes": validation.ast_nodes if validation else 0,
                        "helper_calls": list(validation.helper_calls) if validation else [],
                    },
                ))
            await adapter.send_agent_event(agent_run.record_stage(
                "analysis_work",
                "Running isolated analysis tasks",
                ", ".join(operation.title() for operation in operations),
            ))

            outcomes = await asyncio.gather(*[
                asyncio.to_thread(run_governed_python_analysis, rows, python_code)
                if operation == "python"
                else asyncio.to_thread(run_isolated_analysis, rows, operation)
                for operation in operations
            ], return_exceptions=True)
            for operation, subtask, outcome in zip(operations, subtasks, outcomes):
                if isinstance(outcome, BaseException):
                    store.finish_agent_subtask(
                        subtask_id=subtask["id"], parent_run_id=agent_run.run_id,
                        account_id=account_id, portal_user_id=int(portal_user["id"]),
                        status="failed", result_summary=str(outcome),
                        metadata={
                            "operation": operation,
                            "error_type": type(outcome).__name__,
                            "input_row_count": min(len(rows), 5000),
                            "database_queried": False,
                            "code_source": code_source,
                            "code_hash": validation.code_hash if validation else "",
                            "ast_nodes": validation.ast_nodes if validation else 0,
                            "rows_sent_to_llm": 0,
                        },
                    )
                    results.append((operation, outcome))
                    continue
                store.finish_agent_subtask(
                    subtask_id=subtask["id"], parent_run_id=agent_run.run_id,
                    account_id=account_id, portal_user_id=int(portal_user["id"]),
                    status="completed", result_summary=outcome.summary,
                    metadata={
                        "operation": operation,
                        "source_result_id": getattr(adapter, "last_result_id", None) or "",
                        "input_row_count": min(len(rows), 5000),
                        "database_queried": False,
                        "rows_sent_to_llm": 0,
                        "code_source": code_source,
                        **outcome.metadata,
                    },
                )
                results.append((operation, outcome))

            completed = [(operation, result) for operation, result in results
                         if not isinstance(result, BaseException)]
            failed = [(operation, result) for operation, result in results
                      if isinstance(result, BaseException)]
            if not completed:
                raise RuntimeError("All isolated analysis tasks failed.")

            bullets = [result.summary for _, result in completed]
            if planner_used and plan_explanation:
                bullets.insert(0, plan_explanation)
            if failed:
                bullets.append(
                    "Could not complete: " + ", ".join(operation for operation, _ in failed)
                )
            await websocket.send_json({
                "type": "assistant_analysis",
                "action": "python_analysis" if custom_python else "analyze_result",
                "title": f"{analysis_title} completed",
                "body": (
                    "I analyzed only the governed rows already returned to this conversation. "
                    + (
                        "A metadata-only planner produced the validated calculation; zero result "
                        "rows or sample values were sent to the model."
                        if planner_used else
                        "No database query or model call was made for these calculations."
                    )
                ),
                "bullets": bullets,
                "secondary": (
                    f"{len(completed)} of {len(operations)} child tasks completed in isolated, "
                    "time-bounded workers."
                ),
                "result_scope": {
                    "badge": "Returned result only",
                    "note": f"Based on {min(len(rows), 5000)} released rows.",
                },
            })

            # Prefer the most decision-useful non-empty derived table in the
            # artifact pane; the chat keeps the concise explanation.
            priority = {"python": 0, "correlation": 1, "outliers": 2, "trend": 3, "profile": 4}
            candidates = sorted(completed, key=lambda item: priority.get(item[0], 9))
            operation, chosen = next(
                ((op, result) for op, result in candidates if result.rows), candidates[0]
            )
            chart_type = detect_chart_type(chosen.rows, f"{operation} analysis")
            chart = build_chart_payload(
                chosen.rows, chart_type,
                title=analysis_title if operation == "python" else f"{operation.title()} analysis",
                question=text,
            ) if chart_type else None
            response = build_assistant_response(
                question=(
                    analysis_title
                    if operation == "python"
                    else f"{operation.title()} analysis of the returned result"
                ),
                rows=chosen.rows,
                sql="",
                duration_ms=0,
                chart=chart,
                data_source="Isolated Python worker",
                display_context={"result_operation": "analysis_work"},
            )
            response["answer"] = _analysis_artifact_answer(
                operation,
                chosen.summary,
                len(chosen.rows),
                int(chosen.metadata.get("input_rows") or len(rows)),
            )
            # The worker summary above is authoritative. Suppress the generic
            # table insight, which reasons about the analysis-output schema as
            # if it were a business ranking.
            response["insight_summary"] = ""
            response["decision_signal"] = {}
            response["trust"].update({
                "operation": (
                    "Governed Python analysis" if custom_python else "Bounded result analysis"
                ),
                "database_queried": False,
                "rows_sent_to_llm": 0,
                "source_result_id": getattr(adapter, "last_result_id", None) or "",
                "worker_isolation": "spawned process with hard timeout",
                "code_source": code_source,
                "code_hash": validation.code_hash if validation else "",
                "ast_nodes": validation.ast_nodes if validation else 0,
                "planner_input": "column metadata only" if planner_used else "none",
                "child_tasks": [
                    {
                        "id": task["id"],
                        "operation": operation_name,
                        "status": "failed" if isinstance(result, BaseException) else "completed",
                    }
                    for task, (operation_name, result) in zip(subtasks, results)
                ],
            })
            response["analysis_artifact"] = True
            await adapter.send_assistant_response(adapter.make_event(text), response)
            if agent_run:
                agent_run.record_assistant_message(
                    "; ".join(bullets),
                    message_type="analysis_summary",
                    metadata={
                        "source_result_id": getattr(adapter, "last_result_id", None) or "",
                        "child_task_count": len(subtasks),
                    },
                )
                agent_run.complete_if_running()
                await adapter.send_agent_event(agent_run.event("agent_run_finished"))
        except asyncio.CancelledError:
            if agent_run:
                agent_run.cancel()
                await adapter.send_agent_event(agent_run.event("agent_run_finished"))
            raise
        except Exception as exc:
            log.warning("isolated analysis work failed: %s", exc)
            if agent_run:
                agent_run.fail(exc)
                await adapter.send_agent_event(agent_run.event("agent_run_finished"))
            await websocket.send_json({
                "type": "assistant_error",
                "action": "python_analysis" if custom_python else "analyze_result",
                "content": "I could not complete the governed result analysis.",
                "detail": str(exc)[:240],
            })
        finally:
            await websocket.send_json({"type": "typing", "active": False})

    try:
        while True:
            data = await websocket.receive_json()
            if not isinstance(data, dict):
                await websocket.send_json({
                    "type": "assistant_error",
                    "role": "assistant",
                    "content": "I could not read that message. Please send the question again.",
                })
                continue

            msg_type = _ws_text_value(data.get("type"), "type", "value")
            action = _ws_text_value(data.get("action"), "action", "value")
            context = data.get("context")
            if not isinstance(context, dict):
                context = {}
            text = _ws_text_value(
                data.get("text"), "text", "question", "value", "label"
            )
            # table_hint: FQN hint from clicked suggested question
            table_hint = _ws_text_value(
                data.get("table_hint"), "fqn", "table_hint", "table", "value"
            )
            # schema_hint: schema name selected by the user in the portal UI
            # e.g. "HR" or "PHARMACY" — filters allowed_tables to that schema only
            schema_hint = _ws_text_value(
                data.get("schema_hint"), "schema_hint", "schema", "name", "value"
            ).upper()

            # ── history_sync: browser restores prior-session history ──────────
            if msg_type == "history_sync":
                incoming = data.get("history")
                if isinstance(incoming, list):
                    load_fn = getattr(adapter, "load_history", None)
                    if callable(load_fn):
                        load_fn(incoming)
                        log.debug("history_sync: loaded %d turn(s) from client", len(incoming))
                continue

            # ── cancel: user clicked Stop on an in-flight question ───────────
            # Cancelling the task cleanly interrupts it at its next `await`
            # (LLM calls are plain asyncio awaits, so this stops SQL
            # generation immediately). DB execution runs in a thread pool
            # (asyncio.wait_for around run_in_executor) — cancelling only
            # stops the pipeline from waiting on it; the query may keep
            # running server-side to completion in the background, but no
            # result is ever sent since nothing is listening anymore.
            if msg_type == "cancel":
                if current_query_task and not current_query_task.done():
                    current_query_task.cancel()
                    try:
                        await current_query_task
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        log.debug("cancel: task raised during shutdown: %s", e)
                    async with adapter.send_lock:
                        # "system" (not assistant_error) — the frontend
                        # prefixes assistant_error with a warning triangle,
                        # which reads like something went wrong. Stopping is
                        # what the user asked for; it should look calm, like
                        # the existing "Connected as ..." system notices.
                        await websocket.send_json({
                            "type": "system",
                            "content": "Query stopped.",
                        })
                        await websocket.send_json({"type": "typing", "active": False})
                continue

            # ── result_chat: inline card chat — always DuckDB, no routing ────
            # Sent by the inline mini-chat panel inside each result card.
            # result_id tags the response so the browser renders it inside
            # the correct card rather than the main thread.
            # The browser submits only opaque row handles. Filtering,
            # summaries, and chart reconstruction all happen in-process;
            # neither selected values nor remaining rows are sent to an LLM.
            if msg_type == "result_exclusion":
                result_id = _ws_text_value(data.get("result_id"), "id", "value")
                raw_tokens = data.get("row_tokens")
                row_tokens = (
                    [token for token in raw_tokens if isinstance(token, str)][:200]
                    if isinstance(raw_tokens, list)
                    else []
                )
                start_ms = int(time.time() * 1000)
                question_id = make_llm_audit_request_id()
                parent_question_id = getattr(adapter, "last_question_id", "") or ""
                trace_id = _trace_create(
                    account_id=account_id,
                    question_id=question_id,
                    parent_question_id=parent_question_id,
                    question="Exclude selected rows from the returned result",
                    portal_user_id=portal_user.get("id") if portal_user else None,
                    platform_user_id=zoom_user_id or "",
                    session_id=getattr(adapter, "session_id", "") or "",
                    request_source="portal",
                    route="deterministic_result_exclusion",
                )
                _trace_step(
                    trace_id,
                    "receive_exclusion",
                    output_summary={
                        "opaque_handles": len(row_tokens),
                        "raw_values_received": 0,
                    },
                )
                try:
                    session_id = getattr(adapter, "session_id", "") or ""
                    excluded = result_cache.exclude_rows(
                        session_id,
                        row_tokens,
                        result_id=result_id or None,
                    )
                    excluded_snapshot = result_cache.get_snapshot(
                        session_id,
                        result_id or None,
                    )
                    adopt_cached_snapshot(
                        adapter,
                        excluded_snapshot,
                        question_id=question_id,
                    )
                    safe_rows = _sanitize_rows(excluded["rows"])
                    duration_ms = int(time.time() * 1000) - start_ms

                    chart_payload = None
                    try:
                        chart_type = detect_chart_type(
                            safe_rows,
                            question=excluded.get("question") or "Filtered result",
                            column_formats=excluded.get("column_formats") or {},
                        )
                        if chart_type:
                            chart_payload = build_chart_payload(
                                safe_rows,
                                chart_type,
                                title="Filtered result",
                                question=excluded.get("question") or "Filtered result",
                                column_formats=excluded.get("column_formats") or {},
                            )
                    except Exception as chart_exc:
                        log.debug("Filtered result chart generation skipped: %s", chart_exc)

                    from core.response_builder import build_assistant_response

                    response = build_assistant_response(
                        question=(excluded.get("question") or "Result") + " (selected rows excluded)",
                        rows=safe_rows,
                        sql=excluded.get("sql") or "",
                        duration_ms=duration_ms,
                        chart=chart_payload,
                        data_source="governed result cache",
                        column_formats=excluded.get("column_formats") or {},
                        question_id=question_id,
                    )
                    response["result_exclusion"] = {
                        "result_id": result_id,
                        "excluded_count": excluded["excluded_count"],
                        "rows_before": excluded["rows_before"],
                        "rows_after": excluded["rows_after"],
                        "llm_invoked": False,
                        "rows_sent_to_llm": 0,
                        "database_queried": False,
                        "audit_request_id": question_id,
                    }
                    response.setdefault("trust", {}).update({
                        "question_id": question_id,
                        "parent_question_id": parent_question_id,
                        "operation": "Deterministic result exclusion",
                        "llm_invoked": False,
                        "rows_sent_to_llm": 0,
                        "database_queried": False,
                    })

                    with llm_audit_scope(
                        account_id=account_id,
                        question="Exclude selected rows from the returned result",
                        enabled=bool(client.get("enable_llm_audit")),
                        request_id=question_id,
                        question_id=question_id,
                        component="result_exclusion",
                    ):
                        from core.llm_audit import record_llm_blocked

                        record_llm_blocked(
                            "result_exclusion",
                            "Deterministic cache exclusion; "
                            f"selected={excluded['excluded_count']}; "
                            f"rows_before={excluded['rows_before']}; "
                            f"rows_after={excluded['rows_after']}; "
                            "rows_sent_to_llm=0; database_queried=false.",
                        )

                    _trace_step(
                        trace_id,
                        "exclude_cached_rows",
                        output_summary={
                            "excluded_count": excluded["excluded_count"],
                            "rows_before": excluded["rows_before"],
                            "rows_after": excluded["rows_after"],
                            "llm_invoked": False,
                            "rows_sent_to_llm": 0,
                            "database_queried": False,
                        },
                    )
                    _trace_finish(
                        trace_id,
                        status="success",
                        answer_type="table",
                        row_count=excluded["rows_after"],
                        duration_ms=duration_ms,
                        final_answer_summary="Governed cached result exclusion completed without an LLM or database call",
                    )
                    await adapter.send_assistant_response(
                        adapter.make_event("Exclude selected rows"),
                        response,
                    )
                except (ValueError, LookupError) as exc:
                    _trace_finish(
                        trace_id,
                        status="error",
                        answer_type="error",
                        error_message=str(exc),
                    )
                    await websocket.send_json({
                        "type": "result_exclusion_error",
                        "result_id": result_id,
                        "content": str(exc),
                    })
                except Exception as exc:
                    log.exception("Governed result exclusion failed: %s", exc)
                    _trace_finish(
                        trace_id,
                        status="error",
                        answer_type="error",
                        error_message=str(exc),
                    )
                    await websocket.send_json({
                        "type": "result_exclusion_error",
                        "result_id": result_id,
                        "content": "The filtered view could not be created. Please retry.",
                    })
                continue

            if msg_type == "result_chat":
                rc_question = _ws_text_value(
                    data.get("question"), "question", "text", "value", "label"
                )
                rc_result_id = _ws_text_value(data.get("result_id"), "id", "value")
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

                    _rc_source_id = (
                        rc_result_id
                        if result_cache.has_result(_sid, result_id=rc_result_id)
                        else getattr(adapter, "last_result_id", None)
                    )
                    _rc_snapshot = result_cache.get_snapshot(_sid, _rc_source_id)
                    _rc_schema = list(_rc_snapshot.get("schema") or [])
                    _rc_stats: dict = {}
                    _rc_formats = dict(_rc_snapshot.get("column_formats") or {})
                    _rc_currency = [
                        name for name, value in _rc_formats.items()
                        if str(value).lower() == "currency"
                    ]
                    _rc_db_cfg      = get_client_db(account_id) or {}
                    _rc_history     = _result_chat_histories.get(rc_result_id, [])

                    _rc_provider, _rc_model, _rc_key, _rc_az = resolve_provider(
                        client, purpose="query"
                    )

                    async def _complete_result_plan(**kwargs):
                        return await llm_complete(
                            provider=_rc_provider,
                            model=_rc_model,
                            api_key=_rc_key,
                            **kwargs,
                            **_rc_az,
                        )

                    _rc_planner_request_id = make_llm_audit_request_id()
                    with llm_audit_scope(
                        account_id=account_id,
                        question="Plan a cached-result analysis from metadata",
                        enabled=bool(client.get("enable_llm_audit")),
                        request_id=_rc_planner_request_id,
                        question_id=_rc_question_id,
                        component="result_metadata_planner",
                    ):
                        _rc_followup = await run_governed_result_followup(
                            rc_question,
                            _sid,
                            complete=_complete_result_plan,
                            source_result_id=_rc_source_id,
                        )

                    _trace_step(
                        _rc_trace_id,
                        "governed_result_followup",
                        output_summary={
                            "status": _rc_followup.status,
                            **_rc_followup.evidence,
                        },
                        status="success" if _rc_followup.executed else "error",
                    )

                    if _rc_followup.executed and _rc_followup.outcome is not None:
                        _rc_outcome = _rc_followup.outcome
                        _derived = _rc_outcome.snapshot
                        _rc_rows = _sanitize_rows(list(_derived.get("rows") or []))
                        _rc_sql = str(_derived.get("sql") or "")
                        _rc_formats = dict(_derived.get("column_formats") or {})
                        _rc_display_formats = dict(
                            (_derived.get("metadata") or {}).get("display_formats") or {}
                        )
                        _rc_currency = [
                            name for name, value in _rc_formats.items()
                            if str(value).lower() == "currency"
                        ]
                        _rc_dur_ms = int(time.time() * 1000) - _rc_start_ms

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

                        adopt_cached_snapshot(
                            adapter,
                            _derived,
                            question_id=_rc_question_id,
                        )

                        _rc_display_payload = build_assistant_response(
                            question=rc_question,
                            rows=_rc_rows,
                            sql=_rc_sql,
                            duration_ms=_rc_dur_ms,
                            chart=_rc_chart,
                            data_source="governed_cache",
                            column_formats=_rc_formats,
                            display_formats=_rc_display_formats,
                            display_context={"result_operation": _rc_outcome.operation},
                            question_id=_rc_question_id,
                        )
                        _rc_display_data = dict(_rc_display_payload.get("data") or {})

                        _rc_history.append({
                            "question": rc_question,
                            "row_count": len(_rc_rows),
                            "operation": _rc_outcome.operation,
                        })
                        _result_chat_histories[rc_result_id] = _rc_history[-5:]
                        _log_q(
                            account_id, rc_question, _rc_sql, len(_rc_rows), True, "",
                            "governed_result_cache", "duckdb", 0, 0, _rc_dur_ms,
                            portal_user_id=_rc_pu_id, zoom_user_id=zoom_user_id,
                            question_id=_rc_question_id,
                            parent_question_id=_rc_parent_qid,
                        )
                        await websocket.send_json({
                            "type": "result_chat_response",
                            "result_id": rc_result_id,
                            "derived_result_id": _rc_outcome.derived_result_id,
                            "parent_result_id": _rc_outcome.source_result_id,
                            "question": rc_question,
                            "sql": _rc_sql,
                            "rows": list(_rc_display_data.get("rows") or []),
                            "row_count": len(_rc_rows),
                            "source": "governed_cache",
                            "source_note": (
                                "Computed locally from the cached result. "
                                "No result values were sent to the model."
                            ),
                            "currency_columns": _rc_currency,
                            "column_formats": _rc_formats,
                            "display_formats": _rc_display_formats,
                            "diagnostics": dict(_rc_display_data.get("diagnostics") or {}),
                            "kpi": _rc_display_payload.get("kpi"),
                            "chart": _rc_chart,
                            "narration": _rc_outcome.message or None,
                            "trust": _rc_followup.evidence,
                        })
                        _trace_finish(
                            _rc_trace_id,
                            status="success",
                            answer_type="table",
                            row_count=len(_rc_rows),
                            duration_ms=_rc_dur_ms,
                            final_answer_summary=(
                                "Result-chat answered from governed session cache"
                            ),
                        )
                        continue

                    if _rc_followup.status in {"blocked", "error", "missing"}:
                        detail = (
                            "The request was stopped locally. No cached rows, sample values, "
                            "source SQL, or bound literals were sent to the model."
                            if _rc_followup.status == "blocked"
                            else "Run the business question again or use an exact result column."
                        )
                        await websocket.send_json({
                            "type": "result_chat_error",
                            "result_id": rc_result_id,
                            "content": _rc_followup.reason or "The cached result could not be updated.",
                            "detail": detail,
                        })
                        _trace_finish(
                            _rc_trace_id,
                            status="error",
                            answer_type="cache_transform_error",
                            error_message=_rc_followup.reason,
                        )
                        continue

                    if _rc_followup.status == "clarification" and _rc_followup.outcome is not None:
                        await websocket.send_json({
                            "type": "result_chat_clarification",
                            "result_id": rc_result_id,
                            "prompt": (
                                _rc_followup.outcome.clarification_prompt
                                or "Which result value did you mean?"
                            ),
                            "options": list(_rc_followup.outcome.clarification_options or []),
                        })
                        _trace_finish(
                            _rc_trace_id,
                            status="success",
                            answer_type="cache_transform_clarification",
                            final_answer_summary="Asked for confirmation instead of guessing.",
                        )
                        continue

                    # The metadata-only cache engine cannot answer this request.
                    # Continue through the governed production database fallback.
                    _rc_sql = "CANNOT_GENERATE"
                    _trace_update(
                        _rc_trace_id,
                        route="result_chat_db_fallback",
                        sql_validation_status="cannot_generate",
                    )

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
                                    _fb_system = _fb_system + "\n\n" + "\n".join(_fb_hist_lines)

                                # Original SQL anchor — unified for both aggregate and
                                # identifier results. The LLM can use it as a subquery,
                                # CTE, or just keep the same WHERE conditions.
                                _orig_q    = _cached_result.get("question", "")
                                _drill_ctx = _build_metadata_followup_context(
                                    original_question = _orig_q,
                                    follow_up_question= rc_question,
                                    schema            = _rc_schema,
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
                                    if not _fb_ok and _fb_code in ("unknown_table", "unknown_column", "date_key_format", "anti_join_shape", "fanout_aggregate", "parse"):
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
                            _fb_display_payload = build_assistant_response(
                                question=rc_question,
                                rows=_fb_rows,
                                sql=_fb_sql,
                                duration_ms=_fb_dur,
                                data_source="database",
                                question_id=_rc_question_id,
                            )
                            _fb_display_data = dict(_fb_display_payload.get("data") or {})
                            await websocket.send_json({
                                "type":             "result_chat_response",
                                "result_id":        rc_result_id,
                                "question":         rc_question,
                                "sql":              _fb_sql,
                                "rows":             list(_fb_display_data.get("rows") or []),
                                "row_count":        len(_fb_rows),
                                "source":           "database",
                                "source_note":      "Answer required a full database query.",
                                "currency_columns": list(_fb_display_data.get("currency_columns") or []),
                                "column_formats":   dict(_fb_display_data.get("column_formats") or {}),
                                "display_formats":  dict(_fb_display_data.get("display_formats") or {}),
                                "diagnostics":      dict(_fb_display_data.get("diagnostics") or {}),
                                "kpi":              _fb_display_payload.get("kpi"),
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
                pending = get_pending(
                    account_id,
                    zoom_user_id,
                    session_id=adapter.session_id,
                )
                if not pending:
                    await websocket.send_json({"type": "error", "content": "That clarification is no longer active. Please ask the question again."})
                    continue
                cmeta = pending.get("clarification_meta") or {}

                # Login-time report prompt reply (chip click) -- resolved
                # separately from the generic clarification flow below since
                # this isn't refining a data question; picking a report (or
                # declining) delivers it directly instead of feeding into
                # combine_with_clarification + handle_query.
                if cmeta.get("source") == "login_report_prompt":
                    from core.dispatcher import _deliver_report_via_adapter
                    from core.report_engine import list_promptable_reports

                    selected_id = str(data.get("option_id") or "").strip()
                    free_text = _ws_text_value(data.get("text"), "text", "question", "value", "label")
                    opts = cmeta.get("options") or []
                    selected = next((o for o in opts if str(o.get("id") or "") == selected_id), None)
                    if not selected and free_text:
                        selected = resolve_option_text(opts, free_text)
                    clear_pending(
                        account_id, zoom_user_id, session_id=adapter.session_id
                    )

                    report = None
                    if selected and selected.get("id") != "no_thanks":
                        reports = list_promptable_reports(account_id, portal_user)
                        report = next((r for r in reports if str(r["id"]) == str(selected.get("value"))), None)
                    if report:
                        await _deliver_report_via_adapter(account_id, portal_user, report, adapter.make_event(""), adapter)
                    else:
                        await websocket.send_json({"type": "system", "content": "No worries — skipping today's reports."})
                    continue

                # Presentation follow-ups remain session-local. A confirmed
                # ambiguous reference, or a date/currency/column choice, must
                # execute against the cached snapshot rather than becoming a
                # fresh source-SQL question.
                if cmeta.get("source") in {
                    "result_reference_confirmation", "local_result_command",
                }:
                    opts = cmeta.get("options") or []
                    selected_id = str(data.get("option_id") or "").strip()
                    free_text = _ws_text_value(
                        data.get("text"), "text", "question", "value", "label"
                    )
                    selected = next(
                        (option for option in opts if str(option.get("id") or "") == selected_id),
                        None,
                    )
                    if not selected and free_text:
                        selected = resolve_option_text(opts, free_text)
                    if not selected:
                        await adapter.send_clarification_prompt(
                            adapter.make_event(pending["original_q"]),
                            cmeta.get("question") or "Please choose one option.",
                            opts,
                        )
                        continue

                    clear_pending(
                        account_id, zoom_user_id, session_id=adapter.session_id
                    )
                    if cmeta.get("source") == "result_reference_confirmation":
                        if str(selected.get("id") or "") == "new-question":
                            await websocket.send_json({
                                "type": "message",
                                "role": "assistant",
                                "content": (
                                    "Understood. Please restate the new business question, "
                                    "and I’ll answer it from the governed data source."
                                ),
                            })
                            await websocket.send_json({"type": "typing", "active": False})
                            continue
                        resolved_text = pending["original_q"]
                        resolved_command = compile_confirmed_result_presentation(
                            resolved_text
                        )
                    else:
                        resolved_text = str(
                            selected.get("resolved_question")
                            or selected.get("value")
                            or selected.get("label")
                            or ""
                        ).strip()
                        resolved_command = parse_result_command(resolved_text)

                    if resolved_command is None:
                        await websocket.send_json({
                            "type": "assistant_error",
                            "role": "assistant",
                            "content": "I could not apply that display choice. Please try again.",
                        })
                        await websocket.send_json({"type": "typing", "active": False})
                        continue
                    await websocket.send_json({"type": "typing", "active": True})
                    await _run_local_result_command(resolved_text, resolved_command)
                    continue

                opts = cmeta.get("options") or []
                selected_id = str(data.get("option_id") or "").strip()
                free_text = _ws_text_value(
                    data.get("text"), "text", "question", "value", "label"
                )

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
                        if cmeta.get("source") == "metric_date_context":
                            selected = resolve_date_option_text(
                                cmeta.get("all_options") or opts,
                                free_text,
                            )
                        else:
                            selected = resolve_option_text(opts, free_text)
                    if not selected:
                        send_prompt = getattr(adapter, "send_clarification_prompt", None)
                        if callable(send_prompt):
                            retry_question = (
                                "I couldn't match that business date unambiguously. "
                                "Choose a suggested business date or type a more "
                                "specific business name."
                                if cmeta.get("source") == "metric_date_context"
                                else cmeta.get("question")
                                or "Please choose one option."
                            )
                            await send_prompt(
                                adapter.make_event(pending["original_q"]),
                                retry_question,
                                opts,
                            )
                        else:
                            await websocket.send_json({"type": "error", "content": "Please choose one of the available clarification options."})
                        continue
                    selected_text = str(selected.get("value") or selected.get("label") or "").strip()
                    selected_opt_id = str(selected.get("id") or "") or None   # Fix #2
                    option_is_visible = any(
                        str(option.get("id") or "") == selected_opt_id
                        for option in opts
                    )
                    combined, term_hint = combine_with_clarification(
                        pending["original_q"],
                        selected_text,
                        cmeta,
                        selected_option_id=(
                            selected_opt_id if option_is_visible else None
                        ),
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
                _clarification_event = adapter.make_event(combined)
                if cmeta.get("source") == "metric_date_context" and opts:
                    _clarification_event.raw["_clarification_selected_source"] = (
                        "metric_date_context"
                    )
                    _clarification_event.raw["_clarification_selected_option"] = dict(
                        selected
                    )
                attach_clarification_resolution(_clarification_event, pending)
                clear_pending(
                    account_id, zoom_user_id, session_id=adapter.session_id
                )
                log.info(
                    "WS clarification resolved for '%s' with reply '%s'",
                    pending["original_q"][:80], log_label[:80],
                )
                clarification_run = None
                try:
                    try:
                        clarification_run = AgentRunSession.resume(
                            account_id=account_id,
                            portal_user_id=int(portal_user["id"]),
                            external_thread_id=adapter.thread_id,
                            reply=log_label,
                        )
                    except Exception as audit_exc:
                        log.warning(
                            "Clarification agent resume unavailable; continuing query: %s",
                            audit_exc,
                        )
                    if clarification_run:
                        await adapter.send_agent_event(
                            clarification_run.event("agent_run_started")
                        )
                    with activate_agent_run(clarification_run):
                        await handle_query(
                            account_id,
                            _clarification_event,
                            adapter,
                            combined,
                            portal_user,
                            is_clarification=True,
                        )
                    if clarification_run:
                        clarification_run.complete_if_running()
                        await adapter.send_agent_event(
                            clarification_run.event("agent_run_finished")
                        )
                except Exception as e:
                    log.error("WS clarification handle_query error: %s", e)
                    if clarification_run:
                        try:
                            clarification_run.fail(e)
                            await adapter.send_agent_event(
                                clarification_run.event("agent_run_finished")
                            )
                        except Exception as audit_exc:
                            log.debug("Clarification agent audit failed: %s", audit_exc)
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
                                    account_id=account_id,
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
                                    account_id=account_id,
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
                                    account_id=account_id,
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
                await websocket.send_json({
                    "type": "assistant_error",
                    "role": "assistant",
                    "content": "I could not read that question. Please type it again.",
                })
                await websocket.send_json({"type": "typing", "active": False})
                continue

            # Opt-in "explain your plan first" preview -- checked before
            # every other intent so a pending preview's confirm/correct
            # reply is never swallowed by the report-builder or
            # result-command gates below. A pending preview means the
            # PREVIOUS turn asked to preview a plan; this turn is either a
            # confirmation ("go ahead") -> run the original question, or
            # anything else -> treated as a correction, run the original
            # question enriched with this reply (same proven pattern as
            # every other correction/fallback path in this file).
            _pp_session_id = getattr(adapter, "session_id", "") or ""
            _pending_preview = pending_plan_previews.get(account_id, _pp_session_id)
            if _pending_preview is not None:
                pending_plan_previews.clear(account_id, _pp_session_id)
                _pp_question = (
                    _pending_preview.question if _PLAN_PREVIEW_CONFIRM_RE.match(text)
                    else f"{_pending_preview.question} -- {text}"
                )
                if current_query_task and not current_query_task.done():
                    current_query_task.cancel()
                current_query_task = asyncio.create_task(
                    _run_main_question(_pp_question, table_hint, schema_hint)
                )
                continue

            _pp_match = _PLAN_PREVIEW_INTENT_RE.match(text)
            if _pp_match:
                _pp_question = (_pp_match.group("question") or "").strip()
                if not _pp_question:
                    await websocket.send_json({
                        "type": "message",
                        "content": (
                            "Sure -- ask me a question and I'll explain my plan before "
                            'running it, e.g. "explain your plan: what was net revenue '
                            'for last 7 days".'
                        ),
                    })
                    await websocket.send_json({"type": "typing", "active": False})
                    continue
                _pp_db_cfg = get_client_db(account_id) or {}
                _preview = build_plan_preview(
                    _pp_question, account_id, _pp_db_cfg.get("db_type", "azure_sql"),
                )
                pending_plan_previews.set(account_id, _pp_session_id, _preview)
                await websocket.send_json({
                    "type": "message",
                    "content": f'{_preview.summary} Say "go ahead" to run it, or tell me what to change.',
                })
                await websocket.send_json({"type": "typing", "active": False})
                continue

            # Ana-style dashboard creation/refinement. This is detected before
            # cached-result transforms because "add this to my dashboard" is
            # an artifact action, not a data operation.
            if (
                _DASHBOARD_CREATE_INTENT_RE.search(text)
                or _DASHBOARD_ADD_INTENT_RE.search(text)
                or _DASHBOARD_ADD_QUERY_RE.search(text)
                or _DASHBOARD_CHART_TYPE_RE.search(text)
                or _DASHBOARD_RENAME_INTENT_RE.search(text)
                or _DASHBOARD_PUBLISH_INTENT_RE.search(text)
                or _DASHBOARD_FILTER_INTENT_RE.search(text)
                or _DASHBOARD_TAB_INTENT_RE.search(text)
                or _DASHBOARD_SCHEDULE_INTENT_RE.search(text)
                or _DASHBOARD_SHARE_INTENT_RE.search(text)
                or _DASHBOARD_ROLLBACK_INTENT_RE.search(text)
            ):
                if current_query_task and not current_query_task.done():
                    current_query_task.cancel()
                current_query_task = asyncio.create_task(_run_dashboard_chat(text))
                continue

            # Conversational report/playbook building -- available to every
            # portal user (no role gate). Detected before any result-cache
            # command since "build a report" has nothing to do with a cached
            # result.
            if _REPORT_BUILDER_INTENT_RE.search(text):
                if current_query_task and not current_query_task.done():
                    current_query_task.cancel()
                current_query_task = asyncio.create_task(_run_report_builder_chat(text))
                continue

            # Deep analysis is explicit and operates only on the most recent
            # governed result. Detect it before general cached-result routing
            # so phrases such as "find outliers in this result" create an
            # auditable work run and artifact instead of a one-line transform.
            if _ANALYSIS_WORK_INTENT_RE.search(text) or _CUSTOM_PYTHON_INTENT_RE.search(text):
                if current_query_task and not current_query_task.done():
                    current_query_task.cancel()
                current_query_task = asyncio.create_task(_run_analysis_work(text))
                continue

            # Conversational result operations run before every insight/LLM
            # route. Recognised commands fail closed: a sensitive value in an
            # exclusion command can never fall through into a model prompt.
            result_command = parse_result_command(text)
            if result_command is not None:
                if current_query_task and not current_query_task.done():
                    current_query_task.cancel()
                current_query_task = asyncio.create_task(
                    _run_local_result_command(
                        text, result_command, table_hint, schema_hint,
                    )
                )
                continue

            # Natural-language analytics over the cached result use a model
            # only to choose a constrained operation from column metadata.
            # Cached rows, sample values, source SQL, and locally bound filter
            # literals are never included in that prompt.
            _cache_snapshot = result_cache.get_snapshot(
                getattr(adapter, "session_id", "") or "",
                getattr(adapter, "last_result_id", None),
            )
            _cache_columns = [
                str(column.get("name") or "")
                for column in (_cache_snapshot or {}).get("schema", [])
                if isinstance(column, dict) and column.get("name")
            ]
            if needs_result_reference_confirmation(text, bool(_cache_snapshot)):
                prompt = "Are you referring to the previous result?"
                options = [
                    {
                        "id": "use-previous-result",
                        "label": "Yes — use the previous result",
                        "value": "use_previous_result",
                    },
                    {
                        "id": "new-question",
                        "label": "No — this is a new question",
                        "value": "new_question",
                    },
                ]
                save_pending(
                    account_id,
                    zoom_user_id,
                    text,
                    clarification_meta=prepare_clarification_meta(
                        adapter.make_event(text),
                        {
                            "source": "result_reference_confirmation",
                            "question": prompt,
                            "options": options,
                        },
                        source="result_reference_confirmation",
                    ),
                    session_id=adapter.session_id,
                )
                await adapter.send_clarification_prompt(
                    adapter.make_event(text), prompt, options,
                )
                await websocket.send_json({"type": "typing", "active": False})
                continue
            if _RECONCILE_INTENT_RE.search(text) and _cache_snapshot and _cache_snapshot.get("sql"):
                if current_query_task and not current_query_task.done():
                    current_query_task.cancel()
                current_query_task = asyncio.create_task(_run_reconcile_chat(_cache_snapshot))
                continue

            if is_metadata_result_question(text):
                _route_cached_analysis = True
            else:
                _route_cached_analysis = should_route_to_result_cache(
                    text,
                    bool(_cache_snapshot),
                    cached_col_names=_cache_columns,
                )
            if _route_cached_analysis:
                if current_query_task and not current_query_task.done():
                    current_query_task.cancel()
                current_query_task = asyncio.create_task(
                    _run_metadata_result_planner(text, table_hint, schema_hint)
                )
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
                            account_id=account_id,
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
            # Runs as a background task (not awaited here) so this loop stays
            # free to receive a "cancel" message while the question runs. A
            # new question implicitly supersedes an unfinished one.
            if current_query_task and not current_query_task.done():
                current_query_task.cancel()
            current_query_task = asyncio.create_task(
                _run_main_question(text, table_hint, schema_hint)
            )

    except WebSocketDisconnect:
        log.info("WebSocket chat disconnected: user=%d account=%s", user_id, account_id)
    except Exception as e:
        log.exception("WebSocket error for user %d: %s", user_id, e)
        try:
            await websocket.send_json({
                "type":    "error",
                "content": "Connection error. Please refresh and try again.",
            })
        except Exception:
            pass
    finally:
        # Don't leak a question-answering task past the connection it was
        # answering into — cancel it so it stops hitting the (closed) socket
        # and the LLM/DB calls it's waiting on are released promptly.
        if current_query_task and not current_query_task.done():
            current_query_task.cancel()

