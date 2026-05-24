"""
portal/routes.py

User-facing portal — completely separate from /admin.
Served at /portal/*

Routes:
  GET  /portal/login              — login page
  POST /portal/login              — authenticate
  GET  /portal/logout             — clear session
  GET  /portal/register?token=xxx — registration page (one-time link from bot)
  POST /portal/register           — complete registration
  GET  /portal/dashboard          — personal pinned chart dashboard
  GET  /portal/change-password    — change password form
  POST /portal/change-password    — save new password
  GET  /portal/pin-confirm        — confirm pin a chart (from bot link)
  POST /portal/pin-confirm        — save pinned chart
  POST /portal/unpin              — remove pinned chart
  GET  /portal/kb                 — view KB files for user's tables
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import re
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Request, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import store
from core.schema import run_query
from core.chart import detect_chart_type, build_chart_payload
from core.semantic_layer import build_semantic_layer_tables, find_semantic_field
from core.portal_notifications import portal_notification_hub

log = logging.getLogger("querybot.portal")

router    = APIRouter(prefix="/portal")
templates = Jinja2Templates(
    directory=str(Path(__file__).parent / "templates")
)

_COOKIE = "qb_portal_session"  # different from admin cookie


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _session_secret() -> str:
    return os.getenv("PORTAL_SESSION_SECRET") or os.getenv("SESSION_SECRET") or "change-me-in-production"


def _sign_session_value(user_id: int) -> str:
    payload = str(user_id).encode()
    sig = hmac.new(_session_secret().encode(), payload, hashlib.sha256).hexdigest()
    token = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    return f"{token}.{sig}"


def _read_session_value(raw: str) -> int | None:
    try:
        token, sig = raw.split(".", 1)
        padding = "=" * (-len(token) % 4)
        payload = base64.urlsafe_b64decode((token + padding).encode())
        expected = hmac.new(_session_secret().encode(), payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        return int(payload.decode())
    except Exception:
        return None


def _cookie_secure(request: Request) -> bool:
    return request.url.scheme == "https"


def _set_portal_cookie(resp: RedirectResponse, request: Request, user_id: int) -> None:
    resp.set_cookie(
        _COOKIE,
        _sign_session_value(user_id),
        httponly=True,
        samesite="lax",
        secure=_cookie_secure(request),
    )


def _get_portal_user(request: Request) -> dict | None:
    """Return portal user from signed session cookie, or None."""
    raw = request.cookies.get(_COOKIE)
    if not raw:
        return None
    user_id = _read_session_value(raw)
    if not user_id:
        return None
    try:
        return store.get_user(int(user_id))
    except Exception:
        return None


def _resp(request, name, ctx=None):
    return templates.TemplateResponse(request=request, name=name, context=ctx or {})


def _login_redirect():
    return RedirectResponse("/portal/login", status_code=303)


def _get_portal_user_from_socket(websocket: WebSocket) -> dict | None:
    raw = websocket.cookies.get(_COOKIE)
    if not raw:
        return None
    user_id = _read_session_value(raw)
    if not user_id:
        return None
    try:
        return store.get_user(int(user_id))
    except Exception:
        return None


@router.websocket("/ws/notifications")
async def portal_notifications_ws(websocket: WebSocket):
    user = _get_portal_user_from_socket(websocket)
    if not user:
        await websocket.close(code=4401)
        return
    await portal_notification_hub.connect(
        websocket,
        account_id=user["account_id"],
        user_id=int(user["id"]),
    )
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.debug("Portal notification socket closed: %s", exc)
    finally:
        await portal_notification_hub.disconnect(websocket)


def _guess_dynamic_suggestions(
    account_id: str,
    allowed_tables: list[str] | None,
    max_items: int = 6,
) -> list[str]:
    """
    Build chat suggestions from the client's business glossary.
    
    This is industry-neutral: no hardcoded table or term names. For each
    active term a user is allowed to see (based on table access), synthesize
    a natural question using the term plus its kind — metrics become
    trend/breakdown/top-N questions, dimensions become breakdown questions,
    filters become filtered-list questions.
    
    Falls back to empty list when the glossary has no entries yet.
    """
    try:
        terms = store.list_terms(account_id, active_only=True)
    except Exception:
        return []
    if not terms:
        return []

    allowed_upper = (
        {str(t).upper() for t in allowed_tables} if allowed_tables else None
    )

    suggestions: list[str] = []
    seen: set[str] = set()

    def _add(q: str) -> None:
        key = q.lower().strip()
        if key and key not in seen:
            seen.add(key)
            suggestions.append(q)

    def _term_is_visible(term: dict) -> bool:
        """Term is shown if user has access to at least one of its tables,
        or if the term is not bound to any table (global)."""
        if allowed_upper is None:
            return True
        tbls_raw = (term.get("tables_involved") or "").strip()
        if not tbls_raw:
            return True  # unbounded term — always visible
        tbls = {s.strip().upper() for s in tbls_raw.split(",") if s.strip()}
        return bool(tbls & allowed_upper)

    # Prefer metric terms — they give the best questions. Then dimensions,
    # then filters, then entities. Within a kind, terms with richer
    # definitions come first (they read more naturally).
    priority = {"metric": 0, "dimension": 1, "filter": 2, "entity": 3}
    terms_sorted = sorted(
        terms,
        key=lambda t: (
            priority.get(t.get("kind", "metric"), 99),
            -(len(t.get("definition", "")) + len(t.get("aliases", ""))),
        ),
    )

    for term in terms_sorted:
        if len(suggestions) >= max_items:
            break
        if not _term_is_visible(term):
            continue

        # Use the term's definition if it reads like a noun phrase,
        # else fall back to the term itself.
        name = (term.get("term") or "").strip()
        if not name:
            continue
        definition = (term.get("definition") or "").strip()

        # Pick the most natural label the user would recognize — aliases
        # are often more conversational than the canonical term.
        aliases = [
            a.strip() for a in (term.get("aliases") or "").split(",") if a.strip()
        ]
        natural_name = aliases[0] if aliases else name

        kind = term.get("kind", "metric")
        if kind == "metric":
            # Generate a couple of natural questions per metric
            _add(f"Show {natural_name} trend over time")
            if len(suggestions) < max_items:
                _add(f"What is our total {natural_name}?")
        elif kind == "dimension":
            _add(f"Break down activity by {natural_name}")
        elif kind == "filter":
            # Definition tends to read better for filters
            label = definition or natural_name
            _add(f"Show records where {label}")
        elif kind == "entity":
            _add(f"How many {natural_name} do we have?")

    return suggestions[:max_items]


def _normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _is_human_metric_name(name: str) -> bool:
    value = (name or "").strip()
    if not value:
        return False
    if "_" in value:
        return False
    if re.fullmatch(r"[A-Z0-9_]+", value):
        return False
    return True


def _collect_visible_business_phrases(
    account_id: str,
    allowed_tables: list[str] | None,
) -> set[str]:
    phrases: set[str] = set()
    allowed_upper = {str(t).upper() for t in allowed_tables} if allowed_tables else None
    try:
        terms = store.list_terms(account_id, active_only=True)
    except Exception:
        terms = []

    for term in terms:
        if allowed_upper is not None:
            tbls_raw = (term.get("tables_involved") or "").strip()
            if tbls_raw:
                tbls = {s.strip().upper() for s in tbls_raw.split(",") if s.strip()}
                if tbls and not (tbls & allowed_upper):
                    continue
        canonical = _normalize_phrase(term.get("term") or "")
        if canonical:
            phrases.add(canonical)
        for alias in (term.get("aliases") or "").split(","):
            alias_norm = _normalize_phrase(alias)
            if alias_norm:
                phrases.add(alias_norm)
    return phrases


def _question_matches_visible_business_language(question: str, visible_phrases: set[str]) -> bool:
    q = _normalize_phrase(question)
    if not q or not visible_phrases:
        return False
    q_words = {w for w in re.split(r"[^a-z0-9]+", q) if len(w) >= 3}
    for phrase in visible_phrases:
        if phrase in q:
            return True
        phrase_words = {w for w in re.split(r"[^a-z0-9]+", phrase) if len(w) >= 3}
        if phrase_words and len(q_words & phrase_words) >= min(2, len(phrase_words)):
            return True
    return False


def _guess_safe_metric_suggestions(
    account_id: str,
    allowed_tables: list[str] | None = None,
    max_items: int = 6,
) -> list[str]:
    """
    Conservative fallback suggestions backed by deterministic metric SQL.

    These prompts are intentionally narrow so the chat UI does not suggest
    trend/breakdown questions that the workspace cannot reliably answer.
    """
    try:
        metrics = store.list_metrics(account_id)
    except Exception:
        return []
    if not metrics:
        return []

    visible_phrases = _collect_visible_business_phrases(account_id, allowed_tables)
    suggestions: list[str] = []
    seen: set[str] = set()

    for metric in sorted(
        metrics,
        key=lambda m: (
            -(len(m.get("description", "")) + len(m.get("synonyms", ""))),
            m.get("name", ""),
        ),
    ):
        name = (metric.get("name") or "").strip()
        sql_template = (metric.get("sql_template") or "").strip()
        if not name or not sql_template or not _is_human_metric_name(name):
            continue
        prompt = f"What is our total {name}?"
        if visible_phrases and not _question_matches_visible_business_language(prompt, visible_phrases):
            continue
        key = prompt.lower()
        if key in seen:
            continue
        seen.add(key)
        suggestions.append(prompt)
        if len(suggestions) >= max_items:
            break

    return suggestions


def _compact_number(value: int | float | None) -> str:
    """Format KPI numbers for compact UI display."""
    try:
        n = float(value or 0)
    except Exception:
        n = 0
    sign = "-" if n < 0 else ""
    n = abs(n)
    for suffix, threshold in (("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if n >= threshold:
            compact = n / threshold
            text = f"{compact:.1f}".rstrip("0").rstrip(".")
            return f"{sign}{text}{suffix}"
    return f"{sign}{int(n):,}"


def _query_limit_status(account_id: str) -> dict:
    """Monthly query allowance using the same counter as admin billing."""
    client = store.get_client(account_id) or {}
    limit = int(client.get("query_limit_monthly") or 500)
    used = int(store.get_monthly_query_count(account_id) or 0)
    remaining = max(limit - used, 0)
    pct = 0 if limit <= 0 else min(round(used / limit * 100), 100)
    blocked = limit > 0 and used >= limit
    warning = limit > 0 and pct >= 80 and not blocked
    status = {
        "used": used,
        "limit": limit,
        "remaining": remaining,
        "limit_pct": pct,
        "warning": warning,
        "blocked": blocked,
        "used_label": _compact_number(used),
        "limit_label": _compact_number(limit),
        "remaining_label": _compact_number(remaining),
    }
    if blocked:
        status.update({
            "level": "blocked",
            "title": "Monthly query limit reached",
            "message": f"{used}/{limit} queries used this month. Ask your admin to increase the limit.",
        })
    elif warning:
        status.update({
            "level": "warning",
            "title": "Monthly query limit warning",
            "message": f"{used}/{limit} queries used this month. Your workspace is above 80% of its limit.",
        })
    else:
        status.update({
            "level": "normal",
            "title": "Monthly query limit",
            "message": f"{remaining} queries remaining this month.",
        })
    return status


def _get_available_schemas(user: dict) -> list[dict]:
    """
    Return the distinct schemas visible to this user based on their allowed
    tables and the client's discovered schema.

    Returns a sorted list of dicts:
      { "name": "HR", "table_count": 3 }

    "All schemas" is handled in the template — this returns only the named
    schemas. If the user has access to just one schema, the selector is
    hidden (single-schema clients don't need it).
    """
    import store as _store
    import json
    from core.schema import load_schema_json

    account_id = user["account_id"]
    allowed    = _store.get_allowed_tables(user)  # None = admin (all tables)

    client     = _store.get_client(account_id) or {}
    state_data = json.loads(client.get("state_data") or "{}")
    schema_dir = state_data.get("schema_dir", "")

    if not schema_dir:
        return []

    all_known = {str(k).upper() for k in load_schema_json(schema_dir).keys()}
    if not all_known:
        return []

    # Narrow to user's allowed set
    if allowed is not None:
        allowed_upper = {t.upper() for t in allowed}
        # Expand allowed to include bare names for matching
        effective = {t for t in all_known if (
            t in allowed_upper or
            t.split(".")[-1] in allowed_upper or
            any(t.endswith(f".{a}") or t == a for a in allowed_upper)
        )}
    else:
        effective = all_known  # admin — sees everything

    # Extract schema names from FQNs — only from 2-part or 3-part names
    schema_counts: dict[str, int] = {}
    for fqn in effective:
        parts = fqn.split(".")
        if len(parts) >= 2:
            schema_name = parts[-2].upper()
            # Skip generic/system schemas
            if schema_name not in {"DBO", "SYS", "INFORMATION_SCHEMA", "GUEST"}:
                schema_counts[schema_name] = schema_counts.get(schema_name, 0) + 1
        elif len(parts) == 1 and parts[0]:
            # Bare name — can't determine schema
            pass

    # Also include dbo if it has tables (common default schema)
    for fqn in effective:
        parts = fqn.split(".")
        if len(parts) >= 2 and parts[-2].upper() == "DBO":
            schema_counts["DBO"] = schema_counts.get("DBO", 0) + 1

    return sorted(
        [{"name": name, "table_count": count}
         for name, count in schema_counts.items()],
        key=lambda x: x["name"],
    )


def _build_chat_suggestions(user: dict) -> list[dict]:
    """
    Return up to 6 structured suggestion dicts: {"question": str, "fqn": str}.
    The fqn travels with the suggestion so the chat UI can pass it as a schema
    hint when the user clicks — fixing schema-unaware SQL generation.
    """
    try:
        import store
        from core.suggestions import get_suggestions

        account_id = user["account_id"]
        allowed    = store.get_allowed_tables(user)
        allowed_set = set(allowed) if allowed else None

        client     = store.get_client(account_id) or {}
        import json as _json
        state_data = _json.loads(client.get("state_data") or "{}")
        kb_dir     = state_data.get("kb_dir", "")
        schema_dir = state_data.get("schema_dir", "")

        suggestions = get_suggestions(
            account_id=account_id,
            kb_dir=kb_dir,
            allowed_tables=allowed_set,
            n=6,
            schema_dir=schema_dir,
        )

        # Fallback: glossary-based synthesis (returns plain strings — wrap them)
        if len(suggestions) < 4:
            extra = _guess_safe_metric_suggestions(account_id, allowed, max_items=6)
            for q in extra:
                if not any(s["question"] == q for s in suggestions):
                    suggestions.append({"question": q, "fqn": ""})
                if len(suggestions) >= 6:
                    break

        return suggestions[:6]

    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Login / logout
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/login", response_class=HTMLResponse)
async def portal_login_page(request: Request):
    if _get_portal_user(request):
        return RedirectResponse("/portal/dashboard", status_code=303)
    return _resp(request, "portal_login.html", {
        "error": request.query_params.get("error", "")
    })


@router.post("/login")
async def portal_login_submit(
    request: Request,
    account_id: str = Form(...),
    email:      str = Form(...),
    password:   str = Form(...),
):
    # Validate client exists
    client = store.get_client(account_id)
    if not client:
        return _resp(request, "portal_login.html",
                     {"error": "Account ID not found."})

    user = store.get_user_by_email(account_id, email)
    if not user or not store.verify_password(user, password):
        return _resp(request, "portal_login.html",
                     {"error": "Invalid email or password."})

    # Temp password — force change
    if user.get("is_temp_pw"):
        resp = RedirectResponse("/portal/change-password?forced=1", status_code=303)
        _set_portal_cookie(resp, request, user["id"])
        return resp

    resp = RedirectResponse("/portal/dashboard", status_code=303)
    _set_portal_cookie(resp, request, user["id"])
    return resp


@router.get("/logout")
async def portal_logout():
    resp = RedirectResponse("/portal/login", status_code=303)
    resp.delete_cookie(_COOKIE)
    return resp


# ══════════════════════════════════════════════════════════════════════════════
# Registration
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/register", response_class=HTMLResponse)
async def portal_register_page(request: Request, token: str = ""):
    if not token:
        return _resp(request, "portal_register.html",
                     {"error": "Invalid or missing registration link.", "token": ""})

    token_data = _peek_token(token)
    if not token_data:
        return _resp(request, "portal_register.html",
                     {"error": "This link has expired or already been used. "
                               "Message the bot again to get a new link.", "token": ""})

    client = store.get_client(token_data["account_id"])
    return _resp(request, "portal_register.html", {
        "token":        token,
        "account_id":   token_data["account_id"],
        "client_name":  client.get("client_name", "") if client else "",
        "zoom_user_id": token_data["zoom_user_id"],
        "error":        "",
    })


@router.post("/register")
async def portal_register_submit(
    request:      Request,
    token:        str = Form(...),
    name:         str = Form(...),
    email:        str = Form(...),
    password:     str = Form(...),
    confirm_pw:   str = Form(...),
):
    token_data = _peek_token(token)
    if not token_data:
        return _resp(request, "portal_register.html",
                     {"error": "Link expired or already used. Message the bot for a new one.",
                      "token": "", "account_id": "", "client_name": "", "zoom_user_id": ""})

    account_id   = token_data["account_id"]
    zoom_user_id = token_data["zoom_user_id"]

    if len(name.strip()) < 2:
        return _re_render_register(request, token, account_id, zoom_user_id,
                                   "Please enter your full name.")
    if "@" not in email:
        return _re_render_register(request, token, account_id, zoom_user_id,
                                   "Please enter a valid email address.")
    if len(password) < 8:
        return _re_render_register(request, token, account_id, zoom_user_id,
                                   "Password must be at least 8 characters.")
    if password != confirm_pw:
        return _re_render_register(request, token, account_id, zoom_user_id,
                                   "Passwords do not match.")

    token_data = store.consume_registration_token(token)
    if not token_data:
        return _re_render_register(request, token, account_id, zoom_user_id,
                                   "This registration link is no longer valid.")

    existing = store.get_user_by_email(account_id, email)
    if existing:
        store.link_zoom_user(existing["id"], zoom_user_id)
        resp = RedirectResponse("/portal/dashboard", status_code=303)
        _set_portal_cookie(resp, request, existing["id"])
        return resp

    user_id, _ = store.create_user(account_id, name.strip(), email.strip())
    store.change_password(user_id, password, is_temp=False)
    store.link_zoom_user(user_id, zoom_user_id)

    log.info("New user registered: %s for account %s", email, account_id)

    resp = RedirectResponse("/portal/dashboard?welcome=1", status_code=303)
    _set_portal_cookie(resp, request, user_id)
    return resp


def _peek_token(token: str) -> dict | None:
    """Check token validity without consuming it."""
    from store.db import get_db
    from datetime import datetime, timezone
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM registration_token WHERE token=? AND used=0", (token,)
        ).fetchone()
    if not row:
        return None
    row = dict(row)
    expiry = datetime.strptime(row["expires_at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) > expiry:
        return None
    return row


def _re_render_register(request, token, account_id, zoom_user_id, error):
    client = store.get_client(account_id)
    return _resp(request, "portal_register.html", {
        "token": token, "account_id": account_id,
        "client_name": client.get("client_name", "") if client else "",
        "zoom_user_id": zoom_user_id, "error": error,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/dashboard", response_class=HTMLResponse)
async def portal_dashboard(request: Request):
    user = _get_portal_user(request)
    if not user:
        return _login_redirect()

    charts        = store.list_pinned_charts(user["id"])
    allowed       = store.get_allowed_tables(user)
    client        = store.get_client(user["account_id"]) or {}
    group_tables  = store.get_group_tables(user["group_id"]) if user.get("group_id") else []
    monthly_count = store.get_monthly_query_count(user["account_id"])
    query_status  = _query_limit_status(user["account_id"])
    token_status  = store.get_monthly_token_status(user["account_id"])
    token_status["used_label"] = _compact_number(token_status.get("total_tokens"))
    token_status["limit_label"] = _compact_number(token_status.get("limit"))
    token_status["remaining_label"] = (
        "Unlimited" if token_status.get("unlimited")
        else _compact_number(token_status.get("remaining"))
    )

    # Refresh all pinned charts — re-execute SQL against live DB
    rendered_charts = []
    db_cfg = None
    if charts:
        db_cfg = store.get_db_config(charts[0]["db_config_id"]) if charts[0].get("db_config_id") else None

    for chart in charts:
        chart_db = store.get_db_config(chart["db_config_id"]) if chart.get("db_config_id") else db_cfg
        chart_data = _refresh_chart(chart, chart_db)
        rendered_charts.append(chart_data)

    return _resp(request, "portal_dashboard.html", {
        "user":          user,
        "client":        client,
        "charts":        rendered_charts,
        "allowed_tables": sorted(allowed) if allowed else None,
        "group_tables":  group_tables,
        "monthly_count": monthly_count,
        "query_status":  query_status,
        "token_status":  token_status,
        "welcome":       request.query_params.get("welcome") == "1",
    })


def _refresh_chart(chart: dict, db_cfg: dict | None) -> dict:
    """Re-execute stored SQL and prepare interactive chart data."""
    result = dict(chart)
    result["chart_json"] = None
    result["error"] = None
    result["row_count"] = 0

    if not db_cfg:
        result["error"] = "Database not configured"
        return result

    try:
        rows = run_query(db_cfg["credentials"], db_cfg["db_type"], chart["sql_query"])
        result["row_count"] = len(rows)
        if rows:
            chart_type = chart.get("chart_type") or detect_chart_type(rows, question=chart.get("question", ""))
            payload = build_chart_payload(rows, chart_type, title=chart["title"]) if chart_type else None
            if payload:
                payload["color_palette"] = chart.get("color_palette") or "default"
                payload["chart_id"] = chart["id"]   # lets dashboard JS call update-chart
                result["chart_json"] = json.dumps(payload)
        store.update_chart_refreshed(chart["id"])
    except Exception as e:
        result["error"] = str(e)[:120]
        log.warning("Chart refresh failed for chart %d: %s", chart["id"], e)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Change password
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/change-password", response_class=HTMLResponse)
async def change_pw_page(request: Request):
    user = _get_portal_user(request)
    if not user:
        return _login_redirect()
    forced = request.query_params.get("forced") == "1"
    return _resp(request, "portal_change_password.html", {
        "user": user, "forced": forced, "error": "", "saved": False
    })


@router.post("/change-password")
async def change_pw_submit(
    request:    Request,
    current_pw: str = Form(""),
    new_pw:     str = Form(...),
    confirm_pw: str = Form(...),
):
    user = _get_portal_user(request)
    if not user:
        return _login_redirect()

    forced = not bool(current_pw)  # forced = no current password required

    # If not forced, verify current password
    if not forced and not store.verify_password(user, current_pw):
        return _resp(request, "portal_change_password.html", {
            "user": user, "forced": False,
            "error": "Current password is incorrect.", "saved": False
        })

    if len(new_pw) < 8:
        return _resp(request, "portal_change_password.html", {
            "user": user, "forced": forced,
            "error": "New password must be at least 8 characters.", "saved": False
        })
    if new_pw != confirm_pw:
        return _resp(request, "portal_change_password.html", {
            "user": user, "forced": forced,
            "error": "Passwords do not match.", "saved": False
        })

    store.change_password(user["id"], new_pw, is_temp=False)
    return RedirectResponse("/portal/dashboard", status_code=303)


# ══════════════════════════════════════════════════════════════════════════════
# Pin chart
# ══════════════════════════════════════════════════════════════════════════════

def _peek_pin_token(token: str) -> dict | None:
    """Check pin token validity without consuming it."""
    from store.db import get_db
    from datetime import datetime, timezone
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM pin_token WHERE token=?", (token,)
        ).fetchone()
    if not row:
        return None
    row = dict(row)
    expiry = datetime.strptime(row["expires_at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) > expiry:
        return None
    return row


def _consume_pin_token(token: str) -> dict | None:
    """Retrieve and delete a pin token. Returns None if expired or not found."""
    from store.db import get_db
    from datetime import datetime, timezone
    with get_db() as conn:
        try:
            row = conn.execute(
                "SELECT * FROM pin_token WHERE token=?", (token,)
            ).fetchone()
            if not row:
                return None
            row = dict(row)
            expiry = datetime.strptime(row["expires_at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) > expiry:
                conn.execute("DELETE FROM pin_token WHERE token=?", (token,))
                return None
            conn.execute("DELETE FROM pin_token WHERE token=?", (token,))
            return row
        except Exception:
            return None


@router.get("/pin-confirm", response_class=HTMLResponse)
async def pin_confirm_page(request: Request, token: str = ""):
    user = _get_portal_user(request)
    if not user:
        return _login_redirect()

    if not token:
        return _resp(request, "portal_pin_confirm.html", {
            "user": user, "error": "Invalid or missing pin token.",
            "token": "", "uid": "", "aid": "", "question": "", "sql": "", "ct": "bar", "dbid": "",
        })

    pin_data = _peek_pin_token(token)
    if not pin_data or pin_data["user_id"] != user["id"]:
        return _resp(request, "portal_pin_confirm.html", {
            "user": user,
            "error": "This pin link is invalid or has expired. Ask the bot again for a new one.",
            "token": "", "uid": "", "aid": "", "question": "", "sql": "", "ct": "bar", "dbid": "",
        })

    return _resp(request, "portal_pin_confirm.html", {
        "user":     user,
        "token":    token,
        "uid":      pin_data["user_id"],
        "aid":      pin_data["account_id"],
        "question": pin_data["question"],
        "sql":      pin_data["sql_query"],
        "ct":       pin_data["chart_type"],
        "dbid":     pin_data["db_config_id"],
        "error":    "",
    })


@router.post("/pin-confirm")
async def pin_confirm_submit(
    request:  Request,
    token:    str = Form(...),
    title:    str = Form(...),
):
    user = _get_portal_user(request)
    if not user:
        return _login_redirect()

    pin_data = _consume_pin_token(token)
    if not pin_data or pin_data["user_id"] != user["id"]:
        return _resp(request, "portal_pin_confirm.html", {
            "user": user,
            "error": "This pin link is invalid or has expired.",
            "token": "", "uid": "", "aid": "", "question": "", "sql": "", "ct": "bar", "dbid": "",
        })

    store.pin_chart(
        user_id=user["id"],
        account_id=pin_data["account_id"],
        title=title.strip() or pin_data["question"][:50],
        question=pin_data["question"],
        sql_query=pin_data["sql_query"],
        chart_type=pin_data["chart_type"],
        db_config_id=pin_data["db_config_id"],
    )
    return RedirectResponse("/portal/dashboard?pinned=1", status_code=303)


@router.post("/api/pin-chart")
async def pin_chart_api(request: Request):
    user = _get_portal_user(request)
    if not user:
        return JSONResponse({"ok": False, "error": "Authentication required."}, status_code=401)

    try:
        payload = await request.json()
    except Exception:
        payload = {}

    token          = str(payload.get("token") or "").strip()
    title          = str(payload.get("title") or "").strip()
    # Allow the frontend to send the currently-active chart type / palette
    # (user may have toggled type or changed palette before pinning)
    type_override    = str(payload.get("chart_type") or "").strip() or None
    palette_override = str(payload.get("color_palette") or "default").strip()
    if not token:
        return JSONResponse({"ok": False, "error": "Missing pin token."}, status_code=400)

    pin_data = _consume_pin_token(token)
    if not pin_data or pin_data["user_id"] != user["id"]:
        return JSONResponse({"ok": False, "error": "This pin request is invalid or has expired."}, status_code=400)

    store.pin_chart(
        user_id=user["id"],
        account_id=pin_data["account_id"],
        title=title or pin_data["question"][:50],
        question=pin_data["question"],
        sql_query=pin_data["sql_query"],
        chart_type=type_override or pin_data["chart_type"],
        db_config_id=pin_data["db_config_id"],
        color_palette=palette_override,
    )
    return JSONResponse({"ok": True})


@router.post("/api/update-chart")
async def update_chart_api(request: Request):
    """Update chart_type, color_palette, or title for a pinned chart."""
    user = _get_portal_user(request)
    if not user:
        return JSONResponse({"ok": False, "error": "Authentication required."}, status_code=401)
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    chart_id = int(payload.get("chart_id") or 0)
    if not chart_id:
        return JSONResponse({"ok": False, "error": "chart_id required."}, status_code=400)
    store.update_pinned_chart(
        chart_id=chart_id,
        user_id=user["id"],
        title=str(payload["title"]).strip() if "title" in payload else None,
        chart_type=str(payload["chart_type"]).strip() if "chart_type" in payload else None,
        color_palette=str(payload["color_palette"]).strip() if "color_palette" in payload else None,
    )
    return JSONResponse({"ok": True})


@router.post("/unpin")
async def unpin_chart(request: Request, chart_id: int = Form(...)):
    user = _get_portal_user(request)
    if not user:
        return _login_redirect()
    store.delete_pinned_chart(chart_id, user["id"])
    return RedirectResponse("/portal/dashboard", status_code=303)


# ══════════════════════════════════════════════════════════════════════════════
# Semantic Layer viewer (user sees metadata for their group's tables only)
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/kb", response_class=HTMLResponse)
async def portal_kb(request: Request):
    user = _get_portal_user(request)
    if not user:
        return _login_redirect()

    client     = store.get_client(user["account_id"]) or {}
    state_data = json.loads(client.get("state_data") or "{}")
    kb_dir     = state_data.get("kb_dir", "")
    schema_dir = state_data.get("schema_dir", "")
    allowed    = store.get_allowed_tables(user)
    approved_feedback, pending_feedback = store.semantic_feedback_maps(user["account_id"])

    semantic_tables = build_semantic_layer_tables(
        kb_dir=kb_dir,
        schema_dir=schema_dir,
        allowed_tables=allowed,
        approved_feedback=approved_feedback,
        pending_feedback=pending_feedback,
    )
    schemas = sorted({t["schema"] or "DEFAULT" for t in semantic_tables})
    selected_schema = (request.query_params.get("schema") or "").upper()
    if selected_schema not in schemas:
        selected_schema = schemas[0] if schemas else ""
    visible_tables = [
        t for t in semantic_tables
        if (t["schema"] or "DEFAULT").upper() == selected_schema
    ]

    return _resp(request, "portal_kb.html", {
        "user":              user,
        "client":            client,
        "semantic_tables":   semantic_tables,
        "visible_tables":    visible_tables,
        "schemas":           schemas,
        "selected_schema":   selected_schema,
        "pending_count":     store.count_semantic_field_feedback(user["account_id"]),
        "saved":             request.query_params.get("saved") == "1",
    })


@router.get("/api/semantic-feedback/updates")
async def portal_semantic_feedback_updates(request: Request):
    user = _get_portal_user(request)
    if not user:
        return JSONResponse({"ok": False, "error": "Authentication required."}, status_code=401)
    rows = store.list_recent_reviewed_semantic_feedback(
        user["account_id"],
        int(user["id"]),
        limit=50,
    )
    return JSONResponse({"ok": True, "items": rows})


@router.get("/api/query-limit-status")
async def portal_query_limit_status(request: Request):
    user = _get_portal_user(request)
    if not user:
        return JSONResponse({"ok": False, "error": "Authentication required."}, status_code=401)
    return JSONResponse({"ok": True, "query_status": _query_limit_status(user["account_id"])})


@router.get("/api/history")
async def portal_query_history(request: Request):
    """Return the last 40 successful query traces for the current user's account."""
    user = _get_portal_user(request)
    if not user:
        return JSONResponse({"ok": False, "error": "Authentication required."}, status_code=401)
    traces = store.list_answer_traces(user["account_id"], limit=40)
    items = []
    for t in traces:
        items.append({
            "id":         t.get("id"),
            "question":   t.get("question") or "",
            "sql":        t.get("sql") or "",
            "row_count":  t.get("row_count") or 0,
            "duration_ms":t.get("duration_ms") or 0,
            "created_at": (t.get("created_at") or "")[:16].replace("T", " "),
            "success":    bool(t.get("sql")),  # has SQL = answered
        })
    return JSONResponse({"ok": True, "items": items})


@router.get("/api/export-csv")
async def portal_export_csv(request: Request, trace_id: int | None = None):
    """
    Export the result of a specific trace (or most recent) as a CSV download.
    Reads the stored row data from the trace record.
    """
    import csv, io as _io
    user = _get_portal_user(request)
    if not user:
        return JSONResponse({"ok": False, "error": "Authentication required."}, status_code=401)

    trace = None
    if trace_id:
        trace = store.get_answer_trace(trace_id)
        # verify it belongs to this account
        if trace and trace.get("account_id") != user["account_id"]:
            trace = None
    if not trace:
        # fallback: most recent
        traces = store.list_answer_traces(user["account_id"], limit=1)
        trace  = traces[0] if traces else None

    if not trace:
        return JSONResponse({"ok": False, "error": "No query found to export."}, status_code=404)

    # rows are stored as JSON in the trace
    import json as _json
    raw_rows = trace.get("result_rows") or trace.get("rows") or "[]"
    if isinstance(raw_rows, str):
        try:
            rows = _json.loads(raw_rows)
        except Exception:
            rows = []
    else:
        rows = raw_rows

    if not rows:
        return JSONResponse({"ok": False, "error": "No rows in this query result."}, status_code=404)

    buf = _io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    buf.seek(0)

    question_slug = (trace.get("question") or "query")[:40].lower()
    question_slug = "".join(c if c.isalnum() else "_" for c in question_slug).strip("_")
    filename = f"querybot_{question_slug}.csv"

    from fastapi.responses import StreamingResponse as _SR
    return _SR(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/kb/feedback")
async def portal_kb_feedback(
    request: Request,
    table_fqn:          str = Form(...),
    column_name:        str = Form(...),
    suggested_meaning:  str = Form(""),
    suggested_use_case: str = Form(""),
    user_comment:       str = Form(""),
):
    user = _get_portal_user(request)
    if not user:
        return _login_redirect()

    client     = store.get_client(user["account_id"]) or {}
    state_data = json.loads(client.get("state_data") or "{}")
    approved_feedback, pending_feedback = store.semantic_feedback_maps(user["account_id"])
    tables = build_semantic_layer_tables(
        kb_dir=state_data.get("kb_dir", ""),
        schema_dir=state_data.get("schema_dir", ""),
        allowed_tables=store.get_allowed_tables(user),
        approved_feedback=approved_feedback,
        pending_feedback=pending_feedback,
    )
    found = find_semantic_field(tables, table_fqn, column_name)
    if not found:
        raise HTTPException(status_code=403, detail="Field is not available to this user.")

    table, field = found
    if not (suggested_meaning.strip() or suggested_use_case.strip() or user_comment.strip()):
        return RedirectResponse(
            f"/portal/kb?schema={table['schema']}&error=empty",
            status_code=303,
        )

    feedback_id = store.save_semantic_field_feedback(
        account_id=user["account_id"],
        portal_user_id=user["id"],
        table_fqn=table["fqn"],
        schema_name=table["schema"],
        table_name=table["table"],
        column_name=field["column"],
        current_meaning=field.get("meaning", ""),
        current_use_case=field.get("use_case", ""),
        suggested_meaning=suggested_meaning.strip(),
        suggested_use_case=suggested_use_case.strip(),
        user_comment=user_comment.strip(),
        confidence_score=int(field.get("confidence") or 0),
    )
    try:
        from core.admin_notifications import notify_semantic_feedback_changed
        await notify_semantic_feedback_changed(
            account_id=user["account_id"],
            action="created",
            feedback_id=feedback_id,
        )
    except Exception as exc:
        log.warning("Admin semantic feedback notification failed: %s", exc)
    return RedirectResponse(f"/portal/kb?schema={quote(table['schema'] or '')}&saved=1", status_code=303)

# ══════════════════════════════════════════════════════════════════════════════
# Internal Chat UI
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/chat", response_class=HTMLResponse)
async def portal_chat(request: Request):
    """Internal chat UI — enabled per client by admin."""
    user = _get_portal_user(request)
    if not user:
        return _login_redirect()

    import store as _store
    client = _store.get_client(user["account_id"]) or {}
    if not client.get("chat_ui_enabled"):
        return _resp(request, "portal_chat.html", {
            "user":    user,
            "enabled": False,
            "client":  client,
        })

    suggestions = _build_chat_suggestions(user)

    # Build the list of schemas the user has access to — used for the schema
    # selector tab bar in the chat UI.
    available_schemas = _get_available_schemas(user)
    query_status = _query_limit_status(user["account_id"])
    token_usage = _store.get_monthly_token_usage(user["account_id"], user.get("id"))
    token_usage["total_label"] = _compact_number(token_usage.get("total_tokens"))
    token_usage["input_label"] = _compact_number(token_usage.get("tokens_in"))
    token_usage["output_label"] = _compact_number(token_usage.get("tokens_out"))

    return _resp(request, "portal_chat.html", {
        "user":               user,
        "enabled":            True,
        "client":             client,
        "suggestions":        suggestions,
        "available_schemas":  available_schemas,
        "query_status":       query_status,
        "token_usage":        token_usage,
    })
