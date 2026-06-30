"""
admin/routes.py

Complete admin UI.

Pages:
  /admin               → dashboard
  /admin/setup         → first-run wizard
  /admin/login         → login
  /admin/platforms     → Zoom/Teams/Slack credentials
  /admin/databases     → Snowflake/Oracle/Azure SQL credentials
  /admin/system        → API keys, LLM models, password
  /admin/clients       → all tenants (searchable)
  /admin/clients/{id}  → detail: state, DB assign, LLM override, query limit
  /admin/clients/{id}/kb          → list/edit KB markdown files
  /admin/clients/{id}/billing     → usage + cost + CSV export
"""

import base64
import csv
import asyncio
import hashlib
import hmac
import io
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Request, Form, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import store
from store.db import get_db as _get_db
from store.database import DATABASE_URL, get_saved_pg_url, save_pg_url
from store.config_store import get_db_config
from core.llm_audit import llm_audit_scope, make_llm_audit_request_id
from core.log_export import (
    DEFAULT_EXPORT_TIME,
    DEFAULT_LOG_SCHEMA,
    get_export_state,
    is_log_export_enabled,
    provision_external_log_store,
    sync_external_logs,
    reset_egress_and_sync,
    reset_all_and_sync,
    diagnose_external_log_store,
)
from core.admin_notifications import (
    admin_notification_hub,
    notify_kb_build_changed,
    notify_semantic_feedback_changed,
    semantic_feedback_summary,
)
from core.portal_notifications import notify_portal_semantic_feedback_changed

log = logging.getLogger("querybot.admin")

# Per-account stop events for in-progress KB builds.
# Set the event to request cancellation; cleared on build start.
_kb_stop_events: dict[str, asyncio.Event] = {}

def _sync_all_log_exports_bg() -> None:
    """Push logs to all log-export-enabled DB configs. Runs in a background thread."""
    try:
        from store.config_store import list_db_configs
        for cfg in list_db_configs():
            if is_log_export_enabled(cfg.get("credentials", {})):
                try:
                    sync_external_logs(cfg)
                except Exception as _exc:
                    log.warning("Event-triggered log sync failed for db_config_id=%s: %s",
                                cfg.get("id"), _exc)
    except Exception as _exc:
        log.warning("Event-triggered log sync (enumerate configs) failed: %s", _exc)


router    = APIRouter(prefix="/admin")
templates = Jinja2Templates(
    directory=str(Path(__file__).parent / "templates")
)

# ── Custom Jinja2 filters ──────────────────────────────────────────────
import json as _json_mod

def _jinja_from_json(value, default=None):
    """Parse a JSON string to a Python object. Returns default on error."""
    if not value:
        return default if default is not None else []
    if isinstance(value, (list, dict)):
        return value
    try:
        return _json_mod.loads(value)
    except Exception:
        return default if default is not None else []

templates.env.filters["from_json"] = _jinja_from_json

_COOKIE = "querybot_session"

# ── LLM choices shown in dropdowns ───────────────────────────────────────────
QUERY_MODELS = [
    ("anthropic",    "claude-sonnet-4-6", "Claude Sonnet 4.6 — recommended"),
    ("anthropic",    "claude-haiku-4-5",  "Claude Haiku 4.5 — fast & cheap"),
    ("anthropic",    "claude-opus-4-6",   "Claude Opus 4.6 — most capable"),
    ("openai",       "gpt-4o",            "GPT-4o (OpenAI)"),
    ("openai",       "gpt-4o-mini",       "GPT-4o Mini (OpenAI) — cheapest"),
    ("azure_openai", "gpt-4o",            "GPT-4o (Azure OpenAI) — use your deployment name"),
    ("azure_openai", "gpt-4o-mini",       "GPT-4o Mini (Azure OpenAI) — use your deployment name"),
    ("azure_openai", "gpt-35-turbo",      "GPT-3.5 Turbo (Azure OpenAI)"),
]
KB_MODELS = [
    ("anthropic",    "claude-opus-4-5",  "Claude Opus 4.5 — best quality (recommended)"),
    ("anthropic",    "claude-sonnet-4-6","Claude Sonnet 4.6"),
    ("openai",       "gpt-4o",           "GPT-4o (OpenAI)"),
    ("azure_openai", "gpt-4o",           "GPT-4o (Azure) — use your deployment name"),
]


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def _session_secret() -> str:
    return os.getenv("ADMIN_SESSION_SECRET") or os.getenv("SESSION_SECRET") or "change-me-in-production"

def _sign_admin_session() -> str:
    payload = b"admin"
    sig = hmac.new(_session_secret().encode(), payload, hashlib.sha256).hexdigest()
    token = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    return f"{token}.{sig}"

def _is_auth(request: Request) -> bool:
    raw = request.cookies.get(_COOKIE, "")
    if not raw:
        return False
    try:
        token, sig = raw.split(".", 1)
        padding = "=" * (-len(token) % 4)
        payload = base64.urlsafe_b64decode((token + padding).encode())
        expected = hmac.new(_session_secret().encode(), payload, hashlib.sha256).hexdigest()
        return payload == b"admin" and hmac.compare_digest(sig, expected)
    except Exception:
        return False

def _is_admin_cookie(raw: str) -> bool:
    if not raw:
        return False
    try:
        token, sig = raw.split(".", 1)
        padding = "=" * (-len(token) % 4)
        payload = base64.urlsafe_b64decode((token + padding).encode())
        expected = hmac.new(_session_secret().encode(), payload, hashlib.sha256).hexdigest()
        return payload == b"admin" and hmac.compare_digest(sig, expected)
    except Exception:
        return False

def _is_ws_auth(websocket: WebSocket) -> bool:
    return _is_admin_cookie(websocket.cookies.get(_COOKIE, ""))

def _set_admin_cookie(resp: RedirectResponse, request: Request) -> None:
    resp.set_cookie(
        _COOKIE,
        _sign_admin_session(),
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )

def _first_run() -> bool:
    return not store.get_system("admin_password_hash")

def _resp(request, name, ctx=None):
    return templates.TemplateResponse(request=request, name=name, context=ctx or {})


def _eval_case_files(account_id: str) -> list[Path]:
    root = Path("evals") / "clients" / account_id
    if not root.exists():
        return []
    return sorted(root.glob("**/golden_questions.y*ml")) + sorted(root.glob("**/golden_questions.json"))


async def _run_default_evals_async(account_id: str, *, generate: bool = True, execute: bool = False) -> list[int]:
    """Run all configured eval case files for a client. Best-effort."""
    from evals.run import run_eval_suite

    run_ids: list[int] = []
    for case_file in _eval_case_files(account_id):
        schema = case_file.parent.name or "default"
        out_dir = Path("evals") / "reports" / account_id / schema / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        try:
            _, run_id = await run_eval_suite(
                account_id=account_id,
                schema=schema,
                cases_path=case_file,
                generate=generate,
                execute=execute,
                out_dir=out_dir,
            )
            run_ids.append(run_id)
        except Exception as exc:
            log.warning("Automatic eval failed for %s/%s: %s", account_id, case_file, exc)
    return run_ids


def _run_default_evals_background(account_id: str) -> None:
    try:
        asyncio.run(_run_default_evals_async(account_id, generate=True, execute=False))
    except Exception as exc:
        log.warning("Background eval trigger failed for %s: %s", account_id, exc)


def _safe_child_path(base_dir: str, filename: str) -> Path | None:
    if not base_dir or not filename:
        return None
    base = Path(base_dir).resolve()
    target = (base / Path(filename).name).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        return None
    return target


def _default_regulated_rules(pack: dict) -> list[dict]:
    """Seed a conservative draft policy for a regulated tenant."""
    rules: list[dict] = []
    sensitive = set(pack.get("sensitive_tags", []))
    prohibited_llm = set(pack.get("prohibited_llm_tags", []))
    for role in ("admin", "analyst"):
        for tag in pack.get("classification_tags", []):
            for action in ("query_execution", "result_release"):
                rules.append({
                    "name": f"{role}: {action} {tag}",
                    "subject_type": "role", "subject_id": role,
                    "resource_type": "classification", "resource_pattern": tag,
                    "action": action,
                    "effect": "mask" if tag in sensitive else "allow",
                    "mask_strategy": "partial" if tag in {"FINANCIAL", "PAYMENT"} else "redact",
                    "cache_ttl_seconds": int(pack.get("default_cache_ttl_seconds") or 0),
                })
            rules.append({
                "name": f"{role}: aggregate charts for {tag}",
                "subject_type": "role", "subject_id": role,
                "resource_type": "classification", "resource_pattern": tag,
                "action": "chart", "effect": "allow",
                "aggregate_only": tag in sensitive,
            })
            rules.append({
                "name": f"{role}: exports denied for {tag}",
                "subject_type": "role", "subject_id": role,
                "resource_type": "classification", "resource_pattern": tag,
                "action": "export", "effect": "deny",
                "mandatory": tag in sensitive,
            })
            rules.append({
                "name": f"{role}: LLM context {tag}",
                "subject_type": "role", "subject_id": role,
                "resource_type": "classification", "resource_pattern": tag,
                "action": "llm_context",
                "effect": "deny" if tag in prohibited_llm else "mask",
                "mask_strategy": "redact",
                "mandatory": tag in prohibited_llm,
            })
        for action in ("query_execution", "result_release", "chart", "alert", "cache_read"):
            rules.append({
                "name": f"{role}: {action} internal data",
                "subject_type": "role", "subject_id": role,
                "resource_type": "classification", "resource_pattern": "*",
                "action": action, "effect": "allow",
                "cache_ttl_seconds": int(pack.get("default_cache_ttl_seconds") or 0),
            })
    return rules


# ── Client Readiness / Health Score ──────────────────────────────────────────

def _client_health_score(account_id: str, client: dict | None = None) -> dict:
    """
    Compute the 8-component readiness score (0-100 pts).

    Components
    ----------
    1.  DB connected          10 pts
    2.  Tables selected       10 pts
    3.  Masking reviewed      10 pts  (optional; credit given once state_data contains the key)
    4.  KB ready              20 pts  (10 partial while KB_BUILDING)
    5.  Eval baseline passed  20 pts
    6.  Users assigned        10 pts
    7.  Feedback queue clear  10 pts
    8.  Query success ≥80 %   10 pts

    Returns
    -------
    dict with keys: score, max, pct, grade, color, components
    """
    if client is None:
        client = store.get_client(account_id) or {}

    state      = client.get("state", "NEW")
    state_data = json.loads(client.get("state_data") or "{}")

    components: list[dict] = []

    # 1 — DB connected
    db_id  = client.get("db_config_id")
    db_ok  = bool(db_id and store.get_db_config(db_id))
    components.append({
        "key": "db", "label": "Database connected", "points": 10,
        "earned": 10 if db_ok else 0, "ok": db_ok,
        "hint": "Assign a database in the Settings tab." if not db_ok else "",
    })

    # 2 — Tables selected for KB
    kb_tables  = state_data.get("kb_tables") or []
    tables_ok  = len(kb_tables) > 0
    components.append({
        "key": "tables", "label": "Tables selected", "points": 10,
        "earned": 10 if tables_ok else 0, "ok": tables_ok,
        "hint": "Open Schema & KB Setup → Step 2 to select tables." if not tables_ok else "",
    })

    # 3 — Masking reviewed (optional; credit once the masking_config key exists in state_data)
    masking_reviewed = "masking_config" in state_data
    components.append({
        "key": "masking", "label": "Masking reviewed", "points": 10,
        "earned": 10 if masking_reviewed else 0, "ok": masking_reviewed,
        "hint": "Open Schema & KB Setup → Field Masking to review PII fields." if not masking_reviewed else "",
    })

    # 4 — KB generated
    if state == "READY":
        kb_pts, kb_ok = 20, True
    elif state == "KB_BUILDING":
        kb_pts, kb_ok = 10, False   # partial
    else:
        kb_pts, kb_ok = 0, False
    components.append({
        "key": "kb", "label": "Knowledge Base ready", "points": 20,
        "earned": kb_pts, "ok": kb_ok,
        "partial": state == "KB_BUILDING",
        "hint": "Run 'Generate Knowledge Base' in Schema & KB Setup." if not kb_ok else "",
    })

    # 5 — Eval baseline
    eval_run  = store.latest_eval_run(account_id)
    eval_ok   = bool(eval_run and (eval_run.get("pass_count") or 0) > 0)
    components.append({
        "key": "evals", "label": "Eval baseline passed", "points": 20,
        "earned": 20 if eval_ok else 0, "ok": eval_ok,
        "hint": "Run an eval suite from the Evals tab." if not eval_ok else "",
    })

    # 6 — Users assigned
    users     = store.list_users(account_id)
    users_ok  = len(users) > 0
    components.append({
        "key": "users", "label": "Users assigned", "points": 10,
        "earned": 10 if users_ok else 0, "ok": users_ok,
        "hint": "Add portal users in the Users tab." if not users_ok else "",
    })

    # 7 — Semantic feedback queue
    pending      = store.count_semantic_field_feedback(account_id)
    feedback_ok  = pending == 0
    components.append({
        "key": "feedback", "label": "Feedback queue clear", "points": 10,
        "earned": 10 if feedback_ok else 0, "ok": feedback_ok,
        "partial": not feedback_ok,
        "hint": f"{pending} item{'s' if pending != 1 else ''} need review in the KB tab." if not feedback_ok else "",
    })

    # 8 — Query success rate ≥ 80 %
    stats      = store.get_query_stats(account_id)
    total_q    = stats.get("total") or 0
    succeeded  = stats.get("succeeded") or 0
    if total_q > 0:
        rate       = succeeded / total_q
        success_pts = 10 if rate >= 0.8 else (5 if rate >= 0.5 else 0)
        success_ok  = rate >= 0.8
        rate_label  = f"{rate:.0%}"
    else:
        success_pts, success_ok, rate_label = 0, False, "No data"
    components.append({
        "key": "success_rate", "label": "Query success ≥80 %", "points": 10,
        "earned": success_pts, "ok": success_ok,
        "hint": f"Current rate: {rate_label}. Review failed queries in the Query Log tab." if not success_ok else "",
    })

    score = sum(c["earned"] for c in components)
    pct   = score  # max is 100

    if pct >= 90:
        grade, color = "A", "green"
    elif pct >= 75:
        grade, color = "B", "green"
    elif pct >= 50:
        grade, color = "C", "amber"
    else:
        grade, color = "D", "red"

    return {
        "score":      score,
        "max":        100,
        "pct":        pct,
        "grade":      grade,
        "color":      color,
        "components": components,
    }


# Realtime admin notifications.
@router.get("/api/semantic-feedback/pending-count")
async def semantic_feedback_pending_count_api(request: Request):
    if not _is_auth(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return JSONResponse(semantic_feedback_summary())


@router.websocket("/ws/notifications")
async def admin_notifications_ws(websocket: WebSocket):
    if not _is_ws_auth(websocket):
        await websocket.close(code=4401)
        return

    await admin_notification_hub.connect(websocket)
    try:
        await websocket.send_json({
            "type": "semantic_feedback_pending",
            "action": "snapshot",
            "summary": semantic_feedback_summary(),
        })
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.debug("Admin notification socket closed: %s", exc)
    finally:
        await admin_notification_hub.disconnect(websocket)


# ── Login / logout ────────────────────────────────────────────────────────────

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return _resp(request, "login.html", {"error": None})

@router.post("/login")
async def login_submit(request: Request, password: str = Form(...)):
    stored = store.get_system("admin_password_hash", "")
    if not stored:
        return RedirectResponse("/admin/setup", status_code=303)
    if _hash(password) != stored:
        return _resp(request, "login.html", {"error": "Incorrect password"})
    resp = RedirectResponse("/admin", status_code=303)
    _set_admin_cookie(resp, request)
    return resp

@router.get("/logout")
async def logout():
    resp = RedirectResponse("/admin/login", status_code=303)
    resp.delete_cookie(_COOKIE)
    return resp


# ── First-time setup ──────────────────────────────────────────────────────────

@router.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request):
    if not _first_run():
        return RedirectResponse("/admin", status_code=303)
    return _resp(request, "setup.html", {
        "error": None,
        "query_models": QUERY_MODELS,
        "kb_models": KB_MODELS,
    })

@router.post("/setup")
async def setup_submit(
    request: Request,
    admin_password: str = Form(...),
    anthropic_key:  str = Form(""),
    openai_key:     str = Form(""),
    default_provider: str = Form("anthropic"),
    default_model:  str = Form("claude-sonnet-4-6"),
    kb_model:       str = Form("claude-opus-4-5"),
):
    if len(admin_password) < 8:
        return _resp(request, "setup.html", {
            "error": "Password must be at least 8 characters",
            "query_models": QUERY_MODELS, "kb_models": KB_MODELS,
        })
    pw_hash = _hash(admin_password)
    store.set_system("admin_password_hash",   pw_hash)
    store.set_system("default_llm_provider",  default_provider)
    store.set_system("default_llm_model",     default_model)
    store.set_system("kb_llm_model",          kb_model)
    if anthropic_key:
        store.set_system("anthropic_api_key", anthropic_key)
    if openai_key:
        store.set_system("openai_api_key",    openai_key)
    resp = RedirectResponse("/admin", status_code=303)
    _set_admin_cookie(resp, request)
    return resp


# ── Dashboard ─────────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    if _first_run():
        return RedirectResponse("/admin/setup", status_code=303)
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)

    clients   = store.list_clients()
    platforms = store.list_platforms()
    dbs       = store.list_db_configs()
    stats     = store.get_query_stats()

    plat_map = {p["id"]: p["name"] for p in platforms}
    db_map   = {d["id"]: f"{d['name']} ({d['label']})" for d in dbs}
    for c in clients:
        c["platform_name"] = plat_map.get(c["platform_config_id"], "—")
        c["db_name"]       = db_map.get(c["db_config_id"], "Not assigned")

    return _resp(request, "dashboard.html", {
        "clients": clients, "platforms": platforms,
        "dbs": dbs, "stats": stats,
    })


# ── System config ─────────────────────────────────────────────────────────────

@router.get("/system", response_class=HTMLResponse)
async def system_page(request: Request):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    cfg = store.get_all_system()
    masked = {k: (store.mask(v) if "key" in k or "password" in k else v)
              for k, v in cfg.items()}

    # Database backend state
    saved_pg_url  = get_saved_pg_url()
    active_pg_url = DATABASE_URL  # what is actually running right now
    active_backend = "postgres" if active_pg_url.startswith(("postgresql://", "postgres://")) else "sqlite"
    saved_backend  = "postgres" if saved_pg_url else "sqlite"
    restart_needed = active_backend != saved_backend or (
        active_backend == "postgres" and active_pg_url != saved_pg_url
    )
    # Show host only (strip credentials) for display
    def _pg_host(url: str) -> str:
        import re as _re
        m = _re.search(r"@([^/]+)", url)
        return m.group(1) if m else url[:40]

    return _resp(request, "system.html", {
        "cfg": masked,
        "query_models":   QUERY_MODELS,
        "kb_models":      KB_MODELS,
        "saved":          request.query_params.get("saved"),
        "error":          request.query_params.get("error"),
        # database backend
        "active_backend":  active_backend,
        "saved_backend":   saved_backend,
        "saved_pg_url":    store.mask(saved_pg_url) if saved_pg_url else "",
        "active_pg_host":  _pg_host(active_pg_url) if active_pg_url else "",
        "restart_needed":  restart_needed,
    })

@router.get("/system/azure-deployments")
async def azure_deployments_api(request: Request):
    """Fetch available model deployments from the saved Azure OpenAI resource.

    Tries the configured API version then falls back through a list of known
    versions.  Handles both *.openai.azure.com and *.cognitiveservices.azure.com
    endpoints — the latter requires newer preview versions and may still not
    expose the listing API on older multi-service accounts.
    """
    if not _is_auth(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    cfg = store.get_all_system()
    endpoint    = (cfg.get("azure_openai_endpoint") or "").strip().rstrip("/")
    api_key     = (cfg.get("azure_openai_api_key")  or "").strip()
    api_version = (cfg.get("azure_openai_api_version") or "2024-02-01").strip()

    if not endpoint:
        return JSONResponse({"ok": False, "error": "No endpoint saved yet — fill in the Endpoint URL and save first."}, status_code=400)
    if not api_key:
        return JSONResponse({"ok": False, "error": "No API key saved yet — fill in the Azure API key and save first."}, status_code=400)

    # Fix 1 — auto-add missing scheme; reject http://
    if not endpoint.startswith(("http://", "https://")):
        endpoint = "https://" + endpoint
    if endpoint.startswith("http://"):
        return JSONResponse({"ok": False, "error": "Endpoint must use https://, not http://."}, status_code=400)

    # Fix 2 — strip paths that look like accidentally-pasted /openai/* URLs,
    # but KEEP /api/projects/* paths required for Azure AI Foundry project endpoints.
    from urllib.parse import urlparse, urlunparse
    _p = urlparse(endpoint)
    _path = _p.path.rstrip("/")
    if _path.startswith("/openai"):
        endpoint = urlunparse((_p.scheme, _p.netloc, "", "", "", ""))
    else:
        endpoint = urlunparse((_p.scheme, _p.netloc, _path, "", "", ""))

    _FALLBACK_VERSIONS = [
        "2025-01-01-preview",
        "2024-12-01-preview",
        "2024-10-21",
        "2024-05-01-preview",
        "2023-09-01-preview",
        "2023-05-15",
    ]
    versions_to_try = [api_version] + [v for v in _FALLBACK_VERSIONS if v != api_version]

    import httpx
    headers = {"api-key": api_key}
    last_error = ""
    async with httpx.AsyncClient(timeout=12) as client:
        for version in versions_to_try:
            url = f"{endpoint}/openai/deployments?api-version={version}"
            try:
                resp = await client.get(url, headers=headers)
            except httpx.TimeoutException:
                return JSONResponse({"ok": False, "error": "Request timed out — check the endpoint URL is correct."}, status_code=400)
            except Exception as exc:
                return JSONResponse({"ok": False, "error": f"Could not reach Azure: {exc}"}, status_code=400)

            if resp.status_code == 401:
                return JSONResponse({"ok": False, "error": "API key rejected (401) — check your Azure OpenAI key in Step 1."}, status_code=400)

            # Fix 3 — 403 is a permissions/firewall issue, not a version issue; stop immediately
            if resp.status_code == 403:
                return JSONResponse({"ok": False, "error": (
                    "Permission denied (403) — the key is valid but the request was blocked. "
                    "Check: (1) Azure Portal → your resource → Networking — add this server's IP if "
                    "public access is restricted, or (2) the key may lack the "
                    "'Cognitive Services OpenAI Contributor' RBAC role."
                )}, status_code=400)

            # Fix 4 — 429 rate limit applies across all versions; stop immediately
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After", "")
                msg = "Rate limited by Azure (429) — too many requests."
                if retry_after:
                    msg += f" Retry after {retry_after}s."
                return JSONResponse({"ok": False, "error": msg}, status_code=400)

            if resp.status_code == 404:
                last_error = f"404 on api-version={version}"
                continue

            if not resp.is_success:
                try:
                    detail = resp.json().get("error", {}).get("message", resp.text[:200])
                except Exception:
                    detail = resp.text[:200]
                last_error = f"HTTP {resp.status_code}: {detail}"
                continue

            # Fix 5 — wrap JSON parse so a proxy HTML page on a 200 doesn't cause a 500
            try:
                data = resp.json()
            except Exception:
                return JSONResponse({"ok": False, "error": (
                    f"Azure returned HTTP {resp.status_code} but the response was not valid JSON. "
                    "The endpoint URL may point to a proxy or load balancer — verify it in Azure Portal."
                )}, status_code=400)

            deployments = sorted(
                [
                    {
                        "name":   d.get("id", ""),
                        "model":  d.get("model", ""),
                        "status": d.get("status", "unknown"),
                    }
                    for d in (data.get("data") or [])
                    if d.get("id")
                ],
                key=lambda d: d["name"],
            )
            return JSONResponse({
                "ok": True,
                "deployments": deployments,
                "api_version_used": version,
            })

    # All versions exhausted.  For *.cognitiveservices.azure.com this is expected
    # on older multi-service accounts — they require manual name entry.
    is_cogs = "cognitiveservices.azure.com" in endpoint
    detail_hint = (
        "Azure AI Services / Cognitive Services resources (cognitiveservices.azure.com) "
        "only expose the deployment listing API on newer accounts. "
        "If this is a fresh resource and fetch still fails, the account may require "
        "management-plane access (subscription + AAD token) which is not supported here."
        if is_cogs else
        "All known API versions returned errors."
    )
    return JSONResponse({
        "ok": False,
        "error": (
            f"Could not list deployments automatically. {detail_hint} "
            "Use Step 3 below to type the deployment name manually — "
            "find it in Azure AI Foundry (ai.azure.com) → Models + endpoints. "
            f"(Last error: {last_error})"
        ),
    }, status_code=400)


@router.get("/system/test-connection")
async def test_llm_connection(request: Request, provider: str = ""):
    """Quick liveness check for each LLM provider using saved credentials.

    Returns {"ok": true, "model": "..."} on success or {"ok": false, "error": "..."}.
    For Azure OpenAI with a deployment name configured, sends a 1-token completion.
    Without a deployment name, verifies the key + endpoint are accepted by Azure.
    """
    if not _is_auth(request):
        raise HTTPException(status_code=401, detail="Not authenticated")

    import httpx
    cfg = store.get_all_system()

    # ── Anthropic ────────────────────────────────────────────────────────────
    if provider == "anthropic":
        api_key = (cfg.get("anthropic_api_key") or "").strip()
        if not api_key:
            return JSONResponse({"ok": False, "error": "No API key saved — enter it in Step 1 and save."})
        try:
            async with httpx.AsyncClient(timeout=12) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={"model": "claude-haiku-4-5-20251001", "max_tokens": 1,
                          "messages": [{"role": "user", "content": "hi"}]},
                )
            if resp.status_code == 401:
                return JSONResponse({"ok": False, "error": "API key rejected — check your Anthropic key."})
            if resp.is_success or resp.status_code in (400, 529):
                return JSONResponse({"ok": True, "model": "claude-haiku-4-5-20251001"})
            return JSONResponse({"ok": False, "error": f"HTTP {resp.status_code}"})
        except httpx.TimeoutException:
            return JSONResponse({"ok": False, "error": "Request timed out reaching api.anthropic.com."})
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)})

    # ── OpenAI ───────────────────────────────────────────────────────────────
    if provider == "openai":
        api_key = (cfg.get("openai_api_key") or "").strip()
        if not api_key:
            return JSONResponse({"ok": False, "error": "No API key saved — enter it in Step 1 and save."})
        try:
            async with httpx.AsyncClient(timeout=12) as client:
                resp = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            if resp.status_code == 401:
                return JSONResponse({"ok": False, "error": "API key rejected — check your OpenAI key."})
            if resp.is_success:
                return JSONResponse({"ok": True, "model": "GPT-4o available"})
            return JSONResponse({"ok": False, "error": f"HTTP {resp.status_code}"})
        except httpx.TimeoutException:
            return JSONResponse({"ok": False, "error": "Request timed out reaching api.openai.com."})
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)})

    # ── Azure OpenAI ─────────────────────────────────────────────────────────
    if provider == "azure_openai":
        endpoint   = (cfg.get("azure_openai_endpoint")    or "").strip().rstrip("/")
        api_key    = (cfg.get("azure_openai_api_key")     or "").strip()
        api_version = (cfg.get("azure_openai_api_version") or "2024-02-01").strip()
        deployment  = (cfg.get("azure_query_deployment_name") or "").strip()

        if not endpoint:
            return JSONResponse({"ok": False, "error": "No endpoint saved — fill in the Endpoint URL and save."})
        if not api_key:
            return JSONResponse({"ok": False, "error": "No API key saved — fill in the Azure API key and save."})

        if not endpoint.startswith(("http://", "https://")):
            endpoint = "https://" + endpoint
        from urllib.parse import urlparse, urlunparse
        _p = urlparse(endpoint)
        _path = _p.path.rstrip("/")
        if _path.startswith("/openai"):
            endpoint = urlunparse((_p.scheme, _p.netloc, "", "", "", ""))
        else:
            endpoint = urlunparse((_p.scheme, _p.netloc, _path, "", "", ""))

        is_foundry = "services.ai.azure.com" in endpoint

        try:
            async with httpx.AsyncClient(timeout=12) as client:
                if deployment:
                    if is_foundry:
                        # Azure AI Foundry endpoints use the v1 inference API:
                        # no api-version query param, no deployment in URL — model in body.
                        url = f"{endpoint}/openai/v1/chat/completions"
                        resp = await client.post(
                            url,
                            headers={"api-key": api_key, "content-type": "application/json"},
                            json={"model": deployment, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                        )
                    else:
                        # Classic Azure OpenAI: deployment in URL, api-version in query
                        url = (f"{endpoint}/openai/deployments/{deployment}"
                               f"/chat/completions?api-version={api_version}")
                        resp = await client.post(
                            url,
                            headers={"api-key": api_key, "content-type": "application/json"},
                            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                        )
                else:
                    # No deployment name yet — verify key + endpoint are accepted
                    check_url = (f"{endpoint}/openai/v1/models" if is_foundry
                                 else f"{endpoint}/openai/deployments?api-version={api_version}")
                    resp = await client.get(check_url, headers={"api-key": api_key})

            if resp.status_code == 401:
                return JSONResponse({"ok": False, "error": "API key rejected (401) — check your Azure OpenAI key."})
            if resp.status_code == 403:
                return JSONResponse({"ok": False, "error": (
                    "Permission denied (403) — key accepted but access blocked. "
                    "Check the resource firewall (Azure Portal → Networking) or RBAC role."
                )})
            if resp.status_code == 404 and not deployment:
                # cognitiveservices.azure.com returns 404 on the listing endpoint —
                # that is expected; the key was accepted (no 401) so credentials are valid.
                return JSONResponse({"ok": True, "model": "Credentials accepted — save a deployment name to test completion"})
            if resp.is_success:
                label = f"deployment: {deployment}" if deployment else "endpoint reachable"
                return JSONResponse({"ok": True, "model": label})
            try:
                detail = resp.json().get("error", {}).get("message", resp.text[:150])
            except Exception:
                detail = resp.text[:150]
            return JSONResponse({"ok": False, "error": f"HTTP {resp.status_code}: {detail}"})

        except httpx.TimeoutException:
            return JSONResponse({"ok": False, "error": "Request timed out — check the endpoint URL."})
        except Exception as exc:
            return JSONResponse({"ok": False, "error": f"Could not reach Azure: {exc}"})

    return JSONResponse({"ok": False, "error": f"Unknown provider '{provider}'."})


@router.get("/system/test-pg")
async def test_pg_connection(request: Request, url: str = ""):
    """Test a PostgreSQL connection string without saving it."""
    if not _is_auth(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not url or url.startswith("•"):
        # No URL provided — test the currently saved one
        url = get_saved_pg_url()
    if not url:
        return JSONResponse({"ok": False, "error": "No PostgreSQL URL configured yet."})
    if not url.startswith(("postgresql://", "postgres://")):
        return JSONResponse({"ok": False, "error": "URL must start with postgresql:// or postgres://"})
    try:
        import psycopg2
    except ImportError:
        return JSONResponse({
            "ok": False,
            "error": "psycopg2 is not installed. Run: pip install psycopg2-binary",
        })
    try:
        conn = psycopg2.connect(url, connect_timeout=5)
        cur = conn.cursor()
        cur.execute("SELECT version()")
        version = (cur.fetchone() or [""])[0].split(",")[0]  # e.g. "PostgreSQL 16.2"
        conn.close()
        return JSONResponse({"ok": True, "version": version})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)[:300]})


@router.get("/system/qdrant-status")
async def qdrant_status(request: Request):
    """Return Qdrant running status and detected manager — no auth needed (read-only health check)."""
    from core.qdrant_manager import get_status, detect_manager, manager_label
    status  = get_status()
    manager = detect_manager()
    return JSONResponse({
        "running": status["running"],
        "version": status["version"],
        "manager": manager,
        "manager_label": manager_label(manager),
    })


@router.post("/system/qdrant-restart")
async def qdrant_restart(request: Request):
    """Restart Qdrant using whichever manager is detected on this host."""
    if not _is_auth(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    from core.qdrant_manager import detect_manager, restart
    manager = detect_manager()
    ok, msg = restart(manager)
    return JSONResponse({"ok": ok, "message": msg, "manager": manager})


@router.get("/ping")
async def admin_ping():
    """Liveness check — polled by the UI after a restart to detect when the new process is up."""
    return JSONResponse({"ok": True})


@router.post("/system/restart")
async def system_restart(request: Request):
    """Schedule a graceful restart via os.execv(). Cross-platform: Linux exec-replace, Windows spawn+exit."""
    if not _is_auth(request):
        raise HTTPException(status_code=401, detail="Not authenticated")

    def _do_restart():
        time.sleep(1.5)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    threading.Thread(target=_do_restart, daemon=True).start()
    return JSONResponse({"ok": True})


@router.post("/system")
async def system_save(
    request: Request,
    anthropic_key:           str = Form(""),
    openai_key:              str = Form(""),
    azure_openai_key:        str = Form(""),
    azure_openai_endpoint:   str = Form(""),
    azure_api_version:       str = Form(""),
    azure_query_deployment:  str = Form(""),
    azure_kb_deployment:     str = Form(""),
    default_provider:        str = Form(""),
    default_model:           str = Form(""),
    kb_model:                str = Form(""),
    database_backend:        str = Form(""),
    database_url:            str = Form(""),
):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    # Only update keys if they were actually changed (not blank or masked)
    if anthropic_key and not anthropic_key.startswith("•"):
        store.set_system("anthropic_api_key", anthropic_key)
    if openai_key and not openai_key.startswith("•"):
        store.set_system("openai_api_key", openai_key)
    if azure_openai_key and not azure_openai_key.startswith("•"):
        store.set_system("azure_openai_api_key", azure_openai_key)
    if azure_openai_endpoint.strip():
        store.set_system("azure_openai_endpoint", azure_openai_endpoint.strip())
    if azure_api_version.strip():
        store.set_system("azure_openai_api_version", azure_api_version.strip())
    if azure_query_deployment.strip():
        store.set_system("azure_query_deployment_name", azure_query_deployment.strip())
    if azure_kb_deployment.strip():
        store.set_system("azure_kb_deployment_name", azure_kb_deployment.strip())
    if default_provider:
        store.set_system("default_llm_provider", default_provider)
    if default_model:
        store.set_system("default_llm_model", default_model)
    if kb_model:
        store.set_system("kb_llm_model", kb_model)
    # Database backend: write to data/pg_url (empty = SQLite, URL = PostgreSQL)
    if database_backend == "sqlite":
        save_pg_url("")
    elif database_backend == "postgres" and database_url.strip() and not database_url.startswith("•"):
        save_pg_url(database_url.strip())
    return RedirectResponse("/admin/system?saved=1", status_code=303)

@router.post("/system/password")
async def system_password(
    request: Request,
    new_password:     str = Form(...),
    confirm_password: str = Form(...),
):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    cfg = store.get_all_system()
    masked = {k: (store.mask(v) if "key" in k or "password" in k else v)
              for k, v in cfg.items()}
    if new_password != confirm_password:
        return _resp(request, "system.html", {
            "cfg": masked, "query_models": QUERY_MODELS, "kb_models": KB_MODELS,
            "error": "Passwords do not match",
        })
    if len(new_password) < 8:
        return _resp(request, "system.html", {
            "cfg": masked, "query_models": QUERY_MODELS, "kb_models": KB_MODELS,
            "error": "Password must be at least 8 characters",
        })
    new_hash = _hash(new_password)
    store.set_system("admin_password_hash", new_hash)
    resp = RedirectResponse("/admin/system?saved=1", status_code=303)
    _set_admin_cookie(resp, request)
    return resp


# ── Platform configs ──────────────────────────────────────────────────────────

@router.get("/platforms", response_class=HTMLResponse)
async def platforms_page(request: Request):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    platforms = store.list_platforms()
    for p in platforms:
        p["creds_masked"] = {k: store.mask(v) for k, v in p["credentials"].items()}
    return _resp(request, "platforms.html", {
        "platforms": platforms,
        "platform_fields": store.PLATFORM_FIELDS,
        "saved": request.query_params.get("saved"),
        "error": request.query_params.get("error"),
    })

@router.post("/platforms/save")
async def platform_save(request: Request):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    form          = await request.form()
    platform_type = (form.get("platform_type") or "").strip()
    name          = (form.get("name") or "").strip()
    platform_id   = int(form.get("platform_id")) if form.get("platform_id") else None

    required = store.PLATFORM_FIELDS.get(platform_type, [])
    creds = {}
    for field in required:
        val = (form.get(field) or "").strip()
        if val and not val.startswith("•"):
            creds[field] = val
        elif platform_id:
            existing = store.get_platform(platform_id)
            creds[field] = (existing or {}).get("credentials", {}).get(field, "")
    try:
        store.save_platform(platform_type, name, creds, platform_id)
        return RedirectResponse("/admin/platforms?saved=1", status_code=303)
    except ValueError as e:
        # Re-render with form values intact so the user doesn't have to retype everything
        platforms = store.list_platforms()
        for p in platforms:
            p["creds_masked"] = {k: store.mask(v) for k, v in p["credentials"].items()}
        return _resp(request, "platforms.html", {
            "platforms":       platforms,
            "platform_fields": store.PLATFORM_FIELDS,
            "error":           str(e),
            "prefill": {
                "platform_type": platform_type,
                "name":          name,
                **creds,
            },
        })

@router.post("/platforms/delete")
async def platform_delete(request: Request, platform_id: int = Form(...)):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    store.delete_platform(platform_id)
    return RedirectResponse("/admin/platforms", status_code=303)


# ── Database configs ──────────────────────────────────────────────────────────

def _normalize_table_ref(value) -> str:
    """Normalize a schema/table reference without guessing missing parts."""
    parts = [
        p.strip().strip('"').strip("'").strip("[]")
        for p in str(value or "").split(".")
        if p.strip().strip('"').strip("'").strip("[]")
    ]
    return ".".join(parts).upper()


def _parse_selected_schema_tables(raw) -> list[str]:
    """
    Parse selected KB table refs from JSON, CSV, or newline text.

    The admin schema pickers store fully qualified refs such as
    DB.SCHEMA.TABLE so non-default schemas can be discovered later.
    """
    if not raw:
        return []

    data = raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except Exception:
            data = raw.replace("\r", "\n").replace(",", "\n").split("\n")

    if not isinstance(data, (list, tuple, set)):
        return []

    seen: set[str] = set()
    selected: list[str] = []
    for item in data:
        ref = _normalize_table_ref(item)
        if ref and ref not in seen:
            seen.add(ref)
            selected.append(ref)
    return selected


def _selected_schemas_from_tables(tables: list[str]) -> list[str]:
    seen: set[str] = set()
    schemas: list[str] = []
    for ref in tables:
        parts = ref.split(".")
        if len(parts) >= 3:
            schema_ref = ".".join(parts[:2])
        elif len(parts) == 2:
            schema_ref = parts[0]
        else:
            continue
        if schema_ref not in seen:
            seen.add(schema_ref)
            schemas.append(schema_ref)
    return schemas


def _db_connection_error_message(db_type: str, err: Exception | str) -> str:
    raw = str(err)
    low = raw.lower()
    if db_type == "azure_sql":
        if "can't open lib" in low or "odbc driver 18" in low and "file not found" in low:
            return (
                "Azure SQL ODBC Driver 18 is not available to the app process. "
                "Install msodbcsql18 on the VM, then restart QueryBot. Raw error: "
                f"{raw}"
            )
        if "hyt00" in low or "login timeout" in low or "timed out" in low:
            return (
                "Azure SQL login timed out. The VM could not reach the SQL server in time. "
                "Check Azure SQL firewall rules, private endpoint/VNet access, server name, "
                "port 1433, SQL authentication, and whether a serverless database is paused. "
                f"Raw error: {raw}"
            )
    return f"Connection failed: {raw}"


def _db_credentials_from_form(form) -> tuple[str, str, int | None, dict]:
    db_type = (form.get("db_type") or "").strip()
    name    = (form.get("name") or "").strip()
    db_id   = int(form.get("db_id")) if form.get("db_id") else None

    def g(k): return (form.get(k) or "").strip()

    if db_type == "snowflake":
        creds = {"account": g("sf_account"), "user": g("sf_user"),
                 "password": g("sf_password"), "warehouse": g("sf_warehouse"),
                 "database": g("sf_database"), "schema": g("sf_schema") or "PUBLIC",
                 "role": g("sf_role")}
    elif db_type == "oracle":
        creds = {"user": g("ora_user"), "password": g("ora_password"),
                 "dsn": g("ora_dsn"), "schema": g("ora_schema")}
    elif db_type == "azure_sql":
        creds = {"server": g("az_server"), "database": g("az_database"),
                 "user": g("az_user"), "password": g("az_password"),
                 "schema": g("az_schema") or "dbo",
                 "driver": g("az_driver") or "ODBC Driver 18 for SQL Server"}
    else:
        creds = {}
    if creds:
        creds.update({
            "log_export_enabled": "1" if form.get("log_export_enabled") else "0",
            "log_schema": g("log_schema") or DEFAULT_LOG_SCHEMA,
            "log_export_time": g("log_export_time") or DEFAULT_EXPORT_TIME,
        })
        selected_tables = _parse_selected_schema_tables(form.get("selected_schema_tables"))
        if selected_tables:
            creds["selected_schema_tables"] = selected_tables
            creds["selected_schemas"] = _selected_schemas_from_tables(selected_tables)
    return db_type, name, db_id, creds


def _preserve_existing_db_secret_values(db_id: int | None, creds: dict) -> dict:
    if not db_id:
        return creds
    existing = store.get_db_config(db_id)
    if existing:
        for k, v in creds.items():
            if not v:
                creds[k] = existing["credentials"].get(k, "")
        if "selected_schema_tables" not in creds and existing["credentials"].get("selected_schema_tables"):
            creds["selected_schema_tables"] = existing["credentials"]["selected_schema_tables"]
            creds["selected_schemas"] = existing["credentials"].get("selected_schemas", [])
    return creds


def _provision_log_store_background(db_id: int) -> None:
    raw = store.get_db_config(db_id)
    if not raw:
        return
    try:
        provision_external_log_store(raw)
    except Exception as e:
        log.warning("External log table provisioning failed for db_config_id=%s: %s", db_id, e)


@router.get("/databases", response_class=HTMLResponse)
async def databases_page(request: Request):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    dbs = store.list_db_configs()
    for d in dbs:
        creds = d["credentials"]
        d["creds_masked"] = {
            k: (store.mask(v) if k == "password" else v)
            for k, v in creds.items()
        }
        d["log_export_enabled"] = is_log_export_enabled(creds)
        d["log_schema"] = creds.get("log_schema") or DEFAULT_LOG_SCHEMA
        d["log_export_time"] = creds.get("log_export_time") or DEFAULT_EXPORT_TIME
        d["log_export_state"] = get_export_state(int(d["id"])) if d["log_export_enabled"] else None
        d["selected_schema_tables"] = _parse_selected_schema_tables(creds.get("selected_schema_tables"))
        d["selected_schema_count"] = len(d["selected_schema_tables"])
        # Non-secret credentials for the Edit button data attributes (no passwords)
        db_type = d["db_type"]
        if db_type == "snowflake":
            d["edit_creds"] = {
                "account":   creds.get("account",   ""),
                "warehouse": creds.get("warehouse",  ""),
                "database":  creds.get("database",  ""),
                "schema":    creds.get("schema",    "PUBLIC"),
                "user":      creds.get("user",      ""),
                "role":      creds.get("role",      ""),
            }
        elif db_type == "oracle":
            d["edit_creds"] = {
                "dsn":    creds.get("dsn",    ""),
                "user":   creds.get("user",   ""),
                "schema": creds.get("schema", ""),
            }
        else:  # azure_sql
            d["edit_creds"] = {
                "server":   creds.get("server",   ""),
                "database": creds.get("database", ""),
                "schema":   creds.get("schema",   "dbo"),
                "user":     creds.get("user",     ""),
                "driver":   creds.get("driver",   "ODBC Driver 18 for SQL Server"),
            }
        d["edit_log"] = {
            "enabled": d["log_export_enabled"],
            "schema":  d["log_schema"],
            "time":    d["log_export_time"],
        }
    return _resp(request, "databases.html", {
        "dbs": dbs,
        "saved": request.query_params.get("saved"),
        "error": request.query_params.get("error"),
    })

@router.post("/databases/save")
async def database_save(request: Request, bg: BackgroundTasks):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    form    = await request.form()
    db_type, name, db_id, creds = _db_credentials_from_form(form)

    if not db_type:
        return RedirectResponse("/admin/databases?error=Please+select+a+database+type",
                                status_code=303)
    if not name:
        return RedirectResponse("/admin/databases?error=Please+enter+a+friendly+name",
                                status_code=303)

    if db_type not in store.DB_REQUIRED_FIELDS:
        return RedirectResponse(f"/admin/databases?error=Unknown+type+{db_type}",
                                status_code=303)

    # Preserve existing encrypted values for blank fields when editing
    creds = _preserve_existing_db_secret_values(db_id, creds)

    try:
        saved_id = store.save_db_config(db_type, name, creds, db_id)
        if is_log_export_enabled(creds):
            bg.add_task(_provision_log_store_background, saved_id)
            saved_msg = "Database saved. Log table provisioning started."
        else:
            saved_msg = "Database saved successfully."
        return RedirectResponse(f"/admin/databases?saved={quote(saved_msg)}", status_code=303)
    except ValueError as e:
        return RedirectResponse(f"/admin/databases?error={quote(str(e))}", status_code=303)


@router.post("/databases/test")
async def database_test(request: Request):
    if not _is_auth(request):
        return JSONResponse({"status": "error", "message": "Not authenticated"}, status_code=401)

    form = await request.form()
    db_type, _name, db_id, creds = _db_credentials_from_form(form)
    if db_type not in store.DB_REQUIRED_FIELDS:
        return JSONResponse(
            {"status": "error", "message": "Please select a database type first."},
            status_code=400,
        )
    creds = _preserve_existing_db_secret_values(db_id, creds)

    missing = [field for field in store.DB_REQUIRED_FIELDS[db_type] if not creds.get(field)]
    if missing:
        return JSONResponse(
            {"status": "error", "message": "Missing required field(s): " + ", ".join(missing)},
            status_code=400,
        )

    try:
        from core.schema import test_connection
        loop = asyncio.get_running_loop()
        details = await asyncio.wait_for(
            loop.run_in_executor(None, test_connection, creds, db_type),
            # 25 s — comfortably above the 15 s ODBC connection timeout so
            # the driver can return a real error rather than us cancelling it.
            timeout=25,
        )
        return JSONResponse({
            "status": "ok",
            "message": "Connection successful.",
            "details": details,
        })
    except asyncio.TimeoutError:
        return JSONResponse({
            "status": "error",
            "message": _db_connection_error_message(
                db_type,
                "Connection test timed out (25 s). "
                "Check the server name, that the host is reachable from this VM, "
                "and that the firewall allows inbound connections from the VM's IP address.",
            ),
        })
    except Exception as e:
        log.warning("DB connection test failed (%s): %s", db_type, e)
        return JSONResponse({
            "status": "error",
            "message": _db_connection_error_message(db_type, e),
        })


@router.post("/databases/discover-schema")
async def database_discover_schema(request: Request):
    """Return a schema tree from the unsaved database form values."""
    if not _is_auth(request):
        return JSONResponse({"status": "error", "message": "Not authenticated"}, status_code=401)

    form = await request.form()
    db_type, _name, db_id, creds = _db_credentials_from_form(form)
    if db_type not in store.DB_REQUIRED_FIELDS:
        return JSONResponse(
            {"status": "error", "message": "Please select a database type first."},
            status_code=400,
        )
    creds = _preserve_existing_db_secret_values(db_id, creds)

    missing = [field for field in store.DB_REQUIRED_FIELDS[db_type] if not creds.get(field)]
    if missing:
        return JSONResponse(
            {"status": "error", "message": "Missing required field(s): " + ", ".join(missing)},
            status_code=400,
        )

    try:
        from core.schema_discovery import discover_schema_tree
        tree = await discover_schema_tree(db_type, creds, timeout_seconds=45)
        return JSONResponse({"status": "ok", "tree": tree})
    except TimeoutError:
        return JSONResponse({
            "status": "error",
            "message": _db_connection_error_message(db_type, "Schema discovery timed out."),
        })
    except Exception as e:
        log.warning("DB schema discovery failed (%s): %s", db_type, e)
        return JSONResponse({
            "status": "error",
            "message": _db_connection_error_message(db_type, e),
        })


@router.post("/databases/{db_id}/logs/provision")
async def database_logs_provision(request: Request, db_id: int):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    raw = store.get_db_config(db_id)
    if not raw:
        return RedirectResponse(
            f"/admin/databases?error={quote('Database connection not found')}",
            status_code=303,
        )
    try:
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, provision_external_log_store, raw),
            timeout=90,
        )
        return RedirectResponse(
            f"/admin/databases?saved={quote('Log tables ready in ' + result['schema'])}",
            status_code=303,
        )
    except asyncio.TimeoutError:
        return RedirectResponse(
            f"/admin/databases?error={quote('Log table provisioning timed out')}",
            status_code=303,
        )
    except Exception as e:
        log.warning("External log table provisioning failed for db_config_id=%s: %s", db_id, e)
        return RedirectResponse(
            f"/admin/databases?error={quote('Log table provisioning failed: ' + str(e)[:160])}",
            status_code=303,
        )


@router.post("/databases/{db_id}/logs/sync")
async def database_logs_sync(request: Request, db_id: int):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    raw = store.get_db_config(db_id)
    if not raw:
        return RedirectResponse(
            f"/admin/databases?error={quote('Database connection not found')}",
            status_code=303,
        )
    try:
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, sync_external_logs, raw),
            timeout=120,
        )
        msg = (f"Synced {result['query_count']} query, "
               f"{result['llm_count']} LLM, "
               f"{result.get('egress_count', 0)} egress rows to {result['schema']}")
        return RedirectResponse(f"/admin/databases?saved={quote(msg)}", status_code=303)
    except asyncio.TimeoutError:
        return RedirectResponse(
            f"/admin/databases?error={quote('Log sync timed out')}",
            status_code=303,
        )
    except Exception as e:
        log.warning("External log sync failed for db_config_id=%s: %s", db_id, e)
        return RedirectResponse(
            f"/admin/databases?error={quote('Log sync failed: ' + str(e)[:160])}",
            status_code=303,
        )


@router.post("/databases/{db_id}/logs/reset-all")
async def database_logs_reset_all(request: Request, db_id: int):
    """Truncate ALL three external log tables and re-export everything from local SQLite."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    raw = store.get_db_config(db_id)
    if not raw:
        return RedirectResponse(
            f"/admin/databases?error={quote('Database config not found')}",
            status_code=303,
        )
    try:
        loop   = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, reset_all_and_sync, raw),
            timeout=180,
        )
        msg = (
            f"Full reset complete — re-exported "
            f"{result['query_count']} query, "
            f"{result['llm_count']} LLM, "
            f"{result['egress_count']} egress rows to {result['schema']}"
        )
        return RedirectResponse(f"/admin/databases?saved={quote(msg)}", status_code=303)
    except asyncio.TimeoutError:
        return RedirectResponse(
            f"/admin/databases?error={quote('Reset timed out (>3 min)')}",
            status_code=303,
        )
    except Exception as e:
        log.warning("Full log reset failed for db_config_id=%s: %s", db_id, e)
        return RedirectResponse(
            f"/admin/databases?error={quote('Full reset failed: ' + str(e)[:160])}",
            status_code=303,
        )


@router.post("/databases/{db_id}/logs/reset-egress")
async def database_logs_reset_egress(request: Request, db_id: int):
    """Truncate external KB_DATA_EGRESS_LOG and re-export all local rows from scratch."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    raw = store.get_db_config(db_id)
    if not raw:
        return RedirectResponse(
            f"/admin/databases?error={quote('Database config not found')}",
            status_code=303,
        )
    try:
        loop   = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, reset_egress_and_sync, raw),
            timeout=120,
        )
        msg = f"Egress log reset and re-exported {result['egress_count']} rows to {result['schema']}"
        return RedirectResponse(f"/admin/databases?saved={quote(msg)}", status_code=303)
    except asyncio.TimeoutError:
        return RedirectResponse(
            f"/admin/databases?error={quote('Reset timed out')}",
            status_code=303,
        )
    except Exception as e:
        log.warning("Egress reset failed for db_config_id=%s: %s", db_id, e)
        return RedirectResponse(
            f"/admin/databases?error={quote('Egress reset failed: ' + str(e)[:160])}",
            status_code=303,
        )


@router.get("/databases/{db_id}/logs/diagnose")
async def database_logs_diagnose(request: Request, db_id: int):
    """
    Return a JSON diagnostic showing local vs external egress row counts,
    whether the external table exists, and any missing columns.
    Useful for debugging why sync is not exporting egress rows.
    """
    if not _is_auth(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    raw = store.get_db_config(db_id)
    if not raw:
        return JSONResponse({"error": "Database config not found"}, status_code=404)
    try:
        loop   = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, diagnose_external_log_store, raw),
            timeout=30,
        )
    except asyncio.TimeoutError:
        result = {"error": "Diagnostic timed out (>30s)"}
    except Exception as exc:
        result = {"error": str(exc)}
    return JSONResponse(result)


@router.post("/databases/delete")
async def database_delete(request: Request, db_id: int = Form(...)):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    store.delete_db_config(db_id)
    return RedirectResponse("/admin/databases", status_code=303)


# ── Clients ───────────────────────────────────────────────────────────────────

@router.get("/clients", response_class=HTMLResponse)
async def clients_page(request: Request):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    search  = request.query_params.get("q", "").strip()
    clients = store.list_clients(search or None)
    dbs     = store.list_db_configs()
    db_map  = {d["id"]: f"{d['name']} ({d['label']})" for d in dbs}
    for c in clients:
        c["db_name"]     = db_map.get(c["db_config_id"], "Not assigned")
        c["health_score"] = _client_health_score(c["account_id"], c)
    return _resp(request, "clients.html", {
        "clients": clients, "search": search,
    })


@router.get("/clients/new", response_class=HTMLResponse)
async def client_new_page(request: Request):
    """Form to register a new client manually from the admin panel."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    dbs = store.list_db_configs()
    return _resp(request, "client_new.html", {
        "dbs":   dbs,
        "error": request.query_params.get("error"),
    })


@router.post("/clients/create")
async def client_create(
    request: Request,
    account_id:    str = Form(...),
    client_name:   str = Form(...),
    platform_type: str = Form(""),
    db_config_id:  str = Form(""),
    business_desc: str = Form(""),
    portal_only:   str = Form(""),
):
    """Create a client row (state = NEW) and redirect to the setup page."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)

    from urllib.parse import quote
    account_id    = account_id.strip()
    client_name   = client_name.strip()
    is_portal_only = bool(portal_only)

    if not account_id or not client_name:
        return RedirectResponse(
            f"/admin/clients/new?error={quote('Account ID and client name are required')}",
            status_code=303)

    if not is_portal_only and platform_type not in ("zoom", "teams", "slack"):
        return RedirectResponse(
            f"/admin/clients/new?error={quote('Select a platform or enable web-portal-only mode')}",
            status_code=303)

    if store.get_client(account_id):
        return RedirectResponse(
            f"/admin/clients/new?error={quote('A client with this account ID already exists')}",
            status_code=303)

    # portal-only clients have no chat platform. We store 'web' in the code
    # but must satisfy the existing DB CHECK constraint (zoom|teams|slack) on
    # older databases. Use 'zoom' as a neutral placeholder — platform_config_id
    # is NULL so platform_type is never used for portal-only clients anyway.
    effective_platform = "zoom" if is_portal_only else platform_type
    store.upsert_client(account_id, effective_platform)
    store.update_client_meta(
        account_id,
        client_name   = client_name,
        db_config_id  = int(db_config_id) if db_config_id else None,
        portal_only   = 1 if is_portal_only else 0,
        chat_ui_enabled = 1 if is_portal_only else None,
    )

    # Seed the business description into state_data so the Setup page pre-fills it
    if business_desc.strip():
        store.update_client_state(
            account_id, "NEW",
            {"business_desc": business_desc.strip()},
            business_desc.strip(),
        )

    log.info("Admin registered new client %s (%s, portal_only=%s)",
             account_id, client_name, is_portal_only)
    return RedirectResponse(
        f"/admin/clients/{account_id}/setup?saved={quote('Client registered — run schema discovery to continue')}",
        status_code=303)


@router.get("/clients/{account_id}", response_class=HTMLResponse)
async def client_detail(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client  = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)

    queries       = store.get_recent_queries(account_id, limit=50)
    # Fix #6 — filters on the LLM audit tab
    audit_component = (request.query_params.get("audit_component") or "").strip()
    audit_status    = (request.query_params.get("audit_status") or "").strip()
    llm_calls     = store.get_recent_llm_calls(
        account_id,
        limit=100,
        component=audit_component,
        status=audit_status,
    )
    stats         = store.get_query_stats(account_id)
    all_dbs       = store.list_db_configs()
    monthly_count = store.get_monthly_query_count(account_id)
    limit         = client.get("query_limit_monthly") or 500
    limit_pct     = min(round(monthly_count / limit * 100), 100)
    token_status  = store.get_monthly_token_status(account_id)

    # Failed queries for error panel
    failed_queries = [q for q in queries if not q["success"]][:10]

    # Top questions frequency map
    from collections import Counter
    q_counter = Counter(
        q["question"] for q in queries if q.get("question") and q["success"]
    )
    top_questions = q_counter.most_common(5)

    # KB / schema file counts
    state_data       = __import__("json").loads(client.get("state_data") or "{}")
    kb_dir           = state_data.get("kb_dir", "")
    schema_dir       = state_data.get("schema_dir", "")
    kb_file_count    = len(list(Path(kb_dir).glob("*.md")))    if kb_dir    and Path(kb_dir).exists()    else 0
    schema_file_count= len(list(Path(schema_dir).glob("*.md")))if schema_dir and Path(schema_dir).exists() else 0
    semantic_pending_count = store.count_semantic_field_feedback(account_id)

    # DB name for display
    db_map   = {d["id"]: f"{d['name']} ({d['label']})" for d in all_dbs}
    db_name  = db_map.get(client.get("db_config_id"), "Not assigned")

    # System model for display
    system_model = store.get_system("default_llm_model", "claude-sonnet-4-6")

    health_score = _client_health_score(account_id, client)

    return _resp(request, "client_detail.html", {
        "client":          client,
        "queries":         queries,
        "llm_calls":       llm_calls,
        "audit_component": audit_component,
        "audit_status":    audit_status,
        "stats":           stats,
        "all_dbs":         all_dbs,
        "query_models":    QUERY_MODELS,
        "saved":           request.query_params.get("saved"),
        "cost_rates":      store.LLM_COST_RATES,
        "monthly_count":   monthly_count,
        "limit_pct":       limit_pct,
        "token_status":    token_status,
        "failed_queries":  failed_queries,
        "top_questions":   top_questions,
        "kb_file_count":   kb_file_count,
        "schema_file_count": schema_file_count,
        "semantic_pending_count": semantic_pending_count,
        "db_name":         db_name,
        "system_model":    system_model,
        "health_score":    health_score,
    })


@router.get("/clients/{account_id}/health-score")
async def client_health_score_api(request: Request, account_id: str):
    """JSON health-score payload — used by the live widget on the detail page."""
    if not _is_auth(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    client = store.get_client(account_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    return JSONResponse(_client_health_score(account_id, client))


@router.get("/clients/{account_id}/traces", response_class=HTMLResponse)
async def client_traces(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    traces = store.list_answer_traces(account_id, limit=100)
    selected = None
    trace_id = request.query_params.get("trace_id")
    if trace_id:
        try:
            selected = store.get_answer_trace(int(trace_id))
        except Exception:
            selected = None
    return _resp(request, "client_traces.html", {
        "client": client,
        "traces": traces,
        "selected": selected,
    })


@router.get("/clients/{account_id}/evals", response_class=HTMLResponse)
async def client_evals(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    runs = store.list_eval_runs(account_id, limit=50)
    selected = None
    run_id = request.query_params.get("run_id")
    if run_id:
        try:
            selected = store.get_eval_run(int(run_id))
        except Exception:
            selected = None
    case_files = _eval_case_files(account_id)
    return _resp(request, "client_evals.html", {
        "client": client,
        "runs": runs,
        "selected": selected,
        "case_files": [str(p) for p in case_files],
        "latest": runs[0] if runs else None,
    })


@router.post("/clients/{account_id}/evals/run")
async def client_evals_run(
    request: Request,
    account_id: str,
    cases_path: str = Form(""),
    schema_name: str = Form(""),
    generate: str = Form(""),
    execute: str = Form(""),
):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    case_files = _eval_case_files(account_id)
    allowed = {str(p) for p in case_files}
    if not cases_path and case_files:
        cases_path = str(case_files[0])
    if not cases_path or cases_path not in allowed:
        return RedirectResponse(
            f"/admin/clients/{account_id}/evals?error={quote('No valid eval case file found')}",
            status_code=303,
        )
    from evals.run import run_eval_suite
    case_file = Path(cases_path)
    schema = schema_name.strip() or case_file.parent.name or "default"
    out_dir = Path("evals") / "reports" / account_id / schema / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    _, run_id = await run_eval_suite(
        account_id=account_id,
        schema=schema,
        cases_path=case_file,
        generate=bool(generate),
        execute=bool(execute),
        out_dir=out_dir,
    )
    return RedirectResponse(
        f"/admin/clients/{account_id}/evals?run_id={run_id}",
        status_code=303,
    )


@router.post("/clients/{account_id}/update")
async def client_update(
    request: Request,
    account_id: str,
    client_name:         str = Form(""),
    db_config_id:        str = Form(""),
    llm_provider:        str = Form(""),
    llm_model:           str = Form(""),
    query_limit_monthly: str = Form(""),
    token_limit_monthly: str = Form(""),
    enable_llm_audit:    str = Form(""),
    portal_only:         str = Form(""),
):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    is_portal_only = 1 if portal_only else 0
    store.update_client_meta(
        account_id,
        client_name         = client_name.strip() or None,
        db_config_id        = int(db_config_id) if db_config_id else None,
        llm_provider        = llm_provider or None,
        llm_model           = llm_model or None,
        query_limit_monthly = int(query_limit_monthly) if query_limit_monthly else None,
        token_limit_monthly = int(token_limit_monthly) if token_limit_monthly else 0,
        enable_llm_audit    = 1 if enable_llm_audit else 0,
        portal_only         = is_portal_only,
        # Portal-only clients always have the internal chat UI enabled
        chat_ui_enabled     = 1 if is_portal_only else None,
    )
    return RedirectResponse(f"/admin/clients/{account_id}?saved=1", status_code=303)


@router.post("/clients/{account_id}/reset")
async def client_reset(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    import shutil
    # Delete local schema/KB files
    shutil.rmtree(Path("clients") / account_id, ignore_errors=True)
    # Delete Qdrant vector index for this client
    try:
        from core.vector_store import delete_kb_for_client
        delete_kb_for_client(account_id)
    except Exception as _e:
        log.warning("client_reset: could not clear Qdrant vectors for %s: %s", account_id, _e)
    # Delete the DB row (cascades to query_log, portal_user, etc.)
    store.delete_client(account_id)
    log.info("Admin reset client %s", account_id)
    return RedirectResponse("/admin/clients", status_code=303)


# ── KB editor ─────────────────────────────────────────────────────────────────

@router.get("/clients/{account_id}/kb", response_class=HTMLResponse)
async def kb_list(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    state_data = json.loads(client.get("state_data") or "{}")
    kb_dir     = state_data.get("kb_dir", "")
    schema_dir = state_data.get("schema_dir", "")

    kb_files     = sorted(Path(kb_dir).glob("*.md"))     if kb_dir     and Path(kb_dir).exists()     else []
    schema_files = sorted(Path(schema_dir).glob("*.md")) if schema_dir and Path(schema_dir).exists() else []
    feedback_status = (request.query_params.get("feedback_status") or "pending").strip().lower()
    if feedback_status not in {"pending", "approved", "rejected", "all"}:
        feedback_status = "pending"
    semantic_feedback = store.list_semantic_field_feedback(
        account_id,
        status=None if feedback_status == "all" else feedback_status,
        limit=250,
    )
    if kb_dir:
        try:
            from core.semantic_kb_patch import locate_kb_file_for_feedback
            for item in semantic_feedback:
                item["patch_file"] = locate_kb_file_for_feedback(
                    kb_dir=kb_dir,
                    table_fqn=item.get("table_fqn", ""),
                    table_name=item.get("table_name", ""),
                    schema_name=item.get("schema_name", ""),
                )
        except Exception as exc:
            log.debug("Could not resolve Semantic Layer patch filenames: %s", exc)

    view = (request.query_params.get("view") or "fields").strip().lower()
    if view not in {"fields", "reviews", "files"}:
        view = "fields"

    semantic_tables: list[dict] = []
    field_overrides: dict = {"version": 1, "tables": {}}
    if kb_dir and Path(kb_dir).exists():
        try:
            from core.field_overrides import load_field_overrides
            from core.semantic_layer import build_semantic_layer_tables
            approved_feedback, pending_feedback = store.semantic_feedback_maps(account_id)
            field_overrides = load_field_overrides(account_id)
            semantic_tables = build_semantic_layer_tables(
                kb_dir=kb_dir,
                schema_dir=schema_dir,
                approved_feedback=approved_feedback,
                pending_feedback=pending_feedback,
                field_overrides=field_overrides,
            )
        except Exception as exc:
            log.warning("Could not build admin field editor for %s: %s", account_id, exc)

    needs_context = [
        {
            "table": table["table"],
            "table_fqn": table["fqn"],
            "schema": table["schema"],
            "column": field["column"],
        }
        for table in semantic_tables
        for field in table["fields"]
        if field.get("needs_context")
    ]
    field_count = sum(len(table.get("fields") or []) for table in semantic_tables)
    approved_field_count = sum(
        1
        for table in semantic_tables
        for field in table.get("fields") or []
        if field.get("approved")
    )
    pending_review_count = store.count_semantic_field_feedback(account_id, "pending")

    return _resp(request, "client_kb.html", {
        "client":       client,
        "kb_files":     [f.name for f in kb_files],
        "schema_files": [f.name for f in schema_files],
        "kb_dir":       kb_dir,
        "schema_dir":   schema_dir,
        "saved":        request.query_params.get("saved"),
        "file_view":    request.query_params.get("file"),
        "file_content": _read_kb_file(
            kb_dir,
            schema_dir,
            request.query_params.get("file", ""),
            request.query_params.get("type", "kb"),
        ),
        "semantic_feedback": semantic_feedback,
        "semantic_feedback_status": feedback_status,
        "feedback_saved": request.query_params.get("feedback"),
        "feedback_msg":   request.query_params.get("feedback_msg", ""),
        "field_error": request.query_params.get("field_error", ""),
        "needs_context":  needs_context,
        "semantic_tables": semantic_tables,
        "field_count": field_count,
        "approved_field_count": approved_field_count,
        "pending_review_count": pending_review_count,
        "view": view,
        "selected_table": request.query_params.get("table", ""),
    })


@router.post("/clients/{account_id}/semantic-feedback/{feedback_id}/review")
async def semantic_feedback_review(
    request: Request,
    account_id: str,
    feedback_id: int,
    status: str = Form(...),
    admin_note: str = Form(""),
):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    if status not in {"approved", "rejected"}:
        raise HTTPException(status_code=400, detail="Invalid review status")

    # Fetch the feedback item before updating so we have all field values
    all_items = store.list_semantic_field_feedback(account_id, limit=500)
    item = next((i for i in all_items if i["id"] == feedback_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Feedback item not found")

    patched_file = ""
    msg = ""
    if status == "approved":
        # ── Step 1: patch the KB file and re-embed ────────────────────────────
        client     = store.get_client(account_id)
        state_data = json.loads(client.get("state_data") or "{}")
        kb_dir     = state_data.get("kb_dir", "")

        if not kb_dir:
            return RedirectResponse(
                f"/admin/clients/{account_id}/kb"
                f"?view=reviews&feedback=error&feedback_msg=KB+directory+not+configured"
                f"&feedback_status=pending#semantic-feedback",
                status_code=303,
            )

        from core.semantic_kb_patch import apply_approved_feedback, locate_kb_file_for_feedback
        patched_file = locate_kb_file_for_feedback(
            kb_dir=kb_dir,
            table_fqn=item["table_fqn"],
            table_name=item["table_name"],
            schema_name=item.get("schema_name", ""),
        )
        success, msg = apply_approved_feedback(
            account_id=account_id,
            kb_dir=kb_dir,
            table_fqn=item["table_fqn"],
            table_name=item["table_name"],
            schema_name=item.get("schema_name", ""),
            column_name=item["column_name"],
            approved_meaning=item["suggested_meaning"],
            approved_use_case=item.get("suggested_use_case", ""),
            user_comment=item.get("user_comment", ""),
            admin_note=admin_note,
            persist_override=True,
        )

        if not success:
            # KB patch failed — do NOT mark as approved, keep pending
            return RedirectResponse(
                f"/admin/clients/{account_id}/kb"
                f"?view=reviews&feedback=error&feedback_msg={quote(msg)}"
                f"&feedback_status=pending#semantic-feedback",
                status_code=303,
            )

        log.info("Semantic KB patch applied for %s / %s: %s",
                 account_id, item["column_name"], msg)

    # ── Step 2: update SQLite status ─────────────────────────────────────────
    ok = store.review_semantic_field_feedback(
        feedback_id,
        account_id,
        status=status,
        admin_note=admin_note,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Feedback item not found")

    try:
        await notify_semantic_feedback_changed(
            account_id=account_id,
            action=status,
            feedback_id=feedback_id,
        )
    except Exception as exc:
        log.warning("Admin semantic feedback notification failed: %s", exc)

    try:
        await notify_portal_semantic_feedback_changed(
            account_id=account_id,
            portal_user_id=item.get("portal_user_id"),
            feedback_id=feedback_id,
            status=status,
            table_fqn=item.get("table_fqn", ""),
            column_name=item.get("column_name", ""),
            suggested_meaning=item.get("suggested_meaning", ""),
            suggested_use_case=item.get("suggested_use_case", ""),
            admin_note=admin_note,
        )
    except Exception as exc:
        log.warning("Portal semantic feedback notification failed: %s", exc)

    if status == "approved":
        asyncio.create_task(_run_default_evals_async(account_id, generate=True, execute=False))

    target_view = "files" if status == "approved" and patched_file else "reviews"
    redirect = (
        f"/admin/clients/{account_id}/kb"
        f"?view={target_view}&feedback={status}&feedback_status={status}"
    )
    if msg:
        redirect += f"&feedback_msg={quote(msg)}"
    if patched_file:
        redirect += f"&file={quote(patched_file)}#kb-editor"
    else:
        redirect += "#semantic-feedback"
    return RedirectResponse(
        redirect,
        status_code=303,
    )


@router.post("/clients/{account_id}/kb/save")
async def kb_save(
    request: Request,
    account_id: str,
    background_tasks: BackgroundTasks,
    filename:    str = Form(...),
    content:     str = Form(...),
    file_type:   str = Form("kb"),
):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client     = store.get_client(account_id)
    state_data = json.loads(client.get("state_data") or "{}")

    if file_type == "schema":
        target_dir = state_data.get("schema_dir", "")
    else:
        target_dir = state_data.get("kb_dir", "")

    if not target_dir:
        return RedirectResponse(f"/admin/clients/{account_id}/kb?view=files&saved=error",
                                status_code=303)

    filepath = _safe_child_path(target_dir, filename)
    if filepath is None:
        raise HTTPException(status_code=400, detail="Invalid filename")
    filepath.write_text(content, encoding="utf-8")

    # Re-embed only for KB files (schema MDs don't go into Qdrant)
    if file_type == "kb":
        try:
            from core.knowledge import re_embed_file
            re_embed_file(target_dir, account_id, filename)
            background_tasks.add_task(_run_default_evals_background, account_id)
        except Exception as e:
            log.warning("Re-embed failed for %s: %s", filename, e)

    log.info("Admin edited %s for client %s", filename, account_id)
    return RedirectResponse(
        f"/admin/clients/{account_id}/kb?view=files&saved=1"
        f"&file={quote(filename)}&type={quote(file_type)}",
        status_code=303
    )


@router.post("/clients/{account_id}/kb/fields/save")
async def kb_field_save(
    request: Request,
    account_id: str,
    background_tasks: BackgroundTasks,
    table_fqn: str = Form(...),
    column_name: str = Form(...),
    meaning: str = Form(...),
    use_case: str = Form(""),
    synonyms: str = Form(""),
    admin_note: str = Form(""),
):
    """Persist and immediately apply an administrator-owned field override."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    if not meaning.strip():
        return RedirectResponse(
            f"/admin/clients/{account_id}/kb?view=fields&field_error="
            f"{quote('Business meaning is required')}",
            status_code=303,
        )

    state_data = json.loads(client.get("state_data") or "{}")
    kb_dir = state_data.get("kb_dir", "")
    schema_dir = state_data.get("schema_dir", "")
    if not kb_dir or not Path(kb_dir).exists():
        return RedirectResponse(
            f"/admin/clients/{account_id}/kb?view=fields&field_error="
            f"{quote('Build the Knowledge Base before editing fields')}",
            status_code=303,
        )

    from core.field_overrides import load_field_overrides, parse_synonyms
    from core.semantic_kb_patch import apply_approved_feedback
    from core.semantic_layer import build_semantic_layer_tables, find_semantic_field

    approved_feedback, pending_feedback = store.semantic_feedback_maps(account_id)
    semantic_tables = build_semantic_layer_tables(
        kb_dir=kb_dir,
        schema_dir=schema_dir,
        approved_feedback=approved_feedback,
        pending_feedback=pending_feedback,
        field_overrides=load_field_overrides(account_id),
    )
    found = find_semantic_field(semantic_tables, table_fqn, column_name)
    if not found:
        raise HTTPException(status_code=404, detail="Field not found in this Knowledge Base")
    table, field = found

    success, msg = apply_approved_feedback(
        account_id=account_id,
        kb_dir=kb_dir,
        table_fqn=table["fqn"],
        table_name=table["table"],
        schema_name=table["schema"],
        column_name=field["column"],
        approved_meaning=meaning.strip(),
        approved_use_case=use_case.strip(),
        approved_synonyms=parse_synonyms(synonyms),
        admin_note=admin_note.strip(),
        persist_override=True,
        infer_synonyms=False,
    )
    if not success:
        return RedirectResponse(
            f"/admin/clients/{account_id}/kb?view=fields&field_error={quote(msg)}"
            f"&table={quote(table['fqn'])}",
            status_code=303,
        )

    background_tasks.add_task(_run_default_evals_background, account_id)
    return RedirectResponse(
        f"/admin/clients/{account_id}/kb?view=fields&saved=field"
        f"&table={quote(table['fqn'])}"
        f"#field-{quote(table['table'])}-{quote(field['column'])}",
        status_code=303,
    )


@router.post("/clients/{account_id}/kb/context-hints")
async def kb_save_context_hints(request: Request, account_id: str):
    """Save admin-supplied column context hints (resolves [NEEDS CONTEXT] flags)."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    form = await request.form()
    # Form fields are named  hint__{table}__{column}
    merged: dict[str, dict[str, str]] = {}
    for key, value in form.items():
        if key.startswith("hint__") and value.strip():
            parts = key.split("__", 2)
            if len(parts) == 3:
                _, tbl, col = parts
                merged.setdefault(tbl, {})[col] = value.strip()

    ctx_dir = Path("clients") / account_id
    ctx_dir.mkdir(parents=True, exist_ok=True)
    ctx_file = ctx_dir / "column_context.json"
    # Merge with any existing hints so saving one table doesn't wipe others
    existing: dict = {}
    if ctx_file.exists():
        try:
            existing = json.loads(ctx_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    for tbl, hints in merged.items():
        existing.setdefault(tbl, {}).update(hints)
    ctx_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    log.info("Column context hints saved for %s (%d tables)", account_id, len(merged))
    return RedirectResponse(
        f"/admin/clients/{account_id}/kb?saved=context",
        status_code=303,
    )


def _read_kb_file(
    kb_dir: str,
    schema_dir: str,
    filename: str,
    file_type: str = "kb",
) -> str:
    if not filename:
        return ""
    directories = [schema_dir] if file_type == "schema" else [kb_dir]
    for d in directories:
        if d:
            p = _safe_child_path(d, filename)
            if p and p.exists():
                return p.read_text(encoding="utf-8")
    return ""


# ── Billing export ────────────────────────────────────────────────────────────

@router.get("/clients/{account_id}/billing", response_class=HTMLResponse)
async def billing_page(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client    = store.get_client(account_id)
    month     = request.query_params.get("month", "")
    stats     = store.get_query_stats(account_id, month or None)
    breakdown = store.get_monthly_breakdown(account_id)
    pricing_rows = store.get_all_pricing()
    return _resp(request, "billing.html", {
        "client": client, "stats": stats,
        "breakdown": breakdown, "month": month,
        "pricing_rows": pricing_rows,
        "saved_pricing": request.query_params.get("saved_pricing"),
        "error_pricing": request.query_params.get("error_pricing"),
    })


@router.post("/clients/{account_id}/billing/pricing/save")
async def billing_pricing_save(
    request: Request,
    account_id: str,
    model:      str   = Form(""),
    tokens_in:  str   = Form(""),
    tokens_out: str   = Form(""),
):
    """Save or update a single model pricing rate (USD per 1M tokens)."""
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    model = model.strip()
    if not model:
        return RedirectResponse(
            f"/admin/clients/{account_id}/billing?error_pricing=Model+name+required",
            status_code=303,
        )
    try:
        t_in  = float(tokens_in)
        t_out = float(tokens_out)
        if t_in < 0 or t_out < 0:
            raise ValueError("Rates must be non-negative")
    except (ValueError, TypeError):
        return RedirectResponse(
            f"/admin/clients/{account_id}/billing?error_pricing=Invalid+rate+values",
            status_code=303,
        )
    store.save_pricing(model, t_in, t_out)
    return RedirectResponse(
        f"/admin/clients/{account_id}/billing?saved_pricing=1",
        status_code=303,
    )


@router.get("/clients/{account_id}/billing/known-prices")
async def billing_known_prices(request: Request, account_id: str):
    """Return a static lookup of current model pricing (USD per 1M tokens).

    Sources: OpenAI pricing page, Anthropic pricing page, Google pricing page.
    Updated 2026-06. All prices are public list prices — no discount/commitment tiers.
    """
    if not _is_auth(request):
        raise HTTPException(status_code=401)

    PRICES = [
        # ── OpenAI ──────────────────────────────────────────────────────────
        {"model": "gpt-4o",                  "provider": "OpenAI",    "tokens_in": 2.50,  "tokens_out": 10.00},
        {"model": "gpt-4o-2024-11-20",       "provider": "OpenAI",    "tokens_in": 2.50,  "tokens_out": 10.00},
        {"model": "gpt-4o-mini",             "provider": "OpenAI",    "tokens_in": 0.15,  "tokens_out": 0.60},
        {"model": "gpt-4o-mini-2024-07-18",  "provider": "OpenAI",    "tokens_in": 0.15,  "tokens_out": 0.60},
        {"model": "gpt-4-turbo",             "provider": "OpenAI",    "tokens_in": 10.00, "tokens_out": 30.00},
        {"model": "gpt-4",                   "provider": "OpenAI",    "tokens_in": 30.00, "tokens_out": 60.00},
        {"model": "gpt-3.5-turbo",           "provider": "OpenAI",    "tokens_in": 0.50,  "tokens_out": 1.50},
        {"model": "o1",                      "provider": "OpenAI",    "tokens_in": 15.00, "tokens_out": 60.00},
        {"model": "o1-mini",                 "provider": "OpenAI",    "tokens_in": 1.10,  "tokens_out": 4.40},
        {"model": "o3",                      "provider": "OpenAI",    "tokens_in": 10.00, "tokens_out": 40.00},
        {"model": "o3-mini",                 "provider": "OpenAI",    "tokens_in": 1.10,  "tokens_out": 4.40},
        {"model": "o4-mini",                 "provider": "OpenAI",    "tokens_in": 1.10,  "tokens_out": 4.40},
        # ── Anthropic ───────────────────────────────────────────────────────
        {"model": "claude-opus-4-8",         "provider": "Anthropic", "tokens_in": 15.00, "tokens_out": 75.00},
        {"model": "claude-sonnet-4-6",       "provider": "Anthropic", "tokens_in": 3.00,  "tokens_out": 15.00},
        {"model": "claude-haiku-4-5",        "provider": "Anthropic", "tokens_in": 0.80,  "tokens_out": 4.00},
        {"model": "claude-3-5-sonnet-20241022","provider":"Anthropic", "tokens_in": 3.00,  "tokens_out": 15.00},
        {"model": "claude-3-5-haiku-20241022","provider": "Anthropic", "tokens_in": 0.80,  "tokens_out": 4.00},
        {"model": "claude-3-opus-20240229",  "provider": "Anthropic", "tokens_in": 15.00, "tokens_out": 75.00},
        {"model": "claude-3-sonnet-20240229","provider": "Anthropic", "tokens_in": 3.00,  "tokens_out": 15.00},
        {"model": "claude-3-haiku-20240307", "provider": "Anthropic", "tokens_in": 0.25,  "tokens_out": 1.25},
        # ── Google ──────────────────────────────────────────────────────────
        {"model": "gemini-2.5-pro",          "provider": "Google",    "tokens_in": 1.25,  "tokens_out": 10.00},
        {"model": "gemini-2.5-flash",        "provider": "Google",    "tokens_in": 0.15,  "tokens_out": 0.60},
        {"model": "gemini-1.5-pro",          "provider": "Google",    "tokens_in": 1.25,  "tokens_out": 5.00},
        {"model": "gemini-1.5-flash",        "provider": "Google",    "tokens_in": 0.075, "tokens_out": 0.30},
    ]

    # Mark which models are already in the DB
    existing = {r["model"] for r in store.get_all_pricing()}
    for p in PRICES:
        p["exists"] = p["model"] in existing

    return JSONResponse({"ok": True, "prices": PRICES})


@router.get("/clients/{account_id}/billing/export.csv")
async def billing_export(request: Request, account_id: str):
    if not _is_auth(request):
        raise HTTPException(status_code=401)

    client    = store.get_client(account_id) or {}
    breakdown = store.get_monthly_breakdown(account_id)
    stats     = store.get_query_stats(account_id)

    buf = io.StringIO()
    writer = csv.writer(buf)

    # Header block
    writer.writerow(["QueryBot — Usage Report"])
    writer.writerow(["Client", client.get("client_name", account_id)])
    writer.writerow(["Account ID", account_id])
    writer.writerow(["Report generated", __import__("datetime").datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")])
    writer.writerow([])

    # Summary
    writer.writerow(["Summary"])
    writer.writerow(["Total queries", stats.get("total", 0)])
    writer.writerow(["Successful", stats.get("succeeded", 0)])
    writer.writerow(["Total tokens in", stats.get("total_tokens_in", 0)])
    writer.writerow(["Total tokens out", stats.get("total_tokens_out", 0)])
    writer.writerow(["Total cost (USD)", f"${(stats.get('total_cost_usd') or 0):.4f}"])
    writer.writerow([])

    # Daily breakdown
    writer.writerow(["Date", "Queries", "Successful", "Tokens In", "Tokens Out", "Cost (USD)"])
    for row in breakdown:
        writer.writerow([
            row["date"], row["total_queries"], row["successful"],
            row["tokens_in"], row["tokens_out"],
            f"${(row['cost_usd'] or 0):.4f}",
        ])

    buf.seek(0)
    filename = f"querybot_usage_{account_id[:12]}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── Status API ────────────────────────────────────────────────────────────────

@router.get("/api/status")
async def api_status(request: Request):
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    clients = store.list_clients()
    stats   = store.get_query_stats()
    return {
        "clients_total": len(clients),
        "clients_ready": sum(1 for c in clients if c["state"] == "READY"),
        "total_queries": stats.get("total", 0),
        "total_cost_usd": round(stats.get("total_cost_usd") or 0, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Groups management
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/clients/{account_id}/compliance", response_class=HTMLResponse)
async def compliance_page(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    from core.compliance.packs import list_packs

    profile = store.get_compliance_profile(account_id)
    return _resp(request, "client_compliance.html", {
        "client": client,
        "profile": profile,
        "packs": list_packs(),
        "classifications": store.list_classifications(account_id),
        "rules": store.list_policy_rules(
            account_id, int(profile.get("active_policy_version") or 0) or None
        ),
        "row_policies": store.list_row_policies(
            account_id, int(profile.get("active_policy_version") or 0) or None
        ),
        "purposes": store.list_purposes(account_id),
        "agreements": store.list_provider_agreements(account_id),
        "assessment": store.get_latest_assessment(account_id),
        "versions": store.list_policy_versions(account_id),
        "decisions": store.list_decisions(account_id, limit=30),
        "users": store.list_users(account_id),
        "saved": request.query_params.get("saved"),
        "error": request.query_params.get("error"),
    })


@router.post("/clients/{account_id}/compliance/profile")
async def compliance_save_profile(request: Request, account_id: str):
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    form = await request.form()
    industry = str(form.get("industry") or "standard")
    pack_key = {
        "banking": "banking_v1",
        "healthcare_pharmacy": "healthcare_pharmacy_v1",
    }.get(industry, "")
    if not pack_key:
        store.save_compliance_profile(
            account_id, mode="standard", industry="standard",
            jurisdictions=[], frameworks=[], policy_pack_key="",
            policy_pack_version="", lifecycle_state="DRAFT",
            enforcement_mode="shadow",
        )
        return RedirectResponse(
            f"/admin/clients/{account_id}/compliance?saved=standard", status_code=303
        )

    from core.compliance.classifier import (
        import_legacy_masking, import_schema_classifications,
    )
    from core.compliance.packs import get_pack

    pack = get_pack(pack_key)
    jurisdictions = [
        str(value) for key, value in form.multi_items() if key == "jurisdictions"
    ]
    frameworks = list((pack.get("framework_versions") or {}).keys())
    store.save_compliance_profile(
        account_id, mode="regulated", industry=industry,
        jurisdictions=jurisdictions, frameworks=frameworks,
        policy_pack_key=pack_key, policy_pack_version=pack["version"],
        lifecycle_state="CLASSIFICATION_PENDING", enforcement_mode="shadow",
        invalidated_reason="Regulated profile or jurisdiction changed.",
    )
    version = store.create_policy_version(
        account_id,
        {"pack": pack, "jurisdictions": jurisdictions, "frameworks": frameworks},
        created_by="admin", change_summary=f"Applied {pack_key}", status="active",
    )
    store.activate_policy_version(account_id, version, activated_by="admin")
    store.replace_purposes(account_id, pack.get("default_purposes", []))
    store.replace_policy_rules(account_id, version, _default_regulated_rules(pack))

    client = store.get_client(account_id) or {}
    state_data = json.loads(client.get("state_data") or "{}")
    import_schema_classifications(
        account_id, state_data.get("schema_dir", ""), industry
    )
    import_legacy_masking(account_id, state_data.get("masking_config") or {})
    return RedirectResponse(
        f"/admin/clients/{account_id}/compliance?saved=profile", status_code=303
    )


@router.post("/api/clients/{account_id}/compliance/classifications")
async def compliance_save_classification(request: Request, account_id: str):
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    body = await request.json()
    store.save_classification(
        account_id, str(body.get("table_fqn") or ""),
        str(body.get("column_name") or ""),
        sensitivity=str(body.get("sensitivity") or "INTERNAL"),
        identifiability=str(body.get("identifiability") or "NONE"),
        tags=list(body.get("tags") or []),
        confidence=float(body.get("confidence") or 1.0),
        reviewed=bool(body.get("reviewed", True)), reviewed_by="admin",
        mask_strategy=str(body.get("mask_strategy") or "redact"),
        source="admin",
    )
    store.save_compliance_profile(
        account_id, lifecycle_state="POLICY_PENDING", enforcement_mode="shadow",
        invalidated_reason="Data classification changed.",
    )
    return JSONResponse({"ok": True})


@router.post("/api/clients/{account_id}/compliance/policies")
async def compliance_save_policies(request: Request, account_id: str):
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    body = await request.json()
    profile = store.get_compliance_profile(account_id)
    rules = list(body.get("rules") or [])
    version = store.create_policy_version(
        account_id,
        {"rules": rules, "purposes": store.list_purposes(account_id), "profile": profile},
        created_by="admin",
        change_summary=str(body.get("change_summary") or "Policy rules updated"),
        status="active",
    )
    store.replace_policy_rules(account_id, version, rules)
    store.activate_policy_version(account_id, version, activated_by="admin")
    store.save_compliance_profile(
        account_id, lifecycle_state="POLICY_PENDING", enforcement_mode="shadow",
        invalidated_reason="Policy rules changed.",
    )
    return JSONResponse({"ok": True, "version": version})


@router.post("/clients/{account_id}/compliance/agreement")
async def compliance_save_agreement(request: Request, account_id: str):
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    form = await request.form()
    frameworks = [
        item.strip() for item in str(form.get("frameworks") or "").split(",")
        if item.strip()
    ]
    artifact_ref = str(form.get("artifact_ref") or "").strip()
    store.save_provider_agreement(account_id, {
        "provider": str(form.get("provider") or ""),
        "agreement_type": str(form.get("agreement_type") or ""),
        "frameworks": frameworks, "artifact_ref": artifact_ref,
        "artifact_hash": hashlib.sha256(artifact_ref.encode()).hexdigest() if artifact_ref else "",
        "signed_at": str(form.get("signed_at") or "") or None,
        "expires_at": str(form.get("expires_at") or "") or None,
    })
    return RedirectResponse(
        f"/admin/clients/{account_id}/compliance?saved=agreement", status_code=303
    )


@router.post("/clients/{account_id}/compliance/controls")
async def compliance_save_controls(request: Request, account_id: str):
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    form = await request.form()
    store.save_compliance_profile(
        account_id,
        identity_control=str(form.get("identity_control") or "password"),
        managed_secrets_enabled=bool(form.get("managed_secrets_enabled")),
        immutable_audit_enabled=bool(form.get("immutable_audit_enabled")),
        external_audit_destination=str(form.get("external_audit_destination") or "").strip(),
        enforcement_mode="shadow",
        invalidated_reason="Production security controls changed.",
    )
    return RedirectResponse(
        f"/admin/clients/{account_id}/compliance?saved=controls", status_code=303
    )


@router.post("/api/clients/{account_id}/compliance/row-policies")
async def compliance_save_row_policies(request: Request, account_id: str):
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    body = await request.json()
    profile = store.get_compliance_profile(account_id)
    version = int(profile.get("active_policy_version") or 0)
    if not version:
        raise HTTPException(status_code=409, detail="Create a policy profile first.")
    store.replace_row_policies(account_id, version, list(body.get("row_policies") or []))
    store.save_compliance_profile(
        account_id, lifecycle_state="POLICY_PENDING", enforcement_mode="shadow",
        invalidated_reason="Row policies changed.",
    )
    return JSONResponse({"ok": True, "version": version})


@router.post("/api/clients/{account_id}/compliance/simulate")
async def compliance_simulate(request: Request, account_id: str):
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    body = await request.json()
    user = store.get_user(int(body.get("user_id"))) if body.get("user_id") else {
        "id": "simulation", "role": str(body.get("role") or "analyst"),
    }
    from core.compliance.models import ResourceRef
    from core.compliance.policy_engine import evaluate, resolve_context

    resources = []
    for value in body.get("resources") or []:
        value = str(value).strip()
        table, _, column = value.rpartition(".")
        resources.append(ResourceRef(table=table or value, column=column if table else ""))
    context = resolve_context(
        account_id, user, action=str(body.get("action") or "query_execution"),
        channel=str(body.get("channel") or "portal"),
        purpose_id=str(body.get("purpose_id") or ""),
        provider=str(body.get("provider") or ""),
    )
    decision = evaluate(context, resources, record=False)
    return JSONResponse({
        "allowed": decision.allowed, "effective_allowed": decision.effective_allowed,
        "shadow": decision.shadow, "reason_code": decision.reason_code,
        "explanation": decision.explanation, "masking": decision.masking,
        "aggregate_only": [item.key for item in decision.aggregate_only],
        "cache_ttl_seconds": decision.cache_ttl_seconds,
        "export_allowed": decision.export_allowed,
        "policy_version": decision.policy_version,
    })


@router.post("/clients/{account_id}/compliance/assess")
async def compliance_assess(request: Request, account_id: str):
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    from core.compliance.readiness import assess

    result = assess(account_id)
    store.save_compliance_profile(account_id, lifecycle_state=result["state"])
    return RedirectResponse(
        f"/admin/clients/{account_id}/compliance?saved=assessed", status_code=303
    )


@router.post("/clients/{account_id}/compliance/activate")
async def compliance_activate(request: Request, account_id: str):
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    from core.compliance.readiness import activate_pilot

    result = activate_pilot(account_id, activated_by="admin")
    status = "activated" if not result["critical_failed"] else "blocked"
    return RedirectResponse(
        f"/admin/clients/{account_id}/compliance?saved={status}", status_code=303
    )


@router.post("/clients/{account_id}/compliance/break-glass")
async def compliance_break_glass(request: Request, account_id: str):
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    form = await request.form()
    if _hash(str(form.get("password") or "")) != store.get_system("admin_password_hash", ""):
        return RedirectResponse(
            f"/admin/clients/{account_id}/compliance?error=Re-authentication+failed",
            status_code=303,
        )
    duration = min(max(int(form.get("duration_minutes") or 15), 1), 60)
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=duration)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    actions = [
        item.strip()
        for item in str(form.get("actions") or "query_execution,result_release").split(",")
        if item.strip()
    ]
    resources = [
        item.strip() for item in str(form.get("resources") or "*").split(",")
        if item.strip()
    ]
    user_id = str(form.get("user_id") or "")
    grant_id = store.create_break_glass_grant(
        account_id, user_id, str(form.get("incident_ref") or ""),
        str(form.get("reason") or ""), resources, actions, expires_at,
        created_by="admin",
    )
    store.log_policy_decision(
        account_id=account_id, user_id=user_id, action="break_glass",
        purpose_id="incident_response", channel="admin", allowed=True,
        reason_code="break_glass_grant_created", resources=resources,
        obligations={"grant_id": grant_id, "expires_at": expires_at, "export_allowed": False},
        policy_version=int(store.get_compliance_profile(account_id).get("active_policy_version") or 0),
    )
    await admin_notification_hub.broadcast({
        "type": "break_glass_grant",
        "account_id": account_id,
        "user_id": user_id,
        "grant_id": grant_id,
        "expires_at": expires_at,
        "incident_ref": str(form.get("incident_ref") or ""),
    })
    return RedirectResponse(
        f"/admin/clients/{account_id}/compliance?saved=break-glass", status_code=303
    )


@router.get("/clients/{account_id}/groups", response_class=HTMLResponse)
async def groups_page(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client     = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    groups     = store.list_groups(account_id)
    state_data = json.loads(client.get("state_data") or "{}")
    schema_dir = state_data.get("schema_dir", "")
    # Load all discovered table names for the assignment checkboxes
    all_tables = sorted(store.load_schema_tables(schema_dir)) if schema_dir else []
    # Attach current table list to each group
    for g in groups:
        g["tables"] = store.get_group_tables(g["id"])
    return _resp(request, "client_groups.html", {
        "client":     client,
        "groups":     groups,
        "all_tables": all_tables,
        "saved":      request.query_params.get("saved"),
        "error":      request.query_params.get("error"),
    })


@router.post("/clients/{account_id}/groups/create")
async def group_create(
    request:     Request,
    account_id:  str,
    name:        str = Form(...),
    description: str = Form(""),
):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    if not name.strip():
        return RedirectResponse(f"/admin/clients/{account_id}/groups?error=Name+required", status_code=303)
    store.create_group(account_id, name.strip(), description.strip())
    return RedirectResponse(f"/admin/clients/{account_id}/groups?saved=1", status_code=303)


@router.post("/clients/{account_id}/groups/{group_id}/tables")
async def group_save_tables(
    request:    Request,
    account_id: str,
    group_id:   int,
):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    form   = await request.form()
    tables = [v for k, v in form.multi_items() if k == "tables"]
    store.set_group_tables(group_id, account_id, tables)
    return RedirectResponse(f"/admin/clients/{account_id}/groups?saved=1", status_code=303)


@router.post("/clients/{account_id}/groups/{group_id}/delete")
async def group_delete(request: Request, account_id: str, group_id: int):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    store.delete_group(group_id)
    return RedirectResponse(f"/admin/clients/{account_id}/groups", status_code=303)


# ══════════════════════════════════════════════════════════════════════════════
# Users management
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/clients/{account_id}/users", response_class=HTMLResponse)
async def users_page(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    users  = store.list_users(account_id)
    groups = store.list_groups(account_id)
    return _resp(request, "client_users.html", {
        "client":    client,
        "users":     users,
        "groups":    groups,
        "saved":     request.query_params.get("saved"),
        "new_user":  request.query_params.get("new_user"),
        "temp_pw":   request.query_params.get("temp_pw"),
        "error":     request.query_params.get("error"),
    })


@router.post("/clients/{account_id}/users/create")
async def user_create(
    request:    Request,
    account_id: str,
    name:       str = Form(...),
    email:      str = Form(...),
    group_id:   str = Form(""),
    role:       str = Form("analyst"),
    password:   str = Form(""),
    confirm_password: str = Form(""),
):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    if not name.strip() or not email.strip():
        return RedirectResponse(
            f"/admin/clients/{account_id}/users?error=Name+and+email+required",
            status_code=303)

    # If admin supplied an explicit password, validate it matches and is strong enough
    explicit_pw = password.strip()
    if explicit_pw:
        if explicit_pw != confirm_password.strip():
            return RedirectResponse(
                f"/admin/clients/{account_id}/users?error=Passwords+do+not+match",
                status_code=303)
        if len(explicit_pw) < 8:
            return RedirectResponse(
                f"/admin/clients/{account_id}/users?error=Password+must+be+at+least+8+characters",
                status_code=303)

    try:
        gid  = int(group_id) if group_id else None
        # Pass the explicit password if provided; None triggers a temp-pw flow
        uid, plain_pw = store.create_user(
            account_id, name.strip(), email.strip(), gid, role,
            password=explicit_pw or None,
        )
        from urllib.parse import quote
        # is_temp: if no explicit password was provided a temp one was generated
        is_temp = "0" if explicit_pw else "1"
        return RedirectResponse(
            f"/admin/clients/{account_id}/users?saved=1&new_user={quote(name)}"
            f"&temp_pw={quote(plain_pw)}&is_temp={is_temp}",
            status_code=303)
    except Exception as e:
        from urllib.parse import quote
        return RedirectResponse(
            f"/admin/clients/{account_id}/users?error={quote(str(e))}",
            status_code=303)


@router.post("/clients/{account_id}/users/{user_id}/update")
async def user_update(
    request:    Request,
    account_id: str,
    user_id:    int,
    name:       str = Form(""),
    group_id:   str = Form(""),
    role:       str = Form(""),
    is_active:  str = Form(""),
):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    store.update_user(
        user_id,
        name     = name.strip() or None,
        group_id = int(group_id) if group_id else None,
        role     = role or None,
        is_active= int(is_active) if is_active in ("0","1") else None,
    )
    return RedirectResponse(f"/admin/clients/{account_id}/users?saved=1", status_code=303)


@router.post("/clients/{account_id}/users/{user_id}/reset-password")
async def user_reset_password(request: Request, account_id: str, user_id: int):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    temp_pw = store.reset_user_password(user_id)
    user    = store.get_user(user_id)
    from urllib.parse import quote
    return RedirectResponse(
        f"/admin/clients/{account_id}/users?saved=1"
        f"&new_user={user['name'] if user else ''}&temp_pw={quote(temp_pw)}",
        status_code=303)


@router.post("/clients/{account_id}/users/{user_id}/delete")
async def user_delete(request: Request, account_id: str, user_id: int):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    store.delete_user(user_id)
    return RedirectResponse(f"/admin/clients/{account_id}/users", status_code=303)


# ══════════════════════════════════════════════════════════════════════════════
# Metric registry — Step 3
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
# Schema drift detection
# ══════════════════════════════════════════════════════════════════════════════

def _compute_schema_drift(old: dict, new: dict) -> dict:
    """
    Diff two _schema.json dicts and return a structured change report.

    Each key in the dicts is a fully-qualified table name (e.g. "DB.SCHEMA.TABLE").
    Each value has a "columns" list of {"name": str, "type": str, ...} dicts.

    Returns:
      {
        "has_changes":    bool,
        "added_tables":   [fqn, ...],
        "removed_tables": [fqn, ...],
        "column_changes": {
          fqn: {
            "added":        [col_name, ...],
            "removed":      [col_name, ...],
            "type_changes": [{"column": name, "old_type": t1, "new_type": t2}, ...]
          }
        },
        "detected_at": ISO-8601 string
      }
    """
    old_map = {k.upper(): k for k in old}
    new_map = {k.upper(): k for k in new}
    old_keys = set(old_map)
    new_keys = set(new_map)

    added_tables   = sorted(new_keys - old_keys)
    removed_tables = sorted(old_keys - new_keys)

    column_changes: dict[str, dict] = {}
    for uk in sorted(old_keys & new_keys):
        old_cols = {c["name"].upper(): c for c in (old[old_map[uk]].get("columns") or [])}
        new_cols = {c["name"].upper(): c for c in (new[new_map[uk]].get("columns") or [])}
        added_cols   = sorted(new_cols.keys() - old_cols.keys())
        removed_cols = sorted(old_cols.keys() - new_cols.keys())
        type_changes = [
            {"column": nc, "old_type": old_cols[nc].get("type",""), "new_type": new_cols[nc].get("type","")}
            for nc in sorted(old_cols.keys() & new_cols.keys())
            if old_cols[nc].get("type") != new_cols[nc].get("type")
        ]
        if added_cols or removed_cols or type_changes:
            column_changes[uk] = {
                "added":        added_cols,
                "removed":      removed_cols,
                "type_changes": type_changes,
            }

    return {
        "has_changes":    bool(added_tables or removed_tables or column_changes),
        "added_tables":   added_tables,
        "removed_tables": removed_tables,
        "column_changes": column_changes,
        "detected_at":    datetime.now(timezone.utc).isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Entity Graph — admin routes
# ══════════════════════════════════════════════════════════════════════════════

def _auto_populate_entity_graph(account_id: str, schema_dir: str) -> tuple[int, int]:
    """
    Insert auto-detected entities/relationships with status='suggested'.
    Implementation lives in core.graph_autopopulate (shared with the
    post-KB-build sync); this wrapper keeps the discovery call site stable.
    """
    from core.graph_autopopulate import auto_populate_from_schema
    return auto_populate_from_schema(account_id, schema_dir)


@router.get("/clients/{account_id}/graph", response_class=HTMLResponse)
async def graph_page(request: Request, account_id: str):
    """Interactive entity graph builder page."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    entities      = store.list_entities(account_id, active_only=False)
    relationships = store.list_relationships(account_id, active_only=False)

    # Run health check on page load so the badge is server-rendered
    from core.graph_health import check_graph_health
    try:
        health = check_graph_health(account_id).to_dict()
    except Exception:
        health = {"score": None, "error_count": 0, "warning_count": 0,
                  "issues": [], "unmapped_tables": [], "entity_severity": {},
                  "has_schema": False}

    pending_count = (
        sum(1 for e in entities if e.get("status") == "suggested" and e.get("is_active", 1))
        + sum(1 for r in relationships if r.get("status") == "suggested" and r.get("is_active", 1))
    )

    # Build entity→column list for the join editor dropdowns (entity name uppercase → [col, ...])
    entity_columns: dict = {}
    try:
        import json as _json
        from pathlib import Path as _Path
        from core.schema import _normalize_schema
        _state = store.get_client_state(account_id)
        _schema_dir = _state.get("schema_dir", "")
        if _schema_dir:
            _schema_json = _Path(_schema_dir) / "_schema.json"
            if _schema_json.exists():
                _master = _normalize_schema(_json.loads(_schema_json.read_text(encoding="utf-8")))
                for _fqn, _tbl in _master.items():
                    _bare = _fqn.split(".")[-1].upper()
                    _cols = [c.get("name","") for c in _tbl.get("columns",[]) if c.get("name","")]
                    if _cols:
                        entity_columns[_bare] = _cols
    except Exception:
        pass

    return _resp(request, "client_graph.html", {
        "client":         client,
        "entities":       entities,
        "relationships":  relationships,
        "health":         health,
        "pending_count":  pending_count,
        "entity_columns": entity_columns,
        "saved":          request.query_params.get("saved"),
        "error":          request.query_params.get("error"),
    })


@router.post("/clients/{account_id}/graph/toggle-suggested")
async def graph_toggle_suggested(request: Request, account_id: str):
    """Toggle whether unreviewed (suggested) graph rows may feed SQL generation."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id) or {}
    current = client.get("graph_use_suggested")
    current = 1 if current is None else int(current)
    store.update_client_meta(account_id, graph_use_suggested=0 if current else 1)
    log.info("Admin toggled graph_use_suggested → %d for %s", 0 if current else 1, account_id)
    return RedirectResponse(f"/admin/clients/{account_id}/graph", status_code=303)


@router.post("/clients/{account_id}/graph/entities/create")
async def graph_entity_create(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    form = await request.form()
    entity_name = (form.get("entity_name") or "").strip()
    if not entity_name:
        return RedirectResponse(f"/admin/clients/{account_id}/graph?error=name_required", status_code=303)
    store.save_entity(
        account_id   = account_id,
        entity_name  = entity_name,
        table_name   = (form.get("table_name") or "").strip(),
        schema_name  = (form.get("schema_name") or "").strip(),
        pk_column    = (form.get("pk_column") or "").strip(),
        display_name = (form.get("display_name") or "").strip(),
        description  = (form.get("description") or "").strip(),
        entity_type  = (form.get("entity_type") or "dimension").strip(),
        is_active    = 1 if form.get("is_active") else 0,
    )
    return RedirectResponse(f"/admin/clients/{account_id}/graph?saved=1", status_code=303)


@router.post("/clients/{account_id}/graph/entities/{entity_name}/delete")
async def graph_entity_delete(request: Request, account_id: str, entity_name: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    store.delete_entity(account_id, entity_name)
    return RedirectResponse(f"/admin/clients/{account_id}/graph?saved=1", status_code=303)


@router.post("/clients/{account_id}/graph/relationships/create")
async def graph_rel_create(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    form = await request.form()
    _from_entity = (form.get("from_entity") or "").strip()
    _to_entity   = (form.get("to_entity") or "").strip()
    _from_col    = (form.get("from_column") or "").strip()
    _to_col      = (form.get("to_column") or "").strip()
    _join_type   = (form.get("join_type") or "INNER").strip()
    _label       = (form.get("label") or "").strip()
    store.save_relationship(
        account_id        = account_id,
        from_entity       = _from_entity,
        to_entity         = _to_entity,
        from_column       = _from_col,
        to_column         = _to_col,
        relationship_type = (form.get("relationship_type") or "many_to_one").strip(),
        join_type         = _join_type,
        label             = _label,
    )
    # Sync confirmed relationship into the structured semantic model (S2-2).
    try:
        from core.semantic_model import patch_relationship
        _sm_state  = store.get_client_state(account_id)
        _sm_kb_dir = (_sm_state or {}).get("kb_dir") or ""
        if _sm_kb_dir and _from_entity and _to_entity:
            _fe = store.get_entity(account_id, _from_entity)
            _te = store.get_entity(account_id, _to_entity)
            if _fe and _te:
                def _qtable(e: dict) -> str:
                    sn = (e.get("schema_name") or "").strip()
                    tn = (e.get("table_name") or "").strip()
                    return f"{sn}.{tn}" if sn else tn
                patch_relationship(
                    kb_dir=_sm_kb_dir,
                    from_table=_qtable(_fe),
                    to_table=_qtable(_te),
                    from_column=_from_col,
                    to_column=_to_col,
                    join_type=_join_type,
                    display_column=_label,
                    status="approved",
                )
    except Exception as _sm_exc:
        log.warning("patch_relationship (rel_create) skipped: %s", _sm_exc)
    return RedirectResponse(f"/admin/clients/{account_id}/graph?saved=1", status_code=303)


@router.post("/clients/{account_id}/graph/relationships/{rel_id}/delete")
async def graph_rel_delete(request: Request, account_id: str, rel_id: int):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    store.delete_relationship(account_id, rel_id)
    return RedirectResponse(f"/admin/clients/{account_id}/graph?saved=1", status_code=303)


@router.post("/clients/{account_id}/graph/properties/save")
async def graph_property_save(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    form = await request.form()
    store.save_entity_property(
        account_id   = account_id,
        entity_name  = (form.get("entity_name") or "").strip(),
        column_name  = (form.get("column_name") or "").strip(),
        role         = (form.get("role") or "dimension").strip(),
        display_name = (form.get("display_name") or "").strip(),
        synonyms     = (form.get("synonyms") or "").strip(),
    )
    return RedirectResponse(f"/admin/clients/{account_id}/graph?saved=1", status_code=303)


@router.get("/clients/{account_id}/graph/api/graph.json")
async def graph_json_api(request: Request, account_id: str):
    """JSON snapshot of the full entity graph — used by the live diagram."""
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    graph = store.get_full_graph(account_id)
    return JSONResponse(graph)


@router.get("/clients/{account_id}/graph/api/resolve")
async def graph_resolve_api(request: Request, account_id: str):
    """Test the resolver against a sample question — admin diagnostic endpoint."""
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    question = request.query_params.get("q", "")
    db_type  = request.query_params.get("db_type", "azure_sql")
    from core.graph_resolver import resolve_for_question
    result = resolve_for_question(
        question=question,
        account_id=account_id,
        db_type=db_type,
    )
    return JSONResponse(result)



@router.post("/clients/{account_id}/graph/api/entities")
async def graph_api_entity_upsert(request: Request, account_id: str):
    """JSON API — create or update entity (for canvas drag saves)."""
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    data = await request.json()
    eid = store.save_entity(
        account_id    = account_id,
        entity_name   = data.get("entity_name", "").strip(),
        table_name    = data.get("table_name", "").strip(),
        schema_name   = data.get("schema_name", "").strip(),
        pk_column     = data.get("pk_column", "").strip(),
        display_name  = data.get("display_name", "").strip(),
        description   = data.get("description", "").strip(),
        entity_type   = data.get("entity_type", "dimension"),
        is_active     = int(data.get("is_active", 1)),
        pos_x         = float(data.get("pos_x", 120)),
        pos_y         = float(data.get("pos_y", 120)),
        color         = data.get("color", "#4F86C6"),
        entity_filter = data.get("entity_filter", "").strip(),
    )
    return JSONResponse({"status": "ok", "id": eid})


@router.delete("/clients/{account_id}/graph/api/entities/{entity_name}")
async def graph_api_entity_delete(request: Request, account_id: str, entity_name: str):
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    store.delete_entity(account_id, entity_name)
    return JSONResponse({"status": "ok"})


@router.patch("/clients/{account_id}/graph/api/entities/{entity_name}/position")
async def graph_api_entity_position(request: Request, account_id: str, entity_name: str):
    """Update only pos_x/pos_y without touching other entity fields."""
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    data = await request.json()
    with _get_db() as conn:
        conn.execute(
            "UPDATE entity_graph SET pos_x=?, pos_y=? WHERE account_id=? AND entity_name=?",
            (float(data.get("pos_x", 0)), float(data.get("pos_y", 0)), account_id, entity_name),
        )
    return JSONResponse({"status": "ok"})


@router.patch("/clients/{account_id}/graph/api/entities/{entity_name}/type")
async def graph_api_entity_type(request: Request, account_id: str, entity_name: str):
    """Update only entity_type (and color) without touching other entity fields."""
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    data = await request.json()
    new_type  = data.get("entity_type", "dimension")
    new_color = data.get("color", "")
    with _get_db() as conn:
        if new_color:
            conn.execute(
                "UPDATE entity_graph SET entity_type=?, color=? WHERE account_id=? AND entity_name=?",
                (new_type, new_color, account_id, entity_name),
            )
        else:
            conn.execute(
                "UPDATE entity_graph SET entity_type=? WHERE account_id=? AND entity_name=?",
                (new_type, account_id, entity_name),
            )
    return JSONResponse({"status": "ok"})


@router.patch("/clients/{account_id}/graph/api/entities/{entity_name}/filter")
async def graph_api_entity_filter(request: Request, account_id: str, entity_name: str):
    """Update only entity_filter (row-level WHERE clause) without touching other fields."""
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    data = await request.json()
    entity_filter = (data.get("entity_filter") or "").strip()
    with _get_db() as conn:
        conn.execute(
            "UPDATE entity_graph SET entity_filter=? WHERE account_id=? AND entity_name=?",
            (entity_filter, account_id, entity_name),
        )
    return JSONResponse({"status": "ok"})


@router.post("/clients/{account_id}/graph/api/relationships")
async def graph_api_rel_upsert(request: Request, account_id: str):
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    data = await request.json()
    raw_jc = data.get("join_conditions") or []
    join_conditions = [
        {"from_col": str(c.get("from_col","")).strip(), "to_col": str(c.get("to_col","")).strip()}
        for c in raw_jc if isinstance(c, dict)
        if str(c.get("from_col","")).strip() and str(c.get("to_col","")).strip()
    ]
    _api_from_entity = data.get("from_entity", "").strip()
    _api_to_entity   = data.get("to_entity", "").strip()
    _api_from_col    = data.get("from_column", "").strip()
    _api_to_col      = data.get("to_column", "").strip()
    _api_join_type   = data.get("join_type", "LEFT")
    _api_label       = data.get("label", "").strip()
    rid = store.save_relationship(
        account_id        = account_id,
        from_entity       = _api_from_entity,
        to_entity         = _api_to_entity,
        from_column       = _api_from_col,
        to_column         = _api_to_col,
        relationship_type = data.get("relationship_type", "many_to_one"),
        join_type         = _api_join_type,
        label             = _api_label,
        join_conditions   = join_conditions,
        where_clause      = data.get("where_clause", "").strip(),
        rel_id            = int(data.get("rel_id") or 0),
    )
    # Sync confirmed relationship into the structured semantic model (S2-2).
    try:
        from core.semantic_model import patch_relationship
        _sm_state  = store.get_client_state(account_id)
        _sm_kb_dir = (_sm_state or {}).get("kb_dir") or ""
        if _sm_kb_dir and _api_from_entity and _api_to_entity:
            _fe = store.get_entity(account_id, _api_from_entity)
            _te = store.get_entity(account_id, _api_to_entity)
            if _fe and _te:
                def _qtable(e: dict) -> str:
                    sn = (e.get("schema_name") or "").strip()
                    tn = (e.get("table_name") or "").strip()
                    return f"{sn}.{tn}" if sn else tn
                # Use first join_condition pair if available, else top-level columns.
                _jc_from = join_conditions[0]["from_col"] if join_conditions else _api_from_col
                _jc_to   = join_conditions[0]["to_col"]   if join_conditions else _api_to_col
                patch_relationship(
                    kb_dir=_sm_kb_dir,
                    from_table=_qtable(_fe),
                    to_table=_qtable(_te),
                    from_column=_jc_from,
                    to_column=_jc_to,
                    join_type=_api_join_type,
                    display_column=_api_label,
                    status="approved",
                )
    except Exception as _sm_exc:
        log.warning("patch_relationship (api_upsert) skipped: %s", _sm_exc)
    return JSONResponse({"status": "ok", "id": rid})


@router.post("/clients/{account_id}/graph/api/relationships/validate-all")
async def graph_api_rel_validate_all(request: Request, account_id: str):
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    client = store.get_client(account_id)
    if not client:
        raise HTTPException(status_code=404)

    try:
        body = await request.json()
    except Exception:
        body = {}
    execute = bool(body.get("execute", False))
    timeout_seconds = int(body.get("timeout_seconds") or 20)

    from core.relationship_validator import validate_relationship

    relationships = store.list_relationships(account_id, active_only=True)
    if not relationships:
        return JSONResponse({"status": "ok", "results": []})

    loop = asyncio.get_running_loop()

    # Fire all validations concurrently so a 20-rel graph doesn't take 400s.
    futures = [
        loop.run_in_executor(
            None,
            lambda rid=rel["id"]: validate_relationship(
                account_id,
                int(rid),
                execute=execute,
                timeout_seconds=timeout_seconds,
            ),
        )
        for rel in relationships
    ]
    validation_results = await asyncio.gather(*futures, return_exceptions=True)

    results = []
    for rel, result in zip(relationships, validation_results):
        if isinstance(result, Exception):
            log.warning("validate-all error for rel %s: %s", rel["id"], result)
            results.append({"relationship_id": int(rel["id"]), "status": "warning",
                            "message": f"Validation error: {result}"})
            continue
        store.update_relationship_validation(
            account_id,
            int(rel["id"]),
            result.status,
            row_count_estimate=result.row_count_estimate,
            join_multiplicity=result.join_multiplicity,
        )
        results.append(result.to_dict())

    return JSONResponse({"status": "ok", "results": results})


@router.post("/clients/{account_id}/graph/api/relationships/{rel_id}/validate")
async def graph_api_rel_validate(request: Request, account_id: str, rel_id: int):
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    client = store.get_client(account_id)
    if not client:
        raise HTTPException(status_code=404)

    try:
        body = await request.json()
    except Exception:
        body = {}
    execute = bool(body.get("execute", False))
    timeout_seconds = int(body.get("timeout_seconds") or 20)

    from core.relationship_validator import validate_relationship

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: validate_relationship(
            account_id,
            rel_id,
            execute=execute,
            timeout_seconds=timeout_seconds,
        ),
    )
    store.update_relationship_validation(
        account_id,
        rel_id,
        result.status,
        row_count_estimate=result.row_count_estimate,
        join_multiplicity=result.join_multiplicity,
    )
    return JSONResponse({"status": "ok", "result": result.to_dict()})


@router.delete("/clients/{account_id}/graph/api/relationships/{rel_id}")
async def graph_api_rel_delete(request: Request, account_id: str, rel_id: int):
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    store.delete_relationship(account_id, rel_id)
    return JSONResponse({"status": "ok"})


@router.post("/clients/{account_id}/graph/api/relationships/bulk")
async def graph_api_rel_bulk(request: Request, account_id: str):
    """Bulk update and/or delete relationships in one round-trip.

    Body JSON:
      updates: [{id, from_entity, to_entity, from_column, to_column,
                 join_type, relationship_type, label, where_clause}, …]
      deletes: [id, …]
    """
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    data    = await request.json()
    updates = data.get("updates", [])
    deletes = data.get("deletes", [])

    for rel_id in deletes:
        try:
            store.delete_relationship(account_id, int(rel_id))
        except Exception:
            pass

    for u in updates:
        store.save_relationship(
            account_id        = account_id,
            from_entity       = str(u.get("from_entity", "")).strip(),
            to_entity         = str(u.get("to_entity", "")).strip(),
            from_column       = str(u.get("from_column", "")).strip(),
            to_column         = str(u.get("to_column", "")).strip(),
            relationship_type = str(u.get("relationship_type", "many_to_one")),
            join_type         = str(u.get("join_type", "LEFT")),
            label             = str(u.get("label", "")).strip(),
            where_clause      = str(u.get("where_clause", "")).strip(),
            rel_id            = int(u.get("id") or 0),
        )

    return JSONResponse({"status": "ok", "updated": len(updates), "deleted": len(deletes)})


@router.post("/clients/{account_id}/graph/api/relationships/bulk")
async def graph_api_rel_bulk(request: Request, account_id: str):
    """Bulk update + delete relationships in one round-trip (used by Bulk Relationship Manager)."""
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    data    = await request.json()
    updates = data.get("updates", [])
    deletes = data.get("deletes", [])

    for rel_id in deletes:
        store.delete_relationship(account_id, int(rel_id))

    for u in updates:
        store.save_relationship(
            account_id        = account_id,
            from_entity       = str(u.get("from_entity", "")).strip(),
            to_entity         = str(u.get("to_entity", "")).strip(),
            from_column       = str(u.get("from_column", "")).strip(),
            to_column         = str(u.get("to_column", "")).strip(),
            relationship_type = str(u.get("relationship_type", "many_to_one")),
            join_type         = str(u.get("join_type", "LEFT")),
            label             = str(u.get("label", "")).strip(),
            where_clause      = str(u.get("where_clause", "")).strip(),
            rel_id            = int(u.get("id") or 0),
        )

    return JSONResponse({"status": "ok", "updated": len(updates), "deleted": len(deletes)})


@router.post("/clients/{account_id}/graph/api/relationships/purge-audit")
async def graph_api_purge_audit_joins(request: Request, account_id: str):
    """
    Delete all unconfirmed relationships whose FK column looks like an audit /
    ETL housekeeping column (AZ_ prefix, _UPD_TS suffix, high prevalence, etc.).

    Only removes rows with status != 'confirmed' so admin-approved joins are safe.
    Returns {deleted: N, kept: M, audit_cols: [list of column names removed]}.
    """
    if not _is_auth(request):
        raise HTTPException(status_code=401)

    from core.schema import _is_audit_column

    rels = store.list_relationships(account_id, active_only=False)

    # Build prevalence map: how many relationships use each from_column name?
    col_counts: dict[str, int] = {}
    for r in rels:
        c = (r.get("from_column") or "").upper()
        col_counts[c] = col_counts.get(c, 0) + 1

    total_rels = max(len(rels), 1)

    deleted      = 0
    kept         = 0
    audit_cols   = set()

    for r in rels:
        # Never touch confirmed / admin-approved relationships
        if r.get("status") == "confirmed":
            kept += 1
            continue

        col_upper = (r.get("from_column") or "").upper()
        prevalence_count = col_counts.get(col_upper, 0)

        if _is_audit_column(col_upper, total_rels, prevalence_count):
            store.delete_relationship(account_id, int(r["id"]))
            audit_cols.add(col_upper)
            deleted += 1
        else:
            kept += 1

    return JSONResponse({
        "status":     "ok",
        "deleted":    deleted,
        "kept":       kept,
        "audit_cols": sorted(audit_cols),
    })


@router.post("/clients/{account_id}/graph/api/properties")
async def graph_api_prop_save(request: Request, account_id: str):
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    data = await request.json()
    store.save_entity_property(
        account_id   = account_id,
        entity_name  = data.get("entity_name", ""),
        column_name  = data.get("column_name", ""),
        role         = data.get("role", "dimension"),
        display_name = data.get("display_name", ""),
        synonyms     = data.get("synonyms", ""),
    )
    # Sync to semantic layer business terms
    if data.get("display_name") and data.get("column_name"):
        try:
            store.save_term(
                account_id   = account_id,
                term         = data["display_name"].strip(),
                column_name  = data["column_name"].strip(),
                table_hint   = "",
                is_active    = 1,
                source       = "entity_graph",
            )
        except Exception:
            pass
    return JSONResponse({"status": "ok"})


@router.get("/clients/{account_id}/graph/api/schema-tables")
async def graph_api_schema_tables(request: Request, account_id: str):
    """
    Return all discovered tables for this client, grouped by schema.
    Used to populate the table dropdown in the entity and relationship modals.
    Response: [{"fqn": "DB.SCHEMA.TABLE", "schema_name": "dbo",
                "table_name": "DIM_CUSTOMER", "column_count": 12}]
    """
    if not _is_auth(request):
        raise HTTPException(status_code=401)

    client = store.get_client(account_id)
    if not client:
        raise HTTPException(status_code=404)

    state      = store.get_client_state(account_id)
    schema_dir = (state or {}).get("schema_dir") or ""

    from pathlib import Path as _Path
    import json as _json

    schema_path = _Path(schema_dir) / "_schema.json" if schema_dir else None
    if not schema_path or not schema_path.exists():
        return JSONResponse([])

    from core.schema import _normalize_schema as _ns
    master = _ns(_json.loads(schema_path.read_text(encoding="utf-8")))
    tables = []
    for fqn, info in master.items():
        parts       = fqn.split(".")
        table_name  = parts[-1]
        schema_name = parts[-2] if len(parts) >= 2 else ""
        tables.append({
            "fqn":          fqn,
            "schema_name":  schema_name,
            "table_name":   table_name,
            "column_count": len(info.get("columns", [])),
        })

    tables.sort(key=lambda t: (t["schema_name"], t["table_name"]))
    return JSONResponse(tables)


@router.get("/clients/{account_id}/graph/api/columns")
async def graph_api_columns(request: Request, account_id: str):
    """
    Return column list for a specific table FQN.
    Query param: fqn=DB.SCHEMA.TABLE
    Response: [{"name": "CustomerID", "type": "int", "nullable": false}]
    """
    if not _is_auth(request):
        raise HTTPException(status_code=401)

    fqn = request.query_params.get("fqn", "").strip()
    if not fqn:
        return JSONResponse([])

    client = store.get_client(account_id)
    if not client:
        raise HTTPException(status_code=404)

    state      = store.get_client_state(account_id)
    schema_dir = (state or {}).get("schema_dir") or ""

    from pathlib import Path as _Path
    import json as _json

    schema_path = _Path(schema_dir) / "_schema.json" if schema_dir else None
    if not schema_path or not schema_path.exists():
        return JSONResponse([])

    from core.schema import _normalize_schema as _ns
    master = _ns(_json.loads(schema_path.read_text(encoding="utf-8")))
    # Exact match first, then case-insensitive fallback
    info = master.get(fqn)
    if info is None:
        fqn_upper = fqn.upper()
        for k, v in master.items():
            if k.upper() == fqn_upper:
                info = v
                break
    if info is None:
        return JSONResponse([])

    cols = [
        {
            "name":     c.get("name", ""),
            "type":     c.get("type", ""),
            "nullable": c.get("nullable", True),
        }
        for c in info.get("columns", [])
    ]
    return JSONResponse(cols)


@router.get("/clients/{account_id}/graph/api/health")
async def graph_health_api(request: Request, account_id: str):
    """
    Returns a full health report for the entity graph of an account.

    Checks for schema drift, orphaned relationships, disconnected entities,
    missing properties, and tables not yet mapped to any entity.

    Response shape:
      {score, entity_count, relationship_count, property_count, has_schema,
       issues: [{severity, code, entity, message}],
       unmapped_tables: [{fqn, table_name, schema_name}],
       entity_severity: {entity_name: "error"|"warning"|"info"},
       error_count, warning_count, info_count}
    """
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    client = store.get_client(account_id)
    if not client:
        raise HTTPException(status_code=404)

    from core.graph_health import check_graph_health
    try:
        report = check_graph_health(account_id)
        return JSONResponse(report.to_dict())
    except Exception as exc:
        log.exception("graph health check failed for %s", account_id)
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/clients/{account_id}/graph/api/suggest")
async def graph_suggest(request: Request, account_id: str):
    """
    LLM suggestion pipeline.
    Reads _schema.json + _join_map.md from the client's schema_dir,
    calls GPT-4o to predict entities, field roles, and relationships,
    and saves them with status='suggested' and LLM-generated confidence scores.
    Role-playing dimensions (same table, multiple FK columns pointing to it)
    are detected deterministically and generate separate entity entries.
    """
    if not _is_auth(request):
        raise HTTPException(status_code=401)

    client = store.get_client(account_id)
    if not client:
        raise HTTPException(status_code=404)

    state = store.get_client_state(account_id)
    schema_dir = (state or {}).get("schema_dir") or ""
    if not schema_dir:
        return JSONResponse({"status": "error",
                             "message": "Schema not discovered yet. Run Discovery first."})

    from pathlib import Path as _Path
    import json as _json

    schema_path = _Path(schema_dir) / "_schema.json"
    join_map_path = _Path(schema_dir) / "_join_map.md"

    if not schema_path.exists():
        return JSONResponse({"status": "error",
                             "message": "_schema.json not found. Run Discovery first."})

    from core.schema import _normalize_schema
    schema = _normalize_schema(_json.loads(schema_path.read_text()))
    join_map = join_map_path.read_text() if join_map_path.exists() else ""

    # ── Deterministic pre-pass: detect role-playing dimensions ───────────────
    # A role-playing dimension = two or more FK-looking columns in a fact table
    # both ending in the same suffix that matches a dimension table name.
    role_playing: dict[str, list[dict]] = {}  # target_table -> [{"fk_col", "fact_table", "suggested_name"}]

    fact_tables = [
        t for t in schema
        if t.upper().startswith(("FACT_", "FCT_")) or "_FACT" in t.upper() or "_FCT" in t.upper()
    ]
    dim_tables  = {
        t.upper(): t for t in schema
        if not (t.upper().startswith(("FACT_", "FCT_")) or "_FACT" in t.upper() or "_FCT" in t.upper())
    }

    from core.date_roles import detect_date_role, find_date_dimension_key, is_date_dimension_table

    def _col_name(c: dict) -> str:
        return c.get("name") or c.get("COLUMN_NAME") or c.get("column_name") or ""

    def _col_type(c: dict) -> str:
        return c.get("type") or c.get("DATA_TYPE") or c.get("data_type") or ""

    date_dim_names = [
        t for t, info in schema.items()
        if is_date_dimension_table(t, info.get("columns", []))
    ]

    for fact in fact_tables:
        fact_info = schema[fact]
        col_names = [_col_name(c) for c in fact_info.get("columns", []) if _col_name(c)]
        # Group columns that end with the same suffix (e.g. DateID, DateId)
        suffix_groups: dict[str, list[str]] = {}
        for col in col_names:
            # Look for pattern: <prefix>DateID, <prefix>CustomerID
            import re as _re
            m = _re.match(r'^([A-Za-z]+)(Id|ID|_id|_ID)$', col)
            if m:
                suffix = m.group(2)
                base   = m.group(1).upper().rstrip("ID").rstrip("_")
                # Check if any dim table name contains this base
                for dim_upper, dim_name in dim_tables.items():
                    if base in dim_upper or dim_upper in base:
                        suffix_groups.setdefault(dim_name, []).append(col)

        for dim_name, fk_cols in suffix_groups.items():
            if len(fk_cols) >= 2:
                # Role-playing! Multiple FK columns pointing to same dim table
                for fk_col in fk_cols:
                    import re as _re2
                    # Build suggested entity name: strip the Id/ID suffix, use prefix
                    prefix = _re2.sub(r'(Id|ID|_id|_ID)$', '', fk_col)
                    role_playing.setdefault(dim_name, []).append({
                        "fk_col": fk_col,
                        "fact_table": fact,
                        "suggested_entity_name": prefix,
                    })

        # Date-role dimensions: multiple *_DT_DMS_KEY columns in the fact table
        # can point to the same physical date dimension but mean different
        # business dates (Invoice Date, Order Date, Delivery Date, etc.).
        if date_dim_names:
            date_dim = date_dim_names[0]
            for col in col_names:
                role = detect_date_role(col)
                if not role:
                    continue
                role_playing.setdefault(date_dim, []).append({
                    "fk_col": col,
                    "fact_table": fact,
                    "suggested_entity_name": role.label,
                    "relationship_label": role.label,
                })

    # ── LLM schema analysis ─────────────────────────────────────────────────
    db_cfg_id = client.get("db_config_id")
    db_cfg = store.get_db_config(db_cfg_id) if db_cfg_id else {}
    db_type = (db_cfg or {}).get("db_type", "azure_sql")

    # Chunk the schema so large databases are fully analysed (20 tables per
    # LLM call). A hard ceiling of 100 tables guards cost — and is REPORTED
    # in the response instead of silently truncating.
    _CHUNK = 20
    _MAX_TABLES = 100
    _all_tables = list(schema.items())
    truncated_tables = max(len(_all_tables) - _MAX_TABLES, 0)
    _chunks = [
        _all_tables[i:i + _CHUNK]
        for i in range(0, min(len(_all_tables), _MAX_TABLES), _CHUNK)
    ]

    def _chunk_summary(chunk: list) -> str:
        lines = []
        for tbl_name, tbl_info in chunk:
            cols = [f"{_col_name(c)}:{_col_type(c)}" for c in tbl_info.get("columns", [])[:15] if _col_name(c)]
            lines.append(f"{tbl_name}: {', '.join(cols)}")
        return "\n".join(lines)

    system_msg = (
        "You are a senior data warehouse architect. Given a database schema, "
        "output a structured JSON describing business entities, their field roles, "
        "and relationships between them.\n\n"
        "Rules:\n"
        "- entity_type: 'fact' for transaction/event tables (FACT_ prefix, many numeric cols), "
        "'dimension' for lookup/reference tables, 'bridge' for many-to-many junction tables\n"
        "- field role: 'metric' = numeric aggregatable (revenue, quantity, amount), "
        "'dimension' = groupable text/category, 'date' = date/datetime, "
        "'identifier' = PK or FK column, 'filter' = status/type/flag column, 'ignore' = internal\n"
        "- confidence_score: 0-100. High (85+) when type + name make the role obvious. "
        "Lower (50-70) when ambiguous. Be honest — do not inflate scores.\n"
        "- For synonyms: list 2-4 natural business terms users might say for this column. "
        "Be specific to the domain. Leave empty string if none obvious.\n\n"
        "Return ONLY valid JSON. No markdown. No explanation. No preamble."
    )

    _json_format_block = (
        "Output JSON with this exact structure:\n"
        '{"entities": [{"entity_name": "Customer", "table_name": "DIM_CUSTOMER", '
        '"schema_name": "dbo", "entity_type": "dimension", "pk_column": "CustomerID", '
        '"display_name": "Customer", "description": "...", "confidence_score": 85, '
        '"fields": [{"column_name": "Revenue", "role": "metric", "display_name": "Revenue", '
        '"synonyms": "revenue, income, sales", "confidence_score": 90}]}], '
        '"relationships": [{"from_entity": "Prescription", "to_entity": "Customer", '
        '"from_column": "CustomerID", "to_column": "CustomerID", '
        '"relationship_type": "many_to_one", "join_type": "LEFT", '
        '"label": "placed by", "confidence_score": 88}]}'
    )

    suggestions: dict = {"entities": [], "relationships": []}
    chunk_errors: list[str] = []
    try:
        from core.llm import llm_complete, resolve_provider
        from core.llm_audit import llm_audit_scope, make_llm_audit_request_id

        provider, model, api_key, az_kw = resolve_provider(client)
        request_id = make_llm_audit_request_id()

        with llm_audit_scope(
            account_id=account_id,
            question=f"Graph schema suggestion for {account_id}",
            enabled=bool(client.get("enable_llm_audit")),
            request_id=request_id,
            question_id=request_id,
            component="graph_suggestion",
        ):
            for _ci, _chunk in enumerate(_chunks):
                user_msg = (
                    f"Database type: {db_type}\n\n"
                    f"Schema (table: columns):\n{_chunk_summary(_chunk)}\n\n"
                    f"Join relationships detected:\n{join_map[:1500] if join_map else 'None detected'}\n\n"
                    + _json_format_block
                )
                try:
                    raw, _, _ = await llm_complete(
                        system_msg, user_msg, provider, model, api_key,
                        max_tokens=3000, temperature=0.2, **az_kw,
                    )
                    clean = raw.strip()
                    if clean.startswith("```"):
                        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                    parsed = _json.loads(clean)
                    suggestions["entities"].extend(parsed.get("entities", []))
                    suggestions["relationships"].extend(parsed.get("relationships", []))
                except Exception as _cexc:
                    chunk_errors.append(f"chunk {_ci + 1}/{len(_chunks)}: {_cexc}")
                    log.warning("Graph suggestion chunk %d/%d failed for %s: %s",
                                _ci + 1, len(_chunks), account_id, _cexc)

        if not suggestions["entities"] and chunk_errors:
            return JSONResponse({"status": "error", "message": "; ".join(chunk_errors)})

    except Exception as exc:
        log.error("Graph suggestion LLM call failed for %s: %s", account_id, exc)
        return JSONResponse({"status": "error", "message": str(exc)})

    # ── Save entities and fields ─────────────────────────────────────────────
    import math as _math
    _seen_ent_names: set[str] = set()
    entities = []
    for _e in suggestions.get("entities", []):
        _nm = (_e.get("entity_name") or "").strip()
        if not _nm or _nm in _seen_ent_names:
            continue
        _seen_ent_names.add(_nm)
        entities.append(_e)
    rels     = suggestions.get("relationships", [])
    saved_entities = 0
    saved_fields   = 0
    saved_rels     = 0

    # Assign canvas positions in a grid
    svg_cx, svg_cy = 600, 400
    for i, ent in enumerate(entities):
        angle = (i / max(len(entities), 1)) * 2 * _math.pi
        radius = 280 if ent.get("entity_type") == "dimension" else 0
        px = svg_cx + radius * _math.cos(angle) - 80
        py = svg_cy + radius * _math.sin(angle) - 36
        if ent.get("entity_type") == "fact":
            px = svg_cx - 80 + (i * 220)
            py = svg_cy - 36

        store.save_entity(
            account_id       = account_id,
            entity_name      = ent.get("entity_name", ""),
            table_name       = ent.get("table_name", ""),
            schema_name      = ent.get("schema_name", ""),
            pk_column        = ent.get("pk_column", ""),
            display_name     = ent.get("display_name", ""),
            description      = ent.get("description", ""),
            entity_type      = ent.get("entity_type", "dimension"),
            is_active        = 1,
            pos_x            = float(px),
            pos_y            = float(py),
            color            = "#F59E0B" if ent.get("entity_type") == "fact" else "#4F86C6",
            confidence_score = int(ent.get("confidence_score", 70)),
            status           = "suggested",
            generated_by     = "llm",
            reason           = f"LLM schema analysis classified {ent.get('table_name','')} as {ent.get('entity_type','dimension')}",
        )
        saved_entities += 1

        for field in ent.get("fields", []):
            _syn = field.get("synonyms", "") or ""
            if isinstance(_syn, list):
                _syn = ", ".join(str(s) for s in _syn)
            store.save_entity_property(
                account_id       = account_id,
                entity_name      = ent.get("entity_name", ""),
                column_name      = field.get("column_name", ""),
                role             = field.get("role", "dimension"),
                display_name     = field.get("display_name", ""),
                synonyms         = _syn,
                confidence_score = int(field.get("confidence_score", 70)),
                status           = "suggested",
            )
            saved_fields += 1

    # ── Save role-playing entities (deterministic — overrides LLM) ──────────
    for dim_name, roles in role_playing.items():
        dim_info  = schema.get(dim_name, {})
        dim_schema = dim_info.get("schema", "")
        dim_cols   = [_col_name(c) for c in dim_info.get("columns", []) if _col_name(c)]
        pk_hint    = (
            find_date_dimension_key(dim_info.get("columns", []))
            if is_date_dimension_table(dim_name, dim_info.get("columns", []))
            else next((c for c in dim_cols if c.upper().endswith("ID")), dim_cols[0] if dim_cols else "")
        )
        n_total    = len(roles)
        for ri, role_info in enumerate(roles):
            rp_name = role_info["suggested_entity_name"]
            import re as _re3
            disp = _re3.sub(r'(?<=[a-z])(?=[A-Z])', ' ', rp_name).strip()

            angle = (ri / max(n_total, 1)) * 2 * _math.pi
            px = svg_cx + 340 * _math.cos(angle + _math.pi) - 80
            py = svg_cy + 340 * _math.sin(angle + _math.pi) - 36

            store.save_entity(
                account_id       = account_id,
                entity_name      = rp_name,
                table_name       = dim_name,
                schema_name      = dim_schema,
                pk_column        = pk_hint,
                display_name     = disp,
                description      = f"Role-playing dimension: {dim_name} used as {disp}",
                entity_type      = "dimension",
                is_active        = 1,
                pos_x            = float(px),
                pos_y            = float(py),
                color            = "#7C3AED",  # purple = role-playing
                confidence_score = 92,         # high — deterministic detection
                status           = "suggested",
                generated_by     = "heuristic",
                reason           = f"Role-playing dimension: {role_info['fk_col']} in {role_info['fact_table']} points to {dim_name}",
            )

            # Save relationship from fact to this role-playing entity
            fact_name_entity = next(
                (e.get("entity_name","") for e in entities
                 if e.get("table_name","").upper() == role_info["fact_table"].upper()),
                role_info["fact_table"]
            )
            store.save_relationship(
                account_id        = account_id,
                from_entity       = fact_name_entity,
                to_entity         = rp_name,
                from_column       = role_info["fk_col"],
                to_column         = pk_hint,
                relationship_type = "many_to_one",
                join_type         = "LEFT",
                label             = role_info.get("relationship_label") or disp,
                confidence_score  = 92,
                status            = "suggested",
                generated_by      = "heuristic",
                reason            = f"Role-playing join: {role_info['fk_col']} → {dim_name}.{pk_hint}",
            )
            saved_entities += 1

    # ── Save LLM-suggested relationships (one per table pair) ───────────────
    for rel in rels:
        fe = rel.get("from_entity", "")
        te = rel.get("to_entity", "")
        if not fe or not te or fe == te:
            continue
        store.upsert_relationship_by_pair(
            account_id        = account_id,
            from_entity       = fe,
            to_entity         = te,
            from_column       = rel.get("from_column", ""),
            to_column         = rel.get("to_column", ""),
            relationship_type = rel.get("relationship_type", "many_to_one"),
            join_type         = rel.get("join_type", "LEFT"),
            label             = rel.get("label", ""),
            confidence_score  = int(rel.get("confidence_score", 70)),
            status            = "suggested",
            generated_by      = "llm",
            reason            = f"LLM suggested join on {rel.get('from_column', '')}",
        )
        saved_rels += 1

    return JSONResponse({
        "status": "ok",
        "saved_entities": saved_entities,
        "saved_fields":   saved_fields,
        "saved_rels":     saved_rels,
        "role_playing_detected": {k: len(v) for k, v in role_playing.items()},
        "tables_analysed": min(len(_all_tables), _MAX_TABLES),
        "chunks": len(_chunks),
        "tables_skipped": truncated_tables,
        "chunk_errors": chunk_errors,
    })


@router.post("/clients/{account_id}/graph/api/confirm/entity/{entity_name:path}")
async def graph_confirm_entity(request: Request, account_id: str, entity_name: str):
    """Confirm an LLM-suggested entity — marks it confirmed in the graph."""
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    with __import__("store.db", fromlist=["get_db"]).get_db() as conn:
        conn.execute(
            "UPDATE entity_graph SET status='confirmed', confidence_score=100 "
            "WHERE account_id=? AND entity_name=?",
            (account_id, entity_name)
        )
    return JSONResponse({"status": "ok"})


@router.post("/clients/{account_id}/graph/api/confirm/property")
async def graph_confirm_property(request: Request, account_id: str):
    """Confirm an LLM-suggested field — marks confirmed + syncs to semantic layer at 100%."""
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    data = await request.json()
    store.confirm_entity_property(
        account_id  = account_id,
        entity_name = data.get("entity_name",""),
        column_name = data.get("column_name",""),
    )
    return JSONResponse({"status": "ok"})


@router.post("/clients/{account_id}/graph/api/confirm/relationship/{rel_id}")
async def graph_confirm_rel(request: Request, account_id: str, rel_id: int):
    """Confirm an LLM-suggested relationship."""
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    with __import__("store.db", fromlist=["get_db"]).get_db() as conn:
        conn.execute(
            "UPDATE entity_relationships SET status='confirmed', confidence_score=100 "
            "WHERE id=? AND account_id=?",
            (rel_id, account_id)
        )
    return JSONResponse({"status": "ok"})

@router.post("/clients/{account_id}/graph/api/reject/entity/{entity_name:path}")
async def graph_reject_entity(request: Request, account_id: str, entity_name: str):
    """Reject a suggested entity — marks it rejected and deactivates it."""
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    with __import__("store.db", fromlist=["get_db"]).get_db() as conn:
        conn.execute(
            "UPDATE entity_graph SET status='rejected', is_active=0 "
            "WHERE account_id=? AND entity_name=?",
            (account_id, entity_name),
        )
    return JSONResponse({"status": "ok"})


@router.post("/clients/{account_id}/graph/api/reject/relationship/{rel_id}")
async def graph_reject_rel(request: Request, account_id: str, rel_id: int):
    """Reject a suggested relationship."""
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    with __import__("store.db", fromlist=["get_db"]).get_db() as conn:
        conn.execute(
            "UPDATE entity_relationships SET status='rejected', is_active=0 "
            "WHERE id=? AND account_id=?",
            (rel_id, account_id),
        )
    return JSONResponse({"status": "ok"})


@router.post("/clients/{account_id}/graph/api/bulk-accept")
async def graph_bulk_accept(request: Request, account_id: str):
    """Accept all suggested entities and relationships at or above min_confidence."""
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    data = await request.json()
    min_conf = int(data.get("min_confidence", 85))
    with __import__("store.db", fromlist=["get_db"]).get_db() as conn:
        e_rows = conn.execute(
            "UPDATE entity_graph SET status='confirmed', confidence_score=100 "
            "WHERE account_id=? AND status='suggested' AND COALESCE(confidence_score,0) >= ? "
            "RETURNING entity_name",
            (account_id, min_conf),
        ).fetchall()
        r_rows = conn.execute(
            "UPDATE entity_relationships SET status='confirmed', confidence_score=100 "
            "WHERE account_id=? AND status='suggested' AND COALESCE(confidence_score,0) >= ? "
            "RETURNING id",
            (account_id, min_conf),
        ).fetchall()
    return JSONResponse({
        "status":           "ok",
        "entities_accepted":      len(e_rows),
        "relationships_accepted": len(r_rows),
    })


@router.get("/clients/{account_id}/graph/api/properties/{entity_name}")
async def graph_api_props_get(request: Request, account_id: str, entity_name: str):
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    props = store.list_entity_properties(account_id, entity_name)
    return JSONResponse({"status": "ok", "properties": props})

@router.get("/clients/{account_id}/metrics", response_class=HTMLResponse)
async def metrics_page(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client  = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    metrics = store.list_metrics(account_id, active_only=False)
    # Resolve db_type so the formula helper shows correct dialect syntax
    db_type = "azure_sql"  # safe default
    db_cfg_id = client.get("db_config_id")
    if db_cfg_id:
        raw = store.get_db_config(db_cfg_id)
        if raw:
            db_type = raw.get("db_type", "azure_sql")
    return _resp(request, "client_metrics.html", {
        "client":  client,
        "metrics": metrics,
        "saved":   request.query_params.get("saved"),
        "error":   request.query_params.get("error"),
        "db_type": db_type,
    })


@router.get("/clients/{account_id}/metrics/check-collision")
async def metric_check_collision(
    request:    Request,
    account_id: str,
    phrases:    str = "",          # comma-separated metric name + synonyms being entered
    skip_id:    int = 0,           # metric_id to exclude (for edit form, skip self)
):
    """
    Check whether any of the supplied phrases (metric name + synonyms being
    typed by the admin) collide with existing active business-term aliases.

    Returns:
      { "collisions": [ { "metric_phrase": "...", "term": "...", "term_aliases": "..." }, ... ] }

    Called live from JavaScript in the metric add/edit form so the admin
    gets an inline warning before saving.
    """
    if not _is_auth(request):
        raise HTTPException(status_code=401, detail="Not authenticated")

    incoming = {p.strip().lower() for p in phrases.split(",") if p.strip()}
    if not incoming:
        return JSONResponse({"collisions": []})

    # Load active business terms for this account
    terms = store.list_terms(account_id, active_only=True)
    collisions = []
    for t in terms:
        term_forms = {(t.get("term") or "").strip().lower()}
        for alias in (t.get("aliases") or "").split(","):
            a = alias.strip().lower()
            if a:
                term_forms.add(a)
        hits = incoming & term_forms
        if hits:
            collisions.append({
                "metric_phrase": sorted(hits)[0],
                "term":          t.get("term", ""),
                "term_aliases":  t.get("aliases", ""),
            })

    return JSONResponse({"collisions": collisions})


@router.post("/clients/{account_id}/metrics/create")
async def metric_create(
    request:      Request,
    account_id:   str,
    name:         str = Form(...),
    synonyms:     str = Form(""),
    sql_template: str = Form(...),
    description:  str = Form(""),
    formula_type: str = Form("expression"),
    result_format: str = Form("number"),
    required_columns: str = Form(""),
    allowed_dimensions: str = Form(""),
    metric_builder_config: str = Form(""),
    example_questions: str = Form(""),
    grain: str = Form(""),
    category: str = Form(""),
    default_time_column: str = Form(""),
    base_table:   str = Form(""),
    base_entity:  str = Form(""),
):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    if not name.strip() or not sql_template.strip():
        from urllib.parse import quote
        return RedirectResponse(
            f"/admin/clients/{account_id}/metrics?error={quote('Name and SQL are required')}",
            status_code=303)

    try:
        from core.metric_builder import compile_metric_builder_config, merge_required_columns
        compiled = compile_metric_builder_config(metric_builder_config)
        if compiled and (formula_type or "").strip().lower() == "expression":
            sql_template = compiled.formula
            required_columns = merge_required_columns(required_columns, compiled.required_columns)
            metric_builder_config = compiled.config_json
        elif (formula_type or "").strip().lower() != "expression":
            metric_builder_config = ""
    except Exception as exc:
        from urllib.parse import quote
        return RedirectResponse(
            f"/admin/clients/{account_id}/metrics?error={quote('Metric builder filter is invalid: ' + str(exc))}",
            status_code=303)

    # Resolve db_type so save_metric can run the validator correctly
    db_type = "azure_sql"
    client  = store.get_client(account_id)
    if client and client.get("db_config_id"):
        raw = store.get_db_config(client["db_config_id"])
        if raw:
            db_type = raw.get("db_type", "azure_sql")

    # Validate ${MetricName} references before saving
    _ref_errors = store.validate_metric_refs(account_id, sql_template.strip())
    if _ref_errors:
        from urllib.parse import quote
        return RedirectResponse(
            f"/admin/clients/{account_id}/metrics?error={quote(_ref_errors[0])}",
            status_code=303)

    try:
        store.save_metric(account_id, {
            "name":                name.strip(),
            "synonyms":            synonyms.strip(),
            "sql_template":        sql_template.strip(),
            "description":         description.strip(),
            "formula_type":        formula_type.strip(),
            "result_format":       result_format.strip(),
            "required_columns":    required_columns.strip(),
            "allowed_dimensions":  allowed_dimensions.strip(),
            "metric_builder_config": metric_builder_config.strip(),
            "example_questions":   example_questions.strip(),
            "grain":               grain.strip(),
            "category":            category.strip(),
            "default_time_column": default_time_column.strip(),
            "base_table":          base_table.strip(),
            "base_entity":         base_entity.strip(),
        }, db_type=db_type)
    except ValueError as dup_exc:
        from urllib.parse import quote
        return RedirectResponse(
            f"/admin/clients/{account_id}/metrics?error={quote(str(dup_exc))}",
            status_code=303,
        )

    # Sync approved metric formula into the structured semantic model (S2-1).
    try:
        from core.semantic_model import patch_metric_approval
        _sm_state = store.get_client_state(account_id)
        _sm_kb_dir = (_sm_state or {}).get("kb_dir") or ""
        if _sm_kb_dir:
            _req_cols = [c.strip() for c in required_columns.split(",") if c.strip()]
            _base_parts = base_table.strip().split(".")
            patch_metric_approval(
                kb_dir=_sm_kb_dir,
                table_name=_base_parts[-1],
                schema_name=_base_parts[-2] if len(_base_parts) >= 2 else "",
                metric_name=name.strip(),
                column_name=_req_cols[0] if _req_cols else "",
                sql_template=sql_template.strip(),
                is_active=True,
            )
    except Exception as _sm_exc:
        log.warning("patch_metric_approval (create) skipped for %s: %s", name, _sm_exc)

    return RedirectResponse(f"/admin/clients/{account_id}/metrics?saved=1", status_code=303)


@router.post("/clients/{account_id}/metrics/{metric_id}/update")
async def metric_update(
    request:      Request,
    account_id:   str,
    metric_id:    int,
    name:         str = Form(...),
    synonyms:     str = Form(""),
    sql_template: str = Form(...),
    description:  str = Form(""),
    formula_type: str = Form("query"),
    result_format: str = Form("number"),
    required_columns: str = Form(""),
    allowed_dimensions: str = Form(""),
    metric_builder_config: str = Form(""),
    example_questions: str = Form(""),
    grain: str = Form(""),
    category: str = Form(""),
    default_time_column: str = Form(""),
    base_table:   str = Form(""),
    base_entity:  str = Form(""),
    is_active:    str = Form("1"),
):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)

    try:
        from core.metric_builder import compile_metric_builder_config, merge_required_columns
        compiled = compile_metric_builder_config(metric_builder_config)
        if compiled and (formula_type or "").strip().lower() == "expression":
            sql_template = compiled.formula
            required_columns = merge_required_columns(required_columns, compiled.required_columns)
            metric_builder_config = compiled.config_json
        elif (formula_type or "").strip().lower() != "expression":
            metric_builder_config = ""
    except Exception as exc:
        from urllib.parse import quote
        return RedirectResponse(
            f"/admin/clients/{account_id}/metrics?error={quote('Metric builder filter is invalid: ' + str(exc))}",
            status_code=303)

    # Resolve db_type for re-validation
    db_type = "azure_sql"
    client  = store.get_client(account_id)
    if client and client.get("db_config_id"):
        raw = store.get_db_config(client["db_config_id"])
        if raw:
            db_type = raw.get("db_type", "azure_sql")

    # Validate ${MetricName} references before updating
    _ref_errors_upd = store.validate_metric_refs(account_id, sql_template.strip())
    if _ref_errors_upd:
        from urllib.parse import quote
        return RedirectResponse(
            f"/admin/clients/{account_id}/metrics?error={quote(_ref_errors_upd[0])}",
            status_code=303)

    store.update_metric(metric_id, {
        "name":                name.strip(),
        "synonyms":            synonyms.strip(),
        "sql_template":        sql_template.strip(),
        "description":         description.strip(),
        "formula_type":        formula_type.strip(),
        "result_format":       result_format.strip(),
        "required_columns":    required_columns.strip(),
        "allowed_dimensions":  allowed_dimensions.strip(),
        "metric_builder_config": metric_builder_config.strip(),
        "example_questions":   example_questions.strip(),
        "grain":               grain.strip(),
        "category":            category.strip(),
        "default_time_column": default_time_column.strip(),
        "base_table":          base_table.strip(),
        "base_entity":         base_entity.strip(),
        "is_active":           int(is_active),
    }, account_id=account_id, db_type=db_type)

    # Sync updated metric formula (or deprecation) into the structured semantic model (S2-1 / S2-3).
    try:
        from core.semantic_model import patch_metric_approval
        _sm_state = store.get_client_state(account_id)
        _sm_kb_dir = (_sm_state or {}).get("kb_dir") or ""
        if _sm_kb_dir:
            _req_cols = [c.strip() for c in required_columns.split(",") if c.strip()]
            _base_parts = base_table.strip().split(".")
            patch_metric_approval(
                kb_dir=_sm_kb_dir,
                table_name=_base_parts[-1],
                schema_name=_base_parts[-2] if len(_base_parts) >= 2 else "",
                metric_name=name.strip(),
                column_name=_req_cols[0] if _req_cols else "",
                sql_template=sql_template.strip(),
                is_active=bool(int(is_active)),
            )
    except Exception as _sm_exc:
        log.warning("patch_metric_approval (update) skipped for %s: %s", name, _sm_exc)

    return RedirectResponse(f"/admin/clients/{account_id}/metrics?saved=1", status_code=303)


@router.post("/clients/{account_id}/metrics/{metric_id}/deprecate")
async def metric_deprecate(request: Request, account_id: str, metric_id: int):
    """Soft-delete: marks metric as deprecated/inactive. Preserves history.
    Accepts both JSON (fetch) and form POST (legacy fallback).
    """
    if not _is_auth(request):
        wants_json = "application/json" in request.headers.get("content-type", "")
        if wants_json:
            return JSONResponse({"status": "error", "detail": "Not authenticated"}, status_code=401)
        return RedirectResponse("/admin/login", status_code=303)
    store.deprecate_metric(metric_id, account_id)
    wants_json = "application/json" in request.headers.get("content-type", "")
    if wants_json:
        return JSONResponse({"status": "ok", "metric_id": metric_id})
    return RedirectResponse(f"/admin/clients/{account_id}/metrics", status_code=303)


@router.post("/clients/{account_id}/metrics/{metric_id}/delete")
async def metric_delete(request: Request, account_id: str, metric_id: int):
    """Hard-delete. Enforces account ownership. Returns JSON when called via fetch."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    store.delete_metric(metric_id, account_id)
    # Fetch callers (X-Requested-With: fetch) expect JSON, not a redirect
    if request.headers.get("X-Requested-With") == "fetch":
        from fastapi.responses import JSONResponse as _JSON
        return _JSON({"ok": True})
    return RedirectResponse(f"/admin/clients/{account_id}/metrics", status_code=303)


@router.post("/clients/{account_id}/metrics/validate")
async def metrics_validate(request: Request, account_id: str):
    """
    Live validation endpoint — returns structured errors/warnings for a metric
    definition without saving anything. Called by the create/edit form UI.
    """
    if not _is_auth(request):
        return JSONResponse({"status": "error", "detail": "Not authenticated"}, status_code=401)

    body = await request.json()
    if not body.get("sql_template", "").strip():
        return JSONResponse({"status": "error", "detail": "Formula/SQL is required"})

    client = store.get_client(account_id)
    if not client:
        return JSONResponse({"status": "error", "detail": "Account not found"}, status_code=404)

    # Resolve db_type for function allowlist
    db_type = "azure_sql"
    db_cfg_id = client.get("db_config_id")
    if db_cfg_id:
        raw = store.get_db_config(db_cfg_id)
        if raw:
            db_type = raw.get("db_type", "azure_sql")

    from core.metric_validator import validate_metric, load_schema_columns
    schema_columns = load_schema_columns(account_id)

    # Check metric refs separately so we can surface a clear error
    _sql = (body.get("sql_template") or "").strip()
    _ref_errs = store.validate_metric_refs(account_id, _sql)
    if _ref_errs:
        return JSONResponse({
            "status":      "ok",
            "valid":       False,
            "errors":      _ref_errs,
            "warnings":    [],
            "formula_ast": {},
        })

    result = validate_metric(body, db_type=db_type, schema_columns=schema_columns)
    return JSONResponse({
        "status":      "ok",
        "valid":       result.valid,
        "errors":      result.errors,
        "warnings":    result.warnings,
        "formula_ast": result.formula_ast,
    })


@router.post("/clients/{account_id}/metrics/test-formula")
async def metrics_test_formula(request: Request, account_id: str):
    """Run a metric formula as a live SELECT against the account's DB and return the result."""
    if not _is_auth(request):
        return JSONResponse({"status": "error", "detail": "Not authenticated"}, status_code=401)

    body = await request.json()
    formula = (body.get("formula") or "").strip()
    if not formula:
        return JSONResponse({"status": "error", "detail": "Formula is empty"})

    client = store.get_client(account_id)
    if not client:
        return JSONResponse({"status": "error", "detail": "Account not found"}, status_code=404)

    db_cfg_id = client.get("db_config_id")
    if not db_cfg_id:
        return JSONResponse({"status": "error", "detail": "No database configured for this account"})

    raw_cfg = store.get_db_config(db_cfg_id)
    if not raw_cfg:
        return JSONResponse({"status": "error", "detail": "Database config not found"})

    db_type = raw_cfg.get("db_type", "azure_sql")
    creds   = raw_cfg.get("credentials", {})

    # Find the table to anchor the query — prefer base_table from request, else first in schema
    import json as _json
    from pathlib import Path as _Path

    base_table_raw = (body.get("base_table") or "").strip()

    def _fqn_to_sql(fqn: str) -> str:
        parts = fqn.split(".")
        if db_type == "azure_sql" and len(parts) >= 2:
            return f"[{parts[-2]}].[{parts[-1]}]"
        if db_type == "snowflake" and len(parts) >= 3:
            return f'"{parts[0]}"."{parts[1]}"."{parts[2]}"'
        if db_type == "oracle" and len(parts) >= 2:
            return f'"{parts[-2]}"."{parts[-1]}"'
        return ""

    first_table_sql = None
    master = {}

    state      = store.get_client_state(account_id)
    schema_dir = (state or {}).get("schema_dir") or ""
    schema_path = _Path(schema_dir) / "_schema.json" if schema_dir else None
    if schema_path and schema_path.exists():
        master = _json.loads(schema_path.read_text(encoding="utf-8"))

    if base_table_raw:
        first_table_sql = _fqn_to_sql(base_table_raw)

    if not first_table_sql:
        for fqn in master:
            first_table_sql = _fqn_to_sql(fqn)
            if first_table_sql:
                break

    if not master and not first_table_sql:
        return JSONResponse({"status": "error", "detail": "No tables found in schema — run discovery first"})

    # ── Detect which schema columns the formula references ──────────────────
    import re as _re
    # Extract bare identifiers (skip SQL keywords and function names)
    _SQL_KW = {
        "SELECT","FROM","WHERE","AND","OR","NOT","IN","IS","NULL","AS","CASE",
        "WHEN","THEN","ELSE","END","BETWEEN","LIKE","TOP","LIMIT","DISTINCT",
        "WITH","NOLOCK","SUM","AVG","COUNT","MIN","MAX","COALESCE","NULLIF",
        "ISNULL","NVL","CAST","CONVERT","ROUND","ABS","FLOOR","CEILING","IF",
        "IIF","DATEDIFF","DATEADD","DATE","LEFT","RIGHT","MID","LEN","TRIM",
        "UPPER","LOWER","REPLACE","SUBSTRING","CONCAT","ROWNUM",
    }
    tokens = [t.upper() for t in _re.findall(r'\b[A-Za-z_][A-Za-z0-9_]{2,}\b', formula)
              if t.upper() not in _SQL_KW]

    # Build column→table_fqn map from _schema.json
    col_to_tables: dict[str, list[str]] = {}  # col_upper → [fqn, ...]
    if master:
        for fqn_key, tbl_data in master.items():
            if not isinstance(tbl_data, dict):
                continue
            for col in (tbl_data.get("columns") or []):
                col_name = (col.get("name") if isinstance(col, dict) else str(col) or "").upper()
                if col_name:
                    col_to_tables.setdefault(col_name, []).append(fqn_key)

    # Map each token to its table (first match wins when unambiguous)
    table_to_cols: dict[str, list[str]] = {}   # fqn → [col, ...]
    unresolved: list[str] = []
    for tok in tokens:
        if tok in col_to_tables:
            fqn_key = col_to_tables[tok][0]    # pick first table that has this column
            table_to_cols.setdefault(fqn_key, []).append(tok)
        # else: could be a literal/alias/keyword we didn't strip — ignore

    multi_table = len(table_to_cols) > 1
    single_table_fqn = next(iter(table_to_cols)) if len(table_to_cols) == 1 else None

    def _tbl_sql(fqn: str) -> str:
        parts = fqn.split(".")
        if db_type == "azure_sql" and len(parts) >= 2:
            return f"[{parts[-2]}].[{parts[-1]}]"
        if db_type == "snowflake" and len(parts) >= 3:
            return f'"{parts[0]}"."{parts[1]}"."{parts[2]}"'
        if db_type == "oracle" and len(parts) >= 2:
            return f'"{parts[-2]}"."{parts[-1]}"'
        return fqn

    try:
        from core.schema import _az_connect, _sf_connect, _ora_connect

        def _get_conn():
            if db_type == "azure_sql":   return _az_connect(creds)
            if db_type == "snowflake":   return _sf_connect(creds)
            return _ora_connect(creds)

        def _run_probe(sql: str):
            conn = _get_conn()
            try:
                cur = conn.cursor()
                cur.execute(sql)
                row = cur.fetchone()
                return row[0] if row else None
            finally:
                conn.close()

        loop = asyncio.get_running_loop()

        if multi_table:
            # Per-table column existence probes
            probe_results = []
            all_ok = True
            for fqn_key, cols in table_to_cols.items():
                tbl_sql = _tbl_sql(fqn_key)
                col_list = ", ".join(cols)
                if db_type == "azure_sql":
                    probe = f"SELECT TOP 1 {col_list} FROM {tbl_sql} WITH (NOLOCK)"
                elif db_type == "snowflake":
                    probe = f"SELECT {col_list} FROM {tbl_sql} LIMIT 1"
                else:
                    probe = f"SELECT {col_list} FROM {tbl_sql} WHERE ROWNUM <= 1"
                try:
                    await asyncio.wait_for(loop.run_in_executor(None, _run_probe, probe), timeout=15)
                    probe_results.append(f"✓ {fqn_key.split('.')[-1]} ({col_list})")
                except Exception as exc:
                    probe_results.append(f"✗ {fqn_key.split('.')[-1]} ({col_list}): {exc}")
                    all_ok = False

            summary = f"Multi-table formula — {len(table_to_cols)} tables probed:\n" + "\n".join(probe_results)
            if all_ok:
                return JSONResponse({"status": "ok", "result": summary})
            else:
                return JSONResponse({"status": "error", "detail": summary})

        else:
            # Single table — run the full formula expression
            tbl_sql = _tbl_sql(single_table_fqn) if single_table_fqn else first_table_sql
            if not tbl_sql:
                return JSONResponse({"status": "error", "detail": "No tables found in schema — run discovery first"})
            if db_type == "azure_sql":
                probe = f"SELECT TOP 1 ({formula}) AS _result FROM {tbl_sql} WITH (NOLOCK)"
            elif db_type == "snowflake":
                probe = f"SELECT ({formula}) AS _result FROM {tbl_sql} LIMIT 1"
            else:
                probe = f"SELECT ({formula}) AS _result FROM {tbl_sql} WHERE ROWNUM <= 1"

            result = await asyncio.wait_for(loop.run_in_executor(None, _run_probe, probe), timeout=20)
            if result is not None and not isinstance(result, (int, float, str, bool)):
                result = str(result)
            return JSONResponse({"status": "ok", "result": result})

    except asyncio.TimeoutError:
        return JSONResponse({"status": "error", "detail": "Query timed out (20 s)"})
    except Exception as exc:
        return JSONResponse({"status": "error", "detail": str(exc)})


# ══════════════════════════════════════════════════════════════════════════════
# Model Health — semantic model health dashboard (S4-1 / S4-2)
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/clients/{account_id}/model-health", response_class=HTMLResponse)
async def model_health_page(request: Request, account_id: str):
    """Render the semantic model health dashboard."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        raise HTTPException(status_code=404)

    state  = store.get_client_state(account_id)
    kb_dir = (state or {}).get("kb_dir") or ""

    health: dict = {"has_model": False}
    if kb_dir:
        try:
            from core.semantic_model import get_model_health
            health = get_model_health(kb_dir)
        except Exception as exc:
            log.warning("model_health_page: failed for %s: %s", account_id, exc)

    return _resp(request, "client_model_health.html", {
        "client": client,
        "health": health,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Date Roles — semantic model date dimension admin (S3-1)
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/clients/{account_id}/date-roles", response_class=HTMLResponse)
async def date_roles_page(request: Request, account_id: str):
    """List all date roles detected in the structured semantic model."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        raise HTTPException(status_code=404)

    state = store.get_client_state(account_id)
    kb_dir = (state or {}).get("kb_dir") or ""

    date_roles: list[dict] = []
    has_model = False

    if kb_dir:
        try:
            from core.semantic_model import load_semantic_model
            model = load_semantic_model(kb_dir)
            if model:
                has_model = True
                # Use top-level date_roles for the canonical list — it aggregates
                # across all fact tables and deduplicates by fact_table+fact_column.
                date_roles = [dict(dr) for dr in (model.get("date_roles") or [])]
                # Sort: unapproved first (need attention), then by fact_table + name
                date_roles.sort(key=lambda r: (
                    0 if r.get("status") != "approved" else 1,
                    str(r.get("fact_table") or ""),
                    str(r.get("name") or ""),
                ))
        except Exception as exc:
            log.warning("date_roles_page: failed to load semantic model for %s: %s", account_id, exc)

    return _resp(request, "client_date_roles.html", {
        "client":     client,
        "date_roles": date_roles,
        "has_model":  has_model,
        "saved":      request.query_params.get("saved"),
        "error":      request.query_params.get("error"),
    })


@router.post("/clients/{account_id}/date-roles/approve")
async def date_role_approve(
    request:         Request,
    account_id:      str,
    fact_table:      str = Form(...),
    fact_column:     str = Form(...),
    dimension_table: str = Form(""),
    dimension_key:   str = Form(""),
    business_role:   str = Form(""),
):
    """Approve (and optionally edit) a date role entry in the semantic model."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)

    client = store.get_client(account_id)
    if not client:
        raise HTTPException(status_code=404)

    state = store.get_client_state(account_id)
    kb_dir = (state or {}).get("kb_dir") or ""
    if not kb_dir:
        from urllib.parse import quote
        return RedirectResponse(
            f"/admin/clients/{account_id}/date-roles?error={quote('KB directory not configured')}",
            status_code=303,
        )

    try:
        from core.semantic_model import patch_date_role
        changed = patch_date_role(
            kb_dir=kb_dir,
            fact_table=fact_table.strip(),
            fact_column=fact_column.strip(),
            dimension_table=dimension_table.strip(),
            dimension_key=dimension_key.strip(),
            business_role=business_role.strip(),
            status="approved",
        )
        if not changed:
            from urllib.parse import quote
            return RedirectResponse(
                f"/admin/clients/{account_id}/date-roles?error={quote('Date role not found in semantic model')}",
                status_code=303,
            )
    except Exception as exc:
        log.exception("date_role_approve: unexpected error for %s: %s", account_id, exc)
        from urllib.parse import quote
        return RedirectResponse(
            f"/admin/clients/{account_id}/date-roles?error={quote(str(exc))}",
            status_code=303,
        )

    return RedirectResponse(f"/admin/clients/{account_id}/date-roles?saved=1", status_code=303)


# ══════════════════════════════════════════════════════════════════════════════
# Business glossary (semantic layer) — CRUD + auto-populate
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/clients/{account_id}/glossary", response_class=HTMLResponse)
async def glossary_page(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    terms = store.list_terms(account_id, active_only=False)
    # Parse clarification_options JSON for each term so template can render
    import json as _json
    for t in terms:
        opts_raw = t.get("clarification_options") or ""
        if opts_raw:
            try:
                t["clarification_options_parsed"] = _json.loads(opts_raw)
            except Exception:
                t["clarification_options_parsed"] = []
        else:
            t["clarification_options_parsed"] = []
    stats = store.glossary_stats(account_id)
    return _resp(request, "client_glossary.html", {
        "client": client,
        "terms":  terms,
        "stats":  stats,
        "saved":  request.query_params.get("saved"),
        "error":  request.query_params.get("error"),
    })


@router.post("/clients/{account_id}/glossary/create")
async def glossary_create(
    request:              Request,
    account_id:           str,
    term:                 str = Form(...),
    kind:                 str = Form("metric"),
    canonical_expression: str = Form(""),
    tables_involved:      str = Form(""),
    aliases:              str = Form(""),
    definition:           str = Form(""),
    requires_clarification: str = Form("0"),
    clarification_options_json: str = Form(""),
):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    from urllib.parse import quote
    if not term.strip():
        return RedirectResponse(
            f"/admin/clients/{account_id}/glossary?error={quote('Term is required')}",
            status_code=303)
    # Parse optional JSON-formatted clarification options
    import json as _json
    opts = None
    if clarification_options_json.strip():
        try:
            opts = _json.loads(clarification_options_json)
            if not isinstance(opts, list):
                opts = None
        except Exception:
            return RedirectResponse(
                f"/admin/clients/{account_id}/glossary?error={quote('Clarification options must be valid JSON array')}",
                status_code=303)
    try:
        store.save_term(
            account_id=account_id,
            term=term.strip(),
            kind=kind.strip() or "metric",
            canonical_expression=canonical_expression.strip(),
            tables_involved=tables_involved.strip(),
            aliases=aliases.strip(),
            definition=definition.strip(),
            requires_clarification=(requires_clarification == "1"),
            clarification_options=opts,
            source="manual",
        )
    except Exception as e:
        log.error("glossary_create failed: %s", e)
        return RedirectResponse(
            f"/admin/clients/{account_id}/glossary?error={quote(str(e)[:100])}",
            status_code=303)
    return RedirectResponse(
        f"/admin/clients/{account_id}/glossary?saved=1", status_code=303)


@router.post("/clients/{account_id}/glossary/{term_id}/update")
async def glossary_update(
    request:              Request,
    account_id:           str,
    term_id:              int,
    term:                 str = Form(...),
    kind:                 str = Form("metric"),
    canonical_expression: str = Form(""),
    tables_involved:      str = Form(""),
    aliases:              str = Form(""),
    definition:           str = Form(""),
    requires_clarification: str = Form("0"),
    clarification_options_json: str = Form(""),
    is_active:            str = Form("1"),
):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    from urllib.parse import quote
    import json as _json
    opts = None
    if clarification_options_json.strip():
        try:
            opts = _json.loads(clarification_options_json)
            if not isinstance(opts, list):
                opts = None
        except Exception:
            return RedirectResponse(
                f"/admin/clients/{account_id}/glossary?error={quote('Clarification options must be valid JSON array')}",
                status_code=303)
    # Mark as manual when the admin edits — otherwise a KB rebuild would overwrite
    store.save_term(
        account_id=account_id,
        term=term.strip(),
        kind=kind.strip() or "metric",
        canonical_expression=canonical_expression.strip(),
        tables_involved=tables_involved.strip(),
        aliases=aliases.strip(),
        definition=definition.strip(),
        requires_clarification=(requires_clarification == "1"),
        clarification_options=opts,
        source="manual",
        term_id=term_id,
    )
    store.set_term_active(term_id, account_id, is_active == "1")
    return RedirectResponse(
        f"/admin/clients/{account_id}/glossary?saved=1", status_code=303)


@router.post("/clients/{account_id}/glossary/{term_id}/delete")
async def glossary_delete(request: Request, account_id: str, term_id: int):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    store.delete_term(term_id, account_id)
    return RedirectResponse(
        f"/admin/clients/{account_id}/glossary", status_code=303)


@router.post("/clients/{account_id}/glossary/auto-populate")
async def glossary_auto_populate(request: Request, account_id: str):
    """Re-run term extraction from Stage 1 KB markdown files.
    
    Does NOT overwrite manually-edited entries (source='manual').
    Only refreshes source='kb_extracted' rows.
    """
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    from urllib.parse import quote
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    state_data = json.loads(client.get("state_data") or "{}")
    kb_dir = state_data.get("kb_dir", "")
    if not kb_dir:
        return RedirectResponse(
            f"/admin/clients/{account_id}/glossary?error={quote('KB has not been built for this client yet')}",
            status_code=303)
    try:
        count = store.extract_terms_from_kb(account_id, kb_dir)
        msg = f"Auto-populated {count} terms from the Knowledge Base"
        return RedirectResponse(
            f"/admin/clients/{account_id}/glossary?saved={quote(msg)}",
            status_code=303)
    except Exception as e:
        log.error("glossary auto-populate failed: %s", e)
        return RedirectResponse(
            f"/admin/clients/{account_id}/glossary?error={quote(str(e)[:100])}",
            status_code=303)


@router.post("/clients/{account_id}/metrics/harvest")
async def metrics_harvest(request: Request, account_id: str):
    """Step 4 — Trigger manual query log harvest from admin panel."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client     = store.get_client(account_id)
    state_data = json.loads(client.get("state_data") or "{}")
    chroma_dir = state_data.get("chroma_dir", "")
    if chroma_dir:
        from core.examples import harvest_and_embed
        added = harvest_and_embed(account_id, chroma_dir)
        from urllib.parse import quote
        return RedirectResponse(
            f"/admin/clients/{account_id}/metrics?saved=1",
            status_code=303)
    return RedirectResponse(f"/admin/clients/{account_id}/metrics", status_code=303)


# ══════════════════════════════════════════════════════════════════════════════
# Internal chat UI toggle — per client
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/clients/{account_id}/toggle-chat-ui")
async def toggle_chat_ui(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client  = store.get_client(account_id) or {}
    current = client.get("chat_ui_enabled", 0)
    store.update_client_meta(account_id, chat_ui_enabled=0 if current else 1)
    return RedirectResponse(f"/admin/clients/{account_id}?saved=1", status_code=303)


# ══════════════════════════════════════════════════════════════════════════════
# Admin KB generation — schema discovery + KB build from browser
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/clients/{account_id}/setup", response_class=HTMLResponse)
async def client_setup_page(request: Request, account_id: str):
    """Schema discovery + KB generation admin page."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)

    state_data   = json.loads(client.get("state_data") or "{}")
    schema_dir   = state_data.get("schema_dir", "")
    kb_dir       = state_data.get("kb_dir", "")
    state        = client.get("state", "NEW")
    db_cfg       = None
    kb_tables    = _parse_selected_schema_tables(state_data.get("kb_tables"))
    kb_table_source = "client" if kb_tables else "database"

    raw_db = store.list_db_configs()
    db_map = {d["id"]: d for d in raw_db}
    if client.get("db_config_id"):
        raw = db_map.get(client["db_config_id"])
        if raw:
            default_tables = _parse_selected_schema_tables(raw["credentials"].get("selected_schema_tables"))
            if not kb_tables:
                kb_tables = default_tables
            db_cfg = {
                "db_type": raw["db_type"],
                "name": raw["name"],
                "default_table_count": len(default_tables),
            }

    # Count files
    from pathlib import Path as _Path
    schema_files = sorted(_Path(schema_dir).glob("*.md")) if schema_dir and _Path(schema_dir).exists() else []
    kb_files     = sorted(_Path(kb_dir).glob("*.md"))     if kb_dir     and _Path(kb_dir).exists()     else []

    # Business description stored in state
    biz_desc = state_data.get("business_desc", client.get("business_desc", ""))

    # Parse per-schema descriptions for the multi-schema UI
    # Returns (overall_text, {SCHEMA: description})
    try:
        from core.knowledge import _parse_schema_descriptions
        _biz_overall, _biz_schemas = _parse_schema_descriptions(biz_desc)
    except Exception:
        _biz_overall, _biz_schemas = biz_desc, {}
    biz_desc_parsed = {"overall": _biz_overall, "schemas": _biz_schemas}

    # Build schema breakdown from selected tables so admin can see which
    # schemas are included in the KB scope before running discovery/build.
    schema_breakdown: dict[str, int] = {}
    for fqn in (kb_tables or []):
        parts = fqn.split(".")
        if len(parts) >= 2:
            schema_name = parts[-2].upper()
            schema_breakdown[schema_name] = schema_breakdown.get(schema_name, 0) + 1

    # KB data egress summary — shown in the Data Egress section on this page
    egress_summary = store.get_kb_egress_summary(account_id)

    # Initial masking config for the Field Masking section
    saved_masking_config = state_data.get("masking_config") or {}

    # Schema drift — populated after re-discovery when columns/tables changed
    schema_drift = state_data.get("schema_drift") or {}

    return _resp(request, "client_setup.html", {
        "client":               client,
        "state":                state,
        "db_cfg":               db_cfg,
        "schema_files":         schema_files,
        "kb_files":             kb_files,
        "kb_tables":            kb_tables,
        "kb_table_count":       len(kb_tables),
        "kb_table_source":      kb_table_source if kb_tables else "none",
        "schema_breakdown":     schema_breakdown,   # {schema_name: table_count}
        "biz_desc":             biz_desc,
        "biz_desc_parsed":      biz_desc_parsed,    # {overall, schemas: {SCHEMA: text}}
        "egress_summary":       egress_summary,
        "saved_masking_config": saved_masking_config,  # for JS init
        "schema_drift":         schema_drift,           # populated after re-discovery
        "saved":                request.query_params.get("saved"),
        "error":                request.query_params.get("error"),
    })


@router.get("/clients/{account_id}/setup/status")
async def admin_setup_status(request: Request, account_id: str):
    if not _is_auth(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    client = store.get_client(account_id)
    if not client:
        return JSONResponse({"status": "error", "message": "Client not found"}, status_code=404)
    state_data = json.loads(client.get("state_data") or "{}")
    kb_dir = state_data.get("kb_dir", "")
    schema_dir = state_data.get("schema_dir", "")
    from pathlib import Path as _Path
    kb_files = (
        [f.name for f in sorted(_Path(kb_dir).glob("*.md"))]
        if kb_dir and _Path(kb_dir).exists() else []
    )
    schema_files = (
        [f.name for f in sorted(_Path(schema_dir).glob("*.md")) if not f.name.startswith("_")]
        if schema_dir and _Path(schema_dir).exists() else []
    )
    progress = state_data.get("kb_progress") or {}
    return JSONResponse({
        "status": "ok",
        "state": client.get("state", "NEW"),
        "progress": progress,
        "kb_file_count": len(kb_files),
        "schema_file_count": len(schema_files),
        "business_desc": state_data.get("business_desc", client.get("business_desc", "")),
    })




@router.get("/clients/{account_id}/egress-log")
async def client_egress_log(request: Request, account_id: str):
    """JSON API — full KB egress log for a client. Used by admin tooling."""
    if not _is_auth(request):
        raise HTTPException(status_code=401)
    rows = store.list_kb_egress(account_id, limit=500)
    return JSONResponse({"status": "ok", "rows": rows, "count": len(rows)})


@router.get("/clients/{account_id}/schema-tree")
async def admin_schema_tree(request: Request, account_id: str, refresh: str = "0"):
    """
    Return the full database → schema → {tables, views} catalogue for a
    client's connected database as JSON.  Used by the admin panel to populate
    the schema-browser UI without a page reload.

    Query params:
      refresh=1   bypass the 24-hour in-memory cache and re-query the DB

    Response shapes:
      200  { "status": "ok",      "tree": { ... } }
      202  { "status": "cached",  "tree": { ... } }   (served from cache)
      400  { "status": "error",   "message": "..." }
    """
    if not _is_auth(request):
        raise HTTPException(status_code=401, detail="Not authenticated")

    client = store.get_client(account_id)
    if not client:
        return JSONResponse({"status": "error", "message": "Client not found"}, status_code=404)

    db_cfg_id = client.get("db_config_id")
    if not db_cfg_id:
        return JSONResponse(
            {"status": "error", "message": "No database assigned to this client — set one in Settings first"},
            status_code=400,
        )

    raw = get_db_config(db_cfg_id)
    if not raw:
        return JSONResponse({"status": "error", "message": "DB config not found"}, status_code=400)

    from core.schema_discovery import discover_schema_tree, get_cached_tree, set_cached_tree

    # Serve from cache unless caller explicitly requests a refresh
    force_refresh = (refresh == "1")
    if not force_refresh:
        cached = get_cached_tree(db_cfg_id)
        if cached is not None:
            return JSONResponse({"status": "cached", "tree": cached})

    creds   = raw["credentials"]
    db_type = raw["db_type"]

    try:
        tree = await discover_schema_tree(db_type, creds, timeout_seconds=45)
    except TimeoutError as e:
        return JSONResponse({"status": "error", "message": _db_connection_error_message(db_type, e)})
    except Exception as e:
        log.error("Schema tree discovery failed for %s: %s", account_id, e)
        return JSONResponse(
            {"status": "error", "message": _db_connection_error_message(db_type, e)},
        )

    set_cached_tree(db_cfg_id, tree)

    # Count totals for the log
    total = sum(
        len(objs["tables"]) + len(objs["views"])
        for db in tree.values()
        for objs in db.values()
    )
    log.info("Schema tree: %s → %d databases, %d total objects", account_id, len(tree), total)
    return JSONResponse({"status": "ok", "tree": tree})


@router.get("/clients/{account_id}/api/columns")
async def admin_columns_api(request: Request, account_id: str):
    """
    Return a flat, sorted list of all columns across the client's database.
    Used by the formula editor for ThoughtSpot-style column autocomplete.

    Source priority:
      1. _schema.json on disk  — written by discover_and_write(), always has
         full column+type data and survives server restarts.
      2. No schema found        — returns no_schema so the UI shows a hint.

    Response shapes:
      200 { "status": "ok",        "columns": [ {table, column, type, fqn}, ... ] }
      200 { "status": "no_schema", "columns": [] }
      404 { "status": "error",     "message": "..." }
    """
    if not _is_auth(request):
        raise HTTPException(status_code=401, detail="Not authenticated")

    client = store.get_client(account_id)
    if not client:
        return JSONResponse({"status": "error", "message": "Client not found"}, status_code=404)

    # ── Read from _schema.json (primary, disk-based, survives restarts) ──────
    import json as _json
    from pathlib import Path

    state      = store.get_client_state(account_id) or {}
    schema_dir = state.get("schema_dir", "")
    columns: list[dict] = []

    if schema_dir:
        schema_path = Path(schema_dir) / "_schema.json"
        if schema_path.exists():
            try:
                master = _json.loads(schema_path.read_text(encoding="utf-8"))
                for fqn_key, tbl_data in master.items():
                    if not isinstance(tbl_data, dict):
                        continue
                    # Extract the bare table name from the FQN key
                    # Keys are like "DB.SCHEMA.TABLE" or "SCHEMA.TABLE" or "TABLE"
                    parts    = fqn_key.split(".")
                    tbl_name = parts[-1]           # last segment = table name
                    for col in (tbl_data.get("columns") or []):
                        if isinstance(col, dict):
                            col_name = col.get("name") or ""
                            col_type = col.get("type") or ""
                        else:
                            col_name = str(col)
                            col_type = ""
                        if col_name:
                            columns.append({
                                "table":     tbl_name,
                                "table_fqn": fqn_key,
                                "column":    col_name,
                                "type":      col_type,
                                "fqn":       f"{tbl_name}.{col_name}",
                            })
            except Exception as exc:
                log.warning("admin_columns_api: failed to read _schema.json for %s: %s", account_id, exc)

    if not columns:
        return JSONResponse({"status": "no_schema", "columns": []})

    columns.sort(key=lambda c: (c["table"].lower(), c["column"].lower()))
    return JSONResponse({"status": "ok", "columns": columns})


@router.post("/clients/{account_id}/kb-tables")
async def admin_save_kb_tables(
    request: Request,
    account_id: str,
):
    """
    Save the admin's table selection for KB generation.
    Body: JSON  { "tables": ["TABLE_A", "TABLE_B", ...] }
    Tables are stored as uppercase refs, preferably DB.SCHEMA.TABLE, in
    state_data.kb_tables.
    Returns JSON { "status": "ok", "count": N }
    """
    if not _is_auth(request):
        raise HTTPException(status_code=401, detail="Not authenticated")

    client = store.get_client(account_id)
    if not client:
        return JSONResponse({"status": "error", "message": "Client not found"}, status_code=404)

    try:
        body = await request.json()
        tables = _parse_selected_schema_tables(body.get("tables") or [])
        # masking_config: { "DB.SCHEMA.TABLE": {"mode": "none"|"all"|"selective"|"auto",
        #                                        "masked_fields": ["ColA", ...]} }
        raw_mc = body.get("masking_config") or {}
        masking_config = {k.upper(): v for k, v in raw_mc.items()
                          if isinstance(v, dict) and v.get("mode") in ("none","all","selective","auto")}
    except Exception:
        return JSONResponse({"status": "error", "message": "Invalid JSON body"}, status_code=400)

    if not tables:
        return JSONResponse({"status": "error", "message": "No tables provided"}, status_code=400)

    # Merge into existing state_data — don't overwrite schema_dir / kb_dir etc.
    state_data = json.loads(client.get("state_data") or "{}")
    state_data["kb_tables"]      = tables
    state_data["masking_config"] = masking_config
    # Remove old synthetic_flags key if present from a previous install
    state_data.pop("synthetic_flags", None)
    store.update_client_state(
        account_id,
        client.get("state") or "PENDING",
        state_data,
    )

    mask_count = sum(1 for v in masking_config.values() if v.get("mode") != "none")
    log.info("KB tables saved for %s: %d tables, %d with masking — %s",
             account_id, len(tables), mask_count,
             ", ".join(tables[:8]) + ("…" if len(tables) > 8 else ""))
    return JSONResponse({"status": "ok", "count": len(tables), "tables": tables,
                         "masking_config": masking_config})


@router.get("/clients/{account_id}/kb-tables")
async def admin_get_kb_tables(request: Request, account_id: str):
    """Return the currently saved KB table selection for a client."""
    if not _is_auth(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    client = store.get_client(account_id)
    if not client:
        return JSONResponse({"status": "error", "message": "Client not found"}, status_code=404)
    state_data = json.loads(client.get("state_data") or "{}")
    tables = _parse_selected_schema_tables(state_data.get("kb_tables"))
    masking_config = state_data.get("masking_config") or {}
    source = "client" if tables else "none"
    if not tables and client.get("db_config_id"):
        raw = get_db_config(client["db_config_id"])
        if raw:
            tables = _parse_selected_schema_tables(raw["credentials"].get("selected_schema_tables"))
            source = "database" if tables else "none"
    return JSONResponse({"status": "ok", "tables": tables, "count": len(tables),
                         "source": source, "masking_config": masking_config})


@router.post("/clients/{account_id}/kb-tables/masking")
async def admin_save_masking_only(request: Request, account_id: str):
    """Save only the masking_config without requiring table re-selection."""
    if not _is_auth(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    client = store.get_client(account_id)
    if not client:
        return JSONResponse({"status": "error", "message": "Client not found"}, status_code=404)
    body = await request.json()
    raw_mc = body.get("masking_config") or {}
    masking_config = {
        k.upper(): v for k, v in raw_mc.items()
        if isinstance(v, dict) and v.get("mode") in ("none", "all", "selective", "auto")
    }
    current_state = client.get("state") or "configured"
    state_data = json.loads(client.get("state_data") or "{}")
    state_data["masking_config"] = masking_config
    store.update_client_state(account_id, current_state, state_data)
    # Immediately reflect the masking config in the egress log
    # so the admin panel shows the change without waiting for KB rebuild
    updated = store.update_egress_masking(account_id, masking_config)
    return JSONResponse({"status": "ok", "count": len(masking_config), "egress_updated": updated})


@router.get("/clients/{account_id}/setup/mask-preview")
async def admin_mask_preview(request: Request, account_id: str, fqn: str = ""):
    """
    Fetch up to 3 real rows for a table, apply the saved masking config,
    and return a side-by-side before/after so admins can verify masking works.

    Query param: fqn — fully-qualified table name (DB.SCHEMA.TABLE, upper-cased).
    Response: {status, columns, before:[row,...], after:[row,...], masked_fields:[...]}
    """
    if not _is_auth(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not fqn:
        raise HTTPException(status_code=400, detail="fqn query param required")

    client = store.get_client(account_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    db_cfg_id = client.get("db_config_id")
    if not db_cfg_id:
        return JSONResponse({"status": "no_db", "message": "No database configured"})

    raw_cfg = store.get_db_config(db_cfg_id)
    if not raw_cfg:
        return JSONResponse({"status": "no_db", "message": "Database config not found"})

    db_type = raw_cfg.get("db_type", "azure_sql")
    creds   = raw_cfg.get("credentials", {})
    parts   = fqn.upper().split(".")  # [DB, SCHEMA, TABLE] or [SCHEMA, TABLE]

    # Load masking config for this table
    state_data     = store.get_client_state(account_id)
    masking_config = state_data.get("masking_config") or {}
    # Match FQN key case-insensitively
    mc_entry: dict = {}
    for k, v in masking_config.items():
        if k.upper() == fqn.upper():
            mc_entry = v
            break

    mask_mode     = mc_entry.get("mode", "selective")
    masked_fields = set(f.upper() for f in (mc_entry.get("masked_fields") or []))

    if mask_mode == "none" or not masked_fields:
        return JSONResponse({
            "status": "no_masking",
            "message": "No masking configured for this table",
        })

    try:
        from core.schema import _az_connect, _sf_connect, _ora_connect
        from core.masking import mask_rows, detect_sensitive_columns

        def _fetch_sample() -> tuple[list[dict], list[dict]]:
            """Returns (col_defs, sample_rows) — 3 real rows."""
            col_defs: list[dict] = []
            rows:     list[dict] = []

            if db_type == "azure_sql":
                tbl    = parts[-1]
                schema = parts[-2] if len(parts) >= 2 else "dbo"
                db     = parts[0]  if len(parts) >= 3 else ""
                conn   = _az_connect(creds)
                try:
                    cur = conn.cursor()
                    if db:
                        cur.execute(f"USE [{db}]")
                    cur.execute(
                        "SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
                        "WHERE TABLE_SCHEMA=? AND TABLE_NAME=? ORDER BY ORDINAL_POSITION",
                        schema, tbl,
                    )
                    col_defs = [{"name": r[0].upper(), "type": r[1]} for r in cur.fetchall()]
                    if col_defs:
                        cols_sql = ", ".join(f"[{c['name']}]" for c in col_defs)
                        cur.execute(f"SELECT TOP 3 {cols_sql} FROM [{schema}].[{tbl}] WITH (NOLOCK)")
                        col_names = [c["name"] for c in col_defs]
                        rows = [dict(zip(col_names, row)) for row in cur.fetchall()]
                finally:
                    conn.close()

            elif db_type == "snowflake":
                tbl    = parts[-1]
                schema = parts[-2] if len(parts) >= 2 else "PUBLIC"
                db     = parts[0]  if len(parts) >= 3 else ""
                conn   = _sf_connect(creds)
                try:
                    cur = conn.cursor()
                    query = ("SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
                             "WHERE TABLE_NAME = %s AND TABLE_SCHEMA = %s")
                    params: list = [tbl, schema]
                    if db:
                        query += " AND TABLE_CATALOG = %s"
                        params.append(db)
                    cur.execute(query + " ORDER BY ORDINAL_POSITION", params)
                    col_defs = [{"name": str(r[0]).upper(), "type": r[1]} for r in cur.fetchall()]
                    if col_defs:
                        cols_sql = ", ".join(f'"{c["name"]}"' for c in col_defs)
                        db_prefix = f'"{db}".' if db else ""
                        cur.execute(f'SELECT {cols_sql} FROM {db_prefix}"{schema}"."{tbl}" LIMIT 3')
                        col_names = [c["name"] for c in col_defs]
                        rows = [dict(zip(col_names, row)) for row in cur.fetchall()]
                finally:
                    conn.close()

            else:  # oracle
                tbl   = parts[-1]
                owner = parts[-2] if len(parts) >= 2 else creds.get("username", "").upper()
                conn  = _ora_connect(creds)
                try:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT COLUMN_NAME, DATA_TYPE FROM ALL_TAB_COLUMNS "
                        "WHERE OWNER = :owner AND TABLE_NAME = :tbl ORDER BY COLUMN_ID",
                        {"owner": owner, "tbl": tbl},
                    )
                    col_defs = [{"name": str(r[0]).upper(), "type": r[1]} for r in cur.fetchall()]
                    if col_defs:
                        cols_sql = ", ".join(f'"{c["name"]}"' for c in col_defs)
                        cur.execute(
                            f'SELECT {cols_sql} FROM "{owner}"."{tbl}" FETCH FIRST 3 ROWS ONLY'
                        )
                        col_names = [c["name"] for c in col_defs]
                        rows = [dict(zip(col_names, row)) for row in cur.fetchall()]
                finally:
                    conn.close()

            return col_defs, rows

        loop = asyncio.get_running_loop()
        col_defs, real_rows = await asyncio.wait_for(
            loop.run_in_executor(None, _fetch_sample), timeout=20
        )

    except asyncio.TimeoutError:
        return JSONResponse({"status": "timeout", "message": "DB query timed out"})
    except Exception as exc:
        log.warning("mask-preview failed for %s fqn=%s: %s", account_id, fqn, exc)
        return JSONResponse({"status": "error", "message": str(exc)})

    if not real_rows:
        return JSONResponse({"status": "empty", "message": "Table has no rows to preview"})

    # Apply masking to a copy
    from core.masking import mask_rows as _mask_rows
    masked_rows = _mask_rows(
        [dict(r) for r in real_rows],
        masked_fields,
        col_defs,
        seed_key=account_id,
    )

    # Serialise — convert non-JSON types (Decimal, date, etc.) to string
    import decimal, datetime as _dt

    def _safe(v):
        if v is None:
            return None
        if isinstance(v, (int, float, str, bool)):
            return v
        if isinstance(v, decimal.Decimal):
            return float(v)
        if isinstance(v, (_dt.date, _dt.datetime)):
            return str(v)
        return str(v)

    def _safe_row(row: dict) -> dict:
        return {k: _safe(v) for k, v in row.items()}

    return JSONResponse({
        "status":        "ok",
        "fqn":           fqn,
        "columns":       [c["name"] for c in col_defs],
        "masked_fields": sorted(masked_fields),
        "before":        [_safe_row(r) for r in real_rows],
        "after":         [_safe_row(r) for r in masked_rows],
    })


@router.get("/clients/{account_id}/setup/column-sensitivity")
async def admin_column_sensitivity(request: Request, account_id: str, fqn: str = ""):
    """
    Return columns for a table plus auto-detected PII fields.

    Query param: fqn — fully-qualified table name (DB.SCHEMA.TABLE, upper-cased).
    First tries _schema.json written by discovery; falls back to a live DB query
    so fields are visible in the masking section BEFORE discovery has run.
    Response: {status, columns:[{name,type}], auto_masked:[colname,...], strategy_map:{}}
    """
    if not _is_auth(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not fqn:
        raise HTTPException(status_code=400, detail="fqn query param required")

    client = store.get_client(account_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    # ── 1. Try _schema.json first (fast path after discovery) ───────────────
    schema_dir  = Path("clients") / account_id / "schema"
    schema_path = schema_dir / "_schema.json"
    columns: list[dict] = []

    if schema_path.exists():
        try:
            from core.schema import _normalize_schema as _ns2
            schema_data = _ns2(json.loads(schema_path.read_text(encoding="utf-8")))
            fqn_upper = fqn.upper()
            for key, meta in schema_data.items():
                if key.upper() == fqn_upper:
                    columns = meta.get("columns") or []
                    break
        except Exception as exc:
            log.warning("column-sensitivity: failed to read _schema.json for %s: %s", account_id, exc)

    # ── 2. Live DB fallback when schema not yet discovered / table not found ─
    if not columns:
        db_cfg_id = client.get("db_config_id")
        if not db_cfg_id:
            return JSONResponse({"status": "no_db", "columns": [], "auto_masked": [], "strategy_map": {}})

        raw_cfg = store.get_db_config(db_cfg_id)
        if not raw_cfg:
            return JSONResponse({"status": "no_db", "columns": [], "auto_masked": [], "strategy_map": {}})

        db_type = raw_cfg.get("db_type", "azure_sql")
        creds   = raw_cfg.get("credentials", {})
        parts   = fqn.split(".")   # DB.SCHEMA.TABLE  (upper-cased)

        try:
            from core.schema import _az_connect, _sf_connect, _ora_connect

            def _fetch_columns_live() -> list[dict]:
                """Query INFORMATION_SCHEMA / ALL_TAB_COLUMNS for the table's columns."""
                result: list[dict] = []

                if db_type == "azure_sql":
                    # parts: [DB, SCHEMA, TABLE]  (may be 2 or 3 elements)
                    tbl    = parts[-1]
                    schema = parts[-2] if len(parts) >= 2 else "dbo"
                    db     = parts[0]  if len(parts) >= 3 else ""
                    conn   = _az_connect(creds)
                    try:
                        cur = conn.cursor()
                        if db:
                            cur.execute(f"USE [{db}]")
                        cur.execute(
                            "SELECT COLUMN_NAME, DATA_TYPE "
                            "FROM INFORMATION_SCHEMA.COLUMNS "
                            "WHERE TABLE_SCHEMA=? AND TABLE_NAME=? "
                            "ORDER BY ORDINAL_POSITION",
                            schema, tbl,
                        )
                        for row in cur.fetchall():
                            result.append({"name": row[0].upper(), "type": row[1]})
                    finally:
                        conn.close()

                elif db_type == "snowflake":
                    # parts: [DB, SCHEMA, TABLE]
                    tbl    = parts[-1]
                    schema = parts[-2] if len(parts) >= 2 else "PUBLIC"
                    db     = parts[0]  if len(parts) >= 3 else ""
                    conn   = _sf_connect(creds)
                    try:
                        cur = conn.cursor()
                        query = (
                            "SELECT COLUMN_NAME, DATA_TYPE "
                            "FROM INFORMATION_SCHEMA.COLUMNS "
                            "WHERE TABLE_NAME = %s AND TABLE_SCHEMA = %s"
                        )
                        params: list = [tbl, schema]
                        if db:
                            query += " AND TABLE_CATALOG = %s"
                            params.append(db)
                        query += " ORDER BY ORDINAL_POSITION"
                        cur.execute(query, params)
                        for row in cur.fetchall():
                            result.append({"name": str(row[0]).upper(), "type": row[1]})
                    finally:
                        conn.close()

                else:  # oracle
                    # parts: [OWNER/SCHEMA, TABLE]  (2 parts)
                    tbl   = parts[-1]
                    owner = parts[-2] if len(parts) >= 2 else creds.get("username", "").upper()
                    conn  = _ora_connect(creds)
                    try:
                        cur = conn.cursor()
                        cur.execute(
                            "SELECT COLUMN_NAME, DATA_TYPE "
                            "FROM ALL_TAB_COLUMNS "
                            "WHERE OWNER = :owner AND TABLE_NAME = :tbl "
                            "ORDER BY COLUMN_ID",
                            {"owner": owner, "tbl": tbl},
                        )
                        for row in cur.fetchall():
                            result.append({"name": str(row[0]).upper(), "type": row[1]})
                    finally:
                        conn.close()

                return result

            loop    = asyncio.get_running_loop()
            columns = await asyncio.wait_for(
                loop.run_in_executor(None, _fetch_columns_live), timeout=20
            )
        except asyncio.TimeoutError:
            log.warning("column-sensitivity live fallback timed out for %s fqn=%s", account_id, fqn)
            return JSONResponse({"status": "timeout", "columns": [], "auto_masked": [], "strategy_map": {}})
        except Exception as exc:
            log.warning("column-sensitivity live fallback failed for %s fqn=%s: %s", account_id, fqn, exc)
            return JSONResponse({"status": "error", "columns": [], "auto_masked": [], "strategy_map": {}})

    if not columns:
        return JSONResponse({"status": "not_found", "columns": [], "auto_masked": [], "strategy_map": {}})

    # ── 3. PII detection ─────────────────────────────────────────────────────
    auto_masked: list[str] = []
    strategy_map: dict[str, str] = {}
    try:
        from core.masking import detect_sensitive_columns
        detected     = detect_sensitive_columns(columns)
        auto_masked  = list(detected.keys())
        strategy_map = detected   # {col_name: strategy_name}
    except Exception:
        pass

    return JSONResponse({
        "status": "ok",
        "columns": columns,
        "auto_masked": auto_masked,
        "strategy_map": strategy_map,
    })


@router.post("/clients/{account_id}/setup/delete-kb")
async def admin_delete_kb_only(request: Request, account_id: str):
    """
    Delete only the KB files and vector store for this client.
    Keeps schema discovery files intact so KB can be rebuilt without
    re-running discovery.
    """
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)

    from urllib.parse import quote as _quote
    state_data = json.loads(client.get("state_data") or "{}")
    kb_dir     = state_data.get("kb_dir", "")

    # Resolve to absolute path so deletion works regardless of working directory
    kb_path = Path(kb_dir).resolve() if kb_dir else None

    # 1. Delete KB markdown and JSON files from disk
    deleted_files = 0
    if kb_path and kb_path.exists():
        for f in kb_path.iterdir():
            if f.is_file():
                try:
                    f.unlink()
                    deleted_files += 1
                except Exception as _e:
                    log.warning("Could not delete KB file %s: %s", f, _e)
    else:
        log.warning("delete-kb: kb_dir not found on disk (%s) — skipping file deletion", kb_dir)

    # 2. Clear Qdrant vectors for this account
    try:
        from core.vector_store import delete_kb_for_client
        delete_kb_for_client(account_id)
        log.info("Qdrant KB vectors cleared for %s", account_id)
    except Exception as _e:
        log.warning("Could not clear Qdrant KB for %s: %s", account_id, _e)

    # 3. Clear validated examples (SQL-pair cache)
    try:
        with _get_db() as _conn:
            _conn.execute("DELETE FROM validated_examples WHERE account_id=?", (account_id,))
    except Exception as _e:
        log.warning("Could not clear validated_examples for %s: %s", account_id, _e)

    # 4. Clear KB-extracted business terms so stale terms don't survive a rebuild
    try:
        with _get_db() as _conn:
            _conn.execute(
                "DELETE FROM business_term WHERE account_id=? AND source='kb_extracted'",
                (account_id,),
            )
        log.info("KB-extracted business terms cleared for %s", account_id)
    except Exception as _e:
        log.warning("Could not clear business_term for %s: %s", account_id, _e)

    # 5. Roll state back to SCHEMA_READY so the KB step shows as pending
    try:
        from core.pipeline_context import save_state
        next_state = dict(state_data)
        next_state.pop("kb_progress", None)
        save_state(account_id, "SCHEMA_READY", next_state)
    except Exception as _e:
        log.error("delete-kb: save_state failed for %s: %s", account_id, _e)
        # State update failed — redirect with warning rather than crashing
        return RedirectResponse(
            f"/admin/clients/{account_id}/setup?error={_quote('KB files deleted but state update failed. Refresh the page.')}",
            status_code=303,
        )

    log.info("KB deleted for %s: %d files removed", account_id, deleted_files)
    return RedirectResponse(
        f"/admin/clients/{account_id}/setup?saved={_quote('KB deleted — schema discovery is still intact. Re-run KB generation when ready.')}",
        status_code=303,
    )


@router.post("/clients/{account_id}/setup/stop-kb")
async def admin_stop_kb_build(request: Request, account_id: str):
    """
    Signal a running KB build to stop after the current table finishes.
    Rolls state back to SCHEMA_READY so the page shows the build as cancelled.
    """
    if not _is_auth(request):
        return JSONResponse({"ok": False, "error": "Unauthorized"}, status_code=401)

    ev = _kb_stop_events.get(account_id)
    if ev is not None:
        ev.set()
        log.info("Stop signal sent for KB build of %s", account_id)
        return JSONResponse({"ok": True, "message": "Stop signal sent — build will finish the current table then stop."})

    # No active build — maybe it already finished or the server restarted.
    # Defensively roll state to SCHEMA_READY if still in KB_BUILDING.
    client = store.get_client(account_id)
    if client:
        state_data = json.loads(client.get("state_data") or "{}")
        if client.get("state") == "KB_BUILDING":
            from core.pipeline_context import save_state
            fallback_state = dict(state_data)
            fallback_state.pop("kb_progress", None)
            save_state(account_id, "SCHEMA_READY", fallback_state)
            log.info("KB stop: no active task found for %s, rolled state to SCHEMA_READY", account_id)

    return JSONResponse({"ok": True, "message": "No active KB build found."})


@router.post("/clients/{account_id}/setup/discover")
async def admin_discover_schema(
    request: Request,
    account_id: str,
    bg: BackgroundTasks,
):
    """Trigger schema discovery from admin panel."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)

    db_cfg_id = client.get("db_config_id")
    if not db_cfg_id:
        from urllib.parse import quote
        return RedirectResponse(
            f"/admin/clients/{account_id}/setup?error={quote('No database assigned — go to Settings first')}",
            status_code=303)

    raw = get_db_config(db_cfg_id)
    if not raw:
        return RedirectResponse(
            f"/admin/clients/{account_id}/setup?error=DB+config+not+found",
            status_code=303)

    creds   = raw["credentials"]   # already decrypted by get_db_config()
    db_type = raw["db_type"]
    schema_dir = str(Path("clients") / account_id / "schema")

    # Read the admin's table selection — None means "all tables" (legacy fallback)
    state_data_existing = json.loads(client.get("state_data") or "{}")
    db_default_tables   = _parse_selected_schema_tables(creds.get("selected_schema_tables"))
    kb_tables_selected  = (
        _parse_selected_schema_tables(state_data_existing.get("kb_tables"))
        or db_default_tables
        or None
    )
    allowed_set         = set(kb_tables_selected) if kb_tables_selected else None

    async def _do_discover():
        try:
            from core.schema import discover_and_write
            from core.pipeline_context import save_state
            # ── Snapshot old schema for drift detection ───────────────────────
            _old_schema: dict = {}
            _old_schema_path = Path(schema_dir) / "_schema.json"
            if _old_schema_path.exists():
                try:
                    _old_schema = json.loads(_old_schema_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            # Clean the schema dir so tables removed from the source DB don't
            # linger on disk from a previous discovery run.
            sp = Path(schema_dir)
            if sp.exists():
                for f in sp.iterdir():
                    if f.is_file():
                        try:
                            f.unlink()
                        except Exception:
                            pass
            _masking_config = state_data_existing.get("masking_config") or None
            count = discover_and_write(creds, db_type, schema_dir,
                                       allowed_tables=allowed_set,
                                       masking_config=_masking_config,
                                       seed_key=account_id)
            next_state = dict(state_data_existing)
            next_state["schema_dir"] = schema_dir
            # ── Schema drift detection ─────────────────────────────────────────
            if _old_schema:
                try:
                    _new_schema_path = Path(schema_dir) / "_schema.json"
                    if _new_schema_path.exists():
                        from core.schema import _normalize_schema as _ns
                        _new_schema = _ns(json.loads(_new_schema_path.read_text(encoding="utf-8")))
                        _drift = _compute_schema_drift(_ns(_old_schema), _new_schema)
                        if _drift["has_changes"]:
                            next_state["schema_drift"] = _drift
                            log.info(
                                "Schema drift for %s: +%d/-%d tables, %d tables with column changes",
                                account_id, len(_drift["added_tables"]),
                                len(_drift["removed_tables"]), len(_drift["column_changes"]),
                            )
                        else:
                            # Clear any previous drift report
                            next_state.pop("schema_drift", None)
                except Exception as _dex:
                    log.warning("Schema drift detection failed for %s: %s", account_id, _dex)
            if kb_tables_selected:
                next_state.setdefault("kb_tables", kb_tables_selected)
            save_state(account_id, "SCHEMA_READY", next_state)
            if allowed_set:
                log.info("Admin schema discovery: %d/%d selected tables written for %s",
                         count, len(allowed_set), account_id)
            else:
                log.info("Admin schema discovery: %d tables for %s (all tables)", count, account_id)
            # ── Egress log (discovery) ───────────────────────────────────────
            # schema.json now carries fields_sent, row_count_sent, synthetic_used,
            # synthetic_override — written by _discover_* functions.
            try:
                import json as _json
                from pathlib import Path as _Path
                _schema_path = _Path(schema_dir) / "_schema.json"
                if _schema_path.exists():
                    from core.schema import _normalize_schema as _ns3
                    _schema = _ns3(_json.loads(_schema_path.read_text()))
                    for _tkey, _tmeta in _schema.items():
                        _tparts    = _tkey.split(".")
                        _tname     = _tparts[-1]
                        _tschema   = _tparts[-2] if len(_tparts) >= 2 else ""
                        _tdb       = _tparts[-3] if len(_tparts) >= 3 else ""
                        _col_count = len(_tmeta.get("columns") or [])
                        # Resolve sample_mode from what was actually written
                        _syn_used     = _tmeta.get("synthetic_used", False)
                        _row_ct       = _tmeta.get("row_count_sent", 0)
                        _mf           = _tmeta.get("masked_fields") or []
                        _mk_mode      = _tmeta.get("mask_mode", "auto")
                        if _syn_used:
                            _smode = "synthetic"
                        elif _mf:
                            _smode = "masked"
                        elif _row_ct > 0:
                            _smode = "real"
                        else:
                            _smode = "none"
                        store.log_kb_egress(
                            account_id=account_id,
                            operation="discovery",
                            db_type=db_type,
                            table_name=_tname,
                            sample_mode=_smode,
                            database_name=_tdb,
                            schema_name=_tschema,
                            column_count=_col_count,
                            distinct_col_count=0,
                            triggered_by="admin",
                            fields_sent=_tmeta.get("fields_sent") or [],
                            row_count_sent=_row_ct,
                            masked_fields=_mf,
                            mask_mode=_mk_mode,
                            mask_replacement_map=_tmeta.get("mask_replacement_map") or {},
                        )
            except Exception as _elog_exc:
                log.warning("KB egress log (discovery) write failed for %s: %s",
                            account_id, _elog_exc)
            # ── Entity graph auto-populate ─────────────────────────────────
            try:
                _schema_path = Path(schema_dir) / "_schema.json"
                _schema_scope = []
                if _schema_path.exists():
                    try:
                        _schema_scope = list(json.loads(_schema_path.read_text(encoding="utf-8")).keys())
                    except Exception as _scope_ex:
                        log.warning("Entity graph scope read failed for %s: %s", account_id, _scope_ex)
                _pruned = store.prune_entity_graph_to_tables(account_id, _schema_scope)
                if any(_pruned.values()):
                    log.info(
                        "Entity graph pruned for %s to current schema scope: %s",
                        account_id, _pruned,
                    )
                _ent, _rel = _auto_populate_entity_graph(account_id, schema_dir)
                log.info(
                    "Entity graph auto-populated for %s: %d entities, %d relationships (suggested)",
                    account_id, _ent, _rel,
                )
            except Exception as _gex:
                log.warning("Entity graph auto-populate failed for %s: %s", account_id, _gex)
            import threading as _threading
            _threading.Thread(target=_sync_all_log_exports_bg, daemon=True).start()
        except Exception as e:
            log.error("Admin schema discovery failed for %s: %s", account_id, e)

    bg.add_task(_do_discover)
    # Return JSON when called from JavaScript (AJAX), redirect otherwise
    if request.headers.get("accept", "").startswith("application/json") or \
       request.headers.get("x-requested-with") == "XMLHttpRequest":
        return JSONResponse({"status": "ok", "message": "Discovery started"})
    from urllib.parse import quote
    return RedirectResponse(
        f"/admin/clients/{account_id}/setup?saved={quote('Schema discovery started — refresh in 60 seconds')}",
        status_code=303)


@router.post("/clients/{account_id}/setup/build-kb")
async def admin_build_kb(
    request: Request,
    account_id: str,
    bg: BackgroundTasks,
    business_desc: str = Form(...),
):
    """Build Knowledge Base from admin panel with a business description."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    from urllib.parse import quote

    if not business_desc.strip():
        return RedirectResponse(
            f"/admin/clients/{account_id}/setup?error={quote('Business description is required')}",
            status_code=303)

    state_data = json.loads(client.get("state_data") or "{}")
    schema_dir = state_data.get("schema_dir", "")

    if not schema_dir or not Path(schema_dir).exists():
        return RedirectResponse(
            f"/admin/clients/{account_id}/setup?error={quote('Run schema discovery first')}",
            status_code=303)

    db_cfg_id = client.get("db_config_id")
    raw       = get_db_config(db_cfg_id) if db_cfg_id else None
    if not raw:
        return RedirectResponse(
            f"/admin/clients/{account_id}/setup?error={quote('No database assigned')}",
            status_code=303)

    creds   = raw["credentials"]   # already decrypted by get_db_config()
    db_type = raw["db_type"]
    kb_dir  = str(Path("clients") / account_id / "kb")
    # chroma_dir is kept in state_data for backward-compat reads but Qdrant
    # uses account_id as the tenant key — no filesystem path needed.
    chroma_dir = account_id

    # Register a fresh stop event for this build so the stop route can cancel it
    _stop_ev = asyncio.Event()
    _kb_stop_events[account_id] = _stop_ev

    async def _do_build():
        try:
            from core.pipeline_context import save_state
            from core.dispatcher import _run_example_validation, _run_log_harvest
            from core.knowledge import build_kb
            from core.llm import resolve_provider

            # Retain current per-table KB files so build_kb can compare its
            # full input fingerprint and skip unchanged tables. build_kb also
            # removes files for tables that have left the selected scope.
            kp = Path(kb_dir)
            kp.mkdir(parents=True, exist_ok=True)

            # Clear validated_examples rows for this account. The ChromaDB
            # validated_examples collection will be re-upserted by
            # _run_example_validation below. The kb_store collection is
            # dropped & recreated by core.knowledge._embed_kb_files.
            try:
                with _get_db() as _conn:
                    _conn.execute(
                        "DELETE FROM validated_examples WHERE account_id=?",
                        (account_id,))
            except Exception as _e:
                log.warning("Could not clear validated_examples for %s: %s",
                            account_id, _e)

            total_tables = len([
                f for f in Path(schema_dir).glob("*.md")
                if not f.name.startswith("_")
            ])
            initial_progress = {
                "status": "building",
                "phase": "starting",
                "step": "Starting Knowledge Base generation",
                "current": 0,
                "total": total_tables,
                "percent": 0,
                "current_table": "",
            }
            building_state = dict(state_data)
            building_state.update({
                "schema_dir": schema_dir,
                "kb_dir": kb_dir,
                "chroma_dir": chroma_dir,
                "business_desc": business_desc,
                "kb_progress": initial_progress,
            })
            save_state(account_id, "KB_BUILDING", building_state)
            await notify_kb_build_changed(
                account_id=account_id,
                status="building",
                progress=initial_progress,
            )

            async def _on_kb_progress(progress: dict):
                progress_payload = {
                    "status": "building",
                    **progress,
                }
                building_state["kb_progress"] = progress_payload
                save_state(account_id, "KB_BUILDING", building_state)
                await notify_kb_build_changed(
                    account_id=account_id,
                    status="building",
                    progress=progress_payload,
                )

            provider, model, api_key, az_kw = resolve_provider(client, purpose="kb")
            with llm_audit_scope(
                account_id=account_id,
                question=f"KB build for {client.get('client_name') or account_id}",
                enabled=bool(client.get("enable_llm_audit")),
                request_id=make_llm_audit_request_id(),
                component="kb_build",
            ):
                count = await build_kb(
                    schema_dir=schema_dir, kb_dir=kb_dir, chroma_dir=chroma_dir,
                    business_desc=business_desc, provider=provider, model=model,
                    api_key=api_key, extra_kwargs=az_kw,
                    progress_callback=_on_kb_progress,
                    stop_event=_stop_ev,
                    account_id=account_id,
                    db_type=db_type,
                )

            # If user requested a stop, roll back to SCHEMA_READY and exit
            if _stop_ev.is_set():
                _kb_stop_events.pop(account_id, None)
                stopped_state = dict(building_state)
                stopped_state.pop("kb_progress", None)
                save_state(account_id, "SCHEMA_READY", stopped_state)
                await notify_kb_build_changed(
                    account_id=account_id,
                    status="stopped",
                    progress={"status": "stopped", "step": "Build stopped by user"},
                )
                log.info("KB build cancelled by user for %s (%d tables done)", account_id, count)
                return

            # ── Egress log (KB build) ─────────────────────────────────────────
            # Read audit fields written by _discover_* into _schema.json.
            # This faithfully records what was actually sent — no re-evaluation.
            try:
                import json as _json
                from pathlib import Path as _Path
                _schema_path = _Path(schema_dir) / "_schema.json"
                if _schema_path.exists():
                    from core.schema import _normalize_schema as _ns4
                    _schema = _ns4(_json.loads(_schema_path.read_text()))
                    for _tkey, _tmeta in _schema.items():
                        _tparts    = _tkey.split(".")
                        _tname     = _tparts[-1]
                        _tschema   = _tparts[-2] if len(_tparts) >= 2 else ""
                        _tdb       = _tparts[-3] if len(_tparts) >= 3 else ""
                        _col_count = len(_tmeta.get("columns") or [])
                        _syn_used  = _tmeta.get("synthetic_used", False)
                        _mf        = _tmeta.get("masked_fields") or []
                        _mk_mode   = _tmeta.get("mask_mode", "auto")
                        from core.synthetic import should_use_synthetic
                        if _syn_used or should_use_synthetic(_tname):
                            _smode = "synthetic"
                        elif _mf:
                            _smode = "masked"
                        elif _tmeta.get("row_count_sent", 0) > 0:
                            _smode = "real"
                        else:
                            _smode = "none"
                        store.log_kb_egress(
                            account_id=account_id,
                            operation="kb_build",
                            db_type=db_type,
                            table_name=_tname,
                            sample_mode=_smode,
                            database_name=_tdb,
                            schema_name=_tschema,
                            column_count=_col_count,
                            distinct_col_count=0,
                            triggered_by="admin",
                            fields_sent=_tmeta.get("fields_sent") or [],
                            row_count_sent=_tmeta.get("row_count_sent", 0),
                            masked_fields=_mf,
                            mask_mode=_mk_mode,
                            mask_replacement_map=_tmeta.get("mask_replacement_map") or {},
                        )
            except Exception as _elog_exc:
                log.warning("KB egress log (kb_build) write failed for %s: %s",
                            account_id, _elog_exc)

            validating_progress = {
                "status": "building",
                "phase": "validating",
                "step": "Validating generated examples",
                "current": count,
                "total": count,
                "percent": 100,
                "current_table": "",
            }
            building_state["kb_progress"] = validating_progress
            save_state(account_id, "KB_BUILDING", building_state)
            await notify_kb_build_changed(
                account_id=account_id,
                status="building",
                progress=validating_progress,
            )

            # Auto-extract business terms from Stage 1 KB docs into the
            # semantic glossary. Safe to re-run — manual entries are never
            # overwritten, only 'kb_extracted' entries refresh.
            try:
                import store as _store
                term_count = _store.extract_terms_from_kb(account_id, kb_dir)
                log.info("Glossary auto-populated: %d terms for %s",
                         term_count, account_id)
            except Exception as _e:
                log.warning("Glossary auto-populate failed for %s: %s",
                            account_id, _e)

            # Build suggestion cache so portal has questions from day 1
            try:
                from core.suggestions import build_suggestion_cache as _bsc
                _bsc(kb_dir)
            except Exception as _e:
                log.debug("Suggestion cache (admin): %s", _e)

            # Step 2: Validate examples + Step 4: Harvest
            db_for_val = {"credentials": creds, "db_type": db_type, "id": db_cfg_id}
            await _run_example_validation(account_id, kb_dir, chroma_dir, db_for_val)
            harvesting_progress = {
                "status": "building",
                "phase": "harvesting",
                "step": "Harvesting query examples",
                "current": count,
                "total": count,
                "percent": 100,
                "current_table": "",
            }
            building_state["kb_progress"] = harvesting_progress
            save_state(account_id, "KB_BUILDING", building_state)
            await notify_kb_build_changed(
                account_id=account_id,
                status="building",
                progress=harvesting_progress,
            )
            await _run_log_harvest(account_id, chroma_dir)

            # ── Entity graph auto-populate + KB enrichment ────────────────────
            # The KB build just paid for LLM-written business descriptions and
            # synonyms — harvest them into the entity graph so the semantic
            # model is pre-built and laid out, ready for admin review.
            _graph_summary: dict = {}
            try:
                from core.graph_autopopulate import sync_graph_after_kb_build
                _graph_summary = sync_graph_after_kb_build(account_id, schema_dir, kb_dir)
                log.info(
                    "Entity graph synced after KB build for %s: +%d entities, "
                    "+%d joins, %d descriptions, %d properties enriched",
                    account_id,
                    _graph_summary.get("entities_added", 0),
                    _graph_summary.get("relationships_added", 0),
                    _graph_summary.get("descriptions_enriched", 0),
                    _graph_summary.get("properties_enriched", 0),
                )
            except Exception as _gex:
                log.warning("Entity graph sync after KB build failed for %s: %s",
                            account_id, _gex)

            complete_progress = {
                "status": "ready",
                "phase": "complete",
                "step": "Success - Knowledge Base stored and ready",
                "current": count,
                "total": count,
                "percent": 100,
                "current_table": "",
            }
            ready_state = dict(building_state)
            ready_state.update({
                "schema_dir": schema_dir,
                "kb_dir": kb_dir,
                "chroma_dir": chroma_dir,
                "business_desc": business_desc,
                "kb_progress": complete_progress,
            })
            save_state(account_id, "READY", ready_state, business_desc)
            await notify_kb_build_changed(
                account_id=account_id,
                status="ready",
                progress=complete_progress,
            )
            if _graph_summary:
                try:
                    from core.admin_notifications import notify_graph_ready
                    await notify_graph_ready(account_id=account_id, summary=_graph_summary)
                except Exception as _ngex:
                    log.debug("Graph-ready notification failed for %s: %s", account_id, _ngex)

            log.info("Admin KB build complete: %d tables for %s", count, account_id)
            _kb_stop_events.pop(account_id, None)
            asyncio.create_task(_run_default_evals_async(account_id, generate=True, execute=False))
            import threading as _threading
            _threading.Thread(target=_sync_all_log_exports_bg, daemon=True).start()
        except Exception as e:
            _kb_stop_events.pop(account_id, None)
            log.error("Admin KB build failed for %s: %s", account_id, e)
            failed_state = dict(state_data)
            failed_state["schema_dir"] = schema_dir
            failed_state["kb_progress"] = {
                "status": "failed",
                "phase": "failed",
                "step": str(e),
                "current": 0,
                "total": 0,
                "percent": 0,
                "current_table": "",
            }
            save_state(account_id, "SCHEMA_READY", failed_state)
            await notify_kb_build_changed(
                account_id=account_id,
                status="failed",
                progress=failed_state["kb_progress"],
            )

    bg.add_task(_do_build)
    return RedirectResponse(
        f"/admin/clients/{account_id}/setup?saved={quote('KB generation started - live progress will update on this page.')}",
        status_code=303)


# =============================================================================
# Learning Queue  (Day 3-4 — governed self-learning admin UI)
# =============================================================================

# =============================================================================
# Pending Platform Users  (admin approval — no registration link)
# =============================================================================

@router.get("/clients/{account_id}/pending-users")
async def admin_pending_users(request: Request, account_id: str):
    """List platform users waiting for admin approval."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)

    status_filter = request.query_params.get("status", "pending")
    saved = request.query_params.get("saved", "")

    pending = store.list_pending_users(account_id, status=status_filter)
    groups  = store.list_groups(account_id)
    counts  = {
        "pending":  store.get_pending_user_count(account_id),
        "approved": len(store.list_pending_users(account_id, status="approved")),
        "rejected": len(store.list_pending_users(account_id, status="rejected")),
    }
    return _resp(request, "client_pending_users.html", {
        "client":        client,
        "pending_users": pending,
        "groups":        groups,
        "counts":        counts,
        "status_filter": status_filter,
        "saved":         saved,
    })


@router.post("/clients/{account_id}/pending-users/{pending_id}/approve")
async def admin_pending_users_approve(
    request: Request, account_id: str, pending_id: int,
):
    """Approve a pending platform user, assign a group, send proactive notification."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)

    form         = await request.form()
    group_id_raw = str(form.get("group_id", "")).strip()
    reviewer_note = str(form.get("reviewer_note", "")).strip()[:500]
    group_id     = int(group_id_raw) if group_id_raw.isdigit() else None

    pending = store.get_pending_user(pending_id, account_id)
    if not pending:
        return RedirectResponse(
            f"/admin/clients/{account_id}/pending-users?saved={quote('Pending user not found.')}",
            status_code=303,
        )

    portal_user = store.approve_pending_user(
        pending_id=pending_id,
        account_id=account_id,
        group_id=group_id,
        reviewer_id="admin",
        reviewer_note=reviewer_note,
    )

    msg = f"Access approved for {pending.get('display_name') or pending.get('platform_user_id', '?')}."

    # Proactive notification to the user via their platform
    try:
        import json as _json
        platform_type = pending.get("platform_type", "teams")

        try:
            conv_ref = _json.loads(pending.get("conversation_ref") or "{}")
        except (ValueError, TypeError):
            conv_ref = {}

        if platform_type == "teams" and conv_ref.get("service_url"):
            teams_platforms = store.list_platforms("teams")
            active_teams = [p for p in teams_platforms if p.get("is_active")]
            if active_teams:
                from gateway.teams_adapter import TeamsAdapter
                from gateway.base import PlatformEvent
                adapter = TeamsAdapter(active_teams[0]["credentials"])
                synthetic_event = PlatformEvent(
                    account_id = account_id,
                    user_id    = pending["platform_user_id"],
                    channel_id = pending["conversation_ref"],
                    text       = "",
                    platform   = "teams",
                )
                group_name = ""
                if group_id:
                    g = store.get_group(group_id)
                    group_name = f" ({g['name']})" if g else ""
                await adapter.send_message(
                    synthetic_event,
                    f"✅ *Your access has been approved!*\n\n"
                    f"You now have access to QueryBot{group_name}.\n"
                    f"Go ahead and ask your first question — I'm ready.",
                )
                log.info("Proactive approval message sent to %s", pending["platform_user_id"])
    except Exception as exc:
        log.warning("Proactive approval message failed for pending_id=%d: %s", pending_id, exc)
        msg += " (notification could not be sent — user will see access on next message)"

    return RedirectResponse(
        f"/admin/clients/{account_id}/pending-users?saved={quote(msg)}",
        status_code=303,
    )


@router.post("/clients/{account_id}/pending-users/{pending_id}/reject")
async def admin_pending_users_reject(
    request: Request, account_id: str, pending_id: int,
):
    """Reject a pending platform user."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)

    form          = await request.form()
    reviewer_note = str(form.get("reviewer_note", "")).strip()[:500]

    pending = store.get_pending_user(pending_id, account_id)
    if not pending:
        return RedirectResponse(
            f"/admin/clients/{account_id}/pending-users?saved={quote('Pending user not found.')}",
            status_code=303,
        )

    store.reject_pending_user(
        pending_id=pending_id,
        account_id=account_id,
        reviewer_id="admin",
        reviewer_note=reviewer_note,
    )

    # Notify the user they were rejected so they can contact the admin
    try:
        import json as _json
        platform_type = pending.get("platform_type", "teams")

        try:
            conv_ref = _json.loads(pending.get("conversation_ref") or "{}")
        except (ValueError, TypeError):
            conv_ref = {}

        if platform_type == "teams" and conv_ref.get("service_url"):
            teams_platforms = store.list_platforms("teams")
            active_teams = [p for p in teams_platforms if p.get("is_active")]
            if active_teams:
                from gateway.teams_adapter import TeamsAdapter
                from gateway.base import PlatformEvent
                adapter = TeamsAdapter(active_teams[0]["credentials"])
                synthetic_event = PlatformEvent(
                    account_id = account_id,
                    user_id    = pending["platform_user_id"],
                    channel_id = pending["conversation_ref"],
                    text       = "",
                    platform   = "teams",
                )
                await adapter.send_message(
                    synthetic_event,
                    "Your access request was not approved. "
                    "Please contact your administrator for assistance.",
                )
    except Exception as exc:
        log.warning("Proactive rejection message failed for pending_id=%d: %s", pending_id, exc)

    name = pending.get("display_name") or pending.get("platform_user_id", "?")
    msg  = f"Access rejected for {name}."
    return RedirectResponse(
        f"/admin/clients/{account_id}/pending-users?saved={quote(msg)}",
        status_code=303,
    )


@router.get("/api/clients/{account_id}/pending-users/count")
async def admin_pending_users_count(request: Request, account_id: str):
    """Return JSON count of pending platform users — used for dashboard badges."""
    if not _is_auth(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    count = store.get_pending_user_count(account_id)
    return {"count": count}


@router.get("/clients/{account_id}/learning-queue")
async def admin_learning_queue(request: Request, account_id: str):
    """Display the admin Learning Queue for a workspace."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)

    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)

    from store.learning_store import list_candidates

    # Status filter from query string; default to pending_review
    status_filter = request.query_params.get("status", "pending_review")
    saved = request.query_params.get("saved", "")

    # Map filter → list_candidates status arg
    lc_status = None if status_filter == "all" else status_filter

    candidates = list_candidates(account_id, status=lc_status, limit=100)

    # Build stats for counter chips
    def _count(s):
        return len(list_candidates(account_id, status=s, limit=9999))

    stats = {
        "pending":      _count("pending_review"),
        "positive":     _count("approved"),
        "rejected":     _count("rejected"),
        "known_failure": _count("known_failure"),
        "revoked":      _count("revoked"),
        "total":        len(list_candidates(account_id, status=None, limit=9999)),
    }

    # Governed Qdrant count — how many approved Q&A pairs are live in the vector store
    governed_count = 0
    try:
        from core.governed_store import get_governed_count
        governed_count = get_governed_count(account_id)
    except Exception:
        pass

    # Parse evidence JSON for each candidate so the template can iterate it
    import json as _json
    for c in candidates:
        raw_ev = c.get("evidence")
        if isinstance(raw_ev, str):
            try:
                c["evidence"] = _json.loads(raw_ev)
            except Exception:
                c["evidence"] = {}
        elif not isinstance(raw_ev, dict):
            c["evidence"] = {}

    return _resp(request, "client_learning_queue.html", {
        "client":          client,
        "candidates":      candidates,
        "status_filter":   status_filter,
        "stats":           stats,
        "saved":           saved,
        "flag_enabled":    bool(client.get("enable_feedback_collection")),
        "governed_count":  governed_count,
    })


@router.post("/clients/{account_id}/learning-queue/toggle-flag")
async def admin_learning_queue_toggle_flag(request: Request, account_id: str):
    """Toggle enable_feedback_collection on/off for a client."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id) or {}
    current = int(client.get("enable_feedback_collection") or 0)
    store.update_client_meta(account_id, enable_feedback_collection=0 if current else 1)
    log.info("Admin toggled enable_feedback_collection → %d for %s", 0 if current else 1, account_id)
    return RedirectResponse(f"/admin/clients/{account_id}/learning-queue", status_code=303)


@router.post("/clients/{account_id}/learning-queue/backfill-governed")
async def admin_learning_queue_backfill_governed(request: Request, account_id: str):
    """Backfill all approved learning candidates into the querybot_governed Qdrant collection."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    try:
        from core.governed_store import backfill_approved_candidates
        count = backfill_approved_candidates(account_id)
        msg = f"Backfill complete — {count} approved candidate(s) synced to vector store."
    except Exception as exc:
        log.error("backfill_governed error for %s: %s", account_id, exc)
        msg = f"Backfill error: {exc}"
    return RedirectResponse(
        f"/admin/clients/{account_id}/learning-queue?saved={quote(msg)}",
        status_code=303,
    )


@router.post("/clients/{account_id}/learning-queue/{candidate_id}/review")
async def admin_learning_queue_review(
    request: Request,
    account_id: str,
    candidate_id: str,
):
    """Approve, reject, or mark known_failure on a learning candidate."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)

    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)

    form = await request.form()
    action = str(form.get("action", "")).strip()
    reviewer_note = str(form.get("reviewer_note", "")).strip()[:500]

    VALID_ACTIONS = {"approve", "reject", "known_failure"}
    if action not in VALID_ACTIONS:
        return RedirectResponse(
            f"/admin/clients/{account_id}/learning-queue?saved={quote('Unknown action — no change made.')}",
            status_code=303,
        )

    # Map UI action → DB status
    status_map = {
        "approve":       "approved",
        "reject":        "rejected",
        "known_failure": "known_failure",
    }
    new_status = status_map[action]

    try:
        from store.learning_store import update_candidate_status, get_candidate
        # Admin auth uses cookie-based HMAC (not SessionMiddleware), so no session user id
        reviewer_id = "admin"
        update_candidate_status(
            candidate_id=candidate_id,
            status=new_status,
            reviewer_id=reviewer_id,
            reviewer_note=reviewer_note,
        )
        label_map = {"approved": "approved", "rejected": "rejected", "known_failure": "marked as known failure"}
        msg = f"Candidate {candidate_id[:8]}... {label_map[new_status]}."

        # Governed collection sync is handled by update_candidate_status hooks
        # (_fire_governed_upsert on approved, _fire_governed_delete on revoked).
        # Rejecting or marking known_failure also needs the delete hook.
        if new_status in ("rejected", "known_failure"):
            try:
                from core.governed_store import delete_governed_example
                delete_governed_example(candidate_id=candidate_id, account_id=account_id)
            except Exception as gov_exc:
                log.warning("governed_store delete failed for %s: %s", candidate_id[:8], gov_exc)

    except Exception as exc:
        log.error("admin_learning_queue_review error: %s", exc)
        msg = f"Error: {exc}"

    return RedirectResponse(
        f"/admin/clients/{account_id}/learning-queue?saved={quote(msg)}",
        status_code=303,
    )


@router.post("/clients/{account_id}/learning-queue/{candidate_id}/correct-sql")
async def admin_learning_queue_correct_sql(
    request: Request,
    account_id: str,
    candidate_id: str,
):
    """Save admin-corrected SQL for a learning candidate (sets score to 85, source='admin_correction')."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)

    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)

    form = await request.form()
    corrected_sql = str(form.get("corrected_sql", "")).strip()
    reviewer_note = str(form.get("reviewer_note", "")).strip()[:500]

    if not corrected_sql:
        return RedirectResponse(
            f"/admin/clients/{account_id}/learning-queue?saved={quote('No SQL provided — nothing saved.')}",
            status_code=303,
        )

    try:
        from store.learning_store import (
            set_candidate_corrected_sql, update_candidate_status, get_candidate,
        )
        reviewer_id = "admin"

        # Read status BEFORE writing so we know whether to re-sync Qdrant below.
        pre = get_candidate(candidate_id) or {}
        was_approved = pre.get("status") == "approved"

        set_candidate_corrected_sql(
            candidate_id=candidate_id,
            corrected_sql=corrected_sql,
            reviewer_id=reviewer_id,
        )

        if reviewer_note:
            # Re-queue for explicit re-approval so the corrected SQL goes through review.
            update_candidate_status(
                candidate_id=candidate_id,
                status="pending_review",
                reviewer_id=reviewer_id,
                reviewer_note=reviewer_note,
            )
            # If it was live in Qdrant, remove the stale point so it isn't used until re-approved.
            if was_approved:
                try:
                    from core.governed_store import delete_governed_example
                    delete_governed_example(candidate_id=candidate_id, account_id=account_id)
                except Exception as ge:
                    log.warning("correct-sql: governed delete failed (non-fatal): %s", ge)
        elif was_approved:
            # No reviewer note: candidate stays approved — immediately push corrected SQL to Qdrant.
            try:
                from core.governed_store import upsert_governed_example
                cand = get_candidate(candidate_id) or {}
                upsert_governed_example(
                    candidate_id=candidate_id,
                    account_id=account_id,
                    question=cand.get("question_text", ""),
                    sql=corrected_sql,
                    source="admin_correction",
                    final_score=85,
                    schema_scope=cand.get("schema_scope", ""),
                )
                log.info("correct-sql: governed example re-synced for %s", candidate_id[:8])
            except Exception as ge:
                log.warning("correct-sql: governed upsert failed (non-fatal): %s", ge)

        msg = f"Corrected SQL saved for {candidate_id[:8]}... (score set to 85, source=admin_correction)."
    except Exception as exc:
        log.error("admin_learning_queue_correct_sql error: %s", exc)
        msg = f"Error saving corrected SQL: {exc}"

    return RedirectResponse(
        f"/admin/clients/{account_id}/learning-queue?saved={quote(msg)}",
        status_code=303,
    )


@router.get("/api/clients/{account_id}/learning-queue/count")
async def admin_learning_queue_count(request: Request, account_id: str):
    """Return JSON count of pending_review candidates — used for dashboard badges."""
    if not _is_auth(request):
        raise HTTPException(status_code=401, detail="Not authenticated")

    client = store.get_client(account_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    try:
        from store.learning_store import list_candidates
        pending = list_candidates(account_id, status="pending_review", limit=9999)
        return JSONResponse({"pending_review": len(pending), "account_id": account_id})
    except Exception as exc:
        log.error("admin_learning_queue_count error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
