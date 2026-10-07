"""Admin pages for the new core (docs/core-v2/DESIGN.md): what QueryBot learned.

Kept apart from admin/routes.py so the new core's admin surface can be read,
tested and removed in one piece. Authentication and rendering are the admin
console's own helpers.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any
from urllib.parse import quote

from fastapi import BackgroundTasks, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

import store
# The pages are registered on the admin console's own router (admin/routes.py
# imports this module at its end), so every /admin page sits on one router.
from admin.routes import _is_auth, _resp, router

log = logging.getLogger("querybot.core2")

_BUILD_STALE = dt.timedelta(minutes=45)


def _back(account_id: str, **params: str) -> RedirectResponse:
    query = "&".join(f"{k}={quote(v)}" for k, v in params.items())
    return RedirectResponse(f"/admin/clients/{account_id}/learned" + (f"?{query}" if query else ""), status_code=303)


def _running(build: dict[str, Any] | None) -> bool:
    if not build or build.get("status") != "running":
        return False
    try:
        started = dt.datetime.strptime(str(build["started_at"]), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
    except (TypeError, ValueError):
        return False
    return dt.datetime.now(dt.timezone.utc) - started < _BUILD_STALE


@router.get("/clients/{account_id}/learned", response_class=HTMLResponse)
async def learned_page(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    from core2.bootstrap.service import load_model
    from core2.model.view import learned_view

    db_id = client.get("db_config_id")
    model = None
    problem = ""
    try:
        model = load_model(account_id, db_id)
    except Exception as exc:  # noqa: BLE001 - a model this release cannot read is shown as such, not a 500
        log.warning("core2: stored model for %s could not be read: %s", account_id, exc, exc_info=True)
        problem = "The stored model was written by another release and cannot be read. Learn again to rebuild it."
    build = store.latest_core2_build(account_id, db_id)
    return _resp(request, "client_learned.html", {
        "client": client,
        "view": learned_view(model) if model else None,
        "build": build,
        "building": _running(build),
        "decisions": store.list_core2_overrides(account_id, db_id),
        "problem": problem,
        "has_database": bool(db_id),
        "saved": request.query_params.get("saved"),
        "error": request.query_params.get("error"),
        "engine": store.get_query_engine(account_id),
        "answers": store.list_core2_answers(account_id, 20),
    })


def _run_build(account_id: str) -> None:
    from core2.bootstrap.service import build_workspace

    try:
        build_workspace(account_id)
    except Exception:  # noqa: BLE001 - recorded in the build log by build_workspace, shown on the page
        pass


@router.post("/clients/{account_id}/learned/build")
async def learned_build(request: Request, account_id: str, bg: BackgroundTasks):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    if not client.get("db_config_id"):
        return _back(account_id, error="Connect a database first.")
    if _running(store.latest_core2_build(account_id, client["db_config_id"])):
        return _back(account_id, error="QueryBot is already learning this database.")
    bg.add_task(_run_build, account_id)
    return _back(account_id, saved="building")


def _value(raw: str) -> Any:
    text = raw.strip()
    if text in ("true", "false"):
        return text == "true"
    if text[:1] in "[{\"":
        try:
            return json.loads(text)
        except ValueError:
            return text
    return text


@router.post("/clients/{account_id}/learned/decide")
async def learned_decide(request: Request, account_id: str, target: str = Form(...), field: str = Form(...),
                         value: str = Form(""), note: str = Form("")):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    from core2.model.overrides import ALLOWED

    kind = target.partition(":")[0]
    if field not in ALLOWED.get(kind, set()):
        return _back(account_id, error=f"{field} cannot be changed here.")
    store.set_core2_override(account_id, client.get("db_config_id"), target, field, _value(value), note=note[:500])
    return _back(account_id, saved="decision")


@router.post("/clients/{account_id}/learned/undo")
async def learned_undo(request: Request, account_id: str, target: str = Form(...), field: str = Form(...)):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    store.delete_core2_override(account_id, client.get("db_config_id"), target, field)
    return _back(account_id, saved="undone")


@router.post("/clients/{account_id}/learned/engine")
async def learned_engine(request: Request, account_id: str, engine: str = Form(...)):
    """Who answers the workspace's portal questions: today's pipeline, both side by side, or the new core."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    if engine not in store.core2_store.ENGINES:
        return _back(account_id, error="Choose one of the three ways to answer.")
    if engine != "legacy":
        from core2.bootstrap.service import load_model

        if load_model(account_id, client.get("db_config_id")) is None:
            return _back(account_id, error="QueryBot has to learn this database before the new core can answer.")
    store.set_query_engine(account_id, engine)
    log.info("core2: %s now answers with %s", account_id, engine)
    return _back(account_id, saved="engine")
