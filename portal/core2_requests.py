"""Data guide → Suggest a change: a reader's suggestions, sent to their admin.

Registered on the portal router when portal/routes.py imports it. A reader suggests what a
metric or a field means, other names people use for it, the date a metric is counted by,
or describes a metric that is missing (core2.request_service); they see their suggestions
here with the admin's answer, and can take back one still waiting.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

import store
from portal import routes as _routes
from portal.routes import router

log = logging.getLogger("querybot.portal")


def _shown(request: dict[str, Any]) -> dict[str, Any]:
    """A request as its reader sees it: what they asked, and what became of it."""
    return {"id": request["id"], "kind": request["kind"], "target_kind": request.get("target_kind") or "",
            "target_key": request.get("target_key") or "", "name": request.get("target_name") or "",
            "status": request["status"], "reason": request.get("reason") or "",
            "changes": [{"label": c.get("label"), "value": c.get("shown") or c.get("value")}
                        for c in request.get("changes") or []],
            "note": request.get("note") or "", "sent": request.get("created_at") or "",
            "decided": request.get("decided_at") or ""}


def _refused(text: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": text}, status_code=status)


@router.get("/api/requests")
async def my_requests(request: Request):
    user = _routes._get_portal_user(request)
    if not user:
        return _refused("Sign in again.", 401)
    rows = store.list_core2_requests(user["account_id"], user_id=str(user["id"]), limit=100)
    return JSONResponse({"ok": True, "items": [_shown(r) for r in rows if r["status"] != "withdrawn"]})


@router.post("/api/requests")
async def send_request(request: Request):
    user = _routes._get_portal_user(request)
    if not user:
        return _refused("Sign in again.", 401)
    from core.i18n import get_active_language
    from core2.request_service import RequestRefused, file_request

    try:
        form = await request.json()
    except (ValueError, json.JSONDecodeError):
        form = None
    if not isinstance(form, dict):
        return _refused("The suggestion could not be read.")
    client = store.get_client(user["account_id"]) or {}
    if store.get_query_engine(user["account_id"]) != "core2":
        return _refused("Suggestions go to your admin from here once the new core answers your questions.")
    definition = None
    if form.get("kind") == "new_metric" and form.get("token"):
        # A metric the reader made in a chat: its definition is the one the server kept, found by its token.
        from core2.service import own_metric

        made = own_metric(str(form["token"]), user["account_id"], str(user["id"]))
        if made is None:
            return _refused("That metric is not in your chat any more: define it again, then ask.")
        measure = made["measure"]
        definition = measure.model_dump(mode="json")
        form = {"kind": "new_metric", "name": measure.business_name, "description": made["words"],
                "example": str(form.get("example") or ""), "source": "chat", "question_id": made["question_id"]}
    try:
        filed = file_request(user["account_id"], client, user, form, allowed=store.get_allowed_tables(user),
                             lang=get_active_language(), definition=definition)
    except RequestRefused as exc:
        return _refused(str(exc))
    try:
        from core.admin_notifications import notify_core2_request_changed

        await notify_core2_request_changed(account_id=user["account_id"], action="created", request_id=filed.get("id"))
    except Exception as exc:  # noqa: BLE001 - the request is filed; the admin's page shows it on its next load
        log.warning("Admin notification of a reader's request failed: %s", exc)
    return JSONResponse({"ok": True, "request": _shown(filed)})


@router.post("/api/requests/{request_id}/withdraw")
async def withdraw_request(request: Request, request_id: int):
    user = _routes._get_portal_user(request)
    if not user:
        return _refused("Sign in again.", 401)
    from core2.request_service import RequestRefused, withdraw

    found = store.get_core2_request(user["account_id"], request_id)
    if found is None:
        return _refused("That suggestion is not there any more.", 404)
    try:
        withdraw(user["account_id"], found, user_id=str(user["id"]))
    except RequestRefused as exc:
        return _refused(str(exc))
    return JSONResponse({"ok": True})


@router.post("/api/own-metrics/forget")
async def forget_own_metric(request: Request):
    """Don't keep it: a metric the reader made in a chat leaves that chat."""
    user = _routes._get_portal_user(request)
    if not user:
        return _refused("Sign in again.", 401)
    from core2.service import forget_own_metric as forget

    try:
        token = str((await request.json() or {}).get("token") or "")
    except (ValueError, json.JSONDecodeError, AttributeError):
        token = ""
    if not forget(token, user["account_id"], str(user["id"])):
        return _refused("That metric is not in your chat any more.", 404)
    return JSONResponse({"ok": True})
