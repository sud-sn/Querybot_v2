"""Standing conditions in the portal: the chip row above the chat box, and the reader's own on the Profile page.

Registered on the portal router when portal/routes.py imports it. A reader sets a condition by saying it in the chat
("from now on use the last 2 months", "always leave out 'Wholesale'", core2/standing.py); here they see the ones in
force in a conversation, take one off it (the chip's ×), and list or remove for good the ones kept in their profile.
"""

from __future__ import annotations

import json
import re

from fastapi import Request
from fastapi.responses import JSONResponse

from portal import routes as _routes
from portal.routes import router


def _refused(text: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": text}, status_code=status)


def _key(user: dict, thread_id: str) -> str:
    """The conversation's key, as the chat socket names it (gateway/web_adapter.py, gateway/core2_bridge.py)."""
    from gateway.core2_bridge import _session_key

    thread = re.sub(r"[^a-zA-Z0-9_-]", "", str(thread_id or ""))[:80] or "default"
    return _session_key(str(user["account_id"]), user, thread)


@router.get("/api/chat/standing")
async def chat_standing(request: Request, thread_id: str = ""):
    """The standing conditions in force in a conversation (a reopened thread, a new one)."""
    user = _routes._get_portal_user(request)
    if not user:
        return _refused("Sign in again.", 401)
    from core2.service import standing_in

    return JSONResponse({"ok": True, "standing": standing_in(str(user["account_id"]), user, _key(user, thread_id))})


@router.post("/api/chat/standing/off")
async def chat_standing_off(request: Request):
    """The chip's ×: one condition taken off this conversation; one kept in the profile stays there."""
    user = _routes._get_portal_user(request)
    if not user:
        return _refused("Sign in again.", 401)
    try:
        body = await request.json() or {}
    except (ValueError, json.JSONDecodeError):
        body = {}
    standing_id = str(body.get("id") or "")[:40]
    if not standing_id:
        return _refused("Which condition?")
    from core2.service import standing_taken_off

    kept = standing_taken_off(str(user["account_id"]), user, _key(user, str(body.get("thread_id") or "")),
                              standing_id)
    return JSONResponse({"ok": True, "standing": kept})


@router.get("/api/profile/standing")
async def profile_standing(request: Request):
    """The reader's own standing conditions in this workspace, as their Profile page lists them."""
    user = _routes._get_portal_user(request)
    if not user:
        return _refused("Sign in again.", 401)
    from store import profile_store as P

    rows = P.standing_conditions(int(user["id"]), str(user["account_id"]))
    return JSONResponse({"ok": True, "standing": [{"id": r.get("id"), "words": r.get("words") or "",
                                                   "words_fr": r.get("words_fr") or r.get("words") or ""}
                                                  for r in rows]})


@router.post("/api/profile/standing/remove")
async def profile_standing_remove(request: Request):
    """One of the reader's own standing conditions removed for good: no new answer of theirs uses it."""
    user = _routes._get_portal_user(request)
    if not user:
        return _refused("Sign in again.", 401)
    try:
        body = await request.json() or {}
    except (ValueError, json.JSONDecodeError):
        body = {}
    from store import profile_store as P

    if not P.drop_standing(int(user["id"]), str(user["account_id"]), str(body.get("id") or "")[:40]):
        return _refused("That condition is not in your profile any more.", 404)
    return JSONResponse({"ok": True})
