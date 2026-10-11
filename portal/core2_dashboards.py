"""A dashboard the reader described in the chat (core2/dashboards.py): renamed, tiles taken out or added, saved.

The draft is kept on the server behind a token only its reader can use; the page sends the token and what to
change. A tile the draft offered is added by answering its question as a new conversation would (one AI call, its
own query). Saving writes the dashboard and every tile at once, as one version ("Built from your request"): a new
dashboard, or one of the reader's own. Its tiles are drawn again from their plans, with no AI.
"""

from __future__ import annotations

import json
import logging

from fastapi import Request
from fastapi.responses import JSONResponse

import store
from core2 import dashboards
from portal import routes as _routes
from portal.routes import router

log = logging.getLogger("querybot.portal")


def _refused(text: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": text}, status_code=status)


async def _form(request: Request) -> dict:
    try:
        data = await request.json()
    except (ValueError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _draft(request: Request, token: str):
    user = _routes._get_portal_user(request)
    if not user:
        return None, None, _refused("Sign in again.", 401)
    draft = store.get_dashboard_draft(token, user["account_id"], int(user["id"]))
    if draft is None:
        return user, None, _refused("This dashboard draft is no longer kept: ask for it again in the chat.", 404)
    return user, draft, None


@router.post("/api/core2/dashboard-draft/{token}/rename")
async def rename_draft(request: Request, token: str):
    user, draft, refused = _draft(request, token)
    if refused:
        return refused
    name = " ".join(str((await _form(request)).get("name") or "").split())[:80]
    if not name:
        return _refused("Give the dashboard a name.")
    draft["name"] = name
    store.update_dashboard_draft(token, user["account_id"], int(user["id"]), draft)
    return JSONResponse({"ok": True, "name": name})


@router.post("/api/core2/dashboard-draft/{token}/remove")
async def remove_tile(request: Request, token: str):
    """Take one tile out of the draft (by its place in the draft)."""
    user, draft, refused = _draft(request, token)
    if refused:
        return refused
    try:
        index = int((await _form(request)).get("tile"))
    except (TypeError, ValueError):
        return _refused("Which tile?")
    tiles = draft.get("tiles") or []
    if not 0 <= index < len(tiles):
        return _refused("That tile is not in the draft.")
    removed = tiles.pop(index)
    if not tiles:
        return _refused("A dashboard needs at least one tile.")
    dashboards.relayout(tiles)                       # the rest close up
    draft["tiles"] = tiles
    store.update_dashboard_draft(token, user["account_id"], int(user["id"]), draft)
    return JSONResponse({"ok": True, "removed": removed.get("title"), "left": len(tiles),
                         "layouts": [t.get("layout") or {} for t in tiles]})


@router.post("/api/core2/dashboard-draft/{token}/add")
async def add_tile(request: Request, token: str):
    """Add a tile the draft offered: its question answered as a new conversation, then laid out with the rest."""
    user, draft, refused = _draft(request, token)
    if refused:
        return refused
    if store.get_query_engine(user["account_id"]) != "core2":
        return _refused("Tiles are built by the new core, which no longer answers here.")
    try:
        index = int((await _form(request)).get("extra"))
    except (TypeError, ValueError):
        return _refused("Which tile?")
    extras = draft.get("extras") or []
    if not 0 <= index < len(extras):
        return _refused("That tile is not offered any more.")
    tiles = draft.get("tiles") or []
    if len(tiles) >= 2 * dashboards.MOST_TILES:
        return _refused(f"A dashboard holds at most {2 * dashboards.MOST_TILES} tiles from one request.")
    offered = extras[index]
    tile = dashboards.Tile(title=str(offered.get("title") or "")[:80], kind=str(offered.get("kind") or "breakdown"),
                           question=str(offered.get("question") or ""))
    if tile.kind not in dashboards.HEIGHT:
        tile.kind = "breakdown"
    from gateway.core2_bridge import tile_answer

    status, payload = await tile_answer(user["account_id"], tile.question, user, token)
    if payload is None:
        return _refused("That tile could not be built just now: ask again in a moment.", 503)
    dashboards.answered(tile, lambda _: payload)
    if tile.error:
        return JSONResponse({"ok": False, "error": tile.error, "definable": tile.definable}, status_code=422)
    extras.pop(index)
    tiles.append(dashboards.kept(tile))
    dashboards.relayout(tiles)
    draft["tiles"], draft["extras"] = tiles, extras
    store.update_dashboard_draft(token, user["account_id"], int(user["id"]), draft)
    shown = dashboards.preview_tile(tile, len(tiles) - 1)
    shown["layout"] = tiles[-1]["layout"]
    return JSONResponse({"ok": True, "tile": shown, "layouts": [t.get("layout") or {} for t in tiles],
                         "extras": extras})


@router.post("/api/core2/dashboard-draft/{token}/save")
async def save_draft(request: Request, token: str):
    """Save the draft as a new dashboard, or onto one of the reader's own, as one version."""
    user, draft, refused = _draft(request, token)
    if refused:
        return refused
    form = await _form(request)
    target = form.get("dashboard_id")
    try:
        dashboard_id = int(target) if target not in (None, "", "new") else None
    except (TypeError, ValueError):
        return _refused("Choose a dashboard.")
    client = store.get_client(user["account_id"]) or {}
    name = " ".join(str(form.get("name") or draft.get("name") or "").split())[:80] or "Dashboard"
    saved = store.save_built_dashboard(user["account_id"], int(user["id"]), name=name, tiles=draft.get("tiles") or [],
                                       db_config_id=int(client.get("db_config_id") or 0) or None,
                                       dashboard_id=dashboard_id, thread_id=str(form.get("thread_id") or "default"))
    if saved is None:
        return _refused("That dashboard is not yours to change.", 403)
    store.delete_dashboard_draft(token, user["account_id"], int(user["id"]))
    log.info("A described dashboard was saved for %s: %s tiles on dashboard %s", user["account_id"],
             len(draft.get("tiles") or []), saved["id"])
    return JSONResponse({"ok": True, "dashboard": {"id": saved["id"], "name": saved["name"],
                                                    "url": f"/portal/dashboard?dashboard_id={saved['id']}"},
                         "tiles": len(draft.get("tiles") or [])})
