"""What readers can ask: every question a workspace's data answers, and why the others cannot be.

The question map (core2/question_map.py) is built at Learn's end and after an admin's change; this page shows it
by subject: how many questions are answered, how many found no rows, and each one that cannot be answered with
the reason and where an admin fixes it (a link on the Relationships page, a route to choose). Check now rebuilds
and checks it on the data.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from admin.core2_relationships import _workspace
from admin.routes import _is_auth, _resp, router

log = logging.getLogger("querybot.admin")

# What each kind of gap means to an admin, and where it is fixed.
GAPS = {
    "not_linked": ("Not linked", "No link reaches it, and no table of rows names both.", "relationships"),
    "needs_pick": ("Two routes", "It can be reached two ways; readers are asked which.", "relationships"),
    "unconfirmed": ("Link to confirm", "It is reached through a link an admin has not confirmed yet.",
                    "relationships"),
    "cannot_split": ("Cannot split", "Each one sits under several of the grouping: it would be counted more than once.",
                     ""),
    "refused": ("Not answered", "The engine stopped on it; the reason is below.", ""),
}


@router.get("/clients/{account_id}/askable", response_class=HTMLResponse)
async def askable_page(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client, model = _workspace(account_id)
    if client is None:
        return RedirectResponse("/admin/clients", status_code=303)
    from core2 import question_map as Q

    qmap = Q.load(account_id, client.get("db_config_id")) if model is not None else None
    fresh = qmap is not None and qmap.stamp == Q.stamp_of(account_id, client.get("db_config_id"))
    subjects = Q.coverage(qmap) if qmap is not None else []
    counts = qmap.counts() if qmap is not None else {}
    rows = sum(1 for e in qmap.entries if e.route == "rows") if qmap is not None else 0
    return _resp(request, "client_askable.html", {
        "client": client, "learned": model is not None, "qmap": qmap, "fresh": fresh, "subjects": subjects,
        "answerable": counts.get(Q.ANSWERABLE, 0), "gaps": sum(v for k, v in counts.items() if k != Q.ANSWERABLE),
        "on_rows": rows, "gap_words": GAPS,
    })


@router.post("/clients/{account_id}/askable/api/check")
async def askable_check(request: Request, account_id: str):
    """Rebuild the map on the model as it is now and check it on the data."""
    if not _is_auth(request):
        return JSONResponse({"ok": False, "error": "Sign in again."}, status_code=401)
    client, model = _workspace(account_id)
    if client is None or model is None:
        return JSONResponse({"ok": False, "error": "Solviz has not learned this workspace's data yet."})
    from core2.question_map import ANSWERABLE, build_for

    try:
        qmap = await asyncio.to_thread(build_for, account_id, client)
    except Exception as exc:  # noqa: BLE001 - said to the admin, never raised to the page
        log.warning("Checking the questions of %s failed: %s", account_id, exc, exc_info=True)
        return JSONResponse({"ok": False, "error": f"The check stopped: {exc}"})
    if qmap is None:
        return JSONResponse({"ok": False, "error": "Solviz has not learned this workspace's data yet."})
    counts = qmap.counts()
    return JSONResponse({"ok": True, "answerable": counts.get(ANSWERABLE, 0),
                         "gaps": sum(v for k, v in counts.items() if k != ANSWERABLE)})
