"""One AI call of a workspace, opened from its AI egress log: what was sent, what came back, and whether both match
the fingerprints recorded with them.

A workspace under compliance keeps the exact prompt and response of every AI call, encrypted with the server's key
(core/llm_audit.py::record_llm_call). The log shows a sanitized preview; this page is where an admin, or the auditor
beside them, reads the whole prompt and sees the recorded SHA-256 recomputed from it.
"""

from __future__ import annotations

import json

from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse

import store
from admin.routes import _is_auth, _resp, router


@router.get("/clients/{account_id}/ai-calls/{call_id}", response_class=HTMLResponse)
async def ai_call(request: Request, account_id: str, call_id: int):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    from core.llm_audit import open_llm_call

    opened = open_llm_call(account_id, call_id)
    if opened is None:
        return HTMLResponse("No such AI call in this workspace.", status_code=404)
    try:
        egress = json.loads(opened["row"].get("egress_manifest") or "{}") or {}
    except ValueError:
        egress = {}
    return _resp(request, "client_ai_call.html", {"client": client, "call": opened["row"], "opened": opened,
                                                  "egress": egress, "active_tab": "audit"})
