"""Admin → What Solviz learned → Setup check: the workbook the client's own team answers, and their answers back.

Download makes the workbook from the model as it is now (core2/setup_check.py); upload reads the answers and
checks each change they propose on the data, then shows them on a review page, where an admin picks which to
apply. Applied changes are admin decisions, the same as the Data pages write; corrections no field can hold go to
the Requests queue, and the team's test questions are kept for checking answers against.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

import store
from admin.core2_relationships import _warehouse, _workspace
from admin.routes import _download_header, _is_auth, _resp, router

log = logging.getLogger("querybot.core2")

MAX_BYTES = 10 * 1024 * 1024
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _by(request: Request) -> str:
    from admin.access import identity_of

    identity = identity_of(request)
    return (getattr(identity, "email", "") or getattr(identity, "name", "") or "admin") if identity else "admin"


def _back(account_id: str, error: str) -> RedirectResponse:
    return RedirectResponse(f"/admin/clients/{account_id}/learned?error={quote(error)}#setup-check", status_code=303)


def _connected(account_id: str, client: dict):
    try:
        return _warehouse(account_id, client)
    except Exception as exc:  # noqa: BLE001 - the workbook is made without figures; its checks are said undone
        log.warning("core2 setup check: no warehouse for %s: %s", account_id, exc)
        return None


@router.get("/clients/{account_id}/learned/setup-check.xlsx")
async def setup_check_download(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client, model = _workspace(account_id)
    if client is None:
        return RedirectResponse("/admin/clients", status_code=303)
    if model is None:
        return _back(account_id, "Learn the database first: the setup check is made from what Solviz learned.")
    from core2 import setup_check

    def make() -> bytes:
        suggestions = setup_check.suggestion_rows(model, account_id)
        return setup_check.export(model, workspace=client.get("client_name") or account_id, account_id=account_id,
                                  warehouse=_connected(account_id, client), suggestions=suggestions)

    data = await asyncio.get_running_loop().run_in_executor(None, make)
    name = f"Solviz setup check {(client.get('client_name') or account_id)[:40]} {dt.date.today():%Y-%m-%d}.xlsx"
    return Response(data, media_type=XLSX, headers={"Content-Disposition": _download_header(name)})


@router.post("/clients/{account_id}/learned/setup-check")
async def setup_check_upload(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client, model = _workspace(account_id)
    if client is None:
        return RedirectResponse("/admin/clients", status_code=303)
    if model is None:
        return _back(account_id, "Learn the database first: the setup check is made from what Solviz learned.")
    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        return _back(account_id, "Choose the filled-in setup check (.xlsx) to upload.")
    data = await upload.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        return _back(account_id, "That file is over 10 MB: it is not a setup check.")
    name = os.path.basename(str(getattr(upload, "filename", "") or "setup check.xlsx"))[:200]
    from core2 import setup_check

    def checked() -> list[dict]:
        answers = setup_check.read(data, account_id=account_id)
        return setup_check.review(model, answers, warehouse=_connected(account_id, client))

    try:
        items = await asyncio.get_running_loop().run_in_executor(None, checked)
    except setup_check.SetupCheckError as exc:
        return _back(account_id, str(exc))
    except Exception as exc:  # noqa: BLE001 - said on the page, never a page error
        log.warning("core2 setup check: %s could not be read for %s: %s", name, account_id, exc, exc_info=True)
        return _back(account_id, "The setup check could not be read. Download a new one and fill that in.")
    check_id = store.add_core2_setup_check(account_id, client.get("db_config_id"), file_name=name,
                                           uploaded_by=_by(request), items=items)
    return RedirectResponse(f"/admin/clients/{account_id}/learned/setup-check/{check_id}", status_code=303)


def _tabs(items: list[dict]) -> list[tuple[str, list[dict]]]:
    out: dict[str, list[dict]] = {}
    for item in items:
        out.setdefault(item["tab"], []).append(item)
    return list(out.items())


@router.get("/clients/{account_id}/learned/setup-check/{check_id}", response_class=HTMLResponse)
async def setup_check_review(request: Request, account_id: str, check_id: int):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    check = store.get_core2_setup_check(account_id, check_id)
    if check is None:
        return _back(account_id, "That setup check is not in this workspace.")
    items = check["items"]
    return _resp(request, "client_setup_check.html", {
        "client": client, "check": check, "tabs": _tabs(items),
        "counts": {"changes": sum(1 for i in items if i["ops"] and i["ok"]),
                   "problems": sum(1 for i in items if not i["ok"]),
                   "requests": sum(1 for i in items for o in i["ops"] if o["op"] == "request"),
                   "tests": sum(1 for i in items for o in i["ops"] if o["op"] == "test_question")},
        "saved": request.query_params.get("saved"), "error": request.query_params.get("error"),
    })


@router.post("/clients/{account_id}/learned/setup-check/{check_id}/apply")
async def setup_check_apply(request: Request, account_id: str, check_id: int):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client, model = _workspace(account_id)
    if client is None:
        return RedirectResponse("/admin/clients", status_code=303)
    page = f"/admin/clients/{account_id}/learned/setup-check/{check_id}"
    check = store.get_core2_setup_check(account_id, check_id)
    if check is None:
        return _back(account_id, "That setup check is not in this workspace.")
    if model is None:
        return RedirectResponse(f"{page}?error={quote('The model cannot be read: Learn again first.')}", status_code=303)
    form = await request.form()
    chosen = {int(n) for n in form.getlist("item") if str(n).isdigit()}
    items = [i for i in check["items"] if i.get("n") in chosen and i["ops"] and i["ok"]]
    if not items:
        return RedirectResponse(f"{page}?error={quote('Tick at least one change to apply.')}", status_code=303)
    if not store.close_core2_setup_check(account_id, check_id, status="applying", summary={}):
        return RedirectResponse(f"{page}?error={quote('This setup check has already been applied.')}", status_code=303)
    from core2 import setup_check

    by = _by(request)
    try:
        counts = await asyncio.get_running_loop().run_in_executor(
            None, lambda: setup_check.apply(account_id, client, model, items, by=by, check_id=check_id,
                                            file_name=check.get("file_name") or ""))
    except Exception as exc:  # noqa: BLE001 - what was written stays written; the page says it stopped
        log.error("core2 setup check %s for %s stopped: %s", check_id, account_id, exc, exc_info=True)
        store.close_core2_setup_check(account_id, check_id, status="failed", expect="applying",
                                      summary={"error": str(exc)[:300], "applied": sorted(chosen)})
        return RedirectResponse(f"{page}?error={quote('Applying stopped part way: see the decisions on the Data pages.')}",
                                status_code=303)
    store.close_core2_setup_check(account_id, check_id, status="applied", expect="applying",
                                  summary={**counts, "applied": sorted(chosen), "by": by})
    return RedirectResponse(f"{page}?saved=applied", status_code=303)


@router.post("/clients/{account_id}/learned/setup-check/{check_id}/discard")
async def setup_check_discard(request: Request, account_id: str, check_id: int):
    if not _is_auth(request):
        return JSONResponse({"ok": False}, status_code=401)
    store.close_core2_setup_check(account_id, check_id, status="discarded", summary={"by": _by(request)})
    return RedirectResponse(f"/admin/clients/{account_id}/learned#setup-check", status_code=303)
