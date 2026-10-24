"""Admin → Data → Precomputed copies: metrics kept on this server so the questions they answer need no database trip.

An admin keeps a metric by some of its fields and by day or by month (core2/precompute.py); Solviz makes the copy at
once on the workspace's own connection and again on its schedule (core2/copies.py). Questions the copy holds
exactly are read from it, through the same governance; any other goes to the database. The page lists each copy:
when it was made, how many rows it holds, how many queries it answered, and why it could not be made, if it could
not. A copy keeps the workspace's figures on this server: workspace admins only (admin/access.py).
"""

from __future__ import annotations

from urllib.parse import quote

from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse

import store
from admin.routes import _is_auth, _resp, router
from core2 import copies
from core2 import precompute as PC


def _page(account_id: str, error: str = "", saved: str = "") -> RedirectResponse:
    query = f"?error={quote(error)}" if error else f"?saved={quote(saved)}" if saved else ""
    return RedirectResponse(f"/admin/clients/{account_id}/copies{query}", status_code=303)


def _model(account_id: str, client: dict):
    from core2.bootstrap.service import answering_model

    try:
        return answering_model(account_id, client)
    except Exception:  # noqa: BLE001 - the page says the workspace is not learned yet
        return None


def _who(request: Request) -> str:
    from admin.access import identity_of

    identity = identity_of(request)
    return "" if identity is None or identity.emergency else (identity.name or identity.email or "").strip()


@router.get("/clients/{account_id}/copies", response_class=HTMLResponse)
async def copies_page(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    db_id = int(client.get("db_config_id") or 0)
    model = _model(account_id, client)
    kept = store.list_copies(account_id, db_id) if db_id else []
    for row in kept:
        row["name"] = copies.copy_name(model, row)
        made = copies._when(row.get("made_at"))
        row["fresh"] = made is not None and PC.fresh(made, int(row.get("every_hours") or 6))
        row["fields_text"] = ", ".join(
            (model.attributes[s].business_name or s) if model is not None and s in model.attributes else s
            for s in row.get("fields") or [])
    offered = copies.choices(model) if model is not None else []
    return _resp(request, "client_copies.html", {
        "client": client, "kept": kept, "offered": offered,
        "learned": model is not None, "every": PC.EVERY_HOURS, "max_rows": PC.max_rows(),
        "saved": request.query_params.get("saved"), "error": request.query_params.get("error"),
    })


@router.post("/clients/{account_id}/copies")
async def copies_add(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    db_id = int(client.get("db_config_id") or 0)
    db_config = store.get_db_config(db_id) if db_id else None
    model = _model(account_id, client)
    if not db_config or model is None:
        return _page(account_id, error="This workspace is not learned yet: a copy is kept of a learned metric.")
    form = await request.form()
    metric = str(form.get("metric") or "").strip()
    fields = [str(f) for f in form.getlist("fields") if str(f).strip()]
    grain = str(form.get("grain") or "day")
    date = str(form.get("date") or "").strip()
    try:
        every = int(form.get("every") or 6)
    except ValueError:
        every = 6
    if every not in PC.EVERY_HOURS:
        return _page(account_id, error="Choose how often the copy is made again from the list.")
    if len(fields) > 6:
        return _page(account_id, error="Keep a copy by six fields at most: each one multiplies its rows.")
    from core2.warehouse.dialect import for_db_type
    import datetime as dt

    try:
        # Checked against the model now (the database is read when it is made, in the background).
        PC.define(model, metric=metric, fields=fields, date=date, grain=grain, today=dt.date.today(),
                  dialect=for_db_type(str(db_config.get("db_type") or "")))
    except PC.CopyError as exc:
        return _page(account_id, error=f"That copy cannot be kept: {exc}.")
    copy_id = store.add_copy(account_id, db_id, metric=metric, fields=fields, date_role=date, grain=grain,
                             every_hours=every, by=_who(request))
    copies.start(account_id, copy_id)
    return _page(account_id, saved="added")


@router.post("/clients/{account_id}/copies/{copy_id}/make")
async def copies_make(request: Request, account_id: str, copy_id: int):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    if store.get_copy(account_id, copy_id) is None:
        return _page(account_id, error="That copy is not kept any more.")
    copies.start(account_id, copy_id)
    return _page(account_id, saved="making")


@router.post("/clients/{account_id}/copies/{copy_id}/every")
async def copies_every(request: Request, account_id: str, copy_id: int):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    form = await request.form()
    try:
        every = int(form.get("every") or 0)
    except ValueError:
        every = 0
    if every not in PC.EVERY_HOURS or not store.set_copy_schedule(account_id, copy_id, every):
        return _page(account_id, error="Choose how often the copy is made again from the list.")
    return _page(account_id, saved="schedule")


@router.post("/clients/{account_id}/copies/{copy_id}/remove")
async def copies_remove(request: Request, account_id: str, copy_id: int):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    if not copies.remove(account_id, copy_id):
        return _page(account_id, error="That copy is not kept any more.")
    return _page(account_id, saved="removed")
