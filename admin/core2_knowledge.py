"""Admin → Data → Knowledge base and Dates, for the new core.

Knowledge base: what each table and field means, in the admin's words. A table's "what
one row is"; per field its name, meaning, the words people also use, the names its codes
go by, and whether it is shown, hidden or sensitive. Dates: which date each table is
counted by, the names and words of its dates, a date the build did not find (checked
against the data first) and the month the fiscal year starts in.

Each change is an admin decision (core2.model.overrides): it holds from the next answer
and through the next Learn. Today's knowledge-base files and date roles stay under
Settings → Diagnostics for the earlier pipeline.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

import store
from admin.core2_relationships import _body, _refused, _warehouse, _workspace
from admin.routes import _is_auth, _resp, router

log = logging.getLogger("querybot.core2")


def _dated(checked: dict[str, Any]) -> dict[str, Any]:
    """A check as the page reads it (dates as text, the link check left out)."""
    return {k: (v.isoformat() if isinstance(v, dt.date) else v) for k, v in checked.items() if k != "checked"}


# ── Knowledge base ───────────────────────────────────────────────────────────

@router.get("/clients/{account_id}/knowledge", response_class=HTMLResponse)
async def knowledge_page(request: Request, account_id: str, table: str = ""):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client, model = _workspace(account_id)
    if client is None:
        return RedirectResponse("/admin/clients", status_code=303)
    from core2.model import knowledge

    data = None
    if model is not None and model.tables:
        listed = knowledge.tables(model)
        chosen = table if table in model.tables else next(
            (t["key"] for t in listed if t["kind"] in ("fact", "snapshot") and not t["hidden"]), listed[0]["key"])
        data = {"tables": listed, "table": knowledge.table_view(model, chosen)}
    return _resp(request, "client_knowledge.html", {"client": client, "data": data})


@router.post("/clients/{account_id}/knowledge/api/field")
async def knowledge_save_field(request: Request, account_id: str):
    """Save one field: its name, meaning, words, value names and whether it is shown."""
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    client, model = _workspace(account_id)
    if model is None:
        return _refused("QueryBot has to learn this database first.")
    from core2.bootstrap.service import load_model
    from core2.model import knowledge
    from core2.model.overrides import target

    data = await _body(request)
    key = str(data.get("column") or "")
    try:
        changes = knowledge.field_changes(model, key, data)
    except knowledge.KnowledgeError as exc:
        return _refused(str(exc))
    db_id = client.get("db_config_id")
    for field, value in changes.items():
        store.set_core2_override(account_id, db_id, target("column", key), field, value)
    after = load_model(account_id, db_id)
    view = knowledge.table_view(after, after.columns[key].table)
    return JSONResponse({"ok": True, "changed": sorted(changes),
                         "field": next(f for f in view["fields"] if f["key"] == key)})


@router.post("/clients/{account_id}/knowledge/api/table")
async def knowledge_save_table(request: Request, account_id: str):
    """Save what one row of a table is."""
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    client, model = _workspace(account_id)
    if model is None:
        return _refused("QueryBot has to learn this database first.")
    from core2.model import knowledge
    from core2.model.overrides import target

    data = await _body(request)
    key = str(data.get("table") or "")
    try:
        grain = knowledge.grain_change(model, key, data.get("grain"))
    except knowledge.KnowledgeError as exc:
        return _refused(str(exc))
    if grain is not None:
        store.set_core2_override(account_id, client.get("db_config_id"), target("table", key), "grain_text", grain)
    return JSONResponse({"ok": True, "grain": grain if grain is not None else model.tables[key].grain_text})


# ── Dates ────────────────────────────────────────────────────────────────────

@router.get("/clients/{account_id}/dates", response_class=HTMLResponse)
async def dates_page(request: Request, account_id: str, saved: str = ""):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client, model = _workspace(account_id)
    if client is None:
        return RedirectResponse("/admin/clients", status_code=303)
    from core2.model import knowledge

    data = knowledge.dates_view(model) if model is not None else None
    if data is not None:
        data["fiscal_words"] = {m: knowledge.fiscal_words(m, dt.date.today()) for m in range(1, 13)}
    return _resp(request, "client_dates.html", {"client": client, "data": data, "saved": saved})


@router.post("/clients/{account_id}/dates/api/default")
async def dates_default(request: Request, account_id: str):
    """The date a table is counted by unless a question names another."""
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    client, model = _workspace(account_id)
    if model is None:
        return _refused("QueryBot has to learn this database first.")
    from core2.model.overrides import target

    data = await _body(request)
    table, role = str(data.get("table") or ""), str(data.get("date") or "")
    if table not in model.tables or role not in model.date_roles or model.date_roles[role].table != table:
        return _refused("Choose one of this table's own dates.")
    if model.date_roles[role].kind == "audit":
        return _refused("This date is when the row was written: it cannot be the date a table is counted by.")
    db_id = client.get("db_config_id")
    # One decision per table: a default chosen another way before is replaced, not left to compete.
    for other in model.date_roles.values():
        if other.table == table:
            store.delete_core2_override(account_id, db_id, target("date_role", other.key), "is_default")
    store.set_core2_override(account_id, db_id, target("table", table), "default_date", role)
    return JSONResponse({"ok": True})


@router.post("/clients/{account_id}/dates/api/date")
async def dates_save(request: Request, account_id: str):
    """A date's name and the words people use for it."""
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    client, model = _workspace(account_id)
    if model is None:
        return _refused("QueryBot has to learn this database first.")
    from core2.model import knowledge
    from core2.model.overrides import target

    data = await _body(request)
    key = str(data.get("date") or "")
    try:
        changes = knowledge.date_changes(model, key, data)
    except knowledge.KnowledgeError as exc:
        return _refused(str(exc))
    for field, value in changes.items():
        store.set_core2_override(account_id, client.get("db_config_id"), target("date_role", key), field, value)
    return JSONResponse({"ok": True, "changed": sorted(changes)})


@router.post("/clients/{account_id}/dates/api/check")
async def dates_check(request: Request, account_id: str):
    """Whether a column can be a date of its table, checked against the data."""
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    client, model = _workspace(account_id)
    if model is None:
        return _refused("QueryBot has to learn this database first.")
    from core2.model import knowledge

    warehouse = _warehouse(account_id, client)
    if warehouse is None:
        return _refused("This workspace has no database to check against.")
    try:
        checked = knowledge.check_date(model, str((await _body(request)).get("column") or ""), warehouse)
    except knowledge.KnowledgeError as exc:
        return _refused(str(exc))
    return JSONResponse({**_dated(checked), "ok": bool(checked.get("ok"))})


@router.post("/clients/{account_id}/dates/api/add")
async def dates_add(request: Request, account_id: str):
    """Add a date the build did not find: checked again here, never taken from the page."""
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    client, model = _workspace(account_id)
    if model is None:
        return _refused("QueryBot has to learn this database first.")
    from core2.model import knowledge
    from core2.model.overrides import target

    data = await _body(request)
    warehouse = _warehouse(account_id, client)
    if warehouse is None:
        return _refused("This workspace has no database to check against.")
    column = str(data.get("column") or "")
    try:
        checked = knowledge.check_date(model, column, warehouse)
        role, join = knowledge.new_date(model, column, data.get("name"), checked)
    except knowledge.KnowledgeError as exc:
        return _refused(str(exc))
    db_id = client.get("db_config_id")
    if join is not None:
        store.set_core2_override(account_id, db_id, target("join", join.key), "define", join.model_dump(mode="json"))
    elif role.calendar_join and model.joins[role.calendar_join].trust == "rejected":
        store.set_core2_override(account_id, db_id, target("join", role.calendar_join), "trust", "admin")
    store.set_core2_override(account_id, db_id, target("date_role", role.key), "define", role.model_dump(mode="json"))
    # Kept apart from the definition too: a later Learn that finds this date keeps the admin's name and words.
    store.set_core2_override(account_id, db_id, target("date_role", role.key), "name", role.name)
    words = knowledge.clean_words(data.get("synonyms"))
    if words:
        store.set_core2_override(account_id, db_id, target("date_role", role.key), "synonyms", {"en": words})
    return JSONResponse({"ok": True, "date": role.key})


@router.post("/clients/{account_id}/dates/api/fiscal")
async def dates_fiscal(request: Request, account_id: str):
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    client, model = _workspace(account_id)
    if model is None:
        return _refused("QueryBot has to learn this database first.")
    from core2.model.overrides import target

    try:
        month = int((await _body(request)).get("month") or 0)
    except (TypeError, ValueError):
        month = 0
    if not 1 <= month <= 12:
        return _refused("Choose the month the fiscal year starts in.")
    store.set_core2_override(account_id, client.get("db_config_id"), target("settings", "workspace"),
                             "fiscal_year_start_month", month)
    return JSONResponse({"ok": True, "month": month})
