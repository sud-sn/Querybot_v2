"""Admin → Data → Metrics, for the new core: the numbers people can ask for, and how each is counted.

The list shows every metric the new core answers with (learned from the data, brought
over from today's setup, changed or added by an admin, hidden), how it is counted, the
tables it adds up, its default date and how often it was asked. The editor writes one:
in the admin's words (the workspace's AI writes the formula), or as a formula over any
table's fields with the fields, metrics and functions suggested as they are typed. A
formula is read into the model's own form (core2.model.authoring), never pasted into a
query; it is checked against the data before it is saved, and saved as an admin
decision, so it answers from the next question and survives the next Learn.
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


def _shared(model) -> dict[str, Any]:
    from core2.model import authoring, links

    return {"formats": authoring.FORMAT_WORDS, "over_time": {k: v[2] for k, v in authoring.OVER_TIME.items()},
            "ops": links.OP_WORDS}


@router.get("/clients/{account_id}/measures", response_class=HTMLResponse)
async def measures_page(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client, model = _workspace(account_id)
    if client is None:
        return RedirectResponse("/admin/clients", status_code=303)
    from core2.model import authoring

    rows = []
    if model is not None:
        since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)).strftime("%Y-%m-%d")
        rows = authoring.listing(model, store.core2_measure_uses(account_id, since))
    return _resp(request, "client_measures.html", {
        "client": client, "learned": model is not None, "rows": rows,
        "saved": request.query_params.get("saved", ""),
    })


@router.get("/clients/{account_id}/measures/edit", response_class=HTMLResponse)
async def measure_edit_page(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client, model = _workspace(account_id)
    if client is None:
        return RedirectResponse("/admin/clients", status_code=303)
    if model is None:
        return RedirectResponse(f"/admin/clients/{account_id}/measures", status_code=303)
    from core2.model import authoring

    key = request.query_params.get("key", "")
    measure = model.measures.get(key) if key else None
    if key and measure is None:
        return RedirectResponse(f"/admin/clients/{account_id}/measures", status_code=303)
    dates = {t: [{"key": r.key, "name": r.name} for r in sorted(model.date_roles.values(), key=lambda r: r.name)
                 if r.table == t] for t in model.tables}
    return _resp(request, "client_measure_edit.html", {
        "client": client, "measure": measure,
        "editor": {"state": authoring.editor_state(model, measure), "fields": authoring.fields(model),
                   "metrics": [m for m in authoring.metrics(model) if m["key"] != key],
                   "tables": authoring.tables(model), "links": authoring.links(model),
                   "functions": [{"name": n, "template": t, "what": w} for n, t, w in authoring.FUNCTIONS],
                   "dates": dates, **_shared(model)},
    })


def _conditions(model, data: list[Any]) -> list:
    """A metric's conditions as the editor sends them: a field of any table, compared with a value it can hold."""
    from core2.model.links import LinkError, value_problem
    from core2.model.schema import ColumnFilter

    out = []
    for item in data or []:
        if not isinstance(item, dict) or not item.get("column"):
            continue
        try:
            f = ColumnFilter.model_validate({"column": item.get("column"), "op": item.get("op"),
                                             "values": [v for v in (item.get("values") or []) if str(v).strip()]})
        except Exception:
            raise LinkError("A condition is not complete: choose its field, how it compares and a value.") from None
        if f.column not in model.columns:
            raise LinkError("A condition reads a field that is not in the data any more.")
        if f.op not in ("is_null", "not_null") and not f.values:
            raise LinkError(f"Give the condition on {model.columns[f.column].business_name} a value.")
        if value_problem(model, f):
            raise LinkError(value_problem(model, f))
        out.append(f)
    return out


def _draft(model, data: dict[str, Any], *, key: str = "draft"):
    """The editor's metric, read and checked: a Measure, or the reason it is not one."""
    from core2.model import authoring
    # One rule for how an added metric adds up over time, the same as for one brought over.
    from core2.model.imports import _behaviour
    from core2.model.schema import Measure

    read = authoring.read(model, str(data.get("formula") or ""))
    conditions = _conditions(model, data.get("conditions"))
    choice = authoring.OVER_TIME.get(str(data.get("over_time") or ""))
    additivity, time_aggregation = (choice[0], choice[1]) if choice else _behaviour(model, read.table, read.expr)
    default_date = str(data.get("default_date") or "")
    if default_date and (default_date not in model.date_roles or model.date_roles[default_date].table != read.table):
        default_date = ""
    fmt = str(data.get("format") or "number")
    if fmt not in authoring.FORMAT_WORDS:
        fmt = "number"
    name = " ".join(str(data.get("name") or "").split())[:120]
    words = list(dict.fromkeys(w for w in (" ".join(str(x).split()) for x in data.get("synonyms") or []) if w))[:30]
    return Measure(key=key, slug=key, business_name=name or "This metric", description=str(
        data.get("description") or "").strip()[:500], synonyms={"en": words} if words else {}, table=read.table,
                   expr=read.expr, additivity=additivity, time_aggregation=time_aggregation, format=fmt,
                   default_date=default_date or None, filters=conditions, kind="metric", provenance="admin",
                   status="approved", tested=True, hidden=bool(data.get("draft")))


@router.post("/clients/{account_id}/measures/api/read")
async def measure_read(request: Request, account_id: str):
    """The formula as it is typed: what it reads as, its parts, the tables it adds up; or why it is not one yet."""
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    client, model = _workspace(account_id)
    if model is None:
        return _refused("QueryBot has to learn this database first.")
    from core2.model import authoring
    from core2.plan.catalog import definition

    try:
        read = authoring.read(model, str((await _body(request)).get("formula") or ""))
    except authoring.AuthoringError as exc:
        return JSONResponse({"ok": False, "error": str(exc)})
    from core2.model.schema import Measure

    probe = Measure(key="probe", slug="probe", table=read.table, expr=read.expr)
    return JSONResponse({"ok": True, "formula": authoring.to_text(model, read.expr),
                         "reads_as": definition(model, read.expr, True), "table": read.table,
                         "tables": authoring.counted_on(model, probe), "parts": authoring.parts(model, read.expr),
                         "by": authoring.broken_down_by(model, read.table),
                         "dates": [{"key": r.key, "name": r.name} for r in model.date_roles.values()
                                   if r.table == read.table]})


@router.post("/clients/{account_id}/measures/api/check")
async def measure_check(request: Request, account_id: str):
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    client, model = _workspace(account_id)
    if model is None:
        return _refused("QueryBot has to learn this database first.")
    from core2.model import authoring
    from core2.model.links import LinkError

    try:
        draft = _draft(model, await _body(request))
    except (authoring.AuthoringError, LinkError) as exc:
        return _refused(str(exc))
    warehouse = _warehouse(account_id, client)
    if warehouse is None:
        return _refused("This workspace has no database to check against.")
    return JSONResponse({"ok": True, **authoring.check(model, draft, warehouse, dt.date.today())})


@router.post("/clients/{account_id}/measures/api/describe")
async def measure_describe(request: Request, account_id: str):
    """The admin's words to a metric, written by the workspace's own AI and read like a typed one."""
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    client, model = _workspace(account_id)
    if model is None:
        return _refused("QueryBot has to learn this database first.")
    from core2.bootstrap.ai import metric_writer
    from core2.model import authoring

    description = " ".join(str((await _body(request)).get("description") or "").split())[:1000]
    if not description:
        return _refused("Say what the metric counts, in a sentence.")
    try:
        complete = metric_writer(account_id, client, description=description)
    except Exception as exc:  # noqa: BLE001 - a workspace with no AI configured is told so
        return _refused(f"The AI is not set up for this workspace: {str(exc)[:200]}")
    return JSONResponse({"ok": True, **authoring.describe(model, description, complete)})


@router.post("/clients/{account_id}/measures/api/save")
async def measure_save(request: Request, account_id: str):
    """Save a metric: one of the admin's own is defined whole; a learned one is changed field by field."""
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    client, model = _workspace(account_id)
    if model is None:
        return _refused("QueryBot has to learn this database first.")
    from core2 import ids
    from core2.model import authoring
    from core2.model.links import LinkError
    from core2.model.overrides import target

    data = await _body(request)
    key = str(data.get("key") or "")
    existing = model.measures.get(key) if key else None
    if key and existing is None:
        return _refused("That metric is not in the data any more.")
    name = " ".join(str(data.get("name") or "").split())
    if not name:
        return _refused("Give the metric the name people use for it.")
    if any(m.business_name.casefold() == name.casefold() and m.key != key and not m.hidden
           for m in model.measures.values()):
        return _refused(f"There is already a metric called {name}.")
    try:
        draft = _draft(model, data)
    except (authoring.AuthoringError, LinkError) as exc:
        return _refused(str(exc))
    db_id = client.get("db_config_id")
    if existing is None or key.startswith(authoring.ADMIN_PREFIX):
        taken = {m.slug for m in model.measures.values() if m.key != key}
        slug = existing.slug if existing is not None else ids.unique_slug(ids.slug(name, fallback="metric"), taken)
        key = key or authoring.ADMIN_PREFIX + slug
        measure = draft.model_copy(update={"key": key, "slug": slug})
        store.set_core2_override(account_id, db_id, target("measure", key), "define", measure.model_dump(mode="json"))
        if existing is not None and existing.hidden != measure.hidden:
            store.set_core2_override(account_id, db_id, target("measure", key), "hidden", measure.hidden)
        return JSONResponse({"ok": True, "key": key})
    changes = {"expr": draft.expr.model_dump(mode="json"), "table": draft.table,
               "filters": [f.model_dump(mode="json") for f in draft.filters], "business_name": draft.business_name,
               "synonyms": draft.synonyms, "description": draft.description, "format": draft.format,
               "additivity": draft.additivity, "time_aggregation": draft.time_aggregation,
               "default_date": draft.default_date, "hidden": draft.hidden}
    now = {"expr": existing.expr.model_dump(mode="json"), "table": existing.table,
           "filters": [f.model_dump(mode="json") for f in existing.filters], "business_name": existing.business_name,
           "synonyms": existing.synonyms, "description": existing.description, "format": existing.format,
           "additivity": existing.additivity, "time_aggregation": existing.time_aggregation,
           "default_date": existing.default_date, "hidden": existing.hidden}
    for field, value in changes.items():
        if value != now[field]:
            store.set_core2_override(account_id, db_id, target("measure", key), field, value)
    return JSONResponse({"ok": True, "key": key})


@router.post("/clients/{account_id}/measures/api/hide")
async def measure_hide(request: Request, account_id: str):
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    client, model = _workspace(account_id)
    if model is None:
        return _refused("QueryBot has to learn this database first.")
    from core2.model.overrides import target

    data = await _body(request)
    key = str(data.get("key") or "")
    if key not in model.measures:
        return _refused("That metric is not in the data any more.")
    store.set_core2_override(account_id, client.get("db_config_id"), target("measure", key), "hidden",
                             bool(data.get("hidden")))
    return JSONResponse({"ok": True})
