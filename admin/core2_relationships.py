"""Admin → Data → Relationships, for the new core: how the tables connect, and the rows each leaves out.

The page draws the learned model's links around one table at a time, with a list of
them all. An admin picks a link to change the columns it matches on, the rows of the
table it reaches that it keeps (its conditions), its name when the same tables join
more than one way, or to turn it off; or adds a link the data did not show. A table
holds its name, its kind, its default date and the rows it always leaves out. Every
change can be checked against the data first, and is saved as an admin decision
(core2.model.overrides), so it holds from the next answer and through the next Learn.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

import store
from admin.routes import _is_auth, _resp, router

log = logging.getLogger("querybot.core2")

_TABLE_KINDS = ("fact", "snapshot", "dimension", "bridge", "other")


def _workspace(account_id: str):
    client = store.get_client(account_id)
    if not client:
        return None, None
    from core2.bootstrap.service import load_model

    try:
        model = load_model(account_id, client.get("db_config_id"))
    except Exception as exc:  # noqa: BLE001 - a model this release cannot read is said so on the page
        log.warning("core2: the model of %s could not be read: %s", account_id, exc)
        model = None
    return client, model


def _warehouse(account_id: str, client: dict[str, Any]):
    """The workspace's warehouse for an admin's check: read-only, every table."""
    from core.schema import load_known_tables
    from core2.warehouse.governed import GovernedWarehouse

    db_config = store.get_db_config(int(client.get("db_config_id") or 0)) if client.get("db_config_id") else None
    if not db_config:
        return None
    state = store.get_client_state(account_id) or {}
    return GovernedWarehouse(account_id, None, db_config, known_tables=load_known_tables(state.get("schema_dir", "")),
                             allowed_tables=None)


def _refused(text: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": text}, status_code=status)


async def _body(request: Request) -> dict[str, Any]:
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


@router.get("/clients/{account_id}/relationships", response_class=HTMLResponse)
async def relationships_page(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client, model = _workspace(account_id)
    if client is None:
        return RedirectResponse("/admin/clients", status_code=303)
    from core2.model import links

    return _resp(request, "client_relationships.html", {
        "client": client, "data": links.view(model) if model else None,
        "kinds": {k: links.KIND_WORDS[k] for k in _TABLE_KINDS},
        "cardinalities": links.CARDINALITY_WORDS, "ops": links.OP_WORDS,
    })


def _spec(model, data: dict[str, Any]):
    from core2.model import links

    return links.spec_from(model, data)


@router.post("/clients/{account_id}/relationships/check-link")
async def relationships_check_link(request: Request, account_id: str):
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    client, model = _workspace(account_id)
    if model is None:
        return _refused("QueryBot has to learn this database first.")
    from core2.model import links

    try:
        spec = _spec(model, await _body(request))
    except links.LinkError as exc:
        return _refused(str(exc))
    warehouse = _warehouse(account_id, client)
    if warehouse is None:
        return _refused("This workspace has no database to check against.")
    return JSONResponse({"ok": True, **links.check_link(model, spec, warehouse)})


@router.post("/clients/{account_id}/relationships/save-link")
async def relationships_save_link(request: Request, account_id: str):
    """Save a link: a changed condition, name or confirmation on the same link, or a link of the admin's own
    (a learned one it replaces is turned off, so the next Learn keeps the decision)."""
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    client, model = _workspace(account_id)
    if model is None:
        return _refused("QueryBot has to learn this database first.")
    from core2.model import links
    from core2.model.overrides import target

    data = await _body(request)
    try:
        spec = _spec(model, data)
    except links.LinkError as exc:
        return _refused(str(exc))
    db_id = client.get("db_config_id")
    existing = model.joins.get(str(data.get("key") or ""))
    conditions = [c.model_dump(mode="json") for c in spec.conditions]
    if existing is not None and links.same_link(existing, spec):
        key = existing.key
        if [c.model_dump(mode="json") for c in existing.conditions] != conditions:
            store.set_core2_override(account_id, db_id, target("join", key), "conditions", conditions)
        if existing.trust not in ("admin",):
            store.set_core2_override(account_id, db_id, target("join", key), "trust", "admin")
        return JSONResponse({"ok": True, "key": key})
    key = links.key_for(model, spec)
    if key in model.joins and existing is None:
        return _refused("These tables are already linked on these columns: pick that link to change it.")
    warehouse = _warehouse(account_id, client)
    checked = links.check_link(model, spec, warehouse) if warehouse is not None else {}
    store.set_core2_override(account_id, db_id, target("join", key), "define",
                             links.as_join(model, spec, checked).model_dump(mode="json"))
    if existing is not None:
        _turn_off(account_id, db_id, existing)
    return JSONResponse({"ok": True, "key": key})


def _turn_off(account_id: str, db_id: Any, join) -> None:
    """A link the admin added is taken out; one the data showed is turned off (and stays off)."""
    from core2.model.overrides import target

    object_key = target("join", join.key)
    own = [o for o in store.list_core2_overrides(account_id, db_id)
           if o["object_key"] == object_key and o["field"] == "define"]
    if own:
        for o in store.list_core2_overrides(account_id, db_id):
            if o["object_key"] == object_key:
                store.delete_core2_override(account_id, db_id, object_key, o["field"])
        return
    store.set_core2_override(account_id, db_id, object_key, "trust", "rejected")


@router.post("/clients/{account_id}/relationships/turn-off-link")
async def relationships_turn_off_link(request: Request, account_id: str):
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    client, model = _workspace(account_id)
    if model is None:
        return _refused("QueryBot has to learn this database first.")
    join = model.joins.get(str((await _body(request)).get("key") or ""))
    if join is None:
        return _refused("That link is not in the data any more.")
    _turn_off(account_id, client.get("db_config_id"), join)
    return JSONResponse({"ok": True})


@router.post("/clients/{account_id}/relationships/turn-on-link")
async def relationships_turn_on_link(request: Request, account_id: str):
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    client, model = _workspace(account_id)
    if model is None:
        return _refused("QueryBot has to learn this database first.")
    from core2.model.overrides import target

    key = str((await _body(request)).get("key") or "")
    if key not in model.joins:
        return _refused("That link is not in the data any more.")
    store.set_core2_override(account_id, client.get("db_config_id"), target("join", key), "trust", "admin")
    return JSONResponse({"ok": True})


def _rules(model, table_key: str, data: list[Any]) -> list:
    from core2.model.links import LinkError, value_problem
    from core2.model.schema import ColumnFilter

    rules = []
    for item in data or []:
        if not isinstance(item, dict):
            raise LinkError("A rule is not complete.")
        try:
            rule = ColumnFilter.model_validate({"column": item.get("column"), "op": item.get("op"),
                                                "values": [v for v in (item.get("values") or []) if str(v).strip()]})
        except Exception:
            raise LinkError("A rule is not complete: choose its column, how it compares and a value.") from None
        if model.columns.get(rule.column) is None or model.columns[rule.column].table != table_key:
            raise LinkError("A table leaves out rows by its own columns: choose one of them.")
        if rule.op not in ("is_null", "not_null") and not rule.values:
            raise LinkError(f"Give the rule on {model.columns[rule.column].business_name} a value.")
        problem = value_problem(model, rule)
        if problem:
            raise LinkError(problem)
        rules.append(rule)
    return rules


@router.post("/clients/{account_id}/relationships/check-table")
async def relationships_check_table(request: Request, account_id: str):
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    client, model = _workspace(account_id)
    if model is None:
        return _refused("QueryBot has to learn this database first.")
    from core2.model import links

    data = await _body(request)
    table = str(data.get("table") or "")
    if table not in model.tables:
        return _refused("That table is not in the data any more.")
    try:
        rules = _rules(model, table, data.get("leaves_out"))
    except links.LinkError as exc:
        return _refused(str(exc))
    warehouse = _warehouse(account_id, client)
    if warehouse is None:
        return _refused("This workspace has no database to check against.")
    return JSONResponse({"ok": True, **links.check_table(model, table, rules, warehouse)})


@router.post("/clients/{account_id}/relationships/save-table")
async def relationships_save_table(request: Request, account_id: str):
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    client, model = _workspace(account_id)
    if model is None:
        return _refused("QueryBot has to learn this database first.")
    from core2.model import links
    from core2.model.overrides import target

    data = await _body(request)
    key = str(data.get("table") or "")
    table = model.tables.get(key)
    if table is None:
        return _refused("That table is not in the data any more.")
    try:
        rules = _rules(model, key, data.get("leaves_out"))
    except links.LinkError as exc:
        return _refused(str(exc))
    db_id = client.get("db_config_id")
    object_key = target("table", key)
    name = " ".join(str(data.get("name") or "").split())[:120]
    if name and name != table.business_name:
        store.set_core2_override(account_id, db_id, object_key, "business_name", name)
    kind = str(data.get("kind") or "")
    if kind in _TABLE_KINDS and kind != table.kind:
        store.set_core2_override(account_id, db_id, object_key, "kind", kind)
    default_date = str(data.get("default_date") or "")
    if default_date and default_date != (table.default_date or ""):
        role = model.date_roles.get(default_date)
        if role is None or role.table != key:
            return _refused("Choose one of this table's own dates.")
        store.set_core2_override(account_id, db_id, object_key, "default_date", default_date)
    rules_json = [r.model_dump(mode="json") for r in rules]
    if rules_json != [f.model_dump(mode="json") for f in table.default_filters]:
        store.set_core2_override(account_id, db_id, object_key, "default_filters", rules_json)
    return JSONResponse({"ok": True})
