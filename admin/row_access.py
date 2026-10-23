"""Admin → People → Row access: which rows each person sees.

A rule says who (everyone, a role, a group, one person), the field it is on (a store's region, a customer's
segment: a grouping the workspace learned) and what of it they see: some of its values, their own value of it
(each person's values are kept on this page too), or every row (which lifts that field for them, over the other
rules). A rule on a field narrows its own table and every table linked to it: the sales of those stores, the
orders of those customers (core/compliance/row_links.py). Everything a person reads goes through it: their
questions, dashboards, Home, schedules and alerts, exports.

The page also says, for each person, what the rules leave them, in words, and warns of a person a rule reads a
value of theirs for who has none: they see none of the rows it covers.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

import store
from admin.core2_relationships import _body, _refused, _warehouse, _workspace
from admin.routes import _is_auth, _resp, router

log = logging.getLogger("querybot.row_access")

ROLES = {"analyst": "Analyst", "admin": "Admin"}
MOST_VALUES = 200


def _named(table) -> str:
    return ".".join(p for p in (table.schema_name, table.name) if p).upper()


_NUMBERS = ("integer", "decimal", "float")


def _words(name: str) -> str:
    """A column's name as a label: "DOCTOR_ID" -> "Doctor ID"."""
    return " ".join(w.upper() if w.lower() in ("id", "no", "nr") else w.capitalize()
                    for w in name.replace("_", " ").split()) or name


def fields_of(model) -> list[dict[str, Any]]:
    """The fields a rule can be on: the groupings and names people read the data by, and the ID columns, each with
    the table it is on, as the warehouse spells them. "Store" (a store by its name), "Store › Region", "Store › ID"
    (a store by its own key), "Sales › Store ID" (the key the sales rows point at it with)."""
    if model is None:
        return []
    out: dict[tuple[str, str], dict[str, Any]] = {}

    def add(column_key: str, label: str, owner: str) -> None:
        column = model.columns.get(column_key)
        table = model.tables.get(column.table) if column is not None else None
        if column is None or table is None or getattr(column, "personal", "none") != "none":
            return
        key = (_named(table), column.name)
        if key not in out:
            out[key] = {"table": key[0], "field": column.name, "label": f"{owner} › {label}" if label else owner,
                        "owner": owner, "name": label, "numeric": column.data_type in _NUMBERS}

    for entity in model.entities.values():
        owner = entity.business_name or entity.slug.replace("_", " ").title()
        if entity.label_column:
            add(entity.label_column, "", owner)          # a store by its name: "Store"
        if entity.code_column:
            add(entity.code_column, "Code", owner)
    for attribute in model.attributes.values():
        if attribute.kind not in ("group", "identifier"):
            continue
        entity = model.entities.get(attribute.entity) if attribute.entity else None
        column = model.columns.get(attribute.column)
        table = model.tables.get(column.table) if column is not None else None
        owner = (entity.business_name if entity is not None and entity.business_name
                 else (table.business_name if table is not None and table.business_name else
                       (table.name.title() if table is not None else "")))
        add(attribute.column, attribute.business_name or attribute.slug.split(".")[-1].replace("_", " ").title(),
            owner)
    # The ID columns. A member's own key ("Doctor › ID") narrows the members and every table linked to them; a key
    # a table's rows point at a member with ("Revenue › Doctor ID") narrows those rows and the tables linked to
    # them, never the list of members. Not a calendar's key, nor a key pointing at one: a date is no one's row.
    calendars = {k for k, t in model.tables.items() if t.kind == "calendar"}
    calendars |= {j.to_table for j in model.joins.values() if j.to_calendar}
    to_calendar = {c for j in model.joins.values() if j.to_calendar for c in j.from_columns}
    for entity in model.entities.values():
        if len(entity.key_columns) == 1 and entity.table not in calendars:
            add(entity.key_columns[0], "ID", entity.business_name or entity.slug.replace("_", " ").title())
    pointing = {j.from_columns[0] for j in model.joins.values()
                if len(j.from_columns) == 1 and not j.to_calendar and j.trust != "rejected"}
    for column in model.columns.values():
        if (column.role == "foreign_key" or column.key in pointing) and column.key not in to_calendar \
                and column.table not in calendars and not column.parts:
            table = model.tables.get(column.table)
            if table is not None:
                label = re.sub(r"\bid\b", "ID", column.business_name, flags=re.IGNORECASE) or _words(column.name)
                add(column.key, label, table.business_name or _words(table.name))
    return sorted(out.values(), key=lambda f: (f["owner"].lower(), f["name"].lower()))


def _number(value: Any) -> int | float | None:
    """A typed value as the number it is, or None: "1042" -> 1042, "12.5" -> 12.5, "D-12" -> None."""
    from core.compliance.sql_guard import as_number

    return as_number(value)


def _who(rule: dict[str, Any], groups: dict[str, str], people: dict[str, str]) -> str:
    kind, whom = rule.get("subject_type"), str(rule.get("subject_id") or "")
    if kind == "everyone":
        return "Everyone"
    if kind == "role":
        return f"Every {ROLES.get(whom, whom).lower()}"
    if kind == "group":
        return f"Group {groups.get(whom, '#' + whom)}"
    if kind == "user":
        return people.get(whom, f"Person #{whom}")
    return kind or "?"


def _sees(rule: dict[str, Any]) -> str:
    condition = rule.get("condition") or {}
    if str(condition.get("operator") or "").upper() == "ALL":
        return "every row"
    source = str(condition.get("value_source") or "static")
    if source.startswith("user.attributes."):
        return f"their own {source.split('.', 2)[-1].replace('_', ' ')}"
    values = condition.get("value")
    values = values if isinstance(values, list) else [values]
    shown = ", ".join(str(v) for v in values[:6])
    return f"only {shown}" + (f" and {len(values) - 6} more" if len(values) > 6 else "")


def _field_label(rule: dict[str, Any], fields: list[dict[str, Any]]) -> str:
    table, field = str(rule.get("table_fqn") or ""), str((rule.get("condition") or {}).get("field") or "")
    for f in fields:
        if f["field"].upper() == field.upper() and (f["table"] == table or f["table"].endswith("." + table)
                                                     or table.endswith("." + f["table"])):
            return f["label"]
    return f"{table.title()} › {field}"


def _binds(rule: dict[str, Any], person: dict[str, Any]) -> bool:
    """Does the rule name this person: everyone, their role, a group they are in (any of them), or them?"""
    from types import SimpleNamespace

    from core.compliance.policy_engine import in_group, person_groups

    kind, whom = rule.get("subject_type"), str(rule.get("subject_id") or "")
    if kind == "group":
        ids, names = person_groups(person)
        return in_group(whom, SimpleNamespace(groups=ids, group_names=names))
    return (kind == "everyone" or (kind == "role" and whom == str(person.get("role") or ""))
            or (kind == "user" and whom == str(person.get("id"))))


def _attribute_keys(rules: list[dict[str, Any]]) -> list[str]:
    keys = []
    for rule in rules:
        source = str((rule.get("condition") or {}).get("value_source") or "")
        if source.startswith("user.attributes."):
            key = source.split(".", 2)[-1]
            if key not in keys:
                keys.append(key)
    return keys


def summary(rules: list[dict[str, Any]], people: list[dict[str, Any]], fields: list[dict[str, Any]]) -> list[dict]:
    """For each person, what the rules leave them, and the rules that read a value they have none of."""
    out = []
    for person in people:
        if not person.get("is_active", 1):
            continue
        own = store.user_attributes(person)
        groups = {str(g["id"]): g["name"] for g in person.get("groups") or []}
        bound = [r for r in rules if r.get("enabled", 1) and _binds(r, person)]
        lines, missing = [], []
        by_field: dict[str, list[str]] = {}
        for rule in bound:
            label = _field_label(rule, fields)
            sees = _sees(rule)
            source = str((rule.get("condition") or {}).get("value_source") or "")
            if source.startswith("user.attributes."):
                key = source.split(".", 2)[-1]
                value = own.get(key)
                if not value:
                    missing.append(key.replace("_", " "))
                    sees = f"their own {key.replace('_', ' ')} (none set: no rows)"
                else:
                    sees = "only " + (", ".join(value) if isinstance(value, list) else str(value))
            if rule.get("subject_type") == "group":
                # Which of their groups gives them these rows: a person in several sees each one's.
                sees += f" ({groups.get(str(rule.get('subject_id')), rule.get('subject_id'))})"
            by_field.setdefault(label, []).append(sees)
        for label, ways in by_field.items():
            lines.append(f"{label}: every row" if "every row" in ways else f"{label}: " + "; or ".join(ways))
        out.append({"id": person["id"], "name": person.get("name") or person.get("email"),
                    "lines": lines or ["Every row"], "missing": missing})
    return out


@router.get("/clients/{account_id}/row-access", response_class=HTMLResponse)
async def row_access_page(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client, model = _workspace(account_id)
    if client is None:
        return RedirectResponse("/admin/clients", status_code=303)
    fields = fields_of(model)
    rules = store.list_row_rules(account_id)
    groups = store.list_groups(account_id)
    people = store.list_users(account_id)
    group_names = {str(g["id"]): g["name"] for g in groups}
    person_names = {str(p["id"]): p.get("name") or p.get("email") or "" for p in people}
    sizes = {str(g["id"]): int(g.get("member_count") or 0) for g in groups}
    cards = [{**rule, "who": _who(rule, group_names, person_names), "on": _field_label(rule, fields),
              "sees": _sees(rule), "members": sizes.get(str(rule.get("subject_id"))) if rule.get("subject_type") ==
              "group" else None} for rule in rules]
    keys = _attribute_keys(rules)
    # Opened from a group's page ("Add a row rule for this group"): the form opens for that group.
    who, whom = request.query_params.get("who", ""), request.query_params.get("whom", "")
    opened = {"who": who, "whom": whom} if who == "group" and whom in group_names else None
    return _resp(request, "client_row_access.html", {
        "client": client, "learned": model is not None, "fields": fields, "rules": cards, "groups": groups,
        "open": opened, "pending_count": store.get_pending_user_count(account_id),
        "people": [{"id": p["id"], "name": p.get("name") or p.get("email"), "email": p.get("email"),
                    "group": p.get("group_name") or "", "role": ROLES.get(str(p.get("role") or ""), p.get("role")),
                    "values": store.user_attributes(p), "active": bool(p.get("is_active", 1))} for p in people],
        "roles": ROLES, "keys": keys, "summary": summary(rules, people, fields),
    })


def _rule_from(data: dict[str, Any], fields: list[dict[str, Any]]) -> dict[str, Any]:
    field = next((f for f in fields if f["table"] == str(data.get("table") or "") and
                  f["field"] == str(data.get("field") or "")), None)
    if field is None:
        raise ValueError("Choose the field the rule is on, from the list.")
    sees = str(data.get("sees") or "")
    if sees == "every":
        condition: dict[str, Any] = {"field": field["field"], "operator": "ALL"}
    elif sees == "own":
        condition = {"field": field["field"], "operator": "IN",
                     "value_source": f"user.attributes.{str(data.get('key') or '').strip().lower().replace(' ', '_')}"}
    else:
        values = data.get("values")
        values = values if isinstance(values, list) else str(values or "").split(",")
        typed = [str(v).strip() for v in values if str(v).strip()]
        if field.get("numeric"):
            # A number column compared with a word fails the whole question in most databases: said here, once.
            wrong = next((v for v in typed if _number(v) is None), None)
            if wrong is not None:
                raise ValueError(f"{field['label']} holds numbers: \u201c{wrong}\u201d is not one.")
            typed = [_number(v) for v in typed]
        condition = {"field": field["field"], "operator": "IN", "value_source": "static", "value": typed}
    if field.get("numeric") and sees != "every":
        condition["numeric"] = True                     # its values compared as numbers (core/compliance/sql_guard.py)
    return {"subject_type": str(data.get("who") or ""), "subject_id": str(data.get("whom") or ""),
            "table_fqn": field["table"], "name": field["label"], "condition": condition}


@router.post("/clients/{account_id}/row-access/api/rules")
async def row_rule_save(request: Request, account_id: str):
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    client, model = _workspace(account_id)
    if client is None:
        return _refused("That workspace is not there.", 404)
    data = await _body(request)
    try:
        rule = _rule_from(data, fields_of(model))
        rule_id = store.save_row_rule(account_id, rule, int(data["id"]) if str(data.get("id") or "").isdigit() else None)
    except ValueError as exc:
        return _refused(str(exc))
    return JSONResponse({"ok": True, "id": rule_id})


@router.post("/clients/{account_id}/row-access/api/rules/{rule_id}/delete")
async def row_rule_delete(request: Request, account_id: str, rule_id: int):
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    if not store.delete_row_rule(account_id, rule_id):
        return _refused("That rule is not there any more.", 404)
    return JSONResponse({"ok": True})


@router.post("/clients/{account_id}/row-access/api/people/{user_id}")
async def row_person_values(request: Request, account_id: str, user_id: int):
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    person = store.get_user(user_id)
    if not person or person.get("account_id") != account_id:
        return _refused("That person is not in this workspace.", 404)
    data = await _body(request)
    values = data.get("values") if isinstance(data.get("values"), dict) else {}
    try:
        kept = store.set_user_attributes(user_id, {**store.user_attributes(person), **values})
    except ValueError as exc:
        return _refused(str(exc))
    return JSONResponse({"ok": True, "values": kept})


@router.get("/clients/{account_id}/row-access/api/values")
async def row_field_values(request: Request, account_id: str, table: str = "", field: str = ""):
    """The values a field holds, to choose from: read from the warehouse as the workspace's own work, the first
    MOST_VALUES in order."""
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    client, model = _workspace(account_id)
    if client is None:
        return _refused("That workspace is not there.", 404)
    found = next((f for f in fields_of(model) if f["table"] == table and f["field"] == field), None)
    if found is None:
        return _refused("Choose a field from the list.", 404)
    warehouse = _warehouse(account_id, client)
    if warehouse is None:
        return _refused("The workspace has no database yet.", 409)
    from core2.warehouse.dialect import quote, table_sql

    column = model.columns.get(next(k for k, c in model.columns.items() if c.name == field and
                                    _named(model.tables[c.table]) == table))
    source = model.tables[column.table]
    dialect = warehouse.dialect
    name = quote(field, dialect)
    sql = (f"SELECT DISTINCT {name} AS v FROM {table_sql(source.database, source.schema_name, source.name, dialect)} "
           f"WHERE {name} IS NOT NULL ORDER BY 1")
    try:
        result = warehouse.query(sql, max_rows=MOST_VALUES)
    except Exception as exc:  # noqa: BLE001 - the values are a help; they can be typed
        log.warning("row access: the values of %s.%s could not be read for %s: %s", table, field, account_id, exc)
        return _refused("The values could not be read just now: type them instead.", 502)
    return JSONResponse({"ok": True, "values": [str(r[0]) for r in result.rows if r and r[0] is not None]})
