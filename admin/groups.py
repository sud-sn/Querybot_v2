"""Admin → People → Groups: people who see the same data, gathered once.

A group has members (a person is in every group they are put in), and what is given to the group reaches each of
them: its row rules (Row access), its tables, the dashboards shared with it, the emails and alerts sent to it. The
list shows every group with its size, its rules and its tables; a group's own page adds and takes out people,
renames it, and shows the row rules written for it, with what they leave each member.
"""

from __future__ import annotations

import json
import logging
from urllib.parse import quote

from fastapi import Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

import store
from admin import routes as _admin
from admin.core2_relationships import _body, _refused, _workspace
from admin.routes import _record_user_event, _resp, router

log = logging.getLogger("querybot.groups")


def _is_auth(request: Request) -> bool:
    return _admin._is_auth(request)


def _group_of(account_id: str, group_id: int) -> dict | None:
    group = store.get_group(int(group_id))
    return group if group and group.get("account_id") == account_id else None


def _all_tables(client: dict) -> list[str]:
    """The tables a group can be given: those discovery found for the workspace."""
    try:
        state = json.loads(client.get("state_data") or "{}")
    except (TypeError, ValueError):
        state = {}
    schema_dir = state.get("schema_dir", "")
    return sorted(store.load_schema_tables(schema_dir)) if schema_dir else []


def _rules_by_group(account_id: str) -> dict[str, int]:
    counted: dict[str, int] = {}
    for rule in store.list_row_rules(account_id):
        if rule.get("subject_type") == "group":
            counted[str(rule.get("subject_id") or "")] = counted.get(str(rule.get("subject_id") or ""), 0) + 1
    return counted


@router.get("/clients/{account_id}/groups", response_class=HTMLResponse)
async def groups_page(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    rules = _rules_by_group(account_id)
    groups = store.list_groups(account_id)
    for g in groups:
        g["tables"] = store.get_group_tables(g["id"])
        g["rules"] = rules.get(str(g["id"]), 0)
        g["members"] = store.group_members(g["id"])
    return _resp(request, "client_groups.html", {
        "client": client, "groups": groups,
        "people": sum(1 for p in store.list_users(account_id) if p.get("is_active", 1)),
        "pending_count": store.get_pending_user_count(account_id),
        "saved": request.query_params.get("saved"), "error": request.query_params.get("error"),
    })


@router.post("/clients/{account_id}/groups/create")
async def group_create(request: Request, account_id: str, name: str = Form(""), description: str = Form("")):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    if not store.get_client(account_id):
        return RedirectResponse("/admin/clients", status_code=303)
    try:
        gid = store.create_group(account_id, name, description)
    except ValueError as exc:
        return RedirectResponse(f"/admin/clients/{account_id}/groups?error={quote(str(exc))}", status_code=303)
    return RedirectResponse(f"/admin/clients/{account_id}/groups/{gid}?saved=created", status_code=303)


@router.get("/clients/{account_id}/groups/{group_id}", response_class=HTMLResponse)
async def group_page(request: Request, account_id: str, group_id: int):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client, model = _workspace(account_id)
    if client is None:
        return RedirectResponse("/admin/clients", status_code=303)
    group = _admin._workspace_group(account_id, group_id)
    from admin import row_access

    fields = row_access.fields_of(model)
    members = store.group_members(group_id)
    inside = {m["id"] for m in members}
    people = [p for p in store.list_users(account_id) if p["id"] not in inside]
    rules = [{**r, "on": row_access._field_label(r, fields), "sees": row_access._sees(r)}
             for r in store.list_row_rules(account_id)
             if r.get("subject_type") == "group" and str(r.get("subject_id")) == str(group_id)]
    summary = row_access.summary(store.list_row_rules(account_id), [m for m in members if m.get("is_active", 1)],
                                 fields) if rules else []
    return _resp(request, "client_group.html", {
        "client": client, "group": group, "members": members,
        "candidates": [{"id": p["id"], "name": p.get("name") or p.get("email"), "email": p.get("email") or "",
                        "groups": p.get("group_name") or "", "active": bool(p.get("is_active", 1))} for p in people],
        "rules": rules, "summary": summary, "tables": store.get_group_tables(group_id),
        "all_tables": _all_tables(client), "learned": model is not None,
        "pending_count": store.get_pending_user_count(account_id),
        "saved": request.query_params.get("saved"), "error": request.query_params.get("error"),
    })


@router.post("/clients/{account_id}/groups/{group_id}/members")
async def group_add_members(request: Request, account_id: str, group_id: int):
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    group = _group_of(account_id, group_id)
    if group is None:
        return _refused("That group is not in this workspace.", 404)
    data = await _body(request)
    wanted = data.get("user_ids") if isinstance(data.get("user_ids"), list) else []
    if not all(str(u).isdigit() for u in wanted):
        return _refused("Choose people from the list.")
    before = {m["id"] for m in store.group_members(group_id)}
    try:
        added = store.add_group_members(group_id, account_id, [int(u) for u in wanted])
    except ValueError as exc:
        return _refused(str(exc))
    for person in store.group_members(group_id):
        if person["id"] not in before:
            _record_user_event(request, account_id, person, "group_changed", detail=f"added to {group['name']}")
    return JSONResponse({"ok": True, "added": added, "members": len(before) + added})


@router.post("/clients/{account_id}/groups/{group_id}/members/{user_id}/remove")
async def group_remove_member(request: Request, account_id: str, group_id: int, user_id: int):
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    group = _group_of(account_id, group_id)
    if group is None:
        return _refused("That group is not in this workspace.", 404)
    person = store.get_user(int(user_id))
    if not person or person.get("account_id") != account_id:
        return _refused("That person is not in this workspace.", 404)
    if not store.remove_group_member(group_id, account_id, user_id):
        return _refused(f"{person.get('name') or 'They'} is not in this group any more.", 404)
    _record_user_event(request, account_id, person, "group_changed", detail=f"taken out of {group['name']}")
    return JSONResponse({"ok": True})


@router.post("/clients/{account_id}/groups/{group_id}/rename")
async def group_rename(request: Request, account_id: str, group_id: int):
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    group = _group_of(account_id, group_id)
    if group is None:
        return _refused("That group is not in this workspace.", 404)
    data = await _body(request)
    try:
        store.update_group(group_id, str(data.get("name") or ""), str(data.get("description") or ""))
    except ValueError as exc:
        return _refused(str(exc))
    return JSONResponse({"ok": True})


@router.post("/clients/{account_id}/groups/{group_id}/tables")
async def group_save_tables(request: Request, account_id: str, group_id: int):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    _admin._workspace_group(account_id, group_id)
    form = await request.form()
    store.set_group_tables(group_id, account_id, [v for k, v in form.multi_items() if k == "tables"])
    return RedirectResponse(f"/admin/clients/{account_id}/groups/{group_id}?saved=tables", status_code=303)


@router.post("/clients/{account_id}/groups/{group_id}/delete")
async def group_delete(request: Request, account_id: str, group_id: int):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    group = _admin._workspace_group(account_id, group_id)
    # Its rules name a group that is gone: they go with it, rather than wait for a group made with its number.
    for rule in store.list_row_rules(account_id):
        if rule.get("subject_type") == "group" and str(rule.get("subject_id")) == str(group_id):
            store.delete_row_rule(account_id, int(rule["id"]))
    for person in store.group_members(group_id):
        _record_user_event(request, account_id, person, "group_changed", detail=f"{group['name']} deleted")
    store.delete_group(group_id)
    return RedirectResponse(f"/admin/clients/{account_id}/groups?saved=deleted", status_code=303)


@router.post("/clients/{account_id}/users/{user_id}/groups")
async def user_groups_save(request: Request, account_id: str, user_id: int):
    """The People page's Groups dialog: the groups a person is in, ticked."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    person = _admin._workspace_user(account_id, user_id)
    form = await request.form()
    try:
        wanted = [int(v) for k, v in form.multi_items() if k == "group_ids" and str(v).isdigit()]
        before = list(person.get("group_ids") or [])
        after = store.set_user_groups(user_id, account_id, wanted)
    except ValueError as exc:
        return RedirectResponse(f"/admin/clients/{account_id}/users?error={quote(str(exc))}", status_code=303)
    if after != before:
        names = {g["id"]: g["name"] for g in store.list_groups(account_id)}
        _record_user_event(request, account_id, person, "group_changed",
                           detail=", ".join(names.get(g, f"#{g}") for g in after) or "no group")
    return RedirectResponse(f"/admin/clients/{account_id}/users?saved=1", status_code=303)
