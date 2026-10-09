"""Readers' requests, from the reader who sends one to the admin who decides it.

A reader's suggestion (core2.model.requests) is filed as a request. An admin accepts it as
sent or edited, which writes each change as an admin decision, so it holds from the next
answer and through every Learn, or rejects it with a reason the reader is shown. A reader
the workspace lets edit directly (a portal admin) has their suggestion accepted as they send
it. Every accepted request can be undone: each decision it wrote goes back to what it was,
unless it has been changed again since.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import store
from core2.model import requests as R

log = logging.getLogger("querybot.core2")


class RequestRefused(ValueError):
    """The request cannot be decided as asked; the message is the admin's or the reader's."""


def _model(account_id: str, client: dict[str, Any]):
    from core2.bootstrap.service import load_model

    model = load_model(account_id, client.get("db_config_id"))
    if model is None:
        raise RequestRefused("QueryBot has not learned this workspace's data yet.")
    return model


def file_request(account_id: str, client: dict[str, Any], user: dict[str, Any], form: dict[str, Any], *,
                 allowed: set[str] | None, lang: str = "en") -> dict[str, Any]:
    """File a reader's suggestion; a reader who edits directly has it accepted at once.

    ``allowed`` is the reader's tables as QueryBot names them (None: every table).
    """
    from core2.service import _allowed_model_tables

    model = _model(account_id, client)
    user_id, user_name = str(user.get("id") or ""), str(user.get("name") or user.get("email") or "A reader")
    if len(store.list_core2_requests(account_id, status="waiting", user_id=user_id)) >= 20:
        raise RequestRefused("You have 20 suggestions waiting for your admin: wait for some of them first.")
    example = " ".join(str(form.get("example") or "").split())[:R.MAX_EXAMPLE]
    source = str(form.get("source") or "page")[:20]
    question_id = str(form.get("question_id") or "")[:64]
    if form.get("kind") == "new_metric":
        name = " ".join(str(form.get("name") or "").split())[:120]
        description = " ".join(str(form.get("description") or "").split())
        if not name or len(description) < 10:
            raise RequestRefused("Give the metric a name, and say in a sentence what it counts.")
        if len(description) > R.MAX_NOTE:
            raise RequestRefused(f"Keep the description under {R.MAX_NOTE} characters.")
        request_id = store.add_core2_request(
            account_id, client.get("db_config_id"), kind="new_metric", user_id=user_id, user_name=user_name,
            target_name=name, note=description, example=example, source=source, question_id=question_id)
        return store.get_core2_request(account_id, request_id) or {}
    try:
        name, changes = R.proposal(model, kind=str(form.get("target_kind") or ""), key=str(form.get("target_key") or ""),
                                   meaning=str(form.get("meaning") or ""), names=form.get("names") or "",
                                   date=str(form.get("date") or ""), note=str(form.get("note") or ""),
                                   allowed=_allowed_model_tables(model, allowed))
    except R.RequestError as exc:
        raise RequestRefused(str(exc)) from None
    request_id = store.add_core2_request(
        account_id, client.get("db_config_id"), kind="change", user_id=user_id, user_name=user_name,
        target_kind=str(form.get("target_kind")), target_key=str(form.get("target_key")), target_name=name,
        changes=changes, note=" ".join(str(form.get("note") or "").split()), example=example, source=source,
        question_id=question_id)
    request = store.get_core2_request(account_id, request_id) or {}
    if str(user.get("role") or "") == "admin" and changes:
        request = accept(account_id, client, request, decided_by=f"{user_name} (edits directly)", lang=lang)
    return request


def _checked(model, change: dict[str, Any], edited: Any) -> dict[str, Any]:
    """A change with the admin's edit, checked again against the model as it is now."""
    change = dict(change)
    if edited is None:
        return change
    if change["field"] == "description":
        text = " ".join(str(edited).split())
        if not text or len(text) > R.MAX_MEANING:
            raise RequestRefused(f"What it means has to be between 1 and {R.MAX_MEANING} characters.")
        change["value"] = text
    elif change["field"] == "synonyms":
        words = R.split_names(edited)
        if not words or len(words) > R.MAX_NAMES:
            raise RequestRefused(f"Give between 1 and {R.MAX_NAMES} other names.")
        change["value"] = words
    elif change["field"] == "default_date":
        kind, _, key = change["object_key"].partition(":")
        measure = model.measures.get(key)
        role = model.date_roles.get(str(edited))
        if measure is None or role is None or role.table != measure.table or role.kind == "audit":
            raise RequestRefused("That date cannot count this metric.")
        change["value"], change["shown"] = role.key, role.name
    return change


def accept(account_id: str, client: dict[str, Any], request: dict[str, Any], *, decided_by: str,
           edits: dict[str, Any] | None = None, skip: set[int] | None = None, lang: str = "en") -> dict[str, Any]:
    """Accept a waiting request: each change (as edited, less the skipped) becomes an admin decision."""
    if request.get("status") != "waiting":
        raise RequestRefused("This request has already been decided.")
    model = _model(account_id, client)
    db_id = client.get("db_config_id")
    changes = [_checked(model, c, (edits or {}).get(str(i))) for i, c in enumerate(request.get("changes") or [])
               if i not in (skip or set())]
    writes = []
    for change in changes:
        try:
            writes.append(R.to_write(model, change, lang=lang))
        except R.RequestError as exc:
            raise RequestRefused(str(exc)) from None
    # Claimed before anything is written: two admins cannot both apply it.
    if not store.decide_core2_request(account_id, int(request["id"]), status="accepted", decided_by=decided_by,
                                      changes=changes, applied=[]):
        raise RequestRefused("This request has already been decided.")
    applied = []
    for object_key, field, value in writes:
        before = store.get_core2_override(account_id, db_id, object_key, field)
        store.set_core2_override(account_id, db_id, object_key, field, value, author=decided_by,
                                 note=f"Request #{request['id']} from {request.get('user_name') or 'a reader'}")
        applied.append({"object_key": object_key, "field": field, "value": value, "before": before})
    store.decide_core2_request(account_id, int(request["id"]), status="accepted", decided_by=decided_by,
                               applied=applied, expect=("accepted",))
    return store.get_core2_request(account_id, int(request["id"])) or {}


def reject(account_id: str, request: dict[str, Any], *, decided_by: str, reason: str) -> dict[str, Any]:
    reason = " ".join(str(reason or "").split())[:500]
    if not store.decide_core2_request(account_id, int(request["id"]), status="rejected", decided_by=decided_by,
                                      reason=reason):
        raise RequestRefused("This request has already been decided.")
    return store.get_core2_request(account_id, int(request["id"])) or {}


def withdraw(account_id: str, request: dict[str, Any], *, user_id: str) -> None:
    if str(request.get("user_id")) != str(user_id):
        raise RequestRefused("Only the person who sent a suggestion can take it back.")
    if not store.decide_core2_request(account_id, int(request["id"]), status="withdrawn", decided_by="the reader"):
        raise RequestRefused("Your admin has already decided this suggestion.")


def undo(account_id: str, client: dict[str, Any], request: dict[str, Any], *, decided_by: str) -> list[str]:
    """Put back what an accepted request changed; returns the fields left as they are because they changed since."""
    if request.get("status") != "accepted":
        raise RequestRefused("Only an accepted request can be undone.")
    if not store.decide_core2_request(account_id, int(request["id"]), status="undone", decided_by=decided_by,
                                      reason=request.get("reason") or "", expect=("accepted",)):
        raise RequestRefused("This request has already been undone.")
    db_id = client.get("db_config_id")
    kept: list[str] = []
    for item in reversed(request.get("applied") or []):
        now = store.get_core2_override(account_id, db_id, item["object_key"], item["field"])
        if now is None or now["value"] != item["value"]:
            kept.append(item["field"])
            continue
        before = item.get("before")
        if before is None:
            store.delete_core2_override(account_id, db_id, item["object_key"], item["field"])
        else:
            store.set_core2_override(account_id, db_id, item["object_key"], item["field"], before["value"],
                                     author=before.get("author") or "admin", note=before.get("note") or "")
    return kept


def recent_decisions(account_id: str, user_id: str, *, days: int = 7) -> list[dict[str, Any]]:
    """A reader's requests decided in the last ``days``, for the notice that tells them."""
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    return [r for r in store.list_core2_requests(account_id, user_id=user_id, limit=50)
            if r["status"] in ("accepted", "rejected") and (r.get("decided_at") or "") >= since
            and not str(r.get("decided_by") or "").endswith("(edits directly)")]
