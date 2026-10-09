"""Admin → Data → Requests: changes readers asked for, accepted or rejected here.

Readers suggest from the Data guide (portal/core2_requests.py): what a metric or a
field means, other names people use for it, the date a metric is counted by, how it
should be counted, or a metric that is missing. Each waiting request shows what is there
now beside what was suggested. Accepting writes it as admin decisions (edited first if
need be), so it holds from the next answer and survives every Learn, and tells the reader;
rejecting tells them why. An accepted request can be undone. How a metric is counted, and
a new metric described in words, are written in the metric editor: the request links there
and is marked done once it is.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

import store
from admin.core2_relationships import _body, _refused, _workspace
from admin.routes import _is_auth, _resp, router, templates

log = logging.getLogger("querybot.core2")

KINDS = {"all": "All", "metrics": "Metrics", "meanings": "Meanings", "names": "Other names", "dates": "Dates",
         "new": "New metrics"}
DECIDER = "An admin"


def _waiting_count(account_id: str) -> int:
    try:
        return store.count_core2_requests(account_id).get("waiting", 0)
    except Exception:  # noqa: BLE001 - the menu still draws
        return 0


templates.env.globals["requests_waiting"] = _waiting_count


def _kinds(request: dict[str, Any]) -> set[str]:
    fields = {c.get("field") for c in request.get("changes") or []}
    out = {"all"}
    if request["kind"] == "new_metric":
        out.add("new")
    if request.get("target_kind") == "measure":
        out.add("metrics")
    if "description" in fields:
        out.add("meanings")
    if "synonyms" in fields or "value_names" in fields:
        out.add("names")
    if "default_date" in fields or request.get("target_kind") == "date_role":
        out.add("dates")
    return out


def _now_of(model, change: dict[str, Any]) -> Any:
    """What a changed field holds today (it may have moved since the request was sent)."""
    if model is None:
        return change.get("now")
    kind, _, key = change["object_key"].partition(":")
    thing: Any = {"measure": model.measures, "attribute": model.attributes, "entity": model.entities,
                  "date_role": model.date_roles, "column": model.columns}.get(kind, {}).get(key)
    if thing is None:
        return change.get("now")
    if change["field"] == "description":
        return thing.description
    if change["field"] == "synonyms":
        return list(dict.fromkeys(w for ws in thing.synonyms.values() for w in ws))
    if change["field"] == "value_names":
        return dict(getattr(thing, "value_names", {}) or {})
    if change["field"] == "default_date":
        wanted = thing.default_date or model.tables[thing.table].default_date
        role = model.date_roles.get(wanted or "")
        return role.name if role else ""
    return change.get("now")


def _card(model, request: dict[str, Any]) -> dict[str, Any]:
    target_kind, target_key = request.get("target_kind") or "", request.get("target_key") or ""
    dates = []
    if model is not None and target_kind == "measure" and target_key in model.measures:
        table = model.measures[target_key].table
        dates = [{"key": r.key, "name": r.name} for r in model.date_roles.values()
                 if r.table == table and r.kind != "audit"]
    changes = [{**c, "index": i, "today": _now_of(model, c) if request["status"] == "waiting" else c.get("now")}
               for i, c in enumerate(request.get("changes") or [])]
    made = None
    if request.get("definition") and model is not None:
        # A metric the reader made in a chat: how it is counted, as the admin reads a metric on the Metrics page.
        from core2.model import authoring
        from core2.model.schema import Measure
        from core2.plan.catalog import definition, left_out_words

        try:
            measure = Measure.model_validate(request["definition"])
            made = {"formula": authoring.to_text(model, measure.expr), "reads_as": definition(model, measure.expr),
                    "only": [left_out_words(model, f) for f in measure.filters if f.column in model.columns]}
        except Exception as exc:  # noqa: BLE001 - shown as unreadable; accepting it says why
            made = {"formula": "", "reads_as": f"It cannot be read any more ({str(exc)[:120]}).", "only": []}
    return {**request, "changes": changes, "dates": dates, "kinds": sorted(_kinds(request)), "made": made,
            "editor": target_kind == "measure" and target_key in (model.measures if model is not None else {})}


@router.get("/clients/{account_id}/requests", response_class=HTMLResponse)
async def requests_page(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client, model = _workspace(account_id)
    if client is None:
        return RedirectResponse("/admin/clients", status_code=303)
    status = request.query_params.get("status", "waiting")
    if status not in ("waiting", "accepted", "rejected"):
        status = "waiting"
    kind = request.query_params.get("kind", "all")
    if kind not in KINDS:
        kind = "all"
    shown = store.list_core2_requests(account_id, status=status)
    if status == "accepted":
        shown += store.list_core2_requests(account_id, status="undone")
        shown.sort(key=lambda r: (r.get("decided_at") or "", r["id"]), reverse=True)
    cards = [c for c in (_card(model, r) for r in shown) if kind in c["kinds"]]
    counts = store.count_core2_requests(account_id)
    return _resp(request, "client_requests.html", {
        "client": client, "learned": model is not None, "cards": cards, "status": status, "kind": kind,
        "kinds": KINDS, "counts": {"waiting": counts["waiting"], "accepted": counts["accepted"] + counts["undone"],
                                   "rejected": counts["rejected"]},
    })


async def _tell(account_id: str, decided: dict[str, Any]) -> None:
    """The reader who asked, and every admin page, hear of the decision."""
    try:
        from core.admin_notifications import notify_core2_request_changed
        from core.portal_notifications import notify_portal_semantic_feedback_changed

        if decided.get("user_id"):
            await notify_portal_semantic_feedback_changed(
                account_id=account_id, portal_user_id=int(decided["user_id"]) if str(decided["user_id"]).isdigit()
                else None, feedback_id=f"c2r-{decided['id']}",  # type: ignore[arg-type]
                status="approved" if decided["status"] == "accepted" else "rejected", table_fqn="",
                column_name=decided.get("target_name") or "", admin_note=decided.get("reason") or "")
        await notify_core2_request_changed(account_id=account_id, action=decided["status"], request_id=decided["id"])
    except Exception as exc:  # noqa: BLE001 - decided and stored; the reader sees it on their page
        log.warning("Telling the reader about request %s failed: %s", decided.get("id"), exc)


def _found(account_id: str, request_id: int) -> dict[str, Any] | None:
    return store.get_core2_request(account_id, request_id)


@router.post("/clients/{account_id}/requests/api/{request_id}/accept")
async def request_accept(request: Request, account_id: str, request_id: int):
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    from core2.request_service import RequestRefused, accept

    client = store.get_client(account_id)
    found = _found(account_id, request_id) if client else None
    if found is None:
        return _refused("That request is not there any more.", 404)
    data = await _body(request)
    edits = data.get("edits") if isinstance(data.get("edits"), dict) else {}
    skip = {int(i) for i in data.get("skip") or [] if str(i).isdigit()}
    try:
        decided = accept(account_id, client, found, decided_by=DECIDER, edits=edits, skip=skip)
    except RequestRefused as exc:
        return _refused(str(exc))
    await _tell(account_id, decided)
    return JSONResponse({"ok": True, "status": decided.get("status")})


@router.post("/clients/{account_id}/requests/api/{request_id}/reject")
async def request_reject(request: Request, account_id: str, request_id: int):
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    from core2.request_service import RequestRefused, reject

    found = _found(account_id, request_id)
    if found is None:
        return _refused("That request is not there any more.", 404)
    try:
        decided = reject(account_id, found, decided_by=DECIDER, reason=str((await _body(request)).get("reason") or ""))
    except RequestRefused as exc:
        return _refused(str(exc))
    await _tell(account_id, decided)
    return JSONResponse({"ok": True})


@router.post("/clients/{account_id}/requests/api/{request_id}/undo")
async def request_undo(request: Request, account_id: str, request_id: int):
    if not _is_auth(request):
        return _refused("Signed out.", 401)
    from core2.request_service import RequestRefused, undo

    client = store.get_client(account_id)
    found = _found(account_id, request_id) if client else None
    if found is None:
        return _refused("That request is not there any more.", 404)
    try:
        kept = undo(account_id, client, found, decided_by=DECIDER)
    except RequestRefused as exc:
        return _refused(str(exc))
    return JSONResponse({"ok": True, "kept": kept})
