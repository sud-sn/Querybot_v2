"""Admin pages for the new core (docs/core-v2/DESIGN.md): what QueryBot learned.

Kept apart from admin/routes.py so the new core's admin surface can be read,
tested and removed in one piece. Authentication and rendering are the admin
console's own helpers.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any
from urllib.parse import quote

from fastapi import BackgroundTasks, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

import store
# The pages are registered on the admin console's own router (admin/routes.py
# imports this module at its end), so every /admin page sits on one router.
from admin.routes import _is_auth, _resp, router

log = logging.getLogger("querybot.core2")

# A Learn runs inside the service process that started it (the service runs one worker): one
# another process started was cut off by a restart and will never finish. A large warehouse can
# take hours; past this it is not running, whatever its row says.
_BUILD_STALE = dt.timedelta(hours=6)


def _back(account_id: str, **params: str) -> RedirectResponse:
    query = "&".join(f"{k}={quote(v)}" for k, v in params.items())
    return RedirectResponse(f"/admin/clients/{account_id}/learned" + (f"?{query}" if query else ""), status_code=303)


def _started(build: dict[str, Any]) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(str(build["started_at"]), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
    except (KeyError, TypeError, ValueError):
        return None


def _running(build: dict[str, Any] | None) -> bool:
    if not build or build.get("status") != "running":
        return False
    started = _started(build)
    if started is None or build.get("runner") != store.core2_runner():
        return False
    return dt.datetime.now(dt.timezone.utc) - started < _BUILD_STALE


def _interrupted(build: dict[str, Any] | None) -> bool:
    """A Learn left running by a service that restarted (or one past any sane length)."""
    return bool(build) and (build or {}).get("status") == "running" and not _running(build)


def _lines(build: dict[str, Any] | None) -> list[str]:
    return [line for line in str((build or {}).get("log") or "").splitlines() if line.strip()]


@router.get("/clients/{account_id}/learned/progress")
async def learned_progress(request: Request, account_id: str):
    """What the running Learn is doing, for the learned page to show as it happens."""
    if not _is_auth(request):
        return JSONResponse({"error": "signed out"}, status_code=401)
    client = store.get_client(account_id)
    if not client:
        return JSONResponse({"error": "no such workspace"}, status_code=404)
    build = store.latest_core2_build(account_id, client.get("db_config_id"))
    return JSONResponse({"building": _running(build), "status": (build or {}).get("status") or "",
                         "lines": _lines(build)})


@router.get("/clients/{account_id}/learned", response_class=HTMLResponse)
async def learned_page(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    from core2.bootstrap.service import load_model
    from core2.model.view import learned_view

    db_id = client.get("db_config_id")
    model = None
    problem = ""
    try:
        model = load_model(account_id, db_id)
    except Exception as exc:  # noqa: BLE001 - a model this release cannot read is shown as such, not a 500
        log.warning("core2: stored model for %s could not be read: %s", account_id, exc, exc_info=True)
        problem = "The stored model was written by another release and cannot be read. Learn again to rebuild it."
    build = store.latest_core2_build(account_id, db_id)
    try:
        imported = json.loads((build or {}).get("import_report") or "{}")
    except (TypeError, ValueError):
        imported = {}
    decisions = store.list_core2_overrides(account_id, db_id)
    return _resp(request, "client_learned.html", {
        "client": client,
        "view": learned_view(model) if model else None,
        "build": build,
        "building": _running(build),
        "interrupted": _interrupted(build),
        "build_lines": _lines(build),
        "build_left_out": sum(" Left out " in line for line in _lines(build)),
        "decisions": [d for d in decisions if d["author"] != "import"],
        "brought_over": [d for d in decisions if d["author"] == "import"],
        "imported": imported,
        "problem": problem,
        "has_database": bool(db_id),
        "saved": request.query_params.get("saved"),
        "error": request.query_params.get("error"),
        "engine": store.get_query_engine(account_id),
        "answers": store.list_core2_answers(account_id, 20),
    })


def _run_build(account_id: str) -> None:
    from core2.bootstrap.service import build_workspace

    try:
        build_workspace(account_id)
    except Exception:  # noqa: BLE001 - recorded in the build log by build_workspace, shown on the page
        pass


@router.post("/clients/{account_id}/learned/build")
async def learned_build(request: Request, account_id: str, bg: BackgroundTasks):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    if not client.get("db_config_id"):
        return _back(account_id, error="Connect a database first.")
    if _running(store.latest_core2_build(account_id, client["db_config_id"])):
        return _back(account_id, error="QueryBot is already learning this database.")
    bg.add_task(_run_build, account_id)
    return _back(account_id, saved="building")


def _value(raw: str) -> Any:
    text = raw.strip()
    if text in ("true", "false"):
        return text == "true"
    if text[:1] in "[{\"":
        try:
            return json.loads(text)
        except ValueError:
            return text
    return text


@router.post("/clients/{account_id}/learned/decide")
async def learned_decide(request: Request, account_id: str, target: str = Form(...), field: str = Form(...),
                         value: str = Form(""), note: str = Form("")):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    from core2.model.overrides import ALLOWED

    kind = target.partition(":")[0]
    if field not in ALLOWED.get(kind, set()):
        return _back(account_id, error=f"{field} cannot be changed here.")
    store.set_core2_override(account_id, client.get("db_config_id"), target, field, _value(value), note=note[:500])
    return _back(account_id, saved="decision")


@router.post("/clients/{account_id}/learned/undo")
async def learned_undo(request: Request, account_id: str, target: str = Form(...), field: str = Form(...)):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    store.delete_core2_override(account_id, client.get("db_config_id"), target, field)
    return _back(account_id, saved="undone")


@router.post("/clients/{account_id}/learned/measure")
async def learned_measure(request: Request, account_id: str, target: str = Form(...), name: str = Form(""),
                          synonyms: str = Form(""), adds_up: str = Form("")):
    """A measure's name, the other words readers use for it, and how it adds up over time, in one form."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    from core2.bootstrap.service import load_model
    from core2.model.view import ADDS_UP_CHOICES

    db_id = client.get("db_config_id")
    model = load_model(account_id, db_id) if target.startswith("measure:") else None
    measure = model.measures.get(target.partition(":")[2]) if model else None
    if measure is None:
        return _back(account_id, error="That measure is not in the data any more. Learn again, then change it.")
    text = " ".join(name.split())[:120]
    if text and text != measure.business_name:
        store.set_core2_override(account_id, db_id, target, "business_name", text)
    words = list(dict.fromkeys(w for w in (" ".join(x.split()) for x in synonyms.split(",")) if w))[:30]
    if sorted(words) != sorted({w for ws in measure.synonyms.values() for w in ws}):
        store.set_core2_override(account_id, db_id, target, "synonyms", {"en": words})
    choice = ADDS_UP_CHOICES.get(adds_up)
    if choice and getattr(measure.expr, "agg", None) == "sum" \
            and choice != (measure.additivity, measure.time_aggregation):
        store.set_core2_override(account_id, db_id, target, "additivity", choice[0])
        store.set_core2_override(account_id, db_id, target, "time_aggregation", choice[1])
    return _back(account_id, saved="decision")


@router.post("/clients/{account_id}/learned/import")
async def learned_import(request: Request, account_id: str):
    """Bring today's decisions over again (after an admin changed them in today's setup)."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    from core2.bootstrap.service import bring_over, learned_model

    model = learned_model(account_id, client.get("db_config_id"))
    if model is None:
        return _back(account_id, error="QueryBot has to learn this database first.")
    report = bring_over(account_id, client.get("db_config_id"), model, client)
    if report.get("error"):
        return _back(account_id, error="Today's decisions could not be brought over: " + str(report["error"])[:200])
    return _back(account_id, saved="imported")


@router.post("/clients/{account_id}/learned/engine")
async def learned_engine(request: Request, account_id: str, engine: str = Form(...)):
    """Who answers the workspace's portal questions: today's pipeline, both side by side, or the new core."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    if engine not in store.core2_store.ENGINES:
        return _back(account_id, error="Choose one of the three ways to answer.")
    if engine != "legacy":
        from core2.bootstrap.service import load_model

        if load_model(account_id, client.get("db_config_id")) is None:
            return _back(account_id, error="QueryBot has to learn this database before the new core can answer.")
    store.set_query_engine(account_id, engine)
    log.info("core2: %s now answers with %s", account_id, engine)
    return _back(account_id, saved="engine")
