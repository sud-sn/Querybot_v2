"""Admin → Data → Answers check: the client's own questions, with the answer from their own reports, and what
Solviz answers to each, set side by side (core2/answers_check.py).

The questions come from the setup check's Test questions tab, or are added here one at a time. The check runs by
itself (after a Learn, once decisions stop changing, as a question arrives); "Ask every question again" runs it now,
and each question can be asked again alone after a fix.
"""

from __future__ import annotations

import logging
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

import store
from admin.core2_setup_check import _by
from admin.routes import _is_auth, _resp, router

log = logging.getLogger("querybot.core2")

# Shown worst first: what needs an admin's eye heads the list.
_ORDER = {"different": 0, "unanswered": 1, "close": 2, "waiting": 3, "unchecked": 4, "matches": 5}
_REASONS = {"first": "its first run", "learn": "Solviz learned the database again", "new_questions": "new questions",
            "decisions": "decisions changed", "admin": "an admin asked"}


def _page(account_id: str, error: str = "", saved: str = "") -> RedirectResponse:
    query = f"?error={quote(error)}" if error else f"?saved={quote(saved)}" if saved else ""
    return RedirectResponse(f"/admin/clients/{account_id}/answers-check{query}", status_code=303)


def _rows(account_id: str, db_id: int) -> list[dict]:
    from core2.answers_check import said

    questions = store.list_core2_test_questions(account_id, db_id)
    results = store.core2_answer_results(account_id, db_id, per_question=2)
    out = []
    for q in questions:
        kept = results.get(int(q["id"])) or []
        latest = kept[0] if kept else None
        before = kept[1] if len(kept) > 1 else None
        verdict = latest["verdict"] if latest else "waiting"
        detail = (latest or {}).get("detail") or {}
        figures = [{**f, "got_said": said(f.get("got"), f, f.get("format") or "")} for f in detail.get("figures") or []]
        out.append({"q": q, "verdict": verdict, "detail": detail, "figures": figures, "latest": latest,
                    "was": before["verdict"] if before and before["verdict"] != verdict else ""})
    out.sort(key=lambda r: (_ORDER.get(r["verdict"], 9), r["q"]["id"]))
    return out


@router.get("/clients/{account_id}/answers-check", response_class=HTMLResponse)
async def answers_check_page(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    from core2.answers_check import summary

    db_id = int(client.get("db_config_id") or 0)
    store.release_stopped_core2_answer_checks(account_id, db_id)     # a run a restart stopped never holds the page
    check = store.latest_core2_answer_check(account_id, db_id)
    stamp = store.core2_model_stamp(account_id, db_id) if db_id else None
    whole = store.latest_core2_answer_check(account_id, db_id, finished=True, whole=True)
    # Learned again or decided since every question was last checked: the next run is on its way.
    behind = bool(whole and (check or {}).get("status") != "running" and stamp is not None
                  and list(whole.get("stamp") or []) != [*stamp])
    return _resp(request, "client_answers_check.html", {
        "client": client, "rows": _rows(account_id, db_id), "score": summary(account_id, db_id), "check": check,
        "reason": _REASONS.get((check or {}).get("reason", ""), (check or {}).get("reason", "")),
        "learned": stamp is not None, "behind": behind,
        "saved": request.query_params.get("saved"), "error": request.query_params.get("error"),
    })


@router.get("/clients/{account_id}/answers-check/status")
async def answers_check_status(request: Request, account_id: str):
    if not _is_auth(request):
        return JSONResponse({"ok": False}, status_code=401)
    client = store.get_client(account_id) or {}
    db_id = int(client.get("db_config_id") or 0)
    store.release_stopped_core2_answer_checks(account_id, db_id)
    check = store.latest_core2_answer_check(account_id, db_id) or {}
    return JSONResponse({"ok": True, "id": check.get("id"), "status": check.get("status") or "",
                         "done": int(check.get("done_count") or 0), "total": int(check.get("total") or 0)})


@router.post("/clients/{account_id}/answers-check/run")
async def answers_check_run(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    from core2 import answers_check

    db_id = int(client.get("db_config_id") or 0)
    if store.core2_model_stamp(account_id, db_id) is None:
        return _page(account_id, "Learn the database first: the questions are answered from what Solviz learned.")
    if answers_check.learning(store.latest_core2_build(account_id, db_id)):
        return _page(account_id, "Solviz is learning the database: the questions are checked once it has finished.")
    form = await request.form()
    only = [int(form["question"])] if str(form.get("question") or "").isdigit() else None
    questions = {int(q["id"]) for q in store.list_core2_test_questions(account_id, db_id)}
    if not questions:
        return _page(account_id, "Add a question to check first.")
    if only is not None and not set(only) & questions:
        return _page(account_id, "That question was removed.")
    started = answers_check.start(account_id, mode="ask", reason="admin", only=only, started_by=_by(request))
    if started is None:
        return _page(account_id, "A check is running already; this page shows it as it goes.")
    return _page(account_id, saved="running")


@router.post("/clients/{account_id}/answers-check/questions")
async def answers_check_add(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    form = await request.form()
    question = " ".join(str(form.get("question") or "").split())[:500]
    if not question:
        return _page(account_id, "Write the question the client's team asks.")
    store.add_core2_test_question(account_id, client.get("db_config_id"), question=question,
                                  expected=str(form.get("expected") or "").strip()[:300],
                                  report=str(form.get("report") or "").strip()[:120],
                                  period=str(form.get("period") or "").strip()[:60],
                                  note=f"Added by {_by(request)}"[:300])
    return _page(account_id, saved="added")


@router.post("/clients/{account_id}/answers-check/questions/{question_id}/delete")
async def answers_check_delete(request: Request, account_id: str, question_id: int):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    store.delete_core2_test_question(account_id, client.get("db_config_id"), question_id)
    return _page(account_id, saved="removed")
