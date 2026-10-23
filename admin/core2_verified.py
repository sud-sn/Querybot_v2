"""Admin → Questions → Verified answers: the questions an admin checked, given again as checked (core2/verified.py).

An admin verifies an answer from the question log, or from the questions asked most that nobody has verified yet,
listed here. A reader who asks the same question again is answered from the verified plan with no AI reading it,
on the data of the day they ask, and the answer says who verified it. A verified answer the data no longer
answers (a metric taken out) is set aside and marked here to look at; the fresh reading of it can be verified in
its place.
"""

from __future__ import annotations

from urllib.parse import quote

from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse

import store
from admin.routes import _is_auth, _resp, router


def _page(account_id: str, error: str = "", saved: str = "") -> RedirectResponse:
    query = f"?error={quote(error)}" if error else f"?saved={quote(saved)}" if saved else ""
    return RedirectResponse(f"/admin/clients/{account_id}/verified{query}", status_code=303)


def _who(request: Request) -> str:
    """The name readers see on a verified answer: the admin's name, never their email address ("" when their
    account has none: the answer then says "an admin", in the reader's language)."""
    from admin.access import identity_of

    identity = identity_of(request)
    return "" if identity is None or identity.emergency else (identity.name or "").strip()


@router.get("/clients/{account_id}/verified", response_class=HTMLResponse)
async def verified_page(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    kept = store.list_verified(account_id)
    for entry in kept:
        fresh = store.latest_answer_to(account_id, entry["question"]) if entry["needs_look"] else ""
        entry["fresh"] = fresh
        entry["fresh_said"] = ((store.core2_answered(account_id, fresh) or {}).get("headline") or "") if fresh else ""
    return _resp(request, "client_verified.html", {
        "client": client, "kept": kept, "offered": store.verified_candidates(account_id),
        "saved": request.query_params.get("saved"), "error": request.query_params.get("error"),
    })


@router.post("/clients/{account_id}/verified")
async def verified_add(request: Request, account_id: str):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    if not store.get_client(account_id):
        return RedirectResponse("/admin/clients", status_code=303)
    form = await request.form()
    answered = store.core2_answered(account_id, str(form.get("question_id") or "").strip())
    if answered is None:
        return _page(account_id, "Only an answer to a question asked on its own can be verified: not a follow-up or "
                                 "a suggestion clicked, not one under a reader's standing condition, and not one "
                                 "whose dates came from words with no year (\"since January\"). Ask it again as a "
                                 "new question, then verify that answer.")
    try:
        store.verify_answer(account_id, answered["question"], answered["plan"], by=_who(request),
                            headline=answered.get("headline") or "")
    except ValueError:
        return _page(account_id, "That question has no words to match a reader's question to.")
    return _page(account_id, saved="verified")


@router.post("/clients/{account_id}/verified/{verified_id}/remove")
async def verified_remove(request: Request, account_id: str, verified_id: int):
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    if not store.get_client(account_id):
        return RedirectResponse("/admin/clients", status_code=303)
    store.remove_verified(account_id, verified_id)
    return _page(account_id, saved="removed")
