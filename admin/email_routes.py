"""
admin/email_routes.py

Admin › System › Email (Super admins): how this server sends email, a test email, and the delivery log.

    /admin/system/email                 settings and the log
    /admin/system/email/save            keep the settings (a secret left blank keeps the one saved)
    /admin/system/email/test            send a test email at once and say what happened
    /admin/system/email/{id}/retry      send a failed or skipped email again
    /admin/system/email/{id}            one email: to, subject, status, why, its text

No rule in admin/access.py names these pages, so they are for Super admins only.
"""

from __future__ import annotations

import logging
from urllib.parse import quote

from fastapi import Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from admin import routes as _routes
from core.mail import outbox, render
from core.mail import settings as mail_settings
from core.mail.transport import describe, forget_tokens
from store import email_store

log = logging.getLogger("querybot.admin.email")
router = _routes.router


def _page(request: Request, *, tested: dict | None = None):
    config = mail_settings.load()
    status = request.query_params.get("status") or ""
    kind = request.query_params.get("kind") or ""
    me = _routes._me(request)
    return _routes._resp(request, "system_email.html", {
        "config": config, "describe": describe(config), "missing": config.missing(),
        "has_graph_secret": bool(config.graph_client_secret), "has_smtp_password": bool(config.smtp_password),
        "emails": email_store.list_emails(status=status or None, kind=kind or None, limit=100),
        "counts": email_store.email_counts(), "status": status, "kind": kind,
        "saved": request.query_params.get("saved"), "error": request.query_params.get("error"),
        "tested": tested, "test_to": (me.email if me and not me.emergency else ""),
        "base_url": outbox.portal_base(),
    })


def _auth(request: Request):
    if not _routes._is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    return None


@router.get("/system/email", response_class=HTMLResponse)
async def email_page(request: Request):
    if (stop := _auth(request)) is not None:
        return stop
    return _page(request)


@router.post("/system/email/save")
async def email_save(request: Request):
    if (stop := _auth(request)) is not None:
        return stop
    form = await request.form()
    values = {name: str(form.get(name) or "") for name in mail_settings.KEYS if name in form}
    provider = values.get("provider", "")
    if provider and provider not in mail_settings.PROVIDERS:
        return RedirectResponse("/admin/system/email?error=" + quote("Choose Microsoft 365 or SMTP."), status_code=303)
    port = values.get("smtp_port", "")
    if port and (not port.isdigit() or not 0 < int(port) < 65536):
        return RedirectResponse("/admin/system/email?error=" + quote("The SMTP port is a number."), status_code=303)
    if form.get("clear_graph_secret"):
        values["graph_client_secret"] = ""
        _routes.store.set_system(mail_settings.KEYS["graph_client_secret"], "")
    if form.get("clear_smtp_password"):
        _routes.store.set_system(mail_settings.KEYS["smtp_password"], "")
    mail_settings.save(values)
    forget_tokens()
    log.info("Email settings saved: %s", describe(mail_settings.load()))
    return RedirectResponse("/admin/system/email?saved=1", status_code=303)


@router.post("/system/email/test")
async def email_test(request: Request, to: str = Form("")):
    if (stop := _auth(request)) is not None:
        return stop
    to = to.strip()
    if "@" not in to:
        return RedirectResponse("/admin/system/email?error=" + quote("Give an address to send the test to."),
                                status_code=303)
    config = mail_settings.load()
    title = "Solviz can send email"
    paragraphs = [f"This test was sent through {describe(config)}.",
                  "Follows, scheduled dashboards, alerts and invitations will come the same way."]
    email_id = outbox.queue(kind="test", to=to, subject=title,
                            text=render.text(title, paragraphs, why="Sent from Admin › System › Email."),
                            html=render.page(title, paragraphs, why="Sent from Admin › System › Email."))
    import asyncio

    sent = await asyncio.to_thread(outbox.send_now, email_id) if email_id else {}
    return _page(request, tested=sent or email_store.get_email(email_id) or {})


@router.post("/system/email/{email_id}/retry")
async def email_retry(request: Request, email_id: int):
    if (stop := _auth(request)) is not None:
        return stop
    if not email_store.requeue_email(email_id):
        return RedirectResponse("/admin/system/email?error=" + quote("Only a failed or skipped email can be sent again."),
                                status_code=303)
    import asyncio

    await asyncio.to_thread(outbox.send_now, email_id)
    return RedirectResponse(f"/admin/system/email/{email_id}", status_code=303)


@router.get("/system/email/{email_id}", response_class=HTMLResponse)
async def email_one(request: Request, email_id: int):
    if (stop := _auth(request)) is not None:
        return stop
    email = email_store.get_email(email_id)
    if email is None:
        return RedirectResponse("/admin/system/email", status_code=303)
    return _routes._resp(request, "system_email_one.html", {"email": email})
