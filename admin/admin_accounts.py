"""
admin/admin_accounts.py

Signing in to the admin console as a named admin, and managing admins.

  /admin/login               email and password; or the emergency admin password (the shared one every
                             server had before named admins), while it is on
  /admin/me/setup            after an emergency sign-in: make a named super admin account (the first one on an
                             upgraded server; the page an emergency sign-in lands on until one exists)
  /admin/invite/{token}      an invited admin chooses a password (a one-time link, INVITE_HOURS long)
  /admin/me                  your own account: name, password, workspaces, recent actions
  /admin/admins              super admins: every admin, invite one, change role, workspaces, active
  /admin/clients/{id}/admins workspace admins: the admins of their workspace; invite or remove data stewards

Which pages each role opens is admin/access.py. Every sign-in, refusal and change is in the admin event log.
"""

from __future__ import annotations

import logging
from urllib.parse import quote

from fastapi import Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

import store
from admin import access as admin_access
from admin import credentials as admin_credentials
from admin import routes as _routes
from store.admin_store import MIN_PASSWORD, normalise_email

log = logging.getLogger("querybot.admin.accounts")

router = _routes.router


# Through the module, so a test that stands in for one of these reaches this file too.
def _resp(request, name, ctx=None):
    return _routes._resp(request, name, ctx)


def _me(request):
    return _routes._me(request)


def _set_admin_cookie(resp, request, admin=None):
    _routes._set_admin_cookie(resp, request, admin)


def _ip(request: Request) -> str:
    client = getattr(request, "client", None)
    return str(getattr(client, "host", "") or "") if client else ""


def _event(request: Request, action: str, *, admin: dict | None = None, detail: str = "", account_id: str = "",
           identity: "admin_access.AdminIdentity | None" = None) -> None:
    who = identity or _me(request)
    try:
        store.record_admin_event(
            admin_id=(admin or {}).get("id") if admin else (who.admin_id if who else None),
            admin_email=(admin or {}).get("email") if admin else (who.actor if who else ""),
            action=action, method=request.method, path=str(request.url.path), account_id=account_id,
            status=0, detail=detail, ip=_ip(request))
    except Exception as exc:  # noqa: BLE001 - logged; the sign-in or change stands
        log.warning("Admin event %s could not be recorded: %s", action, exc)


def _login_page(request: Request, *, error: str | None = None, emergency: bool | None = None, email: str = "",
                status: int = 200):
    accounts = store.any_admin_accounts()
    # The emergency sign-in exists only where a shared password was set, and only while it is on.
    available = admin_access.emergency_enabled() and admin_credentials.is_set()
    if emergency is None:
        emergency = str(request.query_params.get("emergency") or "") == "1" or not accounts
    context = {"error": error, "emergency": bool(emergency) and available, "emergency_on": available,
               "accounts": accounts, "email": email}
    if status == 200:
        return _resp(request, "login.html", context)
    return _routes.templates.TemplateResponse(request=request, name="login.html", status_code=status,
                                              context=context)


def _too_many(request: Request, wait: int, *, emergency: bool, email: str = ""):
    response = _login_page(request, error=f"Too many attempts. Wait {max(1, -(-wait // 60))} min and try again.",
                           emergency=emergency, email=email, status=429)
    response.headers["Retry-After"] = str(wait)
    return response


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return _login_page(request)


@router.post("/login")
async def login_submit(request: Request, password: str = Form(...), email: str = Form("")):
    if _routes._first_run():
        return RedirectResponse("/admin/setup", status_code=303)
    email = normalise_email(email if isinstance(email, str) else "")
    host = request.client.host if request.client else ""
    if email:
        # A named admin: attempts are counted per email address and client address.
        identity = store.sign_in_identity(f"admin:{email}", host)
        wait = store.sign_in_wait_seconds(identity)
        admin = None if wait else store.get_admin_by_email(email)
        if not wait and not (admin and admin.get("is_active") and store.verify_admin_password(admin, password)):
            wait = store.record_sign_in_failure(identity, "admin")
            if not wait:
                return _login_page(request, error="That email and password do not match an active admin.",
                                   emergency=False, email=email)
        if wait:
            return _too_many(request, wait, emergency=False, email=email)
        store.clear_sign_in_failures(identity)
        store.upgrade_admin_hash(admin, password)
        store.mark_admin_signed_in(int(admin["id"]))
        admin = store.get_admin(int(admin["id"])) or admin
        _event(request, "sign_in", admin=admin)
        resp = RedirectResponse("/admin", status_code=303)
        _set_admin_cookie(resp, request, admin)
        return resp
    # The emergency admin password (the one every server had before named admins).
    if not admin_access.emergency_enabled():
        return _login_page(request, error="Emergency sign-in is turned off. Sign in with your email and password.",
                           emergency=False)
    if not admin_credentials.is_set():
        return _login_page(request, error="This server has no emergency admin password: sign in with your email.",
                           emergency=False)
    if not admin_credentials.is_readable():
        return _login_page(request, error=_routes._UNREADABLE_ADMIN_PASSWORD, emergency=True)
    # One password and no user name, so attempts are counted per client
    # address. Behind a reverse proxy that is the proxy's address unless
    # uvicorn is told to trust its forwarded headers (docs/UPGRADE_RUNBOOK.md).
    identity = store.sign_in_identity("admin", host)
    wait = store.sign_in_wait_seconds(identity)
    if not wait and not admin_credentials.verify(password):
        wait = store.record_sign_in_failure(identity, "admin")
        if not wait:
            return _login_page(request, error="Incorrect password", emergency=True)
    if wait:
        return _too_many(request, wait, emergency=True)
    store.clear_sign_in_failures(identity)
    admin_credentials.upgrade_hash(password)
    _event(request, "emergency_sign_in", identity=admin_access.AdminIdentity(None, "", "", store.SUPER_ADMIN))
    # Until a named super admin can sign in, the emergency sign-in is for making one.
    target = "/admin" if store.count_active_super_admins() else "/admin/me/setup"
    resp = RedirectResponse(target, status_code=303)
    _set_admin_cookie(resp, request)
    return resp


@router.get("/logout")
async def logout(request: Request):
    who = _me(request)
    if who is not None:
        _event(request, "sign_out", identity=who)
    resp = RedirectResponse("/admin/login", status_code=303)
    resp.delete_cookie(admin_access.COOKIE)
    return resp


# ── The first named super admin, from the emergency sign-in ────────────────────

def _setup_page(request: Request, error: str | None = None, name: str = "", email: str = ""):
    return _resp(request, "admin_account_setup.html", {
        "error": error, "name": name, "email": email, "min_password": MIN_PASSWORD,
        "first": not store.count_active_super_admins()})


@router.get("/me/setup", response_class=HTMLResponse)
async def account_setup_page(request: Request):
    who = _me(request)
    if who is None:
        return RedirectResponse("/admin/login", status_code=303)
    if not who.emergency:
        return RedirectResponse("/admin/me", status_code=303)
    return _setup_page(request)


@router.post("/me/setup")
async def account_setup_submit(request: Request, name: str = Form(""), email: str = Form(""),
                               password: str = Form(""), confirm_password: str = Form("")):
    who = _me(request)
    if who is None:
        return RedirectResponse("/admin/login", status_code=303)
    if not who.emergency:
        return RedirectResponse("/admin/me", status_code=303)
    email = normalise_email(email)
    if not name.strip() or "@" not in email:
        return _setup_page(request, "Give your name and your email address.", name, email)
    if len(password) < MIN_PASSWORD:
        return _setup_page(request, f"Choose a password of at least {MIN_PASSWORD} characters.", name, email)
    if password != confirm_password:
        return _setup_page(request, "The two passwords are not the same.", name, email)
    if store.get_admin_by_email(email):
        return _setup_page(request, "An admin with this email exists already: sign in with it.", name, email)
    admin_id = store.create_admin(email, name.strip(), store.SUPER_ADMIN, password=password,
                                  created_by="emergency sign-in")
    admin = store.get_admin(admin_id)
    _event(request, "admin_created", detail=f"{email} as Super admin", identity=who)
    store.mark_admin_signed_in(admin_id)
    _event(request, "sign_in", admin=admin)
    resp = RedirectResponse("/admin?welcome=1", status_code=303)
    _set_admin_cookie(resp, request, admin)
    return resp


# ── Invitations ──────────────────────────────────────────────────────────────

def _invite_link(request: Request, token: str) -> str:
    return f"{str(request.base_url).rstrip('/')}/admin/invite/{token}"


async def _mail_invite(admin: dict, link: str, who) -> str:
    """The invitation by email when email is set up: "sent", why it was not, or "" (email is off)."""
    import asyncio

    from core.mail import outbox, render
    from core.mail.settings import configured

    if not configured():
        return ""
    title = "You are invited to the QueryBot admin panel"
    inviter = who.label if who is not None and not who.emergency else "A Super admin"
    paragraphs = [f"{inviter} gave you the {admin.get('role_label') or 'admin'} role in QueryBot.",
                  f"Choose your password with the button below. The link works once, for {store.admin_store.INVITE_HOURS} "
                  "hours; then you sign in with this email address."]
    button = ("Choose my password", link)
    email_id = outbox.queue(kind="admin_invite", to=str(admin["email"]), subject=title,
                            text=render.text(title, paragraphs, button=button,
                                             why="Sent because an admin invited you. Not expecting it? Ignore it."),
                            html=render.page(title, paragraphs, button=button,
                                             why="Sent because an admin invited you. Not expecting it? Ignore it."))
    sent = await asyncio.to_thread(outbox.send_now, email_id) if email_id else {}
    if sent.get("status") == "sent":
        return "sent"
    return str(sent.get("last_error") or "it could not be sent")


@router.get("/invite/{token}", response_class=HTMLResponse)
async def invite_page(request: Request, token: str):
    admin = store.admin_for_invite(token)
    return _resp(request, "admin_invite.html", {"admin": admin, "token": token, "error": None,
                                                "min_password": MIN_PASSWORD})


@router.post("/invite/{token}")
async def invite_submit(request: Request, token: str, name: str = Form(""), password: str = Form(""),
                        confirm_password: str = Form("")):
    admin = store.admin_for_invite(token)
    if admin is None:
        return _resp(request, "admin_invite.html", {"admin": None, "token": token, "error": None,
                                                    "min_password": MIN_PASSWORD})

    def again(error: str):
        return _resp(request, "admin_invite.html", {"admin": admin, "token": token, "error": error,
                                                    "min_password": MIN_PASSWORD})
    if len(password) < MIN_PASSWORD:
        return again(f"Choose a password of at least {MIN_PASSWORD} characters.")
    if password != confirm_password:
        return again("The two passwords are not the same.")
    if name.strip() and name.strip() != admin.get("name"):
        store.update_admin(int(admin["id"]), name=name.strip())
    store.set_admin_password(int(admin["id"]), password)
    store.mark_admin_signed_in(int(admin["id"]))
    admin = store.get_admin(int(admin["id"]))
    _event(request, "invite_accepted", admin=admin)
    resp = RedirectResponse("/admin", status_code=303)
    _set_admin_cookie(resp, request, admin)
    return resp


# ── Your own account ─────────────────────────────────────────────────────────

@router.get("/me", response_class=HTMLResponse)
async def my_account(request: Request):
    who = _me(request)
    if who is None:
        return RedirectResponse("/admin/login", status_code=303)
    admin = store.get_admin(who.admin_id) if who.admin_id else None
    names = {c["account_id"]: c.get("client_name") or c["account_id"] for c in store.list_clients()}
    return _resp(request, "admin_me.html", {
        "me": who, "admin": admin, "saved": request.query_params.get("saved"),
        "error": request.query_params.get("error"), "min_password": MIN_PASSWORD,
        "workspaces": [(a, names.get(a, a)) for a in sorted(who.workspaces)],
        "events": store.list_admin_events(admin_id=who.admin_id, limit=15) if who.admin_id else [],
    })


@router.post("/me/password")
async def my_password(request: Request, current_password: str = Form(""), password: str = Form(""),
                      confirm_password: str = Form("")):
    who = _me(request)
    if who is None or who.emergency:
        return RedirectResponse("/admin/login", status_code=303)
    admin = store.get_admin(who.admin_id)
    if not admin or not store.verify_admin_password(admin, current_password):
        return RedirectResponse("/admin/me?error=" + quote("Your current password is not right."), status_code=303)
    if len(password) < MIN_PASSWORD or password != confirm_password:
        return RedirectResponse("/admin/me?error=" + quote(
            f"The new password needs {MIN_PASSWORD} characters or more, typed the same twice."), status_code=303)
    store.set_admin_password(who.admin_id, password)
    _event(request, "password_changed", identity=who)
    resp = RedirectResponse("/admin/me?saved=password", status_code=303)
    _set_admin_cookie(resp, request, store.get_admin(who.admin_id))     # the other sessions end; this one goes on
    return resp


@router.post("/me/name")
async def my_name(request: Request, name: str = Form("")):
    who = _me(request)
    if who is None or who.emergency:
        return RedirectResponse("/admin/login", status_code=303)
    if name.strip():
        store.update_admin(who.admin_id, name=name.strip())
    return RedirectResponse("/admin/me?saved=name", status_code=303)


# ── Admins (super admins) ────────────────────────────────────────────────────

def _admins_page(request: Request, *, invite: tuple[str, str] | None = None, error: str | None = None):
    who = _me(request)
    clients = sorted(store.list_clients(), key=lambda c: (c.get("client_name") or c["account_id"]).casefold())
    return _resp(request, "admin_admins.html", {
        "me": who, "admins": store.list_admins(), "clients": clients,
        "names": {c["account_id"]: c.get("client_name") or c["account_id"] for c in clients},
        "roles": [(r, store.ADMIN_ROLE_LABELS[r]) for r in store.ADMIN_ROLES],
        "invite": invite, "error": error or request.query_params.get("error"),
        "saved": request.query_params.get("saved"), "events": store.list_admin_events(limit=25),
        "emergency_on": admin_access.emergency_enabled(),
    })


def _super_only(request: Request):
    who = _me(request)
    if who is None:
        return None, RedirectResponse("/admin/login", status_code=303)
    if not who.is_super:
        return None, RedirectResponse("/admin", status_code=303)
    return who, None


@router.get("/admins", response_class=HTMLResponse)
async def admins_page(request: Request):
    who, stop = _super_only(request)
    if stop:
        return stop
    return _admins_page(request)


@router.post("/admins/invite")
async def admins_invite(request: Request):
    who, stop = _super_only(request)
    if stop:
        return stop
    form = await request.form()
    email = normalise_email(str(form.get("email") or ""))
    name = str(form.get("name") or "").strip()
    role = str(form.get("role") or store.DATA_STEWARD)
    workspaces = [str(w) for w in form.getlist("workspaces")]
    if role not in store.ADMIN_ROLES:
        return _admins_page(request, error="Choose a role.")
    if "@" not in email:
        return _admins_page(request, error="Give the new admin's email address.")
    if store.get_admin_by_email(email):
        return _admins_page(request, error="An admin with this email exists already.")
    if role != store.SUPER_ADMIN and not workspaces:
        return _admins_page(request, error="Choose the workspaces this admin looks after.")
    known = {c["account_id"] for c in store.list_clients()}
    admin_id = store.create_admin(email, name, role, workspaces=[w for w in workspaces if w in known],
                                  created_by=who.actor)
    token = store.issue_admin_invite(admin_id)
    _event(request, "admin_invited", detail=f"{email} as {store.ADMIN_ROLE_LABELS[role]}", identity=who)
    link = _invite_link(request, token)
    return _admins_page(request, invite=(email, link, await _mail_invite(store.get_admin(admin_id), link, who)))


def _last_super(admin: dict) -> bool:
    return admin["role"] == store.SUPER_ADMIN and admin.get("is_active") and store.count_active_super_admins() <= 1


@router.post("/admins/{admin_id}/update")
async def admins_update(request: Request, admin_id: int):
    who, stop = _super_only(request)
    if stop:
        return stop
    admin = store.get_admin(admin_id)
    if admin is None:
        return RedirectResponse("/admin/admins?error=" + quote("No such admin."), status_code=303)
    form = await request.form()
    role = str(form.get("role") or admin["role"])
    active = str(form.get("is_active") or ("1" if admin.get("is_active") else "0")) == "1"
    workspaces = [str(w) for w in form.getlist("workspaces")]
    if role not in store.ADMIN_ROLES:
        return RedirectResponse("/admin/admins?error=" + quote("Choose a role."), status_code=303)
    if _last_super(admin) and (role != store.SUPER_ADMIN or not active):
        return RedirectResponse("/admin/admins?error=" + quote(
            "This is the only active super admin: make another one first."), status_code=303)
    if role != store.SUPER_ADMIN and not workspaces:
        return RedirectResponse("/admin/admins?error=" + quote(
            "An admin who is not a super admin needs at least one workspace."), status_code=303)
    changes = []
    if role != admin["role"]:
        changes.append(f"role {store.ADMIN_ROLE_LABELS[admin['role']]} to {store.ADMIN_ROLE_LABELS[role]}")
    if active != bool(admin.get("is_active")):
        changes.append("reactivated" if active else "deactivated")
    store.update_admin(admin_id, name=str(form.get("name") or admin.get("name") or ""),
                       role=role if role != admin["role"] else None,
                       is_active=active if active != bool(admin.get("is_active")) else None)
    known = {c["account_id"] for c in store.list_clients()}
    before = set(store.admin_workspaces(admin_id))
    after = set() if role == store.SUPER_ADMIN else {w for w in workspaces if w in known}
    if before != after:
        store.set_admin_workspaces(admin_id, sorted(after))
        changes.append("workspaces " + (", ".join(sorted(after)) or "none"))
    if changes:
        _event(request, "admin_changed", detail=f"{admin['email']}: " + "; ".join(changes), identity=who)
    return RedirectResponse("/admin/admins?saved=1", status_code=303)


@router.post("/admins/{admin_id}/invite")
async def admins_reinvite(request: Request, admin_id: int):
    """A new one-time link: to choose a password, or a new one when it is forgotten."""
    who, stop = _super_only(request)
    if stop:
        return stop
    admin = store.get_admin(admin_id)
    if admin is None:
        return RedirectResponse("/admin/admins?error=" + quote("No such admin."), status_code=303)
    token = store.issue_admin_invite(admin_id)
    _event(request, "admin_link_issued", detail=admin["email"], identity=who)
    link = _invite_link(request, token)
    return _admins_page(request, invite=(admin["email"], link, await _mail_invite(admin, link, who)))


@router.post("/admins/{admin_id}/delete")
async def admins_delete(request: Request, admin_id: int):
    who, stop = _super_only(request)
    if stop:
        return stop
    admin = store.get_admin(admin_id)
    if admin is None:
        return RedirectResponse("/admin/admins", status_code=303)
    if _last_super(admin):
        return RedirectResponse("/admin/admins?error=" + quote(
            "This is the only active super admin: make another one first."), status_code=303)
    if who.admin_id == admin_id:
        return RedirectResponse("/admin/admins?error=" + quote("You cannot remove your own account."),
                                status_code=303)
    store.delete_admin(admin_id)
    _event(request, "admin_removed", detail=admin["email"], identity=who)
    return RedirectResponse("/admin/admins?saved=1", status_code=303)


@router.post("/admins/emergency")
async def admins_emergency(request: Request, state: str = Form("")):
    """Turn the emergency admin password off or on. Off needs a named super admin signed in as themselves."""
    who, stop = _super_only(request)
    if stop:
        return stop
    if state == "off":
        if who.emergency:
            return RedirectResponse("/admin/admins?error=" + quote(
                "Sign in with your own account to turn emergency sign-in off."), status_code=303)
        store.set_system(admin_access.EMERGENCY_KEY, "off")
        _event(request, "emergency_off", identity=who)
    else:
        store.set_system(admin_access.EMERGENCY_KEY, "on")
        _event(request, "emergency_on", identity=who)
    return RedirectResponse("/admin/admins?saved=1", status_code=303)


# ── A workspace's admins (workspace admins and super admins) ─────────────────

def _workspace_admins_page(request: Request, account_id: str, *, invite: tuple[str, str] | None = None,
                           error: str | None = None):
    client = store.get_client(account_id)
    who = _me(request)
    return _resp(request, "client_admins.html", {
        "client": client, "me": who, "admins": store.list_admins(account_id=account_id),
        "invite": invite, "error": error or request.query_params.get("error"),
        "saved": request.query_params.get("saved"),
        "events": store.list_admin_events(account_id=account_id, limit=25),
        "grantable": [(r, store.ADMIN_ROLE_LABELS[r]) for r in
                      ((store.DATA_STEWARD, store.WORKSPACE_ADMIN) if who and who.is_super else (store.DATA_STEWARD,))],
        "pending_count": store.get_pending_user_count(account_id),
    })


@router.get("/clients/{account_id}/admins", response_class=HTMLResponse)
async def workspace_admins(request: Request, account_id: str):
    who = _me(request)
    if who is None or not who.opens(account_id):
        return RedirectResponse("/admin/login", status_code=303)
    if not store.get_client(account_id):
        return RedirectResponse("/admin/clients", status_code=303)
    return _workspace_admins_page(request, account_id)


@router.post("/clients/{account_id}/admins/add")
async def workspace_admins_add(request: Request, account_id: str, email: str = Form(""), name: str = Form(""),
                               role: str = Form(store.DATA_STEWARD)):
    """Give someone this workspace: an existing admin of a lower role, or a new data steward by invitation."""
    who = _me(request)
    if who is None or not who.opens(account_id) or not store.get_client(account_id):
        return RedirectResponse("/admin/login", status_code=303)
    email = normalise_email(email)
    allowed = {store.DATA_STEWARD, store.WORKSPACE_ADMIN} if who.is_super else {store.DATA_STEWARD}
    if role not in allowed:
        return _workspace_admins_page(request, account_id, error="Your role cannot grant that.")
    if "@" not in email:
        return _workspace_admins_page(request, account_id, error="Give the admin's email address.")
    admin = store.get_admin_by_email(email)
    if admin is not None:
        if admin["role"] == store.SUPER_ADMIN:
            return _workspace_admins_page(request, account_id, error=f"{email} is a super admin: every workspace "
                                                                     "is open to them already.")
        if store.ADMIN_ROLE_RANK[admin["role"]] > store.ADMIN_ROLE_RANK[role] and not who.is_super:
            return _workspace_admins_page(request, account_id, error=f"{email} is a {admin['role_label']}: only a "
                                                                     "super admin can give them a workspace.")
        store.add_admin_workspace(int(admin["id"]), account_id)
        _event(request, "admin_workspace_added", detail=email, account_id=account_id, identity=who)
        return RedirectResponse(f"/admin/clients/{account_id}/admins?saved=1", status_code=303)
    admin_id = store.create_admin(email, name.strip(), role, workspaces=[account_id], created_by=who.actor)
    token = store.issue_admin_invite(admin_id)
    _event(request, "admin_invited", detail=f"{email} as {store.ADMIN_ROLE_LABELS[role]}", account_id=account_id,
           identity=who)
    link = _invite_link(request, token)
    return _workspace_admins_page(request, account_id,
                                  invite=(email, link, await _mail_invite(store.get_admin(admin_id), link, who)))


@router.post("/clients/{account_id}/admins/{admin_id}/remove")
async def workspace_admins_remove(request: Request, account_id: str, admin_id: int):
    """Take this workspace away from an admin (a workspace admin may do so for data stewards only)."""
    who = _me(request)
    if who is None or not who.opens(account_id):
        return RedirectResponse("/admin/login", status_code=303)
    admin = store.get_admin(admin_id)
    if admin is None or admin["role"] == store.SUPER_ADMIN:
        return RedirectResponse(f"/admin/clients/{account_id}/admins", status_code=303)
    if not who.is_super and admin["role"] != store.DATA_STEWARD:
        return _workspace_admins_page(request, account_id,
                                      error="Only a super admin can take a workspace from a workspace admin.")
    store.remove_admin_workspace(admin_id, account_id)
    _event(request, "admin_workspace_removed", detail=admin["email"], account_id=account_id, identity=who)
    return RedirectResponse(f"/admin/clients/{account_id}/admins?saved=1", status_code=303)
