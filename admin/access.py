"""
admin/access.py

Who is signed in to the admin console, and what each admin may open.

Every admin page and action is under /admin. Rather than a check in each of the
console's few hundred routes, one middleware (``AdminAccess``) reads the
session, finds the rule for the path, and refuses what the admin's role does
not reach -- a 403 page, a JSON 403 for an API call, a closed socket -- before
any route runs. It records every refusal and every change an admin makes
(store.record_admin_event), so the log names who did what.

The rules (``RULES``) give each path the least role that may open it, and say
whether the path is about one workspace (the account id in it): a workspace
admin or data steward opens only the workspaces it was given. A path no rule
names is for super admins; a new route is closed to the other roles until a
rule opens it.

A session is either a named admin's or the emergency sign-in's (the shared
admin password: a super admin with no name). A named session ends when the
admin's password, role or active state changes; an emergency session ends when
the emergency password changes or emergency sign-in is turned off.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import json
import logging
import re
import secrets
import time
from dataclasses import dataclass, field
from http.cookies import SimpleCookie
from typing import Any, Optional

import store
from admin import credentials as admin_credentials
from core.process_secrets import env_secret_or_random

log = logging.getLogger("querybot.admin.access")

COOKIE = "querybot_session"
SESSION_HOURS = 8
EMERGENCY_KEY = "admin_emergency_sign_in"          # "off" turns the shared password off

SUPER, WORKSPACE, STEWARD = store.SUPER_ADMIN, store.WORKSPACE_ADMIN, store.DATA_STEWARD
RANK = store.ADMIN_ROLE_RANK


@dataclass(frozen=True)
class AdminIdentity:
    admin_id: Optional[int]          # None: the emergency sign-in
    email: str
    name: str
    role: str
    workspaces: frozenset[str] = field(default_factory=frozenset)

    @property
    def emergency(self) -> bool:
        return self.admin_id is None

    @property
    def is_super(self) -> bool:
        return self.role == SUPER

    @property
    def label(self) -> str:
        return "Emergency sign-in" if self.emergency else (self.name or self.email)

    @property
    def role_label(self) -> str:
        return store.ADMIN_ROLE_LABELS.get(self.role, self.role)

    @property
    def actor(self) -> str:
        """How the logs name this admin."""
        return "emergency sign-in" if self.emergency else self.email

    def opens(self, account_id: str) -> bool:
        return self.is_super or account_id in self.workspaces


# ── Sessions ──────────────────────────────────────────────────────────────────

def _secret() -> str:
    # See core/process_secrets.py: a fixed fallback would let anyone who read the source forge a session.
    return env_secret_or_random("ADMIN_SESSION_SECRET", "SESSION_SECRET", purpose="admin_session")


def emergency_enabled() -> bool:
    return str(store.get_system(EMERGENCY_KEY, "") or "").strip().lower() != "off"


def sign_session(admin: Optional[dict] = None, *, issued: Optional[int] = None) -> str:
    """A cookie for one sign-in: a named admin's (``admin``) or the emergency sign-in's (``None``)."""
    stamp = int(time.time()) if issued is None else int(issued)
    if admin is None:
        who, version = 0, admin_credentials.session_version()
    else:
        who, version = int(admin["id"]), int(admin.get("session_version") or 1)
    payload = f"admin:{who}:{version}:{stamp}:{secrets.token_hex(8)}".encode()
    sig = hmac.new(_secret().encode(), payload, hashlib.sha256).hexdigest()
    return f"{base64.urlsafe_b64encode(payload).decode().rstrip('=')}.{sig}"


def read_session(raw: str) -> Optional[AdminIdentity]:
    """The admin a cookie signs in, or None: forged, expired, superseded, deactivated or turned off."""
    if not raw:
        return None
    try:
        token, sig = raw.split(".", 1)
        payload = base64.urlsafe_b64decode((token + "=" * (-len(token) % 4)).encode())
        expected = hmac.new(_secret().encode(), payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        kind, who, version, issued, _nonce = payload.decode().split(":")
        if kind != "admin" or time.time() - int(issued) > SESSION_HOURS * 3600:
            return None
        if int(who) == 0:
            if int(version) != admin_credentials.session_version() or not emergency_enabled():
                return None
            return AdminIdentity(None, "", "", SUPER)
        admin = store.get_admin(int(who))
        if (not admin or not admin.get("is_active") or not admin.get("password_hash")
                or int(admin.get("session_version") or 1) != int(version)):
            return None
        spaces = frozenset() if admin["role"] == SUPER else frozenset(store.admin_workspaces(int(admin["id"])))
        return AdminIdentity(int(admin["id"]), admin["email"], admin.get("name") or "", admin["role"], spaces)
    except Exception:  # noqa: BLE001 - anything unreadable is no session
        return None


def _cookie_from_headers(headers: list[tuple[bytes, bytes]]) -> str:
    for name, value in headers:
        if name.lower() == b"cookie":
            jar = SimpleCookie()
            try:
                jar.load(value.decode("latin-1"))
            except Exception:  # noqa: BLE001
                continue
            if COOKIE in jar:
                return jar[COOKIE].value
    return ""


def identity_of(request: Any) -> Optional[AdminIdentity]:
    """The signed-in admin of a request or socket (read once per request)."""
    state = getattr(request, "state", None)
    cached = getattr(state, "admin_identity", None) if state is not None else None
    if isinstance(cached, AdminIdentity):
        return cached
    if cached is False:
        return None
    cookies = getattr(request, "cookies", None) or {}
    identity = read_session(str(cookies.get(COOKIE, "") or ""))
    try:
        state.admin_identity = identity or False
    except Exception:  # noqa: BLE001 - a state that cannot hold it reads the cookie again next time
        pass
    return identity


# ── Rules ─────────────────────────────────────────────────────────────────────

_A = r"(?P<acct>[^/]+)"
# (path pattern under /admin, least role, about one workspace). First match wins; no match: super admins only.
RULES: list[tuple[re.Pattern, str, bool]] = [(re.compile(p), role, scoped) for p, role, scoped in [
    # Every admin: the home page, the workspace list (theirs only), their own account, the live badge.
    (r"^/?$", STEWARD, False),
    (r"^/clients/?$", STEWARD, False),
    (r"^/me(/.*)?$", STEWARD, False),
    (r"^/ping$", STEWARD, False),
    (r"^/api/status$", STEWARD, False),
    (r"^/ws/notifications$", STEWARD, False),
    (r"^/api/semantic-feedback/pending-count$", STEWARD, False),
    # Making and setting up workspaces, and what changes how a workspace runs: super admins.
    (r"^/clients/(new|create)(/.*)?$", SUPER, False),
    (rf"^/clients/{_A}/(setup|compliance|egress|egress-log|billing|llm-audit\.csv|kb-tables|toggle-chat-ui|reset"
     r"|update|advanced|settings|diagnostics)(/.*)?$", SUPER, True),
    (rf"^/clients/{_A}/learned/(build|engine)$", SUPER, True),
    (rf"^/clients/{_A}/model-health/(compiler/mode|versions/[^/]+/rollback)$", SUPER, True),
    (rf"^/api/clients/{_A}/compliance(/.*)?$", SUPER, True),
    # A workspace's people, groups, table access and admins: workspace admins.
    (rf"^/clients/{_A}/(users|groups|pending-users|admins)(/.*)?$", WORKSPACE, True),
    (rf"^/api/clients/{_A}/pending-users(/.*)?$", WORKSPACE, True),
    # A workspace's data: what QueryBot learned, metrics, dates, relationships, the knowledge base, the readers'
    # requests, the question log and the support pages about them: data stewards.
    (rf"^/clients/{_A}(/(learned|knowledge|relationships|dates|measures|metrics|requests|date-roles|date-contexts"
     r"|kb|glossary|meanings|drafts|learning-queue|semantic-feedback|model-health|mapping|domains|graph|evals|traces"
     r"|queries|readiness|health-score|schema|schema-tree|api|reports|value-index)(/.*)?)?/?$", STEWARD, True),
    (rf"^/api/clients/{_A}/(learning-queue|readiness|metrics|synonym-proposals|domains)(/.*)?$", STEWARD, True),
]]

# Reached without a session: signing in, first setup, accepting an invitation.
PUBLIC = re.compile(r"^/(login|logout|setup|invite/[^/]+)/?$")


@dataclass(frozen=True)
class Decision:
    allowed: bool
    role: str                 # the least role the path needs
    account_id: str = ""      # the workspace the path is about, if any
    reason: str = ""


def rule_for(path: str) -> tuple[str, bool, str]:
    """(least role, about one workspace, that workspace) for a path under /admin."""
    for pattern, role, scoped in RULES:
        found = pattern.match(path)
        if found:
            return role, scoped, (found.groupdict().get("acct") or "") if scoped else ""
    return SUPER, False, ""


def decide(identity: AdminIdentity, path: str) -> Decision:
    """May this admin open this path (under /admin)?"""
    role, scoped, account_id = rule_for(path)
    if identity.is_super:
        return Decision(True, role, account_id)
    if RANK.get(identity.role, 0) < RANK[role]:
        return Decision(False, role, account_id, f"needs {store.ADMIN_ROLE_LABELS[role]}")
    if scoped and account_id not in identity.workspaces:
        return Decision(False, role, account_id, "another workspace")
    return Decision(True, role, account_id)


def can(identity: Optional[AdminIdentity], path: str) -> bool:
    """For pages: whether to show a link (``path`` may include the /admin prefix)."""
    if identity is None:
        return False
    if path.startswith("/admin"):
        path = path[len("/admin"):] or "/"
    return decide(identity, path.split("?", 1)[0]).allowed


def visible_workspaces(identity: Optional[AdminIdentity], rows: list[dict]) -> list[dict]:
    """The workspaces (client rows) an admin may open."""
    if identity is None:
        return []
    return [r for r in rows if identity.opens(str(r.get("account_id")))]


# ── The middleware ────────────────────────────────────────────────────────────

_MUTATING = {"POST", "PUT", "PATCH", "DELETE"}
# Pages that write their own line in the log, naming what changed (admin/admin_accounts.py).
_SELF_LOGGED = re.compile(r"^/(me/(setup|password)|admins(/.*)?|clients/[^/]+/admins/.+)$")
# Checks and previews post a form but change nothing: not worth a line in the log.
_READS = re.compile(r"/(check|check-[a-z-]+|preview|validate|test-formula|test-question|describe|read|simulate"
                    r"|suggest|mask-preview|api/columns|api/check)$")


def _record(identity: Optional[AdminIdentity], action: str, method: str, path: str, account_id: str,
            status: int, detail: str, ip: str) -> None:
    try:
        store.record_admin_event(admin_id=identity.admin_id if identity else None,
                                 admin_email=identity.actor if identity else "", action=action, method=method,
                                 path=path, account_id=account_id, status=status, detail=detail, ip=ip)
    except Exception as exc:  # noqa: BLE001 - the request is answered either way; a lost line is logged
        log.warning("Admin event %s on %s could not be recorded: %s", action, path, exc)


def _forbidden_body(identity: AdminIdentity, decision: Decision) -> bytes:
    where = f" for the {html.escape(decision.account_id)} workspace" if decision.account_id else ""
    need = store.ADMIN_ROLE_LABELS.get(decision.role, decision.role)
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            f"<title>Not available</title><style>body{{font:15px/1.6 system-ui,sans-serif;background:#f3f5f9;"
            f"color:#1b2433;display:grid;place-items:center;min-height:100vh;margin:0}}main{{background:#fff;"
            f"border:1px solid #e1e6ee;border-radius:12px;padding:28px 32px;max-width:460px}}a{{color:#2563eb}}"
            f"</style></head><body><main><h1 style='font-size:20px;margin:0 0 8px'>This page is not open to you</h1>"
            f"<p>You are signed in as {html.escape(identity.label)} ({html.escape(identity.role_label)}). This page "
            f"needs {html.escape(need)}{where}. A super admin can change your role or workspaces.</p>"
            f"<p><a href='/admin'>Back to your workspaces</a></p></main></body></html>").encode()


class AdminAccess:
    """Refuses what an admin's role does not reach, and logs refusals and changes. See the module docstring."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        kind = scope.get("type")
        path_all = str(scope.get("path") or "")
        if kind not in ("http", "websocket") or not (path_all == "/admin" or path_all.startswith("/admin/")):
            await self.app(scope, receive, send)
            return
        path = path_all[len("/admin"):] or "/"
        if PUBLIC.match(path):
            await self.app(scope, receive, send)
            return
        identity = read_session(_cookie_from_headers(scope.get("headers") or []))
        if identity is None:
            await self.app(scope, receive, send)        # the route sends them to sign in, as before
            return
        scope.setdefault("state", {})["admin_identity"] = identity
        method = str(scope.get("method") or ("WS" if kind == "websocket" else "GET")).upper()
        client = scope.get("client") or ("", 0)
        ip = str(client[0] if client else "")
        decision = decide(identity, path)
        if not decision.allowed:
            _record(identity, "refused", method, path_all, decision.account_id, 403, decision.reason, ip)
            if kind == "websocket":
                await send({"type": "websocket.close", "code": 4403})
                return
            accept = b""
            for name, value in scope.get("headers") or []:
                if name.lower() == b"accept":
                    accept = value
            wants_json = b"application/json" in accept or "/api/" in path
            body = (json.dumps({"error": "forbidden", "detail": "Your admin role does not reach this."}).encode()
                    if wants_json else _forbidden_body(identity, decision))
            await send({"type": "http.response.start", "status": 403, "headers": [
                (b"content-type", b"application/json" if wants_json else b"text/html; charset=utf-8"),
                (b"cache-control", b"no-store")]})
            await send({"type": "http.response.body", "body": body})
            return
        if kind != "http" or method not in _MUTATING or _READS.search(path) or _SELF_LOGGED.match(path):
            await self.app(scope, receive, send)
            return
        status = {"code": 0}

        async def watching(message: dict) -> None:
            if message.get("type") == "http.response.start":
                status["code"] = int(message.get("status") or 0)
            await send(message)

        try:
            await self.app(scope, receive, watching)
        finally:
            _record(identity, "change", method, path_all, decision.account_id, status["code"], "", ip)
