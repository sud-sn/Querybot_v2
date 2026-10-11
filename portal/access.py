"""
portal/access.py

What each reader may open in the portal, by their portal role (core/portal_roles.py), checked in one place.

``PortalRoles`` is a middleware in front of every /portal page and API and the chat socket (/ws/chat/...). It
reads the reader's session, finds the action the path needs (``RULES``; first match wins), and refuses what
their role does not reach: a page sends them to their home, an API call gets a 403 that says why in their
language, the chat socket closes. A path no rule names is open to every signed-in reader (their dashboards,
follows, settings, password, notices); which rows and tables they see is decided elsewhere and never widened
here. A request with no session goes on to the route, which sends it to sign in as before.

The chat itself checks again for what is asked inside it: a dashboard request or a change by talking needs a
creator (gateway/core2_bridge.py), and a question on a chat platform needs an asker (core/dispatcher.py).
"""

from __future__ import annotations

import json
import logging
import re
from http.cookies import SimpleCookie
from typing import Any, Optional

from core.portal_roles import can, home_path, refusal

log = logging.getLogger("querybot.portal.access")

_ID = r"[^/]+"
# (methods or None for any, path under /portal or the full socket path, action). First match wins.
RULES: list[tuple[Optional[frozenset[str]], re.Pattern, str]] = [
    (frozenset(m.split()) if m else None, re.compile(p), action) for m, p, action in [
        # Asking: the chat, the data guide, history, suggestions as you type, answer feedback.
        ("GET", r"^/chat/?$", "ask"),
        ("GET", r"^/kb/?$", "ask"),
        ("POST", r"^/kb/feedback$", "ask"),
        ("GET", rf"^/api/history(/{_ID})?$", "ask"),
        ("GET", r"^/api/suggest$", "ask"),
        ("POST", r"^/api/suggestions/event$", "ask"),
        ("GET", r"^/api/query-limit-status$", "ask"),
        ("GET", r"^/api/semantic-feedback/updates$", "ask"),
        ("POST", rf"^/api/answers/{_ID}/feedback$", "ask"),
        ("", r"^/ws/chat/[^/]+$", "ask"),
        # An answer's rows.
        ("GET", r"^/api/export-csv$", "export"),
        # Suggesting metrics and fixes to the admins.
        ("", r"^/api/requests(/.*)?$", "suggest"),
        ("POST", r"^/api/own-metrics/forget$", "suggest"),
        # Answers kept on dashboards of their own, and arranging those.
        ("", r"^/pin-confirm$", "save_answer"),
        ("POST", r"^/api/pin-chart$", "save_answer"),
        ("POST", r"^/unpin$", "save_answer"),
        ("", r"^/api/dashboards$", "own_dashboards"),
        ("POST", r"^/api/update-chart(-layout)?$", "own_dashboards"),
        ("POST", rf"^/api/dashboard/{_ID}/tidy$", "own_dashboards"),
        ("POST", rf"^/dashboard/{_ID}/rollback$", "own_dashboards"),
        # Sharing and publishing, and looking people up to share with.
        ("", rf"^/api/dashboard/{_ID}/shares(/.*)?$", "share"),
        ("POST", rf"^/dashboard/{_ID}/(share|publish)$", "share"),
        ("GET", r"^/api/people$", "share"),
        # Scheduled reports, and dashboards emailed on a schedule.
        ("", r"^/reports(/.*)?$", "reports"),
        ("", rf"^/api/dashboard/{_ID}/schedules(/.*)?$", "reports"),
        # Alerts on a dashboard's number tile: anyone who may open it, for themselves (others: its owner, checked
        # by the route).
        ("", rf"^/api/dashboard/{_ID}/tiles/{_ID}/alerts(/.*)?$", "follow"),
    ]
]


def action_for(method: str, path: str) -> Optional[str]:
    """The action ``path`` needs (under /portal, or a socket's full path), or None: any signed-in reader."""
    for methods, pattern, action in RULES:
        if (methods is None or method in methods) and pattern.match(path):
            return action
    return None


def _cookie(headers: list[tuple[bytes, bytes]], name: str) -> str:
    for key, value in headers:
        if key.lower() == b"cookie":
            jar = SimpleCookie()
            try:
                jar.load(value.decode("latin-1"))
            except Exception:  # noqa: BLE001
                continue
            if name in jar:
                return jar[name].value
    return ""


class PortalRoles:
    """Refuses what a reader's portal role does not reach. See the module docstring."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        kind = scope.get("type")
        full = str(scope.get("path") or "")
        if kind == "websocket" and full.startswith("/ws/chat/"):
            path = full
        elif kind in ("http", "websocket") and full.startswith("/portal/"):
            path = full[len("/portal"):]
        else:
            await self.app(scope, receive, send)
            return
        method = "" if kind == "websocket" else str(scope.get("method") or "GET").upper()
        action = action_for(method, path)
        if action is None:
            await self.app(scope, receive, send)
            return
        from portal import routes as portal_routes   # the session rules live with the portal's pages

        user = portal_routes._session_user(_cookie(scope.get("headers") or [], portal_routes._COOKIE))
        if user is None or can(user, action):
            await self.app(scope, receive, send)        # no session: the route sends them to sign in
            return
        log.info("Portal role %s refused %s %s for user %s", user.get("portal_role"), method or "WS", full,
                 user.get("id"))
        if kind == "websocket":
            await send({"type": "websocket.close", "code": 4403})
            return
        if method == "GET" and "/api/" not in path:
            await send({"type": "http.response.start", "status": 303, "headers": [
                (b"location", home_path(user).encode()), (b"cache-control", b"no-store")]})
            await send({"type": "http.response.body", "body": b""})
            return
        body = json.dumps({"ok": False, "error": "not_allowed", "action": action,
                           "detail": refusal(action, user.get("lang"))}).encode()
        await send({"type": "http.response.start", "status": 403, "headers": [
            (b"content-type", b"application/json"), (b"cache-control", b"no-store")]})
        await send({"type": "http.response.body", "body": body})
