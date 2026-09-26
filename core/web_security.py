"""
core/web_security.py

Requests that carry a session cookie are refused when a browser says they came
from another site, and session cookies are Secure wherever the product is
served over https.

The admin and portal session cookies are SameSite=Lax, which keeps another
site's forms from carrying them -- except from a sibling subdomain, which a
browser counts as the same site, and in any browser that does not implement
SameSite. So every state-changing request to /admin or /portal, and every
handshake of a cookie-authenticated WebSocket, has to come from this origin.
Current browsers say where a request came from in Sec-Fetch-Site; where that
is missing, the Origin header is compared with the Host. A request that says
neither -- curl, a script, a server calling the question API -- carries no
browser session to misuse, and passes.

This is the check Go's net/http CrossOriginProtection makes, and it stands in
for per-form tokens: no form and no fetch call had to change.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

from starlette.responses import PlainTextResponse
from starlette.websockets import WebSocketClose

_PROTECTED_PREFIXES = ("/admin", "/portal")
_PROTECTED_SOCKETS = ("/admin/", "/portal/", "/ws/")
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def is_cross_site(headers: dict[str, str]) -> bool:
    """True when the browser says the request came from another site."""
    site = headers.get("sec-fetch-site")
    if site:
        return site not in ("same-origin", "none")
    origin = headers.get("origin")
    if origin is None:
        return False
    return urlsplit(origin).netloc.lower() != (headers.get("host") or "").lower()


def _guarded(scope) -> bool:
    path = scope.get("path") or ""
    if scope["type"] == "websocket":
        return path.startswith(_PROTECTED_SOCKETS)
    return scope.get("method") not in _SAFE_METHODS and path.startswith(_PROTECTED_PREFIXES)


class RefuseCrossSiteRequests:
    """ASGI middleware: see the module docstring."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket") and _guarded(scope):
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
            if is_cross_site(headers):
                if scope["type"] == "http":
                    await PlainTextResponse("Refused: this request came from another site.",
                                            status_code=403)(scope, receive, send)
                else:
                    await WebSocketClose(code=1008)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def cookie_secure(request) -> bool:
    """Whether a session cookie may only travel over https.

    Behind a proxy that ends TLS the app sees plain http unless uvicorn trusts
    the proxy's forwarded headers, and the cookie went out without Secure. The
    public address the product is configured with says the same thing more
    reliably.
    """
    if request.url.scheme == "https":
        return True
    return os.getenv("PORTAL_BASE_URL", "").strip().lower().startswith("https://")
