"""
Another site cannot act with a signed-in user's session, and the session cookie
only travels over https where the product is served over https.

The session cookies are SameSite=Lax. That keeps another site's forms from
carrying them, except from a sibling subdomain -- which a browser counts as the
same site -- and in browsers without SameSite. Nothing else checked where a
write came from: no form token, no Origin check, on /admin, /portal or the chat
socket. And behind a proxy that ends TLS the app sees plain http, so the cookie
went out without Secure.

Now every state-changing request to /admin or /portal, and every handshake of a
cookie-authenticated socket, is refused when the browser says it came from
another site (Sec-Fetch-Site, or Origin against Host where that header is
missing); a caller that sends neither -- curl, a server -- carries no browser
session and passes. The cookies are Secure when the request is https or the
configured public address is.

The real application (main.app), without its startup work, on a scratch store.
"""

from __future__ import annotations

import os

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.delenv("PORTAL_BASE_URL", raising=False)
    import main

    main.store.init_db()
    return TestClient(main.app)


def _workspace_user(store):
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    email = f"{os.urandom(4).hex()}@example.com"
    store.create_user(account_id, "Ada", email, password="the-right-password")
    return account_id, email


def _sign_in(client, account_id, email, **headers):
    return client.post("/portal/login", headers=headers, follow_redirects=False,
                       data={"account_id": account_id, "email": email, "password": "the-right-password"})


class TestWritesFromAnotherSite:

    @pytest.mark.parametrize("headers", [
        {"sec-fetch-site": "cross-site"},
        {"sec-fetch-site": "same-site"},  # a sibling subdomain
        {"origin": "https://evil.example"},
        {"origin": "null"},
    ])
    def test_are_refused_before_anything_runs(self, app_client, headers):
        import main

        account_id, email = _workspace_user(main.store)
        response = _sign_in(app_client, account_id, email, **headers)
        assert response.status_code == 403
        assert "set-cookie" not in response.headers
        admin = app_client.post("/admin/login", data={"password": "x"}, headers=headers, follow_redirects=False)
        assert admin.status_code == 403

    @pytest.mark.parametrize("headers", [
        {"sec-fetch-site": "same-origin"},
        {"sec-fetch-site": "none"},              # typed into the address bar
        {"origin": "http://testserver"},         # an older browser, same origin
        {},                                      # not a browser at all
    ])
    def test_from_this_site_go_through(self, app_client, headers):
        import main

        account_id, email = _workspace_user(main.store)
        response = _sign_in(app_client, account_id, email, **headers)
        assert response.status_code == 303 and response.headers["location"] == "/portal/dashboard"

    def test_reading_a_page_is_never_refused(self, app_client):
        response = app_client.get("/portal/login", headers={"sec-fetch-site": "cross-site"})
        assert response.status_code == 200

    def test_the_question_api_and_webhooks_are_not_this_checks_business(self, app_client, monkeypatch):
        # Server-to-server endpoints authenticate themselves; the refusal is for
        # the cookie-carrying consoles only.
        monkeypatch.delenv("QUERYBOT_API_KEY", raising=False)
        response = app_client.post("/api/ask", json={}, headers={"sec-fetch-site": "cross-site"})
        assert response.status_code == 503


class TestTheChatSocket:

    def _connect(self, client, account_id, headers):
        with client.websocket_connect(f"/ws/chat/{account_id}", headers=headers) as socket:
            socket.receive_text()

    def test_a_handshake_from_another_site_is_refused(self, app_client):
        with pytest.raises(WebSocketDisconnect) as refused:
            self._connect(app_client, "acct", {"origin": "https://evil.example"})
        assert refused.value.code == 1008

    def test_a_handshake_from_this_site_reaches_the_socket(self, app_client):
        # Unsigned, so the socket itself closes it -- with its own code.
        with pytest.raises(WebSocketDisconnect) as closed:
            self._connect(app_client, "acct", {"origin": "http://testserver"})
        assert closed.value.code != 1008


class TestSecureCookies:

    def test_over_plain_http_with_no_https_address_the_cookie_is_not_secure(self, app_client):
        import main

        account_id, email = _workspace_user(main.store)
        cookie = _sign_in(app_client, account_id, email).headers["set-cookie"]
        assert "secure" not in cookie.lower()

    def test_an_https_public_address_makes_the_cookie_secure_behind_a_plain_http_proxy(
            self, app_client, monkeypatch):
        import main

        monkeypatch.setenv("PORTAL_BASE_URL", "https://bi.example.com")
        account_id, email = _workspace_user(main.store)
        cookies = _sign_in(app_client, account_id, email).headers.get_list("set-cookie")
        assert cookies and all("secure" in c.lower() for c in cookies)
        from admin import credentials

        credentials.set_password("the-admin-password")
        admin = app_client.post("/admin/login", data={"password": "the-admin-password"}, follow_redirects=False)
        assert admin.status_code == 303 and "secure" in admin.headers["set-cookie"].lower()
