"""
A password change ends every other session; sessions end on their own too.

A portal session cookie was a signature of the user id: no expiry, and nothing
a password change, a reset or a deactivation could invalidate -- reactivating
an account brought every old cookie back. The admin console was worse: every
sign-in got the same cookie, a signature of the word "admin", which never
expired and survived any password change. And the admin's own password could
be changed without the current one.

Now a portal cookie carries the user's session version and when it was issued;
a password change, a reset and a deactivation bump the version, and a cookie
older than 7 days is refused. An admin cookie carries the admin session
version, when, and a nonce; setting the admin password (on the System page or
from the server) bumps the version, and a cookie older than 8 hours is
refused. The browser that changed a password gets a new cookie. Both cookies
also expire in the browser.

Real routes, real signed cookies, a scratch store per test.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    from portal import routes

    routes.store.init_db()
    return routes.store


def _portal():
    from portal import routes

    return routes


def _admin():
    from admin import routes

    return routes


def _request(cookie_name=None, cookie=None):
    request = MagicMock()
    request.cookies = {cookie_name: cookie} if cookie else {}
    request.headers = {}
    request.query_params = {}
    request.url.scheme = "http"
    return request


def _set_cookie(response, name):
    for key, value in response.raw_headers:
        text = value.decode()
        if key.lower() == b"set-cookie" and text.startswith(name + "="):
            return text
    return None


def _cookie_value(response, name):
    header = _set_cookie(response, name)
    return header.split(";", 1)[0].split("=", 1)[1] if header else None


def _resp_stub(request, name, ctx=None):
    return {"page": name, **(ctx or {})}


# ── Portal ──────────────────────────────────────────────────────────────────

def _reader(store):
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    email = f"{os.urandom(4).hex()}@example.com"
    user_id, _ = store.create_user(account_id, "Ada", email, password="the-old-password")
    return account_id, user_id, email


def _portal_sign_in(account_id, email, password):
    routes = _portal()
    with patch.object(routes, "_resp", side_effect=_resp_stub):
        response = asyncio.run(routes.portal_login_submit(
            _request(), account_id=account_id, email=email, password=password))
    return response


def _portal_user(cookie):
    routes = _portal()
    return routes._get_portal_user(_request(routes._COOKIE, cookie))


class TestPortalSessions:

    def test_changing_your_password_signs_out_your_other_browsers(self, fresh_store):
        routes = _portal()
        account_id, user_id, email = _reader(fresh_store)
        here = _cookie_value(_portal_sign_in(account_id, email, "the-old-password"), routes._COOKIE)
        elsewhere = _cookie_value(_portal_sign_in(account_id, email, "the-old-password"), routes._COOKIE)
        assert _portal_user(here) and _portal_user(elsewhere)

        with patch.object(routes, "_resp", side_effect=_resp_stub):
            done = asyncio.run(routes.change_pw_submit(
                _request(routes._COOKIE, here), current_pw="the-old-password",
                new_pw="the-new-password", confirm_pw="the-new-password"))
        assert done.headers["location"] == "/portal/dashboard"
        renewed = _cookie_value(done, routes._COOKIE)
        assert _portal_user(renewed)["id"] == user_id
        assert _portal_user(elsewhere) is None
        assert _portal_user(here) is None

    def test_an_admin_reset_signs_the_user_out(self, fresh_store):
        routes = _portal()
        account_id, user_id, email = _reader(fresh_store)
        cookie = _cookie_value(_portal_sign_in(account_id, email, "the-old-password"), routes._COOKIE)
        admin = _admin()
        with patch.object(admin, "_is_auth", return_value=True):
            asyncio.run(admin.user_reset_password(MagicMock(), account_id, user_id))
        # Not even the change-password page: whoever held this session does
        # not get to choose the password the admin just reset.
        assert routes._get_portal_user(_request(routes._COOKIE, cookie), pending_change_ok=True) is None

    def test_a_deactivation_ends_sessions_that_reactivation_does_not_revive(self, fresh_store):
        routes = _portal()
        account_id, user_id, email = _reader(fresh_store)
        cookie = _cookie_value(_portal_sign_in(account_id, email, "the-old-password"), routes._COOKIE)
        fresh_store.update_user(user_id, is_active=0)
        fresh_store.update_user(user_id, is_active=1)
        assert _portal_user(cookie) is None
        again = _cookie_value(_portal_sign_in(account_id, email, "the-old-password"), routes._COOKIE)
        assert _portal_user(again)["id"] == user_id

    def test_other_changes_to_the_user_leave_sessions_alone(self, fresh_store):
        routes = _portal()
        account_id, user_id, email = _reader(fresh_store)
        cookie = _cookie_value(_portal_sign_in(account_id, email, "the-old-password"), routes._COOKIE)
        fresh_store.update_user(user_id, name="Ada Lovelace", role="admin")
        fresh_store.update_user(user_id, is_active=1)
        assert _portal_user(cookie)["id"] == user_id

    def test_a_session_lasts_seven_days(self, fresh_store):
        routes = _portal()
        _, user_id, _ = _reader(fresh_store)
        now = int(time.time())
        assert _portal_user(routes._sign_session_value(user_id, issued=now - 6 * 86400))["id"] == user_id
        assert _portal_user(routes._sign_session_value(user_id, issued=now - 7 * 86400 - 60)) is None

    def test_the_browser_drops_the_cookie_after_seven_days_too(self, fresh_store):
        routes = _portal()
        account_id, _, email = _reader(fresh_store)
        header = _set_cookie(_portal_sign_in(account_id, email, "the-old-password"), routes._COOKIE)
        assert "Max-Age=604800" in header

    def test_a_cookie_from_before_sessions_had_versions_is_refused(self, fresh_store):
        routes = _portal()
        _, user_id, _ = _reader(fresh_store)
        payload = str(user_id).encode()
        sig = hmac.new(routes._session_secret().encode(), payload, hashlib.sha256).hexdigest()
        legacy = base64.urlsafe_b64encode(payload).decode().rstrip("=") + "." + sig
        assert routes._read_session_value(legacy) is None
        assert _portal_user(legacy) is None

    def test_the_chat_socket_closes_on_an_ended_session(self, fresh_store):
        import gateway.webhooks as webhooks

        routes = _portal()
        account_id, user_id, email = _reader(fresh_store)
        cookie = _cookie_value(_portal_sign_in(account_id, email, "the-old-password"), routes._COOKIE)
        fresh_store.change_password(user_id, "changed-elsewhere")
        websocket = MagicMock()
        websocket.cookies = {routes._COOKIE: cookie}
        websocket.query_params = {}
        websocket.close = AsyncMock()
        websocket.accept = AsyncMock()
        asyncio.run(webhooks.ws_chat(websocket, account_id))
        websocket.close.assert_called_once_with(code=4003)
        websocket.accept.assert_not_called()


# ── Admin console ───────────────────────────────────────────────────────────

def _admin_setup(password="the-admin-password"):
    routes = _admin()
    with patch.object(routes, "_resp", side_effect=_resp_stub):
        response = asyncio.run(routes.setup_submit(
            _request(), admin_password=password, anthropic_key="", openai_key="",
            default_provider="anthropic", default_model="m", kb_model="k"))
    return _cookie_value(response, routes._COOKIE)


def _admin_sign_in(password="the-admin-password"):
    routes = _admin()
    with patch.object(routes, "_resp", side_effect=_resp_stub):
        response = asyncio.run(routes.login_submit(_request(), password=password))
    return response


def _admin_ok(cookie):
    routes = _admin()
    return routes._is_auth(_request(routes._COOKIE, cookie))


def _change_admin_password(cookie, current, new):
    routes = _admin()
    with patch.object(routes, "_resp", side_effect=_resp_stub):
        return asyncio.run(routes.system_password(
            _request(routes._COOKIE, cookie), new_password=new, confirm_password=new,
            current_password=current))


class TestAdminSessions:

    def test_every_sign_in_gets_its_own_cookie(self, fresh_store):
        routes = _admin()
        _admin_setup()
        one = _cookie_value(_admin_sign_in(), routes._COOKIE)
        two = _cookie_value(_admin_sign_in(), routes._COOKIE)
        assert one != two and _admin_ok(one) and _admin_ok(two)

    def test_changing_the_admin_password_signs_out_the_other_sessions(self, fresh_store):
        routes = _admin()
        here = _admin_setup()
        elsewhere = _cookie_value(_admin_sign_in(), routes._COOKIE)
        done = _change_admin_password(here, "the-admin-password", "a-new-admin-password")
        assert done.headers["location"] == "/admin/system?saved=password"
        assert _admin_ok(_cookie_value(done, routes._COOKIE))
        assert not _admin_ok(elsewhere) and not _admin_ok(here)

    def test_the_current_password_is_required(self, fresh_store):
        from admin import credentials

        here = _admin_setup()
        shown = _change_admin_password(here, "a-guess", "a-new-admin-password")
        assert shown["error"] == "Current password is incorrect"
        assert credentials.verify("the-admin-password") and _admin_ok(here)
        shown = _change_admin_password(here, "", "a-new-admin-password")
        assert shown["error"] == "Current password is incorrect"

    def test_the_server_side_reset_signs_out_every_admin_session(self, fresh_store):
        from admin import reset_password

        here = _admin_setup()
        answers = iter(["a-new-admin-password", "a-new-admin-password"])
        assert reset_password.main([], prompt=lambda _label: next(answers)) == 0
        assert not _admin_ok(here)

    def test_an_admin_session_lasts_eight_hours(self, fresh_store):
        routes = _admin()
        _admin_setup()
        now = int(time.time())
        assert _admin_ok(routes._sign_admin_session(issued=now - 7 * 3600))
        assert not _admin_ok(routes._sign_admin_session(issued=now - 8 * 3600 - 60))
        header = _set_cookie(_admin_sign_in(), routes._COOKIE)
        assert "Max-Age=28800" in header

    def test_the_old_fixed_cookie_no_longer_signs_in(self, fresh_store):
        routes = _admin()
        _admin_setup()
        payload = b"admin"
        sig = hmac.new(routes._session_secret().encode(), payload, hashlib.sha256).hexdigest()
        old = base64.urlsafe_b64encode(payload).decode().rstrip("=") + "." + sig
        assert not _admin_ok(old)
