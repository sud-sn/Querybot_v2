"""
A temporary password signs in to the password change, and nowhere else.

Signing in with a temporary password redirected to the change-password page,
but the session it issued was already complete: the chat, the dashboards and
every API worked, so the change was a suggestion. The change form decided
"forced" by whether it had been sent a current password (and the page by
?forced=1 in the address), so any signed-in session could set a new password
without knowing the old one. And temporary passwords never expired.

Now the stored flag decides. Until the user chooses their own password, their
session reaches the change-password page and nothing else -- pages send them
there, APIs answer 401, both sockets close. A settled user must give their
current password. The new one must differ from the old. A temporary password
stops signing in after 72 hours, and one issued before this change gets 72
hours from the upgrade.

Real portal routes, real signed cookies, a scratch store per test.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
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


def _routes():
    from portal import routes

    return routes


def _workspace(store):
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    return account_id


def _user(store, account_id, password=None):
    email = f"{os.urandom(4).hex()}@example.com"
    user_id, plain = store.create_user(account_id, "Ada", email, password=password)
    return user_id, plain, email


def _request(cookie=None, lang=None, query=None):
    routes = _routes()
    request = MagicMock()
    request.cookies = {}
    if cookie:
        request.cookies[routes._COOKIE] = cookie
    if lang:
        request.cookies[routes._LANG_COOKIE] = lang
    request.headers = {}
    request.query_params = query or {}
    request.url.scheme = "http"
    return request


def _shown(request_name, ctx=None):
    return {"page": request_name, **(ctx or {})}


def _call(coro):
    routes = _routes()
    with patch.object(routes, "_resp", side_effect=lambda request, name, ctx=None: _shown(name, ctx)):
        return asyncio.run(coro)


def _cookie_from(response):
    routes = _routes()
    for key, value in response.raw_headers:
        if key.lower() == b"set-cookie" and value.decode().startswith(routes._COOKIE + "="):
            return value.decode().split(";", 1)[0].split("=", 1)[1]
    return None


def _sign_in(account_id, email, password, lang=None):
    routes = _routes()
    return _call(routes.portal_login_submit(_request(lang=lang), account_id=account_id,
                                            email=email, password=password))


def _change(cookie, *, current="", new, confirm=None, lang=None):
    routes = _routes()
    return _call(routes.change_pw_submit(_request(cookie, lang=lang), current_pw=current,
                                         new_pw=new, confirm_pw=new if confirm is None else confirm))


def _socket(cookie):
    routes = _routes()
    websocket = MagicMock()
    websocket.cookies = {routes._COOKIE: cookie}
    websocket.query_params = {}
    websocket.close = AsyncMock()
    websocket.accept = AsyncMock()
    return websocket


def _temporary_session(store):
    account_id = _workspace(store)
    user_id, temp, email = _user(store, account_id)
    response = _sign_in(account_id, email, temp)
    return account_id, user_id, temp, email, response


class TestSigningInWithATemporaryPassword:

    def test_it_signs_in_to_the_password_change_only(self, fresh_store):
        routes = _routes()
        _, user_id, _, _, response = _temporary_session(fresh_store)
        assert response.status_code == 303 and response.headers["location"] == "/portal/change-password"
        cookie = _cookie_from(response)
        assert cookie
        assert routes._get_portal_user(_request(cookie)) is None
        assert routes._get_portal_user(_request(cookie), pending_change_ok=True)["id"] == user_id

    @pytest.mark.parametrize("page", ["portal_dashboard", "portal_chat", "notifications_page", "portal_kb"])
    def test_every_page_sends_them_to_the_password_change(self, fresh_store, page):
        routes = _routes()
        _, _, _, _, response = _temporary_session(fresh_store)
        shown = _call(getattr(routes, page)(_request(_cookie_from(response))))
        assert shown.status_code == 303 and shown.headers["location"] == "/portal/change-password"

    def test_an_api_refuses_them(self, fresh_store):
        routes = _routes()
        _, _, _, _, response = _temporary_session(fresh_store)
        answer = _call(routes.list_dashboards_api(_request(_cookie_from(response))))
        assert answer.status_code == 401

    def test_the_chat_socket_refuses_them(self, fresh_store):
        import gateway.webhooks as webhooks

        account_id, _, _, _, response = _temporary_session(fresh_store)
        websocket = _socket(_cookie_from(response))
        asyncio.run(webhooks.ws_chat(websocket, account_id))
        websocket.close.assert_called_once_with(code=4003)
        websocket.accept.assert_not_called()

    def test_the_notification_socket_refuses_them(self, fresh_store):
        routes = _routes()
        _, _, _, _, response = _temporary_session(fresh_store)
        websocket = _socket(_cookie_from(response))
        asyncio.run(routes.portal_notifications_ws(websocket))
        websocket.close.assert_called_once_with(code=4401)

    def test_a_reader_with_no_session_still_goes_to_sign_in(self, fresh_store):
        routes = _routes()
        shown = _call(routes.portal_dashboard(_request()))
        assert shown.headers["location"] == "/portal/login"


class TestChoosingTheirOwn:

    def test_the_page_is_the_forced_one_for_them_and_only_for_them(self, fresh_store):
        routes = _routes()
        account_id, _, _, _, response = _temporary_session(fresh_store)
        assert _call(routes.change_pw_page(_request(_cookie_from(response))))["forced"] is True

        _, password, email = _user(fresh_store, account_id, password="a-password-they-chose")
        settled = _cookie_from(_sign_in(account_id, email, password))
        shown = _call(routes.change_pw_page(_request(settled, query={"forced": "1"})))
        assert shown["forced"] is False

    def test_the_temporary_password_cannot_be_kept(self, fresh_store):
        _, user_id, temp, _, response = _temporary_session(fresh_store)
        shown = _change(_cookie_from(response), new=temp)
        assert shown["error"] == "Choose a password different from your current one."
        assert fresh_store.get_user(user_id)["is_temp_pw"]

    def test_the_refusals_are_in_the_readers_language(self, fresh_store):
        _, _, temp, _, response = _temporary_session(fresh_store)
        cookie = _cookie_from(response)
        assert _change(cookie, new=temp, lang="fr")["error"] == "Choisissez un mot de passe différent de l'actuel."
        assert _change(cookie, new="short", lang="fr")["error"] == (
            "Votre nouveau mot de passe doit contenir au moins 8 caractères.")
        assert _change(cookie, new="long-enough-1", confirm="long-enough-2", lang="fr")["error"] == (
            "Les deux mots de passe ne correspondent pas.")

    def test_their_own_password_ends_the_temporary_state(self, fresh_store):
        routes = _routes()
        _, user_id, _, email, response = _temporary_session(fresh_store)
        cookie = _cookie_from(response)
        done = _change(cookie, new="my-own-password")
        assert done.status_code == 303 and done.headers["location"] == "/portal/dashboard"
        user = fresh_store.get_user(user_id)
        assert not user["is_temp_pw"] and user["temp_pw_expires_at"] is None
        assert fresh_store.verify_password(user, "my-own-password")
        # The change re-issues this browser's session (the old cookie ended
        # with the temporary password), and the new one reaches everything.
        assert routes._get_portal_user(_request(_cookie_from(done)))["id"] == user_id


class TestASettledUser:

    def _settled(self, store):
        account_id = _workspace(store)
        user_id, password, email = _user(store, account_id, password="the-old-password")
        return user_id, password, _cookie_from(_sign_in(account_id, email, password))

    def test_leaving_out_the_current_password_changes_nothing(self, fresh_store):
        user_id, _, cookie = self._settled(fresh_store)
        shown = _change(cookie, current="", new="the-new-password")
        assert shown["error"] == "Your current password is incorrect."
        user = fresh_store.get_user(user_id)
        assert fresh_store.verify_password(user, "the-old-password")
        assert not fresh_store.verify_password(user, "the-new-password")

    def test_a_wrong_current_password_changes_nothing(self, fresh_store):
        user_id, _, cookie = self._settled(fresh_store)
        assert _change(cookie, current="a-guess", new="the-new-password")["error"]
        assert fresh_store.verify_password(fresh_store.get_user(user_id), "the-old-password")

    def test_the_right_current_password_changes_it(self, fresh_store):
        user_id, _, cookie = self._settled(fresh_store)
        done = _change(cookie, current="the-old-password", new="the-new-password")
        assert done.headers["location"] == "/portal/dashboard"
        assert fresh_store.verify_password(fresh_store.get_user(user_id), "the-new-password")


def _utc(text):
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def _set_expiry(store, user_id, value):
    with store.get_db() as conn:
        conn.execute("UPDATE portal_user SET temp_pw_expires_at = ? WHERE id = ?", (value, user_id))


class TestExpiry:

    def test_a_temporary_password_lasts_72_hours(self, fresh_store):
        account_id = _workspace(fresh_store)
        user_id, _, _ = _user(fresh_store, account_id)
        left = _utc(fresh_store.get_user(user_id)["temp_pw_expires_at"]) - datetime.now(timezone.utc)
        assert timedelta(hours=71, minutes=59) <= left <= timedelta(hours=72)

        _set_expiry(fresh_store, user_id, "2000-01-01 00:00:00")
        fresh_store.reset_user_password(user_id)
        left = _utc(fresh_store.get_user(user_id)["temp_pw_expires_at"]) - datetime.now(timezone.utc)
        assert timedelta(hours=71, minutes=59) <= left <= timedelta(hours=72)

        chosen, _, _ = _user(fresh_store, account_id, password="a-password-they-chose")
        assert fresh_store.get_user(chosen)["temp_pw_expires_at"] is None

    def test_an_expired_one_signs_in_nowhere(self, fresh_store):
        routes = _routes()
        account_id, user_id, temp, email, response = _temporary_session(fresh_store)
        cookie = _cookie_from(response)
        _set_expiry(fresh_store, user_id, "2000-01-01 00:00:00")

        refused = _sign_in(account_id, email, temp)
        assert refused["error"] == "This temporary password has expired. Ask your administrator for a new one."
        assert routes._get_portal_user(_request(cookie), pending_change_ok=True) is None
        page = _call(routes.change_pw_page(_request(cookie)))
        assert page.headers["location"] == "/portal/login"

    def test_the_expiry_is_explained_in_french(self, fresh_store):
        account_id, user_id, temp, email, _ = _temporary_session(fresh_store)
        _set_expiry(fresh_store, user_id, "2000-01-01 00:00:00")
        refused = _sign_in(account_id, email, temp, lang="fr")
        assert refused["error"] == ("Ce mot de passe temporaire a expiré. "
                                    "Demandez-en un nouveau à votre administrateur.")

    def test_one_issued_before_the_upgrade_gets_72_hours_from_it(self, fresh_store):
        account_id = _workspace(fresh_store)
        user_id, _, _ = _user(fresh_store, account_id)
        _set_expiry(fresh_store, user_id, None)
        fresh_store.init_db()
        left = _utc(fresh_store.get_user(user_id)["temp_pw_expires_at"]) - datetime.now(timezone.utc)
        assert timedelta(hours=71, minutes=59) <= left <= timedelta(hours=72)

    def test_the_store_says_when_one_has_expired(self, fresh_store):
        user = {"is_temp_pw": 1, "temp_pw_expires_at": "2026-09-26 10:00:00"}
        assert fresh_store.temporary_password_expired(user, now="2026-09-26 10:00:00")
        assert not fresh_store.temporary_password_expired(user, now="2026-09-26 09:59:59")
        assert not fresh_store.temporary_password_expired({**user, "is_temp_pw": 0}, now="2030-01-01 00:00:00")
