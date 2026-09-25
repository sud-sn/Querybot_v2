"""
A new or reset password is shown to the admin once, and never carried in a URL.

Creating a portal user, or resetting one's password, redirected to the users
page with the password in the query string -- temp_pw=... -- a password the
admin chose included. The access log writes every request line, query string
and all, and the browser keeps it in its history; and the page, which says a
temporary password will not be shown again, showed it on every reload. So
rotating an exposed password through the admin page would have written the
new one to the server's log.

The redirect now carries a single-use token. The page shows the password by it
once, for its own workspace, within ten minutes; a reload, another workspace
or a late visit shows none.

A scratch store (QUERYBOT_DB_PATH); no real database.
"""

from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, quote, urlparse

import pytest


def _request(query=None):
    request = MagicMock()
    request.query_params = query or {}
    request.session = {}
    return request


@pytest.fixture
def account():
    import store

    store.init_db()
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    return account_id


def _create(account_id, password=""):
    from admin import routes

    with patch.object(routes, "_is_auth", return_value=True):
        response = asyncio.run(routes.user_create(
            _request(), account_id, name="Dana Roy", email=f"{os.urandom(3).hex()}@example.com",
            group_id="", role="analyst", password=password, confirm_password=password,
        ))
    assert response.status_code == 303
    return response.headers["location"]


def _reset(account_id, user_id):
    from admin import routes

    with patch.object(routes, "_is_auth", return_value=True):
        response = asyncio.run(routes.user_reset_password(_request(), account_id, user_id))
    assert response.status_code == 303
    return response.headers["location"]


def _page(account_id, location):
    from admin import routes

    query = {key: values[0] for key, values in parse_qs(urlparse(location).query).items()}
    with patch.object(routes, "_is_auth", return_value=True), \
            patch.object(routes, "_resp", side_effect=lambda request, name, context: context):
        return asyncio.run(routes.users_page(_request(query), account_id))


def _user(account_id):
    import store

    return next(u for u in store.list_users(account_id) if u["name"] == "Dana Roy")


class TestANewUser:

    def test_the_temporary_password_is_not_in_the_url(self, account):
        location = _create(account)
        shown = _page(account, location)
        assert shown["temp_pw"] and shown["temp_pw"] not in location
        assert "temp_pw" not in parse_qs(urlparse(location).query)
        # What the admin is shown is the password the user signs in with.
        import store

        user = store.get_user_by_email(account, _user(account)["email"])
        assert store.verify_password(user, shown["temp_pw"]) and user["is_temp_pw"]
        assert (shown["new_user"], shown["is_temp"], shown["reveal_kind"]) == ("Dana Roy", "1", "created")

    def test_a_password_the_admin_chose_is_not_in_the_url(self, account):
        location = _create(account, password="Str0ng-pass!")
        assert "Str0ng-pass!" not in location and quote("Str0ng-pass!") not in location
        shown = _page(account, location)
        assert (shown["temp_pw"], shown["is_temp"]) == ("Str0ng-pass!", "0")

    def test_it_is_shown_once(self, account):
        location = _create(account)
        assert _page(account, location)["temp_pw"]
        again = _page(account, location)
        assert (again["temp_pw"], again["new_user"], again["saved"]) == ("", "", "1")


class TestAReset:

    def test_the_new_password_is_shown_once_and_must_be_changed(self, account):
        import store

        _page(account, _create(account))
        location = _reset(account, _user(account)["id"])
        shown = _page(account, location)
        assert shown["temp_pw"] and shown["temp_pw"] not in location
        user = store.get_user(_user(account)["id"])
        assert store.verify_password(user, shown["temp_pw"]) and user["is_temp_pw"]
        assert shown["reveal_kind"] == "reset"
        assert _page(account, location)["temp_pw"] == ""

    def test_another_workspace_is_shown_nothing(self, account):
        import store

        other = f"acct{os.urandom(4).hex()}"
        store.upsert_client(other, "Other Ltd")
        location = _create(account)
        assert _page(other, location)["temp_pw"] == ""

    def test_a_late_visit_is_shown_nothing(self, account):
        from admin import routes

        location = _create(account)
        later = time.monotonic() + routes._REVEAL_TTL_SECONDS + 1
        with patch.object(routes.time, "monotonic", return_value=later):
            assert _page(account, location)["temp_pw"] == ""


    def test_passwords_nobody_came_for_are_not_kept(self, account):
        from admin import routes

        location = _create(account)
        token = parse_qs(urlparse(location).query)["reveal"][0]
        later = time.monotonic() + routes._REVEAL_TTL_SECONDS + 1
        with patch.object(routes.time, "monotonic", return_value=later):
            _create(account)
        assert token not in routes._reveals


class TestThePage:

    def _render(self, account, location):
        from admin import routes

        class _Url:
            path = f"/admin/clients/{account}/users"

        class _FakeRequest:
            url = _Url()
            base_url = "https://querybot.example/"
            query_params: dict = {}
            session = {"admin_id": "admin_user_1"}
            scope = {"type": "http"}
            cookies: dict = {}
            headers: dict = {}

        context = _page(account, location)
        return routes.templates.get_template("client_users.html").render(request=_FakeRequest(), **context)

    def test_a_reset_says_so_and_shows_the_password(self, account):
        _page(account, _create(account))
        location = _reset(account, _user(account)["id"])
        html = self._render(account, location)
        assert "Password reset — Dana Roy" in html
        assert "It will not be shown again." in html
        assert 'id="tempPwCode"' in html

    def test_a_password_the_admin_chose_says_so(self, account):
        html = self._render(account, _create(account, password="Str0ng-pass!"))
        assert "User created — Dana Roy" in html
        assert "Password set by you." in html
