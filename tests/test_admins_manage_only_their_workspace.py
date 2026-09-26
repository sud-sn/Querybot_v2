"""
Admins manage only their own workspace's users, and every change is recorded.

The user routes took the workspace from the URL and never checked the user
against it: tenant A's page could reset tenant B's user and show B's new
password, deactivate B's users, or delete them -- and group routes did the
same with groups. An unknown user id answered "Changes saved". Nothing was
recorded. A reset could only generate a password; a password the admin chose
at creation was permanent, so the admin went on knowing it; the page promised
the sign-in email next to a new password and never showed it; and the page
that shows a password could come back from the browser's cache.

Now a user or group outside the workspace is a 404. Every create, reset,
(de)activation, role, group or name change and deletion is recorded with the
admin session and address it came from, and a user's own password change is
recorded too; the users page lists them. A reset can generate a password or
take one the admin chooses, with "ask them to choose their own" on by
default, as at creation. The reveal shows the email, and the page is sent
no-store.

Real routes, real requests, real store; a scratch database per test.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlencode, urlparse

import pytest
from fastapi import HTTPException
from starlette.requests import Request


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    from admin import routes

    routes.store.init_db()
    with patch.object(routes, "_is_auth", return_value=True):
        yield routes.store


def _routes():
    from admin import routes

    return routes


def _request(path="/admin", query=None, cookie="an-admin-session-cookie", client=("203.0.113.9", 5555)):
    routes = _routes()
    headers = [(b"cookie", f"{routes._COOKIE}={cookie}".encode())] if cookie else []
    return Request({
        "type": "http", "method": "GET", "path": path, "root_path": "", "scheme": "http",
        "query_string": urlencode(query or {}).encode(), "headers": headers,
        "server": ("testserver", 80), "client": client,
    })


def _workspace(store, name="Test Ltd"):
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, name)
    return account_id


def _user(store, account_id, password="the-users-password"):
    email = f"{os.urandom(4).hex()}@example.com"
    user_id, _ = store.create_user(account_id, "Ada", email, password=password)
    return user_id, email


def _run(coro):
    return asyncio.run(coro)


def _reset(account_id, user_id, **form):
    routes = _routes()
    fields = {"mode": "generate", "password": "", "confirm_password": "", "must_change": ""}
    fields.update(form)
    return _run(routes.user_reset_password(_request(), account_id, user_id, **fields))


def _page(account_id, location=None):
    routes = _routes()
    query = {k: v[0] for k, v in parse_qs(urlparse(location).query).items()} if location else {}
    return _run(routes.users_page(_request(f"/admin/clients/{account_id}/users", query), account_id))


def _create(account_id, **form):
    routes = _routes()
    fields = {"name": "Grace", "email": f"{os.urandom(4).hex()}@example.com", "group_id": "",
              "role": "analyst", "password": "", "confirm_password": "", "must_change": ""}
    fields.update(form)
    return _run(routes.user_create(_request(), account_id, **fields)), fields["email"]


def _error(response):
    return parse_qs(urlparse(response.headers["location"]).query).get("error", [""])[0]


class TestAnotherWorkspacesUsers:

    def test_a_reset_is_refused_and_changes_nothing(self, fresh_store):
        mine, theirs = _workspace(fresh_store), _workspace(fresh_store)
        user_id, _ = _user(fresh_store, theirs)
        with pytest.raises(HTTPException) as refused:
            _reset(mine, user_id)
        assert refused.value.status_code == 404
        assert fresh_store.verify_password(fresh_store.get_user(user_id), "the-users-password")
        assert fresh_store.list_user_events(mine) == [] and fresh_store.list_user_events(theirs) == []

    def test_a_deactivation_is_refused(self, fresh_store):
        routes = _routes()
        mine, theirs = _workspace(fresh_store), _workspace(fresh_store)
        user_id, _ = _user(fresh_store, theirs)
        with pytest.raises(HTTPException) as refused:
            _run(routes.user_update(_request(), mine, user_id, name="", group_id="", role="", is_active="0"))
        assert refused.value.status_code == 404
        assert fresh_store.get_user(user_id)["is_active"]

    def test_a_deletion_is_refused(self, fresh_store):
        routes = _routes()
        mine, theirs = _workspace(fresh_store), _workspace(fresh_store)
        user_id, _ = _user(fresh_store, theirs)
        with pytest.raises(HTTPException) as refused:
            _run(routes.user_delete(_request(), mine, user_id))
        assert refused.value.status_code == 404
        assert fresh_store.get_user(user_id)

    def test_an_unknown_user_is_a_404_not_changes_saved(self, fresh_store):
        mine = _workspace(fresh_store)
        with pytest.raises(HTTPException) as refused:
            _reset(mine, 987654)
        assert refused.value.status_code == 404


class TestAnotherWorkspacesGroups:

    def test_a_user_cannot_be_put_in_it(self, fresh_store):
        routes = _routes()
        mine, theirs = _workspace(fresh_store), _workspace(fresh_store)
        user_id, _ = _user(fresh_store, mine)
        their_group = fresh_store.create_group(theirs, "Theirs")
        with pytest.raises(HTTPException) as refused:
            _run(routes.user_update(_request(), mine, user_id, name="", group_id=str(their_group),
                                    role="", is_active=""))
        assert refused.value.status_code == 404
        assert fresh_store.get_user(user_id)["group_id"] is None

    def test_it_cannot_be_deleted_or_given_tables(self, fresh_store):
        routes = _routes()
        mine, theirs = _workspace(fresh_store), _workspace(fresh_store)
        their_group = fresh_store.create_group(theirs, "Theirs")
        with pytest.raises(HTTPException):
            _run(routes.group_delete(_request(), mine, their_group))
        with pytest.raises(HTTPException):
            _run(routes.group_save_tables(_request(), mine, their_group))
        assert fresh_store.get_group(their_group)

    def test_a_new_user_cannot_be_created_in_it(self, fresh_store):
        mine, theirs = _workspace(fresh_store), _workspace(fresh_store)
        their_group = fresh_store.create_group(theirs, "Theirs")
        with pytest.raises(HTTPException):
            _create(mine, group_id=str(their_group))


class TestTheRecord:

    def test_a_reset_records_who_whom_and_when(self, fresh_store):
        mine = _workspace(fresh_store)
        user_id, email = _user(fresh_store, mine)
        _reset(mine, user_id)
        event = fresh_store.list_user_events(mine)[0]
        assert (event["action"], event["user_email"], event["detail"]) == (
            "password_reset", email, "temporary password generated")
        assert event["actor"].startswith("admin session ") and event["actor_ip"] == "203.0.113.9"
        assert event["created_at"]

    def test_every_kind_of_change_is_recorded(self, fresh_store):
        routes = _routes()
        mine = _workspace(fresh_store)
        response, email = _create(mine)
        user_id = fresh_store.get_user_by_email(mine, email)["id"]
        group = fresh_store.create_group(mine, "Analysts")
        _run(routes.user_update(_request(), mine, user_id, name="Grace H", group_id=str(group),
                                role="admin", is_active=""))
        _run(routes.user_update(_request(), mine, user_id, name="", group_id="", role="", is_active="0"))
        _run(routes.user_update(_request(), mine, user_id, name="", group_id="", role="", is_active="1"))
        _run(routes.user_delete(_request(), mine, user_id))
        actions = [e["action"] for e in reversed(fresh_store.list_user_events(mine))]
        assert actions == ["created", "role_changed", "group_changed", "renamed",
                           "deactivated", "reactivated", "deleted"]

    def test_a_users_own_change_is_recorded_as_theirs(self, fresh_store):
        from portal import routes as portal

        mine = _workspace(fresh_store)
        user_id, email = _user(fresh_store, mine)
        request = MagicMock()
        request.cookies = {portal._COOKIE: portal._sign_session_value(user_id)}
        request.headers = {}
        request.url.scheme = "http"
        request.client.host = "198.51.100.7"
        _run(portal.change_pw_submit(request, current_pw="the-users-password",
                                     new_pw="a-new-password", confirm_pw="a-new-password"))
        event = fresh_store.list_user_events(mine)[0]
        assert (event["action"], event["user_email"], event["actor"], event["actor_ip"]) == (
            "password_changed", email, "the user", "198.51.100.7")

    def test_the_users_page_lists_the_changes(self, fresh_store):
        mine = _workspace(fresh_store)
        user_id, email = _user(fresh_store, mine)
        _reset(mine, user_id)
        html = _page(mine).body.decode()
        assert "Recent account changes" in html and email in html and "Password reset" in html


class TestChoosingThePassword:

    def test_a_reset_can_take_a_password_the_user_must_replace(self, fresh_store):
        mine = _workspace(fresh_store)
        user_id, email = _user(fresh_store, mine)
        location = _reset(mine, user_id, mode="choose", password="chosen-by-admin",
                          confirm_password="chosen-by-admin", must_change="1").headers["location"]
        user = fresh_store.get_user(user_id)
        assert fresh_store.verify_password(user, "chosen-by-admin") and user["is_temp_pw"]
        html = _page(mine, location).body.decode()
        assert "chosen-by-admin" in html and email in html

    def test_a_reset_can_take_a_password_the_user_keeps(self, fresh_store):
        mine = _workspace(fresh_store)
        user_id, _ = _user(fresh_store, mine)
        _reset(mine, user_id, mode="choose", password="chosen-by-admin",
               confirm_password="chosen-by-admin", must_change="")
        assert not fresh_store.get_user(user_id)["is_temp_pw"]

    def test_a_mismatch_changes_nothing(self, fresh_store):
        mine = _workspace(fresh_store)
        user_id, _ = _user(fresh_store, mine)
        response = _reset(mine, user_id, mode="choose", password="chosen-by-admin",
                          confirm_password="something-else", must_change="1")
        assert _error(response) == "Passwords do not match"
        assert fresh_store.verify_password(fresh_store.get_user(user_id), "the-users-password")
        assert fresh_store.list_user_events(mine) == []

    def test_a_chosen_password_at_creation_can_be_temporary(self, fresh_store):
        mine = _workspace(fresh_store)
        _, email = _create(mine, password="chosen-at-creation", confirm_password="chosen-at-creation",
                           must_change="1")
        user = fresh_store.get_user_by_email(mine, email)
        assert user["is_temp_pw"] and user["temp_pw_expires_at"]
        assert fresh_store.verify_password(user, "chosen-at-creation")


class TestCreating:

    def test_a_duplicate_email_is_refused_in_words(self, fresh_store):
        mine = _workspace(fresh_store)
        user_id, email = _user(fresh_store, mine)
        fresh_store.update_user(user_id, is_active=0)
        response, _ = _create(mine, email=email.upper())
        assert _error(response) == "A user with this email already exists in this workspace."

    def test_a_database_error_is_not_shown_to_the_admin(self, fresh_store):
        routes = _routes()
        mine = _workspace(fresh_store)
        with patch.object(routes.store, "create_user",
                          side_effect=sqlite3.IntegrityError("UNIQUE constraint failed: portal_user.email")):
            response, _ = _create(mine)
        assert _error(response) == "The user could not be created."


class TestThePageThatShowsAPassword:

    def test_it_is_not_cached_and_names_the_sign_in_email(self, fresh_store):
        mine = _workspace(fresh_store)
        user_id, email = _user(fresh_store, mine)
        page = _page(mine, _reset(mine, user_id).headers["location"])
        assert page.headers["cache-control"] == "no-store"
        html = page.body.decode()
        assert f"Email: <strong>{email}</strong>" in html
        assert "Every session the user had open is signed out." in html
