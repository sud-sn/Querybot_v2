"""A workspace has its own sign-in link, and a reader who follows it types only their email and password.

A reader had to be told an Account ID and type it into a monospace field
before their email -- the step most first sign-ins failed on. The admin's
People page now shows the workspace's link (/portal/login?workspace=<id>),
with a Copy button, and the new-user box says to sign in there. The link
fills the workspace in and starts at the email field; "another workspace"
goes back to the plain page.

The page never looks the workspace up: it says nothing about whether one
exists, as the sign-in itself does not. A reader who came by the link and
mistypes their password tries again on the same link. An admin may name a
workspace with any text, so the link takes any, shown as text.

Signed in and out through the real routes, on a fresh store.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from tests.portal_render import render, visible


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    from portal import routes

    routes.store.init_db()
    return routes.store


def _request(query=None):
    request = MagicMock()
    request.cookies = {}
    request.headers = {}
    request.query_params = query or {}
    request.url.scheme = "http"
    return request


def _call(coro):
    from portal import routes

    with patch.object(routes, "_resp", side_effect=lambda request, name, ctx=None: {"page": name, **(ctx or {})}):
        return asyncio.run(coro)


def test_the_link_fills_the_workspace_in(fresh_store):
    from portal import routes

    shown = _call(routes.portal_login_page(_request({"workspace": "acct-retail"})))
    assert shown["workspace"] == "acct-retail"


@pytest.mark.parametrize("value", ["", "   ", "a\nb", "a\x00b", "x" * 129])
def test_anything_that_cannot_be_a_workspace_id_is_ignored(fresh_store, value):
    from portal import routes

    assert _call(routes.portal_login_page(_request({"workspace": value})))["workspace"] == ""


def test_a_workspace_named_with_any_text_is_filled_in(fresh_store):
    """The new-client form takes any text as the id, so the link must too."""
    from portal import routes

    assert _call(routes.portal_login_page(_request({"workspace": "Acme Retail"})))["workspace"] == "Acme Retail"


def test_what_the_link_names_is_shown_as_text_never_as_markup():
    raw = render("portal_login.html", path="/portal/login", workspace='"><script>alert(1)</script>')
    assert "<script>alert(1)" not in raw
    assert "&lt;script&gt;" in raw or "&#34;&gt;&lt;script" in raw


def test_the_page_asks_only_for_email_and_password():
    page = visible(render("portal_login.html", path="/portal/login", workspace="acct-retail"))
    raw = render("portal_login.html", path="/portal/login", workspace="acct-retail")
    assert 'name="account_id" value="acct-retail"' in raw and 'type="hidden" name="account_id"' in raw
    assert 'id="login-account"' not in raw
    assert "acct-retail" in page and 'href="/portal/login"' in raw, "the way back to the plain page"
    email = raw[raw.index('id="login-email"'):]
    assert "autofocus" in email[:email.index(">")]


def test_without_the_link_the_page_asks_for_the_workspace_too():
    raw = render("portal_login.html", path="/portal/login")
    assert 'id="login-account"' in raw and 'name="from_link"' not in raw


def test_a_mistyped_password_tries_again_on_the_same_link(fresh_store):
    from portal import routes

    account_id = f"acct{os.urandom(4).hex()}"
    fresh_store.upsert_client(account_id, "Test Ltd")
    email = f"{os.urandom(4).hex()}@example.com"
    fresh_store.create_user(account_id, "Ada", email, password="right-password-123")
    shown = _call(routes.portal_login_submit(_request(), account_id=account_id, email=email,
                                             password="wrong", from_link="1"))
    assert shown["page"] == "portal_login.html" and shown["workspace"] == account_id
    shown = _call(routes.portal_login_submit(_request(), account_id=account_id, email=email,
                                             password="wrong", from_link=""))
    assert shown["workspace"] == ""


def test_the_link_signs_in(fresh_store):
    from portal import routes

    account_id = f"acct{os.urandom(4).hex()}"
    fresh_store.upsert_client(account_id, "Test Ltd")
    email = f"{os.urandom(4).hex()}@example.com"
    fresh_store.create_user(account_id, "Ada", email, password="right-password-123")
    response = _call(routes.portal_login_submit(_request(), account_id=account_id, email=email,
                                                password="right-password-123", from_link="1"))
    assert response.status_code == 303


def test_the_people_page_shows_the_link_with_a_copy_button():
    from jinja2 import ChainableUndefined, Environment, FileSystemLoader

    from core.static_assets import asset_url

    env = Environment(loader=FileSystemLoader("admin/templates"), undefined=ChainableUndefined)
    env.globals["asset"] = asset_url
    request = MagicMock()
    request.url.path = "/admin/clients/acct-retail/users"
    request.base_url = "https://querybot.example/"
    html = env.get_template("client_users.html").render(
        request=request, client={"account_id": "acct-retail", "client_name": "Retail", "state": "READY"},
        users=[], groups=[])
    assert "https://querybot.example/portal/login?workspace=acct-retail" in html
    assert "copySignInLink(this)" in html
