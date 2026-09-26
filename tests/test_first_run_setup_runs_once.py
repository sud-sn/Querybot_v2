"""
First-run setup runs once.

The setup form's GET checked that no admin password existed; its POST did not,
so anyone who could reach the server could submit it and replace the admin
password (and the default model settings) and be handed an admin session. And
"no admin password" was read through get_system, which returns "" for a row it
cannot decrypt: on a server whose encryption key had changed, setup reopened
to everyone.

Now setup runs only while no admin password row exists at all, the first
password is stored by one INSERT that only one request can win, the sign-in
page says why a password it cannot read fails, and a lost password is set from
the server with python -m admin.reset_password.

Every test runs the real routes and the real store on a scratch database of
its own. Settings are read back through admin.routes.store, the module the
routes wrote them with: some modules in this suite re-import store under a
different key file, and a value read through another copy fails to decrypt.
"""

from __future__ import annotations

import asyncio
import io
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store

    store.init_db()
    return path


def _request():
    request = MagicMock()
    request.url.scheme = "http"
    return request


def _setup(password, provider="anthropic"):
    from admin import routes

    with patch.object(routes, "_resp", side_effect=lambda request, name, ctx=None: {"page": name, **(ctx or {})}):
        return asyncio.run(routes.setup_submit(
            _request(), admin_password=password, anthropic_key="", openai_key="",
            default_provider=provider, default_model="m", kb_model="k"))


def _login(password):
    from admin import routes

    with patch.object(routes, "_resp", side_effect=lambda request, name, ctx=None: {"page": name, **(ctx or {})}):
        return asyncio.run(routes.login_submit(_request(), password=password))


def _cookie_set(response) -> bool:
    return any(k.lower() == b"set-cookie" for k, _ in getattr(response, "raw_headers", []))


def _unreadable_password_row():
    """A stored admin password encrypted under a key this server no longer has."""
    from store.db import get_db

    with get_db() as conn:
        conn.execute(
            "INSERT INTO system_config (key, value_encrypted) VALUES ('admin_password_hash', ?)",
            ("gAAAAABnot-a-token-this-key-can-read",))


class TestSetupRunsOnce:

    def test_the_first_setup_stores_the_password_and_signs_the_admin_in(self, fresh_store):
        from admin import credentials

        response = _setup("first-password")
        assert response.status_code == 303 and response.headers["location"] == "/admin"
        assert _cookie_set(response)
        assert credentials.verify("first-password")

    def test_a_second_setup_changes_nothing(self, fresh_store):
        from admin import credentials, routes

        _setup("first-password", provider="anthropic")
        response = _setup("attacker-password", provider="openai")
        assert response.status_code == 303 and response.headers["location"] == "/admin/login"
        assert not _cookie_set(response)
        assert credentials.verify("first-password") and not credentials.verify("attacker-password")
        assert routes.store.get_system("default_llm_provider") == "anthropic"

    def test_a_request_that_loses_the_race_changes_nothing(self, fresh_store):
        # Both requests passed the "is this the first run?" check before either
        # stored a password; the store lets only the first one land.
        from admin import credentials, routes

        _setup("first-password", provider="anthropic")
        with patch.object(routes, "_first_run", return_value=True):
            response = _setup("racing-password", provider="openai")
        assert response.headers["location"] == "/admin/login" and not _cookie_set(response)
        assert credentials.verify("first-password") and not credentials.verify("racing-password")
        assert routes.store.get_system("default_llm_provider") == "anthropic"

    def test_after_the_first_run_the_setup_form_is_never_shown_again(self, fresh_store):
        # Not even to say that a password is too short.
        _setup("first-password")
        response = _setup("short")
        assert response.status_code == 303 and response.headers["location"] == "/admin/login"

    def test_a_short_password_is_refused_on_the_first_run(self, fresh_store):
        from admin import credentials

        shown = _setup("short")
        assert shown["page"] == "setup.html" and "8 characters" in shown["error"]
        assert not credentials.is_set()


class TestAPasswordThatCannotBeRead:

    def test_it_is_not_a_first_run(self, fresh_store):
        from admin import routes

        _unreadable_password_row()
        assert routes._first_run() is False
        page = asyncio.run(routes.setup_page(_request()))
        assert page.status_code == 303 and page.headers["location"] == "/admin"

    def test_setup_cannot_replace_it(self, fresh_store):
        from admin import credentials

        _unreadable_password_row()
        response = _setup("attacker-password")
        assert response.headers["location"] == "/admin/login" and not _cookie_set(response)
        assert not credentials.verify("attacker-password")

    def test_the_sign_in_page_says_why_and_how_to_fix_it(self, fresh_store):
        _unreadable_password_row()
        shown = _login("anything-at-all")
        assert shown["page"] == "login.html"
        assert "can't be read" in shown["error"] and "python -m admin.reset_password" in shown["error"]


class TestSignIn:

    def test_with_no_password_yet_it_sends_the_admin_to_setup(self, fresh_store):
        response = _login("anything-at-all")
        assert response.status_code == 303 and response.headers["location"] == "/admin/setup"

    def test_only_the_right_password_signs_in(self, fresh_store):
        _setup("first-password")
        assert _login("wrong-password")["error"] == "Incorrect password"
        response = _login("first-password")
        assert response.status_code == 303 and response.headers["location"] == "/admin"
        assert _cookie_set(response)


class TestTheServerSideReset:

    def _run(self, *answers, argv=(), stdin=None):
        from admin import reset_password

        replies = iter(answers)
        return reset_password.main(list(argv), prompt=lambda _label: next(replies), stdin=stdin)

    def test_it_replaces_a_password_that_cannot_be_read(self, fresh_store):
        from admin import credentials

        _unreadable_password_row()
        assert self._run("new-admin-password", "new-admin-password") == 0
        assert credentials.is_readable() and credentials.verify("new-admin-password")
        assert _login("new-admin-password").headers["location"] == "/admin"

    def test_two_different_answers_change_nothing(self, fresh_store):
        from admin import credentials

        _setup("first-password")
        assert self._run("new-admin-password", "something-else") == 1
        assert credentials.verify("first-password")

    def test_a_short_password_changes_nothing(self, fresh_store):
        from admin import credentials

        _setup("first-password")
        assert self._run("short", "short") == 1
        assert credentials.verify("first-password")

    def test_it_reads_one_line_from_standard_input(self, fresh_store):
        from admin import credentials

        assert self._run(argv=["--stdin"], stdin=io.StringIO("piped-password\n")) == 0
        assert credentials.verify("piped-password")


class TestTheStoreClaim:

    def test_only_the_first_claim_is_stored(self, fresh_store):
        import store

        assert store.claim_system_key("default_llm_model", "first") is True
        assert store.claim_system_key("default_llm_model", "second") is False
        assert store.get_system("default_llm_model") == "first"

    def test_a_row_counts_as_set_even_when_it_cannot_be_read(self, fresh_store):
        import store

        assert store.system_key_is_set("admin_password_hash") is False
        _unreadable_password_row()
        assert store.system_key_is_set("admin_password_hash") is True
        assert store.get_system("admin_password_hash") == ""
