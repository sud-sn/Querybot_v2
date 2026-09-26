"""
No way into an account without its password.

Three doors stood open. The registration route was still live although nothing
issued its links any more, and for an email that already existed it signed the
visitor in as that user without a password. Accounts approved from Teams,
Slack or Zoom carried SHA-256("__platform_user__") -- a string in the source --
and the portal's old-hash fallback accepted it as a password, so anyone who
knew a platform id could sign in as that user. And there were three ways to
hash a password: unsalted SHA-256 compared with != for the admin, PBKDF2 at
200,000 rounds for the portal, and that fallback, which never retired a hash.

Now the registration route is gone, platform-approved accounts carry a marker
no password matches (existing rows are converted at startup, and their
synthetic email lowercased so an admin reset gives a usable password), both
consoles hash with one salted PBKDF2 at 600,000 rounds, and an older hash is
accepted once and replaced at that sign-in.

Real routes and store on a scratch database per test. The suite hashes with
fewer rounds for speed (tests/conftest.py); the production value is checked
here on a fresh copy of the module.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import os
from unittest.mock import MagicMock, patch

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


def _workspace(store):
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    return account_id


def _request():
    request = MagicMock()
    request.cookies = {}
    request.headers = {}
    request.query_params = {}
    request.url.scheme = "http"
    return request


def _sign_in(account_id, email, password):
    routes = _portal()
    with patch.object(routes, "_resp", side_effect=lambda request, name, ctx=None: {"page": name, **(ctx or {})}):
        return asyncio.run(routes.portal_login_submit(_request(), account_id=account_id,
                                                      email=email, password=password))


def _signed_in(response):
    return getattr(response, "status_code", None) == 303


def _set_hash(store, user_id, value):
    with store.get_db() as conn:
        conn.execute("UPDATE portal_user SET password_hash = ? WHERE id = ?", (value, user_id))


def _approved_platform_user(store, account_id, platform_id):
    store.upsert_pending_user(account_id=account_id, platform_type="slack",
                              platform_user_id=platform_id, display_name="Chat User",
                              conversation_ref="{}")
    pending = store.list_pending_users(account_id)[0]
    return store.approve_pending_user(pending["id"], account_id, None)


class TestRegistrationIsGone:

    def test_neither_the_page_nor_the_form_exists(self, fresh_store):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        app = FastAPI()
        app.include_router(_portal().router)
        client = TestClient(app)
        assert client.get("/portal/register?token=anything").status_code == 404
        response = client.post("/portal/register", data={
            "token": "anything", "name": "Mallory", "email": "victim@example.com",
            "password": "whatever-1", "confirm_pw": "whatever-1"})
        assert response.status_code == 404

    def test_the_login_page_no_longer_sends_people_to_registration_links(self, fresh_store):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        app = FastAPI()
        app.include_router(_portal().router)
        client = TestClient(app)
        english = client.get("/portal/login", headers={"accept-language": "en"}).text
        french = client.get("/portal/login", headers={"accept-language": "fr"}).text
        assert "No account yet? Ask your administrator to create one." in english
        assert "Pas encore de compte ? Demandez à votre administrateur d&#39;en créer un." in french
        assert "registration link" not in english and "lien d&#39;inscription" not in french


class TestPlatformApprovedAccounts:

    def test_they_have_no_password_at_all(self, fresh_store):
        from store import passwords

        account_id = _workspace(fresh_store)
        user = _approved_platform_user(fresh_store, account_id, "U0PLATFORM")
        assert user["password_hash"] == passwords.UNUSABLE
        assert user["email"] == "u0platform@platform.internal"
        for guess in ("__platform_user__", "", "!", passwords.UNUSABLE):
            assert not _signed_in(_sign_in(account_id, user["email"], guess))

    def test_the_old_placeholder_never_signs_in_and_is_retired_at_startup(self, fresh_store):
        from store import passwords

        account_id = _workspace(fresh_store)
        user = _approved_platform_user(fresh_store, account_id, "U0LEGACY")
        _set_hash(fresh_store, user["id"], passwords.LEGACY_PLATFORM_PLACEHOLDER)
        # Before startup converts it, the row signs in with nothing.
        assert not _signed_in(_sign_in(account_id, user["email"], "__platform_user__"))
        with fresh_store.get_db() as conn:
            conn.execute("UPDATE portal_user SET email = 'U0LEGACY@platform.internal' WHERE id = ?",
                         (user["id"],))
        fresh_store.init_db()
        row = fresh_store.get_user(user["id"])
        assert row["password_hash"] == passwords.UNUSABLE
        assert row["email"] == "u0legacy@platform.internal"

    def test_an_admin_reset_gives_them_a_password_that_works(self, fresh_store):
        account_id = _workspace(fresh_store)
        user = _approved_platform_user(fresh_store, account_id, "U0MIXEDcase")
        temporary = fresh_store.reset_user_password(user["id"])
        response = _sign_in(account_id, "U0MixedCase@Platform.Internal", temporary)
        assert _signed_in(response) and response.headers["location"] == "/portal/change-password"


class TestOlderHashes:

    def _user(self, store):
        account_id = _workspace(store)
        email = f"{os.urandom(4).hex()}@example.com"
        user_id, _ = store.create_user(account_id, "Ada", email, password="the-users-password")
        return account_id, user_id, email

    def test_an_unsalted_hash_signs_in_once_and_is_replaced(self, fresh_store):
        from store import passwords

        account_id, user_id, email = self._user(fresh_store)
        _set_hash(fresh_store, user_id, hashlib.sha256(b"the-users-password").hexdigest())
        version = fresh_store.get_user(user_id)["session_version"]
        assert _signed_in(_sign_in(account_id, email, "the-users-password"))
        stored = fresh_store.get_user(user_id)["password_hash"]
        assert stored.startswith(f"pbkdf2_sha256${passwords.ITERATIONS}$")
        assert not passwords.needs_rehash(stored)
        assert fresh_store.get_user(user_id)["session_version"] == version
        assert _signed_in(_sign_in(account_id, email, "the-users-password"))

    def test_a_hash_with_fewer_rounds_is_renewed(self, fresh_store):
        from store import passwords

        account_id, user_id, email = self._user(fresh_store)
        with patch.object(passwords, "ITERATIONS", passwords.ITERATIONS // 2):
            weaker = passwords.hash_password("the-users-password")
        _set_hash(fresh_store, user_id, weaker)
        assert _signed_in(_sign_in(account_id, email, "the-users-password"))
        assert fresh_store.get_user(user_id)["password_hash"].startswith(
            f"pbkdf2_sha256${passwords.ITERATIONS}$")

    def test_a_wrong_password_changes_nothing(self, fresh_store):
        account_id, user_id, email = self._user(fresh_store)
        legacy = hashlib.sha256(b"the-users-password").hexdigest()
        _set_hash(fresh_store, user_id, legacy)
        assert not _signed_in(_sign_in(account_id, email, "a-guess"))
        assert fresh_store.get_user(user_id)["password_hash"] == legacy


class TestTheAdminPassword:

    def _setup(self, password="the-admin-password"):
        from admin import routes

        with patch.object(routes, "_resp", side_effect=lambda request, name, ctx=None: {"page": name, **(ctx or {})}):
            request = MagicMock()
            request.url.scheme = "http"
            return asyncio.run(routes.setup_submit(
                request, admin_password=password, anthropic_key="", openai_key="",
                default_provider="anthropic", default_model="m", kb_model="k"))

    def _login(self, password):
        from admin import routes

        with patch.object(routes, "_resp", side_effect=lambda request, name, ctx=None: {"page": name, **(ctx or {})}):
            request = MagicMock()
            request.url.scheme = "http"
            return asyncio.run(routes.login_submit(request, password=password))

    def test_it_is_salted_and_slow(self, fresh_store):
        from admin import credentials, routes
        from store import passwords

        self._setup()
        stored = routes.store.get_system("admin_password_hash")
        assert stored.startswith(f"pbkdf2_sha256${passwords.ITERATIONS}$")
        assert credentials.verify("the-admin-password") and not credentials.verify("a-guess")

    def test_an_old_unsalted_one_signs_in_once_and_is_replaced_without_signing_anyone_out(self, fresh_store):
        from admin import credentials, routes

        self._setup()
        routes.store.set_system("admin_password_hash", hashlib.sha256(b"the-admin-password").hexdigest())
        version = credentials.session_version()
        assert self._login("the-admin-password").headers["location"] == "/admin"
        assert routes.store.get_system("admin_password_hash").startswith("pbkdf2_sha256$")
        assert credentials.session_version() == version
        assert self._login("the-admin-password").headers["location"] == "/admin"

    def test_break_glass_checks_the_same_password(self, fresh_store):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        from admin import routes

        self._setup()
        account_id = _workspace(fresh_store)
        app = FastAPI()
        app.include_router(routes.router)
        client = TestClient(app)
        with patch.object(routes, "_is_auth", return_value=True):
            refused = client.post(f"/admin/clients/{account_id}/compliance/break-glass",
                                  data={"password": "a-guess", "reason": "incident"},
                                  follow_redirects=False)
            granted = client.post(f"/admin/clients/{account_id}/compliance/break-glass",
                                  data={"password": "the-admin-password", "reason": "incident"},
                                  follow_redirects=False)
        assert "Re-authentication+failed" in refused.headers["location"]
        assert "Re-authentication+failed" not in granted.headers["location"]


class TestTheWorkFactor:

    def test_production_hashes_with_600000_rounds(self):
        # A fresh copy of the module: the suite lowers the rounds on the one
        # it imported (tests/conftest.py).
        import store.passwords as imported

        spec = importlib.util.spec_from_file_location("fresh_passwords", imported.__file__)
        fresh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fresh)
        assert fresh.ITERATIONS == 600_000
        stored = fresh.hash_password("a-password")
        assert stored.startswith("pbkdf2_sha256$600000$")
        assert fresh.verify_password(stored, "a-password") and not fresh.verify_password(stored, "b")

    def test_what_needs_renewing(self):
        from store import passwords

        assert passwords.needs_rehash(hashlib.sha256(b"x").hexdigest())
        assert passwords.needs_rehash(f"pbkdf2_sha256${passwords.ITERATIONS - 1}$salt$hex")
        assert not passwords.needs_rehash(passwords.hash_password("x"))
        assert not passwords.needs_rehash(passwords.UNUSABLE)
