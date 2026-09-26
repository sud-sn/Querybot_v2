"""
A password cannot be guessed at the speed of the server, and a failed sign-in
says nothing about what exists.

Neither console limited attempts. The portal's sign-in answered "Account ID
not found." for an unknown workspace -- telling anyone which workspaces exist
-- and "Invalid email or password." otherwise, in English whatever the page's
language, and an unknown email was refused before any password was hashed, so
timing told the same story. And POST /api/ask, which answers with no per-user
restrictions, was open whenever QUERYBOT_API_KEY was unset, which no
deployment file sets.

Now five failures in fifteen minutes are free and each further one doubles the
wait (a minute, up to fifteen), during which the password is not checked; a
success, a new password or an admin's reset clears it, and the server's
admin.reset_password clears the admin console's. Every failed portal sign-in
gets one message in the reader's language and costs one full password hash.
The question API is off until a key is set, and compares keys in constant time.

Real routes and store on a scratch database per test. The suite hashes with
fewer rounds (tests/conftest.py); "the same work" is counted in hash
derivations at whatever the rounds are.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import io
import os
from unittest.mock import patch

import pytest
from starlette.requests import Request


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    from portal import routes

    routes.store.init_db()
    return routes.store


def _workspace_with_user(store, password="the-right-password"):
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    email = f"{os.urandom(4).hex()}@example.com"
    user_id, _ = store.create_user(account_id, "Ada", email, password=password)
    return account_id, user_id, email


def _portal_client():
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from portal import routes

    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def _sign_in(client, account_id, email, password, lang="en"):
    return client.post("/portal/login", data={"account_id": account_id, "email": email, "password": password},
                       headers={"accept-language": lang}, follow_redirects=False)


def _says(response, text):
    return text in html.unescape(response.text)


class _Derivations:
    """Counts password-hash derivations, the unit of sign-in work."""

    def __enter__(self):
        from store import passwords

        self.rounds = []
        real = hashlib.pbkdf2_hmac

        def counting(name, password, salt, rounds, *a, **k):
            self.rounds.append(rounds)
            return real(name, password, salt, rounds, *a, **k)

        self._patch = patch.object(passwords.hashlib, "pbkdf2_hmac", counting)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()


# ── The throttle itself ──────────────────────────────────────────────────────

class TestTheWait:

    def test_five_failures_are_free_then_each_doubles_the_wait_up_to_fifteen_minutes(self, fresh_store):
        identity = fresh_store.sign_in_identity("portal", "acct", "ada@example.com")
        waits = [fresh_store.record_sign_in_failure(identity, "portal", now=1000.0) for _ in range(11)]
        assert waits == [0, 0, 0, 0, 0, 60, 120, 240, 480, 900, 900]
        assert fresh_store.sign_in_wait_seconds(identity, now=1000.0) == 900
        assert fresh_store.sign_in_wait_seconds(identity, now=1000.0 + 900) == 0

    def test_failures_older_than_fifteen_minutes_start_the_count_again(self, fresh_store):
        identity = fresh_store.sign_in_identity("portal", "acct", "ada@example.com")
        for _ in range(5):
            fresh_store.record_sign_in_failure(identity, "portal", now=1000.0)
        assert fresh_store.record_sign_in_failure(identity, "portal", now=1000.0 + 15 * 60) == 0

    def test_what_is_stored_is_a_hash_of_who_typed_what(self, fresh_store):
        identity = fresh_store.sign_in_identity("portal", "acct", "  Ada@Example.com ")
        assert identity == fresh_store.sign_in_identity("portal", "ACCT", "ada@example.com")
        fresh_store.record_sign_in_failure(identity, "portal")
        with fresh_store.get_db() as conn:
            rows = [dict(r) for r in conn.execute("SELECT * FROM sign_in_attempt")]
        assert len(rows) == 1 and "ada" not in str(rows).lower()


# ── The portal ───────────────────────────────────────────────────────────────

class TestThePortalSignIn:

    FAILED_EN = "The account ID, email or password is not right."
    FAILED_FR = "L'identifiant du compte, l'e-mail ou le mot de passe est incorrect."

    def test_every_failure_gets_the_same_answer_in_the_readers_language(self, fresh_store):
        account_id, _uid, email = _workspace_with_user(fresh_store)
        client = _portal_client()
        for lang, message in (("en", self.FAILED_EN), ("fr", self.FAILED_FR)):
            for attempt in ((f"no-such-{account_id}", email, "x"),
                            (account_id, "nobody@example.com", "x"),
                            (account_id, email, "a-wrong-password")):
                response = _sign_in(client, *attempt, lang=lang)
                assert response.status_code == 200 and _says(response, message), (lang, attempt)
                assert not _says(response, "Account ID not found")
            fresh_store.clear_sign_in_scope("portal")

    def test_every_failure_costs_one_full_password_hash(self, fresh_store):
        from store import passwords

        account_id, _uid, email = _workspace_with_user(fresh_store)
        fresh_store.upsert_pending_user(account_id=account_id, platform_type="slack",
                                        platform_user_id="U0PLAT", display_name="Chat User",
                                        conversation_ref="{}")
        pending = fresh_store.list_pending_users(account_id)[0]
        platform = fresh_store.approve_pending_user(pending["id"], account_id, None)
        passwords.spend("warm-up")  # the decoy hash is made once per process
        client = _portal_client()
        for attempt in ((f"no-such-{account_id}", email, "x"),
                        (account_id, "nobody@example.com", "x"),
                        (account_id, email, "a-wrong-password"),
                        (account_id, platform["email"], "x")):
            with _Derivations() as work:
                _sign_in(client, *attempt)
            assert work.rounds == [passwords.ITERATIONS], attempt

    def test_after_a_sixth_failure_even_the_right_password_waits_and_is_not_checked(self, fresh_store):
        account_id, _uid, email = _workspace_with_user(fresh_store)
        client = _portal_client()
        for _ in range(5):
            assert _sign_in(client, account_id, email, "a-wrong-password").status_code == 200
        response = _sign_in(client, account_id, email, "a-wrong-password", lang="fr")
        assert response.status_code == 429 and response.headers["retry-after"] == "60"
        assert _says(response, "Trop de tentatives. Patientez 1 min puis réessayez.")
        with _Derivations() as work:
            response = _sign_in(client, account_id, email, "the-right-password")
        assert response.status_code == 429 and work.rounds == []
        assert _says(response, "Too many attempts. Wait 1 min and try again.")

    def test_when_the_wait_is_over_the_right_password_signs_in_and_clears_the_count(self, fresh_store):
        account_id, _uid, email = _workspace_with_user(fresh_store)
        client = _portal_client()
        for _ in range(6):
            _sign_in(client, account_id, email, "a-wrong-password")
        with fresh_store.get_db() as conn:
            conn.execute("UPDATE sign_in_attempt SET locked_until = 1")
        assert _sign_in(client, account_id, email, "the-right-password").status_code == 303
        with fresh_store.get_db() as conn:
            assert conn.execute("SELECT COUNT(*) FROM sign_in_attempt").fetchone()[0] == 0

    def test_an_admin_reset_ends_the_wait(self, fresh_store):
        account_id, user_id, email = _workspace_with_user(fresh_store)
        client = _portal_client()
        for _ in range(6):
            _sign_in(client, account_id, email, "a-wrong-password")
        temporary = fresh_store.reset_user_password(user_id)
        response = _sign_in(client, account_id, email, temporary)
        assert response.status_code == 303 and response.headers["location"] == "/portal/change-password"

    def test_one_persons_wait_is_not_anothers(self, fresh_store):
        account_id, _uid, email = _workspace_with_user(fresh_store)
        other = f"{os.urandom(4).hex()}@example.com"
        fresh_store.create_user(account_id, "Grace", other, password="graces-password")
        client = _portal_client()
        for _ in range(6):
            _sign_in(client, account_id, email, "a-wrong-password")
        assert _sign_in(client, account_id, other, "graces-password").status_code == 303


# ── The admin console ────────────────────────────────────────────────────────

def _admin_login(password, host="203.0.113.7"):
    from admin import credentials, routes

    request = Request({"type": "http", "method": "POST", "path": "/admin/login", "root_path": "",
                       "scheme": "http", "query_string": b"", "headers": [],
                       "server": ("testserver", 80), "client": (host, 50000)})
    assert credentials.is_set()
    return asyncio.run(routes.login_submit(request, password=password))


class TestTheAdminSignIn:

    @pytest.fixture
    def admin_password(self, fresh_store):
        from admin import credentials

        credentials.set_password("the-admin-password")
        return "the-admin-password"

    def test_after_a_sixth_failure_the_password_waits(self, admin_password):
        for _ in range(5):
            assert _admin_login("a-guess").status_code == 200
        assert _admin_login("a-guess").status_code == 429
        with _Derivations() as work:
            response = _admin_login(admin_password)
        assert response.status_code == 429 and work.rounds == []
        assert b"Too many attempts. Wait 1 min and try again." in response.body

    def test_another_address_is_counted_on_its_own(self, admin_password):
        for _ in range(6):
            _admin_login("a-guess", host="198.51.100.9")
        assert _admin_login(admin_password, host="203.0.113.7").headers["location"] == "/admin"

    def test_the_servers_reset_ends_the_wait(self, admin_password):
        from admin import reset_password

        for _ in range(6):
            _admin_login("a-guess")
        assert reset_password.main(["--stdin"], stdin=io.StringIO("a-new-admin-password\n")) == 0
        assert _admin_login("a-new-admin-password").headers["location"] == "/admin"


# ── The question API ─────────────────────────────────────────────────────────

class TestTheQuestionApi:

    def _ask(self, body):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        from gateway import webhooks

        app = FastAPI()
        app.include_router(webhooks.router)
        with patch.object(webhooks, "handle_query") as pipeline:
            response = TestClient(app).post("/api/ask", json=body)
        return response, pipeline

    def test_it_is_off_until_a_key_is_set(self, fresh_store, monkeypatch):
        monkeypatch.delenv("QUERYBOT_API_KEY", raising=False)
        account_id, _uid, _email = _workspace_with_user(fresh_store)
        response, pipeline = self._ask({"question": "total sales", "account_id": account_id})
        assert response.status_code == 503 and not pipeline.called

    @pytest.mark.parametrize("key", ["", "a-guess", "the-api-key-but-longer"])
    def test_a_wrong_key_is_refused_before_anything_is_looked_up(self, fresh_store, monkeypatch, key):
        monkeypatch.setenv("QUERYBOT_API_KEY", "the-api-key")
        response, pipeline = self._ask({"question": "total sales", "account_id": "no-such-workspace",
                                        "api_key": key})
        assert response.status_code == 401 and not pipeline.called

    def test_the_right_key_is_answered(self, fresh_store, monkeypatch):
        monkeypatch.setenv("QUERYBOT_API_KEY", "the-api-key")
        account_id, _uid, _email = _workspace_with_user(fresh_store)
        response, pipeline = self._ask({"question": "total sales", "account_id": account_id,
                                        "api_key": "the-api-key"})
        assert response.status_code == 200 and pipeline.called


# ── The fields themselves ────────────────────────────────────────────────────

class _Fields:
    """Every input, keyed by name, with its attributes; and the page's scripts."""

    def __init__(self, markup):
        from html.parser import HTMLParser

        fields, scripts = {}, []

        class Parser(HTMLParser):
            in_script = False

            def handle_starttag(self, tag, attrs):
                attrs = {k: (v or "") for k, v in attrs}
                if tag == "input" and attrs.get("name"):
                    fields[attrs["name"]] = attrs
                if tag == "script":
                    self.in_script = True
                    scripts.append("")

            def handle_endtag(self, tag):
                if tag == "script":
                    self.in_script = False

            def handle_data(self, data):
                if self.in_script:
                    scripts[-1] += data

        Parser().feed(markup)
        self.fields, self.scripts, self.text = fields, scripts, markup


class TestThePasswordFields:

    def test_the_sign_in_form_tells_password_managers_what_each_field_is(self, fresh_store):
        page = _Fields(_portal_client().get("/portal/login").text)
        assert page.fields["email"]["autocomplete"] == "username"
        assert page.fields["password"]["autocomplete"] == "current-password"

    def _change_page(self, store, lang):
        account_id = f"acct{os.urandom(4).hex()}"
        store.upsert_client(account_id, "Test Ltd")
        email = f"{os.urandom(4).hex()}@example.com"
        user_id, temporary = store.create_user(account_id, "Ada", email)
        store.set_user_language(user_id, lang)
        client = _portal_client()
        assert _sign_in(client, account_id, email, temporary, lang=lang).status_code == 303
        return _Fields(client.get("/portal/change-password", headers={"accept-language": lang}).text)

    def test_a_new_password_is_marked_new_and_explained_in_the_readers_language(self, fresh_store):
        page = self._change_page(fresh_store, "fr")
        assert page.fields["new_pw"]["autocomplete"] == "new-password"
        assert page.fields["confirm_pw"]["autocomplete"] == "new-password"
        assert "Plus c'est long, plus c'est sûr" in html.unescape(page.text)

    def test_the_live_hint_says_what_the_server_will_check(self, fresh_store):
        import dukpy

        from tests.js_lift import function as lift

        page = self._change_page(fresh_store, "en")
        script = next(s for s in page.scripts if "function passwordHints(" in s)
        hints = dukpy.evaljs(lift(script, "function passwordHints(") + ";"
                             "[passwordHints('seven77', ''), passwordHints('eight888', 'eight888'),"
                             " passwordHints('eight888', 'eight88')]")
        # Seven characters is short and eight is enough, as the server says.
        assert hints == [{"longEnough": False, "match": ""}, {"longEnough": True, "match": "same"},
                         {"longEnough": True, "match": "differ"}]
        catalogue = next(s for s in page.scripts if "window.QB_I18N" in s)
        assert "The two passwords match." in catalogue and "The two passwords differ." in catalogue
