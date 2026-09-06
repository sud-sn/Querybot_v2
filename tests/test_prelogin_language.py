# -*- coding: utf-8 -*-
"""A French customer can reach French before they have an account.

The login and registration screens are the first thing a customer ever sees,
and they were the only screens in the product with no way to choose a
language: the whole switcher lived inside `{% if user %}`. A French speaker on
a browser that asks for English in first position had no way out of English,
and the cookie the pre-auth pages read had nothing to set it.

Signing in then threw away whatever they had managed to set. Login refreshed
the cookie from `user.lang`, and a new account's row has none, so it
normalised to "en" and flipped the reader back to English at the moment they
authenticated.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402


class _Portal(unittest.TestCase):

    def setUp(self):
        import store

        self._dir = tempfile.mkdtemp(prefix="qb-prelogin-")
        self._saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(self._dir, "d.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        store.init_db()
        self.account_id = f"acct-pre-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")
        self.email = "camille@example.test"
        self.password = "Correct-Horse-9"
        # create_user returns (id, plain_password).
        self.user_id, _ = store.create_user(
            account_id=self.account_id, name="Camille", email=self.email,
            password=self.password, role="analyst",
        )
        import main

        self.client = TestClient(main.app)

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._dir, ignore_errors=True)

    def _login(self, client=None):
        return (client or self.client).post(
            "/portal/login",
            data={"account_id": self.account_id, "email": self.email,
                  "password": self.password},
            follow_redirects=False,
        )


class TestTheSignedOutPagesOfferTheChoice(_Portal):

    def test_the_login_page_carries_a_language_switcher(self):
        page = self.client.get("/portal/login").text
        self.assertIn("portal-lang-switch-bare", page)
        self.assertIn("/portal/api/language", page)

    def test_a_visitor_can_set_the_language_without_an_account(self):
        # This used to be a 401, which is what made the switcher pointless
        # before sign-in even if a page had carried one.
        response = self.client.post(
            "/portal/api/language",
            data={"lang": "fr", "next": "/portal/login"},
            follow_redirects=False,
        )
        self.assertIn(response.status_code, (200, 303))
        self.assertEqual(response.cookies.get("qb_lang"), "fr")

    def test_the_choice_actually_renders_the_login_page_in_french(self):
        client = TestClient(self.client.app)
        client.post("/portal/api/language", data={"lang": "fr", "next": "/portal/login"},
                    follow_redirects=False)
        french = client.get("/portal/login").text
        english = TestClient(self.client.app).get("/portal/login").text
        self.assertNotEqual(french, english)
        self.assertIn('lang="fr"', french)

    def test_an_unknown_language_falls_back_rather_than_breaking(self):
        response = self.client.post(
            "/portal/api/language", data={"lang": "klingon", "next": "/portal/login"},
            follow_redirects=False)
        self.assertIn(response.status_code, (200, 303))
        self.assertEqual(self.client.get("/portal/login").status_code, 200)

    def test_the_redirect_target_is_still_confined_to_the_portal(self):
        # The switcher posts the page it was on, and that field is
        # attacker-reachable whether or not anyone is signed in.
        response = self.client.post(
            "/portal/api/language",
            data={"lang": "fr", "next": "https://evil.example/"},
            follow_redirects=False)
        self.assertNotIn("evil.example", response.headers.get("location", ""))


class TestSigningInKeepsTheLanguageTheVisitorChose(_Portal):

    def test_a_new_account_adopts_the_language_read_on_the_way_in(self):
        client = TestClient(self.client.app)
        client.post("/portal/api/language", data={"lang": "fr", "next": "/portal/login"},
                    follow_redirects=False)
        self.assertEqual(client.cookies.get("qb_lang"), "fr")

        self._login(client)
        self.assertEqual(client.cookies.get("qb_lang"), "fr")

    def test_and_writes_it_back_so_it_follows_them_to_another_device(self):
        import store

        client = TestClient(self.client.app)
        client.post("/portal/api/language", data={"lang": "fr", "next": "/portal/login"},
                    follow_redirects=False)
        self._login(client)
        self.assertEqual(store.get_user(self.user_id).get("lang"), "fr")

    def test_a_stored_preference_still_wins_over_the_cookie(self):
        # The row is authoritative when it HAS a language: a preference set on
        # one device must follow the reader to the next, even onto a browser
        # whose cookie says otherwise.
        import store

        store.set_user_language(self.user_id, "fr")
        client = TestClient(self.client.app)
        client.post("/portal/api/language", data={"lang": "en", "next": "/portal/login"},
                    follow_redirects=False)
        self._login(client)
        self.assertEqual(client.cookies.get("qb_lang"), "fr")

    def test_the_forced_password_change_path_keeps_it_too(self):
        """The other login exit, and it carries the language separately.

        A user with a temporary password is redirected to the change-password
        screen instead of the dashboard. That branch sets the cookie on its own
        line, so it can drift from the main one — and a first-ever sign-in is
        exactly when a temporary password is in play.
        """
        import store

        client = TestClient(self.client.app)
        temp_email = "temp@example.test"
        temp_id, temp_password = store.create_user(
            account_id=self.account_id, name="Temp", email=temp_email, role="analyst")
        self.assertTrue(store.get_user_by_email(self.account_id, temp_email)["is_temp_pw"])

        client.post("/portal/api/language", data={"lang": "fr", "next": "/portal/login"},
                    follow_redirects=False)
        response = client.post(
            "/portal/login",
            data={"account_id": self.account_id, "email": temp_email,
                  "password": temp_password},
            follow_redirects=False)
        self.assertIn("change-password", response.headers.get("location", ""))
        self.assertEqual(client.cookies.get("qb_lang"), "fr")
        self.assertEqual(store.get_user(temp_id).get("lang"), "fr")

    def test_a_visitor_who_chose_nothing_is_unaffected(self):
        client = TestClient(self.client.app)
        self._login(client)
        self.assertEqual(client.cookies.get("qb_lang"), "en")


if __name__ == "__main__":
    unittest.main()


class TestTheDashboardFormatsForTheReader(_Portal):
    """The chrome was French around English numbers.

    portal/routes.py resolves the language for the TEMPLATE (via the
    _language_context processor) but nothing activated the ContextVar that
    every server-side formatter reads, so the dashboard's own Python — KPI
    magnitudes, tile values — formatted in English inside a French page.
    """

    def _compact(self, lang, value):
        from core.i18n import activate_language, deactivate_language
        from portal.routes import _compact_number

        token = activate_language(lang)
        try:
            return _compact_number(value)
        finally:
            deactivate_language(token)

    def test_a_grouped_magnitude_uses_the_readers_separator(self):
        """Grouping only ever appears on the compact path.

        Everything from 1,000 up is abbreviated, so the plain-count branch
        only ever sees values below a thousand and has no separator to get
        wrong. 999,999 rounds to "1 000k" — and "1,000K" reads as one-and-a-bit
        to a French reader.
        """
        self.assertEqual(self._compact("fr", 999_999), "1 000k")
        self.assertEqual(self._compact("en", 999_999), "1,000K")

    def test_a_count_below_a_thousand_is_written_plainly_in_both(self):
        for lang in ("en", "fr"):
            self.assertEqual(self._compact(lang, 999), "999")

    def test_a_compact_magnitude_uses_the_readers_decimal_separator(self):
        self.assertEqual(self._compact("fr", 1234), "1,2k")
        self.assertEqual(self._compact("en", 1234), "1.2K")

    def test_a_billion_is_not_abbreviated_b_in_french(self):
        # "B" is the French long-scale billion — a thousand times larger.
        self.assertEqual(self._compact("fr", 2_500_000_000), "2,5Md")
        self.assertEqual(self._compact("en", 2_500_000_000), "2.5B")

    def test_a_round_magnitude_keeps_no_decimal(self):
        self.assertEqual(self._compact("fr", 2_000_000), "2M")
        self.assertEqual(self._compact("en", 2_000_000), "2M")

    def test_a_negative_keeps_its_sign_in_both(self):
        for lang in ("en", "fr"):
            self.assertTrue(self._compact(lang, -1500).startswith("-"))

    def test_the_route_activates_the_language_around_its_own_formatting(self):
        import ast
        import inspect

        import portal.routes as pr

        source = inspect.getsource(pr.portal_dashboard)
        self.assertIn("activate_language", source)
        # And releases it: a leaked activation would answer the NEXT request
        # on this worker in the previous reader's language.
        tree = ast.parse(source.lstrip())
        self.assertTrue(any(isinstance(n, ast.Try) and n.finalbody
                            for n in ast.walk(tree)),
                        "the activation is not released in a finally")
        self.assertIn("deactivate_language", source)
