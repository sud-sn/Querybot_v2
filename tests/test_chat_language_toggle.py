# -*- coding: utf-8 -*-
"""The language toggle in the chat header.

The switcher already existed in the sidebar footer — below Settings and
Logout, and the first thing a collapsed sidebar hides. On the one page a
reader actually spends their time, it was effectively not there.

The tests that matter are the two that are easy to get wrong: switching must
not lose the thread the reader is in, and the toggle must offer the language
they are NOT currently reading.
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

CHAT = (Path(__file__).resolve().parents[1]
        / "portal" / "templates" / "portal_chat.html").read_text(encoding="utf-8")


class _SignedIn(unittest.TestCase):

    def setUp(self):
        import store

        self._dir = tempfile.mkdtemp(prefix="qb-toggle-")
        self._saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(self._dir, "d.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        store.init_db()
        self.account_id = f"acct-tog-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")
        self.email = "camille@example.test"
        self.password = "Correct-Horse-9"
        self.user_id, _ = store.create_user(
            account_id=self.account_id, name="Camille", email=self.email,
            password=self.password, role="analyst")

        import main

        self.client = TestClient(main.app)
        self.client.post("/portal/login",
                         data={"account_id": self.account_id, "email": self.email,
                               "password": self.password}, follow_redirects=False)

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._dir, ignore_errors=True)


class TestTheToggleIsInTheHeader(unittest.TestCase):
    """Markup-level, because placement is the whole request."""

    def test_it_sits_in_the_header_beside_the_status_and_history(self):
        header = CHAT[CHAT.index('<div class="chat-shell-header">'):]
        header = header[:header.index("{% if available_schemas")]
        self.assertIn("chat-lang-switch", header)
        # Last in the right-hand group, so it lands in the corner.
        self.assertGreater(header.index("chat-lang-switch"),
                           header.index('id="historyToggleBtn"'))

    def test_it_posts_the_same_endpoint_the_sidebar_does(self):
        self.assertIn('action="/portal/api/language"', CHAT)

    def test_it_carries_the_page_it_was_on(self):
        # Without this the reader is dropped back to a fresh thread, which is
        # a worse outcome than the untranslated page they started with.
        block = CHAT[CHAT.index("chat-lang-switch"):]
        block = block[:block.index("</form>")]
        self.assertIn('name="next" value="{{ current_portal_path }}"', block)

    def test_each_option_is_a_real_submit_carrying_its_code(self):
        block = CHAT[CHAT.index("chat-lang-switch"):]
        block = block[:block.index("</form>")]
        self.assertIn('type="submit" name="lang" value="{{ option.code }}"', block)

    def test_the_endonym_is_marked_with_its_own_language(self):
        # Or a screen reader reads "FR" with English phonemes.
        block = CHAT[CHAT.index("chat-lang-switch"):]
        block = block[:block.index("</form>")]
        self.assertIn('lang="{{ option.code }}"', block)

    def test_it_is_styled_rather_than_inheriting_the_sidebar_rules(self):
        # The sidebar switcher's styling comes from the nav links beside it;
        # rendered in the header it would be invisible.
        self.assertIn(".chat-lang-option{", CHAT)


class TestTheToggleWorksEndToEnd(_SignedIn):

    def test_the_chat_page_renders_it(self):
        page = self.client.get("/portal/chat").text
        self.assertIn("chat-lang-switch", page)

    def test_it_offers_the_language_the_reader_is_not_reading(self):
        english = self.client.get("/portal/chat").text
        block = english[english.index("chat-lang-switch"):]
        block = block[:block.index("</form>")]
        self.assertIn('value="fr"', block)
        self.assertNotIn('value="en"', block)

    def test_and_the_other_way_round_once_french(self):
        self.client.post("/portal/api/language",
                         data={"lang": "fr", "next": "/portal/chat"},
                         follow_redirects=False)
        french = self.client.get("/portal/chat").text
        block = french[french.index("chat-lang-switch"):]
        block = block[:block.index("</form>")]
        self.assertIn('value="en"', block)
        self.assertNotIn('value="fr"', block)

    def test_switching_actually_translates_the_page(self):
        english = self.client.get("/portal/chat").text
        self.client.post("/portal/api/language",
                         data={"lang": "fr", "next": "/portal/chat"},
                         follow_redirects=False)
        french = self.client.get("/portal/chat").text
        self.assertNotEqual(english, french)
        self.assertIn('lang="fr"', french)

    def test_switching_returns_the_reader_to_the_thread_they_were_in(self):
        response = self.client.post(
            "/portal/api/language",
            data={"lang": "fr", "next": "/portal/chat?thread_id=abc123"},
            follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertIn("thread_id=abc123", response.headers["location"])

    def test_it_writes_the_row_as_well_as_the_cookie(self):
        # The row is what the ANSWER pipeline reads; the cookie is the chrome.
        import store

        self.client.post("/portal/api/language",
                         data={"lang": "fr", "next": "/portal/chat"},
                         follow_redirects=False)
        self.assertEqual(store.get_user(self.user_id)["lang"], "fr")
        self.assertEqual(self.client.cookies.get("qb_lang"), "fr")

    def test_a_crafted_return_path_cannot_send_the_reader_off_site(self):
        response = self.client.post(
            "/portal/api/language",
            data={"lang": "fr", "next": "https://evil.example/"},
            follow_redirects=False)
        self.assertNotIn("evil.example", response.headers.get("location", ""))


if __name__ == "__main__":
    unittest.main()
