"""
Part C4: reconcile-against-a-known-number intent.

The handler was source-scanned, on the stated grounds that _run_reconcile_chat
is a closure inside the large ws_chat handler and so cannot be reached. That is
not true -- tests/test_chat_socket_language.py connects a real TestClient
websocket to the real endpoint -- and the cost of believing it showed up
immediately: moving the card's copy into the message catalogue, which is what
the card needed, broke

    assertIn("excluding internal or administrative records", block)

while the invariant it is named for (the handler offers exactly two testable
hypotheses) was untouched. A grep for a literal cannot tell those two apart.

So the handler is driven over a websocket now, in both shipped languages, and
the assertions are on the frame the browser would actually receive.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_reconcile.db")
os.environ.setdefault("QUERYBOT_DB_PATH", _tmp_db)

from gateway.webhooks import _RECONCILE_INTENT_RE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _reconcile_frame(lang, *, cached=True, text="the real number is 1240, why is yours 1180"):
    """What the browser receives when a reader disputes the number.

    Returns the assistant_analysis frame, or None if the handler never ran --
    which is itself the assertion for the gate tests below.
    """
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    import portal.routes as pr
    import store
    from core.result_cache import result_cache
    from gateway import webhooks

    store.init_db()
    app = FastAPI()
    app.include_router(webhooks.router)
    client = TestClient(app)

    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    store.update_client_meta(account_id, chat_ui_enabled=1)
    user_id, _ = store.create_user(account_id, "Ada", f"{os.urandom(4).hex()}@x.com")
    if lang:
        store.set_user_language(user_id, lang)

    session_id = f"{account_id}:web_{user_id}:thread:t1"
    if cached:
        result_cache.store(session_id, [{"REVENUE": 1180.0}],
                           question="net revenue last week",
                           sql="SELECT SUM(REVENUE) FROM sales")
    client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
    try:
        with client.websocket_connect(f"/ws/chat/{account_id}?thread_id=t1") as ws:
            ws.receive_json()                      # greeting
            ws.send_json({"type": "message", "text": text})
            for _ in range(12):
                frame = ws.receive_json()
                if (frame.get("type") == "assistant_analysis"
                        and frame.get("action") == "reconcile"):
                    return frame
                if frame.get("type") == "typing" and not frame.get("active"):
                    return None
    finally:
        result_cache.clear(session_id)
    return None


class ReconcileIntentRegexTests(unittest.TestCase):
    def test_matches_common_reconciliation_phrasings(self):
        for text in [
            "the real number is 1240, why is yours 1180",
            "the actual number is different from what you gave me",
            "my dashboard shows 1,240 for this metric but you got 1,180",
            "our report shows a different total",
            "that doesn't match what I have",
            "yours says 500 but I expected more",
            "it should be 900 not 850",
            "can you walk me through how you calculated this",
        ]:
            self.assertTrue(_RECONCILE_INTENT_RE.search(text), text)

    def test_does_not_match_unrelated_questions(self):
        for text in [
            "what was net revenue for last 7 days",
            "show me a bar chart",
            "build a report with net revenue",
        ]:
            self.assertFalse(_RECONCILE_INTENT_RE.search(text), text)


class ReconcileChatWiringTests(unittest.TestCase):
    """Every assertion here is on a frame the handler actually sent."""

    def test_the_handler_runs_at_all(self):
        # The control. Without it, a handler that never fires satisfies every
        # "does not say X" assertion below.
        frame = _reconcile_frame("en")
        self.assertIsNotNone(frame, "the reconcile handler never ran")

    def test_it_needs_a_cached_result_with_sql(self):
        """Nothing to reconcile against means the card must not be offered --
        it would restate a definition it does not have."""
        self.assertIsNone(_reconcile_frame("en", cached=False))

    def test_it_wins_over_metadata_result_routing(self):
        """"can you walk me through how you calculated this" matches the
        reconcile pattern and reads as a metadata question. It has to reach
        the card that shows the SQL."""
        frame = _reconcile_frame(
            "en", text="can you walk me through how you calculated this")
        self.assertIsNotNone(frame)

    def test_it_never_promises_to_explain_the_gap(self):
        """QueryBot cannot see the methodology behind an externally-known
        number. Checked in every shipped language, because a translation is
        exactly where the promise could reappear."""
        from core import i18n

        for lang in i18n.SUPPORTED_LANGUAGES:
            with self.subTest(lang=lang):
                body = (_reconcile_frame(lang) or {}).get("body", "").lower()
                self.assertTrue(body)
                self.assertNotIn("i'll explain the difference", body)
                self.assertNotIn("j'expliquerai", body)
        self.assertIn("can't explain the", i18n.t("reply.explain.body", lang="en"))

    def test_it_restates_the_exact_sql_not_a_vague_description(self):
        frame = _reconcile_frame("en")
        self.assertEqual(frame["secondary"], "SELECT SUM(REVENUE) FROM sales")

    def test_it_offers_exactly_two_testable_hypotheses(self):
        chips = _reconcile_frame("en")["follow_up_suggestions"]
        self.assertEqual(len(chips), 2)
        # Each re-runs the reader's own question with one variable changed,
        # which is what makes it testable rather than a guess.
        for chip in chips:
            with self.subTest(chip=chip):
                self.assertTrue(chip["question"].startswith("net revenue last week,"))
        self.assertIn("excluding internal or administrative records",
                      chips[0]["question"])
        self.assertIn("using last calendar month instead", chips[1]["question"])

    def test_a_french_reader_gets_the_card_in_french(self):
        frame = _reconcile_frame("fr")
        self.assertIn("Ma valeur", " ".join(frame["bullets"]))
        self.assertIn("Question posée", " ".join(frame["bullets"]))
        self.assertIn("Lignes renvoyées", " ".join(frame["bullets"]))
        for chip in frame["follow_up_suggestions"]:
            with self.subTest(chip=chip):
                self.assertIn("en ", chip["label"])
                self.assertNotEqual(chip["label"], chip["question"])

    def test_but_the_hypotheses_still_re_plan_in_english(self):
        """The label/wire split. question_normalizer.canonicalise is a
        word-by-word lexicon, not a reverse lookup of the catalogue, so a
        French chip that carried its own text to the planner would depend on
        every word of it happening to be in that lexicon."""
        english = [c["question"] for c in
                   _reconcile_frame("en")["follow_up_suggestions"]]
        french = [c["question"] for c in
                  _reconcile_frame("fr")["follow_up_suggestions"]]
        self.assertEqual(english, french)


class ReconcileFrontendWiringTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / "portal" / "templates" / "portal_chat.html").read_text(encoding="utf-8")

    def test_analysis_cards_render_follow_up_suggestions_as_clickable_chips(self):
        start = self.source.index("function appendAnalysisResponse")
        end = self.source.index("function appendChart")
        block = self.source[start:end]
        self.assertIn("follow_up_suggestions", block)
        self.assertIn("follow-up-chip", block)
        self.assertIn("sendSuggestion(btn)", block)


if __name__ == "__main__":
    unittest.main()
