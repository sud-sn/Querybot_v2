"""
Part C4: reconcile-against-a-known-number intent.

Regex tested directly; the ws_chat wiring and _run_reconcile_chat handler
are source-scanned (this codebase's established convention for logic
embedded in the large ws_chat handler -- see
tests/test_report_builder_chat_wiring.py).
"""

from pathlib import Path
import unittest

from gateway.webhooks import _RECONCILE_INTENT_RE

ROOT = Path(__file__).resolve().parents[1]


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
    def setUp(self):
        self.source = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")

    def test_reconcile_gate_requires_a_cached_result_with_sql(self):
        pos = self.source.index("_RECONCILE_INTENT_RE.search(text)")
        line_end = self.source.index("\n", pos)
        line = self.source[max(0, pos - 40):line_end]
        self.assertIn("_cache_snapshot", line)
        self.assertIn('_cache_snapshot.get("sql")', line)

    def test_reconcile_gate_precedes_metadata_result_question_routing(self):
        reconcile_pos = self.source.index("_RECONCILE_INTENT_RE.search(text)")
        metadata_pos = self.source.index("is_metadata_result_question(text)")
        self.assertLess(reconcile_pos, metadata_pos)

    def test_handler_never_promises_to_explain_the_gap(self):
        start = self.source.index("async def _run_reconcile_chat")
        end = self.source.index("\n    try:\n        while True:", start)
        block = self.source[start:end]
        # The promise is in the catalogue now, and a translation is exactly
        # where one could reappear -- so every shipped language is checked, not
        # just the source of the handler.
        self.assertIn('_t("reply.explain.body")', block)
        from core import i18n
        self.assertIn("can't explain the", i18n.t("reply.explain.body", lang="en"))
        for lang in i18n.SUPPORTED_LANGUAGES:
            said = i18n.t("reply.explain.body", lang=lang).lower()
            self.assertNotIn("i'll explain the difference", said, lang)
            self.assertNotIn("j'expliquerai", said, lang)

    def test_handler_restates_the_exact_sql_not_a_vague_description(self):
        start = self.source.index("async def _run_reconcile_chat")
        end = self.source.index("\n    try:\n        while True:", start)
        block = self.source[start:end]
        self.assertIn('"secondary": sql', block)

    def test_handler_offers_exactly_two_testable_hypotheses(self):
        start = self.source.index("async def _run_reconcile_chat")
        end = self.source.index("\n    try:\n        while True:", start)
        block = self.source[start:end]
        self.assertIn('"follow_up_suggestions": [', block)
        self.assertIn("excluding internal or administrative records", block)
        self.assertIn("using last calendar month instead", block)


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
