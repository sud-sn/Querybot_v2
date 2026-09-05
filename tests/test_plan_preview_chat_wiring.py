"""
gateway/webhooks.py -- the opt-in "explain your plan first" preview intent
gates. Regexes are tested directly; the handler's source is scanned for
correct precedence (this codebase's established convention -- see
tests/test_report_builder_chat_wiring.py).
"""

from pathlib import Path
import unittest

from gateway.webhooks import _PLAN_PREVIEW_INTENT_RE, _PLAN_PREVIEW_CONFIRM_RE

ROOT = Path(__file__).resolve().parents[1]


class PlanPreviewIntentRegexTests(unittest.TestCase):
    def test_matches_trigger_phrasings_with_question(self):
        cases = {
            "explain your plan: what was net revenue last week": "what was net revenue last week",
            "explain plan before running: top 5 customers": "top 5 customers",
            "show me your plan first, revenue by region": "revenue by region",
            "tell me your plan: net revenue": "net revenue",
        }
        for text, expected_question in cases.items():
            match = _PLAN_PREVIEW_INTENT_RE.match(text)
            self.assertIsNotNone(match, text)
            self.assertEqual(match.group("question").strip(), expected_question)

    def test_trigger_with_no_question_captures_empty(self):
        match = _PLAN_PREVIEW_INTENT_RE.match("explain your plan")
        self.assertIsNotNone(match)
        self.assertEqual(match.group("question").strip(), "")

    def test_does_not_match_unrelated_text(self):
        for text in [
            "what was net revenue last week",
            "show me the report",
            "explain the drop",
        ]:
            self.assertIsNone(_PLAN_PREVIEW_INTENT_RE.match(text), text)


class PlanPreviewConfirmRegexTests(unittest.TestCase):
    def test_matches_confirmation_phrasings(self):
        for text in ["go ahead", "yes", "yes!", "ok", "okay", "sure", "proceed", "run it", "do it"]:
            self.assertTrue(_PLAN_PREVIEW_CONFIRM_RE.match(text), text)

    def test_does_not_match_a_correction(self):
        for text in [
            "no, use orders_v2 instead",
            "exclude internal accounts",
            "actually use last month",
        ]:
            self.assertFalse(bool(_PLAN_PREVIEW_CONFIRM_RE.match(text)), text)


class PlanPreviewChatWiringTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")

    def test_pending_preview_gate_precedes_report_builder_and_result_command_gates(self):
        pending_check_pos = self.source.index("pending_plan_previews.get(account_id, _pp_session_id)")
        report_gate_pos = self.source.index("_REPORT_BUILDER_INTENT_RE.search(text)")
        result_cmd_pos = self.source.index("result_command = parse_result_command(text)")
        self.assertLess(pending_check_pos, report_gate_pos)
        self.assertLess(pending_check_pos, result_cmd_pos)

    def test_confirmation_runs_the_original_question_unmodified(self):
        start = self.source.index("_pending_preview = pending_plan_previews.get")
        end = self.source.index("_pp_match = _PLAN_PREVIEW_INTENT_RE.match(text)")
        block = self.source[start:end]
        self.assertIn("_PLAN_PREVIEW_CONFIRM_RE.match(text)", block)
        self.assertIn("_pending_preview.question", block)
        self.assertIn('f"{_pending_preview.question} -- {text}"', block)

    def test_new_preview_is_stored_before_asking_for_confirmation(self):
        start = self.source.index("_pp_match = _PLAN_PREVIEW_INTENT_RE.match(text)")
        end = self.source.index("_REPORT_BUILDER_INTENT_RE.search(text)")
        block = self.source[start:end]
        set_pos = block.index("pending_plan_previews.set(")
        # Anchored on the message id rather than the English sentence: the
        # ordering is the invariant, and the sentence moved to the catalogue.
        send_pos = block.index('_t("reply.plan.preview_suffix"')
        from core import i18n
        assert 'Say "go ahead"' in i18n.t("reply.plan.preview_suffix",
                                          lang="en", summary="")
        self.assertLess(set_pos, send_pos)
        self.assertIn("build_plan_preview(", block)

    def test_never_calls_the_llm_or_executes_sql_to_build_the_preview(self):
        start = self.source.index("_pp_match = _PLAN_PREVIEW_INTENT_RE.match(text)")
        end = self.source.index("_REPORT_BUILDER_INTENT_RE.search(text)")
        block = self.source[start:end]
        self.assertNotIn("llm_complete", block)
        self.assertNotIn("execute_governed_query", block)


if __name__ == "__main__":
    unittest.main()
