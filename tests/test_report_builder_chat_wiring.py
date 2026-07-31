"""
gateway/webhooks.py -- the conversational report/playbook builder intent
gate and _run_report_builder_chat. The regex is tested directly; the
handler's source is scanned for correct precedence and store-call wiring
(this codebase's established convention for verifying logic embedded in
the large ws_chat handler -- see tests/test_metadata_result_planner.py's
MetadataResultPlannerWiringTests).
"""

from pathlib import Path
import unittest

from gateway.webhooks import _REPORT_BUILDER_INTENT_RE

ROOT = Path(__file__).resolve().parents[1]


class ReportBuilderIntentRegexTests(unittest.TestCase):
    def test_matches_build_and_create_and_schedule_phrasings(self):
        for text in [
            "build a report with net revenue",
            "build me a report",
            "create a report for weekly ops",
            "make a report of top customers",
            "set up a report",
            "schedule a report every Monday",
            "schedule me a report",
        ]:
            self.assertTrue(_REPORT_BUILDER_INTENT_RE.search(text), text)

    def test_does_not_match_unrelated_report_mentions(self):
        for text in [
            "what is the report showing",
            "show me my report",
            "delete the report",
            "what was net revenue for last 7 days",
        ]:
            self.assertFalse(_REPORT_BUILDER_INTENT_RE.search(text), text)


class ReportBuilderChatWiringTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")

    def test_report_builder_gate_precedes_result_command_parse(self):
        gate_pos = self.source.index("_REPORT_BUILDER_INTENT_RE.search(text)")
        result_cmd_pos = self.source.index("result_command = parse_result_command(text)")
        self.assertLess(gate_pos, result_cmd_pos)

    def test_handler_uses_the_same_store_calls_as_the_checkbox_form(self):
        start = self.source.index("async def _run_report_builder_chat")
        end = self.source.index("\n    try:\n        while True:", start)
        block = self.source[start:end]
        self.assertIn("report_store.create_report(", block)
        self.assertIn("report_store.add_metric_to_report(", block)
        self.assertIn("report_store.create_subscription(", block)
        self.assertIn("store.get_allowed_tables(portal_user)", block)

    def test_handler_gates_metrics_by_acl_before_sending_to_planner(self):
        start = self.source.index("async def _run_report_builder_chat")
        end = self.source.index("\n    try:\n        while True:", start)
        block = self.source[start:end]
        allowed_pos = block.index("store.get_allowed_tables(portal_user)")
        planner_pos = block.index("parse_report_plan(")
        self.assertLess(allowed_pos, planner_pos)

    def test_handler_never_creates_a_report_when_plan_is_none(self):
        start = self.source.index("async def _run_report_builder_chat")
        end = self.source.index("\n    try:\n        while True:", start)
        block = self.source[start:end]
        none_check_pos = block.index("if plan is None:")
        create_pos = block.index("report_store.create_report(")
        self.assertLess(none_check_pos, create_pos)


if __name__ == "__main__":
    unittest.main()
