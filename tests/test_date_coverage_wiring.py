"""
Phase 1 (agent roadmap): date-range coverage-gap wiring.

core/result_renderer.py::_send_results is a large, heavily-mocked function
with no existing direct-invocation test convention in this suite (every
existing test of it is a source-scan -- see tests/test_result_cache_versioning.py,
tests/test_regulated_llm_boundary.py). This file follows that same
established convention for the new wiring, on top of full behavioral unit
tests for core/date_coverage.py itself (tests/test_date_coverage.py).
"""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ResultRendererCoverageWiringTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / "core" / "result_renderer.py").read_text(encoding="utf-8")

    def test_coverage_check_runs_before_both_rich_and_plaintext_paths(self):
        check_pos = self.source.index("check_date_coverage(")
        rich_pos = self.source.index('rich_sender = getattr(adapter, "send_assistant_response"')
        plaintext_pos = self.source.index("conf_text = format_success_confidence_text")
        self.assertLess(check_pos, rich_pos)
        self.assertLess(check_pos, plaintext_pos)

    def test_coverage_check_never_raises_uncaught(self):
        start = self.source.index("_temporal_policies = (confidence_context.get")
        end = self.source.index('rich_sender = getattr(adapter, "send_assistant_response"')
        block = self.source[start:end]
        self.assertIn("try:", block)
        self.assertIn("except Exception", block)

    def test_rich_response_gets_coverage_caveat_field(self):
        start = self.source.index('rich_sender = getattr(adapter, "send_assistant_response"')
        end = self.source.index("await rich_sender(event, response_payload)")
        block = self.source[start:end]
        self.assertIn('response_payload["coverage_caveat"] = coverage_caveat', block)

    def test_plaintext_reply_prepends_coverage_line_in_both_shapes(self):
        start = self.source.index("conf_text = format_success_confidence_text")
        end = self.source.index("await adapter.send_message(event, reply)")
        block = self.source[start:end]
        self.assertIn("coverage_line = ", block)
        # Both the single-cell and multi-row reply templates must include it.
        self.assertEqual(block.count("{coverage_line}"), 2)

    def test_uses_the_resolved_temporal_policy_not_a_fresh_parse(self):
        # Must reuse the semantic plan's already-resolved temporal_policies
        # (set by core/contextual_dates.py earlier in the pipeline), not
        # re-derive the window from the question text a second time.
        start = self.source.index("_temporal_policies = (confidence_context.get")
        end = self.source.index('rich_sender = getattr(adapter, "send_assistant_response"')
        block = self.source[start:end]
        self.assertIn('"temporal_policies"', block)
        self.assertIn("_temporal_policies[0]", block)


class FrontendCoverageCaveatWiringTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / "portal" / "templates" / "portal_chat.html").read_text(encoding="utf-8")

    def test_coverage_caveat_rendered_from_message_field(self):
        self.assertIn("msg.coverage_caveat", self.source)
        self.assertIn("coverageCaveatHtml", self.source)

    def test_coverage_caveat_slotted_into_the_message_template(self):
        self.assertIn("${coverageCaveatHtml}", self.source)


if __name__ == "__main__":
    unittest.main()
