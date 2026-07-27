"""
core/dispatcher.py — the report front door (_REPORT_ASK_RE, _clean_report_name,
_handle_report_ask). Deterministic, pre-LLM resolution of "what's my report?"
style messages into a named report (store/report_store.py) + rendered reply.

The regex is deliberately anchored to the WHOLE trimmed message — a real
analytical question that happens to use the word "report" ("give me a
report of sales by region") must never be hijacked into this feature, so
those cases are asserted as non-matches, not just "handled gracefully".
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import core.dispatcher as dispatcher


def _arun(coro):
    return asyncio.run(coro)


class ReportAskRegexMatchTests(unittest.TestCase):

    def _matches(self, text: str) -> bool:
        return bool(dispatcher._REPORT_ASK_RE.match(text.strip()))

    def test_bare_report(self):
        self.assertTrue(self._matches("report"))

    def test_my_report(self):
        self.assertTrue(self._matches("my report"))

    def test_todays_report_with_apostrophe(self):
        self.assertTrue(self._matches("today's report"))

    def test_todays_report_without_apostrophe(self):
        self.assertTrue(self._matches("todays report"))

    def test_show_me_the_report(self):
        self.assertTrue(self._matches("show me the report"))

    def test_named_report(self):
        self.assertTrue(self._matches("ops report"))

    def test_the_named_report(self):
        self.assertTrue(self._matches("the daily ops report"))

    def test_whats_my_report_with_question_mark(self):
        self.assertTrue(self._matches("what's my report?"))

    def test_case_insensitive(self):
        self.assertTrue(self._matches("MY REPORT"))


class ReportAskRegexNonMatchTests(unittest.TestCase):
    """These must NOT match — real data questions that merely mention
    'report' as a noun with trailing content, not a whole-message report ask."""

    def _matches(self, text: str) -> bool:
        return bool(dispatcher._REPORT_ASK_RE.match(text.strip()))

    def test_report_of_sales_by_region(self):
        self.assertFalse(self._matches("give me a report of total sales by region"))

    def test_report_with_trailing_qualifier(self):
        self.assertFalse(self._matches("sales report of Q1 2026"))

    def test_unrelated_question_with_report_word_mid_sentence(self):
        self.assertFalse(self._matches("can you generate a report for compliance audits last year"))

    def test_normal_data_question_no_report_word(self):
        self.assertFalse(self._matches("what is total revenue by region"))

    def test_report_as_verb_object_with_more_context(self):
        self.assertFalse(self._matches("report all late deliveries in March"))


class CleanReportNameTests(unittest.TestCase):

    def test_strips_filler_words(self):
        self.assertEqual(dispatcher._clean_report_name("can i get my"), "")

    def test_keeps_real_name_tokens(self):
        self.assertEqual(dispatcher._clean_report_name("ops"), "ops")

    def test_keeps_multi_word_name(self):
        self.assertEqual(dispatcher._clean_report_name("daily ops"), "daily ops")

    def test_empty_input(self):
        self.assertEqual(dispatcher._clean_report_name(""), "")

    def test_mixed_filler_and_real_name(self):
        self.assertEqual(dispatcher._clean_report_name("show me the sales"), "sales")


class HandleReportAskTests(unittest.TestCase):

    def _make_event_adapter(self):
        event = MagicMock()
        adapter = MagicMock()
        adapter.send_message = AsyncMock()
        adapter.send_chart = AsyncMock()
        return event, adapter

    def test_no_reports_configured_sends_setup_message(self):
        event, adapter = self._make_event_adapter()
        with patch("store.report_store.list_reports", return_value=[]):
            _arun(dispatcher._handle_report_ask("acct1", {"id": 1}, "my report", event, adapter))
        adapter.send_message.assert_called_once()
        self.assertIn("ask your admin", adapter.send_message.call_args[0][1].lower())

    def test_single_report_used_without_name(self):
        event, adapter = self._make_event_adapter()
        report = {"id": 1, "name": "Ops Report", "is_default": 0}
        with (
            patch("store.report_store.list_reports", return_value=[report]),
            patch("core.report_engine.build_report_response",
                  return_value={"ok": True, "message": "**Ops Report**", "items": []}) as mock_build,
        ):
            _arun(dispatcher._handle_report_ask("acct1", {"id": 1}, "my report", event, adapter))
        mock_build.assert_called_once_with("acct1", {"id": 1}, report)

    def test_named_report_resolved_by_name(self):
        event, adapter = self._make_event_adapter()
        reports = [{"id": 1, "name": "Ops Report", "is_default": 0}, {"id": 2, "name": "Sales Report", "is_default": 0}]
        with (
            patch("store.report_store.list_reports", return_value=reports),
            patch("store.report_store.get_report_by_name", return_value=reports[1]) as mock_get_by_name,
            patch("core.report_engine.build_report_response",
                  return_value={"ok": True, "message": "**Sales Report**", "items": []}) as mock_build,
        ):
            _arun(dispatcher._handle_report_ask("acct1", {"id": 1}, "sales report", event, adapter))
        mock_get_by_name.assert_called_once_with("acct1", "sales")
        mock_build.assert_called_once_with("acct1", {"id": 1}, reports[1])

    def test_named_report_not_found_lists_available(self):
        event, adapter = self._make_event_adapter()
        reports = [{"id": 1, "name": "Ops Report", "is_default": 0}]
        with (
            patch("store.report_store.list_reports", return_value=reports),
            patch("store.report_store.get_report_by_name", return_value=None),
        ):
            _arun(dispatcher._handle_report_ask("acct1", {"id": 1}, "nonexistent report", event, adapter))
        msg = adapter.send_message.call_args[0][1]
        self.assertIn("couldn't find", msg.lower())
        self.assertIn("Ops Report", msg)

    def test_multiple_reports_no_default_asks_to_pick(self):
        event, adapter = self._make_event_adapter()
        reports = [
            {"id": 1, "name": "Ops Report", "is_default": 0},
            {"id": 2, "name": "Sales Report", "is_default": 0},
        ]
        with patch("store.report_store.list_reports", return_value=reports):
            _arun(dispatcher._handle_report_ask("acct1", {"id": 1}, "my report", event, adapter))
        msg = adapter.send_message.call_args[0][1]
        self.assertIn("which report", msg.lower())
        self.assertIn("Ops Report", msg)
        self.assertIn("Sales Report", msg)

    def test_multiple_reports_with_default_uses_it(self):
        event, adapter = self._make_event_adapter()
        reports = [
            {"id": 1, "name": "Ops Report", "is_default": 0},
            {"id": 2, "name": "Sales Report", "is_default": 1},
        ]
        with (
            patch("store.report_store.list_reports", return_value=reports),
            patch("core.report_engine.build_report_response",
                  return_value={"ok": True, "message": "**Sales Report**", "items": []}) as mock_build,
        ):
            _arun(dispatcher._handle_report_ask("acct1", {"id": 1}, "my report", event, adapter))
        mock_build.assert_called_once_with("acct1", {"id": 1}, reports[1])

    def test_successful_response_sends_message_and_chart_per_item(self):
        event, adapter = self._make_event_adapter()
        report = {"id": 1, "name": "Ops Report", "is_default": 0}
        response = {
            "ok": True,
            "message": "**Ops Report**",
            "items": [
                {"text": "**Revenue**: 1,000", "chart": None},
                {"text": "**Trend** — see chart below.", "chart": {"chart_type": "bar"}},
            ],
        }
        with (
            patch("store.report_store.list_reports", return_value=[report]),
            patch("core.report_engine.build_report_response", return_value=response),
        ):
            _arun(dispatcher._handle_report_ask("acct1", {"id": 1}, "my report", event, adapter))

        # header + 2 item texts = 3 send_message calls
        self.assertEqual(adapter.send_message.call_count, 3)
        adapter.send_chart.assert_called_once_with(event, {"chart_type": "bar"})

    def test_not_ok_response_sends_message_only_no_items(self):
        event, adapter = self._make_event_adapter()
        report = {"id": 1, "name": "Empty Report", "is_default": 0}
        response = {"ok": False, "message": "The report has no metrics.", "items": []}
        with (
            patch("store.report_store.list_reports", return_value=[report]),
            patch("core.report_engine.build_report_response", return_value=response),
        ):
            _arun(dispatcher._handle_report_ask("acct1", {"id": 1}, "my report", event, adapter))
        adapter.send_message.assert_called_once()
        adapter.send_chart.assert_not_called()


if __name__ == "__main__":
    unittest.main()
