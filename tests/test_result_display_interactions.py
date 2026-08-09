import json
import tempfile
import unittest
from pathlib import Path

from core.response_builder import build_assistant_response, sanitize_response_text_fields
from core.result_cache import ResultCache
from core.result_commands import (
    compile_confirmed_result_presentation,
    execute_result_command,
    needs_result_reference_confirmation,
    parse_result_command,
)
from core.semantic_model import build_semantic_model


class ResultDisplayCommandTests(unittest.TestCase):
    def setUp(self):
        self.cache = ResultCache()
        self.cache.store(
            "session",
            [
                {"OrderMonth": "2026-01", "Revenue": 1234.5, "MarginPct": 0.25},
                {"OrderMonth": "2026-02", "Revenue": 900.0, "MarginPct": 0.4},
            ],
            question="revenue by month",
            sql="SELECT OrderMonth, Revenue, MarginPct FROM result",
            result_id="source",
        )

    def execute(self, question):
        command = parse_result_command(question)
        self.assertIsNotNone(command)
        return execute_result_command(
            "session", command, cache=self.cache, source_result_id="source",
        )

    def test_month_and_year_wording_creates_presentation_only_snapshot(self):
        outcome = self.execute("give it in just month and year")
        self.assertTrue(outcome.ok, outcome.message)
        self.assertEqual(outcome.snapshot["rows"][0]["OrderMonth"], "2026-01")
        self.assertEqual(outcome.snapshot["column_formats"]["OrderMonth"], "date")
        self.assertEqual(
            outcome.snapshot["metadata"]["display_formats"]["OrderMonth"],
            {"type": "date", "style": "month_year_short"},
        )
        self.assertTrue(outcome.snapshot["metadata"]["presentation_only"])

    def test_provide_date_as_month_and_year_formats_the_only_date_column(self):
        outcome = self.execute("provide the date as month and year")
        self.assertTrue(outcome.ok, outcome.message)
        self.assertEqual(
            outcome.snapshot["metadata"]["display_formats"]["OrderMonth"],
            {"type": "date", "style": "month_year_short"},
        )
        self.assertEqual(outcome.snapshot["rows"][0]["OrderMonth"], "2026-01")

    def test_explicit_above_result_reference_is_a_generic_format_target(self):
        outcome = self.execute("format the above result as USD currency")
        self.assertFalse(outcome.ok)
        self.assertTrue(outcome.clarification_required)
        self.assertIn("Which column", outcome.clarification_prompt)

    def test_ambiguous_presentation_request_confirms_only_with_cached_result(self):
        self.assertTrue(needs_result_reference_confirmation("change the format", True))
        self.assertFalse(needs_result_reference_confirmation("change the format", False))
        self.assertFalse(needs_result_reference_confirmation("what was revenue by date", True))
        self.assertFalse(
            needs_result_reference_confirmation(
                "provide the date as month and year", True,
            )
        )

    def test_confirmed_ambiguous_date_request_compiles_to_safe_partial_format(self):
        command = compile_confirmed_result_presentation("change the date format")
        self.assertEqual(command.action, "format")
        self.assertEqual(command.format_spec, {"type": "date"})
        outcome = execute_result_command(
            "session", command, cache=self.cache, source_result_id="source",
        )
        self.assertTrue(outcome.clarification_required)
        self.assertIn("Which date format", outcome.clarification_prompt)

    def test_currency_without_code_asks_instead_of_guessing(self):
        outcome = self.execute("format Revenue as currency")
        self.assertFalse(outcome.ok)
        self.assertTrue(outcome.clarification_required)
        self.assertIn("Which currency", outcome.clarification_prompt)
        self.assertEqual([item["label"] for item in outcome.clarification_options], ["USD", "INR", "EUR", "GBP"])

    def test_explicit_currency_and_decimal_precision_are_governed(self):
        outcome = self.execute("format Revenue as INR currency with 0 decimal places")
        self.assertTrue(outcome.ok, outcome.message)
        spec = outcome.snapshot["metadata"]["display_formats"]["Revenue"]
        self.assertEqual(spec["currency_code"], "INR")
        self.assertEqual(spec["fraction_digits"], 0)

    def test_percentage_scale_is_inferred_from_fraction_values(self):
        outcome = self.execute("show MarginPct as percentage")
        self.assertTrue(outcome.ok, outcome.message)
        spec = outcome.snapshot["metadata"]["display_formats"]["MarginPct"]
        self.assertEqual(spec["scale"], "fraction")

    def test_multiple_temporal_columns_require_target_clarification(self):
        cache = ResultCache()
        cache.store(
            "s", [{"OrderDate": "2026-01-02", "ShipDate": "2026-01-03"}],
            result_id="r",
        )
        outcome = execute_result_command(
            "s", parse_result_command("give it in month and year"),
            cache=cache, source_result_id="r",
        )
        self.assertTrue(outcome.clarification_required)
        self.assertIn("Which column", outcome.clarification_prompt)
        self.assertEqual(len(outcome.clarification_options), 2)

    def test_format_metadata_survives_later_local_transform(self):
        formatted = self.execute("format Revenue as INR currency")
        keep = execute_result_command(
            "session", parse_result_command("keep top 1"), cache=self.cache,
            source_result_id=formatted.derived_result_id,
        )
        self.assertTrue(keep.ok)
        self.assertEqual(
            keep.snapshot["metadata"]["display_formats"]["Revenue"]["currency_code"],
            "INR",
        )

    def test_multiple_format_changes_are_applied_to_one_derived_result(self):
        outcome = self.execute(
            "Format OrderMonth as MMM-YY and Revenue as USD currency "
            "with zero decimal places"
        )
        self.assertTrue(outcome.ok, outcome.message)
        formats = outcome.snapshot["metadata"]["display_formats"]
        self.assertEqual(
            formats["OrderMonth"],
            {"type": "date", "style": "month_year_short"},
        )
        self.assertEqual(formats["Revenue"]["currency_code"], "USD")
        self.assertEqual(formats["Revenue"]["fraction_digits"], 0)
        self.assertEqual(outcome.snapshot["rows"][0]["Revenue"], 1234.5)

    def test_to_wording_splits_independent_format_targets(self):
        command = parse_result_command(
            "change OrderMonth to month and year and Revenue to INR with 2 decimals"
        )
        self.assertIsNotNone(command)
        self.assertEqual(len(command.format_spec["batch"]), 2)
        self.assertEqual(command.format_spec["batch"][0]["target_text"], "OrderMonth")
        self.assertEqual(command.format_spec["batch"][1]["target_text"], "Revenue")

    def test_main_local_result_response_forwards_display_metadata(self):
        source = (
            Path(__file__).resolve().parents[1] / "gateway" / "webhooks.py"
        ).read_text(encoding="utf-8")
        self.assertIn("display_formats=display_formats", source)

    def test_portal_wires_result_reference_and_format_clarifications_locally(self):
        source = (
            Path(__file__).resolve().parents[1] / "gateway" / "webhooks.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"source": "result_reference_confirmation"', source)
        self.assertIn('"source": "local_result_command"', source)
        self.assertIn("Are you referring to the previous result?", source)
        self.assertIn("compile_confirmed_result_presentation", source)


class KpiPresentationTests(unittest.TestCase):
    def test_single_metric_is_kpi_not_diagnostic_table(self):
        payload = build_assistant_response(
            question="what is revenue",
            rows=[{"MatchedRows": 64, "NonNullRevenueRows": 64, "Revenue": 52677.25}],
            sql="SELECT diagnostics and revenue",
            duration_ms=20,
            column_formats={"Revenue": "currency"},
        )
        self.assertEqual(payload["data"]["headers"], ["Revenue"])
        self.assertEqual(payload["kpi"]["label"], "Revenue")
        self.assertEqual(payload["kpi"]["state"], "ready")
        self.assertEqual(payload["data"]["diagnostics"]["matched_rows"], 64)
        self.assertNotIn("MatchedRows", payload["data"]["rows"][0])

    def test_null_metric_is_missing_kpi_and_keeps_explanation(self):
        payload = build_assistant_response(
            question="what is revenue",
            rows=[{"MatchedRows": 64, "NonNullRevenueRows": 0, "Revenue": 0}],
            sql="SELECT diagnostics and revenue",
            duration_ms=20,
        )
        self.assertEqual(payload["kpi"]["state"], "missing")
        self.assertIn("all matched values are missing", payload["answer"]["headline"])

    def test_zero_match_does_not_render_false_zero_kpi(self):
        payload = build_assistant_response(
            question="what is revenue",
            rows=[{"MatchedRows": 0, "NonNullRevenueRows": 0, "Revenue": 0}],
            sql="SELECT diagnostics and revenue",
            duration_ms=20,
        )
        self.assertIsNone(payload["kpi"])
        self.assertIn("No matching data", payload["answer"]["headline"])

    def test_kpi_carries_currency_override_and_precision(self):
        payload = build_assistant_response(
            question="what is revenue", rows=[{"Revenue": 52677.25}],
            sql="SELECT revenue", duration_ms=20,
            column_formats={"Revenue": "currency"},
            display_formats={"Revenue": {
                "type": "currency", "currency_code": "INR",
                "fraction_digits": 0, "grouping": True,
            }},
        )
        self.assertEqual(payload["kpi"]["display_format"]["currency_code"], "INR")
        self.assertEqual(payload["kpi"]["display_format"]["fraction_digits"], 0)
        self.assertEqual(payload["answer"]["headline"], "Revenue: ₹52,677")
        self.assertEqual(payload["answer"]["short_value"], "₹52,677")
        self.assertEqual(payload["insight_summary"], "Revenue: ₹52,677.")

    def test_answer_text_uses_requested_percentage_scale_and_date_style(self):
        percentage = build_assistant_response(
            question="what is margin", rows=[{"Margin": 0.126}],
            sql="SELECT margin", duration_ms=20,
            column_formats={"Margin": "percentage"},
            display_formats={"Margin": {
                "type": "percentage", "scale": "fraction",
                "fraction_digits": 1, "grouping": True,
            }},
        )
        self.assertEqual(percentage["answer"]["short_value"], "12.6%")

        period = build_assistant_response(
            question="show the period", rows=[{"Period": "2026-08"}],
            sql="SELECT period", duration_ms=20,
            column_formats={"Period": "date"},
            display_formats={"Period": {"type": "date", "style": "month_year_short"}},
        )
        self.assertEqual(period["answer"]["short_value"], "Aug-26")


class DateRoleCoverageTests(unittest.TestCase):
    def test_native_dates_are_covered_and_encoded_candidates_are_reported(self):
        schema = {
            "OPS.F_EVENT": {
                "schema": "OPS", "table": "F_EVENT",
                "columns": [
                    {"name": "EVENT_ID", "type": "bigint"},
                    {"name": "OCCURRED_AT", "type": "datetime2"},
                    {"name": "LEGACY_PERIOD", "type": "varchar"},
                    {"name": "ETL_LOAD_DATE", "type": "varchar"},
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "_schema.json").write_text(json.dumps(schema), encoding="utf-8")
            model = build_semantic_model(tmp)
        coverage = model["date_role_coverage"]
        self.assertEqual(coverage["native_roles"], 1)
        self.assertEqual(coverage["review_candidate_count"], 1)
        self.assertEqual(coverage["review_candidates"][0]["column"], "LEGACY_PERIOD")
        self.assertEqual(coverage["technical_date_count"], 1)


class ResponseTextContractTests(unittest.TestCase):
    def test_structured_objects_cannot_enter_text_only_response_fields(self):
        payload = sanitize_response_text_fields({
            "answer": {
                "headline": {"bad": "object"},
                "short_value": ["safe", {"bad": "object"}],
            },
            "chart": {"series": [{"name": "Revenue", "data": [1, 2]}]},
            "next_actions": [{"label": {"nested": "object"}, "action": "compare"}],
        })
        self.assertEqual(payload["answer"]["headline"], "")
        self.assertEqual(payload["answer"]["short_value"], "safe")
        self.assertEqual(payload["next_actions"][0]["label"], "")
        self.assertEqual(payload["chart"]["series"][0]["data"], [1, 2])
        self.assertNotIn("[object Object]", json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
