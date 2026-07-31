import json
import unittest

from core.report_planner import (
    build_report_plan_input,
    compile_report_plan_response,
    parse_report_plan,
)


class ReportPlanCompileTests(unittest.TestCase):
    def setUp(self):
        self.metrics = [
            {"id": 11, "name": "Net Revenue", "description": "Booked revenue net of returns"},
            {"id": 12, "name": "Top Customers", "description": "Top 10 customers by spend"},
        ]
        self.built = build_report_plan_input("build me a report", self.metrics)
        self.bindings = self.built.bindings

    def _raw(self, **plan):
        return json.dumps(plan)

    def test_happy_path_two_metrics_no_schedule(self):
        refs = list(self.bindings.keys())
        command, error = compile_report_plan_response(
            self._raw(operation="define_report", name="Weekly Ops", metric_refs=refs, confidence=0.92),
            self.bindings,
        )
        self.assertEqual(error, "")
        self.assertEqual(command.name, "Weekly Ops")
        self.assertEqual(set(command.metric_ids), {11, 12})
        self.assertEqual(command.cadence, "")
        self.assertEqual(command.confidence, 0.92)

    def test_happy_path_with_weekly_schedule(self):
        refs = list(self.bindings.keys())[:1]
        command, error = compile_report_plan_response(
            self._raw(
                operation="define_report", name="Revenue Digest", metric_refs=refs,
                cadence="WEEKLY", day_of_week=2, hour=9,
            ),
            self.bindings,
        )
        self.assertEqual(error, "")
        self.assertEqual(command.cadence, "weekly")
        self.assertEqual(command.day_of_week, 2)
        self.assertEqual(command.hour, 9)

    def test_rejects_metric_ref_not_in_bindings(self):
        command, error = compile_report_plan_response(
            self._raw(operation="define_report", name="X", metric_refs=["METRIC_REF_99"]),
            self.bindings,
        )
        self.assertIsNone(command)
        self.assertIn("not a metric available", error)

    def test_rejects_raw_metric_name_instead_of_ref(self):
        command, error = compile_report_plan_response(
            self._raw(operation="define_report", name="X", metric_refs=["Net Revenue"]),
            self.bindings,
        )
        self.assertIsNone(command)
        self.assertIn("not a metric available", error)

    def test_empty_metric_refs_rejected(self):
        command, error = compile_report_plan_response(
            self._raw(operation="define_report", name="X", metric_refs=[]),
            self.bindings,
        )
        self.assertIsNone(command)
        self.assertIn("No metrics were resolved", error)

    def test_unsupported_operation_rejected(self):
        command, error = compile_report_plan_response(
            self._raw(operation="unsupported"), self.bindings,
        )
        self.assertIsNone(command)
        self.assertIn("could not be built", error)

    def test_unknown_operation_rejected(self):
        command, error = compile_report_plan_response(
            self._raw(operation="delete_report"), self.bindings,
        )
        self.assertIsNone(command)
        self.assertIn("unsupported operation", error)

    def test_unsupported_fields_rejected(self):
        command, error = compile_report_plan_response(
            self._raw(operation="define_report", metric_refs=["METRIC_REF_1"], sql="DROP TABLE x"),
            self.bindings,
        )
        self.assertIsNone(command)
        self.assertIn("unsupported fields", error)

    def test_malformed_json_rejected(self):
        command, error = compile_report_plan_response("not json", self.bindings)
        self.assertIsNone(command)
        self.assertIn("valid JSON", error)

    def test_invalid_cadence_falls_back_to_no_schedule(self):
        command, _ = compile_report_plan_response(
            self._raw(operation="define_report", metric_refs=["METRIC_REF_1"], cadence="hourly"),
            self.bindings,
        )
        self.assertEqual(command.cadence, "")

    def test_out_of_range_hour_and_day_are_clamped(self):
        command, _ = compile_report_plan_response(
            self._raw(
                operation="define_report", metric_refs=["METRIC_REF_1"],
                cadence="weekly", day_of_week=99, hour=99,
            ),
            self.bindings,
        )
        self.assertEqual(command.day_of_week, 6)
        self.assertEqual(command.hour, 23)

    def test_duplicate_metric_refs_deduplicated(self):
        command, _ = compile_report_plan_response(
            self._raw(operation="define_report", metric_refs=["METRIC_REF_1", "METRIC_REF_1"]),
            self.bindings,
        )
        self.assertEqual(command.metric_ids, (11,))

    def test_confidence_defaults_low_when_missing(self):
        command, _ = compile_report_plan_response(
            self._raw(operation="define_report", metric_refs=["METRIC_REF_1"]),
            self.bindings,
        )
        self.assertEqual(command.confidence, 0.0)


class ReportPlanPromptTests(unittest.TestCase):
    def test_prompt_contains_only_metric_metadata_never_row_data(self):
        metrics = [{"id": 5, "name": "Net Revenue", "description": "Booked revenue"}]
        built = build_report_plan_input("weekly revenue report", metrics)
        combined = built.system_prompt + built.user_prompt
        self.assertIn("Net Revenue", combined)
        self.assertIn("Booked revenue", combined)
        self.assertIn("METRIC_REF_1", combined)
        self.assertNotIn("rows", combined.lower().split("metric manifest")[0])

    def test_metric_without_valid_id_is_skipped(self):
        metrics = [{"id": None, "name": "Broken"}, {"id": 7, "name": "Good"}]
        built = build_report_plan_input("report", metrics)
        self.assertEqual(list(built.bindings.values()), [7])


class ParseReportPlanAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_parse_uses_complete_and_compiles_result(self):
        metrics = [{"id": 5, "name": "Net Revenue", "description": ""}]
        captured = {}

        async def complete(**kwargs):
            captured.update(kwargs)
            return json.dumps({"operation": "define_report", "name": "R", "metric_refs": ["METRIC_REF_1"]}), 10, 10

        command, error = await parse_report_plan("build a revenue report", metrics, complete)
        self.assertEqual(error, "")
        self.assertEqual(command.metric_ids, (5,))
        self.assertIn("system", captured)
        self.assertIn("user", captured)

    async def test_parse_handles_complete_failure_gracefully(self):
        async def complete(**kwargs):
            raise RuntimeError("boom")

        command, error = await parse_report_plan("build a report", [], complete)
        self.assertIsNone(command)
        self.assertIn("unavailable", error)


if __name__ == "__main__":
    unittest.main()
