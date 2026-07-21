from pathlib import Path
import unittest

from core.result_planner import (
    build_planner_input,
    compile_planner_response,
    is_metadata_result_question,
    plan_result_command,
)


ROOT = Path(__file__).resolve().parents[1]


class MetadataResultPlannerTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = {
            "row_count": 2,
            "schema": [
                {"name": "DOCTOR_NAME", "type": "TEXT"},
                {"name": "TOTAL_REVENUE", "type": "DOUBLE"},
                {"name": "TOTAL_COST", "type": "DOUBLE"},
            ],
            "column_formats": {
                "TOTAL_REVENUE": "currency",
                "TOTAL_COST": "currency",
            },
            "rows": [
                {"DOCTOR_NAME": "Dr. Priya Shah", "TOTAL_REVENUE": 1250.0, "TOTAL_COST": 800.0},
                {"DOCTOR_NAME": "Dr. Kiran Rao", "TOTAL_REVENUE": 900.0, "TOTAL_COST": 650.0},
            ],
            "sql": "SELECT raw_sensitive_value FROM governed_source",
        }

    def test_metadata_prompt_contains_no_rows_samples_sql_or_bound_values(self):
        planned = build_planner_input(
            "Filter this result where doctor name equals Dr. Priya Shah and revenue is above 1,250",
            self.snapshot,
        )
        combined = planned.system_prompt + planned.user_prompt
        self.assertNotIn("Priya", combined)
        self.assertNotIn("1,250", combined)
        self.assertNotIn("raw_sensitive_value", combined)
        self.assertNotIn("sample_values", combined)
        self.assertIn("VALUE_REF_", planned.user_prompt)
        self.assertIn("DOCTOR_NAME", planned.user_prompt)
        self.assertEqual(planned.metadata["rows_sent_to_llm"], 0)
        self.assertEqual(planned.metadata["sample_values_sent_to_llm"], 0)

    def test_filter_plan_uses_only_local_binding(self):
        planned = build_planner_input(
            "Filter this result where doctor name equals Dr. Priya Shah",
            self.snapshot,
        )
        value_ref = next(iter(planned.bindings))
        command, error = compile_planner_response(
            '{"operation":"filter","dimension":"DOCTOR_NAME",'
            f'"operator":"eq","value_ref":"{value_ref}"}}',
            self.snapshot,
            planned.bindings,
        )
        self.assertEqual(error, "")
        self.assertEqual(command.action, "filter")
        self.assertEqual(command.target_text, "DOCTOR_NAME")
        self.assertEqual(command.value_text, "Dr. Priya Shah")

    def test_raw_model_literal_is_rejected(self):
        command, error = compile_planner_response(
            '{"operation":"filter","dimension":"DOCTOR_NAME",'
            '"operator":"eq","value":"Dr. Priya Shah"}',
            self.snapshot,
            {},
        )
        self.assertIsNone(command)
        self.assertIn("unsupported fields", error)

    def test_unknown_column_and_unknown_operation_are_rejected(self):
        command, _ = compile_planner_response(
            '{"operation":"aggregate","dimension":"WAREHOUSE",'
            '"metric":"TOTAL_REVENUE","aggregation":"sum"}',
            self.snapshot,
            {},
        )
        self.assertIsNone(command)
        command, _ = compile_planner_response(
            '{"operation":"execute_sql","metric":"TOTAL_REVENUE"}',
            self.snapshot,
            {},
        )
        self.assertIsNone(command)

    def test_aggregate_and_ratio_plans_compile_to_local_commands(self):
        aggregate, error = compile_planner_response(
            '{"operation":"aggregate","dimension":"DOCTOR_NAME",'
            '"metric":"TOTAL_REVENUE","aggregation":"avg"}',
            self.snapshot,
            {},
        )
        self.assertEqual(error, "")
        self.assertEqual(aggregate.action, "aggregate")
        self.assertFalse(aggregate.fallback_allowed)

        ratio, error = compile_planner_response(
            '{"operation":"ratio","numerator":"TOTAL_REVENUE",'
            '"denominator":"TOTAL_COST"}',
            self.snapshot,
            {},
        )
        self.assertEqual(error, "")
        self.assertEqual(ratio.action, "ratio")

    def test_contribution_plan_resolves_business_names_to_cached_columns(self):
        snapshot = {
            "row_count": 3,
            "schema": [
                {"name": "PERIOD", "type": "TEXT"},
                {"name": "TOTAL_NET_REVENUE", "type": "DOUBLE"},
            ],
            "rows": [
                {"PERIOD": "2025-06", "TOTAL_NET_REVENUE": 778.0},
                {"PERIOD": "2025-03", "TOTAL_NET_REVENUE": 580.0},
                {"PERIOD": "2025-02", "TOTAL_NET_REVENUE": 566.75},
            ],
        }
        command, error = compile_planner_response(
            '{"operation":"contribution","dimension":"BOOKED_MONTH",'
            '"metric":"NET_REVENUE"}',
            snapshot,
            {},
        )
        self.assertEqual(error, "")
        self.assertEqual(command.action, "contribution")
        self.assertEqual(command.dimension_text, "PERIOD")
        self.assertEqual(command.metric_text, "TOTAL_NET_REVENUE")

    def test_route_requires_explicit_result_context(self):
        self.assertTrue(is_metadata_result_question("Group these results by doctor and average revenue"))
        self.assertFalse(is_metadata_result_question("What is average revenue by doctor?"))
        self.assertFalse(is_metadata_result_question("Explain this result"))


class MetadataResultPlannerAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_receives_only_metadata_and_sanitized_refs(self):
        snapshot = {
            "row_count": 1,
            "schema": [
                {"name": "DOCTOR_NAME", "type": "TEXT"},
                {"name": "REVENUE", "type": "DOUBLE"},
            ],
            "column_formats": {"REVENUE": "currency"},
            "rows": [{"DOCTOR_NAME": "Dr. Priya Shah", "REVENUE": 1250.0}],
        }
        captured = {}

        async def complete(**kwargs):
            captured.update(kwargs)
            return (
                '{"operation":"filter","dimension":"DOCTOR_NAME",'
                '"operator":"eq","value_ref":"VALUE_REF_1"}',
                10,
                10,
            )

        result = await plan_result_command(
            "Filter this result to Dr. Priya Shah",
            snapshot,
            complete,
        )
        self.assertTrue(result.ok, result.reason)
        sent = captured["system"] + captured["user"]
        self.assertNotIn("Priya", sent)
        self.assertNotIn("1250", sent)
        self.assertNotIn("rows\":", sent)
        self.assertEqual(result.metadata["rows_sent_to_llm"], 0)


class MetadataResultPlannerWiringTests(unittest.TestCase):
    def test_planner_route_precedes_insight_and_main_dispatch(self):
        source = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")
        planner_position = source.index("if is_metadata_result_question(text):")
        insight_position = source.index("from core.insight import is_insight_question", planner_position)
        self.assertLess(planner_position, insight_position)

    def test_planner_handler_records_zero_row_exposure(self):
        source = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")
        start = source.index("async def _run_metadata_result_planner")
        end = source.index("\n    try:\n        while True:", start)
        block = source[start:end]
        self.assertIn('component="result_metadata_planner"', block)
        self.assertIn("run_governed_result_followup(", block)
        self.assertNotIn("get_stats(", block)
        governed = (ROOT / "core" / "governed_result_followup.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"rows_sent_to_llm": 0', governed)
        self.assertIn('"sample_values_sent_to_llm": 0', governed)


if __name__ == "__main__":
    unittest.main()
