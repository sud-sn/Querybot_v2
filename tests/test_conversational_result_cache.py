from pathlib import Path
import unittest

from core.result_cache import ResultCache
from core.result_commands import (
    execute_result_command,
    parse_result_command,
)


ROOT = Path(__file__).resolve().parents[1]


class ConversationalResultCacheTests(unittest.TestCase):
    def setUp(self):
        self.cache = ResultCache(max_sessions=4)
        self.session = "regulated:user-1"
        self.rows = [
            {"DOCTOR_NAME": "Dr. Priya Shah", "REVENUE": 1250.0},
            {"DOCTOR_NAME": "Dr. Kiran Rao", "REVENUE": 900.0},
            {"DOCTOR_NAME": "Dr. Anil Bose", "REVENUE": 700.0},
        ]
        self.source_id = self.cache.store(
            self.session,
            self.rows,
            question="Revenue by doctor",
            sql="SELECT DOCTOR_NAME, REVENUE FROM governed_view",
            column_formats={"REVENUE": "currency"},
            result_id="source-1",
        )

    def test_parser_is_conservative(self):
        self.assertEqual(parse_result_command("exclude Dr. Kiran Rao").action, "exclude")
        self.assertEqual(parse_result_command("undo that").action, "undo")
        self.assertEqual(parse_result_command("keep top 2").action, "keep_top")
        self.assertEqual(parse_result_command("sort by revenue descending").action, "sort")
        self.assertIsNone(parse_result_command("What is revenue by doctor?"))
        self.assertEqual(
            parse_result_command(
                "Using this result, show total revenue by doctor"
            ).action,
            "aggregate",
        )

    def test_exclusion_creates_child_and_preserves_source(self):
        outcome = execute_result_command(
            self.session,
            parse_result_command("exclude Dr. Kiran Rao"),
            cache=self.cache,
        )
        self.assertTrue(outcome.ok, outcome.message)
        self.assertEqual(outcome.rows_before, 3)
        self.assertEqual(outcome.rows_after, 2)
        self.assertNotEqual(outcome.derived_result_id, self.source_id)
        self.assertEqual(self.cache.get_snapshot(self.session, self.source_id)["rows"], self.rows)
        self.assertNotIn(
            "Dr. Kiran Rao",
            [row["DOCTOR_NAME"] for row in outcome.snapshot["rows"]],
        )
        self.assertNotIn("Kiran", outcome.snapshot["sql"])
        self.assertEqual(
            outcome.snapshot["metadata"]["metadata_contains_raw_values"], False
        )

    def test_multiple_values_are_excluded_locally(self):
        outcome = execute_result_command(
            self.session,
            parse_result_command("remove Dr. Priya Shah and Dr. Anil Bose"),
            cache=self.cache,
        )
        self.assertTrue(outcome.ok, outcome.message)
        self.assertEqual(outcome.rows_after, 1)
        self.assertEqual(outcome.snapshot["rows"][0]["DOCTOR_NAME"], "Dr. Kiran Rao")

    def test_undo_restores_parent_snapshot(self):
        filtered = execute_result_command(
            self.session,
            parse_result_command("exclude row 1"),
            cache=self.cache,
        )
        self.assertTrue(filtered.ok, filtered.message)
        restored = execute_result_command(
            self.session,
            parse_result_command("undo"),
            cache=self.cache,
        )
        self.assertTrue(restored.ok, restored.message)
        self.assertEqual(restored.derived_result_id, self.source_id)
        self.assertEqual(restored.snapshot["rows"], self.rows)

    def test_snapshot_ids_are_session_scoped(self):
        self.assertEqual(self.cache.get_snapshot("regulated:user-2", self.source_id), {})
        outcome = execute_result_command(
            "regulated:user-2",
            parse_result_command("exclude Dr. Kiran Rao"),
            cache=self.cache,
            source_result_id=self.source_id,
        )
        self.assertFalse(outcome.ok)

    def test_ambiguous_value_fails_closed(self):
        cache = ResultCache()
        cache.store(
            self.session,
            [{"DOCTOR": "Alex", "PATIENT": "Alex", "COUNT": 2}],
            result_id="ambiguous",
        )
        outcome = execute_result_command(
            self.session,
            parse_result_command("exclude Alex"),
            cache=cache,
        )
        self.assertFalse(outcome.ok)
        self.assertIn("more than one field", outcome.message)
        self.assertEqual(cache.get_snapshot(self.session)["row_count"], 1)

    def test_generic_mask_cannot_remove_every_masked_record(self):
        cache = ResultCache()
        cache.store(
            self.session,
            [
                {"DOCTOR": "[REDACTED]", "REVENUE": 10},
                {"DOCTOR": "[REDACTED]", "REVENUE": 20},
            ],
            result_id="masked",
        )
        blocked = execute_result_command(
            self.session,
            parse_result_command("exclude [REDACTED]"),
            cache=cache,
        )
        self.assertFalse(blocked.ok)
        self.assertEqual(cache.get_snapshot(self.session)["row_count"], 2)

        by_position = execute_result_command(
            self.session,
            parse_result_command("exclude row 2"),
            cache=cache,
        )
        self.assertTrue(by_position.ok, by_position.message)
        self.assertEqual(by_position.rows_after, 1)

    def test_parameterized_filter_runs_locally(self):
        outcome = execute_result_command(
            self.session,
            parse_result_command(
                "Filter this result where revenue is greater than 800"
            ),
            cache=self.cache,
        )
        self.assertTrue(outcome.ok, outcome.message)
        self.assertEqual(outcome.rows_after, 2)
        self.assertIn("?", outcome.snapshot["sql"])
        self.assertNotIn("800", outcome.snapshot["sql"])
        self.assertFalse(
            outcome.snapshot["metadata"]["metadata_contains_raw_values"]
        )

    def test_grouped_aggregation_changes_snapshot_schema(self):
        outcome = execute_result_command(
            self.session,
            parse_result_command(
                "Using this result, show total revenue by doctor name"
            ),
            cache=self.cache,
        )
        self.assertTrue(outcome.ok, outcome.message)
        self.assertEqual(outcome.operation, "aggregate")
        self.assertEqual(
            [item["name"] for item in outcome.snapshot["schema"]],
            ["DOCTOR_NAME", "TOTAL_REVENUE"],
        )
        self.assertEqual(
            outcome.snapshot["column_formats"], {"TOTAL_REVENUE": "currency"}
        )

    def test_contribution_is_computed_without_llm(self):
        outcome = execute_result_command(
            self.session,
            parse_result_command(
                "From this result show percentage contribution of revenue by doctor"
            ),
            cache=self.cache,
        )
        self.assertTrue(outcome.ok, outcome.message)
        percentages = [
            row["PERCENTAGE_CONTRIBUTION"] for row in outcome.snapshot["rows"]
        ]
        self.assertAlmostEqual(sum(percentages), 100.0, places=1)
        self.assertEqual(
            outcome.snapshot["column_formats"]["PERCENTAGE_CONTRIBUTION"],
            "percentage",
        )

    def test_profit_percentage_uses_revenue_and_cost_columns(self):
        cache = ResultCache()
        cache.store(
            self.session,
            [{"CLIENT": "A", "REVENUE": 100.0, "COST": 60.0}],
            result_id="profit-source",
        )
        outcome = execute_result_command(
            self.session,
            parse_result_command(
                "Using this result calculate profit percentage"
            ),
            cache=cache,
        )
        self.assertTrue(outcome.ok, outcome.message)
        self.assertEqual(outcome.snapshot["rows"][0]["PROFIT_PERCENTAGE"], 40.0)
        self.assertEqual(
            outcome.snapshot["column_formats"]["PROFIT_PERCENTAGE"],
            "percentage",
        )

    def test_generic_ratio_is_calculated_per_cached_row(self):
        cache = ResultCache()
        cache.store(
            self.session,
            [{"CLIENT": "A", "REVENUE": 100.0, "COST": 40.0}],
            result_id="ratio-source",
        )
        outcome = execute_result_command(
            self.session,
            parse_result_command(
                "Using this result calculate revenue divided by cost"
            ),
            cache=cache,
        )
        self.assertTrue(outcome.ok, outcome.message)
        self.assertEqual(
            outcome.snapshot["rows"][0]["REVENUE_PERCENT_OF_COST"],
            250.0,
        )

    def test_missing_analytic_columns_allow_governed_fallback(self):
        command = parse_result_command(
            "Using this result calculate profit percentage"
        )
        outcome = execute_result_command(self.session, command, cache=self.cache)
        self.assertFalse(outcome.ok)
        self.assertFalse(outcome.handled)
        self.assertTrue(command.fallback_allowed)


class ConversationalResultCacheWiringTests(unittest.TestCase):
    def test_chat_command_route_precedes_insight_and_main_dispatch(self):
        source = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")
        command_position = source.index("result_command = parse_result_command(text)")
        insight_position = source.index('from core.insight import is_insight_question', command_position)
        dispatch_position = source.index("_run_main_question(text, table_hint, schema_hint)", command_position)
        self.assertLess(command_position, insight_position)
        self.assertLess(command_position, dispatch_position)

    def test_local_command_handler_has_zero_exposure_proof(self):
        source = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")
        start = source.index("async def _run_local_result_command")
        end = source.index("\n    async def _run_metadata_result_planner", start)
        block = source[start:end]
        self.assertIn('"rows_sent_to_llm": 0', block)
        self.assertIn('"database_queried": False', block)
        self.assertIn("record_llm_blocked", block)
        self.assertNotIn("llm_complete", block)
        self.assertNotIn("_generate_duckdb_sql", block)
        self.assertNotIn("run_query(", block)

    def test_analytic_cache_miss_routes_to_governed_pipeline(self):
        source = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")
        start = source.index("async def _run_local_result_command")
        end = source.index("\n    async def _run_metadata_result_planner", start)
        block = source[start:end]
        self.assertIn("command, \"fallback_allowed\"", block)
        self.assertIn("await _run_main_question(text, table_hint, schema_hint)", block)

    def test_portal_uses_chat_commands_not_row_checkboxes(self):
        template = (ROOT / "portal" / "templates" / "portal_chat.html").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("dt-row-select", template)
        self.assertNotIn("Exclude selected", template)
        self.assertIn("msg.result_command", template)
        self.assertIn("No LLM call", template)


if __name__ == "__main__":
    unittest.main()
