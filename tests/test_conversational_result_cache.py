from pathlib import Path
import unittest

from core.result_cache import ResultCache
from core.result_commands import (
    ResultCommand,
    execute_result_command,
    parse_result_command,
)
from core.query_router import should_route_to_result_cache
from core.response_builder import build_assistant_response


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
        self.assertEqual(parse_result_command("Undo that change.").action, "undo")
        self.assertIsNone(parse_result_command("What is revenue by doctor?"))
        self.assertEqual(
            parse_result_command(
                "Using this result, show total revenue by doctor"
            ).action,
            "aggregate",
        )

    def test_natural_month_exclusion_matches_period_value(self):
        cache = ResultCache()
        cache.store(
            self.session,
            [
                {"PERIOD": "2025-06", "TOTAL_NET_REVENUE": 778.0},
                {"PERIOD": "2025-03", "TOTAL_NET_REVENUE": 580.0},
                {"PERIOD": "2025-02", "TOTAL_NET_REVENUE": 566.75},
            ],
            result_id="period-source",
        )
        outcome = execute_result_command(
            self.session,
            parse_result_command("Exclude Feb 2025."),
            cache=cache,
        )
        self.assertTrue(outcome.ok, outcome.message)
        self.assertEqual([row["PERIOD"] for row in outcome.snapshot["rows"]], ["2025-06", "2025-03"])

    def test_natural_month_subset_uses_cached_period_values(self):
        cache = ResultCache()
        cache.store(
            self.session,
            [
                {"BOOKED_MONTH": "2025-01", "TOTAL_NET_REVENUE": 100.0},
                {"BOOKED_MONTH": "2025-02", "TOTAL_NET_REVENUE": 200.0},
                {"BOOKED_MONTH": "2025-03", "TOTAL_NET_REVENUE": 300.0},
                {"BOOKED_MONTH": "2025-04", "TOTAL_NET_REVENUE": 400.0},
            ],
            result_id="booked-month-source",
        )
        command = parse_result_command("give me the data only for feb and april")
        self.assertIsNotNone(command)
        self.assertEqual(command.action, "keep_values")
        outcome = execute_result_command(self.session, command, cache=cache)
        self.assertTrue(outcome.ok, outcome.message)
        self.assertEqual(
            [row["BOOKED_MONTH"] for row in outcome.snapshot["rows"]],
            ["2025-02", "2025-04"],
        )
        self.assertIn("?", outcome.snapshot["sql"])
        self.assertNotIn("2025-02", outcome.snapshot["sql"])
        self.assertFalse(outcome.snapshot["metadata"]["metadata_contains_raw_values"])

    def test_natural_month_subset_is_routed_to_the_cached_result(self):
        columns = ["BOOKED_MONTH", "TOTAL_NET_REVENUE"]
        self.assertTrue(
            should_route_to_result_cache(
                "give me the data only for feb and april",
                True,
                columns,
            )
        )
        self.assertTrue(
            should_route_to_result_cache(
                "show only February and April",
                True,
                columns,
            )
        )
        self.assertEqual(
            parse_result_command("Feb and Apr only").action,
            "keep_values",
        )
        self.assertFalse(
            should_route_to_result_cache(
                "give me the data only for feb and april",
                False,
                columns,
            )
        )

    def test_month_subset_can_match_multiple_years_when_year_is_omitted(self):
        cache = ResultCache()
        cache.store(
            self.session,
            [
                {"PERIOD": "2025-02", "REVENUE": 10},
                {"PERIOD": "2026-02", "REVENUE": 20},
                {"PERIOD": "2026-04", "REVENUE": 30},
            ],
            result_id="multi-year-source",
        )
        outcome = execute_result_command(
            self.session,
            parse_result_command("show only February and April"),
            cache=cache,
        )
        self.assertTrue(outcome.ok, outcome.message)
        self.assertEqual(
            [row["PERIOD"] for row in outcome.snapshot["rows"]],
            ["2025-02", "2026-02", "2026-04"],
        )

    def test_keep_top_by_business_metric_sorts_before_limiting(self):
        cache = ResultCache()
        cache.store(
            self.session,
            [
                {"PERIOD": "2025-01", "TOTAL_NET_REVENUE": 100.0},
                {"PERIOD": "2025-02", "TOTAL_NET_REVENUE": 300.0},
                {"PERIOD": "2025-03", "TOTAL_NET_REVENUE": 200.0},
            ],
            result_id="ranking-source",
        )
        outcome = execute_result_command(
            self.session,
            parse_result_command("Keep the top 2 months by net revenue."),
            cache=cache,
        )
        self.assertTrue(outcome.ok, outcome.message)
        self.assertEqual(
            [row["TOTAL_NET_REVENUE"] for row in outcome.snapshot["rows"]],
            [300.0, 200.0],
        )

    def test_ranked_period_subset_is_not_described_as_a_time_trend(self):
        payload = build_assistant_response(
            question="Top 3 months by net revenue from the prior result",
            rows=[
                {"PERIOD": "2025-06", "TOTAL_NET_REVENUE": 778.0},
                {"PERIOD": "2025-03", "TOTAL_NET_REVENUE": 580.0},
                {"PERIOD": "2025-02", "TOTAL_NET_REVENUE": 566.75},
            ],
            sql='SELECT * FROM result ORDER BY "TOTAL_NET_REVENUE" DESC LIMIT 3',
            duration_ms=1,
            display_context={"result_operation": "keep_top"},
        )
        self.assertEqual(payload["analysis_contract"]["mode"], "ranking")
        self.assertNotIn("trended", payload["insight_summary"].lower())

    def test_contribution_uses_only_current_cached_subset(self):
        cache = ResultCache()
        cache.store(
            self.session,
            [
                {"PERIOD": "2025-06", "TOTAL_NET_REVENUE": 778.0},
                {"PERIOD": "2025-03", "TOTAL_NET_REVENUE": 580.0},
                {"PERIOD": "2025-02", "TOTAL_NET_REVENUE": 566.75},
            ],
            result_id="contribution-source",
        )
        outcome = execute_result_command(
            self.session,
            ResultCommand(
                "contribution",
                dimension_text="booked month",
                metric_text="net revenue",
            ),
            cache=cache,
        )
        self.assertTrue(outcome.ok, outcome.message)
        self.assertEqual(len(outcome.snapshot["rows"]), 3)
        self.assertIn("TOTAL_NET_REVENUE", outcome.snapshot["rows"][0])
        self.assertNotIn("TOTAL_TOTAL_NET_REVENUE", outcome.snapshot["rows"][0])
        self.assertAlmostEqual(
            sum(row["PERCENTAGE_CONTRIBUTION"] for row in outcome.snapshot["rows"]),
            100.0,
            places=1,
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
