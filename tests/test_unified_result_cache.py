import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from core.governed_result_followup import adopt_cached_snapshot, run_governed_result_followup
from core.result_cache import ResultCache


ROOT = Path(__file__).resolve().parents[1]


class GovernedResultFollowupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cache = ResultCache(max_sessions=4)
        self.session_id = "account:user"
        self.result_id = self.cache.store(
            self.session_id,
            [
                {"DOCTOR_NAME": "Dr. Priya Shah", "REVENUE": 1200.0},
                {"DOCTOR_NAME": "Dr. Arun Rao", "REVENUE": 900.0},
            ],
            "Revenue by doctor",
            "SELECT protected source query",
            column_formats={"REVENUE": "currency"},
        )

    async def test_explicit_command_executes_without_model(self):
        called = False

        async def complete(**_kwargs):
            nonlocal called
            called = True
            return "", 0, 0

        result = await run_governed_result_followup(
            "keep top 1",
            self.session_id,
            complete=complete,
            source_result_id=self.result_id,
            cache=self.cache,
        )
        self.assertTrue(result.executed)
        self.assertFalse(called)
        self.assertEqual(result.outcome.rows_after, 1)
        self.assertEqual(result.evidence["rows_sent_to_llm"], 0)

    async def test_named_month_subset_executes_without_model_or_database(self):
        source_result_id = self.cache.store(
            self.session_id,
            [
                {"BOOKED_MONTH": "2025-01", "NET_REVENUE": 100.0},
                {"BOOKED_MONTH": "2025-02", "NET_REVENUE": 200.0},
                {"BOOKED_MONTH": "2025-03", "NET_REVENUE": 300.0},
                {"BOOKED_MONTH": "2025-04", "NET_REVENUE": 400.0},
            ],
            "Net revenue by booked month",
            "SELECT governed source query",
        )
        called = False

        async def complete(**_kwargs):
            nonlocal called
            called = True
            return "", 0, 0

        result = await run_governed_result_followup(
            "give me the data only for feb and april",
            self.session_id,
            complete=complete,
            source_result_id=source_result_id,
            cache=self.cache,
        )

        self.assertTrue(result.executed, result.reason)
        self.assertFalse(called)
        self.assertEqual(
            [row["BOOKED_MONTH"] for row in result.outcome.snapshot["rows"]],
            ["2025-02", "2025-04"],
        )
        self.assertEqual(result.evidence["rows_sent_to_llm"], 0)
        self.assertFalse(result.evidence["database_queried"])

    async def test_planner_receives_metadata_not_values_or_sql(self):
        captured = {}

        async def complete(**kwargs):
            captured.update(kwargs)
            return (
                '{"operation":"contribution","dimension":"DOCTOR_NAME",'
                '"metric":"REVENUE","confidence":0.95}',
                20,
                10,
            )

        result = await run_governed_result_followup(
            "Show percentage contribution in this result",
            self.session_id,
            complete=complete,
            source_result_id=self.result_id,
            cache=self.cache,
        )
        self.assertTrue(result.executed, result.reason)
        prompt = captured["system"] + captured["user"]
        self.assertNotIn("Priya", prompt)
        self.assertNotIn("1200", prompt)
        self.assertNotIn("protected source query", prompt)
        self.assertEqual(result.evidence["sample_values_sent_to_llm"], 0)
        self.assertFalse(result.evidence["source_sql_sent_to_llm"])

    async def test_low_confidence_plan_requests_confirmation_instead_of_executing(self):
        async def complete(**_kwargs):
            return (
                '{"operation":"contribution","dimension":"DOCTOR_NAME",'
                '"metric":"REVENUE","confidence":0.4}',
                20,
                10,
            )

        result = await run_governed_result_followup(
            "Show percentage contribution in this result",
            self.session_id,
            complete=complete,
            source_result_id=self.result_id,
            cache=self.cache,
        )
        self.assertEqual(result.status, "clarification")
        self.assertFalse(result.executed)
        self.assertTrue(result.outcome.clarification_required)
        self.assertIn("DOCTOR_NAME", result.outcome.clarification_prompt)
        self.assertIn("REVENUE", result.outcome.clarification_prompt)
        self.assertEqual(result.evidence["planner_confidence"], 0.4)
        # The one confirmation option must resolve back to the exact
        # original question so a confirm reply can be replayed with
        # is_clarification=True.
        self.assertEqual(len(result.outcome.clarification_options), 1)
        self.assertEqual(
            result.outcome.clarification_options[0]["resolved_question"],
            "Show percentage contribution in this result",
        )

    async def test_high_confidence_plan_executes_normally(self):
        async def complete(**_kwargs):
            return (
                '{"operation":"contribution","dimension":"DOCTOR_NAME",'
                '"metric":"REVENUE","confidence":0.95}',
                20,
                10,
            )

        result = await run_governed_result_followup(
            "Show percentage contribution in this result",
            self.session_id,
            complete=complete,
            source_result_id=self.result_id,
            cache=self.cache,
        )
        self.assertTrue(result.executed, result.reason)

    async def test_missing_confidence_field_defaults_low_and_asks_for_confirmation(self):
        # Fail closed: a model that doesn't follow the schema and emit
        # "confidence" at all is not evidence of a trustworthy response --
        # it must default BELOW the clarification threshold, not above it,
        # so a non-compliant response is confirmed rather than trusted.
        async def complete(**_kwargs):
            return (
                '{"operation":"contribution","dimension":"DOCTOR_NAME",'
                '"metric":"REVENUE"}',
                20,
                10,
            )

        result = await run_governed_result_followup(
            "Show percentage contribution in this result",
            self.session_id,
            complete=complete,
            source_result_id=self.result_id,
            cache=self.cache,
        )
        self.assertEqual(result.status, "clarification")
        self.assertFalse(result.executed)
        self.assertEqual(result.evidence["planner_confidence"], 0.0)

    async def test_confirmed_clarification_reply_skips_confidence_gate(self):
        # A confirmed clarification reply must execute even at the same
        # low confidence -- re-litigating a choice the user just made
        # would create an infinite "did you mean" loop.
        async def complete(**_kwargs):
            return (
                '{"operation":"contribution","dimension":"DOCTOR_NAME",'
                '"metric":"REVENUE","confidence":0.2}',
                20,
                10,
            )

        result = await run_governed_result_followup(
            "Show percentage contribution in this result",
            self.session_id,
            complete=complete,
            source_result_id=self.result_id,
            cache=self.cache,
            is_clarification=True,
        )
        self.assertTrue(result.executed, result.reason)

    async def test_bound_literal_planner_failure_is_blocked(self):
        async def complete(**_kwargs):
            return '{"operation":"unsupported"}', 5, 5

        result = await run_governed_result_followup(
            "Identify unusual rows around 1000 in this result",
            self.session_id,
            complete=complete,
            source_result_id=self.result_id,
            cache=self.cache,
        )
        self.assertEqual(result.status, "blocked")
        self.assertGreater(result.evidence["literal_binding_count"], 0)
        self.assertEqual(result.evidence["literal_values_sent_to_llm"], 0)

    async def test_unbound_unsupported_request_can_use_governed_fallback(self):
        async def complete(**_kwargs):
            return '{"operation":"unsupported"}', 5, 5

        result = await run_governed_result_followup(
            "Explain why this result changed",
            self.session_id,
            complete=complete,
            source_result_id=self.result_id,
            cache=self.cache,
        )
        self.assertEqual(result.status, "unsupported")
        self.assertFalse(result.evidence["database_queried"])

    async def test_highest_followup_uses_active_result_without_database_query(self):
        async def complete(**_kwargs):
            return (
                '{"operation":"keep_top","metric":"REVENUE",'
                '"direction":"desc","confidence":0.99}',
                10,
                5,
            )

        result = await run_governed_result_followup(
            "Which one is highest?",
            self.session_id,
            complete=complete,
            cache=self.cache,
        )

        self.assertTrue(result.executed, result.reason)
        self.assertEqual(result.command.action, "keep_top")
        self.assertEqual(result.outcome.rows_after, 1)
        self.assertEqual(result.outcome.snapshot["rows"][0]["REVENUE"], 1200.0)
        self.assertFalse(result.evidence["database_queried"])
        self.assertEqual(result.evidence["rows_sent_to_llm"], 0)

    async def test_date_format_clarification_option_executes_on_active_result(self):
        date_result_id = self.cache.store(
            self.session_id,
            [{"INVOICE_MONTH": "2026-01", "REVENUE": 1200.0}],
            "Revenue by invoice month",
            "SELECT governed date result",
            column_formats={"INVOICE_MONTH": "date", "REVENUE": "currency"},
        )
        first = await run_governed_result_followup(
            "change the date format",
            self.session_id,
            source_result_id=date_result_id,
            cache=self.cache,
        )
        self.assertEqual(first.status, "clarification")
        self.assertIn("Which date format", first.outcome.clarification_prompt)

        selected = first.outcome.clarification_options[0]["resolved_question"]
        resumed = await run_governed_result_followup(
            selected,
            self.session_id,
            source_result_id=date_result_id,
            cache=self.cache,
            is_clarification=True,
        )
        self.assertTrue(resumed.executed, resumed.reason)
        self.assertEqual(
            resumed.outcome.snapshot["metadata"]["display_formats"]["INVOICE_MONTH"]["style"],
            "month_year_short",
        )
        self.assertFalse(resumed.evidence["database_queried"])

    async def test_previous_result_presentation_uses_prior_cached_snapshot(self):
        latest_result_id = self.cache.store(
            self.session_id,
            [{"WAREHOUSE": "A", "REVENUE": 10.0}],
            "Revenue by warehouse",
            "SELECT governed warehouse query",
        )

        result = await run_governed_result_followup(
            "show the previous result as a bar chart",
            self.session_id,
            source_result_id=latest_result_id,
            cache=self.cache,
        )

        self.assertTrue(result.executed, result.reason)
        self.assertEqual(result.outcome.snapshot["parent_result_id"], self.result_id)
        self.assertEqual(
            result.outcome.snapshot["metadata"]["chart_type_override"],
            "bar",
        )
        self.assertEqual(result.evidence["rows_sent_to_llm"], 0)
        self.assertFalse(result.evidence["database_queried"])

    async def test_numbered_result_presentation_uses_requested_snapshot(self):
        latest_result_id = self.cache.store(
            self.session_id,
            [{"WAREHOUSE": "A", "REVENUE": 10.0}],
            "Revenue by warehouse",
            "SELECT governed warehouse query",
        )

        result = await run_governed_result_followup(
            "show result 1 as a line chart",
            self.session_id,
            source_result_id=latest_result_id,
            cache=self.cache,
        )

        self.assertTrue(result.executed, result.reason)
        self.assertEqual(result.outcome.snapshot["parent_result_id"], self.result_id)
        self.assertEqual(
            result.outcome.snapshot["metadata"]["chart_type_override"],
            "line",
        )
        self.assertEqual(result.evidence["rows_sent_to_llm"], 0)
        self.assertFalse(result.evidence["database_queried"])

    async def test_ambiguous_earlier_result_requests_clarification(self):
        self.cache.store(
            self.session_id,
            [{"WAREHOUSE": "A", "REVENUE": 10.0}],
            "Revenue by warehouse",
            "SELECT governed warehouse query",
        )
        latest_result_id = self.cache.store(
            self.session_id,
            [{"MONTH": "2025-01", "REVENUE": 20.0}],
            "Revenue by month",
            "SELECT governed monthly query",
        )

        result = await run_governed_result_followup(
            "show the earlier result as a bar chart",
            self.session_id,
            source_result_id=latest_result_id,
            cache=self.cache,
        )

        self.assertEqual(result.status, "clarification")
        self.assertTrue(result.outcome.clarification_required)
        self.assertEqual(len(result.outcome.clarification_options), 2)
        self.assertEqual(
            [item["value"] for item in result.outcome.clarification_options],
            ["result 1", "result 2"],
        )
        self.assertEqual(
            result.outcome.clarification_options[1]["label"],
            "Result 2: Revenue by warehouse",
        )
        self.assertEqual(result.evidence["rows_sent_to_llm"], 0)
        self.assertFalse(result.evidence["database_queried"])

    async def test_invalid_result_number_requests_available_result(self):
        latest_result_id = self.cache.store(
            self.session_id,
            [{"WAREHOUSE": "A", "REVENUE": 10.0}],
            "Revenue by warehouse",
            "SELECT governed warehouse query",
        )

        result = await run_governed_result_followup(
            "show result 99 as a table",
            self.session_id,
            source_result_id=latest_result_id,
            cache=self.cache,
        )

        self.assertEqual(result.status, "clarification")
        self.assertTrue(result.outcome.clarification_required)
        self.assertEqual(len(result.outcome.clarification_options), 2)
        self.assertIn(
            "not available",
            result.outcome.clarification_prompt.lower(),
        )
        self.assertEqual(result.evidence["rows_sent_to_llm"], 0)
        self.assertFalse(result.evidence["database_queried"])


class UnifiedCacheWiringTests(unittest.TestCase):
    def test_adapter_view_is_adopted_from_canonical_snapshot(self):
        class Adapter:
            last_result = {"db_cfg": {"type": "azure_sql"}}
            last_result_id = None
            last_question_id = None

        adapter = Adapter()
        snapshot = {
            "rows": [{"WAREHOUSE": "A", "REVENUE": 10}],
            "question": "Revenue by warehouse",
            "sql": "SELECT governed",
            "column_formats": {"REVENUE": "currency"},
            "result_id": "derived-1",
            "operation": "filter",
        }
        adopted = adopt_cached_snapshot(adapter, snapshot, question_id="question-1")
        self.assertEqual(adapter.last_result_id, "derived-1")
        self.assertEqual(adapter.last_question_id, "question-1")
        self.assertEqual(adopted["rows"], snapshot["rows"])
        self.assertEqual(adopted["db_cfg"], {"type": "azure_sql"})

    def test_inline_result_chat_uses_governed_engine(self):
        source = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")
        start = source.index('if msg_type == "result_chat":')
        fallback = source.index("# The metadata-only cache engine cannot answer", start)
        block = source[start:fallback]
        self.assertIn("run_governed_result_followup(", block)
        self.assertNotIn("get_stats(", block)
        self.assertNotIn("_generate_duckdb_sql", block)
        self.assertIn("No result values were sent to the model", block)

    def test_database_fallback_context_is_metadata_only(self):
        source = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")
        start = source.index("_drill_ctx = _build_metadata_followup_context(")
        block = source[start:source.index("_fb_sql_raw", start)]
        self.assertNotIn("prev_rows", block)
        self.assertNotIn("original_sql", block)

    def test_web_adapter_snapshot_restores_governed_action_context(self):
        from core.result_cache import result_cache
        from gateway.web_adapter import WebAdapter

        source = WebAdapter(
            AsyncMock(), "tenant", "7", thread_id="action-restore"
        )
        semantic_plan = {
            "enabled": True,
            "available_dimensions": [{"name": "Warehouse"}],
        }
        source.cache_result(
            [{"REVENUE": 10}],
            "Revenue",
            "SELECT governed",
            db_cfg={"id": 41, "db_type": "azure_sql"},
            semantic_plan=semantic_plan,
            contract_version="contract-1",
        )
        snapshot = result_cache.get_snapshot(
            source.session_id, source.last_result_id
        )
        restored = WebAdapter(
            AsyncMock(), "tenant", "7", thread_id="action-restore"
        )
        restored.last_result = {
            "result_id": "newer-card",
            "db_cfg": {"id": 99, "db_type": "snowflake"},
            "semantic_plan": {"enabled": False},
            "rag_context": "context from the wrong card",
        }
        try:
            with patch(
                "store.get_db_config",
                return_value={"id": 41, "db_type": "azure_sql"},
            ):
                restored.adopt_cached_snapshot(snapshot)

            self.assertEqual(restored.last_result_id, source.last_result_id)
            self.assertEqual(restored.last_result["semantic_plan"], semantic_plan)
            self.assertEqual(restored.last_result["db_cfg"]["id"], 41)
            self.assertEqual(restored.last_result["rag_context"], "")
            self.assertEqual(
                restored.last_result["contract_version"], "contract-1"
            )
        finally:
            result_cache.clear(source.session_id)

    def test_derived_snapshot_inherits_only_safe_execution_metadata(self):
        cache = ResultCache(max_sessions=2)
        source_id = cache.store(
            "session",
            [{"REVENUE": 10}],
            "Revenue",
            "SELECT governed",
            metadata={
                "db_config_id": 7,
                "semantic_plan": {"enabled": True},
                "contract_version": "v1",
                "unsafe_extra": "do not inherit",
            },
        )

        derived = cache.derive_snapshot(
            "session",
            source_id,
            [{"REVENUE": 10}],
            question="Revenue formatted",
            operation="format",
        )

        self.assertEqual(derived["metadata"]["db_config_id"], 7)
        self.assertEqual(derived["metadata"]["semantic_plan"], {"enabled": True})
        self.assertEqual(derived["metadata"]["contract_version"], "v1")
        self.assertNotIn("unsafe_extra", derived["metadata"])


if __name__ == "__main__":
    unittest.main()
