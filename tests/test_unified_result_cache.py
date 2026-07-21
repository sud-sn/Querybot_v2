import unittest
from pathlib import Path

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

    async def test_planner_receives_metadata_not_values_or_sql(self):
        captured = {}

        async def complete(**kwargs):
            captured.update(kwargs)
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
        self.assertTrue(result.executed, result.reason)
        prompt = captured["system"] + captured["user"]
        self.assertNotIn("Priya", prompt)
        self.assertNotIn("1200", prompt)
        self.assertNotIn("protected source query", prompt)
        self.assertEqual(result.evidence["sample_values_sent_to_llm"], 0)
        self.assertFalse(result.evidence["source_sql_sent_to_llm"])

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


if __name__ == "__main__":
    unittest.main()
