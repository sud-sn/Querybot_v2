import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from core.conversation_state import (
    ConversationStateStore,
    TurnIntent,
    bypass_analyst_gate,
    classify_turn,
    should_route_as_governed_follow_up,
)


class ConversationStateStoreTests(unittest.TestCase):
    def setUp(self):
        self.now = 1_000.0
        self.store = ConversationStateStore(
            ttl_seconds=120,
            clock=lambda: self.now,
        )

    def _record(self, account="tenant-a", session="session-a", **overrides):
        values = {
            "user_id": "user-a",
            "channel": "portal",
            "question": "Revenue by month",
            "decision": classify_turn("Revenue by month", looks_like_data=True),
            "result_id": "result-a",
            "trace_id": "trace-a",
            "result_schema": ["PERIOD", "REVENUE"],
            "result_metadata": {
                "semantic_version": "semantic-v2",
                "metric_version": "metric-v3",
                "rows": [{"PERIOD": "2025-01", "REVENUE": 10}],
                "raw_values": ["sensitive"],
            },
        }
        values.update(overrides)
        return self.store.record(account, session, **values)

    def test_record_is_metadata_only(self):
        state = self._record()

        self.assertEqual(state.account_id, "tenant-a")
        self.assertEqual(state.session_id, "session-a")
        self.assertEqual(state.result_schema, ("PERIOD", "REVENUE"))
        self.assertEqual(
            state.model_versions,
            {
                "semantic_version": "semantic-v2",
                "metric_version": "metric-v3",
            },
        )
        self.assertFalse(hasattr(state, "rows"))
        self.assertNotIn("rows", state.model_versions)
        self.assertNotIn("raw_values", state.model_versions)

    def test_state_is_isolated_by_tenant_and_session(self):
        self._record()
        self._record(
            account="tenant-b",
            session="session-a",
            result_id="result-b",
        )
        self._record(
            account="tenant-a",
            session="session-b",
            result_id="result-c",
        )

        self.assertEqual(self.store.get("tenant-a", "session-a").result_id, "result-a")
        self.assertEqual(self.store.get("tenant-b", "session-a").result_id, "result-b")
        self.assertEqual(self.store.get("tenant-a", "session-b").result_id, "result-c")

    def test_expired_state_is_removed_lazily(self):
        self._record()
        self.now += 121

        self.assertIsNone(self.store.get("tenant-a", "session-a"))

    def test_clear_and_clear_account(self):
        self._record(session="session-a")
        self._record(session="session-b")
        self._record(account="tenant-b", session="session-c")

        self.store.clear("tenant-a", "session-a")
        self.assertIsNone(self.store.get("tenant-a", "session-a"))
        self.assertIsNotNone(self.store.get("tenant-a", "session-b"))

        self.store.clear_account("tenant-a")
        self.assertIsNone(self.store.get("tenant-a", "session-b"))
        self.assertIsNotNone(self.store.get("tenant-b", "session-c"))

    def test_record_refreshes_ttl_and_preserves_existing_result_handle(self):
        original = self._record()
        self.now += 60
        follow_up = classify_turn(
            "Why did this change?",
            state=original,
            has_cached_result=True,
        )
        updated = self.store.record(
            "tenant-a",
            "session-a",
            question="Why did this change?",
            decision=follow_up,
        )

        self.assertEqual(updated.result_id, "result-a")
        self.assertEqual(updated.trace_id, "trace-a")
        self.assertEqual(updated.expires_at, self.now + 120)
        self.assertEqual(updated.previous_intent, "result_grounded_analysis")

    def test_date_preference_is_thread_and_metric_fact_scoped(self):
        self._record()
        binding = {
            "context_name": "Invoice Date",
            "date_role": "invoice_date",
            "fact_table": "SALES.FACT_REVENUE",
            "fact_column": "INVOICE_DATE_KEY",
            "dimension_table": "COMMON.DIM_DATE",
            "dimension_key": "DATE_KEY",
            "date_value_column": "FULL_DATE",
            "date_key_type": "surrogate_fk",
            "rows": [{"must": "not persist"}],
        }
        self.store.remember_date_preference(
            "tenant-a",
            "session-a",
            binding,
            metric_names=["Revenue"],
            fact_tables=["SALES.FACT_REVENUE"],
        )

        remembered = self.store.get_date_preference(
            "tenant-a",
            "session-a",
            metric_names=["Revenue"],
            fact_tables=["SALES.FACT_REVENUE"],
        )
        self.assertEqual(remembered["fact_column"], "INVOICE_DATE_KEY")
        self.assertNotIn("rows", remembered)
        self.assertEqual(
            self.store.get_date_preference(
                "tenant-a",
                "session-a",
                metric_names=["Inventory"],
                fact_tables=["INVENTORY.FACT_STOCK"],
            ),
            {},
        )

    def test_calendar_preference_is_metadata_only_and_thread_scoped(self):
        self._record()
        self.store.remember_calendar_preference(
            "tenant-a",
            "session-a",
            {
                "basis": "fiscal",
                "fiscal_year_start_month": 4,
                "source": "user_confirmed",
                "rows": [{"must": "not persist"}],
            },
        )

        self.assertEqual(
            self.store.get_calendar_preference("tenant-a", "session-a"),
            {
                "basis": "fiscal",
                "fiscal_year_start_month": 4,
                "source": "user_confirmed",
            },
        )
        self.assertEqual(
            self.store.get_calendar_preference("tenant-a", "another-thread"),
            {},
        )
        self.assertEqual(
            self.store.get_date_preference(
                "tenant-a",
                "another-thread",
                metric_names=["Revenue"],
                fact_tables=["SALES.FACT_REVENUE"],
            ),
            {},
        )

    def test_metadata_state_survives_store_recreation_when_persistence_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "conversation.db")
            with patch.dict(os.environ, {"QUERYBOT_DB_PATH": db_path}):
                first = ConversationStateStore(
                    ttl_seconds=120,
                    clock=lambda: self.now,
                    persist=True,
                )
                first.record(
                    "tenant-a",
                    "thread-1",
                    user_id="user-a",
                    channel="portal",
                    question="Revenue by month",
                    decision=classify_turn("Revenue by month", looks_like_data=True),
                    result_id="result-a",
                    trace_id="trace-a",
                    result_schema=["MONTH", "REVENUE"],
                    result_metadata={
                        "semantic_version": "v3",
                        "rows": [{"sensitive": "must not persist"}],
                    },
                )
                first.remember_date_preference(
                    "tenant-a",
                    "thread-1",
                    {
                        "context_name": "Invoice Date",
                        "fact_table": "SALES.FACT_REVENUE",
                        "fact_column": "INVOICE_DATE_KEY",
                        "dimension_table": "COMMON.DIM_DATE",
                        "dimension_key": "DATE_KEY",
                        "date_value_column": "FULL_DATE",
                        "date_key_type": "surrogate_fk",
                    },
                    metric_names=["Revenue"],
                    fact_tables=["SALES.FACT_REVENUE"],
                )

                recreated = ConversationStateStore(
                    ttl_seconds=120,
                    clock=lambda: self.now,
                    persist=True,
                )
                restored = recreated.get("tenant-a", "thread-1")

                self.assertIsNotNone(restored)
                self.assertEqual(restored.previous_question, "Revenue by month")
                self.assertEqual(restored.result_schema, ("MONTH", "REVENUE"))
                self.assertEqual(restored.model_versions, {"semantic_version": "v3"})
                self.assertFalse(hasattr(restored, "rows"))
                self.assertEqual(
                    recreated.get_date_preference(
                        "tenant-a",
                        "thread-1",
                        metric_names=["Revenue"],
                        fact_tables=["SALES.FACT_REVENUE"],
                    )["fact_column"],
                    "INVOICE_DATE_KEY",
                )

                recreated.clear("tenant-a", "thread-1")
                third = ConversationStateStore(
                    ttl_seconds=120,
                    clock=lambda: self.now,
                    persist=True,
                )
                self.assertIsNone(third.get("tenant-a", "thread-1"))


class TurnClassificationTests(unittest.TestCase):
    def setUp(self):
        self.store = ConversationStateStore(ttl_seconds=120)
        self.state = self.store.record(
            "tenant-a",
            "session-a",
            question="Revenue by booked month in 2025",
            decision=classify_turn(
                "Revenue by booked month in 2025",
                looks_like_data=True,
            ),
            result_id="result-a",
            trace_id="trace-a",
            result_schema=["BOOKED_MONTH", "NET_REVENUE"],
        )

    def classify(self, text, **overrides):
        values = {
            "state": self.state,
            "has_cached_result": True,
            "looks_like_data": False,
        }
        values.update(overrides)
        return classify_turn(text, **values)

    def test_new_business_question_without_prior_result(self):
        decision = classify_turn(
            "What is revenue by customer?",
            looks_like_data=True,
        )

        self.assertEqual(decision.intent, TurnIntent.NEW_DATA_QUERY)
        self.assertFalse(decision.uses_prior_result)

    def test_direct_result_command_is_local_transform(self):
        decision = self.classify(
            "Sort these results by revenue descending",
            direct_result_command=True,
        )

        self.assertEqual(decision.intent, TurnIntent.RESULT_LOCAL_TRANSFORM)
        self.assertTrue(decision.uses_prior_result)
        self.assertEqual(decision.parent_result_id, "result-a")

    def test_natural_month_subset_is_query_refinement(self):
        decision = self.classify("Give me only February and April")

        self.assertEqual(decision.intent, TurnIntent.QUERY_REFINEMENT)
        self.assertTrue(bypass_analyst_gate(decision))

    def test_result_grounded_analysis(self):
        decision = self.classify("Why did this change?")

        self.assertEqual(decision.intent, TurnIntent.RESULT_GROUNDED_ANALYSIS)
        self.assertTrue(decision.uses_prior_result)

    def test_short_superlative_uses_active_result(self):
        decision = self.classify("Which one is highest?")

        self.assertEqual(decision.intent, TurnIntent.RESULT_GROUNDED_ANALYSIS)
        self.assertTrue(decision.uses_prior_result)
        self.assertEqual(decision.parent_result_id, "result-a")

    def test_unrelated_business_question_starts_new_query(self):
        decision = self.classify(
            "Which suppliers have the highest receipt cost?",
            looks_like_data=True,
        )

        self.assertEqual(decision.intent, TurnIntent.NEW_DATA_QUERY)
        self.assertFalse(decision.uses_prior_result)

    def test_clarification_response_bypasses_analyst_gate(self):
        decision = self.classify(
            "Booked revenue",
            is_clarification=True,
            has_cached_result=False,
        )

        self.assertEqual(decision.intent, TurnIntent.CLARIFICATION_RESPONSE)
        self.assertTrue(bypass_analyst_gate(decision))

    def test_reset_context(self):
        decision = self.classify("Start over")

        self.assertEqual(decision.intent, TurnIntent.RESET_CONTEXT)
        self.assertFalse(bypass_analyst_gate(decision))

    def test_greeting_and_safe_off_topic(self):
        self.assertEqual(self.classify("Hello").intent, TurnIntent.GREETING)
        self.assertEqual(
            self.classify("Write me a birthday poem").intent,
            TurnIntent.SAFE_OFF_TOPIC,
        )

    def test_follow_up_language_without_cached_result_does_not_reuse_state(self):
        decision = classify_turn(
            "Only February and April",
            state=self.state,
            has_cached_result=False,
            looks_like_data=True,
        )

        self.assertEqual(decision.intent, TurnIntent.NEW_DATA_QUERY)
        self.assertFalse(decision.uses_prior_result)

    def test_legacy_vague_analysis_routes_with_governed_result(self):
        self.assertTrue(
            should_route_as_governed_follow_up(
                "Analyze this",
                state=self.state,
                has_cached_result=True,
            )
        )

    def test_legacy_vague_analysis_stays_conversational_without_result(self):
        self.assertFalse(
            should_route_as_governed_follow_up(
                "Analyze this",
                state=self.state,
                has_cached_result=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
