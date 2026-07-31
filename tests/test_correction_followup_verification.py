"""
Part C2: does "you used the wrong table, use X instead" already regenerate
correctly today, or does it need new correction-detection code?

Investigated (not guessed) via three code facts, each already covered by
its own existing test elsewhere in this suite:
  1. gateway/web_adapter.py::WebAdapter.get_history() returns prior turns
     with BOTH question AND sql (tests/test_multiturn_followup.py).
  2. core/llm.py::build_sql_system_prompt injects a "## Session context"
     block containing that prior question+SQL whenever conversation_history
     is passed (same file).
  3. core/query_pipeline.py::handle_query (~line 1309-1313, 2172) calls
     adapter.get_history() and passes it through as conversation_history --
     the follow-up text is NOT re-routed through any special "correction"
     path, it flows through the same full SQL-generation call as any other
     question, now WITH the prior SQL visible in context.

Conclusion: the plumbing already exists -- a correction like "wrong table,
use orders_v2 instead" reaches the LLM with the exact prior SQL (so it can
see what "wrong" refers to) plus an explicit instruction not to copy that
SQL verbatim but to generate fresh SQL for the new question. This needed
no new code. These tests lock in that specific combination (prior SQL
visible + "generate fresh, don't copy verbatim" instruction + the
conversation_state refinement classifier recognizing correction phrasing)
as a named regression guard, distinct from the general follow-up-history
tests already in tests/test_multiturn_followup.py.

What this suite CANNOT verify: whether the LLM actually honors the
correction in a live call -- that is model behavior, not code behavior,
and needs a real click-test on a live tenant (e.g. Demo_2) to confirm
quality in practice.
"""

import unittest

from core.llm import build_sql_system_prompt
from core.conversation_state import classify_turn, ConversationState, TurnIntent


class CorrectionPromptContextTests(unittest.TestCase):
    def test_prior_sql_naming_the_wrong_table_is_visible_to_the_model(self):
        history = [{
            "question": "show total revenue by customer",
            "sql": "SELECT c.NAME, SUM(o.TOTAL) FROM DBO.F_ORDERS o JOIN DBO.DIM_CUSTOMER c ON o.CUSTOMER_ID = c.CUSTOMER_ID GROUP BY c.NAME",
            "columns": ["NAME", "TOTAL"],
            "row_count": 12,
        }]
        prompt = build_sql_system_prompt("azure_sql", "KB context here", conversation_history=history)
        self.assertIn("F_ORDERS", prompt)
        self.assertIn("Session context", prompt)

    def test_prompt_explicitly_forbids_copying_prior_sql_verbatim(self):
        history = [{"question": "q", "sql": "SELECT 1 FROM DBO.F_ORDERS", "columns": ["x"], "row_count": 1}]
        prompt = build_sql_system_prompt("azure_sql", "KB context", conversation_history=history)
        self.assertIn("Do NOT copy previous SQL verbatim", prompt)
        self.assertIn("generate fresh SQL for the", prompt)


class CorrectionPhraseClassificationTests(unittest.TestCase):
    def _state(self):
        return ConversationState(
            account_id="acct", session_id="sess",
            result_id="r1", trace_id="t1",
        )

    def test_wrong_table_use_x_instead_is_a_refinement_not_a_fresh_query(self):
        decision = classify_turn(
            "you used the wrong table, use orders_v2 instead",
            state=self._state(),
            has_cached_result=True,
        )
        self.assertEqual(decision.intent, TurnIntent.QUERY_REFINEMENT)
        self.assertTrue(decision.uses_prior_result)

    def test_exclude_internal_accounts_correction_is_a_refinement(self):
        decision = classify_turn(
            "you included internal accounts, please exclude them",
            state=self._state(),
            has_cached_result=True,
        )
        self.assertEqual(decision.intent, TurnIntent.QUERY_REFINEMENT)


if __name__ == "__main__":
    unittest.main()
