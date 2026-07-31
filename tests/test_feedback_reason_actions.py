"""
Part C1: feedback reason_code now actually does something.

- store.learning_store._extract_table_names: best-effort table extraction
  from stored feedback SQL.
- store.learning_store.save_feedback: a "wrong_join" downvote flags the
  actual relationship(s) used for re-review via the existing (previously
  unused for this purpose) store.config_store.flag_relationships_needing_review.
- admin/routes.py::admin_learning_queue attaches feedback_reasons per
  candidate so the template can render why an answer was downvoted.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import store
from store import learning_store
from admin import routes


def _arun(coro):
    return asyncio.run(coro)


class ExtractTableNamesTests(unittest.TestCase):
    def test_extracts_tables_from_simple_join(self):
        sql = "SELECT o.ORDER_ID FROM DBO.F_ORDERS o JOIN DBO.DIM_CUSTOMER c ON o.CUSTOMER_ID = c.CUSTOMER_ID"
        tables = learning_store._extract_table_names(sql)
        self.assertIn("DBO.F_ORDERS", tables)
        self.assertIn("DBO.DIM_CUSTOMER", tables)

    def test_empty_sql_returns_empty_set(self):
        self.assertEqual(learning_store._extract_table_names(""), set())

    def test_unparseable_sql_returns_empty_set_not_raise(self):
        self.assertEqual(learning_store._extract_table_names("not ( valid sql !!!"), set())


class SaveFeedbackWrongJoinTests(unittest.TestCase):
    def setUp(self):
        store.init_db()
        self.account_id = f"acct-fbjoin-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")
        # answer_feedback.user_id is a real FK to portal_user(id) -- must
        # reference a real row, not a fabricated id, or the insert fails
        # under any connection where PRAGMA foreign_keys is enabled.
        self.user_id, _ = store.create_user(
            self.account_id, "Test User", f"fbjoin-{uuid.uuid4().hex[:8]}@example.com",
        )
        store.save_entity(self.account_id, "Orders", "F_ORDERS", schema_name="DBO", status="confirmed")
        store.save_entity(self.account_id, "Customer", "DIM_CUSTOMER", schema_name="DBO", status="confirmed")
        self.rel_id = store.save_relationship(
            self.account_id,
            from_entity="Orders", to_entity="Customer",
            from_column="CUSTOMER_ID", to_column="CUSTOMER_ID",
            status="confirmed",
        )

    def tearDown(self):
        with store.get_db() as conn:
            conn.execute("DELETE FROM answer_feedback WHERE account_id=?", (self.account_id,))
            for table in ("entity_relationships", "entity_graph"):
                conn.execute(f"DELETE FROM {table} WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM portal_user WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM client WHERE account_id=?", (self.account_id,))

    def test_wrong_join_feedback_flags_the_actual_relationship(self):
        question_id = f"q-{uuid.uuid4().hex[:8]}"
        learning_store.save_feedback(
            question_id, self.user_id, self.account_id, -1,
            reason_code="wrong_join",
            sql_text="SELECT o.ORDER_ID FROM DBO.F_ORDERS o JOIN DBO.DIM_CUSTOMER c ON o.CUSTOMER_ID = c.CUSTOMER_ID",
        )
        rel = store.get_relationship(self.account_id, self.rel_id)
        self.assertEqual(rel["validation_status"], "needs_review")

    def test_wrong_metric_feedback_does_not_flag_any_join(self):
        question_id = f"q-{uuid.uuid4().hex[:8]}"
        learning_store.save_feedback(
            question_id, self.user_id, self.account_id, -1,
            reason_code="wrong_metric",
            sql_text="SELECT o.ORDER_ID FROM DBO.F_ORDERS o JOIN DBO.DIM_CUSTOMER c ON o.CUSTOMER_ID = c.CUSTOMER_ID",
        )
        rel = store.get_relationship(self.account_id, self.rel_id)
        self.assertNotEqual(rel["validation_status"], "needs_review")

    def test_positive_rating_never_flags_even_with_wrong_join_reason(self):
        question_id = f"q-{uuid.uuid4().hex[:8]}"
        learning_store.save_feedback(
            question_id, self.user_id, self.account_id, 1,
            reason_code="wrong_join",
            sql_text="SELECT o.ORDER_ID FROM DBO.F_ORDERS o JOIN DBO.DIM_CUSTOMER c ON o.CUSTOMER_ID = c.CUSTOMER_ID",
        )
        rel = store.get_relationship(self.account_id, self.rel_id)
        self.assertNotEqual(rel["validation_status"], "needs_review")

    def test_wrong_join_with_no_sql_text_does_not_raise(self):
        question_id = f"q-{uuid.uuid4().hex[:8]}"
        result = learning_store.save_feedback(
            question_id, self.user_id, self.account_id, -1, reason_code="wrong_join", sql_text="",
        )
        self.assertEqual(result["reason_code"], "wrong_join")


class AdminLearningQueueFeedbackReasonsTests(unittest.TestCase):
    def setUp(self):
        store.init_db()
        self.account_id = f"acct-lqreasons-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")
        self.user_id, _ = store.create_user(
            self.account_id, "Test User", f"lqreasons-{uuid.uuid4().hex[:8]}@example.com",
        )

    def tearDown(self):
        with store.get_db() as conn:
            conn.execute("DELETE FROM answer_feedback WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM learning_candidate WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM portal_user WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM client WHERE account_id=?", (self.account_id,))

    def test_candidate_gets_feedback_reasons_attached(self):
        question_id = f"q-{uuid.uuid4().hex[:8]}"
        learning_store.create_candidate(
            question_id, self.account_id, "what is net revenue", "SELECT 1",
            technical_score=70, evidence={},
        )
        learning_store.save_feedback(
            question_id, self.user_id, self.account_id, -1, reason_code="wrong_filter", sql_text="SELECT 1",
        )
        req = MagicMock()
        req.query_params = {}
        with patch.object(routes, "_is_auth", return_value=True):
            resp = _arun(routes.admin_learning_queue(req, self.account_id))
        ctx = getattr(resp, "context", None)
        # _resp renders a real template in production; assert via the
        # rendered body instead of internals, matching this codebase's
        # established admin-route test convention of checking output.
        body = resp.body.decode("utf-8") if hasattr(resp, "body") else ""
        self.assertIn("wrong filter", body)


if __name__ == "__main__":
    unittest.main()
