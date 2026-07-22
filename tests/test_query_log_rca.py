"""Admin portal failure visibility.

The Query Log / Overview "Recent errors" panels previously showed only the
raw internal error_msg string (or a hover tooltip of it) for a failed
query -- useful for a developer, opaque for anyone else. query_log now
also stores error_code (the validator/execution classification), and the
admin route re-uses core.failure_messages.translate_failure -- the exact
same function chat uses -- to render the same business-readable
explanation admins would otherwise only see via the trace viewer.

Also covers a connected chat-side gap found while building this: several
validator codes introduced in earlier sessions (graph_plan_mismatch,
surrogate_date_conversion, etc.) were missing from _VALIDATION_REASONS, so
a terminal failure with one of those codes fell through to a raw
"❌ {reason}" message in chat too, instead of the translated one.
"""
from __future__ import annotations

import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class QueryLogErrorCodeStoreTests(unittest.TestCase):
    def setUp(self):
        import store
        store.init_db()
        self.store = store
        self.account_id = f"acct-qlrca-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

    def tearDown(self):
        with self.store.get_db() as conn:
            conn.execute("DELETE FROM client WHERE account_id = ?", (self.account_id,))

    def test_error_code_round_trips_on_failed_query(self):
        self.store.log_query(
            account_id=self.account_id,
            question="total net revenue by pharmacy for 2026",
            sql_generated="SELECT ... surrogate misuse ...",
            row_count=0,
            success=False,
            error_msg="DISPENSE_DATE_ID is a surrogate date-dimension key...",
            error_code="surrogate_date_conversion",
        )
        rows = self.store.get_recent_queries(self.account_id, limit=5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["error_code"], "surrogate_date_conversion")
        self.assertFalse(rows[0]["success"])

    def test_error_code_defaults_to_empty_string(self):
        self.store.log_query(
            account_id=self.account_id,
            question="q", sql_generated="SELECT 1",
            row_count=1, success=True,
        )
        rows = self.store.get_recent_queries(self.account_id, limit=5)
        self.assertEqual(rows[0]["error_code"], "")


class FailureMessagesNewCodesTests(unittest.TestCase):
    """The 8 retryable codes that existed but were missing from
    _VALIDATION_REASONS -- confirms each now gets a specific headline
    instead of translate_failure's generic unmapped-code fallback."""

    NEW_CODES = [
        "dialect_mismatch", "production_shape", "top_n_shape",
        "graph_plan_mismatch", "multi_statement", "not_select",
        "reused_plan_empty", "surrogate_date_conversion",
    ]
    GENERIC_FALLBACK = "The generated query did not pass QueryBot's safety and accuracy checks."

    def test_each_new_code_has_a_specific_headline(self):
        from core.failure_messages import translate_failure
        for code in self.NEW_CODES:
            with self.subTest(code=code):
                rca = translate_failure(kind="validation", code=code, reason="technical detail")
                self.assertNotEqual(rca["most_likely_reason"], self.GENERIC_FALLBACK)
                self.assertIn(f"Validation: {code}", rca["technical_notes"])

    def test_truly_unknown_code_still_gets_safe_generic_fallback(self):
        from core.failure_messages import translate_failure
        rca = translate_failure(kind="validation", code="some_future_code", reason="x")
        self.assertEqual(rca["most_likely_reason"], self.GENERIC_FALLBACK)


class AttachFailureRcaHelperTests(unittest.TestCase):
    def test_validator_code_uses_validation_kind(self):
        from admin.routes import _attach_failure_rca
        queries = [{
            "success": False, "error_code": "unknown_column",
            "error_msg": "raw technical text", "sql_generated": "SELECT x",
            "question": "q",
        }]
        _attach_failure_rca(queries)
        rca = queries[0]["rca"]
        self.assertEqual(
            rca["most_likely_reason"],
            "The generated query used a column that does not exist in your data.",
        )

    def test_execution_error_code_uses_execution_kind(self):
        from admin.routes import _attach_failure_rca
        queries = [{
            "success": False, "error_code": "execution_error",
            "error_msg": "Invalid object name 'FOO'.", "sql_generated": "SELECT * FROM FOO",
            "question": "q",
        }]
        _attach_failure_rca(queries)
        rca = queries[0]["rca"]
        self.assertEqual(rca["headline"], "I could not run this query against your database.")

    def test_missing_error_code_falls_back_to_execution_kind(self):
        from admin.routes import _attach_failure_rca
        queries = [{"success": False, "error_code": "", "error_msg": "some old row", "sql_generated": "", "question": "q"}]
        _attach_failure_rca(queries)
        self.assertEqual(queries[0]["rca"]["headline"], "I could not run this query against your database.")

    def test_successful_rows_are_left_untouched(self):
        from admin.routes import _attach_failure_rca
        queries = [{"success": True, "error_code": "", "error_msg": "", "sql_generated": "SELECT 1", "question": "q"}]
        _attach_failure_rca(queries)
        self.assertNotIn("rca", queries[0])


if __name__ == "__main__":
    unittest.main()
