from pathlib import Path
import unittest

from core.result_cache import ResultCache


ROOT = Path(__file__).resolve().parents[1]


class GovernedResultExclusionTests(unittest.TestCase):
    def setUp(self):
        self.cache = ResultCache()
        self.session_id = "regulated-client:user-7"
        self.rows = [
            {"DOCTOR_NAME": "Dr. Priya Shah", "REVENUE": 1250.0},
            {"DOCTOR_NAME": "Dr. Kiran Rao", "REVENUE": 900.0},
            {"DOCTOR_NAME": "Dr. Anil Bose", "REVENUE": 700.0},
        ]
        self.cache.store(
            self.session_id,
            self.rows,
            question="Revenue by doctor",
            sql="SELECT DOCTOR_NAME, SUM(REVENUE) FROM governed_view",
            column_formats={"REVENUE": "currency"},
        )

    def test_tokens_are_opaque_and_aligned(self):
        tokens = self.cache.get_row_tokens(self.session_id)
        self.assertEqual(len(tokens), len(self.rows))
        self.assertEqual(len(set(tokens)), len(tokens))
        for token in tokens:
            self.assertEqual(len(token), 32)
            self.assertNotIn("Priya", token)
            self.assertNotIn("1250", token)

    def test_multiple_rows_are_excluded_without_values_in_request(self):
        tokens = self.cache.get_row_tokens(self.session_id)
        result = self.cache.exclude_rows(self.session_id, [tokens[0], tokens[2]])

        self.assertEqual(result["excluded_count"], 2)
        self.assertEqual(result["rows_before"], 3)
        self.assertEqual(result["rows_after"], 1)
        self.assertEqual(result["rows"], [self.rows[1]])
        self.assertEqual(result["column_formats"], {"REVENUE": "currency"})

    def test_tokens_expire_after_result_changes(self):
        tokens = self.cache.get_row_tokens(self.session_id)
        self.cache.exclude_rows(self.session_id, [tokens[0]])

        with self.assertRaisesRegex(ValueError, "stale"):
            self.cache.exclude_rows(self.session_id, [tokens[1]])

    def test_excluding_every_row_preserves_schema(self):
        tokens = self.cache.get_row_tokens(self.session_id)
        result = self.cache.exclude_rows(self.session_id, tokens)

        self.assertEqual(result["rows"], [])
        self.assertEqual(result["rows_after"], 0)
        self.assertEqual(
            [column["name"] for column in self.cache.get_schema(self.session_id)],
            ["DOCTOR_NAME", "REVENUE"],
        )

    def test_tokens_are_session_scoped(self):
        other_session = "regulated-client:user-8"
        self.cache.store(other_session, self.rows, question="same", sql="SELECT 1")
        token = self.cache.get_row_tokens(self.session_id)[0]

        with self.assertRaisesRegex(ValueError, "stale"):
            self.cache.exclude_rows(other_session, [token])


class GovernedResultExclusionWiringTests(unittest.TestCase):
    def test_portal_sends_only_opaque_row_tokens(self):
        template = (ROOT / "portal" / "templates" / "portal_chat.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("type: 'result_exclusion'", template)
        self.assertIn("row_tokens: selectedTokens", template)
        self.assertIn("Exclude selected", template)
        self.assertNotIn("rows: selectedRows", template)

    def test_websocket_route_records_zero_llm_proof(self):
        source = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")
        start = source.index('if msg_type == "result_exclusion":')
        end = source.index('if msg_type == "result_chat":', start)
        block = source[start:end]

        self.assertIn("result_cache.exclude_rows", block)
        self.assertIn('"rows_sent_to_llm": 0', block)
        self.assertIn('"database_queried": False', block)
        self.assertIn("record_llm_blocked", block)
        self.assertNotIn("llm_complete", block)
        self.assertNotIn("_generate_duckdb_sql", block)


if __name__ == "__main__":
    unittest.main()
