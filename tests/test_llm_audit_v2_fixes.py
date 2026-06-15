"""
Tests for LLM audit v2 fixes (smarter sanitizer, question sanitization,
retention, filters).

Run:  python -m unittest tests.test_llm_audit_v2_fixes
"""

import os
import sys
import sqlite3
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Point DB + encryption key at a temp location BEFORE importing store modules.
_tmpdir = tempfile.mkdtemp(prefix="querybot_audit_v2_")
_tmp_db = str(Path(_tmpdir) / "test_audit.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
os.environ["DB_PATH"] = _tmp_db
os.environ["QUERYBOT_KEY_FILE"] = str(Path(_tmpdir) / "test_key")
# Clear any previously imported store modules so they re-read the env vars.
for _m in list(sys.modules.keys()):
    if _m.startswith("store"):
        del sys.modules[_m]

from core.llm_audit import (
    sanitize_llm_text,
    sanitize_payload_preview,
    record_llm_call,
    llm_audit_scope,
)
from store.db import init_db
import store.db as store_db
from store import config_store


# ──────────────────────────────────────────────────────────────────────────────
# Fix #1 — Quoted-literal masking preserves readability
# ──────────────────────────────────────────────────────────────────────────────
class SmarterQuotedLiteralMaskingTests(unittest.TestCase):

    def test_short_categoricals_are_preserved(self):
        """'Active', 'Late', 'Y', 'N' should NOT be redacted — they're the
        exactly the values an admin needs to see to understand what the LLM
        was doing. They carry no PII on their own."""
        text = "WHERE status = 'Active' AND flag = 'Y' AND attendance = 'Late'"
        out = sanitize_llm_text(text)
        self.assertIn("'Active'", out)
        self.assertIn("'Y'", out)
        self.assertIn("'Late'", out)
        self.assertNotIn("[literal]", out)

    def test_proper_nouns_are_masked(self):
        """'Alice Adams', 'Bob Brown' look like person names → redact."""
        text = "User asked about 'Alice Adams' and 'Bob Brown'"
        out = sanitize_llm_text(text)
        self.assertNotIn("Alice", out)
        self.assertNotIn("Bob", out)
        self.assertIn("'[literal]'", out)

    def test_phrases_with_spaces_are_masked(self):
        """Multi-word values are almost always data, not schema vocab."""
        text = "category = 'Premium Health Plan'"
        out = sanitize_llm_text(text)
        self.assertNotIn("Premium", out)
        self.assertIn("'[literal]'", out)

    def test_long_literals_are_masked(self):
        """Any quoted string ≥10 chars is masked regardless of shape."""
        text = "id_ref = 'ABCDEFGHIJ'"  # 10 chars
        out = sanitize_llm_text(text)
        self.assertIn("'[literal]'", out)


# ──────────────────────────────────────────────────────────────────────────────
# Fix #2 — Long-token regex doesn't eat SCREAMING_SNAKE_CASE column names
# ──────────────────────────────────────────────────────────────────────────────
class LongTokenRegexTests(unittest.TestCase):

    def test_screaming_snake_case_column_names_preserved(self):
        text = "Use COMPOUND_PHARMACY_PRESCRIPTION_HISTORY.PATIENT_ID for the join"
        out = sanitize_llm_text(text)
        self.assertIn("COMPOUND_PHARMACY_PRESCRIPTION_HISTORY", out)
        self.assertNotIn("[token]", out)

    def test_base64_style_api_key_still_masked(self):
        # A real opaque secret: mixed case alnum, no snake_case shape
        text = "api_key=sk1aB2cD3eF4gH5iJ6kL7mN8oP9qR0"
        out = sanitize_llm_text(text)
        self.assertNotIn("sk1aB2cD3eF4gH5iJ6kL7mN8oP9qR0", out)
        self.assertIn("[token]", out)

    def test_long_random_hex_still_masked(self):
        text = "token=a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8"
        out = sanitize_llm_text(text)
        self.assertIn("[token]", out)


# ──────────────────────────────────────────────────────────────────────────────
# Fix #3 — Question field is sanitized before storage
# ──────────────────────────────────────────────────────────────────────────────
class QuestionSanitizationTests(unittest.TestCase):

    def setUp(self):
        # Fresh DB per test — QUERYBOT_DB_PATH is already set at module level
        # and _get_sqlite_connection reads it dynamically on every call.
        db_path = Path(os.environ["QUERYBOT_DB_PATH"])
        for path in [db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")]:
            if path.exists():
                path.unlink(missing_ok=True)
        init_db()
        # Need a parent client row because llm_call_log has FK
        config_store.upsert_client("acct_sanitize", "zoom")

    def test_question_with_email_is_sanitized_on_insert(self):
        with llm_audit_scope(
            account_id="acct_sanitize",
            question="show me alice@example.com orders",
            enabled=True,
            request_id="req_san_1",
            component="sql_generation",
        ):
            record_llm_call(
                llm_provider="openai",
                llm_model="gpt-4o-mini",
                system="sys",
                user="usr",
                status="success",
            )

        rows = config_store.get_recent_llm_calls("acct_sanitize", limit=10)
        self.assertEqual(len(rows), 1)
        self.assertNotIn("alice@example.com", rows[0]["question"])
        self.assertIn("[email]", rows[0]["question"])

    def test_question_with_phone_is_sanitized(self):
        with llm_audit_scope(
            account_id="acct_sanitize",
            question="find customer +1 555 123 4567",
            enabled=True,
            request_id="req_san_2",
            component="sql_generation",
        ):
            record_llm_call(
                llm_provider="openai", llm_model="gpt-4o",
                system="s", user="u", status="success",
            )
        rows = config_store.get_recent_llm_calls("acct_sanitize", limit=10)
        q_texts = [r["question"] for r in rows]
        self.assertFalse(any("555 123 4567" in q for q in q_texts))


# ──────────────────────────────────────────────────────────────────────────────
# Fix #5 — Retention / purge
# ──────────────────────────────────────────────────────────────────────────────
class RetentionPurgeTests(unittest.TestCase):

    def setUp(self):
        db_path = Path(os.environ["QUERYBOT_DB_PATH"])
        for path in [db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")]:
            if path.exists():
                path.unlink(missing_ok=True)
        init_db()
        config_store.upsert_client("acct_retain", "zoom")

    def _insert_row(self, created_at: str, request_id: str = "r", question_id: str = ""):
        with store_db.get_db() as conn:
            conn.execute(
                "INSERT INTO llm_call_log "
                "(account_id, question_id, request_id, question, component, "
                " llm_provider, llm_model, status, payload_hash, "
                " payload_preview_sanitized, prompt_chars, created_at) "
                "VALUES (?, ?, ?, '', 'sql_generation', '', '', 'success', '', '', 0, ?)",
                ("acct_retain", question_id or request_id, request_id, created_at),
            )

    def test_purge_deletes_rows_older_than_n_days(self):
        self._insert_row("2020-01-01 00:00:00", "old1")
        self._insert_row("2020-01-01 00:00:00", "old2")
        # A row from "now"
        self._insert_row("2099-12-31 00:00:00", "future")

        deleted = config_store.purge_old_llm_calls(retention_days=30)
        self.assertEqual(deleted, 2)

        # get_recent_llm_calls now returns grouped dicts — one group per question_id
        groups = config_store.get_recent_llm_calls("acct_retain", limit=10)
        self.assertEqual(len(groups), 1)
        # The remaining group's single call must be "future"
        self.assertEqual(groups[0]["calls"][0]["request_id"], "future")

    def test_purge_with_zero_days_is_a_noop(self):
        self._insert_row("2020-01-01 00:00:00", "old")
        deleted = config_store.purge_old_llm_calls(retention_days=0)
        self.assertEqual(deleted, 0)


# ──────────────────────────────────────────────────────────────────────────────
# Fix #6 — Audit tab filters (component, status)
# ──────────────────────────────────────────────────────────────────────────────
class AuditFilterTests(unittest.TestCase):

    def setUp(self):
        db_path = Path(os.environ["QUERYBOT_DB_PATH"])
        for path in [db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")]:
            if path.exists():
                path.unlink()
        init_db()
        config_store.upsert_client("acct_filter", "zoom")

        # Insert rows — give each a distinct question_id so they form separate
        # groups; this mirrors real usage where each question is independent.
        for comp, status, rid in [
            ("sql_generation",   "success", "r1"),
            ("sql_generation",   "error",   "r2"),
            ("clarification",    "success", "r3"),
            ("clarification",    "error",   "r4"),
            ("analysis",         "success", "r5"),
        ]:
            config_store.log_llm_call(
                account_id="acct_filter",
                request_id=rid,
                question="",
                component=comp,
                llm_provider="openai",
                llm_model="gpt-4o",
                status=status,
                payload_hash="",
                payload_preview_sanitized="",
                prompt_chars=0,
                question_id=rid,  # distinct per row → one group each
            )

    def _all_calls(self, groups):
        """Flatten grouped result back to individual call dicts for easy assertions."""
        return [call for g in groups for call in g["calls"]]

    def test_no_filter_returns_all(self):
        groups = config_store.get_recent_llm_calls("acct_filter", limit=100)
        calls = self._all_calls(groups)
        self.assertEqual(len(calls), 5)

    def test_filter_by_component(self):
        groups = config_store.get_recent_llm_calls(
            "acct_filter", limit=100, component="sql_generation"
        )
        calls = self._all_calls(groups)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(c["component"] == "sql_generation" for c in calls))

    def test_filter_by_status(self):
        groups = config_store.get_recent_llm_calls(
            "acct_filter", limit=100, status="error"
        )
        calls = self._all_calls(groups)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(c["status"] == "error" for c in calls))

    def test_filter_by_component_and_status(self):
        groups = config_store.get_recent_llm_calls(
            "acct_filter", limit=100,
            component="clarification", status="error",
        )
        calls = self._all_calls(groups)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["request_id"], "r4")


if __name__ == "__main__":
    unittest.main()
