"""
tests/test_kb_egress_llm_link.py

Links a kb_data_egress_log row (proves which fields were masked at KB-build
time) to the exact LLM Audit Log entry (proves what masked data was actually
sent in that table's KB-doc-generation prompt) -- closing the gap where an
admin had to manually cross-reference by table name and timestamp between
two separate tabs.

The naive design (a shared request_id column on kb_data_egress_log) doesn't
work: the llm_audit_scope for an entire KB build is opened once and shared
across every table in that build, so request_id alone can't disambiguate
which table a given audit row belongs to. The actual fix tags each table's
kb_table_doc call with question=table_name (core/knowledge.py), then
get_kb_table_doc_audit() looks it up directly -- no schema migration needed.
"""
import os, sys, tempfile, unittest
from pathlib import Path

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_kb_egress_llm_link.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for mod in list(sys.modules.keys()):
    if mod.startswith("store"):
        del sys.modules[mod]
import store.db as db_mod
db_mod.init_db()
import store

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PY = ROOT / "core" / "knowledge.py"
STORE_INIT = ROOT / "store" / "__init__.py"


class GetKbTableDocAuditTests(unittest.TestCase):
    def setUp(self):
        self.account_id = "acct-egress-link-test"
        store.upsert_client(self.account_id, "portal")

    def tearDown(self):
        with store.get_db() as conn:
            conn.execute("DELETE FROM client WHERE account_id = ?", (self.account_id,))

    def _log_kb_table_doc_call(self, table_name: str, *, request_id: str = "req-1",
                                payload: str = "sanitized prompt text"):
        store.log_llm_call(
            account_id=self.account_id,
            request_id=request_id,
            question=table_name,
            component="kb_table_doc",
            llm_provider="anthropic",
            llm_model="claude-sonnet-5",
            status="success",
            payload_hash="deadbeef",
            payload_preview_sanitized=payload,
        )

    def test_no_matching_row_returns_none(self):
        self.assertIsNone(store.get_kb_table_doc_audit(self.account_id, "F_RX_FILL"))

    def test_finds_the_matching_table(self):
        self._log_kb_table_doc_call("F_RX_FILL", payload="masked sample values for F_RX_FILL")
        result = store.get_kb_table_doc_audit(self.account_id, "F_RX_FILL")
        self.assertIsNotNone(result)
        self.assertEqual(result["question"], "F_RX_FILL")
        self.assertEqual(result["payload_preview_sanitized"], "masked sample values for F_RX_FILL")

    def test_does_not_cross_match_a_different_table(self):
        self._log_kb_table_doc_call("F_RX_FILL")
        self.assertIsNone(store.get_kb_table_doc_audit(self.account_id, "F_RX_ORDER"))

    def test_ignores_rows_from_other_components(self):
        store.log_llm_call(
            account_id=self.account_id,
            request_id="req-2",
            question="F_RX_FILL",
            component="sql_generation",
            llm_provider="anthropic",
            llm_model="claude-sonnet-5",
            status="success",
            payload_hash="abc123",
            payload_preview_sanitized="unrelated sql-generation prompt",
        )
        self.assertIsNone(store.get_kb_table_doc_audit(self.account_id, "F_RX_FILL"))

    def test_returns_most_recent_when_table_rebuilt_multiple_times(self):
        self._log_kb_table_doc_call("F_RX_FILL", request_id="req-old", payload="first build")
        self._log_kb_table_doc_call("F_RX_FILL", request_id="req-new", payload="second build (rebuilt)")
        result = store.get_kb_table_doc_audit(self.account_id, "F_RX_FILL")
        self.assertEqual(result["payload_preview_sanitized"], "second build (rebuilt)")

    def test_shared_request_id_across_tables_does_not_cause_confusion(self):
        # Regression guard for the exact bug this feature works around: one
        # KB build shares a single request_id across every table's call.
        self._log_kb_table_doc_call("F_RX_FILL", request_id="shared-build-req", payload="fill doc")
        self._log_kb_table_doc_call("F_RX_ORDER", request_id="shared-build-req", payload="order doc")
        fill_result = store.get_kb_table_doc_audit(self.account_id, "F_RX_FILL")
        order_result = store.get_kb_table_doc_audit(self.account_id, "F_RX_ORDER")
        self.assertEqual(fill_result["payload_preview_sanitized"], "fill doc")
        self.assertEqual(order_result["payload_preview_sanitized"], "order doc")


class WiringGuardTests(unittest.TestCase):
    def test_knowledge_py_tags_kb_table_doc_with_table_name(self):
        src = KNOWLEDGE_PY.read_text(encoding="utf-8")
        self.assertIn('llm_audit_component("kb_table_doc", question=table_name)', src)

    def test_store_exports_get_kb_table_doc_audit(self):
        src = STORE_INIT.read_text(encoding="utf-8")
        self.assertIn("get_kb_table_doc_audit", src)
        self.assertTrue(callable(store.get_kb_table_doc_audit))


if __name__ == "__main__":
    unittest.main()
