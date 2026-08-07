"""
LLM egress hardening — Phase 2: the per-call egress manifest.

The manifest is the structured answer to "what went to the model on this call,
and did any data values go with it". Its whole value rests on values_sent being
DERIVED from the assembled prompt rather than declared by the call site — a
declared flag can be forgotten, a derived one cannot. These tests pin that.
"""

import json
import unittest
from unittest.mock import patch

from core.llm_audit import build_egress_manifest, llm_audit_scope, record_llm_call


class ManifestDerivationTests(unittest.TestCase):
    """values_sent must come from the prompt text, not from a caller's claim."""

    def test_clean_schema_prompt_reports_no_values(self):
        m = build_egress_manifest(
            "You are a SQL generator.\nCOLUMN SYNONYM MAP\nBUSINESS TERM DEFINITIONS",
            "net revenue by product last quarter",
            None,
        )
        self.assertFalse(m["values_sent"])
        self.assertEqual(m["value_sources"], [])

    def test_value_index_block_is_detected(self):
        prompt = (
            "You are a SQL generator.\n"
            "VERIFIED FILTER VALUES (matched against actual database contents):\n"
            "user text 'emco' -> DB.SCH.CUSTOMER.NAME = 'EMCO Corporation'\n"
        )
        m = build_egress_manifest(prompt, "q", None)
        self.assertTrue(m["values_sent"])
        self.assertIn("value_index", m["value_sources"])

    def test_repair_prompt_reports_echoed_sql(self):
        user = (
            "The following SQL failed with this error:\n"
            "SQL: SELECT * FROM T WHERE PRODUCT = 'Lipitor'\n"
            "Error: Conversion failed\n"
        )
        m = build_egress_manifest("", user, None)
        self.assertTrue(m["values_sent"])
        self.assertIn("echoed_sql", m["value_sources"])

    def test_caller_cannot_declare_away_a_detected_leak(self):
        """The scope is descriptive only — it must not be able to claim safety."""
        prompt = "VERIFIED FILTER VALUES (matched against actual database contents):\n"
        scope = {"egress": {"tables": ["T"], "columns": ["T.C"], "values_sent": False}}
        m = build_egress_manifest(prompt, "q", scope)
        self.assertTrue(m["values_sent"], "scope overrode a derived leak")

    def test_content_categories_are_listed(self):
        m = build_egress_manifest(
            "COLUMN SYNONYM MAP\nBUSINESS TERM DEFINITIONS\nSession context",
            "a question",
            None,
        )
        self.assertIn("question", m["content"])
        self.assertIn("synonyms", m["content"])
        self.assertIn("business_terms", m["content"])
        self.assertIn("conversation_history", m["content"])

    def test_tables_and_columns_come_from_scope(self):
        scope = {"egress": {
            "tables": ["PHARMA_LAB.F_RX_FILL", "PHARMA_LAB.D_PRODUCT"],
            "columns": ["PHARMA_LAB.F_RX_FILL.FILL_DATE"],
        }}
        m = build_egress_manifest("sys", "q", scope)
        self.assertEqual(
            m["tables"], ["PHARMA_LAB.D_PRODUCT", "PHARMA_LAB.F_RX_FILL"]
        )
        self.assertEqual(m["columns"], ["PHARMA_LAB.F_RX_FILL.FILL_DATE"])

    def test_manifest_is_bounded(self):
        scope = {"egress": {
            "tables": [f"T{i}" for i in range(500)],
            "columns": [f"T.C{i}" for i in range(5000)],
        }}
        m = build_egress_manifest("sys", "q", scope)
        self.assertLessEqual(len(m["tables"]), 50)
        self.assertLessEqual(len(m["columns"]), 200)

    def test_never_raises_on_malformed_scope(self):
        for bad in ({"egress": {"tables": None}}, {"egress": "nope"}, {"egress": None}):
            m = build_egress_manifest("s", "u", bad)  # type: ignore[arg-type]
            self.assertIn("values_sent", m)

    def test_missing_scope_still_derives_the_flag(self):
        """An un-instrumented call site must still be reported honestly."""
        m = build_egress_manifest("VERIFIED FILTER VALUES", "q", None)
        self.assertTrue(m["values_sent"])
        self.assertEqual(m["tables"], [])


class ManifestPersistenceTests(unittest.TestCase):
    """record_llm_call must write the manifest as JSON on every audited call."""

    def test_manifest_is_written_to_the_audit_row(self):
        captured: dict = {}

        def _fake_log(**kwargs):
            captured.update(kwargs)

        with llm_audit_scope(
            account_id="acct-1",
            question="net revenue by product",
            enabled=True,
            component="sql_generation",
            egress={"tables": ["SALES.ORDERS"], "columns": ["SALES.ORDERS.AMOUNT"]},
        ):
            with patch("store.log_llm_call", _fake_log):
                record_llm_call(
                    llm_provider="anthropic",
                    llm_model="claude",
                    system="COLUMN SYNONYM MAP",
                    user="net revenue by product",
                    status="success",
                    response="SELECT 1",
                )

        self.assertIn("egress_manifest", captured)
        manifest = json.loads(captured["egress_manifest"])
        self.assertEqual(manifest["tables"], ["SALES.ORDERS"])
        self.assertEqual(manifest["columns"], ["SALES.ORDERS.AMOUNT"])
        self.assertFalse(manifest["values_sent"])

    def test_leaky_prompt_is_recorded_as_such(self):
        captured: dict = {}

        with llm_audit_scope(
            account_id="acct-1", question="q", enabled=True, component="sql_generation",
        ):
            with patch("store.log_llm_call", lambda **kw: captured.update(kw)):
                record_llm_call(
                    llm_provider="p", llm_model="m",
                    system="VERIFIED FILTER VALUES (matched against actual database contents):",
                    user="q", status="success",
                )

        manifest = json.loads(captured["egress_manifest"])
        self.assertTrue(manifest["values_sent"])
        self.assertEqual(manifest["value_sources"], ["value_index"])


class SchemaAndStoreTests(unittest.TestCase):
    """The column must exist on both DDL copies and survive a round trip."""

    def test_egress_manifest_column_exists(self):
        from store.db import get_db, init_db

        init_db()
        with get_db() as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(llm_call_log)")}
        self.assertIn("egress_manifest", cols)

    def test_both_ddl_copies_declare_the_column(self):
        """store/db.py keeps a second llm_call_log DDL in the migration path;
        a column added to only one silently diverges on fresh installs."""
        from pathlib import Path

        src = (Path(__file__).resolve().parents[1] / "store" / "db.py").read_text(
            encoding="utf-8"
        )
        blocks = src.split("CREATE TABLE IF NOT EXISTS llm_call_log")[1:]
        self.assertGreaterEqual(len(blocks), 2, "expected two DDL copies to guard")
        for i, block in enumerate(blocks):
            body = block.split(");", 1)[0]
            self.assertIn(
                "egress_manifest", body,
                f"llm_call_log DDL copy #{i + 1} is missing egress_manifest",
            )

    def test_reader_decodes_manifest_and_rolls_up(self):
        """get_recent_llm_calls must expose a dict per call and a group rollup."""
        from pathlib import Path

        src = (Path(__file__).resolve().parents[1] / "store" / "config_store.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('r["egress"] = ', src)
        self.assertIn('"any_values_sent"', src)
        self.assertIn('"value_sources"', src)


if __name__ == "__main__":
    unittest.main()
