"""
LLM egress hardening — Phase 4: audit completeness.

Four gaps from docs/LLM_EGRESS_PLAN.md §3:
  * KB builds shared one request_id across every table, so per-table calls
    could not be told apart.
  * kb_data_egress_log had no correlation id to llm_call_log, and no purge.
  * The tenant's exported copy of the audit trail omitted the whole response
    side, and the egress manifest.
  * There was no offline export for an auditor.
"""

import unittest
from pathlib import Path

from core.llm_audit import (
    get_current_llm_audit_scope,
    llm_audit_component,
    llm_audit_scope,
)

ROOT = Path(__file__).resolve().parents[1]


class KbBuildTraceabilityTests(unittest.TestCase):
    """Each KB component must be individually identifiable."""

    def test_component_can_mint_its_own_request_id(self):
        with llm_audit_scope(
            account_id="a", question="KB build", enabled=True,
            request_id="build-1", question_id="build-1", component="kb_build",
        ):
            with llm_audit_component("kb_table_doc", question="ORDERS", new_request_id=True):
                inner = get_current_llm_audit_scope() or {}
                self.assertNotEqual(inner.get("request_id"), "build-1")
                self.assertEqual(inner.get("question"), "ORDERS")

    def test_grouping_survives_a_fresh_request_id(self):
        """question_id is what buckets the admin view — it must be inherited."""
        with llm_audit_scope(
            account_id="a", question="KB build", enabled=True,
            request_id="build-1", question_id="build-1", component="kb_build",
        ):
            with llm_audit_component("kb_table_doc", question="ORDERS", new_request_id=True):
                self.assertEqual(
                    (get_current_llm_audit_scope() or {}).get("question_id"), "build-1"
                )

    def test_two_components_get_distinct_request_ids(self):
        seen = []
        with llm_audit_scope(
            account_id="a", question="KB build", enabled=True,
            request_id="build-1", question_id="build-1", component="kb_build",
        ):
            for table in ("ORDERS", "CUSTOMERS"):
                with llm_audit_component("kb_table_doc", question=table, new_request_id=True):
                    seen.append((get_current_llm_audit_scope() or {}).get("request_id"))
        self.assertEqual(len(set(seen)), 2, "per-table calls share a request_id")

    def test_default_still_inherits(self):
        """Existing callers must be unaffected unless they opt in."""
        with llm_audit_scope(
            account_id="a", question="q", enabled=True,
            request_id="req-1", question_id="req-1", component="parent",
        ):
            with llm_audit_component("child"):
                self.assertEqual(
                    (get_current_llm_audit_scope() or {}).get("request_id"), "req-1"
                )

    def test_all_kb_components_opt_in(self):
        src = (ROOT / "core" / "knowledge.py").read_text(encoding="utf-8")
        for component in (
            "kb_business_vocab", "kb_table_doc", "kb_query_examples", "kb_query_repair",
        ):
            idx = src.index(f'"{component}"')
            window = src[idx:idx + 200]
            self.assertIn(
                "new_request_id=True", window,
                f"{component} still inherits the build-wide request_id",
            )

    def test_kb_build_scope_sets_an_explicit_question_id(self):
        src = (ROOT / "admin" / "routes.py").read_text(encoding="utf-8")
        idx = src.index('component="kb_build"')
        block = src[max(0, idx - 700):idx]
        self.assertIn("question_id=_kb_build_id", block)
        self.assertIn("request_id=_kb_build_id", block)


class EgressLogCorrelationTests(unittest.TestCase):
    """kb_data_egress_log must be linkable to llm_call_log, and purgeable."""

    def test_request_id_column_exists(self):
        from store.db import get_db, init_db

        init_db()
        with get_db() as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(kb_data_egress_log)")}
        self.assertIn("request_id", cols)

    def test_log_kb_egress_accepts_request_id(self):
        import inspect

        from store.config_store import log_kb_egress

        self.assertIn("request_id", inspect.signature(log_kb_egress).parameters)

    def test_purge_exists_and_is_exported(self):
        import store

        self.assertTrue(hasattr(store, "purge_old_kb_egress"))
        self.assertEqual(store.purge_old_kb_egress(0), 0, "non-positive retention must no-op")

    def test_purge_retention_is_longer_than_llm_calls(self):
        """Egress rows are the long-lived compliance record; purging them on the
        30-day LLM-call schedule would orphan half of every story."""
        import inspect

        from store.config_store import purge_old_kb_egress, purge_old_llm_calls

        egress_default = inspect.signature(purge_old_kb_egress).parameters["retention_days"].default
        llm_default = inspect.signature(purge_old_llm_calls).parameters["retention_days"].default
        self.assertGreater(egress_default, llm_default)

    def test_purge_runs_at_startup(self):
        src = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("purge_old_kb_egress", src)


class ExportCompletenessTests(unittest.TestCase):
    """The tenant's exported copy must carry the response side and manifest."""

    def test_llm_columns_include_response_and_manifest(self):
        from core.log_export import LLM_COLUMNS

        for col in (
            "RESPONSE_HASH", "RESPONSE_PREVIEW_SANITIZED",
            "RESPONSE_CHARS", "EGRESS_MANIFEST",
        ):
            self.assertIn(col, LLM_COLUMNS)

    def test_egress_columns_include_request_id(self):
        from core.log_export import EGRESS_COLUMNS

        self.assertIn("REQUEST_ID", EGRESS_COLUMNS)

    def test_fetched_row_arity_matches_column_lists(self):
        """A SELECT/column-list mismatch silently corrupts every exported row —
        values land under the wrong headers. Asserted against real fetched rows
        rather than by counting commas in the source, which cannot distinguish a
        column separator from one inside a COALESCE."""
        from store.db import get_db, init_db

        from core.log_export import (
            EGRESS_COLUMNS, LLM_COLUMNS,
            _fetch_egress_rows_after, _fetch_llm_rows_after,
        )

        init_db()
        # Seed one row of each so there is something to shape-check. Both tables
        # are append-only audit logs, so a probe row is harmless.
        with get_db() as conn:
            # llm_call_log.account_id is an FK to client; platform_type is the
            # one NOT NULL column without a default.
            conn.execute(
                "INSERT OR IGNORE INTO client (account_id, client_name, platform_type) "
                "VALUES (?, ?, ?)",
                ("acct-export-arity", "export arity probe", "portal"),
            )
            conn.execute(
                "INSERT INTO llm_call_log (account_id, request_id, component, status) "
                "VALUES (?, ?, ?, ?)",
                ("acct-export-arity", "req-arity", "sql_generation", "success"),
            )
            conn.execute(
                "INSERT INTO kb_data_egress_log "
                "(account_id, operation, db_type, table_name, sample_mode) "
                "VALUES (?, ?, ?, ?, ?)",
                ("acct-export-arity", "kb_build", "azure_sql", "T", "none"),
            )

        llm_rows = _fetch_llm_rows_after(0, 1)
        self.assertTrue(llm_rows, "no llm_call_log row fetched to shape-check")
        self.assertEqual(
            len(llm_rows[0]), len(LLM_COLUMNS),
            "LLM export SELECT and LLM_COLUMNS have diverged",
        )

        eg_rows = _fetch_egress_rows_after(0, 1)
        self.assertTrue(eg_rows, "no kb_data_egress_log row fetched to shape-check")
        self.assertEqual(
            len(eg_rows[0]), len(EGRESS_COLUMNS),
            "egress export SELECT and EGRESS_COLUMNS have diverged",
        )


class AuditorExportTests(unittest.TestCase):
    """An auditor needs the evidence offline, not only in the admin table."""

    def test_csv_route_exists_and_is_authenticated(self):
        src = (ROOT / "admin" / "routes.py").read_text(encoding="utf-8")
        idx = src.index('/clients/{account_id}/llm-audit.csv')
        block = src[idx:idx + 4000]
        self.assertIn("_is_auth(request)", block)
        self.assertIn("text/csv", block)

    def test_csv_exposes_the_manifest_verdict(self):
        src = (ROOT / "admin" / "routes.py").read_text(encoding="utf-8")
        idx = src.index('/clients/{account_id}/llm-audit.csv')
        block = src[idx:idx + 3500]
        self.assertIn("Data values sent", block)
        self.assertIn("Value sources", block)

    def test_csv_does_not_export_raw_prompt_text(self):
        """Exporting must not widen exposure beyond what the log already holds."""
        src = (ROOT / "admin" / "routes.py").read_text(encoding="utf-8")
        idx = src.index('/clients/{account_id}/llm-audit.csv')
        block = src[idx:idx + 3500]
        select = block[block.index("SELECT"):block.index("FROM llm_call_log")]
        self.assertNotIn("payload_preview_sanitized", select.replace("payload_hash", ""))
        for forbidden in ("system", "user_prompt", "raw_prompt"):
            self.assertNotIn(forbidden, select)

    def test_admin_page_links_to_the_export(self):
        html = (ROOT / "admin" / "templates" / "client_detail.html").read_text(encoding="utf-8")
        self.assertIn("llm-audit.csv", html)


if __name__ == "__main__":
    unittest.main()
