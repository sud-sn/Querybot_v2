"""
tests/test_kb_egress_log.py

Tests for the KB data egress transparency log:
  1. DB table created on init_db()
  2. log_kb_egress() writes correct rows
  3. list_kb_egress() reads and filters correctly
  4. get_kb_egress_summary() aggregates correctly
  5. Architecture guards — template and routes wired
"""
import os, sys, tempfile, unittest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_egress.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for mod in list(sys.modules.keys()):
    if mod.startswith("store"):
        del sys.modules[mod]
import store.db as db_mod
db_mod.init_db()
import store

SETUP_TMPL  = os.path.join(os.path.dirname(__file__), "..", "admin", "templates", "client_setup.html")
ROUTES_PY   = os.path.join(os.path.dirname(__file__), "..", "admin", "routes.py")
DB_PY       = os.path.join(os.path.dirname(__file__), "..", "store", "db.py")
STORE_INIT  = os.path.join(os.path.dirname(__file__), "..", "store", "__init__.py")
CS_PY       = os.path.join(os.path.dirname(__file__), "..", "store", "config_store.py")


# ── 1  DB table ───────────────────────────────────────────────────────────────
class TestDbTable(unittest.TestCase):

    def test_table_exists_after_init(self):
        with db_mod.get_db() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='kb_data_egress_log'"
            ).fetchone()
        self.assertIsNotNone(row, "kb_data_egress_log table must exist after init_db()")

    def test_table_has_required_columns(self):
        with db_mod.get_db() as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(kb_data_egress_log)").fetchall()}
        for col in ("id","account_id","operation","db_type","table_name",
                    "sample_mode","column_count","distinct_col_count",
                    "triggered_by","created_at","database_name","schema_name"):
            self.assertIn(col, cols, f"Missing column: {col}")

    def test_index_on_account_op(self):
        src = open(DB_PY).read()
        self.assertIn("idx_kb_egress_account_op", src)

    def test_schema_in_db_py(self):
        src = open(DB_PY).read()
        self.assertIn("CREATE TABLE IF NOT EXISTS kb_data_egress_log", src)
        self.assertIn("sample_mode", src)
        self.assertIn("distinct_col_count", src)


# ── 2  log_kb_egress ──────────────────────────────────────────────────────────
class TestLogKbEgress(unittest.TestCase):

    ACC = "test_acc_egress_001"

    def test_writes_synthetic_row(self):
        store.log_kb_egress(
            account_id=self.ACC, operation="kb_build", db_type="azure_sql",
            table_name="FACT_RXFILL", sample_mode="synthetic",
            schema_name="dbo", database_name="CHATBOT_DB",
            column_count=12, distinct_col_count=3,
        )
        rows = store.list_kb_egress(self.ACC, operation="kb_build")
        tbl_rows = [r for r in rows if r["table_name"] == "FACT_RXFILL"]
        self.assertTrue(len(tbl_rows) >= 1)
        self.assertEqual(tbl_rows[0]["sample_mode"], "synthetic")

    def test_writes_real_row(self):
        store.log_kb_egress(
            account_id=self.ACC, operation="kb_build", db_type="azure_sql",
            table_name="DIM_PRODUCT", sample_mode="real",
            column_count=5,
        )
        rows = store.list_kb_egress(self.ACC, operation="kb_build")
        tbl_rows = [r for r in rows if r["table_name"] == "DIM_PRODUCT"]
        self.assertTrue(len(tbl_rows) >= 1)
        self.assertEqual(tbl_rows[0]["sample_mode"], "real")

    def test_writes_discovery_row(self):
        store.log_kb_egress(
            account_id=self.ACC, operation="discovery", db_type="snowflake",
            table_name="DIM_DATE", sample_mode="none",
            column_count=8,
        )
        rows = store.list_kb_egress(self.ACC, operation="discovery")
        disc_rows = [r for r in rows if r["table_name"] == "DIM_DATE"]
        self.assertTrue(len(disc_rows) >= 1)
        self.assertEqual(disc_rows[0]["operation"], "discovery")

    def test_all_fields_stored(self):
        store.log_kb_egress(
            account_id=self.ACC, operation="kb_build", db_type="oracle",
            table_name="ORDERS", sample_mode="synthetic",
            database_name="PRODDB", schema_name="SALES",
            column_count=9, distinct_col_count=2, triggered_by="admin",
        )
        rows = store.list_kb_egress(self.ACC, operation="kb_build")
        r = next((r for r in rows if r["table_name"] == "ORDERS"), None)
        self.assertIsNotNone(r)
        self.assertEqual(r["database_name"], "PRODDB")
        self.assertEqual(r["schema_name"],   "SALES")
        self.assertEqual(r["column_count"],   9)
        self.assertEqual(r["distinct_col_count"], 2)
        self.assertEqual(r["triggered_by"],  "admin")
        self.assertIn("created_at", r)

    def test_does_not_raise_on_bad_account(self):
        """log_kb_egress must swallow errors gracefully — non-critical path."""
        # This should not raise even if something goes wrong
        try:
            store.log_kb_egress(
                account_id="", operation="kb_build", db_type="",
                table_name="", sample_mode="none",
            )
        except Exception as e:
            self.fail(f"log_kb_egress raised unexpectedly: {e}")

    def test_exported_from_store(self):
        self.assertTrue(callable(store.log_kb_egress))


# ── 3  list_kb_egress ─────────────────────────────────────────────────────────
class TestListKbEgress(unittest.TestCase):

    ACC = "test_acc_egress_002"

    def setUp(self):
        for tname, op, mode in [
            ("TABLE_A", "discovery", "none"),
            ("TABLE_B", "kb_build",  "synthetic"),
            ("TABLE_C", "kb_build",  "real"),
            ("TABLE_D", "discovery", "none"),
        ]:
            store.log_kb_egress(
                account_id=self.ACC, operation=op,
                db_type="azure_sql", table_name=tname, sample_mode=mode,
            )

    def test_returns_list(self):
        rows = store.list_kb_egress(self.ACC)
        self.assertIsInstance(rows, list)

    def test_all_rows_for_account(self):
        rows = store.list_kb_egress(self.ACC)
        self.assertGreaterEqual(len(rows), 4)

    def test_filter_discovery(self):
        rows = store.list_kb_egress(self.ACC, operation="discovery")
        ops  = {r["operation"] for r in rows}
        self.assertEqual(ops, {"discovery"})

    def test_filter_kb_build(self):
        rows = store.list_kb_egress(self.ACC, operation="kb_build")
        ops  = {r["operation"] for r in rows}
        self.assertEqual(ops, {"kb_build"})

    def test_isolation_by_account(self):
        other = store.list_kb_egress("completely_different_account")
        tables = {r["table_name"] for r in other}
        self.assertNotIn("TABLE_A", tables)

    def test_rows_are_dicts(self):
        rows = store.list_kb_egress(self.ACC)
        for r in rows:
            self.assertIsInstance(r, dict)
            self.assertIn("table_name", r)
            self.assertIn("sample_mode", r)

    def test_exported_from_store(self):
        self.assertTrue(callable(store.list_kb_egress))


# ── 4  get_kb_egress_summary ──────────────────────────────────────────────────
class TestGetKbEgressSummary(unittest.TestCase):

    ACC = "test_acc_egress_003"

    def setUp(self):
        ops = [
            ("EMPLOYEE",  "kb_build",  "synthetic"),
            ("DIM_DATE",  "kb_build",  "real"),
            ("DIM_STORE", "kb_build",  "real"),
            ("EMPLOYEE",  "discovery", "none"),
            ("DIM_DATE",  "discovery", "none"),
        ]
        for tname, op, mode in ops:
            store.log_kb_egress(
                account_id=self.ACC, operation=op,
                db_type="azure_sql", table_name=tname, sample_mode=mode,
                column_count=5,
            )

    def test_returns_dict(self):
        s = store.get_kb_egress_summary(self.ACC)
        self.assertIsInstance(s, dict)

    def test_counts_total_tables_kb_build(self):
        s = store.get_kb_egress_summary(self.ACC)
        self.assertGreaterEqual(s.get("total_tables_kb_build", 0), 3)

    def test_counts_total_tables_discovery(self):
        s = store.get_kb_egress_summary(self.ACC)
        self.assertGreaterEqual(s.get("total_tables_discovery", 0), 2)

    def test_counts_synthetic(self):
        s = store.get_kb_egress_summary(self.ACC)
        self.assertGreaterEqual(s.get("synthetic_sample_count", 0), 1)

    def test_counts_real(self):
        s = store.get_kb_egress_summary(self.ACC)
        self.assertGreaterEqual(s.get("real_sample_count", 0), 2)

    def test_last_timestamps_present(self):
        s = store.get_kb_egress_summary(self.ACC)
        self.assertIn("last_discovery_at", s)
        self.assertIn("last_kb_build_at",  s)

    def test_discovery_rows_list(self):
        s = store.get_kb_egress_summary(self.ACC)
        self.assertIn("discovery_rows", s)
        self.assertIsInstance(s["discovery_rows"], list)

    def test_kb_build_rows_list(self):
        s = store.get_kb_egress_summary(self.ACC)
        self.assertIn("kb_build_rows", s)
        self.assertIsInstance(s["kb_build_rows"], list)

    def test_empty_account_returns_zeros(self):
        s = store.get_kb_egress_summary("nonexistent_account_xyz")
        self.assertEqual(s.get("real_sample_count", 0), 0)
        self.assertEqual(s.get("synthetic_sample_count", 0), 0)

    def test_exported_from_store(self):
        self.assertTrue(callable(store.get_kb_egress_summary))


# ── 5  Architecture guards ─────────────────────────────────────────────────────
class TestEgressArchitectureGuards(unittest.TestCase):

    def test_store_exports_log_kb_egress(self):
        src = open(STORE_INIT).read()
        self.assertIn("log_kb_egress", src)

    def test_store_exports_list_kb_egress(self):
        src = open(STORE_INIT).read()
        self.assertIn("list_kb_egress", src)

    def test_store_exports_get_kb_egress_summary(self):
        src = open(STORE_INIT).read()
        self.assertIn("get_kb_egress_summary", src)

    def test_config_store_has_log_fn(self):
        src = open(CS_PY).read()
        self.assertIn("def log_kb_egress", src)
        self.assertIn("def list_kb_egress", src)
        self.assertIn("def get_kb_egress_summary", src)

    def test_routes_wire_egress_after_discover(self):
        src = open(ROUTES_PY).read()
        self.assertIn("store.log_kb_egress", src)
        # Called for both discovery and kb_build operations
        self.assertIn('operation="discovery"', src)
        self.assertIn('operation="kb_build"', src)

    def test_routes_wire_egress_after_build(self):
        src = open(ROUTES_PY).read()
        # should_use_synthetic used to determine sample_mode in kb_build
        self.assertIn("should_use_synthetic", src)

    def test_routes_has_egress_log_api(self):
        src = open(ROUTES_PY).read()
        self.assertIn("/egress-log", src)

    def test_routes_has_no_csv_export(self):
        """Egress data flows through the external DB log export — no CSV route."""
        src = open(ROUTES_PY).read()
        self.assertNotIn("egress-log/export.csv", src)

    def test_egress_table_in_log_export_pipeline(self):
        """EGRESS_TABLE must be in log_export.py alongside QUERY and LLM tables."""
        import importlib
        from core import log_export as le
        self.assertTrue(hasattr(le, "EGRESS_TABLE"))
        self.assertEqual(le.EGRESS_TABLE, "KB_DATA_EGRESS_LOG")

    def test_egress_columns_in_log_export_pipeline(self):
        from core import log_export as le
        self.assertTrue(hasattr(le, "EGRESS_COLUMNS"))
        self.assertIn("SAMPLE_MODE", le.EGRESS_COLUMNS)
        self.assertIn("OPERATION",   le.EGRESS_COLUMNS)
        self.assertIn("TABLE_NAME",  le.EGRESS_COLUMNS)

    def test_fetch_egress_rows_after_exists(self):
        from core import log_export as le
        self.assertTrue(callable(getattr(le, "_fetch_egress_rows_after", None)))

    def test_sync_includes_egress_count_in_result(self):
        """sync_external_logs result dict must include egress_count key."""
        src = open(os.path.join(os.path.dirname(__file__), "..", "core", "log_export.py")).read()
        self.assertIn('"egress_count"', src)

    def test_migration_adds_egress_watermark_columns(self):
        """v16 migration adds last_egress_id and last_egress_count."""
        src = open(DB_PY).read()
        self.assertIn('"last_egress_id"', src)
        self.assertIn('"last_egress_count"', src)

    def test_setup_template_points_to_databases_page(self):
        """Setup page must direct admins to Admin → Databases for DB export config."""
        src = open(SETUP_TMPL).read()
        self.assertIn("External Log Export", src)
        self.assertIn("/admin/databases", src)

    def test_setup_template_has_egress_section(self):
        src = open(SETUP_TMPL).read()
        self.assertIn("egress_summary", src)
        self.assertIn("Data egress transparency", src)

    def test_setup_template_shows_sample_mode(self):
        src = open(SETUP_TMPL).read()
        self.assertIn("sample_mode", src)
        self.assertIn("Synthetic", src)
        self.assertIn("Real rows", src)

    def test_setup_template_has_real_rows_warning(self):
        src = open(SETUP_TMPL).read()
        self.assertIn("real_sample_count", src)

    def test_setup_template_shows_per_table_detail(self):
        src = open(SETUP_TMPL).read()
        self.assertIn("kb_build_rows", src)
        self.assertIn("table_name", src)

    def test_setup_page_passes_egress_summary(self):
        src = open(ROUTES_PY).read()
        self.assertIn("egress_summary", src)
        self.assertIn("get_kb_egress_summary", src)

    def test_egress_logged_after_not_before_discover(self):
        """Egress must be logged AFTER discover_and_write completes."""
        src = open(ROUTES_PY).read()
        idx_discover = src.find("count = discover_and_write")
        idx_log      = src.find("log_kb_egress", idx_discover)
        self.assertGreater(idx_log, idx_discover,
            "log_kb_egress must come after discover_and_write")

    def test_egress_logged_after_not_before_build_kb(self):
        """Egress must be logged AFTER build_kb completes."""
        src = open(ROUTES_PY).read()
        idx_build = src.find("count = await build_kb(")
        idx_log   = src.find('operation="kb_build"', idx_build)
        self.assertGreater(idx_log, idx_build,
            "kb_build egress log must come after build_kb call")


if __name__ == "__main__":
    unittest.main()
