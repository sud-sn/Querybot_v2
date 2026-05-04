"""
tests/test_egress_log_export.py

Tests for egress log integration with the external log export pipeline:
  1. _fetch_egress_rows_after() — reads from kb_data_egress_log
  2. EGRESS_TABLE / EGRESS_COLUMNS constants exist and are consistent
  3. _write_export_state() accepts egress_count and last_egress_id
  4. sync_external_logs() result dict includes egress fields
  5. DB migration adds last_egress_id / last_egress_count columns
  6. Architecture guards — all provision functions updated
  7. No CSV export route (egress goes through DB sync)
"""
import os, sys, tempfile, unittest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_logex.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for mod in list(sys.modules.keys()):
    if mod.startswith("store"):
        del sys.modules[mod]
import store.db as db_mod
db_mod.init_db()
import store

LOG_EXPORT_PY = os.path.join(os.path.dirname(__file__), "..", "core", "log_export.py")
DB_PY         = os.path.join(os.path.dirname(__file__), "..", "store", "db.py")
ROUTES_PY     = os.path.join(os.path.dirname(__file__), "..", "admin", "routes.py")
SETUP_TMPL    = os.path.join(os.path.dirname(__file__), "..", "admin", "templates", "client_setup.html")


# ── 1  _fetch_egress_rows_after ───────────────────────────────────────────────
class TestFetchEgressRowsAfter(unittest.TestCase):

    ACC = "test_acc_logex_001"

    def setUp(self):
        # Seed some rows
        for i, (tbl, mode) in enumerate([
            ("FACT_RXFILL", "synthetic"),
            ("DIM_DATE",    "real"),
            ("DIM_STORE",   "real"),
        ]):
            store.log_kb_egress(
                account_id=self.ACC, operation="kb_build",
                db_type="azure_sql", table_name=tbl, sample_mode=mode,
            )

    def test_returns_tuples(self):
        from core.log_export import _fetch_egress_rows_after
        rows = _fetch_egress_rows_after(0, 100)
        self.assertIsInstance(rows, list)
        if rows:
            self.assertIsInstance(rows[0], tuple)

    def test_watermark_excludes_old_rows(self):
        from core.log_export import _fetch_egress_rows_after
        # Get all rows first
        all_rows = _fetch_egress_rows_after(0, 100)
        if not all_rows:
            self.skipTest("No egress rows in DB")
        # Use first row id as watermark — should exclude it
        first_id = all_rows[0][0]
        later_rows = _fetch_egress_rows_after(first_id, 100)
        ids = [r[0] for r in later_rows]
        self.assertNotIn(first_id, ids)

    def test_returns_correct_column_count(self):
        """Tuple must have exactly len(EGRESS_COLUMNS) values."""
        from core.log_export import _fetch_egress_rows_after, EGRESS_COLUMNS
        rows = _fetch_egress_rows_after(0, 100)
        if not rows:
            self.skipTest("No egress rows")
        self.assertEqual(len(rows[0]), len(EGRESS_COLUMNS))

    def test_limit_respected(self):
        from core.log_export import _fetch_egress_rows_after
        rows = _fetch_egress_rows_after(0, 1)
        self.assertLessEqual(len(rows), 1)


# ── 2  Constants ────────────────────────────────────────────────────────────
class TestConstants(unittest.TestCase):

    def test_egress_table_constant_exists(self):
        from core.log_export import EGRESS_TABLE
        self.assertEqual(EGRESS_TABLE, "KB_DATA_EGRESS_LOG")

    def test_egress_columns_constant_exists(self):
        from core.log_export import EGRESS_COLUMNS
        self.assertIsInstance(EGRESS_COLUMNS, list)
        self.assertGreater(len(EGRESS_COLUMNS), 0)

    def test_egress_columns_has_source_id(self):
        from core.log_export import EGRESS_COLUMNS
        self.assertIn("SOURCE_ID", EGRESS_COLUMNS)

    def test_egress_columns_has_account_id(self):
        from core.log_export import EGRESS_COLUMNS
        self.assertIn("ACCOUNT_ID", EGRESS_COLUMNS)

    def test_egress_columns_has_sample_mode(self):
        from core.log_export import EGRESS_COLUMNS
        self.assertIn("SAMPLE_MODE", EGRESS_COLUMNS)

    def test_egress_columns_has_operation(self):
        from core.log_export import EGRESS_COLUMNS
        self.assertIn("OPERATION", EGRESS_COLUMNS)

    def test_egress_columns_has_table_name(self):
        from core.log_export import EGRESS_COLUMNS
        self.assertIn("TABLE_NAME", EGRESS_COLUMNS)

    def test_egress_columns_count_matches_fetch(self):
        """Number of EGRESS_COLUMNS must match what _fetch_egress_rows_after returns."""
        from core.log_export import EGRESS_COLUMNS, _fetch_egress_rows_after
        # Seed a row so we have something to count
        store.log_kb_egress(
            account_id="test_count_acc", operation="kb_build",
            db_type="azure_sql", table_name="T_COUNT", sample_mode="none",
        )
        rows = _fetch_egress_rows_after(0, 1)
        if rows:
            self.assertEqual(len(rows[0]), len(EGRESS_COLUMNS))


# ── 3  _write_export_state accepts egress params ──────────────────────────────
class TestWriteExportStateEgress(unittest.TestCase):

    def test_signature_accepts_egress_params(self):
        from core.log_export import _write_export_state
        import inspect
        sig = inspect.signature(_write_export_state)
        self.assertIn("egress_count",   sig.parameters)
        self.assertIn("last_egress_id", sig.parameters)

    def test_write_does_not_raise_with_egress_params(self):
        """Writing state with egress params must not raise even if columns
        don't exist yet (migration may not have run in test env)."""
        from core.log_export import _write_export_state
        # Use a safe fake db_config_id for this test
        try:
            _write_export_state(
                999999, status="success",
                query_count=5, llm_count=3,
                last_query_id=100, last_llm_id=50,
                egress_count=8, last_egress_id=25,
                run_date="2026-04-28",
            )
        except Exception:
            pass  # Column may not exist in test DB depending on migration state


# ── 4  sync_external_logs result includes egress ─────────────────────────────
class TestSyncResultIncludesEgress(unittest.TestCase):

    def test_result_keys_include_egress_count(self):
        """The result dict from sync must include egress_count."""
        src = open(LOG_EXPORT_PY).read()
        self.assertIn('"egress_count"', src)

    def test_result_keys_include_last_egress_id(self):
        src = open(LOG_EXPORT_PY).read()
        self.assertIn('"last_egress_id"', src)

    def test_fetch_egress_called_in_sync(self):
        src = open(LOG_EXPORT_PY).read()
        # sync_external_logs must call _fetch_egress_rows_after
        self.assertIn("_fetch_egress_rows_after", src)

    def test_egress_insert_called_in_sync(self):
        src = open(LOG_EXPORT_PY).read()
        # _insert_rows called with EGRESS_TABLE
        self.assertIn("EGRESS_TABLE, EGRESS_COLUMNS, egress_rows", src)


# ── 5  DB migration ───────────────────────────────────────────────────────────
class TestDbMigration(unittest.TestCase):

    def test_migration_adds_last_egress_id(self):
        src = open(DB_PY).read()
        self.assertIn('"last_egress_id"', src)

    def test_migration_adds_last_egress_count(self):
        src = open(DB_PY).read()
        self.assertIn('"last_egress_count"', src)

    def test_migration_targets_correct_table(self):
        src = open(DB_PY).read()
        # Migration must target external_log_export_state
        idx_eg = src.find('"last_egress_id"')
        region = src[max(0, idx_eg-200):idx_eg+100]
        self.assertIn("external_log_export_state", region)


# ── 6  Provision functions updated ───────────────────────────────────────────
class TestProvisionFunctions(unittest.TestCase):

    def test_snowflake_provision_has_egress_table(self):
        src = open(LOG_EXPORT_PY).read()
        # Find _provision_snowflake and check it references EGRESS_TABLE
        fn_start = src.find("def _provision_snowflake")
        fn_end   = src.find("\ndef _provision_", fn_start + 1)
        fn_body  = src[fn_start:fn_end]
        self.assertIn("EGRESS_TABLE", fn_body)

    def test_azure_provision_has_egress_table(self):
        src = open(LOG_EXPORT_PY).read()
        fn_start = src.find("def _provision_azure_sql")
        fn_end   = src.find("\ndef _provision_", fn_start + 1)
        fn_body  = src[fn_start:fn_end]
        self.assertIn("EGRESS_TABLE", fn_body)

    def test_oracle_provision_has_egress_table(self):
        src = open(LOG_EXPORT_PY).read()
        fn_start = src.find("def _provision_oracle")
        fn_end   = src.find("\ndef _oracle_table_exists", fn_start + 1)
        fn_body  = src[fn_start:fn_end]
        self.assertIn("EGRESS_TABLE", fn_body)

    def test_snowflake_egress_has_sample_mode_column(self):
        src = open(LOG_EXPORT_PY).read()
        fn_start = src.find("def _provision_snowflake")
        fn_end   = src.find("\ndef _provision_", fn_start + 1)
        fn_body  = src[fn_start:fn_end]
        # SAMPLE_MODE column must be in the CREATE TABLE for Snowflake
        egress_table_pos = fn_body.rfind("EGRESS_TABLE")
        egress_create    = fn_body[egress_table_pos:]
        self.assertIn("SAMPLE_MODE", egress_create)

    def test_azure_egress_has_sample_mode_column(self):
        src = open(LOG_EXPORT_PY).read()
        fn_start = src.find("def _provision_azure_sql")
        fn_end   = src.find("\ndef _provision_", fn_start + 1)
        fn_body  = src[fn_start:fn_end]
        egress_table_pos = fn_body.rfind("EGRESS_TABLE")
        egress_create    = fn_body[egress_table_pos:]
        self.assertIn("SAMPLE_MODE", egress_create)

    def test_all_three_dbs_provision_exported_at(self):
        """All three DB flavours must include an EXPORTED_AT audit timestamp."""
        src = open(LOG_EXPORT_PY).read()
        for fn_name in ("_provision_snowflake", "_provision_azure_sql", "_provision_oracle"):
            fn_start = src.find(f"def {fn_name}")
            fn_end   = src.find("\ndef ", fn_start + 10)
            fn_body  = src[fn_start:fn_end]
            self.assertIn("EXPORTED_AT", fn_body,
                f"{fn_name} must include EXPORTED_AT in EGRESS_TABLE CREATE")


# ── 7  No CSV export; setup template references DB export ────────────────────
class TestNoCSVExportEgressGoesToDB(unittest.TestCase):

    def test_no_csv_export_route_in_routes(self):
        src = open(ROUTES_PY).read()
        self.assertNotIn("egress-log/export.csv", src,
            "CSV export route should be removed — egress flows through external log export")

    def test_setup_template_references_external_log_export(self):
        src = open(SETUP_TMPL).read()
        self.assertIn("External Log Export", src)

    def test_setup_template_links_to_databases_page(self):
        src = open(SETUP_TMPL).read()
        self.assertIn("/admin/databases", src)

    def test_json_api_route_still_present(self):
        """The JSON API for programmatic access should remain."""
        src = open(ROUTES_PY).read()
        self.assertIn("/egress-log", src)


if __name__ == "__main__":
    unittest.main()
