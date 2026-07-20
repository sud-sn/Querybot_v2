"""Sprint 1 governed semantic compiler foundation tests."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class SemanticCompilerSprint1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import store
        import store.database
        import store.db

        cls.tmp = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.tmp.name) / "compiler-tests.db"
        cls.old_database_path = store.database.DB_PATH
        cls.old_db_path = store.db.DB_PATH
        store.database.DB_PATH = cls.db_path
        store.db.DB_PATH = cls.db_path
        store.init_db()

    @classmethod
    def tearDownClass(cls):
        import store.database
        import store.db

        store.database.DB_PATH = cls.old_database_path
        store.db.DB_PATH = cls.old_db_path
        cls.tmp.cleanup()

    def _account(self, suffix: str) -> tuple[str, Path]:
        import store

        account_id = f"compiler-{suffix}"
        kb_dir = Path(self.tmp.name) / account_id / "kb"
        kb_dir.mkdir(parents=True, exist_ok=True)
        store.upsert_client(account_id, "web")
        store.update_client_state(account_id, "READY", {"kb_dir": str(kb_dir)})
        return account_id, kb_dir

    def _compile(self, account_id: str, *, metrics=None):
        from core.semantic_contract import governed_recompile_contract

        model = {"tables": [], "relationships": [], "date_roles": []}
        with patch("core.semantic_model.load_semantic_model", return_value=model), \
             patch("store.list_metrics", return_value=metrics or []), \
             patch("store.list_metric_date_contexts", return_value=[]), \
             patch("store.get_full_graph", return_value={}), \
             patch("store.list_terms", return_value=[]), \
             patch("core.field_overrides.load_field_overrides", return_value={}):
            return governed_recompile_contract(account_id, trigger="test")

    def test_clean_shadow_compile_publishes_and_versions(self):
        import store
        from core.semantic_contract import load_contract

        account_id, kb_dir = self._account("clean")
        result = self._compile(account_id, metrics=[{"name": "Revenue"}])

        self.assertEqual(result["status"], "published")
        self.assertTrue(result["published_version"])
        self.assertEqual(
            load_contract(str(kb_dir))["meta"]["contract_version"],
            result["published_version"],
        )
        summary = store.get_semantic_compiler_summary(account_id)
        self.assertEqual(summary["state"]["mode"], "shadow")
        self.assertEqual(summary["state"]["active_version"], result["published_version"])
        self.assertEqual(summary["latest_run"]["status"], "published")

    def test_source_failure_keeps_last_active_contract(self):
        import store
        from core.semantic_contract import governed_recompile_contract, load_contract

        account_id, kb_dir = self._account("last-good")
        first = self._compile(account_id, metrics=[{"name": "Revenue"}])
        active_before = first["published_version"]

        with patch("core.semantic_model.load_semantic_model", return_value={}), \
             patch("store.list_metrics", side_effect=RuntimeError("registry unavailable")), \
             patch("store.list_metric_date_contexts", return_value=[]), \
             patch("store.get_full_graph", return_value={}), \
             patch("store.list_terms", return_value=[]), \
             patch("core.field_overrides.load_field_overrides", return_value={}):
            failed = governed_recompile_contract(account_id, trigger="broken source")

        self.assertEqual(failed["status"], "invalid")
        self.assertEqual(failed["active_version"], active_before)
        self.assertEqual(
            load_contract(str(kb_dir))["meta"]["contract_version"], active_before,
        )
        summary = store.get_semantic_compiler_summary(account_id)
        self.assertEqual(summary["state"]["active_version"], active_before)
        self.assertGreaterEqual(summary["counts"]["errors"], 1)

        recovered = self._compile(account_id, metrics=[{"name": "Revenue"}])
        self.assertEqual(recovered["status"], "published")
        self.assertEqual(
            store.get_semantic_compiler_summary(account_id)["counts"]["errors"], 0,
        )

    def test_modes_and_records_are_tenant_isolated(self):
        import store

        account_a, _ = self._account("tenant-a")
        account_b, _ = self._account("tenant-b")
        store.set_semantic_compiler_mode(account_a, "enforce")
        self._compile(account_b)

        self.assertEqual(store.get_semantic_compiler_state(account_a)["mode"], "enforce")
        self.assertEqual(store.get_semantic_compiler_state(account_b)["mode"], "shadow")
        self.assertFalse(store.list_semantic_conflicts(account_a))


if __name__ == "__main__":
    unittest.main()
