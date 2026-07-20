"""Sprint 2 detector #1: ambiguous / missing date-role governance.

Promotes two structural gaps that core/contextual_dates.py already has to
work around per-question at compile time, where they can be surfaced to an
admin once instead of silently degrading every matching question:

  1. multiple_default_date_roles (ERROR) — a metric has 2+ date-context
     bindings all marked default.
  2. missing_date_role_binding (WARNING) — a metric's base table has 2+
     candidate business dates but zero governed bindings, so
     _explicit_role_matches (gated on status == "approved") has nothing to
     steer SQL generation with.

Layer 1: unit tests directly on detect_ambiguous_date_roles(contract).
Layer 2: end-to-end through governed_recompile_contract, proving enforce
mode actually blocks publish on the new ERROR and shadow mode does not.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.semantic_conflicts import detect_ambiguous_date_roles, run_all_detectors  # noqa: E402
from core.semantic_contract import compile_contract  # noqa: E402

_MODEL_TWO_DATES = {
    "tables": [{
        "fqn": "ERP.FACT_SALES", "qualified_name": "ERP.FACT_SALES",
        "fields": [],
        "date_roles": [
            {"fact_table": "ERP.FACT_SALES", "fact_column": "ORDER_DT_DMS_KEY", "name": "Order Date"},
            {"fact_table": "ERP.FACT_SALES", "fact_column": "SHIP_DT_DMS_KEY", "name": "Ship Date"},
        ],
    }],
    "date_roles": [],
}
_MODEL_ONE_DATE = {
    "tables": [{
        "fqn": "ERP.FACT_SALES", "qualified_name": "ERP.FACT_SALES",
        "fields": [],
        "date_roles": [
            {"fact_table": "ERP.FACT_SALES", "fact_column": "ORDER_DT_DMS_KEY", "name": "Order Date"},
        ],
    }],
    "date_roles": [],
}


def _compile(*, model=None, metrics=None, date_contexts=None, account_id="acct-dr-unit"):
    with patch("core.semantic_model.load_semantic_model", return_value=model or {}), \
         patch("store.list_metrics", return_value=metrics or []), \
         patch("store.list_metric_date_contexts", return_value=date_contexts or []), \
         patch("store.get_full_graph", return_value={}), \
         patch("store.list_terms", return_value=[]), \
         patch("core.field_overrides.load_field_overrides", return_value={}):
        return compile_contract(account_id, "C:/tmp/dr-unit-kb")


class MultipleDefaultsDetectorTests(unittest.TestCase):
    def test_two_defaults_is_an_error(self):
        contract = _compile(
            model=_MODEL_TWO_DATES,
            metrics=[{"id": 1, "name": "Revenue", "base_table": "ERP.FACT_SALES"}],
            date_contexts=[
                {"metric_id": 1, "context_name": "A", "is_default": 1, "metric_name": "Revenue"},
                {"metric_id": 1, "context_name": "B", "is_default": 1, "metric_name": "Revenue"},
            ],
        )
        conflicts = detect_ambiguous_date_roles(contract)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["code"], "multiple_default_date_roles")
        self.assertEqual(conflicts[0]["severity"], "ERROR")
        self.assertEqual(conflicts[0]["object_id"], "metric:1")

    def test_one_default_among_several_bindings_is_fine(self):
        contract = _compile(
            model=_MODEL_TWO_DATES,
            metrics=[{"id": 1, "name": "Revenue", "base_table": "ERP.FACT_SALES"}],
            date_contexts=[
                {"metric_id": 1, "context_name": "A", "is_default": 1, "metric_name": "Revenue"},
                {"metric_id": 1, "context_name": "B", "is_default": 0, "metric_name": "Revenue"},
            ],
        )
        self.assertEqual(detect_ambiguous_date_roles(contract), [])

    def test_conflict_key_order_independent_across_compiles(self):
        c1 = _compile(
            model=_MODEL_TWO_DATES,
            metrics=[{"id": 5, "name": "X", "base_table": "ERP.FACT_SALES"}],
            date_contexts=[
                {"metric_id": 5, "context_name": "A", "is_default": 1, "metric_name": "X"},
                {"metric_id": 5, "context_name": "B", "is_default": 1, "metric_name": "X"},
            ],
        )
        c2 = _compile(
            model=_MODEL_TWO_DATES,
            metrics=[{"id": 5, "name": "X", "base_table": "ERP.FACT_SALES"}],
            date_contexts=[
                {"metric_id": 5, "context_name": "B", "is_default": 1, "metric_name": "X"},
                {"metric_id": 5, "context_name": "A", "is_default": 1, "metric_name": "X"},
            ],
        )
        key1 = detect_ambiguous_date_roles(c1)[0]["conflict_key"]
        key2 = detect_ambiguous_date_roles(c2)[0]["conflict_key"]
        self.assertEqual(key1, key2)


class MissingBindingDetectorTests(unittest.TestCase):
    def test_unbound_metric_on_ambiguous_table_is_a_warning(self):
        contract = _compile(
            model=_MODEL_TWO_DATES,
            metrics=[{"id": 2, "name": "Shipped Qty", "base_table": "ERP.FACT_SALES"}],
        )
        conflicts = detect_ambiguous_date_roles(contract)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["code"], "missing_date_role_binding")
        self.assertEqual(conflicts[0]["severity"], "WARNING")

    def test_bare_table_name_matches_qualified_model_table(self):
        # base_table is free text (see core/metric_scope.py) — admins commonly
        # set a bare name, not the schema-qualified fqn.
        contract = _compile(
            model=_MODEL_TWO_DATES,
            metrics=[{"id": 2, "name": "Shipped Qty", "base_table": "FACT_SALES"}],
        )
        self.assertEqual(len(detect_ambiguous_date_roles(contract)), 1)

    def test_single_candidate_date_is_not_ambiguous(self):
        contract = _compile(
            model=_MODEL_ONE_DATE,
            metrics=[{"id": 2, "name": "Shipped Qty", "base_table": "ERP.FACT_SALES"}],
        )
        self.assertEqual(detect_ambiguous_date_roles(contract), [])

    def test_metric_with_a_binding_is_not_flagged_even_if_unbound_defaults_check_would_pass(self):
        contract = _compile(
            model=_MODEL_TWO_DATES,
            metrics=[{"id": 2, "name": "Shipped Qty", "base_table": "ERP.FACT_SALES"}],
            date_contexts=[
                {"metric_id": 2, "context_name": "Only", "is_default": 1, "metric_name": "Shipped Qty"},
            ],
        )
        self.assertEqual(detect_ambiguous_date_roles(contract), [])

    def test_metric_with_no_base_table_is_skipped_not_guessed(self):
        contract = _compile(model=_MODEL_TWO_DATES, metrics=[{"id": 2, "name": "No table"}])
        self.assertEqual(detect_ambiguous_date_roles(contract), [])

    def test_base_table_that_matches_nothing_is_skipped(self):
        contract = _compile(
            model=_MODEL_TWO_DATES,
            metrics=[{"id": 2, "name": "X", "base_table": "SOME_OTHER_SCHEMA.UNRELATED_TABLE"}],
        )
        self.assertEqual(detect_ambiguous_date_roles(contract), [])


class DetectorIsolationTests(unittest.TestCase):
    def test_run_all_detectors_survives_a_broken_detector(self):
        def _boom(contract):
            raise RuntimeError("simulated detector bug")

        with patch("core.semantic_conflicts.DETECTORS", (_boom, detect_ambiguous_date_roles)):
            contract = _compile(
                model=_MODEL_TWO_DATES,
                metrics=[{"id": 2, "name": "Shipped Qty", "base_table": "ERP.FACT_SALES"}],
            )
            # Must not raise, and the surviving detector's finding must still land.
            conflicts = run_all_detectors(contract)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["code"], "missing_date_role_binding")


class GovernedCompileEndToEndTests(unittest.TestCase):
    """Proves this detector actually reaches the existing Sprint 1 publish
    gate, not just that the pure function returns the right dict."""

    @classmethod
    def setUpClass(cls):
        import store
        import store.database
        import store.db

        cls.tmp = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.tmp.name) / "date-role-detector-tests.db"
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

    def _account(self, suffix: str):
        import store

        account_id = f"dr-e2e-{suffix}-{uuid.uuid4().hex[:6]}"
        kb_dir = Path(self.tmp.name) / account_id / "kb"
        kb_dir.mkdir(parents=True, exist_ok=True)
        store.upsert_client(account_id, "web")
        store.update_client_state(account_id, "READY", {"kb_dir": str(kb_dir)})
        return account_id, kb_dir

    def _governed_compile(self, account_id, *, metrics, date_contexts=None):
        from core.semantic_contract import governed_recompile_contract

        with patch("core.semantic_model.load_semantic_model", return_value=_MODEL_TWO_DATES), \
             patch("store.list_metrics", return_value=metrics), \
             patch("store.list_metric_date_contexts", return_value=date_contexts or []), \
             patch("store.get_full_graph", return_value={}), \
             patch("store.list_terms", return_value=[]), \
             patch("core.field_overrides.load_field_overrides", return_value={}):
            return governed_recompile_contract(account_id, trigger="test")

    def test_enforce_mode_blocks_publish_on_the_new_error(self):
        import store

        account_id, _ = self._account("enforce")
        store.set_semantic_compiler_mode(account_id, "enforce")
        result = self._governed_compile(
            account_id,
            metrics=[{"id": 1, "name": "Revenue", "base_table": "ERP.FACT_SALES"}],
            date_contexts=[
                {"metric_id": 1, "context_name": "A", "is_default": 1, "metric_name": "Revenue"},
                {"metric_id": 1, "context_name": "B", "is_default": 1, "metric_name": "Revenue"},
            ],
        )
        self.assertEqual(result["status"], "invalid")
        self.assertFalse(result["published_version"])
        codes = {c["code"] for c in result["conflicts"]}
        self.assertIn("multiple_default_date_roles", codes)

    def test_shadow_mode_records_but_does_not_block(self):
        import store

        account_id, kb_dir = self._account("shadow")
        # shadow is the default mode — explicit for clarity/documentation.
        store.set_semantic_compiler_mode(account_id, "shadow")
        result = self._governed_compile(
            account_id,
            metrics=[{"id": 1, "name": "Revenue", "base_table": "ERP.FACT_SALES"}],
            date_contexts=[
                {"metric_id": 1, "context_name": "A", "is_default": 1, "metric_name": "Revenue"},
                {"metric_id": 1, "context_name": "B", "is_default": 1, "metric_name": "Revenue"},
            ],
        )
        self.assertEqual(result["status"], "published")
        self.assertTrue(result["published_version"])
        summary = store.get_semantic_compiler_summary(account_id)
        self.assertGreaterEqual(summary["counts"]["errors"], 1)

    def test_clean_metric_compiles_with_no_new_conflicts(self):
        account_id, _ = self._account("clean")
        result = self._governed_compile(
            account_id,
            metrics=[{"id": 1, "name": "Revenue", "base_table": "ERP.FACT_SALES"}],
            date_contexts=[
                {"metric_id": 1, "context_name": "Only", "is_default": 1, "metric_name": "Revenue"},
            ],
        )
        self.assertEqual(result["status"], "published")
        self.assertEqual(result["conflicts"], [])


if __name__ == "__main__":
    unittest.main()
