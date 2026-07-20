"""Sprint 2 detector #4: stale references after schema drift.

Deliberately narrower than a live-schema diff: the existing Model Health
"Schema Drift" panel (admin/routes.py::_compute_schema_drift) already
compares the live DB schema against the previous discovery snapshot.
Re-deriving that here from a fresh read would duplicate it. This detector
instead cross-references semantic objects (metrics, terms, graph
relationships) against what is ALREADY compiled into the SAME contract
(the schema-derived model + the entity graph) - catching objects
configured outside the KB's current scope, typos, or orphaned graph edges.

Severity is STALE throughout - verified end-to-end that it never blocks
governed_recompile_contract's enforce-mode publish gate, unlike ERROR.
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

from core.semantic_conflicts import detect_stale_references  # noqa: E402
from core.semantic_contract import compile_contract  # noqa: E402

_MODEL = {"tables": [{"fqn": "ERP.FACT_SALES", "qualified_name": "ERP.FACT_SALES", "fields": []}]}
_GRAPH_VALID = {
    "entities": [{"entity_name": "FACT_SALES", "schema_name": "ERP", "table_name": "FACT_SALES"}],
    "relationships": [{"id": 1, "from_entity": "FACT_SALES", "to_entity": "FACT_SALES"}],
}


def _compile(*, metrics=None, terms=None, model=None, graph=None, account_id="acct-stale-unit"):
    with patch("core.semantic_model.load_semantic_model", return_value=model or {}), \
         patch("store.list_metrics", return_value=metrics or []), \
         patch("store.list_metric_date_contexts", return_value=[]), \
         patch("store.get_full_graph", return_value=graph or {}), \
         patch("store.list_terms", return_value=terms or []), \
         patch("core.field_overrides.load_field_overrides", return_value={}):
        return compile_contract(account_id, "C:/tmp/stale-unit-kb")


class StaleMetricBaseTableTests(unittest.TestCase):
    def test_matching_base_table_is_not_flagged(self):
        contract = _compile(
            metrics=[{"id": 1, "name": "Revenue", "base_table": "ERP.FACT_SALES"}], model=_MODEL,
        )
        self.assertEqual(detect_stale_references(contract), [])

    def test_bare_table_name_still_matches(self):
        contract = _compile(
            metrics=[{"id": 1, "name": "Revenue", "base_table": "FACT_SALES"}], model=_MODEL,
        )
        self.assertEqual(detect_stale_references(contract), [])

    def test_unresolvable_base_table_is_stale(self):
        contract = _compile(
            metrics=[{"id": 2, "name": "Ghost", "base_table": "OLD_SCHEMA.DROPPED_TABLE"}], model=_MODEL,
        )
        conflicts = detect_stale_references(contract)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["code"], "stale_metric_base_table")
        self.assertEqual(conflicts[0]["severity"], "STALE")
        self.assertEqual(conflicts[0]["object_id"], "metric:2")

    def test_metric_matched_via_entity_graph_not_just_model(self):
        # A table can be known through the GRAPH even if the schema-derived
        # model doesn't (yet) carry it — either source counts as evidence.
        contract = _compile(
            metrics=[{"id": 1, "name": "Revenue", "base_table": "ERP.FACT_SALES"}],
            model={}, graph=_GRAPH_VALID,
        )
        self.assertEqual(detect_stale_references(contract), [])

    def test_metric_without_base_table_is_not_this_detectors_concern(self):
        contract = _compile(metrics=[{"id": 1, "name": "No table set"}], model=_MODEL)
        self.assertEqual(detect_stale_references(contract), [])


class StaleTermReferenceTests(unittest.TestCase):
    def test_unscoped_term_is_not_flagged(self):
        contract = _compile(terms=[{"id": 1, "term": "generic", "tables_involved": ""}], model=_MODEL)
        self.assertEqual(detect_stale_references(contract), [])

    def test_term_referencing_a_gone_table_is_stale(self):
        contract = _compile(
            terms=[{"id": 2, "term": "ghost term", "tables_involved": "DROPPED.TABLE"}], model=_MODEL,
        )
        conflicts = detect_stale_references(contract)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["code"], "stale_term_table_reference")
        self.assertEqual(conflicts[0]["severity"], "STALE")

    def test_term_valid_if_any_one_of_multiple_tables_still_exists(self):
        contract = _compile(
            terms=[{"id": 3, "term": "partial", "tables_involved": "DROPPED.TABLE, ERP.FACT_SALES"}],
            model=_MODEL,
        )
        self.assertEqual(detect_stale_references(contract), [])


class StaleRelationshipTests(unittest.TestCase):
    def test_relationship_between_existing_entities_is_fine(self):
        contract = _compile(graph=_GRAPH_VALID)
        self.assertEqual(detect_stale_references(contract), [])

    def test_relationship_to_a_deleted_entity_is_stale(self):
        graph = {
            "entities": [{"entity_name": "FACT_SALES", "schema_name": "ERP", "table_name": "FACT_SALES"}],
            "relationships": [{"id": 99, "from_entity": "FACT_SALES", "to_entity": "DIM_CUST"}],
        }
        contract = _compile(graph=graph)
        conflicts = detect_stale_references(contract)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["code"], "stale_relationship_entity")
        self.assertEqual(conflicts[0]["object_id"], "join:99")
        self.assertEqual(conflicts[0]["evidence"]["missing"], ["DIM_CUST"])

    def test_relationship_missing_both_sides_reports_both(self):
        graph = {"entities": [], "relationships": [{"id": 5, "from_entity": "A", "to_entity": "B"}]}
        contract = _compile(graph=graph)
        conflicts = detect_stale_references(contract)
        self.assertEqual(conflicts[0]["evidence"]["missing"], ["A", "B"])


class EmptyContractTests(unittest.TestCase):
    def test_fully_empty_contract_no_crash(self):
        self.assertEqual(detect_stale_references(_compile()), [])


class StaleNeverBlocksEnforceModeTests(unittest.TestCase):
    """The property that matters: STALE severity must never contribute to
    governed_recompile_contract's error_count, so it can never block a
    publish in enforce mode the way an ERROR does."""

    @classmethod
    def setUpClass(cls):
        import store
        import store.database
        import store.db

        cls.tmp = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.tmp.name) / "stale-e2e-tests.db"
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

    def test_stale_reference_does_not_block_enforce_publish(self):
        import store
        from core.semantic_contract import governed_recompile_contract

        account_id = f"stale-e2e-{uuid.uuid4().hex[:6]}"
        kb_dir = Path(self.tmp.name) / account_id / "kb"
        kb_dir.mkdir(parents=True, exist_ok=True)
        store.upsert_client(account_id, "web")
        store.update_client_state(account_id, "READY", {"kb_dir": str(kb_dir)})
        store.set_semantic_compiler_mode(account_id, "enforce")

        with patch("core.semantic_model.load_semantic_model", return_value=_MODEL), \
             patch("store.list_metrics",
                   return_value=[{"id": 2, "name": "Ghost", "base_table": "OLD_SCHEMA.DROPPED_TABLE"}]), \
             patch("store.list_metric_date_contexts", return_value=[]), \
             patch("store.get_full_graph", return_value={}), \
             patch("store.list_terms", return_value=[]), \
             patch("core.field_overrides.load_field_overrides", return_value={}):
            result = governed_recompile_contract(account_id, trigger="test")

        self.assertEqual(result["status"], "published")
        self.assertTrue(result["published_version"])
        codes = {c["code"] for c in result["conflicts"]}
        self.assertIn("stale_metric_base_table", codes)
        summary = store.get_semantic_compiler_summary(account_id)
        self.assertEqual(summary["counts"]["errors"], 0)


if __name__ == "__main__":
    unittest.main()
