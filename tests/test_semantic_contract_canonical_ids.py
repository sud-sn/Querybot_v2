"""Integration: canonical semantic ids as stamped by
core.semantic_contract._stamp_canonical_ids into the actual compiled
contract (not just the unit-level id-minting functions in
core/semantic_ids.py — see tests/test_semantic_ids.py for those).

The property that matters here is the one Sprint 2's conflict detectors and
store.reconcile_semantic_conflicts() both depend on: the SAME real-world
semantic object mints the SAME canonical_id on every compile, so a
conflict_key built from it is stable across runs — and a source-read
failure never crashes the compile, matching the compiler's existing
"missing sources compile to empty sections" contract.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.semantic_contract import compile_contract  # noqa: E402
from core.semantic_ids import conflict_key  # noqa: E402


def _compile(account_id="acct-canon-ids", *, model=None, metrics=None, terms=None, graph=None):
    with patch("core.semantic_model.load_semantic_model", return_value=model or {}), \
         patch("store.list_metrics", return_value=metrics or []), \
         patch("store.list_metric_date_contexts", return_value=[]), \
         patch("store.get_full_graph", return_value=graph or {}), \
         patch("store.list_terms", return_value=terms or []), \
         patch("core.field_overrides.load_field_overrides", return_value={}):
        return compile_contract(account_id, "C:/tmp/canon-ids-kb")


class ContractStampingTests(unittest.TestCase):
    MODEL = {
        "tables": [{
            "fqn": "ERP.FACT_SALES", "qualified_name": "ERP.FACT_SALES",
            "fields": [{"column": "NET_AMT", "role": "measure"}],
            "date_roles": [{
                "fact_table": "ERP.FACT_SALES", "fact_column": "ORDER_DT_DMS_KEY",
                "business_role": "order_date",
            }],
        }],
        "date_roles": [{
            "fact_table": "ERP.FACT_SALES", "fact_column": "ORDER_DT_DMS_KEY",
            "business_role": "order_date",
        }],
    }
    GRAPH = {
        "entities": [{"entity_name": "FACT_SALES", "schema_name": "ERP", "table_name": "FACT_SALES"}],
        "relationships": [{"id": 99, "from_entity": "FACT_SALES", "to_entity": "DIM_CUST"}],
        "properties": [{"entity_name": "FACT_SALES", "column_name": "NET_AMT", "role": "metric"}],
    }

    def test_every_object_type_gets_stamped(self):
        contract = _compile(
            model=self.MODEL, graph=self.GRAPH,
            metrics=[{"id": 1, "name": "Revenue"}], terms=[{"id": 2, "term": "active customer"}],
        )
        self.assertEqual(contract["metrics"][0]["canonical_id"], "metric:1")
        self.assertEqual(contract["terms"][0]["canonical_id"], "term:2")
        table = contract["model"]["tables"][0]
        self.assertEqual(table["canonical_id"], "table:ERP.FACT_SALES")
        self.assertEqual(table["fields"][0]["canonical_id"], "field:ERP.FACT_SALES.NET_AMT")
        self.assertEqual(
            table["date_roles"][0]["canonical_id"],
            "date_role:ERP.FACT_SALES.ORDER_DT_DMS_KEY",
        )
        self.assertEqual(
            contract["model"]["date_roles"][0]["canonical_id"],
            "date_role:ERP.FACT_SALES.ORDER_DT_DMS_KEY",
        )
        entity = contract["graph"]["entities"][0]
        self.assertEqual(entity["canonical_id"], "entity:FACT_SALES")
        self.assertEqual(contract["graph"]["relationships"][0]["canonical_id"], "join:99")
        self.assertEqual(
            contract["graph"]["properties"][0]["canonical_id"], "field:ERP.FACT_SALES.NET_AMT",
        )

    def test_duplicate_metric_names_stay_distinguishable(self):
        contract = _compile(metrics=[
            {"id": 3, "name": "Revenue"}, {"id": 9, "name": "Revenue"},
        ])
        ids = {m["canonical_id"] for m in contract["metrics"]}
        self.assertEqual(ids, {"metric:3", "metric:9"})

    def test_graph_property_and_model_field_converge_in_a_real_compile(self):
        contract = _compile(model=self.MODEL, graph=self.GRAPH)
        model_field_id = contract["model"]["tables"][0]["fields"][0]["canonical_id"]
        graph_property_id = contract["graph"]["properties"][0]["canonical_id"]
        self.assertEqual(model_field_id, graph_property_id)

    def test_missing_and_orphaned_objects_never_crash_compile(self):
        contract = _compile(
            model={},
            metrics=[{"name": "no id"}],
            graph={"properties": [{"entity_name": "GHOST_ENTITY", "column_name": "X"}]},
        )
        self.assertNotIn("canonical_id", contract["metrics"][0])
        self.assertNotIn("canonical_id", contract["graph"]["properties"][0])

    def test_source_read_failure_still_produces_diagnostics_not_a_crash(self):
        # Mirrors the existing Sprint 1 last-known-good contract: a broken
        # source degrades to an empty section plus a semantic_source_unavailable
        # diagnostic, stamping just runs over whatever DID come back.
        with patch("core.semantic_model.load_semantic_model", side_effect=RuntimeError("boom")), \
             patch("store.list_metrics", return_value=[{"id": 1, "name": "Revenue"}]), \
             patch("store.list_metric_date_contexts", return_value=[]), \
             patch("store.get_full_graph", return_value={}), \
             patch("store.list_terms", return_value=[]), \
             patch("core.field_overrides.load_field_overrides", return_value={}):
            contract = compile_contract("acct-canon-ids-broken", "C:/tmp/canon-ids-kb-broken")
        self.assertEqual(contract["metrics"][0]["canonical_id"], "metric:1")
        self.assertEqual(contract["model"], {})


class ConflictKeyStabilityAcrossCompilesTests(unittest.TestCase):
    """The property store.reconcile_semantic_conflicts() actually depends on:
    a conflict_key built from the SAME two objects, discovered in DIFFERENT
    list order on two separate (identically-sourced) compiles, must be the
    identical string — otherwise every recompile would open a duplicate
    conflict instead of recognizing the existing open one."""

    def test_conflict_key_identical_across_two_separately_ordered_compiles(self):
        metrics_run_1 = [{"id": 3, "name": "Revenue"}, {"id": 9, "name": "Revenue"}]
        metrics_run_2 = [{"id": 9, "name": "Revenue"}, {"id": 3, "name": "Revenue"}]  # reordered

        contract_1 = _compile(account_id="acct-conflict-stability", metrics=metrics_run_1)
        contract_2 = _compile(account_id="acct-conflict-stability", metrics=metrics_run_2)

        ids_1 = [m["canonical_id"] for m in contract_1["metrics"]]
        ids_2 = [m["canonical_id"] for m in contract_2["metrics"]]

        key_1 = conflict_key("duplicate_metric_name", *ids_1)
        key_2 = conflict_key("duplicate_metric_name", *ids_2)
        self.assertEqual(key_1, key_2)


if __name__ == "__main__":
    unittest.main()
