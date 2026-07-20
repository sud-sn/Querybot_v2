"""Sprint 2 detectors #8-#9: cross-schema term scope, metric column existence.

detect_cross_schema_term_collisions is deliberately a different check from
detect_synonym_collisions (tests/test_semantic_conflicts_synonyms.py): that
one compares DIFFERENT term rows against each other for a shared phrase;
this one checks whether a SINGLE term row's own tables_involved declaration
internally spans more than one schema.

detect_metric_column_existence is scoped to check the WHOLE compiled model
(not just the metric's own base_table) specifically to avoid false
positives on required_columns legitimately sourced from a joined table.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.semantic_conflicts import (  # noqa: E402
    detect_cross_schema_term_collisions, detect_metric_column_existence,
)
from core.semantic_contract import compile_contract  # noqa: E402


def _compile(*, terms=None, metrics=None, model=None, account_id="acct-schema-col-unit"):
    with patch("core.semantic_model.load_semantic_model", return_value=model or {}), \
         patch("store.list_metrics", return_value=metrics or []), \
         patch("store.list_metric_date_contexts", return_value=[]), \
         patch("store.get_full_graph", return_value={}), \
         patch("store.list_terms", return_value=terms or []), \
         patch("core.field_overrides.load_field_overrides", return_value={}):
        return compile_contract(account_id, "C:/tmp/schema-col-unit-kb")


class CrossSchemaTermScopeTests(unittest.TestCase):
    def test_term_spanning_two_schemas_is_flagged(self):
        contract = _compile(terms=[{
            "id": 1, "term": "active customer",
            "tables_involved": "PHARMACY.DIM_CUSTOMER, PROFITABILITY.DIM_CUSTOMER",
        }])
        conflicts = detect_cross_schema_term_collisions(contract)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["code"], "cross_schema_term_scope")
        self.assertEqual(conflicts[0]["severity"], "WARNING")
        self.assertEqual(conflicts[0]["evidence"]["schemas"], ["PHARMACY", "PROFITABILITY"])

    def test_term_within_one_schema_not_flagged(self):
        contract = _compile(terms=[{
            "id": 2, "term": "ok term",
            "tables_involved": "PHARMACY.DIM_CUSTOMER, PHARMACY.FACT_SALES",
        }])
        self.assertEqual(detect_cross_schema_term_collisions(contract), [])

    def test_bare_table_names_produce_no_false_positive(self):
        # No schema segment to compare — must not be mistaken for a collision.
        contract = _compile(terms=[{
            "id": 3, "term": "bare", "tables_involved": "DIM_CUSTOMER, FACT_SALES",
        }])
        self.assertEqual(detect_cross_schema_term_collisions(contract), [])

    def test_single_table_not_flagged(self):
        contract = _compile(terms=[{
            "id": 4, "term": "single", "tables_involved": "PHARMACY.DIM_CUSTOMER",
        }])
        self.assertEqual(detect_cross_schema_term_collisions(contract), [])

    def test_unscoped_term_not_flagged(self):
        contract = _compile(terms=[{"id": 5, "term": "unscoped", "tables_involved": ""}])
        self.assertEqual(detect_cross_schema_term_collisions(contract), [])

    def test_empty_terms_no_crash(self):
        self.assertEqual(detect_cross_schema_term_collisions(_compile()), [])


class MetricColumnExistenceTests(unittest.TestCase):
    MODEL = {"tables": [{
        "fqn": "ERP.FACT_SALES", "qualified_name": "ERP.FACT_SALES",
        "fields": [{"column": "NET_AMT"}, {"column": "CUST_ID"}],
    }]}

    def test_existing_required_column_not_flagged(self):
        contract = _compile(
            metrics=[{"id": 1, "name": "Revenue", "base_table": "ERP.FACT_SALES",
                      "required_columns": "NET_AMT"}],
            model=self.MODEL,
        )
        self.assertEqual(detect_metric_column_existence(contract), [])

    def test_missing_column_anywhere_in_model_is_flagged(self):
        contract = _compile(
            metrics=[{"id": 2, "name": "Ghost", "base_table": "ERP.FACT_SALES",
                      "required_columns": "TOTALLY_MADE_UP_COL"}],
            model=self.MODEL,
        )
        conflicts = detect_metric_column_existence(contract)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["code"], "metric_column_not_found")
        self.assertEqual(conflicts[0]["severity"], "WARNING")
        self.assertEqual(conflicts[0]["evidence"]["missing_columns"], ["TOTALLY_MADE_UP_COL"])

    def test_column_sourced_from_a_joined_table_is_not_a_false_positive(self):
        model = {"tables": [
            {"fqn": "ERP.FACT_SALES", "qualified_name": "ERP.FACT_SALES",
             "fields": [{"column": "NET_AMT"}]},
            {"fqn": "ERP.DIM_CUSTOMER", "qualified_name": "ERP.DIM_CUSTOMER",
             "fields": [{"column": "REGION"}]},
        ]}
        contract = _compile(
            metrics=[{"id": 3, "name": "Revenue by region", "base_table": "ERP.FACT_SALES",
                      "required_columns": "NET_AMT, REGION"}],
            model=model,
        )
        self.assertEqual(detect_metric_column_existence(contract), [])

    def test_unresolvable_base_table_is_skipped_not_double_flagged(self):
        # detect_stale_references already owns "the table itself is unknown";
        # this detector must not pile on a second finding for the same root cause.
        contract = _compile(
            metrics=[{"id": 4, "name": "X", "base_table": "GHOST.TABLE",
                      "required_columns": "FOO"}],
            model=self.MODEL,
        )
        self.assertEqual(detect_metric_column_existence(contract), [])

    def test_empty_model_skips_check_entirely(self):
        contract = _compile(
            metrics=[{"id": 5, "name": "X", "base_table": "ERP.FACT_SALES",
                      "required_columns": "ANYTHING"}],
        )
        self.assertEqual(detect_metric_column_existence(contract), [])

    def test_metric_without_required_columns_not_flagged(self):
        contract = _compile(
            metrics=[{"id": 6, "name": "X", "base_table": "ERP.FACT_SALES"}],
            model=self.MODEL,
        )
        self.assertEqual(detect_metric_column_existence(contract), [])

    def test_conflict_key_includes_missing_columns(self):
        c1 = _compile(
            metrics=[{"id": 7, "name": "X", "base_table": "ERP.FACT_SALES",
                      "required_columns": "BOGUS_A"}],
            model=self.MODEL, account_id="acct-schema-col-key-1",
        )
        c2 = _compile(
            metrics=[{"id": 7, "name": "X", "base_table": "ERP.FACT_SALES",
                      "required_columns": "BOGUS_B"}],
            model=self.MODEL, account_id="acct-schema-col-key-2",
        )
        key1 = detect_metric_column_existence(c1)[0]["conflict_key"]
        key2 = detect_metric_column_existence(c2)[0]["conflict_key"]
        self.assertNotEqual(key1, key2)  # different missing columns -> different conflicts


if __name__ == "__main__":
    unittest.main()
