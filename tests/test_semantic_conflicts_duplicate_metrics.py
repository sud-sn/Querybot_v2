"""Sprint 2 detector #3: duplicate metric names/synonyms with different
formulas.

metric_registry has no UNIQUE constraint on (account_id, name) - verified
against store/db.py - so two active metrics can share a name today. The
detector exists because neither runtime path that decides what SQL a
question gets (store.match_metric, the deterministic registry route; and
store.list_metric_formula_context, the LLM-route prompt injection) dedupes
such a collision - see core/semantic_conflicts.py's detector docstring for
the exact mechanics.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.semantic_conflicts import detect_duplicate_metric_names  # noqa: E402
from core.semantic_contract import compile_contract  # noqa: E402


def _compile(metrics, account_id="acct-dupmetric-unit"):
    with patch("core.semantic_model.load_semantic_model", return_value={}), \
         patch("store.list_metrics", return_value=metrics), \
         patch("store.list_metric_date_contexts", return_value=[]), \
         patch("store.get_full_graph", return_value={}), \
         patch("store.list_terms", return_value=[]), \
         patch("core.field_overrides.load_field_overrides", return_value={}):
        return compile_contract(account_id, "C:/tmp/dupmetric-unit-kb")


class DuplicateMetricNameTests(unittest.TestCase):
    def test_exact_name_collision_with_different_sql_is_an_error(self):
        contract = _compile([
            {"id": 1, "name": "Revenue", "synonyms": "", "sql_template": "SELECT SUM(NET_AMT) FROM T1"},
            {"id": 2, "name": "Revenue", "synonyms": "", "sql_template": "SELECT SUM(GROSS_AMT) FROM T2"},
        ])
        conflicts = detect_duplicate_metric_names(contract)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["code"], "duplicate_metric_name")
        self.assertEqual(conflicts[0]["severity"], "ERROR")
        self.assertEqual(set(conflicts[0]["object_id"].split(":")), {"metric", "1"})

    def test_synonym_overlap_across_different_names_is_flagged(self):
        contract = _compile([
            {"id": 1, "name": "Net Revenue", "synonyms": "revenue, sales", "sql_template": "A"},
            {"id": 2, "name": "Gross Sales", "synonyms": "revenue", "sql_template": "B"},
        ])
        conflicts = detect_duplicate_metric_names(contract)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["evidence"]["phrase"], "revenue")
        ids = {m["canonical_id"] for m in conflicts[0]["evidence"]["metrics"]}
        self.assertEqual(ids, {"metric:1", "metric:2"})

    def test_whitespace_only_sql_difference_is_not_a_conflict(self):
        contract = _compile([
            {"id": 1, "name": "Revenue", "synonyms": "", "sql_template": "SELECT SUM(NET_AMT)  FROM T1"},
            {"id": 2, "name": "Revenue", "synonyms": "", "sql_template": "SELECT SUM(NET_AMT) FROM T1"},
        ])
        self.assertEqual(detect_duplicate_metric_names(contract), [])

    def test_case_difference_in_sql_is_still_flagged(self):
        # Deliberately NOT normalized away - identifier casing can be
        # semantically significant on a case-sensitive database.
        contract = _compile([
            {"id": 1, "name": "Revenue", "synonyms": "", "sql_template": "SELECT SUM(net_amt) FROM t1"},
            {"id": 2, "name": "Revenue", "synonyms": "", "sql_template": "SELECT SUM(NET_AMT) FROM T1"},
        ])
        self.assertEqual(len(detect_duplicate_metric_names(contract)), 1)

    def test_single_metric_no_conflict(self):
        contract = _compile([{"id": 1, "name": "Revenue", "synonyms": "", "sql_template": "A"}])
        self.assertEqual(detect_duplicate_metric_names(contract), [])

    def test_three_way_collision_reports_all_participants(self):
        contract = _compile([
            {"id": 1, "name": "Revenue", "synonyms": "", "sql_template": "A"},
            {"id": 2, "name": "Revenue", "synonyms": "", "sql_template": "B"},
            {"id": 3, "name": "Revenue", "synonyms": "", "sql_template": "C"},
        ])
        conflicts = detect_duplicate_metric_names(contract)
        self.assertEqual(len(conflicts), 1)
        ids = {m["canonical_id"] for m in conflicts[0]["evidence"]["metrics"]}
        self.assertEqual(ids, {"metric:1", "metric:2", "metric:3"})

    def test_conflict_key_stable_regardless_of_discovery_order(self):
        metrics_a = [
            {"id": 1, "name": "Revenue", "synonyms": "", "sql_template": "A"},
            {"id": 2, "name": "Revenue", "synonyms": "", "sql_template": "B"},
        ]
        metrics_b = list(reversed(metrics_a))
        c1 = _compile(metrics_a, account_id="acct-dupmetric-order-1")
        c2 = _compile(metrics_b, account_id="acct-dupmetric-order-2")
        key1 = detect_duplicate_metric_names(c1)[0]["conflict_key"]
        key2 = detect_duplicate_metric_names(c2)[0]["conflict_key"]
        self.assertEqual(key1, key2)

    def test_metric_missing_id_is_skipped_not_guessed(self):
        contract = _compile([
            {"name": "Revenue", "synonyms": "", "sql_template": "A"},  # no id -> no canonical_id
            {"id": 2, "name": "Revenue", "synonyms": "", "sql_template": "B"},
        ])
        self.assertEqual(detect_duplicate_metric_names(contract), [])

    def test_empty_metrics_no_crash(self):
        self.assertEqual(detect_duplicate_metric_names(_compile([])), [])


if __name__ == "__main__":
    unittest.main()
