"""Sprint 2 detectors #6-#7: competing join paths and cardinality/fan-out risk.

Both reuse core/graph_resolver.py::_edge_weight directly rather than
reimplementing a parallel risk heuristic, so a detector's assessment can
never disagree with what find_join_path() would actually compute for the
same edge when resolving a real question.
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
    detect_cardinality_fanout_risk, detect_competing_join_paths,
)
from core.semantic_contract import compile_contract  # noqa: E402


def _compile(*, graph=None, account_id="acct-graph-unit"):
    with patch("core.semantic_model.load_semantic_model", return_value={}), \
         patch("store.list_metrics", return_value=[]), \
         patch("store.list_metric_date_contexts", return_value=[]), \
         patch("store.get_full_graph", return_value=graph or {}), \
         patch("store.list_terms", return_value=[]), \
         patch("core.field_overrides.load_field_overrides", return_value={}):
        return compile_contract(account_id, "C:/tmp/graph-unit-kb")


class CompetingJoinPathTests(unittest.TestCase):
    def test_different_join_keys_same_pair_is_flagged(self):
        graph = {"relationships": [
            {"id": 1, "from_entity": "ORDERS", "to_entity": "CUSTOMER",
             "from_column": "CUST_ID", "to_column": "CUST_ID",
             "status": "confirmed", "generated_by": "manual"},
            {"id": 2, "from_entity": "ORDERS", "to_entity": "CUSTOMER",
             "from_column": "REFERRED_BY_ID", "to_column": "CUST_ID",
             "status": "confirmed", "generated_by": "manual"},
        ]}
        contract = _compile(graph=graph)
        conflicts = detect_competing_join_paths(contract)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["code"], "competing_join_paths")
        self.assertEqual(conflicts[0]["severity"], "WARNING")
        self.assertEqual(set(conflicts[0]["evidence"]["entities"]), {"ORDERS", "CUSTOMER"})
        self.assertEqual(len(conflicts[0]["evidence"]["join_signatures"]), 2)

    def test_direction_reversed_pair_still_groups_together(self):
        # core/graph_resolver.py's adjacency is undirected — a competing
        # edge stored in the opposite direction must still be caught.
        graph = {"relationships": [
            {"id": 1, "from_entity": "ORDERS", "to_entity": "CUSTOMER",
             "from_column": "CUST_ID", "to_column": "CUST_ID", "status": "confirmed"},
            {"id": 2, "from_entity": "CUSTOMER", "to_entity": "ORDERS",
             "from_column": "REFERRAL_ID", "to_column": "REF_ORDER_ID", "status": "confirmed"},
        ]}
        contract = _compile(graph=graph)
        self.assertEqual(len(detect_competing_join_paths(contract)), 1)

    def test_identical_join_key_repeated_is_not_a_conflict(self):
        graph = {"relationships": [
            {"id": 1, "from_entity": "ORDERS", "to_entity": "CUSTOMER",
             "from_column": "CUST_ID", "to_column": "CUST_ID", "status": "confirmed"},
            {"id": 2, "from_entity": "ORDERS", "to_entity": "CUSTOMER",
             "from_column": "CUST_ID", "to_column": "CUST_ID", "status": "confirmed"},
        ]}
        contract = _compile(graph=graph)
        self.assertEqual(detect_competing_join_paths(contract), [])

    def test_single_edge_no_conflict(self):
        graph = {"relationships": [
            {"id": 1, "from_entity": "ORDERS", "to_entity": "CUSTOMER",
             "from_column": "CUST_ID", "to_column": "CUST_ID", "status": "confirmed"},
        ]}
        contract = _compile(graph=graph)
        self.assertEqual(detect_competing_join_paths(contract), [])

    def test_inactive_edge_excluded(self):
        graph = {"relationships": [
            {"id": 1, "from_entity": "ORDERS", "to_entity": "CUSTOMER",
             "from_column": "CUST_ID", "to_column": "CUST_ID", "status": "confirmed", "is_active": 1},
            {"id": 2, "from_entity": "ORDERS", "to_entity": "CUSTOMER",
             "from_column": "OTHER_ID", "to_column": "CUST_ID", "status": "confirmed", "is_active": 0},
        ]}
        contract = _compile(graph=graph)
        self.assertEqual(detect_competing_join_paths(contract), [])

    def test_manual_vs_llm_edge_weight_gap_is_nonzero(self):
        # A governed manual edge should clearly out-rank an LLM-suggested
        # one — the evidence should show a real gap, not a coin-flip tie.
        graph = {"relationships": [
            {"id": 1, "from_entity": "A", "to_entity": "B",
             "from_column": "X", "to_column": "Y",
             "status": "confirmed", "generated_by": "manual"},
            {"id": 2, "from_entity": "A", "to_entity": "B",
             "from_column": "Z", "to_column": "Y",
             "status": "suggested", "generated_by": "llm"},
        ]}
        contract = _compile(graph=graph)
        conflicts = detect_competing_join_paths(contract)
        self.assertEqual(len(conflicts), 1)
        self.assertGreater(conflicts[0]["evidence"]["weight_gap"], 0)

    def test_empty_graph_no_crash(self):
        self.assertEqual(detect_competing_join_paths(_compile()), [])


class CardinalityFanoutRiskTests(unittest.TestCase):
    def test_high_fanout_confirmed_edge_is_flagged(self):
        graph = {"relationships": [
            {"id": 3, "from_entity": "A", "to_entity": "B", "from_column": "X", "to_column": "Y",
             "status": "confirmed", "generated_by": "llm", "fanout_ratio": 3.5, "confidence_score": 40},
        ]}
        contract = _compile(graph=graph)
        conflicts = detect_cardinality_fanout_risk(contract)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["code"], "high_risk_join_edge")
        self.assertIn("fanout_ratio=3.5", conflicts[0]["evidence"]["reasons"])

    def test_clean_manual_edge_not_flagged(self):
        graph = {"relationships": [
            {"id": 4, "from_entity": "A", "to_entity": "B", "from_column": "X", "to_column": "Y",
             "status": "confirmed", "generated_by": "manual", "confidence_score": 100},
        ]}
        contract = _compile(graph=graph)
        self.assertEqual(detect_cardinality_fanout_risk(contract), [])

    def test_suggested_edge_is_not_this_detectors_concern(self):
        # Unconfirmed edges already carry the review-queue's own signal
        # (Phase 3's review panel) — re-flagging here would be redundant noise.
        graph = {"relationships": [
            {"id": 5, "from_entity": "A", "to_entity": "B", "from_column": "X", "to_column": "Y",
             "status": "suggested", "generated_by": "llm", "fanout_ratio": 5.0, "confidence_score": 20},
        ]}
        contract = _compile(graph=graph)
        self.assertEqual(detect_cardinality_fanout_risk(contract), [])

    def test_broken_edge_not_double_flagged_as_high_risk(self):
        # infinite weight from validation_status=broken is a DIFFERENT
        # problem (an unusable edge) than "usable but risky" — this
        # detector's threshold check explicitly excludes it.
        graph = {"relationships": [
            {"id": 6, "from_entity": "A", "to_entity": "B", "from_column": "X", "to_column": "Y",
             "status": "confirmed", "validation_status": "broken"},
        ]}
        contract = _compile(graph=graph)
        self.assertEqual(detect_cardinality_fanout_risk(contract), [])

    def test_many_to_many_relationship_flagged(self):
        graph = {"relationships": [
            {"id": 7, "from_entity": "A", "to_entity": "B", "from_column": "X", "to_column": "Y",
             "status": "confirmed", "generated_by": "llm", "relationship_type": "many_to_many",
             "confidence_score": 50},
        ]}
        contract = _compile(graph=graph)
        conflicts = detect_cardinality_fanout_risk(contract)
        self.assertEqual(len(conflicts), 1)
        self.assertIn("many_to_many", conflicts[0]["evidence"]["reasons"])

    def test_empty_graph_no_crash(self):
        self.assertEqual(detect_cardinality_fanout_risk(_compile()), [])


if __name__ == "__main__":
    unittest.main()
