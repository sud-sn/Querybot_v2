"""Sprint 2 detector #2: business term and field synonym collisions.

Two checks, deliberately different severities because the runtime
consequence differs in kind, not just degree:

  business_term_phrase_collision (ERROR) — verified against the actual
  runtime path: store.match_terms_in_question() returns EVERY term row
  matching a phrase (not just the best one — see its own docstring), and
  store.build_term_injection() emits one "use this EXACT expression" prompt
  line per surviving match. Two terms colliding on an alias means the
  SQL-generation prompt receives two contradictory instructions for what
  looks like one concept.

  field_synonym_collision (WARNING) — business_candidates only feeds KB
  narrative documentation (core/schema_enrichment.py), not a direct SQL
  substitution, so the consequence is a confusing KB section, not a
  contradictory prompt.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.semantic_conflicts import detect_synonym_collisions  # noqa: E402
from core.semantic_contract import compile_contract  # noqa: E402


def _compile(*, terms=None, model=None, account_id="acct-syn-unit"):
    with patch("core.semantic_model.load_semantic_model", return_value=model or {}), \
         patch("store.list_metrics", return_value=[]), \
         patch("store.list_metric_date_contexts", return_value=[]), \
         patch("store.get_full_graph", return_value={}), \
         patch("store.list_terms", return_value=terms or []), \
         patch("core.field_overrides.load_field_overrides", return_value={}):
        return compile_contract(account_id, "C:/tmp/syn-unit-kb")


class BusinessTermCollisionTests(unittest.TestCase):
    def test_colliding_alias_with_different_expression_is_an_error(self):
        contract = _compile(terms=[
            {"id": 1, "term": "active buyer", "aliases": "active customer",
             "canonical_expression": "CUST.IS_ACTIVE=1"},
            {"id": 2, "term": "engaged customer", "aliases": "active customer",
             "canonical_expression": "CUST.LAST_ORDER_DT > DATEADD(day,-90,GETDATE())"},
        ])
        conflicts = detect_synonym_collisions(contract)
        term_conflicts = [c for c in conflicts if c["code"] == "business_term_phrase_collision"]
        self.assertEqual(len(term_conflicts), 1)
        self.assertEqual(term_conflicts[0]["severity"], "ERROR")
        self.assertEqual(set(term_conflicts[0]["object_id"].split(":")[0] for _ in [0]), {"term"})

    def test_canonical_term_colliding_with_anothers_alias(self):
        # The collision isn't only alias-vs-alias — a term's own primary
        # name can collide with a different term's alias.
        contract = _compile(terms=[
            {"id": 1, "term": "revenue", "aliases": "",
             "canonical_expression": "SUM(NET_AMT)"},
            {"id": 2, "term": "top line", "aliases": "revenue, gross sales",
             "canonical_expression": "SUM(GROSS_AMT)"},
        ])
        conflicts = detect_synonym_collisions(contract)
        term_conflicts = [c for c in conflicts if c["code"] == "business_term_phrase_collision"]
        self.assertEqual(len(term_conflicts), 1)
        self.assertEqual(term_conflicts[0]["evidence"]["phrase"], "revenue")

    def test_same_target_expression_is_not_a_conflict(self):
        # Two terms/aliases pointing at the identical SQL fragment is a
        # harmless duplicate, not an ambiguity.
        contract = _compile(terms=[
            {"id": 3, "term": "net revenue", "aliases": "",
             "canonical_expression": "SUM(NET_AMT)"},
            {"id": 4, "term": "revenue", "aliases": "net revenue",
             "canonical_expression": "SUM(NET_AMT)"},
        ])
        self.assertEqual(
            [c for c in detect_synonym_collisions(contract) if c["code"] == "business_term_phrase_collision"],
            [],
        )

    def test_single_term_no_collision(self):
        contract = _compile(terms=[
            {"id": 1, "term": "revenue", "aliases": "", "canonical_expression": "SUM(NET_AMT)"},
        ])
        self.assertEqual(detect_synonym_collisions(contract), [])

    def test_conflict_key_stable_regardless_of_term_discovery_order(self):
        terms_a = [
            {"id": 1, "term": "active buyer", "aliases": "active customer",
             "canonical_expression": "A"},
            {"id": 2, "term": "engaged customer", "aliases": "active customer",
             "canonical_expression": "B"},
        ]
        terms_b = list(reversed(terms_a))
        c1 = _compile(terms=terms_a, account_id="acct-syn-order-1")
        c2 = _compile(terms=terms_b, account_id="acct-syn-order-2")
        key1 = [c for c in detect_synonym_collisions(c1) if c["code"] == "business_term_phrase_collision"][0]["conflict_key"]
        key2 = [c for c in detect_synonym_collisions(c2) if c["code"] == "business_term_phrase_collision"][0]["conflict_key"]
        self.assertEqual(key1, key2)


class FieldSynonymCollisionTests(unittest.TestCase):
    MODEL_COLLIDING = {
        "tables": [
            {"fqn": "ERP.T1", "qualified_name": "ERP.T1",
             "fields": [{"column": "AMT", "business_candidates": ["revenue", "sales"]}]},
            {"fqn": "ERP.T2", "qualified_name": "ERP.T2",
             "fields": [{"column": "TOTAL", "business_candidates": ["revenue"]}]},
        ],
    }

    def test_same_phrase_two_different_columns_is_a_warning(self):
        contract = _compile(model=self.MODEL_COLLIDING)
        conflicts = [c for c in detect_synonym_collisions(contract) if c["code"] == "field_synonym_collision"]
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["severity"], "WARNING")
        self.assertEqual(conflicts[0]["evidence"]["phrase"], "revenue")
        self.assertEqual(
            set(conflicts[0]["evidence"]["field_ids"]),
            {"field:ERP.T1.AMT", "field:ERP.T2.TOTAL"},
        )

    def test_non_colliding_phrase_not_flagged(self):
        contract = _compile(model=self.MODEL_COLLIDING)
        conflicts = [c for c in detect_synonym_collisions(contract) if c["code"] == "field_synonym_collision"]
        phrases = {c["evidence"]["phrase"] for c in conflicts}
        self.assertNotIn("sales", phrases)  # only on T1.AMT, not colliding

    def test_same_column_repeated_synonym_is_not_a_conflict(self):
        model = {"tables": [{
            "fqn": "ERP.T1", "qualified_name": "ERP.T1",
            "fields": [{"column": "AMT", "business_candidates": ["revenue", "revenue"]}],
        }]}
        contract = _compile(model=model)
        self.assertEqual(detect_synonym_collisions(contract), [])

    def test_empty_model_no_crash(self):
        self.assertEqual(detect_synonym_collisions(_compile()), [])


if __name__ == "__main__":
    unittest.main()
