"""Metric-over-term precedence: when a business term and a registered metric
share a name, the metric always wins.

Enforced in three places, all through one shared helper
(store.semantic_store.filter_metric_colliding_terms):
  1. build_term_injection      — colliding term never reaches the SQL prompt
  2. find_ambiguous_term       — colliding term never triggers a registry
                                 clarification chip (the answer couldn't
                                 influence the final SQL anyway)
  3. check_ambiguity_glossary_first Step 2 — colliding terms excluded from
                                 the "which did you mean" multi-metric menu
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from store import semantic_store  # noqa: E402


def _term(name, *, aliases="", requires_clarification=False, options=None, kind="metric"):
    return {
        "id": 1,
        "term": name,
        "kind": kind,
        "aliases": aliases,
        "canonical_expression": "SUM(NET_AMT)",
        "requires_clarification": requires_clarification,
        "clarification_options": options or [],
        "tables_involved": "",
    }


class FilterMetricCollidingTermsTests(unittest.TestCase):
    def test_colliding_term_dropped_others_kept(self):
        terms = [_term("revenue"), _term("gross margin")]
        with patch.object(semantic_store, "_metric_synonym_set", return_value={"revenue"}):
            kept = semantic_store.filter_metric_colliding_terms("acct", terms)
        self.assertEqual([t["term"] for t in kept], ["gross margin"])

    def test_alias_collision_also_drops(self):
        terms = [_term("net sales", aliases="revenue, turnover")]
        with patch.object(semantic_store, "_metric_synonym_set", return_value={"revenue"}):
            kept = semantic_store.filter_metric_colliding_terms("acct", terms)
        self.assertEqual(kept, [])

    def test_no_metrics_means_no_filtering(self):
        terms = [_term("revenue")]
        with patch.object(semantic_store, "_metric_synonym_set", return_value=set()):
            kept = semantic_store.filter_metric_colliding_terms("acct", terms)
        self.assertEqual(kept, terms)


class FindAmbiguousTermMetricAwareTests(unittest.TestCase):
    OPTS = [{"id": "o1", "label": "Gross", "value": "gross", "valid": True},
            {"id": "o2", "label": "Net", "value": "net", "valid": True}]

    def test_colliding_clarification_term_is_skipped(self):
        # Previously: user asked "revenue by region" → term chip "which
        # revenue did you mean?" even though the registered metric formula
        # would override whatever they picked.
        colliding = _term("revenue", requires_clarification=True, options=self.OPTS)
        with (
            patch.object(semantic_store, "match_terms_in_question", return_value=[colliding]),
            patch.object(semantic_store, "_metric_synonym_set", return_value={"revenue"}),
        ):
            self.assertIsNone(semantic_store.find_ambiguous_term("acct", "revenue by region"))

    def test_second_non_colliding_ambiguous_term_still_triggers(self):
        # Filtering the whole match list (not just checking the first hit)
        # lets a later, legitimately ambiguous term still ask its question.
        colliding = _term("revenue", requires_clarification=True, options=self.OPTS)
        legit = _term("active customer", requires_clarification=True, options=self.OPTS)
        with (
            patch.object(semantic_store, "match_terms_in_question",
                         return_value=[colliding, legit]),
            patch.object(semantic_store, "_metric_synonym_set", return_value={"revenue"}),
        ):
            found = semantic_store.find_ambiguous_term("acct", "revenue for active customers")
        self.assertIsNotNone(found)
        self.assertEqual(found["term"], "active customer")

    def test_non_colliding_clarification_still_works(self):
        legit = _term("active customer", requires_clarification=True, options=self.OPTS)
        with (
            patch.object(semantic_store, "match_terms_in_question", return_value=[legit]),
            patch.object(semantic_store, "_metric_synonym_set", return_value={"revenue"}),
        ):
            found = semantic_store.find_ambiguous_term("acct", "count active customers")
        self.assertEqual(found["term"], "active customer")


class TermInjectionStillSuppressesTests(unittest.TestCase):
    def test_colliding_term_absent_from_prompt_injection(self):
        # The refactor onto the shared helper must preserve the original
        # build_term_injection guarantee.
        terms = [_term("revenue"), _term("gross margin")]
        with patch.object(semantic_store, "_metric_synonym_set", return_value={"revenue"}):
            with patch.object(semantic_store, "match_terms_in_question", return_value=terms):
                injection = semantic_store.build_term_injection("acct", "revenue and gross margin")
        self.assertNotIn("revenue (", injection)
        self.assertIn("gross margin", injection)


class ClarificationWiringTests(unittest.TestCase):
    def test_glossary_multi_path_filters_colliding_terms(self):
        src = (ROOT / "core/clarification.py").read_text(encoding="utf-8")
        fn = src[src.index("async def check_ambiguity_glossary_first"):]
        step2 = fn[:fn.index("# Step 3")]
        self.assertIn("filter_metric_colliding_terms", step2)


if __name__ == "__main__":
    unittest.main()
