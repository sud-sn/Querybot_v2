"""
tests/test_generic_measure_words_are_not_a_metric_match.py

Catalogue case B4 — "show me the inventory value by warehouse" — was refused
with source_fact_mismatch, and the SQL it refused was CORRECT:

    requires EMDW_DMART.PCH_ORD_RCT_FCT as measure fact source(s),
    but the SQL scans EMDW_DMART.ITM_BAL_PRD_FCT, EMDW_DMART.WHS_DMS

The plan was wrong, not the SQL. EMCO registers two metrics and neither is
about inventory, but "Purchase Order Amount" carries the synonym "purchase
order value". The question and that synonym share exactly one token — "value"
— which scored 10, and the only floor anywhere in the matcher is `score > 0`.
That one token made Purchase Order Amount the sole matched metric, its base
table became the authoritative fact, the compiled plan declared it the measure
fact source, and the validator then rejected the model's correct SQL.

A generic quantity word is not evidence about WHICH metric was meant: every
metric in every registry is a value, an amount and a total of something. Real
matches are not this thin — the numbers in the tests below are the actual
scores, and there is two orders of magnitude between a subject-word match and
this one.

These tests execute `resolve_metric_scope` and `_phrase_score` directly. The
scoring functions are pure, so there is nothing to mock and no reason to assert
on source text.
"""

from __future__ import annotations

import unittest

from core.metric_scope import _phrase_score, resolve_metric_scope
from core.source_resolution import GENERIC_MEASURE_WORDS


# The two metrics Emco_test actually has, verbatim from the live registry.
PURCHASE_ORDER_AMOUNT = {
    "name": "Purchase Order Amount",
    "synonyms": (
        "purchase order amount, po amount, purchase order value, "
        "confirmed purchase orders, purchase order total"
    ),
    "formula_type": "expression",
    "sql_template": "SUM(PCH_ORD_LIN_CAD_AMT)",
    "required_columns": "PCH_ORD_LIN_CAD_AMT",
    "base_table": "EMDW_DMART.PCH_ORD_RCT_FCT",
    "category": "PROCUREMENT",
}
REVENUE = {
    "name": "Revenue",
    "synonyms": "revenue, sales revenue, total revenue, net revenue, invoiced revenue",
    "formula_type": "expression",
    "sql_template": "SUM(SOP_CUS_IVC_LIN_AMT)",
    "required_columns": "SOP_CUS_IVC_LIN_AMT",
    "base_table": "EMDW_DMART.CUS_ORD_IVC_FCT",
    "category": "SALES",
}
EMCO_METRICS = [PURCHASE_ORDER_AMOUNT, REVENUE]

COLUMNS = {
    "EMDW_DMART.PCH_ORD_RCT_FCT": {"PCH_ORD_LIN_CAD_AMT": "decimal"},
    "EMDW_DMART.CUS_ORD_IVC_FCT": {"SOP_CUS_IVC_LIN_AMT": "decimal"},
    "EMDW_DMART.ITM_BAL_PRD_FCT": {"BAL_VAL_AMT": "decimal", "WHS_DMS_KEY": "int"},
    "EMDW_DMART.WHS_DMS": {"WHS_DMS_KEY": "int", "WHS_NM": "varchar"},
}


def _matched_names(question: str) -> list[str]:
    result = resolve_metric_scope(EMCO_METRICS, question, COLUMNS)
    return [str(metric.get("name") or "") for metric in result.metrics]


class TestTheDefect(unittest.TestCase):

    def test_the_inventory_question_matches_no_metric(self):
        """The live B4 failure. This tenant has no inventory metric, so the
        honest answer is that nothing matched — which leaves the fact to be
        chosen from schema and graph evidence instead of being pinned wrong."""
        self.assertEqual(_matched_names("show me the inventory value by warehouse"), [])

    def test_the_binding_token_really_was_just_value(self):
        """Names the premise, so this file still means something if the
        synonym list is ever edited."""
        self.assertIn("purchase order value", PURCHASE_ORDER_AMOUNT["synonyms"])
        self.assertEqual(
            _phrase_score(PURCHASE_ORDER_AMOUNT, "show me the inventory value by warehouse"),
            0,
        )

    def test_other_generic_measure_words_behave_the_same(self):
        for question in (
            "show me the stock value by warehouse",
            "what is the total value by warehouse",
            "show me the inventory amount by warehouse",
            "what is the item count by warehouse",
        ):
            with self.subTest(question=question):
                self.assertEqual(_matched_names(question), [])


class TestRealMatchesAreUntouched(unittest.TestCase):
    """The acceptance cases that exist because the OPPOSITE bug was fixed.

    B1 in particular was a source_fact_mismatch in the other direction, so a
    fix here that dampened it would trade one catalogue failure for another.
    """

    def test_b1_purchase_orders_still_matches(self):
        question = "what is the total amount of confirmed purchase orders by profit center"
        self.assertEqual(_matched_names(question), ["Purchase Order Amount"])
        self.assertEqual(_phrase_score(PURCHASE_ORDER_AMOUNT, question), 166)

    def test_b3_revenue_still_matches(self):
        question = "show total revenue by profit centre"
        self.assertEqual(_matched_names(question), ["Revenue"])
        self.assertEqual(_phrase_score(REVENUE, question), 122)

    def test_b2_revenue_by_customer_still_matches(self):
        self.assertEqual(
            _matched_names("what is my revenue by each customer, provide top 5"),
            ["Revenue"],
        )

    def test_the_synonym_itself_still_works_when_actually_used(self):
        """"purchase order value" is a legitimate synonym. Dropping generic-only
        token overlap must not stop it matching when the user says the phrase —
        that path needs the whole phrase, not one token of it."""
        question = "what is the purchase order value for last month"
        self.assertEqual(_matched_names(question), ["Purchase Order Amount"])
        self.assertEqual(_phrase_score(PURCHASE_ORDER_AMOUNT, question), 166)

    def test_a_generic_word_alongside_a_subject_word_still_counts(self):
        """The rule drops overlap that is ONLY generic. "revenue amount" shares
        a subject word too and must be unaffected."""
        self.assertEqual(_matched_names("what is the revenue amount by customer"), ["Revenue"])


class TestTheVocabularyIsShared(unittest.TestCase):

    def test_the_matcher_uses_the_same_list_as_source_resolution(self):
        """Two copies of this list would drift, and the drift would be silent:
        a word in one and not the other reopens exactly this bug."""
        from core import metric_scope

        self.assertIs(metric_scope.GENERIC_MEASURE_WORDS, GENERIC_MEASURE_WORDS)

    def test_the_store_side_prefilter_applies_it_too(self):
        """That scorer builds the candidate list injected into the SQL prompt,
        so a metric surviving there on "value" alone hands the model a formula
        from an unrelated fact table."""
        from store.config_store import _score_metric_for_question

        self.assertEqual(
            _score_metric_for_question(
                PURCHASE_ORDER_AMOUNT, "show me the inventory value by warehouse"
            ),
            0,
        )
        self.assertGreater(
            _score_metric_for_question(
                PURCHASE_ORDER_AMOUNT,
                "what is the total amount of confirmed purchase orders by profit center",
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
