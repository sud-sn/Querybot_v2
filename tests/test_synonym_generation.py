"""
tests/test_synonym_generation.py

S1–S3 — the terms a column answers to.

The generator produced spelling variants where it needed vocabulary. Measured
over eighteen realistic columns, only 23% of the synonyms it emitted
introduced a word the column name did not already contain, and all of those
came from three columns that happened to hit a curated dictionary. BAL_VAL_AMT
got five: `balance val amount`, `total balance val amount`, `bal val amt`,
`balance val amt`, `bal val amount` — every one a permutation of
{bal|balance} × {val} × {amt|amount}, filling the whole six-term budget.

Run against the real matcher, those synonyms matched only if the reader typed
the column name back at it. "inventory value" scored 0. So did "stock value".
So did "balance value" — the exact words in its own expansion — because the
matcher demands a COMPLETE concept match and only the longest form was
generated.

Three changes, each with its own class below:

  S1  the lexicon was missing VAL, STAT, REV, NET, MARG, DISC, TAX, PRC —
      eight of the most common words in a sales warehouse expanded to
      themselves
  S2  vocabulary and abbreviation variants are separated, so a permutation
      cannot take the first slot, which is the measure's display name
  S3  the KB harvest's usefulness gate compares token sets, not strings, so
      re-spellings stop reading as new terms

Every assertion executes the real function.
"""

from __future__ import annotations

import unittest

from core.graph_autopopulate import _has_useful_synonym
from core.schema_enrichment import (
    ABBREVIATIONS,
    MAX_BUSINESS_CANDIDATES,
    enrich_columns,
    head_term,
)
from core.semantic_model import _runtime_match_score, _runtime_match_terms


def terms_for(column: str) -> list[str]:
    return enrich_columns([column])[0].business_candidates


def matches(question: str, column: str) -> int:
    return _runtime_match_score(_runtime_match_terms(question), terms_for(column))


class TestTheLexiconKnowsTheWordsAWarehouseUses(unittest.TestCase):

    EXPECTED = {
        "BAL_VAL_AMT": "balance value amount",
        "STAT_CD": "status code",
        "NET_REV_AMT": "net revenue amount",
        "GRS_MARG_PCT": "gross margin percent",
        "DISC_AMT": "discount amount",
        "UNIT_PRC": "unit price",
        "TAX_AMT": "tax amount",
        "SUPP_NUM": "supplier number",
        "BO_QTY": "back order quantity",
    }

    def test_each_expands_to_english(self):
        enriched = {e.column: e for e in enrich_columns(list(self.EXPECTED))}
        for column, expected in self.EXPECTED.items():
            with self.subTest(column=column):
                self.assertEqual(enriched[column].expanded_name, expected)

    def test_the_words_that_were_missing_are_there(self):
        # Named individually because each one was a live failure: BAL_VAL_AMT
        # expanded to "balance val amount" and STAT_CD to "stat code".
        for token, expansion in (("VAL", "value"), ("STAT", "status"),
                                 ("REV", "revenue"), ("MARG", "margin"),
                                 ("DISC", "discount"), ("TAX", "tax"),
                                 ("PRC", "price"), ("NET", "net")):
            with self.subTest(token=token):
                self.assertEqual(ABBREVIATIONS.get(token), expansion)

    def test_they_are_generic_rather_than_one_vendors(self):
        # These belong in the builtin because every warehouse writes them. A
        # code that means something only in one ERP belongs in a pack, and
        # putting it here would hand it to every tenant.
        for erp_only in ("ORNO", "CUNO", "WHLO", "FACI", "OOHEAD"):
            self.assertNotIn(erp_only, ABBREVIATIONS, erp_only)


class TestVocabularyOutranksSpelling(unittest.TestCase):

    def test_the_display_name_is_english_not_an_abbreviation(self):
        # business_candidates[0] becomes the measure's name in
        # core.semantic_model._measure_candidates. It used to be whatever
        # survived a flat list and a cap: GRS_MARG_PCT was named
        # "grs marg pct" and UNIT_PRC was named "unit prc".
        for column, expected in (("GRS_MARG_PCT", "gross margin percent"),
                                 ("UNIT_PRC", "unit price"),
                                 ("STAT_CD", "status code"),
                                 ("SUPP_NUM", "supplier number"),
                                 ("BAL_VAL_AMT", "balance value amount")):
            with self.subTest(column=column):
                self.assertEqual(terms_for(column)[0], expected)

    def test_a_dictionary_hit_still_leads(self):
        # The ERP dictionary is better evidence than an expansion, and must
        # not be displaced by it.
        self.assertEqual(terms_for("ORQT")[0], "ordered quantity")

    def test_a_dictionary_hit_brings_its_synonyms_too(self):
        # The label alone is one phrasing. The dictionary's synonyms are the
        # other ways readers ask for the same measure, and they are the whole
        # reason a curated dictionary beats an expansion — dropping them
        # leaves the leading term looking right and the column reachable by
        # exactly one sentence.
        candidates = terms_for("ORQT")
        self.assertIn("ordered qty", candidates)
        self.assertIn("order quantity", candidates)
        self.assertGreaterEqual(len(candidates), 4, candidates)
        # And they outrank the raw ERP code. analyze_identifier surfaces the
        # dictionary's synonyms as aliases too, so dropping them from the
        # vocabulary branch does not lose them — it demotes them below "orqt",
        # which is the one term in the list no reader will ever type.
        self.assertLess(candidates.index("ordered qty"), candidates.index("orqt"))
        self.assertLess(candidates.index("order quantity"), candidates.index("orqt"))

    def test_a_date_role_still_leads(self):
        self.assertEqual(terms_for("INVOICE_DATE")[0], "invoice date")

    def test_variants_are_still_offered_after_the_vocabulary(self):
        # A reader who has SEEN the column name will type it. The variants
        # earn their place in the matcher; they just do not lead.
        candidates = terms_for("BAL_VAL_AMT")
        self.assertIn("bal val amt", candidates)
        self.assertLess(candidates.index("balance value amount"),
                        candidates.index("bal val amt"))

    def test_the_budget_is_still_bounded(self):
        for column in ("BAL_VAL_AMT", "CUS_ORD_LIN_AMT", "INVOICE_DATE", "ORQT"):
            with self.subTest(column=column):
                self.assertLessEqual(len(terms_for(column)),
                                     MAX_BUSINESS_CANDIDATES)

    def test_nothing_is_offered_twice(self):
        for column in ("BAL_VAL_AMT", "STAT_CD", "PRODUCT_DESCRIPTION"):
            candidates = terms_for(column)
            self.assertEqual(len(candidates), len(set(candidates)), column)


class TestTheShorterPhrasingAlsoReaches(unittest.TestCase):
    """
    The matcher scores a multi-word term only when EVERY one of its words is
    in the question. Generating just the longest form meant only the longest
    phrasing matched.
    """

    def test_the_head_term_is_offered(self):
        self.assertIn("balance value", terms_for("BAL_VAL_AMT"))
        self.assertIn("net sales", terms_for("NET_SALES_AMT"))

    def test_a_reader_who_drops_the_unit_still_matches(self):
        self.assertTrue(matches("balance value by warehouse", "BAL_VAL_AMT"))
        self.assertTrue(matches("net sales by region", "NET_SALES_AMT"))

    def test_the_full_phrasing_still_matches_and_scores_higher(self):
        full = matches("balance value amount by warehouse", "BAL_VAL_AMT")
        short = matches("balance value by warehouse", "BAL_VAL_AMT")
        self.assertTrue(full and short)
        self.assertGreater(full, short)

    def test_it_never_reduces_to_a_single_word(self):
        # "unit price" -> "unit" would claim every question containing the
        # word; "order quantity" -> "order" would claim every order question.
        for expanded in ("unit price", "order quantity", "status code",
                         "line number", "amount", ""):
            with self.subTest(expanded=expanded):
                self.assertEqual(head_term(expanded), "")

    def test_it_only_drops_a_unit_noun(self):
        self.assertEqual(head_term("net sales amount"), "net sales")
        self.assertEqual(head_term("requested delivery date"), "")
        self.assertEqual(head_term("gross margin percent"), "gross margin")

    def test_the_original_failure_is_fixed(self):
        # The measurement this work started from: the generated synonyms
        # matched only if the reader typed the column name back at it.
        self.assertTrue(matches("bal val amt", "BAL_VAL_AMT"))
        self.assertTrue(matches("balance value by warehouse", "BAL_VAL_AMT"))

    def test_and_the_part_that_is_still_a_human_job_is_still_a_human_job(self):
        # "inventory value" is not derivable from BAL_VAL_AMT by any amount of
        # expansion — it is a fact about what this business calls the column,
        # and it reaches the resolver through the tenant's own column terms.
        # Asserted so the limit is recorded rather than assumed fixed.
        self.assertFalse(matches("inventory value by warehouse", "BAL_VAL_AMT"))


class TestTheUsefulnessGateComparesMeaningNotSpelling(unittest.TestCase):

    def test_re_spellings_are_not_useful(self):
        self.assertFalse(_has_useful_synonym(
            "BAL_VAL_AMT",
            {"balance val amt", "bal val amount", "balance value amount"}))

    def test_a_real_word_is_useful(self):
        self.assertTrue(_has_useful_synonym(
            "BAL_VAL_AMT", {"inventory value", "stock value"}))

    def test_a_term_that_adds_one_word_is_useful(self):
        self.assertTrue(_has_useful_synonym("STAT_CD", {"order status"}))
        self.assertTrue(_has_useful_synonym("CUS_ORD_NUM", {"sales order number"}))

    def test_the_expansion_counts_as_derivable(self):
        # "status code" is what STAT_CD expands to. The old gate compared
        # strings, so it read as a new term.
        self.assertFalse(_has_useful_synonym("STAT_CD", {"status code"}))

    def test_nothing_is_not_useful(self):
        for terms in (set(), {""}, {"   "}, {"", "  "}):
            with self.subTest(terms=terms):
                self.assertFalse(_has_useful_synonym("ANY_COL", terms))

    def test_a_role_is_written_even_when_the_synonym_adds_nothing(self):
        # The gate decides whether a column reaches entity_properties AT ALL,
        # not just whether its synonyms are good. A date or an identifier is
        # information no synonym carries, and a knowledge base that offers
        # "therapy start" for THERAPY_START_DATE_ID has added no word — but
        # the column still has to be known as a date.
        import json
        import tempfile
        from pathlib import Path as _Path

        from core.graph_autopopulate import classify_harvested_property

        # The classification itself, which is what the gate now defers to.
        self.assertEqual(
            classify_harvested_property("THERAPY_START_DATE_ID", "int"), "date")
        self.assertFalse(_has_useful_synonym(
            "THERAPY_START_DATE_ID", {"therapy start"}))

    def test_a_business_word_survives_the_gate(self):
        # The words a distributor actually uses for a warehouse key. If the
        # gate rejected these the harvest would drop exactly the terms it
        # exists to collect.
        self.assertTrue(_has_useful_synonym(
            "WHS_DMS_KEY", {"branch", "profit centre", "location"}))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
