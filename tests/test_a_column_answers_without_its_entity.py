"""
tests/test_a_column_answers_without_its_entity.py

ITM_GRS_WT on the item table is the item's gross weight. It answered only to
its whole name read out -- "item gross weight" -- so "average gross weight by
item", "gross weight of each item" and, in French, "poids brut moyen par
article" reached no column: a reader asking about items does not say "item"
twice.

A column whose name begins with its own dimension's entity now also answers
to the rest of its name, when the rest says something on its own: not one
generic word (ITM_NM is not "name"), and never for a key.

The real planner, under real vocabularies. Synthetic names in a real
inventory warehouse's naming convention; no customer data.
"""

from __future__ import annotations

import unittest

from core.question_normalizer import canonical_question
from core.semantic_planner import build_semantic_field_plan
from core.vocab_packs import _clone_builtin, _merge_pack, activate_vocab, deactivate_vocab, load_pack

TABLES = {
    "MART.ITM_BAL_DLY_FCT": {c: "" for c in (
        "ITM_BAL_DLY_FCT_KEY", "ITM_DMS_KEY", "ITM_GRP_DMS_KEY", "WHS_DMS_KEY", "ON_HND_QTY")},
    "MART.ITM_DMS": {c: "" for c in (
        "ITM_DMS_KEY", "ITM_CD", "ITM_NM", "ITM_GRS_WT", "ITM_NET_WT", "ITM_GRP_DMS_KEY", "ITM_QXR_IND")},
    "MART.ITM_GRP_DMS": {c: "" for c in ("ITM_GRP_DMS_KEY", "ITM_GRP_CD", "ITM_GRP_DSC")},
    "MART.PFT_CTR_DMS": {c: "" for c in ("PFT_CTR_DMS_KEY", "PFT_CTR_NM", "PC_PSL_CD")},
}


def _vocab(*packs):
    vocab = _clone_builtin()
    for pack in packs:
        _merge_pack(vocab, load_pack(pack), pack)
    return vocab


def _fields(question, vocab=None, lang="en"):
    vocab = vocab or _vocab()
    token = activate_vocab(vocab)
    try:
        text = canonical_question(question, lang) if lang == "fr" else question
        plan = build_semantic_field_plan(text, TABLES, vocab=vocab)
    finally:
        deactivate_vocab(token)
    return {(f["table"].split(".")[-1], f["column"]) for f in plan.get("fields", [])}


class TestTheRestOfTheName(unittest.TestCase):

    def test_gross_and_net_weight(self):
        self.assertIn(("ITM_DMS", "ITM_GRS_WT"), _fields("average gross weight by item"))
        self.assertIn(("ITM_DMS", "ITM_GRS_WT"), _fields("gross weight of each item"))
        self.assertIn(("ITM_DMS", "ITM_NET_WT"), _fields("items by net weight"))

    def test_in_french(self):
        vocab = _vocab("infor_m3", "wholesale_distribution")
        self.assertIn(("ITM_DMS", "ITM_GRS_WT"), _fields("poids brut moyen par article", vocab, "fr"))

    def test_the_whole_name_still_answers(self):
        self.assertIn(("ITM_DMS", "ITM_GRS_WT"), _fields("average item gross weight by item"))


class TestWhatDoesNotAnswer(unittest.TestCase):

    def test_one_generic_word_is_not_enough(self):
        """ITM_NM is not "name": a question that says "name" is not about items."""
        self.assertNotIn(("ITM_DMS", "ITM_NM"), _fields("stock on hand by warehouse name"))

    def test_a_key_joins_and_does_not_answer(self):
        """Nor through the display column the planner would swap a key for."""
        fields = _fields("stock on hand by group")
        self.assertNotIn(("ITM_DMS", "ITM_GRP_DMS_KEY"), fields)
        self.assertNotIn(("ITM_GRP_DMS", "ITM_GRP_DSC"), fields)

    def test_a_rest_nothing_reads_does_not_answer(self):
        self.assertNotIn(("ITM_DMS", "ITM_QXR_IND"), _fields("items by qxr indicator"))

    def test_another_entitys_prefix_is_not_the_tables(self):
        """PC_PSL_CD sits on the profit center table under another prefix."""
        self.assertNotIn(("PFT_CTR_DMS", "PC_PSL_CD"), _fields("stock on hand by postal code"))

    def test_only_a_dimension_is_an_entity(self):
        tables = {"MART.ITM_BAL_FCT": {c: "" for c in ("ITM_BAL_FCT_KEY", "ITM_BAL_GRS_WT")}}
        plan = build_semantic_field_plan("gross weight by day", tables, vocab=_vocab())
        self.assertEqual(plan.get("fields", []), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
