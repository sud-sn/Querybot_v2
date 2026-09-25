"""
tests/test_french_stock_questions_reach_the_columns.py

A French reader asking about stock reached nothing. The canonicaliser carries
French grammar and the analytics vocabulary; the nouns of a trade travel with
the pack that knows it, and the distribution pack carried none. So:

    "stock disponible par groupe d'articles"  -> "inventory disponible by groupe articles"
    "valeur du stock par succursale"          -> "valeur of inventory by succursale"
    "commandes en souffrance par acheteur"    -> "orders en souffrance by acheteur"
    "les 10 principaux fournisseurs"          -> "the 10 principaux suppliers"

and no column answered. The distribution pack now carries its French terms --
items and item groups, stock on hand and its states, back orders, branches,
showrooms, buyers, places -- applied only where the pack is selected; and the
lexicon reads the adjectives it lacked ("moyen", "annuelle", "brut", "valeur")
and "les 10 principaux" as a top 10.

Every test runs the real canonicaliser, and the planner, under a real merged
vocabulary. Synthetic column names; no customer data.
"""

from __future__ import annotations

import unittest

from core.question_normalizer import canonical_question
from core.semantic_planner import build_semantic_field_plan
from core.vocab_packs import _clone_builtin, _merge_pack, activate_vocab, deactivate_vocab, load_pack


def _vocab(*packs):
    vocab = _clone_builtin()
    for pack in packs:
        _merge_pack(vocab, load_pack(pack), pack)
    return vocab


class _Under:
    def __init__(self, vocab):
        self.vocab = vocab

    def __enter__(self):
        self.token = activate_vocab(self.vocab)
        return self

    def __exit__(self, *exc):
        deactivate_vocab(self.token)


class TestTheTradesFrenchNouns(unittest.TestCase):

    CASES = (
        ("stock disponible par groupe d'articles", "inventory on hand by item group"),
        ("quantité en stock par article", "quantity on hand by item"),
        ("valeur du stock par succursale", "inventory value by branch"),
        ("commandes en souffrance par acheteur", "back orders by buyer"),
        ("stock réservé par fournisseur", "reserved inventory by supplier"),
        ("poids brut par article", "gross weight by item"),
        ("quantité par unité de mesure", "quantity by unit of measure"),
        ("stock par classe ABC", "inventory by abc class"),
        ("coût moyen par famille de produits", "average cost by product group"),
        ("stock par centre de profit", "inventory by profit center"),
        ("stock par ville et code postal", "inventory by city and postal code"),
        ("demande annuelle par article", "annual demand by item"),
        ("ventes par salle de montre", "sales by showroom"),
        ("taux de service par succursale", "fill rate by branch"),
    )

    def test_each_reads_in_english_under_the_pack(self):
        with _Under(_vocab("wholesale_distribution")):
            for french, english in self.CASES:
                with self.subTest(french=french):
                    self.assertEqual(canonical_question(french, "fr"), english)

    def test_they_apply_only_where_the_pack_is_selected(self):
        with _Under(_vocab()):
            self.assertEqual(canonical_question("valeur du stock par succursale", "fr"),
                             "value of inventory by succursale")

    def test_a_phrase_is_read_before_its_words(self):
        """"groupe d'articles" before "articles", "quantité en stock" before
        "en stock": the longer phrase is the trade's word."""
        with _Under(_vocab("wholesale_distribution")):
            self.assertEqual(canonical_question("articles en stock", "fr"), "items on hand")
            self.assertEqual(canonical_question("groupes d'articles", "fr"), "item groups")


class TestTheLexiconsAdjectivesAndRankings(unittest.TestCase):
    """Grammar, not a trade: these hold with no pack at all."""

    CASES = (
        ("prix moyen", "price average"),
        ("ventes moyennes", "sales average"),
        ("ventes annuelles", "sales annual"),
        ("valeur des ventes", "value of sales"),
        ("marge brute", "gross margin"),
        ("montant brut", "amount gross"),
        ("valeurs aberrantes des ventes", "outliers of sales"),
        ("les 10 principaux clients", "top 10 customers"),
        ("les cinq principales régions", "top 5 regions"),
        ("les principaux fournisseurs", "the top suppliers"),
    )

    def test_each(self):
        with _Under(_vocab()):
            for french, english in self.CASES:
                with self.subTest(french=french):
                    self.assertEqual(canonical_question(french, "fr"), english)

    def test_the_main_one_is_not_a_ranking(self):
        with _Under(_vocab()):
            self.assertEqual(canonical_question("le fournisseur principal", "fr"),
                             "the supplier principal")

    def test_the_first_months_are_still_a_window(self):
        with _Under(_vocab()):
            self.assertEqual(canonical_question("les 3 premiers mois", "fr"), "the first 3 months")


class TestTheQuestionReachesTheColumns(unittest.TestCase):
    """The ERP pack and the distribution pack together, as a distributor on an
    Infor M3 warehouse would select them."""

    TABLES = {
        "MART.ITM_BAL_DLY_FCT": {c: "" for c in (
            "ITM_BAL_DLY_FCT_KEY", "ITM_DMS_KEY", "ITM_GRP_DMS_KEY", "ON_HND_QTY", "ITM_CST")},
        "MART.ITM_GRP_DMS": {c: "" for c in ("ITM_GRP_DMS_KEY", "ITM_GRP_CD", "ITM_GRP_DSC")},
        "MART.ITM_DMS": {c: "" for c in ("ITM_DMS_KEY", "ITM_CD", "ITM_NM")},
    }

    def _fields(self, french):
        vocab = _vocab("infor_m3", "wholesale_distribution")
        with _Under(vocab):
            question = canonical_question(french, "fr")
            plan = build_semantic_field_plan(question, self.TABLES, vocab=vocab)
        return {(f["table"].split(".")[-1], f["column"]) for f in plan.get("fields", [])}

    def test_stock_on_hand_by_item_group(self):
        fields = self._fields("stock disponible par groupe d'articles")
        self.assertIn(("ITM_BAL_DLY_FCT", "ON_HND_QTY"), fields)
        self.assertIn(("ITM_GRP_DMS", "ITM_GRP_DSC"), fields)

    def test_quantity_on_hand_by_item(self):
        fields = self._fields("quantité en stock par article")
        self.assertIn(("ITM_BAL_DLY_FCT", "ON_HND_QTY"), fields)
        self.assertIn(("ITM_DMS", "ITM_NM"), fields)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
