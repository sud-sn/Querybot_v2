# -*- coding: utf-8 -*-
"""A French manufacturer's nouns reached retrieval untranslated.

core.question_normalizer canonicalises a French question into the English the
detectors and the retriever read, and its lexicon is deliberately French
grammar plus the analytics vocabulary: it is not a dictionary and must not
become one. So "rebuts par centre de charge" reached BM25 -- which strips
accents and knows only the English KB -- as "rebuts by centre of charge", with
zero token overlap against a KB that says scrap and work centre.

The nouns of a trade now travel with the pack that knows the trade. A pack may
carry "terms_fr", merged into the tenant's vocabulary and applied by the
canonicaliser before its own lexicon, so a phrase is read whole before "de"
becomes "of". They apply only where the pack is selected: a distributor's
"gamme" is still a product range.

Every test executes the real canonicaliser under a real merged vocabulary.
"""

from __future__ import annotations

import unittest

from core.question_normalizer import canonical_question, canonicalise
from core.vocab_packs import _clone_builtin, _merge_pack, activate_vocab, deactivate_vocab, load_pack

MANUFACTURING = "manufacturing"


def with_pack(*pack_ids):
    vocab = _clone_builtin()
    for pack_id in pack_ids:
        _merge_pack(vocab, load_pack(pack_id), pack_id)
    return vocab


class _Under:
    def __init__(self, vocab):
        self.vocab = vocab

    def __enter__(self):
        self.token = activate_vocab(self.vocab)
        return self

    def __exit__(self, *exc):
        deactivate_vocab(self.token)


class TestTheTradesNounsReachRetrieval(unittest.TestCase):

    def test_scrap_by_work_centre(self):
        with _Under(with_pack(MANUFACTURING)):
            self.assertEqual(canonical_question("rebuts par centre de charge", "fr"), "scrap by work centre")

    def test_a_phrase_is_read_before_its_own_words(self):
        """"ordre de fabrication" contains "de", which the lexicon turns into
        "of", and "quart de travail" begins with a word that is itself a term.
        The longest phrase has to win first or the noun is split."""
        with _Under(with_pack(MANUFACTURING)):
            self.assertEqual(canonical_question("nombre d'ordres de fabrication par usine", "fr"),
                             "count of manufacturing orders by plant")
            self.assertEqual(canonical_question("rebuts par quart de travail", "fr"), "scrap by shift")

    def test_the_analytics_words_still_canonicalise_around_them(self):
        from core.query_semantics import analyze_query_intent
        with _Under(with_pack(MANUFACTURING)):
            out = canonical_question("les 5 meilleurs centres de charge par rendement l'année dernière", "fr")
        self.assertEqual(out, "the 5 best work centres by yield last year")
        self.assertTrue(analyze_query_intent(out).get("wants_top_n"))

    def test_a_production_date_can_be_named(self):
        with _Under(with_pack(MANUFACTURING)):
            self.assertEqual(canonical_question("quantité fabriquée par date de déclaration", "fr"),
                             "manufactured quantity by reported date")


class TestOnlyWhereThePackIsSelected(unittest.TestCase):

    def test_without_the_pack_the_nouns_stay_french(self):
        with _Under(_clone_builtin()):
            self.assertEqual(canonical_question("rebuts par centre de charge", "fr"), "rebuts by centre of charge")

    def test_a_distribution_tenant_keeps_its_own_words(self):
        with _Under(with_pack("wholesale_distribution")):
            self.assertEqual(canonical_question("rebuts par centre de charge", "fr"), "rebuts by centre of charge")

    def test_a_product_range_is_not_a_routing(self):
        """Bare "gamme" is left alone on purpose; the phrase says which."""
        with _Under(with_pack(MANUFACTURING)):
            self.assertEqual(canonical_question("ventes par gamme de produits", "fr"), "sales by gamme of products")
            self.assertEqual(canonical_question("temps par gamme opératoire", "fr"), "time by routing")

    def test_an_english_reader_is_untouched(self):
        with _Under(with_pack(MANUFACTURING)):
            text = "scrap by work centre last month"
            self.assertIs(canonical_question(text, "en"), text)

    def test_a_quoted_value_is_never_rewritten(self):
        with _Under(with_pack(MANUFACTURING)):
            self.assertEqual(canonical_question('rebuts pour le client "Usine Rebut"', "fr"),
                             'scrap for the customer "usine rebut"')

    def test_a_quoted_value_after_a_shortened_phrase_is_still_protected(self):
        """The pack's replacements are shorter than the French they replace,
        so everything after them moves left. The rules that run next must
        read the quotes where they are now, not where they were."""
        with _Under(with_pack(MANUFACTURING)):
            self.assertEqual(
                canonical_question('rebuts par centre de charge client "les 6 derniers mois"', "fr"),
                'scrap by work centre customer "les 6 derniers mois"')

    def test_no_active_vocabulary_is_the_lexicon_alone(self):
        self.assertEqual(canonicalise("rebuts par usine"), "rebuts by usine")


class TestThePackContract(unittest.TestCase):

    def test_the_terms_merge_into_the_vocabulary(self):
        vocab = with_pack(MANUFACTURING)
        self.assertEqual(vocab.french_terms["rebut"], "scrap")
        self.assertEqual(vocab.french_terms["centre de charge"], "work centre")

    def test_a_pack_without_terms_adds_none(self):
        vocab = with_pack("wholesale_distribution")
        self.assertEqual(vocab.french_terms, {})

    def test_every_term_maps_to_english_the_lexicon_leaves_alone(self):
        """The lexicon runs AFTER the pack, over the English it produced. A
        target with a word the lexicon also knows as French ("on", "date")
        would be translated a second time -- so each target is run through the
        canonicaliser on its own, with no pack active, and must come back as
        it went in."""
        for french, english in load_pack(MANUFACTURING)["terms_fr"].items():
            with self.subTest(french=french):
                self.assertEqual(canonicalise(english), english)
                self.assertNotEqual(french.strip().lower(), english.strip().lower())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
