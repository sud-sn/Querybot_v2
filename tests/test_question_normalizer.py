"""
tests/test_question_normalizer.py

The French question normaliser, and the seams it is wired into.

The number that matters is at the bottom: how many analytical intents a French
question recovers against the English one it means. Everything above it exists
because the ways this can go wrong are all silent -- a French question with no
intents detected does not error, it produces a plain grouped SELECT and a
confident narrative answer to a question nobody asked.
"""

from __future__ import annotations

import re

import pytest

from core.insight import detect_analytical_intents, is_causal_question
from core.query_semantics import analyze_query_intent, detect_top_n_intent
from core.question_normalizer import canonical_question, canonicalise


# The corpus. English on the left is what the product already understands;
# French on the right is the same question as a French customer would ask it.
CORPUS = [
    ("Compare 2025 against 2024 by revenue category",
     "Compare 2025 par rapport à 2024 par catégorie de chiffre d'affaires"),
    ("Show me the top 10 customers by margin",
     "Montre-moi les 10 meilleurs clients par marge"),
    ("What is the trend of sales over the last 6 months",
     "Quelle est l'évolution des ventes sur les 6 derniers mois"),
    ("What did each region contribute to revenue",
     "Qu'est-ce que chaque région a contribué au chiffre d'affaires"),
    ("Forecast revenue for the next 3 months",
     "Prévision du chiffre d'affaires pour les 3 prochains mois"),
    ("What is the budget variance by region for Q1 2025",
     "Quel est l'écart budgétaire par région pour T1 2025"),
    ("Show the margin outliers last month",
     "Montre les valeurs aberrantes de marge le mois dernier"),
    ("Breakdown of revenue by product",
     "Répartition du chiffre d'affaires par produit"),
    ("Correlation between margin and revenue",
     "Corrélation entre la marge et le chiffre d'affaires"),
    ("Rolling average of sales over 3 months",
     "Moyenne mobile des ventes sur 3 mois"),
]


def _intents(text):
    return {name for name, value in detect_analytical_intents(text).items() if value}


def _flags(text):
    return {name for name, value in analyze_query_intent(text).items() if value}


# ══════════════════════════════════════════════════════════════════════════════
# English is untouched
# ══════════════════════════════════════════════════════════════════════════════

class TestAnEnglishReaderRunsTheExactBytesTheyRanBefore:

    @pytest.mark.parametrize("english,_french", CORPUS)
    def test_the_text_is_returned_unchanged(self, english, _french):
        assert canonical_question(english, "en") is english

    def test_a_missing_language_is_treated_as_english(self):
        """(user or {}).get('lang') is None on a pre-migration row."""
        assert canonical_question("Show me the top 10", None) == "Show me the top 10"
        assert canonical_question("Show me the top 10", "") == "Show me the top 10"

    def test_an_unsupported_language_is_returned_unchanged(self):
        """German is not canonicalisable here, and half-canonicalising it
        would be worse than leaving it alone."""
        assert canonical_question("Zeig mir die 10 besten Kunden", "de") == \
            "Zeig mir die 10 besten Kunden"

    def test_empty_input_survives(self):
        assert canonical_question("", "fr") == ""
        assert canonical_question(None, "fr") is None


# ══════════════════════════════════════════════════════════════════════════════
# The lexicon
# ══════════════════════════════════════════════════════════════════════════════

class TestTheLexicon:

    def test_a_phrase_beats_its_own_words(self):
        """"chiffre d'affaires" is three words for one measure. Matched word by
        word it would come out as "figure of business"."""
        assert "revenue" in canonicalise("le chiffre d'affaires")
        assert "figure" not in canonicalise("le chiffre d'affaires")

    def test_accents_are_optional(self):
        """They are the first thing a hurried typist drops, and
        core/date_roles.py shreds them anyway."""
        assert canonicalise("prévision") == canonicalise("prevision") == "forecast"
        assert "last year" in canonicalise("l'année dernière")
        assert "last year" in canonicalise("l'annee derniere")

    def test_the_elided_article_does_not_survive_the_word_it_was_attached_to(self):
        """"l'évolution" matches "évolution" and would come out "l'trend"."""
        assert canonicalise("l'évolution") == "trend"
        assert canonicalise("d'anomalies") == "anomalies"

    def test_a_word_is_not_matched_inside_another(self):
        assert "margin" not in canonicalise("margelle")
        assert "month" not in canonicalise("moissonneuse")

    def test_the_pronoun_ca_is_not_read_as_a_measure(self):
        """Folded, the abbreviation for chiffre d'affaires and the pronoun
        "ça" are the same two letters -- so the abbreviation is deliberately
        absent from the lexicon rather than turning "qu'est-ce que ça donne"
        into "what does revenue give"."""
        assert "revenue" not in canonicalise("qu'est-ce que ça donne")


class TestCustomerValuesAreNotRewritten:

    def test_a_quoted_value_is_left_alone(self):
        """A tenant with a customer called "Marge" would otherwise have its own
        data rewritten on the way to retrieval."""
        out = canonicalise("chiffre d'affaires pour 'Marge SA'")
        assert "'marge sa'" in out
        assert out.startswith("revenue")

    def test_the_elision_apostrophe_does_not_break_the_quote_pairing(self):
        """This is the bug a naive `'[^']*'` produces: the apostrophe in
        "d'affaires" pairs with the opening quote, so the real value goes
        unprotected AND half the question goes untranslated."""
        out = canonicalise("Chiffre d'affaires pour 'Banque de France' cette année")
        assert "'banque de france'" in out       # untouched inside the quotes
        assert out.startswith("revenue for")     # translated outside them
        assert "this year" in out

    def test_double_quotes_and_guillemets_protect_too(self):
        assert '"marge"' in canonicalise('ventes de "marge"')
        assert "«marge»" in canonicalise("ventes de «marge»")


class TestTheNumericRules:

    def test_a_relative_window_becomes_the_english_shape(self):
        assert canonicalise("les 6 derniers mois") == "last 6 months"
        assert canonicalise("ces 12 derniers mois") == "last 12 months"
        assert canonicalise("les 3 prochains trimestres") == "next 3 quarters"

    def test_a_french_quarter_becomes_the_letter_the_detector_reads(self):
        """core/multi_period.py's quarter pattern is anchored on Q."""
        assert canonicalise("T1 2025") == "Q1 2025"
        assert canonicalise("le 3e trimestre") == "Q3"

    def test_an_invariant_unit_gets_its_english_plural_from_the_number(self):
        """French "mois" is the same word singular and plural, so only the
        number knows which English form to use."""
        assert canonicalise("sur 3 mois") == "over 3 months"
        assert canonicalise("sur 1 mois") == "over 1 months"


# ══════════════════════════════════════════════════════════════════════════════
# What the detectors then see
# ══════════════════════════════════════════════════════════════════════════════

class TestTheDetectorsRecoverTheIntent:

    @pytest.mark.parametrize("english,french", CORPUS)
    def test_the_canonical_form_detects_what_the_english_does(self, english, french):
        missing = _intents(english) - _intents(canonical_question(french, "fr"))
        assert not missing, (french, sorted(missing))

    def test_the_raw_french_detects_almost_nothing(self):
        """The measurement this whole module exists for. Without it a French
        question reaches SQL generation with no intent detected and nothing
        anywhere reports a problem."""
        raw = sum(len(_intents(fr)) for _, fr in CORPUS)
        english = sum(len(_intents(en)) for en, _ in CORPUS)
        canonical = sum(len(_intents(canonical_question(fr, "fr"))) for _, fr in CORPUS)
        assert english >= 10, "the corpus stopped exercising the detectors"
        assert raw <= 2, f"raw French now detects {raw}; the premise changed"
        assert canonical >= english, (canonical, english)

    def test_the_semantic_flags_come_back_too(self):
        raw = sum(len(_flags(fr)) for _, fr in CORPUS)
        canonical = sum(len(_flags(canonical_question(fr, "fr"))) for _, fr in CORPUS)
        assert canonical > raw * 2, (canonical, raw)

    def test_the_top_ten_question_gets_its_row_limit_back(self):
        """The failure this was built for. Without a limit the reader is handed
        every customer, narrated as the top 10."""
        french = "Montre-moi les 10 meilleurs clients par marge"
        assert detect_top_n_intent(french) is None
        recovered = detect_top_n_intent(canonical_question(french, "fr"))
        assert recovered is not None
        assert recovered.limit == 10
        assert recovered.direction == "descending"

    def test_the_bottom_n_direction_survives_too(self):
        recovered = detect_top_n_intent(
            canonical_question("les 5 pires produits par marge", "fr"))
        assert recovered.limit == 5
        assert recovered.direction == "ascending"

    def test_a_causal_question_reaches_the_causal_route(self):
        french = "Pourquoi les ventes ont-elles baissé ?"
        assert is_causal_question(french) is False
        assert is_causal_question(canonical_question(french, "fr")) is True

    def test_contribute_not_contributed(self):
        """core/contribution_analysis.py matches `contribut(?:ion|e|es|ing)?` --
        the past participle is the one form it does NOT read, so the obvious
        translation of "a contribué" is the one that fails."""
        out = canonical_question(
            "Qu'est-ce que chaque région a contribué au chiffre d'affaires", "fr")
        assert "contribute" in out and "contributed" not in out
        assert "contribution" in _intents(out)


# ══════════════════════════════════════════════════════════════════════════════
# The wiring
# ══════════════════════════════════════════════════════════════════════════════

class TestThePipelineReadsTheCanonicalTextAndKeepsTheReadersOwn:

    def _source(self):
        from pathlib import Path
        return (Path(__file__).resolve().parents[1]
                / "core" / "query_pipeline.py").read_text(encoding="utf-8")

    def test_every_detector_seam_reads_the_canonical_question(self):
        src = self._source()
        for call in (
            "is_causal_question(_analysis_question)",
            "build_generic_query_hints(_analysis_question)",
            "analyze_query_intent(_analysis_question)",
            "detect_top_n_intent(_analysis_question)",
            "detect_analytical_intents(_analysis_question)",
        ):
            assert call in src, call

    def test_retrieval_reads_it_too(self):
        """BM25 strips every non-[A-Za-z0-9_] character and the embedder is
        English-only: measured token overlap between a French question and the
        English KB is zero."""
        src = self._source()
        assert "retriever.retrieve(_analysis_question" in src
        assert "retriever.retrieve(question" not in src
        assert "retrieve_similar_examples(\n            _analysis_question" in src

    def test_the_planner_carrier_is_canonicalised(self):
        src = self._source()
        assert "_semantic_plan_question = canonical_question(" in src

    def test_the_readers_own_words_still_reach_the_trace_and_the_title(self):
        """The canonical form is an ADDED name. If it ever became a
        replacement, a French user's dashboard would fill with English tile
        names -- the tile is titled from the question."""
        src = self._source()
        assert "question=question," in src            # _trace_create
        assert "_analysis_question = canonical_question(" in src
        # The function still takes, and passes on, the reader's own text.
        assert "async def _handle_query_impl(account_id, event, adapter, question," in src

    def test_value_resolution_is_not_given_the_canonical_text(self):
        """Deliberate, and a departure from the plan's own list. Candidate
        phrases are customer VALUES -- a customer called "Marge" or a product
        called "Premier" is a French word that must not be rewritten. Quoted
        spans are protected in the normaliser, but an unquoted one is not, and
        value resolution is the one consumer where a rewrite changes which rows
        come back rather than merely which words are searched."""
        src = self._source()
        assert "extract_candidate_phrases(_analysis_question" not in src

    def test_the_language_comes_from_the_portal_user_not_the_display_context(self):
        """display_context is a cached snapshot on a governed-cache hit, so a
        user who switched to French would keep getting English."""
        src = self._source()
        seam = src[src.index("_analysis_question = canonical_question("):]
        assert '(portal_user or {}).get("lang")' in seam[:300]
