"""
tests/test_a_french_followup_reaches_the_result.py

"Et par entrepôt ?" started a new query against the warehouse.

The cached-result router decides whether a turn is a follow-up about the result
already on screen or a fresh question for the database. It is a stack of English
regexes over the reader's raw words, so a French reader reached none of them.
Measured over ten follow-up pairs against a live cached result:

    English routes 8/10
    French  routes 0/10

Not one. And a follow-up that does not route is not a smaller answer -- it
becomes a FRESH, unrelated query, so "et par entrepôt ?" stops meaning "the
result I am looking at, by warehouse" and becomes "warehouses, over whatever
window and whatever fact the planner picks on its own".

The worst row is the one that names a column outright: "quelle est la moyenne de
NET_SLS_AMT ?" contains a cached column name in full, and still missed, because
the word that had to be recognised alongside it was "average".

A second, independent defect in the same module, and it is not about English at
all. Cached VALUES were compared with `[^a-z0-9]+` over casefolded text, which
does not strip an accent -- it SHREDS it. "Montréal" flattened to "montr" + "al"
while "Montreal" flattened to "montreal", so on EMCO's own branch names the two
never met, in BOTH directions:

    cached warehouse   reader typed    matched
    Montréal           Montreal        no
    Montreal           Montréal        no

A reader could not reference their own data.

The routing fix is the one this session has used throughout: the English phrase
gates read the canonical English as well as the reader's words, and the data
references read the reader's words, folded. One exception: _looks_like_new_query,
the single gate here that REJECTS, reads the canonical text only -- it is an
English starter-word heuristic and untranslated French is not an input it can say
anything about. That is the right input rather than a behaviour difference: asked
over both spellings the answer is the same on every phrasing tried.
"""

from __future__ import annotations

import pytest

from core.query_router import (
    _question_references_cached_value,
    _spellings,
    question_mentions_cached_column,
    should_attempt_cache_followup,
    should_route_to_result_cache,
)
from core.question_normalizer import canonical_question

COLUMNS = ["WHS_DSC", "NET_SLS_AMT", "IVC_MTH"]
ROWS = [
    {"WHS_DSC": "Montréal", "NET_SLS_AMT": 40_000_000.0, "IVC_MTH": "2025-04"},
    {"WHS_DSC": "Toronto", "NET_SLS_AMT": 35_000_000.0, "IVC_MTH": "2025-05"},
    {"WHS_DSC": "Trois-Rivières", "NET_SLS_AMT": 9_000_000.0, "IVC_MTH": "2025-06"},
]

# (English follow-up, its French twin). Every English one routes today.
FOLLOW_UPS = [
    ("and by warehouse?", "et par entrepôt ?"),
    ("what about last month?", "et le mois dernier ?"),
    ("show me the top 3", "montre-moi les 3 premiers"),
    ("only Montreal", "seulement Montréal"),
    ("sort these by revenue", "trie ces résultats par chiffre d'affaires"),
    ("what is the average of NET_SLS_AMT?",
     "quelle est la moyenne de NET_SLS_AMT ?"),
    ("drill into Montreal", "détaille Montréal"),
    ("remove the previous rows", "supprime les lignes précédentes"),
]

# Questions that are NOT about the result on screen. Routing one of these to the
# cache answers a new question from a stale snapshot, which is worse than not
# routing a follow-up.
FRESH_QUESTIONS = [
    ("what is the total net sales for the last 6 months?",
     "quel est le total des ventes nettes des 6 derniers mois ?"),
    ("how many customers do we have?", "combien de clients avons-nous ?"),
    ("show me returns by product category",
     "montre-moi les retours par catégorie de produit"),
    ("net sales by warehouse for 2024",
     "ventes nettes par entrepôt pour 2024"),
]


def routes(question: str, lang: str = "en", *, cached: bool = True,
           rows: list[dict] | None = None) -> bool:
    return should_attempt_cache_followup(
        question, cached, cached_col_names=COLUMNS,
        cached_rows=ROWS if rows is None else rows, lang=lang)


class TestAFrenchFollowUpRoutesLikeItsEnglishTwin:

    @pytest.mark.parametrize("english,french", FOLLOW_UPS)
    def test_the_english_twin_routes(self, english, french):
        """Ground truth. Without it the pair test could pass by routing
        neither."""
        assert routes(english) is True, english

    @pytest.mark.parametrize("english,french", FOLLOW_UPS)
    def test_and_so_does_the_french_one(self, english, french):
        assert routes(french, "fr") is True, french

    def test_a_column_named_in_full_is_enough(self):
        """The row that made the defect plainest: the cached column appears
        verbatim and the turn still became a fresh query, because the word
        beside it was "average"."""
        assert routes("quelle est la moyenne de NET_SLS_AMT ?", "fr") is True

    def test_the_french_word_order_for_a_top_n_is_read(self):
        """French puts the count first, so "les 3 premiers" canonicalised to
        "the 3 top" and every gate downstream is written for "top 3"."""
        assert canonical_question("montre-moi les 3 premiers", "fr") == \
            "show me top 3"
        assert routes("montre-moi les 3 premiers", "fr") is True

    def test_and_the_first_n_months_is_still_a_window_not_a_ranking(self):
        """"les 3 premiers mois" is the FIRST three months. Reading it as a
        top-3 would answer a period question with a ranking."""
        assert canonical_question("les 3 premiers mois", "fr") == \
            "the first 3 months"
        assert canonical_question("les six premiers mois", "fr") == \
            "the first 6 months"


class TestANewQuestionIsStillANewQuestion:
    """Widening the net must not start answering new questions from a stale
    snapshot. That failure is silent and worse than the one being fixed."""

    @pytest.mark.parametrize("english,french", FRESH_QUESTIONS)
    def test_neither_language_is_hijacked_by_the_cache(self, english, french):
        assert routes(english) is False, english
        assert routes(french, "fr") is False, french

    @pytest.mark.parametrize("english,french", FOLLOW_UPS + FRESH_QUESTIONS)
    def test_nothing_routes_without_a_cached_result(self, english, french):
        """A fresh session pays no extra latency and takes no extra risk."""
        assert routes(english, cached=False) is False
        assert routes(french, "fr", cached=False) is False

    def test_both_spellings_are_offered_and_english_gets_only_one(self):
        assert _spellings("montre-moi les 3 premiers", "fr") == (
            "montre-moi les 3 premiers", "show me top 3")
        assert _spellings("show me the top 3", "en") == ("show me the top 3",)
        assert _spellings("show me the top 3", "") == ("show me the top 3",)


class TestTheReaderCanNameTheirOwnData:
    """Cached values are tenant data, not vocabulary. Matching them was
    accent-exact, in both directions."""

    @pytest.mark.parametrize("typed", [
        "drill into Montréal", "drill into Montreal",
        "détaille Montréal", "only montreal", "MONTREAL only",
    ])
    def test_an_accented_warehouse_is_reachable_however_it_is_typed(self, typed):
        assert _question_references_cached_value(
            typed, [{"WHS_DSC": "Montréal"}]) is True

    @pytest.mark.parametrize("typed", [
        "drill into Montréal", "drill into Montreal",
    ])
    def test_and_so_is_a_plain_one(self, typed):
        assert _question_references_cached_value(
            typed, [{"WHS_DSC": "Montreal"}]) is True

    def test_a_hyphenated_accented_value_too(self):
        assert _question_references_cached_value(
            "seulement Trois-Rivières", ROWS) is True
        assert _question_references_cached_value(
            "only Trois Rivieres", ROWS) is True

    def test_a_value_that_is_not_there_is_still_not_there(self):
        """Folding must widen the match, not erase it."""
        assert _question_references_cached_value("drill into Calgary", ROWS) is False
        assert _question_references_cached_value("drill into Vancouver", ROWS) is False

    def test_no_rows_means_no_match(self):
        assert _question_references_cached_value("only Montréal", None) is False
        assert _question_references_cached_value("only Montréal", []) is False

    def test_a_column_name_is_matched_through_an_accent_too(self):
        assert question_mentions_cached_column(
            "la moyenne de la quantité", ["QUANTITE"]) is True
        assert question_mentions_cached_column(
            "la moyenne de quantité commandée",
            ["QUANTITE_COMMANDEE"]) is True

    @pytest.mark.parametrize("value,rewritten", [
        ("Marge", "margin"),
        ("Vente", "sales"),
    ])
    def test_a_value_the_canonicaliser_would_translate_is_still_found(
            self, value, rewritten):
        """The reason cached values are matched on the READER's words. A
        warehouse or channel called "Marge" canonicalises to "margin" and a
        product line called "Vente" to "sales" -- translate the sentence and
        the tenant's own value is gone from it."""
        assert canonical_question(f"seulement {value}", "fr") == \
            f"seulement {rewritten}"
        assert routes(f"seulement {value}", "fr",
                      rows=[{"WHS_DSC": value}]) is True


class TestEnglishRunsTheBytesItRanBefore:

    @pytest.mark.parametrize("english,french", FOLLOW_UPS + FRESH_QUESTIONS)
    def test_the_default_language_needs_no_argument(self, english, french):
        assert should_attempt_cache_followup(
            english, True, cached_col_names=COLUMNS, cached_rows=ROWS) == \
            routes(english, "en")

    @pytest.mark.parametrize("english,french", FOLLOW_UPS + FRESH_QUESTIONS)
    def test_the_phrase_gate_is_unchanged_for_english(self, english, french):
        assert should_route_to_result_cache(
            english, True, COLUMNS) == \
            should_route_to_result_cache(english, True, COLUMNS, lang="en")

    def test_an_unknown_language_is_treated_as_english(self):
        assert routes("and by warehouse?", "de") is True
        assert routes("et par entrepôt ?", "de") is False
