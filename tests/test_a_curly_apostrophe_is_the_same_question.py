"""
tests/test_a_curly_apostrophe_is_the_same_question.py

The apostrophe a French reader actually types is U+2019, not U+0027.

An AZERTY layout, macOS, Word, Google Docs, Outlook and every mobile keyboard
substitute the typographic apostrophe for the ASCII one. The lexicon in
core/question_normalizer.py is written with ASCII apostrophes, matching runs on
folded text, and folding did not touch the apostrophe -- so the two spellings
shared not one key. Measured before the fix, on the exact questions an EMCO
reader types:

    chiffre d'affaires par succursale   -> 'revenue by succursale'
    chiffre d’affaires par succursale   -> 'chiffre d’affaires by succursale'

    ventes d'aujourd'hui                -> 'sales today'
    ventes d’aujourd’hui                -> 'sales d’aujourd’hui'

    depuis le debut de l'annee          -> 'year to date'
    depuis le début de l’année          -> 'depuis the debut of l’year'

The third row is the one that matters most, and it is not a cosmetic loss. The
canonical text is what detect_temporal_window reads; "year to date" carries a
this_year window, and "depuis the debut of l’year" carries none. A question
with no window gets no governed business-date policy, so generation falls
through to the dialect's own date recipes -- which use the SERVER CLOCK. The
reader asked for the year to date and was answered, confidently and with no
caveat, over whatever GETDATE() happened to mean.

So these tests assert EQUIVALENCE between the two spellings and, separately,
that the shared result is the right one. Equivalence alone would pass if both
spellings broke identically.
"""

from __future__ import annotations

import pytest

from core.contextual_dates import detect_temporal_window
from core.conversational import detect_conversational
from core.question_normalizer import _APOSTROPHE_FOLD, _fold, canonical_question

CURLY = "’"

# (ASCII spelling, the canonical English both spellings must reach)
PAIRS = [
    ("chiffre d'affaires par succursale", "revenue by succursale"),
    ("ventes d'aujourd'hui", "sales today"),
    ("depuis le debut de l'annee", "year to date"),
    ("l'evolution des ventes", "trend of sales"),
    ("chiffre d'affaires de l'annee derniere", "revenue of last year"),
    ("marge brute de l'annee", "gross margin of year"),
    ("ventes d'hier", "sales yesterday"),
]


def curly(text: str) -> str:
    return text.replace("'", CURLY)


class TestBothSpellingsCanonicaliseIdentically:

    @pytest.mark.parametrize("ascii_text,expected", PAIRS)
    def test_the_typographic_form_reaches_the_same_canonical_text(
            self, ascii_text, expected):
        assert canonical_question(curly(ascii_text), "fr") == \
            canonical_question(ascii_text, "fr")

    @pytest.mark.parametrize("ascii_text,expected", PAIRS)
    def test_and_that_shared_text_is_the_canonical_english(
            self, ascii_text, expected):
        """Without this, the test above would pass for two identically broken
        spellings."""
        assert canonical_question(ascii_text, "fr") == expected
        assert canonical_question(curly(ascii_text), "fr") == expected

    # Written out rather than read from _APOSTROPHE_FOLD: deriving the cases
    # from the implementation means shrinking the implementation silently
    # shrinks the test, which is exactly the mutation this has to catch.
    SUBSTITUTED = [
        "\u2019",  # right single quotation mark -- the AZERTY/macOS default
        "\u2018",  # left single quotation mark -- paired-quote autocorrect
        "\u02bc",  # modifier letter apostrophe
        "\u2032",  # prime
        "\uff07",  # fullwidth apostrophe
        "\u00b4",  # acute accent, reached for when the layout fights back
        "\u0060",  # grave accent, same
    ]

    @pytest.mark.parametrize("char", SUBSTITUTED)
    def test_every_substituted_apostrophe_folds(self, char):
        """Editors substitute more than one character. Each is folded to the
        ASCII apostrophe the lexicon keys are written with, so one entry covers
        all of them rather than one entry per editor."""
        typed = f"chiffre d{char}affaires par succursale"
        assert canonical_question(typed, "fr") == "revenue by succursale"

    def test_the_fold_table_covers_exactly_those(self):
        assert {chr(code) for code in _APOSTROPHE_FOLD} == set(self.SUBSTITUTED)

    def test_fold_itself_returns_the_ascii_apostrophe(self):
        assert _fold(f"L{CURLY}Année Dernière") == "l'annee derniere"


class TestTheWindowSurvivesTheApostrophe:
    """The canonical text is what the temporal-window detector reads. A lost
    apostrophe is a lost window, and a lost window is a clock-anchored answer
    that validates clean."""

    @pytest.mark.parametrize("ascii_text,kind", [
        ("depuis le debut de l'annee", "this_year"),
        ("ventes de l'annee derniere", "previous_year"),
        ("ventes d'aujourd'hui", "today"),
        ("ventes d'hier", "yesterday"),
    ])
    def test_the_same_window_is_detected_for_both_spellings(
            self, ascii_text, kind):
        from_ascii = detect_temporal_window(
            canonical_question(ascii_text, "fr"))
        from_curly = detect_temporal_window(
            canonical_question(curly(ascii_text), "fr"))
        assert from_ascii == from_curly
        assert from_curly.get("kind") == kind, from_curly

    def test_a_lost_window_is_what_this_prevents(self):
        """Stated as the failure, not the fix: the question below asked for the
        year to date, and with no window nothing anchors it to the governed
        business date."""
        window = detect_temporal_window(
            canonical_question("depuis le début de l’année", "fr"))
        assert window, "no window means a GETDATE()-anchored answer"
        assert window["anchor_policy"] == "latest_available"


class TestAQuotedValueIsStillAValue:
    """A curly-quoted customer name must be protected exactly as an
    ASCII-quoted one is -- folding the quotes must not start translating the
    tenant's own data, nor stop protecting it."""

    def test_a_curly_quoted_value_is_not_translated(self):
        assert canonical_question(
            f"chiffre d{CURLY}affaires pour ‘Banque de France’", "fr",
        ) == canonical_question(
            "chiffre d'affaires pour 'Banque de France'", "fr")

    def test_and_the_value_survives_intact(self):
        out = canonical_question(
            "chiffre d'affaires pour 'Banque de France'", "fr")
        assert "banque de france" in out, out
        assert "banque of france" not in out, out


class TestTheSmalltalkClassifierAgrees:
    """core/conversational.py folds through the same helper and used to repair
    the apostrophe again on its own line. One rule, one place."""

    @pytest.mark.parametrize("ascii_text,kind", [
        ("what's your opinion", "opinion"),
        ("that's wrong", "frustration"),
        ("what's the semantic layer", "semantic_explainer"),
    ])
    def test_both_spellings_classify_the_same(self, ascii_text, kind):
        assert detect_conversational(ascii_text) == kind
        assert detect_conversational(curly(ascii_text)) == kind


class TestEnglishIsUntouched:
    """Folding is for matching French. An English tenant must run the exact
    bytes it ran before -- including the possessive the fold would rewrite."""

    @pytest.mark.parametrize("text", [
        "John's revenue by region",
        "what's the revenue for Q1",
        "revenue by region",
    ])
    def test_the_text_is_returned_verbatim(self, text):
        assert canonical_question(text, "en") == text
        assert canonical_question(text, None) == text
