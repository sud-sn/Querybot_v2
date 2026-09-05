"""
tests/test_cannot_generate_hint_language.py

The "I could not answer that — here is what you CAN ask" hint, in both
languages, and the suggestions in it made to actually work.

_build_cannot_generate_hint is the last thing a reader sees when both the
cached-result engine and the database fallback have failed. Its whole job is
to be acted on: the example questions in it are typed back into the result
chat. That makes them a route, not decoration, and a route has to go
somewhere.

Two things were wrong before the translation, and neither was about French:

  * The summary-total branch told every tenant to ask for "prescriptions with
    their patient details", in a module a steel distributor and a pharmacy
    share. Nothing in the function knows the domain, so the examples no longer
    pretend to.
  * The result-chat path never called core/question_normalizer.py. The main
    question path has done so since the French work began; this branch read
    the reader's own words with hand-written English rules, which is what
    produces this hint for a French reader in the first place. So the hint was
    about to hand them French suggestions down a path that had just failed on
    French.

The English output is asserted byte-identical for every shape except the
de-domained one, against a snapshot taken before the change.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import i18n
from core.question_normalizer import canonical_question
from core.result_renderer import _build_cannot_generate_hint as build_hint

SCHEMA_NUM = [{"name": "NET_AMOUNT"}]
STATS_NUM = {"columns": [{"name": "NET_AMOUNT", "min": 0, "is_currency": True}]}
SCHEMA_MIX = [{"name": "REGION"}, {"name": "NET_AMOUNT"}]
STATS_MIX = {"columns": [
    {"name": "NET_AMOUNT", "min": 0},
    {"name": "REGION", "sample_values": ["North", "South"]},
]}

# Every shape the function can return, named.
SHAPES = {
    "no_schema": ([], {}, {}),
    "summary_single_row": (SCHEMA_NUM, STATS_NUM, {"prev_rows": [{"NET_AMOUNT": 91234}]}),
    "summary_no_rows": (SCHEMA_NUM, STATS_NUM, {"prev_rows": []}),
    "mixed_with_values": (SCHEMA_MIX, STATS_MIX, {"prev_rows": [
        {"REGION": "North", "NET_AMOUNT": 1}, {"REGION": "South", "NET_AMOUNT": 2}]}),
    "mixed_no_prev": (SCHEMA_MIX, STATS_MIX, {}),
}


class _InLanguage:
    def __init__(self, lang):
        self.lang = lang

    def __enter__(self):
        self._token = i18n.activate_language(self.lang)
        return self

    def __exit__(self, *exc):
        i18n.deactivate_language(self._token)
        return False


def hint(shape, lang):
    schema, stats, kw = SHAPES[shape]
    with _InLanguage(lang):
        return build_hint(schema, stats, **kw)


class TheHintIsTranslated(unittest.TestCase):

    def test_every_shape_speaks_both_languages(self):
        for shape in SHAPES:
            with self.subTest(shape=shape):
                english = hint(shape, "en")
                french = hint(shape, "fr")
                self.assertTrue(french.strip(), shape)
                self.assertNotIn("hint.", french)
                # Line by line: these are multi-line messages, and one English
                # line among six translated ones still leaves them different.
                shared = ({line.strip() for line in english.splitlines() if line.strip()}
                          & {line.strip() for line in french.splitlines() if line.strip()})
                self.assertEqual(shared, set(), f"{shape} keeps English lines")

    def test_the_tenants_column_names_ride_through_untranslated(self):
        """A translated column name is a column name that matches nothing."""
        for lang in ("en", "fr"):
            with self.subTest(lang=lang):
                text = hint("mixed_no_prev", lang)
                self.assertIn("`REGION`", text)
                self.assertIn("`NET_AMOUNT`", text)
                self.assertIn("net amount", text)
                self.assertIn("region", text)

    def test_the_named_values_ride_through_untranslated(self):
        for lang in ("en", "fr"):
            with self.subTest(lang=lang):
                self.assertIn("North, South", hint("mixed_with_values", lang))

    def test_the_empty_schema_case_is_translated(self):
        self.assertEqual(hint("no_schema", "en"),
                         "Try asking a fresh question in the main chat.")
        self.assertEqual(
            hint("no_schema", "fr"),
            "Essayez de poser une nouvelle question dans la conversation principale.")


class TheEnglishIsUnchanged(unittest.TestCase):
    """A translation, not a rewrite — except the one branch that named a
    clinical domain in a module every tenant shares."""

    def test_the_summary_columns_shape_is_word_for_word(self):
        self.assertEqual(hint("summary_no_rows", "en"), (
            "The current result only has summary columns: `NET_AMOUNT`.\n"
            "Questions you can ask here:\n"
            "  • 'what is the average net amount'\n"
            "  • 'show rows where net amount is above average'\n"
            "  • 'rank by net amount'\n\n"
            "To see record-level details, ask a fresh question in the main chat."
        ))

    def test_the_mixed_shape_is_word_for_word(self):
        self.assertEqual(hint("mixed_with_values", "en"), (
            "The current result only has these columns: `REGION`, `NET_AMOUNT`.\n"
            "Questions you can ask:\n"
            "  • 'what is the average net amount'\n"
            "  • 'show rows where net amount is above average'\n"
            "  • 'rank by net amount'\n"
            "  • 'filter by region'\n\n"
            "For anything else, ask a fresh question in the main chat — "
            "for example, name them explicitly:\n"
            "  *'... for North, South'*"
        ))

    def test_the_mixed_shape_without_previous_rows_is_word_for_word(self):
        self.assertTrue(hint("mixed_no_prev", "en").endswith(
            "\n\nFor anything else, ask a fresh question in the main chat."))


class TheExamplesNoLongerNameSomeoneElsesDomain(unittest.TestCase):
    """core/result_renderer.py is shared by every tenant. The summary-total
    branch used to tell all of them to ask for prescriptions and patients."""

    CLINICAL = ("prescription", "patient")

    def test_no_shape_in_either_language_names_a_clinical_domain(self):
        for shape in SHAPES:
            for lang in ("en", "fr"):
                with self.subTest(shape=shape, lang=lang):
                    text = hint(shape, lang).lower()
                    for word in self.CLINICAL:
                        self.assertNotIn(word, text)

    def test_the_summary_total_still_says_what_to_do(self):
        english = hint("summary_single_row", "en")
        self.assertIn("There are no identifiers here to drill into.", english)
        self.assertIn("'List the records behind this total'", english)
        self.assertIn("'Break this total down by category'", english)
        self.assertIn("**91234**", english)

    def test_the_summary_total_says_it_in_french_too(self):
        french = hint("summary_single_row", "fr")
        self.assertIn("Ce résultat est un total agrégé", french)
        self.assertIn("'lister les enregistrements derrière ce total'", french)
        self.assertIn("'ventiler ce total par catégorie'", french)
        self.assertIn("**91234**", french)


class EverySuggestionIsAQuestionThePipelineCanRead(unittest.TestCase):
    """The point of this file.

    A suggestion is a route. core/question_normalizer.py is what turns a
    French question into the English the detectors read, and a French
    suggestion it cannot canonicalise fails exactly the way the question that
    produced this hint just did.
    """

    # Every example the hint can emit, and the English the normaliser must
    # produce from it. A new suggestion with no row here fails the
    # completeness test below rather than shipping uncanonicalisable.
    EXPECTED = {
        "quelle est la moyenne de net amount":
            "what is the average of net amount",
        "affiche les lignes avec net amount supérieur à la moyenne":
            "show the rows with net amount above average",
        "classer par net amount": "rank by net amount",
        "filtrer par region": "filter by region",
        "lister les enregistrements derrière ce total":
            "list the records behind this total",
        "ventiler ce total par catégorie":
            "break down this total by category",
    }

    # The one quoted fragment that is not a question: it is the tail of a
    # sentence showing how to name values ("... for North, South").
    NOT_A_QUESTION = "... "

    # An example is a whole line: "  • 'question'" or "  • *'question'*". A
    # bare `'([^']+)'` sweep would pair the apostrophe in "d'aller au détail"
    # with the next one and call the prose between them an example -- the same
    # trap core/question_normalizer.py's _QUOTED_RE documents.
    _EXAMPLE_LINE = re.compile(r"^\s*(?:•\s*)?\*?'([^'\n]+)'\*?\s*$", re.M)

    def _examples(self, lang):
        found = set()
        for shape in SHAPES:
            for quoted in self._EXAMPLE_LINE.findall(hint(shape, lang)):
                if not quoted.startswith(self.NOT_A_QUESTION):
                    found.add(quoted)
        return found

    def test_the_scan_finds_the_examples_at_all(self):
        """Without this, a scan that matched nothing would make the tests
        below pass on an empty set."""
        self.assertGreaterEqual(len(self._examples("en")), 6)

    def test_every_french_suggestion_canonicalises_to_english(self):
        for french, english in self.EXPECTED.items():
            with self.subTest(french=french):
                self.assertEqual(canonical_question(french, "fr"), english)

    def test_the_expected_table_covers_every_shipped_suggestion(self):
        """A suggestion added later without a canonicalisation check fails
        here rather than reaching a reader as a dead end."""
        self.assertEqual(self._examples("fr"), set(self.EXPECTED))

    def test_the_english_suggestions_are_left_alone(self):
        """canonical_question returns English untouched -- that is what makes
        it safe to call unconditionally on this path."""
        for example in self._examples("en"):
            with self.subTest(example=example):
                self.assertEqual(canonical_question(example, "en"), example)
                self.assertEqual(canonical_question(example, None), example)

    def test_no_french_suggestion_survives_canonicalisation_as_french(self):
        """The bar the EXPECTED table encodes, stated as a property: nothing
        the normaliser leaves behind may still be a French verb or article."""
        residue = ("affiche", "lister", "classer", "filtrer", "ventiler",
                   "les ", "des ", "avec ", "derriere", "cette", " ce ")
        for french in self.EXPECTED:
            canonical = canonical_question(french, "fr")
            for word in residue:
                with self.subTest(french=french, word=word.strip()):
                    self.assertNotIn(word, canonical)


if __name__ == "__main__":
    unittest.main()
