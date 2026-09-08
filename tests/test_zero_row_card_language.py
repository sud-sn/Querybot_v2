# -*- coding: utf-8 -*-
"""tests/test_zero_row_card_language.py

F1 · The whole zero-row answer card was English, in a card that was half French.

core/answer_rca.py had no reference to core.i18n at all. Every branch of
build_business_rca returned English literals for headline, most_likely_reason
and suggested_next_step -- and those three are the entire body of the card a
reader gets when their question returns nothing.

The artifact is what makes it plain. Built under an active French language, the
same card carried:

    Confidence: Confiance moyenne (65/100)          <- translated
    Why: La requête s'est exécutée correctement...  <- translated
    I could not find matching records...            <- not
    The filters produced no matching rows.          <- not

build_answer_confidence, sitting beside it in the same builder, had been
localised. The RCA producer had not, and nothing noticed because a failure card
is the one thing nobody demos.

A second consequence: because build_business_rca always returns a non-empty
string for all three keys, the translated fallbacks in
core/answer_formatter.py -- `rca.get(...) or _t("fail.zero_row.*")` -- can never
fire in production. They are kept, because two live tests pass rca={} and
removing the `or` turns "\\n".join([... None ...]) into a TypeError on an answer
path. Reusing fail.zero_row.headline for the four "could not find matching
records" branches is what stops that fallback being a stale English shadow of a
sentence produced somewhere else: it is now the same id.

The wire labels around the prose -- "Kind:", "Most likely reason:", "SQL
tried:" -- stay English on purpose. portal_chat.html PARSES them and renders its
own translated headings from ui.chat.diag.*, so they are protocol, not copy.
A test below pins that distinction so nobody "finishes the job" by translating
them and silently breaks the card's parsing.

Every test executes the real producer or the real card builder.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import i18n  # noqa: E402
from core.answer_rca import build_business_rca  # noqa: E402

# Every shape build_business_rca can return, by the argument that selects it.
SHAPES = {
    "validation": dict(validation_code="failed", row_count=0),
    "unmatched_closest": dict(
        validation_code="ok", row_count=0,
        unmatched_literals=[{"business_name": "Warehouse Name",
                             "column": "WHS_NM", "literal": "Acme",
                             "closest": ["Acme SA", "Acme Inc"]}]),
    "unmatched_plain": dict(
        validation_code="ok", row_count=0,
        unmatched_literals=[{"business_name": "Warehouse Name",
                             "column": "WHS_NM", "literal": "Acme"}]),
    "unmatched_unnamed": dict(
        validation_code="ok", row_count=0,
        unmatched_literals=[{"literal": "Acme"}]),
    "empty_table": dict(validation_code="ok", row_count=0,
                        empty_tables=["SALES.ORDERS"]),
    "join_path": dict(validation_code="ok", row_count=0,
                      tables_used=["A", "B"], graph_context={"enabled": True}),
    "filters": dict(validation_code="ok", row_count=0, tables_used=["A"]),
    "filters_mapped": dict(validation_code="ok", row_count=0,
                           tables_used=["A"], semantic_plan={"enabled": True}),
    "filters_joins": dict(validation_code="ok", row_count=0,
                          tables_used=["A", "B"]),
    "filters_joins_mapped": dict(validation_code="ok", row_count=0,
                                 tables_used=["A", "B"],
                                 semantic_plan={"enabled": True}),
    "success": dict(validation_code="ok", row_count=5),
}

READER_FACING = ("headline", "most_likely_reason", "suggested_next_step")


def rca(shape: str, lang: str) -> dict:
    token = i18n.activate_language(lang)
    try:
        return build_business_rca(**SHAPES[shape])
    finally:
        i18n.deactivate_language(token)


class TestEveryBranchSpeaksTheReadersLanguage(unittest.TestCase):

    def test_no_branch_answers_in_english_to_a_french_reader(self):
        # The whole finding in one assertion: every shape, every reader-facing
        # key, must differ between the two languages.
        for shape in SHAPES:
            english, french = rca(shape, "en"), rca(shape, "fr")
            for key in READER_FACING:
                with self.subTest(shape=shape, key=key):
                    self.assertNotEqual(
                        english[key], french[key],
                        f"{shape}.{key} is the same in both languages")

    def test_the_french_is_actually_french(self):
        # Guards against a "translation" that is the English string with a
        # catalogue id wrapped round it.
        for shape in SHAPES:
            french = rca(shape, "fr")
            with self.subTest(shape=shape):
                joined = " ".join(french[k] for k in READER_FACING)
                self.assertFalse(
                    any(word in joined for word in
                        ("could not", "the filters", "Check whether", "rows")),
                    joined)

    def test_the_english_is_byte_identical_to_what_it_replaced(self):
        # A translation change must not be a copy change. Five live tests in
        # other files assert on these exact sentences.
        english = rca("filters_joins", "en")
        self.assertEqual(
            english["most_likely_reason"],
            "The filters, joins, or selected schema produced no matching rows.")
        self.assertEqual(rca("join_path", "en")["most_likely_reason"],
                         "The selected join path did not produce matching "
                         "records for the current data.")
        self.assertEqual(rca("unmatched_plain", "en")["most_likely_reason"],
                         "There is no Warehouse Name matching 'Acme' in the data.")

    def test_the_em_dash_survived_being_moved_into_the_catalogue(self):
        # U+2014, not a hyphen. Retyping a literal into a catalogue entry is
        # exactly where a character like this is silently downgraded, and
        # nothing else in the repo would notice.
        self.assertIn(
            "—",
            i18n.MESSAGES["fail.zero_row.unmatched.next_step_closest"]["en"])
        self.assertIn(
            "—",
            i18n.MESSAGES["fail.zero_row.unmatched.next_step_closest"]["fr"])
        self.assertIn("—", rca("unmatched_closest", "en")["suggested_next_step"])


class TestTenantDataIsNeverTranslated(unittest.TestCase):

    def test_the_business_name_is_interpolated_not_looked_up(self):
        french = rca("unmatched_closest", "fr")
        self.assertIn("Warehouse Name", french["most_likely_reason"])
        self.assertIn("Acme", french["most_likely_reason"])

    def test_the_value_list_survives_verbatim(self):
        french = rca("unmatched_closest", "fr")
        self.assertIn("'Acme SA'", french["suggested_next_step"])
        self.assertIn("'Acme Inc'", french["suggested_next_step"])

    def test_a_table_name_survives_verbatim(self):
        self.assertIn("SALES.ORDERS", rca("empty_table", "fr")["most_likely_reason"])

    def test_an_unnamed_column_gets_its_own_sentence(self):
        # Not the English word "value" glued into a French sentence: the shape
        # without a name is a different sentence, with its own id.
        french = rca("unmatched_unnamed", "fr")
        self.assertNotIn("value", french["most_likely_reason"])
        self.assertIn("Acme", french["most_likely_reason"])

    def test_the_technical_notes_stay_in_the_machine_s_language(self):
        # They are diagnostics for whoever reads the card to debug it, and they
        # carry SQL codes and table names.
        notes = " ".join(rca("filters", "fr")["technical_notes"])
        self.assertIn("SQL validation:", notes)
        self.assertIn("Row count:", notes)


class TestTheWholeCard(unittest.TestCase):
    """The artifact, built by the real builder."""

    def card(self, lang: str) -> str:
        from core.pipeline_helpers import _build_zero_row_message

        token = i18n.activate_language(lang)
        try:
            return _build_zero_row_message("who bought most", "SELECT 1", {},
                                           "ok", 0)
        finally:
            i18n.deactivate_language(token)

    def test_the_french_card_has_no_english_prose_left(self):
        french = self.card("fr")
        for english in ("I could not find matching records",
                        "The filters produced no matching rows",
                        "Try broadening the filter"):
            with self.subTest(english=english):
                self.assertNotIn(english, french)

    def test_it_carries_the_french_prose(self):
        french = self.card("fr")
        self.assertIn("Je n'ai trouvé aucun enregistrement", french)
        self.assertIn("Les filtres n'ont produit aucune ligne", french)

    def test_the_wire_labels_stay_english_in_both(self):
        # portal_chat.html splits the card on these and renders its own
        # translated headings from ui.chat.diag.*. Translating them here would
        # not finish the job -- it would stop the card being parsed at all.
        for lang in ("en", "fr"):
            card = self.card(lang)
            for label in ("Kind:", "Confidence:", "Most likely reason:",
                          "Suggested next step:", "Why:", "Technical details:",
                          "SQL tried:"):
                with self.subTest(lang=lang, label=label):
                    self.assertIn(label, card)

    def test_the_kind_marker_still_names_the_card(self):
        # It is what lets the portal pick the right kicker without regex-
        # matching a headline that is now translated.
        self.assertIn("Kind: empty", self.card("fr"))

    def test_the_english_card_is_unchanged(self):
        english = self.card("en")
        self.assertIn("I could not find matching records for this question.",
                      english)
        self.assertIn("The filters produced no matching rows.", english)


class TestTheFallbacksAreKeptOnPurpose(unittest.TestCase):
    """core/answer_formatter.py's `rca.get(...) or _t(...)`.

    Unreachable in production -- build_business_rca always populates all three
    keys -- but two live tests pass rca={} deliberately, and without the `or`
    those become "\\n".join([... None ...]) -> TypeError on an answer path,
    which the catalogue's own rules forbid.
    """

    def test_an_empty_rca_still_renders_a_card(self):
        from core.answer_formatter import format_zero_row_business_response

        token = i18n.activate_language("fr")
        try:
            card = format_zero_row_business_response(
                confidence={}, rca={}, sql="SELECT 1",
                sql_preview_fn=lambda sql: sql)
        finally:
            i18n.deactivate_language(token)
        self.assertIn("Je n'ai trouvé aucun enregistrement", card)

    def test_the_fallback_and_the_producer_now_say_the_same_thing(self):
        # They were two English sentences maintained in two places. Reusing the
        # id is what stops them drifting.
        self.assertEqual(rca("filters", "en")["headline"],
                         i18n.MESSAGES["fail.zero_row.headline"]["en"])
        self.assertEqual(rca("filters", "fr")["headline"],
                         i18n.MESSAGES["fail.zero_row.headline"]["fr"])


if __name__ == "__main__":
    unittest.main()
