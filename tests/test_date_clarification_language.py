# -*- coding: utf-8 -*-
"""tests/test_date_clarification_language.py

A French reader was asked, in French, which date to use — and handed
"Confirmed Delivery Date", "Cancelled Order Date", "Invoice Date" and
"Order Date" to choose between.

Seen on a live workspace. Everything the TEMPLATE rendered was correct French:
"Précision demandée", "Rechercher par nom de date métier", "Par exemple : date
de facture", "Continuer", the keyboard hint. Two things came from the server
and had never entered the catalogue at all:

  - the question itself, three English literals inside the pipeline;
  - the role labels on the chips, the English constants in core.date_roles,
    for which no message id existed to translate *to*.

Every test executes the real translator or the real label builder.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.date_roles import DATE_ROLES, translated_label  # noqa: E402
from core.i18n import lookup, t  # noqa: E402
from core.query_pipeline import (  # noqa: E402
    date_clarification_question,
    date_option_label,
)

QUESTION_IDS = ("clar.date.several_defaults",
                "clar.date.no_default_free_text",
                "clar.date.several_valid")


class TheQuestionIsAsked(unittest.TestCase):

    def test_each_form_of_the_question_exists_in_both_languages(self):
        for msg_id in QUESTION_IDS:
            for lang in ("en", "fr"):
                text = t(msg_id, lang=lang)
                self.assertTrue(text and text != msg_id, f"{msg_id}/{lang}")

    def test_the_french_is_not_the_english(self):
        for msg_id in QUESTION_IDS:
            self.assertNotEqual(t(msg_id, lang="en"), t(msg_id, lang="fr"), msg_id)

    def test_the_french_reads_as_french(self):
        # Not merely "different": a template with an English body and one
        # accented word would pass that.
        french = t("clar.date.no_default_free_text", lang="fr")
        for word in ("dates métier", "utiliser", "nom"):
            self.assertIn(word, french)
        self.assertNotIn("business", french.lower())

    def test_the_free_text_form_still_tells_the_reader_they_may_type(self):
        for lang, phrase in (("en", "below"), ("fr", "ci-dessous")):
            self.assertIn(phrase, t("clar.date.no_default_free_text", lang=lang))


class TheRightQuestionIsChosen(unittest.TestCase):
    """Three situations read differently to the person answering, and the
    branch that picks between them was unreachable by any test."""

    def test_several_connected_defaults(self):
        self.assertEqual(
            date_clarification_question(ambiguous=True, allow_free_text=False),
            t("clar.date.several_defaults"))

    def test_ambiguity_wins_over_free_text(self):
        self.assertEqual(
            date_clarification_question(ambiguous=True, allow_free_text=True),
            t("clar.date.several_defaults"))

    def test_no_approved_default_invites_a_typed_answer(self):
        self.assertEqual(
            date_clarification_question(ambiguous=False, allow_free_text=True),
            t("clar.date.no_default_free_text"))

    def test_several_valid_dates_without_free_text(self):
        self.assertEqual(
            date_clarification_question(ambiguous=False, allow_free_text=False),
            t("clar.date.several_valid"))

    def test_each_situation_asks_something_different(self):
        asked = {date_clarification_question(ambiguous=a, allow_free_text=f)
                 for a, f in ((True, False), (False, True), (False, False))}
        self.assertEqual(len(asked), 3)

    def test_the_question_arrives_in_french(self):
        for ambiguous, free_text in ((True, False), (False, True), (False, False)):
            asked = date_clarification_question(
                ambiguous=ambiguous, allow_free_text=free_text, lang="fr")
            english = date_clarification_question(
                ambiguous=ambiguous, allow_free_text=free_text, lang="en")
            self.assertNotEqual(asked, english)
            self.assertIn("date", asked.lower())


class EveryRoleCanBeSpoken(unittest.TestCase):
    """Coverage over the DATA, not over the source: a role added tomorrow with
    no translation fails here rather than reaching a reader in English."""

    def test_every_shipped_role_has_a_message_id(self):
        missing = [r.key for r in DATE_ROLES
                   if lookup(f"date_role.{r.key}", "en") == f"date_role.{r.key}"]
        self.assertEqual(missing, [])

    def test_every_shipped_role_has_a_french_translation(self):
        untranslated = [
            r.key for r in DATE_ROLES
            if lookup(f"date_role.{r.key}", "fr")
            == lookup(f"date_role.{r.key}", "en")
        ]
        self.assertEqual(untranslated, [])

    def test_the_english_still_matches_the_constant(self):
        # The catalogue is now the source of the label a reader sees, so it
        # must not drift from the role it names.
        for role in DATE_ROLES:
            self.assertEqual(translated_label(role.key, "", lang="en"), role.label)

    def test_a_role_nobody_translated_falls_back_rather_than_vanishing(self):
        self.assertEqual(
            translated_label("some_custom_role", "Some Custom Role", lang="fr"),
            "Some Custom Role")

    def test_a_role_with_no_fallback_is_still_readable(self):
        self.assertEqual(translated_label("some_custom_role", "", lang="fr"),
                         "Some Custom Role")

    def test_an_empty_role_returns_the_fallback_untouched(self):
        self.assertEqual(translated_label("", "Whatever", lang="fr"), "Whatever")


class TheChipsCarryTheReadersLanguage(unittest.TestCase):

    ITEM = {"date_role": "confirmed_delivery_date",
            "context_name": "Confirmed Delivery Date"}

    def test_a_chip_is_translated(self):
        self.assertEqual(date_option_label(self.ITEM, lang="fr"),
                         "Date de livraison confirmée")

    def test_english_is_unchanged(self):
        self.assertEqual(date_option_label(self.ITEM, lang="en"),
                         "Confirmed Delivery Date")

    def test_a_disambiguated_label_survives_translation(self):
        # Two roles sharing a name are told apart by the table in the label.
        # Translating that away would make both chips read identically, which
        # is the one thing this card exists to prevent.
        self.assertEqual(
            date_option_label(self.ITEM, "Confirmed Delivery Date (Orders)",
                              lang="fr"),
            "Confirmed Delivery Date (Orders)")

    def test_a_role_the_catalogue_does_not_know_keeps_its_own_name(self):
        self.assertEqual(
            date_option_label({"context_name": "Widget Date"}, lang="fr"),
            "Widget Date")

    def test_an_empty_binding_still_produces_a_chip(self):
        self.assertEqual(date_option_label({}, lang="fr"), "Business date")

    def test_every_role_produces_a_distinct_french_chip(self):
        # A translation collision would offer the reader the same chip twice.
        chips = [date_option_label({"date_role": r.key, "context_name": r.label},
                                   lang="fr")
                 for r in DATE_ROLES]
        self.assertEqual(len(chips), len(set(chips)))


if __name__ == "__main__":
    unittest.main()
