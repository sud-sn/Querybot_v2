# -*- coding: utf-8 -*-
"""tests/test_clarification_questions_language.py

F3, F4 · The clarification card asks its question in English.

Commit 7eb0246 (L3) translated the DATE clarification card — its question, its
chips, and the date-role labels on them. It fixed one card. Every other
clarification the product raises kept asking in English, inside the same
translated shell:

  core/analytical_intent.py   seven questions and every chip on them: which
                              subject, which measure to rank by, how to define
                              a business concept, which comparison window,
                              calendar or fiscal quarters, and which month the
                              fiscal year starts. The month chips read
                              "January".."December" to a French reader.
  core/query_pipeline.py      three more, in the file whose date questions L3
                              had just moved out: which source, which
                              relationship path, and which identifier counts
                              one business event.

The finding named five questions in the first file and two in the second. There
are seven and three: the subject question has two variants chosen by intent,
and the count question is a third sibling in the same function, with the same
shape, that the finding missed.

THE INVARIANT THAT MAKES THIS SAFE. An option's ``label`` is what the reader
sees, and is translated. Its ``value`` is what gets fed back into the planner
as their answer, and stays English: core/dispatcher.py resolves a typed reply
with ``match.get("value") or match.get("label")``, so an option carrying a
French label and no value would put French into the planner's input. Every
option built here carries a non-empty English value, and one test asserts that
for all of them at once rather than trusting it per site.

It also fixes something in the other direction, which is worth stating because
it is the reason the labels matter: a French reader typing "avril" could not
match a chip that said "April". Now they can.

The three query_pipeline questions are lifted to module level, exactly as L3
lifted the date ones, because a literal buried in _handle_query_impl — 6,484
lines needing an event, an adapter, a store and a live LLM before one line runs
— is a literal nothing can test. That is why these went unnoticed while the
card around them was translated.

Every test executes the real producer.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import i18n  # noqa: E402
from core.analytical_intent import plan_analytical_intent  # noqa: E402


def under(lang, fn, *args, **kw):
    token = i18n.activate_language(lang)
    try:
        return fn(*args, **kw)
    finally:
        i18n.deactivate_language(token)


# Every question analytical_intent can raise, by the input that raises it.
QUESTIONS = {
    "subject_snapshot": "what was my today's data",
    "subject_overview": "show me my data",
    "metric_ranking": "rank them",
    "business_definition": "how many churned customers",
    "recent_window": "how many orders decreased recently",
    "calendar_basis": "revenue in Q1",
}


def clarification(case: str, lang: str):
    plan = under(lang, plan_analytical_intent, QUESTIONS[case])
    return getattr(plan, "clarification", None)


class TestTheAnalyticalClarifications(unittest.TestCase):

    def test_every_question_is_asked_in_the_readers_language(self):
        for case in QUESTIONS:
            english, french = clarification(case, "en"), clarification(case, "fr")
            with self.subTest(case=case):
                self.assertIsNotNone(english, f"{case} raised no clarification")
                self.assertNotEqual(english.question, french.question)

    def test_the_french_is_actually_french(self):
        for case in QUESTIONS:
            french = clarification(case, "fr")
            with self.subTest(case=case):
                self.assertFalse(
                    any(word in french.question for word in
                        ("Which", "What", "How should", "Should I interpret")),
                    french.question)

    def test_both_variants_of_the_subject_question_differ(self):
        # The finding counted six constructions and five questions. This site
        # builds two, chosen by intent, and both are live.
        snapshot = clarification("subject_snapshot", "fr").question
        overview = clarification("subject_overview", "fr").question
        self.assertNotEqual(snapshot, overview)
        self.assertIn("instantané", snapshot)

    def test_the_readers_own_words_are_interpolated_not_translated(self):
        french = clarification("business_definition", "fr").question
        self.assertIn("churned", french)

    def test_the_quarter_the_reader_named_survives(self):
        self.assertIn("Q1", clarification("calendar_basis", "fr").question)


class TestTheChipsOnThoseCards(unittest.TestCase):

    def labels(self, case, lang):
        found = clarification(case, lang)
        return [o["label"] for o in (found.options or ())]

    def test_the_comparison_windows_are_translated(self):
        french = self.labels("recent_window", "fr")
        self.assertTrue(french)
        for label in french:
            with self.subTest(label=label):
                self.assertNotIn("previous", label)
                self.assertNotIn("Last", label)

    def test_the_number_of_days_survives_translation(self):
        french = self.labels("recent_window", "fr")
        self.assertTrue(any("7" in lb for lb in french), french)
        self.assertTrue(any("90" in lb for lb in french), french)

    def test_calendar_and_fiscal_are_translated(self):
        french = self.labels("calendar_basis", "fr")
        self.assertEqual(french, ["Trimestres civils", "Trimestres fiscaux"])

    def test_the_fiscal_months_are_the_readers_months(self):
        from core.analytical_intent import _fiscal_month_options

        french = [o["label"] for o in under("fr", _fiscal_month_options)]
        self.assertEqual(french[:3], ["janvier", "février", "mars"])
        english = [o["label"] for o in under("en", _fiscal_month_options)]
        self.assertEqual(english[:3], ["January", "February", "March"])


class TestTheValueStaysEnglish(unittest.TestCase):
    """The invariant, asserted once for every option the module can build.

    core/dispatcher.py resolves a typed reply with
    `match.get("value") or match.get("label")`. An option with a translated
    label and no value would feed French into the planner's input.
    """

    def all_options(self, lang):
        from core.analytical_intent import _fiscal_month_options

        options = list(under(lang, _fiscal_month_options))
        for case in QUESTIONS:
            found = clarification(case, lang)
            options.extend(found.options or ())
        return options

    def test_every_option_carries_a_value(self):
        for lang in ("en", "fr"):
            for option in self.all_options(lang):
                with self.subTest(lang=lang, option=option.get("id")):
                    self.assertTrue(option.get("value"), option)

    def test_the_value_is_the_same_in_both_languages(self):
        english = {o["id"]: o["value"] for o in self.all_options("en")}
        french = {o["id"]: o["value"] for o in self.all_options("fr")}
        self.assertEqual(english, french)

    def test_a_french_reader_can_type_a_french_month(self):
        # The other direction, and the reason the labels matter: before this,
        # a French reader was shown "April" and could type nothing that matched
        # what they were reading.
        from core.analytical_intent import _fiscal_month_options
        from core.dispatcher import resolve_option_text

        options = list(under("fr", _fiscal_month_options))
        matched = under("fr", resolve_option_text, options, "avril")
        self.assertIsNotNone(matched)
        self.assertEqual(matched["value"], "The fiscal year starts in April")


class TestThePipelineClarifications(unittest.TestCase):
    """F4 · three more, in the file whose date questions L3 moved out."""

    def build(self, name, lang, *args):
        import core.query_pipeline as pipeline

        return under(lang, getattr(pipeline, name), *args)

    def test_the_source_question_is_translated(self):
        english = self.build("source_clarification_question", "en")
        french = self.build("source_clarification_question", "fr")
        self.assertNotEqual(english, french)
        self.assertIn("jeux de données", french)

    def test_the_join_question_is_translated(self):
        english = self.build("join_clarification_question", "en")
        french = self.build("join_clarification_question", "fr")
        self.assertNotEqual(english, french)
        self.assertIn("relation", french)

    def test_the_count_question_has_both_of_its_shapes(self):
        # The third sibling, in the same function, that the finding missed.
        one = self.build("count_clarification_question", "fr", "customers", 1)
        several = self.build("count_clarification_question", "fr", "customers", 3)
        self.assertNotEqual(one, several)
        for text in (one, several):
            with self.subTest(text=text[:40]):
                self.assertNotIn("I found", text)
                self.assertIn("customers", text)

    def test_the_english_is_unchanged(self):
        # A translation change must not be a copy change.
        self.assertEqual(
            self.build("source_clarification_question", "en"),
            "I found more than one relevant business dataset. "
            "Which source should I use for this analysis?")
        self.assertEqual(
            self.build("join_clarification_question", "en"),
            "I found more than one equally governed relationship path for this "
            "analysis. Which business relationship should I use?")

    def test_they_are_reachable_without_running_the_pipeline(self):
        # The point of lifting them out. A literal inside _handle_query_impl
        # needs an event, an adapter, a store and a live LLM before one line of
        # it runs, which is why these went unnoticed while the card around them
        # was translated.
        import core.query_pipeline as pipeline

        for name in ("source_clarification_question",
                     "join_clarification_question",
                     "count_clarification_question"):
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(pipeline, name)))


if __name__ == "__main__":
    unittest.main()
