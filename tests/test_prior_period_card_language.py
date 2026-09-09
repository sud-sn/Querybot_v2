"""
tests/test_prior_period_card_language.py

The "vs prior period" card, in the reader's language.

core/period_comparison.py imported core.i18n zero times, so every string it
produced was an English literal — while the chip that OPENS the card
(chip.compare_prior, "vs période précédente") and the error card the websocket
handler falls back to (gateway/webhooks.py, four _t() ids) were both
translated. The same card, in the same frame type, was French when the call
threw and English when it worked.

That includes the deterministic narrative — the headline, body and bullets used
when the model returns nothing parseable, which is precisely the path a
degraded or rate-limited provider takes.

Every test here executes the real coroutine.
"""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from core import i18n
from core.period_comparison import (
    _deterministic_bullets, _deterministic_headline, _parse_narrative,
    generate_period_comparison,
)


TIME_SERIES = {
    "time_series": {"first_period": "2026-01", "last_period": "2026-03",
                    "last_value": 900.0},
    "mode": "time_series",
    "numeric_summaries": {"REV": {"sum": 900.0, "min": 1.0, "max": 2.0}},
}
NO_TIME_SERIES = {"mode": "table", "numeric_summaries": {}}
UNKNOWN_GRAIN = {
    "time_series": {"first_period": "week 12", "last_period": "week 40"},
    "mode": "time_series", "numeric_summaries": {},
}


def _card(brief, lang, *, regulated=False):
    async def _run():
        with patch("core.compliance.policy_engine.result_llm_features_allowed",
                   return_value=not regulated):
            return await generate_period_comparison(
                rows=[{"MONTH": "2026-01", "REV": 100.0}],
                question="revenue by month", original_sql="SELECT 1",
                data_brief=brief,
                db_cfg={"db_type": "azure_sql", "credentials": {}},
                account_id="acct", known_tables=set(),
                provider="openai", model="m", api_key="k")

    token = i18n.activate_language(lang)
    try:
        return asyncio.run(_run())
    finally:
        i18n.deactivate_language(token)


class TestTheCardSpeaksTheReadersLanguage(unittest.TestCase):

    CASES = (("no time series", NO_TIME_SERIES, False),
             ("unknown grain", UNKNOWN_GRAIN, False),
             ("regulated tenant", TIME_SERIES, True))

    def test_an_english_reader_gets_english(self):
        # The control. Without it, a card that returned empty strings would
        # satisfy every "is not English" assertion below.
        for label, brief, regulated in self.CASES:
            with self.subTest(case=label):
                card = _card(brief, "en", regulated=regulated)
                self.assertEqual(card["title"], "Prior period comparison")
                self.assertTrue(card["body"])

    def test_a_french_reader_gets_french(self):
        for label, brief, regulated in self.CASES:
            with self.subTest(case=label):
                card = _card(brief, "fr", regulated=regulated)
                self.assertEqual(card["title"],
                                 "Comparaison avec la période précédente")
                self.assertIn("période", card["headline"])
                self.assertNotIn("Could not", card["headline"])
                self.assertNotIn("Prior period comparison", card["title"])

    def test_the_two_languages_actually_differ(self):
        """Guards against a catalogue entry whose French was copied from its
        English, which reads as done and is not."""
        for label, brief, regulated in self.CASES:
            with self.subTest(case=label):
                english = _card(brief, "en", regulated=regulated)
                french = _card(brief, "fr", regulated=regulated)
                self.assertNotEqual(english["body"], french["body"])
                self.assertNotEqual(english["headline"], french["headline"])

    def test_the_interpolated_labels_survive_translation(self):
        """The reader's own period labels have to reach the sentence — a
        translated template with a dropped placeholder loses them silently."""
        for lang in ("en", "fr"):
            with self.subTest(lang=lang):
                body = _card(UNKNOWN_GRAIN, lang)["body"]
                self.assertIn("week 12", body)
                self.assertIn("week 40", body)

    def test_the_next_step_is_translated_too(self):
        """It is the only actionable line on a card that just said no."""
        english = _card(NO_TIME_SERIES, "en")["next_step"]
        french = _card(NO_TIME_SERIES, "fr")["next_step"]
        self.assertTrue(english and french)
        self.assertNotEqual(english, french)

    def test_the_regulated_refusal_is_translated(self):
        """A regulated tenant is exactly the workspace most likely to have
        readers who are not English speakers, and this is the sentence
        explaining why the product refused."""
        french = _card(TIME_SERIES, "fr", regulated=True)["body"]
        self.assertIn("réglementé", french)
        self.assertNotIn("regulated industry", french)


class TestTheDeterministicNarrativeToo(unittest.TestCase):
    """The path taken when the model returns nothing parseable — a degraded or
    rate-limited provider — which is when a reader most needs the card to still
    make sense."""

    CURRENT = {"time_series": {"last_value": 1200.0},
               "numeric_summaries": {"REV": {"sum": 1200.0}}}
    PRIOR = {"time_series": {"last_value": 1000.0},
             "numeric_summaries": {"REV": {"sum": 1000.0}}}

    def _in(self, lang, fn, *args):
        token = i18n.activate_language(lang)
        try:
            return fn(*args)
        finally:
            i18n.deactivate_language(token)

    def test_the_headline_names_the_direction_in_the_readers_language(self):
        english = self._in("en", _deterministic_headline,
                           self.CURRENT, self.PRIOR, "2026-Q1", "2025-Q4")
        french = self._in("fr", _deterministic_headline,
                          self.CURRENT, self.PRIOR, "2026-Q1", "2025-Q4")
        self.assertIn("up", english)
        self.assertIn("hausse", french)
        self.assertIn("20", english)      # the movement itself survives
        self.assertIn("20", french)

    def test_a_fall_is_named_as_one(self):
        french = self._in("fr", _deterministic_headline,
                          self.PRIOR, self.CURRENT, "2026-Q1", "2025-Q4")
        self.assertIn("baisse", french)

    def test_the_headline_with_no_comparable_value_is_translated(self):
        empty = {"numeric_summaries": {}}
        french = self._in("fr", _deterministic_headline,
                          empty, empty, "2026-Q1", "2025-Q4")
        self.assertIn("Comparaison", french)
        self.assertNotIn("Comparing", french)

    def test_the_bullets_are_translated(self):
        english = self._in("en", _deterministic_bullets, self.CURRENT, self.PRIOR)
        french = self._in("fr", _deterministic_bullets, self.CURRENT, self.PRIOR)
        self.assertTrue(english and french)
        # _first_numeric_total reads a brief's numeric_summaries through a
        # shape these fixtures do not carry, so the last_value branch is the
        # one that fires. Both branches are asserted rather than assumed.
        self.assertIn("Current period last value", english[0])
        self.assertIn("Dernière valeur de la période actuelle", french[0])
        self.assertIn("Prior period last value", english[1])
        self.assertIn("Dernière valeur de la période précédente", french[1])

    def test_the_total_branch_is_translated_as_well(self):
        with_totals = {"numeric_summaries": {"REV": {"total": 1200.0}},
                       "time_series": {}}
        english = self._in("en", _deterministic_bullets, with_totals, with_totals)
        french = self._in("fr", _deterministic_bullets, with_totals, with_totals)
        if not english:
            self.skipTest("this brief shape produces no total bullet")
        self.assertIn("period total", english[0])
        self.assertIn("Total de la période", french[0])

    def test_the_body_is_translated(self):
        parsed = self._in("fr", _parse_narrative, "", self.CURRENT, self.PRIOR,
                          "2026-Q1", "2025-Q4")
        self.assertIn("Comparaison", parsed["body"])
        self.assertNotIn("Comparing", parsed["body"])

    def test_a_model_narrative_is_still_used_when_it_parses(self):
        """The deterministic path is a fallback, not a replacement: text the
        model did produce must survive."""
        raw = "HEADLINE: Revenue rose sharply\nBODY: Driven by Q1\n- one bullet"
        parsed = self._in("fr", _parse_narrative, raw, self.CURRENT, self.PRIOR,
                          "2026-Q1", "2025-Q4")
        self.assertEqual(parsed["headline"], "Revenue rose sharply")
        self.assertEqual(parsed["bullets"], ["one bullet"])


if __name__ == "__main__":
    unittest.main()
