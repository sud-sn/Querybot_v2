"""
tests/test_followup_chip_language.py

The follow-up chips under every answer, in the reader's language, naming
columns the way the business names them, and still re-planning correctly.

core/stat_signals.template_suggestions was twelve English f-strings with no
access to the message catalogue, and it is TIER 1: core/insight.py returns
early as soon as three of them fire, which a plain six-row ranking does. So the
commonest result shape in the product produced three English chips carrying raw
warehouse column codes, under a heading and a kicker that were both translated.
The live output was:

    What makes the top 1 ORDER_CNT account for 80% of REVENUE_AMT?

Three defects in one string: English for a French reader, raw codes instead of
business names, and a COUNT column named as the entity being ranked -- because
the name heuristic had no "_cnt" suffix, so ORDER_CNT was classified as text.

Every test here executes the real generator.
"""
from __future__ import annotations

import unittest

from core import i18n
from core.stat_signals import _suggestion_pairs, compute_signals, template_suggestions


ROWS = [{"WHS_NM": w, "REVENUE_AMT": r, "ORDER_CNT": c} for w, r, c in
        [("Halifax", 800.0, 40), ("Toronto", 90.0, 9), ("Calgary", 60.0, 6),
         ("Regina", 30.0, 3), ("Ottawa", 15.0, 2), ("Laval", 5.0, 1)]]
COLS = ["REVENUE_AMT", "ORDER_CNT", "WHS_NM"]
TYPES = {"REVENUE_AMT": "numeric", "ORDER_CNT": "numeric", "WHS_NM": "text"}


def _pairs(lang, col_types=TYPES):
    token = i18n.activate_language(lang)
    try:
        return _suggestion_pairs(compute_signals(ROWS), COLS, col_types)
    finally:
        i18n.deactivate_language(token)


class TestTheChipsSpeakTheReadersLanguage(unittest.TestCase):

    def test_there_are_chips_at_all(self):
        # The control. Without it every assertion below is satisfied by a
        # generator that returns nothing.
        self.assertEqual(len(_pairs("en")), 3)

    def test_a_french_reader_gets_french_labels(self):
        labels = [p["label"] for p in _pairs("fr")]
        self.assertTrue(labels)
        for label in labels:
            with self.subTest(label=label):
                self.assertNotIn("What makes", label)
                self.assertNotIn("Which ", label)
                self.assertNotIn(" share in ", label)

    def test_and_they_differ_from_the_english_ones(self):
        """Guards against a catalogue entry whose French is a copy of its
        English, which is how a translation silently does nothing."""
        english = [p["label"] for p in _pairs("en")]
        french = [p["label"] for p in _pairs("fr")]
        self.assertEqual(len(english), len(french))
        for en, fr in zip(english, french):
            with self.subTest(en=en):
                self.assertNotEqual(en, fr)

    def test_an_english_reader_is_unaffected(self):
        for pair in _pairs("en"):
            with self.subTest(pair=pair):
                self.assertEqual(pair["label"], pair["question"])


class TestTheWireValueStaysEnglish(unittest.TestCase):
    """The label/wire split, which is what makes translating these safe.

    question_normalizer.canonicalise is a word-by-word lexicon, not a reverse
    lookup of the catalogue -- so sending French chip text to the planner would
    depend on every word of it happening to be in that lexicon. The chip
    carries the English question separately instead, and portal_chat.html puts
    it in data-question, which sendSuggestion already prefers over textContent.
    """

    def test_the_question_is_english_even_for_a_french_reader(self):
        french = _pairs("fr")
        self.assertEqual([p["question"] for p in french],
                         [p["question"] for p in _pairs("en")],
                         "the wire value changed with the reader's language")
        # And it is genuinely the English text, not merely stable: the label
        # beside it differs, which is what makes this a split rather than a
        # translation that did nothing.
        for pair in french:
            with self.subTest(pair=pair):
                self.assertNotEqual(pair["question"], pair["label"])

    def test_the_english_accessor_returns_the_wire_form(self):
        self.assertEqual(template_suggestions(compute_signals(ROWS), COLS, TYPES),
                         [p["question"] for p in _pairs("en")])


class TestTheChipsNameColumnsLikeTheBusiness(unittest.TestCase):

    def test_no_raw_column_code_survives_into_a_chip(self):
        for lang in ("en", "fr"):
            for pair in _pairs(lang):
                for surface in ("label", "question"):
                    with self.subTest(lang=lang, surface=surface, text=pair[surface]):
                        for raw in COLS:
                            self.assertNotIn(raw, pair[surface])

    def test_the_business_name_is_there_instead(self):
        """The other half: not merely absent, but replaced."""
        text = " ".join(p["label"] for p in _pairs("en"))
        self.assertIn("Revenue Amount", text)
        self.assertIn("Warehouse Name", text)


class TestTheEntityIsNotACountColumn(unittest.TestCase):
    """`text_cols[0]` decides what is being ranked. The name heuristic has no
    "_cnt" suffix, so ORDER_CNT was called text and the chip asked about the
    top 1 order count. compute_data_brief already knew it was numeric because
    it looked at the values; the types are threaded through now."""

    def test_with_types_the_entity_is_the_real_dimension(self):
        text = " ".join(p["label"] for p in _pairs("en"))
        self.assertIn("Warehouse Name", text)
        self.assertNotIn("Order Count", text)

    def test_the_name_heuristic_alone_gets_it_wrong(self):
        """Pins the reason the types are needed. If this ever passes, the
        heuristic has been taught the suffix and the threading is belt-and-
        braces rather than the fix."""
        from core.stat_signals import _looks_numeric_col

        self.assertFalse(_looks_numeric_col("ORDER_CNT"))

    def test_and_the_types_are_what_rescue_it(self):
        without = " ".join(p["label"] for p in _pairs("en", col_types=None))
        self.assertIn("Order Count", without)


class TestTheGeneratorHandsThemOn(unittest.TestCase):
    """End to end through the real coroutine, both boundaries mocked."""

    def _chips(self, lang):
        import asyncio
        from unittest.mock import AsyncMock, patch

        import core.compliance.policy_engine as policy_engine
        import core.insight as insight
        import core.llm as llm

        brief = insight.compute_data_brief(ROWS, "revenue by warehouse")
        model = AsyncMock(side_effect=AssertionError(
            "tier 1 supplied three chips; the model must not be called"))
        token = i18n.activate_language(lang)
        try:
            with patch.object(llm, "llm_complete", new=model), \
                 patch.object(llm, "resolve_provider",
                              return_value=("openai", "m", "k", {})), \
                 patch.object(policy_engine, "is_regulated", return_value=False):
                return asyncio.run(insight.generate_followup_suggestions(
                    brief=brief, question="revenue by warehouse",
                    result_scope={}, db_cfg={}, account_id="acct",
                    signals=compute_signals(ROWS)))
        finally:
            i18n.deactivate_language(token)

    def test_a_french_reader_gets_french_chips_from_the_real_entry_point(self):
        chips = self._chips("fr")
        self.assertEqual(len(chips), 3)
        for chip in chips:
            with self.subTest(chip=chip):
                self.assertNotEqual(chip["label"], chip["question"])
                self.assertNotIn("REVENUE_AMT", chip["label"])
                self.assertNotIn("REVENUE_AMT", chip["question"])

    def test_the_brief_supplies_the_types_so_the_entity_is_right(self):
        """The threading, proven from the real brief rather than a hand-built
        dict -- compute_data_brief is what production passes."""
        text = " ".join(c["label"] for c in self._chips("en"))
        self.assertIn("Warehouse Name", text)
        self.assertNotIn("Order Count", text)


if __name__ == "__main__":
    unittest.main()


class TestTheBrowserRendersTheLabelAndWiresTheQuestion(unittest.TestCase):
    """The other end of the split.

    sendSuggestion reads `btn.dataset.question || btn.textContent`, so the
    mechanism for a separate wire value already existed -- the chip markup just
    never set it. Both chip render sites do now, and both go through one
    helper, because there were two of them and only one would have been fixed.
    """

    PAGE = "portal/templates/portal_chat.html"

    def _run(self, entry):
        import json
        import sys

        import pytest

        dukpy = pytest.importorskip("dukpy")
        sys.path.insert(0, "tests")
        from js_lift import function as lift

        with open(self.PAGE, encoding="utf-8") as fh:
            js = lift(fh.read(), "function _followUpChip(entry)")
        return json.loads(dukpy.evaljs(
            js + "\nJSON.stringify(_followUpChip(%s))" % json.dumps(entry)))

    def test_a_pair_keeps_its_two_halves_apart(self):
        got = self._run({"question": "Which warehouses drive revenue?",
                         "label": "Quels entrepôts tirent le chiffre ?"})
        self.assertEqual(got["question"], "Which warehouses drive revenue?")
        self.assertEqual(got["label"], "Quels entrepôts tirent le chiffre ?")

    def test_a_plain_string_still_works(self):
        """The LLM tier and older payloads send strings; a helper that dropped
        them would blank every chip on those paths."""
        got = self._run("Show the top 5")
        self.assertEqual(got, {"question": "Show the top 5", "label": "Show the top 5"})

    def test_a_half_filled_entry_falls_back_rather_than_blanking(self):
        self.assertEqual(self._run({"label": "only a label"}),
                         {"question": "only a label", "label": "only a label"})
        self.assertEqual(self._run({"question": "only a question"}),
                         {"question": "only a question", "label": "only a question"})

    def test_nothing_at_all_is_empty_not_the_string_null(self):
        self.assertEqual(self._run(None), {"question": "", "label": ""})

    def test_both_render_sites_go_through_the_helper(self):
        """A source check, deliberately and narrowly.

        The two chip builders write into innerHTML from a live websocket frame
        and cannot be executed here. What can be pinned is that neither
        decides for itself: one definition, two call sites, and no remaining
        `.map(q => ...escHtml(q)...)` shape that would render a pair object as
        [object Object].
        """
        with open(self.PAGE, encoding="utf-8") as fh:
            page = fh.read()
        self.assertEqual(page.count("_followUpChip"), 3)   # 1 def + 2 call sites
        self.assertIn('data-question="${escHtml(c.question)}"', page)
