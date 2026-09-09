"""
tests/test_remaining_literal_leaks.py

Five more surfaces that produced English prose for a French reader, found by
the second sweep after the first one closed eight of them.

  core/chart_spec.py            13 caption warnings, rendered in the slot
                                DIRECTLY BENEATH a chart whose tab labels, axis
                                titles, annotations and headline prose are all
                                translated.
  core/result_transforms.py     the outliers card, where the English literal
                                SHADOWED a French id that already existed:
                                webhooks sends `stats.get("detail") or
                                _t("reply.outliers.none")`, and the transform
                                supplies a detail for the ordinary "nothing was
                                unusual" outcome — so the French string was
                                only ever reached when the transform said
                                nothing at all.
  core/response_builder.py      the KPI card's caption, on the commonest
                                single-number answer in the product.
  core/result_commands.py       a clarification option's label, plus
                                calendar.month_name, which is English whatever
                                the reader's language.
  core/query_pipeline.py        the fallback question on the value-ambiguity
                                clarification card.

Every test executes the real producer.
"""
from __future__ import annotations

import unittest

from core import i18n


def _in(lang, fn, *args, **kw):
    token = i18n.activate_language(lang)
    try:
        return fn(*args, **kw)
    finally:
        i18n.deactivate_language(token)


class TestChartCaptionWarnings(unittest.TestCase):

    TECHNICAL = [{"CUST_ID": f"C{i}", "REVENUE_AMT": float(i + 1)}
                 for i in range(8)]
    LARGE = [{"SEG": f"s{i}", "REVENUE_AMT": float(i + 1)} for i in range(60)]

    def _warnings(self, rows, lang):
        from core.chart_spec import infer_chart_spec

        return _in(lang, infer_chart_spec, rows, "revenue by thing")["warnings"]

    def test_a_warning_is_produced_at_all(self):
        # The control for every "is not English" assertion below.
        self.assertTrue(self._warnings(self.TECHNICAL, "en"))
        self.assertTrue(self._warnings(self.LARGE, "en"))

    def test_a_french_reader_gets_french_captions(self):
        for rows, name in ((self.TECHNICAL, "technical id"), (self.LARGE, "large")):
            with self.subTest(case=name):
                french = " ".join(self._warnings(rows, "fr"))
                self.assertNotIn("looks like a technical identifier", french)
                self.assertNotIn("may be more readable", french)

    def test_and_they_differ_from_the_english_ones(self):
        for rows, name in ((self.TECHNICAL, "technical id"), (self.LARGE, "large")):
            with self.subTest(case=name):
                self.assertNotEqual(self._warnings(rows, "en"),
                                    self._warnings(rows, "fr"))

    def test_the_column_is_named_the_way_the_business_names_it(self):
        """The caption interpolates a column. It said CUST_ID."""
        for lang in ("en", "fr"):
            with self.subTest(lang=lang):
                text = " ".join(self._warnings(self.TECHNICAL, lang))
                self.assertNotIn("CUST_ID", text)
                self.assertIn("Cust Id", text)

    def test_every_warning_id_resolves_in_every_language(self):
        """A caption built from a missing id renders the id itself, which is
        worse than English."""
        for msg_id in (m for m in i18n.MESSAGES if m.startswith("ui.chart.warn.")):
            for lang in i18n.SUPPORTED_LANGUAGES:
                with self.subTest(msg_id=msg_id, lang=lang):
                    resolved = i18n.lookup(msg_id, lang)
                    self.assertTrue(resolved.strip())
                    self.assertNotEqual(resolved, msg_id)


class TestTheOutliersCard(unittest.TestCase):

    EVEN = [{"BAL_VAL_AMT": float(v)} for v in (10, 11, 12, 13, 14)]
    FLAT = [{"BAL_VAL_AMT": 5.0} for _ in range(4)]
    TINY = [{"BAL_VAL_AMT": 1.0}, {"BAL_VAL_AMT": 2.0}]

    def _detail(self, rows, lang):
        from core.result_transforms import filter_outliers

        return _in(lang, filter_outliers, rows, "BAL_VAL_AMT")[1].get("detail", "")

    def test_the_ordinary_outcome_is_translated(self):
        """"No outliers found" is the commonest press of that chip, and it is
        the branch that supplied a detail and so shadowed the French id."""
        french = self._detail(self.EVEN, "fr")
        self.assertIn("Aucune ligne", french)
        self.assertNotIn("No rows exceed", french)

    def test_so_are_the_other_two_details(self):
        for rows, name in ((self.FLAT, "zero variance"), (self.TINY, "too few")):
            with self.subTest(case=name):
                english = self._detail(rows, "en")
                french = self._detail(rows, "fr")
                self.assertTrue(english)
                self.assertNotEqual(english, french)

    def test_the_column_is_named_the_business_way(self):
        for lang in ("en", "fr"):
            with self.subTest(lang=lang):
                detail = self._detail(self.EVEN, lang)
                self.assertNotIn("BAL_VAL_AMT", detail)
                self.assertIn("Balance Value Amount", detail)

    def test_the_numbers_are_pointed_the_readers_way(self):
        """The threshold is 1.5 in English and 1,5 in French, in a sentence
        that already carries three other numbers."""
        self.assertIn("1.5", self._detail(self.EVEN, "en"))
        self.assertIn("1,5", self._detail(self.EVEN, "fr"))

    def test_a_real_outlier_still_comes_back(self):
        """The control: a transform that always failed would satisfy every
        assertion above."""
        from core.result_transforms import filter_outliers

        rows = [{"BAL_VAL_AMT": float(v)} for v in (1, 1, 1, 1, 1, 500)]
        found, stats = filter_outliers(rows, "BAL_VAL_AMT")
        self.assertTrue(stats["ok"])
        self.assertEqual([r["BAL_VAL_AMT"] for r in found], [500.0])


class TestTheKpiCaption(unittest.TestCase):

    def _note(self, rows, lang, **kw):
        from core.response_builder import _build_kpi_payload

        kw.setdefault("zero_match", None)
        kw.setdefault("null_issue", None)
        return _in(lang, _build_kpi_payload, rows, {}, {}, **kw)["note"]

    def test_all_three_notes_are_translated(self):
        cases = (
            ([{"R": None}], {"null_issue": {"column": "R"}}),
            ([{"R": None}], {}),
            ([{"R": 5.0}], {}),
        )
        for rows, kw in cases:
            with self.subTest(kw=kw, rows=rows):
                english = self._note(rows, "en", **kw)
                french = self._note(rows, "fr", **kw)
                self.assertTrue(english and french)
                self.assertNotEqual(english, french)

    def test_the_french_is_actually_french(self):
        self.assertIn("valeur unique", self._note([{"R": 5.0}], "fr"))


class TestTheClarificationOptionLabels(unittest.TestCase):
    """The label/wire invariant this module already keeps: the label is
    translated, the resolved_question is re-planned and stays English."""

    def test_a_row_option_label_is_translated_and_uses_a_business_name(self):
        import core.result_commands as rc

        english = _in("en", rc._t, "reply.rc.row_option", index=2,
                      column=rc._column_display_label("WHS_NM"))
        french = _in("fr", rc._t, "reply.rc.row_option", index=2,
                     column=rc._column_display_label("WHS_NM"))
        self.assertEqual(english, "Row 2 in Warehouse Name")
        self.assertEqual(french, "Ligne 2 dans Warehouse Name")

    MONTH_ROWS = [{"DMS_DT": "2026-04-15", "A": 1},
                  {"ORDER_DT": "2025-04-20", "A": 2},
                  {"DMS_DT": "2026-05-01", "A": 3}]

    def _month_options(self, lang):
        import core.result_commands as rc

        return _in(lang, rc._build_reference_clarification,
                   self.MONTH_ROWS, "April", action="show")

    def test_the_month_label_comes_from_the_catalogue_not_the_stdlib(self):
        """calendar.month_name is English whatever the reader's language.

        Driven through the real clarification builder rather than rebuilt in
        the test, so it fails if the label stops being assembled from the
        catalogue -- not merely if the catalogue's months change.
        """
        _, english = self._month_options("en")
        _, french = self._month_options("fr")
        self.assertEqual([o["label"] for o in english],
                         ["April 2025", "April 2026"])
        self.assertEqual([o["label"] for o in french],
                         ["avril 2025", "avril 2026"])

    def test_but_the_wire_value_stays_english(self):
        """core/dispatcher.py re-plans resolved_question, so a French label
        must not change what the pipeline is asked."""
        _, french = self._month_options("fr")
        for option in french:
            with self.subTest(option=option):
                self.assertIn("April", option["value"])
                self.assertIn("April", option["resolved_question"])
                self.assertNotIn("avril", option["resolved_question"])

    def test_the_question_above_the_options_is_translated_too(self):
        self.assertEqual(self._month_options("en")[0], "Which month did you mean?")
        self.assertEqual(self._month_options("fr")[0], "De quel mois s'agit-il ?")

    def test_the_value_ambiguity_prompt_is_translated(self):
        english = i18n.lookup("reply.value.which_one", "en")
        french = i18n.lookup("reply.value.which_one", "fr")
        self.assertEqual(english, "Which result value did you mean?")
        self.assertNotEqual(english, french)


if __name__ == "__main__":
    unittest.main()
