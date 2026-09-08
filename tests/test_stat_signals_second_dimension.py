# -*- coding: utf-8 -*-
"""tests/test_stat_signals_second_dimension.py

F19 · The fifth site of the two-dimensional defect, and the last one.

`compute_signals` is not decoration. Its output feeds `template_suggestions`,
which writes the follow-up chips the reader clicks, and `format_signals_for_llm`,
which grounds the model's other suggestions. A false signal here does not sit
quietly in a dict: it becomes an invitation to investigate something that did
not happen.

On "compare revenue by warehouse for the last three months" -- a row per
warehouse per month -- against three warehouses that are perfectly flat across
the quarter, the shipped code reported:

    downward trend in REVENUE_AMT (25% change first→last)

That is Halifax in March (1,200,000) measured against Vancouver in June
(900,000): two different warehouses, a quarter apart, in a quarter where
nothing moved. It read rows[0] and rows[-1] with no period ordering and no
check that a period appears once -- the L1 defect, untouched.

The same function carried the C3 defect beside it. `group_imbalance` divided
one row by a total summed over every period, so a genuine 37.5% leader came out
at 9.4%, below the 0.35 threshold, and the signal vanished rather than merely
understating.

Neither is fixed here with a sixth private implementation. The rules already
exist: core.analysis_evidence._values_in_period_order orders the series and
refuses when a period repeats, core.analysis_contract.collapse_rows_by_label
merges to one row per label and refuses when the measure may not be summed, and
core.response_builder._narrative_label_column picks the dimension the reader
asked about. This module now uses all three, so the three detectors that run on
the same rows stop disagreeing with each other.

Every test executes compute_signals.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.stat_signals import compute_signals  # noqa: E402

PERIODS = ("2026-03", "2026-04", "2026-05", "2026-06")
WAREHOUSES = (("Halifax", 1_200_000.0), ("Calgary", 1_100_000.0),
              ("Vancouver", 900_000.0))


def grid(measure: str = "REVENUE_AMT") -> list[dict]:
    """A row per warehouse per period, every warehouse flat across the quarter."""
    return [{"PERIOD": period, "WHS_NM": name, measure: value}
            for period in PERIODS for name, value in WAREHOUSES]


def flat(measure: str = "REVENUE_AMT") -> list[dict]:
    """The same totals, already one row per warehouse."""
    return [{"WHS_NM": name, measure: value * len(PERIODS)}
            for name, value in WAREHOUSES]


def kinds(rows) -> list[str]:
    return [s["type"] for s in compute_signals(rows)]


def one(rows, kind) -> dict | None:
    return next((s for s in compute_signals(rows) if s["type"] == kind), None)


class TestNoTrendIsInventedAcrossTheGrid(unittest.TestCase):

    def test_a_flat_quarter_produces_no_trend_signal(self):
        self.assertNotIn("temporal", kinds(grid()))

    def test_it_does_not_call_it_flat_either(self):
        # "Broadly flat" would be the right answer by accident. The rows do not
        # form a series at all, and saying anything about its direction --
        # including that it has none -- is a claim about a thing that is not
        # there.
        self.assertNotIn("flat_trend", kinds(grid()))

    def test_a_rising_grid_gets_no_trend_either(self):
        # Refusing is not "detect it anyway when the totals happen to agree".
        # This detector cannot know the measure is additive, so a caller that
        # wants the total's trend has to aggregate before asking.
        rows = [{"PERIOD": p, "WHS_NM": w, "REVENUE_AMT": v * mult}
                for mult, p in enumerate(PERIODS, start=1)
                for w, v in WAREHOUSES]
        self.assertNotIn("temporal", kinds(rows))

    def test_a_genuine_series_still_trends(self):
        # Guards the guard: a fix that refused everything would pass the three
        # tests above.
        rows = [{"PERIOD": p, "REVENUE_AMT": v}
                for p, v in zip(PERIODS, (100.0, 150.0, 200.0, 300.0))]
        signal = one(rows, "temporal")
        self.assertIsNotNone(signal)
        self.assertEqual(signal["direction"], "upward")
        self.assertEqual(signal["value"], 200.0)

    def test_the_series_is_read_in_period_order_not_arrival_order(self):
        # A "top N months by revenue" result carries a period column and is
        # sorted by revenue.
        rows = [{"PERIOD": p, "REVENUE_AMT": v}
                for p, v in zip(PERIODS, (100.0, 150.0, 200.0, 300.0))]
        by_measure = sorted(rows, key=lambda r: -r["REVENUE_AMT"])
        self.assertEqual(one(by_measure, "temporal")["direction"], "upward")

    def test_one_usable_point_is_not_a_flat_trend(self):
        # Two rows, one of which reports nothing. The ordered series then holds
        # a single value, first and last are the same number, and the change
        # comes out at 0% -- "broadly flat", asserted from one point. A
        # direction needs two; one has none, not a neutral one.
        #
        # (compute_signals already returns early below two ROWS, so this is the
        # shape that actually reaches the guard.)
        rows = [{"PERIOD": "2026-03", "REVENUE_AMT": 100.0},
                {"PERIOD": "2026-04", "REVENUE_AMT": None}]
        self.assertNotIn("flat_trend", kinds(rows))
        self.assertNotIn("temporal", kinds(rows))

    def test_a_flat_series_is_still_reported_flat(self):
        rows = [{"PERIOD": p, "REVENUE_AMT": v}
                for p, v in zip(PERIODS, (100.0, 100.2, 100.1, 100.3))]
        self.assertIn("flat_trend", kinds(rows))


class TestTheLeaderIsTheOneTheReaderAskedAbout(unittest.TestCase):

    def test_the_share_is_the_real_share(self):
        # 4.8M of 12.8M. Divided per row it read 9.4% -- under the threshold,
        # so the signal did not appear at all.
        signal = one(grid(), "group_imbalance")
        self.assertIsNotNone(signal, "the imbalance signal disappeared")
        self.assertEqual(signal["value"], 37.5)
        self.assertEqual(signal["leader"], "Halifax")

    def test_the_grouped_and_the_flat_form_agree(self):
        grouped = one(grid(), "group_imbalance")
        already = one(flat(), "group_imbalance")
        for key in ("value", "leader", "col"):
            self.assertEqual(grouped[key], already[key], key)

    def test_it_describes_the_warehouse_not_the_calendar(self):
        # The first text column is the period. Describing the result along that
        # axis reports the imbalance BETWEEN MONTHS -- four periods at 25%
        # each, no leader -- for a question about warehouses.
        self.assertEqual(one(grid(), "group_imbalance")["col"], "WHS_NM")

    def test_a_percentage_gets_no_share_of_a_total_nobody_can_compute(self):
        self.assertNotIn("group_imbalance", kinds(grid("GRS_MARG_PCT")))

    def test_an_even_spread_still_reports_no_leader(self):
        # The threshold is not withdrawn: this must not start reporting leaders
        # it exists to suppress.
        rows = [{"WHS_NM": f"W{n}", "REVENUE_AMT": 100.0} for n in range(10)]
        self.assertNotIn("group_imbalance", kinds(rows))

    def test_a_single_dimension_result_is_untouched(self):
        signal = one(flat(), "group_imbalance")
        self.assertEqual((signal["value"], signal["leader"]), (37.5, "Halifax"))


class TestAFrenchPeriodColumnCanStillTrend(unittest.TestCase):
    """Found by routing this module through the shared ordering rule.

    core.stat_signals._is_temporal_col has recognised French month labels since
    it was written -- the comment beside the pattern says why -- and
    core.analysis_evidence.period_order_key never did. So a French tenant's
    period column was correctly identified as temporal and then, one step
    later, declared to have "no known ordering". Every trend on French-labelled
    periods had been silently refused since b852062, in the detector, in the
    evidence engine, and in everything downstream.

    Refusing what cannot be ordered is right. It was the wrong answer here, and
    it was invisible because a refusal looks exactly like having nothing to
    say.
    """

    FRENCH = [("janvier", 100.0), ("février", 140.0),
              ("mars", 190.0), ("avril", 260.0)]

    def test_a_french_series_reports_its_direction(self):
        rows = [{"MOIS": m, "CHIFFRE": v} for m, v in self.FRENCH]
        signal = one(rows, "temporal")
        self.assertIsNotNone(signal, "the French series produced no trend")
        self.assertEqual(signal["direction"], "upward")

    def test_it_is_ordered_by_the_calendar_not_by_arrival(self):
        rows = [{"MOIS": m, "CHIFFRE": v}
                for m, v in sorted(self.FRENCH, key=lambda p: -p[1])]
        self.assertEqual(one(rows, "temporal")["direction"], "upward")

    def test_unaccented_spellings_order_the_same_way(self):
        # A warehouse export is as likely to hold "fevrier" as "février".
        rows = [{"MOIS": m, "CHIFFRE": v} for m, v in
                [("janvier", 100.0), ("fevrier", 140.0),
                 ("aout", 190.0), ("decembre", 260.0)]]
        self.assertEqual(one(rows, "temporal")["direction"], "upward")

    def test_june_and_july_are_told_apart(self):
        # Both begin "jui". A three-letter prefix reads them as one month and
        # the series folds in on itself.
        from core.analysis_evidence import period_order_key

        self.assertEqual(period_order_key("juin"), (0, 6, 0))
        self.assertEqual(period_order_key("juillet"), (0, 7, 0))

    def test_a_french_place_name_is_still_not_a_month(self):
        from core.analysis_evidence import period_order_key

        for place in ("Marseille", "Marne", "Juilly", "Mayenne", "Aoste"):
            with self.subTest(place=place):
                self.assertIsNone(period_order_key(place))

    def test_a_french_month_with_a_year_orders_too(self):
        from core.analysis_evidence import period_order_key

        self.assertEqual(period_order_key("janvier 2024"), (2024, 1, 0))
        self.assertEqual(period_order_key("2024 mars"), (2024, 3, 0))
        # Accented, which the ASCII character class the pattern used to carry
        # could not match at all.
        self.assertEqual(period_order_key("février 2024"), (2024, 2, 0))
        self.assertEqual(period_order_key("août 2024"), (2024, 8, 0))
        self.assertEqual(period_order_key("2024 décembre"), (2024, 12, 0))


class TestTheGridIsStillAGridWhenItIsSparse(unittest.TestCase):
    """The two holes the first fix left, both reproduced at HEAD before this.

    An adversarial review of the earlier commit rebuilt the finding's exact
    sentence twice over, in English and in French, with that commit in place.
    It narrowed the defect; it did not close it. Two causes, and neither is in
    the detector that was fixed:

    1. _values_in_period_order counted periods only over the rows that REPORT a
       measure. A grid with one non-null cell per period collapses to one pair
       per period and reads as a clean series -- so a quarter of alternating
       warehouses passed the repeat check and produced "upward trend, 300%"
       between two different warehouses three quarters apart. Whether a result
       is one-dimensional is a property of its rows, not of which of them
       happen to be populated.

    2. core.response_builder._looks_temporal -- a fourth private temporal
       classifier -- knew English month names and nothing else. No quarters at
       all, no French. _narrative_label_column asks it whether a repeating
       column is a calendar to skip; answer "no" and the repeating calendar
       becomes the dimension the whole narrative is written about.

    The first fix taught period_order_key French and then leaned on a helper
    that had the same hole.
    """

    QUARTERS = ("Q1 2026", "Q2 2026", "Q3 2026", "Q4 2026")
    MONTHS_FR = ("janvier", "février", "mars", "avril", "mai", "juin")

    def sparse_quarters(self):
        filled = {("Q1 2026", "Halifax"): 100.0, ("Q2 2026", "Calgary"): 200.0,
                  ("Q3 2026", "Halifax"): 300.0, ("Q4 2026", "Calgary"): 400.0}
        return [{"FISCAL_QTR": q, "WHS_NM": w, "REVENUE_AMT": filled.get((q, w))}
                for q in self.QUARTERS for w in ("Halifax", "Calgary")]

    def sparse_months_fr(self):
        filled = {(m, "Halifax" if i % 2 == 0 else "Calgary"): 100.0 * (i + 1)
                  for i, m in enumerate(self.MONTHS_FR)}
        return [{"MOIS": m, "ENTREPOT": w, "CHIFFRE": filled.get((m, w))}
                for m in self.MONTHS_FR for w in ("Halifax", "Calgary")]

    def test_a_sparse_quarter_grid_reports_no_trend(self):
        self.assertNotIn("temporal", kinds(self.sparse_quarters()))

    def test_a_sparse_french_month_grid_reports_no_trend(self):
        self.assertNotIn("temporal", kinds(self.sparse_months_fr()))

    def test_the_quarter_grid_is_described_by_its_warehouse(self):
        # The other half: with the calendar unrecognised, the imbalance signal
        # described the QUARTERS for a question about warehouses.
        signal = one(self.sparse_quarters(), "group_imbalance")
        if signal is not None:
            self.assertEqual(signal["col"], "WHS_NM")

    def test_the_shared_rule_refuses_the_sparse_grid_on_its_own(self):
        # Asserted at the rule itself, not only through compute_signals. With
        # _looks_temporal now recognising quarters, the label-column choice
        # keeps the sparse grid away from the trend branch -- so testing only
        # the signal would leave this fix protected by the other one and
        # untested by either. trend_findings is the public entry point that
        # reaches _values_in_period_order directly.
        from core.analysis_evidence import trend_findings

        self.assertEqual(
            trend_findings(self.sparse_quarters(), "REVENUE_AMT", "FISCAL_QTR"),
            [])

    def test_and_still_accepts_a_series_that_has_a_gap(self):
        from core.analysis_evidence import trend_findings

        rows = [{"FISCAL_QTR": q, "REVENUE_AMT": v}
                for q, v in zip(self.QUARTERS, (100.0, None, 300.0, 400.0))]
        found = trend_findings(rows, "REVENUE_AMT", "FISCAL_QTR")
        self.assertEqual([f.kind for f in found][:1], ["trend_up"])

    def test_a_genuine_quarterly_series_still_trends(self):
        rows = [{"FISCAL_QTR": q, "REVENUE_AMT": v}
                for q, v in zip(self.QUARTERS, (100.0, 200.0, 300.0, 400.0))]
        self.assertEqual(one(rows, "temporal")["direction"], "upward")

    def test_a_genuine_french_series_still_trends(self):
        rows = [{"MOIS": m, "CHIFFRE": v} for m, v in
                zip(self.MONTHS_FR, (100.0, 150.0, 200.0, 260.0, 300.0, 360.0))]
        self.assertEqual(one(rows, "temporal")["direction"], "upward")

    def test_a_series_with_a_missing_reading_is_still_a_series(self):
        # Refusing a sparse GRID must not become refusing a gap. One period
        # that reported nothing is still one row per period.
        rows = [{"FISCAL_QTR": q, "REVENUE_AMT": v}
                for q, v in zip(self.QUARTERS, (100.0, None, 300.0, 400.0))]
        self.assertEqual(one(rows, "temporal")["direction"], "upward")


class TestWhatCountsAsACalendarLabel(unittest.TestCase):
    """core.response_builder._looks_temporal, directly."""

    def temporal(self, labels):
        from core.response_builder import _looks_temporal

        return _looks_temporal(labels)

    def test_quarters_in_both_orders(self):
        self.assertTrue(self.temporal(["Q1 2026", "Q2 2026"]))
        self.assertTrue(self.temporal(["2026-Q1", "2026-Q2"]))

    def test_the_french_quarter_letter(self):
        self.assertTrue(self.temporal(["T1 2026", "T2 2026"]))

    def test_french_months_accented_and_not(self):
        self.assertTrue(self.temporal(["janvier", "février", "mars"]))
        self.assertTrue(self.temporal(["janvier", "fevrier", "aout"]))

    def test_english_months_still_count(self):
        self.assertTrue(self.temporal(["January", "February"]))
        self.assertTrue(self.temporal(["Jan 2024", "Feb 2024"]))
        self.assertTrue(self.temporal(["2026-01", "2026-02"]))

    def test_a_place_is_not_a_month_in_either_language(self):
        # "Mayfield" read as May was already true before quarters and French
        # were added; "Marseille" would have started reading as mars the moment
        # they were. Every token is a whole word now, which settles both.
        for labels in (["Nova", "Maritime", "Mayfield"],
                       ["Marseille", "Marne"],
                       ["Augusta", "Decatur"],
                       ["Januscorp", "Februs"],
                       ["Update", "Mandate"],
                       ["Halifax", "Calgary"]):
            with self.subTest(labels=labels):
                self.assertFalse(self.temporal(labels))


class TestTheThreeDetectorsAgree(unittest.TestCase):
    """The point of using the shared rules rather than a sixth private one."""

    def test_the_leader_matches_the_evidence_engine(self):
        from core.analysis_evidence import CONCENTRATION_LEADER, concentration_findings

        rows = grid()
        signal = one(rows, "group_imbalance")
        finding = next(f for f in concentration_findings(rows, "REVENUE_AMT", "WHS_NM")
                       if f.kind == CONCENTRATION_LEADER)
        self.assertEqual(signal["value"], finding.numbers["share"])
        self.assertEqual(signal["leader"], finding.labels["leader"])

    def test_neither_reports_a_trend_the_other_refuses(self):
        from core.analysis_evidence import build_evidence

        rows = grid()
        self.assertNotIn("temporal", kinds(rows))
        self.assertEqual(
            [f for f in build_evidence(rows).findings
             if str(f.kind).startswith("trend_")], [])


if __name__ == "__main__":
    unittest.main()
