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
