"""
tests/test_trend_reads_period_order.py

The regulated-tenant analysis stated the reverse of the truth.

`trend_findings` documented itself as "a series ordered by period" and then
read `values[0]` and `values[-1]` straight off the rows in whatever order they
arrived. Executed against the shipped code, the identical three months --
100, 200, 300 -- produced:

    rows ascending  (Jan, Feb, Mar)   ->  trend_up
    rows descending (Mar, Feb, Jan)   ->  trend_down

Nothing in SQL guarantees period order. A "top 10 months by revenue" result
carries a period column and is sorted by revenue, so the analysis asserted the
opposite of what happened -- to regulated tenants, in the feature written
specifically for them because no model is allowed near their rows.

The fix orders the series by its period column, and REFUSES when it cannot:
a label with no known ordering produces no trend at all. A confident statement
of the wrong direction is worse than saying nothing, and this feature's whole
value is that its numbers are computed rather than generated.

Every test executes the real detector.
"""

import datetime
import unittest

from core.analysis_evidence import build_evidence, period_order_key, trend_findings

TREND_KINDS = ("trend_up", "trend_down", "trend_flat")


def _trends(rows):
    return [f.kind for f in build_evidence(rows).findings
            if str(f.kind) in TREND_KINDS]


RISING = [
    {"MONTH": "2024-01", "REVENUE": 100.0},
    {"MONTH": "2024-02", "REVENUE": 200.0},
    {"MONTH": "2024-03", "REVENUE": 300.0},
]


class TheSeriesIsReadInPeriodOrder(unittest.TestCase):
    """The defect itself: three orderings of one rising series."""

    def test_ascending_rows_report_a_rise(self):
        self.assertEqual(_trends(RISING), ["trend_up"])

    def test_descending_rows_report_the_same_rise(self):
        """This returned trend_down before the fix."""
        self.assertEqual(_trends(list(reversed(RISING))), ["trend_up"])

    def test_rows_sorted_by_the_measure_report_the_same_rise(self):
        """The shape that actually reaches production: a top-N result is
        ordered by revenue, not by month."""
        by_revenue = sorted(RISING, key=lambda r: -r["REVENUE"])
        self.assertEqual(_trends(by_revenue), ["trend_up"])

    def test_a_genuine_fall_is_still_a_fall(self):
        """Guards the guard: a fix that always answered 'up' would pass the
        three tests above."""
        falling = [{"MONTH": m, "REVENUE": v} for m, v in
                   [("2024-01", 300.0), ("2024-02", 200.0), ("2024-03", 100.0)]]
        self.assertEqual(_trends(falling), ["trend_down"])
        self.assertEqual(_trends(list(reversed(falling))), ["trend_down"])

    def test_the_percentage_is_computed_on_the_ordered_series(self):
        finding = next(f for f in build_evidence(list(reversed(RISING))).findings
                       if str(f.kind) in TREND_KINDS)
        self.assertEqual(finding.numbers["first"], 100.0)
        self.assertEqual(finding.numbers["last"], 300.0)
        self.assertEqual(finding.numbers["pct"], 200.0)


class ItRefusesRatherThanGuesses(unittest.TestCase):

    def test_an_unorderable_label_produces_no_trend(self):
        rows = [{"THING": "alpha", "REVENUE": 100.0},
                {"THING": "beta", "REVENUE": 300.0}]
        self.assertEqual(trend_findings(rows, "REVENUE", "THING"), [])

    def test_labels_that_mix_years_and_bare_months_produce_no_trend(self):
        """Two axes cannot be placed on one line, and sorting them would
        interleave silently."""
        rows = [{"P": "Jan 2024", "REVENUE": 100.0},
                {"P": "February", "REVENUE": 300.0}]
        self.assertEqual(trend_findings(rows, "REVENUE", "P"), [])

    def test_a_non_temporal_result_gets_no_trend_at_all(self):
        rows = [{"REGION": "Nova", "REVENUE": 100.0},
                {"REGION": "Maritime", "REVENUE": 300.0}]
        self.assertEqual(_trends(rows), [])


class ThePeriodKeyKnowsRealWarehouseShapes(unittest.TestCase):

    def test_iso_strings(self):
        self.assertEqual(period_order_key("2024-01"), (2024, 1, 0))
        self.assertEqual(period_order_key("2024-01-15"), (2024, 1, 15))

    def test_the_integer_date_keys_the_date_role_work_documents(self):
        self.assertEqual(period_order_key(202401), (2024, 1, 0))
        self.assertEqual(period_order_key(20240115), (2024, 1, 15))
        self.assertEqual(period_order_key(2024), (2024, 0, 0))

    def test_quarters_both_ways_round(self):
        self.assertEqual(period_order_key("Q1 2024"), (2024, 1, 0))
        self.assertEqual(period_order_key("Q3 2024"), (2024, 7, 0))
        self.assertEqual(period_order_key("2024 Q3"), (2024, 7, 0))

    def test_month_names_with_a_year(self):
        self.assertEqual(period_order_key("Jan 2024"), (2024, 1, 0))
        self.assertEqual(period_order_key("January 2024"), (2024, 1, 0))
        self.assertEqual(period_order_key("2024 Mar"), (2024, 3, 0))

    def test_real_dates(self):
        self.assertEqual(period_order_key(datetime.date(2024, 5, 9)), (2024, 5, 9))
        self.assertEqual(
            period_order_key(datetime.datetime(2024, 5, 9, 13, 30)), (2024, 5, 9))

    def test_a_place_name_is_not_a_month(self):
        """The bug core.stat_signals._is_temporal_col already carries a fix
        for, and which an earlier draft of period_order_key reintroduced by
        matching on a three-letter prefix."""
        for place in ("Nova", "Maritime", "Mayfield", "Augusta", "Decatur",
                      "Julia", "Janus", "Februs", "Aprilia", "Octagon"):
            with self.subTest(place=place):
                self.assertIsNone(period_order_key(place))

    def test_a_real_bare_month_still_orders(self):
        self.assertEqual(period_order_key("March"), (0, 3, 0))
        self.assertEqual(period_order_key("Mar"), (0, 3, 0))
        self.assertEqual(period_order_key("MAY"), (0, 5, 0))

    def test_bare_months_sort_into_calendar_order(self):
        months = ["December", "January", "July", "March"]
        self.assertEqual(
            sorted(months, key=period_order_key),
            ["January", "March", "July", "December"],
        )

    def test_nothing_is_not_a_period(self):
        for value in ("", None, "   ", "12345", "abc"):
            with self.subTest(value=value):
                self.assertIsNone(period_order_key(value))


if __name__ == "__main__":
    unittest.main()
