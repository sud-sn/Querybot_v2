"""
tests/test_insight_single_period.py

Regression: core/insight.py._compute_time_series_brief crashed with
"min() arg is an empty sequence" whenever a time-series-labeled result had
exactly one row — e.g. "which month has the highest number of purchase
receipt" resolving to a single (month, count) row via TOP 1/LIMIT 1. The
period-over-period comparison list (pop_changes) is necessarily empty when
there's only one period, but biggest_period_drop/gain called min()/max() on
it unconditionally.

Confirmed live: the crash happened inside a WebSocket background task after
SQL execution, silently swallowing the answer — the user saw no response at
all, only a server-side "WS bg task error: min() arg is an empty sequence".
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.insight import compute_data_brief, _compute_time_series_brief


class SinglePeriodTimeSeriesTests(unittest.TestCase):
    def test_compute_time_series_brief_does_not_crash_on_one_row(self):
        ts = _compute_time_series_brief(["2026-06"], [812.0])
        self.assertIsNone(ts["biggest_period_drop"])
        self.assertIsNone(ts["biggest_period_gain"])
        self.assertEqual(ts["period_count"], 1)
        self.assertEqual(ts["first_period"], "2026-06")
        self.assertEqual(ts["last_period"], "2026-06")
        self.assertEqual(ts["peak"]["period"], "2026-06")
        self.assertEqual(ts["trough"]["period"], "2026-06")

    def test_compute_data_brief_single_row_which_month_has_highest(self):
        # "which month has highest number of purchase receipt" style result:
        # a TOP 1/LIMIT 1 query collapses to exactly one (month, count) row.
        rows = [{"MONTH": "2026-06", "RECEIPT_COUNT": 812}]
        brief = compute_data_brief(rows, "which month has highest number of purchase receipt")
        self.assertEqual(brief["mode"], "time_series")
        self.assertIsNone(brief["time_series"]["biggest_period_drop"])
        self.assertIsNone(brief["time_series"]["biggest_period_gain"])

    def test_compute_time_series_brief_still_works_for_two_or_more_rows(self):
        # Guardrail: the fix must not break the normal multi-period path.
        ts = _compute_time_series_brief(["2026-05", "2026-06"], [500.0, 812.0])
        self.assertIsNotNone(ts["biggest_period_drop"])
        self.assertIsNotNone(ts["biggest_period_gain"])
        self.assertEqual(ts["biggest_period_gain"]["to_period"], "2026-06")


if __name__ == "__main__":
    unittest.main()
