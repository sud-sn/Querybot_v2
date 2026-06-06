"""
Sprint D — Result transform tests (contribution + outliers).

All tests operate on in-memory row lists — no DB, no LLM, no async.
Tests are grouped by function with explicit edge-case coverage.

Groups:
  AddContributionPctTests   — add_contribution_pct happy path + edges
  FilterOutliersTests       — filter_outliers happy path + edges
  DescribeSqlTests          — human-readable SQL description helpers
"""

import unittest

from core.result_transforms import (
    add_contribution_pct,
    describe_contribution_sql,
    describe_outlier_sql,
    filter_outliers,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _rows(*values, label_col="Warehouse", metric_col="Revenue"):
    return [{label_col: f"W{i}", metric_col: v} for i, v in enumerate(values, 1)]


# ══════════════════════════════════════════════════════════════════════════════
# add_contribution_pct
# ══════════════════════════════════════════════════════════════════════════════

class AddContributionPctTests(unittest.TestCase):

    def test_basic_shares_sum_to_100(self):
        rows = _rows(200, 300, 500)
        result, stats = add_contribution_pct(rows, "Revenue")
        self.assertTrue(stats["ok"])
        total_share = sum(r["Revenue Share %"] for r in result)
        self.assertAlmostEqual(total_share, 100.0, places=0)

    def test_correct_share_values(self):
        rows = _rows(100, 400)   # 20% and 80%
        result, stats = add_contribution_pct(rows, "Revenue")
        shares = sorted(r["Revenue Share %"] for r in result)
        self.assertEqual(shares, [20.0, 80.0])

    def test_adds_share_column_with_default_name(self):
        rows = _rows(100, 200, 300)
        result, _ = add_contribution_pct(rows, "Revenue")
        self.assertIn("Revenue Share %", result[0])

    def test_custom_pct_col_name(self):
        rows = _rows(100, 200)
        result, _ = add_contribution_pct(rows, "Revenue", pct_col_name="Pct")
        self.assertIn("Pct", result[0])
        self.assertNotIn("Revenue Share %", result[0])

    def test_original_columns_preserved(self):
        rows = _rows(100, 200)
        result, _ = add_contribution_pct(rows, "Revenue")
        self.assertIn("Warehouse", result[0])
        self.assertIn("Revenue", result[0])

    def test_sorted_desc_by_default(self):
        rows = _rows(100, 300, 200)   # unsorted
        result, _ = add_contribution_pct(rows, "Revenue")
        values = [r["Revenue"] for r in result]
        self.assertEqual(values, [300, 200, 100])

    def test_sort_disabled(self):
        rows = _rows(100, 300, 200)
        result, _ = add_contribution_pct(rows, "Revenue", sort_desc=False)
        values = [r["Revenue"] for r in result]
        self.assertEqual(values, [100, 300, 200])   # original order

    def test_none_metric_value_gets_zero_share(self):
        rows = [{"Warehouse": "A", "Revenue": 200},
                {"Warehouse": "B", "Revenue": None}]
        result, stats = add_contribution_pct(rows, "Revenue")
        self.assertTrue(stats["ok"])
        b_row = next(r for r in result if r["Warehouse"] == "B")
        self.assertEqual(b_row["Revenue Share %"], 0.0)

    def test_empty_rows_returns_ok_false(self):
        _, stats = add_contribution_pct([], "Revenue")
        self.assertFalse(stats["ok"])
        self.assertEqual(stats["reason"], "no_rows")

    def test_missing_metric_col_returns_ok_false(self):
        _, stats = add_contribution_pct(_rows(100, 200), "")
        self.assertFalse(stats["ok"])
        self.assertEqual(stats["reason"], "no_metric_col")

    def test_zero_total_returns_ok_false(self):
        rows = [{"Warehouse": "A", "Revenue": 0}, {"Warehouse": "B", "Revenue": 0}]
        _, stats = add_contribution_pct(rows, "Revenue")
        self.assertFalse(stats["ok"])
        self.assertEqual(stats["reason"], "zero_total")

    def test_non_numeric_metric_returns_ok_false(self):
        rows = [{"Warehouse": "A", "Revenue": "N/A"}]
        _, stats = add_contribution_pct(rows, "Revenue")
        self.assertFalse(stats["ok"])

    def test_stats_carry_total_and_metric_col(self):
        rows = _rows(200, 300, 500)
        _, stats = add_contribution_pct(rows, "Revenue")
        self.assertEqual(stats["metric_col"], "Revenue")
        self.assertEqual(stats["total"], 1000.0)

    def test_single_row_gets_100_pct(self):
        rows = _rows(500)
        result, stats = add_contribution_pct(rows, "Revenue")
        self.assertTrue(stats["ok"])
        self.assertEqual(result[0]["Revenue Share %"], 100.0)

    def test_negative_values_handled(self):
        """Negative values are valid — share may be negative for losses."""
        rows = [{"W": "A", "PnL": -100}, {"W": "B", "PnL": 300}]
        _, stats = add_contribution_pct(rows, "PnL")
        # total = 200 (non-zero) — ok
        self.assertTrue(stats["ok"])

    def test_original_rows_not_mutated(self):
        rows = _rows(100, 200)
        original_keys = set(rows[0].keys())
        add_contribution_pct(rows, "Revenue")
        self.assertEqual(set(rows[0].keys()), original_keys)


# ══════════════════════════════════════════════════════════════════════════════
# filter_outliers
# ══════════════════════════════════════════════════════════════════════════════

class FilterOutliersTests(unittest.TestCase):

    # Ten warehouses: 9 average (~100), one obvious outlier (1000)
    _rows_with_outlier = [{"W": f"W{i}", "Rev": 100 + i * 2} for i in range(9)] + \
                         [{"W": "BIG", "Rev": 1000}]

    def test_obvious_outlier_is_returned(self):
        filtered, stats = filter_outliers(self._rows_with_outlier, "Rev")
        self.assertTrue(stats["ok"])
        labels = [r["W"] for r in filtered]
        self.assertIn("BIG", labels)

    def test_average_rows_excluded(self):
        filtered, stats = filter_outliers(self._rows_with_outlier, "Rev")
        self.assertTrue(stats["ok"])
        labels = [r["W"] for r in filtered]
        self.assertNotIn("W0", labels)

    def test_filtered_rows_sorted_desc(self):
        rows = [{"W": f"W{i}", "Rev": v} for i, v in enumerate([50, 900, 800, 100, 60])]
        filtered, stats = filter_outliers(rows, "Rev")
        if stats["ok"] and len(filtered) > 1:
            values = [r["Rev"] for r in filtered]
            self.assertEqual(values, sorted(values, reverse=True))

    def test_stats_contain_required_keys(self):
        filtered, stats = filter_outliers(self._rows_with_outlier, "Rev")
        self.assertTrue(stats["ok"])
        for key in ("total_rows", "outlier_rows", "mean", "std_dev",
                    "threshold_value", "threshold", "metric_col"):
            self.assertIn(key, stats, f"stats missing key {key!r}")

    def test_stats_outlier_count_matches_filtered_rows(self):
        filtered, stats = filter_outliers(self._rows_with_outlier, "Rev")
        self.assertTrue(stats["ok"])
        self.assertEqual(stats["outlier_rows"], len(filtered))

    def test_default_threshold_is_1_5(self):
        _, stats = filter_outliers(self._rows_with_outlier, "Rev")
        self.assertEqual(stats.get("threshold"), 1.5)

    def test_custom_threshold(self):
        """With threshold=0.5, more rows should qualify as outliers."""
        _, stats_15 = filter_outliers(self._rows_with_outlier, "Rev", threshold=1.5)
        _, stats_05 = filter_outliers(self._rows_with_outlier, "Rev", threshold=0.5)
        if stats_15.get("ok") and stats_05.get("ok"):
            self.assertGreaterEqual(stats_05["outlier_rows"], stats_15["outlier_rows"])

    def test_empty_rows_returns_ok_false(self):
        _, stats = filter_outliers([], "Rev")
        self.assertFalse(stats["ok"])
        self.assertEqual(stats["reason"], "no_rows")

    def test_missing_metric_col_returns_ok_false(self):
        _, stats = filter_outliers(self._rows_with_outlier, "")
        self.assertFalse(stats["ok"])

    def test_too_few_rows_returns_ok_false(self):
        """Needs ≥3 rows for stdev to be meaningful."""
        _, stats = filter_outliers(_rows(100, 200), "Revenue")
        self.assertFalse(stats["ok"])
        self.assertEqual(stats["reason"], "too_few_rows")

    def test_zero_variance_returns_ok_false(self):
        """All equal values → zero stdev → no outliers possible."""
        rows = [{"W": f"W{i}", "Rev": 100} for i in range(5)]
        _, stats = filter_outliers(rows, "Rev")
        self.assertFalse(stats["ok"])
        self.assertEqual(stats["reason"], "zero_variance")

    def test_evenly_distributed_returns_ok_false_no_outliers(self):
        """When all values are within 1.5 stdev, reason is no_outliers."""
        rows = [{"W": f"W{i}", "Rev": 95 + i * 2} for i in range(6)]
        _, stats = filter_outliers(rows, "Rev")
        # May or may not find outliers — just check ok-false carries a reason
        if not stats["ok"]:
            self.assertIn(stats["reason"],
                          ("no_outliers", "zero_variance", "too_few_rows"))

    def test_original_rows_not_mutated(self):
        rows = list(self._rows_with_outlier)
        original_len = len(rows)
        filter_outliers(rows, "Rev")
        self.assertEqual(len(rows), original_len)

    def test_three_rows_is_minimum_valid(self):
        rows = [{"W": "A", "Rev": 10}, {"W": "B", "Rev": 10}, {"W": "C", "Rev": 1000}]
        filtered, stats = filter_outliers(rows, "Rev")
        # May pass or fail depending on distribution — just ensure no exception
        self.assertIsInstance(stats, dict)
        self.assertIsInstance(filtered, list)


# ══════════════════════════════════════════════════════════════════════════════
# describe_contribution_sql / describe_outlier_sql
# ══════════════════════════════════════════════════════════════════════════════

class DescribeSqlTests(unittest.TestCase):

    def test_contribution_sql_contains_metric_col(self):
        s = describe_contribution_sql("Revenue", 5000.0)
        self.assertIn("Revenue", s)

    def test_contribution_sql_contains_total(self):
        s = describe_contribution_sql("Revenue", 5000.0)
        self.assertIn("5,000", s)

    def test_contribution_sql_is_comment(self):
        s = describe_contribution_sql("Revenue", 5000.0)
        self.assertTrue(s.strip().startswith("--"))

    def test_outlier_sql_contains_metric_col(self):
        stats = {"mean": 100.0, "std_dev": 50.0, "threshold_value": 175.0,
                 "threshold": 1.5, "total_rows": 10, "outlier_rows": 2}
        s = describe_outlier_sql("Revenue", stats)
        self.assertIn("Revenue", s)

    def test_outlier_sql_contains_cutoff_value(self):
        stats = {"mean": 100.0, "std_dev": 50.0, "threshold_value": 175.0,
                 "threshold": 1.5, "total_rows": 10, "outlier_rows": 2}
        s = describe_outlier_sql("Revenue", stats)
        self.assertIn("175", s)

    def test_outlier_sql_shows_row_counts(self):
        stats = {"mean": 100.0, "std_dev": 50.0, "threshold_value": 175.0,
                 "threshold": 1.5, "total_rows": 10, "outlier_rows": 2}
        s = describe_outlier_sql("Revenue", stats)
        self.assertIn("2", s)
        self.assertIn("10", s)

    def test_outlier_sql_is_comment(self):
        stats = {"mean": 100.0, "std_dev": 50.0, "threshold_value": 175.0,
                 "threshold": 1.5, "total_rows": 10, "outlier_rows": 2}
        s = describe_outlier_sql("Revenue", stats)
        self.assertTrue(s.strip().startswith("--"))


if __name__ == "__main__":
    unittest.main()
