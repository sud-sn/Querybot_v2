"""
tests/test_truncated_result_statistics.py

Statistics computed over a truncated fetch were presented as fact.

core/schema.py:run_query caps every query at max_rows (200 by default) and
nothing in core/query_pipeline.py ever raised it. The distribution shapes then
computed their statistics in Python over whatever came back:

    rows  = compute_boxplot(rows, _v_col, _g_col)     # median, Q1, Q3
    rows  = compute_histogram(rows, _h_col)           # bin counts
    _corr = compute_correlation(rows, _x_col, _y_col)
    rows  = compute_cohort_matrix(rows, ...)

The retained rows are not a sample. build_boxplot_sql_hint asks the model for
"UN-AGGREGATED individual rows" with an ORDER BY, and says "Add a TOP/LIMIT
5000 if the table is large" -- a limit run_query ignored. So the rows kept are
the sorted head of one group. Measured against a 10,000-row population whose
true median is 4,999.5, the pre-fix code reported 99.5: 98% low, with no
caveat and no flag. On the EMCO mart that population is 9.2M rows.

The fix detects truncation exactly (overfetch by one, rather than guessing
from len(rows) == cap, which false-alarms on a complete 200-row result) and
refuses the four statistics instead of computing them over a prefix.

Every assertion here fails against the pre-fix code.
"""

import ast
import unittest
from pathlib import Path

import core.schema as schema
from core.cohort_analysis import compute_cohort_matrix
from core.correlation_analysis import compute_correlation
from core.distribution_analysis import compute_boxplot, compute_histogram

REPO = Path(__file__).resolve().parents[1]


def _rows(n, start=0):
    """n rows of one group with strictly increasing values."""
    return [{"region": "North", "amount": float(start + i)} for i in range(n)]

class TruncationIsDetectedExactly(unittest.TestCase):
    """run_query must tell 'exactly max_rows exist' from 'we stopped reading'."""

    def setUp(self):
        self._orig = schema._run_azure_sql
        self.addCleanup(lambda: setattr(schema, "_run_azure_sql", self._orig))

    def _stub(self, available):
        def _fake(cfg, sql, max_rows=200):
            return _rows(available)[:max_rows]

        schema._run_azure_sql = _fake

    def test_more_rows_than_the_cap_is_reported_as_truncated(self):
        self._stub(available=5000)
        out = schema.run_query({}, "azure_sql", "SELECT 1", max_rows=200)
        self.assertEqual(len(out), 200, "must trim back to the caller cap")
        self.assertTrue(
            getattr(out, "truncated", False),
            "5000 available rows behind a 200 cap must report truncated=True",
        )

    def test_exactly_the_cap_is_not_truncated(self):
        """A len(rows) == cap guess would false-alarm here; overfetch does not."""
        self._stub(available=200)
        out = schema.run_query({}, "azure_sql", "SELECT 1", max_rows=200)
        self.assertEqual(len(out), 200)
        self.assertFalse(
            getattr(out, "truncated", False),
            "a complete 200-row result must not be flagged truncated",
        )

    def test_under_the_cap_is_not_truncated(self):
        self._stub(available=17)
        out = schema.run_query({}, "azure_sql", "SELECT 1", max_rows=200)
        self.assertEqual(len(out), 17)
        self.assertFalse(getattr(out, "truncated", False))

    def test_result_is_still_a_plain_list_for_every_existing_caller(self):
        self._stub(available=3)
        out = schema.run_query({}, "azure_sql", "SELECT 1", max_rows=200)
        self.assertIsInstance(out, list)
        self.assertEqual(
            out, [{"region": "North", "amount": float(i)} for i in range(3)]
        )


class StatisticsRefuseOnTruncatedInput(unittest.TestCase):
    """The four row-level statistics must not produce a number from a prefix."""

    def test_boxplot_computes_on_a_complete_result(self):
        out = compute_boxplot(_rows(40), "amount", "region", truncated=False)
        self.assertTrue(any("bp_median" in r for r in out), "control: must compute")

    def test_boxplot_refuses_on_a_truncated_result(self):
        src = _rows(200)
        out = compute_boxplot(src, "amount", "region", truncated=True)
        self.assertFalse(
            any("bp_median" in r for r in out),
            "quartiles over a truncated prefix are wrong, not approximate",
        )
        self.assertEqual(
            out, src, "rows pass through untouched so the table still renders"
        )

    def test_histogram_computes_on_a_complete_result(self):
        out = compute_histogram(_rows(40), "amount", truncated=False)
        self.assertTrue(any("bin_label" in r for r in out), "control: must compute")

    def test_histogram_refuses_on_a_truncated_result(self):
        src = _rows(200)
        out = compute_histogram(src, "amount", truncated=True)
        self.assertFalse(any("bin_label" in r for r in out))
        self.assertEqual(out, src)

    def test_correlation_computes_on_a_complete_result(self):
        rows = [{"x": float(i), "y": float(2 * i)} for i in range(30)]
        res = compute_correlation(rows, "x", "y", truncated=False)
        self.assertIsNotNone(res.pearson_r, "control: must compute")

    def test_correlation_refuses_on_a_truncated_result(self):
        """Sorting on either axis manufactures correlation, so r must not be reported."""
        rows = [{"x": float(i), "y": float(2 * i)} for i in range(200)]
        res = compute_correlation(rows, "x", "y", truncated=True)
        self.assertIsNone(res.pearson_r)
        self.assertIn("truncated", res.interpretation.lower())

    def test_cohort_matrix_refuses_on_a_truncated_result(self):
        src = [
            {"cohort_month": "2024-01", "period_month": 0, "user_count": 1200},
            {"cohort_month": "2024-01", "period_month": 1, "user_count": 960},
        ]
        out = compute_cohort_matrix(
            src, "cohort_month", "period_month", "user_count", truncated=True
        )
        self.assertEqual(out, src, "no retention matrix from a truncated fetch")


class TruncationPropagatesToTheGovernedResult(unittest.TestCase):
    """The flag must survive the boundary that masking transforms cross."""

    def test_governed_result_carries_truncated(self):
        from core.compliance.governed_query import GovernedQueryResult

        field = GovernedQueryResult.__dataclass_fields__.get("truncated")
        self.assertIsNotNone(
            field, "GovernedQueryResult must expose truncation to the pipeline"
        )
        self.assertIs(field.default, False, "default must be the safe, unflagged value")

    def test_flag_is_read_before_masking_can_drop_it(self):
        """Order matters: protect_rows returns a plain list and loses the attribute."""
        src = (REPO / "core" / "compliance" / "governed_query.py").read_text(
            encoding="utf-8"
        )
        fetch_at = src.index("raw_rows = run_query(")
        capture_at = src.index('getattr(raw_rows, "truncated"')
        protect_at = src.index("protect_rows(")
        self.assertLess(fetch_at, capture_at)
        self.assertLess(
            capture_at,
            protect_at,
            "truncation must be captured before rows are transformed",
        )


class PipelinePassesTheFlagToEveryStatistic(unittest.TestCase):
    """Structural guard. A fix that lands but is never wired is this repo's
    recurring failure mode (suggestions.py:391, the claims_display gate,
    stat_signals). The behavioural tests above prove the guards work; this
    proves the pipeline actually reaches them."""

    GUARDED = {
        "compute_boxplot",
        "compute_histogram",
        "compute_correlation",
        "compute_cohort_matrix",
    }

    def test_every_call_site_passes_truncated(self):
        tree = ast.parse(
            (REPO / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        )
        seen = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name in self.GUARDED:
                seen.add(name)
                kwargs = {k.arg for k in node.keywords}
                self.assertIn(
                    "truncated",
                    kwargs,
                    f"{name}() is called without truncated= - the guard cannot fire",
                )
        self.assertEqual(
            seen,
            self.GUARDED,
            f"expected a call site for each of {sorted(self.GUARDED)}, found {sorted(seen)}",
        )


if __name__ == "__main__":
    unittest.main()
