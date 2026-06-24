"""
Chip eligibility tests — updated for the slim chip set.

Chips that exist:
  time_series  : compare, diagnose, compare_prior
  ranking      : contribution
  all modes    : drill_dim (max 2), download_csv

Removed chips (no longer generated):
  explain, analyze, predict, outliers, decide, set_alert
"""

import unittest

from core.response_builder import compute_chip_eligibility


# ── helpers ──────────────────────────────────────────────────────────────────

def _ts_ctx(row_count=6, pct_change=15.0):
    return {
        "mode": "time_series",
        "row_count": row_count,
        "pct_change": pct_change,
        "distribution_stats": {},
        "comparison_stats": {},
    }


def _ts_brief(direction="increasing", period_count=6, overall_pct=15.0):
    return {
        "time_series": {
            "direction": direction,
            "period_count": period_count,
            "overall_pct_change": overall_pct,
            "first_period": "Jan",
            "last_period": "Jun",
            "first_value": 100.0,
            "last_value": 115.0,
        }
    }


def _ranking_ctx(row_count=10, top3_share=75.0, leader_share=45.0, gap=500.0):
    return {
        "mode": "ranking",
        "row_count": row_count,
        "distribution_stats": {
            "top_3_share_pct": top3_share,
            "std_dev": 120.0,
        },
        "comparison_stats": {
            "leader": "Warehouse A",
            "leader_value": 2000.0,
            "leader_share_pct": leader_share,
            "gap": gap,
        },
    }


def _ids(chips):
    return [c["id"] for c in chips]


# ── Removed chips — none of these should appear any more ─────────────────────

class RemovedChipsAbsent(unittest.TestCase):

    def _assert_absent(self, chip_id, ctx, brief=None, plan=None):
        chips = compute_chip_eligibility(ctx, brief=brief, semantic_plan=plan)
        self.assertNotIn(chip_id, _ids(chips), f"removed chip '{chip_id}' must not appear")

    def test_explain_never_generated(self):
        for mode in ("time_series", "ranking", "numeric_table", "single_value"):
            ctx = {"mode": mode, "row_count": 5, "distribution_stats": {}, "comparison_stats": {}}
            self._assert_absent("explain", ctx)

    def test_analyze_never_generated(self):
        ctx = _ts_ctx(row_count=6, pct_change=18.0)
        self._assert_absent("analyze", ctx, brief=_ts_brief(period_count=6, overall_pct=18.0))
        self._assert_absent("analyze", _ranking_ctx(row_count=10))

    def test_predict_never_generated(self):
        ctx = _ts_ctx(row_count=6, pct_change=-22.0)
        brief = _ts_brief(direction="decreasing", period_count=6, overall_pct=-22.0)
        self._assert_absent("predict", ctx, brief=brief)

    def test_outliers_never_generated(self):
        ctx = _ranking_ctx(row_count=8, top3_share=90.0)
        self._assert_absent("outliers", ctx)

    def test_decide_never_generated(self):
        ctx = _ts_ctx(row_count=5, pct_change=12.0)
        self._assert_absent("decide", ctx)

    def test_set_alert_never_generated(self):
        ctx = {"mode": "single_value", "row_count": 1, "distribution_stats": {}, "comparison_stats": {}}
        self._assert_absent("set_alert", ctx)


# ── Time-series chips ─────────────────────────────────────────────────────────

class ChipEligibilityTimeSeries(unittest.TestCase):

    def test_compare_shown_for_significant_change(self):
        ctx = _ts_ctx(row_count=6, pct_change=18.0)
        brief = _ts_brief(period_count=6, overall_pct=18.0)
        chips = compute_chip_eligibility(ctx, brief=brief)
        self.assertIn("compare", _ids(chips))

    def test_compare_suppressed_for_tiny_change(self):
        """< 3 % change → compare not useful."""
        ctx = _ts_ctx(row_count=6, pct_change=1.5)
        brief = _ts_brief(direction="stable", period_count=6, overall_pct=1.5)
        chips = compute_chip_eligibility(ctx, brief=brief)
        self.assertNotIn("compare", _ids(chips))

    def test_diagnose_shown_for_drop_gte_5pct(self):
        ctx = _ts_ctx(row_count=6, pct_change=-12.0)
        brief = _ts_brief(direction="decreasing", period_count=6, overall_pct=-12.0)
        chips = compute_chip_eligibility(ctx, brief=brief)
        self.assertIn("diagnose", _ids(chips))

    def test_diagnose_shown_for_rise_gte_5pct(self):
        ctx = _ts_ctx(row_count=6, pct_change=8.0)
        brief = _ts_brief(direction="increasing", period_count=6, overall_pct=8.0)
        chips = compute_chip_eligibility(ctx, brief=brief)
        self.assertIn("diagnose", _ids(chips))

    def test_diagnose_suppressed_for_small_change(self):
        """< 5 % change → no diagnose chip."""
        ctx = _ts_ctx(row_count=6, pct_change=3.0)
        brief = _ts_brief(period_count=6, overall_pct=3.0)
        chips = compute_chip_eligibility(ctx, brief=brief)
        self.assertNotIn("diagnose", _ids(chips))

    def test_diagnose_label_says_drop_vs_rise(self):
        ctx_drop = _ts_ctx(row_count=4, pct_change=-10.0)
        brief_drop = _ts_brief(direction="decreasing", period_count=4, overall_pct=-10.0)
        chips_drop = compute_chip_eligibility(ctx_drop, brief=brief_drop)
        drop_chip = next(c for c in chips_drop if c["id"] == "diagnose")
        self.assertIn("drop", drop_chip["label"])

        ctx_rise = _ts_ctx(row_count=4, pct_change=10.0)
        brief_rise = _ts_brief(direction="increasing", period_count=4, overall_pct=10.0)
        chips_rise = compute_chip_eligibility(ctx_rise, brief=brief_rise)
        rise_chip = next(c for c in chips_rise if c["id"] == "diagnose")
        self.assertIn("rise", rise_chip["label"])

    def test_compare_prior_shown_with_date_role(self):
        ctx = _ts_ctx(row_count=6, pct_change=10.0)
        brief = _ts_brief(period_count=6, overall_pct=10.0)
        plan = {
            "enabled": True,
            "fields": [{"role": "date_dimension", "column": "DT_DMS_KEY"}],
        }
        chips = compute_chip_eligibility(ctx, brief=brief, semantic_plan=plan)
        self.assertIn("compare_prior", _ids(chips))

    def test_compare_prior_absent_without_date_role(self):
        ctx = _ts_ctx(row_count=6, pct_change=10.0)
        brief = _ts_brief(period_count=6, overall_pct=10.0)
        chips = compute_chip_eligibility(ctx, brief=brief, semantic_plan=None)
        self.assertNotIn("compare_prior", _ids(chips))

    def test_compare_prior_absent_when_plan_disabled(self):
        ctx = _ts_ctx(row_count=6, pct_change=10.0)
        brief = _ts_brief(period_count=6)
        plan = {"enabled": False, "fields": [{"role": "date_dimension"}]}
        chips = compute_chip_eligibility(ctx, brief=brief, semantic_plan=plan)
        self.assertNotIn("compare_prior", _ids(chips))


# ── Ranking chips ─────────────────────────────────────────────────────────────

class ChipEligibilityRanking(unittest.TestCase):

    def test_contribution_shown_when_leader_share_known(self):
        ctx = _ranking_ctx(row_count=5, leader_share=38.0)
        chips = compute_chip_eligibility(ctx)
        self.assertIn("contribution", _ids(chips))

    def test_contribution_hidden_without_leader_share(self):
        ctx = {
            "mode": "ranking",
            "row_count": 5,
            "distribution_stats": {"top_3_share_pct": 70.0},
            "comparison_stats": {},  # no leader_share_pct
        }
        chips = compute_chip_eligibility(ctx)
        self.assertNotIn("contribution", _ids(chips))


# ── Drill-dim chips ───────────────────────────────────────────────────────────

class ChipEligibilityDrillDim(unittest.TestCase):

    def _plan_with_dims(self, names):
        return {
            "enabled": True,
            "fields": [],
            "available_dimensions": [
                {"name": n, "display_column": n.upper(), "status": "approved"}
                for n in names
            ],
        }

    def test_max_two_drill_chips(self):
        """At most 2 drill_dim chips should be generated."""
        ctx = _ts_ctx(row_count=6, pct_change=5.0)
        plan = self._plan_with_dims(["Warehouse", "Customer", "Product", "Region"])
        chips = compute_chip_eligibility(ctx, semantic_plan=plan)
        drill = [c for c in chips if c["id"].startswith("drill_dim:")]
        self.assertLessEqual(len(drill), 2)

    def test_existing_col_skipped(self):
        """A dimension already in the result columns must not get a drill chip."""
        ctx = {
            "mode": "ranking",
            "row_count": 5,
            "text_cols": ["WAREHOUSE"],
            "numeric_cols": [],
            "distribution_stats": {},
            "comparison_stats": {"leader_share_pct": 40.0, "leader": "A"},
        }
        plan = self._plan_with_dims(["Warehouse", "Customer"])
        chips = compute_chip_eligibility(ctx, semantic_plan=plan)
        ids = _ids(chips)
        self.assertNotIn("drill_dim:Warehouse", ids)
        self.assertIn("drill_dim:Customer", ids)


# ── Download CSV ──────────────────────────────────────────────────────────────

class ChipEligibilityDownloadCSV(unittest.TestCase):

    def test_download_csv_always_present_for_non_empty(self):
        for mode in ("time_series", "ranking", "numeric_table"):
            ctx = {"mode": mode, "row_count": 3, "distribution_stats": {}, "comparison_stats": {}}
            chips = compute_chip_eligibility(ctx)
            self.assertIn("download_csv", _ids(chips), f"download_csv missing for mode={mode}")

    def test_download_csv_absent_for_empty_result(self):
        ctx = {"mode": "empty", "row_count": 0, "distribution_stats": {}, "comparison_stats": {}}
        chips = compute_chip_eligibility(ctx)
        self.assertNotIn("download_csv", _ids(chips))


# ── Ordering ──────────────────────────────────────────────────────────────────

class ChipEligibilityOrdering(unittest.TestCase):

    def test_diagnose_before_download_csv(self):
        ctx = _ts_ctx(row_count=6, pct_change=-15.0)
        brief = _ts_brief(direction="decreasing", period_count=6, overall_pct=-15.0)
        chips = compute_chip_eligibility(ctx, brief=brief)
        ids = _ids(chips)
        self.assertIn("diagnose", ids)
        self.assertIn("download_csv", ids)
        self.assertLess(ids.index("diagnose"), ids.index("download_csv"))

    def test_compare_before_diagnose(self):
        ctx = _ts_ctx(row_count=6, pct_change=-15.0)
        brief = _ts_brief(direction="decreasing", period_count=6, overall_pct=-15.0)
        chips = compute_chip_eligibility(ctx, brief=brief)
        ids = _ids(chips)
        self.assertIn("compare", ids)
        self.assertIn("diagnose", ids)
        self.assertLess(ids.index("compare"), ids.index("diagnose"))


# ── Pre-context ───────────────────────────────────────────────────────────────

class ChipEligibilityPreContext(unittest.TestCase):

    def test_compare_pre_context_contains_pct_change(self):
        ctx = _ts_ctx(row_count=6, pct_change=18.5)
        brief = _ts_brief(direction="increasing", period_count=6, overall_pct=18.5)
        chips = compute_chip_eligibility(ctx, brief=brief)
        compare_chip = next((c for c in chips if c["id"] == "compare"), None)
        self.assertIsNotNone(compare_chip)
        self.assertIn("18.5", compare_chip["pre_context"])

    def test_diagnose_pre_context_contains_pct(self):
        ctx = _ts_ctx(row_count=4, pct_change=-10.0)
        brief = _ts_brief(direction="decreasing", period_count=4, overall_pct=-10.0)
        chips = compute_chip_eligibility(ctx, brief=brief)
        chip = next((c for c in chips if c["id"] == "diagnose"), None)
        self.assertIsNotNone(chip)
        self.assertIn("10.0", chip["pre_context"])

    def test_all_chips_have_required_keys(self):
        ctx = _ranking_ctx(row_count=8, top3_share=75.0, leader_share=40.0, gap=300.0)
        chips = compute_chip_eligibility(ctx)
        for chip in chips:
            for key in ("id", "label", "confidence", "pre_context"):
                self.assertIn(key, chip, f"Chip {chip.get('id')} missing key {key!r}")


if __name__ == "__main__":
    unittest.main()
