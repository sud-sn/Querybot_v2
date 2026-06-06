"""
Sprint A — Signal-based chip eligibility tests.

Every test drives ``compute_chip_eligibility`` directly with realistic
``ctx`` / ``brief`` dicts and asserts on *which* chips appear, *which* are
suppressed, and that ``pre_context`` carries meaningful signal text.

Design rules verified here:
  - A 2-row time series must NOT produce predict or analyze chips
  - A flat/stable time series must NOT produce a predict chip
  - A ranking with <3 rows must NOT produce analyze/compare chips
  - A ranking with high concentration MUST produce outliers + contribution chips
  - compare_prior appears only when semantic_plan has a date_dimension field
  - explain is always present for non-empty results
  - decide only appears for mode in (time_series, ranking) with rows >= 3
"""

import unittest

from core.response_builder import compute_chip_eligibility


# ── helpers ──────────────────────────────────────────────────────────────────

def _ts_ctx(row_count=6, pct_change=15.0):
    """Minimal time_series ctx."""
    return {
        "mode": "time_series",
        "row_count": row_count,
        "pct_change": pct_change,
        "distribution_stats": {},
        "comparison_stats": {},
    }


def _ts_brief(direction="increasing", period_count=6, overall_pct=15.0):
    """Minimal time_series brief."""
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


# ── Tests ─────────────────────────────────────────────────────────────────────

class ChipEligibilityExplainTests(unittest.TestCase):

    def test_explain_always_present_for_non_empty(self):
        """explain chip must appear for every non-empty result mode."""
        for mode in ("time_series", "ranking", "numeric_table", "single_value"):
            ctx = {"mode": mode, "row_count": 5, "distribution_stats": {}, "comparison_stats": {}}
            chips = compute_chip_eligibility(ctx)
            self.assertIn("explain", _ids(chips),
                          f"explain missing for mode={mode}")

    def test_explain_absent_for_empty_result(self):
        ctx = {"mode": "empty", "row_count": 0}
        chips = compute_chip_eligibility(ctx)
        self.assertNotIn("explain", _ids(chips))


class ChipEligibilityTimeSeries(unittest.TestCase):

    def test_full_series_has_analyze_compare_predict(self):
        """6-period increasing series → all three time_series chips."""
        ctx = _ts_ctx(row_count=6, pct_change=18.0)
        brief = _ts_brief(direction="increasing", period_count=6, overall_pct=18.0)
        chips = compute_chip_eligibility(ctx, brief=brief)
        ids = _ids(chips)
        self.assertIn("analyze", ids)
        self.assertIn("compare", ids)
        self.assertIn("predict", ids)

    def test_two_row_series_suppresses_analyze_and_predict(self):
        """2-row series has too few periods for trend analysis or prediction."""
        ctx = _ts_ctx(row_count=2, pct_change=20.0)
        brief = _ts_brief(direction="increasing", period_count=2, overall_pct=20.0)
        chips = compute_chip_eligibility(ctx, brief=brief)
        ids = _ids(chips)
        self.assertNotIn("analyze", ids, "analyze must be suppressed for 2-period series")
        self.assertNotIn("predict", ids, "predict must be suppressed for 2-period series")

    def test_stable_series_suppresses_predict(self):
        """Stable direction → prediction chip hidden (no trend to extrapolate)."""
        ctx = _ts_ctx(row_count=6, pct_change=1.0)
        brief = _ts_brief(direction="stable", period_count=6, overall_pct=1.0)
        chips = compute_chip_eligibility(ctx, brief=brief)
        self.assertNotIn("predict", _ids(chips),
                         "predict must not appear for stable trend")

    def test_flat_change_suppresses_compare(self):
        """< 3 % overall change → compare chip hidden (nothing meaningful to compare)."""
        ctx = _ts_ctx(row_count=6, pct_change=1.5)
        brief = _ts_brief(direction="stable", period_count=6, overall_pct=1.5)
        chips = compute_chip_eligibility(ctx, brief=brief)
        self.assertNotIn("compare", _ids(chips),
                         "compare must be suppressed when change < 3%")

    def test_declining_series_has_predict(self):
        """Decreasing trend is still predictable."""
        ctx = _ts_ctx(row_count=5, pct_change=-22.0)
        brief = _ts_brief(direction="decreasing", period_count=5, overall_pct=-22.0)
        chips = compute_chip_eligibility(ctx, brief=brief)
        self.assertIn("predict", _ids(chips))

    def test_decide_present_for_structured_series(self):
        ctx = _ts_ctx(row_count=5, pct_change=12.0)
        brief = _ts_brief(direction="increasing", period_count=5, overall_pct=12.0)
        chips = compute_chip_eligibility(ctx, brief=brief)
        self.assertIn("decide", _ids(chips))

    def test_decide_absent_for_short_series(self):
        """2-row series must NOT get a decide chip — not enough structure."""
        ctx = _ts_ctx(row_count=2, pct_change=10.0)
        brief = _ts_brief(period_count=2, overall_pct=10.0)
        chips = compute_chip_eligibility(ctx, brief=brief)
        self.assertNotIn("decide", _ids(chips))

    def test_compare_prior_chip_from_semantic_plan(self):
        """compare_prior chip shown only when semantic_plan has a date_dimension field."""
        ctx = _ts_ctx(row_count=6, pct_change=10.0)
        brief = _ts_brief(period_count=6, overall_pct=10.0)
        plan_with_date = {
            "enabled": True,
            "fields": [{"role": "date_dimension", "column": "DT_DMS_KEY"}],
        }
        chips_with = compute_chip_eligibility(ctx, brief=brief, semantic_plan=plan_with_date)
        chips_without = compute_chip_eligibility(ctx, brief=brief, semantic_plan=None)
        self.assertIn("compare_prior", _ids(chips_with))
        self.assertNotIn("compare_prior", _ids(chips_without))

    def test_compare_prior_absent_when_plan_disabled(self):
        ctx = _ts_ctx(row_count=6, pct_change=10.0)
        brief = _ts_brief(period_count=6)
        plan_disabled = {"enabled": False, "fields": [{"role": "date_dimension"}]}
        chips = compute_chip_eligibility(ctx, brief=brief, semantic_plan=plan_disabled)
        self.assertNotIn("compare_prior", _ids(chips))


class ChipEligibilityRanking(unittest.TestCase):

    def test_full_ranking_has_analyze_compare_outliers_contribution(self):
        """10-row ranking with high concentration → full chip set."""
        ctx = _ranking_ctx(row_count=10, top3_share=78.0, leader_share=42.0, gap=800.0)
        chips = compute_chip_eligibility(ctx)
        ids = _ids(chips)
        self.assertIn("analyze", ids)
        self.assertIn("compare", ids)
        self.assertIn("outliers", ids)
        self.assertIn("contribution", ids)
        self.assertIn("decide", ids)

    def test_two_row_ranking_suppresses_analyze_compare_decide(self):
        """2 items is not a ranking — most chips should be hidden."""
        ctx = _ranking_ctx(row_count=2, top3_share=95.0, leader_share=65.0, gap=300.0)
        chips = compute_chip_eligibility(ctx)
        ids = _ids(chips)
        self.assertNotIn("analyze", ids)
        self.assertNotIn("compare", ids)
        self.assertNotIn("decide", ids)

    def test_low_concentration_suppresses_outliers(self):
        """Even distribution → outliers chip hidden (top-3 < 60%)."""
        ctx = _ranking_ctx(row_count=10, top3_share=35.0, leader_share=15.0, gap=50.0)
        chips = compute_chip_eligibility(ctx)
        self.assertNotIn("outliers", _ids(chips))

    def test_high_concentration_triggers_outliers(self):
        ctx = _ranking_ctx(row_count=8, top3_share=80.0, leader_share=50.0, gap=600.0)
        chips = compute_chip_eligibility(ctx)
        self.assertIn("outliers", _ids(chips))

    def test_contribution_shows_when_leader_share_known(self):
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

    def test_compare_hidden_when_no_gap(self):
        """Zero gap between leader and runner-up → compare is not useful."""
        ctx = _ranking_ctx(row_count=5, gap=0.0)
        chips = compute_chip_eligibility(ctx)
        self.assertNotIn("compare", _ids(chips))


class ChipEligibilityNumericTable(unittest.TestCase):

    def test_analyze_present_when_std_dev_exists(self):
        ctx = {
            "mode": "numeric_table",
            "row_count": 5,
            "distribution_stats": {"std_dev": 45.2},
            "comparison_stats": {},
        }
        chips = compute_chip_eligibility(ctx)
        self.assertIn("analyze", _ids(chips))

    def test_analyze_absent_without_std_dev(self):
        ctx = {
            "mode": "numeric_table",
            "row_count": 5,
            "distribution_stats": {"std_dev": None},
            "comparison_stats": {},
        }
        chips = compute_chip_eligibility(ctx)
        self.assertNotIn("analyze", _ids(chips))


class ChipEligibilityOrdering(unittest.TestCase):

    def test_explain_is_first_chip(self):
        """explain must always be the first chip shown."""
        ctx = _ranking_ctx(row_count=8, top3_share=75.0, leader_share=40.0)
        chips = compute_chip_eligibility(ctx)
        self.assertTrue(chips, "Expected at least one chip")
        self.assertEqual(chips[0]["id"], "explain")

    def test_decide_before_export_chips(self):
        """decide must appear before download_csv and set_alert (Sprint E chips)."""
        ctx = _ts_ctx(row_count=6, pct_change=14.0)
        brief = _ts_brief(direction="increasing", period_count=6, overall_pct=14.0)
        chips = compute_chip_eligibility(ctx, brief=brief)
        ids = [c["id"] for c in chips]
        self.assertIn("decide", ids)
        decide_pos = ids.index("decide")
        for late_chip in ("download_csv", "set_alert"):
            if late_chip in ids:
                self.assertGreater(
                    ids.index(late_chip), decide_pos,
                    f"{late_chip!r} must appear after 'decide'",
                )


class ChipEligibilityPreContext(unittest.TestCase):

    def test_compare_pre_context_contains_pct_change(self):
        """compare chip pre_context must reference the % change."""
        ctx = _ts_ctx(row_count=6, pct_change=18.5)
        brief = _ts_brief(direction="increasing", period_count=6, overall_pct=18.5)
        chips = compute_chip_eligibility(ctx, brief=brief)
        compare_chip = next((c for c in chips if c["id"] == "compare"), None)
        self.assertIsNotNone(compare_chip)
        self.assertIn("18.5", compare_chip["pre_context"])

    def test_outliers_pre_context_contains_concentration(self):
        ctx = _ranking_ctx(row_count=8, top3_share=82.0)
        chips = compute_chip_eligibility(ctx)
        outlier_chip = next((c for c in chips if c["id"] == "outliers"), None)
        self.assertIsNotNone(outlier_chip)
        self.assertIn("82", outlier_chip["pre_context"])

    def test_all_chips_have_required_keys(self):
        """Every chip must carry id, label, confidence, and pre_context."""
        ctx = _ranking_ctx(row_count=8, top3_share=75.0, leader_share=40.0, gap=300.0)
        chips = compute_chip_eligibility(ctx)
        for chip in chips:
            for key in ("id", "label", "confidence", "pre_context"):
                self.assertIn(key, chip, f"Chip {chip.get('id')} missing key {key!r}")


if __name__ == "__main__":
    unittest.main()
