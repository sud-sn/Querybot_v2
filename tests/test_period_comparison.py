"""
Sprint B — Period comparison tests.

Covers the pure functions in core/period_comparison.py:
  - detect_period_grain
  - compute_prior_period
  - build_prior_period_rewrite_prompt
  - build_period_comparison_narrative_prompt
  - _extract_date_col_hint
  - _clean_sql_response
  - _deterministic_headline / _deterministic_bullets (fallback narrative)

No DB or LLM calls are made — the async generate_period_comparison entry
point is tested only for its deterministic fallback paths (no time_series
brief → immediate fallback dict).
"""

import asyncio
import unittest

from core.period_comparison import (
    GRAIN_MONTHLY,
    GRAIN_QUARTERLY,
    GRAIN_YEARLY,
    GRAIN_UNKNOWN,
    compute_prior_period,
    detect_period_grain,
    build_prior_period_rewrite_prompt,
    build_period_comparison_narrative_prompt,
    _clean_sql_response,
    _extract_date_col_hint,
    _deterministic_headline,
    _deterministic_bullets,
    _parse_narrative,
)


# ═════════════════════════════���════════════════════════════════════════════════
# detect_period_grain
# ══════════════════════════════════════════════════════════════════════════════

class DetectPeriodGrainTests(unittest.TestCase):

    def test_yyyy_mm_format(self):
        self.assertEqual(detect_period_grain(["2024-01", "2024-06"]), GRAIN_MONTHLY)

    def test_yyyy_slash_m_format(self):
        self.assertEqual(detect_period_grain(["2024/1", "2024/12"]), GRAIN_MONTHLY)

    def test_month_name_short(self):
        self.assertEqual(detect_period_grain(["Jan", "Feb", "Mar"]), GRAIN_MONTHLY)

    def test_month_name_full(self):
        self.assertEqual(detect_period_grain(["January", "February"]), GRAIN_MONTHLY)

    def test_month_name_with_year(self):
        self.assertEqual(detect_period_grain(["Jan 2025", "Mar 2025"]), GRAIN_MONTHLY)

    def test_quarterly_q_first(self):
        self.assertEqual(detect_period_grain(["Q1 2024", "Q4 2024"]), GRAIN_QUARTERLY)

    def test_quarterly_year_first(self):
        self.assertEqual(detect_period_grain(["2024 Q1", "2025 Q1"]), GRAIN_QUARTERLY)

    def test_quarterly_with_dash(self):
        self.assertEqual(detect_period_grain(["Q1-2024"]), GRAIN_QUARTERLY)

    def test_yearly(self):
        self.assertEqual(detect_period_grain(["2023", "2024", "2025"]), GRAIN_YEARLY)

    def test_unknown_returns_unknown(self):
        self.assertEqual(detect_period_grain(["Week 1", "Week 2"]), GRAIN_UNKNOWN)

    def test_empty_labels(self):
        self.assertEqual(detect_period_grain([]), GRAIN_UNKNOWN)

    def test_uses_first_label(self):
        # Mixed labels — grain decided from the first recognisable one
        result = detect_period_grain(["2024-03", "not-a-date", "2024-05"])
        self.assertEqual(result, GRAIN_MONTHLY)


# ══════════════════════════════════════════════════════════════════════════════
# compute_prior_period
# ══════════════════════════════════════════════════════════════════════════════

class ComputePriorPeriodMonthlyTests(unittest.TestCase):

    def test_single_month_window(self):
        """One month window → shift back 1 month."""
        pf, pl = compute_prior_period("2025-03", "2025-03", GRAIN_MONTHLY)
        self.assertEqual(pf, "2025-02")
        self.assertEqual(pl, "2025-02")

    def test_three_month_window(self):
        """3-month window (Jan-Mar) → prior = Oct-Dec of previous year."""
        pf, pl = compute_prior_period("2025-01", "2025-03", GRAIN_MONTHLY)
        self.assertEqual(pf, "2024-10")
        self.assertEqual(pl, "2024-12")

    def test_six_month_window(self):
        """6-month window → shift back 6 months."""
        pf, pl = compute_prior_period("2025-01", "2025-06", GRAIN_MONTHLY)
        self.assertEqual(pf, "2024-07")
        self.assertEqual(pl, "2024-12")

    def test_twelve_month_window(self):
        """Full-year monthly window → prior is the previous full year."""
        pf, pl = compute_prior_period("2024-01", "2024-12", GRAIN_MONTHLY)
        self.assertEqual(pf, "2023-01")
        self.assertEqual(pl, "2023-12")

    def test_crosses_year_boundary_back(self):
        """Window ending in Dec → prior window ends in prior Dec."""
        pf, pl = compute_prior_period("2024-10", "2024-12", GRAIN_MONTHLY)
        self.assertEqual(pf, "2024-07")
        self.assertEqual(pl, "2024-09")

    def test_month_zero_padding(self):
        """Output months should be zero-padded to 2 digits."""
        pf, pl = compute_prior_period("2025-02", "2025-02", GRAIN_MONTHLY)
        self.assertRegex(pf, r"^\d{4}-\d{2}$")


class ComputePriorPeriodQuarterlyTests(unittest.TestCase):

    def test_single_quarter(self):
        pf, pl = compute_prior_period("Q2 2025", "Q2 2025", GRAIN_QUARTERLY)
        self.assertEqual(pf, "Q1 2025")
        self.assertEqual(pl, "Q1 2025")

    def test_two_quarter_window(self):
        """2-quarter window → shift back 2 quarters."""
        pf, pl = compute_prior_period("Q1 2025", "Q2 2025", GRAIN_QUARTERLY)
        self.assertEqual(pf, "Q3 2024")
        self.assertEqual(pl, "Q4 2024")

    def test_full_year_quarterly(self):
        """4-quarter year window → prior year."""
        pf, pl = compute_prior_period("Q1 2024", "Q4 2024", GRAIN_QUARTERLY)
        self.assertEqual(pf, "Q1 2023")
        self.assertEqual(pl, "Q4 2023")

    def test_crosses_year_q1(self):
        """Q1 single quarter → prior = Q4 of previous year."""
        pf, pl = compute_prior_period("Q1 2025", "Q1 2025", GRAIN_QUARTERLY)
        self.assertEqual(pf, "Q4 2024")
        self.assertEqual(pl, "Q4 2024")


class ComputePriorPeriodYearlyTests(unittest.TestCase):

    def test_single_year(self):
        pf, pl = compute_prior_period("2025", "2025", GRAIN_YEARLY)
        self.assertEqual(pf, "2024")
        self.assertEqual(pl, "2024")

    def test_two_year_window(self):
        pf, pl = compute_prior_period("2024", "2025", GRAIN_YEARLY)
        self.assertEqual(pf, "2022")
        self.assertEqual(pl, "2023")

    def test_three_year_window(self):
        pf, pl = compute_prior_period("2023", "2025", GRAIN_YEARLY)
        self.assertEqual(pf, "2020")
        self.assertEqual(pl, "2022")


class ComputePriorPeriodUnknownTests(unittest.TestCase):

    def test_unknown_grain_returns_empty(self):
        pf, pl = compute_prior_period("Week 1", "Week 6", GRAIN_UNKNOWN)
        self.assertEqual(pf, "")
        self.assertEqual(pl, "")


# ══════════════════════════════════════════════════════════════════════════════
# build_prior_period_rewrite_prompt
# ══════════════════════════════════════════════════════════════════════════════

class BuildRewritePromptTests(unittest.TestCase):

    def _build(self, sql="SELECT 1", hint=""):
        return build_prior_period_rewrite_prompt(
            sql, "2025-01", "2025-03", "2024-10", "2024-12", hint
        )

    def test_system_prompt_has_cannot_rewrite_sentinel(self):
        system, _ = self._build()
        self.assertIn("CANNOT_REWRITE", system)

    def test_system_prompt_restricts_to_date_filter_only(self):
        system, _ = self._build()
        self.assertIn("ONLY", system)

    def test_user_prompt_contains_original_sql(self):
        sql = "SELECT MONTH, SUM(AMT) FROM FCT WHERE YEAR=2025 GROUP BY MONTH"
        _, user = self._build(sql=sql)
        self.assertIn(sql, user)

    def test_user_prompt_contains_prior_period(self):
        _, user = self._build()
        self.assertIn("2024-10", user)
        self.assertIn("2024-12", user)

    def test_user_prompt_contains_current_period(self):
        _, user = self._build()
        self.assertIn("2025-01", user)
        self.assertIn("2025-03", user)

    def test_date_col_hint_included_when_provided(self):
        _, user = self._build(hint="fact FK: CUS_IVC_DT_DMS_KEY")
        self.assertIn("CUS_IVC_DT_DMS_KEY", user)

    def test_no_hint_block_when_empty(self):
        _, user = self._build(hint="")
        self.assertNotIn("hint", user.lower())


# ══════════════════════════════════════════════════════════════════════════════
# build_period_comparison_narrative_prompt
# ══════════════════════════════════════════════════════════════════════════════

def _full_numeric(total, min_=None, max_=None, mean=None, median=None):
    """Build a numeric_summaries entry with all keys _format_brief_for_prompt expects."""
    return {
        "total": total,
        "min":    min_ if min_ is not None else total * 0.8,
        "max":    max_ if max_ is not None else total * 1.2,
        "mean":   mean if mean is not None else total,
        "median": median if median is not None else total,
        "count":  3,
    }


def _full_ts(last_value, first_value, direction, first_period, last_period):
    """Build a complete time_series dict with all keys _format_brief_for_prompt reads."""
    return {
        "direction":         direction,
        "first_period":      first_period,
        "last_period":       last_period,
        "first_value":       first_value,
        "last_value":        last_value,
        "period_count":      3,
        "overall_pct_change": round((last_value - first_value) / first_value * 100, 1),
        "peak":  {"period": last_period,  "value": last_value},
        "trough": {"period": first_period, "value": first_value},
        "biggest_period_drop": {"from_period": first_period, "to_period": last_period,
                                "absolute_change": 0.0, "pct_change": 0.0},
        "biggest_period_gain": {"from_period": first_period, "to_period": last_period,
                                "absolute_change": last_value - first_value,
                                "pct_change": round((last_value - first_value) / first_value * 100, 1)},
        "volatility":              50.0,
        "half_comparison":         None,
        "longest_decline_streak":  0,
        "longest_increase_streak": 2,
    }


class BuildNarrativePromptTests(unittest.TestCase):

    _current_brief = {
        "row_count": 3,
        "mode": "time_series",
        "time_series": _full_ts(1200.0, 1000.0, "increasing", "Jan", "Mar"),
        "numeric_summaries": {"Revenue": _full_numeric(3600.0)},
    }
    _prior_brief = {
        "row_count": 3,
        "mode": "time_series",
        "time_series": _full_ts(1000.0, 950.0, "stable", "Oct", "Dec"),
        "numeric_summaries": {"Revenue": _full_numeric(3000.0)},
    }

    def test_system_contains_headline_format_requirement(self):
        system, _ = build_period_comparison_narrative_prompt(
            self._current_brief, self._prior_brief,
            "Show revenue by month", "Jan–Mar 2025", "Oct–Dec 2024"
        )
        self.assertIn("HEADLINE:", system)
        self.assertIn("BODY:", system)
        self.assertIn("NEXT:", system)

    def test_user_prompt_contains_both_labels(self):
        _, user = build_period_comparison_narrative_prompt(
            self._current_brief, self._prior_brief,
            "Show revenue by month", "Jan–Mar 2025", "Oct–Dec 2024"
        )
        self.assertIn("Jan–Mar 2025", user)
        self.assertIn("Oct–Dec 2024", user)

    def test_business_context_included_when_provided(self):
        system, _ = build_period_comparison_narrative_prompt(
            self._current_brief, self._prior_brief,
            "revenue", "2025-03", "2024-12",
            business_context="Revenue is net of returns.",
        )
        self.assertIn("Revenue is net of returns.", system)

    def test_no_business_context_block_when_empty(self):
        system, _ = build_period_comparison_narrative_prompt(
            self._current_brief, self._prior_brief,
            "revenue", "2025-03", "2024-12",
        )
        self.assertNotIn("BUSINESS CONTEXT:", system)


# ══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

class CleanSqlResponseTests(unittest.TestCase):

    def test_strips_markdown_fence(self):
        raw = "```sql\nSELECT 1\n```"
        self.assertEqual(_clean_sql_response(raw), "SELECT 1")

    def test_strips_unlabelled_fence(self):
        raw = "```\nSELECT 1\n```"
        self.assertEqual(_clean_sql_response(raw), "SELECT 1")

    def test_plain_sql_unchanged(self):
        sql = "SELECT MONTH, SUM(AMT) FROM FCT GROUP BY MONTH"
        self.assertEqual(_clean_sql_response(sql), sql)

    def test_strips_leading_trailing_whitespace(self):
        self.assertEqual(_clean_sql_response("  SELECT 1  "), "SELECT 1")


class ExtractDateColHintTests(unittest.TestCase):

    def test_returns_empty_when_no_plan(self):
        self.assertEqual(_extract_date_col_hint(None), "")

    def test_returns_empty_when_plan_disabled(self):
        self.assertEqual(_extract_date_col_hint({"enabled": False, "fields": []}), "")

    def test_extracts_date_dimension_field(self):
        plan = {
            "enabled": True,
            "fields": [
                {"role": "date_dimension", "source_key_column": "CUS_IVC_DT_DMS_KEY",
                 "column": "DT_DMS_KEY", "table": "PROFITABILITY.DT_DMS"},
            ],
        }
        hint = _extract_date_col_hint(plan)
        self.assertIn("CUS_IVC_DT_DMS_KEY", hint)
        self.assertIn("DT_DMS_KEY", hint)

    def test_skips_non_date_dimension_fields(self):
        plan = {
            "enabled": True,
            "fields": [
                {"role": "display_dimension", "column": "WHS_DSC"},
            ],
        }
        self.assertEqual(_extract_date_col_hint(plan), "")


# ══════════════════════════════════════════════════════════════════════════════
# Deterministic narrative fallback
# ══════════════════════════════════════════════════════════════════════════════

class DeterministicNarrativeTests(unittest.TestCase):

    _current = {"numeric_summaries": {"Revenue": {"total": 1200.0}}}
    _prior   = {"numeric_summaries": {"Revenue": {"total": 1000.0}}}

    def test_headline_contains_pct_change(self):
        headline = _deterministic_headline(
            self._current, self._prior, "Jan-Mar 2025", "Oct-Dec 2024"
        )
        self.assertIn("20.0%", headline)

    def test_headline_says_up_for_increase(self):
        headline = _deterministic_headline(
            self._current, self._prior, "Jan-Mar 2025", "Oct-Dec 2024"
        )
        self.assertIn("up", headline.lower())

    def test_headline_says_down_for_decrease(self):
        headline = _deterministic_headline(
            self._prior, self._current, "Oct-Dec 2024", "Jan-Mar 2025"
        )
        self.assertIn("down", headline.lower())

    def test_bullets_contain_both_period_totals(self):
        bullets = _deterministic_bullets(self._current, self._prior)
        text = " ".join(bullets)
        self.assertIn("1,200.00", text)
        self.assertIn("1,000.00", text)


# ══════════════════════════════════════════════════════════════════════════════
# _parse_narrative
# ══════════════════════════════════════════════════════════════════════════════

class ParseNarrativeTests(unittest.TestCase):

    _current = {"numeric_summaries": {"Rev": {"total": 500.0}}}
    _prior   = {"numeric_summaries": {"Rev": {"total": 400.0}}}

    def test_parses_structured_response(self):
        raw = (
            "HEADLINE: Revenue is up 25%\n"
            "BODY: The current period outperformed.\n"
            "DETAIL:\n- Current: 500\n- Prior: 400\n- Change: +100\n"
            "NEXT: Drill by warehouse."
        )
        result = _parse_narrative(raw, self._current, self._prior, "Mar", "Dec")
        self.assertEqual(result["headline"], "Revenue is up 25%")
        self.assertIn("Current: 500", result["bullets"])
        self.assertEqual(result["next_step"], "Drill by warehouse.")

    def test_fallback_headline_when_llm_empty(self):
        result = _parse_narrative("", self._current, self._prior, "Mar 2025", "Dec 2024")
        self.assertIn("%", result["headline"])  # deterministic pct change

    def test_fallback_bullets_when_llm_empty(self):
        result = _parse_narrative("", self._current, self._prior, "Mar", "Dec")
        self.assertTrue(len(result["bullets"]) >= 1)


# ══════════════════════════════════════════════════════════════════════════════
# generate_period_comparison — deterministic fallback paths only
# ══════════════════════════════════════════════════════════════════════════════

class GeneratePeriodComparisonFallbackTests(unittest.TestCase):
    """
    Tests that verify the fallback paths WITHOUT hitting the LLM or DB.
    The async function short-circuits early when the brief lacks time_series
    data or when the grain cannot be detected.
    """

    def _run(self, coro):
        return asyncio.run(coro)

    def test_returns_fallback_when_no_time_series_brief(self):
        from core.period_comparison import generate_period_comparison
        result = self._run(generate_period_comparison(
            rows=[{"Month": "2025-01", "Revenue": 100}],
            question="Show revenue by month",
            original_sql="SELECT 1",
            data_brief={"mode": "time_series"},   # no time_series key
            db_cfg={"db_type": "azure_sql"},
            provider="anthropic",
            model="claude-sonnet-4-6",
            api_key="dummy",
        ))
        self.assertEqual(result["type"], "assistant_analysis")
        self.assertEqual(result["action"], "compare_prior")
        self.assertIn("requires", result["body"].lower())

    def test_returns_fallback_when_grain_unknown(self):
        from core.period_comparison import generate_period_comparison
        result = self._run(generate_period_comparison(
            rows=[],
            question="Show revenue by week",
            original_sql="SELECT 1",
            data_brief={
                "mode": "time_series",
                "time_series": {
                    "first_period": "Week 1",
                    "last_period":  "Week 12",
                    "direction":    "increasing",
                    "period_count": 12,
                },
            },
            db_cfg={"db_type": "azure_sql"},
            provider="anthropic",
            model="claude-sonnet-4-6",
            api_key="dummy",
        ))
        self.assertEqual(result["action"], "compare_prior")
        # Weekly grain not supported → fallback with helpful message
        self.assertIn("format", result["body"].lower())


if __name__ == "__main__":
    unittest.main()
