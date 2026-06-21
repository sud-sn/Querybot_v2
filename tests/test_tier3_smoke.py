"""
tests/test_tier3_smoke.py
──────────────────────────
Tier 3 analytics smoke tests — pure Python, no DB, no network.

Covers:
  • Funnel Analysis       (detect + infer + compute)
  • Forecast              (detect + extract periods + infer + compute)
  • Fiscal Calendar       (detect + parse + date range)
  • Histogram             (detect + infer + compute)
  • Box Plot              (detect + infer + compute)
  • What-If Scenarios     (detect + parse + compute)
  • insight.detect_analytical_intents() Tier 3 keys
  • chart.detect_chart_type() new type signals
  • Business model coverage (retail, SaaS, HR, banking, healthcare, manufacturing)

Run: pytest tests/test_tier3_smoke.py -v
"""

import pytest


# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ══════════════════════════════════════════════════════════════════════════════

SALES_FUNNEL = [
    {"stage": "Leads",       "count": 10_000},
    {"stage": "Qualified",   "count": 4_000},
    {"stage": "Proposal",    "count": 1_600},
    {"stage": "Negotiation", "count": 800},
    {"stage": "Won",         "count": 320},
]

HR_FUNNEL = [
    {"hiring_stage": "Applied",      "candidates": 5_000},
    {"hiring_stage": "Screened",     "candidates": 1_500},
    {"hiring_stage": "Interviewed",  "candidates": 500},
    {"hiring_stage": "Offer Sent",   "candidates": 150},
    {"hiring_stage": "Hired",        "candidates": 80},
]

TIME_SERIES = [
    {"month": "2024-01", "revenue": 100_000},
    {"month": "2024-02", "revenue": 105_000},
    {"month": "2024-03", "revenue": 108_000},
    {"month": "2024-04", "revenue": 112_000},
    {"month": "2024-05", "revenue": 118_000},
    {"month": "2024-06", "revenue": 122_000},
    {"month": "2024-07", "revenue": 128_000},
    {"month": "2024-08", "revenue": 130_000},
]

SALARY_DIST = [
    {"employee_id": i, "salary": 60_000 + i * 1_000 + (i % 7) * 3_000}
    for i in range(50)
]

DEPT_SALARY = [
    {"department": "Engineering", "salary": s}
    for s in [90_000, 95_000, 100_000, 105_000, 110_000, 85_000, 120_000, 92_000]
] + [
    {"department": "Sales", "salary": s}
    for s in [65_000, 70_000, 68_000, 75_000, 72_000, 80_000, 60_000]
] + [
    {"department": "Marketing", "salary": s}
    for s in [75_000, 78_000, 80_000, 82_000, 76_000]
]

REVENUE_BY_REGION = [
    {"region": "North", "revenue": 1_200_000},
    {"region": "South", "revenue": 850_000},
    {"region": "East",  "revenue": 1_050_000},
    {"region": "West",  "revenue": 750_000},
]


# ══════════════════════════════════════════════════════════════════════════════
# Funnel Analysis — Detection
# ══════════════════════════════════════════════════════════════════════════════

class TestFunnelDetection:

    @pytest.mark.parametrize("question", [
        "Show the sales funnel from lead to close",
        "What is the conversion rate at each stage?",
        "Where are we losing leads in the pipeline?",
        "Show drop-off rates across the hiring funnel",
        "Funnel analysis of trial to paid conversions",
        "How many applicants make it to the offer stage?",
        "Sales pipeline stage conversion analysis",
        "Show top of funnel to bottom of funnel",
        "Visitor to lead to purchase conversion funnel",
    ])
    def test_funnel_detected(self, question):
        from core.funnel_analysis import detect_funnel_intent
        assert detect_funnel_intent(question), f"Expected funnel match: {question!r}"

    @pytest.mark.parametrize("question", [
        "What is total revenue this quarter?",
        "Show me all employees in engineering",
    ])
    def test_funnel_not_detected(self, question):
        from core.funnel_analysis import detect_funnel_intent
        assert not detect_funnel_intent(question), f"Unexpected funnel match: {question!r}"


# ══════════════════════════════════════════════════════════════════════════════
# Funnel Analysis — Column inference
# ══════════════════════════════════════════════════════════════════════════════

class TestFunnelColumnInference:

    def test_infers_stage_count_cols(self):
        from core.funnel_analysis import infer_funnel_cols
        s, c = infer_funnel_cols(SALES_FUNNEL)
        assert s == "stage"
        assert c == "count"

    def test_infers_hr_funnel_cols(self):
        from core.funnel_analysis import infer_funnel_cols
        s, c = infer_funnel_cols(HR_FUNNEL)
        assert "stage" in s.lower() or "hiring" in s.lower()
        assert c == "candidates"

    def test_empty_returns_empty(self):
        from core.funnel_analysis import infer_funnel_cols
        assert infer_funnel_cols([]) == ("", "")


# ══════════════════════════════════════════════════════════════════════════════
# Funnel Analysis — Computation
# ══════════════════════════════════════════════════════════════════════════════

class TestFunnelComputation:

    def test_top_stage_has_no_conversion_rate(self):
        from core.funnel_analysis import compute_funnel
        rows = compute_funnel(SALES_FUNNEL, "stage", "count")
        # Top of funnel (largest count) → conversion_rate is None
        top = rows[0]
        assert top["conversion_rate"] is None

    def test_conversion_rates_computed(self):
        from core.funnel_analysis import compute_funnel
        rows = compute_funnel(SALES_FUNNEL, "stage", "count")
        # Qualified / Leads = 4000/10000 = 40%
        qualified = next(r for r in rows if r["stage"] == "Qualified")
        assert qualified["conversion_rate"] == pytest.approx(40.0)

    def test_drop_off_computed(self):
        from core.funnel_analysis import compute_funnel
        rows = compute_funnel(SALES_FUNNEL, "stage", "count")
        qualified = next(r for r in rows if r["stage"] == "Qualified")
        assert qualified["drop_off"] == pytest.approx(6_000)

    def test_cumulative_conversion(self):
        from core.funnel_analysis import compute_funnel
        rows = compute_funnel(SALES_FUNNEL, "stage", "count")
        won = next(r for r in rows if r["stage"] == "Won")
        # 320 / 10000 = 3.2%
        assert won["cumulative_conversion"] == pytest.approx(3.2)

    def test_funnel_pct_top_is_100(self):
        from core.funnel_analysis import compute_funnel
        rows = compute_funnel(SALES_FUNNEL, "stage", "count")
        assert rows[0]["funnel_pct"] == pytest.approx(100.0)

    def test_all_rows_have_funnel_pct(self):
        from core.funnel_analysis import compute_funnel
        rows = compute_funnel(SALES_FUNNEL, "stage", "count")
        assert all(r.get("funnel_pct") is not None for r in rows)

    def test_funnel_summary(self):
        from core.funnel_analysis import build_funnel_summary
        summary = build_funnel_summary(SALES_FUNNEL, "stage", "count")
        assert summary.stage_count == 5
        assert summary.top_of_funnel == 10_000
        assert summary.bottom_of_funnel == 320
        assert summary.overall_conversion_pct == pytest.approx(3.2)

    def test_hr_funnel_computation(self):
        from core.funnel_analysis import compute_funnel, infer_funnel_cols
        s, c = infer_funnel_cols(HR_FUNNEL)
        rows = compute_funnel(HR_FUNNEL, s, c)
        assert rows[0]["funnel_pct"] == pytest.approx(100.0)
        assert all("conversion_rate" in r for r in rows)


# ══════════════════════════════════════════════════════════════════════════════
# Forecast — Detection
# ══════════════════════════════════════════════════════════════════════════════

class TestForecastDetection:

    @pytest.mark.parametrize("question", [
        "Forecast revenue for the next 3 months",
        "Predict sales for Q4 based on current trend",
        "Project headcount growth over the next 6 months",
        "What will MRR be in December?",
        "Show a trend projection for the next quarter",
        "Extrapolate the growth trend forward",
        "Growth rate projection for next year",
        "Estimate revenue for the next 4 periods",
    ])
    def test_forecast_detected(self, question):
        from core.forecast import detect_forecast_intent
        assert detect_forecast_intent(question), f"Expected forecast match: {question!r}"

    @pytest.mark.parametrize("question", [
        "Show revenue by region this year",
        "Top 10 products by sales",
    ])
    def test_forecast_not_detected(self, question):
        from core.forecast import detect_forecast_intent
        assert not detect_forecast_intent(question), f"Unexpected forecast match: {question!r}"

    @pytest.mark.parametrize("question,expected", [
        ("forecast next 3 months", 3),
        ("project the next 6 quarters", 6),
        ("predict for 12 months ahead", 12),
        ("what will happen next year", 3),   # default when no number
    ])
    def test_extract_periods(self, question, expected):
        from core.forecast import extract_forecast_periods
        assert extract_forecast_periods(question) == expected


# ══════════════════════════════════════════════════════════════════════════════
# Forecast — Computation
# ══════════════════════════════════════════════════════════════════════════════

class TestForecastComputation:

    def test_appends_n_forecast_rows(self):
        from core.forecast import compute_forecast
        rows = compute_forecast(TIME_SERIES, "month", "revenue", n_periods=3)
        assert len(rows) == len(TIME_SERIES) + 3

    def test_historical_rows_flagged_false(self):
        from core.forecast import compute_forecast
        rows = compute_forecast(TIME_SERIES, "month", "revenue", n_periods=3)
        hist = [r for r in rows if not r["is_forecast"]]
        assert len(hist) == len(TIME_SERIES)

    def test_forecast_rows_flagged_true(self):
        from core.forecast import compute_forecast
        rows = compute_forecast(TIME_SERIES, "month", "revenue", n_periods=3)
        fc = [r for r in rows if r["is_forecast"]]
        assert len(fc) == 3

    def test_forecast_values_are_positive(self):
        from core.forecast import compute_forecast
        rows = compute_forecast(TIME_SERIES, "month", "revenue", n_periods=3)
        fc = [r for r in rows if r["is_forecast"]]
        assert all(r["forecast_value"] > 0 for r in fc)

    def test_upward_trend_projects_upward(self):
        from core.forecast import compute_forecast
        rows = compute_forecast(TIME_SERIES, "month", "revenue", n_periods=3)
        last_actual = TIME_SERIES[-1]["revenue"]
        fc_values = [r["forecast_value"] for r in rows if r["is_forecast"]]
        assert fc_values[-1] > last_actual  # upward trend continues

    def test_period_labels_generated(self):
        from core.forecast import compute_forecast
        rows = compute_forecast(TIME_SERIES, "month", "revenue", n_periods=3)
        fc = [r for r in rows if r["is_forecast"]]
        # After 2024-08, should generate 2024-09, 2024-10, 2024-11
        assert fc[0]["month"] == "2024-09"
        assert fc[1]["month"] == "2024-10"
        assert fc[2]["month"] == "2024-11"

    def test_trend_metadata_on_first_row(self):
        from core.forecast import compute_forecast
        rows = compute_forecast(TIME_SERIES, "month", "revenue", n_periods=3)
        assert "__trend_slope" in rows[0]
        assert rows[0]["__trend_slope"] > 0  # growing series
        assert rows[0].get("__trend_r2") is not None

    def test_period_label_quarter_rollover(self):
        from core.forecast import _next_period_label
        assert _next_period_label("Q3 2024", 1) == "Q4 2024"
        assert _next_period_label("Q4 2024", 1) == "Q1 2025"

    def test_period_label_month_rollover(self):
        from core.forecast import _next_period_label
        assert _next_period_label("2024-11", 1) == "2024-12"
        assert _next_period_label("2024-12", 1) == "2025-01"

    def test_infer_forecast_cols(self):
        from core.forecast import infer_forecast_cols
        p, v = infer_forecast_cols(TIME_SERIES)
        assert p == "month"
        assert v == "revenue"

    def test_too_few_rows_returns_unchanged_with_flags(self):
        from core.forecast import compute_forecast
        rows = compute_forecast([{"month": "2024-01", "revenue": 100}], "month", "revenue", 3)
        # 1 row is < 2, so no regression, but flags still added
        assert all("is_forecast" in r for r in rows)


# ══════════════════════════════════════════════════════════════════════════════
# Fiscal Calendar — Detection
# ══════════════════════════════════════════════════════════════════════════════

class TestFiscalDetection:

    @pytest.mark.parametrize("question", [
        "Show revenue for FY2024",
        "Compare fiscal Q1 vs fiscal Q2",
        "What was performance in the last fiscal year?",
        "Revenue in FY24",
        "Fiscal year ending March 2024 results",
        "Show FQ2 results",
        "Current fiscal quarter performance",
        "Year ended June 2024 revenue",
        "Financial year comparison",
    ])
    def test_fiscal_detected(self, question):
        from core.fiscal_calendar import detect_fiscal_intent
        assert detect_fiscal_intent(question), f"Expected fiscal match: {question!r}"

    @pytest.mark.parametrize("question", [
        "Show revenue for 2024",
        "Monthly sales report",
    ])
    def test_fiscal_not_detected(self, question):
        from core.fiscal_calendar import detect_fiscal_intent
        assert not detect_fiscal_intent(question), f"Unexpected fiscal match: {question!r}"


# ══════════════════════════════════════════════════════════════════════════════
# Fiscal Calendar — Parsing and date ranges
# ══════════════════════════════════════════════════════════════════════════════

class TestFiscalParsing:

    def test_parse_fy2024(self):
        from core.fiscal_calendar import parse_fiscal_reference
        ref = parse_fiscal_reference("Show FY2024 revenue")
        assert ref.fy_year == 2024

    def test_parse_fy24_short(self):
        from core.fiscal_calendar import parse_fiscal_reference
        ref = parse_fiscal_reference("Show FY24 revenue")
        assert ref.fy_year == 2024

    def test_parse_fiscal_q2(self):
        from core.fiscal_calendar import parse_fiscal_reference
        ref = parse_fiscal_reference("Show fiscal Q2 results")
        assert ref.fy_quarter == 2
        assert ref.fy_type == "quarter"

    def test_parse_last_fiscal(self):
        from core.fiscal_calendar import parse_fiscal_reference
        ref = parse_fiscal_reference("Compare last fiscal year performance")
        assert ref.relative == "last"

    def test_calendar_year_range(self):
        from core.fiscal_calendar import parse_fiscal_reference, fiscal_date_range
        ref = parse_fiscal_reference("FY2024 revenue")
        start, end = fiscal_date_range(ref, fiscal_year_start_month=1, current_year=2026)
        assert start == "2024-01-01"
        assert end   == "2024-12-31"

    def test_april_fy_range(self):
        # UK/India fiscal year: April = month 4, FY2024 = Apr 2023 – Mar 2024
        from core.fiscal_calendar import parse_fiscal_reference, fiscal_date_range
        ref = parse_fiscal_reference("FY2024 revenue")
        start, end = fiscal_date_range(ref, fiscal_year_start_month=4, current_year=2026)
        assert start == "2023-04-01"
        assert end   == "2024-03-31"

    def test_july_fy_range(self):
        # ANZ fiscal year: July = month 7, FY2024 = Jul 2023 – Jun 2024
        from core.fiscal_calendar import parse_fiscal_reference, fiscal_date_range
        ref = parse_fiscal_reference("FY2024 results")
        start, end = fiscal_date_range(ref, fiscal_year_start_month=7, current_year=2026)
        assert start == "2023-07-01"
        assert end   == "2024-06-30"

    def test_fiscal_period_label(self):
        from core.fiscal_calendar import fiscal_period_label
        # July fiscal year: 2023-10 → FY2024 Q2
        lbl = fiscal_period_label("2023-10", fiscal_year_start_month=7)
        assert "FY2024" in lbl

    def test_sql_hint_contains_dateadd(self):
        from core.fiscal_calendar import build_fiscal_sql_hint
        hint = build_fiscal_sql_hint("FY2024 revenue", fiscal_year_start_month=7, db_type="azure_sql")
        assert "DATEADD" in hint or "YEAR" in hint
        assert "July" in hint or "month 7" in hint


# ══════════════════════════════════════════════════════════════════════════════
# Histogram — Detection
# ══════════════════════════════════════════════════════════════════════════════

class TestHistogramDetection:

    @pytest.mark.parametrize("question", [
        "Show histogram of order values",
        "What is the distribution of salaries?",
        "Frequency distribution of customer ages",
        "How are loan amounts distributed?",
        "Show the spread of processing times",
        "Bin transaction amounts into ranges",
    ])
    def test_histogram_detected(self, question):
        from core.distribution_analysis import detect_histogram_intent
        assert detect_histogram_intent(question), f"Expected histogram match: {question!r}"

    def test_histogram_not_detected(self):
        from core.distribution_analysis import detect_histogram_intent
        assert not detect_histogram_intent("Show total revenue by region")


# ══════════════════════════════════════════════════════════════════════════════
# Histogram — Computation
# ══════════════════════════════════════════════════════════════════════════════

class TestHistogramComputation:

    def test_returns_n_bins(self):
        from core.distribution_analysis import compute_histogram
        result = compute_histogram(SALARY_DIST, "salary", n_bins=10)
        assert len(result) == 10

    def test_counts_sum_to_total(self):
        from core.distribution_analysis import compute_histogram
        result = compute_histogram(SALARY_DIST, "salary", n_bins=10)
        assert sum(r["count"] for r in result) == len(SALARY_DIST)

    def test_frequency_pct_sums_to_100(self):
        from core.distribution_analysis import compute_histogram
        result = compute_histogram(SALARY_DIST, "salary", n_bins=10)
        assert sum(r["frequency_pct"] for r in result) == pytest.approx(100.0, abs=0.1)

    def test_bin_label_present(self):
        from core.distribution_analysis import compute_histogram
        result = compute_histogram(SALARY_DIST, "salary", n_bins=5)
        assert all("bin_label" in r for r in result)
        assert all("–" in r["bin_label"] for r in result)

    def test_infer_histogram_col(self):
        from core.distribution_analysis import infer_histogram_col
        col = infer_histogram_col(SALARY_DIST)
        assert col == "salary"

    def test_empty_returns_empty(self):
        from core.distribution_analysis import compute_histogram
        assert compute_histogram([], "salary") == []

    def test_single_value_edge_case(self):
        from core.distribution_analysis import compute_histogram
        rows = [{"x": 100} for _ in range(5)]
        result = compute_histogram(rows, "x")
        assert len(result) == 1
        assert result[0]["count"] == 5


# ══════════════════════════════════════════════════════════════════════════════
# Box Plot — Detection
# ══════════════════════════════════════════════════════════════════════════════

class TestBoxplotDetection:

    @pytest.mark.parametrize("question", [
        "Show a box plot of salary by department",
        "Quartile breakdown of revenue by region",
        "What is the spread of processing times by team?",
        "Box and whisker chart of order values",
        "Show the IQR of customer lifetime value",
        "Median salary by department",
        "Show p25 and p75 of transaction amounts",
        "Outliers by region in processing time",
    ])
    def test_boxplot_detected(self, question):
        from core.distribution_analysis import detect_boxplot_intent
        assert detect_boxplot_intent(question), f"Expected boxplot match: {question!r}"

    def test_boxplot_not_detected(self):
        from core.distribution_analysis import detect_boxplot_intent
        assert not detect_boxplot_intent("Show total revenue by region")


# ══════════════════════════════════════════════════════════════════════════════
# Box Plot — Computation
# ══════════════════════════════════════════════════════════════════════════════

class TestBoxplotComputation:

    def test_returns_one_row_per_group(self):
        from core.distribution_analysis import compute_boxplot
        result = compute_boxplot(DEPT_SALARY, "salary", "department")
        assert len(result) == 3  # Engineering, Sales, Marketing

    def test_quartile_order(self):
        from core.distribution_analysis import compute_boxplot
        result = compute_boxplot(DEPT_SALARY, "salary", "department")
        for row in result:
            assert row["bp_q1"] <= row["bp_median"] <= row["bp_q3"]
            assert row["bp_min"] <= row["bp_q1"]
            assert row["bp_q3"] <= row["bp_max"]

    def test_bp_data_has_5_elements(self):
        from core.distribution_analysis import compute_boxplot
        result = compute_boxplot(DEPT_SALARY, "salary", "department")
        for row in result:
            assert len(row["bp_data"]) == 5

    def test_engineering_higher_than_sales(self):
        from core.distribution_analysis import compute_boxplot
        result = compute_boxplot(DEPT_SALARY, "salary", "department")
        eng   = next(r for r in result if r["group"] == "Engineering")
        sales = next(r for r in result if r["group"] == "Sales")
        assert eng["bp_median"] > sales["bp_median"]

    def test_no_group_returns_single_all_row(self):
        from core.distribution_analysis import compute_boxplot
        simple = [{"val": v} for v in [10, 20, 30, 40, 50]]
        result = compute_boxplot(simple, "val")
        assert len(result) == 1
        assert result[0]["group"] == "All"

    def test_outlier_detection(self):
        from core.distribution_analysis import compute_boxplot
        # Add extreme outlier
        rows = [{"g": "A", "val": v} for v in [100, 101, 102, 103, 104, 500]]
        result = compute_boxplot(rows, "val", "g")
        assert 500 in result[0]["bp_outliers"]

    def test_infer_boxplot_cols(self):
        from core.distribution_analysis import infer_boxplot_cols
        g, v = infer_boxplot_cols(DEPT_SALARY)
        assert g == "department"
        assert v == "salary"

    def test_known_quartiles(self):
        from core.distribution_analysis import compute_boxplot
        rows = [{"g": "X", "v": float(i)} for i in range(1, 6)]  # [1,2,3,4,5]
        result = compute_boxplot(rows, "v", "g")
        assert result[0]["bp_median"] == pytest.approx(3.0)
        assert result[0]["bp_q1"]     == pytest.approx(2.0, abs=0.5)
        assert result[0]["bp_q3"]     == pytest.approx(4.0, abs=0.5)


# ══════════════════════════════════════════════════════════════════════════════
# What-If Scenarios — Detection
# ══════════════════════════════════════════════════════════════════════════════

class TestWhatIfDetection:

    @pytest.mark.parametrize("question", [
        "What if revenue grows by 10%?",
        "Show the impact of a 5% price increase",
        "If we reduce cost by 15%, what is the margin?",
        "Scenario: revenue +$500K",
        "What would profit be if headcount drops by 20%?",
        "Simulate the impact of a 10% churn reduction",
        "Sensitivity analysis of revenue to price changes",
        "If sales increase by 25%, what happens to profit?",
        "Optimistic scenario with 15% revenue growth",
    ])
    def test_whatif_detected(self, question):
        from core.whatif import detect_whatif_intent
        assert detect_whatif_intent(question), f"Expected what-if match: {question!r}"

    @pytest.mark.parametrize("question", [
        "What is total revenue this quarter?",
        "Show top products by margin",
    ])
    def test_whatif_not_detected(self, question):
        from core.whatif import detect_whatif_intent
        assert not detect_whatif_intent(question), f"Unexpected what-if match: {question!r}"


# ══════════════════════════════════════════════════════════════════════════════
# What-If Scenarios — Parameter parsing
# ══════════════════════════════════════════════════════════════════════════════

class TestWhatIfParsing:

    def test_parse_positive_pct(self):
        from core.whatif import parse_whatif_params
        p = parse_whatif_params("What if revenue grows by 10%?")
        assert p.delta_pct == pytest.approx(10.0)

    def test_parse_negative_pct(self):
        from core.whatif import parse_whatif_params
        p = parse_whatif_params("If cost reduces by 15%")
        assert p.delta_pct == pytest.approx(-15.0)

    def test_parse_decrease_pct(self):
        from core.whatif import parse_whatif_params
        p = parse_whatif_params("If headcount decreases by 20%")
        assert p.delta_pct == pytest.approx(-20.0)

    def test_parse_col_hint_revenue(self):
        from core.whatif import parse_whatif_params
        p = parse_whatif_params("What if revenue grows by 10%?")
        assert p.col_hint == "revenue"

    def test_parse_col_hint_cost(self):
        from core.whatif import parse_whatif_params
        p = parse_whatif_params("If cost drops by 5%")
        assert p.col_hint == "cost" or p.col_hint == "costs"

    def test_scenario_label_generated(self):
        from core.whatif import parse_whatif_params
        p = parse_whatif_params("What if revenue grows by 10%?")
        assert "+10%" in p.scenario_label or "10%" in p.scenario_label

    def test_no_pct_returns_none(self):
        from core.whatif import parse_whatif_params
        p = parse_whatif_params("What if revenue changes?")
        assert p.delta_pct is None
        assert p.delta_abs is None


# ══════════════════════════════════════════════════════════════════════════════
# What-If Scenarios — Computation
# ══════════════════════════════════════════════════════════════════════════════

class TestWhatIfComputation:

    def test_pct_increase_applied(self):
        from core.whatif import parse_whatif_params, compute_whatif
        params = parse_whatif_params("What if revenue grows by 10%?")
        rows = compute_whatif(REVENUE_BY_REGION, params, target_col="revenue")
        north = next(r for r in rows if r["region"] == "North")
        assert north["scenario_revenue"] == pytest.approx(1_320_000)

    def test_pct_decrease_applied(self):
        from core.whatif import parse_whatif_params, compute_whatif
        params = parse_whatif_params("If revenue drops by 20%")
        rows = compute_whatif(REVENUE_BY_REGION, params, target_col="revenue")
        north = next(r for r in rows if r["region"] == "North")
        assert north["scenario_revenue"] == pytest.approx(960_000)

    def test_all_rows_have_scenario_col(self):
        from core.whatif import parse_whatif_params, compute_whatif
        params = parse_whatif_params("What if revenue grows by 5%?")
        rows = compute_whatif(REVENUE_BY_REGION, params, target_col="revenue")
        assert all("scenario_revenue" in r for r in rows)

    def test_delta_and_delta_pct_present(self):
        from core.whatif import parse_whatif_params, compute_whatif
        params = parse_whatif_params("What if revenue grows by 10%?")
        rows = compute_whatif(REVENUE_BY_REGION, params, target_col="revenue")
        assert all("scenario_delta" in r for r in rows)
        assert all("scenario_delta_pct" in r for r in rows)

    def test_delta_pct_matches_input(self):
        from core.whatif import parse_whatif_params, compute_whatif
        params = parse_whatif_params("What if revenue grows by 10%?")
        rows = compute_whatif(REVENUE_BY_REGION, params, target_col="revenue")
        for r in rows:
            assert r["scenario_delta_pct"] == pytest.approx(10.0)

    def test_original_values_unchanged(self):
        from core.whatif import parse_whatif_params, compute_whatif
        params = parse_whatif_params("What if revenue grows by 10%?")
        rows = compute_whatif(REVENUE_BY_REGION, params, target_col="revenue")
        assert rows[0]["revenue"] == 1_200_000  # original intact

    def test_scenario_label_attached(self):
        from core.whatif import parse_whatif_params, compute_whatif
        params = parse_whatif_params("What if revenue grows by 10%?")
        rows = compute_whatif(REVENUE_BY_REGION, params, target_col="revenue")
        assert all(r.get("__scenario_label") for r in rows)

    def test_col_inferred_from_hint(self):
        from core.whatif import parse_whatif_params, compute_whatif
        params = parse_whatif_params("What if revenue grows by 10%?")
        # Don't specify target_col — let it infer from col_hint
        rows = compute_whatif(REVENUE_BY_REGION, params)
        # Should still find revenue column from hint
        assert any(k.startswith("scenario_") for k in rows[0].keys())


# ══════════════════════════════════════════════════════════════════════════════
# chart.detect_chart_type() structural signals
# ══════════════════════════════════════════════════════════════════════════════

class TestChartTypeStructuralSignals:

    def test_funnel_pct_rows_return_funnel(self):
        from core.chart import detect_chart_type
        rows = [{"stage": "Lead", "count": 100, "funnel_pct": 100.0}]
        assert detect_chart_type(rows) == "funnel"

    def test_is_forecast_rows_return_forecast(self):
        from core.chart import detect_chart_type
        rows = [{"month": "2024-01", "revenue": 100_000, "is_forecast": False}]
        assert detect_chart_type(rows) == "forecast"

    def test_bp_data_rows_return_boxplot(self):
        from core.chart import detect_chart_type
        rows = [{"group": "A", "bp_data": [10, 20, 30, 40, 50]}]
        assert detect_chart_type(rows) == "boxplot"

    def test_bin_label_rows_return_histogram(self):
        from core.chart import detect_chart_type
        rows = [{"bin_label": "10K – 20K", "count": 15, "frequency_pct": 30.0}]
        assert detect_chart_type(rows) == "histogram"


# ══════════════════════════════════════════════════════════════════════════════
# insight.detect_analytical_intents() Tier 3 keys
# ══════════════════════════════════════════════════════════════════════════════

class TestUnifiedIntentTier3:

    def test_funnel_key_present(self):
        from core.insight import detect_analytical_intents
        result = detect_analytical_intents("Show the sales funnel conversion rates")
        assert "funnel" in result
        assert result["funnel"] is True

    def test_forecast_key_present(self):
        from core.insight import detect_analytical_intents
        result = detect_analytical_intents("Forecast revenue for the next 3 months")
        assert "forecast" in result
        assert result["forecast"] is True

    def test_fiscal_key_present(self):
        from core.insight import detect_analytical_intents
        result = detect_analytical_intents("Show FY2024 revenue by region")
        assert "fiscal" in result
        assert result["fiscal"] is True

    def test_histogram_key_present(self):
        from core.insight import detect_analytical_intents
        result = detect_analytical_intents("Show the distribution of salaries")
        assert "histogram" in result
        assert result["histogram"] is True

    def test_boxplot_key_present(self):
        from core.insight import detect_analytical_intents
        result = detect_analytical_intents("Box plot of revenue by region")
        assert "boxplot" in result
        assert result["boxplot"] is True

    def test_whatif_key_present(self):
        from core.insight import detect_analytical_intents
        result = detect_analytical_intents("What if revenue grows by 10%?")
        assert "whatif" in result
        assert result["whatif"] is True

    def test_all_tier3_keys_in_return(self):
        from core.insight import detect_analytical_intents
        result = detect_analytical_intents("what is total revenue?")
        tier3_keys = ("funnel", "forecast", "fiscal", "histogram", "boxplot", "whatif")
        assert all(k in result for k in tier3_keys)

    def test_tier1_and_tier2_still_present(self):
        from core.insight import detect_analytical_intents
        result = detect_analytical_intents("Show 3-month rolling average of revenue")
        all_expected = (
            "window", "relative_date", "contribution", "anomaly", "multi_period",
            "budget_vs_actual", "cohort", "correlation", "pivot",
            "funnel", "forecast", "fiscal", "histogram", "boxplot", "whatif",
        )
        assert all(k in result for k in all_expected)


# ══════════════════════════════════════════════════════════════════════════════
# Business model coverage — all 6 Tier 3 intents
# ══════════════════════════════════════════════════════════════════════════════

BUSINESS_CASES_T3 = {
    "retail": [
        ("funnel",    "Show the retail funnel from store visit to purchase"),
        ("forecast",  "Forecast retail sales for the next 3 months"),
        ("histogram", "Distribution of transaction amounts"),
        ("whatif",    "What if we increase prices by 5%?"),
    ],
    "saas": [
        ("funnel",    "Trial to paid conversion funnel"),
        ("forecast",  "Project MRR for the next 6 months"),
        ("histogram", "Distribution of customer contract values"),
        ("whatif",    "What if churn reduces by 10%?"),
    ],
    "hr": [
        ("funnel",    "Show the hiring funnel from applicant to offer"),
        ("forecast",  "Predict headcount growth over the next quarter"),
        ("boxplot",   "Box plot of salary by department"),
        ("whatif",    "What if headcount decreases by 20%?"),
    ],
    "banking": [
        ("funnel",    "Loan application to approval funnel"),
        ("forecast",  "Forecast loan originations for the next 3 months"),
        ("histogram", "Distribution of loan amounts"),
        ("fiscal",    "Revenue for FY2024 by branch"),
    ],
    "healthcare": [
        ("funnel",    "Patient pathway funnel from referral to discharge"),
        ("forecast",  "Project patient volumes for the next quarter"),
        ("boxplot",   "Quartile breakdown of wait times by department"),
        ("fiscal",    "FY2024 budget vs actual for each ward"),
    ],
    "manufacturing": [
        ("funnel",    "Show the production funnel from raw material to shipped"),
        ("forecast",  "Predict output volumes for the next 6 months"),
        ("histogram", "Distribution of defect rates across batches"),
        ("whatif",    "What if machine uptime improves by 15%?"),
    ],
}

_T3_DETECTOR_MAP = {
    "funnel":    ("core.funnel_analysis",       "detect_funnel_intent"),
    "forecast":  ("core.forecast",              "detect_forecast_intent"),
    "fiscal":    ("core.fiscal_calendar",       "detect_fiscal_intent"),
    "histogram": ("core.distribution_analysis", "detect_histogram_intent"),
    "boxplot":   ("core.distribution_analysis", "detect_boxplot_intent"),
    "whatif":    ("core.whatif",               "detect_whatif_intent"),
}


def _load(module_path: str, fn_name: str):
    import importlib
    return getattr(importlib.import_module(module_path), fn_name)


@pytest.mark.parametrize("domain,cases", list(BUSINESS_CASES_T3.items()))
def test_business_model_tier3_coverage(domain, cases):
    """Every Tier 3 intent must fire for representative questions across all domains."""
    for intent_key, question in cases:
        mod_path, fn_name = _T3_DETECTOR_MAP[intent_key]
        detector = _load(mod_path, fn_name)
        assert detector(question), (
            f"[{domain}] {intent_key} detector missed: {question!r}"
        )
