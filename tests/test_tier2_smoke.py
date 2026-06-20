"""
tests/test_tier2_smoke.py
─────────────────────────
Tier 2 analytics smoke tests — pure Python, no DB, no network.

Covers:
  • Budget vs Actual (detect + compute + infer cols)
  • Cohort Analysis (detect + compute matrix + summary)
  • Correlation Analysis (detect + Pearson r + col inference)
  • Pivot Table (detect + compute + col inference)
  • insight.detect_analytical_intents() Tier 2 keys
  • Business-model coverage (retail, SaaS, HR, banking, healthcare, manufacturing)

Run: pytest tests/test_tier2_smoke.py -v
"""

import pytest

# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ══════════════════════════════════════════════════════════════════════════════

RETAIL_BVA = [
    {"region": "North",  "actual_revenue": 120_000, "budget_revenue": 100_000},
    {"region": "South",  "actual_revenue": 85_000,  "budget_revenue": 100_000},
    {"region": "East",   "actual_revenue": 100_500, "budget_revenue": 100_000},
    {"region": "West",   "actual_revenue": 60_000,  "budget_revenue": 100_000},
]

SAAS_BVA = [
    {"product": "Pro",      "actual_arr": 500_000, "target_arr": 480_000},
    {"product": "Business", "actual_arr": 300_000, "target_arr": 400_000},
    {"product": "Free",     "actual_arr": 10_000,  "target_arr": 12_000},
]

HR_BUDGET = [
    {"department": "Eng",       "actual_headcount_cost": 2_000_000, "planned_headcount_cost": 1_800_000},
    {"department": "Sales",     "actual_headcount_cost": 1_200_000, "planned_headcount_cost": 1_500_000},
    {"department": "Marketing", "actual_headcount_cost": 800_000,   "planned_headcount_cost": 750_000},
]

COHORT_DATA = [
    {"cohort_month": "2024-01", "period_offset": 0, "retained_count": 1000},
    {"cohort_month": "2024-01", "period_offset": 1, "retained_count": 820},
    {"cohort_month": "2024-01", "period_offset": 2, "retained_count": 650},
    {"cohort_month": "2024-01", "period_offset": 3, "retained_count": 520},
    {"cohort_month": "2024-02", "period_offset": 0, "retained_count": 1200},
    {"cohort_month": "2024-02", "period_offset": 1, "retained_count": 960},
    {"cohort_month": "2024-02", "period_offset": 2, "retained_count": 750},
    {"cohort_month": "2024-02", "period_offset": 3, "retained_count": 580},
    {"cohort_month": "2024-03", "period_offset": 0, "retained_count": 900},
    {"cohort_month": "2024-03", "period_offset": 1, "retained_count": 720},
    {"cohort_month": "2024-03", "period_offset": 2, "retained_count": 540},
]

CORR_DATA = [
    {"product": "A", "marketing_spend": 10_000, "revenue": 85_000},
    {"product": "B", "marketing_spend": 25_000, "revenue": 200_000},
    {"product": "C", "marketing_spend": 5_000,  "revenue": 42_000},
    {"product": "D", "marketing_spend": 40_000, "revenue": 330_000},
    {"product": "E", "marketing_spend": 15_000, "revenue": 130_000},
    {"product": "F", "marketing_spend": 30_000, "revenue": 250_000},
    {"product": "G", "marketing_spend": 8_000,  "revenue": 65_000},
    {"product": "H", "marketing_spend": 20_000, "revenue": 170_000},
]

PIVOT_DATA = [
    {"region": "EMEA", "quarter": "Q1", "revenue": 100_000},
    {"region": "EMEA", "quarter": "Q2", "revenue": 120_000},
    {"region": "EMEA", "quarter": "Q3", "revenue": 115_000},
    {"region": "APAC", "quarter": "Q1", "revenue": 80_000},
    {"region": "APAC", "quarter": "Q2", "revenue": 90_000},
    {"region": "APAC", "quarter": "Q3", "revenue": 95_000},
    {"region": "AMER", "quarter": "Q1", "revenue": 200_000},
    {"region": "AMER", "quarter": "Q2", "revenue": 210_000},
    {"region": "AMER", "quarter": "Q3", "revenue": 225_000},
]


# ══════════════════════════════════════════════════════════════════════════════
# Budget vs Actual — Detection
# ══════════════════════════════════════════════════════════════════════════════

class TestBvaDetection:
    """detect_bva_intent() must fire on the right phrases and not on others."""

    from core.budget_vs_actual import detect_bva_intent

    @pytest.mark.parametrize("question", [
        "How does actual revenue compare vs budget this month?",
        "Show me actuals vs target by department",
        "Which regions are over budget?",
        "Which products are under target?",
        "Variance to plan by department",
        "Budget variance report for Q3",
        "Are we on track vs forecast?",
        "Show the spending overrun by category",
        "What is the shortfall vs plan?",
        "Planned vs actual headcount cost",
    ])
    def test_bva_detected(self, question):
        from core.budget_vs_actual import detect_bva_intent
        assert detect_bva_intent(question), f"Expected BvA match: {question!r}"

    @pytest.mark.parametrize("question", [
        "What is total revenue this month?",
        "Show me top 5 products by sales",
        "Which employees were hired in Q1?",
    ])
    def test_bva_not_detected(self, question):
        from core.budget_vs_actual import detect_bva_intent
        assert not detect_bva_intent(question), f"Unexpected BvA match: {question!r}"


# ══════════════════════════════════════════════════════════════════════════════
# Budget vs Actual — Column inference
# ══════════════════════════════════════════════════════════════════════════════

class TestBvaColumnInference:

    def test_infers_actual_budget_by_name_retail(self):
        from core.budget_vs_actual import infer_bva_cols
        a, b = infer_bva_cols(RETAIL_BVA)
        assert "actual" in a.lower()
        assert "budget" in b.lower()

    def test_infers_actual_target_saas(self):
        from core.budget_vs_actual import infer_bva_cols
        a, b = infer_bva_cols(SAAS_BVA)
        assert "actual" in a.lower()
        assert "target" in b.lower()

    def test_infers_actual_plan_hr(self):
        from core.budget_vs_actual import infer_bva_cols
        a, b = infer_bva_cols(HR_BUDGET)
        assert "actual" in a.lower()
        assert "planned" in b.lower()

    def test_empty_rows_returns_empty(self):
        from core.budget_vs_actual import infer_bva_cols
        assert infer_bva_cols([]) == ("", "")


# ══════════════════════════════════════════════════════════════════════════════
# Budget vs Actual — Computation
# ══════════════════════════════════════════════════════════════════════════════

class TestBvaComputation:

    def test_variance_sign_positive(self):
        from core.budget_vs_actual import compute_bva
        rows = compute_bva(RETAIL_BVA, "actual_revenue", "budget_revenue")
        north = next(r for r in rows if r["region"] == "North")
        assert north["variance"] == pytest.approx(20_000)
        assert north["variance_pct"] == pytest.approx(20.0)
        assert north["bva_status"] == "over"

    def test_variance_sign_negative(self):
        from core.budget_vs_actual import compute_bva
        rows = compute_bva(RETAIL_BVA, "actual_revenue", "budget_revenue")
        south = next(r for r in rows if r["region"] == "South")
        assert south["variance"] == pytest.approx(-15_000)
        assert south["bva_status"] == "under"

    def test_on_target_within_2pct(self):
        from core.budget_vs_actual import compute_bva
        rows = compute_bva(RETAIL_BVA, "actual_revenue", "budget_revenue")
        east = next(r for r in rows if r["region"] == "East")
        # 100_500 / 100_000 = 0.5% over → on_target
        assert east["bva_status"] == "on_target"

    def test_all_rows_have_status(self):
        from core.budget_vs_actual import compute_bva
        rows = compute_bva(RETAIL_BVA, "actual_revenue", "budget_revenue")
        assert len(rows) == 4
        assert all("bva_status" in r for r in rows)

    def test_saas_arr_variance(self):
        from core.budget_vs_actual import compute_bva
        rows = compute_bva(SAAS_BVA, "actual_arr", "target_arr")
        pro = next(r for r in rows if r["product"] == "Pro")
        assert pro["variance"] == pytest.approx(20_000)
        biz = next(r for r in rows if r["product"] == "Business")
        assert biz["bva_status"] == "under"

    def test_hr_cost_variance(self):
        from core.budget_vs_actual import compute_bva
        rows = compute_bva(HR_BUDGET, "actual_headcount_cost", "planned_headcount_cost")
        eng = next(r for r in rows if r["department"] == "Eng")
        assert eng["bva_status"] == "over"
        sales = next(r for r in rows if r["department"] == "Sales")
        assert sales["bva_status"] == "under"

    def test_bva_summary(self):
        from core.budget_vs_actual import build_bva_summary
        summary = build_bva_summary(RETAIL_BVA, "actual_revenue", "budget_revenue", "region")
        assert summary.over_count >= 1
        assert summary.under_count >= 1
        assert summary.row_count == 4
        # North is biggest over; West is biggest miss
        assert summary.largest_over["label"] == "North"
        assert summary.largest_miss["label"] == "West"


# ══════════════════════════════════════════════════════════════════════════════
# Cohort Analysis — Detection
# ══════════════════════════════════════════════════════════════════════════════

class TestCohortDetection:

    @pytest.mark.parametrize("question", [
        "Show cohort retention analysis",
        "What is the month 1 retention rate for users who signed up in January?",
        "Customer retention by cohort",
        "Show churn by cohort",
        "How many users from each cohort are still active?",
        "Month 3 retention for the Q1 cohort",
        "User retention analysis by signup month",
        "Subscriber retention curve",
        "Months since sign-up for active users",
    ])
    def test_cohort_detected(self, question):
        from core.cohort_analysis import detect_cohort_intent
        assert detect_cohort_intent(question), f"Expected cohort match: {question!r}"

    @pytest.mark.parametrize("question", [
        "What are total monthly active users?",
        "Show revenue by month",
    ])
    def test_cohort_not_detected(self, question):
        from core.cohort_analysis import detect_cohort_intent
        assert not detect_cohort_intent(question), f"Unexpected cohort match: {question!r}"


# ══════════════════════════════════════════════════════════════════════════════
# Cohort Analysis — Column inference
# ══════════════════════════════════════════════════════════════════════════════

class TestCohortColumnInference:

    def test_infers_standard_cohort_cols(self):
        from core.cohort_analysis import infer_cohort_cols
        c, p, v = infer_cohort_cols(COHORT_DATA)
        assert "cohort" in c.lower()
        assert "period" in p.lower() or "offset" in p.lower()
        assert "count" in v.lower() or "retain" in v.lower()

    def test_empty_returns_empty(self):
        from core.cohort_analysis import infer_cohort_cols
        assert infer_cohort_cols([]) == ("", "", "")


# ══════════════════════════════════════════════════════════════════════════════
# Cohort Analysis — Matrix computation
# ══════════════════════════════════════════════════════════════════════════════

class TestCohortMatrix:

    def test_matrix_has_correct_cohort_count(self):
        from core.cohort_analysis import compute_cohort_matrix
        matrix = compute_cohort_matrix(COHORT_DATA, "cohort_month", "period_offset", "retained_count")
        assert len(matrix) == 3  # 2024-01, 2024-02, 2024-03

    def test_month0_is_100_pct(self):
        from core.cohort_analysis import compute_cohort_matrix
        matrix = compute_cohort_matrix(COHORT_DATA, "cohort_month", "period_offset", "retained_count")
        for row in matrix:
            assert row.get("Month 0") == pytest.approx(100.0), f"Month 0 not 100% for {row['cohort']}"

    def test_retention_decreases_over_time(self):
        from core.cohort_analysis import compute_cohort_matrix
        matrix = compute_cohort_matrix(COHORT_DATA, "cohort_month", "period_offset", "retained_count")
        jan = next(r for r in matrix if r["cohort"] == "2024-01")
        assert jan["Month 1"] < 100.0
        assert jan["Month 2"] < jan["Month 1"]
        assert jan["Month 3"] < jan["Month 2"]

    def test_retention_pct_values(self):
        from core.cohort_analysis import compute_cohort_matrix
        matrix = compute_cohort_matrix(COHORT_DATA, "cohort_month", "period_offset", "retained_count")
        jan = next(r for r in matrix if r["cohort"] == "2024-01")
        # 820/1000 = 82%
        assert jan["Month 1"] == pytest.approx(82.0)
        # 650/1000 = 65%
        assert jan["Month 2"] == pytest.approx(65.0)

    def test_cohort_size_stored(self):
        from core.cohort_analysis import compute_cohort_matrix
        matrix = compute_cohort_matrix(COHORT_DATA, "cohort_month", "period_offset", "retained_count")
        jan = next(r for r in matrix if r["cohort"] == "2024-01")
        assert jan["cohort_size"] == 1000

    def test_incomplete_cohort_has_none_for_missing_periods(self):
        from core.cohort_analysis import compute_cohort_matrix
        matrix = compute_cohort_matrix(COHORT_DATA, "cohort_month", "period_offset", "retained_count")
        # 2024-03 only has periods 0,1,2 — period 3 should be None
        mar = next(r for r in matrix if r["cohort"] == "2024-03")
        assert mar.get("Month 3") is None

    def test_abs_count_preserved(self):
        from core.cohort_analysis import compute_cohort_matrix
        matrix = compute_cohort_matrix(COHORT_DATA, "cohort_month", "period_offset", "retained_count")
        jan = next(r for r in matrix if r["cohort"] == "2024-01")
        assert jan.get("__abs_Month 1") == 820

    def test_summary_stats(self):
        from core.cohort_analysis import compute_cohort_matrix, build_cohort_summary
        matrix = compute_cohort_matrix(COHORT_DATA, "cohort_month", "period_offset", "retained_count")
        summary = build_cohort_summary(matrix)
        assert summary.cohort_count == 3
        assert summary.avg_month1_retention is not None
        # Feb cohort (1200→960 = 80%) vs Jan (82%) vs Mar (720/900 = 80%)
        assert 79 <= summary.avg_month1_retention <= 82


# ══════════════════════════════════════════════════════════════════════════════
# Correlation Analysis — Detection
# ══════════════════════════════════════════════════════════════════════════════

class TestCorrelationDetection:

    @pytest.mark.parametrize("question", [
        "Is there a correlation between marketing spend and revenue?",
        "Show the relationship between price and sales volume",
        "Scatter plot of units sold vs profit margin",
        "Do customer satisfaction scores correlate with repeat purchase rate?",
        "How strongly does ad spend influence revenue?",
        "Show a linear relationship between salary and tenure",
        "Positively correlated metrics in the dataset",
    ])
    def test_correlation_detected(self, question):
        from core.correlation_analysis import detect_correlation_intent
        assert detect_correlation_intent(question), f"Expected correlation match: {question!r}"

    @pytest.mark.parametrize("question", [
        "Show total revenue by region",
        "What is the top product this quarter?",
    ])
    def test_correlation_not_detected(self, question):
        from core.correlation_analysis import detect_correlation_intent
        assert not detect_correlation_intent(question), f"Unexpected correlation match: {question!r}"


# ══════════════════════════════════════════════════════════════════════════════
# Correlation Analysis — Pearson r computation
# ══════════════════════════════════════════════════════════════════════════════

class TestCorrelationComputation:

    def test_strong_positive_correlation(self):
        from core.correlation_analysis import compute_correlation
        result = compute_correlation(CORR_DATA, "marketing_spend", "revenue")
        assert result.pearson_r is not None
        assert result.pearson_r > 0.95
        assert "strong positive" in result.interpretation

    def test_r_squared_is_r_times_r(self):
        from core.correlation_analysis import compute_correlation
        result = compute_correlation(CORR_DATA, "marketing_spend", "revenue")
        assert result.r_squared == pytest.approx(result.pearson_r ** 2, abs=0.001)

    def test_n_matches_row_count(self):
        from core.correlation_analysis import compute_correlation
        result = compute_correlation(CORR_DATA, "marketing_spend", "revenue")
        assert result.n == len(CORR_DATA)

    def test_negative_correlation(self):
        from core.correlation_analysis import compute_correlation
        rows = [
            {"x": 1, "y": 100}, {"x": 2, "y": 80}, {"x": 3, "y": 60},
            {"x": 4, "y": 40},  {"x": 5, "y": 20},
        ]
        result = compute_correlation(rows, "x", "y")
        assert result.pearson_r is not None
        assert result.pearson_r < -0.99
        assert "negative" in result.interpretation

    def test_no_correlation(self):
        from core.correlation_analysis import compute_correlation
        rows = [
            {"x": 1, "y": 50}, {"x": 2, "y": 10}, {"x": 3, "y": 80},
            {"x": 4, "y": 30}, {"x": 5, "y": 60}, {"x": 6, "y": 20},
        ]
        result = compute_correlation(rows, "x", "y")
        assert result.pearson_r is not None
        assert abs(result.pearson_r) < 0.5

    def test_insufficient_data_returns_none_r(self):
        from core.correlation_analysis import compute_correlation
        rows = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
        result = compute_correlation(rows, "x", "y")
        assert result.pearson_r is None
        assert "insufficient" in result.interpretation

    def test_col_inference_picks_mentioned_cols(self):
        from core.correlation_analysis import infer_corr_cols
        x, y = infer_corr_cols(CORR_DATA, "correlation between marketing_spend and revenue")
        assert x == "marketing_spend"
        assert y == "revenue"

    def test_col_inference_fallback_to_first_two_numeric(self):
        from core.correlation_analysis import infer_corr_cols
        x, y = infer_corr_cols(CORR_DATA)
        assert x != ""
        assert y != ""
        assert x != y

    def test_annotate_adds_meta_to_first_row(self):
        from core.correlation_analysis import compute_correlation, annotate_rows_with_correlation
        result = compute_correlation(CORR_DATA, "marketing_spend", "revenue")
        annotated = annotate_rows_with_correlation(CORR_DATA, result)
        assert "__corr_r" in annotated[0]
        assert annotated[0]["__corr_label"] == result.interpretation
        # Other rows should not have the meta key
        assert "__corr_r" not in annotated[1]


# ══════════════════════════════════════════════════════════════════════════════
# Pivot Table — Detection
# ══════════════════════════════════════════════════════════════════════════════

class TestPivotDetection:

    @pytest.mark.parametrize("question", [
        "Pivot revenue by region across quarters",
        "Show a cross-tabulation of channel by segment",
        "Revenue matrix of department by year",
        "Cross-tab of product vs channel",
        "Reshape the data with months as columns",
        "Show a pivot table of region vs product",
    ])
    def test_pivot_detected(self, question):
        from core.pivot_table import detect_pivot_intent
        assert detect_pivot_intent(question), f"Expected pivot match: {question!r}"

    @pytest.mark.parametrize("question", [
        "Show total revenue by region",
        "Top 5 products this month",
    ])
    def test_pivot_not_detected(self, question):
        from core.pivot_table import detect_pivot_intent
        assert not detect_pivot_intent(question), f"Unexpected pivot match: {question!r}"


# ══════════════════════════════════════════════════════════════════════════════
# Pivot Table — Column inference
# ══════════════════════════════════════════════════════════════════════════════

class TestPivotColumnInference:

    def test_infers_three_cols(self):
        from core.pivot_table import infer_pivot_cols
        rk, ck, vk = infer_pivot_cols(PIVOT_DATA)
        assert rk == "region"
        assert ck == "quarter"
        assert vk == "revenue"

    def test_empty_returns_empty(self):
        from core.pivot_table import infer_pivot_cols
        assert infer_pivot_cols([]) == ("", "", "")

    def test_single_text_col_returns_empty(self):
        from core.pivot_table import infer_pivot_cols
        rows = [{"region": "A", "revenue": 100}]
        assert infer_pivot_cols(rows) == ("", "", "")  # only 1 text col


# ══════════════════════════════════════════════════════════════════════════════
# Pivot Table — Computation
# ══════════════════════════════════════════════════════════════════════════════

class TestPivotComputation:

    def test_pivot_row_count(self):
        from core.pivot_table import compute_pivot_table
        pivot = compute_pivot_table(PIVOT_DATA, "region", "quarter", "revenue")
        assert len(pivot) == 3  # EMEA, APAC, AMER

    def test_pivot_column_headers(self):
        from core.pivot_table import compute_pivot_table
        pivot = compute_pivot_table(PIVOT_DATA, "region", "quarter", "revenue")
        headers = set(pivot[0].keys())
        assert "Q1" in headers
        assert "Q2" in headers
        assert "Q3" in headers
        assert "TOTAL" in headers
        assert "region" in headers

    def test_pivot_cell_values(self):
        from core.pivot_table import compute_pivot_table
        pivot = compute_pivot_table(PIVOT_DATA, "region", "quarter", "revenue")
        emea = next(r for r in pivot if r["region"] == "EMEA")
        assert emea["Q1"] == pytest.approx(100_000)
        assert emea["Q2"] == pytest.approx(120_000)

    def test_pivot_total_column(self):
        from core.pivot_table import compute_pivot_table
        pivot = compute_pivot_table(PIVOT_DATA, "region", "quarter", "revenue")
        emea = next(r for r in pivot if r["region"] == "EMEA")
        assert emea["TOTAL"] == pytest.approx(335_000)

    def test_pivot_missing_cell_is_none(self):
        from core.pivot_table import compute_pivot_table
        # Add an extra quarter only for EMEA
        data = PIVOT_DATA + [{"region": "EMEA", "quarter": "Q4", "revenue": 130_000}]
        pivot = compute_pivot_table(data, "region", "quarter", "revenue")
        apac = next(r for r in pivot if r["region"] == "APAC")
        assert apac.get("Q4") is None

    def test_pivot_avg_aggregation(self):
        from core.pivot_table import compute_pivot_table
        # Double entries for EMEA Q1 to test avg
        data = PIVOT_DATA + [{"region": "EMEA", "quarter": "Q1", "revenue": 200_000}]
        pivot = compute_pivot_table(data, "region", "quarter", "revenue", agg="avg")
        emea = next(r for r in pivot if r["region"] == "EMEA")
        # (100_000 + 200_000) / 2 = 150_000
        assert emea["Q1"] == pytest.approx(150_000)

    def test_pivot_count_aggregation(self):
        from core.pivot_table import compute_pivot_table
        data = PIVOT_DATA + [{"region": "EMEA", "quarter": "Q1", "revenue": 50_000}]
        pivot = compute_pivot_table(data, "region", "quarter", "revenue", agg="count")
        emea = next(r for r in pivot if r["region"] == "EMEA")
        assert emea["Q1"] == pytest.approx(2.0)  # 2 EMEA Q1 rows

    def test_banking_pivot_transactions_by_channel(self):
        from core.pivot_table import compute_pivot_table, infer_pivot_cols
        data = [
            {"channel": "ATM",    "txn_type": "Withdrawal", "count": 500},
            {"channel": "ATM",    "txn_type": "Deposit",    "count": 200},
            {"channel": "Online", "txn_type": "Transfer",   "count": 1500},
            {"channel": "Online", "txn_type": "Deposit",    "count": 800},
            {"channel": "Branch", "txn_type": "Withdrawal", "count": 300},
            {"channel": "Branch", "txn_type": "Deposit",    "count": 450},
        ]
        rk, ck, vk = infer_pivot_cols(data)
        pivot = compute_pivot_table(data, rk, ck, vk)
        assert len(pivot) == 3


# ══════════════════════════════════════════════════════════════════════════════
# Unified intent dispatcher — Tier 2 keys
# ══════════════════════════════════════════════════════════════════════════════

class TestUnifiedIntentTier2:

    def test_bva_key_present(self):
        from core.insight import detect_analytical_intents
        result = detect_analytical_intents("Show actuals vs budget by department")
        assert "budget_vs_actual" in result
        assert result["budget_vs_actual"] is True

    def test_cohort_key_present(self):
        from core.insight import detect_analytical_intents
        result = detect_analytical_intents("Show cohort retention analysis")
        assert "cohort" in result
        assert result["cohort"] is True

    def test_correlation_key_present(self):
        from core.insight import detect_analytical_intents
        result = detect_analytical_intents("Is there a correlation between price and demand?")
        assert "correlation" in result
        assert result["correlation"] is True

    def test_pivot_key_present(self):
        from core.insight import detect_analytical_intents
        result = detect_analytical_intents("Pivot revenue by region across quarters")
        assert "pivot" in result
        assert result["pivot"] is True

    def test_tier1_keys_still_present(self):
        from core.insight import detect_analytical_intents
        result = detect_analytical_intents("Show rolling 3-month average of revenue")
        assert all(k in result for k in ("window", "relative_date", "contribution", "anomaly", "multi_period"))

    def test_all_tier2_keys_present_in_return(self):
        from core.insight import detect_analytical_intents
        result = detect_analytical_intents("what is total revenue?")
        tier2_keys = ("budget_vs_actual", "cohort", "correlation", "pivot")
        assert all(k in result for k in tier2_keys)

    def test_non_analytical_question_returns_falsy_tier2(self):
        from core.insight import detect_analytical_intents
        result = detect_analytical_intents("Who are our top customers?")
        # Not a BvA / cohort / correlation / pivot question
        assert not result["budget_vs_actual"]
        assert not result["cohort"]


# ══════════════════════════════════════════════════════════════════════════════
# Business model coverage — all 4 Tier 2 modules
# ══════════════════════════════════════════════════════════════════════════════

BUSINESS_CASES = {
    "retail": [
        ("bva",         "How does actual sales compare vs budget by store?"),
        ("correlation", "Is there a correlation between promotions spend and footfall?"),
        ("pivot",       "Pivot sales by category across seasons"),
    ],
    "saas": [
        ("bva",         "Show ARR actuals vs target by product tier"),
        ("cohort",      "Customer retention by cohort — monthly"),
        ("correlation", "Is NPS correlated with renewal rate?"),
        ("pivot",       "Pivot MRR by plan across months"),
    ],
    "hr": [
        ("bva",         "Actual vs planned headcount cost by department"),
        ("cohort",      "Employee retention by hiring cohort"),
        ("pivot",       "Cross-tab of headcount by department and level"),
    ],
    "banking": [
        ("bva",         "Loan originations vs target by branch"),
        ("correlation", "Scatter plot of credit score vs default rate"),
        ("pivot",       "Pivot transaction volume by channel across quarters"),
    ],
    "healthcare": [
        ("bva",         "Actual vs budgeted patient volumes by department"),
        ("cohort",      "Patient retention by discharge cohort"),
        ("correlation", "Relationship between wait time and patient satisfaction"),
    ],
    "manufacturing": [
        ("bva",         "Actual production vs plan by line"),
        ("correlation", "Is machine uptime correlated with defect rate?"),
        ("pivot",       "Cross-tab of defects by shift and line"),
    ],
}

_DETECTOR_MAP = {
    "bva":         ("core.budget_vs_actual",   "detect_bva_intent"),
    "cohort":      ("core.cohort_analysis",    "detect_cohort_intent"),
    "correlation": ("core.correlation_analysis","detect_correlation_intent"),
    "pivot":       ("core.pivot_table",        "detect_pivot_intent"),
}


def _load_detector(module_path: str, fn_name: str):
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, fn_name)


@pytest.mark.parametrize("domain,cases", [
    (domain, cases) for domain, cases in BUSINESS_CASES.items()
])
def test_business_model_tier2_coverage(domain, cases):
    """Every Tier 2 intent must fire for representative questions across all domains."""
    for intent_key, question in cases:
        mod_path, fn_name = _DETECTOR_MAP[intent_key]
        detector = _load_detector(mod_path, fn_name)
        assert detector(question), (
            f"[{domain}] {intent_key} detector missed: {question!r}"
        )
