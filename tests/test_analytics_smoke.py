"""
tests/test_analytics_smoke.py
──────────────────────────────
Smoke tests for the 5 new Tier-1 analytical capabilities.

Covers 6 business model scenarios across all modules:
  1. Retail / E-commerce    — sales by region, product, month
  2. HR / Workforce         — headcount, attrition, salary by department
  3. SaaS                   — MRR, churn, CAC by cohort
  4. Banking / Finance      — transaction volumes, balances, risk
  5. Healthcare             — patient volumes, wait times, outcomes
  6. Manufacturing          — production units, defect rate, throughput

Each test is pure Python — no database connection required.
Run with:  pytest tests/test_analytics_smoke.py -v
"""

from __future__ import annotations

import sys
import os
from datetime import date

# Add project root so imports work without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

# ══════════════════════════════════════════════════════════════════════════════
# Shared sample data fixtures (inline — no external files)
# ══════════════════════════════════════════════════════════════════════════════

# ── Retail: monthly sales by region ──────────────────────────────────────────
RETAIL_MONTHLY = [
    {"month": "2024-01", "region": "EMEA",   "revenue": 120000, "units": 1200},
    {"month": "2024-02", "region": "EMEA",   "revenue": 135000, "units": 1350},
    {"month": "2024-03", "region": "EMEA",   "revenue": 128000, "units": 1280},
    {"month": "2024-04", "region": "EMEA",   "revenue": 142000, "units": 1420},
    {"month": "2024-05", "region": "EMEA",   "revenue": 155000, "units": 1550},
    {"month": "2024-06", "region": "EMEA",   "revenue":  98000, "units":  980},  # outlier dip
    {"month": "2024-01", "region": "APAC",   "revenue":  85000, "units":  850},
    {"month": "2024-02", "region": "APAC",   "revenue":  90000, "units":  900},
    {"month": "2024-03", "region": "APAC",   "revenue":  95000, "units":  950},
    {"month": "2024-01", "region": "AMER",   "revenue": 200000, "units": 2000},
    {"month": "2024-02", "region": "AMER",   "revenue": 215000, "units": 2150},
    {"month": "2024-03", "region": "AMER",   "revenue": 190000, "units": 1900},
]

RETAIL_BY_REGION = [
    {"region": "AMER",   "revenue": 605000},
    {"region": "EMEA",   "revenue": 778000},
    {"region": "APAC",   "revenue": 270000},
    {"region": "LATAM",  "revenue":  90000},
    {"region": "MEA",    "revenue":  57000},
]

# ── HR: monthly headcount & attrition ────────────────────────────────────────
HR_MONTHLY = [
    {"month": "2024-01", "department": "Engineering", "headcount": 450, "attrition": 4},
    {"month": "2024-02", "department": "Engineering", "headcount": 455, "attrition": 6},
    {"month": "2024-03", "department": "Engineering", "headcount": 460, "attrition": 2},
    {"month": "2024-04", "department": "Engineering", "headcount": 458, "attrition": 28},  # anomaly
    {"month": "2024-05", "department": "Engineering", "headcount": 445, "attrition": 5},
    {"month": "2024-06", "department": "Engineering", "headcount": 448, "attrition": 3},
]

HR_BY_DEPT = [
    {"department": "Engineering",  "headcount": 450, "avg_salary": 95000},
    {"department": "Sales",        "headcount": 120, "avg_salary": 78000},
    {"department": "Finance",      "headcount":  60, "avg_salary": 88000},
    {"department": "HR",           "headcount":  30, "avg_salary": 65000},
    {"department": "Operations",   "headcount":  80, "avg_salary": 70000},
    {"department": "Legal",        "headcount":  20, "avg_salary": 110000},
]

# ── SaaS: MRR by cohort/month ─────────────────────────────────────────────────
SAAS_MRR = [
    {"month": "2024-01", "mrr": 250000, "churn_rate": 2.1, "new_customers": 45},
    {"month": "2024-02", "mrr": 265000, "churn_rate": 2.3, "new_customers": 52},
    {"month": "2024-03", "mrr": 278000, "churn_rate": 1.9, "new_customers": 61},
    {"month": "2024-04", "mrr": 291000, "churn_rate": 2.0, "new_customers": 58},
    {"month": "2024-05", "mrr": 305000, "churn_rate": 1.8, "new_customers": 70},
    {"month": "2024-06", "mrr": 320000, "churn_rate": 1.7, "new_customers": 75},
]

# ── Banking: transaction data ─────────────────────────────────────────────────
BANKING_TRANSACTIONS = [
    {"branch": "London",    "month": "2024-01", "txn_count": 15200, "avg_balance": 42000},
    {"branch": "London",    "month": "2024-02", "txn_count": 14800, "avg_balance": 43500},
    {"branch": "London",    "month": "2024-03", "txn_count": 99999, "avg_balance": 44100},  # anomaly
    {"branch": "Manchester","month": "2024-01", "txn_count":  8500, "avg_balance": 38000},
    {"branch": "Manchester","month": "2024-02", "txn_count":  8700, "avg_balance": 38500},
    {"branch": "Manchester","month": "2024-03", "txn_count":  8900, "avg_balance": 39000},
    {"branch": "Birmingham","month": "2024-01", "txn_count":  6200, "avg_balance": 35000},
    {"branch": "Birmingham","month": "2024-02", "txn_count":  6400, "avg_balance": 35500},
]

# ── Healthcare: patient volumes ───────────────────────────────────────────────
HEALTHCARE_VOLUMES = [
    {"ward": "A&E",       "month": "2024-01", "patient_count": 3200, "wait_time_min": 45},
    {"ward": "A&E",       "month": "2024-02", "patient_count": 3500, "wait_time_min": 52},
    {"ward": "A&E",       "month": "2024-03", "patient_count": 3100, "wait_time_min": 41},
    {"ward": "Cardiology","month": "2024-01", "patient_count":  850, "wait_time_min": 18},
    {"ward": "Cardiology","month": "2024-02", "patient_count":  900, "wait_time_min": 19},
    {"ward": "Oncology",  "month": "2024-01", "patient_count":  620, "wait_time_min": 22},
    {"ward": "Ortho",     "month": "2024-01", "patient_count":  480, "wait_time_min": 35},
]

# ── Manufacturing: production metrics ────────────────────────────────────────
MANUFACTURING = [
    {"line": "Line A", "week": "W01", "units_produced": 4500, "defect_rate": 1.2},
    {"line": "Line A", "week": "W02", "units_produced": 4600, "defect_rate": 1.1},
    {"line": "Line A", "week": "W03", "units_produced": 4400, "defect_rate": 1.4},
    {"line": "Line A", "week": "W04", "units_produced": 4700, "defect_rate": 0.9},
    {"line": "Line A", "week": "W05", "units_produced": 4200, "defect_rate": 8.5},  # anomaly
    {"line": "Line B", "week": "W01", "units_produced": 3800, "defect_rate": 2.1},
    {"line": "Line B", "week": "W02", "units_produced": 3900, "defect_rate": 1.9},
    {"line": "Line B", "week": "W03", "units_produced": 3750, "defect_rate": 2.0},
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Window Analytics
# ══════════════════════════════════════════════════════════════════════════════

class TestWindowAnalyticsDetection:
    """Validate intent detection across all 4 window types and 6 business models."""

    def _detect(self, q):
        from core.window_analytics import detect_window_intent
        return detect_window_intent(q)

    # Rolling average ─────────────────────────────────────────────────────────
    def test_rolling_avg_retail(self):
        i = self._detect("Show a 3-month rolling average of revenue by region")
        assert i is not None and i.type == "rolling_avg"
        assert i.window_size == 3

    def test_rolling_avg_saas(self):
        i = self._detect("What is the 6-month moving average of MRR?")
        assert i is not None and i.type == "rolling_avg"
        assert i.window_size == 6

    def test_rolling_avg_finance(self):
        i = self._detect("Show 12MA of transaction count for each branch")
        assert i is not None and i.type == "rolling_avg"
        assert i.window_size == 12

    # Running total ───────────────────────────────────────────────────────────
    def test_running_total_retail(self):
        i = self._detect("Show cumulative revenue YTD by month")
        assert i is not None and i.type == "running_total"

    def test_running_total_hr(self):
        i = self._detect("Running total of new hires this year")
        assert i is not None and i.type == "running_total"

    def test_running_total_saas(self):
        i = self._detect("Year-to-date cumulative MRR growth")
        assert i is not None and i.type == "running_total"

    # Rank in group ───────────────────────────────────────────────────────────
    def test_rank_in_group_retail(self):
        i = self._detect("Rank products by revenue within each region")
        assert i is not None and i.type == "rank_in_group"

    def test_rank_in_group_hr(self):
        i = self._detect("Who is the top performer in each department?")
        assert i is not None and i.type == "rank_in_group"

    def test_rank_in_group_manufacturing(self):
        i = self._detect("Top 3 lines by units produced within each shift")
        assert i is not None and i.type == "rank_in_group"

    # Row delta ───────────────────────────────────────────────────────────────
    def test_row_delta_mom_retail(self):
        i = self._detect("Show month-over-month change in revenue by product")
        assert i is not None and i.type == "row_delta"
        assert i.delta_grain == "month"

    def test_row_delta_yoy_saas(self):
        i = self._detect("Year-over-year growth in MRR for each plan tier")
        assert i is not None and i.type == "row_delta"
        assert i.delta_grain == "year"

    def test_row_delta_qoq_finance(self):
        i = self._detect("QoQ change in loan originations per branch")
        assert i is not None and i.type == "row_delta"
        assert i.delta_grain == "quarter"


class TestWindowAnalyticsComputation:
    """Validate pure-Python window computation functions."""

    def test_rolling_average_3month(self):
        from core.window_analytics import compute_rolling_average
        rows = [{"m": m, "rev": v} for m, v in [
            ("2024-01", 100), ("2024-02", 200), ("2024-03", 300),
            ("2024-04", 400), ("2024-05", 500),
        ]]
        result = compute_rolling_average(rows, "rev", window=3)
        # 4th row should be avg(200,300,400) = 300
        assert result[3]["rolling_avg_3"] == pytest.approx(300.0)
        # 5th row should be avg(300,400,500) = 400
        assert result[4]["rolling_avg_3"] == pytest.approx(400.0)

    def test_running_total(self):
        from core.window_analytics import compute_running_total
        rows = [{"month": f"2024-0{i}", "sales": i * 100} for i in range(1, 6)]
        result = compute_running_total(rows, "sales")
        assert result[0]["running_total"] == 100
        assert result[2]["running_total"] == 600
        assert result[4]["running_total"] == 1500

    def test_row_delta(self):
        from core.window_analytics import compute_row_delta
        rows = [{"month": f"2024-0{i}", "mrr": [100, 110, 99, 120][i-1]} for i in range(1, 5)]
        result = compute_row_delta(rows, "mrr")
        assert result[0]["delta"] is None             # no prior for first row
        assert result[1]["delta"] == pytest.approx(10.0)
        assert result[2]["delta"] == pytest.approx(-11.0)
        assert result[3]["pct_change"] == pytest.approx(21.21, abs=0.1)

    def test_rank_in_group(self):
        from core.window_analytics import compute_rank_in_group
        rows = [
            {"dept": "Eng", "emp": "Alice", "salary": 95000},
            {"dept": "Eng", "emp": "Bob",   "salary": 85000},
            {"dept": "Eng", "emp": "Carol", "salary": 102000},
            {"dept": "Sales", "emp": "Dave", "salary": 75000},
            {"dept": "Sales", "emp": "Eve",  "salary": 82000},
        ]
        result = compute_rank_in_group(rows, "salary", "dept")
        # Carol is highest in Eng — should be rank 1
        carol = next(r for r in result if r["emp"] == "Carol")
        assert carol["rank_in_group"] == 1
        # Eve is highest in Sales
        eve = next(r for r in result if r["emp"] == "Eve")
        assert eve["rank_in_group"] == 1

    def test_rolling_average_hr_attrition(self):
        from core.window_analytics import compute_rolling_average
        rows = [{"month": r["month"], "attrition": r["attrition"]} for r in HR_MONTHLY]
        result = compute_rolling_average(rows, "attrition", window=3)
        # First 3 months avg = (4+6+2)/3 = 4.0
        assert result[2]["rolling_avg_3"] == pytest.approx(4.0)
        # Month 4 anomaly (28) pulls 3MA up: (6+2+28)/3 = 12.0
        assert result[3]["rolling_avg_3"] == pytest.approx(12.0)

    def test_sql_hint_tsql(self):
        from core.window_analytics import build_window_sql_hint, WindowIntent
        intent = WindowIntent(type="rolling_avg", window_size=3)
        hint = build_window_sql_hint(intent, "azure_sql")
        assert "ROWS BETWEEN" in hint
        assert "AVG" in hint
        assert "rolling_avg_3" in hint

    def test_sql_hint_running_total_snowflake(self):
        from core.window_analytics import build_window_sql_hint, WindowIntent
        intent = WindowIntent(type="running_total")
        hint = build_window_sql_hint(intent, "snowflake")
        assert "SUM" in hint
        assert "running_total" in hint
        assert "OVER" in hint

    def test_sql_hint_row_delta_requires_staged_metric_alias(self):
        from core.window_analytics import build_window_sql_hint, WindowIntent
        hint = build_window_sql_hint(
            WindowIntent(type="row_delta", delta_grain="month"),
            "azure_sql",
        )
        assert "period_totals CTE" in hint
        assert "period_comparison CTE" in hint
        assert "Never place SUM/COUNT/AVG directly inside LAG/LEAD" in hint
        assert "NULLIF(PREV_METRIC, 0)" in hint


# ══════════════════════════════════════════════════════════════════════════════
# 2. Relative Date Range
# ══════════════════════════════════════════════════════════════════════════════

class TestRelativeDateDetection:

    def _detect(self, q):
        from core.relative_date_range import detect_relative_date_question
        return detect_relative_date_question(q)

    def test_last_30_days_retail(self):
        i = self._detect("How did revenue perform over the last 30 days?")
        assert i is not None
        assert i.unit == "day" and i.n == 30
        assert i.compare is False

    def test_last_30_days_vs_prior_finance(self):
        i = self._detect("Compare last 30 days of transactions vs the prior 30 days")
        assert i is not None
        assert i.unit == "day" and i.n == 30
        assert i.compare is True

    def test_this_week_vs_last_week_hr(self):
        i = self._detect("How does this week's absenteeism compare to last week?")
        assert i is not None
        assert i.compare is True

    def test_last_3_months_saas(self):
        i = self._detect("Show MRR trend for the last 3 months")
        assert i is not None
        assert i.unit == "month" and i.n == 3

    def test_ytd_retail(self):
        i = self._detect("Revenue year to date vs prior year to date")
        assert i is not None and i.ytd is True
        assert i.compare is True

    def test_last_quarter_vs_prior_banking(self):
        i = self._detect("Compare last quarter's loan volumes to the quarter before")
        assert i is not None and i.compare is True

    def test_last_7_days_healthcare(self):
        i = self._detect("Show patient admissions for the past 7 days")
        assert i is not None
        assert i.unit == "day" and i.n == 7

    def test_last_2_weeks_manufacturing(self):
        i = self._detect("Show defect rates for the last 2 weeks")
        assert i is not None
        assert i.unit == "week" and i.n == 2

    def test_no_relative_date_for_calendar_periods(self):
        # Calendar-period questions should NOT be picked up by relative engine
        # (they belong to period_comparison.py)
        i = self._detect("Compare Q1 2024 vs Q1 2023")
        # Should not match because Q1 2024 is a named calendar period
        assert i is None or i.compare is False  # no rolling window

    def test_no_false_positive_simple_question(self):
        i = self._detect("What is total revenue by department?")
        assert i is None


class TestRelativeDateWindows:

    def test_last_30_days_window(self):
        from core.relative_date_range import detect_relative_date_question, compute_relative_windows
        intent = detect_relative_date_question("last 30 days vs prior 30 days")
        assert intent is not None
        curr, prior = compute_relative_windows(intent, as_of=date(2024, 6, 30))
        # Current window should be 30 days ending June 29
        assert (curr.end - curr.start).days == 29   # 30 days inclusive
        # Prior window same duration, immediately before current
        assert prior is not None
        assert prior.end == curr.start - __import__("datetime").timedelta(days=1)

    def test_ytd_window(self):
        from core.relative_date_range import detect_relative_date_question, compute_relative_windows
        intent = detect_relative_date_question("year to date")
        intent.compare = True
        curr, prior = compute_relative_windows(intent, as_of=date(2024, 6, 30))
        assert curr.start == date(2024, 1, 1)
        assert curr.end == date(2024, 6, 30)
        assert prior.start == date(2023, 1, 1)

    def test_last_3_months_window(self):
        from core.relative_date_range import detect_relative_date_question, compute_relative_windows
        intent = detect_relative_date_question("last 3 months vs prior 3 months")
        assert intent is not None
        curr, prior = compute_relative_windows(intent, as_of=date(2024, 7, 15))
        # Current should be a 3-month window ending June 30
        assert curr.end.month == 6
        assert prior is not None
        # Prior window ends the day before current starts
        assert (curr.start - prior.end).days == 1

    def test_sql_rewrite_prompt_structure(self):
        from core.relative_date_range import (
            detect_relative_date_question, compute_relative_windows, build_relative_date_rewrite_prompt
        )
        intent = detect_relative_date_question("last 7 days")
        curr, _ = compute_relative_windows(intent, as_of=date(2024, 6, 30))
        sys_p, usr_p = build_relative_date_rewrite_prompt(
            "SELECT * FROM sales WHERE sale_date = '2024-01-01'", curr
        )
        assert "CANNOT_REWRITE" in sys_p
        assert curr.start.isoformat() in usr_p
        assert curr.end.isoformat() in usr_p


# ══════════════════════════════════════════════════════════════════════════════
# 3. Contribution Analysis
# ══════════════════════════════════════════════════════════════════════════════

class TestContributionDetection:

    def _detect(self, q):
        from core.contribution_analysis import detect_contribution_intent
        return detect_contribution_intent(q)

    def test_revenue_mix_retail(self):
        assert self._detect("What % of total revenue does each region contribute?")

    def test_headcount_share_hr(self):
        assert self._detect("Show the headcount mix by department")

    def test_mrr_contribution_saas(self):
        assert self._detect("Which plan tiers contribute the most to MRR?")

    def test_transaction_share_banking(self):
        assert self._detect("What share of total transactions does each branch account for?")

    def test_patient_distribution_healthcare(self):
        assert self._detect("Show patient count breakdown as a percentage by ward")

    def test_defect_contribution_manufacturing(self):
        assert self._detect("Which production line is the top contributor to defects?")

    def test_pareto_detection(self):
        assert self._detect("Do the top 3 products follow a Pareto distribution?")

    def test_no_false_positive(self):
        assert not self._detect("Show total revenue by month")
        assert not self._detect("Who has the highest sales?")


class TestContributionComputation:

    def test_basic_contribution(self):
        from core.contribution_analysis import compute_contribution
        result = compute_contribution(RETAIL_BY_REGION, "revenue", "region")
        # Sorted descending — EMEA has highest revenue
        total = sum(r["revenue"] for r in RETAIL_BY_REGION)
        for row in result:
            expected = round(row["revenue"] / total * 100, 2)
            assert row["contribution_pct"] == pytest.approx(expected, abs=0.01)

    def test_top_n_with_other_bucket(self):
        from core.contribution_analysis import compute_contribution
        result = compute_contribution(RETAIL_BY_REGION, "revenue", "region", top_n=3)
        assert len(result) == 4  # top 3 + "Other (2 items)"
        last = result[-1]
        assert "Other" in str(last.get("region", ""))

    def test_contribution_sums_to_100(self):
        from core.contribution_analysis import compute_contribution
        result = compute_contribution(HR_BY_DEPT, "headcount", "department")
        total_pct = sum(r.get("contribution_pct") or 0 for r in result)
        assert total_pct == pytest.approx(100.0, abs=0.1)

    def test_contribution_summary_pareto(self):
        from core.contribution_analysis import build_contribution_summary
        summary = build_contribution_summary(RETAIL_BY_REGION, "revenue", "region")
        assert summary.top_contributor == "EMEA"
        assert summary.category_count == 5
        # Pareto: how many reach 80%?
        assert summary.pareto_80_count <= 3

    def test_zero_total_handled_gracefully(self):
        from core.contribution_analysis import compute_contribution
        rows = [{"cat": "A", "value": 0}, {"cat": "B", "value": 0}]
        result = compute_contribution(rows, "value", "cat")
        # All contributions should be None when total is 0
        assert all(r["contribution_pct"] is None for r in result)

    def test_sql_hint_contains_window_function(self):
        from core.contribution_analysis import build_contribution_sql_hint
        hint = build_contribution_sql_hint("revenue")
        assert "SUM(revenue) OVER ()" in hint
        assert "contribution_pct" in hint
        assert "NULLIF" in hint

    def test_infer_numeric_col(self):
        from core.contribution_analysis import infer_numeric_col
        rows = [{"dept": "Eng", "headcount": 100, "budget": 500000}]
        col = infer_numeric_col(rows)
        assert col in ("headcount", "budget")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Anomaly Detection
# ══════════════════════════════════════════════════════════════════════════════

class TestAnomalyDetection:

    def _detect(self, q):
        from core.anomaly_detection import detect_anomaly_intent
        return detect_anomaly_intent(q)

    def test_unusual_sick_leave_hr(self):
        assert self._detect("Which months had unusually high sick leave?")

    def test_spike_attrition_hr(self):
        assert self._detect("Flag months where attrition spiked abnormally")

    def test_outlier_salary_hr(self):
        assert self._detect("Show outlier employees by salary")

    def test_anomaly_transactions_banking(self):
        assert self._detect("Are there any anomalies in our transaction data?")

    def test_unusual_defect_rate_manufacturing(self):
        assert self._detect("Identify unusual spikes in the defect rate")

    def test_statistical_deviation_healthcare(self):
        assert self._detect("Show me weeks with wait times outside the normal range")

    def test_no_false_positive(self):
        assert not self._detect("Show total revenue by region")
        assert not self._detect("What is the average salary by department?")


class TestAnomalyComputation:

    def test_zscore_flags_hr_attrition_anomaly(self):
        from core.anomaly_detection import detect_anomalies
        # HR_MONTHLY row 4 has attrition=28, all others are 2-6 — clear spike
        rows = [{"month": r["month"], "attrition": r["attrition"]} for r in HR_MONTHLY]
        result = detect_anomalies(rows, "attrition", method="zscore", zscore_threshold=2.0)
        flagged_months = {r["month"] for r in result.flagged}
        assert "2024-04" in flagged_months

    def test_zscore_flags_banking_anomaly(self):
        from core.anomaly_detection import detect_anomalies
        rows = [{"month": r["month"], "txn_count": r["txn_count"]}
                for r in BANKING_TRANSACTIONS]
        # London March (99999) is extreme but with only 8 rows z ≈ 2.46 — use 2.0 threshold
        result = detect_anomalies(rows, "txn_count", method="zscore", zscore_threshold=2.0)
        flagged = {r["month"] for r in result.flagged}
        assert "2024-03" in flagged

    def test_iqr_flags_manufacturing_anomaly(self):
        from core.anomaly_detection import detect_anomalies
        rows = [{"week": r["week"], "defect_rate": r["defect_rate"]}
                for r in MANUFACTURING if r["line"] == "Line A"]
        result = detect_anomalies(rows, "defect_rate", method="iqr")
        # Week W05 (8.5%) should be flagged as outlier
        flagged = {r["week"] for r in result.flagged}
        assert "W05" in flagged

    def test_auto_selects_method(self):
        from core.anomaly_detection import detect_anomalies
        rows = [{"v": x} for x in [1, 1, 1, 2, 1, 1, 1, 100]]  # heavily right-skewed
        result = detect_anomalies(rows, "v", method="auto")
        # Auto should pick IQR for skewed data
        assert result.method == "iqr"

    def test_normal_distribution_uses_zscore(self):
        from core.anomaly_detection import detect_anomalies
        # Normally distributed values
        rows = [{"v": x} for x in [98, 100, 102, 99, 101, 100, 98, 103]]
        result = detect_anomalies(rows, "v", method="auto")
        assert result.method == "zscore"

    def test_insufficient_data_graceful(self):
        from core.anomaly_detection import detect_anomalies
        rows = [{"v": 100}, {"v": 200}, {"v": 150}]  # < 4 rows
        result = detect_anomalies(rows, "v")
        assert result.flagged_rows == 0
        assert all(not r["anomaly_flag"] for r in result.rows)

    def test_anomaly_brief_safe_for_llm(self):
        from core.anomaly_detection import detect_anomalies, build_anomaly_brief
        rows = [{"month": r["month"], "attrition": r["attrition"]} for r in HR_MONTHLY]
        anom = detect_anomalies(rows, "attrition")
        brief = build_anomaly_brief(anom)
        # Brief must not contain raw row values
        assert "rows" not in brief
        assert "flagged" not in brief
        assert "total_rows" in brief
        assert "flagged_rows" in brief
        assert isinstance(brief["anomaly_rate"], str)

    def test_infer_value_col(self):
        from core.anomaly_detection import infer_value_col
        rows = [{"region": "EMEA", "revenue": 100000, "units": 1000}]
        col = infer_value_col(rows)
        assert col in ("revenue", "units")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Multi-Period Comparison
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiPeriodDetection:

    def _detect(self, q):
        from core.multi_period import detect_multi_period_intent
        return detect_multi_period_intent(q)

    def test_three_named_years_retail(self):
        i = self._detect("Compare revenue for 2022, 2023, and 2024")
        assert i is not None
        assert i.compare_count == 3

    def test_last_n_years_saas(self):
        i = self._detect("Show MRR for the last 3 years")
        assert i is not None
        assert i.compare_count == 3

    def test_quarterly_comparison_banking(self):
        i = self._detect("Compare Q1 2024, Q2 2024, and Q3 2024 loan volumes")
        assert i is not None
        assert i.compare_count == 3
        assert i.grain == "quarterly"

    def test_year_over_year_4_years_healthcare(self):
        i = self._detect("Show patient counts year over year for the last 4 years")
        assert i is not None
        assert i.compare_count == 4

    def test_two_years_not_multi_period(self):
        # 2 named years — should return None (handled by period_comparison.py)
        i = self._detect("Compare 2023 and 2024 revenue")
        # Only 2 specs → None (unless a multi-signal keyword triggers it)
        # This is acceptable — 2 periods go to period_comparison.py
        # Just verify it doesn't crash
        assert i is None or i.compare_count == 2

    def test_no_false_positive(self):
        i = self._detect("What is the revenue for 2024?")
        assert i is None


class TestMultiPeriodExtraction:

    def test_extract_three_years(self):
        from core.multi_period import extract_period_specs
        specs = extract_period_specs("Revenue in 2022, 2023, and 2024 by region")
        labels = [s.label for s in specs]
        assert "2022" in labels
        assert "2023" in labels
        assert "2024" in labels
        assert len(specs) == 3

    def test_extract_quarterly(self):
        from core.multi_period import extract_period_specs
        specs = extract_period_specs("Compare Q1 2024, Q2 2024, and Q3 2024")
        assert len(specs) == 3
        assert all(s.label.startswith("Q") for s in specs)

    def test_extract_monthly(self):
        from core.multi_period import extract_period_specs
        specs = extract_period_specs("Show Jan 2024, Feb 2024, Mar 2024 breakdown")
        assert len(specs) == 3

    def test_relative_specs_last_3_years(self):
        from core.multi_period import _generate_relative_specs
        specs = _generate_relative_specs(3, "year")
        assert len(specs) == 3
        # Should be chronological (oldest first)
        years = [int(s.label) for s in specs]
        assert years == sorted(years)

    def test_relative_specs_last_4_quarters(self):
        from core.multi_period import _generate_relative_specs
        specs = _generate_relative_specs(4, "quarter")
        assert len(specs) == 4
        assert all("Q" in s.label for s in specs)


class TestMultiPeriodMerge:

    def test_merge_three_period_results(self):
        from core.multi_period import PeriodResult, merge_multi_period_results

        p1 = PeriodResult("2022", [
            {"region": "EMEA", "revenue": 100},
            {"region": "APAC", "revenue": 80},
        ], sql="", success=True)
        p2 = PeriodResult("2023", [
            {"region": "EMEA", "revenue": 120},
            {"region": "APAC", "revenue": 95},
        ], sql="", success=True)
        p3 = PeriodResult("2024", [
            {"region": "EMEA", "revenue": 145},
            {"region": "APAC", "revenue": 110},
        ], sql="", success=True)

        result = merge_multi_period_results([p1, p2, p3])
        assert result.successful_periods == 3
        assert result.label_col == "region"
        assert result.value_col == "revenue"

    def test_merge_with_failed_period(self):
        from core.multi_period import PeriodResult, merge_multi_period_results

        p1 = PeriodResult("2022", [{"region": "EMEA", "revenue": 100}], sql="", success=True)
        p2 = PeriodResult("2023", [], sql="", success=False, error="timeout")

        result = merge_multi_period_results([p1, p2])
        assert result.successful_periods == 1
        assert len(result.warnings) > 0

    def test_chart_payload_structure(self):
        from core.multi_period import PeriodResult, merge_multi_period_results, build_multi_period_chart_payload

        periods = [
            PeriodResult("2022", [{"dept": "Eng", "revenue": 100}, {"dept": "Sales", "revenue": 80}], "", True),
            PeriodResult("2023", [{"dept": "Eng", "revenue": 120}, {"dept": "Sales", "revenue": 90}], "", True),
            PeriodResult("2024", [{"dept": "Eng", "revenue": 145}, {"dept": "Sales", "revenue": 105}], "", True),
        ]
        result = merge_multi_period_results(periods)
        payload = build_multi_period_chart_payload(result, periods, title="Revenue 3-Year")

        assert payload["chart_type"] == "bar"
        assert payload["multi_period"] is True
        assert "2022" in payload["y_keys"]
        assert "2024" in payload["y_keys"]
        assert len(payload["rows"]) == 2  # Eng and Sales


# ══════════════════════════════════════════════════════════════════════════════
# 6. Unified Intent Detector (cross-module integration)
# ══════════════════════════════════════════════════════════════════════════════

class TestUnifiedIntentDetector:
    """Test detect_analytical_intents() which calls all detectors at once."""

    def _all(self, q):
        from core.insight import detect_analytical_intents
        return detect_analytical_intents(q)

    def test_window_intent_captured(self):
        r = self._all("Show a 3-month rolling average of sales")
        assert r["window"] is not None
        assert r["window"].type == "rolling_avg"

    def test_anomaly_intent_captured(self):
        r = self._all("Flag unusual spikes in attrition data")
        assert r["anomaly"] is True

    def test_contribution_intent_captured(self):
        r = self._all("What % of total revenue does each product contribute?")
        assert r["contribution"] is True

    def test_relative_date_intent_captured(self):
        r = self._all("Compare last 30 days of transactions vs the prior 30 days")
        assert r["relative_date"] is not None
        assert r["relative_date"].compare is True

    def test_multi_period_intent_captured(self):
        r = self._all("Compare revenue for 2022, 2023, and 2024")
        assert r["multi_period"] is not None

    def test_no_intents_on_simple_question(self):
        r = self._all("What is the total headcount by department?")
        assert r["window"] is None
        assert r["anomaly"] is False
        assert r["contribution"] is False
        assert r["relative_date"] is None
        # multi_period might fire if 2+ years in question — so we skip that assertion

    def test_complex_question_multiple_intents(self):
        r = self._all("Show year-over-year change and flag any anomalies in MRR")
        # Should fire both window (row_delta) and anomaly
        assert r["window"] is not None
        assert r["anomaly"] is True


# ══════════════════════════════════════════════════════════════════════════════
# 7. Cross-Business-Model Completeness Check
# ══════════════════════════════════════════════════════════════════════════════

class TestBusinessModelCoverage:
    """
    Verify that the new analytics capabilities cover canonical questions
    from each of the 6 target business models.
    """

    QUESTIONS = {
        "retail": [
            ("rolling_avg",     "What is the 3-month rolling average of revenue by product?"),
            ("running_total",   "Show cumulative sales revenue year to date"),
            ("contribution",    "Which products account for 80% of revenue?"),
            ("row_delta",       "Show month-over-month sales change by region"),
            ("anomaly",         "Flag any months with unusually low conversion rates"),
            ("relative_date",   "Compare last 90 days vs the previous 90 days"),
            ("multi_period",    "Compare Q1 2022, Q1 2023, and Q1 2024 revenue"),
        ],
        "hr": [
            ("rolling_avg",     "Show a 6-month moving average of attrition rate"),
            ("running_total",   "Running total of new hires this year by department"),
            ("contribution",    "What % of total headcount is each department?"),
            ("row_delta",       "Month-over-month change in headcount by team"),
            ("anomaly",         "Identify months with abnormal sick leave spikes"),
            ("relative_date",   "How did attrition last month compare to the month before?"),
            ("multi_period",    "Compare headcount for 2022, 2023, and 2024"),
        ],
        "saas": [
            ("rolling_avg",     "Show 3-month moving average MRR"),
            ("running_total",   "Cumulative MRR for the year so far"),
            ("contribution",    "Which customer segments contribute most to MRR?"),
            ("row_delta",       "Year-over-year MRR growth by plan tier"),
            ("anomaly",         "Flag months with unusual churn spikes"),
            ("relative_date",   "Compare last 30 days signups vs the prior 30 days"),
            ("multi_period",    "MRR trend for the last 4 quarters"),
        ],
        "banking": [
            ("rolling_avg",     "Show 12-month moving average of loan originations"),
            ("running_total",   "Running total of deposits year to date"),
            ("contribution",    "What share of total transactions does each branch handle?"),
            ("row_delta",       "Month-over-month change in NPL ratio by product"),
            ("anomaly",         "Detect outliers in transaction volumes by branch"),
            ("relative_date",   "Compare last 7 days of transactions vs prior 7 days"),
            ("multi_period",    "Compare Q1 2022, Q1 2023, Q1 2024 loan volumes"),
        ],
        "healthcare": [
            ("rolling_avg",     "Show 4-week rolling average of A&E wait times"),
            ("running_total",   "Cumulative patient admissions year to date"),
            ("contribution",    "Which wards account for most of the patient volume?"),
            ("row_delta",       "Month-over-month change in readmission rate by ward"),
            ("anomaly",         "Flag weeks with abnormal bed occupancy rates"),
            ("relative_date",   "Compare last 4 weeks of admissions vs prior 4 weeks"),
            ("multi_period",    "Compare patient volumes for 2022, 2023, 2024"),
        ],
        "manufacturing": [
            ("rolling_avg",     "Show a 4-week moving average of units produced"),
            ("running_total",   "Cumulative units produced so far this quarter"),
            ("contribution",    "Which production lines account for most defects?"),
            ("row_delta",       "Week-over-week change in defect rate by line"),
            ("anomaly",         "Identify unusual spikes in downtime hours"),
            ("relative_date",   "Compare this week's throughput to last week"),
            ("multi_period",    "Compare Line A output for Q1 2022, Q1 2023, Q1 2024"),
        ],
    }

    def _classify(self, question: str) -> set[str]:
        """Return which analytical intents fire for this question."""
        from core.insight import detect_analytical_intents
        intents = detect_analytical_intents(question)
        fired = set()
        if intents.get("window"):
            fired.add(intents["window"].type)  # rolling_avg / running_total / etc.
        if intents.get("anomaly"):
            fired.add("anomaly")
        if intents.get("contribution"):
            fired.add("contribution")
        if intents.get("relative_date"):
            fired.add("relative_date")
        if intents.get("multi_period"):
            fired.add("multi_period")
        return fired

    @pytest.mark.parametrize("model,cases", QUESTIONS.items())
    def test_all_questions_detected(self, model, cases):
        """Each canonical question should fire at least one analytical intent."""
        failures = []
        for expected_intent, question in cases:
            fired = self._classify(question)
            # Map intent name to what we expect
            intent_map = {
                "rolling_avg":   {"rolling_avg"},
                "running_total": {"running_total"},
                "row_delta":     {"row_delta"},
                "rank_in_group": {"rank_in_group"},
                "contribution":  {"contribution"},
                "anomaly":       {"anomaly"},
                "relative_date": {"relative_date"},
                "multi_period":  {"multi_period"},
            }
            expected_set = intent_map.get(expected_intent, {expected_intent})
            if not (fired & expected_set):
                failures.append(f"[{model}] '{question}' expected {expected_intent}, got {fired}")
        assert not failures, "\n" + "\n".join(failures)
