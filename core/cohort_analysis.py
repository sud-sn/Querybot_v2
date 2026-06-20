"""
core/cohort_analysis.py
────────────────────────
Cohort retention analysis.

Answers questions like:
  "Show user retention by cohort"
  "What is the month-over-month cohort retention rate?"
  "Cohort analysis of customers who signed up in Q1"
  "How many users from the January cohort are still active in month 6?"

Design
──────
• Detection: regex on NL question.
• Post-processing: pivots flat cohort data into a retention matrix.
  Input format expected from SQL:
    cohort_month (or cohort_date, cohort_period)
    period_month (or months_since, period_offset, month_number)
    user_count   (or customer_count, retained_count)
• The pivot creates rows where each cohort = one row, each period = one column.
• Retention is computed as (period_count / month_0_count) × 100.
• Output drives a heatmap chart: x=period offsets, y=cohort months, value=retention%.

Entry points
────────────
  detect_cohort_intent(question) → bool
  infer_cohort_cols(rows) → tuple[str, str, str]  (cohort_col, period_col, value_col)
  compute_cohort_matrix(rows, cohort_col, period_col, value_col) → list[dict]
  build_cohort_sql_hint() → str
  build_cohort_summary(matrix, cohort_col) → CohortSummary
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Detection
# ══════════════════════════════════════════════════════════════════════════════

_COHORT_PATTERNS = [
    re.compile(r"\bcohort\b", re.I),
    re.compile(r"\bretention\s+(?:rate|analysis|curve|by|per)\b", re.I),
    re.compile(r"\b(?:user|customer|subscriber|member)\s+retention\b", re.I),
    re.compile(r"\bchurn\s+(?:by\s+cohort|analysis|curve)\b", re.I),
    re.compile(r"\bmonth(?:s)?\s+(?:since\s+)?(?:sign[- ]?up|acquisition|onboarding|first)\b", re.I),
    re.compile(r"\b(?:active|retained)\s+(?:users|customers|subscribers)\s+(?:by|per)\s+(?:month|cohort|period)\b", re.I),
    re.compile(r"\blifetime\s+retention\b", re.I),
    re.compile(r"\bD\d+\s+retention\b", re.I),  # D1, D7, D30 retention
    re.compile(r"\b(?:day|week|month)\s+\d+\s+retention\b", re.I),
]


def detect_cohort_intent(question: str) -> bool:
    """Return True if the question asks for cohort/retention analysis."""
    return any(p.search(question) for p in _COHORT_PATTERNS)


# ══════════════════════════════════════════════════════════════════════════════
# Column inference
# ══════════════════════════════════════════════════════════════════════════════

_COHORT_COL_KEYWORDS = {
    "cohort", "cohort_month", "cohort_date", "cohort_period",
    "signup_month", "acquisition_month", "first_purchase_month",
}
_PERIOD_COL_KEYWORDS = {
    "period", "period_month", "month_number", "months_since",
    "period_offset", "offset", "months_active", "tenure_month",
}
_VALUE_COL_KEYWORDS = {
    "count", "users", "customers", "retained", "active",
    "user_count", "customer_count", "retained_count", "active_users",
}


def _col_score(col: str, kws: set[str]) -> int:
    col_lower = col.lower().replace("-", "_").replace(" ", "_")
    return sum(1 for kw in kws if kw in col_lower)


def infer_cohort_cols(rows: list[dict]) -> tuple[str, str, str]:
    """
    Infer (cohort_col, period_col, value_col) from column names.
    Returns ("", "", "") if inference fails.
    """
    if not rows:
        return "", "", ""
    cols = list(rows[0].keys())

    def _to_float(v: Any) -> float | None:
        try:
            return float(str(v).replace(",", ""))
        except (TypeError, ValueError):
            return None

    # Score each column
    cohort_col = max(cols, key=lambda c: _col_score(c, _COHORT_COL_KEYWORDS), default="")
    period_col = max(cols, key=lambda c: _col_score(c, _PERIOD_COL_KEYWORDS), default="")
    value_col  = max(cols, key=lambda c: _col_score(c, _VALUE_COL_KEYWORDS), default="")

    # Ensure they are all different
    remaining = [c for c in cols if c not in (cohort_col, period_col, value_col)]
    numeric = [c for c in cols if all(_to_float(r.get(c)) is not None for r in rows[:5] if r.get(c) is not None)]

    if not cohort_col or _col_score(cohort_col, _COHORT_COL_KEYWORDS) == 0:
        # First text-ish column
        text_cols = [c for c in cols if c not in numeric]
        cohort_col = text_cols[0] if text_cols else (cols[0] if cols else "")

    if not period_col or _col_score(period_col, _PERIOD_COL_KEYWORDS) == 0 or period_col == cohort_col:
        # Second column
        others = [c for c in cols if c != cohort_col]
        period_col = others[0] if others else ""

    if not value_col or value_col in (cohort_col, period_col):
        # First numeric column that isn't cohort/period
        n_cols = [c for c in numeric if c not in (cohort_col, period_col)]
        value_col = n_cols[0] if n_cols else ""

    return cohort_col, period_col, value_col


# ══════════════════════════════════════════════════════════════════════════════
# Cohort matrix computation
# ══════════════════════════════════════════════════════════════════════════════

def _to_float(v: Any) -> float | None:
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def compute_cohort_matrix(
    rows: list[dict],
    cohort_col: str,
    period_col: str,
    value_col: str,
) -> list[dict]:
    """
    Pivot flat cohort rows into a retention matrix.

    Input rows (example):
      [{"cohort_month": "2024-01", "period_month": 0, "user_count": 1200},
       {"cohort_month": "2024-01", "period_month": 1, "user_count": 960}, ...]

    Output rows (heatmap-ready):
      [{"cohort": "2024-01", "Month 0": 100.0, "Month 1": 80.0, ...},
       {"cohort": "2024-02", "Month 0": 100.0, "Month 1": 75.0, ...}]

    Retention % is relative to the Month 0 count for each cohort.
    Absolute counts preserved in __abs__ variant keys for tooltip.
    """
    if not rows or not cohort_col or not period_col or not value_col:
        return rows

    # Build nested dict: cohort → period → count
    data: dict[str, dict[Any, float]] = {}
    for row in rows:
        cohort = str(row.get(cohort_col, ""))
        period = row.get(period_col)
        val    = _to_float(row.get(value_col))
        if not cohort or period is None or val is None:
            continue
        data.setdefault(cohort, {})[period] = val

    # Sort periods
    all_periods = sorted({p for d in data.values() for p in d.keys()},
                          key=lambda p: (_to_float(p) if _to_float(p) is not None else str(p)))

    # Determine "period 0" key (smallest period value)
    period_0 = all_periods[0] if all_periods else None

    matrix_rows = []
    for cohort in sorted(data.keys()):
        counts = data[cohort]
        base   = counts.get(period_0, 0) or 0
        row_out: dict[str, Any] = {"cohort": cohort}
        for p in all_periods:
            col_name = f"Month {p}" if isinstance(p, (int, float)) and float(p).is_integer() else str(p)
            count    = counts.get(p)
            if count is None:
                pct = None
            elif base > 0:
                pct = round(count / base * 100, 1)
            else:
                pct = 100.0 if count > 0 else 0.0
            row_out[col_name] = pct
            row_out[f"__abs_{col_name}"] = count  # raw count for tooltip
        row_out["cohort_size"] = int(base)
        matrix_rows.append(row_out)

    return matrix_rows


# ══════════════════════════════════════════════════════════════════════════════
# SQL hint
# ══════════════════════════════════════════════════════════════════════════════

def build_cohort_sql_hint() -> str:
    return (
        "COHORT RETENTION HINT:\n"
        "The user wants a cohort retention analysis. "
        "Return a flat table with exactly these columns:\n\n"
        "  cohort_month   — the period in which the user/customer was acquired "
        "(e.g. DATETRUNC('month', first_purchase_date) or FORMAT(signup_date,'yyyy-MM'))\n"
        "  period_offset  — integer months since acquisition (0 = acquisition month, "
        "1 = 1 month later, etc.) — compute as DATEDIFF('month', cohort_month, activity_month)\n"
        "  retained_count — number of users still active in that period\n\n"
        "Example SQL pattern (adjust table/column names):\n"
        "  SELECT\n"
        "    FORMAT(u.signup_date, 'yyyy-MM') AS cohort_month,\n"
        "    DATEDIFF(month, u.signup_date, a.activity_date) AS period_offset,\n"
        "    COUNT(DISTINCT u.user_id) AS retained_count\n"
        "  FROM users u\n"
        "  JOIN activity a ON a.user_id = u.user_id\n"
        "  GROUP BY FORMAT(u.signup_date, 'yyyy-MM'),\n"
        "           DATEDIFF(month, u.signup_date, a.activity_date)\n"
        "  ORDER BY cohort_month, period_offset\n\n"
        "This flat output will be auto-pivoted into a retention heatmap."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Summary for LLM brief
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CohortSummary:
    cohort_count: int
    period_count: int
    avg_month1_retention: float | None    # average retention at period 1 across all cohorts
    avg_month3_retention: float | None
    avg_month6_retention: float | None
    best_cohort: str
    worst_cohort: str
    overall_avg_retention: float | None   # all non-period-0 cells averaged


def build_cohort_summary(matrix: list[dict], cohort_col: str = "cohort") -> CohortSummary:
    if not matrix:
        return CohortSummary(0, 0, None, None, None, "", "", None)

    period_cols = [k for k in matrix[0].keys()
                   if k not in (cohort_col, "cohort", "cohort_size")
                   and not k.startswith("__abs_")]
    n_periods = len(period_cols)
    n_cohorts = len(matrix)

    def _avg_at(col_name: str) -> float | None:
        vals = [_to_float(r.get(col_name)) for r in matrix if r.get(col_name) is not None]
        if not vals:
            return None
        return round(sum(vals) / len(vals), 1)

    m1 = _avg_at("Month 1") if "Month 1" in (matrix[0].keys() if matrix else []) else None
    m3 = _avg_at("Month 3") if "Month 3" in (matrix[0].keys() if matrix else []) else None
    m6 = _avg_at("Month 6") if "Month 6" in (matrix[0].keys() if matrix else []) else None

    # Overall avg retention (skip Month 0 which is always ~100%)
    non_zero_cols = [c for c in period_cols if c != "Month 0"]
    all_vals = [_to_float(r.get(c)) for r in matrix for c in non_zero_cols if r.get(c) is not None]
    overall = round(sum(all_vals) / len(all_vals), 1) if all_vals else None

    # Best/worst cohort by avg retention across periods
    def _cohort_avg(row: dict) -> float:
        vals = [_to_float(row.get(c)) for c in non_zero_cols if row.get(c) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    best  = max(matrix, key=_cohort_avg, default={})
    worst = min(matrix, key=_cohort_avg, default={})

    return CohortSummary(
        cohort_count=n_cohorts,
        period_count=n_periods,
        avg_month1_retention=m1,
        avg_month3_retention=m3,
        avg_month6_retention=m6,
        best_cohort=str(best.get("cohort", "")),
        worst_cohort=str(worst.get("cohort", "")),
        overall_avg_retention=overall,
    )
