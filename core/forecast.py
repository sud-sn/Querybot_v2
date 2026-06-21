"""
core/forecast.py
─────────────────
Quantitative trend forecast — linear extrapolation over time-series data.

Answers questions like:
  "Forecast revenue for the next 3 months"
  "Predict sales for Q4 based on the trend"
  "Project headcount growth over the next 6 months"
  "What will MRR be in December?"
  "Trend projection for website visits"

Design
──────
• Detection: regex on NL question.
• Post-processing: fits a least-squares linear regression to the numeric
  column, then extrapolates N future periods.
• The original rows are returned unchanged with a `is_forecast: False` flag.
  Projected rows are appended with `is_forecast: True` and `forecast_value`.
• Uses Python 3.10+ `statistics.linear_regression()` — falls back to manual
  OLS if unavailable.
• Chart output: "line" chart type (already supported); forecast rows are
  differentiated by `is_forecast` flag so the JS renderer can style them
  with a dotted line.

Entry points
────────────
  detect_forecast_intent(question) → bool
  extract_forecast_periods(question) → int          default 3
  infer_forecast_cols(rows) → tuple[str, str]       (period_col, value_col)
  compute_forecast(rows, period_col, value_col, n_periods) → list[dict]
  build_forecast_sql_hint() → str
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass
from statistics import mean
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Detection
# ══════════════════════════════════════════════════════════════════════════════

_FORECAST_PATTERNS = [
    re.compile(r"\bforecast\b", re.I),
    re.compile(r"\bpredict(?:ion|ed|s)?\b", re.I),
    re.compile(r"\bproject(?:ion|ed|s)?\b", re.I),
    re.compile(r"\btrend\s+(?:line|projection|analysis|forecast|forward)\b", re.I),
    re.compile(r"\b(?:next|coming|future)\s+\d+\s+(?:month|quarter|week|year|period)[s]?\b", re.I),
    re.compile(r"\bwhat\s+will\s+.{0,30}\s+be\s+(?:in|by|next|this)\b", re.I),
    re.compile(r"\bextrapolat(?:e|ion|ed)\b", re.I),
    re.compile(r"\bgrowth\s+(?:rate\s+)?(?:projection|forecast|trend)\b", re.I),
    re.compile(r"\b(?:estimate|project|predict)\s+.{0,20}(?:for\s+the\s+)?next\b", re.I),
]


def detect_forecast_intent(question: str) -> bool:
    """Return True if the question asks for trend forecast or projection."""
    return any(p.search(question) for p in _FORECAST_PATTERNS)


def extract_forecast_periods(question: str) -> int:
    """Extract the number of periods to forecast. Default: 3."""
    m = re.search(r"\b(?:next\s+|coming\s+|future\s+)?(\d+)\s+(?:month|quarter|week|year|period)[s]?\b", question, re.I)
    if m:
        n = int(m.group(1))
        return max(1, min(n, 24))  # cap at 24 periods
    return 3


# ══════════════════════════════════════════════════════════════════════════════
# Column inference
# ══════════════════════════════════════════════════════════════════════════════

_PERIOD_KEYWORDS = {
    "month", "date", "period", "year", "quarter", "week", "day",
    "time", "fiscal", "fy", "q", "jan", "feb", "mar",
}
_VALUE_KEYWORDS = {
    "revenue", "sales", "mrr", "arr", "cost", "profit", "margin",
    "headcount", "count", "visits", "orders", "units", "volume",
    "users", "customers", "churn", "rate",
}


def _col_kw_score(col: str, kws: set[str]) -> int:
    cl = col.lower()
    return sum(1 for kw in kws if kw in cl)


def _to_float(v: Any) -> float | None:
    try:
        return float(str(v).replace(",", "").replace("$", "").replace("%", ""))
    except (TypeError, ValueError):
        return None


def infer_forecast_cols(rows: list[dict]) -> tuple[str, str]:
    """
    Infer (period_col, value_col).
    Returns ("", "") if the data doesn't look like a time series.
    """
    if not rows:
        return "", ""
    cols = list(rows[0].keys())
    numeric = [c for c in cols if _to_float(rows[0].get(c)) is not None]
    text    = [c for c in cols if c not in numeric]

    # Period col: best-scoring text col, else any col with date-looking values
    period_col = max(text, key=lambda c: _col_kw_score(c, _PERIOD_KEYWORDS), default="")
    if not period_col:
        period_col = text[0] if text else ""

    # Value col: best-scoring numeric col
    value_col = max(numeric, key=lambda c: _col_kw_score(c, _VALUE_KEYWORDS), default="")
    if not value_col and numeric:
        value_col = numeric[0]

    return period_col, value_col


# ══════════════════════════════════════════════════════════════════════════════
# Linear regression (pure Python OLS)
# ══════════════════════════════════════════════════════════════════════════════

def _ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Return (slope, intercept) via ordinary least squares."""
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0)
    try:
        # Python 3.10+ statistics.linear_regression
        from statistics import linear_regression  # type: ignore[attr-defined]
        slope, intercept = linear_regression(xs, ys)
        return slope, intercept
    except ImportError:
        pass
    # Manual OLS
    mx = mean(xs)
    my = mean(ys)
    ss_xx = sum((x - mx) ** 2 for x in xs)
    ss_xy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope     = ss_xy / ss_xx if ss_xx != 0 else 0.0
    intercept = my - slope * mx
    return slope, intercept


def _next_period_label(last_label: str, offset: int) -> str:
    """
    Generate the next period label by incrementing a date or period string.
    Examples:
      "2024-01" + 1  → "2024-02"
      "2024-12" + 1  → "2025-01"
      "Q3 2024" + 1  → "Q4 2024"
      "Month 6" + 1  → "Month 7"
      "2024"    + 1  → "2025"
      fallback       → "Forecast +{offset}"
    """
    lbl = last_label.strip()

    # YYYY-MM
    m = re.match(r"^(\d{4})-(\d{2})$", lbl)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        mo += offset
        while mo > 12:
            mo -= 12
            y += 1
        return f"{y}-{mo:02d}"

    # Q[1-4] YYYY or YYYY Q[1-4]
    m = re.match(r"^Q(\d)\s+(\d{4})$", lbl, re.I) or re.match(r"^(\d{4})\s+Q(\d)$", lbl, re.I)
    if m:
        groups = m.groups()
        q, y = (int(groups[0]), int(groups[1])) if "Q" in lbl[:2].upper() else (int(groups[1]), int(groups[0]))
        q += offset
        while q > 4:
            q -= 4
            y += 1
        if "Q" in lbl[:2].upper():
            return f"Q{q} {y}"
        return f"{y} Q{q}"

    # Month N
    m = re.match(r"^(Month|Week|Period)\s+(\d+)$", lbl, re.I)
    if m:
        return f"{m.group(1)} {int(m.group(2)) + offset}"

    # YYYY
    m = re.match(r"^(\d{4})$", lbl)
    if m:
        return str(int(m.group(1)) + offset)

    return f"Forecast +{offset}"


# ══════════════════════════════════════════════════════════════════════════════
# Forecast computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_forecast(
    rows: list[dict],
    period_col: str,
    value_col: str,
    n_periods: int = 3,
) -> list[dict]:
    """
    Fit a linear trend to the value_col series and append n_periods forecast rows.

    Each original row gets `is_forecast: False` and `forecast_value: None`.
    Each appended forecast row gets `is_forecast: True` and the extrapolated
    value in both the value_col and a `forecast_value` key.

    The `__trend_slope` and `__trend_r2` metadata are added to the first row.
    """
    if not rows or not period_col or not value_col:
        return rows

    # Build (x_index, y_value) pairs, skipping nulls
    pairs = []
    for i, row in enumerate(rows):
        v = _to_float(row.get(value_col))
        if v is not None:
            pairs.append((float(i), v))

    if len(pairs) < 2:
        return [{**r, "is_forecast": False, "forecast_value": None} for r in rows]

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    slope, intercept = _ols(xs, ys)

    # R²
    my = mean(ys)
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in pairs)
    r2 = round(1 - ss_res / ss_tot, 4) if ss_tot else None

    # Annotate original rows
    result = []
    for i, row in enumerate(rows):
        new_row = {**row, "is_forecast": False, "forecast_value": None}
        if i == 0:
            new_row["__trend_slope"] = round(slope, 4)
            new_row["__trend_r2"]    = r2
        result.append(new_row)

    # Append forecast rows
    last_label = str(rows[-1].get(period_col, "")) if rows else ""
    n_existing = len(rows)
    for i in range(1, n_periods + 1):
        x_next  = float(n_existing - 1 + i)
        y_next  = slope * x_next + intercept
        y_next  = round(y_next, 4)
        lbl     = _next_period_label(last_label, i)
        fc_row  = {k: None for k in rows[0].keys()}
        fc_row[period_col]  = lbl
        fc_row[value_col]   = y_next
        fc_row["is_forecast"]    = True
        fc_row["forecast_value"] = y_next
        result.append(fc_row)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SQL hint
# ══════════════════════════════════════════════════════════════════════════════

def build_forecast_sql_hint(n_periods: int = 3) -> str:
    return (
        f"TREND FORECAST HINT:\n"
        f"The user wants a {n_periods}-period trend forecast. "
        f"Return the HISTORICAL time-series data — the system will compute the forecast automatically.\n\n"
        f"Required columns:\n"
        f"  1. period_col  — time dimension (e.g. FORMAT(date, 'yyyy-MM'), fiscal_quarter, month_name)\n"
        f"  2. value_col   — the numeric metric to forecast\n\n"
        f"Requirements:\n"
        f"  - Order results chronologically (oldest first)\n"
        f"  - Use a consistent time grain (all monthly, all quarterly, etc.)\n"
        f"  - Include at least 6 periods of history for a reliable trend\n"
        f"  - Aggregate if needed: GROUP BY period, SUM/AVG the metric\n"
        f"  - Do NOT include future rows — the system appends {n_periods} projected periods\n\n"
        f"Example:\n"
        f"  SELECT FORMAT(order_date, 'yyyy-MM') AS month, SUM(revenue) AS revenue\n"
        f"  FROM orders\n"
        f"  GROUP BY FORMAT(order_date, 'yyyy-MM')\n"
        f"  ORDER BY month"
    )
