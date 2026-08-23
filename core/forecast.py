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

from core.forecast_models import fit_series
from core.temporal_columns import infer_series_grain, parse_period_label


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

# One unit per calendar period, by grain. Day ordinals are the wrong unit here:
# January is 31 days and February is 28, so an evenly spaced monthly series
# measured in days is not evenly spaced at all, and fitting it that way tilts
# the slope by around 3% and mislabels it "per period". Counting months instead
# makes consecutive months exactly one apart and a six-month gap exactly six.
_GRAIN_INDEX = {
    "year": lambda d: float(d.year),
    "quarter": lambda d: float(d.year * 4 + (d.month - 1) // 3),
    "month": lambda d: float(d.year * 12 + d.month),
    "week": lambda d: d.toordinal() / 7.0,
    "day": lambda d: float(d.toordinal()),
}


def _period_units(parsed: list, labels: list) -> list[float] | None:
    """x in whole periods, spaced by the calendar rather than by row number.

    For a series with no gaps this is exactly the row index, so the projection
    is unchanged -- which is what makes the switch safe on the gated path, where
    core.forecast_gate has already required a regular cadence. It only changes
    the answer where the spacing is uneven, and there the row index would have
    treated a six-month gap as one step.

    Returns None -- meaning "fall back to row index" -- when a label did not
    parse, the grain is unknown, or the series is not strictly increasing.
    """
    if len(parsed) < 2 or any(d is None for d in parsed):
        return None
    grain, _consistency = infer_series_grain(labels)
    index_of = _GRAIN_INDEX.get(grain)
    if index_of is None:
        return None
    idx = [index_of(d) for d in parsed]
    if any(b <= a for a, b in zip(idx, idx[1:])):
        return None
    gaps = [b - a for a, b in zip(idx, idx[1:])]
    unit = sorted(gaps)[len(gaps) // 2] or 1.0
    return [(i - idx[0]) / unit for i in idx]


def compute_forecast(
    rows: list[dict],
    period_col: str,
    value_col: str,
    n_periods: int = 3,
    model: str = "ols",
    seasonal_period: int = 0,
) -> list[dict]:
    """
    Fit the value_col series and append n_periods projected rows.

    Each original row gets `is_forecast: False` and `forecast_value: None`.
    Each appended row gets `is_forecast: True`, the projection in both the
    value_col and `forecast_value`, and a 95% prediction interval in
    `forecast_low` / `forecast_high`.

    THE INTERVAL IS THE POINT. A bare projected number is a claim about the
    future; the band is what turns it into a forecast. Six flat months of
    revenue project to "about 7.5M" either way, but only one of the two says
    "give or take 400K".

    `model` and `seasonal_period` are chosen by core.forecast_gate from the
    length and seasonality of the series. They default to plain OLS so every
    existing caller keeps exactly the behaviour it had.

    `__trend_slope` and `__trend_r2` stay on row 0 for one release; the full
    picture is in `__forecast_meta`, which core.chart hoists onto the payload.
    """
    if not rows or not period_col or not value_col:
        return rows

    values: list[float] = []
    parsed: list = []
    labels: list = []
    for row in rows:
        v = _to_float(row.get(value_col))
        if v is None:
            continue
        values.append(v)
        labels.append(row.get(period_col))
        parsed.append(parse_period_label(row.get(period_col)))

    if len(values) < 2:
        return [{**r, "is_forecast": False, "forecast_value": None} for r in rows]

    calendar_xs = _period_units(parsed, labels)
    xs = calendar_xs or [float(i) for i in range(len(values))]
    fit = fit_series(
        values, n_periods, model=model, seasonal_period=seasonal_period, xs=xs,
    )
    if fit is None:
        return [{**r, "is_forecast": False, "forecast_value": None} for r in rows]

    # The descriptive trend line the chart captions, which is a straight line
    # even when the projection came from ETS or SARIMAX.
    slope, _intercept = _ols(xs, values)
    my = mean(values)
    ss_tot = sum((y - my) ** 2 for y in values)
    if fit.r2 is not None:
        r2 = round(fit.r2, 4)
    elif ss_tot:
        _s, _i = slope, _intercept
        r2 = round(1 - sum((y - (_s * x + _i)) ** 2 for x, y in zip(xs, values)) / ss_tot, 4)
    else:
        r2 = None

    meta = {
        "model": fit.model,
        "slope": round(slope, 4),
        "r2": r2,
        "backtest_mape": None if fit.backtest_mape is None else round(fit.backtest_mape, 2),
        "horizon": n_periods,
        "n_points": len(values),
        "interval_confidence": 0.95,
        "fitted_on": "period_dates" if calendar_xs else "row_index",
    }
    if fit.fell_back_from:
        meta["fell_back_from"] = fit.fell_back_from

    result = []
    for i, row in enumerate(rows):
        new_row = {**row, "is_forecast": False, "forecast_value": None,
                   "forecast_low": None, "forecast_high": None}
        if i == 0:
            new_row["__trend_slope"] = meta["slope"]
            new_row["__trend_r2"] = r2
            new_row["__forecast_meta"] = meta
        result.append(new_row)

    last_label = str(rows[-1].get(period_col, "")) if rows else ""
    for i in range(1, n_periods + 1):
        y_next = round(fit.predictions[i - 1], 4)
        fc_row = {k: None for k in rows[0].keys()}
        fc_row[period_col] = _next_period_label(last_label, i)
        fc_row[value_col] = y_next
        fc_row["is_forecast"] = True
        fc_row["forecast_value"] = y_next
        fc_row["forecast_low"] = round(fit.lower[i - 1], 4)
        fc_row["forecast_high"] = round(fit.upper[i - 1], 4)
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
