"""
core/distribution_analysis.py
──────────────────────────────
Statistical distribution analysis — histogram binning and box-plot statistics.

Answers questions like:
  "Show the distribution of salaries"
  "Histogram of order values"
  "Box plot of revenue by region"
  "What is the spread of processing times?"
  "Show quartile breakdown of customer lifetime value"
  "Distribution of defect rates by production line"

Design
──────
• Detection: separate `detect_histogram_intent()` and `detect_boxplot_intent()`.
• `compute_histogram()`: bins a numeric column into N equal-width buckets,
  returns one row per bin with (bin_label, bin_min, bin_max, count, frequency_pct).
• `compute_boxplot()`: computes (min, Q1, median, Q3, max) per category group.
  Returns rows formatted for ECharts type:'boxplot'.
• Both are triggered as post-processing in query_pipeline.py if the SQL result
  returns the raw numeric column (not pre-aggregated).

Entry points
────────────
  detect_histogram_intent(question) → bool
  detect_boxplot_intent(question) → bool
  infer_histogram_col(rows) → str
  infer_boxplot_cols(rows) → tuple[str, str]   (group_col, value_col)
  compute_histogram(rows, value_col, n_bins) → list[dict]
  compute_boxplot(rows, value_col, group_col) → list[dict]
  build_histogram_sql_hint() → str
  build_boxplot_sql_hint() → str
"""

from __future__ import annotations

import re
import math
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Detection
# ══════════════════════════════════════════════════════════════════════════════

_HIST_PATTERNS = [
    re.compile(r"\bhistogram\b", re.I),
    re.compile(r"\bdistribution\s+of\b", re.I),
    re.compile(r"\bfrequency\s+(?:distribution|chart|plot)\b", re.I),
    re.compile(r"\bhow\s+(?:are|is)\s+.{0,30}\s+distributed\b", re.I),
    # "spread of X" without requiring a trailing qualifier word
    re.compile(r"\bspread\s+of\b", re.I),
    # "bin X into ranges" — allow arbitrary words between "bin" and "into"
    re.compile(r"\bbin(?:ned|ning|s)?\b.{0,30}\b(?:by|into|range[s]?|bucket[s]?)\b", re.I),
    re.compile(r"\b(?:value|amount|salary|price|cost)\s+(?:range|bucket|band)\b", re.I),
]

_BOX_PATTERNS = [
    re.compile(r"\bbox\s*(?:plot|chart|diagram|and\s+whisker)\b", re.I),
    re.compile(r"\bwhisker\s+(?:plot|chart)\b", re.I),
    re.compile(r"\bquartile[s]?\s+(?:of|by|for|analysis|breakdown)\b", re.I),
    # allow "of" as preposition, and allow an intermediate word before the preposition
    re.compile(r"\b(?:median|iqr|interquartile)\s+(?:\w+\s+)?(?:by|per|for|across|of)\b", re.I),
    re.compile(r"\b(?:p25|p75|q1|q3|p50)\b", re.I),
    # "spread by/per/across" OR "spread of X by"
    re.compile(r"\bspread\s+(?:by|per|across|for)\s+\w+\b", re.I),
    re.compile(r"\bspread\s+of\b.{0,40}\bby\b", re.I),
    re.compile(r"\boutlier[s]?\s+(?:by|per|in)\s+\w+\b", re.I),
]


def detect_histogram_intent(question: str) -> bool:
    return any(p.search(question) for p in _HIST_PATTERNS)


def detect_boxplot_intent(question: str) -> bool:
    return any(p.search(question) for p in _BOX_PATTERNS)


# ══════════════════════════════════════════════════════════════════════════════
# Column inference
# ══════════════════════════════════════════════════════════════════════════════

_NUMERIC_KWS = {
    "value", "amount", "salary", "revenue", "cost", "price", "duration",
    "time", "rate", "score", "age", "days", "hours", "quantity", "count",
}
_GROUP_KWS = {
    "region", "department", "product", "category", "channel", "segment",
    "team", "group", "type", "class", "status",
}


def _to_float(v: Any) -> float | None:
    try:
        return float(str(v).replace(",", "").replace("$", "").replace("%", ""))
    except (TypeError, ValueError):
        return None


def _kw_score(col: str, kws: set[str]) -> int:
    cl = col.lower()
    return sum(1 for kw in kws if kw in cl)


def infer_histogram_col(rows: list[dict]) -> str:
    """Return the best numeric column for histogram binning."""
    if not rows:
        return ""
    cols = list(rows[0].keys())
    numeric = [c for c in cols if _to_float(rows[0].get(c)) is not None]
    if not numeric:
        return ""
    return max(numeric, key=lambda c: _kw_score(c, _NUMERIC_KWS))


def infer_boxplot_cols(rows: list[dict]) -> tuple[str, str]:
    """Return (group_col, value_col) for boxplot."""
    if not rows:
        return "", ""
    cols = list(rows[0].keys())
    numeric = [c for c in cols if _to_float(rows[0].get(c)) is not None]
    text    = [c for c in cols if c not in numeric]
    if not numeric:
        return "", ""
    value_col = max(numeric, key=lambda c: _kw_score(c, _NUMERIC_KWS))
    group_col = max(text, key=lambda c: _kw_score(c, _GROUP_KWS)) if text else ""
    return group_col, value_col


# ══════════════════════════════════════════════════════════════════════════════
# Histogram computation
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_BINS = 10
_MAX_BINS     = 50


def compute_histogram(
    rows: list[dict],
    value_col: str,
    n_bins: int = _DEFAULT_BINS,
) -> list[dict]:
    """
    Bin value_col into n_bins equal-width buckets.

    Returns one row per bin:
      bin_label       — human-readable range label e.g. "10K – 20K"
      bin_min         — lower bound (inclusive)
      bin_max         — upper bound (exclusive, except last)
      count           — number of rows in this bin
      frequency_pct   — count / total * 100
    """
    if not rows or not value_col:
        return rows

    values = [_to_float(r.get(value_col)) for r in rows]
    values = [v for v in values if v is not None]
    if not values:
        return rows

    n_bins = min(max(n_bins, 2), _MAX_BINS)
    lo, hi = min(values), max(values)

    if lo == hi:
        return [{"bin_label": str(lo), "bin_min": lo, "bin_max": lo, "count": len(values), "frequency_pct": 100.0}]

    width = (hi - lo) / n_bins
    total = len(values)

    bins: list[int] = [0] * n_bins
    for v in values:
        idx = min(int((v - lo) / width), n_bins - 1)
        bins[idx] += 1

    def _fmt(v: float) -> str:
        if abs(v) >= 1_000_000:
            return f"{v/1_000_000:.1f}M"
        if abs(v) >= 1_000:
            return f"{v/1_000:.1f}K"
        return f"{v:.2f}".rstrip("0").rstrip(".")

    result = []
    for i, cnt in enumerate(bins):
        b_min = lo + i * width
        b_max = lo + (i + 1) * width
        label = f"{_fmt(b_min)} – {_fmt(b_max)}"
        result.append({
            "bin_label":     label,
            "bin_min":       round(b_min, 6),
            "bin_max":       round(b_max, 6),
            "count":         cnt,
            "frequency_pct": round(cnt / total * 100, 2) if total else 0,
        })
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Box-plot computation
# ══════════════════════════════════════════════════════════════════════════════

def _quartiles(sorted_vals: list[float]) -> tuple[float, float, float]:
    """Return (Q1, median, Q3) for a sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return sorted_vals[0], sorted_vals[0], sorted_vals[0]

    def _percentile(p: float) -> float:
        pos = p / 100 * (n - 1)
        lo, hi = int(pos), min(int(pos) + 1, n - 1)
        frac = pos - lo
        return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

    return _percentile(25), _percentile(50), _percentile(75)


def compute_boxplot(
    rows: list[dict],
    value_col: str,
    group_col: str = "",
) -> list[dict]:
    """
    Compute box-plot statistics per group (or for the entire dataset if no group).

    Returns one row per group:
      group           — group label (or "All" if no group)
      bp_min          — min (or 1.5×IQR lower fence)
      bp_q1           — 25th percentile
      bp_median       — 50th percentile
      bp_q3           — 75th percentile
      bp_max          — max (or 1.5×IQR upper fence)
      bp_mean         — arithmetic mean
      bp_count        — number of data points
      bp_outliers     — list of values outside the whiskers (for ECharts scatter)

    The ECharts boxplot series expects data as [min, Q1, median, Q3, max].
    The rows returned include these as separate keys AND as `bp_data` list.
    """
    if not rows or not value_col:
        return rows

    # Group the values
    groups: dict[str, list[float]] = {}
    for row in rows:
        v = _to_float(row.get(value_col))
        if v is None:
            continue
        g = str(row.get(group_col, "All")) if group_col else "All"
        groups.setdefault(g, []).append(v)

    if not groups:
        return rows

    result = []
    for group, vals in groups.items():
        vals.sort()
        n = len(vals)
        q1, median, q3 = _quartiles(vals)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        whisker_lo  = min((v for v in vals if v >= lower_fence), default=q1)
        whisker_hi  = max((v for v in vals if v <= upper_fence), default=q3)
        outliers    = [v for v in vals if v < lower_fence or v > upper_fence]
        mean_val    = sum(vals) / n

        result.append({
            "group":       group,
            "bp_min":      round(whisker_lo, 4),
            "bp_q1":       round(q1, 4),
            "bp_median":   round(median, 4),
            "bp_q3":       round(q3, 4),
            "bp_max":      round(whisker_hi, 4),
            "bp_mean":     round(mean_val, 4),
            "bp_count":    n,
            "bp_outliers": [round(v, 4) for v in outliers],
            # Pre-built list for ECharts boxplot series data array
            "bp_data":     [round(whisker_lo, 4), round(q1, 4), round(median, 4),
                            round(q3, 4), round(whisker_hi, 4)],
        })

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SQL hints
# ══════════════════════════════════════════════════════════════════════════════

def build_histogram_sql_hint() -> str:
    return (
        "HISTOGRAM / DISTRIBUTION HINT:\n"
        "The user wants to see the distribution of a numeric value. "
        "Return UN-AGGREGATED individual rows — the system will bin them automatically.\n\n"
        "Required:\n"
        "  - Exactly ONE numeric column (the value to distribute)\n"
        "  - Optionally one label column (product, customer, order, etc.)\n\n"
        "Do NOT group or aggregate. Return each individual value as a row.\n"
        "Add a TOP/LIMIT 10000 if the table is large.\n\n"
        "Example:\n"
        "  SELECT order_id, order_value FROM orders\n"
        "  ORDER BY order_value"
    )


def build_boxplot_sql_hint() -> str:
    return (
        "BOX PLOT / QUARTILE HINT:\n"
        "The user wants quartile statistics (min, Q1, median, Q3, max). "
        "Return UN-AGGREGATED individual rows — the system computes quartiles automatically.\n\n"
        "Required:\n"
        "  1. One GROUP column  (department, region, product, etc.)\n"
        "  2. One METRIC column (salary, revenue, duration, etc.)\n\n"
        "Return individual row-level data, NOT pre-aggregated statistics.\n"
        "Add a TOP/LIMIT 5000 if the table is large.\n\n"
        "Example:\n"
        "  SELECT department, salary FROM employees\n"
        "  ORDER BY department, salary"
    )
