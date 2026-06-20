"""
core/window_analytics.py
────────────────────────
Window-function analytics — rolling averages, running totals, rank-in-group,
and period-over-period row-level deltas.

Design philosophy
─────────────────
This module is a detection + prompt-injection layer. The SQL is still generated
by the LLM; we detect the user's intent and inject a precise SQL-hint block
into the generation prompt so the LLM emits correct window function syntax for
the connected database dialect (Azure SQL / Snowflake / Oracle).

Supported patterns (detected from natural language)
────────────────────────────────────────────────────
ROLLING_AVG   "3-month rolling average of revenue"
              "moving average sales"
              "smoothed trend"
RUNNING_TOTAL "cumulative revenue YTD"
              "running total of hires"
              "year-to-date cumulative"
RANK_IN_GROUP "rank employees by salary within each department"
              "top performer in each region"
              "who is #1 per branch?"
ROW_DELTA     "month-over-month change for each row"
              "MoM delta by product"
              "year-over-year growth per category"

Entry points
────────────
  detect_window_intent(question) → WindowIntent | None
  build_window_sql_hint(intent, db_type) → str
  compute_rolling_average(rows, value_col, window) → list[dict]  (post-process)
  compute_running_total(rows, value_col, sort_col) → list[dict]  (post-process)
  compute_row_delta(rows, value_col, label_col) → list[dict]     (post-process)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

# ══════════════════════════════════════════════════════════════════════════════
# Intent data model
# ══════════════════════════════════════════════════════════════════════════════

WindowType = Literal["rolling_avg", "running_total", "rank_in_group", "row_delta"]


@dataclass
class WindowIntent:
    type: WindowType
    window_size: int = 3          # for rolling averages
    partition_col: str = ""       # for rank_in_group
    order_col: str = ""           # sort column hint
    value_col: str = ""           # metric column hint
    delta_grain: str = ""         # "month", "quarter", "year" for row_delta
    raw_match: str = ""           # matched phrase for logging


# ══════════════════════════════════════════════════════════════════════════════
# Detection patterns
# ══════════════════════════════════════════════════════════════════════════════

_ROLLING_PATTERNS = [
    re.compile(r"(\d+)[- ]?(month|week|day|quarter|period)[- ]?(?:rolling|moving|smoothed)\s*(?:average|avg|mean)?", re.I),
    re.compile(r"(?:rolling|moving|smoothed)\s*(?:average|avg|mean)(?:\s+(?:of|for|over)\s+(\d+)\s+(month|week|day|quarter|period))?", re.I),
    re.compile(r"(\d+)\s*(?:ma|sma|ema)\b", re.I),  # 3MA, 6SMA
]

_RUNNING_PATTERNS = [
    re.compile(r"\b(cumulative|running|ytd|year.to.date|month.to.date|qtd)\b", re.I),
    re.compile(r"\brunning\s+(?:total|sum|count)\b", re.I),
    re.compile(r"\bcumulative\s+(?:revenue|sales|count|total|sum|hires|attrition)\b", re.I),
]

_RANK_PATTERNS = [
    re.compile(r"\b(?:rank|ranking|ranked)\s+(?:by|within|per|across|in\s+each)\b", re.I),
    re.compile(r"\b(?:rank|ranking|ranked)\b.{0,60}\b(?:within|in\s+each|per\s+(?:group|category|dept|department|region|shift))\b", re.I),
    re.compile(r"\b(?:top|best|worst|highest|lowest)\s+(?:\d+\s+)?(?:in|per|for|within|across)\s+each\b", re.I),
    re.compile(r"\b(?:top|best|worst|highest|lowest)\b.{0,70}\bwithin\s+each\b", re.I),
    re.compile(r"\b#1\s+(?:in|per|for|within)\b", re.I),
    re.compile(r"\bwho\s+(?:is|are)\s+(?:the\s+)?(?:top|best|worst|highest|lowest)\s+(?:in|per|for|within|across)\s+each\b", re.I),
    re.compile(r"\bwho\s+(?:is|are)\s+(?:the\s+)?(?:top|best|worst|highest|lowest)\s+(?:\w+\s+)?(?:in|per|for|within)\s+each\b", re.I),
]

_DELTA_PATTERNS = [
    re.compile(r"\b(month.over.month|m.o.m|mom|quarter.over.quarter|q.o.q|qoq|year.over.year|y.o.y|yoy)\b", re.I),
    re.compile(r"\b(?:period.over.period|week.over.week|wow)\b", re.I),
    re.compile(r"\b(?:change|delta|diff(?:erence)?|variance|growth)\s+(?:per|for\s+each|by)\s+(?:row|month|quarter|year|week|period)\b", re.I),
    re.compile(r"\b(?:monthly|quarterly|weekly|annual)\s+(?:change|delta|growth|variance)\s+(?:per|for\s+each|by)?\s*(?:row|product|category|item)?\b", re.I),
]


def detect_window_intent(question: str) -> WindowIntent | None:
    """
    Parse the question for window-function intent.
    Returns the first matched WindowIntent, or None.
    """
    q = question.strip()

    # ── Rolling / Moving Average ─────────────────────────────────────────────
    for pat in _ROLLING_PATTERNS:
        m = pat.search(q)
        if m:
            # Try to extract window size from the match
            size = 3
            groups = [g for g in (m.groups() or []) if g and g.isdigit()]
            if groups:
                size = min(int(groups[0]), 52)  # cap at 52 weeks
            return WindowIntent(
                type="rolling_avg",
                window_size=size,
                raw_match=m.group(0),
            )

    # ── Running Total / Cumulative ───────────────────────────────────────────
    for pat in _RUNNING_PATTERNS:
        m = pat.search(q)
        if m:
            return WindowIntent(type="running_total", raw_match=m.group(0))

    # ── Row-Level Delta (MoM, YoY, QoQ) ─────────────────────────────────────
    for pat in _DELTA_PATTERNS:
        m = pat.search(q)
        if m:
            text = m.group(0).lower()
            grain = "month"
            if any(x in text for x in ("year", "yoy", "annual")):
                grain = "year"
            elif any(x in text for x in ("quarter", "qoq")):
                grain = "quarter"
            elif any(x in text for x in ("week", "wow")):
                grain = "week"
            return WindowIntent(type="row_delta", delta_grain=grain, raw_match=m.group(0))

    # ── Rank Within Group ────────────────────────────────────────────────────
    for pat in _RANK_PATTERNS:
        m = pat.search(q)
        if m:
            return WindowIntent(type="rank_in_group", raw_match=m.group(0))

    return None


# ══════════════════════════════════════════════════════════════════════════════
# SQL hint builder
# ══════════════════════════════════════════════════════════════════════════════

def build_window_sql_hint(intent: WindowIntent, db_type: str = "azure_sql") -> str:
    """
    Return a SQL construction hint block to inject into the LLM system prompt.
    The hint is dialect-aware (T-SQL vs Snowflake vs Oracle).
    """
    is_tsql     = db_type in ("azure_sql", "sql_server", "mssql")
    is_oracle   = db_type == "oracle"
    # Snowflake + DuckDB use standard SQL window syntax

    if intent.type == "rolling_avg":
        n = intent.window_size
        if is_tsql:
            example = (
                f"AVG(metric_col) OVER (PARTITION BY category ORDER BY period_col "
                f"ROWS BETWEEN {n - 1} PRECEDING AND CURRENT ROW) AS rolling_avg_{n}"
            )
        else:
            example = (
                f"AVG(metric_col) OVER (PARTITION BY category ORDER BY period_col "
                f"ROWS BETWEEN {n - 1} PRECEDING AND CURRENT ROW) AS rolling_avg_{n}"
            )
        return (
            f"WINDOW FUNCTION HINT — ROLLING AVERAGE:\n"
            f"The user wants a {n}-period rolling average. Use this pattern:\n"
            f"  {example}\n"
            f"Replace metric_col with the actual numeric column and period_col with the date/period column.\n"
            f"If there is no natural partition (single series), omit the PARTITION BY clause.\n"
            f"Include BOTH the raw metric and the rolling average as separate columns.\n"
            f"Order the result by the period column ascending."
        )

    if intent.type == "running_total":
        if is_tsql:
            example = "SUM(metric_col) OVER (PARTITION BY category ORDER BY period_col ROWS UNBOUNDED PRECEDING) AS running_total"
        else:
            example = "SUM(metric_col) OVER (PARTITION BY category ORDER BY period_col) AS running_total"
        return (
            "WINDOW FUNCTION HINT — RUNNING TOTAL:\n"
            "The user wants a cumulative / running total. Use this pattern:\n"
            f"  {example}\n"
            "Replace metric_col with the actual numeric column and period_col with the date/period column.\n"
            "If there is no natural partition (single series), omit the PARTITION BY clause.\n"
            "Include BOTH the period value and the running total as separate columns.\n"
            "Order the result by the period column ascending."
        )

    if intent.type == "rank_in_group":
        if is_oracle:
            rank_fn = "RANK() OVER (PARTITION BY group_col ORDER BY metric_col DESC)"
        else:
            rank_fn = "RANK() OVER (PARTITION BY group_col ORDER BY metric_col DESC)"
        return (
            "WINDOW FUNCTION HINT — RANK WITHIN GROUP:\n"
            "The user wants to rank items within a category. Use this pattern:\n"
            f"  {rank_fn} AS rank_in_group\n"
            "Replace group_col with the partition dimension (e.g. region, department) "
            "and metric_col with the value to rank by.\n"
            "To show only rank 1 per group, wrap in a CTE or subquery and filter WHERE rank_in_group = 1.\n"
            "Include the group column, the metric column, and the rank column in SELECT."
        )

    if intent.type == "row_delta":
        grain_label = intent.delta_grain or "month"
        if is_tsql:
            lag_expr = "LAG(metric_col, 1) OVER (PARTITION BY category ORDER BY period_col)"
        else:
            lag_expr = "LAG(metric_col, 1) OVER (PARTITION BY category ORDER BY period_col)"
        return (
            f"WINDOW FUNCTION HINT — {grain_label.upper()}-OVER-{grain_label.upper()} DELTA:\n"
            "The user wants the change vs the prior period for each row. Use this pattern:\n"
            f"  {lag_expr} AS prev_value,\n"
            f"  metric_col - {lag_expr} AS delta,\n"
            f"  CASE WHEN {lag_expr} <> 0 THEN\n"
            f"    (metric_col - {lag_expr}) * 100.0 / ABS({lag_expr})\n"
            f"  ELSE NULL END AS pct_change\n"
            "Replace metric_col with the actual numeric column and period_col with the date column.\n"
            "If there is no natural partition (single series), omit the PARTITION BY clause.\n"
            "Include the period, current value, previous value, absolute delta, and % change."
        )

    return ""


# ══════════════════════════════════════════════════════════════════════════════
# Pure Python post-processing (no DB required)
# Used when we have rows in memory and want to compute window metrics client-side
# ══════════════════════════════════════════════════════════════════════════════

def _to_float(v: Any) -> float | None:
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def compute_rolling_average(
    rows: list[dict],
    value_col: str,
    window: int = 3,
    label_col: str = "",
) -> list[dict]:
    """
    Compute a rolling average over `rows` ordered as-is.
    Adds a ``rolling_avg`` key to each row dict.
    Rows where the window is incomplete get a partial average.
    """
    result = []
    vals: list[float | None] = []
    col_out = f"rolling_avg_{window}"

    for row in rows:
        v = _to_float(row.get(value_col))
        vals.append(v)
        window_vals = [x for x in vals[-window:] if x is not None]
        avg = round(sum(window_vals) / len(window_vals), 4) if window_vals else None
        result.append({**row, col_out: avg})

    return result


def compute_running_total(
    rows: list[dict],
    value_col: str,
    label_col: str = "",
) -> list[dict]:
    """
    Compute a cumulative running total over `rows` ordered as-is.
    Adds a ``running_total`` key to each row dict.
    """
    result = []
    cumulative = 0.0

    for row in rows:
        v = _to_float(row.get(value_col))
        if v is not None:
            cumulative += v
        result.append({**row, "running_total": round(cumulative, 4) if v is not None else None})

    return result


def compute_row_delta(
    rows: list[dict],
    value_col: str,
    label_col: str = "",
) -> list[dict]:
    """
    Compute period-over-period absolute and % change for each row.
    Adds ``prev_value``, ``delta``, and ``pct_change`` keys.
    """
    result = []
    prev: float | None = None

    for row in rows:
        v = _to_float(row.get(value_col))
        delta = None
        pct   = None
        if v is not None and prev is not None:
            delta = round(v - prev, 4)
            pct   = round((delta / abs(prev)) * 100, 2) if prev != 0 else None
        result.append({**row, "prev_value": prev, "delta": delta, "pct_change": pct})
        if v is not None:
            prev = v

    return result


def compute_rank_in_group(
    rows: list[dict],
    value_col: str,
    group_col: str,
    descending: bool = True,
) -> list[dict]:
    """
    Compute RANK() OVER (PARTITION BY group_col ORDER BY value_col).
    Adds a ``rank_in_group`` key to each row dict.
    Uses dense ranking within each group.
    """
    from collections import defaultdict

    # Group rows
    groups: dict[Any, list[tuple[int, float | None]]] = defaultdict(list)
    for i, row in enumerate(rows):
        g   = row.get(group_col, "")
        v   = _to_float(row.get(value_col))
        groups[g].append((i, v))

    # Compute rank within each group
    ranks: dict[int, int] = {}
    for g, indexed_vals in groups.items():
        # Sort by value (descending by default), None last
        sorted_iv = sorted(
            indexed_vals,
            key=lambda x: (x[1] is None, -(x[1] or 0) if descending else (x[1] or 0)),
        )
        for rank, (orig_idx, _) in enumerate(sorted_iv, start=1):
            ranks[orig_idx] = rank

    return [{**row, "rank_in_group": ranks.get(i)} for i, row in enumerate(rows)]
