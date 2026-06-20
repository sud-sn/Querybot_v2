"""
core/pivot_table.py
────────────────────
Pivot / cross-tab output transformation.

Answers questions like:
  "Pivot sales by region across quarters"
  "Show a cross-tab of product vs month"
  "Revenue matrix: row = department, column = year"
  "Cross-tabulation of channel by customer segment"

Design
──────
• Detection: regex on NL question.
• Post-processing: rotates flat (row_key, col_key, value) rows into a pivot
  table where the col_key values become column headers.
• SQL hint: instructs LLM to produce the un-pivoted flat format that this
  module can then rotate into the final presentation.
• Output: list of dicts where each dict is one row of the pivot with the
  dimension keys as columns.  The chart type is NOT set (pivot is a table view).

Entry points
────────────
  detect_pivot_intent(question) → bool
  infer_pivot_cols(rows) → tuple[str, str, str]  (row_key, col_key, value_key)
  compute_pivot_table(rows, row_key, col_key, value_key) → list[dict]
  build_pivot_sql_hint() → str
"""

from __future__ import annotations

import re
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Detection
# ══════════════════════════════════════════════════════════════════════════════

_PIVOT_PATTERNS = [
    re.compile(r"\bpivot\b", re.I),
    re.compile(r"\bcross[- ]?tab(?:ulation)?\b", re.I),
    re.compile(r"\bmatrix\s+(?:of|by|showing)\b", re.I),
    re.compile(r"\b\w+\s+(?:by|across)\s+\w+\s+(?:by|across)\s+\w+\b", re.I),  # "X by Y across Z"
    re.compile(r"\b(?:row|rows)\s*=\s*\w+.{0,20}(?:column|col)s?\s*=\s*\w+\b", re.I),
    re.compile(r"\b(?:column|col)s?\s+(?:are|as|=)\s+\w+.{0,20}(?:row|rows)\s+(?:are|as|=)\s+\w+\b", re.I),
    re.compile(r"\bspread\s+(?:out|across)\s+(?:months|quarters|years|regions|columns)\b", re.I),
    re.compile(r"\breshape\s+(?:the\s+)?(?:data|result|table)\b", re.I),
]


def detect_pivot_intent(question: str) -> bool:
    """Return True if the question asks for a pivot / cross-tab view."""
    return any(p.search(question) for p in _PIVOT_PATTERNS)


# ══════════════════════════════════════════════════════════════════════════════
# Column inference
# ══════════════════════════════════════════════════════════════════════════════

def _to_float(v: Any) -> float | None:
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def infer_pivot_cols(rows: list[dict]) -> tuple[str, str, str]:
    """
    Infer (row_key, col_key, value_key) for pivot.

    Heuristics:
    • value_key  = first numeric column
    • row_key    = first text column (likely the "outer" dimension)
    • col_key    = second text column (to be spread as columns)
    Returns ("", "", "") if the shape doesn't support pivoting.
    """
    if not rows:
        return "", "", ""

    keys = list(rows[0].keys())
    numeric = [k for k in keys if _to_float(rows[0].get(k)) is not None]
    text    = [k for k in keys if k not in numeric]

    if len(text) < 2 or not numeric:
        return "", "", ""

    return text[0], text[1], numeric[0]


# ══════════════════════════════════════════════════════════════════════════════
# Pivot computation
# ══════════════════════════════════════════════════════════════════════════════

_MAX_PIVOT_COLS = 50    # cap column explosion


def compute_pivot_table(
    rows: list[dict],
    row_key: str,
    col_key: str,
    value_key: str,
    agg: str = "sum",
) -> list[dict]:
    """
    Rotate flat (row_key, col_key, value_key) rows into a pivot table.

    Parameters
    ──────────
    rows      — flat source rows
    row_key   — column whose values become pivot rows
    col_key   — column whose values become pivot column headers
    value_key — numeric metric column to aggregate at each (row, col) intersection
    agg       — aggregation method: "sum" | "avg" | "max" | "min" | "count"

    Returns a list of dicts — one per unique row_key value — where each
    dict has all unique col_key values as columns (plus row_key itself).
    An "TOTAL" column is appended summing all pivot columns.

    Example input:
      [{"region": "EMEA", "quarter": "Q1", "revenue": 100},
       {"region": "EMEA", "quarter": "Q2", "revenue": 120},
       {"region": "APAC", "quarter": "Q1", "revenue": 80}]

    Example output (agg=sum):
      [{"region": "EMEA", "Q1": 100, "Q2": 120, "TOTAL": 220},
       {"region": "APAC", "Q1": 80,  "Q2": None, "TOTAL": 80}]
    """
    if not rows or not row_key or not col_key or not value_key:
        return rows

    # Build nested dict: row_val → col_val → [values]
    data: dict[Any, dict[Any, list[float]]] = {}
    row_order: list[Any] = []
    col_order: list[Any] = []
    seen_cols: set[Any] = set()

    for row in rows:
        rv = row.get(row_key, "")
        cv = row.get(col_key, "")
        vv = _to_float(row.get(value_key))
        if rv not in data:
            data[rv] = {}
            row_order.append(rv)
        if cv not in seen_cols:
            col_order.append(cv)
            seen_cols.add(cv)
        data[rv].setdefault(cv, [])
        if vv is not None:
            data[rv][cv].append(vv)

    # Cap column explosion
    if len(col_order) > _MAX_PIVOT_COLS:
        col_order = col_order[:_MAX_PIVOT_COLS]

    def _agg_values(vals: list[float]) -> float | None:
        if not vals:
            return None
        if agg == "sum":
            return round(sum(vals), 4)
        if agg == "avg":
            return round(sum(vals) / len(vals), 4)
        if agg == "max":
            return round(max(vals), 4)
        if agg == "min":
            return round(min(vals), 4)
        if agg == "count":
            return float(len(vals))
        return round(sum(vals), 4)

    pivot_rows = []
    for rv in row_order:
        row_dict: dict[str, Any] = {row_key: rv}
        row_total = 0.0
        row_has_value = False
        for cv in col_order:
            cell = _agg_values(data[rv].get(cv, []))
            row_dict[str(cv)] = cell
            if cell is not None:
                row_total += cell
                row_has_value = True
        row_dict["TOTAL"] = round(row_total, 4) if row_has_value else None
        pivot_rows.append(row_dict)

    return pivot_rows


# ══════════════════════════════════════════════════════════════════════════════
# SQL hint
# ══════════════════════════════════════════════════════════════════════════════

def build_pivot_sql_hint() -> str:
    return (
        "PIVOT TABLE HINT:\n"
        "The user wants a cross-tabulation (pivot table) of the data. "
        "Return a FLAT, un-pivoted result with exactly 3 columns:\n\n"
        "  1. ROW dimension  — the values that become rows (e.g. department, region)\n"
        "  2. COLUMN dimension — the values that become column headers (e.g. quarter, year)\n"
        "  3. METRIC value   — the numeric value at each (row, column) intersection\n\n"
        "Do NOT use SQL PIVOT syntax — return the un-pivoted flat format.\n"
        "The system will automatically rotate it into a pivot table.\n\n"
        "Example:\n"
        "  SELECT department, quarter, SUM(revenue) AS revenue\n"
        "  FROM sales\n"
        "  GROUP BY department, quarter\n"
        "  ORDER BY department, quarter\n\n"
        "Keep the column dimension to ≤ 20 distinct values to avoid extremely wide tables."
    )
