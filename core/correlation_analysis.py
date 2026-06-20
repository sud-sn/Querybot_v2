"""
core/correlation_analysis.py
─────────────────────────────
Correlation analysis for numerical column pairs.

Answers questions like:
  "Is there a correlation between marketing spend and revenue?"
  "Show the relationship between units sold and profit margin"
  "Scatter plot of price vs sales volume"
  "Do customer satisfaction scores correlate with repeat purchase rate?"

Design
──────
• Detection: regex on NL question.
• Post-processing: computes Pearson r for all numeric column pairs and
  annotates rows with correlation metadata for chart display.
• SQL hint: inject hint to SELECT the two relevant numeric columns for
  scatter analysis (LLM chooses the specific columns based on KB).
• Chart output: "scatter" type with Pearson r and interpretation label
  added to the payload.

Entry points
────────────
  detect_correlation_intent(question) → bool
  infer_corr_cols(rows, question) → tuple[str, str]   (x_col, y_col)
  compute_correlation(rows, x_col, y_col) → CorrelationResult
  build_correlation_sql_hint() → str
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Detection
# ══════════════════════════════════════════════════════════════════════════════

_CORR_PATTERNS = [
    re.compile(r"\bcorrelat(?:ion|e|es|ed|ing)?\b", re.I),
    re.compile(r"\brelationship\s+between\b", re.I),
    re.compile(r"\bscatter\s+(?:plot|chart|graph|diagram)\b", re.I),
    re.compile(r"\bdo(?:es)?\s+.{0,40}\s+(?:affect|impact|influence|predict|drive)\b", re.I),
    re.compile(r"\b(?:x|y)\s+(?:axis|vs\.?)\s+\w", re.I),
    re.compile(r"\b(?:positively|negatively)\s+(?:correlated|related)\b", re.I),
    re.compile(r"\blinear\s+(?:relationship|correlation|regression)\b", re.I),
    re.compile(r"\bhow\s+(?:closely|strongly|well)\s+.{0,30}\s+(?:correlat|relate|match)\b", re.I),
]


def detect_correlation_intent(question: str) -> bool:
    """Return True if the question asks for correlation or scatter analysis."""
    return any(p.search(question) for p in _CORR_PATTERNS)


# ══════════════════════════════════════════════════════════════════════════════
# Column inference
# ══════════════════════════════════════════════════════════════════════════════

def _to_float(v: Any) -> float | None:
    try:
        f = float(str(v).replace(",", "").replace("$", "").replace("%", ""))
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _numeric_cols(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    return [
        col for col in rows[0].keys()
        if sum(1 for r in rows[:20] if _to_float(r.get(col)) is not None) >= min(len(rows[:20]), 3)
    ]


def infer_corr_cols(rows: list[dict], question: str = "") -> tuple[str, str]:
    """
    Infer the two columns to correlate.

    Priority:
    1. Mention in the question (matches column names in the question text)
    2. First two numeric columns
    Returns ("", "") if fewer than 2 numeric columns exist.
    """
    if not rows:
        return "", ""
    num_cols = _numeric_cols(rows)
    if len(num_cols) < 2:
        return "", ""

    # Try to match column names mentioned in the question
    q_lower = question.lower()
    mentioned = [c for c in num_cols if c.lower().replace("_", " ") in q_lower or c.lower() in q_lower]
    if len(mentioned) >= 2:
        return mentioned[0], mentioned[1]

    return num_cols[0], num_cols[1]


# ══════════════════════════════════════════════════════════════════════════════
# Pearson correlation
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CorrelationResult:
    x_col: str
    y_col: str
    pearson_r: float | None
    r_squared: float | None
    interpretation: str     # "strong positive" | "moderate negative" | etc.
    n: int                  # number of valid pairs
    x_mean: float | None
    y_mean: float | None


def _interpret_r(r: float | None) -> str:
    if r is None:
        return "insufficient data"
    abs_r = abs(r)
    direction = "positive" if r > 0 else "negative"
    if abs_r >= 0.8:
        return f"strong {direction}"
    if abs_r >= 0.5:
        return f"moderate {direction}"
    if abs_r >= 0.2:
        return f"weak {direction}"
    return "negligible"


def compute_correlation(
    rows: list[dict],
    x_col: str,
    y_col: str,
) -> CorrelationResult:
    """
    Compute Pearson r between x_col and y_col.
    Annotates each row with pearson_r, interpretation metadata.
    """
    pairs = [
        (_to_float(r.get(x_col)), _to_float(r.get(y_col)))
        for r in rows
        if _to_float(r.get(x_col)) is not None and _to_float(r.get(y_col)) is not None
    ]

    n = len(pairs)
    if n < 4:
        return CorrelationResult(
            x_col=x_col, y_col=y_col, pearson_r=None, r_squared=None,
            interpretation="insufficient data", n=n, x_mean=None, y_mean=None,
        )

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    try:
        mx, my = mean(xs), mean(ys)
        sx, sy = stdev(xs), stdev(ys)
    except Exception:
        return CorrelationResult(
            x_col=x_col, y_col=y_col, pearson_r=None, r_squared=None,
            interpretation="computation error", n=n, x_mean=None, y_mean=None,
        )

    if sx == 0 or sy == 0:
        return CorrelationResult(
            x_col=x_col, y_col=y_col, pearson_r=None, r_squared=None,
            interpretation="no variance", n=n, x_mean=round(mx, 4), y_mean=round(my, 4),
        )

    r = sum((x - mx) * (y - my) for x, y in pairs) / ((n - 1) * sx * sy)
    r = max(-1.0, min(1.0, round(r, 4)))

    return CorrelationResult(
        x_col=x_col,
        y_col=y_col,
        pearson_r=r,
        r_squared=round(r ** 2, 4),
        interpretation=_interpret_r(r),
        n=n,
        x_mean=round(mx, 4),
        y_mean=round(my, 4),
    )


def annotate_rows_with_correlation(
    rows: list[dict],
    result: CorrelationResult,
) -> list[dict]:
    """
    Return rows unchanged (correlation is a summary stat, not per-row).
    Adds a __corr_meta__ key to the first row only for chart payload injection.
    """
    if not rows:
        return rows
    enriched = list(rows)
    enriched[0] = {
        **enriched[0],
        "__corr_r": result.pearson_r,
        "__corr_r2": result.r_squared,
        "__corr_label": result.interpretation,
        "__corr_n": result.n,
    }
    return enriched


# ══════════════════════════════════════════════════════════════════════════════
# SQL hint
# ══════════════════════════════════════════════════════════════════════════════

def build_correlation_sql_hint() -> str:
    return (
        "CORRELATION / SCATTER ANALYSIS HINT:\n"
        "The user wants to visualise the relationship between two numeric columns. "
        "Return a result with:\n"
        "  1. Exactly two numeric metric columns (the X and Y axes of the scatter)\n"
        "  2. Optionally one text/dimension column as a label (e.g. product name, region)\n\n"
        "Do NOT aggregate — return individual data points so each row becomes one scatter dot.\n"
        "If the question mentions specific columns (e.g. 'price vs sales volume'), "
        "select those exact columns.\n"
        "Avoid grouping by more than one dimension; too many categories make scatter unreadable.\n"
        "Add a TOP / LIMIT 500 if the table is large to avoid overloading the chart.\n"
        "Column order: label_col first (text), then x_metric, then y_metric."
    )
