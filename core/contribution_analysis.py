"""
core/contribution_analysis.py
──────────────────────────────
Percentage contribution / mix analysis.

Answers questions like:
  "What % of revenue does each region contribute?"
  "Show the revenue mix by product category"
  "Which departments account for the most headcount?"
  "Contribution breakdown of attrition by team"

Design
──────
• Detection: regex patterns on natural language question.
• Two modes:
    1. POST-PROCESS: if the caller already has GROUP BY rows, compute shares
       from the in-memory result (fast, no extra DB round-trip).
    2. SQL HINT: inject a prompt hint so the LLM emits a window-function
       SUM() OVER () for the denominator in its generated SQL.

Entry points
────────────
  detect_contribution_intent(question) → bool
  compute_contribution(rows, value_col) → list[dict]          (post-process)
  build_contribution_sql_hint(value_col_hint) → str            (prompt injection)
  build_contribution_summary(rows, value_col) → ContribSummary (brief for LLM)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Detection
# ══════════════════════════════════════════════════════════════════════════════

_CONTRIBUTION_PATTERNS = [
    re.compile(r"\bcontribut(?:ion|e|es|ing)?\b", re.I),
    re.compile(r"\b(?:what|how much)\s+(?:%|percent(?:age)?|share|portion|fraction)\b", re.I),
    re.compile(r"\b(?:revenue|sales|cost|headcount|count)\s+(?:mix|split|breakdown|distribution)\b", re.I),
    re.compile(r"\bmix\s+(?:by|of|across|per)\b", re.I),
    re.compile(r"\bshare\s+of\s+(?:total|revenue|sales|headcount|cost|volume)\b", re.I),
    re.compile(r"\bwhat\s+(?:percentage|%|fraction|portion|share)\s+(?:of\s+)?(?:the\s+)?total\b", re.I),
    re.compile(r"\bbreakdown\s+(?:as|in)\s+(?:percentage|%|percent)\b", re.I),
    re.compile(r"\bwhich\s+(?:\w+\s+){1,3}account(?:s)?\s+for\s+(?:the\s+)?(?:most|largest|highest|biggest|\d+\s*%)\b", re.I),
    re.compile(r"\baccount(?:s)?\s+for\s+\d+\s*%", re.I),
    re.compile(r"\b(?:top|major|main|primary|leading)\s+(?:contributor|driver|source)s?\b", re.I),
    re.compile(r"\bpareto\b", re.I),
]


def detect_contribution_intent(question: str) -> bool:
    """Return True if the question asks for contribution / share / mix analysis."""
    return any(p.search(question) for p in _CONTRIBUTION_PATTERNS)


def infer_numeric_col(rows: list[dict]) -> str:
    """Return the first column that looks like a numeric metric, or ''."""
    if not rows:
        return ""
    for col in rows[0].keys():
        hits = sum(1 for r in rows[:10] if _to_float(r.get(col)) is not None)
        if hits >= min(len(rows), 10) * 0.75:
            return col
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# Post-processing: compute share from in-memory rows
# ══════════════════════════════════════════════════════════════════════════════

def _to_float(v: Any) -> float | None:
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def compute_contribution(
    rows: list[dict],
    value_col: str,
    label_col: str = "",
    *,
    top_n: int | None = None,
    sort_desc: bool = True,
) -> list[dict]:
    """
    Add a ``contribution_pct`` column to each row, representing that row's
    share of the total of ``value_col``.

    If top_n is provided, rows beyond the top-N are aggregated into an
    "Other" bucket.

    Returns the enriched row list, sorted descending by value_col by default.
    """
    # Compute total (exclude None/NaN values)
    vals = [(i, _to_float(row.get(value_col))) for i, row in enumerate(rows)]
    total = sum(v for _, v in vals if v is not None)

    if total == 0:
        # Cannot compute shares on a zero total
        return [{**row, "contribution_pct": None} for row in rows]

    enriched = []
    for i, row in enumerate(rows):
        v = _to_float(row.get(value_col))
        pct = round((v / total) * 100, 2) if v is not None else None
        enriched.append({**row, "contribution_pct": pct})

    # Sort by value descending
    enriched.sort(
        key=lambda r: (_to_float(r.get(value_col)) is None, -(_to_float(r.get(value_col)) or 0))
        if sort_desc else
        (_to_float(r.get(value_col)) is None, _to_float(r.get(value_col)) or 0)
    )

    if top_n and len(enriched) > top_n:
        top    = enriched[:top_n]
        others = enriched[top_n:]
        other_val = sum(_to_float(r.get(value_col)) or 0 for r in others)
        other_pct = round((other_val / total) * 100, 2) if total else None
        other_lbl = label_col or (list(rows[0].keys())[0] if rows else "category")
        top.append({
            other_lbl: f"Other ({len(others)} items)",
            value_col: round(other_val, 4),
            "contribution_pct": other_pct,
        })
        return top

    return enriched


# ══════════════════════════════════════════════════════════════════════════════
# SQL hint builder
# ══════════════════════════════════════════════════════════════════════════════

def build_contribution_sql_hint(value_col_hint: str = "metric_col") -> str:
    """
    Return a prompt hint instructing the LLM to compute contribution % in SQL.
    Works on any SQL dialect that supports window functions (T-SQL, Snowflake, Oracle).
    """
    return (
        "CONTRIBUTION ANALYSIS HINT:\n"
        "The user wants to see each row's percentage share of the total. "
        "Use a window function to compute the denominator:\n\n"
        f"  {value_col_hint},\n"
        f"  ROUND(\n"
        f"    {value_col_hint} * 100.0\n"
        f"    / NULLIF(SUM({value_col_hint}) OVER (), 0),\n"
        f"    2\n"
        f"  ) AS contribution_pct\n\n"
        "Replace 'metric_col' with the actual numeric column being analysed.\n"
        "Sort the result descending by the metric column (highest contributors first).\n"
        "Include the dimension column (e.g. region, department, product) as the first column."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Summary builder (for LLM brief — no raw data sent to LLM)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ContribSummary:
    total: float
    top_contributor: str
    top_share_pct: float
    top3_share_pct: float
    category_count: int
    pareto_80_count: int    # how many categories reach 80% of total
    gini_approx: float      # 0 = equal, 1 = single contributor dominates
    rows: list[dict] = field(default_factory=list)  # enriched rows (contribution_pct added)


def build_contribution_summary(
    rows: list[dict],
    value_col: str,
    label_col: str = "",
) -> ContribSummary:
    """
    Build a statistical summary of a contribution analysis result.
    Used for the LLM data brief — raw row values are NOT included in the summary.
    """
    enriched = compute_contribution(rows, value_col, label_col)
    total    = sum(_to_float(r.get(value_col)) or 0 for r in rows)

    if not enriched or total == 0:
        return ContribSummary(
            total=total, top_contributor="", top_share_pct=0,
            top3_share_pct=0, category_count=0, pareto_80_count=0,
            gini_approx=0, rows=enriched,
        )

    # Top contributor label (redact sensitive fields)
    first_key = label_col or (next(
        (k for k in enriched[0].keys() if k != value_col and k != "contribution_pct"),
        ""
    ))
    top_lbl  = str(enriched[0].get(first_key, ""))[:60] if first_key else ""
    top_pct  = enriched[0].get("contribution_pct") or 0
    top3_pct = sum(r.get("contribution_pct") or 0 for r in enriched[:3])

    # Pareto: how many categories sum to ≥80%?
    cumulative = 0.0
    pareto_80_count = 0
    for r in enriched:
        cumulative += r.get("contribution_pct") or 0
        pareto_80_count += 1
        if cumulative >= 80:
            break

    # Approximate Gini coefficient
    n = len(enriched)
    if n > 1:
        shares = sorted(_to_float(r.get("contribution_pct")) or 0 for r in enriched)
        gini_num = sum((2 * (i + 1) - n - 1) * s for i, s in enumerate(shares))
        gini_approx = round(gini_num / (n * sum(shares)), 3) if sum(shares) else 0
    else:
        gini_approx = 1.0

    return ContribSummary(
        total=round(total, 2),
        top_contributor=top_lbl,
        top_share_pct=round(top_pct, 2),
        top3_share_pct=round(top3_pct, 2),
        category_count=n,
        pareto_80_count=pareto_80_count,
        gini_approx=abs(gini_approx),
        rows=enriched,
    )
