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

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from core.i18n import plural as _plural

log = logging.getLogger("querybot.contribution")


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

# ``distribution`` has two materially different analytical meanings:
#
# * composition â€” "distribution of revenue by state" (one aggregate per
#   category, suitable for pie/bar), and
# * statistical distribution â€” "distribution of transaction amounts"
#   (row-level values binned into a histogram).
#
# Keep the composition vocabulary intentionally limited to measures that are
# normally additive.  Schema/metric governance still decides the exact formula;
# this detector only prevents the histogram route from changing the requested
# grain.  Explicit statistical wording always wins.
_COMPOSITION_MEASURE = (
    r"revenue|sales|cost|spend|profit|income|expense|headcount|count|"
    r"quantity|qty|volume|units?|orders?|claims?|payments?|balance|"
    r"inventory(?:\s+value)?|(?:gross|net|billed|paid|outstanding)\s+amount"
)
_STATISTICAL_DISTRIBUTION_RE = re.compile(
    r"\b(histogram|frequency|bins?|buckets?|ranges?|quartiles?|median|"
    r"percentiles?|outliers?|box\s*(?:plot|chart))\b",
    re.I,
)
_COMPOSITION_DISTRIBUTION_PATTERNS = [
    # A category grain is the decisive signal here.  Restricting the phrase
    # immediately after "distribution of" to a small metric vocabulary made
    # valid governed measures such as "net revenue", "booked revenue", and
    # client-specific metric names fall through to the histogram route.  An
    # explicit statistical term is already handled above and always wins.
    re.compile(
        r"\bdistribution\s+of\s+.{1,100}?"
        r"\b(?:by|per|across|grouped\s+by|split\s+by)\b",
        re.I,
    ),
    re.compile(
        rf"\b(?:{_COMPOSITION_MEASURE})(?:\s+\w+){{0,4}}\s+distribution\b"
        r".{0,50}\b(?:by|per|across|grouped\s+by|split\s+by)\b",
        re.I,
    ),
]


def detect_composition_intent(question: str) -> bool:
    """Return True for category share/mix requests, not histograms.

    This is schema-independent and deliberately conservative.  An explicit
    histogram/bin/range/box-plot phrase keeps the request on the statistical
    distribution route even when an additive measure word is present.
    """
    if _STATISTICAL_DISTRIBUTION_RE.search(question or ""):
        return False
    return any(
        pattern.search(question or "")
        for pattern in _COMPOSITION_DISTRIBUTION_PATTERNS
    )


def detect_contribution_intent(question: str) -> bool:
    """Return True if the question asks for contribution / share / mix analysis."""
    return detect_composition_intent(question) or any(
        p.search(question) for p in _CONTRIBUTION_PATTERNS
    )


def infer_numeric_col(rows: list[dict]) -> str:
    """Return the best numeric business measure, avoiding numeric IDs."""
    if not rows:
        return ""
    candidates: list[str] = []
    for col in rows[0].keys():
        hits = sum(1 for r in rows[:10] if _to_float(r.get(col)) is not None)
        if hits >= min(len(rows), 10) * 0.75:
            candidates.append(col)
    if not candidates:
        return ""

    metric_words = re.compile(
        r"(?:revenue|sales|amount|amt|cost|spend|profit|income|expense|"
        r"headcount|count|cnt|quantity|qty|volume|units|value|balance|total)",
        re.I,
    )
    identifier = re.compile(
        r"(?:^|_)(?:id|key|code|num|no|nbr|seq|rank)$",
        re.I,
    )

    def score(column: str) -> tuple[int, int]:
        value = 0
        if metric_words.search(column):
            value += 20
        if identifier.search(column):
            value -= 30
        # Stable result-column order remains the final tie-breaker.
        return value, -candidates.index(column)

    return max(candidates, key=score)


# ══════════════════════════════════════════════════════════════════════════════
# Post-processing: compute share from in-memory rows
# ══════════════════════════════════════════════════════════════════════════════

def infer_label_col(rows: list[dict], value_col: str = "") -> str:
    """The column whose share is being asked about.

    A composition result grouped by two things -- a warehouse and a month --
    has two non-numeric columns, and only one of them is the thing whose share
    the reader wants. The period is not: "what share does each warehouse
    account for" is a question about warehouses, and the months are the axis
    being collapsed.

    Returns "" when it cannot tell, which leaves the caller's behaviour
    unchanged rather than guessing a dimension.
    """
    if not rows:
        return ""
    from core.response_builder import _looks_temporal

    candidates = []
    for col in rows[0].keys():
        if col == value_col or str(col).startswith("_"):
            continue
        sample = [r.get(col) for r in rows[:20]]
        numeric = sum(1 for v in sample if _to_float(v) is not None)
        if numeric >= max(1, len(sample)) * 0.75:
            continue                     # a measure, or a numeric key
        if _looks_temporal([str(v) for v in sample if v is not None]):
            continue                     # the axis being collapsed, not the label
        candidates.append(col)
    # Exactly one non-numeric, non-temporal column is a label we can trust.
    # Two of them is a genuinely ambiguous request, and guessing which one the
    # reader meant is worse than leaving the rows alone.
    return candidates[0] if len(candidates) == 1 else ""


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
    # A result grouped by two things carries a row per label PER PERIOD, and
    # dividing each of those rows by the grand total understates every label by
    # the period count while its real share appears nowhere. Asked "what share
    # of revenue does each warehouse account for", a three-month grid reported
    # Halifax at 17.65% three times against a true 52.94%.
    #
    # Collapse to one row per label first, through the shared rule -- which
    # returns None when the measure may not be summed across the axis being
    # removed (a margin percentage, a period-end balance). There is no honest
    # share in that case, so no share is offered: the same answer this function
    # already gives for a zero total.
    label_col = label_col or infer_label_col(rows, value_col)
    if label_col and rows:
        from core.analysis_contract import collapse_rows_by_label

        collapsed = collapse_rows_by_label(rows, label_col, value_col)
        if collapsed is None:
            log.info(
                "Contribution shares withheld for %s by %s — the measure may "
                "not be summed across the second dimension",
                value_col, label_col,
            )
            return [{**row, "contribution_pct": None} for row in rows]
        if len(collapsed) != len(rows):
            log.info(
                "Contribution collapsed %d rows to %d %s before computing "
                "shares — the result carried a second dimension",
                len(rows), len(collapsed), label_col,
            )
            rows = [{label_col: label, value_col: value}
                    for label, value in collapsed]

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
            # Was an English literal with a hard-coded plural, in the label
            # of a slice the reader sees beside their own categories.
            other_lbl: _plural("ui.contribution.other", len(others)),
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
