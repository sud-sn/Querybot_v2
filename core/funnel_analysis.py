"""
core/funnel_analysis.py
────────────────────────
Funnel / conversion-stage analysis.

Answers questions like:
  "Show the sales pipeline funnel"
  "What is the conversion rate at each stage?"
  "Where are we losing leads?"
  "Show drop-off rates across the hiring funnel"
  "Patient pathway funnel from referral to discharge"
  "Funnel from trial to paid to renewal"

Design
──────
• Detection: regex on NL question.
• Post-processing: expects rows with a stage/step column and a count column
  ordered by stage position. Computes:
    - conversion_rate: pct from previous stage
    - drop_off:        count lost from previous stage
    - cumulative_conversion: pct from stage 0 (top of funnel)
• SQL hint: instructs LLM to return one row per stage with an ordering column.
• Chart type: "funnel" → ECharts type:'funnel'

Entry points
────────────
  detect_funnel_intent(question) → bool
  infer_funnel_cols(rows) → tuple[str, str]   (stage_col, count_col)
  compute_funnel(rows, stage_col, count_col) → list[dict]
  build_funnel_sql_hint() → str
  build_funnel_summary(rows, stage_col, count_col) → FunnelSummary
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Detection
# ══════════════════════════════════════════════════════════════════════════════

_FUNNEL_PATTERNS = [
    re.compile(r"\bfunnel\b", re.I),
    re.compile(r"\bconversion\s+(?:rate|funnel|path|pipeline|journey)\b", re.I),
    re.compile(r"\bdrop.?off\s+(?:rate|at|per|by|funnel)\b", re.I),
    re.compile(r"\b(?:stage|step|phase)\s+(?:by\s+stage|analysis|conversion|drop)\b", re.I),
    re.compile(r"\bpipeline\s+(?:stages?|funnel|conversion|analysis)\b", re.I),
    re.compile(r"\b(?:top|bottom)\s+of\s+(?:the\s+)?funnel\b", re.I),
    re.compile(r"\blead[s]?\s+(?:to\s+)?(?:close|won|deal|conversion)\b", re.I),
    re.compile(r"\bapplicant[s]?\s+(?:to\s+)?(?:offer|hire|interview)\b", re.I),
    re.compile(r"\b(?:trial|free)\s+(?:to\s+)?(?:paid|premium|converted)\b", re.I),
    re.compile(r"\b(?:visit|visitor)[s]?\s+(?:to\s+)?(?:lead|signup|purchase)\b", re.I),
    re.compile(r"\bhow\s+many\s+.{0,20}(?:make\s+it|pass|progress|advance)\s+(?:to|through)\b", re.I),
    re.compile(r"\bwhere\s+(?:are\s+we|do\s+we)\s+(?:losing|losing\s+leads|dropping)\b", re.I),
]


def detect_funnel_intent(question: str) -> bool:
    """Return True if the question asks for funnel / stage conversion analysis."""
    return any(p.search(question) for p in _FUNNEL_PATTERNS)


# ══════════════════════════════════════════════════════════════════════════════
# Column inference
# ══════════════════════════════════════════════════════════════════════════════

_STAGE_KEYWORDS = {
    "stage", "step", "phase", "status", "level", "tier", "funnel", "state",
    "pipeline_stage", "hiring_stage", "journey_stage", "deal_stage",
}
_COUNT_KEYWORDS = {
    "count", "total", "num", "number", "volume", "quantity", "candidates",
    "leads", "contacts", "users", "customers", "deals", "patients",
    "applicants", "visitors", "trials",
}


def _col_score(col: str, kws: set[str]) -> int:
    col_lower = col.lower().replace("-", "_").replace(" ", "_")
    return sum(1 for kw in kws if kw in col_lower)


def _to_float(v: Any) -> float | None:
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def infer_funnel_cols(rows: list[dict]) -> tuple[str, str]:
    """
    Infer (stage_col, count_col).
    Returns ("", "") if inference fails.
    """
    if not rows:
        return "", ""
    cols = list(rows[0].keys())
    numeric = [c for c in cols if _to_float(rows[0].get(c)) is not None]
    text    = [c for c in cols if c not in numeric]

    # Stage = best-scoring text column
    stage_col = max(text or cols, key=lambda c: _col_score(c, _STAGE_KEYWORDS), default="")
    if not stage_col or _col_score(stage_col, _STAGE_KEYWORDS) == 0:
        stage_col = text[0] if text else ""

    # Count = best-scoring numeric column
    count_col = max(numeric, key=lambda c: _col_score(c, _COUNT_KEYWORDS), default="")
    if not count_col and numeric:
        count_col = numeric[0]

    return stage_col, count_col


# ══════════════════════════════════════════════════════════════════════════════
# Funnel computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_funnel(
    rows: list[dict],
    stage_col: str,
    count_col: str,
) -> list[dict]:
    """
    Annotate funnel rows with conversion rates and drop-off counts.

    Input rows must be ordered top-of-funnel first (highest count first,
    or as returned by a stage-ordered SQL query).

    Adds per row:
      conversion_rate      — pct from previous stage (None for first)
      drop_off             — absolute count lost from previous stage
      cumulative_conversion — pct from stage 0 (top of funnel)
      funnel_pct           — each stage as % of top-of-funnel (for chart sizing)
    """
    if not rows or not stage_col or not count_col:
        return rows

    # Sort by count descending so top-of-funnel is first
    # (preserves user order if already sorted; re-sorts otherwise)
    ordered = sorted(rows, key=lambda r: _to_float(r.get(count_col)) or 0, reverse=True)
    top = _to_float(ordered[0].get(count_col)) if ordered else 0

    result = []
    prev = None
    for row in ordered:
        count = _to_float(row.get(count_col))
        if count is None:
            result.append({**row, "conversion_rate": None, "drop_off": None,
                           "cumulative_conversion": None, "funnel_pct": None})
            continue

        if prev is None:
            conv_rate = None
            drop_off  = 0
        else:
            conv_rate = round(count / prev * 100, 2) if prev > 0 else None
            drop_off  = round(prev - count, 4)

        cum_conv = round(count / top * 100, 2) if top > 0 else None
        funnel_pct = round(count / top * 100, 1) if top > 0 else 100.0

        result.append({
            **row,
            "conversion_rate": conv_rate,
            "drop_off": drop_off,
            "cumulative_conversion": cum_conv,
            "funnel_pct": funnel_pct,
        })
        prev = count

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SQL hint
# ══════════════════════════════════════════════════════════════════════════════

def build_funnel_sql_hint() -> str:
    return (
        "FUNNEL ANALYSIS HINT:\n"
        "The user wants to see conversion rates across pipeline/funnel stages. "
        "Return one row per stage with:\n\n"
        "  1. stage_name   — the name of the stage (e.g. 'Lead', 'Qualified', 'Proposal', 'Won')\n"
        "  2. stage_count  — number of records at that stage\n"
        "  3. stage_order  — (optional) integer to control sort order, 1 = top of funnel\n\n"
        "Order results from the top of the funnel (highest count) to the bottom (lowest count).\n"
        "If stages are stored as status values in a transaction table, use:\n"
        "  SELECT stage_column, COUNT(*) AS stage_count\n"
        "  FROM pipeline_table\n"
        "  GROUP BY stage_column\n"
        "  ORDER BY stage_count DESC\n\n"
        "Include ALL stages, even those with zero entries — the system computes conversion rates automatically."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FunnelSummary:
    stage_count: int
    top_of_funnel: float
    bottom_of_funnel: float
    overall_conversion_pct: float | None
    biggest_drop_stage: str
    biggest_drop_pct: float | None


def build_funnel_summary(rows: list[dict], stage_col: str, count_col: str) -> FunnelSummary:
    enriched = compute_funnel(rows, stage_col, count_col)
    if not enriched:
        return FunnelSummary(0, 0, 0, None, "", None)

    top    = _to_float(enriched[0].get(count_col)) or 0
    bottom = _to_float(enriched[-1].get(count_col)) or 0
    overall = round(bottom / top * 100, 2) if top else None

    # Biggest drop-off (largest absolute drop_off)
    drop_rows = [r for r in enriched if r.get("drop_off") is not None and r.get("drop_off", 0) > 0]
    biggest = max(drop_rows, key=lambda r: r.get("drop_off", 0), default={})
    biggest_stage = str(biggest.get(stage_col, ""))
    biggest_drop_pct = None
    if biggest:
        prev_count = top
        for r in enriched:
            if r.get(stage_col) == biggest_stage:
                break
            prev_count = _to_float(r.get(count_col)) or prev_count
        curr = _to_float(biggest.get(count_col))
        if curr is not None and prev_count:
            biggest_drop_pct = round((prev_count - curr) / prev_count * 100, 1)

    return FunnelSummary(
        stage_count=len(enriched),
        top_of_funnel=top,
        bottom_of_funnel=bottom,
        overall_conversion_pct=overall,
        biggest_drop_stage=biggest_stage,
        biggest_drop_pct=biggest_drop_pct,
    )
