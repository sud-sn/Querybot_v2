"""
core/whatif.py
───────────────
What-if / scenario analysis.

Answers questions like:
  "What if revenue grows by 10%?"
  "Show me the impact of a 5% price increase"
  "If we reduce cost by 15%, what is the margin?"
  "What would profit be if headcount drops by 20%?"
  "Scenario: revenue +$500K — show adjusted metrics"

Design
──────
• Detection: regex on NL question for what-if / scenario language.
• `parse_whatif_params()`: extracts (target_col_hint, delta_pct, delta_abs)
  from the question using regex.
• `compute_whatif()`: applies the adjustment to each row and adds scenario
  columns (`scenario_<col>`, `scenario_delta`, `scenario_delta_pct`).
• Also computes a sensitivity table if the question asks "by X%".
• Output can be rendered as a grouped bar chart (original vs scenario).

Entry points
────────────
  detect_whatif_intent(question) → bool
  parse_whatif_params(question) → WhatIfParams
  compute_whatif(rows, params) → list[dict]
  build_whatif_sql_hint() → str
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Detection
# ══════════════════════════════════════════════════════════════════════════════

_WHATIF_PATTERNS = [
    re.compile(r"\bwhat\s+if\b", re.I),
    re.compile(r"\bwhat\s+would\b", re.I),
    re.compile(r"\bif\s+(?:we|i|revenue|sales|cost|price|headcount|margin)\s+.{0,30}(?:increase[sd]?|decrease[sd]?|grow[sn]?|drops?|reduce[sd]?|raise[sd]?|cuts?|change[sd]?)\b", re.I),
    re.compile(r"\bscenario\s*[:\-]", re.I),  # "Scenario: revenue +$500K"
    re.compile(r"\bscenario\s+(?:analysis|planning|model|where|if)\b", re.I),
    re.compile(r"\b(?:sensitivity|impact)\s+(?:analysis|of|if)\b", re.I),
    re.compile(r"\bif\s+.{0,25}(?:\+|\-|up|down|grows?|drops?|increases?|decreases?)\s+(?:by\s+)?\d+\s*%\b", re.I),
    re.compile(r"\b(?:simulate|model)\s+(?:the\s+)?(?:impact|effect|result)\b", re.I),
    re.compile(r"\bshow\s+(?:me\s+)?(?:the\s+)?(?:impact|effect)\s+of\s+(?:a\s+)?(?:\d+\s*%|\$\d+)\b", re.I),
    re.compile(r"\bif\s+(?:price|revenue|cost|spend|margin|headcount|churn)\s+(?:is|was|were|goes|went)\b", re.I),
    re.compile(r"\b(?:optimistic|pessimistic|base)\s+(?:case|scenario)\b", re.I),
]


def detect_whatif_intent(question: str) -> bool:
    """Return True if the question describes a what-if / scenario adjustment."""
    return any(p.search(question) for p in _WHATIF_PATTERNS)


# ══════════════════════════════════════════════════════════════════════════════
# Parameter extraction
# ══════════════════════════════════════════════════════════════════════════════

_PCT_CHANGE_PATTERN = re.compile(
    r"(?P<dir>increase[sd]?|decrease[sd]?|grow[sn]?|drops?|rise[sd]?|reduce[sd]?|"
    r"up|down|cut[s]?|raise[sd]?|\+|\-)\s*(?:by\s+)?(?P<val>\d+(?:\.\d+)?)\s*%",
    re.I,
)
_ABS_CHANGE_PATTERN = re.compile(
    r"(?P<dir>increase[sd]?|decrease[sd]?|grow[sn]?|drops?|reduce[sd]?|up|down|\+|\-)"
    r"\s*(?:by\s+)?\$?\s*(?P<val>[\d,]+(?:\.\d+)?)\s*(?:K|M|B)?",
    re.I,
)
_COL_HINT_WORDS = {
    "revenue", "sales", "cost", "costs", "price", "spend", "margin", "profit",
    "headcount", "churn", "mrr", "arr", "volume", "orders", "units", "rate",
}


@dataclass
class WhatIfParams:
    delta_pct:  float | None = None   # e.g. +10.0 or -5.0
    delta_abs:  float | None = None   # e.g. +500_000
    col_hint:   str = ""              # column keyword from question
    scenario_label: str = "Scenario"


def _parse_multiplier(direction: str, value: float) -> float:
    # Match negative direction words without strict \b at end to handle inflections
    # (e.g. "reduces", "decreases", "drops" all start with the root)
    neg = re.search(r"(?i)(?:decreas|reduc|drop|down|cut|\-)", direction)
    return -value if neg else value


def parse_whatif_params(question: str) -> WhatIfParams:
    """Extract delta_pct, delta_abs, col_hint from the question."""
    params = WhatIfParams()

    # Percentage change
    m = _PCT_CHANGE_PATTERN.search(question)
    if m:
        val = float(m.group("val"))
        params.delta_pct = _parse_multiplier(m.group("dir"), val)

    # Absolute change (fallback if no %)
    if params.delta_pct is None:
        m = _ABS_CHANGE_PATTERN.search(question)
        if m:
            raw = m.group("val").replace(",", "")
            val = float(raw)
            # Handle K/M/B suffixes
            suffix_m = re.search(r"(K|M|B)\b", question[m.end():m.end()+2], re.I)
            if suffix_m:
                mult = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
                val *= mult.get(suffix_m.group(1).upper(), 1)
            params.delta_abs = _parse_multiplier(m.group("dir"), val)

    # Column hint
    q_lower = question.lower()
    matched = [w for w in _COL_HINT_WORDS if w in q_lower]
    if matched:
        params.col_hint = matched[0]

    # Scenario label
    if params.delta_pct is not None:
        sign = "+" if params.delta_pct >= 0 else ""
        params.scenario_label = f"{sign}{params.delta_pct:g}% scenario"
    elif params.delta_abs is not None:
        sign = "+" if params.delta_abs >= 0 else ""
        params.scenario_label = f"{sign}{params.delta_abs:,.0f} scenario"
    else:
        params.scenario_label = "What-If Scenario"

    return params


# ══════════════════════════════════════════════════════════════════════════════
# Scenario computation
# ══════════════════════════════════════════════════════════════════════════════

def _to_float(v: Any) -> float | None:
    try:
        return float(str(v).replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return None


def _infer_target_col(rows: list[dict], col_hint: str) -> str:
    """Find the numeric column whose name best matches col_hint."""
    if not rows:
        return ""
    cols = list(rows[0].keys())
    numeric = [c for c in cols if _to_float(rows[0].get(c)) is not None]
    if not numeric:
        return ""
    if col_hint:
        for c in numeric:
            if col_hint.lower() in c.lower():
                return c
    return numeric[0]


def compute_whatif(
    rows: list[dict],
    params: WhatIfParams,
    target_col: str = "",
) -> list[dict]:
    """
    Apply the what-if adjustment to each row.

    Adds to each row:
      scenario_<col>      — adjusted value
      scenario_delta      — absolute change
      scenario_delta_pct  — percentage change
      __scenario_label    — human-readable scenario description
      __base_col          — original column name (for chart axis labelling)

    target_col is inferred from params.col_hint if not supplied.
    """
    if not rows:
        return rows

    col = target_col or _infer_target_col(rows, params.col_hint)
    if not col:
        return rows

    result = []
    for row in rows:
        base = _to_float(row.get(col))
        if base is None:
            result.append({
                **row,
                f"scenario_{col}": None,
                "scenario_delta": None,
                "scenario_delta_pct": None,
                "__scenario_label": params.scenario_label,
                "__base_col": col,
            })
            continue

        if params.delta_pct is not None:
            adjusted = base * (1 + params.delta_pct / 100)
        elif params.delta_abs is not None:
            adjusted = base + params.delta_abs
        else:
            adjusted = base

        adjusted = round(adjusted, 4)
        delta    = round(adjusted - base, 4)
        delta_pct = round(delta / abs(base) * 100, 2) if base != 0 else None

        result.append({
            **row,
            f"scenario_{col}": adjusted,
            "scenario_delta": delta,
            "scenario_delta_pct": delta_pct,
            "__scenario_label": params.scenario_label,
            "__base_col": col,
        })

    return result


def compute_sensitivity_table(
    rows: list[dict],
    col: str,
    pct_steps: list[float] | None = None,
) -> list[dict]:
    """
    Return a sensitivity table showing the effect of different % adjustments.

    Default steps: -20%, -10%, -5%, 0%, +5%, +10%, +20%
    Returns one row per step, one column per original row (by first text column as label).
    """
    steps = pct_steps or [-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0]
    text_cols = [c for c in (rows[0].keys() if rows else []) if _to_float(rows[0].get(c)) is None]
    label_col = text_cols[0] if text_cols else None

    result = []
    for pct in steps:
        row_out: dict[str, Any] = {"pct_change": pct}
        total = 0.0
        for r in rows:
            lbl = str(r.get(label_col, "")) if label_col else ""
            base = _to_float(r.get(col))
            adj  = round(base * (1 + pct / 100), 4) if base is not None else None
            if lbl:
                row_out[lbl] = adj
            if adj is not None:
                total += adj
        row_out["TOTAL"] = round(total, 4)
        result.append(row_out)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SQL hint
# ══════════════════════════════════════════════════════════════════════════════

def build_whatif_sql_hint(params: WhatIfParams | None = None) -> str:
    if params and params.col_hint:
        col_note = f"Focus on columns related to '{params.col_hint}'."
    else:
        col_note = "Return the key numeric metrics (revenue, cost, margin, etc.)."

    return (
        "WHAT-IF / SCENARIO ANALYSIS HINT:\n"
        "The user wants to model the impact of a change on the current data. "
        "Return the ACTUAL current values — the system will compute the scenario automatically.\n\n"
        f"{col_note}\n\n"
        "Return:\n"
        "  1. A dimension column (department, product, region, etc.) as the first column\n"
        "  2. The numeric metric column(s) to be adjusted\n\n"
        "Do NOT apply the percentage/amount adjustment in SQL — return the real current values.\n"
        "Do NOT project or forecast — return current snapshot data only."
    )
