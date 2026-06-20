"""
core/budget_vs_actual.py
─────────────────────────
Budget vs Actual variance analysis.

Answers questions like:
  "How does actual revenue compare to budget this month?"
  "Show variance to plan by department"
  "Which regions are over or under target?"
  "Actuals vs forecast — where are we off plan?"

Design
──────
• Detection: regex against NL question.
• Post-processing: adds variance, variance_pct, and status columns to rows
  that contain both an actual and a budget/target/plan column.
• SQL hint: inject hint to select both actual and budget columns and compute
  variance inline so the LLM emits the right SELECT expression.
• Waterfall trigger: when enabled in chart dispatch, the variance breakdown
  drives a waterfall chart showing over/under per category.

Entry points
────────────
  detect_bva_intent(question) → bool
  infer_bva_cols(rows) → tuple[str, str]   (actual_col, budget_col)
  compute_bva(rows, actual_col, budget_col) → list[dict]
  build_bva_sql_hint(actual_hint, budget_hint) → str
  build_bva_summary(rows, actual_col, budget_col) → BvASummary
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# ══════════════════════════════════════════════════════════════════════════════
# Detection
# ══════════════════════════════════════════════════════════════════════════════

_BVA_BUDGET_WORDS = r"(?:budget(?:ed)?|target(?:ed)?|plan(?:ned)?|forecast(?:ed)?|baseline)"
_BVA_ACTUAL_WORDS = r"(?:actual[s]?|real|achieved|realized|realised|delivered)"

_BVA_PATTERNS = [
    re.compile(rf"\bvs\.?\s*{_BVA_BUDGET_WORDS}\b", re.I),
    re.compile(rf"\b{_BVA_BUDGET_WORDS}\s+vs\.?\s*{_BVA_ACTUAL_WORDS}\b", re.I),
    re.compile(rf"\b{_BVA_ACTUAL_WORDS}\s+vs\.?\s*{_BVA_BUDGET_WORDS}\b", re.I),
    re.compile(r"\b(?:variance|variances)\s+(?:to|from|against|vs\.?)\s*(?:budget(?:ed)?|target|plan(?:ned)?|forecast)\b", re.I),
    re.compile(r"\b(?:over|under)\s+(?:budget|target|plan|forecast)\b", re.I),
    re.compile(r"\b(?:budget(?:ed)?|target|plan(?:ned)?|forecast)\s+(?:variance|deviation|gap|miss|hit|achievement)\b", re.I),
    re.compile(r"\b(?:on|off)\s+(?:budget|target|plan|track)\b", re.I),
    re.compile(r"\b(?:attainment|achievement)\s+(?:vs|against|rate|percentage|%)\b", re.I),
    re.compile(r"\bplan(?:ned)?\s+vs\.?\s*(?:actual|achieved)\b", re.I),
    re.compile(r"\b(?:shortfall|overrun|underspend|overspend)\b", re.I),
]


def detect_bva_intent(question: str) -> bool:
    """Return True if the question asks for budget vs actual / target variance analysis."""
    return any(p.search(question) for p in _BVA_PATTERNS)


# ══════════════════════════════════════════════════════════════════════════════
# Column inference
# ══════════════════════════════════════════════════════════════════════════════

_ACTUAL_KEYWORDS = {
    "actual", "actuals", "achieved", "real", "ytd", "current",
    "realized", "realised", "delivered", "result",
}
_BUDGET_KEYWORDS = {
    "budget", "budgeted", "target", "plan", "planned", "forecast",
    "forecasted", "baseline", "expected",
}


def _token_set(col_name: str) -> set[str]:
    raw = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", col_name or "")
    return {t for t in re.split(r"[\s_\-]+", raw.lower()) if t}


def infer_bva_cols(rows: list[dict]) -> tuple[str, str]:
    """
    Infer (actual_col, budget_col) from column names.
    Returns ("", "") if not found.
    """
    if not rows:
        return "", ""
    cols = list(rows[0].keys())
    actuals = [c for c in cols if _token_set(c) & _ACTUAL_KEYWORDS]
    budgets = [c for c in cols if _token_set(c) & _BUDGET_KEYWORDS]
    if actuals and budgets:
        return actuals[0], budgets[0]
    # Fallback: first two numeric-ish columns
    numeric = [c for c in cols if _to_float(rows[0].get(c)) is not None]
    if len(numeric) >= 2:
        return numeric[0], numeric[1]
    return "", ""


# ══════════════════════════════════════════════════════════════════════════════
# Post-processing
# ══════════════════════════════════════════════════════════════════════════════

def _to_float(v: Any) -> float | None:
    try:
        return float(str(v).replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return None


_ON_TARGET_PCT = 2.0   # within ±2% is "on target"


def compute_bva(
    rows: list[dict],
    actual_col: str,
    budget_col: str,
) -> list[dict]:
    """
    Augment each row with:
      variance        = actual − budget
      variance_pct    = (actual − budget) / |budget| × 100
      bva_status      = "over" | "under" | "on_target" | null
    """
    result = []
    for row in rows:
        actual = _to_float(row.get(actual_col))
        budget = _to_float(row.get(budget_col))
        if actual is None or budget is None:
            result.append({**row, "variance": None, "variance_pct": None, "bva_status": None})
            continue
        var = round(actual - budget, 4)
        if budget != 0:
            var_pct = round((actual - budget) / abs(budget) * 100, 2)
        else:
            var_pct = None
        if var_pct is None:
            status = None
        elif abs(var_pct) <= _ON_TARGET_PCT:
            status = "on_target"
        elif var > 0:
            status = "over"
        else:
            status = "under"
        result.append({**row, "variance": var, "variance_pct": var_pct, "bva_status": status})
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SQL hint
# ══════════════════════════════════════════════════════════════════════════════

def build_bva_sql_hint(actual_hint: str = "actual_col", budget_hint: str = "budget_col") -> str:
    return (
        "BUDGET VS ACTUAL HINT:\n"
        "The user wants to compare actuals against a budget/target. "
        "Select both columns and compute variance inline:\n\n"
        f"  {actual_hint} AS actual,\n"
        f"  {budget_hint} AS budget,\n"
        f"  ({actual_hint} - {budget_hint}) AS variance,\n"
        f"  ROUND(\n"
        f"    ({actual_hint} - {budget_hint}) * 100.0\n"
        f"    / NULLIF(ABS({budget_hint}), 0),\n"
        f"    2\n"
        f"  ) AS variance_pct\n\n"
        "Replace 'actual_col' and 'budget_col' with the real column names.\n"
        "Order results so the largest (absolute) variances appear first.\n"
        "Include the dimension column (department, region, product) as the first column."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Summary for LLM brief
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BvASummary:
    total_actual: float
    total_budget: float
    total_variance: float
    total_variance_pct: float | None
    over_count: int
    under_count: int
    on_target_count: int
    largest_miss: dict        # {"label": ..., "variance": ..., "variance_pct": ...}
    largest_over: dict        # {"label": ..., "variance": ..., "variance_pct": ...}
    row_count: int


def build_bva_summary(
    rows: list[dict],
    actual_col: str,
    budget_col: str,
    label_col: str = "",
) -> BvASummary:
    enriched = compute_bva(rows, actual_col, budget_col)
    total_a = sum(_to_float(r.get(actual_col)) or 0 for r in rows)
    total_b = sum(_to_float(r.get(budget_col)) or 0 for r in rows)
    total_v = round(total_a - total_b, 4)
    total_v_pct = round(total_v / abs(total_b) * 100, 2) if total_b else None

    over   = [r for r in enriched if r.get("bva_status") == "over"]
    under  = [r for r in enriched if r.get("bva_status") == "under"]
    on_tgt = [r for r in enriched if r.get("bva_status") == "on_target"]

    lbl = label_col or (next((k for k in (rows[0].keys() if rows else [])
                               if k not in (actual_col, budget_col)), "") if rows else "")

    def _lbl(r):
        return str(r.get(lbl, ""))[:40] if lbl else ""

    worst = min(enriched, key=lambda r: _to_float(r.get("variance")) or 0, default={})
    best  = max(enriched, key=lambda r: _to_float(r.get("variance")) or 0, default={})

    return BvASummary(
        total_actual=round(total_a, 2),
        total_budget=round(total_b, 2),
        total_variance=total_v,
        total_variance_pct=total_v_pct,
        over_count=len(over),
        under_count=len(under),
        on_target_count=len(on_tgt),
        largest_miss={"label": _lbl(worst), "variance": worst.get("variance"), "variance_pct": worst.get("variance_pct")},
        largest_over={"label": _lbl(best),  "variance": best.get("variance"),  "variance_pct": best.get("variance_pct")},
        row_count=len(enriched),
    )
