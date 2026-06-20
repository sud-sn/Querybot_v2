"""
core/multi_period.py
─────────────────────
Multi-period comparison engine — side-by-side comparison of 3 or more periods.

Fills the gap in period_comparison.py which handles exactly 2 periods.

Supported patterns
──────────────────
  "Compare Q1 2024, Q1 2023, and Q1 2022"
  "Revenue for January across the last 5 years"
  "Show the last 3 years side by side"
  "Compare this quarter, last quarter, and the quarter before"
  "3-year trend by department"
  "Year over year for the last 4 years"

Design
──────
• Period extraction: a dedicated parser pulls structured period specs
  from the question — named periods, relative references, or year lists.
• SQL rewriting: reuses the same narrow LLM rewrite approach as
  period_comparison.py — one rewrite per additional period, then executed
  in parallel (asyncio.gather).
• Result merging: period results are merged into a multi-series payload
  where each period becomes a series in a grouped bar or line chart.
• Graceful degradation: if any period fails, the others still render;
  failed periods are noted in a warning list.

Entry points
────────────
  detect_multi_period_intent(question) → MultiPeriodIntent | None
  extract_period_specs(question) → list[PeriodSpec]
  merge_multi_period_results(period_rows) → MultiPeriodResult
  build_multi_period_chart_payload(result) → dict
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Data models
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PeriodSpec:
    label: str          # human-readable: "Q1 2024", "2023", "Jan 2024"
    raw_text: str       # original matched text from question


@dataclass
class MultiPeriodIntent:
    period_specs: list[PeriodSpec]
    grain: str          # "monthly" | "quarterly" | "yearly"
    compare_count: int  # number of periods to compare
    raw_match: str = ""


@dataclass
class PeriodResult:
    label: str
    rows: list[dict]
    sql: str
    success: bool
    error: str = ""


@dataclass
class MultiPeriodResult:
    periods: list[PeriodResult]
    label_col: str      # the common dimension column
    value_col: str      # the common metric column
    successful_periods: int
    warnings: list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Detection & period extraction
# ══════════════════════════════════════════════════════════════════════════════

# Named year references: "2022", "2023", "2024"
_YEAR_RE = re.compile(r"\b(20\d{2})\b")

# Quarter references: "Q1 2024", "2024 Q1", "Q1/2024"
_QUARTER_RE = re.compile(r"\bQ([1-4])[\s\-/]?(20\d{2})\b|\b(20\d{2})[\s\-/]?Q([1-4])\b", re.I)

# Month references: "Jan 2024", "January 2024"
_MONTH_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s*(20\d{2})\b",
    re.I,
)

# "last N years / quarters / months"
_LAST_N_PERIODS = re.compile(r"\blast\s+(\d+)\s+(year|quarter|month)s?\b", re.I)

# Relative quarter references: "this quarter", "last quarter", "quarter before"
_REL_QUARTER = re.compile(r"\b(this|last|previous|prior)\s+quarter\b", re.I)
_REL_YEAR    = re.compile(r"\b(this|last|previous|prior)\s+year\b", re.I)

# Multi-period signals: "compare X, Y, and Z" or "across the last N"
_MULTI_SIGNAL = re.compile(
    r"\b(compare|contrast|side.by.side|vs\.?\s|versus|across\s+(?:the\s+)?last\s+\d+|"
    r"(\d+)[- ]year|year.over.year.for.the.last)\b",
    re.I,
)


def detect_multi_period_intent(question: str) -> MultiPeriodIntent | None:
    """
    Return a MultiPeriodIntent if the question implies comparing 3+ periods.
    Returns None if it's a 2-period or single-period question.
    """
    q = question.strip()

    # "last N years/quarters/months" — explicit multi-period
    m = _LAST_N_PERIODS.search(q)
    if m:
        n    = int(m.group(1))
        unit = m.group(2).lower()
        if n >= 2:
            grain = {"year": "yearly", "quarter": "quarterly", "month": "monthly"}.get(unit, "yearly")
            specs = _generate_relative_specs(n, unit)
            return MultiPeriodIntent(
                period_specs=specs,
                grain=grain,
                compare_count=n,
                raw_match=m.group(0),
            )

    # Named period extraction
    specs = extract_period_specs(q)
    if len(specs) >= 3:
        grain = _infer_grain(specs)
        return MultiPeriodIntent(
            period_specs=specs,
            grain=grain,
            compare_count=len(specs),
            raw_match=q[:80],
        )

    # 2 named periods — only if a multi-signal word is present and the
    # existing period_comparison engine can't handle it (e.g. non-calendar dates)
    if len(specs) == 2 and _MULTI_SIGNAL.search(q):
        grain = _infer_grain(specs)
        return MultiPeriodIntent(
            period_specs=specs,
            grain=grain,
            compare_count=2,
            raw_match=q[:80],
        )

    return None


def extract_period_specs(question: str) -> list[PeriodSpec]:
    """
    Extract all named period references from a question, in order of appearance.
    Deduplicates overlapping matches (quarter takes priority over bare year).
    """
    found: list[tuple[int, int, PeriodSpec]] = []  # (start, end, spec)

    # Quarter matches — highest priority
    for m in _QUARTER_RE.finditer(question):
        if m.group(1):  # Qn YYYY
            label = f"Q{m.group(1)} {m.group(2)}"
        else:           # YYYY Qn
            label = f"Q{m.group(4)} {m.group(3)}"
        found.append((m.start(), m.end(), PeriodSpec(label=label, raw_text=m.group(0))))

    # Month + year matches
    for m in _MONTH_RE.finditer(question):
        # Check not overlapping with a quarter match
        if not any(s <= m.start() < e for s, e, _ in found):
            label = f"{m.group(1).capitalize()} {m.group(2)}"
            found.append((m.start(), m.end(), PeriodSpec(label=label, raw_text=m.group(0))))

    # Bare year matches — only where no quarter/month match covers them
    for m in _YEAR_RE.finditer(question):
        if not any(s <= m.start() < e for s, e, _ in found):
            label = m.group(1)
            found.append((m.start(), m.end(), PeriodSpec(label=label, raw_text=m.group(0))))

    # Sort by position in text and deduplicate
    found.sort(key=lambda x: x[0])
    return [spec for _, _, spec in found]


def _infer_grain(specs: list[PeriodSpec]) -> str:
    if not specs:
        return "yearly"
    first = specs[0].label
    if re.match(r"Q\d", first, re.I):
        return "quarterly"
    if re.match(r"[A-Za-z]{3,9}\s+\d{4}", first):
        return "monthly"
    return "yearly"


def _generate_relative_specs(n: int, unit: str) -> list[PeriodSpec]:
    """
    Generate PeriodSpec list for "last N years/quarters/months".
    Labels are relative strings that the SQL rewrite prompt will interpret.
    """
    from datetime import date

    today = date.today()
    specs = []

    if unit == "year":
        for i in range(n):
            year = today.year - i - 1  # last year = today.year-1
            specs.append(PeriodSpec(label=str(year), raw_text=str(year)))

    elif unit == "quarter":
        q_now = (today.month - 1) // 3 + 1
        y_now = today.year
        for i in range(n):
            # Shift backward by i quarters from "last quarter"
            total_q = (y_now * 4 + q_now - 1) - 1 - i  # -1 for "last"
            year  = total_q // 4
            qtr   = (total_q % 4) + 1
            specs.append(PeriodSpec(label=f"Q{qtr} {year}", raw_text=f"Q{qtr} {year}"))

    elif unit == "month":
        import calendar
        y, mo = today.year, today.month
        for i in range(n):
            total_m = (y * 12 + mo - 1) - 1 - i
            year  = total_m // 12
            month = (total_m % 12) + 1
            abbr  = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][month]
            specs.append(PeriodSpec(label=f"{abbr} {year}", raw_text=f"{abbr} {year}"))

    return specs[::-1]  # chronological order (oldest first)


# ══════════════════════════════════════════════════════════════════════════════
# Result merging
# ══════════════════════════════════════════════════════════════════════════════

def _to_float(v: Any) -> float | None:
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _infer_columns(rows: list[dict]) -> tuple[str, str]:
    """Infer (label_col, value_col) from the first row."""
    if not rows:
        return "", ""
    keys = list(rows[0].keys())
    value_col = ""
    label_col = keys[0]
    for k in keys:
        # First fully-numeric column is the value
        if all(_to_float(r.get(k)) is not None for r in rows[:5] if r.get(k) is not None):
            value_col = k
            break
    return label_col, value_col


def merge_multi_period_results(period_results: list[PeriodResult]) -> MultiPeriodResult:
    """
    Merge results from multiple period queries into a unified multi-series structure.

    The merged structure has one row per unique label_col value, with one
    value column per period.

    E.g. for 3 years of regional sales:
      [{"region": "EMEA", "2022": 1200, "2023": 1450, "2024": 1680}, ...]
    """
    successful = [p for p in period_results if p.success and p.rows]
    warnings   = [f"Period '{p.label}' failed: {p.error}" for p in period_results if not p.success]

    if not successful:
        return MultiPeriodResult(
            periods=period_results,
            label_col="",
            value_col="",
            successful_periods=0,
            warnings=["No periods returned data."] + warnings,
        )

    # Infer column names from first successful result
    label_col, value_col = _infer_columns(successful[0].rows)
    if not value_col:
        return MultiPeriodResult(
            periods=period_results,
            label_col=label_col,
            value_col="",
            successful_periods=len(successful),
            warnings=["Could not identify a numeric value column."] + warnings,
        )

    # Build merged lookup: label_value → {period_label: metric_value}
    merged: dict[Any, dict[str, Any]] = {}
    for period in successful:
        for row in period.rows:
            key = row.get(label_col, "")
            if key not in merged:
                merged[key] = {label_col: key}
            v = _to_float(row.get(value_col))
            merged[key][period.label] = v

    merged_rows = list(merged.values())

    return MultiPeriodResult(
        periods=period_results,
        label_col=label_col,
        value_col=value_col,
        successful_periods=len(successful),
        warnings=warnings,
    )


def build_multi_period_chart_payload(
    result: MultiPeriodResult,
    period_results: list[PeriodResult],
    title: str = "",
) -> dict:
    """
    Build a chart payload compatible with the existing portal_dashboard.js
    buildDashboardOption() format.

    Uses a grouped bar chart: x-axis = dimension labels, one series per period.
    """
    successful = [p for p in period_results if p.success and p.rows]
    if not successful:
        return {}

    label_col, value_col = _infer_columns(successful[0].rows)
    if not label_col or not value_col:
        return {}

    # Collect all unique x-axis labels (dimension values)
    all_labels: list[str] = []
    seen: set[str] = set()
    for period in successful:
        for row in period.rows:
            lbl = str(row.get(label_col, ""))
            if lbl not in seen:
                all_labels.append(lbl)
                seen.add(lbl)

    # Build one row-dict per label with a column per period
    lookup: dict[str, dict[str, Any]] = {lbl: {label_col: lbl} for lbl in all_labels}
    for period in successful:
        for row in period.rows:
            lbl = str(row.get(label_col, ""))
            lookup[lbl][period.label] = _to_float(row.get(value_col))

    merged_rows = [lookup[lbl] for lbl in all_labels]

    period_labels = [p.label for p in successful]

    return {
        "type": "chart",
        "chart_type": "bar",
        "title": title or f"Comparison: {', '.join(period_labels)}",
        "x_key": label_col,
        "y_keys": period_labels,
        "rows": merged_rows,
        "multi_period": True,
        "period_labels": period_labels,
        "warnings": result.warnings or [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# SQL rewrite prompt (one per additional period)
# ══════════════════════════════════════════════════════════════════════════════

def build_multi_period_rewrite_prompt(
    original_sql: str,
    target_period: PeriodSpec,
    current_period_label: str = "",
    date_col_hint: str = "",
) -> tuple[str, str]:
    """
    Build (system, user) prompt to rewrite base SQL for a specific named period.
    Mirrors the approach in period_comparison.py for consistency.
    """
    hint = f"\nDate column hint: {date_col_hint}" if date_col_hint else ""
    system = (
        "You are a SQL date-range modifier. "
        "Your ONLY task is to rewrite the date filter to target the specified period. "
        "RULES:\n"
        "1. Change ONLY the date/period filter — nothing else.\n"
        "2. Return ONLY the SQL — no explanation, no markdown fences.\n"
        "3. If you cannot rewrite safely, return: CANNOT_REWRITE\n"
        "4. Preserve all aliases and table qualifiers exactly."
    )
    current_note = (
        f"The original SQL currently targets: {current_period_label}\n" if current_period_label else ""
    )
    user = (
        f"Original SQL:\n{original_sql}\n\n"
        f"{current_note}"
        f"Rewrite the date filter to target period: {target_period.label}"
        f"{hint}\n\n"
        f"Return only the rewritten SQL."
    )
    return system, user
