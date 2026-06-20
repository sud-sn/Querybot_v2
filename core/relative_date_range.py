"""
core/relative_date_range.py
───────────────────────────
Custom (relative) date range comparison engine.

Handles questions that reference rolling windows rather than calendar-aligned
periods — the gap that period_comparison.py cannot cover.

Supported expressions
─────────────────────
  "last 30 days vs the prior 30 days"
  "last 7 days"
  "this week vs last week"
  "last 90 days vs the 90 days before that"
  "this month vs last month"
  "last 3 months"
  "year to date vs prior year to date"
  "last quarter vs the quarter before"
  "past 2 weeks"

Design
──────
• Fully deterministic date arithmetic — no LLM needed to compute windows.
• The LLM is used only to (a) rewrite base SQL to filter each window, and
  (b) generate the comparison narrative (same pattern as period_comparison.py).
• All date arithmetic is in UTC; timezone-naive datetimes are used for simplicity.

Entry points
────────────
  detect_relative_date_question(question) → RelativeDateIntent | None
  compute_relative_windows(intent, as_of)  → (current_window, prior_window)
  build_relative_date_rewrite_prompt(...)  → (system, user)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta, datetime
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# Data model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DateWindow:
    start: date
    end: date
    label: str

    def as_iso(self) -> tuple[str, str]:
        return self.start.isoformat(), self.end.isoformat()


@dataclass
class RelativeDateIntent:
    """Parsed relative-date intent from a natural language question."""
    unit: str          # "day", "week", "month", "quarter", "year"
    n: int             # number of units (e.g. 30 for "last 30 days")
    compare: bool      # True if question asks for comparison vs prior window
    ytd: bool = False  # year-to-date mode
    raw_match: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# Detection patterns
# ══════════════════════════════════════════════════════════════════════════════

# "last N days/weeks/months/quarters/years"
_LAST_N = re.compile(
    r"\b(?:last|past|previous|prior|trailing)\s+(\d+)\s+"
    r"(day|week|month|quarter|year)s?\b",
    re.I,
)

# "this week / this month / this quarter / this year"
_THIS_PERIOD = re.compile(
    r"\bthis\s+(week|month|quarter|year)\b", re.I
)

# "last week / last month / last quarter / last year"
_LAST_PERIOD = re.compile(
    r"\blast\s+(week|month|quarter|year)\b", re.I
)

# YTD / MTD / QTD
_TD_PATTERNS = re.compile(
    r"\b(ytd|year.?to.?date|mtd|month.?to.?date|qtd|quarter.?to.?date)\b", re.I
)

# Comparison markers
_COMPARE_MARKERS = re.compile(
    r"\b(?:vs\.?|versus|compare[d]?(?:\s+to)?|against|prior|before\s+that|same\s+period)\b", re.I
)


def detect_relative_date_question(question: str) -> RelativeDateIntent | None:
    """
    Return a RelativeDateIntent if the question references a relative/rolling
    date window.  Returns None for questions that are already covered by the
    calendar-period engine (YYYY-MM, Q1 2024, etc.).
    """
    q = question.strip()
    has_compare = bool(_COMPARE_MARKERS.search(q))

    # YTD / MTD / QTD
    td = _TD_PATTERNS.search(q)
    if td:
        text = td.group(1).lower().replace("-", "").replace(" ", "")
        if "ytd" in text or "year" in text:
            return RelativeDateIntent(unit="year", n=1, compare=has_compare,
                                      ytd=True, raw_match=td.group(0))
        if "mtd" in text or "month" in text:
            return RelativeDateIntent(unit="month", n=1, compare=has_compare,
                                      ytd=True, raw_match=td.group(0))
        if "qtd" in text or "quarter" in text:
            return RelativeDateIntent(unit="quarter", n=1, compare=has_compare,
                                      ytd=True, raw_match=td.group(0))

    # "last N days/weeks/months"
    m = _LAST_N.search(q)
    if m:
        n    = int(m.group(1))
        unit = m.group(2).lower().rstrip("s")
        return RelativeDateIntent(unit=unit, n=n, compare=has_compare,
                                   raw_match=m.group(0))

    # "last week / last month / last quarter / last year"
    m = _LAST_PERIOD.search(q)
    if m:
        unit = m.group(1).lower()
        unit_map = {"week": ("day", 7), "month": ("month", 1),
                    "quarter": ("quarter", 1), "year": ("year", 1)}
        unit_key, n = unit_map.get(unit, ("day", 7))
        return RelativeDateIntent(unit=unit_key, n=n, compare=has_compare,
                                   raw_match=m.group(0))

    # "this week / this month" — only when compare intent is present
    m = _THIS_PERIOD.search(q)
    if m and has_compare:
        unit = m.group(1).lower()
        unit_map = {"week": ("day", 7), "month": ("month", 1),
                    "quarter": ("quarter", 1), "year": ("year", 1)}
        unit_key, n = unit_map.get(unit, ("day", 7))
        return RelativeDateIntent(unit=unit_key, n=n, compare=True,
                                   raw_match=m.group(0))

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Date window computation (pure, deterministic)
# ══════════════════════════════════════════════════════════════════════════════

def _start_of_week(d: date) -> date:
    return d - timedelta(days=d.weekday())  # Monday


def _start_of_month(d: date) -> date:
    return d.replace(day=1)


def _start_of_quarter(d: date) -> date:
    q_start_month = ((d.month - 1) // 3) * 3 + 1
    return d.replace(month=q_start_month, day=1)


def _start_of_year(d: date) -> date:
    return d.replace(month=1, day=1)


def _add_months(d: date, months: int) -> date:
    """Add (or subtract) N months from a date, clamping to end of month."""
    year  = d.year + (d.month + months - 1) // 12
    month = (d.month + months - 1) % 12 + 1
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    return d.replace(year=year, month=month, day=min(d.day, last_day))


def _add_quarters(d: date, quarters: int) -> date:
    return _add_months(d, quarters * 3)


def compute_relative_windows(
    intent: RelativeDateIntent,
    as_of: date | None = None,
) -> tuple[DateWindow, DateWindow | None]:
    """
    Compute the current and (optionally) prior date window for the intent.

    Returns (current_window, prior_window).
    prior_window is None when intent.compare is False.
    """
    today = as_of or date.today()

    if intent.ytd:
        # YTD: Jan 1 → today, prior YTD: Jan 1 prev year → same date prev year
        if intent.unit == "year":
            curr_start = _start_of_year(today)
            curr_end   = today
            prior_start = curr_start.replace(year=curr_start.year - 1)
            prior_end   = today.replace(year=today.year - 1)
        elif intent.unit == "month":
            curr_start = _start_of_month(today)
            curr_end   = today
            prior_start = _add_months(curr_start, -1)
            prior_end   = _add_months(today, -1)
        else:  # quarter
            curr_start = _start_of_quarter(today)
            curr_end   = today
            prior_start = _add_quarters(curr_start, -1)
            prior_end   = _add_quarters(today, -1)

        current_window = DateWindow(
            start=curr_start,
            end=curr_end,
            label=f"{curr_start.isoformat()} to {curr_end.isoformat()}",
        )
        prior_window = DateWindow(
            start=prior_start,
            end=prior_end,
            label=f"{prior_start.isoformat()} to {prior_end.isoformat()}",
        ) if intent.compare else None
        return current_window, prior_window

    # Rolling N days/weeks/months/quarters/years
    if intent.unit == "day":
        span      = timedelta(days=intent.n)
        curr_end  = today - timedelta(days=1)       # yesterday (full days only)
        curr_start = curr_end - span + timedelta(days=1)
        prior_end  = curr_start - timedelta(days=1)
        prior_start = prior_end - span + timedelta(days=1)
    elif intent.unit == "week":
        span       = timedelta(weeks=intent.n)
        curr_end   = _start_of_week(today) - timedelta(days=1)
        curr_start = curr_end - span + timedelta(days=1)
        prior_end  = curr_start - timedelta(days=1)
        prior_start = prior_end - span + timedelta(days=1)
    elif intent.unit == "month":
        curr_end   = _start_of_month(today) - timedelta(days=1)
        curr_start = _add_months(curr_end.replace(day=1), -(intent.n - 1))
        prior_end  = curr_start - timedelta(days=1)
        prior_start = _add_months(prior_end.replace(day=1), -(intent.n - 1))
    elif intent.unit == "quarter":
        curr_end   = _start_of_quarter(today) - timedelta(days=1)
        curr_start = _add_quarters(curr_end.replace(day=1), -(intent.n - 1))
        prior_end  = curr_start - timedelta(days=1)
        prior_start = _add_quarters(prior_end.replace(day=1), -(intent.n - 1))
    else:  # year
        curr_end   = _start_of_year(today) - timedelta(days=1)
        curr_start = curr_end.replace(year=curr_end.year - intent.n + 1, month=1, day=1)
        prior_end  = curr_start - timedelta(days=1)
        prior_start = prior_end.replace(year=prior_end.year - intent.n + 1, month=1, day=1)

    def _lbl(s: date, e: date) -> str:
        return s.isoformat() if s == e else f"{s.isoformat()} to {e.isoformat()}"

    current_window = DateWindow(start=curr_start, end=curr_end,
                                 label=_lbl(curr_start, curr_end))
    prior_window = DateWindow(start=prior_start, end=prior_end,
                               label=_lbl(prior_start, prior_end)) if intent.compare else None
    return current_window, prior_window


# ══════════════════════════════════════════════════════════════════════════════
# SQL rewrite prompt
# ══════════════════════════════════════════════════════════════════════════════

def build_relative_date_rewrite_prompt(
    original_sql: str,
    window: DateWindow,
    date_col_hint: str = "",
) -> tuple[str, str]:
    """
    Build (system, user) prompt to rewrite base SQL to target the given window.
    Mirrors the pattern from period_comparison.py for consistency.
    """
    hint_block = (
        f"\nDate column hint from schema: {date_col_hint}" if date_col_hint else ""
    )
    system = (
        "You are a SQL date-range modifier. "
        "Your ONLY task is to rewrite the date filter in the SQL below to target a specific date range. "
        "RULES:\n"
        "1. Change ONLY the date filter — do NOT alter SELECT columns, GROUP BY, JOINs, metrics, or aggregations.\n"
        "2. Return ONLY the modified SQL — no explanation, no markdown fences.\n"
        "3. Use BETWEEN or >= / <= operators for the date range.\n"
        "4. If you cannot safely rewrite the date filter, return exactly: CANNOT_REWRITE\n"
        "5. Preserve all aliases, table qualifiers, and bracket notation exactly."
    )
    user = (
        f"Original SQL:\n{original_sql}\n\n"
        f"Target date window: {window.start.isoformat()} to {window.end.isoformat()}"
        f"{hint_block}\n\n"
        f"Rewrite the SQL so it retrieves only data within this date window.\n"
        f"Add a date filter if none exists; replace any existing date filter.\n"
        f"Use ISO date literals ('YYYY-MM-DD')."
    )
    return system, user
