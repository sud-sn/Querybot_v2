"""
core/fiscal_calendar.py
────────────────────────
Fiscal year / non-calendar period support.

Answers questions like:
  "Show revenue for FY2024"
  "Compare fiscal Q1 vs fiscal Q2"
  "What was performance in the last fiscal year?"
  "Show MTD and YTD figures using our fiscal calendar"
  "Revenue in fiscal year ending March 2024"

Design
──────
• Detection: regex on NL question for fiscal year / FY references.
• `parse_fiscal_reference()`: extracts (fy_year, fy_quarter, fy_period_type)
  from the question.
• `fiscal_date_range()`: converts a fiscal period to a (start_date, end_date)
  tuple given the fiscal year start month.
• `build_fiscal_sql_hint()`: generates dialect-aware SQL date arithmetic for
  the fiscal calendar.

The fiscal_year_start_month comes from db_cfg["fiscal_year_start_month"] which
defaults to 1 (January, i.e. calendar year).

Common configurations:
  - US Federal / Microsoft: October  (month 10)
  - UK financial year:      April    (month 4)
  - ANZ financial year:     July     (month 7)
  - India financial year:   April    (month 4)

Entry points
────────────
  detect_fiscal_intent(question) → bool
  parse_fiscal_reference(question) → FiscalRef
  fiscal_date_range(ref, fiscal_year_start_month) → tuple[str, str]
  build_fiscal_sql_hint(question, fiscal_year_start_month, db_type) → str
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Detection
# ══════════════════════════════════════════════════════════════════════════════

_FISCAL_PATTERNS = [
    re.compile(r"\bFY\s*\d{2,4}\b", re.I),
    re.compile(r"\bfiscal\s+(?:year|quarter|month|Q[1-4]|half)\b", re.I),
    re.compile(r"\bfinancial\s+(?:year|quarter|month)\b", re.I),
    re.compile(r"\bfiscal\s+Q[1-4]\b", re.I),
    re.compile(r"\bFQ[1-4]\b", re.I),
    re.compile(r"\b(?:last|current|this|next)\s+fiscal\s+(?:year|quarter)\b", re.I),
    re.compile(r"\byear\s+(?:ending|ended)\s+(?:March|June|September|December|march|june|september|december)\b", re.I),
    re.compile(r"\bfiscal\s+(?:ytd|mtd|qtd)\b", re.I),
]


def detect_fiscal_intent(question: str) -> bool:
    """Return True if the question references fiscal year periods."""
    return any(p.search(question) for p in _FISCAL_PATTERNS)


# ══════════════════════════════════════════════════════════════════════════════
# Fiscal period parsing
# ══════════════════════════════════════════════════════════════════════════════

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


@dataclass
class FiscalRef:
    fy_year:    int | None = None       # e.g. 2024
    fy_quarter: int | None = None       # 1-4
    fy_type:    str = "year"            # "year" | "quarter" | "half" | "ytd"
    relative:   str = ""               # "last" | "current" | "next" | ""
    year_end_month: int | None = None   # if "year ending March" style


def parse_fiscal_reference(question: str) -> FiscalRef:
    ref = FiscalRef()

    # FY2024 / FY 24 / FY24
    m = re.search(r"\bFY\s*(\d{2,4})\b", question, re.I)
    if m:
        yr = int(m.group(1))
        ref.fy_year = yr if yr > 100 else 2000 + yr
        ref.fy_type = "year"

    # fiscal Q1/Q2/Q3/Q4
    m = re.search(r"\bfiscal\s+Q(\d)\b|\bFQ(\d)\b", question, re.I)
    if m:
        ref.fy_quarter = int(m.group(1) or m.group(2))
        ref.fy_type = "quarter"

    # "year ending March" / "year ended June"
    m = re.search(r"\byear\s+end(?:ing|ed)\s+(\w+)\b", question, re.I)
    if m:
        month_name = m.group(1).lower()
        ref.year_end_month = _MONTH_MAP.get(month_name[:3])

    # relative: last / current / this / next
    m = re.search(r"\b(last|current|this|next)\s+fiscal\b", question, re.I)
    if m:
        ref.relative = m.group(1).lower()
        if ref.relative == "this":
            ref.relative = "current"

    return ref


# ══════════════════════════════════════════════════════════════════════════════
# Date range computation
# ══════════════════════════════════════════════════════════════════════════════

def fiscal_date_range(
    ref: FiscalRef,
    fiscal_year_start_month: int = 1,
    current_year: int = 2026,
) -> tuple[str, str]:
    """
    Return (start_date_str, end_date_str) for the described fiscal period.
    Dates in ISO format (YYYY-MM-DD).
    """
    fsm = fiscal_year_start_month  # 1-12

    # Determine the fiscal year number
    fy = ref.fy_year or current_year
    if ref.relative == "last":
        fy -= 1
    elif ref.relative == "next":
        fy += 1

    # For fiscal years starting in months other than Jan:
    # FY2024 starting in April means April 2023 – March 2024
    # (fiscal year is named by the calendar year in which it ends)
    if fsm == 1:
        fy_start = (fy, fsm)
        fy_end   = (fy, 12)
    else:
        # FY named by end calendar year
        start_calendar_year = fy - 1
        fy_start = (start_calendar_year, fsm)
        # End month = month before start month, in the named year
        end_month = fsm - 1 if fsm > 1 else 12
        end_year  = fy if fsm > 1 else fy - 1
        fy_end    = (end_year, end_month)

    if ref.fy_type == "quarter" and ref.fy_quarter:
        # Each fiscal quarter = 3 months starting from fy_start
        q = ref.fy_quarter - 1
        start_mo = ((fsm - 1 + q * 3) % 12) + 1
        start_yr = fy_start[0] + ((fsm - 1 + q * 3) // 12)
        end_mo   = ((fsm - 1 + q * 3 + 2) % 12) + 1
        end_yr   = fy_start[0] + ((fsm - 1 + q * 3 + 2) // 12)
        # End date = last day of end_mo
        import calendar
        last_day = calendar.monthrange(end_yr, end_mo)[1]
        return f"{start_yr}-{start_mo:02d}-01", f"{end_yr}-{end_mo:02d}-{last_day:02d}"

    import calendar
    last_day = calendar.monthrange(fy_end[0], fy_end[1])[1]
    return f"{fy_start[0]}-{fy_start[1]:02d}-01", f"{fy_end[0]}-{fy_end[1]:02d}-{last_day:02d}"


# ══════════════════════════════════════════════════════════════════════════════
# SQL hint
# ══════════════════════════════════════════════════════════════════════════════

_MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}


def build_fiscal_sql_hint(
    question: str,
    fiscal_year_start_month: int = 1,
    db_type: str = "azure_sql",
) -> str:
    fsm  = fiscal_year_start_month
    fsmn = _MONTH_NAMES.get(fsm, "January")
    ref  = parse_fiscal_reference(question)
    is_tsql = db_type in ("azure_sql", "sql_server", "mssql")
    is_snowflake = db_type == "snowflake"

    try:
        start_dt, end_dt = fiscal_date_range(ref, fsm)
        date_range_clause = f"BETWEEN '{start_dt}' AND '{end_dt}'"
    except Exception:
        date_range_clause = None

    if fsm == 1:
        # Calendar year — standard YEAR() function works
        if is_tsql:
            fy_expr = "YEAR(date_col)"
        else:
            fy_expr = "EXTRACT(YEAR FROM date_col)"
    else:
        # Fiscal year — shift by (12 - fsm + 1) months
        shift = 13 - fsm  # months to add so fiscal year aligns with calendar year
        if is_tsql:
            fy_expr = f"YEAR(DATEADD(month, {shift}, date_col))"
        elif is_snowflake:
            fy_expr = f"YEAR(DATEADD('month', {shift}, date_col))"
        else:
            fy_expr = f"EXTRACT(YEAR FROM date_col + INTERVAL '{shift} months')"

    if fsm == 1:
        fq_expr_comment = "Fiscal quarter = calendar quarter (fiscal year starts January)"
        if is_tsql:
            fq_expr = "DATEPART(quarter, date_col)"
        else:
            fq_expr = "EXTRACT(QUARTER FROM date_col)"
    else:
        fq_comment_shift = (fsm - 1)
        if is_tsql:
            fq_expr = f"DATEPART(quarter, DATEADD(month, {13 - fsm}, date_col))"
        elif is_snowflake:
            fq_expr = f"EXTRACT(QUARTER FROM DATEADD('month', {13 - fsm}, date_col))"
        else:
            fq_expr = f"EXTRACT(QUARTER FROM date_col + INTERVAL '{13 - fsm} months')"
        fq_expr_comment = f"Fiscal Q1 starts {fsmn}"

    hint = (
        f"FISCAL CALENDAR HINT:\n"
        f"The organisation's fiscal year starts in {fsmn} (month {fsm}). "
    )

    if date_range_clause:
        hint += (
            f"The referenced fiscal period maps to:\n"
            f"  date_col {date_range_clause}\n\n"
        )

    hint += (
        f"For fiscal year grouping use:\n"
        f"  {fy_expr}  AS fiscal_year\n\n"
        f"For fiscal quarter grouping use:\n"
        f"  {fq_expr}  AS fiscal_quarter  -- {fq_expr_comment}\n\n"
        f"Replace 'date_col' with the actual date column name.\n"
        f"Always qualify date filters with the fiscal date expressions above rather than "
        f"calendar YEAR() or MONTH() when the user says 'fiscal' or 'FY'."
    )
    return hint


# ══════════════════════════════════════════════════════════════════════════════
# Fiscal period label builder (for display)
# ══════════════════════════════════════════════════════════════════════════════

def fiscal_period_label(
    calendar_date_str: str,
    fiscal_year_start_month: int = 1,
) -> str:
    """
    Convert a calendar date string (YYYY-MM-DD or YYYY-MM) to a fiscal period label.
    E.g. "2023-07-01" with fsm=4 → "FY2024 Q2"
    """
    try:
        parts = calendar_date_str[:7].split("-")
        y, mo = int(parts[0]), int(parts[1])
        fsm = fiscal_year_start_month
        if fsm == 1:
            fy = y
        else:
            fy = y if mo >= fsm else y  # stays in same calendar year
            fy = y + 1 if mo >= fsm else y  # named by calendar year it ends in
        shift = 13 - fsm
        shifted_mo = ((mo - 1 + shift) % 12) + 1
        fq = (shifted_mo - 1) // 3 + 1
        return f"FY{fy} Q{fq}"
    except Exception:
        return calendar_date_str
