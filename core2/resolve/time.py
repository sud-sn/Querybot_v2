"""Time, resolved by code: windows to half-open date ranges, units and periods.

The planner writes time as the user said it ("last 6 months", "since January",
"1 to 7 June"); this module turns it into exact ``[start, end)`` dates. Relative
windows are anchored on today, or on the day after the data ends when the data
stopped more than a week ago (the answer then says so). Weeks start on Monday.
Fiscal units use the model's fiscal year start month.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from core2.plan.ir import Window

DAY = dt.timedelta(days=1)
STALE_AFTER = dt.timedelta(days=7)


@dataclass
class Range:
    start: dt.date | None           # inclusive; None = from the beginning of the data
    end: dt.date | None             # exclusive; None = to the end of the data
    notes: list[str] = field(default_factory=list)

    def contains(self, day: dt.date) -> bool:
        return (self.start is None or day >= self.start) and (self.end is None or day < self.end)


def _month_add(day: dt.date, months: int) -> dt.date:
    index = day.year * 12 + (day.month - 1) + months
    return dt.date(index // 12, index % 12 + 1, 1)


def unit_start(day: dt.date, unit: str, *, fiscal_start: int | None = None) -> dt.date:
    """The first day of the ``unit`` holding ``day``. Fiscal quarters and years start in ``fiscal_start``."""
    if unit == "day":
        return day
    if unit == "week":
        return day - dt.timedelta(days=day.weekday())
    if unit == "month":
        return day.replace(day=1)
    start = fiscal_start or 1
    months_into_year = (day.month - start) % 12
    year_start = _month_add(day.replace(day=1), -months_into_year)
    if unit == "quarter":
        return _month_add(year_start, (months_into_year // 3) * 3)
    if unit == "year":
        return year_start
    raise ValueError(f"no such unit: {unit}")


def add_units(day: dt.date, unit: str, n: int) -> dt.date:
    if unit == "day":
        return day + dt.timedelta(days=n)
    if unit == "week":
        return day + dt.timedelta(weeks=n)
    if unit == "month":
        return _month_add(day, n)
    if unit == "quarter":
        return _month_add(day, 3 * n)
    if unit == "year":
        return _month_add(day, 12 * n)
    raise ValueError(f"no such unit: {unit}")


def anchor(today: dt.date, last_data: dt.date | None) -> tuple[dt.date, str]:
    """The day relative windows count back from, and a note when it is not today."""
    if last_data is not None and today - last_data > STALE_AFTER:
        return last_data + DAY, (f"Your data ends {last_data.day} {last_data:%b %Y}, so recent periods are counted "
                                 "back from there.")
    return today, ""


def resolve_window(window: Window, *, today: dt.date, first_data: dt.date | None = None,
                   last_data: dt.date | None = None, fiscal_start: int | None = None) -> Range:
    """``window`` as exact dates. ``between`` dates are inclusive; the range is half-open."""
    fs = fiscal_start if window.fiscal else 1
    kind = window.kind
    if kind == "all":
        return Range(None, None)
    if kind == "between":
        start = window.start
        end = window.end + DAY if window.end else None
        if start and end and end <= start:
            start, end = window.end, window.start + DAY if window.start else None
        return Range(start, end)
    if kind == "since":
        # Up to and including today: dates planned ahead (a due date) are not "since".
        return Range(window.start, today + DAY)
    if kind == "until":
        return Range(None, window.end + DAY if window.end else None)
    unit = window.unit or "month"
    base, note = anchor(today, last_data)
    current = unit_start(base, unit, fiscal_start=fs)
    notes = [note] if note else []
    if kind == "last":
        n = window.n or 1
        if window.include_current:
            return Range(add_units(current, unit, -(n - 1)), base + DAY if base == today else base, notes)
        return Range(add_units(current, unit, -n), current, notes)
    if kind in ("this", "to_date"):
        return Range(current, (base + DAY) if base == today else base, notes)
    if kind == "previous":
        return Range(add_units(current, unit, -1), current, notes)
    raise ValueError(f"no such window: {kind}")


def year_back(day: dt.date) -> dt.date:
    """The same calendar day a year earlier (29 February becomes the 28th)."""
    try:
        return day.replace(year=day.year - 1)
    except ValueError:
        return day.replace(year=day.year - 1, day=28)


def shift(rng: Range, kind: str, *, unit: str | None = None) -> Range:
    """The comparison range: the period before (same length), or the same period a year earlier."""
    if rng.start is None or rng.end is None:
        raise ValueError("a comparison needs a bounded period")
    if kind == "same_period_last_year":
        return Range(year_back(rng.start), year_back(rng.end))
    # The previous period of the same shape: whole months stay whole months.
    if rng.start.day == 1 and rng.end.day == 1:
        months = (rng.end.year - rng.start.year) * 12 + rng.end.month - rng.start.month
        return Range(_month_add(rng.start, -months), rng.start)
    length = rng.end - rng.start
    return Range(rng.start - length, rng.start)


def periods(rng: Range, grain: str, *, fiscal_start: int | None = None) -> list[dt.date]:
    """The period starts covering a bounded range, in order."""
    if rng.start is None or rng.end is None:
        return []
    unit = {"fiscal_month": "month", "fiscal_quarter": "quarter", "fiscal_year": "year"}.get(grain, grain)
    fs = fiscal_start if grain.startswith("fiscal") else None
    out, day = [], unit_start(rng.start, unit, fiscal_start=fs)
    while day < rng.end:
        out.append(day)
        day = add_units(day, unit, 1)
    return out


def partial_periods(starts: list[dt.date], grain: str, *, first_data: dt.date | None, last_data: dt.date | None,
                    rng: Range | None = None, fiscal_start: int | None = None) -> list[dt.date]:
    """Periods the data (or the window) covers only in part: never compared as if whole."""
    unit = {"fiscal_month": "month", "fiscal_quarter": "quarter", "fiscal_year": "year"}.get(grain, grain)
    out = []
    for start in starts:
        end = add_units(start, unit, 1)
        covered_from = max(x for x in (start, first_data, rng.start if rng else None) if x is not None)
        covered_to = min(x for x in (end, (last_data + DAY) if last_data else None, rng.end if rng else None)
                         if x is not None)
        if covered_from > start or covered_to < end:
            out.append(start)
    return out


def label(start: dt.date, grain: str, *, fiscal_start: int | None = None) -> str:
    """How a period reads: Mar 2026, Week of 29 Jun 2026, Q1 2026, 2026, FY2026 Q3."""
    if grain == "day":
        return f"{start.day} {start:%b %Y}"
    if grain == "week":
        return f"Week of {start.day} {start:%b %Y}"
    if grain in ("month", "fiscal_month"):
        return f"{start:%b %Y}"
    if grain == "quarter":
        return f"Q{(start.month - 1) // 3 + 1} {start.year}"
    if grain == "year":
        return str(start.year)
    fs = fiscal_start or 1
    fiscal_year = start.year + (1 if fs != 1 and start.month >= fs else 0)
    if grain == "fiscal_quarter":
        return f"FY{fiscal_year} Q{((start.month - fs) % 12) // 3 + 1}"
    if grain == "fiscal_year":
        return f"FY{fiscal_year}"
    return start.isoformat()
