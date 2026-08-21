"""Decide whether a result column is really a time axis, and what its cadence is.

Five detectors already answer the first question and disagree with each other:
``core.stat_signals._is_temporal_col``, ``core.insight._looks_temporal``,
``core.response_builder._looks_temporal``, ``core.chart_spec._looks_temporal_values``
and ``core.result_verifier._time_columns``. Each is calibrated for its own job --
a chart axis guess that is wrong costs a slightly odd chart, so those can afford
to be generous.

A forecast cannot. Projecting future periods off a column that is not a time
axis produces a confident, plausible, wrong number, so this module takes the
strictest of the five (``result_verifier``, the only one that rejects an encoded
integer key like ``CUSTOMER_PERIOD_KEY=900001``) and adds what a forecast needs
on top: parsing a period label to a real date, and inferring the observed
cadence of a series.

Deliberately NOT a refactor of the other five. Retro-fitting this strictness to
a chart-axis guess would start refusing charts that are fine today. New
consumers only.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from typing import Any

# Same vocabulary as core/result_verifier.py:18 -- kept identical on purpose, so
# the two agree about what a date-ish NAME looks like.
_TIME_NAME_RE = re.compile(
    r"(?:^|_)(?:date|dt|day|week|month|quarter|year|period|prd|yyyymm|yyyymmdd|fiscal|calendar)(?:_|$)",
    re.I,
)
_TIME_VALUE_RE = re.compile(
    r"^(?:\d{4}(?:[-/]\d{1,2})?(?:[-/]\d{1,2})?|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[- /]\d{2,4}|"
    r"q[1-4][ -]?\d{2,4})$",
    re.I,
)

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Approximate day counts, used only to name the cadence of an observed gap.
_GRAIN_BY_DAYS: tuple[tuple[str, int, int], ...] = (
    ("day", 1, 1),
    ("week", 7, 7),
    ("month", 28, 31),
    ("quarter", 89, 92),
    ("year", 365, 366),
)

_SAMPLE_SIZE = 20

# A parsed period outside this range is not a period, it is an identifier that
# happens to be digits. 900001 parses as year 9000 month 01 and would otherwise
# become a legitimate-looking date on a forecast axis.
_MIN_YEAR, _MAX_YEAR = 1900, 2199


def _bounded(value: date | None) -> date | None:
    if value is None or not (_MIN_YEAR <= value.year <= _MAX_YEAR):
        return None
    return value


def _encoded_time_value(value: Any) -> bool:
    """True for a YYYYMMDD / YYYYMM integer key that is a real calendar date.

    This is the guard that keeps a surrogate key out of the time axis: 900001
    parses as digits but is not a month of any year.
    """
    text = str(value or "").strip()
    match = re.fullmatch(r"(\d{4})(\d{2})(\d{2})?", text)
    if not match:
        return False
    try:
        year = int(match.group(1))
        date(year, int(match.group(2)), int(match.group(3) or 1))
    except ValueError:
        return False
    return 1900 <= year <= 2199


def _looks_like_a_period(value: Any) -> bool:
    if isinstance(value, (date, datetime)):
        return True
    text = str(value or "").strip()
    return bool(_TIME_VALUE_RE.match(text)) or _encoded_time_value(text)


def is_temporal_result_column(col_name: Any, values: list[Any]) -> bool:
    """Whether this column can serve as the time axis of a forecast.

    A date-like NAME is strong evidence but never sufficient on its own -- the
    values must also be plausible calendar values, or an integer surrogate key
    named ``*_PERIOD_KEY`` becomes a time axis and the projection is nonsense.
    A column with a non-date-like name may still qualify if every sampled value
    is unambiguously a period.
    """
    sample = [value for value in (values or [])[:_SAMPLE_SIZE] if value is not None]
    named = bool(_TIME_NAME_RE.search(str(col_name or "")))
    if not sample:
        # Nothing to corroborate with. Trust the name, refuse otherwise: an
        # empty column cannot carry a forecast either way.
        return named
    every_value_is_a_period = all(_looks_like_a_period(value) for value in sample)
    if named:
        return every_value_is_a_period
    return every_value_is_a_period


def parse_period_label(value: Any) -> date | None:
    """Turn a period label into the date it starts on, or None.

    Handles every shape the product actually emits: native dates, ISO strings,
    YYYYMMDD / YYYYMM integer keys, "Q2 2026", "Mar 2026", bare years.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, Decimal):
        value = int(value)
    text = str(value or "").strip()
    if not text:
        return None

    # YYYYMMDD / YYYYMM encoded key
    match = re.fullmatch(r"(\d{4})(\d{2})(\d{2})?", text)
    if match:
        try:
            return _bounded(date(int(match.group(1)), int(match.group(2)), int(match.group(3) or 1)))
        except ValueError:
            return None

    # ISO-ish: 2026, 2026-03, 2026-03-17 (also with slashes)
    match = re.fullmatch(r"(\d{4})(?:[-/](\d{1,2}))?(?:[-/](\d{1,2}))?", text)
    if match:
        try:
            return _bounded(date(int(match.group(1)), int(match.group(2) or 1), int(match.group(3) or 1)))
        except ValueError:
            return None

    # Q2 2026 / 2026-Q2
    match = re.fullmatch(r"q([1-4])[ -]?(\d{4})", text, re.I) or re.fullmatch(
        r"(\d{4})[ -]?q([1-4])", text, re.I
    )
    if match:
        groups = match.groups()
        quarter, year = (groups[0], groups[1]) if groups[0].isdigit() and len(groups[0]) == 1 else (groups[1], groups[0])
        try:
            return _bounded(date(int(year), (int(quarter) - 1) * 3 + 1, 1))
        except ValueError:
            return None

    # Mar 2026 / March 2026
    match = re.fullmatch(r"([a-z]{3,9})[- /](\d{2,4})", text, re.I)
    if match:
        month = _MONTHS.get(match.group(1)[:3].lower())
        if month:
            year = int(match.group(2))
            if year < 100:
                year += 2000
            try:
                return _bounded(date(year, month, 1))
            except ValueError:
                return None
    return None


def infer_series_grain(labels: list[Any]) -> tuple[str, float]:
    """Name the observed cadence of a series and how consistently it holds.

    Returns ``(grain, consistency)`` where consistency is the share of
    consecutive gaps that match the modal gap, 0.0-1.0. ``("", 0.0)`` when the
    labels cannot be parsed or there are fewer than two of them.

    This reads the DATA. ``core.contextual_dates.requested_temporal_grain``
    reads the QUESTION. They answer different things and a disagreement between
    them is worth a caveat, never a refusal -- a user asking "monthly" of a
    weekly table is asking a reasonable question.
    """
    parsed = [parse_period_label(label) for label in (labels or [])]
    parsed = [value for value in parsed if value is not None]
    if len(parsed) < 2:
        return "", 0.0

    gaps = [
        (parsed[i + 1] - parsed[i]).days
        for i in range(len(parsed) - 1)
        if (parsed[i + 1] - parsed[i]).days > 0
    ]
    if not gaps:
        return "", 0.0

    def _bucket(days: int) -> str:
        for name, low, high in _GRAIN_BY_DAYS:
            if low <= days <= high:
                return name
        return ""

    buckets = [_bucket(gap) for gap in gaps]
    named = [bucket for bucket in buckets if bucket]
    if not named:
        return "", 0.0
    grain, count = Counter(named).most_common(1)[0]
    return grain, count / len(gaps)


def seasonal_period_for_grain(grain: str) -> int:
    """How many periods make one seasonal cycle at this grain, 0 when none applies."""
    return {"day": 7, "week": 52, "month": 12, "quarter": 4}.get(str(grain or "").lower(), 0)
