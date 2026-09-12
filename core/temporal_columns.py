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


# ── Is this numeric column a period, or is it the measure? ────────────────────
#
# A different question from is_temporal_result_column above, and it needs a
# different answer. That one asks "can this be a forecast's time axis", and its
# generosity is affordable there: it accepts a column whose values all LOOK like
# periods even when the name says nothing, because a forecast is refused for many
# other reasons too. Asked instead "is this the measure", that generosity is a
# wrong number: ORD_QTY holding [2020, 1500, 3200] passes it, and stealing the
# measure leaves the answer describing quantities as if they were calendar years.
#
# So this one requires the NAME and the VALUES to agree, and lets a measure
# suffix veto both. Measured on an Infor M3 mart, the cost of getting it wrong in
# the other direction was: IVC_YR classified as the measure, an answer reading
# "6 records — Invoice Yr ranges 2,020 to 2,025, avg 2,022.50", and -- because
# the year was the only numeric column left standing as a label candidate -- no
# trend, no ranking and no chart for "net sales by year" at all.

_PERIOD_NAME_RE = re.compile(
    r"(?:^|_)(?:date|dt|day|dow|week|wk|month|mth|quarter|qtr|year|yr|"
    r"period|prd|yyyymm|yyyymmdd|fiscal|fy|calendar)(?:_|$)",
    re.I,
)
# A column carrying one of these is a measure even when it also carries a period
# token: YR_TO_DT_AMT is an amount, and DLV_DAY_CNT is a count of days.
_MEASURE_NAME_RE = re.compile(
    r"(?:^|_)(?:amt|amount|qty|quantity|cnt|count|sum|tot|total|val|value|"
    r"pct|percent|percentage|rate|ratio|prc|price|cost|margin|avg|average|"
    r"bal|balance|min|max|stddev|variance)(?:_|$)",
    re.I,
)
# Month and quarter NUMBERS are read; day and week numbers are not. A column
# named *_MTH holding 1..12 is overwhelmingly a calendar month, and mistaking a
# duration in months for one costs a chart axis. Durations in days and weeks are
# ordinary business measures -- LEAD_TIME_DAY, AGE_WK -- and mistaking one of
# those for a period would steal the measure, which is the failure being fixed.
_UNIT_NUMBER_RANGES: tuple[tuple[re.Pattern[str], int, int], ...] = (
    (re.compile(r"(?:^|_)(?:month|mth)(?:_|$)", re.I), 1, 12),
    (re.compile(r"(?:^|_)(?:quarter|qtr)(?:_|$)", re.I), 1, 4),
)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (float, Decimal)):
        return int(value) if float(value).is_integer() else None
    text = str(value or "").strip()
    if not re.fullmatch(r"-?\d+", text):
        return None
    return int(text)


def _is_bare_year(value: Any) -> bool:
    number = _as_int(value)
    return number is not None and _MIN_YEAR <= number <= _MAX_YEAR


def _is_encoded_period(value: Any) -> bool:
    """A YYYYMM or YYYYMMDD integer that names a real month or day."""
    number = _as_int(value)
    if number is None:
        return False
    text = str(number)
    if len(text) == 6:
        year, month = int(text[:4]), int(text[4:])
        return _MIN_YEAR <= year <= _MAX_YEAR and 1 <= month <= 12
    if len(text) == 8:
        return _encoded_time_value(text)
    return False


def _is_calendar_period_value(value: Any) -> bool:
    if isinstance(value, (date, datetime)):
        return True
    return (_is_bare_year(value) or _is_encoded_period(value)
            or bool(_TIME_VALUE_RE.match(str(value or "").strip())))


def is_calendar_period_column(col_name: Any, values: list[Any]) -> bool:
    """Whether this column names a calendar period rather than a measure.

    Both halves must agree, and a measure suffix vetoes both:

        IVC_YR        [2020 .. 2025]        -> True
        IVC_PRD       [202401, 202402]      -> True
        IVC_MTH       [1 .. 12]             -> True
        ORD_QTY       [2020, 1500, 3200]    -> False   values only
        FISCAL_YR     ["north", "south"]    -> False   name only
        YR_TO_DT_AMT  [2024.50]             -> False   measure suffix
        DLV_DAY_CNT   [1, 2, 3]             -> False   measure suffix
        LEAD_TIME_DAY [14, 21, 30]          -> False   a duration, not a day
    """
    name = str(col_name or "")
    if not name or not _PERIOD_NAME_RE.search(name):
        return False
    if _MEASURE_NAME_RE.search(name):
        return False
    sample = [value for value in (values or [])[:_SAMPLE_SIZE]
              if value is not None and str(value).strip() != ""]
    if not sample:
        # A period name with nothing to corroborate is not evidence enough to
        # take a column out of the measure pool.
        return False
    if all(_is_calendar_period_value(value) for value in sample):
        return True
    for pattern, low, high in _UNIT_NUMBER_RANGES:
        if not pattern.search(name):
            continue
        numbers = [_as_int(value) for value in sample]
        if all(number is not None and low <= number <= high
               for number in numbers):
            return True
    return False


def period_columns(rows: list[dict], candidates: list[str]) -> list[str]:
    """Which of these columns are calendar periods, in the order given."""
    if not rows:
        return []
    return [
        column for column in candidates
        if is_calendar_period_column(column, [row.get(column) for row in rows])
    ]


def labels_are_bare_years(labels: list[Any]) -> bool:
    """Whether every one of these labels is a four-digit calendar year.

    Shared rather than copied because the two ``_looks_temporal`` classifiers
    (core/response_builder.py and core/insight.py) both needed it and the file
    they live in already documents four copies of that rule drifting apart.

    ALL of them, not any: "Depot 2019" is a warehouse, and a single year found
    somewhere inside a joined sample of labels is not a time axis. A year series
    is every label being nothing but a year.
    """
    sample = [str(label).strip() for label in (labels or [])[:_SAMPLE_SIZE]
              if label is not None and str(label).strip() != ""]
    if len(sample) < 2:
        return False
    return all(_is_bare_year(label) for label in sample)
