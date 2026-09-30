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
import unicodedata
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
# A month's word, whole or abbreviated, in English and as French writes it,
# accents folded -- the answer names a French reader's periods "mars 2025",
# "février 2025", and a French warehouse's labels are written the same way.
# Looked up whole: read by its first three letters, "Octane 87" was October
# 2087, "Junior 12" June 2012 and "Marseille 13" March 2013.
_MONTH_WORDS = {
    **_MONTHS, "january": 1, "february": 2, "march": 3, "april": 4, "june": 6, "july": 7, "august": 8,
    "sept": 9, "september": 9, "october": 10, "november": 11, "december": 12,
    "janvier": 1, "janv": 1, "fevrier": 2, "fevr": 2, "fev": 2, "mars": 3, "avril": 4, "avr": 4,
    "mai": 5, "juin": 6, "juillet": 7, "juil": 7, "aout": 8, "septembre": 9,
    "octobre": 10, "novembre": 11, "decembre": 12,
}

# A French month as a whole label -- "mars", "mars 2025", "févr. 25" -- the way
# the answer names a French reader's periods. A label that only holds a month's
# word is no month: a brand called "Mars Bar", a street "Rue de Juin".
FRENCH_MONTH_LABEL_RE = re.compile(
    r"(?:janvier|janv|f[ée]vrier|f[ée]vr|mars|avril|avr|mai|juin|juillet|juil|ao[uû]t|septembre|sept"
    r"|octobre|novembre|d[ée]cembre|d[ée]c)\.?(?:[\s-]+(?:19|20)?\d{2})?", re.I)

# A French day as a whole label -- "1 mars 2025", "1er mars 2025", "jeudi 4 mars
# 2025" -- a period of a daily series.
FRENCH_DAY_LABEL_RE = re.compile(
    r"(?:(?:lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\s+)?\d{1,2}(?:er)?\s+"
    r"(?:janvier|janv|f[ée]vrier|f[ée]vr|mars|avril|avr|mai|juin|juillet|juil|ao[uû]t|septembre|sept"
    r"|octobre|novembre|d[ée]cembre|d[ée]c)\.?\s+(?:19|20)\d{2}", re.I)

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

    # Q2 2026 / 2026-Q2, and T2 2026 as French writes a quarter (trimestre)
    match = re.fullmatch(r"[qt]([1-4])[ -]?(\d{4})", text, re.I) or re.fullmatch(
        r"(\d{4})[ -]?[qt]([1-4])", text, re.I
    )
    if match:
        groups = match.groups()
        quarter, year = (groups[0], groups[1]) if groups[0].isdigit() and len(groups[0]) == 1 else (groups[1], groups[0])
        try:
            return _bounded(date(int(year), (int(quarter) - 1) * 3 + 1, 1))
        except ValueError:
            return None

    # Mar 2026 / March 2026 / mars 2026 / févr. 2026, and Mar-26 or Mar/26.
    # Two digits after a space are a day, not a year: "Sep 24" is the 24th of
    # September, and was read as September 2024 -- three days of sales became
    # three Septembers, 2024 to 2026.
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    match = re.fullmatch(r"([a-z]{3,9})\.?(?:[- /](\d{4})|[-/](\d{2}))", folded, re.I)
    if match:
        month = _MONTH_WORDS.get(match.group(1).lower())
        if month:
            year = int(match.group(2) or match.group(3))
            if year < 100:
                year += 2000
            try:
                return _bounded(date(year, month, 1))
            except ValueError:
                return None
    return None


# A day written out with its year. "Sep 24" is no day to place: it is the 24th of September or September 2024.
_DAY_LABEL_FORMATS = ("%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%d-%b-%Y", "%d/%b/%Y")


def parse_month_name(value: Any) -> int | None:
    """The month a label names alone -- "March", "Mar", "mars", "janv." -- as 1 to 12, or None."""
    folded = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    return _MONTH_WORDS.get(folded.strip().rstrip(".").lower())


def parse_day_label(value: Any) -> date | None:
    """The day a label names -- "4 March 2025", "March 4, 2025", "1er mars 2025",
    "jeudi 4 mars 2025" -- or None."""
    text = str(value or "").strip()
    if not text:
        return None
    if FRENCH_DAY_LABEL_RE.fullmatch(text):
        found = re.search(r"(\d{1,2})(?:er)?\s+([^\W\d_]+)\.?\s+((?:19|20)\d{2})", text)
        folded = unicodedata.normalize("NFKD", found.group(2)).encode("ascii", "ignore").decode().lower()
        month = _MONTH_WORDS.get(folded)
        if month is None:
            return None
        try:
            return date(int(found.group(3)), month, int(found.group(1)))
        except ValueError:
            return None
    for fmt in _DAY_LABEL_FORMATS:
        try:
            return _bounded(datetime.strptime(text, fmt).date())
        except ValueError:
            continue
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
    r"period|prd|yyyymm|yyyymmdd|ym|fiscal|fy|calendar"
    # And as French names them: PERIODE, MOIS, ANNEE, TRIMESTRE.
    r"|periode|mois|annee|an|trimestre|semaine|jour|exercice)(?:_|$)",
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
    (re.compile(r"(?:^|_)(?:month|mth|mois)(?:_|$)", re.I), 1, 12),
    (re.compile(r"(?:^|_)(?:quarter|qtr|trimestre)(?:_|$)", re.I), 1, 4),
)


# A week's key (202530: the 30th week of 2025) and a fiscal period's (202513:
# the 13th period of a year kept in thirteen) are a year and the period's
# number, which runs past 12. Read as a calendar month they were no period at
# all, and the column a measure. Only a column named for such a period may
# hold them: CUSTOMER_KEY 202530 is a customer.
_NUMBERED_KEYS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"(?:^|_)(?:week|weeks|wk|wks|isoweek|semaine|semaines)(?:_|$)", re.I), 53),
    (re.compile(r"(?:^|_)(?:fiscal|fisc|fscl|fp|fyp|fiscale|fiscales|exercice|exercices|accounting|acct|posting|gl"
                r"|comptable|comptables)(?:_|$)", re.I), 17),
)


def _is_a_numbered_key(value: Any, highest: int) -> bool:
    """A year and a period's number, 1 to ``highest``: 202530."""
    number = _as_int(value)
    text = str(number) if number is not None else ""
    return len(text) == 6 and _MIN_YEAR <= int(text[:4]) <= _MAX_YEAR and 1 <= int(text[4:]) <= highest


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
    # Read as its words whatever the case and the accents: "OrderMonth" is
    # ORDER_MONTH, "Période" PERIODE.
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(col_name or ""))
    name = re.sub(r"[^A-Za-z0-9]+", "_", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode())
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
    for pattern, highest in _NUMBERED_KEYS:
        if pattern.search(name) and all(_is_a_numbered_key(value, highest) for value in sample):
            return True
    for pattern, low, high in _UNIT_NUMBER_RANGES:
        if not pattern.search(name):
            continue
        numbers = [_as_int(value) for value in sample]
        if all(number is not None and low <= number <= high
               for number in numbers):
            return True
    return False


# The last word of a column's name is what it holds: PROFIT_CENTRE_COUNT
# counts profit centres, SALES_QTY is a quantity sold, and neither is money,
# whatever the words before say.
_COUNT_HEAD_WORDS = frozenset({
    "count", "counts", "cnt", "number", "num", "nbr", "qty", "quantity", "quantities", "units",
})


def names_a_count(col_name: Any) -> bool:
    """Whether a column's name is a count or a quantity: its last word, in any
    case and separator ("PROFIT_CENTRE_COUNT", "salesQty"), or a count "of"
    something ("NUMBER_OF_INVOICES")."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(col_name or ""))
    words = [word for word in re.split(r"[^a-z0-9]+", spaced.lower()) if word]
    if not words:
        return False
    return words[-1] in _COUNT_HEAD_WORDS or (
        len(words) > 2 and words[0] in _COUNT_HEAD_WORDS and words[1] == "of"
    )


def names_a_measure(col_name: Any) -> bool:
    """Whether a column's name carries a measure token: AMT, QTY, VAL, PCT ...

    The veto is_calendar_period_column applies, for a caller that reads a period
    from the NAME alone: YR_TO_DT_AMT carries a date token and is an amount.
    """
    return bool(_MEASURE_NAME_RE.search(str(col_name or "")))


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
