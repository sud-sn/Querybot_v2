from __future__ import annotations

import logging
import functools
import json
import math
import re
from datetime import date, datetime
from decimal import Decimal
from statistics import mean, median, stdev
from typing import Any

from core.analysis_contract import collapse_rows_by_label
from core.display_formats import normalize_display_format
from core.i18n import (
    count_noun as _count_noun,
    format_date as _format_date,
    format_decimal as _format_decimal,
    format_percent as _format_percent,
    number_format as _number_format,
    plural as _t_plural,
    t as _t,
)
from core.clarification import extract_original_question
from core.query_semantics import detect_top_n_intent
from core.temporal_columns import (
    infer_series_grain, parse_day_label, parse_month_name, parse_period_label, period_columns)

log = logging.getLogger("querybot.response_builder")

_PREVIEW_ROW_CAP = 200
_RESULT_FORMATS = {"number", "currency", "percentage", "date", "text"}

_TEXT_ONLY_RESPONSE_KEYS = {
    "content", "text", "headline", "short_value", "insight_summary",
    "label", "title", "message", "reason", "description", "next_step",
    "executive_summary", "scope_badge", "duration_label", "data_source",
    "question", "question_id", "sql",
}


def sanitize_response_text_fields(value: Any, *, parent_key: str = "") -> Any:
    """Keep structured objects out of response fields consumed as text.

    Browser coercion of an accidental object produces ``[object Object]``.
    Enforce the response contract at the final payload boundary while leaving
    legitimate structured fields (chart, data, confidence, diagnostics) intact.
    """
    if parent_key in _TEXT_ONLY_RESPONSE_KEYS:
        if isinstance(value, list):
            scalar_items = [
                str(item) for item in value
                if isinstance(item, (str, int, float, bool))
            ]
            return " · ".join(scalar_items)
        if isinstance(value, dict):
            return ""
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        return ""
    if isinstance(value, dict):
        return {
            key: sanitize_response_text_fields(item, parent_key=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_response_text_fields(item) for item in value]
    return value

_CURRENCY_NAME_RE = re.compile(
    r"\b(revenue|amount|cost|price|total|sales|charge|fee|payment|spend|"
    r"value|income|profit|loss|margin|earning|billing|invoice|budget|"
    r"gross|net|balance|credit|debit|cash|dollar|usd|gbp|eur|salary|"
    r"wage|commission|rebate|discount|tax|surcharge|reimbursement)\b",
    re.IGNORECASE,
)
_PERCENT_NAME_RE = re.compile(r"\b(percent|percentage|pct|rate|ratio|share)\b", re.IGNORECASE)
_DATE_NAME_RE = re.compile(
    r"\b(date|dt|period|prd|yyyymm|yyyymmdd|year|month|quarter|week|day"
    r"|mois|periode|p\u00e9riode|trimestre|annee|ann\u00e9e|semaine|jour)\b",
    re.IGNORECASE,
)
_VALUE_TOKENS = {
    "amount", "avg", "average", "balance", "charge", "cost", "count",
    "earning", "fee", "gross", "income", "invoice", "loss", "margin",
    "net", "payment", "pct", "percent", "percentage", "price", "profit",
    "quantity", "rate", "ratio", "revenue", "sales", "share", "spend",
    "sum", "tax", "total", "value",
}
_DIMENSION_TOKENS = {
    "code", "date", "day", "description", "flag", "id", "identifier", "item",
    "key", "month", "name", "num", "number", "period", "product", "rank",
    "warehouse", "week", "year",
}
_FORMAT_STOP_TOKENS = {
    "a", "an", "and", "as", "by", "for", "from", "in", "is", "my", "of",
    "on", "per", "show", "the", "to", "total", "what", "with",
}


_CURRENCY_SYMBOLS = {
    "USD": "$", "INR": "₹", "EUR": "€", "GBP": "£",
    "CAD": "CA$", "AUD": "A$", "JPY": "¥",
}


def _format_number(
    value: Any,
    fmt: str | None = None,
    display_format: dict | None = None,
) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(num):
        return str(value)
    spec = normalize_display_format(display_format)
    fmt = _normalise_result_format(spec.get("type") or fmt)
    grouping = spec.get("grouping", True)
    digits = spec.get("fraction_digits")
    # Comma groups and a dot decimal are English. A French reader reads that
    # comma as the decimal point, so "1,234" is one and a bit rather than a
    # thousand -- the same digits, off by a factor of a thousand, with nothing
    # on screen to say so. core/i18n.py owns the pair, because the browser
    # formats the same numbers into table cells and the two must agree.
    number_spec = _number_format()

    def render(value: float, places: int, *, grouped: bool = True) -> str:
        return _format_decimal(value, places, grouping=grouped and grouping)

    if spec.get("style") == "compact" and abs(num) >= 1000:
        for divisor, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
            if abs(num) >= divisor:
                compact_digits = 1 if digits is None else digits
                compact = render(num / divisor, compact_digits, grouped=False)
                # rstrip on the language's own decimal separator: "1,0M" in
                # French is one point zero, and stripping a "." there would
                # leave the zero on.
                compact = compact.rstrip("0").rstrip(str(number_spec["decimal"]))
                return f"{compact}{suffix}"
    if fmt == "currency":
        digits = 2 if digits is None else digits
        code = str(spec.get("currency_code") or "USD")
        symbol = _CURRENCY_SYMBOLS.get(code, f"{code} ")
        absolute = render(abs(num), digits)
        gap = str(number_spec["currency_gap"])
        amount = (f"{absolute}{gap}{symbol.strip()}"
                  if number_spec["currency_after"] else f"{symbol}{absolute}")
        if num < 0 and spec.get("accounting"):
            return f"({amount})"
        return f"{'-' if num < 0 else ''}{amount}"
    if fmt == "percentage":
        if spec.get("scale") == "fraction":
            num *= 100
        digits = 2 if digits is None else digits
        return f"{render(num, digits)}{number_spec['percent_gap']}%"
    if digits is not None:
        return render(num, digits)
    # The >= 1000 test is on the value BEFORE rounding, which is why 999.999
    # comes back ungrouped as "1000.00". Preserved deliberately: this is a
    # translation, not a rounding change.
    if abs(num) >= 1000:
        return render(num, 0 if num.is_integer() else 2)
    return render(num, 0 if num.is_integer() else 2, grouped=False)


def _format_display_value(
    value: Any,
    fmt: str | None = None,
    display_format: dict | None = None,
) -> str:
    spec = normalize_display_format(display_format)
    if spec.get("type") == "date" or _normalise_result_format(fmt) == "date":
        parsed: date | None = None
        if isinstance(value, datetime):
            parsed = value.date()
        elif isinstance(value, date):
            parsed = value
        else:
            text = str(value or "").strip()
            match = re.match(r"^(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?", text)
            if not match:
                match = re.fullmatch(r"(\d{4})(\d{2})(\d{2})?", text)
            if match:
                try:
                    candidate = date(
                        int(match.group(1)),
                        int(match.group(2)),
                        int(match.group(3) or 1),
                    )
                    parsed = candidate if 1900 <= candidate.year <= 2199 else None
                except ValueError:
                    parsed = None
            if parsed is None:
                # A period written as its name or its year: 2021 is the year
                # 2021 -- grouped as a number it read "2,021" -- and "Q2 2025"
                # stays the quarter it names.
                part = _period_value(value)
                if part and part[1] == "year":
                    return str(part[0].year)
                if part and part[1] == "quarter":
                    return _format_date(part[0], "quarter")
                parsed = part[0] if part else None
        if parsed:
            # strftime("%B") reads the process C locale, which is English on
            # every server this runs on. core/i18n.py owns the month names,
            # because portal_base.html's window.qbDate formats the same
            # columns in the browser and the two must not disagree.
            return _format_date(parsed, spec.get("style") or "iso")
    if _normalise_result_format(fmt) == "text" and spec.get("type") in (None, "text"):
        return _text_cell(value)  # a week's or a fiscal period's key: its digits, never "202,530"
    return _format_number(value, fmt, spec)


# ── A period is shown by its name ────────────────────────────────────────────
# A month is "March 2025", a quarter "Q2 2025", a year "2025", in the reader's
# language -- "mars 2025", "T2 2025" -- and the same in the table, the
# headline, the sentences and the chart. The warehouse groups by 2025-03-01,
# 202503 or 2025-04-01 for the second quarter; those are how the SQL keys a
# period, not how a reader names one. The table said "2025-03" under a
# headline naming the same month "2025-03", a chart axis "Mar 2025", and the
# second quarter "2025-04" -- April.
_PERIOD_STYLES = {"month": "month_year_long", "quarter": "quarter", "year": "year", "day": "iso_date"}
# What a column's name says its periods are, read as words ("OrderMonth" is
# order month, "CAL_YM" a year and month). A week's or a fiscal period's key
# is no calendar month -- 202503 is the third week, or the third period of a
# fiscal year -- and a date named as one stays the day it is.
_WEEK_NAME_WORDS = frozenset({"week", "weeks", "wk", "wks", "isoweek", "semaine", "semaines"})
# The periods of a fiscal year, or of the books, are no calendar's months --
# in French, as an accounting system names them, too: PERIODE_FISCALE,
# PERIODE_EXERCICE, ACCOUNTING_PERIOD, GL_PERIOD, POSTING_PERIOD.
_FISCAL_NAME_WORDS = frozenset({"fiscal", "fisc", "fscl", "fp", "fy", "fyp", "fiscale", "fiscales", "fiscaux",
                                "exercice", "exercices", "accounting", "acct", "gl", "posting",
                                "comptable", "comptables"})
_FISCAL_PART_WORDS = frozenset({"period", "periods", "per", "prd", "periode", "month", "months", "mth", "mois",
                                "quarter", "qtr", "trimestre", "week", "wk"})
_MONTH_NAME_WORDS = frozenset({"month", "months", "mth", "mths", "mois", "yyyymm", "ym", "yearmonth", "yrmth"})
_QUARTER_NAME_WORDS = frozenset({"quarter", "quarters", "qtr", "qtrs", "trimestre", "trimestres"})
_YEAR_NAME_WORDS = frozenset({"year", "years", "yr", "yrs", "yyyy", "annee", "annees", "exercice", "fy"})
_DAY_NAME_WORDS = frozenset({"date", "dates", "dt", "day", "days", "jour", "jours", "yyyymmdd"})


def _column_words(column: str) -> list[str]:
    """A column's name as its words, lower case and without accents."""
    import unicodedata

    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(column or ""))
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return [word for word in re.split(r"[^A-Za-z0-9]+", text.lower()) if word]


def _named_grain(column: str) -> str:
    """The grain a column's name says its periods are at -- "week",
    "fiscal", "month", "quarter", "year", "day" -- or "" where it says none."""
    words = set(_column_words(column))
    if words & _WEEK_NAME_WORDS:
        return "week"
    if words & _FISCAL_NAME_WORDS:
        # A fiscal year is named by its year; a fiscal month or period by no
        # calendar month.
        return "fiscal" if words & _FISCAL_PART_WORDS or not words & _YEAR_NAME_WORDS else "year"
    for grain, names in (("month", _MONTH_NAME_WORDS), ("quarter", _QUARTER_NAME_WORDS),
                         ("year", _YEAR_NAME_WORDS), ("day", _DAY_NAME_WORDS)):
        if words & names:
            return grain
    return ""


# A month or a quarter the question names -- "January revenue for the last 3
# years", "Q1 revenue for each of the last 3 years" -- is what its periods are:
# the Januaries, the first quarters. The window's unit ("years") is how far back
# it reaches, and the plan's grain is that unit. Named so, bare: a month with
# its year ("since January 2024"), a range ("Jan-Jun") or a bound ("from
# March") is where a window starts and ends, and names no period.
_MONTH_NUMBERS = {"january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "april": 4, "apr": 4, "may": 5,
                  "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sept": 9,
                  "sep": 9, "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12}
_MONTH_NAMED_RE = re.compile(
    r"\b(?:january|february|march|april|june|july|august|september|october|november|december"
    r"|jan|feb|apr|jun|jul|aug|sept?|oct|nov)\b|\b(?:in|for|of|during)\s+may\b", re.I)
_QUARTER_NAMED_RE = re.compile(
    r"\bq([1-4])\b|\b(first|second|third|fourth)\s+quarter\b", re.I)
_WINDOW_BOUND_WORDS = frozenset({"since", "from", "until", "till", "through", "thru", "before", "after", "between",
                                 "to", "by", "depuis", "jusqu", "avant", "apres"})
_RANGE_WORDS = (r"(?:january|february|march|april|may|june|july|august|september|october|november|december"
                r"|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec|q[1-4])")
_RANGE_JOINER = r"\s*(?:-|\u2013|,|\b(?:to|through|thru|and|or|et)\b)\s*"
# A day may stand beside either month of a range: "January 5 - March 10", "1er March to 31 March".
_RANGE_DAY = r"\d{1,2}(?:st|nd|rd|th|er)?"
_RANGE_AFTER_RE = re.compile(rf"(?:\s+{_RANGE_DAY})?" + _RANGE_JOINER + rf"(?:the\s+)?(?:{_RANGE_DAY}\s+)?" + _RANGE_WORDS + r"\b", re.I)
_RANGE_BEFORE_RE = re.compile(r"\b" + _RANGE_WORDS + rf"(?:\s+{_RANGE_DAY})?" + _RANGE_JOINER + rf"(?:the\s+)?(?:{_RANGE_DAY}\s*)?$", re.I)
_YEAR_AFTER_RE = re.compile(r"\s*,?\s*(?:of\s+)?(?:19|20)\d{2}\b")
# A month with a day beside it -- "January 1st", "1er mars", "the first of January" -- is a date, never a
# period: sales on that day are the day's.
_DAY_AFTER_RE = re.compile(r"\s*\d{1,2}(?:st|nd|rd|th)?\b", re.I)
_DAY_BEFORE_RE = re.compile(r"(?:\b\d{1,2}(?:st|nd|rd|th|er)?|\b(?:first|last)\s+day)\s+(?:of\s+)?$|\b(?:first|last)\s+of\s+$", re.I)
# A month or quarter that qualifies something other than the periods: a cohort, a promotion, a campaign, a plan to
# compare with -- "revenue from the January cohort", "against the Q1 target" -- what is left out of them --
# "excluding January", "not counting Q1" -- the event that dates a group of customers -- "who joined in March",
# "contracts signed in January" -- or when a year starts: "our fiscal year starts in July", "starting April". The rows
# are not that month's. The words are those the question is canonicalised to, which leaves a French "hors", "sauf",
# "sans" and "acquis" as they are.
_QUALIFIES_AFTER_RE = re.compile(
    r"\s+(?:cohorts?|promotions?|promos?|campaigns?|launch(?:es)?|intakes?|class(?:es)?"
    r"|forecasts?|targets?|budgets?|plans?|goals?|quotas?|benchmarks?)\b", re.I)
_LEAVES_OUT = (r"excluding|excl\.?|except|without|minus|less|other\s+than|not\s+counting|not\s+including|but\s+not"
               r"|save|ex|leaving\s+out|omitting|skipping|ignoring|barring|besides"
               r"|hors|sauf|sans|mais\s+pas|excepte|a\s+(?:part|share)|en\s+excluant|(?:a\s+)?exclusion\s+of")
_DATES_A_GROUP = (r"(?:joined|acquired|launched|registered|enrolled|onboarded|hired|arrived|started|began|opened|released"
                  r"|introduced|signed(?:\s+up)?|created|acquis|arrives|inscrits?)\s+(?:in|en|of)"
                  r"|(?:cohorts?|cohorte|promotions?|campaigns?|campagne|intakes?)\s+of")
_STARTS_OR_ENDS = r"(?:starts?|starting|begins?|beginning|ends?|ending|commencant|commence)(?:\s+(?:in|en))?"
_QUALIFIES_BEFORE_RE = re.compile(
    rf"\b(?:{_LEAVES_OUT}|{_DATES_A_GROUP}|{_STARTS_OR_ENDS})\s+(?:the\s+)?$", re.I)


def _edge_of_a_window(text: str, match: re.Match) -> bool:
    """Whether a month or quarter the question names is where a window starts or ends -- "since January", "from
    March", "January 2024", "Jan-Jun" -- and no period of its own."""
    # A day or an article before the month -- "from 1st March", "since the 15th of June", "until the March report",
    # "jusqu'au 31 mars" -- is no word of what bounds it.
    day = _DAY_BEFORE_RE.search(text[:match.start()].lower())
    words = re.findall(r"[^\W\d_]+", text[:day.start() if day else match.start()].lower())
    if words[-1:] == ["the"]:
        words.pop()
    before = words[-1:]
    after = text[match.end():]
    # "jusqu'en mars" is two words, and the bound is the first.
    return bool((before and before[0] in _WINDOW_BOUND_WORDS) or words[-2:-1] == ["jusqu"] or _YEAR_AFTER_RE.match(after)
                or _RANGE_AFTER_RE.match(after) or _RANGE_BEFORE_RE.search(text[:match.start()]))


def _a_day_beside(text: str, match: re.Match) -> bool:
    return bool(_DAY_AFTER_RE.match(text[match.end():]) or _DAY_BEFORE_RE.search(text[:match.start()]))


def _names_a_day(canonical: str) -> bool:
    """Whether the question names a day -- "sales on January 1 each year", "le 1er janvier" -- that is no edge of a
    window: its periods are days, however often it asks for them."""
    text = canonical or ""
    return any(_a_day_beside(text, match) and not _edge_of_a_window(text, match) for match in _MONTH_NAMED_RE.finditer(text))


def _bare_named_period(canonical: str) -> str:
    """The month or quarter a question names as what its periods are --
    "month:1" (January), "quarter:1" (the first quarter) -- or "" where it
    names none, or more than one, or names one as the edge of a window, as a
    day's, or as what something else is filtered by."""
    text = canonical or ""
    found: set[str] = set()
    for match in list(_MONTH_NAMED_RE.finditer(text)) + list(_QUARTER_NAMED_RE.finditer(text)):
        if (_edge_of_a_window(text, match) or _a_day_beside(text, match)
                or _QUALIFIES_AFTER_RE.match(text[match.end():]) or _QUALIFIES_BEFORE_RE.search(text[:match.start()])):
            continue
        word = match.group(0).lower().split()[-1]
        if word in _MONTH_NUMBERS:
            found.add(f"month:{_MONTH_NUMBERS[word]}")
        else:
            quarter = match.group(1) or ("first", "second", "third", "fourth").index(match.group(2).lower()) + 1
            found.add(f"quarter:{quarter}")
    return found.pop() if len(found) == 1 else ""


def requested_period_grain(question: str, semantic_plan: dict | None = None) -> str:
    """The calendar grain the answer's periods were asked at -- "month",
    "quarter", "year", "week", "day" -- or "".

    The question's own words first, read in English ("par trimestre" is "by
    quarter" once canonicalised): a grouping it asks for. Then the plan's date
    disclosures, the grain its SQL was compiled at. A longer window's unit is
    no grain: "January revenue for the last 3 years" groups its months by
    nothing. A window in days or weeks is: "sales for the last 3 days" asks
    about days, and its "Sep-24" is the 24th of September, never September
    2024.

    A month or quarter the question names as what its periods are -- "January
    revenue for the last 3 years", "what did Jan sell each year" lists
    Januaries, whatever the window's unit or the grain asked -- follows the
    grain after a bar: "year|month:1" (period_grain names the Januaries of a
    column with it, and only them). Not where it qualifies something else -- "revenue by
    year for the January cohort", "excluding January" -- nor where the question lists
    months: the month is then what is filtered, or one of several. A day the
    question names -- "sales on January 1 each year" -- makes its grain a day."""
    named = grain = unit = ""
    try:
        from core.contextual_dates import detect_temporal_window, explicit_temporal_grain
        from core.i18n import get_active_language
        from core.question_normalizer import canonical_question

        canonical = canonical_question(question or "", get_active_language())
        named = _bare_named_period(canonical)
        grain = explicit_temporal_grain(canonical) or ""
        if grain == "year" and _names_a_day(canonical):
            grain = "day"  # "sales on January 1 each year" are the day's: a year is how often, not what
        if not grain:
            unit = str(detect_temporal_window(canonical).get("unit") or "").lower()
    except Exception as exc:
        log.warning("The grain a question asks its periods at was not read: %s", exc)
        unit = ""
    if not grain:
        for disclosure in (semantic_plan or {}).get("date_disclosures") or []:
            grain = str((disclosure or {}).get("requested_grain") or "").strip().lower() if isinstance(
                disclosure, dict) else ""
            if grain:
                break
    if not grain:
        grain = unit if unit in {"day", "week"} else ""
    return f"{grain}|{named}" if named else grain


def _period_value(value: Any) -> tuple[date, str] | None:
    """A period value as the day it starts and the finest part it names:
    "year" (2025), "quarter" (Q2 2025), "month" (2025-03, 202503, March 2025)
    or "day" (a date, 2025-03-05, 20250305, a timestamp at midnight). A time
    of day is no period: 2025-03-05 14:30 is an instant."""
    if isinstance(value, bool) or isinstance(value, (dict, list, tuple, set)):
        return None
    if isinstance(value, datetime):
        return (value.date(), "day") if value.time() == datetime.min.time() else None
    if isinstance(value, date):
        return value, "day"
    if isinstance(value, float):
        if not value.is_integer():
            return None
        value = int(value)
    text = str(value if value is not None else "").strip()
    # A bucket from DATE_TRUNC or a DATETIME column: midnight is the day.
    text = re.sub(r"^(\d{4}-\d{2}-\d{2})[ T]00:00(?::00(?:\.0+)?)?(?:Z|[+-]00:?00)?$", r"\1", text)
    if not text:
        return None
    compact = re.fullmatch(r"(\d{4})(\d{2})(\d{2})?", text)
    if compact:
        parsed = _parse_compact_date_value(text)
        return (parsed, "day" if compact.group(3) else "month") if parsed else None
    parsed = parse_period_label(text)
    if parsed is None:
        return None
    if re.fullmatch(r"\d{4}", text):
        return parsed, "year"
    if re.fullmatch(r"\d{4}[-/]\d{1,2}", text) or re.fullmatch(r"[^\W\d_]{3,9}\.?(?:[- /]\d{4}|[-/]\d{2})", text):
        return parsed, "month"
    if re.search(r"[qQtT][1-4]", text):
        return parsed, "quarter"
    return parsed, "day"


# A month's word and two digits after a hyphen or a slash: Mar-25 is March
# 2025 in a monthly report, and Oct-01 the first of October in a daily one.
_MONTH_OR_DAY_RE = re.compile(r"[^\W\d_]{3,9}\.?[-/]\d{2}")


def period_grain(values: list, requested: str = "", column: str = "") -> str:
    """The calendar grain a column's values are the periods of: "month",
    "quarter", "year", "day" (a date key only), or "" when they are not all
    periods, or cannot be told from a day.

    A month or quarter bucket is the first day of its period -- every builder
    emits DATEFROMPARTS(YEAR(x), MONTH(x), 1) -- so a column of firsts is a
    column of months, whatever it skips: a month with no sale is no row. Which
    bucket is said by the grain asked for, then by the column's own name: a
    MONTH column of January and April is two months, never two quarters. With
    neither, three or more whole quarters or years in a row are quarters or
    years; firsts that could be either, and those of a fiscal year or quarter
    that starts in a month no calendar quarter does (February, May, August,
    November), are left as the dates they are; and a column named as a date
    stays its dates unless they step month by month. A
    week's key and a fiscal period's are no calendar month, and dates that
    are not all firsts are days: a date key is shown as the day it is, never
    folded into its month, and a real date already reads as one.
    """
    present = [value for value in values if value not in (None, "")]
    parts = [_period_value(value) for value in present]
    if not parts or any(part is None for part in parts):
        return ""
    requested, _, named_period = (requested or "").partition("|")
    named = _named_grain(column)
    precisions = {part[1] for part in parts}
    starts = sorted({part[0] for part in parts})
    compact_days = all(re.fullmatch(r"\d{8}", str(value).strip()) for value in present)
    asked_days = requested in {"week", "day"}
    if named in {"week", "fiscal"} and not _real_dates(present):
        # A week's key and a fiscal period's are no calendar month, written
        # 202530 or "2025-01". A date is one, whatever its column is called:
        # POSTING_MONTH of the first of each month is those months.
        return "day" if precisions == {"day"} and compact_days else ""
    if any(_MONTH_OR_DAY_RE.fullmatch(str(value).strip()) for value in present):
        # "Sep-30" is September 2030 or the 30th of September, and "Oct-01" the
        # first of October or October 2001: only the grain asked for, or the
        # column's own name, says which. With neither the label stays as the
        # warehouse wrote it -- and a day's is never a month.
        if asked_days or named == "day" or not (
                requested in {"month", "quarter", "year"} or named in {"month", "quarter", "year"}):
            return ""
    if precisions == {"year"}:
        return "year"
    if "quarter" in precisions:
        return "quarter" if precisions <= {"quarter"} else ""
    if any(start.day != 1 for start in starts):
        return "day" if compact_days else ""
    quarters = all(start.month in (1, 4, 7, 10) for start in starts)
    januaries = all(start.month == 1 for start in starts)
    if named_period and named not in {"year", "day"} and _the_named_period_each_year(starts, named_period):
        return named_period.partition(":")[0]
    if named_period and named == "day" and _the_named_period_each_year(starts, named_period):
        return ""  # a column named as a date holds the dates a question names a month of, whatever it asks by
    for grain in (requested, named):
        if grain == "month":
            return "month"
        if grain == "quarter":
            if quarters:
                return "quarter"
            # Firsts on the same month of the quarter that is no calendar quarter's first -- a fiscal year
            # that starts in February, May, August or November -- are a fiscal quarter's, not a month's.
            fiscal = len(starts) > 1 and len({(start.year * 12 + start.month) % 3 for start in starts}) == 1
            return "" if fiscal else "month"
        if grain == "year" and januaries:
            return "year"
        # A year asked of firsts that are not all Januaries -- quarters, a
        # fiscal year's July -- says nothing of them: the column's name, else
        # their values, do.
    if precisions == {"month"}:
        # 202503, 2025-03, March 2025: a month by its spelling.
        return "month"
    if len(starts) < 2:
        return ""
    if named == "day" or asked_days:
        # A column named as a date, or asked about by the day or the week, stays
        # its dates unless they step month by month: an ORDER_DATE on the first
        # of March, June and September, or an EFFECTIVE_DATE each January, is
        # those days -- and "average daily sales by month" lists months.
        cadence, consistency = infer_series_grain(starts)
        return "month" if cadence == "month" and consistency >= 0.8 else ""
    steps = [(later.year - earlier.year) * 12 + later.month - earlier.month
             for earlier, later in zip(starts, starts[1:])]
    # Firsts that step only in whole years or quarters may be those, or the
    # months that fell on them: years or quarters where three or more come one
    # after another, else nothing -- a year's total named "January 2024" is as
    # wrong as a January named "2024".
    if januaries and all(step % 12 == 0 for step in steps):
        return "year" if len(starts) >= 3 and set(steps) == {12} else ""
    if quarters and all(step % 3 == 0 for step in steps):
        return "quarter" if len(starts) >= 3 and set(steps) == {3} else ""
    if all(step % 3 == 0 for step in steps):
        # Quarters or years that start in February, May, August or November, or any other month that is no
        # calendar quarter's: a fiscal calendar's, which no month name says.
        return ""
    return "month"


def _the_named_period_each_year(starts: list, named_period: str) -> bool:
    """Whether firsts are all the month or quarter the question names -- one a
    year, since they are distinct: the Januaries of "January revenue for the
    last 3 years", the first quarters of "Q1 revenue for each of the last 3
    years"."""
    kind, _, number = named_period.partition(":")
    month = int(number) if kind == "month" else (int(number) - 1) * 3 + 1
    return {start.month for start in starts} == {month}


def _real_dates(values: list) -> bool:
    """Values that are dates -- as dates, or as ISO days -- and no key written
    with a hyphen ("2025-01", the first week of the year)."""
    return all(
        re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[ T].*)?", str(value).strip())
        for value in values if value not in (None, "")
    )


def _whole_numbers(values: list) -> bool:
    """Values that are whole numbers, as numbers or as digits."""
    return all(
        (isinstance(value, (int, float)) and not isinstance(value, bool))
        or re.fullmatch(r"\d+(?:\.0+)?", str(value).strip()) is not None
        for value in values if value not in (None, "")
    )


def _period_keys(column: str, values: list) -> bool:
    """A calendar period column of keys no calendar names -- a week's
    (WEEK_KEY 202503) or a fiscal period's -- by its name and its values."""
    from core.temporal_columns import is_calendar_period_column

    present = [value for value in values if value not in (None, "")]
    return bool(present) and all(
        re.fullmatch(r"\d{6}(?:\d{2})?(?:\.0+)?", str(value).strip()) for value in present
    ) and is_calendar_period_column(column, present)


def _a_period_axis(column: str, values: list, labels: list, grain: str = "") -> bool:
    """Whether a label column is the periods of a series: periods named, or
    keys of a calendar period no calendar names -- WEEK_KEY 202501..202504 is
    a series of weeks, read by its keys."""
    return (_looks_temporal(labels) or bool(column_period_grain(values, grain, column))
            or _period_keys(column, values))


def chart_axis_style(rows: list[dict], x_key: str, column_formats: dict, display_formats: dict) -> str:
    """The style a chart's time axis names its periods in: the column's own
    (build_display_formats), or "key" for a week's or a fiscal period's key,
    which no calendar names and the axis leaves as it is."""
    style = str((display_formats.get(x_key) or {}).get("style") or "")
    if style:
        return style
    if column_formats.get(x_key) == "text" and _period_keys(x_key, [row.get(x_key) for row in rows]):
        return "key"
    return ""


def column_period_grain(values: list, requested: str = "", column: str = "") -> str:
    """period_grain, for a column: a whole number (202503, 2025) is a period
    only where the column's name says it holds periods -- "BUCKET" of 202503
    is no month, "OrderMonth" and "CAL_YM" are -- so the table, the headline
    and the sentences name the same column's periods, or none of them do."""
    if _whole_numbers(values) and not (
            _DATE_NAME_RE.search(" ".join(_column_words(column))) or _named_grain(column)):
        return ""
    return period_grain(values, requested, column)


def period_style(grain: str) -> str:
    """The display style a period of this grain is shown in, or ""."""
    return _PERIOD_STYLES.get(grain, "")


def narrative_period_labels(labels: list, grain: str = "", column: str = "") -> list[str]:
    """A series' period labels as the table and the headline show them.

    Periods reach the user through THREE paths, not two: the rendered table,
    the KPI headline, and the sentences written about the series. The first
    two go through the display formatter; narration did not, so the same
    answer said "2026-06 closed at $7.4M" in its headline and "trended flat
    from 2026-01-01 to 2026-06-01" three lines below it.

    ``grain`` is the grain the question asked for (requested_period_grain),
    which the table is formatted by too; the rule is period_grain's, so a
    month is "March 2025" here and in the cell beside it. A label that is not
    a period -- a day, a warehouse -- is returned as it came, and so is a
    series only partly made of periods: narration is prose, and a
    half-formatted series reads worse than an unformatted one.
    """
    raw = [str(label) if label is not None else "" for label in labels]
    found = column_period_grain(list(labels), grain, column)
    style = period_style(found)
    if not style:
        return raw
    named: list[str] = []
    for label, text in zip(labels, raw):
        part = _period_value(label)
        named.append(_format_date(part[0], style) if part else text)
    return named


def _sentence_start(label: Any) -> Any:
    """A period's name where it starts a sentence: French writes its months
    in lower case, and "juin 2025 a terminé à 600." opened the card."""
    return label[:1].upper() + label[1:] if isinstance(label, str) else label


def _numeric_cols(rows: list[dict]) -> list[str]:
    cols: list[str] = []
    if not rows:
        return cols
    for h in rows[0].keys():
        ok = True
        seen = False
        for r in rows:
            v = r.get(h)
            if v is None or v == "":
                continue
            seen = True
            try:
                float(str(v).replace(",", ""))
            except (TypeError, ValueError):
                ok = False
                break
        if ok and seen:
            cols.append(h)
    return cols


from core.analysis_evidence import MIN_CATEGORIES_FOR_CONCENTRATION

# The words a numeric column is named with when it names or orders a row
# rather than measures one -- CUSTOMER_NO, SOLD_TO_ID, ORDER_NUMBER, ITEM_KEY,
# ZIP_CODE, SALES_RANK, near core/chart_spec._ID_SUFFIX_RE, which reads them
# for the chart -- and in French first: "Numéro de commande", "Code article".
_IDENTIFIER_LAST = frozenset({
    "ID", "NO", "NUM", "NBR", "NR", "NUMBER", "NUMERO", "KEY", "SK", "PK", "FK", "CD", "CODE", "REF", "SEQ", "RANK"})
_IDENTIFIER_FIRST = frozenset({"NUMERO", "CODE", "CD", "ID"})
# Plurals no S ends: CHILDREN_NO is a number of children. Not STAFF: STAFF_NO
# is an employee's number, and a staff is one team.
_IRREGULAR_PLURALS = frozenset({"CHILDREN", "PEOPLE", "PERSONS", "MEN", "WOMEN", "PERSONNES", "GENS"})
# A name that ends in one of these numbers its rows, or may number a figure:
# PRICE_REF and NET_AMOUNT_NO are prices and amounts, CD_ITEM_NO an item.
# ID, KEY and CODE after a measure word name a row of it: COST_CENTER_ID,
# RATE_CODE.
_WEAK_IDENTIFIER_WORDS = frozenset({"NO", "NUM", "NBR", "NR", "NUMBER", "NUMERO", "REF", "SEQ", "RANK"})
# A name that says so alone: "ID", "KEY" -- not "NUMBER" or "NO", which a
# count is named too.
_IDENTIFIER_ALONE = frozenset({"ID", "KEY", "SK", "PK", "FK", "CODE", "CD", "REF", "SEQ", "RANK"})
# Words that make a number a count or a figure of one: TOTAL_NUMBER, ID_COUNT,
# "Nombre de commandes". A count named for its number first -- NUM_OF_RCT,
# NO_OF_EMPLOYEES, NUM_EMPLOYEES -- is none of _IDENTIFIER_FIRST's.
_COUNTED_WORDS = frozenset({"COUNT", "CNT", "TOTAL", "TOT", "SUM", "AVG", "AVERAGE", "NB", "NBRE", "NOMBRE"})
# And the names of a period's number: WEEK_NO, FISCAL_WEEK_NBR, DAY_NO,
# PERIOD_NO, "Numéro de semaine".
_PERIOD_WORDS = frozenset({
    "WEEK", "WEEKS", "WK", "DAY", "DAYS", "PERIOD", "PER", "PRD", "MONTH", "MTH", "YEAR", "YR", "QUARTER", "QTR",
    "HOUR", "HR", "FISCAL", "FY", "DATE", "DT", "SEMAINE", "JOUR", "MOIS", "ANNEE", "PERIODE", "TRIMESTRE"})


def _identifier_columns(rows: list[dict], candidates: list[str]) -> list[str]:
    """The numeric columns that identify a row: named for a number, a key, a
    code or a rank, and holding whole numbers. A result that carries a
    customer's number beside the customer was headed "ZED CO leads at
    40,101", the number read as the figure, and its analysis totalled
    customer numbers. Not a count, however it is named (NUM_OF_RCT,
    NO_OF_EMPLOYEES, NUM_EMPLOYEES, VISITS_NUMBER, TOTAL_NUMBER), nor a
    period's number (WEEK_NO), nor a figure with a fraction."""
    from core.analysis_contract import measure_additivity
    from core.units_of_measure import _COUNT_NOUNS, _QUANTITY_WORDS, _word_list

    from core.temporal_columns import names_a_measure

    def a_plural(word: str) -> bool:
        return (word.endswith("S") and not word.endswith("SS")) or word in _IRREGULAR_PLURALS or word in {
            "COUNT", "HEADCOUNT"}  # not ACCOUNT, COUNTY or COUNTER: GL_ACCOUNT_NO numbers an account

    def a_measure_ends(words: list[str]) -> bool:
        """Money or another measure ends the name (CD_BALANCE, NET_AMT), or
        stands right before the number word that does (PRICE_REF,
        NET_AMOUNT_NO). Not COST_CENTER_ID, PRICE_LIST_ID, RATE_CODE: an id
        names a row of what the measure word qualifies."""
        return names_a_measure(words[-1]) or (
            len(words) > 1 and words[-1] in _WEAK_IDENTIFIER_WORDS and names_a_measure(words[-2]))

    found = []
    for column in candidates:
        words = _word_list(column)
        # A quantity, a count or a period's number is a figure whatever else
        # its name says, and so is money: CD_BALANCE, PRICE_REF, NET_AMOUNT_NO.
        if not words or set(words) & (_QUANTITY_WORDS | _COUNT_NOUNS | _COUNTED_WORDS | _PERIOD_WORDS) \
                or a_measure_ends(words):
            continue
        if len(words) == 1:
            named = words[0] in _IDENTIFIER_ALONE
        else:
            # A plural beside NUMBER is a count of it: VISITS_NUMBER, CHILDREN_NO.
            counted = words[-1] in {"NUMBER", "NUM", "NO", "NBR", "NR"} and a_plural(words[-2])
            named = not counted and (words[-1] in _IDENTIFIER_LAST or words[0] in _IDENTIFIER_FIRST)
        if not named or measure_additivity(column)[1] == "event_count":
            continue
        values = [row.get(column) for row in rows if row.get(column) not in (None, "")]
        if values and all(_whole_number(value) for value in values):
            found.append(column)
    return found


def _whole_number(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    try:
        return float(value).is_integer()
    except (TypeError, ValueError, OverflowError):
        return False


def _measure_and_label_cols(
    rows: list[dict], numeric_cols: list[str], text_cols: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Split the numeric columns into measures and calendar periods.

    A period stored as an integer parses as a number, so every classifier in
    this file counted it as a measure, and the FIRST numeric column is what the
    narrative describes. On an Infor M3 mart "net sales by year" returns an
    integer IVC_YR beside NET_SLS_AMT, so the answer read

        6 records — Invoice Yr ranges 2,020 to 2,025, avg 2,022.50.

    -- the average of six calendar years, presented as the finding. And because
    the year was numeric, text_cols was EMPTY: the result had no label column,
    so it fell to numeric_table mode and produced no trend, no ranking and no
    chart either. The year was both the measure and the missing axis.

    Periods move to the LABEL candidates rather than being dropped, and they go
    after the text columns so a result carrying both a warehouse name and a year
    still narrates the warehouse. A row's identifier -- a customer's number, an
    order's key (_identifier_columns) -- is set aside, neither measure nor
    label: beside the customer's name it is the same customer again, and
    alone it is no axis to read.

    A result whose numeric columns are ALL periods is left exactly as it was:
    there is no measure to find, and taking the axis away as well would leave
    nothing to say at all. Identifiers that are the only figures left stay the
    figures.
    """
    from core.temporal_columns import period_columns

    period_cols = period_columns(rows, numeric_cols)
    rest = [column for column in numeric_cols if column not in period_cols]
    identifier_cols = _identifier_columns(rows, rest)
    measures = [column for column in rest if column not in identifier_cols]
    if not measures:
        identifier_cols, measures = [], rest
    if not measures or not (period_cols or identifier_cols):
        return numeric_cols, text_cols, []
    return measures, list(text_cols) + period_cols, period_cols


def _normalise_result_format(value: Any) -> str:
    fmt = str(value or "number").strip().lower()
    return fmt if fmt in _RESULT_FORMATS else "number"


def _normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _term_tokens(value: Any) -> set[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value or ""))
    text = text.replace("_", " ").replace("-", " ")
    payload = {
        tok.lower()
        for tok in re.findall(r"[A-Za-z0-9]+", text)
        if tok and tok.lower() not in _FORMAT_STOP_TOKENS
    }
    return payload


def _metric_tokens(metric: dict) -> set[str]:
    raw = " ".join(
        str(metric.get(k) or "")
        for k in ("name", "synonyms", "description", "required_columns")
    )
    tokens = _term_tokens(raw)
    return {t for t in tokens if t not in _FORMAT_STOP_TOKENS}


def _is_dimension_like_column(column: str) -> bool:
    tokens = _term_tokens(column)
    if not tokens:
        return False
    if tokens & _VALUE_TOKENS:
        return False
    return bool(tokens & _DIMENSION_TOKENS)


def _format_matches_column_name(fmt: str, column: str) -> bool:
    if fmt == "currency":
        return bool(_CURRENCY_NAME_RE.search(column))
    if fmt == "percentage":
        return bool(_PERCENT_NAME_RE.search(column))
    if fmt == "date":
        return bool(_DATE_NAME_RE.search(column))
    return False


def _columns_for_metric_format(
    rows: list[dict],
    metric: dict,
    *,
    strict: bool = False,
) -> list[str]:
    if not rows:
        return []

    fmt = _normalise_result_format(metric.get("result_format"))
    if fmt == "number" and not strict:
        return []

    headers = list(rows[0].keys())
    numeric_cols = set(_numeric_cols(rows))
    text_cols = {h for h in headers if h not in numeric_cols}
    metric_terms = _metric_tokens(metric)

    if fmt in {"currency", "percentage", "number"}:
        candidates = [h for h in headers if h in numeric_cols]
    elif fmt == "date":
        candidates = [h for h in headers if h not in numeric_cols or _format_matches_column_name(fmt, h)]
    else:
        candidates = [h for h in headers if h in text_cols]

    scored: list[tuple[int, str]] = []
    for header in candidates:
        header_terms = _term_tokens(header)
        term_match = bool(metric_terms and (header_terms & metric_terms))
        format_name_match = _format_matches_column_name(fmt, header)
        value_name_match = bool(header_terms & _VALUE_TOKENS)
        score = 0
        if term_match:
            score += 5
        if format_name_match:
            score += 4
        if value_name_match and (strict or term_match or format_name_match):
            score += 2
        if _is_dimension_like_column(header) and not format_name_match:
            score -= 4
        if score > 0:
            scored.append((score, header))

    if scored:
        scored.sort(key=lambda item: (-item[0], headers.index(item[1])))
        return [h for _, h in scored]

    value_candidates = [h for h in candidates if not _is_dimension_like_column(h)]
    if strict and len(value_candidates) == 1:
        return value_candidates
    if strict and value_candidates:
        return value_candidates
    return []


def build_column_formats(
    rows: list[dict],
    display_context: dict | None = None,
    explicit_formats: dict | None = None,
) -> dict[str, str]:
    """
    Build a header -> display-format map for the frontend.

    Metric result_format should drive presentation only. SQL remains numeric/date
    friendly so sorting, charting, CSV export, and result-chat calculations keep
    working.
    """
    if not rows:
        return {}

    headers = list(rows[0].keys())
    by_norm = {_normalise_key(h): h for h in headers}
    formats: dict[str, str] = {}

    # A named-period comparison publishes a format for its own period columns.
    # Without one, core/chart_spec.py demotes a whole-number column whose name
    # matches none of its currency/percent/count patterns (TONNAGE_2024) to an
    # identifier, and the chart loses the series entirely. Merged UNDER the
    # caller's explicit formats, which still win.
    period_formats = (
        ((display_context or {}).get("period_comparison") or {}).get("column_formats")
        if isinstance(display_context, dict) else None
    )
    for raw_col, raw_fmt in {**(period_formats or {}), **(explicit_formats or {})}.items():
        header = by_norm.get(_normalise_key(raw_col))
        fmt = _normalise_result_format(raw_fmt)
        # Allow explicit "number" through — it lets callers override currency
        # heuristics for columns that happen to have monetary-sounding names.
        if header:
            formats[header] = fmt

    ctx = display_context or {}
    requested = str(ctx.get("period_grain") or "") if isinstance(ctx, dict) else ""
    metrics = ctx.get("metrics") if isinstance(ctx, dict) else []
    if isinstance(metrics, dict):
        metrics = [metrics]
    if not isinstance(metrics, list):
        metrics = []
    strict = (ctx.get("format_scope") if isinstance(ctx, dict) else "") == "metric_registry"

    # A column of periods is shown by their names, whatever the warehouse
    # keeps them as: an ERP key (202601, 20260131, an INT), a bucket date
    # (2026-01-01, a timestamp at midnight), a year, "Q2 2026" or "Mar-26".
    # A key is a period only where the column's name says so
    # (column_period_grain): 202601 is also a customer number.
    #
    # Marked only at MONTH, QUARTER or YEAR grain -- and a date KEY at its day
    # -- each shown by its name in the style build_display_formats gives it,
    # and this restriction is load-bearing rather than cautious. The date
    # renderers fall through to `YYYY-MM` for a date with no style
    # (portal_chat.html), so declaring a DAILY column a date would collapse
    # every day of a month onto one label and silently merge rows in the
    # reader's eyes. A day-grain date column already displays correctly.
    #
    # The test is the BUCKET SHAPE, not the cadence. A governed month or
    # quarter bucket is always the FIRST DAY of its period -- every builder
    # emits DATEFROMPARTS(YEAR(x), MONTH(x), 1), see
    # core.contextual_dates.format_period_bucket_expression -- so day == 1 is
    # a shape the server itself created and can recognise. Cadence alone
    # cannot tell a bucket from a real day, and the difference is destructive:
    # invoices due on the 15th of each month, and month-END balance dates, both
    # step ~30 days, and relabelling either one "2026-01" erases the exact
    # thing the reader needs.
    #
    # Checked over EVERY rendered row rather than a sample, because the format
    # is applied to every row: a page of month buckets followed by a daily
    # tail would otherwise collapse the tail onto shared labels and silently
    # merge rows on screen. A row with no period is shown empty.
    for header in headers:
        if header in formats:
            continue
        column_values = [row.get(header) for row in rows if row.get(header) not in (None, "")]
        if column_values and column_period_grain(column_values, requested, header) in {
                "month", "quarter", "year", "day"}:
            formats[header] = "date"
        elif _period_keys(header, column_values):
            # A week's or a fiscal period's key is named by no calendar, and
            # is no amount either: 202503, never "202,503".
            formats[header] = "text"

    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        fmt = _normalise_result_format(metric.get("result_format"))
        if fmt == "number" and not strict:
            continue
        for header in _columns_for_metric_format(rows, metric, strict=strict):
            formats.setdefault(header, fmt)

    return formats


def build_display_formats(
    rows: list[dict],
    column_formats: dict[str, str] | None,
    explicit: dict | None = None,
    requested_grain: str = "",
) -> dict[str, dict]:
    """The display style of each column: what the reader asked for, and for
    every period column build_column_formats marked a date, its period's name
    (period_grain): a month "March 2025", a quarter "Q2 2025", a year "2025",
    a date key its day. The table in the portal, the table in a chat, the
    headline, the KPI and the chart axis all read these."""
    headers = list(rows[0].keys()) if rows else []
    formats: dict[str, dict] = {
        header: dict(spec) for header, spec in (explicit or {}).items()
        if header in headers and isinstance(spec, dict)
    }
    for header, fmt in (column_formats or {}).items():
        if fmt != "date" or header in formats or header not in headers:
            continue
        values = [row.get(header) for row in rows if row.get(header) not in (None, "")]
        style = period_style(column_period_grain(values, requested_grain, header))
        if style:
            formats[header] = {"type": "date", "style": style}
    return formats


def _parse_compact_date_value(value: Any) -> date | None:
    text = str(value or "").strip()
    match = re.fullmatch(r"(\d{4})(\d{2})(\d{2})?", text)
    if not match:
        return None
    try:
        parsed = date(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3) or 1),
        )
    except ValueError:
        return None
    return parsed if 1900 <= parsed.year <= 2199 else None


def _text_cols(rows: list[dict], numeric_cols: list[str]) -> list[str]:
    return [h for h in (rows[0].keys() if rows else []) if h not in numeric_cols]


def _ranked_items(rows: list[dict], label_col: str, value_col: str,
                  limit: int = 5) -> list[dict]:
    """The leading labels by value, each appearing exactly once.

    Ranking the raw rows is right when a label appears once. A result grouped
    by two things carries a row per label PER PERIOD, and then the leader and
    the runner-up came back as the same warehouse in two different months,
    with its share divided by a total counted three times over.

    Collapsing to one row per label means summing, and summing is only sound
    when the measure adds up. A margin percentage summed across months is
    arithmetic on nothing; so is a stock balance, which is semi-additive
    precisely because it may not be summed across time. In those cases this
    returns nothing rather than a plausible wrong leader, and the caller's
    existing guards leave the comparison out of the answer.
    """
    totals: dict[str, float] = {}
    for row in rows:
        label = str(row.get(label_col, ""))
        totals[label] = totals.get(label, 0.0) + _to_float_z(row.get(value_col))

    if len(totals) != len(rows):
        from core.analysis_contract import measure_class_for_column
        if measure_class_for_column(value_col) != "additive":
            log.info(
                "No ranking computed: %r repeats across %d rows and %r does "
                "not add up, so one row per label cannot be formed",
                label_col, len(rows), value_col)
            return []

    ordered = sorted(totals.items(), key=lambda pair: pair[1], reverse=True)
    return [{"label": label, "value": value} for label, value in ordered[:limit]]


def _narrative_label_column(rows: list[dict], text_cols: list[str]) -> str:
    """The text column the narrative should speak about.

    The first one is right for a result grouped by a single thing. Grouped by
    TWO -- "revenue by warehouse for the last three months" -- the calendar
    repeats once per warehouse, and describing the result along that axis
    builds a series out of unrelated rows: the live workspace reported a
    +1,437.3% trend between two different warehouses a quarter apart, and
    never mentioned a warehouse in the prose at all.

    So a temporal column whose labels REPEAT is skipped in favour of one that
    does not, because the dimension the reader asked about is the one that is
    not the repeating calendar. A temporal column with distinct labels is a
    real series and stays first.
    """
    if len(text_cols) < 2:
        return text_cols[0]
    for col in text_cols:
        labels = [str(row.get(col, "")) for row in rows]
        if _looks_temporal(labels) and len(set(labels)) != len(labels):
            continue
        # One member on every row is what the question filtered to, not what
        # it breaks down: "stock on hand for the North Depot warehouse" is
        # North Depot's, by unit.
        if len(rows) > 1 and len(set(labels)) == 1:
            continue
        return col
    return text_cols[0]


def _looks_temporal(values: list[str]) -> bool:
    """Do these labels name periods?

    This is the fourth private temporal classifier in the codebase, and it was
    the narrowest: English month names only, with no quarters at all. That is
    load-bearing rather than cosmetic, because _narrative_label_column asks it
    whether a repeating column is a calendar to skip over. Answer "no" for
    "Q1 2026" or "janvier" and the repeating calendar becomes the dimension the
    whole narrative is written about -- which is how a sparse quarter grid
    still reported "upward trend, 300%" between two different warehouses three
    quarters apart, after the trend detector itself had been fixed.

    Quarters and the French month names are added here rather than in a fifth
    place. The remaining copies are core.stat_signals._is_temporal_col (the
    values-and-name classifier), core.insight (which already listed quarters,
    so the two disagreed) and core.analysis_evidence.period_order_key (which
    answers a harder question -- what ORDER, not merely whether). Consolidating
    all four is worth doing and is bigger than this fix; the comment is here so
    the next person finds them together.
    """
    sample = " ".join(v.lower() for v in values[:8] if v)
    # Every token is matched as a WHOLE WORD. The long names used to be
    # substring-matched, which read "Mayfield" as May and would have read
    # "Marseille" as mars the moment French was added. A period label is a
    # whole label; nothing is gained by finding one inside a longer word, and a
    # warehouse or a customer called Mayfield is not a month.
    tokens = [
        # English, full and abbreviated
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
        "oct", "nov", "dec",
        # Grain words, both languages
        "week", "month", "quarter", "year", "date",
        "semaine", "mois", "trimestre", "année", "annee",
    ]
    from core.temporal_columns import FRENCH_DAY_LABEL_RE, FRENCH_MONTH_LABEL_RE, labels_are_bare_years

    # The product ships French tenants whose warehouses hold French period
    # labels ("mars 2025", as core.stat_signals._is_temporal_col has read them
    # since it was written): a label that is a month and nothing else. A brand
    # called "Mars Bar" is no month.
    french = [str(value).strip() for value in values[:8] if value and str(value).strip()]
    return (
        (bool(french) and all(FRENCH_MONTH_LABEL_RE.fullmatch(label) or FRENCH_DAY_LABEL_RE.fullmatch(label)
                              for label in french))
        or bool(re.search(r"\b\d{4}[-/]\d{1,2}([-/]\d{1,2})?\b", sample))
        # Q1 2026, 2026-Q1, T1 2026 (trimestre). Absent entirely before, so a
        # fiscal-quarter column read as an ordinary business dimension. A T
        # is a quarter only beside its year: T1..T4 alone are tiers, and were
        # narrated "trended down 94.4% from T1 to T4".
        or bool(re.search(r"\bq[1-4]\b|\bt[1-4]\s*[-/]?\s*(?:19|20)\d{2}\b|(?:19|20)\d{2}\s*[-/]?\s*t[1-4]\b",
                          sample))
        # A bare year is the coarsest period label there is and was the one
        # shape no classifier read. "Net sales by year" over an M3 mart returns
        # an INTEGER year column, so the series was not merely mis-grained --
        # it was not a series at all, and the answer ranked six calendar years
        # against each other as if they were warehouses. Matched over EVERY
        # label, not found inside the joined sample: a warehouse called "Depot
        # 2019" is not a period. Shared with core/insight.py's copy of this
        # classifier rather than written twice.
        or labels_are_bare_years(values)
        or any(re.search(r"\b" + re.escape(tok) + r"\b", sample)
               for tok in tokens)
    )


def _temporal_sort_value(value: Any) -> tuple[int, int, int, int] | None:
    """Return a sortable date/period key without changing displayed values."""
    if isinstance(value, datetime):
        return value.year, value.month, value.day, 0
    if isinstance(value, date):
        return value.year, value.month, value.day, 0
    text = str(value or "").strip()
    if not text:
        return None
    match = re.match(r"^((?:19|20)\d{2})[-/]?(\d{2})(?:[-/]?(\d{2}))?$", text)
    if match:
        year, month, day = (int(part or 1) for part in match.groups())
        if 1 <= month <= 12 and 1 <= day <= 31:
            return year, month, day, 0
    # A week's or a fiscal period's key past the twelfth (202530): in order.
    match = re.fullmatch(r"((?:19|20|21)\d{2})(1[3-9]|[2-5]\d)", text)
    if match:
        return int(match.group(1)), int(match.group(2)), 0, 1
    # A bare year, as the calendar reads one (1900 to 2199): listed newest
    # first it read "trended down 12.5% from 2025 to 2023" of revenue that rose.
    if re.fullmatch(r"(?:19|20|21)\d{2}", text):
        return int(text), 1, 1, 0
    match = re.match(r"^(?:Q([1-4])\s*[-/]?\s*((?:19|20)\d{2})|((?:19|20)\d{2})\s*[-/]?\s*Q([1-4]))$", text, re.I)
    if match:
        quarter = int(match.group(1) or match.group(4))
        year = int(match.group(2) or match.group(3))
        return year, ((quarter - 1) * 3) + 1, 1, quarter
    for fmt in ("%B %Y", "%b %Y", "%b-%y", "%B-%y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.year, parsed.month, parsed.day, 0
        except ValueError:
            continue
    # A day written out, in English or French: "4 March 2025", "jeudi 4 mars 2025".
    day = parse_day_label(text)
    if day is not None:
        return day.year, day.month, day.day, 0
    # The periods the answer names in French -- "mars 2025", "T2 2025" -- and
    # any other label a period is read from (parse_period_label).
    parsed = parse_period_label(text)
    if parsed is not None and (re.search(r"\d{4}", text) or re.fullmatch(r"[^\W\d_]{3,9}\.?[-/]\d{2}", text)):
        quarter = re.fullmatch(r"(?:[qQtT]([1-4])\s*[-/]?\s*\d{4}|\d{4}\s*[-/]?\s*[qQtT]([1-4]))", text)
        return parsed.year, parsed.month, parsed.day, int((quarter.group(1) or quarter.group(2))) if quarter else 0
    return None


_AGGREGATE_CALL_RE = re.compile(r"\b(?:sum|count|avg|average|min|max|median|stdev|stddev)\s*\(", re.IGNORECASE)


def _first_order_key(clause: str) -> str:
    """The first key of an ORDER BY's clause, without its direction: the text
    up to the first comma, or the parenthesis that closes what holds the
    clause (a window's OVER (...), a subquery)."""
    depth, first = 0, ""
    for char in clause:
        depth += (char == "(") - (char == ")")
        if depth < 0 or (char == "," and depth == 0):
            break
        first += char
    return re.sub(r"\s+(?:ASC|DESC)\b.*$|\s+NULLS\s+(?:FIRST|LAST)\b.*$", "", first.strip(),
                  flags=re.IGNORECASE | re.DOTALL).strip()


def _period_axes(rows: list[dict]) -> list[str]:
    """The result's columns that hold periods: the axis of a series."""
    numeric_cols = _numeric_cols(rows)
    _measures, label_cols, _periods = _measure_and_label_cols(rows, numeric_cols, _text_cols(rows, numeric_cols))
    return [column for column in label_cols
            if _a_period_axis(column, [row.get(column) for row in rows], [str(row.get(column, "")) for row in rows])]


_DATE_WORDS = frozenset({"date", "dates", "datetime", "dt", "ts", "timestamp", "jour",
                         "created", "posted", "updated", "modified", "booked"})
_WORDS_ENDING_IN_DATE = frozenset({"update", "updates", "candidate", "candidates", "mandate", "validate",
                                   "consolidate"})


def _aggregate_calls(text: str) -> list[tuple[str, str]]:
    """Every aggregate call in an expression -- SUM(x), MAX(DATEFROMPARTS(YEAR(d),
    MONTH(d), 1)) -- as its name and its argument, read to the parenthesis that
    closes it however many functions stand inside."""
    text = re.sub(r"'(?:[^']|'')*'", "''", text)
    calls = []
    for match in _AGGREGATE_CALL_RE.finditer(text):
        depth, end = 1, len(text)
        for index in range(match.end(), len(text)):
            depth += (text[index] == "(") - (text[index] == ")")
            if depth == 0:
                end = index
                break
        calls.append((re.match(r"\w+", match.group(0)).group(0).lower(), text[match.end():end]))
    return calls


def _names_a_date(expression: str) -> bool:
    """Whether a column or expression is named as a date: order_date, OrderDate,
    ORDERDATE, DATEKEY, TXN_TS, created_at, INVOICE_DATETIME. Not a word that
    only holds one: lifetime_value, candidate_score. Nor a time or a day, which
    is as often a length of time as a date: lead_time, days_to_ship."""
    words = _column_words(expression)
    return bool(_DATE_WORDS.intersection(words)) or any(
        word.startswith("date") or (word.endswith("date") and len(word) > 4 and word not in _WORDS_ENDING_IN_DATE)
        for word in words)


def _period_order(rows: list[dict]) -> str:
    """How the rows' periods stand in time: "in" where all can be placed and
    they come in its order or the reverse of it, "out" where all can be placed
    and they come in no order of it -- the rows are listed by something else --
    and "" where they cannot all be placed. Labels that cannot be placed --
    "January", "Week 1", "FY25 Q1" -- say nothing of the kind."""
    axes = _period_axes(rows)
    for column in axes:
        keys = [_temporal_sort_value(row.get(column)) for row in rows]
        if all(key is not None for key in keys):
            return "in" if keys == sorted(keys) or keys == sorted(keys, reverse=True) else "out"
    # Month names with no year -- "January", "mars" -- in the calendar's order, the reverse of it, or one month after
    # another from any month of it are in time; in any other they may be a fiscal year's, and say nothing.
    for column in axes:
        months = [parse_month_name(row.get(column)) for row in rows]
        if all(month is not None for month in months) and (
                months == sorted(months) or months == sorted(months, reverse=True)
                or _months_in_a_row(months) or _months_in_a_row(months[::-1])):
            return "in"
    return ""


def _months_in_a_row(months: list[int]) -> bool:
    """Month numbers that each follow the one before across a new year: October to March is 10, 11, 12, 1, 2, 3."""
    return all((later - earlier) % 12 == 1 for earlier, later in zip(months, months[1:]))


def _periods_listed_out_of_time(rows: list[dict]) -> bool:
    return _period_order(rows) == "out"


def _is_a_sequence(rows: list[dict], column: str) -> bool:
    """A column of whole numbers that runs in equal steps -- 1, 2, 3 ... in
    whatever order they stand, MONTHNUM 4..9, SORT_ORDER 10..60, PERIOD_KEY
    301..306, or one number throughout -- is a sort key (SORT_ORDER, MONTHNUM,
    RN), no figure; so are the keys of months or weeks that run on across a new
    year (202411, 202412, 202501). Two rows are a run only as 1, 2 or 0, 1."""
    try:
        values = [_to_float(row.get(column)) for row in rows]
        if len(rows) < 2 or any(value is None or value != int(value) for value in values):
            return False
        ordered = sorted(int(value) for value in values)
    except (OverflowError, ValueError):  # NaN or an infinity is no whole number
        return False
    if len(ordered) == 2:
        return ordered in ([1, 2], [0, 1])
    if len({later - earlier for earlier, later in zip(ordered, ordered[1:])}) == 1:
        return True
    if all(190001 <= value <= 219912 and 1 <= value % 100 <= 12 for value in ordered):
        # A month's key, 202412 then 202501: months in equal steps across a new year.
        months = [value // 100 * 12 + value % 100 for value in ordered]
        return len({later - earlier for earlier, later in zip(months, months[1:])}) == 1
    if all(19001 <= value <= 21994 and 1 <= value % 10 <= 4 for value in ordered):
        # A quarter's key, 20243 then 20244, 20251: quarters in equal steps across a new year.
        quarters = [value // 10 * 4 + value % 10 for value in ordered]
        if len({later - earlier for earlier, later in zip(quarters, quarters[1:])}) == 1:
            return True
    return _weeks_in_a_row(ordered)


def _weeks_in_a_row(keys: list[int]) -> bool:
    """Week keys, YYYYWW, that each follow the one before: 202451, 202452, 202501 -- or 202453 where the year has it."""
    return all(190001 <= key <= 219953 for key in keys) and all(
        later - earlier == 1 or (earlier % 100 in (52, 53) and later == (earlier // 100 + 1) * 100 + 1)
        for earlier, later in zip(keys, keys[1:]))


# What a statement takes a date's parts with, as the parser names them (YEAR, DATEPART(week, d) and EXTRACT are
# Extract, DAYOFWEEK is DayOfWeek, FORMAT(d, 'yyyyMM') is TimeToStr) and, where the parser knows no such function, by
# the name it is called. A date that is one -- DATE_TRUNC, DATEFROMPARTS, EOMONTH -- is no number to list rows by.
_DATE_PART_NODES = ("Year", "Month", "Day", "Quarter", "Week", "WeekOfYear", "DayOfWeek", "DayOfWeekIso", "DayOfMonth",
                    "DayOfYear", "Extract", "YearOfWeek", "YearOfWeekIso", "TimeToStr", "ToChar")
_DATE_PART_NAMES = frozenset({"yearweek", "isoweek", "iso_week", "weekiso", "datepart", "datename", "date_part",
                              "date_format", "strftime"})


def _reads_a_date_part(expression) -> bool:
    """Whether an expression takes a part of a date anywhere in it -- YEAR(d), DATEPART(week, d), YEAR(d) * 100 + MONTH(d)."""
    from sqlglot import exp as sg_exp

    parts = tuple(getattr(sg_exp, name) for name in _DATE_PART_NODES if hasattr(sg_exp, name))
    return expression.find(*parts) is not None or any(
        str(call.name).lower() in _DATE_PART_NAMES for call in expression.find_all(sg_exp.Anonymous))


def _is_the_first_or_last_of_a_date_part(expression) -> bool:
    """Whether an expression only takes the first or last of parts of dates -- MIN(YEAR(d) * 100 + MONTH(d)),
    MIN(YEAR(d) * 12 + MONTH(d) - 1), MAX(DATEPART(week, d)) -- which is a key. Not a figure it reads beside one --
    MAX(CASE WHEN MONTH(d) = 12 THEN amount END) -- nor a length of time between two dates --
    MAX(YEAR(GETDATE()) - YEAR(birth))."""
    from sqlglot import exp as sg_exp

    calls = list(expression.find_all(sg_exp.AggFunc))
    if not calls or any(_reads_a_date_part(sub.this) and _reads_a_date_part(sub.expression)
                        for sub in expression.find_all(sg_exp.Sub)):
        return False
    parts = tuple(getattr(sg_exp, name) for name in _DATE_PART_NODES if hasattr(sg_exp, name))

    def inside_a_part(node) -> bool:
        parent = node.parent
        while parent is not None:
            if isinstance(parent, parts) or (isinstance(parent, sg_exp.Anonymous)
                                             and str(parent.name).lower() in _DATE_PART_NAMES):
                return True
            parent = parent.parent
        return False

    return all(isinstance(call, (sg_exp.Min, sg_exp.Max)) and _kind_of(call.this) == "date"
               and all(inside_a_part(column) for column in call.this.find_all(sg_exp.Column)) for call in calls)


def _kind_of(expression) -> str:
    """What a select list's expression is: "aggregate" where it adds rows up (SUM, COUNT, AVG, MAX), "date" where it
    takes a part of a date and adds up nothing -- YEAR(d), DATEPART(week, d), YEAR(d) * 100 + MONTH(d) -- else
    "other": a column."""
    from sqlglot import exp as sg_exp

    if _is_the_first_or_last_of_a_date_part(expression):
        return "date"
    if isinstance(expression, sg_exp.Window) and isinstance(expression.this, (sg_exp.Rank, sg_exp.DenseRank, sg_exp.RowNumber)):
        ordered = expression.args.get("order")
        if ordered is not None:
            return _kind_of(ordered)  # RANK() OVER (ORDER BY YEAR(d)) counts the periods, not what they add up to
    if expression.find(sg_exp.AggFunc) is not None:
        return "aggregate"
    return "date" if _reads_a_date_part(expression) else "other"


@functools.lru_cache(maxsize=64)
def _select_definitions(sql: str, db_type: str = "azure_sql") -> dict[str, tuple[str, ...]]:
    """For each name a statement's select lists give a column -- lower case, in every scope that does -- what that
    column is (_kind_of): {"y": ("date",), "revenue": ("aggregate",)}. Read, never changed."""
    tree = _parsed_sql(sql, db_type)
    return _definitions_of(tree) if tree is not None else {}


def _definitions_of(tree) -> dict[str, tuple[str, ...]]:
    from sqlglot import exp as sg_exp

    found: dict[str, list[str]] = {}
    for alias in tree.find_all(sg_exp.Alias):
        found.setdefault(str(alias.alias).lower(), []).append(_kind_of(alias.this))
    return {name: tuple(kinds) for name, kinds in found.items()}


def _named_by(key: str, rows: list[dict]) -> str:
    """The name an ORDER BY key goes by, in lower case: a qualified or quoted column's own, or -- for a position, ORDER
    BY 3 -- the name of the result's column in that place."""
    if key.isascii() and key.isdigit() and rows and 0 < int(key) <= len(rows[0]):
        key = str(list(rows[0].keys())[int(key) - 1])
    return re.split(r"\.", key)[-1].strip().strip('"[]`').lower()


def _key_is_a_figure(key: str, rows: list[dict], asked: bool = False, definitions=None) -> bool:
    """Whether an ORDER BY key is a figure of the result -- an aggregate, a
    numeric column of it, or the position of one -- and no period. The first
    or last date of a period is a date: MIN(ORDER_DATE) lists months in time.
    A column the statement defines as a part of a date -- YEAR(d) AS Y,
    YEAR(d) * 100 + MONTH(d) AS SORT_KEY -- is no figure whatever it is
    called (``definitions``: what _select_definitions reads). A column that
    runs in equal steps is a sort key and no figure, unless the question asks
    for the periods ranked: "top 5 months by number of orders" counts them
    5, 4, 3, 2, 1."""
    if not key:
        return False
    calls = _aggregate_calls(key)
    if definitions is not None:
        defined = (definitions() if callable(definitions) else definitions).get(_named_by(key, rows), ())
        if defined and all(kind == "date" for kind in defined):
            return False
    if calls:
        if any(name not in ("min", "max") for name, _ in calls):
            return True  # a sum, a count, an average: a figure
        # MIN(x) or MAX(x) is the first or last date of a period -- or the
        # smallest or largest amount in it. A date by its name (ORDERDATE,
        # TXN_TS, POSTED_ON, created_at) or where it is wrapped in a date
        # function (MONTH(MIN(order_date))); else said by the rows: periods
        # that can be placed and run in no order of time are listed by the
        # amount, and labels that cannot be placed say nothing of the kind.
        if any(_names_a_date(argument) for _, argument in calls):
            return False
        return _periods_listed_out_of_time(rows)
    columns = list(rows[0].keys())
    figures = set(_numeric_cols(rows)) - set(period_columns(rows, _numeric_cols(rows)))
    sort_keys = {column for column in figures if _is_a_sequence(rows, column)}
    if figures - sort_keys and not asked:
        figures -= sort_keys  # a count of 3, 2, 1 that is all the result measures is what it ranks by
    if key.isascii() and key.isdigit():
        position = int(key) - 1
        return 0 <= position < len(columns) and columns[position] in figures
    name = re.split(r"\.", key)[-1].strip().strip('"[]`').lower()
    return any(column.lower() == name for column in figures)


def _ordered_by_a_figure(sql: str, rows: list[dict], asked: bool = False, db_type: str = "azure_sql") -> bool:
    """Does the SQL list its rows by a figure rather than by their periods?

    "Top 3 months by revenue" and "which months had the highest revenue"
    come back ORDER BY the revenue: a ranking of periods. Put back in time
    order and read as a series, the card said "June 2025 closed at 700" and
    "Revenue trended down 12.5% from January 2025 to June 2025" of three
    months picked for their revenue. Only the outermost ORDER BY counts -- a
    window's or a subquery's orders nothing the reader sees -- and only its
    first key: an aggregate, a numeric column of the result, or the
    position of one.
    """
    text = str(sql or "")
    if not rows or not text.strip():
        return False
    return _key_is_a_figure(_outer_order_key(text), rows, asked, lambda: _select_definitions(text, db_type))


def _outer_order_key(sql: str) -> str:
    """The first key of the statement's outermost ORDER BY, or ""."""
    clause = ""
    for match in re.finditer(r"\bORDER\s+BY\b", sql, re.IGNORECASE):
        before = re.sub(r"'(?:[^']|'')*'", "", sql[:match.start()])
        if before.count("(") == before.count(")"):
            clause = sql[match.end():]
    if not clause:
        return ""
    return _first_order_key(re.split(r"\b(?:LIMIT|OFFSET|FETCH)\b|;", clause, maxsplit=1, flags=re.IGNORECASE)[0])


def _ordered_by_the_measure(sql: str, rows: list[dict], db_type: str = "azure_sql") -> bool:
    """Does the outermost ORDER BY list the rows by what they measure -- SUM(x), a COUNT, a column the statement
    defines as one or, where it defines none, the result's first measure, or the position of a column -- and not by a
    column that only holds a number: a year, a key?"""
    key = _outer_order_key(str(sql or ""))
    if key.isdigit() or _aggregate_calls(key):
        return True
    name = re.split(r"\.", key)[-1].strip().strip('"[]`').lower()
    defined = _select_definitions(str(sql or ""), db_type).get(name, ())
    if defined:
        return "aggregate" in defined
    numeric_cols = _numeric_cols(rows)
    measures, _labels, _periods = _measure_and_label_cols(rows, numeric_cols, _text_cols(rows, numeric_cols))
    return measures[0].lower() == name


def _parsed_sql(sql: str, db_type: str = "azure_sql"):
    """The statement's tree, read the way the validator reads it -- and, where
    that does not parse it, as the other dialects do -- or None."""
    import sqlglot

    from core.validator import _DIALECT, normalize_generated_sql

    own = _DIALECT.get(db_type, "snowflake")
    for dialect in dict.fromkeys((own, "tsql", "snowflake", "oracle")):
        try:
            return sqlglot.parse_one(normalize_generated_sql(sql, db_type), read=dialect)
        except Exception:
            continue
    return None


def _a_selects_own_key(select, key) -> str:
    """The text of an ORDER BY key of a select. A position -- ORDER BY 3 -- is the select's own column in that place,
    named as it is: not the column the whole result has there, which a subquery's order has nothing to do with."""
    from sqlglot import exp as sg_exp

    text = key.sql()
    if text.isascii() and text.isdigit() and 0 < int(text) <= len(select.expressions) and not any(
            isinstance(column, sg_exp.Star) for column in select.expressions):
        column = select.expressions[int(text) - 1]
        if isinstance(column, (sg_exp.Alias, sg_exp.Column)):
            return column.alias_or_name
        return column.sql()  # COUNT(*), SUM(x): the call is what it orders by
    return text


def _cut_by_a_figure(sql: str, rows: list[dict], db_type: str = "azure_sql", asked: bool = True) -> bool:
    """Is the result cut to its top periods by a figure inside the statement --
    a subquery's TOP 3 months ... ORDER BY SUM(x) DESC, or a ROW_NUMBER() OVER
    (ORDER BY SUM(x) DESC) over the months kept to its first three -- whatever
    the statement outside says? Only a cut of the periods themselves ranks
    them: the top 5 customers' monthly totals, listed in time, are a series,
    and cut by the periods' own order ("the last 12 months") it is not a
    ranking either."""
    if not rows or not str(sql or "").strip():
        return False
    tree = _parsed_sql(sql, db_type)
    periods = {column.lower() for column in _period_axes(rows)}
    if tree is None or not periods:
        return False
    from sqlglot import exp as sg_exp

    ctes = {str(cte.alias).lower(): cte.this for cte in tree.find_all(sg_exp.CTE)
            if isinstance(cte.this, sg_exp.Select)}

    def outputs(select, depth: int = 0) -> set[str]:
        """What a SELECT hands on: its columns, and a star's, which are those of
        the CTE or subquery it reads."""
        names = {str(name).lower() for name in select.named_selects}
        if "*" not in names or depth > 4:
            return names
        names.discard("*")
        for source in (select.args.get("from_") or select.args.get("from"), *(select.args.get("joins") or [])):
            node = source.this if source is not None else None
            if isinstance(node, sg_exp.Table) and str(node.name).lower() in ctes:
                names |= outputs(ctes[str(node.name).lower()], depth + 1)
            elif isinstance(node, sg_exp.Subquery) and isinstance(node.this, sg_exp.Select):
                names |= outputs(node.this, depth + 1)
        return names

    ranking_windows = (sg_exp.RowNumber, sg_exp.Rank, sg_exp.DenseRank)
    defined = _definitions_of(tree)
    for select in tree.find_all(sg_exp.Select):
        if not outputs(select) & periods:
            continue  # what this cut ranks is not the periods
        keys = []
        order = select.args.get("order")
        if (select.args.get("limit") or select.args.get("fetch")) and order is not None and order.expressions:
            keys.append(_a_selects_own_key(select, order.expressions[0].this))
        for window in select.find_all(sg_exp.Window):
            ordered = window.args.get("order")
            if (window.find_ancestor(sg_exp.Select) is select and isinstance(window.this, ranking_windows)
                    and not window.args.get("partition_by")  # ranks inside each group, never the periods
                    and ordered is not None and ordered.expressions):
                keys.append(ordered.expressions[0].this.sql())
        # A key the statement defines as what rows add up to is a figure whether the question asks for the periods
        # ranked or not: a count of 3, 2, 1 is a count.
        if any(_key_is_a_figure(key, rows, asked or "aggregate" in defined.get(_named_by(key, rows), ()), defined)
               for key in keys):
            return True
    return False


def _listed_by_value(rows: list[dict]) -> bool:
    """With no SQL to read: are the rows listed by their figure -- periods
    that can be placed in time but are not in its order, and figures in the
    order of their values? Where no period can be placed in time nothing says
    the rows are not a series."""
    placed = False
    for column in _period_axes(rows):
        keys = [_temporal_sort_value(row.get(column)) for row in rows]
        if any(key is None for key in keys):
            continue
        if keys == sorted(keys) or keys == sorted(keys, reverse=True):
            return False
        placed = True
    numeric_cols = _numeric_cols(rows)
    figures = [column for column in numeric_cols if column not in set(period_columns(rows, numeric_cols))]
    if not placed or not figures:
        return False
    values = [_to_float(row.get(figures[0])) for row in rows]
    if any(value is None for value in values):
        return False
    return values == sorted(values) or values == sorted(values, reverse=True)


def _ranked_by_a_figure(question: str, sql: str, rows: list[dict], db_type: str = "azure_sql") -> bool:
    """Are the periods of these rows a ranking -- listed by a figure, or the
    top of one -- and no series? Read from the SQL where there is one; else
    from the rows themselves (_listed_by_value).

    Periods that can be placed in time and come in its order, forwards or
    backwards, are listed by their period whatever the SQL sorted them by --
    a sort key that runs in unequal steps, a figure that happens to rise with
    the months -- unless the question asks for them ranked."""
    if not rows:
        return False
    if not str(sql or "").strip():
        return _listed_by_value(rows)
    ranks_periods = _asks_to_rank_periods(question)
    # The ask makes a sort key a figure only where the periods are not in time order, or are two: "monthly revenue
    # in our best year" lists its months in time, counted 1, 2, 3 by a MONTH_NUM.
    asked = ranks_periods and (len(rows) < 3 or _period_order(rows) != "in")
    if _ordered_by_a_figure(sql, rows, asked, db_type):
        # What the question asks to rank stands where the rows are out of time or listed by what they measure: months
        # in time listed by a year, a column that only holds a number, are a series whatever year is asked about.
        return (ranks_periods and (asked or _ordered_by_the_measure(sql, rows, db_type))) or _period_order(rows) != "in"
    return _asks_for_ranked_periods(question) and _cut_by_a_figure(sql, rows, db_type, asked)


def _question_texts(question: str) -> list[str]:
    """The question as written and as canonicalised, which reads the French."""
    from core.i18n import get_active_language
    from core.question_normalizer import canonical_question

    try:
        return [question or "", canonical_question(question or "", get_active_language())]
    except Exception as exc:
        log.warning("The question was not canonicalised to read its ranked periods: %s", exc)
        return [question or ""]


def _asks_for_ranked_periods(question: str) -> bool:
    """Does the question ask for a ranking -- "top 3 months by revenue",
    "which months had the highest revenue", "best months", "les 3 meilleurs
    mois par revenu"? Of anything: it is periods only where a subquery or a
    window cuts the months themselves (_cut_by_a_figure)."""
    from core.analytical_intent import asks_for_ranking
    from core.contextual_dates import ranked_period_grain

    return any(detect_top_n_intent(text) is not None or asks_for_ranking(text) or bool(ranked_period_grain(text))
               for text in _question_texts(question))


_PERIOD_NOUNS = r"(?:day|week|month|quarter|year)s?"
_NOT_A_PREPOSITION = r"(?!(?:by|per|each|in|for|of|and|or|with|on|at|to|from|during|over)\b)"
_RANKS_PERIODS_RE = re.compile(
    r"\b(?:top|bottom|best|worst|highest|lowest|biggest|largest|smallest|busiest|slowest|fastest|strongest|weakest)"
    r"(?:-\w+)?\s+"
    rf"(?:{_NOT_A_PREPOSITION}[\w'-]+\s+)?{_PERIOD_NOUNS}\b"
    rf"|\brank(?:ed|ing|s)?\s+(?:of\s+)?(?:the\s+|our\s+|all\s+)?{_PERIOD_NOUNS}\b"
    rf"|\b{_PERIOD_NOUNS}\s+(?:with|that\s+had|having)\s+the\s+"
    r"(?:most|highest|lowest|least|fewest|best|worst|biggest|largest|smallest)\b", re.I)


def _asks_to_rank_periods(question: str) -> bool:
    """Does the question rank the periods themselves -- "top 3 months by
    revenue", "which 3 months had the most orders", "the worst week", "rank
    months by sales" -- and not some other member across them: "monthly
    revenue for the top 5 customers" and "our best customer by month" list
    months in time."""
    from core.contextual_dates import ranked_period_grain

    return any(_RANKS_PERIODS_RE.search(text) or bool(ranked_period_grain(text)) for text in _question_texts(question))


def _chronological_analysis_rows(rows: list[dict]) -> list[dict]:
    """Sort a copy for temporal analysis while preserving table display order."""
    copied = list(rows)
    if len(copied) < 2:
        return copied
    numeric_cols = _numeric_cols(copied)
    # An integer year sorts chronologically only if it is looked at. Left in the
    # numeric pool it was never a sort candidate, so a year series arrived in
    # whatever order the warehouse returned it and the trend was computed across
    # it anyway.
    _measures, label_cols, _periods = _measure_and_label_cols(
        copied, numeric_cols, _text_cols(copied, numeric_cols))
    for column in label_cols:
        values = [str(row.get(column, "")) for row in copied]
        # A period key (202501) is a period as its ISO spelling is.
        if not _a_period_axis(column, [row.get(column) for row in copied], values):
            continue
        keys = [_temporal_sort_value(row.get(column)) for row in copied]
        if all(key is not None for key in keys):
            return [
                row
                for _key, _index, row in sorted(
                    zip(keys, range(len(copied)), copied),
                    key=lambda item: (item[0], item[1]),
                )
            ]
    return copied


def in_time_order(labels: list, values: list) -> tuple[list, list]:
    """A series' labels and values, earliest period first: the order the rows
    came in is the SQL's, and a series listed newest first said "trended down"
    of figures that rose. Left as they came where a label cannot be placed in
    time, or is one a day reads as a month ("Sep-30")."""
    if len(labels) < 2 or any(_MONTH_OR_DAY_RE.fullmatch(str(label).strip()) for label in labels):
        return labels, values
    keys = [_temporal_sort_value(label) for label in labels]
    if any(key is None for key in keys) or keys == sorted(keys):
        return labels, values
    order = sorted(range(len(labels)), key=lambda index: (keys[index], index))
    return [labels[index] for index in order], [values[index] for index in order]


def _std_dev(values: list[float]) -> float | None:
    """The sample standard deviation of three values or more, none of them NaN or infinite: of those it is no number."""
    if len(values) < 3 or not all(math.isfinite(value) for value in values):
        return None
    return round(stdev(values), 2)


def _safe_pct_change(first: float, last: float) -> float | None:
    if first == 0:
        return None
    return ((last - first) / abs(first)) * 100.0


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _to_float_z(value: Any) -> float:
    """Like _to_float but returns 0.0 for None/unparseable.

    Use this instead of ``_to_float(v) or 0.0`` because the ``or`` idiom
    silently zeroes legitimate negative values (e.g. -500.0 is falsy).
    """
    v = _to_float(value)
    return v if v is not None else 0.0


def _display_label(column: str) -> str:
    """The business name for a column, for prose a reader sees.

    The implementation moved to core.schema_enrichment.display_label, next to
    the vocabulary it consults, because six other producers outside this module
    need the same label and could not reach it here: the semantic plan's
    dimension chips, the chart's axis titles, the KPI caption, the "what you
    can ask" list, and the measure clarification buttons all printed the
    warehouse's spelling because the only copy of this lived in the narrative
    layer.

    Kept as a name because every call site in this module reads better for it.
    """
    from core.schema_enrichment import display_label

    return display_label(column)


def _find_header_by_norm(headers: list[str], norm: str) -> str:
    if not norm:
        return ""
    for header in headers:
        if _normalise_key(header) == norm:
            return header
    for header in headers:
        h_norm = _normalise_key(header)
        if h_norm.endswith(norm) or norm.endswith(h_norm):
            return header
    return ""


_MATCHED_ROWS_HEADER_KEYS = {"matchedrows", "rowcount", "matchcount", "matchedrecords"}


def _find_matched_rows_header(headers: list[str]) -> str:
    """The header name of a diagnostic match-count column (e.g. MatchedRows),
    if this row shape carries one. Shared by detect_null_metric_issue and
    detect_zero_match_result -- the two checks are mutually exclusive by
    construction (matched_rows > 0 vs <= 0), never both true for the same row."""
    return next(
        (h for h in headers if _normalise_key(h) in _MATCHED_ROWS_HEADER_KEYS),
        "",
    )


def result_diagnostic_headers(rows: list[dict]) -> list[str]:
    """Return support columns used to validate aggregate result quality.

    These values remain available to answer/confidence logic but are not
    business measures and therefore should not be rendered as table columns
    or KPI values.
    """
    if len(rows) != 1 or not rows[0]:
        return []
    output: list[str] = []
    for header in rows[0].keys():
        norm = _normalise_key(header)
        if norm in _MATCHED_ROWS_HEADER_KEYS or (
            norm.startswith("nonnull") and norm.endswith("rows")
        ):
            output.append(header)
    return output


def visible_result_rows(rows: list[dict]) -> list[dict]:
    hidden = set(result_diagnostic_headers(rows))
    if not hidden:
        return list(rows)
    return [
        {header: value for header, value in row.items() if header not in hidden}
        for row in rows
    ]


def _result_diagnostics(rows: list[dict]) -> dict[str, Any]:
    headers = result_diagnostic_headers(rows)
    if not headers:
        return {}
    row = rows[0]
    matched = _find_matched_rows_header(headers)
    return {
        "matched_rows": int(_to_float(row.get(matched)) or 0) if matched else None,
        "non_null_counts": {
            header: int(_to_float(row.get(header)) or 0)
            for header in headers
            if _normalise_key(header).startswith("nonnull")
        },
        "hidden_columns": headers,
    }


def _build_kpi_payload(
    rows: list[dict],
    column_formats: dict[str, str],
    display_formats: dict[str, dict],
    *,
    zero_match: bool,
    null_issue: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if zero_match or len(rows) != 1 or len(rows[0]) != 1:
        return None
    column = next(iter(rows[0]))
    value = rows[0].get(column)
    scalar_missing = _is_missing_scalar(value)
    fmt = column_formats.get(column, "number")
    return {
        "label": _display_label(column),
        # As the table's cell is sent: a period as the day it starts, a key as
        # its digits. "2,021" is no date a browser can name -- it read it as
        # the 21st of February 2001.
        "value": _period_cell(value) if fmt == "date" else _text_cell(value) if fmt == "text" else _safe_cell(value),
        "format": fmt,
        "display_format": dict(display_formats.get(column) or {}),
        "state": "missing" if (null_issue or scalar_missing) else "ready",
        "note": (
            _t("ui.kpi.note.null_metric")
            if null_issue
            else _t("ui.kpi.note.no_data")
            if scalar_missing
            else _t("ui.kpi.note.single_value")
        ),
    }


_COMPARISON_PREFIXES = (
    ("CURRENT_", "PREVIOUS_"),
    ("CURRENT_", "PRIOR_"),
    ("THIS_", "LAST_"),
)
_PCT_CHANGE_COLUMNS = ("PCT_CHANGE", "PERCENT_CHANGE", "PCT_DIFF", "CHANGE_PCT")


def _numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def _is_missing_scalar(value: Any) -> bool:
    """Return True when a scalar database result carries no usable value."""
    if value is None:
        return True
    if isinstance(value, float) and not math.isfinite(value):
        return True
    return False


def _single_missing_scalar(rows: list[dict]) -> tuple[str, Any] | None:
    if len(rows) != 1 or len(rows[0]) != 1:
        return None
    column = next(iter(rows[0]))
    value = rows[0].get(column)
    return (column, value) if _is_missing_scalar(value) else None


def _missing_scalar_copy(column: str, question: str) -> dict[str, str]:
    """Build calm, business-facing copy for successful NULL aggregates.

    SQL aggregates such as SUM() return one physical row containing NULL when
    the requested period has no matching facts.  That is an empty analytical
    result, not a value called ``None`` and not a query failure.
    """
    metric = _display_label(column)
    metric_lower = metric[:1].lower() + metric[1:] if metric else "metric"
    temporal = bool(re.search(
        r"\b(today|yesterday|tomorrow|day|week|month|quarter|q[1-4]|year|"
        r"fiscal|calendar|period|date|latest|last|current|previous|prior)\b",
        str(question or ""),
        re.IGNORECASE,
    ))
    target = _t("answer.target_period" if temporal else "answer.target_filters")
    return {
        "headline": _t("answer.no_metric_headline",
                       metric=metric_lower, target=target),
        "short_value": _t("answer.no_data"),
        "comparison": _t("answer.no_metric_comparison"),
        "scope_badge": _t("answer.no_data"),
        "scope_note": _t("answer.no_metric_note",
                         metric=metric_lower, target=target),
    }


def _period_comparison_from_rows(rows: list[dict]) -> dict | None:
    """Recognise a single-row current-vs-previous comparison.

    The SQL for a period comparison returns ONE wide row of paired columns
    (CURRENT_x / PREVIOUS_x, plus a difference and percentage), not a series.
    Narrating it as a trend produced "trended flat 0.0% from 2026-03 to
    2026-03" over data that actually read 2026-03 $500 vs 2026-02 $400, +25%.

    Column-naming convention only -- no tenant vocabulary. Requires a numeric
    pair to call it a comparison; a non-numeric pair (e.g. CURRENT_MONTH /
    PREVIOUS_MONTH) supplies the period labels instead.
    """
    if not rows or len(rows) != 1:
        return None
    row = rows[0]
    if not isinstance(row, dict):
        return None
    upper = {str(k).upper(): k for k in row}

    numeric_pair = None
    label_pair = None
    for cur_prefix, prev_prefix in _COMPARISON_PREFIXES:
        for key_u, key in upper.items():
            if not key_u.startswith(cur_prefix):
                continue
            suffix = key_u[len(cur_prefix):]
            prev_u = f"{prev_prefix}{suffix}"
            if prev_u not in upper:
                continue
            prev_key = upper[prev_u]
            cur_val = _numeric_or_none(row.get(key))
            prev_val = _numeric_or_none(row.get(prev_key))
            is_period_label = bool(re.search(
                r"(?:^|_)(?:DATE|DT|DAY|WEEK|MONTH|QUARTER|YEAR|PERIOD|PRD|YYYYMM|YYYYMMDD)(?:_|$)",
                suffix,
            ))
            if is_period_label and label_pair is None:
                label_pair = (row.get(key), row.get(prev_key))
            elif cur_val is not None and prev_val is not None:
                if numeric_pair is None:
                    numeric_pair = (key, prev_key, cur_val, prev_val)
            elif label_pair is None:
                label_pair = (row.get(key), row.get(prev_key))

    if not numeric_pair:
        return None
    measure_col, previous_col, current_value, previous_value = numeric_pair

    pct = None
    for candidate in _PCT_CHANGE_COLUMNS:
        if candidate in upper:
            pct = _numeric_or_none(row.get(upper[candidate]))
            if pct is not None:
                break
    if pct is None and previous_value:
        pct = (current_value - previous_value) * 100.0 / previous_value

    current_period, previous_period = ("the current period", "the previous period")
    if label_pair and label_pair[0] is not None and label_pair[1] is not None:
        current_period, previous_period = str(label_pair[0]), str(label_pair[1])

    return {
        "measure_column": measure_col,
        "previous_column": previous_col,
        "current_value": row.get(measure_col),
        "previous_value": row.get(previous_col),
        "current_period": current_period,
        "previous_period": previous_period,
        "pct_change": pct,
    }


def _safe_category_label(label: Any, label_column: str) -> str:
    """The category's own name, or "" when the value redactor replaced it.

    core.insight._display_label is imported under an alias here and in every
    other caller: this module already defines a one-argument _display_label
    column prettifier used in five places, and an unaliased import would shadow
    it and raise TypeError on a path with no protection around it.
    """
    from core.insight import _display_label as _redact_value_label

    text = _redact_value_label(str(label or ""), label_column)
    return "" if not text or text == "redacted segment" else text


def _measure_prefix(column: str, label: str) -> str:
    """"NET_AMOUNT_2025" with the label "2025" -> "NET_AMOUNT".

    Empty when the column IS the period label, which is what makes the caller
    say "Total" rather than name a measure it cannot see.
    """
    from core.multi_period import period_alias_suffix

    suffix = period_alias_suffix(label)
    name = str(column or "")
    if suffix and name.upper().endswith("_" + suffix):
        return name[: -(len(suffix) + 1)]
    return "" if name.upper() == suffix else name


def _period_pair_facts(rows: list[dict], plan_labels: list[str] | None) -> dict | None:
    """The arithmetic behind every period-comparison sentence, or None.

    None means "this is not a named-period comparison" and every caller falls
    straight through to the behaviour it had before. The gate is strict on
    purpose: at least two of the plan's OWN period aliases must be present as
    result columns, and every label must parse as a real calendar period.

    Column matching is delegated to core/multi_period.py rather than repeated
    here. The hint, the post-processor and this all have to agree on which
    columns are the periods; two matchers would drift, and the loose one would
    start reading ERP columns like P_QTY as a period.
    """
    if not rows or not plan_labels or len(plan_labels) < 2:
        return None
    try:
        from core.multi_period import (
            period_alias_suffix, period_columns_by_alias, period_parts,
        )
    except Exception:
        return None

    labels = [str(label) for label in plan_labels]
    if not all(period_parts(label) for label in labels):
        return None
    aliases = [period_alias_suffix(label) for label in labels]
    found = period_columns_by_alias(rows, aliases)
    if len(found) < 2:
        return None

    present = [(label, found[alias]) for label, alias in zip(labels, aliases)
               if alias in found]
    (oldest_label, oldest_col), (newest_label, newest_col) = present[0], present[-1]

    numeric_cols = _numeric_cols(rows)
    text_cols = _text_cols(rows, numeric_cols)
    numeric_cols, text_cols, _period_cols = _measure_and_label_cols(
        rows, numeric_cols, text_cols)
    label_col = text_cols[0] if text_cols else ""

    movers: list[dict] = []
    oldest_total = newest_total = 0.0
    for row in rows:
        before, after = _to_float(row.get(oldest_col)), _to_float(row.get(newest_col))
        if before is None or after is None:
            continue          # masked or missing; excluded from every total
        oldest_total += before
        newest_total += after
        movers.append({
            "label": str(row.get(label_col, "")) if label_col else "",
            "change": after - before,
            "pct": ((after - before) * 100.0 / abs(before)) if before else None,
        })
    if not movers:
        return None

    net = sum(mover["change"] for mover in movers)
    gross = sum(abs(mover["change"]) for mover in movers)
    grew = sum(1 for mover in movers if mover["change"] > 0)
    shrank = sum(1 for mover in movers if mover["change"] < 0)
    risers = [mover for mover in movers if mover["change"] > 0]
    fallers = [mover for mover in movers if mover["change"] < 0]
    top_riser = max(risers, key=lambda m: m["change"]) if risers else None
    top_faller = min(fallers, key=lambda m: m["change"]) if fallers else None

    return {
        "labels": [label for label, _ in present],
        "oldest_label": oldest_label, "newest_label": newest_label,
        "oldest_column": oldest_col, "newest_column": newest_col,
        "label_column": label_col,
        "oldest_total": oldest_total, "newest_total": newest_total,
        "total_pct": (net * 100.0 / abs(oldest_total)) if oldest_total else None,
        "net": net,
        # Below this the gains and losses have cancelled and a share of the net
        # is not a number worth printing -- the same floor annotate_period_change
        # applies to SHARE_OF_CHANGE_PCT.
        "share_holds": gross > 0 and abs(net) >= 0.05 * gross,
        "row_count": len(movers), "grew": grew, "shrank": shrank,
        "flat": len(movers) - grew - shrank,
        "top_riser": top_riser, "top_faller": top_faller,
    }


def _period_comparison_summary(
    rows: list[dict],
    plan_labels: list[str] | None,
    column_formats: dict | None = None,
    display_formats: dict | None = None,
) -> str:
    """The note under the card for a named-period comparison, or "".

    "Across 12 revenue categories, 7 grew and 5 shrank between 2024 and 2025.
    Pumps added the most (+400,000, 46% of the net change); Valves fell the
    most (-180,000)."

    Category labels go through core.insight's value redactor, imported under an
    alias: this module already defines a one-argument _display_label column
    prettifier used in five places, and an unaliased import would shadow it and
    raise TypeError on an unprotected path.
    """
    facts = _period_pair_facts(rows, plan_labels)
    if not facts:
        return ""
    column_formats = column_formats or {}
    display_formats = display_formats or {}

    def money(value: float) -> str:
        return _format_display_value(
            value,
            column_formats.get(facts["newest_column"]),
            display_formats.get(facts["newest_column"]),
        )

    def signed(value: float) -> str:
        return ("+" if value > 0 else "") + money(value)

    def named(mover: dict) -> str:
        return _safe_category_label(mover["label"], facts["label_column"])

    # Counted in the reader's language: "across 5 item groups", "sur 5
    # groupes d'articles" -- English plural morphology on the column's name
    # put English inside the French sentence (core.i18n.count_noun).
    counted = _count_noun(
        _display_label(facts["label_column"]) if facts["label_column"] else "",
        facts["row_count"], "answer.groups")
    opening = _t(
        "answer.period.opening", count=facts["row_count"], label=counted,
        grew=_t_plural("answer.period.grew", facts["grew"]),
        shrank=_t_plural("answer.period.shrank", facts["shrank"]),
        old=facts["oldest_label"], new=facts["newest_label"])

    clauses: list[str] = []
    riser, faller = facts["top_riser"], facts["top_faller"]
    if riser:
        share = ""
        if facts["share_holds"] and facts["net"]:
            share = _t("answer.period.share",
                       pct=f"{abs(riser['change'] * 100.0 / facts['net']):.0f}")
        who = named(riser)
        clauses.append(
            _t("answer.period.added_most", who=who,
               change=signed(riser["change"]), share=share) if who
            else _t("answer.period.largest_increase",
                    change=signed(riser["change"]), share=share)
        )
    if faller:
        who = named(faller)
        clauses.append(
            _t("answer.period.fell_most", who=who,
               change=signed(faller["change"])) if who
            else _t("answer.period.largest_decrease",
                    change=signed(faller["change"]))
        )
    if not clauses:
        return _t("answer.period.none_moved", opening=opening)
    detail = "; ".join(clauses)
    return f"{opening} {detail[0].upper()}{detail[1:]}."


def detect_zero_match_result(rows: list[dict]) -> bool:
    """
    True for a single-row diagnostic aggregate whose match-count column is
    itself zero (or negative) -- e.g. [{"MatchedRows": 0, "Revenue": 0}].

    A query like `SELECT COUNT(*) AS MatchedRows, SUM(x) AS Total FROM ...
    WHERE <date filter>` always returns exactly one physical row even when
    nothing matched, so the ordinary "not rows" empty-result check never
    fires -- the answer layer would otherwise present the zero as if it
    were a real, successful single-value answer ("Returned 1 rows").

    Deliberately narrower than "all numeric columns are zero/null": that
    would misfire on a legitimately-zero real answer (e.g. actual $0 profit
    this month). Only fires when the row carries one of the same explicit
    match-count column names detect_null_metric_issue already trusts.
    """
    if len(rows) != 1 or not rows[0]:
        return False
    row = rows[0]
    matched_header = _find_matched_rows_header(list(row.keys()))
    if not matched_header:
        return False
    matched_rows = _to_float(row.get(matched_header))
    return matched_rows is not None and matched_rows <= 0


def detect_null_metric_issue(rows: list[dict]) -> dict[str, Any] | None:
    """
    Detect diagnostic rows where records matched, but a requested metric was
    NULL/missing for every matched record.
    """
    if len(rows) != 1 or not rows[0]:
        return None
    row = rows[0]
    headers = list(row.keys())
    matched_header = _find_matched_rows_header(headers)
    matched_rows = _to_float(row.get(matched_header)) if matched_header else None
    if matched_rows is None or matched_rows <= 0:
        return None

    issues: list[dict[str, Any]] = []
    for header in headers:
        norm = _normalise_key(header)
        if not (norm.startswith("nonnull") and norm.endswith("rows")):
            continue
        non_null_rows = _to_float(row.get(header))
        if non_null_rows is None or non_null_rows > 0:
            continue
        metric_norm = norm[len("nonnull"):-len("rows")]
        metric_header = _find_header_by_norm(headers, metric_norm)
        if not metric_header:
            continue
        metric_value = row.get(metric_header)
        metric_num = _to_float(metric_value)
        if metric_value not in (None, "") and metric_num not in (0, 0.0):
            continue
        issues.append({
            "metric_column": metric_header,
            "non_null_column": header,
            "matched_rows": int(matched_rows),
            "non_null_rows": int(non_null_rows),
            "value": metric_value,
        })

    if not issues:
        return None
    return {
        "matched_rows": int(matched_rows),
        "matched_column": matched_header,
        "issues": issues,
    }


def _best_label(question: str, label_col: str, value_col: str) -> str:
    q = question.strip().rstrip("?")
    if len(q.split()) >= 4:
        return q
    return f"{value_col.replace('_', ' ').title()} by {label_col.replace('_', ' ').title()}"


def _extract_limit(sql: str) -> int | None:
    if not sql:
        return None
    patterns = [
        r"\btop\s*\(\s*(\d+)\s*\)",
        r"\btop\s+(\d+)\b",
        r"\blimit\s+(\d+)\b",
        r"\bfetch\s+first\s+(\d+)\s+rows?\s+only\b",
        r"\bfetch\s+next\s+(\d+)\s+rows?\s+only\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, sql, re.I)
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                return None
    return None


def infer_result_scope(
    rows: list[dict],
    question: str,
    sql: str = "",
    *,
    mode: str = "table",
) -> dict[str, Any]:
    row_count = len(rows)
    lower_sql = (sql or "").lower()
    explicit_limit = _extract_limit(sql)
    # A row limit only truncated the result if the result REACHED it.
    #
    # The Azure prompt tells the model to write TOP 20 by default, so every
    # generated T-SQL query carries a limit whether or not the question asked
    # for one. Reading the mere presence of TOP as "this is a top-N slice"
    # marked a three-warehouse breakdown as limited: the trend sentence, the
    # decision signal and the ranking's runner-up clause were all withheld --
    # each gated on this flag for good reason -- and the ranking explanation
    # was told the reader had asked for a top twenty. Fewer rows back than the
    # limit is proof the limit never bit. Exactly the limit is treated as
    # bound: it may have been, and the cost of assuming so is one withheld
    # sentence rather than a claim about rows that were cut.
    limit_bound = (
        explicit_limit is not None and row_count > 0 and row_count >= explicit_limit
    )
    # The preview cap is a truncation of its own, whether or not the SQL also
    # carried a larger TOP: TOP 500 returning 200 rows was cut by the cap.
    preview_cap_hit = row_count >= _PREVIEW_ROW_CAP and row_count > 0 and not limit_bound
    filtered_subset = " where " in f" {lower_sql} "
    was_limited = limit_bound or preview_cap_hit

    scope: dict[str, Any] = {
        "kind": mode,
        "question": question,
        "row_count": row_count,
        "limit_value": explicit_limit,
        "was_limited": was_limited,
        "is_preview": preview_cap_hit,
        "filtered_subset": filtered_subset,
        "is_top_n": False,
        "n": None,
        "is_complete_distribution": False,
        "is_complete_series": False,
    }

    # A question that asks for the low end of a ranking is answered from it:
    # "which warehouse has the lowest stock" was headed by the warehouse with
    # the most.
    from core.analytical_intent import asks_for_ranking, ranking_direction, writes_an_order

    # And from the order the question writes out, ranking word or not: "par
    # ordre croissant" and "du plus bas au plus élevé" were headed by the
    # largest row, "arrive en tête".
    scope["ascending"] = (asks_for_ranking(question) or writes_an_order(question)) and (
        ranking_direction(question) == "ascending")
    if mode == "ranking":
        # A top-N framing needs a limit that bound, or a question that asked
        # for one ("top 5 customers" returning all 3 that exist is still the
        # reader's top-5 framing, and complete). A defensive TOP that never
        # bit, on a question that never asked, is neither.
        asks_top_n = detect_top_n_intent(question) is not None
        if explicit_limit is not None and (limit_bound or asks_top_n):
            scope["is_top_n"] = True
            scope["n"] = explicit_limit
        scope["is_complete_distribution"] = not was_limited
    elif mode == "time_series":
        scope["is_complete_series"] = not was_limited
        if explicit_limit is not None:
            scope["n"] = explicit_limit

    # badge_key is the stable token; badge and note are what the reader sees.
    # The narration prompt in core/insight.py reads the KEY, so a French portal
    # does not quietly change what the model is asked -- the answer's language
    # is decided by the prompt's own language rule, not by a leaked label.
    badge_key = "returned"
    fields: dict[str, Any] = {}
    if scope["is_top_n"]:
        n = scope["n"] or row_count
        if n == 1:
            badge_key = "top_one"
        else:
            badge_key, fields = "top_n", {"n": n}
    elif mode == "ranking" and scope["is_complete_distribution"]:
        badge_key = "full_distribution"
    elif mode == "time_series" and scope["is_complete_series"]:
        badge_key = "full_series"
    elif mode == "time_series" and limit_bound:
        # A series cut by a binding TOP. A ranking cut the same way says
        # "Top 20 only"; this said "Returned result", which names nothing.
        badge_key, fields = "partial_series", {"n": explicit_limit}
    elif scope["is_preview"]:
        badge_key = "preview"
    elif filtered_subset:
        badge_key = "filtered_subset"

    note = _t(f"answer.scope.{badge_key}_note", **fields)
    scope["badge_key"] = badge_key
    scope["badge"] = _t(f"answer.scope.{badge_key}", **fields)
    scope["note"] = note
    # The badge as a noun phrase, for the middle of a sentence. The analysis
    # card used to write `badge.lower()` there -- which reads as English and
    # cannot work in French, where an inline noun phrase needs its article and
    # the article carries the gender.
    scope["inline"] = _t(f"answer.scope.{badge_key}.inline", **fields)
    scope["analysis_note"] = (
        _t("answer.scope.slice_note")
        if was_limited and mode in {"ranking", "time_series"}
        else note
    )
    return scope


def _per_unit_listing(totals: list[tuple[str, float]], missing: list[str], format_value, value_col: str) -> str:
    """ "220 EA and 1,250 FT": each unit's total, by the unit's name -- the
    card ranks nothing -- three at most, and the rest counted."""
    listed = [f"{format_value(value, value_col)} {unit or _t('answer.per_unit.no_unit')}" for unit, value in totals]
    listed += [_t("answer.per_unit.no_value", unit=unit or _t("answer.per_unit.no_unit")) for unit in missing]
    if len(listed) > 3:
        return _t_plural("answer.per_unit.more", len(listed) - 3, values=", ".join(listed[:3]))
    if len(listed) == 1:
        return listed[0]
    return _t("answer.per_unit.last", values=", ".join(listed[:-1]), last=listed[-1])


# What a row with no unit is labelled (core/unknown_members.py), in either
# language: no unit to rank in.
_NO_UNIT = frozenset({"", "none", "null", "unknown", "inconnu", "inconnue"})


def _leader_in_one_unit(
    rows: list[dict], label_col: str, text_cols: list[str], value_col: str, ascending: bool,
) -> tuple[str, str, float] | None:
    """(unit, label, value) of the first label in the unit most of a quantity
    is kept in, for rows broken down by a label and by unit of measure; None
    for any other result. Rows of one label in one unit are added up."""
    from core.units_of_measure import is_quantity_measure, is_unit_column

    unit_col = next((c for c in text_cols if c != label_col and is_unit_column(c)), None)
    # Money adds up across units, and is not said in one.
    if not unit_col or len(text_cols) != 2 or not is_quantity_measure(value_col):
        return None
    totals: dict[str, dict[str, float]] = {}
    for row in rows:
        unit = str(row.get(unit_col) or "").strip()
        value = _to_float(row.get(value_col))
        if unit.casefold() in _NO_UNIT or value is None:
            continue
        label = row.get(label_col)
        label = _t("member.unknown") if label is None or str(label).strip() in {"", "None", "null", "NULL"} else str(label)
        by_label = totals.setdefault(unit, {})
        by_label[label] = by_label.get(label, 0.0) + value
    if len(totals) < 2 and not any(len(by_label) > 1 for by_label in totals.values()):
        return None
    # The unit the most of them hold any of, then the most of them are kept
    # in -- never one where every one is at nothing -- and only then the
    # larger, to settle a tie: the headline is said in one unit, not ranked
    # across them.
    # Leading, a unit some of them hold any of; lowest, the unit most are
    # kept in, where a zero is the lowest there is.
    if ascending:
        unit = max(totals, key=lambda u: (len(totals[u]), sum(abs(v) for v in totals[u].values()),
                                          -sorted(totals).index(u)))
    else:
        unit = max(totals, key=lambda u: (sum(1 for v in totals[u].values() if v > 0),
                                          sum(1 for v in totals[u].values() if v), len(totals[u]),
                                          sum(abs(v) for v in totals[u].values()), -sorted(totals).index(u)))
    if len(totals[unit]) < 2:
        return None
    ordered = sorted(totals[unit].items(), key=lambda item: (item[1], item[0]), reverse=not ascending)
    # A tie at the top leads nothing.
    if ordered[0][1] == ordered[1][1]:
        return None
    label, value = ordered[0]
    return unit, label, value


def _per_unit_answer(
    rows: list[dict], unit_col: str, value_col: str, question: str, value_fmt: str | None,
    format_value, scope: dict,
) -> dict | None:
    """The card for a quantity's totals, one per unit of measure, or None.

    Each is a total in its own unit (core/units_of_measure.rows_per_unit),
    so none leads, none is above the next, and no one figure is the answer:
    they are listed, and the card says why they are not added.
    """
    from core.units_of_measure import is_quantity_measure, is_unit_column, per_unit_totals, rows_per_unit

    # One unit's total is that total, not a leader: "units sold last 6 months"
    # came back in one unit and was headed "Unknown leads at 939,315.50".
    single = (len(rows) == 1 and is_unit_column(unit_col)
              and is_quantity_measure(value_col, str(value_fmt or "")))
    if not single and not rows_per_unit(rows, unit_col, value_col, question, measure_format=str(value_fmt or "")):
        return None
    found = per_unit_totals(rows, unit_col, value_col)
    if not found or not found[0]:
        return None
    totals, missing = found
    measure = _display_label(value_col) or _t("answer.total")
    if not missing and not any(value for _, value in totals):
        headline = _t("answer.per_unit.all_zero", measure=measure)
    else:
        headline = _t("answer.per_unit", measure=measure,
                      values=_per_unit_listing(totals, missing, format_value, value_col))
    # Said in the note: the portal prints the comparison only beside a single
    # value, and this card has none, so "not added together" never showed.
    note = " ".join(part for part in (scope.get("note", ""), _t("answer.per_unit.not_added")) if part)
    return {
        "headline": headline,
        "short_value": "",
        "comparison": "",
        "scope_badge": scope.get("badge", ""),
        "scope_note": note,
    }


def build_answer(
    rows: list[dict],
    question: str,
    result_scope: dict | None = None,
    column_formats: dict | None = None,
    display_formats: dict | None = None,
    period_labels: list[str] | None = None,
    asked_grain: str | None = None,
) -> dict:
    scope = result_scope or infer_result_scope(rows, question)
    column_formats = column_formats or {}
    display_formats = display_formats or {}
    grain = requested_period_grain(question) if asked_grain is None else asked_grain

    def format_value(value: Any, column: str) -> str:
        return _format_display_value(
            value,
            column_formats.get(column),
            display_formats.get(column),
        )
    if not rows or detect_zero_match_result(rows):
        return {
            "headline": _t("answer.no_match_headline"),
            "short_value": _t_plural("answer.rows", 0),
            "comparison": _t("answer.no_match_hint"),
            "scope_badge": scope.get("badge", ""),
            "scope_note": scope.get("note", ""),
        }

    missing_scalar = _single_missing_scalar(rows)
    if missing_scalar:
        return _missing_scalar_copy(missing_scalar[0], question)

    null_issue = detect_null_metric_issue(rows)
    if null_issue:
        issue = null_issue["issues"][0]
        metric_col = issue["metric_column"]
        fmt = column_formats.get(metric_col)
        value = format_value(_to_float(rows[0].get(metric_col)) or 0, metric_col)
        metric_label = _display_label(metric_col)
        matched = null_issue["matched_rows"]
        return {
            "headline": _t("answer.null_metric_headline",
                           metric=metric_label, value=value),
            "short_value": value,
            "comparison": _t_plural("answer.null_metric_comparison", matched,
                                    metric=metric_label),
            "scope_badge": _t("answer.null_metric_badge"),
            "scope_note": _t_plural("answer.null_metric_note", matched),
        }

    numeric_cols = _numeric_cols(rows)
    text_cols = _text_cols(rows, numeric_cols)
    numeric_cols, text_cols, _period_cols = _measure_and_label_cols(
        rows, numeric_cols, text_cols)

    if len(rows) == 1 and len(rows[0]) == 1:
        col = next(iter(rows[0].keys()))
        val = rows[0][col]
        fmt = column_formats.get(col)
        return {
            "headline": _t("answer.label_value",
                           label=_display_label(col),
                           value=format_value(val, col)),
            "short_value": format_value(val, col),
            "comparison": scope.get("badge") or _t("answer.single_value"),
            "scope_badge": scope.get("badge", ""),
            "scope_note": scope.get("note", ""),
        }

    # A listing (summarize_result_context, _is_listing) is records, and its
    # first numeric column is not a measure to lead on. It takes the
    # record-count headline at the bottom, like a result with no numbers.
    _listing = scope.get("kind") == "text_table"

    if numeric_cols and text_cols and not _listing:
        # A named-period comparison, before the ranking path. Fixing the SQL
        # alone does not fix the answer: build_answer took numeric_cols[0] --
        # the OLDEST period column on a widened result -- and opened the card
        # with "Pumps leads at 3,800,000." and a cross-category gap chip that
        # reads exactly like a year-over-year delta. The target question is not
        # causal, so no narration runs and these deterministic sentences ARE
        # the answer the reader gets.
        _pair = _period_pair_facts(rows, period_labels)
        if _pair:
            newest_col = _pair["newest_column"]
            measure = _display_label(
                _measure_prefix(newest_col, _pair["newest_label"])) or _t("answer.total")
            pct = _pair["total_pct"]
            # A whole sentence per direction. English joins a verb to a
            # percentage; French conjugates ("a augmenté de 12,3 %") and agrees
            # the participle with the measure, so there is no seam in the
            # middle for a translated adverb to slot into.
            rose, fell = _pair["net"] > 0, _pair["net"] < 0
            if pct is None:
                msg_id = ("answer.period_rose_unquantified" if rose
                          else "answer.period_fell_unquantified" if fell
                          else "answer.period_flat")
                headline = _t(msg_id, measure=measure,
                              old=_pair["oldest_label"], new=_pair["newest_label"])
            else:
                msg_id = ("answer.period_rose" if rose
                          else "answer.period_fell" if fell
                          else "answer.period_flat")
                headline = _t(msg_id, measure=measure, pct=_format_percent(abs(pct), 1),
                              old=_pair["oldest_label"], new=_pair["newest_label"])
            riser = _pair["top_riser"] or _pair["top_faller"]
            if riser:
                mover = _safe_category_label(riser["label"], _pair["label_column"])
                change = ("+" if riser["change"] > 0 else "") + format_value(
                    riser["change"], newest_col)
                headline = _t(
                    "answer.period_mover" if mover else "answer.period_mover_unnamed",
                    sentence=headline, mover=mover, change=change)
            comparison = (_t("answer.period_versus", pct=_signed_percent(pct),
                             old=_pair["oldest_label"])
                          if pct is not None
                          else _t("answer.period_compared", old=_pair["oldest_label"]))
            return {
                "headline": headline + ".",
                "short_value": format_value(_pair["newest_total"], newest_col),
                "comparison": comparison,
                "scope_badge": scope.get("badge", ""),
                "scope_note": scope.get("note", ""),
            }

        # The headline card was left behind when summarize_result_context was
        # taught about a second dimension, so it still read text_cols[0] and
        # ranked raw rows. On "compare revenue by warehouse for the last 3
        # months" it led with "2026-05 closed at 900,000" -- a month's name
        # against one warehouse's row -- and its leader and runner-up were the
        # same warehouse in two different periods, which is where the
        # "5,416 above the next result" came from.
        label_col = _narrative_label_column(rows, text_cols)
        value_col = numeric_cols[0]
        value_fmt = column_formats.get(value_col)
        # A quantity's totals, one per unit of measure, are listed, not
        # ranked: "What is our total stock on hand?" was headed "FT leads at
        # 32,402", "19,251 above the next result" -- feet ahead of eaches.
        per_unit = _per_unit_answer(rows, label_col, value_col, question, value_fmt, format_value, scope)
        if per_unit:
            return per_unit
        collapsed = collapse_rows_by_label(rows, label_col, value_col)
        ascending = bool(scope.get("ascending"))
        ordered = sorted(collapsed or [], key=lambda pair: pair[1], reverse=not ascending)
        labels = [str(r.get(label_col, "")) for r in rows]
        # A period key is a period too: 202501..202504 is headed "April 2025
        # closed at", as the sentences below it read the series, not
        # "202504 leads at".
        periods = _a_period_axis(label_col, [r.get(label_col) for r in rows], labels, grain)
        if (periods and len(set(labels)) == len(labels)
                and scope.get("kind") != "ranking"):
            first = rows[0]
            last = rows[-1]
            first_val = _to_float_z(first.get(value_col))
            last_val = _to_float_z(last.get(value_col))
            trend = ("answer.trend_up" if last_val > first_val
                     else "answer.trend_down" if last_val < first_val
                     else "answer.trend_flat")
            # The period in the words the table and the insight sentence use
            # for it, not through the number formatter, which turned an
            # integer year into "2,025".
            last_label = _sentence_start(narrative_period_labels(labels, grain, label_col)[-1]) or _t(
                "answer.latest_period")
            headline = _t("answer.series_close", label=last_label,
                          value=format_value(last_val, value_col))
            comparison = scope.get("badge") or _t(
                trend, value=format_value(first_val, value_col))
            return {
                "headline": headline,
                "short_value": format_value(last_val, value_col),
                "comparison": comparison,
                "scope_badge": scope.get("badge", ""),
                "scope_note": scope.get("note", ""),
            }
        if not ordered:
            # A quantity broken down by something AND by its unit is ranked
            # within one unit -- the one most of it is kept in -- and says so:
            # "stock on hand by warehouse" was "Returned 19 rows", because
            # feet and eaches are not added up to rank a warehouse.
            in_unit = _leader_in_one_unit(rows, label_col, text_cols, value_col, ascending)
            # Nothing anywhere leads: every one is at nothing.
            from core.units_of_measure import is_quantity_measure, is_unit_column

            if (any(is_unit_column(c) for c in text_cols if c != label_col) and is_quantity_measure(value_col)
                    and not any(_to_float(row.get(value_col)) for row in rows)):
                return {
                    "headline": _t("answer.per_unit.all_zero",
                                   measure=_display_label(value_col) or _t("answer.total")),
                    "short_value": "",
                    "comparison": scope.get("badge", ""),
                    "scope_badge": scope.get("badge", ""),
                    "scope_note": scope.get("note", ""),
                }
            if in_unit:
                unit, label, value = in_unit
                return {
                    "headline": _t("answer.lowest_in_unit" if ascending else "answer.leads_in_unit",
                                   unit=unit, label=label, value=format_value(value, value_col)),
                    # No one figure is the answer: it is one unit's.
                    "short_value": "",
                    "comparison": scope.get("badge", ""),
                    "scope_badge": scope.get("badge", ""),
                    "scope_note": " ".join(part for part in (
                        scope.get("note", ""), _t("answer.per_unit.not_added")) if part),
                }
            # Repeats the sum may not merge -- a margin percentage, a balance.
            # A leader among rows that cannot be added together is a made-up
            # ranking, so this falls through to the plain row-count answer.
            return {
                "headline": _t_plural(
                    "answer.returned_rows", len(rows),
                    question=question.strip().rstrip("?") or _t("answer.this_query")),
                "short_value": "",
                "comparison": scope.get("badge", ""),
                "scope_badge": scope.get("badge", ""),
                "scope_note": scope.get("note", ""),
            }
        best_label, best_value = ordered[0]
        # A group with no label -- a fact row whose key matched no row of its
        # dimension, kept by the outer join -- is "Unknown", not "None leads".
        if best_label is None or str(best_label).strip() in {"", "None", "null", "NULL"}:
            best_label = _t("member.unknown")
        # A period ranked by its measure -- "which month had the highest
        # sales" -- is named as the series names it: "2025-10", not the
        # first day of its bucket.
        if periods and len(set(labels)) == len(labels):
            best_label = _sentence_start(dict(zip(labels, narrative_period_labels(labels, grain, label_col))).get(
                str(best_label), best_label))
        best_label = str(best_label or _t("answer.top_result"))
        comparison = scope.get("badge") or _t_plural(
            "answer.across_results", len(ordered))
        if scope.get("is_top_n") and (scope.get("n") or 0) == 1:
            headline = _t("answer.bottom_ranked" if ascending else "answer.top_ranked", label=best_label,
                          value=format_value(best_value, value_col))
            comparison = _t("answer.lowest_row_only" if ascending else "answer.leading_row_only")
        else:
            headline = _t("answer.lowest" if ascending else "answer.leads", label=best_label,
                          value=format_value(best_value, value_col))
        if len(ordered) > 1 and not scope.get("is_top_n"):
            delta = abs(best_value - ordered[1][1])
            comparison = _t("answer.below_next" if ascending else "answer.above_next",
                            delta=format_value(delta, value_col))
        return {
            "headline": headline,
            "short_value": format_value(best_value, value_col),
            "comparison": comparison,
            "scope_badge": scope.get("badge", ""),
            "scope_note": scope.get("note", ""),
        }

    if numeric_cols and not _listing:
        col = numeric_cols[0]
        value_fmt = column_formats.get(col)
        values = [_to_float_z(r.get(col)) for r in rows]
        return {
            "headline": _t_plural(
                "answer.returned_rows", len(rows),
                question=question.strip().rstrip("?") or _t("answer.this_query")),
            "short_value": format_value(values[0], col),
            "comparison": scope.get("badge") or _t(
                "answer.range", low=format_value(min(values), col),
                high=format_value(max(values), col)),
            "scope_badge": scope.get("badge", ""),
            "scope_note": scope.get("note", ""),
        }

    # Pure text result — e.g. a list of names. Show a preview in the chip.
    first_col = list(rows[0].keys())[0]
    preview_items = [str(r.get(first_col, "")) for r in rows[:3] if r.get(first_col)]
    preview = ", ".join(preview_items)
    if len(rows) > 3:
        preview += ", " + _t("answer.more_items", count=len(rows) - 3)
    return {
        "headline": _t_plural(
            "answer.found_results", len(rows),
            question=question.strip().rstrip("?") or _t("answer.your_query")),
        "short_value": _t_plural("answer.rows", len(rows)),
        "comparison": scope.get("badge") or preview or _t("answer.review_records"),
        "scope_badge": scope.get("badge", ""),
        "scope_note": scope.get("note", ""),
    }


_AGGREGATION_RE = re.compile(
    r"\bGROUP\s+BY\b|\bOVER\s*\(|"
    r"\b(?:SUM|COUNT|COUNT_BIG|AVG|MIN|MAX|STRING_AGG|STDEV|STDEVP|VAR|VARP)\s*\(",
    re.IGNORECASE,
)


def _is_listing(
    question: str, sql: str, rows: list[dict],
    numeric_cols: list[str], text_cols: list[str],
) -> bool:
    """Rows that are individual records rather than one row per label.

    A SELECT that aggregated nothing returns records -- invoice lines, order
    headers -- and the first numeric column of a record is an attribute of
    it, not a measure the records compete on. Read as a ranking, twelve
    invoice lines of one customer became

        INV11 leads at 21 (11.3% of total) across 12 invoice nos.
        Volume is spread across the field — no single entry exceeds 12%.

    Records are known by their key: a text column that names an identifier
    (the chart's own rule, _looks_identifier -- IVC_NO, ORD_NO, CUS_ID). A
    two-column label-and-value result read straight off a pre-aggregated
    view carries no such column and stays the ranking it looks like.

    Three things make records a ranking after all: the SQL aggregated (a
    GROUP BY, an aggregate, a window), the reader asked for a top N, or the
    rows are sorted by a measure, descending. Without the SQL nothing can be
    known and the reading stays what it was.
    """
    from core.chart_spec import _looks_identifier

    text = str(sql or "")
    if not text.strip():
        return False
    # A numeric key set aside from the measures and the labels (a row's
    # identifier, _measure_and_label_cols) marks records too, where the label
    # repeats: one customer's invoices.
    keys = list(text_cols)
    set_aside = [column for column in (rows[0] if rows else {}) if column not in numeric_cols + text_cols]
    if text_cols and set_aside and len({str(row.get(text_cols[0])) for row in rows}) < len(rows):
        keys += set_aside
    if not any(_looks_identifier(rows, column) for column in keys):
        return False
    if _AGGREGATION_RE.search(text):
        return False
    if detect_top_n_intent(question) is not None:
        return False
    order = re.search(r"\bORDER\s+BY\b(.*)$", text, re.IGNORECASE | re.DOTALL)
    if order:
        clause = order.group(1)
        for column in numeric_cols:
            if re.search(rf"\b{re.escape(column)}\b[^,]*\bDESC\b", clause, re.IGNORECASE):
                return False
    return True


def summarize_result_context(
    rows: list[dict], question: str, sql: str = "", *, asked_grain: str | None = None,
) -> dict:
    numeric_cols = _numeric_cols(rows)
    text_cols = _text_cols(rows, numeric_cols)
    numeric_cols, text_cols, period_cols = _measure_and_label_cols(
        rows, numeric_cols, text_cols)
    ctx: dict[str, Any] = {
        "question": question,
        "row_count": len(rows),
        "numeric_cols": numeric_cols,
        "text_cols": text_cols,
        # Named so a consumer can tell a period axis from a measure without
        # re-deriving it, and so the split is visible in the trace.
        "period_cols": period_cols,
        # Every column the result has, a row's identifier set aside from both
        # lists above included: no "break down by" what is already there.
        "columns": list(rows[0]) if rows else [],
        "mode": "table",
        "chartable": False,
    }
    if not rows:
        ctx["mode"] = "empty"
        ctx["result_scope"] = infer_result_scope(rows, question, sql, mode="empty")
        return ctx

    # Single scalar result (one row, one column) — set mode so _build_insight_summary
    # can produce a meaningful sentence instead of falling through to return "".
    if len(rows) == 1 and len(rows[0]) == 1:
        col = next(iter(rows[0].keys()))
        raw_val = rows[0][col]
        # Every other branch below normalizes DB values through _to_float/
        # _to_float_z before putting them in ctx; this one didn't, so a raw
        # decimal.Decimal (returned by pyodbc/Azure SQL for SUM() on a
        # numeric/decimal column) rode straight into analysis_contract and
        # broke ws.send_json's JSON encoder — the query succeeded but the
        # user got total silence with no error surfaced anywhere.
        safe_val = _to_float(raw_val)
        if safe_val is None:
            safe_val = "" if raw_val is None else str(raw_val)
        ctx.update({
            "mode": "single_value",
            "value_column": col,
            "value": safe_val,
            "chartable": False,
        })
        ctx["result_scope"] = infer_result_scope(rows, question, sql, mode="single_value")
        return ctx

    if numeric_cols and text_cols:
        label_col = _narrative_label_column(rows, text_cols)
        value_col = numeric_cols[0]
        # Formatted here, at the one place the series is read, so every
        # sentence written about it downstream inherits the same labels the
        # table and the KPI show.
        labels = narrative_period_labels(
            [r.get(label_col, "") for r in rows],
            requested_period_grain(question) if asked_grain is None else asked_grain, label_col)
        values = [_to_float_z(r.get(value_col)) for r in rows]
        ctx.update({
            "label_col": label_col,
            "value_col": value_col,
            "labels": labels,
            "values": values,
            "min_value": min(values),
            "max_value": max(values),
            "avg_value": mean(values),
            "median_value": median(values),
            "chartable": True,
        })
        ctx["top_items"] = _ranked_items(rows, label_col, value_col)
        # A repeating label is not a series axis. The result is grouped by
        # something else as well, so first and last are two different members
        # of that other dimension -- which is how "+1,437.3% from 2026-03 to
        # 2026-06" was computed between two different warehouses.
        from core.units_of_measure import kept_in_several_units

        a_series = (_a_period_axis(label_col, [r.get(label_col) for r in rows], labels)
                    and len(set(labels)) == len(labels)
                    # Periods listed by a figure are a ranking of them wherever
                    # the result is read: the card, the analyst's prompt, the
                    # conversation, the chart's markers.
                    and not _ranked_by_a_figure(question, sql, rows))
        if a_series and kept_in_several_units(rows, value_col):
            # Nor is a quantity kept in feet one month and eaches the next a
            # series of one thing: receipts "trended down 22.2%" from 900 FT to
            # 700 FT across two months of eaches, and the chips asked what drove
            # the drop. The periods are listed; nothing is read across them.
            ctx.update({"mode": "time_series", "several_units": True})
        elif a_series:
            labels, values = in_time_order(labels, values)
            ctx.update({"labels": labels, "values": values})
            first, last = values[0], values[-1]
            pct = _safe_pct_change(first, last)
            diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
            ctx.update({
                "mode": "time_series",
                "first_label": labels[0],
                "last_label": labels[-1],
                "first_value": first,
                "last_value": last,
                "pct_change": pct,
                "avg_step_change": mean(diffs) if diffs else 0.0,
                "volatility": mean(abs(d) for d in diffs) if diffs else 0.0,
                "comparison_stats": {
                    "first_period": labels[0],
                    "first_value": first,
                    "last_period": labels[-1],
                    "last_value": last,
                    "absolute_change": round(last - first, 2),
                    "pct_change": round(pct, 2) if pct is not None else None,
                },
            })
        elif _is_listing(question, sql, rows, numeric_cols, text_cols):
            # Records, not a ranking. The columns and values stay for the
            # sentence about what they add up to; no leader, no share, no
            # chart, and build_answer reads the scope's kind to agree.
            ctx.update({"mode": "text_table", "listing": True, "chartable": False})
            ctx.pop("top_items", None)
        else:
            from core.units_of_measure import kept_in_several_units, per_unit_totals, rows_per_unit

            ctx["mode"] = "ranking"
            # A quantity's totals, one per unit of measure (_per_unit_answer):
            # none leads, none is above the next, none is a share of their sum.
            # The chips and the fallback cards read this context, and on stock
            # by unit they said "FT is ahead of EA by 1,030" and "Leader share
            # of returned total: 85.0%".
            per_unit = (per_unit_totals(rows, label_col, value_col)
                        if rows_per_unit(rows, label_col, value_col, question) else None)
            if per_unit and per_unit[0]:
                ctx.pop("top_items", None)
                ctx["per_unit"] = {"totals": per_unit[0], "missing": per_unit[1]}
                ctx["distribution_stats"] = {"category_count": len(per_unit[0]) + len(per_unit[1])}
                ctx["comparison_stats"] = {}
                ctx["result_scope"] = infer_result_scope(rows, question, sql, mode="ranking")
                return ctx
            # A quantity the rows keep in several units -- items ranked by units
            # sold, each in its own -- is ranked, and its leader leads by what
            # it leads by, but its sum adds eaches to feet: no share of it, and
            # no spread or middle across the rows either.
            mixed_units = kept_in_several_units(rows, value_col)
            # Nor any share of a result the limit cut: the returned rows are not
            # every category, of a rate, or of a balance at several dates
            # (compute_data_brief). The fallback cards said "Leader share of
            # returned total: 35.1%" of the inventory value at three month ends.
            from core.insight import adds_up_across

            scope = infer_result_scope(rows, question, sql, mode="ranking")
            total = (0.0 if mixed_units or scope.get("was_limited")
                     or not adds_up_across(value_col, label_col, [str(label) for label in labels])
                     else sum(values))
            top_items = ctx.get("top_items") or []
            leader = top_items[0] if top_items else None
            runner_up = top_items[1] if len(top_items) > 1 else None
            ctx["distribution_stats"] = {"category_count": len(set(labels))} if mixed_units else {
                "category_count": len(set(labels)),
                "spread": round(max(values) - min(values), 2) if values else 0.0,
                "median_value": round(median(values), 2) if values else 0.0,
                "top_3_share_pct": round(sum(item["value"] for item in top_items[:3]) / total * 100, 1) if total > 0 and top_items else None,
                "std_dev": _std_dev(values),
            }
            comparison_stats = {}
            if leader:
                comparison_stats.update({
                    "leader": leader["label"],
                    "leader_value": leader["value"],
                    "leader_share_pct": round(leader["value"] / total * 100, 1) if total > 0 else None,
                })
            if leader and runner_up:
                comparison_stats.update({
                    "runner_up": runner_up["label"],
                    "runner_up_value": runner_up["value"],
                    "gap": round(leader["value"] - runner_up["value"], 2),
                })
            ctx["comparison_stats"] = comparison_stats
            ctx["result_scope"] = scope
            return ctx
        ctx["result_scope"] = infer_result_scope(rows, question, sql, mode=ctx["mode"])
        return ctx

    if numeric_cols:
        value_col = numeric_cols[0]
        values = [_to_float_z(r.get(value_col)) for r in rows]
        ctx.update({
            "mode": "numeric_table",
            "value_col": value_col,
            "values": values,
            "min_value": min(values),
            "max_value": max(values),
            "avg_value": mean(values),
            "median_value": median(values),
            "distribution_stats": {
                "spread": round(max(values) - min(values), 2),
                "std_dev": _std_dev(values),
            },
        })
        ctx["result_scope"] = infer_result_scope(rows, question, sql, mode="numeric_table")
        return ctx

    ctx["mode"] = "text_table"
    ctx["result_scope"] = infer_result_scope(rows, question, sql, mode="text_table")
    return ctx


_CHIP_THRESHOLD = 68  # minimum confidence to surface a chip


def compute_chip_eligibility(
    ctx: dict,
    brief: dict | None = None,
    semantic_plan: dict | None = None,
    *,
    sql: str = "",
    db_type: str = "",
) -> list[dict]:
    """
    Signal-based chip eligibility.  Replaces the old mode-only ``_dynamic_actions``.

    Every chip is scored against actual data-brief signals — not just the result
    *mode*.  Chips that score below ``_CHIP_THRESHOLD`` are silently omitted so
    the user only sees actions the data can actually support.

    Returns a list of ``{id, label, confidence, pre_context}`` dicts ordered by
    a fixed display priority (explain → analyze → compare → … → decide).
    The ``pre_context`` string is a one-liner explaining *why* the chip is
    relevant (shown as a hover tooltip / subtitle on the button).
    """
    brief = brief or {}
    mode       = ctx.get("mode", "table")
    row_count  = ctx.get("row_count", 0)
    ts         = brief.get("time_series") or {}
    cat        = brief.get("category_breakdown") or {}
    dist       = ctx.get("distribution_stats") or {}
    cmp_stats  = ctx.get("comparison_stats") or {}

    chips: list[dict] = []

    def _add(id_: str, label: str, confidence: int, pre_context: str = "") -> None:
        if confidence >= _CHIP_THRESHOLD:
            chips.append({
                "id": id_,
                "label": label,
                "confidence": confidence,
                "pre_context": pre_context,
            })

    # ── time_series chips ────────────────────────────────────────────────────
    if mode == "time_series":
        direction    = ts.get("direction") or "stable"
        period_count = ts.get("period_count") or row_count
        pct_change   = ts.get("overall_pct_change")
        if pct_change is None:
            pct_change = ctx.get("pct_change") or 0.0

        # compare_period: only meaningful when overall change is non-trivial
        if period_count >= 2 and pct_change is not None and abs(pct_change) >= 3.0:
            sign = "+" if pct_change > 0 else ""
            _add(
                "compare", _t("chip.compare"),
                82 if abs(pct_change) >= 10 else 73,
                _t("chip.compare_hint", pct=f"{sign}{_format_percent(pct_change, 1)}"),
            )

        # diagnose: root-cause chip for significant movement
        if pct_change is not None and abs(pct_change) >= 5.0:
            # Two ids, not one with a noun slot: French moves the noun to the
            # front of the hint ("baisse de 12,3 %") and puts the adjective
            # agreement on the demonstrative in the label.
            _shape = "drop" if pct_change < 0 else "rise"
            _add(
                "diagnose", _t(f"chip.diagnose_{_shape}"),
                88 if abs(pct_change) >= 10 else 80,
                _t(f"chip.diagnose_{_shape}_hint", pct=_format_percent(abs(pct_change), 1)),
            )

        # compare_prior: available when the semantic model knows the date role
        if semantic_plan and semantic_plan.get("enabled"):
            has_date_role = any(
                f.get("role") == "date_dimension"
                for f in (semantic_plan.get("fields") or [])
            )
            if has_date_role:
                _add(
                    "compare_prior", _t("chip.compare_prior"), 70,
                    _t("chip.compare_prior_hint"),
                )

    # ── ranking chips ────────────────────────────────────────────────────────
    elif mode == "ranking":
        # contribution: % share breakdown useful for ranking results. The
        # brief's share, not one worked out from the rows here: the brief has
        # none where the rows are not the whole -- a top five the limit cut,
        # rows in more than one unit, a measure that does not add up -- and
        # the chip offered "FT holds 69% of total" beside a card saying feet
        # and eaches are not added together.
        top5 = cat.get("top_5") or []
        leader = (top5[0].get("label") if top5 else "") or cmp_stats.get("leader") or "top item"
        leader_share = cat.get("leader_share_pct")
        if leader_share is not None and row_count >= 2:
            _add(
                "contribution", _t("chip.contribution"), 78,
                _t("chip.contribution_hint", leader=leader,
                   pct=f"{leader_share:.0f}"),
            )

    # ── drill_dim — "Break down by X" chips ─────────────────────────────────
    # Show at most 2 dimensions that are available in the semantic model but
    # not already present in the current result.
    if semantic_plan and semantic_plan.get("enabled") and row_count >= 1:
        result_cols_upper = {
            c.upper()
            for c in (ctx.get("columns") or (ctx.get("numeric_cols") or []) + (ctx.get("text_cols") or []))
        }
        drill_count = 0
        for dim in (semantic_plan.get("available_dimensions") or []):
            if drill_count >= 2:
                break
            dc = (dim.get("display_column") or "").upper()
            name = (dim.get("name") or "").strip()
            if not dc or not name:
                continue
            if dc in result_cols_upper:
                continue  # already in the result — skip
            if sql:
                from core.drill_dimension import build_deterministic_drill_sql
                if not build_deterministic_drill_sql(sql, dim, db_type or "azure_sql"):
                    continue
            conf = 75 if dim.get("status") == "approved" else 68
            _add(
                f"drill_dim:{name}",
                _t("chip.drill_dim", name=name),
                conf,
                _t("chip.drill_dim_hint", name=name),
            )
            drill_count += 1

    # ── download_csv — available for any non-empty result ────────────────────
    if row_count >= 1 and mode != "empty":
        _add(
            "download_csv", _t("chip.download_csv"), 85,
            _t_plural("chip.download_csv_hint", row_count),
        )

    # Fixed display order. drill_dim chips slot between contribution and download.
    _fixed = {
        "compare": 0, "diagnose": 1, "compare_prior": 2,
        "contribution": 3,
        "download_csv": 90,
    }
    chips.sort(key=lambda c: (
        _fixed.get(c["id"], 50 if c["id"].startswith("drill_dim:") else 99),
        c["id"],
    ))
    return chips


def _dynamic_actions(ctx: dict) -> list[dict]:
    """Deprecated — delegates to ``compute_chip_eligibility``.

    Kept for backward compatibility with any call sites that haven't been
    updated.  No ``brief`` or ``semantic_plan`` context is available here so
    only mode-level signals are used.
    """
    return compute_chip_eligibility(ctx)


# ── Insight Layer helpers — pure statistics, no LLM call ─────────────────────

def _signed_percent(value, digits: int = 1) -> str:
    """A percentage that states its direction, written for the reader.

    "+12.3%" is an English number: French writes "+12,3 %", with a comma and a
    no-break space before the sign. The sign is prepended rather than left to
    the formatter because a leading "+" is a choice these call sites make and
    a bare format_percent has no opinion about it.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    sign = "+" if number > 0 else ""
    return f"{sign}{_format_percent(number, digits)}"


def _movement_suffix(sentence: str, pct: float | None) -> str:
    """Attach the direction to a finished clause, or just close it.

    The clause goes in whole and comes out whole for the same reason the
    headline's does: French conjugates the direction and there is no seam in
    the middle of the sentence for a translated adverb.
    """
    if pct is None:
        return f"{sentence}."
    if pct > 0:
        return _t("answer.note.up", sentence=sentence,
                  pct=_format_percent(abs(pct), 1))
    if pct < 0:
        return _t("answer.note.down", sentence=sentence,
                  pct=_format_percent(abs(pct), 1))
    return _t("answer.note.unchanged", sentence=sentence)


def _build_insight_summary(
    rows: list[dict],
    ctx: dict,
    brief: dict,
    column_formats: dict | None = None,
    display_formats: dict | None = None,
) -> str:
    """
    Generate a one-sentence plain-English summary from the data brief.

    Purely stat-driven — no LLM call, no latency added.
    Returns empty string when there is not enough structure to say anything useful.
    """
    mode = ctx.get("mode", "table")
    row_count = len(rows)
    column_formats = column_formats or {}
    display_formats = display_formats or {}

    def format_value(value: Any, column: str = "") -> str:
        return _format_display_value(
            value,
            column_formats.get(column),
            display_formats.get(column),
        )

    if detect_zero_match_result(rows):
        return _t("answer.no_match_headline")

    missing_scalar = _single_missing_scalar(rows)
    if missing_scalar:
        return _missing_scalar_copy(missing_scalar[0], ctx.get("question", ""))["headline"]

    null_issue = detect_null_metric_issue(rows)
    if null_issue:
        issue = null_issue["issues"][0]
        metric = _display_label(issue["metric_column"])
        return _t_plural("answer.note.null_metric", null_issue["matched_rows"],
                         metric=metric)

    if mode == "single_value":
        raw_col = brief.get("value_column") or ""
        col = _display_label(raw_col)
        val = brief.get("value", "")
        return (_t("answer.note.single_value", label=col,
                   value=format_value(val, raw_col)) if col else "")

    # A comparison of periods the USER named, which arrives as one row per
    # category with a column per period. Deliberately ahead of the single-wide-
    # row check below: that one recognises CURRENT_x/PREVIOUS_x pairs from the
    # compare_prior chip and has no idea what 2024 and 2025 are.
    _period_note = _period_comparison_summary(
        rows, ctx.get("period_labels"), column_formats, display_formats)
    if _period_note:
        return _period_note

    # A period comparison arrives as ONE wide row (current/previous pairs), not
    # a series. Classified as time_series it narrated "trended flat 0.0% from
    # 2026-03 to 2026-03" -- first and last period of a single-row series are
    # the same cell -- while the table correctly showed 2026-03 $500.00 against
    # 2026-02 $400.00, +25%. Correct data with a contradicting summary is worse
    # than an error: the reader gets no signal to distrust it.
    _cmp = _period_comparison_from_rows(rows)
    if _cmp:
        measure = _display_label(_cmp["measure_column"])
        cur = format_value(_cmp["current_value"], _cmp["measure_column"])
        prev = format_value(_cmp["previous_value"], _cmp["previous_column"])
        sentence = _t("answer.note.period_versus", measure=measure, current=cur,
                      current_period=_cmp["current_period"], previous=prev,
                      previous_period=_cmp["previous_period"])
        return _movement_suffix(sentence, _cmp.get("pct_change"))

    if ctx.get("listing"):
        return _listing_summary(rows, ctx, column_formats, format_value)

    if mode == "time_series":
        ts = brief.get("time_series") or {}
        # No statistics of the series, no sentence about it: none is read across
        # a quantity kept in several units (summarize_result_context), and
        # where the brief did not read the labels as periods the sentence
        # named none -- "Inventory Value est resté stable entre  et .".
        if not ts:
            return ""
        observation_count = int(
            ts.get("observation_count")
            or brief.get("row_count")
            or ctx.get("row_count")
            or 0
        )
        # Two endpoints support a comparison, not a trend claim. Avoid
        # presenting one interval as sustained momentum or decline.
        if observation_count == 1:
            raw_value_col = ctx.get("value_col") or ""
            value_col = _display_label(raw_value_col) or _t("answer.value")
            return _t(
                "answer.note.single_observation", measure=value_col,
                value=format_value(ts.get("first_value"), raw_value_col),
                period=ts.get("first_period") or _t("answer.returned_period"))
        if observation_count == 2:
            raw_value_col = ctx.get("value_col") or ""
            value_col = _display_label(raw_value_col) or _t("answer.value")
            sentence = _t(
                "answer.note.changed_from", measure=value_col,
                first=format_value(ts.get("first_value"), raw_value_col),
                first_period=ts.get("first_period") or _t("answer.first_period"),
                last=format_value(ts.get("last_value"), raw_value_col),
                last_period=ts.get("last_period") or _t("answer.second_period"))
            return _movement_suffix(sentence, ts.get("overall_pct_change"))
        # From here down the sentence is a claim about the WHOLE series — a
        # direction, an overall percentage, a peak. On a truncated result it
        # describes wherever the row cap happened to fall: a daily revenue
        # question capped at 200 rows read "trended down 36.6% from 2025-01-02
        # to 2025-07-20", a window that ends where the preview ends and not
        # where the data does, printed one line under a banner withholding
        # median and quartiles because "computing them over a partial result
        # would give a misleading answer". The single-observation and
        # two-endpoint sentences above are safe by contrast: each describes the
        # rows it was given and claims no momentum.
        _scope = brief.get("result_scope") or ctx.get("result_scope") or {}
        if _scope.get("was_limited"):
            return ""
        direction = ts.get("direction", "stable")
        pct = ts.get("overall_pct_change")
        first = ts.get("first_period", "")
        last_ = ts.get("last_period", "")
        raw_value_col = ctx.get("value_col") or ""
        value_col = _display_label(raw_value_col)
        shape = {"increasing": "up", "decreasing": "down"}.get(direction, "flat")
        if pct is not None:
            base = _t(f"answer.note.trended_{shape}", measure=value_col,
                      pct=_format_percent(abs(pct), 1), first=first, last=last_)
        else:
            base = _t(f"answer.note.remained_{shape}", measure=value_col,
                      first=first, last=last_)
        peak = ts.get("peak") or {}
        if peak and direction in ("increasing", "decreasing"):
            base = _t("answer.note.peak", sentence=base,
                      value=format_value(peak.get("value", 0), raw_value_col),
                      period=peak.get("period", ""))
        return base

    if mode == "ranking":
        cat = brief.get("category_breakdown") or {}
        top5 = cat.get("top_5") or []
        if top5:
            leader = top5[0]
            leader_share = cat.get("leader_share_pct")
            # Pluralised because it follows a count: "across 3 warehouse
            # name" read as broken English the moment the label stopped being
            # the raw "whs nm" and started being words -- and in the reader's
            # language: "sur 3 entrepôts", never "sur 3 warehouses".
            count = cat.get("category_count", row_count)
            label_col = _count_noun(_display_label(cat.get("label_column") or ""), count, "answer.entries")
            # The share as the reader's language writes a number: "92,8 %
            # du total", never "92.8 % du total".
            share_str = (_t("answer.note.leader_share", pct=_format_decimal(leader_share, 1, grouping=False))
                         if leader_share else "")
            sentence = _t(
                "answer.note.leads_across", leader=leader["label"],
                value=format_value(leader["value"], ctx.get("value_col") or ""),
                share=share_str, count=count, label=label_col)
            second = _second_measure_sentence(
                rows, ctx, cat.get("label_column") or "", format_value)
            return f"{sentence} {second}" if second else sentence

    if mode == "numeric_table":
        value_col = _display_label(ctx.get("value_col") or "")
        mn = ctx.get("min_value", 0)
        mx = ctx.get("max_value", 0)
        avg = ctx.get("avg_value", 0)
        return _t_plural(
            "answer.note.range_summary", row_count, measure=value_col,
            low=format_value(mn, ctx.get("value_col") or ""),
            high=format_value(mx, ctx.get("value_col") or ""),
            avg=format_value(avg, ctx.get("value_col") or ""))

    return ""


def _second_measure_sentence(rows: list[dict], ctx: dict, label_col: str, format_value) -> str:
    """The measure the ranking sentence did not narrate.

    "Net sales and returns by warehouse" described the sales -- leader, share,
    count -- and never said the word returns. The second measure gets one
    clause: who leads it, at what value, and with what share when it adds up.
    Period columns were already taken out of numeric_cols by
    _measure_and_label_cols, so a year beside the measure is not "second".
    """
    from core.analysis_contract import measure_class_for_column

    measures = list(ctx.get("numeric_cols") or [])
    if len(measures) < 2 or not label_col:
        return ""
    second = measures[1]
    from core.units_of_measure import kept_in_several_units, rows_per_unit

    # A quantity's totals, one per unit of measure, have no leader: "FT leads
    # Units Sold at 380 (91.1% of total)" followed a ranking of sales value by
    # unit. Kept in several units by another label, it leads with no share.
    if rows_per_unit(rows, label_col, second, str(ctx.get("question") or "")):
        return ""
    items = _ranked_items(rows, label_col, second, limit=max(len(rows), 1))
    if not items:
        return ""
    leader = items[0]
    share = ""
    if (measure_class_for_column(second) == "additive" and not kept_in_several_units(rows, second)
            and not (ctx.get("result_scope") or {}).get("was_limited")):
        total = sum(item["value"] for item in items)
        if total > 0:
            share = _t("answer.note.leader_share",
                       pct=_format_decimal(leader["value"] / total * 100, 1, grouping=False))
    return _t("answer.note.second_measure", leader=leader["label"],
              measure=_display_label(second),
              value=format_value(leader["value"], second), share=share)


def _listing_summary(rows: list[dict], ctx: dict, column_formats: dict, format_value) -> str:
    """What a set of records adds up to.

    The count, and the total of the measure that adds up -- the currency
    column when there is one, because invoice lines carry a quantity and a
    unit price beside the amount and the amount is what the reader came for.
    A listing whose numbers do not add up (a rate, a percentage) gets the
    range sentence a numeric table already gets.
    """
    from core.analysis_contract import measure_class_for_column
    from core.chart_spec import _format_for_column

    measures = list(ctx.get("numeric_cols") or [])
    if not measures:
        return ""
    from core.units_of_measure import kept_in_several_units

    # A quantity the records keep in several units has no total to state.
    additive = [
        c for c in measures
        if measure_class_for_column(c) == "additive"
        and not kept_in_several_units(rows, c, measure_format=str((column_formats or {}).get(c) or ""))
    ]
    # The same name rule the chart uses to tell an amount from a count, so
    # the sentence and the axis agree on which column is the money.
    currency = [
        c for c in additive
        if (column_formats or {}).get(c) == "currency"
        or _format_for_column(c, {}) == "currency"
    ]
    count = len(rows)
    chosen = (currency or additive or [None])[0]
    if chosen is None:
        # Nor a range or an average across units.
        ranged = [
            c for c in measures
            if not kept_in_several_units(rows, c, measure_format=str((column_formats or {}).get(c) or ""))
        ]
        if not ranged:
            return ""
        col = ranged[0]
        values = [_to_float_z(r.get(col)) for r in rows]
        return _t_plural(
            "answer.note.range_summary", count, measure=_display_label(col),
            low=format_value(min(values), col), high=format_value(max(values), col),
            avg=format_value(mean(values), col))
    values = [_to_float_z(r.get(chosen)) for r in rows]
    return _t_plural(
        "answer.note.listing_total", count, measure=_display_label(chosen),
        total=format_value(sum(values), chosen),
        low=format_value(min(values), chosen), high=format_value(max(values), chosen))


def _build_anomaly_callouts(brief: dict) -> list[dict]:
    """
    Detect notable statistical patterns from the data brief.

    Returns a list of up to 3 callout dicts:
      {"type": str, "icon": str, "message": str, "severity": "warning"|"success"|"info"}

    Severity → UI colour:
      warning  = amber   (drops, streaks)
      success  = green   (gains)
      info     = blue    (concentration, outliers)
    """
    callouts: list[dict] = []
    mode = brief.get("mode", "table")

    if mode == "time_series":
        ts = brief.get("time_series") or {}
        if int(ts.get("period_count") or brief.get("row_count") or 0) < 3:
            # One point has no interval; two points have one comparison. Do
            # not label that single interval an anomaly or a sustained move.
            return []
        drop = ts.get("biggest_period_drop") or {}
        gain = ts.get("biggest_period_gain") or {}
        streak = ts.get("longest_decline_streak", 0)

        if drop.get("pct_change") is not None and drop["pct_change"] < -10:
            callouts.append({
                "type": "drop", "icon": "↓",
                "message": _t("answer.callout.biggest_drop",
                              old=drop["from_period"], new=drop["to_period"],
                              pct=_signed_percent(drop["pct_change"])),
                "severity": "warning",
            })
        if gain.get("pct_change") is not None and gain["pct_change"] > 10:
            callouts.append({
                "type": "gain", "icon": "↑",
                "message": _t("answer.callout.biggest_gain",
                              old=gain["from_period"], new=gain["to_period"],
                              pct=_signed_percent(abs(gain["pct_change"]))),
                "severity": "success",
            })
        if streak >= 3:
            callouts.append({
                "type": "streak", "icon": "⚠",
                "message": _t_plural("answer.callout.decline_streak", streak),
                "severity": "warning",
            })

    elif mode == "ranking":
        cat = brief.get("category_breakdown") or {}
        # The COLLAPSED category share, and only when there are enough
        # categories for "the top three" to mean anything.
        #
        # This read numeric_summaries[col]["top_3_concentration_pct"] -- the top
        # three ROWS of the raw result -- which core/insight.py had already
        # replaced for the breakdown itself and documented as wrong. It stayed
        # wrong here, and in both directions. Five warehouses over three months:
        # the three leading warehouses hold 91.9% of the total and the callout
        # said nothing at all, because the top three ROWS were three months of
        # one warehouse and came to 45%. Three warehouses: it announced "100.0%
        # of total — highly concentrated", which is what the top three of three
        # always add up to.
        conc = cat.get("top_3_share_pct")
        categories = cat.get("category_count") or 0
        if (conc and conc >= 80
                and categories >= MIN_CATEGORIES_FOR_CONCENTRATION):
            callouts.append({
                "type": "concentration", "icon": "◉",
                "message": _t("answer.callout.concentration", pct=_format_decimal(conc, 1, grouping=False)),
                "severity": "info",
            })
        leader_share = cat.get("leader_share_pct")
        top5 = cat.get("top_5") or []
        if leader_share and leader_share >= 50 and top5 and len(callouts) < 2:
            callouts.append({
                "type": "dominance", "icon": "★",
                "message": _t("answer.callout.dominance",
                              label=top5[0]["label"], pct=_format_decimal(leader_share, 1, grouping=False)),
                "severity": "info",
            })

    # Outlier detection across numeric columns (all modes)
    if len(callouts) < 3:
        for col, stats in (brief.get("numeric_summaries") or {}).items():
            std = stats.get("std_dev")
            mean_v = stats.get("mean")
            mx_v = stats.get("max")
            if std and mean_v and std > 0 and mx_v and mx_v > mean_v + 2.5 * std:
                callouts.append({
                    "type": "outlier", "icon": "◆",
                    "message": _t("answer.callout.outlier",
                                  column=_display_label(col),
                                  high=_format_number(mx_v),
                                  avg=_format_number(mean_v)),
                    "severity": "info",
                })
                break

    return callouts[:3]


def _build_decision_signal(ctx: dict, brief: dict, anomaly_callouts: list[dict]) -> dict:
    """
    Deterministic 'so-what' line — zero LLM, zero latency.

    Turns the existing statistical brief + anomaly callouts into one
    decision-oriented sentence the user can act on, plus a tone for UI colour.

    Returns:
        {"line": str, "tone": "watch"|"positive"|"neutral", "basis": str}
        or {} when there is nothing decision-relevant to say.
    """
    mode = brief.get("mode") or ctx.get("mode", "table")

    # A named-period comparison is ranked by MOVEMENT, and every sentence below
    # is about a level: "X alone holds 62% of the total" computed over the
    # oldest period's column would read as a claim about today.
    if ctx.get("period_labels"):
        return {}

    if mode == "ranking":
        cat = brief.get("category_breakdown") or {}
        # Every share below divides by the sum of the rows that CAME BACK
        # (core/insight.py builds category_breakdown["total"] from `paired`).
        # On a TOP 5 that denominator is the top-5 subtotal, so "20.2% of
        # total" is a share of the five biggest rows and "broadly diversified"
        # is a tautology of the user's own filter — near-equal shares among the
        # five largest is what a top-5 always looks like. The dominance line is
        # worse: it warns about a single point of dependence computed over a
        # population it never saw. The ranking explanation further down already
        # suppresses its runner-up clause on `is_top_n` for exactly this
        # reason; this chain never asked.
        scope = brief.get("result_scope") or ctx.get("result_scope") or {}
        if scope.get("was_limited"):
            return {}
        leader_share = cat.get("leader_share_pct")
        # Same field and same floor as the callout above. This line is the
        # advisory signal, so it was the most expensive place to read the raw
        # rows: on five warehouses over three months it told the reader "volume
        # is spread across the field — broadly diversified" while three of the
        # five held 91.9% of it.
        conc = cat.get("top_3_share_pct")
        if (cat.get("category_count") or 0) < MIN_CATEGORIES_FOR_CONCENTRATION:
            conc = None
        top5 = cat.get("top_5") or []
        leader = top5[0]["label"] if top5 else ""
        if conc is not None and conc >= 80:
            return {
                "line": _t("answer.signal.concentration", pct=f"{conc:.0f}"),
                "tone": "watch", "basis": "concentration",
            }
        if leader_share is not None and leader_share >= 50:
            return {
                "line": _t("answer.signal.dominance", leader=leader,
                           pct=f"{leader_share:.0f}"),
                "tone": "watch", "basis": "dominance",
            }
        if leader_share is not None:
            # "no single entry exceeds {pct}%" is an upper BOUND, not an
            # approximation, so it has to round up. With :.0f a 20.2% leader
            # printed "no single entry exceeds 20%" directly above its own
            # "20.2% of total" bullet — the card contradicting itself by a
            # rounding mode.
            return {
                "line": _t("answer.signal.spread",
                           pct=f"{math.ceil(max(leader_share, 1))}"),
                "tone": "positive", "basis": "spread",
            }

    if mode == "time_series":
        ts = brief.get("time_series") or {}
        period_count = int(ts.get("period_count") or brief.get("row_count") or 0)
        if period_count < 3:
            return {}
        # Every line below is a claim about the WHOLE series -- a direction,
        # an overall percentage. The insight sentence withholds those on a
        # truncated result (03bb5b6) and the ranking branch above checks the
        # same flag; this branch never did, so a daily series cut at the row
        # cap lost its trend sentence and kept "Sustained downward trend
        # (-12% overall)" two lines under it.
        scope = brief.get("result_scope") or ctx.get("result_scope") or {}
        if scope.get("was_limited"):
            return {}
        direction = ts.get("direction", "stable")
        pct = ts.get("overall_pct_change")
        streak = ts.get("longest_decline_streak", 0)
        if direction == "decreasing" and streak >= 3:
            return {
                "line": (_t("answer.signal.decline_pct", pct=_signed_percent(pct, 0))
                         if pct is not None else _t("answer.signal.decline")),
                "tone": "watch", "basis": "decline",
            }
        if direction == "increasing" and pct is not None and pct >= 10:
            return {
                "line": _t("answer.signal.growth", pct=_signed_percent(abs(pct), 0)),
                "tone": "positive", "basis": "growth",
            }
        if direction == "stable":
            return {
                "line": _t("answer.signal.stable"),
                "tone": "neutral", "basis": "stable",
            }

    if mode == "numeric_table":
        outliers = [c for c in anomaly_callouts if c.get("type") == "outlier"]
        if outliers:
            return {
                "line": _t("answer.signal.outlier"),
                "tone": "watch", "basis": "outlier",
            }

    if mode == "single_value":
        # Restate with directional framing only if a comparison exists.
        comp = ctx.get("comparison") or brief.get("comparison")
        if comp:
            return {"line": _t("answer.signal.single", comparison=comp),
                    "tone": "neutral", "basis": "single"}

    return {}


def _why_it_matters(ctx: dict) -> str:
    mode = ctx.get("mode")
    if mode == "time_series":
        pct = ctx.get("pct_change")
        if pct is None:
            return _t("analysis.why.unstable_base")
        # A sentence per direction, not "{pct} {direction} than": French does
        # not build a comparative that way, and "flat" is not a comparative at
        # all -- the English sentence reads "0.0% flat than the starting
        # period", which nobody would have noticed until it was translated.
        shape = "higher" if pct > 0 else "lower" if pct < 0 else "flat"
        return _t(f"analysis.why.{shape}", pct=_format_percent(abs(pct), 1))
    if mode == "ranking":
        top_items = ctx.get("top_items") or []
        if len(top_items) >= 2:
            gap = top_items[0]["value"] - top_items[1]["value"]
            return _t("analysis.why.gap", gap=_format_number(gap))
        return _t("analysis.why.leader_only")
    if mode == "numeric_table":
        return _t("analysis.why.spread")
    if mode == "empty":
        return _t("analysis.why.empty")
    return _t("analysis.why.starting_point")


def build_analysis_response(action: str, contract: dict) -> dict:
    """
    Synchronous fallback for action button clicks when LLM insight is unavailable.

    The preferred path is the async generate_analysis_response() below, which
    uses the LLM insight engine. This function is kept as a zero-latency
    fallback that works without an LLM call.

    Every sentence here is a whole message id. The scope phrase is why: this
    card wrote `scope["badge"].lower()` into the middle of a sentence, which
    reads as English and cannot work in French, where an inline noun phrase
    needs its article and the article carries the gender. infer_result_scope
    publishes scope["inline"] for exactly this position.
    """
    mode = contract.get("mode")
    scope = contract.get("result_scope") or {}
    scope_inline = scope.get("inline") or _t("answer.scope.returned.inline")
    title = _t("analysis.title.default")
    body = ""
    bullets: list[str] = []
    secondary = scope.get("analysis_note", "")

    if action == "explain":
        title = _t("analysis.title.explain_result")
        if mode == "time_series":
            body = _t(
                "analysis.explain.series", scope=scope_inline,
                period=contract.get("last_label") or _t("analysis.latest_period"),
                value=_format_number(contract.get("last_value", 0.0)),
            )
            pct = contract.get("pct_change")
            if pct is not None:
                shape = "up" if pct > 0 else "down" if pct < 0 else "flat"
                bullets.append(_t(f"analysis.explain.direction_{shape}",
                                  pct=_format_percent(abs(pct), 1)))
        elif mode == "ranking":
            top_items = contract.get("top_items") or []
            if top_items:
                body = _t("analysis.explain.ranking", scope=scope_inline,
                          leader=top_items[0]["label"],
                          value=_format_number(top_items[0]["value"]))
                if len(top_items) > 1 and not scope.get("is_top_n"):
                    body = _t("analysis.explain.runner_up", sentence=body,
                              runner_up=top_items[1]["label"],
                              value=_format_number(top_items[1]["value"]))
        elif mode == "numeric_table":
            body = _t_plural(
                "analysis.explain.numeric", contract.get("row_count", 0),
                low=_format_number(contract.get("min_value", 0.0)),
                high=_format_number(contract.get("max_value", 0.0)),
            )
        else:
            body = _t("analysis.explain.concise")

    elif action == "analyze":
        title = _t("analysis.title.detailed")
        if mode == "time_series":
            body = _t("analysis.detail.series",
                      low=_format_number(contract.get("min_value", 0.0)),
                      high=_format_number(contract.get("max_value", 0.0)),
                      average=_format_number(contract.get("avg_value", 0.0)))
            bullets = [
                _t("analysis.detail.avg_step",
                   value=_format_number(contract.get("avg_step_change", 0.0))),
                _t("analysis.detail.volatility",
                   value=_format_number(contract.get("volatility", 0.0))),
            ]
        elif mode == "ranking":
            stats = contract.get("distribution_stats") or {}
            if stats.get("top_3_share_pct") is not None:
                body = _t("analysis.detail.concentrated",
                          pct=_format_percent(stats["top_3_share_pct"], 1))
            else:
                body = _t("analysis.detail.distribution")
            bullets = [
                _t("analysis.detail.category_count",
                   count=stats.get("category_count", contract.get("row_count", 0))),
            ]
            # None for rows in several units: a spread across them is no figure,
            # and a spread of "0" would say the values were all the same.
            if stats.get("spread") is not None:
                bullets.append(_t("analysis.detail.spread_range", value=_format_number(stats["spread"])))
            if stats.get("std_dev") is not None:
                bullets.append(_t("analysis.detail.std_dev",
                                  value=_format_number(stats["std_dev"])))
        elif mode == "numeric_table":
            body = _t_plural(
                "analysis.detail.numeric", contract.get("row_count", 0),
                average=_format_number(contract.get("avg_value", 0.0)),
            )
            bullets = [
                _t("analysis.detail.spread", value=_format_number(
                    (contract.get("distribution_stats") or {}).get("spread", 0.0))),
                _t("analysis.detail.median",
                   value=_format_number(contract.get("median_value", 0.0))),
            ]
        else:
            body = _t("analysis.detail.not_enough")

    elif action == "compare":
        title = _t("analysis.title.comparison")
        if mode == "time_series":
            cmp = contract.get("comparison_stats") or {}
            body = _t(
                "analysis.compare.series",
                last_period=cmp.get("last_period") or _t("answer.latest_period"),
                last_value=_format_number(cmp.get("last_value", 0.0)),
                first_value=_format_number(cmp.get("first_value", 0.0)),
                first_period=cmp.get("first_period") or _t("analysis.first_period"),
            )
            if cmp.get("pct_change") is not None:
                bullets.append(_t("analysis.compare.pct_change",
                                  pct=_format_percent(abs(cmp["pct_change"]), 1)))
        elif mode == "ranking":
            cmp = contract.get("comparison_stats") or {}
            if cmp.get("leader") and cmp.get("runner_up"):
                body = _t("analysis.compare.leader", leader=cmp["leader"],
                          runner_up=cmp["runner_up"],
                          gap=_format_number(cmp.get("gap", 0.0)))
                if cmp.get("leader_share_pct") is not None:
                    bullets.append(_t("analysis.compare.leader_share",
                                      pct=_format_percent(cmp["leader_share_pct"], 1)))
            elif cmp.get("leader"):
                body = _t("analysis.compare.only_one", leader=cmp["leader"])
            else:
                body = _t("analysis.compare.not_comparable")
        else:
            body = _t("analysis.compare.not_yet")

    elif action == "why":
        title = _t("analysis.title.framing")
        body = _why_it_matters(contract)
        bullets = [_t("analysis.why.caveat")]

    elif action == "predict":
        title = _t("analysis.title.forecast")
        if mode == "time_series" and contract.get("row_count", 0) >= 3:
            vals = contract.get("values") or []
            labels = contract.get("labels") or []
            xs = list(range(len(vals)))
            n = len(xs)
            mean_x = sum(xs) / n
            mean_y = sum(vals) / n
            denom = sum((x - mean_x) ** 2 for x in xs) or 1.0
            slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, vals)) / denom
            intercept = mean_y - slope * mean_x
            next_x = n
            forecast = intercept + slope * next_x
            forecast = max(forecast, 0.0)
            vol = contract.get("volatility", 0.0)
            conf = "low" if vol > abs(slope) * 3 else "medium" if vol > abs(slope) else "moderate"
            body = _t("analysis.predict.projection", value=_format_number(forecast))
            secondary = _t(f"analysis.predict.confidence_{conf}")
            bullets = [
                _t("analysis.predict.last_observed", period=labels[-1],
                   value=_format_number(vals[-1])),
                _t("analysis.predict.step_used",
                   value=_format_number(contract.get("avg_step_change", 0.0))),
            ]
        else:
            body = _t("analysis.predict.needs_series")

    elif action == "decide":
        title = _t("analysis.title.next_step")
        # Deterministic advisory fallback (no LLM). Reuse the decision-signal
        # rules so the static path still gives a useful, safe recommendation.
        signal = _build_decision_signal(contract, contract, [])
        body = signal.get("line") or _t("analysis.decide.starting_point")
        bullets = [_t("analysis.decide.finding"), _t("analysis.decide.caveat")]
        secondary = scope.get("note") or _t("analysis.decide.based_on")

    else:
        title = _t("analysis.title.default")
        body = _t("analysis.unsupported")

    # A quantity's totals, one per unit of measure (summarize_result_context):
    # whatever was asked of them, there is no leader, gap, spread or share
    # across them, only the totals themselves.
    per_unit = contract.get("per_unit") or {}
    if per_unit.get("totals") and action in {"explain", "analyze", "compare", "why"}:
        body = _t("analysis.per_unit", values=_per_unit_listing(
            per_unit["totals"], per_unit.get("missing") or [], lambda value, _column: _format_number(value), ""))
        bullets = []

    next_step = ""
    if action == "decide":
        next_step = _t("analysis.decide.next_step")

    return {
        "type": "assistant_analysis",
        "action": action,
        "title": title,
        "body": body,
        "secondary": secondary,
        "bullets": bullets,
        "next_step": next_step,
        "source_question": contract.get("question", ""),
        "mode": mode,
        "result_scope": scope,
    }


def _leading_quantity_per_unit(rows: list[dict]) -> tuple[list[tuple[str, float]], list[str]] | None:
    """The leading measure's total in each unit of measure, where it is a
    quantity the rows keep in several: nothing else about it is a figure of
    anything -- its spread, its leader or its evenness."""
    from core.units_of_measure import is_unit_column, kept_in_several_units, per_unit_totals

    unit_column = next((column for column in rows[0] if is_unit_column(column)), "") if rows else ""
    # The first quantity the rows keep in several units, not the first number:
    # a date key or a count beside it is no quantity of goods.
    measure = next((column for column in _numeric_cols(rows)
                    if column != unit_column and kept_in_several_units(rows, column)), "")
    if not unit_column or not measure:
        return None
    found = per_unit_totals(rows, unit_column, measure)
    return found if found and found[0] else None


def _regulated_analysis_fallback(action: str, rows: list[dict] | None = None) -> dict:
    """The analysis a regulated tenant gets — computed locally, no LLM.

    This used to be a static title and a paragraph explaining why there was no
    analysis, which meant the tenants we most want to serve received the
    weakest version of the product: a table and an apology.

    The findings come from ``core.analysis_evidence``, which runs the
    deterministic analysers over the rows the user is already looking at, and
    the sentences from ``core.analysis_narrative``, which phrases them from
    the message catalogue. No model is called, no row leaves the process, and
    the same rows produce the same sentences every time — so the summary is
    reproducible for an auditor, which the LLM-written version never was.

    Labels are kept: a category leader named here is a name already on the
    user's screen in the result above, and these rows have been through
    ``result_guard`` like every other released row. The label-free form is for
    the egress boundary (``core.analysis_evidence.redact_labels``), not for
    the reader.

    Falls back to the original explanatory paragraph if evidence cannot be
    built — logged at warning, since a silent degradation here is
    indistinguishable from the feature working.
    """
    # A quantity in several units of measure: its totals, one per unit, first,
    # and nothing read across them -- the evidence reads nothing there either.
    # What it finds in the other measures stands beside them: the money of
    # the same rows adds up across units.
    per_unit = _leading_quantity_per_unit(rows or [])
    per_unit_sentence = _t("analysis.per_unit", values=_per_unit_listing(
        per_unit[0], per_unit[1], lambda value, _column: _format_number(value), "")) if per_unit else ""
    if rows:
        try:
            from core.analysis_evidence import build_evidence
            from core.analysis_narrative import build_narrative

            narrative = build_narrative(build_evidence(rows))
            found = list(narrative.sentences) if narrative.finding_kinds or not per_unit_sentence else []
            sentences = [per_unit_sentence] * bool(per_unit_sentence) + found
            if sentences:
                return {
                    "type": "assistant_analysis",
                    "action": action,
                    "title": narrative.title or _t("narrative.title"),
                    "body": " ".join(sentences),
                    "bullets": sentences if found else [],
                    "secondary": _t("narrative.no_values_note"),
                    "computed": True,
                    "evidence_id": narrative.evidence_id,
                    "finding_kinds": list(narrative.finding_kinds),
                    "rows_sent_to_llm": 0,
                }
        except Exception as exc:
            log.warning("computed narrative unavailable for %r: %s", action, exc)

    return {
        "type": "assistant_analysis",
        "action": action,
        "title": _t("analysis.title.unavailable"),
        "body": _t("analysis.regulated_body"),
        "bullets": [],
        "computed": False,
        "rows_sent_to_llm": 0,
    }


async def generate_analysis_response(
    action: str,
    rows: list[dict],
    question: str,
    provider: str,
    model: str,
    api_key: str,
    account_id: str,
    follow_up: str = "",
    original_sql: str = "",
    db_cfg: dict | None = None,
    context: str = "",
    known_tables: set[str] | None = None,
    query_executor=None,
    # Explicit, NOT via extra_kwargs: generate_drilldown_insight forwards
    # **extra_kwargs straight into llm_complete, so an unexpected key there
    # raises TypeError rather than being ignored.
    grounding: dict | None = None,
    **extra_kwargs,
) -> dict:
    """
    Async LLM-powered analysis — the preferred path for action buttons
    and "why" follow-up questions.

    Falls back to the synchronous build_analysis_response() if the LLM
    call fails. Regulated tenants never reach the LLM at all: they get
    _regulated_analysis_fallback, which computes the analysis locally from the
    same rows. The model does not see `rows` for them, and does not see a
    summary of them either.
    """
    from core.compliance.policy_engine import result_llm_features_allowed
    if not result_llm_features_allowed(account_id):
        from core.llm_audit import record_llm_blocked
        record_llm_blocked(
            "analysis",
            f"action={action!r} blocked — regulated tenant, LLM never received result rows. "
            f"Analysis computed locally from {len(rows or [])} released rows.",
        )
        return _regulated_analysis_fallback(action, rows)

    from core.insight import (
        generate_insight,
        generate_drilldown_insight,
        is_insight_question,
    )

    try:
        # A "why" that cannot drill has to say so.
        #
        # The gate below needs all three of db_cfg, original_sql and context,
        # and `context` is the tenant's KB text -- which the adapter blanks
        # whenever a result is restored from a snapshot rather than answered
        # fresh (gateway/web_adapter.py::adopt_cached_snapshot, correctly: the
        # previous answer's context is not this result's). So after a page
        # reload, "why did this happen?" ran ZERO warehouse queries and still
        # returned a card titled "Why this pattern?" that read exactly like one
        # written over three real breakdowns.
        _why_gap = ""
        if action == "why" and not (db_cfg and original_sql and context):
            _why_gap = "no_context"

        # "why" questions with drill-down capability
        if action == "why" and db_cfg and original_sql and context:
            return await generate_drilldown_insight(
                rows=rows,
                question=question,
                follow_up=follow_up,
                original_sql=original_sql,
                db_cfg=db_cfg,
                context=context,
                provider=provider,
                model=model,
                api_key=api_key,
                known_tables=known_tables,
                business_context=context,
                query_executor=query_executor,
                **extra_kwargs,
            )

        # Standard action buttons (explain, analyze, compare, predict) -- and a
        # "why" that had nothing to drill with.
        return await generate_insight(
            rows=rows,
            question=question,
            action=action,
            follow_up=follow_up,
            provider=provider,
            model=model,
            api_key=api_key,
            business_context=context,
            original_sql=original_sql,
            grounding=grounding,
            drilldown_gap=_why_gap,
            **extra_kwargs,
        )

    except Exception as e:
        log.error("Dynamic analysis failed, falling back to static: %s", e)
        # Fall back to synchronous/static analysis
        ctx = summarize_result_context(rows, question, sql=original_sql)
        return build_analysis_response(action, ctx)


def build_assistant_response(
    *,
    question: str,
    rows: list[dict],
    sql: str,
    duration_ms: int,
    chart: dict | None = None,
    data_source: str | None = None,
    confidence: dict | None = None,
    display_context: dict | None = None,
    column_formats: dict | None = None,
    display_formats: dict | None = None,
    semantic_plan: dict | None = None,
    question_id: str = "",
) -> dict:
    from core.insight import compute_data_brief
    from core.clarification import extract_display_question
    display_question = extract_display_question(question).strip() or question
    display_chart = dict(chart) if isinstance(chart, dict) else chart
    if isinstance(display_chart, dict):
        if "title" in display_chart:
            display_chart["title"] = display_question
        if "question" in display_chart:
            display_chart["question"] = display_question
    raw_rows = list(rows)
    zero_match = detect_zero_match_result(raw_rows)
    null_issue = detect_null_metric_issue(raw_rows)
    visible_rows = visible_result_rows(raw_rows)
    # Periods the SQL listed by a figure are a ranking of them, kept in the
    # order they came (_ordered_by_a_figure); so are those a ranking was asked
    # of and a subquery or a window cut to the top of a figure.
    ranked_by_a_figure = _ranked_by_a_figure(display_question, sql, visible_rows, data_source or "azure_sql")
    analysis_rows = list(visible_rows) if ranked_by_a_figure else _chronological_analysis_rows(visible_rows)
    # The grain the periods were asked at, read once: the table, the headline
    # and every sentence name a period by it, so a month is "March 2025" in
    # all three (period_grain).
    grain = requested_period_grain(display_question, semantic_plan)
    ctx = summarize_result_context(analysis_rows, display_question, sql=sql, asked_grain=grain)
    # The periods the user named, published by the pipeline once the widened
    # result actually arrived. Empty for every other answer in the product, and
    # every branch that reads it falls through to its previous behaviour.
    _period_labels = [
        str(label) for label in
        (((display_context or {}).get("period_comparison") or {}).get("labels") or [])
    ]
    if _period_labels:
        ctx["period_labels"] = _period_labels
    result_operation = str((display_context or {}).get("result_operation") or "")
    if ((result_operation in {"keep_top", "sort", "contribution"} or ranked_by_a_figure)
            and ctx.get("mode") == "time_series" and not _period_labels):
        # Period labels sorted by a measure are a ranking, not a chronology.
        # Treating them as a series creates false trend claims from sort order.
        # Not applied to a named-period comparison: its periods are COLUMNS, so
        # the downgrade would relabel a change result as a leaderboard.
        ctx["mode"] = "ranking"
        ctx["result_scope"] = infer_result_scope(visible_rows, display_question, sql, mode="ranking")
    resolved_column_formats = build_column_formats(
        visible_rows,
        display_context={**(display_context or {}), "period_grain": grain},
        explicit_formats=column_formats,
    )
    headers: list[str] = list(visible_rows[0].keys()) if visible_rows else []
    resolved_display_formats = build_display_formats(
        visible_rows, resolved_column_formats, explicit=display_formats, requested_grain=grain)
    if isinstance(display_chart, dict):
        # The chart's time axis names its periods as the table does: the
        # second quarter is "Q2", not "Apr".
        x_style = chart_axis_style(
            visible_rows, str(display_chart.get("x_key") or ""), resolved_column_formats, resolved_display_formats)
        if x_style:
            display_chart["x_style"] = x_style
    answer_rows = raw_rows if (zero_match or null_issue) else analysis_rows
    answer = build_answer(
        answer_rows,
        display_question,
        ctx.get("result_scope"),
        column_formats=resolved_column_formats,
        display_formats=resolved_display_formats,
        period_labels=_period_labels,
        asked_grain=grain,
    )
    brief = compute_data_brief(
        analysis_rows,
        display_question,
        result_scope=ctx.get("result_scope"),
        context=ctx,
        column_formats=resolved_column_formats,
        asked_grain=grain,
    )

    # ── Insight Layer — pure-stats, zero-latency ─────────────────────────────
    # Generate a summary sentence and anomaly callouts from the data brief.
    # These are computed entirely from statistics — no LLM call, no extra latency.
    insight_summary = _build_insight_summary(
        raw_rows if (zero_match or null_issue) else analysis_rows,
        ctx,
        brief,
        resolved_column_formats,
        resolved_display_formats,
    )
    anomaly_callouts = _build_anomaly_callouts(brief)
    decision_signal  = _build_decision_signal(ctx, brief, anomaly_callouts)

    # Include the actual row data (bounded) so the frontend can render a table.
    # Frontend is the ONLY consumer of raw rows — LLM insight path never sees these.
    # We cap at 200 rows to keep WebSocket payload reasonable; full set is already
    # limited by run_query(max_rows=200).
    display_rows: list[dict] = []
    if visible_rows:
        # Send formatted string values for reliable frontend display -- but a
        # period as its key: 202501 grouped as "202,501" is no date the
        # browser can name, and the cell showed that number.
        periods = {header for header, fmt in resolved_column_formats.items() if fmt == "date"}
        keys = {header for header, fmt in resolved_column_formats.items() if fmt == "text"}
        for r in visible_rows[:_PREVIEW_ROW_CAP]:
            display_rows.append({h: _period_cell(r.get(h)) if h in periods
                                 else _text_cell(r.get(h)) if h in keys else _safe_cell(r.get(h))
                                 for h in headers})
    kpi = _build_kpi_payload(
        visible_rows,
        resolved_column_formats,
        resolved_display_formats,
        zero_match=zero_match,
        null_issue=null_issue,
    )

    payload = {
        "type": "assistant_response",
        "question": display_question,
        "answer": answer,
        "chart": display_chart,
        "kpi": kpi,
        "insight_summary": insight_summary,
        "anomaly_callouts": anomaly_callouts,
        "decision_signal": decision_signal,
        "summary": {"executive_summary": ""},
        "next_actions": compute_chip_eligibility(
            ctx,
            brief=brief,
            semantic_plan=semantic_plan,
            sql=sql,
            db_type=data_source or "azure_sql",
        ),
        "analysis_contract": ctx,
        "data_brief": brief,
        "result_scope": ctx.get("result_scope", {}),
        "data": {
            "headers": headers,
            # The same headers in the tenant's own words, for the <th> the
            # reader looks at. The card said "Warehouse Name" in its prose and
            # "WHS_NM" in the table header directly beneath it.
            #
            # A separate map rather than a replacement: "headers" is the KEY
            # into every row dict, the column-format lookup and the CSV export.
            # Renaming it would break the cells, not just relabel them.
            "header_labels": {column: _display_label(column) or column
                              for column in headers},
            "rows": display_rows,
            "total_rows": len(visible_rows),
            "truncated": len(visible_rows) > _PREVIEW_ROW_CAP,
            "column_formats": resolved_column_formats,
            "display_formats": resolved_display_formats,
            "diagnostics": _result_diagnostics(raw_rows),
            "currency_columns": [
                col for col, fmt in resolved_column_formats.items()
                if fmt == "currency"
            ],
        },
        "trust": {
            "sql": sql,
            "row_count": len(raw_rows),
            "duration_label": f"{duration_ms}ms" if duration_ms < 1000 else f"{duration_ms/1000:.1f}s",
            "data_source": data_source or "",
            "scope_badge": ctx.get("result_scope", {}).get("badge", ""),
            "confidence": confidence or {},
            "question_id": question_id,   # public key for feedback API (B3)
            "date_context": list((semantic_plan or {}).get("date_disclosures") or []),
        },
        "confidence": confidence or {},
    }
    return sanitize_response_text_fields(payload)


def _period_cell(val: Any) -> str:
    """A period cell as the day its period starts, ISO-written: 2025-03-01
    for 202503, "Mar-25" and March 2025, 2021-01-01 for the year 2021. The
    browser names it in the column's style, and reads no period of its own: a
    bare 2021 it took for midnight UTC, the last day of 2020 west of
    Greenwich, and "Jan-25" for January 2001. Anything else as any cell."""
    part = _period_value(val)
    return part[0].isoformat() if part else _safe_cell(val)


def _text_cell(val: Any) -> str:
    """A cell shown as text: a key as its digits, never grouped."""
    if isinstance(val, Decimal) and val.is_finite() and val == val.to_integral_value():
        return str(int(val))
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    if isinstance(val, int) and not isinstance(val, bool):
        return str(val)
    return _safe_cell(val)


def _safe_cell(val: Any) -> str:
    """Format a cell value for frontend display. Returns a string."""
    if val is None:
        return ""
    if isinstance(val, float):
        if val != val or val in (float("inf"), float("-inf")):
            return ""
        if val.is_integer():
            return f"{int(val):,}"
        return f"{val:,.4f}".rstrip("0").rstrip(".")
    if isinstance(val, int):
        return f"{val:,}" if abs(val) >= 1000 else str(val)
    if isinstance(val, (dict, list, tuple)):
        try:
            return json.dumps(val, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return ""
    return str(val)
