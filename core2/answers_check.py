"""The client's own questions, checked against the figures in their own reports.

The setup check's last tab (core2/setup_check.py) keeps the team's real questions with the answer their own report
gives ("How many new orders did each region take in Q2?" -> "North: 41, South: 37"). Each is asked here as a reader
would ask it (core2.service.portal_answer), and every figure the report gives is looked for in the answer, read in
code, never by the AI: a figure written "1.2M" matches anything that rounds to it, "North: 41" is looked for on the
answer's North row. Each question is then

    matches     every figure is the report's, as the report rounds it
    close       within 1% (or one step of the report's rounding): worth a look, not yet wrong
    different   a figure is not the report's, or a member the report names is not in the answer
    unanswered  Solviz did not answer (it asked back, refused, or could not reach the database)
    unchecked   the report's answer holds no figure to compare (words only, or nothing written)

A run asks the questions afresh with the workspace's AI ("ask"), or draws the plans of the last answers again on
the data and the decisions as they are now ("replay": no AI, so it runs after every change of decisions). The
minute scheduler (core/notification_scheduler.py) starts one when it is due: everything asked again after a Learn,
replayed once decisions have stopped changing for a while, and a new question asked as it arrives.

    expected_figures(text)        -> the figures a report's answer gives
    compare(expected, payload)    -> the verdict on one answer
    run(account_id, mode, ...)    -> one run, stored; ``start`` runs it in the background
    run_due_answer_checks()       -> the runs due now, every workspace
"""

from __future__ import annotations

import datetime as dt
import itertools
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

log = logging.getLogger("querybot.core2.answers_check")

VERDICTS = ("matches", "close", "different", "unanswered", "unchecked")
_WORST = {"matches": 0, "close": 1, "different": 2}

# Decisions changed this long ago, with none since, before their answers are drawn again: an admin making several
# changes in a row gets one run, after the last.
SETTLE_SECONDS = 120


# ── the report's figures ────────────────────────────────────────────────────

@dataclass
class Figure:
    """One figure a report gives: its value, the member or period it is for ("North", "Q2"), whether it is a
    percentage or money, and its rounding (1 for "41", 0.1 for "12.5%", 100,000 for "1.2M")."""

    value: float
    label: str = ""
    percent: bool = False
    money: bool = False
    step: float = 1.0
    text: str = ""
    carried: bool = False           # the label is the figure before's ("North: $1.2M, 12%"), not its own
    restates: bool = False          # the figure before said again in full ("$3.09M ($3,093,601)")


_SCALES = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "mm": 1e6, "million": 1e6, "millions": 1e6, "mio": 1e6,
           "b": 1e9, "bn": 1e9, "billion": 1e9, "billions": 1e9, "md": 1e9, "mrd": 1e9, "milliard": 1e9,
           "milliards": 1e9}
# A figure as reports write it: "$1.2M", "(1,200)", "12.5%", "1 200 €", "12,5 %". A minus is one only when it is
# stuck to the figure ("North - 41" is a dash); a number running into letters ("27th", "4.2x", "12.5pp") is none.
_NUMBER = re.compile(
    r"(?<![\w.,])(?P<open>\()?(?P<sign>[-−–](?=[$€£¥₹\d.]))?(?P<cur>[$€£¥₹]|(?:USD|CAD|EUR|GBP|AUD|INR)\s?)?"
    r"(?P<sign2>[-−–](?=[\d.]))?"
    r"(?P<num>\d{1,3}(?:(?:,|[   ])\d{3})+(?:\.\d+)?|\d+,\d{1,2}(?![\d,])|\d+(?:\.\d+)?|\.\d+)"
    r"(?:\s?(?P<scale>thousand|millions?|billions?|milliards?|mio|mrd|mn|mm|bn|md|k|m|b)(?![a-z]))?"
    r"(?:\s?(?P<cur2>[$€£]))?"
    r"(?:\s?(?P<pct>%|percent\b|pct\b|per\s?cent\b))?(?P<close>\))?(?!\.?\w)", re.IGNORECASE)
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_MONTH_WORD = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?" \
              r"|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
# Periods written in a report's answer are what a figure is for, never figures: "March 2025", "Jun 25", "Q2 2025",
# "Q2-25", "2025-03", "31/03/2025", "FY2025".
_PERIODS = re.compile(
    rf"\b(?:{_MONTH_WORD}\.?\s+(?:\d{{1,2}},?\s+)?(?:(?:19|20)\d\d|'?\d\d)\b|(?:q[1-4]|h[12])\s*[-/]?\s*(?:fy)?\s*'?(?:19|20)?\d\d"
    rf"|(?:19|20)\d\d\s*[-/]?\s*(?:q[1-4]|h[12])|\d{{4}}-\d{{2}}(?:-\d{{2}})?|\d{{1,2}}/\d{{1,2}}/\d{{2,4}}"
    rf"|fy\s*'?(?:19|20)?\d\d)\b", re.IGNORECASE)
# A question that names a time of its own: a year, a quarter, a month, "the first half", "this week"; "by month" and
# "monthly" only say how it is split.
_NAMES_TIME = re.compile(rf"\b(?:(?:19|20)\d\d|q[1-4]|h[12]|[yqm]td|fy\w*|{_MONTH_WORD.replace('|may|', '|(?-i:May)|')}"
                         r"|quarter|month|week|year|half|today|yesterday)s?\b", re.IGNORECASE)
_A_GRAIN = re.compile(r"\b(?:by|per|each|every|a|month\s+by|week\s+by|quarter\s+by|year\s+by)\s+"
                      r"(?:day|week|month|quarter|year)s?\b", re.IGNORECASE)
_PARTS = re.compile(r"\s*(?:;|\n|\||,\s+(?=\D))\s*")
_LABEL_EDGE = re.compile(r"^[\s:=\-–—(),.~<>≈/]+|[\s:=\-–—(),.~<>≈/]+$")
_LEAD_WORDS = re.compile(r"^(?:is|was|were|are|=|of|to|by|at|and|about|around|nearly|almost|approx(?:imately)?)"
                         r"(?:\s+|$)", re.IGNORECASE)
_TRAIL_WORDS = re.compile(r"(?:\s+(?:was|were|is|are|came\s+to|totall?ed|reached|stood\s+at|of|at|had|made|with))+$",
                          re.IGNORECASE)
# A figure set against another period, not the one asked about: "up 4.1% on 2024", "(+5% YoY)", "vs 37 last year".
_CHANGE_BEFORE = re.compile(r"(?:\b(?:up|down|rose|fell|grew|increased|decreased|growth|change|changed|vs\.?|versus|"
                            r"compared\s+(?:to|with)|against|from)(?:\s+(?:by|of))?|\+)\s*$", re.IGNORECASE)
_CHANGE_AFTER = re.compile(r"^\s*(?:yoy|y/y|mom|qoq|year[- ]on[- ]year|vs\b|versus|on\s+(?:last|the|(?:19|20)\d\d)|"
                           r"higher|lower|more\s+than|less\s+than|growth|up\b|down\b)", re.IGNORECASE)
# The member a figure is for, written after it: "41 in North", "$1.2M for Retail".
_AFTER = re.compile(r"^\s*(?:in|for|at|from|across|within|à|en|pour|chez)\s+(?P<who>[^()\d$€£%]+?)\s*$", re.IGNORECASE)


def _year_like(match: re.Match[str], text: str) -> bool:
    digits = match.group("num")
    if not (len(digits) == 4 and digits.isdigit() and 1900 <= int(digits) <= 2100) or match.group("cur") \
            or match.group("cur2") or match.group("scale") or match.group("pct"):
        return False
    return not re.search(r"[:=]\s*$", text[:match.start()])          # "Total: 2048" is a figure


def _set_against(part: str, m: re.Match[str]) -> bool:
    """Is the figure a comparison beside the one asked about ("up 4.1% on 2024", "(2024: 37)", "+5%")?"""
    if _CHANGE_BEFORE.search(part[:m.start()].rstrip("( ")) or _CHANGE_AFTER.match(part[m.end():]):
        return True
    opened = part[:m.start()].rfind("(")
    return opened > part[:m.start()].rfind(")") and bool(re.search(r"[:=]|\b(?:19|20)\d\d\b", part[opened:m.start()]))


def expected_figures(text: str) -> list[Figure]:
    """The figures a report's answer gives, each with the member or period it is for. "North: 41, South: 37" is
    two figures; "$1.2M" one, rounded to a tenth of a million; "12.5%" a percentage; a year, a date, a member's own
    number ("Store 12:") or a comparison ("up 4% on 2024") is never a figure."""
    out: list[Figure] = []
    for part in _PARTS.split(str(text or "")):
        if not part.strip():
            continue
        # Periods stand out of the way of the figures, kept for the label they are part of.
        masked = _PERIODS.sub(lambda m: "⁣" * len(m.group(0)), part)
        every = list(_NUMBER.finditer(masked))
        years = [m for m in every if _year_like(m, masked)]
        # Beside any other number (even one set against another period), a bare year is when it was.
        found = [m for m in every if not (len(every) > len(years) and m in years)
                 and not re.match(r"\s*:", masked[m.end():]) and not _set_against(masked, m)]
        start, label, mine, taken = 0, "", len(out), False
        # A part that is a figure alone goes on with the member before it ("North: $1.2M, 12%").
        alone = len(found) == 1 and not _LABEL_EDGE.sub("", masked[:found[0].start()]).strip() \
            and not masked[found[0].end():].strip(" .;)")
        for n, m in enumerate(found):
            digits = m.group("num")
            french = bool(re.fullmatch(r"\d+,\d{1,2}", digits))
            digits = digits.replace(",", ".") if french else re.sub(r"[,   ]", "", digits)
            try:
                value = float(digits)
            except ValueError:
                continue
            decimals = len(digits.split(".")[1]) if "." in digits else 0
            word = (m.group("scale") or "").lower()
            money = bool(m.group("cur") or m.group("cur2"))
            # A lone lower-case m is a million only beside money or a decimal point ("$5m", "1.2m"): "41 m" is
            # 41 of something.
            if word == "m" and not money and "." not in digits:
                word = ""
            scale = _SCALES.get(word, 1.0)
            between = part[start:m.start()]
            # "(1,200)" and "Net income: (1,200)" are losses; "45 loans (4.0%)", "North (12%)" and
            # "$3.09M ($3,093,601)" say something more about the figure before.
            lead = part[:m.start()]
            lead = lead[:lead.rfind("(")] if m.group("open") else ""
            negative = bool(m.group("sign") or m.group("sign2")
                            or m.group("open") and m.group("close") and (not lead.strip() or lead.rstrip()[-1:] in ":="))
            value = float(Decimal(digits) * Decimal(str(scale))) * (-1 if negative else 1)
            # Unit words carry the member on ("45 loans, 4.0%"); a name or a separator starts another ("North 41 /
            # South 37"), and so does a figure after one whose member was written after it ("41 in North, 37 in South").
            carried = bool(n) and not taken and not re.search(r"[:=/]|\b[A-Z]", between) \
                and not _PERIODS.search(between)
            if not carried:
                label = _TRAIL_WORDS.sub("", _LEAD_WORDS.sub("", _LABEL_EDGE.sub("", between.strip())).strip())
                label = _LABEL_EDGE.sub("", label).strip()
            if alone and mine > 0 and not label:
                label, carried = out[mine - 1].label, True
            taken = False
            after = part[m.end():found[n + 1].start()] if n + 1 < len(found) else part[m.end():]
            who = _AFTER.match(after.rstrip(" ,;."))
            if who and not carried:
                label = f"{label} {who.group('who').strip(' ,;.')}".strip()   # "41 in North", "$1.2M for Retail"
                taken = True
            written = part[m.start():m.end()].strip()
            if not negative:
                written = written.strip("()")
            restates = bool(n) and bool(m.group("open")) and not between.replace("(", "").strip() \
                and bool(m.group("pct")) == out[-1].percent
            out.append(Figure(value=value, label=label, percent=bool(m.group("pct")), money=money,
                              step=(10 ** -decimals) * scale, text=written, carried=carried and bool(label),
                              restates=restates))
            start = found[n + 1].start() if taken and n + 1 < len(found) else m.end()
    return out


# ── the answer's figures ────────────────────────────────────────────────────

_PERCENT = ("percent", "percentage")
_NOT_ADDED = (*_PERCENT, "ratio", "rate", "days")


@dataclass
class _Cell:
    value: float
    column: str
    header: str
    fmt: str


@dataclass
class _Row:
    label: str
    words: set[str]
    texts: list[set[str]] = field(default_factory=list)      # each text cell's own words
    days: list[dt.date] = field(default_factory=list)
    cells: list[_Cell] = field(default_factory=list)


@dataclass
class _Column:
    name: str
    header: str
    fmt: str
    words: set[str]


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _as_day(value: Any) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str) and re.match(r"^\d{4}-\d{2}-\d{2}", value):
        try:
            return dt.date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


_STOP = {"the", "and", "of", "a", "an", "in", "for", "at", "by", "on", "to", "per", "with", "from", "de", "du",
         "des", "la", "le", "les", "et"}


def _words(text: str) -> set[str]:
    """The words of a name, plurals as their singular: "loans" and "Number of loan originations" share "loan"."""
    return {w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
            for w in re.findall(r"[a-z0-9]+", str(text or "").lower()) if w not in _STOP}


def _day_said(day: dt.date, style: str) -> str:
    """A period as the answer shows it: Q2 2025, Apr 2025, 2025, 1 Apr 2025."""
    if "quarter" in style:
        return f"Q{(day.month - 1) // 3 + 1} {day.year}"
    if "month" in style:
        return f"{day:%b %Y}"
    if style.startswith("year"):
        return str(day.year)
    return f"{day.day} {day:%b %Y}"


def _table(payload: dict[str, Any]) -> tuple[list[_Row], list[_Column]]:
    """The answer's rows (every one, not only the 200 shown) and its figure columns in order."""
    data = payload.get("data") or {}
    formats: dict[str, str] = data.get("column_formats") or {}
    headers: dict[str, str] = data.get("header_labels") or {}
    styles = {c: str((f or {}).get("style") or "") for c, f in (data.get("display_formats") or {}).items()
              if isinstance(f, dict)}
    records = payload.get("export_rows") or data.get("rows") or []
    rows, columns = [], {}
    for row in records:
        texts, days, said, cells = [], [], [], []
        for column, value in row.items():
            fmt = str(formats.get(column) or "")
            day = _as_day(value)
            number = None if fmt in ("text", "date") else _as_number(value)
            if day is not None:
                days.append(day)
                said.append(_day_said(day, styles.get(column, "")))
            elif number is not None:
                header = str(headers.get(column) or column)
                cells.append(_Cell(number, column, header, fmt))
                columns.setdefault(column, _Column(column, header, fmt, _words(header)))
            elif value not in (None, ""):
                texts.append(str(value))
        label = ", ".join([*said, *texts])
        rows.append(_Row(label=label, words=_words(" ".join(texts)), texts=[_words(t) for t in texts], days=days,
                         cells=cells))
    return rows, list(columns.values())


def _period(label: str) -> tuple[int | None, set[int]] | None:
    """The period a report's label names: its year (None when it names none) and its months. "Q2 2025",
    "March 2025", "2025-03", "2025", or "Q2" and "March" alone."""
    text = label.lower()
    year = re.search(r"\b(?:19|20)\d\d\b", text)
    y = int(year.group(0)) if year else None
    quarter = re.search(r"\bq([1-4])\b", text)
    if quarter:
        q = int(quarter.group(1))
        return y, {3 * q - 2, 3 * q - 1, 3 * q}
    half = re.search(r"\bh([12])\b", text)
    if half:
        return y, set(range(1, 7)) if half.group(1) == "1" else set(range(7, 13))
    month = re.search(r"\b\d{4}-(\d{2})\b", text)
    if month:
        return y, {int(month.group(1))}
    named = re.search(rf"\b{_MONTH_WORD}\b", text)
    if named:
        return y, {_MONTHS.index(named.group(0)[:3]) + 1}
    if y is not None and re.fullmatch(r"\s*(?:fy\s*)?(?:19|20)\d\d\s*", text):
        return y, set(range(1, 13))
    return None


# Words that name a period, never a member: "Q2", "H1", "March", "FY".
_PERIOD_WORDS = re.compile(rf"\b(?:q[1-4]|h[12]|fy\d*|{_MONTH_WORD})\b", re.IGNORECASE)
_TOTAL_WORDS = {"total", "overall", "all", "sum", "grand", "everything", "whole"}
_YEARS = frozenset(str(y) for y in range(1900, 2101))
# A report's word for what an answer's header calls otherwise: "Growth" is the answer's "change".
_HEADER_SAME = {"growth": "change", "increase": "change", "decrease": "change", "difference": "change",
                "delta": "change", "variance": "change", "diff": "change"}


def _rows_named(rows: list[_Row], words: set[str]) -> tuple[list[_Row], str]:
    """The rows a member's name points at: the ones it names exactly ("North", never "North East" beside it),
    else the one row holding all its words; more than one is no answer."""
    exact = [r for r in rows if words == r.words or any(words == t for t in r.texts)]
    if exact:
        return exact, ""
    within = [r for r in rows if words <= r.words]
    if len(within) > 1:
        return [], f"names more than one row of Solviz's answer ({'; '.join(r.label for r in within[:3])})"
    return within, ""


def _missing(rows: list[_Row], label: str, words: set[str]) -> str:
    """Why a member the report names is not found, with the answer's row nearest to it ("LN0205, Union Square,
    V. Lang" for "LN0205 / Lakeview / V. Lang": the same loan in another branch)."""
    near = max(rows, key=lambda r: len(words & r.words), default=None)
    if near is not None and len(words) > 1 and 2 * len(words & near.words) >= len(words):
        return f"{label} is not in Solviz's answer; its nearest row reads {near.label}."
    return f"{label} is not in Solviz's answer."


def _bucket(figure: Figure, columns: list[_Column]) -> list[_Column]:
    """The answer's columns a figure can be read from, by kind: a percentage only from a percentage, money from
    money first, any other figure from a count or an amount, never from a share."""
    if figure.percent:
        return [c for c in columns if c.fmt in _PERCENT] or [c for c in columns if c.fmt in ("ratio", "rate")]
    money = [c for c in columns if c.fmt == "currency"]
    plain = [c for c in columns if c.fmt not in _PERCENT and c.fmt != "currency"]
    return (money + plain) if figure.money else (plain + money)


def _verdict(figure: Figure, got: float) -> str:
    d = abs(got - figure.value)
    rel = d / abs(figure.value) if figure.value else (0.0 if d == 0 else math.inf)
    slack = 1e-9 * max(1.0, abs(figure.value))
    # Rounded the way the report rounds it, and never more than 5% off when the report rounds to thousands or more
    # ("$2M" is not $1.55M).
    if d <= figure.step / 2 + slack and (figure.step <= 1 or rel <= 0.05):
        return "matches"
    if rel <= 0.01 or d <= figure.step + slack and rel <= 0.10:
        return "close"
    return "different"


def _one(figure: Figure, rows: list[_Row], columns: list[_Column], index: int, pick: str = "") -> dict[str, Any]:
    said = {"expected": figure.text or str(figure.value), "label": figure.label, "want": figure.value,
            "percent": figure.percent, "step": figure.step}
    differ = {**said, "verdict": "different", "got": None, "where": ""}
    bucket = _bucket(figure, columns)
    if not bucket:
        return {**differ, "note": "Solviz's answer shows no percentage." if figure.percent
                else "Solviz's answer has no figure to compare."}
    words = _words(_PERIOD_WORDS.sub(" ", _PERIODS.sub(" ", figure.label)))
    span = _period(figure.label) if figure.label else None
    headers = set().union(*(c.words for c in columns)) if columns else set()
    total = bool(words & _TOTAL_WORDS)
    in_rows = set().union(*(r.words for r in rows)) if rows else set()
    # "Growth" is the answer's "change" only where no row is called Growth.
    same = {w: _HEADER_SAME[w] for w in words if w in _HEADER_SAME and w not in in_rows}
    base = words - _TOTAL_WORDS - _YEARS
    named_whole: list[_Row] = []
    if base - headers:
        # A label that names a row in full is that row, even where a header shares a word ("Consumer Loans" beside
        # "Loan Amount"); what is left of it names the column. A label made only of header words ("Loans" beside
        # "Loan amount") names the column, never a row that holds the word.
        named_whole, why = _rows_named(rows, base)
        named_whole = [] if why else named_whole
    if named_whole:
        member = base
        measure = {same.get(w, w) for w in words - set().union(*(r.words for r in named_whole))}
    else:
        measure = {same.get(w, w) for w in words}
        member = {w for w in base if w not in headers and same.get(w) not in headers}
    # The column: the one the label names by its header ("Cost amount: $1.92M"), else this kind's figures in order.
    scored = [(len(measure & c.words), i, c) for i, c in enumerate(bucket) if measure & c.words]
    if pick:
        column = next((c for c in bucket if c.name == pick), None)
    else:
        column = max(scored, key=lambda s: (s[0], -s[1]))[2] if scored else bucket[index] if index < len(bucket) \
            else None
    if column is None:
        return {**differ, "note": "Solviz's answer has fewer figures than the report."}
    by = "header" if scored and not pick else "order"
    # The rows: those the member names, then those of the period it names.
    chosen: list[_Row] | None = None
    if member:
        named, why = (named_whole, "") if named_whole else _rows_named(rows, member)
        if why:
            return {**differ, "note": f"{figure.label} {why}."}
        if not named:
            if len(rows) == 1 and not rows[0].texts:
                named = rows               # one figure with no member shown: the question's own filter
            elif scored and not figure.carried:
                named = None               # measure words not in its header ("Net sales: $3.09M"): the figure
            else:
                return {**differ, "note": _missing(rows, figure.label, member)}
        chosen = named
    if span is not None and any(r.days for r in (chosen or rows)):
        year, months = span
        inside = [r for r in (chosen or rows) if any(d.month in months and (year is None or d.year == year)
                                                       for d in r.days)]
        if not inside:
            return {**differ, "note": f"{figure.label} is not in Solviz's answer."}
        if year is None and len({d.year for r in inside for d in r.days}) > 1:
            return {**differ, "note": f"{figure.label} is in more than one year of Solviz's answer: write its year."}
        chosen = inside
    if chosen is None:
        chosen = rows if len(rows) == 1 else None
    if chosen is None or total or len(chosen) > 1:
        # A total, a figure with no member beside several rows, or a member over several periods: the rows added up,
        # or the answer's own total row.
        pool = chosen if chosen is not None and not total else rows
        totals = [r for r in pool if r.words & _TOTAL_WORDS]
        if totals:
            pool = totals[:1]
        if len(pool) > 1 and (column.fmt in _NOT_ADDED or figure.percent):
            return {**differ, "note": "Solviz's answer gives it row by row, with no figure for all of them."}
        cells = [c for r in pool for c in r.cells if c.column == column.name]
        if not cells:
            return {**differ, "note": "Solviz's answer has no figure to compare."}
        got = sum(c.value for c in cells)
        where = column.header if len(pool) == 1 else f"{column.header}, all rows added up"
        if len(pool) == 1 and pool[0].label:
            where = f"{pool[0].label} · {column.header}"
        at = tuple(id(r) for r in pool)
    else:
        row = chosen[0]
        cells = [c for c in row.cells if c.column == column.name]
        if not cells:
            return {**differ, "note": "Solviz's answer has no figure to compare."}
        got = cells[0].value
        where = column.header if len(rows) == 1 or not row.label else f"{row.label} · {column.header}"
        at = (id(row),)
    # A percentage is read as the answer shows it, in points: a share kept as 0.25 shows as 0.3%, and is no 25%.
    return {**said, "verdict": _verdict(figure, got), "got": got, "format": column.fmt, "where": where,
            "_by": by, "_at": at, "_column": column.name}


_SCORE = {"matches": 2, "close": 1}
# The answer's figure for the period before ("Loan amount, a year before", "Fills (before)"), and its changes.
_BEFORE = re.compile(r"\b(?:before|prior|previous|ago|avant)\b|plus\s+t[oô]t", re.IGNORECASE)
_MOVED = re.compile(r"\b(?:change|écart|variation|share|part)\b", re.IGNORECASE)
_A_YEAR = re.compile(r"^\s*(?:fy\s*)?((?:19|20)\d\d)\s*$", re.IGNORECASE)


def _years_on_columns(figures: list[Figure], each: list[dict[str, Any]], rows: list[_Row],
                      columns: list[_Column]) -> None:
    """Two years written as labels over an answer with no dates ("2024: 25,480,000, 2025: 33,983,000" beside
    "Loan amount" and "Loan amount, a year before"): the later year is the answer's figure, the earlier the one
    before, by their headers, never by their values."""
    years = {n: int(m.group(1)) for n, f in enumerate(figures)
             if not f.percent and (m := _A_YEAR.match(f.label)) and each[n].get("_by") == "order"}
    if len(set(years.values())) != 2 or any(r.days for r in rows):
        return
    bucket = _bucket(figures[next(iter(years))], columns)
    before = [c for c in bucket if _BEFORE.search(c.header)]
    now = [c for c in bucket if not _BEFORE.search(c.header) and not _MOVED.search(c.header)]
    if len(before) != 1 or not now:
        return
    late = max(years.values())
    for n, year in years.items():
        one = _one(figures[n], rows, columns, 0, pick=(now[0] if year == late else before[0]).name)
        each[n] = {**one, "_by": "header"}


def _in_its_order(figures: list[Figure], each: list[dict[str, Any]], rows: list[_Row],
                  columns: list[_Column]) -> None:
    """A member's figures written without their headers ("Live Oak: 7,077, 2,048") are read in the report's order
    first; where the report's order is not the answer's ("last year, this year, change" beside "this year, a year
    before, change"), each is read from the column of its kind that holds it, one column each, never one a header
    named, money only from money. The figures must be all the columns of their kind, or each a figure of two
    significant digits or more: "Live Oak: 7,077, 0" is never read off a column of zeros. A figure alone keeps
    the answer's order: "41" beside Orders 50 and Items 41 is no match."""
    groups: dict[tuple[Any, ...], list[int]] = {}
    for n, (figure, one) in enumerate(zip(figures, each)):
        if one.get("_by") == "order" and one.get("got") is not None and not figure.restates:
            groups.setdefault((figure.label.casefold(), one["_at"], figure.percent), []).append(n)
    for (_, at, _), ns in groups.items():
        if len(ns) < 2 or len(ns) > 6 or all(each[n]["verdict"] == "matches" for n in ns):
            continue
        held = {one.get("_column") for one in each if one.get("_by") == "header" and one.get("_at") == at}
        free = [c for c in _bucket(figures[ns[0]], columns) if c.name not in held]
        if len(free) > 8 or len(free) < len(ns):
            continue
        if len(free) > len(ns) and any(abs(figures[n].value) < 10 * figures[n].step for n in ns):
            continue
        tried = {(n, c.name): _one(figures[n], rows, columns, 0, pick=c.name) for n in ns for c in free
                 if not figures[n].money or c.fmt == "currency"}
        best, chosen = sum(_SCORE.get(each[n]["verdict"], 0) for n in ns), None
        for order in itertools.permutations(free, len(ns)):
            if any((n, c.name) not in tried for n, c in zip(ns, order)):
                continue
            score = sum(_SCORE.get(tried[(n, c.name)]["verdict"], 0) for n, c in zip(ns, order))
            if score > best:
                best, chosen = score, order
        for n, c in zip(ns, chosen or ()):
            each[n] = tried[(n, c.name)]


def said(value: Any, figure: dict[str, Any], fmt: str = "") -> str:
    """The answer's figure as the report writes it: "$3.09M ($3,093,601)" beside "$3.09M", "12.47%" beside
    "12.5%", "41" beside "41"."""
    number = _as_number(value)
    if number is None:
        return "—"
    text = str(figure.get("expected") or figure.get("text") or "")
    money = "$" if fmt == "currency" or "$" in text else ""
    sign = "-" if number < 0 else ""
    if figure.get("percent") or fmt in _PERCENT:
        return f"{number:,.2f}%".replace(".00%", "%")
    step = float(figure.get("step") or 1.0)
    whole = (f"{sign}{money}{abs(number):,.0f}" if abs(number) >= 100 or step >= 1
             else f"{sign}{money}{abs(number):,.2f}")
    written = re.search(r"\d\s?(thousand|million|billion|mn|mm|bn|k|m|b)\b", text, re.IGNORECASE)
    if written:
        size = _SCALES[written.group(1).lower()]
        decimals = max(0, round(-math.log10(step / size))) if step < size else 0
        return f"{sign}{money}{abs(number) / size:,.{decimals}f}{'KMB'[[1e3, 1e6, 1e9].index(size)]} ({whole})"
    return whole


def answered(payload: dict[str, Any]) -> bool:
    return bool(((payload.get("data") or {}).get("rows"))) and not payload.get("clarify")


def compare(expected: str, payload: dict[str, Any]) -> dict[str, Any]:
    """The verdict on one answer against the report's: the verdict, each figure set beside the answer's, and
    the answer's headline."""
    headline = str((payload.get("answer") or {}).get("headline") or payload.get("text") or "")[:600]
    if not answered(payload):
        clarify = payload.get("clarify") or {}
        options = ", ".join(map(str, clarify.get("options") or []))
        # A refusal or a reply in words is its own reason, shown as Solviz's answer.
        why = (f"Solviz asked back instead of answering: {clarify.get('question') or headline}"
               + (f" ({options})" if options else "") if clarify else "" if headline else "Solviz gave no answer.")
        return {"verdict": "unanswered", "figures": [], "why": why[:600], "headline": headline}
    rows, columns = _table(payload)
    figures = expected_figures(expected)
    if not figures:
        words = _words(expected)
        if words and len(words) <= 6 and rows:
            # A name, not a figure ("North" for "which region sold most"): the answer's first row is its leader.
            first, _ = _rows_named(rows[:1], words)
            if first:
                return {"verdict": "matches", "figures": [], "why": f"{rows[0].label} comes first, as the report "
                        "says.", "headline": headline}
            if any(_rows_named([r], words)[0] for r in rows[1:]):
                return {"verdict": "different", "figures": [], "why": f"Solviz puts {rows[0].label} first.",
                        "headline": headline}
        return {"verdict": "unchecked", "figures": [], "headline": headline,
                "why": "The report's answer has no figure to compare." if expected.strip()
                else "No answer from the report was given: check it by eye."}
    each, missing, seen = [], set(), {}
    for figure in figures:
        # The nth figure of a kind for one member is read from the nth column of that kind ("$3.09M, $1.92M" for net
        # sales and cost); a figure said again in full is the one before.
        key = (figure.label.casefold(), figure.percent)
        if not figure.restates:
            seen[key] = seen.get(key, -1) + 1
        one = _one(figure, rows, columns, max(seen.get(key, 0), 0))
        if figure.carried and figure.label in missing:
            # The figure before named a member the answer lacks: this one is that member's too.
            one = {**one, "verdict": "different", "got": None, "where": "",
                   "note": f"{figure.label} is not in Solviz's answer."}
        elif one.get("got") is None and figure.label:
            missing.add(figure.label)
        each.append(one)
    _years_on_columns(figures, each, rows, columns)
    _in_its_order(figures, each, rows, columns)
    each = [{k: v for k, v in e.items() if not k.startswith("_")} for e in each]
    verdict = max((e["verdict"] for e in each), key=lambda v: _WORST[v])
    return {"verdict": verdict, "figures": each, "headline": headline}


# ── a run ───────────────────────────────────────────────────────────────────

def asked_text(question: dict[str, Any]) -> str:
    """The question as asked: with the report's period when the question does not name one itself (a year, a
    quarter, a month: "in the first half of 2026" is H1 2026 said already)."""
    text = str(question.get("question") or "").strip()
    period = str(question.get("period") or "").strip()
    if period and period.lower() not in text.lower() and not _NAMES_TIME.search(_A_GRAIN.sub(" ", text)):
        text = f"{text.rstrip(' ?.')}, for {period}?" if text.endswith("?") else f"{text}, for {period}"
    return text


def _ask(account_id: str, question: dict[str, Any], check_id: int) -> dict[str, Any]:
    from core2.service import forget_session, portal_answer

    key = f"answers-check:{account_id}:{check_id}:{question['id']}"
    try:
        return portal_answer(account_id, asked_text(question), None, session_key=key)
    finally:
        forget_session(key)          # a run's questions never push readers' conversations out of memory


def _replay(account_id: str, question: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    from core2.service import portal_replay

    return portal_replay(account_id, plan, None, question=asked_text(question))


def check_one(account_id: str, question: dict[str, Any], *, check_id: int, plan: dict[str, Any] | None = None
              ) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    """One question asked (or its plan drawn again): the verdict, the plan to replay it by, and whether the AI was
    asked."""
    asked = plan is None
    try:
        payload = _ask(account_id, question, check_id) if asked else _replay(account_id, question, plan or {})
    except Exception as exc:  # noqa: BLE001 - one question that fails never stops the run
        log.warning("answers check: %s question %s failed: %s", account_id, question.get("id"), exc, exc_info=True)
        return ({"verdict": "unanswered", "figures": [], "headline": "",
                 "why": f"Solviz could not answer it ({type(exc).__name__}: {str(exc)[:160]})."}, plan, asked)
    detail = compare(str(question.get("expected") or ""), payload)
    detail["asked"] = asked_text(question)
    kept = payload.get("plan") if isinstance(payload.get("plan"), dict) else plan
    return detail, kept, asked


_RUNNING: set[tuple[str, int]] = set()
_RUNNING_LOCK = threading.Lock()


def run(account_id: str, *, mode: str, reason: str, only: list[int] | None = None, started_by: str = "",
        check_id: int | None = None) -> int | None:
    """One run of the answers check, stored as it goes; the run's id, or None while another runs. ``mode`` "ask"
    asks each question afresh; "replay" draws the last answer's plan again (asking afresh those that have none).
    ``only`` limits the run to those test questions."""
    import store

    client = store.get_client(account_id) or {}
    db_id = int(client.get("db_config_id") or 0)
    questions = [q for q in store.list_core2_test_questions(account_id, db_id) if only is None or q["id"] in only]
    if not questions:
        if check_id is not None:
            # Claimed for questions taken off since: closed, never left running.
            store.update_core2_answer_check(check_id, done_count=0, status="done", message="No questions to check.")
        return None
    if check_id is None:
        check_id = store.claim_core2_answer_check(account_id, db_id, mode=mode, reason=reason,
                                                  stamp=store.core2_model_stamp(account_id, db_id),
                                                  total=len(questions), started_by=started_by, whole=only is None)
        if check_id is None:
            return None
    key = (account_id, db_id)
    with _RUNNING_LOCK:
        _RUNNING.add(key)
    try:
        last = store.core2_answer_results(account_id, db_id, per_question=1)
        for n, question in enumerate(questions, start=1):
            previous = (last.get(int(question["id"])) or [{}])[0]
            plan = previous.get("plan") if mode == "replay" else None
            detail, kept, asked = check_one(account_id, question, check_id=check_id, plan=plan)
            store.add_core2_answer_result(check_id, account_id, db_id, int(question["id"]),
                                          verdict=detail["verdict"], detail=detail, plan=kept,
                                          headline=detail.get("headline", ""), asked=asked)
            store.update_core2_answer_check(check_id, done_count=n)
        store.update_core2_answer_check(check_id, status="done")
        store.prune_core2_answer_checks(account_id, db_id)
    except Exception as exc:  # noqa: BLE001 - said on the page; the next run starts afresh
        log.warning("answers check: run %s for %s stopped: %s", check_id, account_id, exc, exc_info=True)
        store.update_core2_answer_check(check_id, status="failed", message=f"Stopped part way: {exc}")
    finally:
        with _RUNNING_LOCK:
            _RUNNING.discard(key)
    return check_id


def start(account_id: str, *, mode: str, reason: str, only: list[int] | None = None, started_by: str = ""
          ) -> int | None:
    """A run started in the background (the page shows it as it goes); its id, or None while another runs."""
    import store

    client = store.get_client(account_id) or {}
    db_id = int(client.get("db_config_id") or 0)
    questions = [q for q in store.list_core2_test_questions(account_id, db_id) if only is None or q["id"] in only]
    if not questions:
        return None
    check_id = store.claim_core2_answer_check(account_id, db_id, mode=mode, reason=reason,
                                              stamp=store.core2_model_stamp(account_id, db_id),
                                              total=len(questions), started_by=started_by, whole=only is None)
    if check_id is None:
        return None
    threading.Thread(target=run, kwargs={"account_id": account_id, "mode": mode, "reason": reason, "only": only,
                                         "started_by": started_by, "check_id": check_id},
                     name=f"answers-check-{account_id}", daemon=True).start()
    return check_id


# ── when a run is due ───────────────────────────────────────────────────────

# The decisions each workspace was last seen with, and since when: a run waits until they stop changing.
_SEEN: dict[tuple[str, int], tuple[str, float]] = {}


def due(account_id: str, db_id: int, *, now: float | None = None) -> tuple[str, str, list[int] | None] | None:
    """The run due for one workspace now: (mode, reason, only those questions), or None.

    After a Learn every question is asked again; once decisions have stopped changing for SETTLE_SECONDS, the
    answers are drawn again on them; a question never checked is asked as it arrives."""
    import store

    now = time.monotonic() if now is None else now
    stamp = store.core2_model_stamp(account_id, db_id)
    if stamp is None:
        return None                                     # not learned yet: nothing to answer with
    if learning(store.latest_core2_build(account_id, db_id)):
        return None                                     # Learn first: its model is the one to check
    store.release_stopped_core2_answer_checks(account_id, db_id)
    questions = store.list_core2_test_questions(account_id, db_id)
    if not questions:
        return None
    with _RUNNING_LOCK:
        if (account_id, db_id) in _RUNNING:
            return None
    # What the questions were last all checked on: one asked again alone says nothing of the others, and a run a
    # restart stopped (running in no thread of this process; claiming the next run lets it go) did not finish.
    last = store.latest_core2_answer_check(account_id, db_id, finished=True, whole=True)
    results = store.core2_answer_results(account_id, db_id, per_question=1)
    fresh = [int(q["id"]) for q in questions if int(q["id"]) not in results]
    if last is None:
        return "ask", "first", None
    was = list(last.get("stamp") or [])
    if not was or int(was[0]) != int(stamp[0]):
        return "ask", "learn", None
    if fresh:
        return "ask", "new_questions", fresh
    digest = str(stamp[3])
    if len(was) > 3 and str(was[3]) == digest:
        _SEEN.pop((account_id, db_id), None)
        return None
    seen = _SEEN.get((account_id, db_id))
    if seen is None or seen[0] != digest:
        _SEEN[(account_id, db_id)] = (digest, now)
        return None
    if now - seen[1] < SETTLE_SECONDS:
        return None
    _SEEN.pop((account_id, db_id), None)
    return "replay", "decisions", None


# A Learn running longer than this was stopped (as the learned page reads it, admin/core2_routes.py).
LEARN_STALE = dt.timedelta(hours=6)


def learning(build: dict[str, Any] | None) -> bool:
    """Is a Learn running now, in this process? One a restart stopped stays 'running' in the store for good."""
    import store

    if not build or build.get("status") != "running" or build.get("runner") != store.core2_runner():
        return False
    try:
        started = dt.datetime.strptime(str(build.get("started_at") or "")[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return False
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - started < LEARN_STALE


def run_due_answer_checks() -> int:
    """Start every run due now, each in the background; how many were started. Called each minute."""
    import store

    started = 0
    for account_id, db_id in store.core2_test_question_workspaces():
        try:
            client = store.get_client(account_id) or {}
            if int(client.get("db_config_id") or 0) != db_id:
                continue                                # questions of a database the workspace no longer uses
            plan = due(account_id, db_id)
            if plan is None:
                continue
            mode, reason, only = plan
            if start(account_id, mode=mode, reason=reason, only=only, started_by="Solviz") is not None:
                started += 1
        except Exception as exc:  # noqa: BLE001 - one workspace never stops the others
            log.warning("answers check: %s not checked: %s", account_id, exc, exc_info=True)
    return started


def summary(account_id: str, db_id: int) -> dict[str, Any]:
    """The scorecard: how many questions each verdict holds, by their latest results."""
    import store

    questions = store.list_core2_test_questions(account_id, db_id)
    results = store.core2_answer_results(account_id, db_id, per_question=1)
    counts = {v: 0 for v in VERDICTS}
    waiting = 0
    for q in questions:
        latest = (results.get(int(q["id"])) or [None])[0]
        if latest is None:
            waiting += 1
        else:
            counts[latest["verdict"]] = counts.get(latest["verdict"], 0) + 1
    return {"total": len(questions), "counts": counts, "waiting": waiting,
            "checked": len(questions) - waiting - counts["unchecked"]}
