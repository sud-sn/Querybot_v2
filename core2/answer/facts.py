"""An analyst's reading of an answer: facts worked out in code from its rows, each with the figures it states.

An analyst looking at a chart says what is typical, what is not, what changed and what to watch, in figures
rounded the way people round them. The findings under an answer (insights.py) were worked out mechanically:
"May 2026 is 9% above Jan 2025" compares two months that happen to be the ends of the window, and "23% below
the average" uses an average one bulk order pushed up. The facts here are what the reading rests on:

* a series: the typical level and the range of whole periods, set apart from any one-off (a period outside the
  median by more than three times the series' usual spread); the latest whole periods against the same ones a
  year before; the pace of the last three periods against the three before; a run of rises or falls; the same
  months high in every year; a period the data covers in part, with where it is heading; and when one-offs
  pull the average away from the typical period, that the median is the one to read;
* a breakdown: how evenly it is spread (each member's share, the top 3, how many make 80%), the leader and how
  far ahead, a long tail, members below zero, a member far above all the others;
* a comparison by member: the net change, how many rose and fell, who carried the change and how much of it,
  and the biggest mover in percent among members with a base worth comparing;
* two groupings: the biggest cell, and the grouping whose mix differs most between members.

Each fact is a sentence in English and in French, the exact figures it states (the checker allows nothing
else), a size that ranks it, the section it belongs to (what stands out, why it moved, what to watch), and how
it was worked out. A period the data covers in part is never compared as a whole one. Facts that need more
than the rows on screen (the same period last year, the target, the members that moved it) are added after
the answer by notes.py, from small queries of their own.
"""

from __future__ import annotations

import datetime as dt
import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from core2.answer.builder import _Columns, _day, _noun, _number, _Units
from core2.compile.compiler import OutColumn
from core2.resolve.resolver import Logical, OutMeasure, adds_up
from core2.resolve.time import label as period_label

STANDS, WHY, WATCH = "stands_out", "why", "watch"
ONE_OFF = 3.0            # a period this many times the usual spread from the median is a one-off
FLAT = 0.02              # a change smaller than 2% is "flat"
SMALL_BASE = 0.02        # a member under 2% of the total is too small to rank by its growth in percent
MAX_FACTS = 12

_FR_MONTHS = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]
_FR_LONG = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
            "novembre", "décembre"]
_EN_LONG = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
            "November", "December"]


@dataclass
class Fact:
    kind: str
    section: str
    en: str
    fr: str
    figures: list[float] = field(default_factory=list)
    size: float = 0.0
    how: str = ""
    query: str = ""              # the small query beyond the answer it came from; "" for the rows on screen
    members: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.en, self.fr = _capital(self.en), _capital(self.fr)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


# ── figures as people write them ──────────────────────────────────────────────


def _trim(text: str) -> str:
    """1.00 -> 1.0, 2.30 -> 2.3: two decimals only where the second says something."""
    return text[:-1] if "." in text and text.endswith("0") and len(text.split(".")[1]) == 2 else text


def amount(value: float, format_: str, *, unit: str | None = None, lang: str = "en") -> str:
    """A figure rounded as a reader rounds it: $1.11M, $124.5K, $4,518, 38.0%, 4,518 orders' worth."""
    v = float(value)
    sign = "−" if v < 0 else ""
    a = abs(v)
    if format_ in ("percent", "percentage"):
        text = f"{a:.1f}%" if a < 10 else f"{a:.0f}%"
        return sign + (text.replace(".", ",").replace("%", " %") if lang == "fr" else text)
    if format_ == "days":
        text = f"{a:.1f}" if a < 10 and not a.is_integer() else f"{a:,.0f}"
        days = "jour" if lang == "fr" else "day"
        return f"{sign}{_local(text, lang)} {days}{'' if a == 1 else 's'}"
    if a >= 1e9:
        text, scale = _trim(f"{a / 1e9:.2f}"), "B" if lang == "en" else " G"
    elif a >= 1e6:
        text, scale = _trim(f"{a / 1e6:.2f}"), "M" if lang == "en" else " M"
    elif a >= 1e4:
        text, scale = f"{a / 1e3:.1f}", "K" if lang == "en" else " k"
    elif a >= 100 or a.is_integer():
        text, scale = f"{a:,.0f}", ""
    else:
        text, scale = f"{a:,.2f}", ""
    text = _local(text, lang)
    if format_ == "currency":
        return f"{sign}${text}{scale}" if lang == "en" else f"{sign}{text}{scale}\u00a0$"
    tail = f" {unit}" if unit else ""
    return f"{sign}{text}{scale}{tail}"


def _local(text: str, lang: str) -> str:
    return text.replace(",", " ").replace(".", ",") if lang == "fr" else text


def about(value: float) -> float:
    """An estimate to two significant figures, as it is said ("about $200K", not "about $199.6K")."""
    if not value:
        return 0.0
    digits = 2 - int(math.floor(math.log10(abs(value)))) - 1
    return round(value, digits)


def percent(ratio: float, *, signed: bool = False, lang: str = "en") -> str:
    """A change or a share: 7.7%, 40%, +93%, −0.5%."""
    p = ratio * 100
    text = f"{abs(p):.1f}" if abs(p) < 10 else f"{abs(p):.0f}"
    sign = ("+" if p > 0 else "−" if p < 0 else "") if signed else ("−" if p < 0 else "")
    text = sign + (text.replace(".", ",") if lang == "fr" else text)
    return f"{text} %" if lang == "fr" else f"{text}%"


def rounded(ratio: float) -> float:
    """A ratio as it is written, as a figure the checker reads back (7.7 for 0.0774)."""
    p = ratio * 100
    return round(p, 1) if abs(p) < 10 else round(p)


def period(day: dt.date, grain: str, *, lang: str = "en", fiscal_start: int | None = None, long: bool = False) -> str:
    if lang == "fr" and grain in ("month", "fiscal_month"):
        return f"{(_FR_LONG if long else _FR_MONTHS)[day.month - 1]} {day.year}"
    if lang == "en" and long and grain in ("month", "fiscal_month"):
        return f"{_EN_LONG[day.month - 1]} {day.year}"
    text = period_label(day, grain, fiscal_start=fiscal_start)
    if lang == "fr":
        text = text.replace("Week of", "Semaine du").replace("FY", "EF ")
        if grain == "day":
            text = f"{day.day} {_FR_MONTHS[day.month - 1]} {day.year}"
    return text


def _word(grain: str, lang: str, plural: bool = False) -> str:
    en = {"day": "day", "week": "week", "month": "month", "quarter": "quarter", "year": "year"}
    fr = {"day": "jour", "week": "semaine", "month": "mois", "quarter": "trimestre", "year": "année"}
    base = grain.replace("fiscal_", "")
    word = (fr if lang == "fr" else en).get(base, base)
    if plural and not word.endswith("s"):
        word += "s"
    return word


# ── one-offs ──────────────────────────────────────────────────────────────────


def spread(values: list[float]) -> float:
    """The usual distance from the median: the median absolute deviation, scaled to read like a standard
    deviation; the mean absolute deviation where more than half the values sit on the median."""
    middle = statistics.median(values)
    mad = statistics.median(abs(v - middle) for v in values) * 1.4826
    return mad or (sum(abs(v - middle) for v in values) / len(values) * 1.2533)


def one_offs(values: list[float]) -> list[int]:
    """The positions of values outside the median by more than three times the usual spread: at least six
    values are read, and at most a quarter of them can be one-offs (more is a pattern, not exceptions)."""
    if len(values) < 6:
        return []
    middle, usual = statistics.median(values), spread(values)
    if not usual:
        return []
    out = [i for i, v in enumerate(values) if abs(v - middle) > ONE_OFF * usual and abs(v - middle) > 0.25 * abs(middle)]
    return out if len(out) <= len(values) // 4 else []


# ── the reading ───────────────────────────────────────────────────────────────


@dataclass
class Reading:
    """What the note works from: the facts, and the answer's figures without its one-offs."""

    facts: list[Fact] = field(default_factory=list)
    lead: dict[str, str] = field(default_factory=dict)        # the analyst's headline, by language
    one_offs: list[dict[str, Any]] = field(default_factory=list)
    shape: str = ""                                          # series | breakdown | comparison | grid | value

    def as_dict(self) -> dict[str, Any]:
        ranked = sorted(self.facts, key=lambda f: -f.size)[:MAX_FACTS]
        return {"shape": self.shape, "lead": self.lead, "one_offs": self.one_offs,
                "facts": [f.as_dict() for f in ranked]}


def read(logical: Logical, cols: _Columns, raw: list[dict], units: _Units, *, truncated: bool = False) -> Reading:
    """The facts an analyst would read off this answer's rows."""
    measures = cols.of("measure")
    if not raw or not measures or logical.intent == "list" or logical.shape is not None:
        return Reading()
    periods = cols.of("period")
    members = [c for c in cols.of("attribute") if c.name != logical.unit_group]
    rows = units.main_rows(raw)
    unit = units.main if units.column is not None and units.main != "Unknown" else None
    m = measures[0]
    out = {o.name: o for o in logical.measures}
    o = out.get(m.name)
    if logical.compare is not None and members and not periods:
        return _comparison(logical, cols, rows, m, o, members[0], unit)
    if periods and not members:
        return _series(logical, rows, m, o, periods[0], unit)
    if members and not periods and logical.compare is None and len(members) + len(cols.of("time")) == 1:
        return _breakdown(logical, cols, rows, m, o, members[0], unit, truncated=truncated)
    if len(members) == 2 and not periods and logical.compare is None:
        return _grid(rows, m, o, members[0], members[1], unit)
    return Reading()


def _sums(o: OutMeasure | None, m: OutColumn) -> bool:
    return o is not None and m.format not in ("percent", "percentage", "days") and adds_up(o)


# ── a series ──────────────────────────────────────────────────────────────────


def _series(logical: Logical, rows: list[dict], m: OutColumn, o: OutMeasure | None, p: OutColumn,
            unit: str | None) -> Reading:
    partial = {d.isoformat() for d in logical.partial}
    points: list[tuple[dt.date, float]] = sorted(
        ((day, v) for r in rows for day in [_day(r[p.name])] for v in [_number(r[m.name])]
         if day is not None and v is not None), key=lambda x: x[0])
    whole = [(d, v) for d, v in points if d.isoformat() not in partial]
    reading = Reading(shape="series")
    if len(whole) < 3:
        return reading
    grain = p.grain or "month"
    fs = logical.fiscal_start
    name = lambda d, lang="en", long=False: period(d, grain, lang=lang, fiscal_start=fs, long=long)  # noqa: E731
    say = lambda v, lang="en": amount(v, m.format, unit=unit, lang=lang)   # noqa: E731
    label, label_fr = m.label, m.label
    sums = _sums(o, m)
    values = [v for _, v in whole]
    odd = one_offs(values)
    usual = [(d, v) for i, (d, v) in enumerate(whole) if i not in odd]
    middle = statistics.median(v for _, v in usual)
    facts = reading.facts
    w_en, w_fr = _word(grain, "en"), _word(grain, "fr")

    for i in odd:
        d, v = whole[i]
        times = v / middle if middle else None
        above = v > middle
        reading.one_offs.append({"period": d.isoformat(), "label": name(d), "value": v, "typical": middle})
        if times and above and times >= 2:
            how_en = f"{times:.1f} times the typical {w_en} ({say(middle)})"
            how_fr = f"{times:.1f}".replace(".", ",") + f" fois le {w_fr} habituel ({say(middle, 'fr')})"
            figures = [v, round(times, 1), middle]
        else:
            how_en = f"against a typical {w_en} of {say(middle)}"
            how_fr = f"pour un {w_fr} habituel de {say(middle, 'fr')}"
            figures = [v, middle]
        facts.append(Fact("one_off", STANDS,
                          f"{name(d, long=True)} ({say(v)}) is a one-off, {how_en}.",
                          f"{name(d, 'fr', long=True)} ({say(v, 'fr')}) est exceptionnel : {how_fr}.",
                          figures, size=3.0 + min(abs(v - middle) / (abs(middle) or 1), 5),
                          how=(f"{name(d)} is outside the median of the whole {w_en}s by more than "
                               f"{ONE_OFF:g} times their usual spread.")))

    # The typical level and the range, without the one-offs.
    lows = min(usual, key=lambda x: (x[1], x[0]))
    highs = max(usual, key=lambda x: (x[1], x[0]))
    steady = middle and (highs[1] - lows[1]) / abs(middle) <= 0.35
    first, last = usual[0][0], usual[-1][0]
    if steady:
        facts.append(Fact("level", STANDS,
                          f"{label} held near {say(middle)} a {w_en} from {name(first)} to {name(last)}"
                          + (", apart from the one-off." if odd else "."),
                          f"{label_fr} est resté proche de {say(middle, 'fr')} par {w_fr} de {name(first, 'fr')} à "
                          f"{name(last, 'fr')}" + (", hors exception." if odd else "."),
                          [middle], size=2.0, how=f"The median of the whole {w_en}s"
                          + (", without the one-off." if odd else ".")))
    if len(usual) >= 4:
        other = "Every other" if odd else "Every"
        facts.append(Fact("range", STANDS,
                          f"{other} {w_en} sits between {say(lows[1])} ({name(lows[0])}) and {say(highs[1])} "
                          f"({name(highs[0])}).",
                          f"{'Tous les autres' if odd else 'Tous les'} {_word(grain, 'fr', True)} se situent entre "
                          f"{say(lows[1], 'fr')} ({name(lows[0], 'fr')}) et {say(highs[1], 'fr')} "
                          f"({name(highs[0], 'fr')}).",
                          [lows[1], highs[1]], size=1.6 if steady else 1.9,
                          how=f"The lowest and highest whole {w_en}s" + (", without the one-off." if odd else ".")))

    # The latest whole periods of the year against the same ones a year before (whole periods only).
    if grain in ("month", "quarter", "week") and sums:
        by_day = dict(whole)
        latest_year = whole[-1][0].year
        these = [d for d, _ in whole if d.year == latest_year]
        before = [d.replace(year=d.year - 1) for d in these if grain != "week"]
        if these and before and all(b in by_day for b in before):
            now = sum(by_day[d] for d in these)
            then = sum(by_day[b] for b in before)
            if then:
                change = (now - then) / abs(then)
                span_en = name(these[0]) if len(these) == 1 else f"{name(these[0])} to {name(these[-1])}"
                span_fr = name(these[0], "fr") if len(these) == 1 else f"{name(these[0], 'fr')} à {name(these[-1], 'fr')}"
                lean = "above" if change > 0 else "below"
                # A total holds its one-offs: the sentence says which one is in it.
                held = [whole[i][0] for i in odd if whole[i][0] in set(these) | set(before)]
                with_en = f", the one-off of {name(held[0])} included" if held else ""
                with_fr = f", exception de {name(held[0], 'fr')} comprise" if held else ""
                facts.append(Fact("year_on_year", STANDS,
                                  f"{span_en} came to {say(now)}, {percent(abs(change))} {lean} the same "
                                  f"{w_en}s of {latest_year - 1} ({say(then)}){with_en}.",
                                  f"De {span_fr} : {say(now, 'fr')}, soit {percent(abs(change), lang='fr')} "
                                  f"{'de plus' if change > 0 else 'de moins'} que sur la même période de "
                                  f"{latest_year - 1} ({say(then, 'fr')}){with_fr}.",
                                  [now, rounded(abs(change)), then], size=2.6 + min(abs(change), 1),
                                  how=f"The whole {w_en}s of {latest_year} against the same {w_en}s a year before."))

    # The pace: the last three whole periods against the three before. A one-off in either three would be all the
    # change, so then it is the average of the others that is compared, and the sentence says the one-off is out.
    n = len(whole)
    recent_at = [i for i in range(n - 3, n) if i not in odd]
    earlier_at = [i for i in range(n - 6, n - 3) if i not in odd]
    without = len(recent_at) < 3 or len(earlier_at) < 3
    if n >= 6 and sums and without and recent_at and earlier_at:
        recent = statistics.fmean(values[i] for i in recent_at)
        earlier = statistics.fmean(values[i] for i in earlier_at)
        if earlier:
            change = (recent - earlier) / abs(earlier)
            flat = abs(change) < FLAT
            lean_en = "flat on" if flat else f"{percent(abs(change))} {'up' if change > 0 else 'down'} on"
            lean_fr = ("stable par rapport à" if flat else
                       f"en {'hausse' if change > 0 else 'baisse'} de {percent(abs(change), lang='fr')} par rapport à")
            facts.append(Fact(
                "pace", STANDS,
                f"Leaving out the one-off, the last three {w_en}s average {say(recent)} a {w_en}, {lean_en} the "
                f"three before ({say(earlier)}).",
                f"Hors exception, les trois derniers {_word(grain, 'fr', True)} font en moyenne {say(recent, 'fr')} "
                f"par {w_fr}, {lean_fr} la moyenne des trois précédents ({say(earlier, 'fr')}).",
                [recent, rounded(abs(change)), earlier], size=1.2 + min(abs(change) * 3, 1.2),
                how=f"The average of the last three whole {w_en}s against the three before, without the one-off."))
    elif n >= 6 and sums:
        recent = sum(v for _, v in whole[-3:])
        earlier = sum(v for _, v in whole[-6:-3])
        if earlier:
            change = (recent - earlier) / abs(earlier)
            if abs(change) < FLAT:
                en = f"The last three {w_en}s are flat against the three before ({percent(change, signed=True)})."
                fr = (f"Les trois derniers {_word(grain, 'fr', True)} sont stables par rapport aux trois "
                      f"précédents ({percent(change, signed=True, lang='fr')}).")
            else:
                en = (f"The last three {w_en}s are {percent(abs(change))} {'up' if change > 0 else 'down'} on "
                      f"the three before.")
                fr = (f"Les trois derniers {_word(grain, 'fr', True)} sont en "
                      f"{'hausse' if change > 0 else 'baisse'} de {percent(abs(change), lang='fr')} par rapport "
                      f"aux trois précédents.")
            facts.append(Fact("pace", STANDS, en, fr, [rounded(abs(change))], size=1.2 + min(abs(change) * 3, 1.2),
                              how=f"The sum of the last three whole {w_en}s against the three before."))

    # A run of rises or falls up to the latest whole period.
    run, direction = 0, 0
    for i in range(len(values) - 1, 0, -1):
        step = (values[i] > values[i - 1]) - (values[i] < values[i - 1])
        if step == 0 or (direction and step != direction):
            break
        direction, run = step, run + 1
    if run >= 3:
        facts.append(Fact("run", WATCH,
                          f"{label} has {'risen' if direction > 0 else 'fallen'} {run} {w_en}s in a row, up to "
                          f"{name(whole[-1][0])}.",
                          f"{label_fr} est en {'hausse' if direction > 0 else 'baisse'} depuis {run} "
                          f"{_word(grain, 'fr', True)} d'affilée, jusqu'à {name(whole[-1][0], 'fr')}.",
                          [run], size=1.4 + run / 10, how=f"Each whole {w_en} against the one before."))

    # The same months high in every year (two years or more of whole months).
    if grain == "month" and len(usual) >= 24:
        years: dict[int, list[tuple[dt.date, float]]] = {}
        for d, v in usual:
            years.setdefault(d.year, []).append((d, v))
        full = {y: pts for y, pts in years.items() if len(pts) == 12}
        if len(full) >= 2:
            tops = [{d.month for d, _ in sorted(pts, key=lambda x: -x[1])[:2]} for pts in full.values()]
            shared = set.intersection(*tops)
            if shared:
                months = sorted(shared)
                en_names = " and ".join(_EN_LONG[k - 1] for k in months)
                fr_names = " et ".join(_FR_LONG[k - 1] for k in months)
                facts.append(Fact("season", STANDS,
                                  f"{en_names} {'are' if len(months) > 1 else 'is'} the high point of every year.",
                                  f"{fr_names.capitalize()} {'sont' if len(months) > 1 else 'est'} le point haut de "
                                  "chaque année.", [], size=1.5,
                                  how="The two highest months of each whole year, the same in every year."))

    # A period the data covers in part: what it holds so far, and where it is heading.
    tail = [(d, v) for d, v in points if d.isoformat() in partial]
    if tail and sums and logical.data_last is not None and grain in ("month", "quarter", "week"):
        d, v = tail[-1]
        end = _period_end(d, grain)
        if d <= logical.data_last < end:
            covered = (logical.data_last - d).days + 1
            length = (end - d).days
            pace = about(v / covered * length) if covered else None
            typical = statistics.median(v2 for _, v2 in usual)
            en = f"{name(d, long=True)} has {covered} days of data so far ({say(v)})"
            fr = f"{name(d, 'fr', long=True)} compte {covered} jours de données ({say(v, 'fr')})"
            figures = [covered, v]
            if pace is not None and covered >= 5:
                normal = typical and abs(pace - typical) / abs(typical) <= 0.15
                en += f", on pace for about {say(pace)}" + (", which would be a normal one." if normal else ".")
                fr += f", soit environ {say(pace, 'fr')} à ce rythme" + (", un niveau habituel." if normal else ".")
                figures.append(pace)
            else:
                en += "."
                fr += "."
            facts.append(Fact("part_period", WATCH, en, fr, figures, size=1.8,
                              how=f"What {name(d)} holds so far, scaled to its {length} days."))

    # One-offs pull the average away from the typical period: the median is the one to read.
    if odd and len(values) >= 6:
        mean = sum(values) / len(values)
        if middle and abs(mean - middle) / abs(middle) >= 0.1:
            facts.append(Fact("average", WATCH,
                              f"The average of these {len(values)} {w_en}s ({say(mean)}) includes the one-off; "
                              f"the median {w_en}, {say(middle)}, is the typical one.",
                              f"La moyenne de ces {len(values)} {_word(grain, 'fr', True)} ({say(mean, 'fr')}) "
                              f"inclut l'exception ; la médiane, {say(middle, 'fr')}, est le {w_fr} type.",
                              [len(values), mean, middle], size=1.3,
                              how="The mean and the median of the whole periods."))

    reading.lead = _series_lead(reading, label, label_fr, middle, steady, first, last, whole, odd, name, say, w_en,
                                w_fr)
    return reading


def _period_end(start: dt.date, grain: str) -> dt.date:
    if grain == "week":
        return start + dt.timedelta(days=7)
    months = 3 if grain == "quarter" else 1
    year, month = divmod(start.month - 1 + months, 12)
    return dt.date(start.year + year, month + 1, 1)


def _series_lead(reading: Reading, label: str, label_fr: str, middle: float, steady: bool, first: dt.date,
                 last: dt.date, whole: list, odd: list[int], name, say, w_en: str, w_fr: str) -> dict[str, str]:
    """The analyst's headline for a series: its typical level when steady, else its range; a one-off by name."""
    if not steady:
        return {}
    en = f"{label} held near {say(middle)} a {w_en} from {name(first)} to {name(last)}."
    fr = f"{label_fr} est resté proche de {say(middle, 'fr')} par {w_fr} de {name(first, 'fr')} à {name(last, 'fr')}."
    if len(odd) == 1:
        d, v = whole[odd[0]]
        en += f" {name(d, long=True)} was a one-off at {say(v)}."
        fr += f" {_capital(name(d, 'fr', long=True))} est une exception, à {say(v, 'fr')}."
    elif len(odd) > 1:
        en += f" {len(odd)} {w_en}s were one-offs."
        fr += f" {len(odd)} périodes font exception."
    return {"en": en, "fr": fr}


# ── a breakdown ───────────────────────────────────────────────────────────────


def _breakdown(logical: Logical, cols: _Columns, rows: list[dict], m: OutColumn, o: OutMeasure | None,
               g: OutColumn, unit: str | None, *, truncated: bool) -> Reading:
    reading = Reading(shape="breakdown")
    values = [(str(r[g.name]), v) for r in rows if r[g.name] not in (None, "")
              for v in [_number(r[m.name])] if v is not None]
    if len(values) < 3 or len({v for _, v in values}) == 1:
        return reading
    say = lambda v, lang="en": amount(v, m.format, unit=unit, lang=lang)   # noqa: E731
    noun = _noun(g.label)
    ranked = sorted(values, key=lambda x: (-x[1], x[0]))
    facts = reading.facts
    sums = _sums(o, m)
    limited = bool(logical.limit) or truncated
    upward = bool(logical.sort) and not logical.sort[0][1]
    below = [(n, v) for n, v in values if v < 0]
    total = sum(v for _, v in values)
    (first, v1), (second, v2) = ranked[0], ranked[1]

    if sums and not below and not limited and total > 0:
        shares = [v / total for _, v in ranked]
        lo, hi = min(shares), max(shares)
        top3 = sum(shares[:3])
        n = len(shares)
        even = hi <= 2.5 / n and n >= 5
        if even:
            facts.append(Fact("spread", STANDS,
                              f"{label_of(m)} is spread evenly: each of the {n} {noun} makes between "
                              f"{percent(lo)} and {percent(hi)} of the total, and the top 3 make {percent(top3)}.",
                              f"{label_of(m)} est réparti de façon homogène : chacun des {n} représente entre "
                              f"{percent(lo, lang='fr')} et {percent(hi, lang='fr')} du total, et les 3 premiers "
                              f"{percent(top3, lang='fr')}.",
                              [n, rounded(lo), rounded(hi), 3, rounded(top3)], size=1.9,
                              how=f"Each {singular(noun)}'s share of the total of all {n}."))
        else:
            running, k = 0.0, 0
            for s in shares:
                running += s
                k += 1
                if running >= 0.8:
                    break
            if n >= 8 and k <= n // 2:
                facts.append(Fact("concentration", STANDS,
                                  f"{k} of the {n} {noun} make {percent(running)} of the total.",
                                  f"{k} des {n} font {percent(running, lang='fr')} du total.",
                                  [k, n, rounded(running)], size=2.2,
                                  how=f"The {noun} from the largest, until 80% of the total."))
            elif n >= 4:
                facts.append(Fact("concentration", STANDS,
                                  f"The top 3 of the {n} {noun} make {percent(top3)} of the total.",
                                  f"Les 3 premiers des {n} font {percent(top3, lang='fr')} du total.",
                                  [3, n, rounded(top3)], size=1.7, how="The three largest shares of the total."))
            if n >= 10:
                half = sum(shares[n // 2:])
                if half < 0.1:
                    facts.append(Fact("tail", WATCH,
                                      f"The smaller half of the {noun} make {percent(half)} of the total together.",
                                      f"La moitié la plus petite ne représente que {percent(half, lang='fr')} du total.",
                                      [rounded(half)], size=1.2, how="The shares of the smaller half, added."))
        # The leader, and how far ahead.
        if not upward and v1 > 0:
            share = v1 / total
            if v2 > 0 and v1 >= 1.5 * v2:
                lead_en = f"{first} leads with {say(v1)}, {percent(share)} of {say(total)}, {v1 / v2:.1f} times the next, {second} ({say(v2)})."
                lead_fr = (f"{first} est en tête avec {say(v1, 'fr')}, {percent(share, lang='fr')} de "
                           f"{say(total, 'fr')}, soit {f'{v1 / v2:.1f}'.replace('.', ',')} fois le suivant, {second} "
                           f"({say(v2, 'fr')}).")
                figures = [v1, rounded(share), total, round(v1 / v2, 1), v2]
            else:
                close = v2 > 0 and (v1 - v2) / v1 <= 0.1
                lead_en = (f"{first} leads with {say(v1)}, {percent(share)} of {say(total)}"
                           + (f", just ahead of {second} ({say(v2)})." if close else f"; {second} follows with {say(v2)}."))
                lead_fr = (f"{first} est en tête avec {say(v1, 'fr')}, {percent(share, lang='fr')} de "
                           f"{say(total, 'fr')}" + (f", juste devant {second} ({say(v2, 'fr')})." if close
                                                    else f" ; {second} suit avec {say(v2, 'fr')}."))
                figures = [v1, rounded(share), total, v2]
            reading.lead = {"en": lead_en, "fr": lead_fr}
            facts.append(Fact("leader", STANDS, lead_en, lead_fr, figures, size=1.5, members=[first, second],
                              how=f"The largest {singular(noun)}, its share of the total and the next one."))

    if below and sums:
        together = sum(v for _, v in below)
        if len(below) == 1:
            en = f"{below[0][0]} is below zero, at {say(below[0][1])}."
            fr = f"{below[0][0]} est négatif, à {say(below[0][1], 'fr')}."
            figures = [below[0][1]]
        else:
            en = f"{len(below)} {noun} are below zero, {say(together)} together."
            fr = f"{len(below)} sont négatifs, {say(together, 'fr')} au total."
            figures = [len(below), together]
        facts.append(Fact("below_zero", WATCH, en, fr, figures, size=2.0, members=[n for n, _ in below[:3]],
                          how="The members whose figure is under zero."))

    # A member far above all the others.
    if len(values) >= 6 and not upward:
        middle = statistics.median(v for _, v in values)
        usual = spread([v for _, v in values])
        if middle > 0 and usual and v1 - middle > ONE_OFF * usual and v1 >= 2 * v2:
            facts.append(Fact("outlier", STANDS,
                              f"{first} is far above every other {singular(noun)}: {say(v1)} against a typical "
                              f"{say(middle)}.",
                              f"{first} dépasse nettement tous les autres : {say(v1, 'fr')} pour "
                              f"{say(middle, 'fr')} habituellement.",
                              [v1, middle], size=2.4, members=[first],
                              how=f"Outside the median of the {noun} by more than {ONE_OFF:g} times their usual spread."))

    if not sums and len(values) >= 5:
        middle = statistics.median(v for _, v in values)
        low_name, low = ranked[-1]
        facts.append(Fact("range", STANDS,
                          f"From {say(low)} ({low_name}) to {say(v1)} ({first}); half the {noun} are above "
                          f"{say(middle)}.",
                          f"De {say(low, 'fr')} ({low_name}) à {say(v1, 'fr')} ({first}) ; la moitié est "
                          f"au-dessus de {say(middle, 'fr')}.",
                          [low, v1, middle], size=1.6, members=[low_name, first],
                          how="The lowest, the highest and the median."))
    return reading


def label_of(m: OutColumn) -> str:
    return m.label[:1].upper() + m.label[1:]


def singular(noun: str) -> str:
    from core2.bootstrap.names import singular as one

    return one(noun)


# ── a comparison by member ────────────────────────────────────────────────────


def _comparison(logical: Logical, cols: _Columns, rows: list[dict], m: OutColumn, o: OutMeasure | None,
                g: OutColumn, unit: str | None) -> Reading:
    reading = Reading(shape="comparison")
    change = next((c for c in cols.columns if c.role == "change" and c.measure == m.measure), None)
    prior = next((c for c in cols.columns if c.role == "prior" and c.measure == m.measure), None)
    if change is None:
        return reading
    moves = [(str(r[g.name]), by, _number(r[prior.name]) if prior else None, _number(r[m.name]))
             for r in rows if r[g.name] not in (None, "") for by in [_number(r[change.name])] if by is not None]
    if len(moves) < 3:
        return reading
    say = lambda v, lang="en": amount(v, m.format, unit=unit, lang=lang)   # noqa: E731
    noun = _noun(g.label)
    facts = reading.facts
    net = sum(by for _, by, _, _ in moves)
    up = [x for x in moves if x[1] > 0]
    down = [x for x in moves if x[1] < 0]
    if up and down:
        facts.append(Fact("rose_fell", STANDS,
                          f"{len(up)} of the {len(moves)} {noun} rose and {len(down)} fell.",
                          f"{len(up)} des {len(moves)} sont en hausse et {len(down)} en baisse.",
                          [len(up), len(moves), len(down)], size=1.4, how="Each member's change, counted by sign."))
    if _sums(o, m) and net:
        same = sorted([x for x in moves if (x[1] > 0) == (net > 0)], key=lambda x: -abs(x[1]))
        if same:
            name, by, _, _ = same[0]
            part = by / net
            rise_en, rise_fr = ("rise", "hausse") if net > 0 else ("fall", "baisse")
            if part <= 1:
                others = same[1:3]
                signed = lambda b, lang="en": ("+" if b > 0 else "") + say(b, lang)   # noqa: E731
                tail_en = (", then " + " and ".join(f"{n} ({signed(b)})" for n, b, _, _ in others)) if others else ""
                facts.append(Fact("driver", WHY,
                                  f"{name} {'added' if net > 0 else 'lost'} the most ({signed(by)}, "
                                  f"{percent(part)} of the net {rise_en}){tail_en}.",
                                  f"{name} a le plus {'contribué à la' if net > 0 else 'pesé sur la'} {rise_fr} "
                                  f"({signed(by, 'fr')}, {percent(part, lang='fr')} de la {rise_fr} nette).",
                                  [by, rounded(part), *[b for _, b, _, _ in others[:2]]], size=2.5,
                                  members=[name, *[n for n, _, _, _ in others[:2]]],
                                  how="Each member's change as a share of the net change."))
            opposite = sorted([x for x in moves if (x[1] > 0) != (net > 0)], key=lambda x: -abs(x[1]))[:2]
            if opposite:
                listed_en = " and ".join(f"{n} ({'+' if b > 0 else ''}{say(b)})" for n, b, _, _ in opposite)
                listed_fr = " et ".join(f"{n} ({'+' if b > 0 else ''}{say(b, 'fr')})" for n, b, _, _ in opposite)
                facts.append(Fact("against", WHY,
                                  f"{listed_en} {'fell' if net > 0 else 'rose'} against the trend.",
                                  f"{listed_fr} {'ont baissé' if net > 0 else 'ont progressé'} à contre-courant.",
                                  [b for _, b, _, _ in opposite], size=2.0, members=[n for n, _, _, _ in opposite],
                                  how="The members that moved the other way, largest first."))
    total_before = sum(b for _, _, b, _ in moves if b)
    rated = [(n, by / before, before) for n, by, before, _ in moves
             if before and before > 0 and total_before and before / total_before >= SMALL_BASE]
    if len(rated) >= 3:
        name, ratio, before = max(rated, key=lambda x: (x[1], x[0]))
        if ratio >= 0.1:
            small = total_before and before / total_before < 0.08
            facts.append(Fact("fastest", STANDS,
                              f"{name} grew fastest ({percent(ratio, signed=True)})"
                              + (f", from a small base ({say(before)})." if small else "."),
                              f"{name} progresse le plus vite ({percent(ratio, signed=True, lang='fr')})"
                              + (f", sur une petite base ({say(before, 'fr')})." if small else "."),
                              [rounded(ratio), *([before] if small else [])], size=1.8, members=[name],
                              how=f"The change in percent, among {noun} with at least {SMALL_BASE:.0%} of the total before."))
    return reading


# ── two groupings ─────────────────────────────────────────────────────────────


def _grid(rows: list[dict], m: OutColumn, o: OutMeasure | None, a: OutColumn, b: OutColumn,
          unit: str | None) -> Reading:
    reading = Reading(shape="grid")
    cells = [(str(r[a.name]), str(r[b.name]), v) for r in rows if r[a.name] not in (None, "") and
             r[b.name] not in (None, "") for v in [_number(r[m.name])] if v is not None]
    if len(cells) < 4:
        return reading
    say = lambda v, lang="en": amount(v, m.format, unit=unit, lang=lang)   # noqa: E731
    big = max(cells, key=lambda c: (c[2], c[0], c[1]))
    reading.facts.append(Fact("biggest_cell", STANDS,
                              f"The largest is {big[0]} with {big[1]}, at {say(big[2])}.",
                              f"Le plus important : {big[0]} avec {big[1]}, à {say(big[2], 'fr')}.",
                              [big[2]], size=1.4, members=[big[0], big[1]], how="The largest cell of the grid."))
    if not _sums(o, m):
        return reading
    # The mix: the share of each column within each row, and the column whose share differs most between rows.
    rows_of: dict[str, dict[str, float]] = {}
    for x, y, v in cells:
        rows_of.setdefault(x, {})[y] = rows_of.get(x, {}).get(y, 0.0) + v
    if len(rows_of) < 2:
        return reading
    best = None
    for y in {c[1] for c in cells}:
        shares = [(x, row.get(y, 0.0) / sum(row.values())) for x, row in rows_of.items() if sum(row.values()) > 0]
        if len(shares) < 2:
            continue
        hi, lo = max(shares, key=lambda s: s[1]), min(shares, key=lambda s: s[1])
        gap = hi[1] - lo[1]
        if best is None or gap > best[0]:
            best = (gap, y, hi, lo)
    if best is not None and best[0] >= 0.1:
        gap, y, hi, lo = best
        reading.facts.append(Fact("mix", STANDS,
                                  f"{y} is {percent(hi[1])} of {hi[0]}'s {m.label.lower()} but {percent(lo[1])} of "
                                  f"{lo[0]}'s.",
                                  f"{y} représente {percent(hi[1], lang='fr')} pour {hi[0]} mais "
                                  f"{percent(lo[1], lang='fr')} pour {lo[0]}.",
                                  [rounded(hi[1]), rounded(lo[1])], size=1.9, members=[y, hi[0], lo[0]],
                                  how=f"The share of each {_noun(b.label)} within each {_noun(a.label)}, and where it differs most."))
    return reading
