"""Looking at an answer before it is shown, as an analyst would, and saying what was seen.

An analyst handed a figure checks it before passing it on:

* nothing found: is there anything for those conditions in another period -- the last day with some, or the
  first -- and for everyone else in this period? Then it is this member, not the month; or there is nothing for
  it in any period at all;
* a figure resting on a few days of its period (two days of a month, three months of a year) says so and names
  them; one whose rows begin late in the period, or stop early, says on which day;
* a period that starts before the data does says where the data starts.

Each check is one small query through the ordinary resolver and compiler, so joins, dates, the reader's access and
the rows left out by default are the question's own; at most MAX_CHECKS of them, within BUDGET_SECONDS, and a check
that fails costs only what it would have said, never the answer. What was seen is said in English and in French
with the figures and days it states (core2/agent/talk.py rewords it, and checks every figure against them), what
was checked is listed, and the next questions offered are plans the conversation runs with no AI: clicked, typed,
or answered "yes".
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from core2.answer.builder import _MOIS, _MONTHS, _jour, condition_words, span_words, spoken
from core2.answer.extra import _base, _col, _number, _rows
from core2.answer.facts import amount
from core2.bootstrap.names import plural
from core2.plan.ir import Plan
from core2.resolve.resolver import Condition, Logical, _first_last
from core2.resolve.time import DAY, Range, add_units, periods, unit_start
from core2.resolve.time import label as period_label

log = logging.getLogger("querybot.core2")

MAX_CHECKS = 6               # small queries one answer may ask
BUDGET_SECONDS = 12.0        # checks are not started after this
FEW = 0.4                    # periods with data at or under this share of the period's are "only k of n"
EDGE_DAYS = 3                # data starting this many days into the period still covers it (a holiday, a weekend)

Run = Callable[[Plan], tuple[Any, Any, Any]]       # a plan -> (its logical, its compiled query, its result)

_UNITS = {"day": ("day", "days", "jour", "jours"), "week": ("week", "weeks", "semaine", "semaines"),
          "month": ("month", "months", "mois", "mois")}


@dataclass
class Seen:
    """One thing seen, as the reply says it: its words, and the figures, days and names those words state."""

    kind: str           # never | last_seen | first_seen | between | elsewhere | few_days | starts_late |
    #                     begins_late | stops_early
    en: str
    fr: str
    figures: list[float] = field(default_factory=list)
    days: list[str] = field(default_factory=list)        # ISO days (a month or week by its first day)
    members: list[str] = field(default_factory=list)     # the data's names it states (kept from the AI if need be)


@dataclass
class Offer:
    en: str
    fr: str
    plan: Plan


@dataclass
class Look:
    seen: list[Seen] = field(default_factory=list)
    offers: list[Offer] = field(default_factory=list)
    checked: list[tuple[str, str]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.seen)

    def reply(self) -> dict[str, Any]:
        """The reply as the answer carries it: what was seen in a few sentences, each thing seen with its figures,
        what was checked, and the next questions, each a button the conversation runs with no AI."""
        said = self.seen[:3]
        return {"en": " ".join(s.en for s in said), "fr": " ".join(s.fr for s in said), "by": "facts",
                "seen": [asdict(s) for s in self.seen],
                "checked": [{"en": en, "fr": fr} for en, fr in self.checked],
                "offers": [{"label": o.en, "question": o.en, "label_fr": o.fr, "question_fr": o.fr}
                           for o in self.offers]}


# ── what is looked at ─────────────────────────────────────────────────────────


def looked_at(plan: Plan, logical: Logical) -> bool:
    """Is this an answer the checks read: a figure of dated rows, over its period, not a shape worked out over
    them (members by what they did, ranges, a running series), a comparison, a why or a forecast?"""
    if plan.kind != "query" or plan.intent in ("drivers", "forecast", "list") or plan.shaped():
        return False
    if plan.segment is not None or plan.target is not None or plan.rates or plan.durations:
        return False
    if plan.time.compare is not None or logical.compare is not None or logical.people or logical.shape is not None:
        return False
    if not logical.parts or logical.parts[0].date is None or any(part.snapshot for part in logical.parts):
        return False
    return any(not m.hidden for m in logical.measures)


def look(plan: Plan, logical: Logical, payload: dict[str, Any], run: Run, *, empty: bool,
         budget: float = BUDGET_SECONDS) -> Look | None:
    """What an analyst would say of this answer before showing it, from at most MAX_CHECKS small queries; None
    when there is nothing to say (a busy answer) or nothing it reads."""
    if not looked_at(plan, logical):
        return None
    words = _Words(logical)
    checks = _Checks(run, budget)
    try:
        found = _nothing(plan, logical, words, checks) if empty else _thin(plan, logical, payload, words, checks)
    except Exception as exc:  # noqa: BLE001 - the answer stands as it is
        log.warning("core2: looking at an answer failed: %s", exc, exc_info=True)
        return None
    return found if found else None


# ── nothing found ─────────────────────────────────────────────────────────────


def _nothing(plan: Plan, logical: Logical, words: "_Words", checks: "_Checks") -> Look:
    out = Look()
    window = logical.window
    if window.start is not None or window.end is not None:
        months = checks.series(_base(plan, intent="trend", time={"window": {"kind": "all"}, "grain": "month"}),
                               words.name)
        if months is not None:
            _when_else(plan, logical, words, checks, months, out)
    for f in [f for f in plan.filters if f.op in ("eq", "in") and f.values][:2]:
        _elsewhere(plan, logical, words, checks, f, out)
    return out


def _when_else(plan: Plan, logical: Logical, words: "_Words", checks: "_Checks", months: dict[dt.date, float],
               out: Look) -> None:
    """Anything for these conditions in any period: the last day with some before the period asked about, the
    first after it, or none at all."""
    window = logical.window
    span, span_fr = span_words(window), span_words(window, "fr")
    with_any = sorted(d for d, v in months.items() if _has(v))
    if not with_any:
        out.seen.append(Seen("never", f"There is nothing {words.about} in any period of the data.".replace("  ", " "),
                             f"Il n'y a rien {words.about_fr}, sur aucune période des données.".replace("  ", " "),
                             members=words.members))
        out.checked.append((f"Looked for {words.low} {words.about} in every month of the data: none.".replace("  ", " "),
                            f"Cherché {words.label} {words.about_fr} sur chaque mois des données : rien.".replace("  ", " ")))
        return
    last = first = None
    before = [m for m in with_any if window.start is not None and m < window.start]
    after = [m for m in with_any if window.end is not None and add_units(m, "month", 1) > window.end]
    if before:
        days = checks.series(_month_days(plan, before[-1]), words.name)
        inside = [d for d, v in (days or {}).items() if _has(v) and d < window.start]   # type: ignore[operator]
        last = max(inside) if inside else None
    if after:
        days = checks.series(_month_days(plan, after[0]), words.name)
        inside = [d for d, v in (days or {}).items() if _has(v) and d >= window.end]     # type: ignore[operator]
        first = min(inside) if inside else None
    if last is None and first is None:
        return
    lead = " ".join(w for w in ("Nothing", words.about, span) if w)
    lead_fr = " ".join(w for w in ("Rien", words.about_fr, span_fr) if w)
    if last is not None and first is not None:
        kind = "between"
        en = f"{lead}: the last day before it with any is {_day(last)}, and the next is {_day(first)}."
        fr = f"{lead_fr} : le dernier jour avant avec des données est le {_jour(last)}, et le suivant le {_jour(first)}."
        days = [last.isoformat(), first.isoformat()]
    elif last is not None:
        kind, days = "last_seen", [last.isoformat()]
        en = f"{lead}: the last day with any is {_day(last)}."
        fr = f"{lead_fr} : le dernier jour avec des données est le {_jour(last)}."
    else:
        assert first is not None
        kind, days = "first_seen", [first.isoformat()]
        en = f"{lead}: the first day with any is {_day(first)}."
        fr = f"{lead_fr} : le premier jour avec des données est le {_jour(first)}."
    out.seen.append(Seen(kind, en, fr, days=days, members=words.members))
    out.checked.append((f"Looked for {words.low} {words.about} in every period of the data: "
                        f"{len(with_any)} month{'s' if len(with_any) != 1 else ''} with some.".replace("  ", " "),
                        f"Cherché {words.label} {words.about_fr} sur toutes les périodes des données : "
                        f"{len(with_any)} mois avec des données.".replace("  ", " ")))
    # The next questions: the member over every month, and the month it last (or next) had some.
    out.offers.append(Offer(f"{words.label} by month{words.tail}", f"{words.label} par mois{words.tail_fr}",
                            _base(plan, intent="trend", time={"window": {"kind": "all"}, "grain": "month"})))
    for day in [d for d in (last, first) if d is not None]:
        start = day.replace(day=1)
        end = add_units(start, "month", 1) - DAY
        out.offers.append(Offer(f"{words.label} in {_MONTHS[start.month - 1]} {start.year}{words.tail}",
                                f"{words.label} en {_MOIS[start.month - 1]} {start.year}{words.tail_fr}",
                                _base(plan, time={"window": {"kind": "between", "start": start.isoformat(),
                                                             "end": end.isoformat()}})))


def _elsewhere(plan: Plan, logical: Logical, words: "_Words", checks: "_Checks", f: Any, out: Look) -> None:
    """Everyone else in the same period: without the condition on one field, is there anything?"""
    others = [g for g in plan.filters if g is not f]
    rows = checks.one(_base(plan, filters=[g.model_dump(mode="json", exclude_defaults=True) for g in others]),
                      words.name)
    if rows is None or not _has(rows):
        return
    condition = words.condition_of(f)
    thing = condition.label if condition is not None else f.field.split(".")[-1].replace("_", " ")
    rest = [c for c in words.conditions if c is not condition]
    rest_en = " ".join(w for w in (condition_words(c) for c in rest) if w)
    rest_fr = " ".join(w for w in (condition_words(c, lang="fr") for c in rest) if w)
    span, span_fr = span_words(logical.window), span_words(logical.window, "fr")
    value = float(rows)
    said = spoken(value, words.format)
    said_fr = amount(value, words.format, lang="fr")
    scope = " ".join(w for w in (span, rest_en) if w)
    scope_fr = " ".join(w for w in (span_fr, rest_fr) if w)
    out.seen.append(Seen("elsewhere", f"Across all {plural(thing)}, {words.low} {scope} is {said}.".replace("  ", " "),
                         f"Sans la condition sur {thing}, {words.label} {scope_fr} : {said_fr}.".replace("  ", " "),
                         figures=[value]))
    out.checked.append((f"Looked across all {plural(thing)} {scope}: {said}.".replace("  ", " "),
                        f"Cherché sans la condition sur {thing} {scope_fr} : {said_fr}.".replace("  ", " ")))
    tail = "".join(f", {w}" for w in (condition_words(c, quoted=True) for c in rest) if w)
    tail_fr = "".join(f", {w}" for w in (condition_words(c, quoted=True, lang="fr") for c in rest) if w)
    out.offers.append(Offer(f"{words.label} by {thing} {span}{tail}".replace("  ", " "),
                            f"{words.label} par {thing} {span_fr}{tail_fr}".replace("  ", " "),
                            _base(plan, intent="breakdown", group_by=[f.field],
                                  filters=[g.model_dump(mode="json", exclude_defaults=True) for g in others])))


def _month_days(plan: Plan, month: dt.date) -> Plan:
    end = add_units(month, "month", 1) - DAY
    return _base(plan, intent="trend", time={"window": {"kind": "between", "start": month.isoformat(),
                                                        "end": end.isoformat()}, "grain": "day"})


# ── a figure resting on a few days ───────────────────────────────────────────


def _thin(plan: Plan, logical: Logical, payload: dict[str, Any], words: "_Words", checks: "_Checks") -> Look:
    out = Look()
    window = logical.window
    if window.start is None or window.end is None:
        return out
    role = logical.parts[0].date.role
    first_data, last_data = _first_last(role)
    start, end = window.start, window.end
    if first_data is not None and (first_data - window.start).days > EDGE_DAYS and first_data < window.end:
        start = first_data
        out.seen.append(Seen(
            "starts_late",
            f"The data starts on {_day(first_data)}, so this counts {words.low} from then only, not from "
            f"{_day(window.start)}.",
            f"Les données commencent le {_jour(first_data)} : ce chiffre ne compte {words.label} qu'à partir de "
            f"cette date, pas depuis le {_jour(window.start)}.",
            days=[first_data.isoformat()]))
        out.checked.append((f"Checked where the data starts: {_day(first_data)}.",
                            f"Vérifié où commencent les données : le {_jour(first_data)}."))
    if last_data is not None and last_data + DAY < end:
        end = last_data + DAY          # the data ends inside the period: the answer's own notes say so
    if start >= end:
        return out
    groups = [g for g in logical.groups if g.name != logical.unit_group]
    own = groups[0] if len(groups) == 1 and groups[0].kind == "period" and groups[0].grain in _UNITS else None
    if own is None and (groups or plan.group_by or plan.time.grain):
        return out                     # a breakdown: each member's days are its own
    grain = own.grain if own is not None else "day" if (end - start).days <= 92 else \
        "month" if (end - start).days <= 1100 else None
    if grain is None:
        return out
    if own is not None:
        series = _own_series(payload, own.name, words.name)
    else:
        series = checks.series(_base(plan, intent="trend", time={"grain": grain}), words.name)
    if series is None:
        return out
    every = [p for p in periods(Range(start, end), grain) if p >= unit_start(start, grain)]
    with_any = sorted(p for p, v in series.items() if _has(v) and start <= p + _length(p, grain) - DAY
                      and p < end)
    n, k = len(every), len(with_any)
    if n < 3 or k == 0 or k >= n:
        return out
    one, many, un, des = _UNITS[grain]
    span = span_words(Range(start, end))
    span_fr = span_words(Range(start, end), "fr")
    shown = [_period(p, grain) for p in with_any]
    shown_fr = [_period(p, grain, "fr") for p in with_any]
    if k <= 2 or k <= FEW * n:
        if k <= 3:
            listed, listed_fr = _listed(shown), _listed(shown_fr, "et")
            days = [p.isoformat() for p in with_any]
        else:
            listed = f"the first {shown[0]}, the last {shown[-1]}"
            listed_fr = f"le premier {shown_fr[0]}, le dernier {shown_fr[-1]}"
            days = [with_any[0].isoformat(), with_any[-1].isoformat()]
        only_fr = ("Seule" if k == 1 else "Seules") if grain == "week" else ("Seul" if k == 1 else "Seuls")
        has, a = ("has", "a") if k == 1 else ("have", "ont")
        out.seen.append(Seen(
            "few_days",
            f"Only {k} of the {n} {many} {span} {has} {words.low} {words.about}: {listed}.".replace("  ", " "),
            f"{only_fr} {k} des {n} {des} {span_fr} {a} des données ({words.label} {words.about_fr}) : "
            f"{listed_fr}.".replace("  ", " ").replace("( ", "(").replace(" )", ")"),
            figures=[float(k), float(n)], days=days, members=words.members))
        out.checked.append((f"Counted the {many} {span} with {words.low} {words.about}: {k} of {n}.".replace("  ", " "),
                            f"Compté les {des} {span_fr} avec des données : {k} sur {n}.".replace("  ", " ")))
        if own is None:
            out.offers.append(Offer(f"{words.label} by {one} {span}{words.tail}".replace("  ", " "),
                                    f"{words.label} par {un} {span_fr}{words.tail_fr}".replace("  ", " "),
                                    _base(plan, intent="trend", time={"grain": grain})))
        return out
    # Most periods have some: does it begin late, or stop early, while the data goes on?
    slack = max(2, round(0.1 * n))
    late = every.index(with_any[0]) if with_any[0] in every else 0
    early = len(every) - 1 - every.index(with_any[-1]) if with_any[-1] in every else 0
    if words.conditions and late >= slack:
        out.seen.append(Seen("begins_late",
                             f"The first {one} {span} with {words.low} {words.about} is {shown[0]}.".replace("  ", " "),
                             f"Le premier {un} {span_fr} avec des données ({words.label} {words.about_fr}) : "
                             f"{shown_fr[0]}.".replace("  ", " "),
                             days=[with_any[0].isoformat()], members=words.members))
    if words.conditions and early >= slack and (last_data is None or last_data >= end - DAY):
        out.seen.append(Seen("stops_early",
                             f"The last {one} {span} with {words.low} {words.about} is {shown[-1]}; nothing after "
                             "it.".replace("  ", " "),
                             f"Le dernier {un} {span_fr} avec des données ({words.label} {words.about_fr}) : "
                             f"{shown_fr[-1]} ; rien après.".replace("  ", " "),
                             days=[with_any[-1].isoformat()], members=words.members))
    if out.seen and not out.checked:
        out.checked.append((f"Counted the {many} {span} with {words.low} {words.about}: {k} of {n}.".replace("  ", " "),
                            f"Compté les {des} {span_fr} avec des données : {k} sur {n}.".replace("  ", " ")))
    return out


def _own_series(payload: dict[str, Any], period: str, measure: str) -> dict[dt.date, float] | None:
    rows = payload.get("export_rows") or (payload.get("data") or {}).get("rows") or []
    out: dict[dt.date, float] = {}
    for row in rows:
        day = _as_day(row.get(period))
        if day is not None:
            out[day] = _number(row.get(measure)) or 0.0
    return out


def _length(start: dt.date, grain: str) -> dt.timedelta:
    return add_units(start, grain, 1) - start


# ── checks ────────────────────────────────────────────────────────────────────


class _Checks:
    """The small queries, at most MAX_CHECKS and none started after the budget; a failure is None."""

    def __init__(self, run: Run, budget: float) -> None:
        self.run, self.left, self.until = run, MAX_CHECKS, time.monotonic() + budget

    def _rows(self, plan: Plan) -> tuple[Any, list[dict[str, Any]]] | None:
        if self.left <= 0 or time.monotonic() > self.until:
            return None
        self.left -= 1
        try:
            _, compiled, result = self.run(plan)
        except Exception as exc:  # noqa: BLE001 - only what this check would have said is lost
            log.info("core2: a check of an answer failed: %s", exc)
            return None
        return compiled, _rows(compiled, result)

    def one(self, plan: Plan, measure: str) -> float | None:
        got = self._rows(plan)
        if got is None:
            return None
        compiled, rows = got
        name = _col(compiled, "measure") or measure
        return _number(rows[0].get(name)) if len(rows) == 1 else None

    def series(self, plan: Plan, measure: str) -> dict[dt.date, float] | None:
        got = self._rows(plan)
        if got is None:
            return None
        compiled, rows = got
        period, name = _col(compiled, "period"), _col(compiled, "measure") or measure
        if period is None:
            return None
        out: dict[dt.date, float] = {}
        for row in rows:
            day = _as_day(row.get(period))
            if day is not None:
                out[day] = _number(row.get(name)) or 0.0
        return out


# ── words ─────────────────────────────────────────────────────────────────────


class _Words:
    """How the answer's figure and conditions read, in English and French."""

    def __init__(self, logical: Logical) -> None:
        measure = next(m for m in logical.measures if not m.hidden)
        self.name, self.label, self.format = measure.name, measure.label, measure.format
        # "net amount" mid-sentence, never "net Amount"; "GMV" and "SKU count" keep their capitals.
        self.low = " ".join(w if w[:2].isupper() else w.lower() for w in self.label.split(" "))
        self.conditions: list[Condition] = [c for c in logical.conditions if c.kind != "by"]
        self.about = " ".join(w for w in (condition_words(c) for c in self.conditions) if w)
        self.about_fr = " ".join(w for w in (condition_words(c, lang="fr") for c in self.conditions) if w)
        self.tail = "".join(f", {w}" for w in (condition_words(c, quoted=True) for c in self.conditions) if w)
        self.tail_fr = "".join(f", {w}" for w in (condition_words(c, quoted=True, lang="fr")
                                                  for c in self.conditions) if w)
        self.members = [str(v) for c in self.conditions if c.kind == "member" for v in c.values]

    def condition_of(self, f: Any) -> Condition | None:
        """The answer's condition a plan filter became: the one naming the same values."""
        values = [str(v) for v in f.values]
        return next((c for c in self.conditions if c.kind == "member" and [str(v) for v in c.values] == values),
                    None)


def _has(value: Any) -> bool:
    number = _number(value)
    return number is not None and number != 0


def _as_day(value: Any) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str) and len(value) >= 10:
        try:
            return dt.date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _day(day: dt.date) -> str:
    return f"{day.day} {day:%b %Y}"


def _period(start: dt.date, grain: str, lang: str = "en") -> str:
    if grain == "day":
        return _jour(start) if lang == "fr" else _day(start)
    if grain == "week":
        return f"semaine du {_jour(start)}" if lang == "fr" else f"week of {_day(start)}"
    if lang == "fr":
        return f"{_MOIS[start.month - 1]} {start.year}"
    return period_label(start, grain)


def _listed(items: list[str], word: str = "and") -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" {word} {items[-1]}"
