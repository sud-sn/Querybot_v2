"""What the data holds, read off the model: "what can you tell me about my data?" and its kin.

No AI writes this answer. The model knows its subjects, their measures, the dates
they are counted by and what they can be broken down by; an answer about them is
read off it, so it is never wrong about the data and never invents a measure.

* With nothing named, the whole data: each subject (a table measures are counted
  on), what it measures, what it is broken down by and the dates it covers.
* With a measure named ("what does gross profit mean?", "how is margin
  calculated?"): its definition, how it adds up, the date it is counted by and
  the period covered, and what it can be broken down by.
* With a date ("how far back does the data go?"): the period each covers.
* With a breakdown ("what do you know about customers?"): how many there are,
  what they are grouped by, and the measures that can be broken down by them.
* With a subject ("what can I ask about purchasing?"): that subject in full.

Only what the reader may use is described: a subject outside their tables is not
named. Example questions close the answer, each built from a measure and a
breakdown that reach each other, in a year the data covers.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from core2.bootstrap import names
from core2.model.schema import AggExpr, Attribute, DateRole, Entity, Measure, SemanticModel
from core2.plan.catalog import definition
from core2.plan.ir import TIME_ATTRIBUTES
from core2.resolve import paths as P
from core2.resolve.resolver import find_slug

MEASURES_LISTED = 8
EXAMPLES = 4
SMALL = 50          # a breakdown short enough to read as a chart

# How a measure adds up, said to a reader.
_ADDS = {"additive": "It adds up across any breakdown and period.",
         "non_additive": "It does not add up: it is worked out again for each breakdown and period.",
         "semi_additive": "It is a balance: each period shows its last value, never a sum over time."}
_AGG = {"sum": "total", "avg": "average", "count": "number of", "count_distinct": "number of distinct",
        "min": "smallest", "max": "largest"}
# The measure a reader most likely means when they name none: sales and profit, then values and
# balances, then any amount -- and costs, rates and shares after them.
_LEAD = [re.compile(r"\b(revenue|sales|net|profit)\b", re.IGNORECASE),
         re.compile(r"\b(value|balance|on hand|stock)\b", re.IGNORECASE),
         re.compile(r"\b(amount|amt|total)\b", re.IGNORECASE)]
_SIDE = re.compile(r"\b(cost|discount|tax|price|rate|fee|freight|percent|pct|ratio|rejected)\b", re.IGNORECASE)


def _lead_rank(name: str) -> int:
    return next((i for i, words in enumerate(_LEAD) if words.search(name)), len(_LEAD))


def _counts(m: Measure) -> bool:
    return m.kind == "count" or bool(re.match(r"(number|count|no\.?) of\b", m.business_name, re.IGNORECASE))


@dataclass
class Described:
    headline: str
    sections: list[dict] = field(default_factory=list)       # {"title", "body", "bullets"}
    examples: list[str] = field(default_factory=list)
    note: str = ""                                           # a line under the lead


def _span(first: dt.date | None, last: dt.date | None, granularity: str = "day") -> str:
    if not first or not last:
        return ""
    if granularity == "month" or (first.day == 1 and last.month != first.month):
        return f"{first:%B %Y} to {last:%B %Y}"
    return f"{first.day} {first:%B %Y} to {last.day} {last:%B %Y}"


def _listed(items: list[str], limit: int | None = None) -> str:
    shown = items if limit is None or len(items) <= limit else items[:limit]
    rest = len(items) - len(shown)
    if rest:
        return f"{', '.join(shown)} and {rest} more"
    return shown[0] if len(shown) == 1 else f"{', '.join(shown[:-1])} and {shown[-1]}"


class _Reader:
    """The model as one reader may use it: their tables only, hidden things left out."""

    def __init__(self, model: SemanticModel, allowed: set[str] | None, values: bool, today: dt.date):
        self.model, self.allowed, self.values, self.today = model, allowed, values, today
        self._reach: dict[tuple[str, str], int | None] = {}

    def may(self, table: str) -> bool:
        return table in self.model.tables and (self.allowed is None or table in self.allowed)

    def measures(self, table: str | None = None) -> list[Measure]:
        """Visible measures, the ones a reader most likely means first.

        Amounts before counts ("number of deliveries" is an admin's metric, and still a
        count), rates and averages last; sales, values and balances before costs; an
        admin's measure before a learned one of the same standing.
        """
        found = [m for m in self.model.measures.values()
                 if not m.hidden and m.slug and self.may(m.table) and (table is None or m.table == table)]
        return sorted(found, key=lambda m: (_counts(m), m.additivity == "non_additive",
                                            bool(_SIDE.search(m.business_name)), _lead_rank(m.business_name),
                                            m.kind not in ("metric", "import"), m.format != "currency",
                                            m.business_name.casefold()))

    def lead(self, table: str) -> Measure | None:
        return next((m for m in self.measures(table) if not _counts(m)), None)

    def subjects(self) -> list[str]:
        """Tables measures are counted on, the largest first."""
        tables = {m.table for m in self.measures() if m.kind != "count"} or {m.table for m in self.measures()}
        return sorted(tables, key=lambda t: (-(self.model.tables[t].row_count or 0),
                                             self.model.tables[t].business_name.casefold()))

    def name(self, table: str) -> str:
        t = self.model.tables[table]
        return t.business_name or t.name

    def dates(self, table: str) -> list[DateRole]:
        return sorted((r for r in self.model.date_roles.values() if r.table == table and r.kind != "audit"),
                      key=lambda r: (not r.is_default, r.name.casefold()))

    def steps(self, start: str, goal: str) -> int | None:
        """Joins from a measure's table to a breakdown's, or None when it cannot be reached."""
        key = (start, goal)
        if key not in self._reach:
            if start == goal:
                self._reach[key] = 0
            else:
                found = P.all_paths(self.model, start, goal)
                self._reach[key] = min((len(p.joins) for p in found), default=None)
        return self._reach[key]

    def breakdowns(self, table: str) -> list[Entity]:
        """What a subject's measures can be broken down by: the nearest first."""
        found = [(steps, e) for e in self.model.entities.values()
                 if self.may(e.table) and (steps := self.steps(table, e.table)) is not None]
        return [e for _, e in sorted(found, key=lambda f: (f[0], f[1].business_name.casefold()))]

    def one_way(self, start: str, goal: str) -> bool:
        """Reached without the reader having to name a role (bill-to or ship-to customer)."""
        try:
            return P.best_path(self.model, start, goal) is not None
        except P.Ambiguous:
            return False

    def chartable(self, table: str) -> Entity | None:
        """A breakdown for an example question: another table's, reached one way, short enough for a chart."""
        found = [e for e in self.breakdowns(table) if e.table != table and self.one_way(table, e.table)]
        return min(found, key=lambda e: (not 0 < e.members <= SMALL, self.steps(table, e.table) or 0,
                                         e.members or 0), default=None)

    def word(self, e: Entity) -> str:
        return (e.business_name or e.slug).lower()

    def year(self, table: str) -> int | None:
        """The latest year a subject's data covers, never past this one (budgets are set ahead)."""
        role = next(iter(self.dates(table)), None)
        return min(role.last.year, self.today.year) if role is not None and role.last else None


def _when(reader: _Reader, table: str) -> str:
    year = reader.year(table)
    return f" in {year}" if year else ""


def _example_questions(reader: _Reader, subjects: list[str]) -> list[str]:
    """A question per subject (its lead measure by a short breakdown), then a trend: each one answerable."""
    asked: list[str] = []
    trend: str | None = None
    for table in subjects:
        lead = reader.lead(table)
        if lead is None:
            continue
        by = reader.chartable(table)
        if by is not None:
            asked.append(f"{lead.business_name} by {reader.word(by)}{_when(reader, table)}")
        if trend is None and reader.dates(table):
            trend = f"{lead.business_name} by month{_when(reader, table)}"
        if len(asked) >= EXAMPLES - 1:
            break
    return (asked + ([trend] if trend else []))[:EXAMPLES]


def _subject_section(reader: _Reader, table: str, *, full: bool = False) -> dict:
    measures = [m.business_name for m in reader.measures(table) if m.kind != "count"] or \
        [m.business_name for m in reader.measures(table)]
    measures = list(dict.fromkeys(n.lower() for n in measures))        # two columns can carry one name
    bullets = [f"Measures: {_listed(measures, None if full else MEASURES_LISTED)}"]
    breakdowns = [reader.word(e) for e in reader.breakdowns(table)]
    if breakdowns:
        bullets.append(f"Broken down by: {_listed(breakdowns, None if full else 10)}")
    dates = reader.dates(table)
    if dates:
        named = _listed([f"{r.name.lower()}{' (the default)' if r.is_default and len(dates) > 1 else ''}"
                         for r in dates], None if full else 4)
        span = _span(dates[0].first, dates[0].last, dates[0].granularity)
        bullets.append(f"Dates: {named}{f'; {span}' if span else ''}")
    return {"title": reader.name(table), "body": "", "bullets": bullets}


def _overview(reader: _Reader) -> Described:
    subjects = reader.subjects()
    if not subjects:
        return Described("There is no data here you can ask about yet.")
    # The span of what happened: a budget or target set into next year is not data yet.
    spans = [r for t in subjects for r in reader.dates(t)[:1] if r.first and r.last and r.last <= reader.today]
    first = min((r.first for r in spans if r.first), default=None)
    last = max((r.last for r in spans if r.last), default=None)
    covered = f", with data from {first:%B %Y} to {last:%B %Y}" if first and last else ""
    named = _listed([reader.name(t) for t in subjects])
    head = (f"This data covers {len(subjects)} subjects: {named}{covered}." if len(subjects) > 1
            else f"This data covers {named}{covered}.")
    return Described(head, [_subject_section(reader, t) for t in subjects], _example_questions(reader, subjects),
                     "Ask for a total, a breakdown, a trend, a ranking or a comparison of any of them, "
                     "or what a measure means.")


def _measure_sentence(reader: _Reader, m: Measure) -> str:
    """ "Net amount is the total net amount across order lines." """
    said = definition(reader.model, m.expr, reader.values)
    rows = names.plural(reader.name(m.table).lower())
    if isinstance(m.expr, AggExpr):
        target = reader.model.columns[m.expr.column].business_name.lower() if m.expr.column else "rows"
        where = said.split(" where ", 1)[1] if " where " in said else ""
        counted = names.plural(target) if m.expr.agg in ("count", "count_distinct") else target
        what = f"the {_AGG[m.expr.agg]} {counted}" if target != "rows" else f"the number of {rows}"
        return f"{m.business_name} is {what}{f' across {rows}' if target != 'rows' else ''}" + \
            (f", counting only rows where {where}" if where else "") + "."
    return f"{m.business_name} is worked out as {said}, on {rows}."


def _measure(reader: _Reader, m: Measure) -> Described:
    model = reader.model
    bullets = []
    if m.description:
        bullets.append(m.description)
    bullets.append(_ADDS.get(m.additivity, ""))
    if m.unit:
        bullets.append(f"Amounts are in {m.unit}." if m.format == "currency" else f"It is counted in {m.unit}.")
    dates = reader.dates(m.table)
    wanted = m.default_date or model.tables[m.table].default_date
    default = next((r for r in dates if r.key == wanted), dates[0] if dates else None)
    if default is not None:
        span = _span(default.first, default.last, default.granularity)
        bullets.append(f"It is counted by {default.name.lower()}{f', from {span}' if span else ''}.")
    breakdowns = [reader.word(e) for e in reader.breakdowns(m.table)]
    if breakdowns:
        bullets.append(f"It can be broken down by {_listed(breakdowns)}.")
    if m.status in ("needs_review", "proposed"):
        bullets.append("An admin has not confirmed this measure yet.")
    examples = [f"{m.business_name} by month{_when(reader, m.table)}"] if dates else []
    by = reader.chartable(m.table)
    if by is not None:
        examples.append(f"{m.business_name} by {reader.word(by)}{_when(reader, m.table)}")
    return Described(_measure_sentence(reader, m),
                     [{"title": m.business_name, "body": "", "bullets": [b for b in bullets if b]}], examples)


def _date(reader: _Reader, role: DateRole) -> Described:
    span = _span(role.first, role.last, role.granularity)
    grain = {"month": "monthly", "timestamp": "a date and time"}.get(role.granularity, "daily")
    head = (f"{role.name} ({reader.name(role.table).lower()}) runs from {span}." if span
            else f"{role.name} is a date on {reader.name(role.table).lower()}.")
    counted = list(dict.fromkeys(m.business_name.lower() for m in reader.measures(role.table) if m.kind != "count"))
    bullets = [f"It is {grain}."]
    if counted:
        bullets.append(f"Measures counted by it: {_listed(counted, MEASURES_LISTED)}.")
    for t in reader.subjects():
        other = next(iter(reader.dates(t)), None)
        if t != role.table and other is not None and (said := _span(other.first, other.last, other.granularity)):
            bullets.append(f"{reader.name(t)} ({other.name.lower()}): {said}.")
    lead = reader.lead(role.table)
    examples = [f"{lead.business_name} by month{_when(reader, role.table)}"] if lead is not None else []
    return Described(head, [{"title": role.name, "body": "", "bullets": bullets}], examples)


def _group(reader: _Reader, thing: Entity | Attribute) -> Described:
    model = reader.model
    bullets: list[str] = []
    if isinstance(thing, Entity):
        table, word = thing.table, reader.word(thing)
        label = model.columns[thing.label_column].business_name.lower() if thing.label_column else ""
        head = (f"There are {thing.members:,} {names.plural(word)} in the data" if thing.members
                else f"The data can be broken down by {word}")
        head += f", each named by its {label}." if label and label != word else "."
        own = sorted(a.business_name.lower() for a in model.attributes.values()
                     if model.columns[a.column].table == table and a.column != thing.label_column
                     and not model.columns[a.column].hidden and model.columns[a.column].sensitivity == "none")
        if own:
            bullets.append(f"They can be grouped and filtered by {_listed(own)}.")
    else:
        column = model.columns[thing.column]
        table, word = column.table, thing.business_name.lower()
        count = thing.members or (column.profile.distinct if column.profile else 0)
        p = column.profile
        shown = sorted(column.value_names.get(str(t.value), str(t.value)) for t in p.top if t.value is not None) if (
            reader.values and column.values_allowed and column.sensitivity == "none" and p and p.top
            and 0 < count <= 12) else []
        head = (f"{thing.business_name} has {count:,} values: {_listed(shown)}." if shown
                else f"{thing.business_name} has {count:,} values." if count
                else f"The data can be grouped by {word}.")
    measured = [m for m in reader.measures() if not _counts(m) and reader.steps(m.table, table) is not None]
    if measured:
        bullets.append(f"Measures that can be broken down by {word}: "
                       f"{_listed(list(dict.fromkeys(m.business_name.lower() for m in measured)), MEASURES_LISTED)}.")
    examples = [f"List the {names.plural(word)}"] if isinstance(thing, Entity) else []
    if measured:
        examples.append(f"{measured[0].business_name} by {word}{_when(reader, measured[0].table)}")
    return Described(head, [{"title": thing.business_name, "body": "", "bullets": bullets}], examples)


def _find(reader: _Reader, wanted: str) -> Measure | DateRole | Entity | Attribute | str | None:
    """What an ``about`` entry names, if the reader may use it: a slug, or a subject's name."""
    found = find_slug(reader.model, wanted)
    thing = found[1] if found is not None else None
    if isinstance(thing, (Measure, DateRole, Entity)) and reader.may(thing.table):
        return thing
    if isinstance(thing, Attribute) and thing.column in reader.model.columns \
            and reader.may(reader.model.columns[thing.column].table):
        return thing
    def plain(text: str) -> str:
        return " ".join(names.singular(w) for w in text.strip().lstrip("#").casefold().split())

    return next((t for t in reader.subjects() if plain(reader.name(t)) == plain(wanted)), None)


def describe(model: SemanticModel, about: list[str], *, today: dt.date, allowed: set[str] | None = None,
             values: bool = True) -> Described:
    """The answer to a question about the data itself, for a reader limited to ``allowed`` tables."""
    reader = _Reader(model, allowed, values, today)
    parts: list[Described] = []
    missing: list[str] = []
    for wanted in dict.fromkeys(a for a in about if a and a.strip() and a not in TIME_ATTRIBUTES):
        thing = _find(reader, wanted)
        if isinstance(thing, Measure):
            parts.append(_measure(reader, thing))
        elif isinstance(thing, DateRole):
            parts.append(_date(reader, thing))
        elif isinstance(thing, (Entity, Attribute)):
            parts.append(_group(reader, thing))
        elif isinstance(thing, str):
            parts.append(Described(f"Here is what you can ask about {reader.name(thing)}.",
                                   [_subject_section(reader, thing, full=True)],
                                   _example_questions(reader, [thing])))
        else:
            missing.append(wanted.removeprefix("time:").replace("_", " "))
    if not parts:
        whole = _overview(reader)
        if missing:
            whole.note, whole.headline = whole.headline, f"This data has nothing called {_listed(missing)}."
        return whole
    if len(parts) == 1:
        return parts[0]
    return Described(" ".join(p.headline for p in parts),
                     [s for p in parts for s in p.sections],
                     list(dict.fromkeys(q for p in parts for q in p.examples))[:EXAMPLES])
