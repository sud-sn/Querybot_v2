"""Analytical shapes: figures worked out over the rows of a plain answer.

* per member, then over the members (``plan.per``): "average orders per customer", "median spend per
  customer", "how many customers had 3 or more orders", "90th-percentile days to deliver". Each member's figure
  is the plain answer grouped by the member as well; the summary is taken over those figures. By default the
  members are those with rows in the window; ``everyone`` counts every listed member, one with no rows as 0;
* a rank inside each member of a grouping (``plan.within``): "rank stores within each region", "top 3 products
  per store";
* a series accumulated or smoothed (``plan.rolling``): read back far enough before the first period shown (a
  3-month average of January needs November and December), worked out over a gap-free run of periods, then cut
  to the periods asked for;
* ranges of each member's figure (``plan.bands`` with ``per``): "customers by spend band";
* members with no activity (``plan.absent``): everyone listed, minus those with a matching row in the window.

Ranges of each row's own value ("orders by delivery time") are a grouping of the plain answer (resolver.py).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from core2.bootstrap.names import plural
from core2.model.schema import AggExpr, Attribute, DateRole, Entity, Measure, SemanticModel
from core2.plan.ir import Plan
from core2.resolve import bands as B
from core2.resolve import paths as P
from core2.resolve.resolver import (
    Condition,
    Context,
    DateUse,
    DaysBetween,
    Group,
    Logical,
    OutMeasure,
    Part,
    PartGroup,
    Pred,
    ResolveError,
    _Aliases,
    _attribute_label,
    _date_role_for,
    _documents,
    _duration_dates,
    _entity_label,
    _first_last,
    _named_filter,
    _output_name,
    _PartBuilder,
    _reaches,
    _shown,
    find_slug,
    one_total,
    resolve_plain,
)
from core2.resolve.time import Range, add_units, periods, resolve_window, unit_start

SUMMARY_WORDS = {"avg": "Average", "median": "Median", "min": "Lowest", "max": "Highest"}


@dataclass
class PerShape:
    inner: Logical                 # the plain answer grouped by the member as well
    members: list[str]             # inner group names that make one member (its name and code); [] = each row
    value: str                     # the inner column holding each member's figure
    summary: str
    p: float | None
    op: str | None
    values: list[float]
    member: str                    # "customer", "order", "fill" (each row: the rows' table, "order line")
    figure: str                    # what each member's figure is: "Net amount"
    listed: Part | None = None     # every listed member, counted on its own table (everyone)
    band: PartGroup | None = None  # ranges of each member's figure ("customers by spend band")
    row_level: bool = False        # each row's own value, not added up per member


@dataclass
class WithinShape:
    partition: list[str]           # output names of the grouping each rank starts again in
    by: str                        # the measure ranked by
    desc: bool
    member: str                    # the grouping ranked (ties broken by its name)
    top: int | None
    rank: str = "rank"


@dataclass
class RollingShape:
    kind: str                      # running | moving_average | moving_sum
    periods: int
    reset: str | None
    grain: str
    shown: Range                   # the periods asked for
    read: Range                    # the periods the query reads: further back, for the first points shown
    columns: dict[str, str]        # measure output name -> its rolling column's name
    labels: dict[str, str]         # measure output name -> its rolling column's label
    adds: dict[str, bool]          # measure output name -> whether a period with no rows is 0
    expected: list[dt.date] = field(default_factory=list)   # every period start read, gap-free


@dataclass
class AbsentShape:
    entity: Part                   # the listed members, their name (and code) and key
    key: str                       # the entity's key column key
    activity: Part                 # rows that are activity, by the key they reach
    activity_key: str              # the activity part's output name for that key
    had: Part | None = None
    had_key: str = ""
    had_in: Range | None = None    # when they did have rows ("bought in Q1 but not in Q2"); None: any time
    before: Part | None = None
    before_key: str = ""
    member: str = ""               # "customer"
    activity_label: str = ""       # "orders"
    member_key: str = ""           # the listed part's output name for each member's key (never shown)


@dataclass
class Events:
    """Each member's days with activity, each day once (``core2/compile/shapes.py`` reads them in date order)."""

    part: Part                     # SELECT DISTINCT member key, day [, the member's own groupings]
    key: str                       # the part's output name for each member's key
    day: str                       # the part's output name for the day
    groups: list[str]              # the part's output names of the member's own groupings ("customer segment")
    member: str                    # "customer"
    activity: str                  # "orders"
    grain: str | None              # the periods the answer is in (a series), or None
    window: Range                  # the answer's window: the days it is about


@dataclass
class GapShape:
    events: Events
    summary: str                   # avg | median | min | max | within | nth
    nth: int | None = None
    within_days: int | None = None


@dataclass
class LifecycleShape:
    events: Events
    status: str                    # all | active | new | returning | won_back | lapsed | churn_rate
    columns: dict[str, str]        # each figure worked out -> its output name


@dataclass
class CohortShape:
    events: Events
    periods: int                   # periods after the first shown
    value: str                     # share | count
    cohort: str                    # the output name of the cohort's first period
    age: str                       # the output name of the periods after it


@dataclass
class ActiveShape:
    part: Part                     # the rows, with their groupings (a department) and both dates
    start: DateUse
    end: DateUse
    periods: list[tuple[dt.date, dt.date]]   # each period's first day and the next one's
    status: str                    # all | active | starts | ends | end_rate
    columns: dict[str, str]        # each figure worked out -> its output name
    groups: list[str]              # the part's own groupings, by output name
    things: str                    # "employees"
    start_name: str                # "hire date"
    end_name: str                  # "termination date"
    grain: str


MAX_PERIODS = 60


@dataclass
class OnTimeShape:
    part: Part                     # on time: the rows done (a receipt); ageing: the rows due (an order line)
    due: DateUse
    done: DateUse | None           # on the part's rows (or ones they reach); None: ageing against another table
    grace: int
    ageing: bool
    as_of: dt.date                 # ageing: the day open rows are aged on
    groups: list[str]              # the part's own groupings, by output name (a period first, when a series)
    columns: dict[str, str]        # each figure worked out -> its output name
    band: str = ""                 # ageing: the range grouping's output name
    # Ageing with the done rows on another table (receipts of an order line): (their table, their column pointing
    # at the due rows, the due rows' column it equals). A due row with no such row is open.
    done_elsewhere: tuple[str, str, str] | None = None
    things: str = ""               # "receipts", "purchase order lines"
    due_name: str = ""
    done_name: str = ""
    late: bool = False             # the figure is the share late, not on time


AGEING = ["Not yet due", "1-30 days", "31-60 days", "61-90 days", "Over 90 days"]


LIFECYCLE = {"active": "Active", "new": "New", "returning": "Returning", "won_back": "Won back",
             "lapsed": "Lapsed", "churn_rate": "Churn rate"}


def resolve_shaped(plan: Plan, model: SemanticModel, ctx: Context) -> Logical:
    if plan.pareto is not None:
        _alone(plan, "pareto")
        return _pareto(plan, model, ctx)
    if plan.funnel is not None:
        _alone(plan, "funnel")
        return _funnel(plan, model, ctx)
    if plan.together is not None:
        _alone(plan, "together")
        return _together(plan, model, ctx)
    if plan.gap is not None:
        _alone(plan, "gap")
        return _gap(plan, model, ctx)
    if plan.lifecycle is not None:
        _alone(plan, "lifecycle")
        return _lifecycle(plan, model, ctx)
    if plan.cohort is not None:
        _alone(plan, "cohort")
        return _cohort(plan, model, ctx)
    if plan.active is not None:
        _alone(plan, "active")
        return _active(plan, model, ctx)
    if plan.on_time is not None:
        _alone(plan, "on_time")
        return _on_time(plan, model, ctx)
    shapes = [n for n in ("per", "rolling", "within", "absent") if getattr(plan, n) is not None]
    if plan.bands is not None and plan.bands.per is not None:
        shapes.append("bands")
    if len(shapes) > 1:
        raise ResolveError("unsupported", f"one analytical shape at a time ({' and '.join(shapes)} were asked)")
    if plan.time.compare is not None and shapes:
        raise ResolveError("unsupported", "comparing two periods of this kind of figure is not worked out yet: "
                           "ask for each period on its own")
    if plan.absent is not None:
        return _absent(plan, model, ctx)
    if plan.per is not None or (plan.bands is not None and plan.bands.per is not None):
        return _per(plan, model, ctx)
    if plan.rolling is not None:
        return _rolling(plan, model, ctx)
    if plan.within is not None:
        return _within(plan, model, ctx)
    return resolve_plain(plan, model, ctx)          # ranges of each row's own value: a plain grouping


# ── per member, then over the members ──────────────────────────────────────


def _one_figure(plan: Plan) -> None:
    asked = len(plan.measures) + len(plan.derived) + sum(1 for d in plan.durations if d.measure)
    if asked == 0:
        raise ResolveError("unknown", "a figure per member needs a measure")
    if asked > 1:
        raise ResolveError("unsupported", "one figure per member at a time: ask for each measure on its own")


def _member_of(model: SemanticModel, of: str) -> tuple[str | None, str | None, str]:
    """(the group_by slug, or None; a count measure's distinct column, or None; the member in words)."""
    found = find_slug(model, of)
    if found is not None and found[0] == "entity":
        entity = found[1]
        assert isinstance(entity, Entity)
        return of, None, entity.business_name.lower()
    if found is not None and found[0] == "attribute":
        attribute = found[1]
        assert isinstance(attribute, Attribute)
        return of, None, _attribute_label(model, attribute).lower()
    if found is not None and found[0] == "measure":
        m = found[1]
        assert isinstance(m, Measure)
        column = None
        if isinstance(m.expr, AggExpr) and m.expr.agg == "count_distinct" and m.expr.column:
            column = m.expr.column
        elif isinstance(m.expr, AggExpr) and m.expr.agg == "count" and not m.expr.column and m.table in model.tables:
            table = model.tables[m.table]          # a count of rows: each row, told apart by its key
            key = table.primary_key or table.grain
            column = key[0] if len(key) == 1 else None
        if column is not None:
            word = m.business_name.lower()
            for lead in ("number of ", "count of ", "no. of "):
                if word.startswith(lead):
                    word = word[len(lead):]
            return None, column, _singular(word)
    # "Per booking": the thing a count measure counts ("Number of bookings"), named by itself -- the average
    # revenue per booking is each booking's revenue, then their average.
    counts = {}
    for m in model.measures.values():
        if isinstance(m.expr, AggExpr) and m.expr.agg in ("count", "count_distinct") and not m.expr.filters \
                and not m.filters and m.table in model.tables:
            word = m.business_name.lower()
            for lead in ("number of ", "count of ", "no. of "):
                word = word.removeprefix(lead)
            counts.setdefault(_singular(word), m.slug)
    wanted = _singular(of.replace("_", " ").replace(".", " ").strip().lower())
    if wanted in counts and counts[wanted] != of:
        return _member_of(model, counts[wanted])
    raise ResolveError("unknown", f"nothing to work out each figure for called {of}: name a customer, product or "
                       "other thing the data lists, or a count such as the number of orders",
                       [*sorted(model.entities)[:12], *sorted(counts.values())[:8]])


def _singular(word: str) -> str:
    from core2.bootstrap.names import singular

    return " ".join([*word.split()[:-1], singular(word.split()[-1])]) if word.split() else word


def _per(plan: Plan, model: SemanticModel, ctx: Context) -> Logical:
    per = plan.per
    band = plan.bands if plan.bands is not None and plan.bands.per is not None else None
    if band is not None and band.of not in {d.name for d in plan.durations} | {d.name for d in plan.derived}:
        found = find_slug(model, band.of)
        if found is not None and found[0] == "measure" and plan.measures != [band.of]:
            # "Patients with 1, 2 to 4 and 5 or more fills": each patient's fills are put in ranges, and each
            # range counts its patients, whatever the plan counted beside it (a real model wrote the count of
            # patients as the measure, and the patients per patient were put in ranges).
            plan = plan.model_copy(update={"measures": [band.of], "derived": [],
                                           "durations": [d for d in plan.durations if not d.measure]})
    of = per.of if per is not None else band.per if band is not None else None
    _one_figure(plan)
    slug, column, member = _member_of(model, of) if of else (None, None, "row")
    role = str(plan.via.get(of) or "").removeprefix("role:").strip() if of else ""
    if role and find_slug(model, role) is None:
        member = role.lower()             # each manager, not each employee: the link the grouping follows
    summary = per.summary if per is not None else "avg"
    if per is not None and per.everyone and summary in ("median", "percentile"):
        raise ResolveError("unsupported", "a median or percentile is worked out over the members with rows: ask "
                           "without everyone listed, or for the average over everyone")
    if summary == "percentile" and (per is None or per.p is None):
        raise ResolveError("unsupported", "a percentile needs which one (the 90th: p 90)")
    if summary in ("count", "share") and (per is None or per.op is None):
        raise ResolveError("unsupported", f"a {'share' if summary == 'share' else 'count'} of members needs the "
                           "condition each member's figure meets (\"3 or more orders\": op gte, values [3])")
    group_by = [g for g in plan.group_by if g != slug]
    inner_plan = plan.model_copy(update={
        "per": None, "bands": None, "group_by": [*group_by, *([slug] if slug else [])], "sort": [], "limit": None,
        "intent": "breakdown", "chart": None})
    # Each member apart, a member with no name too (by the key its rows hold): items whose names are missing
    # were one "Unknown" member, their costs averaged together, and none counted as costing more than $1,000.
    inner = resolve_plain(inner_plan, model, replace(ctx, unnamed_by_key=True) if slug in model.entities else ctx)
    if len(inner.measures) != 1:
        raise ResolveError("unsupported", "one figure per member at a time: ask for each measure on its own")
    figure = inner.measures[0]
    members: list[str] = []
    if slug is not None:
        attribute = _entity_label(model, slug) if slug in model.entities else model.attributes[slug]
        members = [g.name for g in inner.groups if g.attribute == attribute.slug]
    elif column is not None:
        owner = model.columns[column].table
        part = inner.parts[0] if len(inner.parts) == 1 else None
        if part is not None and part.table != owner:
            # Counted on a table the rows point at (each prescription, from its fills): the rows' own key to it.
            link = next((j for j in model.joins.values() if j.from_table == part.table and j.to_table == owner
                         and list(j.to_columns) == [column] and P.usable(j)), None)
            column = link.from_columns[0] if link is not None else None
        if part is None or column is None:
            raise ResolveError("unsupported", f"each {member}'s {figure.label.lower()} cannot be worked out: "
                               f"{figure.label} is not on the {plural(member)}' own rows")
        name = _output_name(f"{member.replace(' ', '_')}_key", {g.name for g in inner.groups})
        part.groups.append(PartGroup(name, part.alias, column, "attribute"))
        inner.groups.append(Group(name, member.capitalize(), "attribute"))
        members = [name]
    else:
        # Each row's own value (days to deliver each fill): the rows, not added up.
        if len(inner.parts) != 1 or not _row_value(figure):
            raise ResolveError("unsupported", f"{figure.label} is a total, not a figure each row has: name whose "
                               "figure it is (each customer, each order)")
        inner.parts[0].row_level = True
        member = model.tables[inner.parts[0].table].business_name.lower() or "row"
    outer_groups = [g for g in inner.groups if g.name not in members]
    listed: Part | None = None
    everyone = per is not None and per.everyone
    if everyone:
        if slug is None or slug not in model.entities and not model.attributes[slug].entity:
            raise ResolveError("unsupported", "everyone listed can be counted only for things the data lists "
                               "(customers, products): name one")
        if any(g.kind == "attribute" for g in outer_groups):
            raise ResolveError("unsupported", "everyone listed is counted over the whole list, not within each "
                               "group: ask without the grouping, or for those with rows only")
        listed, _ = _listed(plan, model, ctx, slug if slug in model.entities else model.attributes[slug].entity,
                            _Aliases())
    word = figure.label
    shape = PerShape(inner=inner, members=members, value=figure.name, summary=summary,
                     p=per.p if per is not None else None, op=per.op if per is not None else None,
                     values=list(per.values) if per is not None else [], member=member, figure=word, listed=listed,
                     row_level=not members)
    taken = {g.name for g in outer_groups}
    if band is not None:
        edges = sorted(set(band.edges))
        whole = figure.format in ("number", "count", "days", "integer") and _whole(figure)
        labels = B.labels(edges, whole=whole, money=figure.format == "currency",
                          unit="days" if figure.format == "days" else "")
        name = _output_name("band", taken)
        shape.band = PartGroup(name, "m", None, "band", edges=edges, labels=labels,
                               style={"whole": whole, "money": figure.format == "currency",
                                      "unit": "days" if figure.format == "days" else ""})
        outer_groups = [*outer_groups, Group(name, f"{word} per {member}", "attribute", bands=labels)]
        out = [OutMeasure(name=_output_name(f"{plural(member).replace(' ', '_')}", taken | {name}),
                          label=plural(member).capitalize(), expr=AggExpr(agg="count"), format="number",
                          measure=None)]
        if figure.format not in ("percent", "percentage") and _adds(figure):
            out.append(OutMeasure(name=_output_name("total", taken | {name, out[0].name}), label=word,
                                  expr=AggExpr(agg="sum"), format=figure.format, measure=figure.measure))
        sort: list[tuple[str, bool]] = []
    else:
        label, fmt = per_label(shape)
        if fmt == "same":
            # An average of counts has decimals (2.4 orders per customer); a lowest or highest is one of them.
            fmt = figure.format if figure.format not in ("count", "integer") or summary in ("min", "max") \
                else "number"
        agg = "count" if summary == "count" else "avg"
        out = [OutMeasure(name=_output_name(f"{summary}_{figure.name}", taken), label=label,
                          expr=AggExpr(agg=agg), format=fmt, measure=None)]
        sort = []
        for s in plan.sort:
            if s.by in (*plan.measures, figure.name, figure.label, *(d.name for d in plan.durations)):
                sort.append((out[0].name, s.desc))
            elif s.by in {g.attribute for g in outer_groups} | {g.name for g in outer_groups}:
                sort.append((next(g.name for g in outer_groups if s.by in (g.attribute, g.name)), s.desc))
        if not sort and any(g.kind == "attribute" for g in outer_groups):
            sort = [(out[0].name, True)]
        if not sort and any(g.kind == "period" for g in outer_groups):
            sort = [(next(g.name for g in outer_groups if g.kind == "period"), False)]
    notes = [n for n in inner.notes]
    return dataclasses.replace(
        inner, intent=plan.intent or ("breakdown" if outer_groups else "value"), groups=outer_groups, measures=out,
        sort=sort, sort_asked=bool(plan.sort), limit=plan.limit, share=False, having=[], notes=notes,
        conditions=[*inner.conditions, *([Condition("everyone", member)] if everyone else [])],
        unit_group=None, shape=shape)


def _row_value(o: OutMeasure) -> bool:
    """Has each row a value of ``o`` of its own (days between two dates, an amount on the row)?"""
    if isinstance(o.expr, DaysBetween):
        return True
    return isinstance(o.expr, AggExpr) and o.expr.column is not None and o.expr.agg in ("sum", "avg", "min", "max")


def _whole(o: OutMeasure) -> bool:
    return isinstance(o.expr, DaysBetween) or (isinstance(o.expr, AggExpr) and o.expr.agg in ("count",
                                                                                                "count_distinct"))


def _adds(o: OutMeasure) -> bool:
    return (isinstance(o.expr, AggExpr) and o.expr.agg in ("sum", "count", "count_distinct")) or (
        isinstance(o.expr, DaysBetween) and o.expr.agg == "sum") or one_total(o)


def per_label(shape: PerShape) -> tuple[str, str]:
    """The summary's name and format: "Average net amount per customer", "Customers with 3 or more orders"."""
    figure, member = shape.figure, shape.member
    each = f" per {member}" if not shape.row_level else ""
    if shape.summary == "percentile":
        return f"{_ordinal(shape.p or 50)} percentile of {figure.lower()}{each}", "same"
    if shape.summary in SUMMARY_WORDS:
        return f"{SUMMARY_WORDS[shape.summary]} {figure.lower()}{each}", "same"
    condition = condition_words(shape.op or "", shape.values, figure)
    things = plural(member).capitalize()
    if shape.summary == "count":
        return f"{things} with {condition}", "number"
    return f"Share of {plural(member)} with {condition}", "percent"


def condition_words(op: str, values: list[float], figure: str) -> str:
    def n(v: float) -> str:
        return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"
    what = figure.lower()
    if what.startswith("number of "):
        what = what[len("number of "):]
    if op == "between" and len(values) == 2:
        return f"{n(values[0])} to {n(values[1])} {what}"
    words = {"gte": "or more", "gt": "more than", "lte": "or fewer", "lt": "fewer than", "eq": "exactly",
             "ne": "other than"}
    v = n(values[0]) if values else ""
    if op in ("gte", "lte"):
        return f"{v} {words[op]} {what}"
    return f"{words.get(op, op)} {v} {what}".strip()


def _ordinal(p: float) -> str:
    n = int(p) if float(p).is_integer() else p
    if isinstance(n, int):
        suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"
    return f"{n}th"


def _listed(plan: Plan, model: SemanticModel, ctx: Context, entity_slug: str | None,
            aliases: _Aliases) -> tuple[Part, list]:
    """Every listed member: the entity's own table, its default filters and the plan's filters on the members (a
    column of theirs, or of a table they lead to: a customer's segment, their region). The plan's other filters
    are returned: conditions on the activity rows ("in the West store", "product A"), not on who is listed."""
    if entity_slug is None or entity_slug not in model.entities:
        raise ResolveError("unsupported", "everyone listed can be counted only for things the data lists")
    entity = model.entities[entity_slug]
    builder = _PartBuilder(model, entity.table, aliases, ctx)
    builder._allowed(entity.table)
    part = builder.part
    for df in model.tables[entity.table].default_filters:
        part.preds.append(Pred(part.alias, df.column, df.op, list(df.values)))
    rest = []
    for f in plan.filters:
        found = find_slug(model, f.field)
        if not found or found[0] not in ("attribute", "entity"):
            rest.append(f)
            continue
        attribute = _entity_label(model, f.field) if found[0] == "entity" else found[1]
        assert isinstance(attribute, Attribute)
        column = model.columns[attribute.column]
        if column.table != entity.table and not _reaches(model, entity.table, column.table):
            rest.append(f)
            continue
        _shown(model, attribute, f.field, entity=found[0] == "entity", personal=ctx.personal_shown)
        alias = builder.reach(column.table, label=attribute.business_name, filtering=True) \
            if column.table != entity.table else part.alias
        f, stored = _named_filter(column, f)
        part.preds.append(Pred(alias, column.key, f.op, list(stored)))
    return part, rest


# ── a rank inside each member of a grouping ────────────────────────────────


def _within(plan: Plan, model: SemanticModel, ctx: Context) -> Logical:
    w = plan.within
    assert w is not None
    group_by = list(plan.group_by) if w.group in plan.group_by else [w.group, *plan.group_by]
    if len(group_by) < 2:
        raise ResolveError("unsupported", "a rank within each group needs what is ranked too: \"stores within each "
                           "region\" groups by store and region")
    if not plan.measures and not plan.derived and not plan.durations:
        raise ResolveError("unknown", "a rank needs a measure to rank by")
    inner = resolve_plain(plan.model_copy(update={"within": None, "group_by": group_by, "limit": None}), model, ctx)
    attribute = _entity_label(model, w.group) if w.group in model.entities else model.attributes.get(w.group)
    if attribute is None:
        raise ResolveError("unknown", f"nothing to rank within called {w.group}", list(plan.group_by))
    partition = [g.name for g in inner.groups if g.attribute == attribute.slug]
    ranked = [g for g in inner.groups if g.kind == "attribute" and g.name not in partition]
    if not partition or not ranked:
        raise ResolveError("unsupported", "a rank within each group needs two groupings")
    by = inner.measures[0].name
    desc = True
    for s in plan.sort:
        for o in inner.measures:
            if s.by in (o.name, o.label, getattr(o.measure, "slug", None)):
                by, desc = o.name, s.desc
    top = w.top or plan.limit
    shape = WithinShape(partition=partition, by=by, desc=desc, member=ranked[0].name, top=top,
                        rank=_output_name("rank", {g.name for g in inner.groups} | {o.name for o in inner.measures}))
    word = _attribute_label(model, attribute).lower()
    note = (f"Ranked within each {word}" + (f": the first {top} of each" if top else "") + ".")
    return dataclasses.replace(inner, limit=None, shape=shape,
                               sort=[*((n, False) for n in partition), (shape.rank, False)], notes=[*inner.notes, note])


# ── a series accumulated or smoothed ───────────────────────────────────────


def _rolling(plan: Plan, model: SemanticModel, ctx: Context) -> Logical:
    r = plan.rolling
    assert r is not None
    grain = plan.time.grain or "month"
    step = {"fiscal_month": "month", "fiscal_quarter": "quarter", "fiscal_year": "year"}.get(grain, grain)
    fiscal = model.settings.fiscal_year_start_month
    base = plan.model_copy(update={"rolling": None, "limit": None, "time": plan.time.model_copy(update={"grain": grain})})
    plain = resolve_plain(base, model, ctx)         # the periods shown, as the plain series reads them
    shown = plain.window
    role = next((p.date.role for p in plain.parts if p.date is not None), None)
    first, last = _first_last(role) if role is not None else (None, None)
    first_period = unit_start(first, step, fiscal_start=fiscal if grain.startswith("fiscal") else None) \
        if first is not None else None
    start = shown.start
    if r.kind == "running":
        if start is not None and r.reset == "year":
            start = dt.date(start.year, 1, 1)
        elif start is not None and r.reset == "fiscal_year":
            begins = fiscal or 1
            start = dt.date(start.year if start.month >= begins else start.year - 1, begins, 1)
        # With no reset, a running total starts with the first period shown, as a BI tool's does within its
        # filter: "running total in fiscal year 2026" starts at April 2026's own figure, never at everything
        # before it. Asked over all time, it starts where the data begins.
    elif start is not None:
        start = add_units(start, step, -(r.periods - 1))
    if start is not None and first_period is not None and start < first_period:
        start = first_period        # nothing to read before the data begins
    end = shown.end
    if start is not None and end is not None:
        window = {"kind": "between", "start": start, "end": end - dt.timedelta(days=1)}
    elif start is not None:
        window = {"kind": "since", "start": start}
    elif end is not None:
        window = {"kind": "until", "end": end - dt.timedelta(days=1)}
    else:
        window = {"kind": "all"}
    read_plan = base.model_copy(update={"time": base.time.model_copy(update={
        "window": base.time.window.model_validate(window)})})
    inner = resolve_plain(read_plan, model, ctx)
    columns: dict[str, str] = {}
    labels: dict[str, str] = {}
    adds: dict[str, bool] = {}
    taken = {g.name for g in inner.groups} | {o.name for o in inner.measures}
    for o in inner.measures:
        if o.hidden:
            continue
        adds[o.name] = _adds(o) and not o.semi
        if r.kind in ("running", "moving_sum") and not adds[o.name]:
            raise ResolveError("unsupported", f"{o.label} does not add up over time (an average, a ratio or a "
                               f"level), so a {'running total' if r.kind == 'running' else 'total over periods'} "
                               "of it means nothing: ask for its moving average instead")
        columns[o.name] = _output_name(f"{o.name}_{r.kind}", taken)
        taken.add(columns[o.name])
        labels[o.name] = rolling_label(r.kind, r.periods, r.reset, grain, o.label)
    read = Range(start if start is not None else first_period, end if end is not None else
                 (last + dt.timedelta(days=1) if last is not None else None))
    expected = periods(read, grain, fiscal_start=fiscal) if read.start and read.end else []
    shape = RollingShape(kind=r.kind, periods=r.periods, reset=r.reset, grain=grain, shown=shown, read=inner.window,
                         columns=columns, labels=labels, adds=adds, expected=expected)

    def inside(d: dt.date) -> bool:
        return (shown.start is None or d >= shown.start) and (shown.end is None or d < shown.end)

    return dataclasses.replace(inner, intent="trend", window=shown, limit=None, shape=shape,
                               expected=[d for d in inner.expected if inside(d)],
                               partial=[d for d in inner.partial if inside(d)],
                               notes=[*inner.notes, rolling_note(shape)])


def rolling_label(kind: str, periods: int, reset: str | None, grain: str, label: str) -> str:
    unit = {"fiscal_month": "month", "fiscal_quarter": "quarter", "fiscal_year": "year"}.get(grain, grain)
    if kind == "running":
        if reset == "year":
            return f"{label}, year to date"
        if reset == "fiscal_year":
            return f"{label}, fiscal year to date"
        return f"Running total of {label.lower()}"
    if kind == "moving_average":
        return f"{periods}-{unit} moving average of {label.lower()}"
    return f"{label}, last {periods} {unit}s"


def rolling_note(shape: RollingShape) -> str:
    unit = {"fiscal_month": "month", "fiscal_quarter": "quarter", "fiscal_year": "year"}.get(shape.grain, shape.grain)
    if shape.kind == "running":
        since = {"year": "since the start of each year", "fiscal_year": "since the start of each fiscal year"}.get(
            shape.reset or "", f"from the first {unit} shown" if shape.shown.start is not None else "since the data begins")
        return f"Each {unit} adds to the total {since}."
    each = "the average of" if shape.kind == "moving_average" else "the total of"
    return (f"Each point is {each} that {unit} and the {shape.periods - 1} before it; the first points read "
            f"{unit}s before the period shown, and a {unit} with no rows counts as 0.")


# ── members with no activity ───────────────────────────────────────────────


def _absent(plan: Plan, model: SemanticModel, ctx: Context) -> Logical:
    a = plan.absent
    assert a is not None
    found = find_slug(model, a.of)
    if not found or found[0] != "entity":
        raise ResolveError("unknown", f"nothing listed called {a.of}: name the customers, products or other things "
                           "the data lists", sorted(model.entities)[:12])
    entity = found[1]
    assert isinstance(entity, Entity)
    if len(entity.key_columns) != 1:
        raise ResolveError("unsupported", f"{plural(entity.business_name)} are told apart by more than one column")
    key = entity.key_columns[0]
    found_m = find_slug(model, a.measure)
    if not found_m or found_m[0] != "measure":
        raise ResolveError("unknown", f"no measure called {a.measure}", [])
    measure = found_m[1]
    assert isinstance(measure, Measure)
    if not measure.table:
        raise ResolveError("unsupported", f"{measure.business_name} is worked out over several tables: name the "
                           "activity of one (orders, fills)")
    measure = activity_of(model, measure, entity)
    aliases = _Aliases()

    # The listed members, with the plan's conditions on them; its other conditions are on the activity.
    listed, rest = _listed(plan, model, ctx, a.of, aliases)
    label_attr = _shown(model, _entity_label(model, a.of), a.of, entity=True, personal=ctx.personal_shown)
    out_names: set[str] = set()
    name = _output_name(label_attr.slug.replace(".", "_"), out_names)
    listed.groups.append(PartGroup(name, listed.alias, label_attr.column, "attribute"))
    groups = [Group(name, _attribute_label(model, label_attr), "attribute", attribute=label_attr.slug)]
    code = entity.code_column if entity.code_column and entity.code_column != label_attr.column else None
    if code:
        code_name = _output_name(f"{name}_code", out_names)
        listed.groups.append(PartGroup(code_name, listed.alias, code, "attribute"))
        groups.append(Group(code_name, f"{entity.business_name} code", "member_code", attribute=label_attr.slug))
    key_name = _output_name("member_key", out_names)
    listed.groups.append(PartGroup(key_name, listed.alias, key, "attribute"))   # each member once, by its key

    role = _date_role_for(model, plan, measure, measure.table)
    first, last = _first_last(role) if role is not None else (None, None)
    window = resolve_window(plan.time.window, today=ctx.today, first_data=first, last_data=last,
                            fiscal_start=model.settings.fiscal_year_start_month)
    if plan.time.window.kind == "all":
        window = Range(None, None)

    def period(filters: list) -> tuple[list, Range | None, DateRole | None]:
        """"Bought in Q1 but not in Q2": a filter on one of the activity's dates is its period, not a field. The
        filters left, the period (None: none named) and the date it is read by."""
        kept, rng, on = [], None, None
        for f in filters:
            found_d = find_slug(model, f.field)
            dated = found_d[1] if found_d and found_d[0] == "date" else None
            if not isinstance(dated, DateRole) or f.op not in ("between", "eq", "gte", "gt", "lte", "lt") \
                    or not f.values or (dated.table != measure.table and dated.table not in _documents(model, measure.table)):
                kept.append(f)
                continue
            try:
                days = [dt.date.fromisoformat(str(v)[:10]) for v in f.values]
            except ValueError:
                raise ResolveError("unsupported", f"{dated.name} is a date: give it as a date (2026-01-31)") from None
            one = dt.timedelta(days=1)
            start = days[0] + one if f.op == "gt" else days[0] if f.op in ("between", "eq", "gte") else None
            end = (days[-1] + one if f.op in ("between", "eq", "lte") else days[0] if f.op == "lt" else None)
            rng, on = Range(start, end), dated
        return kept, rng, on

    def activity(filters: list, rng: Range | None, on: DateRole | None = None) -> tuple[Part, str]:
        """The keys of the members with rows like these (in ``rng``, by the date ``on``): each once."""
        builder = _PartBuilder(model, measure.table, aliases, ctx)
        builder._allowed(measure.table)
        part = builder.part
        part.distinct_only = True
        reached = builder.reach(entity.table, label=entity.business_name)
        out = _output_name("member_key", set())
        part.groups.append(PartGroup(out, reached, key, "attribute"))
        part.preds.append(Pred(reached, key, "not_null", []))
        for df in model.tables[measure.table].default_filters:
            part.preds.append(Pred(part.alias, df.column, df.op, list(df.values)))
        for mf in [*measure.filters, *(getattr(measure.expr, "filters", None) or [])]:
            part.preds.append(Pred(part.alias, mf.column, mf.op, list(mf.values)))
        for f in filters:
            found_f = find_slug(model, f.field)
            if not found_f or found_f[0] not in ("attribute", "entity"):
                raise ResolveError("unknown", f"nothing to narrow the activity by called {f.field}", [])
            attribute = _entity_label(model, f.field) if found_f[0] == "entity" else found_f[1]
            assert isinstance(attribute, Attribute)
            _shown(model, attribute, f.field, entity=found_f[0] == "entity", personal=ctx.personal_shown)
            column = model.columns[attribute.column]
            f, stored = _named_filter(column, f)
            alias = builder.reach(column.table, label=attribute.business_name, filtering=True) \
                if column.table != measure.table else part.alias
            part.preds.append(Pred(alias, column.key, f.op, list(stored)))
        if rng is not None and (on or role) is not None:
            # A date only this part reads: never the answer's own window (part.date), which is the absence's.
            part.date_ranges.append((builder.date(on or role), rng))
        return part, out

    without, absent_in, absent_on = period(a.without)
    if absent_in is not None and plan.time.window.kind == "all":
        window, role = absent_in, absent_on        # "not in Q2", the period written as a date filter
    act, act_key = activity([*rest, *without], window)
    shape = AbsentShape(entity=listed, key=key, activity=act, activity_key=act_key, member=entity.business_name.lower(),
                        activity_label=measure.business_name, member_key=key_name)
    if a.had:
        had, had_in, had_on = period(a.had)
        shape.had, shape.had_key = activity([*rest, *had], had_in, had_on)
        shape.had_in = had_in
    if a.active_before:
        if window.start is None:
            raise ResolveError("unsupported", "active before needs a window to be absent in")
        shape.before, shape.before_key = activity(list(rest), Range(None, window.start))
    notes = [absent_note(shape, window, a.active_before)]
    return Logical(intent="list", parts=[listed], groups=groups, measures=[], window=window,
                   sort=[(name, False)], limit=plan.limit, max_rows=ctx.max_rows, notes=notes,
                   fiscal_start=model.settings.fiscal_year_start_month,
                   conditions=[Condition("absent", measure.business_name.lower(), dates=window)], shape=shape)


def activity_of(model: SemanticModel, measure: Measure, entity: Entity) -> Measure:
    """A member's activity is what its rows record, never the member list itself: "new customers each month"
    counted with the number of customers was dated by a date on the customer (its credit limit's last change) and
    found none. The rows of the one fact table pointing at the member (or, of several, the one the business is
    measured by) are its activity, counted."""
    if measure.table != entity.table and model.tables[measure.table].kind != "dimension":
        return measure
    facts = list(dict.fromkeys(
        j.from_table for j in model.joins.values()
        if j.to_table == entity.table and j.trust != "rejected" and not j.to_calendar
        and model.tables[j.from_table].kind == "fact" and model.tables[j.from_table].default_date))
    if len(facts) > 1:
        from core2.plan.analyses import leading

        lead = leading(model)
        facts = [lead.table] if lead is not None and lead.table in facts else facts
    if len(facts) != 1:
        return measure
    counts = [m for m in model.measures.values()
              if m.table == facts[0] and not m.hidden and getattr(m.expr, "agg", "") == "count" and not m.filters
              and not getattr(m.expr, "filters", None)]
    return counts[0] if counts else measure


def absent_note(shape: AbsentShape, window: Range, active_before: bool) -> str:
    from core2.answer.builder import span_words

    things = plural(shape.member)
    activity = shape.activity_label.lower()
    for lead in ("number of ", "count of ", "no. of "):
        activity = activity[len(lead):] if activity.startswith(lead) else activity
    when = span_words(window) if (window.start or window.end) else "ever"
    who = (f"the {things} with {activity} before then" if active_before else
           f"the {things} with {activity} {span_words(shape.had_in)}" if shape.had_in is not None else
           f"every {shape.member} listed")
    return (f"Listed: {who}, minus those with {activity} {when}. The list follows what you may see and the rows "
            "each table leaves out by default.")


# ── each member's days with activity (time between events, lifecycle, cohorts) ─


_CALENDAR_GRAINS = ("day", "week", "month", "quarter", "year")


def _alone(plan: Plan, name: str) -> None:
    others = [n for n in ("per", "rolling", "within", "absent", "gap", "lifecycle", "cohort", "active", "on_time")
              if n != name and getattr(plan, n) is not None]
    if plan.bands is not None:
        others.append("bands")
    if others:
        raise ResolveError("unsupported", f"one analytical shape at a time ({' and '.join([name, *others])} were "
                           "asked)")
    if plan.time.compare is not None:
        raise ResolveError("unsupported", "comparing two periods of this kind of figure is not worked out yet: ask "
                           "for each period on its own")


def _events(plan: Plan, model: SemanticModel, ctx: Context, of: str, measure_slug: str, *,
            read: Callable[[Range], Range], grain: str | None) -> tuple[Events, list[Group]]:
    """Each member's days with activity (the measure's rows), each day once, with the member's own groupings.

    ``read`` turns the answer's window into the days read: a gap's earlier event may be before the window, a
    first order long before it."""
    found = find_slug(model, of)
    if not found or found[0] != "entity":
        raise ResolveError("unknown", f"nothing listed called {of}: name the customers, patients or other things "
                           "the data lists whose activity it is", sorted(model.entities)[:12])
    entity = found[1]
    assert isinstance(entity, Entity)
    if len(entity.key_columns) != 1:
        raise ResolveError("unsupported", f"{plural(entity.business_name)} are told apart by more than one column")
    key = entity.key_columns[0]
    found_m = find_slug(model, measure_slug)
    if not found_m or found_m[0] != "measure":
        raise ResolveError("unknown", f"no measure called {measure_slug}: name the activity (orders, fills)", [])
    measure = found_m[1]
    assert isinstance(measure, Measure)
    if not measure.table:
        raise ResolveError("unsupported", f"{measure.business_name} is worked out over several tables: name the "
                           "activity of one (orders, fills)")
    measure = activity_of(model, measure, entity)
    if grain is not None and grain not in _CALENDAR_GRAINS:
        raise ResolveError("unsupported", f"this figure is worked out by calendar periods: ask by month, quarter or "
                           f"year rather than by {grain.replace('_', ' ')}")
    role = _date_role_for(model, plan, measure, measure.table)
    if role is None:
        raise ResolveError("unsupported", f"{measure.business_name} has no date to put its rows in order by")
    first, last = _first_last(role)
    window = resolve_window(plan.time.window, today=ctx.today, first_data=first, last_data=last,
                            fiscal_start=model.settings.fiscal_year_start_month)
    if plan.time.window.kind == "all":
        window = Range(None, None)

    aliases = _Aliases()
    builder = _PartBuilder(model, measure.table, aliases, ctx)
    builder._allowed(measure.table)
    part = builder.part
    part.distinct_only = True
    reached = builder.reach(entity.table, label=entity.business_name)
    taken: set[str] = set()
    key_out = _output_name("member_key", taken)
    taken.add(key_out)
    part.groups.append(PartGroup(key_out, reached, key, "attribute"))
    part.preds.append(Pred(reached, key, "not_null", []))
    use = builder.date(role)
    day_out = _output_name("event_day", taken)
    taken.add(day_out)
    part.groups.append(PartGroup(day_out, part.alias, None, "period", grain="day", date=use))
    # The days read, never the answer's own window (part.date): a gap's earlier event may be before it.
    part.date_ranges.append((use, read(window)))
    for df in model.tables[measure.table].default_filters:
        part.preds.append(Pred(part.alias, df.column, df.op, list(df.values)))
    for mf in [*measure.filters, *(getattr(measure.expr, "filters", None) or [])]:
        part.preds.append(Pred(part.alias, mf.column, mf.op, list(mf.values)))
    for f in plan.filters:
        found_f = find_slug(model, f.field)
        if not found_f or found_f[0] not in ("attribute", "entity"):
            raise ResolveError("unknown", f"nothing to narrow the activity by called {f.field}", [])
        attribute = _entity_label(model, f.field) if found_f[0] == "entity" else found_f[1]
        assert isinstance(attribute, Attribute)
        _shown(model, attribute, f.field, entity=found_f[0] == "entity", personal=ctx.personal_shown)
        column = model.columns[attribute.column]
        f, stored = _named_filter(column, f)
        alias = builder.reach(column.table, label=attribute.business_name, filtering=True) \
            if column.table != measure.table else part.alias
        part.preds.append(Pred(alias, column.key, f.op, list(stored)))

    # Groupings: the member's own (a customer's segment): each member's days stay one sequence. A grouping of the
    # rows (a store) would cut one customer's orders into several sequences.
    out_groups: list[Group] = []
    names: list[str] = []
    for slug in plan.group_by:
        if slug.startswith("time:"):
            raise ResolveError("unsupported", "by the period of each event only (by month, quarter or year)")
        found_g = find_slug(model, slug)
        if not found_g or found_g[0] not in ("attribute", "entity"):
            raise ResolveError("unknown", f"nothing to group by called {slug}", [])
        attribute = _entity_label(model, slug) if found_g[0] == "entity" else found_g[1]
        assert isinstance(attribute, Attribute)
        column = model.columns[attribute.column]
        if column.table != entity.table and not _reaches(model, entity.table, column.table):
            raise ResolveError("unsupported", f"{_attribute_label(model, attribute)} is not one of each "
                               f"{entity.business_name.lower()}'s own: their activity is one sequence, not split by "
                               "what each row is about")
        _shown(model, attribute, slug, entity=found_g[0] == "entity", personal=ctx.personal_shown)
        alias = builder.reach(column.table, label=attribute.business_name) if column.table != measure.table \
            else part.alias
        name = _output_name(attribute.slug.replace(".", "_"), taken)
        taken.add(name)
        part.groups.append(PartGroup(name, alias, column.key, "attribute"))
        out_groups.append(Group(name, _attribute_label(model, attribute), "attribute", attribute=attribute.slug))
        names.append(name)
    activity = measure.business_name.lower()
    for lead in ("number of ", "count of ", "no. of "):
        activity = activity[len(lead):] if activity.startswith(lead) else activity
    events = Events(part=part, key=key_out, day=day_out, groups=names, member=entity.business_name.lower(),
                    activity=activity, grain=grain, window=window)
    if grain is not None:
        out_groups.insert(0, Group(_output_name("period", taken), grain.capitalize(), "period", grain=grain))
    return events, out_groups


def _gap(plan: Plan, model: SemanticModel, ctx: Context) -> Logical:
    g = plan.gap
    assert g is not None
    if g.nth is not None and g.within_days is not None:
        raise ResolveError("unsupported", "days to the nth event, or the share followed within n days: one at a time")
    if g.nth is not None:
        read = lambda w: Range(None, None)          # noqa: E731 - the first event may be long before the window
    elif g.within_days is not None:
        days = g.within_days
        read = lambda w: Range(None, w.end + dt.timedelta(days=days) if w.end else None)   # noqa: E731
    else:
        read = lambda w: Range(None, w.end)         # noqa: E731 - the earlier event of a gap may be before it
    events, groups = _events(plan, model, ctx, g.of, g.measure, read=read, grain=plan.time.grain)
    summary = "nth" if g.nth is not None else "within" if g.within_days is not None else g.summary
    shape = GapShape(events=events, summary=summary, nth=g.nth, within_days=g.within_days)
    label, fmt = gap_label(shape)
    taken = {gr.name for gr in groups}
    out = [OutMeasure(name=_output_name("days" if fmt == "days" else "share", taken), label=label,
                      expr=AggExpr(agg="avg"), format=fmt, measure=None)]
    period = next((gr.name for gr in groups if gr.kind == "period"), None)
    sort = [(period, False)] if period else [(out[0].name, False)] if groups else []
    return Logical(intent="trend" if period else "breakdown" if groups else "value", parts=[events.part],
                   groups=groups, measures=out, window=events.window, sort=sort, limit=plan.limit,
                   max_rows=ctx.max_rows, notes=[], fiscal_start=model.settings.fiscal_year_start_month,
                   conditions=[], shape=shape)


def gap_label(shape: GapShape) -> tuple[str, str]:
    """The figure's name and format: "Average days between orders", "Days to the 2nd order", "Orders followed by
    another within 30 days"."""
    e = shape.events
    single = _singular(e.activity)
    if shape.summary == "nth":
        return f"Days from the first {single} to the {_ordinal(shape.nth or 2)}", "days"
    if shape.summary == "within":
        return f"{e.activity[:1].upper()}{e.activity[1:]} followed by another within {shape.within_days} days", \
            "percent"
    word = SUMMARY_WORDS.get(shape.summary, "Average")
    return f"{word} days between {e.activity}", "days"


def gap_definition(shape: GapShape, span: str, counted: float | None, reached: float | None) -> str:
    e = shape.events
    things = plural(e.member)
    when = f" {span}" if span else ""
    single = _singular(e.activity)
    if shape.summary == "nth":
        made = f"; {int(reached):,} of the {int(counted):,} {things} have one" if counted is not None and \
            reached is not None else ""
        return (f"For each {e.member} whose first {single} was{when or ' at any time'}, the days from it to their "
                f"{_ordinal(shape.nth or 2)} {single}, then the average over those who made one{made}. Each day "
                f"with {e.activity} counts once.")
    if shape.summary == "within":
        n = f" (of {int(counted):,})" if counted is not None else ""
        return (f"Each {single} of each {e.member}{when}{n}, followed by that {e.member}'s next {single} within "
                f"{shape.within_days} days or not: the share that was. Each day with {e.activity} counts once.")
    how = {"avg": "their average", "median": "the middle value", "min": "the shortest", "max": "the longest"}[
        shape.summary]
    n = f" (the {int(counted):,} gaps)" if counted is not None else ""
    return (f"For each {e.member}, the days from each {single}{when} back to their {single} before it (which may "
            f"be earlier), then {how}{n}. Each day with {e.activity} counts once; a first {single} has no gap.")


# ── new, returning, won back, lapsed ───────────────────────────────────────


def _period_grain(plan: Plan) -> str:
    """The periods a member's activity is counted in: the plan's own, the window's unit, or a month."""
    grain = plan.time.grain or (plan.time.window.unit if plan.time.window.unit in ("month", "quarter", "year")
                                else None) or "month"
    if grain not in ("month", "quarter", "year"):
        raise ResolveError("unsupported", f"members are counted by month, quarter or year, not by "
                           f"{grain.replace('_', ' ')}")
    return grain


def _lifecycle(plan: Plan, model: SemanticModel, ctx: Context) -> Logical:
    lc = plan.lifecycle
    assert lc is not None
    grain = _period_grain(plan)
    # Every period of activity up to the window's end: who is new, who was away and came back.
    events, groups = _events(plan, model, ctx, lc.of, lc.measure, read=lambda w: Range(None, w.end), grain=grain)
    taken = {g.name for g in groups}
    things = plural(events.member)
    shown = ["active", "new", "returning", "won_back", "lapsed"] if lc.status == "all" else [lc.status]
    columns: dict[str, str] = {}
    out: list[OutMeasure] = []
    for status in shown:
        name = _output_name(status, taken)
        taken.add(name)
        columns[status] = name
        label = "Churn rate" if status == "churn_rate" else f"{LIFECYCLE[status]} {things}"
        out.append(OutMeasure(name=name, label=label, expr=AggExpr(agg="sum"),
                              format="percent" if status == "churn_rate" else "number", measure=None))
    shape = LifecycleShape(events=events, status=lc.status, columns=columns)
    period = next(g.name for g in groups if g.kind == "period")
    return Logical(intent="trend", parts=[events.part], groups=groups, measures=out, window=events.window,
                   sort=[(period, False)], limit=plan.limit, max_rows=ctx.max_rows, notes=[],
                   fiscal_start=model.settings.fiscal_year_start_month, conditions=[], shape=shape)


def lifecycle_definition(shape: LifecycleShape) -> str:
    e = shape.events
    period = e.grain or "month"
    things, single = plural(e.member), _singular(e.activity)
    parts = {
        "active": f"active: with {e.activity} in the {period}",
        "new": f"new: their first {single} ever was in the {period}",
        "returning": f"returning: they had {e.activity} the {period} before too",
        "won_back": f"won back: they had {e.activity} before, but none the {period} before",
        "lapsed": f"lapsed: {e.activity} the {period} before, none in this one",
        "churn_rate": f"churn rate: the lapsed as a share of the {things} with {e.activity} the {period} before",
    }
    asked = list(parts) if shape.status == "all" else [shape.status]
    if shape.status == "churn_rate":
        asked = ["lapsed", "churn_rate"]
    said = "; ".join(parts[k] for k in asked)
    return (f"{things[:1].upper()}{things[1:]} by {period}, each counted once a {period}: {said}. The latest "
            f"{period} may not be over yet.")


# ── cohorts ────────────────────────────────────────────────────────────────


def _cohort(plan: Plan, model: SemanticModel, ctx: Context) -> Logical:
    co = plan.cohort
    assert co is not None
    if plan.group_by:
        raise ResolveError("unsupported", "a cohort grid is by first period and periods after it: ask for one "
                           "group of members at a time with a filter instead of a grouping")
    grain = _period_grain(plan)
    # Every period of activity: a member's first one makes their cohort, however long ago.
    events, groups = _events(plan, model, ctx, co.of, co.measure, read=lambda w: Range(None, None), grain=grain)
    first = groups[0]
    things = plural(events.member)
    first.label = f"First {grain}"
    age = Group(_output_name("periods_after", {first.name}), f"{grain.capitalize()}s after the first", "attribute")
    shape = CohortShape(events=events, periods=co.periods, value=co.value, cohort=first.name, age=age.name)
    share = co.value == "share"
    out = [OutMeasure(name="retained" if share else "members",
                      label=f"Share of {things} still with {events.activity}" if share else
                      f"{things[:1].upper()}{things[1:]} with {events.activity}", expr=AggExpr(agg="avg"),
                      format="percent" if share else "number", measure=None)]
    return Logical(intent="breakdown", parts=[events.part], groups=[first, age], measures=out,
                   window=events.window, sort=[(first.name, False), (age.name, False)], limit=plan.limit,
                   max_rows=ctx.max_rows, notes=[], fiscal_start=model.settings.fiscal_year_start_month,
                   conditions=[], shape=shape)


def cohort_definition(shape: CohortShape, span: str) -> str:
    e = shape.events
    things, single, period = plural(e.member), _singular(e.activity), e.grain or "month"
    when = f" {span}" if span else ""
    value = "the share of the cohort" if shape.value == "share" else "how many of them"
    return (f"{things[:1].upper()}{things[1:]} by the {period} of their first {single} ever{when} (each a cohort), "
            f"and {value} with {e.activity} in each {period} after it, up to {shape.periods}; 0 is the first "
            f"{period} itself. Only {period}s that have happened are shown, the latest perhaps not over yet.")


# ── in force on each date ──────────────────────────────────────────────────


def _active(plan: Plan, model: SemanticModel, ctx: Context) -> Logical:
    from core2.plan.ir import Duration

    a = plan.active
    assert a is not None
    start_role, end_role = _duration_dates(model, Duration(name="in force", start=a.start, end=a.end))
    grain = _period_grain(plan)
    first, last = _first_last(start_role)
    window = resolve_window(plan.time.window, today=ctx.today, first_data=first, last_data=last,
                            fiscal_start=model.settings.fiscal_year_start_month)
    if window.start is None:
        window = Range(first, window.end)
    # Never past the data's last date: what was in force after it is not known (every open item would still
    # count, month after month, up to today).
    _, ended_last = _first_last(end_role)
    data_end = max((d for d in (last, ended_last) if d is not None), default=None)
    end_cap = min(d for d in (window.end, ctx.today + dt.timedelta(days=1),
                              data_end + dt.timedelta(days=1) if data_end else None) if d is not None)
    if plan.time.window.kind == "all" and plan.time.grain is None and data_end is not None:
        # "How many subscriptions are active, by plan?": one figure, as of the latest date the data holds.
        starts = [unit_start(data_end, grain, fiscal_start=model.settings.fiscal_year_start_month)]
    else:
        starts = periods(Range(window.start, end_cap), grain)
    if not starts:
        raise ResolveError("unsupported", "no period to count in: name a window that has started")
    if len(starts) > MAX_PERIODS:
        raise ResolveError("unsupported", f"{len(starts)} {grain}s are more than {MAX_PERIODS} counted at once: "
                           "ask for a shorter time, or by quarter or year")
    table = start_role.table
    aliases = _Aliases()
    builder = _PartBuilder(model, table, aliases, ctx)
    builder._allowed(table)
    part = builder.part
    for df in model.tables[table].default_filters:
        part.preds.append(Pred(part.alias, df.column, df.op, list(df.values)))
    for f in plan.filters:
        found_f = find_slug(model, f.field)
        if not found_f or found_f[0] not in ("attribute", "entity"):
            raise ResolveError("unknown", f"nothing to narrow the rows by called {f.field}", [])
        attribute = _entity_label(model, f.field) if found_f[0] == "entity" else found_f[1]
        assert isinstance(attribute, Attribute)
        _shown(model, attribute, f.field, entity=found_f[0] == "entity", personal=ctx.personal_shown)
        column = model.columns[attribute.column]
        f, stored = _named_filter(column, f)
        alias = builder.reach(column.table, label=attribute.business_name, filtering=True) \
            if column.table != table else part.alias
        part.preds.append(Pred(alias, column.key, f.op, list(stored)))
    taken: set[str] = set()
    groups: list[Group] = []
    names: list[str] = []
    for slug in plan.group_by:
        found_g = find_slug(model, slug)
        if not found_g or found_g[0] not in ("attribute", "entity"):
            raise ResolveError("unknown", f"nothing to group by called {slug}", [])
        attribute = _entity_label(model, slug) if found_g[0] == "entity" else found_g[1]
        assert isinstance(attribute, Attribute)
        _shown(model, attribute, slug, entity=found_g[0] == "entity", personal=ctx.personal_shown)
        column = model.columns[attribute.column]
        alias = builder.reach(column.table, label=attribute.business_name) if column.table != table else part.alias
        name = _output_name(attribute.slug.replace(".", "_"), taken)
        taken.add(name)
        part.groups.append(PartGroup(name, alias, column.key, "attribute"))
        groups.append(Group(name, _attribute_label(model, attribute), "attribute", attribute=attribute.slug))
        names.append(name)
    period = Group(_output_name("period", taken), grain.capitalize(), "period", grain=grain)
    taken.add(period.name)
    things = plural(model.tables[table].business_name.lower())
    labels = {"active": f"{things[:1].upper()}{things[1:]} at {grain} end",
              "starts": f"Started ({start_role.name.lower()})", "ends": f"Ended ({end_role.name.lower()})",
              "end_rate": "End rate"}
    shown = ["active", "starts", "ends"] if a.status == "all" else [a.status]
    columns: dict[str, str] = {}
    out: list[OutMeasure] = []
    for status in shown:
        name = _output_name(status, taken)
        taken.add(name)
        columns[status] = name
        out.append(OutMeasure(name=name, label=labels[status], expr=AggExpr(agg="sum"),
                              format="percent" if status == "end_rate" else "number", measure=None))
    shape = ActiveShape(part=part, start=builder.date(start_role, kind="left"), end=builder.date(end_role, kind="left"),
                        periods=[(p, add_units(p, grain, 1)) for p in starts], status=a.status, columns=columns,
                        groups=names, things=things, start_name=start_role.name.lower(),
                        end_name=end_role.name.lower(), grain=grain)
    return Logical(intent="trend", parts=[part], groups=[period, *groups], measures=out,
                   window=Range(starts[0], add_units(starts[-1], grain, 1)), sort=[(period.name, False)],
                   limit=plan.limit, max_rows=ctx.max_rows, notes=[],
                   fiscal_start=model.settings.fiscal_year_start_month, conditions=[], shape=shape)


def active_definition(shape: ActiveShape) -> str:
    things, g = shape.things, shape.grain
    parts = {"active": f"in force at the end of each {g}: a {shape.start_name} on or before its last day, and no "
                       f"{shape.end_name} by then (an empty or far-off {shape.end_name} is none)",
             "starts": f"started: a {shape.start_name} in the {g}", "ends": f"ended: a {shape.end_name} in the {g}",
             "end_rate": f"end rate: those ended in the {g} as a share of those in force when it began"}
    asked = ["active", "starts", "ends"] if shape.status == "all" else [shape.status]
    return f"{things[:1].upper()}{things[1:]} {'; '.join(parts[k] for k in asked)}."


# ── on time, and ageing ────────────────────────────────────────────────────


def _date_role(model: SemanticModel, slug: str) -> DateRole:
    found = find_slug(model, slug)
    if not found or found[0] != "date":
        raise ResolveError("unknown", f"no date called {slug}", [])
    role = found[1]
    assert isinstance(role, DateRole)
    if role.granularity == "month":
        raise ResolveError("unsupported", f"{role.name} is kept by month, so days late cannot be counted from it")
    return role


def _rows_part(plan: Plan, model: SemanticModel, ctx: Context, table: str, taken: set[str]
               ) -> tuple[_PartBuilder, list[Group], list[str]]:
    """The rows of ``table``, narrowed by the plan's filters, with its groupings (each reached from it)."""
    builder = _PartBuilder(model, table, _Aliases(), ctx)
    builder._allowed(table)
    part = builder.part
    for df in model.tables[table].default_filters:
        part.preds.append(Pred(part.alias, df.column, df.op, list(df.values)))
    for f in plan.filters:
        found_f = find_slug(model, f.field)
        if not found_f or found_f[0] not in ("attribute", "entity"):
            raise ResolveError("unknown", f"nothing to narrow the rows by called {f.field}", [])
        attribute = _entity_label(model, f.field) if found_f[0] == "entity" else found_f[1]
        assert isinstance(attribute, Attribute)
        _shown(model, attribute, f.field, entity=found_f[0] == "entity", personal=ctx.personal_shown)
        column = model.columns[attribute.column]
        f, stored = _named_filter(column, f)
        alias = builder.reach(column.table, label=attribute.business_name, filtering=True) \
            if column.table != table else part.alias
        part.preds.append(Pred(alias, column.key, f.op, list(stored)))
    groups: list[Group] = []
    names: list[str] = []
    for slug in plan.group_by:
        found_g = find_slug(model, slug)
        if not found_g or found_g[0] not in ("attribute", "entity"):
            raise ResolveError("unknown", f"nothing to group by called {slug}", [])
        attribute = _entity_label(model, slug) if found_g[0] == "entity" else found_g[1]
        assert isinstance(attribute, Attribute)
        _shown(model, attribute, slug, entity=found_g[0] == "entity", personal=ctx.personal_shown)
        column = model.columns[attribute.column]
        alias = builder.reach(column.table, label=attribute.business_name) if column.table != table else part.alias
        name = _output_name(attribute.slug.replace(".", "_"), taken)
        taken.add(name)
        part.groups.append(PartGroup(name, alias, column.key, "attribute"))
        groups.append(Group(name, _attribute_label(model, attribute), "attribute", attribute=attribute.slug))
        names.append(name)
    return builder, groups, names


def _on_time(plan: Plan, model: SemanticModel, ctx: Context) -> Logical:
    ot = plan.on_time
    assert ot is not None
    due_role, done_role = _date_role(model, ot.due), _date_role(model, ot.done)
    taken: set[str] = set()
    if ot.ageing:
        return _ageing(plan, model, ctx, due_role, done_role, taken)
    # Each row done (a receipt), against the date its own row, or the row it points at (its order line), was due by.
    table = done_role.table
    if due_role.table != table and not _reaches(model, table, due_role.table):
        raise ResolveError("unsupported", f"{due_role.name} is not on the rows of {done_role.name} nor on rows they "
                           "point at")
    builder, groups, names = _rows_part(plan, model, ctx, table, taken)
    part = builder.part
    done = builder.date(done_role)
    due = builder.date(due_role, kind="left")
    first, last = _first_last(done_role)
    window = resolve_window(plan.time.window, today=ctx.today, first_data=first, last_data=last,
                            fiscal_start=model.settings.fiscal_year_start_month)
    if plan.time.window.kind == "all":
        window = Range(None, None)
    part.date_ranges.append((done, window))
    out_groups = list(groups)
    if plan.time.grain is not None:
        name = _output_name("period", taken)
        taken.add(name)
        part.groups.insert(0, PartGroup(name, done.alias, None, "period", grain=plan.time.grain, date=done))
        out_groups.insert(0, Group(name, plan.time.grain.capitalize(), "period", grain=plan.time.grain))
        names.insert(0, name)
    on_time = _output_name("late" if ot.late else "on_time", taken)
    shape = OnTimeShape(part=part, due=due, done=done, grace=ot.grace_days, ageing=False, as_of=ctx.today,
                        groups=names, columns={"on_time": on_time},
                        things=plural(model.tables[table].business_name.lower()), due_name=due_role.name.lower(),
                        done_name=done_role.name.lower(), late=ot.late)
    out = [OutMeasure(name=on_time, label="Late" if ot.late else "On time", expr=AggExpr(agg="avg"),
                      format="percent", measure=None)]
    period = next((g.name for g in out_groups if g.kind == "period"), None)
    # The worst first: the least on time, or the most late.
    sort = [(period, False)] if period else [(on_time, ot.late)] if out_groups else []
    return Logical(intent="trend" if period else "breakdown" if out_groups else "value", parts=[part],
                   groups=out_groups, measures=out, window=window, sort=sort, limit=plan.limit,
                   max_rows=ctx.max_rows, notes=[], fiscal_start=model.settings.fiscal_year_start_month,
                   conditions=[], shape=shape)


def _ageing(plan: Plan, model: SemanticModel, ctx: Context, due_role: DateRole, done_role: DateRole,
            taken: set[str]) -> Logical:
    """The rows due and not done yet, by days past due on the day they are aged on (today, or the window's end)."""
    table = due_role.table
    builder, groups, names = _rows_part(plan, model, ctx, table, taken)
    part = builder.part
    due = builder.date(due_role)
    done: DateUse | None = None
    elsewhere = None
    if done_role.table == table:
        done = builder.date(done_role, kind="left")
    else:
        link = next((j for j in model.joins.values() if j.from_table == done_role.table and j.to_table == table
                     and len(j.from_columns) == 1 and P.usable(j)), None)
        if link is None:
            raise ResolveError("unsupported", f"{done_role.name} is on rows that do not point at the "
                               f"{plural(model.tables[table].business_name.lower())}: nothing tells which are open")
        elsewhere = (done_role.table, link.from_columns[0], link.to_columns[0])
    if plan.time.window.kind in ("between", "until") and plan.time.window.end is not None:
        as_of = plan.time.window.end
    else:
        as_of = ctx.today
    band = _output_name("days_past_due", taken)
    taken.add(band)
    out_groups = [Group(band, "Days past due", "attribute", bands=list(AGEING)), *groups]
    names = [band, *names]
    things = plural(model.tables[table].business_name.lower())
    count = _output_name("open", taken)
    taken.add(count)
    out = [OutMeasure(name=count, label=f"Open {things}", expr=AggExpr(agg="count"), format="number", measure=None)]
    columns = {"open": count}
    for slug in plan.measures[:1]:
        found = find_slug(model, slug)
        if not found or found[0] != "measure":
            raise ResolveError("unknown", f"no measure called {slug}", [])
        m = found[1]
        assert isinstance(m, Measure)
        if m.table != table or not isinstance(m.expr, AggExpr) or m.expr.agg not in ("sum", "count"):
            raise ResolveError("unsupported", f"{m.business_name} is not a total of the "
                               f"{things}' own rows: ask for the open {things} alone")
        name = _output_name(m.slug, taken)
        taken.add(name)
        out.append(OutMeasure(name=name, label=f"{m.business_name} open", expr=m.expr, format=m.format, measure=m))
        columns["measure"] = name
    shape = OnTimeShape(part=part, due=due, done=done, grace=0, ageing=True, as_of=as_of, groups=names[1:],
                        columns=columns, band=band, done_elsewhere=elsewhere, things=things,
                        due_name=due_role.name.lower(), done_name=done_role.name.lower())
    return Logical(intent="breakdown", parts=[part], groups=out_groups, measures=out, window=Range(None, None),
                   sort=[], limit=plan.limit, max_rows=ctx.max_rows, notes=[],
                   fiscal_start=model.settings.fiscal_year_start_month, conditions=[], shape=shape)


def on_time_definition(shape: OnTimeShape, span: str, counted: float | None, late: float | None) -> str:
    if shape.ageing:
        open_how = (f"no {shape.done_name} yet" if shape.done_elsewhere is None else
                    f"no row with a {shape.done_name} pointing at it yet")
        return (f"{shape.things[:1].upper()}{shape.things[1:]} with {open_how}, by the days from their "
                f"{shape.due_name} to {shape.as_of:%d %B %Y}: not yet due when the {shape.due_name} is still ahead.")
    grace = f", or within {shape.grace} days after it" if shape.grace else ""
    when = f" {span}" if span else ""
    n = f" Of {int(counted):,}, {int(late or 0):,} were late." if counted is not None else ""
    if shape.late:
        grace = f" by more than {shape.grace} days" if shape.grace else ""
        return (f"Each of the {shape.things}{when} late when its {shape.done_name} is after its {shape.due_name}"
                f"{grace}; those without both dates are left out.{n}")
    return (f"Each of the {shape.things}{when} on time when its {shape.done_name} is on or before its "
            f"{shape.due_name}{grace}; those without both dates are left out.{n}")


# ── who makes most of a figure: ABC classes, the first 80%, deciles ────────

ABC = ["A", "B", "C", "D"]


@dataclass
class ParetoShape:
    inner: Logical                 # each member's figure: the plain answer grouped by the member
    members: list[str]             # inner group names that make one member (its name first)
    value: str                     # the inner column holding each member's figure
    mode: str                      # abc | pareto | ntile
    cuts: list[float]              # cumulative % each class (or the share listed) ends at
    groups: int                    # ntile: how many groups
    listed: bool                   # each member listed (pareto always), or one row a class
    member: str                    # "product"
    figure: str                    # "Net amount"
    columns: dict[str, str]        # what each output column is: class, value, share, running, members


def _pareto(plan: Plan, model: SemanticModel, ctx: Context) -> Logical:
    pa = plan.pareto
    assert pa is not None
    slug, _, member = _member_of(model, pa.of)
    if slug is None:
        raise ResolveError("unsupported", "classes are worked out for things the data lists: name whose figure "
                           "(each product, each customer)")
    if plan.group_by:
        raise ResolveError("unsupported", "classes are worked out over every member at once: ask without a "
                           "grouping, or for one group by its quoted name")
    cuts = sorted({float(c) for c in pa.cuts if 0 < float(c) < 100})
    if not cuts:
        raise ResolveError("unsupported", "a share between 0 and 100% is needed (80 for the first 80%)")
    inner_plan = plan.model_copy(update={
        "pareto": None, "measures": [pa.measure], "derived": [], "durations": [], "group_by": [slug], "sort": [],
        "limit": None, "intent": "breakdown", "chart": None,
        "time": plan.time.model_copy(update={"grain": None, "compare": None})})
    inner = resolve_plain(inner_plan, model, replace(ctx, unnamed_by_key=True) if slug in model.entities else ctx)
    if len(inner.measures) != 1:
        raise ResolveError("unsupported", "one figure at a time: ask for each measure on its own")
    figure = inner.measures[0]
    if not _adds(figure):
        raise ResolveError("unsupported", f"{figure.label} is not a total (an average or a ratio has no share of "
                           "a whole): ask for a figure that adds up, such as an amount or a count")
    attribute = _entity_label(model, slug) if slug in model.entities else model.attributes[slug]
    members = [g.name for g in inner.groups if g.attribute == attribute.slug]
    member_groups = [g for g in inner.groups if g.name in members]
    taken = {g.name for g in inner.groups}

    def name(hint: str) -> str:
        out = _output_name(hint, taken)
        taken.add(out)
        return out

    things = plural(member)
    word = figure.label
    columns: dict[str, str] = {}
    groups: list[Group] = []
    out: list[OutMeasure] = []
    listed = pa.mode == "pareto" or pa.members
    if listed:
        groups = list(member_groups)
        columns["value"] = name(figure.name)
        out.append(OutMeasure(columns["value"], word, AggExpr(agg="sum"), figure.format, figure.measure))
        columns["share"] = name("share")
        out.append(OutMeasure(columns["share"], f"Share of {word.lower()}", AggExpr(agg="sum"), "percent", None))
        columns["running"] = name("running_share")
        out.append(OutMeasure(columns["running"], "Running share", AggExpr(agg="sum"), "percent", None))
        if pa.mode != "pareto":
            columns["class"] = name("decile" if pa.mode == "ntile" else "class")
            groups.append(Group(columns["class"], _ntile_word(pa.groups) if pa.mode == "ntile" else "Class",
                                "attribute"))
    else:
        columns["class"] = name("decile" if pa.mode == "ntile" else "class")
        labels = [str(i) for i in range(1, pa.groups + 1)] if pa.mode == "ntile" else ABC[:len(cuts) + 1]
        groups.append(Group(columns["class"], _ntile_word(pa.groups) if pa.mode == "ntile" else "Class",
                            "attribute", bands=labels))
        columns["members"] = name(things.replace(" ", "_"))
        out.append(OutMeasure(columns["members"], things.capitalize(), AggExpr(agg="count"), "number", None))
        columns["value"] = name(figure.name)
        out.append(OutMeasure(columns["value"], word, AggExpr(agg="sum"), figure.format, figure.measure))
        columns["share"] = name("share")
        out.append(OutMeasure(columns["share"], f"Share of {word.lower()}", AggExpr(agg="sum"), "percent", None))
    shape = ParetoShape(inner=inner, members=members, value=figure.name, mode=pa.mode, cuts=cuts, groups=pa.groups,
                        listed=listed, member=member, figure=word, columns=columns)
    return dataclasses.replace(
        inner, intent="rank" if listed else "breakdown", groups=groups, measures=out,
        sort=[(columns["value"], True)] if listed else [], sort_asked=False, limit=plan.limit, share=False, having=[],
        notes=list(inner.notes), unit_group=None, shape=shape)


def _ntile_word(groups: int) -> str:
    return {10: "Decile", 4: "Quartile", 5: "Quintile", 3: "Tercile", 20: "Ventile"}.get(groups, "Group")


def pareto_definition(shape: ParetoShape, span: str) -> str:
    when = f" {span}" if span else ""
    things, word = plural(shape.member), shape.figure.lower()
    if shape.mode == "ntile":
        kind = _ntile_word(shape.groups).lower()
        return (f"{things[:1].upper()}{things[1:]} ranked by their {word}{when}, from the largest, and cut into "
                f"{shape.groups} groups of the same number of {things}: {kind} 1 holds the largest.")
    if shape.mode == "pareto":
        return (f"{things[:1].upper()}{things[1:]} ranked by their {word}{when}, from the largest: those listed make "
                f"up the first {shape.cuts[0]:g}% of the total, the one that takes it past {shape.cuts[0]:g}% "
                "included.")
    edges = [0.0, *shape.cuts, 100.0]
    parts = [f"{ABC[i]} the {things} making {'the first ' if i == 0 else 'the next '}"
             f"{edges[i + 1] - edges[i]:g}%" if i < len(shape.cuts) else f"{ABC[i]} the rest"
             for i in range(len(shape.cuts) + 1)]
    return (f"{things[:1].upper()}{things[1:]} ranked by their {word}{when}, from the largest: " + "; ".join(parts) +
            ". A class holds the member that takes the total past its edge.")


# ── funnels: rows through their dates in order ─────────────────────────────


@dataclass
class FunnelShape:
    part: Part
    stages: list[DateUse]
    labels: list[str]              # each stage in words ("Order date")
    things: str                    # "orders"
    columns: dict[str, str]        # stage, reached, of_first, of_previous, days


def _funnel(plan: Plan, model: SemanticModel, ctx: Context) -> Logical:
    fu = plan.funnel
    assert fu is not None
    roles: list[DateRole] = []
    for slug in fu.stages:
        found = find_slug(model, slug)
        if not found or found[0] != "date":
            raise ResolveError("unknown", f"no date called {slug}", sorted(model.date_roles)[:12])
        role = found[1]
        assert isinstance(role, DateRole)
        roles.append(role)
    if len({r.key for r in roles}) != len(roles):
        raise ResolveError("unsupported", "each stage of a funnel is a date of its own")
    table = roles[0].table
    other = next((r for r in roles if r.table != table), None)
    if other is not None:
        raise ResolveError("unsupported", f"a funnel follows the dates of the same rows: {other.name} is not on the "
                           f"{plural(model.tables[table].business_name.lower())}")
    if plan.group_by or plan.time.grain:
        raise ResolveError("unsupported", "a funnel is drawn for its rows all together: ask for one group by its "
                           "quoted name, or for one period")
    taken: set[str] = set()
    builder, _, _ = _rows_part(plan, model, ctx, table, taken)
    part = builder.part
    uses = [builder.date(roles[0]), *(builder.date(r, kind="left") for r in roles[1:])]
    first, last = _first_last(roles[0])
    window = resolve_window(plan.time.window, today=ctx.today, first_data=first, last_data=last,
                            fiscal_start=model.settings.fiscal_year_start_month)
    if plan.time.window.kind == "all":
        window = Range(None, None)
    part.date_ranges.append((uses[0], window))
    labels = [r.name[:1].upper() + r.name[1:] for r in roles]
    things = plural(model.tables[table].business_name.lower())
    names = {k: _output_name(k, taken) for k in ("stage", "reached", "of_first", "of_previous", "days")}
    groups = [Group(names["stage"], "Stage", "attribute", bands=labels)]
    out = [OutMeasure(names["reached"], f"{things[:1].upper()}{things[1:]}", AggExpr(agg="count"), "number", None),
           OutMeasure(names["of_first"], "Share of the first stage", AggExpr(agg="sum"), "percent", None),
           OutMeasure(names["of_previous"], "Share of the stage before", AggExpr(agg="sum"), "percent", None),
           OutMeasure(names["days"], "Average days from the stage before", AggExpr(agg="avg"), "days", None)]
    shape = FunnelShape(part=part, stages=uses, labels=labels, things=things, columns=names)
    return Logical(intent="breakdown", parts=[part], groups=groups, measures=out, window=window, sort=[],
                   limit=None, max_rows=ctx.max_rows, notes=[], fiscal_start=model.settings.fiscal_year_start_month,
                   conditions=[], shape=shape)


def funnel_definition(shape: FunnelShape, span: str) -> str:
    stage = shape.labels[0].lower()
    when = f" with {'an' if stage[:1] in 'aeiou' else 'a'} {stage} {span}" if span else ""
    return (f"{shape.things[:1].upper()}{shape.things[1:]}{when}, through "
            f"{', '.join(lbl.lower() for lbl in shape.labels)}: a stage is reached when its date and every one before "
            "it are filled; the days are from the stage before, over those that reached both.")


# ── bought together ────────────────────────────────────────────────────────


@dataclass
class TogetherShape:
    part: Part                     # each basket's items: the rows, by basket and item
    basket: str                    # the part's group holding the basket
    item: str                      # the part's group holding the item's name
    anchor: list[object]           # the item names a quoted filter kept: the pairs that hold one of them
    item_word: str                 # "product"
    basket_word: str               # "order"
    columns: dict[str, str]        # item, with, together, support, confidence, lift


def _together(plan: Plan, model: SemanticModel, ctx: Context) -> Logical:
    to = plan.together
    assert to is not None
    found = find_slug(model, to.of)
    if not found or found[0] != "entity":
        raise ResolveError("unknown", f"nothing the data lists called {to.of}: name the items (each product)",
                           sorted(model.entities)[:12])
    entity = found[1]
    assert isinstance(entity, Entity)
    counted = find_slug(model, to.basket)
    if not counted or counted[0] != "measure" or getattr(counted[1].expr, "agg", "") not in ("count", "count_distinct"):
        raise ResolveError("unsupported", f"{to.basket} does not count baskets: name a count of orders (or "
                           "invoices, visits) that the items are on")
    _, column, basket_word = _member_of(model, to.basket)
    if column is None:
        raise ResolveError("unsupported", f"{to.basket} does not count baskets: name a count of orders (or "
                           "invoices, visits) that the items are on")
    label = _entity_label(model, to.of)
    # The item's own filter (a quoted product) keeps the pairs holding it, not the rows of its baskets.
    anchor: list[object] = []
    rest = []
    for f in plan.filters:
        if f.field in (to.of, label.slug) and f.op in ("eq", "in"):
            _, stored = _named_filter(model.columns[label.column], f)
            anchor += list(stored)
        else:
            rest.append(f)
    inner_plan = plan.model_copy(update={
        "together": None, "measures": [to.basket], "derived": [], "durations": [], "group_by": [to.of],
        "filters": rest, "sort": [], "limit": None, "intent": "breakdown", "chart": None,
        "time": plan.time.model_copy(update={"grain": None, "compare": None})})
    inner = resolve_plain(inner_plan, model, ctx)
    if len(inner.parts) != 1:
        raise ResolveError("unsupported", f"the {plural(entity.business_name.lower())} and the {plural(basket_word)} "
                           "are not on the same rows")
    part = inner.parts[0]
    owner = model.columns[column].table
    if part.table != owner:
        link = next((j for j in model.joins.values() if j.from_table == part.table and j.to_table == owner
                     and list(j.to_columns) == [column] and P.usable(j)), None)
        if link is None:
            raise ResolveError("unsupported", f"each {basket_word} of the "
                               f"{plural(entity.business_name.lower())} cannot be told from their rows")
        column = link.from_columns[0]
    item = next((g.name for g in inner.groups if g.attribute == label.slug), None)
    if item is None:
        raise ResolveError("unsupported", f"the {plural(entity.business_name.lower())} have no name to show")
    basket = _output_name("basket", {g.name for g in inner.groups})
    part.groups = [g for g in part.groups if g.name == item]
    part.groups.append(PartGroup(basket, part.alias, column, "attribute"))
    part.measures = []
    part.distinct_only = True
    taken = {basket, item}
    names = {k: _output_name(k, taken) for k in ("item", "with_item", "together", "support", "confidence", "lift")}
    word = entity.business_name
    groups = [Group(names["item"], word, "attribute"), Group(names["with_item"], "Bought with", "attribute")]
    baskets = plural(basket_word)
    out = [OutMeasure(names["together"], f"{baskets[:1].upper()}{baskets[1:]} with both", AggExpr(agg="count"),
                      "number", None),
           OutMeasure(names["support"], f"Share of {baskets}", AggExpr(agg="sum"), "percent", None),
           OutMeasure(names["confidence"], f"Share of the first's {baskets}", AggExpr(agg="sum"), "percent", None),
           OutMeasure(names["lift"], "Lift", AggExpr(agg="sum"), "number", None)]
    shape = TogetherShape(part=part, basket=basket, item=item, anchor=anchor, item_word=word.lower(),
                          basket_word=basket_word, columns=names)
    return dataclasses.replace(inner, intent="rank", groups=groups, measures=out, sort=[(names["together"], True)],
                               sort_asked=False, limit=plan.limit or 20, share=False, having=[], unit_group=None,
                               shape=shape)


def together_definition(shape: TogetherShape, span: str) -> str:
    when = f" {span}" if span else ""
    baskets = plural(shape.basket_word)
    return (f"Pairs of {plural(shape.item_word)} on the same {shape.basket_word}{when}: how many {baskets} hold "
            f"both, their share of all {baskets}, how many of the {baskets} with the first also hold the second, "
            "and the lift (above 1: bought together more often than chance would have it).")


# ── a figure against its target ────────────────────────────────────────────


@dataclass
class TargetInfo:
    actual: str                    # the actual's label
    target: str                    # the target's label
    difference: str                # the derived difference's name
    percent: str                   # the derived % of target's name
    lower_is_better: bool          # a cost against its budget: under it is good


TARGET_SHARE = "% of target"
TARGET_DIFFERENCE = "Difference"


_PERIOD_COLUMNS = ("year", "quarter", "month", "week")


def _by_period(plan: Plan, model: SemanticModel, tables: tuple[str, str]) -> Plan:
    """"By quarter" read as one table's own quarter column (a quota kept a row a quarter, with its year and quarter
    as numbers): the other figure has no such column, so both are counted by the period, each by its own date
    ("by year and quarter" is by quarter)."""
    if plan.time.grain is not None:
        return plan

    def period(slug: str) -> str:
        a = model.attributes.get(slug)
        column = model.columns.get(a.column) if a is not None else None
        word = a.business_name.strip().lower() if a is not None else ""
        return word if column is not None and column.table in tables and word in _PERIOD_COLUMNS else ""

    periods = [g for g in plan.group_by if period(g)]
    if not periods:
        return plan
    finest = max((period(g) for g in periods), key=_PERIOD_COLUMNS.index)
    return plan.model_copy(update={"group_by": [g for g in plan.group_by if g not in periods],
                                   "time": plan.time.model_copy(update={"grain": finest})})


_FLIPPED = {"lt": "gt", "lte": "gte", "gt": "lt", "gte": "lte", "eq": "eq", "ne": "ne"}


def _on_target(plan: Plan, actual: Measure, target: Measure) -> list:
    """The plan's filters, with a figure set against its target said as the difference ("which stores missed
    target": the actual below the target is a difference below 0), and one on a difference or share the AI
    worked out itself said as the answer's own ("Difference", "% of target")."""
    from core2.resolve.resolver import _plain

    own = {_plain(d.name): d for d in plan.derived}
    out = []
    for f in plan.filters:
        names = {actual.slug, target.slug}
        if f.field in names and len(f.values) == 1 and f.values[0] in names - {f.field} and f.op in _FLIPPED:
            op = f.op if f.field == actual.slug else _FLIPPED[f.op]
            out.append(f.model_copy(update={"field": TARGET_DIFFERENCE, "op": op, "values": [0], "total": True}))
            continue
        made = own.get(_plain(f.field))
        if made is not None and set(made.measures) == names:
            name = TARGET_DIFFERENCE if made.op == "difference" else TARGET_SHARE if made.op == "ratio" else None
            if name is not None:
                values = f.values
                if made.op == "difference" and made.measures[0] == target.slug:   # target less actual
                    values, f = [-float(v) for v in f.values], f.model_copy(update={"op": _FLIPPED.get(f.op, f.op)})
                elif made.op == "ratio" and made.scale != 100:
                    values = [float(v) * 100 / (made.scale or 1) for v in f.values]
                out.append(f.model_copy(update={"field": name, "values": values, "total": True}))
                continue
        out.append(f)
    return out


def with_target(plan: Plan, model: SemanticModel) -> tuple[Plan, TargetInfo]:
    """The plan of a figure against its target: both figures, their difference, the actual as a % of the target."""
    from core2.plan.ir import Derived

    tg = plan.target
    assert tg is not None
    measures: list[Measure] = []
    for slug in (tg.actual, tg.target):
        found = find_slug(model, slug)
        if not found or found[0] != "measure":
            raise ResolveError("unknown", f"no measure called {slug}", sorted(model.measures)[:12])
        assert isinstance(found[1], Measure)
        measures.append(found[1])
    actual, target = measures
    if actual.slug == target.slug:
        raise ResolveError("unsupported", "a figure against its target needs two figures")
    from core2.bootstrap import names
    from core2.plan.analyses import TARGET_WORDS

    said = {names.singular(w) for text in (target.business_name, target.slug, model.tables[target.table].business_name
                                           if target.table in model.tables else "") for w in names.tokens(text)}
    if not said & (TARGET_WORDS | {"plan", "expected", "objective"}):
        # "Total qty ordered vs scrapped": two figures side by side, not the quantity ordered against a target of
        # scrapped quantity ("Order quantity was 34770% of scrapped quantity").
        raise ResolveError("unsupported", f"{target.business_name} is no target, budget or quota. For "
                           f"{actual.business_name} and {target.business_name} side by side, put both in measures; "
                           "for one as a share of the other, a derived ratio")
    derived = [Derived(name=TARGET_DIFFERENCE, op="difference", measures=[actual.slug, target.slug]),
               Derived(name=TARGET_SHARE, op="ratio", measures=[actual.slug, target.slug], scale=100.0)]
    plan = _by_period(plan, model, (actual.table, target.table))
    plan = plan.model_copy(update={"filters": _on_target(plan, actual, target)})
    plan = plan.model_copy(update={"target": None, "measures": [actual.slug, target.slug], "derived": derived,
                                   "intent": plan.intent or ("trend" if plan.time.grain else "breakdown"
                                                             if plan.group_by else "value")})
    return plan, TargetInfo(actual.business_name, target.business_name, TARGET_DIFFERENCE, TARGET_SHARE,
                            actual.better == "lower")
