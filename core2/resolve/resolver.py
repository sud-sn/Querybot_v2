"""From a plan to a logical query: which tables, joins, dates, filters and periods.

The resolver owns the rules the answer depends on: one join path per attribute
(paths.py), the date each measure is counted by, exact windows (time.py), a
snapshot's last day per period, placeholder dates left out, default filters
applied, two facts aggregated separately and lined up on their shared
groupings. It writes no SQL; it records what it decided, in words, for the
answer.
"""

from __future__ import annotations

import datetime as dt
import difflib
import re
from dataclasses import dataclass, field

from core2 import ids
from core2.bootstrap.names import plural
from core2.model.schema import (
    AggExpr,
    Attribute,
    Calendar,
    Column,
    ColumnFilter,
    DateRole,
    Entity,
    Measure,
    MeasureExpr,
    OpExpr,
    RefExpr,
    SemanticModel,
)
from core2.plan.ir import TIME_ATTRIBUTES, Filter, Plan
from core2.resolve import paths as P
from core2.resolve.time import (Range, add_units, partial_periods, periods, resolve_window, shift, year_back,
                                year_basis)
from core2.warehouse import dialect as D

# Aliases a validator screens as statements, whatever their quoting.
_UNSAFE_ALIASES = {"call", "load", "get", "put", "copy", "exec", "execute", "function", "procedure", "merge", "grant",
                   "revoke", "drop", "delete", "insert", "update", "create", "alter", "truncate", "use", "set"}


class ResolveError(Exception):
    """The plan cannot be answered as it stands; ``kind`` says why, ``options`` what could be chosen."""

    def __init__(self, kind: str, message: str, options: list[str] | None = None):
        super().__init__(message)
        self.kind = kind            # unknown | ambiguous | unconfirmed | unsupported | denied | empty
        self.message = message
        self.options = options or []
        self.field: str | None = None   # for "ambiguous": the slug whose link the reader must name (plan.via)
        self.unlinked: str | None = None   # for "not linked": the table key the measure's table does not reach


@dataclass
class Context:
    today: dt.date
    allowed_tables: set[str] | None = None      # table keys the reader may use; None = all
    max_rows: int = 5000
    split_units: bool = False                   # a reader's answer: quantities kept apart by unit (issue E4)


@dataclass
class DateUse:
    role: DateRole
    alias: str               # where the date lives: the calendar's alias, or the measure table's
    mode: str                # day_calendar | month_calendar | date | timestamp | yyyymmdd | yyyymm
    column: str              # column key of the date (or of the period key)
    calendar: Calendar | None = None
    join: Joined | None = None


@dataclass
class Joined:
    alias: str
    table: str
    on: list[tuple[str, str, str]]   # (left alias, left column key, right column key)
    kind: str = "left"
    what: str = ""


@dataclass
class OutMeasure:
    name: str                # output column
    label: str
    expr: MeasureExpr        # column keys resolved against this part's base alias
    format: str
    measure: Measure | None
    semi: bool = False


@dataclass
class PartGroup:
    name: str
    alias: str
    column: str | None       # column key, for attributes
    kind: str                # attribute | period | time
    grain: str | None = None
    time_attr: str | None = None
    date: DateUse | None = None


@dataclass
class Pred:
    alias: str
    column: str
    op: str
    values: list
    text: str = ""


@dataclass
class Part:
    table: str
    alias: str
    joins: list[Joined] = field(default_factory=list)
    measures: list[OutMeasure] = field(default_factory=list)
    groups: list[PartGroup] = field(default_factory=list)
    preds: list[Pred] = field(default_factory=list)
    date: DateUse | None = None
    date_ranges: list[tuple[DateUse, Range]] = field(default_factory=list)
    snapshot: bool = False
    distinct_only: bool = False


@dataclass
class Group:
    name: str
    label: str
    kind: str                # attribute | member_code | period | time
    grain: str | None = None
    attribute: str | None = None


@dataclass
class Logical:
    intent: str
    parts: list[Part]
    groups: list[Group]
    measures: list[OutMeasure]          # output order (the first part's objects, for formats and labels)
    window: Range
    compare: Range | None = None
    sort: list[tuple[str, bool]] = field(default_factory=list)
    limit: int | None = None
    share: bool = False
    having: list[Pred] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    partial: list[dt.date] = field(default_factory=list)
    fiscal_start: int | None = None
    max_rows: int = 5000
    unit_group: str | None = None       # the grouping that keeps different units apart, when one was added
    unit_order: list[str] = field(default_factory=list)   # its units, the most used first (when profiled)
    expected: list[dt.date] = field(default_factory=list)  # a series' periods within the data: one with no row is a gap
    data_last: dt.date | None = None    # the last day the series' date has data (periods after it have none yet)


class _Aliases:
    def __init__(self) -> None:
        self.taken: set[str] = set()

    def new(self, hint: str) -> str:
        """A table alias that never needs quoting: the governed executor's row policies
        name a table by its alias, unquoted ("order" would make ``WHERE order.region``)."""
        base = re.sub(r"[^a-z0-9_]", "_", ids.slug(hint))[:24] or "t"
        if base in _UNSAFE_ALIASES or base.upper() in D.RESERVED or base[0].isdigit():
            base = f"t_{base}"
        return ids.unique_slug(base, self.taken)


def _output_name(text: str, taken: set[str]) -> str:
    base = ids.slug(text)[:40] or "value"
    if base in _UNSAFE_ALIASES:
        base = f"{base}_value"
    return ids.unique_slug(base, taken)


def _slugs(model: SemanticModel, kind: str) -> list[str]:
    if kind == "measure":
        return sorted(m.slug for m in model.measures.values() if not m.hidden)
    if kind == "date":
        return sorted(r.slug for r in model.date_roles.values() if r.kind != "audit")
    return sorted([*model.entities, *model.attributes, *TIME_ATTRIBUTES])


def _closest(name: str, candidates: list[str], n: int = 6) -> list[str]:
    """The names a mistyped or invented slug most likely meant (for the repair round and the user)."""
    words = set(re.split(r"[._:]", name.casefold()))
    by_words = [c for c in candidates if words & set(re.split(r"[._:]", c.casefold()))]
    near = difflib.get_close_matches(name, candidates, n=n, cutoff=0.5)
    return list(dict.fromkeys([*near, *by_words]))[:n] or candidates[:n]


def find_slug(model: SemanticModel, slug: str) -> tuple[str, object] | None:
    """What a slug names: a measure, an attribute, an entity (its label) or a date."""
    for m in model.measures.values():
        if m.slug == slug:
            return "measure", m
    if slug in model.attributes:
        return "attribute", model.attributes[slug]
    if slug in model.entities:
        return "entity", model.entities[slug]
    for r in model.date_roles.values():
        if r.slug == slug:
            return "date", r
    return None


def _can_lack(model: SemanticModel, path: P.Path | None, column: Column) -> bool:
    """Whether some rows may reach no member: a link that misses, or a value that is empty."""
    if path is not None and any(j.match_rate < 0.999 for j in path.joins):
        return True
    p = column.profile
    table_rows = model.tables[column.table].row_count or 0
    return p is None or p.non_null < table_rows


def _member_identity(model: SemanticModel, attribute: Attribute) -> str | None:
    """The code (or key) telling apart members that share a name, when the names repeat."""
    column = model.columns[attribute.column]
    entity = next((e for e in model.entities.values() if e.label_column == column.key), None)
    if entity is None or column.profile is None:
        return None
    if column.profile.distinct >= (model.tables[column.table].row_count or 0):
        return None
    if entity.code_column:
        return entity.code_column
    return entity.key_columns[0] if len(entity.key_columns) == 1 else None


def _entity_label(model: SemanticModel, slug: str) -> Attribute:
    entity = model.entities[slug]
    column = entity.label_column or entity.code_column or (entity.key_columns[0] if entity.key_columns else None)
    for a in model.attributes.values():
        if a.column == column:
            return a
    raise ResolveError("unknown", f"{entity.business_name} has no name to show")


class _PartBuilder:
    """Joins for one measure table: every table joined once per path, aliases reused."""

    def __init__(self, model: SemanticModel, table: str, aliases: _Aliases, ctx: Context):
        self.model = model
        self.ctx = ctx
        self.aliases = aliases
        self.part = Part(table=table, alias=aliases.new(model.tables[table].slug or "f"))
        self.by_path: dict[tuple[str, ...], str] = {(): self.part.alias}
        self.paths_used: dict[str, P.Path] = {}

    def _allowed(self, table: str) -> None:
        if self.ctx.allowed_tables is not None and table not in self.ctx.allowed_tables:
            raise ResolveError("denied", f"{self.model.tables[table].business_name} is not available to you")

    def reach(self, table: str, *, through: str | None = None, role: str | None = None, label: str = "") -> str:
        """The alias holding ``table``, joining along the one path rule (or through the named role)."""
        self._allowed(table)
        if role:
            chosen = P.with_role(self.model, self.part.table, table, role, through=through)
            if not chosen:
                raise ResolveError("unknown", f"{label or self.model.tables[table].business_name} is not reached as "
                                   f"{role}", P.role_names(self.model, self.part.table, table))
            return self._walk(table, chosen[0])
        try:
            path = P.best_path(self.model, self.part.table, table, through=through)
        except P.Ambiguous as exc:
            # The choice is offered by role ("Bill-to", "Ship-to"): what the plan's via names.
            options = [", ".join(P.roles(p)) or p.describe(self.model) for p in exc.options]
            raise ResolveError("ambiguous", f"{label or self.model.tables[table].business_name} can be reached "
                               "more than one way", options) from None
        if path is None:
            waiting = P.all_paths(self.model, self.part.table, table, through=through, unconfirmed=True)
            if waiting:
                links = [j for j in waiting[0].joins if not P.usable(j)]
                raise ResolveError(
                    "unconfirmed", f"{label or self.model.tables[table].business_name} is reached through a link "
                    "an admin has not confirmed yet", [f"{self.model.columns[j.from_columns[0]].business_name} -> "
                                                       f"{self.model.tables[j.to_table].business_name}" for j in links])
            error = ResolveError("unsupported", f"{label or self.model.tables[table].business_name} is not linked to "
                                 f"{self.model.tables[self.part.table].business_name}")
            error.unlinked = table
            raise error
        return self._walk(table, path)

    def _walk(self, table: str, path: P.Path) -> str:
        alias = self.part.alias
        walked: tuple[str, ...] = ()
        for j in path.joins:
            walked = walked + (j.key,)
            if walked not in self.by_path:
                self._allowed(j.to_table)
                new = self.aliases.new(j.role or self.model.tables[j.to_table].slug)
                self.part.joins.append(Joined(
                    alias=new, table=j.to_table,
                    on=[(alias, f, t) for f, t in zip(j.from_columns, j.to_columns)], kind="left",
                    what=j.role or self.model.tables[j.to_table].business_name))
                self.by_path[walked] = new
            alias = self.by_path[walked]
        if path.joins:
            self.paths_used[table] = path
        return alias

    def date(self, role: DateRole) -> DateUse:
        """Where a date role's date lives, joining its calendar once under the role's own name."""
        model = self.model
        column = model.columns[role.column]
        if role.calendar:
            calendar = model.calendars[role.calendar]
            key = ("calendar", role.key)
            if key not in self.by_path:
                self._allowed(role.calendar)
                join = model.joins.get(role.calendar_join or "")
                if join is None:
                    raise ResolveError("unsupported", f"{role.name} has no calendar link")
                alias = self.aliases.new(role.slug or role.name)
                self.part.joins.append(Joined(alias=alias, table=role.calendar,
                                              on=[(self.part.alias, join.from_columns[0], join.to_columns[0])],
                                              kind="inner", what=role.name))
                self.by_path[key] = alias
            alias = self.by_path[key]
            if calendar.grain == "month":
                return DateUse(role, alias, "month_calendar", calendar.key_column or "", calendar)
            return DateUse(role, alias, "day_calendar", calendar.date_column or "", calendar)
        if column.data_type == "date":
            return DateUse(role, self.part.alias, "date", column.key)
        if column.data_type == "timestamp":
            return DateUse(role, self.part.alias, "timestamp", column.key)
        if role.granularity == "month":
            return DateUse(role, self.part.alias, "yyyymm", column.key)
        return DateUse(role, self.part.alias, "yyyymmdd", column.key)


def _measure_expr(model: SemanticModel, measure: Measure) -> MeasureExpr:
    """References to other measures inlined: the compiler sees only aggregates and operations."""
    def inline(expr: MeasureExpr, depth: int = 0) -> MeasureExpr:
        if depth > 5:
            raise ResolveError("unsupported", f"{measure.business_name} refers to itself")
        if isinstance(expr, RefExpr):
            return inline(model.measures[expr.measure].expr, depth + 1)
        if isinstance(expr, OpExpr):
            return OpExpr(op=expr.op, args=[inline(a, depth + 1) for a in expr.args], scale=expr.scale)
        return expr

    return inline(measure.expr)


def _with_filters(expr: MeasureExpr, filters: list[ColumnFilter]) -> MeasureExpr:
    if not filters:
        return expr
    if isinstance(expr, AggExpr):
        return AggExpr(agg=expr.agg, column=expr.column, filters=[*expr.filters, *filters])
    if isinstance(expr, OpExpr):
        return OpExpr(op=expr.op, args=[_with_filters(a, filters) for a in expr.args], scale=expr.scale)
    return expr


def _date_role_for(model: SemanticModel, plan: Plan, measure: Measure, table: str) -> DateRole | None:
    if plan.time.date:
        found = find_slug(model, plan.time.date)
        if not found or found[0] != "date":
            raise ResolveError("unknown", f"no date called {plan.time.date}", _closest(plan.time.date, _slugs(model, "date")))
        role = found[1]
        assert isinstance(role, DateRole)
        if role.table == table:
            return role
    key = measure.default_date or model.tables[table].default_date
    return model.date_roles.get(key) if key else None


def _first_last(role: DateRole) -> tuple[dt.date | None, dt.date | None]:
    """The first and last day the role's data covers (a monthly date covers its whole last month)."""
    last = role.last
    if last is not None and role.granularity == "month":
        last = add_units(last.replace(day=1), "month", 1) - dt.timedelta(days=1)
    return role.first, last


def resolve(plan: Plan, model: SemanticModel, ctx: Context) -> Logical:
    if plan.kind != "query":
        raise ResolveError("unsupported", f"a {plan.kind} plan is not a query")
    aliases = _Aliases()
    out_names: set[str] = set()
    notes: list[str] = []

    # Measures, and the table each is counted on.
    chosen: list[tuple[Measure | None, MeasureExpr, str, str, str]] = []   # (measure, expr, table, label, format)
    for slug in dict.fromkeys(plan.measures):      # a measure named twice is shown once
        found = find_slug(model, slug)
        if not found or found[0] != "measure":
            raise ResolveError("unknown", f"no measure called {slug}", _closest(slug, _slugs(model, "measure")))
        m = found[1]
        assert isinstance(m, Measure)
        expr = _with_filters(_measure_expr(model, m), m.filters)
        chosen.append((m, expr, m.table, m.business_name, m.format))
        if m.filters:
            notes.append(f"{m.business_name} counts only rows where " + "; ".join(
                f"{model.columns[f.column].business_name} {f.op.replace('_', ' ')} {', '.join(map(str, f.values))}"
                for f in m.filters))
        if m.kind == "proposed" or m.status == "proposed":
            notes.append(f"{m.business_name} is a proposed measure, not yet confirmed.")
    for d in plan.derived:
        operands = []
        for slug in d.measures:
            found = find_slug(model, slug)
            if not found or found[0] != "measure":
                raise ResolveError("unknown", f"no measure called {slug}", _closest(slug, _slugs(model, "measure")))
            operands.append(found[1])
        a, b = operands
        assert isinstance(a, Measure) and isinstance(b, Measure)
        if a.table != b.table:
            raise ResolveError("unsupported", f"{d.name} combines measures from two tables")
        op = {"ratio": "ratio", "difference": "subtract", "sum": "add", "product": "multiply"}[d.op]
        expr = OpExpr(op=op, args=[_with_filters(_measure_expr(model, a), a.filters),  # type: ignore[arg-type]
                                   _with_filters(_measure_expr(model, b), b.filters)], scale=d.scale)
        chosen.append((None, expr, a.table, d.name, "percent" if d.scale == 100 else "number"))
        sign = {"ratio": "÷", "difference": "−", "sum": "+", "product": "×"}[d.op]
        scaled = f" × {d.scale:g}" if d.scale != 1 else ""
        notes.append(f"{d.name} = {a.business_name} {sign} {b.business_name}{scaled}, worked out for this question "
                     "(not a saved measure yet).")

    # Groupings asked for.
    wanted: list[tuple[str, Attribute | None, str | None]] = []   # (slug, attribute, time attribute)
    for slug in plan.group_by:
        if slug in TIME_ATTRIBUTES:
            wanted.append((slug, None, slug.split(":", 1)[1]))
            continue
        found = find_slug(model, slug)
        if not found or found[0] not in ("attribute", "entity"):
            raise ResolveError("unknown", f"nothing to group by called {slug}", _closest(slug, _slugs(model, "group")))
        attribute = _entity_label(model, slug) if found[0] == "entity" else found[1]
        assert isinstance(attribute, Attribute)
        wanted.append((slug, attribute, None))

    intent = plan.intent or ("trend" if plan.time.grain else ("breakdown" if wanted else "value"))
    if not chosen:
        if wanted and intent in ("list", "breakdown", "rank", "count", "value"):
            return _members(plan, model, ctx, wanted, aliases, intent)
        raise ResolveError("unknown", "the question names no measure", _slugs(model, "measure")[:12])
    unit_slug = _unit_split(plan, model, ctx, intent, chosen, wanted)
    if unit_slug is not None:
        wanted.append((unit_slug, model.attributes[unit_slug], None))

    # One part per measure table (and per snapshot kind: levels and flows of one
    # snapshot table are counted over different rows).
    by_part: dict[tuple[str, bool], list[tuple[Measure | None, MeasureExpr, str, str, str]]] = {}
    for item in chosen:
        m = item[0]
        semi = bool(m and m.additivity == "semi_additive" and m.time_aggregation == "last")
        by_part.setdefault((item[2], semi), []).append(item)

    first_role: DateRole | None = None
    builders: list[tuple[_PartBuilder, DateRole | None]] = []
    for (table, semi), items in by_part.items():
        builder = _PartBuilder(model, table, aliases, ctx)
        builder._allowed(table)
        role = _date_role_for(model, plan, items[0][0] or _fallback_measure(model, table), table)
        builder.part.snapshot = semi
        builders.append((builder, role))
        first_role = first_role or role
        if plan.time.date and role and role.slug != plan.time.date:
            notes.append(f"{model.tables[table].business_name} is counted by {role.name}.")

    fiscal_start = model.settings.fiscal_year_start_month
    first, last = _first_last(first_role) if first_role else (None, None)
    window = resolve_window(plan.time.window, today=ctx.today, first_data=first, last_data=last,
                            fiscal_start=fiscal_start)
    notes += window.notes
    notes += year_basis(plan.time.window, window, fiscal_start=fiscal_start)
    compare: Range | None = None
    if plan.time.compare:
        c = plan.time.compare
        if c.kind == "window" and c.window:
            compare = resolve_window(c.window, today=ctx.today, first_data=first, last_data=last,
                                     fiscal_start=fiscal_start)
        elif c.kind == "previous_period" and plan.time.window.kind in ("this", "to_date") and window.start \
                and window.end:
            # "This month so far" against the same days of last month, not the days just before it.
            before = add_units(window.start, plan.time.window.unit or "month", -1)
            compare = Range(before, min(before + (window.end - window.start), window.start))
        else:
            compare = shift(window, c.kind)
        intent = "compare"
        if not all(semi for _, semi in by_part):
            window, compare, cut = _like_for_like(window, compare, last, c.kind)
            if cut:
                notes.append(cut)

    # A date the question names, that some rows lack (9999-12-31 for "not left yet"), keeps to the rows
    # that have it, over all time too: "how many have left" counts those with a termination date.
    named_gaps = any(role is not None and role.slug == plan.time.date and (
        role.placeholder_share > 0.001 or role.coverage < 0.999) for _, role in builders)
    uses_time = bool(plan.time.grain) or plan.time.window.kind != "all" or compare is not None or any(
        t for _, _, t in wanted) or named_gaps
    groups: list[Group] = []
    group_names: dict[str, str] = {}
    if plan.time.grain:
        groups.append(Group("period", "Period", "period", plan.time.grain))
        group_names["__period__"] = "period"
        out_names.add("period")
    identity_names: dict[str, tuple[str, str]] = {}     # slug -> (output name, column key) of a member's code
    for slug, attribute, time_attr in wanted:
        if time_attr:
            name = _output_name(time_attr, out_names)
            groups.append(Group(name, time_attr.replace("_", " ").capitalize(), "time"))
        else:
            assert attribute is not None
            name = _output_name(attribute.slug.replace(".", "_"), out_names)
            groups.append(Group(name, attribute.business_name, "attribute", attribute=attribute.slug))
            identity = _member_identity(model, attribute)
            if identity is not None:
                code_name = _output_name(f"{name}_code", out_names)
                groups.append(Group(code_name, f"{attribute.business_name} code", "member_code",
                                    attribute=attribute.slug))
                identity_names[slug] = (code_name, identity)
        group_names[slug] = name

    parts: list[Part] = []
    measures_out: list[OutMeasure] = []
    for (builder, role), items in zip(builders, by_part.values()):
        part = builder.part
        if role is not None and (uses_time or part.snapshot):
            part.date = builder.date(role)
            if part.date.mode in ("day_calendar", "month_calendar") and not uses_time and not part.snapshot:
                part.date = None
        elif uses_time and role is None:
            raise ResolveError("unsupported", f"{model.tables[part.table].business_name} has no date")
        for m, expr, table, label, fmt in items:
            name = _output_name(m.slug if m else label, out_names)
            out = OutMeasure(name=name, label=label, expr=expr, format=fmt, measure=m,
                             semi=bool(m and m.additivity == "semi_additive"))
            part.measures.append(out)
            measures_out.append(out)
        for slug, attribute, time_attr in wanted:
            name = group_names[slug]
            if time_attr:
                if part.date is None:
                    raise ResolveError("unsupported", f"{model.tables[part.table].business_name} has no date")
                part.groups.append(PartGroup(name, part.date.alias, None, "time", time_attr=time_attr, date=part.date))
                continue
            assert attribute is not None
            column = model.columns[attribute.column]
            through, link_role = _via(model, plan, slug)
            alias = _reach(builder, column.table, slug, through, link_role, attribute.business_name)
            part.groups.append(PartGroup(name, alias, column.key, "attribute"))
            if slug in identity_names:
                # Names repeat: two members called the same must stay two rows.
                code_name, code_column = identity_names[slug]
                part.groups.append(PartGroup(code_name, alias, code_column, "attribute"))
            if plan.limit and slug != unit_slug:
                # A top n ranks real members; rows with none are not a member (a
                # breakdown keeps them as one empty group, so it adds up to the total).
                part.preds.append(Pred(alias, column.key, "not_null", []))
                if _can_lack(model, builder.paths_used.get(column.table), column):
                    notes.append(f"Rows with no {attribute.business_name.lower()} are left out of the ranking.")
        if plan.time.grain:
            if part.date is None:
                raise ResolveError("unsupported", f"{model.tables[part.table].business_name} has no date")
            part.groups.insert(0, PartGroup("period", part.date.alias, None, "period", grain=plan.time.grain,
                                            date=part.date))
        if part.date is not None and (window.start or window.end or compare is not None or plan.time.grain):
            rng = window
            if compare is not None:
                rng = Range(min(x for x in (window.start, compare.start) if x) if window.start or compare.start else None,
                            max(x for x in (window.end, compare.end) if x) if window.end or compare.end else None)
            part.date_ranges.append((part.date, rng))
        if role is not None and role.whole_year_share > 0 and not part.date_ranges:
            # A row keyed to month 00 holds a whole year: never added to its months, even over all time.
            part.date = part.date or builder.date(role)
            part.date_ranges.append((part.date, Range(None, None)))
        if role is not None and role.whole_year_share > 0:
            notes.append(f"{model.tables[part.table].business_name}: rows for a whole year (month 00) are left "
                         "out; only months are counted.")
        parts.append(part)

    # Filters: attributes, other dates, totals.
    having: list[Pred] = []
    for f in plan.filters:
        found = find_slug(model, f.field)
        if not found:
            raise ResolveError("unknown", f"nothing to filter on called {f.field}", _closest(f.field, _slugs(model, "group") + _slugs(model, "date") + _slugs(model, "measure")))
        kind, obj = found
        if kind == "measure":
            m = obj
            assert isinstance(m, Measure)
            target = next((o for o in measures_out if o.measure is m), None)
            if target is None:
                raise ResolveError("unsupported", f"a filter on {m.business_name} needs it in the answer")
            having.append(Pred("", target.name, f.op, list(f.values), f"{m.business_name} {_op_words(f)}"))
            continue
        # Each measure table takes the filters it can reach; in a question about two
        # tables, a filter on one table's own column limits that table's measures only.
        took: list[Part] = []
        missed: list[Part] = []
        unreachable: ResolveError | None = None
        for builder, _ in builders:
            part = builder.part
            if kind == "date":
                filter_role = obj
                assert isinstance(filter_role, DateRole)
                if filter_role.table != part.table:
                    missed.append(part)
                    continue
                use = builder.date(filter_role)
                try:
                    values = [dt.date.fromisoformat(str(v)) for v in f.values]
                except ValueError:
                    raise ResolveError("unsupported", f"{filter_role.name} needs dates, not {f.values}") from None
                part.date_ranges.append((use, _date_filter_range(f, values)))
                took.append(part)
                continue
            attribute = _entity_label(model, f.field) if kind == "entity" else obj
            assert isinstance(attribute, Attribute)
            column = model.columns[attribute.column]
            through, link_role = _via(model, plan, f.field)
            try:
                alias = _reach(builder, column.table, f.field, through, link_role, attribute.business_name)
            except ResolveError as exc:
                if exc.kind != "unsupported":
                    raise
                unreachable = exc
                missed.append(part)
                continue
            part.preds.append(Pred(alias, column.key, f.op, list(f.values),
                                   f"{attribute.business_name} {_op_words(f)}"))
            took.append(part)
        if not took:
            if unreachable is not None:
                raise unreachable
            raise ResolveError("unsupported", f"{getattr(obj, 'name', f.field)} is not a date of what was asked")
        label = getattr(obj, "business_name", "") or getattr(obj, "name", "") or f.field
        if missed:
            limited = ", ".join(o.label for p in took for o in p.measures)
            notes.append(f"{label} {_op_words(f)}: this limits {limited} only.")
        else:
            notes.append(f"{label} {_op_words(f)}.")

    # Admin-approved default filters on the measure tables.
    for builder, _ in builders:
        owner = model.tables[builder.part.table]
        for df in owner.default_filters:
            builder.part.preds.append(Pred(builder.part.alias, df.column, df.op, list(df.values)))
            notes.append(f"{owner.business_name}: {model.columns[df.column].business_name} "
                         f"{df.op.replace('_', ' ')} {', '.join(map(str, df.values))} (a default filter).")

    # How it was answered, in words.
    for builder, role in builders:
        part = builder.part
        if part.date is not None and role is not None:
            notes.insert(0, f"{', '.join(o.label for o in part.measures)} by {role.name.lower()}.")
        # A table joined only to keep units apart is not one the reader asked about: no path note.
        unit_column = model.attributes[unit_slug].column if unit_slug is not None else None
        asked = {model.columns[g.column].table for g in part.groups if g.column and g.column != unit_column} | {
            model.columns[p.column].table for p in part.preds if p.column in model.columns}
        for table, path in builder.paths_used.items():
            if unit_column and table == model.columns[unit_column].table and table not in asked:
                continue
            alternatives = [p for p in P.all_paths(model, part.table, table) if p.joins != path.joins]
            if alternatives:
                reached = model.tables[table].business_name
                how = "each row's own" if len(path.joins) == 1 else f"through {path.describe(model)}"
                routes = sorted({model.tables[p.joins[0].to_table].business_name.lower() if len(p.joins) > 1 else
                                 (p.joins[0].role or reached).lower() for p in alternatives[:3]})
                notes.append(f"{reached}: {how} (it could also come through the {' or the '.join(routes)}).")
        for j in model.joins.values():
            if j.from_table == part.table and j.trust == "proposed" and any(
                    step.key == j.key for p in builder.paths_used.values() for step in p.joins):
                notes.append(f"{1 - j.match_rate:.0%} of {model.tables[part.table].business_name.lower()} rows have "
                             f"no matching {model.tables[j.to_table].business_name.lower()}: they show as Unknown.")
        missing = 1 - part.date.role.coverage if part.date else 0.0    # empty or a placeholder (-1, 9999-12-31)
        if part.date and missing > 0.001 and uses_time:
            notes.append(f"Rows with no {part.date.role.name.lower()} ({missing:.0%}) are left out.")
        if part.snapshot:
            notes.append(f"{', '.join(o.label for o in part.measures)}: taken at the last "
                         f"{'snapshot' if not plan.time.grain else 'snapshot of each period'}, not added up over time.")
        grouped = {g.column for g in part.groups if g.column}
        filtered = {p.column for p in part.preds}
        for o in part.measures:
            column_key = getattr(o.measure.expr, "column", None) if o.measure else None
            for flag in model.quality:
                if flag.kind == "unit_mix" and flag.object == column_key and not (
                        o.measure and o.measure.unit_column and o.measure.unit_column in grouped | filtered):
                    units = ", ".join(str(u) for u in flag.data.get("units", [])[:4])
                    notes.append(f"{o.label} adds up different units{f' ({units})' if units else ''}: "
                                 "group by unit of measure to keep them apart.")
                elif flag.kind == "constant" and flag.object == column_key and o.measure \
                        and getattr(o.measure.expr, "agg", None) in ("avg", "min", "max"):
                    # Summed, a constant still counts rows (a headcount of 1 per row); averaged, it says nothing.
                    notes.append(f"{o.label} is {flag.data.get('value')} on every row: it tells nothing apart.")
                elif flag.kind == "outlier_period" and flag.object == column_key \
                        and (part.date is None or part.date.role.is_default):
                    month = _month_of(str(flag.data.get("period", "")))
                    if month and _overlaps(month, window, compare):
                        notes.append(f"{month:%b %Y} stands out: {o.label.lower()} of "
                                     f"{float(flag.data.get('value') or 0):,.0f} against a typical "
                                     f"{float(flag.data.get('typical') or 0):,.0f} a month. Worth checking before "
                                     "relying on it.")
        for flag in model.quality:
            if flag.kind == "listed_vs_active" and flag.object == part.table:    # a list's members, counted
                via = model.columns.get(str(flag.data.get("via")))
                listed = plural(model.tables[part.table].business_name.lower())
                used_in = plural(model.tables[via.table].business_name.lower()) if via else "the data"
                notes.append(f"{int(flag.data.get('listed') or 0):,} {listed} are listed; "
                             f"{int(flag.data.get('used') or 0):,} appear in {used_in}.")
        owner_columns = {c.key for c in model.columns.values() if c.table == part.table}
        for flag in model.quality:
            if flag.kind == "status_column" and flag.object in owner_columns and flag.object not in filtered \
                    and not model.tables[part.table].default_filters:
                cancel_like = [str(v) for v in flag.data.get("cancel_like", [])]
                if cancel_like:
                    notes.append(f"Includes rows whose {model.columns[flag.object].business_name.lower()} is "
                                 f"{' or '.join(cancel_like)}; an admin can leave them out by default.")

    # Partial periods of a series, and the periods it should hold (between the first and last data).
    partial: list[dt.date] = []
    expected: list[dt.date] = []
    if plan.time.grain and first_role:
        bounded = Range(window.start or first, window.end or ((last + dt.timedelta(days=1)) if last else None))
        if bounded.start and bounded.end:
            starts = periods(bounded, plan.time.grain, fiscal_start=fiscal_start)
            partial = partial_periods(starts, plan.time.grain, first_data=first, last_data=last, rng=window,
                                      today=ctx.today,
                                      fiscal_start=fiscal_start)
            step = {"fiscal_month": "month", "fiscal_quarter": "quarter", "fiscal_year": "year"}.get(
                plan.time.grain, plan.time.grain)
            expected = [s for s in starts if (first is None or add_units(s, step, 1) > first)
                        and (last is None or s <= last)]

    sort = []
    for s in plan.sort:
        name = _sort_name(s.by, measures_out, group_names, model)
        if not name:
            # Dropped, the order fell back to the periods: "the month with the highest discount rate"
            # answered with the first month, as if it were the highest.
            raise ResolveError("unknown", f"nothing to sort by called {s.by}",
                               [*(o.measure.slug if o.measure is not None else o.label for o in measures_out),
                                *group_names, "change", "pct_change", "share", "period"])
        sort.append((name, s.desc))
    if not sort:
        if plan.time.grain:
            sort = [("period", False)]
        elif intent == "compare" and measures_out:
            sort = [(f"{measures_out[0].name}_change", True)]
        elif intent in ("rank", "breakdown", "share") and measures_out and wanted:
            sort = [(measures_out[0].name, True)]
    limit = plan.limit
    unit_order: list[str] = []
    if unit_slug is not None:
        unit = model.attributes[unit_slug]
        notes.append(f"{', '.join(o.label for o in measures_out if o.measure and o.measure.unit_column)} is shown "
                     f"per {unit.business_name.lower()}: different units are not added together.")
        profile = model.columns[unit.column].profile
        unit_order = [str(t.value) for t in sorted((profile.top or []) if profile else [], key=lambda t: -t.count)
                      if t.value not in (None, "")]
    return Logical(intent=intent, parts=parts, groups=groups, measures=measures_out, window=window, compare=compare,
                   sort=sort, limit=limit, share=intent == "share", having=having, notes=_unique(notes),
                   partial=partial, fiscal_start=fiscal_start, max_rows=ctx.max_rows,
                   unit_group=group_names.get(unit_slug) if unit_slug else None, unit_order=unit_order,
                   expected=expected, data_last=last if plan.time.grain else None)


_SPLIT_INTENTS = {"value", "breakdown", "rank", "trend", "compare", "count"}


def _unit_split(plan: Plan, model: SemanticModel, ctx: Context, intent: str, chosen: list,
                wanted: list[tuple[str, Attribute | None, str | None]]) -> str | None:
    """The unit to keep a quantity apart by, when its rows hold different units and the answer would add
    them up (issue E4): "1,200 EA and 300 FT", never "1,500". None when the question already groups or
    filters by the unit, when measures come from more than one table, or when the unit cannot be reached
    plainly (another table the reader may not use, or two ways to get there)."""
    if not ctx.split_units or intent not in _SPLIT_INTENTS or len({item[2] for item in chosen}) != 1:
        return None
    for m, *_ in chosen:
        if m is None or not m.unit_column:
            continue
        column = getattr(m.expr, "column", None)
        if not any(f.kind == "unit_mix" and f.object == column for f in model.quality):
            continue
        unit_column = model.columns[m.unit_column]
        attribute = next((a for a in model.attributes.values() if a.column == m.unit_column), None)
        if attribute is None or unit_column.hidden:
            return None
        named = {a.column for _, a, _ in wanted if a is not None}
        filtered = {model.attributes[f.field].column for f in plan.filters if f.field in model.attributes}
        if m.unit_column in named | filtered:
            return None
        if unit_column.table != m.table:
            if ctx.allowed_tables is not None and unit_column.table not in ctx.allowed_tables:
                return None
            try:
                if P.best_path(model, m.table, unit_column.table) is None:
                    return None
            except P.Ambiguous:
                return None
        return attribute.slug
    return None


def _via(model: SemanticModel, plan: Plan, slug: str) -> tuple[str | None, str | None]:
    """How the plan says ``slug`` is reached: through an entity's table ("the customer's home store"),
    or by the role of a link ("Bill to customer", "Ship to customer")."""
    named = plan.via.get(slug)
    if not named:
        return None, None
    found = find_slug(model, named)
    if found and isinstance(found[1], Entity):
        return found[1].table, None
    return None, named.removeprefix("role:").strip()


def _reach(builder: _PartBuilder, table: str, slug: str, through: str | None, role: str | None, label: str) -> str:
    try:
        return builder.reach(table, through=through, role=role, label=label)
    except ResolveError as exc:
        if exc.kind == "ambiguous" and exc.field is None:
            exc.field = slug                 # the reply names a link for this slug
        raise


def _like_for_like(window: Range, compare: Range, last: dt.date | None, kind: str) -> tuple[Range, Range, str]:
    """A current period the data covers only in part, set against the same part of the other one.

    "This month vs last month" on the 10th compares the 1st to the 10th of each
    month; "this year vs last year" compares up to the same day of each year. A
    whole period set against a part of one is a fall that never happened.
    """
    if last is None or None in (window.start, window.end, compare.start, compare.end):
        return window, compare, ""
    assert window.start and window.end and compare.start and compare.end
    covered_to = last + dt.timedelta(days=1)
    if not window.start < covered_to < window.end:
        return window, compare, ""
    if kind == "same_period_last_year":
        prior_end = year_back(covered_to)
    else:
        prior_end = compare.start + (covered_to - window.start)
    prior_end = min(prior_end, compare.end)
    days = (covered_to - window.start).days
    note = (f"The data runs to {last.day} {last:%b %Y}, so both periods are compared over their first "
            f"{days} day{'s' if days != 1 else ''}.")
    return Range(window.start, covered_to, window.notes), Range(compare.start, prior_end, compare.notes), note


def measure_dates(plan: Plan, model: SemanticModel) -> tuple[DateRole | None, dt.date | None, dt.date | None]:
    """The date the plan's first measure is counted by, and the first and last day its data covers."""
    measure: Measure | None = None
    for slug in [*plan.measures, *[s for d in plan.derived for s in d.measures]]:
        found = find_slug(model, slug)
        if found and found[0] == "measure":
            measure = found[1]  # type: ignore[assignment]
            break
    if measure is None:
        return None, None, None
    role = _date_role_for(model, plan, measure, measure.table)
    if role is None:
        return None, None, None
    first, last = _first_last(role)
    return role, first, last


def _fallback_measure(model: SemanticModel, table: str) -> Measure:
    return next(m for m in model.measures.values() if m.table == table)


def _members(plan: Plan, model: SemanticModel, ctx: Context, wanted: list, aliases: _Aliases, intent: str) -> Logical:
    """A list of members (no measure): the distinct values of the grouping, from its own table."""
    slug, attribute, time_attr = wanted[0]
    if time_attr or attribute is None:
        raise ResolveError("unsupported", "listing periods needs a measure")
    column = model.columns[attribute.column]
    builder = _PartBuilder(model, column.table, aliases, ctx)
    builder._allowed(column.table)
    part = builder.part
    part.distinct_only = True
    out_names: set[str] = set()
    groups = []
    for slug_, attr, _ in wanted:
        col = model.columns[attr.column]
        alias = builder.reach(col.table, label=attr.business_name)
        name = _output_name(attr.slug.replace(".", "_"), out_names)
        part.groups.append(PartGroup(name, alias, col.key, "attribute"))
        groups.append(Group(name, attr.business_name, "attribute", attribute=attr.slug))
    for f in plan.filters:
        found = find_slug(model, f.field)
        if not found or found[0] not in ("attribute", "entity"):
            raise ResolveError("unsupported", f"listing cannot filter on {f.field}")
        attr = _entity_label(model, f.field) if found[0] == "entity" else found[1]
        assert isinstance(attr, Attribute)
        col = model.columns[attr.column]
        part.preds.append(Pred(builder.reach(col.table), col.key, f.op, list(f.values)))
    return Logical(intent="list", parts=[part], groups=groups, measures=[], window=Range(None, None),
                   sort=[(groups[0].name, False)], limit=plan.limit, max_rows=ctx.max_rows,
                   notes=[f"{model.tables[column.table].business_name}: {attribute.business_name.lower()} as listed."])


def _date_filter_range(f: Filter, values: list[dt.date]) -> Range:
    one = dt.timedelta(days=1)
    if f.op == "between" and len(values) == 2:
        return Range(min(values), max(values) + one)
    if f.op in ("eq", "in") and values:
        return Range(min(values), max(values) + one)
    if f.op in ("gte", "gt") and values:
        return Range(values[0] + (one if f.op == "gt" else dt.timedelta(0)), None)
    if f.op in ("lte", "lt") and values:
        return Range(None, values[0] + (one if f.op == "lte" else dt.timedelta(0)))
    raise ResolveError("unsupported", f"a date filter with {f.op} is not supported")


def _op_words(f: Filter) -> str:
    values = ", ".join(str(v) for v in f.values)
    return {"eq": f"is {values}", "in": f"is {values}", "ne": f"is not {values}", "not_in": f"is not {values}",
            "gt": f"above {values}", "gte": f"at least {values}", "lt": f"below {values}", "lte": f"at most {values}",
            "between": f"between {' and '.join(str(v) for v in f.values)}", "contains": f"contains {values}",
            "starts_with": f"starts with {values}", "is_null": "is empty", "not_null": "is filled"}.get(f.op, f.op)


def _sort_name(by: str, measures: list[OutMeasure], groups: dict[str, str], model: SemanticModel) -> str:
    if by in ("change", "pct_change", "share") and measures:
        return f"{measures[0].name}_{by}"
    if by == "period":
        return "period"
    plain = " ".join(re.findall(r"[a-z0-9]+", by.casefold()))
    for o in measures:
        # A measure by its slug; one worked out for the question by the name the plan gave it
        # ("Discount rate"), however it is written.
        if o.measure is not None and o.measure.slug == by or o.name == by or \
                " ".join(re.findall(r"[a-z0-9]+", o.label.casefold())) == plain:
            return o.name
    if by in groups:
        return groups[by]
    return ""


def _unique(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _month_of(period: str) -> dt.date | None:
    """'2025-11' as the first day of that month."""
    try:
        return dt.date(int(period[:4]), int(period[5:7]), 1)
    except ValueError:
        return None


def _overlaps(month: dt.date, *ranges: Range | None) -> bool:
    """Does any range (open-ended where unbounded) cover part of ``month``?"""
    end = add_units(month, "month", 1)
    return any(r is not None and (r.start is None or r.start < end) and (r.end is None or r.end > month)
               for r in ranges)
