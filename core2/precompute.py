"""Copies of a metric kept on this server, so the questions they answer need no trip to the database.

As Databricks' materialized metric views, Looker's aggregate tables and Cube's pre-aggregations do: an admin keeps
a metric by some of its fields and by day or by month (Data > Precomputed copies). Solviz makes the copy on the
workspace's own connection, keeps it on this server (a DuckDB file), and makes it again on the copy's schedule.
The connection to the client's database stays read-only: nothing is ever written there.

A question is answered from a copy only when the copy gives exactly the rows the database would:

    the same rows       the question counts the copy's table, through links the copy was made with, under the same
                        always-applied conditions ("cancelled orders left out"), and groups and filters only by the
                        copy's fields and its date
    sums that add up    a sum, a count, a smallest and a largest add up again; an average is its sum over its count;
                        a count of different values only when the question asks for exactly the copy's fields and
                        its day or month (one customer in two months is one customer in the quarter, not two)
    whole periods       a copy by month answers months, quarters and years, never "the last 90 days"
    nothing else        a question needing anything more (a field the copy does not keep, the rows left out by
                        default, a shape worked out over each member) is asked of the database, as before

The copy's query is the question's own logical query moved onto the copy, written by the same compiler; its
columns must come out the same as the database query's, or the copy is not used. Only the fetch changes:
QueryBot's governance (allowed tables, masking, aggregate-only fields, the decision log) runs on the database
query as it always does, and a reader with a row rule on its tables is always answered by the database
(core/compliance/governed_query.py). The answer says the copy was used and when it was made.

A copy is used while it is fresh (made within twice its schedule) and while the metric, its fields and its links
are as they were when it was made (its fingerprint); otherwise the database answers and the copy is made again.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from sqlglot import exp

from core2.compile import compiler as C
from core2.model import formula
from core2.model.schema import AggExpr, Column, MeasureExpr, OpExpr, SemanticModel, SqlExpr, Table
from core2.plan.ir import Plan, TimeSpec, Window
from core2.resolve.resolver import (Context, DateUse, Logical, OutMeasure, Part, PartGroup, Pred, ResolveError,
                                    TimePred, resolve)
from core2.resolve.time import Range
from core2.warehouse import dialect as D

log = logging.getLogger(__name__)

GRAINS = ("day", "month")
EVERY_HOURS = (1, 3, 6, 12, 24)
TABLE = "kept_rows"              # the copy's table in its file
_KEY = "solviz_copy"             # its table key in the model a question is moved onto
_PERIOD = "p"
_DAY_TIMES = frozenset({"day_of_week", "month_of_year", "quarter_of_year", "is_weekend"})
_MONTH_TIMES = frozenset({"month_of_year", "quarter_of_year"})
_MONTH_GRAINS = frozenset({"month", "quarter", "year", "fiscal_month", "fiscal_quarter", "fiscal_year"})
_EXACT_GRAINS = {"day": frozenset({"day"}), "month": frozenset({"month", "fiscal_month"})}
_BYTE_ORDER = frozenset({"duckdb", "snowflake"})     # warehouses that put names in order as the copy does
_DUCK_TYPES = {"text": "VARCHAR", "integer": "BIGINT", "decimal": "DECIMAL(38, 10)", "float": "DOUBLE",
               "date": "DATE", "timestamp": "TIMESTAMP", "boolean": "BOOLEAN"}


class CopyError(ValueError):
    """Why a copy cannot be kept as asked, in words for the admin."""


def max_rows() -> int:
    """The most rows a copy may hold: QUERYBOT_COPY_MAX_ROWS (500,000 when unset)."""
    try:
        return max(1000, int(os.getenv("QUERYBOT_COPY_MAX_ROWS") or 500_000))
    except ValueError:
        return 500_000


def copies_dir() -> Path:
    """Where copies are kept: QUERYBOT_COPIES_DIR, or data/precomputed beside the app's own database (made again
    from the warehouse whenever they are missing: nothing there needs a backup)."""
    folder = Path(os.getenv("QUERYBOT_COPIES_DIR") or "data/precomputed")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ── what a copy holds ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Leaf:
    """One aggregate of the metric as the copy keeps it: its own column, or a sum and a count for an average."""

    expr: AggExpr | SqlExpr
    kind: str                        # sum | count | count_distinct | min | max | avg
    columns: tuple[str, ...]         # the copy's columns: ("c0",), or ("c0", "c1") for an average's sum and count
    data_type: str                   # what the sum or the smallest is of ("" for a count)

    @property
    def key(self) -> str:
        return _leaf_key(self.expr)


@dataclass(frozen=True)
class Dim:
    path: tuple                      # how the copy's rows reach the field's table (_paths)
    column: str                      # the field's column key
    name: str                        # its column in the copy


@dataclass
class Definition:
    metric: str
    table: str
    dims: list[Dim]
    date_role: str
    stored: str                      # how the copy holds the date: date | yyyymmdd | yyyymm
    grain: str                       # day | month
    conditions: frozenset            # the always-applied conditions it was made under (_pred_key)
    joins: frozenset                 # the links it was made through (_paths steps), its date's calendar left out
    calendar: tuple | None           # its date's calendar step, when its date is read through one
    leaves: list[Leaf]
    sql: str                         # the query that makes it, in the warehouse's language
    checks: list[tuple[str, str]]    # (query, why): each must find no row (a link matching two rows for a key)
    columns: list[tuple[str, str]]   # its columns and their core2 data types, in order
    fingerprint: str = ""
    dialect: str = ""                # the warehouse's: SQL Server compares names as its copy must too


def _leaf_key(expr: AggExpr | SqlExpr) -> str:
    return json.dumps(expr.model_dump(mode="json"), sort_keys=True)


def _values_key(values: list) -> str:
    return json.dumps(list(values), sort_keys=True, default=str)


def _paths(part: Part) -> dict[str, tuple] | None:
    """Each alias of ``part`` as the way its rows reach it: one step a link (its table, its column pairs, inner
    or left, its own conditions). None when a step starts from more than one place."""
    steps = {j.alias: j for j in part.joins}
    known: dict[str, tuple] = {part.alias: ()}

    def path(alias: str, seen: frozenset = frozenset()) -> tuple | None:
        if alias in known:
            return known[alias]
        j = steps.get(alias)
        if j is None or alias in seen:
            return None
        lefts = {left for left, _, _ in j.on}
        if len(lefts) != 1:
            return None
        before = path(next(iter(lefts)), seen | {alias})
        if before is None:
            return None
        step = (j.table, tuple(sorted((lc, rc) for _, lc, rc in j.on)), "inner" if j.kind == "inner" else "left",
                tuple(sorted((c, op, _values_key(v)) for c, op, v in j.conds)))
        known[alias] = (*before, step)
        return known[alias]

    for alias in [part.alias, *steps]:
        if path(alias) is None:
            return None
    return known


def _pred_key(paths: dict[str, tuple], p: Pred) -> tuple:
    return (paths.get(p.alias), p.column, p.op, _values_key(p.values))


def _leaves_of(expr: MeasureExpr, name: str) -> list[AggExpr | SqlExpr]:
    if isinstance(expr, AggExpr):
        if expr.agg == "list":
            raise CopyError(f"{name} lists values: a copy keeps figures")
        return [expr]
    if isinstance(expr, SqlExpr):
        _top(expr, name)
        return [expr]
    if isinstance(expr, OpExpr):
        return [leaf for arg in expr.args for leaf in _leaves_of(arg, name)]
    raise CopyError(f"{name} is worked out from other metrics in a way a copy cannot keep")


def _top(expr: SqlExpr, name: str) -> exp.Expression:
    """A formula's one aggregate over each row's arithmetic ("SUM(qty * price)"); anything else is refused."""
    tree = formula.parse_stored(expr.sql)
    if not isinstance(tree, formula.AGGREGATES) or any(isinstance(n, formula.AGGREGATES) for n in tree.this.walk()):
        raise CopyError(f"{name} is a formula a copy cannot keep: one SUM, COUNT, AVG, MIN or MAX of each row's "
                        "arithmetic only")
    if isinstance(tree, (exp.Sum, exp.Avg)) and isinstance(tree.this, exp.Distinct):
        # The different values of each day, added up again, count a value seen on two days twice.
        raise CopyError(f"{name} adds up only the different values: a copy cannot add it up again")
    return tree


def _kind(expr: AggExpr | SqlExpr) -> str:
    if isinstance(expr, AggExpr):
        return expr.agg
    tree = formula.parse_stored(expr.sql)
    if isinstance(tree, exp.Count):
        return "count_distinct" if isinstance(tree.this, exp.Distinct) else "count"
    return {exp.Sum: "sum", exp.Avg: "avg", exp.Min: "min", exp.Max: "max"}[type(tree)]


def _with_top(expr: SqlExpr, top: type) -> SqlExpr:
    """The same formula under another aggregate (an average's sum, or its count)."""
    tree = formula.parse_stored(expr.sql)
    return expr.model_copy(update={"sql": top(this=tree.this.copy()).sql()})


def _data_type(model: SemanticModel, expr: AggExpr | SqlExpr, kind: str) -> str:
    if kind in ("count", "count_distinct"):
        return "integer"
    numbers = ("integer", "decimal", "float")
    if isinstance(expr, AggExpr) and expr.column and expr.column in model.columns:
        dtype = model.columns[expr.column].data_type
        if kind in ("min", "max"):
            return dtype            # the earliest date, the first name: kept as what they are
        return dtype if dtype in numbers else "float"
    if isinstance(expr, SqlExpr) and kind in ("min", "max"):
        types = {model.columns[c].data_type for c in expr.columns if c in model.columns}
        tree = formula.parse_stored(expr.sql)
        if len(types) == 1 and isinstance(tree.this, exp.Column):
            return next(iter(types))
        if not types <= set(numbers):
            raise CopyError("the smallest or largest of a worked-out name or date cannot be kept in a copy")
    return "float"


# ── defining a copy ──────────────────────────────────────────────────────────

def define(model: SemanticModel, *, metric: str, fields: list[str], date: str = "", grain: str = "day",
           today: dt.date, dialect: str, link_repeats: Callable[[Any], bool] | None = None) -> Definition:
    """What a copy of ``metric`` by ``fields`` and by ``grain`` on its date holds, and the query that makes it;
    CopyError when it cannot be kept, saying why."""
    measure = model.measures.get(metric)
    if measure is None or measure.hidden:
        raise CopyError("this metric is not in the data any more")
    name = measure.business_name or measure.slug
    if grain not in GRAINS:
        raise CopyError("a copy is kept by day or by month")
    if measure.additivity == "semi_additive" or measure.time_aggregation:
        raise CopyError(f"{name} is a balance read on a day, never added up over days: a copy cannot keep it")
    role = model.date_roles.get(date or measure.default_date or "")
    if role is None:
        table = model.tables.get(measure.table)
        role = model.date_roles.get(table.default_date or "") if table is not None else None
    if role is None:
        raise CopyError(f"{name} has no date to keep it by")
    for slug in fields:
        attribute = model.attributes.get(slug)
        column = model.columns.get(attribute.column) if attribute is not None else None
        if column is None:
            raise CopyError(f"{slug} is not a field of this data any more")
        label = attribute.business_name or slug
        if column.hidden:
            raise CopyError(f"{label} is hidden from questions: a copy never keeps it")
        if column.sensitivity != "none" or column.personal != "none":
            raise CopyError(f"{label} holds personal or confidential data: a copy never keeps it")
    plan = Plan(kind="query", intent="breakdown" if fields else "trend", measures=[measure.slug],
                group_by=list(fields), time=TimeSpec(date=role.slug or None, grain=grain, window=Window(kind="all")))
    try:
        logical = resolve(plan, model, Context(today=today, split_units=True, link_repeats=link_repeats,
                                               max_rows=max_rows()))
    except ResolveError as exc:
        raise CopyError(str(exc)) from exc
    if len(logical.parts) != 1 or logical.shape is not None or logical.absence is not None \
            or logical.target is not None or logical.people:
        raise CopyError(f"{name} by these fields is worked out over more than one table's rows: a copy keeps one")
    part = logical.parts[0]
    if part.semis or part.day_preds or part.time_preds or part.age_preds or part.segments or part.absent \
            or part.row_level or part.distinct_only or part.snapshot or part.date is None:
        raise CopyError(f"{name} by these fields needs more than adding up rows: a copy cannot keep it")
    paths = _paths(part)
    if paths is None:
        raise CopyError("its links cannot be followed one step at a time")
    use = part.date
    calendar = paths.get(use.alias) if use.alias != part.alias else None
    writer = C._Compiler(logical, model, dialect)
    dims: list[Dim] = []
    selects: list[exp.Expression] = []
    columns: list[tuple[str, str]] = []
    for g in part.groups:
        if g.kind == "period":
            continue
        if g.kind != "attribute" or g.column is None:
            raise CopyError(f"{g.name} is not a field a copy can keep")
        for alias, key in [(g.alias, g.column), *([g.by_key] if g.by_key else [])]:
            dim = Dim(paths[alias], key, f"d{len(dims)}")
            if any(d.path == dim.path and d.column == key for d in dims):
                continue
            dims.append(dim)
            selects.append(writer.col(alias, key).as_(dim.name))
            columns.append((dim.name, model.columns[key].data_type))
    period, stored, dtype = _period(writer, use, grain, dialect)
    selects.append(period.as_(_PERIOD))
    columns.append((_PERIOD, dtype))
    leaves: list[Leaf] = []
    seen: set[str] = set()
    for m in part.measures:
        for expr in _leaves_of(m.expr, name):
            if _leaf_key(expr) in seen:
                continue
            seen.add(_leaf_key(expr))
            kind = _kind(expr)
            at = len(columns)
            if kind == "avg":
                parts_ = _avg_parts(expr)
                leaf = Leaf(expr, kind, (f"c{at}", f"c{at + 1}"), _data_type(model, expr, "sum"))
                for column, made, dt_ in zip(leaf.columns, parts_, ("float", "integer")):
                    selects.append(writer.aggregate(made, part).as_(column))
                    columns.append((column, dt_))
            else:
                leaf = Leaf(expr, kind, (f"c{at}",), _data_type(model, expr, kind))
                if leaf.data_type == "text" and dialect not in _BYTE_ORDER:
                    # The first name of each day, and the first of those again: in the copy's order, which is not
                    # the warehouse's (SQL Server ignores capitals, PostgreSQL follows its language).
                    raise CopyError(f"{name} is the first or last of a name in the database's own order: a copy "
                                    "cannot keep it")
                selects.append(writer.aggregate(expr, part).as_(leaf.columns[0]))
                columns.append((leaf.columns[0], leaf.data_type))
            leaves.append(leaf)
    query = exp.select(*selects).from_(writer.table(part.table, part.alias))
    joins: set[tuple] = set()
    for j in part.joins:
        step = paths[j.alias]
        # Its date's calendar is joined keeping every row: one with no date there is kept with none, so a question
        # with no date (a total of all time) counts it as the database does, and one with a date leaves it out.
        kind = "left" if step == calendar else ("inner" if j.kind == "inner" else "left")
        query = query.join(writer.table(j.table, j.alias), on=writer.on(j), join_type=kind)
        if step != calendar:
            joins.add(step)
    conditions = [writer.predicate(writer.col(p.alias, p.column), p.op, p.values, p.column) for p in part.preds]
    if conditions:
        query = query.where(exp.and_(*conditions))
    query = query.group_by(*[e.this.copy() for e in selects[:len(dims) + 1]])
    checks = [(_repeats(writer, model, j, dialect), f"{model.tables[j.table].business_name or j.table} holds "
               "some of its rows twice for the same key: a copy through it would count them twice")
              for j in part.joins]
    if stored == "yyyymm" and use.mode == "yyyymmdd":
        checks.append((_not_days(writer, part, use, dialect), f"some of {role.name or 'its date'}'s numbers are no "
                       "real day (a 00th day or a 13th month): keep this copy by day"))
    defn = Definition(metric=metric, table=part.table, dims=dims, date_role=role.key, stored=stored, grain=grain,
                      conditions=frozenset(_pred_key(paths, p) for p in part.preds), joins=frozenset(joins),
                      calendar=calendar, leaves=leaves, sql=D.render(query, dialect), checks=checks, columns=columns,
                      dialect=dialect)
    defn.fingerprint = hashlib.sha256(json.dumps({
        "sql": defn.sql, "dims": [dataclasses.astuple(d) for d in dims], "leaves": [l.key for l in leaves],
        "conditions": sorted(map(str, defn.conditions)), "joins": sorted(map(str, defn.joins)),
        "calendar": str(calendar), "stored": stored, "grain": grain, "role": role.key,
        "fiscal": model.settings.fiscal_year_start_month,
        "types": columns}, sort_keys=True, default=str).encode()).hexdigest()
    return defn


def _avg_parts(expr: AggExpr | SqlExpr) -> tuple[AggExpr | SqlExpr, AggExpr | SqlExpr]:
    """An average's sum and count: the same rows' values, the empty ones counted by neither."""
    if isinstance(expr, AggExpr):
        return expr.model_copy(update={"agg": "sum"}), expr.model_copy(update={"agg": "count"})
    return _with_top(expr, exp.Sum), _with_top(expr, exp.Count)


def _period(writer: Any, use: DateUse, grain: str, dialect: str) -> tuple[exp.Expression, str, str]:
    """The copy's date column: each row's date as the question's query reads it, or its month."""
    named = use.role.name or "its date"
    if use.mode == "year_quarter":
        raise CopyError(f"{named} is kept as a year and a quarter: a copy cannot keep it")
    if use.mode in ("month_calendar", "yyyymm"):
        if grain == "day":
            raise CopyError(f"{named} is kept by month: keep the copy by month")
        return writer.stored_date(use), "yyyymm", "integer"
    if use.mode == "yyyymmdd":
        stored = writer.stored_date(use)
        return (stored, "yyyymmdd", "integer") if grain == "day" else (D.floor_div(stored, 100), "yyyymm", "integer")
    return D.period_start(writer.date_value(use), grain, dialect), "date", "date"


def _repeats(writer: Any, model: SemanticModel, j: Any, dialect: str) -> str:
    """A row when ``j``'s table holds more than one row for a key it is reached by (under the link's conditions)."""
    keys = [writer.col("t", rc) for _, _, rc in j.on]
    kept = [exp.not_(exp.Is(this=k.copy(), expression=exp.Null())) for k in keys]
    kept += [writer.predicate(writer.col("t", c), op, list(v), c) for c, op, v in j.conds]
    return (exp.select(*[k.copy() for k in keys]).from_(writer.table(j.table, "t")).where(exp.and_(*kept))
            .group_by(*[k.copy() for k in keys])
            .having(exp.GT(this=exp.Count(this=exp.Star()), expression=exp.Literal.number(1)))).sql(dialect=dialect)


def _not_days(writer: Any, part: Part, use: DateUse, dialect: str) -> str:
    """A row when a date number between the real bounds is no real day of a real month (20250100, 20251301)."""
    n = writer.stored_date(use)
    month = D.mod(D.floor_div(n.copy(), 100), 100)
    bad = exp.or_(exp.EQ(this=D.mod(n.copy(), 100), expression=exp.Literal.number(0)),
                  exp.not_(exp.Between(this=month, low=exp.Literal.number(1), high=exp.Literal.number(12))))
    inside = exp.Between(this=n.copy(), low=exp.Literal.number(19010101), high=exp.Literal.number(90000101))
    query = exp.select(exp.Literal.number(1)).from_(writer.table(part.table, part.alias))
    if use.alias != part.alias:
        j = next(j for j in part.joins if j.alias == use.alias)
        query = query.join(writer.table(j.table, j.alias), on=writer.on(j), join_type="inner")
    return query.where(exp.and_(inside, bad)).sql(dialect=dialect)


# ── answering from a copy ────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Kept:
    """A copy's column as a question's query reads it: its leaf, re-added from the copy's rows."""

    leaf: Leaf
    exact: bool

    # read by the compiler as a measure's expression (core2.compile.compiler._Compiler.aggregate)


class _OnTheCopy(C._Compiler):
    """The compiler, adding up a copy's columns again where the database query adds up rows."""

    as_compared = False      # a copy of SQL Server's data: its names compared ignoring capitals and trailing spaces

    def predicate(self, column: exp.Expression, op: str, values: list, column_key: str | None = None) -> exp.Expr:
        stored = self.model.columns.get(column_key or "")
        if self.as_compared and stored is not None and stored.data_type == "text" \
                and op in ("eq", "in", "ne", "not_in") and all(isinstance(v, str) for v in values):
            # 'EAST ' finds East, as SQL Server does (the copy holds one spelling of each: kept_as_compared).
            folded = exp.Upper(this=exp.Anonymous(this="RTRIM", expressions=[column]))
            return super().predicate(folded, op, [v.rstrip().upper() for v in values])
        return super().predicate(column, op, values, column_key)

    def aggregate(self, expr: Any, part: Part, condition: exp.Expr | None = None) -> exp.Expression:
        if not isinstance(expr, _Kept):
            return super().aggregate(expr, part, condition)

        def kept(column: str) -> exp.Expression:
            value: exp.Expression = self.col(part.alias, f"{_KEY}.{column}")
            return exp.Case(ifs=[exp.If(this=condition.copy(), true=value)]) if condition is not None else value

        leaf = expr.leaf
        if leaf.kind == "sum":
            return D.total(kept(leaf.columns[0]), leaf.data_type, self.d)
        if leaf.kind in ("count", "count_distinct"):
            # A count of no rows is 0 in the database, never empty.
            return exp.Coalesce(this=exp.Sum(this=kept(leaf.columns[0])), expressions=[exp.Literal.number(0)])
        if leaf.kind == "min":
            return exp.Min(this=kept(leaf.columns[0]))
        if leaf.kind == "max":
            return exp.Max(this=kept(leaf.columns[0]))
        total, count = (exp.Sum(this=kept(c)) for c in leaf.columns)
        return D.div(exp.Cast(this=total, to=exp.DataType.build("DOUBLE")),
                     exp.Nullif(this=count, expression=exp.Literal.number(0)))


def _aligned(rng: Range | None, grain: str) -> bool:
    if rng is None or grain == "day":
        return True
    return all(d is None or d.day == 1 for d in (rng.start, rng.end))


def _same_date(use: DateUse | None, role: str) -> bool:
    return use is not None and use.role.key == role


def moved(defn: Definition, logical: Logical, model: SemanticModel) -> Logical | None:
    """``logical`` moved onto the copy, when the copy holds exactly its rows; None when it does not."""
    if len(logical.parts) != 1 or logical.shape is not None or logical.absence is not None \
            or logical.target is not None or logical.people:
        return None
    part = logical.parts[0]
    if part.table != defn.table or part.semis or part.day_preds or part.age_preds or part.segments \
            or part.absent or part.row_level or part.distinct_only or part.snapshot:
        return None
    paths = _paths(part)
    if paths is None:
        return None
    use = part.date
    uses = [u for u in [use, *(u for u, _ in part.date_ranges), *(t.date for t in part.time_preds),
                        *(g.date for g in part.groups if g.kind in ("period", "time"))] if u is not None]
    if any(not _same_date(u, defn.date_role) or paths.get(u.alias, ()) not in ((), defn.calendar) for u in uses):
        return None
    dated = {u.alias for u in uses}
    # The same rows: through links the copy was made with (its calendar aside), under the same conditions.
    for j in part.joins:
        step = paths[j.alias]
        if step not in defn.joins and not (j.alias in dated and step == defn.calendar):
            return None
    if any(step[-1][2] == "inner" for step in defn.joins) and \
            not {s for s in defn.joins if s[-1][2] == "inner"} <= set(paths.values()):
        return None
    dims = {(d.path, d.column): d for d in defn.dims}
    copy_use = DateUse(use.role, part.alias, defn.stored, f"{_KEY}.{_PERIOD}") if use is not None else None

    def on_copy(u: DateUse) -> DateUse:
        """A use of the date on the copy's period: the question's own date stays the one its window is on; any other
        (a filter on the same date, "ordered before March") stays another, kept to its own range."""
        return copy_use if u is use and copy_use is not None else DateUse(u.role, part.alias, defn.stored,
                                                                           f"{_KEY}.{_PERIOD}")

    as_compared = defn.dialect == "tsql"     # SQL Server: names compared ignoring capitals and trailing spaces
    preds: list[Pred] = []
    held: set[tuple] = set()
    for p in part.preds:
        key = _pred_key(paths, p)
        if key in defn.conditions:
            held.add(key)
            continue
        dim = dims.get((paths.get(p.alias), p.column))
        if dim is None:
            return None
        if as_compared and p.op in ("gt", "gte", "lt", "lte", "between") and \
                model.columns[dim.column].data_type == "text":
            return None      # names in SQL Server's order, which is not the copy's
        preds.append(Pred(part.alias, f"{_KEY}.{dim.name}", p.op, list(p.values), p.text))
    if held != set(defn.conditions):
        return None          # the question counts rows the copy left out ("including cancelled orders")
    times = _DAY_TIMES if defn.grain == "day" else _MONTH_TIMES
    time_preds: list[TimePred] = []
    for t in part.time_preds:
        if not _same_date(t.date, defn.date_role) or t.name not in times:
            return None
        time_preds.append(dataclasses.replace(t, date=on_copy(t.date)))
    groups: list[PartGroup] = []
    grouped: set[tuple] = set()
    exact = True
    for g in part.groups:
        if g.kind == "period":
            if not _same_date(g.date, defn.date_role) or g.grain is None or \
                    (defn.grain == "month" and g.grain not in _MONTH_GRAINS):
                return None
            exact = exact and g.grain in _EXACT_GRAINS[defn.grain]
            grouped.add(("period",))
            groups.append(dataclasses.replace(g, alias=part.alias, date=on_copy(g.date)))
        elif g.kind == "time":
            if not _same_date(g.date, defn.date_role) or g.time_attr not in times:
                return None
            exact = False
            groups.append(dataclasses.replace(g, alias=part.alias, date=on_copy(g.date)))
        elif g.kind in ("attribute", "band") and g.column is not None and g.days is None and g.age_on is None:
            dim = dims.get((paths.get(g.alias), g.column))
            if dim is None:
                return None
            by_key = None
            if g.by_key is not None:
                kd = dims.get((paths.get(g.by_key[0]), g.by_key[1]))
                if kd is None:
                    return None
                by_key = (part.alias, f"{_KEY}.{kd.name}")
                grouped.add((kd.path, kd.column))
            exact = exact and g.kind == "attribute"
            grouped.add((dim.path, dim.column))
            groups.append(dataclasses.replace(g, alias=part.alias, column=f"{_KEY}.{dim.name}", by_key=by_key))
        else:
            return None
    # Exactly the copy's own rows, one per group: a count of different values can be read as it is.
    exact = exact and ("period",) in grouped and grouped >= set(dims)
    ranges = [r for _, r in part.date_ranges]
    if not all(_aligned(r, defn.grain) for r in [*ranges, logical.window, logical.compare]):
        return None
    # Names put in order ("regions A to Z"): the order is the database's, which is the copy's only where names are
    # compared byte by byte (SQL Server ignores capitals; PostgreSQL follows its locale).
    texts = {g.name for g in part.groups if g.column in model.columns and model.columns[g.column].data_type == "text"}
    if defn.dialect not in _BYTE_ORDER and any(name in texts for name, _ in logical.sort):
        return None
    leaves = {leaf.key: leaf for leaf in defn.leaves}

    def onto(expr: MeasureExpr) -> Any:
        if isinstance(expr, (AggExpr, SqlExpr)):
            leaf = leaves.get(_leaf_key(expr))
            if leaf is None or (leaf.kind == "count_distinct" and not exact):
                raise LookupError
            return _Kept(leaf, exact)
        if isinstance(expr, OpExpr):
            return OpExpr.model_construct(op=expr.op, args=[onto(a) for a in expr.args], scale=expr.scale)
        raise LookupError

    measures: list[OutMeasure] = []
    for m in part.measures:
        if m.overlaps or m.semi or not isinstance(m.expr, (AggExpr, SqlExpr, OpExpr)):
            return None
        try:
            measures.append(dataclasses.replace(m, expr=onto(m.expr)))
        except LookupError:
            return None
    copied = Part(table=_KEY, alias=part.alias, measures=measures, groups=groups, preds=preds,
                  time_preds=time_preds, date=copy_use,
                  date_ranges=[(on_copy(u), r) for u, r in part.date_ranges])
    by_name = {m.name: m for m in measures}
    return dataclasses.replace(logical, parts=[copied],
                               measures=[by_name.get(m.name, m) for m in logical.measures])


def copy_model(defn: Definition, model: SemanticModel) -> SemanticModel:
    """The model a question moved onto the copy is written against: the copy's one table, each field as the
    workspace defines it (its names for codes, its type), and its date and figures."""
    columns: dict[str, Column] = {}
    for d in defn.dims:
        original = model.columns[d.column]
        columns[f"{_KEY}.{d.name}"] = original.model_copy(update={
            "key": f"{_KEY}.{d.name}", "table": _KEY, "name": d.name, "parts": [], "hidden": False,
            "raw_type": original.data_type, "profile": None})
    for name, dtype in defn.columns:
        if name.startswith("d"):
            continue
        columns[f"{_KEY}.{name}"] = Column(key=f"{_KEY}.{name}", table=_KEY, name=name, data_type=dtype,
                                           raw_type=dtype)
    table = Table(key=_KEY, name=TABLE, columns=list(columns))
    return SemanticModel(client_id=model.client_id, version=model.version, settings=model.settings,
                         tables={_KEY: table}, columns=columns)


def copy_query(defn: Definition, logical: Logical, model: SemanticModel, compiled: C.Compiled) -> str | None:
    """The copy's query giving exactly the rows ``compiled`` gives on the database, or None."""
    on_copy = moved(defn, logical, model)
    if on_copy is None:
        return None
    try:
        writer = _OnTheCopy(on_copy, copy_model(defn, model), "duckdb")
        writer.as_compared = defn.dialect == "tsql"
        written = writer.compile()
    except (C.CompileError, KeyError, ValueError) as exc:
        log.info("core2: a question fitting a copy could not be written on it: %s", exc)
        return None
    if [c.name for c in written.columns] != [c.name for c in compiled.columns] or written.row_cap != compiled.row_cap:
        return None
    return written.sql


# ── kept on this server ──────────────────────────────────────────────────────

def _folded(dialect: str) -> Callable[[str], str]:
    """A column's name as the database hands it back: Snowflake and Oracle give an unquoted name in capitals."""
    if dialect not in ("snowflake", "oracle"):
        return lambda name: name
    return lambda name: name if name.upper() in D.RESERVED or not C._GENERATED.fullmatch(name) else name.upper()


def read(path: Path, sql: str, max_rows: int, dialect: str) -> Any:
    """The copy's rows for ``sql``, as the database would hand them back: one dict per row, at most ``max_rows``
    (and whether there were more)."""
    import duckdb

    from core.schema import QueryRows

    con = duckdb.connect()
    try:
        con.execute(f"ATTACH '{path.as_posix()}' AS kept (READ_ONLY)")
        con.execute("USE kept")
        # Nothing a query sends reaches another file of this server, and the setting cannot be turned back on.
        con.execute("SET enable_external_access = false")
        con.execute("SET lock_configuration = true")
        cur = con.execute(sql)
        fold = _folded(dialect)
        names = [fold(str(d[0])) for d in cur.description or ()]
        rows = cur.fetchmany(max_rows + 1) if max_rows > 0 else cur.fetchall()
    finally:
        con.close()
    truncated = max_rows > 0 and len(rows) > max_rows
    if truncated:
        rows = rows[:max_rows]
    return QueryRows([dict(zip(names, r)) for r in rows], truncated=truncated)


def write(path: Path, defn: Definition, rows: list[tuple]) -> int:
    """The copy's file, written beside its old one and then put in its place; its size in bytes."""
    import duckdb

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".making")
    temp.unlink(missing_ok=True)
    con = duckdb.connect(str(temp))
    try:
        con.execute(f"CREATE TABLE {TABLE} (" + ", ".join(
            f"{name} {_DUCK_TYPES.get(dtype, 'VARCHAR')}" for name, dtype in defn.columns) + ")")
        placeholders = ", ".join("?" for _ in defn.columns)
        batch: list[tuple] = []
        for row in rows:
            batch.append(tuple(_plain(v, dtype) for v, (_, dtype) in zip(row, defn.columns)))
            if len(batch) >= 20_000:
                con.executemany(f"INSERT INTO {TABLE} VALUES ({placeholders})", batch)
                batch = []
        if batch:
            con.executemany(f"INSERT INTO {TABLE} VALUES ({placeholders})", batch)
        con.execute("CHECKPOINT")
    finally:
        con.close()
    os.replace(temp, path)
    return path.stat().st_size


def compared_ignoring_case(model: SemanticModel, defn: Definition, ask: Callable[[str], list[tuple]]) -> None:
    """SQL Server: a copy compares its names ignoring capitals and trailing spaces, as SQL Server's usual collations
    do. CopyError for a field whose collation counts capitals (_CS, _BIN), or whose collation cannot be read.
    ``ask`` runs a query on the warehouse and hands back its rows."""
    if defn.dialect != "tsql":
        return
    for d in defn.dims:
        column = model.columns[d.column]
        if column.data_type != "text":
            continue
        table = model.tables[column.table]
        field = column.business_name or column.name
        quoted = [str(s).replace("'", "''") for s in (table.schema_name, table.name, column.name)]
        sql = ("SELECT COLLATION_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = '{}' AND TABLE_NAME = '{}' "
               "AND COLUMN_NAME = '{}'").format(*quoted)
        try:
            found = [str(r[0] or "") for r in ask(sql)]
        except Exception as exc:  # noqa: BLE001 - said on the admin's page
            raise CopyError(f"how the database compares {field}'s names could not be read: {str(exc)[:200]}") from exc
        if not found:
            raise CopyError(f"how the database compares {field}'s names could not be read")
        counted = next((f for f in found if "_CS" in f.upper() or "_BIN" in f.upper()), "")
        if counted:
            raise CopyError(f"{field} counts capitals as different (its collation is {counted}): a copy compares "
                            "names as most SQL Server databases do, ignoring them, so it cannot keep this field")


def kept_as_compared(defn: Definition, rows: list[tuple]) -> None:
    """CopyError when the copy's names would not compare as the warehouse compares them: SQL Server counts 'East'
    and 'east ' as one name (and hands back either), the copy as two."""
    if defn.dialect != "tsql":
        return
    at = {name: i for i, (name, _) in enumerate(defn.columns)}
    for d in defn.dims:
        i = at[d.name]
        if dict(defn.columns)[d.name] != "text":
            continue
        seen: dict[str, str] = {}
        for row in rows:
            value = row[i]
            if not isinstance(value, str):
                continue
            key = value.rstrip().casefold()
            if seen.setdefault(key, value) != value:
                field = d.column.rsplit(".", 1)[-1]
                raise CopyError(f"some of {field}'s names differ only in capitals or trailing spaces "
                                f"('{seen[key]}', '{value}'), which SQL Server counts as one: a copy cannot keep them")


def _plain(value: Any, dtype: str) -> Any:
    """A value as the copy keeps it: a day as a date (Oracle's TRUNC gives a date and time)."""
    if dtype == "date" and isinstance(value, dt.datetime):
        return value.date()
    return value


# ── the workspace's warehouse, with its copies ───────────────────────────────

@dataclass
class Ready:
    """A copy ready to answer: its row, its file, and its definition as the model reads it now."""

    id: int
    name: str
    metric: str
    fields: list[str]
    date: str
    grain: str
    every_hours: int
    made_at: dt.datetime
    path: Path
    fingerprint: str
    _defined: dict[int, Definition | None] = field(default_factory=dict, repr=False)

    def definition(self, model: SemanticModel, dialect: str, table: str = "") -> Definition | None:
        """Its definition as ``model`` reads it now, when it is the one it was made from (and of ``table``)."""
        measure = model.measures.get(self.metric)
        if measure is None or (table and measure.table != table):
            return None
        key = id(model)
        if key not in self._defined:
            try:
                defn = define(model, metric=self.metric, fields=self.fields, date=self.date, grain=self.grain,
                              today=dt.date.today(), dialect=dialect)
                self._defined[key] = defn if defn.fingerprint == self.fingerprint else None
            except CopyError:
                self._defined[key] = None
        return self._defined[key]


def fresh(made_at: dt.datetime, every_hours: int, now: dt.datetime | None = None) -> bool:
    """Made within twice its schedule (a missed making or two is waited out; then the database answers)."""
    now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    return now - made_at < dt.timedelta(hours=2 * max(1, every_hours))


class Precomputed:
    """The workspace's warehouse with the copies it keeps: a query a copy answers exactly is read from the copy,
    through the same governance; any other goes to the database.

    One answer's figures come from one place: its first query decides. Read from a copy, the answer's other
    queries are read from copies where they hold them (a figure only the database has is read there, and said);
    asked of the database, all of them are, so a breakdown read now never sits beside a total from a copy made
    hours ago. ``taken()`` ends an answer on this thread and hands over what it read: a dashboard's tiles, answered
    at once on threads of their own, each say only their own."""

    def __init__(self, inner: Any, copies: list[Ready], seen: Callable[[int], None] | None = None):
        self.inner = inner
        self.copies = copies
        self.seen = seen
        self._here = threading.local()

    @property
    def used(self) -> list[Ready]:
        found = getattr(self._here, "used", None)
        if found is None:
            found = self._here.used = []
        return found

    @property
    def mixed(self) -> bool:
        """Did an answer read from a copy also read a figure from the database?"""
        return bool(getattr(self._here, "mixed", False))

    def taken(self) -> list[Ready]:
        found = list(self.used)
        self.used.clear()
        self._here.mode = None
        self._here.mixed = False
        return found

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @property
    def dialect(self) -> str:
        return str(self.inner.dialect)

    @property
    def db_type(self) -> str:
        return str(self.inner.db_type)

    def query(self, sql: str, *, max_rows: int | None = None) -> Any:
        found = C.compiled_from(sql)
        mode = getattr(self._here, "mode", None)
        chosen: tuple[Ready, str] | None = None
        if found is not None and mode != "database":
            logical, model, compiled = found
            for kept in self.copies:
                try:
                    defn = kept.definition(model, self.dialect, logical.parts[0].table) \
                        if len(logical.parts) == 1 else None
                    on_copy = copy_query(defn, logical, model, compiled) if defn is not None else None
                except Exception as exc:  # noqa: BLE001 - a question a copy cannot take is the database's, as ever
                    log.warning("core2: copy %s not considered for a query: %s", kept.id, exc, exc_info=True)
                    continue
                if on_copy is not None:
                    chosen = (kept, on_copy)
                    break
        if chosen is None:
            if found is not None:
                # The answer's first question to the data decides where all its figures come from.
                if mode is None:
                    self._here.mode = "database"
                elif mode == "copy":
                    self._here.mixed = True
            return self.inner.query(sql, max_rows=max_rows)
        kept, on_copy = chosen
        dialect = self.dialect

        def local(cap: int) -> Any:
            return read_as_cut(kept, on_copy, cap, dialect)

        try:
            result = self.inner.query(sql, max_rows=max_rows, local=local)
        except Exception as exc:  # noqa: BLE001 - a copy that cannot be read: the database answers, and it is said
            if not _from_copy(exc):
                raise
            log.warning("core2: copy %s not used; the database answers: %s", kept.id, exc)
            if mode is None:
                self._here.mode = "database"
            else:
                self._here.mixed = True
            return self.inner.query(sql, max_rows=max_rows)
        if getattr(result, "source", "") != "copy":
            # The database answered after all (a row rule applies to this reader): so does the rest of the answer.
            if mode is None:
                self._here.mode = "database"
            else:
                self._here.mixed = True
        else:
            self._here.mode = "copy"
            self.used.append(kept)
            if self.seen is not None:
                try:
                    self.seen(kept.id)
                except Exception as exc:  # noqa: BLE001 - a count on the admin page, never the answer
                    log.warning("core2: copy %s use not counted: %s", kept.id, exc)
        return result


class _Unreadable(RuntimeError):
    """The copy cannot give the database's rows: its file could not be read (taken away by a newer making, or
    damaged), or its rows would be cut where the database cuts them otherwise."""


def read_as_cut(kept: Ready, sql: str, cap: int, dialect: str) -> Any:
    """The copy's rows for ``sql``; _Unreadable when they cannot be the database's. Rows cut at ``cap`` are cut in
    the copy's order of names, the database's only where it compares names byte by byte."""
    try:
        rows = read(kept.path, sql, cap, dialect)
    except Exception as exc:  # noqa: BLE001 - told apart by Precomputed.query: the database answers instead
        raise _Unreadable(f"copy {kept.id} could not be read: {exc}") from exc
    if getattr(rows, "truncated", False) and dialect not in _BYTE_ORDER:
        raise _Unreadable(f"copy {kept.id} holds more rows than are shown, cut in another order than the database's")
    return rows


def _from_copy(exc: BaseException) -> bool:
    """Was it the copy's file that failed, not the governance or the database?"""
    seen: BaseException | None = exc
    while seen is not None:
        if isinstance(seen, _Unreadable):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def said(payload: dict[str, Any], used: list[Ready], model: SemanticModel, now: dt.datetime | None = None, *,
         mixed: bool = False) -> None:
    """The answer says it was read from a copy, and when that copy was made (in the reader's language too);
    ``mixed``: and that a figure no copy held was read from the database now."""
    if not used or payload.get("type") != "assistant_response":
        return
    now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    kept = max(used, key=lambda r: now - r.made_at)        # the oldest copy read, if several were
    age = now - kept.made_at
    minutes = int(age.total_seconds() // 60)
    when = ("less than a minute ago" if minutes < 1 else f"{minutes} minute{'s' if minutes != 1 else ''} ago"
            if minutes < 60 else f"{minutes // 60} hour{'s' if minutes // 60 != 1 else ''} ago")
    quand = ("il y a moins d'une minute" if minutes < 1 else f"il y a {minutes} minute{'s' if minutes != 1 else ''}"
             if minutes < 60 else f"il y a {minutes // 60} heure{'s' if minutes // 60 != 1 else ''}")
    every = f"every {kept.every_hours} hour{'s' if kept.every_hours != 1 else ''}"
    chaque = f"toutes les {kept.every_hours} heures" if kept.every_hours != 1 else "toutes les heures"
    en = (f"Read from Solviz's copy of {kept.name}, kept on its server and made {when} "
          f"({kept.made_at:%d %b %H:%M} UTC; it is made again {every}): anything recorded since is not in it yet.")
    fr = (f"Lu dans la copie de {kept.name}, gardée par Solviz sur son serveur et faite {quand} "
          f"({kept.made_at:%d/%m %H:%M} UTC ; elle est refaite {chaque}) : ce qui a été enregistré depuis n'y est "
          "pas encore.")
    if mixed:
        en += " A figure the copy does not hold was read from the database now."
        fr += " Un chiffre que la copie ne contient pas a été lu dans la base de données à l'instant."
    caveats = payload.setdefault("coverage_caveats", [])
    if en not in caveats:
        caveats.append(en)
        payload.setdefault("coverage_caveats_fr", {})[en] = fr
    payload.setdefault("trust", {})["copy"] = {"name": kept.name, "made_at": kept.made_at.strftime("%Y-%m-%d %H:%M"),
                                               "every_hours": kept.every_hours}


def folder_for(account_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", account_id)[:80] or "workspace"
    return copies_dir().resolve() / safe


def file_for(account_id: str, copy_id: int) -> Path:
    """A copy's file: one per making, so an answer reading the last one is never cut short by the next."""
    import uuid

    return folder_for(account_id) / f"copy-{int(copy_id)}-{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}.duckdb"


def files_of(account_id: str, copy_id: int) -> list[Path]:
    """Every file of the copy on this server, made by any making."""
    return sorted(folder_for(account_id).glob(f"copy-{int(copy_id)}-*.duckdb"))


_MAKING: set[int] = set()
_MAKING_LOCK = threading.Lock()
