"""SQL for the analytical shapes (core2/resolve/shapes.py), on every warehouse.

* per member, then over the members: the plain answer grouped by the member too (``m``), summarised a level up.
  A median or percentile is the window form (``PERCENTILE_CONT(p) WITHIN GROUP (ORDER BY v) OVER (PARTITION BY
  ...)``, the only one SQL Server has), then one value per group; averages are never whole-number averages;
* ranges of each member's figure: the members counted (and their figures totalled) per range;
* a rank inside each member of a grouping: ``ROW_NUMBER()`` over the plain answer, kept to the first n a level up
  (a window is never filtered where it is worked out);
* a series accumulated or smoothed: the plain series over the periods read (further back than shown); the running
  or moving figure is worked out from its rows (core2/answer/shaped.py);
* who had none: the listed members with ``NOT EXISTS`` a row of activity in the window.

Figures an answer only states (how many members were counted, how many had none) are columns of the query that
are not among its columns: the builder never shows them, the answer's words read them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from sqlglot import exp

from core2.compile.compiler import CompileError, OutColumn, _Compiler, _double, _num, equal_keys
from core2.model.schema import AggExpr
from core2.resolve import bands as B
from core2.resolve.resolver import Logical, OutMeasure, Part, PartGroup
from core2.resolve.shapes import AbsentShape, PerShape, RollingShape, WithinShape
from core2.warehouse import dialect as D

MEMBERS = "members_counted"     # per member: how many members (or rows) have a figure
LISTED = "members_listed"       # per member over everyone: how many are listed
TOTAL = "members_total"         # who had none: how many there are in all (the list may be cut)
CUTS = (20, 40, 60, 80)         # the percentiles ranges picked from the data are cut near


def shaped(c: _Compiler) -> tuple[exp.Select, list[OutColumn], str]:
    shape = c.q.shape
    if isinstance(shape, PerShape):
        return _per(c, shape)
    if isinstance(shape, WithinShape):
        return _within(c, shape)
    if isinstance(shape, RollingShape):
        return _rolling(c, shape)
    if isinstance(shape, AbsentShape):
        return _absent(c, shape)
    raise CompileError(f"no SQL for {type(shape).__name__}")


def _not_null(value: exp.Expression) -> exp.Expr:
    return exp.not_(exp.Is(this=value.copy(), expression=exp.Null()))


def percentile(c: _Compiler, value: exp.Expression, p: float, partition: list[exp.Expression]) -> exp.Window:
    """The p-th percentile of ``value`` (interpolated) over each partition, on every row of it. SQL Server sorts
    no values first by itself and refuses a NULLS LAST inside WITHIN GROUP; the others are told nulls last."""
    ordered = exp.Ordered(this=value.copy(), desc=False, nulls_first=c.d == "tsql")
    within = exp.WithinGroup(this=exp.PercentileCont(this=_num(round(p / 100, 6))),
                             expression=exp.Order(expressions=[ordered]))
    return exp.Window(this=within, partition_by=[e.copy() for e in partition])


# ── per member, then over the members ──────────────────────────────────────


def _per(c: _Compiler, shape: PerShape) -> tuple[exp.Select, list[OutColumn], str]:
    body, _, _ = _Compiler(shape.inner, c.model, c.d).shaped_body()
    v = c.out("m", shape.value)
    # A member is one with a name (its code only keeps two of one name apart: a member with no code is one).
    keep = [_not_null(c.out("m", n)) for n in shape.members[:1]]
    plain = [g for g in c.q.groups if shape.band is None or g.name != shape.band.name]
    groups = [c.out("m", g.name) for g in plain]
    columns = [c.group_column(g) for g in c.q.groups]
    named = [e.copy().as_(c.name(g.name)) for e, g in zip(groups, plain)]
    source = body.subquery(c.name("m"))

    if shape.band is not None:
        band = c.band(v, shape.band)
        level = exp.select(*named, band.as_(c.name(shape.band.name)))
        counted, *total = c.q.measures
        level = level.select(exp.Count(this=exp.Star()).as_(c.name(counted.name)))
        if total:
            level = level.select(exp.Sum(this=v.copy()).as_(c.name(total[0].name)))
        level = level.from_(source).where(exp.and_(*keep, _not_null(v)))
        level = level.group_by(*[e.copy() for e in groups], band.copy())
        columns += [OutColumn(o.name, o.label, "measure", o.format, None) for o in c.q.measures]
        return exp.select(*[c.out("q", col.name) for col in columns]).from_(level.subquery(c.name("q"))), columns, "q"

    out = c.q.measures[0]
    if shape.summary in ("median", "percentile"):
        p = 50.0 if shape.summary == "median" else float(shape.p or 50)
        ranked = exp.select(*named, v.copy().as_(c.name("v")), percentile(c, v, p, groups).as_(c.name("pct")))
        ranked = ranked.from_(source)
        if keep:
            ranked = ranked.where(exp.and_(*keep))
        t = [c.out("t", g.name) for g in plain]
        level = exp.select(*[e.copy().as_(c.name(g.name)) for e, g in zip(t, plain)],
                           exp.Max(this=c.out("t", "pct")).as_(c.name(out.name)),
                           exp.Count(this=c.out("t", "v")).as_(c.name(MEMBERS))).from_(ranked.subquery(c.name("t")))
        if t:
            level = level.group_by(*[e.copy() for e in t])
        extra = [MEMBERS]
    else:
        listed: exp.Expression | None = None
        level = exp.select(*named).from_(source)
        if shape.listed is not None:
            # How many are listed, beside every member's figure (one row: the same for every group).
            count_all = OutMeasure(LISTED, "", AggExpr(agg="count"), "count", None)
            every = c.part_select(dataclasses.replace(shape.listed, measures=[count_all], groups=[]))
            level = level.join(every.subquery(c.name("l")), join_type="cross")
            listed = exp.Max(this=c.out("l", LISTED))
        level = level.select(_summary(c, shape, v, listed).as_(c.name(out.name)),
                             exp.Count(this=v.copy()).as_(c.name(MEMBERS)))
        extra = [MEMBERS]
        if listed is not None:
            level = level.select(listed.copy().as_(c.name(LISTED)))
            extra.append(LISTED)
        if keep:
            level = level.where(exp.and_(*keep))
        if groups:
            level = level.group_by(*[e.copy() for e in groups])
    columns.append(OutColumn(out.name, out.label, "measure", out.format, None))
    outer = exp.select(*[c.out("q", col.name) for col in columns], *[c.out("q", n) for n in extra])
    return outer.from_(level.subquery(c.name("q"))), columns, "q"


def _summary(c: _Compiler, shape: PerShape, v: exp.Expression, listed: exp.Expression | None) -> exp.Expression:
    """The summary over the members' figures; over everyone listed (``listed``), a member with no rows is 0."""
    rows = exp.Count(this=exp.Star())
    missing = D.sub(listed.copy(), rows.copy()) if listed is not None else None
    if shape.summary == "avg":
        if listed is not None:
            return D.div(_double(exp.Sum(this=v.copy())), exp.Nullif(this=listed.copy(), expression=_num(0)))
        return exp.Avg(this=_double(v.copy()))
    if shape.summary in ("min", "max"):
        agg = exp.Min(this=v.copy()) if shape.summary == "min" else exp.Max(this=v.copy())
        if listed is None:
            return agg
        beyond = exp.GT(this=agg.copy(), expression=_num(0)) if shape.summary == "min" \
            else exp.LT(this=agg.copy(), expression=_num(0))
        return exp.Case(ifs=[exp.If(this=exp.and_(exp.GT(this=listed.copy(), expression=rows.copy()), beyond),
                                    true=_num(0))], default=agg)
    met = c.predicate(v.copy(), shape.op or "eq", list(shape.values))
    count: exp.Expression = exp.Count(this=exp.Case(ifs=[exp.If(this=met, true=_num(1))]))
    if missing is not None and meets(0.0, shape.op or "eq", list(shape.values)):
        count = D.add(count, missing)
    if shape.summary == "count":
        return count
    whole = listed.copy() if listed is not None else exp.Count(this=v.copy())
    return D.mul(D.div(_double(count), exp.Nullif(this=whole, expression=_num(0))), 100)


def meets(value: float, op: str, values: list[float]) -> bool:
    """Does ``value`` meet the condition (a member with no rows, as 0, counted when it does)?"""
    if op == "between":
        return len(values) == 2 and values[0] <= value <= values[1]
    if not values:
        return False
    x = values[0]
    return {"eq": value == x, "ne": value != x, "gt": value > x, "gte": value >= x, "lt": value < x,
            "lte": value <= x}.get(op, False)


# ── a rank inside each member of a grouping ────────────────────────────────


def _within(c: _Compiler, shape: WithinShape) -> tuple[exp.Select, list[OutColumn], str]:
    body, columns, _ = c.shaped_body()
    order = [exp.Ordered(this=c.out("w", shape.by), desc=shape.desc, nulls_first=False),
             exp.Ordered(this=c.out("w", shape.member), desc=False, nulls_first=False)]
    number = exp.Window(this=exp.RowNumber(), partition_by=[c.out("w", n) for n in shape.partition],
                        order=exp.Order(expressions=order))
    ranked = exp.select(*[c.out("w", col.name) for col in columns], number.as_(c.name(shape.rank)))
    ranked = ranked.from_(body.subquery(c.name("w")))
    columns = [*columns, OutColumn(shape.rank, "Rank", "rank", "count")]
    outer = exp.select(*[c.out("t", col.name) for col in columns]).from_(ranked.subquery(c.name("t")))
    if shape.top:
        outer = outer.where(exp.LTE(this=c.out("t", shape.rank), expression=_num(shape.top)))
    return outer, columns, "t"


# ── a series accumulated or smoothed ───────────────────────────────────────


def _rolling(c: _Compiler, shape: RollingShape) -> tuple[exp.Select, list[OutColumn], str]:
    """The plain series over the periods read; each measure's running or moving figure is a column of its own,
    worked out from the rows, and the measure itself is kept beside it."""
    read = _Compiler(dataclasses.replace(c.q, window=shape.read, shape=None), c.model, c.d)
    body, columns, alias = read.shaped_body()
    out: list[OutColumn] = []
    for col in columns:
        if col.role == "measure" and col.name in shape.columns:
            out.append(OutColumn(shape.columns[col.name], shape.labels[col.name], "measure", col.format,
                                 col.measure, computed=True))
            out.append(dataclasses.replace(col, role="base"))
        else:
            out.append(col)
    return body, out, alias


# ── who had none ───────────────────────────────────────────────────────────


def _absent(c: _Compiler, shape: AbsentShape) -> tuple[exp.Select, list[OutColumn], str]:
    listed = c.part_select(shape.entity)
    key = c.col(shape.entity.alias, shape.key)

    def has(part: Part, out_key: str, alias: str) -> exp.Exists:
        rows = c.part_select(part).subquery(c.name(alias))
        same = equal_keys(c.model, c.out(alias, out_key), shape.key, key.copy(), shape.key, c.d)
        return exp.Exists(this=exp.select(_num(1)).from_(rows).where(same))

    kept: list[exp.Expr] = [exp.not_(has(shape.activity, shape.activity_key, "had_none"))]
    if shape.had is not None:
        kept.append(has(shape.had, shape.had_key, "had_some"))
    if shape.before is not None:
        kept.append(has(shape.before, shape.before_key, "had_before"))
    listed = listed.where(exp.and_(*kept))
    listed = listed.select(exp.Window(this=exp.Count(this=exp.Star())).as_(c.name(TOTAL)))
    columns = [c.group_column(g) for g in c.q.groups]
    outer = exp.select(*[c.out("q", col.name) for col in columns], c.out("q", TOTAL))
    return outer.from_(listed.subquery(c.name("q"))), columns, "q"


# ── ranges picked from the data ────────────────────────────────────────────


def _unset_band(logical: Logical) -> tuple[Part | None, PartGroup] | None:
    """The range grouping still without edges (they are picked from the data): its part (None for each member's
    figure) and its grouping."""
    shape = logical.shape
    if isinstance(shape, PerShape) and shape.band is not None and not shape.band.edges:
        return None, shape.band
    for part in logical.parts:
        for g in part.groups:
            if g.kind == "band" and not g.edges:
                return part, g
    return None


def edges_sql(logical: Logical, model, dialect: str) -> str | None:
    """A query of the values a range grouping without edges splits: lowest, highest and the cut percentiles."""
    found = _unset_band(logical)
    if found is None:
        return None
    part, g = found
    c = _Compiler(logical, model, dialect)
    if part is None:
        shape = logical.shape
        assert isinstance(shape, PerShape)
        body, _, _ = _Compiler(shape.inner, model, dialect).shaped_body()
        keep = [_not_null(c.out("m", n)) for n in shape.members[:1]]
        values = exp.select(c.out("m", shape.value).as_(c.name("v"))).from_(body.subquery(c.name("m")))
        if keep:
            values = values.where(exp.and_(*keep))
    else:
        value = c.days(*g.days) if g.days is not None else c.col(g.alias, g.column) if g.column else None
        if value is None:
            raise CompileError(f"{g.name} has no value to put in ranges")
        values = c.part_select(dataclasses.replace(part, groups=[], measures=[], distinct_only=False,
                                                   row_level=True)).select(value.as_(c.name("v")))
    v = c.out("e", "v")
    cut = exp.select(v.copy().as_(c.name("v")), *[percentile(c, v, p, []).as_(c.name(f"p{p}")) for p in CUTS])
    cut = cut.from_(values.subquery(c.name("e"))).where(_not_null(v))
    t = c.out("t", "v")
    stats = exp.select(exp.Min(this=t).as_(c.name("low")), exp.Max(this=t.copy()).as_(c.name("high")),
                       *[exp.Max(this=c.out("t", f"p{p}")).as_(c.name(f"p{p}")) for p in CUTS])
    return D.render(stats.from_(cut.subquery(c.name("t"))), dialect)


def fill_edges(logical: Logical, model, dialect: str, query: Callable[[str], list[tuple]]) -> Logical:
    """``logical`` with the edges of its range grouping picked from the data (round numbers near the values that
    split it into about equal parts), when the question named none."""
    sql = edges_sql(logical, model, dialect)
    if sql is None:
        return logical
    part, g = _unset_band(logical)  # type: ignore[misc]
    rows = query(sql)
    row = rows[0] if rows else None
    numbers = [None if x is None else float(x) for x in row] if row else []
    edges: list[float] = []
    if numbers and numbers[0] is not None and numbers[1] is not None:
        low, high, *cuts = numbers
        edges = B.edges_from(cuts, low, high, whole=bool(g.style.get("whole")))
    g.edges = edges
    g.labels = B.labels(edges, whole=bool(g.style.get("whole")), unit=str(g.style.get("unit") or ""),
                        money=bool(g.style.get("money")))
    for group in logical.groups:
        if group.name == g.name:
            group.bands = list(g.labels)
    return logical
