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
from core2.resolve.shapes import (AGEING, AbsentShape, ActiveShape, CohortShape, Events, GapShape, LifecycleShape,
                                  OnTimeShape, PerShape, RollingShape, WithinShape)
from core2.resolve.time import Range
from core2.warehouse import dialect as D

MEMBERS = "members_counted"     # per member: how many members (or rows) have a figure
REACHED = "members_reached"     # days to the nth event: how many members had one
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
    if isinstance(shape, GapShape):
        return _gap(c, shape)
    if isinstance(shape, LifecycleShape):
        return _lifecycle(c, shape)
    if isinstance(shape, CohortShape):
        return _cohort(c, shape)
    if isinstance(shape, ActiveShape):
        return _active(c, shape)
    if isinstance(shape, OnTimeShape):
        return _ageing(c, shape) if shape.ageing else _on_time(c, shape)
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


# ── each member's days with activity, in order ─────────────────────────────


def within(c: _Compiler, value: exp.Expression, rng: Range) -> list[exp.Expr]:
    """``value`` (a DATE) inside ``[start, end)``; nothing for an open side."""
    out: list[exp.Expr] = []
    if rng.start is not None:
        out.append(exp.GTE(this=value.copy(), expression=D.date_literal(rng.start, c.d)))
    if rng.end is not None:
        out.append(exp.LT(this=value.copy(), expression=D.date_literal(rng.end, c.d)))
    return out


def sequence(c: _Compiler, e: Events) -> exp.Subquery:
    """Each member's days with activity, each with the one before and after it and its place (1 for the first):
    ``s.<key>, s.<day>, s.prev_day, s.next_day, s.event_n`` and the member's own groupings."""
    ev = c.part_select(e.part).subquery(c.name("ev"))
    key, day = c.out("ev", e.key), c.out("ev", e.day)

    def over(fn: exp.Expression) -> exp.Window:
        return exp.Window(this=fn, partition_by=[key.copy()],
                          order=exp.Order(expressions=[exp.Ordered(this=day.copy(), desc=False, nulls_first=False)]))

    return exp.select(key.copy().as_(c.name(e.key)), day.copy().as_(c.name(e.day)),
                      *[c.out("ev", g).as_(c.name(g)) for g in e.groups],
                      over(exp.Lag(this=day.copy())).as_(c.name("prev_day")),
                      over(exp.Lead(this=day.copy())).as_(c.name("next_day")),
                      over(exp.RowNumber()).as_(c.name("event_n"))).from_(ev).subquery(c.name("s"))


def _grouped(c: _Compiler, e: Events, alias: str, dated: exp.Expression) -> list[tuple[exp.Expression, str]]:
    """The answer's groupings over a level of rows: the period of ``dated`` (a series), then the member's own."""
    out: list[tuple[exp.Expression, str]] = []
    period = next((g for g in c.q.groups if g.kind == "period"), None)
    if period is not None and e.grain is not None:
        out.append((D.period_start(dated.copy(), e.grain, c.d), period.name))
    out += [(c.out(alias, g), g) for g in e.groups]
    return out


def _gap(c: _Compiler, shape: GapShape) -> tuple[exp.Select, list[OutColumn], str]:
    e = shape.events
    seq = sequence(c, e)
    day = c.out("s", e.day)
    m = c.q.measures[0]
    columns = [c.group_column(g) for g in c.q.groups] + [OutColumn(m.name, m.label, "measure", m.format, None)]
    hidden = [MEMBERS]
    if shape.summary == "nth":
        key = c.out("s", e.key)
        n = c.out("s", "event_n")
        firsts = exp.select(
            key.copy().as_(c.name(e.key)), *[c.out("s", g).as_(c.name(g)) for g in e.groups],
            exp.Min(this=exp.Case(ifs=[exp.If(this=exp.EQ(this=n.copy(), expression=_num(1)), true=day.copy())]))
            .as_(c.name("first_day")),
            exp.Min(this=exp.Case(ifs=[exp.If(this=exp.EQ(this=n.copy(), expression=_num(shape.nth or 2)),
                                              true=day.copy())])).as_(c.name("nth_day")),
        ).from_(seq).group_by(key.copy(), *[c.out("s", g) for g in e.groups]).subquery(c.name("f"))
        first, nth = c.out("f", "first_day"), c.out("f", "nth_day")
        groups = _grouped(c, e, "f", first)
        body = exp.select(*[x.as_(c.name(n_)) for x, n_ in groups],
                          exp.Avg(this=_double(D.days_between(first.copy(), nth.copy(), c.d))).as_(c.name(m.name)),
                          exp.Count(this=exp.Star()).as_(c.name(MEMBERS)),
                          exp.Count(this=nth.copy()).as_(c.name(REACHED))).from_(firsts)
        conds = within(c, first, e.window)
        hidden.append(REACHED)
    elif shape.summary == "within":
        nxt = c.out("s", "next_day")
        soon = exp.and_(_not_null(nxt), exp.LTE(this=D.days_between(day.copy(), nxt.copy(), c.d),
                                                 expression=_num(shape.within_days or 0)))
        flag = exp.Case(ifs=[exp.If(this=soon, true=exp.Literal.number("100.0"))], default=exp.Literal.number("0.0"))
        groups = _grouped(c, e, "s", day)
        body = exp.select(*[x.as_(c.name(n_)) for x, n_ in groups], exp.Avg(this=flag).as_(c.name(m.name)),
                          exp.Count(this=exp.Star()).as_(c.name(MEMBERS))).from_(seq)
        conds = within(c, day, e.window)
    else:
        prev = c.out("s", "prev_day")
        gap = D.days_between(prev.copy(), day.copy(), c.d)
        groups = _grouped(c, e, "s", day)
        conds = [_not_null(prev), *within(c, day, e.window)]
        if shape.summary == "median":
            inner = exp.select(*[x.as_(c.name(n_)) for x, n_ in groups],
                               percentile(c, gap, 50, [x for x, _ in groups]).as_(c.name("v")))
            inner = inner.from_(seq).where(exp.and_(*conds)).subquery(c.name("g"))
            names = [n_ for _, n_ in groups]
            body = exp.select(*[c.out("g", n_).as_(c.name(n_)) for n_ in names],
                              exp.Max(this=c.out("g", "v")).as_(c.name(m.name)),
                              exp.Count(this=exp.Star()).as_(c.name(MEMBERS))).from_(inner)
            if names:
                body = body.group_by(*[c.out("g", n_) for n_ in names])
            outer = exp.select(*[c.out("q", col.name) for col in columns], *[c.out("q", h) for h in hidden])
            return outer.from_(body.subquery(c.name("q"))), columns, "q"
        agg = {"avg": lambda v: exp.Avg(this=_double(v)), "min": lambda v: exp.Min(this=v),
               "max": lambda v: exp.Max(this=v)}[shape.summary]
        body = exp.select(*[x.as_(c.name(n_)) for x, n_ in groups], agg(gap).as_(c.name(m.name)),
                          exp.Count(this=exp.Star()).as_(c.name(MEMBERS))).from_(seq)
    if conds:
        body = body.where(exp.and_(*conds))
    if groups:
        body = body.group_by(*[x.copy() for x, _ in groups])
    outer = exp.select(*[c.out("q", col.name) for col in columns], *[c.out("q", h) for h in hidden])
    return outer.from_(body.subquery(c.name("q"))), columns, "q"


# ── new, returning, won back, lapsed ───────────────────────────────────────

_MONTHS = {"month": 1, "quarter": 3, "year": 12}


def member_periods(c: _Compiler, e: Events) -> exp.Subquery:
    """Each member's periods with activity, each once, with the period before and after it that had any and their
    first: ``mp.<key>, mp.period_start, mp.prev_period, mp.next_period, mp.first_period`` and their own groupings."""
    assert e.grain is not None
    ev = c.part_select(e.part).subquery(c.name("ev"))
    key = c.out("ev", e.key)
    periods = exp.select(key.copy().as_(c.name(e.key)),
                         D.period_start(c.out("ev", e.day), e.grain, c.d).as_(c.name("period_start")),
                         *[c.out("ev", g).as_(c.name(g)) for g in e.groups]).distinct().from_(ev)
    k, p = c.out("pp", e.key), c.out("pp", "period_start")

    def over(fn: exp.Expression, ordered: bool = True) -> exp.Window:
        order = exp.Order(expressions=[exp.Ordered(this=p.copy(), desc=False, nulls_first=False)]) if ordered \
            else None
        return exp.Window(this=fn, partition_by=[k.copy()], order=order)

    return exp.select(k.copy().as_(c.name(e.key)), p.copy().as_(c.name("period_start")),
                      *[c.out("pp", g).as_(c.name(g)) for g in e.groups],
                      over(exp.Lag(this=p.copy())).as_(c.name("prev_period")),
                      over(exp.Lead(this=p.copy())).as_(c.name("next_period")),
                      over(exp.Min(this=p.copy()), ordered=False).as_(c.name("first_period"))) \
        .from_(periods.subquery(c.name("pp"))).subquery(c.name("mp"))


def _flag(condition: exp.Expr) -> exp.Expression:
    return exp.Case(ifs=[exp.If(this=condition, true=_num(1))], default=_num(0))


def _lifecycle(c: _Compiler, shape: LifecycleShape) -> tuple[exp.Select, list[OutColumn], str]:
    e = shape.events
    assert e.grain is not None
    months = _MONTHS[e.grain]
    p, prev, nxt, first = (c.out("mp", n) for n in ("period_start", "prev_period", "next_period", "first_period"))
    period = next(g for g in c.q.groups if g.kind == "period")
    own = [c.out("mp", g).as_(c.name(g)) for g in e.groups]
    zero = _num(0)

    def row(at: exp.Expression, active, new, returning, lapsed, before) -> exp.Select:
        return exp.select(at.as_(c.name(period.name)), *[x.copy() for x in own],
                          active.as_(c.name("is_active")), new.as_(c.name("is_new")),
                          returning.as_(c.name("is_returning")), lapsed.as_(c.name("is_lapsed")),
                          before.as_(c.name("was_active")))

    mp = member_periods(c, e)
    # Each member's period with activity, as it stands in that period...
    now = row(p.copy(), _num(1), _flag(exp.EQ(this=p.copy(), expression=first.copy())),
              _flag(exp.EQ(this=prev.copy(), expression=D.add_months(p.copy(), -months, c.d))),
              zero.copy(), zero.copy()).from_(mp)
    # ... and in the period after it: lapsed when it had none then.
    after = D.add_months(p.copy(), months, c.d)
    lapsed = _flag(exp.or_(exp.Is(this=nxt.copy(), expression=exp.Null()),
                           exp.NEQ(this=nxt.copy(), expression=after.copy())))
    then = row(after, zero.copy(), zero.copy(), zero.copy(), lapsed, _num(1)).from_(mp.copy())
    both = exp.union(now, then, distinct=False).subquery(c.name("lc"))
    at = c.out("lc", period.name)
    total = {n: exp.Sum(this=c.out("lc", f"is_{n}")) for n in ("active", "new", "returning", "lapsed")}
    total["was"] = exp.Sum(this=c.out("lc", "was_active"))
    figures = {
        "active": total["active"], "new": total["new"], "returning": total["returning"],
        "won_back": D.sub(D.sub(total["active"].copy(), total["new"].copy()), total["returning"].copy()),
        "lapsed": total["lapsed"],
        "churn_rate": D.div(D.mul(_double(total["lapsed"].copy()), 100),
                            exp.Nullif(this=total["was"].copy(), expression=_num(0))),
    }
    names = [period.name, *e.groups]
    body = exp.select(at.copy().as_(c.name(period.name)), *[c.out("lc", g).as_(c.name(g)) for g in e.groups],
                      *[figures[k].copy().as_(c.name(n)) for k, n in shape.columns.items()])
    if shape.status == "churn_rate":
        body = body.select(total["lapsed"].copy().as_(c.name("lapsed_count")),
                           total["was"].copy().as_(c.name("active_before")))
    # The periods asked for, and none after the last with activity (everyone would have lapsed then).
    last = exp.select(exp.Max(this=D.period_start(c.out("ev2", e.day), e.grain, c.d))).from_(
        c.part_select(e.part).subquery(c.name("ev2")))
    conds = [*within(c, at, e.window), exp.LTE(this=at.copy(), expression=last.subquery())]
    body = body.from_(both).where(exp.and_(*conds)).group_by(*[c.out("lc", n) for n in names])
    columns = [c.group_column(g) for g in c.q.groups] + [OutColumn(m.name, m.label, "measure", m.format, None)
                                                         for m in c.q.measures]
    hidden = ["lapsed_count", "active_before"] if shape.status == "churn_rate" else []
    outer = exp.select(*[c.out("q", col.name) for col in columns], *[c.out("q", h) for h in hidden])
    return outer.from_(body.subquery(c.name("q"))), columns, "q"


# ── cohorts ────────────────────────────────────────────────────────────────


def _months(c: _Compiler, value: exp.Expression) -> exp.Expression:
    """A month's place in time: year * 12 + month."""
    return D.add(D.mul(D.date_part(value.copy(), "year", c.d), 12), D.date_part(value.copy(), "month", c.d))


def _cohort(c: _Compiler, shape: CohortShape) -> tuple[exp.Select, list[OutColumn], str]:
    e = shape.events
    assert e.grain is not None
    step = _MONTHS[e.grain]
    mp = member_periods(c, e)
    p, first = c.out("mp", "period_start"), c.out("mp", "first_period")
    months = D.sub(_months(c, p), _months(c, first))
    # Whole periods, as a whole number (a floor is a decimal on some warehouses, and months are counted in whole ones).
    age = exp.Cast(this=D.floor_div(months, step) if step > 1 else months, to=exp.DataType.build("INT"))
    # How many of each cohort had activity n periods after their first.
    cells = (exp.select(first.copy().as_(c.name("cohort")), age.copy().as_(c.name("age")),
                        exp.Count(this=exp.Star()).as_(c.name("active")))
             .from_(mp).where(exp.and_(*within(c, first, e.window), exp.LTE(this=age.copy(), expression=_num(
                 shape.periods)))).group_by(first.copy(), age.copy()))
    last = exp.select(exp.Max(this=D.period_start(c.out("ev2", e.day), e.grain, c.d))).from_(
        c.part_select(e.part).subquery(c.name("ev2")))
    # Every cohort against every n seen in any of them, up to the last period with activity: a cohort with no one
    # active in a period that has happened shows 0, not nothing.
    co, ag = c.out("co", "cohort"), c.out("ag", "age")
    grid = (exp.select(co.copy().as_(c.name("cohort")), ag.copy().as_(c.name("age")))
            .from_(exp.select(c.out("c1", "cohort")).distinct().from_(cells.subquery(c.name("c1")))
                   .subquery(c.name("co")))
            .join(exp.select(c.out("c2", "age")).distinct().from_(cells.subquery(c.name("c2")))
                  .subquery(c.name("ag")), join_type="cross")
            .where(exp.LTE(this=D.add_months(co.copy(), D.mul(ag.copy(), step), c.d), expression=last.subquery())))
    g, cl, sz = c.out("g", "cohort"), c.out("cl", "active"), c.out("sz", "active")
    active = exp.Coalesce(this=cl.copy(), expressions=[_num(0)])
    size = exp.select(c.out("c3", "cohort"), c.out("c3", "active")).from_(cells.subquery(c.name("c3"))).where(
        exp.EQ(this=c.out("c3", "age"), expression=_num(0)))
    m = c.q.measures[0]
    value = D.div(D.mul(_double(active.copy()), 100), exp.Nullif(this=sz.copy(), expression=_num(0))) \
        if shape.value == "share" else active.copy()
    body = (exp.select(g.copy().as_(c.name(shape.cohort)), c.out("g", "age").as_(c.name(shape.age)),
                       value.as_(c.name(m.name)), sz.copy().as_(c.name(MEMBERS)))
            .from_(grid.subquery(c.name("g")))
            .join(cells.subquery(c.name("cl")), on=exp.and_(
                exp.EQ(this=c.out("cl", "cohort"), expression=g.copy()),
                exp.EQ(this=c.out("cl", "age"), expression=c.out("g", "age"))), join_type="left")
            .join(size.subquery(c.name("sz")), on=exp.EQ(this=c.out("sz", "cohort"), expression=g.copy())))
    columns = [c.group_column(gr) for gr in c.q.groups] + [OutColumn(m.name, m.label, "measure", m.format, None)]
    outer = exp.select(*[c.out("q", col.name) for col in columns], c.out("q", MEMBERS))
    return outer.from_(body.subquery(c.name("q"))), columns, "q"


# ── in force on each date ──────────────────────────────────────────────────


def _active(c: _Compiler, shape: ActiveShape) -> tuple[exp.Select, list[OutColumn], str]:
    """One read of the rows: for each period, how many were in force at its end (and when it began), started in
    it and ended in it; then a row per period, a level up."""
    part = shape.part
    rows = c.part_select(dataclasses.replace(part, groups=[], measures=[], row_level=True, distinct_only=False))
    group_exprs = [c.group_expr(g) for g in part.groups]
    wide = rows.select(*[e.copy().as_(c.name(g.name)) for e, g in zip(group_exprs, part.groups)], append=False)

    def before(use, day) -> exp.Expr:
        return c.range_condition(use, Range(None, day))

    def in_force(day) -> exp.Expression:
        # Started before ``day``, and not ended before it (an empty or placeholder end is no end).
        return exp.Case(ifs=[exp.If(this=before(shape.start, day), true=exp.Case(
            ifs=[exp.If(this=before(shape.end, day), true=_num(0))], default=_num(1)))], default=_num(0))

    def inside(use, lo, hi) -> exp.Expression:
        return _flag(c.range_condition(use, Range(lo, hi)))

    need = set(shape.columns)
    for i, (lo, hi) in enumerate(shape.periods):
        if "active" in need:
            wide = wide.select(exp.Sum(this=in_force(hi)).as_(c.name(f"a{i}")))
        if "starts" in need:
            wide = wide.select(exp.Sum(this=inside(shape.start, lo, hi)).as_(c.name(f"s{i}")))
        if "ends" in need or "end_rate" in need:
            wide = wide.select(exp.Sum(this=inside(shape.end, lo, hi)).as_(c.name(f"e{i}")))
        if "end_rate" in need:
            wide = wide.select(exp.Sum(this=in_force(lo)).as_(c.name(f"b{i}")))
    if group_exprs:
        wide = wide.group_by(*[e.copy() for e in group_exprs])
    period = next(g for g in c.q.groups if g.kind == "period")
    hidden = ["ended_count", "in_force_before"] if "end_rate" in need else []
    branches = []
    for i, (lo, _) in enumerate(shape.periods):
        w = lambda n: c.out("w", n)  # noqa: E731
        figures = {"active": w(f"a{i}"), "starts": w(f"s{i}"), "ends": w(f"e{i}"),
                   "end_rate": D.div(D.mul(_double(w(f"e{i}")), 100), exp.Nullif(this=w(f"b{i}"), expression=_num(0)))}
        branch = exp.select(D.date_literal(lo, c.d).as_(c.name(period.name)),
                            *[w(g).as_(c.name(g)) for g in shape.groups],
                            *[figures[k].as_(c.name(n)) for k, n in shape.columns.items()])
        if hidden:
            branch = branch.select(w(f"e{i}").as_(c.name(hidden[0])), w(f"b{i}").as_(c.name(hidden[1])))
        branches.append(branch.from_(exp.to_table(c.name("w").name).as_(c.name("w"))))
    union = branches[0]
    for b in branches[1:]:
        union = exp.union(union, b, distinct=False)
    columns = [c.group_column(g) for g in c.q.groups] + [OutColumn(m.name, m.label, "measure", m.format, None)
                                                         for m in c.q.measures]
    outer = exp.select(*[c.out("q", col.name) for col in columns], *[c.out("q", h) for h in hidden])
    outer = outer.from_(union.subquery(c.name("q"))).with_(c.name("w"), as_=wide)
    return outer, columns, "q"


# ── on time, and ageing ────────────────────────────────────────────────────

LATE = "late_count"


def _on_time(c: _Compiler, shape: OnTimeShape) -> tuple[exp.Select, list[OutColumn], str]:
    """Each row done in the window, against its due date: the share on time, how many had both dates and how many
    were late."""
    part = shape.part
    assert shape.done is not None
    rows = c.part_select(dataclasses.replace(part, groups=[], measures=[], row_level=True, distinct_only=False))
    group_exprs = [c.group_expr(g) for g in part.groups]
    late = c.days(shape.due, shape.done)          # NULL without both dates
    on_time = exp.Case(ifs=[exp.If(this=exp.Is(this=late.copy(), expression=exp.Null()), true=exp.Null()),
                            exp.If(this=exp.LTE(this=late.copy(), expression=_num(shape.grace)),
                                   true=exp.Literal.number("100.0"))], default=exp.Literal.number("0.0"))
    m = c.q.measures[0]
    body = rows.select(*[e.copy().as_(c.name(g.name)) for e, g in zip(group_exprs, part.groups)],
                       exp.Avg(this=on_time).as_(c.name(m.name)), exp.Count(this=late.copy()).as_(c.name(MEMBERS)),
                       exp.Sum(this=_flag(exp.GT(this=late.copy(), expression=_num(shape.grace)))).as_(c.name(LATE)),
                       append=False)
    if group_exprs:
        body = body.group_by(*[e.copy() for e in group_exprs])
    columns = [c.group_column(g) for g in c.q.groups] + [OutColumn(m.name, m.label, "measure", m.format, None)]
    outer = exp.select(*[c.out("q", col.name) for col in columns], c.out("q", MEMBERS), c.out("q", LATE))
    return outer.from_(body.subquery(c.name("q"))), columns, "q"


def _ageing(c: _Compiler, shape: OnTimeShape) -> tuple[exp.Select, list[OutColumn], str]:
    """The rows due and not done, put in ranges of days past due on the day they are aged on."""
    part = shape.part
    rows = c.part_select(dataclasses.replace(part, groups=[], measures=[], row_level=True, distinct_only=False))
    conds: list[exp.Expr] = [c.range_condition(shape.due, Range(None, None))]   # a real due date
    if shape.done is not None:
        # Not done: no real done date (empty, or a placeholder).
        conds.append(exp.EQ(this=_flag(c.range_condition(shape.done, Range(None, None))), expression=_num(0)))
    if shape.done_elsewhere is not None:
        table, pointer, key = shape.done_elsewhere
        conds.append(exp.not_(exp.Exists(this=exp.select(_num(1)).from_(c.table(table, "done_rows")).where(
            equal_keys(c.model, c.col("done_rows", pointer), pointer, c.col(part.alias, key), key, c.d)))))
    days = D.days_between(c.date_value(shape.due), D.date_literal(shape.as_of, c.d), c.d)
    edges = [(0, AGEING[0]), (30, AGEING[1]), (60, AGEING[2]), (90, AGEING[3])]
    band = exp.Case(ifs=[exp.If(this=exp.LTE(this=days.copy(), expression=_num(n)), true=exp.Literal.string(label))
                         for n, label in edges], default=exp.Literal.string(AGEING[4]))
    group_exprs = [band, *[c.group_expr(g) for g in part.groups]]
    names = [shape.band, *[g.name for g in part.groups]]
    figures = [exp.Count(this=exp.Star()).as_(c.name(shape.columns["open"]))]
    if "measure" in shape.columns:
        m = next(o for o in c.q.measures if o.name == shape.columns["measure"])
        figures.append(c.aggregate(m.expr, part).as_(c.name(m.name)))
    body = rows.select(*[e.copy().as_(c.name(n)) for e, n in zip(group_exprs, names)], *figures, append=False)
    body = body.where(exp.and_(*conds)).group_by(*[e.copy() for e in group_exprs])
    columns = [c.group_column(g) for g in c.q.groups] + [OutColumn(o.name, o.label, "measure", o.format,
                                                                   o.measure.key if o.measure else None)
                                                         for o in c.q.measures]
    outer = exp.select(*[c.out("q", col.name) for col in columns])
    return outer.from_(body.subquery(c.name("q"))), columns, "q"


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
