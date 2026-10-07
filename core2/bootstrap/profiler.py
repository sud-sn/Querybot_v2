"""Profiling: aggregates computed inside the warehouse, one table at a time.

Three queries per table at most, each generated here (never by a model) and
checked read-only before it is sent:

1. counts, distinct counts, ranges, signs, whole-number shares, text lengths and
   time-of-day shares for every column (in chunks for very wide tables);
2. date-shape checks for integer columns whose range could be ``yyyymmdd`` or
   ``yyyymm`` numbers;
3. the most common values of low-cardinality columns, only where the caller's
   ``values_allowed`` says common values may be read.

Tables over ``sample_threshold`` rows are profiled on a sample; their distinct
counts are then estimates and are marked so.
"""

from __future__ import annotations

import datetime as dt
import decimal
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlglot import exp

from core2.bootstrap.inventory import InvColumn, InvTable
from core2.model.schema import ColumnProfile, TopValue
from core2.warehouse import dialect as D
from core2.warehouse.runner import Warehouse


@dataclass
class ProfileOptions:
    sample_threshold: int = 5_000_000
    sample_rows: int = 1_000_000
    exact_distinct_up_to: int = 2_000_000
    low_cardinality: int = 50
    columns_per_query: int = 40
    values_allowed: Callable[[str, str], bool] = field(default=lambda table_key, column: True)


@dataclass
class TableProfile:
    rows: int
    sampled: bool
    columns: dict[str, ColumnProfile]


def _num(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, decimal.Decimal)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        value = value.normalize() if value == value.to_integral() else value
        return format(value, "f")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _count_if(condition: exp.Expr) -> exp.Expression:
    return exp.Sum(this=exp.Case(ifs=[exp.If(this=condition, true=exp.Literal.number(1))],
                                 default=exp.Literal.number(0)))


def _source(table: InvTable, dialect: str, rows: int, options: ProfileOptions) -> tuple[str, bool]:
    name = D.table_sql(table.database, table.schema, table.name, dialect)
    if rows > options.sample_threshold:
        return f"{name} {D.sample_clause(dialect, rows=options.sample_rows, total_rows=rows)}", True
    return name, False


def _stats(column: InvColumn, dialect: str, exact: bool) -> list[tuple[str, exp.Expression]]:
    c = exp.column(D.ident(column.name, dialect))
    out: list[tuple[str, exp.Expression]] = [("non_null", exp.Count(this=c.copy()))]
    kind = column.data_type
    if kind != "other":
        distinct = (exp.Count(this=exp.Distinct(expressions=[c.copy()])) if exact
                    else D.approx_distinct(c.copy(), dialect))
        out.append(("distinct", distinct))
    if kind in ("integer", "decimal", "float"):
        out += [("min", exp.Min(this=c.copy())), ("max", exp.Max(this=c.copy())),
                ("negatives", _count_if(exp.LT(this=c.copy(), expression=exp.Literal.number(0)))),
                ("zeros", _count_if(exp.EQ(this=c.copy(), expression=exp.Literal.number(0))))]
        if kind != "integer":
            out.append(("whole", _count_if(exp.EQ(this=c.copy(), expression=exp.Floor(this=c.copy())))))
    elif kind == "text":
        length = D.text_length(c.copy(), dialect)
        out += [("min", exp.Min(this=c.copy())), ("max", exp.Max(this=c.copy())),
                ("min_len", exp.Min(this=length.copy())),
                ("avg_len", exp.Avg(this=exp.Cast(this=length.copy(), to=exp.DataType.build("FLOAT")))),
                ("max_len", exp.Max(this=length.copy()))]
    elif kind == "date":
        out += [("min", exp.Min(this=c.copy())), ("max", exp.Max(this=c.copy()))]
    elif kind == "timestamp":
        out += [("min", exp.Min(this=c.copy())), ("max", exp.Max(this=c.copy())),
                ("timed", _count_if(exp.NEQ(this=c.copy(), expression=D.day_start(c.copy(), dialect))))]
    return out


def _between(value: exp.Expression, low: int, high: int) -> exp.Expr:
    return exp.Between(this=value, low=exp.Literal.number(low), high=exp.Literal.number(high))


def _mod(value: exp.Expression, n: int) -> exp.Expression:
    return exp.Mod(this=value, expression=exp.Literal.number(n))


def _yyyymmdd(c: exp.Expression) -> exp.Expr:
    month = _mod(exp.Floor(this=exp.Div(this=c.copy(), expression=exp.Literal.number(100))), 100)
    return exp.and_(_between(c.copy(), 19000101, 21001231), _between(_mod(c.copy(), 100), 1, 31),
                    _between(month, 1, 12))


def _yyyymm(c: exp.Expression) -> exp.Expr:
    # Month 00 is allowed: period tables often hold a row for the whole year (202200).
    return exp.and_(_between(c.copy(), 190000, 210012), _between(_mod(c.copy(), 100), 0, 12))


def profile_table(warehouse: Warehouse, table: InvTable, options: ProfileOptions | None = None) -> TableProfile:
    options = options or ProfileOptions()
    dialect = warehouse.dialect
    rows = table.row_count
    if rows is None:
        rows = int(warehouse.query(f"SELECT COUNT(1) FROM {D.table_sql(table.database, table.schema, table.name, dialect)}").rows[0][0])
    source, sampled = _source(table, dialect, rows, options)
    exact = not sampled and rows <= options.exact_distinct_up_to
    profiles = {c.name: ColumnProfile(rows=rows, sampled=sampled, distinct_is_approx=not exact) for c in table.columns}

    # 1. aggregates
    chunks = [table.columns[i:i + options.columns_per_query]
              for i in range(0, len(table.columns), options.columns_per_query)] or [[]]
    sample_rows = rows
    for chunk in chunks:
        plan: list[tuple[str, str]] = [("", "__rows")]
        selects: list[exp.Expression] = [exp.Count(this=exp.Literal.number(1))]
        for column in chunk:
            for stat, expression in _stats(column, dialect, exact):
                plan.append((column.name, stat))
                selects.append(expression)
        query = exp.select(*[e.as_(f"s{i}") for i, e in enumerate(selects)]).from_(exp.to_table("__SRC__"))
        sql = query.sql(dialect=dialect).replace("__SRC__", source, 1)
        values = warehouse.query(sql).rows[0]
        for (name, stat), value in zip(plan, values):
            if stat == "__rows":
                sample_rows = int(value or 0)
                continue
            p = profiles[name]
            if stat in ("non_null", "distinct", "negatives", "zeros", "timed", "whole"):
                number = int(value or 0)
                if stat == "non_null":
                    p.non_null = number
                elif stat == "distinct":
                    p.distinct = number
                elif stat == "negatives":
                    p.negatives = number
                elif stat == "zeros":
                    p.zeros = number
                elif stat == "timed":
                    p.time_share = number / p.non_null if p.non_null else 0.0
                elif stat == "whole":
                    p.integer_share = number / p.non_null if p.non_null else None
            elif stat in ("min", "max"):
                setattr(p, stat, _text(value))
                setattr(p, f"{stat}_num", _num(value) if not isinstance(value, (dt.date, str)) else None)
            elif stat in ("min_len", "avg_len", "max_len"):
                setattr(p, stat, _num(value))
    if sampled:
        # Counts were taken on the sample: scale them to the table.
        scale = rows / sample_rows if sample_rows else 1.0
        for p in profiles.values():
            p.non_null = int(round(p.non_null * scale))
            for name in ("negatives", "zeros"):
                if getattr(p, name) is not None:
                    setattr(p, name, int(round(getattr(p, name) * scale)))
    for column in table.columns:
        if column.data_type == "integer":
            profiles[column.name].integer_share = 1.0 if profiles[column.name].non_null else None

    # 2. date shapes of integers
    shaped: list[tuple[str, str, exp.Expression]] = []
    for column in table.columns:
        p = profiles[column.name]
        if column.data_type not in ("integer", "decimal") or p.min_num is None or p.max_num is None:
            continue
        c = exp.column(D.ident(column.name, dialect))
        if p.max_num >= 19000101 and p.max_num <= 99991231:
            shape = _yyyymmdd(c)
            shaped.append((column.name, "yyyymmdd", _count_if(shape)))
        elif p.max_num >= 190001 and p.max_num <= 999912:
            shape = _yyyymm(c)
            shaped.append((column.name, "yyyymm", _count_if(shape)))
        else:
            continue
        real = exp.and_(shape.copy(), exp.LT(this=c.copy(), expression=exp.Literal.number(
            99991231 if shaped[-1][1] == "yyyymmdd" else 999912)), exp.GT(this=c.copy(), expression=exp.Literal.number(
                19000101 if shaped[-1][1] == "yyyymmdd" else 190001)))
        first = exp.Case(ifs=[exp.If(this=real, true=c.copy())])
        shaped.append((column.name, "min_valid", exp.Min(this=first)))
        shaped.append((column.name, "max_valid", exp.Max(this=first.copy())))
    if shaped:
        query = exp.select(*[e.as_(f"s{i}") for i, (_, _, e) in enumerate(shaped)]).from_(exp.to_table("__SRC__"))
        values = warehouse.query(query.sql(dialect=dialect).replace("__SRC__", source, 1)).rows[0]
        for (name, pattern, _), value in zip(shaped, values):
            p = profiles[name]
            if pattern in ("min_valid", "max_valid"):
                setattr(p, "date_min" if pattern == "min_valid" else "date_max", _text(value))
                continue
            valid = int(value or 0)
            if sampled:
                valid = int(round(valid * rows / sample_rows)) if sample_rows else valid
            # Placeholder keys (-1, 0, 19000101, 99991231) are not dates but do not
            # disqualify a date key either: most values must be dates.
            if p.non_null and valid >= 0.9 * p.non_null:
                p.pattern = pattern

    # 3. common values
    low = [c for c in table.columns
           if c.data_type in ("text", "integer", "boolean")
           and 0 < profiles[c.name].distinct <= options.low_cardinality
           and options.values_allowed(table.key, c.name)]
    if low:
        branches = []
        for i, column in enumerate(low):
            c = exp.column(D.ident(column.name, dialect))
            branches.append(exp.select(exp.Literal.number(i).as_("k"), D.as_text(c.copy(), dialect).as_("v"),
                                       exp.Count(this=exp.Literal.number(1)).as_("n"))
                            .from_(exp.to_table("__SRC__")).group_by(c.copy()))
        union: exp.Expression = branches[0]
        for branch in branches[1:]:
            union = exp.union(union, branch, distinct=False)
        sql = union.sql(dialect=dialect).replace("__SRC__", source)
        found: dict[int, list[TopValue]] = {}
        for k, v, n in warehouse.query(sql).rows:
            found.setdefault(int(k), []).append(TopValue(value=None if v is None else str(v), count=int(n)))
        for i, column in enumerate(low):
            profiles[column.name].top = sorted(found.get(i, []), key=lambda t: (-t.count, t.value or ""))

    for column in table.columns:
        p = profiles[column.name]
        if p.pattern is None:
            p.pattern = _pattern(column, p)
    return TableProfile(rows=rows, sampled=sampled, columns=profiles)


_YES_NO = {"y", "n", "yes", "no", "t", "f", "true", "false", "oui", "non"}


def _pattern(column: InvColumn, p: ColumnProfile) -> str | None:
    kind = column.data_type
    if kind in ("date", "timestamp"):
        return kind
    if kind == "boolean":
        return "flag"
    if kind == "integer":
        if p.min_num == 0 and p.max_num == 1:
            return "flag01"
        if p.min_num is not None and p.max_num is not None and 1900 <= p.min_num and p.max_num <= 2100 \
                and p.distinct <= 200:
            return "yyyy"
        return "number"
    if kind in ("decimal", "float"):
        return "number"
    if kind == "text":
        tops = {(t.value or "").strip().lower() for t in (p.top or [])}
        if tops and tops <= _YES_NO and p.distinct <= 2:
            return "flag_yn"
        if p.avg_len is not None and p.avg_len > 60:
            return "free_text"
        if p.max_len is not None and p.max_len <= 12:
            return "code"
        return "text"
    return None
