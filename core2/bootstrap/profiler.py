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
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlglot import exp

from core2.bootstrap.inventory import InvColumn, InvTable
from core2.bootstrap.journal import answers, journal_of, reason
from core2.model.schema import ColumnProfile, TopValue
from core2.warehouse import dialect as D
from core2.warehouse.runner import Warehouse

log = logging.getLogger("querybot.core2")


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
    # Columns the warehouse would not compute, with its reason: left out, never guessed.
    unread: dict[str, str] = field(default_factory=dict)


def _num(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, decimal.Decimal)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


# Dates outside these are placeholders a system writes for "none" or "still open" (1900-01-01,
# 1753-01-01, 9999-12-31): the same bounds the compiler keeps every date question within.
FIRST_REAL_DAY = dt.date(1901, 1, 1)
AFTER_REAL_DAY = dt.date(9000, 1, 1)


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


def _readable(column: InvColumn, dialect: str) -> exp.Expression:
    """The column as the warehouse can count, compare and sort it.

    Azure SQL cannot compare or sort its old text and ntext types (no DISTINCT,
    MIN, MAX or LEN on them), and reads a uniqueidentifier best as its text.
    """
    c = exp.column(D.ident(column.name, dialect))
    if dialect == "tsql":
        base = column.raw_type.strip().upper().split("(")[0].strip()
        if base in ("TEXT", "NTEXT"):
            return exp.Cast(this=c, to=exp.DataType.build("NVARCHAR(MAX)", dialect="tsql"))
        if base == "UNIQUEIDENTIFIER":
            return exp.Cast(this=c, to=exp.DataType.build("CHAR(36)"))
    return c


_UNREAD: Any = object()   # a value the warehouse would not compute


def _select(warehouse: Warehouse, items: list[tuple[str, str, exp.Expression]], source: str, dialect: str,
            unread: dict[str, str], table: str) -> list[Any]:
    """Run [(column, stat, expression)] as one SELECT on the table; the values in the same order.

    A warehouse can refuse one column's expression: a type it cannot compare,
    arithmetic it will not do on that type. With every column in one query, that
    refusal stopped the table and with it the whole Learn. So each column is then
    asked alone. One it still refuses is left out: its values come back as
    _UNREAD and it is named in ``unread`` with the warehouse's reason. When nothing
    can be read, the first error is raised: that is the table or the connection.
    """
    def run(part: list[tuple[str, str, exp.Expression]]) -> list[Any]:
        query = exp.select(*[e.as_(f"s{i}") for i, (_, _, e) in enumerate(part)]).from_(exp.to_table("__SRC__"))
        return list(warehouse.query(query.sql(dialect=dialect).replace("__SRC__", source, 1)).rows[0])

    try:
        return run(items)
    except Exception as first:  # noqa: BLE001 - which column it was is found below, and said
        if not answers(warehouse):
            raise
        log.warning("core2: %s refused a profile query; reading its columns one at a time: %s", table, reason(first))
        values: list[Any] = [_UNREAD] * len(items)
        groups: dict[str, list[int]] = {}
        for i, (name, _, _) in enumerate(items):
            groups.setdefault(name, []).append(i)
        read_any = False
        refused: list[tuple[str, Exception]] = []
        for name, positions in groups.items():
            try:
                got = run([items[i] for i in positions])
            except Exception as error:  # noqa: BLE001 - this column is left out and named
                if name:
                    refused.append((name, error))
                continue
            read_any = True
            for i, value in zip(positions, got):
                values[i] = value
        if not read_any:
            raise first   # the table itself, not a column: its caller decides
        for name, refusal in refused:
            if name not in unread:
                unread[name] = reason(refusal)
                journal_of(warehouse).leave_out(f"{table}.{name}", refusal)
        return values


def _stats(column: InvColumn, dialect: str, exact: bool) -> list[tuple[str, exp.Expression]]:
    c = _readable(column, dialect)
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
    month = _mod(D.floor_div(c.copy(), 100), 100)
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
    if sampled:
        try:
            warehouse.query(f"SELECT COUNT(1) FROM {source}")
        except Exception as error:  # noqa: BLE001 - a view is not sampled on Azure SQL: its first rows then
            if not answers(warehouse):
                raise
            name = D.table_sql(table.database, table.schema, table.name, dialect)
            source = D.aliased(D.first_rows(name, dialect, rows=options.sample_rows), "first_rows", dialect)
            journal_of(warehouse).step(f"{table.name} cannot be sampled ({reason(error)}): "
                                       f"reading its first {options.sample_rows:,} rows instead")
    exact = not sampled and rows <= options.exact_distinct_up_to
    profiles = {c.name: ColumnProfile(rows=rows, sampled=sampled, distinct_is_approx=not exact) for c in table.columns}
    unread: dict[str, str] = {}

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
        values = _select(warehouse, [(name, stat, e) for (name, stat), e in zip(plan, selects)], source, dialect,
                         unread, table.name)
        for (name, stat), value in zip(plan, values):
            if value is _UNREAD:
                continue
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
            shaped.append((column.name, "whole_year", _count_if(exp.and_(shape.copy(), exp.EQ(
                this=_mod(c.copy(), 100), expression=exp.Literal.number(0))))))
        else:
            continue
        if shaped[-1][1] == "whole_year":
            # The first and last real months: a whole-year row (month 00) is not a month.
            shape = exp.and_(shape, exp.Between(this=_mod(c.copy(), 100), low=exp.Literal.number(1),
                                                high=exp.Literal.number(12)))
        real = exp.and_(shape.copy(), exp.LT(this=c.copy(), expression=exp.Literal.number(
            99991231 if shaped[-1][1] == "yyyymmdd" else 999912)), exp.GT(this=c.copy(), expression=exp.Literal.number(
                19000101 if shaped[-1][1] == "yyyymmdd" else 190001)))
        first = exp.Case(ifs=[exp.If(this=real, true=c.copy())])
        shaped.append((column.name, "min_valid", exp.Min(this=first)))
        shaped.append((column.name, "max_valid", exp.Max(this=first.copy())))
    if shaped:
        values = _select(warehouse, shaped, source, dialect, unread, table.name)
        for (name, pattern, _), value in zip(shaped, values):
            if value is _UNREAD:
                continue
            p = profiles[name]
            if pattern in ("min_valid", "max_valid"):
                setattr(p, "date_min" if pattern == "min_valid" else "date_max", _text(value))
                continue
            if pattern == "whole_year":
                count = int(value or 0)
                p.whole_year_rows = int(round(count * rows / sample_rows)) if sampled and sample_rows else count
                continue
            valid = int(value or 0)
            if sampled:
                valid = int(round(valid * rows / sample_rows)) if sample_rows else valid
            # Placeholder keys (-1, 0, 19000101, 99991231) are not dates but do not
            # disqualify a date key either: most values must be dates.
            if p.non_null and valid >= 0.9 * p.non_null:
                p.pattern = pattern

    # 2b. placeholder dates in date columns: counted apart, and left out of the first and last dates
    dated: list[tuple[str, str, exp.Expression]] = []
    for column in table.columns:
        p = profiles[column.name]
        if column.data_type not in ("date", "timestamp") or not p.min or not p.max:
            continue
        if FIRST_REAL_DAY.isoformat() <= p.min[:10] and p.max[:10] < AFTER_REAL_DAY.isoformat():
            continue
        c = exp.column(D.ident(column.name, dialect))
        real = exp.and_(exp.GTE(this=c.copy(), expression=D.date_literal(FIRST_REAL_DAY, dialect)),
                        exp.LT(this=c.copy(), expression=D.date_literal(AFTER_REAL_DAY, dialect)))
        other = exp.and_(exp.not_(exp.Is(this=c.copy(), expression=exp.Null())), exp.not_(real.copy()))
        dated.append((column.name, "placeholders", exp.Sum(this=exp.Case(
            ifs=[exp.If(this=other, true=exp.Literal.number(1))], default=exp.Literal.number(0)))))
        real_day = exp.Case(ifs=[exp.If(this=real, true=c.copy())])
        dated.append((column.name, "min_valid", exp.Min(this=real_day)))
        dated.append((column.name, "max_valid", exp.Max(this=real_day.copy())))
    if dated:
        values = _select(warehouse, dated, source, dialect, unread, table.name)
        for (name, stat, _), value in zip(dated, values):
            if value is _UNREAD:
                continue
            p = profiles[name]
            if stat == "placeholders":
                count = int(value or 0)
                p.placeholder_rows = int(round(count * rows / sample_rows)) if sampled and sample_rows else count
            else:
                setattr(p, "date_min" if stat == "min_valid" else "date_max", _text(value))

    # 3. common values
    low = [c for c in table.columns
           if c.data_type in ("text", "integer", "boolean")
           and 0 < profiles[c.name].distinct <= options.low_cardinality
           and options.values_allowed(table.key, c.name)]
    if low:
        branches = []
        for i, column in enumerate(low):
            readable = _readable(column, dialect)
            branches.append(exp.select(exp.Literal.number(i).as_("k"), D.as_text(readable.copy(), dialect).as_("v"),
                                       exp.Count(this=exp.Literal.number(1)).as_("n"))
                            .from_(exp.to_table("__SRC__")).group_by(readable.copy()))

        def common(part: list[exp.Select]) -> list[tuple]:
            union: exp.Expression = part[0]
            for branch in part[1:]:
                union = exp.union(union, branch, distinct=False)
            return warehouse.query(union.sql(dialect=dialect).replace("__SRC__", source)).rows

        try:
            counted = common(branches)
        except Exception as error:  # noqa: BLE001 - common values are a help; one column's refusal loses only its own
            if not answers(warehouse):
                raise
            log.warning("core2: %s refused the common-values query; reading them one column at a time: %s",
                        table.name, reason(error))
            counted = []
            for column, branch in zip(low, branches):
                try:
                    counted += common([branch])
                except Exception as alone:  # noqa: BLE001 - this column shows no common values
                    journal_of(warehouse).leave_out(f"the common values of {table.name}.{column.name}", alone)
        found: dict[int, list[TopValue]] = {}
        for k, v, n in counted:
            found.setdefault(int(k), []).append(TopValue(value=None if v is None else str(v), count=int(n)))
        for i, column in enumerate(low):
            profiles[column.name].top = sorted(found.get(i, []), key=lambda t: (-t.count, t.value or ""))

    for column in table.columns:
        p = profiles[column.name]
        if p.pattern is None:
            p.pattern = _pattern(column, p)
    return TableProfile(rows=rows, sampled=sampled, columns=profiles, unread=unread)


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
