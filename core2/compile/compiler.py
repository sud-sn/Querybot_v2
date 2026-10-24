"""From a logical query to SQL, through a syntax tree, for each warehouse.

Nothing here decides meaning: the resolver did. The compiler only spells it:

* every table joined once per path under its own alias, a calendar once per date
  role (inner: a row without a real date is not counted on a date);
* windows as half-open ranges on the stored date, so they use the warehouse's
  indexes and pruning; date numbers (yyyymmdd, yyyymm) compared as numbers;
* placeholder dates (the calendar's Unknown row, 0, 19000101, 99991231, month 00)
  never reach a period: they fall outside every range, and a date number that is
  no date converts to NULL instead of failing the query;
* periods are their first day; weeks start on Monday on every warehouse; fiscal
  quarters and years start in the model's fiscal month;
* a snapshot is read on its last day in each period (and in each compared window);
* two measure tables are aggregated separately and lined up on their groupings
  with a null-safe full join; a measure over both (sales less refunds) is worked
  out once they are lined up, from each table's own total;
* a link's own conditions ("only the current customer row") are part of its join;
* a comparison is two conditional aggregates over one scan, with the change and
  the change in percent computed beside them; a share is of the shown total;
* one outer query sorts (nulls last everywhere, ties broken by the groupings) and
  caps the rows.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.scope import traverse_scope

from core2.bootstrap.inventory import normalize_type
from core2.model import formula
from core2.model.schema import AggExpr, Join, MeasureExpr, OpExpr, RefExpr, SemanticModel, SqlExpr, value_names
from core2.resolve.resolver import (Combined, DateUse, DaysBetween, DaysPred, Group, Joined, Logical, MemberAbsent,
                                    Part, PartGroup, Semi)
from core2.resolve.time import Range, add_units
from core2.warehouse import dialect as D

_LOW = dt.date(1901, 1, 1)       # dates before this are placeholders
_HIGH = dt.date(9000, 1, 1)      # and from this on
_NUMBER_TYPES = ("integer", "decimal", "float")
_GENERATED = re.compile(r"[a-z_][a-z0-9_]*")


class CompileError(ValueError):
    pass


@dataclass
class OutColumn:
    name: str
    label: str
    role: str            # period | attribute | member_code | time | measure | prior | change | pct_change | share
    format: str = ""
    measure: str | None = None      # the measure's key, for measures and their derived columns
    grain: str | None = None        # for the period
    flag: bool = False              # a yes/no grouping (its two members are a thing and its opposite)
    numbered: bool = False          # its members are numbers (fill number 0, GL account 4000), said with its name
    ordinal: bool = False           # its members are amounts (transit days, fill number): drawn in number order
    bands: list[str] = field(default_factory=list)   # a range grouping: its ranges' names, in their own order
    computed: bool = False          # worked out from the rows after the query (a running total): not in its SQL
    missing: str = ""               # the name of the rows with no member, where the link names none ("No customer
                                    # recorded"); "Unknown" otherwise
    dated: bool = False             # a list's rows by the day of their own date, headed by its name ("Order date")


@dataclass
class Compiled:
    sql: str
    dialect: str
    columns: list[OutColumn] = field(default_factory=list)
    row_cap: int = 0                # the query returns at most this many rows


def _num(value: int | float) -> exp.Expression:
    return exp.Literal.number(value)


def _double(value: exp.Expression) -> exp.Expression:
    return exp.Cast(this=value, to=exp.DataType.build("DOUBLE"))


def text_date(column: Any) -> bool:
    """A column the database keeps as text that Learn found holds dates: read as a date wherever it is used."""
    return column.data_type in ("date", "timestamp") and normalize_type(column.raw_type) == "text"


def column_sql(model: SemanticModel, key: str, table: exp.Identifier | str | None, dialect: str) -> exp.Expression:
    """A column of the model as the warehouse reads it; a name held in parts ("first" and "last") as the parts
    joined by a space, an empty part left out."""
    column = model.columns[key]
    if not column.parts:
        stored = exp.column(D.ident(column.name, dialect), table=table)
        return D.text_date(stored, column.data_type, dialect) if text_date(column) else stored
    joined: exp.Expression | None = None
    for part in column.parts:
        piece = exp.Coalesce(this=exp.column(D.ident(model.columns[part].name, dialect), table=table),
                             expressions=[exp.Literal.string("")])
        joined = piece if joined is None else exp.DPipe(this=exp.DPipe(this=joined, expression=exp.Literal.string(" ")),
                                                        expression=piece)
    return exp.Trim(this=joined)


def equal_keys(model: SemanticModel, left: exp.Expression, left_key: str, right: exp.Expression, right_key: str,
               dialect: str) -> exp.Expr:
    """The two sides of a link equal: a code kept as text ('0007') against a number key (7) compared without
    leading zeros (D.same_number), anything else as it is."""
    a, b = model.columns[left_key].data_type, model.columns[right_key].data_type
    if {a, b} == {"text", "integer"}:
        return D.same_number(left, right, dialect) if a == "text" else D.same_number(right, left, dialect)
    return exp.EQ(this=left, expression=right)


class _Compiler:
    def __init__(self, logical: Logical, model: SemanticModel, dialect: str):
        self.q = logical
        self.model = model
        self.d = dialect

    # ── names ──────────────────────────────────────────────────────────────
    def name(self, text: str) -> exp.Identifier:
        """A name this query makes up (an alias, an output column): plain unless reserved.

        Stored names keep their stored spelling (``D.ident``); made-up ones are
        lower-case slugs, written the same way wherever they appear, so a warehouse
        that folds unquoted names to upper case reads them all alike.
        """
        return exp.to_identifier(text, quoted=text.upper() in D.RESERVED or not _GENERATED.fullmatch(text))

    def col(self, alias: str, column_key: str) -> exp.Expression:
        return column_sql(self.model, column_key, self.name(alias), self.d)

    def out(self, alias: str, name: str) -> exp.Column:
        return exp.column(self.name(name), table=self.name(alias))

    def table(self, table_key: str, alias: str) -> exp.Table:
        t = self.model.tables[table_key]
        table = D.table_expr(t.database, t.schema_name, t.name, self.d)
        table.set("alias", exp.TableAlias(this=self.name(alias)))
        return table

    def on(self, j: Joined) -> exp.Expr:
        keys = [equal_keys(self.model, self.col(left, lc), lc, self.col(j.alias, rc), rc, self.d) for left, lc, rc in j.on]
        kept = [self.predicate(self.col(j.alias, column), op, list(values), column) for column, op, values in j.conds]
        return exp.and_(*keys, *kept)

    def exists(self, semi: Semi) -> exp.Expr:
        """A filter across a bridge: the part's row has at least one match (kept once, however many match)."""
        first = semi.joins[0]
        inner = exp.select(_num(1)).from_(self.table(first.table, first.alias))
        for j in semi.joins[1:]:
            inner = inner.join(self.table(j.table, j.alias), on=self.on(j), join_type="inner")
        kept = [self.on(first)] + [self.predicate(self.col(p.alias, p.column), p.op, p.values, p.column)
                                   for p in semi.preds]
        found = exp.Exists(this=inner.where(exp.and_(*kept)))
        return exp.Not(this=found) if semi.negated else found

    def at(self, part: Part, column_key: str) -> exp.Column:
        """A column a measure reads: on the part's own table, or where the part reached its table."""
        owner = self.model.columns[column_key].table
        return self.col(part.reached.get(owner, part.alias) if owner != part.table else part.alias, column_key)

    # ── values ─────────────────────────────────────────────────────────────
    def literal(self, value: object, column_key: str | None = None) -> exp.Expression:
        data_type = self.model.columns[column_key].data_type if column_key else ""
        if isinstance(value, bool):
            if self.d in ("tsql", "oracle"):
                return _num(1 if value else 0)
            return exp.Boolean(this=value)
        if isinstance(value, dt.date):
            return D.date_literal(value, self.d)
        if isinstance(value, str) and data_type in _NUMBER_TYPES:
            try:
                number = float(value.strip())
            except ValueError:
                return exp.Literal.string(value)
            return _num(int(number) if number.is_integer() and data_type == "integer" else number)
        if isinstance(value, str) and data_type in ("date", "timestamp"):
            try:
                return D.date_literal(dt.date.fromisoformat(value.strip()[:10]), self.d)
            except ValueError:
                return exp.Literal.string(value)
        if isinstance(value, (int, float)):
            if data_type == "text":
                return exp.Literal.string(str(value))
            return _num(value)
        return exp.Literal.string(str(value))

    def predicate(self, column: exp.Expression, op: str, values: list, column_key: str | None = None) -> exp.Expr:
        vals = [self.literal(v, column_key) for v in values]
        if op in ("eq", "in", "ne", "not_in"):
            if not vals:
                raise CompileError(f"a {op} filter needs a value")
            stored = self.model.columns.get(column_key or "")
            if stored is not None and stored.data_type == "text" and not stored.values_allowed \
                    and all(isinstance(v, str) for v in values):
                # A name on a column whose members are never read (a product's description, a country the old
                # masking rule keeps): the stored spelling is not known to put it in, so "Regency Cakestand 3 Tier"
                # finds REGENCY CAKESTAND 3 TIER, in either case.
                column = exp.Upper(this=column)
                vals = [exp.Literal.string(str(v).upper()) for v in values]
            inner = exp.EQ(this=column, expression=vals[0]) if len(vals) == 1 else exp.In(this=column, expressions=vals)
            if op in ("eq", "in"):
                return inner
            # "not X" keeps the rows with no value at all
            return exp.or_(exp.not_(inner), exp.Is(this=column.copy(), expression=exp.Null()))
        if op in ("gt", "gte", "lt", "lte"):
            if not vals:
                raise CompileError(f"a {op} filter needs a value")
            kind = {"gt": exp.GT, "gte": exp.GTE, "lt": exp.LT, "lte": exp.LTE}[op]
            return kind(this=column, expression=vals[0])
        if op == "between":
            if len(vals) != 2:
                raise CompileError("a between filter needs two values")
            return exp.Between(this=column, low=vals[0], high=vals[1])
        if op in ("contains", "starts_with"):
            text = str(values[0]).upper() if values else ""
            pattern = f"%{text}%" if op == "contains" else f"{text}%"
            return exp.Like(this=exp.Upper(this=column), expression=exp.Literal.string(pattern))
        if op == "is_null":
            return exp.Is(this=column, expression=exp.Null())
        if op == "not_null":
            return exp.not_(exp.Is(this=column, expression=exp.Null()))
        raise CompileError(f"unsupported filter {op}")

    # ── dates ──────────────────────────────────────────────────────────────
    def stored_date(self, use: DateUse) -> exp.Expression:
        """The stored date (a DATE, a timestamp, or an order-preserving date number). A year and a quarter kept as
        two numbers are their quarter's first day as a yyyymmdd number, in plain arithmetic every warehouse writes
        alike: a fiscal year named by the year it ends starts in the workspace's fiscal month of the year before."""
        if use.mode != "year_quarter":
            return self.col(use.alias, use.column)
        year, quarter = self.col(use.alias, use.column), self.col(use.alias, use.quarter or "")
        # The workspace's fiscal year, also for a question worked out without one (a list of members).
        start = (self.q.fiscal_start or self.model.settings.fiscal_year_start_month or 1) if use.role.fiscal else 1
        # Months from January of the year it starts in: a quarter starting past December is in the next year.
        offset = D.add(_num(start - 1), exp.Mul(this=_num(3), expression=exp.Paren(this=D.sub(quarter, _num(1)))))
        later = exp.Case(ifs=[exp.If(this=exp.GTE(this=offset.copy(), expression=_num(12)), true=_num(1))],
                         default=_num(0))
        first_year = D.sub(year, _num(1)) if start != 1 else year
        calendar_year = D.add(first_year, later)
        month = D.sub(D.add(offset.copy(), _num(1)), exp.Mul(this=_num(12), expression=later.copy()))
        return D.add(D.add(exp.Mul(this=exp.Paren(this=calendar_year), expression=_num(10000)),
                           exp.Mul(this=exp.Paren(this=month), expression=_num(100))), _num(1))

    def date_value(self, use: DateUse) -> exp.Expression:
        """The date a row is counted on, as a DATE or timestamp."""
        stored = self.stored_date(use)
        if use.mode in ("day_calendar", "date", "timestamp"):
            return stored
        return D.from_number(stored, "yyyymm" if use.mode in ("month_calendar", "yyyymm") else "yyyymmdd", self.d)

    def range_condition(self, use: DateUse, rng: Range) -> exp.Expr:
        """``use`` within ``[start, end)``; an open end stops at the placeholder bounds."""
        c = self.stored_date(use)
        lo, hi = rng.start or _LOW, rng.end or _HIGH
        if use.mode == "year_quarter":
            # Each quarter the period touches, in full (the resolver says so when the period cuts one): its first
            # day before the period's end, its last after the period's start.
            touched = add_units(lo, "month", -3) if lo != _LOW else lo
            return exp.and_(
                exp.GT(this=c.copy(), expression=_num(touched.year * 10000 + touched.month * 100 + touched.day)),
                exp.LT(this=c.copy(), expression=_num(hi.year * 10000 + hi.month * 100 + hi.day)),
                exp.Between(this=self.col(use.alias, use.quarter or ""), low=_num(1), high=_num(4)))
        if use.mode in ("month_calendar", "yyyymm"):
            last = hi - dt.timedelta(days=1)
            return exp.and_(
                exp.GTE(this=c.copy(), expression=_num(lo.year * 100 + lo.month)),
                exp.LTE(this=c.copy(), expression=_num(last.year * 100 + last.month)),
                exp.Between(this=D.mod(c.copy(), 100), low=_num(1), high=_num(12)))
        if use.mode == "yyyymmdd":
            return exp.and_(
                exp.GTE(this=c.copy(), expression=_num(lo.year * 10000 + lo.month * 100 + lo.day)),
                exp.LT(this=c.copy(), expression=_num(hi.year * 10000 + hi.month * 100 + hi.day)))
        return exp.and_(exp.GTE(this=c.copy(), expression=D.date_literal(lo, self.d)),
                        exp.LT(this=c.copy(), expression=D.date_literal(hi, self.d)))

    def days(self, start: DateUse, end: DateUse) -> exp.Expression:
        """Days from ``start`` to ``end`` on each row; NULL where either is missing or a placeholder."""
        for use in (start, end):
            if use.mode in ("month_calendar", "yyyymm", "year_quarter"):
                kept = "quarter" if use.mode == "year_quarter" else "month"
                raise CompileError(f"{use.role.name or 'this date'} is kept by {kept}: it has no days")
        real = exp.and_(self.range_condition(start, Range(None, None)), self.range_condition(end, Range(None, None)))
        return exp.Case(ifs=[exp.If(this=real, true=D.days_between(self.date_value(start), self.date_value(end),
                                                                      self.d))])

    def days_condition(self, p: DaysPred) -> exp.Expr:
        return self.predicate(self.days(p.start, p.end), p.op, list(p.values))

    def bucket(self, use: DateUse, grain: str) -> exp.Expression:
        monthly = use.mode in ("month_calendar", "yyyymm")
        if monthly and grain in ("day", "week"):
            raise CompileError(f"{use.role.name or 'this date'} is monthly: it has no {grain}s")
        if use.mode == "year_quarter" and grain in ("day", "week", "month", "fiscal_month"):
            raise CompileError(f"{use.role.name or 'this date'} is kept by quarter: it has no "
                               f"{grain.replace('fiscal_', '')}s")
        value = self.date_value(use)
        if grain in ("month", "fiscal_month"):
            return value if monthly else D.period_start(value, "month", self.d)
        if grain in ("fiscal_quarter", "fiscal_year"):
            start = self.q.fiscal_start or 1
            month_start = value if monthly else D.period_start(value, "month", self.d)
            month = D.date_part(value.copy(), "month", self.d)
            back = D.mod(D.add(D.sub(month, start), 12), 3 if grain == "fiscal_quarter" else 12)
            return D.add_months(month_start, exp.Neg(this=exp.Paren(this=back)), self.d)
        return D.period_start(value, grain, self.d)

    def time_attribute(self, use: DateUse, name: str) -> exp.Expression:
        """Computed from the date itself, so it means the same on every warehouse and calendar."""
        value = self.date_value(use)
        if use.mode == "year_quarter" and name not in ("quarter_of_year",):
            raise CompileError(f"{use.role.name or 'this date'} is kept by quarter: it has no "
                               f"{name.replace('_', ' ')}")
        if name == "day_of_week":
            if use.mode in ("month_calendar", "yyyymm"):
                raise CompileError(f"{use.role.name or 'this date'} is monthly: it has no weekdays")
            return D.date_part(value, "isodow", self.d)
        if name == "month_of_year":
            return D.date_part(value, "month", self.d)
        if name == "quarter_of_year":
            return D.date_part(value, "quarter", self.d)
        if name == "is_weekend":
            return exp.Case(ifs=[exp.If(this=exp.GTE(this=D.date_part(value, "isodow", self.d), expression=_num(6)),
                                        true=_num(1))], default=_num(0))
        if name == "hour_of_day":
            if use.mode != "timestamp":
                raise CompileError(f"{use.role.name or 'this date'} has no time of day")
            return D.date_part(value, "hour", self.d)
        # The calendar's own facts: what its row for the day says (a holiday, a business day, the fiscal week).
        column = use.calendar.attributes.get(name) if use.calendar is not None else None
        if column is None or name not in ("business_day", "holiday", "holiday_name", "fiscal_week"):
            raise CompileError(f"no time attribute {name} for {use.role.name or 'this date'}")
        held = self.col(use.alias, column)
        if name in ("business_day", "holiday"):
            yes = use.calendar.flags.get(name, 1) if use.calendar is not None else 1
            return exp.Case(ifs=[exp.If(this=exp.EQ(this=held, expression=self.literal(yes, column)), true=_num(1))],
                            default=_num(0))
        if name == "holiday_name":
            return exp.Nullif(this=exp.Trim(this=held), expression=exp.Literal.string(""))
        return held

    def band(self, value: exp.Expression, g: PartGroup) -> exp.Expression:
        """Each row's range: the first whose upper edge is above its value; no value, no range."""
        ifs = [exp.If(this=exp.LT(this=value.copy(), expression=_num(edge)), true=exp.Literal.string(label))
               for edge, label in zip(g.edges, g.labels)]
        last = exp.Literal.string(g.labels[-1]) if g.labels else exp.Null()
        if not ifs:
            return exp.Case(ifs=[exp.If(this=exp.not_(exp.Is(this=value, expression=exp.Null())), true=last)])
        return exp.Case(ifs=[exp.If(this=exp.Is(this=value.copy(), expression=exp.Null()), true=exp.Null()), *ifs],
                        default=last)

    def age(self, birth: exp.Expression, on: dt.date) -> exp.Expression:
        """Whole years from ``birth`` to ``on``: a year less before the birthday; none for a placeholder date
        (before 1901, or after ``on``). The same on every warehouse: years and month-days compared as numbers."""
        years = D.sub(_num(on.year), D.date_part(birth.copy(), "year", self.d))
        birthday = D.add(D.mul(D.date_part(birth.copy(), "month", self.d), 100), D.date_part(birth.copy(), "day", self.d))
        before = exp.Case(ifs=[exp.If(this=exp.GT(this=birthday, expression=_num(on.month * 100 + on.day)),
                                      true=_num(1))], default=_num(0))
        real = exp.and_(exp.GTE(this=birth.copy(), expression=D.date_literal(_LOW, self.d)),
                        exp.LTE(this=birth.copy(), expression=D.date_literal(on, self.d)))
        return exp.Case(ifs=[exp.If(this=real, true=D.sub(years, before))])

    def group_expr(self, g: PartGroup) -> exp.Expression:
        if g.kind == "band":
            if g.days is not None:
                return self.band(self.days(*g.days), g)
            if g.age_on is not None and g.column is not None:
                return self.band(self.age(self.col(g.alias, g.column), g.age_on), g)
            if g.column is None:
                raise CompileError(f"{g.name} has no value to put in ranges")
            return self.band(self.col(g.alias, g.column), g)
        if g.kind == "period":
            if g.date is None or g.grain is None:
                raise CompileError("a period needs a date and a grain")
            bucket = self.bucket(g.date, g.grain)
            months = self._compare_months()
            if months:
                # "2024 against 2023 by quarter": each row of the compared window is counted under the period it
                # is set against, a whole number of months on, so Q1 2023 stands beside Q1 2024.
                return exp.Case(ifs=[exp.If(this=self.range_condition(g.date, self.q.compare),
                                            true=D.add_months(bucket.copy(), months, self.d))], default=bucket)
            return bucket
        if g.kind == "time":
            if g.date is None or g.time_attr is None:
                raise CompileError("a time attribute needs a date")
            return self.time_attribute(g.date, g.time_attr)
        if g.kind == "segment":
            from core2.compile.shapes import segment_label

            if g.segment is None:
                raise CompileError(f"{g.name} has no segment")
            return segment_label(self, g.segment)
        if g.column is None:
            raise CompileError(f"{g.name} has no column")
        named = self.named_values(self.col(g.alias, g.column), g.column)
        if g.by_key is not None:
            # No row names this member: it is shown by the key its rows hold, "0123 (no name)".
            return exp.Coalesce(this=D.as_text(named, self.d), expressions=[
                exp.DPipe(this=D.as_text(self.col(*g.by_key), self.d), expression=exp.Literal.string(" (no name)"),
                          safe=True)])
        return named

    def _compare_months(self) -> int:
        """Whole months from the compared window to the question's, when both start on the same day of a month."""
        window, compare = self.q.window, self.q.compare
        if compare is None or window.start is None or compare.start is None or window.start.day != compare.start.day:
            return 0
        return (window.start.year - compare.start.year) * 12 + window.start.month - compare.start.month

    def _numbered(self, slug: str | None) -> bool:
        """Are the attribute ``slug``'s members numbers or codes (a fill number, an NPI), read with its name?"""
        attribute = self.model.attributes.get(slug) if slug else None
        return attribute is not None and attribute.kind in ("number", "identifier")

    def _ordinal(self, slug: str | None) -> bool:
        """Are the attribute ``slug``'s members amounts (3 days, fill number 2), not codes that only name a thing?"""
        attribute = self.model.attributes.get(slug) if slug else None
        return attribute is not None and attribute.kind == "number"

    def _missing(self, g: Group) -> str:
        """The name of the rows grouped under no member of ``g``'s attribute, when every one of them is a row
        whose link names none: an attribute of another table that is never empty there (each customer has a
        segment), so the sales with no segment are the sales with no customer -- "No customer recorded". Reached
        through a role, it is the role that names none: the sales by each rep's manager that have none are "No
        manager recorded", never "No sales rep recorded" (each has its rep). Empty when an empty value could be
        the member's own, or a row's number a member the other table does not hold ("Unknown")."""
        attribute = self.model.attributes.get(g.attribute) if g.attribute else None
        column = self.model.columns.get(attribute.column) if attribute is not None else None
        table = self.model.tables.get(column.table) if column is not None else None
        profile = column.profile if column is not None else None
        if table is None or not table.business_name or not self.q.parts \
                or any(p.table == column.table for p in self.q.parts) \
                or (column.nullable and (profile is None or not profile.rows or profile.non_null < profile.rows)):
            return ""
        steps = [self._reached_by(p, g.name) for p in self.q.parts]
        if not all(self._every_number_found(p, step) for p, step in zip(self.q.parts, steps)):
            return ""
        names = {step.what or table.business_name for step in steps if step is not None}
        if len(names) != 1:
            return ""
        name = names.pop()
        return f"No {name if name[:2].isupper() else name[:1].lower() + name[1:]} recorded"

    @staticmethod
    def _reached_by(part: Part, name: str) -> Joined | None:
        """The step by which ``part``'s rows reach the table of its grouping ``name`` (a sale's rep, or that rep's
        manager)."""
        group = next((pg for pg in part.groups if pg.name == name), None)
        return next((j for j in part.joins if group is not None and j.alias == group.alias), None)

    def _every_number_found(self, part: Part, step: Joined | None) -> bool:
        """Does every filled number on the way from ``part``'s rows to ``step`` name a row there? Only Learn's own
        measure of a link on all of its rows says so (core2/bootstrap/joins.py). A link measured on a sample of a
        large table, brought over from today's setup or added by an admin, one that keeps only some rows of its
        table (a condition), or whose numbers were not all found, may leave a row whose number names no member: its
        empty rows stay "Unknown"."""
        steps = {j.alias: j for j in part.joins}
        if step is None:
            return False
        while step is not None:
            left, right = {c for _, c, _ in step.on}, {c for _, _, c in step.on}
            link = next((j for j in self.model.joins.values() if j.trust != "rejected" and (
                (set(j.from_columns), set(j.to_columns)) in ((left, right), (right, left)))), None)
            if link is None or link.orphan_rows or link.match_rate < 1.0 or link.conditions or step.conds \
                    or not self._measured_on_every_row(link):
                return False
            step = steps.get(next((a for a, _, _ in step.on), ""))
        return True

    def _measured_on_every_row(self, link: Join) -> bool:
        """Did Learn test ``link`` on every row of its table, and find each filled number?"""
        table = self.model.tables.get(link.from_table)
        rows = table.row_count if table is not None else 0
        return bool(rows) and any(e.kind == "test_run" and not (e.data or {}).get("unmatched")
                                  and int((e.data or {}).get("rows") or 0) >= rows for e in link.evidence)

    def _flag(self, slug: str | None) -> bool:
        """Is the attribute ``slug`` a yes/no flag (its members a thing and its opposite)?"""
        attribute = self.model.attributes.get(slug) if slug else None
        return attribute is not None and attribute.column in self.model.columns and \
            self.model.columns[attribute.column].role == "flag"

    def named_values(self, value: exp.Expression, column_key: str) -> exp.Expression:
        """A text column's values as readers see them: the names an admin gave its codes ("C" -> "Cancelled"),
        any other value as stored."""
        column = self.model.columns[column_key]
        names = value_names(column)
        if not names:
            return value
        if column.data_type == "text":
            return exp.Case(ifs=[exp.If(this=exp.EQ(this=value.copy(), expression=exp.Literal.string(code)),
                                        true=exp.Literal.string(name))
                                 for code, name in sorted(names.items())], default=value.copy())
        if column.role != "flag":
            return value
        # A yes/no flag: its words, never 1 and 0 (no value stays no value).
        return exp.Case(ifs=[exp.If(this=exp.EQ(this=value.copy(), expression=(
            exp.Boolean(this=code == "1") if column.data_type == "boolean" else exp.Literal.number(code))),
            true=exp.Literal.string(name)) for code, name in sorted(names.items())])

    # ── measures ───────────────────────────────────────────────────────────
    def aggregate(self, expr: MeasureExpr | DaysBetween, part: Part,
                  condition: exp.Expr | None = None) -> exp.Expression:
        if isinstance(expr, DaysBetween):
            days = self.days(expr.start, expr.end)
            if condition is not None:
                days = exp.Case(ifs=[exp.If(this=condition.copy(), true=days)])
            if expr.agg == "avg":
                return exp.Avg(this=_double(days))      # whole days averaged are not truncated
            if expr.agg == "sum":
                return exp.Sum(this=days)
            return exp.Min(this=days) if expr.agg == "min" else exp.Max(this=days)
        if isinstance(expr, AggExpr):
            value: exp.Expression = self.at(part, expr.column) if expr.column else _num(1)
            conds = [self.predicate(self.at(part, f.column), f.op, list(f.values), f.column)
                     for f in expr.filters]
            if condition is not None:
                conds.append(condition.copy())
            if conds:
                value = exp.Case(ifs=[exp.If(this=exp.and_(*conds), true=value)])
            if expr.agg == "sum":
                return D.total(value, self.model.columns[expr.column].data_type if expr.column else "", self.d)
            if expr.agg == "count":
                return exp.Count(this=value)
            if expr.agg == "count_distinct":
                return exp.Count(this=exp.Distinct(expressions=[value]))
            if expr.agg == "avg":
                return exp.Avg(this=_double(value))      # an integer average is not truncated
            if expr.agg == "list":
                return D.listed(value, self.d)
            if expr.agg == "min":
                return exp.Min(this=value)
            return exp.Max(this=value)
        if isinstance(expr, OpExpr):
            if len(expr.args) != 2:
                raise CompileError(f"{expr.op} takes two measures")
            a, b = (self.aggregate(x, part, condition) for x in expr.args)
            return self.operate(expr, a, b)
        if isinstance(expr, SqlExpr):
            return self.formula(expr, part, condition)
        raise CompileError(f"cannot compile {type(expr).__name__}")

    def row_value(self, expr: MeasureExpr | DaysBetween, part: Part) -> exp.Expression:
        """A measure's value on each row, before it is added up: the days between its two dates, or its column
        (none where the measure's own conditions leave the row out)."""
        if isinstance(expr, DaysBetween):
            return self.days(expr.start, expr.end)
        if isinstance(expr, AggExpr) and expr.column and expr.agg in ("sum", "avg", "min", "max"):
            value: exp.Expression = self.at(part, expr.column)
            conds = [self.predicate(self.at(part, f.column), f.op, list(f.values), f.column) for f in expr.filters]
            return exp.Case(ifs=[exp.If(this=exp.and_(*conds), true=value)]) if conds else value
        raise CompileError("this measure is a total: it has no value on each row")

    def operate(self, expr: OpExpr, a: exp.Expression, b: exp.Expression) -> exp.Expression:
        if expr.op in ("add", "subtract"):
            # A sum over no rows is nought when added or taken away (a division with no revenue accounts still has
            # its expenses as a loss), never an empty figure.
            a, b = (exp.Coalesce(this=x, expressions=[_num(0)]) if isinstance(x, (exp.Sum, exp.Count)) else x
                    for x in (a, b))
        if expr.op == "ratio":
            result: exp.Expression = D.div(_double(a), exp.Nullif(this=b, expression=_num(0)))
        elif expr.op == "subtract":
            result = D.sub(a, b)
        elif expr.op == "add":
            result = D.add(a, b)
        else:
            result = D.mul(a, b)
        return D.mul(result, expr.scale) if expr.scale != 1 else result

    def combined(self, expr: MeasureExpr, alias: str, suffix: str) -> exp.Expression:
        """A measure over several tables, from their totals lined up under ``alias``.

        A table with no rows for a group adds nothing to it (a store with no returns
        has refunds of 0), so its total counts as 0 there; a ratio over a 0 is empty.
        A figure against its target is the exception: a period with no target set (or no
        figure yet) has no difference, never the whole figure (or the whole target) as one.
        """
        if isinstance(expr, RefExpr):
            if self.q.target is not None:
                return self.out(alias, f"{expr.measure}{suffix}")
            return exp.Coalesce(this=self.out(alias, f"{expr.measure}{suffix}"), expressions=[_num(0)])
        if isinstance(expr, OpExpr):
            if len(expr.args) != 2:
                raise CompileError(f"{expr.op} takes two measures")
            a, b = (self.combined(x, alias, suffix) for x in expr.args)
            return self.operate(expr, a, b)
        raise CompileError(f"cannot combine {type(expr).__name__}")

    def formula(self, expr: SqlExpr, part: Part, condition: exp.Expr | None) -> exp.Expression:
        """An imported formula (core2.model.formula), written from its checked form: every column
        is this part's, quoted as the warehouse spells it; for a comparison each aggregate counts
        only its period's rows; an average is never an integer division."""
        tree = formula.parse_stored(expr.sql)
        by_name = {self.model.columns[k].name.casefold(): k for k in expr.columns if k in self.model.columns}
        for column in list(tree.find_all(exp.Column)):
            # An admin's formula names each field by its key (fields of two tables can share a name);
            # an imported one by its own table's spelling.
            key = column.name if column.name in expr.columns else by_name.get(column.name.casefold())
            if key is None or key not in self.model.columns:
                raise CompileError(f"a formula reads {column.name}, which is not in the data any more")
            column.replace(self.at(part, key))
        kept = [self.predicate(self.at(part, f.column), f.op, list(f.values), f.column) for f in expr.filters]
        if kept:
            condition = exp.and_(*kept, *([condition] if condition is not None else []))
        for agg in list(tree.find_all(*formula.AGGREGATES)):
            arg = agg.this
            if condition is not None:
                if isinstance(arg, exp.Star):
                    arg = _num(1)
                if isinstance(arg, exp.Distinct):
                    arg.set("expressions", [exp.Case(ifs=[exp.If(this=condition.copy(), true=e)])
                                            for e in arg.expressions])
                else:
                    arg = exp.Case(ifs=[exp.If(this=condition.copy(), true=arg)])
            if isinstance(agg, exp.Avg):
                arg = _double(arg)
            if arg is not agg.this:
                agg.set("this", arg)
        if not isinstance(tree, exp.Expression):
            raise CompileError("an imported formula must be an expression")
        return tree

    # ── one measure table ──────────────────────────────────────────────────
    def part_select(self, part: Part) -> exp.Select:
        select = exp.select().from_(self.table(part.table, part.alias))
        for j in part.joins:
            select = select.join(self.table(j.table, j.alias), on=self.on(j),
                                 join_type="inner" if j.kind == "inner" else "left")
        for seg in part.segments:
            # Each member's activity in one row (their first day, their days in the period), joined by their key.
            from core2.compile.shapes import segment_summary

            select = select.join(segment_summary(self, seg), join_type="left", on=exp.EQ(
                this=self.col(seg.alias, seg.column), expression=self.out(seg.name, "k")))
        conds: list[exp.Expr] = [self.predicate(self.col(p.alias, p.column), p.op, p.values, p.column)
                                       for p in part.preds]
        if part.segments:
            from core2.compile.shapes import segment_test

            conds += [segment_test(self, seg, seg.status) for seg in part.segments if seg.split is None]
        conds += [self.exists(semi) for semi in part.semis]
        conds += [self.none_like(a, i) for i, a in enumerate(part.absent)]
        conds += [self.days_condition(p) for p in part.day_preds]
        conds += [self.predicate(self.time_attribute(t.date, t.name), t.op, list(t.values)) for t in part.time_preds]
        conds += [self.predicate(self.age(self.col(a.alias, a.column), a.on), a.op, list(a.values))
                  for a in part.age_preds]
        dated = False
        for use, rng in part.date_ranges:
            dated = dated or use is part.date
            conds.append(self.window_condition(use) if use is part.date else self.range_condition(use, rng))
        if part.date is not None and not dated:
            conds.append(self.range_condition(part.date, Range(None, None)))   # placeholders never count

        group_exprs: list[exp.Expression] = []
        for g in part.groups:
            e = self.group_expr(g)
            group_exprs.append(e)
            select = select.select(e.copy().as_(self.name(g.name)))

        if part.distinct_only:
            select = select.distinct()
            return select.where(exp.and_(*conds)) if conds else select
        if part.row_level:
            # Each row's own value (days from order to delivery of each fill), summarised a level up.
            for m in part.measures:
                select = select.select(self.row_value(m.expr, part).as_(self.name(m.name)))
            if part.snapshot and part.date is not None:
                conds.append(self.snapshot_days(part))
            return select.where(exp.and_(*conds)) if conds else select

        current = prior = None
        if self.q.compare is not None:
            if part.date is None:
                raise CompileError("a comparison needs a date")
            current = self.range_condition(part.date, self.q.window)
            prior = self.range_condition(part.date, self.q.compare)
        for m in part.measures:
            if isinstance(m.expr, Combined):
                raise CompileError(f"{m.label} is worked out over several tables, not in one")
            select = select.select(self.aggregate(m.expr, part, current).as_(self.name(m.name)))
            if prior is not None:
                select = select.select(self.aggregate(m.expr, part, prior).as_(self.name(f"{m.name}_prior")))
        if part.snapshot:
            if part.date is None:
                raise CompileError("a snapshot needs its date")
            conds.append(self.snapshot_days(part))
        if conds:
            select = select.where(exp.and_(*conds))
        if group_exprs:
            select = select.group_by(*[e.copy() for e in group_exprs])
        return select

    def none_like(self, a: MemberAbsent, n: int = 0) -> exp.Expr:
        """The member of this row had no row of activity like ``a.activity`` (NOT EXISTS, by the member's key)."""
        alias = f"had_none{n or ''}"
        rows = self.part_select(a.activity).subquery(self.name(alias))
        same = equal_keys(self.model, self.out(alias, a.activity_key), a.activity_column or a.column,
                          self.col(a.alias, a.column), a.column, self.d)
        return exp.not_(exp.Exists(this=exp.select(_num(1)).from_(rows).where(same)))

    def snapshot_days(self, part: Part) -> exp.Expr:
        """The stored date is the last snapshot day of its period (and of its compared window).

        The last day is the table's, not each member's: a member with no row on that
        day had nothing on hand, and an older row of it must not be carried forward.
        """
        assert part.date is not None
        use = part.date
        on_calendar = use.alias != part.alias
        inner_use = dataclasses.replace(use, alias="lsd" if on_calendar else "ls")
        inner = exp.select(exp.Max(this=self.stored_date(inner_use))).from_(self.table(part.table, "ls"))
        if on_calendar:
            j = next(j for j in part.joins if j.alias == use.alias)
            on = exp.and_(*[equal_keys(self.model, self.col("ls", lc), lc, self.col("lsd", rc), rc, self.d)
                            for _, lc, rc in j.on])
            inner = inner.join(self.table(j.table, "lsd"), on=on, join_type="inner")
        if any(u is use for u, _ in part.date_ranges):
            inner = inner.where(self.window_condition(inner_use))
        else:
            inner = inner.where(self.range_condition(inner_use, Range(None, None)))
        buckets: list[exp.Expression] = []
        period = next((g for g in part.groups if g.kind == "period"), None)
        if period is not None and period.grain:
            buckets.append(self.bucket(inner_use, period.grain))
        if self.q.compare is not None:
            # Each compared window has its own last day.
            buckets.append(exp.Case(ifs=[exp.If(this=self.range_condition(inner_use, self.q.window), true=_num(1))],
                                    default=_num(2)))
        if buckets:
            inner = inner.group_by(*buckets)
        return exp.In(this=self.stored_date(use), query=inner.subquery())

    def window_condition(self, use: DateUse) -> exp.Expr:
        """The question's window on its own date; with a comparison, either of the two windows."""
        if self.q.compare is not None:
            return exp.or_(self.range_condition(use, self.q.window), self.range_condition(use, self.q.compare))
        return self.range_condition(use, self.q.window)

    # ── the whole query ────────────────────────────────────────────────────
    def compile(self) -> Compiled:
        q = self.q
        if q.shape is not None:
            from core2.compile.shapes import shaped

            outer, columns, alias = shaped(self)
        else:
            outer, columns, alias = self.shaped_body()
        known = {c.name for c in columns if not c.computed}
        order: list[exp.Ordered] = []
        used: set[str] = set()
        for name, desc in q.sort:
            if name in known and name not in used:
                order.append(exp.Ordered(this=self.out(alias, name), desc=desc, nulls_first=False))
                used.add(name)
        for g in q.groups:      # ties broken by the groupings, always the same way (ranges in their own order)
            if g.name not in used:
                order.append(exp.Ordered(this=self.band_order(self.out(alias, g.name), g.bands) if g.bands
                                         else self.out(alias, g.name), desc=False, nulls_first=False))
                used.add(g.name)
        if order:
            outer = outer.order_by(*order)
        cap = min(q.limit, q.max_rows) if q.limit else q.max_rows + 1
        outer = outer.limit(cap)
        sql = D.render(outer, self.d)
        check_references(sql, self.d)
        return Compiled(sql=sql, dialect=self.d, columns=columns, row_cap=cap)

    def band_order(self, column: exp.Expression, labels: list[str]) -> exp.Expression:
        """A range's place among its ranges (to sort by): 1 for the first."""
        return exp.Case(ifs=[exp.If(this=exp.EQ(this=column.copy(), expression=exp.Literal.string(label)),
                                    true=_num(i + 1)) for i, label in enumerate(labels)], default=_num(len(labels) + 1))

    def shaped_body(self) -> tuple[exp.Select, list[OutColumn], str]:
        """The answer's rows before they are sorted and capped: the query, its columns, and the alias its columns
        are read under (q, or r when a level above worked out a change, a share or a filter on totals)."""
        q = self.q
        selects = [self.part_select(p) for p in q.parts]
        body = selects[0] if len(selects) == 1 else self._line_up(selects)
        if any(isinstance(m.expr, Combined) for m in q.measures):
            body = self._combine(body)

        columns: list[OutColumn] = [self.group_column(g) for g in q.groups]
        for m in q.measures:
            key = m.measure.key if m.measure else None
            columns.append(OutColumn(m.name, m.label, "list" if m.format == "list" else "measure", m.format, key))
            if q.compare is not None and m.format != "list":     # the products sold are no amount to compare
                columns.append(OutColumn(f"{m.name}_prior", m.label, "prior", m.format, key))

        # Output columns are always listed by name, never alias.*: QueryBot's production
        # query rules refuse a star projection, and a listed column is what was asked for. Each group's people,
        # when age is used, are read beside them and never shown (core2/resolve/age.py).
        kept = [q.people] if q.people else []
        outer = exp.select(*[self.out("q", c.name) for c in columns], *[self.out("q", n) for n in kept]) \
            .from_(body.subquery(self.name("q")))
        if q.compare is not None:
            for m in (x for x in q.measures if x.format != "list"):
                cur, pri = self.out("q", m.name), self.out("q", f"{m.name}_prior")
                # A member missing on one side changed by all of the other ("new this month"); missing on both,
                # nothing is known, and the change is no value, never 0.
                change = exp.Case(ifs=[exp.If(this=exp.and_(exp.Is(this=cur.copy(), expression=exp.Null()),
                                                            exp.Is(this=pri.copy(), expression=exp.Null())),
                                              true=exp.Null())],
                                  default=D.sub(exp.Coalesce(this=cur, expressions=[_num(0)]),
                                                exp.Coalesce(this=pri, expressions=[_num(0)])))
                outer = outer.select(change.as_(self.name(f"{m.name}_change")))
                outer = outer.select(D.div(_double(change.copy()), exp.Nullif(this=pri.copy(), expression=_num(0)))
                                     .as_(self.name(f"{m.name}_pct_change")))
                key = m.measure.key if m.measure else None
                columns.append(OutColumn(f"{m.name}_change", m.label, "change", m.format, key))
                columns.append(OutColumn(f"{m.name}_pct_change", m.label, "pct_change", "percent", key))
                if m.name in q.change_share:
                    # Each group's part of the whole change, over every group (before a filter on totals keeps
                    # some): a group that went the other way has a negative part, one that moved more than the
                    # whole did has more than all of it. The whole is each period's and each unit's own: a quarter's
                    # regions add up to that quarter's change, kilograms never to a change counted in boxes.
                    within = [self.out("q", g.name) for g in q.groups if g.kind == "period" or g.name == q.unit_group]
                    whole = exp.Window(this=exp.Sum(this=change.copy()), partition_by=within or None)
                    outer = outer.select(D.div(_double(change.copy()), exp.Nullif(this=whole, expression=_num(0)))
                                         .as_(self.name(f"{m.name}_share_of_change")))
                    columns.append(OutColumn(f"{m.name}_share_of_change", m.label, "share_of_change", "percent", key))
        if q.share and q.measures:
            m = next((x for x in q.measures if x.name == q.share_figure), q.measures[0])
            # Of everything shown, or of each member of a grouping ("% of each region's sales by category").
            total = exp.Window(this=exp.Sum(this=self.out("q", m.name)),
                               partition_by=[self.out("q", n) for n in q.share_within] or None)
            outer = outer.select(D.div(_double(self.out("q", m.name)), exp.Nullif(this=total, expression=_num(0)))
                                 .as_(self.name(f"{m.name}_share")))
            columns.append(OutColumn(f"{m.name}_share", m.label, "share", "percent", m.measure.key if m.measure else None))

        alias = "q"
        if q.compare is not None or q.share or q.having:
            # Sorted (and filtered on totals) one level up, where the change and the
            # share are real columns: Azure SQL cannot sort on an alias inside an
            # expression, and its nulls-last ordering is one. A share is of everything
            # shown before a filter on totals.
            alias = "r"
            outer = exp.select(*[self.out(alias, c.name) for c in columns], *[self.out(alias, n) for n in kept]) \
                .from_(outer.subquery(self.name(alias)))
            if q.having:
                outer = outer.where(exp.and_(*[self.predicate(self.out(alias, h.column), h.op, h.values)
                                               for h in q.having]))
        return outer, columns, alias

    def group_column(self, g: Group) -> OutColumn:
        return OutColumn(g.name, g.label, g.kind, "date" if g.kind == "period" else "",
                         grain=g.grain if g.kind == "period" else None, flag=self._flag(g.attribute),
                         numbered=self._numbered(g.attribute), ordinal=self._ordinal(g.attribute), bands=list(g.bands),
                         missing=self._missing(g) if g.kind == "attribute" else "", dated=g.dated)

    def _combine(self, body: exp.Select) -> exp.Select:
        """The groupings and every shown measure, those over several tables worked out from their parts."""
        suffixes = ["", "_prior"] if self.q.compare is not None else [""]
        level = exp.select(*[self.out("b", g.name) for g in self.q.groups])
        for m in self.q.measures:
            for suffix in suffixes:
                value = self.combined(m.expr.expr, "b", suffix) if isinstance(m.expr, Combined) \
                    else self.out("b", f"{m.name}{suffix}")
                level = level.select(value.as_(self.name(f"{m.name}{suffix}")))
        return level.from_(body.subquery(self.name("b")))

    def _line_up(self, selects: list[exp.Select]) -> exp.Select:
        """Each measure table aggregated on its own, then joined on the groupings (nulls matching nulls)."""
        names = [g.name for g in self.q.groups]
        aliases = [f"p{i + 1}" for i in range(len(selects))]
        if self.d == "postgres" and names:
            return self._line_up_on_keys(selects, names, aliases)

        def merged(name: str, upto: int) -> exp.Expression:
            """The grouping value from the first ``upto`` parts that have the row."""
            if upto == 1:
                return self.out(aliases[0], name)
            return exp.Coalesce(this=self.out(aliases[0], name),
                                expressions=[self.out(a, name) for a in aliases[1:upto]])

        line = exp.select(*[merged(n, len(selects)).as_(self.name(n)) for n in names])
        for part, alias in zip(self.q.parts, aliases):
            for m in part.measures:
                line = line.select(self.out(alias, m.name).as_(self.name(m.name)))
                if self.q.compare is not None:
                    line = line.select(self.out(alias, f"{m.name}_prior").as_(self.name(f"{m.name}_prior")))
        line = line.from_(selects[0].subquery(self.name(aliases[0])))
        for i, select in enumerate(selects[1:], start=1):
            if not names:
                line = line.join(select.subquery(self.name(aliases[i])), join_type="cross")
                continue
            conds = []
            for n in names:
                left, right = merged(n, i), self.out(aliases[i], n)
                conds.append(exp.or_(exp.EQ(this=left, expression=right),
                                     exp.and_(exp.Is(this=left.copy(), expression=exp.Null()),
                                              exp.Is(this=right.copy(), expression=exp.Null()))))
            line = line.join(select.subquery(self.name(aliases[i])), on=exp.and_(*conds), join_type="full")
        return line

    def _line_up_on_keys(self, selects: list[exp.Select], names: list[str], aliases: list[str]) -> exp.Select:
        """The same rows for PostgreSQL, which runs a FULL JOIN only on plain equality ("FULL JOIN is only
        supported with merge-joinable or hash-joinable join conditions"), so nulls cannot match there.

        Each part is named once (WITH); the groupings any part has, each once (UNION counts nulls as
        equal), are the rows; each part is joined to them, nulls matching nulls, which a LEFT JOIN may do.
        """
        def same(a: exp.Expression, b: exp.Expression) -> exp.Expression:
            return exp.or_(exp.EQ(this=a, expression=b),
                           exp.and_(exp.Is(this=a.copy(), expression=exp.Null()),
                                    exp.Is(this=b.copy(), expression=exp.Null())))

        keys: exp.Query | None = None
        for alias in aliases:
            part = exp.select(*[self.out(alias, n) for n in names]).from_(exp.Table(this=self.name(alias)))
            keys = part if keys is None else exp.union(keys, part, distinct=True)
        line = exp.select(*[self.out("k", n).as_(self.name(n)) for n in names])
        for part, alias in zip(self.q.parts, aliases):
            for m in part.measures:
                line = line.select(self.out(alias, m.name).as_(self.name(m.name)))
                if self.q.compare is not None:
                    line = line.select(self.out(alias, f"{m.name}_prior").as_(self.name(f"{m.name}_prior")))
        line = line.from_(keys.subquery(self.name("k")))
        for alias in aliases:
            line = line.join(exp.Table(this=self.name(alias)),
                             on=exp.and_(*[same(self.out(alias, n), self.out("k", n)) for n in names]),
                             join_type="left")
        for select, alias in zip(selects, aliases):
            line = line.with_(self.name(alias), as_=select)
        return line


def check_references(sql: str, dialect: str) -> None:
    """Every qualified column names a table or subquery of its own scope, as ``dialect`` folds names.

    DuckDB ignores the case of names, so a query that runs there can still name
    ``q`` where Snowflake or Oracle defined ``Q``; this catches it without a warehouse.
    A correlated subquery (a filter across a bridge: EXISTS) may also name the tables of the scopes around it.
    """
    tree = normalize_identifiers(sqlglot.parse_one(sql, read=dialect), dialect=dialect)
    for scope in traverse_scope(tree):
        sources = set(scope.sources)
        outer = scope.parent if scope.is_subquery else None
        while outer is not None:
            sources |= set(outer.sources)
            outer = outer.parent
        stars = [c for c in scope.expression.find_all(exp.Column) if isinstance(c.this, exp.Star)
                 and c.find_ancestor(exp.Select) is scope.expression]
        for column in [*scope.columns, *stars]:
            if column.table and column.table not in sources:
                raise CompileError(f"{column.sql(dialect=dialect)} names no table here ({dialect})")


def compile_query(logical: Logical, model: SemanticModel, dialect: str) -> Compiled:
    """SQL for ``dialect`` (snowflake | tsql | oracle | postgres | duckdb), read back by the dialect before it is returned.

    The logical query each SQL was written from is remembered (the last few hundred), so a query sent to the
    warehouse can be answered from a copy kept on this server when the copy holds exactly its rows
    (core2/precompute.py): the same SQL is the same question, whichever answer asked it."""
    compiled = _Compiler(logical, model, dialect).compile()
    with _WRITTEN_LOCK:
        _WRITTEN[compiled.sql] = (logical, model, compiled)
        _WRITTEN.move_to_end(compiled.sql)
        while len(_WRITTEN) > _WRITTEN_KEPT:
            _WRITTEN.popitem(last=False)
    return compiled


_WRITTEN: OrderedDict[str, tuple[Logical, SemanticModel, Compiled]] = OrderedDict()
_WRITTEN_LOCK = threading.Lock()
_WRITTEN_KEPT = 256


def compiled_from(sql: str) -> tuple[Logical, SemanticModel, Compiled] | None:
    """The logical query (with its model) ``sql`` was written from by compile_query, while it is remembered."""
    with _WRITTEN_LOCK:
        return _WRITTEN.get(sql)


def repeats_sql(model: SemanticModel, join: Join, dialect: str) -> str:
    """A query returning a row when ``join``'s target holds more than one row for some key: a key value
    found twice among the rows its conditions keep. None when every key is its own row."""
    writer = _Compiler(None, model, dialect)  # type: ignore[arg-type] - a check needs no logical query
    keys = [writer.col("t", c) for c in join.to_columns]
    kept = [exp.not_(exp.Is(this=k.copy(), expression=exp.Null())) for k in keys]
    kept += [writer.predicate(writer.col("t", f.column), f.op, list(f.values), f.column) for f in join.conditions]
    query = (exp.select(*[k.copy() for k in keys]).from_(writer.table(join.to_table, "t")).where(exp.and_(*kept))
             .group_by(*[k.copy() for k in keys])
             .having(exp.GT(this=exp.Count(this=exp.Star()), expression=exp.Literal.number(1))))
    return query.sql(dialect=dialect)


def latest_days_sql(model: SemanticModel, roles: list, dialect: str, *,
                    before: dt.date | None = None) -> list[tuple[str, list[str]]]:
    """Queries for the last real day each date role holds now, before ``before``: [(sql, role keys)], one query
    per table (each role's latest date in one pass over it) and one per date kept through a calendar. Each date is
    read as an answer reads it (through its calendar, a text or a number date as the date it is), placeholder days
    and numbers that are no date (a 13th month) left out. A year and a quarter kept as two numbers (a target's)
    hold no day of their own and are not read."""
    writer = _Compiler(None, model, dialect)  # type: ignore[arg-type] - a date's reach needs no logical query
    window = Range(None, before)
    plain: dict[str, list] = {}
    out: list[tuple[str, list[str]]] = []
    for role in roles:
        if role.quarter_column or role.table not in model.tables or role.column not in model.columns:
            continue
        if not role.calendar:
            # A date kept as text is read through a pattern that a value shaped like a date but none ('0000-00-00')
            # can still fail on some databases: on its own, so it never keeps the table's other dates from being read.
            text = text_date(model.columns[role.column])
            plain.setdefault(f"{role.table}\x00{role.key}" if text else role.table, []).append(role)
            continue
        calendar, join = model.calendars.get(role.calendar), model.joins.get(role.calendar_join or "")
        if calendar is None or join is None:
            continue
        use = DateUse(role, "c", "month_calendar" if calendar.grain == "month" else "day_calendar",
                      (calendar.key_column if calendar.grain == "month" else calendar.date_column) or "", calendar)
        query = exp.select(exp.Max(this=writer.stored_date(use))).from_(writer.table(role.table, "t")).join(
            writer.table(role.calendar, "c"), join_type="inner", on=writer.on(Joined(
                alias="c", table=role.calendar, on=[("t", join.from_columns[0], join.to_columns[0])])))
        out.append((query.where(writer.range_condition(use, window)).sql(dialect=dialect), [role.key]))
    for group, held in plain.items():
        table = group.split("\x00")[0]
        latest = []
        for role in held:
            column = model.columns[role.column]
            mode = ("date" if column.data_type == "date" else "timestamp" if column.data_type == "timestamp"
                    else "yyyymm" if role.granularity == "month" else "yyyymmdd")
            use = DateUse(role, "t", mode, column.key)
            real = writer.range_condition(use, window)
            if mode == "yyyymmdd":
                # A number shaped as a date only: the 13th month or the 32nd day is no date.
                real = exp.and_(real, exp.Between(this=D.mod(writer.stored_date(use), 10000), low=_num(101),
                                                  high=_num(1231)),
                                exp.Between(this=D.mod(writer.stored_date(use), 100), low=_num(1), high=_num(31)))
            latest.append(exp.Max(this=exp.Case(ifs=[exp.If(this=real, true=writer.stored_date(use))])))
        out.append((exp.select(*latest).from_(writer.table(table, "t")).sql(dialect=dialect), [r.key for r in held]))
    return out


def row_conditions(model: SemanticModel, dialect: str, alias: str, filters: list) -> list[exp.Expr]:
    """Conditions on one table's rows (a link's, or a table's always-leave-out rules), written as
    every query writes them: "not X" keeps the rows with no value."""
    writer = _Compiler(None, model, dialect)  # type: ignore[arg-type] - conditions need no logical query
    return [writer.predicate(writer.col(alias, f.column), f.op, list(f.values), f.column) for f in filters]
