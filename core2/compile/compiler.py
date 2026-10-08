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
  with a null-safe full join;
* a comparison is two conditional aggregates over one scan, with the change and
  the change in percent computed beside them; a share is of the shown total;
* one outer query sorts (nulls last everywhere, ties broken by the groupings) and
  caps the rows.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.scope import traverse_scope

from core2.model import formula
from core2.model.schema import AggExpr, MeasureExpr, OpExpr, SemanticModel, SqlExpr
from core2.resolve.resolver import DateUse, DaysBetween, DaysPred, Joined, Logical, Part, PartGroup
from core2.resolve.time import Range
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

    def col(self, alias: str, column_key: str) -> exp.Column:
        return exp.column(D.ident(self.model.columns[column_key].name, self.d), table=self.name(alias))

    def out(self, alias: str, name: str) -> exp.Column:
        return exp.column(self.name(name), table=self.name(alias))

    def table(self, table_key: str, alias: str) -> exp.Table:
        t = self.model.tables[table_key]
        table = D.table_expr(t.database, t.schema_name, t.name, self.d)
        table.set("alias", exp.TableAlias(this=self.name(alias)))
        return table

    def on(self, j: Joined) -> exp.Expr:
        return exp.and_(*[exp.EQ(this=self.col(left, lc), expression=self.col(j.alias, rc)) for left, lc, rc in j.on])

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
    def stored_date(self, use: DateUse) -> exp.Column:
        """The stored date (a DATE, a timestamp, or an order-preserving date number)."""
        return self.col(use.alias, use.column)

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
            if use.mode in ("month_calendar", "yyyymm"):
                raise CompileError(f"{use.role.name or 'this date'} is kept by month: it has no days")
        real = exp.and_(self.range_condition(start, Range(None, None)), self.range_condition(end, Range(None, None)))
        return exp.Case(ifs=[exp.If(this=real, true=D.days_between(self.date_value(start), self.date_value(end),
                                                                      self.d))])

    def days_condition(self, p: DaysPred) -> exp.Expr:
        return self.predicate(self.days(p.start, p.end), p.op, list(p.values))

    def bucket(self, use: DateUse, grain: str) -> exp.Expression:
        monthly = use.mode in ("month_calendar", "yyyymm")
        if monthly and grain in ("day", "week"):
            raise CompileError(f"{use.role.name or 'this date'} is monthly: it has no {grain}s")
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
        raise CompileError(f"no time attribute {name}")

    def group_expr(self, g: PartGroup) -> exp.Expression:
        if g.kind == "period":
            if g.date is None or g.grain is None:
                raise CompileError("a period needs a date and a grain")
            return self.bucket(g.date, g.grain)
        if g.kind == "time":
            if g.date is None or g.time_attr is None:
                raise CompileError("a time attribute needs a date")
            return self.time_attribute(g.date, g.time_attr)
        if g.column is None:
            raise CompileError(f"{g.name} has no column")
        return self.col(g.alias, g.column)

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
            value: exp.Expression = self.col(part.alias, expr.column) if expr.column else _num(1)
            conds = [self.predicate(self.col(part.alias, f.column), f.op, list(f.values), f.column)
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
            if expr.agg == "min":
                return exp.Min(this=value)
            return exp.Max(this=value)
        if isinstance(expr, OpExpr):
            if len(expr.args) != 2:
                raise CompileError(f"{expr.op} takes two measures")
            a, b = (self.aggregate(x, part, condition) for x in expr.args)
            if expr.op == "ratio":
                result: exp.Expression = D.div(_double(a), exp.Nullif(this=b, expression=_num(0)))
            elif expr.op == "subtract":
                result = D.sub(a, b)
            elif expr.op == "add":
                result = D.add(a, b)
            else:
                result = D.mul(a, b)
            return D.mul(result, expr.scale) if expr.scale != 1 else result
        if isinstance(expr, SqlExpr):
            return self.formula(expr, part, condition)
        raise CompileError(f"cannot compile {type(expr).__name__}")

    def formula(self, expr: SqlExpr, part: Part, condition: exp.Expr | None) -> exp.Expression:
        """An imported formula (core2.model.formula), written from its checked form: every column
        is this part's, quoted as the warehouse spells it; for a comparison each aggregate counts
        only its period's rows; an average is never an integer division."""
        tree = formula.parse_stored(expr.sql)
        by_name = {self.model.columns[k].name.casefold(): k for k in expr.columns if k in self.model.columns}
        for column in list(tree.find_all(exp.Column)):
            key = by_name.get(column.name.casefold())
            if key is None:
                raise CompileError(f"an imported formula reads {column.name}, which is not in the data any more")
            column.replace(self.col(part.alias, key))
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
        conds: list[exp.Expr] = [self.predicate(self.col(p.alias, p.column), p.op, p.values, p.column)
                                       for p in part.preds]
        conds += [self.days_condition(p) for p in part.day_preds]
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

        current = prior = None
        if self.q.compare is not None:
            if part.date is None:
                raise CompileError("a comparison needs a date")
            current = self.range_condition(part.date, self.q.window)
            prior = self.range_condition(part.date, self.q.compare)
        for m in part.measures:
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
            on = exp.and_(*[exp.EQ(this=self.col("ls", lc), expression=self.col("lsd", rc)) for _, lc, rc in j.on])
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
        selects = [self.part_select(p) for p in q.parts]
        body = selects[0] if len(selects) == 1 else self._line_up(selects)

        columns: list[OutColumn] = []
        for g in q.groups:
            columns.append(OutColumn(g.name, g.label, g.kind, "date" if g.kind == "period" else "",
                                     grain=g.grain if g.kind == "period" else None))
        for m in q.measures:
            key = m.measure.key if m.measure else None
            columns.append(OutColumn(m.name, m.label, "measure", m.format, key))
            if q.compare is not None:
                columns.append(OutColumn(f"{m.name}_prior", m.label, "prior", m.format, key))

        # Output columns are always listed by name, never alias.*: QueryBot's production
        # query rules refuse a star projection, and a listed column is what was asked for.
        outer = exp.select(*[self.out("q", c.name) for c in columns]).from_(body.subquery(self.name("q")))
        if q.compare is not None:
            for m in q.measures:
                cur, pri = self.out("q", m.name), self.out("q", f"{m.name}_prior")
                change = D.sub(exp.Coalesce(this=cur, expressions=[_num(0)]),
                               exp.Coalesce(this=pri, expressions=[_num(0)]))
                outer = outer.select(change.as_(self.name(f"{m.name}_change")))
                outer = outer.select(D.div(_double(change.copy()), exp.Nullif(this=pri.copy(), expression=_num(0)))
                                     .as_(self.name(f"{m.name}_pct_change")))
                key = m.measure.key if m.measure else None
                columns.append(OutColumn(f"{m.name}_change", m.label, "change", m.format, key))
                columns.append(OutColumn(f"{m.name}_pct_change", m.label, "pct_change", "percent", key))
        if q.share and q.measures:
            m = q.measures[0]
            total = exp.Window(this=exp.Sum(this=self.out("q", m.name)))
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
            outer = exp.select(*[self.out(alias, c.name) for c in columns]).from_(outer.subquery(self.name(alias)))
            if q.having:
                outer = outer.where(exp.and_(*[self.predicate(self.out(alias, h.column), h.op, h.values)
                                               for h in q.having]))

        known = {c.name for c in columns}
        order: list[exp.Ordered] = []
        used: set[str] = set()
        for name, desc in q.sort:
            if name in known and name not in used:
                order.append(exp.Ordered(this=self.out(alias, name), desc=desc, nulls_first=False))
                used.add(name)
        for g in q.groups:      # ties broken by the groupings, always the same way
            if g.name not in used:
                order.append(exp.Ordered(this=self.out(alias, g.name), desc=False, nulls_first=False))
                used.add(g.name)
        if order:
            outer = outer.order_by(*order)
        cap = min(q.limit, q.max_rows) if q.limit else q.max_rows + 1
        outer = outer.limit(cap)
        sql = D.render(outer, self.d)
        check_references(sql, self.d)
        return Compiled(sql=sql, dialect=self.d, columns=columns, row_cap=cap)

    def _line_up(self, selects: list[exp.Select]) -> exp.Select:
        """Each measure table aggregated on its own, then joined on the groupings (nulls matching nulls)."""
        names = [g.name for g in self.q.groups]
        aliases = [f"p{i + 1}" for i in range(len(selects))]

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


def check_references(sql: str, dialect: str) -> None:
    """Every qualified column names a table or subquery of its own scope, as ``dialect`` folds names.

    DuckDB ignores the case of names, so a query that runs there can still name
    ``q`` where Snowflake or Oracle defined ``Q``; this catches it without a warehouse.
    """
    tree = normalize_identifiers(sqlglot.parse_one(sql, read=dialect), dialect=dialect)
    for scope in traverse_scope(tree):
        sources = set(scope.sources)
        stars = [c for c in scope.expression.find_all(exp.Column) if isinstance(c.this, exp.Star)
                 and c.find_ancestor(exp.Select) is scope.expression]
        for column in [*scope.columns, *stars]:
            if column.table and column.table not in sources:
                raise CompileError(f"{column.sql(dialect=dialect)} names no table here ({dialect})")


def compile_query(logical: Logical, model: SemanticModel, dialect: str) -> Compiled:
    """SQL for ``dialect`` (snowflake | tsql | oracle | duckdb), read back by the dialect before it is returned."""
    return _Compiler(logical, model, dialect).compile()
