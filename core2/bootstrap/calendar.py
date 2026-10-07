"""Finding calendar tables, whatever they are called, and reading their columns.

A calendar is recognised from its data: a date column that is unique, never
empty and runs day by day without gaps (a handful of sentinel rows such as
1900-01-01 "Unknown" aside), usually with a key column that maps one to one onto
the date. Each other column is then tested against the date: is it the year, the
month number, the ISO day of week, the Monday the week starts on, the month's
name in whatever language, a fiscal year and from which month it runs? Only
columns that agree with the date on (almost) every row become attributes, so a
calendar's own period columns can be trusted to match the date.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from sqlglot import exp

from core2.bootstrap.inventory import Inventory, InvTable
from core2.bootstrap.keys import TableKeys
from core2.bootstrap.profiler import TableProfile
from core2.model.schema import Evidence
from core2.warehouse import dialect as D
from core2.warehouse.runner import Warehouse

MIN_DAYS = 300
AGREE = 0.995
_LOW = dt.date(1901, 1, 1)
_HIGH = dt.date(9000, 1, 1)


@dataclass
class CalendarFinding:
    table: str                      # table key
    date_column: str                # "" for a period table
    key_column: str | None
    smart_key: bool                 # the key is the date written as yyyymmdd
    first: dt.date
    last: dt.date
    contiguous: bool
    real_rows: int
    placeholders: list = field(default_factory=list)          # key (or date) values meaning "no date"
    attributes: dict[str, str] = field(default_factory=dict)  # canonical attribute -> column name
    fiscal_year_start_month: int | None = None
    fiscal_year_named_by: str | None = None
    evidence: list[Evidence] = field(default_factory=list)
    grain: str = "day"
    year_rows: bool = False


def _as_date(value: object) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if value is None:
        return None
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _count_if(condition: exp.Expr) -> exp.Expression:
    return exp.Sum(this=exp.Case(ifs=[exp.If(this=condition, true=exp.Literal.number(1))],
                                 default=exp.Literal.number(0)))


class _Probe:
    def __init__(self, warehouse: Warehouse, table: InvTable, date_column: str):
        self.w = warehouse
        self.d = warehouse.dialect
        self.source = D.table_sql(table.database, table.schema, table.name, self.d)
        self.date = exp.column(D.ident(date_column, self.d))
        self.real = exp.and_(exp.GT(this=self.date.copy(), expression=D.date_literal(_LOW, self.d)),
                             exp.LT(this=self.date.copy(), expression=D.date_literal(_HIGH, self.d)))

    def col(self, name: str) -> exp.Column:
        return exp.column(D.ident(name, self.d))

    def part(self, name: str) -> exp.Expression:
        return D.date_part(self.date.copy(), name, self.d)

    def run(self, selects: list[exp.Expression], *, where: exp.Expr | None = None,
            group: exp.Expression | None = None) -> list[tuple]:
        query = exp.select(*[e.as_(f"s{i}") for i, e in enumerate(selects)]).from_(exp.to_table("__SRC__"))
        query = query.where((where or self.real).copy())
        if group is not None:
            query = query.group_by(group)
        return self.w.query(query.sql(dialect=self.d).replace("__SRC__", self.source, 1)).rows


def _integer_hypotheses(p: _Probe) -> dict[str, exp.Expression]:
    isodow = p.part("isodow")
    num = exp.Literal.number
    return {
        "year": p.part("year"),
        "quarter": p.part("quarter"),
        "month": p.part("month"),
        "day_of_month": p.part("day"),
        "day_of_week": isodow,
        "day_of_week_sunday_first": D.add(D.mod(isodow.copy(), 7), 1),
        "iso_week": p.part("isoweek"),
        "is_weekend": exp.Case(ifs=[exp.If(this=exp.GTE(this=isodow.copy(), expression=num(6)), true=num(1))],
                               default=num(0)),
        "year_month": D.add(D.mul(p.part("year"), 100), p.part("month")),
    }


def _date_hypotheses(p: _Probe) -> dict[str, exp.Expression]:
    return {f"{grain}_start": D.period_start(p.date.copy(), grain, p.d) for grain in ("week", "month", "quarter", "year")}


def _check(p: _Probe, table: InvTable, candidates: dict[str, list[str]],
           hypotheses: dict[str, exp.Expression], real_rows: int) -> dict[str, str]:
    """Columns agreeing with a hypothesis on (almost) every real row -> {attribute: column}."""
    tests = [(name, column, exp.EQ(this=p.col(column), expression=expression.copy()))
             for name, expression in hypotheses.items() for column in candidates.get(name, [])]
    if not tests:
        return {}
    values = p.run([_count_if(t) for _, _, t in tests])[0]
    found: dict[str, str] = {}
    for (name, column, _), value in zip(tests, values):
        if real_rows and int(value or 0) >= AGREE * real_rows and name not in found and column not in found.values():
            found[name] = column
    return found


def _calendar_in(warehouse: Warehouse, table: InvTable, profile: TableProfile, keys: TableKeys,
                 date_column: str) -> CalendarFinding | None:
    p = _Probe(warehouse, table, date_column)
    rows, distinct, first, last = p.run([exp.Count(this=exp.Literal.number(1)),
                                         exp.Count(this=exp.Distinct(expressions=[p.date.copy()])),
                                         exp.Min(this=p.date.copy()), exp.Max(this=p.date.copy())])[0]
    first, last = _as_date(first), _as_date(last)
    if not rows or first is None or last is None or rows != distinct or distinct < MIN_DAYS:
        return None
    span = (last - first).days + 1
    if distinct < 0.98 * span:
        return None
    finding = CalendarFinding(table=table.key, date_column=date_column, key_column=None, smart_key=False,
                              first=first, last=last, contiguous=distinct == span, real_rows=int(rows))
    finding.evidence.append(Evidence(kind="calendar_days", weight=1.0, data={"days": int(rows)},
                                     detail=f"{date_column} holds one row per day from {first} to {last}"))

    # The key: another unique column, ideally the date written as yyyymmdd.
    others = [c for c in keys.unique_columns if c != date_column
              and table.type_of(c) in ("integer", "text", "decimal")]
    if others:
        smart = D.add(D.add(D.mul(p.part("year"), 10000), D.mul(p.part("month"), 100)), p.part("day"))
        # Only numbers can be the date written as yyyymmdd; comparing text with a
        # number fails outright on most warehouses ("Jan 01, 1950" is a label).
        numeric = [c for c in others if table.type_of(c) in ("integer", "decimal")]
        values = p.run([_count_if(exp.EQ(this=p.col(c), expression=smart.copy())) for c in numeric])[0] \
            if numeric else []
        smart_keys = [c for c, v in zip(numeric, values) if int(v or 0) >= AGREE * rows]
        ranked = smart_keys or sorted(others, key=lambda c: (table.type_of(c) != "integer",
                                                             [x.name for x in table.columns].index(c)))
        finding.key_column = ranked[0]
        finding.smart_key = bool(smart_keys)
        sentinel_where = exp.or_(exp.not_(p.real.copy()), exp.Is(this=p.date.copy(), expression=exp.Null()))
        placeholder_rows = p.run([p.col(finding.key_column)], where=sentinel_where)
        finding.placeholders = sorted({r[0] for r in placeholder_rows if r[0] is not None},
                                      key=lambda v: (str(type(v)), v))[:20]

    # Attributes.
    integers = [c.name for c in table.columns if c.name not in (date_column, finding.key_column)
                and table.type_of(c.name) in ("integer", "decimal")]
    dates = [c.name for c in table.columns if c.name != date_column and table.type_of(c.name) == "date"]
    texts = [c.name for c in table.columns if table.type_of(c.name) == "text"
             and c.name != finding.key_column]
    int_hyp = _integer_hypotheses(p)
    found = _check(p, table, {name: integers for name in int_hyp}, int_hyp, finding.real_rows)
    date_hyp = _date_hypotheses(p)
    found.update(_check(p, table, {name: dates for name in date_hyp}, date_hyp, finding.real_rows))

    # Names: one value per month (12 overall), per weekday (7), per quarter (4), in any language.
    for attribute, part, count in (("month_name", "month", 12), ("day_name", "isodow", 7), ("quarter_name", "quarter", 4)):
        named = [t for t in texts if t not in found.values()
                 and count <= profile.columns[t].distinct <= count + len(finding.placeholders) + 1]
        if not named:
            continue
        rows_per_group = p.run([exp.Count(this=exp.Distinct(expressions=[p.col(t)])) for t in named],
                               group=p.part(part))
        for i, t in enumerate(named):
            if len(rows_per_group) == count and all(int(r[i] or 0) == 1 for r in rows_per_group) \
                    and attribute not in found:
                found[attribute] = t

    # Fiscal year: a year-like column that differs from the calendar year by a
    # fixed offset that changes at one month boundary.
    year_like = [c for c in integers if c not in found.values()
                 and (profile.columns[c].min_num or 0) >= 1900 and (profile.columns[c].max_num or 0) <= 2200]
    for column in year_like:
        diff = D.sub(p.col(column), p.part("year"))
        by_month = p.run([p.part("month"), exp.Min(this=diff.copy()), exp.Max(this=diff.copy())], group=p.part("month"))
        offsets = {int(m): (int(lo), int(hi)) for m, lo, hi in by_month if lo is not None}
        if len(offsets) != 12 or any(lo != hi for lo, hi in offsets.values()):
            continue
        shift = [offsets[m][0] for m in range(1, 13)]
        if len(set(shift)) != 2:
            continue
        start = next(m for m in range(2, 13) if shift[m - 1] != shift[m - 2])
        if set(shift) == {0, 1}:
            named_by = "end"
        elif set(shift) == {-1, 0}:
            named_by = "start"
        else:
            continue
        found["fiscal_year"] = column
        finding.fiscal_year_start_month, finding.fiscal_year_named_by = start, named_by
        fiscal_month = D.add(D.mod(D.add(D.sub(p.part("month"), start), 12), 12), 1)
        fiscal_quarter = D.add(exp.Floor(this=D.div(D.sub(fiscal_month.copy(), 1), 3)), 1)
        rest = [c for c in integers if c not in found.values()]
        found.update(_check(p, table, {"fiscal_month": rest, "fiscal_quarter": rest},
                            {"fiscal_month": fiscal_month, "fiscal_quarter": fiscal_quarter}, finding.real_rows))
        break

    finding.attributes = found
    if found:
        finding.evidence.append(Evidence(kind="calendar_attributes", weight=0.5,
                                         detail="agree with the date on every day: " + ", ".join(
                                             f"{c} ({a.replace('_', ' ')})" for a, c in sorted(found.items()))))
    return finding


def find_calendars(warehouse: Warehouse, inventory: Inventory, profiles: dict[str, TableProfile],
                   keys: dict[str, TableKeys]) -> dict[str, CalendarFinding]:
    """Calendars by table key."""
    out: dict[str, CalendarFinding] = {}
    for key, table in inventory.tables.items():
        profile = profiles[key]
        for column in table.columns:
            cp = profile.columns[column.name]
            if column.data_type not in ("date", "timestamp") or (column.data_type == "timestamp" and cp.time_share):
                continue
            # Unique apart from a few sentinel rows, and at least most of a year of days.
            if cp.distinct < MIN_DAYS or cp.distinct < 0.95 * cp.non_null:
                continue
            finding = _calendar_in(warehouse, table, profile, keys[key], column.name)
            if finding:
                out[key] = finding
                break
        if key not in out:
            period = _period_table(warehouse, table, profile, keys[key])
            if period:
                out[key] = period
    return out


def _period_table(warehouse: Warehouse, table: InvTable, profile: TableProfile,
                  keys: TableKeys) -> CalendarFinding | None:
    """A table of months keyed yyyymm (with year rows as month 00), checked against its own columns."""
    if len(keys.primary_key) != 1 or profile.rows > 100_000:
        return None
    key = keys.primary_key[0]
    kp = profile.columns[key]
    if kp.pattern != "yyyymm" or kp.distinct < 12:
        return None
    d = warehouse.dialect
    k = exp.column(D.ident(key, d))
    real = exp.and_(exp.Between(this=k.copy(), low=exp.Literal.number(190000), high=exp.Literal.number(299912)),
                    exp.Between(this=D.mod(k.copy(), 100), low=exp.Literal.number(0), high=exp.Literal.number(12)))
    year = exp.Floor(this=D.div(k.copy(), 100))
    month = D.mod(k.copy(), 100)
    integers = [c.name for c in table.columns if c.name != key and table.type_of(c.name) in ("integer", "decimal")]
    if not integers:
        return None
    tests = [(name, column, exp.EQ(this=exp.column(D.ident(column, d)), expression=expr.copy()))
             for name, expr in (("year", year), ("month", month)) for column in integers]
    selects = [exp.Count(this=exp.Literal.number(1)), _count_if(exp.EQ(this=month.copy(), expression=exp.Literal.number(0))),
               exp.Min(this=exp.Case(ifs=[exp.If(this=exp.GT(this=month.copy(), expression=exp.Literal.number(0)),
                                                 true=k.copy())])),
               exp.Max(this=exp.Case(ifs=[exp.If(this=exp.GT(this=month.copy(), expression=exp.Literal.number(0)),
                                                 true=k.copy())]))] + [_count_if(t) for _, _, t in tests]
    query = exp.select(*[s.as_(f"s{i}") for i, s in enumerate(selects)]).from_(exp.to_table("__SRC__")).where(real)
    values = warehouse.query(query.sql(dialect=d).replace("__SRC__", D.table_sql(table.database, table.schema,
                                                                                 table.name, d), 1)).rows[0]
    real_rows, year_rows, first, last = int(values[0] or 0), int(values[1] or 0), values[2], values[3]
    if real_rows < 12 or first is None:
        return None
    found: dict[str, str] = {}
    for (name, column, _), value in zip(tests, values[4:]):
        if int(value or 0) >= AGREE * real_rows and name not in found and column not in found.values():
            found[name] = column
    if not found:
        return None
    sentinel = exp.not_(real.copy())
    rows = warehouse.query(exp.select(k.copy()).from_(exp.to_table("__SRC__")).where(sentinel).sql(dialect=d)
                           .replace("__SRC__", D.table_sql(table.database, table.schema, table.name, d), 1),
                           max_rows=20).rows
    first_day = dt.date(int(first) // 100, int(first) % 100, 1)
    last_day = dt.date(int(last) // 100, int(last) % 100, 1)
    finding = CalendarFinding(table=table.key, date_column="", key_column=key, smart_key=True, first=first_day,
                              last=last_day, contiguous=False, real_rows=real_rows,
                              placeholders=sorted({r[0] for r in rows if r[0] is not None})[:20],
                              attributes=found, grain="month", year_rows=year_rows > 0)
    finding.evidence.append(Evidence(kind="period_table", weight=1.0, data={"periods": real_rows},
                                     detail=f"{key} is a month written yyyymm on {real_rows:,} rows"
                                            + (f" ({year_rows:,} whole-year rows as month 00)" if year_rows else "")))
    finding.evidence.append(Evidence(kind="calendar_attributes", weight=0.5, detail="agree with the period on every row: "
                                     + ", ".join(f"{c} ({a})" for a, c in sorted(found.items()))))
    return finding
