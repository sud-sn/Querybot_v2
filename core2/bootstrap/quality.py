"""Data-quality notes: what an answer should mention about the data it used.

Each flag is computed from evidence the bootstrap already has (profiles, join
tests, date roles), except outlier periods, which take one monthly total per
main measure. A flag never changes an answer; it adds one plain sentence to
answers that touch the flagged object.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import datetime as dt
import statistics

from sqlglot import exp

from core2.bootstrap import names
from core2.bootstrap.calendar import CalendarFinding
from core2.bootstrap.dates import DateCandidate
from core2.bootstrap.inventory import Inventory
from core2.bootstrap.joins import JoinFinding
from core2.bootstrap.measures import MeasureFinding
from core2.bootstrap.profiler import TableProfile
from core2.model.schema import QualityFlag
from core2.bootstrap.journal import attempt
from core2.warehouse import dialect as D
from core2.warehouse.runner import Warehouse

_STATUS_WORDS = {"status", "sts", "state", "stat", "stage"}
_LOW = dt.date(1901, 1, 1)
_HIGH = dt.date(8999, 12, 31)
# Codes that usually mean a row should not count. Single letters other than C, X
# and V are left out: D is as often "delivered" as "deleted".
_CANCEL_VALUES = {"c", "x", "v", "cnl", "can", "cxl", "cancel", "cancelled", "canceled", "void", "voided", "rev",
                  "reversed", "credit", "returned", "rejected", "deleted"}


def _flag(object_key: str, kind: str, message: str, severity: str = "info", **data: object) -> QualityFlag:
    return QualityFlag.model_validate({"key": f"{kind}:{object_key}", "object": object_key, "kind": kind,
                                       "message": message, "severity": severity, "data": dict(data)})


def find_quality(warehouse: Warehouse, inventory: Inventory, profiles: dict[str, TableProfile],
                 calendars: dict[str, CalendarFinding], joins: list[JoinFinding],
                 dates: dict[str, list[DateCandidate]], measures: list[MeasureFinding],
                 kinds: dict[str, str], *, outliers: bool = True) -> list[QualityFlag]:
    flags: list[QualityFlag] = []
    col_key = {}
    for key, table in inventory.tables.items():
        for column in table.columns:
            col_key[(key, column.name)] = f"{key}.{column.name.casefold()}"

    for m in measures:
        if m.column is None:
            continue
        p = profiles[m.table].columns[m.column]
        label = names.readable(m.column)
        if p.distinct == 1 and p.non_null:
            flags.append(_flag(col_key[(m.table, m.column)], "constant",
                               f"{label} is the same ({p.min}) on every row", "warning", value=p.min))
        if (p.negatives or 0) > 0 and m.format in ("integer", "number") and m.additivity == "additive":
            flags.append(_flag(col_key[(m.table, m.column)], "negative_values",
                               f"{p.negatives:,} rows have a negative {label.lower()}", "warning",
                               rows=p.negatives))
        units = [u for u in m.unit_values if u.strip()]
        if m.unit_column and len(units) > 1:
            flags.append(_flag(col_key[(m.table, m.column)], "unit_mix",
                               f"{label} mixes units ({', '.join(units[:6])}{', ...' if len(units) > 6 else ''}): "
                               "totals add different units", "warning", units=units))

    for j in joins:
        if j.trust in ("proposed", "declared") and j.match_rate < 0.99 and j.non_null:
            share = 1 - j.match_rate
            flags.append(_flag(col_key[(j.from_table, j.from_column)], "low_match_rate",
                               f"{share:.1%} of {inventory.tables[j.from_table].name} rows have a "
                               f"{names.readable(j.from_column).lower()} found nowhere in "
                               f"{inventory.tables[j.to_table].name}", "warning",
                               unmatched_rows=j.unmatched, match_rate=j.match_rate))

    for key, roles in dates.items():
        for c in roles:
            if c.placeholder_share > 0.001:
                flags.append(_flag(col_key[(key, c.column)], "placeholder_dates",
                                   f"{c.placeholder_share:.0%} of rows have no {names.readable(c.column).lower()} yet",
                                   share=c.placeholder_share))
            if c.kind == "audit" and any(e.kind == "load_clustering" for e in c.evidence):
                flags.append(_flag(col_key[(key, c.column)], "load_timestamp",
                                   f"{names.readable(c.column)} records when rows were written, not a business date"))

    linked = {(j.from_table, c) for j in joins for c in j.from_columns}
    dated = {(k, c.column) for k, roles in dates.items() for c in roles}
    for key, table in inventory.tables.items():
        if kinds.get(key) not in ("fact", "snapshot"):
            continue
        for column in table.columns:
            p = profiles[key].columns[column.name]
            if column.data_type not in ("text", "integer") or not p.top or not 2 <= p.distinct <= 12:
                continue
            if (key, column.name) in linked or (key, column.name) in dated:
                continue   # a key into a status table is filtered by that table's labels, not by its numbers
            words = set(names.tokens(column.name))
            values = {(t.value or "").strip().lower() for t in p.top}
            # A value on most rows is the normal state (C for closed), never a cancellation.
            counted = sum(t.count for t in p.top) or 1
            cancels = sorted({(t.value or "").strip() for t in p.top
                              if (t.value or "").strip().lower() in _CANCEL_VALUES and t.count <= counted / 2})
            if (words & _STATUS_WORDS and not names.opaque(column.name)) or (cancels and len(values) <= 6):
                flags.append(_flag(col_key[(key, column.name)], "status_column",
                                   f"{names.readable(column.name)} has codes {', '.join(sorted(v.value or '' for v in p.top))}: "
                                   "should some rows (cancelled, void, reversed) be left out of totals?",
                                   values=[t.value for t in p.top], cancel_like=cancels))

    # Members listed in a dimension but never used by the facts that point at it.
    for j in joins:
        if j.to_calendar or kinds.get(j.to_table) != "dimension" or kinds.get(j.from_table) not in ("fact", "snapshot"):
            continue
        used = profiles[j.from_table].columns[j.from_column].distinct - j.unmatched_values
        listed = profiles[j.to_table].rows
        if listed >= 10 and used < 0.8 * listed:
            flags.append(_flag(j.to_table, "listed_vs_active",
                               f"{listed:,} {inventory.tables[j.to_table].name} are listed; {used:,} appear in "
                               f"{inventory.tables[j.from_table].name}", listed=listed, used=used,
                               via=col_key[(j.from_table, j.from_column)]))

    if outliers:
        flags += _outlier_periods(warehouse, inventory, calendars, dates, measures, joins)
    unique: dict[str, QualityFlag] = {}
    for f in flags:
        unique.setdefault(f.key, f)
    return list(unique.values())


def _month_rows(warehouse: Warehouse, sql: str) -> list[tuple[Any, ...]]:
    return warehouse.query(sql, max_rows=2000).rows


def _outlier_periods(warehouse: Warehouse, inventory: Inventory, calendars: dict[str, CalendarFinding],
                     dates: dict[str, list[DateCandidate]], measures: list[MeasureFinding],
                     joins: list[JoinFinding]) -> list[QualityFlag]:
    out: list[QualityFlag] = []
    d = warehouse.dialect
    for key, roles in dates.items():
        default = next((c for c in roles if c.is_default and c.granularity in ("day", "timestamp")), None)
        if default is None:
            continue
        table = inventory.tables[key]
        candidates = [m for m in measures if m.table == key and m.column and m.agg == "sum"][:4]
        if not candidates:
            continue
        f = exp.to_table("__SRC__").as_("f")
        if default.via_calendar:
            cal = calendars[default.via_calendar.to_table]
            ct = inventory.tables[cal.table]
            day: exp.Expression = exp.column(D.ident(cal.date_column, d), table="c")
            query = exp.select().from_(f).join(
                exp.to_table("__CAL__").as_("c"),
                on=exp.EQ(this=exp.column(D.ident(default.column, d), table="f"),
                          expression=exp.column(D.ident(default.via_calendar.to_column, d), table="c")))
            cal_sql = D.table_sql(ct.database, ct.schema, ct.name, d)
        else:
            day = exp.column(D.ident(default.column, d), table="f")
            if table.type_of(default.column) not in ("date", "timestamp"):
                # A yyyymmdd key with no calendar to join: the date it holds. Compared with
                # dates as a number it is refused (Azure SQL 206, "operand type clash").
                day = D.from_number(day, "yyyymmdd", d)
            query = exp.select().from_(f)
            cal_sql = ""
        month = D.period_start(day, "month", d)
        query = query.select(month.as_("m"), *[D.total(exp.column(D.ident(m.column or "", d), table="f"),
                                                       table.type_of(m.column or ""), d).as_(f"s{i}")
                                               for i, m in enumerate(candidates)])
        query = query.where(exp.Between(this=day.copy(), low=D.date_literal(_LOW, d), high=D.date_literal(_HIGH, d)))
        query = query.group_by(month.copy())
        sql = query.sql(dialect=d).replace("__SRC__", D.table_sql(table.database, table.schema, table.name, d), 1)
        if cal_sql:
            sql = sql.replace("__CAL__", cal_sql, 1)
        rows: list[tuple[Any, ...]] = attempt(warehouse, f"the unusual-month check on {table.name}",
                                              partial(_month_rows, warehouse, sql), [])
        if len(rows) < 6:
            continue
        for i, m in enumerate(candidates):
            series = [(r[0], float(r[i + 1] or 0)) for r in rows]
            values = [v for _, v in series]
            median = statistics.median(values)
            mad = statistics.median([abs(v - median) for v in values]) or 0.0
            if mad == 0:
                continue
            worst = max(series, key=lambda s: abs(s[1] - median))
            z = abs(worst[1] - median) / (1.4826 * mad)
            if z >= 8:
                month_text = str(worst[0])[:7]
                out.append(_flag(f"{key}.{(m.column or '').casefold()}", "outlier_period",
                                 f"{month_text} {names.readable(m.column or '').lower()} is far from every other "
                                 f"month ({worst[1]:,.0f} against a typical {median:,.0f})", "warning",
                                 period=month_text, value=worst[1], typical=median))
    return out

