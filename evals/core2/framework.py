"""Synthetic warehouses with known answers, for core2's evaluation.

A domain is written once, with descriptive names, as pandas frames plus the
ground truth a perfect learner would arrive at (table kinds, keys, joins and their
roles, the calendar, date roles and defaults, measures and how they add up,
labels, status codes, data-quality traps). :func:`materialize` renders it in one
naming style into DuckDB and maps the truth onto the physical names, so every
style is graded against the same facts.

Keys are declared the way warehouses declare them: ``descriptive`` and ``pascal``
declare primary and foreign keys, ``warehouse`` declares primary keys only, and
``generic`` declares nothing. Foreign keys are never created as DuckDB
constraints (DuckDB enforces them, and real warehouses hold orphans); they are
returned as declarations, the way discovery reports them.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import duckdb
import pandas as pd

from evals.core2 import naming


@dataclass
class TableDef:
    name: str                   # descriptive (logical) name
    kind: str                   # fact | snapshot | dimension | bridge | calendar
    data: pd.DataFrame          # columns in warehouse order, descriptive names
    types: dict[str, str]       # column -> DuckDB type
    primary_key: list[str]
    business_name: str = ""


@dataclass
class JoinTruth:
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    role: str | None = None     # business name of the role when the pair joins more than one way
    declared: bool = True       # declared as a foreign key where the style declares keys
    trust: str = "verified"     # what the data supports: verified (>= 99% match) or proposed
    also: tuple[tuple[str, str], ...] = ()   # the key's other column pairs: (rx_number, fill_number) -> fills
    cast: bool = False          # the two sides are stored as different types ('00042' against 42)

    def pairs(self) -> tuple[tuple[str, str], ...]:
        """Every (from column, to column) of the key, the first one first."""
        return ((self.from_column, self.to_column), *self.also)


@dataclass
class DateTruth:
    table: str
    column: str
    name: str                   # "Order date"
    kind: str                   # event | snapshot | planned | due | validity | audit
    via_calendar: bool
    default: bool = False


@dataclass
class MeasureTruth:
    table: str
    column: str | None          # None: a count of rows
    agg: str                    # sum | count | count_distinct | avg | min | max
    additivity: str             # additive | semi_additive | non_additive
    format: str                 # currency | percent | count | number | integer
    name: str
    unit_column: str | None = None          # "table.column"
    time_aggregation: str | None = None     # last | average, for semi-additive


@dataclass
class Truth:
    kinds: dict[str, str]
    primary_keys: dict[str, list[str]]
    joins: list[JoinTruth]
    calendar: dict[str, Any] | None
    dates: list[DateTruth]
    measures: list[MeasureTruth]
    labels: dict[str, str]                  # entity table -> label column
    codes: dict[str, str] = field(default_factory=dict)
    statuses: dict[str, dict[str, Any]] = field(default_factory=dict)   # "table.column" -> facts
    quality: list[dict[str, Any]] = field(default_factory=list)          # expected flags
    not_measures: list[str] = field(default_factory=list)               # "table.column" numbers that are not measures
    sensitive: dict[str, str] = field(default_factory=dict)             # "table.column" -> pii | confidential


@dataclass
class Domain:
    name: str
    tables: list[TableDef]
    truth: Truth
    today: dt.date
    description: str = ""
    abbreviations: dict[str, str] = field(default_factory=dict)   # extra warehouse-style abbreviations

    def table(self, name: str) -> TableDef:
        return next(t for t in self.tables if t.name == name)


@dataclass
class Built:
    """One domain in one naming style, loaded into DuckDB."""

    domain: Domain
    style: str
    con: duckdb.DuckDBPyConnection
    tables: dict[str, str]                  # logical -> physical table
    columns: dict[str, dict[str, str]]      # logical table -> {logical column: physical}
    declared_fks: list[dict[str, Any]]
    declared_pks: dict[str, list[str]]      # physical table -> physical columns

    def t(self, logical: str) -> str:
        return self.tables[logical]

    def c(self, logical: str) -> tuple[str, str]:
        """``"table.column"`` (logical) -> (physical table, physical column)."""
        table, column = logical.split(".", 1)
        return self.tables[table], self.columns[table][column]

    def logical_of(self) -> dict[tuple[str, str], str]:
        """(physical table, physical column) -> ``"table.column"`` (logical), case-folded keys."""
        out: dict[tuple[str, str], str] = {}
        for table, physical in self.tables.items():
            for column, phys_col in self.columns[table].items():
                out[(physical.casefold(), phys_col.casefold())] = f"{table}.{column}"
        return out


def materialize(domain: Domain, style: str, con: duckdb.DuckDBPyConnection | None = None) -> Built:
    con = con or duckdb.connect()
    spec = {t.name: (t.kind, list(t.data.columns)) for t in domain.tables}
    names = naming.rename_map(spec, style, salt=domain.name, abbreviations=domain.abbreviations)
    declare_pk = style in ("descriptive", "pascal", "warehouse")
    declare_fk = style in ("descriptive", "pascal")

    declared_pks: dict[str, list[str]] = {}
    for table in domain.tables:
        physical, cols = names[table.name]
        col_defs = [f'"{cols[c]}" {table.types[c]}' for c in table.data.columns]
        if declare_pk and table.primary_key:
            pk = [cols[c] for c in table.primary_key]
            col_defs.append("PRIMARY KEY (" + ", ".join(f'"{c}"' for c in pk) + ")")
            declared_pks[physical] = pk
        con.execute(f'CREATE TABLE "{physical}" ({", ".join(col_defs)})')
        frame = table.data.rename(columns=cols)
        con.register("_frame", frame)
        con.execute(f'INSERT INTO "{physical}" SELECT * FROM _frame')
        con.unregister("_frame")

    declared_fks: list[dict[str, Any]] = []
    if declare_fk:
        for j in domain.truth.joins:
            if not j.declared:
                continue
            parent, pcols = names[j.from_table]
            ref, rcols = names[j.to_table]
            for ordinal, (from_column, to_column) in enumerate(j.pairs(), start=1):
                declared_fks.append({
                    "constraint_name": f"fk_{parent}_{pcols[j.from_column]}".lower(),
                    "parent_schema": "main", "parent_table": parent, "parent_col": pcols[from_column],
                    "ref_schema": "main", "ref_table": ref, "ref_col": rcols[to_column],
                    "ordinal": ordinal, "enforced": False, "source": "synthetic",
                })
    return Built(domain=domain, style=style, con=con,
                 tables={t: names[t][0] for t in names},
                 columns={t: names[t][1] for t in names},
                 declared_fks=declared_fks, declared_pks=declared_pks)


# ── calendar helper shared by domains ────────────────────────────────────────

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


UNKNOWN_DAY = (-1, dt.date(1900, 1, 1), "Unknown")


def calendar_frame(start: dt.date, end: dt.date, *, fiscal_start_month: int = 1,
                   placeholders: list[tuple[int, dt.date, str]] | None = None) -> pd.DataFrame:
    """A conventional date dimension keyed yyyymmdd, with placeholder rows.

    Fiscal years start in ``fiscal_start_month`` and are named by the calendar
    year they end in (fiscal 2026 = April 2025 to March 2026 when it starts in April).
    ``placeholders`` are ``(key, sentinel date, label)`` rows meaning "no date";
    the default is one ``-1 / 1900-01-01 / Unknown`` row, ``[]`` adds none.
    """
    days = pd.date_range(start, end, freq="D")
    month = days.month
    fiscal_month = ((month - fiscal_start_month) % 12) + 1
    fiscal_year = days.year + (month >= fiscal_start_month).astype(int) if fiscal_start_month != 1 else days.year
    frame = pd.DataFrame({
        "date_key": (days.year * 10000 + month * 100 + days.day).astype("int64"),
        "full_date": days.date,
        "year": days.year.astype("int64"),
        "quarter": days.quarter.astype("int64"),
        "month_number": month.astype("int64"),
        "month_name": [MONTHS[m - 1] for m in month],
        "month_start": (days - pd.to_timedelta(days.day - 1, unit="D")).date,
        "week_start": (days - pd.to_timedelta(days.dayofweek, unit="D")).date,
        "day_of_week": (days.dayofweek + 1).astype("int64"),
        "day_name": [DAYS[d] for d in days.dayofweek],
        "fiscal_year": pd.Index(fiscal_year).astype("int64"),
        "fiscal_quarter": ((fiscal_month - 1) // 3 + 1).astype("int64"),
        "fiscal_month": pd.Index(fiscal_month).astype("int64"),
        "is_weekend": (days.dayofweek >= 5).astype("int64"),
    })
    rows = [UNKNOWN_DAY] if placeholders is None else placeholders
    if rows:
        extra = []
        for key, sentinel, label in rows:
            row = {c: None for c in frame.columns}
            row.update({"date_key": key, "full_date": sentinel, "month_name": label, "day_name": label})
            extra.append(row)
        sentinel_rows = [r for r in extra if r["date_key"] < int(frame["date_key"].iloc[0])]
        late_rows = [r for r in extra if r not in sentinel_rows]
        frame = pd.concat([pd.DataFrame(sentinel_rows), frame, pd.DataFrame(late_rows)], ignore_index=True)
        for column in ("year", "quarter", "month_number", "day_of_week", "fiscal_year", "fiscal_quarter",
                       "fiscal_month", "is_weekend"):
            frame[column] = frame[column].astype("Int64")
    return frame


CALENDAR_TYPES = {
    "date_key": "INTEGER", "full_date": "DATE", "year": "INTEGER", "quarter": "INTEGER", "month_number": "INTEGER",
    "month_name": "VARCHAR", "month_start": "DATE", "week_start": "DATE", "day_of_week": "INTEGER",
    "day_name": "VARCHAR", "fiscal_year": "INTEGER", "fiscal_quarter": "INTEGER", "fiscal_month": "INTEGER",
    "is_weekend": "INTEGER",
}

CALENDAR_ATTRIBUTES = {
    "year": "year", "quarter": "quarter", "month": "month_number", "month_name": "month_name",
    "month_start": "month_start", "week_start": "week_start", "day_of_week": "day_of_week", "day_name": "day_name",
    "fiscal_year": "fiscal_year", "fiscal_quarter": "fiscal_quarter", "fiscal_month": "fiscal_month",
    "is_weekend": "is_weekend",
}


def date_key(day: dt.date) -> int:
    return day.year * 10000 + day.month * 100 + day.day
