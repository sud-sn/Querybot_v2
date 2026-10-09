"""A snapshot still loading is not read as the latest balance.

A balance is read on its snapshot's last day. When that day holds a fraction of the rows the days
before it held (two warehouses of five hundred, loaded so far), it is a load in progress, not the
state of the business: the answer is read on the last complete day instead, and says so. Only the
question's own window is moved; a comparison window keeps its days.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import statistics

from sqlglot import exp

from core2.compile.compiler import _Compiler, _num
from core2.model.schema import SemanticModel
from core2.resolve.resolver import Logical
from core2.resolve.time import Range
from core2.warehouse.runner import Warehouse

log = logging.getLogger("querybot.core2")

LOOKED_AT = 4          # the latest snapshot days compared
INCOMPLETE = 0.5       # under half the rows of the days before it: still loading


def _day(value: object, mode: str) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value).strip()
    try:
        if mode in ("month_calendar", "yyyymm") and len(text) == 6:
            return dt.date(int(text[:4]), int(text[4:]), 1)
        if len(text) == 8 and text.isdigit():
            return dt.date(int(text[:4]), int(text[4:6]), int(text[6:]))
        return dt.date.fromisoformat(text[:10])
    except ValueError:
        return None


def _said(day: dt.date) -> str:
    return f"{day.day} {day:%b %Y}"


def complete_snapshots(logical: Logical, model: SemanticModel, warehouse: Warehouse) -> Logical:
    """``logical`` with its window ending before a snapshot day that is still loading, and a note saying so."""
    compiler = _Compiler(logical, model, warehouse.dialect)
    for part in logical.parts:
        if not part.snapshot or part.date is None:
            continue
        use = part.date
        on_calendar = use.alias != part.alias
        inner_use = dataclasses.replace(use, alias="lsd" if on_calendar else "ls")
        stored = compiler.stored_date(inner_use)
        query = exp.select(stored.copy().as_("d"), exp.Count(this=_num(1)).as_("n")).from_(
            compiler.table(part.table, "ls"))
        if on_calendar:
            j = next(j for j in part.joins if j.alias == use.alias)
            on = exp.and_(*[exp.EQ(this=compiler.col("ls", lc), expression=compiler.col("lsd", rc)) for _, lc, rc in j.on])
            query = query.join(compiler.table(j.table, "lsd"), on=on, join_type="inner")
        query = query.where(compiler.range_condition(inner_use, logical.window)).group_by(stored.copy()) \
            .order_by(exp.Ordered(this=stored.copy(), desc=True)).limit(LOOKED_AT)
        try:
            rows = warehouse.query(query.sql(dialect=warehouse.dialect)).rows
        except Exception as exc:  # noqa: BLE001 - the answer is still given; the check is said to be skipped
            log.warning("core2 could not check the latest snapshot of %s: %s", model.tables[part.table].name, exc)
            continue
        if len(rows) < 3:
            continue
        latest_value, latest_n = rows[0][0], int(rows[0][1] or 0)
        before = [int(r[1] or 0) for r in rows[1:]]
        usual = statistics.median(before)
        latest, complete = _day(latest_value, use.mode), _day(rows[1][0], use.mode)
        if latest is None or complete is None or not usual or latest_n >= INCOMPLETE * usual:
            continue
        end = latest if logical.window.end is None else min(logical.window.end, latest)
        logical.window = Range(logical.window.start, end, list(logical.window.notes))
        for other in logical.parts:
            if other.snapshot and other.date is not None and not any(u is other.date for u, _ in other.date_ranges):
                other.date_ranges.append((other.date, logical.window))   # a question with no period is now bounded
        table = model.tables[part.table].business_name.lower() or model.tables[part.table].name
        logical.notes.append(
            f"The {table} snapshot of {_said(latest)} holds {latest_n:,} rows against about {usual:,.0f} in the "
            f"snapshots before it: it looks incomplete, so this is as of {_said(complete)}, the last complete one.")
    return logical
