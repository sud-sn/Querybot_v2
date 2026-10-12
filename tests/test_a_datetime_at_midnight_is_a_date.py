"""A datetime column at midnight on every row is read as a date.

Warehouses often keep a date in a datetime column (Azure SQL's datetime, a load turned into a timestamp): every
value at 00:00. Learn read such a column as a moment, so a daily stock table dated by it was a table of
events, not a snapshot, and its stock was added up across every day: a stock value hundreds of times too
large. A timestamp column with no time of day on any row is now a day, as a date column is: the table is a
snapshot, its levels are taken at the last day, and the date is still read and filtered as before. A
timestamp that does carry times stays a moment.

A small warehouse in DuckDB, invented data; figures are checked against hand-written SQL.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 6, 30)
DAYS = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(120)]


def _learned(timed: bool):
    con = duckdb.connect()
    con.execute("CREATE TABLE depots (depot_id INTEGER PRIMARY KEY, depot_name VARCHAR)")
    con.execute("CREATE TABLE stock_by_day (balance_date TIMESTAMP, depot_id INTEGER, part_id INTEGER, "
                "on_hand_qty INTEGER, on_hand_value DECIMAL(14, 2))")
    con.executemany("INSERT INTO depots VALUES (?, ?)", [(1, "Northgate"), (2, "Riverside"), (3, "Hillcrest")])
    rows = []
    for i, day in enumerate(DAYS):
        at = dt.datetime.combine(day, dt.time(14, 30) if timed else dt.time(0, 0))
        for depot in (1, 2, 3):
            for part in range(1, 21):
                qty = 100 + (part * 7 + depot * 3 + i) % 50
                rows.append((at, depot, part, qty, qty * (2.5 + part)))
    con.executemany("INSERT INTO stock_by_day VALUES (?, ?, ?, ?, ?)", rows)
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t-midnight", options=BuildOptions(workers=1))
    return con, model


@pytest.fixture(scope="module")
def midnight():
    return _learned(timed=False)


def _value(model):
    return next(m for m in model.measures.values() if getattr(m.expr, "column", "") and
                m.expr.column.endswith(".on_hand_value"))


def test_a_stock_table_dated_at_midnight_is_a_snapshot(midnight):
    _, model = midnight
    table = next(t for t in model.tables.values() if t.name == "stock_by_day")
    assert table.kind == "snapshot"
    value = _value(model)
    assert (value.additivity, value.time_aggregation) == ("semi_additive", "last"), value.evidence


def test_its_stock_value_is_the_last_days_not_every_day_added(midnight):
    con, model = midnight
    plan = {"kind": "query", "intent": "value", "measures": [_value(model).slug]}
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, t: json.dumps(plan),
                        index=MemberIndex(), today=TODAY)
    payload = answer_question("what is our stock worth", services, Session())
    data = payload.get("data") or {}
    headers = data.get("headers") or []
    assert headers, payload["answer"]["headline"]
    want = con.execute("SELECT SUM(on_hand_value) FROM stock_by_day "
                       "WHERE balance_date = (SELECT MAX(balance_date) FROM stock_by_day)").fetchone()[0]
    assert [float(r[headers[-1]]) for r in data["rows"]] == [pytest.approx(float(want))]


def test_the_midnight_date_is_read_by_the_day(midnight):
    _, model = midnight
    role = next(r for r in model.date_roles.values() if model.columns[r.column].name == "balance_date")
    assert (role.granularity, role.kind) == ("day", "snapshot")


def test_a_timestamp_with_times_is_still_a_moment():
    _, model = _learned(timed=True)
    role = next(r for r in model.date_roles.values() if model.columns[r.column].name == "balance_date")
    assert role.granularity == "timestamp"
