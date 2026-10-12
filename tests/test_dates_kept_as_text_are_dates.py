"""Dates a database keeps as text ("2024-03-01 15:40:46") are dates.

A database loaded from files or moved from another one often keeps its dates in text columns. Learn read
only date and timestamp types, so such a database had no date at all: "revenue by month" was refused with
"Order has no date", and so was every question about a period. A text column whose first and last values are
dates written year first, and nearly every value of which reads as one, is now a date column; every query reads
it as a date, a value that is not one as no date (never an error that stops the question).

An invented shop whose order dates are text: one with the time of day, one a plain date often left empty.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest
import sqlglot

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse import dialect as D
from core2.warehouse.runner import DuckDBWarehouse


@pytest.fixture(scope="module")
def shop():
    con = duckdb.connect()
    con.execute("CREATE TABLE orders (order_id INTEGER, order_date VARCHAR, shipped_date VARCHAR, "
                "batch_code VARCHAR, note_date VARCHAR, customer VARCHAR, freight DECIMAL(10,2))")
    con.execute("CREATE TABLE order_lines (order_id INTEGER, product VARCHAR, quantity INTEGER, "
                "unit_price DECIMAL(10,2))")
    for i in range(1, 601):
        made = dt.datetime(2024, 1, 1) + dt.timedelta(days=i, hours=(i * 7) % 24, minutes=i % 60)
        shipped = (made + dt.timedelta(days=3)).strftime("%Y-%m-%d") if i % 10 else None
        # A code that starts like a date, and a column with a date in only some rows: neither is a date column.
        note = made.strftime("%Y-%m-%d") if i % 4 else "see notes"
        con.execute("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (i, made.strftime("%Y-%m-%d %H:%M:%S"), shipped, f"2024-{i % 12 + 1:02d}-B{i % 7}", note,
                     f"C{i % 40}", 5 + i % 9))
        for j in range(1 + i % 3):
            con.execute("INSERT INTO order_lines VALUES (?, ?, ?, ?)", (i, f"P{(i + j) % 25}", 1 + (i * j) % 5, 10))
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t-text-dates",
                        options=BuildOptions(workers=1))
    return con, model


def _ask(shop, plan):
    con, model = shop
    services = Services(model=model, warehouse=DuckDBWarehouse(con),
                        complete=lambda s, t: json.dumps({"kind": "query", **plan}), index=MemberIndex(),
                        today=dt.date(2025, 9, 1))
    return answer_question("q", services, Session())


def _column(model, name):
    return next(c for c in model.columns.values() if c.name == name)


def test_text_holding_dates_is_learned_as_a_date(shop):
    _, model = shop
    assert _column(model, "order_date").data_type == "timestamp"
    assert _column(model, "shipped_date").data_type == "date"
    roles = {model.columns[r.column].name for r in model.date_roles.values()}
    assert {"order_date", "shipped_date"} <= roles


def test_text_that_only_starts_like_a_date_stays_text(shop):
    _, model = shop
    assert _column(model, "batch_code").data_type == "text"
    assert _column(model, "note_date").data_type == "text"     # a quarter of its rows are no date


def test_a_date_kept_as_text_is_no_personal_detail(shop):
    _, model = shop
    assert _column(model, "shipped_date").personal == "none"


def test_a_month_by_month_series_reads_the_text_as_dates(shop):
    con, _ = shop
    payload = _ask(shop, {"intent": "trend", "measures": ["number_of_orders"], "time": {
        "grain": "month", "window": {"kind": "between", "start": "2024-01-01", "end": "2024-06-30"}}})
    got = [(r["period"][:7], r["number_of_orders"]) for r in payload["data"]["rows"]]
    want = con.execute("SELECT strftime(CAST(order_date AS TIMESTAMP), '%Y-%m'), COUNT(*) FROM orders "
                       "WHERE CAST(order_date AS TIMESTAMP) < TIMESTAMP '2024-07-01' GROUP BY 1 ORDER BY 1").fetchall()
    assert got == want


def test_a_period_on_the_other_text_date(shop):
    con, _ = shop
    payload = _ask(shop, {"intent": "value", "measures": ["freight"], "time": {
        "date": "shipped_date", "window": {"kind": "between", "start": "2024-03-01", "end": "2024-03-31"}}})
    want = con.execute("SELECT SUM(freight) FROM orders WHERE shipped_date BETWEEN '2024-03-01' AND '2024-03-31'"
                       ).fetchone()[0]
    assert payload["data"]["rows"][0]["freight"] == float(want)


def test_the_hour_of_a_text_timestamp(shop):
    payload = _ask(shop, {"intent": "breakdown", "measures": ["number_of_orders"], "group_by": ["time:hour_of_day"],
                          "time": {"window": {"kind": "between", "start": "2024-01-01", "end": "2024-12-31"}}})
    assert len(payload["data"]["rows"]) == 24


def test_an_order_line_is_dated_by_its_orders_text_date(shop):
    con, _ = shop
    payload = _ask(shop, {"intent": "value", "measures": ["quantity"], "time": {
        "window": {"kind": "between", "start": "2024-02-01", "end": "2024-02-29"}}})
    want = con.execute("SELECT SUM(l.quantity) FROM order_lines l JOIN orders o USING (order_id) "
                       "WHERE o.order_date >= '2024-02-01' AND o.order_date < '2024-03-01'").fetchone()[0]
    assert payload["data"]["rows"][0]["quantity"] == want


@pytest.mark.parametrize("dialect, word", [("tsql", "TRY_CONVERT"), ("snowflake", "TRY_TO_TIMESTAMP"),
                                           ("postgres", "CASE WHEN"), ("oracle", "REGEXP_LIKE"),
                                           ("duckdb", "TRY_CAST")])
def test_every_database_reads_a_text_date_without_failing_on_one_that_is_not(dialect, word):
    read = D.text_date(sqlglot.exp.column("d"), "timestamp", dialect).sql(dialect=dialect)
    assert word in read
    sqlglot.parse_one(f"SELECT {read}", read=dialect)
