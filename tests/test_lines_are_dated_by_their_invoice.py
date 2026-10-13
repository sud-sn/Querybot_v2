"""Invoice lines are dated by their invoice, though the invoice holds no figure of its own.

Found by the sector stress test on a public online retailer's data (October 2026), kept as a retailer's database
keeps it: invoices (number, date, customer, country) and their lines (invoice number, line number, product,
quantity, unit price). The invoices hold no number to add up, so Learn took them for a list like the customers,
and the date of a list is never what an answer about other rows is counted by: every question about the
year's sales or units -- the money is all on the lines -- was asked back ("I could not work out how to answer
that as asked"). A table its lines point at by their own key (invoice and line number), and dated, is a
document: a fact, counted, whose date its lines are dated by.

Invented rows only.
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


@pytest.fixture(scope="module")
def shop():
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE customers AS SELECT (12000 + range)::VARCHAR AS customer_id,
               ['United Kingdom', 'France', 'Germany', 'Spain'][range % 4 + 1] AS country,
               DATE '2010-01-01' + ((range * 7) % 97)::INTEGER AS first_order_date FROM range(300);
        CREATE TABLE products AS SELECT 'P' || lpad(range::VARCHAR, 4, '0') AS stock_code,
               'Product number ' || range || repeat(' deluxe', range % 3) AS description FROM range(200);
        CREATE TABLE invoices AS SELECT (500000 + range)::VARCHAR AS invoice_no,
               TIMESTAMP '2010-01-04 09:00:00' + INTERVAL (range * 13) HOUR AS invoice_date,
               (12000 + (range * 7) % 300)::VARCHAR AS customer_id,
               ['United Kingdom', 'France', 'Germany', 'Spain'][range % 4 + 1] AS ship_country,
               range % 23 = 0 AS is_cancellation FROM range(1500);
        CREATE TABLE invoice_lines AS SELECT (500000 + range // 4)::VARCHAR AS invoice_no, range % 4 + 1 AS line_no,
               'P' || lpad(((range * 11) % 200)::VARCHAR, 4, '0') AS stock_code, range % 12 + 1 AS quantity,
               round(0.85 + (range % 40) * 0.25, 2) AS unit_price FROM range(6000);
    """)
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t-shop", options=BuildOptions(workers=1))
    return con, warehouse, model


def _table(model, name):
    return next(t for k, t in model.tables.items() if k.endswith("." + name))


def test_the_invoices_are_documents_its_lines_are_dated_by(shop):
    _, _, model = shop
    assert _table(model, "invoices").kind == "fact"
    assert _table(model, "customers").kind == "dimension"


def test_units_sold_by_month_are_counted_by_the_invoice_date(shop):
    con, warehouse, model = shop
    quantity = next(m for m in model.measures.values() if getattr(m.expr, "column", "") and
                    m.expr.column.endswith("invoice_lines.quantity"))
    plan = {"kind": "query", "measures": [quantity.slug], "time": {"grain": "month", "window": {
        "kind": "between", "start": "2010-01-01", "end": "2010-03-31"}}}
    services = Services(model=model, warehouse=warehouse, complete=lambda system, user: json.dumps(plan),
                        index=MemberIndex(), today=dt.date(2010, 12, 31))
    payload = answer_question("Units sold by month in Q1 2010", services, Session())
    rows = (payload.get("data") or {}).get("rows")
    assert rows, payload.get("answer") or payload
    want = con.execute("""SELECT date_trunc('month', i.invoice_date)::DATE, SUM(l.quantity)
                          FROM invoice_lines l JOIN invoices i USING (invoice_no)
                          WHERE i.invoice_date >= DATE '2010-01-01' AND i.invoice_date < DATE '2010-04-01'
                          GROUP BY 1 ORDER BY 1""").fetchall()
    assert [r[quantity.slug] if quantity.slug in r else list(r.values())[-1] for r in rows] == [w[1] for w in want]
