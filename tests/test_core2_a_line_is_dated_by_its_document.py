"""An order line with no date of its own is dated by its order.

On a public sample of an operational database, sales order lines keep their quantities and prices while the
order date is on the order header. "Top 10 products by quantity ordered in 2024" and "how many order lines had a
discount in 2024" were refused: "Sales order detail has no date". A table with no date of its own, whose rows
belong to one dated document, is dated by that document's date, joined through the line's link to it.

Invented data only.
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

TODAY = dt.date(2026, 6, 15)


@pytest.fixture(scope="module")
def shop():
    con = duckdb.connect()
    con.execute("CREATE TABLE Product (ProductID INTEGER, Name VARCHAR, ListPrice DECIMAL(10,2))")
    con.execute("INSERT INTO Product SELECT i, 'Product ' || i || repeat('x', i % 7), 10 + i FROM range(1, 61) t(i)")
    con.execute("CREATE TABLE SalesOrderHeader (SalesOrderID INTEGER, OrderDate DATE, CustomerID INTEGER, "
                "SubTotal DECIMAL(12,2))")
    con.execute("INSERT INTO SalesOrderHeader SELECT i, DATE '2024-01-01' + CAST(i % 730 AS INTEGER), i % 300 + 1, "
                "100 + i % 900 FROM range(1, 3001) t(i)")
    con.execute("CREATE TABLE SalesOrderDetail (SalesOrderID INTEGER, SalesOrderDetailID INTEGER, ProductID INTEGER, "
                "OrderQty INTEGER, UnitPrice DECIMAL(10,2), UnitPriceDiscount DECIMAL(4,2))")
    con.execute("INSERT INTO SalesOrderDetail SELECT o, o * 10 + l, CAST(hash(o * 10 + l) % 60 AS INTEGER) + 1, (o + l) % 5 + 1, "
                "10 + (o % 50), CASE WHEN (o + l) % 9 = 0 THEN 0.1 ELSE 0 END FROM range(1, 3001) a(o), range(1, 4) b(l)")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="lines", options=BuildOptions(workers=1))
    return con, model


def _ask(shop, plan: dict, question: str) -> dict:
    con, model = shop
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, t: json.dumps(plan),
                        index=MemberIndex(), today=TODAY)
    return answer_question(question, services, Session())


def _slug(model, table: str, column: str) -> str:
    key = next(t.key for t in model.tables.values() if t.name == table)
    return next(m.slug for m in model.measures.values()
                if m.table == key and getattr(m.expr, "column", "") == f"{key}.{column.casefold()}")


def test_order_lines_are_counted_by_their_orders_date(shop):
    con, model = shop
    qty = _slug(model, "SalesOrderDetail", "OrderQty")
    payload = _ask(shop, {"kind": "query", "intent": "trend", "measures": [qty], "time": {
        "grain": "month", "window": {"kind": "between", "start": "2024-01-01", "end": "2024-12-31"}}},
        "Quantity ordered by month in 2024")
    want = {r[0]: float(r[1]) for r in con.execute(
        "SELECT month(h.OrderDate), SUM(d.OrderQty) FROM SalesOrderDetail d JOIN SalesOrderHeader h "
        "ON d.SalesOrderID = h.SalesOrderID WHERE year(h.OrderDate) = 2024 GROUP BY 1").fetchall()}
    got = sorted(float(next(v for k, v in r.items() if k == qty)) for r in payload["export_rows"])
    assert got == sorted(want.values()), payload["answer"]["headline"]


def test_top_products_in_a_year_rank_their_order_lines(shop):
    con, model = shop
    qty = _slug(model, "SalesOrderDetail", "OrderQty")
    product = next(e.slug for e in model.entities.values() if model.tables[e.table].name == "Product")
    payload = _ask(shop, {"kind": "query", "intent": "rank", "measures": [qty], "group_by": [product],
                          "sort": [{"by": qty, "desc": True}], "limit": 3,
                          "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}},
                   "Top 3 products by quantity ordered in 2025")
    want = [float(r[0]) for r in con.execute(
        "SELECT SUM(d.OrderQty) v FROM SalesOrderDetail d JOIN SalesOrderHeader h ON d.SalesOrderID = h.SalesOrderID "
        "WHERE year(h.OrderDate) = 2025 GROUP BY d.ProductID ORDER BY v DESC, d.ProductID LIMIT 3").fetchall()]
    assert [float(r[qty]) for r in payload["export_rows"]] == want, payload["answer"]["headline"]


def test_a_line_with_only_a_due_date_and_a_change_stamp_is_dated_by_its_order():
    """ "Rejected quantity by vendor in 2025" was refused, "Purchase order detail has no date": the purchase order
    lines keep a due date and the time each row was last changed, neither of which is when the line was ordered."""
    con = duckdb.connect()
    con.execute("CREATE TABLE Vendor (VendorID INTEGER, Name VARCHAR)")
    con.execute("INSERT INTO Vendor SELECT i, 'Vendor ' || i || repeat('v', i % 5) FROM range(1, 41) t(i)")
    con.execute("CREATE TABLE PurchaseOrderHeader (PurchaseOrderID INTEGER, VendorID INTEGER, OrderDate DATE, "
                "Freight DECIMAL(10,2))")
    con.execute("INSERT INTO PurchaseOrderHeader SELECT i, i % 40 + 1, DATE '2024-01-01' + CAST(i % 700 AS INTEGER), "
                "5 + i % 30 FROM range(1, 2001) t(i)")
    con.execute("CREATE TABLE PurchaseOrderDetail (PurchaseOrderID INTEGER, PurchaseOrderDetailID INTEGER, "
                "DueDate DATE, OrderQty INTEGER, RejectedQty INTEGER, ModifiedDate TIMESTAMP)")
    con.execute("INSERT INTO PurchaseOrderDetail SELECT o, o * 10 + l, DATE '2024-01-15' + CAST(o % 700 AS INTEGER), "
                "10 + (o + l) % 20, CASE WHEN (o + l) % 7 = 0 THEN (o + l) % 5 ELSE 0 END, "
                "TIMESTAMP '2026-03-01 10:00:00' + INTERVAL (o % 60) MINUTE FROM range(1, 2001) a(o), range(1, 4) b(l)")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="po", options=BuildOptions(workers=1))
    rejected = _slug(model, "PurchaseOrderDetail", "RejectedQty")
    vendor = next(e.slug for e in model.entities.values() if model.tables[e.table].name == "Vendor")
    payload = _ask((con, model), {"kind": "query", "intent": "rank", "measures": [rejected], "group_by": [vendor],
                                  "sort": [{"by": rejected, "desc": True}], "limit": 3,
                                  "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}},
                   "Rejected quantity by vendor in 2025, top 3")
    want = [float(r[0]) for r in con.execute(
        "SELECT SUM(d.RejectedQty) v FROM PurchaseOrderDetail d JOIN PurchaseOrderHeader h "
        "ON d.PurchaseOrderID = h.PurchaseOrderID WHERE year(h.OrderDate) = 2025 GROUP BY h.VendorID "
        "ORDER BY v DESC LIMIT 3").fetchall()]
    assert [float(r[rejected]) for r in payload["export_rows"]] == want, payload["answer"]["headline"]
