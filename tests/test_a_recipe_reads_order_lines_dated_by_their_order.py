"""Dashboard recipes on a classic order book: orders with a date, their lines with a quantity and a unit price.

The lines have no date of their own and no amount: Learn works out the sales amount as the quantity times the
price on each line, added up, and a question dates a line by its order. The recipes read figures only on rows with
their own date, so this data had no sales at all ("needs sales with an amount and a date"); the orders, due on a
required date and shipped on another, were read as purchase orders for a "Supplier performance" dashboard; and the
line count was the "number of orders" an average order was worked out from.

Now a figure on lines is dated by their one order, as a question dates it; supplier performance needs a table of
purchases; an order is counted on the orders; a profit and loss needs a ledger or a cost (a sales amount alone is
none); and one SUM around the whole definition (units times their price) adds up, so its shares, ABC classes and
running totals are worked out as a summed column's.

A small warehouse in DuckDB, invented data; figures are checked against hand-written SQL.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import pytest

from core2 import recipe_book, recipes
from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.model.schema import SqlExpr
from core2.plan.ir import Plan
from core2.plan.values import MemberIndex
from core2.service import Services, Session, _answer_question
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 6, 30)


@pytest.fixture(scope="module")
def shop():
    con = duckdb.connect()
    con.execute("CREATE TABLE categories (category_id INTEGER PRIMARY KEY, category_name VARCHAR)")
    con.execute("CREATE TABLE suppliers (supplier_id INTEGER PRIMARY KEY, company_name VARCHAR, country VARCHAR)")
    con.execute("CREATE TABLE products (product_id INTEGER PRIMARY KEY, product_name VARCHAR, category_id INTEGER, "
                "supplier_id INTEGER)")
    con.execute("CREATE TABLE customers (customer_id VARCHAR PRIMARY KEY, company_name VARCHAR, country VARCHAR)")
    con.execute("CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id VARCHAR, order_date DATE, "
                "required_date DATE, shipped_date DATE, freight DECIMAL(10, 2))")
    con.execute("CREATE TABLE order_details (order_id INTEGER, product_id INTEGER, unit_price DECIMAL(10, 2), "
                "quantity INTEGER, PRIMARY KEY (order_id, product_id))")
    cats = ["Beverages", "Condiments", "Confections", "Dairy", "Grains", "Meat", "Produce", "Seafood"]
    con.executemany("INSERT INTO categories VALUES (?, ?)", list(enumerate(cats, 1)))
    con.executemany("INSERT INTO suppliers VALUES (?, ?, ?)",
                    [(s, f"Supplier {s:02d}", ["UK", "France", "Japan", "Brazil"][s % 4]) for s in range(1, 13)])
    price = {p: round(4 + (p * 37 % 60) + p / 10, 2) for p in range(1, 41)}
    con.executemany("INSERT INTO products VALUES (?, ?, ?, ?)",
                    [(p, f"Product {p:02d}", p % 8 + 1, p % 12 + 1) for p in range(1, 41)])
    con.executemany("INSERT INTO customers VALUES (?, ?, ?)",
                    [(f"C{c:03d}", f"Customer {c:03d}", ["Germany", "USA", "Spain", "Italy", "Canada"][c % 5])
                     for c in range(1, 61)])
    orders, lines = [], []
    for o in range(1, 1201):
        day = dt.date(2024, 7, 1) + dt.timedelta(days=o * 7 % 700)
        orders.append((o, f"C{o * 13 % 60 + 1:03d}", day, day + dt.timedelta(days=28),
                       day + dt.timedelta(days=3 + o % 9), round(5 + o % 40 * 1.5, 2)))
        for n in range(1 + o % 4):
            p = (o * 7 + n * 11) % 40 + 1
            lines.append((o, p, price[p], 1 + (o + n * 3) % 15))
    con.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", orders)
    con.executemany("INSERT INTO order_details VALUES (?, ?, ?, ?)", lines)
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t-order-book", options=BuildOptions(workers=1))
    return con, model


def _sales(model):
    return next(m for m in model.measures.values() if m.business_name == "Sales amount")


def _run(shop, plan: dict) -> dict:
    con, model = shop
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=None, index=MemberIndex(), today=TODAY)
    return _answer_question("q", services, Session(), planned=Plan.model_validate({"kind": "query", **plan}))


def test_a_figure_on_lines_is_dated_by_their_order(shop):
    _, model = shop
    sales = _sales(model)
    assert isinstance(sales.expr, SqlExpr) and not sales.default_date
    assert recipes.dated(model, sales)
    assert sales in recipe_book.figures(model, lambda t: True)


def test_the_sales_recipes_are_built_on_the_lines_and_the_orders(shop):
    _, model = shop
    states = {c["key"]: c for c in recipe_book.catalogue(model, today=TODAY)}
    assert states["sales_overview"]["state"] != "no"
    assert states["product_and_category"]["state"] != "no"
    assert states["supplier_performance"]["state"] == "no"          # a customer's order due on a date is no purchase
    assert states["profit_and_loss"]["state"] == "no"               # sales alone are no profit and loss
    built = recipe_book.build(model, "sales_overview", today=TODAY)
    titles = [t.title for t in built.tiles]
    assert "Number of orders" in titles and "Number of order details" not in titles
    assert "Average order" in titles


def test_the_average_order_is_the_sales_over_the_orders(shop):
    con, model = shop
    built = recipe_book.build(model, "sales_overview", today=TODAY)
    tile = next(t for t in built.tiles if t.title == "Average order")
    payload = _run(shop, {**tile.plan, "time": {"window": {"kind": "between", "start": "2025-01-01",
                                                            "end": "2025-12-31"}}})
    (row,) = payload["data"]["rows"]
    sales, orders = con.execute(
        "SELECT SUM(d.quantity * d.unit_price), COUNT(DISTINCT o.order_id) FROM orders o "
        "JOIN order_details d ON d.order_id = o.order_id WHERE o.order_date BETWEEN '2025-01-01' AND '2025-12-31'"
    ).fetchone()
    assert list(row.values())[0] == pytest.approx(float(sales) / orders, rel=1e-6)


def test_one_sum_of_each_lines_value_has_abc_classes_and_shares(shop):
    con, model = shop
    sales = _sales(model)
    year = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}
    payload = _run(shop, {"intent": "rank", "pareto": {"of": "product", "measure": sales.slug, "mode": "pareto",
                                                       "cuts": [80]}, "time": year})
    rows = payload["data"]["rows"]
    want = con.execute(
        "SELECT p.product_name, SUM(d.quantity * d.unit_price) AS v FROM order_details d "
        "JOIN orders o ON o.order_id = d.order_id JOIN products p ON p.product_id = d.product_id "
        "WHERE o.order_date BETWEEN '2025-01-01' AND '2025-12-31' GROUP BY 1 ORDER BY v DESC").fetchall()
    total = sum(float(v) for _, v in want)
    running, top = 0.0, []
    for name, v in want:
        if running >= 0.8 * total:
            break
        running += float(v)
        top.append(name)
    shown = [next(v for k, v in r.items() if isinstance(v, str)) for r in rows]
    assert shown == top
    running = _run(shop, {"intent": "trend", "measures": [sales.slug], "rolling": {"kind": "running"},
                          "time": {**year, "grain": "month"}})
    months = running["data"]["rows"]
    assert len(months) == 12
    assert months[-1][f"{sales.slug}_running"] == pytest.approx(total, rel=1e-6)    # the year's sales, added up


def _learn(con: duckdb.DuckDBPyConnection, name: str):
    warehouse = DuckDBWarehouse(con)
    return build_model(warehouse, from_duckdb(warehouse), client_id=name, options=BuildOptions(workers=1))


def test_customers_payments_are_a_rental_shops_sales_and_a_suppliers_are_not():
    """A rental shop keeps its takings as customers' payments ("Amount" on each): they are its sales, titled as
    such; payments made to suppliers never are."""
    con = duckdb.connect()
    con.execute("CREATE TABLE customer (customer_id INTEGER PRIMARY KEY, first_name VARCHAR, last_name VARCHAR, "
                "city VARCHAR)")
    con.executemany("INSERT INTO customer VALUES (?, ?, ?, ?)",
                    [(c, f"Ann{c}", f"Lee{c}", ["Leeds", "York", "Bath"][c % 3]) for c in range(1, 81)])
    con.execute("CREATE TABLE payment (payment_id INTEGER PRIMARY KEY, customer_id INTEGER, amount DECIMAL(8, 2), "
                "payment_date DATE)")
    con.executemany("INSERT INTO payment VALUES (?, ?, ?, ?)",
                    [(p, p % 80 + 1, [0.99, 2.99, 4.99][p % 3], dt.date(2025, 1, 1) + dt.timedelta(days=p % 500))
                     for p in range(1, 3001)])
    con.execute("CREATE TABLE ticket (ticket_id INTEGER PRIMARY KEY, customer_id INTEGER, opened_date DATE, "
                "topic VARCHAR, handling_minutes INTEGER)")
    con.executemany("INSERT INTO ticket VALUES (?, ?, ?, ?, ?)",
                    [(t, t % 80 + 1, dt.date(2025, 1, 1) + dt.timedelta(days=t % 500), ["Late", "Damaged"][t % 2],
                      5 + t * 7 % 55) for t in range(1, 2001)])
    model = _learn(con, "t-rental-shop")
    states = {c["key"]: c["state"] for c in recipe_book.catalogue(model, today=TODAY)}
    assert states["sales_overview"] != "no" and states["customer_health"] != "no"
    built = recipe_book.build(model, "sales_overview", today=TODAY)
    assert built.tiles[0].title == "Sales"
    assert not any("ticket" in t.title.lower() for t in built.tiles)     # support tickets are no orders

    con = duckdb.connect()
    con.execute("CREATE TABLE vendor (vendor_id INTEGER PRIMARY KEY, vendor_name VARCHAR)")
    con.executemany("INSERT INTO vendor VALUES (?, ?)", [(v, f"Vendor {v}") for v in range(1, 31)])
    con.execute("CREATE TABLE vendor_payment (payment_id INTEGER PRIMARY KEY, vendor_id INTEGER, "
                "amount DECIMAL(10, 2), payment_date DATE)")
    con.executemany("INSERT INTO vendor_payment VALUES (?, ?, ?, ?)",
                    [(p, p % 30 + 1, 100 + p % 70 * 3.5, dt.date(2025, 1, 1) + dt.timedelta(days=p % 500))
                     for p in range(1, 2001)])
    states = {c["key"]: c["state"] for c in recipe_book.catalogue(_learn(con, "t-paying-vendors"), today=TODAY)}
    assert states["sales_overview"] == "no"


def test_the_lines_value_is_the_sales_over_the_invoice_total():
    """An invoice keeps its total and its lines their quantity and price: the lines' value is the sales a recipe
    shows, as it is the one that splits by product."""
    con = duckdb.connect()
    con.execute("CREATE TABLE track (track_id INTEGER PRIMARY KEY, track_name VARCHAR, genre VARCHAR)")
    con.executemany("INSERT INTO track VALUES (?, ?, ?)",
                    [(t, f"Track {t:03d}", ["Rock", "Jazz", "Blues", "Latin", "Pop"][t % 5]) for t in range(1, 301)])
    con.execute("CREATE TABLE customer (customer_id INTEGER PRIMARY KEY, company VARCHAR, country VARCHAR)")
    con.executemany("INSERT INTO customer VALUES (?, ?, ?)",
                    [(c, f"Company {c:02d}", ["Brazil", "USA", "Canada", "France"][c % 4]) for c in range(1, 41)])
    con.execute("CREATE TABLE invoice (invoice_id INTEGER PRIMARY KEY, customer_id INTEGER, invoice_date DATE, "
                "total DECIMAL(10, 2))")
    con.execute("CREATE TABLE invoice_line (invoice_line_id INTEGER PRIMARY KEY, invoice_id INTEGER, "
                "track_id INTEGER, unit_price DECIMAL(6, 2), quantity INTEGER)")
    lines, invoices, n = [], [], 0
    for i in range(1, 801):
        total = 0.0
        for j in range(1 + i % 5):
            n += 1
            price = 0.99 if (i + j) % 4 else 1.99
            lines.append((n, i, (i * 7 + j * 13) % 300 + 1, price, 1))
            total += price
        invoices.append((i, i % 40 + 1, dt.date(2025, 1, 1) + dt.timedelta(days=i % 540), round(total, 2)))
    con.executemany("INSERT INTO invoice VALUES (?, ?, ?, ?)", invoices)
    con.executemany("INSERT INTO invoice_line VALUES (?, ?, ?, ?, ?)", lines)
    model = _learn(con, "t-music-shop")
    built = recipe_book.build(model, "sales_overview", today=TODAY)
    first = built.tiles[0]
    assert model.measures[next(k for k, m in model.measures.items() if m.slug == first.plan["measures"][0])] \
        .table.endswith("invoice_line")
    titles = [t.title for t in built.tiles]
    assert any(t.startswith("Top 10 tracks") for t in titles)
    assert any(t.endswith("by genre") for t in titles)
