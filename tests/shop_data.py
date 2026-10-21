"""
tests/shop_data.py

An invented shop's warehouse, learned as Learn learns one, for the report tests.

Not a test module. Its shape is the one that broke the reports: order lines keyed by their order number and line
number (no table of orders anywhere), returns keyed the same way and pointing at the lines they return, two
customers who share a name, walk-in sales with no customer on file, and cancelled lines a table leaves out by
default. ``ask`` puts a question through the whole answer path with the AI stood in by the plan it wrote.

Invented names and figures only.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.values import build_index
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 8, 24)            # the data runs 3 Jan 2025 - 23 Aug 2026
STORES = [(1, "North Point"), (2, "Quay Street"), (3, "Hill Road")]
# Two customers share a name, in different cities: told apart by their number.
CUSTOMERS = [(1, "Ari Lane", "Retail", "Easton"), (2, "Ari Lane", "Business", "Weston"),
             (3, "Bo Marsh", "Retail", "Easton"), (4, "Cy Pell", "Business", "Norton"),
             (5, "Di Rowe", "Retail", "Weston"), (6, "Ed Sims", "Retail", "Norton")]
PRODUCTS = [(1, "Dish Soap", "Household"), (2, "Paper Towels", "Household"), (3, "Sponges", "Household"),
            (4, "Rice", "Pantry")]


def _lines() -> list[tuple]:
    """(order_id, line, order_date, customer_id, store_id, product_id, quantity, net_amount, status): three orders
    every third day, one in seven by a walk-in with no customer on file, one in twenty-nine cancelled."""
    out, order = [], 1000
    for day in range(0, 600, 3):
        when = dt.date(2025, 1, 3) + dt.timedelta(days=day)
        for k in range(3):
            order += 1
            customer = None if (day + k) % 7 == 0 else (day // 3 + k) % 6 + 1
            products = [1 + (day + k) % 4, 1 + (day + k + 1) % 4] if k != 1 else [1 + day % 4]
            for line, product in enumerate(products, 1):
                quantity = 1 + (day + line) % 4
                out.append((order, line, when, customer, (day + k) % 3 + 1, product, quantity,
                            round(quantity * (40 + 15 * product), 2),
                            "Cancelled" if (day + k) % 29 == 0 else "Completed"))
    return out


def build():
    """(connection, model, member index) of the shop, learned."""
    con = duckdb.connect()
    con.execute("CREATE TABLE stores (store_id INTEGER PRIMARY KEY, store_name VARCHAR NOT NULL)")
    con.execute("CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, customer_name VARCHAR NOT NULL, "
                "segment VARCHAR NOT NULL, city VARCHAR NOT NULL)")
    con.execute("CREATE TABLE products (product_id INTEGER PRIMARY KEY, product_name VARCHAR NOT NULL, "
                "category VARCHAR NOT NULL)")
    con.execute("CREATE TABLE order_lines (order_id INTEGER, line INTEGER, order_date DATE NOT NULL, "
                "customer_id INTEGER REFERENCES customers(customer_id), store_id INTEGER NOT NULL REFERENCES "
                "stores(store_id), product_id INTEGER NOT NULL REFERENCES products(product_id), quantity INTEGER, "
                "net_amount DECIMAL(12,2), status VARCHAR, PRIMARY KEY (order_id, line))")
    # The units that came back are a "quantity" too: two figures of one name on two tables.
    con.execute("CREATE TABLE returns (order_id INTEGER, line INTEGER, return_date DATE, quantity INTEGER, "
                "refund_amount DECIMAL(12,2), PRIMARY KEY (order_id, line), "
                "FOREIGN KEY (order_id, line) REFERENCES order_lines(order_id, line))")
    con.executemany("INSERT INTO stores VALUES (?, ?)", STORES)
    con.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", CUSTOMERS)
    con.executemany("INSERT INTO products VALUES (?, ?, ?)", PRODUCTS)
    rows = _lines()
    con.executemany("INSERT INTO order_lines VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.executemany("INSERT INTO returns VALUES (?, ?, ?, ?, ?)",
                    [(o, n, d + dt.timedelta(days=9), 1, round(a / q, 2)) for o, n, d, _c, _s, _p, q, a, st in rows
                     if st == "Completed" and (o + n) % 11 == 0])
    warehouse = DuckDBWarehouse(con)
    # Discovery reports the returns' two-column key to the order lines.
    declared = [{"constraint_name": "fk_returns_line", "parent_table": "returns", "parent_col": column,
                 "ref_table": "order_lines", "ref_col": column, "ordinal": n, "enforced": False}
                for n, column in enumerate(("order_id", "line"), 1)]
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=declared), client_id="t-shop",
                        options=BuildOptions(workers=1))

    def fetch(slug: str) -> list[object]:
        column = model.columns[model.attributes[slug].column]
        table = model.tables[column.table]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM "{table.name}"').rows]

    return con, model, build_index(model, fetch)


def ask(shop, question: str, plan: dict, session: Session | None = None) -> dict:
    """The answer to ``question``, the AI stood in by the plan it wrote; ``session`` carries a thread."""
    con, model, index = shop
    written = json.dumps({"kind": "query", **plan})
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda stable, tail: written,
                        index=index, today=TODAY)
    return answer_question(question, services, session if session is not None else Session())


def rows(payload: dict) -> list[dict]:
    """Every row of the answer, as exported."""
    return payload.get("export_rows") or (payload.get("data") or {}).get("rows") or []


def headline(payload: dict) -> str:
    return (payload.get("answer") or {}).get("headline") or ""
