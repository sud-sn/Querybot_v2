"""Order lines dated by their order are what happened, as much as returns dated by their own day.

"Which products are most often bought together with the Barista One Espresso Machine?" answered "QuickBoil Kettle,
on 2 orders": the baskets were read on the returns. An order counted by product is counted on rows that name both,
and of those the rows of what happened come first; returns have a return date of their own, while order lines,
dated only by their order, were taken for undated rows and put after them. A line dated by its document is dated:
the order lines are what was bought, and the returns are said to name both too.

Invented data only; every figure is checked against SQL written by hand.
"""
from __future__ import annotations

import datetime as dt

import duckdb
import pytest

from core2.bootstrap.inventory import from_duckdb
from core2.bootstrap.build import BuildOptions, build_model
from core2.compile.compiler import compile_query
from core2.plan.ir import Plan
from core2.resolve.resolver import Context, resolve
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 6, 15)


def _fk(parent, col, ref, rcol):
    return {"constraint_name": f"fk_{parent}_{col}", "parent_schema": "main", "parent_table": parent,
            "parent_col": col, "ref_schema": "main", "ref_table": ref, "ref_col": rcol, "ordinal": 1,
            "enforced": False, "source": "synthetic"}


@pytest.fixture(scope="module")
def shop():
    con = duckdb.connect()
    con.execute("CREATE TABLE products (product_id INTEGER PRIMARY KEY, product_name VARCHAR)")
    con.execute("CREATE TABLE orders (order_id INTEGER PRIMARY KEY, order_date DATE, ship_date DATE, "
                "customer_name VARCHAR, freight_amount DECIMAL(12,2))")
    con.execute("CREATE TABLE order_lines (order_id INTEGER, line_no INTEGER, product_id INTEGER, quantity INTEGER, "
                "net_amount DECIMAL(12,2))")
    con.execute("CREATE TABLE returns (return_id INTEGER PRIMARY KEY, order_id INTEGER, line_no INTEGER, "
                "product_id INTEGER, return_date DATE, return_qty INTEGER, credit_amount DECIMAL(12,2))")
    names = ["Espresso Machine", "Kettle", "Toaster", "Blender", "Air Fryer", "Mixer"]
    con.executemany("INSERT INTO products VALUES (?, ?)", list(enumerate(names, 1)))
    lines, returns, rid = [], [], 1
    for o in range(1, 601):
        day = dt.date(2025, 1, 1) + dt.timedelta(days=o % 300)
        con.execute("INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
                    [o, day, day + dt.timedelta(days=2 + o % 4), f"Customer {o % 40}", 5.0 + o % 11])
        items = [1, 2 + (o % 5)] if o % 3 else [3 + (o % 3), 2 + (o % 4) % 4 + 1]
        for n, p in enumerate(dict.fromkeys(items), 1):
            lines.append((o, n, p, 1 + o % 3, 10.0 * p + o % 7))
            if o % 23 == 0:
                returns.append((rid, o, n, p, dt.date(2025, 3, 1) + dt.timedelta(days=o % 200), 1, 9.5))
                rid += 1
    con.executemany("INSERT INTO order_lines VALUES (?, ?, ?, ?, ?)", lines)
    con.executemany("INSERT INTO returns VALUES (?, ?, ?, ?, ?, ?, ?)", returns)
    fks = [_fk("order_lines", "order_id", "orders", "order_id"), _fk("order_lines", "product_id", "products",
                                                                       "product_id"),
           _fk("returns", "order_id", "orders", "order_id"), _fk("returns", "product_id", "products", "product_id")]
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=fks), client_id="eval-lines",
                        options=BuildOptions(workers=1))
    return con, model


def _orders(model):
    """The count of orders: the measure counting the orders table's own rows."""
    return next(m.slug for m in model.measures.values() if m.table.endswith(".orders") and m.kind == "count")


def test_orders_by_product_are_counted_on_the_order_lines(shop):
    con, model = shop
    orders = _orders(model)
    product = next(e.slug for e in model.entities.values() if "product" in e.slug)
    plan = Plan.model_validate({"kind": "query", "intent": "breakdown", "measures": [orders], "group_by": [product]})
    logical = resolve(plan, model, Context(today=TODAY))
    sql = compile_query(logical, model, "duckdb").sql
    got = {r[0]: r[-1] for r in con.execute(sql).fetchall()}
    want = dict(con.execute("SELECT p.product_name, COUNT(DISTINCT l.order_id) FROM order_lines l JOIN products p "
                            "USING (product_id) GROUP BY 1").fetchall())
    assert got == want, sql


def test_bought_together_reads_the_baskets_on_the_order_lines(shop):
    con, model = shop
    orders = _orders(model)
    product = next(e.slug for e in model.entities.values() if "product" in e.slug)
    plan = Plan.model_validate({"kind": "query", "intent": "breakdown", "together": {"of": product, "basket": orders}})
    sql = compile_query(resolve(plan, model, Context(today=TODAY)), model, "duckdb").sql
    rows = con.execute(sql).fetchall()
    top = max(rows, key=lambda r: r[2])
    want = con.execute("""SELECT a.p, b.p, COUNT(*) n FROM (SELECT DISTINCT order_id, product_id p FROM order_lines) a
        JOIN (SELECT DISTINCT order_id, product_id p FROM order_lines) b ON a.order_id = b.order_id AND a.p <> b.p
        GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 1""").fetchone()
    assert top[2] == want[2], (top, want, sql)


def test_a_basket_named_by_what_it_is_is_counted_by_the_count_of_those_rows(shop):
    """The AI names the basket by what it is ("order") where the plan wants the count of them: the same pairs."""
    con, model = shop
    product = next(e.slug for e in model.entities.values() if "product" in e.slug)
    by_list = Plan.model_validate({"kind": "query", "intent": "breakdown",
                                   "together": {"of": product, "basket": "order"}})
    by_count = Plan.model_validate({"kind": "query", "intent": "breakdown",
                                    "together": {"of": product, "basket": _orders(model)}})
    rows = [sorted(con.execute(compile_query(resolve(p, model, Context(today=TODAY)), model, "duckdb").sql).fetchall())
            for p in (by_list, by_count)]
    assert rows[0] == rows[1] and rows[0]


def test_a_basket_named_by_the_item_itself_is_the_document_its_rows_belong_to(shop):
    """The AI puts the quoted item where the basket goes: the baskets are still the orders the lines belong to."""
    con, model = shop
    product = next(e.slug for e in model.entities.values() if "product" in e.slug)
    by_item = Plan.model_validate({"kind": "query", "intent": "breakdown",
                                   "together": {"of": product, "basket": "Espresso Machine"}})
    by_count = Plan.model_validate({"kind": "query", "intent": "breakdown",
                                    "together": {"of": product, "basket": _orders(model)}})
    rows = [sorted(con.execute(compile_query(resolve(p, model, Context(today=TODAY)), model, "duckdb").sql).fetchall())
            for p in (by_item, by_count)]
    assert rows[0] == rows[1] and rows[0]
