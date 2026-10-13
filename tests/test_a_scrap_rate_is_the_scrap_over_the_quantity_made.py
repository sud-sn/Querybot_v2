"""A scrap rate is the quantity scrapped over the quantity made, both figures of the same work orders.

Found by the sector stress test on a manufacturer's database (October 2026): "what was our scrap rate in 2024?"
and "the rejection rate on received goods" were planned as rates of the quantity made with the condition
"scrapped quantity above 0" -- and refused, a rate's condition being only a member of a field ("nothing to narrow
the rows by called scrapped_quantity"). A rate of a total whose condition is another total of the same rows is
the one over the other; a rate of a count with a condition on a figure counts the rows whose figure meets it.

Invented rows only.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.compile.compiler import compile_query
from core2.plan.ir import Plan
from core2.resolve.resolver import Context, resolve
from core2.warehouse.runner import DuckDBWarehouse


@pytest.fixture(scope="module")
def shop():
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE products AS SELECT range + 1 AS product_id, 'Product ' || range || repeat('x', range % 3) AS product_name
        FROM range(50);
        CREATE TABLE work_orders AS SELECT range + 1 AS work_order_id, range % 50 + 1 AS product_id,
               DATE '2024-01-01' + (range % 600)::INTEGER AS start_date, 20 + range % 80 AS order_quantity,
               CASE WHEN range % 9 = 0 THEN range % 6 + 1 ELSE 0 END AS scrapped_quantity FROM range(6000);
    """)
    warehouse = DuckDBWarehouse(con)
    return con, build_model(warehouse, from_duckdb(warehouse), client_id="t-scrap", options=BuildOptions(workers=1))


def _value(shop, rate: dict) -> float:
    con, model = shop
    logical = resolve(Plan.model_validate({"kind": "query", "rates": [rate]}), model, Context(today=dt.date(2026, 1, 1)))
    return float(con.execute(compile_query(logical, model, "duckdb").sql).fetchone()[0])


def _slug(model, column):
    return next(m.slug for m in model.measures.values() if getattr(m.expr, "column", "") and
                m.expr.column.endswith("." + column) and m.expr.agg == "sum")


def test_the_scrap_rate_is_the_scrapped_over_the_ordered(shop):
    con, model = shop
    rate = {"name": "Scrap rate", "measure": _slug(model, "order_quantity"),
            "where": [{"field": _slug(model, "scrapped_quantity"), "op": "gt", "values": [0]}]}
    want = con.execute("SELECT 100.0 * SUM(scrapped_quantity) / SUM(order_quantity) FROM work_orders").fetchone()[0]
    assert _value(shop, rate) == pytest.approx(float(want))


def test_the_share_of_work_orders_with_scrap_counts_those_rows(shop):
    con, model = shop
    count = next(m.slug for m in model.measures.values() if m.table.endswith(".work_orders") and m.kind == "count"
                 and getattr(m.expr, "agg", "") == "count")
    rate = {"name": "Share with scrap", "measure": count,
            "where": [{"field": _slug(model, "scrapped_quantity"), "op": "gt", "values": [0]}]}
    want = con.execute("SELECT 100.0 * COUNT(*) FILTER (WHERE scrapped_quantity > 0) / COUNT(*) FROM work_orders"
                       ).fetchone()[0]
    assert _value(shop, rate) == pytest.approx(float(want))
