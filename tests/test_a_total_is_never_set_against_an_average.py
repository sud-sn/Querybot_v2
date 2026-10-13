"""A figure worked out from a total and an average of the same rows is sent back to the planner.

Found by the sector stress test on the TPC-DS retail benchmark (October 2026): "the 2002 store discount rate
(discount over list price)" was planned as the discounts added up over the list price per unit, averaged -- a
discount rate of 71,022,329.7%. A total against an average is no figure at all; both must be totals, and the
total the average is the unit of (the extended list price) is named to the planner.

Invented rows only.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.ir import Plan
from core2.plan.planner import check
from core2.warehouse.runner import DuckDBWarehouse


@pytest.fixture(scope="module")
def model():
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE item AS SELECT range + 1 AS item_id, 'Item ' || range || repeat('x', range % 3) AS item_name FROM range(200);
        CREATE TABLE store_sales AS SELECT range + 1 AS sale_id, range % 200 + 1 AS item_id,
               DATE '2002-01-01' + (range % 365)::INTEGER AS sold_date, range % 9 + 1 AS quantity,
               round(10 + (range * 7) % 90 + 0.5, 2) AS list_price FROM range(9000);
        ALTER TABLE store_sales ADD COLUMN ext_list_price DECIMAL(12, 2);
        ALTER TABLE store_sales ADD COLUMN ext_discount_amt DECIMAL(12, 2);
        UPDATE store_sales SET ext_list_price = quantity * list_price,
                               ext_discount_amt = round(quantity * list_price * (sale_id % 4) / 10.0, 2);
    """)
    warehouse = DuckDBWarehouse(con)
    return build_model(warehouse, from_duckdb(warehouse), client_id="t-avg", options=BuildOptions(workers=1))


def _slug(model, column):
    return next(m.slug for m in model.measures.values() if getattr(m.expr, "column", "") and
                m.expr.column.endswith("." + column))


def test_a_discount_over_a_unit_price_is_sent_back_with_the_total_to_use(model):
    plan = Plan.model_validate({"kind": "query", "derived": [{"name": "Discount rate", "op": "ratio", "scale": 100,
                                                            "measures": [_slug(model, "ext_discount_amt"),
                                                                         _slug(model, "list_price")]}]})
    problems, _ = check(plan, model, dt.date(2003, 1, 1))
    assert problems and _slug(model, "ext_list_price") in problems[0], problems


def test_a_discount_over_the_extended_list_price_stands(model):
    plan = Plan.model_validate({"kind": "query", "derived": [{"name": "Discount rate", "op": "ratio", "scale": 100,
                                                            "measures": [_slug(model, "ext_discount_amt"),
                                                                         _slug(model, "ext_list_price")]}]})
    assert check(plan, model, dt.date(2003, 1, 1))[0] == []
