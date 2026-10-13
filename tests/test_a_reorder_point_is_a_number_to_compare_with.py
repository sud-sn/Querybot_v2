"""A product's reorder point and safety stock level are numbers to compare with, though they take few values.

Found by the sector stress test on a manufacturer's database (October 2026): the products' REORDER_POINT and
SAFETY_STOCK_LEVEL take a handful of values each (375, 750 ...), so Learn kept them as groupings, like a pack
size, and "products whose stock is below their reorder point" could not be asked. A line a member's figures are
held against is a number, however few its values; a few-valued number that names no such line (an education
level) is still a grouping.

Invented rows only.
"""

from __future__ import annotations

import duckdb

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.warehouse.runner import DuckDBWarehouse


def test_a_reorder_point_is_a_number_and_an_education_level_a_group():
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE products AS SELECT range + 1 AS product_id, 'Product ' || range || repeat('x', range % 3) AS product_name,
               [375, 750, 3][range % 3 + 1] AS reorder_point, [500, 1000, 4][range % 3 + 1] AS safety_stock_level
        FROM range(400);
        CREATE TABLE customers AS SELECT range + 1 AS customer_id, 'Customer ' || range || repeat('y', range % 4) AS customer_name,
               range % 5 + 1 AS education_level FROM range(600);
        CREATE TABLE sales AS SELECT range + 1 AS sale_id, range % 400 + 1 AS product_id, range % 600 + 1 AS customer_id,
               DATE '2024-01-01' + (range % 500)::INTEGER AS sale_date, range % 9 + 1 AS quantity FROM range(8000);
    """)
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t-rop", options=BuildOptions(workers=1))
    kinds = {model.columns[a.column].name: a.kind for a in model.attributes.values()}
    assert kinds["reorder_point"] == "number" and kinds["safety_stock_level"] == "number", kinds
    assert kinds["education_level"] == "group", kinds
