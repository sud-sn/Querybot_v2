"""A short list is named by its names though one of them is missing.

Found by the sector stress test on the TPC-DS retail benchmark (October 2026): of its five warehouses, four have
a name and one none. The names were filled on four rows in five, short of the nine in ten a name must be, so
every answer by warehouse showed the business codes instead ("AAAAAAAABAAAAAAA leads with 4.29M"). On a short
list, all names but one is enough: the one with none is shown by its code.

Invented rows only.
"""

from __future__ import annotations

import duckdb

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.warehouse.runner import DuckDBWarehouse


def test_four_warehouses_of_five_named_are_named():
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE warehouse AS SELECT range + 1 AS w_warehouse_sk, 'AAAAAAAA' || chr(66 + range::INTEGER) || 'AAAAAAA'
               AS w_warehouse_id, CASE WHEN range < 4 THEN ['North hub', 'Coastal depot', 'Valley store', 'East yard'][range + 1]
               END AS w_warehouse_name, 10000 + range * 2500 AS w_warehouse_sq_ft FROM range(5);
        CREATE TABLE inventory AS SELECT DATE '2024-01-07' + (w.range * 7)::INTEGER AS inv_date, i.range + 1 AS inv_item_sk,
               i.range % 5 + 1 AS inv_warehouse_sk, (i.range * 13 + w.range) % 400 AS inv_quantity_on_hand
        FROM range(30) w, range(100) i;
    """)
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t-wh", options=BuildOptions(workers=1))
    entity = next(e for e in model.entities.values() if e.table.endswith(".warehouse"))
    assert entity.label_column and entity.label_column.endswith(".w_warehouse_name"), entity.label_column
