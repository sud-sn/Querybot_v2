"""A name finds its member in any case, on a column whose members are never read.

Found by the sector stress test on a real online retailer's invoices (October 2026): "How many of the "Regency
Cakestand 3 Tier" did we sell in 2011?" was answered "no value". The product's description is stored in capitals
(REGENCY CAKESTAND 3 TIER), and the old masking rule keeps a description's members from being read (free text),
so the name could not be put in the stored spelling first, and the warehouse compared the text exactly. On such a
column the comparison ignores case; on a column whose members are read, the name is spelled as stored before
the query, as before.

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
        CREATE TABLE products AS SELECT 'P' || range AS stock_code,
               ['GARDEN TRUG SET OF 2', 'TEA LIGHT HOLDER BLUE', 'MINI CAKE STAND'][range % 3 + 1] || ' ' || range
               AS description FROM range(60);
        CREATE TABLE invoice_lines AS SELECT range + 1 AS line_id, 'P' || (range % 60) AS stock_code,
               DATE '2025-01-01' + (range % 300)::INTEGER AS invoice_date, 1 + range % 7 AS quantity,
               round(2.5 + range % 11, 2) AS unit_price FROM range(3000);
    """)
    warehouse = DuckDBWarehouse(con)
    return con, build_model(warehouse, from_duckdb(warehouse), client_id="t-case", options=BuildOptions(workers=1))


def _total(shop, value: str, *, readable: bool) -> float | None:
    con, model = shop
    description = next(c for c in model.columns.values() if c.name == "description")
    description.values_allowed = readable
    quantity = next(m.slug for m in model.measures.values() if getattr(m.expr, "column", "").endswith(".quantity"))
    product = next(e.slug for e in model.entities.values() if e.table.endswith(".products"))
    plan = Plan.model_validate({"kind": "query", "measures": [quantity],
                                "filters": [{"field": product, "op": "eq", "values": [value]}]})
    sql = compile_query(resolve(plan, model, Context(today=dt.date(2026, 1, 1))), model, "duckdb").sql
    return con.execute(sql).fetchone()[0]


def test_a_name_in_another_case_finds_the_member_whose_values_are_not_read(shop):
    con, _ = shop
    want = con.execute("SELECT SUM(quantity) FROM invoice_lines WHERE stock_code = 'P7'").fetchone()[0]
    assert _total(shop, "Tea Light Holder Blue 7", readable=False) == want


def test_a_column_whose_members_are_read_compares_as_stored(shop):
    """There the name was put in the stored spelling first (the member index): the text compares exactly."""
    assert _total(shop, "Tea Light Holder Blue 7", readable=True) is None
    assert _total(shop, "TEA LIGHT HOLDER BLUE 7", readable=True) is not None
