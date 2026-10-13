"""What was sold is a line's price less the discount given on it, when no column holds it.

Found by the sector stress test on a public wholesale distributor's data (October 2026): order lines keep the
extended price, a discount rate between 0 and 10% and a tax rate, as the TPC-H benchmark keeps them, and no
column holds the price after its discount. Learn found the extended price (summed) and the two rates
(averaged), but not revenue: every question about revenue, sales, the top customers or the discount given
was answered with the gross price -- 21 of the book's 60 steps. A line amount less a rate of it, worked out on
each row and added up, is net sales; the amount times the rate is the discount given; a tax rate is a tax on
what was charged. When a net amount is on the row already, it is the figure, and nothing is worked out.

Invented rows only.
"""

from __future__ import annotations

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.ir import Plan
import datetime as dt

from core2.resolve.resolver import Context, resolve
from core2.compile.compiler import compile_query
from core2.warehouse.runner import DuckDBWarehouse

LINES = """
    CREATE TABLE orders AS SELECT range + 1 AS o_orderkey, DATE '2024-01-01' + (range % 700)::INTEGER AS o_orderdate,
           (range * 7) % 300 + 1 AS o_custkey FROM range(4000);
    CREATE TABLE lineitem AS SELECT range // 4 + 1 AS l_orderkey, range % 4 + 1 AS l_linenumber,
           (range * 13) % 500 + 1 AS l_partkey, range % 50 + 1 AS l_quantity,
           round((range % 50 + 1) * (900 + (range * 31) % 1100) / 10.0, 2) AS l_extendedprice,
           ((range * 7) % 11) / 100.0 AS l_discount, ((range * 3) % 9) / 100.0 AS l_tax FROM range(16000);
"""


def _model(sql: str):
    con = duckdb.connect()
    con.execute(sql)
    warehouse = DuckDBWarehouse(con)
    return con, build_model(warehouse, from_duckdb(warehouse), client_id="t-net", options=BuildOptions(workers=1))


@pytest.fixture(scope="module")
def lines():
    return _model(LINES)


def _figure(con, model, measure) -> float:
    logical = resolve(Plan(kind="query", measures=[measure.slug]), model, Context(today=dt.date(2026, 1, 1)))
    sql = compile_query(logical, model, "duckdb").sql
    return float(con.execute(sql).fetchone()[0])


def _named(model, name):
    return next(m for m in model.measures.values() if m.business_name == name)


def test_net_sales_are_the_extended_price_less_its_discount(lines):
    con, model = lines
    net = _named(model, "Net sales")
    assert {"revenue", "sales"} <= set(net.synonyms["en"])
    want = con.execute("SELECT SUM(l_extendedprice * (1 - l_discount)) FROM lineitem").fetchone()[0]
    assert _figure(con, model, net) == pytest.approx(float(want), rel=1e-9)


def test_the_discount_given_and_the_tax_charged_are_added_up(lines):
    con, model = lines
    given = con.execute("SELECT SUM(l_extendedprice * l_discount) FROM lineitem").fetchone()[0]
    taxed = con.execute("SELECT SUM(l_extendedprice * (1 - l_discount) * l_tax) FROM lineitem").fetchone()[0]
    assert _figure(con, model, _named(model, "Discount amount")) == pytest.approx(float(given), rel=1e-9)
    assert _figure(con, model, _named(model, "Tax amount")) == pytest.approx(float(taxed), rel=1e-9)
    assert _named(model, "Discount").additivity == "non_additive", "the rate itself is still averaged"


def test_a_net_amount_on_the_row_is_the_figure(lines):
    con, model = _model(LINES + """
        ALTER TABLE lineitem ADD COLUMN l_net_amount DECIMAL(12, 2);
        UPDATE lineitem SET l_net_amount = round(l_extendedprice * (1 - l_discount), 2);
    """)
    assert not [m for m in model.measures.values() if m.business_name == "Net sales"]
    assert not [m for m in model.measures.values() if ".less." in m.key]


def test_a_line_total_already_after_its_discount_is_not_discounted_again():
    """A sales order detail as AdventureWorks keeps it: the line total is quantity x unit price x (1 - discount)."""
    con, model = _model("""
        CREATE TABLE sales_order_header AS SELECT range + 1 AS sales_order_id,
               DATE '2024-01-01' + (range % 700)::INTEGER AS order_date FROM range(3000);
        CREATE TABLE sales_order_detail AS SELECT range // 3 + 1 AS sales_order_id, range % 3 + 1 AS line_number,
               range % 12 + 1 AS order_qty, round(5 + (range * 17) % 400 + 0.99, 2) AS unit_price,
               [0.0, 0.0, 0.0, 0.02, 0.05, 0.1][range % 6 + 1] AS unit_price_discount FROM range(9000);
        ALTER TABLE sales_order_detail ADD COLUMN line_total DECIMAL(14, 4);
        UPDATE sales_order_detail SET line_total = order_qty * unit_price * (1 - unit_price_discount);
    """)
    assert not [m for m in model.measures.values() if ".less." in m.key], \
        [m.business_name for m in model.measures.values()]
