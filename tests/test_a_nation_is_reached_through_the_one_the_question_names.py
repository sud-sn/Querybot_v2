"""A field the rows reach by two routes is reached through the one the question names, and the answer says which.

Found by the sector stress test on the TPC-H distribution benchmark (October 2026): an order line reaches a nation
through its customer and through its supplier. "The 5 best French customers in 1997" and "revenue by nation in
AMERICA" were counted by the supplier's nation, the shorter route, never said: every figure was another nation's.
A question that names one of the routes ("customers", "supplier nation") takes it; one that names neither takes
the shorter and says so ("Nation is the supplier's; ask "by customer nation" for the customer's"); the catalog
lists the routes for the planner to name.

Invented rows only.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.compile.compiler import compile_query
from core2.plan.catalog import catalog_text
from core2.plan.ir import Plan
from core2.plan.planner import route_named
from core2.resolve.resolver import Context, resolve
from core2.warehouse.runner import DuckDBWarehouse

NATIONS = ["FRANCE", "GERMANY", "BRAZIL", "CANADA", "JAPAN", "KENYA"]


@pytest.fixture(scope="module")
def shop():
    con = duckdb.connect()
    con.execute(f"""
        CREATE TABLE nation AS SELECT range AS n_nationkey, {NATIONS}[range + 1] AS n_name FROM range(6);
        CREATE TABLE customer AS SELECT range + 1 AS c_custkey, 'Customer#' || lpad(range::VARCHAR, 6, '0') AS c_name,
               range % 6 AS c_nationkey FROM range(600);
        CREATE TABLE supplier AS SELECT range + 1 AS s_suppkey, 'Supplier#' || lpad(range::VARCHAR, 4, '0') AS s_name,
               (range * 5 + 1) % 6 AS s_nationkey FROM range(60);
        CREATE TABLE orders AS SELECT range + 1 AS o_orderkey, (range * 7) % 600 + 1 AS o_custkey,
               DATE '1996-01-01' + (range % 700)::INTEGER AS o_orderdate FROM range(6000);
        CREATE TABLE lineitem AS SELECT range // 3 + 1 AS l_orderkey, range % 3 + 1 AS l_linenumber,
               (range * 11) % 60 + 1 AS l_suppkey, round(100 + (range * 37) % 5000 + 0.5, 2) AS l_extendedprice
        FROM range(18000);
    """)
    warehouse = DuckDBWarehouse(con)
    return con, build_model(warehouse, from_duckdb(warehouse), client_id="t-route", options=BuildOptions(workers=1))


def _slugs(model):
    price = next(m.slug for m in model.measures.values() if getattr(m.expr, "column", "") and
                 m.expr.column.endswith(".l_extendedprice"))
    nation = next(a.slug for a in model.attributes.values() if model.columns[a.column].name == "n_name") \
        if any(model.columns[a.column].name == "n_name" for a in model.attributes.values()) else \
        next(e.slug for e in model.entities.values() if e.table.endswith(".nation"))
    return price, nation


def _run(shop, plan: Plan):
    con, model = shop
    logical = resolve(plan, model, Context(today=dt.date(1998, 1, 1)))
    return logical, {r[0]: float(r[1]) for r in con.execute(compile_query(logical, model, "duckdb").sql).fetchall()}


def test_the_catalog_lists_both_routes(shop):
    _, model = shop
    assert "through (name one in via): customer, supplier" in catalog_text(model)


def test_customers_of_a_nation_are_reached_through_the_customer(shop):
    con, model = shop
    price, nation = _slugs(model)
    plan = Plan(kind="query", measures=[price], group_by=[nation])
    routed = route_named(plan, model, "revenue of our customers by nation")
    assert routed.via == {nation: "customer"}
    _, got = _run(shop, routed)
    want = dict(con.execute("""SELECT n.n_name, SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey
                               JOIN customer c ON o.o_custkey = c.c_custkey JOIN nation n ON c.c_nationkey = n.n_nationkey
                               GROUP BY 1""").fetchall())
    assert got == pytest.approx({k: float(v) for k, v in want.items()})


def test_a_nation_the_question_names_no_route_for_is_the_shorter_and_said(shop):
    _, model = shop
    price, nation = _slugs(model)
    plan = Plan(kind="query", measures=[price], group_by=[nation])
    assert route_named(plan, model, "revenue by nation") is plan
    logical, _ = _run(shop, plan)
    assert any("is the supplier's; ask \"by customer" in n for n in logical.notes), logical.notes


def test_a_supplier_nation_is_reached_through_the_supplier(shop):
    _, model = shop
    price, nation = _slugs(model)
    plan = Plan(kind="query", measures=[price], group_by=[nation])
    assert route_named(plan, model, "revenue by supplier nation").via == {nation: "supplier"}
