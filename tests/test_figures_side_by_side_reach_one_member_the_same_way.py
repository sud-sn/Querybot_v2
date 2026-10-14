"""Figures from several tables meet at one member the same way, and a route is never picked on a guess.

A data model's relationships make a dimension shared: sales and refunds "by region" are both by the same region.
The new core gave each table its own best route to the grouping and lined the totals up by value, so sales could
be by their store's region and refunds by their customer's, side by side as if they were one. And when two routes
of the same length and trust reached a member (a shipment's region through its store, or through its customer),
the one with the smaller join key won, said only in a note. Now:

* tables of one question that part ways are sent along the first table's route when they can follow it (refunds
  through their order reach the order's store's region); when they cannot, the answer says so with its figures;
* of two equally good routes of several steps, one is taken, always the same, as a governed model's active
  relationship: the answer says which and how to ask for the other ("the customer's region"), and an admin makes
  the other the default by naming the role of the first one's link;
* a link no one measured, that the data would not check at question time, is not followed (it might count rows
  more than once): the reader is told why, and it is checked again next time.

An invented warehouse; every number is compared with SQL written by hand.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 6, 30)
H1 = {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}}


def _fk(parent: str, column: str, ref: str, ref_column: str) -> dict:
    return {"constraint_name": f"fk_{parent}_{column}", "parent_schema": "main", "parent_table": parent,
            "parent_col": column, "ref_schema": "main", "ref_table": ref, "ref_col": ref_column, "ordinal": 1}


DECLARED = [_fk("stores", "region_id", "regions", "region_id"),
            _fk("customers", "region_id", "regions", "region_id"),
            _fk("orders", "store_id", "stores", "store_id"),
            _fk("returns", "order_id", "orders", "order_id"),
            _fk("returns", "customer_id", "customers", "customer_id"),
            _fk("complaints", "customer_id", "customers", "customer_id"),
            _fk("shipments", "store_id", "stores", "store_id"),
            _fk("shipments", "customer_id", "customers", "customer_id")]


def _warehouse() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("CREATE TABLE regions (region_id INTEGER PRIMARY KEY, region_name VARCHAR)")
    con.execute("INSERT INTO regions VALUES (1, 'North'), (2, 'South'), (3, 'West')")
    con.execute("CREATE TABLE stores (store_id INTEGER PRIMARY KEY, store_name VARCHAR, region_id INTEGER)")
    con.execute("INSERT INTO stores SELECT i, 'Store ' || i, 1 + i % 3 FROM range(1, 10) t(i)")
    con.execute("CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, customer_name VARCHAR, region_id INTEGER)")
    # A customer's region is not their stores': the two routes give different answers.
    con.execute("INSERT INTO customers SELECT i, 'Customer ' || i, 1 + (i * 2) % 3 FROM range(1, 31) t(i)")
    con.execute("CREATE TABLE orders (order_id INTEGER PRIMARY KEY, order_date DATE, store_id INTEGER, "
                "order_amount DECIMAL(12, 2))")
    con.execute("INSERT INTO orders SELECT i, DATE '2026-01-01' + CAST(i % 170 AS INTEGER), 1 + i % 9, "
                "round(30 + (i * 37) % 300, 2) FROM range(1, 401) t(i)")
    con.execute("CREATE TABLE returns (return_id INTEGER PRIMARY KEY, return_date DATE, order_id INTEGER, "
                "customer_id INTEGER, refund_amount DECIMAL(12, 2))")
    con.execute("INSERT INTO returns SELECT i, DATE '2026-01-05' + CAST(i % 160 AS INTEGER), 1 + (i * 7) % 400, "
                "1 + i % 30, round(5 + (i * 13) % 60, 2) FROM range(1, 121) t(i)")
    con.execute("CREATE TABLE complaints (complaint_id INTEGER PRIMARY KEY, complaint_date DATE, "
                "customer_id INTEGER, complaint_cost DECIMAL(12, 2))")
    con.execute("INSERT INTO complaints SELECT i, DATE '2026-01-03' + CAST(i % 150 AS INTEGER), 1 + i % 30, "
                "round(10 + (i * 11) % 40, 2) FROM range(1, 81) t(i)")
    con.execute("CREATE TABLE shipments (shipment_id INTEGER PRIMARY KEY, ship_date DATE, store_id INTEGER, "
                "customer_id INTEGER, ship_weight DECIMAL(12, 2))")
    con.execute("INSERT INTO shipments SELECT i, DATE '2026-01-02' + CAST(i % 165 AS INTEGER), 1 + (i * 5) % 9, "
                "1 + (i * 7) % 30, round(1 + (i * 17) % 25, 2) FROM range(1, 201) t(i)")
    return con


@pytest.fixture(scope="module")
def shop():
    con = _warehouse()
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=DECLARED), client_id="routes",
                        options=BuildOptions(workers=1))
    return con, model


def _measure(model, column: str) -> str:
    return next(m.slug for m in model.measures.values() if not m.hidden
                and m.expr.model_dump().get("agg") == "sum" and model.columns.get(m.expr.model_dump().get("column")
                                                                                   or "") is not None
                and model.columns[m.expr.model_dump()["column"]].name == column)


def _region(model) -> str:
    return next(a.slug for a in model.attributes.values() if a.column in model.columns
                and model.tables[model.columns[a.column].table].name == "regions"
                and model.columns[a.column].name == "region_name")


class Planner:
    def __init__(self, *plans):
        self.plans = [json.dumps({"kind": "query", **p}) for p in plans]

    def __call__(self, stable, tail):
        return self.plans.pop(0)


def _ask(con, model, session, question: str, *plans):
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=Planner(*plans),
                        index=MemberIndex(), today=TODAY)
    return answer_question(question, services, session)


def _by(payload, measure_slug: str) -> dict[str, float]:
    data = payload.get("data") or {}
    headers = data.get("headers") or []
    name = next(h for h in headers if h.startswith(measure_slug) or measure_slug.startswith(h))
    return {str(r[headers[0]]): float(r[name] or 0) for r in data.get("rows") or []}


def _truth(con, sql: str) -> dict[str, float]:
    return {str(k): float(v) for k, v in con.execute(sql).fetchall()}


# ── tables that part ways are sent along one route ───────────────────────────


def test_refunds_by_region_follow_their_orders_store_like_the_sales_beside_them(shop):
    con, model = shop
    sales, refunds, region = _measure(model, "order_amount"), _measure(model, "refund_amount"), _region(model)
    payload = _ask(con, model, Session(), "Sales and refunds by region",
                   {"intent": "breakdown", "measures": [sales, refunds], "group_by": [region], "time": H1})
    assert payload.get("data"), payload["answer"]
    want = _truth(con, """SELECT g.region_name, SUM(r.refund_amount) FROM returns r JOIN orders o USING (order_id)
                           JOIN stores s USING (store_id) JOIN regions g ON g.region_id = s.region_id
                           WHERE r.return_date BETWEEN '2026-01-01' AND '2026-06-30' GROUP BY 1""")
    assert _by(payload, refunds) == pytest.approx(want), "the order's store's region, as the sales beside them"
    assert not payload.get("coverage_caveats") or "not reached the same way" not in " ".join(
        payload["coverage_caveats"])


def test_figures_that_cannot_meet_say_each_is_by_its_own_member(shop):
    con, model = shop
    sales, cost, region = _measure(model, "order_amount"), _measure(model, "complaint_cost"), _region(model)
    payload = _ask(con, model, Session(), "Sales and complaint costs by region",
                   {"intent": "breakdown", "measures": [sales, cost], "group_by": [region], "time": H1})
    assert payload.get("data"), payload["answer"]
    said = " ".join(payload.get("coverage_caveats") or [])
    assert "not reached the same way by every figure" in said and "store's" in said and "customer's" in said
    want = _truth(con, """SELECT g.region_name, SUM(c.complaint_cost) FROM complaints c JOIN customers u
                           USING (customer_id) JOIN regions g ON g.region_id = u.region_id
                           WHERE c.complaint_date BETWEEN '2026-01-01' AND '2026-06-30' GROUP BY 1""")
    assert _by(payload, cost) == pytest.approx(want)


def test_a_filter_on_the_shared_member_takes_the_same_route(shop):
    con, model = shop
    sales, refunds, region = _measure(model, "order_amount"), _measure(model, "refund_amount"), _region(model)
    payload = _ask(con, model, Session(), 'Sales and refunds in "North"',
                   {"intent": "value", "measures": [sales, refunds], "time": H1,
                    "filters": [{"field": region, "op": "eq", "values": ["North"]}]})
    (row,) = payload["data"]["rows"]
    (want,) = con.execute("""SELECT SUM(r.refund_amount) FROM returns r JOIN orders o USING (order_id)
                              JOIN stores s USING (store_id) JOIN regions g ON g.region_id = s.region_id
                              WHERE g.region_name = 'North'
                              AND r.return_date BETWEEN '2026-01-01' AND '2026-06-30'""").fetchone()
    assert float(row[refunds]) == pytest.approx(float(want)), (payload["data"], payload["trust"]["sql"])


# ── two equally good routes: one is taken, and the answer says so ─────────


def _customer(model) -> str:
    return next(e.slug for e in model.entities.values() if model.tables[e.table].name == "customers" and not e.role)


def _shipped_by_region(through: str) -> str:
    return f"""SELECT g.region_name, SUM(s.ship_weight) FROM shipments s JOIN {through}s t USING ({through}_id)
               JOIN regions g ON g.region_id = t.region_id
               WHERE s.ship_date BETWEEN '2026-01-01' AND '2026-06-30' GROUP BY 1"""


def test_two_equally_good_routes_take_one_and_say_how_to_ask_for_the_other(shop):
    con, model = shop
    weight, region = _measure(model, "ship_weight"), _region(model)
    plan = {"intent": "breakdown", "measures": [weight], "group_by": [region], "time": H1}
    first = _ask(con, model, Session(), "Shipped weight by region", plan)
    again = _ask(con, model, Session(), "Shipped weight by region", plan)
    assert first.get("data") and not first.get("clarify"), first["answer"]
    assert _by(first, weight) == _by(again, weight), "always the same route"
    said = " ".join(first.get("coverage_caveats") or [])
    taken = "store" if "here is the store's" in said else "customer"
    other = "customer" if taken == "store" else "store"
    assert f"ask for the {other}'s region" in said.lower(), said
    assert _by(first, weight) == pytest.approx(_truth(con, _shipped_by_region(taken)))
    # The other route, asked for by whose region it is: no caution then, as no other route was as good.
    chosen = _ask(con, model, Session(), "Shipped weight by the customer's region",
                  {**plan, "via": {region: _customer(model)}})
    assert _by(chosen, weight) == pytest.approx(_truth(con, _shipped_by_region("customer")))
    assert "can differ" not in " ".join(chosen.get("coverage_caveats") or [])


def test_a_route_whose_link_has_a_role_is_never_the_default(shop):
    con, model = shop
    from core2.model.schema import SemanticModel
    from core2.resolve import paths as P

    named = SemanticModel.model_validate(model.model_dump())
    shipments = next(k for k, t in named.tables.items() if t.name == "shipments")
    customers = next(k for k, t in named.tables.items() if t.name == "customers")
    goal = next(k for k, t in named.tables.items() if t.name == "regions")
    link = next(j for j in named.joins.values() if j.from_table == shipments and j.to_table == customers)
    named.joins[link.key] = link.model_copy(update={"role": "Receiving customer"})
    path = P.best_path(named, shipments, goal)
    assert path is not None and customers not in path.tables and not path.others, "the store's region, said once"


def test_one_route_better_than_the_others_is_taken_without_asking(shop):
    con, model = shop
    sales, region = _measure(model, "order_amount"), _region(model)
    payload = _ask(con, model, Session(), "Sales by region",
                   {"intent": "breakdown", "measures": [sales], "group_by": [region], "time": H1})
    assert payload.get("data") and not payload.get("clarify")


# ── a link the data will not check is not followed ───────────────────────────


def test_a_link_the_data_will_not_check_is_not_followed_and_says_why(shop, monkeypatch):
    con, model = shop
    unmeasured = model.model_copy(deep=True)
    unmeasured.version = 4242
    link = next(j for j in unmeasured.joins.values() if unmeasured.tables[j.from_table].name == "orders"
                and unmeasured.tables[j.to_table].name == "stores")
    link.target_checked, link.provenance = False, "declared"     # declared by the database, never measured

    class Refusing(DuckDBWarehouse):
        def query(self, sql, *, max_rows=None):
            if "HAVING" in sql.upper():
                raise RuntimeError("permission denied")
            return super().query(sql, max_rows=max_rows)

    sales, store = _measure(unmeasured, "order_amount"), next(
        a.slug for a in unmeasured.attributes.values() if a.column in unmeasured.columns
        and unmeasured.tables[unmeasured.columns[a.column].table].name == "stores"
        and unmeasured.columns[a.column].name == "store_name")
    services = Services(model=unmeasured, warehouse=Refusing(con),
                        complete=Planner({"intent": "breakdown", "measures": [sales], "group_by": [store],
                                          "time": H1}), index=MemberIndex(), today=TODAY)
    payload = answer_question("Sales by store", services, Session())
    assert payload.get("data") is None
    assert "could not be checked against the data" in payload["answer"]["headline"], payload["answer"]
