"""A table reached two ways is never guessed: the question says which role, or is asked.

Orders name two customers, the one billed and the one shipped to. "Sales by
customer" could mean either, so the new core asks, offering the roles by name;
the reply names one, and the answer follows that link and no other. A role-less
link (the line's own store) needs no choice, and a named role elsewhere ("the
customer's home store") is followed when the question names it.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import numpy as np
import pandas as pd

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.catalog import catalog_text
from core2.plan.ir import Plan
from core2.plan.values import MemberIndex
from core2.resolve.resolver import Context, ResolveError, resolve
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import _Words, golden, learn, same_rows
from evals.core2.framework import materialize

TODAY = dt.date(2026, 6, 15)


def _learn(tables: dict[str, pd.DataFrame]) -> tuple[DuckDBWarehouse, object]:
    con = duckdb.connect()
    for name, frame in tables.items():
        con.register("_f", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _f")
        con.unregister("_f")
    warehouse = DuckDBWarehouse(con)
    return warehouse, build_model(warehouse, from_duckdb(warehouse), client_id="t", options=BuildOptions(workers=1))


def _orders() -> tuple[DuckDBWarehouse, object]:
    rng = np.random.default_rng(3)
    customers = pd.DataFrame({"customer_id": np.arange(1, 41),
                              "customer_name": [f"Client {i:02d}" for i in range(1, 41)]})
    n = 3000
    bill = rng.integers(1, 41, n)
    ship = np.where(rng.random(n) < 0.6, bill, rng.integers(1, 41, n))
    orders = pd.DataFrame({"order_id": np.arange(1, n + 1), "bill_to_customer_id": bill,
                           "ship_to_customer_id": ship, "order_amount": np.round(rng.uniform(10, 900, n), 2)})
    return _learn({"customers": customers, "orders": orders})


class Recorded:
    def __init__(self, *answers: str):
        self.answers, self.sent = list(answers), []

    def __call__(self, stable: str, tail: str) -> str:
        self.sent.append(tail)
        return self.answers.pop(0)


def test_two_roles_are_asked_about_by_name_and_the_reply_is_followed():
    warehouse, model = _orders()
    roles = sorted(j.role for j in model.joins.values() if j.to_table.endswith("customers") and j.role)
    assert [r.lower() for r in roles] == ["bill to customer", "ship to customer"], roles
    measure = next(m.slug for m in model.measures.values() if (getattr(m.expr, "column", None) or "").endswith(
        "order_amount"))
    customer = next(e.slug for e in model.entities.values() if e.table.endswith("customers"))
    assert f"roles (name one in via): {', '.join(roles)}" in catalog_text(model)    # what the AI is shown
    plain = {"kind": "query", "intent": "breakdown", "measures": [measure], "group_by": [customer]}
    with_role = {**plain, "via": {customer: roles[1]}, "follow_up": "refine"}
    ai = Recorded(json.dumps(plain), json.dumps(with_role))
    services = Services(model=model, warehouse=warehouse, complete=ai, index=MemberIndex(), today=TODAY)
    session = Session()
    learned = len(warehouse.log)          # the bootstrap's own reads
    asked = answer_question("sales by customer", services, session)
    assert sorted(asked["clarify"]["options"]) == roles and len(warehouse.log) == learned   # asked, nothing run
    answered = answer_question(roles[1], services, session)
    sql = warehouse.log[-1]
    assert "PREVIOUS QUESTION: sales by customer" in ai.sent[-1]
    got = warehouse.query(sql)
    expected = warehouse.query("SELECT c.customer_name, SUM(o.order_amount) FROM orders o "
                               "LEFT JOIN customers c ON c.customer_id = o.ship_to_customer_id GROUP BY 1")
    assert not same_rows(["name", "amount"], expected.rows, got.columns, got.rows, order_matters=False)
    assert answered["data"]["total_rows"] == 40

    # The role as a reader or the AI may write it: other spelling, or without its noun.
    from core2.compile.compiler import compile_query

    def sql_for(role: str) -> str:
        logical = resolve(Plan.model_validate({**plain, "via": {customer: role}}), model, Context(today=TODAY))
        return compile_query(logical, model, "duckdb").sql

    assert sql_for("ship-to") == sql_for("SHIP TO CUSTOMER") == sql
    assert sql_for("Bill to") != sql_for("Ship to")


def test_a_role_the_question_names_is_followed_where_a_plain_link_also_exists():
    domain = domains.build("retail")
    built, model = learn(domain, "descriptive")
    case = next(c for c in golden("retail")["questions"] if c["id"] == "retail-012")
    plan = _Words(model, built).plan({**case["plan"], "via": {}})
    region = plan.group_by[0]
    logical = resolve(Plan.model_validate({**plan.model_dump(), "via": {region: "Home store"}}), model,
                      Context(today=TODAY))
    from core2.compile.compiler import compile_query

    got = DuckDBWarehouse(built.con).query(compile_query(logical, model, "duckdb").sql)
    expected = DuckDBWarehouse(materialize(domain, "descriptive").con).query(case["reference_sql"])
    assert not same_rows(expected.columns, expected.rows, got.columns, got.rows, order_matters=False)
    try:
        resolve(Plan.model_validate({**plan.model_dump(), "via": {region: "Head office"}}), model,
                Context(today=TODAY))
    except ResolveError as exc:
        assert exc.kind == "unknown" and "Home store" in exc.options
    else:
        raise AssertionError("a role no link has was followed")


def test_a_shortened_role_two_roles_could_mean_is_not_guessed():
    rng = np.random.default_rng(5)
    locations = pd.DataFrame({"location_id": np.arange(1, 31), "location_name": [f"Site {i:02d}" for i in range(1, 31)]})
    n = 2000
    shipments = pd.DataFrame({"shipment_id": np.arange(1, n + 1), "ship_from_location_id": rng.integers(1, 31, n),
                              "ship_to_location_id": rng.integers(1, 31, n),
                              "shipped_weight": np.round(rng.uniform(1, 500, n), 1)})
    _, model = _learn({"locations": locations, "shipments": shipments})
    measure = next(m.slug for m in model.measures.values() if (getattr(m.expr, "column", None) or "").endswith(
        "shipped_weight"))
    location = next(e.slug for e in model.entities.values() if e.table.endswith("locations"))
    plan = {"kind": "query", "intent": "breakdown", "measures": [measure], "group_by": [location]}

    def joined_on(role: str) -> str:
        logical = resolve(Plan.model_validate({**plan, "via": {location: role}}), model, Context(today=TODAY))
        from core2.compile.compiler import compile_query

        sql = compile_query(logical, model, "duckdb").sql
        return "ship_to_location_id" if "ship_to_location_id" in sql else "ship_from_location_id"

    assert joined_on("ship to") == "ship_to_location_id" and joined_on("Ship-from") == "ship_from_location_id"
    try:
        joined_on("ship")
    except ResolveError as exc:
        assert exc.kind == "unknown" and len(exc.options) == 2, exc.options
    else:
        raise AssertionError("'ship' was taken for one of two roles")
