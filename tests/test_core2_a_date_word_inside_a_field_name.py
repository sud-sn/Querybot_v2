"""A date's word inside the name of a field the question groups by is no date it asked for.

Asked of the AdventureWorks sample with a real model (gpt-4o), "Order subtotal by ship method in 2024" was dated
by the ship date: "ship" is the method's word. The measure's own date is taken; "by ship date" or "shipped in"
still picks the ship date. Synthetic data only.
"""

from __future__ import annotations

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.ir import Plan
from core2.plan.planner import date_in_a_name
from core2.warehouse.runner import DuckDBWarehouse


@pytest.fixture(scope="module")
def model():
    con = duckdb.connect()
    con.execute("CREATE TABLE ShipMethod (ShipMethodID INTEGER, Name VARCHAR)")
    con.execute("INSERT INTO ShipMethod VALUES (1, 'Ground truck'), (2, 'Air cargo'), (3, 'Overnight')")
    con.execute("CREATE TABLE OrderHeader (OrderID INTEGER, ShipMethodID INTEGER, OrderDate DATE, ShipDate DATE, "
                "Subtotal DECIMAL(12,2))")
    con.execute("INSERT INTO OrderHeader SELECT i, i % 3 + 1, DATE '2024-01-01' + CAST(i % 700 AS INTEGER), "
                "DATE '2024-01-01' + CAST(i % 700 + 3 + i % 9 AS INTEGER), 20 + i % 400 FROM range(1, 3001) t(i)")
    warehouse = DuckDBWarehouse(con)
    return build_model(warehouse, from_duckdb(warehouse), client_id="ship", options=BuildOptions(workers=1))


def _plan(model, group_by: list[str]) -> Plan:
    ship = next(r for r in model.date_roles.values() if r.column.endswith(".shipdate"))
    subtotal = next(m for m in model.measures.values() if m.expr.column and m.expr.column.endswith(".subtotal"))
    return Plan.model_validate({"kind": "query", "intent": "breakdown", "measures": [subtotal.slug],
                                "group_by": group_by, "time": {"date": ship.slug, "window": {
                                    "kind": "between", "start": "2024-01-01", "end": "2024-12-31"}}})


def _method(model) -> str:
    return next(a.slug for a in model.attributes.values() if a.column.endswith("shipmethod.name"))


@pytest.mark.parametrize("question, kept", [
    ("Order subtotal by ship method in 2024", False),
    ("Order subtotal by ship method, by ship date, in 2024", True),
    ("Order subtotal shipped in 2024 by ship method", True),
    ("Order subtotal by ship date in 2024", True),
])
def test_a_date_word_inside_a_grouping_name_is_not_a_date(model, question, kept):
    plan = _plan(model, [_method(model)])
    assert (date_in_a_name(plan, model, question).time.date == plan.time.date) is kept


def test_a_date_with_no_grouping_of_that_name_stands(model):
    plan = _plan(model, [])
    assert date_in_a_name(plan, model, "Order subtotal by ship method in 2024") is plan
