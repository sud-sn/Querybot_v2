"""A figure "per order" or "per booking" is each document's figure, then their average.

"What was the average service revenue per booking?" was planned with each booking as the thing the figure is
worked out for, named by its word ("booking"), and refused twice: only lists (customers, technicians) and count
measures were accepted, so the reader got "I could not work out how to answer that". A word that names what
a count measure counts ("Number of orders" counts orders) is that count's thing.

The synthetic retail warehouse (invented data); figures are checked against hand-written SQL.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.framework import materialize


@pytest.fixture(scope="module")
def retail():
    built = materialize(domains.build("retail"), "descriptive")
    warehouse = DuckDBWarehouse(built.con)
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=built.declared_fks), client_id="t-per-doc",
                        options=BuildOptions(workers=1))
    return built.con, model


@pytest.mark.parametrize("word", ["order", "orders", "Order"])
def test_the_average_per_order_is_the_average_of_each_orders_total(retail, word):
    con, model = retail
    plan = {"intent": "value", "measures": ["net_amount"], "per": {"of": word, "summary": "avg"},
            "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}}
    services = Services(model=model, warehouse=DuckDBWarehouse(con),
                        complete=lambda s, t: json.dumps({"kind": "query", **plan}), index=MemberIndex(),
                        today=dt.date(2026, 6, 14))
    payload = answer_question("average net sales per order in 2025", services, Session())
    rows = (payload.get("data") or {}).get("rows")
    assert rows, payload["answer"]["headline"]
    want = con.execute("SELECT AVG(t) FROM (SELECT o.order_number, SUM(o.net_amount) t FROM order_lines o "
                       "JOIN calendar c ON c.date_key = o.order_date_key WHERE year(c.full_date) = 2025 "
                       "GROUP BY 1)").fetchone()[0]
    got = next(v for v in rows[0].values() if isinstance(v, (int, float)))
    assert abs(got - float(want)) < 0.01


def test_a_question_that_cannot_be_planned_offers_the_nearest_ones(retail):
    """Two plans that do not check out ("received in full": a filter on a figure no field holds) end in the
    figures the planner tried, for the period it read, one click each -- never a dead end."""
    con, model = retail
    plan = {"intent": "value", "measures": ["net_amount"], "group_by": ["store.name"],
            "filters": [{"field": "net_amount_in_full", "op": "eq", "values": [1]}],
            "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}}
    services = Services(model=model, warehouse=DuckDBWarehouse(con),
                        complete=lambda s, t: json.dumps({"kind": "query", **plan}), index=MemberIndex(),
                        today=dt.date(2026, 6, 14))
    payload = answer_question("net sales by store received in full in 2025", services, Session())
    options = payload["clarify"]["options"]
    assert options and options[0].endswith("by store in 2025") and options[1].endswith("in 2025")
    assert [c["question"] for c in payload["follow_up_suggestions"]] == options
    assert payload["trust"]["stopped"]                  # why, for the reader who opens "how this was worked out"


def test_a_vague_question_offers_the_main_figures_for_its_year(retail):
    from core2.plan.planner import main_figures

    offered = main_figures(retail[1], "show me the 2025 numbers")
    assert offered and all(o.endswith(" in 2025") for o in offered)
    assert any(o.startswith("Net amount") for o in offered)
