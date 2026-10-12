"""The biggest orders, journals or movements are ranked by their own number.

A document number counted to count the documents ("Number of orders" counts distinct order numbers) was a
measure's column and nothing else: "the 3 biggest orders of 2025" could not be grouped by order, and was
answered as the top 3 order lines or refused. The number is now also a grouping that tells the documents
apart (never one whose members are read for the AI).

The synthetic retail warehouse (invented data); figures are checked against hand-written SQL.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.values import MemberIndex, listable
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.framework import materialize


@pytest.fixture(scope="module")
def retail():
    built = materialize(domains.build("retail"), "descriptive")
    warehouse = DuckDBWarehouse(built.con)
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=built.declared_fks), client_id="t-docs",
                        options=BuildOptions(workers=1))
    return built.con, model


def test_the_order_number_is_a_grouping_and_still_counts_orders(retail):
    _, model = retail
    assert model.attributes["order_line.order_number"].kind == "identifier"
    assert "order_line.order_number" not in listable(model)
    assert any(m.slug == "number_of_orders" for m in model.measures.values())


def test_the_three_biggest_orders(retail):
    con, model = retail
    plan = {"intent": "rank", "measures": ["net_amount"], "group_by": ["order_line.order_number"], "limit": 3,
            "sort": [{"by": "net_amount", "desc": True}],
            "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}}
    services = Services(model=model, warehouse=DuckDBWarehouse(con),
                        complete=lambda s, t: json.dumps({"kind": "query", **plan}), index=MemberIndex(),
                        today=dt.date(2026, 6, 14))
    payload = answer_question("the 3 biggest orders of 2025", services, Session())
    want = con.execute("SELECT o.order_number, SUM(o.net_amount) n FROM order_lines o JOIN calendar c "
                       "ON c.date_key = o.order_date_key WHERE year(c.full_date) = 2025 GROUP BY 1 "
                       "ORDER BY n DESC LIMIT 3").fetchall()
    got = [(r["order_line_order_number"], r["net_amount"]) for r in payload["data"]["rows"]]
    assert [g[0] for g in got] == [w[0] for w in want]
    assert all(abs(g[1] - float(w[1])) < 0.01 for g, w in zip(got, want))
