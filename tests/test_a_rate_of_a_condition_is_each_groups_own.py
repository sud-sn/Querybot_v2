"""A rate of a condition -- the cancellation rate by region -- is each group's rows that meet it over all of
that group's rows.

"Cancellation rate by region" and "reversal rate by division" had no plan: the planner fell back on an on-time
share from the order date to a return date (100% everywhere) or gave every division 50%. A rate names the figure
counted and the condition on its rows; in each group it is the figure where the condition holds over the whole
figure, and rows a table leaves out by default (cancelled ones) are brought back when the condition is about them.

The synthetic retail warehouse (invented data); figures are checked against hand-written SQL.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.ir import Plan
from core2.plan.planner import settle
from core2.plan.values import MemberIndex
from core2.resolve.resolver import Context, ResolveError, resolve
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.framework import materialize

Y25 = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}
RATE = {"name": "Cancellation rate", "measure": "number_of_orders",
        "where": [{"field": "order_line.status_code", "op": "eq", "values": ["C"]}]}


@pytest.fixture(scope="module")
def retail():
    built = materialize(domains.build("retail"), "descriptive")
    warehouse = DuckDBWarehouse(built.con)
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=built.declared_fks), client_id="t-rate",
                        options=BuildOptions(workers=1))
    return built.con, model


def test_the_cancellation_rate_of_each_region(retail):
    con, model = retail
    plan = {"intent": "rank", "rates": [RATE], "group_by": ["region"], "time": Y25,
            "sort": [{"by": "Cancellation rate", "desc": True}]}
    services = Services(model=model, warehouse=DuckDBWarehouse(con),
                        complete=lambda s, t: json.dumps({"kind": "query", **plan}), index=MemberIndex(),
                        today=dt.date(2026, 6, 14))
    payload = answer_question("cancellation rate by region in 2025", services, Session())
    want = con.execute(
        "SELECT r.region_name, 100.0 * COUNT(DISTINCT CASE WHEN o.status_code = 'C' THEN o.order_number END) "
        "/ COUNT(DISTINCT o.order_number) v FROM order_lines o JOIN calendar c ON c.date_key = o.order_date_key "
        "JOIN stores s ON s.store_id = o.store_id JOIN regions r ON r.region_id = s.region_id "
        "WHERE year(c.full_date) = 2025 GROUP BY 1 ORDER BY v DESC").fetchall()
    got = [(r["region_name"], r["cancellation_rate"]) for r in payload["data"]["rows"]]
    assert [g[0] for g in got] == [w[0] for w in want]
    assert all(abs(g[1] - float(w[1])) < 0.001 for g, w in zip(got, want))


def test_rows_left_out_by_default_are_brought_back_for_their_rate(retail):
    _, model = retail
    from core2.model.schema import ColumnFilter

    leaves_out = model.model_copy(deep=True)
    lines = next(t for t in leaves_out.tables.values() if t.key.endswith(".order_lines"))
    status = next(c.key for c in leaves_out.columns.values() if c.table == lines.key and c.name == "status_code")
    lines.default_filters = [ColumnFilter(column=status, op="ne", values=["C"])]
    plan = Plan.model_validate({"kind": "query", "intent": "breakdown", "rates": [RATE], "group_by": ["region"],
                                "time": Y25})
    assert settle(plan, leaves_out, "cancellation rate by region in 2025", None).include_left_out


def test_the_condition_is_on_the_figures_own_rows(retail):
    _, model = retail
    plan = Plan.model_validate({"kind": "query", "intent": "breakdown", "group_by": ["region"], "time": Y25,
                                "rates": [{"name": "Rate", "measure": "number_of_orders",
                                           "where": [{"field": "region", "op": "eq", "values": ["East"]}]}]})
    with pytest.raises(ResolveError) as raised:
        resolve(plan, model, Context(today=dt.date(2026, 6, 14)))
    assert "own rows" in raised.value.message
