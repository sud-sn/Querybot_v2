"""A rate the planner names as a slug ("cancellation_rate") is said in words in the answer.

Found by the sector stress test (October 2026): "Claim rejection rate by payer in 2025" was answered as
"rejection_rate in 2025: Northstar PBM leads with 14.5%", the chart titled "rejection_rate by payer" and the
follow-up chip "Monthly rejection_rate". A figure worked out for the question is named for readers ("Rejection
rate"); a sort or filter written with the slug still finds it.

Invented rows only.
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

Y25 = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}


@pytest.fixture(scope="module")
def retail():
    built = materialize(domains.build("retail"), "descriptive")
    warehouse = DuckDBWarehouse(built.con)
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=built.declared_fks), client_id="t-named",
                        options=BuildOptions(workers=1))
    return built.con, model


def test_a_rate_named_as_a_slug_is_said_in_words(retail):
    con, model = retail
    plan = {"kind": "query", "intent": "rank", "group_by": ["region"], "time": Y25,
            "rates": [{"name": "cancellation_rate", "measure": "number_of_orders",
                       "where": [{"field": "order_line.status_code", "op": "eq", "values": ["C"]}]}],
            "sort": [{"by": "cancellation_rate", "desc": True}]}
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, t: json.dumps(plan),
                        index=MemberIndex(), today=dt.date(2026, 6, 14))
    payload = answer_question("cancellation rate by region in 2025", services, Session())
    assert (payload.get("data") or {}).get("rows"), payload
    chart = payload.get("chart") or {}
    said = [payload["answer"]["headline"], chart.get("title", ""),
            *(r.get("label", "") for r in (chart.get("column_roles") or {}).values())]
    assert not any("cancellation_rate" in s for s in said), said
    assert payload["answer"]["headline"].startswith("Cancellation rate") and chart["title"].startswith("Cancellation rate")
    rates = [r[next(k for k in r if "rate" in k.lower())] for r in payload["data"]["rows"]]
    assert rates == sorted(rates, reverse=True)       # the sort written with the slug still finds the figure
