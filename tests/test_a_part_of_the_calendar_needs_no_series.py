"""Grouped by a part of the calendar, an answer is no series of the periods that part is made of.

"Busiest day of the week in 2025 by number of orders" was planned as a daily series and the weekday, so
every date was counted apart under its weekday, and "Thursday 20 Nov 2025, 24 orders" led instead of
Tuesday with 626. A grouping by a part of the calendar that each period of the asked grain falls in once
(a weekday for days, a month of the year for days, weeks or months, a quarter of the year for months and
quarters) replaces the series: the counts are of each part. A series by month beside the weekday ("monthly
orders by weekday") is still two groupings.

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

Y25 = {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}


@pytest.fixture(scope="module")
def retail():
    built = materialize(domains.build("retail"), "descriptive")
    warehouse = DuckDBWarehouse(built.con)
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=built.declared_fks), client_id="t-dow",
                        options=BuildOptions(workers=1))
    return built.con, model


def _rows(con, model, plan):
    services = Services(model=model, warehouse=DuckDBWarehouse(con),
                        complete=lambda s, t: json.dumps({"kind": "query", **plan}), index=MemberIndex(),
                        today=dt.date(2026, 6, 14))
    payload = answer_question("q", services, Session())
    rows = (payload.get("data") or {}).get("rows")
    assert rows, payload["answer"]["headline"]
    return rows


def test_the_busiest_weekday_counts_each_weekday_not_each_date(retail):
    con, model = retail
    rows = _rows(con, model, {"intent": "rank", "measures": ["number_of_orders"], "group_by": ["time:day_of_week"],
                              "time": {"grain": "day", "window": Y25},
                              "sort": [{"by": "number_of_orders", "desc": True}], "limit": 1})
    want = con.execute("SELECT dayname(c.full_date), COUNT(DISTINCT o.order_number) n FROM order_lines o "
                       "JOIN calendar c ON c.date_key = o.order_date_key WHERE year(c.full_date) = 2025 "
                       "GROUP BY 1 ORDER BY n DESC LIMIT 1").fetchone()
    assert [(r["day_of_week"], r["number_of_orders"]) for r in rows] == [want]


def test_a_monthly_series_by_weekday_keeps_both(retail):
    con, model = retail
    rows = _rows(con, model, {"intent": "trend", "measures": ["number_of_orders"], "group_by": ["time:day_of_week"],
                              "time": {"grain": "month", "window": Y25}})
    assert len({r["period"] for r in rows}) == 12 and len({r["day_of_week"] for r in rows}) == 7
