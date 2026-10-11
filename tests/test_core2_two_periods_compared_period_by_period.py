"""Two years compared quarter by quarter (or month by month): each period beside the same period of the other.

Asked of public samples with a real model (gpt-4o), "Compare the order subtotal of 2024 with 2023 by quarter" and
"Compare it with 2024" (a monthly series) were grouped by each row's own quarter or month: 2023's figures came out
as rows of their own, beside nothing, and the answer said "Q3 rose most; Q3 fell most". A row of the compared
period is now counted under the period it is set against, a whole number of months on. Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def _ask(retail, plan: dict, question: str) -> dict:
    built, model = retail
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: json.dumps(plan),
                        index=MemberIndex(), today=TODAY)
    return answer_question(question, services, Session())


def _by_quarter(con, year: int) -> dict[int, float]:
    return {int(q): float(v) for q, v in con.execute(
        f"SELECT (order_date_key // 100 % 100 - 1) // 3 + 1, SUM(net_amount) FROM order_lines "
        f"WHERE order_date_key // 10000 = {year} GROUP BY 1").fetchall()}


@pytest.mark.parametrize("compare", [
    {"kind": "window", "window": {"kind": "between", "start": "2024-01-01", "end": "2024-12-31"}},
    {"kind": "same_period_last_year"},
])
def test_each_quarter_stands_beside_the_same_quarter_a_year_before(retail, compare):
    plan = {"kind": "query", "intent": "compare", "measures": ["net_amount"],
            "time": {"grain": "quarter", "window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"},
                     "compare": compare}}
    payload = _ask(retail, plan, "Compare net amount of 2025 with 2024 by quarter")
    rows = payload["export_rows"]
    assert len(rows) == 4, rows
    now, before = _by_quarter(retail[0].con, 2025), _by_quarter(retail[0].con, 2024)
    for row in rows:
        quarter = (int(str(row["period"])[5:7]) - 1) // 3 + 1
        assert str(row["period"]).startswith("2025"), row
        assert row["net_amount"] == pytest.approx(now[quarter]) and row["net_amount_prior"] == pytest.approx(
            before[quarter]), row


def test_compare_it_after_a_series_keeps_the_series(retail):
    """ "Compare it with 2024" after the monthly claim cost of one class in 2025 was answered with the two years'
    totals: "it" is the series on screen, month by month against the other year's."""
    from core2.plan.ir import Plan
    from core2.plan.planner import settle

    window = {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}
    series = Plan.model_validate({"kind": "query", "intent": "trend", "measures": ["net_amount"],
                                  "time": {"grain": "month", "window": window}})
    ai = Plan.model_validate({"kind": "query", "intent": "compare", "measures": ["net_amount"], "follow_up": "refine",
                              "time": {"grain": "month", "window": window, "compare": {
                                  "kind": "window", "window": {"kind": "between", "start": "2024-01-01",
                                                              "end": "2024-12-31"}}}})
    assert settle(ai, retail[1], "Compare it with 2024", series).time.grain == "month"
    by_store = ai.model_copy(update={"group_by": ["store.name"]})       # members compared: one total each
    assert settle(by_store, retail[1], "Which store grew the most against 2024?", series).time.grain is None


def test_a_series_by_month_is_not_also_grouped_by_the_months_name(retail):
    """A dashboard's "Encounters by month" was planned month by month and grouped by the month's name too: each
    month a series of one point. Inside one year the name is the series' own axis; over several years it is not."""
    from core2.plan.ir import Plan
    from core2.plan.planner import one_time_axis

    def plan(start: str, end: str) -> Plan:
        return Plan.model_validate({"kind": "query", "intent": "trend", "measures": ["net_amount"],
                                    "group_by": ["time:month_of_year"], "time": {
                                        "grain": "month", "window": {"kind": "between", "start": start, "end": end}}})

    assert one_time_axis(plan("2025-01-01", "2025-12-31"), retail[1], "Net amount by month").group_by == []
    several = plan("2024-01-01", "2025-12-31")
    assert one_time_axis(several, retail[1], "Net amount by month of year, 2024 and 2025") is several
