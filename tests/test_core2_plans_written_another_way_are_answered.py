"""Plans a real model wrote another way than the planner's rules, answered instead of refused.

Asked of the AdventureWorks sample with a real model (gpt-4o):

* "How many customers placed an order in 2024?" was written with the year as a filter on the order date, and
  refused twice ("Order date is not a date of what was asked"): a date filter between two days, with no period of
  its own, is the plan's period;
* "How many order lines had a discount in 2024?" was written as the number of lines with a filter on the
  discount, and refused ("a filter on Discount needs it in the answer"): a condition on a figure of the rows
  counted, not asked for itself, keeps the rows whose own figure meets it.

Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.ir import Plan
from core2.plan.planner import dates_as_window
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


def _figure(payload: dict) -> float:
    return next(v for v in payload["export_rows"][0].values() if isinstance(v, (int, float)))


def test_a_year_written_as_a_date_filter_is_the_plans_period(retail):
    plan = {"kind": "query", "intent": "value", "measures": ["number_of_customers"],
            "filters": [{"field": "order_date", "op": "between", "values": ["2025-01-01", "2025-12-31"]}],
            "time": {"date": "order_date"}}
    moved = dates_as_window(Plan.model_validate(plan), retail[1], "")
    assert not moved.filters and moved.time.window.kind == "between" and moved.time.date == "order_date"
    assert (moved.time.window.start, moved.time.window.end) == (dt.date(2025, 1, 1), dt.date(2025, 12, 31))
    payload = _ask(retail, plan, "How many customers placed an order in 2025?")
    expected = retail[0].con.execute("SELECT COUNT(DISTINCT customer_id) FROM order_lines "
                                     "WHERE order_date_key BETWEEN 20250101 AND 20251231").fetchone()[0]
    assert _figure(payload) == expected


def test_a_date_filter_beside_a_period_stands(retail):
    plan = Plan.model_validate({"kind": "query", "intent": "value", "measures": ["number_of_order_lines"],
                                "filters": [{"field": "ship_date", "op": "between",
                                             "values": ["2025-01-01", "2025-01-31"]}],
                                "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}})
    assert dates_as_window(plan, retail[1], "") is plan


def test_rows_counted_by_a_condition_on_their_own_figure(retail):
    plan = {"kind": "query", "intent": "value", "measures": ["number_of_order_lines"],
            "filters": [{"field": "discount_amount", "op": "gt", "values": [0]}]}
    payload = _ask(retail, plan, "How many order lines had a discount?")
    expected = retail[0].con.execute("SELECT COUNT(*) FROM order_lines WHERE discount_amount > 0").fetchone()[0]
    assert 0 < expected < retail[0].con.execute("SELECT COUNT(*) FROM order_lines").fetchone()[0]
    assert _figure(payload) == expected
    notes = " ".join([*(payload.get("trust", {}).get("date_context") or []), *(payload.get("notes") or [])])
    assert "whose own discount amount is above 0" in notes.lower(), notes


def test_a_condition_on_a_documents_total_is_not_read_row_by_row(retail):
    """ "How many orders had a net amount over 100?" is of each order's total: never the orders with one line over
    100. The plan is sent back rather than answered row by row."""
    from core2.plan.planner import check

    plan = Plan.model_validate({"kind": "query", "intent": "value", "measures": ["number_of_orders"],
                                "filters": [{"field": "net_amount", "op": "gt", "values": [100]}]})
    problems, _ = check(plan, retail[1], TODAY, "How many orders had a net amount over 100?")
    assert problems and "needs it in the answer" in problems[0], problems
