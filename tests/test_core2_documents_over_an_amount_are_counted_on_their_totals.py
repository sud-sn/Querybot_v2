"""How many orders were over an amount counts orders whose own total is over it.

"How many orders over $20,000 did we get in 2026?" was planned as the number of orders with a condition on the gross
amount, and answered 11,844 (of 15,000 orders): the condition read each line, or the year's total, never each
order's total. A condition on a figure, asked of the things a count counts, is on each one's own total: the orders
whose lines add up to more than the amount.

Invented retail data; every figure is checked against SQL written by hand.
"""
from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from evals.core2 import domains
from evals.core2.compile_eval import learn
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 6, 15)
MAY = {"window": {"kind": "between", "start": "2026-05-01", "end": "2026-05-31"}}


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return built, model


def _ask(retail, question, plan):
    built, model = retail
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con),
                        complete=lambda s, t: json.dumps({"kind": "query", **plan}), index=MemberIndex(), today=TODAY)
    payload = answer_question(question, services, Session())
    assert payload.get("type") == "assistant_response" and "export_rows" in payload, payload.get("answer")
    return payload


def _hand(retail, over: float) -> int:
    return retail[0].con.execute(f"""SELECT COUNT(*) FROM (SELECT order_number FROM order_lines
        WHERE order_date_key BETWEEN 20260501 AND 20260531 GROUP BY 1 HAVING SUM(net_amount) > {over})""").fetchone()[0]


def test_orders_over_an_amount_are_the_orders_whose_total_is_over_it(retail):
    over = float(retail[0].con.execute("""SELECT MEDIAN(t) FROM (SELECT SUM(net_amount) t FROM order_lines
        WHERE order_date_key BETWEEN 20260501 AND 20260531 GROUP BY order_number)""").fetchone()[0])
    want = _hand(retail, over)
    assert 0 < want
    for plan in ({"intent": "count", "measures": ["number_of_orders"], "time": MAY,
                  "filters": [{"field": "net_amount", "op": "gt", "values": [over]}]},
                 {"intent": "count", "measures": ["number_of_orders", "net_amount"], "time": MAY,
                  "filters": [{"field": "net_amount", "op": "gt", "values": [over]}]}):
        payload = _ask(retail, f"How many orders over ${over:,.0f} did we get in May 2026?", plan)
        assert want in [v for row in payload["export_rows"] for v in row.values()], payload["export_rows"]
