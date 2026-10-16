"""A share of repeat members asked for is their share, not their number.

"What % of customers who bought in 2025 ordered more than once that year?" was planned as the repeat customers
counted (593) where the question asks what part of the customers who bought they are (95.3%). Repeat (or one-time)
members asked as a share are each member's count of activity, and the share of them with more than one (or one).

Invented retail data; every figure is checked against SQL written by hand.
"""
from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.values import build_index
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn
from evals.core2.framework import materialize

TODAY = dt.date(2026, 6, 15)
Q1 = {"kind": "between", "start": "2025-01-01", "end": "2025-03-31"}


@pytest.fixture(scope="module")
def retail():
    domain = domains.build("retail")
    built, model = learn(domain, "descriptive")
    warehouse = DuckDBWarehouse(materialize(domain, "descriptive").con)

    def fetch(slug):
        column = model.columns[model.attributes[slug].column]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM '
                                              f'"{model.tables[column.table].name}"').rows]

    return model, warehouse, build_index(model, fetch)


def _ask(retail, question, plan):
    model, warehouse, index = retail
    services = Services(model=model, warehouse=warehouse, complete=lambda s, t: json.dumps({"kind": "query", **plan}),
                        index=index, today=TODAY)
    payload = answer_question(question, services, Session())
    assert payload.get("type") == "assistant_response" and "export_rows" in payload, payload.get("answer")
    return payload


def _hand(retail, cancelled_out):
    # The product leaves cancelled lines out by default where the model says so; the hand count follows its SQL.
    out = "AND status_code <> 'C'" if cancelled_out else ""
    rows = retail[1].query(f"""SELECT COUNT(*) FILTER (WHERE n > 1), COUNT(*) FILTER (WHERE n = 1), COUNT(*) FROM (
        SELECT customer_id, COUNT(DISTINCT order_number) n FROM order_lines
        WHERE order_date_key BETWEEN 20250101 AND 20250331 {out} GROUP BY 1)""").rows
    return [int(v) for v in rows[0]]


def _figure(payload):
    row = payload["export_rows"][0]
    return [float(v) for v in row.values() if isinstance(v, (int, float))]


@pytest.mark.parametrize("status,question,pick", [
    ("repeat", "What % of customers who bought in Q1 2025 ordered more than once in it?", 0),
    ("one_time", "What share of Q1 2025 customers ordered only once?", 1),
])
def test_a_share_of_repeat_or_one_time_customers_is_their_share(retail, status, question, pick):
    plan = {"intent": "value", "time": {"window": Q1},
            "lifecycle": {"of": "customer", "measure": "number_of_orders", "status": status}}
    payload = _ask(retail, question, plan)
    sql = (payload.get("trust") or {}).get("sql") or ""
    counts = _hand(retail, "'C'" in sql)
    assert 0 < counts[pick] < counts[2]
    want = 100.0 * counts[pick] / counts[2]
    assert len(payload["export_rows"]) == 1
    assert any(v == pytest.approx(want, abs=0.05) or v == pytest.approx(want / 100, abs=0.0005)
               for v in _figure(payload)), (want, payload["export_rows"], payload["answer"]["headline"])


def test_repeat_customers_counted_stay_a_count(retail):
    plan = {"intent": "value", "time": {"window": Q1},
            "lifecycle": {"of": "customer", "measure": "number_of_orders", "status": "repeat"}}
    payload = _ask(retail, "How many repeat customers did we have in Q1 2025?", plan)
    assert "%" not in payload["answer"]["headline"]


def test_each_customers_count_of_customers_goes_back_to_be_planned_again(retail):
    # The same question planned as each customer's number of customers (always 1), the share of those over 1: 0%.
    from core2.plan.ir import Plan
    from core2.plan.planner import check

    plan = Plan.model_validate({"kind": "query", "intent": "share", "measures": ["number_of_customers"],
                                "time": {"window": Q1},
                                "per": {"of": "customer", "summary": "share", "op": "gt", "values": [1.0]}})
    problems, _ = check(plan, retail[0], TODAY, "What % of customers who bought in Q1 2025 ordered more than once?")
    assert any("counts the customers themselves" in p and "number_of_orders" in p for p in problems), problems
