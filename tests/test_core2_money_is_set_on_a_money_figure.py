"""An amount of money in the question is set on a money figure, never on a weight, a quantity or a date.

"How many items in the top group cost more than $1,000 each?": a real model (gpt-4o) set the condition on a date,
was told the date was no number, and set it on the items' net weight, answering 0. The repair round is now told
that the amount is money, and which figures are money (those the question's words name first).

Synthetic data only.
"""

from __future__ import annotations

import datetime as dt

import pytest

from core2.plan.ir import Plan
from core2.plan.planner import check
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)


@pytest.fixture(scope="module")
def model():
    return learn(domains.build("retail"), "descriptive")[1]


@pytest.mark.parametrize("field, question, flagged", [
    ("quantity", "How many products cost more than $1,000 each?", True),          # money on a quantity
    ("order_date", "How many orders were over 1,000 dollars?", True),             # money on a date
    ("product.list_price", "How many products cost more than $1,000 each?", False),
    ("net_amount", "How many stores sold more than $1k?", False),
    ("quantity", "How many products sold more than 1000 units?", False),          # no money said
])
def test_an_amount_of_money_is_set_on_a_money_figure(model, field, question, flagged):
    plan = Plan.model_validate({"kind": "query", "measures": ["number_of_products"],
                                "filters": [{"field": field, "op": "gt", "values": [1000]}]})
    problems = [p for p in check(plan, model, TODAY, question)[0] if "is money" in p]
    assert bool(problems) == flagged, problems
    if flagged and "cost" in question:
        assert "cost_amount" in problems[0]
