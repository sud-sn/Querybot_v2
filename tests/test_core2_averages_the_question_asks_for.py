"""An average the question asks for is an average.

Found asking public sample warehouses through the chat with a real model (gpt-4o):

* "Which territory had the highest average order subtotal in 2024?" was planned as the subtotal added up, and named
  the territory with the most sales: a figure that adds up, written after "average", is each row's figure
  averaged (the average order in each territory);
* "Average vacation hours by gender", "what is the average list price of our bikes?" and "average service grade by
  shift" were refused or summed: a number each member has (a product's list price) is averaged when a plan
  measures it.

Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.ir import Plan
from core2.plan.planner import average_asked
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def _plan(**parts) -> Plan:
    return Plan.model_validate({"kind": "query", "intent": "rank", "measures": ["net_amount"], **parts})


@pytest.mark.parametrize("question, averaged", [
    ("Which store had the highest average net amount in 2025?", True),
    ("Average net amount by store", True),
    ("Average net amount per customer by store", False),       # what to average over: the AI's to plan
    ("Which store had the highest net amount in 2025?", False),
])
def test_a_figure_after_average_is_each_rows_figure_averaged(retail, question, averaged):
    planned = average_asked(_plan(group_by=["store"]), retail[1], question)
    assert (planned.per is not None and planned.per.of is None and planned.per.summary == "avg") is averaged


def _ask(retail, plan: dict, question: str) -> dict:
    built, model = retail
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: json.dumps(plan),
                        index=MemberIndex(), today=TODAY)
    return answer_question(question, services, Session())


def test_the_average_order_by_store_is_the_average_of_its_rows(retail):
    built, _ = retail
    payload = _ask(retail, {"kind": "query", "intent": "rank", "measures": ["net_amount"], "group_by": ["store"],
                            "sort": [{"by": "net_amount", "desc": True}], "limit": 1},
                   "Which store had the highest average net amount?")
    best = built.con.execute("SELECT s.store_name, AVG(o.net_amount) v FROM order_lines o JOIN stores s "
                             "ON o.store_id = s.store_id GROUP BY 1 ORDER BY v DESC LIMIT 1").fetchone()
    row = payload["export_rows"][0]
    assert row["store_name"] == best[0]
    assert any(isinstance(v, float) and v == pytest.approx(float(best[1])) for v in row.values()), row


def test_a_number_each_product_has_is_averaged_by_category(retail):
    built, _ = retail
    payload = _ask(retail, {"kind": "query", "intent": "breakdown", "measures": ["product.list_price"],
                            "group_by": ["category"]}, "Average list price by category")
    want = {k: float(v) for k, v in built.con.execute(
        "SELECT c.category_name, AVG(p.list_price) FROM products p JOIN categories c "
        "ON p.category_id = c.category_id GROUP BY 1").fetchall()}
    got = {r["category_name"]: next(v for k, v in r.items() if k != "category_name") for r in payload["export_rows"]}
    assert got == pytest.approx(want), payload["answer"]["headline"]
    assert payload["answer"]["headline"].startswith("Average list price"), payload["answer"]["headline"]


def test_an_average_of_a_highest_figure_is_the_number_each_member_has():
    """ "Average list price by product line" took a price history's list price, kept as each product's highest: the
    list price each product has, averaged over the products."""
    from core2.plan.ir import Plan
    from core2.plan.planner import average_asked
    from evals.core2 import domains
    from evals.core2.compile_eval import learn

    model = learn(domains.build("retail"), "descriptive")[1].model_copy(deep=True)
    highest = next(m for m in model.measures.values() if m.slug == "unit_price")
    highest.expr = highest.expr.model_copy(update={"agg": "max"})
    highest.business_name = "List price"
    plan = Plan.model_validate({"kind": "query", "intent": "breakdown", "measures": ["unit_price"],
                                "group_by": ["category.name"], "sort": [{"by": "unit_price", "desc": True}]})
    averaged = average_asked(plan, model, "Average list price by category")
    assert averaged.measures == ["product.list_price"] and averaged.sort[0].by == "product.list_price"
    assert average_asked(plan, model, "Highest list price by category") is plan
