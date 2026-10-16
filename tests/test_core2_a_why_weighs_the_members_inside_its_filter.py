"""A "why" narrowed by a property of a thing still weighs each of those things; a blank never explains a change.

"Why did Ontario revenue fall in Q3?" was answered by promotion ("Unknown": the orders with no promotion, nearly all
of them) and by product status, never by customer -- yet one lost customer was the whole fall. The question was
narrowed by the customers' province, and every grouping of the customer was then left out as if one customer were
left. Only a filter on the thing itself (its name, its code) leaves one member; a filter on a property of it
(a province, a segment) leaves many, each of which may explain the change. And a blank member (no promotion, no
reason given) is shown with the rest but never named as what drove a change.

Invented retail data; every figure is checked against SQL written by hand.
"""
from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.answer.drivers import Grouping, Mover, rank, score, telling
from core2.plan.values import build_index
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn
from evals.core2.framework import materialize

TODAY = dt.date(2026, 6, 15)
MAY_VS_APRIL = {"window": {"kind": "between", "start": "2026-05-01", "end": "2026-05-31"},
                "compare": {"kind": "window", "window": {"kind": "between", "start": "2026-04-01", "end": "2026-04-30"}}}


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


def _why(retail, question, filters):
    model, warehouse, index = retail
    plan = {"kind": "query", "intent": "drivers", "measures": ["net_amount"], "filters": filters, "time": MAY_VS_APRIL}
    services = Services(model=model, warehouse=warehouse, complete=lambda s, t: json.dumps(plan), index=index,
                        today=TODAY)
    return answer_question(question, services, Session())


def test_a_why_narrowed_by_a_property_of_customers_still_weighs_each_customer(retail):
    wholesale = [{"field": "customer.segment", "op": "eq", "values": ["Wholesale"]}]
    payload = _why(retail, 'why did net sales of "Wholesale" customers change in May 2026 against April 2026?',
                   wholesale)
    checked = {g["slug"]: g for g in payload["drivers"]["groupings"]}
    assert "customer" in checked, sorted(checked)
    hand = dict(retail[1].query("""
        SELECT c.customer_name,
               SUM(CASE WHEN o.order_date_key BETWEEN 20260501 AND 20260531 THEN o.net_amount ELSE 0 END)
             - SUM(CASE WHEN o.order_date_key BETWEEN 20260401 AND 20260430 THEN o.net_amount ELSE 0 END)
        FROM order_lines o JOIN customers c ON c.customer_id = o.customer_id
        WHERE c.segment = 'Wholesale' AND o.order_date_key BETWEEN 20260401 AND 20260531
        GROUP BY 1""").rows)
    leaders = checked["customer"]["leaders"]
    assert leaders
    for leader in leaders:
        name = next(n for n in hand if leader["member"].startswith(n))
        assert round(leader["change"], 2) == round(float(hand[name]), 2), name


def test_a_why_narrowed_to_one_customer_does_not_split_it_by_customer(retail):
    one = retail[1].query("SELECT customer_name FROM customers ORDER BY customer_id LIMIT 1").rows[0][0]
    payload = _why(retail, f'why did net sales of "{one}" change in May 2026 against April 2026?',
                   [{"field": "customer.name", "op": "eq", "values": [one]}])
    assert "customer" not in {g["slug"] for g in payload["drivers"]["groupings"]}


def test_a_blank_member_is_never_what_drove_a_change():
    # Orders with no promotion: nearly all of them, and most of the fall -- which says nothing about why.
    by_promotion = Grouping("promotion", "Promotion", [Mover("Unknown", 950.0, 800.0, blank=True),
                                                      Mover("Spring sale", 30.0, 25.0),
                                                      Mover("Back to school", 20.0, 15.0)])
    score(by_promotion, -160.0, True)
    assert all(not m.blank for m in by_promotion.leaders)
    assert not telling(by_promotion)
    # One customer lost: a small share of the base, the whole of the fall.
    by_customer = Grouping("customer", "Customer", [Mover("Lost Co", 160.0, 0.0), Mover("Steady Ltd", 420.0, 420.0),
                                                    Mover("Even Inc", 420.0, 420.0)])
    score(by_customer, -160.0, True)
    assert telling(by_customer) and rank([by_promotion, by_customer])[0] is by_customer
