"""The analyses each warehouse's data can give are found from what Learn learned, offered to readers in their own
words, and each one offered is answered.

Members with dated activity (customers and orders, patients and claims) give the time between their events, who
is new, returning or lapsed, and cohorts; a list with a start and an end (employees, subscriptions) gives how many
are in force at each month end; a date things were promised by and the date they came (receipts of order lines)
gives the share on time. Only parties come back or lapse: an item, a cost centre or a ledger account is offered
none of the three, and a movement with an expiry date is no list in force.

Invented data only.
"""

from __future__ import annotations

import datetime as dt

import pytest

from core2.answer.describe import describe
from core2.compile.compiler import compile_query
from core2.plan.analyses import questions, suggestions
from core2.plan.ir import Plan
from core2.resolve.resolver import Context, resolve
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
EXPECTED = {
    "retail": ["Average days between orders per customer", "New, returning and lapsed customers by month",
               "Customer retention by month of first order"],
    "hr": ["Employees at each month end"],
    "purchasing": ["Share of receipts on time"],
    "subscriptions": ["Average days between invoices per customer", "Subscriptions at each month end"],
    "inventory": [],
    "finance": [],
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_warehouse_is_offered_the_analyses_its_data_gives_and_each_is_answered(name):
    built, model = learn(domains.build(name), "descriptive")
    offered = suggestions(model, this_year=TODAY.year)
    said = [q for q, _ in offered]
    for lead in EXPECTED[name]:
        assert any(q.startswith(lead) for q in said), (lead, said)
    if not EXPECTED[name]:
        assert said == []
    for question, plan in offered:
        compiled = compile_query(resolve(Plan.model_validate(plan), model, Context(today=TODAY)), model, "duckdb")
        rows = built.con.execute(compiled.sql).fetchall()
        assert rows, question


def test_the_analyses_close_what_you_can_ask_and_learns_page():
    built, model = learn(domains.build("retail"), "descriptive")
    described = describe(model, [], today=TODAY)
    assert "Customer retention by month of first order" in described.examples
    from core2.model.view import learned_view
    assert questions(model, this_year=TODAY.year)[0] in learned_view(model)["analyses"]
