"""A count of a list's own members grouped by those members is 1 each: the plan is sent back.

Asked of the AdventureWorks sample with a real model (gpt-4o), "What are the most common sales reasons on internet
orders?" was planned as the number of sales reasons by sales reason, and answered "On Promotion has the highest
number of sales reasons, at 1". The repair round is told that how often each occurs is a count of the rows that
record them. Counting the members by another of their fields (stores by region) stands. Synthetic data only.
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


def _rank(measure: str, group: str) -> Plan:
    return Plan.model_validate({"kind": "query", "intent": "rank", "measures": [measure], "group_by": [group],
                                "sort": [{"by": measure, "desc": True}], "limit": 5})


@pytest.mark.parametrize("group", ["store.name", "store.code", "store"])
def test_members_counted_by_themselves_are_sent_back(model, group):
    problems, _ = check(_rank("number_of_stores", group), model, TODAY, "Which stores are the most common?")
    assert any("counts the stores themselves" in p and "each one is 1" in p and "number_of_order_lines" in p
               for p in problems), problems


def test_members_counted_by_another_field_or_rows_by_member_stand(model):
    region = next(a for a in model.attributes if a.startswith("region."))
    assert not check(_rank("number_of_stores", region), model, TODAY, "How many stores in each region?")[0]
    assert not check(_rank("number_of_order_lines", "store.name"), model, TODAY, "Which stores sell most often?")[0]
