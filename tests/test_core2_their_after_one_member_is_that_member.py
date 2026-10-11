""""Their" after a ranking means the members on screen, for another figure too.

Asked of the AdventureWorks sample with a real model (gpt-4o), "What was their sales quota for 2013?" after "Which
sales rep had the highest reseller sales in 2013?" (Linda Mitchell) was asked back: "Which employee's sales quota
do you want to know?". The members on screen are the ones meant: the AI is told so once, and the plan is kept to
them in code, the same period or not, since another figure's ranking would find other members. Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.ir import Plan
from core2.plan.planner import Turn, plan_question, those_members
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
YEAR = {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}


@pytest.fixture(scope="module")
def model():
    return learn(domains.build("retail"), "descriptive")[1]


def _top_store() -> Plan:
    return Plan.model_validate({"kind": "query", "intent": "rank", "measures": ["net_amount"],
                                "group_by": ["store.name"], "sort": [{"by": "net_amount", "desc": True}], "limit": 1,
                                "time": {"window": YEAR}})


def _their_returns() -> dict:
    return {"kind": "query", "intent": "value", "measures": ["number_of_returns"], "follow_up": "refine",
            "time": {"window": YEAR}}


def test_another_figure_for_the_same_period_is_kept_to_the_member_on_screen(model):
    plan = those_members(Plan.model_validate(_their_returns()), model, "What were their returns?",
                         _top_store(), [("store.name", "Store 7")])
    assert any(f.field == "store.name" and f.values == ["Store 7"] for f in plan.filters), plan.filters
    # The same figure in the same period: its own order and limit find the member again.
    same = _top_store().model_copy(update={"follow_up": "refine"})
    assert those_members(same, model, "And their net amount?", _top_store(), [("store.name", "Store 7")]) is same


def test_asking_which_one_is_answered_for_the_member_on_screen(model):
    replies = [{"kind": "clarify", "clarify": {"about": "member", "question": "Which store do you mean?"}},
               _their_returns()]
    seen: list[str] = []

    def complete(stable: str, tail: str) -> str:
        seen.append(tail)
        return json.dumps(replies.pop(0))

    outcome = plan_question(model, "How many returns did their store have?", complete, today=TODAY,
                            history=[Turn("Which store had the highest net amount in 2025?", _top_store())],
                            on_screen=[("store.name", "Store 7")])
    assert "members of the answer above" in seen[-1]
    assert outcome.plan.kind == "query" and any(f.field == "store.name" and f.values == ["Store 7"]
                                                for f in outcome.plan.filters), outcome.plan


def test_a_stand_in_for_the_one_member_on_screen_is_that_member(model):
    """The AI wrote "their" as a filter on "<previous_top_employee>": with one member on screen, it is that one;
    with several and no place word ("first", "top"), it stays for the reader to say which."""
    from core2.plan.planner import pointed_in

    plan = Plan.model_validate({"kind": "query", "intent": "value", "measures": ["number_of_returns"],
                                "filters": [{"field": "store.name", "op": "eq", "values": ["<previous_top_store>"]}]})
    one = pointed_in(plan, model, [], "What were their returns?", [("store.name", "Store 7")])
    assert one.filters[0].values == ["Store 7"]
    two = pointed_in(plan, model, [], "What were their returns?", [("store.name", "Store 7"), ("store.name", "Store 2")])
    assert two.filters[0].values == ["<previous_top_store>"]
