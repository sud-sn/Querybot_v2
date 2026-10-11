"""Counts, conditions and rows as the question means them, whatever the AI wrote.

Found asking real conversations of a real model (gpt-4o), on a copy of an inventory warehouse:

* "How many items in the top group cost more than $1,000 each?" was planned as the number of items, kept only
  if the group's own average cost was above $1,000: 0. A count of a list's members with a condition on another
  figure counts the members whose own figure meets it;
* "How many items does the top one have on hand?" was answered with the stock on hand: "how many" of a thing the
  data lists is sent back to be a count of it, or a list of them;
* "Where do we have back-ordered quantity?" listed every warehouse, most with none: where there is some;
* "What drove the increase in the warehouse that grew the most?" pointed at the one that fell least, every
  warehouse named on screen having fallen: none grew, and none is pointed at;
* "How many different items do we hold from the first supplier on that list?" was asked back ("which supplier?",
  with "top supplier from the previous list" as the one option), the row being in VALUE MATCHES: it is planned.

Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.ir import Plan
from core2.plan.planner import check, counted_each, plan_question, where_we_have
from core2.plan.values import ValueMatch, placed
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def test_a_count_with_a_condition_on_another_figure_counts_the_members_that_meet_it(retail):
    built, model = retail
    ai = Plan.model_validate({"kind": "query", "intent": "count", "measures": ["number_of_products", "net_amount"],
                              "filters": [{"field": "net_amount", "op": "gt", "values": [50000]}]})
    plan = counted_each(ai, model, "How many products sold more than $50,000 each?")
    assert (plan.measures, plan.per.of, plan.per.summary, plan.per.op, plan.per.values) == (
        ["net_amount"], "product", "count", "gt", [50000.0])
    assert not check(plan, model, TODAY, "How many products sold more than $50,000 each?")[0]
    # Split by something, or a single figure: left as the AI wrote it.
    grouped = ai.model_copy(update={"group_by": ["store.name"]})
    assert counted_each(grouped, model, "q") is grouped


def test_how_many_of_a_listed_thing_is_a_count_of_it(retail):
    model = retail[1]
    quantity = Plan.model_validate({"kind": "query", "intent": "value", "measures": ["quantity"],
                                    "filters": [{"field": "store.name", "op": "eq", "values": ["Mall"]}]})
    problems = [p for p in check(quantity, model, TODAY, "How many products does the Mall store sell?")[0]
                if "how many products" in p]
    assert problems and "number_of_products" in problems[0]
    for fine in (quantity.model_copy(update={"measures": ["number_of_products"]}),
                 quantity.model_copy(update={"group_by": ["product.name"]})):
        assert not [p for p in check(fine, model, TODAY, "How many products does the Mall store sell?")[0]
                    if "how many products" in p]
    assert not [p for p in check(quantity, model, TODAY, "How many units did the Mall store sell?")[0]
                if "how many" in p]


def test_where_we_have_a_figure_is_where_it_is_above_zero(retail):
    model = retail[1]
    ai = Plan.model_validate({"kind": "query", "intent": "list", "measures": ["discount_amount"],
                              "group_by": ["store.name"]})
    plan = where_we_have(ai, model, "Where do we have discounts this year?")
    assert [(f.field, f.op, f.values) for f in plan.filters] == [("discount_amount", "gt", [0])]
    assert where_we_have(ai, model, "Discount by store") is ai


def test_the_one_that_grew_the_most_is_one_that_grew():
    shown = [("store.name", "Airport", 900.0, {}, -50.0), ("store.name", "Mall", 500.0, {}, -10.0)]
    assert placed("What drove the increase in the store that grew the most?", shown, []) == []
    assert [m.value for m in placed("What drove the drop in the store that fell the most?", shown, [])] == ["Airport"]
    grew = [("store.name", "Airport", 900.0, {}, 30.0), *shown[1:]]
    assert [m.value for m in placed("What drove the increase in the store that grew the most?", grew, [])] == ["Airport"]


def test_asked_which_member_when_the_question_found_it_is_planned(retail):
    model = retail[1]
    asked = {"kind": "clarify", "clarify": {"about": "member", "question": "Which store do you mean?",
                                            "options": ["top store from the previous list"]}}
    answer = {"kind": "query", "intent": "value", "measures": ["net_amount"],
              "filters": [{"field": "store.name", "op": "eq", "values": ["Airport"]}]}
    replies = [json.dumps(asked), json.dumps(answer)]
    prompts: list[str] = []

    def ai(stable, tail):
        prompts.append(tail)
        return replies.pop(0)

    matches = [ValueMatch("the first store", 9, 24, "store.name", "Airport")]
    outcome = plan_question(model, "How much did the first store on that list sell?", ai, today=TODAY,
                            matches=matches)
    assert outcome.plan.kind == "query" and outcome.plan.filters[0].values == ["Airport"]
    assert "Plan the question for it" in prompts[1] and "Airport" not in prompts[1].split("PROBLEMS:")[1]


def test_two_periods_are_compared_from_the_earlier_to_the_later(retail):
    """ "How did those items do in Q1 2026?" after Q2 was planned as Q1 against Q2: "rose" read backwards."""
    from core2.plan.planner import in_time_order

    q1 = {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}
    q2 = {"kind": "between", "start": "2026-04-01", "end": "2026-06-30"}
    backwards = Plan.model_validate({"kind": "query", "intent": "compare", "measures": ["net_amount"],
                                     "time": {"window": q1, "compare": {"kind": "window", "window": q2}}})
    plan = in_time_order(backwards, retail[1], "How did they do in Q1 2026?")
    assert (str(plan.time.window.start), str(plan.time.compare.window.start)) == ("2026-04-01", "2026-01-01")
    assert in_time_order(plan, retail[1], "q") is plan
