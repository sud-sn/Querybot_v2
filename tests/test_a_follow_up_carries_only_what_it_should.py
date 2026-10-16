"""A follow-up carries what its words point at, and nothing a fresh question leaves behind.

Found asking business readers' questions of a real model (gpt-4o) in the stress test:

* "Prairie Pallets is way up, show me their spend by month" after the top 5 suppliers kept the ranking's
  limit as a row limit on the months: five months of one supplier came back, sorted by value. A series of
  those members runs month by month for each of them, in month order;
* "Spend by supplier country for 2025" after a supplier's months was read as a fragment of that answer (no
  measure is called "spend"), so the supplier and the months were kept: a turn that names a field to split by
  and a period of its own is a whole new question;
* "Break the partner ones down by partner" points at the answer on screen ("the partner ones");
* "Why did they cancel?" was answered as why the figure on screen changed (the members that moved it): it asks
  the reasons people had, which only data holding reasons can answer.

Synthetic data only.
"""

from __future__ import annotations

import pytest

from core2.plan.followup import read_turn
from core2.plan.ir import Plan
from core2.plan.planner import settle
from evals.core2 import domains
from evals.core2.compile_eval import learn

YEAR = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}


@pytest.fixture(scope="module")
def model():
    return learn(domains.build("retail"), "descriptive")[1]


def _plan(**kw) -> Plan:
    return Plan.model_validate({"kind": "query", **kw})


def _top5():
    return _plan(intent="rank", measures=["net_amount"], group_by=["store.name"], time=YEAR, limit=5,
                 sort=[{"by": "net_amount", "desc": True}])


def test_a_series_of_those_members_is_every_month_of_each(model):
    ai = _plan(intent="trend", measures=["net_amount"], group_by=["store.name"],
               time={**YEAR, "grain": "month"}, limit=5, sort=[{"by": "net_amount", "desc": True}],
               filters=[{"field": "store.name", "op": "eq", "values": ["Riverside"]}], follow_up="refine")
    settled = settle(ai, model, "Riverside is way up, show me their sales by month in 2025", _top5())
    assert settled.limit is None
    assert all(s.by == "period" for s in settled.sort)


def test_the_same_members_in_another_period_keep_the_ranking(model):
    ai = _plan(intent="rank", measures=["net_amount"], group_by=["store.name"],
               time={"window": {"kind": "between", "start": "2024-01-01", "end": "2024-12-31"}}, limit=5,
               follow_up="refine")
    assert settle(ai, model, "and how did those do in 2024?", _top5()).limit == 5


def test_a_turn_with_its_own_split_and_period_is_new(model):
    narrowed = _plan(intent="trend", measures=["net_amount"], time={**YEAR, "grain": "month"},
                     filters=[{"field": "store.name", "op": "eq", "values": ["Riverside"]}])
    assert read_turn("Spend by region for 2025", previous=narrowed, model=model).kind == "new"
    assert read_turn("and by region for 2025?", previous=narrowed, model=model).kind != "new"


def test_the_x_ones_point_at_the_answer_on_screen(model):
    assert read_turn("break the online ones down by store", previous=_top5(), model=model).kind == "refine"


@pytest.mark.parametrize("question", ["why did they cancel?", "Why do customers leave?"])
def test_why_people_did_something_is_not_why_the_figure_moved(model, question):
    assert read_turn(question, previous=_top5(), model=model).kind == "open"


@pytest.mark.parametrize("question", ["why did it drop?", "why did they spend less?", "Why?"])
def test_why_the_figure_moved_is_about_the_answer_on_screen(model, question):
    assert read_turn(question, previous=_top5(), model=model).kind == "refine"


def _orders_at_riverside():
    return _plan(intent="value", measures=["number_of_orders"],
                 filters=[{"field": "store.name", "op": "eq", "values": ["Riverside"]}],
                 time={"window": {"kind": "between", "start": "2025-09-01", "end": "2025-09-30"}})


@pytest.mark.parametrize("question", ['spend for "Snack Bars" in 2025?', 'service income for "Snack Bars" in 2025?',
                                      'takings at "Snack Bars" for 2025', 'dépenses pour "Snack Bars" en 2025 ?'])
def test_a_fragment_with_its_own_figure_member_and_period_is_new(model, question):
    # 'service revenue for "Plumbing repiar" in 2025?' after the bookings of one partner in September kept the
    # partner: a word for a figure that is not the one on screen (no measure is called so), a member quoted and a
    # period of its own ask afresh, though the name is misspelt and so found as no member.
    assert read_turn(question, previous=_orders_at_riverside(), model=model).kind == "new"


@pytest.mark.parametrize("question", ['for "Snack Bars" in 2025?', 'takings for "Snack Bars"?', "takings in 2025?",
                                      'and the spend for "Snack Bars" in 2025?'])
def test_a_fragment_missing_one_of_them_follows_on(model, question):
    assert read_turn(question, previous=_orders_at_riverside(), model=model).kind != "new"


def _orders_by_category_at_riverside():
    return _plan(intent="breakdown", measures=["number_of_orders"], group_by=["category.name"],
                 filters=[{"field": "store.name", "op": "eq", "values": ["Riverside"]}], time=YEAR)


@pytest.mark.parametrize("question", ['Combien de commandes "Snack Bars" avons-nous reçues en janvier 2026 ?',
                                      'Quel est le montant des "Snack Bars" en mars 2026 ?'])
def test_a_whole_question_in_another_period_is_new_in_french_too(model, question):
    # 'Combien de réclamations "Fraud" avons-nous reçues en janvier 2026 ?' after one branch's complaints by
    # category in 2025 kept the branch: the French words name no measure of the data, so the code left it to the
    # AI, which kept it. A question word, a member quoted, a figure or a count, and a period other than the one on
    # screen ask afresh.
    assert read_turn(question, previous=_orders_by_category_at_riverside(), model=model).kind == "new"


@pytest.mark.parametrize("question", ['Combien de commandes "Snack Bars" en 2025 ?',
                                      'Combien de commandes en janvier 2026 ?'])
def test_the_same_period_or_no_member_is_not_settled_as_new(model, question):
    # In the period on screen, one bar of it may be what is meant; with no member, nothing says the scope changed.
    assert read_turn(question, previous=_orders_by_category_at_riverside(), model=model).kind != "new"


def _sales_at_riverside_in_march():
    return _plan(intent="value", measures=["net_amount"],
                 filters=[{"field": "store.name", "op": "eq", "values": ["Riverside"]}],
                 time={"window": {"kind": "between", "start": "2026-03-01", "end": "2026-03-31"}})


@pytest.mark.parametrize("question", ["show me the monthly sales for the last 6 months",
                                      "net sales by month over the last 6 months"])
def test_the_figure_on_screen_as_a_series_keeps_its_conditions(model, question):
    # "show me the monthly spend for the last 6 months" after "What did the lab spend in March 2026?" dropped the
    # lab: the figure on screen, month by month over another period, is that lab's.
    assert read_turn(question, previous=_sales_at_riverside_in_march(), model=model).kind == "refine"


@pytest.mark.parametrize("question", ["show me the monthly sales by region for the last 6 months",
                                      'monthly sales for "Lakeside" over the last 6 months',
                                      "monthly refunds for the last 6 months"])
def test_a_series_with_a_split_a_member_or_a_figure_of_its_own_is_not_kept(model, question):
    assert read_turn(question, previous=_sales_at_riverside_in_march(), model=model).kind != "refine"


def _orders_by_store_last_quarter():
    return _plan(intent="breakdown", measures=["number_of_orders"], group_by=["store.name"],
                 time={"window": {"kind": "previous", "unit": "quarter", "n": 1}})


@pytest.mark.parametrize("question", ["avg basket value over a rolling 3 months", "average refund amount in 2024",
                                      "mean order value for the last 6 months"])
def test_an_average_of_another_figure_for_a_period_of_its_own_is_new(model, question):
    # "avg po value over a rolling 3 months" after the purchase orders of last quarter split by buying team kept the
    # split: an average of a figure that is not the one on screen, for its own period, asks afresh.
    assert read_turn(question, previous=_orders_by_store_last_quarter(), model=model).kind == "new"


@pytest.mark.parametrize("question", ["basket value over a rolling 3 months", "the average over a rolling 3 months"])
def test_no_average_of_another_figure_follows_on(model, question):
    # A figure with no average or rate word, or an average of nothing named: the answer on screen goes on (its split
    # kept), as "spend over the last 3 months" after the spend by buying team does.
    assert read_turn(question, previous=_orders_by_store_last_quarter(), model=model).kind != "new"
