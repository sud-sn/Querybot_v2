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
