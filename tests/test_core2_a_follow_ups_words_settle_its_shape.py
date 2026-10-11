"""What a follow-up's own words say about the shape of the answer it changes, whatever the AI kept.

Found asking real conversations of a real model (gpt-4o):

* "And by referral source?" after patients by gender came back by gender and referral source: it splits by that
  field instead;
* "Break that down by booking channel" after fills each month came back month by month and channel: it splits
  the whole period by channel, unless it asks for each month;
* "What was the cost for those?" after the top 10 formulas by gross amount ranked the formulas again by cost:
  it keeps those ten, in that order, and adds the cost;
* "Is that account mostly debits or credits?" after a monthly trend set the two side by side month by month:
  it gives the two totals;
* "Which customer types got the most discount?" after the discount of 2025 was read as a new question and
  answered for all time: a measure on screen ranked another way, for no period of its own, keeps 2025.

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


def test_and_by_a_field_splits_by_it_instead(model):
    before = _plan(intent="breakdown", measures=["net_amount"], group_by=["region.name"], time=YEAR)
    ai = _plan(intent="breakdown", measures=["net_amount"], group_by=["region.name", "store.name"], time=YEAR,
               follow_up="refine")
    assert settle(ai, model, "And by store?", before).group_by == ["store.name"]
    # Asked for both, both are kept.
    assert settle(ai, model, "Split that by region and store", before).group_by == ["region.name", "store.name"]


@pytest.mark.parametrize("question, grain, intent", [
    ("Break that down by store", None, "breakdown"),
    ("Break that down by store for each month", "month", "trend"),
    ("Split it by store, monthly", "month", "trend"),
])
def test_breaking_down_a_trend_splits_the_period_unless_it_asks_for_each_month(model, question, grain, intent):
    before = _plan(intent="trend", measures=["net_amount"], time={**YEAR, "grain": "month"})
    ai = _plan(intent="trend", measures=["net_amount"], group_by=["store.name"], time={**YEAR, "grain": "month"},
               follow_up="refine")
    settled = settle(ai, model, question, before)
    assert (settled.time.grain, settled.intent) == (grain, intent)


def test_another_figure_for_those_keeps_the_members_on_screen(model):
    before = _plan(intent="rank", measures=["net_amount"], group_by=["product.name"], time=YEAR, limit=10,
                   sort=[{"by": "net_amount", "desc": True}])
    ai = _plan(intent="rank", measures=["cost_amount"], group_by=["product.name"], time=YEAR, limit=10,
               sort=[{"by": "cost_amount", "desc": True}], follow_up="refine")
    settled = settle(ai, model, "What was the cost for those?", before)
    assert settled.measures == ["net_amount", "cost_amount"]
    assert (settled.limit, settled.sort[0].by, settled.group_by) == (10, "net_amount", ["product.name"])
    # A new ranking by cost is asked in its own words, and stays one.
    assert settle(ai, model, "Which products cost the most?", before).measures == ["cost_amount"]


def test_mostly_one_or_the_other_sets_the_two_totals_side_by_side(model):
    before = _plan(intent="trend", measures=["net_amount"], time={**YEAR, "grain": "month"})
    ai = _plan(intent="trend", measures=["net_amount", "cost_amount"], time={**YEAR, "grain": "month"},
               follow_up="refine")
    settled = settle(ai, model, "Is it mostly sales or cost?", before)
    assert (settled.time.grain, settled.intent) == (None, "value")
    assert settle(ai, model, "Is it mostly sales or cost, month by month?", before).time.grain == "month"


def test_a_measure_on_screen_ranked_another_way_keeps_the_period(model):
    before = _plan(intent="value", measures=["discount_amount"], time=YEAR)
    reading = read_turn("Which stores got the most discount?", previous=before, model=model)
    assert reading.kind == "refine"
    # Another measure, or a period of its own: a new question.
    assert read_turn("Which stores had the most returns?", previous=before, model=model).kind == "new"
    assert read_turn("Which stores got the most discount in 2024?", previous=before, model=model).kind == "new"
