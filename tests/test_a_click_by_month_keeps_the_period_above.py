"""A member's months, opened from an answer, cover the period of that answer: all of it when it had none.

From a reader's test on Snowflake: revenue by doctor (every month there is), then a click on one doctor and
"by month", came back as "Booked Revenue from Oct 2025 to Sep 2026, for doctor ARI AURON, by month and doctor:
ARI AURON led in every month". Two things were wrong with it:

* the click asked for "the last 12 months", a period the reader never chose; the answer above had none, so its
  months are every month. A period too short to show months (a week, one month) still opens the last 12;
* the split by doctor was kept for a single doctor: one line, headed by the doctor who "led" it. A series split by
  a field it is kept to one member of is now that member's series, and one member's line is never said to lead.
  Asking "by doctor" after one doctor's answer is every doctor again, the doctor's name being the doctor.

The invented retail warehouse; the real service and planner.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.ir import Filter, Plan, TimeSpec
from core2.plan.planner import one_member_series
from core2.plan.thread import kept_from_above
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
APRIL = {"kind": "between", "start": "2026-04-01", "end": "2026-04-30"}
BY_STORE = {"kind": "query", "intent": "breakdown", "measures": ["net_amount"], "group_by": ["store"]}
TOTAL = {"kind": "query", "intent": "value", "measures": ["net_amount"]}


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return model, DuckDBWarehouse(built.con)


class AI:
    def __init__(self, *plans):
        self.plans = list(plans)

    def __call__(self, stable: str, tail: str) -> str:
        return json.dumps(self.plans.pop(0))


def _conversation(retail, *plans):
    model, warehouse = retail
    services = Services(model=model, warehouse=warehouse, complete=AI(*plans), index=MemberIndex(), today=TODAY)
    session = Session()
    return lambda question: answer_question(question, services, session)


# ── what the click asks ─────────────────────────────────────────────────────


def test_a_member_of_an_answer_for_all_time_opens_its_every_month(retail):
    drill = _conversation(retail, BY_STORE)("Net amount by store")["chart"]["drill"]
    assert drill["items"][0]["question"] == 'Net amount by month for "{member}"'
    assert drill["items"][0]["question_fr"] == 'Net amount par mois pour "{member}"'


def test_a_member_of_a_long_period_opens_its_months_in_that_period(retail):
    plan = {**BY_STORE, "time": {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}}}
    drill = _conversation(retail, plan)("Net amount by store in H1 2026")["chart"]["drill"]
    assert drill["items"][0]["question"] == 'Net amount by month for "{member}" from Jan 2026 to Jun 2026'


def test_a_member_of_one_month_opens_its_last_12_months(retail):
    """One month by month is a single point."""
    drill = _conversation(retail, {**BY_STORE, "time": {"window": APRIL}})("Net amount by store in April 2026")
    assert drill["chart"]["drill"]["items"][0]["question"] == 'Net amount by month for "{member}" for the last 12 months'


def test_a_total_for_all_time_offers_its_every_month(retail):
    offered = _conversation(retail, TOTAL)("What is our net amount?")["follow_up_suggestions"]
    trend = [s for s in offered if "by month" in json.dumps(s)]
    assert trend and all("12" not in json.dumps(s) for s in trend), offered


# ── what the click answers ──────────────────────────────────────────────────


def test_a_click_on_a_store_is_that_stores_line_over_every_month(retail):
    _model, warehouse = retail
    store = warehouse.query('SELECT MIN("store_name") FROM "stores"').rows[0][0]
    # The AI kept the split by store from the answer above, and narrowed it to the store clicked.
    clicked = {"kind": "query", "intent": "trend", "measures": ["net_amount"], "group_by": ["store"],
               "filters": [{"field": "store.name", "op": "eq", "values": [store]}],
               "time": {"grain": "month"}, "follow_up": "refine"}
    ask = _conversation(retail, BY_STORE, clicked)
    ask("Net amount by store")
    payload = ask(f'Net amount by month for "{store}"')
    assert not payload["plan"].get("group_by")
    assert payload["plan"].get("time", {}).get("window", {}).get("kind", "all") == "all"
    headline = payload["answer"]["headline"]
    assert "led" not in headline and "by month and" not in headline, headline
    assert payload["chart"]["chart_type"] == "line"


def test_one_members_line_is_never_said_to_lead(retail):
    """A store's months by its region (a store is in one region): a line with no one to lead it."""
    _model, warehouse = retail
    store = warehouse.query('SELECT MIN("store_name") FROM "stores"').rows[0][0]
    plan = {"kind": "query", "intent": "trend", "measures": ["net_amount"], "group_by": ["region.name"],
            "filters": [{"field": "store.name", "op": "eq", "values": [store]}],
            "time": {"grain": "month", "window": {"kind": "between", "start": "2026-01-01", "end": "2026-05-31"}}}
    payload = _conversation(retail, plan)(f'Net amount by month and region for "{store}"')
    assert payload["plan"]["group_by"] == ["region.name"]
    headline = payload["answer"]["headline"]
    assert "led" not in headline and headline.count("for region ") == 1, headline


# ── the rules, in code ──────────────────────────────────────────────────────


NORTH = Filter(field="region.name", op="eq", values=["North"])
MONTHLY = TimeSpec(grain="month")


def test_a_split_kept_to_one_member_is_dropped_from_a_series(retail):
    model, _ = retail
    plan = Plan(intent="trend", measures=["net_amount"], group_by=["region"], filters=[NORTH], time=MONTHLY)
    assert one_member_series(plan, model).group_by == []
    breakdown = plan.model_copy(update={"intent": "breakdown", "group_by": ["region.name", "store"]})
    assert one_member_series(breakdown, model).group_by == ["store"]


@pytest.mark.parametrize("change", [
    {"intent": "rank"},                                                        # a ranking of members keeps them
    {"intent": "share"},
    {"filters": [Filter(field="region.name", op="in", values=["North", "South"])]},   # two members: two lines
    {"filters": [Filter(field="region.name", op="ne", values=["North"])]},
    {"time": TimeSpec()},                                                      # no series: not this rule's
])
def test_other_splits_stay(retail, change):
    model, _ = retail
    plan = Plan(intent="trend", measures=["net_amount"], group_by=["region.name"], filters=[NORTH], time=MONTHLY)
    plan = plan.model_copy(update=change)
    assert one_member_series(plan, model).group_by == ["region.name"]


def test_a_split_by_the_member_lets_go_of_its_name(retail):
    """"By region" after North's answer is every region, whichever name of the region the split uses."""
    model, _ = retail
    before = Plan(intent="trend", measures=["net_amount"], filters=[NORTH], time=MONTHLY)
    for field in ("region", "region.name"):
        ai = Plan(intent="trend", measures=["net_amount"], group_by=[field], time=MONTHLY, follow_up="refine")
        assert NORTH not in kept_from_above(ai, model, "and by region?", before, lambda _p: True).filters
