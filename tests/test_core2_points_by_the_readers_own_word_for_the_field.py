"""A row of the answer on screen is found by its place, whatever the reader calls its field.

"How many different items do we hold from the first supplier on that list?" followed a ranking grouped by a
field called party: the words named no field the answer had, so nothing was found, and the AI wrote a stand-in
("first_supplier") on the right field. When the AI puts a stand-in on the field the answer on screen is grouped
by, the question's one place word ("first", "top", "last") picks that row. When the AI asks which member is meant,
the members of the answer on screen are offered, one click each.

Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.ir import Plan
from core2.plan.planner import pointed_in
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
SCREEN = [("store.name", "Airport", 900.0), ("store.name", "Harbour", 700.0), ("store.name", "Mall", 500.0)]


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def _plan(field: str = "store.name", values=("first_branch",)) -> Plan:
    return Plan.model_validate({"kind": "query", "measures": ["net_amount"],
                                "filters": [{"field": field, "op": "eq", "values": list(values)}]})


@pytest.mark.parametrize("question, kept", [
    ("How much did the first branch on that list sell?", ["Airport"]),
    ("And the last branch on that list?", ["Mall"]),
    ("What about the 2nd branch?", ["Harbour"]),
    ("Compare the first branch with the last branch", ["first_branch"]),       # two places: which one?
    ("How much did that branch sell?", ["first_branch"]),                       # no place word: asked about
])
def test_a_stand_in_on_the_field_on_screen_takes_the_row_at_the_place_named(retail, question, kept):
    assert pointed_in(_plan(), retail[1], [], question, SCREEN).filters[0].values == kept


def test_a_stand_in_on_another_field_is_left_to_be_asked_about(retail):
    plan = _plan("region.name", ("first_region",))
    assert pointed_in(plan, retail[1], [], "the first region on that list", SCREEN).filters[0].values == [
        "first_region"]
    # And a real name is never replaced, whatever is on screen.
    assert pointed_in(_plan(values=("Mall",)), retail[1], [], "the first one", SCREEN).filters[0].values == ["Mall"]


def test_asked_which_member_is_meant_the_members_on_screen_are_offered(retail):
    built, model = retail
    rank = {"kind": "query", "intent": "rank", "measures": ["net_amount"], "group_by": ["store.name"],
            "sort": [{"by": "net_amount", "desc": True}], "limit": 3,
            "time": {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}}}
    ask = {"kind": "clarify", "clarify": {"about": "member", "question": "Which store do you mean?", "options": []}}
    replies = [rank, ask, ask]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con),
                        complete=lambda s, t: json.dumps(replies.pop(0)), index=MemberIndex(), today=TODAY)
    session = Session()
    first = answer_question("Top 3 stores by net amount in Q1 2026", services, session)
    names = [r["store_name"] for r in first["export_rows"]]
    asked = answer_question("How did the one I care about do?", services, session)
    assert asked["clarify"]["options"] == names
    assert [c["question"] for c in asked["follow_up_suggestions"]] == names
