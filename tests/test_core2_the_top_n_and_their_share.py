"""The top N and what share of the total they hold, in one answer; and "those" stay the same members.

Found asking real conversations of a real model (gpt-4o):

* "Which 5 items brought the most gross profit in Q2 2026, and what share of the total is that?" was planned as a
  share of members with no condition ("per": {"of": null, "summary": "share"}) and refused. It is each member's
  share of the total, and the answer says what the five hold together;
* "How did those items do in Q1 2026?" ranked the top 5 of Q1, five other items; "Show their monthly quantity sold
  in 2026" then ranked month-and-item rows and kept the gross profit of the answer before. The members on screen
  are named when a follow-up asks about them in another period or as a series, and stay named for the next
  "their"; the figure it asks for is the one shown.

Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.planner import parse
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
Q1 = {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}
Q2 = {"kind": "between", "start": "2026-04-01", "end": "2026-06-30"}
TOP5 = {"kind": "query", "intent": "rank", "measures": ["net_amount"], "group_by": ["store"],
        "sort": [{"by": "net_amount", "desc": True}], "limit": 5, "time": {"window": Q1}}


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def _ask(retail, replies: list, *questions: str) -> list[dict]:
    built, model = retail
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con),
                        complete=lambda s, t: json.dumps(replies.pop(0)), index=MemberIndex(), today=TODAY)
    session = Session()
    return [answer_question(q, services, session) for q in questions]


def _by_store(built, start: str, end: str) -> dict[str, float]:
    return {k: float(v) for k, v in built.con.execute(
        "SELECT s.store_name, SUM(o.net_amount) FROM order_lines o JOIN stores s ON o.store_id = s.store_id "
        "JOIN calendar d ON o.order_date_key = d.date_key "
        f"WHERE d.full_date BETWEEN DATE '{start}' AND DATE '{end}' GROUP BY 1").fetchall()}


def test_a_share_of_members_with_no_condition_is_each_members_share():
    plan = parse(json.dumps({**TOP5, "per": {"of": None, "summary": "share"}}))
    assert plan.intent == "share" and plan.per is None


def test_the_top_five_say_what_they_hold_together(retail):
    built, model = retail
    share = {**TOP5, "intent": "share"}
    payload = _ask(retail, [share], "Which 5 stores sold the most in Q1 2026, and what share of the total is that?")[0]
    q1 = _by_store(built, "2026-01-01", "2026-03-31")
    top = sorted(q1.values(), reverse=True)[:5]
    headline = payload["answer"]["headline"]
    assert f"the 5 together: ${sum(top):,.2f}, {sum(top) / sum(q1.values()):.0%} of the total" in headline, headline


def test_those_stores_in_another_period_are_the_same_stores(retail):
    built, model = retail
    again = {"kind": "query", "intent": "rank", "measures": ["net_amount"], "group_by": ["store"],
             "sort": [{"by": "net_amount", "desc": True}], "limit": 5, "time": {"window": Q2}, "follow_up": "refine"}
    monthly = {"kind": "query", "intent": "trend", "measures": ["quantity"], "group_by": ["store"],
               "sort": [{"by": "net_amount", "desc": True}], "limit": 5, "follow_up": "refine",
               "time": {"grain": "month", "window": {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}}}
    first, second, third = _ask(retail, [TOP5, again, monthly], "Top 5 stores by net amount in Q1 2026",
                                "How did those stores do in Q2 2026?", "Show their monthly quantity in 2026")
    q1 = _by_store(built, "2026-01-01", "2026-03-31")
    q2 = _by_store(built, "2026-04-01", "2026-06-30")
    five = sorted(q1, key=q1.get, reverse=True)[:5]
    assert [r["store_name"] for r in first["export_rows"]] == five
    assert {r["store_name"]: r["net_amount"] for r in second["export_rows"]} == pytest.approx({s: q2[s] for s in five})
    # Named once, still those five; the figure asked for is the one shown, month by month.
    assert {r["store_name"] for r in third["export_rows"]} == set(five)
    assert "quantity" in third["export_rows"][0] and "net_amount" not in third["export_rows"][0]
    assert len(third["export_rows"]) > 5
