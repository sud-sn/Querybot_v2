"""The date an answer counts by is put right from the answer, kept for the conversation, and fixed for everyone.

Every figure over time is counted by one date of its rows, the one Learn picked: net sales by order date, where a
reader may mean the ship date. The answer's date badge lists the rows' other dates, each a click that counts the
same figure by it again with no AI; the pick holds for the rest of the conversation (a chip above the box, taken off
by a click), and a question that names its own date still wins and is said. "Counted by ship date" in a question,
and "from now on count by ship date", set it in code whatever the AI planned. An answer counted by a date other
than its metric's own offers to make it the metric's date: an admin's click changes it, a reader's goes to the
admin as a suggestion. A date picked when the answer asked which one is kept the same way.

The synthetic retail warehouse: order lines with an order date (Learn's), a ship date, a delivered date and a load
stamp; numbers checked against SQL written by hand.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.model import requests as R
from core2.plan.ir import Plan
from core2.plan.planner import settle
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
Q1 = {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}
SALES_Q1 = {"intent": "value", "measures": ["net_amount"], "time": {"window": Q1}}


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return built.con, model


class AI:
    """The AI's plans, one per question it is asked; asked once more than it was given, it fails the test."""

    def __init__(self, *plans: dict):
        self.plans = [json.dumps({"kind": "query", **p}) for p in plans]

    def __call__(self, stable: str, tail: str) -> str:
        assert self.plans, "the AI was asked for a plan where code should have answered"
        return self.plans.pop(0)


def _chat(retail, *plans: dict):
    con, model = retail
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=AI(*plans), index=MemberIndex(),
                        today=TODAY)
    session = Session()
    return (lambda question: answer_question(question, services, session)), session


def _sales(con, date_key: str, start: str = "2026-01-01", end: str = "2026-03-31") -> float:
    """Net sales of the order lines whose ``date_key`` falls in the period, by hand."""
    return round(float(con.execute(
        f"SELECT SUM(o.net_amount) FROM order_lines o JOIN calendar c ON c.date_key = o.{date_key} "
        f"WHERE c.full_date BETWEEN DATE '{start}' AND DATE '{end}'").fetchone()[0]), 2)


def test_the_answer_lists_the_other_dates_of_its_rows(retail):
    ask, _ = _chat(retail, SALES_Q1)
    payload = ask("net sales in Q1 2026")
    dates = payload["dates"]
    assert dates["used"]["name"] == "Order date"
    assert sorted((o["name"], o["words"], o["words_fr"]) for o in dates["others"]) == [
        ("Delivered date", "By delivered date instead", "Selon delivered date à la place"),
        ("Ship date", "By ship date instead", "Selon ship date à la place")]     # never the load stamp
    assert "fix" not in dates               # counted by the metric's own date: nothing to put right


def test_a_figure_over_no_period_has_no_date_to_choose(retail):
    ask, _ = _chat(retail, {"intent": "value", "measures": ["net_amount"]})
    assert "dates" not in ask("net sales of all time")


def test_another_date_is_one_click_with_no_ai_and_holds_for_the_conversation(retail):
    con, _ = retail
    by_store = {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["store.name"],
                "time": {"window": {"kind": "between", "start": "2026-04-01", "end": "2026-05-31"}}}
    ask, session = _chat(retail, SALES_Q1, by_store)
    ask("net sales in Q1 2026")
    shipped = ask("By ship date instead")
    assert shipped["kpi"]["value"] == pytest.approx(_sales(con, "ship_date_key"))
    assert _sales(con, "ship_date_key") != _sales(con, "order_date_key")
    kept = shipped["standing"]["kept"]
    assert [(s["words"], s["words_fr"]) for s in kept] == [("by ship date", "selon ship date")]
    assert shipped["dates"]["used"]["name"] == "Ship date"
    # The next question, planned by the AI with no date, is counted by the ship date too, and says so.
    stores = ask("and net sales by store for April and May")
    total = sum(r["net_amount"] for r in stores["data"]["rows"])
    assert total == pytest.approx(_sales(con, "ship_date_key", "2026-04-01", "2026-05-31"), abs=0.05)
    assert stores["standing"]["applied"] == ["by ship date"]


def test_the_rows_own_date_again_lets_the_kept_one_go(retail):
    con, _ = retail
    ask, session = _chat(retail, SALES_Q1)
    ask("net sales in Q1 2026")
    ask("By ship date instead")
    back = ask("By order date instead")
    assert back["kpi"]["value"] == pytest.approx(_sales(con, "order_date_key"))
    assert not [s for s in session.standing if s.date]


def test_a_question_that_names_its_own_date_wins_and_is_said(retail):
    con, _ = retail
    ask, _ = _chat(retail, SALES_Q1, SALES_Q1)
    ask("net sales in Q1 2026")
    ask("By ship date instead")
    delivered = ask("net sales in Q1 2026, counted by delivered date")
    assert delivered["kpi"]["value"] == pytest.approx(_sales(con, "delivered_date_key"))
    assert delivered["standing"]["aside"] == ["by ship date (your question names its own date)"]


@pytest.mark.parametrize("question", ["Net sales by month in 2025, counted by ship date",
                                      "net sales in 2025 using the ship date", "net sales in 2025 based on ship date"])
def test_the_date_the_question_names_is_used_whatever_the_ai_planned(retail, question):
    _, model = retail
    planned = Plan.model_validate({"kind": "query", "intent": "value", "measures": ["net_amount"],
                                   "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}})
    assert settle(planned, model, question, None).time.date == "ship_date"


@pytest.mark.parametrize("question", ["net sales by store in 2025", "orders shipped by date in 2025",
                                      "net sales in 2025 by ship mode"])
def test_no_date_named_leaves_the_plan_as_it_is(retail, question):
    _, model = retail
    planned = Plan.model_validate({"kind": "query", "intent": "value", "measures": ["net_amount"],
                                   "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}})
    assert settle(planned, model, question, None).time.date is None


def test_count_by_a_date_from_now_on_is_a_standing_condition(retail):
    con, _ = retail
    ask, session = _chat(retail, SALES_Q1)
    said = ask("From now on count by ship date")
    assert [s["words"] for s in said["standing"]["kept"]] == ["by ship date"]
    assert ask("net sales in Q1 2026")["kpi"]["value"] == pytest.approx(_sales(con, "ship_date_key"))


def test_an_answer_counted_by_another_date_offers_to_make_it_the_metrics_own(retail):
    _, model = retail
    ask, _ = _chat(retail, SALES_Q1)
    ask("net sales in Q1 2026")
    fix = ask("By ship date instead")["dates"]["fix"]
    assert (fix["measure"], fix["date_name"], fix["was"]) == ("Net amount", "Ship date", "Order date")
    assert fix["words_admin"] == "Make ship date the date of Net amount for everyone"
    assert fix["words_reader"] == "Suggest ship date for Net amount to my admin"
    # What the click files: the change the Requests page accepts (an admin's at once).
    name, changes = R.proposal(model, kind=fix["target_kind"], key=fix["target_key"], date=fix["date"])
    assert name == "Net amount"
    assert [(c["field"], c["now"], c["shown"]) for c in changes] == [("default_date", "Order date", "Ship date")]


def test_a_date_picked_when_asked_is_kept_too(retail):
    con, _ = retail
    count = {"intent": "count", "measures": ["number_of_customers"], "time": {"window": Q1}}
    ask, session = _chat(retail, count, {**count, "time": {"window": {"kind": "between", "start": "2026-04-01",
                                                                      "end": "2026-05-31"}}})
    asked = ask("how many customers in Q1 2026?")
    assert "Return date" in asked["clarify"]["options"]
    ask("return date")
    # A question of its own, later in the conversation (a follow-up would carry the date anyway): not asked again.
    again = ask("New question: how many customers in April and May 2026?")
    assert not again.get("clarify"), "the date picked above is not asked again"
    assert [s["words"] for s in again["standing"]["kept"]] == ["by return date"]
    want = con.execute("SELECT COUNT(DISTINCT r.customer_id) FROM returns r JOIN calendar c ON c.date_key = "
                       "r.return_date_key WHERE c.full_date BETWEEN DATE '2026-04-01' AND DATE '2026-05-31'"
                       ).fetchone()[0]
    assert again["kpi"]["value"] == want
