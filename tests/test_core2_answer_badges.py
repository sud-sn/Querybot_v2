"""Each new-core answer says what it counted in short labels above its sentence.

The redesigned answer card opens with badges: the period, and the date the
rows were counted by. They used to be readable only inside the sentence
("Net amount from Jan 2026 to Jun 2026: ...") or in the notes at the bottom.
The badges come from the answer's own plan (its window, what it is compared
with, the date each part was counted by), never from parsing its sentence.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
H1 = {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}
APRIL = {"kind": "between", "start": "2026-04-01", "end": "2026-04-30"}


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return built.con, model


def _badges(retail, plan: dict) -> list[dict]:
    con, model = retail
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    return answer_question("q", services, Session())["answer"]["badges"]


def test_one_number_names_its_month_and_its_date(retail):
    plan = {"intent": "value", "measures": ["net_amount"], "time": {"window": APRIL}}
    assert _badges(retail, plan) == [{"kind": "period", "text": "April 2026"},
                                     {"kind": "date", "text": "by order date"}]


def test_a_span_within_one_year_is_written_once(retail):
    plan = {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["store"], "time": {"window": H1}}
    assert _badges(retail, plan)[0] == {"kind": "period", "text": "Jan–Jun 2026"}


def test_a_span_across_years_names_both(retail):
    plan = {"intent": "trend", "measures": ["net_amount"],
            "time": {"grain": "month", "window": {"kind": "between", "start": "2025-03-01", "end": "2026-02-28"}}}
    assert _badges(retail, plan)[0] == {"kind": "period", "text": "Mar 2025 – Feb 2026"}


def test_a_comparison_names_both_periods(retail):
    plan = {"intent": "compare", "measures": ["net_amount"], "group_by": ["store"],
            "time": {"window": APRIL, "compare": {"kind": "previous_period"}}}
    assert _badges(retail, plan)[0] == {"kind": "period", "text": "April 2026 vs March 2026"}


def test_the_date_badge_is_the_date_the_rows_were_counted_by(retail):
    plan = {"intent": "count", "measures": ["number_of_customers"],
            "time": {"date": "return_date", "window": APRIL}}
    assert {"kind": "date", "text": "by return date"} in _badges(retail, plan)


def test_an_answer_with_no_period_has_no_badges(retail):
    assert _badges(retail, {"intent": "list", "group_by": ["store"]}) == []


# ── a series broken down by a member says who led, not how many rows ────────


def _headline(retail, plan: dict) -> str:
    con, model = retail
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    return answer_question("q", services, Session())["answer"]["headline"]


def test_a_series_by_member_names_who_led_first_and_last(retail):
    said = _headline(retail, {"intent": "trend", "measures": ["net_amount"], "group_by": ["region.name"],
                              "time": {"grain": "month", "window": H1}})
    assert not said.endswith("rows."), said
    assert said.startswith("Net amount from Jan 2026 to Jun 2026, by month and region: ")
    assert " led in Jan 2026 ($" in said and " in May 2026 ($" in said, "June is partly covered: May is the last whole month"


def test_a_series_by_member_that_one_member_always_led_says_so(retail):
    said = _headline(retail, {"intent": "trend", "measures": ["margin_percent"], "group_by": ["region.name"],
                              "time": {"grain": "quarter",
                                       "window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}})
    assert "led in every quarter from Q1 2025 to Q4 2025, among the 5 regions" in said, said
