"""A "why" explains the two periods the reader means: named in its words, or on screen, never the AI's guess.

Found asking real conversations of a real model (gpt-4o):

* "Why did it change from the first to the second half of fiscal year 2026?" kept the months of the trend on
  screen, and was answered as February against March: a year's two halves are worked out in code, and a series
  is compared month by month only when nothing names the two periods;
* "Why did it grow?" after "Which month grew the most?" showed May against May a year before, and was answered
  as May against June: the one period on screen is explained, against the period it was compared with;
* "Pourquoi cette différence ?" after the same figure on 17 August, then on 27 May, was answered as 27 May against
  the day before: the two dates on screen are compared;
* "Why did it grow?" of a figure that fell: the answer says it fell before the reasons.

Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.answer.drivers import _against_the_words, base_plan
from core2.plan.ir import Plan
from core2.plan.planner import Turn, _halves, why_periods
from core2.plan.values import MemberIndex
from core2.resolve.resolver import Context
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
D = dt.date.fromisoformat


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


@pytest.fixture
def april(retail):
    """The retail model with a fiscal year from April, named by the year it ends in."""
    model = retail[1].model_copy(deep=True)
    model.settings.fiscal_year_start_month = 4
    return model


def _why(**time) -> Plan:
    return Plan.model_validate({"kind": "query", "intent": "drivers", "measures": ["net_amount"], "time": time,
                                "follow_up": "refine"})


@pytest.mark.parametrize("question, second, first", [
    ("Why did it change from the first to the second half of fiscal year 2026?",
     ("2025-10-01", "2026-03-31"), ("2025-04-01", "2025-09-30")),
    ("Why did FY26 change from the first half to the second half?", ("2025-10-01", "2026-03-31"),
     ("2025-04-01", "2025-09-30")),
    ("What drove the change from H1 to H2 2025?", ("2025-07-01", "2025-12-31"), ("2025-01-01", "2025-06-30")),
])
def test_a_years_two_halves_are_worked_out(april, question, second, first):
    window, prior = _halves(question, april)
    assert (window.start, window.end) == (D(second[0]), D(second[1]))
    assert (prior.start, prior.end) == (D(first[0]), D(first[1]))
    planned = why_periods(_why(grain="month", window={"kind": "between", "start": "2025-04-01",
                                                      "end": "2026-03-31"}), april, question, [])
    assert planned.time.grain is None and planned.time.compare.window == prior and planned.time.window == window


def test_one_half_alone_is_no_comparison(april):
    assert _halves("Sales in the first half of 2025", april) is None


@pytest.mark.parametrize("compare, prior", [
    ({"kind": "same_period_last_year"}, None),
    ({"kind": "window", "window": {"kind": "between", "start": "2025-01-01", "end": "2025-06-30"}},
     ("2025-05-01", "2025-05-31")),
])
def test_the_one_period_on_screen_is_explained_against_its_match(retail, compare, prior):
    grew = Plan.model_validate({"kind": "query", "intent": "trend", "measures": ["net_amount"], "time": {
        "grain": "month", "window": {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"},
        "compare": compare}, "sort": [{"by": "change", "desc": True}], "limit": 1})
    ai = _why(grain="month", window={"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}, compare=compare)
    planned = why_periods(ai, retail[1], "Why did it grow?", [Turn("Which month grew the most?", grew,
                                                                   periods=["2026-05-01"])])
    assert (planned.time.window.start, planned.time.window.end, planned.time.grain) == (
        D("2026-05-01"), D("2026-05-31"), None)
    if prior is None:
        assert planned.time.compare.kind == "same_period_last_year"
    else:
        assert (planned.time.compare.window.start, planned.time.compare.window.end) == (D(prior[0]), D(prior[1]))
    # Several periods on screen: the AI's periods stand.
    assert why_periods(ai, retail[1], "Why did it grow?", [Turn("Monthly", grew, periods=["2026-04-01",
                                                                                        "2026-05-01"])]) is ai


def test_the_two_dates_on_screen_are_compared(retail):
    def on(day: str) -> Plan:
        return Plan.model_validate({"kind": "query", "intent": "breakdown", "measures": ["net_amount"],
                                    "group_by": ["store.name"],
                                    "time": {"window": {"kind": "between", "start": day, "end": day}}})

    history = [Turn("Net amount by store on 17 August?", on("2025-08-17")), Turn("And on 27 May?", on("2025-05-27"))]
    ai = _why(window={"kind": "between", "start": "2025-05-27", "end": "2025-05-27"})
    planned = why_periods(ai, retail[1], "Pourquoi cette différence ?", history)
    assert planned.time.window.start == D("2025-08-17") and planned.time.compare.window.start == D("2025-05-27")
    # A period of its own: the question's words stand.
    assert why_periods(ai, retail[1], "Why did it change in March 2025?", history) is ai


def test_two_periods_named_are_compared_whole_not_by_their_last_months(retail):
    plan = _why(grain="month", window={"kind": "between", "start": "2025-10-01", "end": "2026-03-31"},
                compare={"kind": "window", "window": {"kind": "between", "start": "2025-04-01", "end": "2025-09-30"}})
    base, notes = base_plan(plan, retail[1], Context(today=TODAY))
    assert base.time.window == plan.time.window and base.time.compare == plan.time.compare
    assert not any("last complete" in n for n in notes)
    # A series with nothing to compare it with: its last month against the one before, as before.
    series, notes = base_plan(_why(grain="month", window={"kind": "between", "start": "2025-10-01",
                                                          "end": "2026-03-31"}), retail[1], Context(today=TODAY))
    assert any("last complete month" in n for n in notes)


@pytest.mark.parametrize("question, change, said", [
    ("Why did it grow?", -5.0, "It did not grow, it fell. "),
    ("Why did sales drop in May?", 5.0, "It did not fall, it rose. "),
    ("Why did it grow?", 5.0, ""),
    ("Why did it change?", -5.0, ""),
])
def test_a_figure_that_moved_the_other_way_says_so_first(question, change, said):
    assert _against_the_words(question, change) == said


def test_why_after_the_month_that_grew_most_explains_that_month(retail):
    built, model = retail
    window = {"kind": "between", "start": "2026-01-01", "end": "2026-05-31"}
    grew = {"kind": "query", "intent": "trend", "measures": ["net_amount"],
            "time": {"grain": "month", "window": window, "compare": {"kind": "same_period_last_year"}},
            "sort": [{"by": "change", "desc": True}], "limit": 1}
    why = {"kind": "query", "intent": "drivers", "measures": ["net_amount"], "follow_up": "refine",
           "time": {"grain": "month", "window": window, "compare": {"kind": "same_period_last_year"}}}
    replies = [grew, why]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con),
                        complete=lambda s, t: json.dumps(replies.pop(0)), index=MemberIndex(), today=TODAY)
    session = Session()
    first = answer_question("Which month grew the most against last year, January to May 2026?", services, session)
    month = next(v for k, v in first["export_rows"][0].items() if k.startswith("period"))
    start = D(str(month)[:10])
    payload = answer_question("Why did it grow?", services, session)
    words = f"{start:%b %Y}" if f"{start:%b %Y}" in payload["answer"]["headline"] else f"{start:%B %Y}"
    assert words in payload["answer"]["headline"], payload["answer"]["headline"]
    last_year = f"{start.replace(year=start.year - 1):%b %Y}"
    assert last_year in payload["answer"]["headline"] or \
        f"{start.replace(year=start.year - 1):%B %Y}" in payload["answer"]["headline"]
