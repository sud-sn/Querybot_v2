"""A period said in a reader's words has one exact meaning, worked out in code, and the answer shows its dates.

* "The last 2 months" are the two complete months before this one.
* "The last 2 months from today", "rolling 2 months", "the past 60 days" run up to today, today included; days
  always do ("the last 30 days").
* "This month so far", "month to date", "MTD" run from the month's first day to today.
* "The last 12 months", "trailing 12 months", "trailing year", "TTM" are the twelve complete months before this one.
* "The same time last year" compares with the same dates a year before.

The planner (an AI) writes the window; code reads the reader's words again and puts right a window it read
differently. Invented data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.ir import Compare, Plan, TimeSpec, Window
from core2.plan.planner import period_meant
from core2.plan.values import MemberIndex
from core2.resolve.time import Range, resolve_window
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
D = dt.date


def _plan(window: dict, compare: dict | None = None) -> Plan:
    return Plan(kind="query", measures=["net_amount"],
                time=TimeSpec(window=Window(**window), compare=Compare(**compare) if compare else None))


# ── the dates each meaning gives ───────────────────────────────────────────


@pytest.mark.parametrize("window, start, end", [
    ({"kind": "rolling", "unit": "month", "n": 2}, D(2026, 4, 16), D(2026, 6, 16)),
    ({"kind": "rolling", "unit": "day", "n": 60}, D(2026, 4, 17), D(2026, 6, 16)),
    ({"kind": "rolling", "unit": "week", "n": 4}, D(2026, 5, 19), D(2026, 6, 16)),
    ({"kind": "rolling", "unit": "year", "n": 1}, D(2025, 6, 16), D(2026, 6, 16)),
    ({"kind": "last", "unit": "month", "n": 2}, D(2026, 4, 1), D(2026, 6, 1)),
    ({"kind": "last", "unit": "month", "n": 12}, D(2025, 6, 1), D(2026, 6, 1)),
    ({"kind": "to_date", "unit": "month"}, D(2026, 6, 1), D(2026, 6, 16)),
])
def test_each_meaning_has_its_dates(window, start, end):
    got = resolve_window(Window(**window), today=TODAY, last_data=TODAY)
    assert (got.start, got.end) == (start, end)


def test_a_rolling_period_counts_back_from_where_stale_data_ends():
    got = resolve_window(Window(kind="rolling", unit="month", n=2), today=TODAY, last_data=D(2026, 5, 31))
    assert (got.start, got.end) == (D(2026, 4, 1), D(2026, 6, 1))
    assert got.notes and "31 May 2026" in got.notes[0]


# ── the reader's words, read again in code ─────────────────────────────────


@pytest.mark.parametrize("question, written, meant", [
    # Rolling, up to today.
    ("net sales in the last 2 months from today", {"kind": "last", "unit": "month", "n": 2},
     {"kind": "rolling", "unit": "month", "n": 2}),
    ("net sales for the last two months up to today", {"kind": "last", "unit": "month", "n": 2},
     {"kind": "rolling", "unit": "month", "n": 2}),
    ("net sales, rolling 3 months", {"kind": "last", "unit": "month", "n": 3},
     {"kind": "rolling", "unit": "month", "n": 3}),
    ("net sales in the past 60 days", {"kind": "last", "unit": "month", "n": 2},
     {"kind": "rolling", "unit": "day", "n": 60}),
    ("net sales over the last 30 days", {"kind": "last", "unit": "day", "n": 30},
     {"kind": "rolling", "unit": "day", "n": 30}),
    ("ventes nettes des 2 derniers mois jusqu'à aujourd'hui", {"kind": "last", "unit": "month", "n": 2},
     {"kind": "rolling", "unit": "month", "n": 2}),
    ("ventes nettes des 60 derniers jours", {"kind": "last", "unit": "day", "n": 60},
     {"kind": "rolling", "unit": "day", "n": 60}),
    ("ventes nettes sur 3 mois glissants", {"kind": "last", "unit": "month", "n": 3},
     {"kind": "rolling", "unit": "month", "n": 3}),
    # Complete periods.
    ("net sales in the last 2 months", {"kind": "rolling", "unit": "month", "n": 2},
     {"kind": "last", "unit": "month", "n": 2}),
    ("net sales over the last 12 months", {"kind": "rolling", "unit": "month", "n": 12},
     {"kind": "last", "unit": "month", "n": 12}),
    ("net sales, trailing 12 months", {"kind": "this", "unit": "year"}, {"kind": "last", "unit": "month", "n": 12}),
    ("net sales for the trailing year", {"kind": "previous", "unit": "year"},
     {"kind": "last", "unit": "month", "n": 12}),
    ("TTM net sales", {"kind": "all"}, {"kind": "last", "unit": "month", "n": 12}),
    ("ventes nettes des 12 derniers mois", {"kind": "rolling", "unit": "month", "n": 12},
     {"kind": "last", "unit": "month", "n": 12}),
    # So far.
    ("net sales this month so far", {"kind": "this", "unit": "month"}, {"kind": "to_date", "unit": "month"}),
    ("net sales month to date", {"kind": "previous", "unit": "month"}, {"kind": "to_date", "unit": "month"}),
    ("MTD net sales", {"kind": "all"}, {"kind": "to_date", "unit": "month"}),
    ("net sales so far this quarter", {"kind": "previous", "unit": "quarter"}, {"kind": "to_date", "unit": "quarter"}),
    ("ventes nettes depuis le début du mois", {"kind": "previous", "unit": "month"},
     {"kind": "to_date", "unit": "month"}),
])
def test_the_words_choose_the_period(question, written, meant):
    got = period_meant(_plan(written), None, question)
    assert got.time.window.model_dump(exclude_defaults=True) == Window(**meant).model_dump(exclude_defaults=True)


@pytest.mark.parametrize("question, window", [
    ("net sales in the last 2 months including this one", {"kind": "last", "unit": "month", "n": 2,
                                                           "include_current": True}),
    ("net sales in March 2026", {"kind": "between", "start": "2026-03-01", "end": "2026-03-31"}),
    ("net sales by store", {"kind": "all"}),
    ("net sales last month", {"kind": "previous", "unit": "month"}),
])
def test_a_period_the_words_do_not_reword_stands(question, window):
    plan = _plan(window)
    assert period_meant(plan, None, question) is plan


def test_the_same_time_last_year_is_a_comparison():
    got = period_meant(_plan({"kind": "to_date", "unit": "month"}), None,
                       "net sales this month so far against the same time last year")
    assert got.time.compare == Compare(kind="same_period_last_year")
    # Said with no period of its own, it is left to the planner.
    plan = _plan({"kind": "all"})
    assert period_meant(plan, None, "same time last year") is plan


# ── on the answer, against SQL written by hand ─────────────────────────────


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def _ask(learned, question: str, plan: dict) -> dict:
    built, model = learned
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    payload = answer_question(question, services, Session())
    assert payload.get("type") == "assistant_response", payload.get("answer")
    return payload


def _net(learned, start: dt.date, last: dt.date) -> float:
    return float(learned[0].con.execute(f"""
        SELECT SUM(o.net_amount) FROM order_lines o JOIN calendar d ON o.order_date_key = d.date_key
        WHERE d.full_date BETWEEN DATE '{start}' AND DATE '{last}'""").fetchone()[0])


@pytest.mark.parametrize("question, written, start, last, badge", [
    # The planner read "from today" as complete months: put right, and the answer says its dates.
    ("Net sales in the last 2 months from today", {"kind": "last", "unit": "month", "n": 2},
     D(2026, 4, 16), D(2026, 6, 15), "16 Apr 2026 – 15 Jun 2026"),
    ("Net sales in the last 2 months", {"kind": "last", "unit": "month", "n": 2},
     D(2026, 4, 1), D(2026, 5, 31), "Apr–May 2026"),
    ("Net sales in the past 60 days", {"kind": "last", "unit": "month", "n": 2},
     D(2026, 4, 17), D(2026, 6, 15), "17 Apr 2026 – 15 Jun 2026"),
    ("Net sales this month so far", {"kind": "this", "unit": "month"},
     D(2026, 6, 1), D(2026, 6, 15), "1 Jun 2026 – 15 Jun 2026"),
])
def test_the_answer_is_for_those_dates_and_shows_them(retail, question, written, start, last, badge):
    got = _ask(retail, question, {"measures": ["net_amount"], "time": {"window": written}})
    assert float(got["kpi"]["value"]) == pytest.approx(_net(retail, start, last))
    periods = [b["text"] for b in got["answer"]["badges"] if b["kind"] == "period"]
    assert periods == [badge]


# ── a quarter or year not called fiscal is the calendar's ─────────────────


@pytest.mark.parametrize("question, written, start, end", [
    # Read as the second quarter of a fiscal year starting in April (July to September 2024).
    ("Net sales from Regular shoppers by store in Q2 2025", {"kind": "between", "start": "2024-07-01",
                                                             "end": "2024-09-30"}, D(2025, 4, 1), D(2025, 6, 30)),
    ("net sales 2025 Q4", {"kind": "between", "start": "2025-01-01", "end": "2025-03-31"},
     D(2025, 10, 1), D(2025, 12, 31)),
    ("ventes nettes au T3 2024", {"kind": "between", "start": "2024-10-01", "end": "2024-12-31"},
     D(2024, 7, 1), D(2024, 9, 30)),
    ("net sales in 2025 by store", {"kind": "between", "start": "2024-04-01", "end": "2025-03-31"},
     D(2025, 1, 1), D(2025, 12, 31)),
])
def test_a_quarter_or_year_not_called_fiscal_is_the_calendars(question, written, start, end):
    from core2.plan.planner import calendar_meant

    got = calendar_meant(_plan(written), None, question)
    assert (got.time.window.kind, got.time.window.start, got.time.window.end, got.time.window.fiscal) == \
        ("between", start, end, False)


@pytest.mark.parametrize("question, window", [
    ("net sales in Q2 FY2025", {"kind": "between", "start": "2024-07-01", "end": "2024-09-30"}),
    ("fiscal 2025 net sales", {"kind": "between", "start": "2024-04-01", "end": "2025-03-31"}),
    ("net sales in 2024 and 2025", {"kind": "between", "start": "2024-01-01", "end": "2025-12-31"}),
    ("net sales in March 2025", {"kind": "between", "start": "2025-03-01", "end": "2025-03-31"}),
    ("net sales Q4 2024 to Q1 2025", {"kind": "between", "start": "2024-10-01", "end": "2025-03-31"}),
    ("net sales in Q2 2025", {"kind": "between", "start": "2025-04-01", "end": "2025-06-30"}),
])
def test_a_fiscal_or_already_right_period_stands(question, window):
    from core2.plan.planner import calendar_meant

    plan = _plan(window)
    assert calendar_meant(plan, None, question) is plan
