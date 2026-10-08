"""When the data stopped months ago, "this month" is the data's last month, never an empty one.

A workspace whose data ends on 30 June 2026, asked on 8 October 2026 "which
customers bought <an item> this month?", answered "No rows match from
Jul 2026 to Jun 2026". Relative periods are counted back from the day after the
data ends (1 July), and "this month" was the month holding that day: July, cut
off where the data ends -- a period ending before it began. The same for this
quarter, this fiscal year (starting in July) and fiscal year to date. "This
month" is now the month holding the data's last day; "last month" is unchanged.
"""

from __future__ import annotations

import datetime as dt

import pytest

from core2.plan.ir import Window
from core2.resolve.time import resolve_window

TODAY = dt.date(2026, 10, 8)
ENDS = dt.date(2026, 6, 30)
D = dt.date


@pytest.mark.parametrize("window, start, end", [
    ({"kind": "this", "unit": "month"}, D(2026, 6, 1), D(2026, 7, 1)),
    ({"kind": "this", "unit": "quarter"}, D(2026, 4, 1), D(2026, 7, 1)),
    ({"kind": "this", "unit": "year"}, D(2026, 1, 1), D(2026, 7, 1)),
    ({"kind": "this", "unit": "year", "fiscal": True}, D(2025, 7, 1), D(2026, 7, 1)),
    ({"kind": "this", "unit": "quarter", "fiscal": True}, D(2026, 4, 1), D(2026, 7, 1)),
    ({"kind": "to_date", "unit": "year", "fiscal": True}, D(2025, 7, 1), D(2026, 7, 1)),
    ({"kind": "last", "unit": "month", "n": 3, "include_current": True}, D(2026, 4, 1), D(2026, 7, 1)),
    # Unchanged: complete periods counted back from the day after the data ends.
    ({"kind": "previous", "unit": "month"}, D(2026, 6, 1), D(2026, 7, 1)),
    ({"kind": "last", "unit": "month", "n": 3}, D(2026, 4, 1), D(2026, 7, 1)),
    ({"kind": "previous", "unit": "year", "fiscal": True}, D(2025, 7, 1), D(2026, 7, 1)),
])
def test_a_relative_period_is_one_the_data_has(window, start, end):
    got = resolve_window(Window.model_validate(window), today=TODAY, last_data=ENDS, fiscal_start=7)
    assert (got.start, got.end) == (start, end)
    assert got.start < got.end
    assert got.notes == ["Your data ends 30 Jun 2026, so recent periods are counted back from there."]


@pytest.mark.parametrize("window, start, end", [
    ({"kind": "this", "unit": "month"}, D(2026, 10, 1), D(2026, 10, 9)),
    ({"kind": "this", "unit": "year", "fiscal": True}, D(2026, 7, 1), D(2026, 10, 9)),
    ({"kind": "last", "unit": "month", "n": 3, "include_current": True}, D(2026, 8, 1), D(2026, 10, 9)),
])
def test_with_current_data_this_month_is_still_today(window, start, end):
    got = resolve_window(Window.model_validate(window), today=TODAY, last_data=dt.date(2026, 10, 7), fiscal_start=7)
    assert (got.start, got.end, got.notes) == (start, end, [])


# ── which year: the calendar's or the fiscal ────────────────────────────────
# "Sales in 2025" in a workspace whose fiscal years run July to June: a reader
# cannot tell from "Jan 2025 to Dec 2025" alone which year was meant, nor from
# "Jul 2025 to Jun 2026" that "this fiscal year" was taken as asked.

CALENDAR = "Counted by the calendar: fiscal years here run July to June. Ask for the fiscal year to count that way."


@pytest.mark.parametrize("window, said", [
    ({"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}, CALENDAR),        # "in 2025"
    ({"kind": "between", "start": "2026-04-01", "end": "2026-06-30"}, CALENDAR),        # "in Q2 2026"
    ({"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}, CALENDAR),        # "in H1 2026"
    ({"kind": "this", "unit": "year"}, CALENDAR),
    ({"kind": "previous", "unit": "quarter"}, CALENDAR),
    ({"kind": "this", "unit": "year", "fiscal": True}, "Fiscal years here run July to June."),
    ({"kind": "between", "start": "2025-07-01", "end": "2026-06-30", "fiscal": True}, "Fiscal years here run July to June."),
    ({"kind": "previous", "unit": "month"}, None),
    ({"kind": "last", "unit": "month", "n": 3}, None),
    ({"kind": "between", "start": "2026-03-10", "end": "2026-05-20"}, None),
    ({"kind": "between", "start": "2026-02-01", "end": "2026-04-30"}, None),             # three months, not a quarter
    ({"kind": "all"}, None),
])
def test_a_year_or_quarter_says_which_year_where_the_fiscal_year_is_not_the_calendars(window, said):
    from core2.resolve.time import year_basis

    w = Window.model_validate(window)
    got = resolve_window(w, today=TODAY, last_data=dt.date(2026, 10, 7), fiscal_start=7)
    assert year_basis(w, got, fiscal_start=7) == ([said] if said else [])


@pytest.mark.parametrize("fiscal_start", [None, 1])
def test_where_the_fiscal_year_is_the_calendars_nothing_is_said(fiscal_start):
    from core2.resolve.time import year_basis

    w = Window.model_validate({"kind": "between", "start": "2025-01-01", "end": "2025-12-31"})
    assert year_basis(w, resolve_window(w, today=TODAY, fiscal_start=fiscal_start), fiscal_start=fiscal_start) == []


def test_the_answer_carries_which_year_it_counted():
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve
    from evals.core2 import domains
    from evals.core2.compile_eval import learn

    _, model = learn(domains.build("retail"), "descriptive")
    model.settings.fiscal_year_start_month = 7
    measure = next(m.slug for m in model.measures.values() if m.slug == "net_amount")
    plan = Plan.model_validate({"intent": "value", "measures": [measure],
                                "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}})
    assert CALENDAR in resolve(plan, model, Context(today=TODAY)).notes
    model.settings.fiscal_year_start_month = 1
    assert CALENDAR not in resolve(plan, model, Context(today=TODAY)).notes
