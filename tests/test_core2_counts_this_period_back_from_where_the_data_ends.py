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
