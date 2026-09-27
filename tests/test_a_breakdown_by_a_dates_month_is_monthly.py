"""
A breakdown by a date's month is monthly.

A question broken down by a calendar unit was read as a series only where it
said "by month" word for word. Business users name the date they mean in the
same breath -- "sales by ship month", "orders by due month", "revenue by fiscal
year" -- and the date was found, but the breakdown was not: on a made-up
retailer's warehouse "sales amount by due month in 2025" was answered with one
number, the year's total by due date, where twelve were asked for.

A breakdown by a calendar unit is read wherever the date it is on is named
between "by" and the unit ("by ship month", "by order date month", "by fiscal
year"), as the date role already was. A window's words are never that date --
"stock by warehouse last month" is broken down by warehouse, over last month --
nor is "year to date".

tests/star_harness.py keeps each sale's order, due and ship days.
"""

from __future__ import annotations

import datetime as dt

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("breakdown-grain")) as built:
        yield built


def _day(line, when: str) -> dt.date:
    return {"order": line[2], "due": line[2] + dt.timedelta(days=14), "ship": line[3]}[when]


def _monthly(when: str, value) -> list:
    totals: dict = {}
    for line in star.orders():
        day = _day(line, when)
        if day.year == 2025:
            totals.setdefault(day.month, []).append(line)
    return [value(lines) for _month, lines in sorted(totals.items())]


def _amount(lines) -> float:
    return sum(line[7] * star.PRODUCTS[line[4]][4] for line in lines)


class TestTheProductAnswers:

    def test_sales_by_due_month(self, warehouse):
        answer = star.ask(warehouse, "Sales amount by due month in 2025")
        assert answer["model_wrote_sql"] is False
        assert [row["SALES_AMOUNT"] for row in answer["rows"]] == _monthly("due", _amount)

    def test_orders_by_ship_month(self, warehouse):
        answer = star.ask(warehouse, "Number of orders by ship month in 2025")
        assert answer["model_wrote_sql"] is False
        assert [row["NUMBER_OF_ORDERS"] for row in answer["rows"]] == _monthly(
            "ship", lambda lines: len({line[0] for line in lines}))

    def test_sales_by_due_month_over_the_last_six_months(self, warehouse):
        answer = star.ask(warehouse, "Sales amount by due month over the last 6 months")
        assert answer["model_wrote_sql"] is False
        by_month = dict(zip(range(1, 13), _monthly("due", _amount)))
        assert len(answer["rows"]) == 6
        assert [row["SALES_AMOUNT"] for row in answer["rows"]] == [
            by_month[row["PERIOD"].month] for row in answer["rows"]]

    def test_sales_by_month(self, warehouse):
        answer = star.ask(warehouse, "Sales amount by month in 2025")
        assert [row["SALES_AMOUNT"] for row in answer["rows"]] == _monthly("order", _amount)

    def test_sales_by_due_date_in_2025_is_one_number(self, warehouse):
        answer = star.ask(warehouse, "Sales amount by due date in 2025")
        assert [row["SALES_AMOUNT"] for row in answer["rows"]] == [sum(_monthly("due", _amount))]


class TestTheWords:

    @pytest.mark.parametrize("question,grain", [
        ("Sales by ship month in 2025", "month"),
        ("Sales by order date month", "month"),
        ("Revenue by fiscal year", "year"),
        ("Units sold per week", "week"),
        ("Sales by quarter", "quarter"),
        ("Stock by warehouse last month", ""),
        ("Sales by customer year to date", ""),
        ("Sales by warehouse", ""),
        ("Sales by product in the month of May", ""),
    ])
    def test_the_breakdown(self, question, grain):
        from core.contextual_dates import breakdown_grain

        assert breakdown_grain(question) == grain

    def test_the_requested_grain(self):
        from core.contextual_dates import requested_temporal_grain

        assert requested_temporal_grain("Sales by ship month in 2025") == "month"
        # The window's unit where no breakdown is asked for, as before.
        assert requested_temporal_grain("Stock by warehouse last month") == "month"
        assert requested_temporal_grain("Sales over the last 6 months") == "month"
