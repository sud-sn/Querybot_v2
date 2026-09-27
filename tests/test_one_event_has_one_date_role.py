"""
One event has one date role.

A fact commonly keeps each of its dates twice: as a key into the calendar and as
a date of its own -- OrderDateKey beside OrderDate, ShipDateKey beside ShipDate.
Discovery proposed a date role for each, two "Order Date"s on one fact, and an
administrator approving what was proposed approved both. Every question that
named the order, the shipment or the due date was then asked which of the two it
meant -- "how many orders were shipped in 2025?", "sales amount by month of
order date in 2025" -- though they are the same date.

A date column and a calendar key that date the same event -- the same words, less
the key's -- are one business date, read through the calendar, which answers by
fiscal period and month name where the bare date cannot. A date of its own with
no calendar twin is still a date role.

tests/star_harness.py keeps the order, due and ship dates of its sales both ways.
"""

from __future__ import annotations

import datetime as dt

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("one-date-role")) as built:
        yield built


def _lines(when, year: int = 2025):
    """Order lines whose date (placed, due or shipped) falls in ``year``."""
    for line in star.orders():
        placed, shipped = line[2], line[3]
        day = {"placed": placed, "due": placed + dt.timedelta(days=14), "shipped": shipped}[when]
        if day.year == year:
            yield line


def _amount(line) -> float:
    return line[7] * star.PRODUCTS[line[4]][4]


class TestTheProductAnswers:

    def test_orders_shipped(self, warehouse):
        answer = star.ask(warehouse, "How many orders were shipped in 2025?")
        assert answer["model_wrote_sql"] is False
        assert [list(row.values()) for row in answer["rows"]] == [[len({line[0] for line in _lines("shipped")})]]

    def test_sales_by_due_date(self, warehouse):
        answer = star.ask(warehouse, "Sales amount by due date in 2025")
        assert answer["model_wrote_sql"] is False
        assert [list(row.values()) for row in answer["rows"]] == [[sum(_amount(line) for line in _lines("due"))]]

    def test_sales_by_month_of_order_date(self, warehouse):
        answer = star.ask(warehouse, "Sales amount by month of order date in 2025")
        assert answer["model_wrote_sql"] is False
        by_month: dict = {}
        for line in _lines("placed"):
            by_month[line[2].month] = by_month.get(line[2].month, 0) + _amount(line)
        assert sorted(row["SALES_AMOUNT"] for row in answer["rows"]) == sorted(by_month.values())


def _role(column: str, key_type: str, dimension: str = "") -> dict:
    return {"fact_column": column, "date_key_type": key_type, "dimension_table": dimension}


class TestTheRoles:

    def test_a_date_and_its_calendar_key_are_one_role(self):
        from core.semantic_model import _one_role_per_event

        roles = [_role("OrderDateKey", "surrogate_fk", "dbo.DimDate"), _role("OrderDate", "timestamp"),
                 _role("ORD_DT_DMS_KEY", "surrogate_fk", "DT_DMS"), _role("ORD_DT", "native_date")]
        assert [r["fact_column"] for r in _one_role_per_event(roles)] == ["OrderDateKey", "ORD_DT_DMS_KEY"]

    def test_a_date_with_no_calendar_twin_is_a_role(self):
        from core.semantic_model import _one_role_per_event

        roles = [_role("DateKey", "surrogate_fk", "dbo.DimDate"), _role("MovementDate", "native_date"),
                 _role("ShipDate", "timestamp")]
        assert [r["fact_column"] for r in _one_role_per_event(roles)] == ["DateKey", "MovementDate", "ShipDate"]

    @pytest.mark.parametrize("column,event", [
        ("OrderDateKey", "ORDER_DATE"), ("ORDER_DATE_ID", "ORDER_DATE"), ("ORD_DT_DMS_KEY", "ORD_DT"),
        ("ShipDate", "SHIP_DATE"), ("DateKey", "DATE"), ("Key", "KEY"),
    ])
    def test_the_event_a_column_dates(self, column, event):
        from core.semantic_model import _event_of

        assert _event_of(column) == event
