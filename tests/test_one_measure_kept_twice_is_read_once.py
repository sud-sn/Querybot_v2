"""
One measure kept twice is read once.

A business keeps its stock day by day for the recent past and at each month's
end for its history, and registers a metric on each, named alike: "Units in
Stock" and "Month-End Units in Stock". "Stock by month" matched both. The
question's source was resolved -- the month-end snapshot, for a monthly
question; the daily one for "stock by date" -- but both metrics stayed, both
snapshots' dates were governed, and the reader was told "I couldn't build a
trusted join plan ... the confirmed relationships do not connect Month End
Date, Date to the rest of the question".

Of two metrics that are one measure kept at two frequencies, the one on the
question's source is read. Metrics of different measures are not touched, and
a question whose source is not settled -- "stock by product category" -- still
asks which dataset (tests/test_a_dataset_the_reader_chooses_is_the_one_read.py).

tests/star_harness.py keeps the stock day by day for the last quarter of 2025,
and at each month's end since January 2024.
"""

from __future__ import annotations

import datetime as dt
import re

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("kept-twice")) as built:
        yield built


def _stock(day: dt.date) -> int:
    return sum(star.units_in_stock(product, day) for product in star.PRODUCTS)


def _reads(sql: str) -> set[str]:
    return set(re.findall(r"\[(FACTPRODUCTINVENTORY\w*)\]", sql.upper()))


class TestTheProductAnswers:

    def test_stock_by_month(self, warehouse):
        answer = star.ask(warehouse, "Stock by month in 2025")
        assert answer["model_wrote_sql"] is False
        assert _reads(answer["sql"]) == {"FACTPRODUCTINVENTORYMONTHLY"}
        assert [row["MONTH_END_UNITS_IN_STOCK"] for row in answer["rows"]] == [
            _stock(day) for day in star.month_ends() if day.year == 2025]

    def test_in_french(self, warehouse):
        answer = star.ask(warehouse, "Stock par mois", "fr")
        assert _reads(answer["sql"]) == {"FACTPRODUCTINVENTORYMONTHLY"}
        assert [row["MONTH_END_UNITS_IN_STOCK"] for row in answer["rows"]] == [
            _stock(day) for day in star.month_ends()]

    def test_stock_by_date(self, warehouse):
        answer = star.ask(warehouse, "Stock by date")
        assert _reads(answer["sql"]) == {"FACTPRODUCTINVENTORY"}
        assert [list(row.values()) for row in answer["rows"]] == [[_stock(dt.date(2025, 12, 31))]]


def _metric(name: str, table: str) -> dict:
    return {"name": name, "base_table": table, "sql_template": "SUM(Units)"}


_DAILY = _metric("Units in Stock", "dbo.FactStock")
_MONTHLY = _metric("Month-End Units in Stock", "dbo.FactStockMonthly")
_SALES = _metric("Units Sold", "dbo.FactSales")


class TestTheRule:

    @pytest.mark.parametrize("fact,kept", [
        ("DBO.FACTSTOCKMONTHLY", ["Month-End Units in Stock"]),
        ("DBO.FACTSTOCK", ["Units in Stock"]),
        ("DBO.FACTSALES", ["Units in Stock", "Month-End Units in Stock"]),
    ])
    def test_one_of_two_siblings(self, fact, kept):
        from core.source_resolution import siblings_on_source

        assert [m["name"] for m in siblings_on_source([_DAILY, _MONTHLY], fact, {})] == kept

    def test_different_measures_are_kept(self):
        from core.source_resolution import siblings_on_source

        assert [m["name"] for m in siblings_on_source([_SALES, _DAILY], "DBO.FACTSTOCK", {})] == [
            "Units Sold", "Units in Stock"]
