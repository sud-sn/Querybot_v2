"""
A calendar's date is a date, whatever it is called.

A filtered total that returns one number must say when rows matched but held
no value -- "revenue for customer 123" -- and the validator asks for that
wherever a WHERE clause looks a member up by an equality on a column that is
not a date. It told a date from a key by the column's name alone. A calendar
names its date for what it is in the calendar, not for being a date: on a
made-up retailer's warehouse FullDateAlternateKey reads as a key, so a stock
level read at the last snapshot of 2024 -- `FullDateAlternateKey IN (the
snapshot's date)` -- was taken for a lookup, the governed compiler's SQL was
declined, and the question went to the model.

A column the schema declares a date or a timestamp is a date filter, whatever
its name. A name two tables share is one only where both declare it a date; a
key or a code is still a lookup.

tests/star_harness.py keeps the stock twice: day by day for the last quarter of
2025, and at each month's end since 2024.
"""

from __future__ import annotations

import datetime as dt

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("date-typed")) as built:
        yield built


def _stock(day: dt.date) -> int:
    return sum(star.units_in_stock(product, day) for product in star.PRODUCTS)


class TestTheProductAnswers:

    def test_stock_at_the_end_of_a_year(self, warehouse):
        answer = star.ask(warehouse, "Units in stock at the end of 2025")
        assert answer["model_wrote_sql"] is False
        assert [list(row.values()) for row in answer["rows"]] == [[_stock(dt.date(2025, 12, 31))]]

    def test_stock_at_the_end_of_a_year_in_french(self, warehouse):
        answer = star.ask(warehouse, "Unités en stock à la fin de 2025", "fr")
        assert answer["model_wrote_sql"] is False
        assert [list(row.values()) for row in answer["rows"]] == [[_stock(dt.date(2025, 12, 31))]]

    def test_month_end_stock_at_the_end_of_a_year(self, warehouse):
        answer = star.ask(warehouse, "Month-end units in stock at the end of 2024")
        assert answer["model_wrote_sql"] is False
        assert [list(row.values()) for row in answer["rows"]] == [[_stock(dt.date(2024, 12, 31))]]


_KNOWN = {"DW.DBO.FACTSTOCK", "DW.DBO.DIMDATE", "DW.DBO.DIMPROMOTION"}
_COLUMNS = {
    "DW.DBO.FACTSTOCK": {"PRODUCTKEY": "int", "DATEKEY": "int", "UNITSBALANCE": "int", "PROMOTIONKEY": "int"},
    "DW.DBO.DIMDATE": {"DATEKEY": "int", "FULLDATEALTERNATEKEY": "date", "CALENDARYEAR": "smallint",
                       "LOADEDAT": "TIMESTAMP_NTZ(9)", "OPENINGTIME": "time"},
}


def _validated(where: str, columns: dict | None = None) -> str:
    from core.validator import validate_sql_detailed

    sql = ("SELECT SUM(f.UnitsBalance) AS UNITS FROM dw.dbo.FactStock AS f "
           "JOIN dw.dbo.DimDate AS d ON f.DateKey = d.DateKey WHERE " + where)
    return validate_sql_detailed(sql, _KNOWN, "azure_sql", _KNOWN, columns or _COLUMNS, {}).code


class TestTheValidator:

    @pytest.mark.parametrize("where", [
        "d.FullDateAlternateKey IN (SELECT MAX(x.FullDateAlternateKey) FROM dw.dbo.DimDate AS x)",
        "d.FullDateAlternateKey = CAST('2024-12-31' AS date)",
        "d.LoadedAt = CAST('2024-12-31' AS date)",
    ])
    def test_a_date_filter_needs_no_null_diagnostics(self, where):
        assert _validated(where) == "ok"

    @pytest.mark.parametrize("where", [
        "f.ProductKey IN (1, 2)",
        "d.CalendarYear = 2024",
        "d.OpeningTime = '09:00'",
    ])
    def test_a_lookup_still_does(self, where):
        assert _validated(where) == "null_aggregate_diagnostic"

    def test_a_name_another_table_keeps_as_a_key_is_a_lookup(self):
        shared = {**_COLUMNS, "DW.DBO.DIMPROMOTION": {"PROMOTIONKEY": "int", "FULLDATEALTERNATEKEY": "int"}}
        assert _validated("d.FullDateAlternateKey = CAST('2024-12-31' AS date)", shared) == "null_aggregate_diagnostic"
