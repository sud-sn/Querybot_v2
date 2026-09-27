"""
A calendar's columns are read whatever their spelling.

"Sales by day of the week" is grouped by the calendar's own weekday number, and
the product finds that number among the calendar's columns by name: DAY_OF_WEEK,
WEEKDAY_NUMBER, DOW. A calendar spells its columns as its warehouse does --
DayNumberOfWeek, MonthNumberOfYear, CalendarQuarter -- and the query reads them
upper-cased, DAYNUMBEROFWEEK, where no snake-case name was ever equal to them.
The calendar was found to keep only a date, the governed compiler had no
weekday to group by and declined, and the question went to the model.

A calendar column is now matched as one word, its separators and case left
out, against the snake-case names and the "<unit> number of <cycle>" spellings
calendars commonly use. Matching is still of the whole name: YEAR_END_FLAG is
no year and MONTHLY_BUDGET no month.

tests/star_harness.py keeps two years of orders on a calendar spelled in
camel case (DayNumberOfWeek, EnglishDayNameOfWeek, MonthNumberOfYear,
CalendarQuarter, CalendarYear), its week starting on Sunday.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("calendar-spelling")) as built:
        yield built


def _sales_by_weekday() -> list[float]:
    totals = [0.0] * 7
    for line in star.orders():
        totals[line[2].isoweekday() % 7] += line[7] * star.PRODUCTS[line[4]][4]
    return totals


class TestTheProductAnswers:

    def test_sales_by_day_of_the_week(self, warehouse):
        answer = star.ask(warehouse, "Sales by day of the week")
        assert answer["model_wrote_sql"] is False
        assert "DAYNUMBEROFWEEK" in answer["sql"].upper()
        assert [(row["PERIOD"], row["SALES_AMOUNT"]) for row in answer["rows"]] == list(zip(
            ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"], _sales_by_weekday()))

    def test_in_french(self, warehouse):
        answer = star.ask(warehouse, "Ventes par jour de la semaine", "fr")
        assert answer["model_wrote_sql"] is False
        assert [row["SALES_AMOUNT"] for row in answer["rows"]] == _sales_by_weekday()

    def test_sales_by_quarter(self, warehouse):
        answer = star.ask(warehouse, "Sales by quarter in 2025")
        expected = [0.0] * 4
        for line in star.orders():
            if line[2].year == 2025:
                expected[(line[2].month - 1) // 3] += line[7] * star.PRODUCTS[line[4]][4]
        assert [row["SALES_AMOUNT"] for row in answer["rows"]] == expected


class TestTheRule:

    def test_a_camel_case_calendar_as_the_query_reads_it(self):
        from core.contextual_dates import infer_calendar_attributes

        columns = {name.upper(): {} for name in (
            "DateKey", "FullDateAlternateKey", "DayNumberOfWeek", "EnglishDayNameOfWeek", "DayNumberOfMonth",
            "WeekNumberOfYear", "EnglishMonthName", "MonthNumberOfYear", "CalendarQuarter", "CalendarYear")}
        assert infer_calendar_attributes("dw.dbo.DimDate", {"DW.DBO.DIMDATE": columns},
                                         date_value_column="FullDateAlternateKey") == {
            "date": "FULLDATEALTERNATEKEY", "year": "CALENDARYEAR", "month_number": "MONTHNUMBEROFYEAR",
            "month_name": "ENGLISHMONTHNAME", "quarter": "CALENDARQUARTER", "week": "WEEKNUMBEROFYEAR",
            "day": "DAYNUMBEROFMONTH", "day_of_week": "DAYNUMBEROFWEEK", "day_name": "ENGLISHDAYNAMEOFWEEK"}

    def test_a_snake_case_calendar_is_read_as_before(self):
        from core.contextual_dates import infer_calendar_attributes

        columns = {name: {} for name in ("CAL_DT", "CAL_YR", "MTH_NO", "MTH_NM", "QTR_NO", "DAY_OF_WEEK")}
        assert infer_calendar_attributes("OPS.CAL_DMS", {"OPS.CAL_DMS": columns}, date_value_column="CAL_DT") == {
            "date": "CAL_DT", "year": "CAL_YR", "month_number": "MTH_NO", "month_name": "MTH_NM",
            "quarter": "QTR_NO", "day_of_week": "DAY_OF_WEEK"}

    def test_a_name_that_only_contains_a_unit_is_not_it(self):
        from core.contextual_dates import infer_calendar_attributes

        columns = {name: {} for name in ("CalendarDate", "YearEndFlag", "MonthlyBudget", "WeekdayFlag")}
        assert infer_calendar_attributes("dbo.Cal", {"dbo.Cal": columns}) == {"date": "CalendarDate"}

    def test_the_shorter_of_spellings(self):
        from core.contextual_dates import infer_calendar_attributes

        columns = {name: {} for name in ("MonthOfYear", "WeekOfYear", "DayNameOfWeek")}
        assert infer_calendar_attributes("dbo.Cal", {"dbo.Cal": columns}) == {
            "month_number": "MonthOfYear", "week": "WeekOfYear", "day_name": "DayNameOfWeek"}
