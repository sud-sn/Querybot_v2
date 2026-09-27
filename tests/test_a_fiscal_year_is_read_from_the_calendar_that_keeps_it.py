"""
A fiscal year is read from the calendar that keeps it.

"Sales in fiscal year 2025" was answered with a question: "Which month does
your fiscal year start?" -- on a warehouse whose calendar says, on every row,
which fiscal year and quarter the day is in. The reader was asked for what the
warehouse already knew, and once they answered, the product still did not
read it: the calendar's FiscalYear column, bound as a measure, sent the
question to the model, which was refused. French "Ventes de l'exercice 2025"
was refused without the question. And a start month is not all a fiscal year
is: "fiscal 2025" is July 2024 to June 2025 in one business and July 2025 to
June 2026 in the next.

The calendar's fiscal year and quarter are now calendar attributes like its
year and month. A fiscal year or quarter the question names -- "fiscal year
2025", "FY25", "fiscal Q1 2025", "Q1 FY2025", "exercice 2025", or a bare "Q1
2025" once the reader has said quarters are fiscal -- is kept on them, as the
calendar numbers it: `FiscalYear = 2025`. Where every calendar of the
warehouse keeps the fiscal year the question needs, no start month is asked
for. A relative fiscal period ("last fiscal year") names no year to read, and
is asked about as before; so is a warehouse whose calendar keeps no fiscal
year. The validator reads a filter on the plan's calendar columns as a period,
not as a lookup of one record.

tests/star_harness.py keeps two years of order lines and a calendar whose
fiscal year starts in July and is numbered by the year it ends in.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("fiscal-year")) as built:
        yield built


def _sales(first: dt.date, last: dt.date) -> float:
    return sum(line[7] * star.PRODUCTS[line[4]][4] for line in star.orders() if first <= line[2] <= last)


def _asked(answer: dict) -> list:
    return [body for kind, body in answer["replies"] if kind == "clarify"]


def _forget_the_threads_calendar() -> None:
    """A basis an earlier question in the harness's one thread settled is
    that thread's until it is forgotten."""
    from core import query_pipeline

    query_pipeline.conversation_state_store.clear(star.ACCOUNT, f"{star.ACCOUNT}:portal:harness")


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("Sales in fiscal year 2025", "en"),
        ("Sales in FY25", "en"),
        ("Ventes de l'exercice 2025", "fr"),
    ])
    def test_a_fiscal_year(self, warehouse, question, lang):
        _forget_the_threads_calendar()
        answer = star.ask(warehouse, question, lang)
        assert _asked(answer) == [] and answer["model_wrote_sql"] is False
        assert [list(row.values()) for row in answer["rows"]] == [
            [_sales(dt.date(2024, 7, 1), dt.date(2025, 6, 30))]]

    def test_a_fiscal_quarter(self, warehouse):
        _forget_the_threads_calendar()
        answer = star.ask(warehouse, "Sales in fiscal Q1 2025")
        assert _asked(answer) == []
        assert [list(row.values()) for row in answer["rows"]] == [
            [_sales(dt.date(2024, 7, 1), dt.date(2024, 9, 30))]]

    def test_a_quarter_the_reader_calls_fiscal(self, warehouse):
        _forget_the_threads_calendar()
        answer = star.ask(warehouse, "Sales in Q2 2025", choose="Fiscal quarters")
        assert [list(row.values()) for row in answer["rows"]] == [
            [_sales(dt.date(2024, 10, 1), dt.date(2024, 12, 31))]]

    def test_the_months_of_a_fiscal_year(self, warehouse):
        _forget_the_threads_calendar()
        answer = star.ask(warehouse, "Sales by month in fiscal year 2025")
        months = [dt.date(2024 + (month < 7), month, 1) for month in (*range(7, 13), *range(1, 7))]
        assert [(row["PERIOD"], row["SALES_AMOUNT"]) for row in answer["rows"]] == [
            (first, _sales(first, (first + dt.timedelta(days=31)).replace(day=1) - dt.timedelta(days=1)))
            for first in months]

    def test_a_date_whose_calendar_cannot_keep_it_is_not_all_dates(self, warehouse, monkeypatch):
        # The warehouse keeps a fiscal year, so the start month is not asked
        # for; were the date the question binds on a calendar that keeps none,
        # the fiscal year is kept nowhere, and every date is no answer.
        import core.contextual_dates as contextual_dates
        from core.i18n import t

        _forget_the_threads_calendar()
        monkeypatch.setattr(contextual_dates, "fiscal_period_window", lambda *args, **kwargs: {})
        answer = star.ask(warehouse, "Sales in fiscal year 2024")
        assert answer["executed"] == [] and answer["rows"] == []
        assert ("message", t("clar.date.window_not_applied")) in answer["replies"]

    def test_a_relative_fiscal_period_is_still_asked_about(self, warehouse):
        _forget_the_threads_calendar()
        answer = star.ask(warehouse, "Sales last fiscal year")
        assert [body["question"] for body in _asked(answer)] == ["Which month does your fiscal year start?"]


class TestTheReading:

    @pytest.mark.parametrize("question,basis,read", [
        ("sales in fiscal year 2025", "unresolved", (2025, 0, "FY2025")),
        ("sales in fy2025", "unresolved", (2025, 0, "FY2025")),
        ("sales in fy 25", "unresolved", (2025, 0, "FY2025")),
        ("sales for fiscal 2024 by month", "unresolved", (2024, 0, "FY2024")),
        ("sales in financial year 2025", "unresolved", (2025, 0, "FY2025")),
        ("ventes de l'exercice 2025", "unresolved", (2025, 0, "FY2025")),
        ("sales in fiscal q1 2025", "unresolved", (2025, 1, "FY2025 Q1")),
        ("sales in q3 fy25", "unresolved", (2025, 3, "FY2025 Q3")),
        ("sales in fq2 fy2025", "unresolved", (2025, 2, "FY2025 Q2")),
        ("sales in the second fiscal quarter of 2025", "unresolved", (2025, 2, "FY2025 Q2")),
        ("sales in q4 2025", "fiscal", (2025, 4, "FY2025 Q4")),
    ])
    def test_a_fiscal_period_named(self, question, basis, read):
        from core.contextual_dates import read_fiscal_period

        period = read_fiscal_period(question, basis)
        assert (period["fiscal_year"], period["fiscal_quarter"], period["label"]) == read

    @pytest.mark.parametrize("question,basis", [
        ("sales in q4 2025", "calendar"),
        ("sales in q4 2025", "unresolved"),
        ("sales in 2025", "fiscal"),
        ("sales last fiscal year", "unresolved"),
        ("sales in fiscal year 2024 and 2025", "unresolved"),
        ("sales fy24 vs fy25", "unresolved"),
        ("sales in fiscal q1 and q2 2025", "unresolved"),
    ])
    def test_none_named(self, question, basis):
        from core.contextual_dates import read_fiscal_period

        assert read_fiscal_period(question, basis) == {}

    def test_the_calendars_fiscal_columns(self):
        from core.contextual_dates import infer_calendar_attributes

        def found(*columns):
            attributes = infer_calendar_attributes("dbo.Cal", {"dbo.Cal": {name: "int" for name in columns}})
            return {key: value for key, value in attributes.items() if key.startswith("fiscal")}

        assert found("FiscalYear", "FiscalQuarter") == {"fiscal_year": "FiscalYear", "fiscal_quarter": "FiscalQuarter"}
        assert found("FSCL_YR", "FSCL_QTR") == {"fiscal_year": "FSCL_YR", "fiscal_quarter": "FSCL_QTR"}
        assert found("FISCAL_YEAR_START", "FISCALYEARLABEL") == {}

    def test_the_window_is_the_calendars(self):
        from core.contextual_dates import fiscal_period_window

        calendar = {"date": "FullDate", "fiscal_year": "FiscalYear", "fiscal_quarter": "FiscalQuarter"}
        assert fiscal_period_window("sales in fiscal q2 2025", calendar)["calendar_filter"] == {
            "fiscal_year": 2025, "fiscal_quarter": 2}
        assert fiscal_period_window("sales in fy2025", {"date": "FullDate", "fiscal_year": "FiscalYear"}) == {
            "kind": "named_period", "anchor_policy": "stated", "period_grain": "fiscal_year", "label": "FY2025",
            "calendar_filter": {"fiscal_year": 2025}}
        assert fiscal_period_window("sales in fiscal q2 2025", {"date": "FullDate", "fiscal_year": "FiscalYear"}) == {}
        assert fiscal_period_window("sales in fy2025", {"date": "FullDate", "year": "CalendarYear"}) == {}


class TestThePlanner:

    @pytest.mark.parametrize("question,kept,asked", [
        ("Sales in fiscal year 2025", ["fiscal_year"], False),
        ("Sales in fiscal Q1 2025", ["fiscal_year", "fiscal_quarter"], False),
        ("Sales in fiscal year 2025", [], True),
        ("Sales in fiscal Q1 2025", ["fiscal_year"], True),
        ("Sales last fiscal year", ["fiscal_year", "fiscal_quarter"], True),
    ])
    def test_the_start_month_is_asked_only_where_the_calendar_cannot_say(self, question, kept, asked):
        from core.analytical_intent import plan_analytical_intent

        plan = plan_analytical_intent(question, calendar_profile={"fiscal_attributes": kept})
        assert ("fiscal_year_start_month" in plan.unresolved_slots) is asked

    @pytest.mark.parametrize("tables,kept", [
        ({"DimDate": [("DateKey", "int"), ("FullDateAlternateKey", "date"), ("CalendarYear", "smallint"),
                      ("FiscalYear", "smallint"), ("FiscalQuarter", "tinyint")],
          "FactSales": [("OrderDateKey", "int"), ("SalesAmount", "money")]}, ["fiscal_year", "fiscal_quarter"]),
        ({"DimDate": [("DateKey", "int"), ("FullDateAlternateKey", "date"), ("CalendarYear", "smallint"),
                      ("FiscalYear", "smallint")]}, ["fiscal_year"]),
        ({"DimDate": [("DateKey", "int"), ("FullDateAlternateKey", "date"), ("CalendarYear", "smallint")]}, []),
        # Two calendars: a fiscal year only one of them keeps is not the warehouse's.
        ({"DimDate": [("DateKey", "int"), ("FullDateAlternateKey", "date"), ("CalendarYear", "smallint"),
                      ("FiscalYear", "smallint")],
          "DimPeriod": [("PeriodDate", "date"), ("Year", "int")]}, []),
    ])
    def test_the_fiscal_years_the_warehouse_keeps(self, tmp_path, tables, kept):
        from core.query_pipeline import _calendar_fiscal_attributes

        schema = {f"WH.dbo.{name}": {"columns": [{"name": column, "type": declared} for column, declared in columns],
                                     "pk_columns": [], "schema": "dbo", "database": "WH"}
                  for name, columns in tables.items()}
        (tmp_path / "_schema.json").write_text(json.dumps(schema), encoding="utf-8")
        assert _calendar_fiscal_attributes(str(tmp_path)) == kept

    def test_a_warehouse_it_cannot_read_keeps_none(self, tmp_path):
        from core.query_pipeline import _calendar_fiscal_attributes

        assert _calendar_fiscal_attributes(str(tmp_path / "missing")) == []


_POLICY = {
    "kind": "named_period", "label": "FY2025 Q1", "calendar_filter": {"fiscal_year": 2025, "fiscal_quarter": 1},
    "fact_table": "dbo.FactSales", "fact_column": "OrderDateKey", "dimension_table": "dbo.DimDate",
    "dimension_key": "DateKey", "date_table": "dbo.DimDate", "date_column": "FullDate", "role_alias": "order_date",
    "date_key_type": "surrogate_fk", "business_role": "Order Date",
    "calendar_attributes": {"date": "FullDate", "fiscal_year": "FiscalYear", "fiscal_quarter": "FiscalQuarter"},
}


class TestTheModelIsToldTheSame:

    def test_the_repair_guidance(self):
        from core.query_pipeline import _governed_date_anchor_repair_lines

        lines = _governed_date_anchor_repair_lines({"temporal_policies": [_POLICY]})
        assert "dbo.DimDate.FiscalYear = 2025 and dbo.DimDate.FiscalQuarter = 1" in lines
        assert "None" not in lines

    def test_the_field_plan(self):
        from core.semantic_planner import format_semantic_field_plan

        date = {"term": "Order Date", "table": "dbo.DimDate", "column": "FullDate", "role": "date"}
        text = format_semantic_field_plan({"fields": [date], "joins": [], "temporal_policies": [_POLICY]})
        assert "order_date.FiscalYear = 2025 and order_date.FiscalQuarter = 1" in text
        assert "None" not in text


class TestTheValidator:

    _TABLES = {"DBO.FACTSALES": {"ORDERDATEKEY": "int", "CUSTOMERKEY": "int", "SALESAMOUNT": "money"},
               "DBO.DIMDATE": {"DATEKEY": "int", "FULLDATE": "date", "FISCALYEAR": "smallint"}}

    def _errors(self, where: str, calendar_columns: frozenset[str]):
        import sqlglot

        from core.validator import _find_null_aggregate_diagnostic_errors

        sql = ("SELECT SUM(f.SalesAmount) AS SALES FROM dbo.FactSales AS f "
               f"JOIN dbo.DimDate AS d ON f.OrderDateKey = d.DateKey WHERE {where}")
        return _find_null_aggregate_diagnostic_errors(sql, sqlglot.parse_one(sql, read="tsql"), self._TABLES,
                                                      calendar_columns)

    def test_a_calendar_column_keeps_a_period(self):
        assert self._errors("d.FiscalYear = 2025", frozenset({"FISCALYEAR"})) == []

    def test_a_record_is_still_a_record(self):
        assert self._errors("f.CustomerKey = 5", frozenset({"FISCALYEAR"}))
        assert self._errors("d.FiscalYear = 2025", frozenset())

    def test_the_plans_calendar_columns(self):
        from core.validator import _plan_calendar_columns

        assert _plan_calendar_columns({"semantic_plan": {"temporal_policies": [_POLICY]}}) == frozenset(
            {"FULLDATE", "FISCALYEAR", "FISCALQUARTER"})
        assert _plan_calendar_columns({}) == frozenset()
