"""
A period is read where it is kept.

A business often keeps one measure twice: day by day for the recent past, and
at each month's end for its history. Its administrator registers both, named
alike -- "Units in Stock" and "Month-End Units in Stock". A reader asking for
"units in stock at the end of 2024" uses the first name, and the daily snapshot
starts long after 2024: the answer was empty where the business keeps the
figure.

Where a question names a period, the metric's own table holds no row in it by
its governed date, and a sibling's table -- a metric named alike once the words
for how often it is kept are left out -- holds rows in it, the sibling answers,
and the reader is told. A period the metric's own table keeps is read there; a
period no table keeps a figure of -- rows with an empty measure keep none -- is
still no data (tests/test_a_period_the_question_names_is_compiled.py asks for
one on a month-end snapshot whose stock column is empty).

tests/star_harness.py keeps the stock day by day for the last quarter of 2025,
and at each month's end since January 2024.
"""

from __future__ import annotations

import datetime as dt

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("period-kept")) as built:
        yield built


def _stock(day: dt.date, products=None) -> int:
    return sum(star.units_in_stock(product, day) for product in (products or star.PRODUCTS))


def _told(answer: dict) -> list[str]:
    return [text for kind, text in answer["replies"] if kind == "message" and "Month-End Units in Stock" in text]


def _category(product: int) -> str:
    return star.CATEGORIES[star.SUBCATEGORIES[star.PRODUCTS[product][3]][2]][0]


class TestTheProductAnswers:

    def test_a_year_the_daily_snapshot_does_not_keep(self, warehouse):
        answer = star.ask(warehouse, "Units in stock at the end of 2024")
        assert answer["model_wrote_sql"] is False
        assert [list(row.values()) for row in answer["rows"]] == [[_stock(dt.date(2024, 12, 31))]]
        assert "FACTPRODUCTINVENTORYMONTHLY" in answer["sql"].upper()
        told = _told(answer)
        assert len(told) == 1 and "Units in Stock" in told[0] and "2024" in told[0]

    def test_in_french(self, warehouse):
        answer = star.ask(warehouse, "Unités en stock à la fin de 2024", "fr")
        assert [list(row.values()) for row in answer["rows"]] == [[_stock(dt.date(2024, 12, 31))]]
        told = _told(answer)
        assert len(told) == 1 and "n’a pas de chiffres pour 2024" in told[0]

    def test_a_month_it_keeps_at_the_month_end(self, warehouse):
        answer = star.ask(warehouse, "Units in stock in March 2024")
        assert [list(row.values()) for row in answer["rows"]] == [[_stock(dt.date(2024, 3, 31))]]

    def test_by_category(self, warehouse):
        answer = star.ask(warehouse, "Units in stock at the end of 2024 by product category")
        assert answer["model_wrote_sql"] is False
        expected: dict[str, int] = {}
        for product in star.PRODUCTS:
            expected[_category(product)] = expected.get(_category(product), 0) + _stock(
                dt.date(2024, 12, 31), [product])
        # The figures are the month-end snapshot's, and named for it.
        assert {row["PRODUCT_CATEGORY"]: row["MONTH_END_UNITS_IN_STOCK"] for row in answer["rows"]} == expected

    def test_a_year_the_daily_snapshot_keeps_is_read_there(self, warehouse):
        answer = star.ask(warehouse, "Units in stock at the end of 2025")
        assert [list(row.values()) for row in answer["rows"]] == [[_stock(dt.date(2025, 12, 31))]]
        assert "FACTPRODUCTINVENTORYMONTHLY" not in answer["sql"].upper()
        assert _told(answer) == []

    def test_a_year_no_table_keeps_is_no_data(self, warehouse):
        answer = star.ask(warehouse, "Units in stock at the end of 2023")
        assert all(value is None for row in answer["rows"] for value in row.values())
        assert "FACTPRODUCTINVENTORYMONTHLY" not in answer["sql"].upper()
        assert _told(answer) == []


def _metric(name: str, table: str, metric_id: int = 0) -> dict:
    return {"id": metric_id, "name": name, "base_table": table, "synonyms": ""}


def _role(table: str, column: str, *, default: bool = False, status: str = "approved", **extra) -> dict:
    return {"fact_table": table, "fact_column": column, "status": status, "is_default": default,
            "date_key_type": "surrogate_fk", "dimension_table": "dbo.DimDate", "dimension_key": "DateKey",
            "date_value_column": "FullDate", **extra}


class TestTheSiblings:

    @pytest.mark.parametrize("name,measure", [
        ("Month-End Units in Stock", "units in stock"),
        ("Month-end allocated quantity", "allocated quantity"),
        ("Stock on hand at month end", "stock on hand"),
        ("Stock at the end of the month", "stock"),
        ("Daily sales", "sales"),
        ("Stock en fin de mois", "stock"),
        ("Ventes mensuelles", "ventes"),
        ("Units in Stock", "units in stock"),
        ("Sales by end date", "sales by end date"),
    ])
    def test_the_measure_a_name_keeps(self, name, measure):
        from core.period_siblings import measured

        assert measured(name) == measure

    def test_a_sibling_is_on_another_table(self):
        from core.period_siblings import period_siblings

        daily = _metric("Units in Stock", "dbo.FactStock")
        metrics = [daily, _metric("Month-End Units in Stock", "dbo.FactStockMonthly"),
                   _metric("Monthly Units in Stock", "dbo.FactStock"),
                   _metric("Month-End Inventory Value", "dbo.FactStockMonthly")]
        assert [m["name"] for m in period_siblings(daily, metrics)] == ["Month-End Units in Stock"]

    def test_the_date_a_metric_is_read_by(self):
        from core.period_siblings import governed_date

        metric = _metric("Units in Stock", "dbo.FactStock", 7)
        roles = [_role("dbo.FactStock", "LoadKey"), _role("dbo.FactStock", "DateKey", default=True),
                 _role("dbo.FactStockMonthly", "MonthEndKey", default=True)]
        assert governed_date(metric, roles)["fact_column"] == "DateKey"
        binding = {"metric_id": 7, "fact_table": "dbo.FactStock", "fact_column": "LoadKey", "is_default": 1}
        assert governed_date(metric, roles, [binding])["fact_column"] == "LoadKey"
        assert governed_date(metric, [_role("dbo.FactStock", "LoadKey")])["fact_column"] == "LoadKey"
        assert governed_date(metric, [_role("dbo.FactStock", "LoadKey"), _role("dbo.FactStock", "DateKey")]) == {}
        assert governed_date(metric, [_role("dbo.FactStock", "DateKey", default=True, status="generated")]) == {}


class TestTheProbe:

    def test_a_calendar_key(self):
        from core.period_siblings import build_period_rows_probe_sql

        sql = build_period_rows_probe_sql(_role("dbo.FactStock", "DateKey"), "2024-01-01", "2025-01-01")
        assert "JOIN [dbo].[DimDate] AS period_date ON period_rows.[DateKey] = period_date.[DateKey]" in sql
        assert "period_date.[FullDate] >= CAST('2024-01-01' AS date)" in sql
        assert "period_date.[FullDate] < CAST('2025-01-01' AS date)" in sql

    def test_a_period_key(self):
        from core.period_siblings import build_period_rows_probe_sql

        role = _role("dbo.FactStockMonthly", "PeriodKey", date_key_type="yyyymm_integer",
                     dimension_table="", dimension_key="", date_value_column="PeriodKey")
        sql = build_period_rows_probe_sql(role, "2024-01-01", "2025-01-01")
        assert "JOIN" not in sql
        assert "TRY_CONVERT(date, CONVERT(varchar(6), period_rows.[PeriodKey]) + '01', 112) >= " in sql

    def test_a_figure_not_a_row(self):
        from core.period_siblings import build_period_rows_probe_sql

        sql = build_period_rows_probe_sql(_role("dbo.FactStock", "DateKey"), "2024-01-01", "2025-01-01",
                                          measures=("UnitsBalance", "UnitCost"))
        assert sql.endswith("AND (period_rows.[UnitCost] IS NOT NULL OR period_rows.[UnitsBalance] IS NOT NULL)")

    def test_an_unfinished_date_is_not_probed(self):
        from core.period_siblings import build_period_rows_probe_sql

        assert build_period_rows_probe_sql(_role("dbo.FactStock", "DateKey", dimension_key=""),
                                           "2024-01-01", "2025-01-01") == ""

    @pytest.mark.parametrize("rows,kept", [([("2024-12-31",)], True), ([(None,)], False), ([], False),
                                           ([{"last_date_in_period": None}], False)])
    def test_what_the_probe_says(self, rows, kept):
        from core.period_siblings import clear_cache, keeps_period

        clear_cache()
        period = {"start": "2024-01-01", "end": "2025-01-01"}
        assert keeps_period("acct", _role("dbo.FactStock", "DateKey"), period, lambda sql: rows) is kept

    def test_a_failed_probe_tells_nothing_and_is_asked_again(self):
        from core.period_siblings import clear_cache, keeps_period

        clear_cache()
        period = {"start": "2024-01-01", "end": "2025-01-01"}
        asked: list[str] = []

        def failing(sql):
            asked.append(sql)
            raise RuntimeError("timeout")

        assert keeps_period("acct", _role("dbo.FactStock", "DateKey"), period, failing) is None
        assert keeps_period("acct", _role("dbo.FactStock", "DateKey"), period, failing) is None
        assert len(asked) == 2

    def test_an_answer_is_kept_for_the_next_question(self):
        from core.period_siblings import clear_cache, keeps_period

        clear_cache()
        period = {"start": "2024-01-01", "end": "2025-01-01"}
        asked: list[str] = []

        def probe(sql):
            asked.append(sql)
            return [(None,)]

        for _ in range(2):
            assert keeps_period("acct", _role("dbo.FactStock", "DateKey"), period, probe) is False
        assert len(asked) == 1
        assert keeps_period("acct", _role("dbo.FactStock", "DateKey"), period, probe, scope="region-7") is False
        assert len(asked) == 2
        assert keeps_period("acct", _role("dbo.FactStock", "DateKey"), period, probe, measures=("Units",)) is False
        assert len(asked) == 3


class TestTheChoice:

    @staticmethod
    def _choose(kept: dict):
        from core.period_siblings import sibling_for_period

        daily = _metric("Units in Stock", "dbo.FactStock")
        monthly = _metric("Month-End Units in Stock", "dbo.FactStockMonthly")
        roles = [_role("dbo.FactStock", "DateKey", default=True),
                 _role("dbo.FactStockMonthly", "MonthEndKey", default=True)]
        period = {"start": "2024-01-01", "end": "2025-01-01", "label": "2024"}
        return sibling_for_period(daily, [daily, monthly], period, roles, [],
                                  lambda date, metric: kept[date["fact_column"]])

    def test_the_sibling_that_keeps_it(self):
        assert self._choose({"DateKey": False, "MonthEndKey": True})["metric"]["name"] == "Month-End Units in Stock"

    def test_the_metric_that_keeps_it(self):
        assert self._choose({"DateKey": True, "MonthEndKey": True}) == {}

    def test_nobody_keeps_it(self):
        assert self._choose({"DateKey": False, "MonthEndKey": False}) == {}

    def test_it_cannot_be_told(self):
        assert self._choose({"DateKey": None, "MonthEndKey": True}) == {}
