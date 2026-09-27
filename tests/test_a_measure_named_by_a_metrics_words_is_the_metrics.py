"""
A measure named by a metric's words is the metric's.

The field plan reads the question against the warehouse's column names before
the business's metrics are matched. Where a column is named for the measure --
a daily stock snapshot's UnitsInStock -- the question "month-end units in stock
at the end of 2024" found it there, and the plan required the daily snapshot
beside the month-end one the approved metric "Month-End Units in Stock" reads.
The confirmed relationships join no two snapshots, and the reader was told "I
couldn't build a trusted join plan for this question ... the confirmed
relationships do not connect Month End Date to the rest of the question".

A measure the question names by a matched metric's name or synonym is that
metric's, read on the metric's own table: the field found on another table by
the column's name is left out of the plan, with its joins and its claim to be
the measure's table.

tests/star_harness.py keeps the stock day by day for the last quarter of 2025
(its count named UnitsInStock), and at each month's end since January 2024.
"""

from __future__ import annotations

import datetime as dt

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("metric-words")) as built:
        yield built


def _stock(day: dt.date) -> int:
    return sum(star.units_in_stock(product, day) for product in star.PRODUCTS)


class TestTheProductAnswers:

    def test_month_end_stock_at_the_end_of_a_year(self, warehouse):
        answer = star.ask(warehouse, "Month-end units in stock at the end of 2024")
        assert answer["model_wrote_sql"] is False
        assert [list(row.values()) for row in answer["rows"]] == [[_stock(dt.date(2024, 12, 31))]]

    def test_month_end_stock_by_category(self, warehouse):
        answer = star.ask(warehouse, "Month-end units in stock by product category")
        assert answer["model_wrote_sql"] is False
        expected: dict[str, int] = {}
        for product in star.PRODUCTS:
            category = star.CATEGORIES[star.SUBCATEGORIES[star.PRODUCTS[product][3]][2]][0]
            expected[category] = expected.get(category, 0) + star.units_in_stock(product, dt.date(2025, 12, 31))
        assert {row["PRODUCT_CATEGORY"]: row["MONTH_END_UNITS_IN_STOCK"] for row in answer["rows"]} == expected

    def test_the_daily_count_is_read_on_its_own_table(self, warehouse):
        answer = star.ask(warehouse, "Units in stock at the end of 2025")
        assert [list(row.values()) for row in answer["rows"]] == [[_stock(dt.date(2025, 12, 31))]]
        assert "FACTPRODUCTINVENTORYMONTHLY" not in answer["sql"].upper()


def _field(term: str, table: str, column: str, role: str = "measure") -> dict:
    return {"term": term, "table": table, "column": column, "role": role}


def _metric(name: str, table: str, synonyms: str = "") -> dict:
    return {"name": name, "base_table": table, "synonyms": synonyms}


class TestTheRule:

    def test_a_column_named_like_the_metric_on_another_table(self):
        from core.semantic_planner import measure_tables_named_by_a_metric

        fields = [_field("unitsinstock", "DW.dbo.FactStock", "UNITSINSTOCK"),
                  _field("warehouse", "DW.dbo.DimWarehouse", "NAME", role="display_dimension")]
        assert measure_tables_named_by_a_metric(
            fields, [_metric("Month-End Units in Stock", "dbo.FactStockMonthly")]) == ["DW.dbo.FactStock"]

    def test_by_a_synonym(self):
        from core.semantic_planner import measure_tables_named_by_a_metric

        fields = [_field("revenue", "DW.dbo.FactResellerSales", "REVENUE")]
        assert measure_tables_named_by_a_metric(
            fields, [_metric("Sales Amount", "dbo.FactInternetSales", "sales, revenue")]) == ["DW.dbo.FactResellerSales"]

    def test_a_column_on_the_metrics_own_table(self):
        from core.semantic_planner import measure_tables_named_by_a_metric

        fields = [_field("unitsinstock", "DW.dbo.FactStockMonthly", "UNITSINSTOCK")]
        assert measure_tables_named_by_a_metric(
            fields, [_metric("Month-End Units in Stock", "dbo.FactStockMonthly")]) == []

    def test_a_measure_the_metric_does_not_name(self):
        from core.semantic_planner import measure_tables_named_by_a_metric

        fields = [_field("freight", "DW.dbo.FactShipments", "FREIGHT")]
        assert measure_tables_named_by_a_metric(
            fields, [_metric("Month-End Units in Stock", "dbo.FactStockMonthly")]) == []

    @pytest.mark.parametrize("term,phrase,said", [
        ("unitsinstock", "Month-End Units in Stock", True),
        ("allocated quantity", "Month-end allocated quantity", True),
        ("quantite allouee", "quantité allouée en fin de mois", True),
        ("stockin", "Month-End Units in Stock", False),
        ("qty", "Allocated quantity", False),
        ("stock", "Stock on hand", True),
        ("hand", "Handling fee", False),
    ])
    def test_said_in_a_phrase(self, term, phrase, said):
        from core.semantic_planner import _said_in

        assert _said_in(term, phrase) is said

    def test_the_plan_less_a_table(self):
        from core.semantic_planner import without_tables

        plan = {"fields": [_field("unitsinstock", "DW.dbo.FactStock", "UNITSINSTOCK"),
                           {**_field("warehouse", "DW.dbo.DimWarehouse", "NAME", "display_dimension"),
                            "enforcement": "required"}],
                "joins": [{"from": "DW.dbo.FactStock", "to": "DW.dbo.DimWarehouse"}],
                "fact_anchor": "DW.dbo.FactStock", "required_tables": ["DW.dbo.FactStock", "DW.dbo.DimWarehouse"]}
        left_out = without_tables(plan, ["dbo.FactStock"])
        assert left_out == ["DW.dbo.FactStock.UNITSINSTOCK", "DW.dbo.FactStock (the measure's table)"]
        assert [field["term"] for field in plan["fields"]] == ["warehouse"]
        assert plan["joins"] == [] and plan["fact_anchor"] == ""
        assert plan["required_tables"] == ["DW.dbo.DimWarehouse"]
