"""
A dataset the reader chooses is the one read.

Where a question could be read on more than one dataset -- "stock by product
category" on the daily inventory snapshot or on the month-end one -- the
reader is asked which, and the answer they choose settles it. It did not:

* The choice is kept as the card offered it (DBO.FACTRESELLERSALES) and the
  model spells its facts as the warehouse does (dbo.FactResellerSales).
  Matched by spelling, the dataset the reader chose was "no longer a governed
  fact", the choice was dropped, and the question failed after the reader had
  answered it.
* Where the choice held, the metrics of the datasets the reader turned down
  stayed matched to the question, their dates were governed beside the chosen
  one's, and the join plan failed on a date the chosen table does not have:
  "the confirmed relationships do not connect Date to the rest of the
  question".
* The choices were the tables' names upper-cased into one word --
  "Factresellersales" -- where the warehouse names them in words.

tests/star_harness.py keeps the stock twice: day by day for the last quarter
of 2025, and at each month's end since 2024; its metrics are "Units in Stock"
and "Month-End Units in Stock".
"""

from __future__ import annotations

import datetime as dt
import re

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("chosen-dataset")) as built:
        yield built


def _stock_by_category(french: bool = False) -> dict[str, int]:
    totals: dict[str, int] = {}
    for product in star.PRODUCTS:
        names = star.CATEGORIES[star.SUBCATEGORIES[star.PRODUCTS[product][3]][2]]
        category = names[1] if french else names[0]
        totals[category] = totals.get(category, 0) + star.units_in_stock(product, dt.date(2025, 12, 31))
    return totals


def _reads(sql: str) -> set[str]:
    return set(re.findall(r"\[(FACTPRODUCTINVENTORY\w*)\]", sql.upper()))


class TestTheProductAnswers:

    def test_the_datasets_are_offered_by_name(self, warehouse):
        answer = star.ask(warehouse, "Stock by product category")
        asked = [payload for kind, payload in answer["replies"] if kind == "clarify"]
        assert [option["label"] for option in asked[0]["options"]] == ["Product Inventory",
                                                                        "Product Inventory Monthly"]

    @pytest.mark.parametrize("label,table,column", [
        ("Product Inventory Monthly", "FACTPRODUCTINVENTORYMONTHLY", "MONTH_END_UNITS_IN_STOCK"),
        ("Product Inventory", "FACTPRODUCTINVENTORY", "UNITS_IN_STOCK"),
    ])
    def test_the_chosen_dataset_answers(self, warehouse, label, table, column):
        answer = star.ask(warehouse, "Stock by product category", choose=label)
        assert answer["model_wrote_sql"] is False
        assert _reads(answer["sql"]) == {table}
        assert {row["PRODUCT_CATEGORY"]: row[column] for row in answer["rows"]} == _stock_by_category()

    def test_in_french(self, warehouse):
        answer = star.ask(warehouse, "Stock par catégorie de produits", "fr", choose="Product Inventory Monthly")
        assert _reads(answer["sql"]) == {"FACTPRODUCTINVENTORYMONTHLY"}
        assert {row["PRODUCT_CATEGORY"]: row["MONTH_END_UNITS_IN_STOCK"]
                for row in answer["rows"]} == _stock_by_category(french=True)


_MODEL = {"tables": [
    {"qualified_name": "dbo.FactInternetSales", "table": "FactInternetSales", "schema": "dbo", "type": "fact"},
    {"qualified_name": "dbo.FactResellerSales", "table": "FactResellerSales", "schema": "dbo", "type": "fact"},
    {"qualified_name": "dbo.DimReseller", "table": "DimReseller", "schema": "dbo", "type": "dimension"},
]}


class TestTheChoice:

    @pytest.mark.parametrize("kept", ["DBO.FACTRESELLERSALES", "dbo.FactResellerSales", "FactResellerSales"])
    def test_a_governed_fact_however_it_is_spelled(self, kept):
        from core.source_resolution import governed_source_fact

        assert governed_source_fact(kept, {"candidates": []}, _MODEL) == "DBO.FACTRESELLERSALES"

    def test_named_as_the_scope_names_it(self):
        from core.source_resolution import governed_source_fact

        scope = {"candidates": [{"table": "DBO.FACTRESELLERSALES"}]}
        assert governed_source_fact("dbo.FactResellerSales", scope, {}) == "DBO.FACTRESELLERSALES"

    @pytest.mark.parametrize("kept", ["dbo.DimReseller", "dbo.FactReturns", ""])
    def test_what_is_no_governed_fact(self, kept):
        from core.source_resolution import governed_source_fact

        assert governed_source_fact(kept, {"candidates": []}, _MODEL) == ""

    def test_the_metrics_of_the_chosen_dataset(self):
        from core.source_resolution import metrics_on_source

        metrics = [{"name": "Sales Amount", "base_table": "dbo.FactInternetSales", "sql_template": "SUM(SalesAmount)"},
                   {"name": "Reseller Sales", "base_table": "dbo.FactResellerSales", "sql_template": "SUM(SalesAmount)"}]
        assert [m["name"] for m in metrics_on_source(metrics, "DBO.FACTRESELLERSALES", {})] == ["Reseller Sales"]
        assert metrics_on_source(metrics, "DBO.FACTRETURNS", {}) == []


class TestTheNames:

    @pytest.mark.parametrize("table,label", [
        ({"qualified_name": "dbo.FactResellerSales", "entity": "Factresellersales"}, "Reseller Sales"),
        ({"qualified_name": "dbo.FactResellerSales"}, "Fact Reseller Sales"),
        ({"qualified_name": "dbo.FactProductInventoryMonthly", "entity": "x"}, "Product Inventory Monthly"),
        ({"qualified_name": "OPS.INV_BAL_FCT", "entity": "Inv Bal"}, "Inv Bal"),
    ])
    def test_a_dataset_is_offered_in_words(self, table, label):
        from core.source_resolution import _business_source_label

        assert _business_source_label(table) == label
