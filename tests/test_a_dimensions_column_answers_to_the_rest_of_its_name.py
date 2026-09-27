"""
A dimension's column answers to the rest of its name.

A sales territory keeps its country in SalesTerritoryCountry, its region in
SalesTerritoryRegion, its group in SalesTerritoryGroup: each named for the
territory, then for what it is. A reader asks for "sales by country", "sales
by region", "sales by territory group" -- and the field planner, which heard
only the whole name, bound nothing, and the question went to the model and
was refused. French "Ventes par pays" with it. An underscore warehouse's
dimension (ITM on ITM_DMS) was already read this way; a table named Dim... or
DIM_... was not.

A column of a dimension named with a dimension affix -- Dim, DIM_, _DIM, or
one the vocabulary declares -- now answers to the rest of its name after the
dimension's entity, and to the entity's head noun with it: "country",
"territory country". A lone word is thin evidence -- "line" is a product's
line and also an order line -- so it answers only where the question asks by
it ("by country", "for each region", "which group"). The name must read with
confidence, the rest must say something on its own ("name" does not), and a
key never answers: it joins.

tests/star_harness.py keeps a sales territory named in camel case, with its
region, country and group.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("rest-of-name")) as built:
        yield built


def _sales_by(position: int, year: int | None = None) -> dict:
    totals: dict = {}
    for line in star.orders():
        if year is None or line[2].year == year:
            name = star.TERRITORIES[line[6]][position]
            totals[name] = totals.get(name, 0) + line[7] * star.PRODUCTS[line[4]][4]
    return totals


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [("Sales by country", "en"), ("Ventes par pays", "fr")])
    def test_sales_by_country(self, warehouse, question, lang):
        answer = star.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert {row["COUNTRY"]: row["SALES_AMOUNT"] for row in answer["rows"]} == _sales_by(1)

    def test_sales_by_region_in_a_year(self, warehouse):
        answer = star.ask(warehouse, "Sales by region in 2025")
        assert {row["REGION"]: row["SALES_AMOUNT"] for row in answer["rows"]} == _sales_by(0, 2025)

    def test_sales_by_territory_group(self, warehouse):
        answer = star.ask(warehouse, "Sales by territory group")
        assert {row["TERRITORY_GROUP"]: row["SALES_AMOUNT"] for row in answer["rows"]} == _sales_by(2)


class TestTheRule:

    @pytest.mark.parametrize("table,entity", [
        ("SalesDW.dbo.DimSalesTerritory", "SALES_TERRITORY"),
        ("DIM_CUSTOMER", "CUSTOMER"),
        ("CUSTOMER_DIM", "CUSTOMER"),
        ("FactInternetSales", ""),
        ("Dim", ""),
    ])
    def test_the_entity_a_dimension_is_named_for(self, table, entity):
        from core.semantic_planner import _dimension_entity

        assert _dimension_entity(table) == entity

    def test_an_affix_the_vocabulary_declares(self):
        import re
        from types import SimpleNamespace

        from core.semantic_planner import _dimension_entity

        vocab = SimpleNamespace(dimension_patterns=[re.compile(r"^LKP_"), re.compile(r"_REF")])
        assert _dimension_entity("LKP_REGION", vocab) == "REGION"
        assert _dimension_entity("LKP_REGION") == ""
        # An affix is at an end of the name, never inside it.
        assert _dimension_entity("SALES_REF_REGION", vocab) == ""

    @pytest.mark.parametrize("table,column,question,phrases", [
        ("dbo.DimSalesTerritory", "SalesTerritoryCountry", "sales by country", {"country", "territory country"}),
        ("dbo.DimSalesTerritory", "SalesTerritoryCountry", "sales for each country", {"country", "territory country"}),
        # Used, not asked by: the lone word is not the column's.
        ("dbo.DimSalesTerritory", "SalesTerritoryCountry", "sales in the country", {"territory country"}),
        ("dbo.DimProduct", "ProductLine", "how many order lines did we have", {"product line"}),
        ("dbo.DimSalesTerritory", "SalesTerritoryGroup", "sales by territory group", {"territory group"}),
        ("DIM_CUSTOMER", "CUSTOMER_CITY", "sales by city", {"city", "customer city"}),
        ("dbo.DimCustomer", "CustomerCommuteDistance", "customers by commute distance",
         {"commute distance", "customer commute distance"}),
        # A key joins; a generic word says nothing; a compact code is not read.
        ("dbo.DimSalesTerritory", "SalesTerritoryKey", "sales by key", set()),
        ("dbo.DimSalesTerritory", "SalesTerritoryAlternateKey", "sales by alternate key", set()),
        ("DIM_CUSTOMER", "CUSTOMER_REGION_ID", "customers by region id", set()),
        ("dbo.DimSalesTerritory", "SalesTerritoryName", "sales by name", set()),
        ("DIM_CUSTOMER", "CUSTOMER_XQZ", "sales by xqz", set()),
        # Not a dimension's own column.
        ("dbo.DimSalesTerritory", "CountryCode", "sales by country code", set()),
    ])
    def test_the_rest_of_a_name(self, table, column, question, phrases):
        from core.semantic_planner import _aliases_within_entity

        assert _aliases_within_entity(table, column, question_norm=question) == phrases

    def test_an_underscore_dimension_is_read_as_before(self):
        from core.semantic_planner import _aliases_within_entity

        assert _aliases_within_entity("ITM_DMS", "ITM_GRS_WT") == {"gross weight"}
        assert _aliases_within_entity("ITM_DMS", "ITM_NM") == set()
