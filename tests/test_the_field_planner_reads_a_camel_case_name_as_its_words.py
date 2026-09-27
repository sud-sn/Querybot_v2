"""
The field planner reads a camel-case name as its words.

A camel-case warehouse was taught to be read as its words (the semantic
model reads ProductKey as a key and SalesAmount as an amount), but the field
planner -- which decides which column a question's words name, and what each
column is -- read every name after it had been upper-cased. PRODUCTKEY has no
_KEY to find and no humps to split, so a product's key was typed a quantity
to add up; SALESTERRITORYCOUNTRY answered to nothing but itself. And SQL
Server's money was not a number: a list price or a yearly income was an
attribute, never what an average reads. "Average list price by product
category" asked the reader which inventory dataset to use; "Average yearly
income by gender" was refused for want of a measure.

The planner now reads each name as the warehouse spells it
(core.schema.load_schema_spellings): its phrases, its entity and its role
come from its words, while the column it emits stays the upper-case name
every later step compares. Money, real and double are numbers. An
underscore warehouse's names are their own words, and read as before.

tests/star_harness.py keeps a product's list price and a customer's yearly
income, in money, and every name in camel case.
"""

from __future__ import annotations

import json

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("camel-planner")) as built:
        yield built


def _category(product: int) -> str:
    return star.CATEGORIES[star.SUBCATEGORIES[star.PRODUCTS[product][3]][2]][0]


def _average(pairs) -> dict:
    groups: dict = {}
    for key, value in pairs:
        groups.setdefault(key, []).append(value)
    return {key: sum(values) / len(values) for key, values in groups.items()}


class TestTheProductAnswers:

    def test_average_list_price_by_product_category(self, warehouse):
        answer = star.ask(warehouse, "Average list price by product category")
        assert answer["model_wrote_sql"] is False
        assert {row["PRODUCT_CATEGORY"]: row["AVERAGE_LIST_PRICE"] for row in answer["rows"]} == _average(
            (_category(product), row[4]) for product, row in star.PRODUCTS.items())

    def test_average_yearly_income_by_gender(self, warehouse):
        answer = star.ask(warehouse, "Average yearly income by gender")
        assert answer["model_wrote_sql"] is False
        assert {row["GENDER"]: row["AVERAGE_YEARLY_INCOME"] for row in answer["rows"]} == _average(
            (row[3], row[4]) for row in star.CUSTOMERS.values())


class TestTheRule:

    @pytest.mark.parametrize("column,declared,role", [
        ("ProductKey", "int", "dimension"),
        ("ProductSubcategoryKey", "int", "dimension"),
        ("OrderDateKey", "int", "date_key"),
        ("SalesAmount", "money", "measure"),
        ("ListPrice", "money", "measure"),
        ("UnitCost", "smallmoney", "measure"),
        ("Weight", "real", "measure"),
        ("Color", "nvarchar", "attribute"),
        # An underscore warehouse, as before.
        ("ON_HND_QTY", "decimal", "measure"),
        ("WHS_DMS_KEY", "int", "dimension"),
        ("ITM_NM", "varchar", "attribute"),
    ])
    def test_a_columns_role(self, column, declared, role):
        from core.semantic_planner import _role_for_column

        assert _role_for_column(column, declared) == role

    @pytest.mark.parametrize("column,phrases", [
        ("SalesTerritoryCountry", {"sales territory country"}),
        # A word no lexicon knows is still a word.
        ("BinLocation", {"bin location"}),
        ("EnglishProductName", {"english product name", "english product"}),
        ("SalesAmount", {"sales amount", "sales"}),
        ("WAREHOUSE_NAME", {"warehouse name", "warehouse"}),
    ])
    def test_the_phrases_a_column_answers_to(self, column, phrases):
        from core.semantic_planner import _aliases_for_column

        assert phrases <= _aliases_for_column(column)

    @pytest.mark.parametrize("column", ["LastName", "FirstName", "LAST_NAME", "MiddleName"])
    def test_a_part_of_a_persons_name_labels_no_entity(self, column):
        # "Sales by due month over the last six months" is not by surname.
        from core.semantic_planner import _aliases_for_column

        assert not {"last", "first", "middle"} & _aliases_for_column(column)

    def test_the_plan_reads_the_warehouses_spelling(self):
        from core.semantic_planner import build_semantic_field_plan

        columns = {"DBO.FACTSALES": {"PRODUCTKEY": "int", "ORDERQUANTITY": "smallint"}}
        spelled = {"PRODUCTKEY": "ProductKey", "ORDERQUANTITY": "OrderQuantity"}

        def roles(**kwargs):
            plan = build_semantic_field_plan("order quantity by product key", columns, **kwargs)
            return {field["column"]: field["role"] for field in plan["fields"]}

        assert roles(spellings=spelled) == {"PRODUCTKEY": "dimension", "ORDERQUANTITY": "measure"}
        assert roles()["PRODUCTKEY"] == "measure"

    def test_the_spellings_a_warehouse_keeps(self, tmp_path):
        from core.schema import load_schema_spellings

        schema = {"SalesDW.dbo.DimSalesTerritory": {"columns": [
            {"name": "SalesTerritoryKey", "type": "int"}, {"name": "SalesTerritoryCountry", "type": "nvarchar"}]},
            "__fk_constraints": []}
        (tmp_path / "_schema.json").write_text(json.dumps(schema), encoding="utf-8")
        assert load_schema_spellings(str(tmp_path)) == {
            "SALESDW.DBO.DIMSALESTERRITORY": "SalesDW.dbo.DimSalesTerritory",
            "DBO.DIMSALESTERRITORY": "dbo.DimSalesTerritory",
            "DIMSALESTERRITORY": "DimSalesTerritory",
            "SALESTERRITORYKEY": "SalesTerritoryKey",
            "SALESTERRITORYCOUNTRY": "SalesTerritoryCountry",
        }
        assert load_schema_spellings(str(tmp_path / "missing")) == {}
