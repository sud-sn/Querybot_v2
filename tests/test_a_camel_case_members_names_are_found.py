"""
A camel-case member's names are found.

"Sales of Summit Tent in 2025" was answered with every product's sales: a
total of 27,225 where the tent sold 8,540. The question names a product, and
a member the question names is filtered on -- when the value index knows it.
It did not: the index keeps the names of dimension tables, and a dimension
table was one spelled DIM_PRODUCT or ITEM_DIM, never DimProduct. The
product's name was no member, the question named none, and the governed
compiler answered for every product. A warehouse named in camel case kept its
places out of the index the same way: "WarehouseCity" read as one word held
no CITY.

A table's and a column's names are read as their words: DimProduct is a
dimension table, EnglishProductName its product's name, and
WarehouseCity a city. A product the question names is a member.

tests/star_harness.py keeps eight products in a table named DimProduct.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("camel-members")) as built:
        yield built


def _sales_2025() -> float:
    return sum(line[7] * star.PRODUCTS[line[4]][4] for line in star.orders() if line[2].year == 2025)


class TestTheProductAnswers:

    def test_a_product_the_question_names_is_not_every_product(self, warehouse):
        answer = star.ask(warehouse, "Sales of Summit Tent in 2025")
        assert _sales_2025() not in [value for row in answer["rows"] for value in row.values()]

    def test_the_product_is_a_member(self, warehouse):
        from core.value_resolver import resolve_literals

        verified = resolve_literals(star.ACCOUNT, "Sales of Summit Tent in 2025")["verified"]
        assert [(item["column"], item["value"]) for item in verified] == [("EnglishProductName", "Summit Tent")]


def _columns(table: str, *columns: tuple[str, str]) -> dict:
    return {f"SalesDW.dbo.{table}": {"columns": [{"name": name, "type": kind} for name, kind in columns]}}


class TestTheRule:

    @pytest.mark.parametrize("table,dimension", [
        ("DimProduct", True),
        ("DIM_PRODUCT", True),
        ("ITM_DMS", True),
        ("FactInternetSales", False),
        # One word is no dimension of anything.
        ("DIMPRODUCT", False),
    ])
    def test_a_dimension_table_by_its_words(self, table, dimension):
        from core.value_index import _is_dimension_table

        assert _is_dimension_table(table) is dimension

    def test_the_columns_worth_indexing(self):
        from core.value_index import select_filterable_columns

        schema = {
            **_columns("DimProduct", ("EnglishProductName", "nvarchar"), ("ProductKey", "int")),
            **_columns("DimSalesTerritory", ("SalesTerritoryCountry", "nvarchar")),
            **_columns("DimWarehouse", ("WarehouseCity", "nvarchar")),
            **_columns("FactInternetSales", ("SalesOrderNumber", "nvarchar")),
        }
        assert sorted(column["column"] for column in select_filterable_columns(schema)) == [
            "EnglishProductName", "SalesTerritoryCountry", "WarehouseCity"]
