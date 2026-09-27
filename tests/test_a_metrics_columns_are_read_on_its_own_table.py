"""
A metric's columns are read on its own table.

A fact and the dimensions it is broken down by often share their keys' names:
the sales fact's ProductKey is the product table's ProductKey, its CustomerKey
the customer table's. A metric that counts one of them -- "Number of Products
Sold" is COUNT(DISTINCT ProductKey) -- was written into the governed query as
its formula stands, beside the join to the table the question breaks it down
by, and the warehouse refused it: "Ambiguous reference to column name
ProductKey". "Number of products sold by product category in 2025" failed
after every other step had been settled.

In the compiled query a metric's bare references to its own table's columns
are read on that table: COUNT(DISTINCT fact_rows.ProductKey). A reference the
formula already qualifies, a function's name and a string are left as written.

tests/star_harness.py keeps two years of order lines and the metrics "Number
of Products Sold" (COUNT(DISTINCT ProductKey)) and "Number of Buying
Customers" (COUNT(DISTINCT CustomerKey)).
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("own-columns")) as built:
        yield built


def _category(product: int) -> str:
    return star.CATEGORIES[star.SUBCATEGORIES[star.PRODUCTS[product][3]][2]][0]


class TestTheProductAnswers:

    def test_products_sold_by_category(self, warehouse):
        answer = star.ask(warehouse, "Number of products sold by product category in 2025")
        assert answer["model_wrote_sql"] is False
        sold: dict[str, set[int]] = {}
        for line in star.orders():
            if line[2].year == 2025:
                sold.setdefault(_category(line[4]), set()).add(line[4])
        assert {row["PRODUCT_CATEGORY"]: row["NUMBER_OF_PRODUCTS_SOLD"] for row in answer["rows"]} == {
            category: len(products) for category, products in sold.items()}

    def test_buying_customers_by_gender(self, warehouse):
        answer = star.ask(warehouse, "Number of buying customers by gender")
        assert answer["model_wrote_sql"] is False
        buyers: dict[str, set[int]] = {}
        for line in star.orders():
            buyers.setdefault(star.CUSTOMERS[line[5]][3], set()).add(line[5])
        assert {row["GENDER"]: row["NUMBER_OF_BUYING_CUSTOMERS"] for row in answer["rows"]} == {
            gender: len(customers) for gender, customers in buyers.items()}


_FACT = {"PRODUCTKEY": "int", "SALESAMOUNT": "money", "ORDERQUANTITY": "int", "STATUS": "nvarchar"}


class TestTheRule:

    @pytest.mark.parametrize("formula,read", [
        ("COUNT(DISTINCT ProductKey)", "COUNT(DISTINCT fact_rows.ProductKey)"),
        ("SUM(SalesAmount) / NULLIF(SUM(OrderQuantity), 0)",
         "SUM(fact_rows.SalesAmount) / NULLIF(SUM(fact_rows.OrderQuantity), 0)"),
        ("SUM([SalesAmount])", "SUM(fact_rows.[SalesAmount])"),
        ("SUM(f.SalesAmount)", "SUM(f.SalesAmount)"),
        ("SUM(CASE WHEN Status = 'SalesAmount' THEN SalesAmount END)",
         "SUM(CASE WHEN fact_rows.Status = 'SalesAmount' THEN fact_rows.SalesAmount END)"),
        ("COUNT(*)", "COUNT(*)"),
        ("SUM(Freight)", "SUM(Freight)"),
    ])
    def test_a_formula_read_on_its_table(self, formula, read):
        from core.pipeline_helpers import formula_read_on

        assert formula_read_on(formula, _FACT) == read

    def test_a_column_named_like_a_function(self):
        from core.pipeline_helpers import formula_read_on

        assert formula_read_on("SUM(Total) + Total", {"TOTAL": "int", "SUM": "int"}) == (
            "SUM(fact_rows.Total) + fact_rows.Total")
