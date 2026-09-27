"""
A member the question names filters the answer.

"Sales of Climbing products in 2025", "sales of Summit Tent in 2025", "how
many products are in Climbing?" -- a question that names a member was left
to the model, every time: the governed compilers wrote no filter on a value,
so they stood aside for the planner, which is told the value the index
found. Where the planner is a stand-in, or strays, the question named the
one thing it was about and was answered by guesswork.

A member the value index finds as the value of one column -- or of a label
and its twin in another language, "Camping" in a category's English and
French names -- is now written as a filter by the grouped compiler, its
table joined along the governed path as a breakdown's is. A single total
filtered so is told apart from no rows: beside it, how many rows matched
and how many held a value, and a total of none is 0. A breakdown by the
member's own column is the member ("sales by product for the Climbing
category"), and a population the member narrows is counted on its own
table: "how many products are in Climbing" counts the category's products,
not the products sold there. A member named with the period beside it --
"Summit Tent 2025" -- is the member. Two members of one column, a value the
index found in two columns, or one it could not find whole are still the
planner's.

tests/star_harness.py keeps eight products in six subcategories of three
categories, named in English and French, and two years of sales.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("member-filter")) as built:
        yield built


def _category(product: tuple, lang: int = 0) -> str:
    return star.CATEGORIES[star.SUBCATEGORIES[product[3]][2]][lang]


def _sales_2025(keep, name: int = 1) -> dict[str, float]:
    """2025 sales of the products ``keep`` keeps, by name."""
    sales: dict[str, float] = {}
    for line in star.orders():
        product = star.PRODUCTS[line[4]]
        if line[2].year == 2025 and keep(product):
            sales[product[name]] = sales.get(product[name], 0) + line[7] * product[4]
    return sales


def _by_product(answer: dict) -> dict[str, float]:
    assert answer["model_wrote_sql"] is False
    return {row["PRODUCT"]: row["SALES_AMOUNT"] for row in answer["rows"]}


class TestTheProductAnswers:

    def test_one_products_sales(self, warehouse):
        answer = star.ask(warehouse, "Sales of Summit Tent in 2025")
        assert answer["model_wrote_sql"] is False
        assert [row["SALES_AMOUNT"] for row in answer["rows"]] == list(
            _sales_2025(lambda product: product[1] == "Summit Tent").values())

    @pytest.mark.parametrize("question", [
        "Sales of Climbing products in 2025",
        "Sales by product for the Climbing category in 2025",
    ])
    def test_a_categorys_products(self, warehouse, question):
        assert _by_product(star.ask(warehouse, question)) == _sales_2025(
            lambda product: _category(product) == "Climbing")

    def test_a_member_its_french_label_keeps_too(self, warehouse):
        assert _by_product(star.ask(warehouse, "Sales of Camping products in 2025")) == _sales_2025(
            lambda product: _category(product) == "Camping")

    def test_in_french(self, warehouse):
        assert _by_product(star.ask(warehouse, "Ventes des produits Escalade en 2025", "fr")) == _sales_2025(
            lambda product: _category(product, 1) == "Escalade", name=2)

    @pytest.mark.parametrize("question", [
        "How many products are in Climbing?",
        "How many products are in the Climbing category?",
    ])
    def test_the_products_a_category_keeps(self, warehouse, question):
        answer = star.ask(warehouse, question)
        assert answer["model_wrote_sql"] is False
        assert "FACTINTERNETSALES" not in answer["sql"].upper()
        assert [row["PRODUCT_COUNT"] for row in answer["rows"]] == [
            sum(1 for product in star.PRODUCTS.values() if _category(product) == "Climbing")]

    @pytest.mark.parametrize("question", [
        # A country the territory's region and country both keep.
        "Sales in France in 2025",
        # Two products, either or both.
        "Sales of Summit Tent and Ridge Tent in 2025",
    ])
    def test_still_the_planners(self, warehouse, question):
        assert star.ask(warehouse, question)["model_wrote_sql"] is True


def _context(named: list[str], verified: list[dict]) -> dict:
    return {"named_members": named, "verified_members": verified}


_CLIMBING = {"phrase": "Climbing", "table": "dbo.DimProductCategory", "column": "EnglishProductCategoryName",
             "value": "Climbing"}
_TENT = {"phrase": "Summit Tent", "table": "dbo.DimProduct", "column": "EnglishProductName", "value": "Summit Tent"}
_RIDGE = {"phrase": "Ridge Tent", "table": "dbo.DimProduct", "column": "EnglishProductName", "value": "Ridge Tent"}


class TestTheRule:

    @pytest.mark.parametrize("context,filters", [
        (_context([], []), []),
        (_context(["Climbing"], [_CLIMBING]), [_CLIMBING]),
        (_context(["Climbing", "Summit Tent"], [_CLIMBING, _TENT]), [_CLIMBING, _TENT]),
        # Named, but not found as one column's value.
        (_context(["France"], []), None),
        (_context(["Climbing", "France"], [_CLIMBING]), None),
        # Two members of one column.
        (_context(["Summit Tent", "Ridge Tent"], [_TENT, _RIDGE]), None),
    ])
    def test_the_filters(self, context, filters):
        from core.pipeline_helpers import verified_member_filters

        assert verified_member_filters(context) == filters

    def test_a_member_a_label_and_its_twin_keep(self):
        from core.query_pipeline import _verified_members

        resolved = {"several": [
            {"phrase": "Camping", "value": "Camping", "columns": [
                {"table_fqn": "dbo.DimProductCategory", "column": "EnglishProductCategoryName"},
                {"table_fqn": "dbo.DimProductCategory", "column": "FrenchProductCategoryName"}]},
            {"phrase": "France", "value": "France", "columns": [
                {"table_fqn": "dbo.DimSalesTerritory", "column": "SalesTerritoryRegion"},
                {"table_fqn": "dbo.DimSalesTerritory", "column": "SalesTerritoryCountry"}]},
        ]}
        assert _verified_members(resolved) == [{"phrase": "Camping", "table": "dbo.DimProductCategory",
                                                "column": "EnglishProductCategoryName", "value": "Camping"}]

    @pytest.mark.parametrize("question,population", [
        ("How many products are in Climbing?", "product"),
        ("How many products are in the Climbing category?", "product"),
        ("how many products do we have in Climbing", "product"),
        # The member narrows something else.
        ("How many products are in Climbing stores?", ""),
        ("Sales in Climbing", ""),
    ])
    def test_a_population_its_members_narrow(self, question, population):
        from core.analytical_intent import population_in_members

        assert population_in_members(question, [_CLIMBING]) == population

    def test_a_member_named_beside_the_period(self):
        from core.query_pipeline import _without_the_stated_period

        resolved = {"verified": [{"phrase": "Summit Tent"}],
                    "narrowed": [{"phrase": "Summit Tent 2025", "dropped": ["2025"]},
                                 {"phrase": "Summit Tents Large", "dropped": ["large"]}]}
        kept = _without_the_stated_period(resolved, "sales of summit tent in 2025")
        assert [item["phrase"] for item in kept["narrowed"]] == ["Summit Tents Large"]

    @pytest.mark.parametrize("expression,safe", [
        ("SUM(fact_rows.SalesAmount)", "COALESCE(SUM(fact_rows.SalesAmount), 0)"),
        ("SUM(a) - SUM(b * (1 - c))", "COALESCE(SUM(a), 0) - COALESCE(SUM(b * (1 - c)), 0)"),
        ("COUNT(DISTINCT k)", "COUNT(DISTINCT k)"),
    ])
    def test_a_total_of_none_is_zero(self, expression, safe):
        from core.pipeline_helpers import _null_safe_sums

        assert _null_safe_sums(expression) == safe

    @pytest.mark.parametrize("db_type,literal", [("azure_sql", "N'Men''s Jacket'"), ("snowflake", "'Men''s Jacket'")])
    def test_a_members_literal(self, db_type, literal):
        from core.pipeline_helpers import _member_literal

        assert _member_literal("Men's Jacket", db_type) == literal
