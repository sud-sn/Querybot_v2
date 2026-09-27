"""
Every product is counted, not the ones sold.

"How many products do we have?" was refused on a warehouse that keeps a
metric Number of Products Sold, and so was "how many customers do we have"
beside Number of Buying Customers. The product knew to count a population on
the table that defines it -- the product table -- but the question's words
matched the metric too, and the metric reads the sales table: with it in
scope, a count on the product table was refused for reading another table
than the metric's, and the model's attempt was refused for the same reason.
"How many products do we have in each category?" was refused as well: the
breakdown the question asks for is the product category, and a count kept
only a dimension named "category" whole, setting "Product Category" aside as
a modifier of the count.

A population count now leaves out the metrics that read any table but the
population's own -- they are about what its members did, not how many there
are -- and a count keeps a dimension the question names by the last words of
its name. "How many products did we sell" is still the metric's.

tests/star_harness.py keeps eight products in three categories, four
customers, and the metrics Number of Products Sold and Number of Buying
Customers, both on the sales table.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("population")) as built:
        yield built


def _products_by_category(lang: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _code, _name, _french, subcategory, *_prices in star.PRODUCTS.values():
        english, french = star.CATEGORIES[star.SUBCATEGORIES[subcategory][2]]
        name = french if lang == "fr" else english
        counts[name] = counts.get(name, 0) + 1
    return counts


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang,table,count", [
        ("How many products do we have?", "en", "DIMPRODUCT", len(star.PRODUCTS)),
        ("Combien de produits avons-nous ?", "fr", "DIMPRODUCT", len(star.PRODUCTS)),
        ("How many customers do we have?", "en", "DIMCUSTOMER", len(star.CUSTOMERS)),
    ])
    def test_the_whole_population(self, warehouse, question, lang, table, count):
        answer = star.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert table in answer["sql"].upper() and "FACTINTERNETSALES" not in answer["sql"].upper()
        assert [list(row.values()) for row in answer["rows"]] == [[count]]

    @pytest.mark.parametrize("question,lang", [
        ("How many products do we have in each category?", "en"),
        ("Combien de produits avons-nous par catégorie ?", "fr"),
    ])
    def test_the_population_in_each_category(self, warehouse, question, lang):
        answer = star.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert {row["PRODUCT_CATEGORY"]: row["PRODUCT_COUNT"] for row in answer["rows"]} == _products_by_category(lang)

    def test_the_model_is_not_handed_the_products_sold(self, warehouse):
        # Two breakdowns are the model's to write; the metric is no part of it.
        answer = star.ask(warehouse, "How many products do we have in each subcategory and category?")
        assert answer["prompts"]
        assert not any("Number of Products Sold" in prompt for prompt in answer["prompts"])

    def test_the_products_sold_are_still_the_metrics(self, warehouse):
        answer = star.ask(warehouse, "How many products did we sell in 2025?")
        assert "FACTINTERNETSALES" in answer["sql"].upper()
        assert [list(row.values()) for row in answer["rows"]] == [
            [len({line[4] for line in star.orders() if line[2].year == 2025})]]


_SOLD = {"name": "Number of Products Sold", "_resolved_source_tables": ["DBO.FACTINTERNETSALES"]}
_LISTED = {"name": "Products Listed", "_resolved_source_tables": ["DBO.DIMPRODUCT"]}
_ACROSS = {"name": "Products With Stock", "_resolved_source_tables": ["DBO.DIMPRODUCT", "DBO.FACTPRODUCTINVENTORY"]}
_UNPLACED = {"name": "Draft"}


def _population_plan() -> dict:
    return {"count_target": {"status": "selected", "selected": {"table": "dbo.DimProduct", "column": "ProductKey"}}}


class TestTheRule:

    def test_a_metric_on_another_table_is_left_out(self):
        from core.analytical_request_plan import metrics_on_the_population

        chosen = metrics_on_the_population(
            _population_plan(), {"population_entity": "product"}, [_SOLD, _LISTED, _ACROSS, _UNPLACED])
        assert [metric["name"] for metric in chosen] == ["Products Listed", "Draft"]

    def test_a_metric_named_by_its_source_tables(self):
        from core.analytical_request_plan import metrics_on_the_population

        metrics = [{"name": "Sold", "source_tables": ["dbo.FactInternetSales"]},
                   {"name": "Listed", "base_table": "dbo.DimProduct"},
                   {"name": "Shipped", "base_table": "dbo.FactInternetSales"}]
        assert [m["name"] for m in metrics_on_the_population(
            _population_plan(), {"population_entity": "product"}, metrics)] == ["Listed"]

    @pytest.mark.parametrize("plan,intent", [
        # Not a population count: a count of what members did.
        (_population_plan(), {"population_entity": ""}),
        # No table was found to define the population.
        ({}, {"population_entity": "product"}),
    ])
    def test_otherwise_the_metrics_are_as_they_were(self, plan, intent):
        from core.analytical_request_plan import metrics_on_the_population

        assert metrics_on_the_population(plan, intent, [_SOLD, _LISTED]) == [_SOLD, _LISTED]

    @pytest.mark.parametrize("term,kept", [
        ("Product Category", True),
        ("category", True),
        ("Subcategory", False),
        ("Category Manager", False),
        ("Customer", False),
    ])
    def test_a_dimension_named_by_the_last_words_of_its_name(self, term, kept):
        from core.pipeline_context import _scope_semantic_plan_to_analytical_request

        plan = {**_population_plan(), "fields": [
            {"term": term, "table": "dbo.DimProductCategory", "column": "Name",
             "role": "display_dimension", "enforcement": "required"}]}
        _scope_semantic_plan_to_analytical_request(
            plan, {"measure_semantics": "count_distinct_business_identifier", "dimensions": ["category"]})
        assert (plan["fields"][0]["enforcement"] == "required") is kept
