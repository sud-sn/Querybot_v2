"""
The members a count counts are not its breakdown.

"How many products did we sell in 2025?" is read with "Number of Products
Sold", COUNT(DISTINCT ProductKey) on the sales. The planner also bound
"products" to the product's name, through the same key, and the answer was
grouped by it: one row per product, each counting one. And the join graph,
reading "products" again from the question, required the product table, so a
question whose words bound nothing else was refused for a join the answer
never needed.

Where the question is a count -- not a ranking or a lookup of the members, and
not broken down "by product" -- a field the question binds through the very
key a matched metric counts names the members counted: it is set aside, and
its table is no join the question requires. "Top 3 products by sales" and
"number of products sold by product" still read the product.

tests/star_harness.py keeps two years of order lines and the metric "Number of
Products Sold" (synonym "products sold").
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("counted-members")) as built:
        yield built


def _sold(year: int) -> set[int]:
    return {line[4] for line in star.orders() if line[2].year == year}


class TestTheProductAnswers:

    def test_how_many_products_were_sold(self, warehouse):
        answer = star.ask(warehouse, "How many products did we sell in 2025?")
        assert answer["model_wrote_sql"] is False
        assert "DIMPRODUCT" not in answer["sql"].upper()
        assert [list(row.values()) for row in answer["rows"]] == [[len(_sold(2025))]]

    def test_by_product_is_by_product(self, warehouse):
        answer = star.ask(warehouse, "Number of products sold by product in 2025")
        assert len(answer["rows"]) == len(_sold(2025))
        assert all(row["NUMBER_OF_PRODUCTS_SOLD"] == 1 for row in answer["rows"])

    def test_a_ranking_of_the_members_reads_them(self, warehouse):
        answer = star.ask(warehouse, "Top 3 products by sales in 2025")
        assert len(answer["rows"]) == 3 and all(row.get("PRODUCT") for row in answer["rows"])


def _plan() -> dict:
    return {"fields": [
        {"term": "Product", "table": "dbo.DimProduct", "column": "EnglishProductName", "role": "display_dimension",
         "source_key_column": "ProductKey", "enforcement": "required"},
        {"term": "product category", "table": "dbo.DimProductCategory", "column": "EnglishProductCategoryName",
         "role": "display_dimension", "source_key_column": "ProductCategoryKey", "enforcement": "required"},
    ]}


_SOLD = {"name": "Number of Products Sold", "sql_template": "COUNT(DISTINCT ProductKey)",
         "base_table": "dbo.FactSales"}


class TestTheRule:

    @pytest.mark.parametrize("formula,keys", [
        ("COUNT(DISTINCT ProductKey)", {"PRODUCTKEY"}),
        ("count( distinct f.[CustomerKey] )", {"CUSTOMERKEY"}),
        ("COUNT(DISTINCT ProductKey) / 2", set()),
        ("SUM(ProductKey)", set()),
        ("COUNT(*)", set()),
    ])
    def test_the_key_a_metric_counts(self, formula, keys):
        from core.analytical_request_plan import counted_keys

        assert counted_keys([{"sql_template": formula}]) == keys

    def test_a_count_sets_its_members_aside(self):
        from core.analytical_request_plan import COUNTED_MEMBERS, demote_counted_members

        plan = _plan()
        demote_counted_members(plan, {"intent": "metric_query", "dimensions": ["product category"]}, [_SOLD])
        assert [(f["term"], f["enforcement"], f.get("demotion_reason")) for f in plan["fields"]] == [
            ("Product", "optional", COUNTED_MEMBERS), ("product category", "required", None)]

    @pytest.mark.parametrize("intent", [
        {"intent": "ranking", "dimensions": []},
        {"intent": "entity_lookup", "dimensions": []},
        {"intent": "metric_query", "dimensions": ["product"]},
    ])
    def test_a_ranking_a_lookup_or_a_breakdown_keeps_them(self, intent):
        from core.analytical_request_plan import demote_counted_members

        plan = _plan()
        demote_counted_members(plan, intent, [_SOLD])
        assert [f["enforcement"] for f in plan["fields"]] == ["required", "required"]

    def test_a_metric_that_counts_another_key(self):
        from core.analytical_request_plan import demote_counted_members

        plan = _plan()
        demote_counted_members(plan, {"intent": "metric_query", "dimensions": []},
                               [{**_SOLD, "sql_template": "COUNT(DISTINCT SalesOrderNumber)"}])
        assert [f["enforcement"] for f in plan["fields"]] == ["required", "required"]

    def test_the_graph_leaves_an_excluded_entity_out(self):
        from core.graph_resolver import detect_entities

        graph = {"entities": [
            {"entity_name": "FactSales", "table_name": "FactSales", "schema_name": "dbo", "entity_type": "fact",
             "display_name": "Sales"},
            {"entity_name": "DimProduct", "table_name": "DimProduct", "schema_name": "dbo",
             "entity_type": "dimension", "display_name": "Product"}], "relationships": []}
        assert "DimProduct" in detect_entities("how many products did we sell", graph)
        assert "DimProduct" not in detect_entities("how many products did we sell", graph,
                                                   excluded_entities={"DimProduct"})
        assert "DimProduct" in detect_entities("how many products did we sell", graph,
                                               required_entities={"DimProduct"}, excluded_entities={"DimProduct"})
