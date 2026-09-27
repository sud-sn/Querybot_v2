"""
Members are ranked by their own attribute.

"Top 3 products by list price", "which product has the highest list price?",
"top 3 customers by yearly income" -- each was answered with a question:
"What measure should I use to rank them?" A ranking was read for a
registered metric alone, and the list price and the yearly income are
columns the warehouse keeps, not metrics anyone registered. Asked anyway,
the product then matched "products" to Number of Products Sold, read the
sales table, and asked which dataset to use.

A ranking that names a measure the warehouse keeps is not asked which
measure. Members ranked by an attribute their own table keeps -- one value
each -- are read on that table, ranked by it and shown beside it; a count
matched by the members' name alone is not asked for. "Top 3 products" with
no measure still asks which, and "top 3 products by sales" is the sales
metric's.

tests/star_harness.py keeps eight products with a list price and four
customers with a yearly income.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("ranked-attribute")) as built:
        yield built


def _by_price() -> list[tuple[str, float]]:
    return sorted(((english, price) for _code, english, _french, _sub, price, _cost in star.PRODUCTS.values()),
                  key=lambda product: -product[1])


class TestTheProductAnswers:

    def test_the_top_products_by_list_price(self, warehouse):
        answer = star.ask(warehouse, "Top 3 products by list price")
        assert answer["model_wrote_sql"] is False
        assert [(row["PRODUCT"], row["LIST_PRICE"]) for row in answer["rows"]] == _by_price()[:3]

    def test_the_product_with_the_lowest_list_price(self, warehouse):
        answer = star.ask(warehouse, "Which product has the lowest list price?")
        assert (answer["rows"][0]["PRODUCT"], answer["rows"][0]["LIST_PRICE"]) == _by_price()[-1]

    def test_the_top_customers_by_yearly_income(self, warehouse):
        answer = star.ask(warehouse, "Top 3 customers by yearly income")
        ranked = sorted(star.CUSTOMERS.values(), key=lambda customer: -customer[4])[:3]
        assert [(row["CUSTOMER_ID"], row["YEARLY_INCOME"]) for row in answer["rows"]] == [
            (code, income) for code, _first, _last, _gender, income in ranked]

    def test_a_ranking_with_no_measure_still_asks(self, warehouse):
        answer = star.ask(warehouse, "Top 3 products")
        assert answer["executed"] == []
        assert "What measure should I use to rank them?" in str(answer["replies"])

    def test_a_ranking_by_a_metric_is_the_metrics(self, warehouse):
        answer = star.ask(warehouse, "Top 3 products by sales in 2025")
        assert "FACTINTERNETSALES" in answer["sql"].upper()


_COLUMNS = {f"dbo.{name.split('.')[-1]}": {column["name"]: column["type"] for column in table["columns"]}
            for name, table in star.SCHEMA.items()}
_MODEL = {"tables": [{"qualified_name": "dbo.DimProduct", "type": "dimension"},
                     {"qualified_name": "dbo.DimProductCategory", "type": "dimension"}]}
_PRICE = {"term": "list price", "table": "dbo.DimProduct", "column": "LISTPRICE", "role": "measure"}
_PRODUCT = {"term": "Product", "table": "dbo.DimProduct", "column": "EnglishProductName", "role": "display_dimension"}
_CATEGORY = {"term": "Product Category", "table": "dbo.DimProductCategory", "column": "Name",
             "role": "display_dimension"}
_SOLD = {"name": "Number of Products Sold", "synonyms": "products sold", "sql_template": "COUNT(DISTINCT ProductKey)"}


class TestTheRule:

    @pytest.mark.parametrize("question,named", [
        ("top 3 products by list price", True),
        ("which product has the highest list price", True),
        ("top 3 products", False),
    ])
    def test_a_measure_the_question_names(self, question, named):
        from core.semantic_planner import names_a_measure

        assert names_a_measure(question, _COLUMNS) is named

    def test_members_ranked_by_their_attribute(self):
        from core.analytical_request_plan import aggregated_attribute

        ranked = aggregated_attribute("top 3 products by list price", {"fields": [_PRICE, _PRODUCT]}, _MODEL)
        assert (ranked["aggregation"], ranked["target_column"], ranked.get("ranked")) == ("MAX", "LISTPRICE", True)

    @pytest.mark.parametrize("question,fields", [
        # A breakdown on another table is no member ranked by its own value.
        ("top 3 product categories by list price", [_PRICE, _CATEGORY]),
        # Nor is a question that ranks nothing.
        ("list price by product", [_PRICE, _PRODUCT]),
    ])
    def test_no_ranking_of_an_attribute(self, question, fields):
        from core.analytical_request_plan import aggregated_attribute

        assert aggregated_attribute(question, {"fields": fields}, _MODEL) == {}

    def test_the_average_is_still_an_average(self):
        from core.analytical_request_plan import aggregated_attribute

        average = aggregated_attribute("average list price of our products", {"fields": [_PRICE, _PRODUCT]}, _MODEL)
        assert (average["aggregation"], average.get("ranked")) == ("AVG", None)

    @pytest.mark.parametrize("question,asked", [
        ("top 3 products by list price", []),
        ("top 5 categories by products sold", [_SOLD]),
        ("how many products did we sell", [_SOLD]),
    ])
    def test_a_count_named_by_its_members_alone(self, question, asked):
        from core.analytical_request_plan import metrics_asked_for

        assert metrics_asked_for(question, [_SOLD]) == asked

    def test_the_ranked_members_are_the_breakdown(self):
        from core.analytical_request_plan import demote_beside_an_attribute

        plan = {"source_scope": {"selected_fact": "dbo.DimProduct", "source_kind": "master"},
                "fields": [dict(_PRICE), dict(_PRODUCT, enforcement="required")]}
        demote_beside_an_attribute(plan, {"intent": "ranking", "entity_grain": "product", "dimensions": []})
        assert plan["fields"][1]["enforcement"] == "required"
