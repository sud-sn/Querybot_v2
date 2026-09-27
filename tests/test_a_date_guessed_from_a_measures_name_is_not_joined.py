"""
A date guessed from a measure's name is not joined.

The join graph's first pass reads the question broadly, before the business's
metrics are matched, and takes an event word for the date of that event:
"order quantity by product category" named the Order Date for it. The date
resolver, which reads a measure's name as the measure's, found no date asked
for -- no period, no grain, no snapshot, no date named -- and the governed
compiler rightly wrote no date join. But the guess survived into the
authoritative graph as a required edge, the validator refused the compiled SQL
for missing it, and the question went to the model: "I could not build a
trusted query for this question". The same question in French, whose words are
not the event's, was answered.

Where the question asks for no date, no guessed date is required. A date the
question does ask for -- "by order month" -- is joined as before, and where
reading the question's dates failed, the guess stands as it did.

tests/star_harness.py keeps two years of order lines with an Order Date, a Ship
Date and a Due Date, and the metric "Order Quantity".
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("guessed-date")) as built:
        yield built


def _category(product: int) -> str:
    return star.CATEGORIES[star.SUBCATEGORIES[star.PRODUCTS[product][3]][2]][0]


def _by_category(measure) -> dict[str, float]:
    totals: dict[str, float] = {}
    for line in star.orders():
        product, quantity = line[4], line[7]
        category = _category(product)
        totals[category] = totals.get(category, 0) + measure(product, quantity)
    return totals


class TestTheProductAnswers:

    def test_a_measure_named_for_an_event(self, warehouse):
        answer = star.ask(warehouse, "Order quantity by product category")
        assert answer["model_wrote_sql"] is False
        assert "DATEKEY" not in answer["sql"].upper()
        assert {row["PRODUCT_CATEGORY"]: row["ORDER_QUANTITY"] for row in answer["rows"]} == _by_category(
            lambda product, quantity: quantity)

    def test_beside_another_measure(self, warehouse):
        answer = star.ask(warehouse, "Sales amount and order quantity by product category")
        assert answer["model_wrote_sql"] is False
        assert {row["PRODUCT_CATEGORY"]: (row["SALES_AMOUNT"], row["ORDER_QUANTITY"])
                for row in answer["rows"]} == {
            category: (_by_category(lambda product, quantity: quantity * star.PRODUCTS[product][4])[category],
                       quantity)
            for category, quantity in _by_category(lambda product, quantity: quantity).items()}

    def test_a_date_the_question_asks_for_is_joined(self, warehouse):
        answer = star.ask(warehouse, "Order quantity by order month")
        assert answer["model_wrote_sql"] is False
        assert "ORDERDATEKEY" in answer["sql"].upper()
        assert len(answer["rows"]) == 24
        assert sum(row["ORDER_QUANTITY"] for row in answer["rows"]) == sum(line[7] for line in star.orders())


def _graph() -> dict:
    return {"entities": [
        {"entity_name": name, "table_name": table, "schema_name": "dbo", "entity_type": kind}
        for name, table, kind in (("FactSales", "FactSales", "fact"), ("DimProduct", "DimProduct", "dimension"),
                                  ("Order Date", "DimDate", "dimension"))]}


def _aligned(resolution: dict | None) -> dict:
    from core.semantic_resolution import build_planner_alignment

    return build_planner_alignment(
        graph=_graph(), graph_ctx={"detected": ["FactSales", "Order Date", "DimProduct"]}, semantic_plan={},
        metric_formula_tables={"dbo.FactSales"}, date_context_resolution=resolution)


class TestTheRule:

    def test_no_date_asked(self):
        from core.contextual_dates import NO_DATE_ASKED

        aligned = _aligned({"status": "none", "reason": NO_DATE_ASKED})
        assert aligned["required_entities"] == ["DimProduct", "FactSales"]
        assert aligned["dropped_date_entities"] == ["Order Date"]

    @pytest.mark.parametrize("resolution", [
        {"status": "none", "reason": "no governed date context"},
        {"status": "error", "reason": "timeout"},
        {"status": "ambiguous", "options": []},
        None,
    ])
    def test_a_date_asked_or_unread_keeps_the_guess(self, resolution):
        aligned = _aligned(resolution)
        assert "Order Date" in aligned["required_entities"]
        assert aligned["dropped_date_entities"] == []

    def test_what_the_resolver_says_of_a_question_with_no_date(self):
        from core.contextual_dates import no_date_asked, resolve_contextual_date_binding

        role = {"fact_table": "dbo.FactSales", "fact_column": "OrderDateKey", "context_name": "Order Date",
                "status": "approved", "is_default": True}
        assert no_date_asked(resolve_contextual_date_binding(
            "order quantity by product category", matched_metrics=[], bindings=[], date_roles=[role]))
        assert not no_date_asked(resolve_contextual_date_binding(
            "order quantity by order month", matched_metrics=[], bindings=[], date_roles=[role]))
        assert not no_date_asked({"status": "none", "reason": "no governed date context"})
        assert not no_date_asked(None)
