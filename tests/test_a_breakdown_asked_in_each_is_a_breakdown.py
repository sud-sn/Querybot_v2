"""
A breakdown asked "in each" is a breakdown.

"Number of orders in each warehouse" asks for one number per warehouse. The
product reads a breakdown from "by", "per", "for each", "across" -- not from
"in each" or "for every", the commonest ways to ask a count per member. On a
warehouse with no warehouse table the question was read as a count with
nothing to break it down by, and the governed compiler answered it: one total,
48 orders, for a question that asked for a number per warehouse. Its one
defence against dropping a breakdown the planner did not bind read the same
words and saw none.

"In each", "in every", "for every" and "within each" now name a breakdown as
"by" does, wherever the question's breakdowns are read: an unbound one is
never answered with a total, and a head count "in each category" is a count
per category. "Do we sell", "offer" and "carry" ask for a catalogue as "do we
have" does -- "combien de produits vendons-nous" among them -- while "did we
sell in 2025" is still activity, not a population.

tests/star_harness.py keeps two years of orders and no warehouse or store.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("in-each")) as built:
        yield built


def _category(product: int) -> str:
    return star.CATEGORIES[star.SUBCATEGORIES[star.PRODUCTS[product][3]][2]][0]


class TestTheProductAnswers:

    @pytest.mark.parametrize("question", ["Number of orders in each warehouse", "Sales in every store"])
    def test_a_breakdown_nothing_binds_is_no_total(self, warehouse, question):
        answer = star.ask(warehouse, question)
        assert answer["rows"] == []

    def test_a_breakdown_that_binds(self, warehouse):
        answer = star.ask(warehouse, "Number of orders in each product category")
        orders: dict[str, set[str]] = {}
        for line in star.orders():
            orders.setdefault(_category(line[4]), set()).add(line[0])
        assert {row["PRODUCT_CATEGORY"]: row["NUMBER_OF_ORDERS"] for row in answer["rows"]} == {
            category: len(numbers) for category, numbers in orders.items()}


class TestTheRule:

    @pytest.mark.parametrize("question,dimensions", [
        ("number of orders in each warehouse", ["warehouse"]),
        ("sales in every store", ["store"]),
        ("how many stores do we have for every region", ["region"]),
        ("sales within each territory", ["territory"]),
        ("sales in 2025", []),
    ])
    def test_the_breakdowns_a_question_names(self, question, dimensions):
        from core.analytical_intent import _DIMENSION_RE

        assert [match.group(1).strip() for match in _DIMENSION_RE.finditer(question)] == dimensions

    @pytest.mark.parametrize("question,population", [
        ("How many products are there in each category?", "product"),
        ("How many customers do we have in each country?", "customer"),
        ("How many products do we sell?", "product"),
        ("How many brands do we carry?", "brand"),
        ("How many products did we sell in 2025?", ""),
        ("How many suppliers delivered late?", ""),
    ])
    def test_a_population_counted(self, question, population):
        from core.analytical_intent import detect_population_count

        assert detect_population_count(question) == population

    def test_in_french(self):
        from core.analytical_intent import detect_population_count
        from core.question_normalizer import canonical_question

        assert detect_population_count(canonical_question("Combien de produits vendons-nous ?", "fr")) == "product"
