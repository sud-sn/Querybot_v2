"""
A count and a superlative are a ranking.

"Top 10 items by inventory value" was read as a ranking of ten; "the 10 items
with the largest inventory value" was not, nor its French twin "les 10
articles ayant la plus grande valeur de stock" -- compiled as the value of
every item, unordered, and the ten largest were not among the rows shown.
"The 3 largest warehouses", "les 3 entrepôts qui ont le moins de stock" and
"les 3 plus grands entrepôts" are rankings too, from the end their superlative
names.

And a ranking of warehouses by a quantity, kept per unit of measure, read
each row's unit through its item and grouped by it: the placeholder item's
unit was a group of its own, the validator refused the ranking for it, and
"bottom 2 warehouses by stock on hand" was left to the model. The placeholder
item is left out by the fact's own key.

A synthetic tenant (tests/answer_harness.py); the warehouse and the model are
the only stand-ins.
"""

from __future__ import annotations

import pytest

from core.query_semantics import detect_top_n_intent
from core.question_normalizer import canonical_question
from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("counted")) as built:
        yield built


def _read(question: str, lang: str) -> tuple[int, str] | None:
    found = detect_top_n_intent(canonical_question(question, lang) if lang == "fr" else question)
    return (found.limit, found.direction) if found else None


class TestTheCountAndTheEnd:

    @pytest.mark.parametrize("question,lang,expected", [
        ("The 10 items with the largest inventory value", "en", (10, "descending")),
        ("the 3 largest warehouses by inventory value", "en", (3, "descending")),
        ("the 2 items with the fewest units", "en", (2, "ascending")),
        ("Les 10 articles ayant la plus grande valeur de stock", "fr", (10, "descending")),
        ("Les 10 articles avec le plus de stock", "fr", (10, "descending")),
        ("Les 3 entrepôts qui ont le moins de stock", "fr", (3, "ascending")),
        ("Les 3 plus grands entrepôts", "fr", (3, "descending")),
    ])
    def test_a_counted_superlative(self, question, lang, expected):
        assert _read(question, lang) == expected

    @pytest.mark.parametrize("question,lang,expected", [
        ("Top 10 items by inventory value", "en", (10, "descending")),
        ("units sold in the last 3 months with the most sales", "en", None),
        ("the 5 items with stock", "en", None),
    ])
    def test_what_was_read_already_and_what_is_not_one(self, question, lang, expected):
        assert _read(question, lang) == expected


def _rows(answer: dict, measure: str) -> list[dict]:
    (run,) = [run for run in answer["executed"] if f"AS {measure}" in run["sql"]]
    return run["rows"]


class TestTheProductRanks:

    def test_the_items_with_the_largest_value(self, warehouse):
        answer = harness.ask(warehouse, "The 2 items with the largest inventory value")
        assert answer["model_wrote_sql"] is False
        value: dict = {}
        for _whs, item, _buyer, _created, on_hand, _allocated, cost in harness.STOCK:
            value[harness.ITEMS[item][1]] = value.get(harness.ITEMS[item][1], 0) + on_hand * cost
        largest = sorted(value.items(), key=lambda pair: pair[1], reverse=True)[:2]
        assert [(row["ITEM"], row["INVENTORY_VALUE"]) for row in _rows(answer, "INVENTORY_VALUE")] == (
            pytest.approx(largest))

    @pytest.mark.parametrize("question,lang", [
        ("Bottom 2 warehouses by stock on hand", "en"),
        ("Les 2 entrepôts qui ont le moins de stock en main", "fr"),
    ])
    def test_the_warehouses_with_the_least_stock(self, warehouse, question, lang):
        answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        on_hand: dict = {}
        for whs, item, _buyer, _created, quantity, _allocated, _cost in harness.STOCK:
            key = (harness.WAREHOUSES[whs][1], harness.ITEMS[item][3])
            on_hand[key] = on_hand.get(key, 0) + quantity
        least = sorted(on_hand.items(), key=lambda pair: pair[1])[:2]
        assert [((row["WAREHOUSE"], row["UNT_OF_MSR"]), row["STOCK_ON_HAND"])
                for row in _rows(answer, "STOCK_ON_HAND")] == least
