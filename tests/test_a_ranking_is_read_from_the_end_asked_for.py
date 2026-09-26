"""
A ranking is read from the end the question asks for.

"Which warehouse has the lowest stock on hand?" was compiled with the same
ORDER BY ... DESC as "highest", and its answer card, which always led with the
largest row, named the warehouse with the most. "Bottom 2 items" could not be
compiled at all -- the validator refused the descending order it was given --
and went to the model. And "which item has the most stock on hand?" was not a
ranking: "most" is not "highest", so the answer was an unordered breakdown.

Now a superlative after "which <thing> has" is a ranking in English and French
("quel entrepôt a le moins de stock"), the compiler sorts from the end the
question asks for -- an explicit Top/Bottom-N's direction, else its words --
and the card names the lowest when the lowest was asked for. "At least 100"
and "the most recent month" are not rankings.

A synthetic tenant (tests/answer_harness.py); the warehouse and the model are
the only stand-ins.
"""

from __future__ import annotations

import pytest

from core import analytical_intent, i18n
from core.analytical_intent import plan_analytical_intent
from core.question_normalizer import canonical_question
from tests import answer_harness as harness


# Read at call time, so each test fails on a tree without them rather than the
# module failing to import.
def asks_for_ranking(question: str) -> bool:
    return analytical_intent.asks_for_ranking(question)


def ranking_direction(question: str) -> str:
    return analytical_intent.ranking_direction(question)


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("rankings")) as built:
        yield built


class TestWhichEndIsAskedFor:

    @pytest.mark.parametrize("question,direction", [
        ("Which item has the most stock on hand?", "descending"),
        ("Which warehouse has the least stock on hand?", "ascending"),
        ("Which warehouse has the lowest stock on hand?", "ascending"),
        ("what item has the fewest units", "ascending"),
        ("Bottom 2 items by stock on hand", "ascending"),
        ("which item group has the highest inventory value", "descending"),
        ("Top 5 warehouses with at least 100 units", "descending"),
    ])
    def test_in_english(self, question, direction):
        assert asks_for_ranking(question) and ranking_direction(question) == direction
        assert plan_analytical_intent(question).intent == "ranking"

    @pytest.mark.parametrize("question,direction", [
        ("Quel article a le plus de stock en main ?", "descending"),
        ("Quel entrepôt a le moins de stock en main ?", "ascending"),
        ("Quel entrepôt a le stock le plus bas ?", "ascending"),
    ])
    def test_in_french_as_typed_and_as_canonicalised(self, question, direction):
        for text in (question, canonical_question(question, "fr")):
            assert asks_for_ranking(text) and ranking_direction(text) == direction

    @pytest.mark.parametrize("question", [
        "which warehouses have at least 100 units",
        "units sold in the most recent month",
        "at most 5 items per order",
        "Quel est le stock le plus récent ?",
    ])
    def test_what_is_not_a_ranking(self, question):
        assert not asks_for_ranking(question)
        assert not asks_for_ranking(canonical_question(question, "fr"))


def _answered(answer: dict) -> dict:
    (run,) = [run for run in answer["executed"] if "AS STOCK_ON_HAND" in run["sql"]]
    return run


def _stock_by(position: int) -> dict:
    """Stock on hand at the newest snapshot per item or warehouse name."""
    totals: dict = {}
    for whs, item, _buyer, _created, on_hand, _allocated, _cost in harness.STOCK:
        name = harness.ITEMS[item][1] if position == 1 else harness.WAREHOUSES[whs][1]
        totals[name] = totals.get(name, 0) + on_hand
    return totals


class TestTheProductSortsFromThatEnd:

    @pytest.mark.parametrize("question,lang", [
        ("Which warehouse has the lowest stock on hand?", "en"),
        ("Quel entrepôt a le moins de stock en main ?", "fr"),
    ])
    def test_the_lowest_first(self, warehouse, question, lang):
        answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        run = _answered(answer)
        assert "STOCK_ON_HAND ASC" in run["sql"]
        values = [row["STOCK_ON_HAND"] for row in run["rows"]]
        assert values == sorted(values)

    def test_the_most_first(self, warehouse):
        answer = harness.ask(warehouse, "Which item has the most stock on hand?")
        assert answer["model_wrote_sql"] is False
        run = _answered(answer)
        assert "STOCK_ON_HAND DESC" in run["sql"]
        assert run["rows"][0]["ITEM"] == max(_stock_by(1).items(), key=lambda pair: pair[1])[0]

    def test_a_bottom_n_is_compiled(self, warehouse):
        answer = harness.ask(warehouse, "Bottom 2 items by stock on hand")
        assert answer["model_wrote_sql"] is False
        assert [row["ITEM"] for row in _answered(answer)["rows"]] == [
            name for name, _ in sorted(_stock_by(1).items(), key=lambda pair: pair[1])[:2]]

    def test_its_card_names_the_lowest(self, warehouse):
        answer = harness.ask(warehouse, "Bottom 1 item by stock on hand")
        (card,) = [payload["answer"] for kind, payload in answer["replies"]
                   if isinstance(payload, dict) and payload.get("answer")]
        lowest = min(_stock_by(1).items(), key=lambda pair: pair[1])[0]
        assert card["headline"].startswith("Lowest-ranked result: " + lowest)
        assert "lowest row" in card["comparison"]


class TestTheCardNamesTheEndAskedFor:

    ROWS = [{"WAREHOUSE": "NORTH DEPOT", "STOCK": 900}, {"WAREHOUSE": "SOUTH DEPOT", "STOCK": 350},
            {"WAREHOUSE": "EAST DEPOT", "STOCK": 120}]

    @staticmethod
    def _answer(question: str, lang: str = "en") -> dict:
        from core.response_builder import build_answer, infer_result_scope

        token = i18n.activate_language(lang)
        try:
            rows = TestTheCardNamesTheEndAskedFor.ROWS
            return build_answer(rows, question, infer_result_scope(rows, question, mode="ranking"))
        finally:
            i18n.deactivate_language(token)

    def test_the_lowest(self):
        card = self._answer("Which warehouse has the lowest stock?")
        assert card["headline"] == "EAST DEPOT is lowest at 120."
        assert card["comparison"] == "230 below the next result"

    def test_the_lowest_in_french(self):
        card = self._answer("Quel entrepôt a le moins de stock ?", "fr")
        assert card["headline"].startswith("EAST DEPOT est le plus bas")

    def test_the_highest_is_unchanged(self):
        card = self._answer("Which warehouse has the highest stock?")
        assert card["headline"] == "NORTH DEPOT leads at 900."
        assert card["comparison"] == "550 above the next result"
