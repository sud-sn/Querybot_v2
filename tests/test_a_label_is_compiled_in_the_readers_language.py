"""
A label is compiled in the reader's language.

A warehouse built for two languages keeps a label twice -- ITM_NM and
ITM_FR_NM -- and a French reader is shown the French one where it is filled
and the English one where it is blank (core/label_language.py). The validator
refuses a label in the other language, but the governed compiler wrote the
column the plan named, the English one, so its answer to a French reader was
thrown away: "quel article a le plus de stock en main ?" and "les 5
principaux groupes d'articles par valeur du stock" were compiled, refused,
and left to the model.

A synthetic tenant (tests/answer_harness.py) whose items keep a French name,
blank for two of them; the warehouse and the model are the only stand-ins. It
selects no vocabulary pack, so it reads "article" as no column: its French is
asked for by name, which chooses the label language the same way a French
reader does.
"""

from __future__ import annotations

import pytest

from core import label_language
from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("labels")) as built:
        yield built


def _shown(item: int, lang: str) -> str:
    """The item's label as a reader of ``lang`` is shown it."""
    return (harness.ITEM_FR_NAMES.get(item) or harness.ITEMS[item][1]) if lang == "fr" else harness.ITEMS[item][1]


def _on_hand_by_item() -> dict[int, float]:
    totals: dict = {}
    for _whs, item, _buyer, _created, on_hand, _allocated, _cost in harness.STOCK:
        totals[item] = totals.get(item, 0) + on_hand
    return totals


def _answered(answer: dict) -> list[dict]:
    (run,) = [run for run in answer["executed"] if "AS STOCK_ON_HAND" in run["sql"]]
    return run["rows"]


class TestTheProductAnswers:
    """Asked for in French -- the language a French reader's labels are shown
    in, and the one a question can name."""

    def test_the_leader_in_french(self, warehouse):
        answer = harness.ask(warehouse, "Which item has the most stock on hand, in French?")
        assert answer["model_wrote_sql"] is False
        leader = max(_on_hand_by_item().items(), key=lambda pair: pair[1])[0]
        assert _answered(answer)[0]["ITEM"] == _shown(leader, "fr")

    def test_a_blank_french_label_falls_back(self, warehouse):
        answer = harness.ask(warehouse, "Stock on hand by item in French")
        assert answer["model_wrote_sql"] is False
        shown = {(row["ITEM"], row["UNT_OF_MSR"]): row["STOCK_ON_HAND"] for row in _answered(answer)}
        expected: dict = {}
        for item, total in _on_hand_by_item().items():
            key = (_shown(item, "fr"), harness.ITEMS[item][3])
            expected[key] = expected.get(key, 0) + total
        assert shown == pytest.approx(expected)

    def test_an_english_reader_is_shown_english(self, warehouse):
        answer = harness.ask(warehouse, "Which item has the most stock on hand?")
        assert answer["model_wrote_sql"] is False
        leader = max(_on_hand_by_item().items(), key=lambda pair: pair[1])[0]
        assert _answered(answer)[0]["ITEM"] == _shown(leader, "en")


class TestTheExpression:
    """What the compiler writes for one label column, read from the plan."""

    PLAN = {"label_policies": [{"kind": "label_language", "table": "WH.MART.ITM_DMS",
                                "twins": [{"base": "ITM_NM", "twin": "ITM_FR_NM", "language": "fr"}]}]}

    @staticmethod
    def _expression(*args):
        return label_language.reader_label_expression(*args)

    def test_a_french_reader(self):
        assert self._expression("d", "MART.ITM_DMS", "ITM_NM", {**self.PLAN, "label_language": "fr"}) == (
            "COALESCE(NULLIF(TRIM(d.[ITM_FR_NM]), ''), d.[ITM_NM])")

    def test_an_english_reader_of_a_french_column(self):
        assert self._expression("d", "MART.ITM_DMS", "ITM_FR_NM", {**self.PLAN, "label_language": "en"}) == (
            "d.[ITM_NM]")

    def test_a_table_without_the_twin(self):
        # A twin belongs to its table: an archive of items keeps ITM_NM and no
        # French copy of it.
        assert self._expression("d", "MART.ITM_ARCHIVE", "ITM_NM", {**self.PLAN, "label_language": "fr"}) == ""
