"""
The join graph reads a question in English as well as in the reader's words.

"Valeur du stock par classe ABC (volume)" was answered with the manual ABC
class on the sample tenant, where "Inventory value by ABC class (volume)" was
answered with the volume class. The snapshot reaches the class table by four
keys, one per way of classing an item, and the join graph takes the key whose
relationship the question names. It read the reader's own words: "classe ABC
(volume)" names no relationship called "abc class volume", so the four were
tied, and the first was taken. Every other detector reads the question's
canonical English; the graph did not, and told the French reader the join was
a toss-up -- in English: "name *Seller* to use the other one" -- where the
English reader who asked the same was told nothing.

"Manuelle" was not read as manual either, nor "fréquence" as frequency.

A synthetic tenant (tests/answer_harness.py) whose snapshot names each row's
buyer and seller, two keys into one party table, each its own relationship.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests import answer_harness as harness

_ON_HAND, _COST = 4, 6
_RELATIONSHIP_NOTICE = ("relationship", "relation")


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("two-parties")) as built:
        yield built


def _value_by(party_of_row) -> dict:
    """The newest snapshot's inventory value by the name of each row's party."""
    totals: dict = {}
    for n, row in enumerate(harness.STOCK):
        name = harness.PARTIES[party_of_row(n, row)][1]
        totals[name] = totals.get(name, 0) + row[_ON_HAND] * row[_COST]
    return totals


def _by_buyer() -> dict:
    return _value_by(lambda n, row: row[2])


def _by_seller() -> dict:
    return _value_by(lambda n, row: harness.SELLER[n])


def _answered(answer: dict, role: str) -> dict:
    assert answer["model_wrote_sql"] is False
    return {row[role]: row["INVENTORY_VALUE"] for row in answer["rows"]}


def _notices(answer: dict) -> list[str]:
    return [str(body) for kind, body in answer["replies"]
            if kind == "message" and any(word in str(body) for word in _RELATIONSHIP_NOTICE)]


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("Inventory value by buyer", "en"),
        ("Valeur du stock par acheteur", "fr"),
    ])
    def test_by_buyer_without_a_toss_up(self, warehouse, question, lang):
        answer = harness.ask(warehouse, question, lang)
        assert _answered(answer, "BUYER") == pytest.approx(_by_buyer())
        assert _notices(answer) == []

    def test_by_seller(self, warehouse):
        answer = harness.ask(warehouse, "Inventory value by seller")
        assert _answered(answer, "SELLER") == pytest.approx(_by_seller())
        assert _notices(answer) == []


class TestTheGraphReadsBoth:

    def test_the_resolver_is_given_the_english(self, warehouse):
        # The graph the pipeline holds is resolved on the reader's words and
        # their English, never on the reader's words alone.
        from core import query_pipeline

        asked = []
        original = query_pipeline._graph_resolve

        def spy(*args, **kwargs):
            asked.append(kwargs.get("question"))
            return original(*args, **kwargs)

        with patch.object(query_pipeline, "_graph_resolve", spy):
            harness.ask(warehouse, "Valeur du stock par acheteur", "fr")
        assert asked and all("by buyer" in text for text in asked)
        assert any("Valeur du stock par acheteur" in text for text in asked)

    def test_an_english_question_is_read_once(self, warehouse):
        from core import query_pipeline

        asked = []
        original = query_pipeline._graph_resolve

        def spy(*args, **kwargs):
            asked.append(kwargs.get("question"))
            return original(*args, **kwargs)

        with patch.object(query_pipeline, "_graph_resolve", spy):
            harness.ask(warehouse, "Inventory value by buyer")
        # Once, never beside a copy of itself; the last pass reads it less the
        # measure's own name (core/semantic_model.py, without_measure_names).
        assert asked and all("\n" not in text and text.strip().endswith("by buyer") for text in asked)
        assert any(text.strip() == "Inventory value by buyer" for text in asked)


class TestTheWords:

    @pytest.mark.parametrize("question,canonical", [
        ("Valeur du stock par classe ABC (manuelle)", "value of inventory by classe abc (manual)"),
        ("Valeur du stock par classe ABC (fréquence)", "value of inventory by classe abc (frequency)"),
    ])
    def test_in_french(self, question, canonical):
        from core.question_normalizer import canonical_question

        assert canonical_question(question, "fr") == canonical
