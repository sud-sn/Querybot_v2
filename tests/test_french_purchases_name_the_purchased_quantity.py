"""
French purchases name the purchased quantity.

"Achats par entrepôt" -- purchases by warehouse -- was refused on the sample
tenant: "the semantic layer has not resolved the governed measure". Nothing
read "achats" as purchases, so the question reached the pipeline as "achats by
warehouse"; and the list of candidate metrics was scored on the reader's own
words alone, where the purchased quantity's French names ("quantité achetée",
"unités achetées") share no word with "achats". "Purchases by warehouse" was
answered.

"Achats" is read as purchases now, and the candidate metrics are scored in
English and in the reader's words, the better of the two, as every other
metric matcher scores them.

A synthetic tenant (tests/answer_harness.py) whose monthly fact keeps the
purchased quantity; the warehouse and the model are the only stand-ins.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("french-purchases")) as built:
        yield built


def _purchased(key) -> dict:
    """The purchased quantity of every monthly row but the whole-year one, by key(row)."""
    totals: dict = {}
    for row in harness.MOVES:
        whs, item, _period, _sold, purchased, *_rest = row
        totals[key(whs, item)] = totals.get(key(whs, item), 0) + purchased
    return totals


def _answered(answer: dict, *labels: str) -> dict:
    assert answer["model_wrote_sql"] is False
    return {tuple(row[label] for label in labels): row["PURCHASED_QUANTITY"] for row in answer["rows"]}


class TestTheProductAnswers:

    def test_purchases_by_warehouse(self, warehouse):
        answer = harness.ask(warehouse, "Achats par entrepôt", "fr")
        assert _answered(answer, "WAREHOUSE", "UNT_OF_MSR") == pytest.approx(
            _purchased(lambda whs, item: (harness.WAREHOUSES[whs][1], harness.ITEMS[item][3])))

    def test_total_purchases(self, warehouse):
        answer = harness.ask(warehouse, "Total des achats en 2025", "fr")
        assert _answered(answer, "UNT_OF_MSR") == pytest.approx(
            _purchased(lambda whs, item: (harness.ITEMS[item][3],)))


class TestTheWords:

    @pytest.mark.parametrize("question,canonical", [
        ("Achats par entrepôt", "purchases by warehouse"),
        ("Le dernier achat", "the last purchase"),
    ])
    def test_achats_are_purchases(self, question, canonical):
        from core.question_normalizer import canonical_question

        assert canonical_question(question, "fr") == canonical


_PURCHASED = {"name": "Purchased quantity",
              "synonyms": "purchased quantity, quantity purchased, units purchased, quantité achetée, unités achetées"}
_RESERVED_IN_FRENCH = {"name": "Réservé", "synonyms": "quantité réservée"}


class TestTheCandidates:

    @staticmethod
    def _names(question: str, reader_question: str = "") -> list[str]:
        from store.config_store import list_metric_formula_context

        found = list_metric_formula_context("acct", question, metrics=[_PURCHASED, _RESERVED_IN_FRENCH],
                                            reader_question=reader_question)
        return [metric["name"] for metric in found]

    def test_read_in_english(self):
        assert self._names("purchases by warehouse", "Achats par entrepôt") == ["Purchased quantity"]

    def test_and_in_the_readers_words(self):
        # A metric named only in French is found by the reader's words.
        assert self._names("reserved quantity by warehouse", "Quantité réservée par entrepôt") == ["Réservé"]

    def test_the_pipeline_offers_both(self, warehouse):
        # The store the pipeline holds: a module that imports "store" afresh
        # leaves another in sys.modules, which the pipeline never calls.
        from core import query_pipeline

        offered = []
        original = query_pipeline.store.list_metric_formula_context

        def spy(account_id, question, *args, **kwargs):
            offered.append((question, kwargs.get("reader_question")))
            return original(account_id, question, *args, **kwargs)

        with patch.object(query_pipeline.store, "list_metric_formula_context", spy):
            harness.ask(warehouse, "Achats par entrepôt", "fr")
        assert offered[0] == ("purchases by warehouse", "Achats par entrepôt")
