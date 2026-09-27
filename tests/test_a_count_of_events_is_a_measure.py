"""
A count of events a row keeps is a measure of its own.

"How many receipts did we have in 2022?" was refused on the sample tenant,
and so were "receipts by division", "the top 5 item groups by number of
receipts" and their French twins. The monthly snapshot keeps the receipts of
each month on each row (NUM_OF_RCT), beside the returns, the deliveries and
the physical inventory counts; no metric summed them, and read as a count of
receipts the question waited on a receipt identifier no snapshot keeps.

Each such count now has a starter metric of its own, named for its event in
English and French and proposed to the administrator like every starter
metric. "Réception(s)" is read as "receipt(s)", and "inventaires physiques" as
physical inventory counts.

A synthetic tenant (tests/answer_harness.py) whose monthly snapshot keeps its
receipts; the warehouse and the model are the only stand-ins.
"""

from __future__ import annotations

import pytest

from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("event-counts")) as built:
        yield built


def _receipts_in(year: int) -> dict:
    """Receipts of the year's months per warehouse; the whole-year row is not a month."""
    totals: dict = {}
    for whs, _item, period, _sold, _bought, receipts, _cost in harness.MOVES:
        if period // 100 == year and period % 100:
            name = harness.WAREHOUSES[whs][1]
            totals[name] = totals.get(name, 0) + receipts
    return totals


def _answered(answer: dict) -> list[dict]:
    (run,) = [run for run in answer["executed"] if "AS NUMBER_OF_RECEIPTS" in run["sql"]]
    return run["rows"]


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("How many receipts did we have in 2025?", "en"),
        ("Combien de réceptions avons-nous eues en 2025 ?", "fr"),
    ])
    def test_the_receipts_of_a_year(self, warehouse, question, lang):
        answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        (row,) = _answered(answer)
        assert row["NUMBER_OF_RECEIPTS"] == sum(_receipts_in(2025).values())

    @pytest.mark.parametrize("question,lang", [
        ("Receipts by warehouse in 2025", "en"),
        ("Nombre de réceptions par entrepôt en 2025", "fr"),
    ])
    def test_the_receipts_of_each_warehouse(self, warehouse, question, lang):
        answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert {row["WAREHOUSE"]: row["NUMBER_OF_RECEIPTS"] for row in _answered(answer)} == _receipts_in(2025)


class TestTheProposals:

    def test_the_tenants_receipts_are_proposed(self, warehouse):
        import store

        metrics = {metric["name"]: metric for metric in store.list_metrics(harness.ACCOUNT)}
        assert metrics["Number of receipts"]["sql_template"] == "SUM(NUM_OF_RCT)"

    @staticmethod
    def _proposed(*columns: str) -> dict:
        from core.starter_metrics import starter_metrics

        model = {"tables": [{
            "type": "fact", "fact_type": "periodic_snapshot", "qualified_name": "WH.STOCK_MONTHLY",
            "fields": [{"column": column, "role": "measure"} for column in columns],
        }]}
        return {metric.name: metric for metric in starter_metrics(model)}

    def test_each_count_is_named_for_its_event(self):
        proposed = self._proposed("NUM_OF_RCT", "NUM_OF_RET", "NBR_OF_DLV", "NUM_OF_PHY_INV", "NUM_OF_XFR_ORD")
        assert {name: metric.sql_template for name, metric in proposed.items()} == {
            "Number of receipts": "SUM(NUM_OF_RCT)",
            "Number of returns": "SUM(NUM_OF_RET)",
            "Number of deliveries": "SUM(NBR_OF_DLV)",
            "Number of physical inventory counts": "SUM(NUM_OF_PHY_INV)",
        }

    def test_what_is_not_a_count_of_the_period(self):
        # Last year's count, and a quantity received rather than a count.
        assert self._proposed("NUM_OF_RCT_PY", "RCT_QTY") == {}

    def test_each_count_is_named_in_french(self):
        proposed = self._proposed("NUM_OF_RCT", "NUM_OF_RET", "NBR_OF_DLV", "NUM_OF_PHY_INV")
        synonyms = {synonym for metric in proposed.values() for synonym in metric.synonyms}
        assert {"nombre de réceptions", "nombre de retours", "nombre de livraisons",
                "nombre d'inventaires physiques"} <= synonyms


class TestTheFrench:

    @pytest.mark.parametrize("french,english", [
        ("Combien de réceptions avons-nous eues en 2022 ?", "receipts"),
        ("Livraisons et inventaires physiques par mois en 2022", "physical inventory counts"),
    ])
    def test_the_events_are_read(self, french, english):
        from core.question_normalizer import canonical_question

        assert english in canonical_question(french, "fr")
