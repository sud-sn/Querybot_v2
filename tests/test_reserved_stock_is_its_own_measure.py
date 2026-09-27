"""
Reserved and back-ordered stock are measures of their own.

A warehouse can keep three parts of its stock side by side -- allocated,
reserved and back-ordered -- and the sample tenant keeps all three, with a
reserved total no allocated column comes near. "Reserved" had been made a
synonym of the allocated quantity, so "reserved quantity by warehouse" was
answered with the allocated quantity. A reserved column and a back-ordered
column now each have a starter metric, named in English and French, and
"reserved" names the reserved one.

A synthetic tenant (tests/answer_harness.py) whose daily snapshot keeps an
allocated, a reserved and a back-ordered quantity; the warehouse and the model
are the only stand-ins.
"""

from __future__ import annotations

import pytest

from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("reserved")) as built:
        yield built


def _by_warehouse(amounts: list) -> dict:
    """An amount of the newest snapshot per warehouse name and unit, row for row with STOCK."""
    totals: dict = {}
    for (whs, item, *_rest), amount in zip(harness.STOCK, amounts):
        key = (harness.WAREHOUSES[whs][1], harness.ITEMS[item][3])
        totals[key] = totals.get(key, 0) + amount
    return totals


def _answered(answer: dict, measure: str) -> dict:
    (run,) = [run for run in answer["executed"] if f"AS {measure}" in run["sql"]]
    return {(row["WAREHOUSE"], row["UNT_OF_MSR"]): row[measure] for row in run["rows"]}


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("Reserved quantity by warehouse", "en"),
        ("Stock réservé par entrepôt", "fr"),
    ])
    def test_reserved(self, warehouse, question, lang):
        answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert _answered(answer, "RESERVED_QUANTITY") == pytest.approx(_by_warehouse(harness.RESERVED))

    @pytest.mark.parametrize("question,lang", [
        ("Back orders by warehouse", "en"),
        ("Commandes en souffrance par entrepôt", "fr"),
    ])
    def test_back_ordered(self, warehouse, question, lang):
        answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert _answered(answer, "BACK_ORDERED_QUANTITY") == pytest.approx(_by_warehouse(harness.BACK_ORDERED))

    def test_allocated_is_still_allocated(self, warehouse):
        answer = harness.ask(warehouse, "Allocated quantity by warehouse")
        assert answer["model_wrote_sql"] is False
        allocated = [row[5] for row in harness.STOCK]
        assert _answered(answer, "ALLOCATED_QUANTITY") == pytest.approx(_by_warehouse(allocated))


class TestTheProposals:

    @staticmethod
    def _metrics() -> dict:
        import store

        return {metric["name"]: metric for metric in store.list_metrics(harness.ACCOUNT)}

    def test_each_part_is_proposed_on_its_own_column(self, warehouse):
        metrics = self._metrics()
        assert metrics["Reserved quantity"]["sql_template"] == "SUM(RSV_QTY)"
        assert metrics["Back-ordered quantity"]["sql_template"] == "SUM(RSV_BCK_ORD_QTY)"

    def test_reserved_is_not_a_name_of_allocated(self, warehouse):
        assert "réserv" not in self._metrics()["Allocated quantity"]["synonyms"]
        assert "reserved" not in self._metrics()["Allocated quantity"]["synonyms"]

    def test_a_back_order_column_is_not_reserved_stock(self):
        from core.starter_metrics import starter_metrics

        # The fields starter_metrics reads of a model: a snapshot fact's measures.
        model = {"tables": [{
            "type": "fact", "fact_type": "periodic_snapshot", "qualified_name": "WH.STOCK_SNAPSHOT",
            "fields": [{"column": "ON_HND_QTY", "role": "measure"}, {"column": "RSV_BCK_ORD_QTY", "role": "measure"}],
        }]}
        assert {metric.name: metric.sql_template for metric in starter_metrics(model)} == {
            "Stock on hand": "SUM(ON_HND_QTY)", "Back-ordered quantity": "SUM(RSV_BCK_ORD_QTY)"}
