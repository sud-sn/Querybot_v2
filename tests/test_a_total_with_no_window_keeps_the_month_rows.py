"""
A total with no window reads a period table's month rows only.

The monthly snapshot keeps a row for each whole year beside that year's months
(core/period_rows.py), and the validator refuses any read that can take both.
A question with a period -- "units sold in 2025" -- left the year rows out on
its own, their key decoding to no date. One with none -- "purchases by
warehouse", "how many units did we sell?" -- was compiled without that, the
validator refused the compiled SQL, and every such question went to the model.
On the sample tenant "purchases by warehouse" was one of them.

The governed compiler now keeps the month rows of a period table whatever the
question's dates.

A synthetic tenant (tests/answer_harness.py) whose monthly snapshot keeps a
whole-year row beside its months; the warehouse and the model are the only
stand-ins.
"""

from __future__ import annotations

import pytest

from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("month-rows")) as built:
        yield built


def _sold(by_warehouse: bool) -> dict:
    """Units sold of every month row, per unit (and warehouse); the year row is not a month."""
    totals: dict = {}
    for whs, item, period, sold, _bought, _receipts, _cost in harness.MOVES:
        assert period % 100, "the harness keeps its year row apart, in YEAR_ROW"
        unit = harness.ITEMS[item][3]
        key = (harness.WAREHOUSES[whs][1], unit) if by_warehouse else unit
        totals[key] = totals.get(key, 0) + sold
    return totals


def _answered(answer: dict) -> list[dict]:
    (run,) = [run for run in answer["executed"] if "AS UNITS_SOLD" in run["sql"]]
    return run["rows"]


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("Units sold by warehouse", "en"),
        ("Unités vendues par entrepôt", "fr"),
    ])
    def test_each_warehouse(self, warehouse, question, lang):
        answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert {(row["WAREHOUSE"], row["UNT_OF_MSR"]): row["UNITS_SOLD"] for row in _answered(answer)} == (
            pytest.approx(_sold(by_warehouse=True)))

    def test_all_of_it(self, warehouse):
        answer = harness.ask(warehouse, "How many units did we sell?")
        assert answer["model_wrote_sql"] is False
        assert {row["UNT_OF_MSR"]: row["UNITS_SOLD"] for row in _answered(answer)} == (
            pytest.approx(_sold(by_warehouse=False)))

    def test_a_named_period_is_unchanged(self, warehouse):
        answer = harness.ask(warehouse, "Units sold in 2025")
        assert answer["model_wrote_sql"] is False
        assert {row["UNT_OF_MSR"]: row["UNITS_SOLD"] for row in _answered(answer)} == (
            pytest.approx(_sold(by_warehouse=False)))

    def test_a_table_without_period_rows_is_untouched(self, warehouse):
        answer = harness.ask(warehouse, "Stock on hand by warehouse")
        assert answer["model_wrote_sql"] is False
        assert not any("% 100" in run["sql"] for run in answer["executed"])
