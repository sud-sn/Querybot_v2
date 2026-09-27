"""
Each measure a question names is one it asks for.

"Deliveries and physical inventory counts by month in 2022" was answered on
the sample tenant with the physical counts alone, in English and French. Both
are registered measures, and both were named in their own words; but a metric
is kept only within a few points of the best-scoring one -- a window meant to
keep a close second READING of one measure -- and a longer name scores more,
so "physical inventory counts" left "deliveries" out of the answer.

A metric whose own name, or one of its synonyms, the question says whole --
where no measure already kept is named -- is now kept as well, however far its
score is from the first. A name said inside another's is that one's.

A synthetic tenant (tests/answer_harness.py) whose monthly fact keeps the units
sold and the receipts, each a registered metric.
"""

from __future__ import annotations

import pytest

from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("two-measures")) as built:
        yield built


def _by_month_and_unit_2025() -> dict:
    totals: dict = {}
    for _whs, item, period, sold, _bought, receipts, _cost in harness.MOVES:
        if 202501 <= period <= 202512:
            key = (f"2025-{period % 100:02d}", harness.ITEMS[item][3])
            units, counted = totals.get(key, (0, 0))
            totals[key] = (units + sold, counted + receipts)
    return totals


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("Units sold and receipts by month in 2025", "en"),
        ("Unités vendues et réceptions par mois en 2025", "fr"),
    ])
    def test_both_measures(self, warehouse, question, lang):
        answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert {
            (str(row["PERIOD"])[:7], row["UNT_OF_MSR"]): (row["UNITS_SOLD"], row["NUMBER_OF_RECEIPTS"])
            for row in answer["rows"]
        } == _by_month_and_unit_2025()


def _metric(name: str, synonyms: str = "") -> dict:
    return {"name": name, "synonyms": synonyms, "base_table": "MART.ITM_BAL_PRD_FCT",
            "sql_template": "SUM(X)", "formula_type": "expression"}


DELIVERIES = _metric("Number of deliveries", "number of deliveries, delivery count, deliveries")
COUNTS = _metric("Number of physical inventory counts", "physical inventory counts, physical counts")
REVENUE = _metric("Revenue", "revenue, sales")
NET_REVENUE = _metric("Net revenue", "net revenue, net sales")
MONTH_END_VALUE = _metric("Month-end inventory value", "month-end inventory value")
VALUE = _metric("Inventory value", "inventory value, stock value")


def _scope(question: str, *metrics: dict) -> list[str]:
    from core.metric_scope import resolve_metric_scope

    return sorted(metric["name"] for metric in resolve_metric_scope(list(metrics), question, None).metrics)


class TestTheScope:

    def test_two_measures_named(self):
        assert _scope("deliveries and physical inventory counts by month in 2022", DELIVERIES, COUNTS) == [
            "Number of deliveries", "Number of physical inventory counts"]

    def test_a_name_said_inside_another(self):
        assert _scope("net revenue by region", REVENUE, NET_REVENUE) == ["Net revenue"]

    def test_a_measure_not_named_whole(self):
        assert _scope("inventory value by warehouse", VALUE, MONTH_END_VALUE) == ["Inventory value"]

    def test_one_measure_named(self):
        assert _scope("physical inventory counts by month", DELIVERIES, COUNTS) == [
            "Number of physical inventory counts"]
