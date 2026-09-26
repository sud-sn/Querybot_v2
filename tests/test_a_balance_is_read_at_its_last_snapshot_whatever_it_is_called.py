"""
A balance is read at its last snapshot, whatever the question calls it.

"Available quantity by warehouse" asks for a level: stock on hand less what is
allocated, held at each daily snapshot. The pipeline knew -- the metric is
semi-additive, and date resolution was entered on that alone -- but the date
plan then asked the question's wording again, and "available quantity" says
none of inventory, stock, on hand or balance. The plan carried no
latest-snapshot rule. The governed compiler, with no snapshot to read at,
declined; and the model's prompt had no SNAPSHOT RULE and no required
anchor, so nothing stopped a query that added every day's snapshot
together. The date plan now takes the pipeline's decision, made from the
resolved metric and measure as well as the words.

A synthetic tenant (tests/answer_harness.py); the warehouse and the model are
the only stand-ins.
"""

from __future__ import annotations

import pytest

from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("levels")) as built:
        yield built


def _latest(value) -> dict:
    """{(warehouse, unit): value} over the newest snapshot's rows."""
    totals: dict = {}
    for whs, item, _buyer, _created, on_hand, allocated, _cost in harness.STOCK:
        key = (harness.WAREHOUSES[whs][1], harness.ITEMS[item][3])
        totals[key] = totals.get(key, 0) + value(on_hand, allocated)
    return totals


class TestTheProductReadsTheLastSnapshot:

    @pytest.mark.parametrize("question,lang", [
        ("Available quantity by warehouse", "en"),
        ("Quantité disponible par entrepôt", "fr"),
    ])
    def test_available_quantity_by_warehouse(self, warehouse, question, lang):
        answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert {(row["WAREHOUSE"], row["UNT_OF_MSR"]): row["AVAILABLE_QUANTITY"] for row in answer["rows"]} == (
            pytest.approx(_latest(lambda on_hand, allocated: on_hand - allocated)))

    def test_allocated_quantity_is_not_added_across_snapshots(self, warehouse):
        answer = harness.ask(warehouse, "What is our allocated quantity?")
        assert answer["model_wrote_sql"] is False
        by_unit: dict = {}
        for (_whs, unit), value in _latest(lambda _on_hand, allocated: allocated).items():
            by_unit[unit] = by_unit.get(unit, 0) + value
        assert {row["UNT_OF_MSR"]: row["ALLOCATED_QUANTITY"] for row in answer["rows"]} == pytest.approx(by_unit)


class TestTheModelIsHeldToItToo:

    def test_its_prompt_carries_the_snapshot_rule(self, warehouse):
        # A condition on a value is not the compiler's to write: the model
        # writes this one, under the same rule.
        answer = harness.ask(warehouse, "Available quantity by warehouse for warehouses with more than 100")
        assert answer["model_wrote_sql"] is True
        assert any("SNAPSHOT RULE" in prompt and "REQUIRED ANCHOR" in prompt for prompt in answer["prompts"])


_BALANCE_DATE = {"fact_table": "MART.ITM_BAL_DLY_FCT", "fact_column": "ITM_BAL_EFC_DT_DMS_KEY",
                 "dimension_table": "MART.DT_DMS", "dimension_key": "DT_DMS_KEY", "date_value_column": "DMS_DT",
                 "date_key_type": "surrogate_fk", "business_role": "Balance date", "name": "Balance date",
                 "temporal_grain": "day"}


def _kinds(plan: dict) -> list[str]:
    return [str(policy.get("kind")) for policy in plan.get("temporal_policies") or []]


class TestTheDatePlanTakesTheCallersDecision:

    def test_a_level_the_caller_named(self):
        from core.contextual_dates import build_contextual_date_plan

        assert _kinds(build_contextual_date_plan(
            _BALANCE_DATE, "available quantity by warehouse", snapshot=True)) == ["latest_snapshot"]

    def test_several_dates_take_it_too(self):
        from core.contextual_dates import build_contextual_date_plan_many

        assert _kinds(build_contextual_date_plan_many(
            [_BALANCE_DATE], "available quantity by warehouse", snapshot=True)) == ["latest_snapshot"]

    def test_with_no_decision_the_wording_decides(self):
        from core.contextual_dates import build_contextual_date_plan

        assert _kinds(build_contextual_date_plan(_BALANCE_DATE, "available quantity by warehouse")) == []
        assert _kinds(build_contextual_date_plan(_BALANCE_DATE, "stock on hand by warehouse")) == ["latest_snapshot"]

    def test_a_flow_is_not_read_at_a_snapshot(self):
        from core.contextual_dates import build_contextual_date_plan

        assert _kinds(build_contextual_date_plan(_BALANCE_DATE, "stock movements by warehouse", snapshot=False)) == []

    def test_a_window_the_question_states_still_wins(self):
        from core.contextual_dates import build_contextual_date_plan

        assert _kinds(build_contextual_date_plan(
            _BALANCE_DATE, "available quantity in the last 3 months", snapshot=True)) == ["last_n"]
