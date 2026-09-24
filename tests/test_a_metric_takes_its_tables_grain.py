"""
A registry metric over a snapshot fact reads one snapshot, like its column does.

"Reserved quantity by warehouse", answered through a registered metric
SUM(RSV_QTY) whose base table is a daily balance fact, summed the reserved
quantity of every day on file. The column's name says nothing about time, so
the metric classifier called it additive -- while the semantic model, reading
the same column on the same table, calls it a level (semi-additive) and the
field-level check would have pinned the question to one snapshot. The metric
path is the governed one, so the less careful reading won whenever a metric
matched.

The metric classifier now reads its base table's grain the way the semantic
model does: on a periodic snapshot a plain quantity or amount is a level, and a
movement or a count beside it still adds up across periods.

Drives question_has_snapshot_intent, the gate _handle_query_impl calls with the
matched registry metrics. Synthetic tables in a mart's naming convention; no
customer data.
"""

from __future__ import annotations

import pytest

from core.analysis_contract import measure_class_for_metric
from core.contextual_dates import question_has_snapshot_intent

DAILY_BALANCES = "MART.ITM_BAL_DLY_FCT"
MONTHLY_BALANCES = "[MART].[ITM_BAL_PRD_FCT]"
MOVEMENTS = "MART.STK_MVT_FCT"
INVOICES = "MART.CUS_ORD_IVC_FCT"


def _metric(sql_template: str, base_table: str, **extra) -> dict:
    return {
        "name": "Metric", "sql_template": sql_template, "formula_type": "expression",
        "base_table": base_table, **extra,
    }


class TestAPlainQuantityOnASnapshotIsALevel:

    @pytest.mark.parametrize("formula", ["SUM(RSV_QTY)", "SUM(ORD_QTY)", "SUM(BCK_ORD_AMT)"])
    def test_the_question_reads_one_snapshot(self, formula):
        metric = _metric(formula, DAILY_BALANCES)
        assert question_has_snapshot_intent(
            "reserved quantity by warehouse", matched_metrics=[metric],
        ) is True
        assert measure_class_for_metric(metric) == "semi_additive"

    def test_a_bracketed_base_table_reads_the_same(self):
        metric = _metric("SUM(RSV_QTY)", MONTHLY_BALANCES)
        assert measure_class_for_metric(metric) == "semi_additive"


class TestWhatStillAddsUp:

    def test_a_movement_on_the_same_snapshot(self):
        metric = _metric("SUM(PCH_QTY)", MONTHLY_BALANCES)
        assert measure_class_for_metric(metric) == "additive"
        assert question_has_snapshot_intent(
            "purchased quantity by warehouse", matched_metrics=[metric],
        ) is False

    def test_a_count_of_events_on_the_same_snapshot(self):
        assert measure_class_for_metric(_metric("SUM(NUM_OF_RCT)", MONTHLY_BALANCES)) == "additive"

    def test_the_same_quantity_on_a_transaction_fact(self):
        metric = _metric("SUM(IVC_QTY)", INVOICES)
        assert measure_class_for_metric(metric) == "additive"
        assert question_has_snapshot_intent(
            "invoiced quantity by customer", matched_metrics=[metric],
        ) is False

    def test_the_same_quantity_on_a_table_of_movements(self):
        metric = _metric("SUM(ITM_QTY)", MOVEMENTS)
        assert measure_class_for_metric(metric) == "additive"
        assert question_has_snapshot_intent(
            "units moved by warehouse", matched_metrics=[metric],
        ) is False

    def test_a_metric_with_no_base_table_reads_its_names_alone(self):
        assert measure_class_for_metric(_metric("SUM(RSV_QTY)", "")) == "additive"


class TestWhatTheRuleLeavesAlone:

    def test_an_admin_declaration_still_decides(self):
        metric = _metric("SUM(RSV_QTY)", DAILY_BALANCES, aggregation_semantics="additive")
        assert measure_class_for_metric(metric) == "additive"

    def test_a_unit_cost_is_never_summed_and_a_valuation_is_a_level(self):
        assert measure_class_for_metric(_metric("AVG(ITM_CST)", DAILY_BALANCES)) == "non_additive"
        assert measure_class_for_metric(
            _metric("SUM(RSV_QTY * ITM_CST)", DAILY_BALANCES)
        ) == "semi_additive"
