"""
A dimension reached from two facts is one dimension.

"Which item has the most stock on hand?" matched the item key on the daily
snapshot and on the monthly fact equally well. The lexical planner called that
a tie and kept the item as a hint, and the governed compiler, which writes only
what is bound for certain, left the question to free-form generation. The key
on the question's own fact now outranks the same key on another fact; with no
fact to prefer it stays a hint, since which fact it came through would be a
guess.

Bound on its own, the item then had no join: the lexical planner builds joins
only between the fields it bound, and here the metric, not a field, supplies
the measure. The compiler now reaches a requested dimension through the edges
the entity graph resolved for the same question -- the ones the validator
holds the SQL to -- when the plan has none. "Inventory value by item group"
is answered through the item, two joins away.

And a word bound twice, once for certain and once as a losing hint ("product
group" on the product-group table and on the item-group table), is one
breakdown: the hint is not a second field the compiler failed to write.

A synthetic tenant (tests/answer_harness.py); the warehouse and the model are
the only stand-ins.
"""

from __future__ import annotations

import pytest

from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("dimensions")) as built:
        yield built


def _totals(key) -> dict[str, float]:
    totals: dict[str, float] = {}
    for _whs, item, _buyer, _created, on_hand, _allocated, cost in harness.STOCK:
        name, value = key(item, on_hand, cost)
        totals[name] = totals.get(name, 0) + value
    return totals


def _named(rows: list[dict]) -> dict[str, float]:
    """{first text: the number} from rows of names, units and one number."""
    return {
        next(v for v in row.values() if isinstance(v, str)):
        next(float(v) for v in row.values() if isinstance(v, (int, float)) and not isinstance(v, bool))
        for row in rows
    }


class TestTheProductWritesTheSql:

    @pytest.mark.parametrize("question", [
        "Which item has the most stock on hand?",
        "What is our total stock on hand by item?",
    ])
    def test_stock_by_item(self, warehouse, question):
        answer = harness.ask(warehouse, question)
        assert answer["model_wrote_sql"] is False
        assert _named(answer["rows"]) == pytest.approx(
            _totals(lambda item, on_hand, _cost: (harness.ITEMS[item][1], on_hand)))

    def test_each_item_keeps_its_unit(self, warehouse):
        rows = harness.ask(warehouse, "What is our total stock on hand by item?")["rows"]
        assert {row["ITEM"]: row["UNT_OF_MSR"] for row in rows} == {
            harness.ITEMS[item][1]: harness.ITEMS[item][3] for item in {line[1] for line in harness.STOCK}}

    def test_the_item_is_reached_from_the_question_s_own_fact(self, warehouse):
        sql = harness.ask(warehouse, "What is our total stock on hand by item?")["sql"].upper()
        assert "LEFT JOIN [MART].[ITM_DMS]" in sql and "ITM_BAL_PRD_FCT" not in sql

    def test_a_dimension_two_joins_away(self, warehouse):
        answer = harness.ask(warehouse, "What is the inventory value by item group?")
        assert answer["model_wrote_sql"] is False
        assert _named(answer["rows"]) == pytest.approx(_totals(
            lambda item, on_hand, cost: (harness.GROUPS[harness.ITEMS[item][2]][1], on_hand * cost)))


# ── The pieces ───────────────────────────────────────────────────────────────

_TWO_FACTS = {
    "MART.ITM_BAL_DLY_FCT": {"ITM_BAL_DLY_FCT_KEY": "bigint", "ITM_DMS_KEY": "int", "ON_HND_QTY": "decimal"},
    "MART.ITM_BAL_PRD_FCT": {"ITM_BAL_PRD_FCT_KEY": "bigint", "ITM_DMS_KEY": "int", "SLD_QTY": "decimal"},
    "MART.ITM_DMS": {"ITM_DMS_KEY": "int", "ITM_NM": "nvarchar"},
}


class TestTheQuestionsOwnFactHoldsTheKey:

    @staticmethod
    def _item(**kwargs) -> dict:
        from core.semantic_planner import build_semantic_field_plan

        plan = build_semantic_field_plan("Which item has the most?", _TWO_FACTS, None, **kwargs)
        return next(field for field in plan["fields"] if field["term"] == "item")

    def test_the_key_on_its_own_fact_wins(self):
        item = self._item(preferred_fact_tables={"MART.ITM_BAL_DLY_FCT"})
        assert item.get("enforcement") != "optional" and not item.get("ambiguous_source")
        assert (item["table"], item["column"], item["source_key_table"]) == (
            "MART.ITM_DMS", "ITM_NM", "MART.ITM_BAL_DLY_FCT")

    def test_with_no_fact_to_prefer_it_stays_a_hint(self):
        item = self._item()
        assert item.get("enforcement") == "optional" and item.get("ambiguous_source")


_COLUMNS = {
    "MART.ITM_BAL_DLY_FCT": {"ITM_BAL_DLY_FCT_KEY": "bigint", "ITM_DMS_KEY": "int", "ITM_BAL_EFC_DT_DMS_KEY": "int",
                             "ON_HND_QTY": "decimal"},
    "MART.ITM_DMS": {"ITM_DMS_KEY": "int", "ITM_NM": "nvarchar"},
    "MART.WHS_DMS": {"WHS_DMS_KEY": "int", "WHS_DSC": "nvarchar"},
    "MART.ITM_GRP_DMS": {"ITM_GRP_DMS_KEY": "int", "ITM_GRP_DSC": "nvarchar"},
    "MART.DT_DMS": {"DT_DMS_KEY": "int", "DMS_DT": "date"},
}
_ITEM_EDGE = {"from_schema": "MART", "from_table": "ITM_BAL_DLY_FCT", "to_schema": "MART", "to_table": "ITM_DMS",
              "conditions": [["ITM_DMS_KEY", "ITM_DMS_KEY"]], "join_type": "LEFT"}


def _stock_by_item(*extra_fields: dict, edges: tuple = (_ITEM_EDGE,)) -> dict:
    """The context the pipeline hands the compiler for "stock on hand by
    item": the item bound on its own, so the plan's joins reach only the date."""
    return {
        "question": "stock on hand by item",
        "canonical_question": "stock on hand by item",
        "metric_formulas": [{"name": "Stock on hand", "formula_type": "expression", "sql_template": "SUM(ON_HND_QTY)",
                             "base_table": "MART.ITM_BAL_DLY_FCT"}],
        "analytical_request_plan": {"status": "compiled", "intent": "metric_query",
                                    "source_fact": "MART.ITM_BAL_DLY_FCT", "source_facts": ["MART.ITM_BAL_DLY_FCT"]},
        "graph_context": {"resolved_edges": list(edges)},
        "semantic_plan": {
            "fields": [{"term": "item", "table": "MART.ITM_DMS", "column": "ITM_NM", "role": "display_dimension",
                        "enforcement": "required", "display_required": True}, *extra_fields],
            "joins": [{"from": "MART.ITM_BAL_DLY_FCT", "to": "MART.DT_DMS",
                       "conditions": [["ITM_BAL_EFC_DT_DMS_KEY", "DT_DMS_KEY"]], "enforcement": "required"}],
            "temporal_policies": [{"kind": "latest_snapshot", "amount": 1, "unit": "period",
                                   "fact_table": "MART.ITM_BAL_DLY_FCT", "fact_column": "ITM_BAL_EFC_DT_DMS_KEY",
                                   "dimension_table": "MART.DT_DMS", "dimension_key": "DT_DMS_KEY",
                                   "date_column": "DMS_DT", "date_key_type": "surrogate_fk",
                                   "role_alias": "balance_date"}],
        },
    }


def _compile(context: dict) -> str:
    from core.pipeline_helpers import compile_governed_temporal_metric_sql

    return compile_governed_temporal_metric_sql("azure_sql", set(_COLUMNS), set(_COLUMNS), _COLUMNS, context)


class TestTheGraphReachesWhatThePlanDidNot:

    def test_the_resolved_edge_is_the_join(self):
        sql = _compile(_stock_by_item())
        assert ("LEFT JOIN [MART].[ITM_DMS] AS business_dimension "
                "ON fact_rows.[ITM_DMS_KEY] = business_dimension.[ITM_DMS_KEY]") in sql
        assert "business_dimension.[ITM_NM] AS ITEM" in sql

    def test_with_no_edge_there_is_no_answer(self):
        assert _compile(_stock_by_item(edges=())) == ""


class TestAWordBoundTwiceIsOneBreakdown:

    def test_the_hint_for_a_written_word_is_not_another_field(self):
        rival = {"term": "Item", "table": "MART.ITM_GRP_DMS", "column": "ITM_GRP_DSC", "role": "display_dimension",
                 "enforcement": "optional", "display_required": True}
        assert "business_dimension.[ITM_NM] AS ITEM" in _compile(_stock_by_item(rival))

    def test_a_hint_for_another_word_still_is(self):
        other = {"term": "warehouse", "table": "MART.WHS_DMS", "column": "WHS_DSC", "role": "display_dimension",
                 "enforcement": "optional", "display_required": True}
        assert _compile(_stock_by_item(other)) == ""
