"""
A question about an approved metric is answered by the product's own SQL,
from the metric's table.

On a distribution mart set up exactly as the checklist says -- every join
discovered and checked against the rows, the date roles approved with their
defaults, the starter metrics accepted -- "inventory value by warehouse" was
refused with "the confirmed entity graph cannot conform every fact", and
"what is our total stock on hand?" was handed to the model to write. Five
things stood between the question and the governed compiler that already
existed for it:

  * A metric's source tables were every table holding one of its columns:
    the daily inventory value, SUM(ON_HND_QTY * ITM_CST), also claimed the
    monthly fact because it has an ITM_CST too, and the planner then demanded
    a path from the monthly fact to the daily balance date.
  * Two planners bound "warehouse" to the same column; the merge kept the
    first one's copy, a tie it had marked optional, and dropped the second
    planner's joins with its duplicate. The field, left with no path, never
    reached the compiler.
  * The grouped compiler refused a balance read at its latest snapshot, the
    rule every stock question gets.
  * It wrote every join INNER, so its SQL failed the validator on a graph edge
    the data check had made LEFT.
  * A lone metric with no dimension was left to the scalar compiler, which
    reads rolling windows only.

And a total of a quantity must stay per unit of measure: the compiler now
groups it by the item's unit, as the validator requires.

Reaching more questions, the compilers must also know what they cannot write.
They write no member filter, no condition on a value and no breakdown the
planner left unbound, and each of these came back as a total over every
member: "stock on hand for BRASS ELBOW EA" as every item's stock, "stock by
warehouse for warehouses with more than 100 units" without the condition,
"which item has the most stock" and "sales by item group" with no item at
all. Such a question now goes to the planner. The window and the
period-comparison compilers had the same gaps and get the same rules.

These build a synthetic tenant with tests/answer_harness.py and ask through
the real pipeline; the warehouse and the model are the only stand-ins.
"""

from __future__ import annotations

import pytest

from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("answers")) as built:
        yield built


def _latest():
    return [row for row in harness.STOCK]


def _value_by_warehouse() -> dict[str, float]:
    totals: dict[str, float] = {}
    for whs, _itm, _byr, _crt, on_hand, _alc, cost in _latest():
        name = harness.WAREHOUSES[whs][1]
        totals[name] = totals.get(name, 0) + on_hand * cost
    return totals


def _stock_by_unit() -> dict[str, float]:
    totals: dict[str, float] = {}
    for _whs, itm, _byr, _crt, on_hand, _alc, _cost in _latest():
        unit = harness.ITEMS[itm][3]
        totals[unit] = totals.get(unit, 0) + on_hand
    return totals


def _pairs(rows: list[dict]) -> dict[str, float]:
    """{label: number} from rows of one text and one number."""
    out = {}
    for row in rows:
        text = next(str(v) for v in row.values() if isinstance(v, str))
        number = next(float(v) for v in row.values() if isinstance(v, (int, float)) and not isinstance(v, bool))
        out[text] = number
    return out


# ── Through the pipeline ─────────────────────────────────────────────────────

class TestTheProductWritesTheSql:

    @pytest.mark.parametrize("question,lang", [
        ("Inventory value by warehouse", "en"),
        ("Valeur du stock par entrepôt", "fr"),
    ])
    def test_a_value_by_warehouse(self, warehouse, question, lang):
        answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert _pairs(answer["rows"]) == pytest.approx(_value_by_warehouse())

    def test_it_reads_the_metric_s_own_table_and_its_latest_snapshot(self, warehouse):
        # The monthly fact also has an ITM_CST; the earlier snapshot holds
        # 1,000 more of everything.
        sql = harness.ask(warehouse, "Inventory value by warehouse")["sql"].upper()
        assert "ITM_BAL_DLY_FCT" in sql and "ITM_BAL_PRD_FCT" not in sql
        assert "IN (SELECT SNAPSHOT_DATE FROM SNAPSHOT_DATES)" in sql

    def test_its_joins_are_the_ones_the_graph_requires(self, warehouse):
        sql = harness.ask(warehouse, "Inventory value by warehouse")["sql"].upper()
        assert "LEFT JOIN [MART].[DT_DMS]" in sql

    def test_one_value_is_one_number(self, warehouse):
        answer = harness.ask(warehouse, "What is our inventory value?")
        assert answer["model_wrote_sql"] is False
        assert [list(row.values()) for row in answer["rows"]] == [[pytest.approx(sum(_value_by_warehouse().values()))]]

    @pytest.mark.parametrize("question,lang", [
        ("What is our total stock on hand?", "en"),
        ("Quel est notre stock total en main ?", "fr"),
    ])
    def test_a_quantity_is_totalled_per_unit(self, warehouse, question, lang):
        answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert _pairs(answer["rows"]) == pytest.approx(_stock_by_unit())

    def test_across_all_units_when_asked(self, warehouse):
        answer = harness.ask(warehouse, "What is our total stock on hand across all units?")
        assert answer["model_wrote_sql"] is False
        assert [list(row.values()) for row in answer["rows"]] == [[pytest.approx(sum(_stock_by_unit().values()))]]


class TestWhatItDoesNotAnswerForEveryMember:

    @pytest.mark.parametrize("question,lang", [
        ("What is the stock on hand for BRASS ELBOW EA?", "en"),
        ("Quel est le stock en main pour BRASS ELBOW EA ?", "fr"),
        ("What is the inventory value by warehouse for warehouses with more than 2000?", "en"),
        ("Quelle est la valeur du stock par entrepôt pour les entrepôts de plus de 2000 ?", "fr"),
        ("Quel article a le plus de stock en main ?", "fr"),
    ])
    def test_it_goes_to_the_planner(self, warehouse, question, lang):
        answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is True
        assert not any("ON_HND_QTY" in run["sql"].upper() for run in answer["executed"])


# ── The pieces ───────────────────────────────────────────────────────────────

_COLUMNS = {
    "MART.ITM_BAL_DLY_FCT": {"ITM_BAL_DLY_FCT_KEY": "bigint", "WHS_DMS_KEY": "int", "ITM_BAL_EFC_DT_DMS_KEY": "int",
                             "ON_HND_QTY": "decimal", "SLD_QTY": "decimal"},
    "MART.WHS_DMS": {"WHS_DMS_KEY": "int", "WHS_DSC": "varchar"},
    "MART.DT_DMS": {"DT_DMS_KEY": "int", "DMS_DT": "date"},
}


def _snapshot_context(metric_name: str, template: str) -> dict:
    return {
        "question": "by warehouse",
        "metric_formulas": [{"name": metric_name, "formula_type": "expression", "sql_template": template,
                             "base_table": "MART.ITM_BAL_DLY_FCT"}],
        "analytical_request_plan": {"status": "compiled", "intent": "metric_query",
                                    "source_fact": "MART.ITM_BAL_DLY_FCT", "source_facts": ["MART.ITM_BAL_DLY_FCT"]},
        "semantic_plan": {
            "fields": [{"term": "warehouse", "table": "MART.WHS_DMS", "column": "WHS_DSC",
                        "role": "display_dimension", "enforcement": "required", "display_required": True}],
            "joins": [{"from": "MART.ITM_BAL_DLY_FCT", "to": "MART.WHS_DMS", "conditions": [["WHS_DMS_KEY", "WHS_DMS_KEY"]],
                       "enforcement": "required"},
                      {"from": "MART.ITM_BAL_DLY_FCT", "to": "MART.DT_DMS",
                       "conditions": [["ITM_BAL_EFC_DT_DMS_KEY", "DT_DMS_KEY"]], "enforcement": "required"}],
            "temporal_policies": [{"kind": "latest_snapshot", "amount": 1, "unit": "period",
                                   "fact_table": "MART.ITM_BAL_DLY_FCT", "fact_column": "ITM_BAL_EFC_DT_DMS_KEY",
                                   "dimension_table": "MART.DT_DMS", "dimension_key": "DT_DMS_KEY",
                                   "date_column": "DMS_DT", "date_key_type": "surrogate_fk",
                                   "role_alias": "balance_date"}],
        },
    }


def _window_context(question: str, fields: list[dict] | None = None) -> dict:
    return {
        "question": question,
        "canonical_question": question,
        "metric_formulas": [{"name": "Units sold", "formula_type": "expression", "sql_template": "SUM(SLD_QTY)",
                             "base_table": "MART.ITM_BAL_DLY_FCT"}],
        "analytical_request_plan": {"status": "compiled", "intent": "metric_query",
                                    "source_fact": "MART.ITM_BAL_DLY_FCT", "source_facts": ["MART.ITM_BAL_DLY_FCT"]},
        "semantic_plan": {
            "fields": list(fields or []),
            "joins": [{"from": "MART.ITM_BAL_DLY_FCT", "to": "MART.DT_DMS",
                       "conditions": [["ITM_BAL_EFC_DT_DMS_KEY", "DT_DMS_KEY"]], "enforcement": "required"}],
            "temporal_policies": [{"kind": "last_n", "amount": 3, "unit": "month",
                                   "fact_table": "MART.ITM_BAL_DLY_FCT", "fact_column": "ITM_BAL_EFC_DT_DMS_KEY",
                                   "dimension_table": "MART.DT_DMS", "dimension_key": "DT_DMS_KEY",
                                   "date_column": "DMS_DT", "date_key_type": "surrogate_fk",
                                   "role_alias": "balance_date"}],
        },
    }


_ITEM_OPTIONAL = {"term": "item", "table": "MART.ITM_DMS", "column": "ITM_NM", "role": "display_dimension",
                  "enforcement": "optional", "display_required": True}


def _compile(context):
    from core.pipeline_helpers import compile_governed_temporal_metric_sql

    return compile_governed_temporal_metric_sql("azure_sql", set(_COLUMNS), set(_COLUMNS), _COLUMNS, context)


class TestWhatTheCompilersLeaveToThePlanner:

    @staticmethod
    def _stock_by_warehouse(question: str = "stock on hand by warehouse", **extra) -> dict:
        context = _snapshot_context("Stock on hand", "SUM(ON_HND_QTY)")
        context["question"] = context["canonical_question"] = question
        context.update(extra)
        return context

    def test_the_breakdown_it_was_given_is_written(self):
        assert "[WHS_DSC] AS WAREHOUSE" in _compile(self._stock_by_warehouse())

    def test_a_member_the_question_names(self):
        assert _compile(self._stock_by_warehouse(named_members=["NORTH DEPOT"])) == ""

    def test_a_condition_on_a_value(self):
        assert _compile(self._stock_by_warehouse("stock on hand by warehouse over 1,000 units")) == ""

    def test_a_breakdown_nobody_bound(self):
        assert _compile(self._stock_by_warehouse("stock on hand by warehouse and by item")) == ""

    def test_a_field_asked_to_be_seen_that_the_planner_left_optional(self):
        context = self._stock_by_warehouse("stock on hand")
        context["semantic_plan"]["fields"][0]["enforcement"] = "optional"
        assert _compile(context) == ""

    def test_a_modifier_the_planner_demoted_is_not_asked_to_be_seen(self):
        context = self._stock_by_warehouse("stock on hand")
        context["semantic_plan"]["fields"][0].update(
            enforcement="optional", demotion_reason="event modifier is not requested output grain")
        assert "SUM(fact_rows.ON_HND_QTY) AS STOCK_ON_HAND" in _compile(context)


class TestTheWindowCompilerToo:

    def test_a_total_over_a_window_is_compiled(self):
        assert "SUM(SLD_QTY)" in _compile(_window_context("units sold in the last 3 months"))

    def test_a_breakdown_nobody_bound(self):
        assert _compile(_window_context("units sold by item in the last 3 months")) == ""

    def test_a_field_asked_to_be_seen_that_the_planner_left_optional(self):
        assert _compile(_window_context("units sold in the last 3 months", [_ITEM_OPTIONAL])) == ""


class TestThePeriodComparisonRepairToo:

    @staticmethod
    def _repair(context):
        from core.pipeline_helpers import attempt_governed_temporal_metric_repair

        return attempt_governed_temporal_metric_repair(
            "SELECT 1", "azure_sql", set(_COLUMNS), set(_COLUMNS), _COLUMNS, context)

    def test_a_comparison_is_compiled(self):
        assert "period_comparison" in self._repair(_window_context("compare units sold this month vs last month"))

    def test_a_breakdown_nobody_bound(self):
        assert self._repair(_window_context("compare units sold by item this month vs last month")) == ""

    def test_a_member_the_question_names(self):
        context = _window_context("compare units sold this month vs last month")
        context["named_members"] = ["BRASS ELBOW EA"]
        assert self._repair(context) == ""


def _canonical(question: str) -> dict:
    from core.question_normalizer import canonicalise

    return {"question": question, "canonical_question": canonicalise(question)}


class TestWhatIsAConditionOnAValue:

    @pytest.mark.parametrize("question", [
        "stock by warehouse for warehouses with more than 100 units", "items with stock over 1,000",
        "items below average stock", "sales above $5000", "items with less than 5% margin",
        "stock between 10 and 20 units", "items with at least 5 units",
        "Quel est notre stock pour les entrepôts de plus de 100 unités ?",
        "articles dont le stock est inférieur à 5", "articles au-dessus de 100",
    ])
    def test_a_condition(self, question):
        from core.pipeline_helpers import _left_to_the_planner

        assert _left_to_the_planner(_canonical(question)) == "states a condition on a value"

    @pytest.mark.parametrize("question", [
        "units sold over the last 6 months", "stock received more than 3 months ago",
        "Quel article a le plus de stock en main ?", "units sold between 2024 and 2025",
        "average stock on hand", "top 5 warehouses by stock", "stock on hand by warehouse",
    ])
    def test_words_that_are_not_one(self, question):
        from core.pipeline_helpers import _left_to_the_planner

        assert _left_to_the_planner(_canonical(question)) == ""


class TestWhatIsABreakdown:

    @pytest.mark.parametrize("question,count", [
        ("Inventory value by warehouse", 1), ("Valeur du stock par entrepôt", 1),
        ("ventes par groupe d'articles", 1), ("Which item has the most stock on hand?", 1),
        ("Quel article a le plus de stock en main ?", 1), ("stock on hand by warehouse and by item", 2),
        ("sales by month and warehouse", 1), ("units sold by month in the last 6 months", 0),
        ("What is our total stock on hand across all units?", 0), ("What is our inventory value?", 0),
    ])
    def test_what_the_question_asks_to_be_broken_down_by(self, question, count):
        from core.pipeline_helpers import _requested_breakdowns

        assert _requested_breakdowns(_canonical(question)["canonical_question"]) == count


class TestTheLatestSnapshotRule:

    def _compile(self, context):
        from core.pipeline_helpers import compile_governed_temporal_metric_sql

        return compile_governed_temporal_metric_sql("azure_sql", set(_COLUMNS), set(_COLUMNS), _COLUMNS, context)

    def test_a_balance_is_read_at_its_last_snapshot(self):
        sql = self._compile(_snapshot_context("Stock on hand", "SUM(ON_HND_QTY)"))
        assert "snapshot_dates" in sql and "IN (SELECT snapshot_date FROM snapshot_dates)" in sql

    def test_a_flow_has_no_snapshot_to_be_read_at(self):
        # Summed under the snapshot rule it would add every date together.
        assert self._compile(_snapshot_context("Units sold", "SUM(SLD_QTY)")) == ""


class TestAMetricReadsItsOwnTable:

    COLUMNS = {"MART.ITM_BAL_DLY_FCT": {"ON_HND_QTY": "decimal", "ITM_CST": "decimal"},
               "MART.ITM_BAL_PRD_FCT": {"SLD_QTY": "decimal", "ITM_CST": "decimal"},
               "MART.ITM_DMS": {"ITM_LIST_PRC": "decimal"}}

    def test_a_column_its_table_holds_is_read_there(self):
        from core.metric_scope import metric_source_tables

        metric = {"base_table": "MART.ITM_BAL_DLY_FCT", "sql_template": "SUM(ON_HND_QTY * ITM_CST)",
                  "required_columns": "ON_HND_QTY, ITM_CST"}
        assert {t.split(".")[-1] for t in metric_source_tables(metric, self.COLUMNS)} == {"ITM_BAL_DLY_FCT"}

    def test_a_column_its_table_lacks_brings_the_table_that_has_it(self):
        from core.metric_scope import metric_source_tables

        metric = {"base_table": "MART.ITM_BAL_DLY_FCT", "sql_template": "SUM(ON_HND_QTY * ITM_LIST_PRC)",
                  "required_columns": "ON_HND_QTY, ITM_LIST_PRC"}
        assert {t.split(".")[-1] for t in metric_source_tables(metric, self.COLUMNS)} == {
            "ITM_BAL_DLY_FCT", "ITM_DMS"}


def _field(table, column, enforcement="required", **extra):
    return {"term": "warehouse", "table": table, "column": column, "role": "display_dimension",
            "enforcement": enforcement, **extra}


class TestTwoPlannersAgreeing:

    def test_a_field_one_planner_already_bound_keeps_the_other_s_joins(self):
        from core.pipeline_context import _merge_semantic_plans

        lexical = {"enabled": True, "fields": [_field("WH.MART.WHS_DMS", "WHS_DSC")], "joins": []}
        model = {"enabled": True, "fields": [_field("MART.WHS_DMS", "WHS_DSC"),
                                             {"term": "value", "table": "MART.ITM_BAL_DLY_FCT", "column": "ON_HND_QTY",
                                              "role": "measure", "enforcement": "required"}],
                 "joins": [{"from": "MART.ITM_BAL_DLY_FCT", "to": "MART.WHS_DMS",
                            "conditions": [["WHS_DMS_KEY", "WHS_DMS_KEY"]]}]}
        merged = _merge_semantic_plans(lexical, model)
        assert [(j["from"], j["to"]) for j in merged["joins"]] == [("MART.ITM_BAL_DLY_FCT", "MART.WHS_DMS")]

    def test_a_certain_binding_settles_a_tie(self):
        from core.pipeline_context import _merge_semantic_plans

        lexical = {"enabled": True, "joins": [],
                   "fields": [_field("WH.MART.WHS_DMS", "WHS_DSC", "optional", ambiguous_source=True)]}
        model = {"enabled": True, "joins": [], "fields": [_field("MART.WHS_DMS", "WHS_DSC")]}
        merged = _merge_semantic_plans(lexical, model)
        assert [f["enforcement"] for f in merged["fields"]] == ["required"]

    def test_a_tie_nobody_settled_stays_a_hint(self):
        from core.pipeline_context import _merge_semantic_plans

        lexical = {"enabled": True, "joins": [],
                   "fields": [_field("WH.MART.WHS_DMS", "WHS_DSC", "optional", ambiguous_source=True)]}
        model = {"enabled": True, "joins": [], "fields": [_field("MART.WHS_DMS", "WHS_DSC", "optional")]}
        assert [f["enforcement"] for f in _merge_semantic_plans(lexical, model)["fields"]] == ["optional"]


class TestTheGraphDecidesTheJoin:

    CONTEXT = {"graph_context": {"resolved_edges": [
        {"from_schema": "MART", "from_table": "ITM_BAL_DLY_FCT", "to_schema": "MART", "to_table": "DT_DMS",
         "join_type": "LEFT"},
        {"from_schema": "MART", "from_table": "ITM_BAL_DLY_FCT", "to_schema": "MART", "to_table": "WHS_DMS",
         "join_type": "INNER"},
    ]}}

    def test_left_where_the_graph_says_left(self):
        from core.pipeline_helpers import _graph_join_type

        assert _graph_join_type(self.CONTEXT, "MART.ITM_BAL_DLY_FCT", "MART.DT_DMS") == "LEFT JOIN"
        assert _graph_join_type(self.CONTEXT, "MART.ITM_BAL_DLY_FCT", "MART.WHS_DMS") == "JOIN"

    def test_an_outer_edge_the_other_way_round_is_not_rewritten(self):
        from core.pipeline_helpers import _graph_join_type

        assert _graph_join_type(self.CONTEXT, "MART.DT_DMS", "MART.ITM_BAL_DLY_FCT") is None


class TestAQuantityKeepsItsUnit:

    POLICIES = [{"kind": "units_of_measure", "fact_table": "MART.ITM_BAL_DLY_FCT",
                 "quantities": ["ON_HND_QTY", "ALC_ON_HND_QTY"],
                 "unit_columns": [{"table": "MART.ITM_BAL_DLY_FCT", "column": "UNT_OF_MSR"},
                                  {"table": "MART.ITM_DMS", "column": "UNT_OF_MSR"}],
                 "item_columns": [{"table": "MART.ITM_BAL_DLY_FCT", "column": "ITM_DMS_KEY"}],
                 "unit_joins": [{"table": "MART.ITM_DMS", "fact_column": "ITM_DMS_KEY", "key": "ITM_DMS_KEY"}]}]

    def test_a_total_of_a_quantity_is_kept_by_the_item_s_unit(self):
        from core.units_of_measure import unit_for_total

        assert unit_for_total("SUM(ON_HND_QTY)", "MART.ITM_BAL_DLY_FCT", self.POLICIES) == {
            "table": "MART.ITM_DMS", "column": "UNT_OF_MSR",
            "join": {"table": "MART.ITM_DMS", "fact_column": "ITM_DMS_KEY", "key": "ITM_DMS_KEY"}}

    def test_a_value_adds_up_across_units(self):
        from core.units_of_measure import unit_for_total

        assert unit_for_total("SUM(ON_HND_QTY * ITM_CST)", "MART.ITM_BAL_DLY_FCT", self.POLICIES) is None

    def test_another_fact_s_total_is_not_this_policy_s(self):
        from core.units_of_measure import unit_for_total

        assert unit_for_total("SUM(ON_HND_QTY)", "MART.ITM_BAL_PRD_FCT", self.POLICIES) is None
