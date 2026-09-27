"""
The table named for a phrase is its one breakdown.

"Top 5 product groups by inventory value" was never answered on the sample
tenant, in English or French. Its snapshot keeps an item group and a product
group, and the two planners read "product group" as two tables: the lexical
planner as the item group, which the vocabulary also calls a product group,
and the semantic model as the product group's own table. The merged plan kept
both, the answer was asked to group by both, and the compiler, handed two
breakdowns for one phrase, declined; the question went to the model.

Where one of the tables a phrase is read as is named for it -- as the tenant's
vocabulary spells the table's name -- it is the breakdown, and the other
readings are set aside. Where none is, or several are, nothing is decided.

Both planners and the merge are the product's own, over a synthetic mart of
the sample's shape.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

FACT = "WH.MART.ITM_BAL_DLY_FCT"


def _table(name: str, *columns: tuple[str, str]) -> dict:
    return {"database": "WH", "schema": "MART", "table": name,
            "columns": [{"name": column, "type": kind} for column, kind in columns], "pk_columns": [columns[0][0]]}


SCHEMA = {
    FACT: _table("ITM_BAL_DLY_FCT", ("ITM_BAL_DLY_FCT_KEY", "bigint"), ("ITM_DMS_KEY", "int"),
                 ("ITM_GRP_DMS_KEY", "int"), ("PDC_GRP_DMS_KEY", "int"), ("ITM_BAL_EFC_DT_DMS_KEY", "int"),
                 ("ON_HND_QTY", "decimal"), ("ITM_CST", "decimal")),
    "WH.MART.ITM_DMS": _table("ITM_DMS", ("ITM_DMS_KEY", "int"), ("ITM_CD", "nvarchar"), ("ITM_NM", "nvarchar"),
                              ("ITM_GRP_DMS_KEY", "int"), ("PDC_GRP_DMS_KEY", "int")),
    "WH.MART.ITM_GRP_DMS": _table("ITM_GRP_DMS", ("ITM_GRP_DMS_KEY", "int"), ("ITM_GRP_CD", "nvarchar"),
                                  ("ITM_GRP_DSC", "nvarchar")),
    "WH.MART.PDC_GRP_DMS": _table("PDC_GRP_DMS", ("PDC_GRP_DMS_KEY", "int"), ("PDC_GRP_CD", "nvarchar"),
                                  ("PDC_GRP_DSC", "nvarchar")),
    "WH.MART.DT_DMS": _table("DT_DMS", ("DT_DMS_KEY", "int"), ("DMS_DT", "date"), ("YR", "int")),
}
COLUMNS = {fqn: {column["name"]: column["type"] for column in meta["columns"]} for fqn, meta in SCHEMA.items()}


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    from core.semantic_model import write_semantic_model

    root = tmp_path_factory.mktemp("two-groups")
    (root / "schema").mkdir()
    (root / "schema" / "_schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
    write_semantic_model(schema_dir=str(root / "schema"), kb_dir=str(root / "kb"))
    return str(root / "kb")


@pytest.fixture
def distribution_words():
    """French read as the distribution pack reads it: "groupes de produits"
    are product groups."""
    from core.vocab_packs import _clone_builtin, _merge_pack, activate_vocab, deactivate_vocab, load_pack

    vocab = _clone_builtin()
    _merge_pack(vocab, load_pack("wholesale_distribution"), "wholesale_distribution")
    token = activate_vocab(vocab)
    try:
        yield
    finally:
        deactivate_vocab(token)


def _merged(kb: str, question: str, lang: str = "en") -> dict:
    """The plan the pipeline holds: both planners on the question's English,
    merged."""
    from core.pipeline_context import _merge_semantic_plans
    from core.question_normalizer import canonical_question
    from core.semantic_model import build_runtime_semantic_plan
    from core.semantic_planner import build_semantic_field_plan

    english = canonical_question(question, lang) if lang != "en" else question
    lexical = build_semantic_field_plan(english, COLUMNS, None, preferred_fact_tables={FACT})
    model = build_runtime_semantic_plan(kb, question=english)
    return _merge_semantic_plans(lexical, model)


def _breakdowns(plan: dict) -> list[tuple[str, str]]:
    return [(field["table"].split(".")[-1], field["column"]) for field in plan["fields"]
            if field.get("role") == "display_dimension" and field.get("enforcement") != "optional"]


def _set_aside(plan: dict) -> list[tuple[str, str]]:
    return [(field["table"].split(".")[-1], field.get("demotion_reason")) for field in plan["fields"]
            if field.get("enforcement") == "optional"]


class TestThePlan:

    def test_product_groups_are_the_product_group(self, kb):
        plan = _merged(kb, "Top 5 product groups by inventory value")
        assert _breakdowns(plan) == [("PDC_GRP_DMS", "PDC_GRP_DSC")]
        assert _set_aside(plan) == [("ITM_GRP_DMS", "another table is named for the phrase")]

    def test_in_french(self, kb, distribution_words):
        plan = _merged(kb, "Les 5 principaux groupes de produits par valeur du stock", "fr")
        assert _breakdowns(plan) == [("PDC_GRP_DMS", "PDC_GRP_DSC")]

    def test_item_groups_are_the_item_group(self, kb):
        plan = _merged(kb, "Top 5 item groups by inventory value")
        assert _breakdowns(plan) == [("ITM_GRP_DMS", "ITM_GRP_DSC")]
        assert _set_aside(plan) == []


def _field(term: str, table: str, column: str, **extra) -> dict:
    return {"term": term, "table": table, "column": column, "role": "display_dimension", "display_required": True,
            **extra}


def _merge(*fields: dict) -> list[tuple[str, str | None]]:
    from core.pipeline_context import _merge_semantic_plans

    plan = _merge_semantic_plans({"enabled": True, "joins": [], "fields": [dict(field) for field in fields]})
    return [(field["table"].split(".")[-1], field.get("enforcement")) for field in plan["fields"]]


class TestTheRule:

    def test_no_table_named_for_the_phrase_decides_nothing(self):
        assert _merge(_field("category", "MART.ITM_GRP_DMS", "ITM_GRP_DSC"),
                      _field("category", "MART.PDC_GRP_DMS", "PDC_GRP_DSC", enforcement="required")) == [
            ("ITM_GRP_DMS", None), ("PDC_GRP_DMS", "required")]

    def test_the_named_table_wins_whichever_planner_read_it(self):
        assert _merge(_field("product group", "MART.PDC_GRP_DMS", "PDC_GRP_DSC"),
                      _field("Product Group", "MART.ITM_GRP_DMS", "ITM_GRP_DSC", enforcement="required")) == [
            ("PDC_GRP_DMS", None), ("ITM_GRP_DMS", "optional")]

    def test_two_tables_named_for_the_phrase_decide_nothing(self):
        assert _merge(_field("product group", "SALES.PDC_GRP_DMS", "PDC_GRP_DSC"),
                      _field("Product Group", "MART.PDC_GRP_DMS", "PDC_GRP_DSC", enforcement="required")) == [
            ("PDC_GRP_DMS", None), ("PDC_GRP_DMS", "required")]

    def test_a_plural_is_the_same_phrase(self):
        assert _merge(_field("product groups", "MART.ITM_GRP_DMS", "ITM_GRP_DSC"),
                      _field("Product Group", "MART.PDC_GRP_DMS", "PDC_GRP_DSC", enforcement="required")) == [
            ("ITM_GRP_DMS", "optional"), ("PDC_GRP_DMS", "required")]

    def test_only_one_phrase_is_decided(self):
        # The item group is asked for by its own name beside the product group.
        assert _merge(_field("product group", "MART.PDC_GRP_DMS", "PDC_GRP_DSC"),
                      _field("item group", "MART.ITM_GRP_DMS", "ITM_GRP_DSC")) == [
            ("PDC_GRP_DMS", None), ("ITM_GRP_DMS", None)]

    def test_a_measure_is_not_a_breakdown(self):
        measure = _field("product group", "MART.ITM_GRP_DMS", "ITM_GRP_CNT", role="measure")
        assert _merge(_field("product group", "MART.PDC_GRP_DMS", "PDC_GRP_DSC"), measure) == [
            ("PDC_GRP_DMS", None), ("ITM_GRP_DMS", None)]

    def test_a_reading_already_set_aside_stays_as_it_was(self):
        hint = _field("product group", "MART.ITM_GRP_DMS", "ITM_GRP_DSC", enforcement="optional",
                      demotion_reason="a tie")
        from core.pipeline_context import _merge_semantic_plans

        plan = _merge_semantic_plans({"enabled": True, "joins": [], "fields": [
            _field("product group", "MART.PDC_GRP_DMS", "PDC_GRP_DSC"), hint]})
        assert plan["fields"][1]["demotion_reason"] == "a tie"

    def test_a_broken_vocabulary_costs_the_choice_not_the_plan(self):
        import core.source_resolution as source_resolution

        def broken(*args, **kwargs):
            raise RuntimeError("vocabulary unavailable")

        with patch.object(source_resolution, "_business_source_label", broken):
            assert _merge(_field("product group", "MART.ITM_GRP_DMS", "ITM_GRP_DSC"),
                          _field("product group", "MART.PDC_GRP_DMS", "PDC_GRP_DSC")) == [
                ("ITM_GRP_DMS", None), ("PDC_GRP_DMS", None)]


def _ranking_context(plan: dict) -> dict:
    """What the pipeline hands the compiler for "top 5 product groups by
    inventory value", with the merged plan and the graph's edges."""
    edges = [{"from_schema": "MART", "from_table": "ITM_BAL_DLY_FCT", "to_schema": "MART", "to_table": table,
              "conditions": [[key, key]], "join_type": "LEFT"}
             for table, key in (("PDC_GRP_DMS", "PDC_GRP_DMS_KEY"), ("ITM_GRP_DMS", "ITM_GRP_DMS_KEY"))]
    policy = {"kind": "latest_snapshot", "amount": 1, "unit": "period", "fact_table": "MART.ITM_BAL_DLY_FCT",
              "fact_column": "ITM_BAL_EFC_DT_DMS_KEY", "dimension_table": "MART.DT_DMS", "dimension_key": "DT_DMS_KEY",
              "date_column": "DMS_DT", "date_key_type": "surrogate_fk", "role_alias": "balance_date"}
    return {
        "question": "top 5 product groups by inventory value",
        "canonical_question": "top 5 product groups by inventory value",
        "metric_formulas": [{"name": "Inventory value", "formula_type": "expression",
                             "sql_template": "SUM(ON_HND_QTY * ITM_CST)", "base_table": "MART.ITM_BAL_DLY_FCT"}],
        "analytical_request_plan": {"status": "compiled", "intent": "ranking", "top_n": 5,
                                    "source_fact": "MART.ITM_BAL_DLY_FCT", "source_facts": ["MART.ITM_BAL_DLY_FCT"]},
        "graph_context": {"resolved_edges": edges},
        "semantic_plan": {**plan, "joins": [*(plan.get("joins") or []), {
            "from": "MART.ITM_BAL_DLY_FCT", "to": "MART.DT_DMS", "enforcement": "required",
            "conditions": [["ITM_BAL_EFC_DT_DMS_KEY", "DT_DMS_KEY"]]}], "temporal_policies": [policy]},
    }


class TestTheAnswer:

    def test_the_ranking_is_compiled_by_the_product_group(self, kb):
        from core.pipeline_helpers import compile_governed_temporal_metric_sql

        plan = _merged(kb, "Top 5 product groups by inventory value")
        columns = {fqn.split(".", 1)[1]: types for fqn, types in COLUMNS.items()}
        sql = compile_governed_temporal_metric_sql("azure_sql", set(columns), set(columns), columns,
                                                   _ranking_context(plan))
        assert "PDC_GRP_DSC" in sql and "ITM_GRP_DSC" not in sql
        assert "SELECT\n    TOP (5) business_dimension.[PDC_GRP_DSC] AS PRODUCT_GROUP" in sql
