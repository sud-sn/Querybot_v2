"""
A word that begins a longer name the question says is that name's.

"Inventory value by product group" was never answered on the sample tenant, in
English or French. The item's dimension is also a product -- its key's prefix
is read both ways -- and a dimension is asked for by the last word of its name
("receipts by division"), so "by product" was taken for the item, though the
question went on to say "product group", the name of a dimension of its own.
The answer was asked to group by the item and the product group at once, and
the compiler declined.

A word the question asks by, where the words after it complete a longer name
some dimension is asked for by, is that name's. "By product" alone, and "which
product", are still the item.

The semantic model and both planners are the product's own, over a synthetic
mart of the sample's shape.
"""

from __future__ import annotations

import json

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

    root = tmp_path_factory.mktemp("product-words")
    (root / "schema").mkdir()
    (root / "schema" / "_schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
    write_semantic_model(schema_dir=str(root / "schema"), kb_dir=str(root / "kb"))
    return str(root / "kb")


@pytest.fixture
def distribution_words():
    """French read as the distribution pack reads it: "groupe de produits" is
    a product group."""
    from core.vocab_packs import _clone_builtin, _merge_pack, activate_vocab, deactivate_vocab, load_pack

    vocab = _clone_builtin()
    _merge_pack(vocab, load_pack("wholesale_distribution"), "wholesale_distribution")
    token = activate_vocab(vocab)
    try:
        yield
    finally:
        deactivate_vocab(token)


def _english(question: str, lang: str) -> str:
    from core.question_normalizer import canonical_question

    return canonical_question(question, lang) if lang != "en" else question


def _model_tables(kb: str, question: str, lang: str = "en") -> list[str]:
    from core.semantic_model import build_runtime_semantic_plan

    plan = build_runtime_semantic_plan(kb, question=_english(question, lang))
    return [field["table"].split(".")[-1] for field in plan["fields"] if field.get("role") == "display_dimension"]


def _merged(kb: str, question: str, lang: str = "en") -> dict:
    from core.pipeline_context import _merge_semantic_plans
    from core.semantic_model import build_runtime_semantic_plan
    from core.semantic_planner import build_semantic_field_plan

    english = _english(question, lang)
    return _merge_semantic_plans(build_semantic_field_plan(english, COLUMNS, None, preferred_fact_tables={FACT}),
                                 build_runtime_semantic_plan(kb, question=english))


class TestThePlan:

    def test_by_product_group(self, kb):
        assert _model_tables(kb, "Inventory value by product group") == ["PDC_GRP_DMS"]

    def test_in_french(self, kb, distribution_words):
        assert _model_tables(kb, "Valeur du stock par groupe de produits", "fr") == ["PDC_GRP_DMS"]

    @pytest.mark.parametrize("question", [
        "Inventory value by product",
        "Which product has the most stock on hand",
    ])
    def test_a_product_alone_is_still_the_item(self, kb, question):
        assert _model_tables(kb, question) == ["ITM_DMS"]


class TestTheAnswer:

    def test_by_product_group_is_compiled(self, kb):
        from core.pipeline_helpers import compile_governed_temporal_metric_sql

        plan = _merged(kb, "Inventory value by product group")
        columns = {fqn.split(".", 1)[1]: types for fqn, types in COLUMNS.items()}
        policy = {"kind": "latest_snapshot", "amount": 1, "unit": "period", "fact_table": "MART.ITM_BAL_DLY_FCT",
                  "fact_column": "ITM_BAL_EFC_DT_DMS_KEY", "dimension_table": "MART.DT_DMS",
                  "dimension_key": "DT_DMS_KEY", "date_column": "DMS_DT", "date_key_type": "surrogate_fk",
                  "role_alias": "balance_date"}
        context = {
            "question": "inventory value by product group",
            "canonical_question": "inventory value by product group",
            "metric_formulas": [{"name": "Inventory value", "formula_type": "expression",
                                 "sql_template": "SUM(ON_HND_QTY * ITM_CST)", "base_table": "MART.ITM_BAL_DLY_FCT"}],
            "analytical_request_plan": {"status": "compiled", "intent": "metric_query",
                                        "source_fact": "MART.ITM_BAL_DLY_FCT",
                                        "source_facts": ["MART.ITM_BAL_DLY_FCT"]},
            "graph_context": {"resolved_edges": [
                {"from_schema": "MART", "from_table": "ITM_BAL_DLY_FCT", "to_schema": "MART",
                 "to_table": "PDC_GRP_DMS", "conditions": [["PDC_GRP_DMS_KEY", "PDC_GRP_DMS_KEY"]],
                 "join_type": "LEFT"}]},
            "semantic_plan": {**plan, "joins": [*(plan.get("joins") or []), {
                "from": "MART.ITM_BAL_DLY_FCT", "to": "MART.DT_DMS", "enforcement": "required",
                "conditions": [["ITM_BAL_EFC_DT_DMS_KEY", "DT_DMS_KEY"]]}], "temporal_policies": [policy]},
        }
        sql = compile_governed_temporal_metric_sql("azure_sql", set(columns), set(columns), columns, context)
        assert "business_dimension.[PDC_GRP_DSC] AS PRODUCT_GROUP" in sql
        assert "ITM_NM" not in sql


class TestTheWords:

    @pytest.mark.parametrize("question,asked", [
        ("inventory value by product group", False),
        ("inventory value by product groups", False),
        ("which product group has the most stock", False),
        ("inventory value by product", True),
        ("inventory value by product, then by month", True),
        ("inventory value by product grouping", True),
        ("inventory value by product group and by product", True),
    ])
    def test_a_word_is_its_own_only_where_no_longer_name_goes_on(self, question, asked):
        from core.semantic_model import _asked_by

        assert _asked_by(question, "product", {"product group", "item group", "product"}) is asked

    def test_with_no_longer_names_a_word_is_asked_by_as_before(self):
        from core.semantic_model import _asked_by

        assert _asked_by("inventory value by product group", "product") is True
