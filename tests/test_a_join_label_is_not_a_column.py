"""
A join's label is a phrase, never a column.

In the join graph a relationship carries a label: a verb or a role -- "places",
"bought by", a date role's name. Whenever an admin confirmed or edited a join,
admin/routes.py wrote that label into the semantic model's display_column for
the join, the model kept it through every rebuild, and the SQL prompt printed
it as an instruction: "prefer display MART.CUSTOMER_DIM.bought by" -- a column
that does not exist, handed to the model as governed guidance.

The label now goes where the model keeps a join's phrases (use_when), which
is what it is. A display column is kept only when it is a column of the table
the join reaches -- when the model is patched, when it is rebuilt, and when
the prompt is written, so a model saved before this change stops misleading
the SQL model without being rebuilt.

A synthetic star, modelled by the product's own builder in a temp directory;
the store is a scratch database.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest


def _table(key, *columns):
    return {"columns": [{"name": n, "type": t, "nullable": False, "comment": ""} for n, t in columns],
            "pk_columns": [key], "row_count": 1000, "comment": "", "schema": "MART", "database": "WH"}


SCHEMA = {
    "WH.MART.SALES_FACT": _table("SALES_KEY", ("SALES_KEY", "bigint"), ("CUSTOMER_SK", "int"),
                                 ("NET_AMT", "decimal(18,2)")),
    "WH.MART.CUSTOMER_DIM": _table("CUSTOMER_SK", ("CUSTOMER_SK", "int"), ("CUSTOMER_NAME", "varchar(80)"),
                                   ("CUSTOMER_LEGAL_NAME", "varchar(120)")),
}
JOIN = dict(from_table="MART.SALES_FACT", to_table="MART.CUSTOMER_DIM",
            from_column="CUSTOMER_SK", to_column="CUSTOMER_SK")


@pytest.fixture
def kb():
    from core.semantic_model import write_semantic_model

    kb_dir = tempfile.mkdtemp()
    with open(os.path.join(kb_dir, "_schema.json"), "w", encoding="utf-8") as handle:
        json.dump(SCHEMA, handle)
    write_semantic_model(schema_dir=kb_dir, kb_dir=kb_dir, account_id="acct")
    return kb_dir


def _join(kb_dir):
    from core.semantic_model import load_semantic_model

    return next(r for r in load_semantic_model(kb_dir)["relationships"]
                if str(r.get("to_table")).upper().endswith("CUSTOMER_DIM")
                and {"from_column": "CUSTOMER_SK", "to_column": "CUSTOMER_SK"} in
                [{k: str(v).upper() for k, v in c.items()} for c in r.get("conditions") or []])


def _prompt(kb_dir):
    from core.semantic_model import build_runtime_semantic_context

    return build_runtime_semantic_context(kb_dir, question="net amount of sales bought by customer", max_lines=40)


def _pollute(kb_dir, value="bought by"):
    """A model saved before this change: an approved join with its label as display column."""
    from core.semantic_model import MODEL_JSON, load_semantic_model

    model = load_semantic_model(kb_dir)
    for rel in model["relationships"]:
        if str(rel.get("to_table")).upper().endswith("CUSTOMER_DIM"):
            rel.update(display_column=value, status="approved", use_when=[*rel.get("use_when", []), value])
    with open(os.path.join(kb_dir, MODEL_JSON), "w", encoding="utf-8") as handle:
        json.dump(model, handle)


class TestPatchingAJoin:

    def test_the_label_becomes_a_phrase_and_not_the_display_column(self, kb):
        from core.semantic_model import patch_relationship

        patch_relationship(kb_dir=kb, **JOIN, label="bought by", status="approved")
        rel = _join(kb)
        assert rel.get("display_column") != "bought by"
        assert "bought by" in rel.get("use_when", [])

    def test_a_label_passed_as_a_display_column_is_not_kept_as_one(self, kb):
        from core.semantic_model import patch_relationship

        patch_relationship(kb_dir=kb, **JOIN, display_column="bought by", status="approved")
        assert _join(kb).get("display_column") != "bought by"

    def test_a_real_column_is_kept_as_the_display_column(self, kb):
        from core.semantic_model import patch_relationship

        patch_relationship(kb_dir=kb, **JOIN, display_column="CUSTOMER_LEGAL_NAME", status="approved")
        assert _join(kb)["display_column"] == "CUSTOMER_LEGAL_NAME"

    def test_confirming_again_clears_a_label_saved_as_a_column(self, kb):
        from core.semantic_model import patch_relationship

        _pollute(kb)
        patch_relationship(kb_dir=kb, **JOIN, label="bought by", status="approved")
        assert _join(kb).get("display_column") != "bought by"


class TestTheSqlPromptNamesOnlyRealColumns:

    def test_a_model_saved_before_this_change_does_not_tell_the_model_to_display_a_verb(self, kb):
        _pollute(kb)
        prompt = _prompt(kb)
        assert "Relationship" in prompt                       # the join itself is still offered
        assert "bought by" not in [line.split("prefer display ")[-1].split(".")[-1]
                                   for line in prompt.splitlines() if "prefer display" in line]

    def test_a_real_display_column_is_still_recommended(self, kb):
        _pollute(kb, value="CUSTOMER_LEGAL_NAME")
        assert "prefer display MART.CUSTOMER_DIM.CUSTOMER_LEGAL_NAME" in _prompt(kb)


class TestARebuild:

    def test_a_rebuild_does_not_carry_a_label_forward_as_a_column(self, kb):
        from core.semantic_model import write_semantic_model

        _pollute(kb)
        write_semantic_model(schema_dir=kb, kb_dir=kb, account_id="acct")
        rel = _join(kb)
        assert rel.get("display_column") != "bought by"
        assert rel.get("status") == "approved"                # the approval itself survives

    def test_a_rebuild_keeps_an_approved_real_display_column(self, kb):
        from core.semantic_model import write_semantic_model

        assert _join(kb)["display_column"] != "CUSTOMER_LEGAL_NAME"     # not the builder's own pick
        _pollute(kb, value="CUSTOMER_LEGAL_NAME")
        write_semantic_model(schema_dir=kb, kb_dir=kb, account_id="acct")
        assert _join(kb)["display_column"] == "CUSTOMER_LEGAL_NAME"


class TestConfirmingAJoinInTheGraph:

    def test_the_admin_s_confirmation_puts_the_label_among_the_phrases(self, kb):
        import store
        from admin.routes import _sync_relationship_to_model

        store.init_db()
        account_id = f"acct{os.urandom(4).hex()}"
        store.upsert_client(account_id, "Test Ltd")
        store.update_client_state(account_id, "READY", {"schema_dir": kb, "kb_dir": kb})
        store.save_entity(account_id=account_id, entity_name="Sales", table_name="SALES_FACT", schema_name="MART")
        store.save_entity(account_id=account_id, entity_name="Customer", table_name="CUSTOMER_DIM", schema_name="MART")
        _sync_relationship_to_model(account_id, {
            "from_entity": "Sales", "to_entity": "Customer", "from_column": "CUSTOMER_SK",
            "to_column": "CUSTOMER_SK", "join_type": "LEFT", "label": "bought by"})
        rel = _join(kb)
        assert rel.get("display_column") != "bought by" and "bought by" in rel.get("use_when", [])
        assert rel["status"] == "approved"


class TestAcceptingAGraphChatProposal:

    def test_an_accepted_edit_puts_the_label_among_the_phrases_too(self, kb):
        import asyncio
        from unittest.mock import MagicMock, patch

        import store
        import admin.routes as routes

        store.init_db()
        account_id = f"acct{os.urandom(4).hex()}"
        store.upsert_client(account_id, "Test Ltd")
        store.update_client_state(account_id, "READY", {"schema_dir": kb, "kb_dir": kb})
        store.save_entity(account_id=account_id, entity_name="Sales", table_name="SALES_FACT", schema_name="MART")
        store.save_entity(account_id=account_id, entity_name="Customer", table_name="CUSTOMER_DIM", schema_name="MART")
        rel_id = store.save_relationship(account_id, "Sales", "Customer", "CUSTOMER_SK", "CUSTOMER_SK",
                                         join_type="LEFT")
        before = dict(store.get_relationship(account_id, rel_id))
        proposal_id = store.create_graph_change_proposal(
            account_id, "edit_join", "relationship", str(rel_id), before, dict(before, label="bought by"))
        with patch.object(routes, "_is_auth", return_value=True), patch.object(routes, "_after_semantic_approval"):
            asyncio.run(routes.graph_accept_change_proposal(MagicMock(), account_id, proposal_id))
        rel = _join(kb)
        assert rel.get("display_column") != "bought by" and "bought by" in rel.get("use_when", [])
