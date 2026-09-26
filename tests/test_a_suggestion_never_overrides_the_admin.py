"""
A suggestion never overrides what an admin confirmed or rejected in the join graph.

The graph keeps, for every table, column role and join, whether the admin
confirmed it, rejected it, or has not looked yet. The join resolver reads only
confirmed rows. Three writers ignored that:

  * The LLM's graph Suggest saved every table and column it classified as
    'suggested' through upserts that overwrote the row. A confirmed table went
    back to 'suggested' -- out of the resolver -- with the model's name and
    description over the admin's, its row filter cleared and its node moved.
    A confirmed column role was demoted the same way, and a rejected table or
    column came back as a fresh suggestion.
  * A rejection sets a join inactive, and both relationship upserts looked
    only at active rows, so the next graph sync or Suggest inserted the join
    again: the admin rejected it on every KB build, even when a declared
    foreign key was the source.
  * Suggest saved each role-playing join (Order Date, Ship Date...) with an
    insert, so every run added another copy of every one.

These drive the store and the real Suggest route against a scratch database,
with only the model's reply stood in for.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def account():
    import store

    store.init_db()
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    return account_id


def _entity(account_id, name="Customer", **overrides):
    import store

    values = dict(table_name="CUSTOMER_DIM", schema_name="MART", pk_column="CUSTOMER_KEY",
                  display_name="Customer", description="Who bought", entity_type="dimension",
                  pos_x=300.0, pos_y=200.0, entity_filter="IS_TEST = 0")
    values.update(overrides)
    return store.save_entity(account_id=account_id, entity_name=name, **values)


def _suggest_entity(account_id, name="Customer"):
    """What the LLM's Suggest writes for a table it classified."""
    import store

    store.save_entity(account_id=account_id, entity_name=name, table_name="CUSTOMER_DIM", schema_name="MART",
                      display_name="Client (AI)", description="AI guess", entity_type="dimension",
                      pos_x=880.0, pos_y=400.0, confidence_score=70, status="suggested", generated_by="llm")


# ── Tables ───────────────────────────────────────────────────────────────────

class TestATableTheAdminDecidedStaysDecided:

    def test_a_confirmed_table_keeps_its_status_its_words_its_filter_and_its_place(self, account):
        import store

        _entity(account)
        _suggest_entity(account)
        row = store.get_entity(account, "Customer")
        assert (row["status"], row["display_name"], row["description"], row["entity_filter"]) == \
            ("confirmed", "Customer", "Who bought", "IS_TEST = 0")
        assert (row["pos_x"], row["pos_y"]) == (300.0, 200.0)

    def test_a_rejected_table_is_not_suggested_again(self, account):
        import store

        _entity(account, status="rejected", is_active=0)
        _suggest_entity(account)
        row = store.get_entity(account, "Customer")
        assert (row["status"], row["is_active"]) == ("rejected", 0)

    def test_a_table_still_awaiting_review_takes_the_newer_suggestion(self, account):
        import store

        _entity(account, status="suggested", display_name="Old guess")
        _suggest_entity(account)
        assert store.get_entity(account, "Customer")["display_name"] == "Client (AI)"

    def test_the_admin_s_own_edit_still_applies(self, account):
        import store

        _entity(account)
        _entity(account, display_name="Buyer", entity_filter="")
        row = store.get_entity(account, "Customer")
        assert (row["display_name"], row["entity_filter"]) == ("Buyer", "")

    def test_the_admin_can_confirm_a_table_they_had_rejected(self, account):
        import store

        _entity(account, status="rejected", is_active=0)
        _entity(account)
        row = store.get_entity(account, "Customer")
        assert (row["status"], row["is_active"]) == ("confirmed", 1)


# ── Column roles ─────────────────────────────────────────────────────────────

def _property(account_id, **overrides):
    import store

    values = dict(role="metric", display_name="Net sales", synonyms="revenue")
    values.update(overrides)
    store.save_entity_property(account_id=account_id, entity_name="Sales", column_name="NET_AMT", **values)


def _suggest_property(account_id):
    import store

    store.save_entity_property(account_id=account_id, entity_name="Sales", column_name="NET_AMT",
                               role="dimension", display_name="Net amount (AI)", synonyms="",
                               confidence_score=60, status="suggested", generated_by="llm")


def _stored_property(account_id):
    import store

    return next(p for p in store.list_entity_properties(account_id, "Sales") if p["column_name"] == "NET_AMT")


class TestAColumnRoleTheAdminDecidedStaysDecided:

    def test_a_confirmed_role_keeps_its_status_and_its_words(self, account):
        _property(account)
        _suggest_property(account)
        prop = _stored_property(account)
        assert (prop["status"], prop["role"], prop["display_name"], prop["synonyms"]) == \
            ("confirmed", "metric", "Net sales", "revenue")

    def test_a_rejected_role_is_not_suggested_again(self, account):
        import store

        _property(account, status="suggested")
        store.reject_entity_property(account, "Sales", "NET_AMT")
        _suggest_property(account)
        assert _stored_property(account)["status"] == "rejected"

    def test_a_role_still_awaiting_review_takes_the_newer_suggestion(self, account):
        _property(account, status="suggested")
        _suggest_property(account)
        assert _stored_property(account)["display_name"] == "Net amount (AI)"


# ── Joins ────────────────────────────────────────────────────────────────────

def _rejected_join(account_id, key=""):
    import store

    rel_id = store.upsert_relationship_by_identity(
        account_id, "Sales", "Customer", "CUSTOMER_KEY", "CUSTOMER_KEY",
        relationship_key=key or "FK:SALES:CUSTOMER", status="suggested")
    with store.get_db() as conn:
        conn.execute("UPDATE entity_relationships SET status='rejected', is_active=0 WHERE id=?", (rel_id,))
    return rel_id


def _joins(account_id):
    import store

    return [(r["from_entity"], r["to_entity"], r["from_column"], r["status"], r["is_active"])
            for r in store.list_relationships(account_id, active_only=False)]


class TestAJoinTheAdminRejectedStaysRejected:

    def test_a_graph_sync_does_not_bring_back_a_rejected_key(self, account):
        import store

        rejected = _rejected_join(account)
        again = store.upsert_relationship_by_identity(
            account, "Sales", "Customer", "CUSTOMER_KEY", "CUSTOMER_KEY",
            relationship_key="FK:SALES:CUSTOMER", status="suggested")
        assert again == rejected
        assert _joins(account) == [("Sales", "Customer", "CUSTOMER_KEY", "rejected", 0)]

    def test_nor_does_a_declared_foreign_key(self, account):
        import store

        _rejected_join(account)
        store.upsert_relationship_by_identity(
            account, "Sales", "Customer", "CUSTOMER_KEY", "CUSTOMER_KEY",
            relationship_key="FK:SALES:CUSTOMER", status="confirmed", source_enforced=True,
            generated_by="db_schema")
        assert [j for j in _joins(account) if j[4] == 1] == []

    def test_the_suggest_upsert_does_not_bring_back_the_same_join(self, account):
        import store

        _rejected_join(account)
        store.upsert_relationship_by_pair(account, "Sales", "Customer", "CUSTOMER_KEY", "CUSTOMER_KEY",
                                          generated_by="llm")
        assert [j for j in _joins(account) if j[4] == 1] == []

    def test_a_different_join_between_the_same_tables_can_still_be_suggested(self, account):
        import store

        _rejected_join(account)
        store.upsert_relationship_by_pair(account, "Sales", "Customer", "BILL_TO_KEY", "CUSTOMER_KEY",
                                          generated_by="llm")
        assert ("Sales", "Customer", "BILL_TO_KEY", "suggested", 1) in _joins(account)

    def test_a_join_a_sync_merely_retired_is_not_a_rejection(self, account):
        import store

        rel_id = store.upsert_relationship_by_identity(
            account, "Sales", "Customer", "CUSTOMER_KEY", "CUSTOMER_KEY",
            relationship_key="FK:SALES:CUSTOMER", status="suggested")
        with store.get_db() as conn:
            conn.execute("UPDATE entity_relationships SET is_active=0 WHERE id=?", (rel_id,))
        store.upsert_relationship_by_identity(
            account, "Sales", "Customer", "CUSTOMER_KEY", "CUSTOMER_KEY",
            relationship_key="FK:SALES:CUSTOMER", status="suggested")
        assert ("Sales", "Customer", "CUSTOMER_KEY", "suggested", 1) in _joins(account)


# ── The Suggest route itself ─────────────────────────────────────────────────

SCHEMA = {
    "SALES_FACT": {"schema": "MART", "columns": [
        {"name": "SALES_KEY", "type": "bigint"}, {"name": "CUSTOMER_KEY", "type": "int"},
        {"name": "ORDER_DATE_KEY", "type": "int"}, {"name": "SHIP_DATE_KEY", "type": "int"},
        {"name": "NET_AMT", "type": "decimal(18,2)"}]},
    "CUSTOMER_DIM": {"schema": "MART", "columns": [
        {"name": "CUSTOMER_KEY", "type": "int"}, {"name": "CUSTOMER_NAME", "type": "varchar(80)"}]},
    "DATE_DIM": {"schema": "MART", "columns": [
        {"name": "DATE_KEY", "type": "int"}, {"name": "CALENDAR_DATE", "type": "date"},
        {"name": "YEAR_NUM", "type": "int"}, {"name": "MONTH_NUM", "type": "int"}]},
}

MODEL_REPLY = {
    "entities": [
        {"entity_name": "Sales", "table_name": "SALES_FACT", "schema_name": "MART", "entity_type": "fact",
         "display_name": "Sales (AI)", "description": "AI", "confidence_score": 80,
         "fields": [{"column_name": "NET_AMT", "role": "dimension", "display_name": "Net amount (AI)",
                     "synonyms": [], "confidence_score": 60}]},
        {"entity_name": "Customer", "table_name": "CUSTOMER_DIM", "schema_name": "MART",
         "entity_type": "dimension", "display_name": "Client (AI)", "description": "AI", "confidence_score": 80,
         "fields": [{"column_name": "CUSTOMER_NAME", "role": "dimension", "display_name": "Customer name",
                     "synonyms": ["client"], "confidence_score": 75}]},
    ],
    "relationships": [
        {"from_entity": "Sales", "to_entity": "Customer", "from_column": "CUSTOMER_KEY",
         "to_column": "CUSTOMER_KEY", "relationship_type": "many_to_one", "join_type": "LEFT",
         "label": "bought by", "confidence_score": 70},
    ],
}


def _run_suggest(account_id):
    import admin.routes as routes

    async def _model(*args, **kwargs):
        return json.dumps(MODEL_REPLY), 0, 0

    with patch.object(routes, "_is_auth", return_value=True), \
            patch.object(routes, "_after_semantic_approval"), \
            patch("core.llm.llm_complete", side_effect=_model), \
            patch("core.llm.resolve_provider", return_value=("test", "test", "k", {})):
        response = asyncio.run(routes.graph_suggest(MagicMock(), account_id))
    return json.loads(response.body)


@pytest.fixture
def workspace(account):
    import store

    schema_dir = tempfile.mkdtemp()
    with open(os.path.join(schema_dir, "_schema.json"), "w", encoding="utf-8") as handle:
        json.dump(SCHEMA, handle)
    store.update_client_state(account, "READY", {"schema_dir": schema_dir, "kb_dir": schema_dir})
    return account


class TestTheSuggestRoute:

    def test_it_leaves_the_admin_s_table_column_and_rejected_join_as_they_were(self, workspace):
        import store

        _entity(workspace)
        _property(workspace)
        _rejected_join(workspace, key="")
        with store.get_db() as conn:     # rejected as the review route does, on the pair Suggest proposes
            conn.execute("UPDATE entity_relationships SET relationship_key='' WHERE account_id=?", (workspace,))
        assert _run_suggest(workspace).get("status") != "error"
        customer = store.get_entity(workspace, "Customer")
        assert (customer["status"], customer["display_name"], customer["entity_filter"]) == \
            ("confirmed", "Customer", "IS_TEST = 0")
        assert (_stored_property(workspace)["status"], _stored_property(workspace)["role"]) == ("confirmed", "metric")
        assert not [j for j in _joins(workspace) if j[:3] == ("Sales", "Customer", "CUSTOMER_KEY") and j[4] == 1]

    def test_running_it_twice_adds_no_second_copy_of_a_role_playing_join(self, workspace):
        import store

        _run_suggest(workspace)
        first = sorted((r["to_entity"], r["from_column"]) for r in store.list_relationships(workspace)
                       if r["from_column"] in ("ORDER_DATE_KEY", "SHIP_DATE_KEY"))
        _run_suggest(workspace)
        second = sorted((r["to_entity"], r["from_column"]) for r in store.list_relationships(workspace)
                        if r["from_column"] in ("ORDER_DATE_KEY", "SHIP_DATE_KEY"))
        assert len(first) == 2 and second == first

    def test_a_column_role_the_model_proposed_says_it_came_from_the_model(self, workspace):
        import store

        _run_suggest(workspace)
        prop = next(p for p in store.list_entity_properties(workspace, "Customer")
                    if p["column_name"] == "CUSTOMER_NAME")
        assert (prop["status"], prop["generated_by"]) == ("suggested", "llm")
