"""
Editing the join graph changes what was edited, and nothing else.

Six ways the graph editor corrupted the model it edits:

  * Editing a table in its dialog -- the name, the description -- put it back
    at (120, 120) on the canvas and dropped its row filter, the WHERE clause
    every query through that table depends on. The dialog does not send
    either, and the save defaulted what it was not sent.
  * Editing a suggested join from the review queue posted it without its id,
    so the save added a second join beside the suggestion, which stayed.
  * Saving an edited suggestion left it 'suggested' in the graph while the
    semantic model recorded it approved.
  * Confirming a suggestion skipped the check every saved join gets, so a
    suggested fact-to-fact join became a confirmed one in one click; nor did
    the confirmation reach the semantic model.
  * The editor offered "OUTER JOIN", which the planner writes out as it
    stands -- valid SQL in no dialect.
  * Deleting a join kept no snapshot, and left it in the semantic model.

The real admin routes and store on a scratch database (QUERYBOT_DB_PATH), a
semantic model built by the real writer, and the editor's own JavaScript run in
a JavaScript engine.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import dukpy
import pytest

from core.semantic_model import load_semantic_model, write_semantic_model
from tests import js_lift

GRAPH_PAGE = Path(__file__).resolve().parents[1] / "admin" / "templates" / "client_graph.html"
SCHEMA = {
    "WH.MART.SLS_FCT": {"database": "WH", "schema": "MART", "table": "SLS_FCT", "columns": [
        {"name": "SLS_FCT_KEY", "type": "bigint"}, {"name": "CUS_DMS_KEY", "type": "int"},
        {"name": "NET_AMT", "type": "decimal"}]},
    "WH.MART.SHP_FCT": {"database": "WH", "schema": "MART", "table": "SHP_FCT", "columns": [
        {"name": "SHP_FCT_KEY", "type": "bigint"}, {"name": "SLS_FCT_KEY", "type": "bigint"},
        {"name": "SHP_QTY", "type": "decimal"}]},
    "WH.MART.CUS_DMS": {"database": "WH", "schema": "MART", "table": "CUS_DMS", "columns": [
        {"name": "CUS_DMS_KEY", "type": "int"}, {"name": "CUS_NM", "type": "varchar"}]},
}


@pytest.fixture
def graph(tmp_path):
    import store
    from core.pipeline_context import save_state

    store.init_db()
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    (tmp_path / "_schema.json").write_text(json.dumps(SCHEMA))
    kb_dir = str(tmp_path)
    write_semantic_model(schema_dir=kb_dir, kb_dir=kb_dir)
    save_state(account_id, "READY", {"kb_dir": kb_dir, "schema_dir": kb_dir})
    store.save_entity(account_id, "Sales", "SLS_FCT", schema_name="MART", entity_type="fact")
    store.save_entity(account_id, "Shipments", "SHP_FCT", schema_name="MART", entity_type="fact")
    store.save_entity(account_id, "Customer", "CUS_DMS", schema_name="MART", entity_type="dimension",
                      pos_x=300, pos_y=400, entity_filter="CUS_STS = 'A'")
    return account_id, kb_dir


def _call(route, *args, body=None):
    """An admin request to a real route; the contract recompile and the
    evaluation run an approval starts are not under test."""
    from admin import routes

    request = MagicMock()

    async def _json():
        return body or {}

    request.json = _json
    with patch.object(routes, "_is_auth", return_value=True), \
            patch.object(routes, "_after_semantic_approval"):
        return asyncio.run(getattr(routes, route)(request, *args))


def _suggest(account_id, from_entity, to_entity, column, to_column=None):
    import store

    return store.save_relationship(account_id, from_entity, to_entity, column, to_column or column,
                                   status="suggested", confidence_score=80, join_type="LEFT")


def _joins(account_id, from_entity, to_entity):
    import store

    return [rel for rel in store.list_relationships(account_id, active_only=False)
            if rel["from_entity"] == from_entity and rel["to_entity"] == to_entity]


def _model_joins(kb_dir, from_table, to_table):
    return [rel for rel in load_semantic_model(kb_dir).get("relationships") or []
            if rel.get("from_table") == from_table and rel.get("to_table") == to_table]


class TestEditingATable:

    def test_its_row_filter_and_position_survive_the_dialog(self, graph):
        import store

        account_id, _kb = graph
        _call("graph_api_entity_upsert", account_id, body={
            "entity_name": "Customer", "table_name": "CUS_DMS", "schema_name": "MART",
            "display_name": "Client", "description": "Who buys", "entity_type": "dimension"})
        customer = store.get_entity(account_id, "Customer")
        assert customer["display_name"] == "Client"
        assert (customer["pos_x"], customer["pos_y"]) == (300, 400)
        assert customer["entity_filter"] == "CUS_STS = 'A'"

    def test_what_the_request_does_send_is_saved(self, graph):
        import store

        account_id, _kb = graph
        _call("graph_api_entity_upsert", account_id, body={
            "entity_name": "Customer", "table_name": "CUS_DMS", "schema_name": "MART",
            "pos_x": 10, "pos_y": 20, "entity_filter": ""})
        customer = store.get_entity(account_id, "Customer")
        assert (customer["pos_x"], customer["pos_y"], customer["entity_filter"]) == (10, 20, "")


class TestEditingASuggestedJoin:

    def test_saving_it_confirms_that_join_rather_than_adding_one(self, graph):
        account_id, _kb = graph
        rel_id = _suggest(account_id, "Sales", "Customer", "CUS_DMS_KEY")
        response = _call("graph_api_rel_upsert", account_id, body={
            "rel_id": rel_id, "from_entity": "Sales", "to_entity": "Customer",
            "from_column": "CUS_DMS_KEY", "to_column": "CUS_DMS_KEY", "join_type": "INNER",
            "label": "Sold to"})
        assert response.status_code == 200
        joins = _joins(account_id, "Sales", "Customer")
        assert [(j["id"], j["status"], j["join_type"], j["label"]) for j in joins] == [
            (rel_id, "confirmed", "INNER", "Sold to")]

    @pytest.mark.parametrize("editing, expected", [(7, 7), (0, None)])
    def test_the_editor_posts_the_id_of_the_join_it_was_opened_on(self, editing, expected):
        source = js_lift.function(GRAPH_PAGE.read_text(encoding="utf-8"), "function joinSavePayload(")
        payload = dukpy.evaljs(source + "; joinSavePayload(dukpy.fields, dukpy.editing)", fields={
            "from_entity": "Sales", "to_entity": "Customer", "from_column": "CUS_DMS_KEY",
            "to_column": "CUS_DMS_KEY", "join_type": "LEFT", "relationship_type": "many_to_one"},
            editing=editing)
        assert payload.get("rel_id") == expected
        assert payload["join_conditions"] == [{"from_col": "CUS_DMS_KEY", "to_col": "CUS_DMS_KEY"}]


class TestConfirmingASuggestion:

    def test_a_fact_to_fact_join_is_refused_and_stays_a_suggestion(self, graph):
        account_id, _kb = graph
        rel_id = _suggest(account_id, "Shipments", "Sales", "SLS_FCT_KEY")
        response = _call("graph_confirm_rel", account_id, rel_id)
        assert response.status_code == 422
        assert "fact-to-fact" in json.loads(response.body)["message"]
        assert [j["status"] for j in _joins(account_id, "Shipments", "Sales")] == ["suggested"]

    def test_a_sound_join_is_confirmed_in_the_graph_and_the_model(self, graph):
        account_id, kb_dir = graph
        rel_id = _suggest(account_id, "Sales", "Customer", "CUS_DMS_KEY")
        assert _call("graph_confirm_rel", account_id, rel_id).status_code == 200
        assert [j["status"] for j in _joins(account_id, "Sales", "Customer")] == ["confirmed"]
        assert [rel["status"] for rel in _model_joins(kb_dir, "MART.SLS_FCT", "MART.CUS_DMS")] == ["approved"]


class TestJoinTypes:

    def test_outer_is_not_offered(self):
        page = GRAPH_PAGE.read_text(encoding="utf-8")
        script = "var ENTITY_COLUMNS = {};\n" + "\n".join(
            js_lift.function(page, signature) for signature in (
                "function _esc(", "function _jsArg(", "function _dlId(", "function _colEl(",
                "function _buildJoinEditForm("))
        html = dukpy.evaljs(script + "; _buildJoinEditForm(dukpy.rel, [])", rel={
            "id": 1, "from_entity": "Sales", "to_entity": "Customer", "join_type": "LEFT"})
        assert "LEFT JOIN" in html and "INNER JOIN" in html
        assert "OUTER" not in html

    def test_a_join_saved_as_outer_is_stored_as_left(self, graph):
        account_id, _kb = graph
        _call("graph_api_rel_upsert", account_id, body={
            "from_entity": "Sales", "to_entity": "Customer", "from_column": "CUS_DMS_KEY",
            "to_column": "CUS_DMS_KEY", "join_type": "OUTER"})
        assert [j["join_type"] for j in _joins(account_id, "Sales", "Customer")] == ["LEFT"]

    def test_a_stored_outer_join_becomes_left(self, graph):
        import store

        account_id, _kb = graph
        store.save_relationship(account_id, "Sales", "Customer", "CUS_DMS_KEY", "CUS_DMS_KEY",
                                join_type="OUTER")
        store.init_db()  # the startup migrations
        assert [j["join_type"] for j in _joins(account_id, "Sales", "Customer")] == ["LEFT"]


class TestDeletingAJoin:

    def _confirmed_join(self, account_id):
        _call("graph_api_rel_upsert", account_id, body={
            "from_entity": "Sales", "to_entity": "Customer", "from_column": "CUS_DMS_KEY",
            "to_column": "CUS_DMS_KEY", "join_type": "LEFT"})
        return _joins(account_id, "Sales", "Customer")[0]["id"]

    @pytest.mark.parametrize("route", ["graph_api_rel_delete", "graph_rel_delete"])
    def test_a_snapshot_is_kept_and_restores_it(self, graph, route):
        import store

        account_id, kb_dir = graph
        rel_id = self._confirmed_join(account_id)
        assert _model_joins(kb_dir, "MART.SLS_FCT", "MART.CUS_DMS")
        _call(route, account_id, rel_id)
        assert _joins(account_id, "Sales", "Customer") == []
        assert _model_joins(kb_dir, "MART.SLS_FCT", "MART.CUS_DMS") == []
        version = store.list_graph_versions(account_id)[0]
        assert "Sales -> Customer" in version["label"]
        _call("graph_versions_restore", account_id, version["id"])
        assert [j["from_column"] for j in _joins(account_id, "Sales", "Customer")] == ["CUS_DMS_KEY"]

    def test_a_bulk_delete_keeps_one_snapshot(self, graph):
        import store

        account_id, kb_dir = graph
        rel_id = self._confirmed_join(account_id)
        before = len(store.list_graph_versions(account_id))
        _call("graph_api_rel_bulk", account_id, body={"deletes": [rel_id], "updates": []})
        assert len(store.list_graph_versions(account_id)) == before + 1
        assert _model_joins(kb_dir, "MART.SLS_FCT", "MART.CUS_DMS") == []
