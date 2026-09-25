# -*- coding: utf-8 -*-
"""The admin accepts joins on what the warehouse's rows say, not on names.

The graph's review list offered one bulk action for joins: "Accept all >=
85%", where the percentage is how alike the column names are. Whether the
keys actually find their rows was recorded by the profiler, shown nowhere in
the list, and checked by nothing an admin could press -- the validate-all
route had no button, and would have fired every probe at once.

Now:
  * each suggested join in the list says what the data said about it;
  * "Check joins against the data" profiles the ones not yet checked, one at
    a time on a bounded slice (core.relationship_validator);
  * "Accept joins the data confirms" confirms exactly the joins whose rows
    vouch for them -- the same rule the list's badge shows, computed once on
    the server;
  * the KB build profiles the joins its graph sync adds, as discovery does.

Routes are called directly, with only the admin session stubbed. Synthetic
tables; no customer data.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from unittest.mock import patch

import pytest

from core.relationship_validator import (
    STATUS_VALID,
    RelationshipValidationResult,
    data_confirms,
)
from tests.js_lift import function as lift

SOUND = {"validation_status": "valid", "join_multiplicity": "one_to_one_or_many_to_one",
         "match_rate": 99.6, "fanout_ratio": 1.0}


class TestWhatTheDataConfirms:

    def test_a_join_whose_rows_all_find_one_row_is_confirmed(self):
        assert data_confirms(SOUND)

    @pytest.mark.parametrize("change", [
        {"validation_status": "warning"},
        {"validation_status": "untested"},
        {"match_rate": 98.9},
        {"fanout_ratio": 1.2},
        {"join_multiplicity": "one_to_many_or_many_to_many"},
        {"match_rate": None},
        {"fanout_ratio": -1.0},
    ], ids=["warning", "untested", "orphans", "fans out", "duplicates", "no rate", "no fanout"])
    def test_anything_less_is_not(self, change):
        assert not data_confirms({**SOUND, **change})


class _JsonRequest:
    def __init__(self, payload: dict):
        self._payload = payload

    async def json(self):
        return self._payload


@pytest.fixture
def workspace():
    import store

    store.init_db()
    account = f"acct-joins-{uuid.uuid4().hex[:8]}"
    store.upsert_client(account, "Test Ltd")
    for name in ("ITM_BAL_PRD_FCT", "ITM_DMS", "WHS_DMS", "PRD_DMS"):
        store.save_entity(account, name, name, status="confirmed")
    store.save_entity(account, "UOM_DMS", "UOM_DMS", status="suggested", confidence_score=95)
    ids = {}
    for to_entity, key in (("ITM_DMS", "ITM_DMS_KEY"), ("WHS_DMS", "WHS_DMS_KEY"), ("PRD_DMS", "PRD_DMS_KEY")):
        store.save_relationship(account, "ITM_BAL_PRD_FCT", to_entity, key, key,
                                status="suggested", confidence_score=60, generated_by="heuristic")
    for rel in store.list_relationships(account, active_only=False):
        ids[rel["to_entity"]] = int(rel["id"])
    # Items: every key finds its row. Warehouses: a tenth find none. Periods:
    # never checked.
    store.update_relationship_validation(account, ids["ITM_DMS"], "valid", join_multiplicity=SOUND["join_multiplicity"],
                                         match_rate=100.0, orphan_rate=0.0, null_fk_rate=0.0, fanout_ratio=1.0)
    store.update_relationship_validation(account, ids["WHS_DMS"], "warning", join_multiplicity=SOUND["join_multiplicity"],
                                         match_rate=90.0, orphan_rate=10.0, null_fk_rate=0.0, fanout_ratio=1.0)
    return account, ids


def _statuses(account):
    import store

    return {rel["to_entity"]: rel["status"] for rel in store.list_relationships(account, active_only=False)}


class TestTheRoutes:

    def test_accept_joins_the_data_confirms_accepts_exactly_those(self, workspace):
        import admin.routes as routes

        account, _ = workspace
        request = _JsonRequest({"action": "accept", "kinds": ["rel", "entity"], "evidence": "data"})
        with patch.object(routes, "_is_auth", return_value=True), patch.object(routes, "_after_semantic_approval"):
            body = json.loads(asyncio.run(routes.graph_bulk_review(request, account)).body)
        assert body["relationships"] == 1 and body["entities"] == 0
        assert _statuses(account) == {"ITM_DMS": "confirmed", "WHS_DMS": "suggested", "PRD_DMS": "suggested"}

    def test_the_datas_word_does_not_reject(self, workspace):
        import admin.routes as routes

        account, _ = workspace
        request = _JsonRequest({"action": "reject", "kinds": ["rel"], "evidence": "data"})
        with patch.object(routes, "_is_auth", return_value=True), pytest.raises(routes.HTTPException):
            asyncio.run(routes.graph_bulk_review(request, account))
        assert set(_statuses(account).values()) == {"suggested"}

    def test_the_graph_says_which_joins_the_data_confirms(self, workspace):
        import admin.routes as routes

        account, _ = workspace
        with patch.object(routes, "_is_auth", return_value=True):
            graph = json.loads(asyncio.run(routes.graph_json_api(None, account)).body)
        assert {rel["to_entity"]: rel["data_confirmed"] for rel in graph["relationships"]} == {
            "ITM_DMS": True, "WHS_DMS": False, "PRD_DMS": False}

    def test_check_joins_against_the_data_profiles_the_unchecked_ones(self, workspace):
        import admin.routes as routes
        import core.relationship_validator as rv

        account, ids = workspace
        asked = []

        def probe(account_id, rel_id, **kwargs):
            asked.append(rel_id)
            return RelationshipValidationResult(
                rel_id, STATUS_VALID, "ok", checked_by="db",
                join_multiplicity=SOUND["join_multiplicity"], match_rate=100.0,
                orphan_rate=0.0, null_fk_rate=0.0, fanout_ratio=1.0)

        with patch.object(routes, "_is_auth", return_value=True), patch.object(rv, "validate_relationship", probe):
            body = json.loads(asyncio.run(routes.graph_api_rel_profile(None, account)).body)
        assert asked == [ids["PRD_DMS"]]
        assert body["valid"] == 1
        with patch.object(routes, "_is_auth", return_value=True):
            graph = json.loads(asyncio.run(routes.graph_json_api(None, account)).body)
        assert {rel["to_entity"]: rel["data_confirmed"] for rel in graph["relationships"]}["PRD_DMS"] is True

    def test_the_kb_build_checks_the_joins_its_graph_sync_adds(self):
        """Discovery profiled its joins; the build's graph sync added joins
        and left them "untested"."""
        import os
        import tempfile

        import admin.routes as routes
        import core.relationship_validator as rv
        import store

        store.init_db()
        account = f"acct-build-{uuid.uuid4().hex[:8]}"
        store.upsert_client(account, "Test Ltd")
        columns = {"ITM_BAL_PRD_FCT": ("ITM_BAL_PRD_FCT_KEY", "ITM_DMS_KEY", "ON_HND_QTY"),
                   "ITM_DMS": ("ITM_DMS_KEY", "ITM_CD")}
        schema = {f"WH.MART.{table}": {
            "columns": [{"name": c, "type": "int", "nullable": False, "comment": ""} for c in cols],
            "pk_columns": [cols[0]], "row_count": 10, "comment": "", "schema": "MART", "database": "WH",
        } for table, cols in columns.items()}
        schema_dir = tempfile.mkdtemp()
        with open(os.path.join(schema_dir, "_schema.json"), "w", encoding="utf-8") as handle:
            json.dump(schema, handle)
        asked = []

        def probe(account_id, rel_id, **kwargs):
            asked.append(rel_id)
            return RelationshipValidationResult(
                rel_id, STATUS_VALID, "ok", checked_by="db",
                join_multiplicity=SOUND["join_multiplicity"], match_rate=100.0,
                orphan_rate=0.0, null_fk_rate=0.0, fanout_ratio=1.0)

        with patch.object(rv, "validate_relationship", probe):
            summary = asyncio.run(routes._sync_graph_after_build(account, schema_dir, tempfile.mkdtemp()))
        rels = store.list_relationships(account, active_only=True)
        assert summary["relationships_added"] >= 1 and rels
        assert sorted(asked) == sorted(int(rel["id"]) for rel in rels)
        assert {rel["validation_status"] for rel in rels} == {"valid"}
        assert summary["joins_profiled"]["valid"] == len(rels)

    def test_the_kb_builds_profiling_never_fails_the_build(self, workspace):
        import admin.routes as routes
        import core.relationship_validator as rv

        account, _ = workspace

        def unreachable(*args, **kwargs):
            raise RuntimeError("warehouse down")

        with patch.object(rv, "profile_suggested_relationships", unreachable):
            assert asyncio.run(routes._profile_suggested_joins(account, "after the KB build")) == {}


def _evidence(rel: dict) -> str:
    import dukpy

    from pathlib import Path

    page = (Path(__file__).resolve().parents[1] / "admin" / "templates" / "client_graph.html").read_text(
        encoding="utf-8")
    script = f"""
{lift(page, "function _esc(")}
{lift(page, "function _joinEvidenceHtml(r)")}
_joinEvidenceHtml({json.dumps(rel)});
"""
    return dukpy.evaljs(script)


class TestTheReviewListSaysWhatTheDataSaid:

    def test_a_join_never_checked_says_so(self):
        assert "not checked yet" in _evidence({"validation_status": "untested", "match_rate": -1})

    def test_a_confirmed_join_says_so(self):
        text = _evidence({**SOUND, "match_rate": 100.0, "orphan_rate": 0.0, "null_fk_rate": 0.0,
                          "data_confirmed": True})
        assert "ev-ok" in text and "Data confirms: 100.0% of keys find their row" in text

    def test_a_join_with_orphans_and_fan_out_says_both(self):
        text = _evidence({"validation_status": "warning", "match_rate": 90.0, "orphan_rate": 10.0,
                          "null_fk_rate": 0.0, "fanout_ratio": 1.5, "data_confirmed": False})
        assert "ev-warn" in text and "10.0% find none" in text and "fans out 1.50" in text

    def test_a_broken_join_says_so(self):
        assert "does not work" in _evidence({"validation_status": "broken"})
