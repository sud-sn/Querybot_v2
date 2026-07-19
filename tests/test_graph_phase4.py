"""Entity Graph Phase 4 — setup-wizard review step, correction-flagged
joins, and graph version history / export / import.

Real-execution tests for the store layer (snapshot/restore round-trip,
correction flagging, pending counts) and route-level tests via the
established direct-call pattern; static wiring assertions for the canvas
JS and wizard step.
"""
from __future__ import annotations

import asyncio
import json
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _arun(coro):
    return asyncio.run(coro)


class _JsonRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class _GraphAccountTest(unittest.TestCase):
    """Shared per-test account with cleanup across all graph tables."""

    def setUp(self):
        import store
        store.init_db()
        self.store = store
        self.account_id = f"acct-graph-p4-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

    def tearDown(self):
        with self.store.get_db() as conn:
            for table in ("graph_version", "entity_properties",
                          "entity_relationships", "entity_graph"):
                conn.execute(
                    f"DELETE FROM {table} WHERE account_id=?", (self.account_id,)
                )
            conn.execute("DELETE FROM client WHERE account_id=?", (self.account_id,))

    def _seed(self):
        self.store.save_entity(
            self.account_id, "FACT_SALES", "RX.FACT_SALES",
            entity_type="fact", status="confirmed",
        )
        self.store.save_entity(
            self.account_id, "DIM_DRUG", "RX.DIM_DRUG", status="confirmed",
        )
        self.store.save_relationship(
            self.account_id, "FACT_SALES", "DIM_DRUG", "DRUG_KEY", "DRUG_KEY",
            status="confirmed",
        )
        self.store.save_entity_property(
            self.account_id, "FACT_SALES", "NET_AMT", role="metric",
        )


class GraphVersionTests(_GraphAccountTest):
    def test_snapshot_restore_round_trip(self):
        self._seed()
        vid = self.store.save_graph_version(self.account_id, label="baseline")
        self.assertGreater(vid, 0)

        # Mutate: drop an entity and its join, add a stray one.
        self.store.delete_entity(self.account_id, "DIM_DRUG")
        self.store.save_entity(self.account_id, "DIM_STRAY", "RX.DIM_STRAY")
        names_now = {e["entity_name"] for e in self.store.list_entities(self.account_id)}
        self.assertNotIn("DIM_DRUG", names_now)

        version = self.store.get_graph_version(self.account_id, vid)
        counts = self.store.replace_graph_from_snapshot(
            self.account_id, version["snapshot"]
        )
        self.assertEqual(counts["entity_graph"], 2)
        self.assertEqual(counts["entity_relationships"], 1)
        restored = {e["entity_name"] for e in self.store.list_entities(self.account_id)}
        self.assertEqual(restored, {"FACT_SALES", "DIM_DRUG"})
        rels = self.store.list_relationships(self.account_id)
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0]["from_entity"], "FACT_SALES")

    def test_import_forces_target_account_id(self):
        # An export from another tenant must not carry its account key in.
        snapshot = {
            "entities": [{
                "account_id": "someone-else", "entity_name": "DIM_X",
                "table_name": "DIM_X", "entity_type": "dimension",
                "is_active": 1, "status": "confirmed",
            }],
            "relationships": [], "properties": [],
        }
        self.store.replace_graph_from_snapshot(self.account_id, snapshot)
        rows = self.store.list_entities(self.account_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["account_id"], self.account_id)

    def test_unknown_snapshot_columns_are_dropped_not_fatal(self):
        snapshot = {
            "entities": [{
                "entity_name": "DIM_Y", "table_name": "DIM_Y",
                "from_a_future_migration": "whatever", "is_active": 1,
            }],
        }
        self.store.replace_graph_from_snapshot(self.account_id, snapshot)
        self.assertEqual(
            self.store.list_entities(self.account_id)[0]["entity_name"], "DIM_Y"
        )

    def test_restore_route_auto_snapshots_current_state_first(self):
        import admin.routes as routes

        self._seed()
        vid = self.store.save_graph_version(self.account_id, label="baseline")
        with patch.object(routes, "_is_auth", return_value=True):
            resp = _arun(
                routes.graph_versions_restore(_JsonRequest({}), self.account_id, vid)
            )
        self.assertEqual(resp.status_code, 200)
        labels = [v["label"] for v in self.store.list_graph_versions(self.account_id)]
        self.assertTrue(any("pre-restore" in l for l in labels), labels)

    def test_import_route_rejects_non_graph_payload(self):
        import admin.routes as routes
        from fastapi import HTTPException

        with patch.object(routes, "_is_auth", return_value=True):
            with self.assertRaises(HTTPException) as ctx:
                _arun(routes.graph_import(
                    _JsonRequest({"random": "junk"}), self.account_id
                ))
        self.assertEqual(ctx.exception.status_code, 400)


class CorrectionFlagTests(_GraphAccountTest):
    def test_flag_marks_confirmed_joins_touching_changed_tables(self):
        self._seed()
        n = self.store.flag_relationships_needing_review(
            self.account_id, {"DIM_DRUG"}
        )
        self.assertEqual(n, 1)
        rel = self.store.list_relationships(self.account_id)[0]
        self.assertEqual(rel["validation_status"], "needs_review")
        # Already-flagged joins aren't double-counted on a second correction.
        self.assertEqual(
            self.store.flag_relationships_needing_review(self.account_id, {"DIM_DRUG"}), 0
        )

    def test_flag_matches_qualified_table_names(self):
        self._seed()
        n = self.store.flag_relationships_needing_review(
            self.account_id, {"RX.DIM_DRUG"}   # qualified — bare name extracted
        )
        self.assertEqual(n, 1)

    def test_unrelated_tables_flag_nothing(self):
        self._seed()
        self.assertEqual(
            self.store.flag_relationships_needing_review(self.account_id, {"DIM_OTHER"}), 0
        )

    def test_clear_flag_and_pending_count(self):
        self._seed()
        base = self.store.count_pending_graph_reviews(self.account_id)
        self.store.flag_relationships_needing_review(self.account_id, {"DIM_DRUG"})
        self.assertEqual(self.store.count_pending_graph_reviews(self.account_id), base + 1)
        rel_id = self.store.list_relationships(self.account_id)[0]["id"]
        self.assertTrue(
            self.store.clear_relationship_review_flag(self.account_id, rel_id)
        )
        self.assertEqual(self.store.count_pending_graph_reviews(self.account_id), base)

    def test_count_includes_suggested_items(self):
        self._seed()
        self.store.save_entity(
            self.account_id, "DIM_SUGG", "DIM_SUGG",
            status="suggested", confidence_score=70,
        )
        self.store.save_entity_property(
            self.account_id, "FACT_SALES", "MYSTERY_COL", status="suggested",
        )
        self.assertEqual(self.store.count_pending_graph_reviews(self.account_id), 2)
        self.assertEqual(self.store.count_pending_structural_graph_reviews(self.account_id), 1)


class Phase4WiringTests(unittest.TestCase):
    ROUTES = (ROOT / "admin/routes.py").read_text(encoding="utf-8")
    GRAPH = (ROOT / "admin/templates/client_graph.html").read_text(encoding="utf-8")
    SETUP = (ROOT / "admin/templates/client_setup.html").read_text(encoding="utf-8")

    def test_setup_wizard_has_semantic_review_step(self):
        self.assertIn("Semantic Review", self.SETUP)
        self.assertIn("graph_pending_reviews", self.SETUP)
        fn = self.ROUTES[self.ROUTES.index("async def client_setup_page("):]
        fn = fn[:fn.index("\n@router.")]
        self.assertIn("count_pending_structural_graph_reviews", fn)

    def test_correction_route_diffs_tables_and_flags_joins(self):
        fn = self.ROUTES[self.ROUTES.index("async def admin_learning_queue_correct_sql("):]
        fn = fn[:fn.index("\n@router.")]
        self.assertIn("analyze_sql", fn)
        self.assertIn("_orig_tables ^ _corr_tables", fn)
        self.assertIn("flag_relationships_needing_review", fn)
        # Best-effort: flagging failure must never break the correction save.
        self.assertIn("join flagging skipped (non-fatal)", fn)

    def test_review_queue_surfaces_flagged_joins(self):
        self.assertIn("validation_status === 'needs_review'", self.GRAPH)
        self.assertIn("correction flagged", self.GRAPH)
        self.assertIn("clear-review-flag", self.GRAPH)
        self.assertIn("Keep join", self.GRAPH)

    def test_tools_menu_and_version_endpoints_exist(self):
        for needle in ("Version history…", "Export JSON", "Import JSON…",
                       "graph-import-file", 'id="versions-modal"'):
            self.assertIn(needle, self.GRAPH, needle)
        for route in (
            '"/clients/{account_id}/graph/api/versions"',
            '"/clients/{account_id}/graph/api/versions/{version_id}/restore"',
            '"/clients/{account_id}/graph/api/export"',
            '"/clients/{account_id}/graph/api/import"',
            '"/clients/{account_id}/graph/api/relationships/{rel_id}/clear-review-flag"',
        ):
            self.assertIn(route, self.ROUTES, route)

    def test_import_auto_snapshots_before_replacing(self):
        fn = self.ROUTES[self.ROUTES.index("async def graph_import("):]
        fn = fn[:fn.index("\n@router.")]
        snap_pos = fn.index("save_graph_version")
        replace_pos = fn.index("replace_graph_from_snapshot")
        self.assertLess(snap_pos, replace_pos)


if __name__ == "__main__":
    unittest.main()
