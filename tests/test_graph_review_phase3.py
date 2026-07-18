"""Entity Graph Phase 3 — review-first UX.

Backend: reject/property endpoint (completing the confirm/reject symmetry)
and the pre-existing bulk-accept threshold behavior it sits beside.
Frontend: static wiring assertions for the review queue panel, evidence
chips, Test-resolver modal, and auto-layout — matching this codebase's
established pattern for canvas JS that isn't practically unit-testable.
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
    """Minimal stand-in for a FastAPI Request whose .json() the route awaits."""

    def __init__(self, payload: dict):
        self._payload = payload

    async def json(self):
        return self._payload


class GraphReviewEndpointTests(unittest.TestCase):
    def setUp(self):
        import store
        store.init_db()
        self.store = store
        self.account_id = f"acct-graph-p3-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

    def tearDown(self):
        with self.store.get_db() as conn:
            for table in ("entity_properties", "entity_relationships", "entity_graph"):
                conn.execute(
                    f"DELETE FROM {table} WHERE account_id=?", (self.account_id,)
                )
            conn.execute("DELETE FROM client WHERE account_id=?", (self.account_id,))

    def test_reject_property_marks_status_rejected(self):
        import admin.routes as routes

        self.store.save_entity_property(
            self.account_id, "FACT_SALES", "NET_AMT",
            role="metric", confidence_score=70, status="suggested",
            generated_by="llm", reason="Numeric column named like a measure",
        )
        req = _JsonRequest({"entity_name": "FACT_SALES", "column_name": "NET_AMT"})
        with patch.object(routes, "_is_auth", return_value=True):
            resp = _arun(routes.graph_reject_property(req, self.account_id))
        self.assertEqual(resp.status_code, 200)
        props = self.store.list_all_entity_properties(self.account_id)
        row = next(p for p in props if p["column_name"] == "NET_AMT")
        self.assertEqual(row["status"], "rejected")
        # Provenance survives the rejection — the auditor can still see why
        # it was suggested in the first place.
        self.assertEqual(row["generated_by"], "llm")

    def test_bulk_accept_respects_confidence_threshold(self):
        import admin.routes as routes

        self.store.save_entity(
            self.account_id, "DIM_HIGH", "DIM_HIGH",
            status="suggested", confidence_score=92, generated_by="heuristic",
        )
        self.store.save_entity(
            self.account_id, "DIM_LOW", "DIM_LOW",
            status="suggested", confidence_score=40, generated_by="llm",
        )
        self.store.save_relationship(
            self.account_id, "DIM_HIGH", "DIM_LOW", "K", "K",
            status="suggested", confidence_score=90, generated_by="llm",
        )
        req = _JsonRequest({"min_confidence": 85})
        with patch.object(routes, "_is_auth", return_value=True):
            resp = _arun(routes.graph_bulk_accept(req, self.account_id))
        body = json.loads(resp.body)
        self.assertEqual(body["entities_accepted"], 1)
        self.assertEqual(body["relationships_accepted"], 1)
        entities = {
            e["entity_name"]: e for e in self.store.list_entities(self.account_id)
        }
        self.assertEqual(entities["DIM_HIGH"]["status"], "confirmed")
        self.assertEqual(entities["DIM_LOW"]["status"], "suggested")  # below bar


class GraphReviewUiWiringTests(unittest.TestCase):
    SRC = (ROOT / "admin/templates/client_graph.html").read_text(encoding="utf-8")

    def test_toolbar_has_review_test_and_layout_buttons(self):
        for needle in (
            'id="review-btn"', "toggleReviewPanel()",
            "openResolverModal()", "autoLayout()",
        ):
            self.assertIn(needle, self.SRC, needle)

    def test_review_panel_has_progress_evidence_and_bulk_accept(self):
        for needle in (
            'id="review-panel"', 'id="rv-progress-fill"',
            "bulkAcceptHighConf()", "min_confidence: 85",
            "_srcChip", "generated_by", "rv-reason",
        ):
            self.assertIn(needle, self.SRC, needle)

    def test_items_sorted_joins_first(self):
        # A wrong join corrupts every crossing answer — rels outrank entities.
        fn = self.SRC[self.SRC.index("function _pendingItems"):]
        fn = fn[:fn.index("function updateReviewBadge")]
        self.assertLess(fn.index("rels.sort"), fn.index("ents.sort"))

    def test_decisions_hit_confirm_and_reject_endpoints(self):
        for needle in (
            "confirm' : 'reject'}/relationship/",
            "confirm' : 'reject'}/entity/",
            "confirm' : 'reject'}/property",
        ):
            self.assertIn(needle.replace("confirm' : 'reject'", "confirm' : 'reject'"), self.SRC, needle)

    def test_resolver_modal_surfaces_scope_and_skeleton(self):
        for needle in (
            'id="resolver-modal"', "runResolverTest",
            "suggested_fallback", "join_skeleton",
        ):
            self.assertIn(needle, self.SRC, needle)

    def test_auto_layout_persists_positions(self):
        fn = self.SRC[self.SRC.index("function autoLayout"):]
        fn = fn[:fn.index("// ── Keyboard shortcuts")]
        self.assertIn("saveEntityPosition", fn)

    def test_boot_opens_review_panel_when_pending(self):
        boot = self.SRC[self.SRC.index("// ── Boot"):]
        self.assertIn("updateReviewBadge()", boot)
        self.assertIn("_pendingItems().length", boot)

    def test_reject_property_route_exists(self):
        routes_src = (ROOT / "admin/routes.py").read_text(encoding="utf-8")
        self.assertIn(
            '@router.post("/clients/{account_id}/graph/api/reject/property")',
            routes_src,
        )


if __name__ == "__main__":
    unittest.main()
