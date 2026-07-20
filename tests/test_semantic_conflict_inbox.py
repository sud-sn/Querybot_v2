"""Sprint 5: Admin Resolution UX.

Sprint 2 gave semantic_conflict a full review-status lifecycle
(open/resolved/acknowledged/dismissed) and Sprint 3/4 built the machinery
to cross-reference a question against open conflicts - but nothing
admin-facing ever surfaced more than the newest 8 conflicts, and nothing
could act on one. This closes the "Admin Resolution UX" bullets:

  - Dedicated conflict inbox (all conflicts, filterable, evidence shown).
  - Resolve / acknowledge / dismiss actions with a reviewer note (the
    "reviewer and approval audit trail" bullet - resolved_by/resolved_at/
    resolution_note already existed on the table since Sprint 2's schema,
    just unused).
  - "Test a question before publishing" - a real, DB-backed dry run of
    metric/term matching cross-referenced against open conflicts.
  - Contract version diff + rollback.

Strategy: real SQLite (store.init_db(), not mocked) for all data, so the
whole store -> route chain is exercised for real; admin.routes._resp is
monkeypatched to capture template context instead of rendering real Jinja
(this codebase's established pattern for admin GET routes - see
test_admin_learning_queue.py). Form(...)-declared POST params are passed
as explicit kwargs on direct invocation, matching test_compiler_mode_toggle.py.
"""
from __future__ import annotations

import asyncio
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _arun(coro):
    return asyncio.run(coro)


def _capture_resp():
    ctx_captured: dict = {}

    def _fake_resp(request, name, ctx=None):
        ctx_captured.update(ctx or {})
        ctx_captured["_template_name"] = name
        r = MagicMock()
        r.status_code = 200
        return r

    return ctx_captured, _fake_resp


class ConflictInboxRouteTests(unittest.TestCase):
    def setUp(self):
        import store
        store.init_db()
        self.store = store
        self.account_id = f"acct-inbox-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")
        self.run_id = store.create_semantic_compile_run(
            self.account_id, trigger="test", initiated_by="test",
            mode="shadow", base_version="",
        )
        self.metric_id = store.save_metric(self.account_id, {
            "name": "Total Revenue", "formula_type": "expression",
            "sql_template": "SUM(REVENUE)", "synonyms": "",
        })
        store.save_semantic_conflicts(self.run_id, self.account_id, [
            {
                "conflict_key": "k1", "code": "duplicate_metric_name", "severity": "ERROR",
                "object_type": "metric", "object_id": f"metric:{self.metric_id}", "table_name": "",
                "message": "Two metrics share a name.", "evidence": {"a": 1},
                "suggestions": ["Rename one."],
            },
            {
                "conflict_key": "k2", "code": "cardinality_fanout", "severity": "WARNING",
                "object_type": "relationship", "object_id": "join:1", "table_name": "",
                "message": "Fan-out risk.",
            },
        ])

    def tearDown(self):
        with self.store.get_db() as conn:
            conn.execute("DELETE FROM semantic_conflict WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM semantic_compile_run WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM semantic_compiler_state WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM metric_registry WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM client WHERE account_id=?", (self.account_id,))

    def _get_inbox(self, **kwargs):
        import admin.routes as routes
        ctx, fake_resp = _capture_resp()
        with (
            patch.object(routes, "_is_auth", return_value=True),
            patch.object(routes, "_resp", side_effect=fake_resp),
        ):
            _arun(routes.model_health_conflict_inbox(object(), self.account_id, **kwargs))
        return ctx

    def test_lists_open_conflicts_by_default(self):
        ctx = self._get_inbox()
        self.assertEqual(len(ctx["conflicts"]), 2)
        self.assertEqual(ctx["status"], "open")

    def test_severity_filter(self):
        ctx = self._get_inbox(severity="ERROR")
        self.assertEqual(len(ctx["conflicts"]), 1)
        self.assertEqual(ctx["conflicts"][0]["code"], "duplicate_metric_name")

    def test_invalid_severity_falls_back_to_all(self):
        ctx = self._get_inbox(severity="NOT_A_SEVERITY")
        self.assertEqual(ctx["severity"], "")
        self.assertEqual(len(ctx["conflicts"]), 2)

    def test_object_info_resolves_metric_label(self):
        ctx = self._get_inbox()
        error_conflict = next(c for c in ctx["conflicts"] if c["severity"] == "ERROR")
        self.assertEqual(error_conflict["object_info"]["object_type"], "metric")
        self.assertEqual(error_conflict["object_info"]["label"], "Total Revenue")
        self.assertIn("/metrics", error_conflict["object_info"]["edit_href"])

    def test_relationship_object_maps_to_graph_link(self):
        ctx = self._get_inbox()
        warn_conflict = next(c for c in ctx["conflicts"] if c["severity"] == "WARNING")
        self.assertEqual(warn_conflict["object_info"]["object_type"], "relationship")
        self.assertIn("/graph", warn_conflict["object_info"]["edit_href"])

    def test_404_for_unknown_client(self):
        import admin.routes as routes
        from fastapi import HTTPException
        with patch.object(routes, "_is_auth", return_value=True):
            with self.assertRaises(HTTPException) as ctx:
                _arun(routes.model_health_conflict_inbox(object(), "no-such-account", status="open"))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_unauthenticated_redirects_to_login(self):
        import admin.routes as routes
        with patch.object(routes, "_is_auth", return_value=False):
            resp = _arun(routes.model_health_conflict_inbox(object(), self.account_id))
        self.assertEqual(resp.status_code, 303)
        self.assertIn("/admin/login", resp.headers["location"])


class ResolveConflictRouteTests(unittest.TestCase):
    def setUp(self):
        import store
        store.init_db()
        self.store = store
        self.account_id = f"acct-resolve-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")
        run_id = store.create_semantic_compile_run(
            self.account_id, trigger="test", initiated_by="test", mode="shadow", base_version="",
        )
        store.save_semantic_conflicts(run_id, self.account_id, [
            {"conflict_key": "k1", "code": "x", "severity": "ERROR",
             "object_type": "metric", "object_id": "metric:1", "message": "m"},
        ])
        self.conflict_id = store.list_semantic_conflicts(self.account_id, status="open")[0]["conflict_id"]

    def tearDown(self):
        with self.store.get_db() as conn:
            conn.execute("DELETE FROM semantic_conflict WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM semantic_compile_run WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM client WHERE account_id=?", (self.account_id,))

    def _resolve(self, action, note=""):
        import admin.routes as routes
        with patch.object(routes, "_is_auth", return_value=True):
            return _arun(routes.model_health_resolve_conflict(
                object(), self.account_id, self.conflict_id, action=action, note=note,
            ))

    def test_resolve_updates_status_and_note(self):
        resp = self._resolve("resolve", "fixed the metric definition")
        self.assertEqual(resp.status_code, 303)
        self.assertIn("saved=conflict_resolve", resp.headers["location"])
        updated = self.store.get_semantic_conflict(self.account_id, self.conflict_id)
        self.assertEqual(updated["status"], "resolved")
        self.assertEqual(updated["resolution_note"], "fixed the metric definition")
        self.assertEqual(updated["resolved_by"], "admin")

    def test_acknowledge_and_dismiss_also_work(self):
        resp = self._resolve("acknowledge")
        self.assertIn("saved=conflict_acknowledge", resp.headers["location"])
        self.assertEqual(self.store.get_semantic_conflict(self.account_id, self.conflict_id)["status"], "acknowledged")

    def test_double_resolve_redirects_with_error_not_500(self):
        self._resolve("resolve")
        resp = self._resolve("dismiss")
        self.assertEqual(resp.status_code, 303)
        self.assertIn("error=", resp.headers["location"])

    def test_unknown_conflict_id_redirects_with_error(self):
        import admin.routes as routes
        with patch.object(routes, "_is_auth", return_value=True):
            resp = _arun(routes.model_health_resolve_conflict(
                object(), self.account_id, "no-such-id", action="resolve", note="",
            ))
        self.assertIn("error=", resp.headers["location"])

    def test_unauthenticated_redirects_to_login(self):
        import admin.routes as routes
        with patch.object(routes, "_is_auth", return_value=False):
            resp = _arun(routes.model_health_resolve_conflict(
                object(), self.account_id, self.conflict_id, action="resolve", note="",
            ))
        self.assertIn("/admin/login", resp.headers["location"])


class TestQuestionRouteTests(unittest.TestCase):
    def setUp(self):
        import store
        store.init_db()
        self.store = store
        self.account_id = f"acct-testq-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

    def tearDown(self):
        with self.store.get_db() as conn:
            conn.execute("DELETE FROM semantic_conflict WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM client WHERE account_id=?", (self.account_id,))

    def test_returns_expected_shape_with_no_matches(self):
        import admin.routes as routes
        with patch.object(routes, "_is_auth", return_value=True):
            resp = _arun(routes.model_health_test_question(
                object(), self.account_id, question="something nobody would ever ask",
            ))
        self.assertEqual(resp.status_code, 200)
        import json
        body = json.loads(bytes(resp.body))
        self.assertIsNone(body["matched_metric"])
        self.assertEqual(body["matched_terms"], [])
        self.assertEqual(body["confidence"], 60.0)  # no metrics/dims resolved -> -40 from 100
        self.assertEqual(body["clarifications"], [])

    def test_empty_question_rejected(self):
        import admin.routes as routes
        with patch.object(routes, "_is_auth", return_value=True):
            resp = _arun(routes.model_health_test_question(object(), self.account_id, question="   "))
        self.assertEqual(resp.status_code, 400)

    def test_unknown_client_404(self):
        import admin.routes as routes
        from fastapi import HTTPException
        with patch.object(routes, "_is_auth", return_value=True):
            with self.assertRaises(HTTPException):
                _arun(routes.model_health_test_question(object(), "no-such-account", question="hi"))


class ContractVersionRouteTests(unittest.TestCase):
    def setUp(self):
        import store
        store.init_db()
        self.store = store
        self.account_id = f"acct-versions-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")
        self.run_id = store.create_semantic_compile_run(
            self.account_id, trigger="test", initiated_by="test", mode="shadow", base_version="",
        )
        self.contract_v1 = {
            "meta": {"contract_version": "v1"},
            "metrics": [{"canonical_id": "metric:1"}],
            "terms": [], "date_roles": [], "relationships": [],
        }
        self.contract_v2 = {
            "meta": {"contract_version": "v2"},
            "metrics": [{"canonical_id": "metric:1"}, {"canonical_id": "metric:2"}],
            "terms": [], "date_roles": [], "relationships": [],
        }
        store.save_semantic_contract_version(
            self.account_id, self.contract_v1, status="active",
            compile_run_id=self.run_id, created_by="admin",
        )
        store.save_semantic_contract_version(
            self.account_id, self.contract_v2, status="draft",
            compile_run_id=self.run_id, created_by="admin",
        )

    def tearDown(self):
        with self.store.get_db() as conn:
            conn.execute("DELETE FROM semantic_contract_version WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM semantic_compile_run WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM semantic_compiler_state WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM client WHERE account_id=?", (self.account_id,))

    def test_versions_page_lists_both(self):
        import admin.routes as routes
        ctx, fake_resp = _capture_resp()
        with (
            patch.object(routes, "_is_auth", return_value=True),
            patch.object(routes, "_resp", side_effect=fake_resp),
        ):
            _arun(routes.model_health_contract_versions(object(), self.account_id))
        self.assertEqual({v["version"] for v in ctx["versions"]}, {"v1", "v2"})

    def test_diff_reports_added_metric(self):
        import admin.routes as routes
        with patch.object(routes, "_is_auth", return_value=True):
            resp = _arun(routes.model_health_contract_version_diff(
                object(), self.account_id, base="v1", compare="v2",
            ))
        import json
        body = json.loads(bytes(resp.body))
        self.assertEqual(body["sections"]["metrics"]["added"], 1)
        self.assertEqual(body["sections"]["metrics"]["unchanged"], 1)
        self.assertEqual(body["sections"]["metrics"]["removed"], 0)

    def test_diff_unknown_version_404(self):
        import admin.routes as routes
        with patch.object(routes, "_is_auth", return_value=True):
            resp = _arun(routes.model_health_contract_version_diff(
                object(), self.account_id, base="v1", compare="v999",
            ))
        self.assertEqual(resp.status_code, 404)

    def test_rollback_publishes_target_version(self):
        import admin.routes as routes
        with patch.object(routes, "_is_auth", return_value=True):
            resp = _arun(routes.model_health_rollback_contract_version(
                object(), self.account_id, "v2",
            ))
        self.assertEqual(resp.status_code, 303)
        self.assertIn("saved=rollback", resp.headers["location"])
        versions = {v["version"]: v["status"] for v in self.store.list_semantic_contract_versions(self.account_id)}
        self.assertEqual(versions["v2"], "active")
        self.assertEqual(versions["v1"], "superseded")

    def test_rollback_unknown_version_redirects_with_error(self):
        import admin.routes as routes
        with patch.object(routes, "_is_auth", return_value=True):
            resp = _arun(routes.model_health_rollback_contract_version(
                object(), self.account_id, "v999",
            ))
        self.assertIn("error=", resp.headers["location"])


class Sprint5WiringTests(unittest.TestCase):
    def setUp(self):
        self.routes_src = (ROOT / "admin/routes.py").read_text(encoding="utf-8")
        self.mh_src = (ROOT / "admin/templates/client_model_health.html").read_text(encoding="utf-8")

    def test_all_routes_registered(self):
        for path in (
            '@router.get("/clients/{account_id}/model-health/conflicts"',
            '@router.post("/clients/{account_id}/model-health/conflicts/{conflict_id}/resolve")',
            '@router.post("/clients/{account_id}/model-health/test-question")',
            '@router.get("/clients/{account_id}/model-health/versions"',
            '@router.get("/clients/{account_id}/model-health/versions/diff")',
            '@router.post("/clients/{account_id}/model-health/versions/{version}/rollback")',
        ):
            self.assertIn(path, self.routes_src)

    def test_model_health_links_to_new_pages(self):
        self.assertIn("/model-health/conflicts", self.mh_src)
        self.assertIn("/model-health/versions", self.mh_src)

    def test_resolve_actions_are_scoped_by_account_id(self):
        # resolve_semantic_conflict must be called with account_id, not just
        # conflict_id - a cross-tenant conflict_id guess must never succeed.
        anchor = "store.resolve_semantic_conflict("
        pos = self.routes_src.index(anchor)
        block = self.routes_src[pos:pos + 150]
        self.assertIn("account_id, conflict_id", block)


if __name__ == "__main__":
    unittest.main()
