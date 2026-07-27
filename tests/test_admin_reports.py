"""
Admin report-builder UI routes (admin/routes.py: reports_page, report_create,
report_update, report_delete, report_metric_add, report_metric_remove).

Real SQLite (store.init_db(), not mocked) so the whole store -> route chain
is exercised for real; admin.routes._resp is monkeypatched to capture
template context instead of rendering real Jinja (this codebase's
established pattern for admin GET routes — see test_semantic_conflict_inbox.py
and test_admin_learning_queue.py). Form(...)-declared POST params are passed
as explicit kwargs on direct invocation.
"""

import asyncio
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import store


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


class AdminReportsRouteTests(unittest.TestCase):
    def setUp(self):
        store.init_db()
        self.account_id = f"acct-admrpt-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")
        self.metric_id = store.save_metric(self.account_id, {
            "name": "Revenue", "synonyms": "", "sql_template": "SELECT SUM(1) AS x",
            "base_table": "SALES.REVENUE",
        })

    def tearDown(self):
        with store.get_db() as conn:
            conn.execute("DELETE FROM report_metric WHERE report_id IN "
                         "(SELECT id FROM report WHERE account_id=?)", (self.account_id,))
            conn.execute("DELETE FROM report WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM metric_registry WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM client WHERE account_id=?", (self.account_id,))

    def _make_get_request(self):
        req = MagicMock()
        req.query_params = {}
        return req

    def _get_reports_page(self):
        import admin.routes as routes
        ctx, fake_resp = _capture_resp()
        with (
            patch.object(routes, "_is_auth", return_value=True),
            patch.object(routes, "_resp", side_effect=fake_resp),
        ):
            _arun(routes.reports_page(self._make_get_request(), self.account_id))
        return ctx

    def test_unauthenticated_redirects_to_login(self):
        import admin.routes as routes
        with patch.object(routes, "_is_auth", return_value=False):
            result = _arun(routes.reports_page(object(), self.account_id))
        self.assertEqual(result.status_code, 303)
        self.assertIn("/admin/login", result.headers["location"])

    def test_unknown_client_redirects_to_clients_list(self):
        import admin.routes as routes
        with patch.object(routes, "_is_auth", return_value=True):
            result = _arun(routes.reports_page(object(), "nonexistent-account"))
        self.assertEqual(result.status_code, 303)
        self.assertIn("/admin/clients", result.headers["location"])

    def test_empty_reports_page_context(self):
        ctx = self._get_reports_page()
        self.assertEqual(ctx["reports"], [])
        self.assertEqual(len(ctx["all_metrics"]), 1)
        self.assertEqual(ctx["all_metrics"][0]["name"], "Revenue")

    def test_create_report_appears_on_page(self):
        import admin.routes as routes
        with patch.object(routes, "_is_auth", return_value=True):
            result = _arun(routes.report_create(
                MagicMock(), self.account_id, name="Daily Ops", description="desc", is_default="1",
            ))
        self.assertEqual(result.status_code, 303)
        self.assertIn("saved=1", result.headers["location"])

        ctx = self._get_reports_page()
        self.assertEqual(len(ctx["reports"]), 1)
        self.assertEqual(ctx["reports"][0]["name"], "Daily Ops")
        self.assertEqual(ctx["reports"][0]["is_default"], 1)
        self.assertEqual(ctx["reports"][0]["metrics"], [])

    def test_create_report_blank_name_errors(self):
        import admin.routes as routes
        with patch.object(routes, "_is_auth", return_value=True):
            result = _arun(routes.report_create(
                MagicMock(), self.account_id, name="   ", description="", is_default="0",
            ))
        self.assertIn("error=", result.headers["location"])

    def test_update_report(self):
        import admin.routes as routes
        from store import report_store

        r = report_store.create_report(self.account_id, "Old Name")
        with patch.object(routes, "_is_auth", return_value=True):
            _arun(routes.report_update(
                MagicMock(), self.account_id, r["id"],
                name="New Name", description="updated", is_default="1", is_active="1",
            ))
        fetched = report_store.get_report(r["id"], self.account_id)
        self.assertEqual(fetched["name"], "New Name")
        self.assertEqual(fetched["is_default"], 1)

    def test_delete_report(self):
        import admin.routes as routes
        from store import report_store

        r = report_store.create_report(self.account_id, "To Delete")
        with patch.object(routes, "_is_auth", return_value=True):
            _arun(routes.report_delete(MagicMock(), self.account_id, r["id"]))
        self.assertIsNone(report_store.get_report(r["id"], self.account_id))

    def test_add_metric_to_report(self):
        import admin.routes as routes
        from store import report_store

        r = report_store.create_report(self.account_id, "R")
        with patch.object(routes, "_is_auth", return_value=True):
            result = _arun(routes.report_metric_add(
                MagicMock(), self.account_id, r["id"], metric_id=self.metric_id, sort_order=0,
            ))
        self.assertIn("saved=1", result.headers["location"])
        metrics = report_store.list_report_metrics(r["id"])
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["name"], "Revenue")

    def test_add_metric_from_other_account_rejected(self):
        import admin.routes as routes
        from store import report_store

        other_account = f"acct-admrpt-other-{uuid.uuid4().hex[:8]}"
        store.upsert_client(other_account, "portal")
        try:
            other_metric_id = store.save_metric(other_account, {
                "name": "Cost", "synonyms": "", "sql_template": "SELECT SUM(1) AS x",
            })
            r = report_store.create_report(self.account_id, "R")
            with patch.object(routes, "_is_auth", return_value=True):
                result = _arun(routes.report_metric_add(
                    MagicMock(), self.account_id, r["id"], metric_id=other_metric_id, sort_order=0,
                ))
            self.assertIn("error=", result.headers["location"])
            self.assertEqual(report_store.list_report_metrics(r["id"]), [])
        finally:
            with store.get_db() as conn:
                conn.execute("DELETE FROM metric_registry WHERE account_id=?", (other_account,))
                conn.execute("DELETE FROM client WHERE account_id=?", (other_account,))

    def test_remove_metric_from_report(self):
        import admin.routes as routes
        from store import report_store

        r = report_store.create_report(self.account_id, "R")
        report_store.add_metric_to_report(r["id"], self.metric_id)
        with patch.object(routes, "_is_auth", return_value=True):
            _arun(routes.report_metric_remove(MagicMock(), self.account_id, r["id"], self.metric_id))
        self.assertEqual(report_store.list_report_metrics(r["id"]), [])

    def test_reports_page_includes_assigned_metrics(self):
        from store import report_store

        r = report_store.create_report(self.account_id, "R")
        report_store.add_metric_to_report(r["id"], self.metric_id)
        ctx = self._get_reports_page()
        self.assertEqual(len(ctx["reports"][0]["metrics"]), 1)
        self.assertEqual(ctx["reports"][0]["metrics"][0]["name"], "Revenue")


class ClientReportsTemplateTests(unittest.TestCase):
    """Verify the Jinja template parses and renders with representative
    context — catches template syntax errors that a route-level context
    capture (which never actually invokes Jinja) would miss."""

    def test_template_renders_with_reports_and_without(self):
        import jinja2
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(ROOT / "admin" / "templates")))
        tmpl = env.get_template("client_reports.html")

        client = {"account_id": "acct1", "client_name": "Acme", "state": "READY",
                  "db_config_id": None, "chat_ui_enabled": 0}
        rendered_empty = tmpl.render(
            request=MagicMock(url=MagicMock(path="/admin/clients/acct1/reports")),
            client=client, reports=[], all_metrics=[], saved=None, error=None,
        )
        self.assertIn("No reports defined yet", rendered_empty)

        reports = [{
            "id": 1, "name": "Daily Ops", "description": "Ops summary",
            "is_default": 1, "is_active": 1,
            "metrics": [{"id": 5, "name": "Revenue", "base_table": "SALES.REVENUE"}],
        }]
        all_metrics = [{"id": 5, "name": "Revenue"}]
        rendered_full = tmpl.render(
            request=MagicMock(url=MagicMock(path="/admin/clients/acct1/reports")),
            client=client, reports=reports, all_metrics=all_metrics, saved="1", error=None,
        )
        self.assertIn("Daily Ops", rendered_full)
        self.assertIn("Revenue", rendered_full)
        self.assertIn("default-badge", rendered_full)


if __name__ == "__main__":
    unittest.main()
