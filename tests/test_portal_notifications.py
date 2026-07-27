"""
Portal "My Notifications" page (portal/routes.py: notifications_page,
notifications_delete_alert, notifications_subscribe, notifications_unsubscribe).

Real SQLite (store.init_db()) for report/subscription data; core.alert_engine
is JSON-file backed so its _load/_save are mocked directly (mirrors
tests/test_alert_engine.py's convention). portal.routes._get_portal_user is
mocked per tests/test_portal_feedback_api.py's established pattern, and
_resp is monkeypatched to capture template context instead of rendering
real Jinja (mirrors tests/test_admin_reports.py / test_semantic_conflict_inbox.py).
"""

import asyncio
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import store
from store import report_store


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


class PortalNotificationsRouteTests(unittest.TestCase):
    def setUp(self):
        store.init_db()
        self.account_id = f"acct-portalnotif-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")
        self.user_id, _ = store.create_user(self.account_id, "Test User", f"{uuid.uuid4().hex[:8]}@test.com")
        self.user = {"id": self.user_id, "account_id": self.account_id, "name": "Test User"}

    def tearDown(self):
        with store.get_db() as conn:
            conn.execute("DELETE FROM report_subscription WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM report_metric WHERE report_id IN "
                         "(SELECT id FROM report WHERE account_id=?)", (self.account_id,))
            conn.execute("DELETE FROM report WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM portal_user WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM client WHERE account_id=?", (self.account_id,))

    def _get_page(self, alerts=None):
        import portal.routes as routes
        ctx, fake_resp = _capture_resp()
        with (
            patch.object(routes, "_get_portal_user", return_value=self.user),
            patch.object(routes, "_resp", side_effect=fake_resp),
            patch("core.alert_engine.list_alerts", return_value=alerts or []),
        ):
            req = MagicMock()
            req.query_params = {}
            _arun(routes.notifications_page(req))
        return ctx

    def test_unauthenticated_redirects_to_login(self):
        import portal.routes as routes
        with patch.object(routes, "_get_portal_user", return_value=None):
            result = _arun(routes.notifications_page(MagicMock()))
        self.assertEqual(result.status_code, 303)
        self.assertIn("/portal/login", result.headers["location"])

    def test_empty_state_context(self):
        ctx = self._get_page()
        self.assertEqual(ctx["alerts"], [])
        self.assertEqual(ctx["reports"], [])
        self.assertEqual(ctx["subscriptions"], {})

    def test_alerts_filtered_to_this_user_and_account(self):
        alerts = [
            {"id": "a1", "account_id": self.account_id, "user_id": str(self.user_id), "question": "mine"},
            {"id": "a2", "account_id": self.account_id, "user_id": "999999", "question": "someone else's"},
            {"id": "a3", "account_id": "other-account", "user_id": str(self.user_id), "question": "wrong account"},
        ]
        ctx = self._get_page(alerts)
        self.assertEqual(len(ctx["alerts"]), 1)
        self.assertEqual(ctx["alerts"][0]["question"], "mine")

    def test_reports_and_subscription_state_shown(self):
        r = report_store.create_report(self.account_id, "Daily Ops")
        report_store.create_subscription(self.account_id, self.user_id, r["id"], cadence="weekly")
        ctx = self._get_page()
        self.assertEqual(len(ctx["reports"]), 1)
        self.assertIn(r["id"], ctx["subscriptions"])
        self.assertEqual(ctx["subscriptions"][r["id"]]["cadence"], "weekly")

    def test_delete_alert_only_removes_own_alert(self):
        import portal.routes as routes

        other_user_alert = {
            "id": "a-other", "account_id": self.account_id, "user_id": "999999", "question": "not mine",
        }
        with (
            patch.object(routes, "_get_portal_user", return_value=self.user),
            patch("core.alert_engine.get_alert", return_value=other_user_alert),
            patch("core.alert_engine.delete_alert") as mock_delete,
        ):
            _arun(routes.notifications_delete_alert(MagicMock(), "a-other"))
        mock_delete.assert_not_called()

    def test_delete_own_alert_succeeds(self):
        import portal.routes as routes

        own_alert = {
            "id": "a-mine", "account_id": self.account_id, "user_id": str(self.user_id), "question": "mine",
        }
        with (
            patch.object(routes, "_get_portal_user", return_value=self.user),
            patch("core.alert_engine.get_alert", return_value=own_alert),
            patch("core.alert_engine.delete_alert") as mock_delete,
        ):
            result = _arun(routes.notifications_delete_alert(MagicMock(), "a-mine"))
        mock_delete.assert_called_once_with("a-mine")
        self.assertIn("saved=1", result.headers["location"])

    def test_subscribe_creates_subscription(self):
        import portal.routes as routes

        r = report_store.create_report(self.account_id, "R")
        with patch.object(routes, "_get_portal_user", return_value=self.user):
            result = _arun(routes.notifications_subscribe(
                MagicMock(), report_id=r["id"], cadence="weekly", day_of_week=3, hour=9,
            ))
        self.assertIn("saved=1", result.headers["location"])
        subs = report_store.list_subscriptions(user_id=self.user_id)
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0]["cadence"], "weekly")

    def test_subscribe_to_missing_report_errors(self):
        import portal.routes as routes
        with patch.object(routes, "_get_portal_user", return_value=self.user):
            result = _arun(routes.notifications_subscribe(
                MagicMock(), report_id=999999, cadence="daily", day_of_week=0, hour=8,
            ))
        self.assertIn("error=", result.headers["location"])

    def test_subscribe_to_other_accounts_report_rejected(self):
        import portal.routes as routes

        other_account = f"acct-portalnotif-other-{uuid.uuid4().hex[:8]}"
        store.upsert_client(other_account, "portal")
        try:
            other_report = report_store.create_report(other_account, "Not Yours")
            with patch.object(routes, "_get_portal_user", return_value=self.user):
                result = _arun(routes.notifications_subscribe(
                    MagicMock(), report_id=other_report["id"], cadence="daily", day_of_week=0, hour=8,
                ))
            self.assertIn("error=", result.headers["location"])
            self.assertEqual(report_store.list_subscriptions(user_id=self.user_id), [])
        finally:
            with store.get_db() as conn:
                conn.execute("DELETE FROM report WHERE account_id=?", (other_account,))
                conn.execute("DELETE FROM client WHERE account_id=?", (other_account,))

    def test_unsubscribe_removes_own_subscription(self):
        import portal.routes as routes

        r = report_store.create_report(self.account_id, "R")
        sub = report_store.create_subscription(self.account_id, self.user_id, r["id"])
        with patch.object(routes, "_get_portal_user", return_value=self.user):
            _arun(routes.notifications_unsubscribe(MagicMock(), subscription_id=sub["id"]))
        self.assertEqual(report_store.list_subscriptions(user_id=self.user_id), [])

    def test_unsubscribe_cannot_remove_other_users_subscription(self):
        import portal.routes as routes

        other_user_id, _ = store.create_user(self.account_id, "Other User", f"{uuid.uuid4().hex[:8]}@test.com")
        r = report_store.create_report(self.account_id, "R")
        sub = report_store.create_subscription(self.account_id, other_user_id, r["id"])
        with patch.object(routes, "_get_portal_user", return_value=self.user):
            _arun(routes.notifications_unsubscribe(MagicMock(), subscription_id=sub["id"]))
        # Still there — delete_subscription is scoped to user_id, so my user's
        # attempt on someone else's subscription must be a no-op.
        remaining = report_store.list_subscriptions(account_id=self.account_id)
        self.assertEqual(len(remaining), 1)


class PortalNotificationsTemplateTests(unittest.TestCase):
    """Verify the Jinja template parses and renders — catches template
    syntax errors a mocked-_resp route test would never surface."""

    def test_template_renders_empty_and_populated_states(self):
        import jinja2
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(ROOT / "portal" / "templates")))
        tmpl = env.get_template("portal_notifications.html")

        user = {"id": 1, "name": "Tester", "role": "analyst", "group_name": "Ops"}
        req = MagicMock(url=MagicMock(path="/portal/notifications"))

        rendered_empty = tmpl.render(request=req, user=user, alerts=[], reports=[], subscriptions={}, saved=None, error=None)
        self.assertIn("No alerts set up yet", rendered_empty)
        self.assertIn("No reports have been set up", rendered_empty)

        alerts = [{
            "id": "a1", "question": "Revenue drop?", "metric_col": "Revenue",
            "condition": "change_pct", "threshold": 10.0, "check_interval_minutes": 60,
            "status": "active", "last_checked": "2026-07-27 08:00:00",
        }]
        reports = [{"id": 1, "name": "Daily Ops", "description": "Ops summary"}]
        subscriptions = {1: {"id": 5, "cadence": "weekly", "day_of_week": 2, "hour": 9,
                              "status": "active", "last_sent": None}}
        rendered_full = tmpl.render(
            request=req, user=user, alerts=alerts, reports=reports,
            subscriptions=subscriptions, saved="1", error=None,
        )
        self.assertIn("Revenue drop?", rendered_full)
        self.assertIn("Daily Ops", rendered_full)
        self.assertIn("Unsubscribe", rendered_full)


if __name__ == "__main__":
    unittest.main()
