"""
store/report_store.py — CRUD for named, multi-metric reports. Real DB
integration tests (not mocked), following the pattern established in
tests/test_teams_tenant_mapping.py: init_db() against the actual dev DB,
unique account_id per test, explicit teardown.
"""

import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import store
from store import report_store


class ReportStoreTestsBase(unittest.TestCase):
    def setUp(self):
        store.init_db()
        self.account_id = f"acct-report-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

    def tearDown(self):
        with store.get_db() as conn:
            conn.execute("DELETE FROM report_subscription WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM report_metric WHERE report_id IN "
                         "(SELECT id FROM report WHERE account_id=?)", (self.account_id,))
            conn.execute("DELETE FROM report WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM portal_user WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM metric_registry WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM client WHERE account_id=?", (self.account_id,))

    def _make_metric(self, name="Revenue", base_table="SALES.REVENUE") -> int:
        return store.save_metric(self.account_id, {
            "name": name, "synonyms": "", "sql_template": "SELECT SUM(1) AS x",
            "base_table": base_table,
        })


class ReportCrudTests(ReportStoreTestsBase):

    def test_create_and_get_report(self):
        r = report_store.create_report(self.account_id, "Daily Ops")
        self.assertTrue(r["id"])
        self.assertEqual(r["name"], "Daily Ops")
        self.assertEqual(r["is_default"], 0)

        fetched = report_store.get_report(r["id"], self.account_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["name"], "Daily Ops")

    def test_list_reports_returns_only_this_account(self):
        report_store.create_report(self.account_id, "Report A")
        report_store.create_report(self.account_id, "Report B")
        other_account = f"acct-report-other-{uuid.uuid4().hex[:8]}"
        store.upsert_client(other_account, "portal")
        try:
            report_store.create_report(other_account, "Report C")
            reports = report_store.list_reports(self.account_id)
            self.assertEqual({r["name"] for r in reports}, {"Report A", "Report B"})
        finally:
            with store.get_db() as conn:
                conn.execute("DELETE FROM report WHERE account_id=?", (other_account,))
                conn.execute("DELETE FROM client WHERE account_id=?", (other_account,))

    def test_creating_second_default_clears_first(self):
        r1 = report_store.create_report(self.account_id, "R1", is_default=True)
        r2 = report_store.create_report(self.account_id, "R2", is_default=True)
        reports = {r["id"]: r for r in report_store.list_reports(self.account_id)}
        self.assertEqual(reports[r1["id"]]["is_default"], 0)
        self.assertEqual(reports[r2["id"]]["is_default"], 1)

    def test_update_report_name_and_default(self):
        r = report_store.create_report(self.account_id, "Old Name")
        report_store.update_report(r["id"], self.account_id, {"name": "New Name", "is_default": True})
        fetched = report_store.get_report(r["id"], self.account_id)
        self.assertEqual(fetched["name"], "New Name")
        self.assertEqual(fetched["is_default"], 1)

    def test_update_ignores_disallowed_fields(self):
        r = report_store.create_report(self.account_id, "R")
        report_store.update_report(r["id"], self.account_id, {"account_id": "hijacked"})
        fetched = report_store.get_report(r["id"], self.account_id)
        self.assertEqual(fetched["account_id"], self.account_id)

    def test_delete_report_returns_true_when_found(self):
        r = report_store.create_report(self.account_id, "To Delete")
        self.assertTrue(report_store.delete_report(r["id"], self.account_id))
        self.assertIsNone(report_store.get_report(r["id"], self.account_id))

    def test_delete_report_returns_false_when_missing(self):
        self.assertFalse(report_store.delete_report(999999, self.account_id))

    def test_deleting_report_cascades_to_report_metric(self):
        metric_id = self._make_metric()
        r = report_store.create_report(self.account_id, "Cascade Test")
        report_store.add_metric_to_report(r["id"], metric_id)
        self.assertEqual(len(report_store.list_report_metrics(r["id"])), 1)
        report_store.delete_report(r["id"], self.account_id)
        self.assertEqual(len(report_store.list_report_metrics(r["id"])), 0)


class GetReportByNameTests(ReportStoreTestsBase):

    def test_exact_match_case_insensitive(self):
        report_store.create_report(self.account_id, "Daily Ops Report")
        found = report_store.get_report_by_name(self.account_id, "daily ops report")
        self.assertIsNotNone(found)
        self.assertEqual(found["name"], "Daily Ops Report")

    def test_substring_match_when_unambiguous(self):
        report_store.create_report(self.account_id, "Daily Ops Report")
        found = report_store.get_report_by_name(self.account_id, "ops")
        self.assertIsNotNone(found)
        self.assertEqual(found["name"], "Daily Ops Report")

    def test_ambiguous_substring_returns_none(self):
        report_store.create_report(self.account_id, "Sales Report")
        report_store.create_report(self.account_id, "Sales Summary")
        found = report_store.get_report_by_name(self.account_id, "sales")
        self.assertIsNone(found)

    def test_no_match_returns_none(self):
        report_store.create_report(self.account_id, "Daily Ops Report")
        self.assertIsNone(report_store.get_report_by_name(self.account_id, "nonexistent"))

    def test_empty_name_returns_none(self):
        report_store.create_report(self.account_id, "Daily Ops Report")
        self.assertIsNone(report_store.get_report_by_name(self.account_id, ""))

    def test_inactive_report_not_matched(self):
        r = report_store.create_report(self.account_id, "Retired Report")
        report_store.update_report(r["id"], self.account_id, {"is_active": False})
        self.assertIsNone(report_store.get_report_by_name(self.account_id, "retired report"))


class ReportMetricMembershipTests(ReportStoreTestsBase):

    def test_add_and_list_metrics_in_sort_order(self):
        m1 = self._make_metric("Revenue", "SALES.REVENUE")
        m2 = self._make_metric("Cost", "SALES.COST")
        r = report_store.create_report(self.account_id, "R")
        report_store.add_metric_to_report(r["id"], m2, sort_order=1)
        report_store.add_metric_to_report(r["id"], m1, sort_order=0)

        metrics = report_store.list_report_metrics(r["id"])
        self.assertEqual([m["name"] for m in metrics], ["Revenue", "Cost"])
        self.assertEqual(metrics[0]["base_table"], "SALES.REVENUE")

    def test_add_metric_twice_updates_sort_order_not_duplicates(self):
        m1 = self._make_metric()
        r = report_store.create_report(self.account_id, "R")
        report_store.add_metric_to_report(r["id"], m1, sort_order=0)
        report_store.add_metric_to_report(r["id"], m1, sort_order=5)
        metrics = report_store.list_report_metrics(r["id"])
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["sort_order"], 5)

    def test_remove_metric_from_report(self):
        m1 = self._make_metric()
        r = report_store.create_report(self.account_id, "R")
        report_store.add_metric_to_report(r["id"], m1)
        report_store.remove_metric_from_report(r["id"], m1)
        self.assertEqual(report_store.list_report_metrics(r["id"]), [])

    def test_empty_report_has_no_metrics(self):
        r = report_store.create_report(self.account_id, "Empty")
        self.assertEqual(report_store.list_report_metrics(r["id"]), [])


class SubscriptionTests(ReportStoreTestsBase):

    def _make_user(self) -> int:
        user_id, _ = store.create_user(self.account_id, "Test User", f"{uuid.uuid4().hex[:8]}@test.com")
        return user_id

    def test_create_and_list_subscription(self):
        user_id = self._make_user()
        r = report_store.create_report(self.account_id, "R")
        sub = report_store.create_subscription(self.account_id, user_id, r["id"], cadence="weekly", day_of_week=2, hour=9)
        self.assertEqual(sub["cadence"], "weekly")
        self.assertEqual(sub["day_of_week"], 2)
        self.assertEqual(sub["hour"], 9)
        self.assertEqual(sub["status"], "active")

        subs = report_store.list_subscriptions(account_id=self.account_id)
        self.assertEqual(len(subs), 1)

    def test_invalid_cadence_defaults_to_daily(self):
        user_id = self._make_user()
        r = report_store.create_report(self.account_id, "R")
        sub = report_store.create_subscription(self.account_id, user_id, r["id"], cadence="hourly")
        self.assertEqual(sub["cadence"], "daily")

    def test_hour_and_day_clamped_to_valid_range(self):
        user_id = self._make_user()
        r = report_store.create_report(self.account_id, "R")
        sub = report_store.create_subscription(self.account_id, user_id, r["id"], hour=99, day_of_week=99)
        self.assertEqual(sub["hour"], 23)
        self.assertEqual(sub["day_of_week"], 6)

    def test_resubscribe_updates_existing_row_not_duplicate(self):
        user_id = self._make_user()
        r = report_store.create_report(self.account_id, "R")
        report_store.create_subscription(self.account_id, user_id, r["id"], cadence="daily")
        report_store.create_subscription(self.account_id, user_id, r["id"], cadence="weekly", day_of_week=3)
        subs = report_store.list_subscriptions(account_id=self.account_id)
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0]["cadence"], "weekly")

    def test_update_subscription_status(self):
        user_id = self._make_user()
        r = report_store.create_report(self.account_id, "R")
        sub = report_store.create_subscription(self.account_id, user_id, r["id"])
        report_store.update_subscription(sub["id"], {"status": "paused"})
        subs = report_store.list_subscriptions(account_id=self.account_id)
        self.assertEqual(subs[0]["status"], "paused")

    def test_delete_subscription_scoped_to_user(self):
        user_id = self._make_user()
        other_user_id = self._make_user()
        r = report_store.create_report(self.account_id, "R")
        sub = report_store.create_subscription(self.account_id, user_id, r["id"])

        self.assertFalse(report_store.delete_subscription(sub["id"], other_user_id))
        self.assertTrue(report_store.delete_subscription(sub["id"], user_id))
        self.assertEqual(report_store.list_subscriptions(account_id=self.account_id), [])

    def test_list_subscriptions_filtered_by_user(self):
        user_id = self._make_user()
        other_user_id = self._make_user()
        r = report_store.create_report(self.account_id, "R")
        report_store.create_subscription(self.account_id, user_id, r["id"])
        report_store.create_subscription(self.account_id, other_user_id, r["id"])

        subs = report_store.list_subscriptions(user_id=user_id)
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0]["user_id"], user_id)


class SelfServiceReportTests(ReportStoreTestsBase):
    """Users can create their own reports too, not just admins --
    created_by_user_id is attribution only (shown to admins) and does not
    restrict who can ask for/subscribe to the report; only editing/deleting
    is scoped to the creator via update_own_report/delete_own_report."""

    def _make_user(self) -> int:
        user_id, _ = store.create_user(self.account_id, "Test User", f"{uuid.uuid4().hex[:8]}@test.com")
        return user_id

    def test_admin_created_report_has_no_creator(self):
        r = report_store.create_report(self.account_id, "Admin Report")
        self.assertIsNone(r["created_by_user_id"])

    def test_self_service_report_records_creator(self):
        user_id = self._make_user()
        r = report_store.create_report(self.account_id, "My Report", created_by_user_id=user_id)
        self.assertEqual(r["created_by_user_id"], user_id)

    def test_self_service_report_is_visible_in_the_shared_pool(self):
        # Shared-pool decision: a user-created report is listed and
        # name-resolvable the same as an admin-created one.
        user_id = self._make_user()
        report_store.create_report(self.account_id, "Shared Report", created_by_user_id=user_id)

        names = {r["name"] for r in report_store.list_reports(self.account_id)}
        self.assertIn("Shared Report", names)
        self.assertIsNotNone(report_store.get_report_by_name(self.account_id, "Shared Report"))

    def test_update_own_report_scoped_to_creator(self):
        user_id = self._make_user()
        other_user_id = self._make_user()
        r = report_store.create_report(self.account_id, "Mine", created_by_user_id=user_id)

        self.assertFalse(
            report_store.update_own_report(r["id"], self.account_id, other_user_id, {"name": "Hijacked"})
        )
        self.assertTrue(
            report_store.update_own_report(r["id"], self.account_id, user_id, {"name": "Renamed"})
        )
        self.assertEqual(report_store.get_report(r["id"], self.account_id)["name"], "Renamed")

    def test_update_own_report_cannot_flip_default_or_active(self):
        # is_default/is_active are account-wide policy knobs, not exposed
        # to a non-admin creator via this self-service path.
        user_id = self._make_user()
        r = report_store.create_report(self.account_id, "Mine", created_by_user_id=user_id)

        report_store.update_own_report(r["id"], self.account_id, user_id, {
            "name": "Still Mine", "is_default": True, "is_active": False,
        })
        fetched = report_store.get_report(r["id"], self.account_id)
        self.assertEqual(fetched["name"], "Still Mine")
        self.assertEqual(fetched["is_default"], 0)
        self.assertEqual(fetched["is_active"], 1)

    def test_admin_created_report_not_editable_via_update_own_report(self):
        # created_by_user_id is NULL -- no user_id can match it.
        user_id = self._make_user()
        r = report_store.create_report(self.account_id, "Admin's")

        self.assertFalse(
            report_store.update_own_report(r["id"], self.account_id, user_id, {"name": "Stolen"})
        )

    def test_delete_own_report_scoped_to_creator(self):
        user_id = self._make_user()
        other_user_id = self._make_user()
        r = report_store.create_report(self.account_id, "Mine", created_by_user_id=user_id)

        self.assertFalse(report_store.delete_own_report(r["id"], self.account_id, other_user_id))
        self.assertIsNotNone(report_store.get_report(r["id"], self.account_id))
        self.assertTrue(report_store.delete_own_report(r["id"], self.account_id, user_id))
        self.assertIsNone(report_store.get_report(r["id"], self.account_id))

    def test_admin_created_report_not_deletable_via_delete_own_report(self):
        user_id = self._make_user()
        r = report_store.create_report(self.account_id, "Admin's")

        self.assertFalse(report_store.delete_own_report(r["id"], self.account_id, user_id))
        self.assertIsNotNone(report_store.get_report(r["id"], self.account_id))


if __name__ == "__main__":
    unittest.main()
