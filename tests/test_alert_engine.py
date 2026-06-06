"""
Sprint E — Alert engine tests.

All storage I/O is mocked so these tests never touch the filesystem.
check_alert_now() DB calls are mocked to return controlled row sets.

Groups:
  CreateAlertTests      — create_alert stores correct fields
  ListGetDeleteTests    — list_alerts / get_alert / delete_alert CRUD
  CheckAlertNowTests    — condition evaluation, trigger paths, fallbacks
  InvalidConditionTests — unknown condition defaults to change_pct
"""

import unittest
from unittest.mock import MagicMock, patch

# Module under test — imported once here; tests patch _load/_save at the
# module level to avoid any real filesystem access.
import core.alert_engine as ae


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

def _make_alert(**overrides) -> dict:
    """Return a minimal valid alert definition dict."""
    base = {
        "id": "abc12345",
        "question": "What is total revenue?",
        "sql": "SELECT SUM(Revenue) AS Revenue FROM Sales",
        "metric_col": "Revenue",
        "baseline_value": 1000.0,
        "condition": "change_pct",
        "threshold": 10.0,
        "db_type": "azure_sql",
        "created_at": "2026-06-06T10:00:00",
        "last_checked": None,
        "last_value": None,
        "status": "active",
    }
    base.update(overrides)
    return base


# ══════════════════════════════════════════════════════════════════════════════
# create_alert
# ══════════════════════════════════════════════════════════════════════════════

class CreateAlertTests(unittest.TestCase):

    def setUp(self):
        self._store: list[dict] = []

    def _make_create(self, **kwargs) -> dict:
        saved: list[list] = []

        def _fake_load():
            return list(self._store)

        def _fake_save(alerts):
            self._store.clear()
            self._store.extend(alerts)

        with (
            patch.object(ae, "_load", side_effect=_fake_load),
            patch.object(ae, "_save", side_effect=_fake_save),
        ):
            return ae.create_alert(**kwargs)

    def test_returns_dict(self):
        a = self._make_create(
            question="Rev?", sql="SELECT 1", metric_col="Rev",
            baseline_value=500.0,
        )
        self.assertIsInstance(a, dict)

    def test_id_generated(self):
        a = self._make_create(
            question="Rev?", sql="SELECT 1", metric_col="Rev",
            baseline_value=500.0,
        )
        self.assertTrue(a.get("id"))
        self.assertIsInstance(a["id"], str)

    def test_question_stored(self):
        a = self._make_create(
            question="Total sales?", sql="SELECT 1", metric_col="Sales",
            baseline_value=200.0,
        )
        self.assertEqual(a["question"], "Total sales?")

    def test_sql_stored(self):
        sql = "SELECT SUM(Rev) FROM Sales"
        a = self._make_create(
            question="q", sql=sql, metric_col="Rev", baseline_value=100.0,
        )
        self.assertEqual(a["sql"], sql)

    def test_metric_col_stored(self):
        a = self._make_create(
            question="q", sql="SELECT 1", metric_col="Revenue",
            baseline_value=300.0,
        )
        self.assertEqual(a["metric_col"], "Revenue")

    def test_baseline_value_rounded_to_4dp(self):
        a = self._make_create(
            question="q", sql="SELECT 1", metric_col="X",
            baseline_value=1234.56789,
        )
        self.assertEqual(a["baseline_value"], round(1234.56789, 4))

    def test_default_condition_is_change_pct(self):
        a = self._make_create(
            question="q", sql="SELECT 1", metric_col="X",
            baseline_value=100.0,
        )
        self.assertEqual(a["condition"], "change_pct")

    def test_custom_condition_above(self):
        a = self._make_create(
            question="q", sql="SELECT 1", metric_col="X",
            baseline_value=100.0, condition="above", threshold=500.0,
        )
        self.assertEqual(a["condition"], "above")
        self.assertEqual(a["threshold"], 500.0)

    def test_status_is_active(self):
        a = self._make_create(
            question="q", sql="SELECT 1", metric_col="X",
            baseline_value=100.0,
        )
        self.assertEqual(a["status"], "active")

    def test_created_at_is_string(self):
        a = self._make_create(
            question="q", sql="SELECT 1", metric_col="X",
            baseline_value=100.0,
        )
        self.assertIsInstance(a["created_at"], str)

    def test_db_cfg_credentials_not_stored(self):
        """Sensitive credential keys must not appear in the stored alert."""
        a = self._make_create(
            question="q", sql="SELECT 1", metric_col="X",
            baseline_value=100.0,
            db_cfg={
                "db_type": "azure_sql",
                "credentials": {"user": "sa", "password": "secret"},
                "server": "prod-server.example.com",
            },
        )
        alert_str = str(a)
        self.assertNotIn("secret", alert_str)
        self.assertNotIn("password", alert_str)
        self.assertEqual(a["db_type"], "azure_sql")

    def test_alert_appended_to_store(self):
        self._make_create(
            question="q", sql="SELECT 1", metric_col="X",
            baseline_value=100.0,
        )
        self.assertEqual(len(self._store), 1)

    def test_multiple_creates_accumulate(self):
        for i in range(3):
            self._make_create(
                question=f"q{i}", sql="SELECT 1", metric_col="X",
                baseline_value=float(i),
            )
        self.assertEqual(len(self._store), 3)


# ══════════════════════════════════════════════════════════════════════════════
# list_alerts / get_alert / delete_alert
# ══════════════════════════════════════════════════════════════════════════════

class ListGetDeleteTests(unittest.TestCase):

    def setUp(self):
        self._store = [_make_alert(id=f"id{i}") for i in range(3)]

    def test_list_returns_all(self):
        with patch.object(ae, "_load", return_value=list(self._store)):
            result = ae.list_alerts()
        self.assertEqual(len(result), 3)

    def test_get_alert_found(self):
        with patch.object(ae, "_load", return_value=list(self._store)):
            a = ae.get_alert("id0")
        self.assertIsNotNone(a)
        self.assertEqual(a["id"], "id0")

    def test_get_alert_not_found_returns_none(self):
        with patch.object(ae, "_load", return_value=list(self._store)):
            a = ae.get_alert("nonexistent")
        self.assertIsNone(a)

    def test_delete_alert_returns_true_when_found(self):
        store = list(self._store)

        def _fake_save(alerts):
            store.clear()
            store.extend(alerts)

        with (
            patch.object(ae, "_load", return_value=list(store)),
            patch.object(ae, "_save", side_effect=_fake_save),
        ):
            result = ae.delete_alert("id1")
        self.assertTrue(result)

    def test_delete_alert_returns_false_when_missing(self):
        with (
            patch.object(ae, "_load", return_value=list(self._store)),
            patch.object(ae, "_save"),
        ):
            result = ae.delete_alert("does_not_exist")
        self.assertFalse(result)


# ══════════════════════════════════════════════════════════════════════════════
# check_alert_now — condition evaluation
# ══════════════════════════════════════════════════════════════════════════════

class CheckAlertNowTests(unittest.TestCase):

    _DB_CFG = {"db_type": "azure_sql", "credentials": {}}

    def _run_check(self, alert: dict, returned_rows: list[dict]) -> dict:
        """Run check_alert_now with a mocked DB call."""
        with (
            patch.object(ae, "get_alert", return_value=alert),
            patch.object(ae, "_load", return_value=[alert]),
            patch.object(ae, "_save"),
            patch("core.schema.run_query", return_value=returned_rows),
        ):
            return ae.check_alert_now(alert["id"], self._DB_CFG)

    # — change_pct condition ─────────────────────────────────────────────────

    def test_change_pct_triggered_when_exceeds_threshold(self):
        alert = _make_alert(
            baseline_value=1000.0, condition="change_pct", threshold=10.0
        )
        result = self._run_check(alert, [{"Revenue": 1200.0}])
        self.assertTrue(result["ok"])
        self.assertTrue(result["triggered"])     # 20% change ≥ 10%

    def test_change_pct_not_triggered_below_threshold(self):
        alert = _make_alert(
            baseline_value=1000.0, condition="change_pct", threshold=10.0
        )
        result = self._run_check(alert, [{"Revenue": 1050.0}])
        self.assertTrue(result["ok"])
        self.assertFalse(result["triggered"])    # 5% change < 10%

    def test_change_pct_decrease_also_triggers(self):
        alert = _make_alert(
            baseline_value=1000.0, condition="change_pct", threshold=10.0
        )
        result = self._run_check(alert, [{"Revenue": 800.0}])
        self.assertTrue(result["triggered"])     # −20% ≥ 10%

    # — above condition ──────────────────────────────────────────────────────

    def test_above_triggered(self):
        alert = _make_alert(
            baseline_value=100.0, condition="above", threshold=500.0
        )
        result = self._run_check(alert, [{"Revenue": 600.0}])
        self.assertTrue(result["triggered"])     # 600 > 500

    def test_above_not_triggered(self):
        alert = _make_alert(
            baseline_value=100.0, condition="above", threshold=500.0
        )
        result = self._run_check(alert, [{"Revenue": 400.0}])
        self.assertFalse(result["triggered"])    # 400 < 500

    # — below condition ──────────────────────────────────────────────────────

    def test_below_triggered(self):
        alert = _make_alert(
            baseline_value=100.0, condition="below", threshold=50.0
        )
        result = self._run_check(alert, [{"Revenue": 30.0}])
        self.assertTrue(result["triggered"])     # 30 < 50

    def test_below_not_triggered(self):
        alert = _make_alert(
            baseline_value=100.0, condition="below", threshold=50.0
        )
        result = self._run_check(alert, [{"Revenue": 80.0}])
        self.assertFalse(result["triggered"])    # 80 > 50

    # — result dict contract ─────────────────────────────────────────────────

    def test_result_contains_required_keys(self):
        alert = _make_alert()
        result = self._run_check(alert, [{"Revenue": 1100.0}])
        for key in (
            "ok", "triggered", "alert_id", "metric_col",
            "current_value", "baseline_value", "delta_pct",
            "condition", "threshold", "message", "checked_at",
        ):
            self.assertIn(key, result, f"missing key: {key!r}")

    def test_current_value_correct(self):
        alert = _make_alert(baseline_value=1000.0)
        result = self._run_check(alert, [{"Revenue": 1500.0}])
        self.assertAlmostEqual(result["current_value"], 1500.0, places=2)

    def test_delta_pct_correct(self):
        alert = _make_alert(baseline_value=1000.0)
        result = self._run_check(alert, [{"Revenue": 1200.0}])
        self.assertAlmostEqual(result["delta_pct"], 20.0, places=1)

    def test_message_contains_metric_col(self):
        alert = _make_alert(metric_col="Revenue")
        result = self._run_check(alert, [{"Revenue": 900.0}])
        self.assertIn("Revenue", result["message"])

    # — failure paths ─────────────────────────────────────────────────────────

    def test_alert_not_found_returns_ok_false(self):
        with patch.object(ae, "get_alert", return_value=None):
            result = ae.check_alert_now("missing", self._DB_CFG)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "alert_not_found")

    def test_inactive_alert_returns_ok_false(self):
        alert = _make_alert(status="paused")
        with patch.object(ae, "get_alert", return_value=alert):
            result = ae.check_alert_now(alert["id"], self._DB_CFG)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "alert_inactive")

    def test_no_rows_returns_ok_false(self):
        alert = _make_alert()
        result = self._run_check(alert, [])
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "no_rows")

    def test_non_numeric_metric_returns_ok_false(self):
        alert = _make_alert()
        result = self._run_check(alert, [{"Revenue": "N/A"}])
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "metric_not_numeric")

    def test_query_exception_returns_ok_false(self):
        alert = _make_alert()
        with (
            patch.object(ae, "get_alert", return_value=alert),
            patch("core.schema.run_query", side_effect=Exception("DB timeout")),
        ):
            result = ae.check_alert_now(alert["id"], self._DB_CFG)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "query_failed")


# ══════════════════════════════════════════════════════════════════════════════
# Invalid condition handling
# ══════════════════════════════════════════════════════════════════════════════

class InvalidConditionTests(unittest.TestCase):

    def test_unknown_condition_defaults_to_change_pct(self):
        saved: list[list] = []

        def _fake_load():
            return list(saved)

        def _fake_save(alerts):
            saved.clear()
            saved.extend(alerts)

        with (
            patch.object(ae, "_load", side_effect=_fake_load),
            patch.object(ae, "_save", side_effect=_fake_save),
        ):
            a = ae.create_alert(
                question="q", sql="SELECT 1", metric_col="X",
                baseline_value=100.0, condition="invalid_condition",
            )
        self.assertEqual(a["condition"], "change_pct")

    def test_valid_conditions_accepted(self):
        for cond in ("change_pct", "above", "below"):
            saved: list = []

            def _fake_load():
                return list(saved)

            def _fake_save(alerts):
                saved.clear()
                saved.extend(alerts)

            with (
                patch.object(ae, "_load", side_effect=_fake_load),
                patch.object(ae, "_save", side_effect=_fake_save),
            ):
                a = ae.create_alert(
                    question="q", sql="SELECT 1", metric_col="X",
                    baseline_value=100.0, condition=cond,
                )
            self.assertEqual(a["condition"], cond, f"condition {cond!r} not preserved")


if __name__ == "__main__":
    unittest.main()
