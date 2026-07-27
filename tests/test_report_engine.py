"""
core/report_engine.py — run a report's metrics for a specific user,
respecting per-user table access, and build the three deterministic
messages plus per-metric chart replies.

All DB/governed-query I/O is mocked — mirrors the mocking conventions in
tests/test_alert_engine.py.
"""

import unittest
from unittest.mock import MagicMock, patch

import core.report_engine as re_engine


def _metric(**overrides) -> dict:
    base = {
        "id": 1, "name": "Revenue", "base_table": "SALES.REVENUE",
        "sql_template": "SELECT SUM(amount) AS Revenue FROM Sales",
    }
    base.update(overrides)
    return base


class RunMetricForReportAccessTests(unittest.TestCase):

    def test_denied_when_base_table_not_in_allowed_set(self):
        with patch("store.get_allowed_tables", return_value={"OTHER.TABLE"}):
            result = re_engine.run_metric_for_report("acct1", {"id": 1}, _metric())
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "access_denied")
        self.assertEqual(result["metric_name"], "Revenue")

    def test_denial_never_names_the_table(self):
        with patch("store.get_allowed_tables", return_value={"OTHER.TABLE"}):
            result = re_engine.run_metric_for_report("acct1", {"id": 1}, _metric())
        self.assertNotIn("SALES.REVENUE", str(result))

    def test_admin_unrestricted_none_bypasses_check(self):
        with (
            patch("store.get_allowed_tables", return_value=None),
            patch("store.get_client_state", return_value={}),
            patch("core.pipeline_context.get_client_db", return_value={"db_type": "azure_sql", "credentials": {}}),
            patch("core.compliance.policy_engine.resolve_context") as mock_resolve,
            patch("core.compliance.governed_query.execute_governed_query") as mock_exec,
            patch("core.compliance.policy_engine.evaluate") as mock_eval,
            patch("core.schema.load_known_tables", return_value={}),
            patch("core.schema.load_schema_columns", return_value={}),
        ):
            mock_resolve.return_value = MagicMock(purpose_id="p1")
            mock_governed = MagicMock()
            mock_governed.rows = [{"Revenue": 1000.0}]
            mock_governed.analysis.resources = []
            mock_exec.return_value = mock_governed
            mock_eval.return_value = MagicMock(effective_allowed=True)

            result = re_engine.run_metric_for_report("acct1", {"id": 1}, _metric())
        self.assertTrue(result["ok"])

    def test_metric_with_no_base_table_skips_access_check(self):
        metric = _metric(base_table="")
        with (
            patch("store.get_client_state", return_value={}),
            patch("core.pipeline_context.get_client_db", return_value={"db_type": "azure_sql", "credentials": {}}),
            patch("core.compliance.policy_engine.resolve_context") as mock_resolve,
            patch("core.compliance.governed_query.execute_governed_query") as mock_exec,
            patch("core.compliance.policy_engine.evaluate") as mock_eval,
            patch("core.schema.load_known_tables", return_value={}),
            patch("core.schema.load_schema_columns", return_value={}),
            patch("store.get_allowed_tables") as mock_allowed,
        ):
            mock_resolve.return_value = MagicMock(purpose_id="p1")
            mock_governed = MagicMock()
            mock_governed.rows = [{"Revenue": 500.0}]
            mock_governed.analysis.resources = []
            mock_exec.return_value = mock_governed
            mock_eval.return_value = MagicMock(effective_allowed=True)

            result = re_engine.run_metric_for_report("acct1", {"id": 1}, metric)
        self.assertTrue(result["ok"])
        # get_allowed_tables is still callable elsewhere in the function (the
        # governed-query allowed_tables kwarg) but the presence check itself
        # must not have gated on an empty base_table.
        self.assertNotEqual(result.get("reason"), "access_denied")


_UNSET = object()


class RunMetricForReportExecutionTests(unittest.TestCase):

    def _run(self, *, rows=None, effective_allowed=True, raises=None, db_cfg=_UNSET, no_sql=False):
        metric = _metric(sql_template="" if no_sql else _metric()["sql_template"])
        resolved_db_cfg = {"db_type": "azure_sql", "credentials": {}} if db_cfg is _UNSET else db_cfg
        with (
            patch("store.get_allowed_tables", return_value=None),
            patch("store.get_client_state", return_value={}),
            patch("core.pipeline_context.get_client_db", return_value=resolved_db_cfg),
            patch("core.compliance.policy_engine.resolve_context") as mock_resolve,
            patch("core.compliance.governed_query.execute_governed_query") as mock_exec,
            patch("core.compliance.policy_engine.evaluate") as mock_eval,
            patch("core.schema.load_known_tables", return_value={}),
            patch("core.schema.load_schema_columns", return_value={}),
        ):
            mock_resolve.return_value = MagicMock(purpose_id="p1")
            if raises:
                mock_exec.side_effect = raises
            else:
                mock_governed = MagicMock()
                mock_governed.rows = rows if rows is not None else []
                mock_governed.analysis.resources = []
                mock_exec.return_value = mock_governed
            mock_eval.return_value = MagicMock(effective_allowed=effective_allowed)
            return re_engine.run_metric_for_report("acct1", {"id": 1}, metric)

    def test_no_sql_template_returns_no_sql_reason(self):
        result = self._run(no_sql=True)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "no_sql")

    def test_no_db_cfg_returns_no_database_reason(self):
        result = self._run(db_cfg=None)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "no_database")

    def test_query_exception_returns_query_failed(self):
        result = self._run(raises=Exception("timeout"))
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "query_failed")

    def test_policy_denial_returns_access_denied(self):
        result = self._run(rows=[{"Revenue": 1.0}], effective_allowed=False)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "access_denied")

    def test_no_rows_returns_ok_with_empty_rows_no_chart(self):
        result = self._run(rows=[])
        self.assertTrue(result["ok"])
        self.assertEqual(result["rows"], [])
        self.assertIsNone(result["chart"])

    def test_success_builds_chart(self):
        rows = [{"Month": "Jan", "Revenue": 100.0}, {"Month": "Feb", "Revenue": 120.0}]
        with patch("core.chart.detect_chart_type", return_value="bar"), \
             patch("core.chart.build_chart_payload", return_value={"chart_type": "bar", "rows": rows}) as mock_build:
            result = self._run(rows=rows)
        self.assertTrue(result["ok"])
        self.assertEqual(result["rows"], rows)
        self.assertEqual(result["chart"]["chart_type"], "bar")
        mock_build.assert_called_once()

    def test_chart_build_failure_does_not_fail_the_metric(self):
        rows = [{"Revenue": 100.0}]
        with patch("core.chart.detect_chart_type", side_effect=Exception("chart boom")):
            result = self._run(rows=rows)
        self.assertTrue(result["ok"])
        self.assertIsNone(result["chart"])


class BuildReportResponseTests(unittest.TestCase):

    def test_no_metrics_anywhere_in_account(self):
        with patch("store.list_metrics", return_value=[]):
            response = re_engine.build_report_response("acct1", {"id": 1}, {"id": 1, "name": "R"})
        self.assertFalse(response["ok"])
        self.assertIn("ask your admin", response["message"].lower())
        self.assertEqual(response["items"], [])

    def test_report_has_no_metrics_assigned(self):
        with (
            patch("store.list_metrics", return_value=[_metric()]),
            patch("store.report_store.list_report_metrics", return_value=[]),
        ):
            response = re_engine.build_report_response("acct1", {"id": 1}, {"id": 1, "name": "Empty Report"})
        self.assertFalse(response["ok"])
        self.assertIn("Empty Report", response["message"])
        self.assertEqual(response["items"], [])

    def test_success_builds_one_item_per_metric(self):
        metrics = [_metric(id=1, name="Revenue"), _metric(id=2, name="Cost", base_table="SALES.COST")]
        with (
            patch("store.list_metrics", return_value=metrics),
            patch("store.report_store.list_report_metrics", return_value=metrics),
            patch("core.report_engine.run_metric_for_report") as mock_run,
        ):
            mock_run.side_effect = [
                {"ok": True, "metric_name": "Revenue", "rows": [{"Revenue": 100.0}], "chart": None},
                {"ok": True, "metric_name": "Cost", "rows": [{"Cost": 50.0}], "chart": None},
            ]
            response = re_engine.build_report_response("acct1", {"id": 1}, {"id": 1, "name": "Ops Report"})
        self.assertTrue(response["ok"])
        self.assertEqual(len(response["items"]), 2)
        self.assertIn("Revenue", response["items"][0]["text"])
        self.assertIn("Cost", response["items"][1]["text"])

    def test_denied_metric_message_names_metric_not_table(self):
        metrics = [_metric(id=1, name="Confidential Payroll", base_table="HR.PAYROLL")]
        with (
            patch("store.list_metrics", return_value=metrics),
            patch("store.report_store.list_report_metrics", return_value=metrics),
            patch("core.report_engine.run_metric_for_report",
                  return_value={"ok": False, "reason": "access_denied", "metric_name": "Confidential Payroll"}),
        ):
            response = re_engine.build_report_response("acct1", {"id": 1}, {"id": 1, "name": "Payroll Report"})
        self.assertTrue(response["ok"])
        item_text = response["items"][0]["text"]
        self.assertIn("Confidential Payroll", item_text)
        self.assertNotIn("HR.PAYROLL", item_text)


class FormatMetricLineTests(unittest.TestCase):

    def test_scalar_single_row_single_col(self):
        line = re_engine._format_metric_line({"ok": True, "metric_name": "Revenue", "rows": [{"Revenue": 1234.5}]})
        self.assertIn("Revenue", line)
        self.assertIn("1,234.5", line)

    def test_multi_row_points_to_chart(self):
        line = re_engine._format_metric_line({"ok": True, "metric_name": "Trend", "rows": [{"m": "Jan"}, {"m": "Feb"}]})
        self.assertIn("chart below", line)

    def test_no_rows(self):
        line = re_engine._format_metric_line({"ok": True, "metric_name": "Empty", "rows": []})
        self.assertIn("no data", line)

    def test_access_denied(self):
        line = re_engine._format_metric_line({"ok": False, "reason": "access_denied", "metric_name": "Secret"})
        self.assertIn("Secret", line)
        self.assertIn("access", line.lower())

    def test_generic_failure(self):
        line = re_engine._format_metric_line({"ok": False, "reason": "query_failed", "metric_name": "Broken"})
        self.assertIn("Broken", line)


if __name__ == "__main__":
    unittest.main()
