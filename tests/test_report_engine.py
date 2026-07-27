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


class ApplyLatestDateFilterTests(unittest.TestCase):
    """core.report_engine._apply_latest_date_filter -- sqlglot parse-mutate-
    regenerate anchor injection. Mirrors the exact sqlglot cases manually
    verified during planning (no WHERE, existing WHERE, WHERE+GROUP BY,
    string-literal-containing-clause-text, CTE, UNION)."""

    def test_no_existing_where_injects_new_clause(self):
        sql = "SELECT SUM(Revenue) AS Revenue FROM Sales"
        result = re_engine._apply_latest_date_filter(sql, "Sales", "SnapshotDate", "azure_sql")
        self.assertIn("WHERE", result.upper())
        self.assertIn("MAX(SnapshotDate)".upper(), result.upper())
        self.assertIn("SALES", result.upper())

    def test_existing_where_appends_with_and(self):
        sql = "SELECT SUM(Revenue) AS Revenue FROM Sales WHERE Region = 'US'"
        result = re_engine._apply_latest_date_filter(sql, "Sales", "SnapshotDate", "azure_sql")
        self.assertIn("REGION", result.upper())
        self.assertIn("AND", result.upper())
        self.assertIn("MAX(SNAPSHOTDATE)", result.upper())

    def test_where_followed_by_group_by_filter_lands_in_where(self):
        sql = "SELECT Region, SUM(Revenue) AS Revenue FROM Sales WHERE Region = 'US' GROUP BY Region"
        result = re_engine._apply_latest_date_filter(sql, "Sales", "SnapshotDate", "azure_sql")
        # The MAX(...) anchor must land before GROUP BY, not after.
        where_idx = result.upper().index("WHERE")
        group_idx = result.upper().index("GROUP BY")
        max_idx = result.upper().index("MAX(SNAPSHOTDATE)")
        self.assertTrue(where_idx < max_idx < group_idx)

    def test_string_literal_containing_clause_keywords_not_confused(self):
        sql = "SELECT x FROM t WHERE status = 'GROUP BY THIS'"
        result = re_engine._apply_latest_date_filter(sql, "t", "d", "azure_sql")
        self.assertIn("GROUP BY THIS", result)
        self.assertIn("MAX(d)".upper(), result.upper())

    def test_cte_gets_filter_on_outer_select(self):
        sql = "WITH cte AS (SELECT 1 AS x) SELECT SUM(x) FROM cte"
        result = re_engine._apply_latest_date_filter(sql, "cte", "d", "azure_sql")
        self.assertIsNotNone(result)
        self.assertIn("MAX(d)".upper(), result.upper())

    def test_union_returns_none(self):
        sql = "SELECT a FROM t1 UNION SELECT b FROM t2"
        result = re_engine._apply_latest_date_filter(sql, "t1", "d", "azure_sql")
        self.assertIsNone(result)

    def test_unparseable_sql_returns_none(self):
        sql = "SELECT SELECT FROM WHERE ((("
        result = re_engine._apply_latest_date_filter(sql, "t", "d", "azure_sql")
        self.assertIsNone(result)

    def test_unsafe_fact_table_returns_none(self):
        sql = "SELECT SUM(x) FROM t"
        result = re_engine._apply_latest_date_filter(sql, "t; DROP TABLE users; --", "d", "azure_sql")
        self.assertIsNone(result)

    def test_unsafe_fact_column_returns_none(self):
        sql = "SELECT SUM(x) FROM t"
        result = re_engine._apply_latest_date_filter(sql, "t", "d) OR (1=1", "azure_sql")
        self.assertIsNone(result)

    def test_qualified_table_name_allowed(self):
        sql = "SELECT SUM(x) FROM t"
        result = re_engine._apply_latest_date_filter(sql, "DB.SCHEMA.TABLE", "d", "azure_sql")
        self.assertIsNotNone(result)

    def test_all_three_dialects_produce_valid_output(self):
        sql = "SELECT SUM(x) AS x FROM t"
        for db_type in ("snowflake", "oracle", "azure_sql"):
            result = re_engine._apply_latest_date_filter(sql, "t", "d", db_type)
            self.assertIsNotNone(result, db_type)
            self.assertIn("MAX(D)", result.upper(), db_type)


class ResolveDefaultDateFilterTests(unittest.TestCase):
    """core.report_engine._resolve_default_date_filter -- reuses
    resolve_contextual_date_binding with a synthetic 'today' question since
    reports have no live question text to score phrases against."""

    def test_no_base_table_returns_none_without_any_store_calls(self):
        metric = {"id": 1, "name": "Revenue", "base_table": ""}
        with patch("store.list_metric_date_contexts") as mock_list:
            result = re_engine._resolve_default_date_filter("acct1", metric, {})
        self.assertIsNone(result)
        mock_list.assert_not_called()

    def test_no_bindings_and_no_date_roles_returns_none(self):
        metric = {"id": 1, "name": "Revenue", "base_table": "SALES.REVENUE"}
        with (
            patch("store.list_metric_date_contexts", return_value=[]),
            patch("core.semantic_contract.load_contract", return_value={}),
        ):
            result = re_engine._resolve_default_date_filter("acct1", metric, {"kb_dir": "x"})
        self.assertIsNone(result)

    def test_resolver_status_none_returns_none(self):
        metric = {"id": 1, "name": "Revenue", "base_table": "SALES.REVENUE"}
        with (
            patch("store.list_metric_date_contexts", return_value=[{"is_default": 0}]),
            patch("core.semantic_contract.load_contract", return_value={"model": {"date_roles": []}}),
            patch("core.contextual_dates.resolve_contextual_date_binding", return_value={"status": "none"}),
        ):
            result = re_engine._resolve_default_date_filter("acct1", metric, {"kb_dir": "x"})
        self.assertIsNone(result)

    def test_resolver_status_ambiguous_returns_none(self):
        metric = {"id": 1, "name": "Revenue", "base_table": "SALES.REVENUE"}
        with (
            patch("store.list_metric_date_contexts", return_value=[{"is_default": 0}]),
            patch("core.semantic_contract.load_contract", return_value={"model": {"date_roles": []}}),
            patch("core.contextual_dates.resolve_contextual_date_binding", return_value={"status": "ambiguous"}),
        ):
            result = re_engine._resolve_default_date_filter("acct1", metric, {"kb_dir": "x"})
        self.assertIsNone(result)

    def test_resolver_status_selected_many_returns_none(self):
        metric = {"id": 1, "name": "Revenue", "base_table": "SALES.REVENUE"}
        with (
            patch("store.list_metric_date_contexts", return_value=[{"is_default": 0}]),
            patch("core.semantic_contract.load_contract", return_value={"model": {"date_roles": []}}),
            patch("core.contextual_dates.resolve_contextual_date_binding", return_value={"status": "selected_many"}),
        ):
            result = re_engine._resolve_default_date_filter("acct1", metric, {"kb_dir": "x"})
        self.assertIsNone(result)

    def test_metric_level_default_selected_returns_fact_table_and_column(self):
        metric = {"id": 1, "name": "Revenue", "base_table": "SALES.REVENUE"}
        binding = {"fact_table": "SALES.REVENUE", "fact_column": "INVOICE_DATE_ID"}
        with (
            patch("store.list_metric_date_contexts", return_value=[{"is_default": 1, **binding}]),
            patch("core.semantic_contract.load_contract", return_value={"model": {"date_roles": []}}),
            patch("core.contextual_dates.resolve_contextual_date_binding",
                  return_value={"status": "selected", "binding": binding}) as mock_resolve,
        ):
            result = re_engine._resolve_default_date_filter("acct1", metric, {"kb_dir": "x"})
        self.assertEqual(result, ("SALES.REVENUE", "INVOICE_DATE_ID"))
        # Synthetic "today" question drives the resolver -- no live NL question exists for reports.
        self.assertEqual(mock_resolve.call_args[0][0], "today")

    def test_fact_level_default_role_selected_returns_fact_table_and_column(self):
        metric = {"id": 1, "name": "Revenue", "base_table": "SALES.REVENUE"}
        binding = {"fact_table": "SALES.REVENUE", "fact_column": "SHIP_DATE_ID"}
        with (
            patch("store.list_metric_date_contexts", return_value=[]),
            patch("core.semantic_contract.load_contract",
                  return_value={"model": {"date_roles": [{"is_default": 1, "status": "approved", **binding}]}}),
            patch("core.contextual_dates.resolve_contextual_date_binding",
                  return_value={"status": "selected", "binding": binding}),
        ):
            result = re_engine._resolve_default_date_filter("acct1", metric, {"kb_dir": "x"})
        self.assertEqual(result, ("SALES.REVENUE", "SHIP_DATE_ID"))

    def test_selected_binding_missing_fact_column_returns_none(self):
        metric = {"id": 1, "name": "Revenue", "base_table": "SALES.REVENUE"}
        with (
            patch("store.list_metric_date_contexts", return_value=[{"is_default": 1}]),
            patch("core.semantic_contract.load_contract", return_value={"model": {"date_roles": []}}),
            patch("core.contextual_dates.resolve_contextual_date_binding",
                  return_value={"status": "selected", "binding": {"fact_table": "SALES.REVENUE", "fact_column": ""}}),
        ):
            result = re_engine._resolve_default_date_filter("acct1", metric, {"kb_dir": "x"})
        self.assertIsNone(result)


class RunMetricForReportDateFilterIntegrationTests(unittest.TestCase):
    """Verifies the resolved/injected SQL is what actually reaches
    execute_governed_query -- the real end-to-end wiring, not just the
    two helpers in isolation."""

    def _run(self, *, date_filter):
        captured = {}

        def _fake_execute(creds, db_type, sql, **kwargs):
            captured["sql"] = sql
            governed = MagicMock()
            governed.rows = [{"Revenue": 100.0}]
            governed.analysis.resources = []
            return governed

        with (
            patch("store.get_allowed_tables", return_value=None),
            patch("store.get_client_state", return_value={"kb_dir": "x", "schema_dir": ""}),
            patch("core.pipeline_context.get_client_db",
                  return_value={"db_type": "azure_sql", "credentials": {}}),
            patch("core.report_engine._resolve_default_date_filter", return_value=date_filter),
            patch("core.compliance.policy_engine.resolve_context", return_value=MagicMock(purpose_id="p1")),
            patch("core.compliance.governed_query.execute_governed_query", side_effect=_fake_execute),
            patch("core.compliance.policy_engine.evaluate", return_value=MagicMock(effective_allowed=True)),
            patch("core.schema.load_known_tables", return_value={}),
            patch("core.schema.load_schema_columns", return_value={}),
        ):
            re_engine.run_metric_for_report("acct1", {"id": 1}, _metric())
        return captured.get("sql", "")

    def test_resolvable_default_injects_filter_into_executed_sql(self):
        sql = self._run(date_filter=("SALES.REVENUE", "SNAPSHOT_DATE_ID"))
        self.assertIn("MAX(SNAPSHOT_DATE_ID)".upper(), sql.upper())

    def test_no_resolvable_default_leaves_sql_byte_for_byte_unchanged(self):
        sql = self._run(date_filter=None)
        self.assertEqual(sql, _metric()["sql_template"])

    def test_date_filter_resolution_exception_falls_back_to_original_sql(self):
        with patch("core.report_engine._resolve_default_date_filter", side_effect=Exception("boom")):
            captured = {}

            def _fake_execute(creds, db_type, sql, **kwargs):
                captured["sql"] = sql
                governed = MagicMock()
                governed.rows = [{"Revenue": 100.0}]
                governed.analysis.resources = []
                return governed

            with (
                patch("store.get_allowed_tables", return_value=None),
                patch("store.get_client_state", return_value={"kb_dir": "x", "schema_dir": ""}),
                patch("core.pipeline_context.get_client_db",
                      return_value={"db_type": "azure_sql", "credentials": {}}),
                patch("core.compliance.policy_engine.resolve_context", return_value=MagicMock(purpose_id="p1")),
                patch("core.compliance.governed_query.execute_governed_query", side_effect=_fake_execute),
                patch("core.compliance.policy_engine.evaluate", return_value=MagicMock(effective_allowed=True)),
                patch("core.schema.load_known_tables", return_value={}),
                patch("core.schema.load_schema_columns", return_value={}),
            ):
                result = re_engine.run_metric_for_report("acct1", {"id": 1}, _metric())
        self.assertTrue(result["ok"])  # must not fail the metric
        self.assertEqual(captured["sql"], _metric()["sql_template"])


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


class SubscriptionDueTests(unittest.TestCase):

    def _sub(self, **overrides):
        base = {"cadence": "daily", "hour": 8, "day_of_week": 0, "last_sent": None}
        base.update(overrides)
        return base

    def test_never_sent_and_hour_reached_is_due(self):
        from datetime import datetime
        self.assertTrue(re_engine._subscription_due(self._sub(), datetime(2026, 7, 27, 8, 30)))

    def test_hour_not_yet_reached_not_due(self):
        from datetime import datetime
        self.assertFalse(re_engine._subscription_due(self._sub(hour=9), datetime(2026, 7, 27, 8, 30)))

    def test_daily_already_sent_today_not_due(self):
        from datetime import datetime
        sub = self._sub(last_sent="2026-07-27 08:05:00")
        self.assertFalse(re_engine._subscription_due(sub, datetime(2026, 7, 27, 9, 0)))

    def test_daily_sent_yesterday_is_due_again(self):
        from datetime import datetime
        sub = self._sub(last_sent="2026-07-26 08:05:00")
        self.assertTrue(re_engine._subscription_due(sub, datetime(2026, 7, 27, 9, 0)))

    def test_weekly_wrong_day_not_due(self):
        from datetime import datetime
        # 2026-07-27 is a Monday (weekday 0); subscription wants Wednesday (2).
        sub = self._sub(cadence="weekly", day_of_week=2, hour=8)
        self.assertFalse(re_engine._subscription_due(sub, datetime(2026, 7, 27, 9, 0)))

    def test_weekly_correct_day_is_due(self):
        from datetime import datetime
        sub = self._sub(cadence="weekly", day_of_week=0, hour=8)  # Monday
        self.assertTrue(re_engine._subscription_due(sub, datetime(2026, 7, 27, 9, 0)))

    def test_missing_hour_defaults_to_8(self):
        from datetime import datetime
        sub = {"cadence": "daily", "last_sent": None}
        self.assertFalse(re_engine._subscription_due(sub, datetime(2026, 7, 27, 7, 59)))
        self.assertTrue(re_engine._subscription_due(sub, datetime(2026, 7, 27, 8, 0)))


class RunDueReportDigestsTests(unittest.TestCase):

    def _sub(self, **overrides):
        base = {
            "id": 1, "account_id": "acct1", "user_id": 7, "report_id": 1,
            "cadence": "daily", "hour": 8, "day_of_week": 0, "last_sent": None,
            "status": "active",
        }
        base.update(overrides)
        return base

    def _run(self, subs, *, report=_UNSET, user=_UNSET, response=None):
        captured = {}
        resolved_report = {"id": 1, "name": "R"} if report is _UNSET else report
        resolved_user = {"id": 7} if user is _UNSET else user

        async def _fake_deliver(account_id, user_id, resp):
            captured["account_id"] = account_id
            captured["user_id"] = user_id
            captured["response"] = resp

        with (
            patch("store.report_store.list_subscriptions", return_value=subs),
            patch("store.report_store.get_report", return_value=resolved_report),
            patch("store.get_user", return_value=resolved_user),
            patch("core.report_engine.build_report_response",
                  return_value=response if response is not None else {"ok": True, "message": "**R**", "items": []}) as mock_build,
            patch("core.report_engine._deliver_report_response", side_effect=_fake_deliver),
            patch("store.report_store.update_subscription") as mock_update,
        ):
            re_engine.run_due_report_digests()
        return captured, mock_build, mock_update

    def test_inactive_subscription_skipped(self):
        sub = self._sub(status="paused")
        captured, mock_build, mock_update = self._run([sub])
        mock_build.assert_not_called()
        self.assertEqual(captured, {})

    def test_not_due_subscription_skipped(self):
        sub = self._sub(hour=23)
        with patch("core.report_engine.datetime") as mock_dt:
            from datetime import datetime as real_datetime
            fixed_now = real_datetime(2026, 7, 27, 8, 0, 0)
            mock_dt.now.return_value = fixed_now
            captured, mock_build, mock_update = self._run([sub])
        mock_build.assert_not_called()

    def test_due_subscription_delivers_and_updates_last_sent(self):
        sub = self._sub()
        captured, mock_build, mock_update = self._run([sub])
        mock_build.assert_called_once()
        self.assertEqual(captured["account_id"], "acct1")
        self.assertEqual(captured["user_id"], 7)
        mock_update.assert_called_once()
        args, _ = mock_update.call_args
        self.assertEqual(args[0], 1)
        self.assertIn("last_sent", args[1])

    def test_missing_report_skips_without_raising(self):
        sub = self._sub()
        captured, mock_build, mock_update = self._run([sub], report=None)
        mock_build.assert_not_called()
        mock_update.assert_not_called()

    def test_missing_user_skips_without_raising(self):
        sub = self._sub()
        captured, mock_build, mock_update = self._run([sub], user=None)
        mock_build.assert_not_called()
        mock_update.assert_not_called()

    def test_one_subscription_exception_does_not_block_others(self):
        sub_bad = self._sub(id=1, account_id="acct-bad")
        sub_ok = self._sub(id=2, account_id="acct-ok")

        def _fake_get_report(report_id, account_id):
            if account_id == "acct-bad":
                raise Exception("boom")
            return {"id": 1, "name": "R"}

        captured = {}

        async def _fake_deliver(account_id, user_id, resp):
            captured["account_id"] = account_id

        with (
            patch("store.report_store.list_subscriptions", return_value=[sub_bad, sub_ok]),
            patch("store.report_store.get_report", side_effect=_fake_get_report),
            patch("store.get_user", return_value={"id": 7}),
            patch("core.report_engine.build_report_response", return_value={"ok": True, "message": "R", "items": []}),
            patch("core.report_engine._deliver_report_response", side_effect=_fake_deliver),
            patch("store.report_store.update_subscription"),
        ):
            re_engine.run_due_report_digests()  # must not raise

        self.assertEqual(captured["account_id"], "acct-ok")

    def test_delivery_exception_skips_update_but_does_not_raise(self):
        sub = self._sub()
        with (
            patch("store.report_store.list_subscriptions", return_value=[sub]),
            patch("store.report_store.get_report", return_value={"id": 1, "name": "R"}),
            patch("store.get_user", return_value={"id": 7}),
            patch("core.report_engine.build_report_response", return_value={"ok": True, "message": "R", "items": []}),
            patch("core.report_engine._deliver_report_response", side_effect=Exception("network down")),
            patch("store.report_store.update_subscription") as mock_update,
        ):
            re_engine.run_due_report_digests()  # must not raise
        mock_update.assert_not_called()


class DeliverReportResponseTests(unittest.TestCase):

    def test_sends_header_then_each_item(self):
        import asyncio
        calls = []

        async def _fake_notify(account_id, user_id, message, chart=None):
            calls.append((account_id, user_id, message, chart))

        response = {
            "ok": True,
            "message": "**R**",
            "items": [
                {"text": "**Revenue**: 100", "chart": None},
                {"text": "**Trend** — see chart below.", "chart": {"chart_type": "bar"}},
            ],
        }
        with patch("core.notify.send_proactive_notification", side_effect=_fake_notify):
            asyncio.run(re_engine._deliver_report_response("acct1", 7, response))

        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][2], "**R**")
        self.assertEqual(calls[2][3], {"chart_type": "bar"})

    def test_not_ok_response_sends_only_header(self):
        import asyncio
        calls = []

        async def _fake_notify(account_id, user_id, message, chart=None):
            calls.append(message)

        response = {"ok": False, "message": "no metrics", "items": []}
        with patch("core.notify.send_proactive_notification", side_effect=_fake_notify):
            asyncio.run(re_engine._deliver_report_response("acct1", 7, response))
        self.assertEqual(calls, ["no metrics"])


if __name__ == "__main__":
    unittest.main()
