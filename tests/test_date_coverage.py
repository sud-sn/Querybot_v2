import unittest
from unittest.mock import patch

from core.date_coverage import (
    check_date_coverage,
    _window_to_days,
    _parse_date_value,
    _safe_metric_formula,
)


class WindowToDaysTests(unittest.TestCase):
    def test_days_weeks_months(self):
        self.assertEqual(_window_to_days(7, "day"), 7)
        self.assertEqual(_window_to_days(2, "week"), 14)
        self.assertEqual(_window_to_days(1, "month"), 30)

    def test_zero_or_missing_amount_is_zero(self):
        self.assertEqual(_window_to_days(0, "day"), 0)
        self.assertEqual(_window_to_days(None, "day"), 0)

    def test_unknown_unit_is_zero(self):
        self.assertEqual(_window_to_days(5, "fortnight"), 0)


class ParseDateValueTests(unittest.TestCase):
    def test_parses_iso_date_string(self):
        self.assertEqual(_parse_date_value("2026-07-20").isoformat(), "2026-07-20")

    def test_parses_datetime_string(self):
        self.assertEqual(_parse_date_value("2026-07-20 10:30:00").isoformat(), "2026-07-20")

    def test_none_and_empty_return_none(self):
        self.assertIsNone(_parse_date_value(None))
        self.assertIsNone(_parse_date_value(""))

    def test_garbage_returns_none(self):
        self.assertIsNone(_parse_date_value("not-a-date"))


class SafeMetricFormulaTests(unittest.TestCase):
    def test_accepts_approved_aggregate_expression(self):
        self.assertEqual(
            _safe_metric_formula("SUM(REVENUE_AMOUNT)"),
            "SUM(REVENUE_AMOUNT)",
        )

    def test_rejects_query_and_qualified_expression(self):
        self.assertEqual(_safe_metric_formula("SELECT SUM(x) FROM fact"), "")
        self.assertEqual(_safe_metric_formula("SUM(f.REVENUE_AMOUNT)"), "")


class CheckDateCoverageTests(unittest.TestCase):
    def setUp(self):
        self.db_cfg = {"credentials": {}}
        self.native_policy = {
            "amount": 7, "unit": "day",
            "fact_table": "DBO.F_ORDERS", "fact_column": "ORDER_DATE",
            "date_table": "DBO.F_ORDERS", "date_column": "ORDER_DATE",
            "date_key_type": "native",
        }
        self.surrogate_policy = {
            "amount": 7, "unit": "day",
            "fact_table": "DBO.F_RX_FILL", "fact_column": "FILL_DATE_ID",
            "dimension_table": "DBO.DIM_DATE", "dimension_key": "DATE_ID",
            "date_column": "CALENDAR_DATE",
            "date_key_type": "surrogate_fk",
        }

    def test_full_coverage_returns_none(self):
        with patch("core.contextual_dates.format_required_anchor", return_value="(SELECT MAX(ORDER_DATE) FROM DBO.F_ORDERS)"), \
             patch("core.date_coverage.run_query", side_effect=[
                 [{"AnchorDate": "2026-07-20"}],
                 [{"DaysWithData": 7}],
             ]):
            gap = check_date_coverage(self.db_cfg, self.native_policy, "azure_sql")
        self.assertIsNone(gap)

    def test_partial_coverage_returns_gap(self):
        with patch("core.contextual_dates.format_required_anchor", return_value="(SELECT MAX(ORDER_DATE) FROM DBO.F_ORDERS)"), \
             patch("core.date_coverage.run_query", side_effect=[
                 [{"AnchorDate": "2026-07-20"}],
                 [{"DaysWithData": 3}],
             ]):
            gap = check_date_coverage(self.db_cfg, self.native_policy, "azure_sql")
        self.assertIsNotNone(gap)
        self.assertEqual(gap.requested_days, 7)
        self.assertEqual(gap.actual_days, 3)
        self.assertIn("last 7 days", gap.message)
        self.assertIn("only 3 business dates", gap.message)
        self.assertIn("3 days with data", gap.message)

    def test_partial_coverage_names_metric_and_business_date_role(self):
        policy = dict(self.native_policy)
        policy["business_role"] = "Invoice Date"
        with patch("core.contextual_dates.format_required_anchor", return_value="(SELECT MAX(ORDER_DATE) FROM DBO.F_ORDERS)"), \
             patch("core.date_coverage.run_query", side_effect=[
                 [{"AnchorDate": "2026-07-20"}],
                 [{"DaysWithData": 2}],
             ]):
            gap = check_date_coverage(
                self.db_cfg, policy, "azure_sql", metric_name="Revenue",
            )
        self.assertEqual(
            gap.message,
            "You asked for the last 7 days, but revenue records were found on "
            "only 2 invoice dates (2 days with data), through 2026-07-20. "
            "The result reflects the available data.",
        )

    def test_month_window_is_described_as_months_not_approximate_days(self):
        policy = dict(self.native_policy)
        policy.update({"amount": 6, "unit": "month", "business_role": "Invoice Date"})
        with patch("core.contextual_dates.format_required_anchor", return_value="(SELECT MAX(ORDER_DATE) FROM DBO.F_ORDERS)"), \
             patch("core.date_coverage.run_query", side_effect=[
                 [{"AnchorDate": "2026-07-20"}],
                 [{"DaysWithData": 2}],
             ]):
            gap = check_date_coverage(
                self.db_cfg, policy, "azure_sql", metric_name="Revenue",
            )
        self.assertIn("requested 6-month period", gap.message)
        self.assertNotIn("180-day", gap.message)

    def test_one_missing_day_is_reported(self):
        with patch("core.contextual_dates.format_required_anchor", return_value="(SELECT MAX(ORDER_DATE) FROM DBO.F_ORDERS)"), \
             patch("core.date_coverage.run_query", side_effect=[
                 [{"AnchorDate": "2026-07-20"}],
                 [{"DaysWithData": 6}],
             ]):
            gap = check_date_coverage(self.db_cfg, self.native_policy, "azure_sql")
        self.assertIsNotNone(gap)
        self.assertEqual(gap.actual_days, 6)

    def test_last_two_days_with_one_observed_day_is_reported_clearly(self):
        policy = dict(self.native_policy)
        policy.update({
            "amount": 2,
            "unit": "day",
            "business_role": "Invoice Date",
        })
        with patch("core.contextual_dates.format_required_anchor", return_value="(SELECT MAX(ORDER_DATE) FROM DBO.F_ORDERS)"), \
             patch("core.date_coverage.run_query", side_effect=[
                 [{"AnchorDate": "2026-07-20"}],
                 [{"DaysWithData": 1}],
             ]):
            gap = check_date_coverage(
                self.db_cfg, policy, "azure_sql", metric_name="Revenue",
            )
        self.assertIsNotNone(gap)
        self.assertEqual(gap.requested_days, 2)
        self.assertEqual(gap.actual_days, 1)
        self.assertEqual(
            gap.message,
            "You asked for the last 2 days, but revenue records were found on "
            "only 1 invoice date (1 day with data), through 2026-07-20. "
            "The result reflects the available data.",
        )

    def test_two_row_dates_but_one_nonzero_metric_day_is_reported(self):
        policy = dict(self.native_policy)
        policy.update({
            "amount": 2,
            "unit": "day",
            "business_role": "Invoice Date",
        })
        with patch("core.contextual_dates.format_required_anchor", return_value="(SELECT MAX(ORDER_DATE) FROM DBO.F_ORDERS)"), \
             patch("core.date_coverage.run_query", side_effect=[
                 [{"AnchorDate": "2026-07-20"}],
                 [{"DaysWithData": 2}],
                 [{"DaysWithMetricData": 1}],
             ]) as mock_run:
            gap = check_date_coverage(
                self.db_cfg,
                policy,
                "azure_sql",
                metric_name="Revenue",
                metric_formula="SUM(REVENUE_AMOUNT)",
            )
        self.assertIsNotNone(gap)
        self.assertEqual(gap.actual_days, 2)
        self.assertEqual(gap.metric_active_days, 1)
        self.assertEqual(
            gap.message,
            "You asked for the last 2 days. Records existed on 2 invoice dates, "
            "but Revenue was nonzero on only 1 day, through 2026-07-20. "
            "The result reflects the available metric values.",
        )
        activity_sql = mock_run.call_args_list[2].args[2]
        self.assertIn("GROUP BY ORDER_DATE", activity_sql)
        self.assertIn("HAVING (SUM(REVENUE_AMOUNT)) IS NOT NULL", activity_sql)
        self.assertIn("(SUM(REVENUE_AMOUNT)) <> 0", activity_sql)

    def test_metric_activity_does_not_warn_when_both_days_are_nonzero(self):
        policy = dict(self.native_policy)
        policy.update({"amount": 2, "unit": "day"})
        with patch("core.contextual_dates.format_required_anchor", return_value="(SELECT MAX(ORDER_DATE) FROM DBO.F_ORDERS)"), \
             patch("core.date_coverage.run_query", side_effect=[
                 [{"AnchorDate": "2026-07-20"}],
                 [{"DaysWithData": 2}],
                 [{"DaysWithMetricData": 2}],
             ]):
            gap = check_date_coverage(
                self.db_cfg,
                policy,
                "azure_sql",
                metric_name="Revenue",
                metric_formula="SUM(REVENUE_AMOUNT)",
            )
        self.assertIsNone(gap)

    def test_metric_probe_failure_falls_back_to_row_date_coverage(self):
        policy = dict(self.native_policy)
        policy.update({"amount": 2, "unit": "day"})
        with patch("core.contextual_dates.format_required_anchor", return_value="(SELECT MAX(ORDER_DATE) FROM DBO.F_ORDERS)"), \
             patch("core.date_coverage.run_query", side_effect=[
                 [{"AnchorDate": "2026-07-20"}],
                 [{"DaysWithData": 1}],
                 RuntimeError("formula needs another join"),
             ]):
            gap = check_date_coverage(
                self.db_cfg,
                policy,
                "azure_sql",
                metric_name="Revenue",
                metric_formula="SUM(REVENUE_AMOUNT)",
            )
        self.assertIsNotNone(gap)
        self.assertEqual(gap.actual_days, 1)
        self.assertIsNone(gap.metric_active_days)
        self.assertIn("only 1 business date", gap.message)

    def test_surrogate_fk_uses_join(self):
        with patch("core.contextual_dates.format_required_anchor",
                   return_value="(SELECT MAX(DIM_DATE.CALENDAR_DATE) FROM DBO.DIM_DATE JOIN DBO.F_RX_FILL ON DBO.F_RX_FILL.FILL_DATE_ID = DBO.DIM_DATE.DATE_ID)"), \
             patch("core.date_coverage.run_query", side_effect=[
                 [{"AnchorDate": "2026-07-20"}],
                 [{"DaysWithData": 2}],
             ]) as mock_run:
            gap = check_date_coverage(self.db_cfg, self.surrogate_policy, "azure_sql")
        self.assertIsNotNone(gap)
        self.assertEqual(gap.actual_days, 2)
        coverage_sql = mock_run.call_args_list[1].args[2]
        self.assertIn("JOIN", coverage_sql)
        self.assertIn("DATE_ID", coverage_sql)

    def test_no_temporal_window_returns_none(self):
        policy = dict(self.native_policy)
        policy["amount"] = 0
        gap = check_date_coverage(self.db_cfg, policy, "azure_sql")
        self.assertIsNone(gap)

    def test_missing_fact_fields_returns_none_not_raise(self):
        policy = {"amount": 7, "unit": "day"}
        gap = check_date_coverage(self.db_cfg, policy, "azure_sql")
        self.assertIsNone(gap)

    def test_query_failure_returns_none_not_raise(self):
        with patch("core.contextual_dates.format_required_anchor", return_value="(SELECT MAX(ORDER_DATE) FROM DBO.F_ORDERS)"), \
             patch("core.date_coverage.run_query", side_effect=RuntimeError("db down")):
            gap = check_date_coverage(self.db_cfg, self.native_policy, "azure_sql")
        self.assertIsNone(gap)

    def test_no_anchor_expression_returns_none(self):
        with patch("core.contextual_dates.format_required_anchor", return_value=""):
            gap = check_date_coverage(self.db_cfg, self.native_policy, "azure_sql")
        self.assertIsNone(gap)

    def test_unsafe_identifier_returns_none_not_raise(self):
        policy = dict(self.native_policy)
        policy["fact_table"] = "DBO.F_ORDERS; DROP TABLE x"
        gap = check_date_coverage(self.db_cfg, policy, "azure_sql")
        self.assertIsNone(gap)


if __name__ == "__main__":
    unittest.main()
