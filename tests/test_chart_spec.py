import unittest
from pathlib import Path

from core.chart import build_chart_payload, detect_chart_type
from core.chart_spec import infer_chart_spec


class ChartSpecTests(unittest.TestCase):
    def test_ranking_result_prefers_bar_not_pie(self):
        rows = [
            {"Warehouse": "North", "Revenue": 1000},
            {"Warehouse": "South", "Revenue": 800},
            {"Warehouse": "West", "Revenue": 600},
            {"Warehouse": "East", "Revenue": 400},
            {"Warehouse": "Central", "Revenue": 300},
        ]
        spec = infer_chart_spec(rows, "total revenue by warehouse")
        self.assertEqual(spec["intent"], "ranking")
        self.assertEqual(spec["recommended_type"], "bar")
        self.assertEqual(detect_chart_type(rows, "total revenue by warehouse"), "bar")
        self.assertNotIn("pie", spec["allowed_types"])

    def test_share_question_allows_donut_for_small_composition(self):
        rows = [
            {"ItemGroup": "A", "RevenueShare": 40},
            {"ItemGroup": "B", "RevenueShare": 35},
            {"ItemGroup": "C", "RevenueShare": 25},
        ]
        spec = infer_chart_spec(rows, "show percentage contribution by item group")
        self.assertEqual(spec["intent"], "composition")
        self.assertEqual(spec["recommended_type"], "donut")
        self.assertIn("bar", spec["allowed_types"])

    def test_temporal_result_prefers_trend_chart(self):
        rows = [
            {"InvoiceMonth": "2026-01", "Revenue": 100},
            {"InvoiceMonth": "2026-02", "Revenue": 120},
            {"InvoiceMonth": "2026-03", "Revenue": 180},
        ]
        spec = infer_chart_spec(rows, "monthly revenue trend")
        self.assertEqual(spec["intent"], "trend")
        self.assertEqual(spec["x"]["column"], "InvoiceMonth")
        self.assertIn(spec["recommended_type"], {"area", "line"})

    def test_scatter_question_uses_two_measures(self):
        rows = [
            {"Warehouse": "A", "Revenue": 100, "GrossProfit": 20},
            {"Warehouse": "B", "Revenue": 300, "GrossProfit": 70},
            {"Warehouse": "C", "Revenue": 200, "GrossProfit": 30},
        ]
        spec = infer_chart_spec(rows, "show revenue vs gross profit")
        self.assertEqual(spec["intent"], "correlation")
        self.assertEqual(spec["recommended_type"], "scatter")
        self.assertEqual([y["column"] for y in spec["y"]], ["Revenue", "GrossProfit"])

    def test_single_row_result_becomes_kpi_spec_not_chart_payload(self):
        rows = [{"Revenue": 1200, "GrossProfit": 400}]
        spec = infer_chart_spec(rows, "what is total revenue")
        self.assertEqual(spec["intent"], "kpi")
        self.assertEqual(spec["recommended_type"], "kpi")
        self.assertIsNone(detect_chart_type(rows, "what is total revenue"))

    def test_technical_identifier_dimension_emits_warning(self):
        rows = [
            {"WHS_DMS_KEY": 1000043, "Revenue": 10},
            {"WHS_DMS_KEY": 1000068, "Revenue": 20},
            {"WHS_DMS_KEY": 1000085, "Revenue": 30},
        ]
        spec = infer_chart_spec(rows, "revenue by warehouse")
        self.assertEqual(spec["x"]["column"], "WHS_DMS_KEY")
        self.assertTrue(any("technical identifier" in w for w in spec["warnings"]))

    def test_payload_rejects_invalid_requested_type(self):
        rows = [
            {"Warehouse": "North", "Revenue": 1000},
            {"Warehouse": "South", "Revenue": 800},
        ]
        payload = build_chart_payload(rows, "scatter", title="Revenue by warehouse")
        self.assertEqual(payload["chart_type"], "bar")
        self.assertEqual(payload["requested_chart_type"], "scatter")
        self.assertEqual(payload["x_key"], "Warehouse")
        self.assertEqual(payload["y_keys"], ["Revenue"])

    def test_payload_carries_column_formats(self):
        rows = [
            {"Warehouse": "North", "TotalRevenue": 1000},
            {"Warehouse": "South", "TotalRevenue": 800},
        ]
        payload = build_chart_payload(
            rows,
            "bar",
            title="Revenue by warehouse",
            column_formats={"TotalRevenue": "currency"},
        )
        self.assertEqual(payload["column_formats"], {"TotalRevenue": "currency"})
        self.assertEqual(payload["column_roles"]["TotalRevenue"]["format"], "currency")

    def test_explicit_metric_format_prevents_numeric_result_alias_becoming_identifier(self):
        rows = [
            {"BusinessUnit": "North", "Result": 1001},
            {"BusinessUnit": "South", "Result": 1002},
        ]
        formats = {"Result": "currency"}
        spec = infer_chart_spec(rows, "result by business unit", column_formats=formats)
        self.assertEqual(spec["column_roles"]["Result"]["role"], "measure")
        self.assertEqual(detect_chart_type(rows, "result by business unit", formats), "bar")

        payload = build_chart_payload(
            rows,
            None,
            title="Result by business unit",
            column_formats=formats,
        )
        self.assertEqual(payload["x_key"], "BusinessUnit")
        self.assertEqual(payload["y_keys"], ["Result"])

    def test_payload_preserves_missing_measure_values(self):
        rows = [
            {"InvoiceMonth": "2026-01", "Revenue": 100},
            {"InvoiceMonth": "2026-02", "Revenue": None},
            {"InvoiceMonth": "2026-03", "Revenue": 150},
        ]
        payload = build_chart_payload(
            rows,
            "line",
            title="Monthly revenue trend",
            question="monthly revenue trend",
        )
        self.assertIsNone(payload["rows"][1]["Revenue"])

    def test_inventory_buildup_uses_warehouse_x_and_derived_measure_y(self):
        rows = [
            {
                "Warehouse": "EMCO 822 BURNABY",
                "Total_Purchase_Quantity": 179995.8,
                "Total_Sales_Quantity": 151776.4,
                "Inventory_Buildup": 28219.4,
            },
            {
                "Warehouse": "NOBLE 980 PORT KELLS DC",
                "Total_Purchase_Quantity": 22614.0,
                "Total_Sales_Quantity": 15374.0,
                "Inventory_Buildup": 7240.0,
            },
        ]
        question = "Which warehouses have strong purchase quantity but weak sales quantity, indicating possible inventory buildup?"
        spec = infer_chart_spec(rows, question)
        self.assertEqual(spec["recommended_type"], "bar")
        self.assertEqual(spec["x"]["column"], "Warehouse")
        self.assertEqual(spec["y"][0]["column"], "Inventory_Buildup")
        self.assertNotIn("area", spec["renderable_types"])

        payload = build_chart_payload(rows, "area", title="Inventory buildup", question=question)
        self.assertEqual(payload["chart_type"], "bar")
        self.assertEqual(payload["x_key"], "Warehouse")
        self.assertEqual(payload["y_keys"][0], "Inventory_Buildup")
        self.assertEqual(payload["rows"][0]["Warehouse"], "EMCO 822 BURNABY")

    def test_leakage_question_prioritizes_leakage_measure(self):
        rows = [
            {"Warehouse": "A", "Total_Revenue": 1000, "Gross_Profit": 100, "Profit_Leakage": 900},
            {"Warehouse": "B", "Total_Revenue": 800, "Gross_Profit": 300, "Profit_Leakage": 500},
        ]
        spec = infer_chart_spec(rows, "which warehouses have the highest profit leakage")
        self.assertEqual(spec["x"]["column"], "Warehouse")
        self.assertEqual(spec["y"][0]["column"], "Profit_Leakage")

    def test_all_decimal_currency_columns_are_measures_not_temporal(self):
        # Regression: when every value in a numeric column had a fractional
        # part (real currency amounts with cents), _looks_temporal_values'
        # integer-YYYYMMDD check filtered out all values and all([]) vacuously
        # classified the column as temporal — no measures survived, so
        # detect_chart_type returned None and no chart rendered at all.
        rows = [
            {"CUSTOMER_NAME": "SUMMIT MECHANICAL", "REVENUE": 673520.57, "GROSS_PROFIT": 152466.50},
            {"CUSTOMER_NAME": "NORM'S CASH & CARRY", "REVENUE": 311810.66, "GROSS_PROFIT": 6414.40},
            {"CUSTOMER_NAME": "CASH/VISA-PETERBOROUGH", "REVENUE": 306333.55, "GROSS_PROFIT": 83906.43},
            {"CUSTOMER_NAME": "HAMILTON SMITH LIMITED", "REVENUE": 277972.72, "GROSS_PROFIT": 40669.85},
            {"CUSTOMER_NAME": "PRIMO MECHANICAL INC.", "REVENUE": 266202.62, "GROSS_PROFIT": 64029.27},
        ]
        question = "what is my revenue and gross profit by each top 5 customers by revenue for the last 6 months"
        spec = infer_chart_spec(rows, question)
        self.assertEqual(spec["column_roles"]["REVENUE"]["role"], "measure")
        self.assertEqual(spec["column_roles"]["GROSS_PROFIT"]["role"], "measure")
        self.assertEqual(spec["recommended_type"], "bar")
        self.assertEqual(spec["x"]["column"], "CUSTOMER_NAME")
        self.assertEqual(detect_chart_type(rows, question), "bar")
        payload = build_chart_payload(rows, "bar", title=question, question=question)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["x_key"], "CUSTOMER_NAME")
        self.assertEqual(set(payload["y_keys"]), {"REVENUE", "GROSS_PROFIT"})

    def test_integer_yyyymmdd_key_column_still_temporal(self):
        rows = [
            {"INV_DT_DMS_KEY": 20260101, "REVENUE": 100.50},
            {"INV_DT_DMS_KEY": 20260201, "REVENUE": 120.25},
            {"INV_DT_DMS_KEY": 20260301, "REVENUE": 180.75},
        ]
        spec = infer_chart_spec(rows, "monthly revenue trend")
        self.assertEqual(spec["column_roles"]["INV_DT_DMS_KEY"]["role"], "temporal")
        self.assertEqual(spec["column_roles"]["REVENUE"]["role"], "measure")

    def test_non_trend_question_with_date_column_still_uses_business_dimension(self):
        rows = [
            {"Invoice_Date": "2026-01-01", "Warehouse": "A", "Revenue": 1000},
            {"Invoice_Date": "2026-01-02", "Warehouse": "B", "Revenue": 800},
        ]
        spec = infer_chart_spec(rows, "which warehouse has the highest revenue")
        self.assertEqual(spec["recommended_type"], "bar")
        self.assertEqual(spec["x"]["column"], "Warehouse")


class ChartRendererTemplateTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    CHAT = ROOT / "portal" / "templates" / "portal_chat.html"
    DASH = ROOT / "portal" / "templates" / "portal_dashboard.html"

    def _read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def test_chat_renderer_uses_chart_column_formats(self):
        src = self._read(self.CHAT)
        self.assertIn("function _chartFormatFor", src)
        self.assertIn("payload?.column_formats", src)
        self.assertIn("payload?.column_roles", src)
        self.assertIn("function _fmtChartValue", src)
        self.assertIn("valueFmt(p.value, p.seriesName)", src)

    def test_dashboard_renderer_uses_chart_column_formats(self):
        src = self._read(self.DASH)
        self.assertIn("function _chartFormatFor", src)
        self.assertIn("payload?.column_formats", src)
        self.assertIn("payload?.column_roles", src)
        self.assertIn("function _fmtChartValue", src)
        self.assertIn("valueFmt(p.value, p.seriesName)", src)

    def test_chart_type_controls_are_limited_by_renderable_types(self):
        for path in [self.CHAT, self.DASH]:
            src = self._read(path)
            self.assertIn("renderable_types", src)
            self.assertIn("allowed_types", src)
            self.assertIn("filter", src)

    def test_chart_warnings_render_in_chat_and_dashboard(self):
        for path in [self.CHAT, self.DASH]:
            src = self._read(path)
            self.assertIn("function renderChartWarnings", src)
            self.assertIn("chart_warnings", src)
            self.assertIn("chart-warning", src)

    def test_chart_renderers_preserve_missing_values_and_report_library_failure(self):
        for path in [self.CHAT, self.DASH]:
            src = self._read(path)
            self.assertIn("function _chartNumber", src)
            self.assertIn("return Number.isFinite(n) ? n : null", src)
            self.assertIn("Chart library failed to load", src)
            self.assertNotIn("Number(r?.[k] ?? 0)", src)


if __name__ == "__main__":
    unittest.main()
