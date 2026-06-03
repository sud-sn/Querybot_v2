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


if __name__ == "__main__":
    unittest.main()
