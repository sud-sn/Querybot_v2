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

    def test_share_question_defaults_to_pie_for_small_composition(self):
        rows = [
            {"ItemGroup": "A", "RevenueShare": 40},
            {"ItemGroup": "B", "RevenueShare": 35},
            {"ItemGroup": "C", "RevenueShare": 25},
        ]
        spec = infer_chart_spec(rows, "show percentage contribution by item group")
        self.assertEqual(spec["intent"], "composition")
        self.assertEqual(spec["recommended_type"], "pie")
        self.assertIn("donut", spec["allowed_types"])
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

    def test_month_substring_in_dimension_values_does_not_kill_chart(self):
        # Regression: one value containing a month fragment (MARtin, NOVak,
        # DECker) used to flip the whole dimension column to temporal, leaving
        # no dimension for the bar branch — single-measure results lost their
        # chart entirely.
        rows = [
            {"CUSTOMER_NAME": "MARTIN SUPPLY CO", "REVENUE": 5000.10},
            {"CUSTOMER_NAME": "NOVAK & SONS", "REVENUE": 4000.20},
            {"CUSTOMER_NAME": "DECKER INDUSTRIES", "REVENUE": 3000.30},
        ]
        spec = infer_chart_spec(rows, "revenue by customer")
        self.assertEqual(spec["column_roles"]["CUSTOMER_NAME"]["role"], "dimension")
        self.assertEqual(spec["recommended_type"], "bar")
        self.assertEqual(detect_chart_type(rows, "revenue by customer"), "bar")

    def test_temporal_substring_in_column_name_does_not_kill_chart(self):
        # Regression: substring name matching classified CONSOLIDATED_SALES
        # ("date"), WIDTH ("dt") and OVERTIME_COST ("time") as temporal.
        rows = [
            {"PRODUCT": "A", "CONSOLIDATED_SALES": 900.15, "WIDTH": 12.5, "OVERTIME_COST": 55.25},
            {"PRODUCT": "B", "CONSOLIDATED_SALES": 800.25, "WIDTH": 9.75, "OVERTIME_COST": 44.75},
        ]
        spec = infer_chart_spec(rows, "consolidated sales by product")
        for col in ("CONSOLIDATED_SALES", "WIDTH", "OVERTIME_COST"):
            self.assertEqual(spec["column_roles"][col]["role"], "measure", col)
        self.assertEqual(detect_chart_type(rows, "consolidated sales by product"), "bar")

    def test_temporal_axis_charts_without_trend_keywords(self):
        # Regression: month + measure with no trend wording in the question
        # ("for each of the last 3 months" says neither "trend" nor
        # "by month") used to fall through every branch to table-only.
        rows = [
            {"INVOICE_MONTH": "2026-01", "REVENUE": 100.11},
            {"INVOICE_MONTH": "2026-02", "REVENUE": 120.22},
            {"INVOICE_MONTH": "2026-03", "REVENUE": 130.33},
        ]
        question = "show revenue for each of the last 3 months"
        spec = infer_chart_spec(rows, question)
        self.assertEqual(spec["intent"], "trend")
        self.assertIn(spec["recommended_type"], {"line", "area"})
        self.assertEqual(spec["x"]["column"], "INVOICE_MONTH")
        self.assertIn(detect_chart_type(rows, question), {"line", "area"})

    def test_month_name_values_still_classify_temporal(self):
        rows = [
            {"MO": "Jan", "REVENUE": 100.5},
            {"MO": "Feb", "REVENUE": 120.5},
            {"MO": "Mar", "REVENUE": 130.5},
        ]
        spec = infer_chart_spec(rows, "monthly revenue trend")
        self.assertEqual(spec["column_roles"]["MO"]["role"], "temporal")
        self.assertEqual(spec["column_roles"]["REVENUE"]["role"], "measure")

    def test_measure_named_column_never_value_sniffed_temporal(self):
        # Whole-dollar amounts that happen to sit in the 19xx/20xx year range
        # must stay measures when the column name says revenue/profit/count.
        rows = [
            {"CUSTOMER": "A", "REVENUE": 2019},
            {"CUSTOMER": "B", "REVENUE": 2045},
        ]
        spec = infer_chart_spec(rows, "revenue by customer")
        self.assertEqual(spec["column_roles"]["REVENUE"]["role"], "measure")
        self.assertEqual(detect_chart_type(rows, "revenue by customer"), "bar")

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


class ExplicitChartTypeRequestTests(unittest.TestCase):
    ROWS = [
        {"Customer": "Acme", "Sales": 5000},
        {"Customer": "Globex", "Sales": 4200},
        {"Customer": "Initech", "Sales": 3900},
        {"Customer": "Umbrella", "Sales": 3400},
        {"Customer": "Hooli", "Sales": 3100},
        {"Customer": "Soylent", "Sales": 2800},
        {"Customer": "Stark", "Sales": 2500},
        {"Customer": "Wayne", "Sales": 2200},
        {"Customer": "Wonka", "Sales": 1900},
        {"Customer": "Vandelay", "Sales": 1600},
    ]

    def test_explicit_pie_overrides_ranking_bar_recommendation(self):
        question = "give me the top 10 customer sales in a pie chart"
        spec = infer_chart_spec(self.ROWS, question)
        self.assertEqual(spec["recommended_type"], "pie")
        self.assertEqual(spec["allowed_types"][0], "pie")
        self.assertEqual(spec["x"]["column"], "Customer")
        self.assertEqual([y["column"] for y in spec["y"]], ["Sales"])
        self.assertEqual(detect_chart_type(self.ROWS, question), "pie")

    def test_explicit_donut_wording_still_honored(self):
        question = "top 5 customers by sales as a donut chart"
        self.assertEqual(detect_chart_type(self.ROWS[:5], question), "donut")

    def test_explicit_bar_overrides_share_donut_recommendation(self):
        rows = [
            {"ItemGroup": "A", "RevenueShare": 40},
            {"ItemGroup": "B", "RevenueShare": 35},
            {"ItemGroup": "C", "RevenueShare": 25},
        ]
        question = "show percentage contribution by item group as a bar chart"
        spec = infer_chart_spec(rows, question)
        self.assertEqual(spec["recommended_type"], "bar")

    def test_explicit_line_chart_wording_on_ranking_question(self):
        question = "top 5 customer sales as a line chart"
        spec = infer_chart_spec(self.ROWS[:5], question)
        self.assertEqual(spec["recommended_type"], "line")

    def test_explicit_scatter_without_second_measure_falls_back_with_warning(self):
        question = "top 10 customer sales in a scatter plot"
        spec = infer_chart_spec(self.ROWS, question)
        self.assertEqual(spec["recommended_type"], "bar")
        self.assertTrue(any("scatter chart needs at least two" in w for w in spec["warnings"]))

    def test_scatter_wording_with_two_measures_is_honored(self):
        rows = [
            {"Warehouse": "A", "Revenue": 100, "GrossProfit": 20},
            {"Warehouse": "B", "Revenue": 300, "GrossProfit": 70},
        ]
        question = "show revenue and gross profit as a scatter chart"
        spec = infer_chart_spec(rows, question)
        self.assertEqual(spec["recommended_type"], "scatter")

    def test_no_chart_keyword_leaves_ranking_heuristic_unchanged(self):
        question = "give me the top 10 customer sales"
        spec = infer_chart_spec(self.ROWS, question)
        self.assertEqual(spec["recommended_type"], "bar")

    def test_bare_type_word_without_chart_suffix_is_not_treated_as_explicit(self):
        # "bar" appearing as ordinary business language (not "bar chart")
        # must not be misread as an explicit chart-type request.
        rows = [
            {"Product": "Chocolate Bar", "Sales": 500},
            {"Product": "Soda", "Sales": 300},
        ]
        spec = infer_chart_spec(rows, "sales by product for the chocolate bar line")
        self.assertEqual(spec["recommended_type"], "bar")

    def test_explicit_type_end_to_end_through_build_chart_payload(self):
        question = "give me the top 10 customer sales in a pie chart"
        payload = build_chart_payload(self.ROWS, None, title=question, question=question)
        self.assertEqual(payload["chart_type"], "pie")
        self.assertEqual(payload["x_key"], "Customer")
        self.assertEqual(payload["y_keys"], ["Sales"])


class ChartAnnotationTests(unittest.TestCase):
    """
    core/insight.py's already-computed biggest_period_drop/biggest_period_gain
    reaching the chart payload as an `annotations` field -- previously these
    stayed completely separate from the chart (text-only callouts below it,
    per core/response_builder.py's _build_anomaly_callouts), never anchored
    to an actual point on the chart itself.
    """
    ROOT = Path(__file__).resolve().parents[1]
    CHAT = ROOT / "portal" / "templates" / "portal_chat.html"

    MONTHLY_ROWS = [
        {"Month": "2025-01", "Revenue": 1000.0},
        {"Month": "2025-02", "Revenue": 615.0},   # biggest drop lands here
        {"Month": "2025-03", "Revenue": 1200.0},  # biggest gain lands here
        {"Month": "2025-04", "Revenue": 1150.0},
    ]
    ANNOTATIONS = {
        "biggest_period_drop": {
            "from_period": "2025-01", "to_period": "2025-02",
            "absolute_change": -385.0, "pct_change": -38.5,
        },
        "biggest_period_gain": {
            "from_period": "2025-02", "to_period": "2025-03",
            "absolute_change": 585.0, "pct_change": 95.1,
        },
    }

    def test_annotations_attach_for_bar_and_line_with_matching_period(self):
        for chart_type in ("bar", "line"):
            with self.subTest(chart_type=chart_type):
                payload = build_chart_payload(
                    self.MONTHLY_ROWS, chart_type, question="revenue by month",
                    annotations=self.ANNOTATIONS,
                )
                self.assertIn("annotations", payload)
                self.assertEqual(payload["annotations"]["biggest_period_drop"]["period"], "2025-02")
                self.assertEqual(payload["annotations"]["biggest_period_drop"]["absolute_change"], -385.0)
                self.assertEqual(payload["annotations"]["biggest_period_gain"]["period"], "2025-03")

    def test_annotations_omitted_when_none_given(self):
        payload = build_chart_payload(self.MONTHLY_ROWS, "line", question="revenue by month")
        self.assertNotIn("annotations", payload)

    def test_annotations_omitted_for_non_annotatable_chart_types(self):
        # pie/donut/scatter etc. have no single per-period x-axis position
        # to anchor a drop/gain marker to. Non-temporal share-of-total data
        # (not the monthly trend rows above, which the spec correctly
        # re-routes away from pie) so the effective type actually stays pie.
        share_rows = [
            {"Region": "North", "Revenue": 400.0},
            {"Region": "South", "Revenue": 300.0},
            {"Region": "East", "Revenue": 200.0},
            {"Region": "West", "Revenue": 100.0},
        ]
        payload = build_chart_payload(
            share_rows, "pie", question="revenue share by region as a pie chart",
            annotations=self.ANNOTATIONS,
        )
        self.assertEqual(payload["chart_type"], "pie")
        self.assertNotIn("annotations", payload)

    def test_annotation_dropped_when_referenced_period_not_in_rendered_rows(self):
        # Defensive case: an annotation naming a period that isn't actually
        # in this chart's rows (e.g. a stale/mismatched brief) must never be
        # sent to the frontend to draw a dangling marker.
        stale_annotations = {
            "biggest_period_drop": {
                "from_period": "2024-11", "to_period": "2024-12",
                "absolute_change": -100.0, "pct_change": -10.0,
            },
        }
        payload = build_chart_payload(
            self.MONTHLY_ROWS, "line", question="revenue by month",
            annotations=stale_annotations,
        )
        self.assertNotIn("annotations", payload)

    def test_only_biggest_gain_present_still_attaches(self):
        payload = build_chart_payload(
            self.MONTHLY_ROWS, "line", question="revenue by month",
            annotations={"biggest_period_gain": self.ANNOTATIONS["biggest_period_gain"]},
        )
        self.assertIn("annotations", payload)
        self.assertNotIn("biggest_period_drop", payload["annotations"])
        self.assertIn("biggest_period_gain", payload["annotations"])

    def test_result_renderer_threads_brief_into_chart_payload(self):
        # Wiring guard: core/result_renderer.py must compute the brief
        # BEFORE calling build_chart_payload (not reuse a later-computed
        # one, since build_assistant_response's own brief computation
        # happens after this call site) and pass its time_series dict
        # through as the annotations kwarg -- source-text check since
        # exercising _send_results directly needs a full adapter/event/
        # db_cfg live-query context disproportionate to what's being
        # verified here (a two-line call-order fact).
        renderer_src = (self.ROOT / "core" / "result_renderer.py").read_text(encoding="utf-8")
        self.assertIn("from core.insight import generate_followup_suggestions, compute_data_brief", renderer_src)
        brief_call_pos = renderer_src.index("brief = compute_data_brief(rows, question)")
        payload_call_pos = renderer_src.index("chart_payload = build_chart_payload(")
        self.assertLess(brief_call_pos, payload_call_pos)
        self.assertIn("annotations=chart_annotations,", renderer_src)

    def test_portal_chat_renders_annotations_as_mark_points(self):
        # Wiring guard: the frontend must read payload.annotations and
        # anchor the marker at the series' OWN value at that index (not a
        # derived delta value, which would misplace the marker vertically)
        # -- both the bar and line series attach markPoint using the same
        # helper, only on the first/sole series.
        src = self.CHAT.read_text(encoding="utf-8")
        self.assertIn("function _buildAnnotationMarkPoints(payload, labels, values)", src)
        self.assertIn("coord: [idx, values[idx]]", src)
        self.assertEqual(
            src.count("markPoint: "), 2,
            "expected exactly one markPoint wiring each in the line/area and bar series builders",
        )


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

    def test_pie_renderers_name_category_and_share_in_labels_legends_and_tooltips(self):
        for path in [self.CHAT, self.DASH]:
            src = self._read(path)
            compact = src.replace(" ", "")
            self.assertIn("function _chartColumnLabel", src)
            self.assertIn("Share of total:", src)
            self.assertIn("pieLegend", src)
            self.assertIn("compactPie", src)
            self.assertIn("p.name||'Unspecified'", compact)
            self.assertIn("pieShare(p.value)", src)

    def test_mainstream_chart_tooltips_include_dynamic_dimension_label(self):
        chat = self._read(self.CHAT)
        dashboard = self._read(self.DASH)
        self.assertGreaterEqual(chat.count("escHtml(xLabel)"), 4)
        self.assertGreaterEqual(dashboard.count("_chartEscHtml(xLabel)"), 4)

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


class ChartClickToDrillTests(unittest.TestCase):
    """
    Part D (click-to-drill): clicking a bar/line/area chart mark on the
    portal fires a natural-language follow-up through the exact same
    prefillComposer/sendMessage path sendSuggestion already uses, so it's
    picked up by core/conversation_state.py's refinement classifier
    (matches on the literal substring "break this down") instead of being
    treated as a fresh, unrelated question.

    Source-text wiring guards only -- no headless browser/echarts runtime
    available in this suite, matching the convention already established
    for the annotation and mark-spec tests above.
    """

    ROOT = Path(__file__).resolve().parents[1]
    CHAT = ROOT / "portal" / "templates" / "portal_chat.html"

    def _read(self) -> str:
        return self.CHAT.read_text(encoding="utf-8")

    def test_send_chart_drill_helper_reuses_existing_send_path(self):
        src = self._read()
        self.assertIn("function sendChartDrill(text)", src)
        self.assertIn("prefillComposer(text)", src)

    def test_wire_click_handler_restricted_to_bar_line_area(self):
        src = self._read()
        self.assertIn("function _wireChartDrillClick(chart, payload)", src)
        self.assertIn("['bar', 'line', 'area'].includes(drillType)", src)
        self.assertIn("chart.on('click'", src)
        self.assertIn("params.componentType !== 'series'", src)

    def test_click_phrasing_matches_refinement_classifier_pattern(self):
        src = self._read()
        # Literal substring the conversation_state refinement regex
        # (\bbreak\s+(?:it|this|these|that)\s+down\b) matches on.
        self.assertIn("Break this down for ${label}", src)

    def test_render_chart_into_wires_click_on_initial_render_and_theme_change(self):
        src = self._read()
        start = src.index("function renderChartInto(chartEl, payload)")
        end = src.index("\nfunction ", start + 1)
        block = src[start:end]
        occurrences = block.count("_wireChartDrillClick(")
        # Once for the initial chart instance, once for the fresh instance
        # created inside the qb-theme-change listener (dispose + re-init
        # drops any handler bound to the old instance).
        self.assertEqual(occurrences, 2, block)
        self.assertIn("qb-theme-change", block)


class ChartPaletteValidationTests(unittest.TestCase):
    """
    Regression coverage for the mode-aware _PALETTES rework: the previous
    single-array-for-both-modes palettes all failed the dataviz skill's
    color-science validator outright (near-duplicate adjacent colors,
    several colors outside the OKLCH lightness/chroma bands). node isn't
    available in this environment, so the validator's exact algorithm
    (OKLCH conversion, CVD simulation, ΔE separation, WCAG contrast) is
    re-implemented here in Python and run against both template files'
    palette definitions directly, so a future edit that reintroduces a
    broken palette (or lets the two duplicated template files drift out of
    sync) fails a test instead of shipping unnoticed.
    """
    ROOT = Path(__file__).resolve().parents[1]
    CHAT = ROOT / "portal" / "templates" / "portal_chat.html"
    DASH = ROOT / "portal" / "templates" / "portal_dashboard.html"

    # -- OKLCH / CVD math, ported from the dataviz skill's
    # scripts/validate_palette.js (same thresholds, same Machado-Oliveira-
    # Fernandes 2009 CVD transforms) --
    BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
    CHROMA_FLOOR = 0.10
    CVD_FLOOR = 6.0
    NORMAL_FLOOR = 15.0  # hard gate; sunset/forest are documented exceptions below
    CONTRAST_MIN = 3.0
    SURFACES = {"light": "#ffffff", "dark": "#0f172a"}
    MACHADO = {
        "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                   [-0.003882, -0.048116, 1.051998]],
        "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                   [-0.011820, 0.042940, 0.968881]],
    }

    @classmethod
    def _hex2srgb(cls, h):
        h = h.strip().lstrip("#")
        return [int(h[i:i+2], 16) / 255 for i in (0, 2, 4)]

    @classmethod
    def _s2lin(cls, c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    @classmethod
    def _lin(cls, h):
        return [cls._s2lin(c) for c in cls._hex2srgb(h)]

    @classmethod
    def _rel_lum(cls, h):
        r, g, b = cls._lin(h)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    @classmethod
    def _contrast(cls, a, b):
        hi, lo = sorted([cls._rel_lum(a), cls._rel_lum(b)], reverse=True)
        return (hi + 0.05) / (lo + 0.05)

    @classmethod
    def _cbrt(cls, x):
        return -((-x) ** (1 / 3)) if x < 0 else x ** (1 / 3)

    @classmethod
    def _oklab_from_lin(cls, rgb):
        r, g, b = rgb
        l = cls._cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
        m = cls._cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
        s = cls._cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
        return [
            0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
        ]

    @classmethod
    def _oklab(cls, h):
        return cls._oklab_from_lin(cls._lin(h))

    @classmethod
    def _oklch(cls, h):
        L, a, b = cls._oklab(h)
        return L, (a ** 2 + b ** 2) ** 0.5

    @classmethod
    def _simulate(cls, h, kind):
        r, g, b = cls._lin(h)
        M = cls.MACHADO[kind]
        def clamp(c): return max(0, min(1, c))
        return [clamp(sum(M[row][i] * v for i, v in enumerate((r, g, b)))) for row in range(3)]

    @classmethod
    def _delta_e(cls, h1, h2, kind=None):
        a = cls._oklab_from_lin(cls._simulate(h1, kind) if kind else cls._lin(h1))
        b = cls._oklab_from_lin(cls._simulate(h2, kind) if kind else cls._lin(h2))
        return 100 * sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    @classmethod
    def _extract_palettes(cls, src: str) -> dict:
        """Pull the _PALETTES object's hex arrays out of the raw JS source.

        Parses real brace nesting rather than a fixed-width slice, since a
        fixed window can bleed into the next palette entry when one theme's
        block is shorter than another's.
        """
        import re
        outer_start = src.index("const _PALETTES = {") + len("const _PALETTES = ")
        # Find the matching closing brace for the outer object by depth-counting.
        depth = 0
        i = outer_start
        for i in range(outer_start, len(src)):
            if src[i] == "{":
                depth += 1
            elif src[i] == "}":
                depth -= 1
                if depth == 0:
                    break
        outer_block = src[outer_start + 1:i]  # contents between the outer { and }

        result = {}
        for name_match in re.finditer(r"(\w+):\s*\{", outer_block):
            name = name_match.group(1)
            inner_start = name_match.end()
            depth = 1
            j = inner_start
            for j in range(inner_start, len(outer_block)):
                if outer_block[j] == "{":
                    depth += 1
                elif outer_block[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
            inner_block = outer_block[inner_start:j]
            modes = {}
            for mode_match in re.finditer(r"(light|dark):\s*\[([^\]]*)\]", inner_block):
                hexes = re.findall(r"#[0-9a-fA-F]{6}", mode_match.group(2))
                modes[mode_match.group(1)] = hexes
            result[name] = modes
        return result

    def test_both_templates_define_identical_mode_aware_palettes(self):
        chat_palettes = self._extract_palettes(self.CHAT.read_text(encoding="utf-8"))
        dash_palettes = self._extract_palettes(self.DASH.read_text(encoding="utf-8"))
        self.assertEqual(set(chat_palettes), {"default", "ocean", "sunset", "forest", "candy", "mono"})
        self.assertEqual(chat_palettes, dash_palettes,
                          "portal_chat.html and portal_dashboard.html palettes drifted out of sync")

    def test_default_ocean_candy_pass_every_hard_gate_both_modes(self):
        # These three were tuned to fully pass; a regression here means a
        # future edit broke a previously-clean palette.
        palettes = self._extract_palettes(self.CHAT.read_text(encoding="utf-8"))
        for name in ("default", "ocean", "candy"):
            for mode in ("light", "dark"):
                hexes = palettes[name][mode]
                with self.subTest(palette=name, mode=mode):
                    lo, hi = self.BAND[mode]
                    for c in hexes:
                        L, C = self._oklch(c)
                        self.assertGreaterEqual(L, lo, f"{name}/{mode} {c} below lightness band")
                        self.assertLessEqual(L, hi, f"{name}/{mode} {c} above lightness band")
                        self.assertGreaterEqual(C, self.CHROMA_FLOOR, f"{name}/{mode} {c} below chroma floor")
                    pairs = list(zip(hexes, hexes[1:]))
                    worst_normal = min(self._delta_e(a, b) for a, b in pairs)
                    self.assertGreaterEqual(
                        worst_normal, self.NORMAL_FLOOR,
                        f"{name}/{mode} worst adjacent normal-vision ΔE {worst_normal:.1f} below {self.NORMAL_FLOOR}",
                    )
                    worst_cvd = min(
                        self._delta_e(a, b, kind) for a, b in pairs for kind in ("protan", "deutan")
                    )
                    self.assertGreaterEqual(
                        worst_cvd, self.CVD_FLOOR,
                        f"{name}/{mode} worst adjacent CVD ΔE {worst_cvd:.1f} below the {self.CVD_FLOOR} floor",
                    )

    def test_sunset_and_forest_stay_within_their_documented_ceiling(self):
        # Both are inherently narrow-hue (warm-only / green-only) themes
        # that cannot clear the 15.0 normal-vision floor for all 8 adjacent
        # pairs simultaneously in both modes (see the code comment above
        # _PALETTES) -- this pins their best-achieved worst-pair separation
        # so a future change can't silently make them WORSE than what was
        # already accepted as the ceiling.
        palettes = self._extract_palettes(self.CHAT.read_text(encoding="utf-8"))
        minimums = {"sunset": 14.0, "forest": 9.0}
        for name, floor in minimums.items():
            for mode in ("light", "dark"):
                hexes = palettes[name][mode]
                pairs = list(zip(hexes, hexes[1:]))
                worst_normal = min(self._delta_e(a, b) for a, b in pairs)
                with self.subTest(palette=name, mode=mode):
                    self.assertGreaterEqual(
                        worst_normal, floor,
                        f"{name}/{mode} worst adjacent ΔE {worst_normal:.1f} regressed below its documented floor {floor}",
                    )

    def test_bar_and_line_mark_specs_match_dataviz_skill(self):
        # Pins the concrete mark-spec fixes made to buildChartOption's
        # main bar/line/area path: bars capped at 24px (was 50px for a
        # single series -- more than double the spec's ceiling) with 4px
        # rounded ends (was 8px), lines at 2px with round caps/joins,
        # markers >=8px (was 6px, a hover-target regression too), and a
        # flat ~10% area wash instead of a steep top-to-bottom gradient.
        src = self.CHAT.read_text(encoding="utf-8")
        self.assertIn("barMaxWidth: 24", src)
        self.assertIn("borderRadius: horizontal ? [0, 4, 4, 0] : [4, 4, 0, 0]", src)
        self.assertIn("lineStyle: { width: 2, cap: 'round', join: 'round'", src)
        self.assertIn("symbolSize: rows.length <= 20 ? 8 : 0", src)
        self.assertIn("+ '1A', opacity: 1 }", src)
        self.assertNotIn("barMaxWidth: horizontal ? 20 : (isMulti ? 28 : 50)", src)

    def test_mono_is_reclassified_as_one_hue_ordinal_ramp(self):
        # mono's colors are true near-zero-chroma grayscale -- confirm they
        # stay below the categorical chroma floor (proving it genuinely
        # can't be a categorical palette, not that nobody checked) and that
        # the code comment documenting its ordinal reclassification is
        # still present.
        palettes = self._extract_palettes(self.CHAT.read_text(encoding="utf-8"))
        for mode in ("light", "dark"):
            for c in palettes["mono"][mode]:
                _, C = self._oklch(c)
                self.assertLess(C, self.CHROMA_FLOOR, f"mono/{mode} {c} unexpectedly clears the chroma floor")
        src = self.CHAT.read_text(encoding="utf-8")
        self.assertIn("Reclassified as a one-hue ORDINAL ramp", src)


if __name__ == "__main__":
    unittest.main()
