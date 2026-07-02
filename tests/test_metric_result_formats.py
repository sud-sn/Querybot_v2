import asyncio
import unittest
from decimal import Decimal
from types import SimpleNamespace

from core.response_builder import build_assistant_response, build_column_formats
from core.result_cache import ResultCache
from core.result_renderer import (
    _detect_column_format,
    _format_value,
    _rows_to_table,
    _send_results,
)


class _PlainAdapter:
    def __init__(self):
        self.messages = []

    async def send_message(self, event, message):
        self.messages.append(message)


class _RichAdapter:
    def __init__(self):
        self.payloads = []
        self.cached_formats = {}

    def cache_result(self, rows, question, sql, db_cfg, rag_context, **kwargs):
        self.cached_formats = kwargs.get("column_formats") or {}

    async def send_assistant_response(self, event, payload):
        self.payloads.append(payload)


class MetricResultFormatTests(unittest.TestCase):
    def test_column_name_detection_handles_camel_case(self):
        self.assertEqual(_detect_column_format("TotalRevenue"), "currency")
        self.assertEqual(_detect_column_format("GrossProfit"), "currency")
        self.assertEqual(_detect_column_format("RetentionRate"), "percent")
        self.assertEqual(_detect_column_format("ProfitMargin"), "percent")
        self.assertEqual(_detect_column_format("TotalEmployees"), "number")

    def test_plural_duration_words_do_not_fall_through_to_currency(self):
        # Regression: "AVG_DAYS_TO_PAY" tokenizes to {avg, days, to, pay}.
        # _DIMENSION_KEYWORDS only had the singular "day", not "days", so
        # this fell through to the currency check where "pay" matched,
        # rendering a plain-number row-calculated metric as "$1.00" instead
        # of "1". Confirmed against a real "Avg Days To Pay" metric whose
        # result_format is explicitly "number".
        self.assertEqual(_detect_column_format("AVG_DAYS_TO_PAY"), "number")
        self.assertEqual(_detect_column_format("DAYS_SALES_OUTSTANDING"), "number")
        self.assertEqual(_detect_column_format("WEEKS_ON_HAND"), "number")
        # Currency/percent detection for real money/rate columns must be unaffected
        self.assertEqual(_detect_column_format("TOTAL_SALES"), "currency")
        self.assertEqual(_detect_column_format("TOTAL_REVENUE"), "currency")

    def test_decimal_values_honor_inferred_and_explicit_formats(self):
        value = Decimal("52677.25")
        self.assertEqual(_format_value(value, "TotalRevenue"), "$52,677.25")
        self.assertEqual(
            _format_value(value, "TotalRevenue", "number"),
            "52,677.25",
        )
        self.assertEqual(
            _format_value(Decimal("12.5"), "MetricValue", "percentage"),
            "12.50%",
        )

    def test_plain_table_uses_explicit_metric_formats(self):
        table = _rows_to_table(
            [{"MetricValue": Decimal("52677.25"), "OrderCount": 12}],
            {"MetricValue": "currency", "OrderCount": "number"},
        )
        self.assertIn("$52,677.25", table)
        self.assertIn("12", table)

    def test_currency_metric_formats_measure_not_numeric_dimension(self):
        rows = [
            {"Warehouse": 1000450, "TotalRevenue": 52677.25},
            {"Warehouse": 1000547, "TotalRevenue": 40650.14},
        ]
        formats = build_column_formats(
            rows,
            display_context={
                "format_scope": "metric_registry",
                "metrics": [{"name": "Total Revenue", "result_format": "currency"}],
            },
        )
        self.assertEqual(formats, {"TotalRevenue": "currency"})

    def test_percentage_metric_format_maps_percentage_alias(self):
        rows = [{"Division": "A", "ProfitPct": 12.5}]
        formats = build_column_formats(
            rows,
            display_context={
                "format_scope": "metric_registry",
                "metrics": [{"name": "Profit percentage", "result_format": "percentage"}],
            },
        )
        self.assertEqual(formats, {"ProfitPct": "percentage"})

    def test_llm_metric_context_does_not_format_unrelated_count(self):
        rows = [{"Warehouse": 1000450, "OrderCount": 12}]
        formats = build_column_formats(
            rows,
            display_context={
                "format_scope": "metric_context",
                "metrics": [{"name": "Total Revenue", "result_format": "currency"}],
            },
        )
        self.assertEqual(formats, {})

    def test_response_payload_includes_generic_column_formats(self):
        rows = [{"Customer": "A", "MarginRate": 18.25}]
        payload = build_assistant_response(
            question="show margin rate by customer",
            rows=rows,
            sql="SELECT Customer, MarginRate FROM result",
            duration_ms=25,
            display_context={
                "format_scope": "metric_registry",
                "metrics": [{"name": "Margin Rate", "result_format": "percentage"}],
            },
        )
        self.assertEqual(payload["data"]["column_formats"], {"MarginRate": "percentage"})
        self.assertEqual(payload["data"]["currency_columns"], [])
        self.assertIn("18.25%", payload["answer"]["headline"])

    def test_single_value_currency_metric_formats_headline(self):
        payload = build_assistant_response(
            question="what is total revenue",
            rows=[{"TotalRevenue": 52677.25}],
            sql="SELECT SUM(x) AS TotalRevenue FROM t",
            duration_ms=25,
            display_context={
                "format_scope": "metric_registry",
                "metrics": [{"name": "Total Revenue", "result_format": "currency"}],
            },
        )
        self.assertEqual(payload["answer"]["short_value"], "$52,677.25")
        self.assertIn("$52,677.25", payload["answer"]["headline"])

    def test_result_cache_preserves_explicit_formats(self):
        cache = ResultCache()
        cache.store(
            "s1",
            [{"Warehouse": 1000450, "TotalRevenue": 52677.25, "MarginPct": 10.5}],
            column_formats={"TotalRevenue": "currency", "MarginPct": "percentage"},
        )
        self.assertEqual(
            cache.get_column_formats("s1"),
            {"TotalRevenue": "currency", "MarginPct": "percentage"},
        )
        self.assertEqual(cache.get_currency_columns("s1"), ["TotalRevenue"])

    def test_registry_number_overrides_currency_sounding_alias(self):
        formats = build_column_formats(
            [{"TotalRevenue": Decimal("52677.25")}],
            display_context={
                "format_scope": "metric_registry",
                "metrics": [{"name": "Total Revenue", "result_format": "number"}],
            },
        )
        self.assertEqual(formats, {"TotalRevenue": "number"})

    def test_plain_metric_response_uses_registry_format_with_decimal(self):
        adapter = _PlainAdapter()
        asyncio.run(
            _send_results(
                SimpleNamespace(schema_hint=""),
                adapter,
                "what is total revenue",
                [{"MetricValue": Decimal("52677.25")}],
                "SELECT 52677.25 AS MetricValue",
                25,
                None,
                "acct",
                {"db_type": "azure_sql"},
                display_context={
                    "format_scope": "metric_registry",
                    "metrics": [
                        {"name": "Total Revenue", "result_format": "currency"}
                    ],
                },
            )
        )
        self.assertIn("$52,677.25", adapter.messages[0])

    def test_rich_response_and_cache_receive_inferred_format(self):
        adapter = _RichAdapter()
        asyncio.run(
            _send_results(
                SimpleNamespace(schema_hint=""),
                adapter,
                "show total revenue",
                [{"TotalRevenue": Decimal("52677.25")}],
                "SELECT 52677.25 AS TotalRevenue",
                25,
                None,
                "acct",
                {"db_type": "azure_sql"},
            )
        )
        self.assertEqual(adapter.cached_formats, {"TotalRevenue": "currency"})
        self.assertEqual(
            adapter.payloads[0]["data"]["column_formats"],
            {"TotalRevenue": "currency"},
        )


if __name__ == "__main__":
    unittest.main()
