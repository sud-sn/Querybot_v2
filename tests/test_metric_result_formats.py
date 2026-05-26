import unittest

from core.response_builder import build_assistant_response, build_column_formats
from core.result_cache import ResultCache


class MetricResultFormatTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
