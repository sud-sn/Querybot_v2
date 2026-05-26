import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.metric_builder import compile_metric_builder_config, merge_required_columns


class MetricBuilderTests(unittest.TestCase):
    def test_compiles_sum_with_multiple_filters(self):
        compiled = compile_metric_builder_config({
            "enabled": True,
            "aggregation": "SUM",
            "measure": "SOP_CUS_IVC_LIN_AMT",
            "filters": [
                {"field": "DEL_REC_IND", "operator": "equals", "value": "0"},
                {"field": "ORDER_STATUS", "operator": "not_equals", "value": "Cancelled"},
            ],
        })
        self.assertIsNotNone(compiled)
        self.assertEqual(
            compiled.formula,
            "SUM(CASE WHEN DEL_REC_IND = 0 AND ORDER_STATUS <> 'Cancelled' THEN SOP_CUS_IVC_LIN_AMT ELSE 0 END)",
        )
        self.assertEqual(
            compiled.required_columns,
            ["SOP_CUS_IVC_LIN_AMT", "DEL_REC_IND", "ORDER_STATUS"],
        )

    def test_compiles_count_with_in_filter(self):
        compiled = compile_metric_builder_config({
            "enabled": True,
            "aggregation": "COUNT",
            "measure": "ORDER_ID",
            "filters": [
                {"field": "DIVI", "operator": "in", "value": "100, 200"},
            ],
        })
        self.assertEqual(
            compiled.formula,
            "COUNT(CASE WHEN DIVI IN (100, 200) THEN 1 END)",
        )

    def test_disabled_builder_returns_none(self):
        self.assertIsNone(compile_metric_builder_config({"enabled": False}))

    def test_rejects_sql_in_field_name(self):
        with self.assertRaises(ValueError):
            compile_metric_builder_config({
                "enabled": True,
                "aggregation": "SUM",
                "measure": "SUM(BAD)",
                "filters": [],
            })

    def test_merge_required_columns_deduplicates(self):
        self.assertEqual(
            merge_required_columns("SOP_CUS_IVC_LIN_AMT", ["SOP_CUS_IVC_LIN_AMT", "DEL_REC_IND"]),
            "SOP_CUS_IVC_LIN_AMT, DEL_REC_IND",
        )

    def test_config_json_is_normalized(self):
        compiled = compile_metric_builder_config(json.dumps({
            "enabled": True,
            "aggregation": "avg",
            "measure": "AMOUNT",
            "filters": [{"field": "ACTIVE_FLAG", "operator": "is_not_null", "value": ""}],
        }))
        self.assertIn('"aggregation":"AVG"', compiled.config_json)
        self.assertEqual(
            compiled.formula,
            "AVG(CASE WHEN ACTIVE_FLAG IS NOT NULL THEN AMOUNT END)",
        )


if __name__ == "__main__":
    unittest.main()
