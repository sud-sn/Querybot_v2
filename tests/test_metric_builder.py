import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.metric_builder import compile_metric_builder_config, merge_required_columns
from core.pipeline_helpers import _format_metric_formula_context, _build_row_metric_join_sql


class MetricBuilderTests(unittest.TestCase):
    def test_metric_builder_ui_exposes_row_calculated_controls(self):
        template = (ROOT / "admin" / "templates" / "client_metrics.html").read_text(encoding="utf-8")
        self.assertIn('value="row_calculated"', template)
        self.assertIn("metric-builder-row-expression", template)
        self.assertIn("metric-builder-row-aggregation", template)
        self.assertIn("metric-builder-required-joins", template)
        self.assertIn("insertDateRoleJoinTemplate", template)
        self.assertIn('data-builder-mode="row_calculated"', template)

    def test_metric_builder_ui_wires_schema_autocomplete_to_row_calc_fields(self):
        template = (ROOT / "admin" / "templates" / "client_metrics.html").read_text(encoding="utf-8")
        self.assertIn('.metric-builder-row-expression,.metric-builder-required-joins', template)
        self.assertIn("_joinTableContext", template)
        self.assertIn("_distinctTables", template)
        self.assertIn('editor.matches(".metric-builder-required-joins")', template)

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

    def test_compiles_row_calculated_average_metric(self):
        compiled = compile_metric_builder_config({
            "enabled": True,
            "mode": "row_calculated",
            "aggregation": "AVG",
            "row_expression": (
                "CASE "
                "WHEN due_dt.DMS_DT IS NULL THEN NULL "
                "WHEN pay_dt.DMS_DT IS NOT NULL THEN DATEDIFF(day, due_dt.DMS_DT, pay_dt.DMS_DT) "
                "WHEN pay_dt.DMS_DT IS NULL AND due_dt.DMS_DT < GETDATE() THEN DATEDIFF(day, due_dt.DMS_DT, GETDATE()) "
                "ELSE 0 END"
            ),
            "required_columns": ["DUE_DT_DMS_KEY", "PAY_DT_DMS_KEY"],
            "required_joins": [
                {
                    "alias": "due_dt",
                    "table": "DT_DMS",
                    "from_column": "DUE_DT_DMS_KEY",
                    "to_column": "DT_DMS_KEY",
                    "role": "due date",
                },
                {
                    "alias": "pay_dt",
                    "table": "DT_DMS",
                    "from_column": "PAY_DT_DMS_KEY",
                    "to_column": "DT_DMS_KEY",
                    "role": "payment date",
                },
            ],
        })
        self.assertIsNotNone(compiled)
        self.assertTrue(compiled.formula.startswith("AVG(CAST((CASE WHEN due_dt.DMS_DT IS NULL"))
        self.assertIn("DATEDIFF(day, due_dt.DMS_DT, pay_dt.DMS_DT)", compiled.formula)
        self.assertEqual(compiled.required_columns, ["DUE_DT_DMS_KEY", "PAY_DT_DMS_KEY"])
        cfg = json.loads(compiled.config_json)
        self.assertEqual(cfg["mode"], "row_calculated")
        self.assertEqual(cfg["required_joins"][0]["alias"], "due_dt")

    def test_row_calculated_metric_rejects_query_text(self):
        with self.assertRaises(ValueError):
            compile_metric_builder_config({
                "enabled": True,
                "mode": "row_calculated",
                "aggregation": "AVG",
                "row_expression": "SELECT DATEDIFF(day, a, b) FROM bad",
            })

    def test_row_calculated_metric_rejects_nested_aggregate(self):
        with self.assertRaises(ValueError):
            compile_metric_builder_config({
                "enabled": True,
                "mode": "row_calculated",
                "aggregation": "AVG",
                "row_expression": "SUM(DATEDIFF(day, due_dt.DMS_DT, pay_dt.DMS_DT))",
            })

    def test_metric_prompt_includes_row_join_hints(self):
        compiled = compile_metric_builder_config({
            "enabled": True,
            "mode": "row_calculated",
            "aggregation": "AVG",
            "row_expression": "CASE WHEN pay_dt.DMS_DT IS NOT NULL THEN DATEDIFF(day, due_dt.DMS_DT, pay_dt.DMS_DT) ELSE 0 END",
            "required_columns": ["DUE_DT_DMS_KEY", "PAY_DT_DMS_KEY"],
            "required_joins": [
                {"alias": "due_dt", "table": "DT_DMS", "from_column": "DUE_DT_DMS_KEY", "to_column": "DT_DMS_KEY", "role": "due date"},
                {"alias": "pay_dt", "table": "DT_DMS", "from_column": "PAY_DT_DMS_KEY", "to_column": "DT_DMS_KEY", "role": "payment date"},
            ],
        })
        prompt = _format_metric_formula_context([{
            "name": "Days Between Due and Payment",
            "formula_type": "expression",
            "result_format": "number",
            "synonyms": "payment delay, days to pay",
            "required_columns": ", ".join(compiled.required_columns),
            "metric_builder_config": compiled.config_json,
            "sql_template": compiled.formula,
        }])
        self.assertIn("Row-level metric", prompt)
        self.assertIn("Join DT_DMS AS due_dt ON fact.DUE_DT_DMS_KEY = due_dt.DT_DMS_KEY for due date", prompt)
        self.assertIn("Join DT_DMS AS pay_dt ON fact.PAY_DT_DMS_KEY = pay_dt.DT_DMS_KEY for payment date", prompt)

    def test_row_joins_appended_to_graph_skeleton_azure(self):
        skeleton = "FROM [dbo].[CUS_ORD_IVC_FCT] inv\nINNER JOIN [dbo].[DIM_DATE] d ON inv.[DATE_KEY] = d.[DATE_KEY]"
        compiled = compile_metric_builder_config({
            "enabled": True,
            "mode": "row_calculated",
            "aggregation": "AVG",
            "row_expression": "DATEDIFF(day, due_dt.DMS_DT, pay_dt.DMS_DT)",
            "required_columns": ["DUE_DT_DMS_KEY", "PAY_DT_DMS_KEY"],
            "required_joins": [
                {"alias": "due_dt", "table": "DT_DMS", "from_column": "DUE_DT_DMS_KEY", "to_column": "DT_DMS_KEY", "role": "due date"},
                {"alias": "pay_dt", "table": "DT_DMS", "from_column": "PAY_DT_DMS_KEY", "to_column": "DT_DMS_KEY", "role": "payment date"},
            ],
        })
        metric = {
            "formula_type": "expression",
            "metric_builder_config": compiled.config_json,
            "sql_template": compiled.formula,
        }
        result = _build_row_metric_join_sql([metric], "azure_sql", skeleton)
        self.assertIn("LEFT  JOIN [DT_DMS] due_dt ON inv.[DUE_DT_DMS_KEY] = due_dt.[DT_DMS_KEY]", result)
        self.assertIn("LEFT  JOIN [DT_DMS] pay_dt ON inv.[PAY_DT_DMS_KEY] = pay_dt.[DT_DMS_KEY]", result)

    def test_row_joins_deduplicates_same_join(self):
        skeleton = "FROM [dbo].[FACT] f"
        metric_cfg = json.dumps({
            "enabled": True, "mode": "row_calculated", "aggregation": "AVG",
            "row_expression": "due_dt.DMS_DT",
            "required_joins": [
                {"alias": "due_dt", "table": "DT_DMS", "from_column": "DUE_DT_DMS_KEY", "to_column": "DT_DMS_KEY"},
            ],
        })
        metrics = [
            {"metric_builder_config": metric_cfg},
            {"metric_builder_config": metric_cfg},
        ]
        result = _build_row_metric_join_sql(metrics, "azure_sql", skeleton)
        self.assertEqual(result.count("LEFT  JOIN"), 1)

    def test_row_joins_empty_when_no_anchor(self):
        skeleton = ""
        metric_cfg = json.dumps({
            "enabled": True, "mode": "row_calculated", "aggregation": "AVG",
            "row_expression": "due_dt.DMS_DT",
            "required_joins": [
                {"alias": "due_dt", "table": "DT_DMS", "from_column": "DUE_DT_DMS_KEY", "to_column": "DT_DMS_KEY"},
            ],
        })
        result = _build_row_metric_join_sql([{"metric_builder_config": metric_cfg}], "azure_sql", skeleton)
        self.assertEqual(result, "")

    def test_row_joins_skips_aggregate_metrics(self):
        skeleton = "FROM [dbo].[FACT] f"
        metric_cfg = json.dumps({
            "enabled": True, "mode": "aggregate", "aggregation": "SUM",
            "measure": "AMOUNT", "filters": [],
        })
        result = _build_row_metric_join_sql([{"metric_builder_config": metric_cfg}], "azure_sql", skeleton)
        self.assertEqual(result, "")

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
