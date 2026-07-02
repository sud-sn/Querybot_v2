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

    def test_initialise_metric_builder_does_not_clobber_in_progress_user_edits(self):
        # Regression: _initialiseMetricBuilder only ever runs from the async
        # /api/columns fetch callback. If that fetch is still in flight when
        # the admin opens the edit dialog and pastes/types into the row
        # expression, it used to resolve later and silently overwrite the
        # user's in-progress edit with the stale server-rendered saved
        # config the moment it completed — losing the paste entirely.
        # Confirmed with a live repro harness before this fix: the paste
        # survived once _initialiseMetricBuilder skips value population for
        # a builder marked userEdited, and normal (untouched) builders still
        # populate correctly from the saved config on first load.
        template = (ROOT / "admin" / "templates" / "client_metrics.html").read_text(encoding="utf-8")
        self.assertIn('if(builder.dataset.userEdited === "1"){', template)
        self.assertIn('_mbForFlag.dataset.userEdited = "1";', template)
        # The guard must be the FIRST thing _initialiseMetricBuilder checks,
        # before it reads builder.dataset.config / overwrites any field.
        idx_fn = template.index("function _initialiseMetricBuilder(builder){")
        idx_guard = template.index('builder.dataset.userEdited === "1"', idx_fn)
        idx_overwrite = template.index("rowExpression.value = cfg.row_expression", idx_fn)
        self.assertLess(idx_guard, idx_overwrite)

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


class DateGapMetricTests(unittest.TestCase):
    """Wizard mode: gap between two role-playing date keys."""

    def _config(self, **overrides):
        cfg = {
            "enabled": True,
            "mode": "date_gap",
            "aggregation": "AVG",
            "unit": "day",
            "missing_to": "today_if_overdue_else_zero",
            "date_column": "DMS_DT",
            "invalid_keys": "0, 777",
            "from_join": {"alias": "due_dt", "table": "DT_DMS", "from_column": "DUE_DT_DMS_KEY", "to_column": "DT_DMS_KEY", "role": "due date"},
            "to_join":   {"alias": "pay_dt", "table": "DT_DMS", "from_column": "PAY_DT_DMS_KEY", "to_column": "DT_DMS_KEY", "role": "payment date"},
        }
        cfg.update(overrides)
        return cfg

    def test_azure_formula_matches_dax_semantics(self):
        compiled = compile_metric_builder_config(self._config(), db_type="azure_sql")
        f = compiled.formula
        self.assertIn("WHEN due_dt.DMS_DT IS NULL THEN NULL", f)
        self.assertIn("WHEN pay_dt.DMS_DT IS NOT NULL THEN DATEDIFF(day, due_dt.DMS_DT, pay_dt.DMS_DT)", f)
        self.assertIn("WHEN due_dt.DMS_DT < CAST(GETDATE() AS date) THEN DATEDIFF(day, due_dt.DMS_DT, CAST(GETDATE() AS date))", f)
        self.assertIn("ELSE 0", f)
        self.assertEqual(compiled.required_columns, ["DUE_DT_DMS_KEY", "PAY_DT_DMS_KEY"])

    def test_oracle_uses_date_subtraction(self):
        f = compile_metric_builder_config(self._config(), db_type="oracle").formula
        self.assertIn("(pay_dt.DMS_DT - due_dt.DMS_DT)", f)
        self.assertIn("TRUNC(SYSDATE)", f)
        self.assertNotIn("DATEDIFF", f)

    def test_snowflake_quotes_the_unit(self):
        f = compile_metric_builder_config(self._config(), db_type="snowflake").formula
        self.assertIn("DATEDIFF('day', due_dt.DMS_DT, pay_dt.DMS_DT)", f)
        self.assertIn("CURRENT_DATE", f)

    def test_missing_to_variants(self):
        f_null = compile_metric_builder_config(self._config(missing_to="null")).formula
        self.assertIn("ELSE NULL", f_null)
        f_zero = compile_metric_builder_config(self._config(missing_to="zero")).formula
        self.assertIn("ELSE 0", f_zero)
        f_today = compile_metric_builder_config(self._config(missing_to="today")).formula
        self.assertIn("ELSE DATEDIFF(day, due_dt.DMS_DT, CAST(GETDATE() AS date))", f_today)

    def test_invalid_keys_attach_to_both_joins(self):
        clean = json.loads(compile_metric_builder_config(self._config()).config_json)
        self.assertEqual(clean["invalid_keys"], [0, 777])
        for join in clean["required_joins"]:
            self.assertEqual(join["invalid_keys"], [0, 777])

    def test_clean_config_carries_row_calculated_equivalents(self):
        # Downstream consumers (prompt injection, join skeleton, base-table
        # inference) read row_expression/required_joins/required_columns.
        clean = json.loads(compile_metric_builder_config(self._config()).config_json)
        self.assertEqual(clean["mode"], "date_gap")
        self.assertTrue(clean["row_expression"].startswith("CASE "))
        self.assertEqual(len(clean["required_joins"]), 2)
        self.assertEqual(clean["required_columns"], ["DUE_DT_DMS_KEY", "PAY_DT_DMS_KEY"])

    def test_rejects_same_alias_for_both_roles(self):
        cfg = self._config()
        cfg["to_join"] = dict(cfg["from_join"])
        with self.assertRaises(ValueError):
            compile_metric_builder_config(cfg)

    def test_rejects_missing_join_and_bad_unit_and_bad_missing_to(self):
        with self.assertRaises(ValueError):
            compile_metric_builder_config(self._config(from_join=None))
        with self.assertRaises(ValueError):
            compile_metric_builder_config(self._config(unit="fortnight"))
        with self.assertRaises(ValueError):
            compile_metric_builder_config(self._config(missing_to="whatever"))
        with self.assertRaises(ValueError):
            compile_metric_builder_config(self._config(invalid_keys="0, abc"))

    def test_join_skeleton_excludes_invalid_keys(self):
        compiled = compile_metric_builder_config(self._config())
        metrics = [{"metric_builder_config": compiled.config_json}]
        skeleton = "FROM [EMDW_DMART].[FNN_FCT] fnn"
        joins = _build_row_metric_join_sql(metrics, "azure_sql", skeleton)
        self.assertIn("LEFT  JOIN [DT_DMS] due_dt ON fnn.[DUE_DT_DMS_KEY] = due_dt.[DT_DMS_KEY] AND fnn.[DUE_DT_DMS_KEY] NOT IN (0, 777)", joins)
        self.assertIn("LEFT  JOIN [DT_DMS] pay_dt ON fnn.[PAY_DT_DMS_KEY] = pay_dt.[DT_DMS_KEY] AND fnn.[PAY_DT_DMS_KEY] NOT IN (0, 777)", joins)

    def test_join_skeleton_no_not_in_without_invalid_keys(self):
        compiled = compile_metric_builder_config(self._config(invalid_keys=""))
        metrics = [{"metric_builder_config": compiled.config_json}]
        joins = _build_row_metric_join_sql(metrics, "azure_sql", "FROM [S].[F] f")
        self.assertNotIn("NOT IN", joins)
        self.assertIn("LEFT  JOIN [DT_DMS] due_dt", joins)

    def test_wizard_ui_present_in_template(self):
        template = (ROOT / "admin" / "templates" / "client_metrics.html").read_text(encoding="utf-8")
        self.assertEqual(template.count('value="date_gap"'), 2)  # create + edit forms
        self.assertEqual(template.count('class="metric-builder-date-gap"'), 2)
        for marker in (
            "metric-builder-dg-from", "metric-builder-dg-to", "metric-builder-dg-unit",
            "metric-builder-dg-missing", "metric-builder-dg-invalid", "metric-builder-dg-datecol",
            "metric-builder-dg-aggregation", "loadDateGapRoles",
            'data-builder-mode="date_gap"',
        ):
            self.assertIn(marker, template, marker)


class MetricAiImportTests(unittest.TestCase):
    def test_ai_import_endpoint_exists(self):
        routes_src = (ROOT / "admin" / "routes.py").read_text(encoding="utf-8")
        self.assertIn("metrics/api/ai-import", routes_src)
        self.assertIn("_METRIC_AI_IMPORT_SYSTEM", routes_src)
        # The endpoint must compile the AI config server-side before returning it.
        self.assertIn("compile_metric_builder_config(config, db_type=db_type)", routes_src)

    def test_metric_save_routes_compile_with_db_type(self):
        routes_src = (ROOT / "admin" / "routes.py").read_text(encoding="utf-8")
        self.assertEqual(
            routes_src.count("compile_metric_builder_config(metric_builder_config, db_type=db_type)"),
            2,  # create + update
        )

    def test_ai_import_ui_present_in_template(self):
        template = (ROOT / "admin" / "templates" / "client_metrics.html").read_text(encoding="utf-8")
        self.assertIn("metric-ai-import-text", template)
        self.assertIn("aiImportMetric", template)
        self.assertIn("metrics/api/ai-import", template)

    def test_date_role_endpoint_returns_date_column(self):
        routes_src = (ROOT / "admin" / "routes.py").read_text(encoding="utf-8")
        self.assertIn("_dimension_date_column", routes_src)
        self.assertIn('"date_column": date_col_cache[dim_table]', routes_src)


if __name__ == "__main__":
    unittest.main()
