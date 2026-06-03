import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.graph_resolver import detect_entities, resolve_for_question
from core.llm import build_sql_system_prompt
from core.query_semantics import analyze_query_intent
from core.semantic_planner import build_semantic_field_plan
from core.validator import validate_sql, validate_sql_detailed
from core.answer_confidence import build_answer_confidence
from core.answer_rca import build_business_rca, extract_sql_tables
from core.query_router import should_route_to_result_cache, build_duckdb_system_prompt
from core.response_builder import build_assistant_response, detect_null_metric_issue


KNOWN = {
    "CHATBOT_DB.PROFITABILITY.CUS_ORD_IVC_FCT",
    "PROFITABILITY.CUS_ORD_IVC_FCT",
    "CUS_ORD_IVC_FCT",
    "CHATBOT_DB.PROFITABILITY.ITM_BAL_PRD_FCT",
    "PROFITABILITY.ITM_BAL_PRD_FCT",
    "ITM_BAL_PRD_FCT",
    "CHATBOT_DB.PROFITABILITY.FIFO_BI_SAL_MGP_EXT",
    "PROFITABILITY.FIFO_BI_SAL_MGP_EXT",
    "FIFO_BI_SAL_MGP_EXT",
    "CHATBOT_DB.PROFITABILITY.FIFOBISALMGPEXT",
    "PROFITABILITY.FIFOBISALMGPEXT",
    "FIFOBISALMGPEXT",
    "CHATBOT_DB.PROFITABILITY.OOLINE",
    "PROFITABILITY.OOLINE",
    "OOLINE",
    "CHATBOT_DB.PROFITABILITY.DIM_DIVISION",
    "PROFITABILITY.DIM_DIVISION",
    "DIM_DIVISION",
    "CHATBOT_DB.PROFITABILITY.WHS_DMS",
    "PROFITABILITY.WHS_DMS",
    "WHS_DMS",
    "CHATBOT_DB.PHARMACY.DIMPATIENT",
    "PHARMACY.DIMPATIENT",
    "DIMPATIENT",
}

COLUMNS = {
    "PROFITABILITY.CUS_ORD_IVC_FCT": {
        "ITM_GRP_DMS_KEY": "int",
        "CUS_IVC_DT_DMS_KEY": "int",
        "CUS_IVC_LIN_AMT": "decimal",
        "CUS_ORD_NUM": "varchar",
        "CUS_ORD_LIN_NUM": "int",
        "CUS_ORD_LIN_SFX": "int",
        "WHS_DMS_KEY": "int",
        "CUS_DMS_KEY": "int",
        "SOP_CUS_IVC_LIN_AMT": "decimal",
        "SOP_CUS_IVC_LIN_CST_AMT": "decimal",
        "SOP_CUS_LIN_GRS_PFT_AMT": "decimal",
    },
    "CUS_ORD_IVC_FCT": {
        "ITM_GRP_DMS_KEY": "int",
        "CUS_IVC_DT_DMS_KEY": "int",
        "CUS_IVC_LIN_AMT": "decimal",
        "CUS_ORD_NUM": "varchar",
        "CUS_ORD_LIN_NUM": "int",
        "CUS_ORD_LIN_SFX": "int",
        "WHS_DMS_KEY": "int",
        "CUS_DMS_KEY": "int",
        "SOP_CUS_IVC_LIN_AMT": "decimal",
        "SOP_CUS_IVC_LIN_CST_AMT": "decimal",
        "SOP_CUS_LIN_GRS_PFT_AMT": "decimal",
    },
    "PROFITABILITY.WHS_DMS": {
        "WHS_DMS_KEY": "int",
        "WHS_CD": "varchar",
        "WHS_DSC": "varchar",
    },
    "WHS_DMS": {
        "WHS_DMS_KEY": "int",
        "WHS_CD": "varchar",
        "WHS_DSC": "varchar",
    },
    "PROFITABILITY.ITM_BAL_PRD_FCT": {
        "NUM_OF_RCT": "int",
        "ITM_DMS_KEY": "int",
    },
    "ITM_BAL_PRD_FCT": {
        "NUM_OF_RCT": "int",
        "ITM_DMS_KEY": "int",
    },
    "PROFITABILITY.FIFO_BI_SAL_MGP_EXT": {
        "PCLA": "decimal",
        "FL_DT_TS": "datetime",
        "ORNO": "varchar",
    },
    "FIFO_BI_SAL_MGP_EXT": {
        "PCLA": "decimal",
        "FL_DT_TS": "datetime",
        "ORNO": "varchar",
    },
    "PROFITABILITY.FIFOBISALMGPEXT": {
        "PCLA": "decimal",
        "ORNO": "varchar",
        "PONR": "int",
        "POSX": "int",
    },
    "FIFOBISALMGPEXT": {
        "PCLA": "decimal",
        "ORNO": "varchar",
        "PONR": "int",
        "POSX": "int",
    },
    "PROFITABILITY.OOLINE": {
        "ORNO": "varchar",
        "PONR": "int",
        "POSX": "int",
        "DIVI": "varchar",
    },
    "OOLINE": {
        "ORNO": "varchar",
        "PONR": "int",
        "POSX": "int",
        "DIVI": "varchar",
    },
    "PROFITABILITY.DIM_DIVISION": {
        "DIVI": "varchar",
        "DIVISION_NAME": "varchar",
    },
    "DIM_DIVISION": {
        "DIVI": "varchar",
        "DIVISION_NAME": "varchar",
    },
    "CHATBOT_DB.PHARMACY.DIMPATIENT": {
        "AGE": "int",
        "PATIENT_ID": "varchar",
    },
    "PHARMACY.DIMPATIENT": {
        "AGE": "int",
        "PATIENT_ID": "varchar",
    },
}


class StrictColumnValidationTests(unittest.TestCase):
    def test_rejects_missing_underscore_column(self):
        ok, msg, code = validate_sql(
            "SELECT cus.ITMGRPDMS_KEY FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] cus",
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
        )
        self.assertFalse(ok)
        self.assertEqual(code, "unknown_column")
        self.assertIn("ITM_GRP_DMS_KEY", msg)

    def test_rejects_numofrct_and_suggests_num_of_rct(self):
        result = validate_sql_detailed(
            "SELECT itm.NUMOFRCT FROM [PROFITABILITY].[ITM_BAL_PRD_FCT] itm",
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "unknown_column")
        self.assertIn("NUM_OF_RCT", result.errors[0]["suggestions"])

    def test_wrong_table_column_points_to_candidate_table(self):
        result = validate_sql_detailed(
            "SELECT itm.DIVI, SUM(itm.NUM_OF_RCT) FROM [PROFITABILITY].[ITM_BAL_PRD_FCT] itm GROUP BY itm.DIVI",
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "unknown_column")
        self.assertIn("Exact column exists on", result.reason)
        self.assertTrue(any("DIM_DIVISION" in t for t in result.errors[0]["candidate_tables"]))

    def test_rejects_yoy_hallucinated_year_column(self):
        ok, _, code = validate_sql(
            "SELECT CAST(YEA4 AS INT) AS Year, SUM(PCLA) FROM [PROFITABILITY].[FIFO_BI_SAL_MGP_EXT] GROUP BY CAST(YEA4 AS INT)",
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
        )
        self.assertFalse(ok)
        self.assertEqual(code, "unknown_column")

    def test_rejects_format_on_numeric_date_key(self):
        ok, msg, code = validate_sql(
            "SELECT FORMAT(cus.CUS_IVC_DT_DMS_KEY, 'yyyy-MM') AS PERIOD FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] cus",
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
        )
        self.assertFalse(ok)
        self.assertEqual(code, "date_key_format")
        self.assertIn("Convert YYYYMMDD", msg)

    def test_rejects_filtered_sum_without_null_diagnostics(self):
        result = validate_sql_detailed(
            "SELECT SUM(CUS_IVC_LIN_AMT) AS Revenue "
            "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] "
            "WHERE CUS_DMS_KEY = 1055930",
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "null_aggregate_diagnostic")
        self.assertTrue(result.errors[0]["requires_matched_count"])
        self.assertTrue(result.errors[0]["requires_non_null_count"])
        self.assertTrue(result.errors[0]["requires_null_safe_sum"])

    def test_accepts_filtered_sum_with_null_diagnostics(self):
        result = validate_sql_detailed(
            "SELECT COUNT_BIG(*) AS [MatchedRows], "
            "COUNT(CUS_IVC_LIN_AMT) AS [NonNullRevenueRows], "
            "COALESCE(SUM(CUS_IVC_LIN_AMT), 0) AS [Revenue] "
            "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] "
            "WHERE CUS_DMS_KEY = 1055930",
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
        )
        self.assertTrue(result.ok, result.reason)

    def test_rejects_revenue_query_that_ignores_approved_formula_metric(self):
        metric = {
            "name": "Revenue",
            "synonyms": "total revenue,sales revenue",
            "formula_type": "expression",
            "sql_template": "SUM(SOP_CUS_IVC_LIN_AMT)",
            "required_columns": "SOP_CUS_IVC_LIN_AMT",
        }
        result = validate_sql_detailed(
            "SELECT COUNT_BIG(*) AS [MatchedRows], "
            "COUNT(CUS_IVC_LIN_AMT) AS [NonNullRevenueRows], "
            "COALESCE(SUM(CUS_IVC_LIN_AMT), 0) AS [Revenue] "
            "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] "
            "WHERE CUS_DMS_KEY = 1035573",
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
            {"question": "what is the revenue for the customer 1035573", "metric_formulas": [metric]},
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "metric_formula_mismatch")
        self.assertIn("SOP_CUS_IVC_LIN_AMT", result.reason)

    def test_accepts_revenue_query_that_uses_approved_formula_metric(self):
        metric = {
            "name": "Revenue",
            "synonyms": "total revenue,sales revenue",
            "formula_type": "expression",
            "sql_template": "SUM(SOP_CUS_IVC_LIN_AMT)",
            "required_columns": "SOP_CUS_IVC_LIN_AMT",
        }
        result = validate_sql_detailed(
            "SELECT COUNT_BIG(*) AS [MatchedRows], "
            "COUNT(SOP_CUS_IVC_LIN_AMT) AS [NonNullRevenueRows], "
            "COALESCE(SUM(SOP_CUS_IVC_LIN_AMT), 0) AS [Revenue] "
            "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] "
            "WHERE CUS_DMS_KEY = 1035573",
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
            {"question": "what is the revenue for the customer 1035573", "metric_formulas": [metric]},
        )
        self.assertTrue(result.ok, result.reason)

    def test_rejects_missing_fifo_single_table_null_filter(self):
        sql = (
            "SELECT DISTINCT fif.ORNO AS OrderNumber, fif.POSX AS OrderLineNumber, 0 AS PCLA "
            "FROM [PROFITABILITY].[FIFOBISALMGPEXT] fif "
            "WHERE fif.PCLA IS NULL"
        )
        ok, msg, code = validate_sql(
            sql,
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
            {"intent": {"wants_missing_records": True}},
        )
        self.assertFalse(ok)
        self.assertEqual(code, "anti_join_shape")
        self.assertIn("source table", msg)

    def test_accepts_missing_fifo_left_join_shape(self):
        sql = (
            "SELECT oo.ORNO AS OrderNumber, oo.PONR AS OrderLineNumber, "
            "COALESCE(fif.PCLA, 0) AS PCLA "
            "FROM [PROFITABILITY].[OOLINE] oo "
            "LEFT JOIN [PROFITABILITY].[FIFOBISALMGPEXT] fif "
            "ON oo.ORNO = fif.ORNO AND oo.PONR = fif.PONR AND oo.POSX = fif.POSX "
            "WHERE fif.ORNO IS NULL"
        )
        ok, msg, code = validate_sql(
            sql,
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
            {"intent": {"wants_missing_records": True}},
        )
        self.assertTrue(ok, msg)
        self.assertEqual(code, "ok")

    def test_semantic_plan_maps_division_question_to_correct_tables(self):
        plan = build_semantic_field_plan(
            "For each division, what percentage of total invoice line amount comes from each item group?",
            COLUMNS,
        )
        self.assertTrue(plan["enabled"])
        fields = {(f["term"], f["column"], f["table"]) for f in plan["fields"]}
        self.assertTrue(any(col == "DIVI" and table.endswith("OOLINE") for _, col, table in fields))
        self.assertTrue(any(col == "ITM_GRP_DMS_KEY" and table.endswith("CUS_ORD_IVC_FCT") for _, col, table in fields))
        self.assertTrue(any(col == "CUS_IVC_LIN_AMT" and table.endswith("CUS_ORD_IVC_FCT") for _, col, table in fields))
        self.assertTrue(plan["joins"])
        self.assertFalse(any(col == "AGE" for _, col, _ in fields))

    def test_semantic_plan_respects_selected_schema(self):
        plan = build_semantic_field_plan(
            "For each division, what percentage of total invoice line amount comes from each item group?",
            COLUMNS,
            selected_schema="PROFITABILITY",
        )
        self.assertTrue(plan["enabled"])
        self.assertTrue(all(".PHARMACY." not in f["table"] for f in plan["fields"]))
        self.assertFalse(any(f["column"] == "AGE" for f in plan["fields"]))

    def test_semantic_plan_expands_allowed_fqn_scope(self):
        plan = build_semantic_field_plan(
            "For each division, what percentage of total invoice line amount comes from each item group?",
            COLUMNS,
            allowed_tables={
                "CHATBOT_DB.PROFITABILITY.CUS_ORD_IVC_FCT",
                "CHATBOT_DB.PROFITABILITY.OOLINE",
            },
            selected_schema="PROFITABILITY",
        )
        self.assertTrue(plan["enabled"])
        self.assertTrue(any(f["column"] == "DIVI" and f["table"].endswith("OOLINE") for f in plan["fields"]))
        self.assertTrue(any(f["column"] == "CUS_IVC_LIN_AMT" and f["table"].endswith("CUS_ORD_IVC_FCT") for f in plan["fields"]))

    def test_semantic_plan_validator_rejects_ignored_field_source(self):
        plan = build_semantic_field_plan(
            "For each division, what percentage of total invoice line amount comes from each item group?",
            COLUMNS,
        )
        sql = (
            "SELECT o.DIVI, SUM(c.CUS_IVC_LIN_AMT) AS TotalInvoiceLineAmount "
            "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] c "
            "JOIN [PROFITABILITY].[OOLINE] o "
            "ON c.CUS_ORD_NUM = o.ORNO AND c.CUS_ORD_LIN_NUM = o.PONR AND c.CUS_ORD_LIN_SFX = o.POSX "
            "GROUP BY o.DIVI"
        )
        result = validate_sql_detailed(
            sql,
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
            {"semantic_plan": plan},
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "field_plan_mismatch")
        self.assertIn("ITM_GRP_DMS_KEY", result.reason)

    def test_semantic_plan_validator_accepts_division_item_group_invoice_share_shape(self):
        plan = build_semantic_field_plan(
            "For each division, what percentage of total invoice line amount comes from each item group?",
            COLUMNS,
        )
        sql = (
            "SELECT o.DIVI AS Division, c.ITM_GRP_DMS_KEY AS ItemGroup, "
            "SUM(c.CUS_IVC_LIN_AMT) AS TotalInvoiceLineAmount, "
            "SUM(c.CUS_IVC_LIN_AMT) * 100.0 / NULLIF(SUM(SUM(c.CUS_IVC_LIN_AMT)) "
            "OVER (PARTITION BY o.DIVI), 0) AS PercentageOfDivisionTotal "
            "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] c "
            "JOIN [PROFITABILITY].[OOLINE] o "
            "ON c.CUS_ORD_NUM = o.ORNO AND c.CUS_ORD_LIN_NUM = o.PONR AND c.CUS_ORD_LIN_SFX = o.POSX "
            "GROUP BY o.DIVI, c.ITM_GRP_DMS_KEY"
        )
        result = validate_sql_detailed(
            sql,
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
            {"semantic_plan": plan},
        )
        self.assertTrue(result.ok, result.reason)

    def test_semantic_plan_prefers_warehouse_description_over_key(self):
        plan = build_semantic_field_plan(
            "Which warehouses have high invoice revenue but low gross profit percentage?",
            COLUMNS,
        )
        self.assertTrue(plan["enabled"])
        fields = {(f["term"], f["column"], f["table"], f.get("role")) for f in plan["fields"]}
        self.assertTrue(any(col == "WHS_DSC" and table.endswith("WHS_DMS") and role == "display_dimension"
                            for _, col, table, role in fields))
        self.assertFalse(any(col == "WHS_DMS_KEY" and role == "dimension" for _, col, _, role in fields))
        self.assertTrue(any(
            any(left == "WHS_DMS_KEY" and right == "WHS_DMS_KEY" for left, right in edge.get("conditions", []))
            for edge in plan["joins"]
        ))

    def test_semantic_plan_allows_warehouse_key_when_user_asks_for_key(self):
        plan = build_semantic_field_plan(
            "Show invoice revenue by warehouse key",
            COLUMNS,
        )
        self.assertTrue(plan["enabled"])
        self.assertTrue(any(f["column"] == "WHS_DMS_KEY" for f in plan["fields"]))
        self.assertFalse(any(f["column"] == "WHS_DSC" for f in plan["fields"]))

    def test_semantic_plan_validator_rejects_warehouse_key_as_display_label(self):
        plan = build_semantic_field_plan(
            "Which warehouses have high invoice revenue but low gross profit percentage?",
            COLUMNS,
        )
        sql = (
            "SELECT c.WHS_DMS_KEY AS Warehouse, "
            "SUM(c.SOP_CUS_IVC_LIN_AMT) AS TotalRevenue, "
            "SUM(c.SOP_CUS_LIN_GRS_PFT_AMT) AS GrossProfit "
            "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] c "
            "GROUP BY c.WHS_DMS_KEY"
        )
        result = validate_sql_detailed(
            sql,
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
            {"semantic_plan": plan},
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "field_plan_mismatch")
        self.assertIn("WHS_DSC", result.reason)

    def test_semantic_plan_validator_accepts_warehouse_description_shape(self):
        plan = build_semantic_field_plan(
            "Which warehouses have high invoice revenue but low gross profit percentage?",
            COLUMNS,
        )
        sql = (
            "SELECT w.WHS_DSC AS Warehouse, "
            "SUM(c.SOP_CUS_IVC_LIN_AMT) AS TotalRevenue, "
            "SUM(c.SOP_CUS_LIN_GRS_PFT_AMT) AS GrossProfit "
            "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] c "
            "JOIN [PROFITABILITY].[WHS_DMS] w ON c.WHS_DMS_KEY = w.WHS_DMS_KEY "
            "GROUP BY w.WHS_DSC"
        )
        result = validate_sql_detailed(
            sql,
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
            {"semantic_plan": plan},
        )
        self.assertTrue(result.ok, result.reason)


class IntentAndGraphReliabilityTests(unittest.TestCase):
    def test_missing_record_phrases_are_detected(self):
        for question in (
            "customers with no orders",
            "show rows with no matching invoice",
            "items without receipts",
            "products never sold",
        ):
            self.assertTrue(analyze_query_intent(question)["wants_missing_records"], question)

    def _graph(self):
        return {
            "entities": [
                {"entity_name": "Order", "table_name": "FACT_ORDER", "schema_name": "dbo", "entity_type": "fact", "display_name": "Orders"},
                {"entity_name": "Product", "table_name": "DIM_PRODUCT", "schema_name": "dbo", "entity_type": "dimension", "display_name": "Product"},
                {"entity_name": "Customer", "table_name": "DIM_CUSTOMER", "schema_name": "dbo", "entity_type": "dimension", "display_name": "Customer"},
            ],
            "relationships": [
                {"from_entity": "Order", "to_entity": "Product", "from_column": "ProductID", "to_column": "ProductID", "join_type": "INNER"},
                {"from_entity": "Order", "to_entity": "Customer", "from_column": "CustomerID", "to_column": "CustomerID", "join_type": "INNER"},
            ],
            "properties": [
                {"entity_name": "Product", "column_name": "ProductName", "display_name": "Item Description", "synonyms": "item, items, products"},
            ],
        }

    def test_plural_and_property_matching_detects_product(self):
        found = detect_entities("compare sales by products and item description", self._graph())
        self.assertIn("Product", found)

    def test_metric_formula_table_forces_fact_entity(self):
        found = detect_entities(
            "compare gross margin year over year by product",
            self._graph(),
            required_tables={"FACT_ORDER"},
        )
        self.assertIn("Order", found)
        self.assertIn("Product", found)

    def test_antijoin_resolver_uses_left_join(self):
        result = resolve_for_question(
            "customers with no orders",
            "acct",
            "azure_sql",
            graph=self._graph(),
            intent={"wants_missing_records": True},
        )
        self.assertTrue(result["enabled"])
        self.assertTrue(result["anti_join"])
        self.assertIn("LEFT", result["join_skeleton"])

    def test_graph_prompt_allows_skeleton_inside_cte(self):
        prompt = build_sql_system_prompt(
            "azure_sql",
            "KB",
            graph_context={
                "enabled": True,
                "join_skeleton": "FROM [dbo].[FACT_ORDER] ord INNER JOIN [dbo].[DIM_PRODUCT] pro ON ord.[ProductID] = pro.[ProductID]",
                "detected": ["Order", "Product"],
            },
        )
        self.assertIn("inside the base CTE", prompt)

    def test_antijoin_prompt_keeps_left_join(self):
        prompt = build_sql_system_prompt(
            "azure_sql",
            "KB",
            graph_context={
                "enabled": True,
                "anti_join": True,
                "join_skeleton": "FROM [dbo].[DIM_CUSTOMER] cus LEFT JOIN [dbo].[FACT_ORDER] ord ON cus.[CustomerID] = ord.[CustomerID]",
                "detected": ["Customer", "Order"],
            },
        )
        self.assertIn("ANTI-JOIN GRAPH MODE", prompt)
        self.assertIn("Do not convert these joins back to INNER JOIN", prompt)


class DiagnosticRenderingReliabilityTests(unittest.TestCase):
    def test_zero_row_message_fences_sql_with_underscored_columns(self):
        from main import _build_zero_row_message

        sql = (
            "SELECT c.CUS_ORD_NUM, c.CUS_ORD_LIN_NUM "
            "FROM [profitability].[CUS_ORD_IVC_FCT] c "
            "JOIN [profitability].[OOLINE] o ON c.CUS_ORD_NUM = o.ORNO"
        )
        message = _build_zero_row_message("test", sql, {}, "ok", 0)
        self.assertIn("```sql", message)
        self.assertIn("c.CUS_ORD_NUM", message)
        self.assertIn("c.CUS_ORD_LIN_NUM", message)

    def test_chat_formatter_preserves_sql_code_blocks_before_markdown(self):
        template = (ROOT / "portal" / "templates" / "portal_chat.html").read_text(encoding="utf-8")
        self.assertIn("const codeBlocks = [];", template)
        self.assertIn("return `@@CODEBLOCK${idx}@@`;", template)
        self.assertNotIn("h = h.replace(/_([^_\\n]+)_/g", template)


class BusinessConfidenceRcaTests(unittest.TestCase):
    def test_zero_row_empty_table_lowers_confidence(self):
        confidence = build_answer_confidence(
            validation_code="ok",
            row_count=0,
            retry_count=0,
            has_semantic_plan=True,
            has_graph_context=True,
            tables_used=["PROFITABILITY.CUS_ORD_IVC_FCT", "PROFITABILITY.OOLINE"],
            empty_tables=["PROFITABILITY.OOLINE"],
        )
        self.assertEqual(confidence["level"], "low")
        self.assertTrue(any("no records" in w for w in confidence["warnings"]))

    def test_zero_row_rca_names_empty_source_table(self):
        rca = build_business_rca(
            question="division share by item group",
            row_count=0,
            tables_used=["PROFITABILITY.CUS_ORD_IVC_FCT", "PROFITABILITY.OOLINE"],
            empty_tables=["PROFITABILITY.OOLINE"],
            validation_code="ok",
            retry_count=0,
        )
        self.assertIn("no records", rca["most_likely_reason"])
        self.assertIn("PROFITABILITY.OOLINE", rca["most_likely_reason"])

    def test_successful_query_gets_high_confidence_summary(self):
        confidence = build_answer_confidence(
            validation_code="ok",
            row_count=42,
            retry_count=0,
            has_semantic_plan=True,
            has_graph_context=False,
            tables_used=["PROFITABILITY.CUS_ORD_IVC_FCT"],
        )
        self.assertEqual(confidence["level"], "high")
        self.assertGreaterEqual(confidence["score"], 80)

    def test_null_metric_issue_lowers_confidence(self):
        confidence = build_answer_confidence(
            validation_code="ok",
            row_count=1,
            retry_count=0,
            tables_used=["PROFITABILITY.CUS_ORD_IVC_FCT"],
            null_metric_issue=True,
        )
        self.assertEqual(confidence["level"], "medium")
        self.assertTrue(any("metric values were null" in w for w in confidence["warnings"]))

    def test_null_metric_issue_gets_business_readable_answer(self):
        rows = [{"MatchedRows": 64, "NonNullRevenueRows": 0, "Revenue": 0}]
        self.assertIsNotNone(detect_null_metric_issue(rows))
        payload = build_assistant_response(
            question="what is the revenue for customer 1055930",
            rows=rows,
            sql="SELECT COUNT_BIG(*) AS [MatchedRows], COUNT(CUS_IVC_LIN_AMT) AS [NonNullRevenueRows], COALESCE(SUM(CUS_IVC_LIN_AMT), 0) AS [Revenue] FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] WHERE CUS_DMS_KEY = 1055930",
            duration_ms=20,
        )
        self.assertIn("all matched values are missing", payload["answer"]["headline"])
        self.assertIn("64 matching records", payload["answer"]["comparison"])
        self.assertIn("64 records matched", payload["insight_summary"])

    def test_extract_sql_tables_preserves_schema_names(self):
        sql = (
            "SELECT o.DIVI, c.ITM_GRP_DMS_KEY "
            "FROM [profitability].[CUS_ORD_IVC_FCT] c "
            "JOIN [profitability].[OOLINE] o ON c.CUS_ORD_NUM = o.ORNO"
        )
        tables = extract_sql_tables(sql, "azure_sql")
        self.assertIn("PROFITABILITY.CUS_ORD_IVC_FCT", tables)
        self.assertIn("PROFITABILITY.OOLINE", tables)


class ResultTransformationRoutingTests(unittest.TestCase):
    def test_main_chat_routes_flag_followup_to_cached_result(self):
        self.assertTrue(
            should_route_to_result_cache(
                "for each unique warehouse id flag them Warehouse A and display the revenue",
                True,
                ["Warehouse", "TotalRevenue"],
            )
        )

    def test_duckdb_prompt_prefers_chr_for_letter_flags(self):
        prompt = build_duckdb_system_prompt(
            [{"name": "Warehouse", "type": "BIGINT"}, {"name": "TotalRevenue", "type": "DOUBLE"}],
        )
        self.assertIn("chr(CAST(64 +", prompt)
        self.assertIn("return the original useful columns PLUS the new computed column", prompt)


if __name__ == "__main__":
    unittest.main()
