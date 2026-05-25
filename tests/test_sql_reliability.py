import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.graph_resolver import detect_entities, resolve_for_question
from core.llm import build_sql_system_prompt
from core.query_semantics import analyze_query_intent
from core.validator import validate_sql, validate_sql_detailed


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
}

COLUMNS = {
    "PROFITABILITY.CUS_ORD_IVC_FCT": {
        "ITM_GRP_DMS_KEY": "int",
        "CUS_IVC_DT_DMS_KEY": "int",
        "CUS_IVC_LIN_AMT": "decimal",
        "WHS_DMS_KEY": "int",
    },
    "CUS_ORD_IVC_FCT": {
        "ITM_GRP_DMS_KEY": "int",
        "CUS_IVC_DT_DMS_KEY": "int",
        "CUS_IVC_LIN_AMT": "decimal",
        "WHS_DMS_KEY": "int",
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
    },
    "OOLINE": {
        "ORNO": "varchar",
        "PONR": "int",
        "POSX": "int",
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


if __name__ == "__main__":
    unittest.main()
