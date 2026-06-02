import json
import os
import tempfile
import unittest
from pathlib import Path

from core.date_roles import detect_date_role, find_date_dimension_key, is_date_dimension_table
from core.graph_resolver import resolve_for_question
from core.schema import build_entity_graph_from_schema


class DateRoleDetectionTests(unittest.TestCase):
    def test_detects_common_fact_date_roles(self):
        cases = {
            "CUS_IVC_DT_DMS_KEY": "Invoice Date",
            "CUS_ORD_DT_DMS_KEY": "Order Date",
            "RQD_CUS_DLV_DT_DMS_KEY": "Requested Delivery Date",
            "CFM_CUS_DLV_DT_DMS_KEY": "Confirmed Delivery Date",
            "PLD_CUS_DLV_DT_DMS_KEY": "Planned Delivery Date",
            "PCH_RCT_DT_DMS_KEY": "Receipt Date",
        }
        for column, label in cases.items():
            with self.subTest(column=column):
                role = detect_date_role(column)
                self.assertIsNotNone(role)
                self.assertEqual(role.label, label)

    def test_identifies_date_dimension_key(self):
        cols = [{"name": "DATE_DMS_KEY"}, {"name": "FULL_DATE"}, {"name": "YEAR"}]
        self.assertTrue(is_date_dimension_table("DIM_DATE", cols))
        self.assertEqual(find_date_dimension_key(cols), "DATE_DMS_KEY")


class DateRoleGraphTests(unittest.TestCase):
    def _schema_dir(self) -> str:
        os.makedirs(r"C:\tmp", exist_ok=True)
        tmp = tempfile.mkdtemp(dir=r"C:\tmp")
        schema = {
            "PROFITABILITY.CUS_ORD_IVC_FCT": {
                "columns": [
                    {"name": "CUS_ORD_IVC_FCT_KEY", "type": "bigint"},
                    {"name": "CUS_ORD_DT_DMS_KEY", "type": "int"},
                    {"name": "CUS_IVC_DT_DMS_KEY", "type": "int"},
                    {"name": "RQD_CUS_DLV_DT_DMS_KEY", "type": "int"},
                    {"name": "SOP_CUS_IVC_LIN_AMT", "type": "decimal"},
                ]
            },
            "PROFITABILITY.DIM_DATE": {
                "columns": [
                    {"name": "DATE_DMS_KEY", "type": "int"},
                    {"name": "FULL_DATE", "type": "date"},
                    {"name": "MONTH_NAME", "type": "varchar"},
                ]
            },
        }
        Path(tmp, "_schema.json").write_text(json.dumps(schema), encoding="utf-8")
        return tmp

    def test_resolver_chooses_invoice_date_role_from_question(self):
        graph = {
            "entities": [
                {"entity_name": "Order", "table_name": "CUS_ORD_IVC_FCT", "schema_name": "PROFITABILITY", "entity_type": "fact"},
                {"entity_name": "Invoice Date", "table_name": "DIM_DATE", "schema_name": "PROFITABILITY", "entity_type": "dimension"},
                {"entity_name": "Order Date", "table_name": "DIM_DATE", "schema_name": "PROFITABILITY", "entity_type": "dimension"},
            ],
            "relationships": [
                {"from_entity": "Order", "to_entity": "Invoice Date", "from_column": "CUS_IVC_DT_DMS_KEY", "to_column": "DATE_DMS_KEY", "join_type": "LEFT", "label": "Invoice Date"},
                {"from_entity": "Order", "to_entity": "Order Date", "from_column": "CUS_ORD_DT_DMS_KEY", "to_column": "DATE_DMS_KEY", "join_type": "LEFT", "label": "Order Date"},
            ],
            "properties": [],
        }

        result = resolve_for_question(
            "show revenue by invoice month",
            "acct",
            "azure_sql",
            graph=graph,
        )
        self.assertTrue(result["enabled"], result)
        self.assertIn("CUS_IVC_DT_DMS_KEY", result["join_skeleton"])
        self.assertNotIn("CUS_ORD_DT_DMS_KEY", result["join_skeleton"])

    def test_resolver_chooses_order_date_role_from_question(self):
        graph = {
            "entities": [
                {"entity_name": "Order", "table_name": "CUS_ORD_IVC_FCT", "schema_name": "PROFITABILITY", "entity_type": "fact"},
                {"entity_name": "Invoice Date", "table_name": "DIM_DATE", "schema_name": "PROFITABILITY", "entity_type": "dimension"},
                {"entity_name": "Order Date", "table_name": "DIM_DATE", "schema_name": "PROFITABILITY", "entity_type": "dimension"},
            ],
            "relationships": [
                {"from_entity": "Order", "to_entity": "Invoice Date", "from_column": "CUS_IVC_DT_DMS_KEY", "to_column": "DATE_DMS_KEY", "join_type": "LEFT", "label": "Invoice Date"},
                {"from_entity": "Order", "to_entity": "Order Date", "from_column": "CUS_ORD_DT_DMS_KEY", "to_column": "DATE_DMS_KEY", "join_type": "LEFT", "label": "Order Date"},
            ],
            "properties": [],
        }

        result = resolve_for_question(
            "show revenue by order month",
            "acct",
            "azure_sql",
            graph=graph,
        )
        self.assertTrue(result["enabled"], result)
        self.assertIn("CUS_ORD_DT_DMS_KEY", result["join_skeleton"])
        self.assertNotIn("CUS_IVC_DT_DMS_KEY", result["join_skeleton"])


if __name__ == "__main__":
    unittest.main()
