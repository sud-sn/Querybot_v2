import json
import os
import tempfile
import unittest
from pathlib import Path

from core.date_roles import detect_date_role, find_date_dimension_key, is_date_dimension_table
from core.graph_resolver import resolve_for_question
from core.schema import build_entity_graph_from_schema
from core.semantic_model import build_semantic_model, patch_date_role


class DateRoleDetectionTests(unittest.TestCase):
    def test_detects_common_fact_date_roles(self):
        cases = {
            "CUS_IVC_DT_DMS_KEY": "Invoice Date",
            "CUS_ORD_DT_DMS_KEY": "Order Date",
            "RQD_CUS_DLV_DT_DMS_KEY": "Requested Delivery Date",
            "CFM_CUS_DLV_DT_DMS_KEY": "Confirmed Delivery Date",
            "PLD_CUS_DLV_DT_DMS_KEY": "Planned Delivery Date",
            "PCH_RCT_DT_DMS_KEY": "Receipt Date",
            "CCL_CUS_ORD_DT_DMS_KEY": "Cancelled Order Date",
            "PCH_VLD_DLV_DT_DMS_KEY": "Valid Delivery Date",
            "CUR_CST_DT_DMS_KEY": "Current Cost Date",
            "PRE_CST_DT_DMS_KEY": "Previous Cost Date",
            "PCH_ORD_LIN_CRN_DT_DMS_KEY": "Order Line Creation Date",
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

    def test_detects_plain_warehouse_date_id_and_dynamic_roles(self):
        cases = {
            "WRITTEN_DATE_ID": ("written_date", "Written Date"),
            "ORDER_DATE_ID": ("order_date", "Order Date"),
            "THERAPY_START_DATE_ID": ("therapy_start_date", "Therapy Start Date"),
            "THERAPY_END_DATE_ID": ("therapy_end_date", "Therapy End Date"),
            "BOOKED_DATE_ID": ("booked_date", "Booked Date"),
            "SCHEDULED_FILL_DATE_ID": ("scheduled_fill_date", "Scheduled Fill Date"),
            "DISPENSE_DATE_ID": ("dispense_date", "Dispense Date"),
            "PICKUP_DATE_ID": ("pickup_date", "Pickup Date"),
            "REVERSAL_DATE_ID": ("reversal_date", "Reversal Date"),
            "SUBMIT_DATE_ID": ("submit_date", "Submit Date"),
            "ADJUDICATION_DATE_ID": ("adjudication_date", "Adjudication Date"),
            "PAID_DATE_ID": ("paid_date", "Paid Date"),
            "DENIAL_DATE_ID": ("denial_date", "Denial Date"),
            "INVOICE_DATE_ID": ("invoice_date", "Invoice Date"),
            "DUE_DATE_ID": ("due_date", "Due Date"),
            "PAYMENT_DATE_ID": ("payment_date", "Payment Date"),
            "POST_DATE_ID": ("post_date", "Post Date"),
            "SNAPSHOT_DATE_ID": ("snapshot_date", "Snapshot Date"),
            "EXPIRY_DATE_ID": ("expiry_date", "Expiry Date"),
            "LAST_RECEIPT_DATE_ID": ("last_receipt_date", "Last Receipt Date"),
            "PO_DATE_ID": ("purchase_order_date", "Purchase Order Date"),
            "EXPECTED_DELIVERY_DATE_ID": ("expected_delivery_date", "Expected Delivery Date"),
            "RECEIPT_DATE_ID": ("receipt_date", "Receipt Date"),
            "DISPENSE_DATE_KEY": ("dispense_date", "Dispense Date"),
            "EXPIRY_DT_ID": ("expiry_date", "Expiry Date"),
        }
        for column, expected in cases.items():
            with self.subTest(column=column):
                role = detect_date_role(column)
                self.assertIsNotNone(role)
                self.assertEqual((role.key, role.label), expected)

    def test_fact_date_columns_do_not_make_fact_a_date_dimension(self):
        cols = [
            {"name": "RX_ORDER_ID", "type": "bigint"},
            {"name": "ORDER_DATE_ID", "type": "int"},
            {"name": "CREATED_AT_UTC", "type": "datetime2"},
        ]
        self.assertFalse(is_date_dimension_table("F_RX_ORDER", cols))

    def test_fk_first_discovery_supports_pharma_lab_naming(self):
        schema = {
            "PHARMA_LAB.F_RX_FILL": {
                "schema": "PHARMA_LAB",
                "table": "F_RX_FILL",
                "columns": [
                    {"name": "RX_FILL_ID", "type": "bigint"},
                    {"name": "BOOKED_DATE_ID", "type": "int"},
                    {"name": "DISPENSE_DATE_ID", "type": "int"},
                    {"name": "NET_REVENUE_AMT", "type": "decimal"},
                ],
            },
            "PHARMA_LAB.D_DATE": {
                "schema": "PHARMA_LAB",
                "table": "D_DATE",
                "columns": [
                    {"name": "DATE_ID", "type": "int"},
                    {"name": "CALENDAR_DATE", "type": "date"},
                ],
            },
            "__db_fk_constraints__": [
                {
                    "parent_schema": "PHARMA_LAB", "parent_table": "F_RX_FILL",
                    "parent_col": "BOOKED_DATE_ID", "ref_schema": "PHARMA_LAB",
                    "ref_table": "D_DATE", "ref_col": "DATE_ID",
                },
                {
                    "parent_schema": "PHARMA_LAB", "parent_table": "F_RX_FILL",
                    "parent_col": "DISPENSE_DATE_ID", "ref_schema": "PHARMA_LAB",
                    "ref_table": "D_DATE", "ref_col": "DATE_ID",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "_schema.json").write_text(json.dumps(schema), encoding="utf-8")
            model = build_semantic_model(tmp)

        table = next(t for t in model["tables"] if t["table"] == "F_RX_FILL")
        self.assertEqual(table["type"], "fact")
        roles = {role["fact_column"]: role for role in model["date_roles"]}
        self.assertEqual(set(roles), {"BOOKED_DATE_ID", "DISPENSE_DATE_ID"})
        self.assertEqual(roles["BOOKED_DATE_ID"]["business_role"], "booked_date")
        self.assertEqual(roles["DISPENSE_DATE_ID"]["business_role"], "dispense_date")
        self.assertEqual(roles["DISPENSE_DATE_ID"]["date_value_column"], "CALENDAR_DATE")
        self.assertEqual(roles["DISPENSE_DATE_ID"]["confidence"], 99)

    def test_native_date_type_is_detected_without_date_naming_or_dimension(self):
        schema = {
            "OPS.ACTIVITY_LOG": {
                "schema": "OPS",
                "table": "ACTIVITY_LOG",
                "columns": [
                    {"name": "EVENT_ID", "type": "bigint"},
                    {"name": "BUSINESS_MOMENT", "type": "datetime2"},
                    {"name": "AMOUNT", "type": "decimal"},
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "_schema.json").write_text(json.dumps(schema), encoding="utf-8")
            model = build_semantic_model(tmp)

        role = next(role for role in model["date_roles"] if role["fact_column"] == "BUSINESS_MOMENT")
        self.assertEqual(role["date_key_type"], "timestamp")
        self.assertEqual(role["date_value_column"], "BUSINESS_MOMENT")
        self.assertEqual(role["dimension_table"], "")
        self.assertEqual(role["confidence"], 98)
        self.assertFalse(any(rel.get("from_column") == "BUSINESS_MOMENT" for rel in model["relationships"]))

    def test_declared_fk_detects_arbitrary_surrogate_column_name(self):
        schema = {
            "OPS.F_SALES": {
                "schema": "OPS",
                "table": "F_SALES",
                "columns": [
                    {"name": "SALE_ID", "type": "bigint"},
                    {"name": "BUSINESS_DAY_REF", "type": "int"},
                ],
            },
            "OPS.D_DATE": {
                "schema": "OPS",
                "table": "D_DATE",
                "columns": [
                    {"name": "DATE_ID", "type": "int"},
                    {"name": "CALENDAR_DATE", "type": "date"},
                ],
            },
            "__db_fk_constraints__": [{
                "parent_schema": "OPS", "parent_table": "F_SALES",
                "parent_col": "BUSINESS_DAY_REF", "ref_schema": "OPS",
                "ref_table": "D_DATE", "ref_col": "DATE_ID",
            }],
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "_schema.json").write_text(json.dumps(schema), encoding="utf-8")
            model = build_semantic_model(tmp)

        role = next(role for role in model["date_roles"] if role["fact_column"] == "BUSINESS_DAY_REF")
        self.assertEqual(role["date_key_type"], "surrogate_fk")
        self.assertEqual(role["dimension_table"], "OPS.D_DATE")
        self.assertEqual(role["dimension_key"], "DATE_ID")
        self.assertEqual(role["date_value_column"], "CALENDAR_DATE")
        self.assertEqual(role["confidence"], 99)

    def test_manual_mapping_accepts_direct_and_arbitrary_surrogate_columns(self):
        schema = {
            "OPS.F_EVENT": {
                "schema": "OPS",
                "table": "F_EVENT",
                "columns": [
                    {"name": "EVENT_ID", "type": "bigint"},
                    {"name": "LEGACY_PERIOD_REF", "type": "int"},
                    {"name": "RAW_OCCURRED_ON", "type": "varchar"},
                ],
            },
            "OPS.D_DATE": {
                "schema": "OPS",
                "table": "D_DATE",
                "columns": [
                    {"name": "DATE_ID", "type": "int"},
                    {"name": "CALENDAR_DATE", "type": "date"},
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "_schema.json").write_text(json.dumps(schema), encoding="utf-8")
            model = build_semantic_model(tmp)
            Path(tmp, "_semantic_model.json").write_text(json.dumps(model), encoding="utf-8")

            self.assertTrue(patch_date_role(
                kb_dir=tmp,
                fact_table="OPS.F_EVENT",
                fact_column="RAW_OCCURRED_ON",
                business_role="service_date",
                date_key_type="native_date",
                create_if_missing=True,
            ))
            self.assertTrue(patch_date_role(
                kb_dir=tmp,
                fact_table="OPS.F_EVENT",
                fact_column="LEGACY_PERIOD_REF",
                dimension_table="OPS.D_DATE",
                dimension_key="DATE_ID",
                date_value_column="CALENDAR_DATE",
                business_role="accounting_period",
                date_key_type="surrogate_fk",
                create_if_missing=True,
            ))
            updated = json.loads(Path(tmp, "_semantic_model.json").read_text(encoding="utf-8"))

        roles = {role["fact_column"]: role for role in updated["date_roles"]}
        self.assertEqual(roles["RAW_OCCURRED_ON"]["date_key_type"], "native_date")
        self.assertEqual(roles["RAW_OCCURRED_ON"]["dimension_table"], "")
        self.assertEqual(roles["RAW_OCCURRED_ON"]["date_value_column"], "RAW_OCCURRED_ON")
        self.assertEqual(roles["LEGACY_PERIOD_REF"]["dimension_table"], "OPS.D_DATE")
        self.assertEqual(roles["LEGACY_PERIOD_REF"]["date_value_column"], "CALENDAR_DATE")


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
