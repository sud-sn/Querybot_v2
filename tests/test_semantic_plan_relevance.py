import json
import tempfile
import unittest
from pathlib import Path

from core.graph_resolver import detect_entities
from core.pipeline_context import _merge_semantic_plans
from core.semantic_model import MODEL_JSON, build_runtime_semantic_plan
from core.table_coverage import build_required_fqns


def _tmp_dir():
    root = Path("C:/tmp")
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=str(root))


def _write_runtime_model(kb_dir: Path) -> None:
    kb_dir.mkdir(parents=True, exist_ok=True)
    model = {
        "version": 1,
        "tables": [
            {
                "schema": "PROFITABILITY",
                "table": "CUS_ORD_IVC_FCT",
                "qualified_name": "PROFITABILITY.CUS_ORD_IVC_FCT",
                "dimensions": [
                    {
                        "name": "Customer",
                        "source_key": "CUS_DMS_KEY",
                        "display_table": "PROFITABILITY.CUS_DMS",
                        "display_key": "CUS_DMS_KEY",
                        "display_column": "CUS_NM",
                        "approved_meaning": "Customer name for invoice analysis",
                        "confidence": 100,
                    },
                    {
                        "name": "Profit Center",
                        "source_key": "PFT_CTR_DMS_KEY",
                        "display_table": "PROFITABILITY.PFT_CTR_DMS",
                        "display_key": "PFT_CTR_DMS_KEY",
                        "display_column": "PFT_CTR_NM",
                        "approved_meaning": "Profit center name",
                        "confidence": 100,
                    },
                    {
                        "name": "Warehouse",
                        "source_key": "WHS_DMS_KEY",
                        "display_table": "PROFITABILITY.WHS_DMS",
                        "display_key": "WHS_DMS_KEY",
                        "display_column": "WHS_DSC",
                        "approved_meaning": "Warehouse description",
                        "confidence": 100,
                    },
                ],
                "date_roles": [
                    {
                        "name": "Invoice Date",
                        "business_role": "invoice_date",
                        "fact_column": "CUS_IVC_DT_DMS_KEY",
                        "dimension_table": "PROFITABILITY.DT_DMS",
                        "dimension_key": "DT_DMS_KEY",
                        "synonyms": ["invoice month", "invoice year"],
                        "confidence": 100,
                    },
                    {
                        "name": "Order Line Creation Date",
                        "business_role": "order_line_creation_date",
                        "fact_column": "ORD_LIN_CRN_DT_DMS_KEY",
                        "dimension_table": "PROFITABILITY.DT_DMS",
                        "dimension_key": "DT_DMS_KEY",
                        "synonyms": ["line creation date"],
                        "confidence": 90,
                    },
                ],
            }
        ],
    }
    (kb_dir / MODEL_JSON).write_text(json.dumps(model), encoding="utf-8")


class RuntimeSemanticRelevanceTests(unittest.TestCase):
    def _plan(self, question: str) -> dict:
        with _tmp_dir() as tmp:
            kb_dir = Path(tmp) / "kb"
            _write_runtime_model(kb_dir)
            return build_runtime_semantic_plan(
                str(kb_dir),
                question=question,
                selected_schema="PROFITABILITY",
            )

    def test_invoice_amount_does_not_require_any_date_role(self):
        plan = self._plan(
            "For each division, what percentage of total invoice line amount "
            "comes from each item group?"
        )
        self.assertFalse(any(f.get("role") == "date_dimension" for f in plan.get("fields", [])))
        self.assertFalse(any("DT_DMS" in str(j.get("to", "")).upper() for j in plan.get("joins", [])))

    def test_warehouse_profit_question_requires_only_warehouse_display(self):
        plan = self._plan(
            "Which warehouses generate the highest invoice revenue, "
            "and what is their gross profit percentage?"
        )
        columns = {f.get("column") for f in plan.get("fields", [])}
        self.assertEqual(columns, {"WHS_DSC"})

    def test_profit_leakage_question_does_not_require_customer_or_profit_center(self):
        plan = self._plan(
            "Which warehouses have high invoice revenue but low gross profit "
            "percentage, and what is the profit leakage by warehouse?"
        )
        columns = {f.get("column") for f in plan.get("fields", [])}
        self.assertEqual(columns, {"WHS_DSC"})

    def test_explicit_invoice_month_still_requires_invoice_date_join(self):
        plan = self._plan("Show invoice revenue by warehouse and invoice month")
        columns = {f.get("column") for f in plan.get("fields", [])}
        self.assertIn("WHS_DSC", columns)
        self.assertIn("DT_DMS_KEY", columns)
        self.assertTrue(any("DT_DMS" in str(j.get("to", "")).upper() for j in plan.get("joins", [])))


class GraphDateEntityRelevanceTests(unittest.TestCase):
    GRAPH = {
        "entities": [
            {
                "entity_name": "Customer Invoice",
                "display_name": "Customer Invoice",
                "table_name": "CUS_ORD_IVC_FCT",
                "schema_name": "PROFITABILITY",
                "entity_type": "fact",
            },
            {
                "entity_name": "Purchase Receipt",
                "display_name": "Purchase Receipt",
                "table_name": "PCH_ORD_RCT_FCT",
                "schema_name": "PROFITABILITY",
                "entity_type": "fact",
            },
            {
                "entity_name": "Invoice Date",
                "display_name": "Invoice Date",
                "table_name": "DT_DMS",
                "schema_name": "PROFITABILITY",
                "entity_type": "dimension",
            },
            {
                "entity_name": "Order Line Creation Date",
                "display_name": "Order Line Creation Date",
                "table_name": "DT_DMS",
                "schema_name": "PROFITABILITY",
                "entity_type": "dimension",
            },
        ],
        "relationships": [
            {
                "from_entity": "Customer Invoice",
                "to_entity": "Invoice Date",
                "from_column": "CUS_IVC_DT_DMS_KEY",
                "to_column": "DT_DMS_KEY",
                "label": "Invoice Date",
            },
            {
                "from_entity": "Purchase Receipt",
                "to_entity": "Order Line Creation Date",
                "from_column": "ORD_LIN_CRN_DT_DMS_KEY",
                "to_column": "DT_DMS_KEY",
                "label": "Order Line Creation Date",
            },
        ],
        "properties": [],
    }

    def test_invoice_revenue_does_not_select_invoice_date_entity(self):
        found = detect_entities("show total invoice revenue", self.GRAPH)
        self.assertNotIn("Invoice Date", found)

    def test_invoice_line_amount_does_not_select_line_creation_date(self):
        found = detect_entities("show total invoice line amount", self.GRAPH)
        self.assertNotIn("Order Line Creation Date", found)

    def test_invoice_month_selects_invoice_date_entity(self):
        found = detect_entities("show invoice revenue by invoice month", self.GRAPH)
        self.assertIn("Invoice Date", found)


class TableCoverageExpansionTests(unittest.TestCase):
    def test_ambiguous_dimension_does_not_inject_every_connected_fact(self):
        graph = {
            "entities": [
                {
                    "entity_name": "Warehouse",
                    "table_name": "WHS_DMS",
                    "schema_name": "PROFITABILITY",
                    "entity_type": "dimension",
                },
                {
                    "entity_name": "Sales",
                    "table_name": "CUS_ORD_IVC_FCT",
                    "schema_name": "PROFITABILITY",
                    "entity_type": "fact",
                },
                {
                    "entity_name": "Inventory",
                    "table_name": "ITM_BAL_PRD_FCT",
                    "schema_name": "PROFITABILITY",
                    "entity_type": "fact",
                },
            ],
            "relationships": [
                {"from_entity": "Sales", "to_entity": "Warehouse"},
                {"from_entity": "Inventory", "to_entity": "Warehouse"},
            ],
        }
        required = build_required_fqns(
            {"enabled": True, "detected": ["Warehouse"], "anchor": "Warehouse"},
            graph,
        )
        self.assertEqual(required, {"PROFITABILITY.WHS_DMS"})

    def test_unique_connected_fact_is_injected(self):
        graph = {
            "entities": [
                {
                    "entity_name": "Warehouse",
                    "table_name": "WHS_DMS",
                    "schema_name": "PROFITABILITY",
                    "entity_type": "dimension",
                },
                {
                    "entity_name": "Sales",
                    "table_name": "CUS_ORD_IVC_FCT",
                    "schema_name": "PROFITABILITY",
                    "entity_type": "fact",
                },
            ],
            "relationships": [
                {"from_entity": "Sales", "to_entity": "Warehouse"},
            ],
        }
        required = build_required_fqns(
            {"enabled": True, "detected": ["Warehouse"], "anchor": "Warehouse"},
            graph,
        )
        self.assertEqual(
            required,
            {"PROFITABILITY.WHS_DMS", "PROFITABILITY.CUS_ORD_IVC_FCT"},
        )


class SemanticPlanMergeTests(unittest.TestCase):
    def test_merge_deduplicates_two_and_three_part_table_names(self):
        schema_plan = {
            "enabled": True,
            "fields": [
                {
                    "term": "Warehouse",
                    "table": "CHATBOT_DB.PROFITABILITY.WHS_DMS",
                    "column": "WHS_DSC",
                }
            ],
            "joins": [],
        }
        model_plan = {
            "enabled": True,
            "fields": [
                {
                    "term": "Warehouse",
                    "table": "PROFITABILITY.WHS_DMS",
                    "column": "WHS_DSC",
                }
            ],
            "joins": [],
        }
        merged = _merge_semantic_plans(schema_plan, model_plan)
        self.assertEqual(len(merged["fields"]), 1)

    def test_merge_prunes_join_unrelated_to_required_fields(self):
        plan = {
            "enabled": True,
            "fields": [
                {
                    "term": "Warehouse",
                    "table": "PROFITABILITY.WHS_DMS",
                    "column": "WHS_DSC",
                    "source_table": "PROFITABILITY.CUS_ORD_IVC_FCT",
                }
            ],
            "joins": [
                {
                    "from": "PROFITABILITY.CUS_ORD_IVC_FCT",
                    "to": "PROFITABILITY.WHS_DMS",
                    "conditions": [("WHS_DMS_KEY", "WHS_DMS_KEY")],
                },
                {
                    "from": "PROFITABILITY.CUS_ORD_IVC_FCT",
                    "to": "PROFITABILITY.PFT_CTR_DMS",
                    "conditions": [("PFT_CTR_DMS_KEY", "PFT_CTR_DMS_KEY")],
                },
            ],
        }
        merged = _merge_semantic_plans(plan)
        self.assertEqual(len(merged["joins"]), 1)
        self.assertEqual(merged["joins"][0]["to"], "PROFITABILITY.WHS_DMS")

    def test_merge_preserves_complete_multi_hop_path(self):
        plan = {
            "enabled": True,
            "fields": [
                {
                    "term": "Revenue",
                    "table": "PROFITABILITY.SALES_FCT",
                    "column": "REVENUE_AMT",
                },
                {
                    "term": "Region",
                    "table": "PROFITABILITY.REGION_DMS",
                    "column": "REGION_DSC",
                },
            ],
            "joins": [
                {
                    "from": "PROFITABILITY.SALES_FCT",
                    "to": "PROFITABILITY.CUSTOMER_DMS",
                    "conditions": [("CUS_DMS_KEY", "CUS_DMS_KEY")],
                },
                {
                    "from": "PROFITABILITY.CUSTOMER_DMS",
                    "to": "PROFITABILITY.REGION_DMS",
                    "conditions": [("REGION_DMS_KEY", "REGION_DMS_KEY")],
                },
                {
                    "from": "PROFITABILITY.SALES_FCT",
                    "to": "PROFITABILITY.DATE_DMS",
                    "conditions": [("DATE_DMS_KEY", "DATE_DMS_KEY")],
                },
            ],
        }
        merged = _merge_semantic_plans(plan)
        self.assertEqual(
            {(j["from"], j["to"]) for j in merged["joins"]},
            {
                ("PROFITABILITY.SALES_FCT", "PROFITABILITY.CUSTOMER_DMS"),
                ("PROFITABILITY.CUSTOMER_DMS", "PROFITABILITY.REGION_DMS"),
            },
        )

    def test_advisory_field_is_not_made_mandatory(self):
        plan = {
            "enabled": True,
            "fields": [
                {
                    "term": "Warehouse",
                    "table": "PROFITABILITY.WHS_DMS",
                    "column": "WHS_DSC",
                },
                {
                    "term": "Customer",
                    "table": "PROFITABILITY.CUS_DMS",
                    "column": "CUS_NM",
                    "enforcement": "advisory",
                },
            ],
            "joins": [],
        }
        merged = _merge_semantic_plans(plan)
        self.assertEqual({f["column"] for f in merged["fields"]}, {"WHS_DSC"})
        self.assertEqual({f["column"] for f in merged["advisory_fields"]}, {"CUS_NM"})
