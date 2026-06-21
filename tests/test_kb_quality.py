import json
import tempfile
import unittest
from pathlib import Path

from core.kb_quality import (
    evaluate_kb_quality,
    load_kb_quality_report,
    write_kb_quality_report,
)
from core.llm import build_kb_query_prompt
from core.semantic_model import build_semantic_model


class KBQualityTests(unittest.TestCase):
    def test_unresolved_fact_contract_is_flagged_for_review(self):
        model = {
            "tables": [{
                "table": "SALES_FCT",
                "qualified_name": "ERP.SALES_FCT",
                "type": "fact",
                "grain": "needs_admin_context",
                "fields": [{"column": "AMT", "confidence": 45, "status": "generated"}],
                "measures": [],
                "dimensions": [],
                "date_roles": [],
            }],
            "relationships": [],
        }

        report = evaluate_kb_quality(model)
        codes = {issue["code"] for issue in report["issues"]}

        self.assertEqual(report["status"], "needs_review")
        self.assertIn("grain_needs_review", codes)
        self.assertIn("fact_without_measure", codes)
        self.assertIn("low_confidence_field", codes)

    def test_healthy_model_with_contract_docs_is_ready(self):
        model = {
            "tables": [{
                "table": "SALES_FCT",
                "qualified_name": "ERP.SALES_FCT",
                "type": "fact",
                "grain": "one row per invoice line",
                "grain_status": "approved",
                "fields": [{"column": "AMT", "confidence": 100, "status": "approved"}],
                "measures": [{"column": "AMT", "status": "approved"}],
                "dimensions": [],
                "date_roles": [{"fact_column": "IVC_DT_KEY", "status": "approved"}],
            }],
            "relationships": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "SALES_FCT_kb.md").write_text("# sales", encoding="utf-8")
            Path(tmp, "SALES_FCT_queries.md").write_text("Q: total sales", encoding="utf-8")

            report = write_kb_quality_report(model, tmp)
            persisted = load_kb_quality_report(tmp)

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["score"], 100)
        self.assertEqual(persisted["score"], 100)

    def test_semantic_model_emits_reviewable_grain_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "_schema.json").write_text(json.dumps({
                "ERP.SALES_FCT": {"columns": [
                    {"name": "ORD_LIN_NUM", "type": "int"},
                    {"name": "NET_AMT", "type": "decimal"},
                ]},
            }), encoding="utf-8")

            model = build_semantic_model(tmp)

        table = model["tables"][0]
        self.assertEqual(table["grain"], "one row per transaction line")
        self.assertEqual(table["grain_status"], "generated")
        self.assertEqual(table["grain_confidence"], 80)

    def test_cross_table_prompt_includes_exact_related_columns(self):
        prompt = build_kb_query_prompt(
            "SALES_FCT",
            "# ERP.SALES_FCT\nColumns: WHS_KEY, NET_AMT",
            "Sales analytics",
            related_tables="ERP.SALES_FCT.WHS_KEY = ERP.WHS_DMS.WHS_KEY",
            related_table_columns="- ERP.WHS_DMS: WHS_KEY, WHS_DSC",
        )

        self.assertIn("Exact Columns on Related Tables", prompt)
        self.assertIn("ERP.WHS_DMS: WHS_KEY, WHS_DSC", prompt)
        self.assertIn("Never invent columns such as MONTH", prompt)


if __name__ == "__main__":
    unittest.main()
