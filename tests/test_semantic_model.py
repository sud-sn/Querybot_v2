import json
import tempfile
import unittest
from pathlib import Path

from core.semantic_model import (
    MODEL_JSON,
    MODEL_YAML,
    build_semantic_model,
    build_runtime_semantic_context,
    build_runtime_semantic_plan,
    load_semantic_model,
    patch_field_approval,
    write_semantic_model,
)
from core.validator import validate_sql_detailed


def _tmp_dir():
    root = Path("C:/tmp")
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=str(root))


def _write_schema(path: Path) -> None:
    schema = {
        "CHATBOTDB.PROFITABILITY.CUS_ORD_IVC_FCT": {
            "database": "CHATBOTDB",
            "schema": "PROFITABILITY",
            "table": "CUS_ORD_IVC_FCT",
            "columns": [
                {"name": "CUS_ORD_IVC_FCT_KEY", "type": "bigint"},
                {"name": "WHS_DMS_KEY", "type": "bigint"},
                {"name": "CUS_IVC_DT_DMS_KEY", "type": "bigint"},
                {"name": "SOP_CUS_IVC_LIN_AMT", "type": "decimal"},
                {"name": "DEL_IVC_REC_IND", "type": "int"},
            ],
        },
        "CHATBOTDB.PROFITABILITY.WHS_DMS": {
            "database": "CHATBOTDB",
            "schema": "PROFITABILITY",
            "table": "WHS_DMS",
            "columns": [
                {"name": "WHS_DMS_KEY", "type": "bigint"},
                {"name": "WHS_CD", "type": "nvarchar"},
                {"name": "WHS_DSC", "type": "nvarchar"},
            ],
        },
        "CHATBOTDB.PROFITABILITY.DT_DMS": {
            "database": "CHATBOTDB",
            "schema": "PROFITABILITY",
            "table": "DT_DMS",
            "columns": [
                {"name": "DT_DMS_KEY", "type": "bigint"},
                {"name": "CAL_DT", "type": "date"},
                {"name": "YEAR", "type": "int"},
                {"name": "MONTH", "type": "int"},
            ],
        },
    }
    (path / "_schema.json").write_text(json.dumps(schema), encoding="utf-8")


KNOWN_TABLES = {
    "PROFITABILITY.CUS_ORD_IVC_FCT",
    "PROFITABILITY.WHS_DMS",
    "PROFITABILITY.DT_DMS",
}


TABLE_COLUMNS = {
    "PROFITABILITY.CUS_ORD_IVC_FCT": {
        "CUS_ORD_IVC_FCT_KEY": "bigint",
        "WHS_DMS_KEY": "bigint",
        "CUS_IVC_DT_DMS_KEY": "bigint",
        "SOP_CUS_IVC_LIN_AMT": "decimal",
        "DEL_IVC_REC_IND": "int",
    },
    "PROFITABILITY.WHS_DMS": {
        "WHS_DMS_KEY": "bigint",
        "WHS_CD": "nvarchar",
        "WHS_DSC": "nvarchar",
    },
    "PROFITABILITY.DT_DMS": {
        "DT_DMS_KEY": "bigint",
        "CAL_DT": "date",
        "YEAR": "int",
        "MONTH": "int",
    },
}


class SemanticModelTests(unittest.TestCase):
    def test_build_model_resolves_display_fields_and_date_roles(self):
        with _tmp_dir() as tmp:
            schema_dir = Path(tmp)
            _write_schema(schema_dir)

            model = build_semantic_model(str(schema_dir), business_desc="Profitability data", account_id="Demo")

        fact = next(t for t in model["tables"] if t["table"] == "CUS_ORD_IVC_FCT")
        warehouse_dim = next(d for d in fact["dimensions"] if d["source_key"] == "WHS_DMS_KEY")
        self.assertEqual(warehouse_dim["display_table"], "PROFITABILITY.WHS_DMS")
        self.assertEqual(warehouse_dim["display_column"], "WHS_DSC")
        self.assertEqual(warehouse_dim["code_column"], "WHS_CD")

        rel = next(r for r in model["relationships"] if r["business_role"] == "warehouse")
        self.assertEqual(rel["join_type"], "LEFT")
        self.assertEqual(rel["display_column"], "WHS_DSC")
        self.assertEqual(rel["conditions"][0], {"from_column": "WHS_DMS_KEY", "to_column": "WHS_DMS_KEY"})

        date_role = next(r for r in model["date_roles"] if r["business_role"] == "invoice_date")
        self.assertEqual(date_role["fact_column"], "CUS_IVC_DT_DMS_KEY")
        self.assertEqual(date_role["dimension_table"], "PROFITABILITY.DT_DMS")
        self.assertEqual(date_role["dimension_key"], "DT_DMS_KEY")

    def test_write_model_creates_json_and_yaml_artifacts(self):
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)

            write_semantic_model(
                schema_dir=str(schema_dir),
                kb_dir=str(kb_dir),
                business_desc="Profitability data",
                account_id="Demo",
            )

            self.assertTrue((kb_dir / MODEL_JSON).exists())
            self.assertTrue((kb_dir / MODEL_YAML).exists())
            loaded = load_semantic_model(str(kb_dir))
            self.assertEqual(loaded["account_id"], "Demo")
            self.assertEqual(len(loaded["tables"]), 3)
            self.assertIn("relationships:", (kb_dir / MODEL_YAML).read_text(encoding="utf-8"))

    def test_patch_field_approval_updates_structured_model(self):
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            changed = patch_field_approval(
                kb_dir=str(kb_dir),
                table_fqn="CHATBOTDB.PROFITABILITY.WHS_DMS",
                table_name="WHS_DMS",
                schema_name="PROFITABILITY",
                column_name="WHS_DSC",
                approved_meaning="Warehouse name used for business-facing warehouse labels",
                approved_use_case="Use whenever users ask for warehouse.",
            )

            self.assertTrue(changed)
            model = load_semantic_model(str(kb_dir))
            whs = next(t for t in model["tables"] if t["table"] == "WHS_DMS")
            field = next(f for f in whs["fields"] if f["column"] == "WHS_DSC")
            self.assertEqual(field["status"], "approved")
            self.assertEqual(field["confidence"], 100)
            self.assertIn("Warehouse name used", field["approved_meaning"])
            dim = next(d for d in whs["dimensions"] if d["display_column"] == "WHS_DSC")
            self.assertEqual(dim["status"], "approved")

    def test_runtime_context_includes_display_and_date_guidance(self):
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            context = build_runtime_semantic_context(
                str(kb_dir),
                question="show invoice revenue by warehouses by invoice month",
                selected_schema="PROFITABILITY",
            )

            self.assertIn("STRUCTURED SEMANTIC MODEL CONTEXT", context)
            self.assertIn("WHS_DSC", context)
            self.assertIn("WHS_DMS", context)
            self.assertIn("CUS_IVC_DT_DMS_KEY", context)
            self.assertIn("DT_DMS", context)

    def test_runtime_context_respects_schema_scope(self):
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            context = build_runtime_semantic_context(
                str(kb_dir),
                question="show revenue by warehouses",
                selected_schema="PHARMACY",
            )

            self.assertEqual(context, "")

    def test_runtime_context_missing_model_returns_empty(self):
        with _tmp_dir() as tmp:
            context = build_runtime_semantic_context(
                str(Path(tmp) / "missing_kb"),
                question="show revenue by warehouse",
            )
            self.assertEqual(context, "")

    def test_runtime_plan_requires_warehouse_display_field(self):
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            plan = build_runtime_semantic_plan(
                str(kb_dir),
                question="show revenue by warehouses",
                selected_schema="PROFITABILITY",
            )

            self.assertTrue(plan["enabled"])
            self.assertTrue(any(f["column"] == "WHS_DSC" for f in plan["fields"]))
            self.assertTrue(any(
                any(left == "WHS_DMS_KEY" and right == "WHS_DMS_KEY" for left, right in join.get("conditions", []))
                for join in plan["joins"]
            ))

    def test_runtime_plan_validator_rejects_raw_key_when_display_is_required(self):
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))
            plan = build_runtime_semantic_plan(
                str(kb_dir),
                question="show revenue by warehouses",
                selected_schema="PROFITABILITY",
            )

            sql = (
                "SELECT c.WHS_DMS_KEY AS Warehouse, "
                "SUM(c.SOP_CUS_IVC_LIN_AMT) AS Revenue "
                "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] c "
                "GROUP BY c.WHS_DMS_KEY"
            )
            result = validate_sql_detailed(
                sql,
                KNOWN_TABLES,
                "azure_sql",
                None,
                TABLE_COLUMNS,
                {"semantic_plan": plan},
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.code, "field_plan_mismatch")
            self.assertIn("WHS_DSC", result.reason)

    def test_runtime_plan_allows_key_when_user_asks_for_key(self):
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            plan = build_runtime_semantic_plan(
                str(kb_dir),
                question="show revenue by warehouse key",
                selected_schema="PROFITABILITY",
            )

            self.assertFalse(plan["enabled"])


if __name__ == "__main__":
    unittest.main()
