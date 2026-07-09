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
    get_model_health,
    load_semantic_model,
    patch_date_role,
    patch_field_approval,
    patch_metric_approval,
    patch_relationship,
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

    # ── S2-1: patch_metric_approval ───────────────────────────────────────────

    def test_approved_field_mapping_blocks_nearby_generated_amount_column(self):
        with _tmp_dir() as tmp:
            kb_dir = Path(tmp)
            model = {
                "tables": [
                    {
                        "schema": "EMDW_DMART",
                        "table": "PCH_ORD_RCT_FCT",
                        "qualified_name": "EMDW_DMART.PCH_ORD_RCT_FCT",
                        "fields": [
                            {
                                "column": "PCH_ORD_LIN_AMT",
                                "role": "measure",
                                "status": "generated",
                                "business_candidates": ["purchase order line amount"],
                                "confidence": 65,
                            },
                            {
                                "column": "PCH_ORD_LIN_CAD_AMT",
                                "role": "measure",
                                "status": "approved",
                                "approved_meaning": "CAD purchase order line amount",
                                "approved_use_case": "Used when a question explicitly refers to purchase order amount",
                                "business_candidates": ["Pch Ord Lin Cad Amt field from the selected table."],
                                "confidence": 100,
                            },
                        ],
                        "dimensions": [],
                        "date_roles": [],
                    }
                ],
                "relationships": [],
                "date_roles": [],
            }
            (kb_dir / MODEL_JSON).write_text(json.dumps(model), encoding="utf-8")

            plan = build_runtime_semantic_plan(
                str(kb_dir),
                question="show total purchase order amount by purchase order date",
                selected_schema="EMDW_DMART",
            )

            self.assertTrue(plan["enabled"])
            self.assertIn(
                "EMDW_DMART.PCH_ORD_RCT_FCT.PCH_ORD_LIN_CAD_AMT",
                [f"{f['table']}.{f['column']}" for f in plan["fields"]],
            )

            wrong_sql = (
                "SELECT SUM(pch.PCH_ORD_LIN_AMT) AS TOTAL_PURCHASE_AMOUNT "
                "FROM [EMDW_DMART].[PCH_ORD_RCT_FCT] pch"
            )
            result = validate_sql_detailed(
                wrong_sql,
                {"EMDW_DMART.PCH_ORD_RCT_FCT"},
                "azure_sql",
                None,
                {
                    "EMDW_DMART.PCH_ORD_RCT_FCT": {
                        "PCH_ORD_LIN_AMT": "decimal",
                        "PCH_ORD_LIN_CAD_AMT": "decimal",
                    }
                },
                {"semantic_plan": plan},
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.code, "field_plan_mismatch")
            self.assertIn("PCH_ORD_LIN_CAD_AMT", result.reason)

    def test_approved_field_plan_keeps_best_duplicate_business_quantity_match(self):
        with _tmp_dir() as tmp:
            kb_dir = Path(tmp)
            model = {
                "tables": [
                    {
                        "schema": "EMDW_DMART",
                        "table": "ITM_BAL_PRD_FCT",
                        "qualified_name": "EMDW_DMART.ITM_BAL_PRD_FCT",
                        "fields": [
                            {
                                "column": "PCH_QTY",
                                "role": "measure",
                                "status": "approved",
                                "approved_meaning": "Total quantity of items purchased",
                                "approved_use_case": "Used when a question refers to purchase quantity",
                                "business_candidates": ["purchase quantity"],
                                "confidence": 100,
                            }
                        ],
                        "dimensions": [],
                        "date_roles": [],
                    },
                    {
                        "schema": "EMDW_DMART",
                        "table": "PCH_ORD_RCT_FCT",
                        "qualified_name": "EMDW_DMART.PCH_ORD_RCT_FCT",
                        "fields": [
                            {
                                "column": "PCH_ORD_AUM_QTY",
                                "role": "measure",
                                "status": "approved",
                                "approved_meaning": "Purchase order quantity",
                                "approved_use_case": "Used when a question explicitly refers to purchase quantity",
                                "business_candidates": ["Purchase quantity", "Number of items purchased"],
                                "confidence": 100,
                            }
                        ],
                        "dimensions": [],
                        "date_roles": [],
                    },
                ],
                "relationships": [],
                "date_roles": [],
            }
            (kb_dir / MODEL_JSON).write_text(json.dumps(model), encoding="utf-8")

            plan = build_runtime_semantic_plan(
                str(kb_dir),
                question="what is the number of purchase order quantity by purchase order date",
                selected_schema="EMDW_DMART",
            )
            fields = [f"{f['table']}.{f['column']}" for f in plan["fields"]]

            self.assertIn("EMDW_DMART.PCH_ORD_RCT_FCT.PCH_ORD_AUM_QTY", fields)
            self.assertNotIn("EMDW_DMART.ITM_BAL_PRD_FCT.PCH_QTY", fields)

            correct_sql = (
                "SELECT SUM(pch.PCH_ORD_AUM_QTY) AS TOTAL_PURCHASE_ORDER_QUANTITY "
                "FROM [EMDW_DMART].[PCH_ORD_RCT_FCT] pch"
            )
            result = validate_sql_detailed(
                correct_sql,
                {"EMDW_DMART.PCH_ORD_RCT_FCT", "EMDW_DMART.ITM_BAL_PRD_FCT"},
                "azure_sql",
                None,
                {
                    "EMDW_DMART.PCH_ORD_RCT_FCT": {"PCH_ORD_AUM_QTY": "decimal"},
                    "EMDW_DMART.ITM_BAL_PRD_FCT": {"PCH_QTY": "decimal"},
                },
                {"semantic_plan": plan},
            )
            self.assertTrue(result.ok, result.reason)

    def test_metric_approval_patches_model_measures(self):
        """patch_metric_approval sets status=approved and expression on matching measure."""
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            changed = patch_metric_approval(
                kb_dir=str(kb_dir),
                table_name="CUS_ORD_IVC_FCT",
                schema_name="PROFITABILITY",
                metric_name="Total Invoice Revenue",
                column_name="SOP_CUS_IVC_LIN_AMT",
                sql_template="SUM(SOP_CUS_IVC_LIN_AMT)",
                is_active=True,
            )
            self.assertTrue(changed, "patch_metric_approval returned False")

            model = load_semantic_model(str(kb_dir))
            fact = next(t for t in model["tables"] if t["table"] == "CUS_ORD_IVC_FCT")
            measure = next(m for m in fact["measures"] if m["column"] == "SOP_CUS_IVC_LIN_AMT")
            self.assertEqual(measure["status"], "approved")
            self.assertEqual(measure["confidence"], 100)
            self.assertEqual(measure["expression"], "SUM(SOP_CUS_IVC_LIN_AMT)")

    def test_metric_deprecation_sets_deprecated_status(self):
        """is_active=False marks the measure as deprecated (S2-3 coverage)."""
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            # Approve first
            patch_metric_approval(
                kb_dir=str(kb_dir),
                table_name="CUS_ORD_IVC_FCT",
                schema_name="PROFITABILITY",
                metric_name="Revenue",
                column_name="SOP_CUS_IVC_LIN_AMT",
                sql_template="SUM(SOP_CUS_IVC_LIN_AMT)",
                is_active=True,
            )
            # Deprecate via is_active=False
            changed = patch_metric_approval(
                kb_dir=str(kb_dir),
                table_name="CUS_ORD_IVC_FCT",
                schema_name="PROFITABILITY",
                metric_name="Revenue",
                column_name="SOP_CUS_IVC_LIN_AMT",
                sql_template="SUM(SOP_CUS_IVC_LIN_AMT)",
                is_active=False,
            )
            self.assertTrue(changed)
            model = load_semantic_model(str(kb_dir))
            fact = next(t for t in model["tables"] if t["table"] == "CUS_ORD_IVC_FCT")
            measure = next(m for m in fact["measures"] if m["column"] == "SOP_CUS_IVC_LIN_AMT")
            self.assertEqual(measure["status"], "deprecated",
                             "Inactive metric must be marked deprecated in the semantic model")

    # ── S4-1: get_model_health ───────────────────────────────────────────────

    def test_get_model_health_returns_correct_counts(self):
        """get_model_health reports correct table/field/measure totals and approval coverage."""
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            health = get_model_health(str(kb_dir))

            self.assertTrue(health.get("has_model"), "has_model must be True")
            self.assertEqual(health["tables"]["total"], 3,
                             "3 tables in the test schema")
            self.assertGreater(health["fields"]["total"], 0, "fields.total must be >0")
            self.assertGreater(health["relationships"]["total"], 0,
                               "relationships.total must be >0")
            # Freshly generated model → nothing approved yet
            self.assertEqual(health["fields"]["approved"], 0,
                             "No fields should be approved on a fresh model")
            self.assertIsInstance(health["approval_coverage"]["pct"], float)
            self.assertIn("table_summaries", health)
            self.assertEqual(len(health["table_summaries"]), 3)

    def test_drift_persisted_in_model_after_rebuild(self):
        """write_semantic_model stores _last_drift so get_model_health can surface it."""
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)

            # First build — no old model, drift is clean
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))
            h1 = get_model_health(str(kb_dir))
            self.assertIn("drift", h1)
            self.assertIn("recorded_at", h1["drift"])
            self.assertTrue(h1["drift"].get("clean"), "First build drift must be clean")

            # Approve a field, then drop it from the schema, then rebuild
            patch_field_approval(
                kb_dir=str(kb_dir),
                table_name="WHS_DMS",
                schema_name="PROFITABILITY",
                column_name="WHS_DSC",
                approved_meaning="Warehouse display name",
            )
            import json as _json
            schema = _json.loads((schema_dir / "_schema.json").read_text(encoding="utf-8"))
            schema["CHATBOTDB.PROFITABILITY.WHS_DMS"]["columns"] = [
                c for c in schema["CHATBOTDB.PROFITABILITY.WHS_DMS"]["columns"]
                if c["name"] != "WHS_DSC"
            ]
            (schema_dir / "_schema.json").write_text(_json.dumps(schema), encoding="utf-8")
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            h2 = get_model_health(str(kb_dir))
            self.assertFalse(h2["drift"].get("clean"),
                             "Drift after dropping approved column must not be clean")
            self.assertTrue(
                any(f["column"] == "WHS_DSC" for f in h2["drift"]["removed_approved_fields"]),
                "WHS_DSC must appear in drift.removed_approved_fields",
            )

    # ── S4-missing-model edge case ────────────────────────────────────────────

    def test_get_model_health_missing_model_returns_has_model_false(self):
        with _tmp_dir() as tmp:
            health = get_model_health(str(Path(tmp) / "nonexistent_kb"))
            self.assertFalse(health.get("has_model"))

    # ── S3-1: patch_date_role ────────────────────────────────────────────────

    def test_date_role_approval_patches_model(self):
        """patch_date_role sets status=approved on both top-level and per-table date_roles."""
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            changed = patch_date_role(
                kb_dir=str(kb_dir),
                fact_table="PROFITABILITY.CUS_ORD_IVC_FCT",
                fact_column="CUS_IVC_DT_DMS_KEY",
                dimension_table="PROFITABILITY.DT_DMS",
                dimension_key="DT_DMS_KEY",
                business_role="invoice_date",
                status="approved",
            )
            self.assertTrue(changed, "patch_date_role returned False")

            model = load_semantic_model(str(kb_dir))
            # Check top-level date_roles
            top_dr = next(
                (r for r in model.get("date_roles", [])
                 if r.get("fact_column") == "CUS_IVC_DT_DMS_KEY"),
                None,
            )
            self.assertIsNotNone(top_dr, "Date role not found in top-level date_roles")
            self.assertEqual(top_dr["status"], "approved")
            self.assertEqual(top_dr["confidence"], 100)
            # Check per-table date_roles
            fact = next(t for t in model["tables"] if t["table"] == "CUS_ORD_IVC_FCT")
            table_dr = next(
                (r for r in fact.get("date_roles", [])
                 if r.get("fact_column") == "CUS_IVC_DT_DMS_KEY"),
                None,
            )
            self.assertIsNotNone(table_dr, "Date role not found in per-table date_roles")
            self.assertEqual(table_dr["status"], "approved")

    # ── S3-2: date role SQL enforcement ──────────────────────────────────────

    def test_date_plan_includes_date_join_for_date_question(self):
        """build_runtime_semantic_plan includes date dimension join when question asks about dates."""
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            plan = build_runtime_semantic_plan(
                str(kb_dir),
                question="show revenue by invoice month",
                selected_schema="PROFITABILITY",
            )

            self.assertTrue(plan["enabled"],
                            "Plan must be enabled for a date question")
            has_date_join = any(
                "DT_DMS" in str(j.get("to", "")).upper()
                for j in plan.get("joins", [])
            )
            self.assertTrue(has_date_join,
                            "Plan must include a join to DT_DMS for a date question")
            has_date_field = any(
                f.get("role") == "date_dimension"
                for f in plan.get("fields", [])
            )
            self.assertTrue(has_date_field,
                            "Plan must include a date_dimension field entry")

    # ── S2-2: patch_relationship ──────────────────────────────────────────────

    def test_relationship_patch_updates_join_type_and_display_column(self):
        """patch_relationship updates join_type, display_column, and status to approved."""
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            # The auto-generated relationship for WHS_DMS should be LEFT.
            # Admin confirms it as INNER and sets the display column.
            changed = patch_relationship(
                kb_dir=str(kb_dir),
                from_table="PROFITABILITY.CUS_ORD_IVC_FCT",
                to_table="PROFITABILITY.WHS_DMS",
                from_column="WHS_DMS_KEY",
                to_column="WHS_DMS_KEY",
                join_type="INNER",
                display_column="WHS_DSC",
                status="approved",
            )
            self.assertTrue(changed, "patch_relationship returned False")

            model = load_semantic_model(str(kb_dir))
            rel = next(
                r for r in model["relationships"]
                if r.get("from_table") == "PROFITABILITY.CUS_ORD_IVC_FCT"
                and r.get("to_table") == "PROFITABILITY.WHS_DMS"
            )
            self.assertEqual(rel["join_type"], "INNER",
                             "join_type not updated by patch_relationship")
            self.assertEqual(rel["status"], "approved")
            self.assertEqual(rel["confidence"], 100)
            self.assertEqual(rel["display_column"], "WHS_DSC",
                             "display_column not updated by patch_relationship")


class ApprovedFieldSupersessionTests(unittest.TestCase):
    """Deterministic field redirection: approving CAD_AMT for 'purchase order
    amount' must actively forbid the old generated LIN_AMT — plan avoid list,
    merge pruning, prompt rendering, validator rejection, deterministic repair,
    and rebuild survival."""

    QUESTION = "show total purchase order amount by purchase order date"

    @staticmethod
    def _model() -> dict:
        return {
            "tables": [
                {
                    "schema": "EMDW_DMART",
                    "table": "PCH_ORD_RCT_FCT",
                    "qualified_name": "EMDW_DMART.PCH_ORD_RCT_FCT",
                    "fields": [
                        {
                            "column": "PCH_ORD_LIN_AMT",
                            "role": "measure",
                            "status": "generated",
                            "business_candidates": ["purchase order line amount"],
                            "confidence": 65,
                        },
                        {
                            "column": "PCH_ORD_LIN_CAD_AMT",
                            "role": "measure",
                            "status": "approved",
                            "approved_meaning": "CAD purchase order line amount",
                            "approved_use_case": "Used when a question explicitly refers to purchase order amount",
                            "business_candidates": ["Pch Ord Lin Cad Amt field from the selected table."],
                            "confidence": 100,
                        },
                    ],
                    "dimensions": [],
                    "date_roles": [],
                }
            ],
            "relationships": [],
            "date_roles": [],
        }

    def _plan(self, kb_dir: Path, question: str | None = None) -> dict:
        (kb_dir / MODEL_JSON).write_text(json.dumps(self._model()), encoding="utf-8")
        return build_runtime_semantic_plan(
            str(kb_dir),
            question=question or self.QUESTION,
            selected_schema="EMDW_DMART",
        )

    def test_avoid_list_contains_superseded_rival(self):
        with _tmp_dir() as tmp:
            plan = self._plan(Path(tmp))
            self.assertTrue(plan["enabled"])
            avoid = plan.get("avoid_columns") or []
            self.assertEqual(
                [(a["column"], a["use_instead_column"]) for a in avoid],
                [("PCH_ORD_LIN_AMT", "PCH_ORD_LIN_CAD_AMT")],
            )
            self.assertEqual(avoid[0]["table"], "EMDW_DMART.PCH_ORD_RCT_FCT")

    def test_question_naming_rival_column_is_not_avoided(self):
        with _tmp_dir() as tmp:
            plan = self._plan(
                Path(tmp),
                question="show total purchase order amount from PCH_ORD_LIN_AMT",
            )
            self.assertEqual(plan.get("avoid_columns") or [], [])

    def test_merge_prunes_llm_planned_rival(self):
        from core.pipeline_context import _merge_semantic_plans
        with _tmp_dir() as tmp:
            model_plan = self._plan(Path(tmp))
            llm_plan = {
                "enabled": True,
                "reason": "llm field plan",
                "joins": [],
                "fields": [{
                    "term": "purchase order amount",
                    "table": "EMDW_DMART.PCH_ORD_RCT_FCT",
                    "column": "PCH_ORD_LIN_AMT",
                    "role": "measure",
                    "enforcement": "required",
                    "source": "semantic_field_plan",
                }],
            }
            merged = _merge_semantic_plans(llm_plan, model_plan)
            cols = [f["column"] for f in merged["fields"]]
            self.assertIn("PCH_ORD_LIN_CAD_AMT", cols)
            self.assertNotIn("PCH_ORD_LIN_AMT", cols)
            self.assertEqual(
                [a["column"] for a in merged["avoid_columns"]],
                ["PCH_ORD_LIN_AMT"],
            )

    def test_validator_rejects_sql_using_avoided_column(self):
        with _tmp_dir() as tmp:
            plan = self._plan(Path(tmp))
            table_columns = {
                "EMDW_DMART.PCH_ORD_RCT_FCT": {
                    "PCH_ORD_LIN_AMT": "decimal",
                    "PCH_ORD_LIN_CAD_AMT": "decimal",
                }
            }
            # Even selecting BOTH columns must fail — the rival silently
            # answers the question with the wrong data.
            both_sql = (
                "SELECT SUM(pch.PCH_ORD_LIN_CAD_AMT) AS CAD_AMT, "
                "SUM(pch.PCH_ORD_LIN_AMT) AS TOTAL_PURCHASE_AMOUNT "
                "FROM [EMDW_DMART].[PCH_ORD_RCT_FCT] pch"
            )
            result = validate_sql_detailed(
                both_sql, {"EMDW_DMART.PCH_ORD_RCT_FCT"}, "azure_sql", None,
                table_columns, {"semantic_plan": plan},
            )
            self.assertFalse(result.ok)
            self.assertEqual(result.code, "field_plan_mismatch")
            self.assertIn("admin-approved source", result.reason)
            self.assertIn("PCH_ORD_LIN_CAD_AMT", result.reason)

            good_sql = (
                "SELECT SUM(pch.PCH_ORD_LIN_CAD_AMT) AS TOTAL_PURCHASE_AMOUNT "
                "FROM [EMDW_DMART].[PCH_ORD_RCT_FCT] pch"
            )
            result_ok = validate_sql_detailed(
                good_sql, {"EMDW_DMART.PCH_ORD_RCT_FCT"}, "azure_sql", None,
                table_columns, {"semantic_plan": plan},
            )
            self.assertTrue(result_ok.ok, result_ok.reason)

    def test_deterministic_repair_swaps_avoided_for_approved(self):
        from core.pipeline_helpers import attempt_field_plan_repair
        with _tmp_dir() as tmp:
            plan = self._plan(Path(tmp))
            table_columns = {
                "EMDW_DMART.PCH_ORD_RCT_FCT": {
                    "PCH_ORD_LIN_AMT": "decimal",
                    "PCH_ORD_LIN_CAD_AMT": "decimal",
                }
            }
            wrong_sql = (
                "SELECT SUM(pch.PCH_ORD_LIN_AMT) AS TOTAL_PURCHASE_AMOUNT "
                "FROM [EMDW_DMART].[PCH_ORD_RCT_FCT] pch"
            )
            repaired = attempt_field_plan_repair(
                wrong_sql,
                "azure_sql",
                {"EMDW_DMART.PCH_ORD_RCT_FCT"},
                None,
                table_columns,
                {"semantic_plan": plan},
            )
            self.assertTrue(repaired, "repair returned empty — expected a deterministic swap")
            self.assertIn("PCH_ORD_LIN_CAD_AMT", repaired)
            self.assertNotIn("PCH_ORD_LIN_AMT", repaired.replace("PCH_ORD_LIN_CAD_AMT", ""))

    def test_prompt_renders_do_not_use_line(self):
        from core.semantic_planner import format_semantic_field_plan
        with _tmp_dir() as tmp:
            plan = self._plan(Path(tmp))
            text = format_semantic_field_plan(plan, "azure_sql")
            self.assertIn("Do NOT use EMDW_DMART.PCH_ORD_RCT_FCT.PCH_ORD_LIN_AMT", text)
            self.assertIn("admin-approved source is EMDW_DMART.PCH_ORD_RCT_FCT.PCH_ORD_LIN_CAD_AMT", text)

    def test_avoid_list_survives_kb_rebuild(self):
        from core.semantic_model import preserve_approvals
        with _tmp_dir() as tmp:
            kb_dir = Path(tmp)
            old_model = self._model()
            # Fresh rebuild regenerates BOTH fields as generated.
            new_model = self._model()
            for field in new_model["tables"][0]["fields"]:
                field["status"] = "generated"
                field.pop("approved_meaning", None)
                field.pop("approved_use_case", None)
            merged, _drift = preserve_approvals(old_model, new_model)
            (kb_dir / MODEL_JSON).write_text(json.dumps(merged), encoding="utf-8")
            plan = build_runtime_semantic_plan(
                str(kb_dir), question=self.QUESTION, selected_schema="EMDW_DMART",
            )
            self.assertIn(
                "EMDW_DMART.PCH_ORD_RCT_FCT.PCH_ORD_LIN_CAD_AMT",
                [f"{f['table']}.{f['column']}" for f in plan["fields"]],
            )
            self.assertEqual(
                [a["column"] for a in plan.get("avoid_columns") or []],
                ["PCH_ORD_LIN_AMT"],
            )


if __name__ == "__main__":
    unittest.main()
