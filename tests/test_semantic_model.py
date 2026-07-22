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
    find_default_date_roles,
    get_model_health,
    load_semantic_model,
    patch_date_role,
    patch_field_approval,
    patch_metric_approval,
    patch_relationship,
    semantic_model_fingerprint,
    set_default_date_role,
    write_semantic_model,
)
from core.validator import validate_sql_detailed


def _add_raw_date_role(kb_dir: str, fact_table: str, **overrides) -> None:
    """Directly inject a date_role entry into both the top-level and
    per-table lists, bypassing patch_date_role's field-existence validation
    -- used purely to set up multi-role/native-date test fixtures where the
    extra role's column isn't part of the base schema fixture."""
    model = load_semantic_model(kb_dir)
    entry = {
        "name": "Test Date",
        "business_role": "test_date",
        "fact_table": fact_table,
        "fact_column": "TEST_DATE_COL",
        "dimension_table": "",
        "dimension_key": "",
        "date_value_column": "",
        "date_key_type": "surrogate_fk",
        "synonyms": [],
        "status": "approved",
        "confidence": 100,
        "is_default": False,
    }
    entry.update(overrides)
    model.setdefault("date_roles", []).append(dict(entry))
    for table in model.get("tables", []) or []:
        if str(table.get("qualified_name") or table.get("table") or "") == fact_table:
            table.setdefault("date_roles", []).append(dict(entry))
    kb_path = Path(kb_dir)
    (kb_path / MODEL_JSON).write_text(json.dumps(model, indent=2, sort_keys=True), encoding="utf-8")


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

    def test_patch_field_approval_merges_synonyms_into_business_candidates(self):
        # Business terms entered via the portal Suggest Edit box (or the
        # admin Edit Field form) must reach business_candidates — the SAME
        # list _approved_field_match_values scores against — so adding a
        # term actually improves runtime SQL-field matching, not just
        # documentation on the Semantic Layer page.
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            patch_field_approval(
                kb_dir=str(kb_dir),
                table_fqn="CHATBOTDB.PROFITABILITY.WHS_DMS",
                table_name="WHS_DMS",
                schema_name="PROFITABILITY",
                column_name="WHS_DSC",
                approved_meaning="Warehouse name",
                approved_use_case="Use whenever users ask for warehouse.",
                approved_synonyms=["depot", "facility name"],
            )

            model = load_semantic_model(str(kb_dir))
            whs = next(t for t in model["tables"] if t["table"] == "WHS_DMS")
            field = next(f for f in whs["fields"] if f["column"] == "WHS_DSC")
            candidates_lower = {str(c).lower() for c in field["business_candidates"]}
            self.assertIn("depot", candidates_lower)
            self.assertIn("facility name", candidates_lower)

    def test_patch_field_approval_synonym_dedupes_against_existing_candidate(self):
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            patch_field_approval(
                kb_dir=str(kb_dir), table_fqn="CHATBOTDB.PROFITABILITY.WHS_DMS",
                table_name="WHS_DMS", schema_name="PROFITABILITY", column_name="WHS_DSC",
                approved_meaning="Warehouse name", approved_synonyms=["Warehouse name", "depot"],
            )
            model = load_semantic_model(str(kb_dir))
            whs = next(t for t in model["tables"] if t["table"] == "WHS_DMS")
            field = next(f for f in whs["fields"] if f["column"] == "WHS_DSC")
            candidates_lower = [str(c).lower() for c in field["business_candidates"]]
            self.assertEqual(candidates_lower.count("warehouse name"), 1)

    def test_synonym_only_term_makes_field_an_approved_winner(self):
        # End-to-end: a business term with NO overlap with approved_meaning/
        # approved_use_case must still be enough to win the field for a
        # question phrased using only that term.
        with _tmp_dir() as tmp:
            kb_dir = Path(tmp)
            model = {
                "tables": [{
                    "schema": "EMDW_DMART", "table": "PCH_ORD_RCT_FCT",
                    "qualified_name": "EMDW_DMART.PCH_ORD_RCT_FCT",
                    "fields": [{
                        "column": "PCH_ORD_AUM_QTY", "role": "measure", "status": "approved",
                        "approved_meaning": "Purchase order quantity",
                        "approved_use_case": "Used when a question refers to purchase quantity",
                        "business_candidates": ["Purchase order quantity"],
                        "confidence": 100,
                    }],
                    "dimensions": [], "date_roles": [],
                }],
                "relationships": [], "date_roles": [],
            }
            (kb_dir / MODEL_JSON).write_text(json.dumps(model), encoding="utf-8")
            plan = build_runtime_semantic_plan(
                str(kb_dir), question="number of items purchased by warehouse",
                selected_schema="EMDW_DMART",
            )
            self.assertFalse(plan["enabled"], "control: term absent from candidates must not match")

            patch_field_approval(
                kb_dir=str(kb_dir), table_fqn="EMDW_DMART.PCH_ORD_RCT_FCT",
                table_name="PCH_ORD_RCT_FCT", schema_name="EMDW_DMART",
                column_name="PCH_ORD_AUM_QTY",
                approved_meaning="Purchase order quantity",
                approved_use_case="Used when a question refers to purchase quantity",
                approved_synonyms=["number of items purchased"],
            )
            plan2 = build_runtime_semantic_plan(
                str(kb_dir), question="number of items purchased by warehouse",
                selected_schema="EMDW_DMART",
            )
            self.assertTrue(plan2["enabled"])
            self.assertIn(
                "EMDW_DMART.PCH_ORD_RCT_FCT.PCH_ORD_AUM_QTY",
                [f"{f['table']}.{f['column']}" for f in plan2["fields"]],
            )

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

    # ── Default date role ────────────────────────────────────────────────────

    def test_default_date_role_matches_generic_temporal_question(self):
        """A question with temporal intent but no role-name match still
        surfaces an is_default-flagged role -- the fallback for questions
        like 'total revenue by year' that name no specific date concept."""
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            patch_date_role(
                kb_dir=str(kb_dir),
                fact_table="PROFITABILITY.CUS_ORD_IVC_FCT",
                fact_column="CUS_IVC_DT_DMS_KEY",
                dimension_table="PROFITABILITY.DT_DMS",
                dimension_key="DT_DMS_KEY",
                business_role="invoice_date",
                status="approved",
            )
            self.assertTrue(set_default_date_role(
                str(kb_dir), "PROFITABILITY.CUS_ORD_IVC_FCT", "CUS_IVC_DT_DMS_KEY",
            ))

            # "revenue by year" names no date role -- only the default flag
            # can surface it.
            plan = build_runtime_semantic_plan(
                str(kb_dir),
                question="show total revenue by year",
                selected_schema="PROFITABILITY",
            )
            self.assertTrue(plan["enabled"], "Default role must enable the plan")
            self.assertTrue(
                any("DT_DMS" in str(j.get("to", "")).upper() for j in plan.get("joins", [])),
                "Default role must still add its dimension join",
            )

    def test_multiple_date_roles_without_default_stay_disabled_for_generic_question(self):
        """A fact table with two approved date roles and no default set must
        not guess for a generic temporal question -- ambiguity is left to
        the reactive surrogate-date-misuse validator catch instead."""
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            patch_date_role(
                kb_dir=str(kb_dir),
                fact_table="PROFITABILITY.CUS_ORD_IVC_FCT",
                fact_column="CUS_IVC_DT_DMS_KEY",
                dimension_table="PROFITABILITY.DT_DMS",
                dimension_key="DT_DMS_KEY",
                business_role="invoice_date",
                status="approved",
            )
            _add_raw_date_role(
                str(kb_dir), "PROFITABILITY.CUS_ORD_IVC_FCT",
                fact_column="TEST_DATE_COL", business_role="test_date",
                dimension_table="PROFITABILITY.DT_DMS", dimension_key="DT_DMS_KEY",
                date_value_column="CAL_DT", status="approved",
            )

            plan = build_runtime_semantic_plan(
                str(kb_dir),
                question="show total revenue by year",
                selected_schema="PROFITABILITY",
            )
            self.assertFalse(
                any(f.get("role") == "date_dimension" for f in plan.get("fields", [])),
                "Neither ambiguous role should be forced in without a default",
            )

    def test_native_date_role_included_without_dimension_join(self):
        """A role marked native_date has no dimension to join to -- the fact
        column itself is the usable date value. This previously never
        survived build_runtime_semantic_plan's required-dimension gate."""
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            _add_raw_date_role(
                str(kb_dir), "PROFITABILITY.CUS_ORD_IVC_FCT",
                fact_column="NATIVE_ORDER_DATE", name="Order Date",
                business_role="order_date", dimension_table="",
                dimension_key="", date_value_column="",
                date_key_type="native_date", status="approved",
            )

            plan = build_runtime_semantic_plan(
                str(kb_dir),
                question="show revenue by order date",
                selected_schema="PROFITABILITY",
            )
            self.assertTrue(plan["enabled"])
            native_field = next(
                (f for f in plan.get("fields", []) if f.get("column") == "NATIVE_ORDER_DATE"),
                None,
            )
            self.assertIsNotNone(native_field, "Native date role must appear as a field")
            self.assertEqual(native_field["table"], "PROFITABILITY.CUS_ORD_IVC_FCT")
            self.assertFalse(
                any(
                    j.get("conditions") and j["conditions"][0][0] == "NATIVE_ORDER_DATE"
                    for j in plan.get("joins", [])
                ),
                "Native date role must not add a dimension join",
            )
            native_policy = next(
                (p for p in plan.get("date_key_policies", []) if p.get("column") == "NATIVE_ORDER_DATE"),
                None,
            )
            self.assertIsNotNone(native_policy)
            self.assertEqual(native_policy["date_key_type"], "native_date")

    def test_set_default_date_role_clears_siblings_and_persists(self):
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            patch_date_role(
                kb_dir=str(kb_dir),
                fact_table="PROFITABILITY.CUS_ORD_IVC_FCT",
                fact_column="CUS_IVC_DT_DMS_KEY",
                dimension_table="PROFITABILITY.DT_DMS",
                dimension_key="DT_DMS_KEY",
                business_role="invoice_date",
                status="approved",
            )
            _add_raw_date_role(
                str(kb_dir), "PROFITABILITY.CUS_ORD_IVC_FCT",
                fact_column="TEST_DATE_COL", business_role="test_date",
                dimension_table="PROFITABILITY.DT_DMS", dimension_key="DT_DMS_KEY",
                date_value_column="CAL_DT", status="approved",
            )

            self.assertTrue(set_default_date_role(
                str(kb_dir), "PROFITABILITY.CUS_ORD_IVC_FCT", "CUS_IVC_DT_DMS_KEY",
            ))
            model = load_semantic_model(str(kb_dir))
            by_col = {r["fact_column"]: r for r in model["date_roles"]}
            self.assertTrue(by_col["CUS_IVC_DT_DMS_KEY"]["is_default"])
            self.assertFalse(by_col["TEST_DATE_COL"]["is_default"])
            fact = next(t for t in model["tables"] if t["table"] == "CUS_ORD_IVC_FCT")
            table_by_col = {r["fact_column"]: r for r in fact["date_roles"]}
            self.assertTrue(table_by_col["CUS_IVC_DT_DMS_KEY"]["is_default"])
            self.assertFalse(table_by_col["TEST_DATE_COL"]["is_default"])

            # Switching the default clears the previous one.
            self.assertTrue(set_default_date_role(
                str(kb_dir), "PROFITABILITY.CUS_ORD_IVC_FCT", "TEST_DATE_COL",
            ))
            model = load_semantic_model(str(kb_dir))
            by_col = {r["fact_column"]: r for r in model["date_roles"]}
            self.assertFalse(by_col["CUS_IVC_DT_DMS_KEY"]["is_default"])
            self.assertTrue(by_col["TEST_DATE_COL"]["is_default"])

            self.assertFalse(set_default_date_role(
                str(kb_dir), "PROFITABILITY.CUS_ORD_IVC_FCT", "NO_SUCH_COLUMN",
            ))

    def test_find_default_date_roles_prefers_flag_then_sole_role_then_ambiguous(self):
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            patch_date_role(
                kb_dir=str(kb_dir),
                fact_table="PROFITABILITY.CUS_ORD_IVC_FCT",
                fact_column="CUS_IVC_DT_DMS_KEY",
                dimension_table="PROFITABILITY.DT_DMS",
                dimension_key="DT_DMS_KEY",
                business_role="invoice_date",
                status="approved",
            )

            # Exactly one approved role -- it's the implicit default.
            results = find_default_date_roles(str(kb_dir))
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["fact_column"], "CUS_IVC_DT_DMS_KEY")

            # A second approved role with neither flagged -- ambiguous, skip.
            _add_raw_date_role(
                str(kb_dir), "PROFITABILITY.CUS_ORD_IVC_FCT",
                fact_column="TEST_DATE_COL", business_role="test_date",
                dimension_table="PROFITABILITY.DT_DMS", dimension_key="DT_DMS_KEY",
                date_value_column="CAL_DT", status="approved",
            )
            self.assertEqual(find_default_date_roles(str(kb_dir)), [])

            # Flagging one resolves the ambiguity.
            set_default_date_role(str(kb_dir), "PROFITABILITY.CUS_ORD_IVC_FCT", "TEST_DATE_COL")
            results = find_default_date_roles(str(kb_dir))
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["fact_column"], "TEST_DATE_COL")

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


class GenericCommandVerbFalsePositiveTests(unittest.TestCase):
    """
    Confirmed live defect: an approved field's auto-generated use_case text
    ("Used when a question explicitly refers to list, show or filter by
    <column>") gets comma-split by _semantic_business_phrases' "refers to X"
    extraction into standalone one-word phrases — including a bare "list".
    _runtime_match_score's single-word branch then treats "list" as a valid
    business-term match against ANY question phrased as "list the ..."
    (one of the most common ways to phrase a question), silently forcing an
    unrelated field into the required plan and failing validation with
    field_plan_mismatch — while a completely unrelated, correctly-matched
    metric (Customer Discount Amount) executed fine in the same query.
    """

    QUESTION = "list the discount amount by each customers top 10"

    @staticmethod
    def _model() -> dict:
        return {
            "tables": [
                {
                    "schema": "EMDW_DMART",
                    "table": "PC_DVN_DMS",
                    "qualified_name": "EMDW_DMART.PC_DVN_DMS",
                    "fields": [{
                        "column": "PC_DVN_CD",
                        "role": "attribute",
                        "status": "approved",
                        "approved_meaning": "Pc Dvn Cd field from the selected table.",
                        "approved_use_case": "Used when a question explicitly refers to list, show or filter by pc dvn cd.",
                        "business_candidates": ["PC codes", "PC code", "PC division code", "profit Centre code"],
                        "confidence": 100,
                    }],
                    "dimensions": [], "date_roles": [],
                }
            ],
            "relationships": [], "date_roles": [],
        }

    def test_unrelated_field_not_required_for_list_phrased_question(self):
        with _tmp_dir() as tmp:
            kb_dir = Path(tmp)
            (kb_dir / MODEL_JSON).write_text(json.dumps(self._model()), encoding="utf-8")
            plan = build_runtime_semantic_plan(
                str(kb_dir), question=self.QUESTION, selected_schema="EMDW_DMART",
            )
        fields = [f"{f['table']}.{f['column']}" for f in (plan.get("fields") or [])]
        self.assertNotIn("EMDW_DMART.PC_DVN_DMS.PC_DVN_CD", fields)

    def test_field_still_matches_its_own_real_business_terms(self):
        # The fix must not make this field unmatchable outright — a question
        # that actually names one of its real business terms should still
        # pull it in.
        with _tmp_dir() as tmp:
            kb_dir = Path(tmp)
            (kb_dir / MODEL_JSON).write_text(json.dumps(self._model()), encoding="utf-8")
            plan = build_runtime_semantic_plan(
                str(kb_dir), question="show revenue by profit centre code",
                selected_schema="EMDW_DMART",
            )
        fields = [f"{f['table']}.{f['column']}" for f in (plan.get("fields") or [])]
        self.assertIn("EMDW_DMART.PC_DVN_DMS.PC_DVN_CD", fields)

    def test_list_alone_never_scores_as_a_match(self):
        from core.semantic_model import _runtime_match_score, _terms_for_text
        use_case = "Used when a question explicitly refers to list, show or filter by pc dvn cd."
        q_terms = _terms_for_text(self.QUESTION)
        self.assertEqual(_runtime_match_score(q_terms, [use_case, "list"]), 0)

    def test_command_verb_stopwords_present(self):
        from core.semantic_model import _RUNTIME_MATCH_STOPWORDS
        for word in ("list", "filter", "display", "view", "find", "get",
                     "give", "tell", "provide", "retrieve", "pull", "fetch"):
            self.assertIn(word, _RUNTIME_MATCH_STOPWORDS, word)


class SemanticBusinessPhraseExtractionTests(unittest.TestCase):
    """
    _semantic_business_phrases is the general mechanism the PC_DVN_CD defect
    came from — a stopword-list addition for "list" patches that one word,
    but the underlying extraction had three compounding defects that could
    surface a DIFFERENT noise word for a DIFFERENT field tomorrow. These
    tests target the extraction function itself, not a specific word.
    """

    def test_stacked_noise_prefix_fully_reduces(self):
        # Single-pass stripping only removed "a ", leaving the garbled
        # "question explicitly refers to list" as its own scoreable phrase —
        # noise-extraction artifacts riding along as if they were content.
        from core.semantic_model import _semantic_business_phrases
        phrases = _semantic_business_phrases(
            "Used when a question explicitly refers to list, show or filter by pc dvn cd."
        )
        for p in phrases:
            self.assertNotIn("question", p.lower())
            self.assertNotIn("explicitly", p.lower())

    def test_redundant_used_when_pattern_skipped_once_refers_to_matches(self):
        # "refers to" and "used when/for" both fire on the single most common
        # use_case sentence shape ("Used when a question explicitly refers to
        # X") — "used when/for" only ever produces a noisier duplicate of the
        # exact same information in that case, never anything new.
        from core.semantic_model import _semantic_business_phrases
        phrases = _semantic_business_phrases(
            "Used when a question explicitly refers to purchase order amount"
        )
        self.assertEqual(phrases, ["purchase order amount"])

    def test_used_for_pattern_still_fires_without_refers_to(self):
        # Confirms the skip is conditional, not a blanket removal of the
        # "used when/for" pattern — text with no "refers to" anywhere still
        # needs it.
        from core.semantic_model import _semantic_business_phrases
        self.assertEqual(
            _semantic_business_phrases("Used for calculating gross margin"),
            ["calculating gross margin"],
        )

    def test_content_free_fragment_never_returned(self):
        # General backstop: a fragment with zero meaningful (non-stopword)
        # terms after extraction is dropped outright, regardless of which
        # specific word makes it meaningless — not reliant on that word
        # already being enumerated in _RUNTIME_MATCH_STOPWORDS.
        from core.semantic_model import _semantic_business_phrases
        phrases = _semantic_business_phrases("Used when a question explicitly refers to list")
        self.assertEqual(phrases, [])

    def test_no_duplicate_phrases_across_patterns(self):
        from core.semantic_model import _semantic_business_phrases
        phrases = _semantic_business_phrases(
            "Used when a question explicitly refers to list, show or filter by pc dvn cd."
        )
        self.assertEqual(len(phrases), len(set(p.lower() for p in phrases)))

    def test_comma_separated_synonyms_still_both_extracted(self):
        from core.semantic_model import _semantic_business_phrases
        self.assertEqual(
            _semantic_business_phrases("Used when a question explicitly refers to nationality, country"),
            ["nationality", "country"],
        )

    def test_business_terms_colon_pattern_unaffected(self):
        from core.semantic_model import _semantic_business_phrases
        self.assertEqual(
            _semantic_business_phrases("Business terms: gross margin, net margin"),
            ["gross margin", "net margin"],
        )

    def test_empty_and_whitespace_text_return_empty_list(self):
        from core.semantic_model import _semantic_business_phrases
        self.assertEqual(_semantic_business_phrases(""), [])
        self.assertEqual(_semantic_business_phrases("   "), [])


class SemanticModelFingerprintTests(unittest.TestCase):
    """
    semantic_model_fingerprint underpins the Learning Queue staleness
    feature — a governed few-shot example is stamped with this fingerprint
    at approval time and re-checked against the CURRENT one at retrieval
    time, so an example approved under an older semantic model can be
    de-prioritized after a KB rebuild or field remapping changes it.
    """

    def test_empty_kb_dir_returns_empty_string(self):
        self.assertEqual(semantic_model_fingerprint(""), "")

    def test_missing_model_file_returns_empty_string(self):
        with _tmp_dir() as tmp:
            self.assertEqual(semantic_model_fingerprint(tmp), "")

    def test_same_content_same_fingerprint(self):
        with _tmp_dir() as tmp:
            (Path(tmp) / MODEL_JSON).write_text('{"tables": []}', encoding="utf-8")
            fp1 = semantic_model_fingerprint(tmp)
            fp2 = semantic_model_fingerprint(tmp)
        self.assertNotEqual(fp1, "")
        self.assertEqual(fp1, fp2)

    def test_different_content_different_fingerprint(self):
        with _tmp_dir() as tmp1, _tmp_dir() as tmp2:
            (Path(tmp1) / MODEL_JSON).write_text('{"tables": []}', encoding="utf-8")
            (Path(tmp2) / MODEL_JSON).write_text('{"tables": [1]}', encoding="utf-8")
            fp1 = semantic_model_fingerprint(tmp1)
            fp2 = semantic_model_fingerprint(tmp2)
        self.assertNotEqual(fp1, fp2)

    def test_fingerprint_changes_after_a_field_is_approved(self):
        # The exact real-world trigger: approving a field patches model.json
        # in place (patch_field_approval) — the fingerprint must move so a
        # governed example approved before this rewrite is recognized as
        # stale on the next retrieval.
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))
            before = semantic_model_fingerprint(str(kb_dir))

            patch_field_approval(
                kb_dir=str(kb_dir),
                table_fqn="CHATBOTDB.PROFITABILITY.WHS_DMS",
                table_name="WHS_DMS", schema_name="PROFITABILITY",
                column_name="WHS_DSC", approved_meaning="Warehouse name",
            )
            after = semantic_model_fingerprint(str(kb_dir))
        self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
