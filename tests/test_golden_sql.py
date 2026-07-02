"""
Sprint 1 — Golden SQL regression tests.

Each test covers one end-to-end scenario that the semantic model pipeline
must handle correctly.  These are the concrete acceptance criteria for the
semantic model feature — if any of these break, something real regressed.

G1  Correct SQL (display field + JOIN)            → validator passes
G2  Raw key used as label                          → validator rejects, field_plan_mismatch
G3  Approved field survives KB rebuild             → preserve_approvals works (S1-1 core)
G4  Drift report tracks removed approved column    → schema column dropped is detected
G5  Schema scope isolates PROFITABILITY from PHARMACY → no cross-schema leak in plan
G6  Date question context maps to DT_DMS           → date dimension table injected
G7  Repair note names exact display field + join   → build_field_plan_repair_note (S1-3)
"""

import json
import tempfile
import unittest
from pathlib import Path

from core.semantic_model import (
    build_field_plan_repair_note,
    build_runtime_semantic_context,
    build_runtime_semantic_plan,
    build_semantic_model,
    load_semantic_model,
    patch_field_approval,
    preserve_approvals,
    write_semantic_model,
)
from core.validator import validate_sql_detailed


# ── Fixtures ────────────────────────────────────────────────────────────────

def _tmp_dir():
    root = Path("C:/tmp")
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=str(root), prefix="golden_sql_")


def _schema():
    return {
        "CHATBOTDB.PROFITABILITY.CUS_ORD_IVC_FCT": {
            "database": "CHATBOTDB",
            "schema": "PROFITABILITY",
            "table": "CUS_ORD_IVC_FCT",
            "columns": [
                {"name": "CUS_ORD_IVC_FCT_KEY", "type": "bigint"},
                {"name": "WHS_DMS_KEY",          "type": "bigint"},
                {"name": "CUS_IVC_DT_DMS_KEY",   "type": "bigint"},
                {"name": "SOP_CUS_IVC_LIN_AMT",  "type": "decimal"},
            ],
        },
        "CHATBOTDB.PROFITABILITY.WHS_DMS": {
            "database": "CHATBOTDB",
            "schema": "PROFITABILITY",
            "table": "WHS_DMS",
            "columns": [
                {"name": "WHS_DMS_KEY", "type": "bigint"},
                {"name": "WHS_CD",      "type": "nvarchar"},
                {"name": "WHS_DSC",     "type": "nvarchar"},
            ],
        },
        "CHATBOTDB.PROFITABILITY.DT_DMS": {
            "database": "CHATBOTDB",
            "schema": "PROFITABILITY",
            "table": "DT_DMS",
            "columns": [
                {"name": "DT_DMS_KEY", "type": "bigint"},
                {"name": "CAL_DT",     "type": "date"},
                {"name": "YEAR",       "type": "int"},
                {"name": "MONTH",      "type": "int"},
            ],
        },
        # A second schema — must never bleed into PROFITABILITY-scoped plans
        "CHATBOTDB.PHARMACY.RX_FACT": {
            "database": "CHATBOTDB",
            "schema": "PHARMACY",
            "table": "RX_FACT",
            "columns": [
                {"name": "RX_FACT_KEY",  "type": "bigint"},
                {"name": "DRUG_DMS_KEY", "type": "bigint"},
                {"name": "RX_AMT",       "type": "decimal"},
            ],
        },
        "CHATBOTDB.PHARMACY.DRUG_DMS": {
            "database": "CHATBOTDB",
            "schema": "PHARMACY",
            "table": "DRUG_DMS",
            "columns": [
                {"name": "DRUG_DMS_KEY", "type": "bigint"},
                {"name": "DRUG_NM",      "type": "nvarchar"},
            ],
        },
    }


KNOWN_TABLES = {
    "PROFITABILITY.CUS_ORD_IVC_FCT",
    "PROFITABILITY.WHS_DMS",
    "PROFITABILITY.DT_DMS",
}

TABLE_COLUMNS = {
    "PROFITABILITY.CUS_ORD_IVC_FCT": {
        "CUS_ORD_IVC_FCT_KEY": "bigint",
        "WHS_DMS_KEY":         "bigint",
        "CUS_IVC_DT_DMS_KEY":  "bigint",
        "SOP_CUS_IVC_LIN_AMT": "decimal",
    },
    "PROFITABILITY.WHS_DMS": {
        "WHS_DMS_KEY": "bigint",
        "WHS_CD":      "nvarchar",
        "WHS_DSC":     "nvarchar",
    },
    "PROFITABILITY.DT_DMS": {
        "DT_DMS_KEY": "bigint",
        "CAL_DT":     "date",
        "YEAR":       "int",
        "MONTH":      "int",
    },
}


def _setup(tmp_root: Path):
    """Write schema + semantic model into tmp_root.  Returns (schema_dir, kb_dir)."""
    schema_dir = tmp_root / "schema"
    kb_dir     = tmp_root / "kb"
    schema_dir.mkdir(parents=True, exist_ok=True)
    (schema_dir / "_schema.json").write_text(
        json.dumps(_schema()), encoding="utf-8"
    )
    write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))
    return schema_dir, kb_dir


# ── Tests ────────────────────────────────────────────────────────────────────

class GoldenSqlTests(unittest.TestCase):

    # ── G1: Correct SQL (display field + explicit JOIN) passes validation ─────
    def test_g1_correct_warehouse_sql_passes_validation(self):
        with _tmp_dir() as tmp:
            _, kb_dir = _setup(Path(tmp))
            plan = build_runtime_semantic_plan(
                str(kb_dir),
                question="show revenue by warehouse",
                selected_schema="PROFITABILITY",
            )
            sql = (
                "SELECT w.WHS_DSC AS Warehouse, "
                "SUM(f.SOP_CUS_IVC_LIN_AMT) AS Revenue "
                "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] f "
                "LEFT JOIN [PROFITABILITY].[WHS_DMS] w "
                "    ON f.WHS_DMS_KEY = w.WHS_DMS_KEY "
                "GROUP BY w.WHS_DSC "
                "ORDER BY Revenue DESC"
            )
            result = validate_sql_detailed(
                sql, KNOWN_TABLES, "azure_sql", None,
                TABLE_COLUMNS, {"semantic_plan": plan},
            )
            self.assertTrue(result.ok, f"Expected pass but got: {result.reason}")

    # ── G2: SQL using raw key as label → rejected with field_plan_mismatch ────
    def test_g2_raw_key_as_label_fails_validation(self):
        with _tmp_dir() as tmp:
            _, kb_dir = _setup(Path(tmp))
            plan = build_runtime_semantic_plan(
                str(kb_dir),
                question="show revenue by warehouse",
                selected_schema="PROFITABILITY",
            )
            sql = (
                "SELECT f.WHS_DMS_KEY AS Warehouse, "
                "SUM(f.SOP_CUS_IVC_LIN_AMT) AS Revenue "
                "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] f "
                "GROUP BY f.WHS_DMS_KEY"
            )
            result = validate_sql_detailed(
                sql, KNOWN_TABLES, "azure_sql", None,
                TABLE_COLUMNS, {"semantic_plan": plan},
            )
            self.assertFalse(result.ok)
            self.assertEqual(result.code, "field_plan_mismatch")
            self.assertIn("WHS_DSC", result.reason)

    # ── G3: Admin-approved field survives a full KB rebuild ───────────────────
    def test_g3_approved_field_survives_kb_rebuild(self):
        with _tmp_dir() as tmp:
            schema_dir, kb_dir = _setup(Path(tmp))

            # Approve WHS_DSC
            changed = patch_field_approval(
                kb_dir=str(kb_dir),
                table_fqn="CHATBOTDB.PROFITABILITY.WHS_DMS",
                table_name="WHS_DMS",
                schema_name="PROFITABILITY",
                column_name="WHS_DSC",
                approved_meaning="Warehouse name — use whenever user asks for warehouse",
                approved_use_case="Group by warehouse name in all revenue and profitability reports",
            )
            self.assertTrue(changed, "patch_field_approval returned False")

            # Simulate KB rebuild: regenerate model from same schema
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            # Approval must be preserved after rebuild
            model = load_semantic_model(str(kb_dir))
            whs = next(t for t in model["tables"] if t["table"] == "WHS_DMS")
            field = next(f for f in whs["fields"] if f["column"] == "WHS_DSC")

            self.assertEqual(field["status"], "approved",
                             "Status was reset to 'generated' after rebuild — preserve_approvals failed")
            self.assertEqual(field["confidence"], 100,
                             "Confidence was reset to <100 after rebuild")
            self.assertIn("Warehouse name", field.get("approved_meaning", ""),
                          "approved_meaning lost after rebuild")

    # ── G4: Drift report tracks approved column dropped from schema ───────────
    def test_g4_drift_detects_removed_approved_column(self):
        with _tmp_dir() as tmp:
            schema_dir, kb_dir = _setup(Path(tmp))

            # Approve WHS_DSC
            patch_field_approval(
                kb_dir=str(kb_dir),
                table_fqn="CHATBOTDB.PROFITABILITY.WHS_DMS",
                table_name="WHS_DMS",
                schema_name="PROFITABILITY",
                column_name="WHS_DSC",
                approved_meaning="Warehouse display name",
            )

            # Simulate WHS_DSC being dropped from the DB schema
            schema = _schema()
            schema["CHATBOTDB.PROFITABILITY.WHS_DMS"]["columns"] = [
                c for c in schema["CHATBOTDB.PROFITABILITY.WHS_DMS"]["columns"]
                if c["name"] != "WHS_DSC"
            ]
            (schema_dir / "_schema.json").write_text(json.dumps(schema), encoding="utf-8")

            # Run preserve_approvals against the rebuilt model
            old_model = load_semantic_model(str(kb_dir))
            new_model  = build_semantic_model(str(schema_dir))
            _, drift   = preserve_approvals(old_model, new_model)

            self.assertTrue(
                any(r["column"] == "WHS_DSC" for r in drift["removed_approved_fields"]),
                "WHS_DSC not reported in removed_approved_fields drift report",
            )

    # ── G5: PROFITABILITY plan must not contain PHARMACY table entries ────────
    def test_g5_schema_scope_isolates_plan_from_other_schemas(self):
        with _tmp_dir() as tmp:
            _, kb_dir = _setup(Path(tmp))

            # Even if the question mentions "drug" (a PHARMACY concept),
            # the PROFITABILITY schema scope must filter it out entirely.
            plan = build_runtime_semantic_plan(
                str(kb_dir),
                question="show revenue by drug",
                selected_schema="PROFITABILITY",
            )

            for field in plan.get("fields") or []:
                self.assertNotIn(
                    "PHARMACY", (field.get("table") or "").upper(),
                    f"PHARMACY field leaked into PROFITABILITY-scoped plan: {field}",
                )
            for join in plan.get("joins") or []:
                for side in ("from", "to"):
                    self.assertNotIn(
                        "PHARMACY", (join.get(side) or "").upper(),
                        f"PHARMACY join leaked into PROFITABILITY-scoped plan: {join}",
                    )

    # ── G6: Date question context includes DT_DMS dimension mapping ───────────
    def test_g6_date_question_context_maps_to_date_dimension(self):
        with _tmp_dir() as tmp:
            _, kb_dir = _setup(Path(tmp))

            context = build_runtime_semantic_context(
                str(kb_dir),
                question="show invoice revenue by month",
                selected_schema="PROFITABILITY",
            )

            self.assertIn("STRUCTURED SEMANTIC MODEL CONTEXT", context)
            self.assertIn("DT_DMS", context,
                          "Date dimension table DT_DMS not injected into context")
            self.assertIn("CUS_IVC_DT_DMS_KEY", context,
                          "Invoice date FK column not injected into context")

    # ── G7: Repair note names the exact display field and join key ────────────
    def test_g7_repair_note_names_exact_display_field_and_join(self):
        plan = {
            "enabled": True,
            "fields": [{
                "term":               "Warehouse",
                "table":              "PROFITABILITY.WHS_DMS",
                "column":             "WHS_DSC",
                "role":               "display_dimension",
                "display_required":   True,
                "source_table":       "PROFITABILITY.CUS_ORD_IVC_FCT",
                "source_key_column":  "WHS_DMS_KEY",
                "confidence":         88,
                "source":             "semantic_model",
            }],
            "joins": [],
            "required_tables": ["PROFITABILITY.WHS_DMS"],
        }
        note = build_field_plan_repair_note(plan)

        self.assertIn("WHS_DSC", note,
                      "Repair note does not name the required display column WHS_DSC")
        self.assertIn("WHS_DMS_KEY", note,
                      "Repair note does not name the join key WHS_DMS_KEY")
        self.assertIn("WHS_DMS", note,
                      "Repair note does not name the dimension table WHS_DMS")
        self.assertIn("_DMS_KEY", note,
                      "Repair note missing the anti-pattern reminder about _DMS_KEY in SELECT")
        # Repair note must be more specific than the old generic message
        self.assertNotIn(
            "Use the exact table.column pairs and required joins from the Semantic field-source plan",
            note,
            "Old generic repair message still present — specific note not used",
        )


    # ── G8: Date role join enforced when plan is active ───────────────────────
    def test_g8_date_dim_join_is_optional_not_required(self):
        """
        Date-role fields/joins are marked enforcement="optional" in
        build_runtime_semantic_plan (commit 5990ddc) specifically so a valid
        alternative — deriving the period straight from the fact table's own
        YYYYMMDD key via TRY_CONVERT — isn't blocked just because it skips the
        DT_DMS join. Both the joined and the raw-key-converted forms are
        legitimate SQL and must pass. (Supersedes the older, stricter
        behavior this test originally asserted — see core/validator.py's
        missing_plan_fields loop, which now also respects enforcement.)
        """
        with _tmp_dir() as tmp:
            _, kb_dir = _setup(Path(tmp))
            plan = build_runtime_semantic_plan(
                str(kb_dir),
                question="show revenue by invoice month",
                selected_schema="PROFITABILITY",
            )

            # Plan must be enabled and include the date dimension join
            self.assertTrue(plan["enabled"],
                            "Plan must be enabled for date question")
            self.assertTrue(
                any("DT_DMS" in str(j.get("to", "")).upper() for j in plan.get("joins", [])),
                "Plan must offer a DT_DMS join hint",
            )
            date_fields = [f for f in plan.get("fields", []) if f.get("role") == "date_dimension"]
            self.assertTrue(date_fields, "Plan must include a date_dimension field")
            self.assertTrue(
                all(f.get("enforcement") == "optional" for f in date_fields),
                "Date-role fields must be optional, not hard-required",
            )

            # SQL that derives the period from the raw key — valid, no JOIN needed
            sql_no_join = (
                "SELECT FORMAT(TRY_CONVERT(date, CONVERT(varchar(8), f.CUS_IVC_DT_DMS_KEY), 112), 'yyyy-MM') AS Month, "
                "SUM(f.SOP_CUS_IVC_LIN_AMT) AS Revenue "
                "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] f "
                "GROUP BY FORMAT(TRY_CONVERT(date, CONVERT(varchar(8), f.CUS_IVC_DT_DMS_KEY), 112), 'yyyy-MM')"
            )
            result_no_join = validate_sql_detailed(
                sql_no_join, KNOWN_TABLES, "azure_sql", None,
                TABLE_COLUMNS, {"semantic_plan": plan},
            )
            self.assertTrue(result_no_join.ok,
                            f"Raw-key date derivation must pass, not be forced through DT_DMS: {result_no_join.reason}")

            # SQL that joins the date dimension is also valid and still passes
            sql_good = (
                "SELECT d.MONTH AS Month, SUM(f.SOP_CUS_IVC_LIN_AMT) AS Revenue "
                "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] f "
                "LEFT JOIN [PROFITABILITY].[DT_DMS] d ON f.CUS_IVC_DT_DMS_KEY = d.DT_DMS_KEY "
                "GROUP BY d.MONTH "
                "ORDER BY d.MONTH"
            )
            result_good = validate_sql_detailed(
                sql_good, KNOWN_TABLES, "azure_sql", None,
                TABLE_COLUMNS, {"semantic_plan": plan},
            )
            self.assertTrue(result_good.ok,
                            f"SQL with correct date dim JOIN must pass: {result_good.reason}")


if __name__ == "__main__":
    unittest.main()
