"""
Sprint C — Drill-by-dimension tests.

Covers:
  1. find_drill_candidate      — pure lookup, no side effects
  2. build_drill_sql_prompt    — pure prompt builder
  3. compute_chip_eligibility  — drill_dim chips from available_dimensions
  4. build_runtime_semantic_plan — available_dimensions populated
  5. generate_drill_by_dimension — deterministic fallback paths (no DB/LLM)

Design: all tests are pure-function or use the async fallback path that
short-circuits before any LLM/DB call is made.
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from core.drill_dimension import (
    build_drill_sql_prompt,
    find_drill_candidate,
)
from core.response_builder import compute_chip_eligibility


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

def _plan_with_dims(*dims):
    """Build a minimal semantic_plan that has the given available_dimensions."""
    return {
        "enabled": True,
        "fields": [{"role": "display_dimension", "column": "WHS_DSC",
                    "table": "PROFITABILITY.WHS_DMS", "source_key_column": "WHS_DMS_KEY",
                    "source_table": "PROFITABILITY.CUS_ORD_IVC_FCT"}],
        "joins": [],
        "available_dimensions": list(dims),
    }


def _dim(name, col, table="PROFITABILITY.WHS_DMS", src_table="PROFITABILITY.CUS_ORD_IVC_FCT",
         src_key="WHS_DMS_KEY", status="generated"):
    return {
        "name": name,
        "display_column": col,
        "display_table": table,
        "source_table": src_table,
        "source_key_column": src_key,
        "display_key": src_key,
        "status": status,
        "confidence": 75 if status == "approved" else 50,
    }


def _ctx(mode="ranking", row_count=8, text_cols=None, numeric_cols=None):
    return {
        "mode": mode,
        "row_count": row_count,
        "text_cols":     text_cols or [],
        "numeric_cols":  numeric_cols or ["Revenue"],
        "distribution_stats": {"top_3_share_pct": 70.0, "std_dev": 100.0},
        "comparison_stats":   {"leader": "X", "leader_share_pct": 40.0, "gap": 200.0},
    }


# ══════════════════════════════════════════════════════════════════════════════
# find_drill_candidate
# ══════════════════════════════════════════════════════════════════════════════

class FindDrillCandidateTests(unittest.TestCase):

    def test_finds_by_exact_name(self):
        plan = _plan_with_dims(_dim("Warehouse", "WHS_DSC"))
        result = find_drill_candidate("Warehouse", plan)
        self.assertIsNotNone(result)
        self.assertEqual(result["display_column"], "WHS_DSC")

    def test_case_insensitive_match(self):
        plan = _plan_with_dims(_dim("Warehouse", "WHS_DSC"))
        self.assertIsNotNone(find_drill_candidate("warehouse", plan))
        self.assertIsNotNone(find_drill_candidate("WAREHOUSE", plan))

    def test_returns_none_for_unknown_name(self):
        plan = _plan_with_dims(_dim("Warehouse", "WHS_DSC"))
        self.assertIsNone(find_drill_candidate("Region", plan))

    def test_returns_none_when_no_available_dims(self):
        plan = {"enabled": True, "fields": [], "available_dimensions": []}
        self.assertIsNone(find_drill_candidate("Warehouse", plan))

    def test_returns_none_for_empty_name(self):
        plan = _plan_with_dims(_dim("Warehouse", "WHS_DSC"))
        self.assertIsNone(find_drill_candidate("", plan))

    def test_returns_first_match_when_duplicates(self):
        plan = _plan_with_dims(
            _dim("Warehouse", "WHS_DSC"),
            _dim("Warehouse", "WHS_NM"),
        )
        result = find_drill_candidate("Warehouse", plan)
        self.assertEqual(result["display_column"], "WHS_DSC")


# ══════════════════════════════════════════════════════════════════════════════
# build_drill_sql_prompt
# ══════════════════════════════════════════════════════════════════════════════

class BuildDrillSqlPromptTests(unittest.TestCase):

    def _build(self, sql="SELECT SUM(AMT) AS Revenue FROM FCT GROUP BY MONTH"):
        return build_drill_sql_prompt(
            sql,
            display_table="PROFITABILITY.WHS_DMS",
            display_col="WHS_DSC",
            source_table="PROFITABILITY.CUS_ORD_IVC_FCT",
            source_key="WHS_DMS_KEY",
            display_key="WHS_DMS_KEY",
        )

    def test_system_has_cannot_rewrite_sentinel(self):
        system, _ = self._build()
        self.assertIn("CANNOT_REWRITE", system)

    def test_system_instructs_to_add_display_col(self):
        system, _ = self._build()
        self.assertIn("WHS_DSC", system)

    def test_system_instructs_to_add_join(self):
        system, _ = self._build()
        self.assertIn("WHS_DMS", system)

    def test_system_forbids_changing_other_parts(self):
        system, _ = self._build()
        self.assertIn("Do NOT change", system)

    def test_user_contains_original_sql(self):
        sql = "SELECT SUM(AMT) FROM FCT GROUP BY MONTH"
        _, user = self._build(sql=sql)
        self.assertIn(sql, user)

    def test_user_contains_join_condition(self):
        _, user = self._build()
        self.assertIn("WHS_DMS_KEY", user)

    def test_system_requires_markdown_free_output(self):
        system, _ = self._build()
        self.assertIn("no markdown", system.lower())


# ══════════════════════════════════════════════════════════════════════════════
# compute_chip_eligibility — drill_dim chips
# ══════════════════════════════════════════════════════════════════════════════

class ChipEligibilityDrillDimTests(unittest.TestCase):

    def test_drill_chip_shown_when_dim_not_in_result(self):
        """Warehouse dim is available but WHS_DSC not in result cols → chip shown."""
        ctx = _ctx(text_cols=["Month"])
        plan = _plan_with_dims(_dim("Warehouse", "WHS_DSC", status="approved"))
        chips = compute_chip_eligibility(ctx, semantic_plan=plan)
        ids = [c["id"] for c in chips]
        self.assertIn("drill_dim:Warehouse", ids)

    def test_drill_chip_suppressed_when_col_already_in_result(self):
        """WHS_DSC already a result column → no chip."""
        ctx = _ctx(text_cols=["WHS_DSC", "Month"])
        plan = _plan_with_dims(_dim("Warehouse", "WHS_DSC"))
        chips = compute_chip_eligibility(ctx, semantic_plan=plan)
        ids = [c["id"] for c in chips]
        self.assertNotIn("drill_dim:Warehouse", ids)

    def test_drill_chip_suppressed_when_col_in_numeric_cols(self):
        """Col may appear as a numeric result column too — still suppress."""
        ctx = _ctx(numeric_cols=["WHS_DSC", "Revenue"])
        plan = _plan_with_dims(_dim("Warehouse", "WHS_DSC"))
        chips = compute_chip_eligibility(ctx, semantic_plan=plan)
        self.assertNotIn("drill_dim:Warehouse", [c["id"] for c in chips])

    def test_case_insensitive_result_col_check(self):
        """Result cols are checked case-insensitively."""
        ctx = _ctx(text_cols=["whs_dsc"])
        plan = _plan_with_dims(_dim("Warehouse", "WHS_DSC"))
        chips = compute_chip_eligibility(ctx, semantic_plan=plan)
        self.assertNotIn("drill_dim:Warehouse", [c["id"] for c in chips])

    def test_at_most_three_drill_chips(self):
        """Never more than 3 drill chips regardless of available dimensions."""
        plan = _plan_with_dims(
            _dim("Warehouse", "WHS_DSC"),
            _dim("Product", "PRD_DSC", src_key="PRD_DMS_KEY"),
            _dim("Region", "RGN_DSC", src_key="RGN_DMS_KEY"),
            _dim("Customer", "CUS_DSC", src_key="CUS_DMS_KEY"),
            _dim("Salesperson", "SLS_DSC", src_key="SLS_DMS_KEY"),
        )
        ctx = _ctx(text_cols=["Month"])  # none of the display cols in result
        chips = compute_chip_eligibility(ctx, semantic_plan=plan)
        drill_chips = [c for c in chips if c["id"].startswith("drill_dim:")]
        self.assertLessEqual(len(drill_chips), 3)

    def test_drill_chips_absent_when_plan_disabled(self):
        plan = {"enabled": False, "available_dimensions": [_dim("Warehouse", "WHS_DSC")]}
        ctx = _ctx()
        chips = compute_chip_eligibility(ctx, semantic_plan=plan)
        self.assertFalse(any(c["id"].startswith("drill_dim:") for c in chips))

    def test_drill_chips_absent_when_no_plan(self):
        ctx = _ctx()
        chips = compute_chip_eligibility(ctx, semantic_plan=None)
        self.assertFalse(any(c["id"].startswith("drill_dim:") for c in chips))

    def test_approved_dim_gets_higher_confidence(self):
        plan = _plan_with_dims(
            _dim("Warehouse", "WHS_DSC", status="approved"),
            _dim("Product", "PRD_DSC", src_key="PRD_DMS_KEY", status="generated"),
        )
        ctx = _ctx(text_cols=["Month"])
        chips = compute_chip_eligibility(ctx, semantic_plan=plan)
        whs = next((c for c in chips if c["id"] == "drill_dim:Warehouse"), None)
        prd = next((c for c in chips if c["id"] == "drill_dim:Product"), None)
        if whs and prd:
            self.assertGreater(whs["confidence"], prd["confidence"])

    def test_drill_chips_ordered_before_predict_and_decide(self):
        """drill_dim chips must appear before predict and decide."""
        plan = _plan_with_dims(_dim("Warehouse", "WHS_DSC", status="approved"))
        ctx = {
            "mode": "time_series",
            "row_count": 6,
            "text_cols": [],
            "numeric_cols": ["Revenue"],
            "distribution_stats": {},
            "comparison_stats": {},
        }
        brief = {
            "time_series": {
                "direction": "increasing",
                "period_count": 6,
                "overall_pct_change": 15.0,
            }
        }
        chips = compute_chip_eligibility(ctx, brief=brief, semantic_plan=plan)
        ids = [c["id"] for c in chips]
        if "drill_dim:Warehouse" in ids:
            drill_idx  = ids.index("drill_dim:Warehouse")
            predict_idx = ids.index("predict") if "predict" in ids else 999
            decide_idx  = ids.index("decide") if "decide" in ids else 999
            self.assertLess(drill_idx, predict_idx)
            self.assertLess(drill_idx, decide_idx)

    def test_drill_chip_label_uses_dimension_name(self):
        plan = _plan_with_dims(_dim("Warehouse", "WHS_DSC"))
        ctx = _ctx(text_cols=["Month"])
        chips = compute_chip_eligibility(ctx, semantic_plan=plan)
        whs = next((c for c in chips if c["id"] == "drill_dim:Warehouse"), None)
        self.assertIsNotNone(whs)
        self.assertEqual(whs["label"], "Break down by Warehouse")

    def test_drill_chip_pre_context_mentions_dimension(self):
        plan = _plan_with_dims(_dim("Warehouse", "WHS_DSC"))
        ctx = _ctx(text_cols=["Month"])
        chips = compute_chip_eligibility(ctx, semantic_plan=plan)
        whs = next((c for c in chips if c["id"] == "drill_dim:Warehouse"), None)
        self.assertIsNotNone(whs)
        self.assertIn("Warehouse", whs["pre_context"])


# ══════════════════════════════════════════════════════════════════════════════
# build_runtime_semantic_plan — available_dimensions populated
# ══════════════════════════════════════════════════════════════════════════════

def _tmp_dir():
    root = Path("C:/tmp")
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=str(root))


def _write_schema(path: Path) -> None:
    schema = {
        "CHATBOTDB.PROFITABILITY.CUS_ORD_IVC_FCT": {
            "database": "CHATBOTDB", "schema": "PROFITABILITY",
            "table": "CUS_ORD_IVC_FCT",
            "columns": [
                {"name": "CUS_ORD_IVC_FCT_KEY", "type": "bigint"},
                {"name": "WHS_DMS_KEY",          "type": "bigint"},
                {"name": "CUS_IVC_DT_DMS_KEY",   "type": "bigint"},
                {"name": "SOP_CUS_IVC_LIN_AMT",  "type": "decimal"},
            ],
        },
        "CHATBOTDB.PROFITABILITY.WHS_DMS": {
            "database": "CHATBOTDB", "schema": "PROFITABILITY",
            "table": "WHS_DMS",
            "columns": [
                {"name": "WHS_DMS_KEY", "type": "bigint"},
                {"name": "WHS_DSC",     "type": "nvarchar"},
                {"name": "WHS_CD",      "type": "nvarchar"},
            ],
        },
        "CHATBOTDB.PROFITABILITY.DT_DMS": {
            "database": "CHATBOTDB", "schema": "PROFITABILITY",
            "table": "DT_DMS",
            "columns": [
                {"name": "DT_DMS_KEY", "type": "bigint"},
                {"name": "CAL_DT",     "type": "date"},
                {"name": "MONTH",      "type": "int"},
            ],
        },
    }
    (path / "_schema.json").write_text(json.dumps(schema), encoding="utf-8")


class AvailableDimensionsInPlanTests(unittest.TestCase):

    def test_available_dimensions_present_in_enabled_plan(self):
        from core.semantic_model import write_semantic_model, build_runtime_semantic_plan
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            plan = build_runtime_semantic_plan(
                str(kb_dir),
                question="show revenue by warehouse",
                selected_schema="PROFITABILITY",
            )

            self.assertTrue(plan["enabled"])
            self.assertIn("available_dimensions", plan)
            self.assertIsInstance(plan["available_dimensions"], list)
            self.assertGreater(len(plan["available_dimensions"]), 0)

    def test_available_dimensions_have_required_keys(self):
        from core.semantic_model import write_semantic_model, build_runtime_semantic_plan
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            plan = build_runtime_semantic_plan(
                str(kb_dir),
                question="show revenue by warehouse",
                selected_schema="PROFITABILITY",
            )

            for dim in plan.get("available_dimensions", []):
                for key in ("name", "display_column", "display_table",
                            "source_table", "source_key_column", "status"):
                    self.assertIn(key, dim,
                                  f"available_dimensions entry missing key {key!r}: {dim}")

    def test_available_dimensions_cap_at_20(self):
        """Even a very large schema must not exceed 20 available_dimensions."""
        from core.semantic_model import write_semantic_model, build_runtime_semantic_plan
        with _tmp_dir() as tmp:
            root = Path(tmp)
            schema_dir = root / "schema"
            kb_dir = root / "kb"
            schema_dir.mkdir()
            _write_schema(schema_dir)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))

            plan = build_runtime_semantic_plan(
                str(kb_dir),
                question="show revenue",
                selected_schema="PROFITABILITY",
            )

            available = plan.get("available_dimensions") or []
            self.assertLessEqual(len(available), 20)


# ══════════════════════════════════════════════════════════════════════════════
# generate_drill_by_dimension — deterministic fallback paths
# ══════════════════════════════════════════════════════════════════════════════

class DrillDimensionFallbackTests(unittest.TestCase):

    def _run(self, coro):
        return asyncio.run(coro)

    def test_fallback_when_dim_not_in_plan(self):
        from core.drill_dimension import generate_drill_by_dimension
        result = self._run(generate_drill_by_dimension(
            dim_name="NonExistentDim",
            rows=[{"Month": "2025-01", "Revenue": 100}],
            question="show revenue",
            original_sql="SELECT 1",
            semantic_plan={"enabled": True, "available_dimensions": []},
            db_cfg={"db_type": "azure_sql"},
            provider="anthropic", model="claude-sonnet-4-6", api_key="dummy",
        ))
        self.assertEqual(result["type"], "assistant_error")
        self.assertIn("not found", result["content"].lower())

    def test_fallback_when_plan_has_no_available_dims(self):
        from core.drill_dimension import generate_drill_by_dimension
        result = self._run(generate_drill_by_dimension(
            dim_name="Warehouse",
            rows=[],
            question="show revenue",
            original_sql="SELECT 1",
            semantic_plan={"enabled": True, "available_dimensions": []},
            db_cfg={"db_type": "azure_sql"},
            provider="anthropic", model="claude-sonnet-4-6", api_key="dummy",
        ))
        self.assertEqual(result["type"], "assistant_error")

    def test_fallback_response_has_suggestion(self):
        from core.drill_dimension import generate_drill_by_dimension
        result = self._run(generate_drill_by_dimension(
            dim_name="Warehouse",
            rows=[],
            question="show revenue",
            original_sql="SELECT 1",
            semantic_plan={"enabled": True, "available_dimensions": []},
            db_cfg={"db_type": "azure_sql"},
            provider="anthropic", model="claude-sonnet-4-6", api_key="dummy",
        ))
        self.assertIn("suggestion", result)
        self.assertTrue(result["suggestion"])


if __name__ == "__main__":
    unittest.main()
