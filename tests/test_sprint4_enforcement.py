"""Sprint 4: full runtime enforcement.

Sprint 3 built the resolution plan and cross-referenced it against Sprint
2's open compile-time conflicts, but deliberately did nothing with the
result beyond tracing it (see core/semantic_resolution.py's own module
docstring). Sprint 4 closes both deferred pieces:

  4a. core/query_pipeline.py now actually blocks answering - but only in
      "enforce" mode - when a question resolves onto an ERROR-severity
      compile-time conflict. "off"/"shadow" leave Sprint 3's
      observational-only behavior unchanged.
  4b. core/semantic_resolution.py's new check_sql_plan_coverage() compares
      the generated SQL's table usage against what the resolution plan
      expected. This one stays advisory-only (traced, never blocking) -
      an LLM legitimately touching an unlisted join/lookup table is common
      and not itself a defect, so this follows the same "observe first"
      path Sprint 2's detectors and Sprint 3's plan itself went through.
"""
from __future__ import annotations

import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.semantic_resolution import build_resolution_plan, check_sql_plan_coverage  # noqa: E402


class SqlPlanCoverageTests(unittest.TestCase):
    def test_full_coverage_when_sql_uses_expected_table(self):
        plan = build_resolution_plan(
            account_id="acct1", question="revenue by region",
            semantic_plan={"fields": [{"table": "SALES.ORDERS", "column": "REGION_ID"}]},
        )
        result = check_sql_plan_coverage(
            "SELECT REGION_ID, SUM(REVENUE) FROM SALES.ORDERS GROUP BY REGION_ID",
            plan, "azure_sql",
        )
        self.assertEqual(result["expected_tables"], ["SALES.ORDERS"])
        self.assertEqual(result["unused_expected_tables"], [])
        self.assertEqual(result["coverage_ratio"], 1.0)

    def test_unused_table_flagged_when_sql_misses_it(self):
        plan = build_resolution_plan(
            account_id="acct1", question="revenue by region",
            semantic_plan={"fields": [{"table": "SALES.ORDERS", "column": "REGION_ID"}]},
        )
        result = check_sql_plan_coverage(
            "SELECT REGION_ID FROM LEGACY_SALES", plan, "azure_sql",
        )
        self.assertEqual(result["unused_expected_tables"], ["SALES.ORDERS"])
        self.assertEqual(result["coverage_ratio"], 0.0)

    def test_bare_table_name_matches_fqn_expectation(self):
        plan = build_resolution_plan(
            account_id="acct1", question="q",
            semantic_plan={"fields": [{"table": "SALES.ORDERS", "column": "REGION_ID"}]},
        )
        result = check_sql_plan_coverage("SELECT REGION_ID FROM ORDERS", plan, "azure_sql")
        self.assertEqual(result["unused_expected_tables"], [])

    def test_date_role_table_counted_as_expected(self):
        plan = build_resolution_plan(
            account_id="acct1", question="q",
            date_context_resolution={
                "status": "selected",
                "binding": {"fact_table": "FACT_SALES", "fact_column": "ORDER_DATE"},
            },
        )
        result = check_sql_plan_coverage("SELECT 1 FROM FACT_SALES", plan, "azure_sql")
        self.assertIn("FACT_SALES", result["expected_tables"])
        self.assertEqual(result["unused_expected_tables"], [])

    def test_empty_plan_never_divides_by_zero(self):
        plan = build_resolution_plan(account_id="acct1", question="hi")
        result = check_sql_plan_coverage("SELECT 1", plan, "azure_sql")
        self.assertEqual(result["expected_tables"], [])
        self.assertEqual(result["coverage_ratio"], 1.0)

    def test_never_raises_on_unparseable_sql(self):
        plan = build_resolution_plan(
            account_id="acct1", question="q",
            semantic_plan={"fields": [{"table": "ORDERS", "column": "X"}]},
        )
        # Should not raise even on garbage input - callers treat this as
        # best-effort trace data, not a validator with error semantics.
        result = check_sql_plan_coverage("not even sql {{{", plan, "azure_sql")
        self.assertIn("coverage_ratio", result)


class EnforcementDecisionTests(unittest.TestCase):
    """Mirrors the exact filter core/query_pipeline.py's Sprint 4a block
    applies to a resolution plan's clarifications list."""

    @staticmethod
    def _blocking(plan: dict) -> list[dict]:
        return [c for c in plan.get("clarifications") or [] if c.get("type") == "compile_time_conflict"]

    def test_error_conflict_produces_a_blocking_entry(self):
        plan = build_resolution_plan(
            account_id="acct1", question="q",
            matched_metrics=[{"id": 1, "name": "Revenue"}],
            open_conflicts=[{
                "code": "duplicate_metric_name", "severity": "ERROR",
                "conflict_key": "duplicate_metric_name::metric:1",
                "message": "Two metrics share this name.",
            }],
        )
        self.assertEqual(len(self._blocking(plan)), 1)

    def test_warning_conflict_never_blocks(self):
        plan = build_resolution_plan(
            account_id="acct1", question="q",
            matched_metrics=[{"id": 1, "name": "Revenue"}],
            open_conflicts=[{
                "code": "cardinality_fanout", "severity": "WARNING",
                "conflict_key": "cardinality_fanout::metric:1",
                "message": "Possible fan-out.",
            }],
        )
        self.assertEqual(self._blocking(plan), [])

    def test_conflict_touching_unrelated_object_never_blocks(self):
        plan = build_resolution_plan(
            account_id="acct1", question="q",
            matched_metrics=[{"id": 1, "name": "Revenue"}],
            open_conflicts=[{
                "code": "duplicate_metric_name", "severity": "ERROR",
                "conflict_key": "duplicate_metric_name::metric:999",
                "message": "Unrelated conflict.",
            }],
        )
        self.assertEqual(self._blocking(plan), [])


class Sprint4WiringTests(unittest.TestCase):
    """Static source assertions for the ~1,100-line handle_query, matching
    this codebase's established pattern for that function (see
    ResolutionPlanWiringTests, QuestionScrubWiringTests)."""

    def setUp(self):
        self.src = (ROOT / "core/query_pipeline.py").read_text(encoding="utf-8")

    # ── 4a: enforcement gate ─────────────────────────────────────────────

    def test_enforcement_gate_sits_after_resolution_plan_before_sql_prompt(self):
        plan_pos = self.src.index("build_resolution_plan(")
        gate_pos = self.src.index("_blocking_conflicts = [")
        prompt_pos = self.src.index('system = build_sql_system_prompt(')
        self.assertLess(plan_pos, gate_pos)
        self.assertLess(gate_pos, prompt_pos)

    def test_gate_only_fires_in_enforce_mode(self):
        block = self.src[self.src.index("_blocking_conflicts = ["):self.src.index("_blocking_conflicts = [") + 1200]
        self.assertIn('_compiler_mode == "enforce"', block)
        self.assertIn("store.get_semantic_compiler_state(account_id)", block)

    def test_gate_respects_is_clarification(self):
        anchor = "if _resolution_plan and not is_clarification:"
        self.assertIn(anchor, self.src)

    def test_gate_finishes_trace_and_returns(self):
        block = self.src[self.src.index("_blocking_conflicts = ["):self.src.index("_blocking_conflicts = [") + 1500]
        self.assertIn('answer_type="semantic_conflict_blocked"', block)
        self.assertIn("await adapter.send_message(", block)
        self.assertIn("return", block)

    def test_gate_mode_lookup_is_best_effort(self):
        # A failure reading compiler state must fall back to "shadow"
        # (non-blocking), not raise and break answering.
        pos = self.src.index("_compiler_mode = store.get_semantic_compiler_state(account_id)")
        before = self.src[:pos]
        self.assertEqual(before.rstrip().splitlines()[-1].strip(), "try:")

    # ── 4b: SQL-plan coverage ────────────────────────────────────────────

    def test_coverage_check_sits_after_sql_generation(self):
        gen_pos = self.src.index('"llm_generate_sql"')
        coverage_pos = self.src.index("check_sql_plan_coverage")
        self.assertLess(gen_pos, coverage_pos)

    def test_coverage_check_is_best_effort_and_advisory_only(self):
        pos = self.src.index("from core.semantic_resolution import check_sql_plan_coverage")
        before = self.src[:pos]
        self.assertEqual(before.rstrip().splitlines()[-1].strip(), "try:")
        after = self.src[pos:pos + 700]
        self.assertIn("except Exception as _coverage_exc:", after)
        self.assertNotIn("return", after[:after.index("except Exception as _coverage_exc:")])

    def test_coverage_attached_to_trace(self):
        pos = self.src.index("check_sql_plan_coverage(")
        after = self.src[pos:pos + 600]
        self.assertIn('"sql_plan_coverage"', after)
        self.assertIn("metadata=_sql_plan_coverage", after)


if __name__ == "__main__":
    unittest.main()
