"""Sprint 3: the deterministic query-time resolution plan.

build_resolution_plan() is a pure function (no I/O, no store access) that
assembles four already-computed pipeline structures - store.match_metric's
result, core.graph_resolver's graph_ctx, the merged semantic field plan,
and core.contextual_dates' date_context_resolution - into one structured
object, plus computes confidence/clarifications from cross-referencing
what the question touches against Sprint 2's open compile-time conflicts.

This module does not call any resolver itself and does not gate the
pipeline yet - see core/semantic_resolution.py's own module docstring for
what's deliberately deferred (SQL-plan enforcement, actually blocking on
resolved_deterministically).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.semantic_resolution import build_resolution_plan  # noqa: E402


class FullyResolvedPlanTests(unittest.TestCase):
    def test_all_sections_assembled_from_inputs(self):
        plan = build_resolution_plan(
            account_id="acct1", question="revenue by region this month",
            contract={"meta": {"contract_version": "abc123"}},
            matched_metrics=[{"id": 1, "name": "Revenue"}],
            graph_ctx={
                "edge_ids": [5],
                "resolved_edges": [{
                    "id": 5, "from_entity": "FACT_SALES", "to_entity": "DIM_REGION",
                    "join_type": "INNER", "validation_status": "valid",
                }],
            },
            semantic_plan={"fields": [{
                "term": "region", "table": "ERP.DIM_REGION", "column": "REGION_NAME",
                "enforcement": "required",
            }]},
            date_context_resolution={
                "status": "selected",
                "binding": {
                    "fact_table": "ERP.FACT_SALES", "fact_column": "ORDER_DT",
                    "context_name": "Order Date", "date_role": "order_date",
                    "resolution_source": "default",
                },
            },
            schema_hint="ERP", allowed_tables={"ERP.FACT_SALES", "ERP.DIM_REGION"},
        )
        self.assertEqual(plan["contract_version"], "abc123")
        self.assertEqual(plan["schemas"], ["ERP"])
        self.assertEqual(plan["metrics"], [{
            "canonical_id": "metric:1", "name": "Revenue", "source": "metric_registry",
        }])
        self.assertEqual(plan["dimensions"][0]["canonical_id"], "field:ERP.DIM_REGION.REGION_NAME")
        self.assertEqual(plan["date_roles"][0]["canonical_id"], "date_role:ERP.FACT_SALES.ORDER_DT")
        self.assertEqual(plan["graph_path"][0]["canonical_id"], "join:5")
        self.assertEqual(plan["access_scope"]["schema_hint"], "ERP")
        self.assertEqual(plan["access_scope"]["allowed_tables"], ["ERP.DIM_REGION", "ERP.FACT_SALES"])
        self.assertEqual(plan["confidence"], 100.0)
        self.assertTrue(plan["resolved_deterministically"])
        self.assertEqual(plan["clarifications"], [])

    def test_selected_many_uses_first_binding(self):
        plan = build_resolution_plan(
            account_id="acct1", question="q",
            date_context_resolution={
                "status": "selected_many",
                "bindings": [
                    {"fact_table": "T1", "fact_column": "C1", "context_name": "A", "date_role": "r1"},
                    {"fact_table": "T2", "fact_column": "C2", "context_name": "B", "date_role": "r2"},
                ],
            },
        )
        self.assertEqual(len(plan["date_roles"]), 1)
        self.assertEqual(plan["date_roles"][0]["canonical_id"], "date_role:T1.C1")


class UnresolvedPlanTests(unittest.TestCase):
    def test_nothing_resolved_still_produces_a_valid_plan(self):
        plan = build_resolution_plan(account_id="acct1", question="asdf")
        self.assertEqual(plan["metrics"], [])
        self.assertEqual(plan["dimensions"], [])
        self.assertEqual(plan["date_roles"], [])
        self.assertEqual(plan["graph_path"], [])
        self.assertEqual(plan["confidence"], 60.0)  # 100 - 40 (no structural grounding)
        self.assertTrue(plan["resolved_deterministically"])  # low confidence != blocked

    def test_metric_matched_alone_avoids_the_no_structure_penalty(self):
        plan = build_resolution_plan(
            account_id="acct1", question="revenue", matched_metrics=[{"id": 1, "name": "Revenue"}],
        )
        self.assertEqual(plan["confidence"], 100.0)

    def test_missing_metric_id_is_skipped_not_guessed(self):
        plan = build_resolution_plan(
            account_id="acct1", question="q", matched_metrics=[{"name": "No id field"}],
        )
        self.assertEqual(plan["metrics"], [])

    def test_field_without_table_or_column_is_skipped(self):
        plan = build_resolution_plan(
            account_id="acct1", question="q",
            semantic_plan={"fields": [{"term": "x"}, {"term": "y", "table": "T"}]},
        )
        self.assertEqual(plan["dimensions"], [])


class MultipleMatchedMetricsTests(unittest.TestCase):
    """matched_metrics is plural to match core/query_pipeline.py's own
    _matched_metrics — a comparison question ("revenue vs. margin") can
    legitimately resolve more than one approved metric at once."""

    def test_two_metrics_both_appear_in_the_plan(self):
        plan = build_resolution_plan(
            account_id="acct1", question="revenue vs margin",
            matched_metrics=[
                {"id": 1, "name": "Revenue"}, {"id": 2, "name": "Margin"},
            ],
        )
        self.assertEqual(len(plan["metrics"]), 2)
        ids = {m["canonical_id"] for m in plan["metrics"]}
        self.assertEqual(ids, {"metric:1", "metric:2"})

    def test_conflict_touching_the_second_metric_still_matches(self):
        # Not just the first entry in the list - every matched metric
        # participates in the conflict cross-reference.
        plan = build_resolution_plan(
            account_id="acct1", question="revenue vs margin",
            matched_metrics=[
                {"id": 1, "name": "Revenue"}, {"id": 2, "name": "Margin"},
            ],
            open_conflicts=[{
                "code": "duplicate_metric_name", "severity": "ERROR", "object_id": "metric:2",
                "conflict_key": "duplicate_metric_name::metric:2|metric:9", "message": "collides",
            }],
        )
        self.assertFalse(plan["resolved_deterministically"])
        self.assertEqual(len(plan["clarifications"]), 1)

    def test_one_metric_missing_id_does_not_drop_the_other(self):
        plan = build_resolution_plan(
            account_id="acct1", question="q",
            matched_metrics=[{"name": "no id"}, {"id": 3, "name": "Has id"}],
        )
        self.assertEqual(len(plan["metrics"]), 1)
        self.assertEqual(plan["metrics"][0]["canonical_id"], "metric:3")


class AmbiguousDateContextTests(unittest.TestCase):
    def test_ambiguous_status_is_a_blocking_clarification(self):
        plan = build_resolution_plan(
            account_id="acct1", question="revenue by date",
            matched_metrics=[{"id": 1, "name": "Revenue"}],
            date_context_resolution={
                "status": "ambiguous", "reason": "two dates",
                "options": [{"label": "A"}, {"label": "B"}],
            },
        )
        self.assertFalse(plan["resolved_deterministically"])
        self.assertEqual(len(plan["clarifications"]), 1)
        self.assertEqual(plan["clarifications"][0]["type"], "date_context_ambiguous")
        self.assertEqual(plan["clarifications"][0]["options"], [{"label": "A"}, {"label": "B"}])
        self.assertEqual(plan["confidence"], 75.0)  # 100 - 25

    def test_none_status_produces_no_date_roles_and_no_clarification(self):
        plan = build_resolution_plan(
            account_id="acct1", question="q", date_context_resolution={"status": "none"},
        )
        self.assertEqual(plan["date_roles"], [])
        self.assertEqual(plan["clarifications"], [])


class CompileTimeConflictBridgeTests(unittest.TestCase):
    """The genuinely new capability: cross-referencing what a question
    resolves to against Sprint 2's open conflicts."""

    def test_error_conflict_touching_resolved_metric_blocks(self):
        plan = build_resolution_plan(
            account_id="acct1", question="revenue",
            matched_metrics=[{"id": 1, "name": "Revenue"}],
            open_conflicts=[{
                "code": "duplicate_metric_name", "severity": "ERROR", "object_id": "metric:1",
                "conflict_key": "duplicate_metric_name::metric:1|metric:2", "message": "collides",
            }],
        )
        self.assertFalse(plan["resolved_deterministically"])
        self.assertEqual(len(plan["clarifications"]), 1)
        self.assertEqual(plan["clarifications"][0]["type"], "compile_time_conflict")
        self.assertEqual(plan["advisories"], [])

    def test_error_conflict_matches_via_second_participant_not_just_object_id(self):
        # object_id on the conflict row is metric:2 (whichever participant
        # the detector happened to pick as "first"), but THIS question
        # resolved to metric:1 - the other participant in the same
        # conflict_key. Must still match.
        plan = build_resolution_plan(
            account_id="acct1", question="revenue",
            matched_metrics=[{"id": 1, "name": "Revenue"}],
            open_conflicts=[{
                "code": "duplicate_metric_name", "severity": "ERROR", "object_id": "metric:2",
                "conflict_key": "duplicate_metric_name::metric:1|metric:2", "message": "collides",
            }],
        )
        self.assertEqual(len(plan["clarifications"]), 1)

    def test_warning_conflict_is_advisory_not_blocking(self):
        plan = build_resolution_plan(
            account_id="acct1", question="revenue",
            matched_metrics=[{"id": 1, "name": "Revenue"}],
            open_conflicts=[{
                "code": "unreviewed_restricted_column", "severity": "WARNING", "object_id": "metric:1",
                "conflict_key": "unreviewed_restricted_column::metric:1|field:X.Y", "message": "unreviewed",
            }],
        )
        self.assertTrue(plan["resolved_deterministically"])
        self.assertEqual(plan["clarifications"], [])
        self.assertEqual(len(plan["advisories"]), 1)

    def test_unrelated_conflict_is_ignored_entirely(self):
        plan = build_resolution_plan(
            account_id="acct1", question="revenue",
            matched_metrics=[{"id": 1, "name": "Revenue"}],
            open_conflicts=[{
                "code": "duplicate_metric_name", "severity": "ERROR", "object_id": "metric:99",
                "conflict_key": "duplicate_metric_name::metric:99|metric:100", "message": "unrelated",
            }],
        )
        self.assertEqual(plan["clarifications"], [])
        self.assertEqual(plan["advisories"], [])
        self.assertEqual(plan["confidence"], 100.0)

    def test_conflict_touching_a_join_this_question_used(self):
        plan = build_resolution_plan(
            account_id="acct1", question="q",
            graph_ctx={"edge_ids": [7], "resolved_edges": [{
                "id": 7, "from_entity": "A", "to_entity": "B",
                "join_type": "INNER", "validation_status": "warning",
            }]},
            open_conflicts=[{
                "code": "high_risk_join_edge", "severity": "WARNING", "object_id": "join:7",
                "conflict_key": "high_risk_join_edge::join:7", "message": "risky",
            }],
        )
        self.assertEqual(len(plan["advisories"]), 1)

    def test_malformed_conflict_key_falls_back_to_object_id(self):
        plan = build_resolution_plan(
            account_id="acct1", question="revenue",
            matched_metrics=[{"id": 1, "name": "Revenue"}],
            open_conflicts=[{
                "code": "x", "severity": "ERROR", "object_id": "metric:1", "conflict_key": "",
                "message": "m",
            }],
        )
        self.assertEqual(len(plan["clarifications"]), 1)

    def test_non_dict_conflict_entries_are_skipped_not_crashed_on(self):
        plan = build_resolution_plan(
            account_id="acct1", question="q",
            matched_metrics=[{"id": 1, "name": "Revenue"}],
            open_conflicts=["not-a-dict", None, 42],
        )
        self.assertEqual(plan["clarifications"], [])

    def test_confidence_floor_does_not_go_negative(self):
        conflicts = [
            {
                "code": f"c{i}", "severity": "ERROR", "object_id": "metric:1",
                "conflict_key": f"c{i}::metric:1", "message": "m",
            }
            for i in range(10)
        ]
        plan = build_resolution_plan(
            account_id="acct1", question="q",
            matched_metrics=[{"id": 1, "name": "Revenue"}],
            date_context_resolution={"status": "ambiguous", "reason": "x", "options": []},
            open_conflicts=conflicts,
        )
        self.assertGreaterEqual(plan["confidence"], 0.0)


class ResolutionPlanWiringTests(unittest.TestCase):
    """The plan is only worth anything if it actually gets built during
    real answering. handle_query gets static source assertions, matching
    this codebase's established pattern for the ~1,100-line pipeline
    function (see QuestionScrubWiringTests in test_masking_privacy.py) -
    a real end-to-end run would need a live LLM + DB, which the existing
    suite deliberately avoids for this function."""

    def setUp(self):
        self.src = (ROOT / "core/query_pipeline.py").read_text(encoding="utf-8")

    def test_resolution_plan_built_before_sql_prompt(self):
        self.assertIn("from core.semantic_resolution import build_resolution_plan", self.src)
        build_pos = self.src.index("build_resolution_plan(")
        prompt_pos = self.src.index('system = build_sql_system_prompt(')
        self.assertLess(
            build_pos, prompt_pos,
            "resolution plan must be assembled before the SQL-generation prompt is built",
        )

    def test_resolution_plan_call_wrapped_in_try_except(self):
        # Observational only - a failure here must never break answering.
        anchor = "from core.semantic_resolution import build_resolution_plan"
        pos = self.src.index(anchor)
        before = self.src[:pos]
        self.assertEqual(before.rstrip().splitlines()[-1].strip(), "try:")
        after = self.src[pos:pos + 1500]
        self.assertIn("except Exception as _resolution_plan_exc:", after)

    def test_open_conflicts_fetched_with_open_status(self):
        self.assertIn('store.list_semantic_conflicts(account_id, status="open")', self.src)

    def test_resolution_plan_passed_pipeline_state_by_name(self):
        block = self.src[self.src.index("build_resolution_plan(") : self.src.index("build_resolution_plan(") + 700]
        for kwarg in (
            "matched_metrics=_matched_metrics",
            "graph_ctx=_graph_ctx",
            "semantic_plan=_semantic_plan",
            "date_context_resolution=_date_context_resolution",
            "allowed_tables=query_scope_tables",
            "contract=_contract",
        ):
            self.assertIn(kwarg, block)

    def test_resolution_plan_attached_to_trace(self):
        pos = self.src.index("build_resolution_plan(")
        after = self.src[pos:pos + 1500]
        self.assertIn('_trace_step(\n            trace_id, "resolution_plan"', after)
        self.assertIn("metadata=_resolution_plan", after)


if __name__ == "__main__":
    unittest.main()
