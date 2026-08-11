from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.graph_resolver import resolve_for_question  # noqa: E402
from core.pipeline_trace import (  # noqa: E402
    _trace_create,
    _trace_finish_unclosed,
)
from core.semantic_resolution import (  # noqa: E402
    build_cannot_generate_recovery_prompt,
    build_planner_alignment,
    build_resolution_plan,
    check_sql_plan_coverage,
)


GRAPH = {
    "entities": [
        {
            "entity_name": "F_CLAIM",
            "display_name": "Claim Revenue",
            "entity_type": "fact",
            "schema_name": "PHARMA_LAB",
            "table_name": "F_CLAIM",
            "status": "confirmed",
        },
        {
            "entity_name": "F_RX_FILL",
            "display_name": "Prescription Fill",
            "entity_type": "fact",
            "schema_name": "PHARMA_LAB",
            "table_name": "F_RX_FILL",
            "status": "confirmed",
        },
        {
            "entity_name": "D_DATE",
            "display_name": "Date",
            "entity_type": "dimension",
            "schema_name": "PHARMA_LAB",
            "table_name": "D_DATE",
            "status": "confirmed",
        },
    ],
    "relationships": [
        {
            "id": 96,
            "from_entity": "F_CLAIM",
            "to_entity": "D_DATE",
            "from_column": "SERVICE_DATE_ID",
            "to_column": "DATE_ID",
            "join_type": "INNER",
            "relationship_type": "many_to_one",
            "generated_by": "manual",
            "status": "confirmed",
            "validation_status": "valid",
            "confidence_score": 100,
        },
        {
            "id": 97,
            "from_entity": "F_RX_FILL",
            "to_entity": "D_DATE",
            "from_column": "ORDER_DATE_ID",
            "to_column": "DATE_ID",
            "join_type": "INNER",
            "relationship_type": "many_to_one",
            "generated_by": "manual",
            "status": "confirmed",
            "validation_status": "valid",
            "confidence_score": 100,
        },
    ],
    "properties": [],
}


class PlannerAlignmentTests(unittest.TestCase):
    def test_semantic_fact_replaces_conflicting_lexical_fact(self):
        alignment = build_planner_alignment(
            graph=GRAPH,
            graph_ctx={
                "enabled": True,
                "detected": ["F_CLAIM", "D_DATE"],
                "anchor": "F_CLAIM",
            },
            semantic_plan={
                "enabled": True,
                "fields": [{
                    "term": "revenue",
                    "table": "PHARMA_LAB.F_RX_FILL",
                    "column": "NET_REVENUE_AMT",
                }],
                "required_tables": ["PHARMA_LAB.F_RX_FILL"],
            },
            date_context_resolution={
                "status": "selected",
                "binding": {
                    "fact_table": "PHARMA_LAB.F_RX_FILL",
                    "dimension_table": "PHARMA_LAB.D_DATE",
                    "fact_column": "ORDER_DATE_ID",
                },
            },
        )

        self.assertEqual(alignment["authoritative_fact_entities"], ["F_RX_FILL"])
        self.assertEqual(alignment["dropped_fact_entities"], ["F_CLAIM"])
        self.assertEqual(
            set(alignment["required_entities"]), {"F_RX_FILL", "D_DATE"},
        )

        resolved = resolve_for_question(
            question="what was my ordered revenue for the last 7 days",
            account_id="Demo_2",
            db_type="azure_sql",
            graph=GRAPH,
            required_entities=set(alignment["required_entities"]),
            authoritative_fact_tables=set(alignment["authoritative_fact_tables"]),
        )
        self.assertTrue(resolved["enabled"])
        self.assertEqual(resolved["anchor"], "F_RX_FILL")
        self.assertNotIn("F_CLAIM", resolved["detected"])
        self.assertEqual(resolved["edge_ids"], [97])
        self.assertIn("[PHARMA_LAB].[F_RX_FILL]", resolved["join_skeleton"])
        self.assertNotIn("[PHARMA_LAB].[F_CLAIM]", resolved["join_skeleton"])

    def test_two_governed_facts_remain_available_for_comparison(self):
        alignment = build_planner_alignment(
            graph=GRAPH,
            graph_ctx={"detected": ["F_CLAIM", "D_DATE"], "anchor": "F_CLAIM"},
            semantic_plan={
                "required_tables": [
                    "PHARMA_LAB.F_CLAIM", "PHARMA_LAB.F_RX_FILL",
                ],
            },
        )
        self.assertEqual(
            set(alignment["authoritative_fact_entities"]), {"F_CLAIM", "F_RX_FILL"},
        )
        self.assertEqual(alignment["dropped_fact_entities"], [])

    def test_dimension_only_plan_keeps_existing_fact_anchor(self):
        alignment = build_planner_alignment(
            graph=GRAPH,
            graph_ctx={"detected": ["F_CLAIM"], "anchor": "F_CLAIM"},
            semantic_plan={"required_tables": ["PHARMA_LAB.D_DATE"]},
        )
        self.assertEqual(alignment["authoritative_fact_entities"], [])
        self.assertEqual(
            set(alignment["required_entities"]), {"F_CLAIM", "D_DATE"},
        )

    def test_optional_rival_fact_does_not_drive_alignment(self):
        alignment = build_planner_alignment(
            graph=GRAPH,
            graph_ctx={
                "enabled": True,
                "detected": ["F_RX_FILL", "D_DATE"],
                "anchor": "F_RX_FILL",
            },
            semantic_plan={
                "enabled": True,
                "fields": [
                    {
                        "term": "revenue",
                        "table": "PHARMA_LAB.F_RX_FILL",
                        "column": "NET_REVENUE_AMT",
                        "enforcement": "required",
                    },
                    {
                        "term": "service date",
                        "table": "PHARMA_LAB.F_CLAIM",
                        "column": "SERVICE_DATE_ID",
                        "enforcement": "optional",
                    },
                ],
                "joins": [
                    {
                        "from": "PHARMA_LAB.F_CLAIM",
                        "to": "PHARMA_LAB.D_DATE",
                        "conditions": [("SERVICE_DATE_ID", "DATE_ID")],
                        "enforcement": "optional",
                    },
                ],
                # Deliberately stale derived state: structured enforcement
                # must win and exclude the optional rival fact.
                "required_tables": [
                    "PHARMA_LAB.F_RX_FILL",
                    "PHARMA_LAB.F_CLAIM",
                    "PHARMA_LAB.D_DATE",
                ],
            },
        )

        self.assertEqual(alignment["authoritative_fact_entities"], ["F_RX_FILL"])
        self.assertNotIn("F_CLAIM", alignment["required_entities"])
        self.assertNotIn("PHARMA_LAB.F_CLAIM", alignment["required_tables"])

    def test_resolution_plan_coverage_uses_reconciled_required_tables(self):
        alignment = {
            "required_tables": ["PHARMA_LAB.F_RX_FILL", "PHARMA_LAB.D_DATE"],
        }
        plan = build_resolution_plan(
            account_id="Demo_2",
            question="ordered revenue by month",
            planner_alignment=alignment,
        )
        coverage = check_sql_plan_coverage(
            "SELECT SUM(NET_REVENUE_AMT) FROM PHARMA_LAB.F_RX_FILL",
            plan,
            "azure_sql",
        )
        self.assertEqual(coverage["coverage_ratio"], 0.5)
        self.assertEqual(coverage["unused_expected_tables"], ["PHARMA_LAB.D_DATE"])


class CannotGenerateRecoveryTests(unittest.TestCase):
    def test_recovery_prompt_contains_authoritative_plan(self):
        prompt = build_cannot_generate_recovery_prompt(
            question="ordered revenue for the last 7 days",
            resolution_plan={
                "required_tables": ["PHARMA_LAB.F_RX_FILL", "PHARMA_LAB.D_DATE"],
            },
            graph_ctx={
                "join_skeleton": (
                    "FROM [PHARMA_LAB].[F_RX_FILL] fr "
                    "JOIN [PHARMA_LAB].[D_DATE] dd ON fr.[ORDER_DATE_ID] = dd.[DATE_ID]"
                ),
            },
            semantic_plan={
                "fields": [{
                    "term": "revenue",
                    "table": "PHARMA_LAB.F_RX_FILL",
                    "column": "NET_REVENUE_AMT",
                }],
            },
        )
        self.assertIn("Original question: ordered revenue for the last 7 days", prompt)
        self.assertIn("PHARMA_LAB.F_RX_FILL", prompt)
        self.assertIn("revenue=PHARMA_LAB.F_RX_FILL.NET_REVENUE_AMT", prompt)
        self.assertIn("ORDER_DATE_ID", prompt)
        self.assertIn("one constrained recovery", prompt)

    def test_pipeline_wires_one_recovery_and_terminal_failure(self):
        source = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        self.assertEqual(source.count('component="sql_cannot_generate_recovery"'), 1)
        self.assertIn('metadata={"retry": True, "reason": "cannot_generate"}', source)
        branch = source[source.index('if "CANNOT_GENERATE" in sql.upper():') :]
        self.assertIn('answer_type="cannot_generate"', branch[:1800])
        self.assertIn("_trace_finish(", branch[:1800])

    def test_recovery_prompt_excludes_optional_rival_field(self):
        prompt = build_cannot_generate_recovery_prompt(
            question="revenue by month",
            resolution_plan={"required_tables": ["PHARMA_LAB.F_RX_FILL"]},
            graph_ctx={},
            semantic_plan={
                "fields": [
                    {
                        "term": "revenue",
                        "table": "PHARMA_LAB.F_RX_FILL",
                        "column": "NET_REVENUE_AMT",
                        "enforcement": "required",
                    },
                    {
                        "term": "claim date",
                        "table": "PHARMA_LAB.F_CLAIM",
                        "column": "SERVICE_DATE_ID",
                        "enforcement": "optional",
                    },
                ],
            },
        )

        self.assertIn("NET_REVENUE_AMT", prompt)
        self.assertNotIn("SERVICE_DATE_ID", prompt)


class TraceLifecycleGuardTests(unittest.TestCase):
    def test_unclosed_trace_is_finalized_once(self):
        with patch("core.pipeline_trace.store.create_answer_trace", return_value=321), patch(
            "core.pipeline_trace.store.finish_answer_trace",
        ) as finish:
            self.assertEqual(_trace_create(account_id="a", question="q", question_id="id"), 321)
            _trace_finish_unclosed(
                status="error",
                answer_type="error",
                error_message="guard",
            )
            _trace_finish_unclosed(status="error", answer_type="error")

        finish.assert_called_once_with(
            321,
            status="error",
            answer_type="error",
            error_message="guard",
        )

    def test_public_pipeline_has_finally_guard(self):
        source = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        wrapper = source[source.rindex("async def handle_query(") :]
        self.assertIn("finally:", wrapper)
        self.assertIn("_trace_finish_unclosed(", wrapper)


if __name__ == "__main__":
    unittest.main()
