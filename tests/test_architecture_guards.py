import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from core.clarification import (
    check_ambiguity_glossary_first,
    resolve_option_text,
    combine_with_clarification,
    build_schema_grounded_clarification_hint,
)
from core.insight import build_action_contract, compute_data_brief
from core.query_semantics import build_generic_query_hints
from core.response_builder import build_analysis_response, summarize_result_context
from core.semantic_registry import validated_options


class ArchitectureGuardTests(unittest.TestCase):
    def test_validated_options_filter_out_invalid_entries(self):
        options = [
            {"id": "attendance_status", "label": "Attendance status", "value": "attendance_status", "valid": True},
            {"id": "attrition_count", "label": "Attrition count", "value": "attrition_count", "valid": False},
        ]
        filtered = validated_options(options, assume_legacy_valid=False)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["id"], "attendance_status")

    def test_llm_fallback_does_not_surface_generated_options(self):
        async def _run():
            with patch("core.clarification.find_registry_clarification", return_value=None), \
                 patch("core.clarification._llm_ambiguity_check", return_value=(True, "Which business meaning did you intend?", [{"id": "fake", "label": "Invented", "value": "invented"}])):
                import store
                with patch.object(store, "match_terms_in_question", return_value=[]), \
                     patch.object(store, "list_terms", return_value=[]):
                    return await check_ambiguity_glossary_first(
                        account_id="acct",
                        question="show absenteeism",
                        context="attendance context",
                        provider="test",
                        model="test",
                        api_key="",
                        extra_kwargs={},
                    )

        is_ambiguous, clarifying_q, meta = asyncio.run(_run())
        self.assertTrue(is_ambiguous)
        self.assertEqual(clarifying_q, "Which business meaning did you intend?")
        self.assertEqual(meta["options"], [])
        self.assertTrue(meta["generated_options_ignored"])

    def test_top_one_scope_is_explicit_in_explain_fallback(self):
        rows = [{"nationality": "United Arab Emirates", "employee_count": 305}]
        sql = "SELECT TOP 1 nationality, COUNT(*) AS employee_count FROM employees GROUP BY nationality ORDER BY employee_count DESC"
        ctx = summarize_result_context(rows, "which nationality employees are higher in count", sql)
        explain = build_analysis_response("explain", ctx)

        self.assertEqual(ctx["result_scope"]["badge"], "Top result only")
        self.assertIn("top result only", explain["body"].lower())
        self.assertIn("returned slice", explain["secondary"].lower())

    def test_action_fallbacks_are_differentiated(self):
        rows = [
            {"nationality": "UAE", "employee_count": 305},
            {"nationality": "Indian", "employee_count": 187},
            {"nationality": "Egyptian", "employee_count": 91},
            {"nationality": "Jordanian", "employee_count": 45},
        ]
        sql = "SELECT nationality, COUNT(*) AS employee_count FROM employees GROUP BY nationality ORDER BY employee_count DESC"
        ctx = summarize_result_context(rows, "which nationality employees are higher in count", sql)

        explain = build_analysis_response("explain", ctx)
        analyze = build_analysis_response("analyze", ctx)
        compare = build_analysis_response("compare", ctx)

        self.assertNotEqual(explain["body"], analyze["body"])
        self.assertNotEqual(compare["body"], analyze["body"])
        self.assertIn("concentrated", analyze["body"].lower())
        self.assertIn("ahead of", compare["body"].lower())
        self.assertIn("ranks first", explain["body"].lower())

    def test_safe_payload_redacts_pii_like_labels(self):
        rows = [
            {"employee_name": "Alice Adams", "employee_count": 12},
            {"employee_name": "Bob Brown", "employee_count": 10},
        ]
        sql = "SELECT employee_name, COUNT(*) AS employee_count FROM attendance GROUP BY employee_name ORDER BY employee_count DESC"
        ctx = summarize_result_context(rows, "which employee has the most attendance records", sql)
        brief = compute_data_brief(
            rows,
            "which employee has the most attendance records",
            result_scope=ctx["result_scope"],
            context=ctx,
        )
        contract = build_action_contract("compare", "which employee has the most attendance records", brief)
        payload_json = json.dumps({"brief": brief, "contract": contract})

        self.assertTrue(brief["category_breakdown"]["labels_redacted"])
        self.assertNotIn("Alice Adams", payload_json)
        self.assertNotIn("Bob Brown", payload_json)
        self.assertIn("redacted segment", payload_json)

    def test_resolve_option_text_supports_tolerant_text_match(self):
        options = [
            {"id": "attendance_status", "label": "Absenteeism based on attendance status", "value": "attendance_status"},
            {"id": "attrition_count", "label": "Absenteeism by attrition count", "value": "attrition_count"},
        ]
        resolved = resolve_option_text(options, "yes absenteeism by attrition count")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["id"], "attrition_count")

    def test_free_text_clarification_stays_attached_to_original_question(self):
        combined, injection = combine_with_clarification(
            "find my unique employee counts based on their department who are marked as lete",
            "it is an existing attendance status called late",
            {"source": "llm", "question": "Which attendance status did you mean?", "options": []},
        )
        self.assertIn("find my unique employee counts", combined.lower())
        self.assertIn("clarification for the same request", combined.lower())
        self.assertIn("existing attendance status called late", combined.lower())
        self.assertEqual(injection, "")

    def test_generic_query_hints_cover_distinct_employee_groupings(self):
        hints = build_generic_query_hints(
            "find my unique employee counts based on their department who are marked as lete"
        )
        self.assertIn("count(distinct stable employee key)", hints.lower())
        self.assertIn("group by that category", hints.lower())
        self.assertIn("preserved exactly", hints.lower())

    def test_schema_grounded_clarification_hint_uses_context_values_and_typos(self):
        import store

        fake_terms = [
            {
                "term": "attendance",
                "aliases": "late, absent",
                "tables_involved": "",
            }
        ]
        with patch.object(store, "list_terms", return_value=fake_terms), \
             patch.object(store, "match_terms_in_question", return_value=[{"term": "attendance"}]):
            hint = build_schema_grounded_clarification_hint(
                account_id="acct",
                question="find unique employees who are marked as lete by department",
                context="Attendance status values are 'Late', 'Absent', 'Present'.",
            )

        self.assertIn("schema-grounded interpretation evidence", hint.lower())
        self.assertIn("'lete' -> 'late'", hint.lower())
        self.assertIn("late, absent, present", hint.lower())
        self.assertIn("query intent", hint.lower())

    def test_schema_grounded_hint_preserves_exact_schema_value_from_schema_files(self):
        import store

        fake_terms = [
            {
                "term": "attendance",
                "aliases": "late, absent",
                "tables_involved": "",
            }
        ]
        with patch.object(store, "list_terms", return_value=fake_terms), \
             patch.object(store, "match_terms_in_question", return_value=[{"term": "attendance"}]), \
             patch("core.clarification._schema_distinct_value_candidates", return_value=["Lete", "Absent", "Present"]):
            hint = build_schema_grounded_clarification_hint(
                account_id="acct",
                question="find unique employees who are marked as lete by department",
                context="No useful status values were retrieved from RAG.",
            )

        self.assertIn("exact schema-backed values already present in the user question: lete", hint.lower())
        self.assertNotIn("'lete' -> 'late'", hint.lower())

    def test_exact_schema_value_status_filter_skips_clarification_llm(self):
        import store

        fake_terms = [
            {
                "id": 1,
                "term": "attendance",
                "kind": "metric",
                "aliases": "late, absent",
                "definition": "Attendance status tracking",
                "canonical_expression": "",
                "tables_involved": "",
            }
        ]

        async def _run():
            with patch("core.clarification.find_registry_clarification", return_value=None), \
                 patch.object(store, "list_terms", return_value=fake_terms), \
                 patch.object(store, "match_terms_in_question", return_value=[{"id": 1, "term": "attendance", "kind": "metric"}]), \
                 patch("core.clarification._schema_distinct_value_candidates", return_value=["Lete", "Absent", "Present"]), \
                 patch("core.clarification._llm_ambiguity_check", side_effect=AssertionError("plain LLM should not run")), \
                 patch("core.clarification._llm_ambiguity_check_constrained", side_effect=AssertionError("menu LLM should not run")):
                return await check_ambiguity_glossary_first(
                    account_id="acct",
                    question="find unique employees who are marked as lete by department",
                    context="Unrelated retrieved context.",
                    provider="test",
                    model="test",
                    api_key="",
                    extra_kwargs={},
                )

        is_ambiguous, clarifying_q, meta = asyncio.run(_run())
        self.assertFalse(is_ambiguous)
        self.assertEqual(clarifying_q, "")
        self.assertEqual(meta["source"], "none")
        self.assertEqual(meta["options"], [])


if __name__ == "__main__":
    unittest.main()
