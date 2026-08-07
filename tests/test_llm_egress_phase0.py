"""
LLM egress hardening — Phase 0.

Covers the four Phase-0 items from docs/LLM_EGRESS_PLAN.md:
  A2  raw DB error text no longer reaches the repair prompt verbatim
  A5  drill_dim sits behind the regulated result-LLM boundary
  --  the row-serialising narration function is gone for good
  --  the three result-chat fallback SQL calls are inside an audit scope

The A2 tests are the negative-space kind the plan flagged as missing: they
assert a known data value is ABSENT from text that is about to be handed to a
model, rather than asserting that some helper was called.
"""

import ast
import unittest
from pathlib import Path

from core.failure_messages import scrub_error_for_llm

ROOT = Path(__file__).resolve().parents[1]


class ScrubErrorForLlmTests(unittest.TestCase):
    """A2 — the value a driver echoes back must not survive into a prompt."""

    def test_conversion_error_value_is_masked(self):
        raw = (
            "[Microsoft][ODBC Driver 18 for SQL Server][SQL Server]Conversion "
            "failed when converting the varchar value 'Lipitor 40mg' to data "
            "type int. (SQLExecDirectW)"
        )
        out = scrub_error_for_llm(raw)
        self.assertNotIn("Lipitor 40mg", out)
        self.assertNotIn("Lipitor", out)
        # The repair signal survives: the model still learns it was a
        # varchar -> int conversion problem.
        self.assertIn("Conversion failed", out)
        self.assertIn("int", out)

    def test_driver_prefix_chain_is_stripped(self):
        raw = "[Microsoft][ODBC Driver 18][SQL Server]Invalid object name 'DBO.NOPE'."
        out = scrub_error_for_llm(raw)
        self.assertNotIn("[Microsoft]", out)
        self.assertNotIn("ODBC Driver 18", out)

    def test_patient_identifier_in_error_is_masked(self):
        raw = (
            "Cannot insert duplicate key row in object 'dbo.PATIENT'. "
            "The duplicate key value is (Priya Raghunathan)."
        )
        out = scrub_error_for_llm(raw)
        self.assertNotIn("Priya Raghunathan", out)

    def test_long_account_number_is_masked(self):
        raw = "Arithmetic overflow converting 90210447182 to data type numeric."
        out = scrub_error_for_llm(raw)
        self.assertNotIn("90210447182", out)

    def test_schema_identifiers_are_preserved(self):
        # SCREAMING_SNAKE identifiers are schema, not data — the repair prompt
        # needs them, and llm_audit's heuristics keep them.
        raw = "Invalid column name 'NET_REVENUE'."
        out = scrub_error_for_llm(raw)
        self.assertIn("NET_REVENUE", out)

    def test_empty_and_none_are_safe(self):
        self.assertEqual(scrub_error_for_llm(""), "")
        self.assertEqual(scrub_error_for_llm(None), "")  # type: ignore[arg-type]

    def test_output_is_length_bounded(self):
        raw = "Conversion failed for value 'x'. " * 200
        self.assertLessEqual(len(scrub_error_for_llm(raw)), 400)


class RepairPromptWiringTests(unittest.TestCase):
    """A2 — both repair prompts must route the error through the scrubber."""

    def test_query_pipeline_repair_prompt_scrubs_the_error(self):
        src = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("scrub_error_for_llm(exec_error)", src)
        self.assertNotIn("f\"Error: {exec_error}\\n\"", src)

    def test_webhooks_exec_retry_scrubs_the_error(self):
        src = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")
        self.assertIn("_scrub_err(_exec_err)", src)
        self.assertNotIn("f\"Error: {_exec_err}\\n\"", src)


class NarrationRemovalTests(unittest.TestCase):
    """The row-serialising narration path must stay gone."""

    def test_generate_result_narration_is_not_defined(self):
        src = (ROOT / "core" / "result_renderer.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        names = {
            n.name
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertNotIn("_generate_result_narration", names)

    def test_result_renderer_has_no_llm_entry_point(self):
        """Structural: the module must not import a way to call a model."""
        src = (ROOT / "core" / "result_renderer.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported.add(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name)
        self.assertNotIn("llm_complete", imported)


class DrillDimBoundaryTests(unittest.TestCase):
    """A5 — drill_dim must be gated like its sibling result actions."""

    def test_drill_dim_checks_result_llm_features_allowed(self):
        src = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")
        start = src.index('if action.startswith("drill_dim:")')
        block = src[start:start + 3000]
        self.assertIn("result_llm_features_allowed(account_id)", block)
        self.assertIn("record_llm_blocked", block)

    def test_blocked_drill_dim_records_proof_inside_an_audit_scope(self):
        """record_llm_blocked no-ops without an ambient scope, so the refusal
        must be wrapped or the proof row is silently never written."""
        src = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")
        start = src.index('if action.startswith("drill_dim:")')
        block = src[start:start + 3000]
        gate = block.index("result_llm_features_allowed(account_id)")
        blocked = block.index("record_llm_blocked")
        scope = block.index("llm_audit_scope", gate)
        self.assertLess(scope, blocked, "record_llm_blocked is outside an audit scope")


class FallbackAuditCoverageTests(unittest.TestCase):
    """The three result-chat fallback SQL calls must all be audited."""

    def _fallback_block(self) -> str:
        src = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")
        start = src.index("_fb_request_id = make_llm_audit_request_id()")
        return src[start:start + 16000]

    def test_all_three_fallback_components_are_scoped(self):
        block = self._fallback_block()
        for component in (
            "result_chat_sql_fallback",
            "result_chat_sql_fallback_retry",
            "result_chat_sql_exec_retry",
        ):
            self.assertIn(f'component="{component}"', block)

    def test_fallback_calls_share_one_request_id(self):
        """Three orphan rows would be unusable — they must group as one question."""
        block = self._fallback_block()
        self.assertEqual(block.count("request_id=_fb_request_id"), 3)

    def test_no_unaudited_llm_complete_in_the_fallback_block(self):
        """Every llm_complete in the block must be preceded by a scope."""
        block = self._fallback_block()
        scopes = block.count("with llm_audit_scope(")
        calls = block.count("await llm_complete(")
        self.assertGreaterEqual(
            scopes, calls, f"{calls} llm_complete calls but only {scopes} audit scopes"
        )


if __name__ == "__main__":
    unittest.main()
