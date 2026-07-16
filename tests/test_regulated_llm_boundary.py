"""
Regulated-industry LLM boundary.

For any regulated tenant, the LLM's only job is writing SQL from
schema/sample-value context (still gated by the existing llm_context/BAA
check). Result narration, follow-up suggestions, and why-insights all hand
real query result rows to the LLM as a *second* call after SQL generation,
and none of them go through evaluate()'s per-resource masking — they rely
entirely on column-name pattern matching having caught every sensitive
field. This must be blocked unconditionally for regulated tenants,
independent of BAA status or enforcement_mode (shadow vs enforce) — a
signed agreement covers legal liability, not minimum-necessary exposure.

Covers:
  1. core.compliance.policy_engine.result_llm_features_allowed — the shared
     gate, and that it is independent of enforcement_mode/agreements.
  2. _send_why_insight (core/query_pipeline.py) — real async invocation,
     LLM call skipped for regulated, still made for standard.
  3. Follow-up suggestions gate wired into _send_results
     (core/result_renderer.py) — static source assertion (the surrounding
     function has too many branches/dependencies to exercise end-to-end).
  4. Result-chat narration gate wired into the WebSocket handler
     (gateway/webhooks.py) — static source assertion, same reasoning.
"""

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _arun(coro):
    return asyncio.run(coro)


class ResultLlmFeaturesAllowedTests(unittest.TestCase):
    def test_regulated_mode_returns_false(self):
        from core.compliance.policy_engine import result_llm_features_allowed
        with patch(
            "core.compliance.policy_engine.store.get_compliance_profile",
            return_value={"mode": "regulated"},
        ):
            self.assertFalse(result_llm_features_allowed("acct-1"))

    def test_standard_mode_returns_true(self):
        from core.compliance.policy_engine import result_llm_features_allowed
        with patch(
            "core.compliance.policy_engine.store.get_compliance_profile",
            return_value={"mode": "standard"},
        ):
            self.assertTrue(result_llm_features_allowed("acct-1"))

    def test_unconditional_regardless_of_enforcement_mode_or_agreements(self):
        # This is the whole point of the design: a BAA on file and/or
        # enforce mode must NOT reopen these LLM call sites.
        from core.compliance.policy_engine import result_llm_features_allowed
        with patch(
            "core.compliance.policy_engine.store.get_compliance_profile",
            return_value={"mode": "regulated", "enforcement_mode": "enforce"},
        ):
            self.assertFalse(result_llm_features_allowed("acct-1"))

    def test_exported_from_compliance_package(self):
        from core.compliance import result_llm_features_allowed
        self.assertTrue(callable(result_llm_features_allowed))


class WhyInsightRegulatedGateTests(unittest.TestCase):
    def test_skips_llm_call_for_regulated_tenant(self):
        from core.query_pipeline import _send_why_insight

        adapter = MagicMock()
        adapter.send_message = AsyncMock()
        adapter.send_analysis_response = AsyncMock()
        event = MagicMock()

        with (
            patch(
                "core.compliance.policy_engine.store.get_compliance_profile",
                return_value={"mode": "regulated"},
            ),
            patch(
                "core.response_builder.generate_analysis_response",
                new_callable=AsyncMock,
            ) as mock_gen,
        ):
            _arun(_send_why_insight(
                adapter, event,
                question="why did revenue drop", rows=[{"a": 1}], sql="SELECT 1",
                client={}, account_id="acct-rx", db_cfg={},
            ))
        mock_gen.assert_not_called()
        adapter.send_message.assert_not_called()
        adapter.send_analysis_response.assert_not_called()

    def test_still_calls_llm_for_standard_tenant(self):
        from core.query_pipeline import _send_why_insight

        adapter = MagicMock()
        adapter.send_analysis_response = AsyncMock()
        event = MagicMock()

        with (
            patch(
                "core.compliance.policy_engine.store.get_compliance_profile",
                return_value={"mode": "standard"},
            ),
            patch(
                "core.query_pipeline.resolve_provider",
                return_value=("azure_openai", "gpt-4o", "key", {}),
            ),
            patch(
                "core.response_builder.generate_analysis_response",
                new_callable=AsyncMock,
                return_value={"headline": "Revenue fell"},
            ) as mock_gen,
        ):
            _arun(_send_why_insight(
                adapter, event,
                question="why did revenue drop", rows=[{"a": 1}], sql="SELECT 1",
                client={}, account_id="acct-std", db_cfg={},
            ))
        mock_gen.assert_called_once()
        adapter.send_analysis_response.assert_called_once()

    def test_gate_precedes_llm_audit_scope_and_provider_resolution(self):
        src = _src("core/query_pipeline.py")
        fn = src[src.index("async def _send_why_insight("):]
        fn = fn[:fn.index("\n\n\n")]
        gate_pos = fn.index("result_llm_features_allowed(account_id)")
        try_pos = fn.index("try:")
        self.assertLess(gate_pos, try_pos, "gate must run before any provider/LLM setup")


class FollowUpSuggestionsRegulatedGateTests(unittest.TestCase):
    def test_gate_wired_into_send_results(self):
        src = _src("core/result_renderer.py")
        fn = src[src.index("async def _send_results("):]
        gen_pos = fn.index("await generate_followup_suggestions(")
        gate_pos = fn.index("result_llm_features_allowed(account_id)")
        self.assertLess(gate_pos, gen_pos, "gate must be checked before the LLM call")
        # The gate must sit in the same conditional that guards the call,
        # not just appear somewhere earlier in the function.
        guard_if = fn.rindex("if ", 0, gen_pos)
        self.assertLess(guard_if, gate_pos)


class ResultChatNarrationRegulatedGateTests(unittest.TestCase):
    def test_gate_wired_into_websocket_handler(self):
        src = _src("gateway/webhooks.py")
        narration_pos = src.index("_rc_narration = (")
        block = src[narration_pos:narration_pos + 400]
        self.assertIn("result_llm_features_allowed(account_id)", block)
        self.assertIn("await _generate_result_narration(", block)
        # Regulated path must fall back to empty, not raise or hang.
        self.assertIn('else ""', block)


if __name__ == "__main__":
    unittest.main()
