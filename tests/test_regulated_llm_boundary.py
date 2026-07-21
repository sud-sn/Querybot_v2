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
  5. generate_analysis_response (core/response_builder.py) — the shared
     entry point behind ALL FOUR "why"/explain/analyze/compare/diagnose
     call sites (query_pipeline.py's _send_why_insight, and three WS action
     handlers in webhooks.py that were previously ungated entirely). Real
     async invocation: regulated returns the static fallback with no LLM
     call, standard still calls the LLM.
  6. generate_period_comparison (core/period_comparison.py) — the
     "compare_prior" WS action, previously entirely ungated. Real async
     invocation: regulated returns the fallback before even the SQL-rewrite
     step, standard proceeds normally.
  7. Source-wiring checks confirming account_id=account_id reaches every
     production call site of generate_analysis_response and
     generate_period_comparison (gateway/webhooks.py, core/query_pipeline.py)
     — these are the sites that were found ungated during a full-codebase
     audit of every llm_complete/generate_analysis_response call site.
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
    def test_gate_wired_into_generate_followup_suggestions(self):
        # Moved from the _send_results call site into the shared function
        # itself (matching generate_analysis_response's pattern) so the
        # block gets audit coverage from the function's own llm_audit_scope.
        src = _src("core/insight.py")
        start = src.index("async def generate_followup_suggestions(")
        fn = src[start:start + 2000]
        gate_pos = fn.index("result_llm_features_allowed(account_id)")
        brief_check_pos = fn.index('if not brief:')
        self.assertLess(gate_pos, brief_check_pos, "gate must run before anything else")
        self.assertIn("record_llm_blocked(", fn[:brief_check_pos])

    def test_send_results_no_longer_duplicates_the_gate(self):
        src = _src("core/result_renderer.py")
        fn = src[src.index("async def _send_results("):]
        self.assertNotIn("result_llm_features_allowed(account_id)", fn)


class ResultChatNarrationBoundaryTests(unittest.TestCase):
    def test_cached_result_rows_are_never_sent_for_narration(self):
        src = _src("gateway/webhooks.py")
        start = src.index('if msg_type == "result_chat":')
        end = src.index("# The metadata-only cache engine cannot answer", start)
        block = src[start:end]
        self.assertIn("run_governed_result_followup(", block)
        self.assertIn('"trust": _rc_followup.evidence', block)
        self.assertIn("No result values were sent to the model", block)
        self.assertNotIn("_generate_result_narration", block)

        governed = _src("core/governed_result_followup.py")
        self.assertIn('"rows_sent_to_llm": 0', governed)
        self.assertIn('"sample_values_sent_to_llm": 0', governed)


class GenerateAnalysisResponseRegulatedGateTests(unittest.TestCase):
    """generate_analysis_response is the single shared entry point behind
    _send_why_insight, and (previously ungated) the diagnose/standard
    action-button/why-text-detection handlers in webhooks.py."""

    def test_regulated_returns_static_fallback_no_llm_call(self):
        from core.response_builder import generate_analysis_response

        with (
            patch(
                "core.compliance.policy_engine.store.get_compliance_profile",
                return_value={"mode": "regulated"},
            ),
            patch("core.insight.generate_insight", new_callable=AsyncMock) as mock_insight,
            patch("core.insight.generate_drilldown_insight", new_callable=AsyncMock) as mock_dd,
        ):
            result = _arun(generate_analysis_response(
                action="explain",
                rows=[{"Customer": "Real Name", "Revenue": 5000}],
                question="explain this",
                provider="azure_openai", model="gpt-4o", api_key="key",
                account_id="acct-rx",
            ))
        mock_insight.assert_not_called()
        mock_dd.assert_not_called()
        self.assertEqual(result["type"], "assistant_analysis")
        self.assertIn("only writes SQL queries", result["body"])

    def test_standard_tenant_still_calls_llm(self):
        from core.response_builder import generate_analysis_response

        with (
            patch(
                "core.compliance.policy_engine.store.get_compliance_profile",
                return_value={"mode": "standard"},
            ),
            patch(
                "core.insight.generate_insight", new_callable=AsyncMock,
                return_value={"type": "assistant_analysis", "action": "explain"},
            ) as mock_insight,
        ):
            result = _arun(generate_analysis_response(
                action="explain",
                rows=[{"Customer": "Real Name", "Revenue": 5000}],
                question="explain this",
                provider="azure_openai", model="gpt-4o", api_key="key",
                account_id="acct-std",
            ))
        mock_insight.assert_called_once()
        self.assertEqual(result["action"], "explain")

    def test_account_id_reaches_every_production_call_site(self):
        # query_pipeline.py's _send_why_insight
        src = _src("core/query_pipeline.py")
        fn = src[src.index("async def _send_why_insight("):]
        fn = fn[:fn.index("\n\n\n")]
        self.assertIn("account_id=account_id,", fn)

        # webhooks.py's three previously-ungated call sites
        whsrc = _src("gateway/webhooks.py")
        occurrences = [
            m for m in range(len(whsrc))
            if whsrc.startswith("await generate_analysis_response(", m)
        ]
        self.assertEqual(len(occurrences), 3, "expected diagnose + standard-actions + why-text call sites")
        for start in occurrences:
            block = whsrc[start:start + 600]
            self.assertIn("account_id=account_id,", block)


class GeneratePeriodComparisonRegulatedGateTests(unittest.TestCase):
    def test_regulated_returns_fallback_before_any_llm_or_db_call(self):
        from core.period_comparison import generate_period_comparison

        with (
            patch(
                "core.compliance.policy_engine.store.get_compliance_profile",
                return_value={"mode": "regulated"},
            ),
            patch("core.llm.llm_complete", new_callable=AsyncMock) as mock_llm,
        ):
            result = _arun(generate_period_comparison(
                rows=[{"Month": "2025-01", "Revenue": 100}],
                question="Show revenue by month",
                original_sql="SELECT 1",
                data_brief={
                    "mode": "time_series",
                    "time_series": {
                        "first_period": "2025-01", "last_period": "2025-06",
                        "direction": "increasing", "period_count": 6,
                    },
                },
                db_cfg={"db_type": "azure_sql"},
                account_id="acct-rx",
                provider="azure_openai", model="gpt-4o", api_key="key",
            ))
        mock_llm.assert_not_called()
        self.assertEqual(result["action"], "compare_prior")
        self.assertIn("only writes SQL queries", result["body"])

    def test_account_id_reaches_webhooks_call_site(self):
        src = _src("gateway/webhooks.py")
        start = src.index("_cp_result = await generate_period_comparison(")
        block = src[start:start + 500]
        self.assertIn("account_id=account_id,", block)


if __name__ == "__main__":
    unittest.main()
