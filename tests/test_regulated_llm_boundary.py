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



# Phase-1 fail-closed default (docs/LLM_EGRESS_PLAN.md A7): an account with no
# compliance_profile row is now treated as regulated. Every test in this module
# is about a PROVISIONED tenant whose mode is varied explicitly, so declare the
# profile as existing; without this they would exercise the unprovisioned path
# instead of the mode they patch.
_profile_exists_patcher = None


def setUpModule():
    global _profile_exists_patcher
    from unittest.mock import patch
    _profile_exists_patcher = patch(
        "core.compliance.policy_engine.store.compliance_profile_exists",
        return_value=True,
    )
    _profile_exists_patcher.start()


def tearDownModule():
    if _profile_exists_patcher is not None:
        _profile_exists_patcher.stop()

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
        """No model call; the summary the reader gets is computed locally.
        It used to be silence, which read as the feature being broken."""
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
            patch("core.query_pipeline.resolve_provider") as mock_provider,
        ):
            _arun(_send_why_insight(
                adapter, event,
                question="why did revenue drop",
                rows=[{"REGION": "North", "NET_AMOUNT": 620.0},
                      {"REGION": "South", "NET_AMOUNT": 240.0},
                      {"REGION": "East", "NET_AMOUNT": 110.0}],
                sql="SELECT 1", client={}, account_id="acct-rx", db_cfg={},
            ))
        mock_gen.assert_not_called()
        mock_provider.assert_not_called()
        adapter.send_message.assert_not_called()
        adapter.send_analysis_response.assert_called_once()
        summary = adapter.send_analysis_response.call_args.args[1]
        self.assertTrue(summary["computed"])
        self.assertEqual(summary["rows_sent_to_llm"], 0)

    def test_regulated_tenant_with_nothing_to_say_gets_no_apology(self):
        """A computed summary with no findings is not sent: after every
        answer, a paragraph explaining why there is none would be noise."""
        from core.query_pipeline import _send_why_insight

        adapter = MagicMock()
        adapter.send_message = AsyncMock()
        adapter.send_analysis_response = AsyncMock()
        with (
            patch(
                "core.compliance.policy_engine.store.get_compliance_profile",
                return_value={"mode": "regulated"},
            ),
            patch(
                "core.response_builder._regulated_analysis_fallback",
                return_value={"type": "assistant_analysis", "computed": False, "body": "no analysis"},
            ),
        ):
            _arun(_send_why_insight(
                adapter, MagicMock(),
                question="why did revenue drop", rows=[{"a": 1}], sql="SELECT 1",
                client={}, account_id="acct-rx", db_cfg={},
            ))
        adapter.send_analysis_response.assert_not_called()
        adapter.send_message.assert_not_called()

    def test_a_one_number_answer_gets_no_empty_summary(self):
        """"What is our total stock?" answers with one number. The computed
        summary of one number is "nothing stands out" -- true, and not worth
        a card after every such answer."""
        from core.query_pipeline import _send_why_insight

        adapter = MagicMock()
        adapter.send_message = AsyncMock()
        adapter.send_analysis_response = AsyncMock()
        with patch(
            "core.compliance.policy_engine.store.get_compliance_profile",
            return_value={"mode": "regulated"},
        ):
            _arun(_send_why_insight(
                adapter, MagicMock(),
                question="what is our total stock", rows=[{"TOTAL_STOCK": 5234.0}],
                sql="SELECT 1", client={}, account_id="acct-rx", db_cfg={},
            ))
        adapter.send_analysis_response.assert_not_called()
        adapter.send_message.assert_not_called()

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
    def _suggest(self, mode: str, signals: list[dict]):
        """The real generate_followup_suggestions with the model replaced by
        one that records being asked; returns (chips, model calls, refusals)."""
        from core import insight

        asked, refused = [], []

        async def _model(*args, **kwargs):
            asked.append(args)
            return '["Show the top 3 regions"]', 1, 1

        brief = {
            "row_count": 3,
            "columns": {"REGION": "text", "NET_AMOUNT": "numeric"},
            "category_breakdown": {"label_column": "REGION", "top_5": [{"label": "North"}]},
        }
        with (
            patch(
                "core.compliance.policy_engine.store.get_compliance_profile",
                return_value={"mode": mode},
            ),
            patch("core.llm.llm_complete", side_effect=_model),
            patch("core.llm.resolve_provider", return_value=("azure_openai", "gpt-4o", "key", {})),
            patch("store.get_client", return_value={}),
            patch("core.llm_audit.record_llm_blocked",
                  side_effect=lambda component, reason: refused.append(component)),
        ):
            chips = _arun(insight.generate_followup_suggestions(
                brief=brief, question="net amount by region", result_scope={},
                db_cfg={"db_type": "azure_sql"}, account_id="acct-fu",
                audit_enabled=True, audit_request_id="rq", signals=signals,
            ))
        return chips, asked, refused

    def test_regulated_tenant_gets_the_statistical_questions_without_the_model(self):
        from core.stat_signals import compute_signals

        signals = compute_signals([
            {"REGION": "North", "NET_AMOUNT": 620.0},
            {"REGION": "South", "NET_AMOUNT": 240.0},
            {"REGION": "East", "NET_AMOUNT": 110.0},
        ])
        chips, asked, refused = self._suggest("regulated", signals)
        self.assertEqual(asked, [])
        self.assertTrue(chips)
        self.assertTrue(all(chip.get("question") for chip in chips))

    def test_regulated_tenant_keeps_a_lone_statistical_question(self):
        """Fewer than three statistical questions is where the model would
        fill the gap. Refusing the model must not throw away the one the
        statistics already found."""
        from core.stat_signals import compute_signals

        [pareto] = [signal for signal in compute_signals([
            {"REGION": "North", "NET_AMOUNT": 620.0},
            {"REGION": "South", "NET_AMOUNT": 240.0},
            {"REGION": "East", "NET_AMOUNT": 110.0},
        ]) if signal["type"] == "pareto"]
        chips, asked, refused = self._suggest("regulated", [pareto])
        self.assertEqual(asked, [])
        self.assertEqual(len(chips), 1)
        self.assertIn("Region", chips[0]["question"])
        self.assertEqual(refused, ["followup_suggestions"])

    def test_regulated_tenant_the_model_tier_is_refused_and_audited(self):
        chips, asked, refused = self._suggest("regulated", [])
        self.assertEqual((chips, asked), ([], []))
        self.assertEqual(refused, ["followup_suggestions"])

    def test_standard_tenant_still_asks_the_model_to_fill_the_gap(self):
        chips, asked, refused = self._suggest("standard", [])
        self.assertEqual(len(asked), 1)
        self.assertEqual(refused, [])

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
        self.assertIn('_t("reply.result_chat.local_note")', block)
        # The claim itself is what matters, and it now has to hold in every
        # language the product ships -- a translation that dropped "no result
        # values" would be a governance claim quietly weakened.
        from core import i18n
        self.assertIn("No result values were sent to the model",
                      i18n.t("reply.result_chat.local_note", lang="en"))
        self.assertIn("Aucune valeur du résultat n'a été envoyée au modèle",
                      i18n.t("reply.result_chat.local_note", lang="fr"))
        self.assertNotIn("_generate_result_narration", block)

        governed = _src("core/governed_result_followup.py")
        self.assertIn('"rows_sent_to_llm": 0', governed)
        self.assertIn('"sample_values_sent_to_llm": 0', governed)


class GenerateAnalysisResponseRegulatedGateTests(unittest.TestCase):
    """generate_analysis_response is the single shared entry point behind
    _send_why_insight, and (previously ungated) the diagnose/standard
    action-button/why-text-detection handlers in webhooks.py."""

    ROWS = [
        {"Customer": "Real Name", "Revenue": 5000},
        {"Customer": "Second Name", "Revenue": 400},
        {"Customer": "Third Name", "Revenue": 300},
        {"Customer": "Fourth Name", "Revenue": 200},
        {"Customer": "Fifth Name", "Revenue": 100},
    ]

    def test_regulated_gets_an_analysis_computed_without_any_llm_call(self):
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
                rows=self.ROWS,
                question="explain this",
                provider="azure_openai", model="gpt-4o", api_key="key",
                account_id="acct-rx",
            ))
        mock_insight.assert_not_called()
        mock_dd.assert_not_called()
        self.assertEqual(result["type"], "assistant_analysis")
        # A regulated tenant used to receive a title reading "Not available
        # for this workspace" and a paragraph explaining why -- the segment
        # this product most wants to serve got a table and an apology. The
        # analysis is now computed from the released rows in-process.
        self.assertTrue(result["computed"])
        self.assertTrue(result["bullets"])
        self.assertEqual(result["rows_sent_to_llm"], 0)
        self.assertTrue(result["evidence_id"])
        self.assertNotIn("only writes SQL queries", result["body"])

    def test_the_computed_analysis_states_what_the_numbers_show(self):
        # Not just non-empty: the leader in this fixture holds 83% of the
        # total, and the summary must say so.
        from core.response_builder import generate_analysis_response

        with patch(
            "core.compliance.policy_engine.store.get_compliance_profile",
            return_value={"mode": "regulated"},
        ):
            result = _arun(generate_analysis_response(
                action="explain", rows=self.ROWS, question="explain this",
                provider="azure_openai", model="gpt-4o", api_key="key",
                account_id="acct-rx",
            ))
        self.assertIn("concentration_leader", result["finding_kinds"])
        self.assertIn("83", result["body"])

    def test_a_result_the_analysers_cannot_read_still_answers(self):
        # An empty result has nothing to compute; the tenant must still get a
        # reply rather than an exception out of the analysis path.
        from core.response_builder import generate_analysis_response

        with patch(
            "core.compliance.policy_engine.store.get_compliance_profile",
            return_value={"mode": "regulated"},
        ):
            result = _arun(generate_analysis_response(
                action="explain", rows=[], question="explain this",
                provider="azure_openai", model="gpt-4o", api_key="key",
                account_id="acct-rx",
            ))
        self.assertEqual(result["type"], "assistant_analysis")
        self.assertTrue(result["body"])
        self.assertEqual(result["rows_sent_to_llm"], 0)

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
