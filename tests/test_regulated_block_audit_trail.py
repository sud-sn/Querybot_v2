"""
Proof-of-refusal audit trail for the regulated-tenant LLM boundary.

Before this, result_llm_features_allowed()==False call sites just returned
early — the audit log had NO row for the blocked feature, which looks
identical to "never triggered" rather than "triggered and correctly
refused." record_llm_blocked() writes a genuine llm_call_log row
(status="blocked", reason in the preview column, no prompt ever built) so
an admin/auditor can point at a concrete record instead of an absence.

Covers:
  1. record_llm_blocked — writes when scope is active+enabled, no-ops
     otherwise (mirrors record_llm_call's existing contract).
  2. Each of the four gated call sites writes a blocked row with the
     right component name: generate_analysis_response ("analysis"),
     generate_period_comparison ("compare_prior"),
     generate_followup_suggestions ("followup_suggestions"),
     _send_why_insight (silent to the user, but still audited).
  3. get_recent_llm_calls' any_blocked grouping flag, against a real DB.
  4. Admin template renders a distinct BLOCKED badge (not the red ERROR
     one) and lists the new component names in the filter/glossary.
"""

from __future__ import annotations

import asyncio
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _arun(coro):
    return asyncio.run(coro)


class RecordLlmBlockedTests(unittest.TestCase):
    def test_writes_blocked_row_when_scope_active(self):
        from core.llm_audit import llm_audit_scope, record_llm_blocked

        with patch("store.log_llm_call") as log_call:
            with llm_audit_scope(
                account_id="acct-1", question="why did revenue drop",
                enabled=True, request_id="req1", component="analysis",
            ):
                record_llm_blocked("analysis", "blocked for test reasons")
        log_call.assert_called_once()
        kwargs = log_call.call_args.kwargs
        self.assertEqual(kwargs["status"], "blocked")
        self.assertEqual(kwargs["component"], "analysis")
        self.assertEqual(kwargs["payload_preview_sanitized"], "blocked for test reasons")
        self.assertEqual(kwargs["prompt_chars"], 0)
        self.assertEqual(kwargs["llm_provider"], "")

    def test_noop_without_active_scope(self):
        from core.llm_audit import record_llm_blocked

        with patch("store.log_llm_call") as log_call:
            record_llm_blocked("analysis", "should not be written")
        log_call.assert_not_called()

    def test_noop_when_scope_disabled(self):
        from core.llm_audit import llm_audit_scope, record_llm_blocked

        with patch("store.log_llm_call") as log_call:
            with llm_audit_scope(
                account_id="acct-1", question="q", enabled=False,
                request_id="req1", component="analysis",
            ):
                record_llm_blocked("analysis", "should not be written")
        log_call.assert_not_called()


class GenerateAnalysisResponseBlockedAuditTests(unittest.TestCase):
    def test_regulated_writes_blocked_audit_row(self):
        from core.llm_audit import llm_audit_scope
        from core.response_builder import generate_analysis_response

        with (
            patch(
                "core.compliance.policy_engine.store.get_compliance_profile",
                return_value={"mode": "regulated"},
            ),
            patch("store.log_llm_call") as log_call,
        ):
            with llm_audit_scope(
                account_id="acct-rx", question="explain this",
                enabled=True, request_id="req1", component="analysis",
            ):
                result = _arun(generate_analysis_response(
                    action="explain", rows=[{"a": 1}], question="explain this",
                    provider="azure_openai", model="gpt-4o", api_key="key",
                    account_id="acct-rx",
                ))
        self.assertEqual(result["title"], "Not available for this workspace")
        log_call.assert_called_once()
        self.assertEqual(log_call.call_args.kwargs["status"], "blocked")
        self.assertEqual(log_call.call_args.kwargs["component"], "analysis")


class GeneratePeriodComparisonBlockedAuditTests(unittest.TestCase):
    def test_regulated_writes_blocked_audit_row(self):
        from core.llm_audit import llm_audit_scope
        from core.period_comparison import generate_period_comparison

        with (
            patch(
                "core.compliance.policy_engine.store.get_compliance_profile",
                return_value={"mode": "regulated"},
            ),
            patch("store.log_llm_call") as log_call,
        ):
            with llm_audit_scope(
                account_id="acct-rx", question="compare_prior: q",
                enabled=True, request_id="req1", component="compare_prior",
            ):
                result = _arun(generate_period_comparison(
                    rows=[{"Month": "2025-01", "Revenue": 100}],
                    question="q", original_sql="SELECT 1",
                    data_brief={"mode": "time_series", "time_series": {
                        "first_period": "2025-01", "last_period": "2025-06",
                    }},
                    db_cfg={"db_type": "azure_sql"}, account_id="acct-rx",
                    provider="azure_openai", model="gpt-4o", api_key="key",
                ))
        self.assertEqual(result["action"], "compare_prior")
        log_call.assert_called_once()
        self.assertEqual(log_call.call_args.kwargs["status"], "blocked")
        self.assertEqual(log_call.call_args.kwargs["component"], "compare_prior")


class GenerateFollowupSuggestionsBlockedAuditTests(unittest.TestCase):
    def test_regulated_returns_empty_and_writes_blocked_audit_row(self):
        from core.insight import generate_followup_suggestions

        with (
            patch(
                "core.compliance.policy_engine.store.get_compliance_profile",
                return_value={"mode": "regulated"},
            ),
            patch("store.log_llm_call") as log_call,
        ):
            result = _arun(generate_followup_suggestions(
                brief={"row_count": 5, "columns": {"a": "numeric", "b": "text"}},
                question="q", result_scope={}, db_cfg={}, account_id="acct-rx",
                audit_enabled=True, audit_request_id="req1",
            ))
        self.assertEqual(result, [])
        log_call.assert_called_once()
        self.assertEqual(log_call.call_args.kwargs["status"], "blocked")
        self.assertEqual(log_call.call_args.kwargs["component"], "followup_suggestions")

    def test_standard_tenant_no_blocked_row(self):
        from core.insight import generate_followup_suggestions

        with (
            patch(
                "core.compliance.policy_engine.store.get_compliance_profile",
                return_value={"mode": "standard"},
            ),
            patch("store.log_llm_call") as log_call,
        ):
            _arun(generate_followup_suggestions(
                brief={}, question="q", result_scope={}, db_cfg={}, account_id="acct-std",
                audit_enabled=True, audit_request_id="req1",
            ))
        log_call.assert_not_called()


class SendWhyInsightBlockedAuditTests(unittest.TestCase):
    def test_regulated_stays_silent_but_writes_blocked_audit_row(self):
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
            patch("store.log_llm_call") as log_call,
        ):
            _arun(_send_why_insight(
                adapter, event, question="why did revenue drop",
                rows=[{"a": 1}], sql="SELECT 1", client={"enable_llm_audit": True},
                account_id="acct-rx", db_cfg={},
            ))
        adapter.send_message.assert_not_called()
        adapter.send_analysis_response.assert_not_called()
        log_call.assert_called_once()
        self.assertEqual(log_call.call_args.kwargs["status"], "blocked")
        self.assertEqual(log_call.call_args.kwargs["component"], "analysis")


class AnyBlockedGroupingTests(unittest.TestCase):
    def setUp(self):
        import store
        store.init_db()
        self.store = store
        self.account_id = f"acct-blocked-grouping-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

    def test_group_flags_any_blocked_distinct_from_any_error(self):
        from core.llm_audit import llm_audit_scope, record_llm_blocked

        with llm_audit_scope(
            account_id=self.account_id, question="q", enabled=True,
            request_id="req-blocked-1", question_id="qid-1", component="analysis",
        ):
            record_llm_blocked("analysis", "blocked for grouping test")

        groups = self.store.get_recent_llm_calls(self.account_id, limit=10)
        group = next(g for g in groups if g["question_id"] == "qid-1")
        self.assertEqual(group["any_blocked"], 1)
        self.assertEqual(group["any_error"], 0)
        self.assertEqual(group["calls"][0]["status"], "blocked")

    def test_status_filter_blocked_returns_only_blocked_rows(self):
        from core.llm_audit import llm_audit_scope, record_llm_blocked

        with llm_audit_scope(
            account_id=self.account_id, question="q", enabled=True,
            request_id="req-blocked-2", question_id="qid-2", component="compare_prior",
        ):
            record_llm_blocked("compare_prior", "blocked for filter test")

        groups = self.store.get_recent_llm_calls(self.account_id, limit=10, status="blocked")
        self.assertTrue(all(c["status"] == "blocked" for g in groups for c in g["calls"]))
        self.assertTrue(any(g["question_id"] == "qid-2" for g in groups))


class TemplateRenderingTests(unittest.TestCase):
    def test_blocked_badge_distinct_from_error_badge(self):
        src = _src("admin/templates/client_detail.html")
        # Two real conditional renders (group-level row + per-call row) —
        # the glossary's illustrative badge is a separate, unconditional span.
        self.assertEqual(
            src.count('badge-blue" title="LLM never received the result rows'), 2
        )

    def test_status_dropdown_has_blocked_option(self):
        src = _src("admin/templates/client_detail.html")
        self.assertIn('<option value="blocked"', src)

    def test_component_dropdown_lists_new_components(self):
        src = _src("admin/templates/client_detail.html")
        for comp in ("followup_suggestions", "compare_prior", "result_narration"):
            self.assertIn(comp, src)

    def test_glossary_explains_blocked_status(self):
        src = _src("admin/templates/client_detail.html")
        self.assertIn("proof the safeguard fired", src)


if __name__ == "__main__":
    unittest.main()
