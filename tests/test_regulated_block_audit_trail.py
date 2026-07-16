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


class ResultCacheGetSqlTests(unittest.TestCase):
    """New accessor — needed so the DuckDB follow-up path can re-derive
    which real table.column the cached result actually came from."""

    def test_returns_stored_sql(self):
        from core.result_cache import ResultCache
        cache = ResultCache()
        cache.store("sess-1", [{"a": 1}], question="q", sql="SELECT a FROM t")
        self.assertEqual(cache.get_sql("sess-1"), "SELECT a FROM t")

    def test_returns_empty_string_for_unknown_session(self):
        from core.result_cache import ResultCache
        cache = ResultCache()
        self.assertEqual(cache.get_sql("no-such-session"), "")


class DuckdbCacheBaaRecheckPolicyTests(unittest.TestCase):
    """Exercises the actual policy_engine.evaluate() + analyze_sql()
    combination the new duckdb_cache re-check relies on, without invoking
    the full handle_query() pipeline (impractical to mock end-to-end —
    see AutoImportHookWiringTests for this codebase's established pattern
    of testing large pipeline functions via their component logic +
    static wiring assertions instead)."""

    def _rules(self):
        from admin.routes import _default_regulated_rules
        from core.compliance.packs import get_pack
        pack = get_pack("healthcare_pharmacy_v1")
        return pack, _default_regulated_rules(pack)

    def test_cached_sql_touching_phi_table_without_baa_is_denied(self):
        from unittest.mock import patch
        from core.compliance import policy_engine
        from core.compliance.models import PolicyContext
        from core.compliance.sql_guard import analyze_sql

        pack, rules = self._rules()
        analysis = analyze_sql(
            "SELECT PRESCRIBER_NAME, SUM(REVENUE) AS REVENUE FROM RX.PHARMACY GROUP BY PRESCRIBER_NAME",
            "azure_sql",
        )
        classifications = {
            "RX.PHARMACY.PRESCRIBER_NAME": {"tags": ["PHI", "PII"], "sensitivity": "RESTRICTED"},
            "RX.PHARMACY.REVENUE": {"tags": [], "sensitivity": "INTERNAL"},
        }
        context = PolicyContext(
            account_id="acct-rx", user_id="1", role="analyst",
            purpose_id="patient_care", action="llm_context", policy_version=1,
        )
        profile = {
            "mode": "regulated", "policy_pack_key": "healthcare_pharmacy_v1",
            "active_policy_version": 1, "enforcement_mode": "enforce",
        }
        with (
            patch.object(policy_engine.store, "get_compliance_profile", return_value=profile),
            patch.object(policy_engine.store, "get_classification_map", return_value=classifications),
            patch.object(policy_engine.store, "list_policy_rules", return_value=rules),
            patch.object(policy_engine.store, "list_purposes", return_value=[]),
            patch.object(policy_engine.store, "provider_agreement_valid", return_value=False),
        ):
            decision = policy_engine.evaluate(context, analysis.resources, record=False)
        self.assertFalse(decision.effective_allowed)
        self.assertEqual(decision.reason_code, "provider_agreement_required")

    def test_cached_sql_touching_only_baa_exempt_tagged_columns_is_allowed(self):
        # NOTE: a resource with NO classification tags at all currently has
        # no matching llm_context rule in _default_regulated_rules (its
        # catch-all "*" allow rules cover query_execution/result_release/
        # chart/alert/cache_read but NOT llm_context) and so denies —
        # identical behavior to the pre-existing PRIMARY llm_context check
        # this re-check deliberately mirrors. That's a separate, pre-existing
        # rule-generation gap, not something introduced here. To test the
        # "allowed" path cleanly, use a tag that IS covered (PRESCRIPTION:
        # sensitive but not BAA-gated for healthcare_pharmacy_v1, whose
        # provider_agreement_tags is just ["PHI"]) — this resolves to
        # effect="mask", which still counts as allowed (masked, not denied).
        from unittest.mock import patch
        from core.compliance import policy_engine
        from core.compliance.models import PolicyContext
        from core.compliance.sql_guard import analyze_sql

        pack, rules = self._rules()
        analysis = analyze_sql(
            "SELECT DRUG_NAME, SUM(REVENUE) AS REVENUE FROM RX.SALES GROUP BY DRUG_NAME",
            "azure_sql",
        )
        classifications = {
            "RX.SALES.DRUG_NAME": {"tags": ["PRESCRIPTION"], "sensitivity": "CONFIDENTIAL"},
            "RX.SALES.REVENUE": {"tags": ["PRESCRIPTION"], "sensitivity": "CONFIDENTIAL"},
        }
        context = PolicyContext(
            account_id="acct-rx", user_id="1", role="analyst",
            purpose_id="patient_care", action="llm_context", policy_version=1,
        )
        profile = {
            "mode": "regulated", "policy_pack_key": "healthcare_pharmacy_v1",
            "active_policy_version": 1, "enforcement_mode": "enforce",
        }
        with (
            patch.object(policy_engine.store, "get_compliance_profile", return_value=profile),
            patch.object(policy_engine.store, "get_classification_map", return_value=classifications),
            patch.object(policy_engine.store, "list_policy_rules", return_value=rules),
            patch.object(policy_engine.store, "list_purposes", return_value=[]),
        ):
            decision = policy_engine.evaluate(context, analysis.resources, record=False)
        self.assertTrue(decision.effective_allowed)

    def test_KNOWN_GAP_untagged_resource_denies_llm_context_in_enforce_mode(self):
        """
        Documents a pre-existing gap found while building this re-check, NOT
        introduced by it: _default_regulated_rules' catch-all "*" rules cover
        query_execution/result_release/chart/alert/cache_read but NOT
        llm_context. A resource with zero classification tags (the common
        case — every column gets a classification row at discovery, most
        with tags=[]) has no matching llm_context rule at all, so it denies
        by default (no allow_rule found -> denied=True) once mode=regulated
        AND enforcement_mode=enforce.

        This affects the EXISTING primary llm_context check identically
        (scoped_resources there is built the same way, from every
        classification_map entry in the effective tables, tagged or not),
        not just this new duckdb_cache re-check. Every regulated client
        observed this session had enforcement_mode="shadow" — under shadow,
        effective_allowed = allowed OR shadow = True regardless, which is
        almost certainly why this hasn't been noticed yet. Flagged here as
        a known finding for a separate fix, not addressed by this change.
        """
        from unittest.mock import patch
        from core.compliance import policy_engine
        from core.compliance.models import PolicyContext, ResourceRef

        pack, rules = self._rules()
        context = PolicyContext(
            account_id="acct-rx", user_id="1", role="analyst",
            purpose_id="patient_care", action="llm_context", policy_version=1,
        )
        profile = {
            "mode": "regulated", "policy_pack_key": "healthcare_pharmacy_v1",
            "active_policy_version": 1, "enforcement_mode": "enforce",
        }
        with (
            patch.object(policy_engine.store, "get_compliance_profile", return_value=profile),
            patch.object(
                policy_engine.store, "get_classification_map",
                return_value={"RX.SALES.WAREHOUSE_CODE": {"tags": [], "sensitivity": "INTERNAL"}},
            ),
            patch.object(policy_engine.store, "list_policy_rules", return_value=rules),
            patch.object(policy_engine.store, "list_purposes", return_value=[]),
        ):
            decision = policy_engine.evaluate(
                context, [ResourceRef(table="RX.SALES", column="WAREHOUSE_CODE")], record=False,
            )
        self.assertFalse(decision.effective_allowed, decision.reason_code)


class DuckdbCacheReCheckWiringTests(unittest.TestCase):
    """Static assertions that the re-check is wired into the duckdb_cache
    routing branch, before the LLM call that would embed cached sample
    values — same reasoning as DuckdbCacheBaaRecheckPolicyTests' docstring."""

    def _duckdb_block(self) -> str:
        src = _src("core/query_pipeline.py")
        start = src.index("if _session_id and should_route_to_result_cache(")
        gen_pos = src.index("_duck_sql = await _generate_duckdb_sql(")
        return src[start:gen_pos]

    def test_recheck_gated_on_regulated_mode(self):
        block = self._duckdb_block()
        self.assertIn('compliance_profile.get("mode") == "regulated"', block)

    def test_recheck_uses_cached_sql_not_broad_effective_scope(self):
        block = self._duckdb_block()
        self.assertIn("result_cache.get_sql(_session_id)", block)
        self.assertIn("analyze_sql(_cached_sql", block)
        self.assertIn('action="llm_context"', block)

    def test_denial_writes_blocked_audit_row_and_returns(self):
        block = self._duckdb_block()
        self.assertIn("record_llm_blocked(", block)
        self.assertIn('"duckdb_cache"', block)
        denial_idx = block.index("if not _cache_llm_decision.effective_allowed:")
        after = block[denial_idx:]
        self.assertIn("return", after)
        self.assertIn("blocked by the workspace data policy", after)


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
