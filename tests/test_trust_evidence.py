"""
Trust Evidence panel — monitoring/proof surface for the regulated LLM boundary.

Covers:
  1. store.get_llm_trust_summary — status counts, result-derived component
     split (the headline "result-carrying LLM calls sent: 0" attestation
     number), last-blocked timestamp; real-DB test via record_llm_blocked /
     log_llm_call so the counts come from the same write paths production
     uses.
  2. store.get_policy_decision_counts — allowed/denied counts from the
     hash-chained policy_decision_log.
  3. RESULT_LLM_COMPONENTS stays in sync with the components the runtime
     actually uses for result-derived calls.
  4. compliance_page passes trust context; template renders the panel with
     the attestation, refusal link, and shadow-mode warning.
"""

from __future__ import annotations

import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


class LlmTrustSummaryTests(unittest.TestCase):
    def setUp(self):
        import store
        store.init_db()
        self.store = store
        self.account_id = f"acct-trust-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

    def _log(self, component: str, status: str):
        self.store.log_llm_call(
            account_id=self.account_id, request_id=uuid.uuid4().hex[:12],
            question="q", component=component, llm_provider="azure_openai",
            llm_model="gpt-4o", status=status, payload_hash="",
            payload_preview_sanitized="preview", prompt_chars=10,
        )

    def test_counts_split_by_status_and_result_components(self):
        self._log("sql_generation", "success")
        self._log("sql_generation", "error")
        self._log("analysis", "blocked")
        self._log("compare_prior", "blocked")
        self._log("followup_suggestions", "blocked")

        trust = self.store.get_llm_trust_summary(self.account_id, days=30)
        self.assertEqual(trust["total"], 5)
        self.assertEqual(trust["success"], 1)
        self.assertEqual(trust["error"], 1)
        self.assertEqual(trust["blocked"], 3)
        # The attestation number: no result-derived component ever succeeded.
        self.assertEqual(trust["result_llm_success"], 0)
        self.assertEqual(trust["result_llm_blocked"], 3)
        self.assertTrue(trust["last_blocked_at"])

    def test_result_component_success_is_flagged(self):
        # If a result-derived call ever DID succeed (e.g. pre-regulated
        # history), the attestation number must reflect it honestly.
        self._log("result_narration", "success")
        trust = self.store.get_llm_trust_summary(self.account_id, days=30)
        self.assertEqual(trust["result_llm_success"], 1)

    def test_sql_generation_success_does_not_count_as_result_derived(self):
        self._log("sql_generation", "success")
        trust = self.store.get_llm_trust_summary(self.account_id, days=30)
        self.assertEqual(trust["result_llm_success"], 0)

    def test_empty_account_returns_zeroes(self):
        trust = self.store.get_llm_trust_summary(self.account_id, days=30)
        self.assertEqual(trust["total"], 0)
        self.assertEqual(trust["blocked"], 0)
        self.assertEqual(trust["result_llm_success"], 0)
        self.assertEqual(trust["last_blocked_at"], "")

    def test_blocked_rows_from_record_llm_blocked_are_counted(self):
        # End-to-end through the same write path the runtime gates use.
        from core.llm_audit import llm_audit_scope, record_llm_blocked
        with llm_audit_scope(
            account_id=self.account_id, question="why did revenue drop",
            enabled=True, request_id="req-t1", component="analysis",
        ):
            record_llm_blocked("analysis", "blocked — regulated tenant")
        trust = self.store.get_llm_trust_summary(self.account_id, days=30)
        self.assertEqual(trust["blocked"], 1)
        self.assertEqual(trust["result_llm_blocked"], 1)


class ResultComponentsSyncTests(unittest.TestCase):
    def test_components_cover_every_runtime_result_component(self):
        # Every component name used by the gated call sites must be in
        # RESULT_LLM_COMPONENTS, or the attestation silently undercounts.
        from store import RESULT_LLM_COMPONENTS
        for expected in ("analysis", "diagnose", "compare_prior",
                         "result_narration", "followup_suggestions",
                         "analysis_narrative", "drilldown_planner"):
            self.assertIn(expected, RESULT_LLM_COMPONENTS, expected)
        # sql_generation must NEVER be counted as result-derived — it's the
        # one LLM call regulated tenants legitimately make.
        self.assertNotIn("sql_generation", RESULT_LLM_COMPONENTS)
        self.assertNotIn("duckdb_cache", RESULT_LLM_COMPONENTS)


class PolicyDecisionCountsTests(unittest.TestCase):
    def setUp(self):
        import store
        store.init_db()
        self.store = store
        self.account_id = f"acct-dec-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

    def test_counts_allowed_and_denied(self):
        for allowed in (True, True, False):
            self.store.log_policy_decision(
                account_id=self.account_id, user_id="1",
                action="query_execution", purpose_id="patient_care",
                channel="portal", allowed=allowed, reason_code="test",
                resources=["T.C"], obligations={}, policy_version=1,
            )
        counts = self.store.get_policy_decision_counts(self.account_id, days=30)
        self.assertEqual(counts["allowed"], 2)
        self.assertEqual(counts["denied"], 1)


class TrustPanelWiringTests(unittest.TestCase):
    def test_compliance_page_passes_trust_context(self):
        src = _src("admin/routes.py")
        fn = src[src.index("async def compliance_page("):]
        fn = fn[:fn.index("\n@router.")]
        self.assertIn('"trust": store.get_llm_trust_summary(account_id', fn)
        self.assertIn('"decision_counts": store.get_policy_decision_counts(account_id', fn)
        self.assertIn('"egress_summary": store.get_kb_egress_summary(account_id)', fn)
        self.assertIn('"llm_audit_enabled"', fn)

    def test_template_renders_trust_panel(self):
        src = _src("admin/templates/client_compliance.html")
        self.assertIn("('trust','Trust evidence')", src)
        self.assertIn('id="panel-trust"', src)
        self.assertIn("result-carrying LLM calls sent", src)
        self.assertIn("audit_status=blocked", src)          # link to BLOCKED proof rows
        self.assertIn("Shadow — logging only, not blocking", src)
        self.assertIn("not a regulatory certification", src)  # honest scope caveat

    def test_llm_context_in_catch_all_rules(self):
        # The rule-generation fix that makes enforce mode usable at all.
        from admin.routes import _default_regulated_rules
        from core.compliance.packs import get_pack
        rules = _default_regulated_rules(get_pack("healthcare_pharmacy_v1"))
        catch_all_llm = [
            r for r in rules
            if r["resource_pattern"] == "*" and r["action"] == "llm_context"
            and r["effect"] == "allow"
        ]
        self.assertEqual(len(catch_all_llm), 2, "one per role (admin, analyst)")
        # Order (within each role — evaluate() filters by subject first):
        # per-tag llm_context rules must come before that role's catch-all
        # so tagged resources keep hitting mask/deny first.
        for role in ("admin", "analyst"):
            role_rules = [r for r in rules if r["subject_id"] == role
                          and r["action"] == "llm_context"]
            catch_all_idx = next(
                i for i, r in enumerate(role_rules) if r["resource_pattern"] == "*"
            )
            last_tagged_idx = max(
                i for i, r in enumerate(role_rules) if r["resource_pattern"] != "*"
            )
            self.assertGreater(catch_all_idx, last_tagged_idx, role)


if __name__ == "__main__":
    unittest.main()
