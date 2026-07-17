"""Per-user attestation: internal users with a recorded confidentiality
attestation see unmasked regulated values in query results; everyone else
gets the masked form. Display-side only — the LLM boundary
(result_llm_features_allowed, llm_context gating) never consults this.

Layers under test:
  1. store CRUD — grant / revoke / validity / history listing
  2. execute_governed_query — the actual enforcement choke point, executed
     for real with patched store + run_query (established pattern: see
     DuckdbCacheBaaRecheckPolicyTests in test_regulated_block_audit_trail.py)
  3. admin route + template wiring — static source assertions
"""
from __future__ import annotations

import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


def _src(rel: str) -> str:
    return Path(rel).read_text(encoding="utf-8")


class UserAttestationStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import store
        store.init_db()
        cls.store = store
        cls.account_id = f"acct-attest-{uuid.uuid4().hex[:8]}"
        store.upsert_client(cls.account_id, "portal")

    @classmethod
    def tearDownClass(cls):
        with cls.store.get_db() as conn:
            conn.execute("DELETE FROM user_attestation WHERE account_id=?", (cls.account_id,))
            conn.execute("DELETE FROM client WHERE account_id=?", (cls.account_id,))

    def test_grant_revoke_lifecycle(self):
        user_id = uuid.uuid4().hex[:6]
        self.assertFalse(self.store.user_attestation_valid(self.account_id, user_id))
        att_id = self.store.save_user_attestation(
            self.account_id, user_id,
            attestation_type="hipaa_workforce", document_ref="DMS-42",
            granted_by="admin",
        )
        self.assertTrue(self.store.user_attestation_valid(self.account_id, user_id))
        # Other users and empty ids never validate.
        self.assertFalse(self.store.user_attestation_valid(self.account_id, "someone-else"))
        self.assertFalse(self.store.user_attestation_valid(self.account_id, ""))
        self.assertTrue(self.store.revoke_user_attestation(self.account_id, att_id, "admin"))
        self.assertFalse(self.store.user_attestation_valid(self.account_id, user_id))
        # Revoking twice is a no-op, not an error.
        self.assertFalse(self.store.revoke_user_attestation(self.account_id, att_id, "admin"))

    def test_regrant_after_revoke_creates_new_row_and_validates(self):
        user_id = uuid.uuid4().hex[:6]
        first = self.store.save_user_attestation(self.account_id, user_id)
        self.store.revoke_user_attestation(self.account_id, first)
        second = self.store.save_user_attestation(self.account_id, user_id, document_ref="DMS-99")
        self.assertNotEqual(first, second)
        self.assertTrue(self.store.user_attestation_valid(self.account_id, user_id))
        history = [
            row for row in self.store.list_user_attestations(self.account_id)
            if row["portal_user_id"] == user_id
        ]
        # Full grant/revoke history preserved for the auditor.
        self.assertEqual(len(history), 2)

    def test_attestation_scoped_per_account(self):
        user_id = uuid.uuid4().hex[:6]
        self.store.save_user_attestation(self.account_id, user_id)
        self.assertFalse(self.store.user_attestation_valid("some-other-account", user_id))


class GovernedQueryAttestationTests(unittest.TestCase):
    """Real execution of execute_governed_query's release path with patched
    store + run_query — proves the attested user gets raw values WITH an
    audit row, and everyone else keeps getting masked values."""

    SQL = "SELECT PRESCRIBER_NAME, REVENUE FROM RX.PHARMACY"
    RAW_ROWS = [{"PRESCRIBER_NAME": "Dr. Alice Real", "REVENUE": 1250}]

    def _run(self, *, attested: bool, log_raises: bool = False):
        from admin.routes import _default_regulated_rules
        from core.compliance import governed_query, policy_engine, sql_guard
        from core.compliance.models import PolicyContext
        from core.compliance.packs import get_pack

        # Patch the store object these modules actually hold. Several test
        # modules in this suite del sys.modules["store"] and re-import,
        # splitting `import store` here from the reference core.compliance
        # captured at its own import time — patching a bare `import store`
        # silently patches the wrong module object when the full suite runs.
        # (Same reason DuckdbCacheBaaRecheckPolicyTests patches
        # policy_engine.store.) All three normally share one object; patching
        # each by name stays correct even if they ever diverge.
        stores = {policy_engine.store, governed_query.store, sql_guard.store}

        pack = get_pack("healthcare_pharmacy_v1")
        rules = _default_regulated_rules(pack)
        classifications = {
            "RX.PHARMACY.PRESCRIBER_NAME": {
                "tags": ["PHI", "PII"], "sensitivity": "RESTRICTED",
            },
            "RX.PHARMACY.REVENUE": {"tags": [], "sensitivity": "INTERNAL"},
        }
        profile = {
            "mode": "regulated", "policy_pack_key": "healthcare_pharmacy_v1",
            "active_policy_version": 1, "enforcement_mode": "enforce",
        }
        context = PolicyContext(
            account_id="acct-attest-gq", user_id="7", role="analyst",
            purpose_id="patient_care", action="query_execution", policy_version=1,
        )
        logged: list[dict] = []

        def _fake_log(**kwargs):
            # Only the attestation-release write is failure-injected —
            # evaluate()'s own decision recording must keep working or the
            # whole query dies before reaching the code under test.
            if log_raises and kwargs.get("reason_code") == "attested_unmasked_release":
                raise RuntimeError("audit store unavailable")
            logged.append(kwargs)
            return "audit-id"

        from contextlib import ExitStack
        with ExitStack() as stack:
            for st in stores:
                stack.enter_context(patch.object(st, "get_compliance_profile", return_value=profile))
                stack.enter_context(patch.object(st, "get_classification_map", return_value=classifications))
                stack.enter_context(patch.object(st, "list_policy_rules", return_value=rules))
                stack.enter_context(patch.object(st, "list_purposes", return_value=[]))
                stack.enter_context(patch.object(st, "list_row_policies", return_value=[]))
                stack.enter_context(patch.object(st, "user_attestation_valid", return_value=attested))
                stack.enter_context(patch.object(st, "log_policy_decision", side_effect=_fake_log))
            stack.enter_context(patch.object(governed_query, "run_query",
                                             return_value=[dict(r) for r in self.RAW_ROWS]))
            result = governed_query.execute_governed_query(
                {}, "azure_sql", self.SQL,
                context=context,
                known_tables={"RX.PHARMACY"},
            )
        return result, logged

    def test_unattested_user_gets_masked_values(self):
        result, logged = self._run(attested=False)
        self.assertNotEqual(result.rows[0]["PRESCRIBER_NAME"], "Dr. Alice Real")
        self.assertEqual(result.rows[0]["REVENUE"], 1250)  # untagged column untouched
        self.assertFalse(
            [l for l in logged if l.get("reason_code") == "attested_unmasked_release"]
        )

    def test_attested_user_gets_raw_values_with_audit_row(self):
        result, logged = self._run(attested=True)
        self.assertEqual(result.rows[0]["PRESCRIBER_NAME"], "Dr. Alice Real")
        releases = [l for l in logged if l.get("reason_code") == "attested_unmasked_release"]
        self.assertEqual(len(releases), 1, "every unmasked release must be audit-logged")
        self.assertIn("RX.PHARMACY.PRESCRIBER_NAME", releases[0]["resources"])
        self.assertEqual(releases[0]["user_id"], "7")
        self.assertTrue(releases[0]["allowed"])
        # The waived obligations are preserved in the log for the auditor.
        self.assertIn("masking_waived", releases[0]["obligations"])

    def test_audit_log_failure_falls_back_to_masking(self):
        # An unmasked release without its audit row is worse than a masked
        # result — if the log can't be written, the user sees masked values.
        result, _ = self._run(attested=True, log_raises=True)
        self.assertNotEqual(result.rows[0]["PRESCRIBER_NAME"], "Dr. Alice Real")


class AttestationWiringTests(unittest.TestCase):
    def test_compliance_page_passes_attestations_context(self):
        src = _src("admin/routes.py")
        fn = src[src.index("async def compliance_page("):]
        fn = fn[:fn.index("\n@router.")]
        self.assertIn('"attestations": store.list_user_attestations(account_id)', fn)

    def test_grant_and_revoke_routes_exist(self):
        src = _src("admin/routes.py")
        self.assertIn('@router.post("/clients/{account_id}/compliance/attestation")', src)
        self.assertIn(
            '@router.post("/clients/{account_id}/compliance/attestation/{attestation_id}/revoke")',
            src,
        )

    def test_template_renders_attestation_section(self):
        src = _src("admin/templates/client_compliance.html")
        self.assertIn("User attestations", src)
        self.assertIn("Grant attestation", src)
        self.assertIn("/compliance/attestation", src)
        self.assertIn("attestation_type", src)
        # Revoke is guarded by a confirm dialog — instant re-masking is
        # surprising enough to warrant one.
        self.assertIn("Revoke this attestation?", src)

    def test_enforcement_sits_before_protect_rows(self):
        src = _src("core/compliance/governed_query.py")
        attn_pos = src.index("user_attestation_valid")
        protect_pos = src.index("rows = protect_rows(")
        self.assertLess(attn_pos, protect_pos)
        block = src[attn_pos:protect_pos]
        self.assertIn("attested_unmasked_release", block)
        self.assertIn("log_policy_decision", block)


if __name__ == "__main__":
    unittest.main()
