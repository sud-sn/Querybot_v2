"""
Compliance onboarding integration — regression tests.

Covers:
  1. compliance_save_profile's `next=setup` redirect (new) vs default
     `/compliance` redirect (unchanged behavior) — real route invocation
     against the shared test DB, no mocks needed since every store call
     it makes is safe against an empty/no-schema client.
  2. The discovery auto-import hook in _do_discover: gated on
     profile.mode == "regulated", positioned after the SCHEMA_READY
     state save, never raises (best-effort).
  3. import_schema_classifications is idempotent — re-running it against
     already-classified columns creates zero new rows and never touches
     admin-reviewed data.
  4. client_setup_page passes compliance_profile into its template
     context; compliance_page passes pack into its template context.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock

from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _arun(coro):
    return asyncio.run(coro)


def _make_post_request(form_data: dict):
    from starlette.datastructures import FormData

    req = MagicMock()
    req.query_params = {}

    async def _form():
        return FormData(form_data)

    req.form = _form
    return req


def _make_json_request(body: dict):
    req = MagicMock()
    req.query_params = {}

    async def _json():
        return body

    req.json = _json
    return req


class ComplianceProfileRedirectTests(unittest.TestCase):
    """compliance_save_profile — real route invocation against the shared
    test DB. Every store call it makes (save_compliance_profile,
    create_policy_version, activate_policy_version, replace_purposes,
    replace_policy_rules, import_schema_classifications,
    import_legacy_masking) is safe against a client with no schema/KB yet."""

    def setUp(self):
        import store
        store.init_db()
        self.account_id = f"acct-compliance-onboard-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

    def test_standard_industry_without_next_redirects_to_compliance(self):
        import admin.routes as routes
        from unittest.mock import patch

        req = _make_post_request({"industry": "standard"})
        with patch.object(routes, "_is_auth", return_value=True):
            result = _arun(routes.compliance_save_profile(req, self.account_id))
        self.assertEqual(result.status_code, 303)
        self.assertIn(f"/admin/clients/{self.account_id}/compliance", result.headers["location"])
        self.assertNotIn("/setup", result.headers["location"])

    def test_standard_industry_with_next_setup_redirects_to_setup(self):
        import admin.routes as routes
        from unittest.mock import patch

        req = _make_post_request({"industry": "standard", "next": "setup"})
        with patch.object(routes, "_is_auth", return_value=True):
            result = _arun(routes.compliance_save_profile(req, self.account_id))
        self.assertEqual(result.status_code, 303)
        self.assertIn(f"/admin/clients/{self.account_id}/setup", result.headers["location"])
        self.assertIn("saved=standard", result.headers["location"])

    def test_regulated_industry_with_next_setup_redirects_to_setup(self):
        import admin.routes as routes
        from unittest.mock import patch

        req = _make_post_request({"industry": "banking", "next": "setup"})
        with patch.object(routes, "_is_auth", return_value=True):
            result = _arun(routes.compliance_save_profile(req, self.account_id))
        self.assertEqual(result.status_code, 303)
        self.assertIn(f"/admin/clients/{self.account_id}/setup", result.headers["location"])
        self.assertIn("saved=profile", result.headers["location"])

        import store
        profile = store.get_compliance_profile(self.account_id)
        self.assertEqual(profile["mode"], "regulated")
        self.assertEqual(profile["industry"], "banking")

    def test_regulated_industry_without_next_redirects_to_compliance(self):
        # Unchanged behavior for the existing Compliance-page-native form.
        import admin.routes as routes
        from unittest.mock import patch

        req = _make_post_request({"industry": "healthcare_pharmacy"})
        with patch.object(routes, "_is_auth", return_value=True):
            result = _arun(routes.compliance_save_profile(req, self.account_id))
        self.assertIn(f"/admin/clients/{self.account_id}/compliance", result.headers["location"])
        self.assertNotIn("/setup", result.headers["location"])

    def test_profile_save_remains_recoverable_when_classification_import_fails(self):
        import admin.routes as routes
        from unittest.mock import patch

        req = _make_post_request({"industry": "healthcare_pharmacy", "next": "setup"})
        with (
            patch.object(routes, "_is_auth", return_value=True),
            patch(
                "core.compliance.classifier.import_schema_classifications",
                side_effect=RuntimeError("malformed schema metadata"),
            ),
        ):
            result = _arun(routes.compliance_save_profile(req, self.account_id))

        self.assertEqual(result.status_code, 303)
        self.assertIn(f"/admin/clients/{self.account_id}/setup", result.headers["location"])
        self.assertIn("saved=profile", result.headers["location"])
        self.assertIn("error=classification_import", result.headers["location"])

        import store
        profile = store.get_compliance_profile(self.account_id)
        self.assertEqual(profile["industry"], "healthcare_pharmacy")
        self.assertEqual(profile["mode"], "regulated")


class AutoImportHookWiringTests(unittest.TestCase):
    """_do_discover's auto-classification hook — static source assertions,
    matching this session's established style for background-task code
    that isn't practically unit-testable in isolation (real discovery
    needs a live DB connection)."""

    def test_hook_exists_gated_on_regulated_mode_after_schema_ready(self):
        src = _src("admin/routes.py")
        anchor = 'save_state(account_id, "SCHEMA_READY", next_state)'
        self.assertIn(anchor, src)
        after = src[src.index(anchor):]
        # The auto-import hook must appear after the state transition and
        # before the (pre-existing) egress-log step that follows it.
        hook_pos = after.index("Auto-import compliance classifications")
        egress_pos = after.index("Egress log (discovery)")
        self.assertLess(hook_pos, egress_pos)
        hook_block = after[hook_pos:egress_pos]
        self.assertIn('_compliance_profile.get("mode") == "regulated"', hook_block)
        self.assertIn("import_schema_classifications(", hook_block)
        self.assertIn("except Exception", hook_block)  # best-effort, never fatal


class ImportSchemaClassificationsIdempotencyTests(unittest.TestCase):
    def test_accepts_legacy_table_lists_and_skips_schema_metadata(self):
        import store
        from core.compliance.classifier import import_schema_classifications

        store.init_db()
        account_id = f"acct-classify-mixed-schema-{uuid.uuid4().hex[:8]}"
        store.upsert_client(account_id, "portal")

        with tempfile.TemporaryDirectory() as schema_dir:
            schema = {
                "PHARMA.D_PATIENT": [
                    {"name": "PATIENT_NAME"},
                    {"name": "MEMBER_ID"},
                ],
                "PHARMA.F_RX_FILL": {
                    "columns": [{"name": "RX_NUMBER"}, {"name": "NET_AMOUNT"}],
                },
                "__db_fk_constraints__": [
                    {
                        "from_table": "PHARMA.F_RX_FILL",
                        "from_column": "PATIENT_ID",
                        "to_table": "PHARMA.D_PATIENT",
                        "to_column": "PATIENT_ID",
                    }
                ],
            }
            (Path(schema_dir) / "_schema.json").write_text(
                json.dumps(schema), encoding="utf-8"
            )

            changed = import_schema_classifications(
                account_id, schema_dir, "healthcare_pharmacy"
            )

        self.assertEqual(changed, 4)
        keys = set(store.get_classification_map(account_id))
        self.assertIn("PHARMA.D_PATIENT.PATIENT_NAME", keys)
        self.assertIn("PHARMA.F_RX_FILL.RX_NUMBER", keys)
        self.assertFalse(any(key.startswith("__DB_FK_CONSTRAINTS__") for key in keys))

    def test_ignores_malformed_non_table_and_non_column_values(self):
        import store
        from core.compliance.classifier import import_schema_classifications

        store.init_db()
        account_id = f"acct-classify-malformed-schema-{uuid.uuid4().hex[:8]}"
        store.upsert_client(account_id, "portal")

        with tempfile.TemporaryDirectory() as schema_dir:
            schema = {
                "PHARMA.GOOD_TABLE": {
                    "columns": [None, "PATIENT_NAME", {"name": "DOCTOR_NAME"}],
                },
                "PHARMA.BAD_TABLE": "not a table definition",
                "__audit__": {"fields_sent": 1},
            }
            (Path(schema_dir) / "_schema.json").write_text(
                json.dumps(schema), encoding="utf-8"
            )

            changed = import_schema_classifications(
                account_id, schema_dir, "healthcare_pharmacy"
            )

        self.assertEqual(changed, 1)
        self.assertIn(
            "PHARMA.GOOD_TABLE.DOCTOR_NAME",
            store.get_classification_map(account_id),
        )

    def test_rerun_does_not_reclassify_existing_columns(self):
        import store
        from core.compliance.classifier import import_schema_classifications

        store.init_db()
        account_id = f"acct-classify-idempotent-{uuid.uuid4().hex[:8]}"
        store.upsert_client(account_id, "portal")

        with tempfile.TemporaryDirectory() as schema_dir:
            schema = {
                "ERP.DIM_CUSTOMER": {
                    "columns": [{"name": "EMAIL"}, {"name": "REGION_CD"}],
                }
            }
            (Path(schema_dir) / "_schema.json").write_text(json.dumps(schema), encoding="utf-8")

            first = import_schema_classifications(account_id, schema_dir, "banking")
            self.assertEqual(first, 2)  # EMAIL + REGION_CD both new

            # Admin reviews EMAIL's classification.
            store.save_classification(
                account_id, "ERP.DIM_CUSTOMER", "EMAIL",
                sensitivity="RESTRICTED", identifiability="DIRECT", tags=["PII"],
                confidence=1.0, reviewed=True, reviewed_by="admin",
                mask_strategy="redact", source="admin",
            )

            second = import_schema_classifications(account_id, schema_dir, "banking")
            self.assertEqual(second, 0, "re-import must not create duplicate/overwriting rows")

            classifications = store.list_classifications(account_id)
            email = next(c for c in classifications if c["column_name"] == "EMAIL")
            self.assertTrue(email["reviewed"], "admin review must survive a re-import")

    def test_returns_zero_when_schema_file_missing(self):
        import store
        from core.compliance.classifier import import_schema_classifications

        store.init_db()
        account_id = f"acct-classify-noschema-{uuid.uuid4().hex[:8]}"
        store.upsert_client(account_id, "portal")
        with tempfile.TemporaryDirectory() as schema_dir:
            result = import_schema_classifications(account_id, schema_dir, "banking")
        self.assertEqual(result, 0)

    def test_rerun_refreshes_unreviewed_stale_classification(self):
        # Root-cause fix: a column first classified with EMPTY tags (before
        # its PII pattern existed, or while the profile was still standard)
        # kept that stale row forever because re-import skipped every
        # existing column — so masking never fired. Re-import must now
        # RE-CLASSIFY unreviewed rows against the current classifier.
        import store
        from core.compliance import classifier
        from core.compliance.classifier import import_schema_classifications

        store.init_db()
        account_id = f"acct-classify-refresh-{uuid.uuid4().hex[:8]}"
        store.upsert_client(account_id, "portal")

        with tempfile.TemporaryDirectory() as schema_dir:
            schema = {"RX.DIM_PRESCRIBER": {"columns": [{"name": "DOCTOR_NAME"}]}}
            (Path(schema_dir) / "_schema.json").write_text(json.dumps(schema), encoding="utf-8")

            # Simulate the stale state: an unreviewed auto row with NO tags,
            # as if classified before "doctor" was a recognized PII pattern.
            store.save_classification(
                account_id, "RX.DIM_PRESCRIBER", "DOCTOR_NAME",
                sensitivity="INTERNAL", identifiability="NONE", tags=[],
                confidence=0.55, reviewed=False, reviewed_by="",
                mask_strategy="redact", source="auto",
            )

            changed = import_schema_classifications(account_id, schema_dir, "healthcare_pharmacy")
            self.assertEqual(changed, 1, "stale unreviewed row must be refreshed")

            cmap = store.get_classification_map(account_id)
            doc = cmap["RX.DIM_PRESCRIBER.DOCTOR_NAME"]
            self.assertIn("PII", doc["tags"])
            self.assertFalse(doc.get("reviewed"))

    def test_rerun_leaves_unchanged_unreviewed_row_untouched(self):
        # No-op when detection is identical — must not churn timestamps or
        # report spurious changes.
        import store
        from core.compliance.classifier import import_schema_classifications

        store.init_db()
        account_id = f"acct-classify-noop-{uuid.uuid4().hex[:8]}"
        store.upsert_client(account_id, "portal")

        with tempfile.TemporaryDirectory() as schema_dir:
            schema = {"RX.DIM_PRESCRIBER": {"columns": [{"name": "DOCTOR_NAME"}]}}
            (Path(schema_dir) / "_schema.json").write_text(json.dumps(schema), encoding="utf-8")

            first = import_schema_classifications(account_id, schema_dir, "healthcare_pharmacy")
            self.assertEqual(first, 1)  # created with PII tag
            second = import_schema_classifications(account_id, schema_dir, "healthcare_pharmacy")
            self.assertEqual(second, 0, "identical detection must be a no-op")

    def test_rerun_never_overwrites_admin_reviewed_row(self):
        # Even if the current classifier would tag it differently, an
        # admin's reviewed decision is authoritative and must survive.
        import store
        from core.compliance.classifier import import_schema_classifications

        store.init_db()
        account_id = f"acct-classify-reviewed-{uuid.uuid4().hex[:8]}"
        store.upsert_client(account_id, "portal")

        with tempfile.TemporaryDirectory() as schema_dir:
            schema = {"RX.DIM_PRESCRIBER": {"columns": [{"name": "DOCTOR_NAME"}]}}
            (Path(schema_dir) / "_schema.json").write_text(json.dumps(schema), encoding="utf-8")

            # Admin deliberately set NO tags and reviewed it (their call).
            store.save_classification(
                account_id, "RX.DIM_PRESCRIBER", "DOCTOR_NAME",
                sensitivity="INTERNAL", identifiability="NONE", tags=[],
                confidence=1.0, reviewed=True, reviewed_by="admin",
                mask_strategy="redact", source="admin",
            )
            changed = import_schema_classifications(account_id, schema_dir, "healthcare_pharmacy")
            self.assertEqual(changed, 0)
            cmap = store.get_classification_map(account_id)
            self.assertEqual(cmap["RX.DIM_PRESCRIBER.DOCTOR_NAME"]["tags"], [])


class ClassificationTagEditTests(unittest.TestCase):
    """Fix 2 — the admin can add/edit a column's tags (which drive mask/deny
    rule matching), correcting a missed auto-classification from the panel."""

    def setUp(self):
        import store
        store.init_db()
        self.store = store
        self.account_id = f"acct-tag-edit-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

    def _post(self, body: dict):
        import admin.routes as routes
        from unittest.mock import patch
        with patch.object(routes, "_is_auth", return_value=True):
            return _arun(routes.compliance_save_classification(
                _make_json_request(body), self.account_id
            ))

    def test_admin_can_add_pii_tag(self):
        result = self._post({
            "table_fqn": "RX.DIM_PRESCRIBER", "column_name": "DOCTOR_NAME",
            "tags": ["PII"], "sensitivity": "RESTRICTED",
            "identifiability": "DIRECT", "mask_strategy": "safe_alias_name",
            "reviewed": True,
        })
        self.assertEqual(json.loads(result.body)["ok"], True)
        cmap = self.store.get_classification_map(self.account_id)
        doc = cmap["RX.DIM_PRESCRIBER.DOCTOR_NAME"]
        self.assertIn("PII", doc["tags"])
        self.assertTrue(doc["reviewed"])
        self.assertEqual(doc["mask_strategy"], "safe_alias_name")

    def test_unknown_tag_rejected(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            self._post({
                "table_fqn": "RX.T", "column_name": "C",
                "tags": ["NONSENSE"], "sensitivity": "INTERNAL",
                "identifiability": "NONE", "mask_strategy": "redact",
            })
        self.assertEqual(ctx.exception.status_code, 422)

    def test_template_renders_editable_tag_checkboxes(self):
        src = _src("admin/templates/client_compliance.html")
        self.assertIn("data-tag", src)
        self.assertIn("[data-tag]:checked", src)


class TemplateContextWiringTests(unittest.TestCase):
    def test_setup_page_passes_compliance_profile(self):
        src = _src("admin/routes.py")
        fn = src[src.index("async def client_setup_page("):]
        fn = fn[:fn.index("\n@router.")]
        self.assertIn("compliance_profile = store.get_compliance_profile(account_id)", fn)
        self.assertIn('"compliance_profile":   compliance_profile', fn)

    def test_compliance_page_passes_pack(self):
        src = _src("admin/routes.py")
        fn = src[src.index("async def compliance_page("):]
        fn = fn[:fn.index("\n@router.")]
        self.assertIn("get_pack(profile.get", fn)
        self.assertIn('"pack": get_pack(', fn)

    def test_setup_template_has_industry_step_and_next_field(self):
        src = _src("admin/templates/client_setup.html")
        self.assertIn("Regulated industry", src)
        self.assertIn('name="next" value="setup"', src)
        self.assertIn('action="/admin/clients/{{ client.account_id }}/compliance/profile"', src)

    def test_compliance_template_has_ordered_checklist(self):
        src = _src("admin/templates/client_compliance.html")
        self.assertIn("readiness-checklist", src)
        self.assertIn("_criticals_met", src)
        # Activate button must reference the criticals gate.
        activate_idx = src.index("Activate regulated enforcement")
        preceding = src[max(0, activate_idx - 400):activate_idx]
        self.assertIn("_criticals_met", preceding)


class DefaultRegulatedRulesMaskStrategyTests(unittest.TestCase):
    """_default_regulated_rules — mask_strategy must only be set on rules
    that actually mask. policy_engine.evaluate() treats ANY truthy
    mask_strategy as a signal to redact, even on an effect="allow" rule —
    setting it unconditionally masked every classification tag regardless
    of whether the pack's sensitive_tags actually called for it (e.g.
    PAYMENT for healthcare_pharmacy_v1, which isn't in sensitive_tags and
    was meant to render as effect="allow")."""

    def _rules_by_tag_action(self, rules, tag, action, role="analyst"):
        return next(
            r for r in rules
            if r["resource_pattern"] == tag and r["action"] == action and r["subject_id"] == role
        )

    def test_non_sensitive_tag_gets_allow_without_mask_strategy(self):
        from admin.routes import _default_regulated_rules
        from core.compliance.packs import get_pack

        pack = get_pack("healthcare_pharmacy_v1")
        self.assertNotIn("PAYMENT", pack["sensitive_tags"])
        rules = _default_regulated_rules(pack)
        rule = self._rules_by_tag_action(rules, "PAYMENT", "query_execution")
        self.assertEqual(rule["effect"], "allow")
        self.assertNotIn("mask_strategy", rule)

    def test_sensitive_tag_still_gets_mask_and_strategy(self):
        from admin.routes import _default_regulated_rules
        from core.compliance.packs import get_pack

        pack = get_pack("healthcare_pharmacy_v1")
        self.assertIn("PHI", pack["sensitive_tags"])
        rules = _default_regulated_rules(pack)
        rule = self._rules_by_tag_action(rules, "PHI", "query_execution")
        self.assertEqual(rule["effect"], "mask")
        self.assertEqual(rule["mask_strategy"], "redact")

    def test_sensitive_financial_tag_keeps_partial_strategy(self):
        from admin.routes import _default_regulated_rules
        from core.compliance.packs import get_pack

        pack = get_pack("banking_v1")
        self.assertIn("PCI", pack["sensitive_tags"])
        rules = _default_regulated_rules(pack)
        rule = self._rules_by_tag_action(rules, "PCI", "query_execution")
        self.assertEqual(rule["effect"], "mask")
        self.assertEqual(rule["mask_strategy"], "redact")

    def _evaluate(self, *, purpose_id, resources, classifications, pack, rules):
        from unittest.mock import patch
        from core.compliance import policy_engine
        from core.compliance.models import PolicyContext

        profile = {
            "mode": "regulated", "policy_pack_key": pack["key"],
            "active_policy_version": 1, "enforcement_mode": "enforce",
        }
        context = PolicyContext(
            account_id="rx-a", user_id="1", role="analyst",
            purpose_id=purpose_id, action="query_execution", policy_version=1,
        )
        with (
            patch.object(policy_engine.store, "get_compliance_profile", return_value=profile),
            patch.object(policy_engine.store, "get_classification_map", return_value=classifications),
            patch.object(policy_engine.store, "list_policy_rules", return_value=rules),
            patch.object(policy_engine.store, "list_purposes", return_value=[]),
        ):
            return policy_engine.evaluate(context, resources, record=False)

    def test_end_to_end_non_sensitive_column_renders_unmasked(self):
        # Full evaluate() + protect_rows() pass for a healthcare_pharmacy_v1
        # tenant: a PAYMENT-tagged column, under the purpose that actually
        # grants access to it, comes back with its real value — not masked.
        from admin.routes import _default_regulated_rules
        from core.compliance.models import ResourceRef
        from core.compliance.packs import get_pack
        from core.compliance.result_guard import protect_rows

        pack = get_pack("healthcare_pharmacy_v1")
        rules = _default_regulated_rules(pack)
        decision = self._evaluate(
            purpose_id="pharmacy_operations",
            resources=[ResourceRef("RX.PHARMACY", "SALES_AMOUNT")],
            classifications={"RX.PHARMACY.SALES_AMOUNT": {"tags": ["PAYMENT"], "sensitivity": "CONFIDENTIAL"}},
            pack=pack, rules=rules,
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.masking, {})

        rows = [{"sales_amt": 4567.89}]
        lineage = {"sales_amt": ["RX.PHARMACY.SALES_AMOUNT"]}
        protected = protect_rows(rows, decision, lineage, account_id="rx-a")
        self.assertEqual(protected[0]["sales_amt"], 4567.89)

    def test_end_to_end_sensitive_column_still_masked(self):
        # Same pack/pipeline, but a PHI-tagged column under the purpose
        # that grants PHI access still comes back redacted — the fix must
        # not weaken masking for genuinely sensitive tags.
        from admin.routes import _default_regulated_rules
        from core.compliance.models import ResourceRef
        from core.compliance.packs import get_pack
        from core.compliance.result_guard import protect_rows

        pack = get_pack("healthcare_pharmacy_v1")
        rules = _default_regulated_rules(pack)
        decision = self._evaluate(
            purpose_id="patient_care",
            resources=[ResourceRef("RX.PHARMACY", "PRESCRIBER_NAME")],
            classifications={"RX.PHARMACY.PRESCRIBER_NAME": {"tags": ["PHI"], "sensitivity": "RESTRICTED"}},
            pack=pack, rules=rules,
        )
        self.assertTrue(decision.allowed)
        self.assertIn("RX.PHARMACY.PRESCRIBER_NAME", decision.masking)

        rows = [{"prescriber": "Dr. Jane Smith"}]
        lineage = {"prescriber": ["RX.PHARMACY.PRESCRIBER_NAME"]}
        protected = protect_rows(rows, decision, lineage, account_id="rx-a")
        self.assertEqual(protected[0]["prescriber"], "[REDACTED]")


if __name__ == "__main__":
    unittest.main()
