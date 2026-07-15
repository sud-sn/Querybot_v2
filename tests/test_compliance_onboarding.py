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


if __name__ == "__main__":
    unittest.main()
