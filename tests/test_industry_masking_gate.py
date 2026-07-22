"""
Industry-aware auto-masking + review gate before KB generation.

Covers:
  1. detect_sensitive_columns(columns, industry=) — healthcare/banking catch
     categories the generic PII patterns miss; standard/empty is unaffected
     (regression guard against get_compliance_profile's default
     industry="standard" silently triggering classification for every
     unconfigured client).
  2. masking/confirm route stamps masking_reviewed_at; admin_save_kb_tables
     and compliance_save_profile (both branches) clear it.
  3. client_setup.html: Discover button gating reflects regulated + review
     state; standard clients are never gated.
"""

from __future__ import annotations

import asyncio
import json
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from starlette.datastructures import FormData

ROOT = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _arun(coro):
    return asyncio.run(coro)


def _post_json(body: dict):
    req = MagicMock()
    req.query_params = {}

    async def _json():
        return body

    req.json = _json
    return req


def _post_form(form: dict):
    req = MagicMock()
    req.query_params = {}

    async def _form():
        return FormData(form)

    req.form = _form
    return req


class DetectSensitiveColumnsIndustryTests(unittest.TestCase):
    def test_standard_and_empty_industry_unaffected(self):
        from core.masking import detect_sensitive_columns
        cols = [{"name": "DIAGNOSIS_CD", "type": "varchar(10)"}]
        self.assertEqual(detect_sensitive_columns(cols), {})
        self.assertEqual(detect_sensitive_columns(cols, industry=""), {})
        self.assertEqual(detect_sensitive_columns(cols, industry="standard"), {},
                         "get_compliance_profile's default industry='standard' must never "
                         "silently enable classification for an unconfigured client")

    def test_healthcare_catches_categories_generic_patterns_miss(self):
        from core.masking import detect_sensitive_columns
        for col_name in ("DIAGNOSIS_CD", "NDC_CODE", "RX_DOSAGE"):
            cols = [{"name": col_name, "type": "varchar(20)"}]
            self.assertEqual(detect_sensitive_columns(cols), {}, f"{col_name} should be a clean miss without industry")
            detected = detect_sensitive_columns(cols, industry="healthcare_pharmacy")
            self.assertIn(col_name, detected, f"{col_name} should be caught with healthcare_pharmacy industry")

    def test_banking_catches_and_excludes_correctly(self):
        from core.masking import detect_sensitive_columns
        banking_col = [{"name": "KYC_RISK_RATING", "type": "varchar(20)"}]
        self.assertEqual(detect_sensitive_columns(banking_col, industry="banking"),
                         {"KYC_RISK_RATING": "text_mask"})
        # Banking excludes PHI — a healthcare-only column must not be caught.
        phi_col = [{"name": "DIAGNOSIS_CD", "type": "varchar(10)"}]
        self.assertEqual(detect_sensitive_columns(phi_col, industry="banking"), {})

    def test_already_caught_columns_not_duplicated_or_broken(self):
        from core.masking import detect_sensitive_columns
        cols = [{"name": "EMAIL", "type": "varchar(100)"}]
        # Generic pattern already catches this — industry pass must not alter it.
        self.assertEqual(
            detect_sensitive_columns(cols, industry="healthcare_pharmacy"),
            detect_sensitive_columns(cols),
        )


class SchemaThreadingTests(unittest.TestCase):
    """Static checks that industry is threaded through every real call site —
    exercising the live DB-connected discovery functions isn't practical in
    a unit test, so this confirms the wiring instead of the DB I/O."""

    def test_apply_masking_and_resolve_masking_fields_accept_industry(self):
        import inspect
        from core import schema as schema_mod
        self.assertIn("industry", inspect.signature(schema_mod._apply_masking).parameters)
        self.assertIn("industry", inspect.signature(schema_mod._resolve_masking_fields).parameters)
        self.assertIn("industry", inspect.signature(schema_mod.discover_and_write).parameters)

    def test_auto_mode_branch_passes_industry_through(self):
        src = _src("core/schema.py")
        self.assertIn('detect_sensitive_columns(col_defs, industry=industry)', src)

    def test_all_three_dialects_thread_industry(self):
        src = _src("core/schema.py")
        fn_names = ["_discover_snowflake", "_discover_oracle", "_discover_azure_sql"]
        starts = [src.index(f"def {fn}(") for fn in fn_names]
        for name, start in zip(fn_names, starts):
            sig_line = src[start:src.index("\n", start)]
            self.assertIn('industry: str = ""', sig_line, name)
        # Each dialect's function body passes industry= to both
        # _resolve_masking_fields and _apply_masking (2 occurrences each).
        bounds = starts + [len(src)]
        for i, name in enumerate(fn_names):
            body = src[bounds[i]:bounds[i + 1]]
            self.assertEqual(
                body.count("industry=industry"), 2,
                f"{name} should pass industry= to both _resolve_masking_fields and _apply_masking",
            )
        # discover_and_write's dispatcher passes industry= to all 3.
        dispatch_start = src.index("def discover_and_write(")
        dispatch_body = src[dispatch_start:min(starts)]
        self.assertEqual(dispatch_body.count("industry=industry"), 3)


class MaskingReviewGateTests(unittest.TestCase):
    def setUp(self):
        import store
        store.init_db()
        self.store = store
        self.account_id = f"acct-mask-gate-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

    def test_confirm_route_stamps_timestamp(self):
        import admin.routes as routes
        with patch.object(routes, "_is_auth", return_value=True):
            result = _arun(routes.admin_confirm_masking_review(
                _post_json({"masking_config": {"S.T": {"mode": "auto"}}}), self.account_id
            ))
        body = json.loads(result.body)
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body.get("masking_reviewed_at"))
        state = self.store.get_client_state(self.account_id)
        self.assertIn("masking_reviewed_at", state)

    def test_kb_tables_save_clears_review(self):
        import admin.routes as routes
        with patch.object(routes, "_is_auth", return_value=True):
            _arun(routes.admin_confirm_masking_review(
                _post_json({"masking_config": {"S.T": {"mode": "auto"}}}), self.account_id
            ))
            self.assertIn("masking_reviewed_at", self.store.get_client_state(self.account_id))
            _arun(routes.admin_save_kb_tables(_post_json({"tables": ["S.T"]}), self.account_id))
        self.assertNotIn("masking_reviewed_at", self.store.get_client_state(self.account_id))

    def test_compliance_profile_standard_clears_review(self):
        import admin.routes as routes
        with patch.object(routes, "_is_auth", return_value=True):
            _arun(routes.admin_confirm_masking_review(
                _post_json({"masking_config": {"S.T": {"mode": "auto"}}}), self.account_id
            ))
            _arun(routes.compliance_save_profile(_post_form({"industry": "standard"}), self.account_id))
        self.assertNotIn("masking_reviewed_at", self.store.get_client_state(self.account_id))

    def test_compliance_profile_regulated_clears_review(self):
        import admin.routes as routes
        with patch.object(routes, "_is_auth", return_value=True):
            _arun(routes.admin_confirm_masking_review(
                _post_json({"masking_config": {"S.T": {"mode": "auto"}}}), self.account_id
            ))
            _arun(routes.compliance_save_profile(
                _post_form({"industry": "healthcare_pharmacy"}), self.account_id
            ))
        self.assertNotIn("masking_reviewed_at", self.store.get_client_state(self.account_id))

    def test_column_sensitivity_route_passes_industry(self):
        src = _src("admin/routes.py")
        fn = src[src.index("async def admin_column_sensitivity("):]
        fn = fn[:fn.index("\n@router.")]
        self.assertIn('store.get_compliance_profile(account_id).get("industry"', fn)
        self.assertIn("detect_sensitive_columns(columns, industry=industry)", fn)

    def test_column_sensitivity_refresh_bypasses_discovery_snapshot(self):
        src = _src("admin/routes.py")
        fn = src[src.index("async def admin_column_sensitivity("):]
        fn = fn[:fn.index("\n@router.")]
        self.assertIn('refresh: str = "0"', fn)
        self.assertIn('force_refresh = refresh == "1"', fn)
        self.assertIn("schema_path.exists() and not force_refresh", fn)

    def test_do_discover_passes_industry(self):
        src = _src("admin/routes.py")
        self.assertIn("_discover_industry = store.get_compliance_profile(account_id)", src)
        self.assertIn("industry=_discover_industry", src)


class TemplateGateRenderTests(unittest.TestCase):
    def _render(self, profile: dict, masking_reviewed_at):
        import sys
        sys.path.insert(0, str(ROOT))
        from admin.routes import templates

        class FakeURL:
            path = "/admin/clients/acct-1/setup"

        class FakeRequest:
            def __init__(self):
                self.query_params = {}
                self.url = FakeURL()

            def url_for(self, *a, **k):
                return "#"

        ctx = {
            "request": FakeRequest(),
            "client": {"account_id": "acct-1", "client_name": "Test Co", "state": "NEW"},
            "state": "NEW",
            "db_cfg": None,
            "compliance_profile": profile,
            "masking_reviewed_at": masking_reviewed_at,
            "schema_files": [],
            "kb_files": [],
            "kb_tables": [],
            "kb_table_count": 0,
            "kb_table_source": "none",
            "schema_breakdown": {},
            "biz_desc": "",
            "biz_desc_parsed": {"overall": "", "schemas": {}},
            "egress_summary": {},
            "saved_masking_config": {},
            "schema_drift": {},
            "saved": None,
            "error": None,
        }
        return templates.get_template("client_setup.html").render(ctx)

    def test_standard_client_never_gated(self):
        html = self._render({"mode": "standard", "industry": "standard"}, None)
        self.assertIn("IS_REGULATED = false", html)
        self.assertIn("Skip masking", html)

    def test_regulated_unreviewed_is_locked(self):
        html = self._render(
            {"mode": "regulated", "industry": "healthcare_pharmacy", "policy_pack_key": "healthcare_pharmacy_v1"},
            None,
        )
        self.assertIn("IS_REGULATED = true", html)
        self.assertIn("MASKING_REVIEWED = false", html)
        self.assertNotIn("Skip masking", html)
        self.assertIn("locked until this review is confirmed", html)

    def test_regulated_reviewed_is_unlocked(self):
        html = self._render(
            {"mode": "regulated", "industry": "healthcare_pharmacy", "policy_pack_key": "healthcare_pharmacy_v1"},
            "2026-07-15 12:00:00",
        )
        self.assertIn("MASKING_REVIEWED = true", html)
        self.assertIn("reviewed and confirmed", html)

    def test_refresh_uses_live_columns_and_callable_mask_toggle(self):
        html = self._render(
            {"mode": "regulated", "industry": "healthcare_pharmacy", "policy_pack_key": "healthcare_pharmacy_v1"},
            None,
        )
        self.assertIn("let _forceLiveColumns = false", html)
        self.assertIn("_forceLiveColumns = true", html)
        self.assertIn("&refresh=1", html)
        self.assertIn("function _toggleMaskItem(fqn)", html)
        self.assertIn("window._toggleMaskItem = _toggleMaskItem", html)


if __name__ == "__main__":
    unittest.main()
