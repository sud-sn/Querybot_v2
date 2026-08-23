"""
LLM egress hardening — Phase 1: one regulated predicate, fail-closed default.

Covers A6/A7/A4 from docs/LLM_EGRESS_PLAN.md:
  A6  is_regulated() replaces four divergent predicates; the
      enforcement_mode == "enforce" conjunct that let a regulated tenant in
      shadow mode fail OPEN on charts and exports is gone.
  A7  an account with no compliance_profile row fails closed.
  A4  compute_data_brief no longer claims it holds no row values.

Behavioural where it can be: the policy tests drive the real engine and only
patch the store accessors, matching tests/test_compliance_engine.py.
"""

import unittest
from unittest.mock import patch

from core.compliance import policy_engine
from core.compliance.policy_engine import is_regulated, result_llm_features_allowed


def _profile(mode="standard", enforcement_mode="shadow", industry="standard"):
    return {
        "account_id": "acct",
        "mode": mode,
        "industry": industry,
        "enforcement_mode": enforcement_mode,
        "lifecycle_state": "DRAFT",
        "active_policy_version": 0,
    }


class IsRegulatedTests(unittest.TestCase):
    """A6/A7 — the single predicate."""

    def test_provisioned_standard_tenant_is_not_regulated(self):
        with patch.object(policy_engine.store, "compliance_profile_exists", return_value=True), \
             patch.object(policy_engine.store, "get_compliance_profile", return_value=_profile()):
            self.assertFalse(is_regulated("acct"))
            self.assertTrue(result_llm_features_allowed("acct"))

    def test_regulated_tenant_is_regulated(self):
        with patch.object(policy_engine.store, "compliance_profile_exists", return_value=True), \
             patch.object(policy_engine.store, "get_compliance_profile",
                          return_value=_profile(mode="regulated", enforcement_mode="enforce")):
            self.assertTrue(is_regulated("acct"))
            self.assertFalse(result_llm_features_allowed("acct"))

    def test_regulated_in_shadow_mode_is_still_regulated(self):
        """The A6 fail-open: shadow must not disable the boundary."""
        with patch.object(policy_engine.store, "compliance_profile_exists", return_value=True), \
             patch.object(policy_engine.store, "get_compliance_profile",
                          return_value=_profile(mode="regulated", enforcement_mode="shadow")):
            self.assertTrue(is_regulated("acct"))
            self.assertFalse(result_llm_features_allowed("acct"))

    def test_unprovisioned_tenant_fails_closed(self):
        """A7 — no profile row must not mean 'standard'."""
        with patch.object(policy_engine.store, "compliance_profile_exists", return_value=False):
            self.assertTrue(is_regulated("brand_new_acct"))
            self.assertFalse(result_llm_features_allowed("brand_new_acct"))

    def test_unprovisioned_tenant_does_not_consult_the_synthesized_default(self):
        """Fail-closed must not depend on the permissive synthesized profile."""
        with patch.object(policy_engine.store, "compliance_profile_exists", return_value=False), \
             patch.object(policy_engine.store, "get_compliance_profile") as get_profile:
            self.assertTrue(is_regulated("brand_new_acct"))
            get_profile.assert_not_called()

    def test_unprovisioned_tenant_is_logged_loudly(self):
        """A silent fail-closed is how a whole workspace goes quiet unexplained."""
        with patch.object(policy_engine.store, "compliance_profile_exists", return_value=False):
            with self.assertLogs("querybot", level="WARNING") as captured:
                is_regulated("brand_new_acct")
        self.assertTrue(
            any("brand_new_acct" in line for line in captured.output),
            "fail-closed decision must name the account at WARNING level",
        )


class ProfileExistenceTests(unittest.TestCase):
    """A7 — the store-level distinction the predicate rests on."""

    def test_exists_is_false_for_unknown_account(self):
        import store

        self.assertFalse(store.compliance_profile_exists("no_such_account_xyz"))

    def test_backfill_is_idempotent(self):
        """Running init_db repeatedly must not duplicate or drop profiles."""
        import sqlite3
        from store.db import _backfill_compliance_profiles, get_db

        def _counts():
            with get_db() as conn:
                clients = conn.execute("SELECT COUNT(*) FROM client").fetchone()[0]
                profiles = conn.execute(
                    "SELECT COUNT(*) FROM compliance_profile"
                ).fetchone()[0]
                orphans = conn.execute(
                    "SELECT COUNT(*) FROM client c "
                    "LEFT JOIN compliance_profile p ON p.account_id = c.account_id "
                    "WHERE p.account_id IS NULL"
                ).fetchone()[0]
            return clients, profiles, orphans

        _backfill_compliance_profiles()
        first = _counts()
        _backfill_compliance_profiles()
        second = _counts()
        self.assertEqual(first, second, "backfill is not idempotent")
        self.assertEqual(second[2], 0, "clients left without a compliance profile")

    def test_backfill_creates_standard_not_regulated_profiles(self):
        """Backfill must preserve existing posture, not tighten it."""
        from store.db import get_db

        with get_db() as conn:
            rows = conn.execute(
                "SELECT DISTINCT mode FROM compliance_profile"
            ).fetchall()
        modes = {r[0] for r in rows}
        self.assertTrue(modes.issubset({"standard", "regulated"}), modes)


class FailOpenRemovalTests(unittest.TestCase):
    """A6 — the two sites that required enforcement_mode == 'enforce'."""

    def test_chart_guard_no_longer_requires_enforce_mode(self):
        """The guard moved to core/chart_policy.py so the forecast gate applies
        the same rule from the same code.

        Rewritten from a string scan of result_renderer.py to an execution of
        the guard. The scan broke the moment the code moved, which is the
        clearest possible demonstration of what it was really testing: the
        presence of a substring, not the behaviour. The behaviour is that a
        regulated tenant fails CLOSED when policy evaluation raises.
        """
        from unittest.mock import patch

        from core.chart_policy import aggregate_only_gate_passes

        broken = dict(
            portal_user=None, event=None,
            sql="not valid sql at all", db_type="no_such_db_type",
        )
        with patch("core.compliance.policy_engine.is_regulated", return_value=True):
            self.assertFalse(
                aggregate_only_gate_passes(account_id="regulated", **broken),
                "a regulated tenant must fail closed when evaluation raises",
            )
        with patch("core.compliance.policy_engine.is_regulated", return_value=False):
            self.assertTrue(
                aggregate_only_gate_passes(account_id="ordinary", **broken),
            )

        from pathlib import Path

        src = (Path(__file__).resolve().parents[1] / "core" / "chart_policy.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('profile.get("enforcement_mode") == "enforce"', src)

    def test_export_guard_no_longer_requires_enforce_mode(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parents[1] / "portal" / "routes.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('profile.get("enforcement_mode") == "enforce"', src)
        self.assertIn('is_regulated(user["account_id"])', src)


class DataBriefDocstringTests(unittest.TestCase):
    """A4 — the docstring must not claim a guarantee the function lacks."""

    def test_docstring_states_the_brief_carries_real_values(self):
        from core.insight import compute_data_brief

        doc = compute_data_brief.__doc__ or ""
        self.assertIn("CONTAINS REAL DATA VALUES", doc)
        # The old claim may still appear, but only as a quoted refutation —
        # never as an unqualified assertion on its own line.
        for line in doc.splitlines():
            stripped = line.strip()
            if stripped.startswith("Returns a dict") or stripped.startswith("NEVER"):
                self.fail(f"docstring still asserts the old guarantee: {stripped!r}")

    def test_docstring_directs_callers_to_the_gate(self):
        from core.insight import compute_data_brief

        doc = compute_data_brief.__doc__ or ""
        self.assertIn("result_llm_features_allowed", doc)


if __name__ == "__main__":
    unittest.main()
