"""
LLM egress hardening — Phase 3: close the two real boundary holes.

A1  the value index injected verbatim cell values into the SQL prompt, gated by
    a feature flag rather than by compliance mode.
A3  few-shot KB examples carry literal WHERE values into every SQL call.

Policy is option (b) from docs/LLM_EGRESS_PLAN.md §5: a regulated tenant grounds
only on columns an admin has REVIEWED and classified non-sensitive; anything
else is suppressed, so an incomplete classification workflow degrades to no
grounding rather than to silent exposure.

These are the negative-space tests the plan called out as the missing
verification — they assert a known sensitive value is ABSENT from text about to
be handed to a model.
"""

import unittest
from unittest.mock import patch

from core.examples import format_examples_for_prompt, scrub_example_sql_literals
from core.value_resolver import (
    build_verified_values_injection,
    filter_resolved_for_compliance,
)


def _resolved():
    return {
        "verified": [
            {"phrase": "emco", "table_fqn": "DB.SCH.CUSTOMER", "column": "NAME",
             "value": "EMCO Corporation"},
            {"phrase": "lipitor", "table_fqn": "DB.SCH.PRODUCT", "column": "DRUG_NAME",
             "value": "Lipitor 40mg"},
        ],
        "in_lists": [
            {"phrase": "acme", "table_fqn": "DB.SCH.CUSTOMER", "column": "NAME",
             "values": ["ACME EU", "ACME US"]},
        ],
        "clarify": [{"phrase": "north"}],
    }


def _classifications(reviewed=True, tags=None):
    return {
        "DB.SCH.CUSTOMER.NAME": {"reviewed": reviewed, "tags": tags or ["PII"]},
        "DB.SCH.PRODUCT.DRUG_NAME": {"reviewed": reviewed, "tags": tags or ["PRESCRIPTION"]},
    }


class ValueGroundingComplianceTests(unittest.TestCase):
    """A1 — verified values must not reach a regulated tenant's prompt unless
    the column was admin-reviewed as non-sensitive."""

    def test_non_regulated_tenant_is_unchanged(self):
        with patch("core.compliance.policy_engine.is_regulated", return_value=False):
            out, ev = filter_resolved_for_compliance("acct", _resolved())
        self.assertEqual(len(out["verified"]), 2)
        self.assertEqual(len(out["in_lists"]), 1)
        self.assertFalse(ev["applied"])

    def test_regulated_tenant_drops_sensitive_columns(self):
        with patch("core.compliance.policy_engine.is_regulated", return_value=True), \
             patch("store.get_compliance_profile", return_value={"industry": "healthcare_pharmacy", "policy_pack_key": "healthcare_pharmacy_v1"}), \
             patch("store.get_classification_map", return_value=_classifications()):
            out, ev = filter_resolved_for_compliance("acct", _resolved())
        self.assertEqual(out["verified"], [])
        self.assertEqual(out["in_lists"], [])
        self.assertTrue(ev["applied"])
        self.assertEqual(ev["dropped"], 3)

    def test_regulated_tenant_keeps_reviewed_non_sensitive_columns(self):
        """Option (b): a reviewed, non-sensitive column still grounds."""
        clean = {
            "DB.SCH.CUSTOMER.NAME": {"reviewed": True, "tags": []},
            "DB.SCH.PRODUCT.DRUG_NAME": {"reviewed": True, "tags": ["PRESCRIPTION"]},
        }
        with patch("core.compliance.policy_engine.is_regulated", return_value=True), \
             patch("store.get_compliance_profile", return_value={"industry": "healthcare_pharmacy", "policy_pack_key": "healthcare_pharmacy_v1"}), \
             patch("store.get_classification_map", return_value=clean):
            out, ev = filter_resolved_for_compliance("acct", _resolved())
        kept = [i["column"] for i in out["verified"]]
        self.assertEqual(kept, ["NAME"], "reviewed non-sensitive column was dropped")
        self.assertEqual(len(out["in_lists"]), 1)
        self.assertEqual(ev["dropped"], 1)

    def test_unreviewed_column_is_suppressed(self):
        """An incomplete classification workflow must degrade to suppression."""
        with patch("core.compliance.policy_engine.is_regulated", return_value=True), \
             patch("store.get_compliance_profile", return_value={"industry": "healthcare_pharmacy", "policy_pack_key": "healthcare_pharmacy_v1"}), \
             patch("store.get_classification_map",
                   return_value=_classifications(reviewed=False, tags=[])):
            out, _ = filter_resolved_for_compliance("acct", _resolved())
        self.assertEqual(out["verified"], [])

    def test_unclassified_column_is_suppressed(self):
        with patch("core.compliance.policy_engine.is_regulated", return_value=True), \
             patch("store.get_compliance_profile", return_value={"industry": "healthcare_pharmacy", "policy_pack_key": "healthcare_pharmacy_v1"}), \
             patch("store.get_classification_map", return_value={}):
            out, _ = filter_resolved_for_compliance("acct", _resolved())
        self.assertEqual(out["verified"], [])
        self.assertEqual(out["in_lists"], [])

    def test_filter_failure_suppresses_rather_than_leaks(self):
        with patch("core.compliance.policy_engine.is_regulated", side_effect=RuntimeError("boom")):
            out, ev = filter_resolved_for_compliance("acct", _resolved())
        self.assertEqual(out["verified"], [])
        self.assertEqual(out["in_lists"], [])
        self.assertIn("filter_error", ev["reason"])

    def test_clarify_bucket_survives_filtering(self):
        """clarify carries no values — suppressing it would cost accuracy for
        no compliance benefit."""
        with patch("core.compliance.policy_engine.is_regulated", return_value=True), \
             patch("store.get_compliance_profile", return_value={"industry": "healthcare_pharmacy", "policy_pack_key": "healthcare_pharmacy_v1"}), \
             patch("store.get_classification_map", return_value={}):
            out, _ = filter_resolved_for_compliance("acct", _resolved())
        self.assertEqual(len(out["clarify"]), 1)


class AssembledPromptNegativeSpaceTests(unittest.TestCase):
    """The verification the plan flagged as missing: a known sensitive value
    must be absent from the text actually handed to the model."""

    def test_sensitive_value_absent_from_verified_values_block(self):
        with patch("core.compliance.policy_engine.is_regulated", return_value=True), \
             patch("store.get_compliance_profile", return_value={"industry": "healthcare_pharmacy", "policy_pack_key": "healthcare_pharmacy_v1"}), \
             patch("store.get_classification_map", return_value=_classifications()):
            out, _ = filter_resolved_for_compliance("acct", _resolved())
        block = build_verified_values_injection(out)
        self.assertNotIn("Lipitor 40mg", block)
        self.assertNotIn("EMCO Corporation", block)
        self.assertNotIn("ACME EU", block)
        self.assertEqual(block, "", "no values cleared, so no block should be emitted")

    def test_value_still_present_for_non_regulated_tenant(self):
        """Guard against over-correction — the feature must still work."""
        with patch("core.compliance.policy_engine.is_regulated", return_value=False):
            out, _ = filter_resolved_for_compliance("acct", _resolved())
        block = build_verified_values_injection(out)
        self.assertIn("EMCO Corporation", block)


class ExampleLiteralScrubTests(unittest.TestCase):
    """A3 — stored example SQL carries real values into every prompt."""

    def test_string_literals_are_masked(self):
        sql = "SELECT * FROM RX WHERE PRODUCT_NAME = 'Lipitor 40mg' AND STATUS = 'Denied'"
        out = scrub_example_sql_literals(sql)
        self.assertNotIn("Lipitor 40mg", out)
        self.assertNotIn("Denied", out)
        self.assertIn("'<value>'", out)

    def test_identifiers_and_shape_survive(self):
        sql = "SELECT SUM(NET_REVENUE) FROM PHARMA_LAB.F_RX_FILL WHERE STATUS = 'X' GROUP BY PRODUCT_ID"
        out = scrub_example_sql_literals(sql)
        for keep in ("NET_REVENUE", "PHARMA_LAB.F_RX_FILL", "GROUP BY", "PRODUCT_ID", "SUM("):
            self.assertIn(keep, out)

    def test_escaped_quotes_do_not_break_masking(self):
        sql = "SELECT * FROM T WHERE NAME = 'O''Brien Pharmacy'"
        out = scrub_example_sql_literals(sql)
        self.assertNotIn("Brien", out)

    def test_regulated_prompt_has_no_example_literals(self):
        examples = [{
            "question": "revenue by product",
            "sql": "SELECT SUM(AMT) FROM F WHERE DRUG = 'Lipitor 40mg'",
            "table": "F",
        }]
        with patch("core.compliance.policy_engine.is_regulated", return_value=True):
            text = format_examples_for_prompt(examples, "acct")
        self.assertNotIn("Lipitor 40mg", text)
        self.assertIn("SUM(AMT)", text)

    def test_non_regulated_prompt_keeps_literals(self):
        examples = [{
            "question": "revenue by product",
            "sql": "SELECT SUM(AMT) FROM F WHERE DRUG = 'Lipitor 40mg'",
            "table": "F",
        }]
        with patch("core.compliance.policy_engine.is_regulated", return_value=False):
            text = format_examples_for_prompt(examples, "acct")
        self.assertIn("Lipitor 40mg", text)

    def test_scrub_failure_masks_anyway(self):
        examples = [{"question": "q", "sql": "SELECT * FROM T WHERE X = 'Secret'", "table": "T"}]
        with patch("core.compliance.policy_engine.is_regulated",
                   side_effect=RuntimeError("boom")):
            text = format_examples_for_prompt(examples, "acct")
        self.assertNotIn("Secret", text)

    def test_omitted_account_id_is_backwards_compatible(self):
        examples = [{"question": "q", "sql": "SELECT * FROM T WHERE X = 'Keep'", "table": "T"}]
        self.assertIn("Keep", format_examples_for_prompt(examples))


class WiringTests(unittest.TestCase):
    """Both live call sites must pass account_id, or the scrub never runs."""

    def test_pipeline_and_webhooks_pass_account_id(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        pipeline = (root / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        webhooks = (root / "gateway" / "webhooks.py").read_text(encoding="utf-8")
        self.assertIn("format_examples_for_prompt(examples, account_id)", pipeline)
        self.assertIn("format_examples_for_prompt(_fb_examples, account_id)", webhooks)
        self.assertIn("filter_resolved_for_compliance", pipeline)


if __name__ == "__main__":
    unittest.main()
