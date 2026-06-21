from __future__ import annotations

import unittest
from unittest.mock import patch

from core.compliance.classifier import classify_column
from core.compliance.models import PolicyContext, PolicyDecision, ResourceRef
from core.compliance.policy_engine import evaluate
from core.compliance.result_guard import protect_rows
from core.compliance.sql_guard import analyze_sql, inject_row_policies
from core.compliance import policy_engine, sql_guard


class PolicyEngineTests(unittest.TestCase):
    def _profile(self, **overrides):
        profile = {
            "mode": "regulated",
            "policy_pack_key": "banking_v1",
            "active_policy_version": 3,
            "enforcement_mode": "enforce",
        }
        profile.update(overrides)
        return profile

    def _context(self, action="query_execution"):
        return PolicyContext(
            account_id="bank-a",
            user_id="7",
            role="analyst",
            purpose_id="customer_service",
            action=action,
            policy_version=3,
        )

    def test_standard_mode_preserves_existing_behavior(self):
        with patch.object(policy_engine.store, "get_compliance_profile", return_value={"mode": "standard"}):
            decision = evaluate(
                self._context(),
                [ResourceRef("DBO.CUSTOMERS", "NAME")],
                record=False,
            )
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.export_allowed)

    def test_regulated_mode_defaults_to_deny_without_rule(self):
        with (
            patch.object(policy_engine.store, "get_compliance_profile", return_value=self._profile()),
            patch.object(policy_engine.store, "get_classification_map", return_value={}),
            patch.object(policy_engine.store, "list_policy_rules", return_value=[]),
            patch.object(policy_engine.store, "list_purposes", return_value=[]),
        ):
            decision = evaluate(
                self._context(),
                [ResourceRef("DBO.CUSTOMERS", "REGION")],
                record=False,
            )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason_code, "default_deny")

    def test_specific_mask_rule_applies_before_broad_allow(self):
        classification = {
            "DBO.CUSTOMERS.NAME": {
                "tags": ["PII"],
                "sensitivity": "RESTRICTED",
                "mask_strategy": "redact",
            }
        }
        rules = [
            {
                "subject_type": "role", "subject_id": "analyst",
                "resource_type": "classification", "resource_pattern": "PII",
                "action": "query_execution", "effect": "mask",
                "mask_strategy": "partial", "cache_ttl_seconds": 0,
            },
            {
                "subject_type": "role", "subject_id": "analyst",
                "resource_type": "classification", "resource_pattern": "*",
                "action": "query_execution", "effect": "allow",
                "cache_ttl_seconds": 300,
            },
        ]
        with (
            patch.object(policy_engine.store, "get_compliance_profile", return_value=self._profile()),
            patch.object(policy_engine.store, "get_classification_map", return_value=classification),
            patch.object(policy_engine.store, "list_policy_rules", return_value=rules),
            patch.object(policy_engine.store, "list_purposes", return_value=[]),
        ):
            decision = evaluate(
                self._context(),
                [ResourceRef("DBO.CUSTOMERS", "NAME")],
                record=False,
            )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.masking["DBO.CUSTOMERS.NAME"], "partial")

    def test_mandatory_deny_wins(self):
        rules = [
            {
                "name": "No card export",
                "subject_type": "role", "subject_id": "analyst",
                "resource_type": "classification", "resource_pattern": "PCI",
                "action": "export", "effect": "deny", "mandatory": 1,
            },
            {
                "subject_type": "role", "subject_id": "analyst",
                "resource_type": "classification", "resource_pattern": "*",
                "action": "export", "effect": "allow", "export_allowed": 1,
            },
        ]
        classification = {
            "DBO.CARDS.PAN": {
                "tags": ["PCI"], "sensitivity": "RESTRICTED",
            }
        }
        with (
            patch.object(policy_engine.store, "get_compliance_profile", return_value=self._profile()),
            patch.object(policy_engine.store, "get_classification_map", return_value=classification),
            patch.object(policy_engine.store, "list_policy_rules", return_value=rules),
            patch.object(policy_engine.store, "list_purposes", return_value=[]),
        ):
            decision = evaluate(
                self._context("export"),
                [ResourceRef("DBO.CARDS", "PAN")],
                record=False,
            )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason_code, "mandatory_policy_deny")

    def test_shadow_mode_records_denial_but_does_not_block(self):
        with (
            patch.object(
                policy_engine.store,
                "get_compliance_profile",
                return_value=self._profile(enforcement_mode="shadow"),
            ),
            patch.object(policy_engine.store, "get_classification_map", return_value={}),
            patch.object(policy_engine.store, "list_policy_rules", return_value=[]),
            patch.object(policy_engine.store, "list_purposes", return_value=[]),
        ):
            decision = evaluate(
                self._context(),
                [ResourceRef("DBO.CUSTOMERS", "REGION")],
                record=False,
            )
        self.assertFalse(decision.allowed)
        self.assertTrue(decision.effective_allowed)


class SqlPolicyTests(unittest.TestCase):
    def test_lineage_and_aggregate_detection(self):
        analysis = analyze_sql(
            "SELECT c.customer_name, SUM(t.amount) AS revenue "
            "FROM dbo.customers c JOIN dbo.transactions t ON c.id=t.customer_id "
            "GROUP BY c.customer_name",
            "azure_sql",
        )
        self.assertIn("DBO.CUSTOMERS.CUSTOMER_NAME", analysis.lineage["customer_name"])
        self.assertIn("DBO.TRANSACTIONS.AMOUNT", analysis.lineage["revenue"])
        self.assertIn("revenue", analysis.aggregate_outputs)

    def test_select_star_is_detected_on_source_table(self):
        analysis = analyze_sql("SELECT * FROM dbo.customers", "azure_sql")
        self.assertTrue(analysis.has_star)
        self.assertEqual(analysis.tables, ["DBO.CUSTOMERS"])

    def test_row_policy_uses_structured_user_group_values(self):
        policies = [{
            "id": 1,
            "subject_type": "role",
            "subject_id": "analyst",
            "table_fqn": "DBO.CUSTOMERS",
            "condition": {
                "field": "BRANCH_ID",
                "operator": "IN",
                "value_source": "user.groups",
            },
        }]
        context = PolicyContext(
            account_id="a", role="analyst", groups=["NORTH", "SOUTH"]
        )
        with patch.object(sql_guard.store, "list_row_policies", return_value=policies):
            sql, applied = inject_row_policies(
                "SELECT c.NAME FROM dbo.CUSTOMERS c WHERE c.ACTIVE=1",
                "azure_sql",
                context,
            )
        self.assertIn("c.BRANCH_ID IN ('NORTH', 'SOUTH')", sql)
        self.assertEqual(applied[0]["policy_id"], 1)

    def test_unsupported_row_operator_is_rejected(self):
        policies = [{
            "id": 1,
            "subject_type": "role",
            "subject_id": "analyst",
            "table_fqn": "DBO.CUSTOMERS",
            "condition": {
                "field": "BRANCH_ID",
                "operator": "EXEC",
                "value": "x",
            },
        }]
        with (
            patch.object(sql_guard.store, "list_row_policies", return_value=policies),
            self.assertRaises(ValueError),
        ):
            inject_row_policies(
                "SELECT NAME FROM dbo.CUSTOMERS",
                "azure_sql",
                PolicyContext(account_id="a", role="analyst"),
            )


class ResultProtectionTests(unittest.TestCase):
    def test_masking_happens_by_output_lineage(self):
        decision = PolicyDecision(
            allowed=True,
            reason_code="allow",
            masking={"DBO.CUSTOMERS.ACCOUNT_NUMBER": "partial"},
        )
        protected = protect_rows(
            [{"Customer": "Alice", "Account": "123456789"}],
            decision,
            {
                "Customer": ["DBO.CUSTOMERS.NAME"],
                "Account": ["DBO.CUSTOMERS.ACCOUNT_NUMBER"],
            },
            account_id="tenant-a",
        )
        self.assertEqual(protected[0]["Customer"], "Alice")
        self.assertTrue(protected[0]["Account"].endswith("6789"))
        self.assertNotEqual(protected[0]["Account"], "123456789")

    def test_tokenization_is_deterministic_per_tenant(self):
        decision = PolicyDecision(
            allowed=True,
            reason_code="allow",
            masking={"DBO.PATIENTS.MRN": "tokenize"},
        )
        args = (
            [{"MRN": "P-100"}],
            decision,
            {"MRN": ["DBO.PATIENTS.MRN"]},
        )
        first = protect_rows(*args, account_id="hospital-a")
        second = protect_rows(*args, account_id="hospital-a")
        other = protect_rows(*args, account_id="hospital-b")
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)


class ClassificationTests(unittest.TestCase):
    def test_banking_classifier_detects_card_and_financial_fields(self):
        card = classify_column("card_number", "banking")
        balance = classify_column("account_balance", "banking")
        self.assertIn("PCI", card["tags"])
        self.assertIn("FINANCIAL", balance["tags"])
        self.assertEqual(card["sensitivity"], "RESTRICTED")

    def test_healthcare_classifier_detects_phi_and_prescriptions(self):
        patient = classify_column("patient_diagnosis", "healthcare_pharmacy")
        compound = classify_column("compound_ingredient_name", "healthcare_pharmacy")
        self.assertIn("PHI", patient["tags"])
        self.assertIn("PRESCRIPTION", compound["tags"])


if __name__ == "__main__":
    unittest.main()
