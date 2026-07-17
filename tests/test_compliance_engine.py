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

    def test_reviewed_field_strategy_overrides_generic_mask_rule(self):
        classification = {
            "DBO.DOCTORS.DOCTOR_NAME": {
                "tags": ["PII"], "sensitivity": "RESTRICTED",
                "mask_strategy": "safe_alias_name", "reviewed": 1,
            }
        }
        rules = [{
            "subject_type": "role", "subject_id": "analyst",
            "resource_type": "classification", "resource_pattern": "PII",
            "action": "query_execution", "effect": "mask",
            "mask_strategy": "redact", "cache_ttl_seconds": 0,
        }]
        with (
            patch.object(policy_engine.store, "get_compliance_profile", return_value=self._profile()),
            patch.object(policy_engine.store, "get_classification_map", return_value=classification),
            patch.object(policy_engine.store, "list_policy_rules", return_value=rules),
            patch.object(policy_engine.store, "list_purposes", return_value=[]),
        ):
            decision = evaluate(
                self._context(), [ResourceRef("DBO.DOCTORS", "DOCTOR_NAME")], record=False
            )
        self.assertEqual(decision.masking["DBO.DOCTORS.DOCTOR_NAME"], "safe_alias_name")

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
        self.assertIn("revenue", analysis.mask_exempt_outputs)

    def test_identifier_max_is_not_exempt_from_masking(self):
        analysis = analyze_sql(
            "SELECT MAX(p.doctor_name) AS doctor_name FROM dbo.prescriptions p",
            "azure_sql",
        )
        self.assertIn("doctor_name", analysis.aggregate_outputs)
        self.assertNotIn("doctor_name", analysis.mask_exempt_outputs)

    def test_select_star_is_detected_on_source_table(self):
        analysis = analyze_sql("SELECT * FROM dbo.customers", "azure_sql")
        self.assertTrue(analysis.has_star)
        self.assertEqual(analysis.tables, ["DBO.CUSTOMERS"])

    def test_cte_output_lineage_resolves_to_base_tables(self):
        analysis = analyze_sql(
            "WITH base AS ("
            " SELECT c.customer_name, t.amount"
            " FROM dbo.customers c"
            " JOIN dbo.transactions t ON c.id=t.customer_id"
            ")"
            " SELECT customer_name, SUM(amount) AS revenue"
            " FROM base GROUP BY customer_name",
            "azure_sql",
        )
        self.assertIn("DBO.CUSTOMERS.CUSTOMER_NAME", analysis.lineage["customer_name"])
        self.assertIn("DBO.TRANSACTIONS.AMOUNT", analysis.lineage["revenue"])
        self.assertNotIn("BASE", analysis.tables)
        self.assertIn("revenue", analysis.aggregate_outputs)

    def test_union_output_lineage_includes_every_branch(self):
        analysis = analyze_sql(
            "SELECT customer_name AS label FROM dbo.customers "
            "UNION ALL "
            "SELECT prescriber_name AS label FROM dbo.prescribers",
            "azure_sql",
        )
        self.assertEqual(
            analysis.lineage["label"],
            ["DBO.CUSTOMERS.CUSTOMER_NAME", "DBO.PRESCRIBERS.PRESCRIBER_NAME"],
        )
        self.assertNotIn("label", analysis.mask_exempt_outputs)

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

    def test_cte_row_policy_is_injected_in_the_table_scope(self):
        policies = [{
            "id": 7,
            "subject_type": "role",
            "subject_id": "analyst",
            "table_fqn": "DBO.CUSTOMERS",
            "condition": {"field": "BRANCH_ID", "operator": "=", "value": "NORTH"},
        }]
        context = PolicyContext(account_id="a", role="analyst")
        with patch.object(sql_guard.store, "list_row_policies", return_value=policies):
            rewritten, applied = inject_row_policies(
                "WITH base AS (SELECT c.NAME FROM dbo.CUSTOMERS c) SELECT NAME FROM base",
                "azure_sql",
                context,
            )
        self.assertIn("FROM dbo.CUSTOMERS AS c WHERE c.BRANCH_ID = 'NORTH'", rewritten)
        self.assertNotIn("FROM base WHERE c.BRANCH_ID", rewritten)
        self.assertEqual(applied[0]["policy_id"], 7)


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

    def test_safe_aliases_preserve_grouping_without_revealing_original(self):
        decision = PolicyDecision(
            allowed=True,
            reason_code="allow",
            masking={
                "RX.PRESCRIPTIONS.DOCTOR_NAME": "safe_alias_name",
                "RX.PRESCRIPTIONS.RX_NUMBER": "safe_alias_identifier",
            },
        )
        rows = [{"Doctor": "Dr. Kavitha Rao", "Prescription": "RX-100234"}]
        lineage = {
            "Doctor": ["RX.PRESCRIPTIONS.DOCTOR_NAME"],
            "Prescription": ["RX.PRESCRIPTIONS.RX_NUMBER"],
        }
        first = protect_rows(rows, decision, lineage, account_id="hospital-a")
        second = protect_rows(rows, decision, lineage, account_id="hospital-a")
        other = protect_rows(rows, decision, lineage, account_id="hospital-b")
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertRegex(first[0]["Doctor"], r"^Dr\. [A-Z]-[A-F0-9]{3}$")
        self.assertRegex(first[0]["Prescription"], r"^RX-[A-F0-9]{4}$")
        self.assertNotIn("Kavitha", str(first))
        self.assertNotIn("100234", str(first))

    def test_aggregate_output_is_not_masked_as_an_identifier(self):
        decision = PolicyDecision(
            allowed=True,
            reason_code="allow",
            masking={"RX.PRESCRIPTIONS.RX_NUMBER": "smart_alias"},
        )
        protected = protect_rows(
            [{"PrescriptionCount": 42}],
            decision,
            {"PrescriptionCount": ["RX.PRESCRIPTIONS.RX_NUMBER"]},
            account_id="hospital-a",
            mask_exempt_outputs={"PrescriptionCount"},
        )
        self.assertEqual(protected[0]["PrescriptionCount"], 42)


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

    def test_doctor_and_physician_classify_as_pii_without_name_suffix(self):
        # Regression: DOCTOR/PHYSICIAN/DOCTOR_NM previously got tags=[] since
        # neither contains the substring "name" — they sailed through query
        # results completely unmasked despite the client being onboarded as
        # a regulated healthcare tenant.
        for col in ("DOCTOR", "PHYSICIAN", "DOCTOR_NM", "doctor_name"):
            self.assertIn("PII", classify_column(col, "healthcare_pharmacy")["tags"], col)

    def test_healthcare_alias_strategy_is_field_aware(self):
        self.assertEqual(
            classify_column("DOCTOR_NAME", "healthcare_pharmacy")["mask_strategy"],
            "safe_alias_name",
        )
        self.assertEqual(
            classify_column("RX_NUMBER", "healthcare_pharmacy")["mask_strategy"],
            "safe_alias_identifier",
        )
        self.assertEqual(
            classify_column("PATIENT_DIAGNOSIS", "healthcare_pharmacy")["mask_strategy"],
            "redact",
        )

    def test_unrelated_provider_columns_not_misclassified(self):
        # 'provider' alone is too ambiguous to tag safely (cloud/insurance
        # providers are legitimate business fields, not people) — this fix
        # only adds 'doctor'/'physician', not bare 'provider'.
        self.assertEqual(classify_column("CLOUD_PROVIDER", "healthcare_pharmacy")["tags"], [])


if __name__ == "__main__":
    unittest.main()
