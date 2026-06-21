from pathlib import Path

from evals.regulated_banking import assess_regulated_banking


ROOT = Path(__file__).resolve().parents[1]


def test_regulated_banking_assessment_is_repeatable():
    report = assess_regulated_banking(ROOT)

    assert report["business_suite"]["total"] == 13
    assert report["business_suite"]["pass_rate"] == 100.0
    assert report["controls"]["total"] >= 20
    assert report["controls"]["passed"] > 0


def test_core_banking_security_controls_are_enforced():
    report = assess_regulated_banking(ROOT)
    by_id = {result["id"]: result for result in report["results"]}

    for control_id in {
        "select_only_destructive_block",
        "multi_statement_attack_block",
        "external_exfiltration_block",
        "table_level_least_privilege",
        "direct_identifier_detection",
        "bank_identifier_masking",
        "tenant_scoped_pseudonymization",
        "llm_audit_payload_sanitization",
        "tenant_filtered_vector_retrieval",
        "credential_encryption",
    }:
        assert by_id[control_id]["passed"], by_id[control_id]


def test_material_regulated_data_gaps_remain_visible():
    report = assess_regulated_banking(ROOT)
    by_id = {result["id"]: result for result in report["results"]}

    for control_id in {
        "runtime_sensitive_result_protection",
        "column_level_authorization",
        "wildcard_sensitive_table_guard",
        "row_level_entitlement_policy",
        "enterprise_identity_mfa_sso",
        "tamper_resistant_audit_storage",
        "managed_key_rotation",
    }:
        assert not by_id[control_id]["passed"], by_id[control_id]
        assert by_id[control_id]["remediation"]
