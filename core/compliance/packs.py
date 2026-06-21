"""Versioned policy-pack defaults.

These are technical control mappings, not claims of regulatory certification.
"""

from __future__ import annotations

from copy import deepcopy


PACKS = {
    "banking_v1": {
        "key": "banking_v1",
        "version": "1.0.0",
        "industry": "banking",
        "framework_versions": {
            "GLBA_SAFEGUARDS": "current",
            "PCI_DSS": "4.0.1",
        },
        "classification_tags": ["PII", "PCI", "KYC_AML", "FINANCIAL"],
        "default_purposes": [
            {
                "purpose_key": "customer_service",
                "name": "Customer service",
                "default_for_roles": ["analyst"],
                "permissions": {
                    "PII": ["llm_context", "query_execution", "result_release"],
                    "FINANCIAL": ["llm_context", "query_execution", "result_release", "chart"],
                },
            },
            {
                "purpose_key": "fraud_investigation",
                "name": "Fraud investigation",
                "requires_prompt": True,
                "permissions": {
                    "PII": ["llm_context", "query_execution", "result_release"],
                    "KYC_AML": ["llm_context", "query_execution", "result_release"],
                    "FINANCIAL": ["llm_context", "query_execution", "result_release", "chart"],
                },
            },
        ],
        "default_cache_ttl_seconds": 300,
        "sensitive_tags": ["PII", "PCI", "KYC_AML"],
        "provider_agreement_tags": ["PII", "PCI", "KYC_AML"],
        "prohibited_llm_tags": ["PCI"],
    },
    "healthcare_pharmacy_v1": {
        "key": "healthcare_pharmacy_v1",
        "version": "1.0.0",
        "industry": "healthcare_pharmacy",
        "framework_versions": {
            "HIPAA": "current",
            "USP_795_797_800": "customer-validated",
        },
        "classification_tags": ["PII", "PHI", "PRESCRIPTION", "PAYMENT"],
        "default_purposes": [
            {
                "purpose_key": "patient_care",
                "name": "Treatment and patient care",
                "default_for_roles": ["analyst"],
                "permissions": {
                    "PII": ["llm_context", "query_execution", "result_release"],
                    "PHI": ["llm_context", "query_execution", "result_release"],
                    "PRESCRIPTION": ["llm_context", "query_execution", "result_release", "chart"],
                },
            },
            {
                "purpose_key": "pharmacy_operations",
                "name": "Pharmacy operations",
                "permissions": {
                    "PRESCRIPTION": ["llm_context", "query_execution", "result_release", "chart"],
                    "PAYMENT": ["llm_context", "query_execution", "result_release", "chart"],
                },
            },
        ],
        "default_cache_ttl_seconds": 0,
        "sensitive_tags": ["PII", "PHI", "PRESCRIPTION"],
        "provider_agreement_tags": ["PHI"],
        "prohibited_llm_tags": [],
    },
}


def get_pack(key: str) -> dict:
    return deepcopy(PACKS.get(key, {}))


def list_packs() -> list[dict]:
    return [deepcopy(value) for value in PACKS.values()]
