"""Code-level regulated banking readiness assessment.

This suite is an engineering control check, not a legal compliance
certification. It exercises the controls QueryBot can enforce in code and
reports material gaps that require product or infrastructure work.
"""

from __future__ import annotations

import argparse
import inspect
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from core.llm_audit import sanitize_payload_preview
from core.masking import (
    detect_sensitive_columns,
    mask_rows,
    scrub_embedded_pii,
)
from core.result_renderer import _sanitize_rows
from core.validator import validate_sql_detailed
from evals.business_user import run_business_suite, summarize
from gateway.web_adapter import WebAdapter
from store.user_store import _hash_pw, _verify_pw


@dataclass
class ControlResult:
    id: str
    category: str
    severity: str
    passed: bool
    evidence: str
    remediation: str = ""


def _control(
    control_id: str,
    category: str,
    severity: str,
    check: Callable[[], tuple[bool, str]],
    remediation: str = "",
) -> ControlResult:
    try:
        passed, evidence = check()
    except Exception as exc:
        passed, evidence = False, f"assessment error: {exc}"
    return ControlResult(
        id=control_id,
        category=category,
        severity=severity,
        passed=bool(passed),
        evidence=str(evidence),
        remediation="" if passed else remediation,
    )


def _banking_metadata(root: Path) -> tuple[set[str], dict[str, dict[str, str]]]:
    schema = json.loads(
        (root / "evals" / "sample_banking" / "schema.json").read_text(
            encoding="utf-8"
        )
    )
    known: set[str] = set()
    columns: dict[str, dict[str, str]] = {}
    for fqn, table in schema.items():
        parts = str(fqn).upper().split(".")
        variants = {parts[-1], ".".join(parts[-2:]), str(fqn).upper()}
        table_cols = {
            str(col["name"]).upper(): str(col.get("type") or "")
            for col in table.get("columns") or []
        }
        for variant in variants:
            known.add(variant)
            columns[variant] = table_cols
    return known, columns


def assess_regulated_banking(root: Path | None = None) -> dict:
    root = Path(root or Path(__file__).resolve().parents[1])
    known, table_columns = _banking_metadata(root)
    customers = {"BANKING.CUSTOMERS"}
    non_customer_tables = {
        "BANKING.ACCOUNTS",
        "BANKING.LOANS",
        "BANKING.CUSTOMER_PROFIT",
    }

    def verdict(sql: str, allowed: set[str] | None = None):
        return validate_sql_detailed(
            sql,
            known,
            "azure_sql",
            allowed,
            table_columns,
        )

    controls: list[ControlResult] = []

    controls.append(_control(
        "select_only_destructive_block",
        "query_security",
        "critical",
        lambda: (
            (result := verdict(
                "UPDATE BANKING.ACCOUNTS SET ACCOUNT_STATUS='Closed'"
            )).code == "ddl",
            f"validator returned {result.code}",
        ),
        "Reject every non-SELECT operation before database execution.",
    ))
    controls.append(_control(
        "multi_statement_attack_block",
        "query_security",
        "critical",
        lambda: (
            (result := verdict(
                "SELECT * FROM BANKING.ACCOUNTS; DROP TABLE BANKING.ACCOUNTS"
            )).code == "ddl",
            f"validator returned {result.code}",
        ),
        "Reject multi-statement and destructive payloads.",
    ))
    controls.append(_control(
        "external_exfiltration_block",
        "query_security",
        "critical",
        lambda: (
            (result := verdict(
                "SELECT * FROM OPENROWSET(BULK 'https://example.invalid/x', "
                "SINGLE_CLOB) AS x"
            )).code == "ddl",
            f"validator returned {result.code}",
        ),
        "Block external readers, unload, copy, and network-capable SQL.",
    ))
    controls.append(_control(
        "unknown_table_block",
        "query_security",
        "high",
        lambda: (
            (result := verdict("SELECT * FROM BANKING.WIRE_TRANSFERS")).code
            == "unknown_table",
            f"validator returned {result.code}",
        ),
        "Fail closed when SQL references objects outside discovered metadata.",
    ))
    controls.append(_control(
        "table_level_least_privilege",
        "authorization",
        "critical",
        lambda: (
            (result := verdict(
                "SELECT CUSTOMER_NAME, RISK_RATING FROM BANKING.CUSTOMERS",
                non_customer_tables,
            )).code == "access_denied",
            f"validator returned {result.code}",
        ),
        "Keep table ACL enforcement in both retrieval and SQL validation.",
    ))
    controls.append(_control(
        "authorized_aggregate_query",
        "authorization",
        "high",
        lambda: (
            (result := verdict(
                "SELECT ACCOUNT_TYPE, SUM(BALANCE) AS BALANCE "
                "FROM BANKING.ACCOUNTS GROUP BY ACCOUNT_TYPE",
                {"BANKING.ACCOUNTS"},
            )).ok,
            f"validator returned {result.code}",
        ),
        "Permit approved aggregate use cases without opening customer PII.",
    ))

    banking_columns = [
        {"name": "CUSTOMER_NAME", "type": "varchar(120)"},
        {"name": "EMAIL_ADDRESS", "type": "varchar(160)"},
        {"name": "PHONE_NUMBER", "type": "varchar(40)"},
        {"name": "NATIONAL_ID", "type": "varchar(40)"},
        {"name": "ACCOUNT_NUMBER", "type": "varchar(40)"},
        {"name": "ROUTING_NUMBER", "type": "varchar(40)"},
        {"name": "CARD_NUMBER", "type": "varchar(40)"},
        {"name": "RISK_RATING", "type": "varchar(20)"},
    ]
    detected = detect_sensitive_columns(banking_columns)
    direct_identifiers = {
        "CUSTOMER_NAME",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "NATIONAL_ID",
        "ACCOUNT_NUMBER",
        "ROUTING_NUMBER",
        "CARD_NUMBER",
    }
    controls.append(_control(
        "direct_identifier_detection",
        "privacy",
        "critical",
        lambda: (
            direct_identifiers <= set(detected),
            f"detected fields: {', '.join(sorted(detected))}",
        ),
        "Expand automatic classification for any missing direct identifiers.",
    ))

    sample = [{
        "CUSTOMER_NAME": "Alice Jones",
        "EMAIL_ADDRESS": "alice@example.com",
        "PHONE_NUMBER": "+1 212-555-0199",
        "NATIONAL_ID": "123-45-6789",
        "ACCOUNT_NUMBER": "123456789012",
        "ROUTING_NUMBER": "021000021",
        "CARD_NUMBER": "4111111111111111",
        "RISK_RATING": "High",
    }]
    masked_a = mask_rows(
        [dict(sample[0])],
        direct_identifiers,
        banking_columns,
        seed_key="tenant-bank-a",
    )
    masked_a_repeat = mask_rows(
        [dict(sample[0])],
        direct_identifiers,
        banking_columns,
        seed_key="tenant-bank-a",
    )
    masked_b = mask_rows(
        [dict(sample[0])],
        direct_identifiers,
        banking_columns,
        seed_key="tenant-bank-b",
    )
    controls.append(_control(
        "bank_identifier_masking",
        "privacy",
        "critical",
        lambda: (
            all(masked_a[0][field] != sample[0][field] for field in direct_identifiers),
            "direct identifiers changed before LLM/KB use",
        ),
        "Mask or synthesize all customer, account, routing, and payment identifiers.",
    ))
    controls.append(_control(
        "tenant_scoped_pseudonymization",
        "tenant_isolation",
        "high",
        lambda: (
            masked_a == masked_a_repeat and masked_a != masked_b,
            "same-tenant output is stable and cross-tenant output differs",
        ),
        "Use a tenant-specific secret when deterministic masking is required.",
    ))
    controls.append(_control(
        "embedded_pii_scrubbing",
        "privacy",
        "high",
        lambda: (
            (
                scrubbed := scrub_embedded_pii(
                    "Call alice@example.com at 212-555-0199; SSN 123-45-6789"
                )
            ).find("alice@example.com") == -1
            and "212-555-0199" not in scrubbed
            and "123-45-6789" not in scrubbed,
            f"scrubbed text: {scrubbed}",
        ),
        "Scrub identifiers embedded in notes and free-text fields.",
    ))
    controls.append(_control(
        "llm_audit_payload_sanitization",
        "audit",
        "high",
        lambda: (
            (
                preview := sanitize_payload_preview(
                    "Analyze customer account 123456789012",
                    "Alice Jones alice@example.com phone 212-555-0199",
                )
            ).find("alice@example.com") == -1
            and "212-555-0199" not in preview
            and "123456789012" not in preview,
            "LLM audit preview removed direct identifiers",
        ),
        "Sanitize all audit payload previews before persistence.",
    ))
    controls.append(_control(
        "conversation_sql_literal_sanitization",
        "privacy",
        "high",
        lambda: (
            (
                sanitized := WebAdapter._sanitize_sql_for_history(
                    "SELECT * FROM BANKING.CUSTOMERS "
                    "WHERE CUSTOMER_NAME='Alice Jones'"
                )
            ).find("Alice Jones") == -1,
            f"stored SQL: {sanitized}",
        ),
        "Strip customer values before prior SQL is reused as conversational context.",
    ))

    def vector_filter_check() -> tuple[bool, str]:
        from core.vector_store import QdrantKBRetriever

        retriever = object.__new__(QdrantKBRetriever)
        retriever._account_id = "tenant-bank-a"
        search_filter = retriever._account_filter(["BANKING.ACCOUNTS"])
        clauses = {
            condition.key: getattr(condition.match, "value", None)
            or getattr(condition.match, "any", None)
            for condition in search_filter.must
        }
        ok = (
            clauses.get("account_id") == "tenant-bank-a"
            and "BANKING.ACCOUNTS" in (clauses.get("fqn") or [])
        )
        return ok, f"vector filter clauses: {clauses}"

    controls.append(_control(
        "tenant_filtered_vector_retrieval",
        "tenant_isolation",
        "critical",
        vector_filter_check,
        "Require tenant and table ACL filters inside vector search.",
    ))
    controls.append(_control(
        "password_hashing",
        "identity",
        "high",
        lambda: (
            (hashed := _hash_pw("Correct-Horse-42")).startswith("pbkdf2_sha256$")
            and _verify_pw(hashed, "Correct-Horse-42")
            and not _verify_pw(hashed, "wrong"),
            "portal password uses salted PBKDF2 and constant-time verification",
        ),
        "Use an enterprise identity provider or a modern adaptive password hash.",
    ))

    def credential_encryption_check() -> tuple[bool, str]:
        import store.crypto as crypto

        original = crypto.KEY_FILE
        with tempfile.TemporaryDirectory() as tmp:
            crypto.KEY_FILE = Path(tmp) / "querybot.key"
            try:
                secret = {"password": "bank-db-secret", "server": "db.internal"}
                encrypted = crypto.encrypt(secret)
                restored = crypto.decrypt_json(encrypted)
                ok = restored == secret and "bank-db-secret" not in encrypted
            finally:
                crypto.KEY_FILE = original
        return ok, "database credentials round-trip through Fernet encryption"

    controls.append(_control(
        "credential_encryption",
        "cryptography",
        "critical",
        credential_encryption_check,
        "Store production encryption keys in a managed HSM/Key Vault with rotation.",
    ))

    controls.append(_control(
        "runtime_sensitive_result_protection",
        "privacy",
        "critical",
        lambda: (
            (
                sanitized := _sanitize_rows([{
                    "CUSTOMER_NAME": "Alice Jones",
                    "ACCOUNT_NUMBER": "123456789012",
                }])
            )[0]["CUSTOMER_NAME"] != "Alice Jones"
            and sanitized[0]["ACCOUNT_NUMBER"] != "123456789012",
            "runtime row serialization currently preserves raw selected values",
        ),
        "Add policy-driven masking or denial to query results, exports, charts, and caches.",
    ))
    controls.append(_control(
        "column_level_authorization",
        "authorization",
        "critical",
        lambda: (
            not (
                result := verdict(
                    "SELECT CUSTOMER_NAME, RISK_RATING FROM BANKING.CUSTOMERS",
                    customers,
                )
            ).ok,
            f"sensitive-column query validator returned {result.code}",
        ),
        "Add per-role allow/deny policies for columns and derived sensitive attributes.",
    ))
    controls.append(_control(
        "wildcard_sensitive_table_guard",
        "authorization",
        "critical",
        lambda: (
            not (
                result := verdict(
                    "SELECT * FROM BANKING.CUSTOMERS",
                    customers,
                )
            ).ok,
            f"SELECT * validator returned {result.code}",
        ),
        "Reject SELECT * on sensitive tables and require explicit approved columns.",
    ))
    controls.append(_control(
        "contextual_sensitive_attribute_policy",
        "privacy",
        "high",
        lambda: (
            "RISK_RATING" in detected,
            f"automatic masking classified RISK_RATING={detected.get('RISK_RATING')}",
        ),
        "Support tenant-defined classifications for credit risk, KYC, AML, and vulnerability data.",
    ))
    controls.append(_control(
        "row_level_entitlement_policy",
        "authorization",
        "critical",
        lambda: (
            any(
                name in inspect.signature(validate_sql_detailed).parameters
                for name in ("row_policy", "row_filter", "entitlements")
            ),
            "validator has no branch/region/customer row-entitlement parameter",
        ),
        "Inject and validate mandatory row filters for branch, region, portfolio, and legal entity.",
    ))

    store_source = (
        root / "store" / "config_store.py"
    ).read_text(encoding="utf-8").lower()
    controls.append(_control(
        "broad_audit_retention_policy",
        "retention",
        "high",
        lambda: (
            "purge_old_query" in store_source
            and "purge_old_answer_trace" in store_source
            and "purge_old_kb_egress" in store_source,
            "only LLM-call retention is implemented; query, trace, and egress retention is incomplete",
        ),
        "Add configurable legal retention and deletion policies for every audit/data class.",
    ))

    requirements = (root / "requirements.txt").read_text(encoding="utf-8").lower()
    controls.append(_control(
        "enterprise_identity_mfa_sso",
        "identity",
        "critical",
        lambda: (
            any(name in requirements for name in ("msal", "openid", "authlib"))
            and "mfa" in store_source,
            "local portal passwords are present; no Entra/OIDC MFA integration was detected",
        ),
        "Use Entra ID/External ID with MFA, conditional access, and tenant lifecycle controls.",
    ))

    db_source = (root / "store" / "db.py").read_text(encoding="utf-8").lower()
    deployment_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore").lower()
        for path in [
            root / "store" / "db.py",
            root / "core" / "log_export.py",
            root / "deploy.sh",
        ]
        if path.exists()
    )
    controls.append(_control(
        "tamper_resistant_audit_storage",
        "audit",
        "high",
        lambda: (
            any(
                marker in deployment_text
                for marker in (
                    "immutability_policy",
                    "immutable_storage_with_versioning",
                    "set-azstorageblobimmutabilitypolicy",
                    "worm_retention_days",
                )
            ),
            "no immutable/WORM audit sink configuration was detected",
        ),
        "Export security audit events to immutable or append-only storage with restricted deletion.",
    ))
    controls.append(_control(
        "managed_key_rotation",
        "cryptography",
        "high",
        lambda: (
            "key vault" in (root / "store" / "crypto.py").read_text(
                encoding="utf-8"
            ).lower()
            and "rotation" in (root / "store" / "crypto.py").read_text(
                encoding="utf-8"
            ).lower(),
            "Fernet works, but the production key is a local file without managed rotation",
        ),
        "Move keys to Azure Key Vault or Managed HSM and define rotation/recovery procedures.",
    ))

    banking_results = run_business_suite(root / "evals" / "sample_banking")
    business_summary = summarize(banking_results)
    passed = sum(control.passed for control in controls)
    critical_failures = [
        control.id
        for control in controls
        if not control.passed and control.severity == "critical"
    ]
    return {
        "assessment": "regulated_banking_code_readiness",
        "disclaimer": (
            "Engineering assessment only; it does not certify compliance with "
            "GLBA, PCI DSS, GDPR, DPDP, RBI, or any other regulation."
        ),
        "business_suite": business_summary,
        "controls": {
            "total": len(controls),
            "passed": passed,
            "failed": len(controls) - passed,
            "pass_rate": round(passed / len(controls) * 100, 2),
            "critical_failures": critical_failures,
        },
        "results": [asdict(control) for control in controls],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default="")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any critical control fails.",
    )
    args = parser.parse_args()
    report = assess_regulated_banking()
    for result in report["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"{status:4} [{result['severity']:<8}] "
            f"{result['id']}: {result['evidence']}"
        )
    summary = report["controls"]
    print(
        f"\nControls: {summary['passed']}/{summary['total']} passed "
        f"({summary['pass_rate']}%)"
    )
    print(
        "Banking calculations: "
        f"{report['business_suite']['passed']}/"
        f"{report['business_suite']['total']} passed"
    )
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.strict and summary["critical_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
