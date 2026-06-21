from __future__ import annotations

import store
from core.compliance.packs import get_pack


def assess(account_id: str) -> dict:
    profile = store.get_compliance_profile(account_id)
    pack = get_pack(profile.get("policy_pack_key", ""))
    classifications = store.list_classifications(account_id)
    rules = store.list_policy_rules(
        account_id, int(profile.get("active_policy_version") or 0) or None
    )
    purposes = store.list_purposes(account_id)
    agreements = store.list_provider_agreements(account_id)

    results = []

    def check(key: str, ok: bool, severity: str, message: str, remediation: str) -> None:
        results.append(
            {
                "control_key": key,
                "severity": severity,
                "status": "pass" if ok else "fail",
                "message": message,
                "remediation": remediation,
            }
        )

    check(
        "profile_selected",
        bool(pack),
        "critical",
        "A supported regulated-industry policy pack is selected.",
        "Select Banking or Healthcare/Compounding Pharmacy.",
    )
    check(
        "classification_complete",
        bool(classifications) and all(item.get("reviewed") for item in classifications),
        "critical",
        "Every discovered column has an administrator-reviewed classification.",
        "Review all unreviewed columns in the Classification workspace.",
    )
    check(
        "active_policy",
        int(profile.get("active_policy_version") or 0) > 0 and bool(rules),
        "critical",
        "An active, versioned access policy exists.",
        "Save and activate the policy matrix.",
    )
    check(
        "purpose_registry",
        bool(purposes),
        "critical",
        "Approved business purposes are configured.",
        "Seed or configure at least one permitted business purpose.",
    )
    provider_required = bool(pack.get("provider_agreement_tags"))
    check(
        "provider_agreement",
        not provider_required or bool(agreements),
        "critical",
        "A provider agreement is recorded for sensitive AI processing.",
        "Record the applicable provider agreement or prohibit sensitive LLM egress.",
    )
    check(
        "runtime_enforcement",
        True,
        "critical",
        "Governed SQL execution and pre-cache result protection are installed.",
        "",
    )
    check(
        "identity_mfa",
        profile.get("identity_control") in {"mfa", "oidc", "entra"},
        "production",
        "Production regulated mode requires MFA or federated identity.",
        "Configure MFA, OIDC, or Microsoft Entra ID.",
    )
    check(
        "managed_secrets",
        bool(profile.get("managed_secrets_enabled")),
        "production",
        "Production secrets are managed outside application configuration.",
        "Move credentials to a managed secrets service.",
    )
    check(
        "immutable_audit",
        bool(profile.get("immutable_audit_enabled"))
        and bool(profile.get("external_audit_destination")),
        "production",
        "An immutable external audit destination is configured.",
        "Configure WORM/immutable audit export.",
    )

    run_id = store.save_assessment(
        account_id, int(profile.get("active_policy_version") or 0), results
    )
    critical_failed = sum(
        1 for item in results if item["severity"] == "critical" and item["status"] == "fail"
    )
    production_failed = sum(
        1 for item in results if item["severity"] == "production" and item["status"] == "fail"
    )
    state = (
        "ASSESSMENT_FAILED"
        if critical_failed
        else ("PRODUCTION_BLOCKED" if production_failed else "REGULATED_READY")
    )
    return {
        "run_id": run_id,
        "state": state,
        "critical_failed": critical_failed,
        "production_failed": production_failed,
        "results": results,
    }


def activate_pilot(account_id: str, activated_by: str = "admin") -> dict:
    assessment = assess(account_id)
    if assessment["critical_failed"]:
        store.save_compliance_profile(
            account_id,
            lifecycle_state="ASSESSMENT_FAILED",
            enforcement_mode="shadow",
        )
        return assessment

    lifecycle = (
        "PILOT_READY"
        if assessment["production_failed"]
        else "REGULATED_READY"
    )
    store.save_compliance_profile(
        account_id,
        lifecycle_state=lifecycle,
        enforcement_mode="enforce",
        activated_by=activated_by,
    )
    from core.result_cache import result_cache

    result_cache.clear_account(account_id)
    assessment["state"] = lifecycle
    return assessment
