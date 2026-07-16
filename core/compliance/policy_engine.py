from __future__ import annotations

import fnmatch
from typing import Iterable

import store
from core.compliance.models import PolicyContext, PolicyDecision, ResourceRef
from core.compliance.packs import get_pack


_HARD_DENIED_ACTIONS = {"insert", "update", "delete", "ddl", "cross_tenant"}
_BREAK_GLASS_DENIED_ACTIONS = {"export", "llm_context"}


def result_llm_features_allowed(account_id: str) -> bool:
    """
    False for any regulated tenant, unconditionally.

    Result narration, follow-up suggestions, and why-insights all hand real
    query result rows to the LLM as a second call after SQL generation —
    none of them go through evaluate()'s per-resource masking, so they rely
    entirely on column-name pattern matching having caught every sensitive
    field. Regulated tenants get a stricter boundary instead: the LLM's
    only job is writing SQL from schema/sample-value context (still gated
    by the llm_context/BAA check); it never sees actual answers.

    Deliberately independent of BAA status and enforcement_mode — a signed
    agreement covers legal liability for permitted use, not the
    minimum-necessary-exposure posture this boundary is for.
    """
    return store.get_compliance_profile(account_id).get("mode") != "regulated"


def resolve_context(
    account_id: str,
    user: dict | None,
    *,
    action: str,
    channel: str = "portal",
    purpose_id: str = "",
    provider: str = "",
    break_glass_grant_id: str | None = None,
) -> PolicyContext:
    user = user or {}
    role = str(user.get("role") or "system")
    groups = []
    if user.get("group_id") is not None:
        groups.append(str(user["group_id"]))
    if user.get("group_name"):
        groups.append(str(user["group_name"]))

    profile = store.get_compliance_profile(account_id)
    resolved_purpose = purpose_id
    if not resolved_purpose:
        for purpose in store.list_purposes(account_id):
            if role in purpose.get("default_for_roles", []):
                resolved_purpose = purpose["purpose_key"]
                break
    if not resolved_purpose:
        pack = get_pack(profile.get("policy_pack_key", ""))
        for purpose in pack.get("default_purposes", []):
            if role in purpose.get("default_for_roles", []):
                resolved_purpose = purpose["purpose_key"]
                break

    if not break_glass_grant_id and user.get("id") is not None:
        grant = store.get_active_break_glass_for_user(account_id, str(user.get("id")))
        break_glass_grant_id = grant.get("id") if grant else None

    return PolicyContext(
        account_id=account_id,
        user_id=str(user.get("id") or user.get("user_id") or ""),
        groups=groups,
        purpose_id=resolved_purpose,
        channel=channel,
        action=action,
        policy_version=int(profile.get("active_policy_version") or 0),
        break_glass_grant_id=break_glass_grant_id,
        role=role,
        provider=provider,
        user_attributes=dict(user.get("attributes") or {}),
    )


def _classification_for_resource(
    resource: ResourceRef,
    classification_map: dict[str, dict],
) -> dict | None:
    if not resource.column:
        return None
    resource_key = resource.key.upper()
    direct = classification_map.get(resource_key)
    if direct:
        return direct
    table_parts = resource.table.upper().split(".")
    for key, value in classification_map.items():
        key_parts = key.upper().split(".")
        if len(key_parts) < 2 or key_parts[-1] != resource.column.upper():
            continue
        classified_table = key_parts[:-1]
        if classified_table[-len(table_parts):] == table_parts:
            return value
        if table_parts[-len(classified_table):] == classified_table:
            return value
    return None


def _subject_matches(rule: dict, context: PolicyContext) -> bool:
    subject_type = str(rule.get("subject_type") or "role")
    subject_id = str(rule.get("subject_id") or "")
    if subject_type == "all":
        return True
    if subject_type == "user":
        return subject_id == context.user_id
    if subject_type == "group":
        return subject_id in context.groups
    return subject_id == context.role


def _resource_matches(rule: dict, resource: ResourceRef, classification: dict | None) -> bool:
    resource_type = str(rule.get("resource_type") or "classification")
    pattern = str(rule.get("resource_pattern") or "*").upper()
    if resource_type == "classification":
        values = set((classification or {}).get("tags") or [])
        values.add(str((classification or {}).get("sensitivity") or "").upper())
        return pattern == "*" or pattern in values
    if resource_type == "table":
        return fnmatch.fnmatch(resource.table.upper(), pattern)
    return fnmatch.fnmatch(resource.key.upper(), pattern)


def _purpose_allows(
    context: PolicyContext,
    classifications: Iterable[dict | None],
) -> bool:
    sensitive = {
        tag
        for classification in classifications
        if classification
        for tag in classification.get("tags", [])
    }
    if not sensitive:
        return True
    if not context.purpose_id:
        return False

    purposes = store.list_purposes(context.account_id)
    if purposes:
        selected = next(
            (purpose for purpose in purposes if purpose["purpose_key"] == context.purpose_id),
            None,
        )
        if not selected:
            return False
        allowed = {
            (item["classification"].upper(), item["action"])
            for item in selected.get("permissions", [])
            if item.get("effect") == "allow"
        }
        return all((tag.upper(), context.action) in allowed for tag in sensitive)

    profile = store.get_compliance_profile(context.account_id)
    pack = get_pack(profile.get("policy_pack_key", ""))
    selected = next(
        (
            purpose
            for purpose in pack.get("default_purposes", [])
            if purpose["purpose_key"] == context.purpose_id
        ),
        None,
    )
    if not selected:
        return False
    permissions = selected.get("permissions", {})
    return all(context.action in permissions.get(tag, []) for tag in sensitive)


def evaluate(
    context: PolicyContext,
    resources: list[ResourceRef] | None = None,
    *,
    record: bool = True,
) -> PolicyDecision:
    resources = resources or []
    profile = store.get_compliance_profile(context.account_id)
    if profile.get("mode") != "regulated":
        return PolicyDecision(
            allowed=True,
            reason_code="standard_mode",
            permitted_resources=resources,
            export_allowed=True,
            cache_ttl_seconds=600,
            policy_version=0,
            explanation="Standard tenants retain the existing access model.",
        )

    shadow = profile.get("enforcement_mode") == "shadow"
    pack = get_pack(profile.get("policy_pack_key", ""))
    classifications_map = store.get_classification_map(context.account_id)
    classifications = [
        _classification_for_resource(resource, classifications_map)
        for resource in resources
    ]
    resource_tags = {
        tag
        for classification in classifications
        if classification
        for tag in classification.get("tags", [])
    }
    denied = False
    reason = "default_deny"
    explanation = "No active policy permits this action."
    permitted: list[ResourceRef] = []
    masking: dict[str, str] = {}
    aggregate_only: list[ResourceRef] = []
    cache_ttl = int(pack.get("default_cache_ttl_seconds") or 0)
    export_allowed = False

    if context.action in _HARD_DENIED_ACTIONS:
        denied = True
        reason = "platform_hard_deny"
        explanation = "Regulated mode permits read-only, tenant-scoped operations."
    elif context.action == "llm_context" and resource_tags & set(pack.get("prohibited_llm_tags", [])):
        denied = True
        reason = "prohibited_llm_data"
        explanation = "This data category cannot be sent to the configured LLM."
    elif (
        context.action == "llm_context"
        and resource_tags & set(pack.get("provider_agreement_tags", []))
        and not store.provider_agreement_valid(
            context.account_id,
            context.provider,
            list((pack.get("framework_versions") or {}).keys()),
        )
    ):
        denied = True
        reason = "provider_agreement_required"
        explanation = "A current provider agreement is required for this data category."
    else:
        rules = store.list_policy_rules(context.account_id, context.policy_version or None)
        matching = [
            rule
            for rule in rules
            if rule.get("action") == context.action and _subject_matches(rule, context)
        ]
        mandatory_denies = [
            rule
            for rule in matching
            if rule.get("mandatory")
            and rule.get("effect") == "deny"
            and any(
                _resource_matches(rule, resource, classification)
                for resource, classification in zip(resources, classifications)
            )
        ]
        if mandatory_denies:
            denied = True
            reason = "mandatory_policy_deny"
            explanation = mandatory_denies[0].get("name") or "A mandatory policy denies this action."
        else:
            for resource, classification in zip(resources, classifications):
                applicable = [
                    rule
                    for rule in matching
                    if _resource_matches(rule, resource, classification)
                ]
                deny_rule = next((rule for rule in applicable if rule.get("effect") == "deny"), None)
                allow_rule = next((rule for rule in applicable if rule.get("effect") in {"allow", "mask"}), None)
                if deny_rule or not allow_rule:
                    denied = True
                    reason = "tenant_policy_deny" if deny_rule else "default_deny"
                    explanation = (
                        (deny_rule or {}).get("name")
                        or f"No policy permits {resource.key} for this subject."
                    )
                    break
                permitted.append(resource)
                if allow_rule.get("effect") == "mask" or allow_rule.get("mask_strategy"):
                    strategy = allow_rule.get("mask_strategy") or (
                        classification or {}
                    ).get("mask_strategy", "redact")
                    masking[resource.key] = strategy
                if allow_rule.get("aggregate_only"):
                    aggregate_only.append(resource)
                cache_ttl = min(
                    cache_ttl,
                    int(allow_rule.get("cache_ttl_seconds") or cache_ttl),
                )
                export_allowed = export_allowed or bool(allow_rule.get("export_allowed"))

            if not resources and matching:
                allowed_rule = next((rule for rule in matching if rule.get("effect") == "allow"), None)
                denied = allowed_rule is None
                if allowed_rule:
                    export_allowed = bool(allowed_rule.get("export_allowed"))
                    cache_ttl = int(allowed_rule.get("cache_ttl_seconds") or cache_ttl)
                    reason = "policy_allow"
                    explanation = allowed_rule.get("name") or "Policy permits this action."

            if not denied and not _purpose_allows(context, classifications):
                denied = True
                reason = "purpose_not_permitted"
                explanation = "The selected business purpose does not permit this data use."

    grant = None
    if denied and context.break_glass_grant_id and context.action not in _BREAK_GLASS_DENIED_ACTIONS:
        grant = store.get_active_break_glass_grant(
            context.account_id, context.break_glass_grant_id, context.user_id
        )
        if grant and context.action in grant.get("actions", []):
            allowed_resources = {item.upper() for item in grant.get("resources", [])}
            if all(
                "*" in allowed_resources
                or resource.key.upper() in allowed_resources
                or resource.table.upper() in allowed_resources
                for resource in resources
            ):
                denied = False
                reason = "break_glass_allow"
                explanation = "A current, scoped emergency grant permits this action."
                permitted = list(resources)
                export_allowed = False
                cache_ttl = 0

    allowed = not denied
    obligations = {
        "masking": masking,
        "aggregate_only": [item.key for item in aggregate_only],
        "row_obligations": [],
        "export_allowed": export_allowed,
        "cache_ttl_seconds": cache_ttl,
        "break_glass": bool(grant),
    }
    audit_id = ""
    if record:
        audit_id = store.log_policy_decision(
            account_id=context.account_id,
            user_id=context.user_id,
            action=context.action,
            purpose_id=context.purpose_id,
            channel=context.channel,
            allowed=allowed,
            reason_code=reason,
            resources=[resource.key for resource in resources],
            obligations=obligations,
            policy_version=context.policy_version,
        )
    return PolicyDecision(
        allowed=allowed,
        reason_code=reason,
        permitted_resources=permitted,
        row_obligations=[],
        masking=masking,
        aggregate_only=aggregate_only,
        export_allowed=export_allowed,
        cache_ttl_seconds=cache_ttl,
        policy_version=context.policy_version,
        audit_id=audit_id,
        shadow=shadow,
        explanation=explanation,
    )
