from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from store.db import get_db


def _loads(value: Any, default):
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value or "")
    except Exception:
        return default


def get_compliance_profile(account_id: str) -> dict:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM compliance_profile WHERE account_id=?", (account_id,)
        ).fetchone()
    if not row:
        return {
            "account_id": account_id,
            "mode": "standard",
            "industry": "standard",
            "jurisdictions": [],
            "frameworks": [],
            "policy_pack_key": "",
            "policy_pack_version": "",
            "lifecycle_state": "DRAFT",
            "enforcement_mode": "shadow",
            "active_policy_version": 0,
            "identity_control": "password",
            "managed_secrets_enabled": 0,
            "immutable_audit_enabled": 0,
        }
    result = dict(row)
    result["jurisdictions"] = _loads(result.pop("jurisdictions_json", "[]"), [])
    result["frameworks"] = _loads(result.pop("frameworks_json", "[]"), [])
    return result


def save_compliance_profile(account_id: str, **values: Any) -> dict:
    current = get_compliance_profile(account_id)
    merged = {**current, **values}
    jurisdictions = merged.get("jurisdictions", [])
    frameworks = merged.get("frameworks", [])
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO compliance_profile (
                account_id, mode, industry, jurisdictions_json, frameworks_json,
                policy_pack_key, policy_pack_version, lifecycle_state,
                enforcement_mode, active_policy_version, identity_control,
                managed_secrets_enabled, immutable_audit_enabled,
                external_audit_destination, activated_by, activated_at,
                invalidated_at, invalidated_reason, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))
            ON CONFLICT(account_id) DO UPDATE SET
                mode=excluded.mode,
                industry=excluded.industry,
                jurisdictions_json=excluded.jurisdictions_json,
                frameworks_json=excluded.frameworks_json,
                policy_pack_key=excluded.policy_pack_key,
                policy_pack_version=excluded.policy_pack_version,
                lifecycle_state=excluded.lifecycle_state,
                enforcement_mode=excluded.enforcement_mode,
                active_policy_version=excluded.active_policy_version,
                identity_control=excluded.identity_control,
                managed_secrets_enabled=excluded.managed_secrets_enabled,
                immutable_audit_enabled=excluded.immutable_audit_enabled,
                external_audit_destination=excluded.external_audit_destination,
                activated_by=excluded.activated_by,
                activated_at=excluded.activated_at,
                invalidated_at=excluded.invalidated_at,
                invalidated_reason=excluded.invalidated_reason,
                updated_at=datetime('now')
            """,
            (
                account_id,
                merged.get("mode", "standard"),
                merged.get("industry", "standard"),
                json.dumps(jurisdictions),
                json.dumps(frameworks),
                merged.get("policy_pack_key", ""),
                merged.get("policy_pack_version", ""),
                merged.get("lifecycle_state", "DRAFT"),
                merged.get("enforcement_mode", "shadow"),
                int(merged.get("active_policy_version") or 0),
                merged.get("identity_control", "password"),
                int(bool(merged.get("managed_secrets_enabled"))),
                int(bool(merged.get("immutable_audit_enabled"))),
                merged.get("external_audit_destination", ""),
                merged.get("activated_by", ""),
                merged.get("activated_at"),
                merged.get("invalidated_at"),
                merged.get("invalidated_reason", ""),
            ),
        )
    return get_compliance_profile(account_id)


def create_policy_version(
    account_id: str,
    snapshot: dict,
    *,
    created_by: str = "admin",
    change_summary: str = "",
    status: str = "draft",
) -> int:
    with get_db() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS v FROM compliance_policy_version WHERE account_id=?",
            (account_id,),
        ).fetchone()
        version = int(row["v"] if row else 0) + 1
        conn.execute(
            """
            INSERT INTO compliance_policy_version
                (account_id, version, status, snapshot_json, change_summary, created_by)
            VALUES (?,?,?,?,?,?)
            """,
            (account_id, version, status, json.dumps(snapshot, sort_keys=True), change_summary, created_by),
        )
    return version


def activate_policy_version(account_id: str, version: int, activated_by: str = "admin") -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE compliance_policy_version SET status='superseded' "
            "WHERE account_id=? AND status='active'",
            (account_id,),
        )
        conn.execute(
            """
            UPDATE compliance_policy_version
            SET status='active', activated_by=?, activated_at=datetime('now')
            WHERE account_id=? AND version=?
            """,
            (activated_by, account_id, version),
        )
    save_compliance_profile(account_id, active_policy_version=version)


def list_policy_versions(account_id: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM compliance_policy_version WHERE account_id=? ORDER BY version DESC",
            (account_id,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["snapshot"] = _loads(item.pop("snapshot_json", "{}"), {})
        result.append(item)
    return result


def list_classifications(account_id: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM data_asset_classification
            WHERE account_id=?
            ORDER BY table_fqn, column_name
            """,
            (account_id,),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            tag_rows = conn.execute(
                "SELECT tag FROM data_classification_tag "
                "WHERE classification_id=? ORDER BY tag",
                (item["id"],),
            ).fetchall()
            item["tags"] = [tag["tag"] for tag in tag_rows]
            result.append(item)
    return result


def get_classification_map(account_id: str) -> dict[str, dict]:
    return {
        f"{item['table_fqn'].upper()}.{item['column_name'].upper()}": item
        for item in list_classifications(account_id)
    }


def save_classification(
    account_id: str,
    table_fqn: str,
    column_name: str,
    *,
    sensitivity: str,
    identifiability: str,
    tags: list[str],
    confidence: float = 1.0,
    reviewed: bool = True,
    reviewed_by: str = "admin",
    mask_strategy: str = "redact",
    source: str = "admin",
) -> int:
    table_fqn = table_fqn.upper()
    column_name = column_name.upper()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO data_asset_classification (
                account_id, table_fqn, column_name, sensitivity, identifiability,
                confidence, source, reviewed, reviewed_by, reviewed_at,
                mask_strategy, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,datetime('now'),?,datetime('now'))
            ON CONFLICT(account_id, table_fqn, column_name) DO UPDATE SET
                sensitivity=excluded.sensitivity,
                identifiability=excluded.identifiability,
                confidence=excluded.confidence,
                source=excluded.source,
                reviewed=excluded.reviewed,
                reviewed_by=excluded.reviewed_by,
                reviewed_at=datetime('now'),
                mask_strategy=excluded.mask_strategy,
                updated_at=datetime('now')
            """,
            (
                account_id, table_fqn, column_name, sensitivity.upper(),
                identifiability.upper(), float(confidence), source,
                int(reviewed), reviewed_by, mask_strategy,
            ),
        )
        row = conn.execute(
            "SELECT id FROM data_asset_classification "
            "WHERE account_id=? AND table_fqn=? AND column_name=?",
            (account_id, table_fqn, column_name),
        ).fetchone()
        classification_id = int(row["id"])
        conn.execute(
            "DELETE FROM data_classification_tag WHERE classification_id=?",
            (classification_id,),
        )
        for tag in sorted({str(tag).upper() for tag in tags if str(tag).strip()}):
            conn.execute(
                "INSERT OR IGNORE INTO data_classification_tag (classification_id, tag) VALUES (?,?)",
                (classification_id, tag),
            )
    return classification_id


def list_policy_rules(account_id: str, version: int | None = None) -> list[dict]:
    query = "SELECT * FROM policy_rule WHERE account_id=? AND enabled=1"
    params: list[Any] = [account_id]
    if version:
        query += " AND policy_version=?"
        params.append(version)
    query += " ORDER BY mandatory DESC, id"
    with get_db() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def replace_policy_rules(account_id: str, version: int, rules: list[dict]) -> None:
    with get_db() as conn:
        conn.execute(
            "DELETE FROM policy_rule WHERE account_id=? AND policy_version=?",
            (account_id, version),
        )
        for rule in rules:
            conn.execute(
                """
                INSERT INTO policy_rule (
                    account_id, policy_version, name, subject_type, subject_id,
                    resource_type, resource_pattern, action, effect, mask_strategy,
                    aggregate_only, export_allowed, cache_ttl_seconds, mandatory, enabled
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)
                """,
                (
                    account_id, version, rule.get("name", ""),
                    rule.get("subject_type", "role"), rule.get("subject_id", "analyst"),
                    rule.get("resource_type", "classification"),
                    str(rule.get("resource_pattern", "*")).upper(),
                    rule.get("action", "query_execution"), rule.get("effect", "deny"),
                    rule.get("mask_strategy", ""), int(bool(rule.get("aggregate_only"))),
                    int(bool(rule.get("export_allowed"))),
                    int(rule.get("cache_ttl_seconds") or 0),
                    int(bool(rule.get("mandatory"))),
                ),
            )


def list_row_policies(account_id: str, version: int | None = None) -> list[dict]:
    query = "SELECT * FROM row_policy WHERE account_id=? AND enabled=1"
    params: list[Any] = [account_id]
    if version:
        query += " AND policy_version=?"
        params.append(version)
    with get_db() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["condition"] = _loads(item.pop("condition_json", "{}"), {})
        result.append(item)
    return result


def replace_row_policies(account_id: str, version: int, policies: list[dict]) -> None:
    with get_db() as conn:
        conn.execute(
            "DELETE FROM row_policy WHERE account_id=? AND policy_version=?",
            (account_id, version),
        )
        for policy in policies:
            condition = policy.get("condition") or {}
            if not isinstance(condition, dict) or not condition.get("field"):
                raise ValueError("Each row policy requires a structured condition with a field.")
            conn.execute(
                """
                INSERT INTO row_policy (
                    account_id, policy_version, name, subject_type, subject_id,
                    table_fqn, condition_json, enabled
                ) VALUES (?,?,?,?,?,?,?,1)
                """,
                (
                    account_id, version, policy.get("name", ""),
                    policy.get("subject_type", "role"),
                    str(policy.get("subject_id", "analyst")),
                    str(policy.get("table_fqn", "")).upper(),
                    json.dumps(condition, sort_keys=True),
                ),
            )


def list_purposes(account_id: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM purpose_registry WHERE account_id=? AND enabled=1 ORDER BY name",
            (account_id,),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["default_for_roles"] = _loads(item.pop("default_for_roles", "[]"), [])
            perms = conn.execute(
                "SELECT classification, action, effect FROM purpose_permission WHERE purpose_id=?",
                (item["id"],),
            ).fetchall()
            item["permissions"] = [dict(p) for p in perms]
            result.append(item)
    return result


def replace_purposes(account_id: str, purposes: list[dict]) -> None:
    with get_db() as conn:
        old = conn.execute(
            "SELECT id FROM purpose_registry WHERE account_id=?", (account_id,)
        ).fetchall()
        for row in old:
            conn.execute("DELETE FROM purpose_permission WHERE purpose_id=?", (row["id"],))
        conn.execute("DELETE FROM purpose_registry WHERE account_id=?", (account_id,))
        for purpose in purposes:
            conn.execute(
                """
                INSERT INTO purpose_registry (
                    account_id, purpose_key, name, description, legal_basis_ref,
                    default_for_roles, requires_prompt, enabled
                ) VALUES (?,?,?,?,?,?,?,1)
                """,
                (
                    account_id, purpose["purpose_key"], purpose["name"],
                    purpose.get("description", ""), purpose.get("legal_basis_ref", ""),
                    json.dumps(purpose.get("default_for_roles", [])),
                    int(bool(purpose.get("requires_prompt"))),
                ),
            )
            row = conn.execute(
                "SELECT id FROM purpose_registry WHERE account_id=? AND purpose_key=?",
                (account_id, purpose["purpose_key"]),
            ).fetchone()
            for classification, actions in (purpose.get("permissions") or {}).items():
                for action in actions:
                    conn.execute(
                        """
                        INSERT INTO purpose_permission
                            (purpose_id, classification, action, effect)
                        VALUES (?,?,?,'allow')
                        """,
                        (row["id"], classification.upper(), action),
                    )


def provider_agreement_valid(account_id: str, provider: str, frameworks: list[str]) -> bool:
    today = datetime.now(timezone.utc).date().isoformat()
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT frameworks_json, expires_at FROM provider_agreement
            WHERE account_id=? AND provider=? AND enabled=1
            """,
            (account_id, provider),
        ).fetchall()
    required = {item.upper() for item in frameworks}
    for row in rows:
        if row["expires_at"] and str(row["expires_at"])[:10] < today:
            continue
        available = {str(item).upper() for item in _loads(row["frameworks_json"], [])}
        if not required or required & available:
            return True
    return False


def save_provider_agreement(account_id: str, agreement: dict) -> int:
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO provider_agreement (
                account_id, provider, agreement_type, frameworks_json,
                artifact_ref, artifact_hash, signed_at, expires_at, enabled
            ) VALUES (?,?,?,?,?,?,?,?,1)
            """,
            (
                account_id, agreement.get("provider", ""),
                agreement.get("agreement_type", ""),
                json.dumps(agreement.get("frameworks", [])),
                agreement.get("artifact_ref", ""),
                agreement.get("artifact_hash", ""),
                agreement.get("signed_at"), agreement.get("expires_at"),
            ),
        )
        return int(cur.lastrowid)


def list_provider_agreements(account_id: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM provider_agreement WHERE account_id=? ORDER BY created_at DESC",
            (account_id,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["frameworks"] = _loads(item.pop("frameworks_json", "[]"), [])
        result.append(item)
    return result


def save_user_attestation(
    account_id: str,
    portal_user_id: str,
    *,
    attestation_type: str = "confidentiality",
    document_ref: str = "",
    granted_by: str = "",
) -> int:
    """Record that a named internal user signed a confidentiality/access
    attestation and may see unmasked regulated values in query results.

    Org-level provider agreements (BAA/DPA, see provider_agreement above)
    govern what may reach the LLM provider; this is the per-USER display
    instrument — a different legal document, kept in a separate table so
    an auditor never confuses the two. Re-granting after a revoke inserts
    a fresh row, preserving the full grant/revoke history."""
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO user_attestation (
                account_id, portal_user_id, attestation_type,
                document_ref, granted_by
            ) VALUES (?,?,?,?,?)
            """,
            (account_id, str(portal_user_id), attestation_type, document_ref, granted_by),
        )
        return int(cur.lastrowid)


def revoke_user_attestation(account_id: str, attestation_id: int, revoked_by: str = "") -> bool:
    with get_db() as conn:
        cur = conn.execute(
            """
            UPDATE user_attestation SET revoked_at=datetime('now'), revoked_by=?
            WHERE account_id=? AND id=? AND revoked_at IS NULL
            """,
            (revoked_by, account_id, int(attestation_id)),
        )
        return cur.rowcount > 0


def user_attestation_valid(account_id: str, portal_user_id: str) -> bool:
    if not portal_user_id:
        return False
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT 1 FROM user_attestation
            WHERE account_id=? AND portal_user_id=? AND revoked_at IS NULL
            LIMIT 1
            """,
            (account_id, str(portal_user_id)),
        ).fetchone()
    return row is not None


def list_user_attestations(account_id: str) -> list[dict]:
    """All attestation rows (active and revoked) with the user's display
    name/email joined in for the admin panel."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT ua.*, pu.name AS user_name, pu.email AS user_email
            FROM user_attestation ua
            LEFT JOIN portal_user pu ON CAST(pu.id AS TEXT) = ua.portal_user_id
            WHERE ua.account_id=?
            ORDER BY ua.granted_at DESC, ua.id DESC
            """,
            (account_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def log_policy_decision(
    *,
    account_id: str,
    user_id: str,
    action: str,
    purpose_id: str,
    channel: str,
    allowed: bool,
    reason_code: str,
    resources: list[str],
    obligations: dict,
    policy_version: int,
) -> str:
    audit_id = str(uuid.uuid4())
    with get_db() as conn:
        previous = conn.execute(
            "SELECT record_hash FROM policy_decision_log WHERE account_id=? "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
            (account_id,),
        ).fetchone()
        previous_hash = previous["record_hash"] if previous else ""
        canonical = json.dumps(
            {
                "id": audit_id,
                "account_id": account_id,
                "user_id": user_id,
                "action": action,
                "purpose_id": purpose_id,
                "channel": channel,
                "allowed": bool(allowed),
                "reason_code": reason_code,
                "resources": resources,
                "obligations": obligations,
                "policy_version": policy_version,
                "previous_hash": previous_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        record_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        conn.execute(
            """
            INSERT INTO policy_decision_log (
                id, account_id, user_id, action, purpose_id, channel, allowed,
                reason_code, resource_json, obligation_json, policy_version,
                previous_hash, record_hash
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                audit_id, account_id, user_id, action, purpose_id, channel,
                int(bool(allowed)), reason_code, json.dumps(resources),
                json.dumps(obligations, sort_keys=True), policy_version,
                previous_hash, record_hash,
            ),
        )
    return audit_id


def get_policy_decision_counts(account_id: str, days: int = 30) -> dict:
    """Allowed/denied decision counts for the compliance Trust panel window."""
    from datetime import datetime, timedelta, timezone
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=int(days))
    ).strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        rows = conn.execute(
            "SELECT allowed, COUNT(*) AS n FROM policy_decision_log "
            "WHERE account_id=? AND created_at >= ? GROUP BY allowed",
            (account_id, cutoff),
        ).fetchall()
    counts = {int(row["allowed"]): row["n"] for row in rows}
    return {
        "window_days": int(days),
        "allowed": counts.get(1, 0),
        "denied": counts.get(0, 0),
    }


def list_decisions(account_id: str, limit: int = 100) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM policy_decision_log WHERE account_id=? "
            "ORDER BY created_at DESC LIMIT ?",
            (account_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def log_export_event(
    *,
    account_id: str,
    user_id: str,
    trace_id: str,
    policy_version: int,
    purpose_id: str,
    export_format: str,
    row_count: int,
    columns: list[str],
) -> str:
    export_id = str(uuid.uuid4())
    fingerprint = hashlib.sha256(
        f"{export_id}:{account_id}:{user_id}:{trace_id}".encode("utf-8")
    ).hexdigest()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO export_event (
                id, account_id, user_id, trace_id, policy_version, purpose_id,
                format, row_count, columns_json, fingerprint
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                export_id, account_id, user_id, trace_id, policy_version,
                purpose_id, export_format, row_count, json.dumps(columns), fingerprint,
            ),
        )
    return export_id


def create_break_glass_grant(
    account_id: str,
    user_id: str,
    incident_ref: str,
    reason: str,
    resources: list[str],
    actions: list[str],
    expires_at: str,
    created_by: str = "admin",
) -> str:
    grant_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO break_glass_grant (
                id, account_id, user_id, incident_ref, reason, resource_json,
                action_json, expires_at, created_by
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                grant_id, account_id, user_id, incident_ref, reason,
                json.dumps(resources), json.dumps(actions), expires_at, created_by,
            ),
        )
    return grant_id


def get_active_break_glass_grant(account_id: str, grant_id: str, user_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT * FROM break_glass_grant
            WHERE id=? AND account_id=? AND user_id=? AND revoked_at IS NULL
              AND expires_at > datetime('now')
            """,
            (grant_id, account_id, user_id),
        ).fetchone()
    if not row:
        return None
    result = dict(row)
    result["resources"] = _loads(result.pop("resource_json", "[]"), [])
    result["actions"] = _loads(result.pop("action_json", "[]"), [])
    return result


def get_active_break_glass_for_user(account_id: str, user_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT * FROM break_glass_grant
            WHERE account_id=? AND user_id=? AND revoked_at IS NULL
              AND expires_at > datetime('now')
            ORDER BY expires_at DESC LIMIT 1
            """,
            (account_id, user_id),
        ).fetchone()
    if not row:
        return None
    result = dict(row)
    result["resources"] = _loads(result.pop("resource_json", "[]"), [])
    result["actions"] = _loads(result.pop("action_json", "[]"), [])
    return result


def save_assessment(account_id: str, policy_version: int, results: list[dict]) -> int:
    critical_failed = sum(
        1 for item in results if item["severity"] == "critical" and item["status"] == "fail"
    )
    high_failed = sum(
        1 for item in results if item["severity"] == "high" and item["status"] == "fail"
    )
    passed = sum(1 for item in results if item["status"] == "pass")
    status = "pass" if critical_failed == 0 else "fail"
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO compliance_assessment_run (
                account_id, policy_version, status, critical_failed,
                high_failed, passed_count, completed_at
            ) VALUES (?,?,?,?,?,?,datetime('now'))
            """,
            (account_id, policy_version, status, critical_failed, high_failed, passed),
        )
        run_id = int(cur.lastrowid)
        for item in results:
            conn.execute(
                """
                INSERT INTO compliance_assessment_result
                    (run_id, control_key, severity, status, message, remediation)
                VALUES (?,?,?,?,?,?)
                """,
                (
                    run_id, item["control_key"], item["severity"], item["status"],
                    item.get("message", ""), item.get("remediation", ""),
                ),
            )
    return run_id


def get_latest_assessment(account_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM compliance_assessment_run WHERE account_id=? "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
            (account_id,),
        ).fetchone()
        if not row:
            return None
        result = dict(row)
        findings = conn.execute(
            "SELECT * FROM compliance_assessment_result WHERE run_id=? ORDER BY id",
            (result["id"],),
        ).fetchall()
    result["results"] = [dict(item) for item in findings]
    return result
