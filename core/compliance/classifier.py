from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

import store


_PATTERNS = {
    # account/routing/IBAN/acct.no direct-identifier suffixes are deliberately
    # HERE, not just under FINANCIAL below — FINANCIAL isn't in banking's
    # sensitive_tags (aggregate balance/revenue figures are meant to stay
    # freely queryable), but a SPECIFIC customer's account or routing number
    # is a direct identifier, not an aggregate figure. core/masking.py
    # (KB-time) already treats these as "credit_card"-strategy sensitive;
    # without a PII tag here, query-time masking never matched that intent.
    "PII": re.compile(
        r"(name|email|phone|mobile|address|ssn|social.?security|passport|national.?id|dob|birth|"
        r"doctor|physician|"
        r"account.?(?:number|num|no)\b|routing.?(?:number|num|no)\b|\biban\b|\bacct.?no\b)",
        re.I,
    ),
    "PCI": re.compile(r"(card.?number|pan|cvv|cvc|expiry|cardholder)", re.I),
    "KYC_AML": re.compile(r"(kyc|aml|sanction|pep|risk.?rating|beneficial.?owner)", re.I),
    "FINANCIAL": re.compile(
        r"(account|routing|iban|swift|balance|revenue|profit|loss|income|payment|transaction|credit|debit)",
        re.I,
    ),
    # policy/subscriber/group numbers are HIPAA-listed health-plan
    # beneficiary identifiers (same protected-identifier class as member ID,
    # already covered below) — not clinical data, but still a direct patient
    # identifier.
    "PHI": re.compile(
        r"(patient|diagnos|medical|health|allerg|condition|mrn|member.?id|clinical|"
        r"policy.?(?:number|num|no|id)|subscriber.?id|group.?(?:number|num|no))",
        re.I,
    ),
    # NPI/DEA/license numbers identify the PRESCRIBER, not the patient, but
    # core/masking.py already treats them as sensitive (KB-time) — this was
    # the exact gap that let DOCTOR_NAME-style columns leak: masked in KB
    # samples, invisible to query-time classification. "prescriber_license"
    # etc. already match via the existing "prescriber" alternative below;
    # these add the BARE forms (a plain "LICENSE_NUMBER" or "NPI" column).
    "PRESCRIPTION": re.compile(
        r"(prescription|rx|drug|medication|compound|ingredient|prescriber|dosage|ndc|"
        # \b doesn't break on "_" (it's a word char), so "DEA_NUMBER" needs an
        # explicit underscore/space-tolerant alternative alongside the bare
        # \bnpi\b/\bdea\b forms.
        r"\bnpi\b|\bdea\b|npi[_\s]?(?:number|num|no)|dea[_\s]?(?:number|num|no)|"
        r"licen[sc]e.?(?:number|num|no)|pharmacy.?licen[sc]e|state.?licen[sc]e)",
        re.I,
    ),
    "PAYMENT": re.compile(r"(payment|payer|claim|charge|copay|insurance)", re.I),
}


def _default_mask_strategy(column_name: str, tags: list[str]) -> str:
    name = str(column_name or "")
    if set(tags) & {"PII", "PRESCRIPTION"}:
        if re.search(r"(?:doctor|physician|prescriber|provider|patient|person).*(?:name|_nm)?$", name, re.I):
            return "safe_alias_name"
        # NOTE: policy/subscriber/group numbers are NOT in this identifier
        # regex — they're tagged PHI (not PII/PRESCRIPTION, see _PATTERNS
        # above), so this branch never runs for them; they resolve via the
        # "redact" fallback below instead, which is a safe default and
        # matches how the pre-existing "member_id" PHI pattern already
        # behaves. Only PII/PRESCRIPTION-tagged identifiers get the
        # grouping-preserving alias here.
        if re.search(
            r"(?:rx|prescription|mrn|medical.?record|patient|member|"
            r"npi|dea|licen[sc]e).*(?:id|key|num|number|no)$",
            name, re.I,
        ) or re.fullmatch(r"npi|dea", name, re.I):
            # A bare "NPI"/"DEA" column IS the identifier itself — no
            # id/num/no suffix to require, unlike "RX_NUMBER" etc.
            return "safe_alias_identifier"
    return "partial" if set(tags) & {"FINANCIAL", "PAYMENT"} else "redact"


def classify_column(column_name: str, industry: str) -> dict:
    tags = []
    for tag, pattern in _PATTERNS.items():
        if pattern.search(column_name):
            tags.append(tag)
    if industry == "banking":
        tags = [tag for tag in tags if tag not in {"PHI", "PRESCRIPTION", "PAYMENT"}]
    elif industry == "healthcare_pharmacy":
        tags = [tag for tag in tags if tag not in {"PCI", "KYC_AML"}]
    direct = bool(set(tags) & {"PII", "PCI", "PHI"})
    sensitivity = "RESTRICTED" if direct or set(tags) & {"KYC_AML", "PRESCRIPTION"} else (
        "CONFIDENTIAL" if tags else "INTERNAL"
    )
    confidence = 0.92 if tags else 0.55
    return {
        "tags": tags,
        "sensitivity": sensitivity,
        "identifiability": "DIRECT" if direct else ("INDIRECT" if tags else "NONE"),
        "confidence": confidence,
        "mask_strategy": _default_mask_strategy(column_name, tags),
    }


def import_schema_classifications(account_id: str, schema_dir: str, industry: str) -> int:
    path = Path(schema_dir) / "_schema.json"
    if not path.exists():
        return 0
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if not isinstance(schema, Mapping):
        return 0
    existing = store.get_classification_map(account_id)
    created = 0
    for table_fqn, table in schema.items():
        # Discovery stores non-table metadata (for example FK constraints) in
        # top-level ``__*`` arrays. Legacy schema files also store a table's
        # columns directly as a list instead of under {"columns": [...]}.
        # Neither shape should make applying a compliance profile fail.
        if str(table_fqn).startswith("__"):
            continue
        if isinstance(table, list):
            columns = table
        elif isinstance(table, Mapping):
            columns = table.get("columns", []) or []
        else:
            continue
        if not isinstance(columns, list):
            continue
        for column in columns:
            if not isinstance(column, Mapping):
                continue
            name = str(
                column.get("name")
                or column.get("COLUMN_NAME")
                or column.get("column_name")
                or ""
            ).strip()
            if not name:
                continue
            key = f"{str(table_fqn).upper()}.{name.upper()}"
            current = existing.get(key)
            # Admin-reviewed classifications are authoritative — never
            # overwrite a human decision on re-import.
            if current and current.get("reviewed"):
                continue
            # An existing UNREVIEWED auto-classification is refreshed against
            # the current classifier: re-import used to skip every existing
            # row, so a column first classified before its pattern existed
            # (e.g. DOCTOR_NAME before the "doctor" PII pattern was added)
            # kept its stale empty-tag row forever, and re-applying the
            # profile silently changed nothing. Only rewrite when the
            # detection actually differs, so we don't churn timestamps.
            detected = classify_column(name, industry)
            if current:
                if set(current.get("tags") or []) == set(detected["tags"]):
                    continue  # unchanged — leave it (and its timestamp) alone
            store.save_classification(
                account_id,
                str(table_fqn),
                name,
                sensitivity=detected["sensitivity"],
                identifiability=detected["identifiability"],
                tags=detected["tags"],
                confidence=detected["confidence"],
                reviewed=False,
                reviewed_by="",
                mask_strategy=detected["mask_strategy"],
                source="auto",
            )
            created += 1
    return created


def import_legacy_masking(account_id: str, masking_config: dict) -> int:
    imported = 0
    existing = store.get_classification_map(account_id)
    for table, config in (masking_config or {}).items():
        for column in config.get("masked_fields", []) or []:
            key = f"{str(table).upper()}.{str(column).upper()}"
            if key in existing:
                continue
            store.save_classification(
                account_id,
                table,
                column,
                sensitivity="RESTRICTED",
                identifiability="DIRECT",
                tags=["PII"],
                confidence=1.0,
                reviewed=False,
                reviewed_by="",
                mask_strategy="redact",
                source="legacy_masking",
            )
            imported += 1
    return imported
