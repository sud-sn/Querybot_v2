from __future__ import annotations

import json
import re
from pathlib import Path

import store


_PATTERNS = {
    "PII": re.compile(
        r"(name|email|phone|mobile|address|ssn|social.?security|passport|national.?id|dob|birth|"
        r"doctor|physician)",
        re.I,
    ),
    "PCI": re.compile(r"(card.?number|pan|cvv|cvc|expiry|cardholder)", re.I),
    "KYC_AML": re.compile(r"(kyc|aml|sanction|pep|risk.?rating|beneficial.?owner)", re.I),
    "FINANCIAL": re.compile(
        r"(account|routing|iban|swift|balance|revenue|profit|loss|income|payment|transaction|credit|debit)",
        re.I,
    ),
    "PHI": re.compile(
        r"(patient|diagnos|medical|health|allerg|condition|mrn|member.?id|clinical)",
        re.I,
    ),
    "PRESCRIPTION": re.compile(
        r"(prescription|rx|drug|medication|compound|ingredient|prescriber|dosage|ndc)",
        re.I,
    ),
    "PAYMENT": re.compile(r"(payment|payer|claim|charge|copay|insurance)", re.I),
}


def _default_mask_strategy(column_name: str, tags: list[str]) -> str:
    name = str(column_name or "")
    if set(tags) & {"PII", "PRESCRIPTION"}:
        if re.search(r"(?:doctor|physician|prescriber|provider|patient|person).*(?:name|_nm)?$", name, re.I):
            return "safe_alias_name"
        if re.search(r"(?:rx|prescription|mrn|medical.?record|patient|member).*(?:id|key|num|number|no)$", name, re.I):
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
    existing = store.get_classification_map(account_id)
    created = 0
    for table_fqn, table in schema.items():
        for column in (table or {}).get("columns", []) or []:
            name = str(
                column.get("name")
                or column.get("COLUMN_NAME")
                or column.get("column_name")
                or ""
            ).strip()
            if not name:
                continue
            key = f"{str(table_fqn).upper()}.{name.upper()}"
            if key in existing:
                continue
            detected = classify_column(name, industry)
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
