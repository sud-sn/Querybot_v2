from __future__ import annotations

import hashlib
import hmac
import os
import re
from typing import Any

from core.compliance.models import PolicyDecision


_NAME_FIELD = re.compile(r"(?:^|[._])(?:DOCTOR|PHYSICIAN|PRESCRIBER|PROVIDER|PATIENT|PERSON)(?:_|[.]|$)|(?:NAME|_NM)$", re.I)
_IDENTIFIER_FIELD = re.compile(
    r"(?:^|[._])(?:RX|PRESCRIPTION|MRN|MEDICAL_RECORD|PATIENT|MEMBER)(?:_|[.]|$).*(?:ID|KEY|NUM|NUMBER|NO)$|"
    r"(?:^|[._])(?:RX_NUM|RX_NUMBER|MRN|PRESCRIPTION_ID)(?:[.]|$)",
    re.I,
)


def _pseudonym_secret() -> bytes:
    value = (
        os.getenv("PII_PSEUDONYM_SECRET")
        or os.getenv("PORTAL_SESSION_SECRET")
        or os.getenv("SESSION_SECRET")
        or "querybot-development-pseudonym-secret"
    )
    return value.encode("utf-8")


def _digest(value: str, *, account_id: str, source: str) -> str:
    payload = f"{account_id}|{source.upper()}|{value}".encode("utf-8")
    return hmac.new(_pseudonym_secret(), payload, hashlib.sha256).hexdigest().upper()


def _safe_name_alias(value: str, *, account_id: str, source: str) -> str:
    digest = _digest(value, account_id=account_id, source=source)
    initial = chr(ord("A") + (int(digest[:2], 16) % 26))
    code = digest[2:5]
    upper_source = source.upper()
    if any(token in upper_source for token in ("DOCTOR", "PHYSICIAN", "PRESCRIBER", "PROVIDER")):
        return f"Dr. {initial}-{code}"
    if "PATIENT" in upper_source:
        return f"Patient {initial}-{code}"
    return f"Person {initial}-{code}"


def _safe_identifier_alias(value: str, *, account_id: str, source: str) -> str:
    digest = _digest(value, account_id=account_id, source=source)
    upper_source = source.upper()
    prefix = "RX" if any(token in upper_source for token in ("RX", "PRESCRIPTION")) else (
        "MRN" if "MRN" in upper_source or "MEDICAL_RECORD" in upper_source else "ID"
    )
    return f"{prefix}-{digest[:4]}"


def _mask(value: Any, strategy: str, seed: str, *, source: str = "") -> Any:
    if value is None:
        return None
    strategy = (strategy or "redact").lower()
    text = str(value)
    if strategy in {"safe_alias", "smart_alias"}:
        if _NAME_FIELD.search(source):
            return _safe_name_alias(text, account_id=seed, source=source)
        if _IDENTIFIER_FIELD.search(source):
            return _safe_identifier_alias(text, account_id=seed, source=source)
        return "[REDACTED]"
    if strategy in {"safe_alias_name", "pseudonym_name"}:
        return _safe_name_alias(text, account_id=seed, source=source)
    if strategy in {"safe_alias_identifier", "pseudonym_identifier"}:
        return _safe_identifier_alias(text, account_id=seed, source=source)
    if strategy == "partial_original":
        return text[:1] + ("*" * max(3, len(text) - 1))
    if strategy == "partial":
        return ("*" * max(4, len(text) - 4)) + text[-4:]
    if strategy in {"hash", "tokenize"}:
        digest = _digest(text, account_id=seed, source=source)[:16].lower()
        return f"TKN-{digest}" if strategy == "tokenize" else digest
    if strategy == "null":
        return None
    return "[REDACTED]"


def protect_rows(
    rows: list[dict],
    decision: PolicyDecision,
    lineage: dict[str, list[str]],
    *,
    account_id: str,
    mask_exempt_outputs: set[str] | None = None,
) -> list[dict]:
    if not rows or not decision.masking:
        return [dict(row) for row in rows]
    exempt_keys = {str(name).lower() for name in (mask_exempt_outputs or set())}
    protected = []
    for row in rows:
        item = dict(row)
        for output_column, sources in lineage.items():
            if str(output_column).lower() in exempt_keys:
                continue
            strategies = [
                (decision.masking[source], source)
                for source in sources
                if source in decision.masking
            ]
            if strategies and output_column in item:
                strategy, source = strategies[0]
                item[output_column] = _mask(
                    item[output_column], strategy, account_id, source=source
                )
        protected.append(item)
    return protected
