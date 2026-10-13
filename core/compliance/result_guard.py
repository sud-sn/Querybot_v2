from __future__ import annotations

import hashlib
import hmac
import re
from typing import Any

from core.compliance.models import PolicyDecision
from core.process_secrets import env_secret_or_random


_NAME_FIELD = re.compile(r"(?:^|[._])(?:DOCTOR|PHYSICIAN|PRESCRIBER|PROVIDER|PATIENT|PERSON)(?:_|[.]|$)|(?:NAME|_NM)$", re.I)
_IDENTIFIER_FIELD = re.compile(
    r"(?:^|[._])(?:RX|PRESCRIPTION|MRN|MEDICAL_RECORD|PATIENT|MEMBER)(?:_|[.]|$).*(?:ID|KEY|NUM|NUMBER|NO)$|"
    r"(?:^|[._])(?:RX_NUM|RX_NUMBER|MRN|PRESCRIPTION_ID)(?:[.]|$)",
    re.I,
)


def _pseudonym_secret() -> bytes:
    # See core/process_secrets.py: the literal string this used to fall back
    # to was public, so anyone who had read this file could recompute the
    # HMAC below for a candidate value and de-mask any pseudonym shown to a
    # deployment that never set one of these three env vars.
    value = env_secret_or_random(
        "PII_PSEUDONYM_SECRET", "PORTAL_SESSION_SECRET", "SESSION_SECRET",
        purpose="pii_pseudonym",
    )
    return value.encode("utf-8")


def _digest(value: str, *, account_id: str, source: str) -> str:
    payload = f"{account_id}|{source.upper()}|{value}".encode("utf-8")
    return hmac.new(_pseudonym_secret(), payload, hashlib.sha256).hexdigest().upper()


# How many hex digits an alias keeps. Two people given the same alias are shown as one: their rows read alike and a
# chart keyed by the label adds them together. A letter and 3 digits (26 x 16^3, about 10^5 aliases) gave a table of
# a few thousand patients dozens of shared aliases; a letter and 8 digits (about 10^11) leaves a table of 10^5 people
# an expected 0.05. Identifiers (prescription numbers, record numbers) run to millions, so they keep 12 (16^12).
_NAME_CODE_DIGITS = 8
_IDENTIFIER_CODE_DIGITS = 12


def _safe_name_alias(value: str, *, account_id: str, source: str) -> str:
    digest = _digest(value, account_id=account_id, source=source)
    initial = chr(ord("A") + (int(digest[:2], 16) % 26))
    code = digest[2:2 + _NAME_CODE_DIGITS]
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
    return f"{prefix}-{digest[:_IDENTIFIER_CODE_DIGITS]}"


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
        # The warehouse decides the case of the names it returns: Snowflake and
        # Oracle upper-case unquoted ones, so `customer_name` in the SQL comes
        # back as CUSTOMER_NAME. An exact-case lookup missed it and released
        # the column unmasked, so an output is found by name, whatever the case.
        keys_by_name: dict[str, list] = {}
        for key in item:
            keys_by_name.setdefault(str(key).lower(), []).append(key)
        for output_column, sources in lineage.items():
            if str(output_column).lower() in exempt_keys:
                continue
            strategies = [
                (decision.masking[source], source)
                for source in sources
                if source in decision.masking
            ]
            if not strategies:
                continue
            strategy, source = strategies[0]
            for key in keys_by_name.get(str(output_column).lower(), []):
                item[key] = _mask(item[key], strategy, account_id, source=source)
        protected.append(item)
    return protected
