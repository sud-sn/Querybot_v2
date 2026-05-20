"""
core/masking.py
───────────────
Local, zero-network field-level data masking for KB generation.

Entry points
------------
detect_sensitive_columns(columns)  → {col_name: strategy}
    Regex scan of column names.  Used by the admin UI to pre-check likely-PII
    fields when the admin enables masking on a table.

mask_rows(rows, masked_fields, columns) → list[dict]
    Apply per-column masking strategies to a sample of real rows.
    Only fields listed in masked_fields are altered; everything else is
    returned verbatim so the LLM still sees real categorical values,
    real numeric ranges, real date structures.

Performance
-----------
Faker is imported once and reused.  At ~0.3 ms per field call and
5 rows × 30 columns = 150 calls, total masking time is < 50 ms per table.
Falls back to stdlib-only masking if Faker is not installed.
"""

from __future__ import annotations

import logging
import random
import re
import string
from typing import Any

log = logging.getLogger(__name__)

# ── PII detection patterns ────────────────────────────────────────────────────
# Ordered from most-specific to least-specific.
# Each tuple: (regex applied to lower-cased column name, masking strategy)
_PII_PATTERNS: list[tuple[str, str]] = [
    # Must-redact (secrets / credentials)
    (r"password|passwd|pwd|api[_\s]?key|secret[_\s]?key|private[_\s]?key|"
     r"\btoken\b|hash\b|salt\b|\bpin\b", "redact"),
    # Government IDs
    (r"ssn|social[_\s]?sec|national[_\s]?id|nid\b|sin\b|tax[_\s]?id|"
     r"fiscal[_\s]?id|passport", "ssn"),
    # Financial card numbers
    (r"credit[_\s]?card|card[_\s]?no|card[_\s]?num|pan\b", "credit_card"),
    # Contact
    (r"email|e[_\s]mail", "email"),
    (r"phone|mobile|cell\b|tel\b|fax", "phone"),
    # Name — order matters: first/last before bare 'name'
    (r"first[_\s]?name|fname|given[_\s]?name|forename", "first_name"),
    (r"last[_\s]?name|lname|surname|family[_\s]?name", "last_name"),
    (r"(?<![a-z])name(?![a-z])|full[_\s]?name|display[_\s]?name", "name"),
    # Address components
    (r"address|street|addr\b", "address"),
    (r"(?<![a-z])city(?![a-z])", "city"),
    (r"zip|postal|post[_\s]?code", "zip"),
    (r"(?<![a-z])state(?![a-z])", "state"),
    (r"(?<![a-z])country(?![a-z])", "country"),
    # Date of birth (shift more aggressively than normal dates)
    (r"birth[_\s]?date|date[_\s]?of[_\s]?birth|dob\b|born\b", "birthdate"),
    # Compensation / financial (shift ±15 %)
    (r"salary|wage\b|income|compensation|base[_\s]?pay|annual[_\s]?pay", "salary"),
    # IP / geo
    (r"ip[_\s]?address|ip[_\s]?addr\b", "ip_address"),
    (r"latitude|longitude|lat\b|lng\b|lon\b", "coordinate"),
]


# ── Public API ────────────────────────────────────────────────────────────────

def detect_sensitive_columns(columns: list[dict]) -> dict[str, str]:
    """
    Return {column_name: masking_strategy} for columns likely to contain PII.

    ``columns`` is a list of dicts with at least a ``name`` key (and optionally
    ``type``).  Only columns whose names match a PII pattern are returned.
    """
    result: dict[str, str] = {}
    for col in columns:
        strategy = _strategy_for_name(col.get("name", ""))
        if strategy:
            result[col["name"]] = strategy
    return result


# Human-readable labels for each masking strategy (used in UI + egress log).
STRATEGY_LABELS: dict[str, str] = {
    "redact":        "→ [REDACTED]",
    "name":          "→ fake full name",
    "first_name":    "→ fake first name",
    "last_name":     "→ fake last name",
    "email":         "→ fake email",
    "phone":         "→ fake phone number",
    "address":       "→ fake address",
    "city":          "→ fake city",
    "zip":           "→ fake zip code",
    "state":         "→ fake state",
    "country":       "→ fake country code",
    "ssn":           "→ fake SSN",
    "credit_card":   "→ ****-****-****-XXXX",
    "ip_address":    "→ fake IP address",
    "coordinate":    "→ shifted ±0.5°",
    "birthdate":     "→ shifted ±730 days",
    "date_shift":    "→ shifted ±5 days",
    "salary":        "→ shifted ±15%",
    "numeric_shift": "→ shifted ±10%",
    "text_mask":     "→ format-preserving mask",
}


def get_strategy_map(masked_fields: set[str], col_defs: list[dict]) -> dict[str, str]:
    """
    Return {field_name: strategy_name} for every field in *masked_fields*.

    Uses the same resolution logic as ``mask_rows`` — name-based PII pattern
    first, then type-based fallback.  Strategy names are the raw keys used
    internally (e.g. ``"email"``, ``"numeric_shift"``).
    """
    col_type_map = {c["name"]: c.get("type", "varchar") for c in col_defs}
    result: dict[str, str] = {}
    for field in masked_fields:
        s = _strategy_for_name(field)
        if not s:
            ct = col_type_map.get(field, "varchar").lower()
            if any(t in ct for t in ("int", "bigint", "decimal", "numeric",
                                     "float", "real", "money", "number")):
                s = "numeric_shift"
            elif any(t in ct for t in ("date", "time", "timestamp")):
                s = "date_shift"
            else:
                s = "text_mask"
        result[field] = s
    return result


def mask_rows(
    rows: list[dict],
    masked_fields: set[str],
    columns: list[dict],
) -> list[dict]:
    """
    Apply column-level masking to *rows*.

    - Fields not in *masked_fields* are left unchanged.
    - The masking strategy for each field is resolved from the column name.
      If no named strategy matches, a type-based fallback is used.
    - None values are passed through as None (no masking needed).
    """
    if not rows or not masked_fields:
        return rows

    col_type_map = {c["name"]: c.get("type", "varchar") for c in columns}
    strategy_map: dict[str, str] = {}
    for field in masked_fields:
        s = _strategy_for_name(field)
        if not s:
            ct = col_type_map.get(field, "varchar").lower()
            if any(t in ct for t in ("int", "bigint", "decimal", "numeric",
                                     "float", "real", "money", "number")):
                s = "numeric_shift"
            elif any(t in ct for t in ("date", "time", "timestamp")):
                s = "date_shift"
            else:
                s = "text_mask"
        strategy_map[field] = s

    faker = _get_faker()
    return [_mask_row(row, strategy_map, faker) for row in rows]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _strategy_for_name(col_name: str) -> str | None:
    cn = col_name.lower()
    for pattern, strategy in _PII_PATTERNS:
        if re.search(pattern, cn):
            return strategy
    return None


def _mask_row(row: dict, strategy_map: dict[str, str], faker) -> dict:
    result = dict(row)
    for field, strategy in strategy_map.items():
        if field not in result or result[field] is None:
            continue
        try:
            result[field] = _apply(result[field], strategy, faker)
        except Exception:
            result[field] = "[MASKED]"
    return result


def _apply(value: Any, strategy: str, faker) -> Any:  # noqa: C901
    if strategy == "redact":
        return "[REDACTED]"

    if strategy == "name":
        return faker.name() if faker else _stdlib_name()
    if strategy == "first_name":
        return faker.first_name() if faker else random.choice(_FIRST)
    if strategy == "last_name":
        return faker.last_name() if faker else random.choice(_LAST)
    if strategy == "email":
        return faker.email() if faker else f"user{random.randint(100,9999)}@example.com"
    if strategy == "phone":
        return faker.phone_number() if faker else (
            f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"
        )
    if strategy == "address":
        return faker.street_address() if faker else f"{random.randint(1,9999)} Oak Street"
    if strategy == "city":
        return faker.city() if faker else random.choice(_CITIES)
    if strategy == "zip":
        return faker.zipcode() if faker else f"{random.randint(10000,99999)}"
    if strategy == "state":
        return faker.state_abbr() if faker else random.choice(_STATES)
    if strategy == "country":
        return faker.country_code() if faker else random.choice(_COUNTRIES)
    if strategy == "ssn":
        return f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"
    if strategy == "credit_card":
        return f"****-****-****-{random.randint(1000,9999)}"
    if strategy == "ip_address":
        return f"{random.randint(10,192)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    if strategy == "coordinate":
        try:
            return round(float(value) + random.uniform(-0.5, 0.5), 6)
        except Exception:
            return value
    if strategy == "birthdate":
        return _shift_date(value, max_days=730)
    if strategy == "date_shift":
        return _shift_date(value, max_days=5)
    if strategy == "salary":
        return _shift_numeric(value, pct=0.15)
    if strategy == "numeric_shift":
        return _shift_numeric(value, pct=0.10)
    if strategy == "text_mask":
        return _mask_text(str(value))

    return value  # unknown strategy — pass through


def _shift_date(value: Any, max_days: int = 5) -> Any:
    from datetime import timedelta, date, datetime

    shift = random.randint(-max_days, max_days)
    if isinstance(value, datetime):
        return value + timedelta(days=shift)
    if isinstance(value, date):
        return value + timedelta(days=shift)
    # Try common string formats
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            d = datetime.strptime(str(value), fmt)
            return (d + timedelta(days=shift)).strftime(fmt)
        except ValueError:
            continue
    return value  # unrecognised format — leave as-is


def _shift_numeric(value: Any, pct: float = 0.10) -> Any:
    try:
        n = float(value)
        n_new = n * (1.0 + random.uniform(-pct, pct))
        # Preserve integer-ness
        if isinstance(value, int):
            return max(0, int(round(n_new)))
        if isinstance(value, str) and value.isdigit():
            return max(0, int(round(n_new)))
        return round(n_new, 2)
    except Exception:
        return value


def _mask_text(value: str) -> str:
    """Format-preserving text mask: replace alpha→alpha, digit→digit, keep rest."""
    result = []
    for ch in value:
        if ch.isalpha():
            pool = string.ascii_lowercase if ch.islower() else string.ascii_uppercase
            result.append(random.choice(pool))
        elif ch.isdigit():
            result.append(str(random.randint(0, 9)))
        else:
            result.append(ch)
    return "".join(result)


# ── Faker singleton ───────────────────────────────────────────────────────────

_faker_instance = None


def _get_faker():
    global _faker_instance
    if _faker_instance is not None:
        return _faker_instance
    try:
        from faker import Faker  # type: ignore
        _faker_instance = Faker()
        Faker.seed(42)           # deterministic within a session
        log.debug("masking: Faker loaded")
        return _faker_instance
    except ImportError:
        log.info("masking: Faker not installed — using stdlib fallback")
        _faker_instance = False  # sentinel so we only try once
        return None


# ── Stdlib fallback data ──────────────────────────────────────────────────────

_FIRST = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley",
          "Quinn", "Avery", "Blake", "Cameron", "Dana", "Ellis"]
_LAST  = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
          "Miller", "Davis", "Wilson", "Moore", "Anderson", "Lee"]
_CITIES   = ["Springfield", "Riverside", "Fairview", "Madison", "Georgetown",
             "Burlington", "Salem", "Greenville", "Franklin", "Arlington"]
_STATES   = ["CA", "TX", "NY", "FL", "IL", "WA", "OH", "GA", "NC", "PA"]
_COUNTRIES = ["US", "GB", "CA", "AU", "DE", "FR", "SG", "IN", "JP", "BR"]


def _stdlib_name() -> str:
    return f"{random.choice(_FIRST)} {random.choice(_LAST)}"
