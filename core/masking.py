"""
core/masking.py
───────────────
Local, zero-network field-level data masking for KB generation.

Entry points
------------
detect_sensitive_columns(columns)  → {col_name: strategy}
    Regex scan of column names + type-based free-text detection.
    Used by the admin UI to pre-check likely-PII fields when the admin
    enables masking on a table.

scan_values_for_pii(rows, col_defs)  → {col_name: {pii_type, strategy, confidence}}
    Scan actual sample values for value-level PII patterns (email, SSN, PAN,
    Aadhaar, credit card, IP, phone).  High-confidence hits (>60%) are merged
    into masked_fields in auto mode.

mask_rows(rows, masked_fields, columns, seed_key="") → list[dict]
    Apply per-column masking strategies to a sample of real rows.
    When seed_key is non-empty, each value is masked deterministically:
    the same real value always maps to the same fake value across all
    tables — preserving FK consistency without storing a lookup table.

Performance
-----------
Faker is imported once and reused.  Falls back to stdlib-only masking
if Faker is not installed.  Deterministic path bypasses Faker entirely
and uses a seeded random.Random instance per cell.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac_mod
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
    # Free-text / narrative columns — may contain anything; always redact.
    # Matched AFTER the more specific patterns above so that e.g. "address_line"
    # hits the address strategy, not the free_text strategy.
    (r"\bnotes?\b|\bcomments?\b|\bremarks?\b|description|narrative|"
     r"\bmemo\b|\bmessage\b|address_line|reason_text|feedback|"
     r"\binstructions?\b|\bsummary\b|\btranscript\b|\bdetails?\b|"
     r"\bbody\b|text_field|\bcontent\b|\bfreetext\b|free[_\s]?text",
     "free_text"),
]

# ── Type-based free-text detection ────────────────────────────────────────────
_FREE_TEXT_BASE_TYPES = {
    "text", "ntext", "longtext", "mediumtext", "tinytext",
    "clob", "nclob", "long", "blob",
}
_LONG_VARCHAR_THRESHOLD = 500

_SAFE_LONG_TEXT_NAMES = {
    "status", "type", "flag", "gender", "sex", "category", "class",
    "department", "dept", "division", "group", "team", "role", "level",
    "grade", "rank", "priority", "severity", "state", "phase", "stage",
    "mode", "method", "code", "indicator", "active", "enabled",
    "country", "region", "zone", "area", "branch", "location", "title",
    "currency", "unit", "format", "language", "locale", "timezone",
}


def _is_free_text_type(col_type: str) -> bool:
    """
    Return True if the column type indicates an unbounded or very long text
    field likely to contain unstructured, potentially sensitive content.
    """
    ct = (col_type or "").lower().strip()
    base = ct.split("(")[0].strip()
    if base in _FREE_TEXT_BASE_TYPES:
        return True
    if "varchar" in base or "character varying" in base:
        m = re.search(r"\((-?\d+)\)", ct)
        if m:
            length = int(m.group(1))
            if length < 0 or length > _LONG_VARCHAR_THRESHOLD:
                return True
    return False


# ── Value-level PII regex patterns ────────────────────────────────────────────
# Used by scan_values_for_pii() to detect PII in actual sample data.

_RE_EMAIL   = re.compile(r"^[\w.+\-]+@[\w\-]+\.[\w.]{2,}$", re.IGNORECASE)
_RE_SSN     = re.compile(r"^\d{3}-\d{2}-\d{4}$")
_RE_PAN     = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")            # Indian PAN card
_RE_AADHAAR = re.compile(r"^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$") # Indian Aadhaar
_RE_IP      = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")

# Phone — must have 7-15 digits; allow +, spaces, hyphens, parens, dots
_RE_PHONE_FMT = re.compile(r"^\+?[\d\s\-\(\)\.]{7,20}$")


def _is_phone_like(v: str) -> bool:
    digit_count = sum(c.isdigit() for c in v)
    return 7 <= digit_count <= 15 and bool(_RE_PHONE_FMT.match(v))


def _luhn_check(n: str) -> bool:
    """Validate a digit string using the Luhn algorithm."""
    digits = [int(c) for c in n if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _could_be_credit_card(v: str) -> bool:
    """True if v looks like a credit card number (13-19 digits, passes Luhn)."""
    stripped = _RE_CC_STRIP.sub("", v)
    if not stripped.isdigit():
        return False
    return _luhn_check(stripped)


# ── Value-scan type-skip sets (module-level so they aren't rebuilt each call) ─
_NUMERIC_TYPE_HINTS: frozenset[str] = frozenset({
    "int", "bigint", "smallint", "tinyint", "decimal",
    "numeric", "float", "real", "money", "number", "bit",
})
_DATE_TYPE_HINTS: frozenset[str] = frozenset({
    "date", "time", "datetime", "timestamp",
})

# Pre-compiled regex for stripping separators in credit-card candidates.
# Defined at module level so it isn't recompiled on every _could_be_credit_card call.
_RE_CC_STRIP = re.compile(r"[\s\-]")

# Maps detected PII type → masking strategy
_VALUE_PII_STRATEGY: dict[str, str] = {
    "email":       "email",
    "ssn":         "ssn",
    "pan":         "ssn",        # Indian PAN — treat as government ID
    "aadhaar":     "ssn",        # Indian Aadhaar — treat as government ID
    "credit_card": "credit_card",
    "ip":          "ip_address",
    "phone":       "phone",
}

# Ordered list for stable priority when multiple types could match one value
_VALUE_PII_ORDER = ["email", "ssn", "pan", "aadhaar", "credit_card", "ip", "phone"]


def _value_pii_types(v: str) -> list[str]:
    """Return list of PII types that match value v."""
    matches: list[str] = []
    if _RE_EMAIL.match(v):
        matches.append("email")
    if _RE_SSN.match(v):
        matches.append("ssn")
    if _RE_PAN.match(v):
        matches.append("pan")
    if _RE_AADHAAR.match(v):
        matches.append("aadhaar")
    if _could_be_credit_card(v):
        matches.append("credit_card")
    if _RE_IP.match(v):
        matches.append("ip")
    elif _is_phone_like(v):          # phone check: skip if already matched as IP
        matches.append("phone")
    return matches


def _check_value_pii(values: list[str]) -> tuple[str, float] | None:
    """
    Examine a sample of string values for a single column.

    Returns (pii_type, confidence) if >60 % of values match a single PII
    pattern, else None.  When multiple types match (e.g. Aadhaar also looks
    like a phone number), the highest-priority type wins.
    """
    if not values:
        return None
    total = len(values)
    counts: dict[str, int] = {k: 0 for k in _VALUE_PII_ORDER}
    for v in values:
        for pii_type in _value_pii_types(v):
            counts[pii_type] += 1

    best_type: str | None = None
    best_ratio = 0.0
    for pii_type in _VALUE_PII_ORDER:
        ratio = counts[pii_type] / total
        if ratio >= 0.60 and ratio > best_ratio:
            best_type = pii_type
            best_ratio = ratio
    if best_type:
        return best_type, best_ratio
    return None


def scan_values_for_pii(
    rows: list[dict],
    col_defs: list[dict],
) -> dict[str, dict]:
    """
    Scan actual sample values for value-level PII patterns.

    Returns {col_name: {"pii_type": str, "strategy": str, "confidence": float}}
    for columns where >60 % of non-null values match a PII pattern.

    Only examines string/text columns — numeric and date columns are skipped
    because patterns like SSN or phone would cause false positives on
    employee IDs, revenue figures, etc.
    """
    if not rows:
        return {}

    col_type_map = {c["name"]: c.get("type", "varchar").lower() for c in col_defs}
    result: dict[str, dict] = {}

    for col_name, col_type in col_type_map.items():
        ct_base = col_type.split("(")[0].strip()
        if any(t in ct_base for t in _NUMERIC_TYPE_HINTS):
            continue
        if any(t in ct_base for t in _DATE_TYPE_HINTS):
            continue

        values = [
            str(row[col_name]).strip()
            for row in rows
            if col_name in row and row[col_name] is not None
            and str(row[col_name]).strip()
        ]
        if not values:
            continue

        hit = _check_value_pii(values)
        if hit:
            pii_type, confidence = hit
            strategy = _VALUE_PII_STRATEGY.get(pii_type, "redact")
            result[col_name] = {
                "pii_type":  pii_type,
                "strategy":  strategy,
                "confidence": round(confidence, 2),
            }
            log.info(
                "masking: value-scan detected %s in %r (%.0f%% match → strategy=%s)",
                pii_type, col_name, confidence * 100, strategy,
            )

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def detect_sensitive_columns(columns: list[dict]) -> dict[str, str]:
    """
    Return {column_name: masking_strategy} for columns likely to contain PII.

    Detection runs in two passes:
    1. Name-based regex scan (_PII_PATTERNS).
    2. Type-based free-text detection — flags unbounded text columns whose
       name does not indicate a known-safe business field.
    """
    result: dict[str, str] = {}
    for col in columns:
        col_name = col.get("name", "")
        col_type = col.get("type", "")

        # Pass 1 — name-based
        strategy = _strategy_for_name(col_name)
        if strategy:
            result[col_name] = strategy
            continue

        # Pass 2 — type-based free-text detection
        if _is_free_text_type(col_type):
            name_lower = col_name.lower()
            if not any(hint in name_lower for hint in _SAFE_LONG_TEXT_NAMES):
                result[col_name] = "free_text"
                log.debug(
                    "masking: type-based free_text flag on %r (type=%r)",
                    col_name, col_type,
                )

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
    "free_text":     "→ [REDACTED TEXT]",
}


def strategy_for_field(field: str, col_type: str = "varchar") -> str:
    """Resolve the masking strategy for a field name/type pair."""
    s = _strategy_for_name(field)
    if s:
        return s
    ct = (col_type or "varchar").lower()
    if any(t in ct for t in ("int", "bigint", "decimal", "numeric",
                             "float", "real", "money", "number")):
        return "numeric_shift"
    if any(t in ct for t in ("date", "time", "timestamp")):
        return "date_shift"
    if _is_free_text_type(ct):
        return "free_text"
    return "text_mask"


def get_strategy_map(masked_fields: set[str], col_defs: list[dict]) -> dict[str, str]:
    """
    Return {field_name: strategy_name} for every field in *masked_fields*.
    """
    col_type_map = {c["name"]: c.get("type", "varchar") for c in col_defs}
    result: dict[str, str] = {}
    for field in masked_fields:
        result[field] = strategy_for_field(field, col_type_map.get(field, "varchar"))
    return result


def mask_rows(
    rows: list[dict],
    masked_fields: set[str],
    columns: list[dict],
    seed_key: str = "",
    strategy_overrides: dict[str, str] | None = None,
) -> list[dict]:
    """
    Apply column-level masking to *rows*.

    Parameters
    ----------
    rows             : real sample rows from the DB
    masked_fields    : set of column names to mask
    columns          : [{name, type}, ...] column metadata
    seed_key         : when non-empty, use HMAC-based deterministic masking
                       so the same real value → same fake across all tables.
                       Typically set to the account_id.
    strategy_overrides: per-column strategy overrides (from value-scan).
                       These take highest priority over name/type detection.
    """
    if not rows or not masked_fields:
        return rows

    col_type_map = {c["name"]: c.get("type", "varchar") for c in columns}
    strategy_map: dict[str, str] = {}
    for field in masked_fields:
        # Value-based override takes highest priority
        if strategy_overrides and field in strategy_overrides:
            strategy_map[field] = strategy_overrides[field]
        else:
            strategy_map[field] = strategy_for_field(field, col_type_map.get(field, "varchar"))

    faker = _get_faker()
    return [
        _mask_row(row, strategy_map, faker,
                  seed_key=seed_key, col_type_map=col_type_map)
        for row in rows
    ]


# ── Deterministic masking (Item 3) ────────────────────────────────────────────

def _hmac_seed(value: Any, col_type: str, seed_key: str) -> int:
    """
    Derive a stable integer seed from (value, col_type, seed_key).

    Using HMAC-SHA256 ensures:
    - Same value → same seed across tables (FK consistency).
    - Different values → different seeds (no collisions within strategy).
    - seed_key isolates one account from another.

    Only called from _mask_row when seed_key is non-empty (guarded by
    ``if seed_key:``), so seed_key is always truthy here.
    """
    msg = f"{col_type}:{value}".encode("utf-8")
    digest = _hmac_mod.new(seed_key.encode("utf-8"), msg, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big")


def _apply_det(value: Any, strategy: str, rng: random.Random) -> Any:
    """
    Deterministic masking using a pre-seeded RNG.

    Mirrors _apply() but replaces all random calls with rng.* so the same
    (value, strategy, seed_key) always produces the same output.
    """
    if strategy == "redact":
        return "[REDACTED]"
    if strategy == "free_text":
        return "[REDACTED TEXT]"
    if strategy == "name":
        return f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
    if strategy == "first_name":
        return rng.choice(_FIRST)
    if strategy == "last_name":
        return rng.choice(_LAST)
    if strategy == "email":
        digits = "".join(rng.choices(string.digits, k=4))
        domain = rng.choice(["example.com", "mail.test", "test.org", "sample.net"])
        return f"user{digits}@{domain}"
    if strategy == "phone":
        return (
            f"+1-{rng.randint(200, 999)}-"
            f"{rng.randint(100, 999)}-"
            f"{rng.randint(1000, 9999)}"
        )
    if strategy == "address":
        return f"{rng.randint(1, 9999)} {rng.choice(_STREET_NAMES)} St"
    if strategy == "city":
        return rng.choice(_CITIES)
    if strategy == "zip":
        return str(rng.randint(10000, 99999))
    if strategy == "state":
        return rng.choice(_STATES)
    if strategy == "country":
        return rng.choice(_COUNTRIES)
    if strategy == "ssn":
        return (
            f"{rng.randint(100, 999)}-"
            f"{rng.randint(10, 99)}-"
            f"{rng.randint(1000, 9999)}"
        )
    if strategy == "credit_card":
        return f"****-****-****-{rng.randint(1000, 9999)}"
    if strategy == "ip_address":
        return (
            f"{rng.randint(10, 192)}.{rng.randint(0, 255)}."
            f"{rng.randint(0, 255)}.{rng.randint(1, 254)}"
        )
    if strategy == "coordinate":
        try:
            return round(float(value) + rng.uniform(-0.5, 0.5), 6)
        except Exception:
            return value
    if strategy == "birthdate":
        return _shift_date(value, max_days=730, rng=rng)
    if strategy == "date_shift":
        return _shift_date(value, max_days=5, rng=rng)
    if strategy == "salary":
        return _shift_numeric(value, pct=0.15, rng=rng)
    if strategy == "numeric_shift":
        return _shift_numeric(value, pct=0.10, rng=rng)
    if strategy == "text_mask":
        return _mask_text(str(value), rng=rng)
    return value  # unknown strategy — pass through


# ── Internal helpers ──────────────────────────────────────────────────────────

def _strategy_for_name(col_name: str) -> str | None:
    cn = col_name.lower()
    for pattern, strategy in _PII_PATTERNS:
        if re.search(pattern, cn):
            return strategy
    return None


def _mask_row(
    row: dict,
    strategy_map: dict[str, str],
    faker,
    seed_key: str = "",
    col_type_map: dict[str, str] | None = None,
) -> dict:
    result = dict(row)
    for field, strategy in strategy_map.items():
        if field not in result or result[field] is None:
            continue
        try:
            if seed_key:
                col_type = (col_type_map or {}).get(field, "varchar")
                seed = _hmac_seed(result[field], col_type, seed_key)
                rng = random.Random(seed)
                result[field] = _apply_det(result[field], strategy, rng)
            else:
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
    if strategy == "free_text":
        return "[REDACTED TEXT]"

    return value  # unknown strategy — pass through


def _shift_date(
    value: Any,
    max_days: int = 5,
    rng: random.Random | None = None,
) -> Any:
    from datetime import timedelta, date, datetime
    _rng = rng or random
    shift = _rng.randint(-max_days, max_days)
    if isinstance(value, datetime):
        return value + timedelta(days=shift)
    if isinstance(value, date):
        return value + timedelta(days=shift)
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            d = datetime.strptime(str(value), fmt)
            return (d + timedelta(days=shift)).strftime(fmt)
        except ValueError:
            continue
    return value


def _shift_numeric(
    value: Any,
    pct: float = 0.10,
    rng: random.Random | None = None,
) -> Any:
    _rng = rng or random
    try:
        n = float(value)
        n_new = n * (1.0 + _rng.uniform(-pct, pct))
        if isinstance(value, int):
            return max(0, int(round(n_new)))
        if isinstance(value, str) and value.isdigit():
            return max(0, int(round(n_new)))
        return round(n_new, 2)
    except Exception:
        return value


def _mask_text(value: str, rng: random.Random | None = None) -> str:
    """Format-preserving text mask: replace alpha→alpha, digit→digit, keep rest."""
    _rng = rng or random
    result = []
    for ch in value:
        if ch.isalpha():
            pool = string.ascii_lowercase if ch.islower() else string.ascii_uppercase
            result.append(_rng.choice(pool))
        elif ch.isdigit():
            result.append(str(_rng.randint(0, 9)))
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
        Faker.seed(42)
        log.debug("masking: Faker loaded")
        return _faker_instance
    except ImportError:
        log.info("masking: Faker not installed — using stdlib fallback")
        _faker_instance = False  # sentinel so we only try once
        return None


# ── Stdlib fallback data pools ────────────────────────────────────────────────
# 30 names each so deterministic sampling has good variety

_FIRST = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley",
    "Quinn", "Avery", "Blake", "Cameron", "Dana", "Ellis",
    "Finley", "Harper", "Hayden", "Jamie", "Jesse", "Kendall",
    "Lee", "Logan", "London", "Madison", "Mason", "Peyton",
    "Reese", "River", "Rowan", "Sage", "Skylar", "Sydney",
]
_LAST = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
    "Miller", "Davis", "Wilson", "Moore", "Anderson", "Lee",
    "Taylor", "Thomas", "Jackson", "White", "Harris", "Martin",
    "Thompson", "Young", "Lewis", "Walker", "Hall", "Allen",
    "King", "Wright", "Scott", "Green", "Baker", "Adams",
]
_CITIES = [
    "Springfield", "Riverside", "Fairview", "Madison", "Georgetown",
    "Burlington", "Salem", "Greenville", "Franklin", "Arlington",
    "Clinton", "Milford", "Newport", "Chester", "Ashland",
]
_STATES   = ["CA", "TX", "NY", "FL", "IL", "WA", "OH", "GA", "NC", "PA"]
_COUNTRIES = ["US", "GB", "CA", "AU", "DE", "FR", "SG", "IN", "JP", "BR"]
_STREET_NAMES = [
    "Oak", "Maple", "Cedar", "Pine", "Elm", "Birch",
    "Willow", "Spruce", "Aspen", "Walnut", "Chestnut", "Magnolia",
]


def _stdlib_name() -> str:
    return f"{random.choice(_FIRST)} {random.choice(_LAST)}"
