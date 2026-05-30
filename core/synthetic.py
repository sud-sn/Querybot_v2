"""
core/synthetic.py

Synthetic sample data generator for sensitive database tables.

Generates structurally-realistic fake rows from column metadata alone.
NO real database records are read — everything is generated locally on the VM.

Used in schema discovery for tables whose names suggest PII / sensitive data.
Claude receives fake names, fake IDs, fake dates — but the correct column
structure, data types, and value formats — so it can still write accurate SQL.

No external dependencies (no Faker package required).
Pure Python stdlib only.
"""

import random
import string
from datetime import date, timedelta
from typing import Any

# ── Table name patterns that trigger synthetic generation ──────────────────────
# If any of these substrings appear in a table name (case-insensitive),
# synthetic samples are used instead of real rows.
SENSITIVE_TABLE_PATTERNS = [
    # Patient / person identity
    "patient", "person", "people", "individual", "member", "beneficiary",
    "claimant", "insured", "subscriber", "enrollee",
    # Employee / staff identity
    "employee", "staff", "worker", "personnel", "hr_", "_hr",
    # Customer / contact identity
    "customer", "client", "contact", "user", "account", "prospect",
    # Medical / clinical identity and encounters
    "demographic", "clinical", "medical_record", "health_record",
    "physician", "provider", "prescriber", "caregiver",
    "encounter", "visit", "admission", "discharge",
    # Financial identity
    "taxpayer", "policyholder",
    # ── Transactional / operational tables ──────────────────────────────────
    # These tables contain row-level business events that carry PII in values
    # (customer names, drug names, amounts, dates tied to individuals) even
    # when the table name does not look like an identity table.
    # Prescription / pharmacy
    "prescription", "rx_", "_rx", "dispens", "fill", "refill",
    # Orders / transactions
    "order", "transaction", "invoice", "payment", "receipt", "sale",
    "purchase", "billing", "charge", "claim",
    # Clinical events
    "diagnosis", "procedure", "lab_result", "test_result",
    "medication", "drug_admin", "immuniz",
    # Audit / history / log — row-level event tables
    "audit_log", "audit_trail", "change_log", "event_log",
    "history", "_hist", "hist_",
    # Fact tables — always transactional in a star schema
    "fact_", "_fact",
]


def should_use_synthetic(table_name: str) -> bool:
    """Return True if this table name suggests PII / sensitive content."""
    lower = table_name.lower()
    return any(p in lower for p in SENSITIVE_TABLE_PATTERNS)


# ── Value vocabularies ────────────────────────────────────────────────────────

_FIRST_NAMES = [
    "James", "Sarah", "Robert", "Maria", "William", "Jennifer", "David",
    "Linda", "Michael", "Patricia", "Richard", "Barbara", "Joseph", "Susan",
    "Thomas", "Jessica", "Charles", "Karen", "Christopher", "Nancy", "Daniel",
    "Lisa", "Paul", "Betty", "Mark", "Margaret", "Donald", "Sandra", "George",
    "Ashley", "Kenneth", "Dorothy", "Steven", "Kimberly", "Edward", "Emily",
    "Brian", "Donna", "Ronald", "Michelle", "Anthony", "Carol", "Kevin",
    "Amanda", "Jason", "Melissa", "Matthew", "Deborah", "Raj", "Priya",
    "Chidi", "Amara", "Yuki", "Kenji", "Sofia", "Mateo", "Aisha", "Omar",
]

_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Okonkwo", "Patel", "Kim", "Singh", "Chen",
    "Yamamoto", "Castillo", "Reyes", "Diaz", "Morales", "Gutierrez",
]

_CITIES = [
    "Atlanta", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
    "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
    "Fort Worth", "Columbus", "Charlotte", "Indianapolis", "Seattle", "Denver",
    "Nashville", "Oklahoma City", "El Paso", "Boston", "Portland", "Miami",
    "Minneapolis", "New Orleans", "Cleveland", "Pittsburgh", "Sacramento",
]

_US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

_DIAGNOSIS_CODES = [
    "E11.9", "I10", "J18.9", "K21.0", "M54.5", "F32.9",
    "N18.3", "Z79.4", "J06.9", "K92.1", "G43.909", "I25.10",
]

_DRUG_CODES = [
    "MET500", "AMO250", "LIP40", "ATN25", "OME20",
    "LOS50", "AML10", "SIM20", "MET1000", "PRE5",
]

_INSURANCE_CARRIERS = [
    "BlueCross", "Aetna", "Cigna", "UnitedHealth", "Humana",
    "Anthem", "Centene", "Molina", "WellCare", "BCBS",
]

_STATUS_CODES   = ["A", "I", "C", "P", "R", "N", "H", "T"]
_YN_VALUES      = ["Y", "N"]
_ACTIVE_VALUES  = [0, 1]
_GENDER_VALUES  = ["M", "F", "U"]


def _rnd_digits(n: int) -> str:
    return "".join(random.choices(string.digits, k=n))


def _rnd_alpha(n: int) -> str:
    return "".join(random.choices(string.ascii_uppercase, k=n))


def _rnd_date(years_ago_min: int, years_ago_max: int) -> str:
    start = date.today() - timedelta(days=365 * years_ago_max)
    end   = date.today() - timedelta(days=365 * years_ago_min)
    delta = max((end - start).days, 1)
    return str(start + timedelta(days=random.randint(0, delta)))


def _rnd_datetime(years_ago_min: int, years_ago_max: int) -> str:
    d = _rnd_date(years_ago_min, years_ago_max)
    return f"{d} {random.randint(6, 22):02d}:{random.randint(0, 59):02d}:00"


# ── Column name → generator mapping ──────────────────────────────────────────
# Keys are substrings matched against lowercased column names.
# Order matters — more specific patterns first.

_COLUMN_PATTERNS: list[tuple[str, Any]] = [
    # ── Personal name ────────────────────────────────────────────────────────
    ("first_name",     lambda i: random.choice(_FIRST_NAMES)),
    ("firstname",      lambda i: random.choice(_FIRST_NAMES)),
    ("fname",          lambda i: random.choice(_FIRST_NAMES)),
    ("given_name",     lambda i: random.choice(_FIRST_NAMES)),
    ("last_name",      lambda i: random.choice(_LAST_NAMES)),
    ("lastname",       lambda i: random.choice(_LAST_NAMES)),
    ("lname",          lambda i: random.choice(_LAST_NAMES)),
    ("surname",        lambda i: random.choice(_LAST_NAMES)),
    ("family_name",    lambda i: random.choice(_LAST_NAMES)),
    ("full_name",      lambda i: f"{random.choice(_FIRST_NAMES)} {random.choice(_LAST_NAMES)}"),
    ("patient_name",   lambda i: f"{random.choice(_FIRST_NAMES)} {random.choice(_LAST_NAMES)}"),
    ("member_name",    lambda i: f"{random.choice(_FIRST_NAMES)} {random.choice(_LAST_NAMES)}"),
    # ── Contact ──────────────────────────────────────────────────────────────
    ("email",          lambda i: f"{random.choice(_FIRST_NAMES).lower()}.{random.choice(_LAST_NAMES).lower()}@example.com"),
    ("phone",          lambda i: f"({_rnd_digits(3)}) {_rnd_digits(3)}-{_rnd_digits(4)}"),
    ("mobile",         lambda i: f"({_rnd_digits(3)}) {_rnd_digits(3)}-{_rnd_digits(4)}"),
    ("fax",            lambda i: f"({_rnd_digits(3)}) {_rnd_digits(3)}-{_rnd_digits(4)}"),
    ("telephone",      lambda i: f"({_rnd_digits(3)}) {_rnd_digits(3)}-{_rnd_digits(4)}"),
    # ── Address ──────────────────────────────────────────────────────────────
    ("street",         lambda i: f"{random.randint(100,9999)} {random.choice(_LAST_NAMES)} St"),
    ("address1",       lambda i: f"{random.randint(100,9999)} {random.choice(_LAST_NAMES)} Ave"),
    ("address2",       lambda i: random.choice(["Apt 4B", "Suite 200", "Unit 7", ""])),
    ("address",        lambda i: f"{random.randint(100,9999)} {random.choice(_LAST_NAMES)} Blvd"),
    ("city",           lambda i: random.choice(_CITIES)),
    ("state",          lambda i: random.choice(_US_STATES)),
    ("zip_code",       lambda i: _rnd_digits(5)),
    ("zipcode",        lambda i: _rnd_digits(5)),
    ("zip",            lambda i: _rnd_digits(5)),
    ("postal",         lambda i: _rnd_digits(5)),
    ("country",        lambda i: "US"),
    # ── Identity / government ────────────────────────────────────────────────
    ("ssn",            lambda i: f"***-**-{_rnd_digits(4)}"),
    ("social_sec",     lambda i: f"***-**-{_rnd_digits(4)}"),
    ("tax_id",         lambda i: f"**-***{_rnd_digits(4)}"),
    ("passport",       lambda i: f"{_rnd_alpha(1)}{_rnd_digits(8)}"),
    ("license",        lambda i: f"{_rnd_alpha(1)}{_rnd_digits(7)}"),
    ("drivers",        lambda i: f"{_rnd_alpha(2)}{_rnd_digits(6)}"),
    # ── Dates ────────────────────────────────────────────────────────────────
    ("date_of_birth",  lambda i: _rnd_date(18, 90)),
    ("dob",            lambda i: _rnd_date(18, 90)),
    ("birth_date",     lambda i: _rnd_date(18, 90)),
    ("birthdate",      lambda i: _rnd_date(18, 90)),
    ("hire_date",      lambda i: _rnd_date(1, 15)),
    ("hired",          lambda i: _rnd_date(1, 15)),
    ("enroll",         lambda i: _rnd_date(0, 5)),
    ("start_date",     lambda i: _rnd_date(0, 3)),
    ("end_date",       lambda i: _rnd_date(0, 2)),
    ("created_at",     lambda i: _rnd_datetime(0, 2)),
    ("updated_at",     lambda i: _rnd_datetime(0, 1)),
    ("modified",       lambda i: _rnd_datetime(0, 1)),
    ("timestamp",      lambda i: _rnd_datetime(0, 1)),
    ("date",           lambda i: _rnd_date(0, 2)),
    # ── Demographics ─────────────────────────────────────────────────────────
    ("gender",         lambda i: random.choice(_GENDER_VALUES)),
    ("sex",            lambda i: random.choice(_GENDER_VALUES)),
    ("age",            lambda i: random.randint(18, 85)),
    ("race",           lambda i: random.choice(["W", "B", "H", "A", "O", "U"])),
    ("ethnicity",      lambda i: random.choice(["H", "N", "U"])),
    ("language",       lambda i: random.choice(["EN", "ES", "FR", "ZH", "AR"])),
    # ── Financial ────────────────────────────────────────────────────────────
    ("account_number", lambda i: f"ACC-{_rnd_digits(8)}"),
    ("account_no",     lambda i: f"ACC-{_rnd_digits(8)}"),
    ("acct_num",       lambda i: f"ACC-{_rnd_digits(8)}"),
    ("card_number",    lambda i: f"****-****-****-{_rnd_digits(4)}"),
    ("routing",        lambda i: _rnd_digits(9)),
    ("balance",        lambda i: round(random.uniform(100, 50000), 2)),
    ("salary",         lambda i: round(random.uniform(30000, 200000), 2)),
    ("income",         lambda i: round(random.uniform(20000, 300000), 2)),
    ("amount",         lambda i: round(random.uniform(10, 5000), 2)),
    ("price",          lambda i: round(random.uniform(1, 500), 2)),
    ("cost",           lambda i: round(random.uniform(1, 1000), 2)),
    ("revenue",        lambda i: round(random.uniform(100, 100000), 2)),
    # ── Medical / pharmacy ───────────────────────────────────────────────────
    ("npi",            lambda i: _rnd_digits(10)),
    ("dea",            lambda i: f"{_rnd_alpha(2)}{_rnd_digits(7)}"),
    ("diagnosis",      lambda i: random.choice(_DIAGNOSIS_CODES)),
    ("icd",            lambda i: random.choice(_DIAGNOSIS_CODES)),
    ("drug_code",      lambda i: random.choice(_DRUG_CODES)),
    ("ndc",            lambda i: f"{_rnd_digits(5)}-{_rnd_digits(4)}-{_rnd_digits(2)}"),
    ("rx_number",      lambda i: f"RX-{_rnd_digits(7)}"),
    ("prescription",   lambda i: f"RX-{_rnd_digits(7)}"),
    ("insurance_id",   lambda i: f"INS-{_rnd_digits(5)}"),
    ("insurance",      lambda i: random.choice(_INSURANCE_CARRIERS)),
    ("policy",         lambda i: f"POL-{_rnd_digits(8)}"),
    ("member_id",      lambda i: f"MBR-{_rnd_digits(7)}"),
    ("group_id",       lambda i: f"GRP-{_rnd_digits(6)}"),
    ("copay",          lambda i: round(random.uniform(0, 100), 2)),
    # ── Generic IDs and codes ────────────────────────────────────────────────
    ("patient_id",     lambda i: 10000 + i * 7 + random.randint(1, 50)),
    ("person_id",      lambda i: 10000 + i * 7 + random.randint(1, 50)),
    ("employee_id",    lambda i: 20000 + i * 3 + random.randint(1, 20)),
    ("customer_id",    lambda i: 30000 + i * 5 + random.randint(1, 30)),
    ("member_id",      lambda i: 40000 + i * 4 + random.randint(1, 25)),
    ("user_id",        lambda i: 50000 + i * 6 + random.randint(1, 40)),
    ("status",         lambda i: random.choice(_STATUS_CODES)),
    ("active",         lambda i: random.choice(_ACTIVE_VALUES)),
    ("enabled",        lambda i: random.choice(_ACTIVE_VALUES)),
    ("flag",           lambda i: random.choice(_ACTIVE_VALUES)),
    ("yn",             lambda i: random.choice(_YN_VALUES)),
]


def _generate_value(col_name: str, col_type: str, row_index: int) -> Any:
    """
    Generate a single synthetic value.
    Tries column-name matching first, then falls back to type-based generation.
    """
    col_lower = col_name.lower()

    # Try each pattern in order — first match wins
    for pattern, generator in _COLUMN_PATTERNS:
        if pattern in col_lower:
            try:
                return generator(row_index)
            except Exception:
                break

    # Fall back to type-based generation
    type_upper = (col_type or "").upper()
    base_id = 10000 + row_index * 13

    if any(t in type_upper for t in ("VARCHAR", "CHAR", "TEXT", "STRING", "NVARCHAR", "CLOB")):
        return f"SAMPLE_{_rnd_alpha(3)}{row_index + 1}"
    elif any(t in type_upper for t in ("NUMBER", "INT", "DECIMAL", "FLOAT", "NUMERIC", "BIGINT", "SMALLINT")):
        return base_id + random.randint(0, 999)
    elif any(t in type_upper for t in ("DATE", "TIMESTAMP", "DATETIME")):
        return _rnd_date(0, 2)
    elif "BOOL" in type_upper:
        return random.choice([True, False])
    else:
        return f"VAL_{row_index + 1}"


def generate_synthetic_sample(
    columns: list[dict],
    n_rows: int = 5,
    seed: str = "",
) -> list[dict]:
    """
    Generate n_rows of synthetic data matching the given column definitions.

    Args:
        columns : list of dicts with at minimum 'name' and 'type' keys
        n_rows  : number of synthetic rows to generate (default 5)
        seed    : optional per-account seed (typically account_id). When set,
                  generation is deterministic *within* an account but differs
                  *across* accounts — so the same fake patient names are never
                  reused across clients (avoids cross-client inference). When
                  empty, generation is fully random per call (legacy behaviour).

    Returns:
        list of dicts — same structure as real DB rows but entirely fake
    """
    _saved_state = None
    if seed:
        # Derive a stable 32-bit seed from the account key; save/restore the
        # global RNG state so we don't perturb randomness elsewhere.
        import hashlib
        _saved_state = random.getstate()
        _derived = int(hashlib.sha256(str(seed).encode("utf-8")).hexdigest()[:8], 16)
        random.seed(_derived)
    try:
        rows = []
        for i in range(n_rows):
            row = {}
            for col in columns:
                col_name = col.get("name") or col.get("COLUMN_NAME") or "col"
                col_type = col.get("type") or col.get("DATA_TYPE") or "VARCHAR"
                row[col_name] = _generate_value(col_name, col_type, i)
            rows.append(row)
        return rows
    finally:
        if _saved_state is not None:
            random.setstate(_saved_state)
