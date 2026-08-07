"""
Date-role detection for role-playing date dimensions.

Enterprise fact tables often contain several foreign keys to the same date
dimension: order date, invoice date, requested delivery date, payment date,
receipt date, and so on.  The physical dimension is the same table, but the
business meaning of each FK is different.  These helpers keep that mapping
deterministic so SQL generation does not guess the wrong date.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class DateRole:
    key: str
    label: str
    synonyms: tuple[str, ...]
    priority: int = 50


DATE_ROLES: tuple[DateRole, ...] = (
    DateRole("booked_date", "Booked Date", ("booked date", "booking date", "booked month", "booked year"), 97),
    DateRole("invoice_date", "Invoice Date", ("invoice date", "invoiced date", "billing date", "billed date", "invoice month", "invoice year"), 95),
    DateRole("order_date", "Order Date", ("order date", "ordered date", "sales order date", "order month", "order year"), 92),
    DateRole("cancelled_order_date", "Cancelled Order Date", ("cancelled order date", "canceled order date", "order cancellation date", "cancelled order month"), 90),
    DateRole("requested_delivery_date", "Requested Delivery Date", ("requested delivery date", "requested ship date", "requested delivery month", "due delivery date"), 88),
    DateRole("confirmed_delivery_date", "Confirmed Delivery Date", ("confirmed delivery date", "confirmed ship date", "confirmed delivery month"), 87),
    DateRole("planned_delivery_date", "Planned Delivery Date", ("planned delivery date", "planned ship date", "planned delivery month"), 86),
    DateRole("valid_delivery_date", "Valid Delivery Date", ("valid delivery date", "validated delivery date", "valid ship date", "valid delivery month"), 84),
    DateRole("delivery_date", "Delivery Date", ("delivery date", "ship date", "shipped date", "fulfillment date", "delivery month", "delivery year"), 82),
    DateRole("due_date", "Due Date", ("due date", "payment due date", "invoice due date", "due month"), 78),
    DateRole("payment_date", "Payment Date", ("payment date", "paid date", "collection date", "cash date", "payment month"), 76),
    DateRole("receipt_date", "Receipt Date", ("receipt date", "received date", "purchase receipt date", "receipt month"), 75),
    DateRole("accounting_date", "Accounting Date", ("accounting date", "posting date", "gl date", "ledger date", "accounting month"), 74),
    DateRole("current_cost_date", "Current Cost Date", ("current cost date", "current replacement cost date", "current cost month"), 73),
    DateRole("previous_cost_date", "Previous Cost Date", ("previous cost date", "previous replacement cost date", "prior cost date", "previous cost month"), 72),
    DateRole("order_line_creation_date", "Order Line Creation Date", ("order line creation date", "purchase order line creation date", "line creation date", "line created date"), 70),
    DateRole("creation_date", "Creation Date", ("creation date", "created date", "line creation date", "created month"), 68),
    DateRole("registration_date", "Registration Date", ("registration date", "created date", "entry date", "created month"), 65),
    DateRole("modified_date", "Last Modified Date", ("last modified date", "updated date", "last updated date"), 55),
)

_ROLE_BY_KEY = {role.key: role for role in DATE_ROLES}

_GENERIC_DATE_KEY_RE = re.compile(
    r"(?:^|_)(?:DATE|DT)(?:_(?:DMS_)?(?:KEY|ID))$"
)
_PLAIN_DATE_KEY_RE = re.compile(
    r"(?:^|_)DATE(?:_(?:DMS_)?(?:KEY|ID))$"
)

_COLUMN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:^|_)BOOK(?:ED|ING)?_DT(?:_|$)|(?:^|_)BKD_DT(?:_|$)"), "booked_date"),
    (re.compile(r"(?:^|_)CUS_IVC_DT(?:_|$)|(?:^|_)SLR_IVC_DT(?:_|$)|(?:^|_)IVC_DT(?:_|$)|^IVDT$"), "invoice_date"),
    (re.compile(r"(?:^|_)CCL_.*ORD_DT(?:_|$)|(?:^|_)CANCEL(?:LED|ED)?_.*ORD_DT(?:_|$)"), "cancelled_order_date"),
    (re.compile(r"(?:^|_)CUS_(?:ORD|ORDER)_DT(?:_|$)|(?:^|_)PCH_(?:ORD|ORDER)_DT(?:_|$)|(?:^|_)(?:ORD|ORDER)_DT(?:_|$)|^ORDT$"), "order_date"),
    (re.compile(r"(?:^|_)RQD_.*DLV_DT(?:_|$)|(?:^|_)REQ(?:UESTED)?_.*DLV_DT(?:_|$)|^DWDT$"), "requested_delivery_date"),
    (re.compile(r"(?:^|_)CFM_.*DLV_DT(?:_|$)|(?:^|_)CONF(?:IRMED)?_.*DLV_DT(?:_|$)|^CODT$"), "confirmed_delivery_date"),
    (re.compile(r"(?:^|_)PLD_.*DLV_DT(?:_|$)|(?:^|_)PLANN?ED_.*DLV_DT(?:_|$)|^PLDT$"), "planned_delivery_date"),
    (re.compile(r"(?:^|_)VLD_.*DLV_DT(?:_|$)|(?:^|_)VALID_.*DLV_DT(?:_|$)"), "valid_delivery_date"),
    (re.compile(r"(?:^|_)DLV_DT(?:_|$)|(?:^|_)SHIP_DT(?:_|$)|^DLDT$|^DSDT$"), "delivery_date"),
    (re.compile(r"(?:^|_)DUE_DT(?:_|$)|^DUDT$"), "due_date"),
    (re.compile(r"(?:^|_)PAY(?:MENT)?_DT(?:_|$)|(?:^|_)PYM?T_DT(?:_|$)"), "payment_date"),
    (re.compile(r"(?:^|_)RCT_DT(?:_|$)|(?:^|_)RECEIPT_DT(?:_|$)|(?:^|_)RCV_DT(?:_|$)|^RVDT$"), "receipt_date"),
    (re.compile(r"(?:^|_)ACCT?_DT(?:_|$)|(?:^|_)ACCOUNTING_DT(?:_|$)|^ACDT$"), "accounting_date"),
    (re.compile(r"(?:^|_)CUR_.*CST_DT(?:_|$)|(?:^|_)CURRENT_.*COST_DT(?:_|$)"), "current_cost_date"),
    (re.compile(r"(?:^|_)PRE_.*CST_DT(?:_|$)|(?:^|_)PREV(?:IOUS)?_.*COST_DT(?:_|$)|(?:^|_)PRIOR_.*COST_DT(?:_|$)"), "previous_cost_date"),
    (re.compile(r"(?:^|_)PCH_ORD_LIN_CRN_DT(?:_|$)|(?:^|_)ORD_LIN_CRN_DT(?:_|$)|(?:^|_)LINE_CRN_DT(?:_|$)|(?:^|_)LINE_CREATED?_DT(?:_|$)"), "order_line_creation_date"),
    (re.compile(r"(?:^|_)CRN_DT(?:_|$)|(?:^|_)CREATION_DT(?:_|$)|(?:^|_)CREATED_DT(?:_|$)"), "creation_date"),
    (re.compile(r"(?:^|_)RGDT(?:_|$)|(?:^|_)REG(?:ISTRATION)?_DT(?:_|$)|(?:^|_)CRN_DT(?:_|$)|^RGDT$"), "registration_date"),
    (re.compile(r"(?:^|_)LMDT(?:_|$)|(?:^|_)LST_UPD(?:_|$)|(?:^|_)UPDATED?_DT(?:_|$)|^LMDT$"), "modified_date"),
)

DATE_DIMENSION_TABLE_HINTS = (
    "DATE", "DIM_DATE", "DATE_DIM", "D_DATE", "CALENDAR", "DIM_CALENDAR",
    "DATE_DMS", "PRD_DMS", "PERIOD", "DIM_PERIOD",
)

DATE_DIMENSION_KEY_HINTS = (
    "DATE_DMS_KEY", "DT_DMS_KEY", "PRD_DMS_KEY", "DATE_KEY", "DATE_ID",
    "CALENDAR_DATE_KEY", "CALENDAR_KEY", "DAY_KEY", "PERIOD_KEY",
)

DATE_KEY_TYPES = (
    "surrogate_fk",
    "yyyymmdd_integer",
    "yyyymm_integer",
    "native_date",
    "timestamp",
)

_INTEGER_TYPE_RE = re.compile(
    r"^(?:tinyint|smallint|int|integer|bigint|number|numeric|decimal)(?:\s*\(|\b)",
    re.IGNORECASE,
)
_YYYYMMDD_NAME_RE = re.compile(
    r"(?:^|_)(?:YYYYMMDD|DATE_YYYYMMDD|CALENDAR_DAY_KEY)(?:_|$)|"
    r"(?:^|_)(?:DT_DMS_KEY|DATE_DMS_KEY)$",
    re.IGNORECASE,
)
_YYYYMM_NAME_RE = re.compile(
    r"(?:^|_)(?:YYYYMM|YEAR_MONTH|YR_MONTH|YR_MTH|YEAR_MTH)(?:_|$)|"
    r"(?:^|_)(?:PRD_DMS_KEY|PERIOD_DMS_KEY)$",
    re.IGNORECASE,
)


def infer_encoded_date_key(column_name: str, data_type: str = "") -> dict:
    """Infer a self-contained integer calendar encoding from strong metadata.

    This intentionally does *not* classify arbitrary integer IDs as dates.  A
    direct fallback is emitted only for naming conventions that declare the
    physical representation (YYYYMMDD/day or YYYYMM/month).  Sequential
    ``*_DATE_ID``/``*_DATE_KEY`` values remain governed surrogate keys and
    still require a real date-dimension relationship or an admin mapping.
    """
    raw_type = " ".join(str(data_type or "").strip().lower().split())
    if not _INTEGER_TYPE_RE.match(raw_type):
        return {}
    column = str(column_name or "").strip().strip('"`[]')
    if _YYYYMMDD_NAME_RE.search(column):
        return {
            "date_key_type": "yyyymmdd_integer",
            "temporal_grain": "day",
            "confidence": 98,
            "inference_source": "encoded_date_name",
        }
    if _YYYYMM_NAME_RE.search(column):
        return {
            "date_key_type": "yyyymm_integer",
            "temporal_grain": "month",
            "confidence": 96,
            "inference_source": "encoded_period_name",
        }
    return {}


def physical_date_key_type(data_type: str) -> str:
    """Return the governed type for a physical calendar column, if any.

    SQL Server's bare ``timestamp`` type is a rowversion binary value, not a
    date, so it is deliberately excluded. PostgreSQL timestamp spellings and
    the SQL Server datetime family are treated as temporal values.
    """
    raw = " ".join(str(data_type or "").strip().lower().split())
    if not raw:
        return ""
    base = re.split(r"[\s(]", raw, maxsplit=1)[0]
    if base == "date":
        return "native_date"
    if base in {"datetime", "datetime2", "smalldatetime", "datetimeoffset", "timestamptz"}:
        return "timestamp"
    if raw in {"timestamp with time zone", "timestamp without time zone"}:
        return "timestamp"
    return ""


def normalize_date_key_type(value: str, *, has_date_dimension_fk: bool = False) -> str:
    """Return one governed date-key type.

    A relationship to a date dimension always wins over name-based inference:
    integer IDs such as 4067 are surrogate keys, not YYYYMMDD values.
    """
    if has_date_dimension_fk:
        return "surrogate_fk"
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "surrogate": "surrogate_fk",
        "foreign_key": "surrogate_fk",
        "fk": "surrogate_fk",
        "yyyymmdd": "yyyymmdd_integer",
        "integer_date": "yyyymmdd_integer",
        "yyyymm": "yyyymm_integer",
        "integer_month": "yyyymm_integer",
        "year_month": "yyyymm_integer",
        "date": "native_date",
        "datetime": "timestamp",
        "datetime2": "timestamp",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in DATE_KEY_TYPES else "surrogate_fk"


def classify_date_key(
    column_name: str,
    data_type: str = "",
    *,
    has_date_dimension_fk: bool = False,
    declared_encoding: str = "",
) -> str:
    """Classify how a date-related physical column must be interpreted."""
    if has_date_dimension_fk:
        return "surrogate_fk"
    if declared_encoding:
        return normalize_date_key_type(declared_encoding)
    physical_type = physical_date_key_type(data_type)
    if physical_type:
        return physical_type
    inferred = infer_encoded_date_key(column_name, data_type)
    if inferred:
        return str(inferred["date_key_type"])
    # Integer date keys are deliberately not guessed from a *_DT_* name.
    # YYYYMMDD must be declared/profiled; otherwise treating 4067 as a date
    # causes invalid conversions and, worse, plausible but incorrect periods.
    return "yyyymmdd_integer" if normalize_date_key_type(declared_encoding) == "yyyymmdd_integer" else "surrogate_fk"


def date_key_temporal_grain(date_key_type: str) -> str:
    """Return the finest calendar grain supported by a governed key type."""
    key_type = normalize_date_key_type(date_key_type)
    if key_type == "yyyymm_integer":
        return "month"
    if key_type in {"yyyymmdd_integer", "native_date", "timestamp"}:
        return "day"
    return ""


def normalize_date_role_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def question_has_temporal_intent(question: str) -> bool:
    """Return True when a question explicitly asks for a time/date concept."""
    q = normalize_date_role_text(question)
    if not q:
        return False

    temporal_terms = {
        "date", "dates", "day", "days", "daily", "week", "weeks", "weekly",
        "month", "months", "monthly", "quarter", "quarters", "quarterly",
        "year", "years", "yearly", "period", "periods", "time", "timeline",
        "trend", "trends", "when", "latest", "earliest", "recent", "previous",
        "prior", "current", "yesterday", "today", "tomorrow", "ytd", "mtd",
        "qtd", "yoy", "mom", "wow",
    }
    q_terms = set(q.split())
    if q_terms & temporal_terms:
        return True

    temporal_phrases = (
        "as of",
        "over time",
        "year over year",
        "month over month",
        "week over week",
        "period over period",
        "last year",
        "last month",
        "last week",
        "this year",
        "this month",
        "this week",
    )
    return any(phrase in q for phrase in temporal_phrases)


def detect_date_role(column_name: str, vocab=None) -> DateRole | None:
    col = (column_name or "").strip().strip('"`[]').upper()
    if not col:
        return None
    # Terminology-pack patterns run FIRST so a pack can specialize a column
    # the builtin regexes would miss (e.g. SAP AUDAT → order_date). The 18
    # builtin role keys stay fixed; packs only map new column names to them.
    if vocab is None:
        from core.vocab_packs import get_active_vocab
        vocab = get_active_vocab()
    # Plain warehouse names often spell DATE where ERP models use DT. The
    # shared canonicalizer also exposes camel/Pascal/compact forms, so
    # OrderDate, ORDERDATE, ORD_DT and ORDDT can resolve through one rule set.
    from core.identifier_intelligence import canonical_identifier
    segmented = canonical_identifier(column_name, vocab=vocab) or col
    canonical = re.sub(r"(^|_)DATE(?=_|$)", r"\1DT", segmented)
    candidates = tuple(dict.fromkeys((col, segmented, canonical)))
    for pattern, role_key in getattr(vocab, "date_role_patterns", ()):
        if any(pattern.search(candidate) for candidate in candidates) and role_key in _ROLE_BY_KEY:
            return _ROLE_BY_KEY[role_key]
    # Preserve modifiers in descriptive warehouse names. For example,
    # LAST_RECEIPT_DATE_ID is a distinct role from RECEIPT_DATE_ID.
    if any(_PLAIN_DATE_KEY_RE.search(candidate) for candidate in candidates):
        return derive_date_role(column_name)
    for pattern, role_key in _COLUMN_PATTERNS:
        if any(pattern.search(candidate) for candidate in candidates):
            return _ROLE_BY_KEY[role_key]
    if any(_GENERIC_DATE_KEY_RE.search(candidate) for candidate in candidates):
        return derive_date_role(column_name)
    return None


def derive_date_role(column_name: str) -> DateRole:
    """Derive a reviewable business role from any date-dimension FK column.

    This is intentionally deterministic. A physical FK to a recognized date
    dimension is stronger evidence than a naming dictionary, so unfamiliar
    names such as DISPENSE_DATE_ID and THERAPY_START_DATE_KEY must survive into
    the semantic model instead of being silently dropped.
    """
    label = _label_from_column(column_name)
    key = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "business_date"
    return DateRole(key, label, (label.lower(),), 60)


def is_date_role_column(column_name: str) -> bool:
    return detect_date_role(column_name) is not None


_DMS_KEY_SUFFIX_RE = re.compile(
    r"(?:_DT_DMS_KEY|_DATE_DMS_KEY|_PRD_DMS_KEY|_PERIOD_DMS_KEY)$",
    re.IGNORECASE,
)
_SURROGATE_KEY_SUFFIX_RE = re.compile(r"(?:_ID|_KEY)$", re.IGNORECASE)


def is_plain_surrogate_date_role_column(column_name: str) -> bool:
    """True for a date-role FK column that is a pure sequential surrogate
    key (e.g. DISPENSE_DATE_ID) — not the _DT_DMS_KEY/_DATE_DMS_KEY
    YYYYMMDD-encoded convention (which has its own arithmetic-decode rule
    elsewhere), and not a plain native DATE/DATETIME column such as
    ORDER_DATE (no _ID/_KEY suffix at all — YEAR()/MONTH() is perfectly
    valid directly on those; is_date_role_column() alone matches both
    shapes since it only checks business-role naming, not this distinction).
    """
    col = (column_name or "").strip()
    if not col or _DMS_KEY_SUFFIX_RE.search(col):
        return False
    if not _SURROGATE_KEY_SUFFIX_RE.search(col):
        return False
    return is_date_role_column(col)


def date_role_terms(role: DateRole) -> list[str]:
    terms = [role.label.lower(), role.key.replace("_", " ")]
    terms.extend(role.synonyms)
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        norm = normalize_date_role_text(term)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


_EVENT_DATE_VARIANTS: dict[str, tuple[str, ...]] = {
    "booked_date": ("booking date", "when booked"),
    "invoice_date": ("invoiced date", "billed date", "when invoiced"),
    "order_date": ("ordered date", "date ordered", "when ordered"),
    "cancelled_order_date": ("canceled order date", "when cancelled"),
    "requested_delivery_date": ("requested ship date", "delivery requested date"),
    "confirmed_delivery_date": ("confirmed ship date", "delivery confirmed date"),
    "planned_delivery_date": ("planned ship date",),
    "delivery_date": ("delivered date", "shipped date", "when delivered", "when shipped"),
    "due_date": ("payment due date", "when due"),
    "payment_date": ("paid date", "when paid"),
    "receipt_date": ("received date", "when received"),
    "accounting_date": ("posting date", "gl date", "ledger date"),
    "creation_date": ("created date", "date created", "when created"),
    "registration_date": ("registered date", "when registered"),
    "modified_date": ("updated date", "last updated date", "when updated"),
}


def generated_date_role_synonyms(role: DateRole, column_name: str = "", vocab=None) -> tuple[str, ...]:
    """Return governed business and physical-language variants for a role."""
    values: list[str] = [*date_role_terms(role)]
    if column_name:
        from core.identifier_intelligence import analyze_identifier
        values.extend(analyze_identifier(column_name, vocab=vocab).aliases)
    values.extend(_EVENT_DATE_VARIANTS.get(role.key, ()))
    stem = normalize_date_role_text(role.label)
    if stem.endswith(" date"):
        stem = stem[:-5].strip()
    if stem:
        values.extend(f"{stem} {grain}" for grain in ("day", "week", "month", "quarter", "year"))
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = normalize_date_role_text(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return tuple(result)


def relationship_matches_date_role(question: str, label: str = "", description: str = "") -> bool:
    q = normalize_date_role_text(question)
    if not q:
        return False
    candidates = [label, description]
    label_norm = normalize_date_role_text(label or description)
    for role in DATE_ROLES:
        if label_norm in {normalize_date_role_text(role.label), role.key.replace("_", " ")}:
            candidates.extend(date_role_terms(role))
    for candidate in candidates:
        norm = normalize_date_role_text(candidate)
        if norm and norm in q:
            return True
    return False


def is_date_dimension_table(table_name: str, columns: list[dict] | list[str]) -> bool:
    bare = (table_name or "").split(".")[-1].upper()
    compact = re.sub(r"[^A-Z0-9]+", "_", bare)
    if compact in DATE_DIMENSION_TABLE_HINTS or any(h in compact for h in ("DIM_DATE", "DATE_DIM", "CALENDAR")):
        return True
    col_names = {_column_name(c).upper() for c in columns}
    # Structural inference requires both a canonical date key and a separate
    # business date value. Merely containing ORDER_DATE_ID must not cause a
    # fact table to be mistaken for the date dimension itself.
    return bool(col_names & set(DATE_DIMENSION_KEY_HINTS)) and bool(
        find_date_value_column(columns)
    )


def find_date_dimension_key(columns: list[dict] | list[str]) -> str:
    col_names = [_column_name(c) for c in columns]
    upper_to_actual = {c.upper(): c for c in col_names}
    for hint in DATE_DIMENSION_KEY_HINTS:
        if hint in upper_to_actual:
            return upper_to_actual[hint]
    for col in col_names:
        up = col.upper()
        if ("DATE" in up or up.startswith("DT_") or up.startswith("PRD_")) and (up.endswith("_KEY") or up.endswith("_ID")):
            return col
    return col_names[0] if col_names else ""


def find_date_value_column(columns: list[dict] | list[str]) -> str:
    """Return the business date value column, never the surrogate key."""
    candidates: list[tuple[int, str]] = []
    for raw in columns:
        name = _column_name(raw)
        if not name:
            continue
        upper = name.upper()
        data_type = ""
        if isinstance(raw, dict):
            data_type = str(
                raw.get("type") or raw.get("DATA_TYPE") or raw.get("data_type") or ""
            ).lower()
        if upper in DATE_DIMENSION_KEY_HINTS or upper.endswith(("_KEY", "_ID")):
            continue
        score = 0
        if data_type in {"date", "datetime", "datetime2", "timestamp", "timestamp_ntz", "timestamp_tz"}:
            score += 100
        if upper in {"DMS_DT", "DT", "DATE", "CALENDAR_DATE", "FULL_DATE", "DATE_VALUE"}:
            score += 90
        elif upper in {"DT_DSC", "DATE_DSC", "DATE_DESC", "DATE_DESCRIPTION"}:
            score += 70
        elif "DATE" in upper or upper.endswith("_DT"):
            score += 55
        if score:
            candidates.append((score, name))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def _column_name(col: dict | str) -> str:
    if isinstance(col, dict):
        return str(col.get("name") or col.get("COLUMN_NAME") or "")
    return str(col)


# Physical plumbing that must never reach a business-date label shown to an
# end user: surrogate/foreign key markers and packed-integer encodings.
# A live regression surfaced "Invoice Sk Date" and "Posting Yyyymm Date" as
# clarification buttons — INVOICE_DATE_KEY already rendered correctly as
# "Invoice Date", so only the _SK spelling (and the YYYYMM/YYYYMMDD encodings)
# were leaking storage detail into the business vocabulary.
_KEY_TOKENS = {"key", "id", "sk", "fk", "dimension", "dim", "utc"}
# Packed-integer date encodings. These are never a business word in any
# position, so they are safe to drop wherever they appear.
_ENCODING_TOKENS = {"yyyymmdd", "yyyymm", "yymmdd", "yymm"}
_MONTH_ENCODINGS = {"yyyymm", "yymm"}
# Unambiguous plumbing prefixes (DIM_ORDER_DATE, DMS_SVC_DT).
_PLUMBING_PREFIXES = {"dim", "dms"}


def _strip_plumbing_tokens(tokens: list[str]) -> list[str]:
    """Drop storage plumbing while protecting real business words.

    Surrogate-key markers are a *positional* convention — they suffix a name
    (INVOICE_DATE_SK, ADMIT_DATE_KEY). Removing them wherever they appear
    would corrupt legitimate vocabulary: KEY_ACCOUNT_DATE is a "Key Account"
    date, not an "Account" date. So key tokens are stripped only from the
    tail (and the well-known DIM_/DMS_ prefixes from the head), while packed
    date encodings are stripped anywhere.
    """
    result = [tok for tok in tokens if tok.lower() not in _ENCODING_TOKENS]
    while result and result[0].lower() in _PLUMBING_PREFIXES:
        result.pop(0)
    while result and result[-1].lower() in _KEY_TOKENS:
        result.pop()
    return result


def _label_from_column(column: str) -> str:
    original = (column or "").strip().strip('"`[]')
    text = original.upper()
    text = re.sub(
        r"_(?:DATE|DT)(?:_DMS)?_(?:KEY|ID|SK|FK)$",
        "",
        text,
    )
    text = re.sub(r"_?DMS_KEY$", "", text)
    text = re.sub(r"_(?:KEY|ID|SK|FK)$", "", text)
    text = re.sub(r"_?(?:DATE|DT)$", "", text)
    from core.identifier_intelligence import analyze_identifier
    analysis = analyze_identifier(original)
    raw_parts = analysis.expanded_name.split()
    # A YYYYMM-encoded column is a monthly *period*, not a calendar date —
    # naming it "... Date" both leaks the encoding and misdescribes the grain,
    # which left the monthly inventory role unreachable when a user searched
    # for its real business name ("Inventory Period").
    is_month_grain = any(part in _MONTH_ENCODINGS for part in raw_parts)
    semantic_parts = _strip_plumbing_tokens([
        part for part in raw_parts
        if part not in {"date", "datetime", "timestamp"}
    ])
    if semantic_parts:
        label = " ".join(semantic_parts).title().strip()
        suffix = "Period" if is_month_grain else "Date"
        if suffix not in label and (suffix != "Date" or "Period" not in label):
            label = (label + " " + suffix).strip()
        return label or "Business Date"
    parts = _strip_plumbing_tokens([
        p for p in text.split("_") if p and p not in {"CUS", "PCH", "SLR"}
    ])
    expanded = []
    mini = {
        "ORD": "Order", "IVC": "Invoice", "DLV": "Delivery", "RQD": "Requested",
        "CFM": "Confirmed", "PLD": "Planned", "RCT": "Receipt", "DUE": "Due",
        "ACD": "Accounting", "CRN": "Created", "CUR": "Current", "CST": "Cost",
        "PRE": "Previous", "CCL": "Cancelled", "VLD": "Valid", "LIN": "Line",
        "PCH": "Purchase", "PO": "Purchase Order",
    }
    for part in parts:
        expanded.append(mini.get(part, part.capitalize()))
    label = " ".join(expanded).strip()
    suffix = "Period" if is_month_grain else "Date"
    if suffix not in label and (suffix != "Date" or "Period" not in label):
        label = (label + " " + suffix).strip()
    return label or "Business Date"
