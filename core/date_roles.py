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
    DateRole("invoice_date", "Invoice Date", ("invoice date", "invoiced date", "billing date", "billed date", "invoice month", "invoice year"), 95),
    DateRole("order_date", "Order Date", ("order date", "ordered date", "sales order date", "order month", "order year"), 92),
    DateRole("requested_delivery_date", "Requested Delivery Date", ("requested delivery date", "requested ship date", "requested delivery month", "due delivery date"), 88),
    DateRole("confirmed_delivery_date", "Confirmed Delivery Date", ("confirmed delivery date", "confirmed ship date", "confirmed delivery month"), 87),
    DateRole("planned_delivery_date", "Planned Delivery Date", ("planned delivery date", "planned ship date", "planned delivery month"), 86),
    DateRole("delivery_date", "Delivery Date", ("delivery date", "ship date", "shipped date", "fulfillment date", "delivery month", "delivery year"), 82),
    DateRole("due_date", "Due Date", ("due date", "payment due date", "invoice due date", "due month"), 78),
    DateRole("payment_date", "Payment Date", ("payment date", "paid date", "collection date", "cash date", "payment month"), 76),
    DateRole("receipt_date", "Receipt Date", ("receipt date", "received date", "purchase receipt date", "receipt month"), 75),
    DateRole("accounting_date", "Accounting Date", ("accounting date", "posting date", "gl date", "ledger date", "accounting month"), 74),
    DateRole("registration_date", "Registration Date", ("registration date", "created date", "entry date", "created month"), 65),
    DateRole("modified_date", "Last Modified Date", ("last modified date", "updated date", "last updated date"), 55),
)

_ROLE_BY_KEY = {role.key: role for role in DATE_ROLES}

_COLUMN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:^|_)CUS_IVC_DT(?:_|$)|(?:^|_)SLR_IVC_DT(?:_|$)|(?:^|_)IVC_DT(?:_|$)|^IVDT$"), "invoice_date"),
    (re.compile(r"(?:^|_)CUS_ORD_DT(?:_|$)|(?:^|_)PCH_ORD_DT(?:_|$)|(?:^|_)ORD_DT(?:_|$)|^ORDT$"), "order_date"),
    (re.compile(r"(?:^|_)RQD_.*DLV_DT(?:_|$)|(?:^|_)REQ(?:UESTED)?_.*DLV_DT(?:_|$)|^DWDT$"), "requested_delivery_date"),
    (re.compile(r"(?:^|_)CFM_.*DLV_DT(?:_|$)|(?:^|_)CONF(?:IRMED)?_.*DLV_DT(?:_|$)|^CODT$"), "confirmed_delivery_date"),
    (re.compile(r"(?:^|_)PLD_.*DLV_DT(?:_|$)|(?:^|_)PLANN?ED_.*DLV_DT(?:_|$)|^PLDT$"), "planned_delivery_date"),
    (re.compile(r"(?:^|_)DLV_DT(?:_|$)|(?:^|_)SHIP_DT(?:_|$)|^DLDT$|^DSDT$"), "delivery_date"),
    (re.compile(r"(?:^|_)DUE_DT(?:_|$)|^DUDT$"), "due_date"),
    (re.compile(r"(?:^|_)PAY(?:MENT)?_DT(?:_|$)|(?:^|_)PYM?T_DT(?:_|$)"), "payment_date"),
    (re.compile(r"(?:^|_)RCT_DT(?:_|$)|(?:^|_)RECEIPT_DT(?:_|$)|(?:^|_)RCV_DT(?:_|$)|^RVDT$"), "receipt_date"),
    (re.compile(r"(?:^|_)ACCT?_DT(?:_|$)|(?:^|_)ACCOUNTING_DT(?:_|$)|^ACDT$"), "accounting_date"),
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


def normalize_date_role_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def detect_date_role(column_name: str) -> DateRole | None:
    col = (column_name or "").strip().strip('"`[]').upper()
    if not col:
        return None
    for pattern, role_key in _COLUMN_PATTERNS:
        if pattern.search(col):
            return _ROLE_BY_KEY[role_key]
    if col.endswith("_DT_DMS_KEY") or col.endswith("_DATE_DMS_KEY"):
        return DateRole(
            "business_date",
            _label_from_column(col),
            (_label_from_column(col).lower(),),
            45,
        )
    return None


def is_date_role_column(column_name: str) -> bool:
    return detect_date_role(column_name) is not None


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
    if col_names & set(DATE_DIMENSION_KEY_HINTS):
        return True
    has_date_text = any("DATE" in c or c in {"YEAR", "MONTH", "DAY"} for c in col_names)
    return has_date_text and any(c.endswith("_KEY") or c.endswith("_ID") for c in col_names)


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


def _column_name(col: dict | str) -> str:
    if isinstance(col, dict):
        return str(col.get("name") or col.get("COLUMN_NAME") or "")
    return str(col)


def _label_from_column(column: str) -> str:
    text = column
    text = re.sub(r"_?DMS_KEY$", "", text)
    text = re.sub(r"_?DATE_KEY$", "", text)
    text = re.sub(r"_?DT$", "", text)
    parts = [p for p in text.split("_") if p and p not in {"CUS", "PCH", "SLR"}]
    expanded = []
    mini = {
        "ORD": "Order", "IVC": "Invoice", "DLV": "Delivery", "RQD": "Requested",
        "CFM": "Confirmed", "PLD": "Planned", "RCT": "Receipt", "DUE": "Due",
        "ACD": "Accounting", "CRN": "Created",
    }
    for part in parts:
        expanded.append(mini.get(part, part.capitalize()))
    label = " ".join(expanded).strip()
    if "Date" not in label:
        label = (label + " Date").strip()
    return label or "Business Date"
