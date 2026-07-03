"""
Deterministic schema enrichment for KB generation.

Production ERP/data-warehouse tables often expose cryptic names such as ORNO,
PONR, CUS_IVC_LIN_AMT, and ITM_GRP_DMS_KEY.  This module turns those raw names
into structured, evidence-backed hints before the LLM writes the KB document.
The output is intentionally conservative: it proposes meanings and roles, but
keeps official metric approval in the semantic/metric layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from core.date_roles import date_role_terms, detect_date_role
from core.erp_column_dict import ERP_COLUMN_DICT


ABBREVIATIONS: dict[str, str] = {
    "ABC": "abc",
    "ACC": "account",
    "ACCT": "account",
    "ACT": "active",
    "AGM": "agreement",
    "ALC": "allocated",
    "ALT": "alternate",
    "AMT": "amount",
    "ANN": "annual",
    "AP": "accounts payable",
    "AR": "accounts receivable",
    "APV": "approved",
    "ARA": "area",
    "AVG": "average",
    "BAL": "balance",
    "BCK": "back",
    "BIL": "billing",
    "BILL": "billing",
    "BUM": "base unit of measure",
    "BUS": "business",
    "BYR": "buyer",
    "CAD": "cad",
    "CAT": "category",
    "CCL": "cancelled",
    "CCY": "currency",
    "CD": "code",
    "CFM": "confirmed",
    "CHG": "change",
    "CLS": "class",
    "CLU": "cluster",
    "CO": "company",
    "CNT": "count",
    "CRN": "creation",
    "CST": "cost",
    "CTC": "contact",
    "CUR": "current",
    "CUS": "customer",
    "DCN": "discount",
    "DEL": "deleted",
    "DLV": "delivery",
    "DLVD": "delivered",
    "DMD": "demand",
    "DMS": "dimension",
    "DRC": "direct",
    "DSC": "description",
    "DESC": "description",
    "DT": "date",
    "DVN": "division",
    "EMCO": "company",
    "EQP": "equipment",
    "FCY": "facility",
    "FCT": "fact",
    "FIFO": "first in first out",
    "FRT": "freight",
    "GL": "general ledger",
    "GRP": "group",
    "GRS": "gross",
    "HST": "highest",
    "IND": "indicator",
    "ISP": "inspection",
    "ITM": "item",
    "IVC": "invoice",
    "IVCD": "invoiced",
    "LIN": "line",
    "LIS": "list",
    "LWS": "lowest",
    "MDA": "media",
    "MDL": "model",
    "MFG": "manufacturing",
    "MGP": "margin profit",
    "MNL": "manual",
    "MSR": "measure",
    "NED": "needed",
    "NGV": "negative",
    "NM": "name",
    "NUM": "number",
    "ON": "on",
    "ORD": "order",
    "ORI": "origin",
    "PAY": "payment",
    "PCE": "price",
    "PC": "profit center",
    "PCH": "purchase",
    "PDC": "product",
    "PFT": "profit",
    "PHY": "physical",
    "PIK": "pick",
    "PLD": "planned",
    "PLU": "plus",
    "PRD": "period",
    "PRL": "profile",
    "PRM": "promotion",
    "PRS": "person",
    "PRU": "product",
    "PSV": "positive",
    "PT": "point",
    "PTY": "party",
    "PYE": "payee",
    "PYR": "payer",
    "QTY": "quantity",
    "RCT": "receipt",
    "REC": "record",
    "RET": "return",
    "RJC": "rejected",
    "RGN": "region",
    "RNL": "rental",
    "RPL": "replacement",
    "RPS": "responsible",
    "RQD": "requested",
    "RQS": "requesting",
    "RSV": "reserved",
    "RVD": "received",
    "SAL": "sales",
    "SEG": "segment",
    "SFX": "suffix",
    "SHP": "shipment",
    "SLD": "sold",
    "SLR": "seller",
    "SLY": "supply",
    "SOP": "sales order processing",
    "SRC": "source",
    "STK": "stock",
    "STL": "standard",
    "STS": "status",
    "TOT": "total",
    "TFR": "transfer",
    "TM": "time",
    "TS": "timestamp",
    "TYP": "type",
    "UNT": "unit",
    "UOM": "unit of measure",
    "USR": "user",
    "USD": "usd",
    "VLD": "valid",
    "VOL": "volume",
    "WHS": "warehouse",
    "WRG": "wrong",
    "XCH": "exchange",
    "XTR": "extra",
}

RAW_DATE_CODES = {
    "IVDT", "ORDT", "DLDT", "DWDT", "CODT", "ACDT", "DSDT", "PLDT",
    "RGDT", "LMDT", "DUDT", "RVDT", "LRDT", "BLDT", "ADAT",
}
RAW_IDENTIFIER_CODES = {
    "ORNO", "PONR", "POSX", "DLIX", "IVNO", "CUNO", "SUNO", "PYNO",
    "SMCD", "WHLO", "FACI", "DIVI", "CONO", "JRNO", "JSNO", "VONO",
    "CINO", "AGNO", "PROJ", "ROUT", "BKID",
}
RAW_MEASURE_CODES = {
    "TRQT", "ORQT", "IVQT", "DLQT", "ALQT", "RNQT", "PLQT",
    # alternate-unit quantity variants (QA suffix = qty in alt unit)
    "ORQA", "IVQA", "DLQA", "ALQA", "PLQA", "RNQA", "IVQS", "ORQS", "ORQB",
    "CUAM", "SAAM", "SGAM", "UCOS", "DCOS", "MFAM", "SAPR", "NEPR",
    "PCLA", "OFRA",
    # weight / volume
    "GRWE", "NEWE", "VOL3",
    # AR/financial measures
    "DCAM", "VTAM", "IIAM", "ACBL", "RMBL",
}

# ── Numbered-series patterns ────────────────────────────────────────────────
# Maps a regex (matching the column name) to (human_label_template, role)
# Use {} as placeholder for the digit suffix.
_NUMBERED_SERIES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"^DIC(\d+)$"),  "discount code {}",                    "attribute"),
    (re.compile(r"^DIP(\d+)$"),  "discount percent {}",                 "measure"),
    (re.compile(r"^DIA(\d+)$"),  "discount amount {}",                  "measure"),
    (re.compile(r"^CMP(\d+)$"),  "campaign component {}",               "attribute"),
    (re.compile(r"^ATV(\d+|0)$"),"attribute value {}",                  "attribute"),
    (re.compile(r"^AAV(\d+)$"),  "additional attribute value {}",        "attribute"),
    (re.compile(r"^UCA(\d+|0)$"),"user defined character attribute {}", "attribute"),
    (re.compile(r"^UDN(\d+)$"),  "user defined numeric {}",             "measure"),
    (re.compile(r"^UID(\d+)$"),  "user defined id {}",                  "attribute"),
    (re.compile(r"^UCT(\d+)$"),  "unit conversion type {}",             "attribute"),
    (re.compile(r"^CFE(\d+)$"),  "customer free field {}",              "attribute"),
    (re.compile(r"^CHL(\d+)$"),  "sales channel level {}",              "attribute"),
    (re.compile(r"^SMC(\d+)$"),  "salesman commission code {}",         "attribute"),
    (re.compile(r"^ONK(\d+)$"),  "order number key {}",                 "attribute"),
    (re.compile(r"^DDF(\d+)$"),  "date field {}",                       "date_key"),
    (re.compile(r"^RSC(\d+)$"),  "reason code {}",                      "attribute"),
    (re.compile(r"^FRE(\d+)$"),  "free field {}",                       "attribute"),
    (re.compile(r"^ODI(\d+)$"),  "other discount info {}",              "attribute"),
    (re.compile(r"^MTX(\d+)$"),  "matrix {}",                           "attribute"),
    (re.compile(r"^BOP(\d+)$"),  "back-order priority {}",              "attribute"),
    (re.compile(r"^TEL(\d+)$"),  "terms of delivery {}",                "attribute"),
    (re.compile(r"^LNA(\d+)$"),  "line name {}",                        "attribute"),
    (re.compile(r"^DMA(\d+)$"),  "delivery method alternative {}",      "attribute"),
    (re.compile(r"^DTP(\d+)$"),  "date type {}",                        "date_key"),
    (re.compile(r"^VOL(\d+)$"),  "volume {}",                           "measure"),
]

# ── Infrastructure / platform field patterns ────────────────────────────────
_INFRA_PREFIXES = ("AZ_",)
_INFRA_CAMEL    = {"accountingEntity", "variationNumber", "timestamp", "deleted", "archived"}

KNOWN_JOIN_EQUIVALENTS: dict[str, list[str]] = {
    "CUS_ORD_NUM": ["ORNO"],
    "CUS_ORD_LIN_NUM": ["PONR"],
    "CUS_ORD_LIN_SFX": ["POSX"],
    "DLV_NUM": ["DLIX"],
    "CUS_IVC_NUM": ["IVNO"],
    "CUS_DMS_KEY": ["CUNO", "PYNO"],
    "WHS_DMS_KEY": ["WHLO"],
    "FCY_DMS_KEY": ["FACI"],
}


@dataclass
class EnrichedColumn:
    column: str
    expanded_name: str
    role: str
    business_candidates: list[str]
    confidence: int
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    default_filter: str = ""
    distinct_values: str = ""
    data_type: str = ""
    nullable: str = ""
    join_equivalents: list[str] = field(default_factory=list)
    date_role: str = ""


def _clean_identifier(name: str) -> str:
    return (name or "").strip().strip('"').strip("`").strip()


def _tokens(column: str) -> list[str]:
    raw = _clean_identifier(column).upper()
    return [t for t in re.split(r"[_\W]+", raw) if t]


def _human_join(parts: list[str]) -> str:
    return " ".join(p for p in parts if p).strip()


def _active_vocab(vocab=None):
    """Resolve the effective terminology vocab (client packs merge over builtins)."""
    if vocab is not None:
        return vocab
    from core.vocab_packs import get_active_vocab
    return get_active_vocab()


def _expand_column(column: str, vocab=None) -> tuple[str, list[str]]:
    col = _clean_identifier(column).upper()
    raw = _clean_identifier(column)  # preserve original case for camelCase check
    v = _active_vocab(vocab)

    # 1. ERP dictionary — highest confidence
    if col in v.column_dict:
        label, _ = v.column_dict[col]
        return label.lower(), ["erp dictionary"]

    # 2. Infrastructure / platform fields
    if any(col.startswith(p.upper()) for p in _INFRA_PREFIXES) or raw in _INFRA_CAMEL:
        return f"data platform field: {raw}", ["infrastructure/platform field"]

    # 3. Numbered series patterns (DIC1, ATV3, UCA7, …)
    for pattern, label_tpl, _ in v.numbered_series:
        m = pattern.match(col)
        if m:
            label = label_tpl.format(m.group(1))
            return label, ["numbered series pattern"]

    # 4. Token-by-token abbreviation expansion
    parts: list[str] = []
    evidence: list[str] = []
    for token in _tokens(col):
        expanded = v.abbreviations.get(token)
        if expanded:
            evidence.append(f"abbreviation {token}={expanded}")
            parts.append(expanded)
        else:
            parts.append(token.lower())
    return _human_join(parts), evidence


def _metric_candidates(column: str, expanded: str, role: str, vocab=None) -> list[str]:
    col = _clean_identifier(column).upper()
    v = _active_vocab(vocab)
    candidates: list[str] = []
    date_role = detect_date_role(col, vocab=v)
    if date_role:
        candidates.extend(date_role_terms(date_role))
    if col in v.column_dict:
        label, synonyms = v.column_dict[col]
        candidates.extend([label.lower(), *[s.lower() for s in synonyms[:4]]])
    if role == "measure":
        candidates.append(expanded)
        if "_AMT" in col or col in {"CUAM", "SAAM", "SGAM"}:
            candidates.append(f"total {expanded}")
        if "CST" in col or col in {"UCOS", "DCOS"}:
            candidates.append(f"{expanded} measure")
        if "QTY" in col or col in {"TRQT", "ORQT", "IVQT", "DLQT"}:
            candidates.append(f"total {expanded}")
    elif role in {"dimension_key", "dimension", "identifier"}:
        candidates.append(expanded)
    elif role == "date_key":
        candidates.append(expanded)
    elif role == "status_filter":
        candidates.append(expanded)

    seen: set[str] = set()
    deduped: list[str] = []
    for candidate in candidates:
        normalized = re.sub(r"\s+", " ", candidate).strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped[:6]


def _role_for_column(column: str, data_type: str = "", distinct_values: str = "", vocab=None) -> tuple[str, list[str], list[str], str]:
    col = _clean_identifier(column).upper()
    ctype = (data_type or "").upper()
    distinct = distinct_values or ""
    v = _active_vocab(vocab)
    evidence: list[str] = []
    warnings: list[str] = []
    default_filter = ""

    raw = _clean_identifier(column)
    # Infrastructure / platform fields — exclude from business queries
    if any(col.startswith(p.upper()) for p in _INFRA_PREFIXES) or raw in _INFRA_CAMEL:
        evidence.append("data platform / infrastructure field")
        return "infrastructure", evidence, warnings, default_filter

    if col in {"DELETED", "ARCHIVED"}:
        evidence.append("standard soft-delete/archive flag — apply as optional filter only when user asks for active/non-deleted records")
        return "status_filter", evidence, warnings, default_filter
    if col == "DEL_REC_IND" or re.match(r"^DEL_[A-Z]+_REC_IND$", col):
        evidence.append("deleted-record indicator — do NOT auto-filter; only apply as WHERE condition when user explicitly requests active or non-deleted records")
        return "status_filter", evidence, warnings, default_filter
    if col.endswith("_FCT_KEY") or col.endswith("_KEY") and col.startswith(col.rsplit("_", 1)[0]):
        if col.endswith("_FCT_KEY"):
            evidence.append("fact surrogate key suffix")
            return "surrogate_key", evidence, warnings, default_filter
    date_role = detect_date_role(col, vocab=v)
    if date_role:
        evidence.append("date key naming pattern")
        evidence.append(f"date role={date_role.label}")
        if col.endswith("_DT_DMS_KEY"):
            warnings.append("Treat as YYYYMMDD integer date key unless metadata proves otherwise.")
        return "date_key", evidence, warnings, default_filter
    if col in {"YEA4"}:
        evidence.append("ERP fiscal year code")
        return "date_attribute", evidence, warnings, default_filter
    if col.endswith("_DMS_KEY"):
        evidence.append("dimension key suffix")
        warnings.append("Dimension key values may be displayed with separators; SQL filters should use raw unformatted literals.")
        return "dimension_key", evidence, warnings, default_filter
    if col in v.raw_identifier_codes:
        evidence.append("ERP identifier/dimension code")
        return "identifier", evidence, warnings, default_filter
    if col in v.raw_measure_codes or any(s in col for s in ("_AMT", "_QTY", "_CST", "_PFT", "_PCE", "_RATE")):
        evidence.append("measure naming pattern")
        return "measure", evidence, warnings, default_filter
    if col.endswith("_IND") or col.endswith("_STS") or col.endswith("_STS_DMS_KEY"):
        evidence.append("status/indicator naming pattern")
        return "status_filter", evidence, warnings, default_filter
    if any(t in ctype for t in ("DECIMAL", "NUMERIC", "FLOAT", "REAL", "MONEY")):
        evidence.append("numeric data type")
        if distinct and len(distinct) < 80:
            warnings.append("Numeric field with low-cardinality distinct values may be a code, not a measure.")
        return "measure_candidate", evidence, warnings, default_filter
    return "attribute", evidence, warnings, default_filter


def _confidence(column: str, role: str, evidence: list[str], expanded_name: str, vocab=None) -> int:
    col = _clean_identifier(column).upper()
    score = 35

    if col in _active_vocab(vocab).column_dict:
        score = 95
    elif "numbered series pattern" in evidence:
        score = 72          # we know what it is structurally, not semantically
    elif "infrastructure/platform field" in evidence or role == "infrastructure":
        score = 80          # well-understood, just not a business column
    elif any(e.startswith("abbreviation") for e in evidence):
        score = 70

    if role in {"measure", "date_key", "dimension_key", "status_filter", "surrogate_key"}:
        score += 10
    if role == "measure_candidate":
        score = min(score, 65)
    if role == "infrastructure":
        score = min(score, 80)          # cap infra fields — not business queryable
    if expanded_name.replace(" ", "") == col.lower():
        score = min(score, 45)          # no expansion found at all
    return max(10, min(score, 95))


def parse_schema_markdown(schema_md: str) -> dict[str, dict[str, str]]:
    """
    Parse the schema markdown table emitted by schema discovery.

    Expected row shape:
    | `COLUMN` | type | nullable | distinct values |
    """
    columns: dict[str, dict[str, str]] = {}
    for line in (schema_md or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("| `") or "---" in stripped:
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not cells:
            continue
        match = re.match(r"`([^`]+)`", cells[0])
        if not match:
            continue
        column = _clean_identifier(match.group(1))
        columns[column] = {
            "type": cells[1] if len(cells) > 1 else "",
            "nullable": cells[2] if len(cells) > 2 else "",
            "distinct_values": cells[3] if len(cells) > 3 else "",
        }
    return columns


def enrich_columns(
    columns: list[str],
    schema_md: str = "",
    vocab=None,
) -> list[EnrichedColumn]:
    v = _active_vocab(vocab)
    schema_meta = parse_schema_markdown(schema_md)
    enriched: list[EnrichedColumn] = []

    for raw_column in columns:
        column = _clean_identifier(raw_column)
        meta = schema_meta.get(column, {})
        data_type = meta.get("type", "")
        distinct_values = meta.get("distinct_values", "")
        expanded, expansion_evidence = _expand_column(column, vocab=v)
        role, role_evidence, warnings, default_filter = _role_for_column(column, data_type, distinct_values, vocab=v)
        evidence = [*expansion_evidence, *role_evidence]
        confidence = _confidence(column, role, evidence, expanded, vocab=v)
        join_equivalents = KNOWN_JOIN_EQUIVALENTS.get(column.upper(), [])
        date_role = detect_date_role(column, vocab=v)
        candidates = _metric_candidates(column, expanded, role, vocab=v)
        enriched.append(
            EnrichedColumn(
                column=column,
                expanded_name=expanded,
                role=role,
                business_candidates=candidates,
                confidence=confidence,
                evidence=evidence or ["no deterministic meaning found"],
                warnings=warnings,
                default_filter=default_filter,
                distinct_values=distinct_values,
                data_type=data_type,
                nullable=meta.get("nullable", ""),
                join_equivalents=join_equivalents,
                date_role=date_role.label if date_role else "",
            )
        )
    return enriched


def format_schema_intelligence(table_name: str, columns: list[str], schema_md: str = "", vocab=None) -> str:
    enriched = enrich_columns(columns, schema_md=schema_md, vocab=vocab)
    if not enriched:
        return ""

    lines = [
        "SCHEMA INTELLIGENCE - DETERMINISTIC FIELD ENRICHMENT:",
        "Use this block before interpreting raw column names. It is generated from exact schema names, ERP dictionaries, naming patterns, and sample/distinct metadata.",
        "If confidence is below 70, document the field as a candidate or [NEEDS ADMIN CONTEXT] instead of treating it as an official business definition.",
        "Official metrics still come only from the approved metric registry or admin approval.",
        "",
        f"Table: {table_name}",
        "Field intelligence:",
    ]
    for item in enriched:
        candidate_text = ", ".join(item.business_candidates) if item.business_candidates else "none"
        evidence_text = "; ".join(item.evidence)
        parts = [
            f"- {item.column}: role={item.role}",
            f"expanded='{item.expanded_name}'",
            f"business candidates={candidate_text}",
            f"confidence={item.confidence}",
            f"evidence={evidence_text}",
        ]
        if item.default_filter:
            parts.append(f"default filter candidate={item.default_filter}")
        if item.join_equivalents:
            parts.append(f"known join equivalents={', '.join(item.join_equivalents)}")
        if item.date_role:
            parts.append(f"date role={item.date_role}")
        if item.warnings:
            parts.append("warnings=" + "; ".join(item.warnings))
        lines.append("; ".join(parts))

    filters = [c.default_filter for c in enriched if c.default_filter]
    if filters:
        lines.extend([
            "",
            "Optional status filter columns (apply ONLY when user explicitly requests active/non-deleted records — do NOT add these to every query automatically):",
            *[f"- {f}" for f in filters],
        ])

    joins = [(c.column, eq) for c in enriched for eq in c.join_equivalents]
    if joins:
        lines.append("")
        lines.append("Known cross-table join aliases:")
        for left, right in joins:
            lines.append(f"- {left} may join to ERP/raw-code column {right} when that column exists in another table.")

    date_roles = [c for c in enriched if c.date_role]
    if date_roles:
        lines.append("")
        lines.append("Date role map:")
        for item in date_roles:
            lines.append(f"- {item.date_role}: use `{item.column}` for questions about {item.date_role.lower()}.")

    lines.append("")
    lines.append("Value-format rule: when users type IDs with or without thousands separators, SQL filters must use the raw database value format, not the UI display format.")
    return "\n".join(lines)


def format_column_reference_for_vocab(
    table_name: str,
    columns: list[str],
    schema_md: str = "",
    *,
    max_terms_per_column: int = 3,
) -> str:
    """
    Build a compact table line for the cross-table Business Vocabulary prompt.

    The important bit is preserving exact column names while adding deterministic
    meanings/roles in parentheses. This helps the LLM map terms like "warehouse
    description" or "order line" without treating the expanded words as real SQL
    columns.
    """
    enriched = enrich_columns(columns, schema_md=schema_md)
    if not enriched:
        return f"  {table_name}:"

    parts: list[str] = []
    for item in enriched:
        hint_bits = [f"role={item.role}", f"meaning={item.expanded_name}"]
        if item.business_candidates:
            terms = ", ".join(item.business_candidates[:max_terms_per_column])
            hint_bits.append(f"terms={terms}")
        if item.default_filter:
            hint_bits.append(f"default_filter={item.default_filter}")
        if item.join_equivalents:
            hint_bits.append(f"joins_like={', '.join(item.join_equivalents)}")
        if item.confidence < 70:
            hint_bits.append("needs_admin_context")
        parts.append(f"{item.column} ({'; '.join(hint_bits)})")

    return f"  {table_name}: " + "; ".join(parts)
