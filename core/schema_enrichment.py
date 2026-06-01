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

from core.erp_column_dict import ERP_COLUMN_DICT


ABBREVIATIONS: dict[str, str] = {
    "ABC": "abc",
    "AGM": "agreement",
    "ALC": "allocated",
    "AMT": "amount",
    "ANN": "annual",
    "APV": "approved",
    "ARA": "area",
    "BAL": "balance",
    "BCK": "back",
    "BUM": "base unit of measure",
    "BUS": "business",
    "BYR": "buyer",
    "CAD": "cad",
    "CCL": "cancelled",
    "CCY": "currency",
    "CFM": "confirmed",
    "CLS": "class",
    "CLU": "cluster",
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
    "DT": "date",
    "DVN": "division",
    "EMCO": "company",
    "FCY": "facility",
    "FCT": "fact",
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
    "MNL": "manual",
    "MSR": "measure",
    "NED": "needed",
    "NGV": "negative",
    "NUM": "number",
    "ON": "on",
    "ORD": "order",
    "ORI": "origin",
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
    "TFR": "transfer",
    "TM": "time",
    "TS": "timestamp",
    "TYP": "type",
    "UNT": "unit",
    "USR": "user",
    "VLD": "valid",
    "VOL": "volume",
    "WHS": "warehouse",
    "WRG": "wrong",
    "XCH": "exchange",
    "XTR": "extra",
}

RAW_DATE_CODES = {"IVDT", "ORDT", "DLDT", "DWDT", "CODT", "ACDT", "DSDT", "PLDT", "RGDT", "LMDT", "DUDT", "RVDT"}
RAW_IDENTIFIER_CODES = {"ORNO", "PONR", "POSX", "DLIX", "IVNO", "CUNO", "SUNO", "PYNO", "SMCD", "WHLO", "FACI", "DIVI", "CONO"}
RAW_MEASURE_CODES = {"TRQT", "ORQT", "IVQT", "DLQT", "ALQT", "RNQT", "PLQT", "CUAM", "SAAM", "SGAM", "UCOS", "DCOS", "MFAM", "SAPR", "NEPR", "PCLA", "OFRA"}

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


def _clean_identifier(name: str) -> str:
    return (name or "").strip().strip('"').strip("`").strip()


def _tokens(column: str) -> list[str]:
    raw = _clean_identifier(column).upper()
    return [t for t in re.split(r"[_\W]+", raw) if t]


def _human_join(parts: list[str]) -> str:
    return " ".join(p for p in parts if p).strip()


def _expand_column(column: str) -> tuple[str, list[str]]:
    col = _clean_identifier(column).upper()
    if col in ERP_COLUMN_DICT:
        label, _ = ERP_COLUMN_DICT[col]
        return label.lower(), ["erp dictionary"]

    parts: list[str] = []
    evidence: list[str] = []
    for token in _tokens(col):
        expanded = ABBREVIATIONS.get(token)
        if expanded:
            evidence.append(f"abbreviation {token}={expanded}")
            parts.append(expanded)
        else:
            parts.append(token.lower())
    return _human_join(parts), evidence


def _metric_candidates(column: str, expanded: str, role: str) -> list[str]:
    col = _clean_identifier(column).upper()
    candidates: list[str] = []
    if col in ERP_COLUMN_DICT:
        label, synonyms = ERP_COLUMN_DICT[col]
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


def _role_for_column(column: str, data_type: str = "", distinct_values: str = "") -> tuple[str, list[str], list[str], str]:
    col = _clean_identifier(column).upper()
    ctype = (data_type or "").upper()
    distinct = distinct_values or ""
    evidence: list[str] = []
    warnings: list[str] = []
    default_filter = ""

    if col in {"DELETED", "ARCHIVED"}:
        evidence.append("standard soft-delete/archive flag")
        default_filter = f"{col} = false"
        return "status_filter", evidence, warnings, default_filter
    if col == "DEL_REC_IND":
        evidence.append("standard deleted-record indicator")
        default_filter = "DEL_REC_IND = 0"
        return "status_filter", evidence, warnings, default_filter
    if col.endswith("_FCT_KEY") or col.endswith("_KEY") and col.startswith(col.rsplit("_", 1)[0]):
        if col.endswith("_FCT_KEY"):
            evidence.append("fact surrogate key suffix")
            return "surrogate_key", evidence, warnings, default_filter
    if col.endswith("_DT_DMS_KEY") or col.endswith("_DATE_DMS_KEY") or col in RAW_DATE_CODES:
        evidence.append("date key naming pattern")
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
    if col in RAW_IDENTIFIER_CODES:
        evidence.append("ERP identifier/dimension code")
        return "identifier", evidence, warnings, default_filter
    if col in RAW_MEASURE_CODES or any(s in col for s in ("_AMT", "_QTY", "_CST", "_PFT", "_PCE", "_RATE")):
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


def _confidence(column: str, role: str, evidence: list[str], expanded_name: str) -> int:
    col = _clean_identifier(column).upper()
    score = 35
    if col in ERP_COLUMN_DICT:
        score = 95
    elif any(e.startswith("abbreviation") for e in evidence):
        score = 70
    if role in {"measure", "date_key", "dimension_key", "status_filter", "surrogate_key"}:
        score += 10
    if role == "measure_candidate":
        score = min(score, 65)
    if expanded_name.replace(" ", "") == col.lower():
        score = min(score, 45)
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
) -> list[EnrichedColumn]:
    schema_meta = parse_schema_markdown(schema_md)
    enriched: list[EnrichedColumn] = []

    for raw_column in columns:
        column = _clean_identifier(raw_column)
        meta = schema_meta.get(column, {})
        data_type = meta.get("type", "")
        distinct_values = meta.get("distinct_values", "")
        expanded, expansion_evidence = _expand_column(column)
        role, role_evidence, warnings, default_filter = _role_for_column(column, data_type, distinct_values)
        evidence = [*expansion_evidence, *role_evidence]
        confidence = _confidence(column, role, evidence, expanded)
        join_equivalents = KNOWN_JOIN_EQUIVALENTS.get(column.upper(), [])
        candidates = _metric_candidates(column, expanded, role)
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
            )
        )
    return enriched


def format_schema_intelligence(table_name: str, columns: list[str], schema_md: str = "") -> str:
    enriched = enrich_columns(columns, schema_md=schema_md)
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
        if item.warnings:
            parts.append("warnings=" + "; ".join(item.warnings))
        lines.append("; ".join(parts))

    filters = [c.default_filter for c in enriched if c.default_filter]
    if filters:
        lines.extend(["", "Default filter candidates:", *[f"- {f}" for f in filters]])

    joins = [(c.column, eq) for c in enriched for eq in c.join_equivalents]
    if joins:
        lines.append("")
        lines.append("Known cross-table join aliases:")
        for left, right in joins:
            lines.append(f"- {left} may join to ERP/raw-code column {right} when that column exists in another table.")

    lines.append("")
    lines.append("Value-format rule: when users type IDs with or without thousands separators, SQL filters must use the raw database value format, not the UI display format.")
    return "\n".join(lines)
