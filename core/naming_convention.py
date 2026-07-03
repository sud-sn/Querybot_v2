"""
core/naming_convention.py

Data warehouse naming convention taxonomy.

Complements core/erp_column_dict.py — the two layers work together:

  erp_column_dict  →  VOCABULARY layer  →  ORNO means "Order Number"
  naming_convention →  GRAMMAR layer    →  _DMS_KEY suffix means "surrogate FK —
                                            never display, always JOIN to _DMS table"

This module maps structural suffix/prefix patterns that appear across ALL tables
regardless of domain, giving the LLM:
  1. What role a column plays (measure, FK, display field, status code, audit…)
  2. How to use it correctly in SQL (SUM vs never SUM, SELECT vs never SELECT)
  3. Which patterns are audit/ETL columns to ignore in business queries

The patterns are matched longest-suffix-first so _DMS_KEY wins over _KEY.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Column suffix rules ────────────────────────────────────────────────────────
# Each entry describes a structural suffix pattern.
# Ordered longest-first so more specific patterns match before shorter ones.

@dataclass
class SuffixRule:
    suffix: str
    role: str                          # surrogate_fk | display | measure | ratio |
                                       # semi_additive | date | timestamp | code |
                                       # status | flag | identifier | audit | grain
    meaning: str                       # one-line plain-English meaning
    aggregation: str                   # additive | non_additive | semi_additive |
                                       # identifier | dimension | none
    sql_guidance: str                  # what the LLM should do
    anti_pattern: str = ""             # what the LLM must NOT do
    display_via: str = ""              # for FK suffixes: how to resolve to display
    format_hint: str = ""              # currency | percentage | integer | date | text


COLUMN_SUFFIX_RULES: list[SuffixRule] = [
    # ── Dimension surrogate FKs ───────────────────────────────────────────────
    # _DT_DMS_KEY must come BEFORE _DMS_KEY (longer suffix wins).
    SuffixRule(
        suffix="_DT_DMS_KEY",
        role="date_fk",
        meaning="Integer foreign key to the date dimension (YYYYMMDD format)",
        aggregation="identifier",
        sql_guidance=(
            "Use in JOIN to DT_DMS for calendar attributes (month, quarter, year). "
            "For date range filters use BETWEEN with YYYYMMDD integers, e.g. BETWEEN 20240101 AND 20241231. "
            "Never use FORMAT() or CONVERT() on this column — it is an integer, not a date type."
        ),
        anti_pattern="FORMAT({col}, 'yyyy-MM-dd') — this column is an INT (YYYYMMDD), not a DATE/DATETIME.",
        format_hint="integer",
    ),
    SuffixRule(
        suffix="_DMS_KEY",
        role="surrogate_fk",
        meaning="Surrogate foreign key referencing a _DMS dimension table",
        aggregation="identifier",
        sql_guidance=(
            "NEVER SELECT directly as a display value — it is a numeric surrogate key. "
            "Always JOIN to the corresponding _DMS table and SELECT the _DSC or _NM column instead. "
            "Pattern: JOIN {prefix}_DMS d ON fact.{col} = d.{col} → SELECT d.{prefix}_DSC"
        ),
        anti_pattern=(
            "SELECT {col} AS [entity name] — this returns a meaningless integer like 1000547, "
            "not the warehouse/customer/item name the user expects."
        ),
        display_via="JOIN {prefix}_DMS t ON f.{col} = t.{col} → use t.{prefix}_DSC or t.{prefix}_NM",
    ),
    SuffixRule(
        suffix="_KEY",
        role="surrogate_fk",
        meaning="Surrogate key — primary or foreign key in the dimensional model",
        aggregation="identifier",
        sql_guidance=(
            "Surrogate keys are internal identifiers. Use only in JOIN conditions. "
            "Never expose in SELECT as a business label — resolve to a display column."
        ),
        anti_pattern="GROUP BY {col} for user-facing results — use the corresponding display column.",
    ),

    # ── Display / label columns ───────────────────────────────────────────────
    SuffixRule(
        suffix="_DSC",
        role="display",
        meaning="Human-readable description — the preferred display label for this entity",
        aggregation="dimension",
        sql_guidance="Use in SELECT and GROUP BY for all user-facing results. This is the canonical display field.",
        format_hint="text",
    ),
    SuffixRule(
        suffix="_DESC",
        role="display",
        meaning="Human-readable description (alternate spelling of _DSC)",
        aggregation="dimension",
        sql_guidance="Use in SELECT and GROUP BY for all user-facing results.",
        format_hint="text",
    ),
    SuffixRule(
        suffix="_DESCRIPTION",
        role="display",
        meaning="Full human-readable description",
        aggregation="dimension",
        sql_guidance="Use in SELECT and GROUP BY for all user-facing results.",
        format_hint="text",
    ),
    SuffixRule(
        suffix="_NM",
        role="display",
        meaning="Human-readable name for this entity",
        aggregation="dimension",
        sql_guidance="Use in SELECT and GROUP BY for all user-facing results.",
        format_hint="text",
    ),
    SuffixRule(
        suffix="_NAME",
        role="display",
        meaning="Human-readable name for this entity",
        aggregation="dimension",
        sql_guidance="Use in SELECT and GROUP BY for all user-facing results.",
        format_hint="text",
    ),

    # ── Code columns (short, filterable) ─────────────────────────────────────
    SuffixRule(
        suffix="_CD",
        role="code",
        meaning="Short business code — abbreviated identifier meaningful to users",
        aggregation="dimension",
        sql_guidance=(
            "Use in WHERE filters (exact match) and GROUP BY. "
            "Shorter than _DSC but still a business-meaningful value (e.g. 'MELB01', 'USD', 'A')."
        ),
        format_hint="text",
    ),
    SuffixRule(
        suffix="_CODE",
        role="code",
        meaning="Short business code — abbreviated identifier meaningful to users",
        aggregation="dimension",
        sql_guidance="Use in WHERE filters (exact match) and GROUP BY.",
        format_hint="text",
    ),

    # ── Monetary measures (additive) ──────────────────────────────────────────
    SuffixRule(
        suffix="_AMT",
        role="measure",
        meaning="Monetary amount — additive financial measure",
        aggregation="additive",
        sql_guidance="Safe to SUM across all dimensions. Format as currency.",
        format_hint="currency",
    ),
    SuffixRule(
        suffix="_CST",
        role="measure",
        meaning="Cost amount — additive financial measure",
        aggregation="additive",
        sql_guidance="Safe to SUM across all dimensions. Format as currency.",
        format_hint="currency",
    ),
    SuffixRule(
        suffix="_PFT",
        role="measure",
        meaning="Profit amount — additive financial measure",
        aggregation="additive",
        sql_guidance="Safe to SUM across all dimensions. Format as currency.",
        format_hint="currency",
    ),
    SuffixRule(
        suffix="_REV",
        role="measure",
        meaning="Revenue amount — additive financial measure",
        aggregation="additive",
        sql_guidance="Safe to SUM across all dimensions. Format as currency.",
        format_hint="currency",
    ),

    # ── Quantity measures (additive) ──────────────────────────────────────────
    SuffixRule(
        suffix="_QTY",
        role="measure",
        meaning="Quantity — additive unit measure",
        aggregation="additive",
        sql_guidance="Safe to SUM across all dimensions. Format as integer or decimal.",
        format_hint="integer",
    ),
    SuffixRule(
        suffix="_CNT",
        role="measure",
        meaning="Count — additive integer measure",
        aggregation="additive",
        sql_guidance="Safe to SUM or COUNT. Format as integer.",
        format_hint="integer",
    ),
    SuffixRule(
        suffix="_VOL",
        role="measure",
        meaning="Volume — additive unit measure",
        aggregation="additive",
        sql_guidance="Safe to SUM across all dimensions.",
        format_hint="integer",
    ),
    SuffixRule(
        suffix="_WGT",
        role="measure",
        meaning="Weight — additive unit measure",
        aggregation="additive",
        sql_guidance="Safe to SUM across all dimensions.",
        format_hint="integer",
    ),

    # ── Non-additive ratios / percentages ─────────────────────────────────────
    SuffixRule(
        suffix="_PCT",
        role="ratio",
        meaning="Percentage or rate — non-additive derived measure",
        aggregation="non_additive",
        sql_guidance=(
            "NEVER SUM. Always recalculate from component measures: "
            "SUM(numerator) / NULLIF(SUM(denominator), 0) * 100."
        ),
        anti_pattern=(
            "SUM({col}) — summing percentages produces nonsense (e.g. 847% gross margin). "
            "Always recalculate from the underlying additive components."
        ),
        format_hint="percentage",
    ),
    SuffixRule(
        suffix="_RATE",
        role="ratio",
        meaning="Rate — non-additive derived measure",
        aggregation="non_additive",
        sql_guidance="NEVER SUM. Recalculate from component measures.",
        anti_pattern="SUM({col}) — rates are non-additive and must be recalculated.",
        format_hint="percentage",
    ),
    SuffixRule(
        suffix="_RATIO",
        role="ratio",
        meaning="Ratio — non-additive derived measure",
        aggregation="non_additive",
        sql_guidance="NEVER SUM. Recalculate from component measures.",
        anti_pattern="SUM({col}) — ratios are non-additive.",
        format_hint="percentage",
    ),
    SuffixRule(
        suffix="_PER",
        role="ratio",
        meaning="Per-unit rate — non-additive derived measure",
        aggregation="non_additive",
        sql_guidance="NEVER SUM. Use AVG or recalculate from components.",
        anti_pattern="SUM({col}) — per-unit rates are non-additive.",
        format_hint="percentage",
    ),

    # ── Semi-additive balances ────────────────────────────────────────────────
    SuffixRule(
        suffix="_BAL",
        role="semi_additive",
        meaning="Balance — semi-additive measure (point-in-time snapshot)",
        aggregation="semi_additive",
        sql_guidance=(
            "SUM by entity (product, customer, warehouse) is valid. "
            "Do NOT SUM across time periods — use the latest snapshot instead: "
            "WHERE date_key = (SELECT MAX(date_key) FROM ...)."
        ),
        anti_pattern=(
            "SUM({col}) GROUP BY month — sums a balance across months, "
            "which double-counts. Use MAX date or a snapshot approach."
        ),
        format_hint="currency",
    ),
    SuffixRule(
        suffix="_INV",
        role="semi_additive",
        meaning="Inventory level — semi-additive snapshot measure",
        aggregation="semi_additive",
        sql_guidance="SUM by entity is valid. Do NOT SUM across time. Use latest snapshot for current stock.",
        anti_pattern="SUM({col}) over a date range — overstates inventory by counting the same stock multiple times.",
        format_hint="integer",
    ),

    # ── Date columns ──────────────────────────────────────────────────────────
    SuffixRule(
        suffix="_DT",
        role="date",
        meaning="Business date — transaction or event date (stored as DATE or INT YYYYMMDD)",
        aggregation="none",
        sql_guidance=(
            "Use in WHERE for date range filters. If stored as INT (YYYYMMDD), "
            "filter with BETWEEN 20240101 AND 20241231. "
            "For relative time (last month, YTD), anchor to MAX(date_col) not GETDATE()."
        ),
        format_hint="date",
    ),
    SuffixRule(
        suffix="_DATE",
        role="date",
        meaning="Business date — transaction or event date",
        aggregation="none",
        sql_guidance=(
            "Use in WHERE for date range filters. "
            "For relative time queries, anchor to MAX(date_col) not GETDATE()/SYSDATE."
        ),
        format_hint="date",
    ),

    # ── Timestamp columns (usually audit / ETL) ───────────────────────────────
    SuffixRule(
        suffix="_TS",
        role="timestamp",
        meaning="System timestamp — typically ETL load time or row modification time, NOT a business event date",
        aggregation="none",
        sql_guidance=(
            "Do NOT use for business date filtering. This records when the row was loaded/modified by the ETL pipeline. "
            "Use the corresponding _DT or _DMS_KEY business date column instead."
        ),
        anti_pattern=(
            "WHERE {col} BETWEEN @start AND @end — this filters by ETL load time, "
            "not by the business transaction date. Use the _DT column instead."
        ),
    ),
    SuffixRule(
        suffix="_DTM",
        role="timestamp",
        meaning="System datetime — ETL or system timestamp, not a business event date",
        aggregation="none",
        sql_guidance="Use _DT or date _DMS_KEY columns for business date filtering, not this system datetime.",
        anti_pattern="Use for business date filters — this is a system/ETL datetime.",
    ),

    # ── Status / type / flag columns ──────────────────────────────────────────
    SuffixRule(
        suffix="_STS",
        role="status",
        meaning="Status code — current state of the entity (active, closed, cancelled…)",
        aggregation="dimension",
        sql_guidance="Use in WHERE to filter by state. Check distinct values in the KB for valid codes.",
        format_hint="text",
    ),
    SuffixRule(
        suffix="_TYP",
        role="type",
        meaning="Type code — category or classification of the entity",
        aggregation="dimension",
        sql_guidance="Use in WHERE filters and GROUP BY for type-level analysis.",
        format_hint="text",
    ),
    SuffixRule(
        suffix="_GRP",
        role="group",
        meaning="Group code — higher-level grouping or category",
        aggregation="dimension",
        sql_guidance="Use in GROUP BY for group-level rollups. Use in WHERE to filter by group.",
        format_hint="text",
    ),
    SuffixRule(
        suffix="_FLG",
        role="flag",
        meaning="Boolean flag — yes/no or 0/1 indicator field",
        aggregation="dimension",
        sql_guidance="Use in WHERE filters (= 1 or = 'Y'). Do not SUM unless counting occurrences.",
        anti_pattern="NEVER SUM({col}) as a measure — it is a flag, not a quantity.",
        format_hint="text",
    ),
    SuffixRule(
        suffix="_IND",
        role="flag",
        meaning="Boolean indicator — yes/no or 0/1 flag field",
        aggregation="dimension",
        sql_guidance="Use in WHERE filters. Do not SUM unless counting occurrences.",
        format_hint="text",
    ),
    SuffixRule(
        suffix="_YN",
        role="flag",
        meaning="Yes/No flag — values are typically 'Y' or 'N'",
        aggregation="dimension",
        sql_guidance="Filter with WHERE {col} = 'Y' or WHERE {col} = 'N'.",
        format_hint="text",
    ),

    # ── Number / identifier columns ───────────────────────────────────────────
    SuffixRule(
        suffix="_NUM",
        role="identifier",
        meaning="Business number — a human-assigned reference number (order number, invoice number)",
        aggregation="identifier",
        sql_guidance=(
            "Use in WHERE for exact lookups (WHERE {col} = '12345'). "
            "May appear in GROUP BY if reporting at document level. "
            "Do not SUM — it is a reference number, not a quantity."
        ),
        format_hint="text",
    ),
    SuffixRule(
        suffix="_NBR",
        role="identifier",
        meaning="Business number — human-assigned reference (alternate suffix for _NUM)",
        aggregation="identifier",
        sql_guidance="Use in WHERE for exact lookups. Do not SUM.",
        format_hint="text",
    ),
    SuffixRule(
        suffix="_NO",
        role="identifier",
        meaning="Business number — human-assigned reference identifier",
        aggregation="identifier",
        sql_guidance="Use in WHERE for exact lookups. Do not SUM.",
        format_hint="text",
    ),

    # ── Grain-level suffixes ──────────────────────────────────────────────────
    SuffixRule(
        suffix="_LIN",
        role="grain",
        meaning="Line-level field — this column/table operates at invoice/order line grain",
        aggregation="none",
        sql_guidance=(
            "One row = one line item. To get document-level totals, "
            "GROUP BY the header key (order number, invoice number) before aggregating."
        ),
    ),
    SuffixRule(
        suffix="_HDR",
        role="grain",
        meaning="Header-level field — this column/table operates at document header grain",
        aggregation="none",
        sql_guidance="One row = one document header (order, invoice). Measures here are already at header level.",
    ),
    SuffixRule(
        suffix="_DTL",
        role="grain",
        meaning="Detail-level field — this column/table operates at transaction detail grain",
        aggregation="none",
        sql_guidance=(
            "One row = one detail/line item. To get summary totals, "
            "GROUP BY the parent key before aggregating."
        ),
    ),
]

# ── Table suffix roles ─────────────────────────────────────────────────────────

@dataclass
class TableSuffixRule:
    suffix: str
    table_type: str
    meaning: str
    sql_guidance: str


TABLE_SUFFIX_RULES: list[TableSuffixRule] = [
    TableSuffixRule(
        suffix="_FCT",
        table_type="fact_table",
        meaning="Fact table — transactional records containing measures (amounts, quantities) and FK keys to dimension tables",
        sql_guidance=(
            "Contains the numerical measures you aggregate (SUM, COUNT, AVG). "
            "Join to _DMS tables to resolve dimension keys to display labels. "
            "One row typically represents one business event (invoice line, transaction, receipt)."
        ),
    ),
    TableSuffixRule(
        suffix="_FACT",
        table_type="fact_table",
        meaning="Fact table — transactional records with measures and FK dimension keys",
        sql_guidance="Same as _FCT. Contains additive measures. Join to dimension tables for display labels.",
    ),
    TableSuffixRule(
        suffix="_DMS",
        table_type="dimension_table",
        meaning="Dimension table — reference/lookup data with descriptive attributes for a business entity",
        sql_guidance=(
            "Contains display fields (_DSC, _NM, _CD) and attributes for a single entity (warehouse, customer, item). "
            "Join from fact tables via the matching _DMS_KEY. Never aggregate measures from this table alone."
        ),
    ),
    TableSuffixRule(
        suffix="_DIM",
        table_type="dimension_table",
        meaning="Dimension table — reference/lookup data (alternate suffix for _DMS)",
        sql_guidance="Same as _DMS. Contains display fields. Join from fact tables via the matching FK.",
    ),
    TableSuffixRule(
        suffix="_EXT",
        table_type="extended_table",
        meaning="Extended table — supplementary or externally-sourced data joined to a core table",
        sql_guidance=(
            "Contains additional attributes that extend the main fact or dimension. "
            "Join to the corresponding base table to enrich results."
        ),
    ),
    TableSuffixRule(
        suffix="_STG",
        table_type="staging_table",
        meaning="Staging table — pre-production ETL landing zone, typically not used in business reporting",
        sql_guidance=(
            "Avoid using staging tables for business queries — data may be incomplete or un-validated. "
            "Prefer the corresponding _FCT or _DMS production table."
        ),
    ),
    TableSuffixRule(
        suffix="_RPT",
        table_type="report_table",
        meaning="Report/summary table — pre-aggregated data for faster reporting queries",
        sql_guidance=(
            "Data is already summarised — do not re-aggregate unless joining at a finer grain. "
            "Check the grain documented in the KB before grouping."
        ),
    ),
    TableSuffixRule(
        suffix="_VW",
        table_type="view",
        meaning="Database view — a virtual table combining multiple underlying tables",
        sql_guidance="Treat like a regular table. Check the KB for what underlying tables this view consolidates.",
    ),
    TableSuffixRule(
        suffix="_AGG",
        table_type="aggregate_table",
        meaning="Aggregate table — pre-computed roll-up for performance-critical queries",
        sql_guidance="Data is pre-aggregated. Do not double-aggregate. Use only when the required grain matches.",
    ),
]

# ── Audit / ETL prefix rules ───────────────────────────────────────────────────
# Columns starting with these prefixes are system/pipeline columns — never use
# in business filters, never SELECT as a result column.

@dataclass
class AuditPrefixRule:
    prefix: str
    meaning: str
    guidance: str


AUDIT_PREFIX_RULES: list[AuditPrefixRule] = [
    AuditPrefixRule(
        prefix="AZ_",
        meaning="Azure Data Factory pipeline audit column (load timestamp, batch ID, source tracking)",
        guidance=(
            "IGNORE in all business queries. Never use in WHERE for date filtering — "
            "this records ETL load time, not transaction date. "
            "Never SELECT in results."
        ),
    ),
    AuditPrefixRule(
        prefix="ETL_",
        meaning="ETL pipeline metadata column",
        guidance="IGNORE in all business queries. System-generated ETL tracking column.",
    ),
    AuditPrefixRule(
        prefix="DW_",
        meaning="Data warehouse system metadata column",
        guidance="IGNORE in all business queries. Internal DW management column.",
    ),
    AuditPrefixRule(
        prefix="SYS_",
        meaning="System-generated column — not a business attribute",
        guidance="IGNORE in all business queries.",
    ),
    AuditPrefixRule(
        prefix="META_",
        meaning="Metadata column — pipeline or system tracking",
        guidance="IGNORE in all business queries.",
    ),
    AuditPrefixRule(
        prefix="STG_",
        meaning="Staging metadata column — pre-production ETL state",
        guidance="IGNORE in all business queries. Use the corresponding business column.",
    ),
    AuditPrefixRule(
        prefix="CDC_",
        meaning="Change data capture column — row-level change tracking",
        guidance="IGNORE in all business queries. Used by replication pipelines only.",
    ),
]

# ── Entity prefix vocabulary ────────────────────────────────────────────────────
# Common entity prefixes used in DW column naming.
# These help the LLM understand WHAT entity a column belongs to even when
# the full column name is abbreviated (e.g. WHS_DMS_KEY → Warehouse FK).

ENTITY_PREFIX_VOCABULARY: dict[str, str] = {
    "CUS":   "Customer",
    "ITM":   "Item / Product",
    "ORD":   "Order",
    "IVC":   "Invoice",
    "WHS":   "Warehouse",
    "DT":    "Date / Calendar",
    "PFT":   "Profit Center",
    "FCY":   "Facility / Factory",
    "SLR":   "Seller / Sales Rep",
    "PC":    "Profit Center",
    "DVN":   "Division",
    "DIV":   "Division",
    "RGN":   "Region",
    "PCH":   "Purchase",
    "DLV":   "Delivery",
    "SOP":   "Sales Order Processing (calculated / derived measure)",
    "ITM_GRP": "Item Group",
    "CUS_ORD": "Customer Order",
    "CUS_IVC": "Customer Invoice",
    "PCH_ORD": "Purchase Order",
    "PCH_GRP": "Purchase Group",
    "DLV_TER": "Delivery Territory",
    "DLV_MTH": "Delivery Method",
    "ITM_BUS_ARA": "Item Business Area",
    "ITM_STS": "Item Status",
    "PDC_GRP": "Product Group",
    "EMCO_RGN": "Company Region",
    "PC_DVN": "Profit Center Division",
}


# ── Matching helpers ───────────────────────────────────────────────────────────

def match_column_suffix(column: str) -> SuffixRule | None:
    """
    Return the first matching SuffixRule for this column name.
    Longer suffixes are tested before shorter ones (list is ordered).
    """
    col_upper = (column or "").upper()
    for rule in COLUMN_SUFFIX_RULES:
        if col_upper.endswith(rule.suffix.upper()):
            return rule
    return None


def _nc_vocab(vocab=None):
    """Resolve the terminology vocab; defaults preserve legacy constants."""
    if vocab is not None:
        return vocab
    from core.vocab_packs import get_active_vocab
    return get_active_vocab()


_PACK_FACT_RULE = TableSuffixRule(
    suffix="",
    table_type="fact_table",
    meaning="Fact table (classified by the client's terminology pack) — transactional records containing measures and FK dimension keys",
    sql_guidance=(
        "Contains the numerical measures you aggregate (SUM, COUNT, AVG). "
        "Join to dimension tables to resolve keys to display labels."
    ),
)
_PACK_DIM_RULE = TableSuffixRule(
    suffix="",
    table_type="dimension_table",
    meaning="Dimension table (classified by the client's terminology pack) — lookup/master records with display fields",
    sql_guidance="Join from fact tables to resolve keys; select the display/description columns for output.",
)
_PACK_BRIDGE_RULE = TableSuffixRule(
    suffix="",
    table_type="bridge_table",
    meaning="Bridge table (classified by the client's terminology pack) — resolves many-to-many relationships",
    sql_guidance="Join through this table between the two related entities; do not aggregate its rows directly.",
)


def match_table_suffix(table_name: str, vocab=None) -> TableSuffixRule | None:
    """Return the first matching TableSuffixRule for this table name.

    Terminology-pack classification (explicit table lists, then fact/dim
    patterns) is consulted first so non-DMS warehouses (FACT_SALES, VBAK…)
    classify correctly once a pack is enabled; builtin suffix rules follow.
    """
    tbl_upper = (table_name or "").upper().split(".")[-1]   # bare table name only
    v = _nc_vocab(vocab)
    if tbl_upper in v.fact_tables:
        return _PACK_FACT_RULE
    if tbl_upper in v.dimension_tables:
        return _PACK_DIM_RULE
    for pattern in v.fact_patterns:
        if pattern.search(tbl_upper):
            return _PACK_FACT_RULE
    for pattern in v.dimension_patterns:
        if pattern.search(tbl_upper):
            return _PACK_DIM_RULE
    for pattern in v.bridge_patterns:
        if pattern.search(tbl_upper):
            return _PACK_BRIDGE_RULE
    for rule in TABLE_SUFFIX_RULES:
        if tbl_upper.endswith(rule.suffix.upper()):
            return rule
    return None


def match_audit_prefix(column: str) -> AuditPrefixRule | None:
    """Return the matching AuditPrefixRule if this column is an audit/ETL column."""
    col_upper = (column or "").upper()
    for rule in AUDIT_PREFIX_RULES:
        if col_upper.startswith(rule.prefix.upper()):
            return rule
    return None


def match_entity_prefix(column: str, vocab=None) -> str | None:
    """
    Return the entity domain for this column based on its prefix.
    Tries longest prefix first. Terminology-pack prefixes merge over builtins.
    """
    col_upper = (column or "").upper()
    prefixes = {**ENTITY_PREFIX_VOCABULARY, **_nc_vocab(vocab).entity_prefixes}
    sorted_prefixes = sorted(prefixes.keys(), key=len, reverse=True)
    for prefix in sorted_prefixes:
        if col_upper.startswith(prefix.upper() + "_") or col_upper == prefix.upper():
            return prefixes[prefix]
    return None


# ── Hint generation ────────────────────────────────────────────────────────────

def get_naming_hints(column_names: list[str], table_name: str = "", vocab=None) -> str:
    """
    Return a formatted hint block for columns in `column_names` that match
    a naming convention pattern.  Parallel to get_erp_hints() in erp_column_dict.

    Covers four tiers:
      1. Audit prefix    → IGNORE this column
      2. Column suffix   → role + aggregation rule + SQL guidance
      3. Entity prefix   → which business entity this column belongs to
      4. Table suffix    → what type of table this is (injected once at the top)
    """
    if not column_names:
        return ""

    lines: list[str] = []

    # Table-level hint (once)
    if table_name:
        tbl_rule = match_table_suffix(table_name, vocab=vocab)
        if tbl_rule:
            lines.append(
                f"TABLE TYPE [{table_name}]: {tbl_rule.table_type.upper()} — "
                f"{tbl_rule.meaning} | {tbl_rule.sql_guidance}"
            )
            lines.append("")

    # Per-column hints
    seen_suffixes: set[str] = set()   # avoid repeating the same suffix rule 10 times
    for col in column_names:
        col_upper = col.upper()

        # Tier 1: Audit/ETL columns
        audit = match_audit_prefix(col)
        if audit:
            lines.append(
                f"- {col} [AUDIT/ETL — {audit.meaning}]: {audit.guidance}"
            )
            continue

        # Tier 2: Column suffix rule
        suffix_rule = match_column_suffix(col)
        entity = match_entity_prefix(col, vocab=vocab)

        if suffix_rule:
            entity_part = f" | entity domain: {entity}" if entity else ""
            guidance = suffix_rule.sql_guidance.replace("{col}", col)
            anti = (
                f" | ANTI-PATTERN: {suffix_rule.anti_pattern.replace('{col}', col)}"
                if suffix_rule.anti_pattern else ""
            )
            lines.append(
                f"- {col} [{suffix_rule.role.upper()}{entity_part}]: "
                f"{suffix_rule.meaning} | aggregation={suffix_rule.aggregation} | "
                f"guidance={guidance}{anti}"
            )
        elif entity:
            # Tier 3: Entity prefix only (no matching suffix rule)
            lines.append(f"- {col} [entity domain: {entity}]")

    return "\n".join(lines)


# ── Global KB document ─────────────────────────────────────────────────────────

def build_naming_convention_doc(vocab=None) -> str:
    """
    Build the full _naming_convention.md document embedded as a global KB doc.
    Retrieved at query time to give the LLM structural grammar rules that apply
    across every table — regardless of which table's KB is in context.
    """
    v = _nc_vocab(vocab)
    lines = [
        "# Data Warehouse Naming Convention Reference",
        "",
        "This document describes the structural naming grammar used across all tables.",
        "Use it alongside per-table KB documents to understand column roles and correct SQL patterns.",
        "",
    ]
    pack_ids = [p for p in getattr(v, "source_packs", []) if p != "builtin"]
    if pack_ids:
        lines += [
            f"Client terminology pack(s): {', '.join(pack_ids)}",
            "Pack-specific conventions below merge over (and take precedence for",
            "overlapping codes) the built-in defaults.",
            "",
        ]
    lines += [

        "---",
        "",
        "## Table Types (by suffix)",
        "",
        "| Table Suffix | Type | Meaning | SQL Guidance |",
        "| --- | --- | --- | --- |",
    ]
    for r in TABLE_SUFFIX_RULES:
        lines.append(f"| `{r.suffix}` | {r.table_type} | {r.meaning} | {r.sql_guidance} |")

    lines += [
        "",
        "---",
        "",
        "## Column Suffix Rules",
        "",
        "Suffixes encode the **role** of a column and the **correct way to use it in SQL**.",
        "",
        "| Suffix | Role | Aggregation | SQL Guidance | Anti-Pattern |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in COLUMN_SUFFIX_RULES:
        anti = r.anti_pattern.replace("\n", " ") if r.anti_pattern else "—"
        guidance = r.sql_guidance.replace("\n", " ")
        lines.append(
            f"| `{r.suffix}` | {r.role} | {r.aggregation} | {guidance} | {anti} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Audit / ETL Prefixes — ALWAYS IGNORE IN BUSINESS QUERIES",
        "",
        "| Prefix | Meaning | Guidance |",
        "| --- | --- | --- |",
    ]
    for r in AUDIT_PREFIX_RULES:
        lines.append(f"| `{r.prefix}` | {r.meaning} | {r.guidance} |")

    lines += [
        "",
        "---",
        "",
        "## Entity Prefix Vocabulary",
        "",
        "| Prefix | Business Entity |",
        "| --- | --- |",
    ]
    for prefix, entity in sorted(ENTITY_PREFIX_VOCABULARY.items()):
        lines.append(f"| `{prefix}_` | {entity} |")

    lines += [
        "",
        "---",
        "",
        "## Key Rules Summary",
        "",
        "1. **`_DMS_KEY` columns** — NEVER SELECT directly. Always JOIN to the `_DMS` table and use `_DSC` or `_NM`.",
        "2. **`_PCT`, `_RATE`, `_RATIO` columns** — NEVER SUM. Recalculate as `SUM(num)/SUM(denom)*100`.",
        "3. **`_AMT`, `_QTY`, `_CST`, `_PFT`, `_REV` columns** — safe to SUM (additive).",
        "4. **`_BAL`, `_INV` columns** — semi-additive: SUM by entity OK, never SUM across time.",
        "5. **`_TS`, `_DTM` columns** — system/ETL timestamps, NOT business dates. Use `_DT` or `_DT_DMS_KEY` for date filtering.",
        "6. **`AZ_`, `ETL_`, `DW_`, `SYS_` prefixes** — audit/pipeline columns. Never use in business queries.",
        "7. **`_FCT` tables** — contain measures. Always join to `_DMS` tables to resolve FK keys to display labels.",
        "8. **`_DMS` tables** — contain display fields. Use `_DSC`/`_NM` in SELECT, `_KEY` only for JOIN conditions.",
        "",
    ]
    return "\n".join(lines)
