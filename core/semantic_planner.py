"""
Lightweight semantic field planner.

This module builds a deterministic field-source plan from the discovered schema
before the LLM writes SQL. It is intentionally conservative: it only emits a
plan when a user phrase maps to exact known columns and, when needed, a
reasonable join path can be inferred.
"""

from __future__ import annotations

import re
from collections import deque


_ABBREVIATIONS = {
    # ── Generic dimensional abbreviations ─────────────────────────────────
    "AMT": "amount",
    "BAL": "balance",
    "BUS": "business",
    "CST": "cost",
    "CUS": "customer",
    "DLV": "delivery",
    "DLVD": "delivered",
    "DMS": "dimension",
    "DT": "date",
    "DVN": "division",
    "FCT": "fact",
    "GRP": "group",
    "IVC": "invoice",
    "IVCD": "invoiced",
    "ITM": "item",
    "LIN": "line",
    "NUM": "number",
    "ORD": "order",
    "PCH": "purchase",
    "PFT": "profit",
    "PRD": "period",
    "QTY": "quantity",
    "RCT": "receipt",
    "RPL": "replacement",
    "SFX": "suffix",
    "WHS": "warehouse",
    # ── M3/ERP raw short codes ─────────────────────────────────────────────
    "ACDT": "accounting date",
    "ALQT": "allocated quantity",
    "CONO": "company",
    "CSCD": "country",
    "CUAM": "customer amount",
    "CUCD": "currency",
    "CUNO": "customer",
    "DCOS": "delivery cost",
    "DIVI": "division",
    "DLIX": "delivery",
    "DLDT": "delivery date",
    "DLQT": "delivered quantity",
    "DWDT": "requested delivery date",
    "FACI": "facility",
    "ITDS": "item description",
    "ITGR": "item group",
    "ITNO": "item number",
    "ITTY": "item type",
    "IVDT": "invoice date",
    "IVNO": "invoice number",
    "IVQT": "invoiced quantity",
    "MFAM": "manufacturing amount",
    "ORDT": "order date",
    "ORNO": "order number",
    "ORQT": "ordered quantity",
    "ORST": "order status",
    "ORTP": "order type",
    "PCLA": "profit class",
    "PONR": "order line number",
    "POSX": "order line suffix",
    "SAAM": "sales amount",
    "SAPR": "sales price",
    "SDST": "sales district",
    "SMCD": "salesman",
    "SUNO": "supplier",
    "TRQT": "transaction quantity",
    "UCOS": "unit cost",
    "WHLO": "warehouse",
    "YEA4": "fiscal year",
}

_DIRECT_ALIASES = {
    # ── Dimensional fact columns ───────────────────────────────────────────
    "DIVI": {"division"},
    "ITM_GRP_DMS_KEY": {"item group", "item group key", "product group"},
    "CUS_IVC_LIN_AMT": {
        "invoice line amount",
        "total invoice line amount",
        "invoice amount",
        "sales amount",
    },
    "SOP_CUS_LIN_GRS_PFT_AMT": {
        "gross profit",
        "sales gross profit",
        "customer line gross profit",
    },
    "CUR_ON_HND_QTY": {"current on hand quantity", "on hand quantity"},
    "RCT_BUM_QTY": {"purchase receipt quantity", "receipt quantity"},
    "CUR_RPL_CST_AMT": {"current replacement cost", "replacement cost"},
    # ── M3/ERP raw column aliases ──────────────────────────────────────────
    # RULE: use only QUALIFIED multi-word phrases here — never bare generic words
    # like "quantity", "item", "supplier", "amount" which match too broadly and
    # pull the wrong table into the semantic plan.
    "TRQT": {"transaction quantity", "transaction qty"},           # NOT "quantity"/"volume"/"units"
    "PCLA": {"profit class", "fifo profit", "fifo margin", "margin tier", "fifo layer", "pcla"},
    "SUNO": {"supplier number", "vendor number"},                  # NOT bare "supplier"/"vendor"
    "CUNO": {"customer number"},                                   # NOT bare "customer"/"client"
    "SMCD": {"salesman code", "salesperson code", "sales rep code", "smcd"},
    "CUAM": {"customer amount", "billed amount"},                  # NOT "sales amount"/"revenue"
    "SAAM": {"net sales amount", "gross sales amount"},            # NOT bare "sales"
    "UCOS": {"unit cost", "cost per unit", "cogs per unit"},
    "WHLO": {"warehouse location", "whs"},                         # NOT bare "warehouse" (too broad)
    "ORNO": {"order number", "sales order number"},
    "IVNO": {"invoice number"},
    "IVQT": {"invoiced quantity", "invoiced qty", "billed quantity"},
    "ORQT": {"ordered quantity", "ordered qty", "order quantity"},
    "DLQT": {"delivered quantity", "delivered qty", "shipped quantity"},
    "SAPR": {"sales price", "list price"},                         # NOT bare "price"
    "ITNO": {"item number", "part number"},                        # NOT bare "item"/"product"/"sku"
    "ITGR": {"item group", "product group"},                       # NOT bare "category"
}

_JOIN_SYNONYMS = {
    "CUS_ORD_NUM": {"ORNO"},
    "CUS_ORD_LIN_NUM": {"PONR"},
    "CUS_ORD_LIN_SFX": {"POSX"},
}

_MEASURE_HINTS = ("amount", "profit", "quantity", "cost", "margin", "sales", "invoice")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _table_bare(table: str) -> str:
    return (table or "").upper().split(".")[-1]


def _table_schema(table: str) -> str:
    parts = (table or "").upper().split(".")
    return parts[-2] if len(parts) >= 2 else ""


def _table_variants(table: str) -> set[str]:
    table_u = (table or "").upper()
    parts = table_u.split(".")
    variants = {table_u}
    if parts:
        variants.add(parts[-1])
    if len(parts) >= 2:
        variants.add(".".join(parts[-2:]))
    return {v for v in variants if v}


def _column_words(column: str) -> list[str]:
    words: list[str] = []
    for token in re.split(r"[_\W]+", (column or "").upper()):
        if not token:
            continue
        word = _ABBREVIATIONS.get(token, token.lower())
        if word not in {"key", "dimension"}:
            words.append(word)
    return words


def _aliases_for_column(column: str) -> set[str]:
    col = (column or "").upper()
    aliases = {_norm(col), _norm(" ".join(_column_words(col)))}
    aliases.update(_norm(a) for a in _DIRECT_ALIASES.get(col, set()))
    return {a for a in aliases if a}


def _role_for_column(column: str, col_type: str = "") -> str:
    col = (column or "").upper()
    ctype = (col_type or "").upper()
    if col.endswith("_DT_DMS_KEY") or col.endswith("_DATE_DMS_KEY"):
        return "date_key"
    if col in {"DIVI"} or col.endswith("_DMS_KEY") or col in {"WHLO", "ORNO", "PONR", "POSX"}:
        return "dimension"
    if any(suffix in col for suffix in ("_AMT", "_QTY", "_CST", "_PFT")):
        return "measure"
    if any(token in ctype for token in ("INT", "DECIMAL", "NUMBER", "NUMERIC", "FLOAT")):
        return "measure"
    return "attribute"


def _table_context_score(table: str, question: str) -> int:
    t = _table_bare(table)
    q = _norm(question)
    score = 0
    if any(w in q for w in ("invoice", "sales", "customer order")) and any(x in t for x in ("CUS_ORD_IVC", "OOLINE", "OSBSTD")):
        score += 4
    if any(w in q for w in ("inventory", "on hand", "stock")) and "ITM_BAL" in t:
        score += 4
    if "purchase" in q and "PCH_ORD_RCT" in t:
        score += 4
    if "replacement" in q and "RPL_CST" in t:
        score += 4
    if any(w in q for w in ("fifo", "pcla", "margin")) and ("FIFO" in t or "OOLINE" in t):
        score += 4
    return score


def _score_candidate(table: str, column: str, role: str, question: str, base_tables: set[str]) -> int:
    score = _table_context_score(table, question)
    if table in base_tables:
        score += 6
    if role == "measure":
        score += 3
    if role == "dimension" and any(w in _norm(question) for w in ("by", "per", "each", "for each")):
        score += 2
    if _table_bare(table).startswith("DIM_"):
        score += 1
    if column == "DIVI" and "division" in _norm(question):
        score += 3
    return score


def _contains_alias(alias: str, question_norm: str, question_compact: str) -> bool:
    alias_norm = _norm(alias)
    if not alias_norm:
        return False
    alias_forms = {alias_norm}
    if not alias_norm.endswith("s"):
        alias_forms.add(alias_norm + "s")
        if alias_norm.endswith("y"):
            alias_forms.add(alias_norm[:-1] + "ies")
        if alias_norm.endswith(("x", "z", "ch", "sh")):
            alias_forms.add(alias_norm + "es")
    if alias_norm.endswith("se"):
        alias_forms.add(alias_norm[:-1] + "es")
    if alias_norm == "warehouse":
        alias_forms.add("warehouses")
    for form in alias_forms:
        if re.search(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])", question_norm):
            return True
    alias_compact = _compact(alias_norm)
    # Compact matching is only for long technical forms such as ITMGRPDMSKEY.
    # Short terms must not match inside larger words, e.g. AGE in percentAGE.
    return len(alias_compact) >= 6 and alias_compact in question_compact


def _column_matches_question(column: str, aliases: set[str], question_norm: str, question_compact: str) -> tuple[bool, str]:
    if column == "ITM_DMS_KEY" and "item group" in question_norm:
        return False, ""
    for alias in sorted(aliases, key=len, reverse=True):
        if not alias:
            continue
        if _contains_alias(alias, question_norm, question_compact):
            return True, alias
    return False, ""


def _find_candidates(
    question: str,
    table_columns: dict[str, dict[str, str]],
    allowed_tables: set[str] | None,
    selected_schema: str = "",
) -> list[dict]:
    qn = _norm(question)
    qc = _compact(question)
    selected_schema = (selected_schema or "").upper().strip()
    allowed_expanded: set[str] = set()
    for table in allowed_tables or set():
        allowed_expanded.update(_table_variants(str(table)))
    candidates: list[dict] = []
    for table, cols in table_columns.items():
        table_u = str(table).upper()
        if selected_schema:
            schema_name = _table_schema(table_u)
            if schema_name and schema_name != selected_schema:
                continue
        if allowed_tables is not None:
            variants = _table_variants(table_u)
            if not variants & allowed_expanded:
                continue
        for col, col_type in (cols or {}).items():
            col_u = str(col).upper()
            aliases = _aliases_for_column(col_u)
            matched, term = _column_matches_question(col_u, aliases, qn, qc)
            if not matched:
                continue
            candidates.append({
                "term": term,
                "table": table_u,
                "column": col_u,
                "role": _role_for_column(col_u, str(col_type)),
                "aliases": sorted(aliases),
            })
    return candidates


def _choose_fields(question: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    base_tables = {
        c["table"]
        for c in candidates
        if c["role"] == "measure" and any(h in c["term"] for h in _MEASURE_HINTS)
    }
    chosen_by_term: dict[str, dict] = {}
    for c in candidates:
        key = c["term"]
        score = _score_candidate(c["table"], c["column"], c["role"], question, base_tables)
        current = chosen_by_term.get(key)
        if not current or score > current["_score"]:
            c = dict(c)
            c["_score"] = score
            chosen_by_term[key] = c
    fields = list(chosen_by_term.values())
    fields.sort(key=lambda f: (f["role"] != "measure", f["term"], f["table"], f["column"]))
    for f in fields:
        f.pop("_score", None)
    return fields[:8]


def _question_asks_for_key(term: str, question: str) -> bool:
    q = _norm(question)
    term_norm = _norm(term)
    if not term_norm:
        return False
    key_words = ("key", "id", "identifier", "number")
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(term_norm)}\s+{word}(?![a-z0-9])", q)
        or re.search(rf"(?<![a-z0-9]){word}\s+{re.escape(term_norm)}(?![a-z0-9])", q)
        for word in key_words
    )


def _question_asks_for_code(term: str, question: str) -> bool:
    q = _norm(question)
    term_norm = _norm(term)
    if not term_norm:
        return False
    return bool(
        re.search(rf"(?<![a-z0-9]){re.escape(term_norm)}\s+code(?![a-z0-9])", q)
        or re.search(rf"(?<![a-z0-9])code\s+{re.escape(term_norm)}(?![a-z0-9])", q)
    )


def _key_prefix(column: str) -> str:
    col = (column or "").upper()
    if col.endswith("_DMS_KEY"):
        return col[:-8]
    if col.endswith("_KEY"):
        return col[:-4]
    return ""


def _table_allowed_for_display(table: str, allowed_tables: set[str] | None, selected_schema: str = "") -> bool:
    table_u = (table or "").upper()
    if selected_schema:
        schema_name = _table_schema(table_u)
        if schema_name and schema_name != selected_schema.upper():
            return False
    if allowed_tables is None:
        return True
    allowed_expanded: set[str] = set()
    for allowed in allowed_tables:
        allowed_expanded.update(_table_variants(str(allowed)))
    return bool(_table_variants(table_u) & allowed_expanded)


def _display_table_score(table: str, key_prefix: str) -> int:
    bare = _table_bare(table)
    score = 0
    if bare == f"{key_prefix}_DMS":
        score += 12
    if bare.startswith("DIM_"):
        score += 8
    if "FCT" not in bare and "FACT" not in bare:
        score += 4
    if key_prefix and key_prefix in bare:
        score += 3
    return score


def _find_display_field_for_key(
    key_column: str,
    term: str,
    question: str,
    table_columns: dict[str, dict[str, str]],
    allowed_tables: set[str] | None,
    selected_schema: str = "",
) -> dict | None:
    key_col = (key_column or "").upper()
    prefix = _key_prefix(key_col)
    if not prefix or not key_col.endswith("_KEY"):
        return None
    if _question_asks_for_key(term, question):
        return None

    wants_code = _question_asks_for_code(term, question)
    display_candidates = [f"{prefix}_DSC", f"{prefix}_DESC", f"{prefix}_DESCRIPTION", f"{prefix}_NAME", f"{prefix}_NM"]
    code_candidates = [f"{prefix}_CD", f"{prefix}_CODE"]
    preferred = code_candidates + display_candidates if wants_code else display_candidates + code_candidates

    matches: list[dict] = []
    for table, cols in table_columns.items():
        cols_u = {str(c).upper() for c in (cols or {})}
        if key_col not in cols_u:
            continue
        if not _table_allowed_for_display(table, allowed_tables, selected_schema):
            continue
        for idx, display_col in enumerate(preferred):
            if display_col not in cols_u:
                continue
            matches.append({
                "table": table,
                "column": display_col,
                "source_key_column": key_col,
                "_score": _display_table_score(table, prefix) + (100 - idx),
            })
            break

    if not matches:
        return None
    matches.sort(key=lambda m: m["_score"], reverse=True)
    winner = dict(matches[0])
    winner.pop("_score", None)
    return winner


def _apply_display_dimension_fields(
    fields: list[dict],
    question: str,
    table_columns: dict[str, dict[str, str]],
    allowed_tables: set[str] | None,
    selected_schema: str = "",
) -> list[dict]:
    out: list[dict] = []
    for field in fields:
        col = (field.get("column") or "").upper()
        if field.get("role") == "dimension" and col.endswith("_DMS_KEY"):
            display = _find_display_field_for_key(
                col,
                field.get("term") or "",
                question,
                table_columns,
                allowed_tables,
                selected_schema,
            )
            if display:
                upgraded = dict(field)
                upgraded.update({
                    "table": display["table"],
                    "column": display["column"],
                    "role": "display_dimension",
                    "source_key_column": display["source_key_column"],
                    "source_key_table": field.get("table", ""),
                    "display_required": True,
                })
                out.append(upgraded)
                continue
        out.append(field)
    return out


def _join_edges(table_columns: dict[str, dict[str, str]]) -> dict[str, list[dict]]:
    tables = {str(t).upper(): {str(c).upper() for c in (cols or {})} for t, cols in table_columns.items()}
    graph: dict[str, list[dict]] = {t: [] for t in tables}
    table_list = list(tables)
    for i, left in enumerate(table_list):
        for right in table_list[i + 1:]:
            conditions: list[tuple[str, str]] = []
            common = sorted(tables[left] & tables[right])
            # DIVI is a grouping/filter dimension, not a relational key — exclude it
            # from join conditions so it doesn't create false graph edges.
            conditions.extend((c, c) for c in common if c.endswith("_DMS_KEY") or c in {"CONO", "ORNO", "PONR", "POSX", "DLIX"})
            for lcol, rcols in _JOIN_SYNONYMS.items():
                if lcol in tables[left]:
                    conditions.extend((lcol, rc) for rc in rcols if rc in tables[right])
                if lcol in tables[right]:
                    conditions.extend((rc, lcol) for rc in rcols if rc in tables[left])
            seen = set()
            deduped = []
            for cond in conditions:
                if cond not in seen:
                    seen.add(cond)
                    deduped.append(cond)
            if deduped:
                graph[left].append({"to": right, "conditions": deduped[:5]})
                graph[right].append({"to": left, "conditions": [(r, l) for l, r in deduped[:5]]})
    return graph


def _shortest_join_path(source: str, target: str, graph: dict[str, list[dict]]) -> list[dict]:
    if source == target:
        return []
    queue = deque([(source, [])])
    seen = {source}
    while queue:
        table, path = queue.popleft()
        for edge in graph.get(table, []):
            nxt = edge["to"]
            if nxt in seen:
                continue
            next_path = path + [{"from": table, "to": nxt, "conditions": edge["conditions"]}]
            if nxt == target:
                return next_path
            seen.add(nxt)
            queue.append((nxt, next_path))
    return []


def _build_required_joins(fields: list[dict], table_columns: dict[str, dict[str, str]]) -> list[dict]:
    tables = []
    for f in fields:
        if f["table"] not in tables:
            tables.append(f["table"])
    if len(tables) <= 1:
        return []
    graph = _join_edges(table_columns)
    anchor = next((f["table"] for f in fields if f["role"] == "measure"), tables[0])
    joins: list[dict] = []
    seen_edges: set[tuple[str, str]] = set()
    for table in tables:
        if table == anchor:
            continue
        for edge in _shortest_join_path(anchor, table, graph):
            key = tuple(sorted([edge["from"], edge["to"]]))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            joins.append(edge)
    return joins


def build_semantic_field_plan(
    question: str,
    table_columns: dict[str, dict[str, str]] | None,
    allowed_tables: set[str] | None = None,
    selected_schema: str = "",
) -> dict:
    """Build a conservative field-source plan from exact known schema columns."""
    normalized_columns = {
        str(t).upper(): {str(c).upper(): str(v) for c, v in (cols or {}).items()}
        for t, cols in (table_columns or {}).items()
    }
    candidates = _find_candidates(question, normalized_columns, allowed_tables, selected_schema)
    fields = _choose_fields(question, candidates)
    fields = _apply_display_dimension_fields(fields, question, normalized_columns, allowed_tables, selected_schema)
    if not fields:
        return {"enabled": False, "fields": [], "joins": [], "reason": "no matching semantic fields"}
    joins = _build_required_joins(fields, normalized_columns)
    required_tables = sorted({f["table"] for f in fields})
    return {
        "enabled": True,
        "fields": fields,
        "joins": joins,
        "required_tables": required_tables,
        "reason": "schema-derived semantic field plan",
    }


def format_semantic_field_plan(plan: dict, db_type: str = "azure_sql") -> str:
    if not plan or not plan.get("enabled") or not plan.get("fields"):
        return ""
    lines = [
        "## Semantic field-source plan",
        "Use these exact source fields when the question mentions the mapped business terms.",
        "Do not move a mapped column to another table and do not remove underscores from column names.",
        "",
        "Resolved fields:",
    ]
    for field in plan.get("fields", []):
        expr = f"{field['table']}.{field['column']}"
        # Show a non-binding hint for measures — the LLM should aggregate only when
        # the query is aggregating (not for row-level queries like "show all invoices").
        role_hint = " [measure — apply SUM/COUNT only if aggregating]" if field.get("role") == "measure" else ""
        if field.get("display_required"):
            role_hint = (
                " [business display field - use this in SELECT and GROUP BY; "
                f"use {field.get('source_key_column')} only for JOINs unless the user asks for key/id]"
            )
        lines.append(f"- {field['term']}: {expr}{role_hint}")
    joins = plan.get("joins") or []
    if joins:
        lines.append("")
        lines.append("Required join path:")
        for edge in joins:
            conds = " AND ".join(
                f"{edge['from']}.{left_col} = {edge['to']}.{right_col}"
                for left_col, right_col in edge.get("conditions", [])
            )
            lines.append(f"- {edge['from']} JOIN {edge['to']} ON {conds}")
    return "\n".join(lines)
