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
}

_DIRECT_ALIASES = {
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


def _column_matches_question(column: str, aliases: set[str], question_norm: str, question_compact: str) -> tuple[bool, str]:
    if column == "ITM_DMS_KEY" and "item group" in question_norm:
        return False, ""
    for alias in sorted(aliases, key=len, reverse=True):
        if not alias:
            continue
        if alias in question_norm or _compact(alias) in question_compact:
            return True, alias
    return False, ""


def _find_candidates(question: str, table_columns: dict[str, dict[str, str]], allowed_tables: set[str] | None) -> list[dict]:
    qn = _norm(question)
    qc = _compact(question)
    allowed_expanded = {str(t).upper() for t in allowed_tables or set()}
    candidates: list[dict] = []
    for table, cols in table_columns.items():
        table_u = str(table).upper()
        if allowed_tables is not None:
            variants = {table_u, _table_bare(table_u)}
            parts = table_u.split(".")
            if len(parts) >= 2:
                variants.add(".".join(parts[-2:]))
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


def _join_edges(table_columns: dict[str, dict[str, str]]) -> dict[str, list[dict]]:
    tables = {str(t).upper(): {str(c).upper() for c in (cols or {})} for t, cols in table_columns.items()}
    graph: dict[str, list[dict]] = {t: [] for t in tables}
    table_list = list(tables)
    for i, left in enumerate(table_list):
        for right in table_list[i + 1:]:
            conditions: list[tuple[str, str]] = []
            common = sorted(tables[left] & tables[right])
            conditions.extend((c, c) for c in common if c.endswith("_DMS_KEY") or c in {"CONO", "DIVI", "ORNO", "PONR", "POSX", "DLIX"})
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
) -> dict:
    """Build a conservative field-source plan from exact known schema columns."""
    normalized_columns = {
        str(t).upper(): {str(c).upper(): str(v) for c, v in (cols or {}).items()}
        for t, cols in (table_columns or {}).items()
    }
    candidates = _find_candidates(question, normalized_columns, allowed_tables)
    fields = _choose_fields(question, candidates)
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
        agg = "SUM" if field.get("role") == "measure" else ""
        expr = f"{agg}({field['table']}.{field['column']})" if agg else f"{field['table']}.{field['column']}"
        lines.append(f"- {field['term']}: {expr}")
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
