from __future__ import annotations

import re
from typing import Any


try:
    import sqlglot
    from sqlglot import exp as sg_exp
except Exception:  # pragma: no cover - exercised only when sqlglot is absent
    sqlglot = None
    sg_exp = None


_DIALECT = {
    "azure_sql": "tsql",
    "snowflake": "snowflake",
    "oracle": "oracle",
}


def _clean_part(value: str) -> str:
    return str(value or "").strip().strip("[]").strip('"').strip("`")


def extract_sql_tables(sql: str, db_type: str = "azure_sql") -> list[str]:
    """Return table references used by SQL, excluding CTE names where possible."""
    if not sql:
        return []

    tables: list[str] = []
    seen: set[str] = set()

    if sqlglot is not None and sg_exp is not None:
        dialect = _DIALECT.get(db_type, "tsql")
        tree = None
        for candidate in (dialect, None):
            try:
                tree = sqlglot.parse_one(sql, dialect=candidate)
                break
            except Exception:
                continue
        if tree is not None:
            ctes = {
                str(cte.alias or "").upper()
                for cte in tree.find_all(sg_exp.CTE)
                if cte.alias
            }
            for node in tree.find_all(sg_exp.Table):
                name = _clean_part(getattr(node, "name", "") or "")
                if not name or name.upper() in ctes:
                    continue
                parts = []
                catalog = _clean_part(getattr(node, "catalog", "") or "")
                db = _clean_part(getattr(node, "db", "") or "")
                if catalog:
                    parts.append(catalog)
                if db:
                    parts.append(db)
                parts.append(name)
                ref = ".".join(p for p in parts if p).upper()
                if ref and ref not in seen:
                    seen.add(ref)
                    tables.append(ref)
            if tables:
                return tables

    pattern = re.compile(
        r"\b(?:FROM|JOIN)\s+((?:\[[^\]]+\]|\"[^\"]+\"|`[^`]+`|[A-Za-z_][\w$]*)(?:\s*\.\s*(?:\[[^\]]+\]|\"[^\"]+\"|`[^`]+`|[A-Za-z_][\w$]*)){0,2})",
        re.IGNORECASE,
    )
    for match in pattern.finditer(sql):
        ref = ".".join(_clean_part(p) for p in re.split(r"\s*\.\s*", match.group(1)) if _clean_part(p)).upper()
        if ref and ref not in seen:
            seen.add(ref)
            tables.append(ref)
    return tables


def build_business_rca(
    *,
    question: str = "",
    row_count: int | None = None,
    tables_used: list[str] | None = None,
    empty_tables: list[str] | None = None,
    validation_code: str = "ok",
    retry_count: int = 0,
    graph_context: dict | None = None,
    semantic_plan: dict | None = None,
    unmatched_literals: list[dict] | None = None,
) -> dict[str, Any]:
    tables = [str(t) for t in (tables_used or []) if str(t).strip()]
    empty = [str(t) for t in (empty_tables or []) if str(t).strip()]
    validation = (validation_code or "ok").lower()
    graph = graph_context or {}
    plan = semantic_plan or {}

    technical_notes = [
        f"SQL validation: {validation_code or 'ok'}",
        f"Row count: {0 if row_count is None else row_count}",
        f"Retry count: {retry_count}",
    ]
    if tables:
        technical_notes.append("Tables used: " + ", ".join(tables[:6]))

    if validation not in {"ok", "pass", "trusted_metric"}:
        return {
            "headline": "I could not produce a trusted answer for this question.",
            "most_likely_reason": "The generated SQL did not pass the validation checks.",
            "suggested_next_step": "Ask an administrator to review the field mapping or rephrase the question with a specific metric and table.",
            "technical_notes": technical_notes,
        }

    # A filter value that matches nothing in the actual data is the most
    # specific zero-row explanation available — takes precedence over the
    # generic empty-table / join-path reasons.
    if row_count == 0 and unmatched_literals:
        first = unmatched_literals[0]
        label = first.get("business_name") or first.get("column") or "value"
        reason = f"There is no {label} matching '{first.get('literal')}' in the data."
        closest = [c for c in (first.get("closest") or []) if c]
        if closest:
            listed = ", ".join(f"'{c}'" for c in closest[:3])
            next_step = f"Closest values in your data: {listed} — try one of these."
        else:
            next_step = "Check the spelling, or ask without the filter to see the available values."
        technical_notes.append(
            f"Unmatched filter literal: {first.get('column')} = '{first.get('literal')}'"
        )
        return {
            "headline": "I could not find matching records for this question.",
            "most_likely_reason": reason,
            "suggested_next_step": next_step,
            "technical_notes": technical_notes,
        }

    if row_count == 0 and empty:
        listed = ", ".join(empty[:3])
        return {
            "headline": "I could not find matching records for this question.",
            "most_likely_reason": f"One of the tables needed for this answer has no records: {listed}.",
            "suggested_next_step": "Check whether that source table should contain data for the selected schema, or map the business term to another populated table.",
            "technical_notes": technical_notes,
        }

    if row_count == 0 and (graph.get("enabled") or graph.get("detected")):
        return {
            "headline": "I could not find matching records for this question.",
            "most_likely_reason": "The selected join path did not produce matching records for the current data.",
            "suggested_next_step": "Check whether the relationship keys match in the database, or choose a less restrictive join path.",
            "technical_notes": technical_notes,
        }

    if row_count == 0:
        if plan.get("enabled"):
            reason = "The mapped fields were valid, but the filters, joins, or selected schema produced no matching rows."
        else:
            reason = "The filters, joins, or selected schema produced no matching rows."
        return {
            "headline": "I could not find matching records for this question.",
            "most_likely_reason": reason,
            "suggested_next_step": "Try broadening the filter, checking the selected schema, or confirming the business field mapping.",
            "technical_notes": technical_notes,
        }

    return {
        "headline": "Here are the results I found.",
        "most_likely_reason": "The query completed successfully and returned data.",
        "suggested_next_step": "Use the query details if you want to audit the fields and tables behind the answer.",
        "technical_notes": technical_notes,
    }
