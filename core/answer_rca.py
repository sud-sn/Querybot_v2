from __future__ import annotations

import re
from typing import Any


def _t(msg_id: str, **kw) -> str:
    """Resolve a catalogue id in the reader's language.

    Deferred import, matching core/failure_messages.py: the language comes from
    the request's ContextVar, which _handle_query_impl activates around the
    whole turn. The zero-row card is built inline in that coroutine -- the only
    executor hop nearby wraps the table-count query, not this -- so the
    ContextVar is live and no lang needs threading through.
    """
    from core.i18n import t

    return t(msg_id, **kw)


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
            "headline": _t("fail.zero_row.validation.headline"),
            "most_likely_reason": _t("fail.zero_row.validation.reason"),
            "suggested_next_step": _t("fail.zero_row.validation.next_step"),
            "technical_notes": technical_notes,
        }

    # A filter value that matches nothing in the actual data is the most
    # specific zero-row explanation available — takes precedence over the
    # generic empty-table / join-path reasons.
    if row_count == 0 and unmatched_literals:
        first = unmatched_literals[0]
        # The business name is tenant DATA, so it is interpolated, never
        # translated. A separate id for the unnamed case rather than an English
        # word "value" glued into a French sentence.
        label = first.get("business_name") or first.get("column") or ""
        if label:
            reason = _t("fail.zero_row.unmatched.reason",
                        label=label, literal=first.get("literal"))
        else:
            reason = _t("fail.zero_row.unmatched.reason_unnamed",
                        literal=first.get("literal"))
        closest = [c for c in (first.get("closest") or []) if c]
        if closest:
            listed = ", ".join(f"'{c}'" for c in closest[:3])
            next_step = _t("fail.zero_row.unmatched.next_step_closest",
                           values=listed)
        else:
            next_step = _t("fail.zero_row.unmatched.next_step")
        technical_notes.append(
            f"Unmatched filter literal: {first.get('column')} = '{first.get('literal')}'"
        )
        return {
            "headline": _t("fail.zero_row.headline"),
            "most_likely_reason": reason,
            "suggested_next_step": next_step,
            "technical_notes": technical_notes,
        }

    if row_count == 0 and empty:
        listed = ", ".join(empty[:3])
        return {
            "headline": _t("fail.zero_row.headline"),
            "most_likely_reason": _t("fail.zero_row.empty_table.reason",
                                     tables=listed),
            "suggested_next_step": _t("fail.zero_row.empty_table.next_step"),
            "technical_notes": technical_notes,
        }

    # A join-path explanation is only honest when the SQL actually joins
    # more than one table. graph_context["enabled"]/["detected"] are true
    # for a single-table resolution too (e.g. entities=["F_RX_FILL"],
    # edge_ids=[]) -- blaming "relationship keys"/"join path" there is
    # flatly wrong: there is no join, the query is just a WHERE clause that
    # legitimately matched nothing.
    if row_count == 0 and (graph.get("enabled") or graph.get("detected")) and len(tables) > 1:
        return {
            "headline": _t("fail.zero_row.headline"),
            "most_likely_reason": _t("fail.zero_row.join.reason"),
            "suggested_next_step": _t("fail.zero_row.join.next_step"),
            "technical_notes": technical_notes,
        }

    if row_count == 0:
        # Only mention joins when the SQL genuinely has more than one table --
        # otherwise this is just a single-table WHERE clause that correctly
        # matched nothing, and saying "joins" implies a problem that isn't
        # there.
        # Four whole sentences rather than one with a clause spliced into it.
        # "The {clause} produced no matching rows" cannot be translated: French
        # word order and agreement do not survive an English fragment dropped
        # into the middle of the sentence.
        stem = "filters_joins" if len(tables) > 1 else "filters"
        suffix = "reason_mapped" if plan.get("enabled") else "reason"
        reason = _t(f"fail.zero_row.{stem}.{suffix}")
        return {
            "headline": _t("fail.zero_row.headline"),
            "most_likely_reason": reason,
            "suggested_next_step": _t("fail.zero_row.broaden.next_step"),
            "technical_notes": technical_notes,
        }

    return {
        "headline": _t("rca.success.headline"),
        "most_likely_reason": _t("rca.success.reason"),
        "suggested_next_step": _t("rca.success.next_step"),
        "technical_notes": technical_notes,
    }
