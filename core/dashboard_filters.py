"""Safe SQL-AST compilation for dashboard controls.

Filters are applied to the output of the saved governed query, not spliced
into its text. Values become sqlglot literals and fields must match an output
alias already present in the saved SELECT. The resulting SQL is still passed
through QueryBot's normal validator, row-policy injector, and compliance gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_DIALECT = {"azure_sql": "tsql", "snowflake": "snowflake", "oracle": "oracle"}
_OPERATORS = {"equals", "contains", "gte", "lte"}


@dataclass(frozen=True)
class DashboardFilterResult:
    sql: str
    applied: tuple[str, ...] = ()
    ignored: tuple[str, ...] = ()


def compile_dashboard_filters(
    sql: str,
    db_type: str,
    filters: list[dict],
    values: dict[str, str],
) -> DashboardFilterResult:
    active = [
        item for item in list(filters or [])
        if isinstance(item, dict) and str(values.get(str(item.get("field") or "")) or "").strip()
    ]
    if not active:
        return DashboardFilterResult(str(sql or ""))

    try:
        import sqlglot
        from sqlglot import exp
    except ImportError:
        return DashboardFilterResult(str(sql or ""), ignored=tuple(str(item.get("field") or "") for item in active))

    dialect = _DIALECT.get(str(db_type or "").lower(), "snowflake")
    try:
        tree = sqlglot.parse_one(str(sql or "").strip().rstrip(";"), read=dialect)
    except Exception:
        return DashboardFilterResult(str(sql or ""), ignored=tuple(str(item.get("field") or "") for item in active))
    if not isinstance(tree, exp.Select):
        return DashboardFilterResult(str(sql or ""), ignored=tuple(str(item.get("field") or "") for item in active))

    outputs = {
        _normalise(name): name
        for name in list(getattr(tree, "named_selects", None) or [])
        if name and name != "*"
    }
    wrapper = exp.select("*").from_(tree.subquery("qb_dashboard_source"))
    predicate = None
    applied: list[str] = []
    ignored: list[str] = []

    for item in active:
        requested = str(item.get("field") or "").strip()
        output = outputs.get(_normalise(requested))
        if not output:
            ignored.append(requested)
            continue
        value = str(values.get(requested) or "").strip()[:500]
        operator = str(item.get("operator") or "equals").lower()
        operator = operator if operator in _OPERATORS else "equals"
        column = exp.Column(
            this=exp.Identifier(this=output, quoted=True),
            table=exp.Identifier(this="qb_dashboard_source", quoted=False),
        )
        literal = _literal(exp, value, str(item.get("type") or "text"))
        if operator == "contains":
            condition = exp.Like(this=column, expression=exp.Literal.string(f"%{value}%"))
        elif operator == "gte":
            condition = exp.GTE(this=column, expression=literal)
        elif operator == "lte":
            condition = exp.LTE(this=column, expression=literal)
        else:
            condition = exp.EQ(this=column, expression=literal)
        predicate = condition if predicate is None else exp.and_(predicate, condition)
        applied.append(output)

    if predicate is None:
        return DashboardFilterResult(str(sql or ""), ignored=tuple(ignored))
    wrapper = wrapper.where(predicate)
    return DashboardFilterResult(
        wrapper.sql(dialect=dialect),
        applied=tuple(applied),
        ignored=tuple(ignored),
    )


def _literal(exp, value: str, filter_type: str):
    if str(filter_type or "").lower() == "number" and re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        return exp.Literal.number(value)
    return exp.Literal.string(value)


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
