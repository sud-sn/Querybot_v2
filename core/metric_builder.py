"""
No-code metric formula builder.

Compiles structured metric-builder configuration into the expression syntax the
existing metric registry already understands.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


_IDENT = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*$")
_AGGREGATIONS = {"SUM", "AVG", "COUNT", "MIN", "MAX"}
_ROW_EXPR_BLOCKLIST = re.compile(
    r"(--|/\*|\*/|;|\b(?:SELECT|FROM|JOIN|WITH|UNION|INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|CREATE|TRUNCATE|EXEC|EXECUTE|CALL)\b)",
    re.IGNORECASE,
)
_ROW_EXPR_AGGREGATE = re.compile(r"\b(?:SUM|AVG|COUNT|MIN|MAX)\s*\(", re.IGNORECASE)


@dataclass
class CompiledMetricFormula:
    formula: str
    required_columns: list[str]
    config_json: str


def _clean_identifier(value: str, label: str) -> str:
    ident = (value or "").strip()
    if not ident:
        raise ValueError(f"{label} is required.")
    if not _IDENT.match(ident):
        raise ValueError(f"{label} must be a schema column name, not SQL text.")
    return ident


def _bare_column(value: str) -> str:
    return (value or "").strip().split(".")[-1]


def _is_number(value: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", (value or "").strip()))


def _literal(value: str) -> str:
    raw = (value or "").strip()
    if _is_number(raw):
        return raw
    return "'" + raw.replace("'", "''") + "'"


def _split_values(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s*,\s*", value or "") if part.strip()]


def _split_columns(value: Any) -> list[str]:
    if isinstance(value, list):
        parts = value
    else:
        parts = re.split(r"\s*,\s*", str(value or ""))
    columns: list[str] = []
    for part in parts:
        raw = str(part or "").strip()
        if not raw:
            continue
        columns.append(_bare_column(_clean_identifier(raw, "Required column")))
    return columns


def _normalise_required_joins(value: Any) -> list[dict[str, str]]:
    """
    Store row-calculated metric join hints as metadata for prompt injection.

    These are not executed by the compiler. They tell the SQL generator how to
    expose columns referenced by the row expression, for example joining the
    same date dimension twice as due_dt and pay_dt.
    """
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Required joins must be a JSON array.") from exc
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError("Required joins must be a list.")

    joins: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        alias = (item.get("alias") or "").strip()
        table = (item.get("table") or item.get("to_table") or "").strip()
        from_column = (item.get("from_column") or "").strip()
        to_column = (item.get("to_column") or "").strip()
        role = (item.get("role") or "").strip()
        if not table or not from_column or not to_column:
            continue
        if alias:
            _clean_identifier(alias, "Join alias")
        joins.append({
            "alias": alias,
            "table": _clean_identifier(table, "Join table"),
            "from_column": _bare_column(_clean_identifier(from_column, "Join source column")),
            "to_column": _bare_column(_clean_identifier(to_column, "Join target column")),
            "role": role,
        })
    return joins


def _compile_row_calculated_metric(config: dict[str, Any], aggregation: str) -> CompiledMetricFormula:
    row_expression = (config.get("row_expression") or "").strip()
    if not row_expression:
        raise ValueError("Row-level expression is required.")
    if _ROW_EXPR_BLOCKLIST.search(row_expression):
        raise ValueError("Row-level expression must be a single safe expression, not a query or script.")
    if _ROW_EXPR_AGGREGATE.search(row_expression):
        raise ValueError("Row-level expression cannot contain aggregate functions; choose the aggregation separately.")

    required_columns = _split_columns(config.get("required_columns") or "")
    required_joins = _normalise_required_joins(config.get("required_joins") or [])
    for join in required_joins:
        if join.get("from_column"):
            required_columns.append(join["from_column"])

    if aggregation == "COUNT":
        formula = f"COUNT(CASE WHEN ({row_expression}) IS NOT NULL THEN 1 END)"
    else:
        formula = f"{aggregation}(CAST(({row_expression}) AS float))"

    clean_config = {
        "enabled": True,
        "mode": "row_calculated",
        "aggregation": aggregation,
        "row_expression": row_expression,
        "required_columns": list(dict.fromkeys(col for col in required_columns if col)),
        "required_joins": required_joins,
    }
    return CompiledMetricFormula(
        formula=formula,
        required_columns=clean_config["required_columns"],
        config_json=json.dumps(clean_config, separators=(",", ":")),
    )


def _compile_condition(field: str, operator: str, value: str) -> str:
    field = _clean_identifier(field, "Filter field")
    op = (operator or "equals").strip().lower()
    raw_value = (value or "").strip()

    if op == "equals":
        return f"{field} = {_literal(raw_value)}"
    if op == "not_equals":
        return f"{field} <> {_literal(raw_value)}"
    if op == "greater_than":
        return f"{field} > {_literal(raw_value)}"
    if op == "greater_or_equal":
        return f"{field} >= {_literal(raw_value)}"
    if op == "less_than":
        return f"{field} < {_literal(raw_value)}"
    if op == "less_or_equal":
        return f"{field} <= {_literal(raw_value)}"
    if op == "contains":
        # Embed wildcards inside the literal so the pattern works across all
        # dialects (Azure SQL, Snowflake, Oracle, DuckDB) without string concat.
        escaped = raw_value.replace("'", "''")
        return f"{field} LIKE '%{escaped}%'"
    if op == "in":
        values = _split_values(raw_value)
        if not values:
            raise ValueError("IN filter requires at least one value.")
        return f"{field} IN ({', '.join(_literal(v) for v in values)})"
    if op == "not_in":
        values = _split_values(raw_value)
        if not values:
            raise ValueError("NOT IN filter requires at least one value.")
        return f"{field} NOT IN ({', '.join(_literal(v) for v in values)})"
    if op == "between":
        # Strip thousands-separator commas from numeric-looking tokens before
        # splitting so "1,000, 5,000" yields ["1000", "5000"] not three tokens.
        normalised = re.sub(r"(?<=\d),(?=\d{3}(?:[^,\d]|$))", "", raw_value)
        values = _split_values(normalised)
        if len(values) != 2:
            raise ValueError("BETWEEN filter requires two comma-separated values (e.g. 100, 500).")
        return f"{field} BETWEEN {_literal(values[0])} AND {_literal(values[1])}"
    if op == "is_null":
        return f"{field} IS NULL"
    if op == "is_not_null":
        return f"{field} IS NOT NULL"
    raise ValueError(f"Unsupported filter operator: {operator}")


def _normalise_config(raw_config: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_config, str):
        if not raw_config.strip():
            return {}
        data = json.loads(raw_config)
    else:
        data = dict(raw_config or {})
    if not isinstance(data, dict):
        raise ValueError("Metric builder config must be an object.")
    return data


def compile_metric_builder_config(raw_config: str | dict[str, Any]) -> CompiledMetricFormula | None:
    config = _normalise_config(raw_config)
    if not config or not config.get("enabled"):
        return None

    aggregation = (config.get("aggregation") or "SUM").strip().upper()
    if aggregation not in _AGGREGATIONS:
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    mode = (config.get("mode") or "aggregate").strip().lower()
    if mode in {"row", "row_calculated", "row-level", "row_level"}:
        return _compile_row_calculated_metric(config, aggregation)
    if mode not in {"aggregate", "filtered_aggregate", "measure"}:
        raise ValueError(f"Unsupported metric builder mode: {mode}")

    measure = _clean_identifier(config.get("measure") or "", "Measure field")
    filters = config.get("filters") or []
    if not isinstance(filters, list):
        raise ValueError("Metric filters must be a list.")

    conditions: list[str] = []
    required_columns = [_bare_column(measure)]
    normalised_filters: list[dict[str, str]] = []
    for filt in filters:
        if not isinstance(filt, dict):
            continue
        field = (filt.get("field") or "").strip()
        operator = (filt.get("operator") or "equals").strip().lower()
        value = (filt.get("value") or "").strip()
        if not field:
            continue
        condition = _compile_condition(field, operator, value)
        conditions.append(condition)
        required_columns.append(_bare_column(field))
        normalised_filters.append({"field": field, "operator": operator, "value": value})

    predicate = " AND ".join(conditions)
    if not predicate:
        formula = f"{aggregation}({measure})"
    elif aggregation == "SUM":
        formula = f"SUM(CASE WHEN {predicate} THEN {measure} ELSE 0 END)"
    elif aggregation == "COUNT":
        formula = f"COUNT(CASE WHEN {predicate} THEN 1 END)"
    else:
        formula = f"{aggregation}(CASE WHEN {predicate} THEN {measure} END)"

    clean_config = {
        "enabled": True,
        "mode": "aggregate",
        "aggregation": aggregation,
        "measure": measure,
        "filters": normalised_filters,
    }
    deduped_required = list(dict.fromkeys(col for col in required_columns if col))
    return CompiledMetricFormula(
        formula=formula,
        required_columns=deduped_required,
        config_json=json.dumps(clean_config, separators=(",", ":")),
    )


def merge_required_columns(existing: str, required: list[str]) -> str:
    columns = [part.strip() for part in (existing or "").split(",") if part.strip()]
    for col in required or []:
        if col and col not in columns:
            columns.append(col)
    return ", ".join(columns)
