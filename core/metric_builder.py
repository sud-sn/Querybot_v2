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


def _normalise_invalid_keys(value: Any) -> list[int]:
    """Parse sentinel key values (e.g. 0, 777) that mark 'no date' in the fact."""
    if value in (None, ""):
        return []
    if isinstance(value, str):
        parts = re.split(r"\s*,\s*", value.strip())
    elif isinstance(value, list):
        parts = value
    else:
        parts = [value]
    keys: list[int] = []
    for part in parts:
        raw = str(part).strip()
        if not raw:
            continue
        if not re.fullmatch(r"-?\d+", raw):
            raise ValueError(f"Invalid key value must be an integer, got: {raw}")
        n = int(raw)
        if n not in keys:
            keys.append(n)
    return keys


def _normalise_required_joins(value: Any) -> list[dict[str, Any]]:
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

    joins: list[dict[str, Any]] = []
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
        join: dict[str, Any] = {
            "alias": alias,
            "table": _clean_identifier(table, "Join table"),
            "from_column": _bare_column(_clean_identifier(from_column, "Join source column")),
            "to_column": _bare_column(_clean_identifier(to_column, "Join target column")),
            "role": role,
        }
        invalid_keys = _normalise_invalid_keys(item.get("invalid_keys"))
        if invalid_keys:
            join["invalid_keys"] = invalid_keys
        joins.append(join)
    return joins


_DATE_GAP_UNITS = {"day", "month"}
_DATE_GAP_MISSING_TO = {"null", "zero", "today", "today_if_overdue_else_zero"}


def _date_gap_diff_expr(a: str, b: str, unit: str, db_type: str) -> str:
    """Dialect-correct '(b - a) in <unit>' expression for real date columns."""
    if db_type == "oracle":
        if unit == "month":
            return f"MONTHS_BETWEEN({b}, {a})"
        return f"({b} - {a})"
    if db_type == "snowflake":
        return f"DATEDIFF('{unit}', {a}, {b})"
    return f"DATEDIFF({unit}, {a}, {b})"


def _date_gap_today_expr(db_type: str) -> str:
    if db_type == "oracle":
        return "TRUNC(SYSDATE)"
    if db_type == "snowflake":
        return "CURRENT_DATE"
    return "CAST(GETDATE() AS date)"


def _date_gap_join(config: dict[str, Any], key: str, label: str, invalid_keys: list[int]) -> dict[str, Any]:
    join = config.get(key)
    if not isinstance(join, dict):
        raise ValueError(f"{label} date role is required.")
    alias = (join.get("alias") or "").strip()
    table = (join.get("table") or "").strip()
    from_column = (join.get("from_column") or "").strip()
    to_column = (join.get("to_column") or "").strip()
    if not alias or not table or not from_column or not to_column:
        raise ValueError(f"{label} date role needs alias, table, from_column, and to_column.")
    _clean_identifier(alias, f"{label} join alias")
    out: dict[str, Any] = {
        "alias": alias,
        "table": _clean_identifier(table, f"{label} join table"),
        "from_column": _bare_column(_clean_identifier(from_column, f"{label} source column")),
        "to_column": _bare_column(_clean_identifier(to_column, f"{label} target column")),
        "role": (join.get("role") or "").strip(),
    }
    if invalid_keys:
        out["invalid_keys"] = list(invalid_keys)
    return out


def _compile_date_gap_metric(
    config: dict[str, Any], aggregation: str, db_type: str = "azure_sql"
) -> CompiledMetricFormula:
    """
    No-code 'gap between two date roles' wizard.

    The admin picks two role-playing date keys (e.g. Due Date → Payment Date);
    this compiles the same row-calculated shape they would otherwise hand-write:
    two dimension joins plus a CASE row expression, with sentinel keys (0, 777…)
    excluded in the join so an invalid key reads as a missing date — mirroring
    the ISBLANK/LOOKUPVALUE guard pattern of the equivalent DAX measure.
    """
    unit = (config.get("unit") or "day").strip().lower()
    if unit not in _DATE_GAP_UNITS:
        raise ValueError(f"Unsupported date-gap unit: {unit}")
    missing_to = (config.get("missing_to") or "null").strip().lower()
    if missing_to not in _DATE_GAP_MISSING_TO:
        raise ValueError(f"Unsupported missing-date behaviour: {missing_to}")
    date_column = _bare_column(
        _clean_identifier(config.get("date_column") or "", "Dimension date column")
    )

    invalid_keys = _normalise_invalid_keys(config.get("invalid_keys"))
    from_join = _date_gap_join(config, "from_join", "From", invalid_keys)
    to_join = _date_gap_join(config, "to_join", "To", invalid_keys)
    if from_join["alias"] == to_join["alias"]:
        raise ValueError("From and To date roles must use different aliases.")

    from_date = f"{from_join['alias']}.{date_column}"
    to_date = f"{to_join['alias']}.{date_column}"
    today = _date_gap_today_expr(db_type)
    gap = _date_gap_diff_expr(from_date, to_date, unit, db_type)
    gap_to_today = _date_gap_diff_expr(from_date, today, unit, db_type)

    branches = [
        f"WHEN {from_date} IS NULL THEN NULL",
        f"WHEN {to_date} IS NOT NULL THEN {gap}",
    ]
    if missing_to == "zero":
        branches.append("ELSE 0")
    elif missing_to == "today":
        branches.append(f"ELSE {gap_to_today}")
    elif missing_to == "today_if_overdue_else_zero":
        branches.append(f"WHEN {from_date} < {today} THEN {gap_to_today}")
        branches.append("ELSE 0")
    else:  # "null" — CASE falls through to NULL naturally
        branches.append("ELSE NULL")
    row_expression = "CASE " + " ".join(branches) + " END"

    required_columns = list(dict.fromkeys(
        [from_join["from_column"], to_join["from_column"]]
    ))
    clean_config = {
        "enabled": True,
        "mode": "date_gap",
        "aggregation": aggregation,
        "unit": unit,
        "missing_to": missing_to,
        "date_column": date_column,
        "invalid_keys": invalid_keys,
        "from_join": from_join,
        "to_join": to_join,
        # Compiled row_calculated equivalents — every downstream consumer
        # (prompt injection, join skeleton, base-table inference) reads these
        # same fields, so date_gap metrics need no special handling there.
        "row_expression": row_expression,
        "required_columns": required_columns,
        "required_joins": [from_join, to_join],
    }
    if aggregation == "COUNT":
        formula = f"COUNT(CASE WHEN ({row_expression}) IS NOT NULL THEN 1 END)"
    else:
        formula = f"{aggregation}(CAST(({row_expression}) AS float))"
    return CompiledMetricFormula(
        formula=formula,
        required_columns=required_columns,
        config_json=json.dumps(clean_config, separators=(",", ":")),
    )


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


def compile_metric_builder_config(
    raw_config: str | dict[str, Any], db_type: str = "azure_sql"
) -> CompiledMetricFormula | None:
    config = _normalise_config(raw_config)
    if not config or not config.get("enabled"):
        return None

    aggregation = (config.get("aggregation") or "SUM").strip().upper()
    if aggregation not in _AGGREGATIONS:
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    mode = (config.get("mode") or "aggregate").strip().lower()
    if mode == "date_gap":
        return _compile_date_gap_metric(config, aggregation, db_type)
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
