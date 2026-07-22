"""Deterministic conversational transforms over a cached query result.

This module intentionally has no LLM or database dependency. It parses a
small, explicit command language and executes parameterised DuckDB SQL over a
session-scoped result snapshot. Raw result values never appear in generated
SQL, logs, snapshot metadata, or model prompts.
"""

from __future__ import annotations

import re
from calendar import month_abbr, month_name
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

from core.result_cache import ResultCache, result_cache


_UNDO_RE = re.compile(
    r"^\s*(?:undo(?:\s+(?:(?:that|this)(?:\s+change)?|last(?:\s+change)?))?|go\s+back|"
    r"restore\s+(?:the\s+)?(?:previous|original)\s+result)\s*[.!]?\s*$",
    re.IGNORECASE,
)
_EXCLUDE_RE = re.compile(
    r"^\s*(?:exclude|remove|omit|drop)\s+(.+?)"
    r"(?:\s+from\s+(?:this|the|these|my|current|previous)\s+"
    r"(?:result|results|list|rows?))?\s*[.!]?\s*$",
    re.IGNORECASE,
)
_KEEP_TOP_RE = re.compile(
    r"^\s*(?:keep|show)\s+(?:only\s+)?(?:the\s+)?top\s+(\d{1,4})"
    r"(?:\s+(?:rows?|results?|records?))?\s*[.!]?\s*$",
    re.IGNORECASE,
)
_KEEP_TOP_BY_RE = re.compile(
    r"^\s*keep\s+(?:only\s+)?(?:the\s+)?top\s+(\d{1,4})"
    r"(?:\s+[a-z][a-z0-9 _-]*?)?\s+by\s+(.+?)\s*[.!]?\s*$",
    re.IGNORECASE,
)
_SORT_RE = re.compile(
    r"^\s*sort\s+(?:(?:this|the|these|current)\s+(?:result|results|list)\s+)?"
    r"by\s+(.+?)(?:\s+(ascending|asc|descending|desc))?\s*[.!]?\s*$",
    re.IGNORECASE,
)
_KEEP_VALUES_RE = re.compile(
    r"^\s*(?:(?:give|show)(?:\s+me)?\s+)?"
    r"(?:(?:the\s+)?(?:data|rows|results?|records?)\s+)?"
    r"(?:only\s+for|for\s+only|only|keep\s+(?:only\s+)?)\s+"
    r"(.+?)\s*[.!]?\s*$",
    re.IGNORECASE,
)
_KEEP_VALUES_POSTFIX_RE = re.compile(
    r"^\s*(?:(?:give|show)(?:\s+me)?\s+)?"
    r"(?:(?:the\s+)?(?:data|rows|results?|records?)\s+)?"
    r"(?:for\s+)?(.+?)\s+only\s*[.!]?\s*$",
    re.IGNORECASE,
)
_RESULT_CONTEXT_RE = re.compile(
    r"\b(?:this|the|current|previous|cached)\s+"
    r"(?:result|results|data|dataset|rows?|table)\b",
    re.IGNORECASE,
)
_FILTER_OPERATOR_RE = re.compile(
    r"\s+(is\s+greater\s+than|is\s+more\s+than|is\s+less\s+than|"
    r"is\s+at\s+least|is\s+at\s+most|is\s+above|is\s+below|"
    r"is\s+not|not\s+equal\s+to|does\s+not\s+equal|at\s+least|"
    r"at\s+most|greater\s+than|more\s+than|less\s+than|equal\s+to|"
    r"equals|contains|starts\s+with|ends\s+with|above|over|below|under|"
    r">=|<=|!=|<>|=|>|<|is)\s+",
    re.IGNORECASE,
)
_ROW_RE = re.compile(r"^(?:row\s+)?(\d{1,6})(?:st|nd|rd|th)?$", re.IGNORECASE)
_MASKED_MARKERS = {"redacted", "masked", "hidden", "protected", "restricted"}
_AGGREGATIONS = {
    "total": "sum",
    "sum": "sum",
    "average": "avg",
    "avg": "avg",
    "mean": "avg",
    "count": "count",
    "minimum": "min",
    "min": "min",
    "maximum": "max",
    "max": "max",
}


@dataclass(frozen=True)
class ResultCommand:
    action: Literal[
        "exclude", "undo", "keep_top", "keep_values", "sort", "filter", "aggregate",
        "contribution", "profit_percentage", "ratio",
    ]
    target_text: str = ""
    limit: int | None = None
    direction: Literal["asc", "desc"] = "desc"
    metric_text: str = ""
    dimension_text: str = ""
    aggregation: Literal["sum", "avg", "count", "min", "max"] = "sum"
    operator: str = ""
    value_text: str = ""
    numerator_text: str = ""
    denominator_text: str = ""
    fallback_allowed: bool = False


@dataclass
class ResultCommandOutcome:
    handled: bool
    ok: bool = False
    message: str = ""
    snapshot: dict = field(default_factory=dict)
    operation: str = ""
    rows_before: int = 0
    rows_after: int = 0
    affected_count: int = 0
    source_result_id: str = ""
    derived_result_id: str = ""


def parse_result_command(text: str) -> ResultCommand | None:
    """Return a command only when the wording is explicitly result-directed."""
    value = str(text or "").strip()
    if not value:
        return None
    if _UNDO_RE.fullmatch(value):
        return ResultCommand("undo")
    match = _EXCLUDE_RE.fullmatch(value)
    if match:
        target = _clean_target(match.group(1))
        return ResultCommand("exclude", target_text=target) if target else None
    match = _KEEP_TOP_RE.fullmatch(value)
    if match:
        return ResultCommand("keep_top", limit=max(1, min(int(match.group(1)), 1000)))
    match = _KEEP_TOP_BY_RE.fullmatch(value)
    if match:
        return ResultCommand(
            "keep_top",
            limit=max(1, min(int(match.group(1)), 1000)),
            metric_text=_clean_field_text(match.group(2)),
        )
    match = _SORT_RE.fullmatch(value)
    if match:
        direction = "asc" if (match.group(2) or "").lower() in {"asc", "ascending"} else "desc"
        target = _clean_target(match.group(1))
        return ResultCommand("sort", target_text=target, direction=direction) if target else None

    # Month names are resolved against the actual cached period/date values.
    # Requiring a month reference keeps this deterministic shortcut from
    # capturing unrelated source-data questions such as "show only revenue".
    for pattern in (_KEEP_VALUES_RE, _KEEP_VALUES_POSTFIX_RE):
        match = pattern.fullmatch(value)
        if match:
            target = _clean_temporal_subset_target(match.group(1))
            if target and _contains_month_reference(target):
                return ResultCommand("keep_values", target_text=target)

    # Analytical transforms are intentionally limited to explicit references
    # to the current/cached result. This prevents a new business question from
    # being answered from an incomplete prior slice by accident.
    if not _RESULT_CONTEXT_RE.search(value):
        return None

    compact = _strip_result_context(value)

    filter_match = re.search(r"\bwhere\s+(.+)$", compact, re.IGNORECASE)
    if filter_match:
        condition = filter_match.group(1).strip().rstrip(".?")
        operator_match = _FILTER_OPERATOR_RE.search(condition)
        if operator_match:
            column = condition[:operator_match.start()].strip()
            raw_operator = operator_match.group(1).strip().lower()
            raw_value = condition[operator_match.end():].strip().strip("\"'")
            if column and raw_value:
                return ResultCommand(
                    "filter",
                    target_text=column,
                    operator=_normalise_operator(raw_operator),
                    value_text=raw_value,
                )

    contribution_patterns = (
        r"(?:show|calculate|display|give\s+me|what\s+is)?\s*(?:the\s+)?"
        r"(?:percentage|percent|%)\s+(?:contribution\s+)?(?:of\s+)?(.+?)\s+by\s+(?:each\s+)?(.+)",
        r"(?:show|calculate|display|give\s+me|what\s+is)?\s*what\s+percentage\s+of\s+"
        r"(.+?)\s+comes?\s+from\s+(?:each\s+)?(.+)",
    )
    for pattern in contribution_patterns:
        match = re.fullmatch(pattern, compact, re.IGNORECASE)
        if match:
                return ResultCommand(
                    "contribution",
                    metric_text=_clean_field_text(match.group(1)),
                    dimension_text=_clean_field_text(match.group(2)),
                fallback_allowed=True,
            )

    if re.search(r"\b(?:profit|gross\s+margin)\s+(?:percentage|percent|pct|margin)\b", compact, re.IGNORECASE):
        return ResultCommand("profit_percentage", fallback_allowed=True)

    ratio_match = re.search(
        r"(?:calculate|show|what\s+is|give\s+me)?\s*(.+?)\s+"
        r"(?:divided\s+by|as\s+(?:a\s+)?percentage\s+of|/)\s+(.+)$",
        compact,
        re.IGNORECASE,
    )
    if ratio_match:
        return ResultCommand(
            "ratio",
            numerator_text=_clean_field_text(ratio_match.group(1)),
            denominator_text=_clean_field_text(ratio_match.group(2)),
            fallback_allowed=True,
        )

    group_match = re.fullmatch(
        r"(?:group|summarize|summarise)\s+(?:by\s+)?(.+?)\s+and\s+"
        r"(total|sum|average|avg|mean|count|minimum|min|maximum|max)\s+(.+)",
        compact,
        re.IGNORECASE,
    )
    if group_match:
        return ResultCommand(
            "aggregate",
            dimension_text=_clean_field_text(group_match.group(1)),
            aggregation=_AGGREGATIONS[group_match.group(2).lower()],
            metric_text=_clean_field_text(group_match.group(3)),
            fallback_allowed=True,
        )

    aggregate_match = re.fullmatch(
        r"(?:show|calculate|display|give\s+me|what\s+is)?\s*(?:the\s+)?"
        r"(total|sum|average|avg|mean|count|minimum|min|maximum|max)\s+"
        r"(.+?)\s+by\s+(?:each\s+)?(.+)",
        compact,
        re.IGNORECASE,
    )
    if aggregate_match:
        return ResultCommand(
            "aggregate",
            aggregation=_AGGREGATIONS[aggregate_match.group(1).lower()],
            metric_text=_clean_field_text(aggregate_match.group(2)),
            dimension_text=_clean_field_text(aggregate_match.group(3)),
            fallback_allowed=True,
        )
    return None


def execute_result_command(
    session_id: str,
    command: ResultCommand,
    *,
    cache: ResultCache = result_cache,
    source_result_id: str | None = None,
) -> ResultCommandOutcome:
    source = cache.get_snapshot(session_id, source_result_id)
    if not source:
        return ResultCommandOutcome(
            handled=not command.fallback_allowed,
            message="That result is no longer available. Run the business question again.",
        )

    source_id = str(source.get("result_id") or "")
    rows = list(source.get("rows") or [])
    before = len(rows)

    try:
        if command.action == "undo":
            restored = cache.restore_parent(session_id, source_id)
            return ResultCommandOutcome(
                handled=True,
                ok=True,
                message="Restored the previous result.",
                snapshot=restored,
                operation="undo",
                rows_before=before,
                rows_after=int(restored.get("row_count") or 0),
                source_result_id=source_id,
                derived_result_id=str(restored.get("result_id") or ""),
            )

        if not rows:
            return ResultCommandOutcome(
                handled=not command.fallback_allowed,
                message="The current result has no rows to transform.",
                source_result_id=source_id,
            )

        if command.action == "exclude":
            match_groups, error = _resolve_exclusions(rows, command.target_text)
            if error:
                return ResultCommandOutcome(
                    handled=True,
                    message=error,
                    source_result_id=source_id,
                    rows_before=before,
                    rows_after=before,
                )
            predicates: list[str] = []
            parameters: list[Any] = []
            for group in match_groups:
                parts: list[str] = []
                for column, value in group:
                    parts.append(f'{_quote_identifier(column)} IS NOT DISTINCT FROM ?')
                    parameters.append(value)
                predicates.append("(" + " AND ".join(parts) + ")")
            transform_sql = "SELECT * FROM result WHERE NOT (" + " OR ".join(predicates) + ")"
            expected_rows = [
                row for row in rows
                if not any(
                    all(row.get(column) == value for column, value in group)
                    for group in match_groups
                )
            ]
            transformed = cache.query(
                session_id,
                transform_sql,
                result_id=source_id,
                parameters=parameters,
            )
            if transformed != expected_rows:
                transformed = expected_rows
            affected = before - len(transformed)
            if affected <= 0:
                return ResultCommandOutcome(
                    handled=True,
                    message="I found the value locally, but it did not remove any rows.",
                    source_result_id=source_id,
                    rows_before=before,
                    rows_after=before,
                )
            snapshot = cache.derive_snapshot(
                session_id,
                source_id,
                transformed,
                question=str(source.get("question") or "Result"),
                operation="exclude",
                sql=transform_sql,
                metadata={
                    "affected_count": affected,
                    "parameter_count": len(parameters),
                    "metadata_contains_raw_values": False,
                },
            )
            return ResultCommandOutcome(
                handled=True,
                ok=True,
                message=f"Created a filtered result with {affected} row{'s' if affected != 1 else ''} excluded.",
                snapshot=snapshot,
                operation="exclude",
                rows_before=before,
                rows_after=len(transformed),
                affected_count=affected,
                source_result_id=source_id,
                derived_result_id=str(snapshot.get("result_id") or ""),
            )

        if command.action == "keep_top":
            limit = int(command.limit or 1)
            # ``limit`` is locally parsed and clamped to 1..1000. DuckDB does
            # not consistently preserve parameterized LIMIT clauses through
            # the cache validator/fallback path, so emit the safe integer.
            order_column = ""
            if command.metric_text:
                order_column, error = _resolve_column(rows, command.metric_text)
                if error:
                    return _command_error(command, source_id, before, error)
            order_clause = (
                f" ORDER BY {_quote_identifier(order_column)} DESC NULLS LAST"
                if order_column else ""
            )
            transform_sql = f"SELECT * FROM result{order_clause} LIMIT {limit}"
            transformed = cache.query(
                session_id, transform_sql, result_id=source_id,
            )
            expected_rows = list(rows)
            if order_column:
                populated = [row for row in rows if row.get(order_column) is not None]
                null_rows = [row for row in rows if row.get(order_column) is None]
                expected_rows = sorted(
                    populated, key=lambda row: row.get(order_column), reverse=True,
                ) + null_rows
            expected_rows = expected_rows[:limit]
            if transformed != expected_rows:
                transformed = expected_rows
            snapshot = cache.derive_snapshot(
                session_id,
                source_id,
                transformed,
                question=str(source.get("question") or "Result"),
                operation="keep_top",
                sql=transform_sql,
                metadata={
                    "limit": limit,
                    "order_column": order_column,
                    "metadata_contains_raw_values": False,
                },
            )
            return ResultCommandOutcome(
                handled=True,
                ok=True,
                message=f"Kept the first {len(transformed)} rows from the current result.",
                snapshot=snapshot,
                operation="keep_top",
                rows_before=before,
                rows_after=len(transformed),
                affected_count=max(0, before - len(transformed)),
                source_result_id=source_id,
                derived_result_id=str(snapshot.get("result_id") or ""),
            )

        if command.action == "keep_values":
            column, selected_values, error = _resolve_inclusions(
                rows, command.target_text,
            )
            if error:
                return _command_error(command, source_id, before, error)

            predicates = [
                f'{_quote_identifier(column)} IS NOT DISTINCT FROM ?'
                for _ in selected_values
            ]
            transform_sql = "SELECT * FROM result WHERE " + " OR ".join(predicates)
            transformed = cache.query(
                session_id,
                transform_sql,
                result_id=source_id,
                parameters=selected_values,
            )
            expected_rows = [
                row for row in rows
                if any(row.get(column) == value for value in selected_values)
            ]
            if transformed != expected_rows:
                transformed = expected_rows
            if not transformed:
                return _command_error(
                    command,
                    source_id,
                    before,
                    "Those periods were not found in the current result.",
                )
            snapshot = cache.derive_snapshot(
                session_id,
                source_id,
                transformed,
                question=str(source.get("question") or "Result"),
                operation="filter",
                sql=transform_sql,
                column_formats=dict(source.get("column_formats") or {}),
                metadata={
                    "column": column,
                    "operator": "in",
                    "parameter_count": len(selected_values),
                    "metadata_contains_raw_values": False,
                },
            )
            return _command_success(
                snapshot,
                "filter",
                source_id,
                before,
                f"Kept {len(transformed)} matching rows from the cached result.",
            )

        if command.action == "sort":
            column, error = _resolve_column(rows, command.target_text)
            if error:
                return ResultCommandOutcome(
                    handled=True,
                    message=error,
                    source_result_id=source_id,
                    rows_before=before,
                    rows_after=before,
                )
            direction = "ASC" if command.direction == "asc" else "DESC"
            transform_sql = (
                f"SELECT * FROM result ORDER BY {_quote_identifier(column)} "
                f"{direction} NULLS LAST"
            )
            transformed = cache.query(session_id, transform_sql, result_id=source_id)
            if not transformed and rows:
                populated = [row for row in rows if row.get(column) is not None]
                null_rows = [row for row in rows if row.get(column) is None]
                transformed = sorted(
                    populated,
                    key=lambda row: row.get(column),
                    reverse=command.direction == "desc",
                ) + null_rows
            snapshot = cache.derive_snapshot(
                session_id,
                source_id,
                transformed,
                question=str(source.get("question") or "Result"),
                operation="sort",
                sql=transform_sql,
                metadata={"direction": direction.lower(), "metadata_contains_raw_values": False},
            )
            return ResultCommandOutcome(
                handled=True,
                ok=True,
                message=f"Sorted the cached result {direction.lower()}.",
                snapshot=snapshot,
                operation="sort",
                rows_before=before,
                rows_after=len(transformed),
                source_result_id=source_id,
                derived_result_id=str(snapshot.get("result_id") or ""),
            )

        if command.action == "filter":
            column, error = _resolve_column(rows, command.target_text)
            if error:
                return _command_error(command, source_id, before, error)
            value = _coerce_filter_value(rows, column, command.value_text)
            if _normalise_value(value) in _MASKED_MARKERS:
                return _command_error(
                    command, source_id, before,
                    "A generic masked value cannot be used as a filter. Use a visible value or row number.",
                )
            sql_operator, parameter = _filter_sql(command.operator, value)
            transform_sql = (
                f"SELECT * FROM result WHERE {_quote_identifier(column)} "
                f"{sql_operator} ?"
            )
            expected_rows = [
                row for row in rows
                if _filter_matches(row.get(column), command.operator, value)
            ]
            transformed = cache.query(
                session_id,
                transform_sql,
                result_id=source_id,
                parameters=[parameter],
            )
            if transformed != expected_rows:
                transformed = expected_rows
            formats = dict(source.get("column_formats") or {})
            snapshot = cache.derive_snapshot(
                session_id,
                source_id,
                transformed,
                question=str(source.get("question") or "Result"),
                operation="filter",
                sql=transform_sql,
                column_formats=formats,
                metadata={
                    "column": column,
                    "operator": command.operator,
                    "parameter_count": 1,
                    "metadata_contains_raw_values": False,
                },
            )
            return _command_success(
                snapshot, "filter", source_id, before,
                f"Filtered the cached result to {len(transformed)} matching rows.",
            )

        if command.action == "aggregate":
            dimension, dim_error = _resolve_column(rows, command.dimension_text)
            metric, metric_error = _resolve_metric_column(
                rows, command.metric_text, allow_row_count=command.aggregation == "count",
            )
            if dim_error or metric_error:
                return _command_error(
                    command, source_id, before, dim_error or metric_error,
                )
            if metric != "*" and command.aggregation != "count" and not _column_is_numeric(rows, metric):
                return _command_error(
                    command, source_id, before,
                    "That measure is not numeric in the current result.",
                )
            output_column = _aggregate_output_name(command.aggregation, metric)
            sql_metric = "*" if metric == "*" else _quote_identifier(metric)
            transform_sql = (
                f"SELECT {_quote_identifier(dimension)}, "
                f"{command.aggregation.upper()}({sql_metric}) AS {_quote_identifier(output_column)} "
                f"FROM result GROUP BY {_quote_identifier(dimension)} "
                f"ORDER BY {_quote_identifier(output_column)} DESC NULLS LAST"
            )
            expected_rows = _aggregate_rows(
                rows, dimension, metric, command.aggregation, output_column,
            )
            transformed = cache.query(session_id, transform_sql, result_id=source_id)
            if transformed != expected_rows:
                transformed = expected_rows
            formats = {}
            source_format = (source.get("column_formats") or {}).get(metric)
            if source_format and command.aggregation != "count":
                formats[output_column] = source_format
            snapshot = cache.derive_snapshot(
                session_id,
                source_id,
                transformed,
                question=str(source.get("question") or "Result"),
                operation="aggregate",
                sql=transform_sql,
                column_formats=formats,
                metadata={
                    "dimension": dimension,
                    "metric": metric,
                    "aggregation": command.aggregation,
                    "metadata_contains_raw_values": False,
                },
            )
            return _command_success(
                snapshot, "aggregate", source_id, before,
                f"Summarized the cached result into {len(transformed)} groups.",
            )

        if command.action == "contribution":
            dimension, dim_error = _resolve_column(rows, command.dimension_text)
            metric, metric_error = _resolve_metric_column(rows, command.metric_text)
            if dim_error or metric_error:
                return _command_error(command, source_id, before, dim_error or metric_error)
            if not _column_is_numeric(rows, metric):
                return _command_error(
                    command, source_id, before,
                    "That contribution measure is not numeric in the current result.",
                )
            metric_token = _safe_output_token(metric)
            total_column = (
                metric_token if metric_token.startswith("TOTAL_")
                else f"TOTAL_{metric_token}"
            )
            percent_column = "PERCENTAGE_CONTRIBUTION"
            transform_sql = (
                "WITH grouped AS (SELECT "
                f"{_quote_identifier(dimension)}, SUM({_quote_identifier(metric)}) AS value "
                f"FROM result GROUP BY {_quote_identifier(dimension)}) "
                f"SELECT {_quote_identifier(dimension)}, value AS {_quote_identifier(total_column)}, "
                f"ROUND(value * 100.0 / NULLIF(SUM(value) OVER (), 0), 2) "
                f"AS {_quote_identifier(percent_column)} FROM grouped "
                f"ORDER BY {_quote_identifier(percent_column)} DESC NULLS LAST"
            )
            expected_rows = _contribution_rows(
                rows, dimension, metric, total_column, percent_column,
            )
            transformed = cache.query(session_id, transform_sql, result_id=source_id)
            if transformed != expected_rows:
                transformed = expected_rows
            formats = {percent_column: "percentage"}
            source_format = (source.get("column_formats") or {}).get(metric)
            if source_format:
                formats[total_column] = source_format
            snapshot = cache.derive_snapshot(
                session_id,
                source_id,
                transformed,
                question=str(source.get("question") or "Result"),
                operation="contribution",
                sql=transform_sql,
                column_formats=formats,
                metadata={
                    "dimension": dimension,
                    "metric": metric,
                    "metadata_contains_raw_values": False,
                },
            )
            return _command_success(
                snapshot, "contribution", source_id, before,
                "Calculated percentage contribution from the cached result.",
            )

        if command.action in {"profit_percentage", "ratio"}:
            if command.action == "profit_percentage":
                revenue = _find_semantic_column(
                    rows, ("revenue", "sales amount", "invoice amount", "income"),
                )
                gross_profit = _find_semantic_column(
                    rows, ("gross profit", "profit amount", "gross margin amount"),
                )
                cost = _find_semantic_column(
                    rows, ("cost", "cogs", "cost of goods sold", "expense"),
                )
                if not revenue or not (gross_profit or cost):
                    return _command_error(
                        command, source_id, before,
                        "The cached result needs revenue plus cost or gross profit columns.",
                    )
                numerator = gross_profit or cost
                denominator = revenue
                subtract = not bool(gross_profit)
                output_column = "PROFIT_PERCENTAGE"
            else:
                numerator, num_error = _resolve_column(rows, command.numerator_text)
                denominator, den_error = _resolve_column(rows, command.denominator_text)
                if num_error or den_error:
                    return _command_error(command, source_id, before, num_error or den_error)
                subtract = False
                output_column = (
                    f"{_safe_output_token(numerator)}_PERCENT_OF_"
                    f"{_safe_output_token(denominator)}"
                )
            if not _column_is_numeric(rows, numerator) or not _column_is_numeric(rows, denominator):
                return _command_error(
                    command, source_id, before,
                    "Both calculation fields must be numeric in the cached result.",
                )
            if subtract:
                expression = (
                    f"({_quote_identifier(denominator)} - {_quote_identifier(numerator)}) "
                    f"* 100.0 / NULLIF({_quote_identifier(denominator)}, 0)"
                )
            else:
                expression = (
                    f"{_quote_identifier(numerator)} * 100.0 / "
                    f"NULLIF({_quote_identifier(denominator)}, 0)"
                )
            transform_sql = (
                f"SELECT *, ROUND({expression}, 2) AS {_quote_identifier(output_column)} "
                "FROM result"
            )
            expected_rows = _ratio_rows(
                rows, numerator, denominator, output_column, subtract=subtract,
            )
            transformed = cache.query(session_id, transform_sql, result_id=source_id)
            if transformed != expected_rows:
                transformed = expected_rows
            formats = dict(source.get("column_formats") or {})
            formats[output_column] = "percentage"
            snapshot = cache.derive_snapshot(
                session_id,
                source_id,
                transformed,
                question=str(source.get("question") or "Result"),
                operation=command.action,
                sql=transform_sql,
                column_formats=formats,
                metadata={
                    "numerator": numerator,
                    "denominator": denominator,
                    "metadata_contains_raw_values": False,
                },
            )
            return _command_success(
                snapshot, command.action, source_id, before,
                "Calculated the percentage locally from the cached result.",
            )
    except (LookupError, ValueError) as exc:
        return ResultCommandOutcome(
            handled=not command.fallback_allowed,
            message=str(exc),
            source_result_id=source_id,
        )

    return ResultCommandOutcome(handled=False)


def _command_error(
    command: ResultCommand,
    source_result_id: str,
    row_count: int,
    message: str,
) -> ResultCommandOutcome:
    return ResultCommandOutcome(
        handled=not command.fallback_allowed,
        message=message,
        source_result_id=source_result_id,
        rows_before=row_count,
        rows_after=row_count,
    )


def _command_success(
    snapshot: dict,
    operation: str,
    source_result_id: str,
    rows_before: int,
    message: str,
) -> ResultCommandOutcome:
    rows_after = int(snapshot.get("row_count") or 0)
    return ResultCommandOutcome(
        handled=True,
        ok=True,
        message=message,
        snapshot=snapshot,
        operation=operation,
        rows_before=rows_before,
        rows_after=rows_after,
        affected_count=max(0, rows_before - rows_after),
        source_result_id=source_result_id,
        derived_result_id=str(snapshot.get("result_id") or ""),
    )


def _strip_result_context(value: str) -> str:
    cleaned = re.sub(
        r"\b(?:from|using|in|on)?\s*(?:this|the|current|previous|cached)\s+"
        r"(?:result|results|data|dataset|rows?|table)\b",
        " ",
        str(value or ""),
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.?;")
    return cleaned


def _clean_field_text(value: str) -> str:
    cleaned = str(value or "").strip().strip("\"'").rstrip(".?")
    cleaned = re.sub(r"^(?:the|each|per)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _normalise_operator(value: str) -> str:
    compact = re.sub(r"\s+", " ", str(value or "").strip().lower())
    if compact.startswith("is ") and compact != "is not":
        compact = compact[3:]
    return {
        "is": "eq", "=": "eq", "equals": "eq", "equal to": "eq",
        "is not": "ne", "!=": "ne", "<>": "ne",
        "not equal to": "ne", "does not equal": "ne",
        ">": "gt", "greater than": "gt", "more than": "gt",
        "above": "gt", "over": "gt",
        ">=": "gte", "at least": "gte",
        "<": "lt", "less than": "lt", "below": "lt", "under": "lt",
        "<=": "lte", "at most": "lte",
        "contains": "contains", "starts with": "starts_with", "ends with": "ends_with",
    }.get(compact, compact)


def _coerce_filter_value(rows: list[dict], column: str, raw_value: str) -> Any:
    text = str(raw_value or "").strip().strip("\"'")
    values = [row.get(column) for row in rows if row.get(column) is not None]
    sample = values[0] if values else None
    numeric_text = re.sub(r"[$,%\s]", "", text).replace(",", "")
    if isinstance(sample, bool):
        return text.casefold() in {"true", "yes", "1", "y"}
    if isinstance(sample, (int, float)):
        try:
            number = float(numeric_text)
            return int(number) if isinstance(sample, int) and number.is_integer() else number
        except ValueError:
            return text
    return text


def _filter_sql(operator: str, value: Any) -> tuple[str, Any]:
    if operator == "eq":
        return "IS NOT DISTINCT FROM", value
    if operator == "ne":
        return "IS DISTINCT FROM", value
    if operator in {"gt", "gte", "lt", "lte"}:
        return {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}[operator], value
    if operator == "contains":
        return "ILIKE", f"%{value}%"
    if operator == "starts_with":
        return "ILIKE", f"{value}%"
    if operator == "ends_with":
        return "ILIKE", f"%{value}"
    raise ValueError("That filter operator is not supported locally.")


def _filter_matches(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "eq":
        return _comparable(actual) == _comparable(expected)
    if operator == "ne":
        return _comparable(actual) != _comparable(expected)
    if operator == "contains":
        return str(expected).casefold() in str(actual or "").casefold()
    if operator == "starts_with":
        return str(actual or "").casefold().startswith(str(expected).casefold())
    if operator == "ends_with":
        return str(actual or "").casefold().endswith(str(expected).casefold())
    if actual is None:
        return False
    left, right = _ordered_pair(actual, expected)
    return {
        "gt": left > right,
        "gte": left >= right,
        "lt": left < right,
        "lte": left <= right,
    }[operator]


def _comparable(value: Any) -> Any:
    if isinstance(value, str):
        return value.casefold()
    return value


def _ordered_pair(left: Any, right: Any) -> tuple[Any, Any]:
    try:
        return float(str(left).replace(",", "")), float(str(right).replace(",", ""))
    except (TypeError, ValueError):
        return str(left).casefold(), str(right).casefold()


def _resolve_metric_column(
    rows: list[dict], target: str, *, allow_row_count: bool = False,
) -> tuple[str, str]:
    if allow_row_count and _normalise_value(target) in {
        "row", "rows", "record", "records", "item", "items", "entries",
    }:
        return "*", ""
    return _resolve_column(rows, target)


def _column_is_numeric(rows: list[dict], column: str) -> bool:
    values = [row.get(column) for row in rows if row.get(column) not in (None, "")]
    if not values:
        return False
    try:
        for value in values:
            float(str(value).replace(",", "").replace("$", "").replace("%", ""))
        return True
    except (TypeError, ValueError):
        return False


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", "").replace("$", "").replace("%", ""))
    except (TypeError, ValueError):
        return None


def _aggregate_output_name(aggregation: str, metric: str) -> str:
    token = "ROWS" if metric == "*" else _safe_output_token(metric)
    prefix = {"sum": "TOTAL", "avg": "AVERAGE", "count": "COUNT", "min": "MIN", "max": "MAX"}[aggregation]
    return f"{prefix}_{token}"


def _safe_output_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value or "").upper()).strip("_") or "VALUE"


def _aggregate_rows(
    rows: list[dict], dimension: str, metric: str, aggregation: str, output_column: str,
) -> list[dict]:
    groups: dict[Any, list[Any]] = {}
    for row in rows:
        groups.setdefault(row.get(dimension), []).append(
            1 if metric == "*" else row.get(metric)
        )
    output: list[dict] = []
    for key, values in groups.items():
        populated = [value for value in values if value is not None]
        numbers = [_number(value) for value in populated]
        numbers = [value for value in numbers if value is not None]
        if aggregation == "count":
            aggregate: Any = len(populated)
        elif not numbers:
            aggregate = None
        elif aggregation == "sum":
            aggregate = sum(numbers)
        elif aggregation == "avg":
            aggregate = sum(numbers) / len(numbers)
        elif aggregation == "min":
            aggregate = min(numbers)
        else:
            aggregate = max(numbers)
        output.append({dimension: key, output_column: aggregate})
    return sorted(
        output,
        key=lambda row: (row.get(output_column) is not None, row.get(output_column) or 0),
        reverse=True,
    )


def _contribution_rows(
    rows: list[dict], dimension: str, metric: str,
    total_column: str, percent_column: str,
) -> list[dict]:
    grouped = _aggregate_rows(rows, dimension, metric, "sum", total_column)
    total = sum(float(row.get(total_column) or 0) for row in grouped)
    for row in grouped:
        value = float(row.get(total_column) or 0)
        row[percent_column] = round(value * 100.0 / total, 2) if total else None
    return sorted(
        grouped,
        key=lambda row: (row.get(percent_column) is not None, row.get(percent_column) or 0),
        reverse=True,
    )


def _find_semantic_column(rows: list[dict], candidates: tuple[str, ...]) -> str:
    columns = [str(column) for column in rows[0].keys()]
    normalised = {column: _normalise_value(column) for column in columns}
    for candidate in candidates:
        wanted = _normalise_value(candidate)
        exact = [column for column, value in normalised.items() if value == wanted]
        if exact:
            return exact[0]
        partial = [column for column, value in normalised.items() if wanted in value]
        if partial:
            return sorted(partial, key=lambda column: len(normalised[column]))[0]
    return ""


def _ratio_rows(
    rows: list[dict], numerator: str, denominator: str,
    output_column: str, *, subtract: bool,
) -> list[dict]:
    output: list[dict] = []
    for row in rows:
        numerator_value = _number(row.get(numerator))
        denominator_value = _number(row.get(denominator))
        value = None
        if numerator_value is not None and denominator_value not in (None, 0):
            base = denominator_value - numerator_value if subtract else numerator_value
            value = round(base * 100.0 / denominator_value, 2)
        output.append({**row, output_column: value})
    return output


def _resolve_exclusions(
    rows: list[dict], target_text: str,
) -> tuple[list[list[tuple[str, Any]]], str]:
    row_match = _ROW_RE.fullmatch(target_text)
    if row_match:
        index = int(row_match.group(1)) - 1
        if index < 0 or index >= len(rows):
            return [], "That row number is outside the current result."
        # Match the whole row using every value so duplicate displayed values
        # do not accidentally remove unrelated records.
        return [list(rows[index].items())], ""

    has_list_separator = bool(re.search(r",|\band\b", target_text, re.IGNORECASE))
    if not has_list_separator:
        whole_matches = _find_value_matches(rows, target_text)
        if len(whole_matches) == 1:
            return [[whole_matches[0]]], ""
        if len(whole_matches) > 1:
            return [], (
                "That value appears in more than one field in the current result. "
                "Use a more specific value or say `exclude row N`."
            )

    targets = [
        _clean_target(part)
        for part in re.split(r"\s*(?:,|\band\b)\s*", target_text, flags=re.IGNORECASE)
        if _clean_target(part)
    ]
    if len(targets) <= 1:
        return [], "I could not find that value in the current result."

    resolved: list[list[tuple[str, Any]]] = []
    for target in targets:
        matches = _find_value_matches(rows, target)
        if not matches:
            return [], "One of those values was not found in the current result."
        if len(matches) > 1:
            return [], (
                "One of those values appears in more than one field. "
                "Use a more specific value or say `exclude row N`."
            )
        resolved.append([matches[0]])
    return resolved, ""


def _resolve_inclusions(
    rows: list[dict], target_text: str,
) -> tuple[str, list[Any], str]:
    """Resolve natural month names to one cached temporal column locally."""
    targets = [
        _clean_temporal_subset_target(part)
        for part in re.split(r"\s*(?:,|\band\b|&)\s*", target_text, flags=re.IGNORECASE)
        if _clean_temporal_subset_target(part)
    ]
    if not targets:
        return "", [], "No periods were provided."

    matches_by_target: list[dict[str, list[Any]]] = []
    for target in targets:
        matches = _find_temporal_value_matches(rows, target)
        if not matches:
            return "", [], f"I could not find {target!r} in the current result."
        grouped: dict[str, list[Any]] = {}
        for column, value in matches:
            if value not in grouped.setdefault(column, []):
                grouped[column].append(value)
        matches_by_target.append(grouped)

    common_columns = set(matches_by_target[0])
    for grouped in matches_by_target[1:]:
        common_columns &= set(grouped)
    if not common_columns:
        return "", [], "Those periods do not resolve to one field in the current result."

    def column_score(column: str) -> tuple[int, int]:
        tokens = _semantic_tokens(column)
        semantic = (
            3 if "month" in tokens
            else 2 if "period" in tokens
            else 1 if "date" in tokens
            else 0
        )
        return semantic, -len(column)

    ranked = sorted(common_columns, key=column_score, reverse=True)
    if len(ranked) > 1 and column_score(ranked[0]) == column_score(ranked[1]):
        return "", [], "More than one date field matches. Name the result column explicitly."
    column = ranked[0]
    selected: list[Any] = []
    for grouped in matches_by_target:
        for value in grouped[column]:
            if value not in selected:
                selected.append(value)
    return column, selected, ""


def _find_temporal_value_matches(
    rows: list[dict], target: str,
) -> list[tuple[str, Any]]:
    reference = _month_reference(target)
    if reference is None:
        return _find_value_matches(rows, target)
    target_year, target_month = reference
    unique: dict[tuple[str, str], tuple[str, Any]] = {}
    for row in rows:
        for column, value in row.items():
            actual = _value_year_month(value)
            if actual is None:
                continue
            actual_year, actual_month = actual
            if actual_month != target_month:
                continue
            if target_year is not None and actual_year != target_year:
                continue
            unique[(str(column), repr(value))] = (str(column), value)
    return list(unique.values())


def _find_value_matches(rows: list[dict], target: str) -> list[tuple[str, Any]]:
    wanted = _normalise_value(target)
    wanted_temporal = _temporal_key(target)
    if not wanted or wanted in _MASKED_MARKERS:
        return []
    unique: dict[tuple[str, str], tuple[str, Any]] = {}
    for row in rows:
        for column, value in row.items():
            if value is None or isinstance(value, (dict, list, tuple, set)):
                continue
            actual = _normalise_value(value)
            if not actual:
                continue
            actual_temporal = _temporal_key(value)
            if (
                actual == wanted
                or wanted.endswith(actual)
                or actual.endswith(wanted)
                or (wanted_temporal and actual_temporal == wanted_temporal)
            ):
                key = (str(column), repr(value))
                unique[key] = (str(column), value)
    return list(unique.values())


def _resolve_column(rows: list[dict], target: str) -> tuple[str, str]:
    columns = [str(column) for column in rows[0].keys()]
    wanted = _normalise_value(target)
    exact = [column for column in columns if _normalise_value(column) == wanted]
    if len(exact) == 1:
        return exact[0], ""
    partial = [
        column for column in columns
        if wanted and (wanted in _normalise_value(column) or _normalise_value(column) in wanted)
    ]
    if len(partial) == 1:
        return partial[0], ""
    semantic = _semantic_column_matches(rows, target, columns)
    if len(semantic) == 1:
        return semantic[0], ""
    if len(semantic) > 1:
        return "", "That field name is ambiguous. Use the exact result column name."
    if not partial:
        return "", "That field is not present in the current result."
    return "", "That field name is ambiguous. Use the exact result column name."


def resolve_result_column(rows: list[dict], target: str) -> tuple[str, str]:
    """Public, local-only result-column resolver used by governed planners."""
    if not rows:
        return "", "The current result has no columns to resolve."
    return _resolve_column(rows, target)


_COLUMN_NOISE = {
    "total", "sum", "summed", "average", "avg", "mean", "amount",
    "value", "number", "count", "percentage", "percent", "pct", "each",
}


def _semantic_tokens(value: Any) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", str(value or "").casefold())
        if token not in _COLUMN_NOISE
    }


def _semantic_column_matches(
    rows: list[dict], target: str, columns: list[str],
) -> list[str]:
    wanted = _semantic_tokens(target)
    matches = [
        column for column in columns
        if wanted and (
            wanted.issubset(_semantic_tokens(column))
            or _semantic_tokens(column).issubset(wanted)
        )
    ]
    if matches:
        return matches

    if wanted & {"month", "date", "period", "year", "quarter", "week"}:
        temporal = []
        for column in columns:
            column_tokens = _semantic_tokens(column)
            values = [row.get(column) for row in rows if row.get(column) is not None]
            if (
                column_tokens & {"month", "date", "period", "year", "quarter", "week"}
                or (values and sum(bool(_temporal_key(value)) for value in values[:20]) >= min(2, len(values)))
            ):
                temporal.append(column)
        return temporal
    return []


_MONTHS = {
    name.casefold(): index
    for index, name in enumerate(month_name)
    if index and name
}
_MONTHS.update({
    name.casefold(): index
    for index, name in enumerate(month_abbr)
    if index and name
})


def _temporal_key(value: Any) -> str:
    """Canonicalise common user/display date forms without database access."""
    if isinstance(value, datetime):
        return f"day:{value:%Y-%m-%d}"
    if isinstance(value, date):
        return f"day:{value:%Y-%m-%d}"
    text = str(value or "").strip().casefold().rstrip(".")
    if not text:
        return ""
    match = re.fullmatch(r"(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?", text)
    if match:
        year, month, day = int(match.group(1)), int(match.group(2)), match.group(3)
        if 1 <= month <= 12:
            return (
                f"day:{year:04d}-{month:02d}-{int(day):02d}"
                if day else f"month:{year:04d}-{month:02d}"
            )
    words = re.findall(r"[a-z]+|\d{4}", text)
    year = next((int(word) for word in words if word.isdigit() and len(word) == 4), None)
    month = next((_MONTHS[word] for word in words if word in _MONTHS), None)
    if year and month:
        return f"month:{year:04d}-{month:02d}"
    return ""


def _month_reference(value: Any) -> tuple[int | None, int] | None:
    text = str(value or "").strip().casefold().rstrip(".")
    words = re.findall(r"[a-z]+|\d{4}", text)
    month = next((_MONTHS[word] for word in words if word in _MONTHS), None)
    if month is None:
        return None
    year = next((int(word) for word in words if word.isdigit() and len(word) == 4), None)
    return year, month


def _contains_month_reference(value: Any) -> bool:
    return _month_reference(value) is not None


def _value_year_month(value: Any) -> tuple[int | None, int] | None:
    if isinstance(value, (date, datetime)):
        return value.year, value.month
    temporal = _temporal_key(value)
    match = re.fullmatch(r"(?:day|month):(\d{4})-(\d{2})(?:-\d{2})?", temporal)
    if match:
        return int(match.group(1)), int(match.group(2))
    return _month_reference(value)


def _normalise_value(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _clean_target(value: str) -> str:
    target = str(value or "").strip().strip("\"'")
    target = re.sub(
        r"^(?:doctor|dr\.?|customer|patient|supplier|warehouse|product|item)\s+",
        "",
        target,
        flags=re.IGNORECASE,
    )
    return target.strip().strip("\"'")


def _clean_temporal_subset_target(value: str) -> str:
    target = str(value or "").strip().strip("\"'").rstrip(".?!")
    target = re.sub(
        r"^(?:the\s+)?(?:months?|periods?|dates?)\s+(?:of\s+)?",
        "",
        target,
        flags=re.IGNORECASE,
    )
    target = re.sub(
        r"\s+(?:from|in)\s+(?:this|the|current|previous|cached)\s+"
        r"(?:result|results|data|dataset|rows?|table)$",
        "",
        target,
        flags=re.IGNORECASE,
    )
    return target.strip()


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'
