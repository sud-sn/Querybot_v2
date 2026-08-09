"""Deterministic post-execution verification for analytical result shapes.

SQL validation proves that a query is safe and schema-valid.  It does not prove
that the returned shape answers the requested analysis.  This module compares
metadata-only intent/resolution plans with output column names, value types, and
row counts.  It never calls an LLM and never persists or logs returned values.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
import re
from typing import Any


_TIME_NAME_RE = re.compile(
    r"(?:^|_)(?:date|day|week|month|quarter|year|period|fiscal|calendar)(?:_|$)",
    re.I,
)
_TIME_VALUE_RE = re.compile(
    r"^(?:\d{4}(?:[-/]\d{1,2})?(?:[-/]\d{1,2})?|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[- /]\d{2,4}|"
    r"q[1-4][ -]?\d{2,4})$",
    re.I,
)
_CONTRIBUTION_RE = re.compile(r"(?:contribution|share|percent|percentage|pct)", re.I)


def _plan_dict(plan: Any) -> dict[str, Any]:
    if isinstance(plan, dict):
        return dict(plan)
    if is_dataclass(plan):
        return asdict(plan)
    to_dict = getattr(plan, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        return dict(value) if isinstance(value, dict) else {}
    return {}


def _normalise(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _numeric_columns(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    numeric: list[str] = []
    for column in columns:
        values = [row.get(column) for row in rows[:50] if row.get(column) is not None]
        if values and all(
            isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)
            for value in values
        ):
            numeric.append(column)
    return numeric


def _time_columns(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    found: list[str] = []
    for column in columns:
        if _TIME_NAME_RE.search(str(column)):
            found.append(column)
            continue
        values = [row.get(column) for row in rows[:20] if row.get(column) is not None]
        if values and all(
            isinstance(value, (date, datetime))
            or bool(_TIME_VALUE_RE.match(str(value).strip()))
            for value in values
        ):
            found.append(column)
    return found


def _column_matches(term: str, columns: list[str]) -> bool:
    needle = _normalise(term)
    if not needle:
        return False
    return any(
        needle in _normalise(column) or _normalise(column) in needle
        for column in columns
        if _normalise(column)
    )


def verify_result_shape(
    rows: list[dict[str, Any]] | None,
    *,
    analytical_plan: Any = None,
    resolution_plan: dict[str, Any] | None = None,
    request_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an auditable shape-verification report for one executed result.

    A failed report is a confidence signal, not an authorization decision.
    The SQL/policy validators remain responsible for blocking unsafe queries.
    """

    rows = [row for row in (rows or []) if isinstance(row, dict)]
    intent_plan = _plan_dict(analytical_plan)
    resolution = dict(resolution_plan or {})
    compiled_request = dict(request_plan or {})
    columns = list(rows[0].keys()) if rows else []
    numeric = _numeric_columns(rows, columns)
    temporal = _time_columns(rows, columns)
    warnings: list[str] = []
    errors: list[str] = []
    checks: list[str] = []

    if not rows:
        return {
            "status": "empty",
            "score": 0,
            "checks": [],
            "warnings": [],
            "errors": ["The query returned no rows to verify."],
            "row_count": 0,
            "columns": [],
            "numeric_columns": [],
            "time_columns": [],
            "metric_binding_source": (
                "metric_registry" if resolution.get("metrics") else "unresolved"
            ),
        }

    resolved_metrics = [
        str(metric.get("name") or "")
        for metric in resolution.get("metrics") or []
        if isinstance(metric, dict) and metric.get("name")
    ]
    if resolved_metrics:
        checks.append("Approved metric binding was present in the resolution plan.")
        if not numeric:
            errors.append("The result has no numeric output for the resolved metric.")
    elif intent_plan.get("metrics") and not numeric:
        errors.append("The requested measure did not produce a numeric output column.")

    intent = str(intent_plan.get("intent") or "").casefold()
    output = str(intent_plan.get("output") or "auto").casefold()
    dimensions = [str(value) for value in intent_plan.get("dimensions") or [] if str(value)]
    missing_dimensions = [value for value in dimensions if not _column_matches(value, columns)]
    if missing_dimensions:
        warnings.append(
            "Requested dimension is not visible in the output columns: "
            + ", ".join(missing_dimensions[:3])
            + "."
        )
    elif dimensions:
        checks.append("Requested dimensions are represented in the result shape.")

    if intent == "trend":
        if len(rows) < 2:
            errors.append("A trend requires at least two returned periods.")
        if not temporal:
            errors.append("A trend requires a visible date or period column.")
        if not numeric:
            errors.append("A trend requires a numeric measure column.")
        if len(rows) >= 2 and temporal and numeric:
            checks.append("Trend output contains multiple periods and a numeric measure.")

    if intent == "ranking":
        top_n = intent_plan.get("top_n")
        if top_n is not None:
            try:
                if len(rows) > int(top_n):
                    warnings.append(
                        f"The request asked for top {int(top_n)}, but {len(rows)} rows were returned."
                    )
            except (TypeError, ValueError):
                pass
        if not numeric:
            errors.append("A ranking requires a numeric measure column.")
        if len(columns) < 2:
            errors.append("A ranking requires an entity/dimension and a measure.")

    if intent == "distribution":
        if len(rows) < 2:
            errors.append("A distribution requires at least two categories.")
        if not numeric or len(columns) < 2:
            errors.append("A distribution requires a category and numeric measure.")
        share_columns = [column for column in numeric if _CONTRIBUTION_RE.search(column)]
        for column in share_columns[:1]:
            total = sum(float(row.get(column) or 0) for row in rows)
            scale = 100.0 if total > 1.5 else 1.0
            tolerance = 2.0 if scale == 100.0 else 0.02
            if abs(total - scale) > tolerance:
                warnings.append(
                    f"Returned {column} values sum to {total:.2f}, not approximately {scale:.0f}."
                )
            else:
                checks.append("Distribution percentages reconcile to the full returned series.")

    if intent == "comparison":
        if len(rows) < 2 and len(numeric) < 2:
            errors.append("A comparison requires multiple rows or multiple measure columns.")

    if output == "kpi":
        if len(rows) != 1:
            errors.append(f"A KPI request should return one row, not {len(rows)}.")
        if not numeric:
            errors.append("A KPI request requires a numeric value.")

    if output in {"line chart", "area chart"} and not temporal:
        errors.append(f"A {output} requires a date or period axis.")
    if output in {"pie chart", "bar chart", "chart", "graph"} and not numeric:
        errors.append("The requested chart requires a numeric measure.")

    observed_policies = [
        policy for policy in (compiled_request.get("temporal_operations") or [])
        if str(policy.get("kind") or "") == "latest_n_observed"
    ]
    for policy in observed_policies:
        amount = max(1, int(policy.get("amount") or 1))
        if not temporal:
            errors.append("Latest observed periods require a visible business-date column.")
            continue
        if len(rows) > amount:
            errors.append(
                f"The request asked for the latest {amount} observed periods, but {len(rows)} period rows were returned."
            )
        date_column = temporal[0]
        visible_values = [row.get(date_column) for row in rows if row.get(date_column) is not None]
        if len(visible_values) != len({str(value) for value in visible_values}):
            errors.append("Latest observed periods were not returned at one row per distinct business date.")
        if rows and len(rows) <= amount and temporal and not errors:
            checks.append(f"Result is grouped into the latest {len(rows)} distinct observed period(s).")

    if not errors and not warnings:
        checks.append("Returned columns and row shape are consistent with the analytical request.")

    score = max(0, 100 - (30 * len(errors)) - (10 * len(warnings)))
    status = "fail" if errors else "warning" if warnings else "pass"
    return {
        "status": status,
        "score": score,
        "checks": checks[:6],
        "warnings": warnings[:6],
        "errors": errors[:6],
        "row_count": len(rows),
        "columns": columns,
        "numeric_columns": numeric,
        "time_columns": temporal,
        "metric_binding_source": (
            "metric_registry" if resolved_metrics else "semantic_field_or_schema"
        ),
        "resolved_metrics": resolved_metrics,
    }
