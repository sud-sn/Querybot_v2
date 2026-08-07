"""Governed analytical-shape contracts for generated SQL.

The metric registry remains authoritative.  This module covers the common case
where a user asks for a new dimensional breakdown that has no pre-written SQL:
it records the resolved measure/dimension evidence and constrains the LLM and
validator to a grouped composition query instead of a row-level distribution.
It contains no tenant-specific vocabulary or physical schema names.
"""

from __future__ import annotations

import re
from typing import Any

from core.contribution_analysis import detect_composition_intent


_IDENTIFIER_RE = re.compile(r"(?:^|_)(?:ID|KEY|CODE|NUM|NO|NBR|SEQ)$", re.I)
_NON_ADDITIVE_RE = re.compile(r"(?:PCT|PERCENT|RATE|RATIO|AVG|AVERAGE)", re.I)
_SEMI_ADDITIVE_RE = re.compile(
    r"(?:BALANCE|ON_HAND|INVENTORY|SNAPSHOT|ENDING|CLOSING)", re.I
)
_ADDITIVE_RE = re.compile(
    r"(?:AMT|AMOUNT|REVENUE|SALES|COST|SPEND|PROFIT|INCOME|EXPENSE|"
    r"QTY|QUANTITY|VOLUME|UNITS|COUNT|CNT)",
    re.I,
)


def _measure_class(column: str, field: dict[str, Any]) -> str:
    declared = str(
        field.get("aggregation_semantics")
        or field.get("aggregation")
        or ""
    ).strip().lower()
    if declared in {"additive", "semi_additive", "semi-additive", "non_additive", "non-additive"}:
        return declared.replace("-", "_")
    name = str(column or "")
    if _NON_ADDITIVE_RE.search(name):
        return "non_additive"
    if _SEMI_ADDITIVE_RE.search(name):
        return "semi_additive"
    if _ADDITIVE_RE.search(name):
        return "additive"
    return "unknown"


def measure_class_for_metric(metric: dict[str, Any]) -> str:
    """Classify a registry metric as additive / semi_additive / non_additive.

    An admin-declared `aggregation_semantics` (or `aggregation`) is
    authoritative; otherwise the measure columns named in the approved formula
    are inspected. Semi-additive wins any tie because it is the restrictive
    reading — a balance summed across time is wrong, whereas an additive
    measure aggregated at one grain is merely narrower than it needed to be.

    Tenant-neutral: reads only registry metadata and column naming, never
    physical schema names or a client's vocabulary.
    """
    declared = str(
        metric.get("aggregation_semantics")
        or metric.get("aggregation")
        or ""
    ).strip().lower().replace("-", "_")
    if declared in {"additive", "semi_additive", "non_additive"}:
        return declared

    formula = str(
        metric.get("sql_template")
        or metric.get("formula")
        or metric.get("expression")
        or ""
    )
    source = f"{formula} {metric.get('column') or ''}"
    classes = {
        _measure_class(token, {})
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", source)
    }
    for candidate in ("semi_additive", "non_additive", "additive"):
        if candidate in classes:
            return candidate
    return "unknown"


def build_analysis_contract(
    question: str,
    *,
    analytical_intents: dict[str, Any] | None = None,
    semantic_plan: dict[str, Any] | None = None,
    metric_formulas: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a tenant-neutral contract for SQL generation and validation."""
    intents = analytical_intents or {}
    composition = bool(intents.get("contribution")) or detect_composition_intent(question)
    if not composition:
        return {"enabled": False, "mode": "none"}

    measures: list[dict[str, Any]] = []
    dimensions: list[dict[str, Any]] = []

    for metric in metric_formulas or []:
        formula = str(metric.get("sql_template") or "").strip()
        if not formula:
            continue
        measures.append({
            "name": str(metric.get("name") or metric.get("label") or "approved metric"),
            "formula": formula,
            "source": "approved_metric",
            "classification": str(metric.get("aggregation_semantics") or "approved"),
            "authoritative": True,
        })

    for field in (semantic_plan or {}).get("fields") or []:
        role = str(field.get("role") or "").lower()
        column = str(field.get("column") or "")
        table = str(field.get("table") or field.get("source_table") or "")
        if not column:
            continue
        if role in {"measure", "measure_candidate"} and not _IDENTIFIER_RE.search(column):
            if not measures:
                measures.append({
                    "name": str(field.get("term") or column),
                    "table": table,
                    "column": column,
                    "source": "semantic_field",
                    "classification": _measure_class(column, field),
                    "authoritative": str(field.get("source") or "").startswith("approved"),
                    "confidence": field.get("confidence"),
                })
        elif role in {
            "dimension", "display_dimension", "identifier", "attribute",
            "date_dimension", "dimension_key",
        }:
            dimensions.append({
                "name": str(field.get("term") or column),
                "table": table,
                "column": column,
                "source": str(field.get("source") or "semantic_field"),
                "authoritative": field.get("enforcement") == "required",
            })

    return {
        "enabled": True,
        "mode": "composition",
        "requires_grouping": True,
        "requires_aggregate": True,
        "requires_row_values": False,
        "measure_source": (
            "approved_metric" if any(m.get("source") == "approved_metric" for m in measures)
            else "semantic_field" if measures
            else "schema_grounded"
        ),
        "measures": measures,
        "dimensions": dimensions,
        "chart_policy": {
            "preferred": "pie",
            "max_default_categories": 6,
            "fallback": "bar",
            "requires_non_negative_values": True,
            "requires_positive_total": True,
        },
    }


def format_analysis_contract(contract: dict[str, Any]) -> str:
    """Format the contract as concise, copy-safe SQL-generation guidance."""
    if not contract or not contract.get("enabled"):
        return ""
    lines = [
        "GOVERNED ANALYSIS CONTRACT:",
        "- Intent: categorical composition/share, not a statistical histogram.",
        "- Return one row per requested business category.",
        "- SELECT the category plus an aggregated measure and GROUP BY the category.",
        "- Do NOT return individual measure rows for automatic binning.",
        "- Preserve governed joins, metric filters, row grain, and tenant scope.",
    ]
    for measure in contract.get("measures") or []:
        if measure.get("source") == "approved_metric":
            lines.append(
                f"- Approved measure '{measure.get('name')}': use this exact formula: "
                f"{measure.get('formula')}"
            )
            continue
        ref = ".".join(
            part for part in (measure.get("table"), measure.get("column")) if part
        )
        classification = str(measure.get("classification") or "unknown")
        lines.append(
            f"- Schema-grounded measure '{measure.get('name')}': {ref or 'resolve from the schema'} "
            f"({classification})."
        )
        if classification == "non_additive":
            lines.append(
                "- This measure is non-additive: never SUM the stored rate/ratio; derive it from governed components or return CANNOT_GENERATE."
            )
        elif classification == "semi_additive":
            lines.append(
                "- This measure is semi-additive: aggregate only at one governed snapshot/date grain; do not sum across snapshots."
            )
        elif classification == "unknown":
            lines.append(
                "- Aggregation semantics are not proven: do not invent a formula; use explicit KB guidance or return CANNOT_GENERATE."
            )
    if not contract.get("measures"):
        lines.append(
            "- No registered metric matched. Resolve exactly one additive measure from schema/KB evidence; if multiple business definitions remain, return CANNOT_GENERATE rather than guessing."
        )
    return "\n".join(lines)
