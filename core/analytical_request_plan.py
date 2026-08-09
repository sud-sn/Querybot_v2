"""Compile a question's resolved semantics into one executable request plan."""

from __future__ import annotations

from typing import Any


def compile_analytical_request_plan(
    question: str,
    semantic_plan: dict[str, Any] | None,
    *,
    matched_metrics: list[dict[str, Any]] | None = None,
    analysis_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan = semantic_plan or {}
    source_scope = plan.get("source_scope") or {}
    selected_fact = str(source_scope.get("selected_fact") or plan.get("fact_anchor") or "")
    fields = [f for f in (plan.get("fields") or []) if f.get("enforcement") != "optional"]
    measures = [
        {"term": f.get("term"), "table": f.get("table"), "column": f.get("column")}
        for f in fields if str(f.get("role") or "").lower() in {"measure", "measure_candidate"}
    ]
    dimensions = [
        {"term": f.get("term"), "table": f.get("table"), "column": f.get("column")}
        for f in fields if str(f.get("role") or "").lower() in {
            "dimension", "display_dimension", "attribute", "date_dimension", "contextual_date",
        }
    ]
    temporal = [dict(p) for p in (plan.get("temporal_policies") or [])]
    output_shape = "table"
    if any(str(p.get("kind") or "") == "latest_n_observed" for p in temporal):
        output_shape = "time_series"
    elif (analysis_contract or {}).get("mode"):
        output_shape = str((analysis_contract or {}).get("mode") or "table")
    return {
        "version": 1,
        "question": str(question or ""),
        "status": "compiled" if (selected_fact or measures or dimensions) else "unresolved",
        "source_fact": selected_fact,
        "measures": measures,
        "dimensions": dimensions,
        "temporal_operations": temporal,
        "joins": [dict(j) for j in (plan.get("joins") or []) if j.get("enforcement") != "optional"],
        "metrics": [
            {"id": m.get("id"), "name": m.get("name"), "formula": m.get("formula") or m.get("sql_formula")}
            for m in (matched_metrics or [])
        ],
        "output_shape": output_shape,
        "prohibitions": [
            "no_direct_fact_to_fact_join",
            "no_unapproved_field_substitution",
            "no_server_clock_for_relative_business_dates",
        ],
    }


def format_analytical_request_plan(plan: dict[str, Any] | None) -> str:
    if not plan or plan.get("status") != "compiled":
        return ""
    lines = ["## Executable analytical request plan"]
    if plan.get("source_fact"):
        lines.append(f"- Single measure fact: {plan['source_fact']}")
    if plan.get("measures"):
        lines.append("- Measures: " + ", ".join(
            f"{m.get('table')}.{m.get('column')}" for m in plan["measures"]
        ))
    if plan.get("dimensions"):
        lines.append("- Output dimensions: " + ", ".join(
            f"{d.get('table')}.{d.get('column')}" for d in plan["dimensions"]
        ))
    lines.append(f"- Required output shape: {plan.get('output_shape') or 'table'}")
    lines.append("This plan is authoritative. If a SQL draft conflicts with it, regenerate the SQL; do not reinterpret the question.")
    return "\n".join(lines)
