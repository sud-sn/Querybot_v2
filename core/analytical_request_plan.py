"""Compile a question's resolved semantics into one executable request plan."""

from __future__ import annotations

from typing import Any


def _entity_physical_table(entity: dict[str, Any]) -> str:
    table = str(entity.get("table_name") or "").strip()
    if not table:
        return ""
    if "." in table:
        return table
    schema = str(entity.get("schema_name") or entity.get("schema") or "").strip()
    return f"{schema}.{table}" if schema else table


def _same_table(left: Any, right: Any) -> bool:
    left_text = str(left or "").strip().strip("[]\"`").upper()
    right_text = str(right or "").strip().strip("[]\"`").upper()
    return bool(
        left_text
        and right_text
        and (
            left_text == right_text
            or left_text.split(".")[-1] == right_text.split(".")[-1]
        )
    )


def compile_analytical_request_plan(
    question: str,
    semantic_plan: dict[str, Any] | None,
    *,
    matched_metrics: list[dict[str, Any]] | None = None,
    analysis_contract: dict[str, Any] | None = None,
    graph_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan = semantic_plan or {}
    source_scope = plan.get("source_scope") or {}
    selected_fact = str(source_scope.get("selected_fact") or plan.get("fact_anchor") or "")
    selected_facts = [
        str(value) for value in (source_scope.get("selected_facts") or []) if value
    ]
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
    graph = graph_context if isinstance(graph_context, dict) else {}
    join_plan = graph.get("join_plan") if isinstance(graph.get("join_plan"), dict) else {}
    entity_map = {
        str(entity.get("entity_name") or ""): entity
        for entity in graph.get("entities") or []
        if isinstance(entity, dict)
    }
    source_facts: list[str] = []
    subrequests: list[dict[str, Any]] = []
    if str(join_plan.get("status") or "") == "requires_isolated_aggregation":
        common_dimensions = list(join_plan.get("common_dimensions") or [])
        for index, isolated in enumerate(join_plan.get("isolated_fact_plans") or [], start=1):
            entity_name = str(isolated.get("fact_entity") or "")
            physical = _entity_physical_table(entity_map.get(entity_name, {}))
            if not physical:
                continue
            source_facts.append(physical)
            subrequests.append({
                "id": f"fact_subplan_{index}",
                "fact_entity": entity_name,
                "source_fact": physical,
                "measures": [m for m in measures if _same_table(m.get("table"), physical)],
                "dimensions": [dict(d) for d in dimensions],
                "temporal_operations": [dict(item) for item in temporal],
                "group_by": common_dimensions,
                "aggregate_before_join": True,
                "physical_fact_must_appear_in_own_cte": True,
            })
    # Source arbitration can identify a multi-grain compound request even when
    # the entity graph has no explicit fact-to-fact path (the safe and common
    # case). Compile those facts into isolated subplans directly rather than
    # collapsing them to the first cadence or asking a misleading one-option
    # source clarification.
    if not subrequests and len(selected_facts) > 1:
        shared_dimensions = list(dict.fromkeys(
            str(d.get("term") or "") for d in dimensions if d.get("term")
        ))
        for index, physical in enumerate(selected_facts, start=1):
            source_facts.append(physical)
            subrequests.append({
                "id": f"fact_subplan_{index}",
                "source_fact": physical,
                "measures": [m for m in measures if _same_table(m.get("table"), physical)],
                "dimensions": [dict(d) for d in dimensions],
                "temporal_operations": [dict(item) for item in temporal],
                "group_by": shared_dimensions,
                "aggregate_before_join": True,
                "physical_fact_must_appear_in_own_cte": True,
            })
    if not source_facts and selected_fact:
        source_facts = [selected_fact]

    compiled = {
        "version": 1,
        "question": str(question or ""),
        "status": "compiled" if (selected_fact or source_facts or measures or dimensions) else "unresolved",
        "source_fact": selected_fact,
        "source_facts": source_facts,
        "subrequests": subrequests,
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
    if subrequests:
        compiled["combination"] = {
            "mode": "join_aggregated_subplans",
            "shared_dimensions": list(
                join_plan.get("common_dimensions")
                or subrequests[0].get("group_by")
                or []
            ),
            "join_inputs": "aggregated_subplans_only",
        }
    return compiled


def format_analytical_request_plan(plan: dict[str, Any] | None) -> str:
    if not plan or plan.get("status") != "compiled":
        return ""
    lines = ["## Executable analytical request plan"]
    if plan.get("subrequests"):
        lines.append("- Compound request: execute each fact subplan independently.")
        for subplan in plan["subrequests"]:
            lines.append(
                f"  - {subplan.get('id')}: aggregate {subplan.get('source_fact')} "
                f"to {', '.join(subplan.get('group_by') or []) or 'the requested output grain'}"
            )
        lines.append("- Combine only the aggregated subplan outputs; never join physical fact rows.")
    elif plan.get("source_fact"):
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
