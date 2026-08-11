"""Compile a question's resolved semantics into one executable request plan."""

from __future__ import annotations

import re
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
    analytical_intent_plan: dict[str, Any] | None = None,
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

    intent_plan = analytical_intent_plan if isinstance(analytical_intent_plan, dict) else {}
    intent = str(intent_plan.get("intent") or "").strip().casefold()
    measure_semantics = str(intent_plan.get("measure_semantics") or "").strip()
    counted_entity = str(intent_plan.get("counted_entity") or "").strip()
    derived_measure = {}
    if measure_semantics == "count_distinct_business_identifier" and counted_entity:
        count_resolution = plan.get("count_target") or {}
        exact_target = (
            count_resolution.get("selected")
            if count_resolution.get("status") == "selected"
            and isinstance(count_resolution.get("selected"), dict)
            else {}
        )
        derived_measure = {
            "semantics": measure_semantics,
            "business_entity": counted_entity,
            "aggregation": "count_distinct",
            "identifier_policy": "governed_stable_business_identifier",
            "target_table": exact_target.get("table") or "",
            "target_column": exact_target.get("column") or "",
            "business_name": exact_target.get("business_name") or "",
            "business_meaning": exact_target.get("business_meaning") or "",
            "confidence": exact_target.get("confidence"),
            "resolution_reason": count_resolution.get("reason") or "",
            "forbidden_substitutions": [
                "registered_metric_without_question_evidence",
                "amount_or_value_column",
                "quantity_column",
                "display_name",
                "line_or_row_surrogate",
            ],
        }

    question_text = str(question or "")
    change_direction = ""
    if re.search(r"\b(?:reduced|decreased|declined|fewer|dropped|fell)\b", question_text, re.I):
        change_direction = "decrease"
    elif re.search(r"\b(?:increased|grew|grown|more|rose|rising)\b", question_text, re.I):
        change_direction = "increase"
    comparison_requested = bool(
        intent == "comparison"
        or intent_plan.get("comparison")
        or re.search(
            r"\b(?:versus|vs\.?|compared?\s+(?:to|with)|previous|prior|baseline)\b",
            question_text,
            re.I,
        )
    )
    analytical_recipe: dict[str, Any] = {}
    if (
        derived_measure.get("semantics") == "count_distinct_business_identifier"
        and (change_direction or comparison_requested)
    ):
        analytical_recipe = {
            "kind": "period_over_period_entity_change",
            "measure": "count_distinct_business_identifier",
            "direction": change_direction or "compare",
            "entity_grain": str(intent_plan.get("entity_grain") or ""),
            "period_policy": "equal_non_overlapping_governed_windows",
            "required_outputs": [
                "entity_identifier",
                "current_period_count",
                "prior_period_count",
                "absolute_change",
                "percentage_change",
            ],
            "filter_policy": (
                "return_only_decreases" if change_direction == "decrease"
                else "return_only_increases" if change_direction == "increase"
                else "return_all_comparable_entities"
            ),
            "zero_baseline_policy": "preserve_and_label_not_comparable",
        }

    registered_metrics = [
        {"id": m.get("id"), "name": m.get("name"), "formula": m.get("formula") or m.get("sql_formula")}
        for m in (matched_metrics or [])
    ]
    has_measure = bool(measures or registered_metrics or (
        derived_measure.get("target_table") and derived_measure.get("target_column")
    ))
    temporal_requested = bool(
        intent_plan.get("time_range")
        or intent_plan.get("quarter_periods")
        or intent_plan.get("date_role") not in {None, "", "unresolved"}
        or temporal
    )
    source_required = bool(
        has_measure
        or temporal_requested
        or intent in {
            "comparison", "trend", "ranking", "distribution",
            "daily_snapshot", "metric_query", "causal_analysis",
        }
        or measure_semantics == "count_distinct_business_identifier"
    )
    missing_slots: list[str] = []
    if source_required and not source_facts:
        missing_slots.append("source_fact")
    if measure_semantics == "count_distinct_business_identifier" and not (
        derived_measure.get("target_table") and derived_measure.get("target_column")
    ):
        missing_slots.append("count_target")
    if intent in {"ranking", "trend", "comparison", "distribution", "metric_query"} and not has_measure:
        missing_slots.append("measure")
    if temporal_requested and not temporal:
        missing_slots.append("date_role")
    if intent == "comparison" and not str(intent_plan.get("comparison") or "").strip():
        missing_slots.append("comparison_window")

    has_any_binding = bool(source_facts or measures or dimensions or registered_metrics or derived_measure)
    status = "incomplete" if missing_slots else "compiled" if has_any_binding else "unresolved"
    compiled = {
        "version": 1,
        "question": str(question or ""),
        "status": status,
        "intent": intent,
        "confidence": intent_plan.get("confidence"),
        "missing_slots": list(dict.fromkeys(missing_slots)),
        "source_fact": selected_fact,
        "source_facts": source_facts,
        "subrequests": subrequests,
        "measures": measures,
        "dimensions": dimensions,
        "temporal_operations": temporal,
        "joins": [dict(j) for j in (plan.get("joins") or []) if j.get("enforcement") != "optional"],
        "metrics": registered_metrics,
        "derived_measure": derived_measure,
        "analytical_recipe": analytical_recipe,
        "filters": list(intent_plan.get("filters") or []),
        "time_range": str(intent_plan.get("time_range") or ""),
        "quarter_periods": list(intent_plan.get("quarter_periods") or []),
        "calendar_basis": str(intent_plan.get("calendar_basis") or ""),
        "comparison": str(intent_plan.get("comparison") or ""),
        "entity_grain": str(intent_plan.get("entity_grain") or ""),
        "top_n": intent_plan.get("top_n"),
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
    if plan.get("intent"):
        lines.append(f"- Analytical intent: {plan['intent']}")
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
    derived = plan.get("derived_measure") or {}
    if derived.get("semantics") == "count_distinct_business_identifier":
        entity = str(derived.get("business_entity") or "business event")
        target_table = str(derived.get("target_table") or "")
        target_column = str(derived.get("target_column") or "")
        if target_table and target_column:
            lines.append(
                f"- Exact derived measure: COUNT(DISTINCT {target_table}.{target_column}) "
                f"for {derived.get('business_name') or entity}. This exact table and "
                "column are authoritative."
            )
        else:
            lines.append(
                f"- Derived measure: COUNT(DISTINCT the governed stable {entity} business "
                "identifier)."
            )
        lines.append(
            "- Do not substitute revenue, amount, quantity, a display name, a line key, "
            "a row surrogate, or an unrelated registered metric."
        )
    recipe = plan.get("analytical_recipe") or {}
    if recipe.get("kind") == "period_over_period_entity_change":
        lines.append(
            "- Analytical recipe: calculate the exact governed distinct-event count "
            "for two equal, non-overlapping governed periods at the requested entity grain."
        )
        lines.append(
            "- Return current count, prior count, absolute change, and percentage change; "
            "do not compare amount, revenue, quantity, or row counts."
        )
        if recipe.get("direction") == "decrease":
            lines.append("- Keep entities whose current count is lower than their prior count.")
        elif recipe.get("direction") == "increase":
            lines.append("- Keep entities whose current count is higher than their prior count.")
    if plan.get("dimensions"):
        lines.append("- Output dimensions: " + ", ".join(
            f"{d.get('table')}.{d.get('column')}" for d in plan["dimensions"]
        ))
    if plan.get("time_range"):
        lines.append(f"- Requested time range: {plan['time_range']}")
    if plan.get("comparison"):
        lines.append(f"- Comparison operation: {plan['comparison']}")
    if plan.get("entity_grain"):
        lines.append(f"- Required business grain: {plan['entity_grain']}")
    if plan.get("top_n"):
        lines.append(f"- Final result limit: Top {plan['top_n']}")
    lines.append(f"- Required output shape: {plan.get('output_shape') or 'table'}")
    lines.append("This plan is authoritative. If a SQL draft conflicts with it, regenerate the SQL; do not reinterpret the question.")
    return "\n".join(lines)
