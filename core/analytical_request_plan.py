"""Compile a question's resolved semantics into one executable request plan."""

from __future__ import annotations

import logging
import re
from typing import Any

from core.analytical_intent import _singular
from core.semantic_model import _is_measure_binding

log = logging.getLogger("querybot.analytical_request_plan")


def _entity_physical_table(entity: dict[str, Any]) -> str:
    table = str(entity.get("table_name") or "").strip()
    if not table:
        return ""
    if "." in table:
        return table
    schema = str(entity.get("schema_name") or entity.get("schema") or "").strip()
    return f"{schema}.{table}" if schema else table


def _table_matches(left: Any, right: Any) -> bool:
    left_parts = [part for part in re.split(r"[.\[\]`]", str(left or "").upper()) if part]
    right_parts = [part for part in re.split(r"[.\[\]`]", str(right or "").upper()) if part]
    if not left_parts or not right_parts:
        return False
    return left_parts[-2:] == right_parts[-2:] or left_parts[-1] == right_parts[-1]


def _is_compiled_measure(field: dict[str, Any], selected_fact: str) -> bool:
    """Use generic attribute bindings as measures only on the chosen fact.

    Runtime semantic contracts can label an approved numeric fact field as an
    ``attribute``.  The same generic role can also describe a name or category
    on a dimension, so treating every non-key attribute as a measure would turn
    labels into aggregations.  Explicit measure roles remain authoritative;
    the fallback is restricted to the already governed source fact.
    """
    role = str(field.get("role") or "").strip().lower()
    if role in {"measure", "measure_candidate"}:
        return True
    return bool(
        selected_fact
        and _table_matches(field.get("table"), selected_fact)
        and _is_measure_binding(field)
    )
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


def counted_snapshot_key(model: dict[str, Any] | None, fact_table: str, dated_column: str) -> str:
    """The key of the snapshot a count of a periodic snapshot's records reads:
    each record repeats under every snapshot, and only its latest is counted.
    "" for any other table, and when the date the question names is the
    snapshot's own -- each snapshot's records are then counted in its period."""
    table = next((
        t for t in (model or {}).get("tables") or []
        if isinstance(t, dict) and _same_table(t.get("qualified_name") or t.get("table"), fact_table)
    ), None)
    if not table or str(table.get("fact_type") or "") != "periodic_snapshot":
        return ""
    key = next((
        str(role.get("fact_column") or "")
        for role in (model or {}).get("date_roles") or []
        if isinstance(role, dict) and role.get("is_default") and role.get("fact_column")
        and _same_table(role.get("fact_table"), fact_table)
    ), "")
    return "" if key.upper() == str(dated_column or "").upper() else key


# Words that name a table's rows as such, in either word order: "item-warehouse
# records", and the French "fiches article-entrepot" read as "records
# item-warehouse".
_RECORD_WORDS = frozenset({"record", "row", "line", "entry"})


def _counted_words(noun: str) -> set[str]:
    return {_singular(word) for word in re.split(r"[\s-]+", str(noun or "").casefold()) if word}


def _names_rows(noun: str) -> bool:
    return bool(_counted_words(noun) & _RECORD_WORDS)


def _record_key(fields: list[dict[str, Any]], noun: str, fact_table: str) -> str:
    """The fact's key for the records a count names: "item" is the item key
    the planner bound "items" through. "" for a noun no field is named for
    whole: "active items" are not all the items, and "item-warehouse records"
    are the table's rows."""
    for field in fields:
        if _counted_words(str(field.get("term") or "")) != _counted_words(noun):
            continue
        key = str(field.get("source_key_column") or "")
        source = str(field.get("source_key_table") or field.get("source_table") or "")
        if key and _same_table(source, fact_table):
            return key
    return ""


_RECORD_COUNT_REASON = "the records a date the question names"


def _names_the_counted(field: dict[str, Any], counted: set[str], intent_plan: dict[str, Any] | None) -> bool:
    """A field that names only the records a count counts: "items" in "items
    created by month". Never one the question groups by: "warehouse" in
    "item-warehouse records created in 2025 by warehouse"."""
    term = _counted_words(str(field.get("term") or ""))
    grouped = [_counted_words(value) for value in (intent_plan or {}).get("dimensions") or []]
    return bool(term) and term <= counted and term not in grouped and str(field.get("role") or "").lower() in {
        "dimension", "display_dimension", "attribute",
    }


def record_count_by_named_date(
    semantic_plan: dict[str, Any] | None,
    intent_plan: dict[str, Any] | None,
    matched_metrics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The records a date the question names, when that is what it counts.

    "Items created by month in 2025" counts the items whose record was created
    in each month: the question names the creation date, kept on the request's
    own fact, asks for no measure, and counts what the fact keeps -- an entity
    by its key, or the rows it names as records. {} for anything else: a
    metric, a measure, a date the question did not name or one on another
    table, and a noun the fact keeps no key of ("sales invoiced by month").
    """
    plan = semantic_plan or {}
    noun = str((intent_plan or {}).get("record_count") or "").strip()
    if not noun or matched_metrics:
        return {}
    fact = str((plan.get("source_scope") or {}).get("selected_fact") or plan.get("fact_anchor") or "")
    temporal = [p for p in plan.get("temporal_policies") or [] if isinstance(p, dict)]
    if not fact or len(temporal) != 1:
        return {}
    policy = temporal[0]
    if str(policy.get("resolution_source") or "") != "explicit_date_role" or not _same_table(
        policy.get("fact_table"), fact
    ):
        return {}
    if any(
        str(field.get("role") or "").lower() in {"measure", "measure_candidate"}
        and field.get("enforcement") != "optional"
        for field in plan.get("fields") or [] if isinstance(field, dict)
    ):
        return {}
    key = _record_key([field for field in plan.get("fields") or [] if isinstance(field, dict)], noun, fact)
    if not (key or _names_rows(noun)):
        return {}
    return {
        "noun": noun, "fact": fact, "key": key,
        "dated_by": str(policy.get("business_role") or ""),
        "dated_column": str(policy.get("fact_column") or ""),
    }


def demote_counted_records(
    semantic_plan: dict[str, Any] | None,
    intent_plan: dict[str, Any] | None,
    matched_metrics: list[dict[str, Any]] | None = None,
) -> None:
    """Leave the records a count names out of its breakdown, before the graph
    is resolved for the plan: the planner bound "items" in "items created by
    month" to the item's name, and "warehouse" in "item-warehouse records" to
    the warehouse's, and either would be grouped by and joined."""
    decision = record_count_by_named_date(semantic_plan, intent_plan, matched_metrics)
    if not decision:
        return
    plan = semantic_plan or {}
    words = _counted_words(decision["noun"])
    fields = [field for field in plan.get("fields") or [] if isinstance(field, dict)]
    counted = [field for field in fields if _names_the_counted(field, words, intent_plan)]
    for field in counted:
        field["enforcement"] = "optional"
        field["demotion_reason"] = "the records the question counts"
    # And the joins that reached only them.
    kept = {str(field.get("table") or "") for field in fields if field.get("enforcement") != "optional"}
    for join in plan.get("joins") or []:
        target = str(join.get("to") or join.get("to_table") or "")
        if any(_same_table(target, field.get("table")) for field in counted) and not any(
            _same_table(target, table) for table in kept
        ):
            join["enforcement"] = "optional"


def demote_counted_population(
    semantic_plan: dict[str, Any] | None,
    intent_plan: dict[str, Any] | None,
) -> None:
    """A population is counted on its own table, and is not also its
    breakdown: "how many warehouses do we have" bound "warehouses" to the
    warehouse's name, grouped by it, and joined the warehouse table from a
    fact that was never read -- the graph then required that join of a count
    that has none."""
    plan = semantic_plan or {}
    entity = str((intent_plan or {}).get("population_entity") or "")
    master = str((((plan.get("count_target") or {}).get("selected")) or {}).get("table") or "")
    if not entity or not master:
        return
    words = _counted_words(entity)
    for field in plan.get("fields") or []:
        if isinstance(field, dict) and _names_the_counted(field, words, intent_plan):
            field["enforcement"] = "optional"
            field["demotion_reason"] = "the population the question counts"
    for join in plan.get("joins") or []:
        if isinstance(join, dict) and not _same_table(join.get("from") or join.get("from_table"), master):
            join["enforcement"] = "optional"


def demote_what_the_question_does_not_compute(
    semantic_plan: dict[str, Any] | None,
    intent_plan: dict[str, Any] | None,
    matched_metrics: list[dict[str, Any]] | None = None,
    date_bindings: list[dict[str, Any]] | None = None,
) -> None:
    """Set aside what a question names but neither computes nor is cut by, in
    the one order that reads each right: a calendar's time attributes first,
    since while one is bound as a measure no count of records is read -- "items
    created by week of year" was answered as a sum of weeks by item -- then the
    records a count names, and the population it counts."""
    demote_calendar_measures(semantic_plan, date_bindings)
    demote_counted_records(semantic_plan, intent_plan, matched_metrics)
    demote_counted_population(semantic_plan, intent_plan)


def demote_calendar_measures(
    semantic_plan: dict[str, Any] | None,
    date_bindings: list[dict[str, Any]] | None,
) -> None:
    """A calendar's time attributes are how a question is cut, never what it
    adds up: "items created by week of year" bound the calendar's week number
    as a measure the answer had to compute, and the query was refused, where
    "by week of the year" -- which matched the column's name less well -- was
    answered. The attributes are the ones the chosen date's calendar was
    found to keep (its year, quarter, month, week, weekday and their names);
    a flag or a count the calendar keeps is not among them, and stays."""
    calendars = [
        (
            str((binding or {}).get("dimension_table") or ""),
            {
                str(column).upper()
                for attribute, column in ((binding or {}).get("calendar_attributes") or {}).items()
                if attribute != "date" and column
            },
        )
        for binding in date_bindings or []
    ]
    for field in (semantic_plan or {}).get("fields") or []:
        if (
            isinstance(field, dict)
            and field.get("role") == "measure"
            and field.get("enforcement") != "optional"
            and any(
                _same_table(field.get("table"), table) and str(field.get("column") or "").upper() in attributes
                for table, attributes in calendars
            )
        ):
            field["enforcement"] = "optional"
            field["demotion_reason"] = "a calendar's time attribute, not a measure"


def compile_analytical_request_plan(
    question: str,
    semantic_plan: dict[str, Any] | None,
    *,
    matched_metrics: list[dict[str, Any]] | None = None,
    analysis_contract: dict[str, Any] | None = None,
    graph_context: dict[str, Any] | None = None,
    analytical_intent_plan: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan = semantic_plan or {}
    source_scope = plan.get("source_scope") or {}
    selected_fact = str(source_scope.get("selected_fact") or plan.get("fact_anchor") or "")
    selected_facts = [
        str(value) for value in (source_scope.get("selected_facts") or []) if value
    ]
    fields = [f for f in (plan.get("fields") or []) if f.get("enforcement") != "optional"]

    # ── The measure decides the measure fact ─────────────────────────────────
    # `source_scope.selected_fact` is business-source arbitration: useful, but
    # lexical/heuristic evidence. It was winning over the compiled plan, and
    # this plan is what the validator enforces — so a wrong pick here rejects
    # correct SQL rather than just answering oddly.
    #
    # Live case: "what is the total amount of confirmed purchase orders by
    # profit center". Source arbitration scored CUS_ORD_IVC_FCT (customer
    # invoices) 33 on an approved-metric binding while the compiled plan
    # required PCH_ORD_RCT_FCT.PCH_ORD_LIN_CAD_AMT. This plan then declared
    # CUS_ORD_IVC_FCT the measure fact, the model generated correct SQL over
    # PCH_ORD_RCT_FCT, and source_fact_mismatch blocked the right answer.
    #
    # So a required measure field, or the governed fact anchor derived from one,
    # outranks arbitration. A source the USER explicitly chose still wins over
    # both — that is a governed decision, not a guess.
    _user_confirmed_source = (
        str(source_scope.get("reason") or "") == "user-confirmed governed source"
    )
    # A population count is anchored on the table that DEFINES the population.
    # There is no measure to arbitrate over, and letting an incidental numeric
    # column on that master claim the source would move the request off the
    # very table the governed count target lives on.
    _governed_master_source = (
        str(source_scope.get("source_kind") or "").casefold() == "master"
    )
    if not _user_confirmed_source and not _governed_master_source:
        # Governance strength, not list order: enforcement="required" is the
        # structured semantic model's approved binding, while an unset value is
        # the LLM field planner's suggestion — which is how "purchase" bound to
        # an inventory quantity on a purchase-order question. Ordering by list
        # position would let that suggestion claim the measure fact.
        _ranked_measures = sorted(
            (
                field for field in fields
                if str(field.get("role") or "").strip().lower()
                in {"measure", "measure_candidate"}
            ),
            key=lambda field: (
                0 if str(field.get("enforcement") or "").strip().lower() == "required"
                else 1
            ),
        )
        _governed_candidates = [
            str(plan.get("fact_anchor") or ""),
            *(str(field.get("table") or "") for field in _ranked_measures),
        ]
        for _candidate in _governed_candidates:
            if not _candidate or not selected_fact:
                continue
            if _table_matches(_candidate, selected_fact):
                break
            log.info(
                "Analytical request plan: business-source arbitration chose %s "
                "but the compiled plan's required measure lives on %s — "
                "anchoring the measure fact on %s",
                selected_fact, _candidate, _candidate,
            )
            selected_fact = _candidate
            selected_facts = [
                value for value in selected_facts
                if _table_matches(value, _candidate)
            ] or [_candidate]
            break

    metric_sources: list[str] = []
    for metric in matched_metrics or []:
        values = (
            metric.get("_resolved_source_tables")
            or metric.get("source_tables")
            or ([metric.get("base_table")] if metric.get("base_table") else [])
        )
        for value in values:
            table = str(value or "").strip()
            if table and table not in metric_sources:
                metric_sources.append(table)
    # A validated formula metric is authoritative source evidence even when
    # no physical measure field was emitted by the semantic field planner.
    if not selected_fact and len(metric_sources) == 1:
        selected_fact = metric_sources[0]
    if not selected_facts and selected_fact:
        selected_facts = [selected_fact]
    measures = [
        {"term": f.get("term"), "table": f.get("table"), "column": f.get("column")}
        for f in fields if _is_compiled_measure(f, selected_fact)
    ]
    dimensions = [
        {"term": f.get("term"), "table": f.get("table"), "column": f.get("column")}
        for f in fields if not _is_compiled_measure(f, selected_fact) and str(f.get("role") or "").lower() in {
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

    # The records a date the question names, counted. A periodic snapshot
    # repeats each record under every snapshot, so its latest snapshot alone
    # is counted.
    record_count = record_count_by_named_date(plan, intent_plan, matched_metrics)
    if record_count and not derived_measure and not measures:
        derived_measure = {
            "semantics": "count_distinct_business_identifier" if record_count["key"] else "count_records",
            "business_entity": record_count["noun"],
            "aggregation": "count_distinct" if record_count["key"] else "count",
            "target_table": selected_fact,
            "target_column": record_count["key"],
            "dated_by": record_count["dated_by"],
            "snapshot_column": counted_snapshot_key(model, selected_fact, record_count["dated_column"]),
            "resolution_reason": _RECORD_COUNT_REASON,
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
        {
            "id": m.get("id"),
            "name": m.get("name"),
            # The column is `sql_template`. Neither "formula" nor "sql_formula"
            # exists on a metric_registry row, so the compiled plan has been
            # carrying a null formula for every metric it ever described --
            # silently, because every consumer treats it as optional.
            "formula": m.get("sql_template") or m.get("formula") or m.get("sql_formula"),
            "source_tables": list(
                m.get("_resolved_source_tables")
                or m.get("source_tables")
                or ([m.get("base_table")] if m.get("base_table") else [])
            ),
        }
        for m in (matched_metrics or [])
    ]
    has_measure = bool(measures or registered_metrics or (
        derived_measure.get("target_table") and derived_measure.get("target_column")
    ) or derived_measure.get("semantics") == "count_records")
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
        # What the compiler chose BETWEEN, not just what it chose.
        #
        # Business-source arbitration weighs several facts and keeps one;
        # metric matching can resolve to several source tables; a question
        # with two temporal policies has two date roles it could anchor on.
        # All of that was computed and thrown away, so a plan that was a
        # coin-flip and a plan that was determined looked identical
        # downstream -- and the product presented both with the same
        # confidence. core.candidate_selection reads these to decide when a
        # question is worth more than one query.
        "considered_facts": sorted({
            str(value) for value in (
                list(selected_facts) + list(source_facts) + list(metric_sources)
            ) if value
        }),
        "considered_date_roles": sorted({
            str((policy or {}).get("date_role") or (policy or {}).get("column") or "")
            for policy in temporal
            if isinstance(policy, dict)
        } - {""}),
        "considered_metrics": sorted({
            str(m.get("name") or "") for m in registered_metrics if m.get("name")
        }),
        "source_kind": str(source_scope.get("source_kind") or "fact"),
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
    elif plan.get("source_fact") and str(plan.get("source_kind") or "").casefold() == "master":
        lines.append(
            f"- Single source table: {plan['source_fact']}. It defines the "
            "population being counted, so select from it without joining a fact."
        )
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
