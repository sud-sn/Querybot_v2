"""
core/query_pipeline.py
──────────────────────
Main query pipeline extracted from main.py.

Covers the full governed query pipeline, including metadata-only cached-result
follow-ups and governed source-query fallback.
"""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import replace as _dataclass_replace
import logging
import time

import store
from gateway import PlatformEvent
from core.llm import llm_complete, build_sql_system_prompt, resolve_provider
from core.examples import retrieve_similar_examples, format_examples_for_prompt
from core.clarification import (
    check_ambiguity_glossary_first, save_pending,
    build_schema_grounded_clarification_hint, extract_original_question,
    can_request_clarification, clarification_session_id,
    clarification_progress, prepare_clarification_meta,
    selected_clarification_option,
)
from core.analytical_intent import plan_analytical_intent
from core.analytical_request_plan import compile_analytical_request_plan
from core.schema import (
    load_known_tables,
    load_schema_columns,
    query_wait_timeout_seconds as _query_wait_timeout,
    run_query,
)
from core.knowledge import load_retriever
from core.validator import normalize_generated_sql, validate_sql
from core.query_semantics import (
    analyze_query_intent,
    build_generic_query_hints,
    detect_top_n_intent,
)
from core.graph_resolver import (
    resolve_for_question as _graph_resolve,
    date_role_entity_for_binding,
    entity_name_for_table,
    infer_connected_default_date_fact,
    is_date_role_entity,
)
from core.llm_audit import llm_audit_scope, make_llm_audit_request_id
from core.result_cache import result_cache
from core.query_router import should_route_to_result_cache, should_attempt_cache_followup
from core.result_regrain import (
    build_regrain_sql,
    parse_trend_regrain_request,
    regrain_question_text,
    resolve_regrain_grain,
    temporal_policy_from_plan,
)
from core.governed_result_followup import (
    adopt_cached_snapshot,
    contextualize_source_query_fallback,
    run_governed_result_followup,
)
from core.semantic_planner import build_semantic_field_plan
from core.source_resolution import resolve_source_scope, source_clarification_options
from core.count_target_resolver import (
    count_target_clarification_options,
    resolve_count_target,
    resolve_population_count_target,
)
from core.semantic_model import (
    build_runtime_semantic_context, build_runtime_semantic_plan,
    build_field_plan_repair_note,
)
from core.date_roles import (
    date_key_temporal_grain,
    question_has_temporal_intent,
    normalize_date_key_type,
)
from core.contextual_dates import (
    build_contextual_date_plan,
    build_contextual_date_plan_many,
    detect_temporal_window,
    enrich_date_binding_calendar_attributes,
    find_explicit_date_roles,
    question_has_snapshot_intent,
    requested_temporal_grain,
    resolve_contextual_date_binding,
)
from core.metric_scope import metric_source_tables, resolve_metric_scope
from core.answer_rca import extract_sql_tables
from core.pipeline_context import (
    get_state, get_client_db, _merge_semantic_plans,
    _scope_semantic_plan_to_analytical_request,
    check_query_limit, check_token_limit,
)
from core.pipeline_helpers import (
    _extract_kb_synonym_injection, _send_live_stage, _sql_preview,
    _count_tables_for_zero_row, _build_zero_row_message,
    _format_metric_formula_context, _extract_metric_formula_tables,
    _build_row_metric_join_sql, attempt_field_plan_repair,
    attempt_governed_temporal_metric_repair, compile_governed_temporal_metric_sql,
    _clamp_kb_doc, _clamp_prompt_context, reused_plan_is_stale_for_graph,
    reused_plan_semantic_staleness_code,
    allow_progressive_sql_repair,
)
from core.pipeline_trace import (
    _log_q, _trace_create, _trace_update, _trace_step, _trace_finish,
    _trace_finish_unclosed, _create_learning_candidate,
)
from core.result_renderer import (
    _send_results, _inject_distinct_if_needed,
)
from core.compliance.governed_query import (
    PolicyDeniedError, execute_governed_query,
)
from core.compliance.models import ResourceRef
from core.compliance.policy_engine import evaluate as evaluate_policy, resolve_context
from core.conversation_state import conversation_state_store
from core.semantic_plan_utils import required_semantic_tables

log = logging.getLogger("querybot")


def _calendar_profile_for_request(
    client_state: dict | None,
    db_cfg: dict | None,
    thread_preference: dict | None,
) -> dict:
    """Resolve calendar metadata without inventing a tenant default.

    An approved workspace setting wins over a thread preference. A fiscal
    start month alone is sufficient evidence that the workspace uses a
    fiscal calendar; otherwise a missing basis remains unresolved so the
    analytical planner can ask the user.
    """
    state = client_state if isinstance(client_state, dict) else {}
    database = db_cfg if isinstance(db_cfg, dict) else {}
    thread = thread_preference if isinstance(thread_preference, dict) else {}
    nested = state.get("calendar_profile")
    configured = dict(nested) if isinstance(nested, dict) else {}
    for key in (
        "basis",
        "calendar_basis",
        "calendar_mode",
        "fiscal_year_start_month",
    ):
        if configured.get(key) in (None, ""):
            value = state.get(key)
            if value in (None, ""):
                value = database.get(key)
            if value not in (None, ""):
                configured[key] = value

    configured_basis = str(
        configured.get("basis")
        or configured.get("calendar_basis")
        or configured.get("calendar_mode")
        or ""
    ).strip().casefold()
    if configured_basis in {"financial", "fiscal_year", "financial_year"}:
        configured_basis = "fiscal"
    try:
        configured_start = int(configured.get("fiscal_year_start_month") or 0)
    except (TypeError, ValueError):
        configured_start = 0
    if not 1 <= configured_start <= 12:
        configured_start = 0
    if configured_start and not configured_basis:
        configured_basis = "fiscal"

    profile = dict(thread)
    if configured_basis in {"calendar", "fiscal"}:
        profile["basis"] = configured_basis
        profile["source"] = "workspace_config"
    if configured_start:
        profile["fiscal_year_start_month"] = configured_start
        profile["source"] = "workspace_config"
    return profile


def _sql_completion_token_budget(
    question: str,
    semantic_plan: dict | None = None,
    *,
    retry: bool = False,
) -> int:
    """Return an output budget proportionate to the required SQL shape.

    A flat 512-token ceiling truncated legitimate period-comparison CTEs in
    production.  The parser then saw only the first half of the query and
    could not reach the semantic validators that would have corrected a wrong
    date role.  Output-token limits are ceilings (billing remains based on
    actual output), so give complex analytical shapes room to finish while
    keeping compact lookups bounded.
    """
    q = str(question or "").casefold()
    plan = semantic_plan or {}
    complex_terms = (
        "compare", "comparison", "versus", " vs ", "trend", "variance",
        "change", "growth", "contribution", "share", "running total",
        "moving average", "rank", "percentile", "current month",
        "last month", "previous month", "current year", "last year",
    )
    complex_shape = bool(
        plan.get("temporal_policies")
        and any(term in q for term in complex_terms)
    ) or len(plan.get("joins") or []) > 2
    if complex_shape:
        return 1536 if retry else 1280
    return 1024 if retry else 768


def _unknown_column_is_cross_schema(reason: str, schema_hint: str) -> bool:
    """Return whether an unknown column exists only outside the locked schema.

    Validator messages preserve human-readable casing. Keep parsing
    case-insensitive without mixing an uppercased value with a mixed-case
    delimiter, which previously raised ``IndexError`` in the SQL repair path.
    """
    if not schema_hint:
        return False
    _prefix, marker, suffix = (reason or "").casefold().partition(
        "exact column exists on"
    )
    return bool(marker) and schema_hint.casefold() not in suffix[:120]


def _entity_field_unavailable_reason(errors: list[dict] | None) -> str:
    """Explain when a missing qualified field exists only on other entities.

    Moving an exact column from one table to another is not a spelling repair:
    it can change the requested business entity (for example, prescriber state
    into pharmacy state).  Return a business-readable terminal explanation so
    the pipeline asks for a choice instead of executing a semantically different
    query. Unqualified columns and same-table spelling suggestions remain
    eligible for the existing repair paths.
    """

    def _bare_table(value: str) -> str:
        return str(value or "").strip().strip("[]\"`").split(".")[-1].upper()

    def _business_name(table: str) -> str:
        name = _bare_table(table)
        for prefix in ("FACT_", "DIM_", "BRIDGE_", "F_", "D_", "BR_"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        return name.replace("_", " ").strip().lower() or "requested table"

    conflicts: list[str] = []
    for error in errors or []:
        if str(error.get("code") or "") != "unknown_column":
            continue
        source = _bare_table(error.get("table") or "")
        column = str(error.get("column") or "").strip().upper()
        suggestions = [item for item in (error.get("suggestions") or []) if item]
        if not source or not column or suggestions:
            continue
        candidates: list[str] = []
        for candidate in error.get("candidate_tables") or []:
            bare = _bare_table(candidate)
            if bare and bare != source and bare not in candidates:
                candidates.append(bare)
        if not candidates:
            continue
        alternatives = ", ".join(
            f"{_business_name(candidate)} ({candidate})"
            for candidate in candidates[:4]
        )
        conflicts.append(
            f"The field {column} is not available for {_business_name(source)} "
            f"({source}). It is available for {alternatives}. I did not substitute "
            "one of those entities because that would change the meaning of your question."
        )
    return " ".join(conflicts[:3])


def _resolved_fact_tables(
    graph_context: dict,
    graph: dict,
    semantic_plan: dict | None = None,
    metric_tables: set[str] | None = None,
) -> set[str]:
    """Collect only physical fact tables already implicated by the question."""
    detected = set(graph_context.get("detected") or [])
    if graph_context.get("anchor"):
        detected.add(str(graph_context.get("anchor")))
    entities_by_name = {
        str(entity.get("entity_name") or ""): entity
        for entity in graph.get("entities") or []
    }

    def add_if_fact(table_ref: str, *, preserve_unknown: bool = False) -> None:
        if not table_ref:
            return
        entity_name = entity_name_for_table(graph, table_ref)
        entity = entities_by_name.get(entity_name, {})
        if str(entity.get("entity_type") or "").lower() == "fact":
            facts.add(table_ref)
        elif preserve_unknown and not entity_name:
            # A metric formula may reference a fact not yet represented in an
            # incomplete graph. Preserve that stronger governed signal.
            facts.add(table_ref)

    facts: set[str] = set()
    for table in metric_tables or set():
        add_if_fact(str(table), preserve_unknown=True)
    for entity in graph.get("entities") or []:
        if entity.get("entity_name") not in detected:
            continue
        if str(entity.get("entity_type") or "").lower() != "fact":
            continue
        table = str(entity.get("table_name") or "")
        schema = str(entity.get("schema_name") or "")
        if table:
            facts.add(f"{schema}.{table}" if schema else table)
    for field in (semantic_plan or {}).get("fields") or []:
        # A business-term field mapping (e.g. "revenue" -> F_RX_FILL.
        # NET_REVENUE_AMT) is just as strong a governed signal as a metric
        # formula's base table above -- same preserve_unknown rationale:
        # the current graph-resolution pass may have anchored on a
        # different table entirely (e.g. via unrelated KB-retrieval
        # scoring), which must not silently drop this table from the
        # fact-scope used for admin-approved default date-role lookup.
        source = str(field.get("source_table") or field.get("table") or "")
        add_if_fact(source, preserve_unknown=True)
    return {table for table in facts if table}


def _graph_with_exact_date_edges(graph: dict, bindings: list[dict]) -> dict:
    """Make the configured role-playing edge authoritative for graph planning."""
    relationships = list(graph.get("relationships") or [])
    remove_ids: set[int] = set()
    additions: list[dict] = []
    promoted_any = False
    for binding in bindings or []:
        if normalize_date_key_type(binding.get("date_key_type")) != "surrogate_fk":
            continue
        fact_entity = entity_name_for_table(graph, str(binding.get("fact_table") or ""))
        # Role-aware. When role-playing dates are modelled as separate entities
        # sharing one physical dimension, a table lookup returns an arbitrary
        # role — and this function would then rewrite THAT role's edge with this
        # binding's columns, silently corrupting the query-local graph. The
        # resolver already falls back to the sole owner of the dimension table,
        # so "" means genuinely ambiguous: leave the graph alone.
        dim_entity = date_role_entity_for_binding(graph, binding)
        fact_col = str(binding.get("fact_column") or "").upper()
        dim_col = str(binding.get("dimension_key") or "").upper()
        if not all((fact_entity, dim_entity, fact_col, dim_col)):
            continue
        parallel: list[tuple[int, dict]] = []
        exact_indexes: set[int] = set()
        for index, edge in enumerate(relationships):
            left = str(edge.get("from_entity") or "")
            right = str(edge.get("to_entity") or "")
            if {left, right} != {fact_entity, dim_entity}:
                continue
            parallel.append((index, edge))
            from_col = str(edge.get("from_column") or "").upper()
            to_col = str(edge.get("to_column") or "").upper()
            if (
                left == fact_entity and from_col == fact_col and to_col == dim_col
            ) or (
                right == fact_entity and from_col == dim_col and to_col == fact_col
            ):
                exact_indexes.add(index)
        if exact_indexes:
            remove_ids.update(index for index, _edge in parallel if index not in exact_indexes)
            # The persisted Entity Graph deliberately keeps discovered edges
            # in ``suggested`` state until an administrator reviews them.
            # Selecting an *approved* Date Role is already an independent,
            # stronger governance decision for this exact physical edge.  Make
            # that edge executable in the query-local graph without mutating
            # the persisted graph.  Previously an exact suggested edge was
            # retained verbatim, so resolve_for_question() removed it from the
            # confirmed subgraph and SQL generation never received the join
            # skeleton that validation subsequently required.
            for index in exact_indexes:
                promoted = dict(relationships[index])
                promoted.update({
                    "generated_by": "date_role",
                    "status": "confirmed",
                    "confidence_score": 100,
                })
                relationships[index] = promoted
                promoted_any = True
            continue

        # An approved date role is stronger evidence than a missing or stale
        # graph suggestion. Materialize its exact edge for this query only;
        # the persisted graph remains unchanged and can still be reviewed by
        # an administrator.
        if parallel:
            template = dict(parallel[0][1])
            forward = str(template.get("from_entity") or "") == fact_entity
            template.update({
                "from_entity": fact_entity if forward else dim_entity,
                "to_entity": dim_entity if forward else fact_entity,
                "from_column": fact_col if forward else dim_col,
                "to_column": dim_col if forward else fact_col,
                "join_type": template.get("join_type") or "LEFT",
                "generated_by": "date_role",
                "status": "confirmed",
                "validation_status": template.get("validation_status") or "untested",
                "confidence_score": 100,
            })
            remove_ids.update(index for index, _edge in parallel)
            additions.append(template)
        else:
            additions.append({
                "from_entity": fact_entity,
                "to_entity": dim_entity,
                "from_column": fact_col,
                "to_column": dim_col,
                "join_type": "LEFT",
                "relationship_type": "many_to_one",
                "generated_by": "date_role",
                "status": "confirmed",
                "validation_status": "untested",
                "confidence_score": 100,
            })
    if not remove_ids and not additions and not promoted_any:
        return graph
    return {
        **graph,
        "relationships": [
            edge for index, edge in enumerate(relationships) if index not in remove_ids
        ] + additions,
    }


def _table_matches_policy_scope(table: str, scope: set[str]) -> bool:
    table = table.upper()
    return any(
        table == candidate
        or table.endswith("." + candidate)
        or candidate.endswith("." + table)
        for candidate in scope
    )


def _graph_entities_for_verified_values(resolved: dict, graph: dict) -> set[str]:
    """Map filter values to graph entities by physical table.

    "narrowed" is included deliberately. That bucket holds a value we refused
    to SUBSTITUTE — "Calgary Distribution Centre East" is not "Calgary
    Distribution Centre" — but the near-miss still tells us which table the
    user is filtering on, and that is a separate question from which value to
    use. Leaving it out made the warehouse table undetectable, so a question
    naming a warehouse by value resolved to the fact alone and came back as
    "I couldn't find the right tables or columns", which is not what went
    wrong. Including it plans the query against the right table, filters on
    the user's own words, and lets the zero-row explanation name the closest
    real value.
    """
    table_refs: set[str] = set()
    for bucket in ("verified", "in_lists", "narrowed"):
        for item in (resolved or {}).get(bucket) or []:
            ref = str(item.get("table_fqn") or "").upper()
            ref = ref.replace("[", "").replace("]", "").replace('"', "").strip()
            if ref:
                table_refs.add(ref)

    matched: set[str] = set()
    for entity in (graph or {}).get("entities") or []:
        table = str(entity.get("table_name") or "").upper()
        schema = str(entity.get("schema_name") or "").upper()
        if not table:
            continue
        candidates = {table, f"{schema}.{table}" if schema else table}
        if any(
            ref in candidates
            or ref.endswith("." + table)
            or any(candidate and ref.endswith("." + candidate) for candidate in candidates)
            for ref in table_refs
        ):
            name = str(entity.get("entity_name") or "")
            if name:
                matched.add(name)
    return matched

def _format_insight_markdown(insight: dict) -> str:
    """Render an assistant_analysis payload as chat-channel markdown.
    The portal renders the raw payload natively (send_analysis_response);
    this is the fallback for adapters that only speak text."""
    parts: list[str] = []
    headline = (insight.get("headline") or "").strip()
    if headline:
        parts.append(f"*{headline}*")
    body = (insight.get("body") or "").strip()
    if body:
        parts.append(body)
    bullets = [str(b).strip() for b in (insight.get("bullets") or []) if str(b).strip()]
    if bullets:
        parts.append("\n".join(f"  • {b}" for b in bullets))
    next_step = (insight.get("next_step") or "").strip()
    if next_step:
        parts.append(f"_Next step: {next_step}_")
    return "\n\n".join(parts).strip()


async def _send_why_insight(
    adapter, event, *,
    question: str,
    rows: list,
    sql: str,
    client: dict,
    account_id: str,
    db_cfg: dict,
    rag_context: str = "",
    known_tables: set | None = None,
    query_executor=None,
    question_id: str = "",
) -> None:
    """Generate and send a causal analysis of `rows` after the factual answer.
    Best-effort: the factual answer is already on the wire, so any failure
    here is logged and swallowed — never surfaced as a user-facing error."""
    from core.compliance.policy_engine import result_llm_features_allowed
    if not result_llm_features_allowed(account_id):
        # Silent skip (matches this function's own failure-swallowing
        # contract) — but still leave a proof-of-refusal audit row, since
        # this is an auto-triggered second message the user never asked
        # for, not an action button click that deserves a visible reply.
        with llm_audit_scope(
            account_id=account_id,
            question=f"why: {question}"[:500],
            enabled=bool(client.get("enable_llm_audit")),
            request_id=make_llm_audit_request_id(),
            question_id=question_id,
            component="analysis",
        ):
            from core.llm_audit import record_llm_blocked
            record_llm_blocked(
                "analysis",
                "why-insight blocked — regulated tenant, LLM never received result rows.",
            )
        return
    try:
        from core.response_builder import generate_analysis_response
        provider, model, api_key, az_kwargs = resolve_provider(client, purpose="query")
        with llm_audit_scope(
            account_id=account_id,
            question=f"why: {question}"[:500],
            enabled=bool(client.get("enable_llm_audit")),
            request_id=make_llm_audit_request_id(),
            question_id=question_id,
            component="analysis",
        ):
            insight = await generate_analysis_response(
                action="why",
                rows=rows,
                question=question,
                provider=provider,
                model=model,
                api_key=api_key,
                account_id=account_id,
                follow_up=question,
                original_sql=sql,
                db_cfg=db_cfg,
                context=rag_context,
                known_tables=known_tables,
                query_executor=query_executor,
                **az_kwargs,
            )
        _send_analysis = getattr(adapter, "send_analysis_response", None)
        if callable(_send_analysis):
            await _send_analysis(event, insight)
            return
        text = _format_insight_markdown(insight)
        if text:
            await adapter.send_message(event, text)
    except Exception as exc:
        log.warning("Why-insight after factual answer failed (answer already sent): %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# Query pipeline — table-aware
# ══════════════════════════════════════════════════════════════════════════════

# How a date is physically carried, said in business words. Two roles on the
# same fact routinely share a business name ("Snapshot Date" stored both as a
# real date and as a YYYYMMDD code); this is what tells them apart on screen
# without naming a column, table, or key.
_DATE_KEY_QUALIFIERS = {
    "native_date": "calendar date",
    "timestamp": "timestamp",
    "yyyymmdd_integer": "date code",
    "yyyymm_integer": "month code",
    "surrogate_fk": "calendar reference",
}


def _date_option_identity(binding: dict) -> tuple[str, str]:
    """Physical identity of a date role, used to key and to order stably."""
    return (
        str(binding.get("fact_table") or "").upper(),
        str(binding.get("fact_column") or "").upper(),
    )


# Preference between physical encodings of ONE business date. A native date is
# safest to filter and anchor on; an encoded integer is the least safe.
_DATE_ENCODING_RANK = {
    "native_date": 60,
    "timestamp": 55,
    "surrogate_fk": 50,
    "yyyymmdd_integer": 40,
    "yyyymm_integer": 30,
    "date_string": 20,
}


def _date_option_business_identity(binding: dict) -> tuple[str, str, str]:
    """Business identity of a date role: same role, same fact, same grain.

    Two rows with this identity mean the same thing to a business user and
    carry the same effective join contract — they differ only in how the date
    is physically stored. Asking which storage format to use is not a
    meaningful clarification, so they collapse into one option.
    """
    role = " ".join(
        str(
            binding.get("context_name")
            or binding.get("date_role")
            or ""
        ).replace("_", " ").casefold().split()
    )
    return (
        str(binding.get("fact_table") or "").upper(),
        role,
        str(binding.get("temporal_grain") or "").casefold(),
    )


def _unique_date_bindings(bindings: list[dict]) -> list[dict]:
    """Collapse duplicate roles while keeping the strongest metadata.

    Two passes. The first removes exact physical duplicates (the same fact
    column offered twice). The second collapses rows that are the SAME business
    date on the same fact at the same grain but stored differently (a native
    SNAPSHOT_DATE beside an integer SNAPSHOT_YYYYMMDD): those would otherwise
    render as two buttons a business user cannot choose between — the defect
    behind "duplicate Last Modified Date options". Genuinely distinct roles
    (Invoice Date vs Order Date) always survive as separate options.
    """
    chosen: dict[tuple[str, str], dict] = {}

    def strength(item: dict) -> tuple[int, int, int, str]:
        status = str(item.get("governance_status") or item.get("status") or "").casefold()
        return (
            1 if status == "approved" else 0,
            1 if item.get("is_default") else 0,
            int(item.get("priority") or 0),
            str(item.get("context_name") or item.get("date_role") or ""),
        )

    for binding in bindings or []:
        identity = _date_option_identity(binding)
        if not any(identity):
            continue
        current = chosen.get(identity)
        if current is None or strength(binding) > strength(current):
            chosen[identity] = binding

    def encoding_strength(item: dict) -> tuple[int, int, int, int, str]:
        status = str(item.get("governance_status") or item.get("status") or "").casefold()
        return (
            1 if status == "approved" else 0,
            1 if item.get("is_default") else 0,
            _DATE_ENCODING_RANK.get(
                normalize_date_key_type(item.get("date_key_type")), 0
            ),
            int(item.get("priority") or 0),
            str(item.get("fact_column") or ""),
        )

    collapsed: dict[tuple[str, str, str], dict] = {}
    for identity in sorted(chosen):
        binding = chosen[identity]
        business = _date_option_business_identity(binding)
        # A row with no business role name cannot be judged equivalent to
        # anything; keep it keyed on its own physical identity.
        key = business if business[1] else (identity[0], identity[1], "")
        current = collapsed.get(key)
        if current is None or encoding_strength(binding) > encoding_strength(current):
            collapsed[key] = binding
    return [
        collapsed[key]
        for key in sorted(collapsed, key=lambda item: _date_option_identity(collapsed[item]))
    ]


def _date_option_labels(bindings: list[dict]) -> dict[tuple[str, str], str]:
    """Give every physical date role one stable, unique business label.

    Ambiguity here is normal, not exceptional: the same business date is
    often stored twice on a fact. The previous scheme appended a positional
    ordinal ("Snapshot Date (Day data 1)"), which had two faults -- it said
    nothing a user could choose on, and the number came from the order the
    bindings happened to arrive in, so the same menu entry could point at a
    different column on the next request.

    Disambiguate by *how the date is stored*, which is a property of the role
    itself and therefore identical on every render. Where that is still not
    enough, fall back to an ordinal derived from sorted physical identity so
    it is at least reproducible.
    """
    base_labels: dict[tuple[str, str], str] = {}
    for binding in bindings:
        identity = _date_option_identity(binding)
        if identity in base_labels:
            continue
        base_labels[identity] = str(
            binding.get("context_name")
            or binding.get("date_role")
            or "Business date"
        ).strip() or "Business date"

    qualifiers: dict[tuple[str, str], str] = {}
    for binding in bindings:
        identity = _date_option_identity(binding)
        qualifiers.setdefault(
            identity,
            _DATE_KEY_QUALIFIERS.get(
                normalize_date_key_type(binding.get("date_key_type")), ""
            ),
        )

    by_base: dict[str, list[tuple[str, str]]] = {}
    for identity, label in base_labels.items():
        by_base.setdefault(label.casefold(), []).append(identity)

    resolved: dict[tuple[str, str], str] = {}
    for identities in by_base.values():
        if len(identities) == 1:
            identity = identities[0]
            resolved[identity] = base_labels[identity]
            continue
        by_qualifier: dict[str, list[tuple[str, str]]] = {}
        for identity in identities:
            by_qualifier.setdefault(qualifiers.get(identity, ""), []).append(identity)
        for qualifier, same in by_qualifier.items():
            for identity in sorted(same):
                base = base_labels[identity]
                if qualifier and len(same) == 1:
                    resolved[identity] = f"{base} ({qualifier})"
                    continue
                # Two roles sharing a name AND a storage format are genuinely
                # different physical dates that the tenant's metadata gives no
                # business words to tell apart. Name the source field rather
                # than a bare ordinal: a positional number says nothing a user
                # can choose on, while the field name is at least actionable
                # (and tells an admin which role needs a distinct name).
                field = str(identity[1] or "").strip()
                if qualifier and field:
                    resolved[identity] = f"{base} ({qualifier} {field})"
                elif field:
                    resolved[identity] = f"{base} ({field})"
                else:
                    resolved[identity] = base
    return resolved


def _governed_date_anchor_repair_lines(
    semantic_plan: dict, db_type: str = "azure_sql"
) -> str:
    """Build the "REQUIRED ANCHOR" guidance shared by every repair path that
    can surface a broken governed date-role join -- temporal_anchor_* directly,
    and graph_plan_mismatch when the missing entity-graph edge IS the date-role
    join (naming the target column/edge alone isn't enough for the LLM to
    independently derive a correctly fact-scoped anchor subquery; that's the
    exact class of mistake this whole date-role system exists to prevent).
    """
    from core.contextual_dates import format_date_value_expression, format_required_anchor

    lines = []
    for policy in (semantic_plan or {}).get("temporal_policies") or []:
        fact_table = str(policy.get("fact_table") or "")
        fact_column = str(policy.get("fact_column") or "")
        date_table = str(policy.get("dimension_table") or policy.get("date_table") or "")
        date_key = str(policy.get("dimension_key") or "")
        date_column = str(policy.get("date_column") or "")
        if not (fact_table and fact_column and date_table and date_column):
            continue
        key_type = str(policy.get("date_key_type") or "")
        join_rule = (
            f"{fact_table}.{fact_column} = {date_table}.{date_key}"
            if date_key else "direct fact date/period (no surrogate join)"
        )
        date_expression = format_date_value_expression(
            date_table, date_column, key_type, db_type
        )
        lines.append(
            f"- JOIN/FIELD: {join_rule}; filter and anchor on "
            f"{date_expression}.\n"
            f"- REQUIRED ANCHOR (copy this exact subquery as the anchor; "
            f"do not build your own): {format_required_anchor(policy, db_type)}"
        )
    return "\n".join(lines)


async def _handle_query_impl(account_id, event, adapter, question, portal_user, is_clarification=False):
    start_ms = int(time.time() * 1000)
    # Set from every governed execution below. Row-level statistics (quartiles,
    # histogram bins, correlation, cohort matrices) are only meaningful over a
    # complete result, so the post-processing block consults this before
    # computing any of them.
    _rows_truncated = False
    state    = get_state(account_id)
    db_cfg   = get_client_db(account_id)
    client   = store.get_client(account_id) or {}

    # Activate the client's terminology packs for this request. Default (no
    # packs) equals the builtin vocabulary, so behavior is unchanged for
    # clients that have not enabled any pack. ContextVars do not cross
    # run_in_executor threads — anything vocab-dependent called through an
    # executor must take vocab= explicitly.
    from core.vocab_packs import vocab_for_account, activate_vocab
    _vocab = vocab_for_account(account_id)
    activate_vocab(_vocab)
    audit_enabled = bool(client.get("enable_llm_audit"))
    audit_request_id = make_llm_audit_request_id()

    # Channel-agnostic why-route: explicitly causal questions get the factual
    # answer first (normal pipeline below), then a causal analysis of the
    # fresh rows as a second message. Clarification replies are exempt — the
    # causal wording there belongs to the original question, already handled.
    from core.insight import is_causal_question
    _why_mode = bool(not is_clarification and is_causal_question(question))

    # Identity passed through every query-log row for audit + billing.
    pu_id  = portal_user.get("id") if portal_user else None
    zid    = event.user_id or ""
    trace_id = _trace_create(
        account_id=account_id,
        question_id=audit_request_id,
        question=question,
        portal_user_id=pu_id,
        platform_user_id=zid,
        session_id=getattr(adapter, "session_id", "") or "",
        request_source=getattr(event, "platform", "") or "",
    )
    _trace_step(trace_id, "receive_question", output_summary={"question_id": audit_request_id})

    def _save_pending_clarification(
        original_question: str,
        prompt_context: str,
        meta: dict | None,
    ) -> None:
        if not event.user_id:
            return
        source = str((meta or {}).get("source") or "clarification")
        save_pending(
            account_id,
            event.user_id,
            original_question,
            prompt_context,
            clarification_meta=prepare_clarification_meta(
                event,
                meta,
                source=source,
            ),
            session_id=clarification_session_id(adapter, event),
        )

    _event_raw = getattr(event, "raw", None)
    _confirmed_date_role = selected_clarification_option(
        event, "metric_date_context"
    )
    _confirmed_join_path = selected_clarification_option(event, "graph_join_path")
    _graph_resolution_question = question
    if _confirmed_join_path.get("label") or _confirmed_join_path.get("value"):
        _graph_resolution_question = (
            extract_original_question(question)
            + " "
            + str(
                _confirmed_join_path.get("label")
                or _confirmed_join_path.get("value")
            )
        ).strip()

    if not db_cfg:
        _trace_finish(trace_id, status="error", answer_type="error", error_message="No database assigned")
        await adapter.send_message(event, "⚠️ No database assigned. Contact your administrator.")
        return

    await _send_live_stage(adapter, event, "authorization", "Checking access", "Verifying workspace access and available data.")

    within_limit, used, limit = check_query_limit(account_id)
    if not within_limit:
        _trace_finish(trace_id, status="error", answer_type="error", error_message="Monthly query limit reached")
        await adapter.send_message(event, f"❌ Monthly query limit reached ({used}/{limit}).")
        return
    if used >= int(limit * 0.8):
        await adapter.send_message(event, f"⚠️ {used}/{limit} queries used this month.")

    within_token_limit, tokens_used, token_limit = check_token_limit(account_id)
    if not within_token_limit:
        _trace_finish(trace_id, status="error", answer_type="error", error_message="Monthly token limit reached")
        await adapter.send_message(event, f"❌ Monthly token limit reached ({tokens_used}/{token_limit}).")
        return
    if token_limit and tokens_used >= int(token_limit * 0.8):
        await adapter.send_message(event, f"⚠️ {tokens_used}/{token_limit} tokens used this month.")

    try:
        provider, model, api_key, az_kwargs = resolve_provider(client, purpose="query")
        _trace_update(trace_id, llm_provider=provider, llm_model=model, db_type=db_cfg.get("db_type", ""))
    except RuntimeError as e:
        _trace_finish(trace_id, status="error", answer_type="error", error_message=str(e))
        await adapter.send_message(event, f"⚠️ Config error: {e}")
        return

    # Determine this user's allowed tables
    # all_known      : every base table in the connected DB (authoritative).
    # allowed_tables : user's permitted subset (uppercase). None = admin.
    # effective      : intersection — what this user can actually see.
    allowed_tables = store.get_allowed_tables(portal_user) if portal_user else None
    all_known      = load_known_tables(state.get("schema_dir", ""))
    all_columns    = load_schema_columns(state.get("schema_dir", ""))

    if allowed_tables is None:
        effective = all_known  # admin — unrestricted
    else:
        allowed_tables = {t.upper() for t in allowed_tables}
        effective = {t for t in all_known if t in allowed_tables}

    # ── Schema scoping — applied when user selects a specific schema tab ──────
    # schema_hint comes from the portal chat schema selector (e.g. "HR").
    # We narrow effective and allowed_tables to only tables in that schema.
    # This scopes RAG retrieval, SQL generation, and validation to that schema.
    schema_hint = (getattr(event, "schema_hint", "") or "").upper().strip()
    if not schema_hint:
        # Teams and other chat channels do not expose the portal's schema
        # selector. Resolve the same scope automatically when the user can
        # access exactly one schema, keeping RAG and validation channel-neutral.
        _available_schemas = {
            parts[-2]
            for table in effective
            if len(parts := str(table or "").upper().split(".")) >= 2
        }
        if len(_available_schemas) == 1:
            schema_hint = next(iter(_available_schemas))
    if schema_hint:
        def _in_schema(fqn: str) -> bool:
            """True if the FQN's schema part matches the selected schema."""
            parts = fqn.upper().split(".")
            # DB.SCHEMA.TABLE → parts[-2] is schema
            # SCHEMA.TABLE    → parts[-2] is schema
            # TABLE           → no schema, keep it (can't filter)
            if len(parts) >= 2:
                return parts[-2] == schema_hint
            return True  # bare name — keep

        effective = {t for t in effective if _in_schema(t)}
        if allowed_tables is not None:
            allowed_tables = {t for t in allowed_tables if _in_schema(t)}

        if not effective:
            _trace_finish(
                trace_id,
                status="error",
                answer_type="error",
                error_message=f"No tables available in selected schema {schema_hint}",
            )
            await adapter.send_message(event,
                f"⚠️ No tables from the **{schema_hint}** schema are available to you. "
                f"Switch to a different schema or ask your administrator to grant access.")
            return

    _trace_update(
        trace_id,
        selected_schema=schema_hint,
        allowed_tables_snapshot=sorted(effective),
    )
    _trace_step(
        trace_id,
        "resolve_user_permissions",
        output_summary={"tables_available": len(effective), "schema": schema_hint or ""},
    )

    if portal_user and allowed_tables is not None and not effective:
        _trace_finish(trace_id, status="error", answer_type="error", error_message="No table access assigned")
        await adapter.send_message(event,
            "🔒 *No table access assigned.*\n\n"
            "Your account is not yet linked to any tables in this workspace. "
            "Please contact your administrator to request access before you "
            "can ask data questions.")
        return

    if not effective:
        _trace_finish(trace_id, status="error", answer_type="error", error_message="No tables available")
        await adapter.send_message(event,
            "⚠️ No tables are available to query. Contact your administrator.")
        return

    # ── Step 2.5: Tier-2 DuckDB routing — answer from cached result set ──────
    # When the user's question clearly refers to the already-returned data
    # ("who is below average?", "rank these", "show outliers"), run the query
    # against the in-memory DuckDB result cache instead of hitting the
    # production database.  Fast, private, supports full analytic SQL.
    # Structured analytical preflight. This plan contains only user text and
    # governed catalog names; no result rows are exposed. Materially missing
    # slots are clarified before retrieval/SQL work, while complete plans are
    # injected as a compact control block into the generation prompt below.
    _planner_session_id = str(getattr(adapter, "session_id", "") or "")
    _thread_calendar_preference = (
        conversation_state_store.get_calendar_preference(
            account_id,
            _planner_session_id,
        )
        if _planner_session_id
        else {}
    )
    _confirmed_count_target = selected_clarification_option(event, "count_target")
    _planner_calendar_profile = _calendar_profile_for_request(
        state,
        db_cfg,
        _thread_calendar_preference,
    )
    # Initialized before the try so later stages that re-plan (the trend
    # re-grain fallback below) always have these in scope, even when catalog
    # loading failed open.
    _planner_metrics: list[dict] = []
    _planner_terms: list[dict] = []
    try:
        _planner_metrics = []
        for _metric in store.list_metrics(account_id):
            _base_table = str(_metric.get("base_table") or "").upper().strip()
            if _base_table and not any(
                table == _base_table or table.endswith("." + _base_table)
                for table in effective
            ):
                continue
            _planner_metrics.append(_metric)
        _planner_terms = []
        for _term in store.list_terms(account_id):
            _term_tables = {
                value.strip().upper()
                for value in str(_term.get("tables_involved") or "").split(",")
                if value.strip()
            }
            if _term_tables and not any(
                any(table == allowed or table.endswith("." + allowed) for table in effective)
                for allowed in _term_tables
            ):
                continue
            _planner_terms.append(_term)
        _analytical_plan = plan_analytical_intent(
            question,
            metrics=_planner_metrics,
            terms=_planner_terms,
            calendar_profile=_planner_calendar_profile,
        )
    except Exception as _plan_exc:
        log.warning("Analytical intent planning failed open: %s", _plan_exc)
        _analytical_plan = plan_analytical_intent(
            question,
            calendar_profile=_planner_calendar_profile,
        )

    # Keep an explicit or clarified calendar choice available to later turns
    # in this thread. This stores only calendar metadata, never result values,
    # and does not change the tenant's approved configuration.
    if (
        _planner_session_id
        and _analytical_plan.calendar_basis in {"calendar", "fiscal"}
        and _analytical_plan.calendar_basis_source == "question"
    ):
        conversation_state_store.remember_calendar_preference(
            account_id,
            _planner_session_id,
            {
                "basis": _analytical_plan.calendar_basis,
                "fiscal_year_start_month": _analytical_plan.fiscal_year_start_month,
                "source": "user_confirmed",
            },
        )

    _analytical_plan_context = _analytical_plan.prompt_context()
    _trace_step(
        trace_id,
        "analytical_intent_plan",
        input_summary={"question": extract_original_question(question)},
        output_summary=_analytical_plan.to_dict(),
    )

    _planner_has_cached_result = bool(
        _planner_session_id and result_cache.has_result(_planner_session_id)
    )
    _calendar_slot_requires_answer = bool(
        _analytical_plan.clarification
        and _analytical_plan.clarification.slot
        in {"calendar_basis", "fiscal_year_start_month"}
    )
    if _analytical_plan.needs_clarification and (
        _calendar_slot_requires_answer or not _planner_has_cached_result
    ):
        _plan_clarification = _analytical_plan.clarification
        _plan_source = f"analytical_plan:{_plan_clarification.slot}"
        if can_request_clarification(event, _plan_source):
            _plan_options = [dict(option) for option in _plan_clarification.options]
            _plan_meta = {
                "source": _plan_source,
                "question": _plan_clarification.question,
                "options": _plan_options,
                "slot": _plan_clarification.slot,
                "plan": _analytical_plan.to_dict(),
            }
            if event.user_id:
                _save_pending_clarification(
                    question,
                    "",
                    _plan_meta,
                )
            send_prompt = getattr(adapter, "send_clarification_prompt", None)
            if callable(send_prompt) and _plan_options:
                await send_prompt(event, _plan_clarification.question, _plan_options)
            else:
                await adapter.send_message(event, _plan_clarification.question)
            _trace_finish(
                trace_id,
                status="success",
                answer_type="clarification",
                final_answer_summary=f"Requested analytical slot: {_plan_clarification.slot}",
                duration_ms=int(time.time() * 1000) - start_ms,
            )
            return

        _round_count, _max_rounds, _ = clarification_progress(event)
        if _round_count >= _max_rounds:
            _trace_finish(
                trace_id,
                status="error",
                answer_type="clarification_limit",
                error_message=f"Unresolved analytical slot: {_plan_clarification.slot}",
                duration_ms=int(time.time() * 1000) - start_ms,
            )
            await adapter.send_message(
                event,
                "I still do not have enough governed context to answer accurately. "
                f"Please restate the request and specify {_plan_clarification.slot.replace('_', ' ')}.",
            )
            return

    compliance_profile = store.get_compliance_profile(account_id)

    # ── Front-door PII scrub (regulated tenants) ─────────────────────────────
    # Everything below this line may embed `question` into an LLM prompt or an
    # embedding call (SQL generation, RAG retrieval, repair retries). Schema
    # samples and result rows are masked elsewhere, but the user can type PHI
    # directly ("why was John Smith's claim denied?") — scrub it here, at the
    # single point every handle_query caller flows through. The dispatcher
    # scrubs before its off-topic classifier too; this call is idempotent so
    # double-scrubbing is harmless, and this one also covers the direct
    # handle_query callers in gateway/webhooks.py that bypass the dispatcher.
    # The trace row above already captured the original text (tenant-local,
    # never sent to the LLM), preserving audit evidence of what was removed.
    if compliance_profile.get("mode") == "regulated":
        from core.masking import scrub_question_pii
        question, _scrubbed = scrub_question_pii(
            question, compliance_profile.get("industry", "")
        )
        if _scrubbed:
            _trace_step(
                trace_id,
                "question_pii_scrub",
                output_summary={"scrubbed_question": question},
            )
            await adapter.send_message(
                event,
                "ℹ️ Personal identifiers in your question were removed before "
                "processing, per this workspace's data policy. Results may be "
                "less specific — try filtering by an ID or category instead "
                "of a person's name or contact details.",
            )

    compliance_context = resolve_context(
        account_id,
        portal_user,
        action="query_execution",
        channel=getattr(event, "platform", "") or "portal",
        purpose_id=getattr(event, "purpose_id", "") or "",
        provider=provider,
        break_glass_grant_id=getattr(event, "break_glass_grant_id", None),
    )

    def _execute_with_policy(candidate_sql: str, semantic: dict | None = None):
        context = resolve_context(
            account_id,
            portal_user,
            action="query_execution",
            channel=getattr(event, "platform", "") or "portal",
            purpose_id=compliance_context.purpose_id,
            provider=provider,
            break_glass_grant_id=compliance_context.break_glass_grant_id,
        )
        return execute_governed_query(
            db_cfg["credentials"],
            db_cfg["db_type"],
            candidate_sql,
            context=context,
            known_tables=all_known,
            table_columns=all_columns,
            allowed_tables=effective,
            semantic_context=semantic,
        )

    if compliance_profile.get("mode") == "regulated":
        classification_map = store.get_classification_map(account_id)
        scoped_resources = []
        for key in classification_map:
            table, _, column = key.rpartition(".")
            if _table_matches_policy_scope(table, effective):
                scoped_resources.append(ResourceRef(table=table, column=column))
        llm_context = resolve_context(
            account_id,
            portal_user,
            action="llm_context",
            channel=getattr(event, "platform", "") or "portal",
            purpose_id=compliance_context.purpose_id,
            provider=provider,
            break_glass_grant_id=compliance_context.break_glass_grant_id,
        )
        llm_decision = evaluate_policy(llm_context, scoped_resources)
        _trace_step(
            trace_id,
            "regulated_llm_context",
            output_summary={
                "allowed": llm_decision.effective_allowed,
                "reason": llm_decision.reason_code,
                "policy_version": llm_decision.policy_version,
            },
            status="success" if llm_decision.effective_allowed else "error",
        )
        if not llm_decision.effective_allowed:
            _trace_finish(
                trace_id,
                status="error",
                answer_type="policy_denied",
                error_message=llm_decision.explanation,
            )
            await adapter.send_message(
                event,
                "This request is blocked by the workspace data policy. "
                f"Reason: {llm_decision.explanation}",
            )
            return

    # Sprint 3e — cheap contract-version stamp, fetched early so it's
    # available to the result-cache staleness check right below and to
    # every _send_results() call site further down (the duckdb-cache and
    # metric-registry routes both return before the full contract load
    # later in this function). load_contract() is mtime-cached, so this
    # costs one extra stat() call, not a re-parse.
    from core.semantic_contract import load_contract as _load_contract_early
    _contract_version = (
        _load_contract_early(state.get("kb_dir", "")).get("meta") or {}
    ).get("contract_version", "")

    _session_id = getattr(adapter, "session_id", None)
    _cached_cols = [s["name"] for s in result_cache.get_schema(_session_id)] if _session_id else []
    if _session_id and result_cache.has_result(_session_id):
        cache_context = resolve_context(
            account_id,
            portal_user,
            action="cache_read",
            channel=getattr(event, "platform", "") or "portal",
            purpose_id=compliance_context.purpose_id,
            provider=provider,
        )
        cache_decision = evaluate_policy(cache_context, [])
        if not cache_decision.effective_allowed:
            result_cache.clear(_session_id)
        else:
            _cached_contract_version = result_cache.get_contract_version(_session_id)
            if (
                _cached_contract_version and _contract_version
                and _cached_contract_version != _contract_version
            ):
                # The semantic contract was recompiled since these rows were
                # cached (a metric/field/join changed meaning) — stale rows
                # must not silently answer a follow-up under new semantics.
                _trace_step(
                    trace_id, "result_cache_stale",
                    output_summary={
                        "cached_version": _cached_contract_version,
                        "current_version": _contract_version,
                    },
                )
                result_cache.clear(_session_id)
            elif compliance_profile.get("mode") == "regulated":
                _trace_step(
                    trace_id,
                    "regulated_cache_read",
                    output_summary={"reason": cache_decision.reason_code},
                )
    _has_cached_result = bool(_session_id and result_cache.has_result(_session_id))
    _regex_routes_to_cache = bool(
        _session_id
        and should_route_to_result_cache(
            question,
            _has_cached_result,
            cached_col_names=_cached_cols,
        )
    )
    # should_attempt_cache_followup (core/query_router.py) additionally gives
    # the metadata-only LLM planner a second opinion for phrasings the regex
    # gate above misses ("drill into North") whenever a cached result is
    # active AND the question shows a positive sign of naming something
    # already on screen -- see its docstring for why this is safe to widen
    # (never fires on a fresh session; the planner's own "unsupported"
    # fallback is unchanged for genuinely new questions). Cached rows are
    # only fetched lazily, when the regex gate didn't already resolve this,
    # to avoid the extra snapshot read on the common fast path.
    _cached_rows_for_gate = (
        list(result_cache.get_snapshot(_session_id).get("rows") or [])
        if (_has_cached_result and not _regex_routes_to_cache)
        else None
    )
    _route_to_cached_result = bool(
        _session_id
        and should_attempt_cache_followup(
            question, _has_cached_result,
            cached_col_names=_cached_cols,
            cached_rows=_cached_rows_for_gate,
        )
    )
    if _route_to_cached_result:
        _trace_update(trace_id, route="governed_result_cache")
        _trace_step(
            trace_id, "route",
            output_summary="governed_result_cache" if _regex_routes_to_cache else "governed_result_cache_second_opinion",
        )
        await _send_live_stage(
            adapter,
            event,
            "retrieving_context",
            "Analysing results",
            "Running a governed analysis on the previously returned data.",
        )

        _cache_provider, _cache_model, _cache_key, _cache_az = resolve_provider(
            client, purpose="query"
        )

        async def _complete_cache_plan(**kwargs):
            return await llm_complete(
                provider=_cache_provider,
                model=_cache_model,
                api_key=_cache_key,
                **kwargs,
                **_cache_az,
            )

        _planner_request_id = make_llm_audit_request_id()
        with llm_audit_scope(
            account_id=account_id,
            question="Plan a cached-result analysis from metadata",
            enabled=bool(client.get("enable_llm_audit")),
            request_id=_planner_request_id,
            question_id=audit_request_id,
            component="result_metadata_planner",
        ):
            _cache_followup = await run_governed_result_followup(
                question,
                _session_id,
                complete=_complete_cache_plan,
                source_result_id=getattr(adapter, "last_result_id", None),
                is_clarification=is_clarification,
            )

        _trace_step(
            trace_id,
            "governed_result_followup",
            output_summary={
                "status": _cache_followup.status,
                **_cache_followup.evidence,
            },
            status="success" if _cache_followup.executed else "error",
        )

        if _cache_followup.status == "clarification" and _cache_followup.outcome is not None:
            _clarification_outcome = _cache_followup.outcome
            _clarification_options = list(
                _clarification_outcome.clarification_options or []
            )
            _clarification_prompt = (
                _clarification_outcome.clarification_prompt
                or "Which result value did you mean?"
            )
            if event.user_id:
                _save_pending_clarification(
                    question,
                    context,
                    {
                        "source": "governed_result_cache",
                        "question": _clarification_prompt,
                        "options": _clarification_options,
                        "source_result_id": getattr(adapter, "last_result_id", None),
                    },
                )
            send_prompt = getattr(adapter, "send_clarification_prompt", None)
            if callable(send_prompt):
                await send_prompt(
                    event,
                    _clarification_prompt,
                    _clarification_options,
                )
            else:
                option_lines = "\n".join(
                    f"- {option.get('label', '')}"
                    for option in _clarification_options
                )
                await adapter.send_message(
                    event,
                    f"{_clarification_prompt}\n\n{option_lines}",
                )
            _trace_finish(
                trace_id,
                status="success",
                answer_type="clarification",
                final_answer_summary="Requested clarification for a cached-result operation",
                duration_ms=int(time.time() * 1000) - start_ms,
            )
            return

        if _cache_followup.executed and _cache_followup.outcome is not None:
            _cache_outcome = _cache_followup.outcome
            _cache_snapshot = _cache_outcome.snapshot
            _cache_rows = list(_cache_snapshot.get("rows") or [])
            _cache_sql = str(_cache_snapshot.get("sql") or "")
            _cache_formats = dict(_cache_snapshot.get("column_formats") or {})
            _cache_duration = int(time.time() * 1000) - start_ms

            adopt_cached_snapshot(
                adapter,
                _cache_snapshot,
                question_id=audit_request_id,
            )

            _log_q(
                account_id, question, _cache_sql, len(_cache_rows), True, "",
                "governed_result_cache", "duckdb", 0, 0, _cache_duration,
                portal_user_id=pu_id, zoom_user_id=zid,
                question_id=audit_request_id,
            )
            await _send_results(
                event,
                adapter,
                question,
                _cache_rows,
                _cache_sql,
                _cache_duration,
                portal_user,
                account_id,
                db_cfg,
                question_id=audit_request_id,
                display_context=dict(_cache_snapshot.get("metadata") or {}),
                explicit_column_formats=_cache_formats,
                contract_version=_contract_version,
                cache_result=False,
            )
            _trace_finish(
                trace_id,
                status="success",
                answer_type="table",
                row_count=len(_cache_rows),
                duration_ms=_cache_duration,
                final_answer_summary=(
                    "Answered from the governed session cache; no result values were sent to the model"
                ),
            )
            return

        if _cache_followup.status == "blocked":
            _trace_finish(
                trace_id,
                status="error",
                answer_type="cache_transform_blocked",
                error_message=_cache_followup.reason,
            )
            await adapter.send_message(
                event,
                "I could not safely apply that operation to the cached result. "
                "Use an exact result column name or a row number. No cached values "
                "were sent to the model or source database.",
            )
            return

        if _cache_followup.status in {"error", "missing"}:
            _trace_finish(
                trace_id,
                status="error",
                answer_type="cache_transform_error",
                error_message=_cache_followup.reason,
            )
            await adapter.send_message(
                event,
                _cache_followup.reason
                or "The cached result could not be updated. Run the business question again.",
            )
            return
        else:
            _trace_step(
                trace_id,
                "governed_result_fallback",
                output_summary={
                    "reason": _cache_followup.reason,
                    "cached_values_forwarded": False,
                },
            )
            _source_result_id = getattr(adapter, "last_result_id", None)
            _contextualized_question = contextualize_source_query_fallback(
                question,
                _session_id,
                source_result_id=_source_result_id,
            )
            if _contextualized_question != question:
                question = _contextualized_question
                # The first analytical plan was built before cache routing,
                # when the turn contained only the short follow-up. Rebuild it
                # from the inherited source intent so metric, window, grain,
                # and comparison semantics stay aligned with the re-query.
                try:
                    _analytical_plan = plan_analytical_intent(
                        question,
                        metrics=_planner_metrics,
                        terms=_planner_terms,
                        calendar_profile=_planner_calendar_profile,
                    )
                except Exception as _lineage_plan_exc:
                    log.warning(
                        "Source-query lineage planning failed open: %s",
                        _lineage_plan_exc,
                    )
                    _analytical_plan = plan_analytical_intent(
                        question,
                        calendar_profile=_planner_calendar_profile,
                    )
                _analytical_plan_context = _analytical_plan.prompt_context()
                _trace_step(
                    trace_id,
                    "source_query_lineage",
                    output_summary={
                        "source_result_id": str(_source_result_id or ""),
                        "parent_question_preserved": True,
                        "analytical_plan_rebuilt": True,
                        "cached_values_forwarded": False,
                    },
                )
                log.info(
                    "Source-query fallback preserved governed parent intent "
                    "for result %s",
                    str(_source_result_id or "")[:16],
                )

    # Unsupported cache requests continue through the governed source-query pipeline.

    # ── Step 2.6: Trend re-grain of the parent answer ────────────────────────
    # "What was my revenue for the past 5 days?" answers with one total. The
    # follow-up "provide the trend" is neither a question about those rows (a
    # total cannot be un-aggregated, so the result cache has nothing to give)
    # nor a new question (three words carry no metric, so re-deriving the
    # metric, the business date and the window loses all three). It is the SAME
    # query at a finer grain.
    #
    # So compile it that way: take the parent's already-validated SQL, add the
    # parent's OWN governed business date to the SELECT and GROUP BY, order it
    # chronologically, and leave everything else — crucially the WHERE clause,
    # and therefore the window — untouched. The daily series then always sums
    # back to the total the user is looking at, and the date can only be the
    # role that answer was already governed by. No LLM is involved.
    _regrain_request = (
        parse_trend_regrain_request(question)
        if (_session_id and _has_cached_result and not is_clarification)
        else None
    )
    if _regrain_request:
        _regrain_snapshot = result_cache.get_snapshot(
            _session_id, getattr(adapter, "last_result_id", None)
        ) or result_cache.get_snapshot(_session_id)
        _regrain_parent_sql = str(_regrain_snapshot.get("sql") or "")
        _regrain_parent_question = str(_regrain_snapshot.get("question") or "")
        _regrain_policy = temporal_policy_from_plan(
            (_regrain_snapshot.get("metadata") or {}).get("semantic_plan") or {}
        )
        _regrain_grain = resolve_regrain_grain(_regrain_request, _regrain_policy)
        _regrain_sql, _regrain_refusal = build_regrain_sql(
            _regrain_parent_sql,
            _regrain_policy,
            _regrain_grain,
            db_cfg.get("db_type", "azure_sql"),
        )
        _regrain_question = regrain_question_text(
            _regrain_parent_question, _regrain_grain
        )
        _trace_step(
            trace_id,
            "trend_regrain",
            input_summary={
                "follow_up": _regrain_request.matched_phrase,
                "parent_question": _regrain_parent_question,
            },
            output_summary={
                "grain": _regrain_grain,
                "business_role": _regrain_policy.get("business_role") or "",
                "compiled": bool(_regrain_sql),
                "reason": _regrain_refusal,
            },
            status="success" if _regrain_sql else "error",
        )
        log.info(
            "Trend re-grain for %s: grain=%s role=%s parent=%r compiled=%s%s",
            account_id, _regrain_grain,
            _regrain_policy.get("business_role") or "",
            _regrain_parent_question[:60], bool(_regrain_sql),
            "" if _regrain_sql else f" reason={_regrain_refusal}",
        )
        if _regrain_sql:
            _trace_update(trace_id, route="trend_regrain", generated_sql=_regrain_sql)
            _trace_step(
                trace_id, "route",
                output_summary={"route": "trend_regrain", "grain": _regrain_grain},
            )
            await _send_live_stage(
                adapter, event, "executing_query", "Building the trend",
                "Re-running the previous answer's governed query, grouped by its "
                "approved business date.",
            )
            _regrain_t0 = time.time()
            try:
                _loop = asyncio.get_running_loop()
                try:
                    governed = await asyncio.wait_for(
                        _loop.run_in_executor(
                            None, _execute_with_policy, _regrain_sql,
                        ),
                        timeout=_query_wait_timeout(db_cfg),
                    )
                except asyncio.TimeoutError:
                    await adapter.send_message(
                        event,
                        "⏱ The trend query timed out after 3 minutes. Try a "
                        "narrower window or a coarser period.",
                    )
                    _trace_finish(
                        trace_id, status="error", answer_type="timeout",
                        error_message="Trend re-grain timed out",
                    )
                    return
                rows = governed.rows
                _rows_truncated = bool(getattr(governed, "truncated", False))
                _regrain_sql = governed.sql
                duration_ms = int(time.time() * 1000) - start_ms
            except PolicyDeniedError as policy_error:
                _trace_finish(
                    trace_id, status="error", answer_type="policy_denied",
                    error_message=str(policy_error),
                )
                await adapter.send_message(event, str(policy_error))
                return
            except Exception as _regrain_exc:
                # Fall through to the fallback below rather than failing the
                # turn: the restated parent question still carries the parent's
                # metric, business date and window.
                log.warning(
                    "Trend re-grain execution failed for %s (%s); "
                    "falling back to the governed pipeline",
                    account_id, _regrain_exc,
                )
            else:
                # The query has already run against the production database, so
                # bookkeeping must never be able to discard the answer or send
                # the turn back through the pipeline for a second execution.
                try:
                    _log_q(account_id, _regrain_question, _regrain_sql, len(rows), True, "",
                           "trend_regrain", "deterministic", 0, 0, duration_ms,
                           portal_user_id=pu_id, zoom_user_id=zid,
                           question_id=audit_request_id)
                    _trace_update(
                        trace_id,
                        sql_validation_status="derived_from_validated_parent",
                        query_row_count=len(rows),
                        query_duration_ms=duration_ms,
                    )
                    _trace_step(
                        trace_id, "execute_sql", input_summary=_regrain_sql,
                        output_summary={"rows": len(rows)},
                        duration_ms=int((time.time() - _regrain_t0) * 1000),
                    )
                    _add_history = getattr(adapter, "add_to_history", None)
                    if callable(_add_history) and rows:
                        _add_history(
                            question=_regrain_question,
                            sql=_regrain_sql,
                            columns=list(rows[0].keys()) if rows else [],
                            row_count=len(rows),
                        )
                except Exception as _regrain_log_exc:
                    log.warning(
                        "Trend re-grain bookkeeping failed for %s (%s) — the "
                        "answer below was still produced and delivered",
                        account_id, _regrain_log_exc,
                    )
                await _send_results(
                    event, adapter, _regrain_question, rows, _regrain_sql,
                    duration_ms, portal_user, account_id, db_cfg,
                    question_id=audit_request_id,
                    confidence_context={
                        # The parent SQL was validated before it ever executed;
                        # this query is that SQL plus one governed date column.
                        "validation_code": "derived_from_validated_parent",
                        "has_semantic_plan": True,
                        # Carry the parent's governed plan onto this answer so a
                        # further re-grain ("now by week") resolves against the
                        # same date role instead of losing it one hop later.
                        "semantic_plan": dict(
                            (_regrain_snapshot.get("metadata") or {}).get(
                                "semantic_plan"
                            ) or {}
                        ),
                        "tables_used": extract_sql_tables(
                            _regrain_sql, db_cfg.get("db_type", "azure_sql"),
                        ),
                    },
                    display_context={
                        **dict(_regrain_snapshot.get("metadata") or {}).get(
                            "display_formats", {}
                        ),
                        "format_scope": "trend_regrain",
                    },
                    contract_version=_contract_version,
                )
                _trace_finish(
                    trace_id, status="success", answer_type="table",
                    row_count=len(rows), duration_ms=duration_ms,
                    final_answer_summary=(
                        f"Re-grained the previous answer by {_regrain_grain}"
                    ),
                )
                return
        # Not compilable (or execution failed). Restate the PARENT question at
        # the requested grain so the normal pipeline resolves the same metric,
        # the same approved date role and the same window — rather than trying
        # to answer "provide the trend" on its own, which carries none of them.
        if _regrain_parent_question:
            log.info(
                "Trend re-grain fallback for %s: answering %r through the "
                "governed pipeline", account_id, _regrain_question,
            )
            question = _regrain_question
            # Re-plan against the question we are actually going to answer.
            # The plan built from "provide the trend" describes nothing, and
            # every stage below reads it (counted entity, calendar basis,
            # prompt context).
            try:
                _analytical_plan = plan_analytical_intent(
                    question,
                    metrics=_planner_metrics,
                    terms=_planner_terms,
                    calendar_profile=_planner_calendar_profile,
                )
                _analytical_plan_context = _analytical_plan.prompt_context()
            except Exception as _regrain_plan_exc:
                log.warning(
                    "Trend re-grain re-planning failed for %s: %s",
                    account_id, _regrain_plan_exc,
                )

    # ── Step 3: Metric registry — deterministic SQL for known metrics ────────
    # If the question matches a defined metric, assemble SQL without the LLM.
    # Downstream query scope. For unrestricted admins this remains None unless
    # they explicitly select a schema tab; then the selected schema must also
    # constrain retrieval, prompt grounding, validation, and repair.
    query_scope_tables = effective if (allowed_tables is not None or schema_hint) else None

    matched_metric = store.match_metric(account_id, question)
    if matched_metric:
        _trace_update(trace_id, route="metric_registry", generated_sql=matched_metric["sql_template"].strip())
        _trace_step(trace_id, "route", output_summary={"route": "metric_registry", "metric": matched_metric.get("name", "")})
        await _send_live_stage(adapter, event, "metric_registry", "Using known metric", "Found a trusted metric definition for this question.")
        sql_from_metric = matched_metric["sql_template"].strip()
        # Warn when user asks for a dimensional breakdown of a query-type metric
        import re as _re_grp
        if matched_metric.get("formula_type") == "query" and _re_grp.search(r'\b(by|per|for each|grouped by|split by|breakdown)\b', question, _re_grp.IGNORECASE):
            await adapter.send_message(
                event,
                f"ℹ️ **{matched_metric.get('label') or matched_metric.get('name', 'This metric')}** is a fixed SQL query — "
                "it returns an overall value and cannot be broken down by individual dimensions. "
                "Showing the overall result:",
            )
        log.info("Metric registry hit: %s → %s", matched_metric["name"], sql_from_metric[:60])
        _metric_exec_t0 = time.time()
        try:
            await _send_live_stage(adapter, event, "executing_query", "Running query", "Executing the trusted metric query against your database.")
            _loop = asyncio.get_running_loop()
            try:
                governed = await asyncio.wait_for(
                    _loop.run_in_executor(None, _execute_with_policy, sql_from_metric),
                    timeout=_query_wait_timeout(db_cfg),
                )
            except asyncio.TimeoutError:
                await adapter.send_message(
                    event,
                    "⏱ Query timed out after 3 minutes. Try adding a filter (e.g. date range or specific customer) to narrow the result.",
                )
                _trace_finish(trace_id, status="error", answer_type="timeout", error_message="Metric query timed out after 60s")
                return
            rows = governed.rows
            _rows_truncated = bool(getattr(governed, "truncated", False))
            sql_from_metric = governed.sql
            duration_ms = int(time.time()*1000) - start_ms
            _log_q(account_id, question, sql_from_metric, len(rows), True, "",
                   "metric_registry", "deterministic", 0, 0, duration_ms,
                   portal_user_id=pu_id, zoom_user_id=zid,
                   question_id=audit_request_id)
            _trace_update(
                trace_id,
                sql_validation_status="trusted_metric",
                query_row_count=len(rows),
                query_duration_ms=duration_ms,
            )
            _trace_step(trace_id, "execute_sql", input_summary=sql_from_metric, output_summary={"rows": len(rows)}, duration_ms=duration_ms)
            # Record metric-registry hits in conversation history too so
            # follow-up questions ("filter to top 5", "break that down by region")
            # can reference the returned columns and SQL shape.
            _add_history = getattr(adapter, "add_to_history", None)
            if callable(_add_history) and rows:
                _add_history(
                    question=extract_original_question(question),
                    sql=sql_from_metric,
                    columns=list(rows[0].keys()) if rows else [],
                    row_count=len(rows),
                )
            _metric_result_verification = {}
            try:
                from core.result_verifier import verify_result_shape
                _metric_result_verification = verify_result_shape(
                    rows,
                    analytical_plan=_analytical_plan,
                    resolution_plan={
                        "metrics": [{
                            "name": matched_metric.get("name") or "",
                            "source": "metric_registry",
                        }],
                    },
                )
                _trace_step(
                    trace_id,
                    "result_shape_verification",
                    output_summary={
                        "status": _metric_result_verification.get("status"),
                        "score": _metric_result_verification.get("score"),
                        "row_count": _metric_result_verification.get("row_count"),
                        "columns": _metric_result_verification.get("columns") or [],
                        "metric_binding_source": "metric_registry",
                    },
                )
            except Exception as _metric_verification_exc:
                log.debug("Trusted metric result-shape verification skipped: %s", _metric_verification_exc)
            await _send_results(event, adapter, question, rows, sql_from_metric,
                                duration_ms, portal_user, account_id, db_cfg,
                                question_id=audit_request_id,
                                confidence_context={
                                    "validation_code": "trusted_metric",
                                    "has_semantic_plan": True,
                                    "tables_used": extract_sql_tables(
                                        sql_from_metric,
                                        db_cfg.get("db_type", "azure_sql"),
                                    ),
                                    "result_verification": _metric_result_verification,
                                },
                                display_context={
                                    "format_scope": "metric_registry",
                                    "metrics": [matched_metric],
                                },
                                contract_version=_contract_version)
            if _why_mode and rows:
                await _send_why_insight(
                    adapter, event,
                    question=question, rows=rows, sql=sql_from_metric,
                    client=client, account_id=account_id, db_cfg=db_cfg,
                    known_tables=all_known,
                    query_executor=lambda _cfg, _s: _execute_with_policy(_s),
                    question_id=audit_request_id,
                )
            _trace_finish(trace_id, status="success", answer_type="table", row_count=len(rows), duration_ms=duration_ms, final_answer_summary="Answered by metric registry")
            return
        except PolicyDeniedError as policy_error:
            _trace_finish(
                trace_id,
                status="error",
                answer_type="policy_denied",
                error_message=policy_error.decision.explanation,
            )
            await adapter.send_message(
                event,
                "This metric is blocked by the workspace data policy. "
                f"Reason: {policy_error.decision.explanation}",
            )
            return
        except Exception as e:
            _trace_step(trace_id, "execute_sql", input_summary=sql_from_metric, output_summary=str(e), status="error", duration_ms=int((time.time() - _metric_exec_t0) * 1000))
            log.warning("Metric registry SQL failed, falling through to LLM: %s", e)
            # Fall through to normal LLM pipeline

    # ── RAG — scoped to effective tables ─────────────────────────────────────
    # The retriever filters by doc_id (table name), not by substring match,
    # so disallowed tables never leak into the LLM prompt. allowed_tables
    # is passed through explicitly; None means admin/unrestricted.
    _kb_phase_t0 = time.time()
    _weak_retrieval = False
    _retrieval_unscored = False
    try:
        await _send_live_stage(adapter, event, "retrieving_context", "Understanding your data", "Retrieving the most relevant schema, examples, and business context.")
        import re as _re
        retriever    = load_retriever(account_id)   # Qdrant — no filesystem path needed

        _grouping = bool(_re.search(
            r"\b(by|per|grouped by|breakdown|split by|each|for each)\s+\w",
            question.lower()
        ))
        _n = 10 if _grouping else 8

        rag_filter = query_scope_tables
        relevant_kbs = retriever.retrieve(question, n=_n, allowed_tables=rag_filter)
        _weak_retrieval = bool(getattr(retriever, "last_retrieval_weak", False))
        _retrieval_unscored = bool(getattr(retriever, "last_retrieval_unscored", False))

        pinned    = [d for d in relevant_kbs if retriever._is_global(d)]
        table_kbs = [d for d in relevant_kbs if not retriever._is_global(d)]

        if _grouping:
            fact_patterns = retriever.retrieve_fact_patterns(
                question, n=2, allowed_tables=rag_filter,
            )
            for fp in fact_patterns:
                if fp not in (pinned + table_kbs):
                    table_kbs.insert(0, fp)

        # ── Multi-schema coherence (no schema_hint = "All" mode) ─────────────
        # When the user is in "All" mode across multiple schemas, the semantic
        # search can return KB docs from different schemas. If one schema
        # dominates the top results (≥60%, ≥2 docs), do a focused re-retrieval
        # scoped only to that schema's tables so the LLM gets clean, single-
        # schema context instead of a mix.
        if not schema_hint and table_kbs:
            _schema_votes: dict[str, int] = {}
            for _doc in table_kbs:
                _first_line = _doc.splitlines()[0].strip().lstrip("#").strip()
                _parts = _first_line.upper().split(".")
                if len(_parts) >= 2:
                    _sch = _parts[-2].strip("[]")
                    if _sch and _sch not in {"DBO", "SYS", "INFORMATION_SCHEMA", "GUEST"}:
                        _schema_votes[_sch] = _schema_votes.get(_sch, 0) + 1
            if _schema_votes:
                _dominant_sch = max(_schema_votes, key=_schema_votes.get)
                _total_votes  = sum(_schema_votes.values())
                _dom_ratio    = _schema_votes[_dominant_sch] / _total_votes
                if _dom_ratio >= 0.6 and _total_votes >= 2 and len(_schema_votes) > 1:
                    # Build a focused filter for just the dominant schema
                    _base_pool = effective if effective else all_known
                    _focused = {
                        t for t in _base_pool
                        if len(t.split(".")) >= 2
                        and t.upper().split(".")[-2].strip("[]") == _dominant_sch
                    }
                    if _focused:
                        _focused_kbs = retriever.retrieve(
                            question, n=_n, allowed_tables=_focused
                        )
                        _focused_table_kbs = [
                            d for d in _focused_kbs if not retriever._is_global(d)
                        ]
                        if len(_focused_table_kbs) >= 2:
                            table_kbs = _focused_table_kbs
                            log.info(
                                "Multi-schema: re-retrieved focused on %s "
                                "(ratio=%.0f%%, schemas_seen=%d)",
                                _dominant_sch, _dom_ratio * 100, len(_schema_votes),
                            )

        relevant_kbs = [_clamp_kb_doc(d) for d in (pinned + table_kbs)[:7]]
        context = "\n\n---\n\n".join(relevant_kbs)
        # Per-table retrieval telemetry: which tables were candidates, their
        # best cross-encoder score, and whether the relevance floor kept them.
        # Re-read weak flag too — the focused re-retrieval above may have
        # replaced the first retrieve()'s stats, and what fed the prompt is
        # what matters. Persisted on the trace so every "wrong table" report
        # is diagnosable, and aggregated by store.get_kb_doc_quality for the
        # Model Health KB doc-quality ranking.
        _retrieval_stats = list(getattr(retriever, "last_retrieval_stats", []) or [])
        _weak_retrieval = bool(getattr(retriever, "last_retrieval_weak", False))
        _retrieval_unscored = bool(getattr(retriever, "last_retrieval_unscored", False))
        _trace_update(
            trace_id,
            route="normal_sql",
            retrieved_kb_chunk_ids=store.kb_chunk_refs(relevant_kbs),
            retrieved_kb_scores=_retrieval_stats,
        )
        _trace_step(
            trace_id, "retrieve_kb",
            output_summary={
                "chunks": len(relevant_kbs),
                "weak_retrieval": _weak_retrieval,
                "tables": _retrieval_stats,
            },
            duration_ms=int((time.time() - _kb_phase_t0) * 1000),
        )

        # ── Step 2: Retrieve validated SQL examples — few-shot grounding ─────
        # Examples now live in Qdrant alongside KB docs — no chroma_dir needed
        _examples_t0 = time.time()
        examples = retrieve_similar_examples(
            question, account_id, n=3,
            allowed_tables=rag_filter,
            schema_scope=schema_hint,
            kb_dir=state.get("kb_dir", ""),
        )
        if examples:
            context = format_examples_for_prompt(examples, account_id) + "\n\n---\n\n" + context
            log.info("Injected %d validated examples into prompt", len(examples))
            _trace_step(
                trace_id, "retrieve_examples",
                output_summary={"examples": len(examples)},
                duration_ms=int((time.time() - _examples_t0) * 1000),
            )

    except Exception as e:
        log.error("RAG retrieval failed: %s", e)
        _trace_finish(trace_id, status="error", answer_type="error", error_message=f"RAG retrieval failed: {e}")
        await adapter.send_message(event, "⚠️ Knowledge Base not ready.")
        return

    # ── Compiled semantic contract — the single runtime truth source ─────────
    # All approved semantics (model, metrics, graph, terms) come from ONE
    # versioned artifact, recompiled on every admin approval. The consumers
    # below take the contract's sections instead of re-reading each store, so
    # the contract_version stamped on this answer is exactly what was used.
    # Every consumer falls back to its own store read when the contract is
    # absent (accounts that predate the first compile).
    from core.semantic_contract import load_contract
    _contract = load_contract(state.get("kb_dir", ""))
    _contract_version = (_contract.get("meta") or {}).get("contract_version", "")
    if _contract_version:
        _trace_update(trace_id, contract_version=_contract_version)
    _contract_model = _contract.get("model") if _contract else None
    _contract_metrics = _contract.get("metrics") if _contract else None
    _contract_terms = _contract.get("terms") if _contract else None

    # SQL generation — inject any matched business-glossary terms as grounding hints
    term_injection = store.build_term_injection(
        account_id, question, query_scope_tables, terms=_contract_terms,
    )
    schema_grounded_hint = build_schema_grounded_clarification_hint(
        account_id,
        question,
        context,
        allowed_tables=query_scope_tables,
    )
    # Value grounding: resolve user-typed filter literals ("emco corp") to
    # exact database values ("EMCO Corporation") via the per-client value
    # index, so the LLM writes WHERE literals that actually exist. The index
    # covers high-cardinality display columns that schema discovery's
    # 30-distinct-value cap excludes from the KB entirely.
    verified_values_hint = ""
    _value_clarify: list[dict] = []
    _resolved_values: dict = {}
    try:
        from core.value_index import value_index_enabled
        from core.value_resolver import (
            resolve_literals, build_verified_values_injection,
            build_known_terms, filter_resolved_for_compliance,
        )
        if value_index_enabled(state):
            _known_terms = build_known_terms(account_id, all_columns)
            _resolved_values = resolve_literals(
                account_id, question, allowed_tables=query_scope_tables,
                known_terms=_known_terms,
            )
            # Regulated tenants only ground on columns an admin has reviewed as
            # non-sensitive; everything else is dropped before it can reach the
            # prompt. Applied here rather than inside the injection builder so
            # the clarify bucket (which carries no values) still works, and so
            # the decision is recorded in the trace.
            _resolved_values, _value_egress = filter_resolved_for_compliance(
                account_id, _resolved_values
            )
            verified_values_hint = build_verified_values_injection(_resolved_values)
            _value_clarify = _resolved_values.get("clarify") or []
            if verified_values_hint or _value_clarify or _value_egress.get("dropped"):
                _trace_step(
                    trace_id,
                    "value_resolution",
                    output_summary={
                        "verified": len(_resolved_values.get("verified") or []),
                        "in_lists": len(_resolved_values.get("in_lists") or []),
                        "clarify": len(_value_clarify),
                        "compliance_filtered": _value_egress.get("dropped", 0),
                        "compliance_applied": bool(_value_egress.get("applied")),
                    },
                )
    except Exception as _vr_exc:
        log.debug("Value resolution skipped: %s", _vr_exc)
    generic_hints = build_generic_query_hints(question)
    query_intent = analyze_query_intent(question)
    top_n_intent = detect_top_n_intent(question)
    # Candidate metrics are account-wide. We delay injecting/enforcing them
    # until graph + semantic planning has inferred the question's schema/domain.
    _metric_candidates = store.list_metric_formula_context(
        account_id, question, limit=10, metrics=_contract_metrics,
    )
    # Metric logic this user composed in this thread. Never in metric_registry,
    # so it steers nobody else's answers; its ACL is re-checked on every read,
    # so a table revoked mid-thread kills it.
    #
    # Prepended HERE rather than later because this list feeds both metric-scope
    # passes, and the early one at the source-resolution stage is what makes the
    # pipeline anchor on the fact the user actually named.
    _adhoc_metrics: list[dict] = []
    try:
        _adhoc_metrics = store.active_session_metrics(
            account_id, _planner_session_id, allowed_tables,
        )
        if _adhoc_metrics:
            _metric_candidates = _adhoc_metrics + list(_metric_candidates or [])
            log.info(
                "Session metric draft(s) in scope for %s: %s",
                account_id, [m.get("name") for m in _adhoc_metrics],
            )
    except Exception as _adhoc_exc:
        log.debug("Session metric drafts unavailable: %s", _adhoc_exc)
    _matched_metrics: list[dict] = []
    metric_formula_context = ""
    _metric_formula_tables: set[str] = set()

    # If the question came from a suggested-question click, inject the FQN hint
    # so the LLM uses the correct table name format for this DB type.
    table_hint_str = getattr(event, "table_hint", "") or ""
    if table_hint_str:
        parts = table_hint_str.upper().split(".")
        db_type_hint = db_cfg.get("db_type", "")
        if db_type_hint == "azure_sql":
            # Azure SQL only supports 2-part: [SCHEMA].[TABLE]
            sql_name = f"[{parts[-2]}].[{parts[-1]}]" if len(parts) >= 2 else f"[{parts[-1]}]"
        elif db_type_hint == "oracle":
            # Oracle uses OWNER.TABLE
            sql_name = f"{parts[-2]}.{parts[-1]}" if len(parts) >= 2 else parts[-1]
        else:
            # Snowflake supports DATABASE.SCHEMA.TABLE
            sql_name = ".".join(parts)
        # Deliberately a NAMING hint, not a table choice. It said "This question
        # is about the table X", which is a claim the click cannot support: the
        # fqn was attached to the chip at KB-build time, and the same question
        # typed by hand carried no such claim. So one question produced two
        # different answers depending on whether it was clicked or typed, and
        # the clicked one was steered toward a table chosen before the question
        # was planned. What the click genuinely tells us is which name to
        # render if that table is used at all.
        table_hint_injection = (
            f"TABLE NAME FORMAT: if this query uses {table_hint_str}, write it "
            f"as exactly {sql_name}. This came from the suggestion the user "
            f"clicked and does NOT restrict which tables the query may use — "
            f"follow the governed plan and the question itself for that, "
            f"exactly as you would for a typed question."
        )
    else:
        table_hint_injection = ""

    selected_schema_injection = ""
    if schema_hint:
        selected_schema_injection = (
            f"ACTIVE SCHEMA LOCK — {schema_hint}:\n"
            f"The user has explicitly locked the query to the {schema_hint} schema.\n"
            f"MANDATORY RULES:\n"
            f"1. Use ONLY tables and columns that belong to the {schema_hint} schema.\n"
            f"2. NEVER use column names from any other schema (e.g. if a column name appears "
            f"in a different schema's documents it is FORBIDDEN here — do not copy it).\n"
            f"3. Every column name you write MUST appear verbatim in the Knowledge Base "
            f"documents provided in this prompt.\n"
            f"4. If the user asks for a concept (e.g. 'revenue') and you cannot find its "
            f"exact column name in the {schema_hint} KB documents, return CANNOT_GENERATE "
            f"and explain which column is missing — NEVER borrow a name from another schema."
        )

    # Scan already-retrieved KB for Business Synonyms / Key Metrics → compact map.
    # This runs even when the glossary DB is empty and guards against the LLM
    # inventing CamelCase column names for well-known business terms.
    kb_synonym_injection = _extract_kb_synonym_injection(context)

    context_parts = [
        part for part in (
            _analytical_plan_context,
            selected_schema_injection,
            table_hint_injection,
            term_injection,
            kb_synonym_injection,
            schema_grounded_hint,
            verified_values_hint,
            generic_hints,
            context,
        )
        if part
    ]
    context_with_terms = "\n\n".join(context_parts)

    # Tables whose KB document this request has already appended to the prompt.
    # Coverage runs once per planning stage (graph, date role, metric formulas,
    # planner reconciliation) and each call only sees the RAG docs, so without a
    # shared ledger the same table is re-fetched from Qdrant and re-appended
    # every stage — one 90 kB fact document went in four times on the live trace.
    _injected_kb_tables: set[str] = set()

    # Value-resolution ambiguity across DIFFERENT columns ("Emco" matches a
    # customer name AND an item description) can't be settled deterministically
    # or by the LLM — ask the user, mirroring the metric-scope clarification.
    if _value_clarify and can_request_clarification(event, "value_resolver"):
        _vc = _value_clarify[0]
        _vc_options = [
            {
                "label": f"{opt['value']} ({opt.get('business_name') or opt['column']})",
                "value": opt["value"],
            }
            for opt in (_vc.get("options") or [])[:5]
        ]
        if _vc_options:
            clarifying_q = (
                f"'{_vc['phrase']}' matches more than one thing in your data. "
                f"Which one did you mean?"
            )
            if event.user_id:
                _save_pending_clarification(
                    question,
                    context_with_terms,
                    {
                        "term": _vc["phrase"],
                        "options": _vc_options,
                        "source": "value_resolver",
                    },
                )
            send_prompt = getattr(adapter, "send_clarification_prompt", None)
            if callable(send_prompt):
                await send_prompt(event, clarifying_q, _vc_options)
            else:
                option_lines = "\n".join(f"  • {o['label']}" for o in _vc_options)
                await adapter.send_message(
                    event,
                    f"❓ {clarifying_q}\n\n{option_lines}\n\n"
                    "_Reply with one of the options above._",
                )
            _trace_finish(
                trace_id,
                status="success",
                answer_type="clarification",
                final_answer_summary="Requested clarification for an ambiguous filter value",
                duration_ms=int(time.time() * 1000) - start_ms,
            )
            return

    try:
        _semantic_model_context = build_runtime_semantic_context(
            state.get("kb_dir", ""),
            question=question,
            selected_schema=schema_hint,
            model=_contract_model,
            glossary=_planner_terms,
        )
        if _semantic_model_context:
            context_with_terms = _semantic_model_context + "\n\n" + context_with_terms
            _trace_step(
                trace_id,
                "semantic_model_context",
                output_summary={"enabled": True, "schema": schema_hint or "ALL"},
            )
    except Exception as _sm_exc:
        log.debug("Structured semantic model context skipped: %s", _sm_exc)

    # Multi-turn memory: inject conversation history for web portal sessions
    _conv_history = []
    _history_fn = getattr(adapter, "get_history", None)
    if callable(_history_fn):
        _conv_history = _history_fn()

    # Entity graph — deterministic JOIN skeleton resolution
    # Loads the client's entity graph once and resolves JOINs from the question
    # before the LLM is called so the LLM never guesses table relationships.
    _graph_ctx: dict = {}
    _full_graph: dict = {}
    try:
        _full_graph = (
            dict(_contract.get("graph") or {}) if _contract.get("graph")
            else store.get_full_graph(account_id)
        )
        # When the user selected a specific schema, restrict the graph to
        # entities that belong to that schema so the resolver never proposes
        # JOINs to tables from a different schema (which validation would reject).
        if schema_hint and _full_graph.get("entities"):
            _sh = schema_hint.upper()
            _filtered_entities = [
                e for e in _full_graph["entities"]
                if not e.get("schema_name") or e.get("schema_name", "").upper() == _sh
            ]
            _in_schema_names = {e["entity_name"] for e in _filtered_entities}
            _full_graph = {
                "entities": _filtered_entities,
                "relationships": [
                    r for r in _full_graph.get("relationships", [])
                    if r["from_entity"] in _in_schema_names
                    and r["to_entity"] in _in_schema_names
                ],
                "properties": _full_graph.get("properties", []),
            }
        if _full_graph.get("entities"):
            _value_required_entities = _graph_entities_for_verified_values(
                _resolved_values, _full_graph,
            )
            # A generic temporal question ("for 2026") names no specific date
            # concept, so detect_entities() alone would never pull in the
            # fact table that owns the date column (or its date dimension)
            # unless the question happens to imply that fact table for some
            # other reason. Force them in via required_entities whenever an
            # admin-flagged (or sole) default date role exists — otherwise
            # the semantic field plan's later "join to the date dimension"
            # instruction gets built but the LLM is told not to act on it
            # (see the "do NOT introduce new joins" rule in build_sql_system_
            # prompt, which defers to the graph skeleton whenever both are
            # active), so the join never actually lands in the SQL.
            _pregraph_date_roles = (
                [_confirmed_date_role]
                if _confirmed_date_role
                else find_explicit_date_roles(
                    question,
                    list((_contract_model or {}).get("date_roles") or []),
                )
            )
            for _date_role in _pregraph_date_roles:
                _fact_entity = entity_name_for_table(
                    _full_graph, str(_date_role.get("fact_table") or "")
                )
                if _fact_entity:
                    _value_required_entities.add(_fact_entity)
                if normalize_date_key_type(
                    str(_date_role.get("date_key_type") or "surrogate_fk")
                ) == "surrogate_fk":
                    # Role-aware: several role-playing date entities normally
                    # share one physical date dimension, so a table lookup
                    # would force an arbitrary role (this is how an approved
                    # Invoice Date could require "Last Modified Date").
                    # Returns "" when several roles own the dimension and none
                    # matches this binding — forcing an arbitrary one there is
                    # precisely the defect, so force nothing and let the exact
                    # date edge come from _graph_with_exact_date_edges().
                    _dim_entity = date_role_entity_for_binding(
                        _full_graph, _date_role
                    )
                    if _dim_entity:
                        _value_required_entities.add(_dim_entity)
            _resolution_graph = _graph_with_exact_date_edges(
                _full_graph,
                _pregraph_date_roles,
            )
            _graph_ctx = _graph_resolve(
                question=_graph_resolution_question,
                account_id=account_id,
                db_type=db_cfg.get("db_type", "azure_sql"),
                graph=_resolution_graph,
                intent=query_intent,
                required_entities=_value_required_entities,
                metric_formula_tables=set(),
            )
            if _graph_ctx.get("enabled"):
                log.info(
                    "Graph resolved for %s: entities=%s anchor=%s planning_status=%s "
                    "schema_filter=%s",
                    account_id, _graph_ctx.get("detected"), _graph_ctx.get("anchor"),
                    _graph_ctx.get("planning_status") or "selected",
                    schema_hint or "none",
                )
                _trace_step(
                    trace_id,
                    "entity_graph_resolution",
                    input_summary={"question": question, "schema": schema_hint or "all"},
                    output_summary={
                        "entities": _graph_ctx.get("detected") or [],
                        "anchor": _graph_ctx.get("anchor") or "",
                        "edge_ids": _graph_ctx.get("edge_ids") or [],
                    },
                    metadata={
                        "resolved_edges": _graph_ctx.get("resolved_edges") or [],
                        "join_skeleton": _graph_ctx.get("join_skeleton") or "",
                    },
                )
    except Exception as _gex:
        # Not a skip — a failure, and one that removes join governance from the
        # rest of this request. Every downstream graph check reads _graph_ctx,
        # so an empty one is indistinguishable from "this workspace has no
        # entity graph": the validator stops checking joins, the reuse guard
        # stops rejecting stale plans, and the LLM's own joins execute with
        # nothing to compare them against. At debug level none of that was
        # visible; the answer just came back looking normal.
        log.error(
            "Entity-graph resolution FAILED for %s (%s) — join governance is "
            "not in force for this request; any joins in the answer are the "
            "model's own",
            account_id, question[:200], exc_info=True,
        )
        if not _graph_ctx.get("enabled"):
            _graph_ctx = {"enabled": False, "resolution_error": type(_gex).__name__}
        _trace_step(
            trace_id,
            "entity_graph_resolution",
            input_summary={"question": question, "schema": schema_hint or "all"},
            output_summary=str(_gex)[:500],
            status="error",
            metadata={"join_governance": "not_enforced"},
        )

    # ── Table coverage guarantee ──────────────────────────────────────────────
    # After graph resolution we know which tables are required (from detected
    # entities) and which are already covered by the retrieved KB docs.
    # For any gap — a table the graph needs but RAG missed — we do a direct
    # Qdrant filter fetch (not a semantic search) and append the KB doc to
    # context_with_terms so the LLM sees every table's column definitions.
    #
    # Why this matters: dense + BM25 retrieval ranks by similarity to the
    # *question*.  Secondary JOIN tables (e.g. a patient dim that's never
    # mentioned by name) often score below the cutoff and are dropped.  The
    # LLM then hallucinates column names for those tables → CANNOT_GENERATE.
    #
    # Capped at 3 gap-fill docs; all failures are swallowed so this never
    # blocks SQL generation.
    if _graph_ctx.get("enabled"):
        try:
            from core.table_coverage import build_required_fqns, guarantee_table_coverage
            _required_fqns = build_required_fqns(_graph_ctx, _full_graph)
            if _required_fqns:
                _gap_docs = guarantee_table_coverage(
                    account_id    = account_id,
                    required_fqns = _required_fqns,
                    retrieved_docs = relevant_kbs,   # what actually went into context
                    rag_filter    = rag_filter,       # ACL scope (None = admin)
                    max_fill      = 3,
                    already_injected = _injected_kb_tables,
                )
                if _gap_docs:
                    context_with_terms = (
                        context_with_terms
                        + "\n\n---\n\n"
                        + "\n\n---\n\n".join(_gap_docs)
                    )
                    log.info(
                        "Table coverage: injected %d gap-fill doc(s) into prompt "
                        "for %s — missing tables now covered",
                        len(_gap_docs), sorted(_required_fqns),
                    )
        except Exception as _cov_exc:
            log.debug("Table coverage guarantee skipped: %s", _cov_exc)

    # Deterministic field-plan builders scan the question text for literal
    # aliases/business terms. On a clarification retry, `question` is the
    # original question PLUS the raw clarification wrapper (e.g. a chip
    # label like "Synonyms: customer id, customer key") — that wrapper is
    # UI metadata, not natural language, and can spuriously match column
    # aliases (e.g. "key" against a *_DMS_KEY column). Strip it back out
    # for field-plan purposes only; the LLM-facing prompt still gets the
    # full `question` with clarification context further below.
    _structured_semantic_question = ""
    _structured_temporal_window: dict = {}
    if isinstance(_event_raw, dict):
        _structured_semantic_question = str(
            _event_raw.get("_clarification_semantic_question") or ""
        ).strip()
        _raw_temporal_window = _event_raw.get("_clarification_temporal_window")
        if isinstance(_raw_temporal_window, dict) and _raw_temporal_window.get("kind"):
            _structured_temporal_window = dict(_raw_temporal_window)
    _semantic_plan_question = (
        _structured_semantic_question or extract_original_question(question)
    )

    _semantic_plan = {}
    _source_model: dict = {}
    _preferred_facts: set[str] = set()
    _source_scope: dict = {"status": "none", "selected_fact": "", "candidates": []}
    _semantic_model_plan: dict = {}
    try:
        # Phase 3: the model's fact classifications let the planner lock the
        # measure's fact. Without them it falls back to prior behaviour.
        _planner_fact_tables = {
            str(t.get("qualified_name") or t.get("table") or "")
            for t in ((_contract_model or {}).get("tables") or [])
            if str(t.get("type") or "").lower() == "fact"
            and (t.get("qualified_name") or t.get("table"))
        }
        if not _planner_fact_tables:
            try:
                from core.semantic_model import load_semantic_model
                _planner_fact_tables = {
                    str(t.get("qualified_name") or t.get("table") or "")
                    for t in ((load_semantic_model(state.get("kb_dir", "")) or {}).get("tables") or [])
                    if str(t.get("type") or "").lower() == "fact"
                    and (t.get("qualified_name") or t.get("table"))
                }
            except Exception as _pf_exc:
                _planner_fact_tables = set()
                log.warning(
                    "Measure-first anchoring inactive — no fact classifications "
                    "available for %s: %s", account_id, _pf_exc,
                )
        if not _planner_fact_tables:
            log.warning(
                "Measure-first anchoring inactive for %s: model has no fact "
                "tables, so a rival fact's field can still be hard-required",
                account_id,
            )
        _source_model = _contract_model
        if not _source_model:
            try:
                from core.semantic_model import load_semantic_model
                _source_model = load_semantic_model(state.get("kb_dir", "")) or {}
            except Exception:
                _source_model = {}
        _resolved_clarification_sources = {
            str(value).strip()
            for value in (
                _event_raw.get("_clarification_resolved_sources") or []
                if isinstance(_event_raw, dict) else []
            )
            if str(value).strip()
        }
        _source_resolution_question = (
            question
            if is_clarification and "source_scope" in _resolved_clarification_sources
            else _semantic_plan_question
        )
        # Metric candidates are already tenant-scoped and phrase-matched. Run
        # an early, conservative metric pass solely to give source arbitration
        # approved physical-table evidence. The full graph-aware metric scope
        # still runs below and remains authoritative for formula enforcement.
        _early_metric_scope = resolve_metric_scope(
            _metric_candidates,
            _semantic_plan_question,
            all_columns,
            selected_schema=schema_hint,
            limit=6,
        )
        _early_metric_tables: set[str] = set()
        if not _early_metric_scope.ambiguous:
            for _early_metric in _early_metric_scope.metrics:
                _early_metric_tables.update(metric_source_tables(_early_metric, all_columns))

        _source_scope = resolve_source_scope(
            _source_resolution_question,
            _source_model,
            vocab=_vocab,
            selected_schema=schema_hint,
            authoritative_fact_tables=_early_metric_tables,
        )
        _confirmed_source = selected_clarification_option(event, "source_scope")
        _confirmed_source_table = str(
            _confirmed_source.get("value")
            or _confirmed_source.get("table")
            or ""
        ).strip()
        if _confirmed_source_table:
            _known_source_candidates = {
                str(candidate.get("table") or "").strip()
                for candidate in (_source_scope.get("candidates") or [])
            }
            _known_fact_tables = {
                str(table.get("qualified_name") or table.get("table") or "").strip()
                for table in (_source_model.get("tables") or [])
                if str(table.get("type") or "").lower() == "fact"
            }
            if (
                _confirmed_source_table in _known_source_candidates
                or _confirmed_source_table in _known_fact_tables
            ):
                _source_scope = {
                    **_source_scope,
                    "status": "selected",
                    "selected_fact": _confirmed_source_table,
                    "reason": "user-confirmed governed source",
                }
            else:
                log.warning(
                    "Ignoring stale source clarification for %s: %s is no longer a governed fact",
                    account_id,
                    _confirmed_source_table,
                )
        _preferred_facts = ({str(_source_scope.get("selected_fact") or "")} | {
            str(value) for value in (_source_scope.get("selected_facts") or [])
        }) - {""}
        log.info(
            "Business source resolution for %s: status=%s selected=%s candidates=%s",
            account_id, _source_scope.get("status"),
            _source_scope.get("selected_fact"),
            [f"{c.get('table')}:{c.get('score')}" for c in (_source_scope.get("candidates") or [])[:4]],
        )
        if (
            _source_scope.get("status") == "ambiguous"
            and can_request_clarification(event, "source_scope")
        ):
            _source_options = source_clarification_options(_source_scope)
            if _source_options:
                _source_question = (
                    "I found more than one relevant business dataset. "
                    "Which source should I use for this analysis?"
                )
                _save_pending_clarification(
                    _semantic_plan_question,
                    context_with_terms,
                    {
                        "source": "source_scope",
                        "question": _source_question,
                        "options": _source_options,
                    },
                )
                _send_source_prompt = getattr(adapter, "send_clarification_prompt", None)
                if callable(_send_source_prompt):
                    await _send_source_prompt(event, _source_question, _source_options)
                else:
                    await adapter.send_message(
                        event,
                        _source_question + "\n\n" + "\n".join(
                            f"- {option['label']}" for option in _source_options
                        ),
                    )
                _trace_finish(
                    trace_id,
                    status="success",
                    answer_type="clarification",
                    final_answer_summary="Requested clarification for an ambiguous business source",
                    duration_ms=int(time.time() * 1000) - start_ms,
                )
                return
        _semantic_plan = build_semantic_field_plan(
            _semantic_plan_question,
            all_columns,
            query_scope_tables,
            selected_schema=schema_hint,
            vocab=_vocab,
            fact_tables=_planner_fact_tables,
            preferred_fact_tables=_preferred_facts,
        )
        _semantic_plan["source_scope"] = _source_scope
        if _semantic_plan.get("enabled"):
            _trace_step(
                trace_id,
                "semantic_field_plan",
                output_summary={
                    "fields": [
                        f"{f.get('term')}={f.get('table')}.{f.get('column')}"
                        for f in _semantic_plan.get("fields", [])
                    ],
                    "joins": len(_semantic_plan.get("joins") or []),
                },
            )
            log.info(
                "Semantic field plan for %s: fields=%s joins=%d",
                account_id,
                [
                    f"{f.get('term')}={f.get('table')}.{f.get('column')}"
                    for f in _semantic_plan.get("fields", [])
                ],
                len(_semantic_plan.get("joins") or []),
            )
    except Exception as _sp_exc:
        # Not "skipped" — this is the deterministic field plan failing, and an
        # empty dict removes four separate guarantees at once: the term→column
        # bindings, the required join path, the supersession block that forbids
        # a retired column, and the temporal policy that scopes the window to
        # the governed date. Every one of those is silently absent from the
        # prompt afterwards, so the model answers from raw KB prose while the
        # answer looks exactly like a planned one. At debug level none of it
        # was visible.
        _semantic_plan = {"planning_failed": True}
        log.error(
            "Deterministic semantic field planning FAILED for %s on %r — this "
            "answer has no field bindings, no required joins, no superseded-"
            "column list and no temporal policy",
            account_id, _semantic_plan_question[:200], exc_info=True,
        )
        _trace_step(
            trace_id,
            "semantic_field_plan",
            input_summary={"question": _semantic_plan_question},
            output_summary=str(_sp_exc)[:500],
            status="error",
            metadata={"field_governance": "not_enforced"},
        )

    try:
        _semantic_model_plan = build_runtime_semantic_plan(
            state.get("kb_dir", ""),
            question=_semantic_plan_question,
            selected_schema=schema_hint,
            model=_contract_model,
            preferred_fact_tables=({
                str(_source_scope.get("selected_fact") or "")
            } | {
                str(value) for value in (_source_scope.get("selected_facts") or [])
            }) - {""},
            glossary=_planner_terms,
        )
        _semantic_model_plan["source_scope"] = _source_scope
        if _semantic_model_plan.get("enabled"):
            _smp_field_summary = [
                f"{f.get('term')}={f.get('table')}.{f.get('column')}"
                for f in _semantic_model_plan.get("fields", [])
            ]
            _trace_step(
                trace_id,
                "semantic_model_plan",
                output_summary={
                    "fields": _smp_field_summary,
                    "joins": len(_semantic_model_plan.get("joins") or []),
                },
            )
            # Only _trace_step recorded this (silent to console) -- the
            # sibling build_semantic_field_plan block above already logs to
            # console on a match; this one didn't, so a "business term never
            # matched at all" report was indistinguishable in the logs from
            # "matched fine, just got dropped downstream" (the fact-scope
            # bug fixed earlier this session).
            log.info(
                "Structured semantic model plan for %s: fields=%s joins=%d",
                account_id, _smp_field_summary, len(_semantic_model_plan.get("joins") or []),
            )
    except Exception as _smp_exc:
        _semantic_model_plan = {}
        log.warning("Structured semantic model planning failed — approved field "
                    "enforcement inactive for this question: %s", _smp_exc)

    _semantic_plan = _merge_semantic_plans(_semantic_plan, _semantic_model_plan)
    # Source resolution is an analytical request constraint, not a field-plan
    # implementation detail.  Preserve it even when one of the two independent
    # field planners produced no bindings and was therefore skipped by the
    # merge helper.
    _semantic_plan["source_scope"] = _source_scope

    # A business-event count is not fully governed until the expression
    # inside COUNT(DISTINCT ...) is resolved. Resolve it from this tenant's
    # structured semantic model and, when evidence is close, ask using
    # business meanings rather than exposing physical identifiers.
    _count_target_resolution: dict = {}
    if _analytical_plan.counted_entity:
        _count_target_resolution = resolve_count_target(
            _analytical_plan.counted_entity,
            _source_model,
            source_scope=_source_scope,
            confirmed_option=_confirmed_count_target,
        )
        _trace_step(
            trace_id,
            "count_target_resolution",
            output_summary={
                "status": _count_target_resolution.get("status"),
                "entity": _count_target_resolution.get("entity"),
                "selected": (
                    (_count_target_resolution.get("selected") or {}).get("business_name")
                ),
                "candidate_count": len(_count_target_resolution.get("candidates") or []),
            },
            metadata=_count_target_resolution,
        )
        log.info(
            "Count target resolution for %s: entity=%s status=%s selected=%s candidates=%s",
            account_id,
            _analytical_plan.counted_entity,
            _count_target_resolution.get("status"),
            (_count_target_resolution.get("selected") or {}).get("business_name"),
            [
                f"{item.get('business_name')}:{item.get('score')}"
                for item in (_count_target_resolution.get("candidates") or [])[:4]
            ],
        )
        if _count_target_resolution.get("status") == "ambiguous":
            _count_options = count_target_clarification_options(_count_target_resolution)
            _count_question = (
                (
                    f"I found one possible business identifier for counting "
                    f"{_analytical_plan.counted_entity}s, but its event grain is not "
                    "approved strongly enough for me to assume it. Does this meaning "
                    "represent one business event for this question?"
                )
                if len(_count_options) == 1
                else (
                    f"I found more than one possible business identifier for counting "
                    f"{_analytical_plan.counted_entity}s. Which meaning represents one "
                    "business event for this question?"
                )
            )
            if _count_options and can_request_clarification(event, "count_target"):
                _save_pending_clarification(
                    _semantic_plan_question,
                    context_with_terms,
                    {
                        "source": "count_target",
                        "question": _count_question,
                        "options": _count_options,
                    },
                )
                _send_count_prompt = getattr(adapter, "send_clarification_prompt", None)
                if callable(_send_count_prompt):
                    await _send_count_prompt(event, _count_question, _count_options)
                else:
                    await adapter.send_message(
                        event,
                        _count_question + "\n\n" + "\n".join(
                            f"- {option['label']}" for option in _count_options
                        ),
                    )
                _trace_finish(
                    trace_id,
                    status="success",
                    answer_type="clarification",
                    final_answer_summary="Requested governed count identifier clarification",
                    duration_ms=int(time.time() * 1000) - start_ms,
                )
                return
        if _count_target_resolution.get("status") == "missing":
            await adapter.send_message(
                event,
                f"I understand that you want to count {_analytical_plan.counted_entity}s, "
                "but the semantic layer does not yet identify which business field "
                f"represents one {_analytical_plan.counted_entity}. Please ask your "
                "administrator to approve that business identifier before I calculate it.",
            )
            _trace_finish(
                trace_id,
                status="error",
                answer_type="semantic_mapping_required",
                error_message="No governed count target",
                duration_ms=int(time.time() * 1000) - start_ms,
            )
            return
        if _count_target_resolution.get("status") != "selected":
            await adapter.send_message(
                event,
                "I could not confirm a safe business identifier to count. Please choose "
                "one of the business meanings when prompted or ask your administrator "
                "to approve the intended identifier.",
            )
            _trace_finish(
                trace_id,
                status="error",
                answer_type="semantic_mapping_required",
                error_message="Count target remained ambiguous",
                duration_ms=int(time.time() * 1000) - start_ms,
            )
            return

        _semantic_plan["count_target"] = _count_target_resolution
        _target_fact = str(
            (_count_target_resolution.get("selected") or {}).get("table") or ""
        )
        _source_changed_by_count = False
        _current_fact = str(_source_scope.get("selected_fact") or "")
        _target_parts = [
            part for part in _target_fact.upper().replace("[", "").replace("]", "").split(".")
            if part
        ]
        _current_parts = [
            part for part in _current_fact.upper().replace("[", "").replace("]", "").split(".")
            if part
        ]
        _same_count_fact = bool(
            _target_parts
            and _current_parts
            and (
                _target_parts[-2:] == _current_parts[-2:]
                or _target_parts[-1] == _current_parts[-1]
            )
        )
        if _target_fact and not _same_count_fact:
            _source_scope = {
                "status": "selected",
                "selected_fact": _target_fact,
                "selected_facts": [],
                "candidates": [],
                "reason": "governed count target source",
            }
            # The governed identifier is stronger evidence of the counted
            # event than an earlier lexical rival such as a shipment fact that
            # merely contains ORDER_SK. Keep exactly one fact authoritative.
            _preferred_facts.clear()
            _preferred_facts.add(_target_fact)
            _semantic_plan["source_scope"] = _source_scope
            _source_changed_by_count = True

        # Count resolution may establish the authoritative event fact only
        # after the two field planners above have already run. Rebuild both
        # planners with that confirmed fact so fields, joins, required tables
        # and subsequent Date Roles all consume the same source decision.
        if _source_changed_by_count:
            try:
                _replanned_fields = build_semantic_field_plan(
                    _semantic_plan_question,
                    all_columns,
                    query_scope_tables,
                    selected_schema=schema_hint,
                    vocab=_vocab,
                    fact_tables=_planner_fact_tables,
                    preferred_fact_tables=_preferred_facts,
                )
                _replanned_model = build_runtime_semantic_plan(
                    state.get("kb_dir", ""),
                    question=_semantic_plan_question,
                    selected_schema=schema_hint,
                    model=_contract_model,
                    preferred_fact_tables=_preferred_facts,
                    glossary=_planner_terms,
                )
                _semantic_plan = _merge_semantic_plans(
                    _replanned_fields,
                    _replanned_model,
                )
                _semantic_model_plan = _replanned_model
                _semantic_plan["source_scope"] = _source_scope
                _semantic_plan["count_target"] = _count_target_resolution
                _trace_step(
                    trace_id,
                    "semantic_plan_rebuilt_after_count_source",
                    output_summary={
                        "source_fact": _target_fact,
                        "fields": len(_semantic_plan.get("fields") or []),
                        "joins": len(_semantic_plan.get("joins") or []),
                    },
                )
                log.info(
                    "Rebuilt semantic plan for %s after governed count target "
                    "established source fact %s: fields=%d joins=%d",
                    account_id,
                    _target_fact,
                    len(_semantic_plan.get("fields") or []),
                    len(_semantic_plan.get("joins") or []),
                )
            except Exception as _count_replan_exc:
                log.exception(
                    "Count source established for %s but semantic-plan rebuild failed",
                    account_id,
                )
                await adapter.send_message(
                    event,
                    "I resolved the business event to count, but I could not compile "
                    "a consistent governed field and join plan for it. No query was run.",
                )
                _trace_finish(
                    trace_id,
                    status="error",
                    answer_type="semantic_plan_incomplete",
                    error_message=f"Count source plan rebuild failed: {_count_replan_exc}",
                    duration_ms=int(time.time() * 1000) - start_ms,
                )
                return

    # "How many customers do we have" asks for the size of a population, not
    # for how many members had activity. Source arbitration answers it from a
    # fact, which silently drops every member with no row there — a customer
    # who has never been invoiced simply does not exist in the answer. Count it
    # from the table that DEFINES the population instead.
    #
    # Advisory by construction: this runs only when the strict business-event
    # path found nothing, and it governs the question only if the semantic
    # layer resolves the population to exactly one master table. Anything less
    # leaves the question exactly as it was, so no question that answers today
    # can start refusing because of this.
    elif _analytical_plan.population_entity:
        _population_resolution = resolve_population_count_target(
            _analytical_plan.population_entity, _source_model,
        )
        _population_selected = _population_resolution.get("selected") or {}
        _population_table = str(_population_selected.get("table") or "")
        _trace_step(
            trace_id,
            "population_count_resolution",
            output_summary={
                "status": _population_resolution.get("status"),
                "entity": _population_resolution.get("entity"),
                "selected": (
                    f"{_population_table}.{_population_selected.get('column')}"
                    if _population_selected else ""
                ),
                "candidate_count": len(_population_resolution.get("candidates") or []),
            },
            metadata=_population_resolution,
        )
        log.info(
            "Population count resolution for %s: entity=%s status=%s selected=%s reason=%s",
            account_id,
            _analytical_plan.population_entity,
            _population_resolution.get("status"),
            f"{_population_table}.{_population_selected.get('column')}"
            if _population_selected else "",
            _population_resolution.get("reason"),
        )
        if _population_resolution.get("status") == "selected" and _population_table:
            _population_scope = {
                "status": "selected",
                "selected_fact": _population_table,
                "selected_facts": [],
                "candidates": [],
                "source_kind": "master",
                "reason": "governed population master table",
            }
            try:
                # Build the replacement plan BEFORE committing anything: a
                # half-applied population target leaves the compiled request
                # demanding a count target it no longer has, which refuses a
                # question the ordinary path could still have answered.
                _population_plan = _merge_semantic_plans(
                    build_semantic_field_plan(
                        _semantic_plan_question,
                        all_columns,
                        query_scope_tables,
                        selected_schema=schema_hint,
                        vocab=_vocab,
                        fact_tables=_planner_fact_tables,
                        preferred_fact_tables={_population_table},
                    ),
                    build_runtime_semantic_plan(
                        state.get("kb_dir", ""),
                        question=_semantic_plan_question,
                        selected_schema=schema_hint,
                        model=_contract_model,
                        preferred_fact_tables={_population_table},
                        glossary=_planner_terms,
                    ),
                )
            except Exception as _population_replan_exc:
                log.warning(
                    "Population master resolved for %s but the semantic-plan "
                    "rebuild failed — answering without it: %s",
                    account_id, _population_replan_exc,
                )
            else:
                _count_target_resolution = _population_resolution
                _semantic_plan = _population_plan
                _semantic_plan["source_scope"] = _population_scope
                _semantic_plan["count_target"] = _population_resolution
                _source_scope = _population_scope
                _preferred_facts.clear()
                _preferred_facts.add(_population_table)
                # The population entity only becomes the governed counted
                # entity now that a master table has been resolved for it.
                # Every stage below reads the plan, so replace the plan rather
                # than the single slot.
                _analytical_plan = _dataclass_replace(
                    _analytical_plan,
                    counted_entity=_analytical_plan.population_entity,
                    measure_semantics="count_distinct_business_identifier",
                )
                log.info(
                    "Population count for %s governed by master %s.%s: "
                    "fields=%d joins=%d",
                    account_id,
                    _population_table,
                    _population_selected.get("column"),
                    len(_semantic_plan.get("fields") or []),
                    len(_semantic_plan.get("joins") or []),
                )

    # Merged-plan contents, unconditionally. The validator enforces THIS object,
    # so when it rejects correct SQL this line is the ground truth for which
    # bindings were actually in force — inferring it from the rejection message
    # cost several wrong diagnoses.
    log.info(
        "Merged semantic plan for %s: %d field(s) %s | %d join(s) | required_tables=%s",
        account_id,
        len(_semantic_plan.get("fields") or []),
        [
            f"{f.get('term')}={f.get('table')}.{f.get('column')}"
            f"[role={f.get('role')},enf={f.get('enforcement')}]"
            for f in (_semantic_plan.get("fields") or [])
        ],
        len(_semantic_plan.get("joins") or []),
        _semantic_plan.get("required_tables"),
    )

    # Single-fact scoping has to run on the MERGED plan, not just the model
    # plan: the LLM field planner is a second, independent source of required
    # fields, and a rival fact's column arriving from there survives the merge
    # untouched. Live consequence when this only ran inside
    # build_runtime_semantic_plan — "total revenue by warehouse" still failed
    # field_plan_mismatch demanding ERP_ITM_BAL_PRD_FCT.WHS_DMS_KEY, and the
    # repair "satisfied" it by joining the ERP fact to F_SALES_INVOICE. That
    # raw fact-to-fact join passed validation and fanned revenue out from
    # 1050.00 to 2700.00 — a wrong answer where the old behaviour merely
    # refused to answer.
    try:
        from core.semantic_model import _scope_plan_to_single_fact, load_semantic_model

        # Table roles must come from somewhere that is actually populated.
        # _contract_model is None whenever no contract has been compiled, and
        # passing an empty table list made every table look like a non-fact —
        # the scope check then found fewer than two facts and returned having
        # done nothing at all. Silent no-op: the rival-fact requirement stayed
        # required and the repair still built the fan-out join. Fall back to
        # the on-disk model, which is the same source build_runtime_semantic_plan
        # reads when no contract model is supplied.
        _anchor_tables = (_contract_model or {}).get("tables") or []
        if not _anchor_tables:
            _anchor_tables = (
                load_semantic_model(state.get("kb_dir", "")) or {}
            ).get("tables") or []

        if not _anchor_tables:
            log.warning(
                "Merged-plan fact scoping skipped for %s: no table roles "
                "available from the contract model or %s — a rival fact's "
                "field can stay hard-required and drive a fan-out repair",
                account_id, state.get("kb_dir", ""),
            )
        else:
            # Hand the validator the fact list so its raw fact-to-fact join
            # guard works without an approved entity graph. The graph-based
            # guard is inert while relationships sit in review, which is
            # exactly when a fan-out join is most likely to slip through.
            _semantic_plan["known_fact_tables"] = sorted({
                str(t.get("qualified_name") or t.get("table") or "")
                for t in _anchor_tables
                if str(t.get("type") or "").lower() == "fact"
                and (t.get("qualified_name") or t.get("table"))
            })
            _compound_sources = {
                str(value) for value in (_source_scope.get("selected_facts") or []) if value
            }
            _plan_anchor = ""
            if not _compound_sources:
                _plan_anchor = _scope_plan_to_single_fact(
                    _semantic_plan.get("fields") or [],
                    _semantic_plan.get("joins") or [],
                    _anchor_tables,
                    preferred_fact_tables=_preferred_facts,
                    # Provenance decides whether arbitration may override the
                    # plan's required measure — see _scope_plan_to_single_fact.
                    preferred_source_reason=str(_source_scope.get("reason") or ""),
                )
            if _plan_anchor:
                _semantic_plan["fact_anchor"] = _plan_anchor
                _trace_step(
                    trace_id,
                    "semantic_plan_fact_anchor",
                    output_summary={"anchor": _plan_anchor},
                )

            # Required tables are derived state. Rebuild them after rival-fact
            # demotion so optional retrieval noise cannot remain a hard table
            # requirement later in validation/prompt coverage.
            _semantic_plan["required_tables"] = sorted(
                required_semantic_tables(_semantic_plan)
            )
    except Exception as _anchor_exc:
        log.warning(
            "Merged-plan fact scoping failed — a rival fact's field may stay "
            "hard-required for this question: %s", _anchor_exc, exc_info=True,
        )

    # ── Table-coverage fallback when the entity graph didn't resolve ──────────
    # The graph-driven gap-fill above is gated on _graph_ctx["enabled"] — an
    # entity with no table_name, or a disabled/empty graph, silently killed
    # the ONLY structural recovery for retrieval misses. The merged semantic
    # plan is a second independent source of "tables this answer definitely
    # needs": approved-field and planner mappings name exact tables. When the
    # graph produced nothing, gap-fill against those instead.
    if not _graph_ctx.get("enabled"):
        try:
            from core.table_coverage import guarantee_table_coverage
            _plan_fqns = {
                str(table).upper()
                for table in required_semantic_tables(_semantic_plan)
                if "." in str(table)
            }
            if _plan_fqns:
                _plan_gap_docs = guarantee_table_coverage(
                    account_id     = account_id,
                    required_fqns  = _plan_fqns,
                    retrieved_docs = relevant_kbs,
                    rag_filter     = rag_filter,
                    max_fill       = 3,
                    already_injected = _injected_kb_tables,
                )
                if _plan_gap_docs:
                    context_with_terms = (
                        context_with_terms
                        + "\n\n---\n\n"
                        + "\n\n---\n\n".join(_plan_gap_docs)
                    )
                    log.info(
                        "Table coverage (semantic-plan fallback): injected %d "
                        "gap-fill doc(s) — graph was unavailable",
                        len(_plan_gap_docs),
                    )
        except Exception as _plan_cov_exc:
            log.debug("Semantic-plan coverage fallback skipped: %s", _plan_cov_exc)

    # Build entity→schema lookup from the (possibly schema-filtered) full graph.
    # Metrics have a base_entity field; this map lets metric_source_schemas() use
    # the entity graph's schema_name directly — more reliable than parsing bare
    # base_table names or matching required_columns against _schema.json.
    _entity_schema_map: dict[str, str] = {
        e["entity_name"]: (e.get("schema_name") or "").upper().strip()
        for e in (_full_graph.get("entities") or [])
        if (e.get("schema_name") or "").strip()
    }

    # Metric scoping happens after graph + semantic planning so "revenue by
    # prescriber" can choose the Pharmacy revenue metric, while "revenue by
    # warehouse" can choose the Profitability metric.  In All-schema mode, a
    # bare "revenue" question remains ambiguous and should ask the user.
    _metric_scope = resolve_metric_scope(
        _metric_candidates,
        question,
        all_columns,
        selected_schema=schema_hint,
        graph_context=_graph_ctx,
        graph=_full_graph,
        semantic_plan=_semantic_plan,
        entity_schema_map=_entity_schema_map or None,
        limit=6,
    )
    # A pinned draft IS the user's disambiguation -- they just defined, in this
    # thread, exactly what they want computed. So do not ask which of the
    # registry's rival definitions they meant.
    #
    # But suppressing the question is only half of it. Leaving the rivals in
    # scope was worse than asking: their source tables union into the graph
    # resolution, and on the live warehouse a two-table ratio pulled in ten
    # entities (SUP_DMS, ITM_DMS, WHS_DMS, PC_DVN_DMS ...) and the answer came
    # back "the confirmed entity graph cannot reach SUP_DMS" -- a question about
    # customers refused over a supplier table nobody mentioned.
    if _metric_scope.ambiguous and _adhoc_metrics:
        log.info(
            "Ambiguous metric scope for %s narrowed to the thread's own draft: %s",
            account_id, [m.get("name") for m in _adhoc_metrics],
        )
        _metric_scope = dataclasses.replace(_metric_scope, metrics=[], ambiguous=False)
    if (
        _metric_scope.ambiguous
        and can_request_clarification(event, "metric_scope")
    ):
        options = _metric_scope.options or []
        clarifying_q = (
            "I found more than one revenue definition. Which one should I use?"
        )
        if event.user_id and options:
            _save_pending_clarification(
                question,
                context_with_terms,
                {
                    "term": "revenue",
                    "options": [{"label": opt, "value": opt} for opt in options],
                    "source": "metric_scope",
                },
            )
        send_prompt = getattr(adapter, "send_clarification_prompt", None)
        if callable(send_prompt) and options:
            await send_prompt(
                event,
                clarifying_q,
                [{"label": opt, "value": opt} for opt in options],
            )
        else:
            option_lines = "\n".join(f"  • {opt}" for opt in options[:5])
            await adapter.send_message(
                event,
                f"❓ {clarifying_q}\n\n{option_lines}\n\n"
                "_Reply with one of the options above._",
            )
        _trace_finish(
            trace_id,
            status="success",
            answer_type="clarification",
            final_answer_summary="Requested clarification for an ambiguous metric definition",
            duration_ms=int(time.time() * 1000) - start_ms,
        )
        return

    # Work on request-local copies. Source resolution is execution evidence,
    # not persistent registry metadata, and must not leak into later requests.
    _matched_metrics = _metric_scope.metrics
    # resolve_metric_scope drops anything scoring <= 0, and a follow-up turn
    # ("now break that down by region") does not repeat the metric's name. A
    # draft the user defined in this thread has to survive that, or their own
    # definition silently stops applying one question after they made it.
    _pinned_adhoc = [m for m in _adhoc_metrics if m.get("_pinned_thread_metric")]
    if _pinned_adhoc:
        _already = {str(m.get("name") or "").casefold() for m in _matched_metrics}
        _matched_metrics = [
            m for m in _pinned_adhoc if str(m.get("name") or "").casefold() not in _already
        ] + list(_matched_metrics)
    _matched_metrics = [dict(metric) for metric in _matched_metrics]
    if _matched_metrics:
        try:
            # Ad-hoc drafts are not registry rows. increment_metric_usage is an
            # UPDATE ... WHERE name IN (...), so a draft sharing a name with a
            # real metric would bump that metric's usage instead.
            store.increment_metric_usage(account_id, [
                m.get("name") for m in _matched_metrics
                if m.get("name") and not m.get("_adhoc")
            ])
        except Exception as _usage_exc:
            log.debug("Metric usage increment skipped: %s", _usage_exc)
    metric_formula_context = _format_metric_formula_context(_matched_metrics, account_id=account_id)
    _metric_formula_tables = set()
    for _metric in _matched_metrics:
        _resolved_metric_tables = metric_source_tables(_metric, all_columns)
        _metric["_resolved_source_tables"] = sorted(_resolved_metric_tables)
        _metric_formula_tables.update(_resolved_metric_tables)
    # The field plan was built from schema names before metric matching ran, so
    # it could hard-require a measure the registry does not use. Two validators
    # then demand contradictory columns and the repair loop oscillates until it
    # gives up. Resolve it in the registry's favour now that both are known.
    if _matched_metrics and isinstance(_semantic_plan, dict):
        from core.semantic_planner import demote_measures_governed_by_a_metric
        _demoted_measures = demote_measures_governed_by_a_metric(
            _semantic_plan.get("fields") or [], _matched_metrics,
        )
        if _demoted_measures:
            log.info(
                "Measure fields demoted for %s — an approved metric already "
                "governs the measure on that fact: %s",
                account_id, ", ".join(_demoted_measures),
            )
            _trace_step(
                trace_id,
                "measure_field_demoted_for_metric",
                output_summary={
                    "demoted": _demoted_measures,
                    "metrics": [m.get("name") for m in _matched_metrics],
                },
            )

    if metric_formula_context:
        # Prepend metric formulas BEFORE the KB context so the LLM reads them
        # first and they take precedence over any similar-column documentation
        # in the 10,000+ chars of KB content that follows.
        context_with_terms = metric_formula_context + "\n\n" + context_with_terms

    # Resolve role-playing dates after metric scoping. The same measure can use
    # invoice date for sales, accounting date for inventory sales, and another
    # approved role for a different context. The result is compiled into the
    # semantic plan so both generation and validation receive the same rule.
    _date_context_resolution: dict = {"status": "none"}
    _selected_date_bindings: list[dict] = []
    # Everything downstream of this gate -- date-role resolution, the date plan,
    # and the fact arbitration that consumes it -- is silent when the gate is
    # closed. Record the decision itself so an absent arbitration log can be
    # read as "the gate was shut" rather than "the hook is broken".
    _temporal_intent = question_has_temporal_intent(_semantic_plan_question)
    _snapshot_intent = question_has_snapshot_intent(_semantic_plan_question)
    log.info(
        "Date-context gate for %r: temporal_intent=%s snapshot_intent=%s -> %s",
        (_semantic_plan_question or "")[:80], _temporal_intent, _snapshot_intent,
        "entering date resolution" if (_temporal_intent or _snapshot_intent) else "SKIPPED",
    )
    if _temporal_intent or _snapshot_intent:
        try:
            _metric_ids = [
                int(metric.get("id") or 0) for metric in _matched_metrics
                if int(metric.get("id") or 0) > 0
            ]
            # Contracts compiled by older releases may not carry metric IDs.
            # Load the tenant-scoped bindings and match by ID when available,
            # otherwise by canonical metric name. This keeps existing clients
            # working immediately after deployment without a forced rebuild.
            _all_date_bindings = store.list_metric_date_contexts(account_id)
            _metric_names = {
                str(metric.get("name") or "").strip().casefold()
                for metric in _matched_metrics if metric.get("name")
            }
            _date_bindings = [
                binding for binding in _all_date_bindings
                if (
                    int(binding.get("metric_id") or 0) in _metric_ids
                    or str(binding.get("metric_name") or "").strip().casefold() in _metric_names
                )
            ]
            _date_fact_scope = _resolved_fact_tables(
                _graph_ctx,
                _full_graph,
                semantic_plan=_semantic_plan,
                metric_tables=_metric_formula_tables,
            )
            # Explicit/metric/count source selection is stronger than rival
            # facts admitted by broad retrieval. Date roles must remain on the
            # selected event fact(s), including opaque ERP and M3 tables.
            if _preferred_facts:
                _date_fact_scope = set(_preferred_facts)
            _date_roles = list((_contract_model or {}).get("date_roles") or [])
            _explicit_date_roles = find_explicit_date_roles(
                _semantic_plan_question,
                _date_roles,
            )
            _date_fact_inference: dict = {"status": "not_needed"}

            # A generic temporal question may identify only a dimension, for
            # example "patient count by state today". In that case the
            # configured default date belongs to a connected fact, not the
            # dimension itself. Infer that fact deterministically from the
            # governed graph, but only when no stronger metric or explicit
            # role has already established the business event.
            if (
                not _date_fact_scope
                and not _matched_metrics
                and not _date_bindings
                and not _explicit_date_roles
            ):
                _default_fact_tables = {
                    str(role.get("fact_table") or "")
                    for role in _date_roles
                    if str(role.get("status") or "") == "approved"
                    and bool(role.get("is_default"))
                    and role.get("fact_table")
                }
                if not _default_fact_tables:
                    _requested_grain = requested_temporal_grain(
                        _semantic_plan_question
                    )
                    _grain_order = {
                        "day": 1, "week": 2, "month": 3,
                        "quarter": 4, "year": 5,
                    }
                    _inferred_roles = [
                        role for role in _date_roles
                        if str(role.get("status") or "").casefold() == "generated"
                        and int(role.get("confidence") or 0) >= 95
                        and normalize_date_key_type(role.get("date_key_type"))
                        in {"yyyymmdd_integer", "yyyymm_integer"}
                        and role.get("inference_source")
                    ]
                    if _requested_grain:
                        _inferred_roles = [
                            role for role in _inferred_roles
                            if _grain_order.get(
                                str(role.get("temporal_grain") or "")
                                or date_key_temporal_grain(role.get("date_key_type")),
                                99,
                            ) <= _grain_order.get(_requested_grain, 0)
                        ]
                    elif question_has_snapshot_intent(_semantic_plan_question):
                        _known_grains = [
                            str(role.get("temporal_grain") or "")
                            or date_key_temporal_grain(role.get("date_key_type"))
                            for role in _inferred_roles
                        ]
                        if _known_grains:
                            _finest_grain = min(
                                _known_grains,
                                key=lambda grain: _grain_order.get(grain, 99),
                            )
                            _inferred_roles = [
                                role for role in _inferred_roles
                                if (
                                    str(role.get("temporal_grain") or "")
                                    or date_key_temporal_grain(role.get("date_key_type"))
                                ) == _finest_grain
                            ]
                    _default_fact_tables = {
                        str(role.get("fact_table") or "")
                        for role in _inferred_roles
                        if role.get("fact_table")
                    }
                _requested_semantic_tables = required_semantic_tables(_semantic_plan)
                _date_dimension_tables = {
                    str(role.get("dimension_table") or "")
                    for role in _date_roles
                    if role.get("dimension_table")
                }
                _client = store.get_client(account_id) or {}
                _suggested_setting = _client.get("graph_use_suggested")
                if _suggested_setting is None:
                    _allow_suggested_dates = True
                elif isinstance(_suggested_setting, str):
                    _allow_suggested_dates = (
                        _suggested_setting.strip().casefold()
                        not in {"0", "false", "off", "no"}
                    )
                else:
                    _allow_suggested_dates = bool(_suggested_setting)
                _date_fact_inference = infer_connected_default_date_fact(
                    _full_graph,
                    requested_entities=set(_graph_ctx.get("detected") or []),
                    requested_tables=_requested_semantic_tables,
                    candidate_fact_tables=_default_fact_tables,
                    excluded_tables=_date_dimension_tables,
                    allow_suggested=_allow_suggested_dates,
                )
                if _date_fact_inference.get("status") == "selected":
                    _date_fact_scope.add(
                        str(_date_fact_inference.get("fact_table") or "")
                    )
                elif _date_fact_inference.get("status") == "ambiguous":
                    _date_fact_scope.update(
                        str(item.get("fact_table") or "")
                        for item in (_date_fact_inference.get("candidates") or [])
                        if item.get("fact_table")
                    )
                _trace_step(
                    trace_id,
                    "default_date_fact_inference",
                    input_summary={
                        "entities": sorted(set(_graph_ctx.get("detected") or [])),
                        "semantic_tables": sorted(_requested_semantic_tables),
                    },
                    output_summary={
                        "status": _date_fact_inference.get("status") or "none",
                        "fact_table": _date_fact_inference.get("fact_table") or "",
                        "candidate_facts": [
                            item.get("fact_table")
                            for item in (_date_fact_inference.get("candidates") or [])
                        ],
                        "reason": _date_fact_inference.get("reason") or "",
                    },
                    metadata={
                        "graph_scope": _date_fact_inference.get("graph_scope") or "",
                        "edge_ids": _date_fact_inference.get("edge_ids") or [],
                    },
                )
            _date_context_resolution = resolve_contextual_date_binding(
                _semantic_plan_question,
                matched_metrics=_matched_metrics,
                bindings=_date_bindings,
                date_roles=_date_roles,
                required_fact_tables=_date_fact_scope,
                confirmed_date_role=_confirmed_date_role,
                remembered_date_role=(
                    conversation_state_store.get_date_preference(
                        account_id,
                        clarification_session_id(adapter, event),
                        metric_names=_metric_names,
                        fact_tables=_date_fact_scope,
                    )
                    if not _confirmed_date_role
                    else {}
                ),
            )
            # Diagnostic: the "selected"/"selected_many" branch below already
            # logs a resolved binding, but a "none" or "ambiguous" outcome was
            # previously silent -- there was no way to tell, from the logs
            # alone, whether an admin-configured default date role failed to
            # engage because the fact scope came back empty, because no
            # approved default exists for that fact, or because resolution
            # landed on a completely different branch. Always log the status/
            # reason and the inputs that drove it so a "no answer" report can
            # be diagnosed from logs instead of re-deriving this trace by hand.
            log.info(
                "Date-role resolution for %s: status=%s reason=%s fact_scope=%s "
                "matched_metrics=%s date_bindings=%d date_roles=%d",
                account_id,
                _date_context_resolution.get("status") or "",
                _date_context_resolution.get("reason") or "",
                sorted(_date_fact_scope),
                [m.get("name") for m in _matched_metrics if m.get("name")],
                len(_date_bindings),
                len(_date_roles),
            )
            if _date_context_resolution.get("status") == "unsupported_grain":
                _requested = str(
                    _date_context_resolution.get("requested_grain") or "requested"
                )
                _available = str(
                    _date_context_resolution.get("available_grain") or "coarser"
                )
                _available_option = (
                    (_date_context_resolution.get("options") or [{}])[0]
                )
                _available_label = str(
                    _available_option.get("context_name")
                    or _available_option.get("date_role")
                    or "the available business date"
                )
                await adapter.send_message(
                    event,
                    f"I can’t return a trustworthy **{_requested}-level** result from "
                    f"this source. **{_available_label}** is available only at "
                    f"**{_available} grain**, so using it would invent finer dates. "
                    f"Ask for a {_available}-level result, or configure/connect a "
                    f"source with {_requested}-level history.",
                )
                _trace_finish(
                    trace_id,
                    status="success",
                    answer_type="clarification",
                    final_answer_summary="Requested a compatible temporal grain",
                    duration_ms=int(time.time() * 1000) - start_ms,
                )
                return
            if (
                _date_fact_inference.get("status") == "selected"
                and _date_context_resolution.get("status") == "selected"
                and str(
                    (_date_context_resolution.get("binding") or {}).get(
                        "resolution_source"
                    ) or ""
                ) != "inferred_encoded_fact_date"
            ):
                _date_context_resolution["binding"]["resolution_source"] = (
                    "connected_dimension_default"
                )
                _date_context_resolution["reason"] = (
                    "default date of the uniquely connected fact"
                )
            if (
                _date_context_resolution.get("status") == "ambiguous"
                and can_request_clarification(event, "metric_date_context")
            ):
                _all_date_bindings = _unique_date_bindings(list(
                    _date_context_resolution.get("all_options")
                    or _date_context_resolution.get("options")
                    or []
                ))
                _visible_date_bindings = _unique_date_bindings(list(
                    _date_context_resolution.get("options") or []
                ))[:4]
                _stable_date_labels = _date_option_labels(_all_date_bindings)

                def _date_choice_option(item: dict, index: int) -> dict:
                    base_label = str(
                        item.get("context_name")
                        or item.get("date_role")
                        or "Business date"
                    ).strip()
                    label = _stable_date_labels.get(
                        _date_option_identity(item), base_label
                    )
                    option = {
                        "id": f"date_role_{index}",
                        "label": label,
                        # Carry the disambiguated label, not the shared base
                        # name: the selection echo and the free-text matcher
                        # both read this, and the base name alone cannot say
                        # which of two same-named roles the user picked.
                        "value": label,
                        "allow_free_text": bool(
                            _date_context_resolution.get("allow_free_text")
                        ),
                    }
                    # These fields came from the server-side semantic contract
                    # and let the retry use the exact physical role selected.
                    for key in (
                        "context_name", "aliases", "date_role", "fact_table",
                        "fact_column", "dimension_table", "dimension_key",
                        "date_value_column", "date_key_type", "is_default",
                        "priority", "resolution_source", "governance_status",
                        "temporal_grain", "inference_source", "inferred_fallback",
                    ):
                        if key in item:
                            option[key] = item.get(key)
                    return option

                _all_date_options = [
                    _date_choice_option(item, index)
                    for index, item in enumerate(_all_date_bindings, start=1)
                ]
                # Browser suggestions contain business labels only. Exact
                # physical fields stay in the server-side option objects used
                # to compile the governed date plan after selection.
                _business_date_suggestions: list[str] = []
                _seen_business_date_suggestions: set[str] = set()
                for _option in _all_date_options:
                    for _candidate in (
                        _option.get("label"),
                        _option.get("value"),
                    ):
                        _suggestion = str(_candidate or "").strip()
                        _suggestion_key = _suggestion.casefold()
                        if (
                            _suggestion
                            and _suggestion_key not in _seen_business_date_suggestions
                        ):
                            _seen_business_date_suggestions.add(_suggestion_key)
                            _business_date_suggestions.append(_suggestion)
                _visible_identities = {
                    (
                        str(item.get("fact_table") or "").upper(),
                        str(item.get("fact_column") or "").upper(),
                    )
                    for item in _visible_date_bindings
                }
                _date_options = [
                    option for option in _all_date_options
                    if (
                        str(option.get("fact_table") or "").upper(),
                        str(option.get("fact_column") or "").upper(),
                    ) in _visible_identities
                ][:4]
                for _option in _date_options:
                    _option["business_suggestions"] = list(
                        _business_date_suggestions
                    )
                if _date_fact_inference.get("status") == "ambiguous":
                    _date_question = (
                        "More than one connected business event has a default date. "
                        "Which date context should I use?"
                    )
                else:
                    if _date_context_resolution.get("allow_free_text"):
                        _date_question = (
                            "I found these relevant business dates, but none is an "
                            "unambiguous approved default. Which date should I use? "
                            "If it is not listed, enter its business name below."
                        )
                    else:
                        _date_question = (
                            "This metric has more than one valid business date. "
                            "Which date context should I use?"
                        )
                if event.user_id and _date_options:
                    _pending_temporal_window = detect_temporal_window(
                        _semantic_plan_question
                    )
                    _save_pending_clarification(
                        question,
                        context_with_terms,
                        {
                            "term": "business date",
                            "question": _date_question,
                            "options": _date_options,
                            "all_options": _all_date_options,
                            "allow_free_text": bool(
                                _date_context_resolution.get("allow_free_text")
                            ),
                            "source": "metric_date_context",
                            # Keep the visible follow-up in original_q for the
                            # chat transcript, but persist the executable
                            # lineage independently for the resumed compiler.
                            "semantic_question": _semantic_plan_question,
                            "temporal_window": _pending_temporal_window,
                        },
                    )
                send_prompt = getattr(adapter, "send_clarification_prompt", None)
                if callable(send_prompt) and _date_options:
                    await send_prompt(event, _date_question, _date_options)
                else:
                    option_lines = "\n".join(
                        f"  - {option['label']}" for option in _date_options
                    )
                    await adapter.send_message(
                        event,
                        f"{_date_question}\n\n{option_lines}\n\n"
                        "_Reply with one of the options above._",
                    )
                _trace_finish(
                    trace_id,
                    status="success",
                    answer_type="clarification",
                    final_answer_summary="Requested clarification for an ambiguous business date",
                    duration_ms=int(time.time() * 1000) - start_ms,
                )
                return
            if _date_context_resolution.get("status") in {"selected", "selected_many"}:
                # Enrich the approved role from the live schema at query time.
                # This lets an existing date dimension expose month/year/
                # quarter columns immediately, without a KB rebuild, while
                # keeping the role's native date as the filter/anchor value.
                if _date_context_resolution.get("status") == "selected_many":
                    _date_context_resolution["bindings"] = [
                        enrich_date_binding_calendar_attributes(binding, all_columns)
                        for binding in (_date_context_resolution.get("bindings") or [])
                    ]
                else:
                    _date_context_resolution["binding"] = (
                        enrich_date_binding_calendar_attributes(
                            _date_context_resolution.get("binding") or {},
                            all_columns,
                        )
                    )
                if _date_context_resolution.get("status") == "selected_many":
                    _date_plan = build_contextual_date_plan_many(
                        _date_context_resolution.get("bindings") or [],
                        _semantic_plan_question,
                        temporal_window=_structured_temporal_window,
                    )
                else:
                    _date_plan = build_contextual_date_plan(
                        _date_context_resolution.get("binding") or {},
                        _semantic_plan_question,
                        temporal_window=_structured_temporal_window,
                    )
                _expected_temporal_window = (
                    _structured_temporal_window
                    or detect_temporal_window(_semantic_plan_question)
                )
                if (
                    _expected_temporal_window
                    and _date_plan.get("enabled")
                    and not _date_plan.get("temporal_policies")
                ):
                    # Fail closed instead of running an unbounded query when a
                    # known user window cannot be compiled into the date plan.
                    log.error(
                        "Temporal constraint lost before date-plan merge for %s: %s",
                        account_id,
                        _expected_temporal_window,
                    )
                    await adapter.send_message(
                        event,
                        "I retained your requested time period, but could not "
                        "safely apply it to the selected business date. No "
                        "unbounded query was run. Please choose another date "
                        "context or ask your administrator to review this Date Role.",
                    )
                    _trace_finish(
                        trace_id,
                        status="error",
                        answer_type="temporal_plan_error",
                        error_message="Temporal window was not compiled into the date plan",
                        duration_ms=int(time.time() * 1000) - start_ms,
                    )
                    return
                if _date_plan.get("enabled"):
                    _source_scope_before_date_merge = dict(
                        (_semantic_plan or {}).get("source_scope") or {}
                    )
                    _semantic_plan = _merge_semantic_plans(_semantic_plan, _date_plan)
                    if _source_scope_before_date_merge and not _semantic_plan.get("source_scope"):
                        _semantic_plan["source_scope"] = _source_scope_before_date_merge
                    _selected_date_bindings = (
                        _date_context_resolution.get("bindings")
                        if _date_context_resolution.get("status") == "selected_many"
                        else [_date_context_resolution.get("binding") or {}]
                    )
                    if any(
                        str(binding.get("resolution_source") or "")
                        == "thread_date_preference"
                        for binding in _selected_date_bindings
                    ):
                        _thread_date_label = str(
                            (_selected_date_bindings[0] or {}).get("context_name")
                            or (_selected_date_bindings[0] or {}).get("date_role")
                            or "the previously selected business date"
                        )
                        await adapter.send_message(
                            event,
                            f"Using **{_thread_date_label}** for this metric in "
                            "the current thread. Name a different business date "
                            "at any time to change it.",
                        )
                    _inferred_binding = next(
                        (
                            binding for binding in _selected_date_bindings
                            if bool(binding.get("inferred_fallback"))
                            or str(binding.get("resolution_source") or "")
                            == "inferred_encoded_fact_date"
                        ),
                        None,
                    )
                    if _inferred_binding:
                        _inferred_label = str(
                            _inferred_binding.get("context_name")
                            or _inferred_binding.get("date_role")
                            or "Business date"
                        )
                        _inferred_table = str(
                            _inferred_binding.get("fact_table") or ""
                        ).split(".")[-1]
                        _inferred_column = str(
                            _inferred_binding.get("fact_column") or ""
                        )
                        _inferred_grain = str(
                            _inferred_binding.get("temporal_grain") or "calendar"
                        )
                        await adapter.send_message(
                            event,
                            f"Using inferred **{_inferred_label}** from "
                            f"`{_inferred_table}.{_inferred_column}` at "
                            f"**{_inferred_grain} grain**. It is a deterministic "
                            "encoded date on the resolved fact, but it is not an "
                            "admin-approved Date Role or metric date.",
                        )
                    _date_graph = _graph_with_exact_date_edges(
                        _full_graph,
                        _selected_date_bindings,
                    )
                    _entities_by_name = {
                        str(_entity.get("entity_name") or ""): _entity
                        for _entity in (_date_graph.get("entities") or [])
                        if _entity.get("entity_name")
                    }
                    # The governed roles this question actually resolved to.
                    # Resolved by physical edge, because role-playing roles
                    # share one date dimension and a table lookup would return
                    # an arbitrary one.
                    _governed_date_entities = {
                        _resolved_entity
                        for _resolved_entity in (
                            date_role_entity_for_binding(_date_graph, _binding_candidate)
                            for _binding_candidate in _selected_date_bindings
                        )
                        if _resolved_entity
                    }
                    # Carry the broad pass forward, minus any *other* business
                    # date it guessed lexically. Requiring a second date role
                    # beside the resolved one forces two edges into the same
                    # date dimension, which silently returns no rows.
                    _date_required_entities = {
                        _detected_entity
                        for _detected_entity in (_graph_ctx.get("detected") or [])
                        if _detected_entity in _governed_date_entities
                        or not _governed_date_entities
                        or not is_date_role_entity(
                            _entities_by_name.get(_detected_entity, {})
                        )
                    }
                    _date_required_entities |= _governed_date_entities
                    for _date_table in _date_plan.get("required_tables") or []:
                        _date_entity = entity_name_for_table(_date_graph, str(_date_table))
                        # Never let a shared date-dimension table re-introduce a
                        # role the governed binding did not select.
                        if _date_entity and (
                            _date_entity in _date_required_entities
                            or not is_date_role_entity(
                                _entities_by_name.get(_date_entity, {})
                            )
                        ):
                            _date_required_entities.add(_date_entity)
                    log.info(
                        "Date-role graph scoping for %s: governed=%s required=%s",
                        account_id,
                        sorted(_governed_date_entities),
                        sorted(_date_required_entities),
                    )
                    _scoped_graph_ctx = _graph_resolve(
                        question=_semantic_plan_question,
                        account_id=account_id,
                        db_type=db_cfg.get("db_type", "azure_sql"),
                        graph=_date_graph,
                        intent=query_intent,
                        required_entities=_date_required_entities,
                        metric_formula_tables=_metric_formula_tables,
                    )
                    if _scoped_graph_ctx.get("enabled"):
                        _graph_ctx = _scoped_graph_ctx
                    _binding = (
                        _date_context_resolution.get("binding")
                        or ((_date_context_resolution.get("bindings") or [{}])[0])
                    )
                    try:
                        from core.table_coverage import guarantee_table_coverage
                        _date_required_tables = {
                            str(table).upper()
                            for table in (_date_plan.get("required_tables") or [])
                            if table
                        }
                        _date_gap_docs = guarantee_table_coverage(
                            account_id=account_id,
                            required_fqns=_date_required_tables,
                            retrieved_docs=relevant_kbs,
                            rag_filter=rag_filter,
                            max_fill=2,
                            already_injected=_injected_kb_tables,
                        )
                        if _date_gap_docs:
                            context_with_terms = (
                                context_with_terms
                                + "\n\n---\n\n"
                                + "\n\n---\n\n".join(_date_gap_docs)
                            )
                    except Exception as _date_cov_exc:
                        log.debug("Contextual date table coverage skipped: %s", _date_cov_exc)
                    _trace_step(
                        trace_id,
                        "date_context_resolution",
                        output_summary={
                            "metric": _binding.get("metric_name") or "",
                            "context": _binding.get("context_name") or "",
                            "date_role": _binding.get("date_role") or "",
                            "fact_column": _binding.get("fact_column") or "",
                            "source": _binding.get("resolution_source") or "",
                        },
                    )
                    log.info(
                        "Date context resolved for %s: metric=%s context=%s role=%s column=%s",
                        account_id,
                        _binding.get("metric_name") or "",
                        _binding.get("context_name") or "",
                        _binding.get("date_role") or "",
                        _binding.get("fact_column") or "",
                    )
        except Exception as _date_ctx_exc:
            log.warning("Contextual date resolution skipped for %s: %s", account_id, _date_ctx_exc)

    # Fetch KB docs for every table referenced by the selected metric formulas.
    # This is deliberately after metric scoping; otherwise a generic metric from
    # another schema can pollute an All-schema question.
    if _metric_formula_tables:
        try:
            from core.table_coverage import guarantee_table_coverage
            _mf_gap_docs = guarantee_table_coverage(
                account_id    = account_id,
                required_fqns = _metric_formula_tables,
                retrieved_docs = relevant_kbs,
                rag_filter    = None,   # metric tables are admin-approved; bypass per-user ACL
                max_fill      = 4,
                already_injected = _injected_kb_tables,
            )
            if _mf_gap_docs:
                context_with_terms = (
                    context_with_terms
                    + "\n\n---\n\n"
                    + "\n\n---\n\n".join(_mf_gap_docs)
                )
                log.info(
                    "Metric formula coverage: injected %d doc(s) for scoped metric tables %s",
                    len(_mf_gap_docs), sorted(_metric_formula_tables),
                )
        except Exception as _mf_exc:
            log.debug("Scoped metric formula table coverage skipped: %s", _mf_exc)

    # Row-calculated metric joins → promote from text hints to deterministic SQL.
    # After metric scoping we know which metrics are active. If any are
    # row-calculated and the graph resolver has produced a join skeleton, append
    # the required joins so the LLM treats them as hard constraints (part of the
    # "MUST use this exact structure" block) rather than optional instructions.
    # Final planner reconciliation. The earlier graph pass is intentionally
    # broad so it can help metric scoping. Before SQL generation, compile the
    # now-known semantic fields, metric formulas, and date role into one
    # authoritative fact scope and resolve the graph again. The resulting
    # graph is the exact object shared by the prompt and validator.
    _planner_alignment: dict = {}
    try:
        from core.semantic_resolution import build_planner_alignment
        _planner_alignment = build_planner_alignment(
            graph=_full_graph,
            graph_ctx=_graph_ctx,
            semantic_plan=_semantic_plan,
            metric_formula_tables=_metric_formula_tables,
            date_context_resolution=_date_context_resolution,
        )
        if _full_graph.get("entities") and _planner_alignment.get("enabled"):
            _aligned_graph = _graph_with_exact_date_edges(
                _full_graph, _selected_date_bindings,
            )
            _aligned_graph_ctx = _graph_resolve(
                question=_graph_resolution_question,
                account_id=account_id,
                db_type=db_cfg.get("db_type", "azure_sql"),
                graph=_aligned_graph,
                intent=query_intent,
                required_entities=set(_planner_alignment.get("required_entities") or []),
                metric_formula_tables=_metric_formula_tables,
                authoritative_fact_tables=set(
                    _planner_alignment.get("authoritative_fact_tables") or []
                ),
                selected_edge_ids=_confirmed_join_path.get("edge_ids") or [],
            )
            if _aligned_graph_ctx.get("enabled"):
                _graph_ctx = _aligned_graph_ctx
            # The narrowing this pass performed is the difference between
            # answering on the metric's approved business date and joining an
            # unrelated one beside it, so make it readable in production logs
            # rather than only in the trace metadata below.
            log.info(
                "Graph resolved (authoritative) for %s: entities=%s anchor=%s "
                "planning_status=%s governed_dates=%s dropped_dates=%s dropped_facts=%s",
                account_id,
                _graph_ctx.get("detected"),
                _graph_ctx.get("anchor"),
                _graph_ctx.get("planning_status") or "selected",
                _planner_alignment.get("governed_date_entities") or [],
                _planner_alignment.get("dropped_date_entities") or [],
                _planner_alignment.get("dropped_fact_entities") or [],
            )

        # Near-tied relationship paths are now ranked deterministically rather
        # than escalated (core/graph_resolver.py). Say which one was used, so the
        # choice is visible and the user can redirect, instead of stopping the
        # answer to ask a question only a data modeller could answer.
        _ranked_rel = _graph_ctx.get("ranked_relationship") or {}
        if _ranked_rel.get("chosen") and _ranked_rel.get("alternatives"):
            try:
                await adapter.send_message(
                    event,
                    f"ℹ️ Using the **{_ranked_rel['chosen']}** relationship to reach "
                    f"{_ranked_rel.get('target')}. "
                    f"Ask again naming *{_ranked_rel['alternatives'][0]}* to use "
                    "the other one.",
                )
                _trace_step(
                    trace_id,
                    "relationship_path_ranked",
                    output_summary={
                        "target": _ranked_rel.get("target"),
                        "chosen": _ranked_rel.get("chosen"),
                        "alternatives": _ranked_rel.get("alternatives"),
                    },
                )
            except Exception as _rank_exc:
                log.debug("Ranked-relationship disclosure skipped: %s", _rank_exc)

        if (
            _graph_ctx.get("planning_status") == "clarification_required"
            and _graph_ctx.get("clarification_options")
            and can_request_clarification(event, "graph_join_path")
        ):
            _join_options = list(_graph_ctx.get("clarification_options") or [])[:5]
            _join_question = (
                "I found more than one equally governed relationship path for "
                "this analysis. Which business relationship should I use?"
            )
            _save_pending_clarification(
                _semantic_plan_question,
                context_with_terms,
                {
                    "source": "graph_join_path",
                    "question": _join_question,
                    "options": _join_options,
                },
            )
            _send_join_prompt = getattr(adapter, "send_clarification_prompt", None)
            if callable(_send_join_prompt):
                await _send_join_prompt(event, _join_question, _join_options)
            else:
                await adapter.send_message(
                    event,
                    _join_question + "\n\n" + "\n".join(
                        f"- {option.get('label')}" for option in _join_options
                    ),
                )
            _trace_finish(
                trace_id,
                status="success",
                answer_type="clarification",
                final_answer_summary=(
                    "Requested clarification for ambiguous governed join path"
                ),
                duration_ms=int(time.time() * 1000) - start_ms,
            )
            return

        if (
            _graph_ctx.get("enabled")
            and _graph_ctx.get("planning_status") == "blocked"
        ):
            _graph_reason = str(
                _graph_ctx.get("reason")
                or "The confirmed entity graph does not contain a safe path for every requested business entity."
            ).strip()
            _missing_graph_entities = list(
                _graph_ctx.get("missing_entities") or []
            )
            _graph_block_message = (
                "I couldn't build a trusted join plan for this question. "
                + _graph_reason
            )
            if _missing_graph_entities:
                _graph_block_message += (
                    "\n\nMissing governed path for: "
                    + ", ".join(str(name) for name in _missing_graph_entities)
                    + "."
                )
            _graph_block_message += (
                "\n\nAsk an administrator to confirm the required relationship "
                "in the Entity Graph, then try the question again."
            )
            await adapter.send_message(event, _graph_block_message)
            _trace_finish(
                trace_id,
                status="error",
                answer_type="validation_error",
                error_message=_graph_reason,
                final_answer_summary="Blocked SQL generation because the governed join path was incomplete",
                duration_ms=int(time.time() * 1000) - start_ms,
            )
            return

        _trace_step(
            trace_id,
            "planner_reconciliation",
            input_summary={
                "previous_anchor": _planner_alignment.get("previous_anchor") or "",
                "required_tables": _planner_alignment.get("required_tables") or [],
            },
            output_summary={
                "anchor": _graph_ctx.get("anchor") or "",
                "entities": _graph_ctx.get("detected") or [],
                "dropped_fact_entities": (
                    _planner_alignment.get("dropped_fact_entities") or []
                ),
                "governed_date_entities": (
                    _planner_alignment.get("governed_date_entities") or []
                ),
                "dropped_date_entities": (
                    _planner_alignment.get("dropped_date_entities") or []
                ),
            },
            metadata={
                **_planner_alignment,
                "resolved_edges": _graph_ctx.get("resolved_edges") or [],
                "join_skeleton": _graph_ctx.get("join_skeleton") or "",
            },
        )

        # The authoritative pass can introduce a table that the broad first
        # pass did not request. Guarantee its KB document before the prompt is
        # clamped and sent to the model.
        from core.table_coverage import guarantee_table_coverage
        _alignment_gap_docs = guarantee_table_coverage(
            account_id=account_id,
            required_fqns=set(_planner_alignment.get("required_tables") or []),
            retrieved_docs=relevant_kbs,
            rag_filter=rag_filter,
            max_fill=4,
            already_injected=_injected_kb_tables,
        )
        if _alignment_gap_docs:
            context_with_terms += "\n\n---\n\n" + "\n\n---\n\n".join(_alignment_gap_docs)
    except Exception as _align_exc:
        log.warning(
            "Pre-generation planner reconciliation skipped for %s: %s",
            account_id, _align_exc,
        )

    if _graph_ctx.get("enabled") and _matched_metrics:
        try:
            _row_joins = _build_row_metric_join_sql(
                _matched_metrics,
                db_cfg.get("db_type", "azure_sql"),
                _graph_ctx.get("join_skeleton", ""),
            )
            if _row_joins:
                _graph_ctx = {
                    **_graph_ctx,
                    "join_skeleton": _graph_ctx["join_skeleton"] + "\n" + _row_joins,
                }
                log.info(
                    "Row-metric joins appended to graph skeleton for %s", account_id
                )
        except Exception as _rmj_exc:
            log.debug("Row-metric join injection skipped: %s", _rmj_exc)

    # ── Analytical intent detection — inject SQL hints into system prompt ──────
    # Detects window functions, anomaly, contribution, relative date patterns
    # from the question and appends precise SQL construction hints so the LLM
    # emits the correct syntax without needing general training on window funcs.
    _analytic_hints: list[str] = []
    _analysis_contract: dict = {"enabled": False, "mode": "none"}
    try:
        from core.insight import detect_analytical_intents
        from core.window_analytics import build_window_sql_hint
        from core.anomaly_detection import build_anomaly_sql_hint
        from core.contribution_analysis import build_contribution_sql_hint
        from core.analysis_contract import (
            build_analysis_contract,
            format_analysis_contract,
        )

        _intents = detect_analytical_intents(question)
        _analysis_contract = build_analysis_contract(
            question,
            analytical_intents=_intents,
            semantic_plan=_semantic_plan,
            metric_formulas=_matched_metrics,
        )
        _analysis_contract_hint = format_analysis_contract(_analysis_contract)
        if _analysis_contract_hint:
            _analytic_hints.append(_analysis_contract_hint)
            log.info(
                "analysis_contract: mode=%s measure_source=%s measures=%d dimensions=%d",
                _analysis_contract.get("mode"),
                _analysis_contract.get("measure_source"),
                len(_analysis_contract.get("measures") or []),
                len(_analysis_contract.get("dimensions") or []),
            )

        if _intents.get("window"):
            _analytic_hints.append(
                build_window_sql_hint(_intents["window"], db_cfg.get("db_type", "azure_sql"))
            )
            log.info("analytic_intent: window=%s", _intents["window"].type)

        if _intents.get("anomaly"):
            _analytic_hints.append(build_anomaly_sql_hint(db_cfg.get("db_type", "azure_sql")))
            log.info("analytic_intent: anomaly=True")

        if _intents.get("contribution"):
            _analytic_hints.append(build_contribution_sql_hint())
            log.info("analytic_intent: contribution=True")

        if _intents.get("relative_date"):
            ri = _intents["relative_date"]
            _temporal_window = detect_temporal_window(_semantic_plan_question)
            if _temporal_window.get("kind") == "latest_n_observed":
                _analytic_hints.append(
                    "OBSERVED PERIOD HINT: Select exactly the latest "
                    f"{_temporal_window.get('n') or ri.n} distinct observed "
                    f"{_temporal_window.get('unit') or ri.unit} values from the selected fact. "
                    "Do not subtract a number from MAX(date), and do not anchor to the full "
                    "calendar dimension. Aggregate once per selected observed period and "
                    "return the periods in chronological order."
                )
            else:
                _analytic_hints.append(
                    f"RELATIVE DATE HINT: The user is asking about a rolling window of "
                    f"{ri.n} {ri.unit}(s). Use dynamic date arithmetic rather than hardcoded dates. "
                    f"For SQL Server/Azure SQL use DATEADD; for Snowflake use DATEADD or interval syntax; "
                    f"for Oracle use INTERVAL or ADD_MONTHS. "
                    f"{'Also compute the prior window for comparison.' if ri.compare else ''}"
                )
            log.info("analytic_intent: relative_date=%s", ri.unit)

        # ── Tier 2 SQL hints ──────────────────────────────────────────────────
        if _intents.get("budget_vs_actual"):
            from core.budget_vs_actual import build_bva_sql_hint
            _analytic_hints.append(build_bva_sql_hint())
            log.info("analytic_intent: budget_vs_actual=True")

        if _intents.get("cohort"):
            from core.cohort_analysis import build_cohort_sql_hint
            _analytic_hints.append(build_cohort_sql_hint())
            log.info("analytic_intent: cohort=True")

        if _intents.get("correlation"):
            from core.correlation_analysis import build_correlation_sql_hint
            _analytic_hints.append(build_correlation_sql_hint())
            log.info("analytic_intent: correlation=True")

        if _intents.get("pivot"):
            from core.pivot_table import build_pivot_sql_hint
            _analytic_hints.append(build_pivot_sql_hint())
            log.info("analytic_intent: pivot=True")

        # ── Tier 3 SQL hints ──────────────────────────────────────────────────
        if _intents.get("funnel"):
            from core.funnel_analysis import build_funnel_sql_hint
            _analytic_hints.append(build_funnel_sql_hint())
            log.info("analytic_intent: funnel=True")

        if _intents.get("forecast"):
            from core.forecast import build_forecast_sql_hint, extract_forecast_periods
            _n_fc = extract_forecast_periods(question)
            _analytic_hints.append(build_forecast_sql_hint(_n_fc))
            log.info("analytic_intent: forecast=True periods=%d", _n_fc)

        if _intents.get("fiscal") or _analytical_plan.calendar_basis == "fiscal":
            from core.fiscal_calendar import build_fiscal_sql_hint
            _fiscal_month = _analytical_plan.fiscal_year_start_month
            if _fiscal_month:
                _analytic_hints.append(
                    build_fiscal_sql_hint(
                        question,
                        _fiscal_month,
                        db_cfg.get("db_type", "azure_sql"),
                    )
                )
                log.info("analytic_intent: fiscal=True start_month=%d", _fiscal_month)
            else:
                # The analytical preflight normally returns a clarification
                # before reaching this point. Never silently turn an unknown
                # fiscal calendar into a January calendar if a bounded-loop
                # edge case reaches hint construction.
                log.warning(
                    "Fiscal SQL hint withheld: fiscal year start month is unresolved"
                )

        if _intents.get("histogram"):
            from core.distribution_analysis import build_histogram_sql_hint
            _analytic_hints.append(build_histogram_sql_hint())
            log.info("analytic_intent: histogram=True")

        if _intents.get("boxplot"):
            from core.distribution_analysis import build_boxplot_sql_hint
            _analytic_hints.append(build_boxplot_sql_hint())
            log.info("analytic_intent: boxplot=True")

        if _intents.get("whatif"):
            from core.whatif import parse_whatif_params, build_whatif_sql_hint
            _wi_params = parse_whatif_params(question)
            _analytic_hints.append(build_whatif_sql_hint(_wi_params))
            log.info("analytic_intent: whatif=True col_hint=%s", _wi_params.col_hint)
            if hasattr(event, '__dict__'):
                event.__dict__['_whatif_params'] = _wi_params

        # Store intents on event so _send_results can post-process the result
        if hasattr(event, '__dict__'):
            event.__dict__['_analytic_intents'] = _intents
    except Exception as _ai_exc:
        log.debug("Analytical intent detection skipped: %s", _ai_exc)
        _intents = {}
        _analysis_contract = {"enabled": False, "mode": "none"}

    if _analytic_hints:
        context_with_terms = (
            context_with_terms
            + "\n\n---\n\n"
            + "\n\n".join(_analytic_hints)
        )

    # Compile the resolved facts, fields, dates, joins and output shape into a
    # single plan before SQL generation. This is dataset-neutral: every value
    # comes from this tenant's semantic model, metrics and Date Roles.
    _semantic_plan = _scope_semantic_plan_to_analytical_request(
        _semantic_plan,
        _analytical_plan.to_dict(),
    )
    _analytical_request_plan = compile_analytical_request_plan(
        _semantic_plan_question,
        _semantic_plan,
        matched_metrics=_matched_metrics,
        analysis_contract=_analysis_contract,
        graph_context=_graph_ctx,
        analytical_intent_plan=_analytical_plan.to_dict(),
    )
    _semantic_plan["analytical_request_plan"] = _analytical_request_plan
    _trace_step(
        trace_id,
        "analytical_request_plan",
        output_summary={
            "status": _analytical_request_plan.get("status"),
            "source_fact": _analytical_request_plan.get("source_fact"),
            "measures": len(_analytical_request_plan.get("measures") or []),
            "dimensions": len(_analytical_request_plan.get("dimensions") or []),
            "temporal_operations": len(_analytical_request_plan.get("temporal_operations") or []),
            "subrequests": len(_analytical_request_plan.get("subrequests") or []),
        },
        metadata=_analytical_request_plan,
    )
    if _analytical_request_plan.get("status") == "incomplete":
        _missing_plan_slots = list(
            _analytical_request_plan.get("missing_slots") or []
        )
        log.warning(
            "Analytical request plan incomplete for %s: missing=%s question=%r",
            account_id,
            _missing_plan_slots,
            _semantic_plan_question[:120],
        )
        _slot_labels = {
            "source_fact": "the business event dataset to analyse",
            "measure": "the governed measure to calculate",
            "count_target": "the stable business identifier to count",
            "date_role": "the business date to use",
            "comparison_window": "the periods or windows to compare",
        }
        _missing_copy = ", ".join(
            _slot_labels.get(slot, slot.replace("_", " "))
            for slot in _missing_plan_slots[:3]
        )
        _trace_finish(
            trace_id,
            status="error",
            answer_type="semantic_plan_incomplete",
            error_message=f"Missing compiled analytical slots: {_missing_plan_slots}",
            duration_ms=int(time.time() * 1000) - start_ms,
        )
        await adapter.send_message(
            event,
            "I understand the analytical request, but I cannot compile a trusted "
            f"query until the semantic layer resolves {_missing_copy}. "
            "Please name the business measure or event more specifically, or ask "
            "an administrator to approve the missing semantic mapping.",
        )
        return

    # Final safety cap after every prepend/append is done. Priority blocks
    # (metric formulas, semantic model context) sit at the HEAD of the string,
    # so tail truncation only ever sacrifices the lowest-priority material.
    context_with_terms = _clamp_prompt_context(context_with_terms)

    # Sprint 3 — assemble the structured resolution plan (what got matched,
    # by canonical id, cross-referenced against open compile-time conflicts)
    # for trace visibility. Building/tracing it never gates SQL generation
    # and a failure here must never break answering — the actual enforcement
    # decision (Sprint 4a) is a separate, explicit step just below so a bug
    # in this assembly step fails open, not closed.
    _resolution_plan: dict | None = None
    try:
        from core.semantic_resolution import build_resolution_plan
        _open_conflicts = store.list_semantic_conflicts(account_id, status="open")
        _resolution_plan = build_resolution_plan(
            account_id=account_id,
            question=question,
            contract=_contract,
            matched_metrics=_matched_metrics,
            graph_ctx=_graph_ctx,
            semantic_plan=_semantic_plan,
            date_context_resolution=_date_context_resolution,
            schema_hint=schema_hint,
            allowed_tables=query_scope_tables,
            open_conflicts=_open_conflicts,
            planner_alignment=_planner_alignment,
        )
        _trace_step(
            trace_id, "resolution_plan",
            output_summary={
                "confidence": _resolution_plan["confidence"],
                "resolved_deterministically": _resolution_plan["resolved_deterministically"],
                "clarifications": len(_resolution_plan["clarifications"]),
                "advisories": len(_resolution_plan["advisories"]),
            },
            metadata=_resolution_plan,
        )
    except Exception as _resolution_plan_exc:
        log.debug("resolution plan build skipped: %s", _resolution_plan_exc)

    # Sprint 4a — full runtime enforcement. A question that resolves onto an
    # ERROR-severity compile-time conflict (Sprint 2's detectors) is blocked
    # instead of silently answered under contested semantics — but only in
    # "enforce" mode. "off"/"shadow" leave answering exactly as Sprint 3 left
    # it (observational only). Mirrors governed_recompile_contract's severity
    # contract (ERROR blocks, WARNING/INFO/STALE are advisory-only) and the
    # ambiguous-date-context clarification pattern above it in this same
    # function. Best-effort: any failure here falls through to normal
    # answering rather than blocking on an enforcement-check bug.
    if _resolution_plan and not is_clarification:
        _blocking_conflicts = [
            c for c in _resolution_plan.get("clarifications") or []
            if c.get("type") == "compile_time_conflict"
        ]
        if _blocking_conflicts:
            try:
                _compiler_mode = store.get_semantic_compiler_state(account_id).get("mode", "shadow")
            except Exception:
                _compiler_mode = "shadow"
            if _compiler_mode == "enforce":
                _conflict_lines = "\n".join(
                    f"  - {c['message']}" for c in _blocking_conflicts if c.get("message")
                )
                _trace_finish(
                    trace_id, status="error", answer_type="semantic_conflict_blocked",
                    error_message="; ".join(c.get("message", "") for c in _blocking_conflicts),
                )
                await adapter.send_message(
                    event,
                    "I can't answer this confidently yet — it touches part of the "
                    "semantic model with an unresolved conflict:\n\n"
                    f"{_conflict_lines}\n\n"
                    "Please ask an admin to resolve this in Model Health, then try again.",
                )
                return

    # Compile the common fully-governed case before free-form generation.  A
    # single approved expression metric plus a single approved rolling Date
    # Role already determines the complete query; an LLM should not be asked
    # to rediscover those physical choices.
    _effective_temporal_window = (
        dict(_structured_temporal_window)
        or detect_temporal_window(_semantic_plan_question)
    )
    _generation_semantic_context = {
        "intent": query_intent,
        "top_n": top_n_intent.to_dict() if top_n_intent else None,
        "question": question,
        "production_sql": True,
        "graph_context": _graph_ctx,
        "semantic_plan": _semantic_plan,
        "metric_formulas": _matched_metrics,
        "analysis_contract": _analysis_contract,
        "resolution_plan": _resolution_plan,
        "planner_alignment": _planner_alignment,
        "analytical_request_plan": _analytical_request_plan,
        # Keep the request-level window beside the compiled semantic plan.
        # This is deliberately redundant: it lets cache eligibility fail
        # closed if a future plan merge ever drops the temporal policy.
        "temporal_window": _effective_temporal_window,
    }
    if (
        _effective_temporal_window
        and not list((_semantic_plan or {}).get("temporal_policies") or [])
    ):
        log.error(
            "Temporal request reached SQL generation without a compiled policy "
            "for %s: window=%s semantic_question=%r",
            account_id,
            _effective_temporal_window,
            _semantic_plan_question[:160],
        )
        _trace_finish(
            trace_id,
            status="error",
            answer_type="temporal_plan_error",
            error_message="Effective temporal window missing from semantic contract",
            duration_ms=int(time.time() * 1000) - start_ms,
        )
        await adapter.send_message(
            event,
            "I retained your requested time period, but could not compile it "
            "into the selected business-date contract. I did not run an "
            "unbounded query. Please retry the request; if it persists, ask "
            "an administrator to review the applicable Date Role.",
        )
        return
    # ── Resolve the business-date anchor once, not per question ───────────────
    # "The newest date present in this fact" is a question no SQL shape can
    # answer without reading the fact, and it changes only when the warehouse
    # loads. Probe it once, cache it per account+fact+key, and the compiled SQL
    # below carries a literal window instead of a subquery — so nothing reads the
    # fact merely to discover its own latest date. Still data-relative, never the
    # clock; see core/date_anchor.py.
    _anchor_policies: list[dict] = []
    try:
        from core.date_anchor import resolve_business_anchor

        _anchor_policies = [
            policy for policy in ((_semantic_plan or {}).get("temporal_policies") or [])
            if isinstance(policy, dict)
            and str(policy.get("anchor_policy") or "") == "latest_available"
        ]
        if len(_anchor_policies) == 1:
            _resolved_anchor = resolve_business_anchor(
                account_id,
                _anchor_policies[0],
                db_cfg.get("db_type", "azure_sql"),
                lambda probe_sql: _execute_with_policy(probe_sql).rows,
            )
            if _resolved_anchor.get("value"):
                _generation_semantic_context["resolved_date_anchor"] = _resolved_anchor
                # Also on the plan itself, so the plan travels self-describing.
                # The result cache carries semantic_plan into anything built
                # from an answer — an alert, a saved report — and those need to
                # know WHICH date the relative window resolved to in order to
                # move it later. Without it they re-run a query pinned to the
                # day they were created on, forever.
                if isinstance(_semantic_plan, dict):
                    _semantic_plan["resolved_date_anchor"] = _resolved_anchor
                _trace_step(
                    trace_id,
                    "business_date_anchor",
                    output_summary={
                        "value": _resolved_anchor["value"],
                        "source": _resolved_anchor["source"],
                        "cached": _resolved_anchor.get("cached", False),
                        "probe_ms": _resolved_anchor.get("probe_ms", 0),
                    },
                )
    except Exception as _anchor_exc:
        log.warning(
            "Business-date anchor resolution skipped for %s: %s — the compiled "
            "SQL keeps its in-query anchor", account_id, _anchor_exc,
        )

    # ── Disclose a current-period window before answering it ─────────────────
    # "today"/"yesterday"/"this month" anchor on the latest date the data holds,
    # which is correct — but answering them with a bare number is not. On a
    # warehouse loaded to 2025-04-17, "what is today's revenue" returns a
    # confident figure from sixteen months ago and says nothing about which day
    # it is. A believable stale number is worse than a zero, because a zero
    # prompts a question.
    #
    # This deliberately sits OUTSIDE the anchor-resolved branch. The first
    # version was nested inside it, and on a warehouse slow enough to need this
    # disclosure the probe is exactly what times out — so the one mechanism that
    # would have disclosed the staleness was skipped precisely when it mattered.
    # Without a probed date we cannot name the day, but we can still say the
    # answer is data-relative rather than calendar-relative.
    try:
        from datetime import date as _date

        _window_kind = str(
            (_anchor_policies[0] if _anchor_policies else {}).get("kind") or ""
        )
        if _window_kind in {"today", "yesterday", "this_week", "this_month",
                            "this_quarter", "this_year"}:
            _spoken_kind = _window_kind.replace("_", " ")
            _anchor_value = str(
                (_generation_semantic_context.get("resolved_date_anchor") or {})
                .get("value") or ""
            )
            _drift_days = None
            if _anchor_value:
                _anchor_date = _date.fromisoformat(_anchor_value)
                _drift_days = (_date.today() - _anchor_date).days
            if _drift_days is not None and _drift_days > 1:
                # Say where the date came from. The anchor may have been read
                # from cache rather than from the warehouse on this request, and
                # a cached value stated as bare fact is exactly how a stale
                # answer passes for a current one: the banner was the one thing
                # on screen that looked authoritative about the date, and it was
                # willing to name a specific day it had not checked.
                _anchor_meta = (_generation_semantic_context.get("resolved_date_anchor") or {})
                _anchor_checked = "just now" if not _anchor_meta.get("cached") else "from cache"
                await adapter.send_message(
                    event,
                    f"ℹ️ The most recent business data is "
                    f"**{_anchor_date.strftime('%d %b %Y')}** ({_drift_days} days "
                    f"ago), so \"{_spoken_kind}\" is answered as of that date "
                    f"rather than the calendar date. "
                    f"_(read {_anchor_checked}; if your data has just been "
                    f"reloaded, ask an administrator to refresh the business "
                    f"date.)_",
                )
            elif _drift_days is None:
                await adapter.send_message(
                    event,
                    f"ℹ️ \"{_spoken_kind}\" is answered against the most recent "
                    "business date present in your data, which may be earlier "
                    "than the calendar date.",
                )
            if _drift_days is None or _drift_days > 1:
                _trace_step(
                    trace_id,
                    "stale_relative_date_disclosed",
                    output_summary={
                        "window_kind": _window_kind,
                        "anchor": _anchor_value or "unresolved",
                        "drift_days": _drift_days,
                    },
                )
    except Exception as _drift_exc:
        log.warning(
            "Current-period disclosure skipped for %s: %s", account_id, _drift_exc,
        )

    try:
        _compiled_governed_sql = compile_governed_temporal_metric_sql(
            db_cfg["db_type"],
            all_known,
            query_scope_tables,
            all_columns,
            _generation_semantic_context,
        )
    except Exception as _compile_exc:
        _compiled_governed_sql = ""
        log.debug("Governed rolling metric compilation skipped: %s", _compile_exc)

    # The governed compilers decline by returning "" from ~90 different guard
    # clauses, almost all of them silent. When the inputs looked compilable and
    # nothing came back, the question quietly leaves the governed contract for
    # free-form generation — indistinguishable in the logs from "this question
    # was never eligible". Say so once, with the inputs, so a wrong answer can
    # be traced to the fallback instead of being re-derived by hand.
    if not _compiled_governed_sql:
        _compilable_policies = list((_semantic_plan or {}).get("temporal_policies") or [])
        _compilable_metrics = [
            metric for metric in (_matched_metrics or [])
            if str(metric.get("formula_type") or "query").lower() == "expression"
        ]
        if len(_compilable_policies) == 1 and len(_compilable_metrics) == 1:
            log.warning(
                "Governed compiler declined for %s despite one temporal policy "
                "(kind=%s role=%s) and one expression metric (%s) — answering "
                "through free-form SQL generation instead",
                account_id,
                _compilable_policies[0].get("kind") or "",
                _compilable_policies[0].get("business_role") or "",
                _compilable_metrics[0].get("name") or "",
            )

    system = build_sql_system_prompt(
        db_cfg["db_type"], context_with_terms,
        conversation_history=_conv_history or None,
        graph_context=_graph_ctx or None,
        semantic_plan=_semantic_plan or None,
        question=question,
    )
    _sql_generation_max_tokens = _sql_completion_token_budget(
        question,
        _semantic_plan,
    )
    _sql_repair_max_tokens = _sql_completion_token_budget(
        question,
        _semantic_plan,
        retry=True,
    )
    try:
        _reused_plan = None if _compiled_governed_sql else store.find_reusable_validated_sql_plan(
            account_id=account_id,
            question=question,
            selected_schema=schema_hint,
            allowed_tables=sorted(effective),
            db_type=db_cfg["db_type"],
            contract_version=_contract_version,
        )
    except Exception as _plan_exc:
        _reused_plan = None
        log.warning("Governed SQL plan lookup skipped for %s: %s", account_id, _plan_exc)
    if _reused_plan and reused_plan_is_stale_for_graph(
        str(_reused_plan.get("sql_generated") or ""), _graph_ctx, db_cfg["db_type"],
    ):
        log.info(
            "Discarding stale reusable SQL plan trace=%s for %s: references tables "
            "outside the current entity-graph detection (likely cached before a "
            "resolver/validator fix) — falling through to fresh generation.",
            _reused_plan.get("trace_id"), account_id,
        )
        _reused_plan = None
    if _reused_plan:
        _reuse_staleness_code = reused_plan_semantic_staleness_code(
            str(_reused_plan.get("sql_generated") or ""),
            all_known,
            db_cfg["db_type"],
            query_scope_tables,
            all_columns,
            _generation_semantic_context,
            _effective_temporal_window,
        )
        if _reuse_staleness_code:
            log.info(
                "Discarding reusable SQL plan trace=%s for %s: current semantic "
                "contract validation failed with %s - falling through to fresh "
                "governed generation.",
                _reused_plan.get("trace_id"), account_id, _reuse_staleness_code,
            )
            _trace_step(
                trace_id,
                "discard_stale_reused_sql_plan",
                input_summary={
                    "source_trace_id": _reused_plan.get("trace_id"),
                    "source_query_log_id": _reused_plan.get("query_log_id"),
                },
                output_summary={"validation_code": _reuse_staleness_code},
                metadata={"reason": "current_semantic_contract_mismatch"},
            )
            _reused_plan = None
    try:
        if _compiled_governed_sql:
            await _send_live_stage(
                adapter,
                event,
                "compiling_sql",
                "Compiling governed query",
                "Using the approved metric formula and business date mapping.",
            )
        elif _reused_plan:
            await _send_live_stage(
                adapter,
                event,
                "reusing_sql",
                "Using validated query",
                "Revalidating a successful governed query plan for this workspace.",
            )
        else:
            await _send_live_stage(adapter, event, "generating_sql", "Generating query", "Translating the business question into SQL.")
        _llm_gen_t0 = time.time()
        with llm_audit_scope(
            account_id=account_id,
            question=question,
            enabled=audit_enabled,
            request_id=audit_request_id,
            question_id=audit_request_id,
            component="sql_generation",
            # Descriptive half of the egress manifest. `effective` is the
            # ACL-filtered table set this prompt describes, so the audit row
            # states what the model was actually shown rather than everything
            # the workspace owns. The values_sent half is computed from the
            # assembled prompt inside build_egress_manifest — not from here.
            egress={
                "tables": sorted(effective),
                "columns": sorted(
                    f"{t}.{c}"
                    for t in effective
                    for c in (all_columns.get(t) or [])
                ),
            },
        ):
            if _compiled_governed_sql:
                sql = _compiled_governed_sql
                tok_in = tok_out = 0
                log.info(
                    "Compiled governed rolling metric SQL for %s without an LLM retry",
                    account_id,
                )
            elif _reused_plan:
                sql = str(_reused_plan.get("sql_generated") or "")
                tok_in = tok_out = 0
                log.info(
                    "Reused governed SQL plan for %s from trace=%s source=%s rows=%s",
                    account_id,
                    _reused_plan.get("trace_id"),
                    _reused_plan.get("request_source") or "unknown",
                    _reused_plan.get("query_row_count"),
                )
            else:
                sql, tok_in, tok_out = await llm_complete(
                    system, question, provider, model, api_key,
                    temperature=0.0,
                    max_tokens=_sql_generation_max_tokens,
                    **az_kwargs,
                )
            _trace_update(
                trace_id,
                generated_sql=sql,
                prompt_tokens=tok_in,
                completion_tokens=tok_out,
            )
            _trace_step(
                trace_id,
                (
                    "compile_governed_temporal_metric"
                    if _compiled_governed_sql
                    else "reuse_validated_sql_plan" if _reused_plan
                    else "llm_generate_sql"
                ),
                output_summary={
                    "tokens_in": tok_in,
                    "tokens_out": tok_out,
                    "source_trace_id": _reused_plan.get("trace_id") if _reused_plan else None,
                    "source_query_log_id": _reused_plan.get("query_log_id") if _reused_plan else None,
                    "source_request_source": _reused_plan.get("request_source") if _reused_plan else None,
                    "source_row_count": _reused_plan.get("query_row_count") if _reused_plan else None,
                    "authorization_match": "referenced_tables_subset" if _reused_plan else None,
                    "max_output_tokens": (
                        _sql_generation_max_tokens
                        if not (_reused_plan or _compiled_governed_sql)
                        else 0
                    ),
                    "compiler": "governed_metric_date" if _compiled_governed_sql else None,
                },
                duration_ms=int((time.time() - _llm_gen_t0) * 1000),
            )
            # Sprint 4b — advisory-only: does the generated SQL actually
            # touch the tables the resolution plan expected? Trace-level
            # visibility only, same as Sprint 3's plan itself — a mismatch
            # here does not block or retry generation.
            if _resolution_plan:
                try:
                    from core.semantic_resolution import check_sql_plan_coverage
                    _sql_plan_coverage = check_sql_plan_coverage(
                        sql, _resolution_plan, db_cfg["db_type"],
                    )
                    _trace_step(
                        trace_id, "sql_plan_coverage",
                        output_summary={
                            "coverage_ratio": _sql_plan_coverage["coverage_ratio"],
                            "unused_expected_tables": _sql_plan_coverage["unused_expected_tables"],
                        },
                        metadata=_sql_plan_coverage,
                    )
                except Exception as _coverage_exc:
                    log.debug("sql plan coverage check skipped: %s", _coverage_exc)
    except Exception as e:
        _log_q(account_id, question, "", 0, False, str(e), provider, model, 0, 0,
               int(time.time()*1000)-start_ms,
               portal_user_id=pu_id, zoom_user_id=zid,
               question_id=audit_request_id, error_code="llm_error")
        _trace_finish(trace_id, status="error", answer_type="error", error_message=f"AI error: {e}")
        await adapter.send_message(event, f"⚠️ AI error: {e}")
        return

    if sql.startswith("```"):
        sql = "\n".join(sql.split("\n")[1:]).rsplit("```", 1)[0].strip()

    # ── Safety net: inject SELECT DISTINCT for list-entity questions ──────────
    # Fires only when the LLM forgot DISTINCT on a non-aggregate list query.
    # Silently skipped for aggregate / GROUP BY / already-DISTINCT queries.
    sql = _inject_distinct_if_needed(sql, question)
    sql = normalize_generated_sql(sql, db_cfg["db_type"])

    # A sentinel is not accepted as the final result when the deterministic
    # planners produced enough structure to try a constrained recovery. This
    # is deliberately one attempt (never a loop) and it reuses the exact final
    # graph/semantic plan that validation will enforce below.
    if "CANNOT_GENERATE" in sql.upper() and not _reused_plan:
        try:
            from core.semantic_resolution import build_cannot_generate_recovery_prompt
            _recovery_user = build_cannot_generate_recovery_prompt(
                question=question,
                resolution_plan=_resolution_plan,
                graph_ctx=_graph_ctx,
                semantic_plan=_semantic_plan,
            )
            await _send_live_stage(
                adapter,
                event,
                "recovering_sql",
                "Resolving query plan",
                "Retrying once with the approved tables, fields, dates, and joins.",
            )
            _recovery_t0 = time.time()
            with llm_audit_scope(
                account_id=account_id,
                question=question,
                enabled=audit_enabled,
                request_id=audit_request_id,
                question_id=audit_request_id,
                component="sql_cannot_generate_recovery",
            ):
                _recovered_sql, _recovery_tok_in, _recovery_tok_out = await llm_complete(
                    system,
                    _recovery_user,
                    provider,
                    model,
                    api_key,
                    temperature=0.0,
                    max_tokens=_sql_repair_max_tokens,
                    **az_kwargs,
                )
            tok_in += _recovery_tok_in
            tok_out += _recovery_tok_out
            if _recovered_sql.startswith("```"):
                _recovered_sql = "\n".join(
                    _recovered_sql.split("\n")[1:]
                ).rsplit("```", 1)[0].strip()
            _recovered_sql = _inject_distinct_if_needed(_recovered_sql, question)
            _recovered_sql = normalize_generated_sql(_recovered_sql, db_cfg["db_type"])
            _recovery_succeeded = (
                "CANNOT_GENERATE" not in _recovered_sql.upper()
                and len(_recovered_sql) > 10
            )
            _trace_step(
                trace_id,
                "llm_generate_sql",
                output_summary={
                    "tokens_in": _recovery_tok_in,
                    "tokens_out": _recovery_tok_out,
                    "recovered": _recovery_succeeded,
                },
                status="success" if _recovery_succeeded else "error",
                metadata={"retry": True, "reason": "cannot_generate"},
                duration_ms=int((time.time() - _recovery_t0) * 1000),
            )
            if _recovery_succeeded:
                sql = _recovered_sql
                _trace_update(
                    trace_id,
                    generated_sql=sql,
                    prompt_tokens=tok_in,
                    completion_tokens=tok_out,
                    sql_validation_status="recovered_pending_validation",
                )
        except Exception as _recovery_exc:
            log.warning(
                "CANNOT_GENERATE recovery failed for %s: %s",
                account_id, str(_recovery_exc)[:160],
            )
            _trace_step(
                trace_id,
                "llm_generate_sql",
                output_summary=str(_recovery_exc),
                status="error",
                metadata={"retry": True, "reason": "cannot_generate"},
            )

    # CANNOT_GENERATE — try clarification before giving up (Approach B)
    if "CANNOT_GENERATE" in sql.upper():
        _trace_update(trace_id, generated_sql=sql, sql_validation_status="cannot_generate")
        _log_q(account_id, question, "", 0, False, "CANNOT_GENERATE",
               provider, model, tok_in, tok_out, int(time.time()*1000)-start_ms,
               portal_user_id=pu_id, zoom_user_id=zid,
               question_id=audit_request_id, error_code="cannot_generate")
        # Finalize before the optional clarification call. Even if that
        # secondary provider call fails, the answer trace cannot remain
        # indefinitely in its initial "started" state.
        _trace_finish(
            trace_id,
            status="error",
            answer_type="cannot_generate",
            error_message="CANNOT_GENERATE after constrained recovery",
            duration_ms=int(time.time() * 1000) - start_ms,
        )

        # A clarification reply may expose a different missing slot. Re-check
        # ambiguity under the source-aware round limit instead of skipping all
        # later clarification opportunities.
        if can_request_clarification(event):
            with llm_audit_scope(
                account_id=account_id,
                question=question,
                enabled=audit_enabled,
                request_id=audit_request_id,
            question_id=audit_request_id,
                component="clarification",
            ):
                is_ambiguous, clarifying_q, cmeta = await check_ambiguity_glossary_first(
                    account_id, question, context, provider, model, api_key, az_kwargs,
                    allowed_tables=query_scope_tables,
                )
            _clarification_source = str((cmeta or {}).get("source") or "llm")
            if (
                is_ambiguous
                and clarifying_q
                and event.user_id
                and can_request_clarification(event, _clarification_source)
            ):
                cmeta = dict(cmeta or {})
                cmeta["question"] = clarifying_q
                opts = (cmeta or {}).get("options") or []
                if opts:
                    _save_pending_clarification(question, context, cmeta)
                    send_prompt = getattr(adapter, "send_clarification_prompt", None)
                    if callable(send_prompt):
                        await send_prompt(event, clarifying_q or "I need a bit more context to answer that.", opts)
                    else:
                        # Plain-text fallback: list the options so the user can
                        # reply with one of them. Without this, they see only
                        # the clarifying question and have to guess what to say.
                        option_lines = []
                        for o in opts[:5]:
                            lbl = (o.get("label") or o.get("value") or "").strip()
                            if lbl:
                                option_lines.append(f"  • {lbl}")
                        options_text = "\n".join(option_lines)
                        await adapter.send_message(event,
                            f"❓ I need a bit more context to answer that.\n\n"
                            f"{clarifying_q}\n\n"
                            f"{options_text}\n\n"
                            f"_Reply with one of the options above (or type your own)._"
                        )
                else:
                    _save_pending_clarification(question, context, cmeta)
                    await adapter.send_message(event,
                        f"❓ I need a bit more context to answer that.\n\n"
                        f"{clarifying_q}\n\n"
                        f"_Reply in plain language and I'll continue with your original question._"
                    )
                _trace_finish(
                    trace_id,
                    status="success",
                    answer_type="clarification",
                    final_answer_summary=(
                        "Requested clarification after constrained SQL recovery failed"
                    ),
                    duration_ms=int(time.time() * 1000) - start_ms,
                )
                _trace_update(trace_id, error_message="")
                return
        from core.failure_messages import suggest_closest_terms
        _closest = suggest_closest_terms(question, account_id, state.get("kb_dir", ""))
        _closest_line = (
            f"Closest known terms in your data: {', '.join(_closest)}.\n\n" if _closest else ""
        )
        await adapter.send_message(event,
                "❓ I couldn't find the right tables or columns to answer that.\n\n"
                + _closest_line +
                "Try rephrasing — for example:\n"
                "  • Be more specific about the metric you want\n"
                "  • Include a time range (last month, this year)\n"
                "  • Mention the specific column or category name\n\n"
                "If this is a business concept not in the data, ask your "
                "administrator to add it to the Metric Registry."
            )
        return

    # ── Validate + Execute with ONE unified retry on failure ────────────────
    # Retry fires on:
    #   * validator failure (unknown_table / parse)   — LLM picks a valid table
    #   * execution failure                           — LLM fixes the SQL
    # Retry does NOT fire on:
    #   * access_denied — the user genuinely lacks permission; we need the
    #                     admin to intervene, not a different SQL.
    #   * ddl / cannot_generate — already terminal.
    # Only the currently-failing query is repaired. A first repair is always
    # bounded, and one additional attempt is admitted later only when the
    # validator reason code changes (proof that the first repair progressed).
    rows        = None
    exec_error  = None
    last_reason = ""
    last_code   = ""

    await _send_live_stage(adapter, event, "validating_sql", "Checking query safety", "Verifying table access, structure, and execution safety.")
    retry_count = 0
    semantic_context = _generation_semantic_context
    _validate_t0 = time.time()
    ok, reason, code = validate_sql(
        sql, all_known, db_cfg["db_type"], query_scope_tables, all_columns, semantic_context
    )
    _validate_ms = int((time.time() - _validate_t0) * 1000)

    # When one governed expression metric and one approved Date Role fully
    # determine a current-vs-previous period comparison, compile it from that
    # executable contract before asking the LLM to retry. This repairs missing
    # role joins, surrogate-key parsing, and truncated comparison CTEs with the
    # same dynamic path for every client/schema.
    if not ok and code in {
        "field_plan_mismatch", "graph_plan_mismatch", "surrogate_date_conversion",
        "temporal_anchor_missing", "temporal_anchor_mismatch", "temporal_role_mismatch",
        "temporal_anchor_unscoped", "observed_period_shape", "source_fact_mismatch",
        "period_comparison_shape", "parse",
    }:
        _temporal_repair_t0 = time.time()
        try:
            _governed_temporal_sql = attempt_governed_temporal_metric_repair(
                sql, db_cfg["db_type"], all_known, query_scope_tables,
                all_columns, semantic_context,
            )
        except Exception as _temporal_rep_exc:
            _governed_temporal_sql = ""
            log.debug("Governed temporal repair skipped: %s", _temporal_rep_exc)
        if _governed_temporal_sql:
            _trace_step(
                trace_id,
                "governed_temporal_repair",
                input_summary=sql,
                output_summary=_governed_temporal_sql,
                metadata={"mode": "deterministic", "date_role": "approved"},
                duration_ms=int((time.time() - _temporal_repair_t0) * 1000),
            )
            sql = _governed_temporal_sql
            ok, reason, code = True, "OK", "ok"

    # Display-field plan mismatches are mechanically fixable from the plan
    # itself (add the dimension join, swap key → display column). Try that
    # before burning an LLM retry — and before surfacing a validator error.
    if not ok and code == "field_plan_mismatch":
        _repair_t0 = time.time()
        try:
            _repaired_sql = attempt_field_plan_repair(
                sql, db_cfg["db_type"], all_known, query_scope_tables,
                all_columns, semantic_context,
            )
        except Exception as _rep_exc:
            _repaired_sql = ""
            log.debug("Field-plan repair skipped: %s", _rep_exc)
        if _repaired_sql:
            _trace_step(
                trace_id,
                "field_plan_repair",
                input_summary=sql,
                output_summary=_repaired_sql,
                metadata={"mode": "deterministic"},
                duration_ms=int((time.time() - _repair_t0) * 1000),
            )
            sql = _repaired_sql
            ok, reason, code = True, "OK", "ok"

    # A single validator suggestion is deterministic schema evidence, not an
    # LLM guess. Apply it locally and require the complete query to validate
    # again before execution; ambiguous cases still use the governed retry.
    if not ok and code == "unknown_column":
        from core.validator import validate_sql_detailed, repair_unambiguous_unknown_columns
        _column_validation = validate_sql_detailed(
            sql, all_known, db_cfg["db_type"], query_scope_tables,
            all_columns, semantic_context,
        )
        _column_repair = repair_unambiguous_unknown_columns(
            sql, _column_validation, db_cfg["db_type"],
        )
        if _column_repair:
            _repaired_validation = validate_sql_detailed(
                _column_repair, all_known, db_cfg["db_type"], query_scope_tables,
                all_columns, semantic_context,
            )
            if _repaired_validation.ok:
                _trace_step(
                    trace_id,
                    "unknown_column_repair",
                    input_summary=sql,
                    output_summary=_column_repair,
                    metadata={"mode": "deterministic", "errors": _column_validation.errors},
                )
                sql = _column_repair
                ok, reason, code = True, "OK", "ok"
        else:
            _entity_field_reason = _entity_field_unavailable_reason(
                _column_validation.errors
            )
            if _entity_field_reason:
                # A cross-table substitution is a change of business meaning,
                # not a safe SQL repair. Stop before the LLM retry and explain
                # which entities actually expose the requested field.
                reason = _entity_field_reason
                code = "entity_field_unavailable"

    _trace_update(
        trace_id,
        generated_sql=sql,
        sql_validation_status="pass" if ok else "fail",
        sql_validation_error="" if ok else reason,
    )
    _trace_step(
        trace_id,
        "validate_sql",
        input_summary=sql,
        output_summary=reason,
        status="success" if ok else "error",
        metadata={"code": code},
        duration_ms=_validate_ms,
    )

    if ok:
        _exec_t0 = time.time()
        try:
            await _send_live_stage(adapter, event, "executing_query", "Running query", "Executing the SQL against your connected data source.")
            _loop = asyncio.get_running_loop()
            governed = await asyncio.wait_for(
                _loop.run_in_executor(None, _execute_with_policy, sql, semantic_context),
                timeout=_query_wait_timeout(db_cfg),
            )
            rows = governed.rows
            _rows_truncated = bool(getattr(governed, "truncated", False))
            sql = governed.sql
            _trace_step(trace_id, "execute_sql", input_summary=sql, output_summary={"rows": len(rows)}, duration_ms=int((time.time() - _exec_t0) * 1000))
        except asyncio.TimeoutError:
            exec_error = "Query timed out after 3 minutes. Try narrowing your question with a filter (e.g. date range or specific customer)."
            _trace_step(trace_id, "execute_sql", input_summary=sql, output_summary=exec_error, status="error", duration_ms=int((time.time() - _exec_t0) * 1000))
            log.warning("Query timed out for %s", account_id)
        except PolicyDeniedError as policy_error:
            rows = None
            exec_error = None
            ok = False
            last_reason = policy_error.decision.explanation or "Blocked by regulated data policy."
            last_code = policy_error.decision.reason_code
            _trace_step(
                trace_id,
                "policy_enforcement",
                input_summary=sql,
                output_summary={
                    "reason": last_code,
                    "audit_id": policy_error.decision.audit_id,
                },
                status="error",
                duration_ms=int((time.time() - _exec_t0) * 1000),
            )
        except Exception as first_error:
            exec_error = str(first_error)
            _trace_step(trace_id, "execute_sql", input_summary=sql, output_summary=exec_error, status="error", duration_ms=int((time.time() - _exec_t0) * 1000))
            log.warning("First execution failed for %s: %s",
                        account_id, exec_error[:100])
    else:
        last_reason, last_code = reason, code

    # A historically successful plan can become stale as source data, row
    # policies, or semantic filters change. Do not present an unexplained empty
    # result merely because the SQL came from the governed plan cache. Route it
    # through the existing single repair/regeneration path; the regenerated SQL
    # must still pass the current validator and policy executor.
    if _reused_plan and ok and exec_error is None and rows is not None and len(rows) == 0:
        last_code = "reused_plan_empty"
        last_reason = (
            "The reused governed SQL plan executed successfully but returned no rows "
            "under the current data and policy scope. Generate a fresh governed plan."
        )
        _trace_step(
            trace_id,
            "reused_sql_plan_empty",
            input_summary=sql,
            output_summary={
                "rows": 0,
                "source_trace_id": _reused_plan.get("trace_id"),
                "source_query_log_id": _reused_plan.get("query_log_id"),
            },
            status="error",
        )
        log.warning(
            "Reusable SQL plan returned zero rows for %s (source_trace=%s); regenerating",
            account_id,
            _reused_plan.get("trace_id"),
        )
        rows = None
        ok = False

    # A freshly generated (not reused-cache) query with an active date-role
    # temporal filter that comes back with zero rows is much more likely to
    # be a wrong anchor/offset than a genuine business zero -- unlike a
    # non-date-filtered zero (e.g. "how many orders did X cancel today"),
    # which is very often a legitimate answer and must NOT be retried away
    # into a fabricated non-zero result. Scoped narrowly to date-filtered
    # questions only, mirroring the reused_plan_empty repair path above but
    # for SQL that was never cached in the first place.
    elif (
        not _reused_plan and ok and exec_error is None and rows is not None
        and len(rows) == 0 and (_semantic_plan or {}).get("temporal_policies")
    ):
        last_code = "zero_row_fresh"
        last_reason = (
            "The generated SQL executed successfully but returned no rows for a "
            "date-filtered question. Generate a fresh query -- try a different "
            "table, join, or date anchor; do not repeat the same restrictive filter."
        )
        _trace_step(
            trace_id,
            "zero_row_fresh_date_filtered",
            input_summary=sql,
            output_summary={"rows": 0},
            status="error",
        )
        log.info(
            "Fresh date-filtered query returned zero rows for %s; regenerating",
            account_id,
        )
        rows = None
        ok = False

    _sql_repair_reason_codes = {
        "unknown_table", "unknown_column", "date_key_format", "dialect_mismatch",
        "production_shape", "period_comparison_shape", "composition_shape",
        "anti_join_shape", "fanout_aggregate", "top_n_shape", "graph_plan_mismatch",
        "field_plan_mismatch", "metric_formula_mismatch", "null_aggregate_diagnostic",
        "parse", "multi_statement", "not_select", "reused_plan_empty",
        "zero_row_fresh", "surrogate_date_conversion", "temporal_anchor_missing",
        "temporal_anchor_mismatch", "temporal_role_mismatch",
        "temporal_anchor_unscoped", "observed_period_shape", "source_fact_mismatch",
    }
    _initial_repair_code = str(last_code or code or "").strip().casefold()
    _repair_reason_codes_seen = {_initial_repair_code} if _initial_repair_code else set()
    _llm_repair_attempts = 0
    # Keep the explicit tuple on this assignment for the established wiring
    # guards that audit newly-added validator codes. The normalized set above
    # is reused by the progressive-repair gate below.
    retryable = (not ok and (last_code or code) in ("unknown_table", "unknown_column", "date_key_format", "dialect_mismatch", "production_shape", "period_comparison_shape", "composition_shape", "anti_join_shape", "fanout_aggregate", "top_n_shape", "graph_plan_mismatch", "field_plan_mismatch", "metric_formula_mismatch", "null_aggregate_diagnostic", "parse", "multi_statement", "not_select", "reused_plan_empty", "zero_row_fresh", "surrogate_date_conversion", "temporal_anchor_missing", "temporal_anchor_mismatch", "temporal_role_mismatch", "temporal_anchor_unscoped", "observed_period_shape", "source_fact_mismatch", "order_alias_mismatch")) or (exec_error is not None)

    # A statement timeout is not a defect in the SQL, so there is nothing for a
    # repair to fix. Rewriting it cannot make the database faster: the retry
    # spends another full timeout window, costs a generation call, and can
    # replace a clear "the query timed out" answer with a misleading validator
    # error from the regenerated SQL. Observed live — a governed revenue query
    # timed out at 120 s, the retry ran 120 s more and then failed on the
    # entity-graph join plan, so four minutes bought a wrong explanation.
    from core.failure_messages import is_query_timeout
    from core.schema import _query_timeout_seconds

    _timed_out = exec_error is not None and is_query_timeout(exec_error)
    if _timed_out:
        retryable = False
        _trace_step(
            trace_id,
            "execution_timeout_no_repair",
            input_summary=sql,
            output_summary={
                "timeout_seconds": _query_timeout_seconds(db_cfg),
                "repair_attempted": False,
            },
            status="error",
        )
        log.warning(
            "Execution timed out for %s after %ss — not repairing: the SQL is "
            "valid and a rewrite cannot make the database faster",
            account_id, _query_timeout_seconds(db_cfg),
        )

    if retryable:
        if exec_error is not None:
            import re as _re_retry
            # Extract column name(s) flagged as invalid by the DB engine.
            # SQL Server / Azure SQL format: "Invalid column name 'XYZ'"
            bad_cols = _re_retry.findall(
                r"Invalid column name '([^']+)'", exec_error, _re_retry.IGNORECASE
            )
            col_fix_note = ""
            if bad_cols:
                cols_list = ", ".join(f"'{c}'" for c in bad_cols)
                col_fix_note = (
                    f"\n⚠️  COLUMN NAME ERROR: The column(s) {cols_list} do NOT exist in the database.\n"
                    f"These column names were invented — they are NOT in the schema.\n"
                    f"MANDATORY: Look at the 'COLUMN SYNONYM MAP' and 'BUSINESS TERM DEFINITIONS' "
                    f"sections in the system prompt to find the EXACT column name for each concept.\n"
                    f"Also check the 'Session context' section — if the previous turn returned a column "
                    f"that represents the same concept, reuse that EXACT column name verbatim.\n"
                    f"NEVER guess, never use CamelCase variants of column names.\n"
                )
            from core.failure_messages import scrub_error_for_llm

            retry_user = (
                f"The following SQL failed with this error:\n"
                f"SQL: {sql}\n"
                f"Error: {scrub_error_for_llm(exec_error)}\n"
                f"{col_fix_note}\n"
                f"The original question was: {question}\n\n"
                f"Rewrite the SQL to fix the error. Use ONLY column names that appear "
                f"verbatim in the Knowledge Base (system prompt). "
                f"Return only the corrected SQL, no explanation."
            )
        else:
            validation_repair_note = ""
            if last_code == "unknown_column":
                # When a schema is locked, "switch to the table where the column exists" is
                # wrong if that table lives in a different schema — detect this and override
                # the repair instruction so the LLM stays in the selected schema.
                _cross_schema = _unknown_column_is_cross_schema(
                    last_reason, schema_hint,
                )
                if _cross_schema:
                    validation_repair_note = (
                        f"\nSCHEMA-LOCKED COLUMN REPAIR RULE:\n"
                        f"- The column you used does NOT exist in the {schema_hint} schema.\n"
                        f"- The validator found it in a DIFFERENT schema — you are FORBIDDEN "
                        f"from using that table or column.\n"
                        f"- Search the {schema_hint} Knowledge Base documents in this prompt "
                        f"for the correct column that represents the same concept.\n"
                        f"- If no matching column exists in {schema_hint}, return CANNOT_GENERATE "
                        f"and state exactly which column is missing from the {schema_hint} schema.\n"
                        f"- Do NOT copy column names from other schemas under any circumstance.\n"
                    )
                else:
                    validation_repair_note = (
                        "\nUNKNOWN COLUMN REPAIR RULE:\n"
                        "- The SQL used a column on a table where that column does not exist.\n"
                        "- If the validator lists 'Exact column exists on', switch the source table "
                        "or add the required JOIN to that table.\n"
                        "- Do not keep the same table alias and do not retry the same invalid column/table pair.\n"
                        "- If no exact column exists anywhere in the KB context, use the closest exact "
                        "business synonym/column from the KB or return CANNOT_GENERATE.\n"
                    )
            elif last_code == "anti_join_shape":
                validation_repair_note = (
                    "\nANTI-JOIN REPAIR RULE:\n"
                    "- Prefer source rows WHERE NOT EXISTS (SELECT 1 FROM missing_side WHERE governed_key_match).\n"
                    "- If using LEFT JOIN, WHERE must test a join key from the RIGHT/missing side with IS NULL.\n"
                    "- The FROM table must be the source/parent table containing the records to list.\n"
                    "- Never null-test the FROM table's own key; that can make the result impossible.\n"
                )
            elif last_code == "composition_shape":
                validation_repair_note = (
                    "\nCOMPOSITION QUERY-SHAPE REPAIR REQUIRED:\n"
                    "- This is a category composition/share request, not a histogram.\n"
                    "- Return exactly one row per requested category.\n"
                    "- SELECT the requested category plus the governed aggregate measure.\n"
                    "- GROUP BY the requested category and preserve the governed join path and filters.\n"
                    "- Do not return individual measure rows for bins, ranges, or frequency analysis.\n"
                    "- If the measure is not additive and no approved formula is available, return CANNOT_GENERATE rather than summing it.\n"
                )
            elif last_code == "fanout_aggregate":
                validation_repair_note = (
                    "\nGRAIN-PRESERVING AGGREGATE REPAIR RULE:\n"
                    "- Do not SUM/AVG/COUNT a parent-grain value after a one-to-many or many-to-many join.\n"
                    "- Remove tables not required by the requested outputs or filters.\n"
                    "- Otherwise pre-aggregate the child to the governed join key, use EXISTS, or COUNT(DISTINCT approved_parent_key).\n"
                )
            elif last_code == "date_key_format":
                validation_repair_note = (
                    "\nDATE-KEY REPAIR RULE:\n"
                    "- Decode integer calendar keys before FORMAT/YEAR/MONTH/DATEPART.\n"
                    "- YYYYMMDD uses varchar(8); YYYYMM period keys use varchar(6) plus '01'.\n"
                    "- Copy the exact nullable conversion from the semantic date-key plan and exclude invalid/sentinel values.\n"
                )
            elif last_code == "top_n_shape":
                _top_limit = top_n_intent.limit if top_n_intent else "N"
                _top_direction = (
                    "ASC" if top_n_intent and top_n_intent.direction == "ascending" else "DESC"
                )
                _tie_instruction = (
                    "Use TOP (N) WITH TIES or RANK/DENSE_RANK <= N because the user explicitly requested ties."
                    if top_n_intent and top_n_intent.tie_policy == "include_ties"
                    else "Return exactly N rows with TOP (N), or ROW_NUMBER() followed by rn <= N. Do not use RANK/DENSE_RANK."
                )
                _scope_instruction = (
                    "Use ROW_NUMBER() OVER (PARTITION BY the requested group ORDER BY metric) and filter rn <= N."
                    if top_n_intent and top_n_intent.per_group
                    else "Apply the limit to the final ordered result set."
                )
                validation_repair_note = (
                    f"\nTOP-{_top_limit} SEMANTIC REPAIR RULE:\n"
                    f"- {_scope_instruction}\n"
                    f"- {_tie_instruction}\n"
                    f"- Order the requested metric {_top_direction} and use the business dimension as a stable secondary order.\n"
                    "- Do NOT compare the metric to MIN/MAX from a Top-N CTE; that changes a row-limit request into a threshold and can return zero rows on ties.\n"
                )
            elif last_code == "dialect_mismatch":
                validation_repair_note = (
                    f"\nDIALECT REPAIR REQUIRED:\n"
                    f"- Rewrite every incompatible construct using only the configured {db_cfg['db_type']} syntax.\n"
                    "- Replace the actual row-limit, date, null, cast, and identifier syntax identified by the validator.\n"
                    "- Do not merely rename aliases or return the same incompatible syntax.\n"
                )
            elif last_code == "production_shape":
                validation_repair_note = (
                    "\nPRODUCTION QUERY-SHAPE REPAIR REQUIRED:\n"
                    "- List every projected output column explicitly; SELECT * and alias.* are not allowed.\n"
                    "- Every non-CROSS JOIN must have an exact ON/USING relationship from the available schema/entity graph.\n"
                    "- Do not use CROSS JOIN or comma joins unless the original question explicitly asks for a cartesian product.\n"
                    "- For above/below-average group comparisons, use a grouped CTE followed by AVG(metric_alias) OVER () in a scored CTE.\n"
                )
            elif last_code == "period_comparison_shape":
                validation_repair_note = (
                    "\nPERIOD-COMPARISON STRUCTURE REQUIRED:\n"
                    "- In a period_totals CTE, group by the resolved business date period and calculate the approved metric once.\n"
                    "- In a period_comparison CTE, calculate LAG(METRIC) or LEAD(METRIC) from the metric alias; never use LAG(SUM(...)).\n"
                    "- In the final SELECT, calculate difference and percentage from METRIC and PREV_METRIC.\n"
                    "- Keep NULLIF(PREV_METRIC, 0) around the denominator and balance every parenthesis.\n"
                    "- Preserve the exact governed date-role JOIN and use a native date value directly when the date dimension exposes one.\n"
                )
            elif last_code == "parse":
                _date_contract_lines = _governed_date_anchor_repair_lines(
                    _semantic_plan or {}
                )
                validation_repair_note = (
                    "\nINCOMPLETE OR MALFORMED SQL REPAIR REQUIRED:\n"
                    "- Discard the truncated draft and return one complete SQL statement.\n"
                    "- Every WITH/CTE must be closed and followed by a final SELECT.\n"
                    "- Prefer one period_totals CTE plus conditional aggregation or LAG; "
                    "do not duplicate a long current-period query into separate incomplete CTEs.\n"
                    "- Do not shorten SQL with ellipses (...).\n"
                    "- Preserve the approved metric formula and the governed date-role "
                    "join; never parse a surrogate date key as YYYYMMDD.\n"
                    + (
                        "- Copy this governed date-role contract exactly:\n"
                        + _date_contract_lines
                        if _date_contract_lines
                        else ""
                    )
                )
            elif last_code == "field_plan_mismatch":
                # Narrow the note to the field(s) that ACTUALLY failed --
                # without this, a plan defining both a display field (e.g.
                # Supplier Name, already satisfied) and a governed date
                # field (e.g. Snapshot Date, actually broken) always
                # returned display-field guidance and never once reached
                # the date branch, no matter which field the validator
                # flagged.
                from core.validator import validate_sql_detailed as _vsd_for_field_repair

                _field_violated_errors = _vsd_for_field_repair(
                    sql, all_known, db_cfg["db_type"], query_scope_tables, all_columns, semantic_context,
                ).errors
                validation_repair_note = build_field_plan_repair_note(
                    _semantic_plan or {}, _field_violated_errors,
                )
            elif last_code == "graph_plan_mismatch":
                validation_repair_note = (
                    "\nENTITY-GRAPH REPAIR REQUIRED:\n"
                    "- Copy the exact FROM/JOIN skeleton from ENTITY GRAPH JOIN PLAN into the base query or base CTE.\n"
                    "- Preserve every composite key condition connected with AND.\n"
                    "- Preserve LEFT JOIN where the graph marks the relationship optional.\n"
                    "- Do not substitute a nearby key or invent an alternative join path.\n"
                )
                # A missing entity-graph edge to the governed date-role
                # dimension is checked BEFORE field_plan_mismatch in the
                # validator, so it fires first and this generic "add the
                # join" note is all the LLM gets on retry. Without the exact
                # anchor subquery, it reliably regresses to the fact-only
                # surrogate-key anti-pattern, which then fails a SECOND,
                # different check (field_plan_mismatch) with no further
                # retry available. Detect that the missing edge IS the
                # date-role join (its dimension table is named in the
                # validator's own error text) and append the same
                # copy-pasteable anchor the temporal_anchor_* path uses.
                _date_contract_lines = _governed_date_anchor_repair_lines(_semantic_plan or {})
                _date_dim_names = {
                    str(_p.get("dimension_table") or _p.get("date_table") or "")
                    for _p in (_semantic_plan or {}).get("temporal_policies") or []
                    if (_p.get("dimension_table") or _p.get("date_table"))
                }
                if _date_contract_lines and any(
                    name and name in (last_reason or "") for name in _date_dim_names
                ):
                    validation_repair_note += (
                        "\nGOVERNED DATE-ROLE JOIN REQUIRED (this is one of the missing "
                        "entity-graph edges above):\n" + _date_contract_lines
                    )
            elif last_code == "metric_formula_mismatch":
                # Inject the EXACT approved formula(s) verbatim — do not rely on
                # the LLM finding them in the KB context, which can be overridden.
                _metric_formulas_inline = ""
                for _mf in _matched_metrics:
                    if ((_mf.get("formula_type") or "query").lower() == "expression"
                            and _mf.get("sql_template")):
                        _metric_formulas_inline += (
                            f"\n  Metric name : {_mf.get('name', 'metric')}\n"
                            f"  EXACT formula: {_mf['sql_template'].strip()}\n"
                            f"  Required columns (must appear in SELECT): "
                            f"{_mf.get('required_columns', 'see formula above')}\n"
                        )
                validation_repair_note = (
                    "\nAPPROVED METRIC FORMULA REPAIR RULE:\n"
                    "- Your SQL used a column that is NOT the approved metric formula.\n"
                    "- You MUST replace your aggregate expression with the EXACT formula below:\n"
                    f"{_metric_formulas_inline if _metric_formulas_inline else '  See the APPROVED METRIC FORMULAS section in the prompt.'}\n"
                    "- The formula MUST appear in the SELECT clause — not just in a WHERE filter.\n"
                    "- Do NOT use CUS_IVC_LIN_AMT or any other similar column as a substitute "
                    "for the approved metric, even if the Knowledge Base suggests it.\n"
                    "- Keep all other query structure (GROUP BY, JOINs, date filters) unchanged.\n"
                    "- Only replace the aggregate expression itself.\n"
                )
            elif last_code == "null_aggregate_diagnostic":
                validation_repair_note = (
                    "\nNULL-AWARE AGGREGATE REPAIR RULE:\n"
                    "- This is a filtered single-row aggregate such as revenue for one customer/key.\n"
                    "- Include COUNT_BIG(*) AS [MatchedRows] for Azure SQL, or COUNT(*) AS MatchedRows for other DBs.\n"
                    "- Include COUNT(metric_column) AS [NonNullMetricRows] for every SUM metric.\n"
                    "- Wrap every SUM metric with COALESCE(SUM(metric_column), 0) so missing values render as 0.\n"
                    "- Keep the user's filter value unchanged.\n"
                )
            elif last_code == "surrogate_date_conversion":
                _date_contract_lines = _governed_date_anchor_repair_lines(_semantic_plan or {})
                validation_repair_note = (
                    "\nSURROGATE DATE-KEY REPAIR REQUIRED:\n"
                    "- The flagged column is a sequential surrogate key, not an encoded calendar date — "
                    "do NOT wrap it in TRY_CONVERT/CONVERT/CAST, and do NOT pass it to YEAR/MONTH/DAY/"
                    "DATEPART/DATEADD/DATEDIFF directly.\n"
                    "- The validation error above names the exact table and column to JOIN to and filter/group on "
                    "— use that EXACT column verbatim.\n"
                    "- If no exact table/column is named above, find the date dimension's real calendar column "
                    "verbatim in the schema context (KB documents / table columns in this prompt). "
                    "Do NOT invent, abbreviate, or guess a column name (e.g. 'YR') that does not appear there.\n"
                    + ("- Copy this governed date-role contract exactly:\n" + _date_contract_lines
                       if _date_contract_lines else "")
                )
            elif last_code in {"temporal_anchor_missing", "temporal_anchor_mismatch", "temporal_role_mismatch", "temporal_anchor_unscoped"}:
                _date_contract_lines = _governed_date_anchor_repair_lines(_semantic_plan or {})
                validation_repair_note = (
                    "\nGOVERNED RELATIVE-DATE REPAIR REQUIRED:\n"
                    "- Never compare a fact surrogate date ID to DATEADD, a date literal, or MAX(the ID).\n"
                    "- If the fact exposes a native date, filter that native date directly.\n"
                    "- Otherwise use the exact fact-key to date-dimension join below, and use the date dimension's native calendar field.\n"
                    "- Derive the business clock from the REQUIRED ANCHOR subquery below — never from "
                    "MAX(CALENDAR_DATE) over an unrestricted date dimension, which includes future "
                    "calendar rows with no matching fact records.\n"
                    + (_date_contract_lines if _date_contract_lines else
                       "- Use the exact date-role JOIN and calendar field supplied in the semantic plan.\n")
                )
            elif last_code == "reused_plan_empty":
                validation_repair_note = (
                    "\nSTALE REUSED-PLAN REGENERATION REQUIRED:\n"
                    "- The previous governed SQL was valid but returned zero rows under the current scope.\n"
                    "- Generate a fresh query from the current Knowledge Base, semantic plan, date roles, and entity graph.\n"
                    "- Preserve the user's metric, date range, and requested grain.\n"
                    "- Do not repeat the same restrictive predicate or join path unless the current semantic plan explicitly requires it.\n"
                    "- Use only exact tables, columns, approved metric formulas, and governed joins supplied in this prompt.\n"
                )
            elif last_code == "zero_row_fresh":
                _date_contract_lines = _governed_date_anchor_repair_lines(_semantic_plan or {})
                validation_repair_note = (
                    "\nZERO-ROW DATE-FILTERED QUERY — REGENERATION REQUIRED:\n"
                    "- Your SQL was valid and executed successfully but returned zero rows for "
                    "a question with a date filter.\n"
                    "- Try a different table, join path, or date anchor — do not repeat the "
                    "same restrictive filter unless the semantic plan explicitly requires it.\n"
                    "- Preserve the user's metric, date range, and requested grain.\n"
                    + (_date_contract_lines if _date_contract_lines else "")
                )
            retry_user = (
                f"The following SQL failed validation with: {last_reason}\n"
                f"SQL: {sql}\n\n"
                f"The original question was: {question}\n\n"
                f"{validation_repair_note}\n"
                f"Rewrite the SQL using only tables and columns that appear in "
                f"the provided knowledge base context. Return only the corrected "
                f"SQL, no explanation."
            )

        # Keep semantic-plan guardrails during repair. Otherwise the retry can
        # pass validation while still using a raw dimension key as a business
        # label instead of the planned display field.
        _retry_plan = _semantic_plan or None
        _retry_semantic_context = dict(semantic_context)

        try:
            await _send_live_stage(adapter, event, "repairing_query", "Repairing query", "Fixing a validation or execution issue before retrying.")
            _retry_llm_t0 = time.time()
            _llm_repair_attempts += 1
            with llm_audit_scope(
                account_id=account_id,
                question=question,
                enabled=audit_enabled,
                request_id=audit_request_id,
            question_id=audit_request_id,
                component="sql_repair",
            ):
                sql_retry, _, _ = await llm_complete(
                    build_sql_system_prompt(
                        db_cfg["db_type"],
                        context_with_terms,
                        graph_context=_graph_ctx or None,
                        semantic_plan=_retry_plan,
                        question=question,
                    ),
                    retry_user, provider, model, api_key,
                    temperature=0.0,
                    max_tokens=_sql_repair_max_tokens,
                    **az_kwargs,
                )
            # Retry timings accumulate onto the same buckets as the first
            # attempt (bucket aggregation sums by step_name), not separate rows.
            _trace_step(trace_id, "llm_generate_sql", output_summary={"retry": True},
                        metadata={"repair_attempt": _llm_repair_attempts},
                        duration_ms=int((time.time() - _retry_llm_t0) * 1000))
            if sql_retry.startswith("```"):
                sql_retry = "\n".join(sql_retry.split("\n")[1:]).rsplit("```", 1)[0].strip()

            sql_retry = _inject_distinct_if_needed(sql_retry, question)
            sql_retry = normalize_generated_sql(sql_retry, db_cfg["db_type"])

            if "CANNOT_GENERATE" not in sql_retry.upper() and len(sql_retry) > 10:
                _retry_validate_t0 = time.time()
                ok2, reason2, code2 = validate_sql(
                    sql_retry, all_known, db_cfg["db_type"], query_scope_tables, all_columns, _retry_semantic_context)
                _trace_step(
                    trace_id, "validate_sql", input_summary=sql_retry, output_summary=reason2,
                    status="success" if ok2 else "error", metadata={"code": code2, "retry": True},
                    duration_ms=int((time.time() - _retry_validate_t0) * 1000),
                )
                if ok2:
                    _retry_exec_t0 = time.time()
                    try:
                        await _send_live_stage(adapter, event, "executing_query", "Retrying query", "Running the corrected query against your data.")
                        _loop = asyncio.get_running_loop()
                        governed = await asyncio.wait_for(
                            _loop.run_in_executor(None, _execute_with_policy, sql_retry, _retry_semantic_context),
                            timeout=_query_wait_timeout(db_cfg),
                        )
                        rows = governed.rows
                        _rows_truncated = bool(getattr(governed, "truncated", False))
                        sql         = governed.sql
                        exec_error  = None
                        ok, last_reason, last_code = True, "OK", "ok"
                        retry_count += 1
                        log.info("Retry succeeded for %s", account_id)
                        _trace_step(trace_id, "execute_sql", input_summary=sql, output_summary={"rows": len(rows), "retry": True}, duration_ms=int((time.time() - _retry_exec_t0) * 1000))
                    except asyncio.TimeoutError:
                        exec_error = "Retry query timed out after 3 minutes."
                        log.warning("Retry query timed out for %s", account_id)
                        _trace_step(trace_id, "execute_sql", input_summary=sql_retry, output_summary=exec_error, status="error", duration_ms=int((time.time() - _retry_exec_t0) * 1000))
                    except PolicyDeniedError as policy_error:
                        exec_error = None
                        ok = False
                        last_reason = policy_error.decision.explanation or "Blocked by regulated data policy."
                        last_code = policy_error.decision.reason_code
                        log.warning(
                            "Retry denied by policy for %s: %s",
                            account_id,
                            last_code,
                        )
                        _trace_step(trace_id, "policy_enforcement", input_summary=sql_retry, output_summary={"reason": last_code, "retry": True}, status="error", duration_ms=int((time.time() - _retry_exec_t0) * 1000))
                    except Exception as retry_exec_err:
                        exec_error = str(retry_exec_err)
                        log.warning("Retry execution failed for %s: %s",
                                    account_id, exec_error[:100])
                        _trace_step(trace_id, "execute_sql", input_summary=sql_retry, output_summary=exec_error, status="error", duration_ms=int((time.time() - _retry_exec_t0) * 1000))
                else:
                    last_reason, last_code = reason2, code2
                    log.warning("Retry still invalid for %s: %s",
                                account_id, reason2[:120])
        except Exception as retry_err:
            log.warning("Retry LLM call failed for %s: %s",
                        account_id, str(retry_err)[:100])

    # A first repair can legitimately clear one validator layer and expose a
    # different one (for example graph_plan_mismatch -> field_plan_mismatch).
    # Permit one more code-directed attempt only when the reason code changed;
    # the same code twice is a non-progress loop and remains terminal.
    _progressive_repair_allowed = bool(
        not ok
        and exec_error is None
        and (last_code or "") in _sql_repair_reason_codes
        and allow_progressive_sql_repair(
            _repair_reason_codes_seen,
            last_code,
            _llm_repair_attempts,
            max_attempts=2,
        )
    )
    if _progressive_repair_allowed:
        _prior_repair_code = next(iter(_repair_reason_codes_seen), "validation_error")
        _repair_reason_codes_seen.add(str(last_code or "").strip().casefold())
        _progressive_user = (
            "The previous repair made progress but exposed a different validation failure.\n"
            f"Previous failure code: {_prior_repair_code}\n"
            f"Current failure code: {last_code}\n"
            f"Current validation reason: {last_reason}\n"
            f"Current SQL: {sql_retry}\n\n"
            f"Original question: {question}\n\n"
            "Fix only the current failure while preserving the corrections already made. "
            "Follow the executable analytical request plan, approved metric formulas, exact "
            "date-role contract, and entity-graph joins in the system prompt. Return one "
            "complete SQL SELECT statement and no explanation."
        )
        try:
            await _send_live_stage(
                adapter,
                event,
                "repairing_query",
                "Completing query repair",
                "The first correction exposed another validation issue; applying one bounded follow-up repair.",
            )
            _progressive_t0 = time.time()
            _llm_repair_attempts += 1
            with llm_audit_scope(
                account_id=account_id,
                question=question,
                enabled=audit_enabled,
                request_id=audit_request_id,
                question_id=audit_request_id,
                component="sql_repair_progressive",
            ):
                _progressive_sql, _, _ = await llm_complete(
                    build_sql_system_prompt(
                        db_cfg["db_type"],
                        context_with_terms,
                        graph_context=_graph_ctx or None,
                        semantic_plan=_retry_plan,
                        # Without this the question-gate has nothing to read, and
                        # the repair prompt arrives carrying every optional rule —
                        # broader than the prompt that just failed.
                        question=question,
                    ),
                    _progressive_user,
                    provider,
                    model,
                    api_key,
                    temperature=0.0,
                    max_tokens=_sql_repair_max_tokens,
                    **az_kwargs,
                )
            _trace_step(
                trace_id,
                "llm_generate_sql",
                output_summary={"retry": True, "repair_attempt": _llm_repair_attempts},
                duration_ms=int((time.time() - _progressive_t0) * 1000),
            )
            if _progressive_sql.startswith("```"):
                _progressive_sql = "\n".join(_progressive_sql.split("\n")[1:]).rsplit("```", 1)[0].strip()
            _progressive_sql = _inject_distinct_if_needed(_progressive_sql, question)
            _progressive_sql = normalize_generated_sql(_progressive_sql, db_cfg["db_type"])

            if "CANNOT_GENERATE" not in _progressive_sql.upper() and len(_progressive_sql) > 10:
                _progressive_validate_t0 = time.time()
                _progressive_ok, _progressive_reason, _progressive_code = validate_sql(
                    _progressive_sql,
                    all_known,
                    db_cfg["db_type"],
                    query_scope_tables,
                    all_columns,
                    _retry_semantic_context,
                )
                _trace_step(
                    trace_id,
                    "validate_sql",
                    input_summary=_progressive_sql,
                    output_summary=_progressive_reason,
                    status="success" if _progressive_ok else "error",
                    metadata={
                        "code": _progressive_code,
                        "retry": True,
                        "repair_attempt": _llm_repair_attempts,
                    },
                    duration_ms=int((time.time() - _progressive_validate_t0) * 1000),
                )
                if _progressive_ok:
                    _progressive_exec_t0 = time.time()
                    try:
                        await _send_live_stage(
                            adapter,
                            event,
                            "executing_query",
                            "Retrying query",
                            "Running the corrected query against your data.",
                        )
                        _loop = asyncio.get_running_loop()
                        governed = await asyncio.wait_for(
                            _loop.run_in_executor(
                                None,
                                _execute_with_policy,
                                _progressive_sql,
                                _retry_semantic_context,
                            ),
                            timeout=_query_wait_timeout(db_cfg),
                        )
                        rows = governed.rows
                        _rows_truncated = bool(getattr(governed, "truncated", False))
                        sql = governed.sql
                        exec_error = None
                        ok, last_reason, last_code = True, "OK", "ok"
                        retry_count = _llm_repair_attempts
                        log.info(
                            "Progressive repair succeeded for %s after %d attempts",
                            account_id,
                            _llm_repair_attempts,
                        )
                        _trace_step(
                            trace_id,
                            "execute_sql",
                            input_summary=sql,
                            output_summary={
                                "rows": len(rows),
                                "retry": True,
                                "repair_attempt": _llm_repair_attempts,
                            },
                            duration_ms=int((time.time() - _progressive_exec_t0) * 1000),
                        )
                    except asyncio.TimeoutError:
                        exec_error = "Progressive repair query timed out after 3 minutes."
                        log.warning("Progressive repair query timed out for %s", account_id)
                    except PolicyDeniedError as policy_error:
                        exec_error = None
                        ok = False
                        last_reason = policy_error.decision.explanation or "Blocked by regulated data policy."
                        last_code = policy_error.decision.reason_code
                    except Exception as progressive_exec_err:
                        exec_error = str(progressive_exec_err)
                        log.warning(
                            "Progressive repair execution failed for %s: %s",
                            account_id,
                            exec_error[:100],
                        )
                else:
                    last_reason, last_code = _progressive_reason, _progressive_code
                    log.warning(
                        "Progressive repair still invalid for %s: %s",
                        account_id,
                        _progressive_reason[:120],
                    )
        except Exception as progressive_repair_err:
            log.warning(
                "Progressive repair LLM call failed for %s: %s",
                account_id,
                str(progressive_repair_err)[:100],
            )

    # ── Terminal failure handling ────────────────────────────────────────────
    # Raw reasons/errors stay in query_log + answer_trace (audit unchanged);
    # only the chat message is translated to business language with a next step.
    #
    # exec_error is None here on the validation path: when the FIRST attempt
    # failed validation and the REPAIRED query then passed validation but
    # failed at execution (e.g. the database was paused/unreachable),
    # exec_error holds that real terminal event — reporting the stale
    # pre-repair validation message instead would hide an infra outage
    # behind a validator code (live case: 'temporal_anchor_missing' shown
    # while the actual final failure was "Database 'chatbot_db' unavailable").
    if not ok and exec_error is None:
        _log_q(account_id, question, sql, 0, False, last_reason, provider, model,
               tok_in, tok_out, int(time.time()*1000) - start_ms,
               portal_user_id=pu_id, zoom_user_id=zid,
               question_id=audit_request_id, error_code=last_code or "")
        _trace_finish(trace_id, status="error", answer_type="error", error_message=last_reason)
        from core.failure_messages import translate_failure, suggest_closest_terms, _VALIDATION_REASONS
        if (last_code or "").lower() in _VALIDATION_REASONS:
            _suggest = (
                suggest_closest_terms(question, account_id, state.get("kb_dir", ""))
                if (last_code or "").lower() in {"unknown_column", "cannot_generate", "field_plan_mismatch"}
                else []
            )
            _rca = translate_failure(
                kind="validation", code=last_code, reason=last_reason,
                sql=sql, question=question,
                suggestions=_suggest,
            )
            from core.answer_formatter import format_failure_business_response
            await adapter.send_message(event, format_failure_business_response(
                rca=_rca, sql=sql, sql_preview_fn=_sql_preview,
            ))
        else:
            # Policy denials and other non-validator codes already carry a
            # business-written explanation — pass through untouched.
            await adapter.send_message(event, f"❌ {last_reason}")
        return

    # _timed_out was decided from the FIRST execution, to suppress the repair
    # loop. But exec_error is reassigned by the retry and progressive-repair
    # executions below it, either of which can itself time out. Reading the
    # stale flag here reported those as a generic execution failure — "the
    # database could not be reached" — for a query that in fact ran to the
    # statement timeout, hiding the one diagnosis that matters and dropping the
    # index guidance. Re-derive it from the error actually being reported.
    _timed_out = _timed_out or (
        exec_error is not None and is_query_timeout(exec_error)
    )

    if exec_error is not None or rows is None:
        _log_q(account_id, question, sql, 0, False,
               exec_error or "Unknown error",
               provider, model, tok_in, tok_out,
               int(time.time()*1000) - start_ms,
               portal_user_id=pu_id, zoom_user_id=zid,
               question_id=audit_request_id, error_code="execution_error")
        _trace_finish(trace_id, status="error", answer_type="error", error_message=exec_error or "Unknown error")
        from core.failure_messages import translate_failure
        from core.answer_formatter import format_failure_business_response
        _rca = translate_failure(
            kind="execution", exception_text=exec_error or "Unknown error",
            sql=sql, question=question,
        )
        if _timed_out:
            # "The database did not respond in time" is true but not actionable.
            # The governed plan names the exact column this question filters and
            # joins on, so name the index that would make it fast instead.
            from core.failure_messages import build_query_timeout_guidance

            _timeout_guidance = build_query_timeout_guidance(
                _semantic_plan,
                timeout_seconds=_query_timeout_seconds(db_cfg),
            )
            _rca["most_likely_reason"] = _timeout_guidance["reason"]
            _rca["suggested_next_step"] = _timeout_guidance["next_step"]
            _rca.setdefault("technical_notes", []).append(
                "The SQL was valid and governed; it was not repaired, because "
                "rewriting a query cannot make the database faster."
            )
        if not ok and last_code:
            # A validation failure preceded this: the repaired query passed
            # validation but died at execution. Keep the history visible so
            # the two failures aren't conflated.
            _rca.setdefault("technical_notes", []).append(
                f"A repaired query passed validation but failed to execute. "
                f"Original validation failure: {last_code}."
            )
        await adapter.send_message(event, format_failure_business_response(
            rca=_rca, sql=sql, sql_preview_fn=_sql_preview,
        ))
        return

    duration_ms = int(time.time()*1000) - start_ms
    _log_q(account_id, question, sql, len(rows), True, "", provider, model,
           tok_in, tok_out, duration_ms,
           portal_user_id=pu_id, zoom_user_id=zid,
           question_id=audit_request_id)
    # A repair may have replaced the first draft. Persist only the SQL that
    # actually executed so another channel can reuse the validated plan.
    _trace_update(
        trace_id,
        generated_sql=sql,
        sql_validation_status="pass",
        sql_validation_error="",
        query_row_count=len(rows),
    )

    # Zero rows: clarify only when an ambiguity signal exists and the bounded
    # loop permits a distinct next question. Blind LLM ambiguity checks
    # on every empty result set waste tokens and confuse users whose filter
    # was legitimately correct but the data genuinely has no rows.
    if len(rows) == 0 and event.user_id and can_request_clarification(event):
        _zr_matches = store.match_terms_in_question(account_id, question, query_scope_tables)
        _zr_has_required = any(
            m.get("requires_clarification") and m.get("clarification_options")
            for m in _zr_matches
        )
        _zr_has_multi_metric = len([m for m in _zr_matches if m.get("kind") == "metric"]) >= 2

        if not (_zr_has_required or _zr_has_multi_metric):
            # No ambiguity signal: return business-readable RCA for the empty result.
            _zr_tables = extract_sql_tables(sql, db_cfg.get("db_type", "azure_sql"))
            _zr_counts = await asyncio.to_thread(_count_tables_for_zero_row, db_cfg, _zr_tables)
            _zr_empty_tables = [table for table, count in _zr_counts.items() if count == 0]
            _trace_finish(trace_id, status="success", answer_type="empty", row_count=0, duration_ms=duration_ms, final_answer_summary="Query returned no rows")
            await adapter.send_message(event, _build_zero_row_message(
                question,
                sql,
                _graph_ctx,
                last_code or code or "ok",
                retry_count,
                tables_used=_zr_tables,
                empty_tables=_zr_empty_tables,
                semantic_plan=_semantic_plan,
                account_id=account_id,
            ))
            return

        with llm_audit_scope(
            account_id=account_id,
            question=question,
            enabled=audit_enabled,
            request_id=audit_request_id,
            question_id=audit_request_id,
            component="clarification",
        ):
            is_ambiguous, clarifying_q, cmeta = await check_ambiguity_glossary_first(
                account_id, question, context, provider, model, api_key, az_kwargs,
                allowed_tables=query_scope_tables,
            )
        _clarification_source = str((cmeta or {}).get("source") or "llm")
        if (
            is_ambiguous
            and clarifying_q
            and can_request_clarification(event, _clarification_source)
        ):
            cmeta = dict(cmeta or {})
            cmeta["question"] = clarifying_q
            opts = (cmeta or {}).get("options") or []
            if opts:
                _save_pending_clarification(question, context, cmeta)
                send_prompt = getattr(adapter, "send_clarification_prompt", None)
                if callable(send_prompt):
                    await send_prompt(event, clarifying_q or "I need a bit more context to answer that.", opts)
                else:
                    await adapter.send_message(event,
                        f"The query ran successfully but returned *no results*.\n\n"
                        f"❓ {clarifying_q}\n\n"
                        f"_Reply with one of the listed clarification options and I'll rerun the query._"
                    )
            else:
                _save_pending_clarification(question, context, cmeta)
                await adapter.send_message(event,
                    f"The query ran successfully but returned *no results*.\n\n"
                    f"❓ {clarifying_q}\n\n"
                    f"_Reply in plain language and I'll continue with your original question._"
                )
            _trace_finish(
                trace_id,
                status="success",
                answer_type="clarification",
                row_count=0,
                duration_ms=duration_ms,
                final_answer_summary="Query returned no rows; requested clarification",
            )
            return

    await _send_live_stage(adapter, event, "formatting_results", "Preparing results", "Formatting the answer and any chart for display.")
    # Record this turn in conversation history (web portal only)
    _add_history = getattr(adapter, "add_to_history", None)
    if callable(_add_history) and rows:
        _add_history(
            question=extract_original_question(question),
            sql=sql,
            columns=list(rows[0].keys()) if rows else [],
            row_count=len(rows),
        )
    from core.metric_semantics import detect_derived_metric_gap
    _resolved_graph_edges = (_graph_ctx or {}).get("resolved_edges") or []
    _graph_fanout_risk = bool((_graph_ctx or {}).get("fanout_risk_facts"))
    if not _graph_fanout_risk:
        for _edge in _resolved_graph_edges:
            try:
                _edge_fanout = float(_edge.get("fanout_ratio") or 0)
            except (TypeError, ValueError):
                _edge_fanout = 0.0
            if _edge.get("many_to_many") or _edge_fanout > 1.01:
                _graph_fanout_risk = True
                break
    _confidence_context = {
        "validation_code": last_code or code or "ok",
        "retry_count": retry_count,
        # The fetch stopped at its row cap, so `rows` is the head of a larger
        # result. Row-level statistics were refused upstream; the renderer
        # turns this into a caveat so a missing chart has a stated reason.
        "rows_truncated": _rows_truncated,
        "has_semantic_plan": bool((_semantic_plan or {}).get("enabled")),
        # Planning raised, as opposed to finding nothing to bind.
        "semantic_planning_failed": bool((_semantic_plan or {}).get("planning_failed")),
        "has_graph_context": bool((_graph_ctx or {}).get("enabled") or (_graph_ctx or {}).get("detected")),
        # Resolution raised rather than returning "no graph". The difference
        # matters to the reader: a single-table answer needs no join governance,
        # but a multi-table one built while governance was down has joins
        # nothing checked.
        "graph_resolution_failed": bool((_graph_ctx or {}).get("resolution_error")),
        "tables_used": extract_sql_tables(sql, db_cfg.get("db_type", "azure_sql")),
        # "open order quantity" with no approved formula anywhere = the LLM
        # could only SUM a raw column; surface that as a confidence warning
        # instead of presenting a business-wrong number with full confidence.
        "derived_metric_gap": detect_derived_metric_gap(
            question,
            has_metric_formula=bool(metric_formula_context),
            has_term_expression=bool(term_injection),
        ),
        # Everything above the relevance floor was dropped — the KB context
        # the SQL was built on matched this question only weakly.
        "weak_retrieval": _weak_retrieval,
        # The re-ranker produced no scores at all, so the relevance floor did
        # not run and weak_retrieval above could never become True. Retrieval
        # is unfiltered here, not confirmed relevant.
        "retrieval_unscored": _retrieval_unscored,
        "graph_scope": str((_graph_ctx or {}).get("graph_scope") or ""),
        "fanout_risk": _graph_fanout_risk,
        # Carried through to cache_result so compare_prior can read them.
        "semantic_plan": _semantic_plan or {},
        # The exact entity_relationships rows this question's JOIN path
        # resolved to (core.graph_resolver.resolve_for_question's own
        # "resolved_edges") -- lets _send_results check for a known-lossy
        # join (core/join_coverage.py) without re-parsing the generated SQL.
        "graph_edges": (_graph_ctx or {}).get("resolved_edges") or [],
    }

    # ── Post-processing: apply contribution / anomaly analytics ──────────────
    # Run after DB execution, before _send_results.  Augments rows in-place
    # with computed columns so the frontend can render them directly.
    _post_intents = getattr(event, '_analytic_intents', None) or _intents if '_intents' in dir() else {}
    if rows and _post_intents:
        try:
            if _post_intents.get("contribution") and not any("contribution_pct" in r for r in rows[:1]):
                from core.contribution_analysis import compute_contribution, infer_numeric_col as _inc
                _val_col = _inc(rows)
                if _val_col:
                    rows = compute_contribution(rows, _val_col)
                    log.info("post_process: contribution_pct added for col=%s", _val_col)

            if _post_intents.get("anomaly") and not any("anomaly_flag" in r for r in rows[:1]):
                from core.anomaly_detection import detect_anomalies, infer_value_col as _ivc
                _val_col = _ivc(rows)
                if _val_col:
                    _anom_result = detect_anomalies(rows, _val_col)
                    rows = _anom_result.rows
                    log.info(
                        "post_process: anomaly detection complete col=%s flagged=%d/%d",
                        _val_col, _anom_result.flagged_rows, _anom_result.total_rows,
                    )

            # ── Tier 2 post-processing ────────────────────────────────────────
            if _post_intents.get("budget_vs_actual") and not any("variance" in r for r in rows[:1]):
                from core.budget_vs_actual import infer_bva_cols, compute_bva
                _a_col, _b_col = infer_bva_cols(rows)
                if _a_col and _b_col:
                    rows = compute_bva(rows, _a_col, _b_col)
                    log.info("post_process: bva variance added actual=%s budget=%s", _a_col, _b_col)

            if _post_intents.get("cohort") and not any("cohort" in str(list(r.keys())) for r in rows[:1]):
                from core.cohort_analysis import infer_cohort_cols, compute_cohort_matrix
                _cohort_col, _period_col, _value_col = infer_cohort_cols(rows)
                if _cohort_col and _period_col and _value_col:
                    rows = compute_cohort_matrix(rows, _cohort_col, _period_col, _value_col,
                                                 truncated=_rows_truncated)
                    log.info("post_process: cohort matrix built cohort=%s period=%s", _cohort_col, _period_col)

            if _post_intents.get("correlation") and not any("__corr_r" in r for r in rows[:1]):
                from core.correlation_analysis import infer_corr_cols, compute_correlation, annotate_rows_with_correlation
                _x_col, _y_col = infer_corr_cols(rows, question)
                if _x_col and _y_col:
                    _corr = compute_correlation(rows, _x_col, _y_col, truncated=_rows_truncated)
                    rows = annotate_rows_with_correlation(rows, _corr)
                    log.info("post_process: correlation r=%.4f (%s) n=%d", _corr.pearson_r or 0, _corr.interpretation, _corr.n)

            if _post_intents.get("pivot") and not any("TOTAL" in r for r in rows[:1]):
                from core.pivot_table import infer_pivot_cols, compute_pivot_table
                _rk, _ck, _vk = infer_pivot_cols(rows)
                if _rk and _ck and _vk:
                    rows = compute_pivot_table(rows, _rk, _ck, _vk)
                    log.info("post_process: pivot table built row=%s col=%s value=%s", _rk, _ck, _vk)

            # ── Tier 3 post-processing ────────────────────────────────────────
            if _post_intents.get("funnel") and not any("funnel_pct" in r for r in rows[:1]):
                from core.funnel_analysis import infer_funnel_cols, compute_funnel
                _s_col, _c_col = infer_funnel_cols(rows)
                if _s_col and _c_col:
                    rows = compute_funnel(rows, _s_col, _c_col)
                    log.info("post_process: funnel computed stage=%s count=%s", _s_col, _c_col)

            if _post_intents.get("forecast") and not any("is_forecast" in r for r in rows[:1]):
                from core.forecast import infer_forecast_cols, compute_forecast, extract_forecast_periods
                _p_col, _v_col = infer_forecast_cols(rows)
                if _p_col and _v_col:
                    _n_fc = extract_forecast_periods(question)
                    rows = compute_forecast(rows, _p_col, _v_col, _n_fc)
                    log.info("post_process: forecast appended periods=%d period=%s value=%s", _n_fc, _p_col, _v_col)

            if _post_intents.get("histogram") and not any("bin_label" in r for r in rows[:1]):
                from core.distribution_analysis import infer_histogram_col, compute_histogram
                _h_col = infer_histogram_col(rows)
                if _h_col:
                    rows = compute_histogram(rows, _h_col, truncated=_rows_truncated)
                    log.info("post_process: histogram binned col=%s bins=%d", _h_col, len(rows))

            if _post_intents.get("boxplot") and not any("bp_data" in r for r in rows[:1]):
                from core.distribution_analysis import infer_boxplot_cols, compute_boxplot
                _g_col, _v_col = infer_boxplot_cols(rows)
                if _v_col:
                    rows = compute_boxplot(rows, _v_col, _g_col, truncated=_rows_truncated)
                    log.info("post_process: boxplot computed group=%s value=%s", _g_col, _v_col)

            if _post_intents.get("whatif") and not any(k.startswith("scenario_") for k in (rows[0] if rows else {})):
                from core.whatif import compute_whatif
                _wi_p = getattr(event, '_whatif_params', None)
                if _wi_p:
                    rows = compute_whatif(rows, _wi_p)
                    log.info("post_process: what-if scenario applied delta_pct=%s", _wi_p.delta_pct)

        except Exception as _pp_exc:
            log.debug("Post-processing analytics skipped: %s", _pp_exc)

    # SQL/schema validation proves the statement is safe to run; this second,
    # deterministic check verifies that the returned columns and row shape
    # still match the requested analytical intent (trend, ranking,
    # distribution, KPI, and so on).  It is metadata-only and never logs row
    # values.  A mismatch lowers answer confidence instead of silently showing
    # a schema-valid but business-wrong shape.
    try:
        from core.result_verifier import verify_result_shape

        _result_verification = verify_result_shape(
            rows,
            analytical_plan=_analytical_plan,
            resolution_plan=_resolution_plan,
            request_plan=_analytical_request_plan,
        )
        _confidence_context["result_verification"] = _result_verification
        _trace_step(
            trace_id,
            "result_shape_verification",
            output_summary={
                "status": _result_verification.get("status"),
                "score": _result_verification.get("score"),
                "row_count": _result_verification.get("row_count"),
                "columns": _result_verification.get("columns") or [],
                "metric_binding_source": _result_verification.get("metric_binding_source"),
                "warnings": _result_verification.get("warnings") or [],
                "errors": _result_verification.get("errors") or [],
            },
        )
    except Exception as _verification_exc:
        # This is the product's stated defence against a schema-valid but
        # business-wrong answer — the check that a "top 5 by region" question
        # came back with five rows and a region column. Failing it open at debug
        # left the answer scored exactly as if the shape had been confirmed:
        # verification contributes +5 when it passes and nothing at all when it
        # is absent, so a silent failure reads as "no issues found".
        log.error(
            "Result-shape verification FAILED for %s on %r — the answer's shape "
            "was not checked against what was asked",
            account_id, question[:200], exc_info=True,
        )
        _confidence_context["result_verification"] = {
            "status": "unavailable",
            "errors": [
                "The result could not be checked against the shape of the "
                "question, so nothing confirms it answers what was asked."
            ],
        }
        _trace_step(
            trace_id,
            "result_shape_verification",
            output_summary=str(_verification_exc)[:500],
            status="error",
            metadata={"shape_verified": False},
        )

    await _send_results(event, adapter, question, rows, sql, duration_ms,
                        portal_user, account_id, db_cfg,
                        rag_context=context, question_id=audit_request_id,
                        confidence_context=_confidence_context,
                        display_context={
                            "format_scope": "metric_context",
                            "metrics": _matched_metrics,
                        },
                        contract_version=_contract_version)
    # Only a successful query turns the clarification choice into a thread
    # preference. Failed validation/execution must not teach the session a
    # potentially unusable date role, and this metadata is never promoted to
    # the tenant's governed semantic model.
    if _confirmed_date_role and _selected_date_bindings:
        try:
            _confirmed_binding = next(
                (
                    binding for binding in _selected_date_bindings
                    if str(binding.get("resolution_source") or "")
                    == "user_confirmed_date_role"
                ),
                {},
            )
            if _confirmed_binding:
                conversation_state_store.remember_date_preference(
                    account_id,
                    clarification_session_id(adapter, event),
                    _confirmed_binding,
                    metric_names=[
                        metric.get("name") for metric in _matched_metrics
                        if metric.get("name")
                    ],
                    fact_tables=[_confirmed_binding.get("fact_table")],
                )
        except Exception as _date_preference_exc:
            log.debug(
                "Thread date preference could not be persisted: %s",
                _date_preference_exc,
            )
    if _why_mode and rows:
        await _send_why_insight(
            adapter, event,
            question=question, rows=rows, sql=sql,
            client=client, account_id=account_id, db_cfg=db_cfg,
            rag_context=context,
            known_tables=all_known,
            query_executor=lambda _cfg, _s: _execute_with_policy(_s),
            question_id=audit_request_id,
        )
    _trace_finish(trace_id, status="success", answer_type="table", row_count=len(rows), duration_ms=duration_ms, final_answer_summary="Answered from database query")

    # ── Learning loop: persist quality candidate ──────────────────────────────
    # Runs AFTER the response is already on the wire — zero user-facing latency.
    # Gated by enable_feedback_collection so it's a no-op until the pilot is on.
    if client.get("enable_feedback_collection"):
        _create_learning_candidate(
            account_id        = account_id,
            question_id       = audit_request_id,
            question          = question,
            sql               = sql,
            validation_passed = ok,
            had_repair        = retry_count > 0,
            repair_succeeded  = retry_count > 0,   # if we're here after retries, repair worked
            row_count         = len(rows),
            confidence_ctx    = _confidence_context,
            schema_scope      = schema_hint,
            kb_dir            = state.get("kb_dir", ""),
            schema_dir        = state.get("schema_dir", ""),
        )


async def handle_query(
    account_id, event, adapter, question, portal_user, is_clarification=False,
):
    """Run the governed pipeline with a terminal answer-trace guarantee."""
    # The client's vocabulary — the ERP pack an admin selected plus the
    # clients/<id>/vocab.json overlay where they add tenant-specific
    # abbreviations — was installed only during the KB build and the graph
    # autopopulate. Nothing installed it for a QUERY, so every query-time
    # consumer called get_active_vocab(), got the ContextVar's None, and fell
    # back to the builtin. An abbreviation the admin added expanded correctly in
    # the generated KB text and then failed to expand the question that needed
    # it, so the doc documenting that term was not retrieved.
    _vocab_token = None
    try:
        from core.vocab_packs import activate_vocab, vocab_for_account
        _vocab_token = activate_vocab(vocab_for_account(account_id))
    except Exception as _vocab_exc:
        log.error(
            "Could not activate the vocabulary pack for %s (%s) — this question "
            "is resolved against the builtin vocabulary only, so tenant-specific "
            "abbreviations will not expand",
            account_id, _vocab_exc, exc_info=True,
        )
    try:
        return await _handle_query_impl(
            account_id,
            event,
            adapter,
            question,
            portal_user,
            is_clarification=is_clarification,
        )
    except Exception as exc:
        _trace_finish_unclosed(
            status="error",
            answer_type="error",
            error_message=f"Unhandled query pipeline error: {exc}",
        )
        raise
    finally:
        if _vocab_token is not None:
            from core.vocab_packs import deactivate_vocab
            deactivate_vocab(_vocab_token)
        # Every expected return calls _trace_finish explicitly. This final
        # guard catches future early-return regressions without changing the
        # user-facing response or swallowing an exception.
        _trace_finish_unclosed(
            status="error",
            answer_type="error",
            error_message="Query pipeline ended without a terminal outcome",
        )


