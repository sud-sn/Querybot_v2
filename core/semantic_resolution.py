"""
Sprint 3 — the deterministic query-time resolution plan.

Today, "what does this question mean" is resolved by four INDEPENDENT
pipeline steps in core/query_pipeline.py, each producing its own partial
structure that gets text-injected into the SQL-generation prompt
separately, with no single object anyone can point to and say "this is
what got resolved, and how confident are we":

    core.metric_scope.resolve_metric_scope() -> _matched_metrics list[dict]
    core.graph_resolver.resolve_for_question()      -> graph_ctx dict
    core.semantic_planner.build_semantic_field_plan()
      + core.semantic_model.build_runtime_semantic_plan()
                                        -> merged semantic_plan dict
    core.contextual_dates.resolve_contextual_date_binding()
                                        -> date_context_resolution dict

build_resolution_plan() is a pure function that ASSEMBLES those four
already-computed structures into the ONE structured plan the Sprint 2/3
plan document describes — it does not re-derive or second-guess any of
them (this module has no I/O and calls none of the resolvers itself; the
caller passes in what it already computed). Two things it adds that don't
exist anywhere today:

  1. A single confidence score and resolved_deterministically flag,
     computed from signals that were previously scattered across four
     return values with no shared vocabulary.

  2. Cross-referencing what THIS question actually touches (by canonical
     id) against the Sprint 2 detectors' OPEN compile-time conflicts —
     the first place compile-time governance (Sprint 1/2) and query-time
     answering meet. An ERROR-severity conflict touching a resolved
     object becomes a blocking clarification; WARNING/INFO/STALE become
     non-blocking advisories, mirroring the same severity contract
     governed_recompile_contract already uses for publish-blocking.

Sprint 4 built on top of this module:
  - check_sql_plan_coverage() (below) compares generated SQL's table usage
    against this plan — advisory-only, never blocking; an LLM legitimately
    touching an unlisted join/lookup table is common and not a defect.
  - core/query_pipeline.py now actually gates on `clarifications` — but
    only in "enforce" mode (see store.get_semantic_compiler_state). In
    "off"/"shadow" this plan is still built and traced only, exactly as
    Sprint 3 left it.
"""

from __future__ import annotations

from typing import Any

from core.semantic_ids import date_role_id, field_id, join_id, metric_id


def _touched_canonical_ids(
    *,
    matched_metrics: list[dict[str, Any]] | None,
    graph_ctx: dict[str, Any] | None,
    semantic_plan: dict[str, Any] | None,
    date_binding: dict[str, Any] | None,
) -> set[str]:
    """Every canonical id this question's resolution actually references —
    the set open conflicts get cross-referenced against."""
    touched: set[str] = set()

    for metric in matched_metrics or []:
        if isinstance(metric, dict) and metric.get("id") is not None:
            try:
                touched.add(metric_id(metric["id"]))
            except (TypeError, ValueError):
                pass

    for edge_id in (graph_ctx or {}).get("edge_ids") or []:
        if edge_id is not None:
            touched.add(join_id(edge_id))

    for field in (semantic_plan or {}).get("fields") or []:
        if isinstance(field, dict) and field.get("table") and field.get("column"):
            touched.add(field_id(str(field["table"]), str(field["column"])))

    if isinstance(date_binding, dict) and date_binding.get("fact_table") and date_binding.get("fact_column"):
        touched.add(date_role_id(str(date_binding["fact_table"]), str(date_binding["fact_column"])))

    return touched


def _conflict_participant_ids(conflict: dict[str, Any]) -> set[str]:
    """Recover every canonical id a conflict names, from its conflict_key
    (core.semantic_ids.conflict_key() format: "code::id1|id2|..."). Falls
    back to object_id alone if the key is missing or malformed — never
    raises, a conflict row is display data here, not something this module
    can afford to crash on."""
    key = str(conflict.get("conflict_key") or "")
    _, _, ids_part = key.partition("::")
    ids = {part for part in ids_part.split("|") if part}
    if not ids and conflict.get("object_id"):
        ids = {str(conflict["object_id"])}
    return ids


def build_resolution_plan(
    *,
    account_id: str,
    question: str,
    contract: dict[str, Any] | None = None,
    matched_metrics: list[dict[str, Any]] | None = None,
    graph_ctx: dict[str, Any] | None = None,
    semantic_plan: dict[str, Any] | None = None,
    date_context_resolution: dict[str, Any] | None = None,
    schema_hint: str = "",
    allowed_tables: set[str] | None = None,
    open_conflicts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Assemble the structured resolution plan for one question from whatever
    the pipeline already resolved. Every parameter is optional and defaults
    to "nothing resolved" — a half-resolved question (e.g. no metric
    matched, LLM route) still produces a valid, inspectable plan instead of
    this function needing a special case for every combination of what
    happened to be available.

    matched_metrics is a LIST (not a single dict) to match
    core/query_pipeline.py's own _matched_metrics — a comparison question
    ("revenue vs. margin") can legitimately have more than one relevant
    approved metric, and every one of them should participate in both the
    plan's metrics section and the conflict cross-reference below, not just
    whichever happened to be first.
    """
    contract = contract or {}
    graph_ctx = graph_ctx or {}
    semantic_plan = semantic_plan or {}
    date_context_resolution = date_context_resolution or {}
    open_conflicts = open_conflicts or []
    matched_metrics = matched_metrics or []

    date_binding: dict[str, Any] | None = None
    if date_context_resolution.get("status") in {"selected", "selected_many"}:
        date_binding = date_context_resolution.get("binding") or next(
            iter(date_context_resolution.get("bindings") or []), None,
        )

    metrics: list[dict[str, Any]] = []
    for metric in matched_metrics:
        if isinstance(metric, dict) and metric.get("id") is not None:
            metrics.append({
                "canonical_id": metric_id(metric["id"]),
                "name": metric.get("name") or "",
                "source": "metric_registry",
            })

    dimensions: list[dict[str, Any]] = []
    for field in semantic_plan.get("fields") or []:
        if not isinstance(field, dict) or not field.get("table") or not field.get("column"):
            continue
        dimensions.append({
            "canonical_id": field_id(str(field["table"]), str(field["column"])),
            "term": field.get("term") or "",
            "table": field.get("table"),
            "column": field.get("column"),
            "enforcement": field.get("enforcement") or "required",
        })

    date_roles: list[dict[str, Any]] = []
    if date_binding:
        date_roles.append({
            "canonical_id": date_role_id(
                str(date_binding.get("fact_table") or ""), str(date_binding.get("fact_column") or ""),
            ),
            "context_name": date_binding.get("context_name") or "",
            "date_role": date_binding.get("date_role") or "",
            "resolution_source": date_binding.get("resolution_source") or "",
        })

    graph_path: list[dict[str, Any]] = []
    for edge in graph_ctx.get("resolved_edges") or []:
        if not isinstance(edge, dict):
            continue
        graph_path.append({
            "canonical_id": join_id(edge["id"]) if edge.get("id") is not None else "",
            "from_entity": edge.get("from_entity") or "",
            "to_entity": edge.get("to_entity") or "",
            "join_type": edge.get("join_type") or "",
            "validation_status": edge.get("validation_status") or "untested",
        })

    schemas = sorted({schema_hint.upper()}) if schema_hint else []

    touched = _touched_canonical_ids(
        matched_metrics=matched_metrics, graph_ctx=graph_ctx,
        semantic_plan=semantic_plan, date_binding=date_binding,
    )

    clarifications: list[dict[str, Any]] = []
    advisories: list[dict[str, Any]] = []

    fanout_risk_facts = graph_ctx.get("fanout_risk_facts") or []
    if fanout_risk_facts:
        advisories.append({
            "code": "graph_fanout_risk",
            "severity": "WARNING",
            "message": (
                "Join skeleton connects fact table(s) " + ", ".join(fanout_risk_facts)
                + " to the anchor only through a shared dimension key — flattening this "
                "into one join chain multiplies rows before aggregation and can silently "
                "inflate sums/counts. The SQL prompt was instructed to pre-aggregate these "
                "via CTEs instead of a flat join."
            ),
            "object_id": ", ".join(fanout_risk_facts),
        })

    if date_context_resolution.get("status") == "ambiguous":
        clarifications.append({
            "type": "date_context_ambiguous",
            "message": date_context_resolution.get("reason") or "Multiple valid business dates for this metric.",
            "options": date_context_resolution.get("options") or [],
        })

    for conflict in open_conflicts:
        if not isinstance(conflict, dict):
            continue
        if not (_conflict_participant_ids(conflict) & touched):
            continue
        entry = {
            "code": conflict.get("code") or "",
            "severity": conflict.get("severity") or "",
            "message": conflict.get("message") or "",
            "object_id": conflict.get("object_id") or "",
        }
        if conflict.get("severity") == "ERROR":
            clarifications.append({"type": "compile_time_conflict", **entry})
        else:
            advisories.append(entry)

    confidence = 100.0
    if not metrics and not dimensions:
        # Nothing structural resolved at all — the LLM is working with no
        # deterministic grounding beyond retrieved KB text.
        confidence -= 40.0
    if date_context_resolution.get("status") == "ambiguous":
        confidence -= 25.0
    confidence -= min(30.0, 15.0 * sum(1 for c in clarifications if c.get("type") == "compile_time_conflict"))
    confidence -= min(20.0, 5.0 * len(advisories))
    confidence = max(0.0, min(100.0, confidence))

    return {
        "contract_version": (contract.get("meta") or {}).get("contract_version", ""),
        "account_id": account_id,
        "question": question,
        "schemas": schemas,
        "metrics": metrics,
        "dimensions": dimensions,
        "date_roles": date_roles,
        "graph_path": graph_path,
        "access_scope": {
            "schema_hint": schema_hint or "",
            "allowed_tables": sorted(allowed_tables) if allowed_tables is not None else None,
        },
        "clarifications": clarifications,
        "advisories": advisories,
        "confidence": round(confidence, 1),
        "resolved_deterministically": not clarifications,
    }


def check_sql_plan_coverage(
    sql: str, plan: dict[str, Any], db_type: str = "azure_sql",
) -> dict[str, Any]:
    """
    Sprint 4b — advisory-only comparison of what the generated SQL actually
    touches against what the resolution plan expected, table-level only
    (dimensions and date_roles are the only plan sections that carry table
    names; metrics and graph_path do not).

    Never blocks and is not a validator: a mismatch is often legitimate (the
    LLM may reasonably touch an unlisted join or lookup table that
    metric_scope/semantic_plan never resolved), so this is pure trace-level
    visibility for now — the same "observe first" step Sprint 2's detectors
    and Sprint 3's resolution plan itself both went through before anything
    built on them could gate an answer.
    """
    from core.answer_rca import extract_sql_tables
    from core.semantic_conflicts import _table_fqn_variants

    sql_tables = extract_sql_tables(sql, db_type)
    sql_variants: set[str] = set()
    for table in sql_tables:
        sql_variants |= _table_fqn_variants(table)

    expected_tables: set[str] = set()
    for dim in plan.get("dimensions") or []:
        table = dim.get("table")
        if table:
            expected_tables.add(str(table))
    for role in plan.get("date_roles") or []:
        canonical_id = str(role.get("canonical_id") or "")
        if canonical_id.startswith("date_role:") and "." in canonical_id:
            expected_tables.add(canonical_id.split(":", 1)[1].rsplit(".", 1)[0])

    expected = sorted(expected_tables)
    unused = [table for table in expected if not (_table_fqn_variants(table) & sql_variants)]
    covered = len(expected) - len(unused)
    coverage_ratio = 1.0 if not expected else round(covered / len(expected), 2)

    return {
        "sql_tables": sql_tables,
        "expected_tables": expected,
        "unused_expected_tables": unused,
        "coverage_ratio": coverage_ratio,
    }
