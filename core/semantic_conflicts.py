"""
Sprint 2 — governed semantic conflict detectors.

Pure functions over an already-compiled contract body (see
core/semantic_contract.py). No I/O, no store access, no exceptions escape a
single detector — one detector's bug must never take down the whole compile,
matching the compiler's existing "a source failure degrades, it never
crashes" philosophy for _compile_contract_internal.

Every detector returns dicts shaped exactly like
core.semantic_contract._source_conflict's output, because that shape is
already store.save_semantic_conflicts' input contract and already flows
through the Model Health panel, the mode-gated publish block, and
store.reconcile_semantic_conflicts — nothing downstream needs to change to
consume a new detector.

conflict_key MUST be built via core.semantic_ids.conflict_key() from the
canonical_id(s) already stamped onto the contract by
_stamp_canonical_ids() — never from display names — so the same real
misconfiguration reconciles to the same open conflict row across compiles
instead of opening a duplicate every run.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.semantic_ids import conflict_key

log = logging.getLogger("querybot.semantic_conflicts")


def _metric_canonical_ids(contract: dict[str, Any]) -> dict[int, str]:
    """metric_registry.id -> canonical_id, for detectors that only have the
    integer id (as date_contexts rows do) and need the stamped id to build a
    conflict_key with."""
    out: dict[int, str] = {}
    for metric in contract.get("metrics") or []:
        if not isinstance(metric, dict):
            continue
        mid = metric.get("id")
        cid = metric.get("canonical_id")
        if mid is not None and cid:
            try:
                out[int(mid)] = cid
            except (TypeError, ValueError):
                continue
    return out


def _table_fqn_variants(fqn: str) -> set[str]:
    """Same expansion core/schema.py::load_known_tables uses for FQN
    membership tests — a metric's base_table is free-text (bare name or
    FQN; see core/metric_scope.py's own multi-strategy resolution), so
    matching it to a compiled model table needs the same tolerance."""
    upper = str(fqn or "").upper().strip()
    if not upper:
        return set()
    variants = {upper}
    parts = upper.split(".")
    if len(parts) >= 2:
        variants.add(parts[-1])
        variants.add(".".join(parts[-2:]))
    return variants


def _table_by_base_table(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """base_table-variant -> compiled model table dict, for matching a
    metric's free-text base_table against the compiled model's tables."""
    lookup: dict[str, dict[str, Any]] = {}
    for table in (contract.get("model") or {}).get("tables") or []:
        if not isinstance(table, dict):
            continue
        fqn = str(table.get("qualified_name") or table.get("fqn") or "")
        for variant in _table_fqn_variants(fqn):
            lookup[variant] = table
    return lookup


def detect_ambiguous_date_roles(contract: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Two structural date-governance gaps, promoted from what
    core/contextual_dates.py already has to work around at query time:

    1. multiple_default_date_roles (ERROR) — a metric has more than one
       metric_date_context row marked is_default. The runtime resolver
       assumes exactly one default per metric; two is an outright
       misconfiguration, not a risk to weigh.

    2. missing_date_role_binding (WARNING) — a metric's base_table has 2+
       candidate business dates (model.tables[].date_roles) but the metric
       has ZERO governed date_context bindings. core/contextual_dates.py's
       _explicit_role_matches only matches date roles with
       status == "approved" (set via the /date-roles/approve route,
       core/semantic_model.py::patch_date_role) — a table's date roles sit
       at status="generated"/"needs_review" until an admin approves them,
       so an unbound metric on such a table has NO governed date steering
       at query time: SQL generation picks a date column with no
       contract-level guidance at all.
    """
    conflicts: list[dict[str, Any]] = []
    metric_ids = _metric_canonical_ids(contract)
    metrics_by_id = {
        int(m["id"]): m for m in (contract.get("metrics") or [])
        if isinstance(m, dict) and m.get("id") is not None
    }

    # ── Check 1: multiple defaults per metric ────────────────────────────
    by_metric: dict[int, list[dict[str, Any]]] = {}
    for binding in contract.get("date_contexts") or []:
        if not isinstance(binding, dict):
            continue
        try:
            mid = int(binding.get("metric_id") or 0)
        except (TypeError, ValueError):
            continue
        if mid:
            by_metric.setdefault(mid, []).append(binding)

    for mid, bindings in by_metric.items():
        defaults = [b for b in bindings if b.get("is_default")]
        if len(defaults) <= 1:
            continue
        metric_cid = metric_ids.get(mid)
        if not metric_cid:
            continue
        metric_name = str(metrics_by_id.get(mid, {}).get("name") or bindings[0].get("metric_name") or mid)
        context_names = [str(b.get("context_name") or "") for b in defaults]
        conflicts.append({
            "conflict_key": conflict_key("multiple_default_date_roles", metric_cid),
            "code": "multiple_default_date_roles",
            "severity": "ERROR",
            "object_type": "metric",
            "object_id": metric_cid,
            "schema_name": "",
            "table_name": "",
            "origin": "date_governance",
            "message": (
                f'Metric "{metric_name}" has {len(defaults)} date contexts marked '
                f"default ({', '.join(context_names)}) — exactly one is required."
            ),
            "evidence": {"metric_id": mid, "default_context_names": context_names},
            "suggestions": ["Mark only one date context as the default for this metric."],
        })

    # ── Check 2: base table has unresolved date-role ambiguity ──────────
    table_lookup = _table_by_base_table(contract)
    bound_metric_ids = set(by_metric.keys())
    for mid, metric in metrics_by_id.items():
        if mid in bound_metric_ids:
            continue  # has at least one governed binding — check 1 covers its shape
        metric_cid = metric_ids.get(mid)
        base_table = str(metric.get("base_table") or "").strip()
        if not metric_cid or not base_table:
            continue  # can't structurally resolve this metric's table — skip, don't guess
        table = None
        for variant in _table_fqn_variants(base_table):
            if variant in table_lookup:
                table = table_lookup[variant]
                break
        if table is None:
            continue
        date_roles = [r for r in (table.get("date_roles") or []) if isinstance(r, dict)]
        if len(date_roles) < 2:
            continue
        table_cid = table.get("canonical_id") or ""
        role_labels = [str(r.get("name") or r.get("business_role") or "") for r in date_roles]
        conflicts.append({
            "conflict_key": conflict_key(
                "missing_date_role_binding", metric_cid, table_cid or base_table,
            ),
            "code": "missing_date_role_binding",
            "severity": "WARNING",
            "object_type": "metric",
            "object_id": metric_cid,
            "schema_name": "",
            "table_name": str(table.get("qualified_name") or table.get("fqn") or base_table),
            "origin": "date_governance",
            "message": (
                f'Metric "{metric.get("name") or mid}" targets a table with '
                f"{len(date_roles)} candidate business dates ({', '.join(role_labels)}) "
                "but has no governed date-context binding — questions may resolve "
                "to an arbitrary date column."
            ),
            "evidence": {
                "metric_id": mid, "base_table": base_table,
                "candidate_date_roles": [r.get("canonical_id") for r in date_roles],
            },
            "suggestions": [
                "Add a metric date-context binding to make one of these dates the governed default.",
                "Or approve just the one unambiguous date role on the Date Roles page.",
            ],
        })

    return conflicts


def detect_synonym_collisions(contract: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Two distinct phrase-collision checks — same shape, different severity,
    because the two consequences are different in kind, not just degree:

    1. business_term_phrase_collision (ERROR) — two DIFFERENT active
       business_term rows share a matched phrase (term text or an alias)
       but resolve to a different canonical_expression. business_term has
       UNIQUE(account_id, term), but nothing constrains two different rows
       from listing the SAME alias. This is not a recoverable runtime
       ambiguity: store.match_terms_in_question() returns every matching
       row (see its docstring — "Find ALL business terms that appear"), and
       store.build_term_injection() emits one prompt line per surviving
       match, each with its own `use this EXACT expression` instruction.
       Two colliding terms means the SQL-generation prompt receives two
       CONTRADICTORY instructions for what looks like one concept — a
       correctness bug in the prompt itself, not a graceful clarification.

    2. field_synonym_collision (WARNING) — the same business_candidates
       phrase (core/schema_enrichment.py's per-column synonym list, used
       for KB narrative documentation, not direct SQL substitution) is a
       candidate name for 2+ DIFFERENT physical columns. Milder than #1:
       it can make the KB's business-vocabulary section ambiguous to a
       reader, but it doesn't inject contradictory SQL instructions the way
       a term collision does.

    Both checks skip a phrase whose colliders all point at the SAME
    canonical target — that's a harmless duplicate, not a conflict.
    """
    conflicts: list[dict[str, Any]] = []

    # ── Check 1: business term phrase collisions ─────────────────────────
    phrase_to_terms: dict[str, list[dict[str, Any]]] = {}
    for term in contract.get("terms") or []:
        if not isinstance(term, dict) or not term.get("canonical_id"):
            continue
        phrases = {str(term.get("term") or "").strip().lower()}
        for alias in str(term.get("aliases") or "").split(","):
            alias = alias.strip().lower()
            if alias:
                phrases.add(alias)
        for phrase in phrases:
            if phrase:
                phrase_to_terms.setdefault(phrase, []).append(term)

    for phrase, terms in phrase_to_terms.items():
        if len(terms) < 2:
            continue
        distinct_exprs = {str(t.get("canonical_expression") or "").strip() for t in terms}
        if len(distinct_exprs) < 2:
            continue  # same target under two names — not a conflict
        participant_ids = sorted({t["canonical_id"] for t in terms})
        conflicts.append({
            "conflict_key": conflict_key("business_term_phrase_collision", *participant_ids),
            "code": "business_term_phrase_collision",
            "severity": "ERROR",
            "object_type": "business_term",
            "object_id": participant_ids[0],
            "schema_name": "",
            "table_name": "",
            "origin": "business_terms",
            "message": (
                f'"{phrase}" matches {len(terms)} business terms with different meanings: '
                + "; ".join(
                    f'{t.get("term")} -> {t.get("canonical_expression")}' for t in terms
                )
            ),
            "evidence": {
                "phrase": phrase,
                "terms": [
                    {
                        "canonical_id": t["canonical_id"],
                        "term": t.get("term"),
                        "canonical_expression": t.get("canonical_expression"),
                    }
                    for t in terms
                ],
            },
            "suggestions": [
                "Rename or remove the colliding alias from one of the terms.",
                "Merge the terms if they are meant to describe the same thing.",
            ],
        })

    # ── Check 2: field business_candidates collisions ────────────────────
    phrase_to_fields: dict[str, list[dict[str, Any]]] = {}
    for table in (contract.get("model") or {}).get("tables") or []:
        if not isinstance(table, dict):
            continue
        for field in table.get("fields") or []:
            if not isinstance(field, dict) or not field.get("canonical_id"):
                continue
            for phrase in field.get("business_candidates") or []:
                phrase_norm = str(phrase or "").strip().lower()
                if phrase_norm:
                    phrase_to_fields.setdefault(phrase_norm, []).append(field)

    for phrase, fields in phrase_to_fields.items():
        distinct_ids = sorted({f["canonical_id"] for f in fields})
        if len(distinct_ids) < 2:
            continue
        conflicts.append({
            "conflict_key": conflict_key("field_synonym_collision", *distinct_ids),
            "code": "field_synonym_collision",
            "severity": "WARNING",
            "object_type": "field",
            "object_id": distinct_ids[0],
            "schema_name": "",
            "table_name": "",
            "origin": "schema_enrichment",
            "message": (
                f'"{phrase}" is a candidate business name for {len(distinct_ids)} '
                "different columns."
            ),
            "evidence": {"phrase": phrase, "field_ids": distinct_ids},
            "suggestions": [
                "Scope the synonym to one column, or remove it from the others' candidates.",
            ],
        })

    return conflicts


def _normalize_sql(sql: str) -> str:
    """Whitespace-only normalization for SQL comparison. Case is preserved
    deliberately — two templates differing only in identifier casing are
    NOT guaranteed equivalent on a case-sensitive-identifier database, so
    treating them as "the same formula" would be an assumption this module
    has no grounds to make."""
    return re.sub(r"\s+", " ", str(sql or "")).strip()


def detect_duplicate_metric_names(contract: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Two or more active metric_registry rows share a name/synonym phrase but
    carry different SQL. metric_registry has no UNIQUE constraint on
    (account_id, name) — verified against store/db.py, not assumed — so
    this is possible today, and the consequence is worse than an admin
    typo: BOTH of the metric-aware paths that decide what SQL a question
    gets iterate every metric and match by phrase without deduping a
    collision —

      store.match_metric() (the deterministic registry route, checked
      BEFORE any LLM involvement) returns the first phrase match in
      whatever order list_metrics()'s "ORDER BY name" happens to produce —
      the "losing" metric's formula is silently unreachable.

      store.list_metric_formula_context() (SQL-generation prompt context
      for the LLM route) has no such dedup either — the prompt can receive
      two "use this EXACT sql_template" instructions for what looks like
      one concept, the same contradictory-prompt failure mode as a
      colliding business term (see detect_synonym_collisions).

    Phrase collisions whose SQL is identical are skipped — that's a
    harmless duplicate synonym, not a conflict.
    """
    conflicts: list[dict[str, Any]] = []
    phrase_to_metrics: dict[str, list[dict[str, Any]]] = {}
    for metric in contract.get("metrics") or []:
        if not isinstance(metric, dict) or not metric.get("canonical_id"):
            continue
        phrases = {str(metric.get("name") or "").strip().lower()}
        for syn in str(metric.get("synonyms") or "").split(","):
            syn = syn.strip().lower()
            if syn:
                phrases.add(syn)
        for phrase in phrases:
            if phrase:
                phrase_to_metrics.setdefault(phrase, []).append(metric)

    for phrase, metrics in phrase_to_metrics.items():
        distinct_by_id = {m["canonical_id"]: m for m in metrics}
        if len(distinct_by_id) < 2:
            continue
        distinct_sql = {_normalize_sql(m.get("sql_template")) for m in distinct_by_id.values()}
        if len(distinct_sql) < 2:
            continue
        participant_ids = sorted(distinct_by_id.keys())
        conflicts.append({
            "conflict_key": conflict_key("duplicate_metric_name", *participant_ids),
            "code": "duplicate_metric_name",
            "severity": "ERROR",
            "object_type": "metric",
            "object_id": participant_ids[0],
            "schema_name": "",
            "table_name": "",
            "origin": "metric_registry",
            "message": (
                f'"{phrase}" matches {len(distinct_by_id)} metrics with different formulas: '
                + "; ".join(f'{m.get("name")} (id={m.get("id")})' for m in distinct_by_id.values())
            ),
            "evidence": {
                "phrase": phrase,
                "metrics": [
                    {
                        "canonical_id": cid,
                        "name": m.get("name"),
                        "sql_template": m.get("sql_template"),
                    }
                    for cid, m in distinct_by_id.items()
                ],
            },
            "suggestions": [
                "Rename or deprecate one of the colliding metrics.",
                "Remove the shared name/synonym from one of them.",
            ],
        })

    return conflicts


def _known_table_variants(contract: dict[str, Any]) -> set[str]:
    """Every table identifier this SAME compile has direct evidence of —
    the union of the schema-derived model's tables and the entity graph's
    entities, each expanded into fqn/schema.table/bare-name variants (same
    tolerance _table_fqn_variants already applies for base_table matching).
    """
    known: set[str] = set()
    for table in (contract.get("model") or {}).get("tables") or []:
        if isinstance(table, dict):
            fqn = str(table.get("qualified_name") or table.get("fqn") or "")
            known.update(_table_fqn_variants(fqn))
    for entity in (contract.get("graph") or {}).get("entities") or []:
        if isinstance(entity, dict):
            from core.semantic_ids import resolve_entity_table_fqn
            known.update(_table_fqn_variants(resolve_entity_table_fqn(entity)))
    return known


def detect_stale_references(contract: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Semantic objects that point at a table this SAME compile has no
    evidence of — deliberately narrower than a live-schema diff: the
    existing Model Health "Schema Drift" panel (admin/routes.py's
    _compute_schema_drift) already compares the current DB schema against
    the previous discovery snapshot; re-deriving that here from a live
    read would duplicate it. This detector instead cross-references
    semantic objects against what's ALREADY compiled into this contract
    (the schema-derived model and the entity graph), catching a metric or
    term that was configured against a table outside the KB's current
    scope, a typo'd base_table, or a graph relationship whose entity was
    deleted without cleaning up the edge (entity_relationships has no
    foreign key tying from_entity/to_entity back to entity_graph.entity_name
    — verified against store/db.py — so an orphaned edge is a real
    possibility, not a hypothetical).

    Severity is STALE, not ERROR or WARNING: per the compiler's severity
    table this means "revalidate or deprecate", and deliberately does NOT
    contribute to governed_recompile_contract's error_count, so it can
    never block a publish on its own — a stale reference means someone
    should look at it, not that the contract is wrong.
    """
    conflicts: list[dict[str, Any]] = []
    known_tables = _known_table_variants(contract)

    for metric in contract.get("metrics") or []:
        if not isinstance(metric, dict) or not metric.get("canonical_id"):
            continue
        base_table = str(metric.get("base_table") or "").strip()
        if not base_table:
            continue  # nothing to validate — not this detector's concern
        if not (_table_fqn_variants(base_table) & known_tables):
            conflicts.append({
                "conflict_key": conflict_key("stale_metric_base_table", metric["canonical_id"]),
                "code": "stale_metric_base_table",
                "severity": "STALE",
                "object_type": "metric",
                "object_id": metric["canonical_id"],
                "schema_name": "",
                "table_name": base_table,
                "origin": "metric_registry",
                "message": (
                    f'Metric "{metric.get("name") or metric.get("id")}" targets '
                    f'"{base_table}", which is not in the current semantic model or entity graph.'
                ),
                "evidence": {"base_table": base_table},
                "suggestions": [
                    "Update the metric's base table, or deprecate it if the table was removed.",
                ],
            })

    for term in contract.get("terms") or []:
        if not isinstance(term, dict) or not term.get("canonical_id"):
            continue
        involved = [t.strip() for t in str(term.get("tables_involved") or "").split(",") if t.strip()]
        if not involved:
            continue  # unscoped term — nothing to validate
        if not any(_table_fqn_variants(t) & known_tables for t in involved):
            conflicts.append({
                "conflict_key": conflict_key("stale_term_table_reference", term["canonical_id"]),
                "code": "stale_term_table_reference",
                "severity": "STALE",
                "object_type": "business_term",
                "object_id": term["canonical_id"],
                "schema_name": "",
                "table_name": ", ".join(involved),
                "origin": "business_terms",
                "message": (
                    f'Business term "{term.get("term")}" references '
                    f"{', '.join(involved)}, none of which are in the current "
                    "semantic model or entity graph."
                ),
                "evidence": {"tables_involved": involved},
                "suggestions": ["Update tables_involved, or deactivate the term."],
            })

    entity_names = {
        str(e.get("entity_name") or "").upper()
        for e in (contract.get("graph") or {}).get("entities") or []
        if isinstance(e, dict) and e.get("entity_name")
    }
    for rel in (contract.get("graph") or {}).get("relationships") or []:
        if not isinstance(rel, dict) or not rel.get("canonical_id"):
            continue
        missing = [
            side for side in (str(rel.get("from_entity") or ""), str(rel.get("to_entity") or ""))
            if side and side.upper() not in entity_names
        ]
        if missing:
            conflicts.append({
                "conflict_key": conflict_key("stale_relationship_entity", rel["canonical_id"]),
                "code": "stale_relationship_entity",
                "severity": "STALE",
                "object_type": "relationship",
                "object_id": rel["canonical_id"],
                "schema_name": "",
                "table_name": "",
                "origin": "entity_graph",
                "message": (
                    f'Join {rel.get("from_entity")} -> {rel.get("to_entity")} references '
                    f"{' and '.join(missing)}, which no longer exist as active graph entities."
                ),
                "evidence": {"from_entity": rel.get("from_entity"), "to_entity": rel.get("to_entity"),
                             "missing": missing},
                "suggestions": ["Delete the orphaned join, or re-create the missing entity."],
            })

    return conflicts


def detect_compliance_gaps(contract: dict[str, Any]) -> list[dict[str, Any]]:
    """
    A metric references a RESTRICTED column (direct PII/PHI/PCI
    identifiability — the highest data_asset_classification.sensitivity
    tier) whose classification has not been human-reviewed yet.

    contract["classifications"] (added as a proper compiled source
    alongside metrics/terms/graph — see _compile_contract_internal) is
    keyed "TABLE_FQN.COLUMN", the exact shape core.semantic_ids.field_id()
    produces minus its "field:" prefix, so a metric's base_table + each
    required_columns token (free-text, tokenized the same way
    core/metric_scope.py's _split_required_columns already does) resolves
    straight to a classification lookup.

    Severity is deliberately review-status-dependent, not a flat block:

      WARNING — unreviewed. The column's mask_strategy is an auto-suggestion,
      not confirmed, so this metric's actual masking behavior at query time
      is not yet trustworthy — an admin should review the classification
      before relying on the metric's output.

      INFO — already reviewed. Nothing to fix; this exists purely as an
      audit trail entry ("this published metric knowingly touches a
      RESTRICTED column, and here is its confirmed mask_strategy") per the
      compiler's own severity table definition of INFO.

    Deliberately narrower than a full policy-engine replica: aggregate-only
    exemptions, purposes, and role-based rules all live in
    core/compliance/policy_engine.py and require tenant policy configuration
    this compiler doesn't have. This detector only answers "does an approved
    metric touch a RESTRICTED column, and has that column's masking been
    confirmed" — visibility, not enforcement.
    """
    from core.metric_scope import _split_required_columns

    conflicts: list[dict[str, Any]] = []
    classifications = contract.get("classifications") or {}
    if not isinstance(classifications, dict) or not classifications:
        return conflicts

    # classification keys are always fully-qualified ("ERP.FACT_SALES.COL"),
    # but base_table is free text and often bare ("FACT_SALES" — see
    # core/metric_scope.py's own multi-strategy resolution). Index by bare
    # column name first, then filter candidates by table-variant overlap,
    # rather than trying to construct the exact qualified key directly.
    by_column: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for key, classification in classifications.items():
        table_part, _, col_part = str(key).rpartition(".")
        if col_part:
            by_column.setdefault(col_part, []).append((table_part, classification))

    for metric in contract.get("metrics") or []:
        if not isinstance(metric, dict) or not metric.get("canonical_id"):
            continue
        base_table = str(metric.get("base_table") or "").strip()
        required = _split_required_columns(str(metric.get("required_columns") or ""))
        if not base_table or not required:
            continue
        base_variants = _table_fqn_variants(base_table)
        for column in required:
            classification = None
            table_variant = base_table.upper()
            for table_part, candidate in by_column.get(column, []):
                if _table_fqn_variants(table_part) & base_variants:
                    classification = candidate
                    table_variant = table_part
                    break
            if not classification or classification.get("sensitivity") != "RESTRICTED":
                continue
            reviewed = bool(classification.get("reviewed"))
            conflicts.append({
                "conflict_key": conflict_key(
                    "unreviewed_restricted_column" if not reviewed else "restricted_column_in_metric",
                    metric["canonical_id"], f"{table_variant}.{column}",
                ),
                "code": "unreviewed_restricted_column" if not reviewed else "restricted_column_in_metric",
                "severity": "WARNING" if not reviewed else "INFO",
                "object_type": "metric",
                "object_id": metric["canonical_id"],
                "schema_name": "",
                "table_name": base_table,
                "origin": "compliance",
                "message": (
                    f'Metric "{metric.get("name") or metric.get("id")}" requires '
                    f'"{column}", classified RESTRICTED and '
                    + ("not yet reviewed." if not reviewed else
                       f'reviewed (mask_strategy={classification.get("mask_strategy")}).')
                ),
                "evidence": {
                    "column": column, "sensitivity": classification.get("sensitivity"),
                    "tags": classification.get("tags"), "reviewed": reviewed,
                    "mask_strategy": classification.get("mask_strategy"),
                },
                "suggestions": (
                    ["Review this column's classification on the Compliance page."]
                    if not reviewed else []
                ),
            })

    return conflicts


# Registry of detectors run.py wires into the compiler. Each entry is a pure
# function (contract) -> list[conflict dict]. New Sprint 2 detectors are
# added here, not by editing core/semantic_contract.py.
DETECTORS: tuple[Any, ...] = (
    detect_ambiguous_date_roles,
    detect_synonym_collisions,
    detect_duplicate_metric_names,
    detect_stale_references,
    detect_compliance_gaps,
)


def run_all_detectors(contract: dict[str, Any]) -> list[dict[str, Any]]:
    """Run every registered detector, isolating failures per-detector so one
    broken detector degrades (skipped, nothing reported for it) rather than
    failing the whole compile — the same fail-soft contract
    _compile_contract_internal already applies to its own data sources."""
    conflicts: list[dict[str, Any]] = []
    for detector in DETECTORS:
        try:
            conflicts.extend(detector(contract) or [])
        except Exception as exc:
            log.warning("Conflict detector %s failed, skipping: %s", detector.__name__, exc)
    return conflicts
