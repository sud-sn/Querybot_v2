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


# Registry of detectors run.py wires into the compiler. Each entry is a pure
# function (contract) -> list[conflict dict]. New Sprint 2 detectors are
# added here, not by editing core/semantic_contract.py.
DETECTORS: tuple[Any, ...] = (
    detect_ambiguous_date_roles,
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
