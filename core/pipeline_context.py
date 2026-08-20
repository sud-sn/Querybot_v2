"""
core/pipeline_context.py
────────────────────────
Per-request client context helpers extracted from main.py.

Covers:
  • client_dir          — per-account file-system directory
  • get_state / save_state  — client state bag
  • get_client_db       — active DB config for an account
  • _merge_semantic_plans   — combine multiple semantic field plans
  • check_query_limit   — monthly query-count gate
  • check_token_limit   — monthly token-usage gate
  • get_portal_base     — base URL for portal links
"""

from __future__ import annotations

import json
import os
import re
from collections import deque
from pathlib import Path

import store
from core.semantic_plan_utils import required_semantic_tables


# ── File-system ────────────────────────────────────────────────────────────────

def client_dir(account_id: str) -> Path:
    p = Path("clients") / account_id
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── State ─────────────────────────────────────────────────────────────────────

def get_state(account_id: str) -> dict:
    client = store.get_client(account_id)
    if not client:
        return {"state": "NEW"}
    state_data = json.loads(client.get("state_data") or "{}")
    return {"state": client["state"], **state_data}


def save_state(account_id, state, state_data=None, business_desc=None):
    store.update_client_state(account_id, state, state_data or {}, business_desc)


# ── DB config ─────────────────────────────────────────────────────────────────

def get_client_db(account_id: str) -> dict | None:
    client = store.get_client(account_id)
    if not client:
        return None
    db_config_id = client.get("db_config_id")
    if not db_config_id:
        return None
    return store.get_db_config(db_config_id)


# ── Semantic plan merge ────────────────────────────────────────────────────────

def _semantic_table_identity(table: str) -> str:
    cleaned = str(table or "").upper()
    for char in "[]\"`":
        cleaned = cleaned.replace(char, "")
    parts = [part for part in cleaned.split(".") if part]
    return ".".join(parts[-2:]) if len(parts) >= 2 else cleaned


def _select_relevant_semantic_joins(joins: list[dict], relevant_tables: set[str]) -> list[dict]:
    candidates = [
        join for join in joins
        if join.get("enforcement") != "advisory"
    ]
    targets = {table for table in relevant_tables if table}
    if len(targets) < 2 or not candidates:
        return []

    adjacency: dict[str, list[tuple[str, int]]] = {}
    for index, join in enumerate(candidates):
        left = _semantic_table_identity(join.get("from") or "")
        right = _semantic_table_identity(join.get("to") or "")
        if not left or not right:
            continue
        adjacency.setdefault(left, []).append((right, index))
        adjacency.setdefault(right, []).append((left, index))

    selected_indexes: set[int] = set()
    ordered_targets = sorted(targets)
    for start_index, start in enumerate(ordered_targets):
        for target in ordered_targets[start_index + 1:]:
            queue = deque([(start, [])])
            visited = {start}
            while queue:
                node, path = queue.popleft()
                if node == target:
                    selected_indexes.update(path)
                    break
                for neighbour, edge_index in adjacency.get(node, []):
                    if neighbour in visited:
                        continue
                    visited.add(neighbour)
                    queue.append((neighbour, path + [edge_index]))

    return [
        join for index, join in enumerate(candidates)
        if index in selected_indexes
    ]


def _merge_semantic_plans(*plans: dict | None) -> dict:
    fields: list[dict] = []
    joins: list[dict] = []
    advisory_fields: list[dict] = []
    available_dimensions: list[dict] = []
    seen_fields: set[tuple[str, str]] = set()
    seen_joins: set[tuple[str, str, tuple[tuple[str, str], ...]]] = set()
    reasons: list[str] = []
    date_key_policies: list[dict] = []
    seen_date_policies: set[tuple[str, str, str]] = set()
    temporal_policies: list[dict] = []
    seen_temporal_policies: set[tuple[str, str, str, str]] = set()
    date_disclosures: list[dict] = []
    seen_date_disclosures: set[tuple[str, str, str]] = set()
    source_scope: dict = {}
    fact_anchor = ""

    # Pre-pass: union avoid lists across plans so the main loop can drop a
    # superseded column no matter which plan proposed it.  The LLM field
    # planner routinely re-suggests the old generated column ("purchase order
    # amount" -> PCH_ORD_LIN_AMT) that an admin-approved mapping replaced; the
    # dedup key below is (table, column), not business term, so without this
    # both rivals would survive as required fields.
    avoid_columns: list[dict] = []
    # Measures that share the bare business word the question used. Like the
    # avoid list, this outlives a plan that resolved no fields — it exists
    # precisely because nothing resolved.
    ambiguous_measures: list[str] = []
    # True when a source plan reports that deterministic planning RAISED, as
    # opposed to finding nothing. The merged plan is empty either way; only
    # this tells the difference.
    planning_failed = any(
        (plan or {}).get("planning_failed") for plan in plans
    )
    seen_avoid: set[tuple[str, str]] = set()
    for plan in plans:
        if not plan:
            continue
        candidate_scope = plan.get("source_scope") or {}
        if candidate_scope:
            if not source_scope:
                source_scope = dict(candidate_scope)
            else:
                # Merge missing evidence without allowing a later, weaker
                # plan to erase the already selected metric fact.
                for key, value in candidate_scope.items():
                    if value not in (None, "", [], {}) and not source_scope.get(key):
                        source_scope[key] = value
        if not fact_anchor and plan.get("fact_anchor"):
            fact_anchor = str(plan.get("fact_anchor") or "")
        # Collected BEFORE the enabled check: a plan only reports ambiguous
        # measures when it resolved nothing, which is exactly when it is
        # disabled. Gathering it below the check would mean it could never
        # survive the merge at all.
        for rival in plan.get("ambiguous_measures") or []:
            if rival not in ambiguous_measures:
                ambiguous_measures.append(rival)
        if not plan.get("enabled"):
            continue
        for avoid in plan.get("avoid_columns") or []:
            key = (
                _semantic_table_identity(avoid.get("table") or ""),
                (avoid.get("column") or "").upper(),
            )
            if not key[0] or not key[1] or key in seen_avoid:
                continue
            seen_avoid.add(key)
            avoid_columns.append(avoid)

    for plan in plans:
        if not plan or not plan.get("enabled"):
            continue
        if plan.get("reason"):
            reasons.append(str(plan.get("reason")))
        relevant_tables: set[str] = set()
        for field in plan.get("fields") or []:
            if field.get("enforcement") == "advisory":
                advisory_fields.append(field)
                continue
            key = (
                _semantic_table_identity(field.get("table") or ""),
                (field.get("column") or "").upper(),
            )
            if not key[0] or not key[1] or key in seen_fields:
                continue
            if key in seen_avoid and field.get("source") != "approved_semantic_field":
                continue
            seen_fields.add(key)
            fields.append(field)
            relevant_tables.add(key[0])
            source_table = _semantic_table_identity(
                field.get("source_table") or field.get("source_key_table") or ""
            )
            if source_table:
                relevant_tables.add(source_table)
        available_dimensions.extend(plan.get("available_dimensions") or [])
        for policy in plan.get("date_key_policies") or []:
            policy_key = (
                _semantic_table_identity(policy.get("table") or ""),
                str(policy.get("column") or "").upper(),
                str(policy.get("role_alias") or "").lower(),
            )
            if policy_key[0] and policy_key[1] and policy_key not in seen_date_policies:
                seen_date_policies.add(policy_key)
                date_key_policies.append(policy)
        for policy in plan.get("temporal_policies") or []:
            policy_key = (
                _semantic_table_identity(policy.get("fact_table") or ""),
                str(policy.get("fact_column") or "").upper(),
                str(policy.get("role_alias") or "").lower(),
                str(policy.get("kind") or "").lower(),
            )
            if policy_key[0] and policy_key[1] and policy_key not in seen_temporal_policies:
                seen_temporal_policies.add(policy_key)
                temporal_policies.append(policy)
        for disclosure in plan.get("date_disclosures") or []:
            disclosure_key = (
                _semantic_table_identity(disclosure.get("table") or ""),
                str(disclosure.get("column") or "").upper(),
                str(disclosure.get("label") or "").casefold(),
            )
            if (
                disclosure_key[0]
                and disclosure_key[1]
                and disclosure_key not in seen_date_disclosures
            ):
                seen_date_disclosures.add(disclosure_key)
                date_disclosures.append(disclosure)
        relevant_joins = _select_relevant_semantic_joins(
            plan.get("joins") or [],
            relevant_tables,
        )
        # A role-playing dimension may be joined twice to the same fact using
        # different FKs (for example booked_date and order_date). A shortest
        # path selector would retain only one parallel edge, so exact governed
        # date joins explicitly opt into preservation.
        for join in plan.get("joins") or []:
            if join.get("preserve_all") and join not in relevant_joins:
                relevant_joins.append(join)
        for join in relevant_joins:
            from_table = _semantic_table_identity(join.get("from") or "")
            to_table = _semantic_table_identity(join.get("to") or "")
            conds = tuple(
                (str(left).upper(), str(right).upper())
                for left, right in (join.get("conditions") or [])
            )
            key = (from_table, to_table, conds)
            if not key[0] or not key[1] or key in seen_joins:
                continue
            seen_joins.add(key)
            joins.append(join)
    if not fields:
        return {
            "enabled": False,
            "fields": [],
            "joins": [],
            "required_tables": [],
            "reason": "no semantic fields",
            "source_scope": source_scope,
            "fact_anchor": fact_anchor,
            "avoid_columns": avoid_columns,
            "ambiguous_measures": ambiguous_measures,
        "planning_failed": planning_failed,
        }
    return {
        "enabled": True,
        "fields": fields,
        "joins": joins,
        "required_tables": sorted(required_semantic_tables({
            "fields": fields,
            "joins": joins,
        })),
        "reason": " + ".join(dict.fromkeys(reasons)) or "merged semantic plan",
        "advisory_fields": advisory_fields,
        "available_dimensions": available_dimensions,
        "avoid_columns": avoid_columns,
        "ambiguous_measures": ambiguous_measures,
        "planning_failed": planning_failed,
        "date_key_policies": date_key_policies,
        "temporal_policies": temporal_policies,
        "date_disclosures": date_disclosures,
        "source_scope": source_scope,
        "fact_anchor": fact_anchor,
    }


def _scope_semantic_plan_to_analytical_request(
    plan: dict | None,
    analytical_plan: dict | None,
) -> dict:
    """Remove lexical field bindings that contradict a compiled count intent.

    In phrases such as "warehouses with the most customer orders", customer
    modifies the order event; it is not a requested output dimension. Likewise
    an order-quantity column is not the governed identifier to COUNT DISTINCT.
    Keep those matches as optional hints so they can inform generation without
    becoming validator requirements.
    """
    scoped = plan if isinstance(plan, dict) else {}
    intent = analytical_plan if isinstance(analytical_plan, dict) else {}
    if str(intent.get("measure_semantics") or "") != "count_distinct_business_identifier":
        return scoped

    requested_dimensions = {
        re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip().rstrip("s")
        for value in [intent.get("entity_grain"), *(intent.get("dimensions") or [])]
        if str(value or "").strip()
    }
    target = scoped.get("count_target") or {}
    selected = target.get("selected") if isinstance(target.get("selected"), dict) else {}
    target_table = _semantic_table_identity(selected.get("table") or "")
    target_column = str(selected.get("column") or "").upper()

    for field in scoped.get("fields") or []:
        role = str(field.get("role") or "").casefold()
        term = re.sub(
            r"[^a-z0-9]+", " ", str(field.get("term") or "").casefold()
        ).strip().rstrip("s")
        is_exact_target = (
            target_table
            and target_column
            and _semantic_table_identity(field.get("table") or "") == target_table
            and str(field.get("column") or "").upper() == target_column
        )
        if is_exact_target:
            continue
        if role in {"measure", "metric", "calculated_measure"}:
            field["enforcement"] = "optional"
            field["demotion_reason"] = "derived event count uses governed identifier"
        elif role in {"dimension", "display_dimension", "attribute"} and term:
            if term not in requested_dimensions:
                field["enforcement"] = "optional"
                field["demotion_reason"] = "event modifier is not requested output grain"

    required_table_names = {
        str(field.get("table") or "")
        for field in scoped.get("fields") or []
        if field.get("enforcement") != "optional" and field.get("table")
    }
    if selected.get("table"):
        required_table_names.add(str(selected.get("table")))
    required_identities = {
        _semantic_table_identity(table) for table in required_table_names if table
    }
    scoped["required_tables"] = sorted(required_table_names)
    for join in scoped.get("joins") or []:
        from_table = _semantic_table_identity(join.get("from") or "")
        to_table = _semantic_table_identity(join.get("to") or "")
        if from_table not in required_identities or to_table not in required_identities:
            join["enforcement"] = "optional"
    return scoped


# ── Usage limits ──────────────────────────────────────────────────────────────

def check_query_limit(account_id: str) -> tuple[bool, int, int]:
    client = store.get_client(account_id)
    limit  = (client or {}).get("query_limit_monthly") or 500
    used   = store.get_monthly_query_count(account_id)
    return used < limit, used, limit


def check_token_limit(account_id: str) -> tuple[bool, int, int]:
    client = store.get_client(account_id) or {}
    limit = int(client.get("token_limit_monthly") or 0)
    usage = store.get_monthly_token_status(account_id)
    used = int(usage.get("total_tokens") or 0)
    if limit <= 0:
        return True, used, 0
    return used < limit, used, limit


# ── Portal base URL ────────────────────────────────────────────────────────────

def get_portal_base() -> str:
    return os.getenv("PORTAL_BASE_URL", "http://localhost:8000").rstrip("/")
