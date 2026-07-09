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
from collections import deque
from pathlib import Path

import store


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

    # Pre-pass: union avoid lists across plans so the main loop can drop a
    # superseded column no matter which plan proposed it.  The LLM field
    # planner routinely re-suggests the old generated column ("purchase order
    # amount" -> PCH_ORD_LIN_AMT) that an admin-approved mapping replaced; the
    # dedup key below is (table, column), not business term, so without this
    # both rivals would survive as required fields.
    avoid_columns: list[dict] = []
    seen_avoid: set[tuple[str, str]] = set()
    for plan in plans:
        if not plan or not plan.get("enabled"):
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
        relevant_joins = _select_relevant_semantic_joins(
            plan.get("joins") or [],
            relevant_tables,
        )
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
        return {"enabled": False, "fields": [], "joins": [], "required_tables": [], "reason": "no semantic fields"}
    return {
        "enabled": True,
        "fields": fields,
        "joins": joins,
        "required_tables": sorted(
            {f.get("table") for f in fields if f.get("table")}
            | {j.get("from") for j in joins if j.get("from")}
            | {j.get("to") for j in joins if j.get("to")}
        ),
        "reason": " + ".join(dict.fromkeys(reasons)) or "merged semantic plan",
        "advisory_fields": advisory_fields,
        "available_dimensions": available_dimensions,
        "avoid_columns": avoid_columns,
    }


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
