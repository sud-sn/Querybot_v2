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

def _merge_semantic_plans(*plans: dict | None) -> dict:
    fields: list[dict] = []
    joins: list[dict] = []
    seen_fields: set[tuple[str, str]] = set()
    seen_joins: set[tuple[str, str, tuple[tuple[str, str], ...]]] = set()
    reasons: list[str] = []
    for plan in plans:
        if not plan or not plan.get("enabled"):
            continue
        if plan.get("reason"):
            reasons.append(str(plan.get("reason")))
        for field in plan.get("fields") or []:
            key = ((field.get("table") or "").upper(), (field.get("column") or "").upper())
            if not key[0] or not key[1] or key in seen_fields:
                continue
            seen_fields.add(key)
            fields.append(field)
        for join in plan.get("joins") or []:
            conds = tuple(
                (str(left).upper(), str(right).upper())
                for left, right in (join.get("conditions") or [])
            )
            key = ((join.get("from") or "").upper(), (join.get("to") or "").upper(), conds)
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
        "required_tables": sorted({f.get("table") for f in fields if f.get("table")}),
        "reason": " + ".join(dict.fromkeys(reasons)) or "merged semantic plan",
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
