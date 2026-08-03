"""Governed dashboard source execution and scheduled encrypted caching."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import store
from core.dashboard_filters import compile_dashboard_filters

log = logging.getLogger("querybot.dashboard_refresh")


@dataclass(frozen=True)
class DashboardSourceResult:
    rows: list[dict]
    sql: str
    from_cache: bool = False
    refreshed_at: str = ""
    applied_filters: tuple[str, ...] = ()
    ignored_filters: tuple[str, ...] = ()


def execute_dashboard_source(
    source: dict,
    viewer: dict,
    *,
    filters: list[dict] | None = None,
    filter_values: dict[str, str] | None = None,
    allow_cache: bool = True,
) -> DashboardSourceResult:
    """Execute one saved source using the viewer's current ACL and policies."""
    account_id = str(source.get("account_id") or viewer.get("account_id") or "")
    active_values = {
        str(key): str(value)
        for key, value in (filter_values or {}).items()
        if str(value or "").strip()
    }
    is_owner = int(source.get("user_id") or 0) == int(viewer.get("id") or 0)
    profile = store.get_compliance_profile(account_id)
    compiler = store.get_semantic_compiler_state(account_id)
    contract_version = str(
        compiler.get("active_contract_version")
        or compiler.get("published_version")
        or source.get("semantic_contract_version")
        or ""
    )
    policy_version = int(profile.get("active_policy_version") or 0)

    if allow_cache and is_owner and not active_values:
        cached = store.get_source_cache(int(source["id"]), int(viewer["id"]), account_id)
        if (
            cached
            and int(cached.get("policy_version") or 0) == policy_version
            and str(cached.get("contract_version") or "") == contract_version
        ):
            return DashboardSourceResult(
                rows=list(cached.get("rows") or []),
                sql=str(source.get("sql_query") or ""),
                from_cache=True,
                refreshed_at=str(cached.get("refreshed_at") or ""),
            )

    db_cfg = store.get_db_config(int(source.get("db_config_id") or 0))
    if not db_cfg:
        raise ValueError("Database not configured")
    from core.compliance.governed_query import execute_governed_query
    from core.compliance.policy_engine import resolve_context
    from core.schema import load_known_tables, load_schema_columns

    state = store.get_client_state(account_id)
    known_tables = load_known_tables(state.get("schema_dir", ""))
    table_columns = load_schema_columns(state.get("schema_dir", ""))
    compiled = compile_dashboard_filters(
        str(source.get("sql_query") or ""),
        str(db_cfg.get("db_type") or "azure_sql"),
        list(filters or []),
        active_values,
    )
    context = resolve_context(
        account_id, viewer, action="query_execution", channel="portal"
    )
    governed = execute_governed_query(
        db_cfg["credentials"],
        db_cfg["db_type"],
        compiled.sql,
        context=context,
        known_tables=known_tables,
        table_columns=table_columns,
        allowed_tables=store.get_allowed_tables(viewer),
    )
    if is_owner and not active_values:
        ttl = int(getattr(governed.decision, "cache_ttl_seconds", 0) or 600)
        store.save_source_cache(
            source,
            governed.rows,
            policy_version=policy_version,
            contract_version=contract_version,
            ttl_seconds=ttl,
        )
    return DashboardSourceResult(
        rows=list(governed.rows or []),
        sql=str(governed.sql or compiled.sql),
        applied_filters=compiled.applied,
        ignored_filters=compiled.ignored,
    )


def refresh_dashboard_source(source: dict) -> bool:
    """Refresh one scheduled source as its owner and record any failure."""
    user = store.get_user(int(source.get("user_id") or 0))
    if not user or not user.get("is_active"):
        store.mark_source_cache_error(source, "Dashboard owner is inactive.")
        return False
    try:
        execute_dashboard_source(source, user, allow_cache=False)
        return True
    except Exception as exc:
        store.mark_source_cache_error(source, str(exc))
        log.warning("Dashboard source %s refresh failed: %s", source.get("id"), exc)
        return False


def run_due_dashboard_refreshes() -> dict:
    due = store.list_due_dashboard_sources()
    refreshed = sum(1 for source in due if refresh_dashboard_source(source))
    return {"due": len(due), "refreshed": refreshed, "failed": len(due) - refreshed}
