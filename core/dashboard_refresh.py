"""Governed dashboard source execution and scheduled encrypted caching."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import store
from core.dashboard_filters import compile_dashboard_filters

log = logging.getLogger("querybot.dashboard_refresh")


@dataclass(frozen=True)
class DashboardSourceResult:
    rows: list[dict]
    sql: str
    from_cache: bool = False
    refreshed_at: str = ""
    # Cached rows kept through refreshes that have failed since this time.
    refresh_failed_at: str = ""
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
    # semantic_compiler_state keeps the published version in active_version.
    # This read two names the table has never had, so every dashboard fell
    # back to the version its source was saved under and a cached result was
    # never invalidated by a change in meaning.
    contract_version = str(
        compiler.get("active_version")
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
                refresh_failed_at=str(cached.get("failed_at") or ""),
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
    # pyodbc returns Decimal/date values for common aggregates. Dashboard
    # sources cross a JSON boundary both when cached and when rendered, so
    # normalize once here instead of letting driver-specific values leak.
    from core.result_renderer import _sanitize_rows
    safe_rows = _sanitize_rows(list(governed.rows or []))
    if is_owner and not active_values:
        ttl = _kept_for(account_id, viewer, int(getattr(governed.decision, "cache_ttl_seconds", 0) or 600))
        if ttl:
            store.save_source_cache(
                source,
                safe_rows,
                policy_version=policy_version,
                contract_version=contract_version,
                ttl_seconds=ttl,
            )
    return DashboardSourceResult(
        rows=safe_rows,
        sql=str(governed.sql or compiled.sql),
        applied_filters=compiled.applied,
        ignored_filters=compiled.ignored,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _kept_for(account_id: str, owner: dict, ttl: int) -> int:
    """How long an owner's rows are kept: the source's own cache life, but no longer than access of theirs that
    ends by itself (an attestation's term, an emergency grant) -- nothing is written when it ends, so nothing
    else would stop the rows. 0: not kept, the access ends within the minute a cache lasts at least."""
    ends = store.access_ends_at(account_id, str(owner.get("id") or ""))
    if not ends:
        return ttl
    try:
        # A bare date ends at the start of that day, as validity compares it.
        left = int((datetime.fromisoformat(ends) - _utc_now()).total_seconds())
    except ValueError:
        return 0
    return min(ttl, left) if left >= 60 else 0


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
        failure = store.mark_source_cache_error(source, str(exc))
        log.warning("Dashboard source %s refresh failed (%s in a row; next attempt %s UTC): %s",
                    source.get("id"), failure["failure_count"], failure["next_attempt_at"], exc)
        if failure["owner_due"]:
            _tell_the_owner(source, user, failure)
        return False


def _tell_the_owner(source: dict, owner: dict, failure: dict) -> None:
    """Tell a dashboard's owner, in their language, that its scheduled refresh
    keeps failing: since when, what the dashboard shows meanwhile, and when it
    is tried again. Its failures were logged and nothing else.

    Once per dashboard for a run of failures, however many of its sources
    fail. A notice no channel delivered is not counted as told: the next
    failure tries again.
    """
    import asyncio

    from core.i18n import t
    from core.notify import send_proactive_notification

    claim = store.claim_owner_notice(source)
    if not claim:
        return
    refreshed = str(source.get("cache_refreshed_at") or "")[:16]
    # The rows of the last refresh that worked are shown until they expire, and not after.
    until = str(source.get("cache_expires_at") or "")
    shown = bool(refreshed) and until > _utc_now().strftime("%Y-%m-%d %H:%M:%S")
    message = t(
        "notify.dashboard_refresh_failed" if shown
        else "notify.dashboard_refresh_failed_expired" if refreshed
        else "notify.dashboard_refresh_failed_no_data",
        lang=owner.get("lang") or "en",
        name=source.get("dashboard_name") or source.get("name") or "",
        count=failure["failure_count"],
        since=str(failure["failed_at"])[:16],
        at=refreshed,
        until=until[:16],
        next=str(failure["next_attempt_at"])[:16],
    )
    try:
        delivered = asyncio.run(send_proactive_notification(str(source["account_id"]), int(owner["id"]), message))
    except Exception as exc:
        log.warning("Dashboard source %s: its owner could not be told the refresh is failing: %s",
                    source.get("id"), exc)
        delivered = False
    if not delivered:
        store.release_owner_notice(source, claim)
        log.warning("Dashboard source %s: no channel reached its owner (user %s); the next failure tells them",
                    source.get("id"), owner.get("id"))


def run_due_dashboard_refreshes() -> dict:
    due = store.list_due_dashboard_sources()
    refreshed = 0
    for source in due:
        # One source that cannot even record its failure -- deleted while it
        # was refreshed -- ended the tick, and every source after it waited.
        try:
            refreshed += refresh_dashboard_source(source)
        except Exception:
            log.error("Dashboard source %s: its scheduled refresh could not be recorded", source.get("id"),
                      exc_info=True)
    return {"due": len(due), "refreshed": refreshed, "failed": len(due) - refreshed}
