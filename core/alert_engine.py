"""
core/alert_engine.py

Alert definition storage and check engine for the "Alert me on changes"
chip (Sprint E).

Design principles
─────────────────
• Alerts persist to ``data/alerts.json`` — a plain JSON list.
• Only non-credential DB metadata (db_type) is stored; no passwords or
  connection strings are written to disk.
• check_alert_now() is the single point that touches the live database;
  all other functions are pure JSON CRUD.
• run_due_alert_checks() is called on a schedule by
  core/notification_scheduler.py::scheduled_notification_loop (wired into
  main.py startup/shutdown) — it re-checks every active alert whose
  check_interval_minutes has elapsed and delivers via core/notify.py on
  trigger. This closes the gap this module used to leave to "an external
  scheduler."

Supported conditions
────────────────────
  "change_pct"  (default) — trigger when |current − baseline| / baseline ≥ threshold %
  "above"                 — trigger when current > threshold (absolute value)
  "below"                 — trigger when current < threshold (absolute value)

Public API
──────────
  create_alert(question, sql, metric_col, baseline_value, …) → dict
  list_alerts() → list[dict]
  get_alert(alert_id) → dict | None
  delete_alert(alert_id) → bool
  check_alert_now(alert_id, db_cfg) → dict
  run_due_alert_checks() → None
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("querybot.alert_engine")

_ALERTS_PATH = Path(__file__).parent.parent / "data" / "alerts.json"
_VALID_CONDITIONS = frozenset({"above", "below", "change_pct"})


# ══════════════════════════════════════════════════════════════════════════════
# Internal storage helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load() -> list[dict]:
    """Load alerts from disk; return [] on any read/parse error."""
    try:
        if _ALERTS_PATH.exists():
            return json.loads(_ALERTS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("alert_engine: failed to load alerts: %s", exc)
    return []


def _save(alerts: list[dict]) -> None:
    """Persist alert list to disk atomically."""
    _ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ALERTS_PATH.write_text(
        json.dumps(alerts, indent=2, default=str),
        encoding="utf-8",
    )


def _relative_window_of(semantic_plan: dict | None) -> tuple[dict, str]:
    """The governed temporal policy behind this SQL, and the date it resolved to.

    Returns ({}, "") for a question with no relative window — an alert on
    "revenue in March 2026" is pinned on purpose and must stay pinned.
    """
    plan = semantic_plan or {}
    policies = [
        policy for policy in (plan.get("temporal_policies") or [])
        if str(policy.get("anchor_policy") or "") == "latest_available"
    ]
    if not policies:
        return {}, ""
    resolved = plan.get("resolved_date_anchor") or {}
    return dict(policies[0]), str(resolved.get("value") or "")


def _question_is_relative(question: str) -> bool:
    try:
        from core.contextual_dates import detect_temporal_window
        return bool(detect_temporal_window(question or ""))
    except Exception:
        return False


def _refresh_relative_window(alert: dict, db_cfg: dict) -> tuple[str, str]:
    """Re-anchor the alert's SQL to the newest business date before it runs.

    Returns (sql_to_run, failure_reason). A failure reason means the alert must
    NOT be evaluated: comparing today's threshold against a window frozen on the
    day the alert was created is not a check, it is a fixed answer wearing one.
    """
    policy = alert.get("anchor_policy") or {}
    stored_value = str(alert.get("anchor_value") or "")
    sql = str(alert.get("sql") or "")

    if not policy or not stored_value:
        # Either the question named no relative window (nothing to move), or
        # this alert predates the policy being stored and we cannot know which
        # literal in its SQL was the anchor.
        if _question_is_relative(alert.get("question", "")):
            return sql, "relative_window_not_re_anchorable"
        return sql, ""

    if stored_value not in sql:
        # The anchor literal is not in the SQL, so re-anchoring cannot be done
        # by substitution and we would be guessing about what the query means.
        return sql, "anchor_literal_absent"

    from core.date_anchor import resolve_business_anchor
    from core.schema import run_query

    def _probe(probe_sql: str):
        return run_query(
            db_cfg.get("credentials") or db_cfg,
            db_cfg.get("db_type", alert.get("db_type", "azure_sql")),
            probe_sql,
        )

    try:
        resolved = resolve_business_anchor(
            str(alert.get("account_id") or ""),
            policy,
            db_cfg.get("db_type", alert.get("db_type", "azure_sql")),
            _probe,
        )
    except Exception as exc:
        log.warning(
            "alert_engine: could not re-anchor alert %s (%s) — not evaluating it "
            "against a window frozen on %s", alert.get("id"), exc, stored_value,
        )
        return sql, "anchor_probe_failed"

    current = str((resolved or {}).get("value") or "")
    if not current:
        return sql, "anchor_unavailable"
    if current == stored_value:
        return sql, ""

    log.info(
        "alert_engine: alert %s re-anchored from %s to %s before checking",
        alert.get("id"), stored_value, current,
    )
    return sql.replace(stored_value, current), ""


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def create_alert(
    question: str,
    sql: str,
    metric_col: str,
    baseline_value: float,
    *,
    condition: str = "change_pct",
    threshold: float = 10.0,
    check_interval_minutes: int = 60,
    db_cfg: dict | None = None,
    account_id: str = "",
    user_id: str = "",
    purpose_id: str = "",
    semantic_plan: dict | None = None,
) -> dict:
    """
    Create and persist a new alert definition.

    Parameters
    ──────────
    question       : original user question (stored for display)
    sql            : SQL to re-run on each check
    metric_col     : column name whose value is monitored
    baseline_value : current observed value — the reference point
    condition      : one of "change_pct" | "above" | "below"
    threshold      : interpretation depends on condition:
                       • change_pct → minimum % change to trigger (default 10)
                       • above/below → absolute cutoff value
    check_interval_minutes : how often run_due_alert_checks() re-checks this
                       alert (default 60, floored to a minimum of 15 so a
                       misconfigured alert can't hammer the live DB).
    db_cfg         : DB config — only ``db_type`` is persisted (no secrets)

    Returns
    ───────
    The newly created alert dict (includes ``id`` and ``created_at``).
    """
    if condition not in _VALID_CONDITIONS:
        log.warning(
            "alert_engine: unknown condition %r — defaulting to change_pct",
            condition,
        )
        condition = "change_pct"

    # A relative window ("today", "this month") was resolved to a LITERAL date
    # when this SQL was generated, and the SQL is then re-run unchanged forever.
    # Without the policy that produced that literal there is no way to move the
    # window on later checks, so the alert monitors the day it was created on:
    # a change alert that can never fire, or an above/below alert that fires on
    # every check and never stops. Keep the policy and the literal so
    # _refresh_relative_window() can re-anchor before each comparison.
    anchor_policy, anchor_value = _relative_window_of(semantic_plan)

    alert: dict[str, Any] = {
        "id": str(uuid.uuid4())[:8],
        "question": (question or "").strip(),
        "sql": sql or "",
        "metric_col": metric_col or "",
        "anchor_policy": anchor_policy,
        "anchor_value": anchor_value,
        "baseline_value": round(float(baseline_value), 4),
        "condition": condition,
        "threshold": float(threshold),
        "check_interval_minutes": max(int(check_interval_minutes or 60), 15),
        # Non-secret DB hint — needed to route check_alert_now() calls
        "db_type": str((db_cfg or {}).get("db_type", "azure_sql")),
        "account_id": account_id,
        "user_id": user_id,
        "purpose_id": purpose_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "last_checked": None,
        "last_value": None,
        "status": "active",
    }

    alerts = _load()
    alerts.append(alert)
    _save(alerts)
    log.info(
        "alert_engine: created alert %s — %r (condition=%s threshold=%s)",
        alert["id"], (question or "")[:60], condition, threshold,
    )
    return alert


def list_alerts() -> list[dict]:
    """Return all stored alert definitions (active and inactive)."""
    return _load()


def get_alert(alert_id: str) -> dict | None:
    """Return the alert with ``alert_id``, or ``None`` if not found."""
    return next(
        (a for a in _load() if a.get("id") == alert_id),
        None,
    )


def delete_alert(alert_id: str) -> bool:
    """
    Remove an alert by id.

    Returns ``True`` if it was found and deleted, ``False`` if not found.
    """
    alerts = _load()
    new_alerts = [a for a in alerts if a.get("id") != alert_id]
    if len(new_alerts) == len(alerts):
        return False
    _save(new_alerts)
    log.info("alert_engine: deleted alert %s", alert_id)
    return True


def check_alert_now(alert_id: str, db_cfg: dict) -> dict:
    """
    Re-run the alert's SQL and compare the result to the baseline.

    Pipeline
    ────────
    1. Load alert definition
    2. Execute the stored SQL against the live DB
    3. Extract the metric value from the first result row
    4. Evaluate the condition against the baseline
    5. Persist last_checked and last_value
    6. Return a structured result dict

    Returns
    ───────
    Success dict keys:
      ok, triggered, alert_id, metric_col, current_value, baseline_value,
      delta_pct, condition, threshold, message, checked_at

    Failure dict keys:
      ok=False, reason, (optional: detail, raw_value)
    """
    from core.schema import run_query  # local import keeps module lightweight

    alert = get_alert(alert_id)
    if not alert:
        return {"ok": False, "reason": "alert_not_found", "alert_id": alert_id}
    if alert.get("status") != "active":
        return {"ok": False, "reason": "alert_inactive", "alert_id": alert_id}

    # ── Move the window before running anything ───────────────────────────────
    sql_to_run, window_problem = _refresh_relative_window(alert, db_cfg or {})
    if window_problem:
        log.warning(
            "alert_engine: alert %s monitors a relative window that cannot be "
            "moved (%s) — refusing to compare against the window it was created "
            "on. Recreate the alert to re-anchor it.", alert_id, window_problem,
        )
        return {
            "ok": False, "reason": window_problem, "alert_id": alert_id,
            "detail": (
                "This alert watches a relative period, and its query is fixed "
                "to the date it was created on. Recreate it so it follows the "
                "latest available data."
            ),
        }

    # ── Execute SQL ───────────────────────────────────────────────────────────
    try:
        if alert.get("account_id") and alert.get("user_id"):
            import store
            from core.compliance.governed_query import execute_governed_query
            from core.compliance.policy_engine import evaluate, resolve_context
            from core.schema import load_known_tables, load_schema_columns

            user = store.get_user(int(alert["user_id"]))
            if not user:
                return {"ok": False, "reason": "alert_user_missing", "alert_id": alert_id}
            state = store.get_client_state(alert["account_id"])
            context = resolve_context(
                alert["account_id"], user, action="query_execution",
                channel="alert", purpose_id=alert.get("purpose_id", ""),
            )
            governed = execute_governed_query(
                db_cfg.get("credentials") or db_cfg,
                db_cfg.get("db_type", alert.get("db_type", "azure_sql")),
                sql_to_run,
                context=context,
                known_tables=load_known_tables(state.get("schema_dir", "")),
                table_columns=load_schema_columns(state.get("schema_dir", "")),
                allowed_tables=store.get_allowed_tables(user),
            )
            alert_context = resolve_context(
                alert["account_id"], user, action="alert", channel="alert",
                purpose_id=context.purpose_id,
            )
            alert_decision = evaluate(alert_context, governed.analysis.resources)
            if not alert_decision.effective_allowed:
                return {
                    "ok": False, "reason": alert_decision.reason_code,
                    "detail": alert_decision.explanation, "alert_id": alert_id,
                }
            rows = governed.rows
        else:
            rows = run_query(
                db_cfg.get("credentials") or db_cfg,
                db_cfg.get("db_type", alert.get("db_type", "azure_sql")),
                sql_to_run,
            )
    except Exception as exc:
        return {
            "ok": False, "reason": "query_failed",
            "detail": str(exc)[:120], "alert_id": alert_id,
        }

    if not rows:
        return {"ok": False, "reason": "no_rows", "alert_id": alert_id}

    # ── Resolve metric column ─────────────────────────────────────────────────
    metric_col = alert.get("metric_col", "")
    first_row  = rows[0]

    # Fall back to first numeric column when metric_col is blank or missing
    if not metric_col or metric_col not in first_row:
        metric_col = next(
            (k for k, v in first_row.items() if isinstance(v, (int, float))),
            "",
        )
    if not metric_col:
        return {
            "ok": False, "reason": "metric_not_found",
            "alert_id": alert_id,
        }

    # ── Parse current value ───────────────────────────────────────────────────
    raw_val = first_row.get(metric_col)
    try:
        current = float(str(raw_val).replace(",", ""))
    except (TypeError, ValueError):
        return {
            "ok": False, "reason": "metric_not_numeric",
            "raw_value": raw_val, "alert_id": alert_id,
        }

    # ── Evaluate condition ────────────────────────────────────────────────────
    baseline  = float(alert.get("baseline_value", 0))
    delta_pct = (
        ((current - baseline) / abs(baseline) * 100)
        if baseline != 0 else 0.0
    )
    condition = alert.get("condition", "change_pct")
    threshold = float(alert.get("threshold", 10.0))

    if condition == "above":
        triggered = current > threshold
    elif condition == "below":
        triggered = current < threshold
    else:  # change_pct
        triggered = abs(delta_pct) >= threshold

    direction = "increased" if current > baseline else "decreased"
    message = (
        f"{'⚠️ ALERT' if triggered else '✓ OK'}: "
        f"{metric_col} is now {current:,.2f} "
        f"({direction} {abs(delta_pct):.1f}% from baseline {baseline:,.2f})"
    )

    # ── Persist check state ───────────────────────────────────────────────────
    checked_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    alerts     = _load()
    for a in alerts:
        if a.get("id") == alert_id:
            a["last_checked"] = checked_at
            a["last_value"]   = round(current, 4)
    _save(alerts)

    return {
        "ok":             True,
        "triggered":      triggered,
        "alert_id":       alert_id,
        "metric_col":     metric_col,
        "current_value":  round(current, 4),
        "baseline_value": baseline,
        "delta_pct":      round(delta_pct, 2),
        "condition":      condition,
        "threshold":      threshold,
        "message":        message,
        "checked_at":     checked_at,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Scheduled due-checks — the piece the module docstring originally left out
# ══════════════════════════════════════════════════════════════════════════════

def _alert_due(alert: dict, now: datetime) -> bool:
    """
    True when `alert` hasn't been checked yet, or its check_interval_minutes
    has elapsed since last_checked. Timestamps are naive local time (matching
    create_alert's/check_alert_now's own time.strftime(...) format), so
    comparison stays naive too.
    """
    last_checked = alert.get("last_checked")
    if not last_checked:
        return True
    interval = max(int(alert.get("check_interval_minutes") or 60), 15)
    try:
        last_dt = datetime.strptime(last_checked, "%Y-%m-%dT%H:%M:%S")
    except (TypeError, ValueError):
        return True
    return (now - last_dt).total_seconds() >= interval * 60


def run_due_alert_checks() -> None:
    """
    Re-check every active alert whose interval has elapsed, and proactively
    deliver a notification when triggered. One alert's failure is logged and
    skipped — it never blocks the rest. Synchronous by design (called via
    asyncio.to_thread from core/notification_scheduler.py), but delivery
    itself is async, so each alert's delivery is run to completion inline.
    """
    import asyncio

    from core.pipeline_context import get_client_db
    from core.notify import send_proactive_notification

    now = datetime.now()
    for alert in list_alerts():
        if alert.get("status") != "active":
            continue
        if not _alert_due(alert, now):
            continue

        try:
            db_cfg = get_client_db(alert.get("account_id") or "")
            if not db_cfg:
                log.debug("run_due_alert_checks: no db_cfg for alert %s, skipping", alert.get("id"))
                continue
            result = check_alert_now(alert["id"], db_cfg)
        except Exception as exc:
            log.warning("run_due_alert_checks: check failed for alert %s: %s", alert.get("id"), exc)
            continue

        if not result.get("ok") or not result.get("triggered"):
            continue

        account_id = alert.get("account_id") or ""
        user_id = alert.get("user_id") or ""
        if not account_id or not user_id:
            log.debug("run_due_alert_checks: alert %s triggered but has no account_id/user_id for delivery", alert.get("id"))
            continue
        try:
            asyncio.run(send_proactive_notification(account_id, int(user_id), result["message"]))
        except Exception as exc:
            log.warning("run_due_alert_checks: delivery failed for alert %s: %s", alert.get("id"), exc)
