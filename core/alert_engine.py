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
• Scheduling and push-notification are intentionally out of scope here.
  An external scheduler would call check_alert_now() on a cron.

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
"""

from __future__ import annotations

import json
import logging
import time
import uuid
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
    db_cfg: dict | None = None,
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

    alert: dict[str, Any] = {
        "id": str(uuid.uuid4())[:8],
        "question": (question or "").strip(),
        "sql": sql or "",
        "metric_col": metric_col or "",
        "baseline_value": round(float(baseline_value), 4),
        "condition": condition,
        "threshold": float(threshold),
        # Non-secret DB hint — needed to route check_alert_now() calls
        "db_type": str((db_cfg or {}).get("db_type", "azure_sql")),
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

    # ── Execute SQL ───────────────────────────────────────────────────────────
    try:
        rows = run_query(
            db_cfg.get("credentials") or db_cfg,
            db_cfg.get("db_type", alert.get("db_type", "azure_sql")),
            alert["sql"],
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
