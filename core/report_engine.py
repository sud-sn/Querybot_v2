"""
core/report_engine.py

Named, multi-metric report execution — the "reports" half of the alerts/
reports/notifications feature. A report (store/report_store.py) is an
admin-defined group of metric_registry rows. This module runs each metric
in a report for a specific user, respecting that user's own table-level
access (never running SQL against a table they can't see, and never
naming a denied table in the response — only the metric name), and builds
a chart per metric via the existing core/chart.py pipeline.

Mirrors core/alert_engine.py::check_alert_now's governed-execution pattern
exactly (resolve_context + execute_governed_query + a second evaluate()
gate) so reports get the same compliance guarantees alerts already have.
"""

from __future__ import annotations

import logging
from datetime import datetime

log = logging.getLogger("querybot.report_engine")


def run_metric_for_report(account_id: str, user: dict, metric: dict) -> dict:
    """
    Run one report metric for one user.

    Returns on denial or failure:
      {"ok": False, "reason": ..., "metric_name": ..., "detail": (optional)}
    Returns on success:
      {"ok": True, "metric_name": ..., "rows": [...], "chart": dict|None}
    """
    import store
    from core.pipeline_context import get_client_db

    metric_name = metric.get("name") or "this metric"
    base_table = (metric.get("base_table") or "").strip()

    if base_table:
        allowed_tables = store.get_allowed_tables(user)
        if allowed_tables is not None and base_table not in allowed_tables:
            return {"ok": False, "reason": "access_denied", "metric_name": metric_name}

    sql = (metric.get("sql_template") or "").strip()
    if not sql:
        return {"ok": False, "reason": "no_sql", "metric_name": metric_name}

    db_cfg = get_client_db(account_id)
    if not db_cfg:
        return {"ok": False, "reason": "no_database", "metric_name": metric_name}

    try:
        from core.compliance.governed_query import execute_governed_query
        from core.compliance.policy_engine import evaluate, resolve_context
        from core.schema import load_known_tables, load_schema_columns

        state = store.get_client_state(account_id)
        context = resolve_context(
            account_id, user, action="query_execution",
            channel="report", purpose_id="",
        )
        governed = execute_governed_query(
            db_cfg.get("credentials") or db_cfg,
            db_cfg.get("db_type", "azure_sql"),
            sql,
            context=context,
            known_tables=load_known_tables(state.get("schema_dir", "")),
            table_columns=load_schema_columns(state.get("schema_dir", "")),
            allowed_tables=store.get_allowed_tables(user),
        )
        report_context = resolve_context(
            account_id, user, action="report", channel="report",
            purpose_id=context.purpose_id,
        )
        decision = evaluate(report_context, governed.analysis.resources)
        if not decision.effective_allowed:
            return {"ok": False, "reason": "access_denied", "metric_name": metric_name}
        rows = governed.rows
    except Exception as exc:
        log.warning("run_metric_for_report: query failed for metric %s: %s", metric_name, exc)
        return {"ok": False, "reason": "query_failed", "metric_name": metric_name, "detail": str(exc)[:120]}

    if not rows:
        return {"ok": True, "metric_name": metric_name, "rows": [], "chart": None}

    chart = None
    try:
        from core.chart import build_chart_payload, detect_chart_type
        chart_type = detect_chart_type(rows, metric_name)
        if chart_type:
            chart = build_chart_payload(rows, chart_type, title=metric_name, question=metric_name)
    except Exception as exc:
        log.debug("run_metric_for_report: chart build failed for metric %s: %s", metric_name, exc)

    return {"ok": True, "metric_name": metric_name, "rows": rows, "chart": chart}


def _format_metric_line(result: dict) -> str:
    """One text line summarizing a single metric result for the report message."""
    metric_name = result.get("metric_name", "Metric")
    if not result.get("ok"):
        reason = result.get("reason", "")
        if reason == "access_denied":
            return f"🔒 **{metric_name}** — you don't have access to this metric. Ask your admin for access."
        return f"⚠️ **{metric_name}** — couldn't be computed right now."

    rows = result.get("rows") or []
    if not rows:
        return f"**{metric_name}** — no data."
    if len(rows) == 1 and len(rows[0]) == 1:
        from core.response_builder import _safe_cell
        value = next(iter(rows[0].values()))
        return f"**{metric_name}**: {_safe_cell(value)}"
    return f"**{metric_name}** — see chart below."


def build_report_response(account_id: str, user: dict, report: dict) -> dict:
    """
    Build the full reply for a report ask/digest: three deterministic
    (non-LLM) messages checked in order, then one message + chart per
    accessible metric.

    Returns {"ok": bool, "message": str, "items": [{"text": str, "chart": dict|None}, ...]}
    `items` is empty whenever `ok` is False (nothing to render).
    """
    import store
    from store import report_store

    report_id = report.get("id")
    report_name = report.get("name") or "report"

    all_metrics = store.list_metrics(account_id)
    if not all_metrics:
        return {
            "ok": False,
            "message": "There are no metrics set up for this account yet — ask your admin to add some to the metric registry first.",
            "items": [],
        }

    report_metrics = report_store.list_report_metrics(report_id)
    if not report_metrics:
        return {
            "ok": False,
            "message": f"The \"{report_name}\" report has no metrics assigned yet — ask your admin to add some.",
            "items": [],
        }

    items = []
    for metric in report_metrics:
        result = run_metric_for_report(account_id, user, metric)
        items.append({
            "text": _format_metric_line(result),
            "chart": result.get("chart"),
        })

    return {"ok": True, "message": f"**{report_name}**", "items": items}


# ══════════════════════════════════════════════════════════════════════════════
# Scheduled digests — report_subscription due-checks + delivery
# ══════════════════════════════════════════════════════════════════════════════

def _subscription_due(sub: dict, now) -> bool:
    """
    True once `now.hour` reaches the subscription's configured hour AND
    (for weekly cadence) today is the configured day_of_week, AND it hasn't
    already been sent today. Mirrors core/log_export.py::_is_due's
    hour-gate + date-string-comparison shape (naive local time, matching
    this module's and alert_engine's own timestamp convention).
    """
    hour = int(sub.get("hour") if sub.get("hour") is not None else 8)
    if now.hour < hour:
        return False
    if sub.get("cadence") == "weekly":
        day_of_week = int(sub.get("day_of_week") if sub.get("day_of_week") is not None else 0)
        if now.weekday() != day_of_week:
            return False
    last_sent = sub.get("last_sent")
    if not last_sent:
        return True
    return str(last_sent)[:10] != now.date().isoformat()


async def _deliver_report_response(account_id: str, user_id: int, response: dict) -> None:
    """Send a report response through the shared proactive-delivery
    channel (portal WebSocket + Teams) — used by digests, which run
    outside any live request/response cycle and so have no adapter/event
    in hand the way the dispatcher's live chat ask does."""
    from core.notify import send_proactive_notification

    await send_proactive_notification(account_id, user_id, response["message"])
    if not response.get("ok"):
        return
    for item in response.get("items", []):
        await send_proactive_notification(account_id, user_id, item["text"], chart=item.get("chart"))


def run_due_report_digests() -> None:
    """
    Re-check every active report_subscription whose scheduled time has
    arrived, and deliver a digest when due. One subscription's failure is
    logged and skipped — it never blocks the rest. Synchronous by design
    (called via asyncio.to_thread from core/notification_scheduler.py);
    delivery itself is async and run to completion inline per subscription,
    mirroring alert_engine.run_due_alert_checks.
    """
    import asyncio

    import store
    from store import report_store

    now = datetime.now()
    for sub in report_store.list_subscriptions():
        if sub.get("status") != "active":
            continue
        if not _subscription_due(sub, now):
            continue

        try:
            report = report_store.get_report(sub["report_id"], sub["account_id"])
            if not report:
                log.debug("run_due_report_digests: report %s missing for subscription %s",
                          sub.get("report_id"), sub.get("id"))
                continue
            user = store.get_user(int(sub["user_id"]))
            if not user:
                log.debug("run_due_report_digests: user %s missing for subscription %s",
                          sub.get("user_id"), sub.get("id"))
                continue
            response = build_report_response(sub["account_id"], user, report)
        except Exception as exc:
            log.warning("run_due_report_digests: build failed for subscription %s: %s", sub.get("id"), exc)
            continue

        try:
            asyncio.run(_deliver_report_response(sub["account_id"], int(sub["user_id"]), response))
        except Exception as exc:
            log.warning("run_due_report_digests: delivery failed for subscription %s: %s", sub.get("id"), exc)
            continue

        report_store.update_subscription(sub["id"], {"last_sent": now.strftime("%Y-%m-%d %H:%M:%S")})
