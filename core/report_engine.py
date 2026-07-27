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
