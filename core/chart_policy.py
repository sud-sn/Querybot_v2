"""Whether a derived visual of these values is allowed at all.

A chart and a forecast are the same kind of disclosure: neither shows a new
number, both re-present the result's values in a shape the policy engine has an
opinion about. The aggregate-only rule -- a regulated resource may be charted
only where the SQL aggregated it -- has to apply to both, from one piece of code,
or the two drift apart and the weaker one becomes the way around the stronger.

Lifted verbatim out of core/result_renderer.py, which was the only caller. The
fail-closed-for-regulated-tenants behaviour on evaluation failure is preserved
exactly, including the reason it is written that way: shadow mode governs
whether a *decision* is advisory, not whether a failed evaluation may be ignored.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("querybot.chart_policy")


def aggregate_only_gate_passes(
    *,
    account_id: Any,
    portal_user: Any,
    event: Any,
    sql: str,
    db_type: str = "azure_sql",
    what: str = "Chart",
) -> bool:
    """True when a derived visual of this result may be shown.

    `what` names the caller in the log line only -- "Chart", "Forecast" -- so a
    blocked forecast is distinguishable from a blocked chart when reading logs.
    """
    try:
        from core.compliance.policy_engine import evaluate, resolve_context
        from core.compliance.sql_guard import analyze_sql

        analysis = analyze_sql(sql, db_type or "azure_sql")
        context = resolve_context(
            account_id,
            portal_user,
            action="chart",
            channel=getattr(event, "platform", "") or "portal",
        )
        decision = evaluate(context, analysis.resources)
        aggregate_sources = {
            source
            for output, sources in analysis.lineage.items()
            if output in analysis.aggregate_outputs
            for source in sources
        }
        required_aggregate = {resource.key for resource in decision.aggregate_only}
        if (
            not decision.effective_allowed
            or bool(required_aggregate - aggregate_sources)
        ):
            return False
        return True
    except Exception as exc:
        # Previously also required enforcement_mode == "enforce", which left a
        # regulated tenant in shadow mode with no chart protection at all when
        # policy evaluation failed. Shadow governs whether a *decision* is
        # advisory, not whether a failed evaluation may be ignored.
        from core.compliance.policy_engine import is_regulated

        if is_regulated(account_id):
            log.warning("%s blocked because policy evaluation failed: %s", what, exc)
            return False
        return True
