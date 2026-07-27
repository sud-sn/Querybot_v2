"""
core/notification_scheduler.py

Background loop that drives due alert checks and due report digests.
Modeled directly on core/log_export.py::scheduled_log_export_loop
— same while/try/except/sleep shape — and wired into main.py startup/
shutdown the same way (see app.state.log_export_task).
"""

from __future__ import annotations

import asyncio
import logging

log = logging.getLogger("querybot.notification_scheduler")


def run_due_notifications_once() -> None:
    """Run every due notification check. Each check is independently
    wrapped so one failing check never blocks the others."""
    from core.alert_engine import run_due_alert_checks
    from core.report_engine import run_due_report_digests

    try:
        run_due_alert_checks()
    except Exception as exc:
        log.warning("run_due_alert_checks failed: %s", exc)

    try:
        run_due_report_digests()
    except Exception as exc:
        log.warning("run_due_report_digests failed: %s", exc)


async def scheduled_notification_loop(poll_seconds: int = 60) -> None:
    while True:
        try:
            await asyncio.to_thread(run_due_notifications_once)
        except Exception as exc:
            log.warning("Scheduled notification loop error: %s", exc)
        await asyncio.sleep(poll_seconds)
