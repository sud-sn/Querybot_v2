"""
core/notify.py

Shared proactive-delivery helper for alerts (core/alert_engine.py) and
report digests (core/report_engine.py) — anything that needs to push a
message to a user outside a live request/response cycle.

Two channels, attempted independently, best-effort:
  1. Portal — core.portal_notifications.portal_notification_hub, a live
     WebSocket push. No live session required; a no-op if the user isn't
     currently connected (matches how semantic_feedback_reviewed events
     already work).
  2. Teams — a proactive send via a stored conversation_ref on the user's
     approved pending_platform_user row, mirroring the exact construction
     admin/routes.py's approval-notice proactive send already uses
     (TeamsAdapter + a synthetic PlatformEvent built from conversation_ref).

Never raises — each channel's failure is caught and logged independently
so one broken channel never blocks the other. Says whether any channel
delivered the message: a caller that tells someone once needs to know they
were told.
"""

from __future__ import annotations

import json
import logging

log = logging.getLogger("querybot.notify")


async def send_proactive_notification(
    account_id: str,
    user_id: int,
    message: str,
    chart: dict | None = None,
) -> bool:
    """Deliver `message` (and optional `chart`) to `user_id` via every
    channel available to them. Best-effort on each channel independently.
    True when at least one of them delivered it."""
    portal = await _send_via_portal(account_id, user_id, message, chart)
    teams = await _send_via_teams(account_id, user_id, message, chart)
    return portal or teams


async def _send_via_portal(
    account_id: str, user_id: int, message: str, chart: dict | None,
) -> bool:
    try:
        from core.portal_notifications import portal_notification_hub
        delivered = await portal_notification_hub.deliver(int(user_id), {
            "type": "notification",
            "account_id": account_id,
            "message": message,
            "chart": chart,
        })
        return delivered > 0
    except Exception as exc:
        log.warning("send_proactive_notification: portal delivery failed for user %s: %s", user_id, exc)
        return False


async def _send_via_teams(
    account_id: str, user_id: int, message: str, chart: dict | None,
) -> bool:
    try:
        import store
        pending = store.get_conversation_ref_for_user(account_id, user_id)
        if not pending:
            return False
        try:
            conv_ref = json.loads(pending.get("conversation_ref") or "{}")
        except (ValueError, TypeError):
            conv_ref = {}
        if not conv_ref.get("service_url"):
            return False

        teams_platforms = store.list_platforms("teams")
        active_teams = [p for p in teams_platforms if p.get("is_active")]
        if not active_teams:
            return False

        from gateway.teams_adapter import TeamsAdapter
        from gateway.base import PlatformEvent

        adapter = TeamsAdapter(active_teams[0]["credentials"])
        synthetic_event = PlatformEvent(
            account_id=account_id,
            user_id=pending["platform_user_id"],
            channel_id=pending["conversation_ref"],
            text="",
            platform="teams",
        )
        await adapter.send_message(synthetic_event, message)
        if chart:
            await adapter.send_chart(synthetic_event, chart)
        return True
    except Exception as exc:
        log.warning("send_proactive_notification: Teams delivery failed for user %s: %s", user_id, exc)
        return False
