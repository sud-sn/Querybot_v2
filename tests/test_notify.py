"""
core/notify.py — shared proactive-delivery helper for alerts and report
digests.

send_proactive_notification() attempts the portal WebSocket push and a
proactive Teams send independently, and must never raise even if one (or
both) channel(s) fail. All I/O (portal_notification_hub, TeamsAdapter,
store) is mocked — no real sockets or HTTP calls.
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import core.notify as notify


def _run(coro):
    return asyncio.run(coro)


class PortalDeliveryTests(unittest.TestCase):

    def test_broadcasts_to_portal_hub_with_message_and_chart(self):
        hub = MagicMock()
        hub.broadcast_to_user = AsyncMock()
        with (
            patch("core.portal_notifications.portal_notification_hub", hub),
            patch("store.get_conversation_ref_for_user", return_value=None),
        ):
            _run(notify.send_proactive_notification("acct1", 42, "hello", chart={"chart_type": "bar"}))

        hub.broadcast_to_user.assert_called_once()
        args, _ = hub.broadcast_to_user.call_args
        self.assertEqual(args[0], 42)
        payload = args[1]
        self.assertEqual(payload["type"], "notification")
        self.assertEqual(payload["account_id"], "acct1")
        self.assertEqual(payload["message"], "hello")
        self.assertEqual(payload["chart"], {"chart_type": "bar"})

    def test_portal_failure_does_not_raise_and_teams_still_attempted(self):
        hub = MagicMock()
        hub.broadcast_to_user = AsyncMock(side_effect=Exception("socket gone"))
        with (
            patch("core.portal_notifications.portal_notification_hub", hub),
            patch("store.get_conversation_ref_for_user", return_value=None) as mock_lookup,
        ):
            _run(notify.send_proactive_notification("acct1", 42, "hello"))  # must not raise
        mock_lookup.assert_called_once()


class TeamsDeliveryTests(unittest.TestCase):

    def _pending(self, **overrides):
        base = {
            "platform_user_id": "teams-user-1",
            "conversation_ref": json.dumps({"service_url": "https://smba.example/", "conversation_id": "conv-1"}),
        }
        base.update(overrides)
        return base

    def test_no_pending_record_skips_teams_silently(self):
        hub = MagicMock()
        hub.broadcast_to_user = AsyncMock()
        with (
            patch("core.portal_notifications.portal_notification_hub", hub),
            patch("store.get_conversation_ref_for_user", return_value=None),
            patch("gateway.teams_adapter.TeamsAdapter") as MockAdapter,
        ):
            _run(notify.send_proactive_notification("acct1", 42, "hello"))
        MockAdapter.assert_not_called()

    def test_pending_without_service_url_skips_teams(self):
        hub = MagicMock()
        hub.broadcast_to_user = AsyncMock()
        pending = self._pending(conversation_ref="{}")
        with (
            patch("core.portal_notifications.portal_notification_hub", hub),
            patch("store.get_conversation_ref_for_user", return_value=pending),
            patch("gateway.teams_adapter.TeamsAdapter") as MockAdapter,
        ):
            _run(notify.send_proactive_notification("acct1", 42, "hello"))
        MockAdapter.assert_not_called()

    def test_no_active_teams_platform_skips_send(self):
        hub = MagicMock()
        hub.broadcast_to_user = AsyncMock()
        pending = self._pending()
        with (
            patch("core.portal_notifications.portal_notification_hub", hub),
            patch("store.get_conversation_ref_for_user", return_value=pending),
            patch("store.list_platforms", return_value=[{"is_active": False, "credentials": {}}]),
            patch("gateway.teams_adapter.TeamsAdapter") as MockAdapter,
        ):
            _run(notify.send_proactive_notification("acct1", 42, "hello"))
        MockAdapter.assert_not_called()

    def test_sends_message_and_chart_through_teams_adapter(self):
        hub = MagicMock()
        hub.broadcast_to_user = AsyncMock()
        pending = self._pending()
        adapter_instance = MagicMock()
        adapter_instance.send_message = AsyncMock()
        adapter_instance.send_chart = AsyncMock()
        with (
            patch("core.portal_notifications.portal_notification_hub", hub),
            patch("store.get_conversation_ref_for_user", return_value=pending),
            patch("store.list_platforms", return_value=[{"is_active": True, "credentials": {"app_id": "x"}}]),
            patch("gateway.teams_adapter.TeamsAdapter", return_value=adapter_instance) as MockAdapter,
        ):
            _run(notify.send_proactive_notification("acct1", 42, "hello", chart={"chart_type": "bar"}))

        MockAdapter.assert_called_once_with({"app_id": "x"})
        adapter_instance.send_message.assert_called_once()
        event_arg = adapter_instance.send_message.call_args[0][0]
        self.assertEqual(event_arg.account_id, "acct1")
        self.assertEqual(event_arg.user_id, "teams-user-1")
        self.assertEqual(event_arg.platform, "teams")
        adapter_instance.send_chart.assert_called_once()

    def test_no_chart_skips_send_chart_call(self):
        hub = MagicMock()
        hub.broadcast_to_user = AsyncMock()
        pending = self._pending()
        adapter_instance = MagicMock()
        adapter_instance.send_message = AsyncMock()
        adapter_instance.send_chart = AsyncMock()
        with (
            patch("core.portal_notifications.portal_notification_hub", hub),
            patch("store.get_conversation_ref_for_user", return_value=pending),
            patch("store.list_platforms", return_value=[{"is_active": True, "credentials": {}}]),
            patch("gateway.teams_adapter.TeamsAdapter", return_value=adapter_instance),
        ):
            _run(notify.send_proactive_notification("acct1", 42, "hello"))
        adapter_instance.send_chart.assert_not_called()

    def test_teams_send_failure_does_not_raise(self):
        hub = MagicMock()
        hub.broadcast_to_user = AsyncMock()
        pending = self._pending()
        adapter_instance = MagicMock()
        adapter_instance.send_message = AsyncMock(side_effect=Exception("network error"))
        with (
            patch("core.portal_notifications.portal_notification_hub", hub),
            patch("store.get_conversation_ref_for_user", return_value=pending),
            patch("store.list_platforms", return_value=[{"is_active": True, "credentials": {}}]),
            patch("gateway.teams_adapter.TeamsAdapter", return_value=adapter_instance),
        ):
            _run(notify.send_proactive_notification("acct1", 42, "hello"))  # must not raise

    def test_malformed_conversation_ref_json_skips_teams(self):
        hub = MagicMock()
        hub.broadcast_to_user = AsyncMock()
        pending = self._pending(conversation_ref="not-json")
        with (
            patch("core.portal_notifications.portal_notification_hub", hub),
            patch("store.get_conversation_ref_for_user", return_value=pending),
            patch("gateway.teams_adapter.TeamsAdapter") as MockAdapter,
        ):
            _run(notify.send_proactive_notification("acct1", 42, "hello"))
        MockAdapter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
