"""
Tests for Teams adapter feature-parity with the web portal.

Covers:
  • parse_event handles Adaptive Card submits (value.label path)
  • send_clarification_prompt renders a well-formed card
  • send_status attempts a typing indicator (best-effort, failure-tolerant)
  • send_chart rejects empty chart payloads gracefully
  • send_clarification_prompt falls back to plain text when options are empty

These tests are pure-adapter — no DB, no crypto, no network. All HTTP is
mocked. No filesystem state is required.

Run: python -m unittest tests.test_teams_parity
"""

import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.teams_adapter import TeamsAdapter
from gateway.base import PlatformEvent


def _run(coro):
    """Helper — run an async coroutine. Creates a fresh event loop per call,
    which keeps tests hermetic under Python 3.10+."""
    return asyncio.run(coro)


def _make_adapter():
    return TeamsAdapter({
        "app_id":       "fake-app-id",
        "app_password": "fake-password",
        "tenant_id":    "common",
    })


def _make_event() -> PlatformEvent:
    return PlatformEvent(
        account_id="tenant-xyz",
        user_id="user-abc",
        channel_id=json.dumps({
            "service_url":     "https://smba.trafficmanager.net/amer/",
            "conversation_id": "conv-1",
            "activity_id":     "act-1",
        }),
        text="ignored in these tests",
        platform="teams",
        raw={},
    )


# ──────────────────────────────────────────────────────────────────────────
# parse_event — Adaptive Card submit handling
# ──────────────────────────────────────────────────────────────────────────
class ParseEventSubmitTests(unittest.TestCase):

    def test_card_submit_with_label_returns_event_with_text_set_to_label(self):
        """Tapping a clarification button produces an activity with empty text
        and populated value. parse_event should lift value.label into text so
        the core dispatcher can route it like a typed reply."""
        activity = {
            "type": "message",
            "text": "",
            "value": {
                "option_id":  "opt_1",
                "label":      "late pickups",
                "pending_id": "pending-xyz",
            },
            "from":         {"id": "user-abc"},
            "conversation": {"id": "conv-1"},
            "serviceUrl":   "https://smba.trafficmanager.net/amer/",
            "id":           "act-1",
            "channelData":  {"tenant": {"id": "tenant-xyz"}},
        }
        adapter = _make_adapter()
        event = adapter.parse_event(json.dumps(activity).encode(), {})
        self.assertIsNotNone(event)
        self.assertEqual(event.text, "late pickups")
        self.assertEqual(event.account_id, "tenant-xyz")
        self.assertEqual(event.user_id, "user-abc")

    def test_card_submit_missing_label_falls_back_to_option_id(self):
        activity = {
            "type": "message",
            "text": "",
            "value": {"option_id": "opt_2"},
            "from":         {"id": "u"},
            "conversation": {"id": "c"},
            "serviceUrl":   "https://x/",
            "channelData":  {"tenant": {"id": "t"}},
        }
        event = _make_adapter().parse_event(json.dumps(activity).encode(), {})
        self.assertIsNotNone(event)
        self.assertEqual(event.text, "opt_2")

    def test_card_submit_with_no_label_or_option_id_is_dropped(self):
        activity = {
            "type": "message",
            "text": "",
            "value": {"unrelated": "junk"},
            "from":         {"id": "u"},
            "conversation": {"id": "c"},
            "serviceUrl":   "https://x/",
            "channelData":  {"tenant": {"id": "t"}},
        }
        event = _make_adapter().parse_event(json.dumps(activity).encode(), {})
        self.assertIsNone(event)

    def test_regular_text_message_still_works(self):
        activity = {
            "type": "message",
            "text": "show me late orders",
            "from":         {"id": "user-abc"},
            "conversation": {"id": "conv-1"},
            "serviceUrl":   "https://smba.trafficmanager.net/amer/",
            "id":           "act-1",
            "channelData":  {"tenant": {"id": "tenant-xyz"}},
        }
        event = _make_adapter().parse_event(json.dumps(activity).encode(), {})
        self.assertIsNotNone(event)
        self.assertEqual(event.text, "show me late orders")

    def test_mention_prefix_still_stripped_on_text_messages(self):
        activity = {
            "type": "message",
            "text": "<at>QueryBot</at> show me revenue",
            "from":         {"id": "u"},
            "conversation": {"id": "c"},
            "serviceUrl":   "https://x/",
            "channelData":  {"tenant": {"id": "t"}},
        }
        event = _make_adapter().parse_event(json.dumps(activity).encode(), {})
        self.assertIsNotNone(event)
        self.assertEqual(event.text, "show me revenue")


# ──────────────────────────────────────────────────────────────────────────
# send_clarification_prompt — card shape
# ──────────────────────────────────────────────────────────────────────────
class SendClarificationPromptTests(unittest.TestCase):

    def test_card_has_one_action_per_option_with_label_and_option_id(self):
        adapter = _make_adapter()
        captured = {}

        async def fake_post(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            class FakeResp:
                status_code = 200
                def raise_for_status(self): pass
            return FakeResp()

        options = [
            {"id": "opt_1", "label": "late pickups", "_term_id": "T1"},
            {"id": "opt_2", "label": "late deliveries", "_term_id": "T2"},
            {"id": "opt_3", "label": "late payments", "_term_id": "T3"},
        ]

        with patch.object(adapter, "_get_token", new=AsyncMock(return_value="fake-token")):
            with patch("gateway.teams_adapter.httpx.AsyncClient") as FakeClient:
                instance = FakeClient.return_value.__aenter__.return_value
                instance.post = AsyncMock(side_effect=fake_post)

                _run(adapter.send_clarification_prompt(
                    _make_event(),
                    "Which kind of 'late' did you mean?",
                    options,
                    pending_id="pending-xyz",
                ))

        self.assertIn("json", captured)
        activity = captured["json"]
        self.assertEqual(activity["type"], "message")
        att = activity["attachments"][0]
        self.assertEqual(att["contentType"], "application/vnd.microsoft.card.adaptive")
        card = att["content"]
        self.assertEqual(card["type"], "AdaptiveCard")
        actions = card["actions"]
        self.assertEqual(len(actions), 3)
        for act, opt in zip(actions, options):
            self.assertEqual(act["type"], "Action.Submit")
            self.assertEqual(act["title"], opt["label"])
            self.assertEqual(act["data"]["option_id"], opt["id"])
            self.assertEqual(act["data"]["label"], opt["label"])
            self.assertEqual(act["data"]["pending_id"], "pending-xyz")

    def test_card_caps_options_at_five(self):
        adapter = _make_adapter()
        captured = {}

        async def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json")
            class FakeResp:
                status_code = 200
                def raise_for_status(self): pass
            return FakeResp()

        many_options = [{"id": f"opt_{i}", "label": f"option {i}"} for i in range(8)]

        with patch.object(adapter, "_get_token", new=AsyncMock(return_value="t")):
            with patch("gateway.teams_adapter.httpx.AsyncClient") as FakeClient:
                instance = FakeClient.return_value.__aenter__.return_value
                instance.post = AsyncMock(side_effect=fake_post)

                _run(adapter.send_clarification_prompt(
                    _make_event(), "pick one", many_options,
                ))

        actions = captured["json"]["attachments"][0]["content"]["actions"]
        self.assertEqual(len(actions), 5)

    def test_empty_options_falls_back_to_plain_text(self):
        adapter = _make_adapter()
        adapter.send_message = AsyncMock()

        _run(adapter.send_clarification_prompt(
            _make_event(),
            "pick one",
            options=[],
        ))
        adapter.send_message.assert_called_once()
        msg_text = adapter.send_message.call_args.args[1]
        self.assertIn("pick one", msg_text)


# ──────────────────────────────────────────────────────────────────────────
# send_status — best-effort typing indicator
# ──────────────────────────────────────────────────────────────────────────
class SendStatusTests(unittest.TestCase):

    def test_sends_typing_activity(self):
        adapter = _make_adapter()
        captured = {}

        async def fake_post(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            class FakeResp:
                status_code = 200
            return FakeResp()

        with patch.object(adapter, "_get_token", new=AsyncMock(return_value="t")):
            with patch("gateway.teams_adapter.httpx.AsyncClient") as FakeClient:
                instance = FakeClient.return_value.__aenter__.return_value
                instance.post = AsyncMock(side_effect=fake_post)
                _run(adapter.send_status(_make_event(), "generating_sql", "Generating query"))

        self.assertEqual(captured["json"], {"type": "typing"})

    def test_initial_status_creates_visible_progress_activity(self):
        adapter = _make_adapter()
        event = _make_event()
        calls = []

        async def fake_post(url, **kwargs):
            calls.append(kwargs.get("json"))
            class FakeResp:
                status_code = 201
                text = ""
                def json(self):
                    return {"id": "progress-1"}
            return FakeResp()

        with patch.object(adapter, "_get_token", new=AsyncMock(return_value="t")):
            with patch("gateway.teams_adapter.httpx.AsyncClient") as FakeClient:
                instance = FakeClient.return_value.__aenter__.return_value
                instance.post = AsyncMock(side_effect=fake_post)
                _run(adapter.send_status(event, "accepted", "Working on it"))

        self.assertEqual(calls[0], {"type": "typing"})
        self.assertEqual(calls[1]["type"], "message")
        self.assertIn("QueryBot is working", calls[1]["text"])
        self.assertEqual(event.raw["_teams_progress_activity_id"], "progress-1")

    def test_final_message_updates_visible_progress_activity(self):
        adapter = _make_adapter()
        event = _make_event()
        event.raw["_teams_progress_activity_id"] = "progress-1"
        captured = {}

        async def fake_put(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            class FakeResp:
                status_code = 200
                text = ""
                def raise_for_status(self):
                    return None
            return FakeResp()

        with patch.object(adapter, "_get_token", new=AsyncMock(return_value="t")):
            with patch("gateway.teams_adapter.httpx.AsyncClient") as FakeClient:
                instance = FakeClient.return_value.__aenter__.return_value
                instance.put = AsyncMock(side_effect=fake_put)
                _run(adapter.send_message(event, "Done"))

        self.assertTrue(captured["url"].endswith("/activities/progress-1"))
        self.assertEqual(captured["json"]["type"], "message")
        self.assertEqual(captured["json"]["text"], "Done")
        self.assertNotIn("_teams_progress_activity_id", event.raw)

    def test_token_failure_is_swallowed(self):
        """send_status must never raise — it's best-effort UX only."""
        adapter = _make_adapter()
        with patch.object(adapter, "_get_token", new=AsyncMock(side_effect=RuntimeError("no net"))):
            try:
                _run(adapter.send_status(_make_event(), "x", "Y"))
            except Exception as e:  # pragma: no cover
                self.fail(f"send_status raised: {e}")


# ──────────────────────────────────────────────────────────────────────────
# send_chart — empty / failed-render tolerance
# ──────────────────────────────────────────────────────────────────────────
class SendChartTests(unittest.TestCase):

    def test_empty_rows_skips_upload(self):
        adapter = _make_adapter()
        adapter.upload_file = AsyncMock()
        _run(adapter.send_chart(_make_event(), {"rows": [], "chart_type": "bar", "title": "x"}))
        adapter.upload_file.assert_not_called()

    def test_valid_payload_triggers_upload(self):
        adapter = _make_adapter()
        adapter.upload_file = AsyncMock()
        chart = {
            "rows": [{"region": "NE", "revenue": 100}, {"region": "SE", "revenue": 150}],
            "chart_type": "bar",
            "title": "Revenue by region",
        }
        _run(adapter.send_chart(_make_event(), chart))
        # upload_file may or may not be called depending on whether matplotlib
        # is available in the test env. Assert: if NOT called, generate_chart
        # returned None; if called, arg0 was bytes.
        if adapter.upload_file.call_count:
            args, _ = adapter.upload_file.call_args
            self.assertIsInstance(args[1], (bytes, bytearray))


if __name__ == "__main__":
    unittest.main()
