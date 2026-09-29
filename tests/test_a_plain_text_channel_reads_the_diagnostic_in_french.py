"""
A diagnostic reads in the reader's language on Teams, Slack and Zoom.

A failure or empty-result reply is written with English wire labels --
"Kind: validation", "Most likely reason:", "Suggested next step:" -- that the
portal parses and replaces with its own translated headings. Teams, Slack and
Zoom show the message as it is, so a French reader there read the machine kind
and the English labels around a French body. The plain-text channels now drop
the kind and show each label as the portal's heading, in the reader's language.
Teams' clarification card and its plain-text fallback were English as well.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, patch

import pytest

from core import i18n
from core.answer_formatter import format_failure_business_response, readable_diagnostic


def _failure() -> str:
    return format_failure_business_response(
        rca={"kind": "validation", "headline": "Je n'ai pas pu construire une requête fiable.",
             "most_likely_reason": "La requête a lu un autre jeu de données.",
             "suggested_next_step": "Nommez le jeu de données."},
        sql="SELECT 1",
    )


@pytest.fixture
def french():
    token = i18n.activate_language("fr")
    try:
        yield
    finally:
        i18n.deactivate_language(token)


class TestTheReadableDiagnostic:

    def test_the_kind_is_dropped_and_the_labels_read_in_french(self, french):
        text = readable_diagnostic(_failure())
        assert "Kind:" not in text and "validation" not in text
        for english in ("Most likely reason", "Suggested next step", "SQL tried"):
            assert english not in text
        assert "**Raison la plus probable :**" in text
        assert "**Prochaine étape suggérée :**" in text
        assert "**SQL tenté :**" in text
        assert "La requête a lu un autre jeu de données." in text

    def test_the_labels_read_in_english_for_an_english_reader(self):
        text = readable_diagnostic(_failure())
        assert "Kind:" not in text
        assert "**Most likely reason:**" in text

    def test_a_label_with_its_value_on_the_same_line_keeps_the_value(self, french):
        text = readable_diagnostic("Aucune ligne.\n\nKind: empty\n\nConfidence: Moyenne (50/100)")
        assert "**Confiance :** Moyenne (50/100)" in text

    def test_each_channel_marks_the_heading_its_own_way(self, french):
        assert "*Raison la plus probable :*" in readable_diagnostic(_failure(), emphasis="*")
        assert "\nRaison la plus probable :\n" in readable_diagnostic(_failure(), emphasis="")

    def test_any_other_message_is_unchanged(self):
        message = "Most likely reason: a line a reader typed, not a diagnostic."
        assert readable_diagnostic(message) == message


class _Posted:
    """httpx.AsyncClient standing in for Teams: keeps what was posted."""

    sent: list[dict] = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers=None, json=None, timeout=None):
        _Posted.sent.append(json)

        class _Ok:
            status_code = 200
            text = ""

            def raise_for_status(self):
                return None

            def json(self):
                return {"ok": True}

        return _Ok()


def _teams():
    from gateway.teams_adapter import TeamsAdapter

    adapter = TeamsAdapter.__new__(TeamsAdapter)
    adapter._get_token = AsyncMock(return_value="token")
    return adapter


def _event():
    from gateway.base import PlatformEvent

    channel = json.dumps({"service_url": "https://teams.example", "conversation_id": "c1"})
    return PlatformEvent(account_id="acct", user_id="u1", channel_id=channel, text="", platform="teams")


class TeamsReadsInFrenchTests(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self._token = i18n.activate_language("fr")
        _Posted.sent = []

    def tearDown(self):
        i18n.deactivate_language(self._token)

    async def test_a_diagnostic_is_sent_readable(self):
        with patch("gateway.teams_adapter.httpx.AsyncClient", _Posted):
            await _teams().send_message(_event(), _failure())
        (activity,) = _Posted.sent
        self.assertNotIn("Kind:", activity["text"])
        self.assertNotIn("Most likely reason", activity["text"])
        self.assertIn("**Raison la plus probable :**", activity["text"])

    async def test_the_clarification_card_is_in_french(self):
        with patch("gateway.teams_adapter.httpx.AsyncClient", _Posted):
            await _teams().send_clarification_prompt(
                _event(), "Quelle mesure ?", [{"id": "m1", "label": "Quantité commandée"}])
        (activity,) = _Posted.sent
        shown = [block["text"] for block in activity["attachments"][0]["content"]["body"]]
        self.assertEqual(shown[0], i18n.t("clar.need_more_context", lang="fr"))
        self.assertEqual(shown[2], i18n.t("clar.pick_option_below", lang="fr"))
        self.assertNotIn("I need a bit more context", " ".join(shown))

    async def test_the_clarification_without_options_is_in_french(self):
        adapter = _teams()
        adapter.send_message = AsyncMock()
        await adapter.send_clarification_prompt(_event(), "Quelle mesure ?", [])
        sent = adapter.send_message.call_args[0][1]
        self.assertNotIn("Reply in plain language", sent)
        self.assertIn("Répondez en langage courant", sent)


class SlackAndZoomReadInFrenchTests(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self._token = i18n.activate_language("fr")
        _Posted.sent = []

    def tearDown(self):
        i18n.deactivate_language(self._token)

    async def test_slack_bolds_the_french_heading_its_own_way(self):
        from gateway.slack_adapter import SlackAdapter

        adapter = SlackAdapter.__new__(SlackAdapter)
        adapter._bot_token = "token"
        with patch("gateway.slack_adapter.httpx.AsyncClient", _Posted):
            await adapter.send_message(_event(), _failure())
        (message,) = _Posted.sent
        self.assertNotIn("Kind:", message["text"])
        self.assertIn("*Raison la plus probable :*", message["text"])
        self.assertNotIn("**", message["text"])

    async def test_zoom_shows_the_french_heading_plain(self):
        from gateway.zoom_adapter import ZoomAdapter

        adapter = ZoomAdapter.__new__(ZoomAdapter)
        adapter._bot_jid = "bot"
        adapter._get_token = AsyncMock(return_value="token")
        with patch("gateway.zoom_adapter.httpx.AsyncClient", _Posted):
            await adapter.send_message(_event(), _failure())
        (payload,) = _Posted.sent
        text = payload["content"]["body"][0]["text"]
        self.assertNotIn("Kind:", text)
        self.assertIn("\nRaison la plus probable :\n", text)


if __name__ == "__main__":
    unittest.main()
