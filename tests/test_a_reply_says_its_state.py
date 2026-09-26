"""
A reply says what it means for the conversation in a field, not in its first
character.

The chat ended a run -- marked it failed or waiting on the reader, restored the
composer, showed the error mascot -- when a message's text started with "❌" or
"❓". The icon work removes such characters from the product's pages, and a
client that reads meaning from a glyph breaks the day the glyph goes (or the
day an answer happens to start with one).

Now gateway/base.py reads the marker once, on the server, and the web adapter
sends the state beside the text; the chat's _terminalRunState reads only that
field. These run the real adapter, the real catalogue and the real page
function (dukpy).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import dukpy
import pytest

from core.i18n import MESSAGES
from gateway.web_adapter import WebAdapter

ROOT = Path(__file__).resolve().parents[1]


def message_state(text):
    from gateway.base import message_state as read

    return read(text)


class TestTheServerReadsTheMarkerOnce:

    @pytest.mark.parametrize("text,state", [
        ("❌ Monthly query limit reached (500/500).", "failed"),
        ("  ❓ Which date do you mean?", "needs_input"),
        ("Net sales were $3.06M, up 4.2% on the previous month.", ""),
        ("", ""),
        (None, ""),
    ])
    def test_it_names_the_state(self, text, state):
        assert message_state(text) == state

    @pytest.mark.parametrize("lang", ["en", "fr"])
    def test_every_catalogue_reply_that_marks_itself_has_a_state(self, lang):
        marked = {key: entry[lang] for key, entry in MESSAGES.items()
                  if isinstance(entry, dict) and str(entry.get(lang, "")).lstrip()[:1] in ("❌", "❓")}
        assert "terminal.query_limit_reached" in marked
        for key, text in marked.items():
            assert message_state(text) in ("failed", "needs_input"), key


def _sent(text: str) -> dict:
    ws = AsyncMock()
    adapter = WebAdapter(ws, "acct", "7", thread_id="t1")
    asyncio.run(adapter.send_message(adapter.make_event("q"), text))
    return ws.send_json.await_args.args[0]


class TestTheWebChatReceivesIt:

    def test_a_failure_is_sent_as_failed(self):
        payload = _sent(MESSAGES["terminal.query_limit_reached"]["en"].format(used=500, limit=500))
        assert payload["type"] == "message" and payload["state"] == "failed"

    def test_a_question_back_is_sent_as_needs_input(self):
        payload = _sent("❓ Which date do you mean: order date or ship date?")
        assert payload["state"] == "needs_input"

    def test_an_answer_carries_no_state(self):
        payload = _sent("Net sales were $3.06M.")
        assert "state" not in payload
        assert payload["content"] == "Net sales were $3.06M."


def _run_state(msg: dict) -> str:
    from tests.js_lift import function as lift

    page = (ROOT / "portal" / "templates" / "portal_chat.html").read_text(encoding="utf-8")
    return dukpy.evaljs(lift(page, "function _terminalRunState(msg)") + f"\n_terminalRunState({json.dumps(msg)})")


class TestTheChatReadsTheField:

    def test_failed_ends_the_run_failed(self):
        assert _run_state({"type": "message", "state": "failed", "content": "Monthly query limit reached."}) == "failed"

    def test_needs_input_waits_for_the_reader(self):
        assert _run_state({"type": "message", "state": "needs_input", "content": "Which date?"}) == "waiting_for_user"

    @pytest.mark.parametrize("content", ["❌ Monthly query limit reached.", "❓ Which date?"])
    def test_a_marker_in_the_text_alone_ends_nothing(self, content):
        assert _run_state({"type": "message", "content": content}) == ""

    def test_the_message_handler_decides_by_it(self):
        # The handler for type "message" must ask _terminalRunState, and no
        # page script may test a message's first character for a marker.
        page = (ROOT / "portal" / "templates" / "portal_chat.html").read_text(encoding="utf-8")
        assert "const terminalState = _terminalRunState(msg);" in page
        for source in list((ROOT / "portal" / "templates").glob("*.html")) + list((ROOT / "static" / "js").glob("*.js")):
            text = source.read_text(encoding="utf-8")
            assert "startsWith('❌')" not in text and "startsWith('❓')" not in text, source.name
