# -*- coding: utf-8 -*-
"""The investigation trigger, over the real socket.

"investigate scrap rate at the Lyon plant" has nothing to do with a cached
result and everything to do with starting a bounded, multi-step inquiry
from scratch (core/investigation_planner.py). This proves the trigger
reaches that loop and only that loop -- an ordinary question must never be
diverted into it, and a bare "investigate" with nothing to investigate must
fall through to the ordinary question path.

The planning loop is replaced at its own boundary
(core.investigation_planner.run_investigation) for the routing tests, since
that module's own suite already proves the loop's substance; the one
end-to-end test that runs the real loop proves only the one thing the loop
cannot prove of itself -- that a regulated tenant is refused before this
product's own governed pipeline is ever touched, driven from the same
socket a real reader types into.
"""

from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import AsyncMock

import pytest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_investigation_wiring.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for _mod in list(sys.modules):
    if _mod.startswith("store"):
        del sys.modules[_mod]

import store  # noqa: E402

store.init_db()

import core.investigation as investigation  # noqa: E402
import core.investigation_planner as investigation_planner  # noqa: E402
import gateway.webhooks as wh  # noqa: E402
from core.investigation_planner import InvestigationOutcome  # noqa: E402


def _client_app():
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    app = FastAPI()
    app.include_router(wh.router)
    return TestClient(app)


def _reader():
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    store.update_client_meta(account_id, chat_ui_enabled=1, enable_llm_audit=1)
    user_id, _ = store.create_user(account_id, "Ada", f"{os.urandom(4).hex()}@x.com", password="a-password-they-chose")
    return account_id, user_id


def _drive(account_id, user_id, text, *, drain=12):
    import portal.routes as pr

    client = _client_app()
    client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
    frames = []
    with client.websocket_connect(f"/ws/chat/{account_id}") as ws:
        ws.receive_json()  # greeting or connected line
        ws.send_json({"type": "message", "text": text})
        for _ in range(drain):
            frame = ws.receive_json()
            frames.append(frame)
            if isinstance(frame, dict) and frame.get("type") == "message" and frame.get("content"):
                break
            # typing:false is the one frame _guarded_turn guarantees on every
            # path -- success, refusal, or exception -- so it is always safe
            # to stop here even when no "message" frame was ever sent.
            if isinstance(frame, dict) and frame.get("type") == "typing" and frame.get("active") is False:
                break
    return frames


def _reply(frames):
    for frame in frames:
        if frame.get("type") == "message" and frame.get("content"):
            return frame
    return {}


class TestTheTriggerReachesTheLoop:

    def test_the_synthesis_is_sent_as_the_final_message(self, monkeypatch):
        account_id, user_id = _reader()
        outcome = InvestigationOutcome(
            objective="investigate scrap rate at the Lyon plant",
            synthesis="Scrap at Lyon rose because of work centre WC10.",
            phrasing="llm",
        )

        async def _fake_loop(**kwargs):
            return outcome

        monkeypatch.setattr(investigation_planner, "run_investigation", _fake_loop)
        monkeypatch.setattr(wh, "resolve_provider", lambda *a, **k: ("test", "test", "k", {}))
        frames = _drive(account_id, user_id, "investigate scrap rate at the Lyon plant")
        reply = _reply(frames)
        assert reply.get("content") == outcome.synthesis

    def test_the_agent_run_events_are_sent(self, monkeypatch):
        account_id, user_id = _reader()

        async def _fake_loop(**kwargs):
            return InvestigationOutcome(objective="x", synthesis="Done.", phrasing="template")

        monkeypatch.setattr(investigation_planner, "run_investigation", _fake_loop)
        monkeypatch.setattr(wh, "resolve_provider", lambda *a, **k: ("test", "test", "k", {}))
        frames = _drive(account_id, user_id, "investigate scrap rate at the Lyon plant")
        types = [f.get("type") for f in frames if isinstance(f, dict)]
        assert "agent_run_started" in types
        assert "agent_run_finished" in types

    def test_the_objective_the_loop_receives_is_the_readers_own_text(self, monkeypatch):
        account_id, user_id = _reader()
        seen = {}

        async def _fake_loop(*, objective, max_steps, **kwargs):
            seen["objective"] = objective
            seen["max_steps"] = max_steps
            return InvestigationOutcome(objective=objective, synthesis="Done.", phrasing="template")

        monkeypatch.setattr(investigation_planner, "run_investigation", _fake_loop)
        monkeypatch.setattr(wh, "resolve_provider", lambda *a, **k: ("test", "test", "k", {}))
        _drive(account_id, user_id, "investigate scrap rate at the Lyon plant")
        assert seen["objective"] == "investigate scrap rate at the Lyon plant"
        assert seen["max_steps"] == wh._INVESTIGATION_MAX_STEPS

    def test_a_regulated_tenant_is_refused_through_the_real_loop(self):
        """No monkeypatch on the loop itself: the real gate, the real socket.
        run_query_tool is spied only to prove it is never reached."""
        account_id, user_id = _reader()
        spy = AsyncMock()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(investigation, "run_query_tool", spy)
            mp.setattr(wh, "resolve_provider", lambda *a, **k: ("test", "test", "k", {}))
            frames = _drive(account_id, user_id, "investigate scrap rate at the Lyon plant")
        reply = _reply(frames)
        from core.i18n import t

        assert reply.get("content") == t("investigation.regulated_refusal", "en")
        spy.assert_not_called()


class TestOrdinaryQuestionsAreNeverDiverted:

    def test_a_plain_question_does_not_reach_the_loop(self, monkeypatch):
        account_id, user_id = _reader()
        spy = AsyncMock()
        monkeypatch.setattr(investigation_planner, "run_investigation", spy)
        monkeypatch.setattr(wh, "resolve_provider", lambda *a, **k: ("test", "test", "k", {}))
        _drive(account_id, user_id, "what is total revenue by region")
        spy.assert_not_called()

    def test_bare_investigate_with_nothing_to_investigate_falls_through(self, monkeypatch):
        """resolve_provider is patched here too -- so if the trigger mistakenly
        matched, _run_investigation would reach the loop and the spy WOULD be
        called; without the patch, a real match and no match both end at the
        same "no API key" error, and this test could not tell them apart."""
        account_id, user_id = _reader()
        spy = AsyncMock()
        monkeypatch.setattr(investigation_planner, "run_investigation", spy)
        monkeypatch.setattr(wh, "resolve_provider", lambda *a, **k: ("test", "test", "k", {}))
        _drive(account_id, user_id, "investigate")
        spy.assert_not_called()

    def test_analyzing_an_existing_result_is_not_an_investigation(self, monkeypatch):
        """_ANALYSIS_WORK_INTENT_RE's own territory -- rows already on
        screen -- must keep working exactly as before."""
        account_id, user_id = _reader()
        spy = AsyncMock()
        monkeypatch.setattr(investigation_planner, "run_investigation", spy)
        monkeypatch.setattr(wh, "resolve_provider", lambda *a, **k: ("test", "test", "k", {}))
        _drive(account_id, user_id, "analyze this result deeply")
        spy.assert_not_called()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
