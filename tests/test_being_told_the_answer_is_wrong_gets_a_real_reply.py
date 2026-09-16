# -*- coding: utf-8 -*-
"""Told the answer was wrong, the product recited three tips.

"this is wrong", "that's not what I asked", "c'est faux": the conversational
front door classified these as frustration and sent the catalogue's reply --
an apology and the same three tips, whatever had just been answered. A reader
looking at a total for the wrong period was told to use the thumbs-down button.

The complaint is about a specific answer, so the reply is now about that
answer: the analyst, handed the conversation so far and the result on screen
(where the tenant's policy allows), told the reader has just called the answer
wrong, and asked for one concrete way to correct course. The catalogue's tips
remain the fallback whenever the analyst declines, hands the turn to the query
pipeline, or fails -- a complaint is never answered with silence or a query.

And because the analyst now reads the result's brief, a reply that asserts a
figure the brief does not hold is withheld: the analyst may read the result,
never invent one.

Every test executes the real functions -- the reply builder, the figure check,
the analyst prompt, and the dispatcher's front door end to end -- with the
model replaced at its boundary and the prompt it received read back.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_told_wrong.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for _mod in list(sys.modules):
    if _mod.startswith("store"):
        del sys.modules[_mod]

import store  # noqa: E402

store.init_db()

import core.dispatcher as dispatcher  # noqa: E402

ROWS = [
    {"REGION": "North", "NET_AMOUNT": 100.0},
    {"REGION": "South", "NET_AMOUNT": 60.0},
    {"REGION": "East", "NET_AMOUNT": 25.0},
    {"REGION": "West", "NET_AMOUNT": 5.0},
]
QUESTION = "revenue by region"


def _en(key, **kw):
    from core.i18n import t
    return t(key, "en", **kw)


@pytest.fixture
def model(monkeypatch):
    import core.compliance.policy_engine as pe
    import core.llm

    seen = {"prompts": [], "reply": (
        "Understood. The result on screen is revenue by region across four regions, "
        "led by North; if you meant a period, tell me which and I will rerun it.")}

    async def _complete(system="", user="", *a, **k):
        seen["prompts"].append((str(system), str(user)))
        return seen["reply"], 10, 10

    monkeypatch.setattr(core.llm, "llm_complete", _complete)
    monkeypatch.setattr(dispatcher, "llm_complete", _complete)
    monkeypatch.setattr(dispatcher, "resolve_provider", lambda *a, **k: ("test", "test", "k", {}))
    monkeypatch.setattr(dispatcher, "_build_analyst_context", lambda *a, **k: "Metrics: Net Revenue")
    monkeypatch.setattr(pe, "result_llm_features_allowed", lambda _a: True)
    return seen


def _adapter(with_result: bool):
    from core.result_cache import result_cache
    from gateway.web_adapter import WebAdapter

    adapter = WebAdapter(AsyncMock(), "acct", "7", thread_id="t1")
    result_cache.clear(adapter.session_id)
    if with_result:
        result_cache.store(adapter.session_id, list(ROWS), question=QUESTION, sql="SELECT 1")
    return adapter


class TestTheFigureCheck:

    def test_a_figure_the_brief_holds_passes(self):
        assert dispatcher._reply_asserts_an_unseen_figure(
            "North leads with 100 of the 190 total.", "Leader: North 100.0\nTotal: 190.0") == ""

    def test_a_figure_the_brief_does_not_hold_is_named(self):
        assert dispatcher._reply_asserts_an_unseen_figure(
            "The total is about 2,400.", "Total: 190.0") == "2400"

    def test_small_numbers_are_not_claims(self):
        assert dispatcher._reply_asserts_an_unseen_figure("Across the 4 regions, top 3 lead.", "") == ""

    def test_a_year_the_reader_typed_is_allowed(self):
        assert dispatcher._reply_asserts_an_unseen_figure(
            "For 2023 I would need to rerun it.", "Total: 190\nwhat about 2023?") == ""


class TestTheAnalystNeverInventsAFigure:

    def _ask(self, text, **kwargs):
        return asyncio.run(dispatcher._generate_analyst_reply(text, "acct", {}, **kwargs))

    def test_an_invented_figure_is_withheld(self, model):
        model["reply"] = "The total revenue is 999 across the regions."
        out = self._ask("what did that show?", result_question=QUESTION, result_brief="Total: 190.0")
        assert out == _en("reply.analyst.figure_withheld")

    def test_a_figure_from_the_brief_is_kept(self, model):
        model["reply"] = "The total revenue is 190 across the regions."
        out = self._ask("what did that show?", result_question=QUESTION, result_brief="Total: 190.0")
        assert out == model["reply"]

    def test_without_a_brief_the_reply_is_not_second_guessed(self, model):
        model["reply"] = "QueryBot answers questions about 12,000 tables? No -- about your workspace."
        assert self._ask("what can you do?") == model["reply"]

    def test_told_wrong_reaches_the_prompt(self, model):
        self._ask("this is wrong", told_wrong=True)
        system, _user = model["prompts"][0]
        assert "The reader has just said the last answer was wrong or unhelpful." in system
        assert "Told the answer was wrong" in system

    def test_not_told_wrong_is_not_claimed(self, model):
        self._ask("what can you do?")
        system, _user = model["prompts"][0]
        assert "The reader has just said the last answer was wrong" not in system


class TestTheReplyToAComplaint:

    def _reply(self, text, *, with_result=True, portal_user=None):
        adapter = _adapter(with_result)
        event = adapter.make_event(text)
        try:
            return asyncio.run(dispatcher._reply_to_frustration(
                "acct", event, adapter, text, portal_user, {})), adapter
        finally:
            from core.result_cache import result_cache
            result_cache.clear(adapter.session_id)

    def test_it_is_about_the_result_on_screen(self, model):
        reply, adapter = self._reply("this is wrong")
        assert reply == model["reply"]
        system, _user = model["prompts"][0]
        assert f"Result on screen (it answers: {QUESTION}):" in system
        assert "North" in system
        assert "The reader has just said the last answer was wrong or unhelpful." in system

    def test_the_exchange_is_remembered_for_the_next_turn(self, model):
        _reply, adapter = self._reply("that's not what I asked")
        assert [t["message"] for t in adapter.get_analyst_history()] == ["that's not what I asked"]
        assert adapter.get_analyst_history()[0]["reply"] == model["reply"]

    def test_without_a_result_the_analyst_is_told_of_none(self, model):
        reply, _adapter_ = self._reply("this is wrong", with_result=False)
        assert reply == model["reply"]
        system, _user = model["prompts"][0]
        assert "Result on screen" not in system

    def test_the_catalogue_stands_when_the_analyst_hands_over(self, model):
        model["reply"] = dispatcher._PROCEED_TO_QUERY
        reply, adapter = self._reply("this is wrong")
        assert reply.startswith(_en("reply.frustration.lead"))
        assert adapter.get_analyst_history() == []

    def test_the_catalogue_stands_when_the_model_fails(self, monkeypatch, model):
        async def _boom(*a, **k):
            raise RuntimeError("provider down")

        monkeypatch.setattr(dispatcher, "llm_complete", _boom)
        reply, _adapter_ = self._reply("this is wrong")
        assert reply.startswith(_en("reply.frustration.lead"))

    def test_a_french_complaint_falls_back_in_french(self, monkeypatch, model):
        from core.i18n import activate_language, deactivate_language, t

        async def _boom(*a, **k):
            raise RuntimeError("provider down")

        monkeypatch.setattr(dispatcher, "llm_complete", _boom)
        token = activate_language("fr")
        try:
            reply, _adapter_ = self._reply("c'est faux")
        finally:
            deactivate_language(token)
        assert reply.startswith(t("reply.frustration.lead", "fr"))


class TestTheFrontDoorSendsIt:
    """dispatch() itself, over a real WebAdapter and a READY workspace in the
    temporary store: the complaint reaches the analyst and its reply is what
    is sent."""

    def _drive(self, text, model):
        account_id = f"acct{os.urandom(4).hex()}"
        store.upsert_client(account_id, "Test Ltd")
        store.update_client_state(account_id, "READY", state_data={})
        from core.result_cache import result_cache
        from gateway.web_adapter import WebAdapter

        adapter = WebAdapter(AsyncMock(), account_id, "7", thread_id="t1")
        sent = []

        async def _send_message(_event, content, *a, **k):
            sent.append(content)

        adapter.send_message = _send_message
        result_cache.store(adapter.session_id, list(ROWS), question=QUESTION, sql="SELECT 1")
        try:
            event = adapter.make_event(text)
            asyncio.run(dispatcher.dispatch(account_id, event, adapter, MagicMock(),
                                            {"id": 7, "name": "Ada", "role": "viewer", "group_name": ""}))
        finally:
            result_cache.clear(adapter.session_id)
        return sent

    def test_a_complaint_is_answered_by_the_analyst(self, model):
        sent = self._drive("this is wrong", model)
        assert sent == [model["reply"]], sent
        system, _user = model["prompts"][0]
        assert f"Result on screen (it answers: {QUESTION}):" in system

    def test_thanks_is_still_the_catalogue(self, model):
        sent = self._drive("thanks!", model)
        assert sent == [_en("reply.thanks")], sent
        assert model["prompts"] == []


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
