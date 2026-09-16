# -*- coding: utf-8 -*-
"""The conversational analyst answered about a result it could not see.

A reader with a result on their screen who asked the main chat "is that
good?", "what does NET_AMOUNT mean?" or "what did that show?" was routed to
the conversational analyst -- correctly, none of those is a query -- and the
analyst reasoned over workspace metadata alone: business description,
industry, table and metric names. It could not see the result, so it
answered about the workspace in general, or admitted it had nothing.

The analyst is now handed the result on screen: what it answers and its
statistical brief, the same brief the card's conversational reply sends and
never the row set. Because the brief names real values, it reaches the model
only where result_llm_features_allowed says this tenant's results may; a
regulated workspace keeps the analyst metadata-only, exactly as before.

Every test executes the real functions -- the brief builder, the gate, the
prompt builder, and the dispatcher's guard end to end over a real WebAdapter
-- with the model replaced at its boundary and the prompt it received read
back.
"""

from __future__ import annotations

import asyncio

import pytest

import core.dispatcher as dispatcher

ROWS = [
    {"REGION": "North", "NET_AMOUNT": 100.0},
    {"REGION": "South", "NET_AMOUNT": 60.0},
    {"REGION": "East", "NET_AMOUNT": 25.0},
    {"REGION": "West", "NET_AMOUNT": 5.0},
]
QUESTION = "revenue by region"
SQL = "SELECT REGION, SUM(NET_AMOUNT) AS NET_AMOUNT FROM DBO.F_SALES GROUP BY REGION"


@pytest.fixture
def model(monkeypatch):
    """The real analyst, with the model at the boundary and its prompt kept."""
    import core.llm

    seen = {"prompts": [], "reply": "North leads at 100.0, more than half of the 190.0 total."}

    async def _complete(system="", user="", *a, **k):
        seen["prompts"].append((str(system), str(user)))
        return seen["reply"], 10, 10

    monkeypatch.setattr(core.llm, "llm_complete", _complete)
    monkeypatch.setattr(dispatcher, "llm_complete", _complete)
    monkeypatch.setattr(dispatcher, "resolve_provider", lambda *a, **k: ("test", "test", "k", {}))
    monkeypatch.setattr(dispatcher, "_build_analyst_context", lambda *a, **k: "Metrics: Net Revenue")
    return seen


class TestTheBriefOfTheResultOnScreen:

    def test_it_names_the_leader_and_the_total(self):
        from core.insight import describe_result_for_prompt
        text = describe_result_for_prompt(list(ROWS), QUESTION, sql=SQL)
        assert "North" in text, text
        assert "190" in text, text

    def test_it_is_a_summary_not_the_rows(self):
        from core.insight import describe_result_for_prompt
        rows = [{"REGION": f"R{i:02d}", "NET_AMOUNT": float(1000 - i * 7)} for i in range(20)]
        text = describe_result_for_prompt(rows, QUESTION, sql=SQL)
        assert "R00" in text and "R01" in text, text
        assert "R09" not in text and "937" not in text, text

    def test_nothing_for_nothing(self):
        from core.insight import describe_result_for_prompt
        assert describe_result_for_prompt([], QUESTION) == ""
        assert describe_result_for_prompt(None, QUESTION) == ""  # type: ignore[arg-type]


class TestTheGateOnTheReadersScreen:

    def _snapshot(self, rows=None):
        return {"rows": list(ROWS if rows is None else rows), "question": QUESTION, "sql": SQL}

    def test_an_allowed_tenant_gets_the_brief(self, monkeypatch):
        import core.compliance.policy_engine as pe
        monkeypatch.setattr(pe, "result_llm_features_allowed", lambda _a: True)
        question, brief = dispatcher._result_on_screen_for_analyst("acct", self._snapshot())
        assert question == QUESTION
        assert "North" in brief and "190" in brief, brief

    def test_a_regulated_tenant_keeps_the_analyst_metadata_only(self, monkeypatch):
        import core.compliance.policy_engine as pe
        monkeypatch.setattr(pe, "result_llm_features_allowed", lambda _a: False)
        assert dispatcher._result_on_screen_for_analyst("acct", self._snapshot()) == ("", "")

    def test_the_real_policy_fails_closed_for_a_workspace_without_a_profile(self):
        """No monkeypatch: the policy engine itself, for an account that has
        no compliance profile, withholds the result."""
        assert dispatcher._result_on_screen_for_analyst(
            "acct-no-profile-ever", self._snapshot()) == ("", "")

    def test_no_result_no_brief(self, monkeypatch):
        import core.compliance.policy_engine as pe
        monkeypatch.setattr(pe, "result_llm_features_allowed", lambda _a: True)
        assert dispatcher._result_on_screen_for_analyst("acct", None) == ("", "")
        assert dispatcher._result_on_screen_for_analyst("acct", {}) == ("", "")
        assert dispatcher._result_on_screen_for_analyst("acct", self._snapshot(rows=[])) == ("", "")


class TestThePromptCarriesTheResult:

    def _ask(self, text, **kwargs):
        return asyncio.run(dispatcher._generate_analyst_reply(text, "acct", {}, **kwargs))

    def test_the_result_and_the_rule_for_it_are_in_the_prompt(self, model):
        self._ask("is that good?", result_question=QUESTION,
                  result_brief="Mode: ranking\nLeader: North 100.0\nTotal: 190.0")
        system, _user = model["prompts"][0]
        assert f"Result on screen (it answers: {QUESTION}):" in system
        assert "Leader: North 100.0" in system
        assert "Asked about the result on screen" in system
        assert "name the exact question you could run instead" in system

    def test_without_a_result_the_prompt_says_nothing_about_one(self, model):
        self._ask("is that good?")
        system, _user = model["prompts"][0]
        assert "Result on screen" not in system
        assert "Asked about the result on screen" not in system

    def test_the_sentinel_is_still_demanded(self, model):
        self._ask("is that good?", result_question=QUESTION, result_brief="Total: 190.0")
        system, _user = model["prompts"][0]
        assert dispatcher._PROCEED_TO_QUERY in system

    def test_the_reply_comes_back(self, model):
        # Every figure the canned reply asserts is in this brief; a figure the
        # brief does not hold is withheld (tests/test_being_told_the_answer_is_wrong_gets_a_real_reply.py).
        assert self._ask("is that good?", result_question=QUESTION,
                         result_brief="Leader: North 100.0\nTotal: 190.0") == model["reply"]


class TestTheDispatcherLooksAtTheScreen:
    """End to end through _run_query_with_guard, over a real WebAdapter: the
    result the reader has is the result the analyst is told about."""

    def _drive(self, monkeypatch, text, *, with_result: bool):
        from unittest.mock import AsyncMock

        import core.compliance.policy_engine as pe
        from core.result_cache import result_cache
        from gateway.web_adapter import WebAdapter

        monkeypatch.setattr(pe, "result_llm_features_allowed", lambda _a: True)
        adapter = WebAdapter(AsyncMock(), "acct", "7", thread_id="t1")
        sent = []

        async def _send_message(_event, content, *a, **k):
            sent.append(content)

        adapter.send_message = _send_message
        if with_result:
            result_cache.store(adapter.session_id, list(ROWS), question=QUESTION, sql=SQL)
        try:
            event = adapter.make_event(text)
            asyncio.run(dispatcher._run_query_with_guard(
                "acct", event, adapter, text, {"id": 7}, {"enable_llm_audit": 0}))
        finally:
            result_cache.clear(adapter.session_id)
        return sent

    def test_a_reader_with_a_result_is_answered_about_it(self, model, monkeypatch):
        # "is that good?" with a result on screen is classified as a
        # refinement of it and never reaches the analyst; a question about a
        # column's meaning does, and it is the case this exists for.
        sent = self._drive(monkeypatch, "what does NET_AMOUNT mean?", with_result=True)
        assert model["prompts"], "the analyst never ran"
        system, _user = model["prompts"][0]
        assert f"Result on screen (it answers: {QUESTION}):" in system
        assert "North" in system and "190" in system
        assert sent and sent[-1] == model["reply"], sent

    def test_without_a_result_the_analyst_is_told_of_none(self, model, monkeypatch):
        self._drive(monkeypatch, "what does NET_AMOUNT mean?", with_result=False)
        assert model["prompts"], "the analyst never ran"
        system, _user = model["prompts"][0]
        assert "Result on screen" not in system
