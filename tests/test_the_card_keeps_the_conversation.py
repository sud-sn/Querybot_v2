# -*- coding: utf-8 -*-
"""Talking to a result card, turn after turn, over the real socket.

Four things about the in-card conversation were true before this change and
are the subject of these tests:

  * "thanks", "merci", "bonjour" and "bye" were routed like questions about
    the result -- the metadata planner, then the production-database
    fallback, then the conversational model: three model calls to answer a
    courtesy, and a planner that sometimes read it as a filter.
  * The conversation's memory was keyed by the id the browser addressed, and
    the browser addresses its next question to the card the last transform
    produced -- so every transform reset the memory to nothing.
  * The memory held the reader's questions and never the answers, so "why
    did you say that?" went to a model that had never seen what it said.
  * A plan the governance layer stopped was explained with the planner's own
    diagnostic ("The metadata planner did not return valid JSON") rather than
    a sentence for the reader, in the reader's language.

Every test sends the frames a browser sends and asserts on the frames that
come back. The model is replaced at its boundary (converse_about_result) by a
fake that records what it was given; nothing between the socket and that
boundary is stubbed.
"""

import os
import sys
import tempfile
from unittest.mock import patch

import pytest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_card_keeps_conversation.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for _mod in list(sys.modules):
    if _mod.startswith("store"):
        del sys.modules[_mod]

import store  # noqa: E402

store.init_db()

ROWS = [
    {"REGION": "North", "NET_AMOUNT": 100.0},
    {"REGION": "South", "NET_AMOUNT": 60.0},
    {"REGION": "East", "NET_AMOUNT": 25.0},
    {"REGION": "West", "NET_AMOUNT": 5.0},
]

_REPLY_TYPES = {
    "result_chat_response",
    "result_chat_error",
    "result_chat_clarification",
    "result_chat_message",
}


def _client_app():
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from gateway import webhooks

    app = FastAPI()
    app.include_router(webhooks.router)
    return TestClient(app)


def _reader(lang=None):
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    store.update_client_meta(account_id, chat_ui_enabled=1, enable_llm_audit=1)
    user_id, _ = store.create_user(account_id, "Ada Lovelace", f"{os.urandom(4).hex()}@x.com", password="a-password-they-chose")
    if lang:
        store.set_user_language(user_id, lang)
    return account_id, user_id


class _Card:
    """One result card on one open socket, asked as many times as needed.

    resolve_provider is replaced at the boundary because it raises without an
    API key; the provider it names does not exist, so the planner's own model
    call fails and the branch falls through exactly as it does for a
    workspace whose planner is unreachable.
    """

    def __init__(self, *, lang=None, rows=None):
        self.lang = lang
        self.rows = list(ROWS if rows is None else rows)

    def __enter__(self):
        import gateway.webhooks as wh
        import portal.routes as pr
        from core.result_cache import result_cache

        self._wh = wh
        self._cache = result_cache
        self.client = _client_app()
        self.account_id, self.user_id = _reader(self.lang)
        self.session_id = f"{self.account_id}:web_{self.user_id}:thread:t1"
        result_cache.store(
            self.session_id, self.rows, question="revenue by region",
            sql="SELECT REGION, SUM(NET_AMOUNT) AS NET_AMOUNT FROM DBO.F_SALES GROUP BY REGION",
            column_formats={"NET_AMOUNT": "currency"},
        )
        self._original_provider = wh.resolve_provider
        wh.resolve_provider = lambda *a, **k: ("test", "test", "k", {})
        self.client.cookies.set(pr._COOKIE, pr._sign_session_value(self.user_id))
        self._socket_cm = self.client.websocket_connect(
            f"/ws/chat/{self.account_id}?thread_id=t1")
        self.ws = self._socket_cm.__enter__()
        self.ws.receive_json()  # greeting / connected line
        return self

    def ask(self, typed, *, result_id=""):
        self.ws.send_json({"type": "result_chat", "question": typed, "result_id": result_id})
        for _ in range(12):
            frame = self.ws.receive_json()
            if frame.get("type") in _REPLY_TYPES:
                return frame
        return {}

    def __exit__(self, *exc):
        try:
            self._socket_cm.__exit__(*exc)
        finally:
            self._wh.resolve_provider = self._original_provider
            self._cache.clear(self.session_id)


class _Analyst:
    """A stand-in for converse_about_result that remembers what it was given."""

    def __init__(self, reply="The total is 190."):
        self.reply = reply
        self.calls = []

    async def __call__(self, question, **kwargs):
        # The socket appends this very turn to the same list after the call
        # returns, so what the model SAW is copied here, at call time.
        seen = {**kwargs, "history": [dict(t) for t in (kwargs.get("history") or [])]}
        self.calls.append({"question": question, **seen})
        return self.reply


def _never(*a, **k):
    raise AssertionError("this path must not be reached for a courtesy")


async def _never_async(*a, **k):
    _never()


def _en(key, **kw):
    from core.i18n import t
    return t(key, "en", **kw)


def _fr(key, **kw):
    from core.i18n import t
    return t(key, "fr", **kw)


class TestACourtesyIsAnsweredByTheCardItself:

    def test_thanks_reaches_neither_the_planner_nor_a_model(self):
        import gateway.webhooks as wh
        with patch.object(wh, "run_governed_result_followup", _never_async), \
                patch.object(wh, "converse_about_result", _never_async), \
                _Card() as card:
            reply = card.ask("thanks!")
        assert reply.get("type") == "result_chat_message", reply
        assert reply["content"] == _en("reply.thanks")

    def test_a_french_reader_is_thanked_in_french(self):
        with _Card(lang="fr") as card:
            reply = card.ask("merci beaucoup")
        assert reply.get("type") == "result_chat_message", reply
        assert reply["content"] == _fr("reply.thanks")
        assert reply["content"] != _en("reply.thanks")

    def test_a_greeting_names_the_reader_and_points_at_the_result(self):
        with _Card() as card:
            reply = card.ask("hello")
        assert reply.get("type") == "result_chat_message", reply
        assert reply["content"].startswith(_en("reply.greeting.hello_named", name="Ada"))
        assert reply["content"].endswith(_en("reply.result_chat.greeting_hint"))

    def test_a_goodbye_in_french_is_answered_in_french(self):
        with _Card(lang="fr") as card:
            reply = card.ask("à bientôt")
        assert reply.get("type") == "result_chat_message", reply
        assert reply["content"] == _fr("reply.goodbye")

    @pytest.mark.parametrize("typed, key", [
        ("au revoir", "reply.goodbye"),
        ("merci pour votre aide", "reply.thanks"),
        ("à la prochaine", "reply.goodbye"),
    ])
    def test_a_courtesy_is_heard_before_the_lexicon_rewrites_it(self, typed, key):
        """The card reads the canonical English of a French question, and the
        lexicon turns "au revoir" into "to revoir" and "merci pour votre aide"
        into "merci for votre aide" -- neither of which any courtesy pattern
        knows. The reader's own words have to be heard first."""
        with _Card(lang="fr") as card:
            reply = card.ask(typed)
        assert reply.get("type") == "result_chat_message", reply
        assert reply["content"] == _fr(key)

    def test_a_question_with_a_courtesy_in_front_is_not_a_courtesy(self):
        """"thanks, now what is the total?" is a question. Whatever the card
        makes of it, it must not be the thank-you reply."""
        import gateway.webhooks as wh
        analyst = _Analyst(reply="190 across the four regions.")
        with patch.object(wh, "converse_about_result", analyst), _Card() as card:
            reply = card.ask("thanks, now what is the total?")
        assert reply.get("type") == "result_chat_message", reply
        assert reply["content"] == "190 across the four regions."
        assert analyst.calls and analyst.calls[0]["question"] == "thanks, now what is the total?"


class TestTheConversationFollowsTheCard:

    def test_the_memory_follows_the_card_a_transform_produced(self):
        """After "keep the top 2" the browser talks to the derived card. The
        next question, addressed to that card, must still see the first."""
        import gateway.webhooks as wh
        analyst = _Analyst()
        with patch.object(wh, "converse_about_result", analyst), _Card() as card:
            first = card.ask("keep the top 2")
            assert first.get("type") == "result_chat_response", first
            derived = first["derived_result_id"]
            assert derived
            second = card.ask("what is the total?", result_id=derived)
        assert second.get("type") == "result_chat_message", second
        assert analyst.calls, "the question never reached the conversational reply"
        remembered = analyst.calls[-1]["history"]
        assert [t.get("question") for t in remembered] == ["keep the top 2"], remembered
        assert remembered[0].get("operation") == "keep_top"

    def test_the_reader_hears_what_was_answered_not_only_what_was_asked(self):
        import gateway.webhooks as wh
        analyst = _Analyst(reply="The total is 190.")
        with patch.object(wh, "converse_about_result", analyst), _Card() as card:
            card.ask("what is the total?")
            card.ask("and the average?")
        assert len(analyst.calls) == 2
        remembered = analyst.calls[-1]["history"]
        assert remembered[-1] == {"question": "what is the total?", "reply": "The total is 190."}

    def test_a_courtesy_is_remembered_with_its_answer(self):
        import gateway.webhooks as wh
        analyst = _Analyst()
        with patch.object(wh, "converse_about_result", analyst), _Card() as card:
            card.ask("thanks")
            card.ask("what is the total?")
        remembered = analyst.calls[-1]["history"]
        assert remembered[-1] == {"question": "thanks", "reply": _en("reply.thanks")}


class TestABlockedPlanIsExplainedForTheReader:

    def _blocked(self, *a, **k):
        from core.governed_result_followup import GovernedFollowupResult
        return GovernedFollowupResult(
            "blocked",
            reason="The metadata planner did not return valid JSON.",
            planner_used=True,
            evidence={"planner_used": True, "literal_binding_count": 1},
        )

    async def _blocked_async(self, *a, **k):
        return self._blocked()

    def test_the_planners_diagnostic_never_reaches_the_reader(self):
        import gateway.webhooks as wh
        with patch.object(wh, "run_governed_result_followup", self._blocked_async), \
                _Card() as card:
            reply = card.ask("why is North so high?")
        assert reply.get("type") == "result_chat_error", reply
        assert reply["content"] == _en("reply.result.unsafe_operation")
        assert "valid JSON" not in reply["content"]
        assert reply["detail"] == _en("reply.result_chat.blocked_detail")

    def test_in_the_readers_language(self):
        import gateway.webhooks as wh
        with patch.object(wh, "run_governed_result_followup", self._blocked_async), \
                _Card(lang="fr") as card:
            reply = card.ask("pourquoi North est-il si élevé ?")
        assert reply["content"] == _fr("reply.result.unsafe_operation")


class TestThePromptCarriesTheAnswers:

    def test_a_turn_with_a_reply_is_repeated_with_it(self):
        from core.insight import _converse_user_message
        message = _converse_user_message(
            {"question": "revenue by region", "grounding": {}},
            follow_up="and the average?", mode="ranking", scope_badge="",
            history=[{"question": "keep the top 2", "operation": "keep_top"},
                     {"question": "what is the total?", "reply": "The total is 190."}],
        )
        assert "they asked: keep the top 2 (ran: keep_top)" in message
        assert "they asked: what is the total? -- you answered: The total is 190." in message
        assert message.rstrip().endswith("They have just typed: and the average?")

    def test_a_long_reply_is_cut_not_repeated_whole(self):
        from core.insight import _converse_user_message
        message = _converse_user_message(
            {"question": "q", "grounding": {}}, follow_up="x", mode="table", scope_badge="",
            history=[{"question": "a", "reply": "y" * 1000}],
        )
        assert "y" * 240 in message and "y" * 241 not in message


class TestTheCourtesyReplies:

    def test_only_the_three_courtesies_get_a_reply(self):
        from core.result_conversation import courtesy_reply_for_card, is_card_courtesy
        assert [k for k in ("greeting", "thanks", "goodbye", "opinion", "vague", None)
                if is_card_courtesy(k)] == ["greeting", "thanks", "goodbye"]
        assert courtesy_reply_for_card("opinion") == ""

    def test_a_reader_without_a_name_is_still_greeted(self):
        from core.result_conversation import courtesy_reply_for_card
        assert courtesy_reply_for_card("greeting", {"name": ""}).startswith(_en("reply.greeting.hello"))


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
