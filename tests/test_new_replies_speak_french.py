"""
tests/test_new_replies_speak_french.py

The conversational surfaces added in this branch, in the reader's language.

Three of them are new and none is covered by the existing language suites:

  result_chat_message      the card's conversational reply, and its provenance
                           note
  grounding_caveat         "no supporting breakdown was run", on an analysis
                           card
  the cannot-answer hint   which now actually lists suggestions, so its
                           suggestions are now visible enough to be wrong

The prose the model writes is governed by a prompt rule, not the catalogue, so
it is asserted where it belongs: in the prompt. Everything the SERVER writes
comes from the catalogue and is asserted on the wire.
"""

import json
import os
import sys
import tempfile

import pytest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_new_replies_fr.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for _mod in list(sys.modules):
    if _mod.startswith("store"):
        del sys.modules[_mod]

import store  # noqa: E402

store.init_db()

ROWS = [
    {"REGION": "North", "NET_AMOUNT": 900.0},
    {"REGION": "South", "NET_AMOUNT": 240.0},
    {"REGION": "East", "NET_AMOUNT": 110.0},
]


def _client_app():
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from gateway import webhooks

    app = FastAPI()
    app.include_router(webhooks.router)
    return TestClient(app)


def _reader(lang):
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    store.update_client_meta(account_id, chat_ui_enabled=1)
    user_id, _ = store.create_user(account_id, "Ada", f"{os.urandom(4).hex()}@x.com")
    store.set_user_language(user_id, lang)
    return account_id, user_id


def _drain(ws, limit=25, timeout=5.0):
    import anyio

    out = []
    for _ in range(limit):
        async def _recv():
            with anyio.fail_after(timeout):
                return await ws._send_rx.receive()

        try:
            message = ws.portal.call(_recv)
        except Exception:
            break
        if message.get("type") != "websocket.send" or "text" not in message:
            break
        frame = json.loads(message["text"])
        out.append(frame)
        if frame.get("type") in {
            "result_chat_message", "result_chat_error", "result_chat_response",
        }:
            break
    return out


def _ask_the_card(lang, typed, *, model_replies):
    import core.compliance.policy_engine as pe
    import core.llm
    import gateway.webhooks as wh
    import portal.routes as pr
    from core.result_cache import result_cache

    seen = {"prompts": []}

    async def _complete(system="", user="", *a, **k):
        seen["prompts"].append((str(system), str(user)))
        if "ANSWER THE READER" in str(system):
            return (model_replies, 10, 10) if model_replies else ("", 1, 1)
        return "CANNOT_GENERATE", 1, 1

    client = _client_app()
    account_id, user_id = _reader(lang)
    session_id = f"{account_id}:web_{user_id}:thread:t1"
    result_cache.store(
        session_id, list(ROWS), question="revenue by region",
        sql="SELECT REGION, SUM(NET_AMOUNT) AS NET_AMOUNT FROM DBO.F_SALES GROUP BY REGION",
        metadata={"account_id": account_id, "user_id": str(user_id)},
    )
    originals = (wh.resolve_provider, wh.llm_complete,
                 pe.result_llm_features_allowed, core.llm.llm_complete)
    wh.resolve_provider = lambda *a, **k: ("test", "test", "k", {})
    wh.llm_complete = _complete
    core.llm.llm_complete = _complete
    pe.result_llm_features_allowed = lambda _a: True
    try:
        client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
        with client.websocket_connect(f"/ws/chat/{account_id}?thread_id=t1") as ws:
            ws.receive_json()
            ws.send_json({"type": "result_chat", "question": typed, "result_id": ""})
            frames = _drain(ws)
    finally:
        (wh.resolve_provider, wh.llm_complete,
         pe.result_llm_features_allowed, core.llm.llm_complete) = originals
        result_cache.clear(session_id)
    reply = next((f for f in frames if f.get("type") in {
        "result_chat_message", "result_chat_error", "result_chat_response"}), {})
    return reply, seen


class TestTheConversationalReply:

    def test_the_provenance_note_is_french(self):
        reply, _seen = _ask_the_card(
            "fr", "quel est le total ?",
            model_replies="Les trois régions totalisent 1 250.")
        assert reply["type"] == "result_chat_message", reply
        assert "résumé statistique" in reply["source_note"], reply["source_note"]

    def test_the_english_note_is_english(self):
        reply, _seen = _ask_the_card(
            "en", "what is the total?",
            model_replies="The three regions total 1,250.")
        assert "statistical summary" in reply["source_note"]

    def test_a_french_reader_asks_the_model_for_french(self):
        """The reply itself is prose the model writes, so the language lives in
        the prompt and nowhere else."""
        _reply, seen = _ask_the_card(
            "fr", "quel est le total ?", model_replies="peu importe")
        conversational = [s for s, _u in seen["prompts"] if "ANSWER THE READER" in s]
        assert conversational, "the conversational path never ran"
        assert "en français" in conversational[0]

    def test_an_english_reader_is_told_nothing_extra(self):
        _reply, seen = _ask_the_card(
            "en", "what is the total?", model_replies="never mind")
        conversational = [s for s, _u in seen["prompts"] if "ANSWER THE READER" in s]
        assert conversational and "en français" not in conversational[0]


class TestTheHintWhenNothingCanAnswer:

    def test_the_suggestions_are_french(self):
        """The hint lists questions to type back, so they have to be questions
        a French reader can type -- and the normaliser has to be able to turn
        them back into English for the planner."""
        reply, _seen = _ask_the_card("fr", "pourquoi donc ?", model_replies="")
        assert reply["type"] == "result_chat_error", reply
        content = reply["content"]
        assert "Questions you can ask" not in content, content
        assert "poser" in content or "demander" in content, content

    def test_every_suggestion_survives_canonicalisation(self):
        from core.question_normalizer import canonical_question
        from core.result_commands import parse_result_command

        reply, _seen = _ask_the_card("fr", "pourquoi donc ?", model_replies="")
        suggestions = [
            line.strip(" •-'’")
            for line in reply["content"].splitlines()
            if line.strip().startswith(("•", "-", "'"))
        ]
        assert suggestions, reply["content"]
        for suggestion in suggestions:
            english = canonical_question(suggestion, "fr")
            assert parse_result_command(english) is not None \
                or any(word in english.lower()
                       for word in ("average", "above", "rank", "filter")), \
                (suggestion, english)
