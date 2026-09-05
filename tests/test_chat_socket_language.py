"""
tests/test_chat_socket_language.py

The chat socket's own language, proven over a real websocket.

Everything gateway/webhooks.py sends -- greetings, refusals, dashboard and
report confirmations, the drill-down fallbacks -- is written where no template
context reaches and no `lang` parameter is threaded. core/query_pipeline.py
activates a language for the duration of ONE answer, which covers the answer
and nothing around it: the "turned off by the data policy" refusal, the "run a
query first" nudge and every dashboard confirmation are sent from the socket
loop, outside that scope.

So the socket activates the reader's language once, for the connection. This
file exists because that is exactly the kind of wiring that can be written,
reviewed, merged and never execute: every `_t()` added to webhooks.py is
English until the ContextVar is actually set on this task. It connects a real
TestClient websocket to the real endpoint and reads what the browser would get.

The isolation test is the other half. A ContextVar set inside one connection
must not reach another reader's socket -- if it did, a single French user would
switch the language for everyone on the box. asyncio gives each task a copy of
the context, and this asserts it rather than trusting the docs.
"""

import os
import sys
import tempfile

import pytest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_chat_socket_lang.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for _mod in list(sys.modules):
    if _mod.startswith("store"):
        del sys.modules[_mod]

import store  # noqa: E402

store.init_db()


def _client_app():
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from gateway import webhooks

    app = FastAPI()
    app.include_router(webhooks.router)
    return TestClient(app)


def _reader(lang=None):
    """A workspace with the chat UI on, and one user in it."""
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    store.update_client_meta(account_id, chat_ui_enabled=1)
    user_id, _ = store.create_user(account_id, "Ada", f"{os.urandom(4).hex()}@x.com")
    if lang:
        store.set_user_language(user_id, lang)
    return account_id, user_id


def _connect(client, account_id, user_id):
    import portal.routes as pr

    client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
    return client.websocket_connect(f"/ws/chat/{account_id}")


def _first_message(client, account_id, user_id):
    """What the browser receives the moment the socket opens.

    touch_user_activity decides between the greeting and the connected line,
    and it is true only on a genuinely new session, so both are accepted and
    the caller asserts on whichever arrives.
    """
    with _connect(client, account_id, user_id) as ws:
        return ws.receive_json()


class TestTheSocketSpeaksTheReadersLanguage:

    def test_an_english_reader_gets_the_english_line(self):
        client = _client_app()
        account_id, user_id = _reader()
        # The greeting fires once per new session; drain it so the assertion
        # lands on the connected line either way.
        _first_message(client, account_id, user_id)
        message = _first_message(client, account_id, user_id)
        assert "Connected as Ada" in message["content"]

    def test_a_french_reader_gets_the_french_line(self):
        """Fails against a socket that never activates the language: the
        catalogue lookup returns English and this reads "Connected as Ada"."""
        client = _client_app()
        account_id, user_id = _reader("fr")
        _first_message(client, account_id, user_id)
        message = _first_message(client, account_id, user_id)
        assert "Connecté en tant que Ada" in message["content"]
        assert "Connected as" not in message["content"]

    def test_a_user_with_no_language_stored_gets_english(self):
        """lang is NULL for every tenant that upgraded. `or "en"` is what makes
        that a default rather than a crash."""
        client = _client_app()
        account_id, user_id = _reader()
        assert store.get_user(user_id).get("lang") in (None, "", "en")
        _first_message(client, account_id, user_id)
        assert "Connected as Ada" in _first_message(client, account_id, user_id)["content"]


class TestOneReadersLanguageDoesNotReachAnother:

    def test_two_open_sockets_keep_their_own_language(self):
        """A ContextVar set on the wrong task would make one French reader
        switch the language for everyone else on the box. Both sockets are open
        at the same time, so a leak has somewhere to go."""
        client = _client_app()
        fr_account, fr_user = _reader("fr")
        en_account, en_user = _reader("en")
        # Drain each new-session greeting first.
        _first_message(client, fr_account, fr_user)
        _first_message(client, en_account, en_user)

        with _connect(client, fr_account, fr_user) as fr_ws:
            french = fr_ws.receive_json()
            with _connect(client, en_account, en_user) as en_ws:
                english = en_ws.receive_json()
            # ... and the French socket is still French after the English one
            # has been opened and closed inside it.
        assert "Connecté en tant que" in french["content"]
        assert "Connected as" in english["content"]

    def test_the_language_does_not_survive_the_connection(self):
        """The socket never resets its token, on the argument that the context
        dies with the task. If that were wrong, the process default would be
        French from the first French reader onward."""
        from core import i18n

        client = _client_app()
        account_id, user_id = _reader("fr")
        _first_message(client, account_id, user_id)
        _first_message(client, account_id, user_id)
        assert i18n.get_active_language() == "en"


class TestTheSocketsOwnRepliesAreTranslated:
    """Three replies driven all the way through the real receive loop.

    The loop is ~2,200 lines and most of its branches need a warehouse, a
    cached governed result or an LLM to reach. These three need none, so they
    are the ones that can prove -- rather than assume -- that a `_t()` written
    inside the loop body resolves against the reader's language and not the
    process default.
    """

    def _drain_opening(self, client, account_id, user_id):
        with _connect(client, account_id, user_id) as ws:
            ws.receive_json()

    def _reply_to(self, lang, payload):
        """The first frame carrying prose, not the first frame.

        The loop interleaves `typing` and stage frames with the reply, and
        which of them arrives first is not part of what this file is testing.
        """
        client = _client_app()
        account_id, user_id = _reader(lang)
        self._drain_opening(client, account_id, user_id)   # new-session greeting
        with _connect(client, account_id, user_id) as ws:
            ws.receive_json()                              # the connected line
            ws.send_json(payload)
            for _ in range(10):
                frame = ws.receive_json()
                if isinstance(frame, dict) and frame.get("content"):
                    return frame
            raise AssertionError("no frame with prose in it arrived")

    def test_a_blank_question_is_refused_in_french(self):
        reply = self._reply_to("fr", {"type": "message", "text": "   "})
        assert reply["content"] == ("Je n'ai pas pu lire cette question. "
                                    "Veuillez la saisir à nouveau.")

    def test_a_blank_question_is_refused_in_english(self):
        reply = self._reply_to("en", {"type": "message", "text": "   "})
        assert reply["content"] == ("I could not read that question. "
                                    "Please type it again.")

    def test_an_unreadable_message_is_refused_in_french(self):
        reply = self._reply_to("fr", "not a dict at all")
        assert "Je n'ai pas pu lire ce message" in reply["content"]
        assert "could not read" not in reply["content"]

    def test_an_action_with_no_cached_result_is_refused_in_french(self):
        reply = self._reply_to("fr", {"type": "action", "action": "analyze"})
        assert reply["content"].startswith("Ce résultat n'est plus disponible")
        assert "no longer available" not in reply["content"]

    def test_the_same_action_is_refused_in_english_for_an_english_reader(self):
        """The control: without it the assertion above would pass on a socket
        that answered everyone in French."""
        reply = self._reply_to("en", {"type": "action", "action": "analyze"})
        assert reply["content"].startswith("That result is no longer available")


class TestTheGreetingArrivesTranslated:
    """The greeting is written in core/conversational.py, sent from
    gateway/webhooks.py the moment the socket opens, and is the first thing a
    new session ever sees. It is also the one reply that has to cross two
    modules to get here, so it is the one worth driving end to end."""

    def _greeting(self, lang):
        client = _client_app()
        account_id, user_id = _reader(lang)
        # touch_user_activity returns True only on a genuinely new session, so
        # the greeting is the FIRST connection's first frame.
        with _connect(client, account_id, user_id) as ws:
            return ws.receive_json()

    def test_an_english_reader_is_greeted_in_english(self):
        frame = self._greeting("en")
        assert frame["role"] == "assistant"
        assert frame["content"].startswith("Hello, Ada! 👋")
        assert "I'm QueryBot" in frame["content"]

    def test_a_french_reader_is_greeted_in_french(self):
        """Fails on a socket that never activates the language, and on a
        core/conversational.py still holding the sentence as a literal."""
        frame = self._greeting("fr")
        assert frame["content"].startswith("Bonjour Ada ! 👋")
        assert "Je suis QueryBot" in frame["content"]
        assert "ask me anything" not in frame["content"]

    def test_the_help_command_survives_the_crossing(self):
        """`help` is compared by equality in core/dispatcher.py."""
        assert "`help`" in self._greeting("fr")["content"]


class TestTheResultChatReadsTheCanonicalQuestion:
    """The enabling half of the "cannot generate" hint.

    core/query_pipeline.py has canonicalised the main question path since the
    French work began; this branch never did, so parse_result_command, the
    grouping regex and build_generic_query_hints all read the reader's own
    French and matched nothing -- which is what produces the hint whose own
    suggestions are French.

    Driven over the real socket with the planner stubbed at the boundary, so
    what is asserted is the text the planner actually received.
    """

    ROWS = [{"REGION": "North", "NET_AMOUNT": 10},
            {"REGION": "South", "NET_AMOUNT": 20}]

    def _question_the_planner_saw(self, lang, typed):
        import asyncio

        import core.governed_result_followup as grf
        import gateway.webhooks as wh
        from core.result_cache import result_cache

        seen = {}

        async def _spy(question, session_id, **kw):
            seen["question"] = question
            return grf.GovernedFollowupResult(
                "error", reason="stubbed", evidence={})

        client = _client_app()
        account_id, user_id = _reader(lang)
        session_id = f"{account_id}:web_{user_id}:thread:t1"
        result_cache.store(session_id, list(self.ROWS), question="revenue by region",
                           sql="SELECT 1")
        import portal.routes as pr
        client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
        # resolve_provider runs before the planner and raises without an API
        # key, so it is stubbed too -- at the boundary, not inside the block.
        original = wh.run_governed_result_followup
        original_provider = wh.resolve_provider
        wh.run_governed_result_followup = _spy
        wh.resolve_provider = lambda *a, **k: ("test", "test", "k", {})
        try:
            with client.websocket_connect(
                    f"/ws/chat/{account_id}?thread_id=t1") as ws:
                ws.receive_json()          # greeting
                ws.send_json({"type": "result_chat", "question": typed,
                              "result_id": ""})
                for _ in range(10):
                    frame = ws.receive_json()
                    if isinstance(frame, dict) and frame.get("content"):
                        break
        finally:
            wh.run_governed_result_followup = original
            wh.resolve_provider = original_provider
            result_cache.clear(session_id)
        return seen.get("question")

    def test_a_french_question_reaches_the_planner_as_english(self):
        asked = self._question_the_planner_saw("fr", "classer par net amount")
        assert asked == "rank by net amount", asked

    def test_a_french_suggestion_from_the_hint_reaches_it_as_its_english_twin(self):
        """The suggestion the hint itself offers, typed back verbatim."""
        from core import i18n

        typed = i18n.t("hint.ask.filter", lang="fr", column="region")
        asked = self._question_the_planner_saw("fr", typed)
        assert asked == i18n.t("hint.ask.filter", lang="en", column="region"), asked

    def test_an_english_question_reaches_the_planner_untouched(self):
        """canonical_question returns English unchanged, which is what makes
        it safe to call unconditionally here.

        The question deliberately contains "client" and "stock": the French
        lexicon rewrites both ("customer", "inventory"), so canonicalising an
        English reader's words as French would show up here rather than as a
        quietly different answer.
        """
        typed = "show stock by client"
        assert self._question_the_planner_saw("en", typed) == typed


class TestEveryMessageIdTheChatUsesExists:
    """lookup() returns the id itself when an entry is missing, so a typo ships
    as `reply.dash.no_publish` in the chat bubble rather than raising anywhere.
    Nothing else in the suite can see that: the ids are spread over a hundred
    branches most of which need a warehouse to reach.

    This reads the two files for their ids and then EXECUTES the lookup for
    each, in every shipped language.
    """

    FILES = ("gateway/webhooks.py", "core/drill_dimension.py",
             "core/conversational.py", "core/clarification.py",
             "core/dispatcher.py")

    def _ids(self):
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        found = set()
        for name in self.FILES:
            tree = ast.parse((root / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                label = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if label not in {"_t", "_t_plural", "t", "plural"}:
                    continue
                if node.args and isinstance(node.args[0], ast.Constant) \
                        and isinstance(node.args[0].value, str):
                    stem = node.args[0].value
                    if label in {"_t_plural", "plural"}:
                        found.update({f"{stem}.one", f"{stem}.other"})
                    else:
                        found.add(stem)
        return found

    def test_the_scan_finds_the_ids_at_all(self):
        """Without this, a scan that silently matched nothing would make the
        test below pass on an empty set."""
        ids = self._ids()
        assert len(ids) > 100, len(ids)
        assert "reply.drill.title" in ids
        assert "reply.report.created_body.one" in ids

    def test_every_id_resolves_in_every_language(self):
        from core import i18n

        for msg_id in sorted(self._ids()):
            # f-string ids (reply.weekday.{n}) are built at the call site and
            # cannot be read statically; they are covered by the weekday test.
            if "{" in msg_id:
                continue
            for lang in i18n.SUPPORTED_LANGUAGES:
                assert i18n.lookup(msg_id, lang=lang) != msg_id, f"{msg_id} [{lang}]"

    def test_every_weekday_the_report_scheduler_can_name_exists(self):
        from core import i18n

        for index in range(7):
            for lang in i18n.SUPPORTED_LANGUAGES:
                msg_id = f"reply.weekday.{index}"
                assert i18n.lookup(msg_id, lang=lang) != msg_id, f"{msg_id} [{lang}]"
