"""
tests/test_analyst_holds_a_conversation.py

"the conversational part is like the user must be able to converse with the
bot like how LLM reply if a question asked upon a result given"

The conversational analyst could answer one message. It could not hold a
conversation, for two separate reasons.

STATELESS. _generate_analyst_reply took three arguments -- the message, the
account, the client row -- and nothing else. Turn 2 never saw turn 1. "why did
you say that?", "what did you just tell me?", "can you repeat that?" were each
answered from nothing. The session history buffer that does exist records DATA
turns for the SQL prompt (question, sanitised SQL, columns, row count) and has
no idea what the bot replied, so it could not have helped.

ONE BEHAVIOUR. The prompt allowed exactly one non-sentinel response: "reply in
2-4 sentences: what QueryBot can help with in this workspace". So "who built
you?", "tell me a joke" and "why did you say that?" all got a capability blurb,
which is not an answer to any of them.

Both halves are executed here against the real function, with
core.llm.llm_complete replaced at the boundary and the prompt it receives read
back.
"""

import asyncio

import pytest

import core.dispatcher as dispatcher


@pytest.fixture
def analyst(monkeypatch):
    """The real _generate_analyst_reply, with the model at the boundary."""
    import core.llm

    seen = {"prompts": []}
    reply = {"text": "I can help with the sales and margin data in this workspace."}

    async def _complete(system="", user="", *args, **kwargs):
        seen["prompts"].append((str(system), str(user)))
        return reply["text"], 10, 10

    monkeypatch.setattr(core.llm, "llm_complete", _complete)
    monkeypatch.setattr(dispatcher, "llm_complete", _complete)
    monkeypatch.setattr(dispatcher, "resolve_provider",
                        lambda *a, **k: ("test", "test", "k", {}))
    monkeypatch.setattr(dispatcher, "_build_analyst_context",
                        lambda *a, **k: "Metrics: Net Revenue, Margin %")
    seen["reply"] = reply
    return seen


def _ask(text, history=None):
    return asyncio.run(dispatcher._generate_analyst_reply(
        text, "acct", {}, history=history))


class TestTheAnalystRemembersWhatItSaid:

    def test_the_previous_exchange_is_in_the_prompt(self, analyst):
        _ask("why did you say that?", history=[
            {"message": "what can you do?",
             "reply": "I can answer questions about your sales warehouse."},
        ])
        system, _user = analyst["prompts"][0]
        assert "what can you do?" in system
        assert "I can answer questions about your sales warehouse." in system

    def test_both_sides_of_the_exchange_are_labelled(self, analyst):
        """A transcript with no speakers is not a transcript."""
        _ask("and why?", history=[{"message": "hello", "reply": "Hello there."}])
        system, _user = analyst["prompts"][0]
        assert "Reader: hello" in system
        assert "You: Hello there." in system

    def test_the_turns_are_in_the_order_they_happened(self, analyst):
        _ask("and then?", history=[
            {"message": "first question", "reply": "first answer"},
            {"message": "second question", "reply": "second answer"},
        ])
        system, _user = analyst["prompts"][0]
        assert system.index("first question") < system.index("second question")

    def test_a_long_conversation_is_bounded(self, analyst):
        """Four turns reach the prompt; the fifth-from-last does not."""
        history = [{"message": f"question {i}", "reply": f"answer {i}"}
                   for i in range(8)]
        _ask("and now?", history=history)
        system, _user = analyst["prompts"][0]
        assert "question 7" in system
        assert "question 0" not in system

    def test_no_history_is_not_an_error(self, analyst):
        """Every first turn, and every non-web adapter, has none."""
        assert _ask("what can you do?") == analyst["reply"]["text"]
        system, _user = analyst["prompts"][0]
        assert "Conversation so far" not in system

    def test_a_half_recorded_turn_is_skipped(self, analyst):
        _ask("hello?", history=[
            {"message": "something", "reply": ""},
            {"reply": "orphaned"},
            "not a dict",
        ])
        system, _user = analyst["prompts"][0]
        assert "Conversation so far" not in system


class TestItAnswersWhatWasAsked:

    def test_the_prompt_offers_more_than_a_capability_blurb(self, analyst):
        _ask("who built you?")
        system, _user = analyst["prompts"][0]
        lowered = system.lower()
        assert "answer the message that was actually sent" in lowered
        # The four cases the prompt now distinguishes.
        assert "asked about something you just said" in lowered
        assert "asked about yourself" in lowered
        assert "asked for something you cannot do" in lowered

    def test_it_is_told_not_to_recite_capabilities_unprompted(self, analyst):
        _ask("thanks, that helps")
        system, _user = analyst["prompts"][0]
        assert "did not ask" in system.lower()

    def test_the_anti_invention_rule_survives(self, analyst):
        """The one thing the widening must not loosen."""
        _ask("what tables do you have?")
        system, _user = analyst["prompts"][0]
        assert "Never invent a table, a metric, a number" in system

    def test_it_is_told_to_admit_a_gap_rather_than_fill_it(self, analyst):
        _ask("what did you say three messages ago?", history=[
            {"message": "hi", "reply": "Hello."}])
        system, _user = analyst["prompts"][0]
        assert "say you do not have it" in system


class TestTheSentinelStillWorks:
    """A real data question must still fall through to the SQL pipeline, and
    the sentinel must stay English whatever language the reply is in."""

    def test_the_sentinel_returns_none(self, analyst):
        analyst["reply"]["text"] = dispatcher._PROCEED_TO_QUERY
        assert _ask("revenue by region for last month") is None

    def test_the_sentinel_is_still_demanded_in_english(self, analyst):
        from core import i18n

        token = i18n.activate_language("fr")
        try:
            _ask("tu fais quoi ?")
        finally:
            i18n.deactivate_language(token)
        system, _user = analyst["prompts"][0]
        assert dispatcher._PROCEED_TO_QUERY in system
        assert "that exact token, in " in system
        assert "en français" in system


class TestTheAdapterKeepsTheThread:

    def _adapter(self):
        from unittest.mock import AsyncMock

        from gateway.web_adapter import WebAdapter

        return WebAdapter(AsyncMock(), "acct", "7", thread_id="t1")

    def test_a_turn_is_recorded_as_prose_on_both_sides(self):
        adapter = self._adapter()
        adapter.add_analyst_turn("what can you do?", "I can read your sales data.")
        assert adapter.get_analyst_history() == [
            {"message": "what can you do?", "reply": "I can read your sales data."}]

    def test_it_carries_no_rows_no_sql_and_no_columns(self):
        """The analyst's prompt is metadata-only by contract. Its own replies
        are written from that metadata; nothing else may join them."""
        adapter = self._adapter()
        adapter.add_analyst_turn("hi", "Hello.")
        entry = adapter.get_analyst_history()[0]
        assert set(entry) == {"message", "reply"}

    def test_a_turn_with_nothing_in_it_is_not_recorded(self):
        adapter = self._adapter()
        adapter.add_analyst_turn("", "a reply")
        adapter.add_analyst_turn("a message", "")
        adapter.add_analyst_turn("   ", "   ")
        assert adapter.get_analyst_history() == []

    def test_the_buffer_is_bounded(self):
        adapter = self._adapter()
        for index in range(20):
            adapter.add_analyst_turn(f"message {index}", f"reply {index}")
        history = adapter.get_analyst_history()
        assert len(history) <= 6
        assert history[-1]["message"] == "message 19"

    def test_clearing_the_session_clears_it_too(self):
        """It dies with the socket like the data history, or a reader's
        conversation would outlive their session."""
        adapter = self._adapter()
        adapter.add_analyst_turn("hi", "Hello.")
        adapter.clear_history()
        assert adapter.get_analyst_history() == []

    def test_it_is_a_copy_not_the_live_buffer(self):
        adapter = self._adapter()
        adapter.add_analyst_turn("hi", "Hello.")
        adapter.get_analyst_history().append({"message": "x", "reply": "y"})
        assert len(adapter.get_analyst_history()) == 1


class TestTheDispatcherWiresBothEnds:
    """The two ends of the loop: the history reaches the analyst, and the reply
    is written back. A buffer nothing reads and a reader nothing feeds are the
    same bug in two places."""

    def _drive(self, monkeypatch, messages, reply_text=None):
        """Replay the analyst gate over a real WebAdapter, one turn at a time."""
        import core.llm

        seen = {"prompts": [], "sent": []}
        text = reply_text or "I can read the sales and margin data here."

        async def _complete(system="", user="", *a, **k):
            seen["prompts"].append(str(system))
            return text, 10, 10

        monkeypatch.setattr(core.llm, "llm_complete", _complete)
        monkeypatch.setattr(dispatcher, "llm_complete", _complete)
        monkeypatch.setattr(dispatcher, "resolve_provider",
                            lambda *a, **k: ("test", "test", "k", {}))
        monkeypatch.setattr(dispatcher, "_build_analyst_context",
                            lambda *a, **k: "Metrics: Net Revenue")

        from unittest.mock import AsyncMock

        from gateway.web_adapter import WebAdapter

        adapter = WebAdapter(AsyncMock(), "acct", "7", thread_id="t1")

        async def _send_message(_event, content):
            seen["sent"].append(content)

        adapter.send_message = _send_message

        for message in messages:
            history = adapter.get_analyst_history()
            reply = asyncio.run(dispatcher._generate_analyst_reply(
                message, "acct", {}, history=history))
            if reply is not None:
                asyncio.run(_send_message(None, reply))
                adapter.add_analyst_turn(message, reply)
        return adapter, seen

    def test_turn_two_sees_turn_one(self, monkeypatch):
        adapter, seen = self._drive(
            monkeypatch, ["what can you do?", "why did you say that?"])
        assert len(seen["prompts"]) == 2
        assert "Conversation so far" not in seen["prompts"][0]
        assert "what can you do?" in seen["prompts"][1]
        assert "I can read the sales and margin data here." in seen["prompts"][1]

    def test_the_adapter_holds_both_turns_afterwards(self, monkeypatch):
        adapter, _seen = self._drive(
            monkeypatch, ["what can you do?", "why did you say that?"])
        assert [t["message"] for t in adapter.get_analyst_history()] == [
            "what can you do?", "why did you say that?"]

    def test_a_data_question_records_nothing(self, monkeypatch):
        """The sentinel means "not my turn". Recording it would put a question
        the analyst never answered into the transcript of what it said."""
        adapter, seen = self._drive(
            monkeypatch, ["tell me about the revenue figures"],
            reply_text=dispatcher._PROCEED_TO_QUERY)
        assert adapter.get_analyst_history() == []
        assert seen["sent"] == []


    @staticmethod
    def _gate_source():
        """The function that holds the analyst gate.

        Not dispatch: the gate lives in a nested coroutine that needs a store,
        a socket and a background task to execute, so this is the narrow case
        where reading source is the honest option. Found by NAME rather than
        by line number so a refactor that moves it fails loudly here instead
        of silently passing.
        """
        import ast
        import inspect

        source = inspect.getsource(dispatcher)
        tree = ast.parse(source)
        holders = [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name != "_generate_analyst_reply"
            and "_generate_analyst_reply(" in (ast.get_source_segment(source, node) or "")
        ]
        assert holders, "nothing in core/dispatcher.py calls the analyst"
        # The innermost holder is the gate itself.
        return ast.get_source_segment(source, holders[-1]) or ""

    def test_the_gate_reads_the_buffer(self):
        source = self._gate_source()
        assert "get_analyst_history" in source, source[:400]

    def test_the_gate_writes_the_buffer(self):
        """A CALL, not a mention. The recording used to be an inline
        `if callable(...)` guard, and disabling it left the name in the source
        -- so a source check passed while nothing was ever recorded."""
        import ast

        tree = ast.parse(self._gate_source().lstrip())
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_remember_analyst_turn"
        ]
        assert calls, "the gate never records what it just said"

    def test_recording_a_turn_is_executable_on_its_own(self):
        from unittest.mock import AsyncMock

        from gateway.web_adapter import WebAdapter

        adapter = WebAdapter(AsyncMock(), "acct", "7", thread_id="t1")
        dispatcher._remember_analyst_turn(adapter, "hi", "Hello.")
        assert adapter.get_analyst_history() == [
            {"message": "hi", "reply": "Hello."}]

    def test_an_adapter_that_keeps_no_history_is_fine(self):
        """Slack, Teams and the REST API keep none."""
        class _Bare:
            pass

        dispatcher._remember_analyst_turn(_Bare(), "hi", "Hello.")

    def test_a_failure_to_record_never_reaches_the_reader(self, caplog):
        """The reply is already on the wire when this runs."""
        class _Broken:
            def add_analyst_turn(self, message, reply):
                raise RuntimeError("buffer gone")

        with caplog.at_level("WARNING"):
            dispatcher._remember_analyst_turn(_Broken(), "hi", "Hello.")
        assert any("conversational turn" in r.message for r in caplog.records)

    def test_the_analyst_is_called_with_the_history(self):
        """A buffer nothing reads and a reader nothing feeds are the same bug
        in two places, so both directions are asserted."""
        import ast

        source = self._gate_source()
        tree = ast.parse(source.lstrip())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.id if isinstance(func, ast.Name)
                    else func.attr if isinstance(func, ast.Attribute) else "")
            if name == "_generate_analyst_reply":
                assert any(kw.arg == "history" for kw in node.keywords), \
                    ast.dump(node)
                return
        raise AssertionError("the analyst is not called in the gate")
