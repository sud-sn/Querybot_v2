"""
tests/test_result_card_conversation.py

Talking to a result, rather than commanding it.

The card understood seven operations -- filter, aggregate, sort, limit,
exclude, chart, compare -- and that was the whole of what it could hear.
Everything else got the same reply:

    I could not answer that. The current result only has these columns:
    `REGION`, `NET_AMOUNT`. Questions you can ask:

...followed, because `_rc_stats` was declared empty and never filled, by
nothing at all. So the message whose entire purpose was to show a reader a
question that works showed them an empty list.

The four questions below are the ordinary ones a person asks a table:

    "what is the total?"          a number that is on the screen
    "why is that?"                an interpretation of what is on the screen
    "what does this column mean?" a definition
    "thanks"                      not a question

Three of the four need no query at all, and all four were refused identically.
They are answered now, LAST -- only once the governed cache engine and the
production-database fallback have both declined -- so a question a query can
answer is still answered with data.

Everything here drives the real socket. The model is replaced at
core.llm.llm_complete, which is the boundary; nothing between the frame the
browser sends and the frame it gets back is stubbed.
"""

import os
import sys

import pytest

# The socket harness, the tenant fixture and the frame reader are shared with
# the sibling file rather than rebuilt, so both suites drive the same path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_result_chat_conversation import (  # noqa: E402
    _REPLY_TYPES, ROWS, _ask_the_card, _client_app, _reply,
)

CONVERSATIONAL = [
    "what is the total?",
    "why is that?",
    "what does NET_AMOUNT mean?",
    "thanks!",
]


@pytest.fixture
def model(monkeypatch):
    """The model, and a record of every prompt it was sent.

    The card's own SQL fallback runs first and asks the model to write a
    query; it is answered CANNOT_GENERATE so the conversational path is
    reached, which is exactly the sequence production follows for a question
    no query can answer.
    """
    import core.compliance.policy_engine as pe
    import core.llm
    import gateway.webhooks as wh

    seen = {"prompts": [], "conversational": []}
    reply = {"text": "The three regions total 970 — North alone is 620 of that."}

    async def _complete(system="", user="", *args, **kwargs):
        seen["prompts"].append((str(system), str(user)))
        if "ANSWER THE READER" in str(system):
            seen["conversational"].append((str(system), str(user)))
            return reply["text"], 10, 10
        if "NO_DRILLDOWN" in str(system):
            return "NO_DRILLDOWN", 1, 1
        return "CANNOT_GENERATE", 1, 1

    monkeypatch.setattr(core.llm, "llm_complete", _complete)
    monkeypatch.setattr(wh, "llm_complete", _complete)
    monkeypatch.setattr(pe, "result_llm_features_allowed", lambda _a: True)
    seen["reply"] = reply
    return seen


class TestTheCardAnswersAQuestionItCannotQuery:

    @pytest.mark.parametrize("typed", CONVERSATIONAL)
    def test_it_replies_instead_of_refusing(self, model, typed):
        frame = _reply(_ask_the_card(typed))
        assert frame.get("type") == "result_chat_message", frame
        assert frame["content"] == model["reply"]["text"]

    def test_the_reply_says_where_it_came_from(self, model):
        """The governed cache engine can claim no result values reached the
        model. This path cannot -- the brief carries labels and figures -- so
        it carries its own note rather than borrowing the stronger one."""
        frame = _reply(_ask_the_card("what is the total?"))
        note = frame.get("source_note") or ""
        assert "statistical summary" in note, note
        assert "not the rows" in note, note

    def test_the_readers_own_words_are_the_last_thing_the_model_reads(self, model):
        _reply(_ask_the_card("what is the total?"))
        assert model["conversational"], "the conversational path never ran"
        _system, user = model["conversational"][0]
        assert user.rstrip().endswith("They have just typed: what is the total?"), user

    def test_the_model_is_given_the_number_that_was_asked_for(self, model):
        """"What is the total?" is the most ordinary question there is about a
        table, and the brief computed the total to divide by it and then never
        carried it -- so the one figure a reader asks for by name was the one
        figure the model was never shown."""
        _reply(_ask_the_card("what is the total?"))
        _system, user = model["conversational"][0]
        total = sum(row["NET_AMOUNT"] for row in ROWS)
        assert str(total) in user or str(int(total)) in user, user

    def test_the_model_is_told_what_the_result_answers(self, model):
        _reply(_ask_the_card("why is that?"))
        _system, user = model["conversational"][0]
        assert "The result on screen answers: revenue by region" in user, user

    def test_it_is_never_handed_the_rows(self, model):
        """Same boundary as every other narrative in the product: a
        statistical brief, never the row set.

        A brief does name real values -- the leader, the top five, the tail --
        so "no values reach the model" would be a claim this path cannot make
        and does not make. What it can be held to is that the brief is a
        SUMMARY: a result of twenty rows must not arrive as twenty rows. The
        row below is rank ten of twenty, outside both the top five and the
        bottom three, and it is the whole result set's worth of rows that its
        absence stands for.
        """
        rows = [{"REGION": f"R{i:02d}", "NET_AMOUNT": float(1000 - i * 7)}
                for i in range(20)]
        _reply(_ask_the_card("what is the total?", rows=rows))
        _system, user = model["conversational"][0]
        assert "R09" not in user, user
        assert "937" not in user, user            # its NET_AMOUNT
        assert "R19" not in user, user            # nor the tail
        # The leader and the runner-up are named, so this is not passing
        # because the brief arrived empty.
        assert "R00" in user and "R01" in user, user


class TestTheReplyIsInTheReadersLanguage:
    """Model-written prose, so the language rule has to be in the prompt --
    the catalogue cannot reach it."""

    def test_a_french_reader_gets_a_french_instruction(self):
        from core import i18n
        from core.insight import (
            build_action_contract, build_insight_prompt_from_contract,
            compute_data_brief,
        )

        brief = compute_data_brief(list(ROWS), "revenue by region")
        contract = build_action_contract(
            "converse", "revenue by region", brief,
            follow_up="quel est le total ?")
        token = i18n.activate_language("fr")
        try:
            system, _user = build_insight_prompt_from_contract(
                contract, follow_up="quel est le total ?")
        finally:
            i18n.deactivate_language(token)
        assert "en français" in system, system[:400]

    def test_an_english_reader_is_told_nothing_extra(self):
        """English is the empty rule on purpose."""
        from core.insight import (
            build_action_contract, build_insight_prompt_from_contract,
            compute_data_brief,
        )

        brief = compute_data_brief(list(ROWS), "revenue by region")
        contract = build_action_contract("converse", "revenue by region", brief)
        system, _user = build_insight_prompt_from_contract(contract)
        assert "en français" not in system


class TestACommandIsStillACommand:
    """The conversational reply runs last, and only when nothing else could
    answer. A question the cache engine handles must never reach a model."""

    @pytest.mark.parametrize("typed", ["keep the top 2", "exclude East"])
    def test_a_command_answers_from_the_cache_with_no_model_at_all(
            self, model, typed):
        frame = _reply(_ask_the_card(typed))
        assert frame["type"] == "result_chat_response"
        assert model["conversational"] == [], model["conversational"]


class TestWhenTheModelSaysNothing:

    def test_an_empty_reply_falls_back_to_the_hint(self, monkeypatch):
        import core.compliance.policy_engine as pe
        import core.llm
        import gateway.webhooks as wh

        async def _silent(system="", user="", *a, **k):
            return "", 1, 1

        monkeypatch.setattr(core.llm, "llm_complete", _silent)
        monkeypatch.setattr(wh, "llm_complete", _silent)
        monkeypatch.setattr(pe, "result_llm_features_allowed", lambda _a: True)
        frame = _reply(_ask_the_card("what is the total?"))
        assert frame["type"] == "result_chat_error"

    def test_a_regulated_tenant_reaches_no_model_and_still_gets_an_answer(
            self, monkeypatch):
        """The refusal is the correct behaviour, but it must not be the end of
        the reader's turn: they get the deterministic hint they would have got
        before this feature existed."""
        import core.compliance.policy_engine as pe
        import core.llm
        import gateway.webhooks as wh

        calls = []

        async def _complete(system="", user="", *a, **k):
            calls.append(str(system))
            return "CANNOT_GENERATE", 1, 1

        monkeypatch.setattr(core.llm, "llm_complete", _complete)
        monkeypatch.setattr(wh, "llm_complete", _complete)
        monkeypatch.setattr(pe, "result_llm_features_allowed", lambda _a: False)
        frame = _reply(_ask_the_card("what is the total?"))
        assert frame["type"] == "result_chat_error"
        assert not any("ANSWER THE READER" in c for c in calls), calls


class TestTheHintFinallyHasSuggestionsInIt:
    """`_rc_stats` was declared empty and never filled, and the suggestions
    are built from it -- which columns are numeric, which are text, what
    values they hold. So "Questions you can ask:" was followed by nothing."""

    def test_the_list_is_not_empty(self, monkeypatch):
        import core.compliance.policy_engine as pe
        import core.llm
        import gateway.webhooks as wh

        async def _silent(system="", user="", *a, **k):
            return "", 1, 1

        monkeypatch.setattr(core.llm, "llm_complete", _silent)
        monkeypatch.setattr(wh, "llm_complete", _silent)
        monkeypatch.setattr(pe, "result_llm_features_allowed", lambda _a: True)
        content = _reply(_ask_the_card("what is the total?"))["content"]
        after = content.split("Questions you can ask:", 1)
        assert len(after) == 2, content
        # Count BULLETS, not characters. The section ends with a standing
        # "For anything else, ask a fresh question in the main chat." line, so
        # a truthiness check on the remainder passed with zero suggestions --
        # which is exactly the state this test exists to catch.
        bullets = [line for line in after[1].splitlines()
                   if line.strip().startswith(("•", "-"))]
        assert len(bullets) >= 3, (
            "the hint promised suggestions and listed "
            f"{len(bullets)}:\n{after[1]}")

    def test_the_suggestions_use_the_business_name_not_the_column_code(
            self, monkeypatch):
        import core.compliance.policy_engine as pe
        import core.llm
        import gateway.webhooks as wh

        async def _silent(system="", user="", *a, **k):
            return "", 1, 1

        monkeypatch.setattr(core.llm, "llm_complete", _silent)
        monkeypatch.setattr(wh, "llm_complete", _silent)
        monkeypatch.setattr(pe, "result_llm_features_allowed", lambda _a: True)
        content = _reply(_ask_the_card("what is the total?"))["content"]
        suggestions = content.split("Questions you can ask:", 1)[1]
        assert "net amount" in suggestions, suggestions
        assert "net_amount" not in suggestions.lower().replace("net amount", ""), \
            suggestions


class TestTheEgressIsAudited:
    """A path that sends a tenant's figures to a model has to leave a record,
    and a path that REFUSES to has to leave one too.

    Neither did. llm_audit_component narrows an ambient scope and yields None
    without one (core/llm_audit.py:174); record_llm_blocked reads the scope and
    returns silently without one (core/llm_audit.py:401). The call site opened
    no scope, so a successful conversational egress wrote no row and a
    regulated tenant's refusal wrote no row -- the proof pack would report zero
    refusals for a tenant that did refuse.
    """

    def _rows_written(self, monkeypatch, typed, *, allowed, model_reply):
        import core.compliance.policy_engine as pe
        import core.llm
        import gateway.webhooks as wh
        import store as store_mod

        written = []
        original_log = store_mod.log_llm_call

        def _spy(**kwargs):
            written.append(kwargs)
            return original_log(**kwargs)

        async def _complete(system="", user="", *a, **k):
            if "ANSWER THE READER" in str(system):
                return model_reply, 10, 10
            return "CANNOT_GENERATE", 1, 1

        monkeypatch.setattr(store_mod, "log_llm_call", _spy)
        monkeypatch.setattr(core.llm, "llm_complete", _complete)
        monkeypatch.setattr(wh, "llm_complete", _complete)
        monkeypatch.setattr(pe, "result_llm_features_allowed", lambda _a: allowed)
        _reply(_ask_the_card(typed))
        return written

    def test_a_regulated_refusal_is_recorded(self, monkeypatch):
        rows = self._rows_written(
            monkeypatch, "what is the total?", allowed=False, model_reply="x")
        blocked = [r for r in rows
                   if r.get("component") == "result_conversation"]
        assert blocked, [r.get("component") for r in rows]
        assert any(str(r.get("status") or "") == "blocked" for r in blocked), blocked

    def test_the_refusal_names_this_path(self, monkeypatch):
        rows = self._rows_written(
            monkeypatch, "what is the total?", allowed=False, model_reply="x")
        assert any(r.get("component") == "result_conversation" for r in rows), rows

    def test_a_successful_egress_runs_inside_a_scope(self, monkeypatch):
        """The refusal is the easy half. An egress that leaves no trace is the
        one a compliance review cannot see.

        The row itself is written by llm_complete, which is stubbed here -- so
        what is asserted is the thing that was actually missing: that an audit
        scope is AMBIENT when the call is made, and that it names this
        component. Without one, llm_audit_component yields None and the real
        llm_complete writes nothing.
        """
        import core.compliance.policy_engine as pe
        import core.llm
        import core.llm_audit as audit
        import gateway.webhooks as wh

        scopes = []

        async def _complete(system="", user="", *a, **k):
            if "ANSWER THE READER" in str(system):
                scopes.append(dict(audit._AUDIT_SCOPE.get() or {}))
                return "The three regions total 970.", 10, 10
            return "CANNOT_GENERATE", 1, 1

        monkeypatch.setattr(core.llm, "llm_complete", _complete)
        monkeypatch.setattr(wh, "llm_complete", _complete)
        monkeypatch.setattr(pe, "result_llm_features_allowed", lambda _a: True)
        _reply(_ask_the_card("what is the total?"))
        assert scopes, "the conversational path never ran"
        assert scopes[0].get("account_id"), scopes[0]
        assert scopes[0].get("component") == "result_conversation", scopes[0]

    def test_the_refusal_runs_inside_one_too(self, monkeypatch):
        """record_llm_blocked reads the same scope and returns silently
        without it, which is why the refusal wrote nothing."""
        import core.compliance.policy_engine as pe
        import core.llm_audit as audit

        scopes = []
        original = audit.record_llm_blocked

        def _spy(component, reason):
            scopes.append(dict(audit._AUDIT_SCOPE.get() or {}))
            return original(component, reason)

        monkeypatch.setattr(audit, "record_llm_blocked", _spy)
        monkeypatch.setattr(pe, "result_llm_features_allowed", lambda _a: False)
        _reply(_ask_the_card("what is the total?"))
        assert scopes, "the refusal never ran"
        assert scopes[0].get("account_id"), scopes[0]


class TestTheProviderIsToldWhereToGo:
    """resolve_provider returns a fourth value -- azure_endpoint,
    azure_api_version, or the local model's base URL -- and the socket forwards
    it. converse_about_result declared **extra_kwargs and then dropped them, so
    llm_complete raised "Azure OpenAI endpoint not configured", the handler
    swallowed it, and every Azure and local-model workspace fell straight back
    to the "I could not answer that" hint this feature exists to replace."""

    def _kwargs_seen(self, monkeypatch, **provider_kwargs):
        import asyncio

        import core.compliance.policy_engine as pe
        import core.llm

        from core.result_conversation import converse_about_result

        seen = {}

        async def _complete(system="", user="", *args, **kwargs):
            seen.update(kwargs)
            return "an answer", 10, 10

        monkeypatch.setattr(core.llm, "llm_complete", _complete)
        monkeypatch.setattr(pe, "result_llm_features_allowed", lambda _a: True)
        asyncio.run(converse_about_result(
            "what is the total?",
            rows=list(ROWS),
            result_question="revenue by region",
            account_id="acct",
            provider="azure_openai", model="gpt-4o", api_key="k",
            **provider_kwargs,
        ))
        return seen

    def test_the_azure_location_reaches_the_client(self, monkeypatch):
        seen = self._kwargs_seen(
            monkeypatch,
            azure_endpoint="https://tenant.openai.azure.com",
            azure_api_version="2024-02-01")
        assert seen.get("azure_endpoint") == "https://tenant.openai.azure.com"
        assert seen.get("azure_api_version") == "2024-02-01"

    def test_a_local_models_base_url_reaches_it(self, monkeypatch):
        seen = self._kwargs_seen(
            monkeypatch, azure_endpoint="http://127.0.0.1:11434/v1")
        assert seen.get("azure_endpoint") == "http://127.0.0.1:11434/v1"

    def test_an_anthropic_tenant_is_unaffected(self, monkeypatch):
        seen = self._kwargs_seen(monkeypatch)
        assert "azure_endpoint" not in seen


class TestItAnswersAboutTheCardThatWasAsked:
    """A reader can type into the chat panel of an older card that is still on
    screen. adapter.last_result is the NEWEST answer in the thread, not that
    one -- so the model was summarised a different result's values and the
    reader was told the answer came from the one in front of them."""

    def _two_cards_then_ask(self, monkeypatch, typed):
        import core.compliance.policy_engine as pe
        import core.llm
        import gateway.webhooks as wh
        import portal.routes as pr
        import store
        from core.result_cache import result_cache

        seen = {"conversational": []}

        async def _complete(system="", user="", *a, **k):
            if "ANSWER THE READER" in str(system):
                seen["conversational"].append(str(user))
                return "An answer about that card.", 10, 10
            return "CANNOT_GENERATE", 1, 1

        client = _client_app()
        account_id = f"acct{os.urandom(4).hex()}"
        store.upsert_client(account_id, "Test Ltd")
        store.update_client_meta(
            account_id, chat_ui_enabled=1, enable_llm_audit=1)
        user_id, _ = store.create_user(
            account_id, "Ada", f"{os.urandom(4).hex()}@x.com")
        session_id = f"{account_id}:web_{user_id}:thread:t1"
        older = result_cache.store(
            session_id,
            [{"REGION": "North", "NET_AMOUNT": 900.0},
             {"REGION": "South", "NET_AMOUNT": 240.0}],
            question="revenue by region", sql="SELECT 1",
            metadata={"account_id": account_id, "user_id": str(user_id)})
        newer = result_cache.store(
            session_id,
            [{"DEPARTMENT": "Ops", "HEADCOUNT": 7.0},
             {"DEPARTMENT": "Sales", "HEADCOUNT": 3.0}],
            question="headcount by department", sql="SELECT 2",
            metadata={"account_id": account_id, "user_id": str(user_id)})
        originals = (wh.resolve_provider, wh.llm_complete,
                     pe.result_llm_features_allowed, core.llm.llm_complete)
        wh.resolve_provider = lambda *a, **k: ("test", "test", "k", {})
        wh.llm_complete = _complete
        core.llm.llm_complete = _complete
        pe.result_llm_features_allowed = lambda _a: True
        try:
            client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
            with client.websocket_connect(
                    f"/ws/chat/{account_id}?thread_id=t1") as ws:
                ws.receive_json()
                ws.send_json({"type": "result_chat", "question": typed,
                              "result_id": older})
                for _ in range(12):
                    frame = ws.receive_json()
                    if frame.get("type") in _REPLY_TYPES:
                        break
        finally:
            (wh.resolve_provider, wh.llm_complete,
             pe.result_llm_features_allowed, core.llm.llm_complete) = originals
            result_cache.clear(session_id)
        assert newer != older
        return seen

    def test_the_prompt_describes_the_addressed_card(self, monkeypatch):
        seen = self._two_cards_then_ask(monkeypatch, "what is the total?")
        assert seen["conversational"], "the conversational path never ran"
        prompt = seen["conversational"][0]
        assert "revenue by region" in prompt, prompt
        assert "headcount by department" not in prompt, prompt

    def test_it_carries_the_addressed_cards_figures(self, monkeypatch):
        seen = self._two_cards_then_ask(monkeypatch, "what is the total?")
        prompt = seen["conversational"][0]
        assert "900" in prompt, prompt          # the older card's leader
        assert "Ops" not in prompt, prompt      # the newer card's leader
