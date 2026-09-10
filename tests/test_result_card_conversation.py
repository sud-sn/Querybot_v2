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
    ROWS, _ask_the_card, _reply,
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
        assert after[1].strip(), "the hint promised suggestions and listed none"

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
