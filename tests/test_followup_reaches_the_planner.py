"""
tests/test_followup_reaches_the_planner.py

"revenue by warehouse", then "just the top 3".

The history buffer is real: it is populated, it is in scope, and it reaches the
SQL prompt with the prior turn's question, columns and SQL. What stood in front
of it were two gates that could not see a turn as a continuation, so most
natural follow-ups were refused before the planner ran and the reader was told

    I understand the analytical request, but I cannot compile a trusted query
    until the semantic layer resolves the business event dataset to analyse,
    the governed measure to calculate...

for "just the top 3" -- because the dataset and the measure were in the
PREVIOUS turn, and the compiler was handed this turn's four words alone.

Two separate misses, and they need different fixes:

  "just the top 3"        A deterministic result command in every respect
                          except its verb. parse_result_command's phrasebook
                          was `keep|show` -- the two words nobody types. The
                          fix is vocabulary.

  "and for last month?"   Not a command at all: it needs a NEW query with the
                          previous turn's structure and a different period.
                          The cached-result gate rejects a short question
                          unless it names something on screen, which is right
                          -- word count cannot tell "Revenue yesterday?" from
                          "drill into North". But an elliptical OPENER is a
                          content signal, not a length one: nobody opens a
                          conversation with "and for last month?". Once the
                          gate opens, machinery that already existed
                          (contextualize_source_query_fallback) merges the
                          root question in, the cache declines to serve a
                          period it does not hold, and the merged question
                          goes to the warehouse.

Every test here executes the real classifier, the real parser and the real
gate. The false-positive half is as important as the rest: a first question
that is short must not be answered from a cache the reader has not built.
"""

import pytest

from core.conversation_state import (
    ConversationState, TurnIntent, classify_turn, looks_elliptical,
)
from core.query_router import (
    should_attempt_cache_followup, should_route_to_result_cache,
)
from core.result_commands import parse_result_command

CACHED_COLUMNS = ["REGION", "NET_AMOUNT"]
CACHED_ROWS = [{"REGION": "North", "NET_AMOUNT": 900.0}]


def _state():
    return ConversationState(
        account_id="acct", session_id="sess", user_id="1", channel="web",
        previous_question="revenue by warehouse", result_id="r1",
    )


# ══════════════════════════════════════════════════════════════════════════════
# "just the top 3" — vocabulary
# ══════════════════════════════════════════════════════════════════════════════

REFINEMENTS = [
    "just the top 3",
    "only the top 3",
    "give me the top 3",
    "limit to the top 5",
    "narrow to the top 5",
    "just the first 5",
    "only the last 3",
    "just the top one",
]


class TestTheWayPeopleActuallyNarrowAResult:

    @pytest.mark.parametrize("typed", REFINEMENTS)
    def test_it_parses_as_a_deterministic_command(self, typed):
        command = parse_result_command(typed)
        assert command is not None, f"{typed!r} did not parse at all"
        assert command.action == "keep_top", command

    @pytest.mark.parametrize("typed", REFINEMENTS)
    def test_it_routes_to_the_cached_result(self, typed):
        assert should_route_to_result_cache(typed, True) is True

    @pytest.mark.parametrize("typed,limit", [
        ("just the top 3", 3), ("only the top 3", 3), ("limit to the top 5", 5),
        ("just the top one", 1), ("only the last 3", 3),
    ])
    def test_it_takes_the_number_the_reader_typed(self, typed, limit):
        assert parse_result_command(typed).limit == limit

    def test_the_ones_that_already_worked_still_do(self):
        for typed in ("keep the top 3", "show only the top 3",
                      "show me the top 2 rows"):
            assert parse_result_command(typed).action == "keep_top", typed

    @pytest.mark.parametrize("typed", ["top 3", "top 3 customers",
                                       "top 10 products by revenue"])
    def test_a_bare_top_n_is_still_a_question(self, typed):
        """The one phrasing equally plausible as a FIRST question. Answering it
        from a cache the reader has not built would be worse than the refusal
        this change removes."""
        assert parse_result_command(typed) is None, typed


# ══════════════════════════════════════════════════════════════════════════════
# "and for last month?" — a fragment is a continuation
# ══════════════════════════════════════════════════════════════════════════════

ELLIPTICAL = [
    "and for last month?",
    "what about last month?",
    "how about by product?",
    "same for last quarter",
    "do the same for Q3",
    "and by product",
    "now by warehouse",
    "also by region",
]

STANDALONE = [
    "revenue yesterday?",
    "profit by warehouse",
    "how many open orders are there",
    "what is the total revenue for last month",
    "sales and margin by region",
    "show me the total revenue for every warehouse last year",
]

# A courtesy also opens with "and" or "now", and the first version of this
# change let every one of them through as a governed result refinement -- so
# saying thank you skipped the conversational analyst and went to the SQL
# pipeline. An opener is not the signal on its own; what is LEFT after it has
# to name a slice.
COURTESIES = [
    "and thanks!",
    "and who are you?",
    "now goodbye",
    "and?",
    "next!",
    "same to you",
    "and you?",
    "ok and what can you do?",
]

# Short, opener-led, and genuinely a NEW question rather than a refinement.
# These are allowed to reach a fresh query; what matters is that they are not
# silently answered from the previous result.
NEW_QUESTIONS_THAT_LOOK_ELLIPTICAL = [
    "also unpaid invoices?",
    "and margin?",
    "now show me customers",
]


class TestAFragmentIsRecognisedAsAContinuation:

    @pytest.mark.parametrize("typed", ELLIPTICAL)
    def test_it_is_elliptical(self, typed):
        assert looks_elliptical(typed) is True

    @pytest.mark.parametrize("typed", STANDALONE)
    def test_a_whole_question_is_not(self, typed):
        assert looks_elliptical(typed) is False

    def test_an_opener_followed_by_a_whole_question_is_a_whole_question(self):
        """The opener alone is not the signal -- length is the other half."""
        assert looks_elliptical(
            "and what is the total revenue by region for the last year") is False

    def test_a_conjunction_inside_a_sentence_is_not_an_opener(self):
        assert looks_elliptical("sales and margin by region") is False

    @pytest.mark.parametrize("typed", COURTESIES)
    def test_a_courtesy_is_not_a_refinement(self, typed):
        """An opener with nothing after it that names a slice."""
        assert looks_elliptical(typed) is False

    @pytest.mark.parametrize("typed", NEW_QUESTIONS_THAT_LOOK_ELLIPTICAL)
    def test_a_new_measure_or_entity_is_not_a_refinement(self, typed):
        """"and margin?" asks for a different measure, which needs a fresh
        query. Treating it as a refinement of the result on screen would
        transform rows that do not contain it."""
        assert looks_elliptical(typed) is False

    def test_the_target_has_to_be_in_what_follows_the_opener(self):
        """The remainder is what is searched, not the whole turn -- otherwise
        an opener that IS a target word ("next!") would qualify."""
        assert looks_elliptical("next!") is False
        assert looks_elliptical("next month please") is False
        assert looks_elliptical("and for next month") is True


class TestTheGateOpensForItAndStaysShutForTheRest:

    @pytest.mark.parametrize("typed", ELLIPTICAL)
    def test_a_fragment_reaches_the_planner(self, typed):
        assert should_attempt_cache_followup(
            typed, True,
            cached_col_names=CACHED_COLUMNS, cached_rows=CACHED_ROWS) is True

    @pytest.mark.parametrize("typed", [
        "revenue yesterday?", "profit by warehouse",
        "how many open orders are there",
        "show me the total revenue for every warehouse last year",
    ])
    def test_a_standalone_question_does_not(self, typed):
        """The reasoning the gate was built on, unchanged: word count cannot
        tell "Revenue yesterday?" from "drill into North", so only content
        opens it."""
        assert should_attempt_cache_followup(
            typed, True,
            cached_col_names=CACHED_COLUMNS, cached_rows=CACHED_ROWS) is False

    @pytest.mark.parametrize("typed", ELLIPTICAL)
    def test_nothing_fires_without_a_cached_result(self, typed):
        """A fresh session pays no extra latency, and a first turn that happens
        to start with "and" is not answered from nothing."""
        assert should_attempt_cache_followup(typed, False) is False


class TestTheTurnIsClassifiedAsARefinement:
    """So the turn keeps the result it is continuing from, rather than being
    filed as an unrelated new question."""

    @pytest.mark.parametrize("typed", ELLIPTICAL)
    def test_a_fragment_with_a_live_result_is_a_refinement(self, typed):
        assert classify_turn(
            typed, state=_state(), has_cached_result=True, looks_like_data=True
        ).intent is TurnIntent.QUERY_REFINEMENT

    @pytest.mark.parametrize("typed", STANDALONE)
    def test_a_whole_question_stays_a_new_query(self, typed):
        assert classify_turn(
            typed, state=_state(), has_cached_result=True, looks_like_data=True
        ).intent is TurnIntent.NEW_DATA_QUERY

    @pytest.mark.parametrize("typed", ELLIPTICAL)
    def test_with_no_prior_result_it_is_a_new_query(self, typed):
        assert classify_turn(
            typed, has_cached_result=False, looks_like_data=True
        ).intent is not TurnIntent.QUERY_REFINEMENT

    @pytest.mark.parametrize("typed", ELLIPTICAL)
    def test_a_refinement_skips_the_chit_chat_analyst(self, typed):
        """The gate that used to answer these conversationally instead of
        querying."""
        from core.conversation_state import bypass_analyst_gate

        assert bypass_analyst_gate(classify_turn(
            typed, state=_state(), has_cached_result=True, looks_like_data=True
        )) is True


class TestACourtesyStillReachesTheAnalyst:
    """The regression this predicate had to avoid: saying thank you must not
    become a query."""

    @pytest.mark.parametrize("typed", COURTESIES)
    def test_it_does_not_bypass_the_analyst_gate(self, typed):
        from core.conversation_state import bypass_analyst_gate

        decision = classify_turn(
            typed, state=_state(), has_cached_result=True, looks_like_data=False)
        assert bypass_analyst_gate(decision) is False, decision

    @pytest.mark.parametrize("typed", COURTESIES)
    def test_it_does_not_cost_a_planner_call(self, typed):
        assert should_attempt_cache_followup(
            typed, True,
            cached_col_names=CACHED_COLUMNS, cached_rows=CACHED_ROWS) is False


class TestTheMergedQuestionCarriesBothTurns:
    """What the fresh query is actually asked, once the cache declines to serve
    a period it does not hold."""

    @pytest.mark.parametrize("typed", ["and for last month?", "same for Q3"])
    def test_the_parent_question_is_carried_in(self, typed):
        from core.clarification import combine_with_result_context

        merged = combine_with_result_context("revenue by warehouse", typed)
        assert typed.strip() in merged
        assert "revenue by warehouse" in merged

    @pytest.mark.parametrize("typed", ["and for last month?", "same for Q3"])
    def test_the_reader_still_sees_only_their_own_words(self, typed):
        """The parent question is prompt context, not something to show back."""
        from core.clarification import (
            combine_with_result_context, extract_display_question,
        )

        merged = combine_with_result_context("revenue by warehouse", typed)
        assert extract_display_question(merged) == typed.strip()
