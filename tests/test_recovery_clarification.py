"""
tests/test_recovery_clarification.py

F1 — when the repair ladder runs out, ask instead of giving up.

The pipeline recovers well and then stops. At the end of the ladder the turn
ends with a failure card: accurate, well-written, and the end of the
conversation. Often the product already knows what to ask — a question that
dead-ends on `unknown_column` has usually named something close to a real
metric, and `suggest_closest_terms` has already worked out which. Those
suggestions were printed as "did you mean" prose and thrown away.

Most of this file is about the three rules that keep it from becoming a
machine that interrogates people: only where an answer can change the outcome,
never an invented option, and one question inside the budget that already
exists.

The last class is the one that matters most, and it is not about the decision
at all: it drives a real reply through the real dispatcher helpers, because a
question nobody can answer is worse than no question.
"""

from __future__ import annotations

import pytest

from core.clarification import combine_with_clarification, resolve_option_text
from core.i18n import MESSAGES
from core.recovery_clarification import (
    CLARIFIABLE_CODES,
    MAX_OPTIONS,
    MIN_OPTIONS,
    SOURCE,
    build,
    is_clarifiable,
)

CLOSE_TERMS = ["purchase order quantity", "order line quantity",
               "customer order quantity"]


def asked(**kwargs):
    base = {"code": "unknown_column", "question": "customer ordered qty",
            "suggestions": CLOSE_TERMS}
    base.update(kwargs)
    return build(**base)


class TestOnlyWhereAnAnswerCanChangeTheOutcome:

    @pytest.mark.parametrize("code", sorted(CLARIFIABLE_CODES))
    def test_a_failure_a_reader_can_resolve_is_asked_about(self, code):
        assert asked(code=code).should_ask, code

    @pytest.mark.parametrize("code", [
        "database_unavailable", "execution_timeout", "access_denied",
        "policy_denied", "row_limit_exceeded", "", None,
    ])
    def test_a_failure_a_reader_cannot_resolve_is_not(self, code):
        # Asking "which did you mean?" about a database outage implies the
        # outage was the reader's phrasing, and a product that asks a question
        # it cannot use the answer to is worse than one that says plainly that
        # it failed.
        decision = asked(code=code)
        assert not decision.should_ask, code
        assert "resolve" in decision.reason

    def test_the_code_check_is_case_insensitive(self):
        assert is_clarifiable("UNKNOWN_COLUMN")
        assert is_clarifiable("  Unknown_Column  ")
        assert not is_clarifiable("unknown_columns")


class TestNoOptionIsEverInvented:

    def test_the_options_are_exactly_the_suggestions(self):
        assert list(asked().options) == CLOSE_TERMS

    def test_no_suggestions_means_no_question(self):
        for empty in ([], None, ["", "   "]):
            decision = asked(suggestions=empty)
            assert not decision.should_ask, empty
            assert "no close vocabulary" in decision.reason

    def test_one_suggestion_is_not_a_choice(self):
        # A single suggestion is a correction the product should be making
        # itself, not a question to put to somebody.
        assert MIN_OPTIONS == 2
        assert not asked(suggestions=["purchase order quantity"]).should_ask

    def test_the_menu_is_bounded(self):
        decision = asked(suggestions=[f"term number {i}" for i in range(12)])
        assert len(decision.options) == MAX_OPTIONS

    def test_duplicates_are_collapsed_case_insensitively(self):
        decision = asked(suggestions=["Gross Margin", "gross margin", "net margin"])
        assert decision.options == ("Gross Margin", "net margin")

    def test_a_sentence_is_not_an_option(self):
        long_one = "a phrase far too long to be a column name " * 3
        decision = asked(suggestions=[long_one, "gross margin", "net margin"])
        assert long_one not in decision.options
        assert len(decision.options) == 2


class TestOneQuestionInsideTheExistingBudget:

    def test_it_asks_under_its_own_source(self):
        # The round cap and the already-asked set are keyed on the source, so
        # a recovery question cannot repeat and cannot stack on top of an
        # ambiguity question asked earlier in the same turn.
        assert SOURCE == "recovery"
        assert asked().meta()["source"] == "recovery"

    def test_a_spent_budget_asks_nothing(self):
        decision = asked(allowed=False)
        assert not decision.should_ask
        assert "budget" in decision.reason

    def test_the_budget_is_checked_before_anything_else(self):
        # Not after building a menu nobody will see.
        assert not asked(allowed=False, suggestions=CLOSE_TERMS).options


class TestWhatTheReaderIsShown:

    def test_the_question_comes_from_the_catalogue_in_both_languages(self):
        entry = MESSAGES["recovery.clarify.question"]
        assert entry["en"] and entry["fr"]
        assert entry["en"] != entry["fr"]
        assert asked(lang="fr").question == entry["fr"]
        assert asked(lang="en").question == entry["en"]

    def test_it_does_not_blame_the_reader(self):
        # They picked words the product does not know, which is a gap in the
        # model. A sentence that says so without apologising is the one that
        # gets an answer back.
        text = asked().question.casefold()
        assert "sorry" not in text
        assert "invalid" not in text

    def test_the_metadata_carries_what_the_prompt_needs(self):
        meta = asked().meta()
        assert meta["options"] == [{"label": term, "value": term}
                                   for term in CLOSE_TERMS]
        assert meta["failure_code"] == "unknown_column"
        assert meta["question"]


class TestTheReplyActuallyResumesTheTurn:
    """
    A question nobody can answer is worse than no question.

    These drive the real dispatcher helpers over the real metadata this module
    produces, rather than asserting the metadata looks plausible.
    """

    def _meta(self):
        return asked().meta()

    @pytest.mark.parametrize("typed", [
        "purchase order quantity",      # exactly the option
        "PURCHASE ORDER QUANTITY",      # shouted
        "purchase order",               # a prefix of it
    ])
    def test_a_reply_resolves_to_the_option_it_names(self, typed):
        match = resolve_option_text(self._meta()["options"], typed)
        assert match, typed
        assert match["value"] == "purchase order quantity"

    def test_an_unrelated_reply_does_not_resolve(self):
        assert not resolve_option_text(self._meta()["options"], "something else")

    def test_the_choice_is_combined_with_the_original_question(self):
        combined, _ = combine_with_clarification(
            "customer ordered qty", "purchase order quantity", self._meta())
        # Both halves survive: the original names what was asked and the
        # choice names the vocabulary the workspace holds. Replacing the
        # original with the choice would answer a question nobody asked.
        assert "customer ordered qty" in combined
        assert "purchase order quantity" in combined

    def test_the_choice_is_marked_as_a_clarification_not_a_new_question(self):
        combined, _ = combine_with_clarification(
            "customer ordered qty", "purchase order quantity", self._meta())
        assert "do not treat it as a separate question" in combined

    def test_this_module_does_not_rewrite_the_question_itself(self):
        # The dispatcher already owns that, and a second rewriter here would
        # be a second answer to a question already answered.
        import core.recovery_clarification as module

        assert not hasattr(module, "rewritten_question")


class TestItIsWiredIntoTheTerminalFailure:
    """
    Placement inside _handle_query_impl, which is 6,400 lines and is exactly
    where an insertion lands in the wrong body — this branch has shipped a fix
    that sat between an `except` and a `return`. Everything above executes;
    this checks the one thing execution cannot reach.

    Deliberately no test that the question REUSES the suggestions already
    computed for the failure card. Recomputing them would cost a second scan
    of the glossary on a turn that has already failed, but it would produce
    the same options and the same question, so nothing a test can observe
    changes — and the only assertion available was a substring that the
    translate_failure call three lines below satisfies just as well. A test
    that cannot discriminate is worse than none: it reports coverage it does
    not have.
    """

    def _source(self):
        import inspect

        import core.query_pipeline as pipeline
        return inspect.getsource(pipeline._handle_query_impl)

    def test_it_is_asked_before_the_failure_card_is_sent(self):
        source = self._source()
        ask = source.index("if _recovery_q.should_ask:")
        card = source.index("format_failure_business_response")
        assert ask < card, "the failure card is built before the question is asked"

    def test_it_goes_through_the_shared_budget(self):
        assert "can_request_clarification(event, _RECOVERY_SOURCE)" in self._source()

    def test_it_ends_the_turn_rather_than_falling_through(self):
        source = self._source()
        block = source[source.index("if _recovery_q.should_ask:"):]
        block = block[:block.index("_rca = translate_failure(")]
        assert "return" in block, "asking and then also sending a failure card"
        assert 'answer_type="clarification"' in block

    def test_the_decision_not_to_ask_is_logged(self):
        # Fail-open silence: a recovery question that never fires and never
        # says why is indistinguishable from one that works.
        assert "Recovery clarification not offered" in self._source()
