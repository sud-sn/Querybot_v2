"""
tests/test_insight_contract_grounding.py

Whatever the model is asked to write about, it is given the numbers to write
it from.

`_build_safe_llm_payload` computes statistics from the result's SHAPE --
ranking gets comparison and distribution stats, a time series gets series
stats, a single cell gets its value. `build_action_contract` then chose which
of those to hand the model by naming keys per ACTION. Shape and action are
picked independently, so any pair whose names did not line up produced a
contract with no numbers in it:

    why      on a time series  -> comparison_stats + distribution_stats,
                                  neither of which a time series has: ZERO
    predict  on a ranking      -> time_series_stats, which a ranking has
                                  not got: ZERO
    compare  on a single value -> ZERO
    explain  on a ranking      -> one number, with the leader share, the
                                  runner-up and the category count discarded
                                  three lines after being computed

`why` on a trend is the one LLM narrative an ordinary question can reach, so
the most common analytical answer in the product was written from the question
wording. These tests start at real rows, run the real brief and the real
prompt builder, and assert the figures are in the text the model receives.

The `explain` case is the one to read twice: it always had a number, so it
looked grounded, and a test that checked "is there a number" would have passed
throughout.
"""

import pytest

from core.insight import (
    build_action_contract,
    build_insight_prompt_from_contract,
    compute_data_brief,
)

RANKING = [
    {"REGION": "North", "NET_AMOUNT": 620.0},
    {"REGION": "South", "NET_AMOUNT": 240.0},
    {"REGION": "East", "NET_AMOUNT": 110.0},
    {"REGION": "West", "NET_AMOUNT": 30.0},
]

TREND = [
    {"MONTH": "2024-01", "NET_AMOUNT": 900.0},
    {"MONTH": "2024-02", "NET_AMOUNT": 820.0},
    {"MONTH": "2024-03", "NET_AMOUNT": 640.0},
    {"MONTH": "2024-04", "NET_AMOUNT": 360.0},
]

SINGLE = [{"NET_AMOUNT": 4820.0}]

ACTIONS = ("explain", "analyze", "compare", "why", "predict", "decide")


def _prompt(rows, question, action):
    """Real rows -> real brief -> real contract -> the text the model sees."""
    brief = compute_data_brief(rows, question)
    contract = build_action_contract(action, question, brief)
    _system, user = build_insight_prompt_from_contract(contract)
    return contract, user


class TestEveryActionGetsTheNumbersTheResultHas:

    @pytest.mark.parametrize("action", ACTIONS)
    def test_a_ranking_carries_its_leader_and_its_spread(self, action):
        contract, user = _prompt(RANKING, "revenue by region", action)
        assert contract.get("comparison_stats"), action
        assert contract.get("distribution_stats"), action
        # The leader's own amount, in the prompt, as a number.
        assert "620" in user, f"{action} lost the leading value:\n{user}"
        assert "North" in user, f"{action} lost the leading label:\n{user}"

    @pytest.mark.parametrize("action", ACTIONS)
    def test_a_time_series_carries_its_direction_and_its_endpoints(self, action):
        contract, user = _prompt(TREND, "revenue by month", action)
        assert contract.get("time_series_stats"), action
        assert "900" in user, f"{action} lost the opening value:\n{user}"
        assert "360" in user, f"{action} lost the closing value:\n{user}"

    @pytest.mark.parametrize("action", ACTIONS)
    def test_a_single_cell_carries_its_one_number(self, action):
        _contract, user = _prompt(SINGLE, "total revenue", action)
        assert "4820" in user, f"{action} lost the only figure there was:\n{user}"


class TestTheOnesThatUsedToSendNothing:
    """Named individually because each was a live failure, not a hypothetical."""

    def test_why_on_a_trend_used_to_send_zero_numbers(self):
        contract, user = _prompt(TREND, "why did revenue drop?", "why")
        assert contract["time_series_stats"]["direction"] == "decreasing"
        assert "decreasing" in user

    def test_predict_on_a_ranking_used_to_send_zero_numbers(self):
        contract, _user = _prompt(RANKING, "what happens next?", "predict")
        assert contract.get("comparison_stats", {}).get("leader") == "North"

    def test_compare_on_a_single_cell_used_to_send_zero_numbers(self):
        contract, _user = _prompt(SINGLE, "how does total revenue compare?", "compare")
        assert contract.get("single_value_stats", {}).get("value") == 4820.0

    def test_explain_on_a_ranking_kept_one_number_and_dropped_the_rest(self):
        """The quiet one: it always had a figure, so it always looked fine."""
        contract, user = _prompt(RANKING, "revenue by region", "explain")
        assert contract["headline_number"] == 620.0        # what it always had
        assert contract["comparison_stats"]["runner_up"] == "South"
        assert contract["distribution_stats"]["category_count"] == 4
        assert "South" in user and "240" in user


class TestTheActionStillDecidesTheTask:
    """The fix must not flatten the actions into one prompt."""

    def test_each_action_states_its_own_task(self):
        tasks = {}
        for action in ACTIONS:
            contract, _ = _prompt(RANKING, "revenue by region", action)
            tasks[action] = contract["task"]
        assert len(set(tasks.values())) == len(ACTIONS), tasks

    def test_only_why_and_decide_offer_safe_next_steps(self):
        offered = {
            action
            for action in ACTIONS
            if "safe_next_steps" in _prompt(RANKING, "revenue by region", action)[0]
        }
        assert offered == {"why", "decide"}


class TestTheTotalIsCarriedAndOnlyWhenItMeansSomething:
    """"What is the total?" is the most ordinary question there is about a
    ranking. compute_data_brief worked the total out in order to divide by it
    -- leader_share_pct and top_3_share_pct are both quotients of it -- and
    then dropped it, so the one figure a reader asks for by name was the one
    figure the model was never shown."""

    def test_a_ranking_carries_its_total(self):
        brief = compute_data_brief(RANKING, "revenue by region")
        assert brief["category_breakdown"]["total"] == 1000.0
        contract = build_action_contract("converse", "revenue by region", brief)
        assert contract["distribution_stats"]["total"] == 1000.0

    def test_the_total_reaches_the_text_the_model_reads(self):
        _contract, user = _prompt(RANKING, "what is the total?", "converse")
        assert "1000" in user, user

    @pytest.mark.parametrize("column", [
        "NET_AMOUNT",      # the classifier knows this one
        "GROSS_MARGIN",    # it does not
        "ORDERS",          # nor this
        "ARPU",            # nor this
        "WIDGETS_SHIPPED", # nor any column a tenant invented
    ])
    def test_a_measure_the_classifier_has_no_opinion_about_still_totals(
            self, column):
        """"unknown" is not a refusal.

        measure_class_for_column returns "unknown" for every column name its
        lexicon has never seen -- which is most of them, GROSS_MARGIN, ORDERS
        and ARPU included. Gating on `== "additive"` treated all of those as
        unsummable and silently dropped the total AND both shares from
        ordinary revenue results: a safeguard that fires on the common case is
        a regression.
        """
        rows = [{"REGION": "North", column: 620.0},
                {"REGION": "South", column: 240.0},
                {"REGION": "East", column: 110.0}]
        breakdown = compute_data_brief(rows, "by region")["category_breakdown"]
        assert breakdown.get("total") == 970.0, breakdown
        assert breakdown.get("leader_share_pct") == 63.9, breakdown
        assert breakdown.get("top_3_share_pct") == 100.0, breakdown

    @pytest.mark.parametrize("column,label", [
        ("MARGIN_PCT", "a percentage"),
        ("STOCK_BALANCE", "a semi-additive balance"),
    ])
    def test_a_measure_that_may_not_be_summed_gets_no_total(self, column, label):
        """collapse_rows_by_label consults this rule, but only when it has rows
        to MERGE -- a result whose labels are already distinct is returned
        untouched "because there is nothing to add". Summing ACROSS those
        distinct labels is a different sum and is subject to the rule again.
        Three regions' margin percentages added together is arithmetic on
        nothing, stated with a number."""
        rows = [{"REGION": "North", column: 12.0},
                {"REGION": "South", column: 9.0},
                {"REGION": "East", column: 7.0}]
        breakdown = compute_data_brief(rows, f"{label} by region")["category_breakdown"]
        assert breakdown.get("total") is None, breakdown

    @pytest.mark.parametrize("column", ["MARGIN_PCT", "STOCK_BALANCE"])
    def test_it_gets_no_shares_either(self, column):
        """The shares are quotients of that same total. "North holds 42.8% of
        the total margin percentage" is a sentence with no meaning, and it was
        being stated with two decimal places."""
        rows = [{"REGION": "North", column: 12.0},
                {"REGION": "South", column: 9.0},
                {"REGION": "East", column: 7.0}]
        breakdown = compute_data_brief(rows, "by region")["category_breakdown"]
        assert breakdown.get("leader_share_pct") is None, breakdown
        assert breakdown.get("top_3_share_pct") is None, breakdown

    def test_the_leader_and_the_ranking_survive(self):
        """Withholding the total must not withhold the ranking: which region
        has the highest margin percentage is a real, answerable question."""
        rows = [{"REGION": "North", "MARGIN_PCT": 12.0},
                {"REGION": "South", "MARGIN_PCT": 9.0},
                {"REGION": "East", "MARGIN_PCT": 7.0}]
        breakdown = compute_data_brief(rows, "margin pct by region")["category_breakdown"]
        assert breakdown["top_5"][0] == {"label": "North", "value": 12.0}
        assert breakdown["category_count"] == 3

    @pytest.mark.parametrize("column", ["MARGIN_PCT", "STOCK_BALANCE"])
    def test_the_gap_survives_too_because_a_difference_needs_no_total(
            self, column):
        """"North's margin is 3 points above South's" is true whatever the
        measure -- it subtracts two of the result's own values and divides by
        nothing. The gap was computed inside the `total > 0` branch, so gating
        the total on additivity took the gap down with it."""
        rows = [{"REGION": "North", column: 12.0},
                {"REGION": "South", column: 9.0},
                {"REGION": "East", column: 7.0}]
        breakdown = compute_data_brief(rows, "by region")["category_breakdown"]
        assert breakdown.get("leader_vs_runner_up_gap") == 3.0, breakdown

    def test_the_gap_is_still_there_for_an_additive_measure(self):
        rows = [{"REGION": "North", "NET_AMOUNT": 620.0},
                {"REGION": "South", "NET_AMOUNT": 240.0}]
        breakdown = compute_data_brief(rows, "by region")["category_breakdown"]
        assert breakdown.get("leader_vs_runner_up_gap") == 380.0


class TestTheConversationalContract:
    """The reader typed a sentence at a result. Every other action is a button
    the product chose; this one is the reader's own words."""

    def test_it_states_the_readers_question_as_the_task(self):
        brief = compute_data_brief(RANKING, "revenue by region")
        contract = build_action_contract(
            "converse", "revenue by region", brief, follow_up="what is the total?")
        assert "follow-up" in contract["task"].lower()
        assert contract["follow_up"] == "what is the total?"

    def test_the_typed_words_come_last_so_they_are_what_gets_answered(self):
        brief = compute_data_brief(RANKING, "revenue by region")
        contract = build_action_contract(
            "converse", "revenue by region", brief, follow_up="what is the total?")
        _system, user = build_insight_prompt_from_contract(
            contract, follow_up="what is the total?")
        assert user.rstrip().endswith("They have just typed: what is the total?")

    def test_it_is_not_given_the_cards_scaffolding(self):
        """A three-bullet analysis is the wrong answer to "thank you"."""
        brief = compute_data_brief(RANKING, "revenue by region")
        contract = build_action_contract(
            "converse", "revenue by region", brief, follow_up="thanks")
        system, _user = build_insight_prompt_from_contract(
            contract, follow_up="thanks")
        assert "HEADLINE: ...".lower() not in system.lower()
        assert "Plain prose" in system

    def test_the_earlier_turns_are_in_the_prompt(self):
        """A conversation is turns, not a sequence of unrelated questions."""
        brief = compute_data_brief(RANKING, "revenue by region")
        contract = build_action_contract(
            "converse", "revenue by region", brief, follow_up="and the second?")
        _system, user = build_insight_prompt_from_contract(
            contract, follow_up="and the second?",
            history=[{"question": "which is highest?", "row_count": 1},
                     {"question": "keep the top 2", "operation": "limit"}])
        assert "which is highest?" in user
        assert "keep the top 2" in user

    def test_a_turns_rows_are_never_replayed_into_the_prompt(self):
        brief = compute_data_brief(RANKING, "revenue by region")
        contract = build_action_contract("converse", "revenue by region", brief)
        _system, user = build_insight_prompt_from_contract(
            contract,
            history=[{"question": "which is highest?",
                      "rows": [{"REGION": "SECRETVALUE"}]}])
        assert "SECRETVALUE" not in user, user


class TestTheShapeOfTheResultReachesThePrompt:
    """build_action_contract set result_shape on every contract, with a comment
    saying every action wants it, and the formatter never rendered it -- so no
    answer could say "across 4 regions" or "over 57 rows"."""

    def test_the_row_count_is_in_the_text_the_model_reads(self):
        _contract, user = _prompt(RANKING, "revenue by region", "explain")
        assert "4 rows" in user, user

    def test_the_column_names_are_in_the_text_the_model_reads(self):
        _contract, user = _prompt(RANKING, "revenue by region", "explain")
        assert "REGION" in user and "NET_AMOUNT" in user
