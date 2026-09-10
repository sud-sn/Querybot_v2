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
