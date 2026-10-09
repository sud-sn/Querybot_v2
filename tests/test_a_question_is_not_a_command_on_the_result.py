"""A question that only reads like a command on the result on screen is answered as a question.

"How much discount did we give in 2025, and what is it as a percentage of gross amount?" was
read by the display-format parser as "format <in 2025, and what is it> as a percentage" ("give"
is one of its verbs, "as a percentage" one of its formats). On a new thread there was no result
to format, and the command, which never falls through, said "That result is no longer
available. Run the business question again." No engine saw the question.

A sentence that starts by asking is not a display instruction, and "a percentage of X" is a
ratio, never a display format. And in new-core mode, a command with nothing of today's pipeline
on screen (a new thread, or a new-core answer on screen) goes to the new core, whose follow-ups
these are; one that carries a value of the data ("exclude North") still fails closed.

Invented data only.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from core.result_commands import parse_result_command

ASKED = "How much discount did we give in 2025, and what is it as a percentage of gross amount?"


@pytest.mark.parametrize("question", [
    ASKED,
    "How much discount did we give in 2025, and what is it as a percentage?",
    "show discount as a percentage of gross amount",
    "What do we give away in discounts, as % of sales?",
    "Which stores did we show in March as percent of the total?",
])
def test_a_question_is_not_read_as_a_display_command(question):
    assert parse_result_command(question) is None


@pytest.mark.parametrize("command,target,spec", [
    ("show it as a percentage", "it", {"type": "percentage"}),
    ("can you show it as a percentage?", "it", {"type": "percentage"}),
    ("format margin as percent with 1 decimal", "margin", {"type": "percentage", "fraction_digits": 1}),
    ("give me revenue in USD", "revenue", {"type": "currency", "currency_code": "USD"}),
])
def test_a_display_command_is_still_one(command, target, spec):
    parsed = parse_result_command(command)
    assert parsed is not None and parsed.action == "format"
    assert (parsed.target_text, parsed.format_spec) == (target, spec)


def test_two_display_instructions_in_one_sentence_are_still_both_taken():
    parsed = parse_result_command("format month as MMM-YY and revenue as USD currency with no decimals")
    assert parsed is not None and [r["target_text"] for r in parsed.format_spec["batch"]] == ["month", "revenue"]


# ── in new-core mode, on the socket ─────────────────────────────────────────


@pytest.fixture(scope="module")
def tenant(tmp_path_factory):
    from tests import answer_harness as harness

    with harness.tenant_in(tmp_path_factory.mktemp("not_a_command")) as built:
        yield built


def _conversation(tenant, engine: str, question: str) -> tuple[list[dict], list[str]]:
    """``question`` asked first on a new thread, the workspace on ``engine``: the frames sent, and what reached the new core."""
    import gateway.core2_bridge as bridge
    from tests import answer_harness as harness
    from tests import portal_harness as portal

    reached: list[str] = []

    async def setting(account_id):
        return engine

    def answer(account_id, asked, *args, **kwargs):
        reached.append(asked)
        return {"type": "assistant_response", "engine": "core2", "question": asked,
                "answer": {"headline": "Discount in 2025: $1,200, 4.0% of gross amount."},
                "trust": {"engine": "core2", "sql": "", "question_id": kwargs.get("question_id", "")}}

    with patch.object(bridge, "engine", setting), patch("core2.service.portal_answer", answer), \
            patch("core2.service.portal_summary", return_value=None):
        with portal.SocketConversation(tenant, harness.ACCOUNT, harness.saved_connection(),
                                       idle_seconds=2.0) as conversation:
            frames = conversation.ask(question)["frames"]
    return frames, reached


def _refused(frames: list[dict]) -> bool:
    return any("no longer available" in json.dumps(f) for f in frames)


def test_in_new_core_mode_the_question_reaches_the_new_core(tenant):
    frames, reached = _conversation(tenant, "core2", ASKED)
    assert reached == [ASKED] and not _refused(frames)


@pytest.mark.parametrize("follow_up", ["show the result as a pie", "just the top 3", "sort by value",
                                       "show it as a percentage"])
def test_in_new_core_mode_a_follow_up_with_nothing_on_screen_goes_to_the_new_core(tenant, follow_up):
    """After a new-core answer, today's result is set aside: these are the new core's to answer."""
    assert parse_result_command(follow_up) is not None          # each is a command the parser takes
    frames, reached = _conversation(tenant, "core2", follow_up)
    assert reached == [follow_up] and not _refused(frames)


def test_in_new_core_mode_every_command_is_the_new_cores_to_answer(tenant):
    """New-core mode reads no command of today's pipeline first: "exclude North" is a follow-up of the new core's,
    planned under its own rules for member values (put in placeholders where values are kept from the AI)."""
    frames, reached = _conversation(tenant, "core2", "exclude North from this result")
    assert reached == ["exclude North from this result"] and not _refused(frames)


def test_outside_new_core_mode_a_command_with_nothing_on_screen_is_still_refused(tenant):
    frames, reached = _conversation(tenant, "legacy", "just the top 3")
    assert reached == [] and _refused(frames)


@pytest.mark.parametrize("question", [
    "create a dashboard for stock by warehouse",
    "explain your plan for net sales by region",
    "investigate why returns rose in March",
    "find outliers in this result",
])
def test_in_new_core_mode_no_route_of_todays_pipeline_reads_a_question_first(tenant, question):
    frames, reached = _conversation(tenant, "core2", question)
    assert reached == [question], [f.get("type") for f in frames]


def test_in_new_core_mode_a_workspace_at_its_monthly_limit_is_told_so_and_nothing_runs(tenant):
    with patch("core.pipeline_context.check_query_limit", return_value=(False, 500, 500)):
        frames, reached = _conversation(tenant, "core2", ASKED)
    assert reached == []
    assert any("Monthly query limit reached (500/500)" in str(f.get("content")) for f in frames), frames
