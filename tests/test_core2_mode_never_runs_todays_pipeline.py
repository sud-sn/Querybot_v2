"""In new-core mode only the new core answers, even when its answer cannot be sent.

A frame that could not reach the reader (the socket had closed) used to hand the question to today's
pipeline: it queried the warehouse again for a reader who was gone, and wrote another engine's answer into
the thread's history. Every path through new-core mode is handled by the new core -- answered, its own
"I can't answer that" reply, a failure or a timeout, the monthly limit, a "why" about the answer on screen --
whether its frame is sent or not. Side by side ("compare") and today's pipeline alone ("legacy") are
unchanged: today's pipeline answers first, as before.

Through the real portal socket; invented data only.
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest

from tests.test_core2_answers_in_the_portal import CANNED, WHY, _analyses, _answers, _conversation, _NewCore
from tests.test_core2_answers_in_the_portal import tenant  # noqa: F401 - the fixture

QUESTION = "stock on hand by warehouse"


class _Pipeline:
    """Today's pipeline (core.dispatcher.dispatch as the socket calls it): only whether it ran."""

    def __init__(self):
        self.asked: list[str] = []

    async def __call__(self, account_id, event, adapter, bg, portal_user=None):
        self.asked.append(str(getattr(event, "text", "") or ""))


@contextlib.contextmanager
def _sends(*, fail: bool):
    """Every frame the new core sends fails to reach the reader (``fail``), as on a socket that has closed."""
    import gateway.core2_bridge as bridge

    if not fail:
        yield
        return

    async def closed(adapter, websocket, payload):
        return False

    with patch.object(bridge, "_send", closed):
        yield


def _turn(tenant, engine: str, new_core: _NewCore, *, fail: bool, question: str = QUESTION):  # noqa: F811
    pipeline = _Pipeline()
    stack, conversation = _conversation(tenant, engine, new_core)
    with stack, patch("gateway.webhooks.dispatch", pipeline), _sends(fail=fail):
        turn = conversation.ask(question)
    return turn, pipeline


ANSWERED = {QUESTION: CANNED}


@pytest.mark.parametrize("fail", [False, True], ids=["sent", "not sent"])
def test_an_answer_is_the_new_cores_whether_or_not_it_reaches_the_reader(tenant, fail):  # noqa: F811
    turn, pipeline = _turn(tenant, "core2", _NewCore(ANSWERED), fail=fail)
    assert pipeline.asked == [], "today's pipeline ran in the new core's place"
    assert not turn["executed"], "the warehouse was queried again by today's pipeline"
    if not fail:
        assert [a["answer"]["headline"] for a in _answers(turn)] == ["New core says 42."]


@pytest.mark.parametrize("fail", [False, True], ids=["sent", "not sent"])
def test_what_the_new_core_cannot_answer_is_its_own_reply(tenant, fail):  # noqa: F811
    turn, pipeline = _turn(tenant, "core2", _NewCore({}), fail=fail)
    assert pipeline.asked == [] and not turn["executed"]
    if not fail:
        assert _answers(turn) and all(a.get("engine") == "core2" for a in _answers(turn))


@pytest.mark.parametrize("fail", [False, True], ids=["sent", "not sent"])
def test_a_failure_is_said_and_today_s_pipeline_does_not_answer_instead(tenant, fail):  # noqa: F811
    def broken(*args, **kwargs):
        raise RuntimeError("the warehouse went away")

    pipeline = _Pipeline()
    stack, conversation = _conversation(tenant, "core2", _NewCore({}))
    with stack, patch("core2.service.portal_answer", broken), patch("gateway.webhooks.dispatch", pipeline), \
            _sends(fail=fail):
        turn = conversation.ask(QUESTION)
    assert pipeline.asked == [] and not turn["executed"]


@pytest.mark.parametrize("fail", [False, True], ids=["sent", "not sent"])
def test_at_the_monthly_limit_the_reader_is_told_and_nothing_else_runs(tenant, fail):  # noqa: F811
    import gateway.core2_bridge as bridge

    new_core = _NewCore(ANSWERED)
    pipeline = _Pipeline()
    stack, conversation = _conversation(tenant, "core2", new_core)
    with stack, patch.object(bridge, "_over_limit", lambda *a, **k: "This workspace has reached its monthly limit."), \
            patch("gateway.webhooks.dispatch", pipeline), _sends(fail=fail):
        turn = conversation.ask(QUESTION)
    assert pipeline.asked == [] and new_core.asked == [] and not turn["executed"]
    if not fail:
        assert any("monthly limit" in str(f.get("content") or "") for f in turn["frames"])


def test_a_why_about_the_answer_on_screen_stays_the_new_cores_when_it_cannot_be_sent(tenant):  # noqa: F811
    new_core = _NewCore({})
    pipeline = _Pipeline()
    stack, conversation = _conversation(tenant, "core2", new_core)
    with stack, patch("gateway.webhooks.dispatch", pipeline):
        conversation.ask(QUESTION)
        with _sends(fail=True):
            why = conversation.ask(WHY)
    assert new_core.asked == [QUESTION, WHY]
    assert pipeline.asked == [] and not _analyses(why), "today's analysis or pipeline answered the why"


@pytest.mark.parametrize("engine", ["legacy", "compare"])
def test_the_other_modes_still_answer_with_todays_pipeline_first(tenant, engine):  # noqa: F811
    turn, pipeline = _turn(tenant, engine, _NewCore(ANSWERED), fail=False)
    assert pipeline.asked == [QUESTION]
