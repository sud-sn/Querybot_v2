"""
tests/test_clarification_loop_closes.py

When the bot asks a question, the answer has to reach it.

Two cards on the cached-result path were dead on arrival:

  "Which result did you mean?"       -- two results on screen, the follow-up
                                       could refer to either
  "Did you mean: keep the top 3?"    -- the planner's reading was uncertain

Both are correct behaviour: better to ask than to guess. Both were sent with
no pending clarification saved behind them, so clicking an option produced

    That clarification is no longer active. Please ask the question again.

not sometimes and not after a timeout -- on the first click, every time. The
card even carried `"pending_id": ""` on the way out, because web_adapter fills
that field from get_pending and there was nothing to get.

And the second card could not have closed even with a pending row: the option
carries the reader's own question back, and re-planning identical text at
temperature 0 scores the same confidence, so "Yes, that is what I meant" would
have produced the identical card again. The retry has to say that the reading
is already confirmed.

These drive the real socket: a message frame in, a clarification_prompt out, a
clarification_response back, and the answer read off the wire.
"""

import json
import os
import sys
import tempfile

import pytest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_clarify_loop.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for _mod in list(sys.modules):
    if _mod.startswith("store"):
        del sys.modules[_mod]

import store  # noqa: E402

store.init_db()

ROWS = [
    {"REGION": "North", "NET_AMOUNT": 900.0},
    {"REGION": "South", "NET_AMOUNT": 240.0},
    {"REGION": "East", "NET_AMOUNT": 110.0},
]


def _client_app():
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from gateway import webhooks

    app = FastAPI()
    app.include_router(webhooks.router)
    return TestClient(app)


def _reader():
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    store.update_client_meta(account_id, chat_ui_enabled=1)
    user_id, _ = store.create_user(account_id, "Ada", f"{os.urandom(4).hex()}@x.com")
    return account_id, user_id


def _drain(ws, limit=30, timeout=5.0):
    import anyio

    out = []
    for _ in range(limit):
        async def _recv():
            with anyio.fail_after(timeout):
                return await ws._send_rx.receive()

        try:
            message = ws.portal.call(_recv)
        except Exception:
            break
        if message.get("type") != "websocket.send" or "text" not in message:
            break
        frame = json.loads(message["text"])
        out.append(frame)
        if frame.get("type") == "typing" and frame.get("active") is False:
            break
    return out


def _of_type(frames, kind):
    return next((f for f in frames if f.get("type") == kind), {})


class _Followup:
    """A canned run_governed_result_followup, replaced at the boundary.

    The real one needs a planner LLM to reach either clarification, and both
    of those clarifications are built deterministically from
    core/governed_result_followup.py -- so the OPTIONS here are the real
    ones, produced by the real builders, and only the decision to ask is
    canned.
    """

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    async def __call__(self, question, session_id, **kwargs):
        import core.governed_result_followup as grf

        self.calls.append({"question": question,
                           "is_clarification": kwargs.get("is_clarification")})
        if kwargs.get("is_clarification"):
            # The reader has confirmed. The real code path skips the
            # confidence gate here and executes.
            return grf.GovernedFollowupResult(
                "unsupported", reason="confirmed, nothing left to ask",
                evidence={})
        return grf.GovernedFollowupResult(
            "clarification", outcome=self.outcome,
            reason=self.outcome.message, evidence={})


def _reference_outcome():
    from core.governed_result_followup import _reference_clarification

    return _reference_clarification(
        "rank the rows by net amount",
        [{"question": "revenue by region"}, {"question": "orders by month"}],
        None,
        "Which result did you mean?",
    ).clarification


def _confidence_outcome():
    from core.governed_result_followup import _confidence_clarification
    from core.result_planner import ResultCommand

    return _confidence_clarification(
        "keep the top 3",
        ResultCommand(action="keep_top", limit=3, metric_text="NET_AMOUNT"),
    )


def _ask_and_answer(outcome_fn, *, answer_with="option"):
    """Provoke the clarification, then answer it. Returns (frames, spy)."""
    import gateway.webhooks as wh
    import portal.routes as pr
    from core.result_cache import result_cache

    client = _client_app()
    account_id, user_id = _reader()
    session_id = f"{account_id}:web_{user_id}:thread:t1"
    result_cache.store(
        session_id, list(ROWS), question="revenue by region",
        sql="SELECT REGION, SUM(NET_AMOUNT) AS NET_AMOUNT FROM DBO.F_SALES GROUP BY REGION",
        metadata={"account_id": account_id, "user_id": str(user_id)},
    )
    spy = _Followup(outcome_fn())
    original_followup = wh.run_governed_result_followup
    original_provider = wh.resolve_provider
    wh.run_governed_result_followup = spy
    wh.resolve_provider = lambda *a, **k: ("test", "test", "k", {})
    asked, answered = [], []
    try:
        client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
        with client.websocket_connect(f"/ws/chat/{account_id}?thread_id=t1") as ws:
            ws.receive_json()
            ws.send_json({"type": "message",
                          "text": "rank the rows in this result by net amount"})
            asked = _drain(ws)
            card = _of_type(asked, "clarification_prompt")
            options = card.get("options") or []
            if options:
                frame = {"type": "clarification_response",
                         "pending_id": card.get("pending_id") or ""}
                if answer_with == "option":
                    frame["option_id"] = options[0].get("id")
                else:
                    frame["text"] = options[0].get("label")
                ws.send_json(frame)
                answered = _drain(ws)
    finally:
        wh.run_governed_result_followup = original_followup
        wh.resolve_provider = original_provider
        result_cache.clear(session_id)
    return asked, answered, spy


OUTCOMES = [
    pytest.param(_reference_outcome, id="which-result-did-you-mean"),
    pytest.param(_confidence_outcome, id="did-you-mean"),
]


class TestTheCardIsBackedByAPending:

    @pytest.mark.parametrize("outcome_fn", OUTCOMES)
    def test_the_card_carries_a_pending_id(self, outcome_fn):
        asked, _answered, _spy = _ask_and_answer(outcome_fn)
        card = _of_type(asked, "clarification_prompt")
        assert card, [f.get("type") for f in asked]
        assert card.get("pending_id"), (
            "the card went out with no pending behind it, so its own answer "
            f"cannot be matched to it: {card}")

    @pytest.mark.parametrize("outcome_fn", OUTCOMES)
    def test_the_browser_gets_choices_and_not_the_machinery(self, outcome_fn):
        """`resolved_question` stays server-side on purpose -- the browser
        answers with an option id and the pending row holds the wording, so a
        translated label can never become the text that is re-planned."""
        asked, _answered, _spy = _ask_and_answer(outcome_fn)
        card = _of_type(asked, "clarification_prompt")
        assert card.get("options"), card
        assert all(o.get("id") and o.get("label") for o in card["options"]), \
            card["options"]
        assert not any("resolved_question" in o for o in card["options"]), \
            card["options"]


class TestClickingAnOptionIsNotRefused:

    @pytest.mark.parametrize("outcome_fn", OUTCOMES)
    def test_the_answer_is_not_met_with_no_longer_active(self, outcome_fn):
        _asked, answered, _spy = _ask_and_answer(outcome_fn)
        refusals = [f for f in answered
                    if f.get("type") == "error"
                    and "no longer active" in str(f.get("content") or "")]
        assert not refusals, refusals

    @pytest.mark.parametrize("outcome_fn", OUTCOMES)
    def test_typing_the_answer_works_too(self, outcome_fn):
        """A reader may type "Result 1" instead of clicking it."""
        _asked, answered, _spy = _ask_and_answer(outcome_fn, answer_with="text")
        refusals = [f for f in answered
                    if f.get("type") == "error"
                    and "no longer active" in str(f.get("content") or "")]
        assert not refusals, refusals


class TestTheConfirmedReadingIsNotReLitigated:

    @pytest.mark.parametrize("outcome_fn", OUTCOMES)
    def test_the_retry_says_the_reader_already_confirmed(self, outcome_fn):
        """Without this the retry re-plans identical text at the same
        temperature, scores the same confidence, and asks the same question
        again -- for ever."""
        _asked, _answered, spy = _ask_and_answer(outcome_fn)
        assert len(spy.calls) == 2, spy.calls
        assert spy.calls[0]["is_clarification"] in (False, None)
        assert spy.calls[1]["is_clarification"] is True, spy.calls

    @pytest.mark.parametrize("outcome_fn", OUTCOMES)
    def test_the_retry_carries_the_disambiguated_question(self, outcome_fn):
        """Not the chip's internal token. "result 2" and "confirm" are how the
        option is identified on the wire; neither is anything a reader typed,
        and sending either into the pipeline asks the warehouse about a phrase
        that does not exist."""
        _asked, _answered, spy = _ask_and_answer(outcome_fn)
        # The wording the real builder put on the option the reader picked.
        expected = (outcome_fn().clarification_options or [])[0]["resolved_question"]
        assert spy.calls[1]["question"] == expected, spy.calls[1]
        # And it is not the wire token.
        assert spy.calls[1]["question"] not in {"confirm", "result 1"}

    @pytest.mark.parametrize("outcome_fn", OUTCOMES)
    def test_the_same_card_is_not_sent_back(self, outcome_fn):
        _asked, answered, _spy = _ask_and_answer(outcome_fn)
        assert not _of_type(answered, "clarification_prompt"), [
            f.get("type") for f in answered]
