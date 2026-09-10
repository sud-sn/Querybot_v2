"""
tests/test_result_chat_conversation.py

Talking to a result card, over the real socket, with nothing stubbed between
the frame the browser sends and the frame it gets back.

The `result_chat` branch of gateway/webhooks.py builds its reply with
`build_assistant_response`. That name used to be imported by four *other*
branches of ws_chat -- the row-lasso handler and two chip handlers -- and by
none of the ones that read it. Python decides a name is local to a function
if it is assigned anywhere in that function's body, so the branch-local
`from core.response_builder import ...` made it a local of the whole 4,000-line
`ws_chat`, and every read from a branch that did not run one of those imports
raised `UnboundLocalError: cannot access local variable
'build_assistant_response' where it is not associated with a value`.

The `except Exception` around the branch turned that into
`{"type": "result_chat_error", "content": "Something went wrong."}`, so the
in-card conversation was dead end to end -- for every question the card
successfully understood -- and looked like a model failure from the outside.

These tests send the frame a browser sends and assert on the frame that comes
back, so they fail if the reply is not built. The commands chosen here are the
ones `parse_result_command` handles deterministically: no LLM is involved on
this path, which is why nothing here mocks one.
"""

import os
import sys
import tempfile

import pytest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_result_chat_conv.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for _mod in list(sys.modules):
    if _mod.startswith("store"):
        del sys.modules[_mod]

import store  # noqa: E402

store.init_db()

ROWS = [
    {"REGION": "North", "NET_AMOUNT": 100.0},
    {"REGION": "South", "NET_AMOUNT": 60.0},
    {"REGION": "East", "NET_AMOUNT": 25.0},
    {"REGION": "West", "NET_AMOUNT": 5.0},
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


def _ask_the_card(typed, *, rows=None):
    """Send one result_chat frame over a real socket; return every frame back.

    resolve_provider runs before the planner and raises without an API key, so
    it is replaced at the boundary. The command itself is parsed
    deterministically, so the stub is never called to plan anything -- it only
    stops the branch dying before it reaches the code under test.
    """
    import gateway.webhooks as wh
    import portal.routes as pr
    from core.result_cache import result_cache

    client = _client_app()
    account_id, user_id = _reader()
    session_id = f"{account_id}:web_{user_id}:thread:t1"
    result_cache.store(
        session_id,
        list(ROWS if rows is None else rows),
        question="revenue by region",
        sql="SELECT REGION, SUM(NET_AMOUNT) AS NET_AMOUNT FROM DBO.F_SALES GROUP BY REGION",
        column_formats={"NET_AMOUNT": "currency"},
    )
    frames = []
    original_provider = wh.resolve_provider
    wh.resolve_provider = lambda *a, **k: ("test", "test", "k", {})
    try:
        client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
        with client.websocket_connect(f"/ws/chat/{account_id}?thread_id=t1") as ws:
            ws.receive_json()          # greeting / connected line
            ws.send_json({"type": "result_chat", "question": typed, "result_id": ""})
            for _ in range(12):
                frame = ws.receive_json()
                frames.append(frame)
                if frame.get("type") in {
                    "result_chat_response",
                    "result_chat_error",
                    "result_chat_clarification",
                }:
                    break
    finally:
        wh.resolve_provider = original_provider
        result_cache.clear(session_id)
    return frames


def _reply(frames):
    for frame in frames:
        if frame.get("type") in {
            "result_chat_response",
            "result_chat_error",
            "result_chat_clarification",
        }:
            return frame
    return {}


class TestTheCardAnswersWhatItUnderstands:
    """Every one of these questions crashed before the reply was built."""

    @pytest.mark.parametrize("typed", [
        "keep the top 2",
        "sort by NET_AMOUNT",
        "exclude East",
    ])
    def test_a_command_the_card_understands_comes_back_as_an_answer(self, typed):
        reply = _reply(_ask_the_card(typed))
        assert reply.get("type") == "result_chat_response", (
            f"{typed!r} did not produce an answer frame: {reply!r}"
        )

    def test_keeping_the_top_two_really_returns_two_rows(self):
        """Not just a frame -- the right rows, from the real cache engine."""
        reply = _reply(_ask_the_card("keep the top 2"))
        assert reply["row_count"] == 2
        regions = [r.get("REGION") for r in reply["rows"]]
        assert regions == ["North", "South"], regions

    def test_excluding_a_row_drops_exactly_that_row(self):
        reply = _reply(_ask_the_card("exclude East"))
        assert reply["row_count"] == 3
        assert "East" not in [r.get("REGION") for r in reply["rows"]]

    def test_the_answer_carries_the_rendered_payload_not_just_raw_rows(self):
        """
        build_assistant_response is what produces `diagnostics` and the
        formatted rows. If the branch ever goes back to reading a name it
        cannot see, these keys are the first thing to disappear.
        """
        reply = _reply(_ask_the_card("keep the top 2"))
        assert reply.get("source") == "governed_cache"
        assert isinstance(reply.get("diagnostics"), dict)
        assert reply.get("column_formats", {}).get("NET_AMOUNT") == "currency"

    def test_the_derived_result_is_addressable_for_the_next_turn(self):
        """A conversation needs the card it just produced to have an id."""
        reply = _reply(_ask_the_card("keep the top 2"))
        assert reply.get("derived_result_id"), reply
