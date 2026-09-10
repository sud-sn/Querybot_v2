"""
tests/test_ws_turn_never_locks_the_composer.py

A turn that crashes must still hand the composer back.

Every question in ws_chat runs as a fire-and-forget background task, and the
browser re-enables the composer on exactly one signal:

    portal/templates/portal_chat.html
    if (msg.active) setAgentRunState('running', msg);
    else if (agentRunState === 'running') setProcessing(false);

There is no add_done_callback anywhere in gateway/webhooks.py, and three of
the nine background coroutines had no top-level handler of their own. So an
exception raised before a coroutine's own `except` could run -- or anywhere
inside one of the unguarded ones -- ended the turn with `typing: true` on the
wire and nothing after it. The reader could not type another word, and Stop
could not help them: it acted only on a task that was NOT done, and a crashed
task is done. A page reload was the only way out.

The trigger is not exotic. `resolve_provider` raises when the workspace has no
API key configured yet -- an ordinary state for a tenant mid-setup -- and it is
the first thing several of those coroutines do. In that state every follow-up
about the result on screen was met with a spinner that never stopped.

These tests drive the real socket, crash the boundary, and assert on the frames
the browser would actually receive.
"""

import json
import os
import sys
import tempfile

import pytest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_ws_turn_guard.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for _mod in list(sys.modules):
    if _mod.startswith("store"):
        del sys.modules[_mod]

import store  # noqa: E402

store.init_db()

ROWS = [
    {"REGION": "North", "NET_AMOUNT": 100.0},
    {"REGION": "South", "NET_AMOUNT": 60.0},
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


def _turn(typed, *, then_cancel=False, monkeypatch=None):
    """Send one main-chat message with the provider boundary raising.

    Returns every frame the browser receives. `resolve_provider` is the
    boundary, not the code under test: a workspace with no API key makes it
    raise for real, which is what this reproduces.
    """
    import gateway.webhooks as wh
    import portal.routes as pr
    from core.result_cache import result_cache

    client = _client_app()
    account_id, user_id = _reader()
    session_id = f"{account_id}:web_{user_id}:thread:t1"
    result_cache.store(
        session_id, list(ROWS),
        question="revenue by region",
        sql="SELECT REGION, SUM(NET_AMOUNT) AS NET_AMOUNT FROM DBO.F_SALES GROUP BY REGION",
    )

    def _no_api_key(*_a, **_k):
        raise RuntimeError("No API key configured for provider 'anthropic'")

    frames = []
    original = wh.resolve_provider
    wh.resolve_provider = _no_api_key
    try:
        client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
        with client.websocket_connect(f"/ws/chat/{account_id}?thread_id=t1") as ws:
            ws.receive_json()                       # greeting / connected line
            ws.send_json({"type": "message", "text": typed})
            frames.extend(_drain(ws))
            if then_cancel:
                ws.send_json({"type": "cancel"})
                frames.extend(_drain(ws))
    finally:
        wh.resolve_provider = original
        result_cache.clear(session_id)
    return frames


def _drain(ws, limit=25, timeout=4.0):
    """Read until the spinner is cleared, or until nothing more is coming.

    Timeout-aware on purpose. The failure this file exists for produces
    exactly ONE frame and then silence, and `ws.receive_json()` blocks for
    ever on silence -- so a plain reader here would make these tests HANG when
    the guard is missing instead of failing, which is barely better than
    passing. This reaches through TestClient's blocking portal to the same
    stream `receive()` uses, with a deadline on it.
    """
    import anyio

    out = []
    for _ in range(limit):
        async def _recv():
            with anyio.fail_after(timeout):
                return await ws._send_rx.receive()

        try:
            message = ws.portal.call(_recv)
        except Exception:
            break                       # timed out, or the socket closed
        if message.get("type") != "websocket.send" or "text" not in message:
            break
        frame = json.loads(message["text"])
        out.append(frame)
        if frame.get("type") == "typing" and frame.get("active") is False:
            break
    return out


def _composer_released(frames):
    return any(f.get("type") == "typing" and f.get("active") is False
               for f in frames)


# The three routes whose coroutines had no handler of their own, plus the
# ordinary question path. Each phrase is chosen to reach a different
# create_task site in ws_chat.
ROUTES = [
    pytest.param("what does NET_AMOUNT mean?", id="metadata-result-planner"),
    pytest.param("revenue by region", id="main-question"),
    pytest.param("sort it by revenue", id="result-command-or-planner"),
]


class TestACrashedTurnStillReleasesTheComposer:

    @pytest.mark.parametrize("typed", ROUTES)
    def test_the_spinner_is_always_cleared(self, typed):
        frames = _turn(typed)
        assert _composer_released(frames), (
            f"{typed!r} left the composer locked; frames: "
            f"{[(f.get('type'), f.get('active')) for f in frames]}"
        )

    @pytest.mark.parametrize("typed", ROUTES)
    def test_the_reader_is_told_something_went_wrong(self, typed):
        """A silent spinner-off would be an improvement and still not an
        answer. The reader asked a question."""
        frames = _turn(typed)
        assert any(
            f.get("type") in {"assistant_error", "message", "system"}
            and str(f.get("content") or "").strip()
            for f in frames
        ), [(f.get("type"), str(f.get("content"))[:60]) for f in frames]


class TestStopWorksOnATurnThatAlreadyDied:
    """Stop is the only control the browser leaves enabled once the composer
    is locked, and it did nothing for a task that had already crashed -- which
    is the one case where the reader needs it."""

    def test_pressing_stop_is_answered_rather_than_swallowed(self):
        """`if current_query_task and not current_query_task.done():` skipped
        the whole cancel body -- its own typing:false included -- for a task
        that had already died. Stop produced zero frames."""
        frames = _turn("what does NET_AMOUNT mean?", then_cancel=True)
        stopped = [f for f in frames if f.get("type") == "system"
                   and str(f.get("content") or "").strip()]
        assert stopped, [(f.get("type"), str(f.get("content"))[:50])
                         for f in frames]

    def test_stop_clears_the_spinner_a_second_time(self):
        """Two releases: the guard's, and Stop's own. Either alone is enough
        for the reader, and both firing is what proves Stop is not a no-op on
        a dead task."""
        frames = _turn("what does NET_AMOUNT mean?", then_cancel=True)
        releases = [f for f in frames
                    if f.get("type") == "typing" and f.get("active") is False]
        assert len(releases) >= 2, [
            (f.get("type"), f.get("active")) for f in frames]


class TestEveryBackgroundTurnIsGuarded:
    """The tests above drive three of the nine routes. This one covers the
    rest.

    Reading source, because the other six need a dashboard request, a report
    request, a metric definition, a reconcile phrase or a plan preview to
    reach -- each with its own tenant state -- and a guard that covers three
    sites of a defect class is the same missed-siblings pattern the defect is.
    Read as a syntax tree rather than as text, so it pins the invariant (every
    background turn is wrapped) instead of a spelling, and asks the parser
    which call is actually being scheduled.
    """

    @staticmethod
    def _ws_chat_ast():
        import ast
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1]
                  / "gateway" / "webhooks.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == "ws_chat":
                return node
        raise AssertionError("ws_chat is no longer in gateway/webhooks.py")

    def _scheduled_calls(self):
        """(what is scheduled, line) for every asyncio.create_task in ws_chat."""
        import ast

        out = []
        for node in ast.walk(self._ws_chat_ast()):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "create_task"):
                continue
            if not node.args:
                out.append(("<no argument>", node.lineno))
                continue
            arg = node.args[0]
            name = ""
            if isinstance(arg, ast.Call):
                inner = arg.func
                name = (inner.id if isinstance(inner, ast.Name)
                        else inner.attr if isinstance(inner, ast.Attribute) else "")
            out.append((name or "<not a call>", arg.lineno))
        return out

    def test_the_scan_found_the_turns(self):
        """A scan that stopped matching would pass the test below for ever."""
        assert len(self._scheduled_calls()) >= 9, self._scheduled_calls()

    def test_nothing_is_scheduled_unguarded(self):
        unguarded = [(name, line) for name, line in self._scheduled_calls()
                     if name != "_guarded_turn"]
        assert not unguarded, (
            "a background turn that raises before its own handler runs leaves "
            "the composer locked with no way back but a page reload; these "
            "are not wrapped: " + repr(unguarded))
