"""
tests/test_zero_row_leaves_nothing_current.py

An answer that returned nothing must not leave the previous one current.

result_cache.store refuses an empty row set, and every consumer reads
`entry.rows`, so "the current result is nothing" had no representation in the
cache at all. That absence was not a missing feature; it produced a wrong
answer. A question that found no matching records left the PREVIOUS answer as
the session's current result, so:

    reader:  revenue by region          -> a table, 4 rows
    reader:  revenue for Zorg Ltd       -> "I could not find matching records"
    reader:  just the top 3             -> the top 3 OF REVENUE BY REGION

computed correctly, labelled with the older question, and flagged nowhere. The
reader had no way to know which result they were looking at.

Three doors had to be shut, because closing any two leaves the third open:

  the adapter's own pointers  last_result / last_result_id, which is what a
                              follow-up with no explicit result_id acts on
  the cache's implicit lookup get_snapshot(session_id) with no result_id was
                              still handing back the previous snapshot
  the durable restore         walks this thread's traces newest-first and SKIPS
                              any with no rows, so it stepped over the zero-row
                              answer and restored the one before it

A snapshot addressed BY ID is deliberately untouched throughout: the reader can
still see that card on screen, and its chips, its inline chat and its CSV all
have to keep working.
"""

import json
import os
import sys
import tempfile

import pytest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_zero_row_current.db")
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
    {"REGION": "West", "NET_AMOUNT": 40.0},
]


# ══════════════════════════════════════════════════════════════════════════════
# The cache's own contract
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def cache():
    from core.result_cache import result_cache

    session_id = f"acct{os.urandom(4).hex()}:web_1:thread:t1"
    yield result_cache, session_id
    result_cache.clear(session_id)


class TestTheCacheCanSayNothingIsCurrent:

    def test_an_implicit_lookup_finds_nothing(self, cache):
        result_cache, session_id = cache
        result_cache.store(session_id, list(ROWS),
                           question="revenue by region", sql="SELECT 1")
        assert result_cache.has_result(session_id) is True
        result_cache.mark_no_result(session_id)
        assert result_cache.has_result(session_id) is False
        assert result_cache.get_snapshot(session_id) == {}

    def test_the_card_the_reader_can_still_see_keeps_working(self, cache):
        """Everything on screen is addressed by its own id."""
        result_cache, session_id = cache
        result_id = result_cache.store(
            session_id, list(ROWS), question="revenue by region", sql="SELECT 1")
        result_cache.mark_no_result(session_id)
        assert result_cache.has_result(session_id, result_id=result_id) is True
        assert result_cache.get_snapshot(session_id, result_id)["rows"] == ROWS
        assert result_cache.get_stats(session_id, result_id=result_id)
        assert result_cache.get_schema(session_id, result_id=result_id)

    def test_a_real_answer_supersedes_it(self, cache):
        result_cache, session_id = cache
        result_cache.store(session_id, list(ROWS), question="a", sql="SELECT 1")
        result_cache.mark_no_result(session_id)
        result_cache.store(session_id, [{"N": 1.0}], question="b", sql="SELECT 2")
        assert result_cache.has_no_result(session_id) is False
        assert result_cache.get_snapshot(session_id)["rows"] == [{"N": 1.0}]

    def test_it_says_why_there_is_nothing(self, cache):
        """"the last question found nothing" and "that result expired" are
        different sentences to read, and the caller cannot tell them apart from
        an absent snapshot alone."""
        result_cache, session_id = cache
        assert result_cache.has_no_result(session_id) is False
        result_cache.mark_no_result(session_id)
        assert result_cache.has_no_result(session_id) is True

    def test_clearing_the_session_clears_it(self, cache):
        result_cache, session_id = cache
        result_cache.mark_no_result(session_id)
        result_cache.clear(session_id)
        assert result_cache.has_no_result(session_id) is False

    def test_it_is_idempotent_and_harmless_on_an_empty_session(self, cache):
        result_cache, session_id = cache
        result_cache.mark_no_result(session_id)
        result_cache.mark_no_result(session_id)
        result_cache.mark_no_result("")
        assert result_cache.has_no_result(session_id) is True
        assert result_cache.has_no_result("") is False

    def test_one_session_does_not_speak_for_another(self, cache):
        """The marker is per session, like everything else in this cache."""
        result_cache, session_id = cache
        other = f"acct{os.urandom(4).hex()}:web_2:thread:t1"
        result_cache.store(other, list(ROWS), question="theirs", sql="SELECT 1")
        try:
            result_cache.mark_no_result(session_id)
            assert result_cache.has_result(other) is True
            assert result_cache.has_no_result(other) is False
        finally:
            result_cache.clear(other)


# ══════════════════════════════════════════════════════════════════════════════
# The adapter
# ══════════════════════════════════════════════════════════════════════════════

class TestTheAdapterForgetsTheCurrentResult:

    def _adapter(self, session_suffix="t1"):
        from unittest.mock import AsyncMock

        from gateway.web_adapter import WebAdapter

        return WebAdapter(AsyncMock(), f"acct{os.urandom(4).hex()}", "7",
                          thread_id=session_suffix)

    def test_it_clears_both_pointers_and_marks_the_cache(self):
        from core.result_cache import result_cache

        adapter = self._adapter()
        adapter.cache_result(list(ROWS), "revenue by region", "SELECT 1")
        assert adapter.last_result_id
        assert result_cache.has_result(adapter.session_id) is True
        try:
            adapter.forget_current_result()
            assert adapter.last_result_id is None
            assert adapter.last_result == {}
            assert result_cache.has_no_result(adapter.session_id) is True
        finally:
            result_cache.clear(adapter.session_id)

    def test_a_cache_failure_is_not_the_answers_problem(self, monkeypatch):
        """The answer is already built when this runs."""
        import core.result_cache as rc

        adapter = self._adapter()

        def _explode(_session_id):
            raise RuntimeError("cache gone")

        monkeypatch.setattr(rc.result_cache, "mark_no_result", _explode)
        adapter.forget_current_result()             # must not raise
        assert adapter.last_result_id is None


# ══════════════════════════════════════════════════════════════════════════════
# The consequence, over the real socket
# ══════════════════════════════════════════════════════════════════════════════

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


def _follow_up_after_an_empty_answer(typed, *, frame=None, mark=True):
    """Seed a result, declare the next answer empty, then send `typed`.

    `mark` off reproduces the old behaviour, so each assertion below can be
    read against what it replaces.
    """
    import gateway.webhooks as wh
    import portal.routes as pr
    from core.result_cache import result_cache

    client = _client_app()
    account_id, user_id = _reader()
    session_id = f"{account_id}:web_{user_id}:thread:t1"
    result_id = result_cache.store(
        session_id, list(ROWS), question="revenue by region",
        sql="SELECT REGION, SUM(NET_AMOUNT) AS NET_AMOUNT FROM DBO.F_SALES GROUP BY REGION",
        metadata={"account_id": account_id, "user_id": str(user_id)})
    if mark:
        result_cache.mark_no_result(session_id)
    original = wh.resolve_provider
    wh.resolve_provider = lambda *a, **k: ("test", "test", "k", {})
    try:
        client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
        with client.websocket_connect(f"/ws/chat/{account_id}?thread_id=t1") as ws:
            ws.receive_json()
            ws.send_json(frame or {"type": "message", "text": typed})
            frames = _drain(ws)
    finally:
        wh.resolve_provider = original
        result_cache.clear(session_id)
    return frames, result_id


def _rows_in(frames):
    for frame in frames:
        if frame.get("type") == "assistant_response":
            return list((frame.get("data") or {}).get("rows") or [])
    return None


class TestAFollowUpDoesNotSilentlyActOnTheOlderAnswer:

    def test_the_old_behaviour_really_did_transform_the_previous_result(self):
        """The bug, reproduced. Without the marker "just the top 3" comes back
        as a table -- three rows of the answer before last."""
        frames, _rid = _follow_up_after_an_empty_answer(
            "just the top 3", mark=False)
        rows = _rows_in(frames)
        assert rows is not None, [f.get("type") for f in frames]
        assert len(rows) == 3, rows

    def test_it_now_refuses_instead(self):
        frames, _rid = _follow_up_after_an_empty_answer("just the top 3")
        assert _rows_in(frames) is None, (
            "a follow-up was answered from the result before the empty one")
        assert any(f.get("type") == "assistant_error" for f in frames), \
            [f.get("type") for f in frames]

    def test_the_refusal_says_the_last_question_found_nothing(self):
        """Not "that result is no longer available": nothing expired, and a
        query did run. execute_result_command only sees an absent snapshot, so
        it cannot tell those two apart on its own."""
        frames, _rid = _follow_up_after_an_empty_answer("just the top 3")
        errors = [str(f.get("content") or "") for f in frames
                  if f.get("type") == "assistant_error"]
        assert errors, [f.get("type") for f in frames]
        assert "found no matching records" in errors[0], errors
        assert "no longer available" not in errors[0], errors

    def test_the_composer_is_still_released(self):
        frames, _rid = _follow_up_after_an_empty_answer("just the top 3")
        assert any(f.get("type") == "typing" and f.get("active") is False
                   for f in frames)


class TestTheInCardConversationIsAddressedByIdAndSurvives:

    def _ask_the_card(self, typed, *, mark=True, use_id=True):
        import gateway.webhooks as wh
        import portal.routes as pr
        from core.result_cache import result_cache

        client = _client_app()
        account_id, user_id = _reader()
        session_id = f"{account_id}:web_{user_id}:thread:t1"
        result_id = result_cache.store(
            session_id, list(ROWS), question="revenue by region",
            sql="SELECT REGION, SUM(NET_AMOUNT) AS NET_AMOUNT FROM DBO.F_SALES GROUP BY REGION",
            metadata={"account_id": account_id, "user_id": str(user_id)})
        if mark:
            result_cache.mark_no_result(session_id)
        original = wh.resolve_provider
        wh.resolve_provider = lambda *a, **k: ("test", "test", "k", {})
        try:
            client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
            with client.websocket_connect(
                    f"/ws/chat/{account_id}?thread_id=t1") as ws:
                ws.receive_json()
                ws.send_json({"type": "result_chat", "question": typed,
                              "result_id": result_id if use_id else ""})
                frames = []
                for _ in range(14):
                    frame = ws.receive_json()
                    frames.append(frame)
                    if frame.get("type") in {
                        "result_chat_response", "result_chat_error",
                        "result_chat_message", "result_chat_clarification",
                    }:
                        break
        finally:
            wh.resolve_provider = original
            result_cache.clear(session_id)
        return frames

    def test_the_card_still_answers_when_addressed_by_id(self):
        frames = self._ask_the_card("keep the top 2")
        reply = next((f for f in frames if f.get("type").startswith("result_chat_")
                      and f.get("type") != "result_chat_typing"), {})
        assert reply.get("type") == "result_chat_response", reply
        assert reply.get("row_count") == 2, reply

    def test_a_card_with_no_id_is_told_why_there_is_nothing(self):
        """The panel sends an empty result_id when it has lost track. That
        lands on the implicit lookup, which is exactly the case the marker
        governs -- so the message has to be the honest one."""
        frames = self._ask_the_card("keep the top 2", use_id=False)
        reply = next((f for f in frames if f.get("type") == "result_chat_error"), {})
        assert reply, [f.get("type") for f in frames]
        assert "found no matching records" in str(reply.get("content")), reply


class TestTheDurableRestoreDoesNotResurrectIt:
    """The third door, and the one that stays open longest.

    Result cards outlive the in-process cache, so _restore_durable_thread_result
    rebuilds one from this thread's stored answer traces. It walks them
    newest-first and SKIPS any with no rows -- sensibly, since it cannot restore
    what has none. The effect was that a zero-row answer was stepped over and
    the answer BEFORE it was restored as the session's current result: the same
    staleness the in-process marker prevents, arriving by the other door, and
    surviving a reconnect that clears the marker's own memory.
    """

    def _restore(self, *, mark):
        """Two stored answers, the newer one empty, then an implicit restore."""
        import gateway.webhooks as wh
        from core.result_cache import result_cache

        account_id, user_id = _reader()
        session_id = f"{account_id}:{user_id}:thread:t1"

        def _trace(question_id, question, rows, row_count):
            trace_id = store.create_answer_trace(
                account_id=account_id, question_id=question_id,
                question_text=question, portal_user_id=int(user_id),
                session_id=session_id, request_source="portal")
            store.update_answer_trace(
                trace_id,
                generated_sql=f"SELECT * FROM DBO.F_SALES -- {question_id}")
            if rows:
                store.store_protected_result_rows(account_id, question_id, rows)
            store.finish_answer_trace(
                trace_id, status="success", answer_type="table",
                row_count=row_count)

        _trace("older-with-rows", "revenue by region", list(ROWS), len(ROWS))
        _trace("newer-but-empty", "revenue for Zorg Ltd", [], 0)

        # A real WebAdapter: the restore calls cache_result on it, which a
        # hand-built stub would omit -- and then the restore fails for that
        # reason rather than the one under test.
        from unittest.mock import AsyncMock

        from gateway.web_adapter import WebAdapter

        adapter = WebAdapter(AsyncMock(), account_id, str(user_id),
                             thread_id="t1")
        assert adapter.session_id == session_id, adapter.session_id
        # No in-process snapshot at all: this is the state after a reconnect.
        result_cache.clear(session_id)
        if mark:
            result_cache.mark_no_result(session_id)
        try:
            restored = wh._restore_durable_thread_result(
                account_id, int(user_id), adapter, result_id=None)
        finally:
            result_cache.clear(session_id)
        return restored, adapter

    def test_the_old_behaviour_restored_the_answer_before_the_empty_one(self):
        """The bug at this layer, reproduced: the rowless newer trace is
        skipped and the older one becomes current."""
        restored, adapter = self._restore(mark=False)
        assert restored.get("rows"), restored
        assert restored.get("question") == "revenue by region", restored
        assert adapter.last_result_id

    def test_it_restores_nothing_once_the_answer_is_known_to_be_empty(self):
        restored, adapter = self._restore(mark=True)
        assert restored == {}, restored
        assert adapter.last_result_id is None

    def test_a_card_asked_for_by_id_still_restores(self):
        """The reader can see that card; only the implicit restore is refused."""
        import gateway.webhooks as wh
        from core.result_cache import result_cache

        _restored, adapter = self._restore(mark=True)
        account_id, user_id, _rest = adapter.session_id.split(":", 2)
        session_id = adapter.session_id
        trace_id = store.create_answer_trace(
            account_id=account_id, question_id="asked-for-by-id",
            question_text="revenue by region", portal_user_id=int(user_id),
            session_id=session_id, request_source="portal")
        store.update_answer_trace(
            trace_id, generated_sql="SELECT * FROM DBO.F_SALES")
        store.store_protected_result_rows(
            account_id, "asked-for-by-id", list(ROWS))
        store.finish_answer_trace(
            trace_id, status="success", answer_type="table", row_count=len(ROWS))
        result_cache.mark_no_result(session_id)
        try:
            restored = wh._restore_durable_thread_result(
                account_id, int(user_id), adapter,
                result_id="asked-for-by-id")
        finally:
            result_cache.clear(session_id)
        assert restored.get("rows"), restored


class TestThePipelineActuallyCallsIt:
    """The tests above set the marker by hand. This one checks the pipeline
    does.

    Read as source, because the call sits inside _handle_query_impl -- 6,500
    lines that need a warehouse, a knowledge base and a live socket to reach --
    and read as a syntax tree rather than as text, so it pins the invariant
    (forgotten when rows is empty, BEFORE the diagnostic is sent) instead of a
    spelling.
    """

    @staticmethod
    def _pipeline_ast():
        import ast
        import inspect

        import core.query_pipeline as qp

        source = inspect.getsource(qp)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == "_handle_query_impl":
                return node, source
        raise AssertionError("_handle_query_impl is no longer in the pipeline")

    def _forget_calls(self):
        """Every call of the forgetting hook, with the `if` that guards it."""
        import ast

        node, _source = self._pipeline_ast()
        found = []
        for stmt in ast.walk(node):
            if not isinstance(stmt, ast.If):
                continue
            test = ast.dump(stmt.test)
            for inner in ast.walk(stmt):
                if not isinstance(inner, ast.Call):
                    continue
                # getattr(adapter, "forget_current_result", None) names it, and
                # the call itself is of whatever that was bound to.
                for arg in inner.args:
                    if isinstance(arg, ast.Constant) \
                            and arg.value == "forget_current_result":
                        found.append((stmt.lineno, test))
        return found

    def test_the_pipeline_forgets_the_result_somewhere(self):
        assert self._forget_calls(), (
            "nothing in _handle_query_impl clears the current result, so a "
            "zero-row answer leaves the previous one addressable")

    def test_it_is_guarded_on_an_empty_result(self):
        """Not on something else that happens to be nearby."""
        guards = [test for _line, test in self._forget_calls()]
        assert any("rows" in guard for guard in guards), guards
        assert any("_zero_match_diagnostic" in guard or "len" in guard
                   for guard in guards), guards

    def test_it_happens_before_the_diagnostic_is_sent(self):
        """After the send, the reader's next turn has already been decided by
        the stale pointer -- and after the `return`, it would never run."""
        _node, source = self._pipeline_ast()
        forget_at = min(line for line, _test in self._forget_calls())
        send_at = source[:source.index("_build_zero_row_message(")].count("\n") + 1
        assert forget_at < send_at, (forget_at, send_at)

    def test_the_adapter_actually_has_the_method_it_calls(self):
        """A getattr hook that names a method nobody implements is a no-op that
        looks like a feature."""
        from gateway.web_adapter import WebAdapter

        assert callable(getattr(WebAdapter, "forget_current_result", None))
