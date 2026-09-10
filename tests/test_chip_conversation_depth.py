"""
tests/test_chip_conversation_depth.py

The guided conversation was exactly one press deep.

Every result card offers chips -- drill down by a dimension, % contribution,
outliers, compare with the prior period, CSV. Press one and you get a new card
with new rows, and that card offered nothing: no CSV, no contribution, no drill
of its own. The reader could take one step and then had to type.

Two causes, in the same place. The browser gates a card's chips on an id:

    const actionResultId = trust.result_id || msg.data?.result_id || '';
    const nextActions = actionResultId && Array.isArray(msg.next_actions)
                      ? msg.next_actions : [];

and the chip handlers send their cards straight down the socket rather than
through send_assistant_response, which is the only place that id was ever
stamped. So the server computed next_actions for every chip card, put them on
the frame, and the browser threw them away on arrival.

And the rows behind those cards were not addressable either: contribution and
outliers never cached their derived result at all, and drill_dim cached its own
AFTER sending the frame, so the id would have been the parent's.

Everything here presses a real chip over the real socket and reads the frame
the browser would receive.
"""

import json
import os
import sys
import tempfile

import pytest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_chip_depth.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for _mod in list(sys.modules):
    if _mod.startswith("store"):
        del sys.modules[_mod]

import store  # noqa: E402

store.init_db()

# A shape every chip in this file can work on: one text dimension, one
# additive measure, a clear leader -- and one row far enough from the mean
# that `filter_outliers` has something to find, so the outliers chip produces
# a card rather than its (correct) "nothing stands out" refusal.
ROWS = (
    [{"REGION": f"R{index:02d}", "NET_AMOUNT": 100.0 + index}
     for index in range(12)]
    + [{"REGION": "North", "NET_AMOUNT": 9000.0}]
)


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


def _drain(ws, limit=25, timeout=5.0):
    """Frames until the spinner clears. Timeout-aware: a chip that dies
    silently would otherwise hang this file rather than fail it."""
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


def _press(chip):
    """Seed a result, press one chip on it, return the card that comes back."""
    import gateway.webhooks as wh
    import portal.routes as pr
    from core.result_cache import result_cache

    client = _client_app()
    account_id, user_id = _reader()
    session_id = f"{account_id}:web_{user_id}:thread:t1"
    result_id = result_cache.store(
        session_id, list(ROWS),
        question="revenue by region",
        sql="SELECT REGION, SUM(NET_AMOUNT) AS NET_AMOUNT FROM DBO.F_SALES GROUP BY REGION",
        column_formats={"NET_AMOUNT": "currency"},
        metadata={"account_id": account_id, "user_id": str(user_id)},
    )
    frames = []
    original = wh.resolve_provider
    wh.resolve_provider = lambda *a, **k: ("test", "test", "k", {})
    try:
        client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
        with client.websocket_connect(f"/ws/chat/{account_id}?thread_id=t1") as ws:
            ws.receive_json()                     # greeting / connected line
            ws.send_json({
                "action": chip,
                "context": {},
                "result_id": result_id,
                "action_id": "a1",
            })
            frames = _drain(ws)
    finally:
        wh.resolve_provider = original
    # The session is deliberately NOT cleared: one test reads back the
    # snapshot the chip minted, which only exists while the session does.
    # Every call above uses a fresh random account, so nothing leaks between
    # tests.
    return result_id, frames, session_id


def _card(frames):
    for frame in frames:
        if frame.get("type") == "assistant_response":
            return frame
    return {}


# The two chips that are pure local transforms: no LLM, no warehouse, so they
# run end to end in a test with neither.
LOCAL_CHIPS = ["contribution", "outliers"]


class TestAChipCardCanBePressedAgain:

    @pytest.mark.parametrize("chip", LOCAL_CHIPS)
    def test_the_card_carries_an_id_of_its_own(self, chip):
        parent_id, frames, _session = _press(chip)
        card = _card(frames)
        assert card, [f.get("type") for f in frames]
        stamped = (card.get("trust") or {}).get("result_id") \
            or (card.get("data") or {}).get("result_id") or ""
        assert stamped, (
            f"the {chip} card has no result_id, so the browser drops its "
            f"chips: trust={card.get('trust')} data keys="
            f"{list((card.get('data') or {}).keys())}"
        )

    @pytest.mark.parametrize("chip", LOCAL_CHIPS)
    def test_that_id_is_the_new_card_not_the_one_that_was_pressed(self, chip):
        """The top-level result_id says which card was PRESSED -- the browser
        needs it to place the reply. trust.result_id says what the new card
        IS. They are different ids and were the same one."""
        parent_id, frames, _session = _press(chip)
        card = _card(frames)
        assert card.get("result_id") == parent_id      # where to put the reply
        stamped = (card.get("trust") or {}).get("result_id")
        assert stamped and stamped != parent_id, (
            f"{chip} stamped the parent's id, so pressing a chip on the new "
            f"card would act on the old rows"
        )

    @pytest.mark.parametrize("chip", LOCAL_CHIPS)
    def test_the_new_rows_are_actually_in_the_cache_under_that_id(self, chip):
        """An id the browser can send back but the server cannot resolve is
        no better than none. contribution and outliers never cached their
        derived result at all."""
        from core.result_cache import result_cache

        parent_id, frames, session_id = _press(chip)
        card = _card(frames)
        stamped = (card.get("trust") or {}).get("result_id")
        snapshot = result_cache.get_snapshot(session_id, stamped)
        assert snapshot.get("rows"), (
            f"{chip} produced an id that resolves to nothing: {snapshot!r}")
        # And it is the DERIVED result, not the one that was pressed.
        assert snapshot["rows"] != ROWS, snapshot["rows"][:2]
        result_cache.clear(session_id)

    @pytest.mark.parametrize("chip", LOCAL_CHIPS)
    def test_the_card_still_advertises_what_to_press_next(self, chip):
        parent_id, frames, _session = _press(chip)
        card = _card(frames)
        assert card.get("next_actions"), (
            f"{chip} offered the reader nowhere to go next")


class TestTheCorrelationIdsAreStillRight:
    """The frame has to reach the card that was pressed, or the reply lands in
    the wrong place in the thread."""

    @pytest.mark.parametrize("chip", LOCAL_CHIPS)
    def test_the_click_is_still_correlated(self, chip):
        parent_id, frames, _session = _press(chip)
        card = _card(frames)
        assert card.get("action") == chip
        assert card.get("action_id") == "a1"
        assert card.get("result_id") == parent_id

    def test_an_export_is_not_treated_as_a_card(self, monkeypatch):
        """download_csv produces an assistant_export, which has no rows to
        cache and no chips to gate. It must come back untouched but still
        correlated."""
        parent_id, frames, _session = _press("download_csv")
        export = next((f for f in frames
                       if f.get("type") == "assistant_export"), {})
        assert export, [f.get("type") for f in frames]
        assert export.get("result_id") == parent_id
        assert export.get("action_id") == "a1"
        assert "trust" not in export


class TestARefusalIsOneReplyNotTwo:
    """The contribution handler sent its refusal and then fell through to
    `_bound_action_payload(_ct_resp)` -- a name only the `else` binds. So a
    reader who pressed % contribution on a result with nothing summable got
    the refusal AND a second bubble saying something had gone wrong."""

    def _press_on(self, rows):
        import gateway.webhooks as wh
        import portal.routes as pr
        from core.result_cache import result_cache

        client = _client_app()
        account_id, user_id = _reader()
        session_id = f"{account_id}:web_{user_id}:thread:t1"
        result_id = result_cache.store(
            session_id, list(rows), question="share by region", sql="SELECT 1",
            metadata={"account_id": account_id, "user_id": str(user_id)},
        )
        original = wh.resolve_provider
        wh.resolve_provider = lambda *a, **k: ("test", "test", "k", {})
        try:
            client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
            with client.websocket_connect(
                    f"/ws/chat/{account_id}?thread_id=t1") as ws:
                ws.receive_json()
                ws.send_json({"action": "contribution", "context": {},
                              "result_id": result_id, "action_id": "a1"})
                frames = _drain(ws)
        finally:
            wh.resolve_provider = original
            result_cache.clear(session_id)
        return frames

    def test_a_result_with_nothing_to_share_gets_exactly_one_error(self):
        # No numeric column at all, so add_contribution_pct refuses.
        rows = [{"REGION": "North", "STATUS": "open"},
                {"REGION": "South", "STATUS": "closed"}]
        errors = [f for f in self._press_on(rows)
                  if f.get("type") == "assistant_error"]
        assert len(errors) == 1, [
            (f.get("type"), str(f.get("content"))[:60]) for f in errors]

    def test_the_refusal_says_why_rather_than_just_failing(self):
        rows = [{"REGION": "North", "STATUS": "open"},
                {"REGION": "South", "STATUS": "closed"}]
        errors = [f for f in self._press_on(rows)
                  if f.get("type") == "assistant_error"]
        assert errors and errors[0].get("detail"), errors


class TestPressingAChipLeavesTheAnswerAlone:
    """_bound_action_payload used the adapter's FULL response preparer, which
    does two things beyond stamping the id: it overwrites last_response_payload
    (what "add this to my dashboard" pins) and it consumes pending_dashboard
    (materialising a queued widget from whatever card happens to pass through,
    and raising out of the chip handler if that fails)."""

    def _adapter(self):
        from unittest.mock import AsyncMock

        from gateway.web_adapter import WebAdapter

        return WebAdapter(AsyncMock(), "acct", "7", thread_id="t1")

    def _card(self):
        return {
            "type": "assistant_response",
            "question": "revenue by region (% contribution)",
            "data": {"rows": [{"REGION": "North", "PCT": 90.0}]},
            "trust": {"sql": "SELECT 1"},
        }

    def test_the_answer_stays_the_thing_a_dashboard_would_pin(self):
        adapter = self._adapter()
        answer = {"type": "assistant_response", "question": "revenue by region",
                  "data": {"rows": []}, "trust": {}}
        adapter.prepare_assistant_response_payload(answer)
        assert adapter.last_response_payload is answer

        adapter.stamp_result_id(self._card())
        assert adapter.last_response_payload is answer, (
            "a chip press replaced the answer that a dashboard pin refers to")

    def test_a_queued_dashboard_survives_a_chip_press(self):
        adapter = self._adapter()
        adapter.pending_dashboard = {"name": "Q3 review"}
        adapter.stamp_result_id(self._card())
        assert adapter.pending_dashboard == {"name": "Q3 review"}, (
            "the chip consumed the queued dashboard")

    def test_a_failing_dashboard_cannot_take_the_chip_down_with_it(self):
        """materialize_dashboard raising inside the chip handler cost the
        reader both the chip's answer and the queued widget."""
        adapter = self._adapter()
        adapter.pending_dashboard = {"name": "Q3 review"}

        def _explode(*_a, **_k):
            raise RuntimeError("dashboard service down")

        adapter.materialize_dashboard = _explode
        stamped = adapter.stamp_result_id(self._card())      # must not raise
        assert stamped["trust"]["result_id"] == ""

    def test_the_stamp_still_does_its_one_job(self):
        adapter = self._adapter()
        adapter.last_result_id = "derived-123"
        stamped = adapter.stamp_result_id(self._card())
        assert stamped["trust"]["result_id"] == "derived-123"
        assert stamped["data"]["result_id"] == "derived-123"

    def test_a_non_card_payload_is_returned_untouched(self):
        adapter = self._adapter()
        adapter.last_result_id = "derived-123"
        export = {"type": "assistant_export", "filename": "x.csv"}
        assert adapter.stamp_result_id(export) is export
        assert "trust" not in export

    def test_the_full_preparer_still_does_both(self):
        """The other caller, send_assistant_response, still needs the
        bookkeeping -- splitting the stamp out must not have taken it away."""
        adapter = self._adapter()
        adapter.last_result_id = "answer-1"
        adapter.pending_dashboard = {"name": "Q3 review"}
        adapter.materialize_dashboard = lambda payload, **_k: {"id": "dash-1"}
        answer = {"type": "assistant_response", "question": "revenue",
                  "data": {"rows": []}, "trust": {}}
        prepared = adapter.prepare_assistant_response_payload(answer)
        assert prepared["trust"]["result_id"] == "answer-1"
        assert prepared["dashboard_artifact"] == {"id": "dash-1"}
        assert adapter.pending_dashboard is None
        assert adapter.last_response_payload is prepared
