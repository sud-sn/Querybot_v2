"""A thread reopened from the history list shows the answers it was given, and every thread is listed.

A reader refreshed the page on a new-core answer ("Purchased quantity by quarter in
2026": one line, EA only) and got another answer back: a line per unit, a "Full
distribution" chip, units on a stock-status axis, and no Add to dashboard. A reopened
thread was rebuilt from its rows by today's pipeline, whose wording and chart are another
engine's. Now a new-core answer is kept as it was shown and comes back as it was, with a
new Add to dashboard token; one given before answers were kept comes back as its own
sentence and rows, never as today's pipeline's.

Threads went missing from the list: it was built from the reader's last 200 answers and
skipped every turn that ran no query, and opening a thread read the reader's OLDEST 200
answers, so past 200 a recent thread opened empty. Now the list reads every answer
without its rows, a thread is opened by its own id, and a new-core turn that ran no query
(a question back) is kept with what it said.

On reconnecting (an idle tab, a restarted server) the socket said "Connected as ..." or
greeted the reader again, and the page dropped it between two answers. A thread with
messages now carries on without it.

Through the real socket and the real routes; the new core's answers are stand-ins (the
engine itself is tested elsewhere). Invented data only.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from tests import answer_harness as harness
from tests import portal_harness as portal
from tests.chat_js import source as chat_page_source
from tests.js_lift import function as lift

PLAN = {"intent": "trend", "measures": ["purchased_quantity"], "group_by": [], "time": {"grain": "quarter"}}
ROWS = [{"quarter": "2026-Q1", "qty": 58000}, {"quarter": "2026-Q2", "qty": 458},
        {"quarter": "2026-Q3", "qty": 21830}]
CHARTED = {
    "type": "assistant_response", "engine": "core2", "question": "q",
    "answer": {"headline": "Purchased quantity fell from 58,000 EA in Q1 2026 to 21,830 EA in Q3 2026.",
               "short_value": "21,830 EA", "comparison": "", "scope_badge": "", "scope_note": "",
               "badges": [{"kind": "period", "text": "2026"}]},
    "result_scope": {"badge": "", "note": ""},
    "chart": {"title": "Purchased quantity (EA)", "chart_type": "line", "x_key": "quarter", "y_keys": ["qty"],
              "rows": ROWS, "renderable_types": ["line", "area", "bar"], "allowed_types": ["line", "area", "bar"],
              "recommended_type": "line", "column_formats": {"qty": "integer"},
              "chart_warnings": ["Only EA is drawn: the other 11 units are in the table."]},
    "kpi": None, "insight_summary": "", "anomaly_callouts": [],
    "key_insights": ["Q2 2026 was the low point, at 458 EA."],
    "coverage_caveats": [], "follow_up_suggestions": [{"label": "By warehouse", "question": "By warehouse"}],
    "data": {"headers": ["quarter", "qty"], "header_labels": {"quarter": "Quarter", "qty": "Purchased quantity"},
             "rows": ROWS, "total_rows": 3, "truncated": False, "column_formats": {"qty": "integer"}},
    "trust": {"engine": "core2", "sql": "SELECT quarter, qty FROM dbo.purchases", "row_count": 3,
              "model_version": 1},
    "confidence": {}, "plan": PLAN,
}
ASKED_BACK = {
    "type": "assistant_response", "engine": "core2", "question": "q",
    "answer": {"headline": "Which quantity do you mean: purchased or received?", "short_value": "",
               "comparison": "", "scope_badge": "", "scope_note": ""},
    "chart": None, "kpi": None, "data": None, "trust": {"engine": "core2"}, "confidence": {},
    "insight_summary": "", "anomaly_callouts": [], "coverage_caveats": [],
    "clarify": {"about": "measure", "question": "Which quantity do you mean", "options": ["purchased", "received"]},
    "follow_up_suggestions": [{"label": "purchased", "question": "purchased"},
                              {"label": "received", "question": "received"}],
}


@pytest.fixture(scope="module")
def tenant(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("reopened-threads")) as built:
        yield built


def _answering(frames: dict[str, dict]):
    """portal_answer as the bridge calls it: each question's frame, with the bridge's question id."""
    def answer(account_id, question, *args, **kwargs):
        payload = json.loads(json.dumps(frames[question]))
        payload["question"] = question
        payload["trust"]["question_id"] = kwargs["question_id"]
        if payload.get("data"):
            payload["export_rows"] = payload["data"]["rows"]
        return payload
    return answer


def _ask_all(tenant, frames: dict[str, dict]) -> tuple[list[dict], int]:
    """Ask each question in one thread in new-core mode; the frames sent, and the reader's id."""
    import gateway.core2_bridge as bridge

    async def setting(account_id):
        return "core2"

    sent = []
    with patch.object(bridge, "engine", setting), patch("core2.service.portal_answer", _answering(frames)):
        with portal.SocketConversation(tenant, harness.ACCOUNT, harness.saved_connection(),
                                       idle_seconds=2.0) as conversation:
            for question in frames:
                turn = conversation.ask(question)
                sent += [f for f in turn["frames"] if f.get("type") == "assistant_response"]
            return sent, conversation.user_id


def _portal(user_id: int):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    import portal.routes as pr

    app = FastAPI()
    app.include_router(pr.router)
    client = TestClient(app)
    client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
    return client


def _thread_of(trace_id: int) -> str:
    import store

    return str(store.get_answer_trace(trace_id)["session_id"]).rsplit(":thread:", 1)[-1]


def _turns(client, thread_id: str) -> list[dict]:
    response = client.get(f"/portal/api/history/{thread_id}")
    assert response.status_code == 200, response.text
    return response.json()["turns"]


def _listed(client) -> dict[str, dict]:
    response = client.get("/portal/api/history")
    assert response.status_code == 200, response.text
    return {item["thread_id"]: item for item in response.json()["items"]}


def _without_tokens(frame: dict) -> dict:
    out = {k: v for k, v in frame.items() if k != "pin_token"}
    if isinstance(out.get("chart"), dict):
        out["chart"] = {k: v for k, v in out["chart"].items() if k != "pin_token"}
    return out


def _pinned_plan(token: str) -> dict:
    from store.db import get_db

    with get_db() as conn:
        row = conn.execute("SELECT display_config FROM pin_token WHERE token=?", (token,)).fetchone()
    assert row is not None, "the token Add to dashboard sends is not a pin token"
    return json.loads(row["display_config"])["core2_plan"]


# ── a reopened thread ──────────────────────────────────────────────────────


def test_a_reopened_new_core_answer_is_the_one_shown_and_can_be_added_to_a_dashboard(tenant):
    (shown,), user_id = _ask_all(tenant, {"purchased quantity by quarter in 2026": CHARTED})
    assert shown["engine"] == "core2" and shown["pin_token"]
    (turn,) = _turns(_portal(user_id), _thread_of(int(shown["trace_id"])))
    again = turn["payload"]
    # The same answer: its sentence, findings, chart (one line, EA only, its warning) and table.
    assert _without_tokens(again) == _without_tokens(shown)
    assert "Full distribution" not in json.dumps(again)
    # Add to dashboard again, with a token of its own that carries the answer's plan.
    assert again["pin_token"] and again["pin_token"] != shown["pin_token"]
    assert again["chart"]["pin_token"] == again["pin_token"]
    assert _pinned_plan(again["pin_token"]) == PLAN


def test_a_new_core_question_back_is_kept_in_the_thread_and_the_thread_is_listed(tenant):
    sent, user_id = _ask_all(tenant, {"quantity by quarter": ASKED_BACK,
                                      "purchased quantity by quarter in 2026": CHARTED})
    asked_back, answered = sent
    thread_id = _thread_of(int(asked_back["trace_id"]))
    client = _portal(user_id)
    turns = _turns(client, thread_id)
    assert [t["question"] for t in turns] == ["quantity by quarter", "purchased quantity by quarter in 2026"]
    assert turns[0]["payload"]["answer"]["headline"] == ASKED_BACK["answer"]["headline"]
    assert [c["question"] for c in turns[0]["payload"]["follow_up_suggestions"]] == ["purchased", "received"]
    assert "pin_token" not in turns[0]["payload"]          # it ran no query: nothing to pin
    listed = _listed(client)[thread_id]
    assert listed["question"] == "quantity by quarter" and listed["turn_count"] == 2


def test_a_thread_of_only_a_question_back_is_listed(tenant):
    (asked_back,), user_id = _ask_all(tenant, {"quantity please": ASKED_BACK})
    thread_id = _thread_of(int(asked_back["trace_id"]))
    assert thread_id in _listed(_portal(user_id))


def _reader(account: str, tables: tuple[str, ...] = ("DBO.STOCK",)) -> int:
    """A reader (not an admin) whose group may read ``tables``."""
    import store

    group_id = store.create_group(account, f"grp{os.urandom(3).hex()}")
    store.set_group_tables(group_id, account, list(tables))
    user_id, _ = store.create_user(account, "Rowan Reader", f"rowan-{os.urandom(4).hex()}@example.test",
                                   group_id=group_id, password="a-password-they-chose")
    return int(user_id)


def _trace(account: str, user_id: int, thread_id: str, question: str, *, route: str = "", sql: str = "",
           rows: list[dict] | None = None, frame: dict | None = None, summary: str = "") -> int:
    import store

    trace_id = store.create_answer_trace(account_id=account, question_id=f"c2-{os.urandom(6).hex()}",
                                         question_text=question, portal_user_id=user_id,
                                         session_id=f"{account}:{user_id}:thread:{thread_id}",
                                         request_source="portal", route=route)
    store.update_answer_trace(trace_id, generated_sql=sql, result_rows=rows or [], db_type="azure_sql",
                              query_row_count=len(rows or []), final_answer_summary=summary, status="success",
                              **({"answer_frame": frame} if frame is not None else {}))
    return trace_id


def test_every_thread_is_listed_and_opens_however_many_answers_came_after(tenant):
    user_id = _reader(harness.ACCOUNT)
    first = f"first_{os.urandom(3).hex()}"
    _trace(harness.ACCOUNT, user_id, first, "stock on hand by warehouse",
           sql="SELECT warehouse, qty FROM dbo.stock", rows=[{"warehouse": "Main", "qty": 1200}])
    for n in range(520):       # a busy reader: 520 answers in 52 later threads
        _trace(harness.ACCOUNT, user_id, f"later-{n // 10}", f"question {n}", sql="SELECT 1 AS v",
               rows=[{"v": 1}])
    client = _portal(user_id)
    listed = _listed(client)
    assert first in listed and len(listed) == 53
    assert listed[first]["question"] == "stock on hand by warehouse"
    (turn,) = _turns(client, first)
    assert turn["question"] == "stock on hand by warehouse" and turn["payload"]["data"]["rows"]
    # The newest thread too: opening a thread once read the reader's oldest answers only.
    assert [t["question"] for t in _turns(client, "later-51")] == [f"question {n}" for n in range(510, 520)]


def test_a_thread_id_is_read_exactly_not_as_a_pattern(tenant):
    """"_" in a thread id is a character, not a wildcard: "a_b" is not "aXb"."""
    user_id = _reader(harness.ACCOUNT)
    _trace(harness.ACCOUNT, user_id, "aXb", "another thread", sql="SELECT 1 AS v", rows=[{"v": 1}])
    import store

    client = _portal(user_id)
    assert _turns(client, "a_b") == []
    assert store.list_answer_traces(harness.ACCOUNT, 10, portal_user_id=user_id, thread_id="a_b") == []
    assert store.list_answer_traces(harness.ACCOUNT, 10, portal_user_id=user_id, thread_id="a%") == []
    assert len(_turns(client, "aXb")) == 1


def test_an_answer_given_before_answers_were_kept_is_the_new_cores_sentence_and_rows(tenant):
    import store

    user_id = _reader(harness.ACCOUNT)
    thread_id = f"old_{os.urandom(3).hex()}"
    trace_id = _trace(harness.ACCOUNT, user_id, thread_id, "stock by warehouse", route="core2",
                      sql="SELECT warehouse, stock_on_hand FROM dbo.stock",
                      rows=[{"warehouse": "Main", "stock_on_hand": 1200}, {"warehouse": "North", "stock_on_hand": 300}],
                      summary="Main holds the most stock: 1,200 EA of 1,500 EA.")
    question_id = store.get_answer_trace(trace_id)["question_id"]
    store.log_core2_answer(harness.ACCOUNT, user_id=str(user_id), mode="core2", question="stock by warehouse",
                           status="answered", headline="", sql="", row_count=2, plan=PLAN, duration_ms=5,
                           model_version=1, question_id=question_id)
    (turn,) = _turns(_portal(user_id), thread_id)
    payload = turn["payload"]
    assert payload["engine"] == "core2"
    assert payload["answer"]["headline"] == "Main holds the most stock: 1,200 EA of 1,500 EA."
    assert payload["data"]["rows"] == [{"warehouse": "Main", "stock_on_hand": 1200},
                                       {"warehouse": "North", "stock_on_hand": 300}]
    assert payload["data"]["header_labels"]["stock_on_hand"] == "Stock on hand"
    assert payload["trace_id"] == trace_id and _pinned_plan(payload["pin_token"]) == PLAN
    # Drawn as it was drawn when reopened before answers were kept: never a bare table.
    assert payload["chart"] and payload["chart"]["chart_type"] == "bar"
    assert payload["chart"]["pin_token"] == payload["pin_token"]


def test_a_monthly_answer_given_before_answers_were_kept_comes_back_as_a_chart(tenant):
    """Reopened as a table only, after answers began to be kept: every older trend lost its chart."""
    user_id = _reader(harness.ACCOUNT)
    thread_id = f"oldtrend_{os.urandom(3).hex()}"
    months = [{"period": f"2026-{m:02d}-01", "number_of_receipts": n} for m, n in ((1, 213), (2, 40), (3, 30),
                                                                                    (4, 12), (5, 16))]
    _trace(harness.ACCOUNT, user_id, thread_id, "number of receipts by month", route="core2",
           sql="SELECT period, receipts FROM dbo.stock", rows=months,
           summary="Number of receipts, by month: 213 in Jan 2026, 16 in May 2026.")
    (turn,) = _turns(_portal(user_id), thread_id)
    chart = turn["payload"]["chart"]
    assert chart and chart["chart_type"] in ("line", "area", "bar"), chart
    assert [r["number_of_receipts"] for r in chart["rows"]] == [213, 40, 30, 12, 16]


def test_a_kept_answer_with_rows_but_no_query_is_never_shown(tenant):
    """A turn that ran no query has nothing to check a grant against: it comes back only without rows."""
    user_id = _reader(harness.ACCOUNT)
    thread_id = f"norows_{os.urandom(3).hex()}"
    _trace(harness.ACCOUNT, user_id, thread_id, "which quantity?", route="core2", frame=ASKED_BACK)
    _trace(harness.ACCOUNT, user_id, thread_id, "rows without a query", route="core2",
           frame={**ASKED_BACK, "data": CHARTED["data"]})
    assert [t["question"] for t in _turns(_portal(user_id), thread_id)] == ["which quantity?"]


def test_a_kept_answer_on_a_table_the_reader_has_lost_is_not_shown(tenant):
    user_id = _reader(harness.ACCOUNT, tables=("DBO.STOCK",))
    thread_id = f"lost_{os.urandom(3).hex()}"
    _trace(harness.ACCOUNT, user_id, thread_id, "stock", route="core2", sql="SELECT qty FROM dbo.stock",
           rows=[{"qty": 1}], frame={**CHARTED, "trust": {**CHARTED["trust"], "sql": "SELECT qty FROM dbo.stock"}})
    _trace(harness.ACCOUNT, user_id, thread_id, "purchases", route="core2", sql=CHARTED["trust"]["sql"],
           rows=ROWS, frame=CHARTED)
    assert [t["question"] for t in _turns(_portal(user_id), thread_id)] == ["stock"]


# ── what is kept ───────────────────────────────────────────────────────────


def test_an_answer_is_kept_with_the_question_as_kept_and_without_its_tokens(tenant):
    import gateway.core2_bridge as bridge
    import store

    payload = json.loads(json.dumps(CHARTED))
    payload.update(question="purchases for Jane Doe", pin_token="old-pin",
                   own_metric={"name": "Net buys", "token": "own-token", "formula": "qty - returns"})
    payload["chart"]["pin_token"] = "old-pin"
    payload["trust"]["question_id"] = question_id = f"c2-{os.urandom(6).hex()}"
    with patch("core2.service.question_scrubber", return_value=lambda text: text.replace("Jane Doe", "[name]")):
        trace_id = bridge._keep(harness.ACCOUNT, None, "s:thread:kept", "purchases for Jane Doe", question_id,
                                payload, ROWS, 12)
    kept = json.loads(store.get_answer_trace(trace_id)["answer_frame"])
    assert kept["question"] == "purchases for [name]" and "Jane Doe" not in json.dumps(kept)
    assert "pin_token" not in kept and "pin_token" not in kept["chart"]
    assert kept["own_metric"] == {"name": "Net buys", "formula": "qty - returns"}
    assert kept["chart"] == {k: v for k, v in CHARTED["chart"].items()} and kept["plan"] == PLAN


def test_an_answer_too_large_to_keep_is_reopened_from_its_rows(tenant, monkeypatch):
    import gateway.core2_bridge as bridge
    import store

    monkeypatch.setattr(bridge, "MAX_KEPT_FRAME", 200)
    payload = json.loads(json.dumps(CHARTED))
    payload["trust"]["question_id"] = question_id = f"c2-{os.urandom(6).hex()}"
    with patch("core2.service.question_scrubber", return_value=None):
        trace_id = bridge._keep(harness.ACCOUNT, None, "s:thread:big", "purchases", question_id, payload, ROWS, 12)
    trace = store.get_answer_trace(trace_id)
    assert trace["answer_frame"] == ""
    assert bridge.reopened(harness.ACCOUNT, None, trace, ROWS)["answer"]["headline"] == CHARTED["answer"]["headline"]


# ── reconnecting ───────────────────────────────────────────────────────────


def test_the_socket_marks_its_connected_notice(tenant):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    import gateway.webhooks as wh
    import portal.routes as pr
    import store

    user_id = _reader(harness.ACCOUNT)
    store.touch_user_activity(user_id)       # an active session: "Connected as", not a greeting
    app = FastAPI()
    app.include_router(wh.router)
    client = TestClient(app)
    client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
    with client.websocket_connect(f"/ws/chat/{harness.ACCOUNT}?thread_id=t{os.urandom(3).hex()}") as socket:
        first = socket.receive_json()
    assert first["type"] == "system" and first["connect"] is True and "Rowan Reader" in first["content"]


def _handled(frames: list[dict], *, conversation: bool) -> dict[str, list]:
    """Feed frames to the page's own socket handler; what it put in the thread."""
    import dukpy

    src = chat_page_source()
    harness_js = f"""
var location = {{protocol: 'https:', host: 'portal.example'}};
let _socket = null;
function WebSocket(url) {{ this.url = url; _socket = this; }}
const ACCOUNT_ID = 'acct', THREAD_ID = 'thread-1', DASHBOARD_ID = 0;
let reconnectAttempts = 0, wsReady = false, agentRunState = 'idle', _pendingSuggestion = null;
function t(id) {{ return id; }}
function setConnectionState() {{}}
function safeJsonParse(text) {{ try {{ return JSON.parse(text); }} catch (e) {{ return null; }} }}
const _put = {{bot: [], system: []}};
function appendBot(text, opts) {{ _put.bot.push(text); }}
function appendSystem(text, opts) {{ _put.system.push([text, !!(opts || {{}}).persist]); }}
function setAgentRunState() {{ return true; }}
function _markActiveOutbound() {{}}
function _setStageState() {{}}
function _genieEvent() {{}}
function refreshQueryLimitStatus() {{}}
function showMascotError() {{}}
const _startScreen = {{style: {{display: 'none'}}}};
var document = {{getElementById: function (id) {{ return id === 'welcomeState' ? _startScreen : null; }}}};
function thread() {{ return {{querySelector: function () {{ return {json.dumps(conversation)} ? {{}} : null; }}}}; }}
{lift(src, "function _terminalRunState(msg)")};
{lift(src, "function connect()")}
connect();
{json.dumps(frames)}.forEach(frame => _socket.onmessage({{data: JSON.stringify(frame)}}));
JSON.stringify(_put);
"""
    return json.loads(dukpy.evaljs(harness_js))


CONNECTED = {"type": "system", "content": "Connected as Rowan. Ask me anything about your data.", "connect": True}
GREETING = {"type": "message", "role": "assistant", "content": "Hello, Rowan!", "greeting": True}


def test_a_conversation_the_socket_reconnects_to_carries_on_without_notices():
    put = _handled([CONNECTED, GREETING, {"type": "system", "content": "Stopped."}], conversation=True)
    assert put == {"bot": [], "system": [["Stopped.", True]]}


def test_an_empty_thread_says_it_is_connected_without_keeping_it():
    put = _handled([CONNECTED], conversation=False)
    assert put == {"bot": [], "system": [[CONNECTED["content"], False]]}


def test_an_answer_that_showed_peoples_data_is_withheld_once_the_reader_is_no_longer_cleared(tenant):
    """Shown as given while the reader's attestation holds; once it is revoked, neither the reopened answer
    nor its CSV shows the data again (core/compliance/kept_answers.py)."""
    import store

    user_id = _reader(harness.ACCOUNT, tables=("DBO.PURCHASES",))
    thread_id = f"released_{os.urandom(3).hex()}"
    granted = store.save_user_attestation(harness.ACCOUNT, str(user_id))
    trace_id = _trace(harness.ACCOUNT, user_id, thread_id, "purchases", route="core2", sql=CHARTED["trust"]["sql"],
                      rows=ROWS, frame={**CHARTED, "released": True})
    client = _portal(user_id)
    (turn,) = _turns(client, thread_id)
    assert turn["payload"]["answer"]["headline"] == CHARTED["answer"]["headline"]
    store.revoke_user_attestation(harness.ACCOUNT, granted, "admin")
    (turn,) = _turns(client, thread_id)
    assert turn["payload"].get("withheld") is True and "no longer cleared" in turn["payload"]["answer"]["headline"]
    assert "data" not in turn["payload"]
    exported = client.get(f"/portal/api/export-csv?trace_id={trace_id}")
    assert exported.status_code == 403 and "no longer cleared" in exported.json()["error"]
