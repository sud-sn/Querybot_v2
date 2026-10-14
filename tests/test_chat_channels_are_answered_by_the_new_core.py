"""Teams, Slack, Zoom and the question API are answered by the new core, as the portal is.

The portal had moved to the new core; a question asked in Teams, Slack or Zoom still went to today's pipeline,
which a workspace set up for the new core alone never built (its knowledge base, its readiness), so it was told
"not ready". Now every chat is answered by the new core:

* the person is the Solviz user their chat account is linked to, with their role, tables and row rules; an
  account linked to nobody is asked to register, and a viewer is refused, before any engine;
* the answer is a chat message: its sentence, a small table of its first rows, and where the rest is;
* each chat is a conversation of its own ("and by month?" follows the answer above it in that chat alone);
* a question back comes as buttons where the chat has them; a dashboard is made in the portal;
* a workspace the new core has not learned says it is being set up; ``status`` says whether it is.

The question API answers as the workspace, and follows a conversation when its caller names one.

The real store and dispatcher; the new core's answer is invented where the test is about the chat around it.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest
from fastapi import BackgroundTasks

import store
from core import dispatcher
from gateway import PlatformEvent
from gateway import core2_bridge as bridge


@pytest.fixture()
def workspace():
    store.init_db()
    account = f"chat-{uuid.uuid4().hex[:10]}"
    store.upsert_client(account, "teams")
    people = {}
    for name, role in (("Ada", "asker"), ("Vic", "viewer")):
        user_id, _ = store.create_user(account, name, f"{name.lower()}-{uuid.uuid4().hex[:6]}@test.com",
                                       portal_role=role)
        store.link_zoom_user(user_id, f"chat-{name.lower()}")
        people[name] = store.get_user(user_id)
    yield account, people
    with store.get_db() as conn:
        for table in ("pending_platform_user", "answer_trace", "query_log", "core2_answer_log", "portal_user",
                      "client"):
            try:
                conn.execute(f"DELETE FROM {table} WHERE account_id=?", (account,))
            except Exception:     # noqa: BLE001 - a table this build does not have
                pass


class Chat:
    """A chat platform's adapter: what was said, and the buttons offered."""

    def __init__(self, buttons: bool = True):
        self.said: list[str] = []
        self.asked: list[tuple[str, list[dict]]] = []
        if buttons:
            self.send_clarification_prompt = self._buttons

    async def send_message(self, event, text, **_):
        self.said.append(text)

    async def _buttons(self, event, question, options, pending_id=None):
        self.asked.append((question, options))


def _event(account: str, user: str, text: str, *, platform: str = "teams", chat: str = "conv-1") -> PlatformEvent:
    channel = json.dumps({"service_url": "https://teams.invalid", "conversation_id": chat}) \
        if platform == "teams" else chat
    return PlatformEvent(account_id=account, user_id=user, channel_id=channel, text=text, platform=platform)


def _ask(account: str, event: PlatformEvent, chat: Chat) -> None:
    asyncio.run(dispatcher.dispatch(account, event, chat, BackgroundTasks()))


ANSWER = {
    "type": "assistant_response", "engine": "core2",
    "answer": {"headline": "Net amount is $1.20M, led by Store A ($700.0K).", "short_value": "$1.20M"},
    "chart": {"chart_type": "bar"},
    "key_insights": ["Store A is 58% of the total."],
    "coverage_caveats": [],
    "data": {"headers": ["store", "net_amount"], "header_labels": {"store": "Store", "net_amount": "Net amount"},
             "rows": [{"store": f"Store {chr(65 + i)}", "net_amount": 700000 - i * 10000} for i in range(14)],
             "total_rows": 14, "column_formats": {"store": "text", "net_amount": "currency"}},
    "trust": {"engine": "core2", "sql": "SELECT 1", "row_count": 14},
}


@pytest.fixture()
def new_core(monkeypatch):
    """The new core's answer, invented: who asked, in which conversation."""
    asked: list[dict] = []
    answers: list[dict] = []

    def portal_answer(account_id, question, portal_user, *, session_key, question_id="", progress=None):
        asked.append({"account": account_id, "question": question, "user": portal_user, "session": session_key})
        return json.loads(json.dumps(answers.pop(0) if answers else ANSWER))

    import core2.service

    monkeypatch.setattr(core2.service, "portal_answer", portal_answer)
    monkeypatch.setattr(bridge, "_over_limit", lambda *a, **k: "")
    import core.query_pipeline

    monkeypatch.setattr(core.query_pipeline, "handle_query", lambda *a, **k: pytest.fail("today's pipeline ran"))
    monkeypatch.setattr(dispatcher, "_enqueue_query", lambda *a, **k: pytest.fail("today's pipeline ran"))
    return asked, answers


# ── who is answered ─────────────────────────────────────────────────────────


def test_a_linked_teams_user_is_answered_by_the_new_core_as_themselves(workspace, new_core):
    account, people = workspace
    asked, _ = new_core
    chat = Chat()
    _ask(account, _event(account, "chat-ada", "net amount by store"), chat)
    assert [a["question"] for a in asked] == ["net amount by store"]
    assert asked[0]["user"]["id"] == people["Ada"]["id"], "answered as the linked Solviz user"
    message = chat.said[-1]
    assert message.startswith("**Net amount is $1.20M")
    assert "| Store | Net amount |" in message and "| Store A | $700,000.00 |" in message
    assert "- Store A is 58% of the total." in message


def test_a_workspace_set_up_for_the_new_core_alone_is_answered(workspace, new_core):
    """Today's pipeline never built it (its state stays NEW): it used to say "not ready"."""
    account, _ = workspace
    assert (store.get_client_state(account) or {}).get("state", "NEW") in ("NEW", None, "")
    chat = Chat()
    _ask(account, _event(account, "chat-ada", "net amount by store"), chat)
    assert chat.said and chat.said[-1].startswith("**Net amount")


def test_an_account_linked_to_nobody_is_not_answered(workspace, new_core):
    account, _ = workspace
    asked, _ = new_core
    chat = Chat()
    _ask(account, _event(account, "someone-else", "net amount by store"), chat)
    assert not asked and chat.said and "net amount" not in chat.said[-1].lower()


def test_a_viewer_is_refused_before_any_engine(workspace, new_core):
    account, _ = workspace
    asked, _ = new_core
    chat = Chat()
    _ask(account, _event(account, "chat-vic", "net amount by store", platform="slack"), chat)
    assert not asked and "dashboards shared with you" in chat.said[-1].lower()


# ── the conversation ────────────────────────────────────────────────────────


def test_each_chat_is_a_conversation_of_its_own(workspace, new_core):
    account, _ = workspace
    asked, _ = new_core
    for chat_id in ("conv-1", "conv-1", "conv-2"):
        _ask(account, _event(account, "chat-ada", "net amount by store", chat=chat_id), Chat())
    _ask(account, _event(account, "chat-ada", "net amount by store", platform="zoom", chat="conv-1"), Chat())
    sessions = [a["session"] for a in asked]
    assert sessions[0] == sessions[1], "the same Teams chat follows on"
    assert len({sessions[0], sessions[2], sessions[3]}) == 3, "another chat, or another platform, starts afresh"


def test_a_question_back_comes_as_buttons(workspace, new_core):
    account, _ = workspace
    _, answers = new_core
    back = {**ANSWER, "data": None, "chart": None, "key_insights": [],
            "answer": {"headline": "Which date: order date or ship date?"},
            "clarify": {"about": "date", "question": "Which date", "options": ["order date", "ship date"]}}
    answers.append(back)
    chat = Chat()
    _ask(account, _event(account, "chat-ada", "net amount by month"), chat)
    assert chat.asked == [("Which date", [{"id": "order date", "label": "order date"},
                                          {"id": "ship date", "label": "ship date"}])]
    answers.append(back)
    plain = Chat(buttons=False)
    _ask(account, _event(account, "chat-ada", "net amount by month", platform="zoom"), plain)
    assert plain.said[-1] == "**Which date: order date or ship date?**"


def test_a_dashboard_is_made_in_the_portal(workspace, new_core):
    account, _ = workspace
    asked, _ = new_core
    chat = Chat()
    _ask(account, _event(account, "chat-ada", "Create an executive dashboard"), chat)
    assert not asked and "Dashboards are made in the Solviz portal" in chat.said[-1]


def test_the_monthly_limit_is_said(workspace, new_core, monkeypatch):
    account, _ = workspace
    asked, _ = new_core
    monkeypatch.setattr(bridge, "_over_limit", lambda *a, **k: "This workspace has used its 500 questions.")
    chat = Chat()
    _ask(account, _event(account, "chat-ada", "net amount by store"), chat)
    assert not asked and chat.said[-1] == "This workspace has used its 500 questions."


def test_an_answer_is_kept_in_history_as_asked_in_the_chat(workspace, new_core):
    account, people = workspace
    _ask(account, _event(account, "chat-ada", "net amount by store", platform="slack"), Chat())
    with store.get_db() as conn:
        row = conn.execute("SELECT request_source, route, portal_user_id FROM answer_trace WHERE account_id=?",
                           (account,)).fetchone()
    assert tuple(row) == ("slack", "core2", people["Ada"]["id"])


# ── a workspace not learned ─────────────────────────────────────────────────


def test_a_workspace_not_learned_says_it_is_being_set_up(workspace, monkeypatch):
    """The real new core: no database connected, nothing learned."""
    account, _ = workspace
    monkeypatch.setattr(bridge, "_over_limit", lambda *a, **k: "")
    chat = Chat()
    _ask(account, _event(account, "chat-ada", "net amount by store"), chat)
    assert "being set up" in chat.said[-1]


def test_status_says_whether_the_workspace_is_learned(workspace):
    account, _ = workspace
    chat = Chat()
    _ask(account, _event(account, "chat-ada", "status"), chat)
    assert "being set up" in chat.said[-1] and "Ada" in chat.said[-1]


# ── the message ─────────────────────────────────────────────────────────────


def test_the_message_shows_the_first_rows_and_where_the_rest_is():
    text = bridge.channel_text(json.loads(json.dumps(ANSWER)), link="https://solviz.example/portal/chat")
    rows = [line for line in text.splitlines() if line.startswith("| Store ") and "$" in line]
    assert len(rows) == bridge.CHANNEL_ROWS
    assert text.splitlines()[-1] == ("_The first 10 of 14 rows. The chart and every row are in Solviz: "
                                     "https://solviz.example/portal/chat_")


def test_a_chart_alone_is_said_to_be_in_solviz():
    few = {**ANSWER, "key_insights": [], "data": {**ANSWER["data"], "rows": ANSWER["data"]["rows"][:3],
                                                  "total_rows": 3}}
    assert bridge.channel_text(few, link="L").splitlines()[-1] == "_The chart is in Solviz: L_"
    many = {**ANSWER, "chart": None}
    assert bridge.channel_text(many, link="L").splitlines()[-1] == \
        "_The first 10 of 14 rows. Every row is in Solviz: L_"


def test_one_number_is_its_sentence_alone():
    one = {**ANSWER, "chart": None, "key_insights": [],
           "data": {"headers": ["net_amount"], "rows": [{"net_amount": 1200000}], "total_rows": 1,
                    "column_formats": {"net_amount": "currency"}}}
    assert bridge.channel_text(one, link="x") == "**Net amount is $1.20M, led by Store A ($700.0K).**"


def test_a_cell_never_breaks_the_table():
    odd = {**ANSWER, "chart": None, "key_insights": [],
           "data": {"headers": ["note"], "rows": [{"note": "a | b\nc"}, {"note": None}], "total_rows": 2,
                    "column_formats": {}}}
    lines = bridge.channel_text(odd).splitlines()
    assert "| a / b c |" in lines and "|  |" in lines


# ── the question API ────────────────────────────────────────────────────────


def _api(body: dict, monkeypatch) -> dict:
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from gateway import webhooks

    monkeypatch.setenv("QUERYBOT_API_KEY", "the-api-key")
    app = FastAPI()
    app.include_router(webhooks.router)
    response = TestClient(app).post("/api/ask", json={"api_key": "the-api-key", **body})
    assert response.status_code == 200, response.text
    return response.json()


def test_the_question_api_is_answered_by_the_new_core_as_the_workspace(workspace, new_core, monkeypatch):
    account, _ = workspace
    asked, _ = new_core
    reply = _api({"question": "net amount by store", "account_id": account}, monkeypatch)
    assert reply["answer"].startswith("**Net amount is $1.20M") and asked[0]["user"] is None


def test_the_question_api_follows_a_conversation_its_caller_names(workspace, new_core, monkeypatch):
    account, _ = workspace
    asked, _ = new_core
    for conversation in ("flow-7", "flow-7", ""):
        _api({"question": "net amount by store", "account_id": account, "conversation_id": conversation},
             monkeypatch)
    _api({"question": "net amount by store", "account_id": account}, monkeypatch)
    sessions = [a["session"] for a in asked]
    assert sessions[0] == sessions[1]
    assert len({sessions[0], sessions[2], sessions[3]}) == 3, "no conversation named: each question alone"
