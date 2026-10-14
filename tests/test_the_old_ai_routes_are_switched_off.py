"""The earlier engine's own AI calls are switched off: no data reaches the AI except through the new core.

The new core answers every question, and every AI call it makes goes through its audit, its masking and its
compliance rules. Four doors of the earlier engine still called the AI around it:

* the chat inside each answer card ("Ask about this result") ran the earlier engine's card chat, whose AI calls
  included writing SQL against the database. The new core caches no result for it, so on a new-core card it
  only ever said "No cached result found"; a result left in the cache answered through the earlier engine;
* a reply to a clarification the earlier engine asked ran that engine again;
* the old knowledge-base build sent each table's sampled rows to the AI to write its documents;
* "Suggest" for a table description sent that knowledge base's text (sample values included) to the AI.

Now the card chat says to ask in the box below, where the new core reads the answer above; a stale clarification
is cleared; and both admin routes say Learn has replaced them, without calling the AI. The card has no chat of its
own. The login report offer, the one clarification the new core keeps, still works.

Driven through the real socket and the real admin routes; the AI is replaced and fails the test if it is called.
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import BackgroundTasks, FastAPI
from starlette.testclient import TestClient

ROWS = [{"REGION": "North", "NET_AMOUNT": 100.0}, {"REGION": "South", "NET_AMOUNT": 60.0}]


def _never(*args, **kwargs):
    raise AssertionError("the AI was called")


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.chdir(tmp_path)
    import store

    store.init_db()
    from core import llm
    from gateway import webhooks

    monkeypatch.setattr(llm, "llm_complete", AsyncMock(side_effect=_never))
    monkeypatch.setattr(webhooks, "llm_complete", AsyncMock(side_effect=_never))
    monkeypatch.setattr(webhooks, "resolve_provider", lambda *a, **k: ("test", "test", "k", {}))
    yield store


def _reader(store):
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    store.update_client_meta(account_id, chat_ui_enabled=1, enable_llm_audit=1)
    user_id, _ = store.create_user(account_id, "Ada", f"{os.urandom(4).hex()}@x.com", password="a-password-they-chose")
    return account_id, user_id


def _socket(store, frame: dict, *, before=None) -> list[dict]:
    """Send one frame over the real chat socket; return what comes back until a reply."""
    import portal.routes as pr
    from gateway import webhooks

    app = FastAPI()
    app.include_router(webhooks.router)
    client = TestClient(app)
    account_id, user_id = _reader(store)
    session_id = f"{account_id}:web_{user_id}:thread:t1"
    if before:
        before(account_id, user_id, session_id)
    frames: list[dict] = []
    client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
    with client.websocket_connect(f"/ws/chat/{account_id}?thread_id=t1") as ws:
        ws.receive_json()
        ws.send_json(frame)
        for _ in range(12):
            got = ws.receive_json()
            frames.append(got)
            if got.get("type") in {"result_chat_error", "result_chat_response", "result_chat_message",
                                   "result_chat_clarification", "error", "assistant_error", "message", "system"}:
                break
    return frames


def test_the_card_chat_says_to_ask_below_even_with_a_result_cached(fresh):
    from core.i18n import t
    from core.result_cache import result_cache

    def cached(account_id, user_id, session_id):
        result_cache.store(session_id, list(ROWS), question="revenue by region",
                           sql="SELECT REGION, SUM(NET_AMOUNT) AS NET_AMOUNT FROM F_SALES GROUP BY REGION")

    for typed, card in (("why is North so high?", ""), ("keep the top 1", ""), ("and by month?", "rc-1")):
        frames = _socket(fresh, {"type": "result_chat", "question": typed, "result_id": card}, before=cached)
        reply = frames[-1]
        assert reply["type"] == "result_chat_error" and reply["result_id"] == card, frames
        assert reply["content"] == t("reply.result.card_chat_off"), reply


def test_a_clarification_the_earlier_engine_asked_is_cleared_not_answered(fresh):
    from core.clarification import get_pending, save_pending

    seen = {}

    def asked(account_id, user_id, session_id):
        seen["key"] = (account_id, f"web_{user_id}", session_id)
        save_pending(account_id, f"web_{user_id}", "net sales by region",
                     clarification_meta={"source": "llm", "mode": "single", "options": [
                         {"id": "a", "label": "Order date", "value": "order date"}]},
                     session_id=session_id)

    frames = _socket(fresh, {"type": "clarification_response", "option_id": "a"}, before=asked)
    assert frames[-1]["type"] == "error", frames
    account_id, user, session_id = seen["key"]
    assert not get_pending(account_id, user, session_id=session_id), "the old question is not left waiting"


def test_the_login_report_offer_still_answers(fresh):
    from core.clarification import save_pending

    def offered(account_id, user_id, session_id):
        save_pending(account_id, f"web_{user_id}", "__login_report_prompt__",
                     clarification_meta={"source": "login_report_prompt", "mode": "multi", "options": [
                         {"id": "no_thanks", "label": "No thanks", "value": "no_thanks"}]},
                     session_id=session_id)

    frames = _socket(fresh, {"type": "clarification_response", "option_id": "no_thanks"}, before=offered)
    assert frames[-1]["type"] != "error", frames


def _admin(store):
    from admin import routes

    app = FastAPI()
    app.include_router(routes.router)
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    return routes, TestClient(app), account_id


def test_suggesting_a_table_description_calls_no_ai(fresh):
    routes, client, account_id = _admin(fresh)
    with patch.object(routes, "_is_auth", return_value=True):
        response = client.post(f"/admin/clients/{account_id}/setup/table-description/suggest",
                               json={"table_name": "dbo.ORDERS"})
    assert response.status_code == 410, response.text
    assert "Learn" in response.json()["detail"]


def test_the_old_knowledge_base_build_starts_nothing(fresh, tmp_path):
    from fastapi import Request

    routes, _, account_id = _admin(fresh)
    schema = tmp_path / "schema"
    schema.mkdir()
    (schema / "ORDERS.md").write_text("# dbo.ORDERS\n", encoding="utf-8")
    fresh.update_client_state(account_id, "SCHEMA_READY", {"schema_dir": str(schema)})
    request = Request({"type": "http", "method": "POST", "path": "/", "headers": [], "query_string": b""})
    tasks = BackgroundTasks()
    with patch.object(routes, "_is_auth", return_value=True), \
            patch("core.knowledge.build_kb", AsyncMock(side_effect=_never)):
        response = asyncio.run(routes.admin_build_kb(request, account_id, tasks, business_desc="Orders."))
        asyncio.run(tasks())
    assert response.status_code == 303 and "Learn" in response.headers["location"].replace("+", " ").replace(
        "%20", " "), response.headers
    assert not tasks.tasks
    assert json.loads(fresh.get_client(account_id)["state_data"] or "{}").get("kb_dir") in (None, "")


def test_an_answer_card_has_no_chat_of_its_own():
    pytest.importorskip("dukpy")
    from tests.test_the_answer_card import BY_STORE, Elements, card

    for engine in ("core2", "earlier"):
        shown = Elements(card({**BY_STORE, "engine": engine, "trust": {**BY_STORE["trust"], "engine": engine}}))
        assert not any(e["attrs"].get("data-rc-toggle") for e in shown.found), engine
        assert not shown.first("rc-panel"), engine
