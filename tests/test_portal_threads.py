import asyncio
import json
from unittest.mock import patch

from gateway.web_adapter import WebAdapter
from portal.routes import portal_query_history, portal_query_thread


class _FakeWebSocket:
    async def send_json(self, payload):
        return None


def _response_json(response):
    return json.loads(response.body.decode("utf-8"))


def test_web_adapter_cache_and_history_are_thread_scoped():
    first = WebAdapter(_FakeWebSocket(), "acct", "user", thread_id="thread-a")
    second = WebAdapter(_FakeWebSocket(), "acct", "user", thread_id="thread-b")

    assert first.session_id == "acct:user:thread:thread-a"
    assert second.session_id == "acct:user:thread:thread-b"
    assert first.session_id != second.session_id


def test_history_api_filters_by_portal_user_and_groups_thread_turns():
    user = {"id": 7, "account_id": "acct"}
    traces = [
        {
            "id": 3,
            "portal_user_id": 7,
            "session_id": "acct:web_7:thread:abc",
            "question_text_sanitized": "Follow-up",
            "generated_sql": "SELECT 2",
            "query_row_count": 2,
            "query_duration_ms": 20,
            "status": "success",
            "created_at": "2026-07-31 11:00:00",
        },
        {
            "id": 2,
            "portal_user_id": 7,
            "session_id": "acct:web_7:thread:abc",
            "question_text_sanitized": "Original question",
            "generated_sql": "SELECT 1",
            "query_row_count": 1,
            "query_duration_ms": 10,
            "status": "success",
            "created_at": "2026-07-31 10:00:00",
        },
        {
            "id": 1,
            "portal_user_id": 7,
            "session_id": "acct:web_7",
            "question_text_sanitized": "Legacy question",
            "generated_sql": "SELECT 1",
            "query_row_count": 1,
            "status": "success",
            "created_at": "2026-07-30 10:00:00",
        },
    ]
    with patch("portal.routes._get_portal_user", return_value=user), patch(
        "portal.routes.store.list_answer_traces", return_value=traces
    ) as list_traces:
        response = asyncio.run(portal_query_history(object()))

    payload = _response_json(response)
    list_traces.assert_called_once_with("acct", limit=200, portal_user_id=7)
    assert payload["items"][0]["thread_id"] == "abc"
    assert payload["items"][0]["question"] == "Original question"
    assert payload["items"][0]["turn_count"] == 2
    assert payload["items"][1]["thread_id"] == "legacy-1"


def test_thread_detail_reconstructs_result_for_owner_only():
    user = {"id": 7, "account_id": "acct"}
    traces = [{
        "id": 2,
        "portal_user_id": 7,
        "session_id": "acct:web_7:thread:abc",
        "question_id": "qid-2",
        "question_text_sanitized": "How many?",
        "generated_sql": "SELECT 2 AS total",
        "result_rows": '[{"total": 2}]',
        "query_row_count": 1,
        "query_duration_ms": 12,
        "db_type": "azure_sql",
        "status": "success",
        "created_at": "2026-07-31 10:00:00",
    }]
    with patch("portal.routes._get_portal_user", return_value=user), patch(
        "portal.routes.store.list_answer_traces", return_value=traces
    ) as list_traces:
        response = asyncio.run(portal_query_thread(object(), "abc"))

    payload = _response_json(response)
    assert payload["ok"] is True
    assert payload["turns"][0]["payload"]["trust"]["sql"] == "SELECT 2 AS total"
    assert payload["turns"][0]["payload"]["data"]["rows"] == [{"total": "2"}]
    list_traces.assert_called_once_with(
        "acct", limit=200, portal_user_id=7, oldest_first=True
    )


def test_thread_detail_does_not_match_another_session():
    user = {"id": 7, "account_id": "acct"}
    traces = [{
        "id": 2,
        "session_id": "acct:web_7:thread:other",
        "generated_sql": "SELECT 1",
        "status": "success",
    }]
    with patch("portal.routes._get_portal_user", return_value=user), patch(
        "portal.routes.store.list_answer_traces", return_value=traces
    ):
        response = asyncio.run(portal_query_thread(object(), "abc"))

    assert response.status_code == 404


def test_portal_javascript_opens_history_without_rerunning_question():
    source = open("portal/templates/portal_chat.html", encoding="utf-8").read()

    history_block = source[source.index("function openHistoryThread"):source.index("function filterHistory")]
    assert "sendMessage()" not in history_block
    assert "/portal/api/history/" in history_block
    assert "thread_id=${encodeURIComponent(THREAD_ID)}" in source
    assert "SQL is restored from" in source
