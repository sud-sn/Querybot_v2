import asyncio
from unittest.mock import patch

from core.query_service import run_production_query


def test_production_query_service_captures_final_sql_and_rows():
    async def fake_handle_query(account_id, event, adapter, question, portal_user):
        assert event.platform == "evaluation"
        assert event.schema_hint == "FINANCE"
        await adapter.send_status(event, "generating", "Generating SQL")
        await adapter.send_assistant_response(
            event,
            {
                "type": "assistant_response",
                "data": {"rows": [{"total": "42"}]},
                "trust": {
                    "sql": "SELECT 42 AS total",
                    "row_count": 1,
                    "question_id": "qid-1",
                },
            },
        )

    with patch("core.query_pipeline.handle_query", fake_handle_query), patch(
        "core.query_service._protected_rows", return_value=[{"total": 42}]
    ):
        result = asyncio.run(
            run_production_query("acct", "total?", schema_hint="FINANCE")
        )

    assert result.completed is True
    assert result.sql == "SELECT 42 AS total"
    assert result.rows == [{"total": 42}]
    assert result.row_count == 1
    assert result.question_id == "qid-1"
    assert result.statuses[0]["stage"] == "generating"


def test_production_query_service_returns_clarification_as_non_completed():
    async def fake_handle_query(account_id, event, adapter, question, portal_user):
        await adapter.send_clarification_prompt(
            event,
            "Which revenue definition?",
            [{"id": "net", "label": "Net revenue"}],
        )

    with patch("core.query_pipeline.handle_query", fake_handle_query):
        result = asyncio.run(run_production_query("acct", "revenue?"))

    assert result.completed is False
    assert result.clarification is not None
    assert result.error_message == "Which revenue definition?"
