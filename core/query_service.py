"""Programmatic access to QueryBot's production question pipeline.

This module deliberately does not reproduce retrieval, prompting, validation,
repair, policy, or execution logic.  It captures the structured response
emitted by :func:`core.query_pipeline.handle_query`, which keeps evaluation and
other internal callers on the same path as the portal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from gateway.base import PlatformEvent


@dataclass
class ProductionQueryResult:
    """Structured outcome captured from one production-pipeline query."""

    sql: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    question_id: str = ""
    assistant_response: dict[str, Any] | None = None
    clarification: dict[str, Any] | None = None
    messages: list[str] = field(default_factory=list)
    statuses: list[dict[str, str]] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return bool(self.assistant_response and self.sql)

    @property
    def error_message(self) -> str:
        if self.completed:
            return ""
        if self.clarification:
            return str(self.clarification.get("question") or "clarification required")
        return self.messages[-1] if self.messages else "production pipeline returned no answer"


class _StructuredCaptureAdapter:
    """Minimal in-memory adapter for the production query pipeline."""

    platform_type = "evaluation"
    persistent_typing = False

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.statuses: list[dict[str, str]] = []
        self.clarification: dict[str, Any] | None = None
        self.assistant_response: dict[str, Any] | None = None
        self.analysis_responses: list[dict[str, Any]] = []

    async def send_message(self, event: PlatformEvent, text: str) -> None:
        if str(text or "").strip():
            self.messages.append(str(text).strip())

    async def send_status(
        self,
        event: PlatformEvent,
        stage: str,
        label: str,
        detail: str = "",
    ) -> None:
        self.statuses.append({"stage": stage, "label": label, "detail": detail})

    async def send_clarification_prompt(
        self,
        event: PlatformEvent,
        question: str,
        options: list,
        pending_id: str | None = None,
    ) -> None:
        self.clarification = {
            "question": question,
            "options": options or [],
            "pending_id": pending_id or "",
        }

    async def send_assistant_response(self, event: PlatformEvent, payload: dict) -> None:
        self.assistant_response = payload

    async def send_analysis_response(self, event: PlatformEvent, payload: dict) -> None:
        self.analysis_responses.append(payload)

    async def send_chart(self, event: PlatformEvent, chart: dict) -> None:
        return None

    async def upload_file(
        self,
        event: PlatformEvent,
        file_bytes: bytes,
        filename: str,
        mime_type: str = "image/png",
    ) -> None:
        return None


def _protected_rows(account_id: str, question_id: str) -> list[dict[str, Any]]:
    """Read the post-policy rows recorded by the production result renderer."""
    if not question_id:
        return []
    try:
        from store.trace_store import get_answer_trace_by_question_id

        trace = get_answer_trace_by_question_id(account_id, question_id) or {}
        value = trace.get("result_rows")
        if isinstance(value, str):
            value = json.loads(value or "[]")
        return value if isinstance(value, list) else []
    except Exception:
        return []


async def run_production_query(
    account_id: str,
    question: str,
    *,
    schema_hint: str = "",
    table_hint: str = "",
    portal_user: dict[str, Any] | None = None,
) -> ProductionQueryResult:
    """Run a question through the exact portal SQL/validation/execution path.

    This is intentionally an execution API, not a SQL-only generator.  Callers
    must opt into database execution before using it (the eval CLI does this by
    requiring both ``--generate`` and ``--execute``).
    """
    from core.query_pipeline import handle_query

    adapter = _StructuredCaptureAdapter()
    event = PlatformEvent(
        account_id=account_id,
        user_id="querybot-evaluation",
        channel_id="querybot-evaluation",
        text=question,
        platform="evaluation",
        raw={"internal_evaluation": True},
        table_hint=table_hint,
        schema_hint=schema_hint,
    )
    await handle_query(account_id, event, adapter, question, portal_user)

    payload = adapter.assistant_response or {}
    trust = payload.get("trust") if isinstance(payload.get("trust"), dict) else {}
    question_id = str(trust.get("question_id") or "")
    rows = _protected_rows(account_id, question_id)
    if not rows:
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        preview = data.get("rows")
        rows = preview if isinstance(preview, list) else []

    return ProductionQueryResult(
        sql=str(trust.get("sql") or "").strip(),
        rows=rows,
        row_count=int(trust.get("row_count") or len(rows)),
        question_id=question_id,
        assistant_response=adapter.assistant_response,
        clarification=adapter.clarification,
        messages=list(adapter.messages),
        statuses=list(adapter.statuses),
    )
