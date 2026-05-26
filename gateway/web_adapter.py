"""
gateway/web_adapter.py

WebSocket adapter for the internal chat UI.

Implements the same PlatformAdapter interface as Zoom/Teams/Slack.
The bot core (main.py / dispatch / handle_query) never changes — only
the delivery mechanism changes: messages go to a browser WebSocket
instead of the Zoom/Teams API.

Charts are sent as structured JSON so the browser can render them inline
with an interactive library such as ECharts.
"""

import logging
from collections import deque
from typing import Optional

from fastapi import WebSocket
from gateway.base import PlatformAdapter, PlatformEvent

log = logging.getLogger("querybot.web_adapter")

# Max turns of conversation context injected into SQL prompts.
# 3 turns balances context quality vs prompt size.
_HISTORY_MAXLEN = 3


class WebAdapter(PlatformAdapter):
    """Adapter for browser WebSocket connections."""

    platform_type = "web"

    def __init__(self, websocket: WebSocket, account_id: str, user_id: str):
        super().__init__(credentials={})
        self.ws = websocket
        self._account = account_id
        self._user_id = user_id
        # Per-session result cache for action buttons and "why" follow-ups
        self.last_result: dict | None = None
        self.last_question_id: str | None = None   # stable ID linking a question to all its follow-ups

        # ── Conversation history (multi-turn memory) ──────────────────────
        # Stores the last _HISTORY_MAXLEN successful turns so the SQL
        # generation prompt can resolve follow-up references like
        # "filter to top 5" or "break that down by segment".
        # Only populated for web portal sessions — webhook channels are
        # stateless and use separate per-user DB-backed history.
        self._history: deque = deque(maxlen=_HISTORY_MAXLEN)

    # ── Conversation history API ─────────────────────────────────────────

    @staticmethod
    def _sanitize_sql_for_history(sql: str) -> str:
        """
        Strip quoted string literals from SQL before storing in history.

        The audit sanitizer protects the LLM log — this protects the system
        prompt. SQL WHERE clauses can contain literal values typed by users
        or inferred by the LLM (e.g. WHERE CustomerName = 'John Smith').
        We remove these before the SQL goes back into the next query's
        system prompt via conversation history.

        Keeps SQL structure intact (table names, column names, operators,
        numeric literals) so the LLM can still resolve follow-ups like
        "filter to top 5" or "break that down by segment".
        """
        import re as _re
        # Strip single-quoted string literals (SQL values)
        sql = _re.sub(r"'[^'\n]*'", "''", sql)
        # Strip double-quoted literals that look like values (not identifiers)
        # Heuristic: double-quoted values that are all lowercase or mixed case
        # and not SCREAMING_SNAKE_CASE are likely string values not identifiers.
        sql = _re.sub(r'"[^"\n]{10,}"', '"[value]"', sql)
        return sql

    def add_to_history(
        self,
        question: str,
        sql: str,
        columns: list[str],
        row_count: int,
    ) -> None:
        """
        Record a successful query turn in the session history.
        Only the structural metadata is stored — never raw row values.
        SQL is sanitized to strip quoted string literals before storage
        so WHERE clause values from previous queries do not re-enter
        the LLM system prompt on subsequent turns.
        """
        self._history.append({
            "question":  question,
            "sql":       self._sanitize_sql_for_history(sql),
            "columns":   columns,
            "row_count": row_count,
        })
        log.debug(
            "History: +1 turn (total=%d) q=%r cols=%s",
            len(self._history), question[:60], columns,
        )

    def get_history(self) -> list[dict]:
        """Return conversation history oldest-first, excluding the latest
        turn (which hasn't been returned to the user yet)."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear session history — called on WebSocket close."""
        self._history.clear()

    def load_history(self, history: list[dict]) -> None:
        """
        Hydrate conversation history from client-side localStorage.

        Called once on WebSocket connect when the browser sends a
        ``history_sync`` message containing the turns it persisted from the
        previous session.  Only the last _HISTORY_MAXLEN turns are kept.
        SQL is re-sanitized so any value that slipped through on the client
        side is stripped before it re-enters the LLM prompt.
        """
        self._history.clear()
        for turn in list(history)[-_HISTORY_MAXLEN:]:
            if not isinstance(turn, dict):
                continue
            q = (turn.get("question") or "").strip()
            sql = (turn.get("sql") or "").strip()
            if not q or not sql:
                continue
            self._history.append({
                "question":  q,
                "sql":       self._sanitize_sql_for_history(sql),
                "columns":   turn.get("columns") or [],
                "row_count": int(turn.get("row_count") or 0),
            })
        log.debug("History hydrated from client: %d turn(s)", len(self._history))

    @property
    def session_id(self) -> str:
        """Stable session key for the result cache (user-scoped)."""
        return f"{self._account}:{self._user_id}"

    def cache_result(
        self,
        rows: list[dict],
        question: str,
        sql: str,
        db_cfg: dict | None = None,
        rag_context: str = "",
        question_id: str | None = None,
        column_formats: dict | None = None,
    ) -> None:
        """Cache the last query result for insight follow-ups and Tier-2 DuckDB queries."""
        self.last_result = {
            "rows":           rows,
            "question":       question,
            "sql":            sql,
            "db_cfg":         db_cfg,
            "rag_context":    rag_context,
            "column_formats": column_formats or {},
        }
        # Persist the parent question_id so drilldowns can reference it.
        if question_id:
            self.last_question_id = question_id

        # Also populate the module-level DuckDB result cache so follow-up
        # analytical questions ("who is below average?") can be answered
        # from the already-fetched rows without hitting the production DB.
        try:
            from core.result_cache import result_cache
            result_cache.store(
                self.session_id,
                rows,
                question,
                sql,
                column_formats=column_formats,
            )
        except Exception as _ce:
            log.debug("Result cache store failed (non-critical): %s", _ce)

    async def verify_request(self, body: bytes, headers: dict) -> bool:
        return True

    def parse_event(self, body: bytes, headers: dict) -> Optional[PlatformEvent]:
        return None

    def handle_challenge(self, body: bytes) -> Optional[dict]:
        return None

    async def send_message(self, event: PlatformEvent, text: str) -> None:
        try:
            await self.ws.send_json({
                "type": "message",
                "role": "assistant",
                "content": text,
            })
        except Exception as e:
            log.error("WebSocket send_message failed: %s", e)

    async def send_status(self, event: PlatformEvent, stage: str, label: str, detail: str = "") -> None:
        try:
            await self.ws.send_json({
                "type": "status",
                "stage": stage,
                "label": label,
                "detail": detail,
            })
        except Exception as e:
            log.error("WebSocket send_status failed: %s", e)

    async def send_chart(self, event: PlatformEvent, chart: dict) -> None:
        try:
            await self.ws.send_json({
                "type": "chart",
                "role": "assistant",
                "chart": chart,
            })
        except Exception as e:
            log.error("WebSocket send_chart failed: %s", e)

    async def send_assistant_response(self, event: PlatformEvent, payload: dict) -> None:
        try:
            await self.ws.send_json(payload)
        except Exception as e:
            log.error("WebSocket send_assistant_response failed: %s", e)

    async def send_analysis_response(self, event: PlatformEvent, payload: dict) -> None:
        try:
            await self.ws.send_json(payload)
        except Exception as e:
            log.error("WebSocket send_analysis_response failed: %s", e)


    async def send_clarification_prompt(self, event: PlatformEvent, question: str, options: list[dict], pending_id: str | None = None) -> None:
        try:
            await self.ws.send_json({
                "type": "clarification_prompt",
                "question": question,
                "options": options,
                "pending_id": pending_id or "",
            })
        except Exception as e:
            log.error("WebSocket send_clarification_prompt failed: %s", e)

    async def upload_file(
        self,
        event: PlatformEvent,
        file_bytes: bytes,
        filename: str,
        mime_type: str = "image/png",
    ) -> None:
        # Legacy path for non-interactive image uploads. Kept for compatibility.
        try:
            await self.ws.send_json({
                "type": "file_unavailable",
                "role": "assistant",
                "filename": filename,
                "mime_type": mime_type,
            })
        except Exception as e:
            log.error("WebSocket upload_file failed: %s", e)

    def make_event(self, text: str, channel_id: str = "web") -> PlatformEvent:
        return PlatformEvent(
            account_id=self._account,
            user_id=self._user_id,
            channel_id=channel_id,
            text=text,
            platform="web",
            raw={},
        )
