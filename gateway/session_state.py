"""Governed session state shared by stateful chat platform adapters.

The result cache may contain database rows, so its key must always include the
resolved QueryBot account, platform, and platform user. Conversation history
contains structural metadata only; raw row values are never copied into it.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict, deque


log = logging.getLogger("querybot.channel_session")

_HISTORY_MAXLEN = 3
_MAX_HISTORY_SESSIONS = 1000
_history_lock = threading.RLock()
_history_by_session: OrderedDict[str, deque] = OrderedDict()


def _history_for(session_id: str) -> deque:
    with _history_lock:
        history = _history_by_session.get(session_id)
        if history is None:
            history = deque(maxlen=_HISTORY_MAXLEN)
            _history_by_session[session_id] = history
        else:
            _history_by_session.move_to_end(session_id)
        while len(_history_by_session) > _MAX_HISTORY_SESSIONS:
            _history_by_session.popitem(last=False)
        return history


class GovernedChannelSessionMixin:
    """Adds user-isolated history and result caching to a channel adapter."""

    def _init_governed_session(self) -> None:
        self._session_account = ""
        self._session_user = ""
        self.last_result: dict | None = None
        self.last_question_id: str | None = None
        self.last_result_id: str | None = None

    def bind_session(self, account_id: str, user_id: str) -> None:
        account = str(account_id or "").strip()
        user = str(user_id or "").strip()
        if not account or not user:
            raise ValueError("A governed channel session requires account_id and user_id")
        self._session_account = account
        self._session_user = user
        _history_for(self.session_id)

    @property
    def session_id(self) -> str:
        if not self._session_account or not self._session_user:
            return ""
        platform = str(getattr(self, "platform_type", "channel") or "channel").strip().lower()
        return f"{self._session_account}:{platform}:{self._session_user}"

    @staticmethod
    def _sanitize_sql_for_history(sql: str) -> str:
        cleaned = re.sub(r"'[^'\n]*'", "''", str(sql or ""))
        return re.sub(r'"[^"\n]{10,}"', '"[value]"', cleaned)

    def add_to_history(
        self,
        question: str,
        sql: str,
        columns: list[str],
        row_count: int,
    ) -> None:
        if not self.session_id:
            return
        history = _history_for(self.session_id)
        with _history_lock:
            history.append({
                "question": str(question or ""),
                "sql": self._sanitize_sql_for_history(sql),
                "columns": [str(column) for column in (columns or [])],
                "row_count": int(row_count or 0),
            })

    def get_history(self) -> list[dict]:
        if not self.session_id:
            return []
        with _history_lock:
            return [dict(turn) for turn in _history_for(self.session_id)]

    def clear_history(self) -> None:
        if not self.session_id:
            return
        with _history_lock:
            _history_by_session.pop(self.session_id, None)

    def cache_result(
        self,
        rows: list[dict],
        question: str,
        sql: str,
        db_cfg: dict | None = None,
        rag_context: str = "",
        question_id: str | None = None,
        column_formats: dict | None = None,
        data_brief: dict | None = None,
        semantic_plan: dict | None = None,
        contract_version: str = "",
    ) -> None:
        if not self.session_id:
            log.warning("Skipped result cache for unbound %s adapter", getattr(self, "platform_type", "channel"))
            return
        self.last_result = {
            "rows": rows,
            "question": question,
            "sql": sql,
            "db_cfg": db_cfg,
            "rag_context": rag_context,
            "column_formats": column_formats or {},
            "data_brief": data_brief or {},
            "semantic_plan": semantic_plan or {},
            "result_id": "",
        }
        if question_id:
            self.last_question_id = question_id
        try:
            from core.result_cache import result_cache

            cached_result_id = result_cache.store(
                self.session_id,
                rows,
                question,
                sql,
                column_formats=column_formats,
                result_id=question_id,
                metadata={
                    "account_id": self._session_account,
                    "user_id": self._session_user,
                    "channel": str(getattr(self, "platform_type", "channel")),
                    "metadata_contains_raw_values": False,
                    "contract_version": contract_version,
                },
            )
            self.last_result_id = cached_result_id or None
            self.last_result["result_id"] = cached_result_id or ""
        except Exception as exc:
            log.debug("Channel result cache store failed (non-critical): %s", exc)

    def adopt_cached_snapshot(self, snapshot: dict, *, question_id: str | None = None) -> dict:
        previous = self.last_result if isinstance(self.last_result, dict) else {}
        previous.update({
            "rows": list(snapshot.get("rows") or []),
            "question": str(snapshot.get("question") or previous.get("question") or ""),
            "sql": str(snapshot.get("sql") or previous.get("sql") or ""),
            "column_formats": dict(snapshot.get("column_formats") or {}),
            "result_id": str(snapshot.get("result_id") or ""),
            "result_operation": str(snapshot.get("operation") or "source_query"),
        })
        self.last_result = previous
        self.last_result_id = previous["result_id"] or None
        if question_id:
            self.last_question_id = question_id
        return previous
