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

import asyncio
import logging
import re
from collections import deque
from typing import Optional

from fastapi import WebSocket
from gateway.base import PlatformAdapter, PlatformEvent

log = logging.getLogger("querybot.web_adapter")

# Max turns of conversation context injected into SQL prompts.
# 3 turns balances context quality vs prompt size.
_HISTORY_MAXLEN = 3


def _public_clarification_options(options: list[dict] | None) -> list[dict]:
    """Project server-side choices to business-facing browser fields only."""
    public_options: list[dict] = []
    for option in options or []:
        if not isinstance(option, dict):
            continue
        public_options.append({
            key: option.get(key)
            for key in (
                "id", "label", "value", "allow_free_text",
                "business_suggestions",
            )
            if key in option
        })
    return public_options


class WebAdapter(PlatformAdapter):
    """Adapter for browser WebSocket connections."""

    platform_type = "web"

    def __init__(
        self,
        websocket: WebSocket,
        account_id: str,
        user_id: str,
        thread_id: str = "",
        *,
        portal_user_id: int | None = None,
    ):
        super().__init__(credentials={})
        self.ws = websocket
        self._account = account_id
        self._user_id = user_id
        # Channel identities can be synthetic (for example ``web_42``),
        # while dashboard ownership must always use the authenticated portal
        # user's numeric ID. Keep those two identities separate.
        if portal_user_id is not None:
            self._portal_user_id = int(portal_user_id)
        elif str(user_id).isdigit():
            self._portal_user_id = int(user_id)
        else:
            self._portal_user_id = None
        # A browser thread scopes conversation memory and governed result
        # cache independently from every other thread owned by this user.
        cleaned_thread = re.sub(r"[^a-zA-Z0-9_-]", "", str(thread_id or ""))[:80]
        self._thread_id = cleaned_thread or "default"
        # Per-session result cache for action buttons and "why" follow-ups
        self.last_result: dict | None = None
        self.last_question_id: str | None = None   # stable ID linking a question to all its follow-ups
        self.last_result_id: str | None = None
        self.last_response_payload: dict | None = None
        self.pending_dashboard: dict | None = None

        # Serializes every outgoing frame on this connection. The main WS
        # receive loop now runs question handling as a background asyncio
        # task (so it can keep listening for a "cancel" message) — that task
        # sends through this adapter concurrently with the receive loop's own
        # sends. ASGI/Starlette don't guarantee interleaved sends are safe,
        # so every send_json call below (and the loop's own direct sends,
        # sharing this same lock via `adapter.send_lock`) goes through it.
        self.send_lock: asyncio.Lock = asyncio.Lock()

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
        """Stable result-cache key scoped by account, user, and chat thread."""
        return f"{self._account}:{self._user_id}:thread:{self._thread_id}"

    @property
    def thread_id(self) -> str:
        """Sanitized browser thread id used by durable agent ownership."""
        return self._thread_id

    async def send_agent_event(self, payload: dict) -> None:
        """Send a structured agent activity event without changing chat APIs."""
        try:
            async with self.send_lock:
                await self.ws.send_json(payload)
        except Exception as exc:
            log.debug("WebSocket agent event failed: %s", exc)

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
        """Cache the last query result for insight follow-ups and Tier-2 DuckDB queries."""
        self.last_result = {
            "rows":           rows,
            "question":       question,
            "sql":            sql,
            "db_cfg":         db_cfg,
            "rag_context":    rag_context,
            "column_formats": column_formats or {},
            # Stored so compare_prior and other chip actions can read them
            # without recomputing on the follow-up round-trip.
            "data_brief":     data_brief or {},
            "semantic_plan":  semantic_plan or {},
            "contract_version": str(contract_version or ""),
            "result_id":      "",
        }
        # Persist the parent question_id so drilldowns can reference it.
        if question_id:
            self.last_question_id = question_id

        # Also populate the module-level DuckDB result cache so follow-up
        # analytical questions ("who is below average?") can be answered
        # from the already-fetched rows without hitting the production DB.
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
                    "account_id": self._account,
                    "user_id": self._user_id,
                    "metadata_contains_raw_values": False,
                    "contract_version": contract_version,
                },
            )
            self.last_result_id = cached_result_id or None
            self.last_result["result_id"] = cached_result_id or ""
        except Exception as _ce:
            log.debug("Result cache store failed (non-critical): %s", _ce)

    def adopt_cached_snapshot(
        self,
        snapshot: dict,
        *,
        question_id: str | None = None,
    ) -> dict:
        """Refresh the portal compatibility view from a canonical snapshot."""
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

    async def verify_request(self, body: bytes, headers: dict) -> bool:
        return True

    def parse_event(self, body: bytes, headers: dict) -> Optional[PlatformEvent]:
        return None

    def handle_challenge(self, body: bytes) -> Optional[dict]:
        return None

    async def send_message(self, event: PlatformEvent, text: str) -> None:
        try:
            try:
                from core.agent_runtime import get_active_agent_run
                run = get_active_agent_run()
                if run:
                    run.record_assistant_message(text)
            except Exception as exc:
                log.debug("Agent assistant message audit failed: %s", exc)
            async with self.send_lock:
                await self.ws.send_json({
                    "type": "message",
                    "role": "assistant",
                    "content": text,
                })
        except Exception as e:
            log.error("WebSocket send_message failed: %s", e)

    async def send_status(self, event: PlatformEvent, stage: str, label: str, detail: str = "") -> None:
        try:
            agent_event = None
            try:
                from core.agent_runtime import get_active_agent_run
                run = get_active_agent_run()
                if run:
                    agent_event = run.record_stage(stage, label, detail)
            except Exception as exc:
                log.debug("Agent progress persistence failed: %s", exc)
            async with self.send_lock:
                await self.ws.send_json({
                    "type": "status",
                    "stage": stage,
                    "label": label,
                    "detail": detail,
                })
                if agent_event:
                    await self.ws.send_json(agent_event)
        except Exception as e:
            log.error("WebSocket send_status failed: %s", e)

    async def send_chart(self, event: PlatformEvent, chart: dict) -> None:
        try:
            async with self.send_lock:
                await self.ws.send_json({
                    "type": "chart",
                    "role": "assistant",
                    "chart": chart,
                })
        except Exception as e:
            log.error("WebSocket send_chart failed: %s", e)

    def _record_agent_answer_summary(self, payload: dict) -> None:
        """Persist only the answer summary, never raw result rows."""
        try:
            from core.agent_runtime import get_active_agent_run
            run = get_active_agent_run()
            data = payload.get("data") if isinstance(payload, dict) else None
            summary = ""
            if isinstance(data, dict):
                summary = str(
                    data.get("executive_summary")
                    or data.get("summary")
                    or data.get("answer")
                    or ""
                )
            if run and summary:
                run.record_assistant_message(
                    summary,
                    message_type="answer_summary",
                    metadata={"result_id": self.last_result_id or ""},
                )
        except Exception as exc:
            log.debug("Agent answer summary audit failed: %s", exc)

    def _prepare_assistant_response_payload(self, payload: dict) -> dict:
        if payload.get("type") != "assistant_response":
            return payload
        data = payload.get("data")
        if isinstance(data, dict) and self.last_result_id:
            data["result_id"] = self.last_result_id
        payload.setdefault("trust", {})["result_id"] = self.last_result_id or ""
        self.last_response_payload = payload
        if self.pending_dashboard:
            try:
                payload["dashboard_artifact"] = self.materialize_dashboard(
                    payload,
                    name=str(self.pending_dashboard.get("name") or "Analysis dashboard"),
                    dashboard_id=self.pending_dashboard.get("dashboard_id"),
                )
            finally:
                self.pending_dashboard = None
        return payload

    async def send_assistant_response(self, event: PlatformEvent, payload: dict) -> None:
        try:
            payload = self._prepare_assistant_response_payload(payload)
            self._record_agent_answer_summary(payload)
            async with self.send_lock:
                await self.ws.send_json(payload)
        except Exception as e:
            log.error("WebSocket send_assistant_response failed: %s", e)
            # A serialization failure here (e.g. a non-JSON-serializable type
            # sneaking into the payload) used to mean the user saw NOTHING
            # after "query generated" — send a minimal error frame so the
            # failure is at least visible in chat.
            try:
                async with self.send_lock:
                    await self.ws.send_json({
                        "type": "assistant_error",
                        "role": "assistant",
                        "content": (
                            "Something went wrong while preparing your answer — "
                            "please try asking again."
                        ),
                    })
            except Exception:
                pass

    def queue_dashboard(
        self,
        name: str,
        *,
        dashboard_id: int | None = None,
        tab: str = "Overview",
        chart_type: str = "auto",
        title: str = "",
    ) -> None:
        """Create or extend a dashboard after the next governed answer."""
        self.pending_dashboard = {
            "name": str(name or "Analysis dashboard").strip()[:120],
            "dashboard_id": int(dashboard_id) if dashboard_id else None,
            "tab": str(tab or "Overview").strip()[:60],
            "chart_type": str(chart_type or "auto").strip().lower()[:20],
            "title": str(title or "").strip()[:120],
        }

    def materialize_dashboard(
        self,
        payload: dict | None = None,
        *,
        name: str = "Analysis dashboard",
        dashboard_id: int | None = None,
        tab: str = "Overview",
        chart_type_override: str = "auto",
        title_override: str = "",
    ) -> dict:
        """Persist the current governed answer as a dashboard item.

        Only SQL and presentation metadata are stored. Rows remain in the
        short-lived governed result cache and are re-fetched through the
        existing dashboard compliance path when the dashboard is opened.
        """
        response = payload or self.last_response_payload or {}
        result = self.last_result or {}
        rows = list(result.get("rows") or [])
        sql = str(result.get("sql") or "").strip()
        db_cfg = result.get("db_cfg") or {}
        db_config_id = int(db_cfg.get("id") or 0)
        if not rows or not sql or not db_config_id:
            raise ValueError(
                "Run the business question first so I have a governed result to add."
            )

        from store import dashboard_store, user_store

        if self._portal_user_id is None:
            raise ValueError(
                "Authenticated portal user identity is unavailable for this dashboard action."
            )
        user_id = self._portal_user_id
        dashboard = None
        if dashboard_id:
            dashboard = dashboard_store.get_dashboard(
                int(dashboard_id), user_id, self._account
            )
        if dashboard is None:
            dashboard = dashboard_store.create_dashboard(
                self._account,
                user_id,
                self._thread_id,
                name,
                description="Created conversationally from governed QueryBot results.",
            )

        chart = response.get("chart") if isinstance(response.get("chart"), dict) else {}
        queued = self.pending_dashboard or {}
        tab = str(queued.get("tab") or tab or "Overview")[:60]
        chart_type_override = str(
            queued.get("chart_type") or chart_type_override or "auto"
        ).lower()
        title_override = str(queued.get("title") or title_override or "").strip()[:120]
        if chart:
            item_type = str(chart.get("chart_type") or "bar")
            title = str(chart.get("title") or "").strip()
        elif response.get("kpi"):
            item_type = "kpi"
            title = str((response.get("kpi") or {}).get("label") or "").strip()
        else:
            item_type = "table"
            title = ""
        if chart_type_override in {
            "bar", "line", "area", "pie", "donut", "scatter", "table", "kpi"
        }:
            item_type = chart_type_override
        answer = response.get("answer") or {}
        title = (
            title_override
            or title
            or str(answer.get("headline") or "").strip()
            or str(result.get("question") or "Result").strip()
        )[:120]
        kpi = response.get("kpi") if isinstance(response.get("kpi"), dict) else {}
        display_config = {
            "column_formats": dict(
                response.get("column_formats")
                or chart.get("column_formats")
                or {}
            ),
            "display_formats": dict(response.get("display_formats") or {}),
            "kpi": {
                key: kpi.get(key)
                for key in ("label", "format", "display_format")
                if kpi.get(key) is not None
            },
        }

        source = dashboard_store.create_data_source(
            int(dashboard["id"]),
            user_id,
            self._account,
            name=title,
            question=str(result.get("question") or title),
            sql_query=sql,
            db_config_id=db_config_id,
            semantic_contract_version=str(result.get("contract_version") or ""),
        )

        chart_id = user_store.pin_chart(
            user_id=user_id,
            account_id=self._account,
            title=title,
            question=str(result.get("question") or title),
            sql_query=sql,
            chart_type=item_type,
            db_config_id=db_config_id,
            color_palette=str(chart.get("color_palette") or "default"),
            dashboard_id=int(dashboard["id"]),
            display_config=display_config,
        )
        dashboard_store.add_chart(
            int(dashboard["id"]),
            chart_id,
            user_id,
            self._account,
            data_source_id=int(source["id"]),
            tab=tab,
        )
        dashboard = dashboard_store.get_dashboard(
            int(dashboard["id"]), user_id, self._account
        ) or dashboard
        return {
            "id": int(dashboard["id"]),
            "name": dashboard["name"],
            "status": dashboard["status"],
            "version": int(dashboard["version"]),
            "chart_id": chart_id,
            "data_source_id": int(source["id"]),
            "tab": tab,
            "item_type": item_type,
            "url": f"/portal/dashboard?dashboard_id={int(dashboard['id'])}",
        }

    async def send_analysis_response(self, event: PlatformEvent, payload: dict) -> None:
        try:
            async with self.send_lock:
                await self.ws.send_json(payload)
        except Exception as e:
            log.error("WebSocket send_analysis_response failed: %s", e)

    def _record_agent_clarification(self, question: str, options: list[dict]) -> dict | None:
        try:
            from core.agent_runtime import get_active_agent_run
            run = get_active_agent_run()
            if not run:
                return None
            run.record_assistant_message(
                question,
                message_type="clarification",
                metadata={"option_count": len(options or [])},
            )
            return run.record_stage(
                "waiting_for_user", "I need one detail from you", question,
            )
        except Exception as exc:
            log.debug("Agent clarification audit failed: %s", exc)
            return None

    async def send_clarification_prompt(self, event: PlatformEvent, question: str, options: list[dict], pending_id: str | None = None) -> None:
        try:
            agent_event = self._record_agent_clarification(question, options)
            # The browser only needs business-facing choice data. Physical
            # tables, columns, and join metadata remain in the server-side
            # pending clarification and are never exposed as suggestions.
            public_options = _public_clarification_options(options)
            async with self.send_lock:
                await self.ws.send_json({
                    "type": "clarification_prompt",
                    "question": question,
                    "options": public_options,
                    "pending_id": pending_id or "",
                })
                if agent_event:
                    await self.ws.send_json(agent_event)
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
            async with self.send_lock:
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
