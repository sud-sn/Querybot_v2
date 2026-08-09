"""Metadata-only conversation state and deterministic turn classification.

Result rows remain in the governed result cache. This module keeps only the
handles and schema metadata required to decide whether a message continues the
current analytical thread or starts a new one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Iterable


log = logging.getLogger("querybot.conversation_state")


class TurnIntent(str, Enum):
    NEW_DATA_QUERY = "new_data_query"
    RESULT_LOCAL_TRANSFORM = "result_local_transform"
    QUERY_REFINEMENT = "query_refinement"
    RESULT_GROUNDED_ANALYSIS = "result_grounded_analysis"
    CLARIFICATION_RESPONSE = "clarification_response"
    GREETING = "greeting"
    SAFE_OFF_TOPIC = "safe_off_topic"
    RESET_CONTEXT = "reset_context"


@dataclass(frozen=True)
class TurnDecision:
    intent: TurnIntent
    confidence: float
    reason: str
    parent_result_id: str = ""
    parent_trace_id: str = ""

    @property
    def uses_prior_result(self) -> bool:
        return self.intent in {
            TurnIntent.RESULT_LOCAL_TRANSFORM,
            TurnIntent.QUERY_REFINEMENT,
            TurnIntent.RESULT_GROUNDED_ANALYSIS,
        }


@dataclass
class ConversationState:
    account_id: str
    session_id: str
    user_id: str = ""
    channel: str = ""
    previous_question: str = ""
    previous_intent: str = ""
    result_id: str = ""
    trace_id: str = ""
    result_schema: tuple[str, ...] = field(default_factory=tuple)
    model_versions: dict[str, str] = field(default_factory=dict)
    # Metadata-only semantic choices confirmed by the user in this thread.
    # Values contain schema identities (table/column/date role), never rows.
    date_preferences: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Calendar interpretation confirmed for this thread. This is deliberately
    # separate from a Date Role: it answers whether "Q1" means calendar or
    # fiscal Q1 and, for fiscal calendars, which month starts the year.
    calendar_preference: dict[str, Any] = field(default_factory=dict)
    updated_at: float = 0.0
    expires_at: float = 0.0


class ConversationStateStore:
    """Tenant-isolated metadata state with optional durable persistence.

    Persisted state never contains result rows or sampled values.  It only
    stores the same handles, output column names, and semantic-version labels
    held in memory.  The governed result cache remains the authority for row
    access; after a restart, a stale result handle therefore helps explain the
    prior turn but never grants access to missing rows.
    """

    def __init__(
        self,
        ttl_seconds: int | None = None,
        *,
        clock: Callable[[], float] = time.time,
        persist: bool = False,
    ) -> None:
        configured = ttl_seconds
        if configured is None:
            configured = int(os.getenv("CONVERSATION_STATE_TTL_SECONDS", "1800"))
        self.ttl_seconds = max(60, int(configured))
        self._clock = clock
        self.persist = bool(persist)
        self._states: dict[tuple[str, str], ConversationState] = {}
        self._lock = threading.RLock()
        self._table_ready = False

    def _ensure_table(self) -> None:
        if not self.persist or self._table_ready:
            return
        try:
            from store.database import get_db

            with get_db() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_thread_state (
                        account_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        user_id TEXT DEFAULT '',
                        channel TEXT DEFAULT '',
                        previous_question TEXT DEFAULT '',
                        previous_intent TEXT DEFAULT '',
                        result_id TEXT DEFAULT '',
                        trace_id TEXT DEFAULT '',
                        result_schema TEXT DEFAULT '[]',
                        model_versions TEXT DEFAULT '{}',
                        date_preferences TEXT DEFAULT '{}',
                        calendar_preference TEXT DEFAULT '{}',
                        updated_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        PRIMARY KEY (account_id, session_id)
                    )
                    """
                )
                # Existing installations may already have the runtime-owned
                # table. Keep this migration local to conversation metadata;
                # no application-wide schema migration is required.
                columns = {
                    str(row[1])
                    for row in conn.execute(
                        "PRAGMA table_info(conversation_thread_state)"
                    ).fetchall()
                }
                if "date_preferences" not in columns:
                    conn.execute(
                        "ALTER TABLE conversation_thread_state "
                        "ADD COLUMN date_preferences TEXT DEFAULT '{}'"
                    )
                if "calendar_preference" not in columns:
                    conn.execute(
                        "ALTER TABLE conversation_thread_state "
                        "ADD COLUMN calendar_preference TEXT DEFAULT '{}'"
                    )
            self._table_ready = True
        except Exception as exc:
            # Conversation persistence improves continuity but must never make
            # answering unavailable when the metadata store is temporarily
            # locked or still migrating.
            log.warning("Conversation-state persistence unavailable: %s", exc)

    @staticmethod
    def _from_row(row: Any) -> ConversationState | None:
        if not row:
            return None
        try:
            schema = json.loads(row["result_schema"] or "[]")
            versions = json.loads(row["model_versions"] or "{}")
            date_preferences = json.loads(row["date_preferences"] or "{}")
            calendar_preference = json.loads(row["calendar_preference"] or "{}")
            return ConversationState(
                account_id=str(row["account_id"] or ""),
                session_id=str(row["session_id"] or ""),
                user_id=str(row["user_id"] or ""),
                channel=str(row["channel"] or ""),
                previous_question=str(row["previous_question"] or ""),
                previous_intent=str(row["previous_intent"] or ""),
                result_id=str(row["result_id"] or ""),
                trace_id=str(row["trace_id"] or ""),
                result_schema=tuple(str(value) for value in schema or ()),
                model_versions={str(k): str(v) for k, v in (versions or {}).items()},
                date_preferences={
                    str(key): dict(value)
                    for key, value in (date_preferences or {}).items()
                    if isinstance(value, dict)
                },
                calendar_preference=(
                    dict(calendar_preference)
                    if isinstance(calendar_preference, dict)
                    else {}
                ),
                updated_at=float(row["updated_at"] or 0),
                expires_at=float(row["expires_at"] or 0),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            log.warning("Ignoring invalid persisted conversation state: %s", exc)
            return None

    def _load_persisted(self, account_id: str, session_id: str) -> ConversationState | None:
        self._ensure_table()
        if not self._table_ready:
            return None
        try:
            from store.database import get_db

            with get_db() as conn:
                row = conn.execute(
                    """
                    SELECT account_id, session_id, user_id, channel,
                           previous_question, previous_intent, result_id,
                           trace_id, result_schema, model_versions,
                           date_preferences, calendar_preference,
                           updated_at, expires_at
                    FROM conversation_thread_state
                    WHERE account_id=? AND session_id=?
                    """,
                    (account_id, session_id),
                ).fetchone()
            return self._from_row(row)
        except Exception as exc:
            log.warning("Could not load persisted conversation state: %s", exc)
            return None

    def _persist(self, state: ConversationState) -> None:
        self._ensure_table()
        if not self._table_ready:
            return
        try:
            from store.database import get_db

            with get_db() as conn:
                conn.execute(
                    """
                    INSERT INTO conversation_thread_state
                        (account_id, session_id, user_id, channel,
                         previous_question, previous_intent, result_id,
                         trace_id, result_schema, model_versions,
                         date_preferences, calendar_preference,
                         updated_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(account_id, session_id) DO UPDATE SET
                        user_id=excluded.user_id,
                        channel=excluded.channel,
                        previous_question=excluded.previous_question,
                        previous_intent=excluded.previous_intent,
                        result_id=excluded.result_id,
                        trace_id=excluded.trace_id,
                        result_schema=excluded.result_schema,
                        model_versions=excluded.model_versions,
                        date_preferences=excluded.date_preferences,
                        calendar_preference=excluded.calendar_preference,
                        updated_at=excluded.updated_at,
                        expires_at=excluded.expires_at
                    """,
                    (
                        state.account_id,
                        state.session_id,
                        state.user_id,
                        state.channel,
                        state.previous_question,
                        state.previous_intent,
                        state.result_id,
                        state.trace_id,
                        json.dumps(list(state.result_schema), separators=(",", ":")),
                        json.dumps(state.model_versions, sort_keys=True, separators=(",", ":")),
                        json.dumps(state.date_preferences, sort_keys=True, separators=(",", ":")),
                        json.dumps(state.calendar_preference, sort_keys=True, separators=(",", ":")),
                        state.updated_at,
                        state.expires_at,
                    ),
                )
        except Exception as exc:
            log.warning("Could not persist conversation state: %s", exc)

    def _delete_persisted(self, account_id: str, session_id: str = "") -> None:
        self._ensure_table()
        if not self._table_ready:
            return
        try:
            from store.database import get_db

            with get_db() as conn:
                if session_id:
                    conn.execute(
                        "DELETE FROM conversation_thread_state WHERE account_id=? AND session_id=?",
                        (account_id, session_id),
                    )
                else:
                    conn.execute(
                        "DELETE FROM conversation_thread_state WHERE account_id=?",
                        (account_id,),
                    )
        except Exception as exc:
            log.warning("Could not clear persisted conversation state: %s", exc)

    @staticmethod
    def _key(account_id: Any, session_id: Any) -> tuple[str, str]:
        return str(account_id or ""), str(session_id or "")

    def get(self, account_id: Any, session_id: Any) -> ConversationState | None:
        key = self._key(account_id, session_id)
        if not all(key):
            return None
        now = self._clock()
        with self._lock:
            state = self._states.get(key)
            if state is None and self.persist:
                state = self._load_persisted(*key)
                if state is not None:
                    self._states[key] = state
            if state is not None and state.expires_at <= now:
                self._states.pop(key, None)
                self._delete_persisted(*key)
                return None
            return state

    def record(
        self,
        account_id: Any,
        session_id: Any,
        *,
        user_id: Any = "",
        channel: Any = "",
        question: str = "",
        decision: TurnDecision | None = None,
        result_id: Any = "",
        trace_id: Any = "",
        result_schema: Iterable[Any] = (),
        result_metadata: dict[str, Any] | None = None,
    ) -> ConversationState | None:
        key = self._key(account_id, session_id)
        if not all(key):
            return None
        now = self._clock()
        prior = self.get(*key)
        versions = dict(prior.model_versions) if prior else {}
        for name in (
            "semantic_version",
            "metric_version",
            "schema_version",
            "contract_version",
        ):
            value = (result_metadata or {}).get(name)
            if value is not None:
                versions[name] = str(value)
        schema = tuple(str(value) for value in result_schema if value is not None)
        state = ConversationState(
            account_id=key[0],
            session_id=key[1],
            user_id=str(user_id or (prior.user_id if prior else "")),
            channel=str(channel or (prior.channel if prior else "")),
            previous_question=str(question or ""),
            previous_intent=(decision.intent.value if decision else ""),
            result_id=str(result_id or (prior.result_id if prior else "")),
            trace_id=str(trace_id or (prior.trace_id if prior else "")),
            result_schema=schema or (prior.result_schema if prior else ()),
            model_versions=versions,
            date_preferences=dict(prior.date_preferences) if prior else {},
            calendar_preference=dict(prior.calendar_preference) if prior else {},
            updated_at=now,
            expires_at=now + self.ttl_seconds,
        )
        with self._lock:
            self._states[key] = state
        self._persist(state)
        return state

    @staticmethod
    def _date_scope_keys(
        metric_names: Iterable[Any] = (),
        fact_tables: Iterable[Any] = (),
    ) -> list[str]:
        metrics = sorted({
            str(value or "").strip().casefold()
            for value in metric_names
            if str(value or "").strip()
        })
        facts = sorted({
            str(value or "").strip().strip("[]`").upper()
            for value in fact_tables
            if str(value or "").strip()
        })
        if metrics and facts:
            return [
                f"metric:{metric}|fact:{fact}"
                for metric in metrics
                for fact in facts
            ]
        if metrics:
            return [f"metric:{metric}" for metric in metrics]
        return [f"fact:{fact}" for fact in facts]

    def get_date_preference(
        self,
        account_id: Any,
        session_id: Any,
        *,
        metric_names: Iterable[Any] = (),
        fact_tables: Iterable[Any] = (),
    ) -> dict[str, Any]:
        """Return a user-confirmed date role scoped to this analytical topic."""
        state = self.get(account_id, session_id)
        if state is None:
            return {}
        for key in self._date_scope_keys(metric_names, fact_tables):
            preference = state.date_preferences.get(key)
            if isinstance(preference, dict) and preference:
                return dict(preference)
        return {}

    def remember_date_preference(
        self,
        account_id: Any,
        session_id: Any,
        binding: dict[str, Any],
        *,
        metric_names: Iterable[Any] = (),
        fact_tables: Iterable[Any] = (),
    ) -> ConversationState | None:
        """Remember schema metadata for the thread without governing the tenant."""
        keys = self._date_scope_keys(metric_names, fact_tables)
        if not keys or not isinstance(binding, dict):
            return self.get(account_id, session_id)
        allowed = {
            "context_name", "aliases", "date_role", "fact_table",
            "fact_column", "dimension_table", "dimension_key",
            "date_value_column", "date_key_type", "governance_status",
        }
        safe_binding = {
            key: value for key, value in binding.items()
            if key in allowed and isinstance(value, (str, int, float, bool, type(None)))
        }
        if not safe_binding.get("fact_table") or not safe_binding.get("fact_column"):
            return self.get(account_id, session_id)
        key = self._key(account_id, session_id)
        now = self._clock()
        with self._lock:
            prior = self.get(*key)
            if prior is None:
                prior = ConversationState(account_id=key[0], session_id=key[1])
            preferences = dict(prior.date_preferences)
            for scope_key in keys:
                preferences[scope_key] = dict(safe_binding)
            prior.date_preferences = preferences
            prior.updated_at = now
            prior.expires_at = now + self.ttl_seconds
            self._states[key] = prior
        self._persist(prior)
        return prior

    def get_calendar_preference(
        self,
        account_id: Any,
        session_id: Any,
    ) -> dict[str, Any]:
        """Return the metadata-only calendar choice for this thread."""
        state = self.get(account_id, session_id)
        if state is None or not isinstance(state.calendar_preference, dict):
            return {}
        return dict(state.calendar_preference)

    def remember_calendar_preference(
        self,
        account_id: Any,
        session_id: Any,
        preference: dict[str, Any],
    ) -> ConversationState | None:
        """Remember a user-confirmed calendar basis without governing a tenant."""
        if not isinstance(preference, dict):
            return self.get(account_id, session_id)
        basis = str(preference.get("basis") or "").strip().casefold()
        if basis not in {"calendar", "fiscal"}:
            return self.get(account_id, session_id)
        safe: dict[str, Any] = {"basis": basis}
        if basis == "fiscal":
            try:
                start_month = int(preference.get("fiscal_year_start_month") or 0)
            except (TypeError, ValueError):
                start_month = 0
            if 1 <= start_month <= 12:
                safe["fiscal_year_start_month"] = start_month
        source = str(preference.get("source") or "user_confirmed").strip()
        if source:
            safe["source"] = source[:40]

        key = self._key(account_id, session_id)
        if not all(key):
            return None
        now = self._clock()
        with self._lock:
            prior = self.get(*key)
            if prior is None:
                prior = ConversationState(account_id=key[0], session_id=key[1])
            prior.calendar_preference = safe
            prior.updated_at = now
            prior.expires_at = now + self.ttl_seconds
            self._states[key] = prior
        self._persist(prior)
        return prior

    def clear(self, account_id: Any, session_id: Any) -> None:
        key = self._key(account_id, session_id)
        with self._lock:
            self._states.pop(key, None)
        self._delete_persisted(*key)

    def clear_account(self, account_id: Any) -> None:
        account = str(account_id or "")
        with self._lock:
            for key in [key for key in self._states if key[0] == account]:
                self._states.pop(key, None)
        self._delete_persisted(account)


_RESET_RE = re.compile(
    r"^\s*(?:start\s+over|new\s+(?:question|topic|analysis)|"
    r"clear\s+(?:the\s+)?context|forget\s+(?:that|the\s+previous\s+result))[\s.!]*$",
    re.I,
)
_GREETING_RE = re.compile(r"^\s*(?:hi|hello|hey|good\s+(?:morning|afternoon|evening))[\s!,.]*$", re.I)
_REFINEMENT_RE = re.compile(
    r"\b(?:instead|only|also|exclude|include|remove|keep|filter|limit|"
    r"sort|order\s+(?:it|these|those)|break\s+(?:it|this|these|that)\s+down|"
    r"group\s+(?:it|this|these|that)\s+by|for\s+(?:jan(?:uary)?|feb(?:ruary)?|"
    r"mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?))\b",
    re.I,
)
_ANALYSIS_RE = re.compile(
    r"\b(?:why|explain|analy[sz]e|what\s+(?:changed|drove|caused)|"
    r"compare|summari[sz]e|insight|contribution|variance|trend)\b",
    re.I,
)
_DEICTIC_RE = re.compile(
    r"\b(?:this|that|these|those|it|them|one|ones|the\s+result|the\s+rows|above|previous)\b",
    re.I,
)
_RESULT_SUPERLATIVE_RE = re.compile(
    r"^\s*(?:which|what)\s+(?:one|ones|row|item)\s+(?:is|are)\s+"
    r"(?:the\s+)?(?:highest|lowest|best|worst|first|second|next)\b",
    re.I,
)


def classify_turn(
    text: str,
    *,
    state: ConversationState | None = None,
    has_cached_result: bool = False,
    direct_result_command: bool = False,
    is_clarification: bool = False,
    looks_like_data: bool = False,
) -> TurnDecision:
    """Classify a turn without exposing result rows to a model."""

    value = str(text or "").strip()
    parent_result_id = state.result_id if state else ""
    parent_trace_id = state.trace_id if state else ""

    def decision(intent: TurnIntent, confidence: float, reason: str) -> TurnDecision:
        return TurnDecision(
            intent=intent,
            confidence=confidence,
            reason=reason,
            parent_result_id=parent_result_id,
            parent_trace_id=parent_trace_id,
        )

    if is_clarification:
        return decision(TurnIntent.CLARIFICATION_RESPONSE, 1.0, "pending clarification")
    if _RESET_RE.match(value):
        return decision(TurnIntent.RESET_CONTEXT, 1.0, "explicit context reset")
    if _GREETING_RE.match(value):
        return decision(TurnIntent.GREETING, 1.0, "greeting")

    has_prior = bool(has_cached_result and parent_result_id)
    if has_prior and direct_result_command:
        return decision(
            TurnIntent.RESULT_LOCAL_TRANSFORM,
            1.0,
            "deterministic governed result command",
        )
    if has_prior and _REFINEMENT_RE.search(value):
        return decision(
            TurnIntent.QUERY_REFINEMENT,
            0.95,
            "refinement language with an active governed result",
        )
    if (
        has_prior
        and _ANALYSIS_RE.search(value)
        and (_DEICTIC_RE.search(value) or len(value.split()) <= 6)
    ):
        return decision(
            TurnIntent.RESULT_GROUNDED_ANALYSIS,
            0.9,
            "analysis language with an active governed result",
        )
    if has_prior and _RESULT_SUPERLATIVE_RE.search(value):
        return decision(
            TurnIntent.RESULT_GROUNDED_ANALYSIS,
            0.92,
            "superlative reference to the active governed result",
        )
    if has_prior and _DEICTIC_RE.search(value):
        return decision(
            TurnIntent.QUERY_REFINEMENT,
            0.82,
            "reference to the active governed result",
        )
    if looks_like_data:
        return decision(TurnIntent.NEW_DATA_QUERY, 0.9, "business-data request")
    return decision(TurnIntent.SAFE_OFF_TOPIC, 0.65, "no governed data intent detected")


def bypass_analyst_gate(decision: TurnDecision) -> bool:
    return decision.intent in {
        TurnIntent.RESULT_LOCAL_TRANSFORM,
        TurnIntent.QUERY_REFINEMENT,
        TurnIntent.RESULT_GROUNDED_ANALYSIS,
        TurnIntent.CLARIFICATION_RESPONSE,
    }


def should_route_as_governed_follow_up(
    text: str,
    *,
    state: ConversationState | None,
    has_cached_result: bool,
) -> bool:
    """Return whether a legacy conversational turn refers to a governed result."""
    return bypass_analyst_gate(
        classify_turn(
            text,
            state=state,
            has_cached_result=has_cached_result,
        )
    )


conversation_state_store = ConversationStateStore(persist=True)
