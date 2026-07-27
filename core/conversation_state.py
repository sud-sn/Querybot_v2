"""Metadata-only conversation state and deterministic turn classification.

Result rows remain in the governed result cache. This module keeps only the
handles and schema metadata required to decide whether a message continues the
current analytical thread or starts a new one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
import re
import threading
import time
from typing import Any, Callable, Iterable


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
    updated_at: float = 0.0
    expires_at: float = 0.0


class ConversationStateStore:
    """Tenant-isolated in-memory state with lazy TTL expiry."""

    def __init__(
        self,
        ttl_seconds: int | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        configured = ttl_seconds
        if configured is None:
            configured = int(os.getenv("CONVERSATION_STATE_TTL_SECONDS", "1800"))
        self.ttl_seconds = max(60, int(configured))
        self._clock = clock
        self._states: dict[tuple[str, str], ConversationState] = {}
        self._lock = threading.RLock()

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
            if state is not None and state.expires_at <= now:
                self._states.pop(key, None)
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
            updated_at=now,
            expires_at=now + self.ttl_seconds,
        )
        with self._lock:
            self._states[key] = state
        return state

    def clear(self, account_id: Any, session_id: Any) -> None:
        with self._lock:
            self._states.pop(self._key(account_id, session_id), None)

    def clear_account(self, account_id: Any) -> None:
        account = str(account_id or "")
        with self._lock:
            for key in [key for key in self._states if key[0] == account]:
                self._states.pop(key, None)


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
    r"\b(?:this|that|these|those|it|them|the\s+result|the\s+rows|above|previous)\b",
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


conversation_state_store = ConversationStateStore()
