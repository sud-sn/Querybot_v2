"""Opt-in "explain your plan first" preview.

A plain-English description of the deterministic entity-graph resolution
QueryBot would use to answer a question, shown BEFORE any SQL is generated
or executed. Reuses the exact resolver the real pipeline already calls
before SQL generation (core.graph_resolver.resolve_for_question) -- this
is NOT a second, separate plan-only LLM call: graph resolution is a pure,
non-LLM, non-DB-touching function, so building this preview is cheap and
side-effect-free.

Explicitly opt-in via a trigger phrase (gateway/webhooks.py's own
_PLAN_PREVIEW_INTENT_RE) -- never a default gate on every question, so the
normal fast, direct-answer path is completely unaffected when this isn't
invoked.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass


def _t(msg_id: str, **kw) -> str:
    """Resolve a catalogue id in the reader's language.

    Deferred import, matching the other producers in core/.
    """
    from core.i18n import t

    return t(msg_id, **kw)


@dataclass(frozen=True)
class PlanPreview:
    question: str
    summary: str
    tables: tuple[str, ...]
    graph_scope: str


def build_plan_preview(question: str, account_id: str, db_type: str) -> PlanPreview:
    """Resolve the deterministic entity-graph plan for `question` and
    describe it in plain English. Never calls the LLM, never touches the
    live database."""
    import store
    from core.graph_resolver import resolve_for_question

    graph = store.get_full_graph(account_id)
    resolution = resolve_for_question(question, account_id, db_type, graph=graph)

    anchor = str(resolution.get("anchor") or "")
    if not resolution.get("enabled") or not anchor:
        return PlanPreview(
            question=question,
            summary=_t("reply.plan.no_match"),
            tables=(),
            graph_scope="",
        )

    detected = [d for d in (resolution.get("detected") or []) if d and d != anchor]
    tables = tuple([anchor] + detected)
    graph_scope = str(resolution.get("graph_scope") or "confirmed")

    # Four whole sentences, not one assembled from clauses. The summary is
    # interpolated into the translated reply.plan.preview_suffix, so building
    # it out of English fragments produced a French sentence with an English
    # clause welded into it -- and a fragment like "joined to" cannot carry
    # French agreement or word order wherever the caller happens to drop it.
    #
    # Table names are tenant DATA: bolded here, interpolated, never looked up.
    unreviewed = ".unreviewed" if graph_scope == "suggested_fallback" else ""
    if detected:
        summary = _t(
            f"reply.plan.joined_tables{unreviewed}",
            anchor=f"**{anchor}**",
            others=", ".join(f"**{t}**" for t in detected))
    else:
        summary = _t(f"reply.plan.single_table{unreviewed}",
                     table=f"**{anchor}**")

    return PlanPreview(
        question=question,
        summary=summary,
        tables=tables,
        graph_scope=graph_scope,
    )


class PendingPlanPreviewStore:
    """In-memory, TTL-based pending-preview tracker, one per
    (account_id, session_id) — mirrors
    core.conversation_state.ConversationStateStore's shape exactly."""

    def __init__(self, ttl_seconds: int | None = None) -> None:
        configured = ttl_seconds
        if configured is None:
            configured = int(os.getenv("PLAN_PREVIEW_TTL_SECONDS", "300"))
        self.ttl_seconds = max(30, int(configured))
        self._pending: dict[tuple[str, str], tuple[PlanPreview, float]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(account_id, session_id) -> tuple[str, str]:
        return str(account_id or ""), str(session_id or "")

    def set(self, account_id, session_id, preview: PlanPreview) -> None:
        key = self._key(account_id, session_id)
        with self._lock:
            self._pending[key] = (preview, time.time() + self.ttl_seconds)

    def get(self, account_id, session_id) -> PlanPreview | None:
        key = self._key(account_id, session_id)
        now = time.time()
        with self._lock:
            entry = self._pending.get(key)
            if entry is None:
                return None
            preview, expires_at = entry
            if expires_at <= now:
                self._pending.pop(key, None)
                return None
            return preview

    def clear(self, account_id, session_id) -> None:
        with self._lock:
            self._pending.pop(self._key(account_id, session_id), None)


pending_plan_previews = PendingPlanPreviewStore()
