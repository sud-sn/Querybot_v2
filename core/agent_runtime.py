"""Bounded, auditable orchestration around the existing governed query path.

This module deliberately does not generate, validate, or execute SQL.  Its
single read-only ``query_data`` tool delegates to the established pipeline and
records enough durable state for a user and an auditor to understand the run.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator

import store
from core.llm_audit import sanitize_llm_text

log = logging.getLogger("querybot.agent_runtime")

_active_run: ContextVar["AgentRunSession | None"] = ContextVar(
    "querybot_active_agent_run", default=None,
)

_BLOCKED_ANSWERS = {
    "policy_denied", "semantic_conflict_blocked", "cache_transform_blocked",
    "authorization_denied", "compliance_blocked",
}
_FAILED_ANSWERS = {
    "error", "timeout", "cannot_generate", "execution_error",
    "validation_error",
}


def _safe_text(value: Any, limit: int = 1200) -> str:
    return sanitize_llm_text(str(value or ""), limit=limit)


def _sanitize_objective(account_id: str, text: str) -> str:
    cleaned = str(text or "")
    try:
        from core.compliance.policy_engine import is_regulated

        profile = store.get_compliance_profile(account_id)
        # Union, not replacement: is_regulated adds unprovisioned tenants, while
        # the mode check keeps any future non-standard posture covered. Scrubbing
        # is a safety measure, so this predicate may widen but must never narrow.
        if is_regulated(account_id) or str(profile.get("mode") or "standard").lower() != "standard":
            from core.masking import scrub_question_pii
            cleaned, _ = scrub_question_pii(
                cleaned, str(profile.get("industry") or ""),
            )
    except Exception as exc:
        log.debug("Agent objective PII scrub skipped: %s", exc)
    return _safe_text(cleaned, 4000)


@dataclass
class AgentRunSession:
    account_id: str
    portal_user_id: int
    external_thread_id: str
    thread_id: str
    run_id: str
    step_index: int = 1
    status: str = "running"
    current_stage: str = "understanding"
    tool_name: str = "query_data"
    label: str = "Understanding your request"
    detail: str = ""

    @classmethod
    def start(
        cls,
        *,
        account_id: str,
        portal_user: dict,
        external_thread_id: str,
        objective: str,
        selected_schema: str = "",
        purpose_id: str = "",
        max_tool_calls: int = 5,
        initial_tool: str = "query_data",
        initial_label: str = "Understanding your request",
        initial_detail: str = "Read-only governed query pipeline",
        initial_metadata: dict | None = None,
    ) -> "AgentRunSession":
        user_id = int(portal_user["id"])
        objective_sanitized = _sanitize_objective(account_id, objective)
        allowed_tables = store.get_allowed_tables(portal_user)
        profile = store.get_compliance_profile(account_id)
        compiler = store.get_semantic_compiler_state(account_id)
        contract_version = str(
            compiler.get("active_contract_version")
            or compiler.get("published_version")
            or ""
        )
        thread = store.ensure_agent_thread(
            account_id=account_id,
            portal_user_id=user_id,
            external_thread_id=external_thread_id,
            title=objective_sanitized[:160],
        )
        run = store.create_agent_run(
            thread_id=thread["id"],
            account_id=account_id,
            portal_user_id=user_id,
            objective_sanitized=objective_sanitized,
            selected_schema=selected_schema,
            purpose_id=purpose_id,
            allowed_tables_snapshot=sorted(allowed_tables or []),
            policy_version=int(profile.get("active_policy_version") or 0),
            contract_version=contract_version,
            max_tool_calls=max_tool_calls,
        )
        session = cls(
            account_id=account_id,
            portal_user_id=user_id,
            external_thread_id=external_thread_id,
            thread_id=thread["id"],
            run_id=run["id"],
            tool_name=_safe_text(initial_tool, 96) or "query_data",
            label=_safe_text(initial_label, 200) or "Understanding your request",
            detail=_safe_text(initial_detail, 500),
        )
        store.add_agent_message(
            thread_id=session.thread_id,
            run_id=session.run_id,
            account_id=account_id,
            portal_user_id=user_id,
            role="user",
            content_sanitized=objective_sanitized,
            metadata={"kind": "objective"},
        )
        store.create_agent_step(
            run_id=session.run_id,
            account_id=account_id,
            portal_user_id=user_id,
            step_index=session.step_index,
            tool_name=session.tool_name,
            label=session.label,
            detail=session.detail,
            metadata={
                "read_only": True,
                "governed": True,
                "tenant_scoped": True,
                "uses_existing_acl": True,
                "uses_existing_semantic_contract": True,
                "uses_existing_sql_validation": True,
                "uses_existing_compliance_policy": True,
                **(initial_metadata or {}),
            },
        )
        return session

    @classmethod
    def resume(
        cls,
        *,
        account_id: str,
        portal_user_id: int,
        external_thread_id: str,
        reply: str,
    ) -> "AgentRunSession | None":
        run = store.get_waiting_agent_run(
            account_id=account_id,
            portal_user_id=portal_user_id,
            external_thread_id=external_thread_id,
        )
        if not run:
            return None
        session = cls(
            account_id=account_id,
            portal_user_id=portal_user_id,
            external_thread_id=external_thread_id,
            thread_id=run["thread_id"],
            run_id=run["id"],
            status="running",
            current_stage="resolving_clarification",
            label="Applying your clarification",
        )
        store.update_agent_run(
            run_id=session.run_id,
            account_id=account_id,
            portal_user_id=portal_user_id,
            status="running",
            current_stage=session.current_stage,
        )
        store.update_agent_step(
            run_id=session.run_id,
            account_id=account_id,
            portal_user_id=portal_user_id,
            step_index=session.step_index,
            status="running",
            label=session.label,
            detail="Clarification received; resuming governed query",
        )
        store.add_agent_message(
            thread_id=session.thread_id,
            run_id=session.run_id,
            account_id=account_id,
            portal_user_id=portal_user_id,
            role="user",
            message_type="clarification",
            content_sanitized=_sanitize_objective(account_id, reply),
            metadata={"kind": "clarification_reply"},
        )
        return session

    def record_stage(self, stage: str, label: str, detail: str = "") -> dict:
        self.current_stage = _safe_text(stage, 96)
        self.label = _safe_text(label, 200)
        self.detail = _safe_text(detail, 500)
        store.update_agent_run(
            run_id=self.run_id,
            account_id=self.account_id,
            portal_user_id=self.portal_user_id,
            status="running",
            current_stage=self.current_stage,
        )
        store.update_agent_step(
            run_id=self.run_id,
            account_id=self.account_id,
            portal_user_id=self.portal_user_id,
            step_index=self.step_index,
            status="running",
            label=self.label,
            detail=self.detail,
        )
        return self.event("agent_run_progress")

    def link_trace(self, trace_id: int) -> None:
        store.update_agent_run(
            run_id=self.run_id,
            account_id=self.account_id,
            portal_user_id=self.portal_user_id,
            answer_trace_id=trace_id,
        )
        store.update_agent_step(
            run_id=self.run_id,
            account_id=self.account_id,
            portal_user_id=self.portal_user_id,
            step_index=self.step_index,
            answer_trace_id=trace_id,
        )

    def record_trace_outcome(self, trace_id: int | None, **fields: Any) -> None:
        answer_type = str(fields.get("answer_type") or "").lower()
        trace_status = str(fields.get("status") or "").lower()
        error = _safe_text(fields.get("error_message") or "", 1000)
        access_block = any(
            phrase in error.lower()
            for phrase in ("no table access", "not authorized", "permission denied", "blocked by")
        )
        if answer_type == "clarification":
            self._finish("waiting_for_user", "waiting_for_user", answer_type, trace_id, error)
        elif answer_type in _BLOCKED_ANSWERS or trace_status == "blocked" or access_block:
            self._finish("blocked", "blocked", answer_type or "policy_blocked", trace_id, error)
        elif answer_type in _FAILED_ANSWERS or trace_status in {"failed", "error"}:
            self._finish("failed", "failed", answer_type or "error", trace_id, error)
        else:
            self._finish("completed", "completed", answer_type or "answer", trace_id, error)

    def record_assistant_message(
        self, text: str, *, message_type: str = "text", metadata: dict | None = None,
    ) -> None:
        cleaned = _sanitize_objective(self.account_id, text)
        if not cleaned:
            return
        store.add_agent_message(
            thread_id=self.thread_id,
            run_id=self.run_id,
            account_id=self.account_id,
            portal_user_id=self.portal_user_id,
            role="assistant",
            message_type=message_type,
            content_sanitized=cleaned,
            metadata=metadata or {},
        )

    def complete_if_running(self) -> None:
        row = store.get_agent_run(
            account_id=self.account_id,
            portal_user_id=self.portal_user_id,
            run_id=self.run_id,
        )
        if row and row.get("status") in {"queued", "planning", "running", "reviewing"}:
            self._finish("completed", "completed", "answer", row.get("answer_trace_id"), "")

    def cancel(self) -> None:
        self._finish("cancelled", "cancelled", "cancelled", None, "Cancelled by user")

    def fail(self, error: Any) -> None:
        self._finish("failed", "failed", "error", None, _safe_text(error, 1000))

    def _finish(
        self,
        run_status: str,
        step_status: str,
        outcome_type: str,
        trace_id: int | None,
        error: str,
    ) -> None:
        self.status = run_status
        self.current_stage = run_status
        store.update_agent_step(
            run_id=self.run_id,
            account_id=self.account_id,
            portal_user_id=self.portal_user_id,
            step_index=self.step_index,
            status=step_status,
            label=self.label,
            detail=error or self.detail,
            answer_trace_id=trace_id,
            policy_reason_code=outcome_type if run_status == "blocked" else None,
        )
        store.update_agent_run(
            run_id=self.run_id,
            account_id=self.account_id,
            portal_user_id=self.portal_user_id,
            status=run_status,
            current_stage=run_status,
            answer_trace_id=trace_id,
            outcome_type=outcome_type,
            error_code=outcome_type if run_status in {"failed", "blocked"} else "",
            error_message=error,
        )

    def event(self, event_type: str) -> dict:
        return {
            "type": event_type,
            "run_id": self.run_id,
            "run_status": self.status,
            "stage": self.current_stage,
            "tool": self.tool_name,
            "tool_label": self.label,
            "detail": self.detail,
            "governed": True,
            "read_only": True,
        }


def get_active_agent_run() -> AgentRunSession | None:
    return _active_run.get()


@contextmanager
def activate_agent_run(session: AgentRunSession | None) -> Iterator[AgentRunSession | None]:
    token = _active_run.set(session)
    try:
        yield session
    finally:
        _active_run.reset(token)
