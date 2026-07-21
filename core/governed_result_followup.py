"""Single governed orchestration path for cached-result follow-ups.

The model may plan from result metadata, but cached rows, sample values,
source SQL, and locally bound literals never leave the process. All approved
plans compile to ``ResultCommand`` and execute against the session-local cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from core.result_cache import ResultCache, result_cache
from core.result_commands import (
    ResultCommand,
    ResultCommandOutcome,
    execute_result_command,
    parse_result_command,
)
from core.result_planner import plan_result_command


FollowupStatus = Literal["executed", "unsupported", "blocked", "missing", "error"]


@dataclass(frozen=True)
class GovernedFollowupResult:
    status: FollowupStatus
    command: ResultCommand | None = None
    outcome: ResultCommandOutcome | None = None
    reason: str = ""
    planner_used: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def executed(self) -> bool:
        return self.status == "executed" and bool(self.outcome and self.outcome.ok)


def adopt_cached_snapshot(
    adapter: Any,
    snapshot: dict,
    *,
    question_id: str | None = None,
) -> dict:
    """Synchronize an adapter's compatibility view from the canonical cache.

    ``ResultCache`` owns rows and snapshot lineage. ``last_result`` remains a
    lightweight compatibility view for existing chart, drilldown, and insight
    actions, but no caller should update that view independently.
    """
    adopt = getattr(adapter, "adopt_cached_snapshot", None)
    if callable(adopt):
        return adopt(snapshot, question_id=question_id)

    previous = getattr(adapter, "last_result", None)
    if not isinstance(previous, dict):
        previous = {}
    previous.update({
        "rows": list(snapshot.get("rows") or []),
        "question": str(snapshot.get("question") or previous.get("question") or ""),
        "sql": str(snapshot.get("sql") or previous.get("sql") or ""),
        "column_formats": dict(snapshot.get("column_formats") or {}),
        "result_id": str(snapshot.get("result_id") or ""),
        "result_operation": str(snapshot.get("operation") or "source_query"),
    })
    adapter.last_result = previous
    adapter.last_result_id = previous["result_id"] or None
    if question_id:
        adapter.last_question_id = question_id
    return previous


def _evidence(*, planner_used: bool, planner_metadata: dict | None = None) -> dict[str, Any]:
    metadata = dict(planner_metadata or {})
    return {
        "mode": "metadata_only" if planner_used else "deterministic",
        "planner_used": planner_used,
        "rows_sent_to_llm": 0,
        "sample_values_sent_to_llm": 0,
        "source_sql_sent_to_llm": False,
        "literal_values_sent_to_llm": 0,
        "literal_values_logged": 0,
        "database_queried": False,
        "column_count_disclosed": int(metadata.get("column_count_disclosed") or 0),
        "row_count_disclosed": int(metadata.get("row_count_disclosed") or 0),
        "literal_binding_count": int(metadata.get("literal_binding_count") or 0),
    }


async def run_governed_result_followup(
    question: str,
    session_id: str,
    *,
    complete: Callable[..., Awaitable[tuple[str, int, int]]] | None = None,
    source_result_id: str | None = None,
    cache: ResultCache = result_cache,
) -> GovernedFollowupResult:
    """Plan and execute one result follow-up without disclosing cached values."""
    snapshot = cache.get_snapshot(session_id, source_result_id)
    if not snapshot:
        return GovernedFollowupResult(
            "missing",
            reason="The cached result is no longer available.",
            evidence=_evidence(planner_used=False),
        )

    command = parse_result_command(question)
    if command is not None:
        outcome = execute_result_command(
            session_id,
            command,
            cache=cache,
            source_result_id=str(snapshot.get("result_id") or source_result_id or ""),
        )
        return GovernedFollowupResult(
            "executed" if outcome.ok else "error",
            command=command,
            outcome=outcome,
            reason=outcome.message,
            planner_used=False,
            evidence=_evidence(planner_used=False),
        )

    if complete is None:
        return GovernedFollowupResult(
            "unsupported",
            reason="This follow-up requires metadata planning.",
            evidence=_evidence(planner_used=False),
        )

    planned = await plan_result_command(question, snapshot, complete)
    evidence = _evidence(planner_used=True, planner_metadata=planned.metadata)
    evidence["literal_binding_count"] = planned.binding_count

    if not planned.ok or planned.command is None:
        # A bound literal can be a regulated or identifying value. Never pass
        # the original wording to another model when local compilation fails.
        status: FollowupStatus = "blocked" if planned.binding_count else "unsupported"
        return GovernedFollowupResult(
            status,
            reason=planned.reason,
            planner_used=True,
            evidence=evidence,
        )

    outcome = execute_result_command(
        session_id,
        planned.command,
        cache=cache,
        source_result_id=str(snapshot.get("result_id") or source_result_id or ""),
    )
    return GovernedFollowupResult(
        "executed" if outcome.ok else "error",
        command=planned.command,
        outcome=outcome,
        reason=outcome.message,
        planner_used=True,
        evidence=evidence,
    )
