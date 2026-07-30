"""Single governed orchestration path for cached-result follow-ups.

The model may plan from result metadata, but cached rows, sample values,
source SQL, and locally bound literals never leave the process. All approved
plans compile to ``ResultCommand`` and execute against the session-local cache.
"""

from __future__ import annotations

import re
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


FollowupStatus = Literal[
    "executed", "clarification", "unsupported", "blocked", "missing", "error"
]

# Below this, a compiled-but-uncertain planner interpretation is confirmed
# with the user instead of executed silently. Adjustable; start conservative.
_CONFIDENCE_THRESHOLD = 0.7


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


_RESULT_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}
_NUMBERED_RESULT_RE = re.compile(
    r"\b(?:result|answer|query)\s*(?:number|no\.?|#)?\s*(\d+)\b",
    re.IGNORECASE,
)
_ORDINAL_RESULT_RE = re.compile(
    rf"\b({'|'.join(_RESULT_ORDINALS)})\s+(?:result|answer|query)\b",
    re.IGNORECASE,
)
_PREVIOUS_RESULT_RE = re.compile(
    r"\b(?:the\s+)?(?:previous|prior)\s+(?:result|answer|query)\b"
    r"|\bthe\s+one\s+before\b",
    re.IGNORECASE,
)
_EARLIER_RESULT_RE = re.compile(
    r"\b(?:the\s+)?earlier\s+(?:result|answer|query)\b",
    re.IGNORECASE,
)
_CURRENT_RESULT_RE = re.compile(
    r"\b(?:the\s+)?(?:current|latest|last|this|that)\s+"
    r"(?:result|answer|query)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _ResultReference:
    question: str
    result_id: str | None
    clarification: ResultCommandOutcome | None = None


def _short_question(value: str, limit: int = 72) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else f"{text[: limit - 3].rstrip()}..."


def _reference_clarification(
    question: str,
    summaries: list[dict],
    match: re.Match[str] | None,
    message: str,
) -> _ResultReference:
    options: list[dict[str, str]] = []
    for index, summary in enumerate(summaries, start=1):
        reference = f"result {index}"
        resolved = (
            f"{question[:match.start()]}{reference}{question[match.end():]}"
            if match is not None
            else f"{question} using {reference}"
        )
        options.append({
            "id": reference.replace(" ", "-"),
            "label": f"Result {index}: {_short_question(summary.get('question', ''))}",
            "value": reference,
            "resolved_question": resolved.strip(),
        })
    outcome = ResultCommandOutcome(
        handled=True,
        ok=False,
        message=message,
        clarification_required=True,
        clarification_prompt=message,
        clarification_options=options,
    )
    return _ResultReference(question=question, result_id=None, clarification=outcome)


def _describe_command(command: ResultCommand) -> str:
    """Plain-English restatement of a compiled command for a confirmation
    prompt -- column names and an already-locally-resolved filter value are
    safe to show back to the same user who is about to confirm them; this
    never leaves the process."""
    if command.action == "filter":
        return f"filter to rows where {command.target_text} {command.operator} {command.value_text}"
    if command.action == "aggregate":
        return f"{command.aggregation} of {command.metric_text} grouped by {command.dimension_text}"
    if command.action == "contribution":
        return f"show each {command.dimension_text}'s contribution to {command.metric_text}"
    if command.action == "ratio":
        return f"{command.numerator_text} divided by {command.denominator_text}"
    if command.action == "profit_percentage":
        return "compute profit percentage"
    if command.action == "sort":
        return f"sort by {command.target_text} ({command.direction})"
    if command.action == "keep_top":
        suffix = f" by {command.metric_text}" if command.metric_text else ""
        return f"keep the top {command.limit} rows{suffix}"
    return command.action


def _confidence_clarification(question: str, command: ResultCommand) -> ResultCommandOutcome:
    """Confirm an uncertain planner interpretation instead of executing it
    silently. The single option's resolved_question is the ORIGINAL question
    text unchanged -- on confirm, the dispatcher's clarification-reply path
    re-submits it with is_clarification=True, which skips this gate on that
    retry (see run_governed_result_followup) rather than risking an
    infinite low-confidence loop on a near-deterministic (temperature=0.0)
    re-plan of the identical text."""
    description = _describe_command(command)
    prompt = f"Did you mean: {description}?"
    return ResultCommandOutcome(
        handled=True,
        ok=False,
        message=prompt,
        clarification_required=True,
        clarification_prompt=prompt,
        clarification_options=[{
            "id": "confirm",
            "label": f"Yes — {description}",
            "value": "confirm",
            "resolved_question": question,
        }],
    )


def _resolve_result_reference(
    question: str,
    session_id: str,
    source_result_id: str | None,
    cache: ResultCache,
) -> _ResultReference:
    """Resolve conversational result references without reading cached values."""
    summaries = cache.list_snapshot_summaries(session_id)
    if not summaries:
        return _ResultReference(question=question, result_id=source_result_id)

    numbered = _NUMBERED_RESULT_RE.search(question)
    ordinal = _ORDINAL_RESULT_RE.search(question)
    previous = _PREVIOUS_RESULT_RE.search(question)
    earlier = _EARLIER_RESULT_RE.search(question)
    current = _CURRENT_RESULT_RE.search(question)
    match = numbered or ordinal or previous or earlier or current
    if match is None:
        return _ResultReference(
            question=question,
            result_id=source_result_id or str(summaries[-1]["result_id"]),
        )

    index: int | None = None
    if numbered:
        index = int(numbered.group(1)) - 1
    elif ordinal:
        index = _RESULT_ORDINALS[ordinal.group(1).lower()] - 1
    elif previous:
        current_index = len(summaries) - 1
        if source_result_id:
            current_index = next(
                (
                    idx
                    for idx, item in enumerate(summaries)
                    if item["result_id"] == source_result_id
                ),
                current_index,
            )
        index = current_index - 1
    elif earlier:
        candidates = summaries[:-1]
        if len(candidates) == 1:
            index = 0
        else:
            return _reference_clarification(
                question,
                candidates,
                earlier,
                "Which earlier result do you mean?",
            )
    else:
        index = len(summaries) - 1
        if source_result_id and current and current.group(0).lower().strip().startswith(
            ("this", "that", "the current")
        ):
            index = next(
                (
                    idx
                    for idx, item in enumerate(summaries)
                    if item["result_id"] == source_result_id
                ),
                index,
            )

    if index is None or index < 0 or index >= len(summaries):
        return _reference_clarification(
            question,
            summaries,
            match,
            "That result is not available. Which cached result should I use?",
        )

    normalized = f"{question[:match.start()]}this result{question[match.end():]}".strip()
    return _ResultReference(
        question=normalized,
        result_id=str(summaries[index]["result_id"]),
    )


async def run_governed_result_followup(
    question: str,
    session_id: str,
    *,
    complete: Callable[..., Awaitable[tuple[str, int, int]]] | None = None,
    source_result_id: str | None = None,
    cache: ResultCache = result_cache,
    is_clarification: bool = False,
) -> GovernedFollowupResult:
    """Plan and execute one result follow-up without disclosing cached values."""
    reference = _resolve_result_reference(
        question,
        session_id,
        source_result_id,
        cache,
    )
    if reference.clarification is not None:
        return GovernedFollowupResult(
            "clarification",
            outcome=reference.clarification,
            reason=reference.clarification.message,
            evidence=_evidence(planner_used=False),
        )

    source_result_id = reference.result_id
    question = reference.question
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
        status: FollowupStatus = (
            "clarification"
            if outcome.clarification_required
            else ("executed" if outcome.ok else "error")
        )
        return GovernedFollowupResult(
            status,
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

    # A clarification reply confirming this exact interpretation already
    # went through this gate once; the confidence check exists to catch
    # guesses before they execute, not to re-litigate a choice the user
    # just made.
    if not is_clarification and planned.confidence < _CONFIDENCE_THRESHOLD:
        clarification_outcome = _confidence_clarification(question, planned.command)
        evidence["planner_confidence"] = planned.confidence
        return GovernedFollowupResult(
            "clarification",
            command=planned.command,
            outcome=clarification_outcome,
            reason=clarification_outcome.message,
            planner_used=True,
            evidence=evidence,
        )

    outcome = execute_result_command(
        session_id,
        planned.command,
        cache=cache,
        source_result_id=str(snapshot.get("result_id") or source_result_id or ""),
    )
    status = (
        "clarification"
        if outcome.clarification_required
        else ("executed" if outcome.ok else "error")
    )
    return GovernedFollowupResult(
        status,
        command=planned.command,
        outcome=outcome,
        reason=outcome.message,
        planner_used=True,
        evidence=evidence,
    )
