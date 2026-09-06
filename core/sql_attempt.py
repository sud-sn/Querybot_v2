"""One try at answering a question: validate it, repair what is mechanical, run it.

This was inline in ``core.query_pipeline._handle_query_impl``, and being inline
was the constraint that blocked everything else. A candidate query has to be
*executed* before anything can judge it, and execution sat a thousand lines
below generation through validation and the compliance gate — so there was no
way to produce a second candidate, and no way to scope one to a subject area
without threading a parameter through every frame in between.

Extracted, the sequence is a function of its inputs:

    generate → :func:`validate_with_repairs` → :func:`execute` → :class:`Attempt`

Two things it deliberately does not do. It does not send live stages or write
trace steps — those are the pipeline's, and passing them in as callbacks keeps
this callable from a test without a websocket. And it does not decide anything:
whether an attempt is *good enough* belongs to
``core.candidate_selection``, which compares attempts against each other.

The deterministic repairs are the interesting part of validation. Three
validator codes are mechanically fixable from the plan itself — a governed
period comparison compiled from an approved metric and date role, a display
field swapped for the key it was planned as, an unambiguous column name — and
each is tried before an LLM repair is worth spending. They run in a fixed
order, and each one re-validates: a repair that produces different invalid SQL
is not an improvement.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("querybot.sql_attempt")


@dataclass(frozen=True)
class ValidationScope:
    """Everything the validator and the deterministic repairs need.

    Assembled once per question and passed to every attempt, so two candidates
    are judged against identical scope — which is the whole basis for
    comparing them.
    """

    known_tables: set[str]
    db_type: str
    allowed_tables: set[str] | None = None
    table_columns: dict[str, dict[str, str]] | None = None
    semantic_context: dict | None = None


@dataclass
class Attempt:
    """One candidate query and everything observed about it."""

    sql: str
    source: str = "primary"
    ok: bool = False
    reason: str = ""
    code: str = ""
    rows: list[dict] | None = None
    exec_error: str | None = None
    truncated: bool = False
    timed_out: bool = False
    policy_denied: Any = None
    repairs: tuple[str, ...] = ()
    validate_ms: int = 0
    execute_ms: int = 0

    @property
    def executed(self) -> bool:
        return self.rows is not None

    @property
    def row_count(self) -> int:
        return len(self.rows or [])


# The validator codes a governed temporal recompile can fix. Kept here rather
# than at the call site because the repair and the codes that trigger it are
# one fact.
_TEMPORAL_REPAIR_CODES = frozenset({
    "field_plan_mismatch", "graph_plan_mismatch", "surrogate_date_conversion",
    "temporal_anchor_missing", "temporal_anchor_mismatch", "temporal_role_mismatch",
    "temporal_anchor_unscoped", "observed_period_shape", "source_fact_mismatch",
    "period_comparison_shape", "parse",
})


def validate_with_repairs(
    sql: str,
    scope: ValidationScope,
    *,
    on_repair: Callable[..., None] | None = None,
) -> Attempt:
    """Validate, applying the repairs that are mechanical rather than guesswork.

    ``on_repair(kind, before, after, metadata, duration_ms)`` is called for
    each repair that landed, so the pipeline can trace it -- with the time it
    took, because a repair step recorded at the store layer's default of zero
    is a phase that vanishes from the duration breakdown. Returns an :class:`Attempt` carrying
    the (possibly repaired) SQL and the verdict — never executed.

    Each repair re-validates before it is accepted: SQL that is differently
    invalid is not progress, and accepting it would spend the LLM repair budget
    on a query the deterministic path had already made worse.
    """
    from core.validator import validate_sql

    started = time.time()
    current = sql
    repairs: list[str] = []
    ok, reason, code = validate_sql(
        current, scope.known_tables, scope.db_type, scope.allowed_tables,
        scope.table_columns, scope.semantic_context,
    )

    if not ok and code in _TEMPORAL_REPAIR_CODES:
        repaired = _try_repair(
            "governed_temporal_repair", current, scope,
            {"mode": "deterministic", "date_role": "approved"}, on_repair,
        )
        if repaired:
            current, ok, reason, code = repaired, True, "OK", "ok"
            repairs.append("governed_temporal_repair")

    if not ok and code == "field_plan_mismatch":
        repaired = _try_repair(
            "field_plan_repair", current, scope, {"mode": "deterministic"}, on_repair,
        )
        if repaired:
            current, ok, reason, code = repaired, True, "OK", "ok"
            repairs.append("field_plan_repair")

    if not ok and code == "unknown_column":
        current, ok, reason, code, applied = _repair_unknown_column(
            current, scope, reason, code, on_repair,
        )
        if applied:
            repairs.append("unknown_column_repair")

    return Attempt(
        sql=current, ok=ok, reason=reason, code=code,
        repairs=tuple(repairs),
        validate_ms=int((time.time() - started) * 1000),
    )


def _try_repair(kind, sql, scope, metadata, on_repair) -> str:
    """Run one deterministic repair; "" when it does not apply or fails.

    Fail-open on purpose — a repair that raises must cost its own improvement
    and nothing else — but logged, because a repair that silently never fires
    looks exactly like a repair that never applies.
    """
    from core.pipeline_helpers import (
        attempt_field_plan_repair, attempt_governed_temporal_metric_repair,
    )

    repair = {
        "governed_temporal_repair": attempt_governed_temporal_metric_repair,
        "field_plan_repair": attempt_field_plan_repair,
    }[kind]
    started = time.time()
    try:
        repaired = repair(
            sql, scope.db_type, scope.known_tables, scope.allowed_tables,
            scope.table_columns, scope.semantic_context,
        )
    except Exception as exc:
        log.warning("%s skipped: %s", kind, exc)
        return ""
    if not repaired:
        return ""
    if on_repair:
        on_repair(kind, sql, repaired, metadata,
                  int((time.time() - started) * 1000))
    return repaired


def _repair_unknown_column(sql, scope, reason, code, on_repair):
    """Apply an unambiguous column suggestion, or explain why it is not one.

    A single validator suggestion is deterministic schema evidence, not a
    guess. But a suggestion that moves a column to a DIFFERENT table changes
    the business entity being asked about — prescriber state for pharmacy
    state — so that case becomes a terminal explanation rather than a repair.
    """
    from core.validator import repair_unambiguous_unknown_columns, validate_sql_detailed

    started = time.time()
    try:
        detail = validate_sql_detailed(
            sql, scope.known_tables, scope.db_type, scope.allowed_tables,
            scope.table_columns, scope.semantic_context,
        )
        candidate = repair_unambiguous_unknown_columns(sql, detail, scope.db_type)
    except Exception as exc:
        log.warning("unknown_column repair skipped: %s", exc)
        return sql, False, reason, code, False

    if candidate:
        try:
            revalidated = validate_sql_detailed(
                candidate, scope.known_tables, scope.db_type, scope.allowed_tables,
                scope.table_columns, scope.semantic_context,
            )
        except Exception as exc:
            log.warning("unknown_column re-validation skipped: %s", exc)
            return sql, False, reason, code, False
        if revalidated.ok:
            if on_repair:
                on_repair("unknown_column_repair", sql, candidate,
                          {"mode": "deterministic", "errors": detail.errors},
                          int((time.time() - started) * 1000))
            return candidate, True, "OK", "ok", True
        return sql, False, reason, code, False

    from core.query_pipeline import _entity_field_unavailable_reason

    entity_reason = _entity_field_unavailable_reason(detail.errors)
    if entity_reason:
        return sql, False, entity_reason, "entity_field_unavailable", False
    return sql, False, reason, code, False


async def execute(
    attempt: Attempt,
    *,
    executor: Callable[[str, dict | None], Any],
    semantic_context: dict | None,
    timeout: float,
    timeout_message: str,
) -> Attempt:
    """Run a validated attempt, recording what happened rather than raising.

    Every failure mode the pipeline distinguishes is preserved as state on the
    attempt: a timeout is not an execution error (a repair cannot make the
    database faster), and a policy denial is not a failure of the SQL. Callers
    read the fields; nothing here decides what to do about them.
    """
    if not attempt.ok:
        return attempt

    from core.compliance.governed_query import PolicyDeniedError

    started = time.time()
    try:
        loop = asyncio.get_running_loop()
        governed = await asyncio.wait_for(
            loop.run_in_executor(None, executor, attempt.sql, semantic_context),
            timeout=timeout,
        )
        attempt.rows = governed.rows
        attempt.truncated = bool(getattr(governed, "truncated", False))
        attempt.sql = governed.sql
    except asyncio.TimeoutError:
        attempt.exec_error = timeout_message
        attempt.timed_out = True
        log.warning("Attempt %r timed out", attempt.source)
    except PolicyDeniedError as denied:
        attempt.rows = None
        attempt.exec_error = None
        attempt.ok = False
        attempt.policy_denied = denied.decision
        attempt.reason = denied.decision.explanation or "Blocked by regulated data policy."
        attempt.code = denied.decision.reason_code
    except Exception as exc:
        attempt.exec_error = str(exc)
        log.warning("Attempt %r execution failed: %s", attempt.source, str(exc)[:120])
    attempt.execute_ms = int((time.time() - started) * 1000)
    return attempt


async def run_attempt(
    sql: str,
    scope: ValidationScope,
    *,
    executor: Callable[[str, dict | None], Any],
    timeout: float,
    timeout_message: str,
    source: str = "primary",
    on_repair: Callable[[str, str, str, dict], None] | None = None,
    on_validated: Callable[[Attempt], None] | None = None,
    on_executing: Callable[[], Any] | None = None,
) -> Attempt:
    """Validate (repairing what is mechanical) and execute, in one call.

    The unit a candidate goes through. Two candidates run through identical
    scope and identical repairs, which is what makes comparing them mean
    anything.

    ``on_validated`` fires once the verdict is known and before execution;
    ``on_executing`` fires only when the attempt is about to run, and may be
    a coroutine function (the pipeline uses it to send a live stage). Both are
    the caller's observability, kept out of here so this stays callable from a
    test with no websocket and no trace store.
    """
    attempt = validate_with_repairs(sql, scope, on_repair=on_repair)
    attempt.source = source
    if on_validated is not None:
        try:
            on_validated(attempt)
        except Exception as exc:
            # Observability must never cost the answer -- but it must be
            # visible when it breaks, or a trace that silently stopped
            # recording looks identical to a run that had nothing to record.
            log.warning("on_validated callback failed for %r: %s", source, exc)
    if attempt.ok and on_executing is not None:
        try:
            result = on_executing()
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            log.warning("on_executing callback failed for %r: %s", source, exc)
    return await execute(
        attempt, executor=executor, semantic_context=scope.semantic_context,
        timeout=timeout, timeout_message=timeout_message,
    )
