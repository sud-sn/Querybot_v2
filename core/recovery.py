"""Read a trace the way the run actually went, not the way it was written.

The pipeline already recovers: a query that fails validation is repaired, and
a repair that exposes a *different* failure code gets one more attempt while
the same code twice is treated as a non-progress loop
(``core.pipeline_helpers.allow_progressive_sql_repair``, budget of two). That
is the bounded re-plan working.

What was missing is how it reads afterwards. Every attempt is written to the
trace with the status it had at the time, so a run that failed twice and then
succeeded looks like two defects and an answer — and an operator scanning
traces for problems finds problems that were already solved. A failed attempt
that a later attempt corrected is a *step in a recovery*, not an unresolved
error, and saying so is the difference between a trace that explains the run
and one that indicts it.

Superseding is presentation only. The original status is preserved on every
step, because the audit record must say what happened and a report that
quietly rewrites failures into successes is worth less than no report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("querybot.recovery")

# Steps that produce a candidate query. These are what the re-plan budget
# counts: one generation plus at most MAX_REPLANS repairs.
ATTEMPT_STEPS = frozenset({
    "llm_generate_sql",
    "sql_repair",
    "progressive_sql_repair",
    "reuse_validated_sql_plan",
    "compile_governed_temporal_metric",
})

# Steps that can be superseded by a later success. A superset of the above,
# because the *outcome* steps -- validation, the zero-row checks -- are where
# a failed attempt actually shows up, and they are what an operator scanning
# for errors sees. They do not count toward the budget: validate_sql runs
# once per attempt, so counting it would make a normal one-repair run read as
# twice over budget.
SUPERSEDABLE_STEPS = ATTEMPT_STEPS | {
    "validate_sql",
    "zero_row_fresh_date_filtered",
    "reused_plan_empty",
}

# The statuses that mean an attempt did not work out.
FAILED_STATUSES = frozenset({"error", "failed", "invalid"})

# What a superseded step's status becomes for display.
SUPERSEDED = "superseded"

# The pipeline's own re-plan budget. Stated here so a reader of a trace and
# the code that enforces it cannot drift apart -- and asserted against the
# enforcing call in the tests.
MAX_REPLANS = 2


@dataclass(frozen=True)
class RecoverySummary:
    """What was retried on this run, and whether it worked."""

    attempts: int = 0
    superseded: int = 0
    recovered: bool = False
    unresolved: tuple[str, ...] = ()
    within_budget: bool = True

    @property
    def clean(self) -> bool:
        """Did this run reach its answer first time, with nothing left failing?

        Unresolved failures count. Without them, a run whose single attempt
        failed outright read as clean -- one attempt, nothing superseded --
        and the reader was told nothing at all about the one run that most
        needed explaining.
        """
        return self.attempts <= 1 and not self.superseded and not self.unresolved


def _status(step: dict) -> str:
    return str((step or {}).get("status") or "success").strip().casefold()


def _name(step: dict) -> str:
    return str((step or {}).get("step_name") or "").strip()


def mark_superseded(steps: list[dict]) -> list[dict]:
    """Relabel failed attempts that a later attempt corrected.

    Returns new dicts; the input is not mutated, and every returned step keeps
    its ``original_status`` so the audit record still says what happened.

    Only failures BEFORE a success are superseded. A failure after the last
    success is the run's actual outcome and stays an error — relabelling that
    would turn a broken run into a tidy one, which is the opposite of the
    point.
    """
    if not steps:
        return []

    last_success = -1
    for index, step in enumerate(steps):
        if _name(step) in SUPERSEDABLE_STEPS and _status(step) not in FAILED_STATUSES:
            last_success = index

    out: list[dict] = []
    for index, step in enumerate(steps):
        item = dict(step)
        item["original_status"] = _status(step)
        item["superseded"] = False
        if (
            index < last_success
            and _name(step) in SUPERSEDABLE_STEPS
            and _status(step) in FAILED_STATUSES
        ):
            item["superseded"] = True
            item["status"] = SUPERSEDED
        out.append(item)
    return out


def summarise(steps: list[dict]) -> RecoverySummary:
    """How much recovery this run needed, and whether it got there.

    ``unresolved`` names the steps that failed and were never corrected —
    the ones an operator should actually look at.
    """
    marked = mark_superseded(steps)
    attempts = sum(1 for step in marked if _name(step) in ATTEMPT_STEPS)
    superseded = sum(1 for step in marked if step.get("superseded"))
    unresolved = tuple(
        _name(step) for step in marked
        if step.get("original_status") in FAILED_STATUSES
        and not step.get("superseded")
    )
    return RecoverySummary(
        attempts=attempts,
        superseded=superseded,
        recovered=bool(superseded and not unresolved),
        unresolved=unresolved,
        # One generation plus at most MAX_REPLANS repairs. More than that
        # means the budget was not enforced on the path that ran.
        within_budget=attempts <= MAX_REPLANS + 1,
    )


def describe(summary: RecoverySummary, *, lang: str = "en") -> str:
    """The one line a trace reader sees about recovery on this run."""
    from core.i18n import t

    if summary.clean:
        return ""
    if summary.recovered:
        return t("recovery.corrected", lang=lang, attempts=summary.attempts)
    if summary.unresolved:
        return t("recovery.unresolved", lang=lang,
                 step=summary.unresolved[0], attempts=summary.attempts)
    return t("recovery.retried", lang=lang, attempts=summary.attempts)


def annotate_trace(trace: dict[str, Any] | None) -> dict[str, Any] | None:
    """A trace with its steps relabelled and a recovery summary attached.

    Best-effort: an unreadable trace is returned as it came rather than
    lost. A trace view that 500s because of a presentation nicety is worse
    than one that shows the raw statuses.
    """
    if not trace:
        return trace
    try:
        steps = list(trace.get("steps") or [])
        annotated = dict(trace)
        annotated["steps"] = mark_superseded(steps)
        summary = summarise(steps)
        annotated["recovery"] = {
            "attempts": summary.attempts,
            "superseded": summary.superseded,
            "recovered": summary.recovered,
            "unresolved": list(summary.unresolved),
            "within_budget": summary.within_budget,
            "clean": summary.clean,
        }
        return annotated
    except Exception as exc:
        log.warning("Trace recovery annotation skipped: %s", exc)
        return trace
