"""One correction pass, guided by what actually happened when the SQL ran.

A first attempt fails in two ways this product could already see and did not
act on:

**The database rejected it.** That path exists and feeds the scrubbed driver
message back. What it never carried was the *diagnosis* — ``sanitize_db_error``
has an ordered matcher that turns "Invalid object name" into a plain reason and
a next step, and the repair prompt was handed the raw sentence instead of the
reading of it.

**It ran, and the shape is wrong.** ``core.result_verifier`` already says so —
"A trend requires at least two returned periods", "A ranking requires a numeric
measure column" — and its report was computed only at the very end, to score
confidence. A verifier complaint was a footnote on an answer nobody corrected.

The literature this comes from is specific: execution guidance cuts join and
logic errors by 20-40%, and the first correction pass carries nearly all of the
gain. So exactly one pass, and only when there is something concrete to say.

**Non-destructive, and that is the whole safety argument.** The existing repair
ladder overwrites ``rows`` and ``sql`` and, when a retry fails, leaves the turn
reporting the retry's failure. That is right for an attempt that produced
nothing — there is nothing to lose. It is wrong here: a shape complaint means
the first attempt DID return usable rows, and a correction that fails would
turn a slightly-wrong answer into no answer at all. So a correction is adopted
only when ``is_improvement`` says the new result is strictly better, and the
original is kept otherwise.

Nothing in this module executes SQL, calls a model, or reads the store. It
decides and it phrases; the caller runs.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass

log = logging.getLogger("querybot.execution_correction")

# How many verifier complaints reach the prompt. A model handed nine
# simultaneous demands satisfies the last one; the report lists them in the
# order the verifier found them, which is the order they matter in.
MAX_COMPLAINTS = 4

# A complaint longer than this is a paragraph, and the verifier does not write
# paragraphs -- a long one means something interpolated more than it meant to.
MAX_COMPLAINT_CHARS = 300


@dataclass(frozen=True)
class CorrectionDecision:
    """Whether to spend one correction pass, and what to say if so."""

    should_correct: bool = False
    kind: str = ""
    complaints: tuple[str, ...] = ()
    diagnosis: str = ""
    next_step: str = ""

    def as_dict(self) -> dict:
        return {
            "should_correct": self.should_correct,
            "kind": self.kind,
            "complaints": list(self.complaints),
            "diagnosis": self.diagnosis,
            "next_step": self.next_step,
        }


def _clean_complaints(raw) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for item in raw or []:
        text = " ".join(str(item or "").split())[:MAX_COMPLAINT_CHARS].strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return tuple(out[:MAX_COMPLAINTS])


def diagnose_execution_error(exec_error: str) -> CorrectionDecision:
    """The reading of a database error, beside the error itself.

    ``sanitize_db_error`` already owns the matcher table that turns a driver
    sentence into a plain reason and a next step. The repair prompt was built
    from the raw sentence and never asked for the reading, so the model had to
    re-derive from "Invalid object name 'X'" what the product already knew.

    THE RETURNED TEXT IS NOT SCRUBBED. When no pattern matches,
    ``sanitize_db_error`` falls back to the driver's own first sentence and
    returns it verbatim -- deliberately, so support can search on it. That is
    right for the user's own error card and wrong for a prompt: any caller
    putting ``diagnosis`` or ``next_step`` in front of a model must pass it
    through ``core.failure_messages.scrub_error_for_llm`` first. The repair
    prompt in core/query_pipeline.py did not, and shipped unmasked values to
    the model two lines above the same error being masked.
    """
    if not str(exec_error or "").strip():
        return CorrectionDecision()

    from core.failure_messages import sanitize_db_error

    info = sanitize_db_error(exec_error)
    return CorrectionDecision(
        should_correct=True,
        kind="execution_error",
        diagnosis=str(info.get("plain_reason") or ""),
        next_step=str(info.get("next_step") or ""),
    )


def needs_shape_correction(
    *,
    ok: bool,
    exec_error: str | None,
    rows,
    verification: dict | None,
    already_corrected: bool = False,
) -> CorrectionDecision:
    """Should one pass be spent on a result the verifier objects to?

    Only for a result that is otherwise fine: it validated, the database
    accepted it, and it came back with rows. A result that failed anything
    else is the existing repair ladder's business, and stacking a second
    correction on top of it is how one pass becomes four.

    Warnings are not grounds. The verifier separates them from errors
    deliberately -- "numeric output was returned, but no column is labelled
    with the approved metric name" is a note for the reader, not a defect
    worth spending a model call and a second execution on.
    """
    if already_corrected:
        return CorrectionDecision()
    if not ok or exec_error or not rows:
        return CorrectionDecision()

    report = dict(verification or {})
    if str(report.get("status") or "") == "empty":
        # No rows to verify. The zero-row paths above this one own that case
        # and know things this does not, such as whether a date filter was
        # involved.
        return CorrectionDecision()

    complaints = _clean_complaints(report.get("errors"))
    if not complaints:
        return CorrectionDecision()
    return CorrectionDecision(
        should_correct=True, kind="result_shape", complaints=complaints,
    )


def correction_prompt(
    question: str,
    sql: str,
    decision: CorrectionDecision,
    *,
    scrubbed_error: str = "",
) -> str:
    """The user message for the single correction attempt.

    Says what was asked, what was run, what was wrong and — where the product
    knows one — what to do about it. Deliberately does not restate the schema
    or the knowledge base: those are in the system prompt already, and a repair
    prompt that repeats them buries the one new fact it exists to deliver.
    """
    lines = [
        "The SQL below ran but its result does not answer the question that "
        "was asked."
        if decision.kind == "result_shape"
        else "The SQL below failed when the database ran it.",
        "",
        f"Question: {question}",
        f"SQL: {sql}",
    ]
    if scrubbed_error:
        lines += ["", f"Database error: {scrubbed_error}"]
    if decision.diagnosis:
        lines += ["", f"What that means: {decision.diagnosis}"]
    if decision.next_step:
        lines.append(f"How it is usually fixed: {decision.next_step}")
    if decision.complaints:
        lines += ["", "The result was checked against what was asked and failed on:"]
        lines += [f"- {complaint}" for complaint in decision.complaints]
    lines += [
        "",
        "Rewrite the SQL so the result answers the question. Change only what "
        "the failure requires — keep the same measure, the same filters and "
        "the same date range unless one of them is the problem. Use only "
        "tables and columns from the knowledge base in the system prompt. "
        "Return only the corrected SQL, no explanation.",
    ]
    return "\n".join(lines)


def _error_count(report: dict | None) -> int:
    return len((dict(report or {}).get("errors") or []))


def _score(report: dict | None) -> int:
    try:
        return int(dict(report or {}).get("score") or 0)
    except (TypeError, ValueError):
        return 0


def is_improvement(before: dict | None, after: dict | None,
                   *, after_rows=None) -> tuple[bool, str]:
    """Is the corrected result strictly better than the one it would replace?

    Strictly, because the original is a real answer that a reader can use. The
    bar for throwing it away is not "the new one also works" — it is "the new
    one fixes what was wrong without losing anything".

    Score alone is the wrong test and was the first thing tried: a correction
    can raise the score while leaving every complaint standing, because the
    score also rewards things the complaint is not about.
    """
    if not after_rows:
        return False, "the corrected query returned no rows"

    before_errors, after_errors = _error_count(before), _error_count(after)
    if after_errors > before_errors:
        return False, "the correction introduced new shape failures"
    if after_errors == before_errors:
        return False, "the correction did not resolve the shape failure"
    if _score(after) < _score(before):
        # Fewer complaints but a worse overall report means something else
        # regressed while the named problem was being fixed.
        return False, "the correction fixed one failure and cost more elsewhere"
    return True, f"resolved {before_errors - after_errors} shape failure(s)"


async def run_correction(
    attempt,
    *,
    question: str,
    verify,
    generate,
    execute,
    already_corrected: bool = False,
    on_trace=None,
):
    """The whole correction pass, with its four dependencies handed in.

    ``verify(rows) -> report``       — core.result_verifier.verify_result_shape
    ``generate(prompt) -> sql``      — one LLM call, "" when it cannot
    ``execute(sql) -> attempt``      — validate, repair, run under policy
    ``on_trace(record)``             — optional, called once when a pass ran

    The orchestration lives here rather than in a closure inside
    ``_handle_query_impl`` for one reason: a 6,400-line function cannot be
    called from a test, so the checks on the code inside it become source
    scans — which is exactly what happened to candidate selection before it
    moved out. Everything below is executed by tests against real
    ``verify_result_shape`` reports.

    Returns ``(attempt_to_use, record)``. The record is ``{}`` when no pass was
    spent, so a caller can tell "corrected and discarded" from "never tried".
    """
    if getattr(attempt, "policy_denied", None) is not None:
        return attempt, {}

    try:
        before = verify(getattr(attempt, "rows", None))
    except Exception as exc:  # noqa: BLE001
        # Verification is a check, not a gate: it failing must not cost the
        # answer. It must be LOUD, though -- a silent failure here reads
        # exactly like "the shape was fine".
        log.error("Shape verification before correction failed: %s", exc, exc_info=True)
        return attempt, {}

    decision = needs_shape_correction(
        ok=bool(getattr(attempt, "ok", False)),
        exec_error=getattr(attempt, "exec_error", None),
        rows=getattr(attempt, "rows", None),
        verification=before,
        already_corrected=already_corrected,
    )
    if not decision.should_correct:
        return attempt, {}

    complaints = list(decision.complaints)
    try:
        # The question is passed rather than read off the attempt: Attempt
        # carries the SQL and what happened to it, not what was asked, and a
        # getattr default here would have quietly sent an empty question in a
        # prompt whose whole job is "make this answer the question".
        raw = await generate(correction_prompt(
            question, getattr(attempt, "sql", ""), decision))
    except Exception as exc:  # noqa: BLE001
        log.warning("Shape correction generation failed: %s", exc)
        return attempt, {"attempted": False, "reason": "generation_failed",
                         "complaints": complaints}
    if not raw:
        return attempt, {"attempted": False, "reason": "no_usable_sql",
                         "complaints": complaints}

    try:
        corrected = await execute(raw)
    except Exception as exc:  # noqa: BLE001
        log.warning("Shape correction execution failed: %s", exc)
        return attempt, {"attempted": True, "adopted": False,
                         "reason": "execution_failed", "complaints": complaints}

    try:
        after = verify(getattr(corrected, "rows", None))
    except Exception as exc:  # noqa: BLE001
        log.error("Shape verification after correction failed: %s", exc, exc_info=True)
        return attempt, {"attempted": True, "adopted": False,
                         "reason": "verification_failed", "complaints": complaints}

    adopted, why = is_improvement(before, after,
                                  after_rows=getattr(corrected, "rows", None))
    record = {
        "attempted": True,
        "adopted": adopted,
        "reason": why,
        "complaints": complaints,
        "errors_before": len(before.get("errors") or []),
        "errors_after": len(after.get("errors") or []),
        "score_before": before.get("score"),
        "score_after": after.get("score"),
    }
    if on_trace is not None:
        try:
            on_trace(record)
        except Exception as exc:  # noqa: BLE001
            log.warning("Shape correction trace failed: %s", exc)
    return (corrected if adopted else attempt), record
