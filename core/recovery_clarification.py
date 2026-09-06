"""When the repair ladder runs out, ask instead of giving up.

The pipeline recovers well and then stops. A question that fails validation is
repaired; a repair that exposes a different failure code gets one more attempt;
the same code twice is a non-progress loop and terminal. At that point the turn
ends with a failure card — accurate, well-written, and the end of the
conversation. The reader is told what went wrong and left to work out what to
type next.

Often the product already knows what to ask. A question that dead-ends on
``unknown_column`` has usually named something close to a real metric, and
``suggest_closest_terms`` has already computed which — those suggestions are
printed on the failure card as "did you mean" prose and then thrown away. Turn
them into one focused question with those options and the turn resumes instead
of ending.

Three rules keep this from becoming a machine that interrogates people:

**Only where a person's answer can change the outcome.** A failure whose cause
is a paused database, a policy refusal or a timeout is not a question for the
reader; asking one wastes their time and implies the failure was their fault.
``CLARIFIABLE_CODES`` is a list, not a fallback.

**Never invent an option.** Every option comes from vocabulary the workspace
actually holds. A "did you mean" list of things that do not exist sends the
reader round the loop again and costs the product its credibility on the one
screen where it has already failed once.

**One question, inside the existing budget.** ``can_request_clarification``
owns the round cap and the per-source dedup, and this asks under the source
``recovery`` so it can be asked at most once per turn and never twice for the
same thing.

Nothing here resumes the turn. The reply travels the same road every other
clarification does -- ``core.dispatcher`` resolves it against the options with
``resolve_option_text`` and re-runs through ``combine_with_clarification``,
which wraps the original question with the choice rather than rewriting it. A
second rewriter here would be a second answer to a question already answered,
and the two would drift.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass, field

log = logging.getLogger("querybot.recovery_clarification")

# The clarification source this path asks under. The round cap and the
# already-asked set are keyed on it, so a recovery question cannot repeat and
# cannot be stacked on top of an ambiguity question asked earlier in the turn.
SOURCE = "recovery"

# Failures a person can actually resolve. Deliberately a list rather than a
# default: asking "which did you mean?" about a database outage implies the
# outage was the reader's phrasing, and a product that asks a question it
# cannot use the answer to is worse than one that says plainly that it failed.
CLARIFIABLE_CODES = frozenset({
    "unknown_column",
    "cannot_generate",
    "field_plan_mismatch",
    "metric_formula_mismatch",
    "graph_plan_mismatch",
})

# Below two options there is no choice to offer -- one suggestion is a
# correction the product should be making itself, not a question.
MIN_OPTIONS = 2

# Above this a "did you mean" list is a menu nobody reads.
MAX_OPTIONS = 4


@dataclass(frozen=True)
class RecoveryQuestion:
    """One focused question, or the decision not to ask one."""

    should_ask: bool = False
    question: str = ""
    options: tuple[str, ...] = ()
    reason: str = ""
    code: str = ""

    def meta(self) -> dict:
        """The clarification metadata the pipeline's pending-prompt store wants."""
        return {
            "source": SOURCE,
            "question": self.question,
            "options": [{"label": option, "value": option} for option in self.options],
            "failure_code": self.code,
        }

    def as_dict(self) -> dict:
        return {"should_ask": self.should_ask, "question": self.question,
                "options": list(self.options), "reason": self.reason,
                "code": self.code}


def _clean_options(raw) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for item in raw or []:
        text = " ".join(str(item or "").split())
        if not text or len(text) > 60:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return tuple(out[:MAX_OPTIONS])


def is_clarifiable(code: str) -> bool:
    return str(code or "").strip().casefold() in CLARIFIABLE_CODES


def build(
    *,
    code: str,
    question: str,
    suggestions,
    allowed: bool = True,
    lang: str | None = None,
) -> RecoveryQuestion:
    """Decide whether to ask, and what.

    ``suggestions`` are the vocabulary the workspace holds that is closest to
    what the reader typed -- already computed for the failure card by
    ``core.failure_messages.suggest_closest_terms``. They are the only source
    of options: this module never writes one of its own.

    ``allowed`` is the caller's answer from ``can_request_clarification``.
    Passed in rather than read here because the budget lives on the event, and
    a decision module that reaches for request state is a decision module that
    cannot be tested.
    """
    if not allowed:
        return RecoveryQuestion(reason="the clarification budget for this turn is spent",
                               code=str(code or ""))
    if not is_clarifiable(code):
        return RecoveryQuestion(
            reason="this failure is not one a reader's answer can resolve",
            code=str(code or ""))

    options = _clean_options(suggestions)
    if len(options) < MIN_OPTIONS:
        # One suggestion is a correction the product should make itself, and
        # none is a question with no answers, which is a dead end wearing a
        # question mark.
        return RecoveryQuestion(
            reason="the workspace holds no close vocabulary to offer",
            code=str(code or ""))

    from core.i18n import t

    return RecoveryQuestion(
        should_ask=True,
        question=t("recovery.clarify.question", lang=lang),
        options=options,
        reason=f"{len(options)} close terms available",
        code=str(code or ""),
    )
