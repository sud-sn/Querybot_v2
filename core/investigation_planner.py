"""
core/investigation_planner.py
──────────────────────────────
Ask several questions instead of one, and prove the summary before it ships.

An objective that needs more than one fact ("why did scrap rise in Q2") gets
one query and one guess today. This runs a small, bounded loop instead: the
first step always asks the objective itself, as a real governed question
(core.investigation.run_query_tool); every step after that is a model's
choice, made from nothing but the labels and figures each prior step
actually found (never the row set), to either ask one more question or stop
and summarize.

The summary is checked before a reader ever sees it, the same discipline
core.situation_phraser applies to a failed turn: every figure it states must
be one a step actually found, every quoted term or column-like identifier
must come from a step or the objective, and it must be short. A summary that
fails the check, or a planner that cannot be reached at all, is replaced by
a template built from the steps alone -- correct, dull, and always available.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from core.investigation import ToolResult

log = logging.getLogger("querybot.investigation_planner")

_MAX_TOKENS = 500
_MAX_SYNTHESIS_CHARS = 900
_PLAN_KEYS = {"action", "question", "reason", "synthesis"}


@dataclass(frozen=True)
class PlannerDecision:
    """What the planner chose to do next, or "" action on a parse failure."""
    action: str = ""
    question: str = ""
    synthesis: str = ""


@dataclass
class InvestigationStep:
    """One step of the trail: what was asked, and what it found or why not."""
    index: int
    question: str
    result: ToolResult


@dataclass
class InvestigationOutcome:
    """The whole run, for the caller to render and log."""
    objective: str
    steps: list[InvestigationStep] = field(default_factory=list)
    synthesis: str = ""
    phrasing: str = "template"  # "llm" once a checked model summary replaces it
    refused: str = ""  # non-empty only when the run never started at all


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:limit]


def _evidence_text(objective: str, steps: list[InvestigationStep]) -> str:
    """Everything a synthesis is allowed to draw a figure or a name from."""
    parts = [objective]
    for step in steps:
        parts.append(step.question)
        parts.append(step.result.brief if step.result.ok else step.result.error)
    return "\n".join(p for p in parts if p)


def build_planner_prompt(
    objective: str, steps: list[InvestigationStep], *, steps_left: int, lang: str | None = None,
) -> tuple[str, str]:
    """The system and user halves of "what next, given what we know so far"."""
    from core.i18n import prompt_language_rule

    system = (
        "You are QueryBot's investigation planner. An objective needs more than "
        "one governed question to answer well; you decide, one step at a time, "
        "what the next question should be, using only the figures and labels "
        "each earlier step actually found -- never invent a figure, a table, "
        "or a column name.\n\n"
        + prompt_language_rule(lang, shape="prose") +
        "Return exactly one JSON object and no markdown. Allowed top-level keys: "
        'action, question, reason, synthesis. action is "query" or "finish". '
        "For \"query\", question is the next business question to ask -- plain "
        "language, no SQL, preserving any time range or filter the objective "
        "named -- and reason is one short sentence on why it helps. For "
        '"finish", synthesis is two to four sentences answering the objective '
        "from what the steps found, citing figures exactly as given and naming "
        "which step each comes from when it helps the reader judge it. If the "
        f"steps so far already answer the objective, finish. {steps_left} more "
        "question(s) may be asked after this decision; finish immediately if "
        "that number is 0."
    )
    lines = [f"Objective: {objective}"]
    for step in steps:
        if step.result.ok:
            lines.append(f"Step {step.index} asked: {step.question}\nFound: {step.result.brief}")
        else:
            lines.append(f"Step {step.index} asked: {step.question}\nCould not be answered: {step.result.error}")
    return system, "\n\n".join(lines)


def _parse_json_object(raw_response: str) -> dict | None:
    import json

    raw = str(raw_response or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def parse_planner_decision(raw_response: str) -> PlannerDecision:
    """The planner's next move, or an empty decision on anything malformed.

    An empty decision (action="") is the caller's signal to finish the
    investigation from its own template rather than trust a reply that did
    not keep to the contract -- the same "verify or fall back" rule the
    synthesis itself is held to.
    """
    plan = _parse_json_object(raw_response)
    if plan is None or set(plan) - _PLAN_KEYS:
        return PlannerDecision()
    action = str(plan.get("action") or "").strip().lower()
    if action == "query":
        question = _clip(plan.get("question"), 400)
        if not question:
            return PlannerDecision()
        return PlannerDecision(action="query", question=question)
    if action == "finish":
        synthesis = _clip(plan.get("synthesis"), _MAX_SYNTHESIS_CHARS)
        if not synthesis:
            return PlannerDecision()
        return PlannerDecision(action="finish", synthesis=synthesis)
    return PlannerDecision()


def verify_synthesis(synthesis: str, objective: str, steps: list[InvestigationStep]) -> tuple[bool, str]:
    """Is ``synthesis`` only a restatement of what the steps actually found?

    Reuses the exact check core.situation_phraser applies to a reworded
    failure: every figure beyond a small, uncomputed-looking number must be
    one a step's brief (or the objective, or a failed step's own reason)
    states, within a rounding step; every quoted term or warehouse-shaped
    identifier must appear in that same evidence.
    """
    from core.situation_phraser import _IDENTIFIER_RE, _QUOTED_RE, _ROUNDING_TOLERANCE
    from core.analysis_narrative import SMALL_NUMBER_CEILING, numbers_in

    synthesis = str(synthesis or "").strip()
    if not synthesis:
        return False, "empty"
    if len(synthesis) > _MAX_SYNTHESIS_CHARS:
        return False, "too_long"
    evidence = _evidence_text(objective, steps)
    known = numbers_in(evidence)
    for value in numbers_in(synthesis):
        if abs(value) <= SMALL_NUMBER_CEILING:
            continue
        if any(abs(value - k) <= _ROUNDING_TOLERANCE for k in known):
            continue
        return False, f"uncomputed_figure:{value:g}"
    folded = evidence.lower()
    for match in _QUOTED_RE.finditer(synthesis):
        term = next((g for g in match.groups() if g), "").strip()
        if term and term.lower() not in folded:
            return False, f"unknown_term:{term[:40]}"
    for ident in _IDENTIFIER_RE.findall(synthesis):
        if ident.lower() not in folded:
            return False, f"unknown_identifier:{ident}"
    return True, ""


def template_synthesis(objective: str, steps: list[InvestigationStep], *, lang: str | None = None) -> str:
    """A correct, dull synthesis built from the steps alone -- no model call.

    The fallback for a rejected or unreachable planner: one line per step,
    in the order they were asked, each showing exactly what it found or why
    it could not be answered.
    """
    from core.i18n import t as _t

    lines = [_t("investigation.template.headline", lang=lang, objective=objective)]
    for step in steps:
        if step.result.ok:
            lines.append(_t(
                "investigation.template.step_found", lang=lang,
                index=step.index, question=step.question, brief=step.result.brief,
            ))
        else:
            lines.append(_t(
                "investigation.template.step_failed", lang=lang,
                index=step.index, question=step.question, error=step.result.error,
            ))
    return "\n".join(lines)


async def run_investigation(
    *,
    objective: str,
    account_id: str,
    portal_user: dict,
    run_id: str,
    max_steps: int,
    complete: Callable[..., Awaitable[tuple[str, int, int]]],
    lang: str | None = None,
    on_step: Callable[[InvestigationStep], None] | None = None,
) -> InvestigationOutcome:
    """Run the whole loop: the objective as step one, the planner after that.

    ``complete`` is the model call (system, user, temperature, max_tokens) ->
    (text, tokens_in, tokens_out) -- the same shape every planner in this
    product takes, so the caller supplies whichever provider/model/api key
    this tenant is configured with.

    A regulated tenant (result_llm_features_allowed_for_investigation) is
    refused before the first question is even asked: every step's brief
    would otherwise reach a planning model, which is exactly what that
    tenant's policy withholds.

    ``on_step``, if given, is called synchronously the moment each step
    finishes -- before the planner is even asked for the next one -- so a
    caller with a UI can show real progress ("asking: ...", "found: ...")
    instead of silence until the whole loop returns. A callback that raises
    is logged and ignored; a UI update must never be able to end the turn a
    real answer was already found in.
    """
    from core.investigation import result_llm_features_allowed_for_investigation, run_query_tool

    objective = _clip(objective, 400)
    max_steps = max(1, int(max_steps))

    if not result_llm_features_allowed_for_investigation(account_id):
        from core.llm_audit import record_llm_blocked

        record_llm_blocked(
            "investigation_planner",
            "Investigation refused before the first question -- regulated "
            "tenant, no step's findings may reach a planning model.",
        )
        from core.i18n import t as _t

        return InvestigationOutcome(objective=objective, refused=_t("investigation.regulated_refusal", lang=lang))

    steps: list[InvestigationStep] = []
    question = objective
    while True:
        result = await run_query_tool(
            account_id=account_id, portal_user=portal_user, run_id=run_id, question=question,
        )
        step = InvestigationStep(index=len(steps) + 1, question=question, result=result)
        steps.append(step)
        if on_step is not None:
            try:
                on_step(step)
            except Exception as exc:  # noqa: BLE001 - a UI hook must never break the loop
                log.warning("Investigation on_step callback failed: %s", exc)
        if len(steps) >= max_steps:
            break
        try:
            system, user = build_planner_prompt(
                objective, steps, steps_left=max_steps - len(steps), lang=lang,
            )
            raw, _tok_in, _tok_out = await complete(
                system=system, user=user, temperature=0.2, max_tokens=_MAX_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001 - an unreachable planner ends the loop, not the turn
            log.warning("Investigation planner unavailable after step %d: %s", len(steps), exc)
            break
        decision = parse_planner_decision(raw)
        if decision.action == "finish":
            ok, why = verify_synthesis(decision.synthesis, objective, steps)
            if ok:
                return InvestigationOutcome(
                    objective=objective, steps=steps, synthesis=decision.synthesis, phrasing="llm",
                )
            log.warning("Investigation synthesis rejected (%s); using the template", why)
            break
        if decision.action != "query":
            log.warning("Investigation planner returned no usable decision after step %d", len(steps))
            break
        question = decision.question

    return InvestigationOutcome(
        objective=objective, steps=steps,
        synthesis=template_synthesis(objective, steps, lang=lang),
        phrasing="template",
    )
