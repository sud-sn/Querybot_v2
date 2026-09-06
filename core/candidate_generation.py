"""Ask the same question a second way, when the plan left the choice open.

``core.analytical_request_plan`` records what it chose *between*
(``considered_facts``, ``considered_date_roles``, ``considered_metrics``), and
``core.candidate_selection.ambiguity_of`` reads those to say when a plan was a
coin-flip. This module turns that into work: one extra generation per open
decision, each pinned to the alternative the compiler discarded.

A variant is the same prompt with one constraint appended. That is
deliberate — a variant built from a different prompt would differ in ways the
verifier cannot attribute, and the whole point of running two is that the only
difference between them is the decision under test.

Directives are prompt text, not user copy: they go to a model, never to a
reader, so they stay English and are not in the message catalogue.

Nothing here calls a model. It says *what* to ask for; the pipeline owns the
call, because the generation call carries the audit scope, the provider
routing and the token budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("querybot.candidate_generation")


@dataclass(frozen=True)
class CandidateSpec:
    """One way of asking: a label for the trace, and what to constrain."""

    source: str
    directive: str = ""
    dimension: str = ""
    choice: str = ""

    @property
    def is_primary(self) -> bool:
        return not self.directive


PRIMARY = CandidateSpec(source="primary")


def _alternatives(plan: dict, key: str, chosen: str) -> list[str]:
    """The values the compiler weighed and did not pick, in a stable order."""
    values = [str(v).strip() for v in (plan.get(key) or []) if str(v).strip()]
    chosen_key = str(chosen or "").strip().casefold()
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        folded = value.casefold()
        if folded == chosen_key or folded in seen:
            continue
        seen.add(folded)
        out.append(value)
    return out


def _primary_date_role(plan: dict) -> str:
    for policy in plan.get("temporal_operations") or []:
        if isinstance(policy, dict):
            role = str(policy.get("date_role") or policy.get("column") or "").strip()
            if role:
                return role
    return ""


def _primary_metric(plan: dict) -> str:
    for metric in plan.get("metrics") or []:
        name = str((metric or {}).get("name") or "").strip()
        if name:
            return name
    return ""


def specs_for(request_plan: dict | None, *, limit: int = 3) -> list[CandidateSpec]:
    """The candidates worth generating for this plan, primary first.

    Returns exactly ``[PRIMARY]`` when the plan is determined — which is most
    questions, and the reason this does not multiply the cost of the product.
    Ordered so the first alternatives generated are the ones on the decision
    the plan is least sure about: a different business date changes the answer
    more often than a different metric name.
    """
    plan = dict(request_plan or {})
    specs: list[CandidateSpec] = [PRIMARY]

    for role in _alternatives(plan, "considered_date_roles", _primary_date_role(plan)):
        specs.append(CandidateSpec(
            source=f"variant:date_role={role}",
            dimension="date_role", choice=role,
            directive=(
                f"BUSINESS DATE OVERRIDE: anchor every date filter and every "
                f"period grouping on {role}. Do not use any other date column "
                f"for the period this question asks about."
            ),
        ))

    for fact in _alternatives(plan, "considered_facts", str(plan.get("source_fact") or "")):
        specs.append(CandidateSpec(
            source=f"variant:fact={fact}",
            dimension="fact", choice=fact,
            directive=(
                f"SOURCE FACT OVERRIDE: measure this question from {fact}. "
                f"Join to dimensions from there using only approved "
                f"relationships; do not measure from another fact table."
            ),
        ))

    for metric in _alternatives(plan, "considered_metrics", _primary_metric(plan)):
        specs.append(CandidateSpec(
            source=f"variant:metric={metric}",
            dimension="metric", choice=metric,
            directive=(
                f"METRIC OVERRIDE: the measure this question asks for is "
                f"\"{metric}\". Use that metric's approved formula and its "
                f"source table."
            ),
        ))

    if len(specs) > limit:
        log.info("Capping %d candidate specs to %d", len(specs), limit)
    return specs[:limit]


def variant_user_message(question: str, spec: CandidateSpec) -> str:
    """The user half of a variant's generation call.

    The same question with one constraint appended, so the only difference
    between two candidates is the decision under test. A variant built from a
    different prompt would differ in ways the verifier cannot attribute, which
    would make comparing them meaningless.
    """
    if spec.is_primary:
        return question
    return f"{question}\n\n{spec.directive}"
