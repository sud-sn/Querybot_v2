"""Choose between SQL candidates by running them, not by trusting the first.

The pipeline generates one query and accepts it if it validates. When the
semantic plan leaves a genuine choice open — an ambiguous grain, two
reachable date roles, two candidate join paths — that first query is a guess
the product then presents with full confidence.

The literature on this is unusually clear. Execution-guided selection and
self-correction is where the accuracy is: LitE-SQL reports 72.1% on BIRD with
vector schema linking plus execution-guided self-correction, and execution
guidance is reported to cut schema-linking, join and logic errors by 20-40%.
The same literature is equally clear that **agreement is a weak selector** —
the most-agreed answer is often not the correct one, with an upper bound
around 14% above what majority voting achieves. So candidates are ranked by a
*verifier*, and agreement only breaks ties between candidates the verifier
already ranked equally.

We already own the verifier. ``core.result_verifier.verify_result_shape``
compares metadata-only intent against the executed shape, never calls an LLM
and never logs a value; using it as the selector rather than as a post-hoc
confidence signal is the whole change.

Two rules keep this honest:

* **Cost is spent only where there is real ambiguity.** ``ambiguity_of`` reads
  the compiled plan; when the plan is determined — most questions — one
  candidate is generated and nothing changes.
* **When nothing verifies, clarify.** Picking the least-bad candidate from a
  set that all disagree is how a product returns a confident wrong number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("querybot.candidate_selection")

# How many candidates to generate when the plan is ambiguous. Three is enough
# for a verifier to have something to choose between and cheap enough that an
# ambiguous question costs roughly what a retry already costs.
DEFAULT_CANDIDATE_COUNT = 3
MAX_CANDIDATE_COUNT = 5

# A verification score at or above this is a candidate the verifier is happy
# with. Below it the candidate ran and returned rows but does not match what
# was asked for.
VERIFIED_SCORE = 70

# Two headline figures within this relative distance count as agreeing.
# Deliberately tight: agreement is a tie-breaker, and a loose band would let
# two materially different answers vote for each other.
AGREEMENT_TOLERANCE = 0.005


# ══════════════════════════════════════════════════════════════════════════════
# What makes a plan ambiguous
# ══════════════════════════════════════════════════════════════════════════════

# Each entry is a key on the compiled analytical request plan and what it
# means for that key to hold more than one distinct value. These are the
# `considered_*` lists core.analytical_request_plan emits -- what the compiler
# chose BETWEEN, which it used to compute and discard. Kept as a table rather
# than a chain of ifs so a new source of ambiguity is one line, and so each
# reason can be named in the trace.
_AMBIGUITY_KEYS: tuple[tuple[str, str], ...] = (
    ("considered_facts", "fact"),
    ("considered_date_roles", "date_role"),
    ("considered_metrics", "metric"),
    ("candidate_grains", "grain"),
    ("candidate_join_paths", "join_path"),
)


def ambiguity_of(request_plan: dict | None) -> list[str]:
    """Which decisions the compiled plan left open, if any.

    An empty list means the plan is determined and one candidate is correct
    by construction — which is the common case, and the reason this does not
    triple the cost of the product.
    """
    plan = dict(request_plan or {})
    reasons: list[str] = []
    for key, label in _AMBIGUITY_KEYS:
        values = plan.get(key)
        if isinstance(values, (list, tuple, set)) and len(set(map(str, values))) > 1:
            reasons.append(label)
    # An explicit flag from the compiler outranks the heuristics above: it
    # knows things the candidate lists do not carry.
    if plan.get("ambiguous"):
        reasons.append(str(plan.get("ambiguity_reason") or "compiler"))
    return reasons


def candidate_count_for(request_plan: dict | None, *, cap: int = DEFAULT_CANDIDATE_COUNT) -> int:
    """How many candidates this question is worth.

    One when the plan is determined. Otherwise one per open decision, capped —
    a plan with four open decisions is a plan that should have asked a
    clarifying question, not one worth sixteen queries.
    """
    reasons = ambiguity_of(request_plan)
    if not reasons:
        return 1
    return max(2, min(int(cap), MAX_CANDIDATE_COUNT, len(reasons) + 1))


# ══════════════════════════════════════════════════════════════════════════════
# A candidate
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Candidate:
    """One SQL attempt and everything observed about it."""

    sql: str
    source: str = "primary"
    validated: bool = False
    validation_reason: str = ""
    executed: bool = False
    error: str = ""
    rows: list[dict] = field(default_factory=list)
    verification: dict = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """Did this candidate validate, execute, and return something?"""
        return bool(self.validated and self.executed and not self.error)

    @property
    def score(self) -> int:
        try:
            return int(self.verification.get("score") or 0)
        except (TypeError, ValueError):
            return 0

    @property
    def verified(self) -> bool:
        return self.usable and self.score >= VERIFIED_SCORE

    @property
    def row_count(self) -> int:
        return len(self.rows or [])

    def headline(self) -> float | None:
        """The single figure this candidate is really asserting, if there is one.

        The first numeric cell of the first row: for a scalar answer that is
        the answer, and for a grouped answer it is the leading group's
        measure. Used only to break ties between candidates the verifier
        ranked equally — never to choose one.
        """
        for row in (self.rows or [])[:1]:
            for value in row.values():
                try:
                    number = float(str(value).replace(",", ""))
                except (TypeError, ValueError):
                    continue
                if number == number:      # not NaN
                    return number
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Selection
# ══════════════════════════════════════════════════════════════════════════════

def agreeing_group(candidates: list[Candidate]) -> list[Candidate]:
    """The largest set of candidates asserting the same headline figure."""
    with_headline = [(c, c.headline()) for c in candidates]
    with_headline = [(c, h) for c, h in with_headline if h is not None]
    best: list[Candidate] = []
    for _, anchor in with_headline:
        scale = max(abs(anchor), 1.0)
        group = [
            c for c, h in with_headline
            if abs(h - anchor) / scale <= AGREEMENT_TOLERANCE
        ]
        if len(group) > len(best):
            best = group
    return best


def rank(candidates: list[Candidate]) -> list[Candidate]:
    """Usable candidates, best first.

    Ordered by the verifier's score, then by whether the candidate returned
    rows at all, then by a stable key. Agreement is deliberately NOT in this
    ordering — it settles ties in ``select`` and nothing more, because the
    most-agreed answer is often not the right one.
    """
    usable = [c for c in candidates if c.usable]
    return sorted(
        usable,
        key=lambda c: (-c.score, 0 if c.row_count else 1, c.source, c.sql),
    )


def select(candidates: list[Candidate]) -> tuple[Candidate | None, str]:
    """Pick the candidate to answer with, or nothing and a reason to clarify.

    Returns ``(chosen, reason)``. ``chosen`` is None when the product should
    ask rather than answer — which is the outcome whenever no candidate
    satisfies the verifier. Presenting the least-bad member of a set that all
    disagree is exactly how a confident wrong number reaches a user.
    """
    if not candidates:
        return None, "no_candidates"

    usable = rank(candidates)
    if not usable:
        failed = next((c for c in candidates if c.error), None)
        if failed is not None:
            return None, f"all_candidates_failed:{failed.error[:120]}"
        invalid = next((c for c in candidates if not c.validated), None)
        if invalid is not None:
            return None, f"all_candidates_invalid:{invalid.validation_reason[:120]}"
        return None, "all_candidates_failed"

    verified = [c for c in usable if c.verified]
    if not verified:
        if len(usable) == 1:
            # Nothing to choose between, and the single-candidate path is the
            # pipeline's existing behaviour. Returning it with a reason keeps
            # the trace honest without changing what a determined plan does.
            return usable[0], "single_candidate_unverified"
        return None, "no_candidate_verified"

    if len(verified) == 1:
        return verified[0], "single_verified_candidate"

    top_score = verified[0].score
    tied = [c for c in verified if c.score == top_score]
    if len(tied) == 1:
        return tied[0], "highest_verification_score"

    agreeing = agreeing_group(tied)
    if len(agreeing) > 1:
        # Ordered by rank already, so the first of the agreeing set is the
        # best-verified member of it.
        for candidate in tied:
            if candidate in agreeing:
                return candidate, f"agreement_of_{len(agreeing)}_candidates"

    # Tied on verification and disagreeing on the number. This is the case
    # worth refusing: two queries the verifier likes equally, returning
    # different answers, means the question was ambiguous in a way the plan
    # did not capture.
    if len(tied) > 1 and len({
        round(h, 6) for h in (c.headline() for c in tied) if h is not None
    }) > 1:
        return None, "verified_candidates_disagree"

    return tied[0], "highest_verification_score"


def summarise(candidates: list[Candidate], chosen: Candidate | None,
              reason: str) -> dict[str, Any]:
    """The record of what was tried and why one was picked.

    Goes into the pipeline trace: a "why did it answer with that query"
    question should be answerable from the record rather than re-derived, and
    a selection nobody can explain is one nobody will trust.
    """
    return {
        "candidates": len(candidates),
        "usable": sum(1 for c in candidates if c.usable),
        "verified": sum(1 for c in candidates if c.verified),
        "reason": reason,
        "chosen_source": chosen.source if chosen else "",
        "chosen_score": chosen.score if chosen else 0,
        "scores": [
            {
                "source": c.source,
                "validated": c.validated,
                "executed": c.executed,
                "score": c.score,
                "rows": c.row_count,
                "error": (c.error or "")[:120],
            }
            for c in candidates
        ],
    }
