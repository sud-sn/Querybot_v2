"""Which subject area a question belongs to, and what to do when two can answer.

A tenant's analytics do not arrive as one undifferentiated schema. There is a
sales area, a supply-chain area, a finance area, each with its own vocabulary
and its own facts — and today a question is answered against all of them at
once, so nothing can say which area an answer came from or notice that two of
them disagree about it.

The reference product this was measured against binds one app per turn and,
in its deep mode, lets the agent wander into others mid-loop. That reads as
cross-app answering and is genuinely more than we do — but an agent that
reads two apps has no mechanism to notice they disagree, because nothing
asked it to look. Corroboration is the cheaper and stronger move: answer from
the primary, run the same request against the secondary, and compare. A
comparison step cannot miss a disagreement; a loop that was never told to
look can miss every one.

Routing here is deliberately the same shape ``core.metric_scope`` already
uses for metrics: whole-phrase matches on the words a domain declares, scored
against the question. A question that matches nothing routes nowhere and the
existing full-workspace behaviour is unchanged, so adding domains cannot make
an un-domained workspace worse.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("querybot.domains")

# A domain must beat the runner-up by this proportion to be chosen outright.
# Below it, two domains are both plausible and the question is a candidate
# for corroboration rather than a routing decision.
DECISIVE_MARGIN = 0.35

# Below this, no domain is a real match and the workspace answers as it
# always has. Adding domains must not make an unrouted question worse.
MIN_ROUTE_SCORE = 2.0

_WORD = re.compile(r"[a-z0-9]+")
_SPLIT = re.compile(r"[,;|\n]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(str(text or "").casefold()))


def _phrases(domain: dict) -> list[str]:
    """Every phrase that names this domain."""
    out = [str(domain.get("name") or "")]
    out.extend(part for part in _SPLIT.split(str(domain.get("synonyms") or "")))
    return [p.strip() for p in out if p.strip()]


@dataclass(frozen=True)
class DomainMatch:
    name: str
    score: float
    tables: tuple[str, ...] = ()
    matched: tuple[str, ...] = ()


@dataclass(frozen=True)
class Routing:
    """Where a question should be answered, and what could confirm it."""

    primary: DomainMatch | None = None
    secondary: DomainMatch | None = None
    considered: tuple[DomainMatch, ...] = ()
    reason: str = ""

    @property
    def routed(self) -> bool:
        return self.primary is not None

    @property
    def should_corroborate(self) -> bool:
        """Can a second domain answer the same question?

        True only when a runner-up scored close to the winner. Running every
        question twice would double the cost of the product to confirm
        answers nothing disputed.
        """
        return self.primary is not None and self.secondary is not None


def score_domain(question: str, domain: dict) -> DomainMatch:
    """How well this domain matches the question.

    A whole-phrase hit on the domain's name or a synonym is worth far more
    than incidental token overlap: "supply" appearing inside a sentence about
    supplier payments should not route a finance question to the supply-chain
    area. Same weighting shape as ``core.metric_scope._phrase_score``, and for
    the same reason -- one token of overlap once bound a question to entirely
    the wrong metric.
    """
    text = str(question or "").casefold()
    question_tokens = _tokens(question)
    score = 0.0
    matched: list[str] = []

    for phrase in _phrases(domain):
        pattern = re.escape(phrase.casefold())
        if re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", text):
            score += 10.0 + len(_tokens(phrase)) * 2.0
            matched.append(phrase)

    # Table names contribute, weakly: a question naming a table in this domain
    # belongs to it, but table names are rarely what a business user types.
    tables = [str(t) for t in (domain.get("tables") or [])]
    table_tokens: set[str] = set()
    for table in tables:
        table_tokens |= _tokens(table.replace(".", " ").replace("_", " "))
    overlap = question_tokens & table_tokens
    score += len(overlap) * 1.0

    # The description is the weakest signal and is capped, so a long
    # description cannot outweigh an exact name match on another domain.
    description_overlap = question_tokens & _tokens(domain.get("description"))
    score += min(len(description_overlap), 3) * 0.5

    return DomainMatch(
        name=str(domain.get("name") or ""),
        score=round(score, 3),
        tables=tuple(tables),
        matched=tuple(matched + sorted(overlap)),
    )


def route(question: str, domains: list[dict]) -> Routing:
    """Pick the domain to answer in, and the one that could confirm it.

    Returns an unrouted ``Routing`` when nothing matches, and the caller then
    answers exactly as it did before domains existed.
    """
    if not domains:
        return Routing(reason="no_domains")

    scored = sorted(
        (score_domain(question, d) for d in domains),
        key=lambda m: (-m.score, m.name),
    )
    best = scored[0]
    if best.score < MIN_ROUTE_SCORE:
        return Routing(considered=tuple(scored), reason="no_domain_matched")

    runner_up = scored[1] if len(scored) > 1 else None
    if runner_up is None or runner_up.score < MIN_ROUTE_SCORE:
        return Routing(primary=best, considered=tuple(scored),
                       reason="single_matching_domain")

    margin = (best.score - runner_up.score) / max(best.score, 1.0)
    if margin >= DECISIVE_MARGIN:
        return Routing(primary=best, considered=tuple(scored),
                       reason="decisive_match")

    # Two plausible areas. Answer from the better one and check the other --
    # this is the case the whole module exists for.
    return Routing(primary=best, secondary=runner_up, considered=tuple(scored),
                   reason="two_plausible_domains")


def allowed_tables_for(routing: Routing, *, existing: set[str] | None = None) -> set[str] | None:
    """The table scope a routed question should run under.

    Intersected with whatever scope the caller already had, never replacing
    it: a domain narrows what a user may see, it must never widen it. A user
    with access to three tables who asks a question routed to a domain of
    twenty gets three, not twenty.
    """
    if routing.primary is None:
        return existing
    domain_tables = {t.upper() for t in routing.primary.tables}
    if existing is None:
        return domain_tables
    return {t for t in existing if t.upper() in domain_tables}


# ══════════════════════════════════════════════════════════════════════════════
# Corroboration
# ══════════════════════════════════════════════════════════════════════════════

# Two figures within this relative distance are the same answer. Wider than
# the candidate-selection tolerance on purpose: two subject areas computing
# the same measure from different facts will differ by rounding, by late
# arrivals, by a day's lag in a load -- and calling that a contradiction
# would cry wolf on every question.
CORROBORATION_TOLERANCE = 0.01


@dataclass(frozen=True)
class Corroboration:
    """Whether a second source agrees with the answer given."""

    checked: bool = False
    agrees: bool = False
    primary_value: float | None = None
    secondary_value: float | None = None
    difference: float | None = None
    relative_difference: float | None = None
    primary_source: str = ""
    secondary_source: str = ""
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


def _headline(rows: list[dict] | None) -> float | None:
    """The single figure a result is really asserting, if there is one."""
    for row in (rows or [])[:1]:
        for value in row.values():
            try:
                number = float(str(value).replace(",", ""))
            except (TypeError, ValueError):
                continue
            if number == number:
                return number
    return None


def corroborate(
    primary_rows: list[dict] | None,
    secondary_rows: list[dict] | None,
    *,
    primary_source: str = "",
    secondary_source: str = "",
) -> Corroboration:
    """Does the second source agree with the first?

    Deliberately reports "not checked" rather than "agrees" whenever there is
    nothing comparable — an empty secondary result, or a result with no
    figure in it. Reporting agreement that was never established is worse
    than reporting nothing, because a user reads "confirmed" as evidence.
    """
    base = Corroboration(primary_source=primary_source,
                         secondary_source=secondary_source)
    if not primary_rows:
        return Corroboration(**{**base.__dict__, "reason": "primary_empty"})
    if not secondary_rows:
        return Corroboration(**{**base.__dict__, "reason": "secondary_empty"})

    first = _headline(primary_rows)
    second = _headline(secondary_rows)
    if first is None or second is None:
        return Corroboration(**{**base.__dict__, "reason": "no_comparable_figure"})

    difference = second - first
    scale = max(abs(first), abs(second), 1.0)
    relative = abs(difference) / scale
    agrees = relative <= CORROBORATION_TOLERANCE
    return Corroboration(
        checked=True,
        agrees=agrees,
        primary_value=first,
        secondary_value=second,
        difference=round(difference, 6),
        relative_difference=round(relative, 6),
        primary_source=primary_source,
        secondary_source=secondary_source,
        reason="agreement" if agrees else "disagreement",
        detail={
            "primary_rows": len(primary_rows),
            "secondary_rows": len(secondary_rows),
        },
    )


def describe(result: Corroboration, *, lang: str = "en") -> str:
    """The one line a user is shown about the second source."""
    from core.i18n import format_decimal, format_percent, t

    if not result.checked:
        return ""
    if result.agrees:
        return t("corroboration.agrees", lang=lang,
                 source=result.secondary_source)
    return t(
        "corroboration.disagrees", lang=lang,
        source=result.secondary_source,
        primary=format_decimal(result.primary_value, lang=lang),
        secondary=format_decimal(result.secondary_value, lang=lang),
        gap=format_percent((result.relative_difference or 0.0) * 100, 1, lang=lang),
    )
