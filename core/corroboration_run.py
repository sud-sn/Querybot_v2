"""Asking the same question of a second subject area, and comparing the answers.

``core.domains`` decides that two areas could both answer a question and
``core.domains.corroborate`` compares two results. Between them sat the part
that actually costs something: producing the second result. This module is
that part.

It is a second retrieval and a second generation, not a parameter. The
knowledge base the primary answer was written against was retrieved for the
primary domain's tables; a query generated from that context and then pointed
at another area's tables would differ from the primary in ways the comparison
cannot attribute — a "disagreement" that is an artefact of the prompt rather
than a conflict in the data. So the second opinion gets its own retrieval, its
own prompt, and its own scope, and the only thing carried over is the question.

Two rules run through everything here:

**A second opinion may never cost the answer.** Every step is allowed to fail,
and every failure reports "not checked" rather than degrading what the primary
produced. A user who is told two areas disagree when in fact the second query
never ran has been told something false about their data.

**A second opinion is still an answer this user is shown**, so it runs under
the same guards. Its scope is the runner-up domain intersected with what the
user could already see (``core.domains.secondary_scope``), its SQL goes
through ``core.sql_attempt.run_attempt`` — same validator, same repairs — and
its execution goes through ``execute_governed_query`` like every other query
in the product.

The subtle one is the semantic context. The validator's guards read it, and
one of them is the raw fact-to-fact join guard: ``_raw_multi_fact_errors``
returns nothing at all when ``semantic_plan.known_fact_tables`` is absent,
which is how a revenue question once joined two facts and reported 2700.00
against a true 1050.00. Handing the corroborating attempt an empty context
would switch that guard off for exactly the query nobody is reading closely.
``corroborating_semantic_context`` therefore carries the fact list across —
and it can, because the list is a property of the workspace's compiled
semantic model, not of the question or the domain that asked it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from core.domains import Corroboration, corroborate, describe
from core.sql_attempt import Attempt, ValidationScope

log = logging.getLogger("querybot.corroboration")

# How many knowledge-base documents the second opinion is given. Smaller than
# the primary's budget on purpose: this is one query against one subject area,
# and a wider context buys nothing but tokens on every ambiguous question.
MAX_CONTEXT_DOCUMENTS = 6

# What survives from the primary's semantic context into the corroborating
# one. Everything here is a property of the QUESTION rather than of the plan
# that was compiled for the primary domain -- the intent it expressed, the
# top-N it asked for, its own text -- so it is still true of a query written
# against a different area.
#
# Everything NOT here is deliberately dropped: `graph_context`,
# `resolution_plan`, `analytical_request_plan` and the compiled `semantic_plan`
# fields are all resolved against the primary's tables, and checking a
# secondary-scope query against them would fail every corroboration on a
# plan mismatch that says nothing about the data.
CARRIED_CONTEXT_KEYS = ("question", "intent", "top_n")


@dataclass(frozen=True)
class SecondOpinion:
    """What a second subject area said, or why it did not say anything."""

    checked: bool = False
    domain: str = ""
    reason: str = ""
    sql: str = ""
    row_count: int = 0
    duration_ms: int = 0
    corroboration: Corroboration = field(default_factory=Corroboration)
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def agrees(self) -> bool:
        return self.checked and self.corroboration.agrees

    @property
    def disagrees(self) -> bool:
        return self.checked and not self.corroboration.agrees

    def as_dict(self) -> dict[str, Any]:
        """The shape the pipeline puts on the trace and the confidence context."""
        return {
            "checked": self.checked,
            "agrees": self.agrees,
            "domain": self.domain,
            "reason": self.reason,
            "row_count": self.row_count,
            "duration_ms": self.duration_ms,
            "primary_value": self.corroboration.primary_value,
            "secondary_value": self.corroboration.secondary_value,
            "relative_difference": self.corroboration.relative_difference,
            "detail": dict(self.detail),
        }


def _not_checked(domain: str, reason: str, **extra) -> SecondOpinion:
    return SecondOpinion(checked=False, domain=domain, reason=reason, **extra)


def corroborating_semantic_context(primary: dict | None) -> dict:
    """The semantic context a corroborating attempt is validated against.

    Question-level facts are carried over, plan-level ones are not — see
    ``CARRIED_CONTEXT_KEYS`` for why.

    Two things are always set regardless of what the primary held:

    ``production_sql`` keeps the shape rules on (no ``SELECT *``, no
    unrequested cartesian product), because those are properties of generated
    SQL rather than of a plan.

    ``semantic_plan.known_fact_tables`` carries the workspace's fact list
    across. This is the one that matters. The validator's raw fact-to-fact
    guard reads exactly this key and returns *nothing at all* when fewer than
    two facts are known, so an empty context does not weaken the guard, it
    removes it — and the corroborating query is the one nobody reads before it
    is compared against the answer a user is being shown.
    """
    source = primary or {}
    context: dict[str, Any] = {
        key: source[key] for key in CARRIED_CONTEXT_KEYS if key in source
    }
    context["production_sql"] = True
    facts = [
        str(name) for name in
        ((source.get("semantic_plan") or {}).get("known_fact_tables") or [])
        if str(name or "").strip()
    ]
    if len(facts) < 2:
        # Not fatal — a workspace with one fact table cannot fan out across
        # two — but it is the condition under which the guard is inert, and
        # a guard that is inert for a reason nobody recorded is one nobody
        # notices has stopped working.
        log.info(
            "Corroborating attempt carries %d known fact table(s); the raw "
            "fact-to-fact join guard needs two to fire", len(facts),
        )
    context["semantic_plan"] = {"known_fact_tables": facts}
    return context


def corroborating_scope(primary: ValidationScope, tables: set[str]) -> ValidationScope:
    """The validation scope the corroborating attempt runs under.

    ``tables`` is the second opinion's scope and must already be the
    intersection of the runner-up domain with the user's own permissions —
    ``core.domains.secondary_scope`` computes it at the one point in the
    pipeline that still holds the un-narrowed scope. It is passed as
    ``allowed_tables`` unconditionally, including for an unrestricted admin
    whose primary scope was ``None``: "run this under the second area's
    tables" is the whole point, and leaving it unrestricted would let the
    second opinion answer from the first area's facts and agree with itself.

    ``known_tables`` and ``table_columns`` are the workspace's schema and are
    the same for every scope; narrowing them would turn a permission failure
    into an unknown-table error and lose the distinction.
    """
    return ValidationScope(
        known_tables=primary.known_tables,
        db_type=primary.db_type,
        allowed_tables=set(tables or set()),
        table_columns=primary.table_columns,
        semantic_context=corroborating_semantic_context(primary.semantic_context),
    )


def corroborating_user_message(question: str, domain: str) -> str:
    """The user half of the corroborating generation call.

    The same question, plus the one instruction that keeps a disagreement
    meaningful: answer from this area or answer not at all. Without it the
    model will reach for whatever it can measure, and the comparison then
    reports a "disagreement" between two different questions — the loudest
    possible way to be wrong, because a user reads a contradiction as
    evidence of a data problem.
    """
    area = str(domain or "").strip() or "this subject area"
    return (
        f"{question}\n\n"
        f"SECOND OPINION: answer this question from the {area} tables in the "
        f"knowledge base above, and from those only. Measure the same thing "
        f"the question asks for. If {area} cannot answer this question, reply "
        f"with exactly CANNOT_GENERATE — do not answer a nearby question and "
        f"do not reach for a table that is not listed above."
    )


def build_context(documents: list[str], *, limit: int = MAX_CONTEXT_DOCUMENTS) -> str:
    """The knowledge-base half of the corroborating prompt.

    Same separator the primary path uses, so the prompt the model sees has the
    shape it was tuned on. Empty when retrieval found nothing for this scope,
    and an empty context is a reason to stop rather than to generate: SQL
    written with no schema in front of it is a guess, and a guess that
    disagrees with the answer is worse than no second opinion at all.
    """
    docs = [str(d).strip() for d in (documents or []) if str(d or "").strip()]
    return "\n\n---\n\n".join(docs[:limit])


async def run_second_opinion(
    question: str,
    *,
    domain: str,
    tables: set[str],
    primary_rows: list[dict] | None,
    primary_domain: str = "",
    scope: ValidationScope,
    retrieve: Callable[[set[str]], list[str]],
    generate: Callable[[str, str], Awaitable[str]],
    execute: Callable[[str, ValidationScope], Awaitable[Attempt]],
) -> SecondOpinion:
    """Answer the question again from a second subject area, and compare.

    ``retrieve``, ``generate`` and ``execute`` are the three boundaries this
    needs — the vector store, the model, and the governed executor. They are
    injected rather than reached for so the sequence between them is callable
    from a test with no Qdrant, no provider key and no warehouse; everything
    in between is this module's, and is what the tests actually check.

    Never raises. Every failure path returns a ``SecondOpinion`` that reports
    what was not checked and why, because the primary answer is already
    correct and complete without this.
    """
    started = time.time()

    def _elapsed() -> int:
        return int((time.time() - started) * 1000)

    if not domain:
        return _not_checked("", "no_secondary_domain")
    if not tables:
        return _not_checked(domain, "no_visible_tables")
    if not primary_rows:
        # Nothing to corroborate against. Checked before anything is spent:
        # the comparison would return "primary_empty" after a retrieval, a
        # generation and a query, and report exactly the same thing.
        return _not_checked(domain, "primary_empty")

    try:
        documents = retrieve(set(tables))
    except Exception as exc:  # noqa: BLE001
        log.warning("Second-opinion retrieval failed for %s: %s", domain, exc)
        return _not_checked(domain, "retrieval_failed", duration_ms=_elapsed())

    context = build_context(documents)
    if not context:
        log.info("No knowledge base under %s; skipping the second opinion", domain)
        return _not_checked(domain, "no_context", duration_ms=_elapsed())

    from core.llm import build_sql_system_prompt

    try:
        # No graph context and no semantic plan: both were resolved against
        # the primary's tables, and a prompt that describes one area's joins
        # while asking for another area's answer is worse than no plan at all.
        system = build_sql_system_prompt(scope.db_type, context, question=question)
        raw = await generate(system, corroborating_user_message(question, domain))
    except Exception as exc:  # noqa: BLE001
        log.warning("Second-opinion generation failed for %s: %s", domain, exc)
        return _not_checked(domain, "generation_failed", duration_ms=_elapsed())

    # Deferred: core.query_pipeline imports this module.
    from core.query_pipeline import clean_generated_sql

    try:
        sql = clean_generated_sql(raw or "", question, scope.db_type)
    except Exception as exc:  # noqa: BLE001
        log.warning("Second-opinion SQL could not be cleaned for %s: %s", domain, exc)
        return _not_checked(domain, "generation_failed", duration_ms=_elapsed())

    if not sql or "CANNOT_GENERATE" in sql.upper():
        # The second area was asked to refuse rather than guess, so a refusal
        # is the instruction working, not a fault.
        log.info("%s declined the second opinion", domain)
        return _not_checked(domain, "not_answerable_there", duration_ms=_elapsed())

    try:
        attempt = await execute(sql, corroborating_scope(scope, tables))
    except Exception as exc:  # noqa: BLE001
        log.warning("Second-opinion attempt failed for %s: %s", domain, exc)
        return _not_checked(domain, "attempt_failed", sql=sql, duration_ms=_elapsed())

    if not attempt.ok or attempt.rows is None:
        return _not_checked(
            domain,
            "invalid_there" if not attempt.ok else "execution_failed",
            sql=attempt.sql or sql,
            duration_ms=_elapsed(),
            detail={"code": attempt.code, "reason": attempt.reason[:400]},
        )

    result = corroborate(
        primary_rows, attempt.rows,
        primary_source=primary_domain, secondary_source=domain,
    )
    return SecondOpinion(
        checked=result.checked,
        domain=domain,
        reason=result.reason,
        sql=attempt.sql or sql,
        row_count=len(attempt.rows),
        duration_ms=_elapsed(),
        corroboration=result,
        detail={"truncated": bool(attempt.truncated)},
    )


def describe_second_opinion(payload: dict | None, *, lang: str | None = None) -> str:
    """The one line a reader gets about the second area.

    Takes the dict :meth:`SecondOpinion.as_dict` produces, because that is
    what survives onto the confidence context — the dataclass does not cross
    that boundary. Empty for a run that did not complete: a reader takes
    either sentence as a statement about their data, and "the second area was
    never asked" is not one of them.
    """
    data = payload or {}
    if not data.get("checked"):
        return ""
    return describe(
        Corroboration(
            checked=True,
            agrees=bool(data.get("agrees")),
            primary_value=data.get("primary_value"),
            secondary_value=data.get("secondary_value"),
            relative_difference=data.get("relative_difference"),
            secondary_source=str(data.get("domain") or ""),
        ),
        lang=lang,
    )
