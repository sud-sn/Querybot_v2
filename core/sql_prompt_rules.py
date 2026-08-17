"""Decide which SQL-prompt rules a given question actually needs.

The SQL system prompt carries ~59 named rules and sends every one of them on
every question. Two costs follow. Attention dilutes -- a rule competing with
58 others is followed probabilistically, which is why prompt-only fixes hold
"most of the time" rather than always. And the curve bends the wrong way:
each new failure mode adds rule #60, #61, weakening #1-59 further.

Now that the pipeline compiles an analytical request *before* building the
prompt, we know in advance whether a question involves a period comparison, a
moving average, an anti-join, and so on. Rules that cannot apply are dropped.

Safety contract -- this module is deliberately biased toward inclusion:

  * A rule is omitted ONLY when a signal positively says it cannot apply.
  * No question text (or no plan) means every gated rule is included, so the
    prompt is byte-identical to the pre-gating behaviour.
  * Any exception inside a predicate includes the rule.

Omitting a needed rule silently degrades SQL. Including a redundant one costs
a few hundred tokens. The asymmetry decides the default.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("querybot.sql_prompt_rules")


@dataclass
class RuleContext:
    """Everything a gate is allowed to look at."""

    question: str = ""
    table_context: str = ""
    semantic_plan: dict[str, Any] = field(default_factory=dict)
    graph_context: dict[str, Any] = field(default_factory=dict)

    @property
    def q(self) -> str:
        return (self.question or "").lower()

    @property
    def request_plan(self) -> dict[str, Any]:
        plan = self.semantic_plan or {}
        return plan.get("analytical_request_plan") or {}

    @property
    def has_question(self) -> bool:
        """Without a question there is no signal, so nothing may be dropped."""
        return bool((self.question or "").strip())

    def asks(self, pattern: str) -> bool:
        try:
            return bool(re.search(pattern, self.q))
        except re.error:  # pragma: no cover - a malformed pattern must not gate
            return True


# ── Signals ──────────────────────────────────────────────────────────────
# Each returns True when the rule is *needed*. Phrasing lists stay generous;
# a false positive costs tokens, a false negative costs correctness.

_COMPARISON_WORDS = (
    r"year[- ]over[- ]year|yoy|prior year|previous year|last year|year before|"
    r"vs\.? *(?:last|previous|prior)|compared? (?:to|with)|change (?:from|since)|"
    r"growth|decline|trend"
)
_PERIOD_STEP_WORDS = (
    r"month[- ]over[- ]month|mom\b|quarter[- ]over[- ]quarter|qoq\b|"
    r"previous month|prior month|last month vs|sequential"
)
_MOVING_AVG_WORDS = r"moving average|rolling (?:average|mean|sum|\d+)|trailing \d+"
_CORRELATION_WORDS = r"correlat|scatter|relationship between|against each other|plot .* (?:vs|against)"
_ANTI_JOIN_WORDS = (
    r"never|without|missing|no (?:orders|sales|activity|transactions)|"
    r"not been|n't been|not yet|haven'?t|hasn'?t|didn'?t|with no\b|lacking"
)
_INTERVAL_WORDS = (
    r"(?:average|avg|mean|typical) (?:time|days|gap|interval|duration)|"
    r"time between|days between|gap between|how long between|between .* and .* orders"
)
_BENCHMARK_WORDS = (
    r"above (?:the )?average|below (?:the )?average|above|below|better than|worse than|"
    r"outperform|underperform|benchmark|exceed"
)


def _plan_has_comparison(ctx: RuleContext) -> bool:
    plan = ctx.request_plan
    if not plan:
        return False
    if plan.get("comparison_periods") or plan.get("comparison"):
        return True
    for op in plan.get("temporal_operations") or []:
        kind = str((op or {}).get("kind") or (op or {}).get("type") or "").lower()
        if any(token in kind for token in ("compare", "prior", "yoy", "mom", "qoq", "delta")):
            return True
    return False


def _needs_period_comparison(ctx: RuleContext) -> bool:
    return ctx.asks(_COMPARISON_WORDS) or _plan_has_comparison(ctx)


def _needs_period_step(ctx: RuleContext) -> bool:
    return ctx.asks(_PERIOD_STEP_WORDS) or _plan_has_comparison(ctx)


def _needs_moving_average(ctx: RuleContext) -> bool:
    return ctx.asks(_MOVING_AVG_WORDS)


def _needs_correlation(ctx: RuleContext) -> bool:
    return ctx.asks(_CORRELATION_WORDS)


def _needs_anti_join(ctx: RuleContext) -> bool:
    if ctx.asks(_ANTI_JOIN_WORDS):
        return True
    # The graph resolver flags an anti-join shape independently of phrasing.
    return bool((ctx.graph_context or {}).get("anti_join"))


def _needs_interval(ctx: RuleContext) -> bool:
    return ctx.asks(_INTERVAL_WORDS)


def _needs_benchmark(ctx: RuleContext) -> bool:
    return ctx.asks(_BENCHMARK_WORDS)


# ── Registry ─────────────────────────────────────────────────────────────
# Only narrowly-scoped, high-token rules are gated. Universal safety rules
# (column-hallucination, dialect syntax, field-plan, fact-to-fact joins)
# are never gated -- they apply to every query by construction.

GATED_RULES: dict[str, Callable[[RuleContext], bool]] = {
    "year_over_year":   _needs_period_comparison,
    "month_over_month": _needs_period_step,
    "moving_average":   _needs_moving_average,
    "correlation":      _needs_correlation,
    "anti_join":        _needs_anti_join,
    "avg_interval":     _needs_interval,
    "benchmark":        _needs_benchmark,
}


def rule_applies(rule_id: str, ctx: RuleContext | None) -> bool:
    """True when `rule_id` should be included in the SQL system prompt.

    Unknown rule ids, a missing context, an empty question, or a raising
    predicate all resolve to True -- see the safety contract above.
    """
    if ctx is None or not ctx.has_question:
        return True
    predicate = GATED_RULES.get(rule_id)
    if predicate is None:
        return True
    try:
        return bool(predicate(ctx))
    except Exception:
        log.exception("sql prompt rule gate %r raised; including the rule", rule_id)
        return True


def applied_rules(ctx: RuleContext | None) -> dict[str, bool]:
    """Gate decisions for every rule -- for tracing and tests."""
    return {rule_id: rule_applies(rule_id, ctx) for rule_id in GATED_RULES}
