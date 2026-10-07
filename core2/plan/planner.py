"""From a question to a typed plan: the one step where an AI decides about a question (DESIGN §7).

The prompt is a stable prefix (rules, the plan's schema, the workspace catalog and
a few examples written in the workspace's own words), byte-identical for a model
version so providers cache it, then a short tail (today, the conversation, the
member names found in the question, the question). The answer must be one JSON
object that validates as a :class:`Plan`; the plan is then checked against the
model by resolving it. A plan that fails gets one repair round with the problems
and the allowed options; one that still fails becomes a clarification, never a
guess.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from core2.bootstrap import names
from core2.model.schema import Measure, SemanticModel
from core2.plan.catalog import catalog_text
from core2.plan.ir import Clarify, Plan, schema_text
from core2.plan.values import Masked, ValueMatch, mask, placeholder, unmask
from core2.resolve import paths as P
from core2.resolve.resolver import Context, ResolveError, resolve

Complete = Callable[[str, str], str]       # (stable system prompt, question tail) -> the model's text

RULES = """You turn a business question into a query plan for QueryBot. You never write SQL.
Answer with exactly one JSON object that matches the PLAN SCHEMA. No prose, no markdown.

How to plan:
1. kind: "query" for questions about the data. "clarify" only when the question is genuinely ambiguous
   (two measures fit equally, a date that could mean two dates, a name that fits several members): put the
   question and 2-5 options from the catalog in clarify. "describe_data" for "what data do you have",
   "what can I ask". "smalltalk" for greetings and thanks. "unsupported" when nothing in the catalog
   answers it; say why in notes.
2. measures: the measure slugs the question names. Never swap in a different measure because it is close
   (sales is not margin, quantity is not value, count of rows is not count of orders). When the question
   needs a ratio, difference, sum or product of two measures (a margin %, revenue per order), put it in
   derived with a short name, the op and the two slugs (scale 100 for a percentage).
3. group_by: attribute or entity slugs for "by X", "per X", "each X", "which X"; an entity slug groups by
   its name. A time: attribute for "by weekday", "by month of the year", "weekends". When the catalog lists
   roles for an entity and the question names one ("by ship-to customer", "the customer's home store"),
   put it in via: {"<the group_by slug>": "<role>"}; with no role named, leave via empty.
4. time.grain makes a series: "monthly", "by month", "weekly", "per quarter", "trend", "over time" (month
   when unsaid). No grain for one total.
5. time.window, written as the user said it; never compute relative dates yourself:
   "last 6 months" -> {"kind": "last", "unit": "month", "n": 6} (complete months before the current one);
   "this year", "year to date", "so far this year" -> {"kind": "this", "unit": "year"};
   "last month" -> {"kind": "previous", "unit": "month"}; "last year" -> {"kind": "previous", "unit": "year"};
   "in March 2026" -> {"kind": "between", "start": "2026-03-01", "end": "2026-03-31"};
   "since January" -> {"kind": "since", "start": "<this year>-01-01"};
   fiscal periods ("this fiscal year", "FY2026", "Q3 FY26") set "fiscal": true.
   No window when the user gave none: everything is the default.
6. time.date: a date slug only when the question names which date ("by ship date", "invoiced in May",
   "expiring in 2028"); otherwise null and the measure's default date is used. A second date in the
   question ("ordered in March and delivered in April") is a filter on that date slug with ISO dates.
7. time.compare: "vs last year", "year over year" -> {"kind": "same_period_last_year"};
   "vs last month", "month over month", "vs the previous period" -> {"kind": "previous_period"};
   "from February to March", "March vs February" -> window March and
   {"kind": "window", "window": February}; the intent is then "compare".
8. filters: a member goes in a filter on its attribute with op "eq" (or "in" for several), using the exact
   stored value from VALUE MATCHES when one is given; "excluding X" -> "ne"/"not_in". A condition on a
   total ("customers with more than 10,000 in sales") is a filter on the measure slug.
9. sort and limit: "top 5" -> sort by the measure (desc) with limit 5; "bottom 5", "least" -> desc false.
   "grew the most" -> sort by "change" desc; "biggest drop" -> "change" with desc false.
10. intent: value | breakdown | trend | compare | rank | share | list | count. "List the X", "what are the
   X" with no measure -> intent list with group_by [X] and no measures. "How many X" -> intent count with
   the "Number of X" measure. "Share of", "% of total", "contribution" -> intent share.
11. Follow-ups: PREVIOUS PLAN is the plan behind the last answer. When the new question changes it ("by
   week instead", "only for X", "what about returns", "top 5 of those"), return the whole new plan with
   follow_up "refine"; a question about something else is follow_up "new".
12. notes: one short line for each assumption the user did not state.
"""


@dataclass
class Turn:
    question: str
    plan: Plan | None = None
    masked: Masked | None = None        # what the AI was shown, when member values are withheld


@dataclass
class Outcome:
    plan: Plan
    repaired: bool = False
    problems: list[str] = field(default_factory=list)   # what the last attempt got wrong, if anything
    raw: list[str] = field(default_factory=list)        # the model's answers, for the record
    masked: Masked | None = None                        # the placeholders used, when values are withheld


def _examples(model: SemanticModel) -> str:
    """Three worked examples in this workspace's own words (deterministic for a model version).

    They use the main fact's leading measure (money first) and an entity it reaches,
    so they never teach a combination the model cannot answer.
    """
    measures = sorted((m for m in model.measures.values() if not m.hidden and m.kind != "count"
                       and m.additivity == "additive" and m.default_date),
                      key=lambda m: (-model.tables[m.table].row_count, m.format != "currency", m.slug))
    if not measures:
        return ""
    m: Measure = measures[0]
    entities = sorted((e for e in model.entities.values()
                       if e.table != m.table and P.all_paths(model, m.table, e.table)), key=lambda e: e.slug)
    lines = ["EXAMPLES (in this workspace's words)"]
    plan: dict[str, Any] = {"kind": "query", "intent": "trend", "measures": [m.slug],
                            "time": {"grain": "month", "window": {"kind": "last", "unit": "month", "n": 6}}}
    lines.append(f"Q: {m.business_name.lower()} by month for the last 6 months\nA: {json.dumps(plan)}")
    if entities:
        e = entities[0]
        plan = {"kind": "query", "intent": "rank", "measures": [m.slug], "group_by": [e.slug],
                "sort": [{"by": m.slug, "desc": True}], "limit": 5,
                "time": {"window": {"kind": "this", "unit": "year"}}}
        lines.append(f"Q: top 5 {names.plural(e.business_name.lower())} by {m.business_name.lower()} this year\n"
                     f"A: {json.dumps(plan)}")
        plan = {"kind": "query", "intent": "compare", "measures": [m.slug], "group_by": [e.slug],
                "time": {"window": {"kind": "previous", "unit": "month"},
                         "compare": {"kind": "previous_period"}},
                "sort": [{"by": "change", "desc": True}]}
        lines.append(f"Q: which {names.plural(e.business_name.lower())} grew the most last month\n"
                     f"A: {json.dumps(plan)}")
    return "\n".join(lines)


def stable_prompt(model: SemanticModel, *, values_allowed: bool = True) -> str:
    """Rules, schema, catalog and examples: identical for every question on a model version."""
    return "\n\n".join(part for part in (
        RULES, "PLAN SCHEMA (JSON Schema)\n" + schema_text(),
        catalog_text(model, values_allowed=values_allowed), _examples(model)) if part)


def question_tail(question: str, *, today: dt.date, history: list[Turn], matches: list[ValueMatch],
                  masked: Masked | None = None, scrub: Callable[[str], str] | None = None,
                  model: SemanticModel | None = None) -> str:
    """Today, the conversation, the member matches and the question (each question scrubbed when asked to).

    With member values withheld (``masked``), the previous turn is shown as the AI saw
    it: its question with placeholders, and its plan's member values as placeholders too.
    """
    clean = scrub or (lambda text: text)
    lines = [f"TODAY: {today.isoformat()} ({today:%A})"]
    if history:
        last = history[-1]
        if masked is None or last.masked is not None:
            asked = last.masked.question if masked is not None and last.masked is not None else last.question
            lines.append(f"PREVIOUS QUESTION: {clean(asked)}")
        if last.plan is not None:
            lines.append("PREVIOUS PLAN: " + _plan_shown(last.plan, model, masked))
    if matches:
        lines.append("VALUE MATCHES (member names found in the question; use these exact values):")
        for m in matches:
            shown = next((t for t, v in masked.values.items() if v == m.value), m.text) if masked else m.text
            stored = shown if masked else m.value
            lines.append(f'- "{shown}" -> {m.attribute} = "{stored}"')
    lines.append(f"QUESTION: {clean(masked.question if masked else question)}")
    return "\n".join(lines)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse(text: str) -> Plan:
    """The one JSON object in the model's answer, as a Plan (fences and stray prose tolerated)."""
    body = _FENCE.sub("", text.strip())
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("the answer holds no JSON object")
    data = json.loads(body[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("the answer is not a JSON object")
    return Plan.model_validate(data)


def check(plan: Plan, model: SemanticModel, today: dt.date) -> list[str]:
    """What stops the plan from being answered, in words the planner can act on."""
    if plan.kind == "clarify":
        return [] if plan.clarify and plan.clarify.question else ["a clarify plan needs clarify.question"]
    if plan.kind != "query":
        return []
    try:
        resolve(plan, model, Context(today=today))
    except ResolveError as exc:
        if exc.kind in ("unconfirmed", "denied", "ambiguous"):
            return []          # not the planner's to fix: the service answers or asks the reader
        options = f" Allowed: {', '.join(exc.options[:12])}." if exc.options else ""
        return [f"{exc.message}.{options}"]
    except ValueError as exc:
        return [str(exc)]
    return []


def _plan_shown(plan: Plan, model: SemanticModel | None, masked: Masked | None) -> str:
    """A plan as the AI may see it: member values as placeholders when values are withheld."""
    if masked is None:
        return plan.model_dump_json(exclude_defaults=True)
    data = plan.model_dump(mode="json", exclude_defaults=True)
    for f in data.get("filters", []):
        if model is not None and (f.get("field") in model.measures or f.get("field") in model.date_roles):
            continue      # amounts and dates, not member values
        f["values"] = [placeholder(v, masked.values) for v in f.get("values", [])]
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _unmask_plan(plan: Plan, masked: Masked | None) -> Plan:
    if masked is None or not masked.values:
        return plan
    data = plan.model_dump()
    for f in data.get("filters", []):
        f["values"] = [unmask(v, masked) for v in f.get("values", [])]
    return Plan.model_validate(data)


def plan_question(model: SemanticModel, question: str, complete: Complete, *, today: dt.date,
                  history: list[Turn] | None = None, matches: list[ValueMatch] | None = None,
                  values_allowed: bool = True, scrub: Callable[[str], str] | None = None) -> Outcome:
    """The question's plan, checked against the model; at most two calls to the AI."""
    matches = matches or []
    history = history or []
    known = history[-1].masked.values if history and history[-1].masked is not None else None
    masked = None if values_allowed else mask(question, matches, known=known)
    stable = stable_prompt(model, values_allowed=values_allowed)
    tail = question_tail(question, today=today, history=history, matches=matches, masked=masked,
                         scrub=scrub, model=model)
    raw: list[str] = []
    problems: list[str] = []
    text = ""
    for attempt in range(2):
        prompt = tail if attempt == 0 else (
            f"{tail}\n\nYOUR PREVIOUS ANSWER:\n{text}\n\nPROBLEMS:\n" + "\n".join(f"- {p}" for p in problems)
            + "\nReturn the corrected plan as one JSON object.")
        text = complete(stable, prompt)
        raw.append(text)
        try:
            plan = _unmask_plan(parse(text), masked)
        except (ValueError, ValidationError) as exc:
            problems = [_short(exc)]
            continue
        problems = check(plan, model, today)
        if not problems:
            return Outcome(plan, repaired=attempt > 0, raw=raw, masked=masked)
    return Outcome(Plan(kind="clarify", clarify=Clarify(
        about="other", question="I could not work out how to answer that from this data. Could you say it "
        "another way, naming what to measure and for when?")), repaired=True, problems=problems, raw=raw,
        masked=masked)


def _short(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:6])
    return str(exc)[:300]
