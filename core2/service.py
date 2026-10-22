"""core2's answer to a question, for any channel (DESIGN §12.1).

:func:`answer_question` takes everything it needs injected (the model, a
warehouse, the AI, the member index, today), so tests run it on DuckDB with a
recorded AI and production runs it on the workspace's governed connection and
configured provider. It returns the frame the web portal renders.

    question -> member names found -> plan (AI, checked, one repair) -> logical query
             -> SQL for the warehouse -> rows (governed) -> answer

Nothing here writes SQL from text: a question the plan cannot express is said so
(``unsupported``), and in compare mode the current pipeline's answer stands.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any

from core2 import ids
from core2.bootstrap import names
from core2.answer import beyond
from core2.answer.builder import _period_badge, against, build_answer, no_member, sparkline
from core2.answer.describe import describe
from core2.answer.drivers import answer_drivers
from core2.answer.extra import extra_facts
from core2.answer.forecast import answer_forecast
from core2.answer.snapshots import complete_snapshots
from core2.answer.suggestions import breakdowns, drills, follow_ups
from core2.compile.compiler import CompileError, compile_query, repeats_sql
from core2.compile.shapes import fill_edges
from core2.date_choice import choices as date_choices
from core2.plan import asks
from core2.plan.chips import french as chip_french, offered
from core2.model.schema import Attribute, DateRole, Entity, Join, Measure, SemanticModel
from core2.plan.followup import NEW_PREFIX, Reading, read_turn
from core2.plan.ir import Clarify, Compare, Filter, Plan, Window
from core2.plan.planner import Complete, Outcome, Turn, last_plan, period_words, plan_question, stands_for
from core2.plan.thread import kept_parts, points_above
from core2 import standing as S
from core2.standing import Standing
from core2.plan.values import (Masked, MemberIndex, ValueMatch, about_people, build_index, listable, normalise,
                                people_column, placed, quoted_spans, which_end, slugs_in)
from core2.resolve.figures import worked_out
from core2.resolve.resolver import Context, ResolveError, find_slug, resolve
from core2.resolve.time import Range, label as period_label, periods
from core2.warehouse.runner import Guarded, QueryFailed, Warehouse

log = logging.getLogger("querybot.core2")

HISTORY = 3          # turns of the conversation the planner sees
END = "end"                                               # Pending.field: "the worst one": which end of the answer
LEFT_OUT_CHIP = {"label": "Include the rows left out", "question": "Include the rows left out by default",
                 "label_fr": "Inclure les lignes écartées", "question_fr": "Inclure les lignes écartées par défaut"}
MAX_ROWS = 5000


@dataclass
class Services:
    model: SemanticModel
    warehouse: Warehouse
    complete: Complete
    index: MemberIndex
    today: dt.date
    values_allowed: bool = True                 # may member values reach the AI (False for regulated tenants)
    allowed_tables: set[str] | None = None      # model table keys the reader may use; None = all
    data_source: str = ""
    scrub: Callable[[str], str] | None = None   # personal data out of the question before the AI (regulated)
    split_units: bool = True                    # a quantity in several units is answered per unit (issue E4)
    # The workspace's AI writing a metric from a reader's words ("X = ..." in the chat); None: not offered.
    write_metric: Callable[[str], Complete] | None = None
    # A described dashboard's card as its tiles land (from the threads that build them); None: shown when done.
    progress: Callable[[dict[str, Any]], None] | None = None
    # The workspace's industry on its Compliance page ("banking"): which kind of business its dashboards are of,
    # where its data could be read as more than one.
    industry: str = ""
    # The business metrics suggested from the data and waiting for an admin (core2/kpi_proposals.py), by name:
    # an executive dashboard says they are coming.
    waiting: Callable[[], list[str]] | None = None
    # Every question the data answers, worked out and checked (core2/question_map.py): the planner is told the
    # routes counted on rows that name both lists, and a question that cannot be answered offers the nearest.
    question_map: Any = None
    # The reader's standing conditions kept in their profile ("always use the last 2 months", core2/standing.py),
    # and how one is kept there or taken out; None: kept for the conversation only.
    always: list[Standing] = field(default_factory=list)
    keep_always: Callable[[Standing], None] | None = None
    drop_always: Callable[[str], None] | None = None
    # The admin's switch: a question about business in general is answered from the AI's general knowledge,
    # labelled as such (core2/answer/beyond.py). On unless the admin turns it off.
    general_knowledge: bool = True
    # The workspace's AI sorting a question the data cannot answer (business in general, something else, or data it
    # does not hold), audited as its own call; None: such a question is said to be beyond the data.
    beyond: Complete | None = None


@dataclass
class Pending:
    """A question back that code completes: the plan waiting for the link the reader names."""

    plan: Plan
    field: str                       # the slug whose link the reply names (plan.via), "time.date", or END
    options: list[str]
    masked: Masked | None = None
    question: str = ""               # END: the question waiting for the end of the answer it means
    phrase: str = ""                 # END: the words that point at an end of the answer ("the worst one")


@dataclass
class Session:
    turns: list[Turn] = field(default_factory=list)
    pending: Pending | None = None
    own: dict[str, Measure] = field(default_factory=dict)     # metrics the reader defined in this chat
    # The chips of the answer on screen that change one setting ("Median instead"), by their words: the plan
    # each runs, with no AI (core2/plan/chips.py).
    offers: dict[str, Plan] = field(default_factory=dict)
    # Standing conditions set in this conversation ("from now on use the last 2 months"), and the reader's own
    # ones from their profile taken off for this conversation only (core2/standing.py).
    standing: list[Standing] = field(default_factory=list)
    paused: set[str] = field(default_factory=set)
    # The first next question the answer on screen offered after looking at itself (core2/agent/look.py): what
    # "yes" runs, with no AI.
    yes: Plan | None = None
    # The other dates the answer on screen could be counted by, by the words of their choice ("By ship date
    # instead"): the date a click picks, kept for the conversation (core2/date_choice.py).
    dates: dict[str, str] = field(default_factory=dict)


_ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3,
             "fifth": 4, "5th": 4, "sixth": 5, "6th": 5, "last": -1}
_FILLER = {"the", "one", "option", "number", "no", "please", "use", "i", "mean", "meant", "it", "is", "via",
           "by", "a", "an"}


def _instead(plan: Plan, model, outcome: Outcome) -> list[tuple[str, Plan]]:
    """The other figures a business ask may mean, when the figure was picked for it ("Gross amount instead"): the
    same answer for each, one click away with no AI."""
    out: list[tuple[str, Plan]] = []
    for slug in outcome.instead:
        measure = next((m for m in model.measures.values() if m.slug == slug), None)
        if measure is not None:
            out.append((f"{measure.business_name} instead",
                        asks.with_figure(plan, model, slug).model_copy(update={"follow_up": "refine"})))
    return out


def _plain(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def choose(reply: str, options: list[str]) -> str | None:
    """The option a reply picks: its name, its number ("2", "the second one"), or a start only it has."""
    text = _plain(reply)
    if not text or not options:
        return None
    plain = [_plain(o) for o in options]
    if text in plain:
        return options[plain.index(text)]
    words = [w for w in text.split() if w not in _FILLER]
    if len(words) == 1:
        word = words[0]
        if word.isdigit() and 1 <= int(word) <= len(options):
            return options[int(word) - 1]
        if word in _ORDINALS and _ORDINALS[word] < len(options):
            return options[_ORDINALS[word]]
    starts = [o for o, p in zip(options, plain) if p.startswith(text)]
    if len(starts) == 1:
        return starts[0]
    inside = [o for o, p in zip(options, plain) if p and re.search(rf"\b{re.escape(p)}\b", text)]
    return inside[0] if len(inside) == 1 else None


def _frame(question: str, text: str, **extra: Any) -> dict[str, Any]:
    answer = {"headline": text, "short_value": "", "comparison": "", "scope_badge": "", "scope_note": ""}
    return {"type": "assistant_response", "engine": "core2", "question": question, "answer": answer, "chart": None,
            "kpi": None, "data": None, "trust": {"engine": "core2"}, "confidence": {}, "insight_summary": "",
            "anomaly_callouts": [], "coverage_caveats": [], "follow_up_suggestions": [], **extra}


def _named_options(model: SemanticModel, options: list[str]) -> list[str]:
    """Options of a question back as readers name them: a metric, field or date by its name, never its slug."""
    from core2.resolve.resolver import find_slug

    out = []
    for option in options:
        found = find_slug(model, option.strip()) if option and re.fullmatch(r"[a-z0-9_.:]+", option.strip()) else None
        thing = found[1] if found else None
        name = (getattr(thing, "business_name", "") or getattr(thing, "name", "")) if thing is not None else ""
        out.append(name or option)
    return list(dict.fromkeys(out))


def _clarification(question: str, clarify: Clarify) -> dict[str, Any]:
    """A question back: the options in the sentence, and each one a chip that answers it."""
    options = [o for o in clarify.options if o][:6]
    asked = clarify.question.rstrip(" ?:.")
    if options:
        listed = options[0] if len(options) == 1 else ", ".join(options[:-1]) + f" or {options[-1]}"
        text = f"{asked}: {listed}?"
    else:
        text = clarify.question
    return _frame(question, text, clarify={"about": clarify.about, "question": clarify.question, "options": options},
                  follow_up_suggestions=[{"label": o, "question": o} for o in options])


# "Which group's price changed the most?": the answer names one member, its first row ("its revenue" is that
# member's). Not "which 10 warehouses ...": a list, where "its" names none.
_LEADER = re.compile(r"^\s*(?:which|what|who)\b(?!\s+\d)[^?]*\b(?:most|least|highest|lowest|largest|biggest|"
                     r"smallest|top|best|worst)\b", re.IGNORECASE)


def _names_a_leader(question: str) -> bool:
    return bool(_LEADER.match(question or "")) and not re.search(r"\b(?:top|bottom|first|last)\s+\d+\b", question,
                                                                 re.IGNORECASE)


def _which_end(question: str, phrase: str, first: str, last: str) -> dict[str, Any]:
    """Asked for "the worst one": whether that is the most or the least depends on what is measured."""
    text = f'Which one do you mean by "{phrase}": {first}, first in the answer above, or {last}, last in it?'
    return _frame(question, text, kind="which_end", clarify={"about": "member", "question": text,
                                                             "options": [first, last]},
                  follow_up_suggestions=[{"label": o, "question": o} for o in (first, last)])


def _read(question: str, last: Turn | None, model: SemanticModel, found: list[ValueMatch]) -> Reading:
    """How the question relates to the answer on screen, by its words (core2/plan/followup.py)."""
    if last is not None and last.plan is not None and last.plan.kind == "clarify":
        return Reading("open", "it may answer the question asked back")
    if last is not None and last.unanswered:
        # "and 2025? split both years" right after "fee income in 2024" was refused: read with that question.
        return Reading("open", "the question before it was not answered")
    return read_turn(question, previous=last.plan if last is not None else None, model=model,
                     members=[m.text for m in found])


def _describe(question: str, plan: Plan, services: Services, *, question_id: str = "",
              masked: Masked | None = None) -> dict[str, Any]:
    """A question about the data itself, answered from the model: a lead, a card per table or the thing asked
    about, what to know, and example questions to ask next (core2/answer/describe.py). The workspace's AI may open
    it with a paragraph written from the cards, after it is shown (``portal_overview``)."""
    about = list(plan.about) or [x for x in [beyond.meaning_asked(question)] if x]
    said = describe(services.model, about, today=services.today, allowed=services.allowed_tables,
                    values=services.values_allowed)
    if said.missing:
        # Asked in the reader's own words ("what does net sales mean?"): the name those words give it.
        from core2.plan.words import named

        about = [(named(services.model, a) or a) if a in said.missing else a for a in about]
        said = describe(services.model, about, today=services.today, allowed=services.allowed_tables,
                        values=services.values_allowed)
    if said.missing:
        # "What does EBITDA mean?" with nothing of that name in the data: business in general, or data it does not
        # hold, answered as such (core2/answer/beyond.py), never the description of everything else. The AI is told
        # what was missing in the words it may see: the plan's own, or the question's as it was given them (people's
        # names scrubbed, members masked), never the words as typed.
        words = [a for a in said.missing if a in plan.about] or \
            [x for x in [beyond.meaning_asked(_as_the_ai_sees(question, services, masked))] if x]
        why = f"this data has nothing called {', '.join(words)}" if words else "this data has nothing of that name"
        return _beyond(question, why, None, services, None, masked)
    frame = _frame(question, said.headline, kind="describe_data", sections=said.sections,
                   follow_up_suggestions=[{"label": q, "question": q} for q in said.examples],
                   trust={"engine": "core2", "data_source": services.data_source, "question_id": question_id,
                          "model_version": services.model.version})
    frame["answer"]["scope_note"] = said.note
    return frame


def _beyond(question: str, why: str, tried: Plan | None, services: Services, session: Session | None,
            masked: Masked | None, *, chat: bool = False) -> dict[str, Any]:
    """A question the data cannot answer, answered as a person would (core2/answer/beyond.py): business in general from
    the AI's general knowledge, labelled, where the admin allows it; anything else declined with questions to try;
    data the workspace does not hold said, with what it does hold, the nearest questions and a button asking the
    admin for it. Kept in the conversation as unanswered, as a refusal is. ``chat``: the planner read it as no
    question about the data ("bye!", "tell me a joke"), so it is never said to be data the workspace lacks."""
    lang = "fr" if beyond.french(question) else "en"
    said = describe(services.model, [], today=services.today, allowed=services.allowed_tables,
                    values=services.values_allowed)
    tables = [s["title"] for s in said.sections if s.get("label")]
    found = beyond.sort(_as_the_ai_sees(question, services, masked), why, tables, services.beyond, lang=lang)
    nearest = _nearest(services, tried) or beyond.nearest(question, found.wanted, said.examples)
    chips = [{"label": q, "question": q} for q in nearest]
    if found.kind == beyond.GENERAL and services.general_knowledge:
        return _refused(question, found.answer, tried, services, why, session, masked, unsupported=True,
                        kind="general_knowledge", general={"label": beyond.LABEL[lang]},
                        follow_up_suggestions=chips)
    if found.kind == beyond.PRODUCT:
        # A question about Solviz the topics in code did not answer: what it does, never another tool's how-to.
        return _refused(question, beyond.about_solviz(lang=lang), tried, services, why, session, masked,
                        unsupported=True, kind="product_help", follow_up_suggestions=chips)
    if chat and found.kind == beyond.NOT_HERE:
        found = beyond.Sorted(beyond.OFF_TOPIC)
    if found.kind in (beyond.OFF_TOPIC, beyond.GENERAL):
        text = beyond.decline(question, lang=lang) if found.kind == beyond.OFF_TOPIC else beyond.general_off(lang=lang)
        return _refused(question, text, tried, services, why, session, masked, unsupported=True, kind=found.kind,
                        follow_up_suggestions=chips)
    holds = describe_holds(said.sections)
    # "Your data does not hold monthly recurring revenue" where a field of that name is there (Learn kept it as each
    # subscription's property): never said, of what the reader may see.
    has = beyond.held(_visible_names(services), f"{found.wanted} {question}")
    lead, note = beyond.not_here(found.wanted, why, holds, lang=lang, has=has)
    frame = _refused(question, lead, tried, services, why, session, masked, unsupported=True, kind="data_not_here",
                     follow_up_suggestions=chips,
                     request_data={"name": (found.wanted or question)[:120], "question": question})
    frame["answer"]["scope_note"] = note
    return frame


def _as_the_ai_sees(question: str, services: Services, masked: Masked | None) -> str:
    """The question as the planner was given it: members masked where values are kept from the AI, people's
    details scrubbed where the workspace scrubs (core2/plan/planner.py::question_tail)."""
    text = masked.question if masked is not None else question
    return services.scrub(text) if services.scrub else text


def _visible_names(services: Services) -> list[str]:
    """The names of what the reader may see: figures, fields and dates on their tables, none hidden or sensitive."""
    model, allowed = services.model, services.allowed_tables
    seen = [c.business_name for c in model.columns.values()
            if not c.hidden and c.sensitivity == "none" and (allowed is None or c.table in allowed)]
    seen += [m.business_name for m in model.measures.values() if not m.hidden and (allowed is None or m.table in allowed)]
    return [n for n in dict.fromkeys(seen) if n]


def describe_holds(sections: list[dict[str, Any]]) -> str:
    """What the reader's data holds, in a few words: what it records, then the lookups ("order lines, returns and
    sales targets, with customers, products and stores")."""
    def listed(items: list[str]) -> str:
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" and {items[-1]}" if items else ""

    happened = [names.plural(s["title"].lower()) for s in sections if s.get("label") in ("Activity", "Snapshots")]
    lookups = [names.plural(s["title"].lower()) for s in sections if s.get("label") == "Lookup"]
    if happened and lookups:
        return f"{listed(happened)}, with {listed(lookups)}"
    return listed(happened or lookups)


_OPS = {"eq": "is", "in": "is one of", "ne": "is not", "not_in": "is none of", "gt": ">", "gte": ">=", "lt": "<",
        "lte": "<=", "between": "between", "contains": "contains", "starts_with": "starts with",
        "is_null": "is empty", "not_null": "is not empty"}


def _span(window: Window) -> str:
    unit = f"{'fiscal ' if window.fiscal else ''}{window.unit or 'period'}"
    n = window.n or 1
    return {"between": f"{window.start} to {window.end}", "since": f"since {window.start}",
            "until": f"until {window.end}", "last": f"the last {n} {unit}{'s' if n > 1 else ''}",
            "rolling": f"the last {n} {unit}{'s' if n > 1 else ''} up to today",
            "this": f"this {unit}", "to_date": f"{unit} to date", "previous": f"the previous {unit}"}.get(window.kind, "")


def _window(plan: Plan, model: SemanticModel) -> str:
    """The plan's dates in words: the date it is on, the window, the grain and the comparison."""
    time = plan.time
    said = [_span(time.window)]
    if time.grain:
        said.append(f"{time.grain} by {time.grain}")
    if time.compare is not None:
        other = _span(time.compare.window) if time.compare.window is not None else ""
        said.append(f"compared with {other or 'the ' + time.compare.kind.replace('_', ' ')}")
    text = ", ".join(part for part in said if part)
    if time.date:
        found = find_slug(model, time.date)
        role = found[1] if found is not None and isinstance(found[1], DateRole) else None
        text = f"{role.name if role is not None and role.name else time.date}: {text}".rstrip(": ")
    return text


def _considered(plan: Plan | None, model: SemanticModel) -> list[dict[str, str]]:
    """What a plan that was not answered asked for, in the model's names and with each one's table.

    A refusal's "How this answer was produced": no query ran, so this is what
    was looked at -- the measure and its table, the breakdown, the filters,
    the dates -- beside why it stopped.
    """
    if plan is None or plan.kind != "query":
        return []

    def table(key: str) -> str:
        return model.tables[key].business_name if key in model.tables else key

    def named(slug: str) -> str:
        found = find_slug(model, slug)
        thing = found[1] if found is not None else None
        if isinstance(thing, Attribute):
            column = model.columns.get(thing.column)
            return f"{thing.business_name or slug} ({table(column.table)})" if column else thing.business_name or slug
        if isinstance(thing, DateRole):
            return f"{thing.name or slug} ({table(thing.table)})"
        if isinstance(thing, (Measure, Entity)):
            return f"{thing.business_name or slug} ({table(thing.table)})"
        return slug.removeprefix("time:").replace("_", " ")

    rows: list[dict[str, str]] = []
    measures = [named(s) for s in plan.measures]
    measures += [f"{d.name} ({d.op} of {' and '.join(named(m) for m in d.measures)})" for d in plan.derived]
    measures += [f"{d.name} (days from {named(d.start)} to {named(d.end)})" for d in plan.durations]
    if measures:
        rows.append({"key": "measure", "value": "; ".join(measures)})
    if plan.group_by:
        rows.append({"key": "by", "value": ", ".join(named(s) for s in plan.group_by)})
    if plan.filters:
        rows.append({"key": "filter", "value": "; ".join(
            f"{named(f.field)} {_OPS.get(f.op, f.op)} {', '.join(str(v) for v in f.values)}".strip()
            for f in plan.filters)})
    dates = _window(plan, model)
    if dates:
        rows.append({"key": "dates", "value": dates})
    return rows


REFUSED, NOT_REACHED = "it was refused", "no answer came back"     # why a turn got no answer, in fixed words


def _unanswered(session: Session, question: str, why: str, masked: Masked | None = None) -> None:
    """The question kept in the conversation though it got no answer: its words and why not, never a plan (nothing
    it tried is carried). Only the latest one is kept, after the answers it follows."""
    session.turns = [t for t in session.turns if not t.unanswered]
    session.turns.append(Turn(question, None, masked, unanswered=why))
    del session.turns[:-(HISTORY + 1)]


def _keep_date(session: Session, model: SemanticModel, date: str, *, asked: bool = False) -> None:
    """The date the reader picked, kept for the rest of the conversation for the figures on its rows (a standing
    condition, shown above the box): it takes the place of any date kept before for those rows. Picking the rows'
    own date again on the badge lets the kept one go; a date picked when the answer asked which one is kept, the
    question having had no date of its own to fall back on."""
    role = model.date_roles.get(date) or next((r for r in model.date_roles.values() if r.slug == date), None)
    if role is None:
        return
    session.standing = [s for s in session.standing
                        if not (s.date and getattr(model.date_roles.get(s.date), "table", None) == role.table)]
    if asked or model.tables[role.table].default_date != role.key:
        session.standing.append(S.dated(role))


def _keep(session: Session, turn: Turn) -> None:
    """A turn kept in the conversation; a question that got no answer before it has been followed or passed by."""
    session.turns = [t for t in session.turns if not t.unanswered]
    session.turns.append(turn)
    del session.turns[:-HISTORY]


def _kept_said(plan: Plan, previous: Plan | None, question: str, model: SemanticModel) -> list[str]:
    """What a follow-up's answer kept from the answer above that its question did not say, in the reader's words:
    "Region is North", "in Q1 2026" -- each one said, so a scope kept is never a surprise."""
    filters, window = kept_parts(plan, previous, question)
    said = []
    for f in filters:
        found = find_slug(model, f.field)
        thing = found[1] if found is not None else None
        name = getattr(thing, "business_name", "") or getattr(thing, "name", "") or f.field.split(".")[-1].replace("_", " ")
        values = ", ".join(str(v) for v in f.values)
        said.append(f"{name} {_OPS.get(f.op, f.op)} {values}".strip())
    seg = plan.segment
    if seg is not None and previous is not None and previous.segment == seg and plan.follow_up == "refine":
        from core2.resolve.segments import WORDS

        found = find_slug(model, seg.of)
        thing = getattr(found[1], "business_name", "") if found is not None else ""
        words = seg.name or f"{WORDS[seg.status]} {names.plural((thing or seg.of).lower())}"
        if not seg.split and words.casefold() not in question.casefold():
            said.append(words)
    if window is not None:
        words = period_words(window)
        if words:
            said.append(words)
    return said


def _refused(question: str, text: str, plan: Plan | None, services: Services, stopped: str,
             session: Session | None = None, masked: Masked | None = None, **extra: Any) -> dict[str, Any]:
    """A refusal, with what was considered and why it stopped under "How this answer was produced".

    ``session``: the conversation, which keeps the question as an unanswered turn: the next question may go on with
    it ("and 2025?"), never with the answer before it as if it had not been asked."""
    if session is not None:
        _unanswered(session, question, REFUSED, masked)
    trust = {"engine": "core2", "considered": _considered(plan, services.model), "stopped": stopped,
             "model_version": services.model.version, "data_source": services.data_source,
             **extra.pop("trust", {})}
    return _frame(question, text, trust=trust, **extra)


_REFUSALS = {
    "denied": "That needs data you do not have access to: {message}.",
    "unconfirmed": "{message}. An admin can confirm it on the What Solviz learned page.",
    "unsupported": "I cannot answer that from this data: {message}.",
    "unknown": "I could not find that in this data: {message}.",
    "empty": "{message}.",
    "sensitive": "{message}.",
}


# A request to change the data, not to read it: "delete the cancelled movements of 2024", "close out all the open
# bookings", "supprime les fiches des employés partis". Solviz reads data; it never deletes, changes or adds
# rows, and says so before any planning (a French one was answered with a count). The rows it names can be shown.
# Money words: what can be put on a record or taken off one ("the late fee", "a $50 credit", "les frais de port").
_MONEY = (r"(?:fees?|charges?|surcharges?|discounts?|credits?|debits?|refunds?|payments?|penalt(?:y|ies)|interest|"
          r"tax(?:es)?|bonus(?:es)?|commissions?|adjustments?|rebates?|shipping|postage|freight|coupons?|vouchers?|"
          r"promo(?:tion)?s?|remises?|frais|cr[ée]dits?|p[ée]nalit[ée]s?|remboursements?|r[ée]ductions?)")
_HOLD = r"(?:holds?|blocks?|freezes?|locks?|suspensions?|restrictions?)"
# "... and show the new revenue", "what would margin be": a what-if, a question about the data as it would be ("... and
# show me the balance" or "I would like it done today" is not).
_WHAT_IF = (r"(?![^.?!]*\b(?:what\s+if|what\s+would|what(?:'s|\s+is|\s+are)\s+the\s+new|how\s+(?:much|many)\s+would|"
            r"how\s+(?:would|does|do|will)\s+[\w-]+|would\s+(?:[\w-]+\s+){0,3}be|(?:revenue|margin|sales|profit)\s+impact|"
            r"affect|and\s+(?:show|tell)\s+(?:me\s+)?(?:the\s+)?(?:new|resulting|impact|effect|revenue|margin)))")
_AMOUNT = r"(?:[$€£]\s?\d[\d,.]*|\d[\d,.]*\s*(?:%|percent|€|\$|dollars?|euros?)(?!\w))"
# Where money goes when it is put on something: a record, never the answer ("put the discount on the chart", "apply the
# promo filter to the results").
_ONTO = (r"\s+(?:to|onto|on|for|à|au|aux|sur)\s+(?!(?:the\s+|la\s+|le\s+|les\s+)?(?:chart|graph|graphique|axis|[xy][\s-]axis|"
         r"results?|table|tableau|answer|view|report|résultats?|lines?|rows?|columns?|colonnes?)\b)")
# Words that make a money word a field on the answer: "a payment date", "a discount amount", "a coupon code".
_FIELD_WORDS = (r"(?:rates?|ratios?|%|percent(?:age)?s?|shares?|counts?|totals?|columns?|colonnes?|figures?|numbers?|"
                r"breakdowns?|trends?|splits?|limits?|dates?|amounts?|codes?|flags?|methods?|status|types?|reasons?|"
                r"values?|levels?|by|per)")
_CHANGE = re.compile(
    r"^\s*(?:(?:please|can you|could you|pls)\s+)?(?:"
    # Verbs that only ever change data: whatever follows.
    r"(?:delete|erase|purge|wipe|supprime[rz]?|efface[rz]?)\b\s+"
    r"(?:(?:all|every|the|these|those|this|that|our|my|tous|toutes|les|la|le|l'|ces|cette|ce)\b\s*)*"
    # Money put on: "add a $50 credit to customer 1042", "add a late fee to every order", "apply the discount to every
    # order" ("add a refund rate", "add a payment date", "put the discount on the chart" are the answer's).
    r"|(?P<money>(?:add|apply|post|put|place|charge|ajoute[rz]?|applique[rz]?)\s+(?:a|an|un|une)\s+(?:"
    + _AMOUNT + r"\s+(?:[\w-]+\s+){0,2}?" + _MONEY + r"\b(?!(?:\s+[\w-]+){0,2}?\s+" + _FIELD_WORDS + r"\b)|"
    r"(?:(?!colonnes?\b|columns?\b)[\w-]+\s+){0,2}?" + _MONEY
    + r"\b(?!(?:\s+[\w-]+){0,2}?\s+" + _FIELD_WORDS + r"\b)(?:\s+(?:of|de)\s+" + _AMOUNT + r"|(?:\s+[\w-]+){0,4}?"
    + _ONTO + r"))" + _WHAT_IF
    + r"|(?:apply|post|put|place|charge|applique[rz]?)\s+(?:the|these|those|la|le|les|l')\s*(?:[\w-]+\s+){0,2}?"
    + _MONEY + r"(?:\s+[\w-]+)?" + _ONTO + _WHAT_IF + r")"
    # A hold put on or lifted: "lift the hold on account 7", "put a hold on customer 1042" ("apply a restriction to only
    # 2025 orders" is a filter).
    r"|(?P<held>(?:lift|release|clear|put|place|apply)\s+(?:the\s+|a\s+|an\s+|any\s+|these\s+|all\s+)?" + _HOLD
    + r"\s+(?:on|from|off|for|to)\s+(?!(?:(?:only|just)\s+)?(?:(?:19|20)\d{2}|this|last|recent)\b|the\s+(?:chart|graph|"
    r"results?|table|answer|view|report)\b)|"
    r"(?:lift|release|clear)\s+(?:the\s+|any\s+|these\s+|all\s+)?" + _HOLD + r"\s*[?.!]?\s*$)"
    # Verbs that are also figures ("credit sales in 2025", "refund rate by store", "write off amounts by region"): a command
    # only before the records they would change ("refund all orders placed by customer 1042", "charge every customer $5
    # per month", "write off these invoices") or a record by its number ("credit customer 1042 with $50", "write off
    # $500 for customer 1042"); never a year ("refund orders 2025 by month"). Nothing is ever written, so a command
    # read as a question is only answered as one; a question read as a command would not be answered at all.
    r"|(?P<owed>(?:credit|debit|charge|bill|refund|reimburse|write\s+off|waive)\s+(?:(?:the|these|those|all|every|each|"
    r"our|my)\s+(?!(?:year|month|week|quarter|period|time|day)s?\b)|(?:" + _AMOUNT + r"\s+(?:for|to|on|from)\s+)?"
    r"(?:customer|client|account|invoice|order|card)s?\s*(?:#|no\.?|number)?\s*(?:[A-Z]+-\d|(?!(?:19|20)\d{2}\b)\d)))"
    # Verbs that also read ("change in sales", "close rate", "add up"): only before what they would change.
    r"|(?:remove|drop|update|change|modify|edit|insert|add|create|close\s+out|close|cancel|void|archive|approve|"
    r"reject|mark|set|assign|merge|rename|reset|increase|raise|decrease|reduce|lower|cut|"
    r"augmente[rz]?|baisse[rz]?|réduis(?:ez)?|diminue[rz]?|retire[rz]?|modifie[rz]?|mets?\s+à\s+jour|mettez\s+à\s+jour|"
    r"ajoute[rz]?|crée[rz]?|annule[rz]?|clôture[rz]?|ferme[rz]?|archive[rz]?|valide[rz]?|renomme[rz]?)\b\s+"
    r"(?:(?:all|every|the|these|those|this|that|our|my|tous|toutes|les|la|le|l'|ces|cette|ce)\b\s*)+)",
    re.IGNORECASE)
_SHOW = {"en": "Show ", "fr": "Montre "}
# A change to the answer on screen, not to the data: "drop the count and the average, and show each balance as % of
# the region's total instead", "remove the cost column", "add their region". Only the verbs that also read can be
# one; "delete" and "purge" never are.
_DESTROYS = re.compile(r"^\s*(?:(?:please|can you|could you|pls)\s+)?(?:delete|erase|purge|wipe|supprime|efface)",
                       re.IGNORECASE)
# Only the verbs that take a figure off or put one on: never one that writes a value ("set", "update", "mark",
# "cancel", "reset", "change"), and never with a new value ("to $500", "by 5%", "as paid").
_EDIT_VERB = re.compile(r"^\s*(?:(?:please|can you|could you|pls)\s+)?(?:drop|remove|hide|add|take\s+out|leave\s+out|"
                        r"enlève|enleve|retire|ajoute|masque)\b", re.IGNORECASE)
_NEW_VALUE = re.compile(r"\b(?:to|by|as|at|à|de)\s+[$€£]?\s?\d|\b(?:to|as)\s+(?:paid|closed|open|cancelled|canceled|done|"
                        r"complete|completed|active|inactive|zero|null|blank|empty)\b", re.IGNORECASE)
# With an answer on screen, "add the units sold", "add the sales rep who took the order", "remove the practice" put a
# field or figure on the answer or take one off; only a new record, or every record at once, is data.
_NEW_RECORD = re.compile(r"\b(?:new|records?|entr(?:y|ies)|nouveau|nouvelle|nouveaux|fiches?)\b|"
                         r"^\s*(?:(?:please|can you|could you|pls)\s+)?\w+(?:\s+out)?\s+(?:(?:all|every|tous|toutes)\b|"
                         r"(?:the\s+|les\s+)?\d)|"     # every record at once; an amount put on ("add the 5 units")
                         r"\b(?:to|into|from|in)\s+(?:the\s+)?(?:database|system|crm|ledger|base|stock|inventory|"
                         r"balance|account)\b", re.IGNORECASE)
# Money put on records or taken off them is data too, however little it names: "add the shipping fee to every order",
# "add the late fee to all customer accounts", "ajoute les frais de port à chaque commande"; so is anything put on or
# taken off a numbered record ("add the coupon to order SO-1001", "remove the credit from customer 1042"), a hold put
# on or lifted ("remove the freeze for these accounts"), or records put on or taken off the books. A field put beside
# the answer is not ("add the category to each order line", "add the cost to each order", "add the credit to each
# customer", "remove the flag column", "add the author to the books"). Only the verbs that can edit the answer reach
# this: "add", "remove", "ajoute", "retire".
_RECORD = r"(?:orders?|invoices?|accounts?|records?|bills?|transactions?|claims?)"
_NUMBERED = (r"(?:(?:order|invoice|account|record|bill|transaction|claim|customer|client|card)\s*(?:#|no\.?|number)?|"
             r"(?:orders|invoices|accounts|records|bills|transactions|claims|customers|clients|cards)\s*(?:#|no\.?|"
             r"numbers?))\s*[A-Z]*-?\d|(?:orders|invoices|accounts|records|bills|transactions|claims|customers|clients|"
             r"cards)\s+[A-Z]*-?\d[\w-]*(?:\s*(?:,|and|&)\s*[A-Z]*-?\d[\w-]*)+")
_ON_RECORDS = re.compile(
    r"^\s*(?:(?:please|can you|could you|pls)\s+)?add\s+(?:the\s+|a\s+|an\s+|these\s+|those\s+)?(?:[\w$€£%.,-]+\s+){0,2}?"
    + _MONEY + r"(?:\s+(?:costs?|fees?|charges?))?\s+(?:to|onto|on)\s+(?:(?:the|every|each|all|these|"
    r"those|this|that|our)\s+)?(?:[\w-]+(?:'s|’s|s'|’)?\s+){0,2}?" + _RECORD
    + r"\b(?!\s+(?:lines?|items?|table|list|report|view|chart|columns?|values?|amounts?|totals?|counts?|sizes?|levels?|"
    r"dates?)\b)|"
    r"\b(?:on|from|off|to|onto|into)\s+(?:(?:the|these|those|this|that)\s+)?" + _NUMBERED + r"|"
    r"\b(?:à|au|aux|sur|de|du)\s+(?:la\s+|le\s+|l')?(?:commande|facture|compte|client)\s*(?:n°|no\.?|numéro|#)?\s*"
    r"[A-Z]*-?\d|"
    r"^\s*(?:ajout|retir|enl[eè]v)\w*\s+(?:la\s+|le\s+|les\s+|l'|une?\s+|des\s+)?(?:[\w-]+\s+){0,2}?" + _MONEY
    + r"(?:\s+(?:de|du|des|d')\s*[\w-]+)?\s+(?:à|au|aux|sur)\s+|"
    r"^\s*(?:(?:please|can you|could you|pls)\s+)?add\s+(?:the\s+|a\s+|an\s+)?(?:[\w-]+\s+)?(?:" + _HOLD
    + r"|tags?|notes?)\s+(?:to|onto|on)\s+(?:(?:the|every|each|all|these|those|this|that|our)\s+)?(?:[\w-]+(?:'s|’s)\s+)?"
    + r"(?:" + _RECORD + r"|customers?|clients?|cards?)\b(?!\s+(?:lines?|items?|table|list|report|view|chart|columns?)\b)|"
    r"^\s*(?:(?:please|can you|could you|pls)\s+)?(?:remove|take\s+off)\s+(?:the\s+|a\s+|any\s+|these\s+|all\s+)?"
    r"(?:" + _HOLD + r"|flags?)\s+(?:on|from|off|for)\b|"
    r"^\s*(?:(?:please|can you|could you|pls)\s+)?remove\s+(?:the\s+|any\s+|these\s+|all\s+)?" + _HOLD + r"\s*[?.!]?\s*$|"
    r"\b(?:" + _RECORD + r"|" + _MONEY + r"|entr(?:y|ies)|rows?|them|it|these|those)\b[^,.;]{0,40}?\b(?:to|into|onto|from|"
    r"off)\s+the\s+"
    r"(?:books|ledger)\b(?!\s+(?:category|genre|section|department|segment|table|columns?)\b)", re.IGNORECASE)
_ANSWER_EDIT = re.compile(
    r"\b(?:instead|column|columns|colonne|colonnes|from\s+the\s+(?:table|answer|list|chart|report|view)|"
    r"as\s+(?:a\s+)?(?:%|percent|share)|"
    r"%\s+of|next\s+to|beside|alongside|"
    r"(?:the|their|its)\s+(?:count|counts|average|averages|avg|total|totals|sum|share|percentage|%|figure|figures|"
    r"cost|costs|margin|change|ratio|rate|number|numbers|chart|table|labels?|decimals?))\b", re.IGNORECASE)


def changes_data(question: str, *, on_screen: bool = False) -> str | None:
    """The rows a request to change the data names, as a question that shows them; None for a question.
    ``on_screen``: an answer is on screen, which "add the units sold" or "drop the practice" changes."""
    found = _CHANGE.match(question or "")
    if not found:
        return None
    put_on = bool(found.group("money") or found.group("owed") or found.group("held"))
    if not put_on and not _DESTROYS.match(question) and _EDIT_VERB.match(question) and not _NEW_VALUE.search(question) \
            and not _ON_RECORDS.search(question) \
            and (_ANSWER_EDIT.search(question) or on_screen and not _NEW_RECORD.search(question)):
        return None          # "drop the count and show ... instead": the answer changes, the data never
    rest = question[found.end():].strip().rstrip("?.!").removesuffix("for me").removesuffix("s'il te plaît")
    named = re.search(r"\b(?:customer|client|account|invoice|order|card)s?\s*(?:#|no\.?|number)?\s*[A-Z]*-?\d[\w-]*"
                      r"(?:\s*(?:,|and|&)\s*[A-Z]*-?\d[\w-]*)*", question, re.IGNORECASE)
    if found.group("owed") and named:
        rest = named.group(0)                            # "credit customer 1042 with $50": that customer
    elif found.group("owed"):
        rest = re.sub(r"^(?:(?:the|these|those|all|every|our|my)\s+)+", "", rest)   # "write off the overdue invoices"
    elif found.group("money") or found.group("held"):
        # "add a $50 credit to customer 1042", "lift the hold on these accounts": the records it would go on or come
        # off, when the request names them
        on = re.findall(r"\b(?:to|onto|on|from|off|for|à|au|aux|sur)\s+(?:(the|these|those|this|that|la|le|les|l')\s+|"
                        r"(every|each|all|chaque|toutes\s+les|tous\s+les)\s+)?([^,;?.!]+)", question, re.IGNORECASE)
        _, every, rest = on[-1] if on else ("", "", "")
        rest = rest.strip()
        if every and every.lower() in ("every", "each", "chaque"):
            # "to every order in 2025": the orders in 2025
            head = re.split(r"\s+(?=(?:in|from|for|of|with|on|at|by|during|since|en|de|du|des)\b)", rest, maxsplit=1)[0]
            rest = (head if head.endswith(("s", "x")) else head + "s") + rest[len(head):]
    rest = rest.removesuffix("please").strip(" ,")
    if re.match(r"(?:to|into|onto|from|off|on|for)\b", rest, re.IGNORECASE):
        rest = ""                                        # "add these to the books": nothing named to show
    if re.match(r"\s*(?:(?:please|can you|could you|pls)\s+)?(?:update|change|modify|edit|mark|set|assign|rename|"
                r"modifie|mets|mettez|renomme)", question, re.IGNORECASE):
        rest = re.sub(r"\s+(?:to|as|en|à|comme)\s+[\w'-]+$", "", rest)      # "... to closed": the new value goes
    if re.match(r"\s*(?:(?:please|can you|could you|pls)\s+)?(?:increase|raise|decrease|reduce|lower|cut|augment|baiss|"
                r"rédui|diminu)", question, re.IGNORECASE):
        # "... by 10%", "... to $50,000": the change goes, the figure it would change stays
        rest = re.sub(r"\s+(?:by|to|of|de|à)\s+[$€]?\s*[\d][\d.,\s]*\s*(?:%|percent|points?|pts|\$|€)?$", "", rest)
    french = bool(re.match(r"\s*(?:supprim|effac|retir|modifi|mets|mettez|ajout|appliqu|cré|annul|clôtur|ferm|archiv|"
                           r"valid|renomm|augment|baiss|rédui|diminu)", question, re.IGNORECASE))
    lead = (_SHOW["fr"] + "les ") if french else (_SHOW["en"] + "the ")
    if french and put_on:
        one = re.match(r"(commande|facture|client|compte)\b", rest, re.IGNORECASE)
        if one:                                          # "au client 1042": le client 1042
            lead = _SHOW["fr"] + ("la " if one.group(1).lower() in ("commande", "facture") else "le ")
    return f"{lead}{rest}" if rest else ""


# A message that only thanks or greets, in English or French: answered as a person would, with no AI.
_THANKS = re.compile(r"^\s*(?:thanks|thank\s+you|thx|ty|cheers|great|perfect|awesome|excellent|merci|parfait|génial|"
                     r"super)(?:[\s,!.]+(?:so\s+much|a\s+lot|very\s+much|again|thanks|thank\s+you|beaucoup|bien|encore|"
                     r"merci|c'est\s+parfait|that's\s+(?:great|perfect)))*[\s!.]*$", re.IGNORECASE)
_HELLO = re.compile(r"^\s*(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|bonjour|salut|bonsoir)"
                    r"(?:[\s,!.]+(?:there|solviz|à\s+vous|tout\s+le\s+monde))*[\s!.]*$", re.IGNORECASE)
_FRENCH_TALK = re.compile(r"\b(?:merci|parfait|génial|bonjour|salut|bonsoir|beaucoup)\b", re.IGNORECASE)


def small_talk(question: str, *, french: bool | None = None) -> str | None:
    """The reply to a message that only thanks or greets ("thanks!", "merci beaucoup", "hi"); None for a question."""
    thanks, hello = bool(_THANKS.match(question or "")), bool(_HELLO.match(question or ""))
    if not thanks and not hello:
        return None
    fr = bool(_FRENCH_TALK.search(question)) if french is None else french
    if thanks:
        return "Avec plaisir. Voulez-vous regarder autre chose ?" if fr else \
            "You're welcome. Is there anything else you'd like to look at?"
    return ("Bonjour ! Posez-moi une question sur vos données, par exemple un total, une tendance ou un classement."
            if fr else "Hello! Ask me about your data, for example a total, a trend or a ranking.")


def answer_question(question: str, services: Services, session: Session, *, question_id: str = "",
                    planned: Plan | None = None) -> dict[str, Any]:
    """The portal frame answering ``question``; the session keeps the plan for follow-ups. ``planned``: the plan
    written for it in code (a start-page card's), run as it is with no AI.

    A question that defines a metric ("X = how it is counted. Show it by month") has it written by the
    workspace's AI and kept in this chat (core2.plan.own_metric); the rest of the question is answered with it,
    and the frame carries the metric so the reader sees how it was counted and can ask their admin to keep it.
    """
    from core2 import dashboards
    from core2.plan import own_metric

    if dashboards.asked_for(question) is not None:
        return _described(question, services, session)     # "build me a dashboard of ...": before planning
    if planned is None:
        # "From now on use the last 2 months", "always leave out 'Wholesale'", "stop using the last 2 months":
        # read and kept in code, never asked of the AI (core2/standing.py); "remove ..." is no change to the data.
        said = S.instruction(question)
        if said is not None:
            return _standing_set(question, *said, services, session)
        gone = S.letting_go(question, _standing_now(services, session))
        if gone is not None:
            return _standing_dropped(question, gone, services, session)
    small = small_talk(question) if planned is None else None
    if small is not None:
        return _frame(question, small, kind="smalltalk")     # "thanks!", "bonjour": answered with no AI
    helped = beyond.product_help(question) if planned is None else None
    if helped is not None:
        # "How do I share a dashboard?": what Solviz does, answered in code with no AI.
        fr = beyond.french(question)
        chip = "Quelles données avez-vous ?" if fr else "What data do you have?"
        return _frame(question, helped[0], kind="product_help", help_topic=helped[1],
                      follow_up_suggestions=[{"label": chip, "question": chip}])
    last = session.turns[-1].plan if session.turns else None
    shown = changes_data(question, on_screen=last is not None and last.kind == "query")
    if shown is not None:
        french = shown.startswith(_SHOW["fr"])
        text = ("Je lis les données sans jamais les modifier : je ne peux ni supprimer, ni changer, ni ajouter de "
                "lignes. Je peux vous montrer celles dont il s'agit." if french else
                "I read the data and never change it: I cannot delete, change or add records. I can show you the "
                "ones you mean.")
        chips = [{"label": shown, "question": shown}] if shown else []
        return _frame(question, text, kind="changes_data", follow_up_suggestions=chips)
    defined = own_metric.definition_in(question) if services.write_metric is not None else None
    made: Measure | None = None
    if defined is not None:
        # Defined again under the same name: the new definition replaces the old one, under the same slug.
        kept = {k: m for k, m in session.own.items() if m.business_name.casefold() != defined.name.casefold()}
        try:
            made = own_metric.write(own_metric.with_own(services.model, kept), defined,
                                    services.write_metric(defined.how), allowed=services.allowed_tables,
                                    taken={m.slug for m in kept.values()})
        except own_metric.OwnMetricError as exc:
            return _frame(question, f"I could not make {defined.name} a metric: {exc}", kind="own_metric_refused")
        except Exception as exc:  # noqa: BLE001 - said to the reader, never raised into their turn
            log.warning("core2: a reader's metric could not be written: %s", exc, exc_info=True)
            return _frame(question, f"I could not make {defined.name} a metric: something went wrong while writing "
                                    "it. Try saying how it is counted another way.", kind="own_metric_refused")
        session.own = {**kept, made.key: made}
    if session.own:
        services = replace(services, model=own_metric.with_own(services.model, session.own))
    last = session.turns[-1] if session.turns else None
    if last is not None and last.unanswered and _plain(last.question) == _plain(question):
        # The question asked again after no answer came back (the AI's per-minute limit, a timeout, a refusal): read
        # as it was the first time, after the answer on screen, never as going on with itself. A margin asked again
        # after a rate limit had kept the figure and the condition of the answer above.
        session.turns = session.turns[:-1]
    kept_before = session.turns[-1] if session.turns else None
    try:
        payload = _answer_question(defined.ask if defined is not None else question, services, session,
                                   question_id=question_id, planned=planned if defined is None else None)
    except Exception:
        # No answer came back (the AI rate limited or timed out): the question still stays in the conversation, so
        # the next one ("the late ones") goes on with it rather than with the answer before it. Kept once: a turn
        # this question already left (a refusal) stands.
        if (session.turns[-1] if session.turns else None) is kept_before:
            _unanswered(session, question, NOT_REACHED)
        raise
    if made is not None:
        payload["question"] = question
        payload["own_metric"] = {**own_metric.card(services.model, made, defined), "measure": made}
    used = {m for m in ((payload.get("plan") or {}).get("measures") or []) if isinstance(m, str)}
    mine = [m.business_name for m in session.own.values() if m.slug in used]
    if mine:
        payload["own_metrics_used"] = mine
    return payload


def _described(question: str, services: Services, session: Session | None = None) -> dict[str, Any]:
    """A dashboard the reader asked for: the recipe the request names, or designed by one AI call over the
    workspace's names (it reads the whole request: "create a revenue dashboard" names a subject, where "revenue"
    alone reads as one tile); "create a dashboard using this" is built around the answer on screen. Each tile is
    then answered as a question of its own (a new conversation each), several at once. The built tiles, with their
    plans, are for the server to save as the reader's (``dashboard_tiles``); the chat is told how far it is while
    they are built, never what they show."""
    from core2 import dashboards

    told = services.progress
    shown = None
    if dashboards.on_screen(dashboards.asked_for(question)):
        last = shown_answer(session)
        if last is None:
            return _frame(question, "There is no answer on screen to build a dashboard from. Ask a question first, "
                                    "then say \u201ccreate a dashboard using this\u201d; or say what the dashboard "
                                    "should show (\u201ca dashboard of net sales by store and month\u201d).",
                          kind="dashboard_refused")
        if session is not None and set(last.plan.get("measures") or []) & {m.slug for m in session.own.values()}:
            return _frame(question, "This answer uses a metric you defined in this chat, which a dashboard cannot "
                                    "draw again: ask your admin to save it for everyone first.",
                          kind="dashboard_refused")
        shown = last

    def answer(asked: str, planned: dict[str, Any] | None = None) -> dict[str, Any]:
        return _answer_question(asked, services, Session(),
                                planned=None if planned is None else Plan.model_validate(planned))

    try:
        draft = dashboards.build(services.model, question, services.complete, answer,
                                 scrub=services.scrub, personal_shown=services.scrub is not None,
                                 progress=None if told is None else lambda d: told(
                                     {"type": "dashboard_building", "engine": "core2", "question": question,
                                      "building": dashboards.status(d)}),
                                 today=services.today, allowed=services.allowed_tables, shown=shown,
                                 industry=services.industry)
    except dashboards.DesignError as exc:
        return _frame(question, f"I could not design that dashboard: {exc}. Try naming what it should show, "
                                "as \"a dashboard of net sales by store and month\".", kind="dashboard_refused")
    built = [t for t in draft.tiles if not t.error]
    if not built:
        return _frame(question, "I could not build any tile of that dashboard: " + "; ".join(
            sorted({t.error for t in draft.tiles})[:3]), kind="dashboard_refused")
    return {"type": "dashboard_built", "engine": "core2", "question": question, "name": draft.name,
            "tabs": list(draft.tabs), "dashboard_tiles": [dashboards.kept(t) for t in built],
            "not_built": dashboards.not_built(draft), "checks": list(draft.checks),
            "missing": list(draft.missing), "instead": dashboards.in_place(draft),
            "widened": dashboards.widened(draft), "waiting": _waiting(services) if draft.executive else []}


def _routes(services: Services) -> str:
    if services.question_map is None:
        return ""
    from core2.question_map import routes_text

    return routes_text(services.question_map)


def _nearest(services: Services, plan: Plan | None) -> list[str]:
    """The questions closest to one that cannot be answered, that this reader can ask: the same figure by
    another grouping, or the same grouping of another figure (core2/question_map.py)."""
    if services.question_map is None or plan is None or plan.kind != "query":
        return []
    from core2.question_map import nearest

    found = nearest(services.question_map, measure=plan.measures[0] if plan.measures else "",
                    group=plan.group_by[0] if plan.group_by else "", allowed=services.allowed_tables)
    return [e.question for e in found]


def _answerable_options(services: Services, options: list[str], tried: Plan | None, most: int = 6) -> list[str]:
    """A question back's options with none the question map knows cannot be answered, and the nearest questions
    that can be put first."""
    from core2.question_map import ANSWERABLE

    failing = {e.question.casefold() for e in services.question_map.entries if e.status != ANSWERABLE}
    nearest = _nearest(services, tried)
    kept = [o for o in options if o and o.casefold() not in failing and o not in nearest]
    return [*nearest, *kept][:most]


def _waiting(services: Services) -> list[str]:
    if services.waiting is None:
        return []
    try:
        return list(services.waiting())
    except Exception as exc:  # noqa: BLE001 - the dashboard is built; the line about what is coming is left out
        log.warning("core2: the business metrics waiting for an admin could not be read: %s", exc)
        return []


def shown_answer(session: Session | None) -> Any:
    """The answer on screen ("this" in "add this to my dashboard"): the last answer of the conversation that drew
    rows, with its question, plan and title; None when there is none."""
    from core2.dashboards import Shown

    last = next((t for t in reversed(session.turns) if t.title and t.plan is not None), None) if session else None
    if last is None:
        return None
    return Shown(last.question, last.plan.model_dump(mode="json", exclude_defaults=True), last.title)


def portal_shown(session_key: str) -> Any:
    """The answer on screen in a portal conversation, for "add this to my dashboard" (gateway/core2_bridge.py)."""
    with _LOCK:
        session = _SESSIONS.get(session_key)
    return shown_answer(session)


def _answer_question(question: str, services: Services, session: Session, *, question_id: str = "",
                     forced: str | None = None, planned: Plan | None = None) -> dict[str, Any]:
    """``forced``: "refine" or "new" when the reader said which ("Ask as a new question", or a button).
    ``planned``: the plan written for the question in code (a recipe's dashboard tile), run as it is (no AI).

    An answer to the AI's plan that looks wrong when it is checked (nothing found because of a condition the
    question never states, a share of exactly 0% or 100%) is asked of the AI once more, with what the check found
    (core2/agent/again.py). The second reading is kept only when it puts that right; the answer says it looked
    again, and the first reading stays one click away. Otherwise the first answer stands, as it was."""
    from core2.agent import again as A

    before = A.state(session)
    payload = _answer_once(question, services, session, question_id=question_id, forced=forced, planned=planned)
    looked = payload.pop(A.KEY, None)
    if looked is None:
        return payload
    asked, how, first, reasons = looked
    after = A.state(session)
    A.restore(session, before)
    try:
        second = _answer_once(asked, services, session, question_id=question_id, forced=how,
                              feedback=A.feedback(first, reasons, values=services.values_allowed))
        second.pop(A.KEY, None)
    except Exception as exc:  # noqa: BLE001 - the first answer stands
        log.info("core2: looking again at %r failed: %s", asked, exc)
        second = None
    if second is not None and A.better(second, first, payload, reasons):
        A.say(second, payload, first, reasons, session, _plain)
        return second
    A.restore(session, after)
    A.unchanged(payload, reasons)
    return payload


def _answer_once(question: str, services: Services, session: Session, *, question_id: str = "",
                 forced: str | None = None, planned: Plan | None = None, feedback: str = "") -> dict[str, Any]:
    """One reading of the question, answered (see _answer_question). ``feedback``: what the check of a first
    reading found, for the AI writing the plan again."""
    start = time.perf_counter()
    model = services.model
    from_ai = False
    pending, session.pending = session.pending, None
    if pending is not None and pending.field == END:
        end = choose(question, pending.options)
        if end is not None:      # the member they meant, in quotes where the words pointed: their question again
            asked = pending.question.replace(pending.phrase, f'"{end}"', 1)
            return _answer_once(asked, services, session, question_id=question_id, forced="refine")
        pending = None
    if forced is None and NEW_PREFIX.match(question) and NEW_PREFIX.sub("", question, count=1).strip():
        question, forced = NEW_PREFIX.sub("", question, count=1).strip(), "new"
    last = session.turns[-1] if session.turns else None
    # The answer on screen: the last turn that got one (a question refused since is said, not drawn).
    seen = next((t for t in reversed(session.turns) if not t.unanswered), None)
    following = False
    matches: list[ValueMatch] = []
    named: list[ValueMatch] = []        # the members the question names (quoted, or beyond doubt): each must be held
    unquoted_named: list[ValueMatch] = []   # those of them written without quotes: said, and one click removes them
    picked = choose(question, pending.options) if pending is not None else None
    offer, session.offers = (session.offers.get(_plain(question)) if forced is None else None), {}
    date_pick, session.dates = (session.dates.get(_plain(question)) if forced is None else None), {}
    kept_date: str | None = None
    asked_date = False          # picked as the answer to "which date?", where no date was the rows' own
    yes, session.yes = session.yes, None
    if offer is None and yes is not None and forced is None and pending is None and _YES.match(question):
        offer = yes             # "yes please" to the answer's own question: its first offer, with no AI
    if planned is not None:
        outcome = Outcome(planned)
    elif offer is not None and (pending is None or picked is None):
        # A chip of the answer on screen (or of a reply that found no answer, "Did you mean ...?"): its plan, run
        # as it is (no AI).
        outcome = Outcome(offer, masked=seen.masked if seen is not None else None)  # the chip says what it changed
        kept_date = date_pick
    elif pending is not None and picked is not None:
        # The reply names one of the links or dates offered: the waiting plan, completed by code.
        outcome = Outcome(_completed(pending.plan, pending.field, picked, model).model_copy(
            update={"follow_up": "refine"}), masked=pending.masked)
        if pending.field == "time.date":
            kept_date = outcome.plan.time.date       # the date picked counts the rest of the conversation too
            asked_date = True
    else:
        # A name the reader put in quotes is a member, and so is one written as people write names ("Dish Soap",
        # "at Riverside") that names one member beyond doubt; ordinary words never narrow the answer. Every
        # name written, quoted or not, is still withheld from the AI where member values may not reach it.
        found = _visible(services.index.quoted(question), model, services.allowed_tables)
        # "Virtual = F", "DOC = 31": the value of the field named before the sign, for that field only.
        assigned = _visible(_assigned(model, services.index, question), model, services.allowed_tables)
        found = [*(m for m in found if not any(a.start < m.end and m.start < a.end for a in assigned)), *assigned]
        unquoted_named = [m for m in _visible(services.index.written_names(question, not_names=_own_names(model),
                                                                           prefer=lambda f: _one_reading(model, question, f)),
                                              model, services.allowed_tables)
                          if not any(f.start < m.end and m.start < f.end for f in found)]
        found = sorted([*found, *unquoted_named], key=lambda m: (m.start, m.attribute))
        named = found
        hidden = [] if services.values_allowed else _visible(services.index.match(question), model,
                                                             services.allowed_tables)
        # "Break the first one down", "monthly sales for that division in 2026": the member of the answer on
        # screen the words point at. A turn that points at one follows that answer, however much else it says.
        # An answer with no list on screen (one store's months) leaves "the third one" pointing at the list before
        # it: the top three stores the series was drawn from.
        listed = seen if seen is not None and seen.shown else next(
            (t for t in reversed(session.turns[-HISTORY:]) if t.shown), None)
        pointed = _visible(placed(question, listed.shown, found, leader=_names_a_leader(listed.question)), model,
                           services.allowed_tables) if listed is not None else []
        end = which_end(question, seen.shown, [*found, *pointed]) if seen is not None and seen.shown \
            and seen.plan is not None and forced != "new" else None
        if end is not None:
            session.pending = Pending(seen.plan, END, [end[1], end[2]], question=question, phrase=end[0])  # type: ignore[arg-type]
            return _which_end(question, *end)
        reading = (Reading(forced, "the reader said so") if forced else
                   Reading("refine", "it points at a member of the answer on screen") if pointed else
                   Reading("refine", "it points at the member the answer above was narrowed to")
                   if points_above(question, seen.plan if seen is not None else None) else
                   _read(question, last, model, found))
        matches = found
        if reading.kind != "new" and pointed:
            matches = sorted([*found, *pointed], key=lambda m: m.start)
        # A new question is planned on its own: nothing of the answer before it can leak in.
        from_ai = True
        outcome = plan_question(model, question, services.complete, today=services.today,
                                history=[] if reading.kind == "new" else session.turns[-HISTORY:], matches=matches,
                                values_allowed=services.values_allowed, scrub=services.scrub, hidden=hidden,
                                written=_visible(services.index.match(question), model, services.allowed_tables),
                                defined=[d for t in session.turns[-HISTORY:] if t.plan is not None for d in t.plan.derived],
                                reading=reading.kind if reading.kind in ("refine", "new") else None,
                                on_screen=list(seen.shown) if seen is not None and reading.kind != "new" else None,
                                routes=_routes(services), feedback=feedback)
        following = seen is not None and seen.plan is not None and seen.plan.kind == "query"
    plan = outcome.plan
    if plan.kind == "smalltalk":
        said = small_talk(question)
        if said is None:
            # "Tell me a joke", "how are you?": not a greeting or thanks, so declined as anything beyond the data is.
            return _beyond(question, "it is not a question about the data", None, services, session, outcome.masked,
                           chat=True)
        return _frame(question, said, kind="smalltalk")
    if plan.kind == "describe_data":
        return _describe(question, plan, services, question_id=question_id, masked=outcome.masked)
    if plan.kind == "unsupported":
        why = " ".join(plan.notes) or "nothing in this data measures it"
        # After a repair round, the plan that was tried first says what was looked at.
        return _beyond(question, why, outcome.tried or plan, services, session, outcome.masked)
    if plan.kind == "clarify" and plan.clarify is not None:
        # Kept in the conversation: the reply ("the first one") answers this question.
        _keep(session, Turn(question, plan, outcome.masked))
        clarify = plan.clarify
        if clarify.about == "member" and clarify.options:
            # An option written as a description of a row ("top supplier from the previous list") is no member.
            clarify = clarify.model_copy(update={"options": [o for o in clarify.options if not stands_for(o)]})
        if clarify.about == "member" and not clarify.options and seen is not None and seen.shown:
            # "Which warehouse do you mean by 'the first one'?": the members of the answer on screen, one click each.
            listed = list(dict.fromkeys(str(s[1]) for s in seen.shown if s[1] is not None))[:5]
            clarify = clarify.model_copy(update={"options": listed})
        if clarify.about != "member" and clarify.options:
            # "number_of_doctors" is what the planner calls it; the reader sees, and clicks, "Number of doctors".
            clarify = clarify.model_copy(update={"options": _named_options(model, clarify.options)})
        if outcome.problems and services.question_map is not None:
            # The plans did not check out: never offer one the data cannot answer either, and offer the nearest
            # it can (core2/question_map.py).
            clarify = clarify.model_copy(update={"options": _answerable_options(services, clarify.options,
                                                                                 outcome.tried)})
        asked = _clarification(question, clarify)
        if outcome.problems:      # the planner's plans did not check out: what it tried, and why not
            asked["trust"] = {**asked["trust"], "considered": _considered(outcome.tried, model),
                              "stopped": " ".join(outcome.problems)}
        return asked

    plan = _roles_named(plan, model, services.index)        # "per manager": the link, never a level called so
    plan = _under_a_parent(plan, question, model)           # 'accounts under "Operating Expenses"': its children
    # A quoted name of a field whose names are not read ("Packaging Desk", a buying team), never a part of it.
    plan = S.quoted_unread(plan, question, model, services.index,
                           [f for s in _standing_now(services, session) for f in s.filters]
                           + (list(seen.plan.filters) if seen is not None and seen.plan is not None else []))
    plan = _whole_names(plan, matches)
    plan = _by_day(plan, question)
    plan = _one_month_named(plan, question)
    plan, unquoted, no_such = _quoted_filters(plan, question, model, _grounded(matches, session), services.index)
    plan = _quoted_lost(plan, question, model, services.index)
    plan = _quoted_as_written(plan, question, model, services.index)
    unknown_left_out: list[tuple[str, str]] = []
    plan, missing = _members_named(plan, model, services.index, unknown_left_out)
    bare = plan
    from core2.plan.planner import check

    if kept_date:
        _keep_date(session, model, kept_date, asked=asked_date)
    plan, applied, aside = S.apply(plan, _standing_now(services, session), question, model,
                                   lambda p: not check(p, model, services.today)[0],
                                   seen.plan if seen is not None else None)

    ctx = Context(today=services.today, allowed_tables=services.allowed_tables, max_rows=MAX_ROWS,
                  personal_shown=services.scrub is not None, link_repeats=_link_check(services))
    shown: list[tuple] = []
    periods: list[str] = []
    try:
        if plan.figures:
            # Figures no metric holds, worked out for this question (core2/resolve/figures.py): the whole answer
            # counts and names them like any other.
            plan, model, _ = worked_out(plan, model, ctx)
            services = replace(services, model=model)
        payload = _compute(question, plan, services, ctx, question_id=question_id, started=start, shown=shown,
                           periods=periods, notes=True)
    except ResolveError as exc:
        if exc.kind != "ambiguous":
            template = _REFUSALS.get(exc.kind, "{message}.")
            stopped = exc.message + (f" ({', '.join(exc.options)})" if exc.kind == "unconfirmed" and exc.options
                                     else "")
            instead = _nearest(services, plan)
            return _refused(question, template.format(message=exc.message), plan, services, stopped, session,
                            outcome.masked, unsupported=exc.kind == "unsupported",
                            **({"follow_up_suggestions": [{"label": q, "question": q} for q in instead]} if instead
                               else {}))
        # Two fair readings (two links, two dates): the one the question's own words name ("posts owned by an elder
        # user": the owner, not the last editor) is answered. Else each one is worked out first: the same figure
        # either way is answered and said so; different figures are asked about, each one's figure beside its name.
        said = _reading_said(question, exc.options) if exc.field and exc.field != "time.date" else None
        before = list(shown), list(periods)       # what a reading tried and dropped leaves behind is put back
        if said is not None:
            try:
                chosen = _completed(plan, exc.field or "", said, model)
                payload = _compute(question, chosen, services, ctx, question_id=question_id, started=start,
                                   shown=shown, periods=periods, notes=True)
                plan = chosen
            except Exception as again:  # noqa: BLE001 - asked about, as before
                log.info("core2: could not answer %r by the reading its words name: %s", question, again)
                said = None
                shown[:], periods[:] = before
        figures = _each_reading(question, plan, exc, services, ctx, question_id=question_id) \
            if exc.field and said is None else {}
        same = _same_either_way(figures, plan)
        answered = payload if said is not None else None
        if same is not None:
            either = _completed(plan, exc.field or "", same, model)
            try:
                answered = _compute(question, either, services, ctx, question_id=question_id, started=start,
                                    shown=shown, periods=periods, notes=True)
            except Exception as again:  # noqa: BLE001 - asked about, as before
                log.info("core2: could not answer %r by its one reading: %s", question, again)
                shown[:], periods[:] = before
        if answered is None:
            _keep(session, Turn(question, plan, outcome.masked))     # the reply names the role this asks for
            if exc.field:
                session.pending = Pending(plan, exc.field, list(exc.options), outcome.masked)
            asked = _clarification(question, Clarify(about="path", question=f"{exc.message}. Which one do you mean",
                                                     options=exc.options))
            if figures:
                _with_numbers(asked, figures)
            return asked
        if said is None:
            plan, payload = either, answered
            _said_same_either_way(payload, exc.message, figures)
    except QueryFailed as exc:     # a refused or failed query is told, never raised to the socket
        log.warning("core2 query failed for %r: %s", question, exc.cause)
        cause = exc.cause
        if unavailable(cause):
            return _refused(question, UNAVAILABLE, plan, services, "The database was not available.", session,
                            outcome.masked, trust={"sql": exc.sql})
        reason = str(cause).split(":", 1)[-1].strip() if "Policy" in type(cause).__name__ else "the database refused it"
        return _refused(question, f"The question was understood but the query could not run ({reason}).", plan,
                        services, f"The query could not run: {reason}", session, outcome.masked,
                        trust={"sql": exc.sql})
    except (CompileError, ValueError) as exc:
        log.warning("core2 could not compile a plan for %r: %s", question, exc)
        return _refused(question, f"I cannot answer that from this data yet: {exc}.", plan, services, str(exc),
                        session, outcome.masked, unsupported=True)
    if plan.intent == "drivers":
        _one_step_down(question, plan, payload, services, ctx, question_id=question_id)
    if plan.figures:
        payload["figures"] = _figures_shown(plan, model)
    for thing, value in unknown_left_out[:3]:
        payload.setdefault("coverage_caveats", []).append(
            f'No {thing} is called "{value}" in the data, so leaving it out changed nothing. Name it as the data '
            f'writes it, or in quotes as the reader sees it in an answer.')
    if missing is not None and _found_nothing(payload):
        # Nothing matched, and a member the plan named is not one the data had when its members were read:
        # say so, not "No rows match", which reads as if that member bought nothing. (A member added since
        # is found by the query, so it is never refused for being new.)
        thing, value = missing
        meant: list[tuple[str, str, str, Plan]] = []
        if _points(str(value)):
            # The planner filtered on the words that point at a row ("lowest", "that division"): they name no member.
            text = (f'I could not tell which {thing} "{value}" means. Name it, or point at one in the answer above: '
                    '"the first one", "the lowest one".')
        else:
            text = _REFUSALS["unknown"].format(message=f'there is no {thing} called "{value}"')
            meant = _did_you_mean(question, plan, str(value), services, ctx, question_id=question_id)
            if meant:
                # "Downtwn Store": the members written most like it, each asked first through the reader's own
                # access, so only one they can see is offered -- with its figure.
                text += " Did you mean " + " or ".join(f'"{name}"' for name, _, _, _ in meant) + "?"
                if len(meant) == 1:
                    text += f" {meant[0][2]}"
            elif plan.follow_up == "refine":
                text += (f' If you mean one of the {names.plural(thing)} in the answer above, say which ("the first '
                         'one", "the lowest one"), or ask for the same answer in another period ("the same ranking '
                         'against last year").')
        refused = _refused(question, text, plan, services, f'No {thing} is called "{value}", and nothing matched.',
                           session, outcome.masked, trust={"sql": (payload.get("trust") or {}).get("sql", "")})
        if meant:
            refused["follow_up_suggestions"] = [{"label": asked, "question": asked, "kind": "look"}
                                                for _, asked, _, _ in meant]
            for _, asked, _, instead in meant:
                session.offers[_plain(asked)] = instead
            session.yes = meant[0][3]
        return refused
    if payload.get("beyond_data") and _found_nothing(payload, zero=True):
        # "What will our total sales be in 2004?" said "no value": the data ends before the period asked for. The
        # answer stays an empty one (a dashboard tile draws it over all the data instead), said as it is.
        last = dt.date.fromisoformat(payload["beyond_data"])
        # The figure and the conditions it was asked under stay said ("Net amount in 2030, excluding segment Mail
        # order"); only its "no value" becomes why.
        old = (payload.get("answer") or {}).get("headline") or ""
        asked = old[:-len(": no value.")] if old.endswith(": no value.") else ""
        said = (f"{asked}: nothing yet. The data ends on {last.day} {last:%b %Y}." if asked else
                f"The data ends on {last.day} {last:%b %Y}: it has nothing yet for the period you asked about.")
        payload.setdefault("answer", {})["headline"] = said
        payload["answer"]["short_value"] = ""
        payload["text"] = (said + " Ask about a period it covers, or for a forecast (\"forecast it for the next 12 "
                           "months\").")
    if plan.group_by and (plan.limit and _found_nothing(payload) or shown and all(s[1] is None for s in shown)):
        _by_key(payload, question, plan, services, ctx, question_id=question_id, shown=shown)
    if plan.time.date and _found_nothing(payload, zero=True):
        plan = _by_main_date(payload, question, plan, services, ctx, question_id=question_id, shown=shown,
                             periods=periods)
    if not plan.include_left_out and _found_nothing(payload, zero=True):
        # A count of nothing too: "How many invoice lines were cancelled in 2025?" said 0, every cancelled line
        # being left out by default, under a filter that asked for exactly those.
        plan = _say_left_out(payload, question, plan, services, ctx, question_id=question_id, shown=shown)
    # Looked at before it is shown, as an analyst would (core2/agent/look.py): nothing found, and why; a figure
    # resting on a few days; a period starting before the data.
    empty = _found_nothing(payload, zero=True)
    # The conditions this question did not write: kept from the answer above, or standing ones.
    carried = kept_parts(plan, seen.plan if seen is not None else None, question)[0] \
        if following and plan.follow_up == "refine" else []
    standing_filters = [f for s in applied for f in s.filters]
    looked = _look(plan, payload, services, ctx, empty=empty, carried=carried, standing=standing_filters)
    payload["plan"] = plan.model_dump(mode="json", exclude_defaults=True)
    chips = [*offered(plan, model), *_instead(plan, model, outcome), *asks.other_readings(plan, model, question),
             *_named_unquoted(payload, plan, model, unquoted_named)]
    read_as = asks.badges(plan, model, outcome.picked)
    if read_as and isinstance(payload.get("answer"), dict):
        # "Read as Net amount": the figure a business ask was read as, when the AI asked which (the others are
        # chips); and how a run rate is worked out.
        payload["answer"]["badges"] = [*(payload["answer"].get("badges") or []), *read_as]
    if chips:
        # Either wording runs the chip's plan: a French reader's click sends the French words.
        session.offers = {_plain(w): changed for words, changed in chips for w in (words, chip_french(words))}
        rest = payload.get("follow_up_suggestions", [])
        if any(changed.include_left_out for _, changed in chips):
            rest = [s for s in rest if s.get("question") != LEFT_OUT_CHIP["question"]]     # the chip says it better
        payload["follow_up_suggestions"] = [*({"label": words, "question": words, "label_fr": chip_french(words),
                                               "question_fr": chip_french(words), "kind": "setting"}
                                              for words, _ in chips), *rest]
    for other in (payload.get("dates") or {}).get("others") or []:
        # Another date of the same rows, a click on the date badge: the same plan counted by it, with no AI, and
        # the date kept for the conversation.
        counted = plan.model_copy(update={"time": plan.time.model_copy(update={"date": other["slug"]}),
                                          "follow_up": "refine"})
        for words in (other["words"], other["words_fr"]):
            session.offers[_plain(words)] = counted
            session.dates[_plain(words)] = other["key"]
    if looked:
        _offer(payload, session, looked, empty=empty)
    dig = payload.pop("_dig", None)
    if dig is not None:
        # The "why" one step down: its next step is a button, and what "yes" runs (no AI).
        steps = [{**o, "kind": "look"} for o in payload["reply"]["offers"]]
        for o in steps:
            session.offers[_plain(o["question"])] = dig
            session.offers[_plain(o["question_fr"])] = dig
        session.yes = dig
        payload["follow_up_suggestions"] = [*steps, *(payload.get("follow_up_suggestions") or [])][:6]
    if unquoted or no_such:
        _say_unquoted(payload, question, unquoted,
                      _visible(services.index.match(question), model, services.allowed_tables), no_such=no_such)
    if outcome.repaired:
        payload["trust"]["plan_repaired"] = True
    if following and plan.follow_up == "refine" and seen is not None:
        # Said on the answer: what it changed, what it kept from the answer above without the question saying so
        # ("North", "Q1 2026"), and the same question asked on its own, one click away.
        payload["following"] = {"question": seen.question, "ask_new": f"New question: {question}"}
        kept = _kept_said(plan, last_plan(session.turns), question, model)
        if kept:
            payload["following"]["kept"] = kept
    _standing_said(payload, services, session, applied, aside)
    # The conversation keeps the plan as asked: a standing condition is put on again at each answer, so one
    # removed is gone from the follow-ups too.
    _keep(session, Turn(question, S.without(plan, applied, bare), outcome.masked, shown, periods,
                        _shown_title(payload)))
    from core2.agent import parts as P

    gaps = []
    if from_ai:
        # Every part of the request, checked against the plan and the answer: one left out is asked for once more
        # (below), and what the answer still lacks is said on it, never dropped without a word.
        gaps = P.missing(question, plan, payload, model, named=named,
                         before=seen.plan if following and plan.follow_up == "refine" and seen is not None else None)
        if gaps:
            payload["parts_left_out"] = P.said(gaps, question, private=not services.values_allowed)
    if from_ai and not feedback:
        from core2.agent import again as A

        reasons = A.suspect(question, plan, payload, looked, kept=[*carried, *standing_filters],
                            empty=empty, model=model, values=services.values_allowed)
        if gaps:
            en, fr = P.reason(gaps, hide=None if services.values_allowed else _hiding(outcome.masked))
            reasons.append(A.Reason("left_out", en, fr))
        if reasons:
            payload[A.KEY] = (question, forced, plan, reasons)
    return payload


# ── standing conditions (core2/standing.py) ────────────────────────────────


def _standing_now(services: Services, session: Session) -> list[Standing]:
    return _in_force(services.always, session)


def _in_force(always: list[Standing], session: Session) -> list[Standing]:
    """The standing conditions in force: the reader's own from their profile (unless taken off here), then this
    conversation's; one condition said twice counts once."""
    out: list[Standing] = []
    for s in [*(a for a in always if a.id not in session.paused), *session.standing]:
        if not any(o.words == s.words for o in out):
            out.append(s)
    return out


def _standing_shown(services: Services, session: Session) -> list[dict[str, Any]]:
    return [s.shown() for s in _standing_now(services, session)]


def _standing_said(payload: dict[str, Any], services: Services, session: Session, applied: list[Standing],
                   aside: list[str]) -> None:
    """On the answer: the standing conditions it was given under, and any set aside for it, and why."""
    kept = _standing_shown(services, session)
    if not kept:
        return
    payload["standing"] = {"kept": kept, "applied": [s.words for s in applied],
                           "applied_fr": [s.words_fr or s.words for s in applied],
                           "aside": [en for en, _ in aside], "aside_fr": [fr for _, fr in aside]}


def _standing_set(question: str, scope: str, words: str, services: Services, session: Session) -> dict[str, Any]:
    """A standing instruction read: the conditions kept, said with what each means, or why none could be."""
    said = S.read(scope, words, services.model, services.index, services.today)  # type: ignore[arg-type]
    fr = S.french(question)
    unknown = ", ".join(f"« {u} »" if fr else f"“{u}”" for u in said.unknown[:3])
    if not said.standing:
        text = ("Je n’ai pas compris quoi garder désormais. Nommez une période (« les 2 derniers mois ») ou les noms à "
                "garder ou à exclure, entre guillemets (seulement « Downtown », sans « Cancelled »)." if fr else
                "I could not tell what to keep from now on. Name a period (\u201cthe last 2 months\u201d) or the "
                "names to keep or leave out, in quotes (only \u201cDowntown\u201d, leave out \u201cCancelled\u201d).")
        if said.unknown:
            text = (f"Rien dans les données ne s’appelle {unknown} : rien n’a été gardé. Nommez-le comme les données "
                    "l’écrivent." if fr else
                    f"Nothing in the data is called {unknown}, so nothing was kept. Name it as the data writes it.")
        frame = _frame(question, text, kind="standing")
        _standing_said(frame, services, session, [], [])
        return frame
    kept_here = scope == "conversation" or services.keep_always is None
    for s in said.standing:
        if kept_here:
            session.standing = [k for k in session.standing if k.id != s.id and not _same_kind(k, s)]
            session.standing.append(s)
        else:
            services.keep_always(s)  # type: ignore[misc]
            services.always = [a for a in services.always if a.id != s.id and not _same_kind(a, s)] + [s]
            session.paused.discard(s.id)
    listed = "; ".join(_meant(s, services, fr) for s in said.standing)
    if scope == "always" and not kept_here:
        text = (f"Enregistré dans votre profil : {listed}. Chacune de vos conversations l’utilise jusqu’à ce que vous "
                "le retiriez, ici ou sur votre page Profil." if fr else
                f"Saved to your profile: {listed}. Every conversation of yours uses it until you remove it, here "
                "or on your Profile page.")
    else:
        text = (f"Désormais dans cette conversation : {listed}. Chaque réponse l’utilise jusqu’à ce que vous le "
                "retiriez." if fr else
                f"From now on in this conversation: {listed}. Every answer uses it until you remove it.")
    if said.unknown:
        text += (f" Rien dans les données ne s’appelle {unknown} : laissé de côté." if fr else
                 f" Nothing in the data is called {unknown}: left out.")
    frame = _frame(question, text, kind="standing")
    _standing_said(frame, services, session, [], [])
    return frame


def _same_kind(a: Standing, b: Standing) -> bool:
    """A new period replaces the period kept; a new condition on a field replaces the one on that field."""
    if a.window is not None and b.window is not None:
        return True
    return bool(a.filters and b.filters and {f.field for f in a.filters} == {f.field for f in b.filters})


def _meant(s: Standing, services: Services, fr: bool = False) -> str:
    """A condition with what it means now: "the last 2 months (1 Apr 2026 – 31 May 2026)", counted back from where
    the data ends when it ended a while ago, as the answers count it (core2/resolve/time.py)."""
    words = (s.words_fr or s.words) if fr else s.words
    if s.window is None:
        return words
    from core2.answer.builder import _jour
    from core2.resolve.time import resolve_window

    from core2.plan.analyses import activities
    from core2.resolve.resolver import _first_last

    # The main activity's date (orders, fills), as the answers count most questions: never a table of targets
    # that runs on into next year. With no members followed (a ledger), the busiest table of what has happened.
    main = next((a.date for a in activities(services.model, most=1) if a.date is not None), None) \
        or _happened(services.model, services.today)
    try:
        rng = resolve_window(s.window, today=services.today, last_data=_first_last(main)[1] if main else None)
    except Exception:  # noqa: BLE001 - the words alone
        return words
    if rng.start is None or rng.end is None:
        return words
    last = rng.end - dt.timedelta(days=1)
    if fr:
        return f"{words} ({_jour(rng.start)} \u2013 {_jour(last)})"
    return f"{words} ({rng.start.day} {rng.start:%b %Y} \u2013 {last.day} {last:%b %Y})"


def _happened(model: SemanticModel, today: dt.date) -> DateRole | None:
    """The date of the busiest table whose figures are dated up to today, never one running on past it (budgets,
    plans, targets): what "the last 2 months" of a ledger counts back from."""
    from core2.resolve.resolver import _first_last

    dated = {m.default_date for m in model.measures.values() if not m.hidden and m.default_date}
    roles = [r for r in model.date_roles.values() if r.key in dated and r.table in model.tables
             and (last := _first_last(r)[1]) is not None and last <= today]
    return max(roles, key=lambda r: (model.tables[r.table].row_count or 0, r.key), default=None)


def _standing_dropped(question: str, gone: list[Standing], services: Services, session: Session) -> dict[str, Any]:
    """Standing conditions removed: this conversation's go; the reader's own are taken off for this conversation,
    and out of their profile when they say so for good."""
    for s in gone:
        if s.scope == "always":
            if re.search(r"\b(?:for\s+good|everywhere|permanently|from\s+my\s+profile|always)\b", question, re.I) \
                    and services.drop_always is not None:
                services.drop_always(s.id)
                services.always = [a for a in services.always if a.id != s.id]
            else:
                session.paused.add(s.id)
        else:
            session.standing = [k for k in session.standing if k.id != s.id]
    fr = S.french(question)
    names_ = "; ".join((s.words_fr or s.words) if fr else s.words for s in gone)
    still = "; ".join((s.words_fr or s.words) if fr else s.words for s in _standing_now(services, session))
    if not gone:
        # "Remove the Germany condition" where none is kept by that name: said, never answered as a question.
        if fr:
            text = ("Aucune condition de ce nom n’est appliquée dans cette conversation. " +
                    (f"Appliqué : {still}" if still else "Aucune condition n’est appliquée"))
        else:
            text = ("No condition by that name is in force in this conversation. " +
                    (f"In force: {still}" if still else "None is in force"))
        frame = _frame(question, text + ".", kind="standing")
        frame["standing"] = {"kept": _standing_shown(services, session), "applied": [], "aside": []}
        return frame
    if fr:
        text = f"Retiré : {names_}. " + (f"Toujours appliqué : {still}" if still else
                                          "Les prochaines réponses utilisent de nouveau toutes les données")
    else:
        text = f"Removed: {names_}. " + (f"Still in force: {still}" if still else
                                         "The next answers use all of the data again")
    if any(s.scope == "always" and s.id in session.paused for s in gone):
        text += (". Il reste dans votre profil pour les autres conversations : retirez-le là-bas pour de bon" if fr
                 else ". It stays in your profile for other conversations: remove it there for good")
    frame = _frame(question, text + ".", kind="standing")
    frame["standing"] = {"kept": _standing_shown(services, session), "applied": [], "aside": []}
    return frame


def _marked(payload: dict[str, Any], plan: Plan, model: SemanticModel) -> dict[str, Any]:
    """An answer grouped by, filtered on or counted for a person or their records ("sales for the first one", a
    prescription's number) is marked as naming people: neither it nor its headline reaches the AI's summary."""
    if any(about_people(model, slug) for slug in slugs_in(plan.model_dump(mode="json", exclude_defaults=True))):
        payload["personal"] = True
    return payload


def _figures_shown(plan: Plan, model: SemanticModel) -> list[dict[str, Any]]:
    """The figures an answer worked out for its question, as the reader sees them: each name and how it was
    counted, in the formula language an admin would save it in (for "Save as a metric")."""
    from core2.model import authoring
    from core2.resolve.figures import PREFIX
    from core2.resolve.resolver import _readable

    shown = []
    for figure in plan.figures:
        m = next((m for m in model.measures.values()
                  if m.key.startswith(PREFIX) and m.business_name == _readable(figure.name)), None)
        if m is not None:
            shown.append({"name": m.business_name, "formula": authoring.to_text(model, m.expr), "format": m.format,
                          "tables": authoring.counted_on(model, m), "measure": m})
    return shown


def _shown_title(payload: dict[str, Any]) -> str:
    """What an answer that drew rows shows, as its dashboard tile is called ("Net amount by region"); "" when it
    drew none (a refusal, a question back), so "this" never means it."""
    if payload.get("data") is None or not payload.get("plan"):
        return ""
    return str((payload.get("chart") or {}).get("title") or (payload.get("kpi") or {}).get("title")
               or payload.get("title") or (payload.get("answer") or {}).get("headline") or "")[:120]


def _compute(question: str, plan: Plan, services: Services, ctx: Context, *, question_id: str,
             started: float, shown: list[tuple] | None = None, periods: list[str] | None = None,
             notes: bool = False) -> dict[str, Any]:
    """The answer to a checked plan: one query, or the several a "why" or a forecast needs.

    ``shown``, when given, is filled with the answer's members in the order shown, for a
    follow-up that names one by its place ("the first one"); ``periods`` with its periods, for a
    "why" that follows the one month on screen.
    """
    model = services.model
    warehouse = Guarded(services.warehouse)
    common: dict[str, Any] = {"data_source": services.data_source, "question_id": question_id,
                              "model_version": model.version}
    if plan.intent in ("drivers", "forecast") and plan.shaped():
        raise ResolveError("unsupported", f"{'why' if plan.intent == 'drivers' else 'a forecast of'} a figure worked "
                           "out per member, in ranges or over a running series is not worked out yet: ask for the "
                           "figure itself")
    if plan.intent == "drivers":
        return _marked(answer_drivers(question, plan, model=model, warehouse=warehouse, ctx=ctx, started=started,
                                      **common), plan, model)
    if plan.intent == "forecast":
        return _marked(answer_forecast(question, plan, model=model, warehouse=warehouse, ctx=ctx, started=started,
                                       **common), plan, model)
    logical = resolve(plan, model, replace(ctx, split_units=services.split_units))
    if logical.people:
        _age_limits(plan, logical, services, ctx, warehouse)
    if any(part.snapshot for part in logical.parts):
        logical = complete_snapshots(logical, model, warehouse)     # a snapshot still loading is not the latest
    # Ranges the question named no edges for: round numbers near the values that split the data evenly.
    logical = fill_edges(logical, model, warehouse.dialect, lambda sql: warehouse.query(sql, max_rows=1).rows)
    compiled = compile_query(logical, model, warehouse.dialect)
    result = warehouse.query(compiled.sql, max_rows=compiled.row_cap)
    rows = _members_listed(compiled, logical, result.columns, _with_a_figure(compiled, result.columns, result.rows))
    payload = build_answer(question, logical, compiled, result.columns, rows,
                           duration_ms=(time.perf_counter() - started) * 1000, truncated=result.truncated,
                           chart_type=plan.chart, **common)
    dates = date_choices(plan, logical, model)
    if dates:
        payload["dates"] = dates       # the badge's other dates of the same rows, and the metric's own put right
    if logical.beyond:
        payload["beyond_data"] = logical.beyond.isoformat()     # the period asked for starts after the data ends
    if logical.share:
        _share_of_the_whole(plan, logical, payload, model, ctx, warehouse)
    columns = [*(g.column for part in logical.parts for g in part.groups),
               *(p.column for part in logical.parts for p in part.preds),
               *(p.column for part in logical.parts for semi in part.semis for p in semi.preds)]
    if logical.people or any(people_column(model.columns.get(c)) for c in columns if c):
        payload["personal"] = True      # its rows name people, or count them by age: no written summary by the AI
    _marked(payload, plan, model)
    if shown is not None:
        shown.extend(_members_shown(logical, payload))
    if periods is not None:
        periods.extend(_periods_shown(logical, payload))
    payload["follow_up_suggestions"] = follow_ups(plan, logical, payload, model, services.allowed_tables)
    chart = payload.get("chart") or {}
    # A bar that is a range of amounts ("5–10") is no member a click could open.
    drill = drills(plan, logical, payload, model, services.allowed_tables) \
        if chart and not (chart.get("other") or {}).get("ranges") and not logical.people else None
    if drill:
        payload["chart"]["drill"] = drill
    # And as fields, for a dashboard tile drilled into in place (portal/routes.py).
    broken = breakdowns(plan, logical, model, services.allowed_tables) if chart and not logical.people else []
    if broken:
        payload["chart"]["breakdowns"] = broken
    context = _against_before(question, plan, services, ctx, warehouse, logical, payload)
    if context:
        payload["key_insights"] = [context, *payload.get("key_insights", [])]
    if notes:
        _more_for_the_note(plan, logical, payload, services, ctx, warehouse)
    return payload


def _with_a_figure(compiled: Any, names: list[str], rows: list[tuple]) -> list[tuple]:
    """The members of a breakdown that have some of the figure asked: revenue in CAD by cost centre listed the nine
    cost centres booking only expenses as "no value", and said "across 12 cost centres" for three. Periods keep
    theirs (a month with nothing is a month of the series), and an answer with no member having any stays as it
    is, to be said so."""
    roles = {c.role for c in compiled.columns}
    # A store with nothing this year but some last year fell to nothing: a figure of the answer too.
    measures = {c.name for c in compiled.columns if c.role in ("measure", "prior") and not c.computed}
    at = [i for i, n in enumerate(names) if n in measures]
    if not at or roles & {"period", "time"} or not roles & {"attribute", "member_code"}:
        return rows
    kept = [r for r in rows if any(r[i] is not None for i in at)]
    return kept or rows


def _members_listed(compiled: Any, logical: Any, names: list[str], rows: list[tuple]) -> list[tuple]:
    """A list of members ("all the customers who bought Dish Soap in the last 90 days") lists members: the rows that
    name none (Dish Soap sold with no customer on file) were counted as one more customer. They are left out, and
    the answer says so."""
    first = next((c for c in compiled.columns if c.role == "attribute"), None)
    if logical.intent != "list" or first is None or not first.missing or first.name not in names:
        return rows
    at = names.index(first.name)
    kept = [r for r in rows if r[at] is not None]
    if kept and len(kept) < len(rows):
        thing = first.missing.removeprefix("No ").removesuffix(" recorded")
        logical.notes.append(f"Rows with {first.missing[:1].lower() + first.missing[1:]} are left out: they name no "
                             f"{thing}.")
        return kept
    return rows


def _more_for_the_note(plan: Plan, logical: Any, payload: dict[str, Any], services: Services, ctx: Context,
                       warehouse: Guarded) -> None:
    """The analyst's next checks (core2/answer/extra.py): what made a one-off, the same period a year before, the
    target, what moved the figure -- each one more small query, asked in the chat only (a dashboard tile says its
    own line), and never at the cost of the answer."""
    model = services.model
    rules = replace(ctx, split_units=services.split_units)

    def run(asked: Plan) -> tuple[Any, Any, Any]:
        lg = resolve(asked, model, rules)
        if any(part.snapshot for part in lg.parts):
            lg = complete_snapshots(lg, model, warehouse)
        compiled = compile_query(lg, model, warehouse.dialect)
        return lg, compiled, warehouse.query(compiled.sql, max_rows=min(compiled.row_cap or 500, 500))

    from core2.answer.compared import last_year_in_full

    try:
        last_year_in_full(plan, logical, payload, run)       # a year so far: last year in full beside it
    except Exception as exc:  # noqa: BLE001 - the answer stands without last year in full
        log.warning("core2 comparison: last year in full not added: %s", exc)
    try:
        more = extra_facts(plan, logical, payload, model, run, allowed=services.allowed_tables)
    except Exception as exc:  # noqa: BLE001 - the answer stands with the facts of its own rows
        log.info("core2 note: the next checks failed: %s", exc)
        return
    if more:
        reading = payload.setdefault("reading", {"shape": "value", "lead": {}, "one_offs": [], "facts": []})
        reading["facts"] = [*(reading.get("facts") or []), *(f.as_dict() for f in more)]


def _share_of_the_whole(plan: Plan, logical: Any, payload: dict[str, Any], model: SemanticModel, ctx: Context,
                        warehouse: Guarded) -> None:
    """"What share of 2025 sales came in on weekends?": the part's share of everything in the period, not of
    itself. A share is worked out over the answer's own rows, so a filter that keeps only the part made it
    100%. Where the answer is one figure kept to one member filter (or to members of one field, the part
    grouped by it), the whole is asked for once more without that filter, and the share is the part over it."""
    from core2.answer.builder import fmt

    named = [f for f in plan.filters if f.op not in ("is_null", "not_null")]   # weekends, a store: the period is not one
    rows = (payload.get("data") or {}).get("rows") or []
    if logical.unit_group or len({f.field for f in named}) != 1 or len(rows) != 1 or len(plan.measures) != 1:
        return
    field = named[0].field
    if any(g != field for g in plan.group_by):
        return
    whole_plan = plan.model_copy(update={"filters": [f for f in plan.filters if f.field != field], "group_by": [],
                                         "intent": "value", "sort": [], "limit": None})
    try:
        whole_logical = resolve(whole_plan, model, ctx)
        if any(part.snapshot for part in whole_logical.parts):
            whole_logical = complete_snapshots(whole_logical, model, warehouse)
        compiled = compile_query(whole_logical, model, warehouse.dialect)
        result = warehouse.query(compiled.sql, max_rows=2)
    except Exception as exc:  # noqa: BLE001 - the answer stands with the share of its own rows
        log.info("core2 could not ask for the whole a share is of: %s", exc)
        return
    measure = next((o for o in logical.measures if not o.hidden), None)
    if measure is None or len(result.rows) != 1:
        return
    names = [c.casefold() for c in result.columns]
    whole = result.rows[0][names.index(measure.name.casefold())] if measure.name.casefold() in names else None
    part = rows[0].get(measure.name)
    if not isinstance(whole, (int, float, Decimal)) or not isinstance(part, (int, float, Decimal)) or not whole:
        return
    share = float(part) / float(whole) * 100
    rows[0][f"{measure.name}_share"] = round(share, 4)
    answer = payload.setdefault("answer", {})
    said = str(answer.get("headline") or "").rstrip(".")
    label = measure.label[:1].lower() + measure.label[1:]
    condition = (model.attributes[field].business_name.lower() if field in model.attributes else
                 field.split(":")[-1].split(".")[-1].removeprefix("is_").replace("_", " "))
    answer["headline"] = f"{said}, {share:.1f}% of all {label} in the period ({fmt(whole, measure.format)})."
    payload.setdefault("coverage_caveats", []).append(
        f"The share is of {fmt(whole, measure.format)}: the same {label} without the condition on {condition}.")


def _age_limits(plan: Plan, logical: Any, services: Services, ctx: Context, warehouse: Guarded) -> None:
    """An age limit stands only where the 5 years just inside it hold at least 11 people (with the question's
    other conditions): two answers either side of it never isolate fewer. Each use of an age is logged."""
    from core2.resolve import age

    age.used(logical, str(getattr(services.warehouse, "account_id", "") or ""))
    for low, high in logical.age_checks:
        checked = resolve(age.edge_check_plan(plan, logical.age_slug, low, high), services.model, ctx)
        compiled = compile_query(checked, services.model, warehouse.dialect)
        result = warehouse.query(compiled.sql, max_rows=2)
        names = [n.casefold() for n in result.columns]
        people = int(result.rows[0][names.index(checked.people.casefold())] or 0) if result.rows else 0
        if people < age.MIN_PEOPLE:
            raise ResolveError("unsupported", f"fewer than {age.MIN_PEOPLE} people are aged {low} to {high} here, so "
                               "an age limit there could point at someone: ask for a wider range of ages")


# Links no one measured, as the data answered for them: asked the first time a question follows one, once per
# workspace, model version, link (its key columns and conditions) and reader -- what a reader's rules leave
# them is what their totals count. At most LINK_CHECKS answers are kept.
LINK_CHECKS = 4096
_LINK_REPEATS: OrderedDict[tuple, bool] = OrderedDict()
_LINK_LOCK = threading.Lock()


def _link_check(services: Services) -> Callable[[Join], bool] | None:
    """Does a link hold more than one target row for some key, as the data says? (resolver Context.link_repeats)"""
    warehouse, model = services.warehouse, services.model
    if not hasattr(warehouse, "query"):
        return None
    account = str(getattr(warehouse, "account_id", "") or "")
    reader = str((getattr(warehouse, "user", None) or {}).get("id") or "")
    unknown: set[str] = set()       # links the data would not check now: not followed, and said so

    def repeats(j: Join) -> bool:
        key = (account, model.version, j.key, tuple(j.to_columns),
               tuple(json.dumps(c.model_dump(mode="json"), sort_keys=True) for c in j.conditions), reader)
        with _LINK_LOCK:
            if key in _LINK_REPEATS:
                _LINK_REPEATS.move_to_end(key)
                return _LINK_REPEATS[key]
        try:
            found = bool(warehouse.query(repeats_sql(model, j, warehouse.dialect), max_rows=1).rows)
        except Exception as exc:  # noqa: BLE001 - not followed this time (it might count rows twice); asked again
            log.warning("core2: could not check whether the link %s of %s matches one row per key; not followed "
                        "until it can be: %s", j.key, account, exc)
            unknown.add(j.key)
            return True
        if found:
            log.warning("core2: the link %s of %s matches more than one row for some keys: questions do not follow "
                        "it", j.key, account)
        with _LINK_LOCK:
            _LINK_REPEATS[key] = found
            while len(_LINK_REPEATS) > LINK_CHECKS:
                _LINK_REPEATS.popitem(last=False)
        return found

    repeats.unknown = unknown         # type: ignore[attr-defined] - read by the resolver's reason (_counted_twice)
    return repeats


def _members_shown(logical: Any, payload: dict[str, Any], most: int = 500) -> list[tuple]:
    """The members of an answer's first grouping, in the order shown: (attribute slug, stored value, its number,
    the row's members of its other groupings by attribute slug, for "that division" of a second grouping, and its
    change on an answer that compares two periods, for "the one that grew the most")."""
    groups = [g for g in logical.groups if g.kind == "attribute" and g.attribute and g.name != logical.unit_group]
    if not groups or any(g.kind == "period" for g in logical.groups):
        return []
    group = groups[0]
    measure = next((o.name for o in logical.measures if not o.hidden), None)
    change = f"{measure}_change" if measure else None

    def member(value: Any) -> str | None:
        return None if no_member(value) else str(value)

    out: list[tuple] = []
    for row in (payload.get("export_rows") or (payload.get("data") or {}).get("rows") or [])[:most]:
        value, number = row.get(group.name), row.get(measure) if measure else None
        moved = row.get(change) if change else None
        out.append((group.attribute, member(value),
                    number if isinstance(number, (int, float)) and not isinstance(number, bool) else None,
                    {g.attribute: member(row.get(g.name)) for g in groups[1:]},
                    moved if isinstance(moved, (int, float)) and not isinstance(moved, bool) else None))
    return out


def _periods_shown(logical: Any, payload: dict[str, Any], most: int = 60) -> list[str]:
    """The periods of an answer by period, in the order shown, each by its first day ("2026-05-01")."""
    period = next((g for g in logical.groups if g.kind == "period"), None)
    if period is None:
        return []
    rows = payload.get("export_rows") or (payload.get("data") or {}).get("rows") or []
    return [str(r[period.name])[:10] for r in rows[:most] if r.get(period.name) not in (None, "")]


UNAVAILABLE = ("The database is not available right now: Azure SQL says so while a paused database is resuming, "
               "or during a failover. Ask again in a minute.")


def unavailable(exc: BaseException | None) -> bool:
    """Is ``exc`` the warehouse being briefly unavailable (Azure SQL resuming from a pause, a failover)?"""
    from core.schema import _azure_transient_number

    for _ in range(5):
        if exc is None:
            return False
        if _azure_transient_number(exc):
            return True
        exc = getattr(exc, "cause", None) or exc.__cause__
    return False


def _against_before(question: str, plan: Plan, services: Services, ctx: Context, warehouse: Guarded,
                    logical: Any, payload: dict[str, Any]) -> str:
    """One number for a period: how it compares with the period before, from one more small query.

    "Gross profit in Q2 2026: $4.20M" says nothing of whether that is good; the
    period before is the first thing a reader checks. Only for a single value of a
    bounded period that is not already a comparison; a failure costs the finding,
    never the answer.
    """
    if plan.group_by or plan.time.grain or plan.time.compare or logical.compare is not None or plan.shaped() \
            or plan.intent not in (None, "value", "count") or len(logical.measures) != 1:
        return ""         # a figure per member, then over them, is not yet compared with the period before
    if logical.window.start is None or logical.window.end is None or any(
            c.kind == "date" for c in logical.conditions):
        return ""                     # "invoiced in March for orders placed in February" has no period before
    rows = (payload.get("data") or {}).get("rows") or []
    if len(rows) != 1 or _found_nothing(payload):
        return ""
    kpi = payload.get("kpi") if isinstance(payload.get("kpi"), dict) else None
    if kpi is not None:
        _trend(plan, services, ctx, warehouse, logical, kpi)
    before = plan.model_copy(update={"intent": "compare", "time": plan.time.model_copy(
        update={"compare": Compare(kind="previous_period")})})
    try:
        compared = resolve(before, services.model, replace(ctx, split_units=services.split_units))
        role = compared.parts[0].date.role if compared.parts and compared.parts[0].date else None
        if compared.compare is None or compared.compare.start is None or role is None or role.first is None \
                or compared.compare.start < role.first:
            return ""                 # data that starts later is not a period with nothing in it
        compiled = compile_query(compared, services.model, warehouse.dialect)
        result = warehouse.query(compiled.sql, max_rows=compiled.row_cap)
        moved = build_answer(question, compared, compiled, result.columns, result.rows)
    except Exception as exc:  # noqa: BLE001 - the answer stands without the comparison
        log.warning("core2 could not compare %r with the period before: %s", question, exc)
        return ""
    said = str((moved.get("answer") or {}).get("comparison") or "")
    if not said or said.startswith("nothing"):
        return ""
    prior = next((c for c in compiled.columns if c.role == "prior"), None)
    names = [n.casefold() for n in result.columns]
    if kpi is not None and prior is not None and len(result.rows) == 1 and prior.name.casefold() in names:
        measure = compared.measures[0].measure if compared.measures else None
        change = against(kpi.get("value"), result.rows[0][names.index(prior.name.casefold())], prior.format,
                         _period_badge(compared.compare),
                         better=measure.better if measure is not None else "higher")
        if change is not None:
            kpi["change"] = change
    return f"{said[:1].upper()}{said[1:]}, the period before."


_TRAILING = {"day": 30, "week": 12, "month": 12, "quarter": 8, "year": 5}   # periods a KPI's trend line shows


def _trend(plan: Plan, services: Services, ctx: Context, warehouse: Guarded, logical: Any, kpi: dict) -> None:
    """A KPI's trend line, from one more small query: the periods up to and including its own (the last 12
    months for a month), or, for a window that is no one period ("Jan–Jun"), the months or days inside it. A
    failure costs the line, never the answer."""
    from core2.resolve.time import add_units, unit_start

    start, end = logical.window.start, logical.window.end
    role = logical.parts[0].date.role if logical.parts and logical.parts[0].date else None
    if start is None or end is None or role is None or role.first is None or plan.time.window.fiscal:
        return
    last = role.last or ctx.today
    grain = None
    for unit in ("day", "week", "month", "quarter", "year"):
        following = add_units(unit_start(start, unit), unit, 1)
        if unit_start(start, unit) == start and (end == following or start < end < following and end > last):
            grain = unit                  # one whole period, or the current one to date
            break
    partial = end - dt.timedelta(days=1) > last     # the data stops inside the last period: it is not whole yet
    if grain is not None:
        first = max(add_units(start, grain, 1 - _TRAILING[grain]), unit_start(role.first, grain))
        window = Window(kind="between", start=first, end=end - dt.timedelta(days=1))
    elif (end - start).days > 62:
        grain, window = "month", Window(kind="between", start=start, end=end - dt.timedelta(days=1))
    elif (end - start).days >= 7:
        grain, window = "day", Window(kind="between", start=start, end=end - dt.timedelta(days=1))
    else:
        return
    trailing = plan.model_copy(update={"intent": "trend", "sort": [], "limit": None,
                                       "time": plan.time.model_copy(update={"grain": grain, "window": window})})
    try:
        series = resolve(trailing, services.model, replace(ctx, split_units=services.split_units))
        compiled = compile_query(series, services.model, warehouse.dialect)
        result = warehouse.query(compiled.sql, max_rows=compiled.row_cap)
    except Exception as exc:  # noqa: BLE001 - the KPI stands without its trend line
        log.warning("core2 could not draw the trend of %r: %s", kpi.get("label"), exc)
        return
    names = [n.casefold() for n in result.columns]
    period = next((c for c in compiled.columns if c.role == "period"), None)
    measure = next((c for c in compiled.columns if c.role == "measure"), None)
    if period is None or measure is None or period.name.casefold() not in names \
            or measure.name.casefold() not in names or len(result.rows) > 400:
        return
    at, of = names.index(period.name.casefold()), names.index(measure.name.casefold())
    found = {str(row[at])[:10]: row[of] for row in result.rows}
    starts = periods(Range(window.start, end), grain)
    values = [_amount(found.get(d.isoformat())) for d in starts]
    drawn = sparkline(values, partial=partial)
    if drawn:
        kpi["trend"] = {**drawn, "grain": grain,
                        "span": f"{period_label(starts[0], grain)} – {period_label(starts[-1], grain)}"}


def _amount(value: Any) -> float | None:
    try:
        return None if value is None or isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        return None


# Words that point at something rather than name it, and nothing more: "lowest", "the first one", "that
# division", "it". A name that only starts so ("Top Customer A", "Best Buy") is a name.
_REFERENCE = re.compile(r"^\s*(?:(?:the\s+)?(?:first|second|third|last|top|bottom|lowest|highest|biggest|smallest|"
                        r"largest|best|worst|previous|same)(?:\s+one)?|(?:that|this|these|those)(?:\s+[\w-]+){1,2}|"
                        r"it|them|one|its|their)\s*$", re.IGNORECASE)


def _points(value: str) -> bool:
    """Does a filter value point at a row rather than name one ("lowest", "that division", "first_one")?"""
    return bool(_REFERENCE.match(value) or _REFERENCE.match(value.replace("_", " ")) or stands_for(value))


def _asks_for(question: str, rules: list) -> bool:
    """Does the question ask about the very rows a table leaves out ("revenue on cancelled lines", "which customer
    type cancels the most")? Read from the values the rules leave out, by the start of each word."""
    asked = {w[:6] for w in re.findall(r"[a-z]{4,}", question.casefold())}
    for _, df in rules:
        left = " ".join(str(v) for v in [*(getattr(df, "shown", None) or []), *df.values])
        if asked & {w[:6] for w in re.findall(r"[a-z]{4,}", left.casefold())}:
            return True
    return False


def _say_left_out(payload: dict[str, Any], question: str, plan: Plan, services: Services, ctx: Context, *,
                  question_id: str = "", shown: list[tuple] | None = None) -> Plan:
    """Nothing matched, and the tables leave rows out by default: when the rows left out would answer it, say so
    and offer them ("No rows match" read as if there were none, when every one was a cancelled order). When the
    question asks about those very rows, it is answered with them, and says so. Returns the plan answered."""
    from core2.plan.catalog import left_out_words

    model = services.model
    measures = [found[1] for found in (find_slug(model, slug) for slug in plan.measures)
                if found is not None and isinstance(found[1], Measure)]
    tables = [model.tables[m.table] for m in measures if m.table in model.tables]
    rules = [(t, df) for t in {t.key: t for t in tables}.values() if t.readers_may_include for df in t.default_filters]
    if not rules:
        return plan
    wider_plan = plan.model_copy(update={"include_left_out": True})
    members: list[tuple] = []
    try:
        wider = _compute(question, wider_plan, services, ctx, question_id=question_id, started=time.perf_counter(),
                         shown=members)
    except Exception as exc:  # noqa: BLE001 - only a better sentence is lost; the answer stands as it is
        log.warning("core2: could not check the rows left out for %r: %s", question, exc)
        return plan
    if _found_nothing(wider, zero=True):      # a count of nothing is 0 with them too: no row is left out
        return plan
    words = "; ".join(left_out_words(model, df) for _, df in rules)
    if _asks_for(question, rules):
        payload.clear()
        payload.update(wider)
        if shown is not None:
            shown[:] = members      # "the first one" after it points into these rows
        trust = payload.setdefault("trust", {})
        trust["date_context"] = [*(trust.get("date_context") or []),
                                 f"Includes the rows left out by default ({words}): the question asks about them."]
        return wider_plan
    answer = payload.setdefault("answer", {})
    said = (answer.get("headline") or "No rows match.").rstrip(".")
    answer["headline"] = f"{said}: every row that matches is left out by default ({words})."
    payload["follow_up_suggestions"] = [dict(LEFT_OUT_CHIP), *(payload.get("follow_up_suggestions") or [])][:4]
    return plan


def _by_main_date(payload: dict[str, Any], question: str, plan: Plan, services: Services, ctx: Context, *,
                  question_id: str = "", shown: list[tuple] | None = None, periods: list[str] | None = None) -> Plan:
    """Nothing by a date the question did not name, the rows having none at all: answered by their table's main
    date, and said. 'Shipments with delivery status "Delivery Exception" in 2025' kept the delivered date of the
    question before it, and said "No rows match": a shipment that was never delivered has no delivered date. Rows
    that have the date, only not in that period ("shipped in June", none shipped then), keep the answer they got.
    Returns the plan answered."""
    model = services.model
    role = next((r for r in model.date_roles.values() if r.slug == plan.time.date), None)
    if role is None or role.is_default or plan.time.window.kind == "all" or role.table not in model.tables:
        return plan
    table = model.tables[role.table]
    main = model.date_roles.get(table.default_date or "")
    if main is None or main.slug == role.slug:
        return plan
    words = {w for w in re.findall(r"[a-z]+", role.name.lower()) if w not in ("date", "day", "time", "on", "of")}
    words |= {w.lower() for ws in role.synonyms.values() for w in ws}
    if words & set(re.findall(r"[a-z]+", question.lower())):
        return plan                     # "delivered in 2025": nothing delivered then is the answer
    ever = plan.model_copy(update={"time": plan.time.model_copy(update={"window": Window(), "compare": None,
                                                                        "grain": None}),
                                   "group_by": [], "sort": [], "limit": None})
    try:
        if not _found_nothing(_compute(question, ever, services, ctx, question_id=question_id,
                                       started=time.perf_counter()), zero=True):
            return plan                 # the rows have that date, only not then: nothing then is the answer
    except Exception as exc:  # noqa: BLE001 - the answer that found nothing stands
        log.warning("core2: could not check the dates of %r: %s", question, exc)
        return plan
    main_plan = plan.model_copy(update={"time": plan.time.model_copy(update={"date": main.slug})})
    members: list[tuple] = []
    spans: list[str] = []
    try:
        again = _compute(question, main_plan, services, ctx, question_id=question_id, started=time.perf_counter(),
                         shown=members, periods=spans)
    except Exception as exc:  # noqa: BLE001 - the answer that found nothing stands
        log.warning("core2: could not answer %r by its main date: %s", question, exc)
        return plan
    if _found_nothing(again):
        return plan
    payload.clear()
    payload.update(again)
    if shown is not None:
        shown[:] = members
    if periods is not None:
        periods[:] = spans
    trust = payload.setdefault("trust", {})
    trust["date_context"] = [f"None of these rows has a {role.name.lower()}: shown by {main.name.lower()}, the "
                             f"main date of {table.business_name.lower()} rows.",
                             *(trust.get("date_context") or [])]
    return main_plan


def _by_key(payload: dict[str, Any], question: str, plan: Plan, services: Services, ctx: Context, *,
            question_id: str = "", shown: list[tuple] | None = None) -> None:
    """A ranking that found nothing, or a breakdown whose every row has no name, asked again with each member
    whose table has no row for its key (so no name) shown by that key: "Which 10 warehouses have the least stock?"
    said "No rows match" on a copy whose warehouse table named none of the warehouses the stock rows held, and
    "which warehouses hold the most of it?" one row, "Unknown". The answer on screen, and what "the first one"
    points at, become those rows."""
    members: list[tuple] = []
    try:
        again = _compute(question, plan, services, replace(ctx, unnamed_by_key=True),
                         question_id=question_id, started=time.perf_counter(), shown=members)
    except Exception as exc:  # noqa: BLE001 - the answer that found nothing stands
        log.warning("core2: could not rank %r by its members' keys: %s", question, exc)
        return
    if _found_nothing(again):
        return
    payload.clear()
    payload.update(again)
    if shown is not None:
        shown[:] = members


CARRIED = 0.4       # the member that moved a figure most is followed when it carried this much of the change


def _one_step_down(question: str, plan: Plan, payload: dict[str, Any], services: Services, ctx: Context, *,
                   question_id: str = "") -> None:
    """A "why" answer followed one step down, as an analyst would: the member that carried most of the change
    (a store), then, inside it alone, what moved it (its product category). One more "why" of the same figure for
    that member, through the same groupings and access; said as steps, with the step itself one click away."""
    from core2.answer.facts import amount

    model = services.model
    found = payload.get("drivers") or {}
    change = _amount(found.get("change"))
    top = next((g for g in found.get("groupings") or [] if g.get("leaders")), None)
    if not change or top is None or float(top.get("explained") or 0) < 0.5:
        return
    lead = top["leaders"][0]
    member, moved = lead.get("member"), _amount(lead.get("change"))
    if no_member(member) or moved is None or abs(moved) < CARRIED * abs(change):
        return
    slug = top["slug"]
    named = model.entities[slug].label_column if slug in model.entities else None
    field = slug if slug in model.attributes else next(
        (a.slug for a in model.attributes.values() if named and a.column == named), None)
    if field is None or about_people(model, field):
        return
    deeper = plan.model_copy(update={"filters": [*plan.filters, Filter(field=field, op="eq", values=[member])],
                                     "group_by": [], "follow_up": "new"})
    try:
        inside = _compute(question, deeper, services, ctx, question_id=question_id, started=time.perf_counter())
    except Exception as exc:  # noqa: BLE001 - the answer stands at its first step
        log.info("core2: could not follow %r one step down: %s", question, exc)
        return
    within = inside.get("drivers") or {}
    nxt = next((g for g in within.get("groupings") or [] if g.get("leaders")), None)
    form = ((payload.get("data") or {}).get("column_formats") or {}).get("change") or "number"
    thing = str(top.get("word") or top.get("label") or "").lower()

    def said(value: float, lang: str = "en") -> str:
        return amount(value, form, lang=lang)

    whole, part = said(change), said(moved)
    if abs(moved) > abs(change):      # one member moved by more than the whole: the others went the other way
        en = (f"{thing.capitalize()} {member} moved by more than the whole change: {part} against {whole} overall, "
              "the rest going the other way.")
        fr = (f"{thing.capitalize()} {member} a bougé plus que l'ensemble : {said(moved, 'fr')} pour "
              f"{said(change, 'fr')} au total, le reste allant dans l'autre sens.")
    else:
        en = f"Most of the change came from {thing} {member}: {part} of {whole}."
        fr = f"L'essentiel de la variation vient de {thing} {member} : {said(moved, 'fr')} sur {said(change, 'fr')}."
    of, sur = ("against", "pour") if abs(moved) > abs(change) else ("of", "sur")
    checked = [{"en": f"Step 1, the change by {thing}: {member} {part} {of} {whole} overall.",
                "fr": f"Étape 1, la variation par {thing} : {member} {said(moved, 'fr')} {sur} "
                      f"{said(change, 'fr')} au total."}]
    figures = [moved, change]
    if nxt is not None and float(nxt.get("explained") or 0) >= 0.5:
        lead2 = nxt["leaders"][0]
        moved2 = _amount(lead2.get("change"))
        thing2 = str(nxt.get("word") or nxt.get("label") or "").lower()
        if not no_member(lead2.get("member")) and moved2 is not None:
            en += f" Inside {member}, {thing2} {lead2['member']} moved most: {said(moved2)}."
            fr += f" Dans {member}, c'est {thing2} {lead2['member']} qui a le plus bougé : {said(moved2, 'fr')}."
            checked.append({"en": f"Step 2, inside {member}, the change by {thing2}: {lead2['member']} {said(moved2)}.",
                            "fr": f"Étape 2, dans {member}, la variation par {thing2} : {lead2['member']} "
                                  f"{said(moved2, 'fr')}."})
            figures.append(moved2)
    else:
        en += f" Inside {member}, nothing stands out: the change is spread out."
        fr += f" Dans {member}, rien ne se détache : la variation est répartie."
        checked.append({"en": f"Step 2, inside {member}: the change is spread out.",
                        "fr": f"Étape 2, dans {member} : la variation est répartie."})
    # The answer's own conditions said too: the words, typed or sent, ask the same ("for customer "X"").
    tail = tail_fr = ""
    try:
        from core2.answer.builder import condition_words

        conditions = [c for c in resolve(plan, model, replace(ctx, split_units=services.split_units)).conditions
                      if c.kind != "by"]
        tail = "".join(f", {w}" for w in (condition_words(c, quoted=True) for c in conditions) if w)
        tail_fr = "".join(f", {w}" for w in (condition_words(c, quoted=True, lang="fr") for c in conditions) if w)
    except Exception:  # noqa: BLE001 - the step is asked by its own words
        pass
    ask = f'Why did it change, for {thing} "{member}"{tail}?'
    ask_fr = f'Pourquoi cela a-t-il varié, pour {thing} « {member} »{tail_fr} ?'
    seen = {"kind": "steps", "en": en, "fr": fr, "figures": figures, "days": [], "members": [member], "about": field}
    payload["reply"] = {"en": en, "fr": fr, "by": "facts", "seen": [seen], "checked": checked, "nothing": False,
                        "offers": [{"label": ask, "question": ask, "label_fr": ask_fr, "question_fr": ask_fr}]}
    payload["_dig"] = deeper          # the step itself, offered with no AI (taken off before the answer is shown)


_READING_WORD = re.compile(r"[a-z]+")
_READING_SKIP = {"to", "of", "the", "by", "through", "a", "an", "and", "via", "on", "for"}


def _stem(word: str) -> str:
    """A word without its ending, as the name of a link and a question both write it: "owned", "owner" -> own;
    "edited", "editor" -> edit; "shipped", "ship" -> ship."""
    if len(word) > 4:
        word = re.sub(r"(?:ing|ers|ors|er|or|ed|es|s)$", "", word)
        if len(word) > 3 and word[-1] == word[-2]:
            word = word[:-1]
    return word


def _reading_said(question: str, options: list[str]) -> str | None:
    """Of the ways a field can be reached ("Last editor user", "Owner user"), the one the question's words name: each
    word of its name that the others do not share is a word of the question, in any form ("posts owned by an elder
    user"). None unless exactly one is."""
    words = lambda text: {_stem(w) for w in _READING_WORD.findall(text.casefold()) if w not in _READING_SKIP}  # noqa: E731
    named = {o: words(o) for o in options}
    if len(named) < 2:
        return None
    shared = set.intersection(*named.values())
    asked = words(question)
    said = [o for o, w in named.items() if w - shared and (w - shared) <= asked]
    return said[0] if len(said) == 1 else None


def _completed(plan: Plan, field: str, option: str, model: SemanticModel) -> Plan:
    """A plan waiting for the link or the date the reader names, completed with ``option``."""
    if field == "time.date":
        # The offered dates are tables' defaults; another table may have a date of the same name.
        named = [r for r in model.date_roles.values() if r.name == option]
        role = next((r for r in named if r.is_default), named[0] if named else None)
        return plan.model_copy(update={"time": plan.time.model_copy(
            update={"date": role.slug if role is not None else option})})
    return plan.model_copy(update={"via": {**plan.via, field: option}})


READINGS = 4        # fair readings worked out before asking which one is meant


def _each_reading(question: str, plan: Plan, exc: ResolveError, services: Services, ctx: Context, *,
                  question_id: str = "") -> dict[str, tuple[float, str]]:
    """Each fair reading's figure for the whole question -- one total, the figure as the answer says it -- for
    the readings that can be worked out."""
    from core2.answer.extra import _base

    out: dict[str, tuple[float, str]] = {}
    if exc.field in plan.group_by and exc.field not in {f.field for f in plan.filters}:
        return out        # only how the rows are grouped differs: every reading has the same total, nothing to say
    for option in list(exc.options)[:READINGS]:
        reading = _completed(plan, exc.field or "", option, services.model)
        if reading.group_by or reading.time.grain:
            reading = _base(reading, intent="value")      # readings compared on the whole
        try:
            got = _compute(question, reading, services, ctx, question_id=question_id, started=time.perf_counter())
        except Exception as again:  # noqa: BLE001 - that reading is offered without its figure
            log.info("core2: could not work out %r read as %s: %s", question, option, again)
            continue
        rows = (got.get("data") or {}).get("rows") or []
        kpi = got.get("kpi") if isinstance(got.get("kpi"), dict) else {}
        value = _amount(kpi.get("value")) if kpi else None
        if value is None and len(rows) == 1:
            value = next((_amount(v) for v in rows[0].values() if _amount(v) is not None), None)
        said = str((got.get("answer") or {}).get("short_value") or "")
        if value is not None and said:
            out[option] = (value, said)
    return out


def _same_either_way(figures: dict[str, tuple[float, str]], plan: Plan) -> str | None:
    """The reading to answer with when every reading gives the same figure and that figure is the whole answer
    (one value: a breakdown read two ways can differ member by member even where the totals agree)."""
    if len(figures) < 2 or plan.group_by or plan.time.grain:
        return None
    values = [v for v, _ in figures.values()]
    top = max(abs(v) for v in values)
    return next(iter(figures)) if max(values) - min(values) <= 0.005 * top else None


def _with_numbers(asked: dict[str, Any], figures: dict[str, tuple[float, str]]) -> None:
    """A question back with each reading's figure beside its name -- unless every one is the same, which says
    nothing."""
    if len({round(v, 6) for v, _ in figures.values()}) < 2:
        return
    clarify = asked.get("clarify") or {}
    options = list(clarify.get("options") or [])
    clarify["figures"] = {o: said for o, (_, said) in figures.items()}
    listed = [f"{o} ({figures[o][1]})" if o in figures else o for o in options]
    head = str(clarify.get("question") or "").rstrip(" ?:.")
    text = f"{head}: {listed[0] if len(listed) == 1 else ', '.join(listed[:-1]) + ' or ' + listed[-1]}?"
    clarify["question"] = text
    asked["answer"]["headline"] = text


def _said_same_either_way(payload: dict[str, Any], message: str, figures: dict[str, tuple[float, str]]) -> None:
    """An answer read more than one way, each to the same figure: said on its reply, so the reader is not asked."""
    options = list(figures)
    said = next(iter(figures.values()))[1]
    seen = {"kind": "same_either_way", "about": "", "days": [], "members": [],
            "figures": [next(iter(figures.values()))[0]],
            "en": f"{message}, and it comes to the same either way ({' or '.join(options)}): {said}.",
            "fr": f"La question se lit de plusieurs façons ({' ou '.join(options)}), et le résultat est le même : "
                  f"{said}."}
    checked = [{"en": "Worked out each reading: " + "; ".join(f"{o} {s}" for o, (_, s) in figures.items()) + ".",
                "fr": "Calculé chaque lecture : " + " ; ".join(f"{o} {s}" for o, (_, s) in figures.items()) + "."}]
    reply = payload.get("reply")
    if isinstance(reply, dict) and reply.get("en"):
        reply.update({"en": f"{seen['en']} {reply['en']}", "fr": f"{seen['fr']} {reply.get('fr') or ''}".strip(),
                      "seen": [seen, *(reply.get("seen") or [])], "checked": [*checked, *(reply.get("checked") or [])]})
    else:
        payload["reply"] = {"en": seen["en"], "fr": seen["fr"], "by": "facts", "seen": [seen], "checked": checked,
                            "offers": [], "nothing": False}


def _named_instead(plan: Plan, value: str, name: str, field: str | None = None) -> Plan:
    """The plan with the member ``value`` it names replaced by ``name`` (of the grouping ``field``, when the name
    is another grouping's)."""
    filters = [f.model_copy(update={"values": [name if str(v) == value else v for v in f.values],
                                    **({"field": field} if field and any(str(v) == value for v in f.values) else {})})
               for f in plan.filters]
    return plan.model_copy(update={"filters": filters, "follow_up": "new"})


def _did_you_mean(question: str, plan: Plan, value: str, services: Services, ctx: Context, *,
                  question_id: str = "", most: int = 2) -> list[tuple[str, str, str, Plan]]:
    """The members written most like a name the data does not have, each asked first through the reader's own
    access and kept only when it has rows: (its name, the question asked with it, its answer's sentence, its
    plan). The grouping's own members first; else the one other grouping holding names written like it ("Key
    Acounts" put on the account, for the cost centre Key Accounts). Never for a person's name."""
    from core2.resolve.resolver import find_slug

    model = services.model
    f = next((f for f in plan.filters if any(str(v) == value for v in f.values)), None)
    if f is None or about_people(model, f.field):
        return []
    # A filter on the store itself ("store") is on its name: the index keeps the members by their name's field.
    found = find_slug(model, f.field)
    field = f.field
    if found is not None and isinstance(found[1], Entity):
        column = found[1].label_column or found[1].code_column
        field = next((a.slug for a in model.attributes.values() if a.column == column), field)
        if about_people(model, field):
            return []
    out = []
    near = [(None, name) for name in services.index.nearest(field, value)]
    if not near:
        anywhere = services.index.nearest_anywhere(value)
        if len({a for a, _ in anywhere}) == 1 and not about_people(model, anywhere[0][0]):
            near = list(anywhere)
    for home, name in near[:most]:
        meant = _named_instead(plan, value, name, home)
        try:
            got = _compute(question, meant, services, ctx, question_id=question_id, started=time.perf_counter())
        except Exception as exc:  # noqa: BLE001 - only the suggestion is lost
            log.info("core2: could not check %r as the member meant: %s", name, exc)
            continue
        if _found_nothing(got, zero=True):
            continue
        asked = question.replace(f'"{value}"', f'"{name}"') if f'"{value}"' in question else \
            re.sub(re.escape(value), f'"{name}"', question, count=1, flags=re.IGNORECASE)
        if asked == question:
            asked = f'{question.rstrip(" ?")} for "{name}"'
        out.append((name, asked, str((got.get("answer") or {}).get("headline") or ""), meant))
    return out


# "Yes", "ok, show me", "oui merci": the whole message says yes to the answer's own offer, and nothing more ("yes,
# and by store" is a question of its own).
_YES = re.compile(r"^\s*(?:yes|yeah|yep|yup|sure|ok(?:ay)?|please|go ahead|do it|do that|show me|let'?s see|"
                  r"oui|ouais|d'accord|vas-y|allez-y|volontiers|bien sûr)(?:[\s,!.]+(?:yes|please|thanks|thank you|"
                  r"do it|do that|show me|go ahead|show it|s'il te plaît|s'il vous plaît|merci|montre|montrez))*"
                  r"[\s!.]*$", re.IGNORECASE)


def _look(plan: Plan, payload: dict[str, Any], services: Services, ctx: Context, *, empty: bool,
          carried: list | None = None, standing: list | None = None) -> Any:
    """The answer looked at before it is shown (core2/agent/look.py); None when there is nothing to say, or the
    answer already says why it found nothing (the data ends before the period, the rows are left out)."""
    from core2.agent.look import look

    headline = str((payload.get("answer") or {}).get("headline") or "")
    if payload.get("beyond_data") or "left out by default" in headline or payload.get("kind") == "which_end":
        return None
    rules = replace(ctx, split_units=services.split_units)
    warehouse = Guarded(services.warehouse)
    try:
        logical = resolve(plan, services.model, rules)
    except Exception:  # noqa: BLE001 - nothing to look at
        return None

    def run(asked: Plan) -> tuple[Any, Any, Any]:
        lg = resolve(asked, services.model, rules)
        compiled = compile_query(lg, services.model, warehouse.dialect)
        return lg, compiled, warehouse.query(compiled.sql, max_rows=min(compiled.row_cap or 500, 500))

    return look(plan, logical, payload, run, empty=empty, carried=carried or [], standing=standing or [])


def _offer(payload: dict[str, Any], session: Session, looked: Any, *, empty: bool) -> None:
    """What the answer saw, said on it, and its next questions as buttons the conversation runs with no AI (the
    first one is what "yes" runs). An answer that found nothing offers only these: "why did it change?" has
    nothing to explain."""
    reply = looked.reply()
    reply["nothing"] = empty        # the reply says it all: no empty figure or table under it
    payload["reply"] = reply
    offers = [{**o, "kind": "look"} for o in reply["offers"]]
    for o in looked.offers:
        for words in (o.en, o.fr):
            session.offers[_plain(words)] = o.plan
    session.yes = looked.offers[0].plan if looked.offers else None
    rest = [] if empty else [s for s in payload.get("follow_up_suggestions") or []
                             if s.get("question") not in {o["question"] for o in offers}]
    payload["follow_up_suggestions"] = [*offers, *rest][:6]


def _found_nothing(payload: dict[str, Any], *, zero: bool = False) -> bool:
    """No rows, or only empty values: a total over nothing comes back as one row with no value (``zero``: or
    a count of nothing, as 0)."""
    rows = (payload.get("data") or {}).get("rows") or []
    if zero and len(rows) == 1:
        return all(v is None or v == 0 and not isinstance(v, bool) for v in rows[0].values())
    return all(all(v is None for v in row.values()) for row in rows)


def _members_named(plan: Plan, model: SemanticModel, index: MemberIndex,
                   unknown_left_out: list[tuple[str, str]] | None = None) -> tuple[Plan, tuple[str, str] | None]:
    """The plan with each member it filters on spelled as stored, and the first one the index does not know.

    "How did those same customers do in 2025?" was planned with customers the AI
    was never told the names of (it sees the plan behind the answer on screen,
    not its rows): the filter matched nothing, and the answer was "No rows
    match", as if they had bought nothing. A member the index does not know is
    returned so that an empty answer can say so; the query still runs, since a
    member added after the index was read is real. One written in another case
    ("retail") is the stored one ("RETAIL"): the warehouse compares text exactly. A member left out that the
    data does not have ("outside the United States" where countries are stored otherwise) leaves nothing out:
    it is added to ``unknown_left_out`` so that the answer says so, never as if something had been left out.
    """
    missing: tuple[str, str] | None = None

    def spelled(listed: list) -> list:
        return [as_stored(f) for f in listed]

    def as_stored(f):
        nonlocal missing
        found = find_slug(model, f.field) if f.op in ("eq", "in", "ne", "not_in") else None
        attribute: Attribute | None = None
        thing = ""
        if found is not None and isinstance(found[1], Attribute):
            attribute = found[1]
            thing = attribute.business_name.lower()
        elif found is not None and isinstance(found[1], Entity):
            entity = found[1]
            column = entity.label_column or entity.code_column
            attribute = next((a for a in model.attributes.values() if a.column == column), None)
            thing = entity.business_name.lower()
        if attribute is None or attribute.slug not in index.attributes:
            return f
        if f.op in ("eq", "in") and all(isinstance(v, str) and index.stored(attribute.slug, v) is None
                                         for v in f.values):
            # "Sales of Lakeshore" put on the region: the one other grouping that has the member, reached from
            # the figure, holds it ("Lakeshore Store", a store).
            elsewhere = _member_home(model, plan, index, [str(v) for v in f.values], attribute.slug)
            if elsewhere is not None:
                return f.model_copy(update={"field": elsewhere[0], "values": elsewhere[1]})
        values = []
        for value in f.values:
            stored = index.stored(attribute.slug, value) if isinstance(value, str) else value
            if stored is None and f.op in ("eq", "in") and missing is None:
                missing = (thing, str(value))
            elif stored is None and f.op in ("ne", "not_in") and unknown_left_out is not None:
                unknown_left_out.append((thing, str(value)))
            values.append(value if stored is None else stored)
        return f.model_copy(update={"values": values})

    filters = spelled(plan.filters)
    absent = plan.absent.model_copy(update={"without": spelled(plan.absent.without),
                                            "had": spelled(plan.absent.had)}) if plan.absent is not None else None
    return plan.model_copy(update={"filters": filters, "absent": absent}), missing


def _table_of(model: SemanticModel, slug: str) -> str | None:
    """The table a grouping or filter slug is on (an attribute's column's, an entity's own)."""
    found = find_slug(model, slug)
    if found is not None and isinstance(found[1], Attribute) and found[1].column in model.columns:
        return model.columns[found[1].column].table
    if found is not None and isinstance(found[1], Entity):
        return found[1].table
    return None


def _role_said(model: SemanticModel, plan: Plan, f: Any, index: MemberIndex) -> str | None:
    """The link role a filter's values name in place of a member ("manager" on the job level: no level is called
    so, and an employee's manager is a link of the employees to themselves); None when they name a member, or no
    such role of the table the filter is on."""
    from core2.resolve import paths as P

    if f.op not in ("eq", "in") or not f.values or not all(isinstance(v, str) for v in f.values):
        return None
    found = find_slug(model, f.field)
    table = _table_of(model, f.field)
    if found is None or table is None:
        return None
    attribute = found[1] if isinstance(found[1], Attribute) else next(
        (a for a in model.attributes.values() if isinstance(found[1], Entity) and a.column == found[1].label_column),
        None)
    if attribute is None or attribute.slug not in index.attributes \
            or any(_member_form(index, attribute.slug, v) != v or index.stored(attribute.slug, v) for v in f.values):
        return None
    starts = {m.table for m in model.measures.values() if m.slug in plan.measures} or {table}
    roles = {r for start in starts for r in P.role_names(model, start, table)}

    def said(text: str) -> tuple[str, ...]:
        return tuple(names.singular(w) for w in _plain(text).split())

    hits = [{r for r in roles if said(str(v)) == said(r)} for v in f.values]
    if any(len(h) != 1 for h in hits) or len(set().union(*hits)) != 1:
        return None
    return hits[0].pop()


def _roles_named(plan: Plan, model: SemanticModel, index: MemberIndex) -> Plan:
    """ "Avg number of direct reports per manager" and "which 5 managers have the most active direct reports?" were
    planned with a job level called "manager", which the data does not have, and refused. "Manager" names the link
    from an employee to another employee: a filter whose values are no member but name such a link is that link --
    the plan's grouping of that table, or what it works out per member, follows it -- and the filter goes. A plan
    that reaches the table no such way keeps the filter, and its refusal."""
    if plan.kind != "query" or not plan.filters:
        return plan
    per = [plan.per.of] if plan.per is not None and plan.per.of else []
    via = dict(plan.via)
    kept = []
    for f in plan.filters:
        role = _role_said(model, plan, f, index)
        table = _table_of(model, f.field) if role else None
        reached = [s for s in [*plan.group_by, *per] if table and _table_of(model, s) == table]
        if role is None or not reached:
            kept.append(f)
            continue
        if any(_plain(via.get(s, "").removeprefix("role:")) == _plain(role) for s in reached):
            continue                                      # the grouping follows the link already
        free = list(dict.fromkeys(s for s in reached if not via.get(s)))
        if len(free) != 1:
            kept.append(f)
            continue
        via[free[0]] = role
    if len(kept) == len(plan.filters):
        return plan
    return plan.model_copy(update={"filters": kept, "via": via})


# "under "Operating Expenses"", "sous « Charges »": the members a heading holds.
_UNDER = re.compile(r"\b(?:under|beneath|below|within|sous)\s+(?:the\s+|le\s+|la\s+|les\s+|l')?[\"“«']?\s*$",
                    re.IGNORECASE)


def _under_a_parent(plan: Plan, question: str, model: SemanticModel) -> Plan:
    """ 'top 5 accts under "Operating Expenses"' filtered the account called so, a heading no line is posted to:
    "No rows match". A quoted member written after "under" (sous) of a list that links to itself (an account's
    parent account, an employee's manager) narrows to the members it holds, by that link; the plan's grouping of
    the same list still lists the members themselves."""
    if plan.kind != "query" or not plan.filters:
        return plan
    from core2.plan.values import quoted_spans

    via, filters = dict(plan.via), list(plan.filters)
    for a, b in quoted_spans(question):
        if not _UNDER.search(question[:a]):
            continue
        name = _plain(question[a:b])
        for i, f in enumerate(filters):
            table = _table_of(model, f.field)
            selves = [j for j in model.joins.values() if j.from_table == j.to_table == table and j.role]
            if f.op not in ("eq", "in") or not any(_plain(str(v)) == name for v in f.values) or len(selves) != 1:
                continue
            field = f.field
            if field in plan.group_by:      # the grouping lists the members: the filter names the list otherwise
                entity = next((e for e in model.entities.values() if e.table == table), None)
                field = entity.slug if entity is not None and field != entity.slug else next(
                    (x.slug for x in model.attributes.values() if entity is not None
                     and x.column == (entity.label_column or entity.code_column)), field)
                if field in plan.group_by:
                    continue
            filters[i] = f.model_copy(update={"field": field})
            via[field] = f"role:{selves[0].role}"
    if via == plan.via:
        return plan
    return plan.model_copy(update={"filters": filters, "via": via})


def _member_home(model: SemanticModel, plan: Plan, index: MemberIndex, values: list[str],
                 chosen: str) -> tuple[str, list[str]] | None:
    """The one grouping, other than ``chosen``, whose members ``values`` all name and that the plan's figures
    reach, with those members as stored; None when there is no such grouping, or more than one."""
    from core2.resolve import paths as P

    homes: dict[str, list[str]] = {}
    for value in values:
        found = {a: v for a, v in index.homes(value) if a != chosen}
        for a in list(homes) or list(found):
            if a in found:
                homes.setdefault(a, []).append(found[a])
            else:
                homes.pop(a, None)
        if not homes:
            return None
    homes = {a: vs for a, vs in homes.items() if len(vs) == len(values)}
    tables = {m.table for m in (find_slug(model, s) for s in plan.measures) if m is not None
              for m in [m[1]] if isinstance(m, Measure)}
    reached = [a for a in homes if a in model.attributes and model.attributes[a].column in model.columns
               and all(model.columns[model.attributes[a].column].table == t
                       or P.all_paths(model, t, model.columns[model.attributes[a].column].table) for t in tables)]
    if len(reached) != 1:
        return None
    return reached[0], homes[reached[0]]


def _a_things_own_name(model: SemanticModel, found: list[ValueMatch]) -> ValueMatch | None:
    """Of a name two groupings share, the reading where it is a thing's own name: "Ontario" names the region,
    while the sales team is only called after it. None unless exactly one of the groupings is the name of a thing
    (a region, a store, a product) and the name is one member of it."""
    labels = {e.label_column for e in model.entities.values() if e.label_column}
    own = [m for m in found if m.attribute in model.attributes and model.attributes[m.attribute].column in labels]
    if len({m.attribute for m in own}) != 1 or len({normalise(str(m.value)) for m in own}) != 1:
        return None
    return own[0]


_ASSIGNED = re.compile(r"([A-Za-z_][\w ]{0,60}?)\s*=\s*(?:'([^'\n]+)'|\"([^\"\n]+)\"|([^\s;,)'\"?!]+))")


def _assigned(model: SemanticModel, index: MemberIndex, question: str) -> list[ValueMatch]:
    """Members written as a field's value ("Virtual = F", "DOC = 31", "status = 'Open'"): the field named before
    the sign, by its column's own name, its business name or its slug; read as if quoted, for that field only."""
    found: list[ValueMatch] = []
    for m in _ASSIGNED.finditer(question or ""):
        group = next(i for i in range(2, 5) if m.group(i) is not None)
        value, start, end = m.group(group), m.start(group), m.end(group)
        if group == 4:
            value = value.rstrip(".")          # the sentence's own full stop ("DOC = 31."), never the code's
            end = start + len(value)
        words = normalise(m.group(1)).split()
        said = {" ".join(words[-n:]) for n in range(1, min(3, len(words)) + 1)}
        for slug, a in model.attributes.items():
            column = model.columns.get(a.column)
            names = {normalise(n) for n in (column.name.replace("_", " ") if column else "", a.business_name,
                                             slug.rsplit(".", 1)[-1].replace("_", " ")) if n}
            if names & said:
                # The value as written, exactly: a code is never part of another ("31" of "131").
                stored = next((v for attr, v in index._entries(normalise(value)) if attr == slug), None)
                if stored is not None:
                    found.append(ValueMatch(value, start, end, slug, stored))
    return found


_SPAN_WORD = re.compile(r"[a-z0-9]+")
_BESIDE_SKIP = {"of", "the", "in", "at", "for", "from", "a", "an", "and", "or", "de", "du", "des", "la", "le", "les"}


def _one_reading(model: SemanticModel, question: str, found: list[ValueMatch]) -> ValueMatch | None:
    """Of a name several fields hold, the one the question means, or None when it stays unclear:

    * a thing's own name ("Ontario" the region, not the sales team called after it);
    * one field repeated on the tables of what happened (a school's district, kept again on its test scores and
      its meal counts): the thing's own copy, where each table is linked to the thing;
    * the field written beside the name ("Alameda County", "the city of Fremont");
    * the plain field over one that qualifies it (the city, not the mailing city);
    * the field the question names elsewhere ("between San Diego and Santa Barbara, which county ...").
    """
    own = _a_things_own_name(model, found)
    if own is not None:
        return own
    cands = [m for m in found if m.attribute in model.attributes]
    if not cands or len({normalise(str(m.value)) for m in cands}) != 1:
        return None

    def name(m: ValueMatch) -> str:
        a = model.attributes[m.attribute]
        return normalise(a.business_name or m.attribute.rsplit(".", 1)[-1].replace("_", " "))

    near_tables: dict[str, set[str]] = {}
    for j in (getattr(model, "joins", None) or {}).values():
        near_tables.setdefault(j.from_table, set()).add(j.to_table)
        near_tables.setdefault(j.to_table, set()).add(j.from_table)

    def reached(table: str) -> set[str]:
        """The tables linked to ``table``, through any number of links."""
        seen, todo = {table}, [table]
        while todo:
            for t in near_tables.get(todo.pop(), ()):
                if t not in seen:
                    seen.add(t)
                    todo.append(t)
        return seen

    fields: dict[str, list[ValueMatch]] = {}
    for m in cands:
        fields.setdefault(name(m), []).append(m)
    readings: dict[str, ValueMatch] = {}
    for field_name, ms in fields.items():
        held = [m for m in ms if model.attributes[m.attribute].entity in model.entities]
        if len(ms) == 1:
            readings[field_name] = ms[0]
        elif len({model.attributes[m.attribute].entity for m in held}) == 1:
            thing = model.entities[model.attributes[held[0].attribute].entity]  # type: ignore[index]
            copies = [m for m in ms if m not in held]
            tied = reached(thing.table)
            if all(model.columns[model.attributes[m.attribute].column].table in tied for m in copies):
                readings[field_name] = held[0]
                continue
            return None                     # one name, unlinked tables: which is meant is unclear
        else:
            return None
    if len(readings) == 1:
        return next(iter(readings.values()))
    start, end = cands[0].start, cands[0].end
    words = lambda text: [w for w in _SPAN_WORD.findall(text.casefold()) if w not in _BESIDE_SKIP]  # noqa: E731
    beside = set(words(question[end:])[:2]) | set(words(question[:start])[-2:])
    tokens = {f: set(f.split()) for f in readings}
    near = [f for f in readings if tokens[f] & beside]
    if len(near) == 1:
        return readings[near[0]]
    plain = [f for f in readings if not any(tokens[g] < tokens[f] for g in readings if g != f)]
    if len(plain) == 1:
        return readings[plain[0]]
    rest = normalise(question[:start] + " " + question[end:])
    named = [f for f in plain if re.search(rf"\b{re.escape(f)}\b", rest)]
    return readings[named[0]] if len(named) == 1 else None


def _own_names(model: SemanticModel) -> frozenset[str]:
    """The names of the model's own figures, groupings and fields ("Net Amount", "Store"), and their words for
    them: written in a question they name what is counted or grouped, never a member to narrow to."""
    names: set[str] = set()
    for m in model.measures.values():
        names |= {m.business_name, m.slug.replace("_", " "), *(w for ws in m.synonyms.values() for w in ws)}
    for a in model.attributes.values():
        names |= {a.business_name, *(w for ws in a.synonyms.values() for w in ws)}
    for e in model.entities.values():
        names |= {e.business_name, e.slug.replace("_", " "), *(w for ws in e.synonyms.values() for w in ws)}
    return frozenset(normalise(n) for n in names if n)


def _grounded(matches: list[ValueMatch], session: Session) -> set[str]:
    """Member values a filter may name: those the reader quoted or pointed at on screen ("the first one"), those
    the answers before already filtered on (a follow-up keeps them), and those a question back offered."""
    values = {normalise(str(m.value)) for m in matches}
    for turn in session.turns[-HISTORY:]:
        if turn.plan is None:
            continue
        for f in turn.plan.filters:
            values |= {normalise(v) for v in f.values if isinstance(v, str)}
        if turn.plan.clarify is not None:
            values |= {normalise(o) for o in turn.plan.clarify.options}
    return values


def _member_form(index: MemberIndex, attribute: str, value: object) -> object:
    """The stored member a value names in the singular or the plural ("Bike" -> "Bikes"); else the value."""
    if not isinstance(value, str) or index.stored(attribute, value) is not None:
        return value
    for other in (f"{value}s", f"{value}es", value[:-1] if value.endswith("s") else "",
                  value[:-2] if value.endswith("es") else ""):
        found = index.stored(attribute, other) if other else None
        if found is not None:
            return found
    return value


_MONTH_WORDS = {**{m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                                "nov", "dec"), 1)},
                **{m: i for i, m in enumerate(("january", "february", "march", "april", "may", "june", "july",
                                                "august", "september", "october", "november", "december"), 1)},
                **{m: i for i, m in enumerate(("janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
                                                "septembre", "octobre", "novembre", "décembre"), 1)},
                "sept": 9, "fevrier": 2, "aout": 8, "decembre": 12}
_MONTH_YEAR = re.compile(r"\b(" + "|".join(sorted(_MONTH_WORDS, key=len, reverse=True)) + r")\.?\s+((?:19|20)\d{2})\b",
                         re.IGNORECASE)
# Words that make a month one end of a span, or set it against another period: the AI's window stands.
_SPAN_WORDS = re.compile(r"\b(?:since|from|to|until|till|through|thru|between|before|after|vs\.?|versus|against|"
                         r"compared?|than|and|depuis|jusqu|entre|avant|après|contre|par\s+rapport|et|ytd|to\s+date|"
                         r"last|previous|past|rolling|trailing)\b|-|–", re.IGNORECASE)


def _one_month_named(plan: Plan, question: str) -> Plan:
    """ "Why did spend spike in nov 2025?" after an answer for 2025 was planned for all of 2025, and the why
    compared December with November. A question naming one month and no other period, nor any span, is that
    month."""
    if plan.kind != "query" or plan.time.compare is not None:
        return plan
    named = _MONTH_YEAR.findall(question)
    years = re.findall(r"\b(?:19|20)\d{2}\b", question)
    if len(named) != 1 or len(years) != 1 or _SPAN_WORDS.search(_MONTH_YEAR.sub(" ", question)):
        return plan
    month, year = _MONTH_WORDS[named[0][0].lower()], int(named[0][1])
    start = dt.date(year, month, 1)
    end = (start.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)
    window = plan.time.window
    if window.kind == "between" and window.start == start and window.end == end:
        return plan
    if not (window.kind == "between" and window.start and window.end and window.start <= start and end <= window.end
            and (window.start, window.end) != (start, end)):
        return plan          # only a window that holds the month and more: the AI kept a wider period
    time = plan.time.model_copy(update={"window": Window(kind="between", start=start, end=end),
                                        "grain": None if plan.intent == "drivers" else plan.time.grain})
    return plan.model_copy(update={"time": time})


_BY_DAY = re.compile(r"\b(?:by|per|each|every)\s+day\b|\bdaily\b|\bday[- ]by[- ]day\b|\bpar\s+jour\b|"
                     r"\bjour\s+par\s+jour\b|\bquotidien", re.IGNORECASE)
_WEEKDAY = re.compile(r"\bday\s+of\s+(?:the\s+)?week\b|\bweekdays?\b|\bweekends?\b|\bjours?\s+de\s+la\s+semaine\b|"
                      r"\b(?:mon|tues|wednes|thurs|fri|satur|sun)days?\b|\b(?:lundi|mardi|mercredi|jeudi|vendredi|"
                      r"samedi|dimanche)s?\b", re.IGNORECASE)


def _by_day(plan: Plan, question: str) -> Plan:
    """ "Paid amount by day in June 2026" is each day of June, never Monday to Sunday: the AI added the day of
    the week to the days, and the answer ranked weekdays. Taken off unless the question names the week's days.
    "Daily spend in March" grouped by the day of the week with no grain at all is the days of March too."""
    if plan.time.grain not in ("day", None) or "time:day_of_week" not in plan.group_by \
            or not _BY_DAY.search(question) or _WEEKDAY.search(question):
        return plan
    return plan.model_copy(update={"group_by": [g for g in plan.group_by if g != "time:day_of_week"],
                                   "time": plan.time.model_copy(update={"grain": "day"}),
                                   "intent": "trend" if plan.intent == "breakdown" else plan.intent})


def _quoted_lost(plan: Plan, question: str, model: SemanticModel, index: MemberIndex) -> Plan:
    """A name the reader quoted that the data does not have, and that no condition of the plan holds: the AI left
    it out ('spend on "Home improvment"', for "Home improvement"), and the answer was for every one. It goes back
    on as written, on the grouping whose member is written most like it, so the answer says there is none and
    offers that one. Never on a person's name, and never when two groupings hold names as close."""
    if plan.kind != "query":
        return plan
    held = [normalise(str(v)) for f in plan.filters for v in f.values]
    # A figure's or a grouping's own name in quotes ('"net sales" by month') is no member written wrong.
    named = {normalise(str(x.business_name)) for x in [*model.measures.values(), *model.attributes.values(),
                                                        *model.entities.values()] if x.business_name}
    added: list[Filter] = []
    for a, b in quoted_spans(question):
        text = question[a:b].strip()
        key = normalise(text)
        if len(key) < 4 or key in named or index.quoted(f'"{text}"') \
                or any(h and (key in h or h in key) for h in held):
            continue
        near = index.nearest_anywhere(text)
        fields = {field for field, _ in near}
        if len(fields) != 1 or about_people(model, next(iter(fields))):
            continue
        added.append(Filter(field=next(iter(fields)), op="eq", values=[text]))
    return plan.model_copy(update={"filters": [*plan.filters, *added]}) if added else plan


_END_MARKS = "?!.;:"


def _quoted_as_written(plan: Plan, question: str, model: SemanticModel, index: MemberIndex) -> Plan:
    """A quoted value filters as the reader wrote it, its last mark kept: "the post titled 'Open source tools for
    visualizing data?'" was filtered on the title without its question mark, and found no post. Where the field's
    members are read, the one stored is taken; where they are not (a free-text title), either way is matched:
    whether the mark is the title's own or the sentence's, only the data knows."""
    if plan.kind != "query" or not plan.filters:
        return plan
    quoted = [question[a:b].strip() for a, b in quoted_spans(question)]
    marked = {q.rstrip(_END_MARKS).strip().casefold(): q for q in quoted if q.rstrip(_END_MARKS) != q}
    if not marked:
        return plan
    filters, changed = [], False
    for f in plan.filters:
        found = find_slug(model, f.field) if f.op in ("eq", "in") else None
        attribute = found[1] if found is not None and isinstance(found[1], Attribute) else None
        if attribute is None:
            filters.append(f)
            continue
        read = attribute.slug in index.attributes
        values = list(f.values)
        for i, v in enumerate(f.values):
            written = marked.get(str(v).strip().casefold()) if isinstance(v, str) else None
            if written is None or written in values:
                continue
            if not read:
                values.append(written)
            elif index.stored(attribute.slug, v) is None and index.stored(attribute.slug, written) is not None:
                values[i] = index.stored(attribute.slug, written)
        if values != list(f.values):
            changed = True
            f = f.model_copy(update={"op": "in" if len(values) > 1 else f.op, "values": values})
        filters.append(f)
    return plan.model_copy(update={"filters": filters}) if changed else plan


def _whole_names(plan: Plan, matches: list[ValueMatch]) -> Plan:
    """A name the reader quoted whole filters as written, never as its pieces: "VersaBase cream", an ingredient,
    was planned as an FDA name "VersaBase" and a dosage form "Cream", and the answer said there was no
    "VersaBase". Filters whose every value is a piece of the quoted name give way to the name itself."""
    written = {normalise(m.text) for m in matches}
    for m in matches:
        quoted = normalise(str(m.value))
        if not m.exact or normalise(m.text) != quoted or len({x.attribute for x in matches
                                                              if normalise(str(x.value)) == quoted}) > 1:
            continue     # a containment, or a name two fields hold: the AI's reading stands
        filters = list(plan.filters)
        if any(normalise(str(v)) == quoted for f in filters for v in f.values):
            continue
        pieces = [f for f in filters if f.op in ("eq", "in") and f.values and all(
            isinstance(v, str) and len(normalise(v)) >= 3 and normalise(v) in quoted and normalise(v) not in written
            for v in f.values)]      # a name quoted on its own is never a piece of another
        if pieces:
            kept = [f for f in filters if f not in pieces]
            plan = plan.model_copy(update={"filters": [*kept, Filter(field=m.attribute, op="eq", values=[m.value])]})
    return plan


def _quoted_filters(plan: Plan, question: str, model: SemanticModel, grounded: set[str],
                    index: MemberIndex) -> tuple[Plan, list[tuple[str, str]], list[tuple[str, str]]]:
    """The plan without any filter whose member is a word of the question the reader did not quote, and what
    was left out.

    "How much stock is available?" was filtered on a stock status called AVAILABLE, and "open orders by
    month" on an order status OPEN: words of the question matched a member and became a WHERE condition.
    A member the question writes narrows the answer only when the reader put it in quotes ("OEM",
    “North”), pointed at it on screen, or a previous answer of the conversation already filtered on it. A
    code the AI chose for what the question means, not one of its words ("units received" -> movement
    type RCV), still filters, and the answer names it. A number or a flag (cancelled = 1) is no name. A name
    the data does not have is kept too: the AI made it up ("those same customers", whose names it was never
    given), and the answer says there is none and asks which one the reader means, never a total for all. A word
    the reader wrote in lower case that no member is called describes what is asked about ("which referral
    partner", where every partner refers): it is left out as well, and returned apart, said with no button to quote
    it.
    """
    said = [normalise(question[a:b]) for a, b in quoted_spans(question)]
    words = {w for w in re.findall(r"[\w#&'.+-]+", normalise(question)) if len(w) >= 3 or w.isdigit()}
    singular_words = words | {names.singular(w) for w in words}

    def written(value: str) -> bool:
        """Is the value one of the question's own words (or made of them)? A one-word name in the singular or the
        plural too ("bike" for "Bikes"); a longer one only as written ("stores" is no word of "Old Town Store")."""
        own = [w for w in re.findall(r"[\w#&'.+-]+", normalise(value)) if len(w) >= 3 or w.isdigit()]
        return any(w in words or len(own) == 1 and names.singular(w) in singular_words for w in own)

    def quoted(value: str) -> bool:
        v = normalise(value)
        if _points(value):
            return True      # "lowest", "that division": no name at all; the reader is asked which one they mean
        return v in grounded or any(v == q or (len(q) >= 3 and (q in v or v in q)) for q in said)

    dropped: list[tuple[str, str]] = []
    no_such: list[tuple[str, str]] = []     # words that describe, which no member of the data is called
    # The rows a table leaves out by default ("Invoice status is CANCELLED"): a filter that asks for them is the
    # question's point ("revenue on cancelled lines"), never a stray word of it.
    left_out = {normalise(str(v)) for t in model.tables.values() for df in t.default_filters
                for v in [*df.values, *(getattr(df, "shown", None) or [])]}

    def kept_of(listed: list) -> list:
        filters = []
        for f in listed:
            found = find_slug(model, f.field) if f.op in ("eq", "in", "ne", "not_in") else None
            if found is None or not isinstance(found[1], (Attribute, Entity)):
                filters.append(f)
                continue
            kept = values_kept(f, found[1])
            if kept:
                filters.append(f.model_copy(update={"values": kept}))
        return filters

    def values_kept(f, found: Attribute | Entity) -> list:
        thing = found.business_name.lower()
        attribute = found if isinstance(found, Attribute) else next(
            (a for a in model.attributes.values()
             if a.column == (found.label_column or found.code_column)), None)
        read = attribute is not None and attribute.slug in index.attributes

        def unknown(value: str) -> bool:      # a member the data does not have: said, never answered for all
            return read and index.stored(attribute.slug, value) is None   # type: ignore[union-attr]

        def described(value: str) -> bool:
            """Written by the reader in lower case, unquoted, right before the word for the things it describes: it
            says what kind ("which referral partner", "the loyal customers"), never a name. A name written with its
            capital ("Wholsale") is asked about when the data has none, and so is a word that is itself what is
            counted ("how many directors in each department": no level is called so)."""
            low = question.lower()
            at = low.find(value.lower())
            if at < 0 or question[at:at + len(value)] != value.lower() or quoted(value):
                return False
            entity = found if isinstance(found, Entity) else \
                model.entities.get(attribute.entity or "") if attribute is not None else None
            after = re.match(r"\s+([a-zà-ÿ]+)", low[at + len(value):])
            nouns = {names.singular(w) for w in re.findall(r"[a-zà-ÿ]+", (entity.business_name or "").lower())} \
                if entity is not None else set()
            return after is not None and names.singular(after.group(1)) in nouns

        amount = attribute is not None and attribute.kind == "number"    # "exactly 3 days": a value, no name
        if read:
            # "Bike" for the category stored as "Bikes": the member it names, then read as any name is.
            f = f.model_copy(update={"values": [_member_form(index, attribute.slug, v) for v in f.values]})  # type: ignore[union-attr]
        kept = [v for v in f.values
                if not isinstance(v, str) or amount or quoted(v) or unknown(v) and not described(v) or not written(v)
                or f.op in ("eq", "in") and normalise(v) in left_out]
        dropped.extend((thing, v) for v in f.values if v not in kept and not unknown(v))
        no_such.extend((thing, v) for v in f.values if v not in kept and unknown(v))
        return kept

    filters = kept_of(plan.filters)
    # "Customers who never bought 'Apples'": the conditions on who had none follow the same rule.
    absent = plan.absent.model_copy(update={"without": kept_of(plan.absent.without),
                                            "had": kept_of(plan.absent.had)}) if plan.absent is not None else None
    if not dropped and filters == plan.filters and (absent is None or absent == plan.absent):
        return plan, [], []
    return plan.model_copy(update={"filters": filters, "absent": absent}), dropped, no_such


def _hiding(masked: Masked | None) -> Callable[[str], str]:
    """The reader's words with every member name the AI was not shown back in its placeholder ("⟨v1⟩")."""
    pairs = sorted(((str(v), ph) for ph, v in (masked.values if masked is not None else {}).items()),
                   key=lambda p: -len(p[0]))

    def hide(words: str) -> str:
        for value, ph in pairs:
            if value:
                words = re.sub(re.escape(value), ph, words, flags=re.IGNORECASE)
        return words
    return hide


def _named_unquoted(payload: dict[str, Any], plan: Plan, model: SemanticModel,
                    unquoted: list[ValueMatch]) -> list[tuple[str, Plan]]:
    """A member the reader named without quotes, beyond doubt ("customers who bought Dish Soap"), narrows the
    answer: said on it, with one click to take it off ('Not only "Dish Soap"': the same plan without it)."""
    out: list[tuple[str, Plan]] = []
    notes = payload.setdefault("trust", {}).setdefault("date_context", [])
    for m in unquoted:
        key = normalise(str(m.value))
        if not any(normalise(str(v)) == key for f in _held_conditions(plan) for v in f.values):
            continue                # not applied: said with what the answer left out (core2/agent/parts.py)
        attribute = model.attributes.get(m.attribute)
        thing = (attribute.business_name if attribute is not None else m.attribute).lower()
        notes.append(f'Narrowed to {thing} "{m.value}", named in your question.')
        out.append((f'Not only "{m.value}"', _without_member(plan, key)))
    return out


def _held_conditions(plan: Plan) -> list[Filter]:
    """The conditions of a plan that hold members: its filters, who had or had none, a rate's condition."""
    out = list(plan.filters)
    if plan.absent is not None:
        out += [*plan.absent.without, *plan.absent.had]
    return out + [w for r in plan.rates for w in r.where]


def _without_member(plan: Plan, key: str) -> Plan:
    """The plan with no condition on the member (its value taken off a list of several)."""
    def cut(filters: list[Filter]) -> list[Filter]:
        out = []
        for f in filters:
            values = [v for v in f.values if normalise(str(v)) != key]
            if len(values) == len(f.values):
                out.append(f)
            elif values:
                out.append(f.model_copy(update={"values": values, "op": "eq" if len(values) == 1 and f.op == "in"
                                                else f.op}))
        return out

    # A rate whose only condition was the member is no rate any more: it goes with it.
    update: dict[str, Any] = {"filters": cut(plan.filters), "follow_up": "refine",
                              "rates": [r.model_copy(update={"where": cut(r.where)}) for r in plan.rates
                                        if cut(r.where)]}
    if plan.absent is not None:
        update["absent"] = plan.absent.model_copy(update={"without": cut(plan.absent.without),
                                                          "had": cut(plan.absent.had)})
    return plan.model_copy(update=update)


def _say_unquoted(payload: dict[str, Any], question: str, dropped: list[tuple[str, str]],
                  written: list[ValueMatch] | None = None, *, no_such: list[tuple[str, str]] = ()) -> None:
    """Said on the answer: which names did not narrow it, and the question again with them in quotes.

    A name dropped from inside a longer member name the question wrote ("Northfield" of "Northfield Industrial and
    Eastgate Industrial", read by the AI as a region and a division) is offered as the whole names written, all of
    that field's at once: quoting "Northfield" alone asked for the region again."""
    notes = payload.setdefault("trust", {}).setdefault("date_context", [])
    # The names it was not narrowed to: a summary naming one would claim what the answer does not show.
    payload["not_narrowed"] = [*(payload.get("not_narrowed") or []), *(str(v) for _, v in [*dropped, *no_such])]
    for thing, value in no_such:
        notes.append(f'No {thing} is called "{value}": the answer is not narrowed by it.')
    chips = []
    whole: list[ValueMatch] = []
    for thing, value in dropped:
        notes.append(f'Not narrowed to {thing} "{value}": a name narrows the answer only when it is in quotes.')
        at = question.lower().find(str(value).lower())
        inside = next((m for m in written or [] if m.start <= at and at + len(str(value)) <= m.end
                       and m.end - m.start > len(str(value))), None) if at >= 0 else None
        if inside is not None:
            whole += [m for m in written or [] if m.attribute == inside.attribute and m not in whole]
        elif at >= 0:
            again = f'{question[:at]}"{question[at:at + len(value)]}"{question[at + len(value):]}'
            chips.append({"label": f'Only "{value}"', "question": again})
    if whole:
        spans = sorted({(m.start, m.end) for m in whole})
        again = question
        for start, end in reversed(spans):
            again = f'{again[:start]}"{again[start:end]}"{again[end:]}'
        names_ = ", ".join(f'"{question[s:e]}"' for s, e in spans[:3])
        chips.insert(0, {"label": f"Only {names_}", "question": again})
    singles = [c for c in chips if c["label"].count('"') == 2]
    if len(singles) > 1:
        # Two names left unquoted ("Starter plans in Canada"): one click quotes them all, as the reader meant.
        again = question
        for _thing, value in sorted(dropped, key=lambda d: -question.lower().find(str(d[1]).lower())):
            at = again.lower().find(str(value).lower())
            if at >= 0 and (at == 0 or again[at - 1] != '"'):
                again = f'{again[:at]}"{again[at:at + len(value)]}"{again[at + len(value):]}'
        listed = " and ".join(c["label"].removeprefix("Only ") for c in singles[:3])
        chips.insert(0, {"label": f"Only {listed}", "question": again})
    if chips:
        payload["follow_up_suggestions"] = (chips + list(payload.get("follow_up_suggestions") or []))[:4]


def _visible(matches: list[ValueMatch], model: SemanticModel, allowed: set[str] | None) -> list[ValueMatch]:
    """Member names only from tables the reader may use."""
    if allowed is None:
        return matches
    return [m for m in matches
            if m.attribute in model.attributes and model.columns[model.attributes[m.attribute].column].table in allowed]


# ── the portal's wiring ────────────────────────────────────────────────────

MEMBERS_HOURS = 12.0          # member names are read again, in the background, after this long


@dataclass
class _Members:
    index: MemberIndex
    listable: frozenset[str]      # the attributes that could be listed when it was read
    read_at: float
    refreshing: bool = False


_INDEXES: OrderedDict[tuple[str, int], _Members] = OrderedDict()
_SESSIONS: OrderedDict[str, Session] = OrderedDict()
_LOCK = threading.Lock()


def forget_session(key: str) -> None:
    """A conversation no one goes on with (the answers check's questions): out of memory at once."""
    with _LOCK:
        _SESSIONS.pop(key, None)


def _session(key: str) -> Session:
    with _LOCK:
        session = _SESSIONS.pop(key, None) or Session()
        _SESSIONS[key] = session
        while len(_SESSIONS) > 2000:
            _SESSIONS.popitem(last=False)
        return session


def _member_index(account_id: str, model: SemanticModel, db_config: dict[str, Any], *,
                  read: bool = True) -> MemberIndex:
    """Members of the attributes QueryBot may list, read under the service connection.

    Read once per model version and kept, then read again in the background every
    MEMBERS_HOURS, so customers and items added since are found by name. An attribute
    the admin hides or marks sensitive is left out at once; one newly allowed is read at
    once. ``read`` False (the admin turned value indexing off) reads nothing.
    """
    if not read:
        return MemberIndex()
    key = (account_id, model.version)
    may = frozenset(listable(model))
    stale = False
    with _LOCK:
        kept = _INDEXES.get(key)
        if kept is not None:
            _INDEXES.move_to_end(key)
            stale = not kept.refreshing and time.monotonic() - kept.read_at > MEMBERS_HOURS * 3600
            kept.refreshing = kept.refreshing or stale
    if kept is None or not may <= kept.listable:
        kept = _read_members(key, model, db_config, may)
    elif stale:
        threading.Thread(target=_read_again, args=(key, model, db_config, may), daemon=True,
                         name=f"core2-members-{account_id}").start()
    return (kept.index if may == kept.listable else kept.index.only(may)).named(model)


def _read_members(key: tuple[str, int], model: SemanticModel, db_config: dict[str, Any],
                  may: frozenset[str]) -> _Members:
    from sqlglot import exp

    from core2.warehouse import dialect as D
    from core2.warehouse.querybot import QueryBotWarehouse

    def fetch(slug: str) -> list[object]:
        from core2.compile.compiler import column_sql

        column = model.columns[model.attributes[slug].column]
        table = model.tables[column.table]
        c = column_sql(model, column.key, None, warehouse.dialect)
        sql = exp.select(c).distinct().from_(D.table_expr(table.database, table.schema_name, table.name,
                                                          warehouse.dialect)).where(
            exp.not_(exp.Is(this=c.copy(), expression=exp.Null()))).sql(dialect=warehouse.dialect)
        return [r[0] for r in warehouse.query(sql, max_rows=50_001).rows]

    with QueryBotWarehouse(str(db_config.get("db_type") or ""), db_config["credentials"]) as warehouse:
        kept = _Members(build_index(model, fetch, budget_seconds=30.0), may, time.monotonic())
    with _LOCK:
        _INDEXES[key] = kept
        _INDEXES.move_to_end(key)
        while len(_INDEXES) > 20:
            _INDEXES.popitem(last=False)
    return kept


def _read_again(key: tuple[str, int], model: SemanticModel, db_config: dict[str, Any],
                may: frozenset[str]) -> None:
    try:
        _read_members(key, model, db_config, may)
    except Exception as exc:  # noqa: BLE001 - the names already read stand, and are tried again after the same wait
        log.warning("core2: member names of %s could not be read again: %s", key[0], exc)
        with _LOCK:
            kept = _INDEXES.get(key)
            if kept is not None:
                kept.refreshing, kept.read_at = False, time.monotonic()


def read_members(account_id: str) -> int:
    """Read the workspace's member names now (at the end of Learn, and on the admin's Refresh values).

    Returns how many names were read; 0 when the workspace has no model, or its admin
    turned value indexing off.
    """
    import store
    from core.value_index import value_index_enabled
    from core2.bootstrap.service import answering_model

    client = store.get_client(account_id) or {}
    db_id = int(client.get("db_config_id") or 0)
    db_config = store.get_db_config(db_id) if db_id else None
    model = answering_model(account_id, client) if db_config else None
    if model is None or db_config is None or not value_index_enabled(store.get_client_state(account_id)):
        return 0
    return len(_read_members((account_id, model.version), model, db_config, frozenset(listable(model))).index.names)


def _allowed_model_tables(model: SemanticModel, allowed: set[str] | None) -> set[str] | None:
    """The reader's allowed tables (as QueryBot names them) as model table keys."""
    if allowed is None:
        return None
    wanted = {ids.norm(name) for name in allowed}
    out = set()
    for key, table in model.tables.items():
        names = {ids.norm(".".join(p for p in parts if p)) for parts in (
            (table.database, table.schema_name, table.name), (table.schema_name, table.name), (table.name,))}
        if names & wanted:
            out.add(key)
    return out


def question_scrubber(account_id: str, model: SemanticModel | None = None) -> Callable[[str], str] | None:
    """Personal data out of a question, for a tenant under compliance; None for one that is not.

    The agent runtime's rule (core/agent_runtime.py): regulated, unprovisioned, or any
    mode but standard. It may widen, never narrow. The same tenants keep member values
    from the AI. Emails, phone numbers and social security numbers; then a name-shaped
    phrase that is not the workspace's own words (``model``, the answering one when not
    given) and, in every industry, what the name recogniser (Presidio) takes for a person
    where it is installed: both read the question as typed, and what either finds is
    withheld (core/masking.py::scrub_names). Both keep the workspace's own names: its
    members, places and words.
    """
    import store
    from core.compliance.policy_engine import under_compliance

    if not under_compliance(account_id):
        return None
    from core.masking import scrub_names, scrub_question_pii

    industry = str((store.get_compliance_profile(account_id) or {}).get("industry") or "")
    read: list[tuple[frozenset[str], frozenset[str]]] = []

    def vocabulary() -> tuple[frozenset[str], frozenset[str]]:
        """The workspace's own words and member names, read the first time a question is scrubbed (a caller that
        only asks whether the workspace is under compliance reads no model)."""
        if not read:
            given = model
            if given is None:
                from core2.bootstrap.service import answering_model

                try:
                    given = answering_model(account_id, store.get_client(account_id) or {})
                except Exception as exc:  # noqa: BLE001 - names are still scrubbed, with fewer words known to be safe
                    log.warning("core2: the scrubber of %s reads no model, so knows fewer of its words: %s",
                                account_id, exc)
            read.append(_vocabulary(given) if given is not None else (frozenset(), frozenset()))
        return read[0]

    def scrub(text: str) -> str:
        said = scrub_question_pii(text, industry, names=False)[0]
        known, phrases = vocabulary()
        return scrub_names(said, known, phrases)

    return scrub


_VOCABULARIES: OrderedDict[int, tuple[SemanticModel, tuple[frozenset[str], frozenset[str]]]] = OrderedDict()
_VOCABULARIES_LOCK = threading.Lock()


def _vocabulary(model: SemanticModel) -> tuple[frozenset[str], frozenset[str]]:
    """The workspace's own words, in lower case (what its tables, fields, metrics and dates are called), and the whole
    names of the members it may list (never a person's), kept per model: each question is scrubbed several times."""
    with _VOCABULARIES_LOCK:
        kept = _VOCABULARIES.get(id(model))
        if kept is not None and kept[0] is model:
            _VOCABULARIES.move_to_end(id(model))
            return kept[1]
    words: set[str] = set()
    phrases: set[str] = set()

    def add(*texts: object) -> None:
        for text in texts:
            words.update(w.casefold() for w in re.findall(r"[^\W\d_][\w'’\-]*", str(text or "")))

    for t in model.tables.values():
        add(t.name, t.business_name)
    for c in model.columns.values():
        add(c.name, c.business_name, *[w for ws in c.synonyms.values() for w in ws])
        if not people_column(c) and c.values_allowed and c.profile and c.profile.top:
            for v in c.profile.top:
                if isinstance(v.value, str):
                    # The member's whole name, and each run of two or more of its words ("Northline Distribution"
                    # of "... 58"). Never one word cut from it: "Spencer" alone is a person's name though a
                    # supplier is called Marks & Spencer.
                    runs = [v.value, *(r for r in re.split(r"[^\w\s'’\-]|\d+", v.value) if len(r.split()) >= 2)]
                    phrases.update(" ".join(w.casefold() for w in run.split()) for run in runs if run.strip())
    for m in model.measures.values():
        add(m.slug, m.business_name, *[w for ws in m.synonyms.values() for w in ws])
    for e in model.entities.values():
        add(e.business_name)
    for r in model.date_roles.values():
        add(r.name)
    out = (frozenset(words), frozenset(phrases))
    with _VOCABULARIES_LOCK:
        _VOCABULARIES[id(model)] = (model, out)
        while len(_VOCABULARIES) > 16:
            _VOCABULARIES.popitem(last=False)
    return out


def personal_columns(model: SemanticModel) -> dict[tuple[str, str], str]:
    """People's data the model knows of: (table, column) as the warehouse spells them, casefolded -> "name" or
    "detail". The governed warehouse masks what comes from them for a reader not cleared to see it."""
    out = {}
    for column in model.columns.values():
        table = model.tables.get(column.table)
        if column.personal != "none" and not column.parts and table is not None:
            out[(table.name.casefold(), column.name.casefold())] = column.personal
    return out


# What a reader is told on a workspace Learn has not read yet (an admin runs it on the Setup page). The old engine
# never answers instead: the new core is the only one.
NOT_LEARNED = ("This workspace is still being set up: Solviz has not learned its database yet. An admin can finish "
               "its setup, and questions are answered from then on.")


def _portal_services(account_id: str, question: str, portal_user: dict[str, Any] | None,
                     question_id: str = "") -> tuple[Services | None, dict[str, Any] | None]:
    """The services a portal answer runs on, or the frame that says why there are none."""
    import store
    from core.schema import load_known_tables
    from core.value_index import value_index_enabled
    from core2.bootstrap.ai import beyond_writer, metric_writer, workspace_planner
    from core2.bootstrap.service import answering_model, keep_decisions_current
    from core2.kpi_proposals import industry_of, waiting_metrics
    from core2.warehouse.governed import GovernedWarehouse

    client = store.get_client(account_id) or {}
    db_id = int(client.get("db_config_id") or 0)
    db_config = store.get_db_config(db_id) if db_id else None
    if not db_config:
        return None, _frame(question, "This workspace is still being set up: it has no database connected yet. "
                                      "An admin can finish its setup.")
    keep_decisions_current(account_id, db_id, client)
    model = answering_model(account_id, client)
    if model is None:
        return None, _frame(question, NOT_LEARNED)
    state = store.get_client_state(account_id) or {}
    allowed = store.get_allowed_tables(portal_user) if portal_user else None
    scrub = question_scrubber(account_id, model=model)
    warehouse = GovernedWarehouse(account_id, portal_user, db_config,
                                  known_tables=load_known_tables(state.get("schema_dir", "")),
                                  allowed_tables=allowed, personal=personal_columns(model) if scrub else {})
    # The admin's "value indexing" switch, as today's pipeline reads it: off, no member
    # name is read from the warehouse or put before the AI.
    indexing = value_index_enabled(state)
    try:
        index = _member_index(account_id, model, db_config, read=indexing)
    except Exception as exc:  # noqa: BLE001 - names are a help to the planner; the question is answered without them
        if unavailable(exc):
            log.warning("core2: the database of %s was not available to read member names: %s", account_id, exc)
            return None, _frame(question, UNAVAILABLE)
        log.warning("core2: member names of %s could not be read; answering without them: %s", account_id, exc,
                    exc_info=True)
        index = MemberIndex()
    tables = _allowed_model_tables(model, allowed)
    return Services(
        model=model, warehouse=warehouse,
        # The AI call log keeps the question as the AI was given it: scrubbed, where the workspace scrubs.
        complete=workspace_planner(account_id, client, question=scrub(question) if scrub else question,
                                   question_id=question_id),
        index=index, today=dt.date.today(),
        values_allowed=scrub is None and indexing, allowed_tables=tables,
        data_source=str(db_config.get("db_type") or ""), scrub=scrub,
        # A reader's own metric is written by the AI from their words, unscrubbed and beside the fields' values:
        # offered only where the question itself may reach the AI as typed.
        write_metric=(lambda words: metric_writer(account_id, client, description=words))
        if scrub is None and indexing else None, industry=industry_of(account_id),
        waiting=lambda: waiting_metrics(account_id, tables), question_map=_question_map(account_id, client),
        general_knowledge=general_knowledge_enabled(state),
        beyond=beyond_writer(account_id, client, question=scrub(question) if scrub else question,
                             question_id=question_id)), None


def general_knowledge_enabled(state: dict[str, Any] | None) -> bool:
    """The admin's switch for answers from the AI's general knowledge (Settings): on unless turned off."""
    flag = (state or {}).get("general_knowledge_enabled") if isinstance(state, dict) else None
    return flag is None or (bool(flag) and str(flag).lower() not in ("0", "false", "no", "off"))


def _question_map(account_id: str, client: dict[str, Any]):
    """The workspace's question map when it was built on the model as it is now; None while it is rebuilt."""
    try:
        from core2.question_map import current

        return current(account_id, client)
    except Exception as exc:  # noqa: BLE001 - the question is answered without it
        log.warning("core2: the question map of %s could not be read: %s", account_id, exc)
        return None


def portal_summary(account_id: str, question: str, payload: dict[str, Any], *, question_id: str = "") -> str | None:
    """A short written summary of a portal answer by the workspace's AI (core2/answer/summary.py), or None.

    What may reach the AI is decided as for the question: where member values are kept from it
    (a tenant under compliance, or value indexing off), an answer that names members gets none,
    and the question goes with its personal data scrubbed.
    """
    import store
    from core.value_index import value_index_enabled
    from core2.answer.summary import eligible, write
    from core2.bootstrap.ai import summary_writer

    client = store.get_client(account_id) or {}
    scrub = question_scrubber(account_id)
    values_allowed = scrub is None and value_index_enabled(store.get_client_state(account_id) or {})
    if not eligible(payload, values_allowed=values_allowed):
        return None
    asked = scrub(question) if scrub else question
    if not _answers_may_reach_ai(account_id, asked, "core2_summary"):
        return None
    complete = summary_writer(account_id, client, question=asked, question_id=question_id)
    return write(payload, asked, complete)


def portal_overview(account_id: str, question: str, payload: dict[str, Any], *, question_id: str = "",
                    lang: str = "en") -> str | None:
    """The opening paragraph of an answer about the data itself, by the workspace's AI (core2/answer/overview.py),
    or None. The cards hold the data's spans and the values its tables leave out, so it is written only where member
    values may reach the AI, and only where the workspace's answers may."""
    import store
    from core.value_index import value_index_enabled
    from core2.answer.overview import eligible, write
    from core2.bootstrap.ai import summary_writer

    client = store.get_client(account_id) or {}
    scrub = question_scrubber(account_id)
    values_allowed = scrub is None and value_index_enabled(store.get_client_state(account_id) or {})
    if not eligible(payload, values_allowed=values_allowed):
        return None
    if not _answers_may_reach_ai(account_id, question, "core2_summary"):
        return None
    complete = summary_writer(account_id, client, question=question, question_id=question_id)
    return write(payload, question, complete, lang=lang)


def portal_note(account_id: str, question: str, payload: dict[str, Any], *, question_id: str = "",
                lang: str = "en") -> dict[str, Any] | None:
    """The analyst's note of a portal answer (core2/answer/notes.py), or None when the answer has no facts.

    The workspace's AI writes it from the facts alone where it may see them: never where the rows name people,
    and where member values are kept from it (a tenant under compliance, or value indexing off) only when the
    facts name no member. Otherwise the note is the facts' own sentences."""
    import store
    from core.value_index import value_index_enabled
    from core2.answer.notes import names_members, write
    from core2.bootstrap.ai import summary_writer

    reading = payload.get("reading")
    if not isinstance(reading, dict) or not reading.get("facts") or payload.get("engine") != "core2":
        return None
    client = store.get_client(account_id) or {}
    scrub = question_scrubber(account_id)
    values_allowed = scrub is None and value_index_enabled(store.get_client_state(account_id) or {})
    personal = bool(payload.get("personal"))
    asked = scrub(question) if scrub else question
    # A regulated workspace's answers never reach the AI: the note is the facts' own sentences. Its refusal is
    # recorded only for a note the AI would have been asked to write.
    would_ask = not personal and (values_allowed or not names_members(reading))
    complete = summary_writer(account_id, client, question=asked, question_id=question_id) \
        if would_ask and _answers_may_reach_ai(account_id, asked, "core2_summary") else None
    return write(reading, complete, lang=lang, values_allowed=values_allowed, personal=personal)


def portal_reply(account_id: str, question: str, payload: dict[str, Any], *, question_id: str = "",
                 lang: str = "en") -> str | None:
    """The answer's reply in the chat, in the workspace's AI's words (core2/agent/talk.py), or None: the facts' own
    sentences stand.

    The AI is given what the checks found and the answer's sentence, never the rows, and only where it may see them:
    never where the answer names people, and where member values are kept from it (a tenant under compliance, or
    value indexing off) only when neither the reply nor the answer names a member; never for a workspace whose
    answers may not reach the AI."""
    import store
    from core.value_index import value_index_enabled
    from core2.agent.talk import names_members, write
    from core2.answer.summary import names_members as answer_names_members
    from core2.bootstrap.ai import summary_writer

    reply = payload.get("reply")
    if not isinstance(reply, dict) or not reply.get("seen") or payload.get("engine") != "core2" \
            or payload.get("personal"):
        return None
    client = store.get_client(account_id) or {}
    scrub = question_scrubber(account_id)
    values_allowed = scrub is None and value_index_enabled(store.get_client_state(account_id) or {})
    # The answer's sentence goes with the reply: it names the members its conditions name, as its rows name theirs.
    if not values_allowed and (names_members(reply) or answer_names_members(payload)):
        return None
    asked = scrub(question) if scrub else question
    if not _answers_may_reach_ai(account_id, asked, "core2_summary"):
        return None
    complete = summary_writer(account_id, client, question=asked, question_id=question_id)
    headline = str((payload.get("answer") or {}).get("headline") or "")
    return write(reply, asked, headline, complete, lang=lang)


def _answers_may_reach_ai(account_id: str, question: str, component: str) -> bool:
    """May an answer (its rows, its figures, its headline) be sent to the workspace's AI? Never for a regulated
    workspace (core/compliance/policy_engine.py::result_llm_features_allowed): the AI writes the plan and never sees
    an answer. A refusal is recorded on the AI egress log, as the old engine's were."""
    from core.compliance.policy_engine import result_llm_features_allowed
    from core.llm_audit import llm_audit_scope, record_llm_blocked

    try:
        allowed = result_llm_features_allowed(account_id)
    except Exception as exc:  # noqa: BLE001 - refused when the policy cannot be read
        log.warning("core2: could not read whether %s may send answers to the AI; it does not: %s", account_id, exc)
        allowed = False
    if not allowed:
        with llm_audit_scope(account_id=account_id, question=question, enabled=True, component=component):
            record_llm_blocked(component, "A regulated workspace's answers never reach the AI: the summary is left "
                                          "out and the analyst note is written from the facts alone.")
    return allowed


def portal_answer(account_id: str, question: str, portal_user: dict[str, Any] | None, *, session_key: str,
                  question_id: str = "", progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """The new core's answer in the web portal (compare mode, or core2 mode). ``progress`` is shown a described
    dashboard's card as its tiles land."""
    from core.portal_roles import can, refusal
    from core2 import dashboards

    if dashboards.asked_for(question) is not None and not can(portal_user, "create_dashboard"):
        # A dashboard made from a request is a creator's (core/portal_roles.py): refused before any AI.
        return _frame(question, refusal("create_dashboard", str((portal_user or {}).get("lang") or "en")))
    services, refused = _portal_services(account_id, question, portal_user, question_id)
    if services is None:
        return refused or _frame(question, "The new core cannot answer here yet.")
    if progress is not None:
        services = replace(services, progress=progress)
    services = _with_profile_standing(services, account_id, portal_user)
    session = _session(session_key)
    payload = answer_question(question, services, session, question_id=question_id,
                              planned=_card_asked(account_id, portal_user, question, session))
    if getattr(services.warehouse, "released", False):
        # People's data went out as stored, to a cleared reader: kept with the answer, so a reopened thread
        # never shows it again once the clearance ends (core/compliance/kept_answers.py).
        payload["released"] = True
    fix = (payload.get("dates") or {}).get("fix")
    if fix:
        # An admin's click makes it the metric's date at once; a reader's is a suggestion (core2/request_service.py).
        fix["admin"] = str((portal_user or {}).get("role") or "") == "admin"
    own = payload.get("own_metric")
    if own and isinstance(own.get("measure"), Measure):
        # The definition stays on the server; the reader's buttons name it by a token only they can use.
        own["token"] = _own_token(account_id, str((portal_user or {}).get("id") or ""), session_key,
                                  own.pop("measure"), own.get("words", ""), question_id)
    for figure in payload.get("figures") or []:
        # A figure worked out for the question: its definition stays here too, and "Ask my admin to save it"
        # names it by a token, as a reader's own metric is (core2/request_service.py re-keys it on acceptance).
        if isinstance(figure.get("measure"), Measure):
            figure["token"] = _own_token(account_id, str((portal_user or {}).get("id") or ""), session_key,
                                         figure.pop("measure"), figure.get("formula", ""), question_id)
    return payload


def _with_profile_standing(services: Services, account_id: str, portal_user: dict[str, Any] | None) -> Services:
    """The reader's own standing conditions ("always use the last 2 months"), read from their profile at each
    question, so one removed on the Profile page is gone from the next answer; and how one is kept or removed."""
    user_id = (portal_user or {}).get("id")
    if not user_id:
        return services
    from store import profile_store as P

    return replace(services, always=_profile_always(account_id, portal_user),
                   keep_always=lambda s: P.keep_standing(int(user_id), account_id, s.data()),
                   drop_always=lambda standing_id: P.drop_standing(int(user_id), account_id, standing_id))


def _profile_always(account_id: str, portal_user: dict[str, Any] | None) -> list[Standing]:
    user_id = (portal_user or {}).get("id")
    if not user_id:
        return []
    from store import profile_store as P

    try:
        return [Standing.of(d) for d in P.standing_conditions(int(user_id), account_id)]
    except Exception as exc:  # noqa: BLE001 - the question is answered without them, and it is logged
        log.warning("core2: standing conditions of reader %s could not be read: %s", user_id, exc)
        return []


def standing_in(account_id: str, portal_user: dict[str, Any] | None, session_key: str) -> list[dict[str, Any]]:
    """The standing conditions in force in a conversation, for the chip row above the box (a reopened thread,
    a new one): the reader's own, unless taken off here, then the conversation's."""
    always = _profile_always(account_id, portal_user)
    with _LOCK:
        session = _SESSIONS.get(session_key) or Session()
        return [s.shown() for s in _in_force(always, session)]


def standing_taken_off(account_id: str, portal_user: dict[str, Any] | None, session_key: str,
                       standing_id: str) -> list[dict[str, Any]]:
    """The chip's ×: a condition taken off this conversation (one of the reader's own, for this conversation only;
    the Profile page removes it for good). The conditions still in force."""
    session = _session(session_key)
    with _LOCK:
        session.standing = [k for k in session.standing if k.id != standing_id]
        session.paused.add(str(standing_id))
    return standing_in(account_id, portal_user, session_key)


def _card_asked(account_id: str, portal_user: dict[str, Any] | None, question: str, session: Session) -> Plan | None:
    """The plan of a start-page card asked as a new conversation's first question (core2/suggest.py): the card
    was checked for this reader with it, so the answer is the one checked, with no AI. Any other question, and a
    card's words typed later in a conversation (read with what came before), is planned as usual."""
    if session.turns or session.pending is not None or not portal_user or not portal_user.get("id"):
        return None
    try:
        from core2.suggest import card_plan

        found = card_plan(account_id, str(portal_user["id"]), question)
        return parse_plan({"kind": "query", **found[0]}) if found else None
    except Exception as exc:  # noqa: BLE001 - planned as any question
        log.warning("core2: a start-page card could not be read for %s: %s", account_id, exc)
        return None


# A reader's metric made in a chat, by token: "Ask my admin to save it for everyone" and "Don't keep it".
_OWN_TOKENS: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _own_token(account_id: str, user_id: str, session_key: str, measure: Measure, words: str,
               question_id: str) -> str:
    import secrets

    token = secrets.token_urlsafe(18)
    with _LOCK:
        _OWN_TOKENS[token] = {"account_id": account_id, "user_id": user_id, "session_key": session_key,
                              "measure": measure, "words": words, "question_id": question_id}
        while len(_OWN_TOKENS) > 2000:
            _OWN_TOKENS.popitem(last=False)
    return token


def own_metric(token: str, account_id: str, user_id: str) -> dict[str, Any] | None:
    """The reader's own metric a token names, if it is theirs."""
    with _LOCK:
        found = _OWN_TOKENS.get(str(token or ""))
    if found is None or found["account_id"] != account_id or found["user_id"] != str(user_id):
        return None
    return found


def forget_own_metric(token: str, account_id: str, user_id: str) -> bool:
    """Take a reader's metric out of the chat it was made in ("Don't keep it")."""
    found = own_metric(token, account_id, user_id)
    if found is None:
        return False
    with _LOCK:
        session = _SESSIONS.get(found["session_key"])
        if session is not None:
            session.own.pop(found["measure"].key, None)
        _OWN_TOKENS.pop(token, None)
    return True


def portal_replay(account_id: str, plan_data: dict[str, Any], portal_user: dict[str, Any] | None, *,
                  question: str = "") -> dict[str, Any]:
    """A pinned answer drawn again: its stored plan run on today's data, as the viewer is allowed to see it.

    No AI is asked: the plan is the one the answer was given from, so a dashboard tile keeps the
    answer's own shape (a comparison stays a dumbbell, a total a number). A window written
    relative to today ("last month") moves on with it.
    """
    services, refused = _portal_services(account_id, question, portal_user)
    if services is None:
        return refused or _frame(question, "The new core cannot answer here yet.")
    plan = parse_plan(plan_data)
    ctx = Context(today=services.today, allowed_tables=services.allowed_tables, max_rows=MAX_ROWS,
                  personal_shown=services.scrub is not None, link_repeats=_link_check(services))
    payload = _compute(question, plan, services, ctx, question_id="", started=time.perf_counter())
    if getattr(services.warehouse, "released", False):
        payload["released"] = True       # as portal_answer: people's data went out as stored, to a cleared reader
    return payload


def parse_plan(data: dict[str, Any]) -> Plan:
    """A stored plan back as a Plan (for replaying recorded answers)."""
    return Plan.model_validate(data)
