"""Follow-up or new question: what the words of a turn say, before any AI is asked.

Most turns say plainly whether they continue the answer on screen. "Only North", "by week instead",
"and refunds", "the first one", "why did it drop?", "as a pie" change that answer; "What were refunds
by return reason last quarter?" names what it measures and for when, and asks afresh. Those are
decided here, the same way every time. A short complete question after a narrowed answer ("How many
orders?" after net sales for one region in March) could mean either: the reader is asked, with two
buttons. What is left is the AI's to read, with the answer before it in view.

Readings: ``refine`` (continues the answer on screen), ``new`` (a question of its own), ``unsure``
(ask which), ``open`` (the AI decides).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core2.model.schema import SemanticModel
from core2.plan.ir import Plan
from core2.plan.words import _SHAPE, _phrases, _same

NEW_PREFIX = re.compile(r"^\s*new question\s*[:\-–]\s*", re.IGNORECASE)

_LEADS = (r"and|also|plus|but|now|then|ok(?:ay)?|what about|how about|what if|only|just|except|excluding|exclude|"
          r"without|instead|same|versus|vs\.?|by|per|for each|in|for")
_CONTINUE = re.compile(
    r"^\s*(?:" + _LEADS + r"|compared?\s+(?:to|with|against)|split|break\s+(?:it|that|this|them|those)\s+down|"
    r"drill|sort(?:ed)?|ordered|ranked|order\s+(?:it|them|by)|rank\s+(?:it|them)|top\s+\d+|bottom\s+\d+|first\s+\d+|lowest\s+first|"
    r"highest\s+first|largest\s+first|smallest\s+first|as\s+an?\b|as\s+the\b|show\s+(?:it|that|this|them|those)\b)\b",
    re.IGNORECASE)
_LEADING = re.compile(r"^\s*(?:" + _LEADS + r")\b", re.IGNORECASE)
# Words that carry the answer on screen along: "Canada only", "last year instead", "and gross sales too".
_CARRY = re.compile(r"\b(?:only|instead|too|alongside|as well|next to it|with it|either|as an?\s+(?:percent(?:age)?|"
                    r"share|ratio|proportion|fraction)\s+of)\b", re.IGNORECASE)
_POINTING = re.compile(
    r"\b(?:it|its|that|those|them|these|this one|that one|the same|above|the first one|the second one|the third one|"
    r"the last one|the lowest one|the highest one|the top one|the bottom one|of those|of these|of them|the largest one|"
    r"the smallest one|the biggest one)\b", re.IGNORECASE)
# A turn that opens by pointing ("the first one, how did it do each month?") follows the answer, however long.
_POINTS_FIRST = re.compile(
    r"^\s*(?:the\s+(?:first|second|third|fourth|fifth|last|lowest|highest|largest|smallest|biggest|top|bottom)"
    r"(?:\s+one)?|that\s+\w+|those\s+\w+|these\s+\w+|it|they|them)\b", re.IGNORECASE)
# "those same customers", "how many of them": the members of the answer on screen, in a turn of any length.
_POINTS_AT_MEMBERS = re.compile(r"\b(?:those|these|them|the same(?!-))\b", re.IGNORECASE)
# "the stores that opened in 2025": a "that" that starts a description points at nothing on screen.
_RELATIVE = re.compile(r"\b(?:that|which|who)\s+(?:\w+ed|are|were|have|has|had|did|do|does|is|was|can|will|bought|"
                       r"sold|made|got|took|left|went|came|hold|holds|pay|pays|use|uses)\b", re.IGNORECASE)
_DISPLAY = re.compile(
    r"\b(?:as an?\s+(?:pie|bar|line|donut|doughnut|ring|table|chart|area|graph)|as a share|share of (?:the )?total|"
    r"as a percent(?:age)? of (?:the )?total|chart it|plot it|graph it|just the table)\b", re.IGNORECASE)
_WHY = re.compile(r"^\s*(?:why|what\s+(?:drove|caused|explains|is behind)|how come|what changed)\b", re.IGNORECASE)
# "Which items explain the change?", "what is behind this drop": about the change the answer on screen shows.
_BEHIND = re.compile(r"\b(?:explain|explains|explained|drove|drive|drives|caused|cause|causes|behind|account\s+for|"
                     r"accounts\s+for)\s+(?:the|this|that)\s+(?:change|increase|decrease|drop|rise|fall|growth|"
                     r"decline|difference|gap|jump|dip|move|movement)\b", re.IGNORECASE)
_ASKS = re.compile(r"^\s*(?:what|what's|whats|which|who|whom|how|when|where|show|list|give|tell|compare|is|are|do|does|"
                   r"did|can|could|find|get|count)\b", re.IGNORECASE)
_ABOUT_DATA = re.compile(r"^\s*(?:what\s+(?:data|information|tables)|what can i ask|what do you (?:have|know)|"
                         r"what(?:'s| is)\s+(?:in|available)|which tables|describe\b|help\b)", re.IGNORECASE)
_SUPERLATIVE = re.compile(r"\b(?:most|least|highest|lowest|biggest|smallest|largest|best|worst|top|bottom|cheapest|"
                          r"dearest|furthest|fastest|slowest|first|last)\b", re.IGNORECASE)
_WHEN = re.compile(
    r"\b(?:today|yesterday|tomorrow|this|last|next|previous|past|latest|current|ytd|mtd|qtd|so far|since|until|"
    r"between|(?:19|20)\d{2}|q[1-4]|h[12]|fy\s?\d{2,4}|january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|week|weekly|month|monthly|quarter|"
    r"quarterly|year|yearly|annual|daily|right now|currently|at the end of)\b", re.IGNORECASE)
_GROUPED = re.compile(r"\b(?:by|per|each|every|for each|across)\s+\w+", re.IGNORECASE)
_CONDITION = re.compile(r"\b(?:(?:less|more|fewer|greater|lower|higher)\s+than|at least|at most|over|under)\s+\$?\d",
                        re.IGNORECASE)
# "net sales for nowhere": a name the member index does not know is still a scope the question sets.
# "for North", "for \"North\"": a name the reader writes, in quotes or not.
_NAMES_ONE = re.compile(r"\b(?:for|at|from|in)\s+[\"\u201c\u00ab']?\s*"
                        r"(?!(?:the|a|an|each|every|all|total|this|our|my|it|them)\b)[a-z]\w*", re.IGNORECASE)
_WHICH = re.compile(r"^\s*(?:which|what)\s+(\w+(?:\s+\w+)?)", re.IGNORECASE)
_GENERIC_TAIL = {"amount", "amt", "quantity", "qty", "value", "val", "count", "cnt", "number", "total",
                 # units a learned name keeps and people leave out: "utilization pct", "data usage gb"
                 "pct", "percent", "gb", "mb", "tb", "kb", "kg", "hrs", "mbps", "usd"}


@dataclass(frozen=True)
class Reading:
    kind: str           # refine | new | unsure | open
    why: str


def _words(question: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9%][A-Za-z0-9%'&.-]*", question)


def _phrases_said(model: SemanticModel) -> list[tuple[str, str, list[str]]]:
    """(kind, slug, words) for every name, and the shorter ways people say it: "Number of claims" as "claims"
    (kind "count": the thing counted, which a question may name without counting it), "Budget amount" as "budget"."""
    out = []
    for kind, slug, _, phrase in _phrases(model):
        out.append((kind, slug, phrase))
        if kind == "measure" and len(phrase) >= 2 and phrase[0] in ("number", "count", "no", "num", "nbr"):
            out.append(("count", slug, phrase[1:]))
        elif kind == "measure" and len(phrase) >= 2 and phrase[-1] in _GENERIC_TAIL:
            shorter = list(phrase)
            while len(shorter) >= 2 and shorter[-1] in _GENERIC_TAIL:
                shorter.pop()
            out.append((kind, slug, shorter))
    return out


def _found(model: SemanticModel | None, question: str, kinds: tuple[str, ...]) -> list[tuple[str, str, list[str]]]:
    if model is None:
        return []
    tokens = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", question)]
    found = []
    for kind, slug, phrase in _phrases_said(model):
        if kind not in kinds:
            continue
        size = len(phrase)
        for i in range(len(tokens) - size + 1):
            run = tokens[i:i + size]
            # A word of a question's shape ("on", "by") is no part of a name, unless the name has it too ("on hand").
            if not any(w in _SHAPE and w not in phrase for w in run) \
                    and all(_same(a, b, size > 1) for a, b in zip(run, phrase)):
                found.append((kind, slug, run))
                break
    return found


def named(model: SemanticModel | None, question: str) -> set[str]:
    """The kinds of name the question uses (measure, field, date), by the model's own words and their abbreviations."""
    return {"measure" if kind == "count" else kind
            for kind, _, _ in _found(model, question, ("measure", "count", "field", "date"))}


def measures_named(model: SemanticModel | None, question: str) -> set[str]:
    """The measure slugs the question names (a thing it counts among them)."""
    return {slug for _, slug, _ in _found(model, question, ("measure", "count"))}


def scoped(question: str, members: int = 0) -> bool:
    """Does the question set its own scope: a period, a grouping, a condition on a number, or a member it names?"""
    return bool(_WHEN.search(question) or _GROUPED.search(question) or _CONDITION.search(question) or members)


def narrowed(plan: Plan | None) -> bool:
    """Did the answer on screen narrow anything a fresh question would not carry: a filter or a period?"""
    if plan is None:
        return False
    window = plan.time.window
    return bool(plan.filters or window.kind != "all" or plan.time.compare is not None)


def read_turn(question: str, *, previous: Plan | None, model: SemanticModel | None = None,
              members: list[str] | int = 0, previous_narrowed: bool | None = None) -> Reading:
    """How this turn relates to the answer on screen (``previous``, the plan behind it), by its words alone.

    ``members``: the member names of the data the question names (North, Riverside Hub), as written.
    ``previous_narrowed``: whether that answer was narrowed, when its plan is not at hand.
    """
    if NEW_PREFIX.match(question):
        return Reading("new", "asked as a new question")
    if _ABOUT_DATA.match(question):
        return Reading("new", "it asks about the data itself")
    if previous is None or previous.kind != "query":
        return Reading("new", "there is no answer on screen to follow")
    words = _words(question)
    short = len(words) <= 8
    pointing = _POINTING.search(_RELATIVE.sub(" ", question))
    if _DISPLAY.search(question):
        return Reading("refine", "it says how to show the answer on screen")
    if _WHY.match(question) and (short or pointing):
        return Reading("refine", "it asks why the answer on screen is what it is")
    if _BEHIND.search(question):
        return Reading("refine", "it asks what is behind the change on screen")
    if _POINTS_FIRST.match(question) or short and pointing:
        return Reading("refine", "it points at the answer on screen")
    if _POINTS_AT_MEMBERS.search(question):
        return Reading("refine", "it points at the members of the answer on screen")
    if short and _CARRY.search(question):
        return Reading("refine", "it carries the answer on screen along")
    found = _found(model, question, ("measure", "count", "field", "date"))
    kinds = {"measure" if kind == "count" else kind for kind, _, _ in found}
    said = {w for kind, _, run in found if kind == "measure" for w in run}
    # A member name that is a word of a measure it names ("paid" of "paid amount") narrows nothing.
    texts = members if isinstance(members, list) else []
    own_members = [m for m in texts if not set(re.findall(r"[a-z0-9]+", m.lower())) <= said] if texts else []
    own = scoped(question, len(own_members) if texts else int(members or 0))
    asks = bool(_ASKS.match(question))
    leading = bool(_LEADING.match(question))
    period = bool(_WHEN.search(question))
    grouped = bool(_GROUPED.search(question))
    # What the turn is about comes before its split ("tickets" in "tickets by category"); what follows "by" is
    # how to split it, never what is measured.
    head = re.split(r"\b(?:by|per|for|in|during|over)\b", question, maxsplit=1, flags=re.IGNORECASE)[0]
    head_found = _found(model, head, ("measure", "count", "field"))
    head_measure = any(k == "measure" for k, _, _ in head_found)      # a measure by name, not a thing counted
    head_members = [m for m in own_members if m.lower() in head.lower()]
    which = _WHICH.match(question)
    if asks and not scoped(question) and len(words) <= 6 and _SUPERLATIVE.search(question):
        # "which store was lowest?": the store is what is ranked, not what is measured
        rest = question[which.end():] if which else question
        if measures_named(model, rest) <= set(previous.measures):
            return Reading("refine", "it asks which of the answer on screen comes first or last")
    if not asks and head_members and not head_measure:
        return Reading("refine", f"it narrows the answer on screen to {head_members[0]}")
    if which and model is not None and _found(model, which.group(1), ("field",)):
        own = True      # "which warehouse type ..." ranks by warehouse type: its own grouping
    if not leading and len(words) >= 4 and period and (head_found or asks and grouped):
        return Reading("new", "it names what it measures, how to split it and for when")
    if _CONTINUE.match(question):
        return Reading("refine", "it starts by carrying on from the answer on screen")
    if not asks and not leading:
        if grouped and head_found:
            return Reading("new", "it names a measure or a thing and how to split it")
        if short and not head_measure and not any(k == "count" for k, _, _ in head_found):
            return Reading("refine", "a fragment that only changes the answer on screen")
    if (kinds & {"measure", "field"}) and not pointing:
        if own:
            return Reading("new", "it names what it measures and its own scope")
        was_narrowed = narrowed(previous) if previous_narrowed is None else previous_narrowed
        if was_narrowed and _NAMES_ONE.search(question):
            return Reading("open", "it names something the data does not hold: the AI reads it")
        if was_narrowed:
            return Reading("unsure", "a complete question after a narrowed answer: it may or may not keep that scope")
        return Reading("new", "it names what it measures, and the answer on screen narrowed nothing")
    return Reading("open", "left to the AI")
