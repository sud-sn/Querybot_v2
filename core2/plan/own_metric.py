"""A metric a reader defines in the chat, for that chat only.

"Margin after returns = net amount minus refunds, divided by net amount, completed orders
only. Show it by month this year." The reader names a metric and says how it is counted;
the workspace's AI writes it (core2.model.authoring.describe, the admin's Add a metric
path), it is read and checked like any metric an admin types, and it answers the rest of
the question. It is the reader's, in that chat: nobody else's answers change until they ask
their admin to save it for everyone and the admin accepts it (Admin -> Data -> Requests).

A definition is recognised by its shape only: "<name> = <how it is counted>", or "define
<name> as <how>"; a question that only asks what a metric is ("what is gross margin?") is
not one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from core2 import ids
from core2.model import authoring
from core2.model.schema import ColumnFilter, Measure, SemanticModel

OWN_PREFIX = "reader:"
MAX_NAME = 60
MAX_DEFINITION = 600

_DEFINE = [
    re.compile(r"^\s*(?:define|let)\s+(?P<name>[^=:.?!]{2,60}?)\s+(?:as|be)\s+(?P<body>.+)$", re.I | re.S),
    re.compile(r"^\s*(?:d[ée]finis(?:sez)?|soit)\s+(?P<name>[^=:.?!]{2,60}?)\s+(?:comme|=)\s+(?P<body>.+)$",
               re.I | re.S),
    re.compile(r"^\s*(?P<name>[A-Za-zÀ-ÿ][^=:.?!]{1,59}?)\s*:?=\s*(?P<body>.+)$", re.S),
]
# The sentence after a definition that asks for an answer with it: "Show it by month this year."
_ASK = re.compile(r"^(?:show|give|plot|chart|list|graph|compare|what|how|which|break|rank|top|and|now|then|"
                  r"montre|montrez|affiche|affichez|donne|donnez|quel|quelle|comment)\b", re.I)
_QUESTION_WORDS = {"what", "how", "which", "why", "when", "where", "who", "is", "are", "does", "do", "can", "show",
                   "quel", "quelle", "comment", "pourquoi"}
# A filter, not a name: "sales where region = East", "revenue for store = 12".
_FILTER_WORDS = {"where", "when", "if", "whose", "with", "by", "in", "from", "to", "between", "for", "and", "or",
                 "not", "is", "equals", "où", "pour", "avec", "dans", "par"}


# A definition says how something is counted: arithmetic, a count, an average, a share, a condition.
_COUNTING = re.compile(r"[+*/×÷-]|\b(?:minus|plus|divided|times|multiplied|over|per|sum|total|count|counted|number|"
                       r"numbers|average|mean|share|ratio|percent|percentage|less|excluding|including|only|distinct|"
                       r"unique|different|at least|how many|moins|divis[ée]e?|fois|somme|nombre|moyenne|part|"
                       r"pourcentage|seulement|sauf|distincts?)\b", re.I)


class OwnMetricError(ValueError):
    """The reader's metric could not be written as they said it; the message is theirs to read."""


@dataclass(frozen=True)
class Defined:
    name: str
    how: str          # how it is counted, in the reader's words
    ask: str          # the rest of the question, with the metric named where it said "it"


def definition_in(question: str) -> Defined | None:
    """The metric a question defines, and what it then asks; None when it defines none."""
    text = " ".join(str(question or "").split())
    for pattern in _DEFINE:
        found = pattern.match(text)
        if not found:
            continue
        name = found.group("name").strip(" '\"")
        words = name.lower().split()
        if not words or words[0] in _QUESTION_WORDS or len(name) > MAX_NAME or len(words) > 6 \
                or any(w in _FILTER_WORDS for w in words):
            return None
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", found.group("body").strip()) if s.strip()]
        how_parts, ask_parts = [], []
        for sentence in sentences:
            (ask_parts if (ask_parts or _ASK.match(sentence)) and how_parts else how_parts).append(sentence)
        how = " ".join(how_parts).rstrip(".").strip()
        if len(how.split()) < 3 or not _COUNTING.search(how):
            return None       # "region = East", "store = 12 for May": a filter; a definition says how it is counted
        ask = " ".join(ask_parts).strip()
        if ask:
            named = re.sub(r"\b(it|them|this|that)\b", name, ask, count=1, flags=re.I)
            ask = named if named != ask else re.sub(r"-(la|le|les)\b", " " + name, ask, count=1, flags=re.I)
        else:
            ask = name
        return Defined(name=name, how=how[:MAX_DEFINITION], ask=ask)
    return None


def write(model: SemanticModel, defined: Defined, complete: Callable[[str, str], str], *,
          allowed: set[str] | None = None, taken: set[str] = frozenset()) -> Measure:
    """The reader's metric, written by the workspace's AI from their words and checked; or OwnMetricError."""
    clash = next((m.business_name for m in model.measures.values()
                  if not m.hidden and m.business_name.casefold() == defined.name.casefold()
                  and not m.key.startswith(OWN_PREFIX)), None)
    if clash:
        raise OwnMetricError(f"There is already a metric called {clash}: give yours another name.")
    out = authoring.describe(model, f"{defined.name}: {defined.how}", complete)
    if out.get("problem"):
        raise OwnMetricError(out["problem"])
    try:
        read = authoring.read(model, out["formula"])
    except authoring.AuthoringError as exc:
        raise OwnMetricError(f"The metric could not be read: {exc}") from None
    conditions = []
    for c in out.get("conditions") or []:
        f = ColumnFilter.model_validate(c)
        from core2.model.links import value_problem

        if value_problem(model, f):
            raise OwnMetricError(value_problem(model, f))
        conditions.append(f)
    used = set(read.tables) | {model.columns[f.column].table for f in conditions}
    hidden = [c for c in [*read.columns, *(f.column for f in conditions)]
              if c in model.columns and (model.columns[c].hidden or model.columns[c].sensitivity != "none")]
    if hidden or (allowed is not None and not used <= allowed):
        raise OwnMetricError("It would use data you cannot ask about.")
    from core2.model.imports import _behaviour

    additivity, time_aggregation = _behaviour(model, read.table, read.expr)
    base = ids.slug(defined.name, fallback="my_metric")
    slug = ids.unique_slug(base, {m.slug for m in model.measures.values()} | set(taken))
    return Measure(key=OWN_PREFIX + slug, slug=slug, business_name=defined.name, description=out.get("description")
                   or defined.how, synonyms={}, table=read.table, expr=read.expr, additivity=additivity,
                   time_aggregation=time_aggregation, format=out.get("format") or "number", filters=conditions,
                   kind="metric", provenance="ai", status="approved", tested=False)


def with_own(model: SemanticModel, own: dict[str, Measure]) -> SemanticModel:
    """The model as this chat answers with it: the reader's own metrics beside everyone's."""
    if not own:
        return model
    copy = model.model_copy(deep=False)
    copy.measures = {**model.measures, **own}
    return copy


def card(model: SemanticModel, measure: Measure, defined: Defined) -> dict[str, Any]:
    """What the chat shows of it: its name, how it is counted, and that it is the reader's alone."""
    from core2.plan.catalog import definition, left_out_words

    return {"name": measure.business_name, "key": measure.key, "formula": authoring.to_text(model, measure.expr),
            "reads_as": definition(model, measure.expr, True), "words": defined.how,
            "only": [left_out_words(model, f, True) for f in measure.filters],
            "tables": authoring.counted_on(model, measure), "format": measure.format}
