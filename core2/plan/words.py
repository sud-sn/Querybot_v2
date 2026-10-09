"""A reader's own words for what the data holds: "sales" for Revenue, "prd qty" for Product quantity.

The planner reads the catalog: each measure, field and date by its name and the words
people also use for it. Readers write what they say at work instead: an everyday word
("sales", "clients"), or an abbreviation ("prd qty", "cust seg", "amt"). Each reader may use
a different one, once. Those readings are found here, without the AI, and handed to it
beside the question as the likeliest meaning; the AI still decides, and a question that
says otherwise wins.

A reading is offered only when a run of the question's words names the WHOLE of a name,
word for word, each word the same, an abbreviation of it (core2.bootstrap.names, the
reading Learn uses for column names) or an everyday word for it; and at least one word is
not written as the name has it (an exact name the AI reads in the catalog itself).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core2.bootstrap import names
from core2.model.schema import SemanticModel

MAX_WORDS = 4          # the longest name matched, in words
MAX_READINGS = 8       # readings handed to the planner for one question

# Everyday words for the same thing in business questions; each group reads as one word.
EVERYDAY = [
    {"sales", "sale", "revenue", "turnover"},
    {"quantity", "qty", "units", "unit"},
    {"customer", "client"},
    {"product", "item", "sku", "article"},
    {"supplier", "vendor"},
    {"employee", "staff", "worker"},
    {"store", "shop", "outlet"},
    {"warehouse", "depot"},
    {"cost", "cogs"},
    {"discount", "markdown"},
]
_EVERYDAY = {word: frozenset(group) for group in EVERYDAY for word in group}

# Words a question uses for its own shape, never part of a name it means.
_SHAPE = {"a", "an", "the", "of", "by", "in", "for", "per", "and", "or", "to", "on", "at", "is", "are", "was", "what",
          "which", "how", "each", "our", "my", "me", "show", "give", "list", "top", "bottom", "last", "this", "next",
          "year", "years", "month", "months", "week", "weeks", "day", "days", "quarter", "today", "yesterday", "vs",
          "versus", "compare", "with", "without", "from", "between", "all", "total", "many", "much", "were", "did"}


@dataclass(frozen=True)
class Reading:
    said: str          # as the question writes it
    kind: str          # measure | field | date
    slug: str
    name: str


def _same(asked: str, named: str, loose: bool) -> bool:
    """One word of the question and one of a name: the same word, an abbreviation of it, or an everyday word for it.

    ``loose`` (a run of two words or more, "cust seg") also takes a shortening the abbreviation list does not
    hold; one word alone does not ("sale" is no shortening of "salesperson").
    """
    a, b = names.singular(asked), names.singular(named)
    if a == b:
        return True
    plain_a, plain_b = names.EXPANSIONS.get(a, a), names.EXPANSIONS.get(b, b)
    if plain_a == plain_b:            # an abbreviation of the same word: "qty" and "quantity"
        return True
    if plain_b in _EVERYDAY.get(plain_a, ()):    # an everyday word for it: "sales" and "revenue"
        return True
    # A shortening the abbreviation list does not hold ("prod", "seg"): three to five letters, an everyday word
    # never, and the name's word clearly longer.
    return loose and 3 <= len(a) <= 5 and a not in _EVERYDAY and len(b) >= len(a) + 2 and names.same_word(a, b)


def _phrases(model: SemanticModel) -> list[tuple[str, str, str, list[str]]]:
    """(kind, slug, name, words) for every name a question may mean: measures, fields and dates, and their words."""
    out: list[tuple[str, str, str, list[str]]] = []

    def add(kind: str, slug: str, name: str, said: list[str]) -> None:
        for phrase in [name, *said]:
            words = [w for w in names.tokens(phrase) if w not in ("of", "the", "a", "an")]
            if 0 < len(words) <= MAX_WORDS:
                out.append((kind, slug, name, words))

    for m in model.measures.values():
        if not m.hidden and m.slug:
            add("measure", m.slug, m.business_name, [w for ws in m.synonyms.values() for w in ws])
    for a in model.attributes.values():
        column = model.columns.get(a.column)
        if column is None or column.hidden or column.sensitivity != "none":
            continue
        entity = model.entities.get(a.entity or "")
        owner = [f"{entity.business_name} {a.business_name}"] if entity is not None and entity.business_name \
            and not a.business_name.lower().startswith(entity.business_name.lower()) else []
        add("field", a.slug, a.business_name, [*owner, *(w for ws in a.synonyms.values() for w in ws)])
    for e in model.entities.values():
        add("field", e.slug, e.business_name, [w for ws in e.synonyms.values() for w in ws])
    for r in model.date_roles.values():
        if r.kind != "audit" and r.slug:
            add("date", r.slug, r.name, [w for ws in r.synonyms.values() for w in ws])
    return out


def readings(model: SemanticModel, question: str) -> list[Reading]:
    """What the question's own words most likely name, where they do not write the name as it is."""
    spans = [(m.group(), m.start(), m.end()) for m in re.finditer(r"[A-Za-z][A-Za-z0-9]*", question)]
    words = [w.lower() for w, _, _ in spans]
    found: dict[tuple[int, int], list[Reading]] = {}
    exact: set[tuple[int, int]] = set()
    for kind, slug, name, phrase in _phrases(model):
        size = len(phrase)
        for i in range(len(words) - size + 1):
            run = words[i:i + size]
            if any(w in _SHAPE for w in run) or not all(_same(a, b, size > 1) for a, b in zip(run, phrase)):
                continue
            key = (i, i + size)
            if all(names.singular(a) == names.singular(b) for a, b in zip(run, phrase)):
                exact.add(key)       # the name as written: the AI reads it in the catalog itself
                continue
            said = question[spans[i][1]:spans[i + size - 1][2]]
            reading = Reading(said, kind, slug, name)
            if reading not in found.setdefault(key, []):
                found[key].append(reading)
    # The longest run wins where runs overlap ("prd qty" over "qty"); a run some name writes exactly says nothing.
    kept: list[Reading] = []
    taken: set[int] = set()
    for (start, end), items in sorted(found.items(), key=lambda kv: (-(kv[0][1] - kv[0][0]), kv[0][0])):
        if (start, end) in exact or any(j in taken for j in range(start, end)):
            continue
        taken.update(range(start, end))
        kept.extend(items[:3])
    return kept[:MAX_READINGS]


def readings_text(model: SemanticModel, question: str) -> str:
    """The readings as the planner is shown them (empty when there are none)."""
    found = readings(model, question)
    if not found:
        return ""
    lines = ["READER'S WORDS (what the question's own words most likely name; use these unless the question says "
             "otherwise):"]
    lines += [f'- "{r.said}" -> {r.kind} {r.slug} ({r.name})' for r in found]
    return "\n".join(lines)
