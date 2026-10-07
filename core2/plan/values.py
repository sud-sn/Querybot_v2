"""Member names in a question, found before the AI sees it (DESIGN §7.3).

"Net sales for northline distribution 58" names a customer: the question is
matched against the members of every attribute QueryBot may list, so the planner
is handed ``customer.name = "Northline Distribution 58"`` instead of guessing a
spelling. Matching is on normalised text (case, accents, quotes and dashes),
keeps ``#``, digits and punctuation inside names, prefers exact whole names, then
the longest, and never lets two matches overlap. There are no stop-word lists:
a member called "Top Value Stores" is found even though "top" is an analysis
word, because the whole name is in the question.

For a tenant whose values may not reach the AI, matched text is replaced by a
placeholder before the prompt and put back after planning.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from core2.model.schema import SemanticModel

MAX_TOKENS = 8
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "`": "'"})
_TOKEN = re.compile(r"[\w#&'./+-]+", re.UNICODE)


def normalise(text: str) -> str:
    """Lower case, no accents, plain quotes and dashes, single spaces."""
    text = unicodedata.normalize("NFKD", text.translate(_QUOTES))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.casefold().split())


def _tokens(text: str) -> list[tuple[str, int, int]]:
    """Words with their spans; trailing sentence punctuation is not part of a word."""
    out = []
    for m in _TOKEN.finditer(text):
        word, start, end = m.group(), m.start(), m.end()
        while word and word[-1] in ".'-/" and len(word) > 1:
            word, end = word[:-1], end - 1
        out.append((word, start, end))
    return out


@dataclass(frozen=True)
class ValueMatch:
    text: str              # as written in the question
    start: int
    end: int
    attribute: str         # attribute slug
    value: str             # the stored member value
    exact: bool = True     # the whole name was written (not a containment)


@dataclass
class MemberIndex:
    """Normalised member names -> (attribute slug, stored value), for the attributes QueryBot may list."""

    names: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    longest: int = 1

    def add(self, attribute: str, values: Iterable[object]) -> None:
        for value in values:
            if value is None:
                continue
            stored = str(value)
            key = normalise(stored)
            if not key or len(key) < 2:
                continue
            entries = self.names.setdefault(key, [])
            if (attribute, stored) not in entries:
                entries.append((attribute, stored))
            self.longest = max(self.longest, min(MAX_TOKENS, len(key.split())))

    def match(self, question: str) -> list[ValueMatch]:
        """Whole member names written in the question, longest first, never overlapping."""
        words = _tokens(question)
        found: list[ValueMatch] = []
        taken: set[int] = set()
        for size in range(min(self.longest, len(words)), 0, -1):
            for i in range(len(words) - size + 1):
                if any(j in taken for j in range(i, i + size)):
                    continue
                start, end = words[i][1], words[i + size - 1][2]
                key = normalise(question[start:end])
                if key.isdigit() and len(key) < 3:
                    continue   # "top 5" is not member "5"
                for attribute, value in self.names.get(key, []):
                    found.append(ValueMatch(question[start:end], start, end, attribute, value))
                if key in self.names:
                    taken.update(range(i, i + size))
        return sorted(found, key=lambda m: (m.start, m.attribute))


def build_index(model: SemanticModel, fetch: Callable[[str], list[object]], *, values_allowed: bool = True,
                max_members: int = 20_000, budget_seconds: float | None = None) -> MemberIndex:
    """Members of every attribute whose values may be listed, from profiles or ``fetch(attribute slug)``.

    Entities' names and codes are read first (they are what questions name); with a
    time budget, attributes not reached in time are left out and logged, never waited on.
    """
    import logging
    import time

    index = MemberIndex()
    if not values_allowed:
        return index
    labels = {e.label_column for e in model.entities.values()} | {e.code_column for e in model.entities.values()}
    start = time.monotonic()
    skipped: list[str] = []
    ordered = sorted(model.attributes.items(), key=lambda item: (item[1].column not in labels, item[0]))
    for slug, attribute in ordered:
        column = model.columns[attribute.column]
        if column.hidden or column.sensitivity != "none" or not column.values_allowed:
            continue
        if column.data_type not in ("text",):
            continue
        p = column.profile
        if p is not None and p.top and p.distinct <= len(p.top):
            index.add(slug, (t.value for t in p.top))
        elif (attribute.members or (p.distinct if p else 0)) <= max_members:
            if budget_seconds is not None and time.monotonic() - start > budget_seconds:
                skipped.append(slug)
                continue
            index.add(slug, fetch(slug))
    if skipped:
        logging.getLogger("querybot.core2").warning(
            "core2: member names of %d attributes not indexed within %.0fs: %s", len(skipped), budget_seconds or 0,
            ", ".join(skipped[:10]))
    return index


@dataclass
class Masked:
    question: str                          # with member names replaced by placeholders
    values: dict[str, str]                 # placeholder -> stored value


def mask(question: str, matches: list[ValueMatch]) -> Masked:
    """Replace matched member names by ⟨v1⟩, ⟨v2⟩... so no member value reaches the AI."""
    out, values, last = [], {}, 0
    by_text: dict[str, str] = {}
    for m in sorted(matches, key=lambda m: m.start):
        if m.start < last:
            continue
        token = by_text.get(m.value) or f"⟨v{len(by_text) + 1}⟩"
        by_text[m.value] = token
        values[token] = m.value
        out.append(question[last:m.start])
        out.append(token)
        last = m.end
    out.append(question[last:])
    return Masked("".join(out), values)


def unmask(value: object, masked: Masked) -> object:
    if isinstance(value, str):
        for token, stored in masked.values.items():
            if value == token:
                return stored
            value = value.replace(token, stored)
    return value
