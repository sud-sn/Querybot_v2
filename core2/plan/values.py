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
from dataclasses import dataclass, field, replace

from core2.model.schema import SemanticModel, value_names

MAX_TOKENS = 8

# Everyday words a question is made of. A member stored as one of them -- a group coded DO, AS, IF or
# HAD, a province ON, a unit ME -- is matched only where the question writes it as stored ("sales for DO"),
# never in its ordinary sense ("how many items do we have", "value on 3 September").
COMMON_WORDS = frozenset("""
a am an and any are as at be been but by can could did do does each else for from go goes had has have he
her him his how i if in into is it its may me might more most my no nor not now of off on one or our ours
out over per she should so some such than that the their them then there these they this those to too two
up us was we were what when where which while who whom why will with would yes yet you your all also both
get got just much many new old only own same see set use very via let lot
au aux ce ces dans de des du elle en est et il ils je la le les leur mais mes mon ne nos notre nous on ou
par pas pour qu que quel quelle qui sa se ses son sont sur ta te tes ton tu un une vos votre vous y
""".split())
# A quarter, a half or a fiscal year ("in Q1 2026", "H1", "FY26") is a period. Groups coded Q1 or H2 are
# matched only where the question names their field just before the code ("pdc group Q1"); "deliveries in
# Q1 2026" was handed to the planner as pdc group = Q1, and every count came back empty.
PERIOD_WORD = re.compile(r"^(?:q[1-4]|h[12]|fy\d{2}(?:\d{2})?)$")
_FIELD_NOUNS = frozenset({"code", "codes", "name", "names", "id", "key", "desc", "description", "label"})
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
    attributes: set[str] = field(default_factory=set)     # the attributes whose members were all read
    aliases: dict[str, list[tuple[str, str]]] = field(default_factory=dict)   # names an admin gave codes

    def _entries(self, key: str) -> list[tuple[str, str]]:
        return [*self.names.get(key, []), *self.aliases.get(key, [])]

    def named(self, model: SemanticModel) -> MemberIndex:
        """This index with the names an admin gave the codes of its attributes ("cancelled" finds "C").

        Taken from the model on every question, not read with the members, so a name given
        since the members were read is found at once.
        """
        aliases: dict[str, list[tuple[str, str]]] = {}
        longest = self.longest
        # A yes/no flag is never read as members, but its words are names: "Sterile" finds the flag at 1.
        flags = {slug for slug, a in model.attributes.items()
                 if a.column in model.columns and model.columns[a.column].role == "flag"}
        for slug in sorted({*self.attributes, *flags}):
            attribute = model.attributes.get(slug)
            column = model.columns.get(attribute.column) if attribute is not None else None
            for code, name in (value_names(column).items() if column is not None else ()):
                key = normalise(name)
                if len(key) >= 2 and (slug, code) not in aliases.get(key, []):
                    aliases.setdefault(key, []).append((slug, code))
                    longest = max(longest, min(MAX_TOKENS, len(key.split())))
        return replace(self, aliases=aliases, longest=longest) if aliases else self

    def add(self, attribute: str, values: Iterable[object]) -> None:
        self.attributes.add(attribute)
        for value in values:
            if value is None:
                continue
            stored = str(value)
            key = normalise(stored)
            if not key:
                continue       # a one-letter code ("A") is kept: it names a member only when it is quoted
            entries = self.names.setdefault(key, [])
            if (attribute, stored) not in entries:
                entries.append((attribute, stored))
            self.longest = max(self.longest, min(MAX_TOKENS, len(key.split())))

    def only(self, attributes: set[str] | frozenset[str]) -> MemberIndex:
        """This index with the members of ``attributes`` alone (one hidden or marked sensitive since is dropped)."""
        out = MemberIndex(attributes=self.attributes & set(attributes))
        for own, theirs in ((out.names, self.names), (out.aliases, self.aliases)):
            for key, entries in theirs.items():
                kept = [entry for entry in entries if entry[0] in attributes]
                if kept:
                    own[key] = kept
                    out.longest = max(out.longest, min(MAX_TOKENS, len(key.split())))
        return out

    def stored(self, attribute: str, value: object) -> str | None:
        """The member ``value`` names, as the data stores it ("retail" -> "RETAIL"); None when there is none.

        Only for an attribute whose members were read: elsewhere nothing can be said.
        """
        return next((v for a, v in self._entries(normalise(str(value))) if a == attribute), None)

    def match(self, question: str, *, explicit: bool = False) -> list[ValueMatch]:
        """Whole member names written in the question, longest first, never overlapping.

        A member that is also an everyday word is matched only where the question writes it exactly as
        stored, in a question that is not all capitals (where case says nothing); ``explicit``: the text is
        one the reader put in quotes, so it names a member however it is written.
        """
        words = _tokens(question)
        found: list[ValueMatch] = []
        taken: set[int] = set()
        cased = any(c.islower() for c in question)
        for size in range(min(self.longest, len(words)), 0, -1):
            for i in range(len(words) - size + 1):
                if any(j in taken for j in range(i, i + size)):
                    continue
                start, end = words[i][1], words[i + size - 1][2]
                key = normalise(question[start:end])
                if key.isdigit() and len(key) < 3 and not explicit:
                    continue   # "top 5" is not member "5"
                if len(key) < 2 and not explicit:
                    continue   # "a" is not class "A": one letter names a member only in quotes
                entries = self._entries(key)
                if key in COMMON_WORDS and not explicit:
                    written = question[start:end]
                    entries = [e for e in entries if cased and e[1] == written]
                if PERIOD_WORD.match(key) and not explicit:
                    before = {normalise(w) for w, _, _ in words[max(0, i - 2):i]}
                    entries = [e for e in entries if before & _field_words(e[0])]
                for attribute, value in entries:
                    found.append(ValueMatch(question[start:end], start, end, attribute, value))
                if entries:
                    taken.update(range(i, i + size))
        return sorted(found, key=lambda m: (m.start, m.attribute))

    def quoted(self, question: str) -> list[ValueMatch]:
        """The member names the reader put in quotes ("North", “North”, « Nord », 'North'): the only words of
        a question that may narrow its answer to a member. Ordinary words that happen to be a member's name
        ("stock", "open", "available") are never read as one."""
        found: list[ValueMatch] = []
        for start, end in quoted_spans(question):
            text = question[start:end]
            # The whole text first: a name with punctuation the words leave out ("A-ITEMS, 60%"), or one letter.
            whole = self._entries(normalise(text))
            if whole:
                found += [ValueMatch(text, start, end, attribute, value) for attribute, value in whole]
                continue
            for m in self.match(text, explicit=True):
                found.append(replace(m, start=m.start + start, end=m.end + start))
        return sorted(found, key=lambda m: (m.start, m.attribute))


# Text in quotes: straight or curly double quotes, guillemets, or single quotes around words (never an
# apostrophe inside one: "customer's", "3' pipe").
_QUOTED = re.compile(r""""([^"\n]+)"|“([^”\n]+)”|«\s*([^»\n]+?)\s*»|(?<![\w])'([^'\n]+)'(?![\w])""")


def quoted_spans(question: str) -> list[tuple[int, int]]:
    """Where the question puts text in quotes: (start, end) of the text inside them."""
    out = []
    for m in _QUOTED.finditer(question or ""):
        group = next(i for i in range(1, 5) if m.group(i) is not None)
        if m.group(group).strip():
            out.append((m.start(group), m.end(group)))
    return out


def _field_words(attribute: str) -> set[str]:
    """The words an attribute's slug names it by ("pdc_group.group_code": pdc, group), not "code" or "name"."""
    return {w for w in re.split(r"[._\s]+", attribute.lower()) if w and w not in _FIELD_NOUNS}


_ORDINAL = re.compile(r"\b(?:the\s+)?(first|1st|top|second|2nd|third|3rd|fourth|4th|fifth|5th|last|bottom)"
                      r"\s+(one|1|[a-z]+)\b", re.IGNORECASE)
_PLACE = {"first": 0, "1st": 0, "top": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3,
          "fifth": 4, "5th": 4, "last": -1, "bottom": -1}
# "The lowest one" is the row with the lowest number, wherever the answer put it.
_BY_VALUE = re.compile(r"\b(?:the\s+)?(highest|largest|biggest|lowest|smallest)\s+(one)\b", re.IGNORECASE)
_HIGH = {"highest", "largest", "biggest"}
# "That division": the one member of that field the answer on screen showed.
_THAT = re.compile(r"\b(that|this)\s+([a-z]+)\b", re.IGNORECASE)
# "this year", "that month": a period, never a member on screen, whatever the answer was grouped by.
_PERIOD_NOUNS = {"year", "month", "quarter", "week", "day", "half", "period", "time", "fiscal", "season"}
# "In the top group", "within the largest division": a row of the answer on screen, named by its field after a
# preposition. Without one ("show the top store in March", "the largest customers") it asks for a ranking.
_END_OF = re.compile(r"\b(?:in|for|of|within|from|at|on|to|about|inside|under|across)\s+(the\s+"
                     r"(top|bottom|highest|largest|biggest|lowest|smallest)\s+([a-z]+))\b", re.IGNORECASE)
# "The number one customer": the first row.
_NUMBER_ONE = re.compile(r"\b(?:the\s+)?(?:number\s+one|no\.?\s*1|#\s*1)\s+([a-z]+)\b", re.IGNORECASE)
# "The warehouse that grew the most", "the biggest mover": the row by its change, on an answer that compares.
_BY_CHANGE = re.compile(r"\b(?:the\s+)?([a-z]+)\s+(?:that|which)\s+(grew|rose|increased|went\s+up|gained|fell|dropped|"
                        r"declined|decreased|went\s+down|lost|changed|moved)\s+(?:the\s+)?most\b", re.IGNORECASE)
_MOVER = re.compile(r"\bthe\s+biggest\s+(mover|riser|gainer|faller|loser|increase|rise|drop|fall|decrease|change)\b",
                    re.IGNORECASE)
_UP = {"grew", "rose", "increased", "went up", "gained", "riser", "gainer", "increase", "rise"}
_DOWN = {"fell", "dropped", "declined", "decreased", "went down", "lost", "faller", "loser", "drop", "fall", "decrease"}
# "Its revenue": the one member on screen, or the one the answer named first when it was asked "which ... most".
_ITS = re.compile(r"\bits\b", re.IGNORECASE)
# "The worst one", "the best supplier": which end that is depends on the measure, so the reader is asked.
_BEST_WORST = re.compile(r"\b(?:the\s+)?(best|worst)\s+([a-z]+)\b", re.IGNORECASE)
# "For the best supplier", "the number one customer's revenue": named by its field, a row of the answer on screen
# only after a preposition or as an owner; "which is the best supplier in 2024?" asks for a ranking.
_PREP_BEFORE = re.compile(r"\b(?:in|for|of|within|from|at|on|to|about|inside|under|across|with|by)\s*$", re.IGNORECASE)


def _a_row(question: str, m: re.Match, noun: str) -> bool:
    """Do these words, ending in ``noun``, name a row of the answer on screen rather than ask for a ranking?"""
    return (noun in ("one", "1") or bool(_PREP_BEFORE.search(question[:m.start()]))
            or question[m.end():].startswith(("'s", "\u2019s")))

Shown = tuple  # (attribute slug, stored value or None, the answer's number for it or None,
#                 optionally the row's members of the answer's other groupings by attribute slug,
#                 and optionally the row's change, on an answer that compares two periods)


def _number(s: Shown, at: int) -> float | None:
    value = s[at] if len(s) > at else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def placed(question: str, shown: list[Shown], taken: list[ValueMatch], *, leader: bool = False) -> list[ValueMatch]:
    """"The first one", "the lowest one", "that division": a member of the answer on screen.

    ``shown`` is that answer's members in the order shown, as (attribute slug, stored value,
    its number). The planner is never given the rows: "break the first one down" had no name to
    filter on and came back as a placeholder, and "which division does the lowest one belong to?"
    was filtered on a profit centre called "lowest". By place ("the first one", "the second
    warehouse", "the last one"), by value ("the lowest one", "the highest one"), or as the one member
    the answer showed ("that one") or of a field it showed ("that division", in whichever of its
    groupings). Also "in the top group", "within the largest division", "the number one customer", "the
    warehouse that grew the most" and "the biggest mover" (by the change, on an answer that compares), and
    "its" (the one member on screen, or the first one when ``leader``: the answer was asked which comes
    first, "which group changed the most?"). Only where the words name a row ("one", or the field's own
    noun, never "the first quarter" or "this year"), and never over a member the question names itself.
    """
    if not shown:
        return []
    words = _field_words(shown[0][0])
    nouns = {"one", "1"} | words
    out: list[ValueMatch] = []

    def free(m: re.Match) -> bool:
        return not any(t.start < m.end() and m.start() < t.end for t in [*taken, *out])

    for m in _ORDINAL.finditer(question):
        word, noun = m.group(1).lower(), m.group(2).lower()
        # "The top store in March" asks for a ranking; "the top one" points at a row.
        if noun not in (nouns if word not in ("top", "bottom") else {"one", "1"}) or not free(m):
            continue
        place = _PLACE[word]
        if place >= len(shown):
            continue
        attribute, value = shown[place][0], shown[place][1]
        if value is not None:
            out.append(ValueMatch(m.group(0), m.start(), m.end(), attribute, value))
    for m in _NUMBER_ONE.finditer(question):
        if m.group(1).lower() in nouns and _a_row(question, m, m.group(1).lower()) and free(m) \
                and shown[0][1] is not None:
            out.append(ValueMatch(m.group(0), m.start(), m.end(), shown[0][0], shown[0][1]))
    numbered = [s for s in shown if s[1] is not None and _number(s, 2) is not None]
    for m in _BY_VALUE.finditer(question):
        if not numbered or not free(m):
            continue
        pick = (max if m.group(1).lower() in _HIGH else min)(numbered, key=lambda s: s[2])
        out.append(ValueMatch(m.group(0), m.start(), m.end(), pick[0], pick[1]))
    for m in _END_OF.finditer(question):
        word, noun = m.group(2).lower(), m.group(3).lower()
        if noun not in words or not free(m):
            continue
        if word in ("top", "bottom"):
            pick = shown[0 if word == "top" else -1]
        elif numbered:
            pick = (max if word in _HIGH else min)(numbered, key=lambda s: s[2])
        else:
            continue
        if pick[1] is not None:
            out.append(ValueMatch(m.group(1), m.start(1), m.end(1), pick[0], pick[1]))
    changed = [s for s in shown if s[1] is not None and _number(s, 4) is not None]
    for m, noun, way in [*((m, m.group(1).lower(), " ".join(m.group(2).lower().split()))
                           for m in _BY_CHANGE.finditer(question)),
                         *((m, "mover", m.group(1).lower()) for m in _MOVER.finditer(question))]:
        if not changed or (noun != "mover" and noun not in nouns) or not _a_row(question, m, noun) or not free(m):
            continue
        key = ((lambda s: _number(s, 4)) if way in _UP else (lambda s: -_number(s, 4)) if way in _DOWN
               else (lambda s: abs(_number(s, 4))))
        pick = max(changed, key=key)
        out.append(ValueMatch(m.group(0), m.start(), m.end(), pick[0], pick[1]))
    # Every field of the answer and its members: the first grouping's, and those of the others on its rows.
    fields: dict[str, set[str]] = {shown[0][0]: {s[1] for s in shown if s[1] is not None}}
    for s in shown:
        for attribute, value in (s[3] if len(s) > 3 and isinstance(s[3], dict) else {}).items():
            fields.setdefault(attribute, set()).update([value] if value is not None else [])
    for m in _THAT.finditer(question):
        noun = m.group(2).lower()
        if noun in _PERIOD_NOUNS:
            continue
        # "that one": the one member of the answer's first grouping; "that division": of the field so named.
        named = ([(shown[0][0], fields[shown[0][0]])] if noun in ("one", "1") and len(fields[shown[0][0]]) == 1 else
                 [(a, v) for a, v in fields.items() if noun in _field_words(a) and len(v) == 1])
        if len(named) == 1 and free(m):
            out.append(ValueMatch(m.group(0), m.start(), m.end(), named[0][0], next(iter(named[0][1]))))
    first = fields[shown[0][0]]
    for m in _ITS.finditer(question):
        if not free(m):
            continue
        if len(first) == 1:
            out.append(ValueMatch(m.group(0), m.start(), m.end(), shown[0][0], next(iter(first))))
        elif leader and shown[0][1] is not None:
            out.append(ValueMatch(m.group(0), m.start(), m.end(), shown[0][0], shown[0][1]))
    return sorted(out, key=lambda v: v.start)


def which_end(question: str, shown: list[Shown], taken: list[ValueMatch]) -> tuple[str, str, str] | None:
    """"The worst one", "the best supplier": a row at one end of the answer on screen, but which end depends on
    what is measured (most rejections is the worst, most revenue the best). The phrase and the members at the
    two ends, for the reader to say which; None when the question says no such thing."""
    if len({s[1] for s in shown if s[1] is not None}) < 2 or shown[0][1] is None or shown[-1][1] is None:
        return None
    nouns = {"one", "1"} | _field_words(shown[0][0])
    for m in _BEST_WORST.finditer(question):
        overlaps = any(t.start < m.end() and m.start() < t.end for t in taken)
        if m.group(2).lower() in nouns and _a_row(question, m, m.group(2).lower()) and not overlaps:
            return m.group(0), str(shown[0][1]), str(shown[-1][1])
    return None


def listable(model: SemanticModel) -> list[str]:
    """The attributes whose member names may be read: text, not hidden, not sensitive, values allowed."""
    out = []
    for slug, attribute in model.attributes.items():
        column = model.columns[attribute.column]
        if column.data_type == "text" and not column.hidden and column.sensitivity == "none" and column.values_allowed:
            out.append(slug)
    return out


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
    may = set(listable(model))
    ordered = sorted(model.attributes.items(), key=lambda item: (item[1].column not in labels, item[0]))
    for slug, attribute in ordered:
        if slug not in may:
            continue
        column = model.columns[attribute.column]
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
    values: dict[str, object]              # placeholder -> stored value


def mask(question: str, matches: list[ValueMatch], known: dict[str, object] | None = None) -> Masked:
    """Replace matched member names by ⟨v1⟩, ⟨v2⟩... so no member value reaches the AI.

    ``known`` holds the placeholders earlier turns used: a value keeps its placeholder
    for the whole conversation, so a follow-up ("and in 2024?") can carry it over.
    """
    values = dict(known or {})
    out, last = [], 0
    for m in sorted(matches, key=lambda m: m.start):
        if m.start < last:
            continue
        out.append(question[last:m.start])
        out.append(placeholder(m.value, values))
        last = m.end
    out.append(question[last:])
    return Masked("".join(out), values)


def placeholder(value: object, values: dict[str, object]) -> str:
    """``value``'s placeholder in ``values``, added when it has none yet."""
    token = next((t for t, v in values.items() if v == value), None)
    if token is None:
        token = f"⟨v{len(values) + 1}⟩"
        values[token] = value
    return token


def unmask(value: object, masked: Masked) -> object:
    if isinstance(value, str):
        for token, stored in masked.values.items():
            if value == token:
                return stored
            value = value.replace(token, str(stored))
    return value
