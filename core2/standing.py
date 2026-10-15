"""Standing conditions: a period or names a reader sets once and every later answer keeps until it is removed.

"From now on use the last 2 months", "only the Downtown store from now on", "for this conversation, leave out
"Cancelled"": every question after it in the conversation is answered for that period and those members, shown as
a chip above the box and on each answer. "Always use the last 2 months" is kept in the reader's profile instead:
every conversation of theirs starts with it, and the Profile page lists it.

Read in code, never by the AI (the names are the workspace's members, found in its member index): a period in
words, and the names that follow "only", "just", "leave out", "exclude" or "without". A name the data does not
hold is said, never guessed.

A question that names its own period, or its own member of the same field, is answered for those, and the answer
says the standing one was set aside for it. A condition that cannot apply to a question's rows (a field they do not
reach) is set aside for that answer and said too.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from core2.model.schema import Attribute, SemanticModel
from core2.plan.followup import names_own_period
from core2.plan.ir import Filter, Plan, TimeSpec, Window
from core2.plan.values import MemberIndex, ValueMatch

Scope = Literal["conversation", "always"]

# "From now on ...", "for the rest of this conversation ...", "désormais ...": for this conversation.
_FOR_NOW = (r"from\s+now\s+on|going\s+forward|from\s+here\s+on|for\s+the\s+rest\s+of\s+(?:this|the)\s+"
            r"(?:conversation|chat|thread)|for\s+(?:this|the)\s+(?:whole\s+)?(?:conversation|chat|thread)|"
            r"until\s+I\s+say\s+otherwise|(?:à|a)\s+partir\s+de\s+maintenant|d(?:é|e)sormais|dor(?:é|e)navant|"
            r"pour\s+(?:toute\s+|la\s+suite\s+de\s+)?cette\s+conversation")
_STARTS = re.compile(rf"^\s*(?:{_FOR_NOW})\b[\s,:;-]*", re.IGNORECASE)
_ENDS = re.compile(rf"[\s,;-]*\b(?:{_FOR_NOW})\s*[.!]?\s*$", re.IGNORECASE)
# "Always use ...", "toujours ...": kept in the reader's profile, for every conversation.
_ALWAYS = re.compile(r"^\s*(?:always|by\s+default|toujours|par\s+d(?:é|e)faut)\b[\s,:;-]*", re.IGNORECASE)
# What a condition says first: "use", "only", "leave out", "exclude" ... (so "always on time" asks nothing).
_SAYS = re.compile(r"^\s*(?:please\s+)?(?:use|only|just|leave\s+out|exclude|excluding|without|except|look\s+at|"
                   r"show|limit|restrict|keep|stick\s+to|focus\s+on|filter|count|answer|work\s+with|in|for|over|"
                   r"utilise[rz]?|seulement|uniquement|exclu(?:re|ez|s)|sans|sauf|garde[rz]?|limite[rz]?|pour|sur)\b",
                   re.IGNORECASE)
# The words before a name it narrows to, or leaves out.
_ONLY = r"(?:only|just|solely|exclusively|stick\s+to|focus\s+on|restrict(?:ed)?\s+to|limit(?:ed)?\s+to|seulement|" \
        r"uniquement|que)"
_LEAVE = r"(?:leave\s+out|leaving\s+out|exclude|excluding|without|except|not|no|ignore|ignoring|drop|sauf|hors|sans|" \
         r"exclu(?:re|ez|s|ant))"
# "Stop using the last 2 months", "remove the standing conditions", "forget the Downtown filter".
_LET_GO = re.compile(r"^\s*(?:please\s+)?(?:stop\s+(?:using|applying|filtering)|stop|forget|remove|drop|clear|cancel|"
                     r"undo|turn\s+off|switch\s+off|no\s+longer\s+use|don'?t\s+use|enl[eè]ve[rz]?|supprime[rz]?|"
                     r"oublie[rz]?|arr[eê]te[rz]?\s+(?:d['’]utiliser|de\s+filtrer)?)\b(.*)$", re.IGNORECASE)
_ALL_OF_THEM = re.compile(r"^\W*(?:all\s+)?(?:the\s+|my\s+|these\s+|those\s+)?(?:standing\s+)?(?:conditions?|filters?|"
                          r"periods?|settings?|them|it|that|those|all|everything|tout|toutes?|les\s+conditions?|"
                          r"les\s+filtres?)\W*$", re.IGNORECASE)

# Words that name no condition: "remove the Downtown condition", "stop using the last 2 months for good".
_FILLER = {"the", "a", "an", "my", "this", "that", "these", "those", "condition", "conditions", "filter", "filters",
           "standing", "period", "for", "good", "now", "please", "from", "profile", "everywhere", "permanently",
           "always", "is", "of", "on", "le", "la", "les", "filtre", "condition", "pour", "de", "du", "des", "toujours"}

_MONTHS = {m: i for i, m in enumerate(("january", "february", "march", "april", "may", "june", "july", "august",
                                       "september", "october", "november", "december"), 1)}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})
_MONTHS.update({m: i for i, m in enumerate(("janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
                                            "septembre", "octobre", "novembre", "décembre"), 1)})
_MONTH = r"(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")"
_UNITS = {"week": "week", "month": "month", "quarter": "quarter", "year": "year", "semaine": "week", "mois": "month",
          "trimestre": "quarter", "année": "year", "annee": "year", "an": "year"}


@dataclass
class Standing:
    """One standing condition: its period, or its filters, and the words the reader sees for it."""

    id: str
    words: str                          # "the last 2 months", "Store is Downtown"
    scope: Scope = "conversation"
    window: Window | None = None
    filters: list[Filter] = field(default_factory=list)
    words_fr: str = ""                  # "les 2 derniers mois", "Store est Downtown"

    def shown(self) -> dict[str, Any]:
        """For the screen: the chip's words (a French reader's too), and where it is kept."""
        return {"id": self.id, "words": self.words, "words_fr": self.words_fr or self.words, "scope": self.scope}

    def data(self) -> dict[str, Any]:
        """As kept in a profile."""
        return {**self.shown(), "window": self.window.model_dump(mode="json", exclude_defaults=True)
                if self.window is not None else None,
                "filters": [f.model_dump(mode="json", exclude_defaults=True) for f in self.filters]}

    @classmethod
    def of(cls, data: dict[str, Any]) -> Standing:
        return cls(id=str(data["id"]), words=str(data.get("words") or ""), scope=data.get("scope") or "always",
                   window=Window(**data["window"]) if data.get("window") else None,
                   filters=[Filter(**f) for f in data.get("filters") or []],
                   words_fr=str(data.get("words_fr") or ""))


@dataclass
class Said:
    """What a reader's standing instruction set, or why it set nothing."""

    standing: list[Standing]
    scope: Scope
    unknown: list[str] = field(default_factory=list)     # names written that the data does not hold
    unquoted: list[ValueMatch] = field(default_factory=list)


def instruction(question: str) -> tuple[Scope, str] | None:
    """A standing instruction's scope and the words of its condition; None for any other question."""
    text = question.strip()
    found = _ALWAYS.match(text)
    if found and _SAYS.match(text[found.end():]):
        return "always", _STARTS.sub("", text[found.end():]).strip(" .!")
    found = _STARTS.match(text)
    if found and text[found.end():].strip(" .!"):
        rest = text[found.end():]
        always = _ALWAYS.match(rest)
        if always and _SAYS.match(rest[always.end():]):
            return "always", rest[always.end():].strip(" .!")
        return "conversation", rest.strip(" .!")
    found = _ENDS.search(text)
    if found and found.start() > 0 and _SAYS.match(text[:found.start()]):
        return "conversation", text[:found.start()].strip(" .!")
    return None


def letting_go(question: str, kept: list[Standing]) -> list[Standing] | None:
    """The standing conditions a question removes ("stop using the last 2 months", "clear the conditions"); None
    when it removes none (and is read as a question)."""
    found = _LET_GO.match(question.strip())
    if found is None or not kept:
        return None
    rest = found.group(1).strip(" .!?")
    rest = re.sub(r"^(?:using|applying|with|the\s+standing|standing)\s+", "", rest, flags=re.IGNORECASE)
    if not rest or _ALL_OF_THEM.match(rest):
        return list(kept)
    said = set(_plain(rest).split()) - _FILLER
    named = [s for s in kept if said and any(said <= set(_plain(w).split()) | _FILLER for w in (s.words, s.words_fr))]
    return named or None


def read(scope: Scope, words: str, model: SemanticModel, index: MemberIndex, today: dt.date) -> Said:
    """The conditions in a standing instruction's words: its period, and the members it narrows to or leaves out."""
    from core2.plan.planner import period_meant, period_words

    out: list[Standing] = []
    window = _window(words, today)
    plan = Plan(kind="query", time=TimeSpec(window=window or Window()))
    meant = period_meant(plan, model, words).time.window
    if meant.kind != "all":
        window = meant
    if window is not None and window.kind != "all":
        out.append(Standing(_id(scope, "period", window.model_dump_json()), period_words(window).removeprefix("in ")
                            or "the period said", scope, window=window, words_fr=_period_fr(window)))
    filters, unknown, unquoted = _members(words, model, index)
    for f in filters:
        out.append(Standing(_id(scope, f.model_dump_json()), _filter_words(f, model), scope, filters=[f],
                            words_fr=_filter_words(f, model, "fr")))
    return Said(out, scope, unknown, unquoted)


def apply(plan: Plan, kept: list[Standing], question: str, model: SemanticModel,
          answers: Callable[[Plan], bool], previous: Plan | None = None
          ) -> tuple[Plan, list[Standing], list[tuple[str, str]]]:
    """The plan under the standing conditions; those applied, and what was set aside for this question and why
    (in English and in French)."""
    if plan.kind != "query" or not kept:
        return plan, [], []
    applied: list[Standing] = []
    aside: list[tuple[str, str]] = []

    def set_aside(s: Standing, why: str) -> None:
        aside.append((f"{s.words} ({why})", f"{s.words_fr or s.words} ({_WHY_FR[why]})"))

    for s in kept:
        if s.window is not None:
            window = plan.time.window
            carried = previous is not None and plan.follow_up == "refine" and window.kind != "all" \
                and window == previous.time.window and all(window != k.window for k in kept if k.window)
            if names_own_period(question) or carried:
                # "Net sales in 2024" under "the last 2 months": the question's own period, and said.
                if window != s.window:
                    set_aside(s, "your question names its own period" if names_own_period(question)
                              else "the period of the answer above is kept")
                continue
            trial = plan.model_copy(update={"time": plan.time.model_copy(update={"window": s.window})})
            if answers(trial):
                plan = trial
                applied.append(s)
            else:
                set_aside(s, "not for these rows")
            continue
        own = [f for f in s.filters if any(g.field == f.field for g in plan.filters)]
        if own:
            if not all(f in plan.filters for f in s.filters):
                set_aside(s, "your question names its own")
            else:
                applied.append(s)
            continue
        trial = plan.model_copy(update={"filters": [*plan.filters, *s.filters]})
        if answers(trial):
            plan = trial
            applied.append(s)
        else:
            set_aside(s, "not for these rows")
    return plan, applied, aside


def without(plan: Plan, applied: list[Standing], bare: Plan) -> Plan:
    """The plan as the question asked it, the standing conditions taken out again: what the conversation keeps,
    so a condition removed is gone from the follow-ups too."""
    if not applied:
        return plan
    added = [f for s in applied for f in s.filters if f not in bare.filters]
    update: dict[str, Any] = {"filters": [f for f in plan.filters if f not in added]}
    if any(s.window is not None for s in applied):
        update["time"] = plan.time.model_copy(update={"window": bare.time.window})
    return plan.model_copy(update=update)


# ── reading the words ───────────────────────────────────────────────────────


def _window(words: str, today: dt.date) -> Window | None:
    """A period the words name that period_meant does not: this/last week, month, quarter or year, a year, a
    quarter, a month, or "since" one."""
    text = " ".join(words.casefold().replace("’", "'").split())
    since = re.search(rf"\bsince\s+(?:{_MONTH}\s+)?((?:19|20)\d\d)\b|\bdepuis\s+(?:{_MONTH}\s+)?((?:19|20)\d\d)\b", text)
    if since:
        month = _MONTHS.get(since.group(1) or since.group(3) or "", 1)
        return Window(kind="since", start=dt.date(int(since.group(2) or since.group(4)), month, 1))
    found = re.search(r"\bq([1-4])\s*((?:19|20)\d\d)\b", text)
    if found:
        q, year = int(found.group(1)), int(found.group(2))
        end = dt.date(year + (q == 4), 1 if q == 4 else 3 * q + 1, 1) - dt.timedelta(days=1)
        return Window(kind="between", start=dt.date(year, 3 * q - 2, 1), end=end)
    found = re.search(rf"\b{_MONTH}\s+((?:19|20)\d\d)\b", text)
    if found:
        month, year = _MONTHS[found.group(1)], int(found.group(2))
        end = dt.date(year + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1)
        return Window(kind="between", start=dt.date(year, month, 1), end=end)
    found = re.search(r"\b((?:19|20)\d\d)\b", text)
    if found:
        year = int(found.group(1))
        return Window(kind="between", start=dt.date(year, 1, 1), end=dt.date(year, 12, 31))
    found = re.search(r"\b(?:this|the\s+current|current)\s+(week|month|quarter|year)\b|"
                      r"\b(?:ce|cette)\s+(semaine|mois|trimestre|ann[ée]e)\b", text)
    if found:
        return Window(kind="this", unit=_UNITS[found.group(1) or found.group(2)])
    found = re.search(r"\b(?:last|previous|the\s+previous)\s+(week|month|quarter|year)\b|"
                      r"\b(?:le|la|l')\s*(semaine|mois|trimestre|ann[ée]e)\s+(?:derni[eè]re?|pr[ée]c[ée]dente?)\b",
                      text)
    if found:
        return Window(kind="previous", unit=_UNITS[found.group(1) or found.group(2)])
    return None


def _members(words: str, model: SemanticModel, index: MemberIndex
             ) -> tuple[list[Filter], list[str], list[ValueMatch]]:
    """The members the words narrow to or leave out: quoted names, and names written right after "only" or
    "leave out"; the names said that the data does not hold; and ordinary words that happen to be members."""
    quoted = index.quoted(words)
    written = [m for m in index.match(words) if not any(q.start <= m.start < q.end for q in quoted)]
    picked: list[tuple[ValueMatch, bool]] = []
    for m in [*quoted, *written]:
        before = words[:m.start]
        leave = re.search(rf"\b{_LEAVE}\b(?:\s+(?:the|any|all|les?|la|des?|du))?(?:\s+[\w'-]+){{0,2}}\s*[\"“«']?\s*$",
                          before, re.IGNORECASE)
        only = re.search(rf"\b{_ONLY}\b(?:\s+(?:the|our|les?|la|des?|du|l'))?(?:\s+[\w'-]+){{0,2}}\s*[\"“«']?\s*$",
                         before, re.IGNORECASE)
        if m in quoted or leave or only:
            picked.append((m, bool(leave) and (not only or leave.start() > only.start())))
    picked = _one_field_each(picked, words, model)
    filters: dict[tuple[str, bool], Filter] = {}
    for m, out in picked:
        key = (m.attribute, out)
        if key in filters:
            f = filters[key]
            filters[key] = f.model_copy(update={"op": "not_in" if out else "in", "values": [*f.values, m.value]})
        else:
            filters[key] = Filter(field=m.attribute, op="ne" if out else "eq", values=[m.value])
    unknown = [words[a:b] for a, b in _spans(words) if not any(m.start >= a and m.end <= b for m in quoted)]
    unquoted = [m for m in written if not any(m is p for p, _ in picked)]
    return list(filters.values()), unknown, unquoted


def _one_field_each(picked: list[tuple[ValueMatch, bool]], words: str, model: SemanticModel
                    ) -> list[tuple[ValueMatch, bool]]:
    """A name held by several fields ("Downtown", a store and a region) read as the one the words name ("the
    Downtown store"), else the first."""
    said = set(re.findall(r"[a-z]+", words.casefold()))
    by_place: dict[int, list[tuple[ValueMatch, bool]]] = {}
    for m, leave in picked:
        by_place.setdefault(m.start, []).append((m, leave))
    out = []
    for same in by_place.values():
        named = [p for p in same if said & _field_words(model, p[0].attribute)]
        out.append(named[0] if named else same[0])
    return out


def _field_words(model: SemanticModel, slug: str) -> set[str]:
    attribute = model.attributes.get(slug)
    words = set(re.findall(r"[a-z]+", slug.casefold().replace("_", " ").replace(".", " ")))
    if isinstance(attribute, Attribute):
        words |= set(re.findall(r"[a-z]+", (attribute.business_name or "").casefold()))
        entity = model.entities.get(attribute.entity or "")
        if entity is not None:
            words |= set(re.findall(r"[a-z]+", (entity.business_name or "").casefold()))
            words |= {w for ws in entity.synonyms.values() for v in ws for w in re.findall(r"[a-z]+", v.casefold())}
    from core2.plan.words import _EVERYDAY

    return words | {f"{w}s" for w in words} | {e for w in words for e in _EVERYDAY.get(w, ())}


_WHY_FR = {"your question names its own period": "votre question nomme sa propre période",
           "the period of the answer above is kept": "la période de la réponse précédente est gardée",
           "your question names its own": "votre question nomme le sien",
           "not for these rows": "ne s’applique pas à ces lignes"}
# A reader writing in French: "désormais ...", "toujours exclure ...", "enlève les 2 derniers mois".
_FRENCH = re.compile(r"\b(?:d(?:é|e)sormais|dor(?:é|e)navant|(?:à|a)\s+partir\s+de\s+maintenant|toujours|par\s+d(?:é|e)faut|"
                     r"seulement|uniquement|exclu(?:re|ez|s)|sans|sauf|les|des|derni(?:er|ère)s?|mois|cette|enl(?:è|e)ve|"
                     r"supprime|oublie|arr(?:ê|e)te|retire)\b", re.IGNORECASE)


def french(question: str) -> bool:
    """Is a standing instruction, or its removal, written in French?"""
    from core2.plan.values import quoted_spans

    plain = question
    for a, b in sorted(quoted_spans(question), reverse=True):     # names in quotes are in no language
        plain = plain[:a] + " " + plain[b:]
    return bool(_FRENCH.search(plain)) and not re.search(r"\b(?:the|from\s+now|use|only|always|stop|remove)\b", plain,
                                                         re.IGNORECASE)


_UNIT_FR = {"day": ("jour", "jours", "m"), "week": ("semaine", "semaines", "f"), "month": ("mois", "mois", "m"),
            "quarter": ("trimestre", "trimestres", "m"), "year": ("année", "années", "f")}


def _period_fr(window: Window) -> str:
    """A standing period in French words: "les 2 derniers mois", "ce trimestre", "en 2025"."""
    from core2.answer.builder import _span_fr

    one, many, gender = _UNIT_FR.get(window.unit or "", ("", "", "m"))
    n = window.n or 1
    last = "dernières" if gender == "f" else "derniers"
    if window.kind in ("last", "rolling") and one:
        said = f"les {n} {last} {many}" if n > 1 else f"{'la' if gender == 'f' else 'le'} {last[:-1]} {one}"
        return said + (" jusqu’à aujourd’hui" if window.kind == "rolling" else "")
    if window.kind == "this" and one:
        return {"week": "cette semaine", "month": "ce mois-ci", "quarter": "ce trimestre", "year": "cette année"}[
            window.unit or "year"]
    if window.kind == "previous" and one:
        return {"week": "la semaine dernière", "month": "le mois dernier", "quarter": "le trimestre dernier",
                "year": "l’année dernière"}[window.unit or "year"]
    if window.kind == "to_date" and one:
        return {"week": "depuis le début de la semaine", "month": "depuis le début du mois",
                "quarter": "depuis le début du trimestre", "year": "depuis le début de l’année"}[window.unit or "year"]
    if window.kind == "between" and window.start and window.end:
        return _span_fr(window.start, window.end + dt.timedelta(days=1))
    if window.kind == "since" and window.start:
        return _span_fr(window.start, None)
    return "la période dite"


def _filter_words(f: Filter, model: SemanticModel, lang: str = "en") -> str:
    attribute = model.attributes.get(f.field)
    name = (attribute.business_name if isinstance(attribute, Attribute) and attribute.business_name
            else f.field.split(".")[-1].replace("_", " "))
    entity = model.entities.get(attribute.entity or "") if isinstance(attribute, Attribute) else None
    if entity is not None and name.casefold() in ("name", "label", "description"):
        name = entity.business_name
    ops = {"eq": "is", "in": "is one of", "ne": "is not", "not_in": "is none of"} if lang == "en" else \
        {"eq": "est", "in": "est l’un de", "ne": "n’est pas", "not_in": "n’est aucun de"}
    values = ", ".join(str(v) for v in f.values)
    return f"{name[:1].upper()}{name[1:]} {ops.get(f.op, f.op)} {values}"


def _spans(words: str) -> list[tuple[int, int]]:
    from core2.plan.values import quoted_spans

    return list(quoted_spans(words))


def _plain(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]
