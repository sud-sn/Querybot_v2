"""Follow-up or new question: what the words of a turn say, before any AI is asked.

Most turns say plainly whether they continue the answer on screen. "Only North", "by week instead",
"and refunds", "the first one", "why did it drop?", "as a pie" change that answer; "What were refunds
by return reason last quarter?" names what it measures and for when, and asks afresh. Those are
decided here, the same way every time, in English and in French. A complete question after a narrowed
answer that names no period of its own ("How many orders?" after net sales for one region in March) is
the AI's to read with the conversation in view, as a person would: the same subject going on keeps the
scope above (core2/plan/thread.py), another subject starts afresh. The reader is never asked which.

Readings: ``refine`` (continues the answer on screen), ``new`` (a question of its own), ``open`` (the AI
decides).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core2.model.schema import SemanticModel
from core2.plan.ir import Plan
from core2.plan.words import _SHAPE, _phrases, _same

NEW_PREFIX = re.compile(r"^\s*new question\s*[:\-–]\s*", re.IGNORECASE)

_LEADS = (r"and|also|plus|but|now|then|ok(?:ay)?|what about|how about|what if|only|just|except|excluding|exclude|"
          r"without|instead|same|versus|vs\.?|by|per|for each|in|for|"
          # the same in French: "et par mois ?", "seulement les clients en ligne", "pour 2025", "hors taxes"
          r"et|aussi|puis|maintenant|ensuite|seulement|uniquement|juste|sauf|hors|sans|plut[oô]t|pareil|idem|"
          r"qu'en est-il|et si|par|pour")
# Words that open a scope rather than carry on ("By store, what were net sales in 2025?", "Pour 2025, quelles sont
# les ventes ?"): a lead only in a short turn, or one that names no period of its own.
_SCOPE_LEAD = re.compile(r"^\s*(?:by|per|for|in|without|excluding|par|pour|sans|hors|en)\b", re.IGNORECASE)
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
    r"\b(?:it|its|that|those|them|these|they|this one|that one|the same(?!-)|above|the first one|the second one|"
    r"the third one|the last one|the lowest one|the highest one|the top one|the bottom one|of those|of these|of them|"
    r"the largest one|the smallest one|the biggest one|the \w+ ones)\b|"
    # "and their cost?", "et leurs ventes ?": theirs on screen near the start (not "stores that beat their target")
    r"^\s*(?:\w+\s+){0,2}(?:their|leurs?)\b|"
    # "ce magasin", "ces clients", "celui-ci": not "ce mois", "cette année", "ces 12 derniers mois", "ce que",
    # "est-ce"
    r"(?<![\w-])(?:ce|cet|cette|ces)\s+(?!(?:que|qui|qu|mois|trimestre|semestre|jour|matin|soir|week-end|ann[ée]es?|"
    r"semaines?|saison|exercice|derni[eè]re?s?|premi[eè]re?s?|\d+)\b)(?!qu')\w+|"
    r"\b(?:eux|celui-ci|celle-ci|ceux-ci|celles-ci|celui-l[aà]|celle-l[aà]|ceux-l[aà]|ci-dessus|"
    r"(?:le|la|les)\s+m[êe]mes?)(?![\w-])", re.IGNORECASE)
# A turn that opens by pointing ("the first one, how did it do each month?") follows the answer, however long.
_POINTS_FIRST = re.compile(
    r"^\s*(?:the\s+(?:first|second|third|fourth|fifth|last|lowest|highest|largest|smallest|biggest|top|bottom)"
    r"(?:\s+one)?|that\s+\w+|those\s+\w+|these\s+\w+|it|they|them)\b", re.IGNORECASE)
# "those same customers", "how many of them": the members of the answer on screen, in a turn of any length.
_POINTS_AT_MEMBERS = re.compile(r"\b(?:those|these|them|the same(?!-))\b|\b(?:for|of|in)\s+each(?:\s+one)?\s*\??\s*$|"
                                # "ces clients", "et leurs ventes", "les mêmes produits"
                                r"(?<![\w-])ces\s+(?!(?:\d+|derni\w*|premi\w*|mois|ann[ée]es?|semaines?|jours)\b)\w|"
                                r"\b(?:eux|ceux-ci|celles-ci)\b|^\s*(?:\w+\s+){0,2}leurs?\b|\b(?:les\s+m[êe]mes)\b|"
                                # "Show their monthly quantity", "and their cost?" (not "customers who raised their
                                # spend", where "their" is the question's own)
                                r"^\s*(?:\w+\s+){0,2}their\b", re.IGNORECASE)
# "the stores that opened in 2025": a "that" that starts a description points at nothing on screen.
_RELATIVE = re.compile(r"\b(?:that|which|who)\s+(?:\w+ed|are|were|have|has|had|did|do|does|is|was|can|will|bought|"
                       r"sold|made|got|took|left|went|came|hold|holds|pay|pays|use|uses)\b", re.IGNORECASE)
_DISPLAY = re.compile(
    r"\b(?:as an?\s+(?:pie|bar|line|donut|doughnut|ring|table|chart|area|graph|heatmap|heat map|scatter|grid)|"
    r"stacked(?: bars?)?|(?:as|on) a map|as a share|share of (?:the )?total|"
    r"as a percent(?:age)? of (?:the )?total|chart it|plot it|graph it|just the table)\b", re.IGNORECASE)
_WHY = re.compile(r"^\s*(?:why|what\s+(?:drove|caused|explains|is behind)|how come|what changed)\b", re.IGNORECASE)
# "Why did they cancel?", "why do customers leave?": the reasons people had, not why the figure on screen moved.
# Answered from data that holds reasons, or said to be none -- never the members that moved the figure.
_WHY_REASON = re.compile(
    r"^\s*why\s+(?:did|do|does|would|are|were|is|was)\s+(?:they|customers|clients|people|users|members|"
    r"subscribers|employees|staff|patients|buyers|he|she)\b(?![^?]*\b(?:drop|dropped|fall|fell|rise|rose|grow|grew|"
    r"increase|increased|decrease|decreased|change|changed|go up|go down|went up|went down|spend|spent|buy|bought|"
    r"order|ordered|pay|paid|use|used)\b)", re.IGNORECASE)
# "Which items explain the change?", "what is behind this drop": about the change the answer on screen shows.
_BEHIND = re.compile(r"\b(?:explain|explains|explained|drove|drive|drives|caused|cause|causes|behind|account\s+for|"
                     r"accounts\s+for)\s+(?:the|this|that)\s+(?:change|increase|decrease|drop|rise|fall|growth|"
                     r"decline|difference|gap|jump|dip|move|movement)\b", re.IGNORECASE)
_ASKS = re.compile(r"^\s*(?:what|what's|whats|which|who|whom|how|when|where|show|list|give|tell|compare|is|are|do|does|"
                   r"did|can|could|find|get|count|"
                   r"quel|quelle|quels|quelles|quel\s+est|combien|qui|où|quand|comment|montre[sz]?|liste[sz]?|"
                   r"donne[sz]?|affiche[sz]?|compare[sz]?|y\s+a-t-il|est-ce)\b", re.IGNORECASE)
_ABOUT_DATA = re.compile(r"^\s*(?:what\s+(?:data|information|tables)|what can i ask|what do you (?:have|know)|"
                         r"what(?:'s| is)\s+(?:in|available)|which tables|describe\b|help\b)", re.IGNORECASE)
_SUPERLATIVE = re.compile(r"\b(?:most|least|highest|lowest|biggest|smallest|largest|best|worst|top|bottom|cheapest|"
                          r"dearest|furthest|fastest|slowest|first|last)\b", re.IGNORECASE)
_WHEN = re.compile(
    r"\b(?:today|yesterday|tomorrow|this|last|next|previous|past|latest|current|ytd|mtd|qtd|ttm|ltm|so far|since|until|"
    r"rolling|trailing|"
    r"between|(?:19|20)\d{2}|q[1-4]|t[1-4]\s+(?:19|20)\d{2}|h[12]|fy\s?\d{2,4}|january|february|march|april|may|june|july|august|september|"
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
# A ranking by a figure ("grew the most", "the lowest"), never a place in time ("last week", "first quarter").
_RANKING = re.compile(r"\b(?:most|least|highest|lowest|best|worst|biggest|smallest|largest|fastest|slowest)\b",
                      re.IGNORECASE)
# A figure of the question's own that no measure is named for: "the average visit duration by team".
_FIGURE = re.compile(r"\b(?:average|avg|mean|median|duration|percent(?:age)?|share\s+of|rate|ratio)\b", re.IGNORECASE)
_COUNTING = re.compile(r"\b(?:how\s+many|number\s+of|count|counts|counted)\b", re.IGNORECASE)
# "Which buyers raised the most purchase orders", "the fewest cancelled subscriptions": a thing counted right after a
# ranking word is the figure ranked, the question's own.
_RANKED_COUNT = r"\b(?:most|fewest|least|more|fewer|le\s+plus\s+de|le\s+moins\s+de)\s+(?:\w+\s+){{0,2}}{word}\b"
# "Which customers in France pay the most?", "who spent the most": a verb of money names the figure, whatever its
# measure is called.
# "the most purchase orders", "the fewest visits": things counted, by whatever name, right after the ranking word.
_RANKED_THINGS = re.compile(r"\b(?:most|fewest)\s+((?:[a-z][\w-]*\s+){0,2}[a-z][\w-]{2,}s)\b(?!\s+than)", re.IGNORECASE)
_MONEY_VERB = re.compile(r"\b(?:pay|pays|paid|paying|spend|spends|spent|spending|buy|buys|bought|sell|sells|sold|"
                         r"earn|earns|earned|billed|invoiced|d[ée]pens[ée]s?|d[ée]pensent|achet[ée]s?|ach[eè]tent|"
                         r"vendu[es]{0,2}|vendent|pay[ée]s?|payent|paient)\b",
                         re.IGNORECASE)
# "List our item groups", "what are the profit centres", "which divisions do we have": a list, which measures nothing.
_LISTS = re.compile(r"^\s*(?:list|name|enumerate|what\s+are\s+(?:the|our|all)|show\s+(?:me\s+)?(?:all|the\s+list)|"
                    r"(?:which|what)\s+\w+(?:\s+\w+)?\s+(?:are\s+there|do\s+we\s+have|exist))\b", re.IGNORECASE)
_GENERIC_TAIL = {"amount", "amt", "quantity", "qty", "value", "val", "count", "cnt", "number", "total",
                 # units a learned name keeps and people leave out: "utilization pct", "data usage gb"
                 "pct", "percent", "gb", "mb", "tb", "kb", "kg", "hrs", "mbps", "usd"}


@dataclass(frozen=True)
class Reading:
    kind: str           # refine | new | open (the AI reads it, with the conversation in view)
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


def _spans(model: SemanticModel | None, question: str,
           kinds: tuple[str, ...]) -> list[tuple[int, int, str, str, list[str]]]:
    """Every name of ``kinds`` the question says: (first word, after its last, kind, slug, its words)."""
    if model is None:
        return []
    tokens = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", question)]
    spans = []
    for kind, slug, phrase in _phrases_said(model):
        if kind not in kinds:
            continue
        size = len(phrase)
        for i in range(len(tokens) - size + 1):
            run = tokens[i:i + size]
            # A word of a question's shape ("on", "by") is no part of a name, unless the name has it too ("on hand").
            if not any(w in _SHAPE and w not in phrase for w in run) \
                    and all(_same(a, b, size > 1) for a, b in zip(run, phrase)):
                spans.append((i, i + size, kind, slug, run))
                break
    return spans


def _found(model: SemanticModel | None, question: str, kinds: tuple[str, ...]) -> list[tuple[str, str, list[str]]]:
    spans = _spans(model, question, kinds)
    # The longer name wins: "profit centre" is the field, not the measure "profit" inside it.
    return [(kind, slug, run) for start, end, kind, slug, run in spans
            if not any(s <= start and end <= e and (e - s) > (end - start) for s, e, *_ in spans)]


def named_longer(model: SemanticModel | None, question: str, slug: str) -> str | None:
    """The measure the question names by a longer name holding ``slug``'s own, where it says ``slug`` nowhere else:
    "how many ingredient lots" names "Number of ingredient lots", not the "Number of ingredients" inside it."""
    spans = _spans(model, question, ("measure", "count"))
    own = [(s, e) for s, e, _, found, _ in spans if found == slug]
    if not own:
        return None
    holding: set[str] = set()
    for start, end in own:
        around = {found for s, e, _, found, _ in spans
                  if found != slug and s <= start and end <= e and (e - s) > (end - start)}
        if not around:
            return None         # said on its own too ("orders and order lines")
        holding |= around
    return next(iter(holding)) if len(holding) == 1 else None


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


# "For all time", "overall": the reader sets the period to everything, which is still saying when.
_ALL_TIME = re.compile(r"\b(?:all\s+time|ever|overall|in\s+total|whole\s+period)\b", re.IGNORECASE)
# "That month", "the same period": the period of the answer above, pointed at rather than named. Never "the same
# time last year": a comparison with a year before the period asked ("YTD against the same time last year").
_THAT_PERIOD = re.compile(r"\b(?:that|the\s+same|same)\s+(?:day|week|month|quarter|half|year|fiscal\s+year|period|"
                          r"time|date|season)\b(?!\s+(?:(?:last|previous|prior)\s+year|a\s+year\s+(?:ago|before|"
                          r"earlier)|(?:the\s+)?year\s+before|l'an\s+(?:dernier|pass[ée]))\b)", re.IGNORECASE)
# A period named outright: a year, a quarter or half, a fiscal year, a month.
_NAMED_PERIOD = re.compile(
    r"\b(?:(?:19|20)\d{2}|q[1-4]|t[1-4]\s+(?:19|20)\d{2}|h[12]|fy\s?\d{2,4}|january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b", re.IGNORECASE)
# "Monthly", "by week", "over time": the question asks for a series.
_SERIES = re.compile(r"\b(?:daily|weekly|monthly|quarterly|yearly|annually|over\s+time|trend|trends|trending|"
                     r"(?:by|per|each|every|a)\s+(?:day|week|month|quarter|year)|"
                     r"(?:day|week|month|quarter|year)\s+(?:by|on|over)\s+(?:day|week|month|quarter|year)|"
                     r"(?:par|chaque)\s+(?:jour|semaine|mois|trimestre|an|ann[ée]e)(?!\s+(?:de|du|sur)\b)|"
                     r"mensuel(?:le)?s?|hebdomadaires?|trimestriel(?:le)?s?|annuel(?:le)?s?)\b",
                     re.IGNORECASE)


# The same in French, and a date written in figures ("le 27/05", "2026"): "Et au 27 mai ?" names a day.
_WHEN_FR = re.compile(r"\b(?:janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[uû]t|septembre|octobre|novembre|"
                      r"d[ée]cembre|mois|ann[ée]e|ann[ée]es|l'an|par\s+an|\d+\s+ans?|trimestre|semestre|semaine|jour|hier|"
                      r"aujourd'hui|depuis|exercice)\b", re.IGNORECASE)
# A period the question names itself, never a series word ("by month", "par mois"): a year, a quarter, a month,
# "last quarter", "this year", "so far", "for all time"; in French "l'an dernier", "cette année", "en mai".
_OWN_PERIOD = re.compile(
    r"\b(?:(?:this|last|next|previous|past|current|rolling|trailing)\s+(?:\d+\s+)?(?:fiscal\s+)?"
    r"(?:day|week|wk|month|mth|mo|quarter|qtr|half|year|yr)s?|ttm|ltm|"
    r"today|yesterday|ytd|mtd|qtd|year\s+to\s+date|so\s+far|since\s+\w+|until\s+\w+|right\s+now|currently|"
    r"at\s+the\s+end\s+of|all\s+time|ever|overall|whole\s+period|"
    r"(?:cette|cet|ce)\s+(?:ann[ée]e|an|mois|trimestre|semestre|semaine)|"
    r"(?:l'|la\s+|le\s+)(?:ann[ée]e|an|mois|trimestre|semestre|semaine)\s+(?:derni[eè]re?|pass[ée]e?|"
    r"pr[ée]c[ée]dente?|prochaine?|en\s+cours)|(?:\d+\s+)?derni(?:er|ère|ers|ères)\s+(?:mois|ann[ée]es?|jours|"
    r"semaines|trimestres)|hier|aujourd'hui|depuis\s+\w+|janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[uû]t|"
    r"septembre|octobre|novembre|d[ée]cembre)\b", re.IGNORECASE)
_DATE_FIGURES = re.compile(r"\b(?:\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?|(?:19|20)\d{2})\b")


def names_own_period(question: str) -> bool:
    """Does the question name a period of its own ("in 2025", "last quarter", "en mai", "for all time")? A series
    word ("by month", "monthly", "par mois") names none: it splits the period on screen."""
    return bool(_NAMED_PERIOD.search(question) or _DATE_FIGURES.search(question) or _OWN_PERIOD.search(question))


def keeps_period(question: str) -> bool:
    """Does a follow-up leave the period of the answer above as it was? It names none ("why did it grow?",
    "show the returns by warehouse"), or points at that one without naming another ("due that month")."""
    named = _NAMED_PERIOD.search(question) or _WHEN_FR.search(question) or _DATE_FIGURES.search(question)
    if not (_WHEN.search(question) or _ALL_TIME.search(question) or named):
        return True
    return bool(_THAT_PERIOD.search(question)) and not named


# "Did we hit target for the quarter?" after a Q1 answer: that quarter, said with "the".
_THE_PERIOD = re.compile(r"\b(?:for|in|over|during|across|of|on)\s+the\s+(day|week|month|quarter|half|year)\b"
                         r"(?!\s+(?:before|after|to\s+date|ahead|ending|so\s+far|just\s+gone))", re.IGNORECASE)


def the_period(question: str) -> str | None:
    """The unit of a period the question names with "the" ("for the quarter"): the one on screen when its window
    is one such period (the caller checks); None when it names none."""
    found = _THE_PERIOD.search(question)
    return found.group(1).lower() if found else None


def points_at_period(question: str) -> bool:
    """Does the question point at the period of the answer above ("due that month", "for the same period")?"""
    return bool(_THAT_PERIOD.search(question))


def asks_series(question: str) -> bool:
    """Does the question ask for figures period by period ("monthly", "by week", "over time")?"""
    return bool(_SERIES.search(question))


def narrowed(plan: Plan | None) -> bool:
    """Did the answer on screen narrow anything a fresh question would not carry: a filter or a period?"""
    if plan is None:
        return False
    window = plan.time.window
    return bool(plan.filters or window.kind != "all" or plan.time.compare is not None)


# Words of a measure's name that say nothing of what it is.
_PLAIN_MEASURE_WORDS = {"amount", "number", "total", "count", "value", "sum", "of", "the", "rows", "row", "quantity",
                        "qty", "invoice", "line", "lines", "net", "gross", "cad", "usd", "sop", "customer", "order"}


def _says_measure_on_screen(model: SemanticModel | None, words: str, previous: Plan) -> bool:
    """Does ``words`` name a measure of the answer on screen by a word of its own ("discount" of "invoice discount
    amount"), never one of the plain words every measure has?"""
    if model is None:
        return False
    said = set(re.findall(r"[a-z]+", words.lower()))
    for slug in previous.measures:
        measure = next((m for m in model.measures.values() if m.slug == slug), None)
        if measure is None:
            continue
        own = set(re.findall(r"[a-z]+", measure.business_name.lower())) - _PLAIN_MEASURE_WORDS
        if own & said or {w + "s" for w in own} & said:
            return True
    return False


# Words of a turn's head that say how to split, or when, never what is measured: "Break the first half of 2026
# down" names no figure, where "Spend" does.
_NO_FIGURE = re.compile(r"\b(?:break|split|cut|divide|slice|show|give|get|me|us|it|down|out|up|now|please|can|could|"
                        r"you|the|a|an|of|for|in|on|over|during|and|then|first|second|third|fourth|half|halves|"
                        r"full|whole|same|that|this|those|these)\b", re.IGNORECASE)


# "... by total spend?", "by revenue": a figure's word ends the question ("by sales rep" is a field).
_BY_FIGURE = re.compile(r"\b(?:spend|spending|spent|revenue|revenues|sales|turnover|amount|value|cost|costs|profit|"
                        r"profits|margin|volume|takings|income|billing|billings)\s*[?.!]*\s*$", re.IGNORECASE)


def _names_a_figure(head: str) -> bool:
    """Does the part of a turn before "by" hold a word for what is measured, once the words of a period and of a
    split command are taken out?"""
    rest = _NO_FIGURE.sub(" ", _WHEN.sub(" ", _NAMED_PERIOD.sub(" ", head)))
    return any(len(w) >= 3 for w in re.findall(r"[^\W\d_]+", rest))


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
    if _WHY_REASON.match(question):
        return Reading("open", "it asks the reasons people had, not why the figure moved")
    if _WHY.match(question) and (short or pointing):
        return Reading("refine", "it asks why the answer on screen is what it is")
    if _BEHIND.search(question):
        return Reading("refine", "it asks what is behind the change on screen")
    if _POINTS_FIRST.match(question) or short and pointing:
        return Reading("refine", "it points at the answer on screen")
    if _POINTS_AT_MEMBERS.search(question):
        return Reading("refine", "it points at the members of the answer on screen")
    if points_at_period(question):
        return Reading("refine", "it points at the period of the answer on screen")
    if short and _CARRY.search(question):
        return Reading("refine", "it carries the answer on screen along")
    found = _found(model, question, ("measure", "count", "field", "date"))
    kinds = {"measure" if kind == "count" else kind for kind, _, _ in found}
    said = {w for kind, _, run in found if kind == "measure" for w in run}
    # A member name that is a word of a measure it names ("paid" of "paid amount") narrows nothing.
    texts = members if isinstance(members, list) else []
    own_members = [m for m in texts if not set(re.findall(r"[a-z0-9]+", m.lower())) <= said] if texts else []
    own = scoped(question, len(own_members) if texts else int(members or 0))
    leading = bool(_LEADING.match(question))
    own_period = names_own_period(question)
    # "By store, what were net sales in 2025?", "Pour 2025, quelles sont les ventes par magasin ?": a long question
    # that opens on its scope and names its own period sets out afresh; it does not carry the answer on.
    scope_opened = bool(_SCOPE_LEAD.match(question)) and len(words) > 6 and own_period
    rest = question.split(",", 1)[1] if scope_opened and "," in question else ""
    asks = bool(_ASKS.match(question) or (rest and _ASKS.match(rest)))
    if scope_opened:
        leading = False
    period = bool(_WHEN.search(question) or _WHEN_FR.search(question))
    grouped = bool(_GROUPED.search(question))
    # What the turn is about comes before its split ("tickets" in "tickets by category"); what follows "by" is
    # how to split it, never what is measured.
    head = re.split(r"\b(?:by|per|for|in|during|over)\b", question, maxsplit=1, flags=re.IGNORECASE)[0]
    head_found = _found(model, head, ("measure", "count", "field"))
    head_measure = any(k == "measure" for k, _, _ in head_found)      # a measure by name, not a thing counted
    head_members = [m for m in own_members if m.lower() in head.lower()]
    which = _WHICH.match(question)
    # "Top 10 customers by total spend": what follows "by" is the figure they are ranked by, no field to split by.
    after_by = re.split(r"\b(?:by|per)\b", question, maxsplit=1, flags=re.IGNORECASE)
    by_figure = model is not None and len(after_by) == 2 and not _found(model, after_by[1], ("field", "date")) \
        and not _WHEN.search(after_by[1]) and (bool(_found(model, after_by[1], ("measure",)))
                                               or bool(_BY_FIGURE.search(after_by[1])))
    if asks and not scoped(question) and len(words) <= 6 and _SUPERLATIVE.search(question):
        # "which store was lowest?": the store is what is ranked, not what is measured
        rest = question[which.end():] if which else question
        if measures_named(model, rest) <= set(previous.measures):
            return Reading("refine", "it asks which of the answer on screen comes first or last")
    if not asks and head_members and not head_measure:
        return Reading("refine", f"it narrows the answer on screen to {head_members[0]}")
    if which and model is not None and _found(model, which.group(1), ("field",)):
        own = True      # "which warehouse type ..." ranks by warehouse type: its own grouping
    # "Which division grew the most compared with H1 2025?" names a field and a period but no figure: the ranking
    # is of the figure on screen (below), never a new question that has none.
    ranked_counts = {slug for k, slug, run in found if k == "count" and re.search(
        _RANKED_COUNT.format(word=re.escape(run[0])), question, re.IGNORECASE)}
    ranked_things = _RANKED_THINGS.search(question)
    ranked_count = bool(ranked_counts) or bool(ranked_things)
    ranks_the_figure = bool(previous.measures and _RANKING.search(question)) and not (
        any(k == "measure" for k, _, _ in found) or _FIGURE.search(question) or ranked_count
        or _MONEY_VERB.search(question))
    if not leading and not pointing and len(words) >= 4 and period and (head_found or asks and grouped) \
            and not ranks_the_figure:
        return Reading("new", "it names what it measures, how to split it and for when")
    # "Which plans have the most cancelled subscriptions?" after tax by country, "Which customers in France pay the
    # most?" after a count of subscriptions: the members of a field of its own, ranked by a figure that is not the
    # one on screen, ask afresh.
    if which and not leading and not pointing and model is not None and _found(model, which.group(1), ("field",)):
        money_on_screen = any(getattr(model.measures.get(m), "format", "") == "currency" for m in previous.measures)
        things = set(re.findall(r"[a-z]+", ranked_things.group(1).lower())) if ranked_things else set()
        named_on_screen = any(things & set(m.split("_")) for m in previous.measures)
        if ranked_counts and not ranked_counts & set(previous.measures) \
                or ranked_things and not ranked_counts and not named_on_screen \
                or _MONEY_VERB.search(question) and _RANKING.search(question) and not money_on_screen:
            return Reading("new", "it ranks the members of a field by a figure of its own")
    # "Spend by supplier country for 2025": a field of the data to split by and a period of its own make a whole
    # question, whatever word it uses for the figure ("spend" is no measure's name): read as new, it keeps nothing
    # of the answer on screen (the supplier it was narrowed to, its months).
    split = re.split(r"\b(?:by|per)\b", question, maxsplit=1, flags=re.IGNORECASE)
    if not leading and not pointing and len(words) >= 4 and period and len(split) == 2 and not ranks_the_figure \
            and _names_a_figure(split[0]) and _found(model, split[1], ("field",)):
        return Reading("new", "it names how to split it and for when")
    if _CONTINUE.match(question) and not scope_opened:
        return Reading("refine", "it starts by carrying on from the answer on screen")
    # Whether the answer on screen set a scope a follow-up keeps (a filter or a period), and whether the question
    # says its own period ("in 2025", "for all time"), which replaces it.
    was_narrowed = narrowed(previous) if previous_narrowed is None else previous_narrowed
    if not asks and not leading:
        if grouped and head_found:
            if was_narrowed and not own_period:
                # "avg basket by education level?" after store sales by marital status in 2002: the same subject
                # split another way, or another one: the AI reads it with the conversation in view, as a person
                # would. A follow-up keeps the filters and the period from above.
                return Reading("open", "it names a figure and a split but no period, after a narrowed answer")
            return Reading("new", "it names a measure or a thing and how to split it")
        if short and not head_measure and not any(k == "count" for k, _, _ in head_found):
            return Reading("refine", "a fragment that only changes the answer on screen")
    if (kinds & {"measure", "field"}) and not pointing:
        # A thing counted names a figure only when it is counted ("how many profit centres"), never when it is
        # how the answer is split ("by profit centre").
        measured = any(k == "measure" for k, _, _ in found) or any(k == "count" for k, _, _ in found) \
            and _COUNTING.search(question) or bool(_FIGURE.search(question)) or by_figure or ranked_count \
            or bool(_RANKING.search(question) and _MONEY_VERB.search(question))
        # "Which devices are firewalls?" lists: only a split or a ranking word is of the figure on screen.
        if not measured and previous.measures and not _LISTS.match(question) \
                and (grouped or _RANKING.search(question)):
            # "Break the first half of 2026 down by profit centre", "which division grew the most?": a split or a
            # ranking that names no figure of its own is of the figure on screen (planned afresh, it had none).
            return Reading("refine", "it splits or ranks the answer on screen and names no figure of its own")
        if own and (grouped or which) and not period and not own_members and not _CONDITION.search(question) \
                and not by_figure \
                and (measures_named(model, head if grouped else question) & set(previous.measures)
                     or which and _says_measure_on_screen(model, question[which.end():], previous)) \
                and previous.time.window.kind != "all":
            # "Show the returns by warehouse" after the returns of Q1, "which customer types got the most discount?"
            # after the discount of 2025: a measure on screen split or ranked another way, for no period of its
            # own, keeps the answer's ("Ask as a new question" is one click away).
            return Reading("refine", "it splits a measure of the answer on screen and names no period of its own")
        # "Who are our top 10 customers by total spend?" after the inactive customers: ranked by a figure of its own
        # (not the one on screen), the question stands on its own, whatever it leaves unsaid.
        by_own_figure = by_figure and not (measures_named(model, after_by[1]) & set(previous.measures))
        if own and (own_period or by_own_figure or not was_narrowed):
            return Reading("new", "it names what it measures and its own scope")
        if was_narrowed:
            # "How many orders?" after net sales for North in March, "what is our margin by category?" after a
            # store's discounts in Q1: the same subject going on (a follow-up, keeping North and March) or another
            # one (new). Never asked back: the AI decides with the conversation in view, as a person would.
            return Reading("open", "a complete question after a narrowed answer: the conversation decides")
        return Reading("new", "it names what it measures, and the answer on screen narrowed nothing")
    return Reading("open", "left to the AI")
