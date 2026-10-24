"""Business asks: what a business asks of its figures in its own words, planned in code.

"What is my revenue comparison?", "How are sales doing vs last year?", "Are we on track to beat last year?", "Which
stores grew the most vs last year?", "Top 5 stores this year and what they sold last year", "What is our run rate?":
each means one thing to a business, and the AI planned each its own way -- it asked back which figure, compared 2025
with 2024, took the fiscal year, set the last 12 months against themselves and this month against the 15 days before
it. The AI now reads which figure the question names and what it is split by; the periods and the shape are set here:

* a comparison ("comparison", "compare", "how are we doing", "how is spend tracking", "on track", "vs last year"):
  this year so far against the same days last year -- month by month when it asks for a comparison or how it is
  doing, one figure otherwise and when it names the year to date ("YTD", "so far");
* who grew or fell ("which stores grew the most vs last year"): each member over the same two periods, the biggest
  rise (or fall) first;
* this year and last year named together ("top 5 stores this year and what they sold last year"): this year so far,
  with the same days last year beside it;
* a run rate: a year at the pace of the last three whole months (their total, times four);
* "the same days last month", "the 12 before": the period before, of the same shape -- never the period itself, nor
  the days just before it.

Another unit ("compared with last month", "vs last quarter") compares that unit so far with the same days of the one
before. A question naming its own dates ("2025 vs 2024", "Q1 against Q4", "over the last 12 months") keeps them.
When the AI asks back which figure such a question means, the figure is picked here (the one a business means, never
a target, a cost or a discount it does not name); the answer says which, and offers the others.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from core2.model.schema import Measure, SemanticModel
from core2.plan.ir import Compare, Derived, Figure, Plan, Sort, TimeSpec, Window

_UNIT = {"year": "year", "yr": "year", "month": "month", "mth": "month", "quarter": "quarter", "qtr": "quarter",
         "week": "week", "wk": "week", "an": "year", "annee": "year", "année": "year", "mois": "month",
         "trimestre": "quarter", "semaine": "week", "ytd": "year", "mtd": "month", "qtd": "quarter", "wtd": "week"}

# The period before, as a reader names it: "last year", "the previous month", "a year ago", "l'an dernier".
_LAST = (r"(?:the\s+)?(?:last|previous|prior)\s+(year|yr|month|mth|quarter|qtr|week|wk)(?!\w)|"
         r"a\s+(year|month)\s+(?:ago|before|earlier)|l['’](an|ann[ée]e)\s+(?:derni[eè]re?|pr[ée]c[ée]dente?|pass[ée]e?)|"
         r"(?:le\s+)?(mois|trimestre)\s+(?:dernier|pr[ée]c[ée]dent)|la\s+(semaine)\s+(?:derni[eè]re|pr[ée]c[ée]dente)")
_LAST_RE = re.compile(rf"\b(?:{_LAST})", re.IGNORECASE)
# Set against it: "vs last year", "compared with the same days last month", "than last year", "beat last year's",
# "par rapport à l'an dernier", "avec l'an dernier".
_AGAINST_LAST = re.compile(
    r"\b(?:vs\.?|versus|against|compared\s+(?:to|with)|comparison\s+(?:to|with)|with|than|on|over|beat(?:ing)?|from|"
    r"par\s+rapport\s+(?:à|a)|contre|compar[ée]e?s?\s+(?:à|a|avec)|avec|face\s+(?:à|a))\s+"
    r"(?:(?:the|la|le)\s+)?(?:(?:same|m[êe]mes?)\s+(?:time|period|p[ée]riode|days?|dates?|jours?|months?)\s+"
    rf"(?:of\s+|in\s+|de\s+|du\s+)?)?(?:{_LAST})", re.IGNORECASE)
_OVER_YEAR = re.compile(r"\b(?:year[\s-]+(?:on|over)[\s-]+year|yoy|y/y|d['’]une\s+ann[ée]e\s+sur\s+l['’]autre)\b",
                        re.IGNORECASE)
_OVER_MONTH = re.compile(r"\b(?:month[\s-]+(?:on|over)[\s-]+month|m/m)\b", re.IGNORECASE)
# This period so far: "this year", "the current month", "year to date", "YTD", "cette année", "ce mois-ci".
_THIS = re.compile(r"\b(?:this|current)\s+(year|month|quarter|week)\b|\b(year|month|quarter|week)[\s-]+to[\s-]+date\b|"
                   r"\b(ytd|mtd|qtd|wtd)\b|\bcette\s+(ann[ée]e|semaine)\b|\bce\s+(mois|trimestre)\b", re.IGNORECASE)
# The period to date named as such ("how does our YTD compare with last year?"): one total against the same days.
_TO_DATE = re.compile(r"\b(?:ytd|mtd|qtd|wtd|fytd)\b|\b(?:year|month|quarter|week)[\s-]+to[\s-]+date\b|\bso\s+far\b|"
                      r"\b[àa]\s+ce\s+jour\b|\bdepuis\s+le\s+d[ée]but\b|\bjusqu['’]\s*(?:ici|[àa]\s+pr[ée]sent)\b",
                      re.IGNORECASE)
# "My revenue comparison" is a comparison with the period before; "compare North and South" is not one.
_COMPARISON = re.compile(r"\bcompar(?:ison|isons|aison|aisons)\b", re.IGNORECASE)
_COMPARE = re.compile(r"\bcompare\b|\bcomparer\b", re.IGNORECASE)       # "compare spend with last year", not "compared"
# "Which stores ...", "quels magasins ...": members, never a month-by-month comparison ("quelle est" is "what is").
_WHICH = re.compile(r"^\s*(?:which|who|quels|quelles)\b", re.IGNORECASE)
_MEMBERS_SIDE_BY_SIDE = re.compile(r"\b(?:and|between|or|et|entre|ou)\b|&", re.IGNORECASE)
_HOW_DOING = re.compile(r"\bhow\s+(?:are|is|am|were|was|have|has)\b[^?.!]{0,60}?\b(?:doing|tracking|performing|"
                        r"going|faring|looking|shaping\s+up)\b|\bcomment\s+(?:vont|va|se\s+porte(?:nt)?|"
                        r"[ée]volue(?:nt)?)\b|\bo[uù]\s+en\s+(?:sont|est|sommes)[\s-]", re.IGNORECASE)
_ON_TRACK = re.compile(r"\bon\s+(?:track|course|pace)\b|\bahead\s+of\s+last\b|\bbehind\s+last\b|"
                       r"\ben\s+(?:bonne\s+)?voie\b", re.IGNORECASE)
_GREW = re.compile(r"\b(?:grew|grown|grow(?:s|ing)?|growth|increas(?:e|ed|es|ing)|rose|risen|improv(?:e|ed|es|ing)|"
                   r"gain(?:ed|s)?|more\s+(?:\w+\s+)?than\s+last|progress[ée](?:s|nt)?|augment[ée](?:s|nt)?|"
                   r"(?:ont|a)\s+cr[uû]|en\s+hausse)\b", re.IGNORECASE)
_FELL = re.compile(r"\b(?:fell|fallen|fall(?:s|ing)?|declin(?:e|ed|es|ing)|drop(?:ped|s)?|decreas(?:e|ed|es|ing)|"
                   r"shr[aiu]nk(?:ing)?|less\s+(?:\w+\s+)?than\s+last|baiss[ée](?:s|nt)?|recul[ée](?:s|nt)?|"
                   r"diminu[ée](?:s|nt)?|en\s+baisse)\b", re.IGNORECASE)
# "Drop the share column", "just drop this", "drop last year", "drop by store": a command opening its clause takes
# something off the answer, no fall ("where did sales drop the most", "a drop in sales", "then drop by 10%", "drop by
# store vs last year" and "then drop sharply" are falls).
_DROP_SAID = re.compile(r"(?:^|[,;.!?]|\band\b|\bthen\b)\s*(?:(?:please|now|just|also|ok(?:ay)?|and|then|so)\s+)*"
                        r"drop\b(?!\s+(?:in|of|below|under|compared|vs|versus|against|over|during)\b)"
                        r"(?!\s+off\b(?!\s+(?:the|these|those)\b|\s+(?:this|that)\s+(?!(?:year|month|quarter|week|"
                        r"period|season|time)\b)))"
                        r"(?!\s+by\s+(?:[$€£]?\d|(?:about|around|over|more|less|nearly|almost|half|one|two|three|four|"
                        r"five|six|seven|eight|nine|ten|twenty|fifty|a\s+(?:lot|third|quarter|fifth|bit))\b))"
                        r"(?!\s+(?:to|from)\s+(?:[$€£]\s?\d|\d[\d.,]*\s?(?:%|percent|[kmb]\b)))"
                        r"(?!\s+(?:(?:sharply|significantly|steeply|dramatically)\b|a\s+lot\b|the\s+most\s*(?:[?.!,]|$|"
                        r"(?:in|this|last|since|over|from|between)\b)))"
                        r"(?!(?!\s+(?:the|this|that|these|those|it|them|its|their|my|our)\b)"
                        r"(?:(?!\b(?:and|then)\b)[^,.;?!])*\b(?:vs\.?|versus|compared|against|year\s+over\s+year|yoy|"
                        r"from\s+(?:last|the\s+previous|previous|a\s+year))\b)(?!\s*(?:[?.!,]|$))", re.IGNORECASE)
# A sentence that asks ("which products did well last year and drop this year") goes on with a fall when "and" or
# "then" joins it; after a comma, or opening its own sentence, "drop" is a command ("which stores sold the most, and
# drop the share column").
_ASKS = re.compile(r"^\s*(?:which|what|where|who|whose|when|how|did|do|does|has|have|is|are|was|were|"
                   r"(?:(?:show|list|give)\s+(?:me\s+)?)?(?:the\s+|all\s+)?\w+\s+(?:whose|that|where|which|who))\b",
                   re.IGNORECASE)


def _without_drops(q: str, *, what: bool = False) -> str:
    """The question with each command to drop something off the answer taken out ("drop the share column");
    ``what``: each one marked instead, so what it drops is taken out with it (``_taken_off``)."""
    cut = _CUT if what else " "

    def taken(m: re.Match) -> str:
        if m.group(0).lstrip()[:1] in ",;.!?":
            # Its own clause: a command, unless a question goes on with a period ("..., then drop this year?").
            follows = q[m.end():]
            asked = _ASKS.match(q[:m.start()]) and re.match(
                r"\s+(?:(?:this|last|next)\s+(?:year|month|quarter|week|period|season|time)|(?:in|during)\s+(?:[\w-]+\s+){0,2}?"
                r"(?:(?:19|20)\d{2}|q[1-4]|jan\w*|feb\w*|mar\w*|apr\w*|may|jun\w*|jul\w*|aug\w*|sep\w*|oct\w*|nov\w*|"
                r"dec\w*))\s*\?\s*$", follows, re.IGNORECASE)
            return m.group(0) if asked else (m.group(0)[:1] + _CUT if what else " ")
        start = max(q.rfind(c, 0, m.start() + 1) for c in ".?!;") + 1
        before = q[start:m.start()]
        return m.group(0) if before.strip() and _ASKS.match(before) else cut
    return _DROP_SAID.sub(taken, q)


# What a command takes off the answer, to the end of its clause: "drop the vs last year column", "remove the change
# and add units" (the change only), "hide the comparison", "without the vs last year". The words of what it takes off
# ask for nothing: "drop the vs last year column" is no comparison, "remove the run rate" no run rate. A command
# takes off what it names on the answer ("remove the ...", "hide last year"), never rows asked for beside a
# comparison ("remove cancelled orders vs last year", "take out orders this year vs last year": take-out orders).
_CUT = "\x00"
_IN_CLAUSE = r"(?:[^,;.!?\x00]|(?<=vs)\.)"            # "vs." is no end of the clause
_CLAUSE_END = r"(?=\s+(?:and|then|et|puis)\b|[,;!?]|(?<!vs)\.|$)"
_DROPPED_WHAT = re.compile(_CUT + _IN_CLAUSE + r"*?" + _CLAUSE_END, re.IGNORECASE)
# "Can you remove ...", "could you please hide ...", "pouvez-vous enlever ...": the command after its courtesy.
_POLITE = re.compile(r"^\s*(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|please\s+|kindly\s+|"
                     r"(?:pouvez|pourriez)[\s-]vous\s+|(?:peux|pourrais)[\s-]tu\s+|merci\s+de\s+)", re.IGNORECASE)
_DETERMINER = r"(?:the|that|this|these|those|it|them|its|their|our|my|la|le|les|l['’]|cette|ce|ces)"
# What names the comparison itself: the period compared with, or the comparison.
_COMPARISON_NAMED = (r"(?:vs\.?|versus|comparisons?|compar\w*|last|prior|previous|yoy|y/y|year|ann[ée]e|an\b|columns?|colonnes?|"
                     r"n-1)")
_TAKEN_OFF = re.compile(
    r"(?:^|(?<=[,;.!?])|\band\b|\bthen\b|\bet\b|\bpuis\b)\s*(?:(?:please|now|just|also|ok(?:ay)?|and|then|so)\s+)*"
    r"(?:remove|hide|take\s+(?:off|out)|leave\s+out|get\s+rid\s+of|enl[eè]ve[rz]?|retire[rz]?|masque[rz]?|"
    r"supprime[rz]?)\s+"
    # Its object names the comparison at once ("the vs last year column", "last year"), or it is something on the
    # answer with no comparison asked beside it in its clause.
    r"(?:(?=(?:" + _DETERMINER + r"\s*)?" + _COMPARISON_NAMED + r"\b)|(?=" + _DETERMINER + r"\b)(?!" + _IN_CLAUSE
    + r"*\b(?:vs\.?|versus|compared|against|par\s+rapport)\b))" + _IN_CLAUSE + r"*?" + _CLAUSE_END, re.IGNORECASE)
_WITHOUT_COMPARED = re.compile(
    r"\b(?:without|sans)\s+(?:the\s+|any\s+|a\s+|la\s+|le\s+|les\s+)?(?:vs\.?|versus|comparisons?|compar\w*|change|"
    r"yoy|y/y|year[\s-]+over[\s-]+year|(?:last|prior|previous)\s+year(?:'s|’s)?|year\s+before|l['’]an\s+dernier|"
    r"l['’]ann[ée]e\s+(?:derni[èe]re|pr[ée]c[ée]dente))\b" + _IN_CLAUSE + r"*?" + _CLAUSE_END, re.IGNORECASE)


def _taken_off(q: str) -> tuple[str, list[str]]:
    """(the question without what its commands take off the answer, what they take off)."""
    gone: list[str] = []

    def cut(m: re.Match) -> str:
        gone.append(m.group(0).lstrip(_CUT))
        return " "
    q = _without_drops(_POLITE.sub("", q or "", count=1), what=True)
    q = _DROPPED_WHAT.sub(cut, q)
    q = _TAKEN_OFF.sub(cut, q)
    return _WITHOUT_COMPARED.sub(cut, q), gone


def without_taken_off(q: str) -> str:
    """The question without what its commands take off the answer: their words ask for nothing."""
    return _taken_off(q)[0]


# The period compared with, named: what taking off the comparison names ("the vs last year column", "the year
# before", "the comparison", "l'année précédente").
_PERIOD_COMPARED = re.compile(r"\b(?:vs\.?|versus|against|compar\w*|yoy|y/y|year[\s-]+over[\s-]+year|"
                              r"(?:last|prior|previous)\s+(?:year|month|quarter|week|period)|"
                              r"(?:year|month|quarter|week|period)\s+before|l['’]an\s+dernier|"
                              r"l['’]ann[ée]e\s+(?:derni[èe]re|pr[ée]c[ée]dente)|n-1)\b", re.IGNORECASE)
# The change between the two periods, taken off alone: "drop the % change vs last year" keeps both years.
_THE_CHANGE = re.compile(r"%|\b(?:change|difference|diff|growth|variance|variation|[ée]cart|[ée]volution)\b",
                         re.IGNORECASE)


def takes_off_comparison(question: str) -> bool:
    """Does a command take the comparison off the answer ("drop the vs last year column", "hide last year")?
    Never the change alone ("drop the % change column", "drop the % change vs last year" keep both years)."""
    q = question or ""
    kept, gone = _taken_off(q)
    if not _PERIOD_COMPARED.search(q) or _PERIOD_COMPARED.search(kept):
        return False
    return not any(_THE_CHANGE.search(g) for g in gone if _PERIOD_COMPARED.search(g))


# "Are we growing?", "Is spend going up?": yes or no about the whole, said against the period before.
_YES_NO = re.compile(r"^\s*(?:and\s+|so\s+)?(?:are|is|have|has|did|was|were)\b|^\s*(?:est-ce\s+que|sommes-nous|"
                     r"avons-nous|est-ce)\b", re.IGNORECASE)
_RUN_RATE = re.compile(r"\brun[\s-]?rate\b|\bannuali[sz](?:ed|e|es|ing)\b|\brythme\s+annuel\b|\bannualis[ée]e?s?\b",
                       re.IGNORECASE)
_FISCAL = re.compile(r"\b(?:fiscal|fy\s?'?\d{0,4}|fytd|exercice|financial\s+year)\b", re.IGNORECASE)
# A period of the question's own: its dates, or a span counted out ("the last 12 months", "past 90 days").
_OWN_DATES = re.compile(
    r"\b(?:(?:19|20)\d\d|fy\s?\d{2,4}|q[1-4]|h[12]|t[1-4]|s[12]|jan(?:uary)?|feb(?:ruary)?|march|apr(?:il)?|june?|"
    r"july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|janvier|f[ée]vrier|mars|avril|"
    r"juin|juillet|ao[uû]t|septembre|octobre|novembre|d[ée]cembre)\b|"
    r"\b(?:last|past|previous|prior|trailing|next|rolling)\s+(?:\d+|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|eighteen|twenty)\b|\b\d+\s+(?:days?|weeks?|months?|quarters?|years?|jours|semaines|mois|"
    r"trimestres|ans)\b|\b(?:ttm|ltm|trailing\s+(?:year|twelve))\b|\b(?:since|between|until)\b", re.IGNORECASE)

# "The same days last month", "the same period of the previous quarter": the unit so far against the one before.
_SAME_DAYS_BEFORE = re.compile(
    r"\bsame\s+(?:days?|dates?|period|time)\s+(?:of\s+|in\s+)?(?:the\s+)?(?:last|previous|prior)\s+(month|quarter|"
    r"week)\b|\bm[êe]mes\s+jours\s+(?:du|de\s+la)\s+(mois|trimestre|semaine)\s+(?:dernier|derni[èe]re|"
    r"pr[ée]c[ée]dente?)\b", re.IGNORECASE)
# "The 12 before", "the 12 months before", "the previous 12": the span before a span counted out.
_SPAN_BEFORE = re.compile(
    r"\b(?:the|les)\s+(?:\d+|two|three|four|six|twelve)\s+(?:\w+\s+)?(?:before|prior|earlier|preceding|"
    r"pr[ée]c[ée]dents?|d['’]avant)\b|\bthe\s+(?:previous|prior|preceding)\s+(?:\d+|two|three|four|six|twelve)\b|"
    r"\b(?:the\s+)?(?:period|span)\s+before\b", re.IGNORECASE)

# What a figure is planned to be, never what happened: chosen only when the question says so.
_PLANNED = re.compile(r"\b(?:target|targets|budget|budgets|budgeted|quota|quotas|forecast|forecasts|objective|"
                      r"objectives|goal|goals|plan|planned|cible|cibles|objectif|objectifs|pr[ée]vision|"
                      r"pr[ée]visions)\b", re.IGNORECASE)
# What a business takes off or pays out beside its revenue: never "the" figure of a comparison unless named.
_BESIDE = re.compile(r"\b(?:cost|costs|cogs|discount|discounts|tax|taxes|vat|refund|refunds|fee|fees|freight|"
                     r"shipping|co[uû]t|co[uû]ts|remise|remises|taxe|taxes|frais)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Ask:
    kind: Literal["compare", "run_rate"]
    unit: str = "year"           # the unit so far, against the one before
    said: bool = False           # the question sets it against the period before in so many words
    by_month: bool = False       # "comparison", "how are we doing": month by month
    moved: bool = False          # grew, rose, fell: who moved most, when the answer is split
    falls: bool = False          # the biggest fall first


def _unit_of(match: "re.Match[str] | None") -> str | None:
    if match is None:
        return None
    word = next((g for g in match.groups() if g), "")
    word = word.casefold().replace("é", "e")
    return _UNIT.get(word, _UNIT.get(word.rstrip("s")))


def ask_of(question: str) -> Ask | None:
    """The business ask the question's words make, or None. What a command takes off the answer asks for nothing
    ("drop the vs last year column" keeps the report, never makes it a comparison)."""
    q = without_taken_off(question or "")
    if _RUN_RATE.search(q):
        return Ask("run_rate")
    if _OWN_DATES.search(q):
        return None
    unit = _unit_of(_AGAINST_LAST.search(q))
    if unit is None and _OVER_YEAR.search(q):
        unit = "year"
    if unit is None and _OVER_MONTH.search(q):
        unit = "month"
    lasts = [u for u in (_unit_of(m) for m in _LAST_RE.finditer(q)) if u]
    this = _unit_of(_THIS.search(q))
    if unit is None and this and this in lasts:
        unit = this                          # "this year ... last year"
    comparing = bool(_COMPARISON.search(q) or _HOW_DOING.search(q) or _ON_TRACK.search(q)) and \
        not _MEMBERS_SIDE_BY_SIDE.search(q)
    grew, fell = bool(_GREW.search(q)), bool(_FELL.search(_without_drops(q)))
    if unit is None and lasts and (comparing or grew or fell):
        unit = lasts[0]                      # "compare revenue with last month"
    said = unit is not None or comparing or ((grew or fell) and bool(_YES_NO.search(q)))
    if not said and not (grew or fell):
        return None
    unit = unit or "year"
    by_month = unit == "year" and bool(_COMPARISON.search(q) or _COMPARE.search(q) or _HOW_DOING.search(q)) \
        and not _WHICH.search(q) and not _TO_DATE.search(q)
    return Ask("compare", unit, said=said, by_month=by_month, moved=grew or fell, falls=fell and not grew)


def compares(question: str) -> bool:
    """Does the question set a period against the one before ("this year and what they sold last year")?"""
    ask = ask_of(question)
    return ask is not None and ask.kind == "compare" and ask.said


def month_by_month(question: str) -> bool:
    """Does the question ask for a comparison month by month ("what is my revenue comparison?")?"""
    ask = ask_of(question)
    return ask is not None and ask.by_month


_TOP = re.compile(r"\b(?:top|first|best|leading|biggest|largest)\s+\d+\b|\b\d+\s+(?:premiers|premières|meilleurs)\b",
                  re.IGNORECASE)
_SPLIT_WORDS = re.compile(r"\b(?:which|who|top|per|each|every|bottom|rank|ranked|ranking|quels?|quelles?|par|chaque)\b",
                          re.IGNORECASE)
_BY = re.compile(r"\bby\s+(?:the\s+|their\s+|its\s+)?([a-z][a-z_]+)", re.IGNORECASE)


def _measured_by(plan: Plan, model: SemanticModel, question: str) -> bool:
    """Does every "by X" of the question name the figure counted ("by principal") rather than a split?"""
    found = [m.group(1).casefold() for m in _BY.finditer(question or "")]
    if not found:
        return False
    named = [m for m in model.measures.values() if m.slug in plan.measures]
    words = {w for m in named for w in re.findall(r"[a-z]+", f"{m.business_name} {m.slug}".casefold())}
    return all(word.rstrip("s") in words or word in words for word in found)


def settled(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """The plan with the periods and the shape the business ask means (see the module's words); as it is for any
    other question, and for a plan with an analytical shape of its own."""
    ask = ask_of(question)
    if ask is None or plan.kind != "query" or plan.target is not None or plan.forecast is not None \
            or plan.drivers is not None or plan.intent in ("drivers", "forecast", "list") or plan.segment is not None:
        return plan
    if ask.kind == "run_rate":
        # A moving sum the AI drew for it is the run rate's own months.
        bare = plan.model_copy(update={"rolling": None}) if plan.rolling is not None else plan
        return plan if bare.shaped() else (run_rate(bare, model, question) or plan)
    if plan.shaped():
        return plan
    members = [g for g in plan.group_by if not g.startswith("time:")]
    if not ask.said and not (ask.moved and members):
        return plan                          # "revenue growth by month": the AI's period over period stands
    fiscal = bool(_FISCAL.search(question)) and ask.unit in ("year", "quarter")
    window = Window(kind="to_date", unit=ask.unit, fiscal=fiscal)
    compare = Compare(kind="same_period_last_year" if ask.unit == "year" else "previous_period")
    grain = plan.time.grain
    if ask.by_month and not members and grain is None:
        grain = "fiscal_month" if fiscal else "month"
    update: dict = {"time": plan.time.model_copy(update={"window": window, "compare": compare, "grain": grain}),
                    "intent": "compare"}
    if ask.moved and members and not any(s.by in ("change", "pct_change") for s in plan.sort):
        update["sort"] = [Sort(by="change", desc=not ask.falls)]
    if not ask.moved and members and plan.limit and _TOP.search(question) and plan.measures \
            and any(s.by in ("change", "pct_change") for s in plan.sort):
        # "Top 3 partners by bookings this year and their bookings last year": the first 3 by this year's figure.
        update["sort"] = [Sort(by=plan.measures[0], desc=True)]
    if members and not _SPLIT_WORDS.search(question) and _measured_by(plan, model, question):
        # "Loans funded this month so far vs the same days last month, by principal" was split by loan: "by
        # principal" says what is counted, and the question asks for no split.
        update.update({"group_by": [g for g in plan.group_by if g.startswith("time:")], "limit": None,
                       "sort": [s for s in update.get("sort", plan.sort) if s.by not in plan.group_by]})
    changed = plan.model_copy(update=update)
    return plan if changed == plan else changed


# The readings of a comparison, each a click away from the others: (words, window, compare, grain).
READINGS = (
    ("This year so far against the same days last year", Window(kind="to_date", unit="year"),
     Compare(kind="same_period_last_year"), "month"),
    ("Last 12 months against the 12 before", Window(kind="last", unit="month", n=12),
     Compare(kind="same_period_last_year"), "month"),
    ("Last month against the month before", Window(kind="last", unit="month", n=1), Compare(kind="previous_period"),
     None),
)

READINGS_FR = {
    "This year so far against the same days last year": "Cette année à ce jour face aux mêmes jours l'an dernier",
    "Last 12 months against the 12 before": "Les 12 derniers mois face aux 12 d'avant",
    "Last month against the month before": "Le mois dernier face au mois d'avant",
}


def other_readings(plan: Plan, model: SemanticModel, question: str) -> list[tuple[str, Plan]]:
    """A comparison read as this year so far against the same days last year, month by month: the other ways a
    business reads "comparison", one click away with no AI -- the last 12 months against the 12 before, month by
    month, and last month against the month before; from either of those, the other two. None for a comparison
    with its own period or members."""
    if plan.kind != "query" or plan.shaped() or any(not g.startswith("time:") for g in plan.group_by) \
            or plan.time.compare is None or plan.time.window.fiscal:
        return []
    ask = ask_of(question)
    words = " ".join(re.findall(r"[a-z0-9]+", (question or "").casefold()))
    asked = ask is not None and ask.kind == "compare" and ask.by_month or any(
        words == " ".join(re.findall(r"[a-z0-9]+", said.casefold()))
        for w, *_ in READINGS for said in (w, READINGS_FR[w]))
    time = plan.time
    now = next((w for w, window, compare, grain in READINGS if time.window == window
                and time.compare.kind == compare.kind and time.compare.window is None and time.grain == grain), None)
    if not asked or now is None:
        return []
    return [(w, plan.model_copy(update={"time": time.model_copy(update={"window": window, "compare": compare,
                                                                        "grain": grain}),
                                        "follow_up": "refine"}))
            for w, window, compare, grain in READINGS if w != now]


def period_before(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "This month so far vs the same days last month" was set against the 15 days before today, and "the last 12
    months against the 12 before" against the same 12 months: the period before is the one of the same shape before
    it -- the same days of the month before, the 12 months before those."""
    if plan.kind != "query":
        return plan
    time = plan.time
    same = _SAME_DAYS_BEFORE.search(question or "")
    if same:
        unit = _unit_of(same)
        window = time.window if time.window.kind in ("to_date", "this") and time.window.unit == unit else \
            Window(kind="to_date", unit=unit)
        if time.compare is not None and time.compare.kind == "previous_period" and window == time.window:
            return plan
        return plan.model_copy(update={"time": time.model_copy(update={
            "window": window, "compare": Compare(kind="previous_period")}), "intent": "compare"})
    other = time.compare.window if time.compare is not None and time.compare.kind == "window" else None
    if other is None or other.kind not in ("last", "rolling", "this", "to_date", "previous"):
        return plan
    alike = (other.kind, other.unit, other.n or 1) == (time.window.kind, time.window.unit, time.window.n or 1)
    if alike or _SPAN_BEFORE.search(question or ""):
        return plan.model_copy(update={"time": time.model_copy(update={"compare": Compare(kind="previous_period")})})
    return plan


RUN_RATE_MONTHS = 3


def run_rate_figure(measure: Measure) -> Derived:
    """A year at the pace of the last three whole months: their total, times four (a figure times a number is its
    product with itself, the number as the scale)."""
    return Derived(name=f"Annual run rate of {measure.business_name}"[:80], op="product",
                   measures=[measure.slug, measure.slug], scale=12 / RUN_RATE_MONTHS)


def run_rate_of(plan: Plan) -> str | None:
    """The measure slug a plan's run rate is worked out from (as run_rate made it), or None."""
    for d in plan.derived:
        if d.op == "product" and d.measures[0] == d.measures[1] and d.scale == 12 / RUN_RATE_MONTHS \
                and d.name.startswith("Annual run rate of "):
            return d.measures[0]
    return None


def run_rate(plan: Plan, model: SemanticModel, question: str = "") -> Plan | None:
    """ "What is our run rate?": the figure the question names over the last three whole months, times four -- one
    figure, split as the question splits it. None when the plan holds no one figure that adds up over time."""
    if run_rate_of(plan) is not None:
        return plan
    known = {m.slug for m in model.measures.values()}
    slugs = list(dict.fromkeys(s for s in plan.measures if s in known))
    if not slugs:
        # Planned as a figure of the AI's own ("[Net amount] * 12 / 3", a ratio of the measure): the one measure it
        # is made of.
        written = " ".join([*(f.formula for f in plan.figures), *(s for d in plan.derived for s in d.measures)])
        names = {m.slug: m for m in model.measures.values()}
        slugs = list(dict.fromkeys(slug for slug, m in names.items() if re.search(
            rf"(?<![\w])(?:{re.escape(slug)}|\[{re.escape(m.business_name)}\])(?![\w])", written, re.IGNORECASE)))
    if len(slugs) != 1:
        return None
    measure = next(m for m in model.measures.values() if m.slug == slugs[0])
    if measure.additivity != "additive":
        return None
    return plan.model_copy(update={
        "measures": [], "derived": [run_rate_figure(measure)], "figures": [], "intent": "value",
        "sort": [s for s in plan.sort if s.by != slugs[0]] or [], "rolling": None,
        "time": TimeSpec(date=plan.time.date, window=Window(kind="last", unit="month", n=RUN_RATE_MONTHS))})


def with_figure(plan: Plan, model: SemanticModel, slug: str) -> Plan:
    """The same answer for another figure ("Gross amount instead"): its run rate, or its comparison."""
    measure = next((m for m in model.measures.values() if m.slug == slug), None)
    if measure is not None and run_rate_of(plan) is not None:
        return plan.model_copy(update={"derived": [run_rate_figure(measure)]})
    return plan.model_copy(update={"measures": [slug, *plan.measures[1:]]})


def planned_unasked(measure: Measure, question: str) -> bool:
    """Is the measure what was planned (a target, a budget, a forecast) while the question names nothing planned?"""
    return bool(_PLANNED.search(measure.business_name or "")) and not _PLANNED.search(question or "")


def target_unasked(plan: Plan, model: SemanticModel, question: str) -> list[str]:
    """ "How is loan volume doing vs last year?" was planned on Loan volume target: a target is what was planned,
    the question asks what happened. A problem for the planner, unless the question names a target; it names the
    figures of the rows the target's own words name ("loan" -> the loans' amounts), so the AI is not left to guess
    (told only "what happened", it took the transactions' amount)."""
    if plan.kind != "query" or plan.target is not None:
        return []
    out = []
    for m in model.measures.values():
        if m.slug in plan.measures and planned_unasked(m, question):
            words = {w.rstrip("s") for w in re.findall(r"[a-z]+", _PLANNED.sub(" ", m.business_name.casefold()))
                     if len(w) > 2}
            tables = {key for w in words for key in _tables_named(model, w)}
            near = [x for x in _money_of(model, tables, question) if x.table != m.table]
            hint = (" The figures of those rows: " + ", ".join(f"{x.slug} ({x.business_name})" for x in near[:4]) + "."
                    ) if near else ""
            out.append(f"{m.slug} ({m.business_name}) is what was planned, and the question names no target, budget "
                       f"or forecast: plan the figure of what happened instead.{hint}")
    return out


_VOLUME = re.compile(r"\b([a-z]+)\s+volumes?\b|\bvolume\s+(?:of|de|des|du)\s+([a-zé]+)", re.IGNORECASE)


def _money_of(model: SemanticModel, tables: set[str], question: str) -> list[Measure]:
    return [x for x in model.measures.values() if x.table in tables and not x.hidden and x.additivity == "additive"
            and x.format == "currency" and not planned_unasked(x, question)]


_KIND_WORDS = {"fact", "facts", "dim", "dimension", "dimensions", "tbl", "table", "f", "d"}


def _tables_named(model: SemanticModel, word: str) -> set[str]:
    """The tables a word names as a whole ("loan" -> Loans, "Loan fact", fact_loans), never every table with it in
    its name (loan applications, loan payments)."""
    word = word.casefold().rstrip("s")
    return {t.key for t in model.tables.values() if t.kind != "calendar" and word in {
        " ".join(w for w in re.findall(r"[a-zé]+", name.rsplit(".", 1)[-1].casefold()) if w not in _KIND_WORDS
                 ).rstrip("s") for name in (t.business_name, t.name) if name}}


def volume_elsewhere(plan: Plan, model: SemanticModel, question: str) -> list[str]:
    """ "Which branches grew their loan volume the most vs last year?" was counted on the transactions' amount: the
    volume of a thing is the money of its own rows (the loans). A problem for the planner, naming those figures, for
    a business ask whose figure is on other rows."""
    if plan.kind != "query" or not plan.measures or ask_of(question) is None:
        return []
    found = _VOLUME.search(question or "")
    if found is None:
        return []
    tables = _tables_named(model, found.group(1) or found.group(2) or "")
    chosen = [m for m in model.measures.values() if m.slug in plan.measures]
    if not tables or not chosen or any(m.table in tables for m in chosen):
        return []
    near = _money_of(model, tables, question)
    if not near:
        return []
    table = model.tables[next(iter(tables))].business_name
    return [f'The question\'s "{found.group(0)}" is counted on the {table} rows, and '
            + ", ".join(f"{m.slug} ({m.business_name})" for m in chosen) + " is not: plan a figure of the "
            f"{table} rows: " + ", ".join(f"{x.slug} ({x.business_name})" for x in near[:4]) + "."]


def figure_problems(plan: Plan, model: SemanticModel, question: str) -> list[str]:
    """What is wrong with the figure a plan counts for the question: a target it never names, a volume on other
    rows."""
    return target_unasked(plan, model, question) or volume_elsewhere(plan, model, question)


def figure_problem(ask: Ask) -> str:
    what = "a run rate" if ask.kind == "run_rate" else "a comparison with the period before"
    return (f"The question is {what} of the figure it names: a business ask Solviz answers at once. Do not ask which "
            "figure is meant: plan the one a business means by the question's word for it -- for revenue or sales "
            "the money taken in, net before gross; never a target, a budget, a cost, a discount or a tax the question "
            "does not name -- with any split the question asks for. Solviz sets the periods, says which figure it "
            "read and offers the others. A question naming no figure takes the main one.")


def _options(plan: Plan, model: SemanticModel) -> list[Measure]:
    from core2.resolve.resolver import find_slug

    out: list[Measure] = []
    by_name = {m.business_name.casefold(): m for m in model.measures.values() if m.business_name and not m.hidden}
    for option in (plan.clarify.options if plan.clarify is not None else []):
        text = re.sub(r"\s+(?:in|for|from|during)\s+.*$", "", str(option)).strip()
        found = find_slug(model, text)
        m = found[1] if found is not None and found[0] == "measure" else by_name.get(text.casefold())
        if isinstance(m, Measure) and m not in out:
            out.append(m)
    return out


def offered(plan: Plan, model: SemanticModel, question: str) -> list[str]:
    """The figures the AI asked back between ("net, gross, discount or cost amount?"), leaving out a target the
    question does not name: the readings the answer offers beside the one counted."""
    return [m.slug for m in _options(plan, model) if not planned_unasked(m, question)]


def figure_counted(plan: Plan) -> str | None:
    """The measure a plan counts: its run rate's, or its first."""
    return run_rate_of(plan) or (plan.measures[0] if plan.measures else None)


def figure_of(plan: Plan, model: SemanticModel, question: str, main: list[Measure]) -> tuple[Plan, list[str]] | None:
    """The figure a business ask means when the AI asked back which one: among the figures it offered (else those
    the question names, else the data's main ones), the first that is no target, cost or discount the question
    leaves unnamed, a figure that adds up before one that does not. The plan for it, and the others the reader may
    mean instead (offered on the answer)."""
    from core2.plan.followup import measures_named

    ask = ask_of(question)
    if ask is None:
        return None
    named = measures_named(model, question)
    choices = _options(plan, model) or [m for m in model.measures.values() if m.slug in named and not m.hidden] \
        or list(main)
    fair = [m for m in choices if not planned_unasked(m, question)
            and not (_BESIDE.search(m.business_name or "") and not _BESIDE.search(question or ""))] or \
        [m for m in choices if not planned_unasked(m, question)]
    if not fair:
        return None
    fair.sort(key=lambda m: m.additivity != "additive")
    chosen = fair[0]
    others = [m.slug for m in choices if m.slug != chosen.slug and not planned_unasked(m, question)][:3]
    return Plan(kind="query", intent="compare", measures=[chosen.slug], time=TimeSpec()), others


# The words of an ask that name no figure: who, the periods, the comparison ("which locations grew the most vs last
# year?", "how are we doing?", "quels magasins ont progressé ?").
_ASK_WORDS = frozenset("""which what who whom how are is am was were do does did have has had we our us my me you your
they their them it its the a an of in on for to with by from at and or than vs versus against compared compare comparison
most least more less best worst top bottom grew grow grows growing grown growth rose risen increased increase increases
fell fallen dropped drop declined decline decreased decrease shrank up down doing tracking performing going faring
looking track course pace beat beating last this current previous prior year years month months quarter quarters week
weeks so far date ytd mtd qtd today now same time period days ago before since overall total show tell give list rank
ranked ranking biggest largest highest lowest smallest gained gain lost improved over on there been be quels quelles quel
quelle qui ont le la les du des de en avec par rapport au aux sur cette ce cet an année annee mois trimestre semaine
dernier dernière derniere plus moins progressé progresse augmenté augmente baissé baisse reculé recule hausse comment
vont va est sont nous notre nos""".split())


def names_no_figure(plan: Plan, model: SemanticModel, question: str) -> bool:
    """Does the question name no figure at all -- only who, the periods and the comparison?"""
    from core2.resolve.resolver import find_slug

    split: set[str] = set()
    for slug in plan.group_by:
        found = find_slug(model, slug)
        if found is not None:
            for word in re.findall(r"[a-zà-ÿ]+", str(getattr(found[1], "business_name", "") or slug).casefold()):
                split |= {word, word + "s", word + "es", word.rstrip("s"), word[:-1] + "ies" if word.endswith("y")
                          else word}
    words = [w for w in re.findall(r"[a-zà-ÿ]+", (question or "").casefold().replace("’", "'"))
             if len(w) > 2 and w not in _ASK_WORDS and w not in split]
    return not words


def main_figure(plan: Plan, model: SemanticModel, question: str, main: list[Measure]) -> tuple[Plan, list[str]] | None:
    """ "Which locations grew the most vs last year?" was counted in batches: a business ask naming no figure is
    about the money the business takes in, its main figure. The plan for it, and the figure the AI chose (offered
    instead); None when the AI chose money already, or the question names a figure."""
    ask = ask_of(question)
    if ask is None or plan.kind != "query" or plan.derived or plan.figures or plan.rates or len(plan.measures) > 1:
        return None
    money = next((m for m in main if m.format == "currency"), None)
    if money is None or not names_no_figure(plan, model, question):
        return None
    chosen = next((m for m in model.measures.values() if plan.measures and m.slug == plan.measures[0]), None)
    if chosen is not None and (chosen.format == "currency" or chosen.slug == money.slug):
        return None
    sort = [s.model_copy(update={"by": money.slug}) if chosen is not None and s.by == chosen.slug else s
            for s in plan.sort]
    return plan.model_copy(update={"measures": [money.slug], "sort": sort}), [chosen.slug] if chosen else []


# A reader's own logic on fields: "line_amount plus travel_fee", "price_charged minus ingredient_cost minus
# labor_cost", "quantity times unit_price" -- each row's value worked out, then added up.
_FIELD = r"[A-Za-z_][\w.]*"
_JOINED = re.compile(rf"(?<![\w.])({_FIELD}(?:\s+(?:plus|minus|times|multiplied\s+by)\s+{_FIELD})+)", re.IGNORECASE)
_SIGN = {"plus": "+", "minus": "-", "times": "*", "multiplied by": "*"}
_AVERAGE_OF = re.compile(r"\b(?:average|avg|mean|moyenne)\s+(?:of\s+)?$", re.IGNORECASE)


def fields_combined(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "line_amount plus travel_fee from booking lines, by service_type, for 2025" was answered with the two totals
    side by side: fields the reader joins with plus, minus or times are one figure, each row's value worked out as
    written and then added up (averaged after "average"). Only fields the question names as the data writes them,
    one table's; a figure the AI wrote already stands."""
    from core2.model import authoring
    from core2.model.schema import AggExpr

    if plan.kind != "query" or len(plan.derived) > 1:
        return plan
    found = _JOINED.search(question or "")
    if found is None:
        return plan
    parts = re.split(r"\s+(plus|minus|times|multiplied\s+by)\s+", found.group(1), flags=re.IGNORECASE)
    names = authoring._Names(model)  # noqa: SLF001 - the editor's own reading of a field's name
    terms = parts[::2]
    raw = {c.name.casefold() for c in model.columns.values()}
    if any("_" not in t and "." not in t and t.casefold() not in raw for t in terms):
        return plan                  # a word that is no field's name ("sales plus returns") is not the reader's logic
    # Each field by its own name; one two tables hold (a travel fee on the booking and on its lines) is the one of
    # the table the other fields are on, or of the table the question names ("from booking lines").
    found_keys = [names._shown(names.bare.get(authoring._plain(t), [])) for t in terms]  # noqa: SLF001
    if not all(found_keys):
        return plan
    sure = {k[0] for k in found_keys if len(k) == 1}
    said = authoring._plain(question)  # noqa: SLF001
    keys: list[str] = []
    for options in found_keys:
        if len(options) > 1:
            options = [k for k in options if any(model.columns[k].table == model.columns[o].table for o in sure)] or \
                [k for k in options if authoring._plain(authoring.table_name(model, model.columns[k].table)) in said]  # noqa: SLF001
        if len(options) != 1:
            return plan
        keys.append(options[0])
    if len({model.columns[k].table for k in keys}) != 1:
        return plan
    body = f"[{authoring.field_ref(model, keys[0])}]"
    said = [model.columns[keys[0]].business_name or model.columns[keys[0]].name]
    for op, key in zip(parts[1::2], keys[1:]):
        word = re.sub(r"\s+", " ", op.casefold())
        body += f" {_SIGN[word]} [{authoring.field_ref(model, key)}]"
        said.append(f"{word} {(model.columns[key].business_name or model.columns[key].name).lower()}")
    agg = "AVG" if _AVERAGE_OF.search(question[:found.start()]) else "SUM"
    formula = f"{agg}({body})"
    if plan.figures:
        # A figure the AI wrote over the same fields stands when it works each row out as written; "SUM(quantity) *
        # SUM(unit price)" (the quantity of the year times the sum of its prices) is put right, its name kept.
        if len(plan.figures) != 1:
            return plan
        own = plan.figures[0]
        try:
            read = set(authoring.read(model, own.formula).columns)
        except authoring.AuthoringError:
            read = set()
        if read != set(keys) or re.sub(r"\s+", "", own.formula).casefold() == re.sub(r"\s+", "", formula).casefold():
            return plan
        return plan.model_copy(update={"figures": [own.model_copy(update={"formula": formula})]})
    used = {m.slug for m in model.measures.values() if isinstance(m.expr, AggExpr) and m.expr.column in keys}
    if plan.derived and not set(plan.derived[0].measures) <= used:
        return plan                  # a figure of other measures than the fields named
    money = any(m.format == "currency" for m in model.measures.values() if m.slug in used)
    figure = Figure(name=" ".join(said)[:80], formula=formula, format="currency" if money else "number")
    # "received_qty minus rejected_qty" planned as the two totals and their difference: the one figure, each row
    # worked out as written; the fields it is made of are no figures of the answer.
    gone = used | {d.name for d in plan.derived}
    kept = [m for m in plan.measures if m not in used]
    return plan.model_copy(update={"measures": kept, "figures": [figure], "derived": [],
                                   "sort": [s.model_copy(update={"by": figure.name}) if s.by in gone else s
                                            for s in plan.sort]})


def badges(plan: Plan, model: SemanticModel, picked: str | None) -> list[dict[str, str]]:
    """What the answer says it read: the figure picked for a business ask, and how a run rate is worked out."""
    out: list[dict[str, str]] = []
    if picked:
        name = next((m.business_name for m in model.measures.values() if m.slug == picked), picked)
        out.append({"kind": "reading", "text": f"Read as {name}", "text_fr": f"Lu comme {name}"})
    if run_rate_of(plan) is not None:
        out.append({"kind": "reading", "text": f"A year at the pace of the last {RUN_RATE_MONTHS} months (their total × 4)",
                    "text_fr": f"Une année au rythme des {RUN_RATE_MONTHS} derniers mois (leur total × 4)"})
    return out
