"""The thread rule: a follow-up keeps the filters and the period of the answer above; a new question keeps none.

Whether a turn follows the answer above or asks afresh is read from its words (core2/plan/followup.py) and, where
the words do not settle it, by the AI with the conversation in view, as a person reading it would. Once that is
decided, what the turn keeps is decided here, in code, so the answer does not depend on what the AI happened to
copy:

* a follow-up keeps every filter of the answer above that it does not replace ("how many orders?" after net sales
  for North in March is the orders of North): only a member of the same field it names ("and South?"), a split by
  that field, "all regions" or "overall" lets one go, and a filter the new figure cannot be narrowed by (checked);
* a follow-up that names one period of its own ("break 1997 down by region" after 1997 against 1996) is no longer
  a comparison unless it still compares; one that asks "which branches ..." of a monthly answer is no longer
  month by month unless it asks for a series;
* a new question carries no filter or period of the answer above that it does not write itself.

The period a follow-up keeps, its date and its links are kept by settle (core2/plan/planner.py).
"""

from __future__ import annotations

import re
from collections.abc import Callable

from core2.bootstrap import names
from core2.model.schema import SemanticModel
from core2.plan.followup import asks_series, names_own_period
from core2.plan import asks
from core2.plan.ir import Filter, Plan, Window

# "Overall", "in total", "for everyone": every filter of the answer above lets go.
_OVERALL = re.compile(r"\b(?:overall|in\s+total|altogether|company[-\s]wide|across\s+the\s+board|everyone|everything|"
                      r"our\s+(?:total|whole|entire)|au\s+total|dans\s+l'ensemble|globalement|tout\s+confondu|"
                      r"notre\s+total)\b", re.IGNORECASE)
# "All regions", "every store", "toutes les régions": the filter on that field lets go.
_ALL_OF = re.compile(r"\b(?:all(?:\s+(?:the|of\s+the|our))?|every|tou(?:s|tes)\s+(?:les|nos))\s+([a-zà-ÿ][\w-]*"
                     r"(?:\s+[a-zà-ÿ][\w-]*)?)", re.IGNORECASE)
# "Without the filter", "not only North", "remove the filter": the reader lets go of the narrowing.
_LET_GO = re.compile(r"\b(?:without|remove|drop|clear)\s+(?:the\s+|that\s+|this\s+)?(?:filters?|conditions?)\b|"
                     r"\bnot\s+only\b|\bsans\s+(?:le\s+|ce\s+)?filtres?\b", re.IGNORECASE)
# The question still compares two periods: "vs", "growth", "rose", "évolution", "par rapport à".
# "Which branches ...", "quels magasins ...": the members of a kind, never "which months" (a series of its own).
_WHICH_MEMBERS = re.compile(r"^\s*(?:which|quels?|quelles?)\s+(?!(?:days?|weeks?|months?|quarters?|years?|periods?|"
                            r"dates?|jours?|semaines?|mois|trimestres?|ann[ée]es?)\b)\w+", re.IGNORECASE)
_COMPARES = re.compile(r"\b(?:vs\.?|versus|against|compar\w*|growth|grew|grow|change[sd]?|rose|rise[sn]?|risen|fell|"
                       r"fall(?:en|s)?|drop(?:ped|s)?|increase[sd]?|decrease[sd]?|yoy|year[-\s]over[-\s]year|"
                       r"than\s+(?:last|the|in)|both|[ée]volution|par\s+rapport|compar[ée]\w*|hausse|baisse)\b",
                       re.IGNORECASE)


def _field_words(model: SemanticModel, field: str) -> set[str]:
    """The words a reader uses for the field a filter is on: "region" for region.region_name, singular."""
    words = set(re.findall(r"[a-z]+", field.lower().replace("_", " ")))
    attribute = model.attributes.get(field)
    if attribute is not None and attribute.business_name:
        words |= set(re.findall(r"[a-zà-ÿ]+", attribute.business_name.lower()))
    return {names.singular(w) for w in words if len(w) > 2} - {"name", "code", "id", "key", "the"}


def member_of(model: SemanticModel, slug: str) -> str:
    """The member a field names: an entity, whether by its slug, its name or its code; any other field itself."""
    if slug in model.entities:
        return slug
    attribute = model.attributes.get(slug)
    if attribute is not None:
        for name, entity in model.entities.items():
            if attribute.column in (entity.label_column, entity.code_column, *entity.key_columns):
                return name
    return slug


def _all_of(question: str, model: SemanticModel, field: str) -> bool:
    """Does the question ask for every member of the field ("all regions", "toutes les régions")?"""
    said = {names.singular(w) for found in _ALL_OF.finditer(question)
            for w in re.findall(r"[a-zà-ÿ]+", found.group(1).lower())}
    return bool(said & _field_words(model, field))


def _written(f: Filter, question: str) -> bool:
    """Does the question write one of the filter's values itself ("North" in "and North by store", "100" in "list
    price over 100")?"""
    text = question.casefold()
    return any(str(v).strip() and (len(str(v)) >= 2 or not isinstance(v, str))
               and re.search(rf"(?<!\w){re.escape(str(v).casefold())}(?!\w)", text) for v in f.values)


def kept_from_above(plan: Plan, model: SemanticModel, question: str, previous: Plan | None,
                    answers: Callable[[Plan], bool]) -> Plan:
    """The plan with what the thread rule keeps from ``previous`` (the plan of the answer above), or leaves out.

    ``answers``: whether a plan can be answered (checked against the model); a filter a follow-up would keep is
    kept only where the plan with it still answers.
    """
    if plan.kind != "query" or previous is None or previous.kind != "query":
        return plan
    plan = figures_carried(plan, previous)
    pointed = _pointed_at(question, previous, plan)
    if pointed:
        # "What did the lab spend in March 2026?" after "R&D Lab #2": the member above, by a word of its name,
        # whatever else the question asks anew.
        trial = plan.model_copy(update={"filters": [*plan.filters, *pointed]})
        plan = trial if answers(trial) else plan
    if plan.follow_up == "new":
        return _nothing_carried(plan, question, previous)
    if plan.follow_up != "refine":
        return plan
    update: dict = {}
    filters = list(plan.filters)
    if not _OVERALL.search(question) and not _LET_GO.search(question):
        have = {f.field for f in filters}
        # A split by the field, by any of its names: "by doctor" after one doctor's name is every doctor.
        split = {member_of(model, g) for g in plan.group_by}
        for f in previous.filters:
            if f.field in have or f in filters or member_of(model, f.field) in split \
                    or _all_of(question, model, f.field):
                continue
            trial = plan.model_copy(update={"filters": [*filters, f]})
            if answers(trial):
                filters.append(f)
    if filters != list(plan.filters):
        update["filters"] = filters
    order = _order_kept(plan, previous, question)
    if order is not None and answers(plan.model_copy(update=order)):
        update.update(order)
    if previous.segment is not None and plan.segment is None and not _OVERALL.search(question) \
            and not _LET_GO.search(question) and answers(plan.model_copy(update={"segment": previous.segment})):
        update["segment"] = previous.segment       # "and by month" after returning customers' sales: still theirs
    lc = previous.lifecycle
    if lc is not None and plan.segment is None and plan.lifecycle is None and plan.measures \
            and lc.status in ("new", "returning", "won_back", "repeat", "one_time") and _THEM.search(question):
        # "And what did we spend with them" after the suppliers won back this month: the spend with those suppliers,
        # not their number again.
        from core2.plan.ir import Segment

        them = Segment(of=lc.of, measure=lc.measure, status=lc.status, name=lc.name)
        if answers(plan.model_copy(update={"segment": them})):
            update["segment"] = them
    if plan.group_by and plan.group_by == previous.group_by and _TOTAL.search(question) \
            and not any(_named_split(model, g, question) for g in plan.group_by):
        # "And total spend by month?" after spend by category: one total a month, no longer by category.
        update["group_by"] = []
        if plan.intent == "breakdown":
            update["intent"] = "trend" if plan.time.grain else "value"
    time = plan.time
    if time.compare is not None and time.compare == previous.time.compare and plan.intent != "drivers" \
            and names_own_period(question) and not _COMPARES.search(question) and not asks.compares(question):
        # "Break 1997 down by region" after 1997 against 1996: 1997 by region, no longer a comparison.
        time = time.model_copy(update={"compare": None})
        update["sort"] = [s for s in plan.sort if s.by not in ("change", "pct_change")]
        if plan.intent == "compare":
            update["intent"] = "breakdown" if plan.group_by else "value"
    added = [g for g in plan.group_by if g not in previous.group_by]
    if time.grain is not None and time.grain == previous.time.grain and added and _WHICH_MEMBERS.match(question) \
            and not asks_series(question):
        # "Which branches were those loans booked at?" after loans by month: the branches, not month by month.
        time = time.model_copy(update={"grain": None})
        if plan.intent == "trend":
            update["intent"] = "breakdown"
    if time != plan.time:
        update["time"] = time
    return plan.model_copy(update=update) if update else plan


# A count written out ("top five by margin").
_COUNT = r"(?:two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty|cinq|dix|vingt)"
# The reader orders the rows anew: "sorted by units", "highest first", "rank them by margin", "trié par".
# An order of the follow-up's own: said ("sorted", "highest first"), or rows ranked anew ("top 5 by orders", "the
# bottom 5", "by units sold instead"); "show units sold instead of the share" or "for the same top 5" keep the order.
_ORDER_SAID = re.compile(r"\b(?:sort(?:ed)?|order(?:ed)?|rank(?:ed)?|highest|lowest|biggest|smallest|largest|most|"
                         r"least|fewest|ascending|descending|tri[ée]e?s?|class(?:é|ée|és|ées)|classe[rz]|ordre|croissant|"
                         r"d[ée]croissant)\b|\b(?:top|bottom|best|worst)\s+(?:[\w-]+\s+){0,2}?by\b|"
                         r"\b(?:top|bottom|first|last|best|worst)\s+(?:\d+|" + _COUNT + r")\s+(?!(?:days?|weeks?|months?|"
                         r"quarters?|years?)\b)(?:[\w-]+\s+){0,2}?by\b|"
                         r"\b(?:\d+|" + _COUNT + r")\s+(?:best|worst|top|bottom|biggest|smallest)\s+(?:[\w-]+\s+){0,2}?by\b|"
                         r"\bbottom\s+(?:\d+|" + _COUNT + r")\b|"
                         # "by units sold rather than margin", "now by margin instead": a ranking said anew where its clause
                         # opens ("show the margin by product rather than the share" swaps a figure, the rows kept).
                         r"(?:^|[,;:]\s*|\b(?:now|just|then|and|but|so|rank(?:ed)?|sort(?:ed)?|order(?:ed)?|about|them|it|these|"
                         r"those)\s+)by\s+(?:the\s+)?"
                         r"(?:[\w-]+\s+){0,3}?[\w-]+,?\s+(?:instead|rather\s+than)\b|"
                         r"\b(?:premiers|derniers|meilleurs|pires)\s+(?:[\w-]+\s+){0,2}?par\b|"
                         r"(?:^|[,;:]\s*|\b(?:maintenant|puis|et|class[ée]s?|tri[ée]s?|les|classe(?:-les)?|trie(?:-les)?)\s+)par\s+"
                         r"(?:[\w-]+\s+){0,3}?[\w-]+,?\s+"
                         r"(?:plut[oô]t|au\s+lieu)\b", re.IGNORECASE)


def _order_kept(plan: Plan, previous: Plan, question: str) -> dict | None:
    """The order of the rows above, for a follow-up on the same rows that says no order of its own ("drop the
    share column and add units sold" after the top 5 products by gross margin: the same 5, ranked as they were),
    when the figure they were ranked by is still on the answer; None otherwise."""
    if plan.sort or not previous.sort or plan.group_by != previous.group_by or not plan.group_by \
            or plan.time.grain or _ORDER_SAID.search(question):
        return None                # "top 5 by orders instead", "by month": an order of its own
    shown = {*plan.measures, *(d.name for d in plan.derived), *(f.name for f in plan.figures), *plan.group_by,
             "period"}
    if plan.time.compare is not None:
        shown |= {"change", "pct_change"}
    if any(s.by not in shown for s in previous.sort):
        return None
    return {"sort": list(previous.sort)}


_POINTER = r"(?:the|that|this|our|le|la|ce|cette|cet)"
_NOT_NAMES = {"the", "and", "for", "of", "inc", "ltd", "llc", "co", "store", "team", "group"}


def points_above(question: str, previous: Plan | None) -> bool:
    """Does the question point at the member the answer above was narrowed to, by a word of its name ("What did
    the lab spend in March?" after "R&D Lab #2")? It follows that answer, however much else it asks anew."""
    return previous is not None and previous.kind == "query" \
        and bool(_pointed_at(question, previous, Plan(kind="query")))


def _pointed_at(question: str, previous: Plan, plan: Plan) -> list[Filter]:
    """The filters above whose member the question points at by a word of its name ("the lab" for "R&D Lab #2",
    "that studio"), when the plan names no member of that field itself."""
    have = {f.field for f in plan.filters}
    out = []
    for f in previous.filters:
        if f.field in have or f.op not in ("eq", "in") or len(f.values) != 1 or not isinstance(f.values[0], str):
            continue
        words = {w for w in re.findall(r"[a-z]+", f.values[0].lower()) if len(w) >= 3 and w not in _NOT_NAMES}
        if any(re.search(rf"\b{_POINTER}\s+{re.escape(w)}s?\b", question, re.IGNORECASE) for w in words):
            out.append(f)
    return out


# Those the answer above counted: "what did we spend with them", "leur chiffre d'affaires".
_THEM = re.compile(r"\b(?:them|they|those|these|their|eux|ils|elles|leurs?|ces)\b", re.IGNORECASE)
# The whole, no longer split: "and total spend by month?", "le total par mois".
_TOTAL = re.compile(r"\b(?:total|overall|le\s+total|au\s+total)\b", re.IGNORECASE)


def _named_split(model: SemanticModel, slug: str, question: str) -> bool:
    """Does the question name the split itself ("total spend by category")?"""
    words = {w for w in re.split(r"[._\s]+", slug.lower()) if len(w) > 2 and w not in ("name", "code")}
    attribute = model.attributes.get(slug)
    if attribute is not None:
        words |= {w for w in re.findall(r"[a-z]+", (attribute.business_name or "").lower()) if len(w) > 2
                  and w not in ("name", "code")}
    said = {names.singular(w) for w in re.findall(r"[a-z]+", question.lower())}
    return bool({names.singular(w) for w in words} & said)


def figures_carried(plan: Plan, previous: Plan) -> Plan:
    """A figure worked out for the answer above and named again ("and by formula" after "the lowest yield"): its
    formula comes with its name, whatever the thread rule keeps, since the name means nothing without it."""
    from core2 import ids

    have = {ids.slug(f.name) for f in plan.figures}
    named = {ids.slug(x) for x in [*plan.measures, *(s.by for s in plan.sort), *(f.field for f in plan.filters),
                                   *(m for d in plan.derived for m in d.measures), *(r.measure for r in plan.rates)]}
    carried = [f for f in previous.figures if ids.slug(f.name) not in have and ids.slug(f.name) in named]
    return plan.model_copy(update={"figures": [*plan.figures, *carried]}) if carried else plan


def derived_as_above(plan: Plan, previous: Plan, model: SemanticModel, question: str = "") -> Plan:
    """A figure worked out above and named again keeps what it was made of: "drop the cost column and add the units
    sold" after a margin made of the cost at standard cost (a figure worked out for that question) was planned with
    the margin made of the cost the AI first named. Each part of a derived figure of the same name that was a
    figure above, and that the AI writes again as what the figure was worked out from (the column its formula
    multiplies, or by the figure's own words: "standard cost", "list_price"), is that figure again; a part the
    reader makes it of ("on the gross amount instead", "use the standard cost amount booked on each line") stays
    theirs."""
    from core2 import ids
    from core2.model import authoring
    from core2.model.schema import AggExpr
    from core2.resolve.resolver import find_slug

    def plain(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", (text or "").lower().replace("_", " ")))

    asked, said = plain(question), (question or "").lower().replace("_", " ")

    def chosen(name: str) -> bool:
        """Does the reader make a figure of this part ("switch the margin to the standard cost amount", "use the standard
        cost amount column for the margin", "work it out on the gross amount instead")? Naming it to show it, look at
        it or sort by it is no choice ("same for 2024 with the standard cost amount shown too", "switch to a bar chart
        with the standard cost amount shown too", "sort on the standard cost amount")."""
        phrase = r"\s+".join(re.escape(w) for w in re.findall(r"[a-z0-9]+", name.lower().replace("_", " ")))
        for h in (re.finditer(rf"\b{phrase}\b", said) if phrase else ()):
            clause = re.split(r"[,.;:?!]|\b(?:and|then)\b", said[:h.start()])[-1]
            if re.match(r"\s*(?:shown|showing|too|as\s+well|beside|alongside|next\s+to)\b", said[h.end():]) \
                    or re.search(r"\b(?:show|showing|display|add|include|plus|also|look\s+at|focus\s+on|report\s+on|"
                                 r"(?:sort|order|rank)(?:ed)?\s+(?:it\s+|them\s+)?(?:by|on))\s+(?:the\s+|its\s+|their\s+)?$",
                                 clause):
                continue
            # A choosing verb that governs the part, not the chart or the period ("change the year to 2024 with ...").
            verb = list(re.finditer(r"\b(?:use|using|switch|replace|swap|change|compute|computed|calculate|calculated|"
                                    r"work(?:ed)?\s+(?:[\w-]+\s+){0,2}?out|based\s+on|made\s+(?:of|from)|minus|less|"
                                    r"subtract|take|via|instead|should\s+be|make|out\s+of)\b", clause))
            between = clause[verb[-1].end():] if verb else ""
            if verb and not re.search(r"\b(?:chart|graph|table|view|year|month|quarter|week|it|this|that|(?:19|20)\d{2})\b",
                                      between):
                return True
            if re.search(r"\b(?:on|from|against|via)\s+(?:the\s+|its\s+|each\s+|their\s+)?$", clause):
                return True
        return False

    # Words the reader names only to take them off ("drop the list price column"): never a choice of part.
    dropped = set().union(*(plain(m.group(1)) for m in re.finditer(
        r"\b(?:drop|remove|without|hide|lose|take\s+(?:off|out)|leave\s+out|instead\s+of)\s+([^,.;]{1,40}?)"
        r"(?=\s+(?:column|and|then)\b|[,.;]|$)",
        (question or "").lower())))
    figures = {ids.slug(f.name): f for f in previous.figures}

    def made_of(slug: str, figure) -> bool:
        """Is this part of the plan what the figure above was worked out from, and not the reader's own choice?"""
        found = find_slug(model, slug)
        column, names = None, plain(slug)
        if found is not None and found[0] in ("measure", "attribute"):
            what = found[1]
            column = (what.expr.column if isinstance(what.expr, AggExpr) else None) if found[0] == "measure" \
                else what.column
            names |= plain(what.business_name) | plain(model.columns[column].name if column in model.columns else "")
            own = plain(what.business_name)
            if own and own <= asked and not own <= dropped and chosen(what.business_name):
                return False           # the reader makes it of this: "use the standard cost amount booked on each line"
        if column and column in model.columns and f"[{authoring.field_ref(model, column)}]" in figure.formula:
            return True
        at = figure.name.lower().split(" at ", 1)
        words = plain(at[1]) if len(at) == 2 else set()
        return bool(words) and words <= names

    above = {d.name.casefold(): d for d in previous.derived}
    derived, changed = [], False
    for d in plan.derived:
        before = above.get(d.name.casefold())
        if before is None or before.op != d.op or len(before.measures) != len(d.measures):
            derived.append(d)
            continue
        parts = [figures[ids.slug(b)].name if ids.slug(b) in figures and ids.slug(b) != ids.slug(m)
                 and made_of(m, figures[ids.slug(b)]) else m
                 for b, m in zip(before.measures, d.measures)]
        changed = changed or parts != list(d.measures)
        derived.append(d.model_copy(update={"measures": parts}))
    return plan.model_copy(update={"derived": derived}) if changed else plan


def _nothing_carried(plan: Plan, question: str, previous: Plan) -> Plan:
    """A new question with the filters or the period of the answer above that it does not write itself, without
    them: the AI copied them from the plan it was shown."""
    update: dict = {}
    kept = [f for f in plan.filters if f not in previous.filters or _written(f, question)]
    if len(kept) != len(plan.filters):
        update["filters"] = kept
    window = plan.time.window
    if window.kind != "all" and window == previous.time.window and not names_own_period(question):
        update["time"] = plan.time.model_copy(update={"window": Window(), "compare": None
                                                     if plan.time.compare == previous.time.compare
                                                     else plan.time.compare})
    return plan.model_copy(update=update) if update else plan


def kept_parts(plan: Plan, previous: Plan | None, question: str) -> tuple[list[Filter], Window | None]:
    """What a follow-up's answer kept from the answer above without the question saying it: its filters, and its
    period (None when the question names its own or there is none). Said on the answer, so the reader sees it."""
    if plan.kind != "query" or previous is None or previous.kind != "query" or plan.follow_up != "refine":
        return [], None
    filters = [f for f in plan.filters if f in previous.filters and not _written(f, question)]
    window = plan.time.window
    same = window.kind != "all" and window == previous.time.window and not names_own_period(question)
    return filters, (window if same else None)
