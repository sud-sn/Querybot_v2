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
from core2.plan.ir import Filter, Plan, Window

# "Overall", "in total", "for everyone": every filter of the answer above lets go.
_OVERALL = re.compile(r"\b(?:overall|in\s+total|altogether|company[-\s]wide|across\s+the\s+board|everyone|everything|"
                      r"au\s+total|dans\s+l'ensemble|globalement|tout\s+confondu)\b", re.IGNORECASE)
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
    time = plan.time
    if time.compare is not None and time.compare == previous.time.compare and plan.intent != "drivers" \
            and names_own_period(question) and not _COMPARES.search(question):
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


def figures_carried(plan: Plan, previous: Plan) -> Plan:
    """A figure worked out for the answer above and named again ("and by formula" after "the lowest yield"): its
    formula comes with its name, whatever the thread rule keeps, since the name means nothing without it."""
    from core2 import ids

    have = {ids.slug(f.name) for f in plan.figures}
    named = {ids.slug(x) for x in [*plan.measures, *(s.by for s in plan.sort), *(f.field for f in plan.filters),
                                   *(m for d in plan.derived for m in d.measures), *(r.measure for r in plan.rates)]}
    carried = [f for f in previous.figures if ids.slug(f.name) not in have and ids.slug(f.name) in named]
    return plan.model_copy(update={"figures": [*plan.figures, *carried]}) if carried else plan


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
