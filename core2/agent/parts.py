"""Nothing left out without a word: every part of a request, checked against the plan and the answer.

A reader asks for several things at once ("revenue, cost, gross margin and margin % for each store"; "the
customers who bought Dish Soap, with their segment and city"). The AI lists those parts before it writes its plan
(Plan.asked): each figure, field, condition, period, comparison, ranking and form, in the reader's words, with what
in the plan gives it, or nothing when the plan cannot. Code then checks each part:

* a part the plan cannot give (the AI said so) is missing;
* a figure or field the plan names but the answer's columns lack (worked out inside another figure, never shown)
  is missing;
* a part whose plan reference is in no part of the plan is missing;
* a member the reader named beyond doubt ("Dish Soap") that no condition of the plan holds is missing.

A missing part is asked of the AI once more (core2/agent/again.py, with what is missing); when the second reading
still lacks it, or is no better, the answer says what it left out, with a button to ask for it. Never a silent drop.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core2.plan.ir import Plan

CHART_WORDS = re.compile(r"\b(?:chart|graph|plot|bars?|lines?|pie|donut|visual|graphique|courbe|camembert)\b", re.I)
# A plan's section names: a part whose plan is one of them ("derived") says where, not which.
SECTIONS = {"derived", "rates", "figures", "durations", "measures", "filters", "groupby", "plan", "per", "within",
            "absent", "bands", "conditions"}
# Figures the answer works out by itself, by the plan's word for them, and what their column's name holds: a share
# of the total (net_amount_share), the figure before (net_amount_prior), a change, a % change, a share of a change.
COMPUTED = {"share": ("share",), "percentshare": ("share",), "pctshare": ("share",), "shareoftotal": ("share",),
            "compare": ("prior", "change"), "change": ("change",), "difference": ("change",),
            "pctchange": ("pctchange", "changepct"), "changepct": ("pctchange", "changepct"),
            "percentchange": ("pctchange", "changepct"), "shareofchange": ("shareofchange",),
            "contribution": ("shareofchange", "contribution")}
# "% of the total increase", "share of the growth": a share of a change, never of the total.
OF_CHANGE = re.compile(r"(?:%|percent|share|pct|proportion).{0,30}\b(?:change|increase|growth|rise|decline|drop|"
                       r"decrease|fall|gain|loss)", re.I)

# Words that ask for an amount ("the amount owed", "their value", "how much is outstanding"), never a count.
AMOUNT_WORDS = re.compile(r"\b(?:amounts?|value|owed|owing|outstanding|balances?|revenue|sales|dollars?|spend|cost|"
                          r"how much|montants?|valeur|d[uû]s?|solde)\b|\$", re.I)

# A follow-up that refines the answer on screen ("put each month's target next to it", "only the top 5, as a bar
# chart"): the figures it showed stay, unless the reader takes one away or asks for another in its place.
ADDS = re.compile(r"\b(?:add|adding|next to|alongside|along with|also|as well|plus|include|including|ajout\w*|"
                  r"aussi|à côté)\b", re.I)
SWAPS = re.compile(r"\b(?:drop|remove|without|exclud\w*|instead|replace|rather|swap|switch|what about|how about|"
                   r"enl[eè]ve\w*|sans|au lieu|plut[oô]t|remplace\w*|et pour)\b", re.I)


@dataclass(frozen=True)
class Gap:
    words: str          # the reader's words for the part ("gross margin", "Dish Soap")
    kind: str           # figure | field | condition | period | comparison | ranking | form | other
    why: str            # no_plan | not_in_plan | not_shown | not_applied


def _plain(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").casefold())


def _close(a: str, b: str) -> bool:
    """Two plain names for the same thing: equal, or one holds the other (net_amount and net_amount_prior)."""
    return bool(a) and bool(b) and (a == b or (len(a) >= 4 and a in b) or (len(b) >= 4 and b in a))


def _columns(payload: dict[str, Any]) -> list[str]:
    data = payload.get("data") or {}
    labels = data.get("header_labels") or {}
    out = [_plain(h) for h in data.get("headers") or []]
    out += [_plain(v) for v in (labels.values() if isinstance(labels, dict) else labels)]
    rows = data.get("rows") or []
    if rows and isinstance(rows[0], dict):
        out += [_plain(k) for k in rows[0]]
    return [c for c in out if c]


def _figure_names(plan: Plan, model: Any) -> dict[str, set[str]]:
    """Each figure the plan works out, by its plain name, with the plain names its column may carry."""
    out: dict[str, set[str]] = {}
    for slug in plan.measures:
        m = next((x for x in model.measures.values() if x.slug == slug), None) if model is not None else None
        names = {_plain(slug), *({_plain(m.business_name)} if m is not None else set())}
        out[_plain(slug)] = names
        if m is not None:
            out[_plain(m.business_name)] = names
    for group in (plan.derived, plan.rates, plan.figures, plan.durations):
        for x in group:
            out[_plain(x.name)] = {_plain(x.name)}
    return out


def _shown(names: set[str], columns: list[str], claimed: frozenset[str] = frozenset()) -> bool:
    """A column carries the name, or starts or ends with it (net_amount_prior, avg_net_amount); never a shorter
    name inside a longer one ("margin" is no column for "gross margin"), nor one inside the column of another figure
    of the plan named exactly so ("margin" is not shown by the column margin_pct)."""
    return any(n and (c == n or c not in claimed and (c.startswith(n) or c.endswith(n)))
               for n in names for c in columns)


def _field_names(ref: str, plan: Plan, model: Any) -> set[str]:
    """The names a field's column may carry: its slug, its column's own name and its business name."""
    out = {ref}
    slug = next((g for g in [*plan.group_by, *plan.via.keys()] if _same_field(ref, _plain(g))), None)
    attribute = model.attributes.get(slug) if model is not None and slug else None
    if attribute is not None:
        out |= {_plain(attribute.business_name), _plain(model.columns[attribute.column].name)}
    return {n for n in out if n}


def _same_field(ref: str, group: str) -> bool:
    """A group_by entry for the field named: the same slug, or an entity and its name ("store", "store.name")."""
    return group == ref or group == ref + "name" or ref == group + "name"


def missing(question: str, plan: Plan, payload: dict[str, Any], model: Any, *,
            named: list[Any] = (), before: Plan | None = None) -> list[Gap]:
    """The parts of the request the answer lacks (see the module's docstring). ``named``: the members the reader
    named beyond doubt (core2.plan.values.MemberIndex.written_names and quoted), each of which a condition must
    hold. ``before``: the plan of the answer a follow-up refines, whose figures and fields stay when it adds."""
    if plan.kind != "query" or payload.get("data") is None or payload.get("unsupported"):
        return []
    columns = _columns(payload)
    figures = _figure_names(plan, model)
    body = _plain(json.dumps(plan.model_dump(mode="json", exclude_defaults=True, exclude={"asked", "notes"})))
    # Where a field the answer does not group by may still be: a condition on it, or an analysis over it (never the
    # date that only sets the period: "order date" asked as a column is no column when it only picks March).
    held_body = _plain(json.dumps(plan.model_dump(mode="json", exclude_defaults=True,
                                                  exclude={"asked", "notes", "time", "sort", "limit", "measures"})))
    groups = [_plain(g) for g in [*plan.group_by, *plan.via.keys()]]
    by_date = _plain(plan.time.date) if plan.time.grain and plan.time.date else ""
    worked = [_plain(x.name) for group in (plan.derived, plan.rates, plan.figures, plan.durations) for x in group]
    # The columns each figure of the plan heads, by its exact name: no other figure is shown by them.
    claimed = frozenset(c for c in columns if c in {*worked, *(_plain(m) for m in plan.measures)})
    gaps: list[Gap] = []
    known = {_plain(m.slug) for m in model.measures.values()} if model is not None else set()
    conditions = bool(_conditions(plan) or plan.segment or plan.within or plan.target or plan.include_left_out)
    figured = bool(plan.measures or worked) or not plan.shaped()
    for part in plan.asked:
        ref = _plain(part.plan)
        if part.plan and part.kind in ("figure", "other") and not figured and AMOUNT_WORDS.search(part.words) \
                and ref not in figures and ref not in known:
            # An amount given to an analysis that only counts ("the amount owed in each bucket" given to the ageing,
            # which counts the open orders): no figure of the plan totals it.
            gaps.append(Gap(part.words, "figure", "not_in_plan"))
            continue
        if not part.plan:
            if part.kind == "form" and not CHART_WORDS.search(part.words):
                continue                    # "a list", "a table": every answer has its table
            if part.kind == "other":
                continue                    # words the AI could not place ("returns": the rows listed)
            gaps.append(Gap(part.words, part.kind, "no_plan"))
            continue
        if ref.startswith("time") and part.kind in ("field", "figure"):
            # "Return date" given as time.date: shown only when the answer is grouped by it (its period column).
            if not plan.time.grain:
                gaps.append(Gap(part.words, "field", "not_shown"))
            continue
        if part.kind == "field" and ref in figures:
            part = part.model_copy(update={"kind": "figure"})      # a figure the AI called a field
        if part.kind in ("figure", "other") and (ref in COMPUTED or OF_CHANGE.search(part.words)) \
                and ref not in figures and ref not in known:
            # Worked out by the answer itself: its column is there ("each category's % of the total increase" is
            # a share of the change, never the % change).
            need = COMPUTED["shareofchange"] if OF_CHANGE.search(part.words) else COMPUTED.get(ref, ())
            if columns and need and not any(t in c for t in need for c in columns):
                gaps.append(Gap(part.words, "figure", "not_shown"))
            continue
        if ref in SECTIONS:
            # "plan": "derived" says where, not which: every figure worked out there must be shown, and the section
            # must hold one; "filters" for a condition, that the plan has one.
            held_there = getattr(plan, {"groupby": "group_by"}.get(ref, ref), None)
            if part.kind in ("figure", "field") and (held_there is not None and not held_there or columns and any(
                    not _shown({w}, columns, claimed) for w in worked)):
                gaps.append(Gap(part.words, part.kind, "not_in_plan" if held_there is not None and not held_there
                                else "not_shown"))
            elif part.kind == "condition" and not conditions:
                gaps.append(Gap(part.words, part.kind, "not_in_plan"))
            continue
        if part.kind == "figure":
            names = figures.get(ref) or next((v for k, v in figures.items() if _close(k, ref)), None)
            if names is None:
                if ref in known:
                    # A measure of the catalog the plan only works out inside another figure ("the number of
                    # customers" inside an average per customer): asked for, so it must be a column.
                    if columns and not _shown({ref}, columns):
                        gaps.append(Gap(part.words, part.kind, "not_shown"))
                    continue
                # A figure the answer works out by itself (the % change of a comparison, a share) is a column.
                if not (columns and _shown({ref}, columns)) and ref not in body:
                    gaps.append(Gap(part.words, part.kind, "not_in_plan"))
                continue
            if columns and not _shown(names, columns, claimed):
                gaps.append(Gap(part.words, part.kind, "not_shown"))
        elif part.kind == "field":
            if by_date and (ref == by_date or _close(ref, by_date)):
                continue                    # the date the answer is grouped by: its period column
            if not any(_same_field(ref, g) for g in groups):
                # A field the reader filters by is no column ("for the West region"): only one to show must be.
                if ref not in held_body:
                    gaps.append(Gap(part.words, part.kind, "not_in_plan"))
                continue
            if columns and not _shown(_field_names(ref, plan, model), columns):
                gaps.append(Gap(part.words, part.kind, "not_shown"))
        elif part.kind == "form":
            chart = payload.get("chart") or {}
            kind = str(chart.get("type") or chart.get("chart_type") or "")
            if CHART_WORDS.search(part.words) and (not chart or kind in ("", "table", "kpi")):
                gaps.append(Gap(part.words, part.kind, "not_shown"))
        elif part.kind == "period":
            if plan.time.window.kind == "all" and not plan.time.grain and plan.time.compare is None:
                gaps.append(Gap(part.words, part.kind, "not_in_plan"))
        elif part.kind == "comparison":
            if plan.time.compare is None and plan.intent not in ("compare", "drivers") and not plan.derived \
                    and plan.target is None and ref not in body:
                gaps.append(Gap(part.words, part.kind, "not_in_plan"))
        elif part.kind == "ranking":
            if plan.limit is None and not plan.sort and plan.within is None and ref not in body:
                gaps.append(Gap(part.words, part.kind, "not_in_plan"))
        elif ref not in body and ref not in ("time", "compare", "limit", "chart"):
            gaps.append(Gap(part.words, part.kind, "not_in_plan"))
    if before is not None and before.kind == "query" and not SWAPS.search(question) and (
            ADDS.search(question) or not _names_a_figure(question, model)) \
            and plan.intent not in ("drivers", "forecast") and not plan.shaped():
        # (An analysis or a why shows its figures in its own columns: "before", "after", "change".)
        # The answer on screen refined: the figures it showed stay, and are shown ("put each month's target next to
        # it" kept the orders in the plan, and the answer against target dropped them).
        now = {_plain(m) for m in plan.measures} | set(worked)
        kept = [*before.measures, *(x.name for g in (before.derived, before.rates, before.figures, before.durations)
                                    for x in g)]
        for name in kept:
            gone = not _shown({_plain(name)}, columns, claimed) if columns else _plain(name) not in now
            if gone:
                gaps.append(Gap(_business(name, model), "figure", "not_kept"))
    held = {_plain(v) for f in _conditions(plan) for v in f.values}
    grouped = {_plain(g) for g in [*plan.group_by, *plan.via.keys()]}
    for m in named:
        if _plain(m.attribute) in grouped:
            continue                        # "compare North and South": a row each of the grouping shown
        if _plain(m.value) not in held and not any(g.kind == "condition" and _close(_plain(g.words), _plain(m.text))
                                                    for g in gaps):
            gaps.append(Gap(m.text, "condition", "not_applied"))
    return _once(gaps)


def _names_a_figure(question: str, model: Any) -> bool:
    """Does the follow-up name a figure of its own ("show units by month")? Then it may be in place of the one shown."""
    if model is None:
        return False
    from core2.plan.followup import measures_named

    return bool(measures_named(model, question))


def _business(slug: str, model: Any) -> str:
    """A measure's or field's name as the answer heads it ("Number of orders", "Store name")."""
    if model is not None:
        m = next((x for x in model.measures.values() if x.slug == slug), None)
        if m is not None:
            return m.business_name.lower()
        a = model.attributes.get(slug)
        if a is not None:
            return a.business_name.lower()
    return slug.replace("_", " ").replace(".", " ")


def _conditions(plan: Plan) -> list[Any]:
    out = list(plan.filters)
    if plan.absent is not None:
        out += [*plan.absent.without, *plan.absent.had]
    out += [w for r in plan.rates for w in r.where]
    return out


def _once(gaps: list[Gap]) -> list[Gap]:
    seen: set[str] = set()
    out = []
    for g in gaps:
        if _plain(g.words) not in seen:
            seen.add(_plain(g.words))
            out.append(g)
    return out


# ── saying it ─────────────────────────────────────────────────────────────────

_KIND_EN = {"figure": "the figure", "field": "the field", "condition": "the condition", "period": "the period",
            "comparison": "the comparison", "ranking": "the ranking", "form": "", "other": ""}
_KIND_FR = {"figure": "le chiffre", "field": "le champ", "condition": "la condition", "period": "la période",
            "comparison": "la comparaison", "ranking": "le classement", "form": "", "other": ""}


def said(gaps: list[Gap], question: str, *, private: bool = False) -> list[dict[str, Any]]:
    """Each part the answer lacks, as the answer says it, with the question that asks for it again. ``private``:
    the workspace keeps member values from the AI, and the reader's words for a part may hold one (never written
    into what the AI is given: the summary's material leaves them out)."""
    out = []
    for g in gaps:
        words = g.words.strip().strip('"“”')
        if g.kind == "condition":
            ask_en = question if f'"{words}"' in question else question.replace(words, f'"{words}"', 1) \
                if words in question else f'Only "{words}"'
            ask_fr = ask_en
        elif g.kind == "form":
            ask_en, ask_fr = "Show it as a chart", "Montre-le sous forme de graphique"
        else:
            ask_en, ask_fr = f"Add {words}", f"Ajoute {words}"
        out.append({"words": words, "kind": g.kind, "why": g.why, "private": private,
                    "en": f"{_KIND_EN.get(g.kind) or ''} {words}".strip() if g.kind in ("figure", "field") else words,
                    "fr": f"{_KIND_FR.get(g.kind) or ''} {words}".strip() if g.kind in ("figure", "field") else words,
                    "ask": ask_en, "ask_fr": ask_fr})
    return out


def reason(gaps: list[Gap], hide: Callable[[str], str] | None = None) -> tuple[str, str]:
    """What the AI is told when it is asked once more, and its French for the reader's list of checks. ``hide``
    puts back the placeholders of the member names the AI may not see (a workspace whose values are kept from it):
    the parts are the reader's words, and a name among them would reach the AI."""
    shown_to_ai = hide or (lambda words: words)

    def one(g: Gap) -> str:
        why = {"no_plan": "the plan gave nothing for it", "not_in_plan": "nothing in the plan gives it",
               "not_shown": "the answer's columns do not show it", "not_applied": "no condition of the plan holds it",
               "not_kept": "the answer on screen showed it and the reader only added to it"}
        return f'"{shown_to_ai(g.words)}" ({g.kind}: {why.get(g.why, g.why)})'

    en = ("The answer leaves out parts of the request: " + "; ".join(one(g) for g in gaps) + ". Write the plan so "
          "the answer gives every part asked for: a figure the reader names is in measures (or, worked out, a derived "
          "listed in asked by its own name), a field or date to show goes in group_by (a date by time.grain), a member "
          "the reader names is a filter, and what the answer showed before stays when the reader only adds to it. Keep "
          "a part in asked with plan null only when nothing in the catalog can give it.")
    if any(OF_CHANGE.search(g.words) for g in gaps):
        en += (" Each member's share of a change (\"% of the total increase\", \"how much of the growth came from "
               "each\") is given by intent \"compare\" with group_by that grouping, the two periods in time.window "
               "and time.compare, and share_of_change true.")
    fr = "La réponse laisse de côté une partie de la demande : " + ", ".join(f"« {g.words} »" for g in gaps) + "."
    return en, fr
