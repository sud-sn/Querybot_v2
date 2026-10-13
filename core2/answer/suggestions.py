"""Next questions worth one click, written so the planner reads them like a reader's own.

Each suggestion follows from the answer on screen and only offers what the new
core answers: a "why" for a bounded period (issue G1), a forecast for a series
at a grain that has one, a breakdown by the grouping the measure's table reaches
first, a month-by-month view of the member leading the answer. Nothing is offered
that would fail when clicked (issue G3): no comparison of a series, no "why"
without a period, no forecast of a daily or yearly series.
"""

from __future__ import annotations

from typing import Any

from core2.answer.builder import condition_words, span_words
from core2.answer.drivers import Grouping, candidates
from core2.model.schema import SemanticModel, value_names
from core2.plan.ir import Plan
from core2.resolve import paths as P
from core2.resolve.resolver import Logical, adds_up, measure_dates
from core2.resolve.time import label as period_label

_FORECAST_GRAINS = {"month": "months", "quarter": "quarters", "week": "weeks"}


def _word(model: SemanticModel, slug: str) -> str:
    """A grouping slug as a reader says it ("customer", not "customer name")."""
    if slug in model.entities:
        name = model.entities[slug].business_name
    elif slug in model.attributes:
        name = model.attributes[slug].business_name
        column = model.columns.get(model.attributes[slug].column)
        yes = value_names(column).get("1") if column is not None and column.role == "flag" else None
        if yes:
            return f"{yes[:1].lower() + yes[1:]} or not"       # "by sterile or not", never "by is sterile"
    else:
        return ""
    return Grouping(slug, name, []).word


def follow_ups(plan: Plan, logical: Logical, payload: dict[str, Any], model: SemanticModel,
               allowed: set[str] | None) -> list[dict[str, str]]:
    """The next questions, each as its words and the question a click sends, in English and in French (a French
    reader's click sends the French question; the data's own names stay as they are)."""
    measures = [m for m in logical.measures]
    if not measures or logical.intent == "list":
        return []
    # The conditions the answer applied go into every next question, so a click never widens it; members in
    # quotes, since a name narrows an answer only when the question quotes it
    # ("... by month in 2025, where days from order to invoice is above 10 days").
    words = _Words(logical, "en")
    fr = _Words(logical, "fr")
    bounded = logical.window.start is not None and logical.window.end is not None
    rows = (payload.get("data") or {}).get("rows") or []
    grouped = [g for g in logical.groups if g.kind == "attribute" and g.name != logical.unit_group]
    period = next((g for g in logical.groups if g.kind == "period"), None)
    out: list[tuple[str, str, str]] = []                          # (kind, English, French)

    if period is not None and not grouped:
        complete = [r for r in rows if r.get(period.name) and str(r.get(period.name)) not in
                    {d.isoformat() for d in logical.partial}]
        if complete and period.grain:
            last = complete[-1][period.name]
            try:
                import datetime as dt

                day = dt.date.fromisoformat(str(last)[:10])
                label = period_label(day, period.grain, fiscal_start=logical.fiscal_start)
                label_fr = _period_fr(day, period.grain, logical.fiscal_start)
                out.append(("why", f"Why did {words.lower} change in {label}{words.tail}?",
                            f"Pourquoi {fr.lower} a-t-il varié en {label_fr}{fr.tail} ?"))
            except ValueError:
                pass
        if period.grain in _FORECAST_GRAINS:
            out.append(("forecast", f"Forecast {words.lower} for the next 3 {_FORECAST_GRAINS[period.grain]}{words.tail}",
                        f"Prévision de {fr.lower} pour {_NEXT_FR[period.grain]}{fr.tail}"))
    elif grouped and period is None:
        top = next((str(r.get(grouped[0].name)) for r in rows
                    if r.get(grouped[0].name) not in (None, "", "Unknown")), "")
        # A class, a stage or a pair an analysis made is no member a question can name.
        if top and (not plan.shaped() or (grouped[0].attribute and measures[0].measure is not None)):
            out.append(("member", f'Monthly {words.lower} for "{top}"{words.tail}',
                        f'{fr.m} par mois pour "{top}"{fr.tail}'))
        if bounded and logical.compare is None:
            out.append(("why", f"Why did {words.lower} change {words.span}{words.tail}?",
                        f"Pourquoi {fr.lower} a-t-il varié {fr.span}{fr.tail} ?"))
        if not logical.share and logical.compare is None and logical.measures and adds_up(logical.measures[0]):
            word = (_word(model, grouped[0].attribute) if grouped[0].attribute else "") or \
                Grouping("", grouped[0].label, []).word
            out.append(("share", f"Share of {words.lower} by {word} {words.span}".strip() + words.tail,
                        f"Part de {fr.lower} par {word} {fr.span}".strip() + fr.tail))
    elif not grouped and period is None:
        if bounded:
            out.append(("why", f"Why did {words.lower} change {words.span}{words.tail}?",
                        f"Pourquoi {fr.lower} a-t-il varié {fr.span}{fr.tail} ?"))
        out.append(("trend", (f"{words.m} by month {words.span}".strip() if bounded
                              else f"{words.m} by month for the last 12 months") + words.tail,
                    (f"{fr.m} par mois {fr.span}".strip() if bounded
                     else f"{fr.m} par mois sur les 12 derniers mois") + fr.tail))

    if len(out) < 3 and not grouped and logical.parts:
        skip = {f.field for f in plan.filters}
        slugs = _apart(model, logical.parts[0].table, _one_member(plan),
                       candidates(model, logical.parts[0].table, allowed=allowed, skip=skip, limit=20))
        word = _word(model, slugs[0]) if slugs else ""
        if word:
            out.append(("by", f"{words.m} by {word} {words.span}".strip() + words.tail,
                        f"{fr.m} par {word} {fr.span}".strip() + fr.tail))
    if plan.shaped():
        # A figure per member, a running total or a rank is not yet explained, forecast or shared out.
        out = [s for s in out if s[0] not in ("why", "forecast", "share")]
    seen: set[str] = set()
    unique = [s for s in out if not (s[1].lower() in seen or seen.add(s[1].lower()))]   # type: ignore[func-returns-value]
    return [{"label": en, "question": en, "label_fr": french, "question_fr": french} for _, en, french in unique[:3]]


_NEXT_FR = {"month": "les 3 prochains mois", "quarter": "les 3 prochains trimestres",
            "week": "les 3 prochaines semaines"}


def _period_fr(day: Any, grain: str, fiscal_start: int | None) -> str:
    """A period as a French question says it: "déc. 2025", "T1 2026", "semaine du 2 juin 2026"."""
    from core2.answer.facts import period

    text = period(day, grain, lang="fr", fiscal_start=fiscal_start)
    if grain in ("quarter", "fiscal_quarter"):
        text = text.replace("Q", "T")
    return text[:1].lower() + text[1:] if text.startswith("Semaine") else text


class _Words:
    """The pieces of a next question in one language: the figure (with what it counted), its period, and the
    conditions the answer applied."""

    def __init__(self, logical: Logical, lang: str):
        before = " ".join(condition_words(c, lang=lang) for c in logical.conditions if c.kind == "activity")
        by = " ".join(condition_words(c, lang=lang) for c in logical.conditions if c.kind == "by")
        rest = ", ".join(w for w in (condition_words(c, quoted=True, lang=lang) for c in logical.conditions
                                     if c.kind not in ("activity", "by")) if w)
        self.tail = f", {rest}" if rest else ""
        label = logical.measures[0].label
        self.m = label + (f" {before}" if before else "")
        self.lower = label.lower() + (f" {before}" if before else "")
        self.span = " ".join(w for w in (span_words(logical.window, lang), by) if w)


def _table_of(model: SemanticModel, slug: str) -> str | None:
    if slug in model.entities:
        return model.entities[slug].table
    attribute = model.attributes.get(slug)
    return model.columns[attribute.column].table if attribute and attribute.column in model.columns else None


def _apart(model: SemanticModel, table: str, fixed: list[str | None], slugs: list[str]) -> list[str]:
    """The groupings a member of each ``fixed`` field splits into more than one part. Not a field of the member's
    own row, nor one the answer's rows reach through it: a store's region, a product's category, a customer's
    segment are one value each, a single bar. A customer's purchases by store stay, the sales row naming its own
    store. Only a member (a store, by its name or code) is so; a customer's city holds customers of every segment.
    ``fixed``: the field grouped by, and each field the answer keeps to one member of (North, drilled into)."""
    owners = set()
    for slug in fixed:
        entity = model.entities.get(slug or "")
        if entity is None and slug in model.attributes:
            column = model.attributes[slug].column
            entity = next((e for e in model.entities.values()
                           if column in (e.label_column, e.code_column, *e.key_columns)), None)
        if entity is not None:
            owners.add(entity.table)
    if not owners:
        return slugs
    kept = []
    for slug in slugs:
        target = _table_of(model, slug)
        try:
            path = P.best_path(model, table, target) if target else None
        except P.Ambiguous:
            path = None
        if target in owners or path is not None and owners & set(path.tables[1:]):
            continue
        kept.append(slug)
    return kept


def _one_member(plan: Plan) -> list[str]:
    """The fields the answer keeps to one member of: "for store Eastview Store", a tile drilled into North."""
    return [f.field for f in plan.filters if f.op in ("in", "eq") and len(f.values) == 1]


def breakdowns(plan: Plan, logical: Logical, model: SemanticModel, allowed: set[str] | None) -> list[dict[str, str]]:
    """What a member of this answer can be broken down by, most telling first, as fields: a dashboard tile's
    drill-down (portal/routes.py) opens "North by store" from the region tile. The same groupings a click's "{member}
    by store" question names in the chat (``drills``), five at most. None for an answer over time, a shaped one (per
    member, bands, a funnel ...), a list, or one grouped by more than one field."""
    if not logical.measures or logical.intent == "list" or not logical.parts or plan.shaped():
        return []
    grouped = [g for g in logical.groups if g.kind == "attribute" and g.name != logical.unit_group]
    if len(grouped) != 1 or any(g.kind == "period" for g in logical.groups):
        return []
    shown = {g.attribute for g in logical.groups if g.attribute} | set(plan.group_by)
    skip = {f.field for f in plan.filters} | shown | {s.split(".", 1)[0] for s in shown}
    table = logical.parts[0].table
    slugs = _apart(model, table, [grouped[0].attribute, *_one_member(plan)],
                   candidates(model, table, allowed=allowed, skip=skip, limit=20))
    out = [{"field": slug, "label": _word(model, slug)} for slug in slugs[:5] if _word(model, slug)]
    # And its months, where the measure is counted by a date ("North by month", as the chat's menu offers it).
    if measure_dates(plan, model)[0] is not None:
        out.append({"field": "time:month", "label": "month"})
    return out


def drills(plan: Plan, logical: Logical, payload: dict[str, Any], model: SemanticModel,
           allowed: set[str] | None) -> dict[str, Any] | None:
    """What a click on the chart opens next: a few questions with the clicked member or period in place.

    A member (a bar, a slice, a dot of a comparison): its month-by-month trend when the
    measure is counted by a date, the member broken down by the grouping its table reaches
    first, and why it changed when the answer covers a bounded period. A period (a point on a
    time line): that period broken down by that grouping, and why the measure changed in it;
    on a line per member, both for the member of the line clicked. "{member}" and "{period}"
    are filled in by the page with what was clicked, as the chart names it. Each question
    keeps the answer's conditions, so a click never widens what was asked.
    """
    chart = payload.get("chart") or {}
    if not chart or not logical.measures or logical.intent == "list":
        return None
    w, fr = _Words(logical, "en"), _Words(logical, "fr")
    bounded = logical.window.start is not None and logical.window.end is not None
    period = next((g for g in logical.groups if g.kind == "period"), None)
    grouped = [g for g in logical.groups if g.kind == "attribute" and g.name != logical.unit_group]
    # Not the grouping on screen again, by any of its names ("store", "store.name").
    shown = {g.attribute for g in logical.groups if g.attribute} | set(plan.group_by)
    skip = {f.field for f in plan.filters} | shown | {s.split(".", 1)[0] for s in shown}
    slugs = candidates(model, logical.parts[0].table, allowed=allowed, skip=skip, limit=20) if logical.parts else []
    if logical.parts:
        slugs = _apart(model, logical.parts[0].table, [grouped[0].attribute if grouped else None, *_one_member(plan)],
                       slugs)
    word = _word(model, slugs[0]) if slugs else ""
    items: list[dict[str, str]] = []

    def item(label: str, question: str, label_fr: str, question_fr: str) -> None:
        items.append({"label": label, "question": question, "label_fr": label_fr, "question_fr": question_fr})

    if period is not None:
        member = "{member} " if grouped else ""
        for_member = ' for "{member}"' if grouped else ""
        pour = ' pour "{member}"' if grouped else ""
        if word:
            item(f"{{member}} in {{period}} by {word}" if grouped else f"{{period}} by {word}",
                 f"{w.m}{for_member} by {word} in {{period}}{w.tail}",
                 f"{{member}} en {{period}} par {word}" if grouped else f"{{period}} par {word}",
                 f"{fr.m}{pour} par {word} en {{period}}{fr.tail}")
        item(f"Why {member}changed in {{period}}", f"Why did {w.lower} change{for_member} in {{period}}{w.tail}?",
             "Pourquoi {member} a varié en {period}" if grouped else "Pourquoi cette variation en {period}",
             f"Pourquoi {fr.lower} a-t-il varié{pour} en {{period}}{fr.tail} ?")
        return {"on": "period", "series": "member" if grouped else "", "items": items}
    if not grouped or (plan.shaped() and not grouped[0].attribute) or chart.get("x_pair"):
        return None
    long_window = bounded and (logical.window.end - logical.window.start).days > 80   # type: ignore[operator]
    if measure_dates(plan, model)[0] is not None:
        when, quand = (w.span, fr.span) if long_window else ("for the last 12 months", "sur les 12 derniers mois")
        item("{member} by month", f'{w.m} by month for "{{member}}" {when}'.strip() + w.tail,
             "{member} par mois", f'{fr.m} par mois pour "{{member}}" {quand}'.strip() + fr.tail)
    if word:
        item(f"{{member}} by {word}", f'{w.m} for "{{member}}" by {word} {w.span}'.strip() + w.tail,
             f"{{member}} par {word}", f'{fr.m} pour "{{member}}" par {word} {fr.span}'.strip() + fr.tail)
    if bounded and logical.compare is None:
        item("Why {member} changed", f'Why did {w.lower} change for "{{member}}" {w.span}{w.tail}?',
             "Pourquoi {member} a varié", f'Pourquoi {fr.lower} a-t-il varié pour "{{member}}" {fr.span}{fr.tail} ?')
    return {"on": "member", "series": "", "items": items} if items else None
