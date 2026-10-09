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
from core2.model.schema import SemanticModel
from core2.plan.ir import Plan
from core2.resolve.resolver import Logical, measure_dates
from core2.resolve.time import label as period_label

_FORECAST_GRAINS = {"month": "months", "quarter": "quarters", "week": "weeks"}


def _word(model: SemanticModel, slug: str) -> str:
    """A grouping slug as a reader says it ("customer", not "customer name")."""
    if slug in model.entities:
        name = model.entities[slug].business_name
    elif slug in model.attributes:
        name = model.attributes[slug].business_name
    else:
        return ""
    return Grouping(slug, name, []).word


def follow_ups(plan: Plan, logical: Logical, payload: dict[str, Any], model: SemanticModel,
               allowed: set[str] | None) -> list[dict[str, str]]:
    measures = [m for m in logical.measures]
    if not measures or logical.intent == "list":
        return []
    # The conditions the answer applied go into every next question, so a click never widens it
    # ("... by month in 2025, where days from order to invoice is above 10 days").
    before = " ".join(condition_words(c) for c in logical.conditions if c.kind == "activity")
    by = " ".join(condition_words(c) for c in logical.conditions if c.kind == "by")
    rest = ", ".join(w for w in (condition_words(c) for c in logical.conditions
                                 if c.kind not in ("activity", "by")) if w)
    tail = f", {rest}" if rest else ""
    m = measures[0].label + (f" {before}" if before else "")
    lower = measures[0].label.lower() + (f" {before}" if before else "")
    span = " ".join(w for w in (span_words(logical.window), by) if w)
    bounded = logical.window.start is not None and logical.window.end is not None
    rows = (payload.get("data") or {}).get("rows") or []
    grouped = [g for g in logical.groups if g.kind == "attribute" and g.name != logical.unit_group]
    period = next((g for g in logical.groups if g.kind == "period"), None)
    out: list[str] = []

    if period is not None and not grouped:
        complete = [r for r in rows if r.get(period.name) and str(r.get(period.name)) not in
                    {d.isoformat() for d in logical.partial}]
        if complete and period.grain:
            last = complete[-1][period.name]
            try:
                import datetime as dt

                label = period_label(dt.date.fromisoformat(str(last)[:10]), period.grain,
                                     fiscal_start=logical.fiscal_start)
                out.append(f"Why did {lower} change in {label}{tail}?")
            except ValueError:
                pass
        if period.grain in _FORECAST_GRAINS:
            out.append(f"Forecast {lower} for the next 3 {_FORECAST_GRAINS[period.grain]}{tail}")
    elif grouped and period is None:
        top = next((str(r.get(grouped[0].name)) for r in rows
                    if r.get(grouped[0].name) not in (None, "", "Unknown")), "")
        if top:
            out.append(f"Monthly {lower} for {top}{tail}")
        if bounded and logical.compare is None:
            out.append(f"Why did {lower} change {span}{tail}?")
        if not logical.share and logical.compare is None:
            out.append(f"Share of {lower} by {Grouping('', grouped[0].label, []).word} {span}".strip() + tail)
    elif not grouped and period is None:
        if bounded:
            out.append(f"Why did {lower} change {span}{tail}?")
        out.append((f"{m} by month {span}".strip() if bounded else f"{m} by month for the last 12 months") + tail)

    if len(out) < 3 and not grouped and logical.parts:
        skip = {f.field for f in plan.filters}
        slugs = candidates(model, logical.parts[0].table, allowed=allowed, skip=skip)
        word = _word(model, slugs[0]) if slugs else ""
        if word:
            out.append(f"{m} by {word} {span}".strip() + tail)
    seen: set[str] = set()
    unique = [s for s in out if not (s.lower() in seen or seen.add(s.lower()))]   # type: ignore[func-returns-value]
    return [{"label": s, "question": s} for s in unique[:3]]


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
    before = " ".join(condition_words(c) for c in logical.conditions if c.kind == "activity")
    by = " ".join(condition_words(c) for c in logical.conditions if c.kind == "by")
    rest = ", ".join(w for w in (condition_words(c) for c in logical.conditions
                                 if c.kind not in ("activity", "by")) if w)
    tail = f", {rest}" if rest else ""
    m = logical.measures[0].label + (f" {before}" if before else "")
    lower = logical.measures[0].label.lower() + (f" {before}" if before else "")
    span = " ".join(w for w in (span_words(logical.window), by) if w)
    bounded = logical.window.start is not None and logical.window.end is not None
    period = next((g for g in logical.groups if g.kind == "period"), None)
    grouped = [g for g in logical.groups if g.kind == "attribute" and g.name != logical.unit_group]
    # Not the grouping on screen again, by any of its names ("store", "store.name").
    shown = {g.attribute for g in logical.groups if g.attribute} | set(plan.group_by)
    skip = {f.field for f in plan.filters} | shown | {s.split(".", 1)[0] for s in shown}
    slugs = candidates(model, logical.parts[0].table, allowed=allowed, skip=skip) if logical.parts else []
    word = _word(model, slugs[0]) if slugs else ""
    items: list[dict[str, str]] = []
    if period is not None:
        member = "{member} " if grouped else ""
        for_member = " for {member}" if grouped else ""
        if word:
            items.append({"label": f"{{member}} in {{period}} by {word}" if grouped else f"{{period}} by {word}",
                          "question": f"{m}{for_member} by {word} in {{period}}{tail}"})
        items.append({"label": f"Why {member}changed in {{period}}",
                      "question": f"Why did {lower} change{for_member} in {{period}}{tail}?"})
        return {"on": "period", "series": "member" if grouped else "", "items": items}
    if not grouped:
        return None
    long_window = bounded and (logical.window.end - logical.window.start).days > 80   # type: ignore[operator]
    if measure_dates(plan, model)[0] is not None:
        when = span if long_window else "for the last 12 months"
        items.append({"label": "{member} by month",
                      "question": f"{m} by month for {{member}} {when}".strip() + tail})
    if word:
        items.append({"label": f"{{member}} by {word}", "question": f"{m} for {{member}} by {word} {span}".strip() + tail})
    if bounded and logical.compare is None:
        items.append({"label": "Why {member} changed", "question": f"Why did {lower} change for {{member}} {span}{tail}?"})
    return {"on": "member", "series": "", "items": items} if items else None
