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

from core2.answer.builder import span_words
from core2.answer.drivers import Grouping, candidates
from core2.model.schema import SemanticModel
from core2.plan.ir import Plan
from core2.resolve.resolver import Logical
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
    m = measures[0].label
    lower = m.lower()
    span = span_words(logical.window)
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
                out.append(f"Why did {lower} change in {label}?")
            except ValueError:
                pass
        if period.grain in _FORECAST_GRAINS:
            out.append(f"Forecast {lower} for the next 3 {_FORECAST_GRAINS[period.grain]}")
    elif grouped and period is None:
        top = next((str(r.get(grouped[0].name)) for r in rows
                    if r.get(grouped[0].name) not in (None, "", "Unknown")), "")
        if top:
            out.append(f"Monthly {lower} for {top}")
        if bounded and logical.compare is None:
            out.append(f"Why did {lower} change {span}?")
        if not logical.share and logical.compare is None:
            out.append(f"Share of {lower} by {Grouping('', grouped[0].label, []).word} {span}".strip())
    elif not grouped and period is None:
        if bounded:
            out.append(f"Why did {lower} change {span}?")
        out.append(f"{m} by month {span}".strip() if bounded else f"{m} by month for the last 12 months")

    if len(out) < 3 and not grouped and logical.parts:
        skip = {f.field for f in plan.filters}
        slugs = candidates(model, logical.parts[0].table, allowed=allowed, skip=skip)
        word = _word(model, slugs[0]) if slugs else ""
        if word:
            out.append(f"{m} by {word} {span}".strip())
    seen: set[str] = set()
    unique = [s for s in out if not (s.lower() in seen or seen.add(s.lower()))]   # type: ignore[func-returns-value]
    return [{"label": s, "question": s} for s in unique[:3]]
