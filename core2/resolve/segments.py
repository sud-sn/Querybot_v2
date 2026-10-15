"""A segment of members narrowing, or splitting, any other figure: "net sales from returning customers by store",
"net sales: new vs returning customers", "sales from new customers by month".

Each row is counted under its member's segment, decided by the member's own activity (all of it, not only the rows
the figure counts), compiled as one row a member joined to the figure's rows (core2/compile/shapes.py
segment_summary):

* new: their first activity ever is in the period -- each row's own period in a series, the question's period
  otherwise, and the row's own day when the question names no period;
* returning: they had activity before it;
* won back: returning, with none in the period before it (the question's whole period only);
* repeat / one-time: activity on 2 or more days of the question's period / on one day only.
"""

from __future__ import annotations

from core2.bootstrap.names import plural
from core2.model.schema import Entity, SemanticModel
from core2.plan.ir import Plan, TimeSpec
from core2.resolve.resolver import (Condition, Context, Group, PartGroup, ResolveError, SegmentUse, _output_name,
                                    find_slug)
from core2.resolve.time import Range

WORDS = {"new": "new", "returning": "returning", "won_back": "won-back", "repeat": "repeat", "one_time": "one-time"}
SPLITS = {"new": ("New", "Returning"), "returning": ("Returning", "New"), "repeat": ("Repeat", "One-time"),
          "one_time": ("One-time", "Repeat")}
WHOLE_PERIOD = ("won_back", "repeat", "one_time")


def attach(plan: Plan, model: SemanticModel, ctx: Context, builders: list, groups: list[Group],
           conditions: list[Condition], window: Range, intent: str, taken: set[str]) -> str:
    """Narrow (or split) every part's rows by the plan's segment; the answer's intent, a breakdown when split."""
    from core2.resolve.shapes import _events, period_before

    seg = plan.segment
    assert seg is not None
    found = find_slug(model, seg.of)
    if not found or found[0] != "entity":
        raise ResolveError("unknown", f"nothing listed called {seg.of}: name the customers, patients or other things "
                           "whose activity decides who is new or returning", sorted(model.entities)[:12])
    entity = found[1]
    assert isinstance(entity, Entity)
    things = plural(entity.business_name.lower())
    grain = plan.time.grain
    if seg.status in WHOLE_PERIOD and grain:
        raise ResolveError("unsupported", f"{WORDS[seg.status]} {things} are counted over the whole period asked: "
                           "ask without a series, or for new or returning ones")
    if seg.split and seg.status not in SPLITS:
        raise ResolveError("unsupported", f"{WORDS[seg.status]} {things} narrow a figure; a split is new against "
                           "returning, or repeat against one-time")
    # Every day of each member's activity up to the period's end: their first ever, and the days in the period.
    activity = Plan(kind="query", time=TimeSpec(window=plan.time.window))
    events, _ = _events(activity, model, ctx, seg.of, seg.measure, read=lambda w: Range(None, w.end), grain=None)
    bounded = events.window.start is not None and events.window.end is not None
    if seg.status in WHOLE_PERIOD and not bounded:
        raise ResolveError("unsupported", f"{WORDS[seg.status]} {things} are counted over a period: name one")
    prior = period_before(plan.time.window, events.window)[0] if seg.status == "won_back" else None
    per_row = grain if grain else ("window" if events.window.start is not None else "day")
    split = SPLITS[seg.status] if seg.split else None
    name = _output_name(f"{split[0]}_or_{split[1]}".lower().replace("-", "_") if split else "segment", taken)
    label = f"{split[0]} or {split[1].lower()} {entity.business_name.lower()}" if split else ""
    key = entity.key_columns[0]
    for builder, role in builders:
        part = builder.part
        alias = builder.reach(entity.table, label=entity.business_name)
        date = part.date
        if per_row != "window" and date is None:
            if role is None:
                raise ResolveError("unsupported", f"{model.tables[part.table].business_name} has no date to tell "
                                   f"new {things} from returning ones")
            date = builder.date(role)
        use = SegmentUse(alias=alias, column=key, events=events, status=seg.status, window=events.window,
                         per_row=per_row, date=date, prior=prior, split=split, name=f"sg{len(part.segments)}")
        part.segments.append(use)
        if split:
            part.groups.append(PartGroup(name, alias, None, "segment", segment=use))
    if split:
        groups.append(Group(name, label, "attribute"))
        return "breakdown" if intent == "value" else intent
    conditions.append(Condition("segment", f"{WORDS[seg.status]} {things}", op=seg.status, values=[things]))
    return intent
