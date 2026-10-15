"""Groups of members an admin names, by what they did: "Regular shoppers" (customers with orders on 2 or more days of
the period asked), "Won-back patients", "Loyal members".

Saved in the metric editor as a decision (core2/model/overrides.py, kind "segment"), so every Learn keeps them; listed
for readers in the Data guide; asked of by name like any figure: "net sales from Regular shoppers by store" narrows a
figure to them (Plan.segment), "how many Regular shoppers last quarter" counts them (Plan.lifecycle).
"""

from __future__ import annotations

import re
from typing import Any

from core2 import ids
from core2.bootstrap import names
from core2.model.schema import SavedSegment, SemanticModel

ADMIN_PREFIX = "admin:"
# The kinds a saved segment can be, in the editor's order, with what each means.
KINDS = {
    "repeat": ("Repeat", "{things} with {activity} on 2 or more days of the period asked"),
    "one_time": ("One-time", "{things} with {activity} on one day only of the period asked"),
    "new": ("New", "{things} whose first {activity_one} ever is in the period asked"),
    "returning": ("Returning", "{things} with {activity} in the period asked who had some before it"),
    "won_back": ("Won back", "returning {things} who had no {activity} in the period before it"),
    "lapsed": ("Lapsed", "{things} with {activity} in the period before and none in the period asked"),
    "loyal": ("Loyal", "{things} with {activity} in every month of the period asked"),
    "active": ("Active", "{things} with {activity} in the period asked"),
}
# The kinds a figure can be narrowed to ("net sales from Regular shoppers"); the others are counted only.
NARROW = ("new", "returning", "won_back", "repeat", "one_time")


class SegmentError(ValueError):
    pass


def choices(model: SemanticModel) -> list[dict[str, str]]:
    """Whose activity a segment can follow, as the editor offers it: "Customers, by their orders"."""
    from core2.plan.analyses import activities

    out, seen = [], set()
    for a in activities(model):
        if (a.entity.slug, a.measure.slug) in seen or a.entity.role:
            continue
        seen.add((a.entity.slug, a.measure.slug))
        out.append({"of": a.entity.slug, "measure": a.measure.slug,
                    "label": f"{names.plural(a.entity.business_name)}, by their {a.things}"})
    return out


def meaning(model: SemanticModel, segment: SavedSegment) -> str:
    """What a segment holds, in a sentence: "Customers with orders on 2 or more days of the period asked"."""
    entity = model.entities.get(segment.of)
    activity = next((m for m in model.measures.values() if m.slug == segment.measure), None)
    things = names.plural((entity.business_name if entity is not None else segment.of).lower())
    done = (activity.business_name if activity is not None else segment.measure).lower()
    done = done.removeprefix("number of ").removeprefix("count of ")
    said = KINDS[segment.kind][1].format(things=things, activity=done, activity_one=names.singular(done))
    return said[:1].upper() + said[1:]


def listing(model: SemanticModel) -> list[dict[str, Any]]:
    """The saved segments, as the metric editor and the Data guide list them."""
    rows = []
    for s in sorted(model.segments.values(), key=lambda s: s.business_name.casefold()):
        rows.append({"key": s.key, "name": s.business_name, "kind": s.kind, "kind_name": KINDS[s.kind][0],
                     "of": s.of, "measure": s.measure, "meaning": meaning(model, s), "description": s.description,
                     "also": [w for ws in s.synonyms.values() for w in ws], "hidden": s.hidden,
                     "example": example(model, s)})
    return rows


def example(model: SemanticModel, segment: SavedSegment) -> str:
    """A question a reader could ask of it: a figure narrowed to it, or a count of it."""
    if segment.kind in NARROW:
        from core2.plan.analyses import leading

        activity = next((m for m in model.measures.values() if m.slug == segment.measure), None)
        # The data's main figure where it is on the activity's rows (net sales on orders), else any amount there.
        figure = leading(model)
        if figure is None or activity is None or figure.table != activity.table:
            figure = next((m for m in model.measures.values() if activity is not None and m.table == activity.table
                           and not m.hidden and getattr(m.expr, "agg", "") == "sum"), None)
        if figure is not None:
            return f"{figure.business_name} from {segment.business_name} last quarter"
    return f"How many {segment.business_name} last quarter?"


def build(model: SemanticModel, data: dict[str, Any], *, key: str = "") -> SavedSegment:
    """A segment from the editor's form, checked against the model: its name, whose activity, which kind."""
    name = " ".join(str(data.get("name") or "").split())
    if not name:
        raise SegmentError("Give the segment the name people use for it.")
    if any(s.business_name.casefold() == name.casefold() and s.key != key for s in model.segments.values()):
        raise SegmentError(f"There is already a segment called {name}.")
    if any(m.business_name.casefold() == name.casefold() and not m.hidden for m in model.measures.values()):
        raise SegmentError(f"{name} is already the name of a metric: name the segment another way.")
    kind = str(data.get("kind") or "")
    if kind not in KINDS:
        raise SegmentError("Choose which members it holds: repeat, new, returning, and so on.")
    of, measure = str(data.get("of") or ""), str(data.get("measure") or "")
    if not any(c["of"] == of and c["measure"] == measure for c in choices(model)):
        raise SegmentError("Choose whose activity it follows, from the list.")
    words = [w.strip() for w in re.split(r"[,;\n]", str(data.get("also") or "")) if w.strip()]
    existing = model.segments.get(key) if key else None
    taken = {s.slug for s in model.segments.values() if s.key != key}
    slug = existing.slug if existing is not None else ids.unique_slug(ids.slug(name, fallback="segment"), taken)
    return SavedSegment(key=key or ADMIN_PREFIX + slug, slug=slug, business_name=name,
                        description=" ".join(str(data.get("description") or "").split())[:500],
                        synonyms={"en": words} if words else {}, of=of, measure=measure,
                        kind=kind, hidden=bool(data.get("hidden")))  # type: ignore[arg-type]


def named(question: str, model: SemanticModel) -> SavedSegment | None:
    """The saved segment a question names, by its name or another word for it ("regular shoppers"); the longest
    name first, so "Loyal VIP customers" wins over "VIP customers"."""
    text = " " + " ".join(re.findall(r"[\w'-]+", (question or "").casefold())) + " "
    found = []
    for s in model.segments.values():
        if s.hidden:
            continue
        for word in [s.business_name, *(w for ws in s.synonyms.values() for w in ws)]:
            plain = " ".join(re.findall(r"[\w'-]+", word.casefold()))
            forms = {plain, names.singular(plain), names.plural(plain)} if plain else set()
            if any(f and f" {f} " in text for f in forms):
                found.append((len(plain), s))
    return max(found, key=lambda f: f[0])[1] if found else None
