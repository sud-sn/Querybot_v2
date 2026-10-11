"""People's ages, read from a birth date an admin allowed, and never a group small enough to point at someone.

A birth date is a personal detail: never shown, sorted, filtered on or sent to the AI. Learn proposes a person's
birth date as "usable through age"; once an admin confirms it on the Knowledge base page, questions may group by
the person's age (in bands) or keep only some ages, and nothing else. The AI is told only that an age exists.

* an age is whole years on the day the question is asked, worked out the same way on every warehouse; a
  placeholder birth date (before 1901, or after today) gives no age, and the people without one are counted apart;
* ages are taken in steps of 5 years ("over 63" is 65 and over), so two answers never differ by fewer years;
* every group of an answer that uses age must hold at least 11 people. A smaller one is not shown, nor anything
  built on it (its share, a total, a chart, a chip or an export); and when only one is hidden, the next smallest
  is hidden too, so the hidden group cannot be worked out from the total;
* an age limit is refused when the 5 years just inside it hold fewer than 11 people, so two answers either side
  of it never isolate fewer than 11;
* an answer that uses age is never summarised by the AI, and each use is logged.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import math
from dataclasses import dataclass

from core2.model.schema import AggExpr, Entity, SemanticModel
from core2.plan.ir import Filter, Plan
from core2.resolve import bands as B

log = logging.getLogger("core2.age")

MIN_PEOPLE = 11
STEP = 5
EDGES = [20, 35, 50, 65, 80]          # age bands when the question names none
PEOPLE = "people_counted"            # each group's distinct people: worked with, never shown
_LOW = dt.date(1901, 1, 1)


@dataclass(frozen=True)
class AgeField:
    slug: str                        # "patient.age"
    entity: Entity
    column: str                      # the birth date's column key (never shown)
    key: str                         # the entity's key column: who is counted once


def age_fields(model: SemanticModel) -> dict[str, AgeField]:
    """The ages an admin allowed, by slug: one per entity whose birth date is confirmed usable through age."""
    out: dict[str, AgeField] = {}
    for column in model.columns.values():
        if column.age != "confirmed" or column.hidden:
            continue
        entity = next((e for e in model.entities.values() if e.table == column.table and len(e.key_columns) == 1),
                      None)
        if entity is not None:
            slug = f"{entity.slug}.age"
            out.setdefault(slug, AgeField(slug, entity, column.key, entity.key_columns[0]))
    return out


def uses_age(plan: Plan, model: SemanticModel) -> bool:
    ages = age_fields(model) if model.columns else {}
    return bool(ages) and (any(g in ages for g in plan.group_by) or any(f.field in ages for f in plan.filters)
                           or (plan.bands is not None and plan.bands.of in ages))


def snap(value: float) -> int:
    return int(STEP * math.floor(float(value) / STEP + 0.5))


@dataclass
class AgeRange:
    """Ages from ``low`` (inclusive) to ``high`` (exclusive), each a multiple of 5; None is open."""

    low: int | None
    high: int | None

    def words(self) -> str:
        if self.low is not None and self.high is not None:
            return f"aged {self.low} to {self.high - 1}"
        if self.low is not None:
            return f"aged {self.low} and over"
        return f"aged under {self.high}"

    def edge_blocks(self) -> list[tuple[int, int]]:
        """The 5 years just inside each limit: what two answers either side of it would differ by."""
        blocks = []
        if self.low is not None:
            blocks.append((self.low, self.low + STEP - 1))
        if self.high is not None and (self.low is None or self.high - STEP != self.low):
            blocks.append((self.high - STEP, self.high - 1))
        return blocks


def age_range(op: str, values: list) -> AgeRange:
    """A condition on age as ages in steps of 5: "over 63" is 65 and over, "under 18" under 20."""
    from core2.resolve.resolver import ResolveError

    try:
        numbers = [float(v) for v in values]
    except (TypeError, ValueError):
        raise ResolveError("unsupported", "an age is compared with a number of years") from None
    if op in ("gte", "gt") and len(numbers) == 1:
        return AgeRange(snap(numbers[0]), None)
    if op in ("lt", "lte") and len(numbers) == 1:
        return AgeRange(None, max(STEP, snap(numbers[0] + (1 if op == "lte" else 0))))
    if op == "between" and len(numbers) == 2:
        low, high = snap(min(numbers)), snap(max(numbers) + 1)
        return AgeRange(low, max(high, low + STEP))
    if op == "eq" and len(numbers) == 1:
        low = int(STEP * math.floor(numbers[0] / STEP))
        return AgeRange(low, low + STEP)
    raise ResolveError("unsupported", "an age is kept above, below or between a number of years")


def resolve_with_age(plan: Plan, model: SemanticModel, ctx) -> object:
    """The plan's answer with age as a grouping or a condition, each group counting its people."""
    from core2.resolve.resolver import (AgePred, Condition, Group, OutMeasure, PartGroup, ResolveError, _Aliases,
                                        _output_name, _PartBuilder, resolve_plain)

    ages = age_fields(model)
    if plan.shaped() and not (plan.bands is not None and plan.bands.of in ages and plan.bands.per is None):
        raise ResolveError("unsupported", "an age is grouped by or kept to a range in a plain answer; a figure per "
                           "member, a running total or a list of who had none is not worked out with it")
    if plan.time.compare is not None:
        raise ResolveError("unsupported", "ages are not compared between two periods: ask for each period")
    if plan.intent == "list":
        raise ResolveError("unsupported", "people are never listed with their age: ask how many there are in each "
                           "age band")
    if any(s.by in ages for s in plan.sort) or any(m in ages for m in plan.measures):
        raise ResolveError("unsupported", "an age is never sorted by or added up: group by it in bands")
    used = {*(g for g in plan.group_by if g in ages), *(f.field for f in plan.filters if f.field in ages),
            *([plan.bands.of] if plan.bands is not None and plan.bands.of in ages else [])}
    if len(used) > 1:
        raise ResolveError("unsupported", "one person's age at a time")
    field = ages[next(iter(used))]
    grouped = field.slug in plan.group_by or (plan.bands is not None and plan.bands.of == field.slug)
    edges = sorted({snap(e) for e in (plan.bands.edges if plan.bands is not None and plan.bands.of == field.slug
                                      else []) if snap(e) > 0}) or EDGES
    ranges = [age_range(f.op, list(f.values)) for f in plan.filters if f.field == field.slug]
    plain = plan.model_copy(update={
        "group_by": [g for g in plan.group_by if g != field.slug],
        "filters": [f for f in plan.filters if f.field != field.slug],
        "bands": None if plan.bands is not None and plan.bands.of == field.slug else plan.bands})
    logical = resolve_plain(plain, model, ctx)
    if len(logical.parts) != 1:
        raise ResolveError("unsupported", "an age is used with measures of one table at a time")
    part = logical.parts[0]
    aliases = _Aliases()
    aliases.taken |= {part.alias, *(j.alias for j in part.joins), *(j.alias for s in part.semis for j in s.joins)}
    builder = _PartBuilder(model, part.table, aliases, ctx)
    builder.part = part
    alias = builder.reach(field.entity.table, label=field.entity.business_name)
    part.reached[field.entity.table] = alias
    on = ctx.today
    for r in ranges:
        if r.low is not None:
            part.age_preds.append(AgePred(alias, field.column, "gte", [r.low], on))
        if r.high is not None:
            part.age_preds.append(AgePred(alias, field.column, "lt", [r.high], on))
    taken = {g.name for g in logical.groups} | {o.name for o in logical.measures}
    groups = list(logical.groups)
    if grouped:
        labels = B.labels(edges, whole=True)
        name = _output_name("age_band", taken)
        part.groups.append(PartGroup(name, alias, field.column, "band", edges=list(edges), labels=labels,
                                     style={"whole": True}, age_on=on))
        groups.append(Group(name, "Age band", "attribute", bands=labels))
    people = _output_name(PEOPLE, taken)
    part.measures.append(OutMeasure(name=people, label="People", expr=AggExpr(agg="count_distinct", column=field.key),
                                    format="count", measure=None, hidden=True))
    who = field.entity.business_name.lower()
    notes = [*logical.notes, f"Ages are whole years on {on.day} {on:%b %Y}, from each {who}'s birth date, taken in "
             f"steps of {STEP} years; a group of fewer than {MIN_PEOPLE} {who}s is not shown."]
    from core2.bootstrap.names import plural

    conditions = [*logical.conditions, *(Condition("age", f"for {plural(who)} {r.words()}") for r in ranges)]
    sort = logical.sort
    if grouped and not plan.sort and not plan.time.grain:
        sort = []            # age bands in their own order
    return dataclasses.replace(logical, groups=groups, notes=notes, conditions=conditions, sort=sort,
                               people=people, age_checks=[b for r in ranges for b in r.edge_blocks()],
                               age_slug=field.slug)


def edge_check_plan(plan: Plan, slug: str, low: int, high: int) -> Plan:
    """The plan counting only the people aged ``low`` to ``high``, with its other conditions and no grouping."""
    return plan.model_copy(update={
        "group_by": [], "sort": [], "limit": None, "bands": None, "intent": "value",
        "time": plan.time.model_copy(update={"grain": None}),
        "filters": [*(f for f in plan.filters if f.field != slug), Filter(field=slug, op="between", values=[low, high])]})


# ── the answer: no group of fewer than 11 people ───────────────────────────


@dataclass
class Protected:
    columns: list[str]
    rows: list[tuple]
    hidden: int = 0               # groups not shown
    unknown: int | None = None    # people with no usable birth date (None: none, or not grouped by age)


def protect(columns: list[str], rows: list[tuple], *, people: str, band: str | None,
            shares: list[str]) -> Protected:
    """The rows an answer may show: every group with at least 11 people; when one alone would be hidden, the next
    smallest too, until what is hidden holds 11 or more. The people with no age are counted apart, never a
    group. A share is of the groups shown."""
    names = [c.casefold() for c in columns]
    at = names.index(people.casefold())
    band_at = names.index(band.casefold()) if band and band.casefold() in names else None
    unknown = None
    kept, hidden = [], []
    for row in rows:
        count = int(row[at] or 0)
        if band_at is not None and row[band_at] is None:
            unknown = (unknown or 0) + count
            continue
        (kept if count >= MIN_PEOPLE else hidden).append(row)
    if hidden and len(rows) > 1:
        kept.sort(key=lambda r: int(r[at] or 0))
        while kept and (len(hidden) < 2 or sum(int(r[at] or 0) for r in hidden) < MIN_PEOPLE):
            hidden.append(kept.pop(0))
        order = {id(r): i for i, r in enumerate(rows)}
        kept.sort(key=lambda r: order[id(r)])
    for share in shares:
        i = names.index(share.casefold()) if share.casefold() in names else None
        if i is None:
            continue
        base = names.index(share.casefold().removesuffix("_share")) if \
            share.casefold().removesuffix("_share") in names else None
        whole = sum(float(r[base] or 0) for r in kept) if base is not None else 0.0
        kept = [tuple((float(r[base] or 0) / whole if whole else None) if j == i else v for j, v in enumerate(r))
                for r in kept]
    return Protected(columns, kept, hidden=len(hidden), unknown=unknown)


def used(logical, account: str = "") -> None:
    """Each use of an age is logged: which age, how it was used, by which workspace."""
    log.info("core2 age used: %s%s, grouped=%s, conditions=%s", logical.age_slug,
             f" by {account}" if account else "", any(g.bands and g.label == "Age band" for g in logical.groups),
             [c.label for c in logical.conditions if c.kind == "age"])
