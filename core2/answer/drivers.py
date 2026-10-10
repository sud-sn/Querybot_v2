"""Why a measure changed: the change between two periods, and the groupings that explain it.

A "why" question ("why did sales drop in March?", "what drove the rise this
year?") compares the period asked about with the one before it, or with the one
the question names. The total change comes first; then the change is broken
down by the groupings a reader would look at first: the measure table's direct
dimensions (customer, product, store), their parents (category, region) and
small categories (segment, status). Each grouping is one governed comparison
query through the ordinary resolver and compiler, so joins, dates, snapshots,
filters and the reader's access are the question path's own.

A grouping explains the change when a few of its members carry most of it and
their share of the measure moved (explanatory power and surprise, after
Adtributor, kept simple): a fall spread evenly over every member points at no
member. A ratio does not decompose into its members' changes; for one, the
members that moved most are shown and the answer says the parts do not add up.

A "why" about a series (a follow-up on a chart) compares its last stretch with
the one before: the last 7 days against the 7 before for a daily series, the
last 4 weeks against the 4 before for a weekly one, and the last complete period
against the one before for a monthly or longer one.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import time
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any

from core2.answer.builder import (answer_badges, bar_chart, conditions_tail, display_value, fmt, frame, scoped_label,
                                  span_words, versus)
from core2.compile.compiler import CompileError, Compiled, compile_query
from core2.model.schema import SemanticModel
from core2.plan.ir import Compare, Filter, Plan, TimeSpec, Window
from core2.resolve import paths as P
from core2.resolve.resolver import Context, Logical, ResolveError, measure_dates, resolve
from core2.resolve.time import DAY, Range, add_units, resolve_window, unit_start
from core2.resolve.time import label as period_label
from core2.warehouse.runner import QueryFailed, QueryResult, Warehouse

log = logging.getLogger("querybot.core2")

MAX_GROUPINGS = 5           # comparison queries beside the total's
EXPLAINED = 0.67            # the share of a change a grouping's leading members should carry
SHOWN = 10                  # members listed per grouping
BUDGET_SECONDS = 60.0       # groupings are not started after this


@dataclass
class Mover:
    member: str
    prior: float
    current: float

    @property
    def change(self) -> float:
        return self.current - self.prior


@dataclass
class Grouping:
    slug: str
    label: str
    movers: list[Mover]
    leaders: list[Mover] = field(default_factory=list)
    explained: float = 0.0       # the share of the change the leaders carry, at most 1
    carried: float = 0.0         # the same, uncapped: above 1 when the rest moved the other way
    surprise: float = 0.0
    shift: float = 0.0           # how far the members' shares moved (half the sum of share changes)
    truncated: bool = False

    @property
    def word(self) -> str:
        """The grouping as a reader says it: "customer", not "customer name"."""
        words = self.label.lower().split()
        while len(words) > 1 and words[-1] in ("name", "description", "desc", "label", "title"):
            words.pop()
        return " ".join(words)


def _number(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float, Decimal)):
        number = float(value)
        return number if math.isfinite(number) else 0.0
    return 0.0


def _signed(value: float, format_: str) -> str:
    return ("+" if value > 0 else "") + fmt(value, format_)


# ── which periods ──────────────────────────────────────────────────────────


def _between(rng: Range) -> Window:
    assert rng.start is not None and rng.end is not None
    return Window(kind="between", start=rng.start, end=rng.end - DAY)


def _series_windows(plan: Plan, model: SemanticModel, ctx: Context) -> tuple[Window, Compare, str] | None:
    """The last stretch of a series against the one before it."""
    grain = plan.time.grain or "month"
    unit = {"fiscal_month": "month", "fiscal_quarter": "quarter", "fiscal_year": "year"}.get(grain, grain)
    fiscal = model.settings.fiscal_year_start_month if grain.startswith("fiscal") else None
    _, first, last = measure_dates(plan, model)
    rng = resolve_window(plan.time.window, today=ctx.today, first_data=first, last_data=last,
                         fiscal_start=model.settings.fiscal_year_start_month)
    start = rng.start or first
    end = rng.end or ((last + DAY) if last else None)
    if last is not None and end is not None:
        end = min(end, last + DAY)
    if start is None or end is None or end <= start:
        return None
    if unit == "day":
        days = min(7, (end - start).days // 2)
        if days < 1:
            return None
        current = Range(end - dt.timedelta(days=days), end)
        prior = Range(end - dt.timedelta(days=2 * days), end - dt.timedelta(days=days))
        words = f"the last {days} days of the series against the {days} before"
    elif unit == "week":
        week_end = unit_start(end, "week")          # whole weeks only: a part-week is not compared
        weeks = (week_end - unit_start(start, "week")).days // 7
        n = min(4, weeks // 2)
        if n < 1:
            return None
        current = Range(week_end - dt.timedelta(weeks=n), week_end)
        prior = Range(week_end - dt.timedelta(weeks=2 * n), week_end - dt.timedelta(weeks=n))
        words = f"the last {n} week{'s' if n > 1 else ''} of the series against the {n} before"
    else:
        holding = unit_start(end - DAY, unit, fiscal_start=fiscal)
        if add_units(holding, unit, 1) > end:          # the last period is partial: the one before it
            holding = add_units(holding, unit, -1)
        if first is not None and add_units(holding, unit, -1) < unit_start(first, unit, fiscal_start=fiscal):
            return None                                 # no period before it in the data
        current = Range(holding, add_units(holding, unit, 1))
        prior = Range(add_units(holding, unit, -1), holding)
        words = f"the last complete {unit} of the series against the one before"
    return _between(current), Compare(kind="window", window=_between(prior)), f"Compared {words}."


def base_plan(plan: Plan, model: SemanticModel, ctx: Context) -> tuple[Plan, list[str]]:
    """The total, as a comparison of the two periods the question is about."""
    keep = plan.measures[:1]
    derived = [] if keep else plan.derived[:1]
    shown = [d for d in plan.durations if d.measure]
    durations = [] if keep or derived else shown[:1]      # "why did the days to invoice go up?"
    if not keep and not derived and not durations:
        raise ResolveError("unknown", "a why question needs the measure that changed",
                           sorted(m.slug for m in model.measures.values() if not m.hidden)[:12])
    notes: list[str] = []
    if len(plan.measures) + len(plan.derived) + len(shown) > 1:
        notes.append("Explained for the first measure asked about.")
    t = plan.time
    window, compare = t.window, t.compare
    if t.grain:
        series = _series_windows(plan, model, ctx)
        if series is not None:
            window, compare, note = series
            notes.append(note)
    if window.kind == "all":
        window = Window(kind="previous", unit="month")
        notes.append("No period was named: the last complete month against the one before.")
    compare = compare or Compare(kind="previous_period")
    base = plan.model_copy(update={
        "intent": "compare", "measures": keep, "derived": derived, "group_by": [], "via": {}, "sort": [],
        "durations": durations + [d for d in plan.durations if not d.measure],    # and those keeping rows
        "limit": None, "drivers": None, "forecast": None,
        "time": TimeSpec(date=t.date, grain=None, window=window, compare=compare)})
    return base, notes


# ── which groupings ────────────────────────────────────────────────────────


MIN_MATCH = 0.9     # a grouping reached through a link that matches fewer rows is not checked


_FLAG_VALUES = {"0", "1", "y", "n", "t", "f", "yes", "no", "true", "false"}


def _a_flag(column) -> bool:
    """A yes/no column (0/1, Y/N): "1 (+2,001,508) and 0 (+78,401)" explains nothing to a reader."""
    top = column.profile.top if column.profile else None
    return bool(top) and all(str(t.value).strip().lower() in _FLAG_VALUES for t in top if t.value is not None)


def candidates(model: SemanticModel, table: str, *, allowed: set[str] | None, skip: set[str]) -> list[str]:
    """Groupings worth checking, most telling first: direct dimensions, their parents, then small categories."""
    scored: list[tuple[tuple, str]] = []
    names_or_keys = {c for e in model.entities.values()
                     for c in (e.label_column, e.code_column, *e.key_columns) if c}

    def hops(owner: str) -> int | None:
        if owner == table:
            return 0
        if allowed is not None and owner not in allowed:
            return None
        try:
            path = P.best_path(model, table, owner)
        except P.Ambiguous:
            return None              # reached two ways: only when the question names the role
        if path is not None and any(j.match_rate < MIN_MATCH for j in path.joins):
            return None              # most rows would read Unknown: it explains nothing
        return len(path.joins) if path is not None else None

    for e in model.entities.values():
        label = model.columns.get(e.label_column or "")
        if e.slug in skip or e.table == table or not 2 <= (e.members or 0) <= 2000 \
                or label is not None and label.personal != "none":
            continue          # people (patients, customers by name) are never ranked as a change's drivers
        n = hops(e.table)
        if n is None or n > 2:
            continue
        scored.append(((0 if n == 1 else 1, e.members, e.slug), e.slug))
    for a in model.attributes.values():
        column = model.columns[a.column]
        members = a.members or (column.profile.distinct if column.profile else 0)
        if a.slug in skip or column.key in names_or_keys or column.hidden or column.sensitivity != "none" \
                or column.personal != "none" or not 2 <= members <= 50 or _a_flag(column):
            continue
        n = hops(column.table)
        if n is None or n > 1:
            continue
        scored.append(((2, members, a.slug), a.slug))
    return [slug for _, slug in sorted(scored)][:MAX_GROUPINGS]


# ── the numbers ────────────────────────────────────────────────────────────


def _columns(compiled: Compiled, result: QueryResult) -> dict[str, int]:
    at = {name.casefold(): i for i, name in enumerate(result.columns)}
    return {c.name: at[c.name.casefold()] for c in compiled.columns if c.name.casefold() in at}


def _totals(compiled: Compiled, result: QueryResult) -> tuple[float, float]:
    at = _columns(compiled, result)
    measure = next(c for c in compiled.columns if c.role == "measure")
    prior = next(c for c in compiled.columns if c.role == "prior")
    if not result.rows:
        return 0.0, 0.0
    row = result.rows[0]
    return _number(row[at[measure.name]]), _number(row[at[prior.name]])


def _movers(compiled: Compiled, result: QueryResult, logical: Logical) -> list[Mover]:
    at = _columns(compiled, result)
    group = next(c for c in compiled.columns if c.role in ("attribute", "time"))
    code = next((c for c in compiled.columns if c.role == "member_code"), None)
    measure = next(c for c in compiled.columns if c.role == "measure")
    prior = next(c for c in compiled.columns if c.role == "prior")
    out = []
    for row in result.rows:
        name = str(display_value(group, row[at[group.name]], logical))
        if code is not None and row[at[code.name]] is not None:
            name = f"{name} ({display_value(code, row[at[code.name]], logical)})"
        out.append(Mover(name, _number(row[at[prior.name]]), _number(row[at[measure.name]])))
    return out


def _kl(p: float, m: float) -> float:
    return p * math.log(p / m) if p > 0 and m > 0 else 0.0


def score(g: Grouping, change: float, additive: bool) -> None:
    """Which members carry the change (in its direction), how much of it, and how surprising their move is."""
    direction = 1.0 if change >= 0 else -1.0
    same = [m for m in g.movers if m.change * direction > 0]
    shares_hold = additive and change != 0 and all(m.prior >= 0 and m.current >= 0 for m in g.movers)
    before = sum(m.prior for m in g.movers)
    after = sum(m.current for m in g.movers)
    if not shares_hold or before <= 0 or after <= 0:
        g.leaders = sorted(same, key=lambda m: (-abs(m.change), m.member))[:3]
        g.carried = sum(abs(m.change) for m in g.leaders) / abs(change) if change else 0.0
        g.explained = min(1.0, g.carried)
        g.surprise = g.explained
        g.shift = 1.0            # shares mean nothing here: the members' own changes decide
        return
    g.shift = 0.5 * sum(abs(m.current / after - m.prior / before) for m in g.movers)

    def surprise(m: Mover) -> float:
        p, q = m.prior / before, m.current / after
        mid = (p + q) / 2
        return 0.5 * (_kl(p, mid) + _kl(q, mid))

    # The leaders are the members carrying the most of the change; whether their move is telling
    # (their share moved, not just their size) is the grouping's surprise and shift.
    leaders: list[Mover] = []
    carried = 0.0
    for m in sorted(same, key=lambda m: (-abs(m.change), m.member)):
        leaders.append(m)
        carried += m.change / change
        if carried >= EXPLAINED or len(leaders) == 3:
            break
    g.leaders = leaders
    g.carried = carried
    g.explained = min(1.0, carried)
    g.surprise = sum(surprise(m) for m in leaders)


SHIFTED = 0.02               # a mix that moved less than 2 points changed evenly: no member drove it


def telling(g: Grouping) -> bool:
    """Whether a grouping explains the change: a few members carry most of it, and the mix moved."""
    return bool(g.leaders) and g.explained >= 0.5 and g.shift >= SHIFTED


def rank(groupings: list[Grouping]) -> list[Grouping]:
    """The most telling first: fewest members needed, then the most of the change carried, then the mix's move."""
    return sorted(groupings, key=lambda g: (not telling(g), len(g.leaders) or 9, -g.explained, -g.shift, g.label))


# ── the answer ─────────────────────────────────────────────────────────────


def _one_unit(base: Plan, model: SemanticModel, ctx: Context) -> tuple[Plan, str | None, str | None]:
    """The comparison in its main unit, when the quantity is kept in several (EA, FT, BX): added up, they
    are no amount. The base plan narrowed to that unit, the unit's grouping, and the unit."""
    try:
        probe = resolve(base.model_copy(update={"intent": "breakdown"}), model, replace(ctx, split_units=True))
    except (ResolveError, ValueError):
        return base, None, None
    if not probe.unit_group or not probe.unit_order:
        return base, None, None
    slug = next((g.attribute for g in probe.groups if g.name == probe.unit_group and g.attribute), None)
    if slug is None:
        return base, None, None
    main = probe.unit_order[0]
    return base.model_copy(update={"filters": [*base.filters, Filter(field=slug, op="eq", values=[main])]}), slug, main


def _unit_unsaid(logical: Logical, unit: str | None) -> Logical:
    """The unit counted in is said with every amount ("422 EA"), not again as a condition of the measure."""
    if unit:
        logical.conditions = [c for c in logical.conditions
                              if not (c.kind == "member" and [str(v) for v in c.values] == [unit])]
    return logical


def _whole_unit(rng: Range) -> str | None:
    """The calendar unit a window is exactly (a month, a quarter, a year), if it is one."""
    if rng.start is None or rng.end is None or rng.start.day != 1:
        return None
    for unit in ("month", "quarter", "year"):
        if unit_start(rng.start, unit) == rng.start and add_units(rng.start, unit, 1) == rng.end:
            return unit
    return None


LOOK_BACK = {"month": 36, "quarter": 12, "year": 5}


def _last_with_data(base: Plan, logical: Logical, model: SemanticModel, ctx: Context,
                    warehouse: Warehouse) -> tuple[Window, str, str] | None:
    """When the period before the one asked about has no data (a month never loaded, a snapshot not
    taken), the last period before it that has: its window, its name and the unit. None when the
    period before has data, or none before it does."""
    rng, before = logical.window, logical.compare
    unit = _whole_unit(rng)
    if unit is None or before is None or before.start is None or rng.start is None:
        return None
    look = add_units(rng.start, unit, -LOOK_BACK[unit])
    series = base.model_copy(update={"intent": "trend", "time": TimeSpec(
        date=base.time.date, grain=unit, window=_between(Range(look, rng.start)), compare=None)})
    trend = resolve(series, model, ctx)
    compiled = compile_query(trend, model, warehouse.dialect)
    result = warehouse.query(compiled.sql, max_rows=compiled.row_cap)
    at = _columns(compiled, result)
    period = next(c for c in compiled.columns if c.role == "period")
    measure = next(c for c in compiled.columns if c.role == "measure")
    from core2.answer.builder import _day

    held = sorted(d for row in result.rows for d in [_day(row[at[period.name]])]
                  if d is not None and row[at[measure.name]] is not None)
    if not held or held[-1] >= before.start:
        return None
    last = held[-1]
    return _between(Range(last, add_units(last, unit, 1))), period_label(last, unit), unit


def answer_drivers(question: str, plan: Plan, *, model: SemanticModel, warehouse: Warehouse, ctx: Context,
                   data_source: str = "", question_id: str = "", model_version: int = 0,
                   started: float | None = None) -> dict[str, Any]:
    started = started if started is not None else time.perf_counter()
    base, notes = base_plan(plan, model, ctx)
    base, unit_slug, unit = _one_unit(base, model, ctx)
    if unit:
        notes.append(f"Counted in {unit}, the unit of most rows; quantities in other units are not added in.")
    total_logical = _unit_unsaid(resolve(base, model, ctx), unit)
    total_compiled = compile_query(total_logical, model, warehouse.dialect)
    result = warehouse.query(total_compiled.sql, max_rows=total_compiled.row_cap)
    current, prior = _totals(total_compiled, result)
    sqls = [("the total", total_compiled.sql)]
    if not prior and (base.time.compare is None or base.time.compare.kind == "previous_period"):
        try:
            found = _last_with_data(base, total_logical, model, ctx, warehouse)
        except (ResolveError, CompileError, ValueError, QueryFailed, StopIteration) as exc:
            log.warning("core2 could not look for the last period with data: %s", exc)
            found = None
        if found is not None:
            window, name, unit_word = found
            assert total_logical.compare is not None and total_logical.compare.start is not None
            empty = period_label(total_logical.compare.start, unit_word)
            base = base.model_copy(update={"time": base.time.model_copy(
                update={"compare": Compare(kind="window", window=window)})})
            total_logical = _unit_unsaid(resolve(base, model, ctx), unit)
            total_compiled = compile_query(total_logical, model, warehouse.dialect)
            result = warehouse.query(total_compiled.sql, max_rows=total_compiled.row_cap)
            current, prior = _totals(total_compiled, result)
            sqls = [("the total", total_compiled.sql)]
            notes.append(f"{empty} has no data: compared with {name}, the last {unit_word} before it that has.")
    change = current - prior
    m = total_logical.measures[0]
    if m.measure is not None:
        additive = m.measure.additivity != "non_additive" and m.format != "percent"
    else:   # a ratio or an average of days is explained by mix and rate, a difference or total by parts
        additive = (bool(base.derived) and base.derived[0].op != "ratio") or any(
            d.measure and d.agg == "sum" for d in base.durations)
    rows_read = len(result.rows)

    asked = list(dict.fromkeys(plan.drivers.dimensions)) if plan.drivers and plan.drivers.dimensions else []
    skip = {f.field for f in plan.filters} | ({unit_slug} if unit_slug else set())
    slugs = asked[:MAX_GROUPINGS] or candidates(model, total_logical.parts[0].table, allowed=ctx.allowed_tables,
                                               skip=skip)
    groupings: list[Grouping] = []
    skipped: list[str] = []
    why_not: list[str] = []
    for slug in slugs:
        if time.perf_counter() - started > BUDGET_SECONDS:
            skipped.append(slug)
            continue
        sub = base.model_copy(update={"group_by": [slug], "via": {k: v for k, v in plan.via.items() if k == slug}})
        try:
            logical = resolve(sub, model, ctx)
            compiled = compile_query(logical, model, warehouse.dialect)
            res = warehouse.query(compiled.sql, max_rows=compiled.row_cap)
        except ResolveError as exc:
            if asked:
                raise                    # the reader named this grouping: say what is wrong with it
            skipped.append(slug)
            why_not.append(f"{_grouping_word(model, slug)} ({str(exc).rstrip('.')})")
            continue
        except (CompileError, ValueError, QueryFailed) as exc:
            if asked:
                raise
            skipped.append(slug)
            why_not.append(f"{_grouping_word(model, slug)} ({str(exc).splitlines()[0][:120].rstrip('.')})")
            continue
        label = next((g.label for g in logical.groups if g.kind in ("attribute", "time")), slug)
        grouping = Grouping(slug, label, _movers(compiled, res, logical), truncated=res.truncated)
        score(grouping, change, additive)
        groupings.append(grouping)
        sqls.append((f"by {label.lower()}", compiled.sql))
        rows_read += len(res.rows)

    ranked = rank(groupings)
    reported = [g for g in ranked if telling(g)][:2]
    fmt_ = m.format
    now, before = span_words(total_logical.window), span_words(total_logical.compare or Range(None, None))
    pct = change / prior if prior else None
    # The conditions the question applied are named with the measure: "Gross profit for profit centre X rose".
    headline = _headline(scoped_label(total_logical, m.label), fmt_, current, prior, change, pct, now, before,
                         reported, ranked, additive, unit=unit)
    tail = conditions_tail(total_logical)

    notes = [*notes, *total_logical.notes]
    if groupings:
        notes.append("Groupings checked: " + ", ".join(g.word for g in groupings) + ".")
    if skipped:
        notes.append(f"{len(skipped)} other grouping{'s' if len(skipped) > 1 else ''} could not be checked"
                     + (": " + "; ".join(why_not[:2]) if why_not else "") + ".")
    if not slugs:
        notes.append("No grouping of this measure's rows could be checked for what moved it.")
    if any(g.truncated for g in groupings):
        notes.append("Some groupings have more members than were read; their smallest members are not shown.")

    headers, labels, formats, records = _table(m.label, fmt_, now, before, change, prior, current, ranked, additive)
    chart = None
    if ranked and ranked[0].movers:
        top = ranked[0]
        shown = sorted(top.movers, key=lambda x: (-abs(x.change), x.member))[:SHOWN]
        chart = bar_chart(f"Change in {m.label.lower()} by {top.word}", "member", top.label,
                          [("change", "Change", formats["change"])],
                          [{"member": x.member, "change": x.change} for x in shown], intent="drivers")
    short_value = fmt(current, fmt_, unit=unit)
    comparison = ""
    if prior or current:
        pct_text = f" ({pct * 100:+.1f}%)" if pct is not None else ""
        comparison = (f"{'up' if change >= 0 else 'down'} {fmt(abs(change), fmt_, unit=unit)}{pct_text} "
                      f"{versus(total_logical.compare or Range(None, None))}")
    suggestions: list[str] = []
    if reported:
        top = reported[0]
        suggestions.append(f"{m.label} by {top.word} by month{tail}")
        leader = next((x.member for x in top.leaders if x.member != "Unknown"), None)
        if leader:
            suggestions.append(f"Monthly {m.label.lower()} for {leader}{tail}")
    payload = frame(question, headline=headline, short_value=short_value, comparison=comparison,
                    caveats=[n for n in notes if n.startswith("The data runs to")][:1], chart=chart, kpi=None,
                    badges=answer_badges(total_logical),
                    suggestions=suggestions[:3], headers=headers, labels=labels, records=records, formats=formats,
                    sql="\n\n".join(f"-- {what}\n{sql}" for what, sql in sqls), row_count=rows_read,
                    duration_ms=(time.perf_counter() - started) * 1000, data_source=data_source,
                    question_id=question_id, notes=notes, model_version=model_version)
    payload["drivers"] = {"change": change, "groupings": [
        {"slug": g.slug, "label": g.label, "explained": round(g.explained, 4),
         "leaders": [{"member": x.member, "change": x.change} for x in g.leaders]} for g in ranked]}
    return payload


def _headline(label: str, format_: str, current: float, prior: float, change: float, pct: float | None,
              now: str, before: str, reported: list[Grouping], ranked: list[Grouping], additive: bool,
              *, unit: str | None = None) -> str:
    if not prior and not current:
        return f"There is no {label.lower()} {now} or {before.removeprefix('in ')}."
    if not change:
        lead = f"{label} did not change: {fmt(current, format_, unit=unit)} {now}, as {before}."
    else:
        moved = "rose" if change > 0 else "fell"
        pct_text = f" ({pct * 100:+.1f}%)" if pct is not None else ""
        lead = (f"{label} {moved} {fmt(abs(change), format_, unit=unit)}{pct_text}, from "
                f"{fmt(prior, format_, unit=unit)} {before} to {fmt(current, format_, unit=unit)} {now}.")
    sentences = [lead]
    what = "rise" if change > 0 else "fall"
    for g in reported:
        listed = [f"{x.member} ({_signed(x.change, format_)})" for x in g.leaders]
        names = listed[0] if len(listed) == 1 else ", ".join(listed[:-1]) + f" and {listed[-1]}"
        share = ""
        if additive and change:
            share = (f", more than the whole {what} (the rest moved the other way)" if g.carried > 1.005
                     else f", {g.explained:.0%} of the {what}")
        sentences.append(f"By {g.word}: {names}{share}.")
    if not reported and ranked and change:
        sentences.append(f"The change is spread out: no single {ranked[0].word} accounts for much of it.")
    if not additive and reported:
        sentences.append("The groups' changes do not add up to the total for a ratio; these moved most.")
    return " ".join(sentences)


def _grouping_word(model: SemanticModel, slug: str) -> str:
    if slug in model.entities:
        return Grouping(slug, model.entities[slug].business_name, []).word
    if slug in model.attributes:
        return Grouping(slug, model.attributes[slug].business_name, []).word
    return slug


def _first_upper(text: str) -> str:
    return text[:1].upper() + text[1:]


def _table(label: str, format_: str, now: str, before: str, change: float, prior: float, current: float,
           ranked: list[Grouping], additive: bool) -> tuple[list[str], dict[str, str], dict[str, str], list[dict]]:
    value_format = {"currency": "currency", "percent": "percentage"}.get(format_, "number")
    headers = ["grouping", "member", "before", "after", "change", "change_pct"] + (["share_of_change"] if additive
                                                                                   else [])
    labels = {"grouping": "Grouping", "member": "Member", "before": _first_upper(before or "before"),
              "after": _first_upper(now or "after"), "change": "Change", "change_pct": "Change %",
              "share_of_change": "Share of the change"}
    formats = {"grouping": "text", "member": "text", "before": value_format, "after": value_format,
               "change": value_format, "change_pct": "percentage", "share_of_change": "percentage"}

    def row(grouping: str, member: str, p: float, c: float) -> dict:
        out = {"grouping": grouping, "member": member, "before": round(p, 6), "after": round(c, 6),
               "change": round(c - p, 6), "change_pct": round((c - p) / p * 100, 4) if p else None}
        if additive:
            out["share_of_change"] = round((c - p) / change * 100, 4) if change else None
        return out

    records = [row("All", f"Total {label.lower()}", prior, current)]
    for g in ranked:
        for x in sorted(g.movers, key=lambda x: (-abs(x.change), x.member))[:SHOWN]:
            records.append(row(g.word[:1].upper() + g.word[1:], x.member, x.prior, x.current))
    return headers, {k: labels[k] for k in headers}, {k: formats[k] for k in headers}, records
