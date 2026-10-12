"""Answers of the analytical shapes (core2/resolve/shapes.py): the figures worked out from the query's rows, and
what each answer says it is.

A running or moving figure is worked out here, not in SQL: over a gap-free run of periods (a month with no rows
adds 0 to a total, and is left out of an average of a level), from periods read before the first one shown, then
cut to the periods asked for. Every analytical answer states its definition ("Median of each customer's net amount
in Q1 2026, over the 1,204 customers with orders"), so a reader never has to guess what was averaged over what.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from core2.bootstrap.names import plural
from core2.compile.compiler import Compiled
from core2.model.schema import AggExpr
from core2.compile.shapes import LATE, LISTED, MEMBERS, REACHED, TOTAL
from core2.resolve.resolver import Logical
from core2.resolve.shapes import (AbsentShape, ActiveShape, CohortShape, FunnelShape, GapShape, LifecycleShape,
                                  OnTimeShape, ParetoShape, PerShape, RollingShape, TargetInfo, TogetherShape,
                                  WithinShape, active_definition, cohort_definition, condition_words,
                                  funnel_definition, gap_definition, lifecycle_definition, on_time_definition,
                                  pareto_definition, together_definition)


def _day(value: Any) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value)[:10]) if value is not None else None
    except ValueError:
        return None


def _float(value: Any) -> float | None:
    try:
        return None if value is None or isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        return None


def rows_for(logical: Logical, compiled: Compiled, columns: list[str], rows: list[tuple]) -> tuple[list[str],
                                                                                                   list[tuple]]:
    """The rows an answer shows: for a running or moving figure, worked out over the rows read and cut to the
    periods asked for; for every other shape, as the query returned them."""
    shape = logical.shape
    if isinstance(shape, RollingShape):
        return _rolled(shape, compiled, columns, rows, logical.fiscal_start or 1)
    return columns, rows


def _year(day: dt.date, reset: str | None, fiscal_start: int) -> int | None:
    if reset == "year":
        return day.year
    if reset == "fiscal_year":
        return day.year if day.month >= fiscal_start else day.year - 1
    return None


def _rolled(shape: RollingShape, compiled: Compiled, columns: list[str], rows: list[tuple],
            fiscal: int) -> tuple[list[str], list[tuple]]:
    names = [n.casefold() for n in columns]
    at = {c.name: names.index(c.name.casefold()) for c in compiled.columns if c.name.casefold() in names}
    period = next((c for c in compiled.columns if c.role == "period" and c.name in at), None)
    if period is None:
        return columns, rows
    keys = [c.name for c in compiled.columns if c.role in ("attribute", "member_code", "time", "unit")
            and c.name in at]
    series: dict[tuple, dict[dt.date, tuple]] = {}
    for row in rows:
        day = _day(row[at[period.name]])
        if day is not None:
            series.setdefault(tuple(row[at[k]] for k in keys), {})[day] = row
    days = sorted(set(shape.expected) | {d for found in series.values() for d in found})
    shown = [d for d in days if (shape.shown.start is None or d >= shape.shown.start)
             and (shape.shown.end is None or d < shape.shown.end)]
    out_columns = [*columns, *shape.columns.values()]
    out: list[tuple] = []
    for key, found in series.items():
        worked: dict[str, list[float | None]] = {}
        for base, name in shape.columns.items():
            if base not in at:
                continue
            values = [_float(found[d][at[base]]) if d in found else None for d in days]
            if shape.adds.get(base):
                values = [0.0 if v is None else v for v in values]
            worked[base] = _work(shape, values, days, fiscal)
        for d in shown:
            i = days.index(d)
            if d in found:
                row = list(found[d])
            else:
                row = [None] * len(columns)
                row[at[period.name]] = d
                for k, value in zip(keys, key):
                    row[at[k]] = value
                for base in shape.columns:
                    if base in at and shape.adds.get(base):
                        row[at[base]] = 0
            out.append((*row, *[worked[base][i] if base in worked else None for base in shape.columns]))
    out.sort(key=lambda r: (_day(r[at[period.name]]) or dt.date.min,
                            *[str(r[at[k]]) if r[at[k]] is not None else "" for k in keys]))
    return out_columns, out


def _work(shape: RollingShape, values: list[float | None], days: list[dt.date], fiscal: int) -> list[float | None]:
    out: list[float | None] = []
    if shape.kind == "running":
        total, year = 0.0, None
        for value, day in zip(values, days):
            this = _year(day, shape.reset, fiscal)
            if this != year:
                total, year = 0.0, this
            total += value or 0.0
            out.append(round(total, 6))
        return out
    n = shape.periods
    for i in range(len(values)):
        if i < n - 1:
            out.append(None)            # fewer periods than it covers before it: no figure yet
            continue
        window = [v for v in values[i - n + 1:i + 1] if v is not None]
        if not window:
            out.append(None)
        elif shape.kind == "moving_sum":
            out.append(round(sum(window), 6))
        else:
            out.append(round(sum(window) / len(window), 6))
    return out


# ── what the answer is ─────────────────────────────────────────────────────


def _hidden(columns: list[str], rows: list[tuple], name: str) -> float | None:
    names = [n.casefold() for n in columns]
    if name.casefold() not in names or not rows:
        return None
    return _float(rows[0][names.index(name.casefold())])


def explain(payload: dict[str, Any], logical: Logical, columns: list[str], rows: list[tuple]) -> None:
    """The answer's definition, and its sentence where the plain one would not say what it is."""
    from core2.answer.builder import span_words

    shape = logical.shape
    span = span_words(logical.window) if (logical.window.start or logical.window.end) else ""
    answer = payload.setdefault("answer", {})
    definition = ""
    if isinstance(shape, PerShape):
        definition = per_definition(shape, span, _hidden(columns, rows, MEMBERS), _hidden(columns, rows, LISTED),
                                    grouped=len(rows) > 1)
    elif isinstance(shape, RollingShape):
        definition = next((n for n in logical.notes if n.startswith("Each ")), "")
    elif isinstance(shape, WithinShape):
        said = within_headline(payload, logical, shape)
        if said:
            answer["headline"] = said
        top = f" the first {shape.top} of each are shown;" if shape.top else ""
        definition = f"Ranked within each group by the figure shown;{top} ties are put in name order."
    elif isinstance(shape, LifecycleShape):
        definition = lifecycle_definition(shape)
    elif isinstance(shape, CohortShape):
        definition = cohort_definition(shape, span)
    elif isinstance(shape, ActiveShape):
        definition = active_definition(shape)
    elif isinstance(shape, OnTimeShape):
        one = len(rows) == 1 and not shape.ageing
        definition = on_time_definition(shape, span, _hidden(columns, rows, MEMBERS) if one else None,
                                        _hidden(columns, rows, LATE) if one else None)
    elif isinstance(shape, GapShape):
        grouped = len(rows) > 1
        definition = gap_definition(shape, span, None if grouped else _hidden(columns, rows, MEMBERS),
                                    None if grouped else _hidden(columns, rows, REACHED))
    elif isinstance(shape, ParetoShape):
        said = pareto_headline(payload, shape, span, _hidden(columns, rows, TOTAL), _hidden(columns, rows, LISTED))
        if said:
            answer["headline"] = said
        definition = pareto_definition(shape, span)
    elif isinstance(shape, FunnelShape):
        said = funnel_headline(payload, shape, span)
        if said:
            answer["headline"] = said
        definition = funnel_definition(shape, span)
    elif isinstance(shape, TogetherShape):
        said = together_headline(payload, shape, span)
        if said:
            answer["headline"] = said
        definition = together_definition(shape, span)
    elif isinstance(shape, AbsentShape):
        total = _hidden(columns, rows, TOTAL)
        things = plural(shape.member)
        when = span or "ever"
        count = int(total) if total is not None else 0
        activity = _noun(shape.activity_label)
        # "Bought in Q1 but not in Q2": who had some then is who the list is of.
        had = f" with {activity} {span_words(shape.had_in)}" if shape.had_in is not None else ""
        if rows:
            answer["headline"] = (f"{count:,} {things if count != 1 else shape.member}{had} had "
                                  f"{'none' if had else f'no {activity}'} {when}" +
                                  (f"; the first {len(rows):,} are listed." if count > len(rows) else "."))
        else:
            answer["headline"] = (f"Every {shape.member}{had} had {activity} {when} too." if had else
                                  f"Every {shape.member} listed had {activity} {when}.")
        definition = next((n for n in logical.notes if n.startswith("Listed:")), "")
    if isinstance(logical.target, TargetInfo):
        said = target_headline(payload, logical, logical.target, span)
        if said:
            answer["headline"] = said
        definition = target_definition(logical.target)
    if any(g.bands for g in logical.groups) and not isinstance(shape, (ParetoShape, FunnelShape)):
        said = ranges_headline(payload, logical)
        if said:
            answer["headline"] = said
        if not definition:
            band = next(g for g in logical.groups if g.bands)
            definition = (f"Each row is put in the range of its own {band.label[:1].lower() + band.label[1:]}; a "
                          "range holds its lower edge and stops before the next.")
    if definition:
        payload["definition"] = definition
        answer["definition"] = definition


def _with(shape: PerShape) -> str:
    """What a member counted has: "with orders", "with a net amount", "with both dates"."""
    figure = shape.figure.lower()
    if figure.startswith("number of "):
        return f"with {figure[len('number of '):]}"
    if figure.startswith("days from "):
        return "with both dates"
    return f"with {article_of(figure)} {figure}"


def article_of(words: str) -> str:
    return "an" if words[:1] in "aeiou" else "a"


def _noun(label: str) -> str:
    """What a count counts: "Number of orders" -> "orders", "Customers" -> "customers"."""
    words = label.lower()
    for lead in ("number of ", "count of ", "no. of "):
        if words.startswith(lead):
            return words[len(lead):]
    return words


def ranges_headline(payload: dict[str, Any], logical: Logical) -> str:
    """ "Orders in Q1 2026 by days from order to delivery: the most, 567 (53%), in 3–7 days; 69 have no value." """
    from core2.answer.builder import fmt, scope_words
    from core2.resolve.resolver import adds_up

    if len(logical.groups) != 1 or not logical.groups[0].bands:
        return ""
    g = logical.groups[0]
    m = next((o for o in logical.measures if not o.hidden), None)
    counted = m is not None and isinstance(m.expr, AggExpr) and m.expr.agg in ("count", "count_distinct")
    if m is None or not (adds_up(m) or counted):
        return ""
    records = payload.get("export_rows") or []
    unknown = {None, "", "Unknown"}
    known = [r for r in records if r.get(g.name) not in unknown and _float(r.get(m.name)) is not None]
    if not known:
        return ""
    total = sum(_float(r.get(m.name)) or 0.0 for r in records)
    best = max(known, key=lambda r: _float(r.get(m.name)) or 0.0)
    value = _float(best.get(m.name)) or 0.0
    what = _noun(m.label) if counted else m.label
    before, span = scope_words(logical)
    lead = " ".join(w for w in (what[:1].upper() + what[1:], before, span) if w)
    share = f" ({value / total:.0%})" if total > 0 and adds_up(m) else ""   # an order in two ranges: no share
    said = f"{lead} by {g.label[:1].lower() + g.label[1:]}: the most, {fmt(value, m.format)}{share}, in {best[g.name]}"
    missing = sum(_float(r.get(m.name)) or 0.0 for r in records if r.get(g.name) in unknown)
    if missing and counted:
        said += f"; {fmt(missing, m.format)} {what} have no value for it"
    return said + "."


def within_headline(payload: dict[str, Any], logical: Logical, shape: WithinShape) -> str:
    """ "Net amount in Q1 2026, the first 3 products in each store: Airport Store — Potatoes Large ($2,399.11),
    …; Downtown Store — …" (the first of each, for the first three groups)."""
    from core2.answer.builder import fmt, scope_words

    records = payload.get("export_rows") or []
    m = next((o for o in logical.measures if o.name == shape.by), None)
    member = next((g for g in logical.groups if g.name == shape.member), None)
    groups = [g for g in logical.groups if g.name in shape.partition]
    if not records or m is None or member is None or not groups:
        return ""
    firsts: dict[tuple, dict] = {}
    for r in records:
        key = tuple(r.get(g.name) for g in groups)
        if key not in firsts or (r.get(shape.rank) or 0) < (firsts[key].get(shape.rank) or 0):
            firsts[key] = r
    things = plural(member.label.lower().removesuffix(" name"))
    where = " ".join(str(g.label).lower().removesuffix(" name") for g in groups)
    before, span = scope_words(logical)
    lead = " ".join(w for w in (m.label, before, span) if w)
    first = f"the first {shape.top} {things} in each {where}" if shape.top else f"{things} ranked in each {where}"
    named = [f"{' / '.join(str(k) for k in key)}: {r.get(member.name)} ({fmt(r.get(m.name), m.format)})"
             for key, r in list(firsts.items())[:3]]
    more = f"; and {len(firsts) - 3:,} more" if len(firsts) > 3 else ""
    return f"{lead}, {first}. Leading each: {'; '.join(named)}{more}."


def per_definition(shape: PerShape, span: str, members: float | None, listed: float | None, *,
                   grouped: bool) -> str:
    """ "Each customer's net amount in Q1 2026, then the median over the 1,204 customers with a net amount." """
    things = plural(shape.member)
    each = f"the {shape.figure.lower()} of each {shape.member}" if shape.row_level else \
        f"each {shape.member}'s {shape.figure.lower()}"
    when = f" {span}" if span else ""
    if shape.band is not None:
        return (f"Each range counts the {things} whose {shape.figure.lower()}{when} falls in it; a range holds "
                "its lower edge and stops before the next.")
    if listed is not None:
        over = f"all {int(listed):,} {things} listed (one with none counted as 0)"
    elif members is not None and not grouped:
        over = f"the {int(members):,} {things} {_with(shape)}"
    else:
        over = f"the {things} {_with(shape)}"
    if shape.summary == "median":
        how = "then the middle value of"
    elif shape.summary == "percentile":
        how = f"then the value {shape.p or 50:g}% of them are at or below, of"
    elif shape.summary == "avg":
        how = "then their average over"
    elif shape.summary in ("min", "max"):
        how = f"then the {'lowest' if shape.summary == 'min' else 'highest'} of"
    else:
        condition = condition_words(shape.op or "", shape.values, shape.figure)
        whole = "counted among" if shape.summary == "count" else "as a share of"
        return f"{each[:1].upper()}{each[1:]}{when}; {things} with {condition}, {whole} {over}."
    return f"{each[:1].upper()}{each[1:]}{when}, {how} {over}."


# ── R4: who makes most, funnels, bought together, against target ───────────


def _records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(payload.get("export_rows") or (payload.get("data") or {}).get("rows") or [])


def _pct_words(value: float | None) -> str:
    return "" if value is None else f"{value:.0f}%" if abs(value) >= 10 else f"{value:.1f}%"


def pareto_headline(payload: dict[str, Any], shape: ParetoShape, span: str, total: float | None,
                    first_cut: float | None = None) -> str:
    """ "18 of 103 customers made the first 80% of net amount in 2025." / "A: 41 of 111 products (37%) made 80%
    of net amount in 2025; B: 31, the next 15%; C: 39, the last 5%." """
    records = _records(payload)
    if not records:
        return ""
    things, word = plural(shape.member), shape.figure.lower()
    when = f" {span}" if span else ""
    of = f" of {int(total):,}" if total else ""
    cols = shape.columns
    if shape.mode == "pareto":
        first = records[0]
        lead = " ".join(str(first.get(n) or "") for n in shape.members[:1]).strip()
        share = _pct_words(_float(first.get(cols["share"])))
        made = int(first_cut) if first_cut is not None else len(records)
        return (f"{made:,}{of} {things} made the first {shape.cuts[0]:g}% of {word}{when}"
                + (f"; the largest, {lead}, made {share}." if lead and share else "."))
    if shape.listed:
        if shape.mode != "abc" or total is None or len(records) < int(total):
            return ""
        first = sum(1 for r in records if r.get(cols["class"]) == "A")
        return f"{first:,}{of} {things} are class A: together they made the first {shape.cuts[0]:g}% of {word}{when}."
    counts = {str(r.get(cols["class"])): (_float(r.get(cols["members"])) or 0, _float(r.get(cols["share"])))
              for r in records}
    if shape.mode == "ntile":
        kind = "decile" if shape.groups == 10 else "quartile" if shape.groups == 4 else "group"
        top, bottom = counts.get("1"), counts.get(str(shape.groups))
        if top is None:
            return ""
        said = f"The top {kind} of {things} ({int(top[0]):,}{of}) made {_pct_words(top[1])} of {word}{when}"
        if bottom is not None:
            said += f"; the bottom {kind}, {_pct_words(bottom[1])}"
        return said + "."
    parts = []
    for i, name in enumerate(sorted(counts)):
        n, share = counts[name]
        if i == 0:
            parts.append(f"{name}: {int(n):,}{of} {things}"
                         + (f" ({n / total:.0%})" if total else "") + f" made {_pct_words(share)} of {word}{when}")
        else:
            parts.append(f"{name}: {int(n):,}, {_pct_words(share)}")
    return "; ".join(parts) + "."


def funnel_headline(payload: dict[str, Any], shape: FunnelShape, span: str) -> str:
    """ "Of 10,256 order lines with an order date in 2025, 9,658 (94%) reached delivered date; the most were lost
    before ship date (6%)." """
    records = _records(payload)
    cols = shape.columns
    if len(records) < 2:
        return ""
    first = _float(records[0].get(cols["reached"])) or 0
    last = records[-1]
    reached = _float(last.get(cols["reached"])) or 0
    stage = shape.labels[0].lower()
    when = f" with {article_of(stage)} {stage} {span}" if span else ""
    said = (f"Of {int(first):,} {shape.things}{when}, {int(reached):,} ({_pct_words(_float(last.get(cols['of_first'])))}"
            f") reached {str(last.get(cols['stage'])).lower()}")
    drops = [(100 - (_float(r.get(cols["of_previous"])) or 100), str(r.get(cols["stage"]))) for r in records[1:]]
    worst = max(drops) if drops else None
    if worst is not None and worst[0] > 0:
        said += f"; the most were lost before {worst[1].lower()} ({_pct_words(worst[0])})"
    return said + "."


def together_headline(payload: dict[str, Any], shape: TogetherShape, span: str) -> str:
    """ "Bought together most often in 2025: Bagels Large and Dish Soap Large, on 8 orders (3.3 times as often as
    chance would have it)." """
    records = _records(payload)
    if not records:
        return f"No two {plural(shape.item_word)} were on the same {shape.basket_word}" + (f" {span}." if span else ".")
    cols = shape.columns
    top = records[0]
    n = int(_float(top.get(cols["together"])) or 0)
    lift = _float(top.get(cols["lift"]))
    when = f" {span}" if span else ""
    baskets = plural(shape.basket_word) if n != 1 else shape.basket_word
    chance = f" ({lift:.1f} times as often as chance would have it)" if lift and lift > 1 else ""
    if shape.anchor:
        share = _pct_words(_float(top.get(cols["confidence"])))
        return (f"Bought most often with {top.get(cols['item'])}{when}: {top.get(cols['with_item'])}, on {n:,} "
                f"{baskets} ({share} of its {plural(shape.basket_word)}).")
    return f"Bought together most often{when}: {top.get(cols['item'])} and {top.get(cols['with_item'])}, on {n:,} {baskets}{chance}."


def target_definition(info: TargetInfo) -> str:
    good = "100% or less is within it" if info.lower_is_better else "100% or more is reached"
    return (f"% of target: {info.actual.lower()} as a share of {info.target.lower()} ({good}); difference: "
            f"{info.actual.lower()} minus {info.target.lower()}.")


def target_headline(payload: dict[str, Any], logical: Logical, info: TargetInfo, span: str) -> str:
    """ "8 of 12 stores reached their target amount in 2025; Eastview Store is furthest above it (245%),
    Downtown Store furthest below (93%)." """
    from core2.answer.builder import fmt

    records = _records(payload)
    share = next((m.name for m in logical.measures if m.label == info.percent), None)
    actual = next((m for m in logical.measures if m.label == info.actual), None)
    target = next((m for m in logical.measures if m.label == info.target), None)
    if share is None or not records:
        return ""
    when = f" {span}" if span else ""
    met = (lambda v: v <= 100) if info.lower_is_better else (lambda v: v >= 100)
    reached = "stayed within" if info.lower_is_better else "reached"
    # A period or member with no actual yet (targets set ahead) is not one that missed it.
    known = [r for r in records if _float(r.get(share)) is not None
             and (actual is None or _float(r.get(actual.name)) is not None)]
    if len(records) == 1 and not logical.groups:
        r = records[0]
        value = _float(r.get(share))
        if value is None or actual is None or target is None:
            return ""
        return (f"{info.actual} was {_pct_words(value)} of {info.target.lower()}{when}: "
                f"{fmt(r.get(actual.name), actual.format)} against {fmt(r.get(target.name), target.format)}.")
    group = next((g for g in logical.groups if g.kind in ("attribute", "period")), None)
    if group is not None and group.kind == "period":
        # A period the data covers only in part is not compared as a whole one (the answer says which).
        part = {d.isoformat() for d in logical.partial}
        known = [r for r in known if str(r.get(group.name))[:10] not in part]
    if group is None or not known:
        return ""
    name = lambda r: str(r.get(group.name))  # noqa: E731
    hits = sum(1 for r in known if met(_float(r.get(share)) or 0))
    best = max(known, key=lambda r: _float(r.get(share)) or 0)
    worst = min(known, key=lambda r: _float(r.get(share)) or 0)
    if info.lower_is_better:
        best, worst = worst, best
    if group.kind == "period":
        periods = plural((group.grain or "period").replace("fiscal_", "fiscal "))
        return (f"{info.actual} {reached} {info.target.lower()} in {hits} of {len(known)} {periods}{when}; the best "
                f"was {_period_name(best.get(group.name), group.grain)} ({_pct_words(_float(best.get(share)))}), "
                f"the lowest {_period_name(worst.get(group.name), group.grain)} "
                f"({_pct_words(_float(worst.get(share)))}).")
    words = group.label.lower().split()
    if len(words) > 1 and words[-1] in ("name", "code", "id", "number"):
        words = words[:-1]                          # "Store name": stores
    things = plural(" ".join(words))
    return (f"{hits} of {len(known)} {things} {reached} their {info.target.lower()}{when}; {name(best)} is furthest "
            f"{'under' if info.lower_is_better else 'above'} it ({_pct_words(_float(best.get(share)))}), "
            f"{name(worst)} furthest {'over' if info.lower_is_better else 'below'} "
            f"({_pct_words(_float(worst.get(share)))}).")


def _period_name(value: Any, grain: str | None) -> str:
    day = _day(value)
    if day is None:
        return str(value)
    if grain in ("year", "fiscal_year"):
        return str(day.year)
    if grain in ("quarter", "fiscal_quarter"):
        return f"Q{(day.month - 1) // 3 + 1} {day.year}"
    if grain == "day":
        return f"{day.day} {day:%b %Y}"
    return f"{day:%b %Y}"
