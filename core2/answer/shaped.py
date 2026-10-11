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
from core2.compile.shapes import LISTED, MEMBERS, TOTAL
from core2.resolve.resolver import Logical
from core2.resolve.shapes import AbsentShape, PerShape, RollingShape, WithinShape, condition_words


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
    elif isinstance(shape, AbsentShape):
        total = _hidden(columns, rows, TOTAL)
        things = plural(shape.member)
        when = span or "ever"
        count = int(total) if total is not None else 0
        activity = _noun(shape.activity_label)
        if rows:
            answer["headline"] = (f"{count:,} {things if count != 1 else shape.member} had no {activity} {when}" +
                                  (f"; the first {len(rows):,} are listed." if count > len(rows) else "."))
        else:
            answer["headline"] = f"Every {shape.member} listed had {activity} {when}."
        definition = next((n for n in logical.notes if n.startswith("Listed:")), "")
    if any(g.bands for g in logical.groups):
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
