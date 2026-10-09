"""What an answer shows, beyond its first sentence: up to three findings from its own rows.

The first sentence says the answer (the leader, the total, the first and last
period). The findings say what a reader would look for next, each worked out
from the rows the query returned, never written by the AI and never about
anything the rows do not hold:

* a breakdown or a ranking: how concentrated it is ("the top 3 of the 12
  stores make 58% of the total"), how far the leader is ahead or how close the
  race is, anything below zero, the range of an average, a rate or days, and
  where a second measure has another leader; a ranking from the lowest says how
  little the lowest make and how far the lowest is behind the next;
* a series: the change from first to last period in percent, the biggest move
  between two periods, a run of rises or falls up to the latest period, the
  latest against the series' average, and its low point;
* a comparison by member: how many rose and fell, how much of the change one
  member made, and who moved most in percent;
* one number: how it compares with the period before, from one more small
  query (the service runs it: the rows of one number cannot say it).

Periods the data only partly covers are left out of every finding, and a
quantity kept apart by unit is read in its main unit only.
"""

from __future__ import annotations

import re
import statistics

from core2.answer.builder import _Columns, _day, _lower, _noun, _number, _Units, fmt
from core2.compile.compiler import OutColumn
from core2.resolve.resolver import Logical, OutMeasure, adds_up
from core2.resolve.time import label as period_label

MAX_FINDINGS = 3


def summarize(logical: Logical, cols: _Columns, raw: list[dict], units: _Units, *,
              truncated: bool = False) -> list[str]:
    """Up to three findings for the answer on screen; none when the rows say nothing more."""
    measures = cols.of("measure")
    if raw and logical.intent == "list":
        return _listing(cols, raw)[:MAX_FINDINGS]
    if not raw or not measures:
        return []
    periods = cols.of("period")
    members = [c for c in cols.of("attribute") if c.name != logical.unit_group]
    rows = units.main_rows(raw)
    unit = units.main if units.column is not None else None
    m = measures[0]
    out = {o.name: o for o in logical.measures}
    found: list[str] = []
    if logical.compare is not None and members and not periods:
        found = _compared(logical, cols, rows, m, out.get(m.name), members[0], unit)
    elif periods and not members:
        found = _series(logical, rows, m, periods[0], unit)
    elif periods and members and logical.compare is None:
        found = _members_over_time(logical, rows, m, periods[0], members[0], unit)
    elif members and not periods and logical.compare is None:
        found = _breakdown(logical, cols, rows, measures, out, members[0], unit, truncated=truncated)
    return found[:MAX_FINDINGS]


# ── a line per member ──────────────────────────────────────────────────────


def _members_over_time(logical: Logical, rows: list[dict], m: OutColumn, p: OutColumn, g: OutColumn,
                       unit: str | None) -> list[str]:
    """Who led in each period, and who rose and fell most from the first period to the last."""
    partial = {d.isoformat() for d in logical.partial}
    series: dict[str, dict] = {}
    for r in rows:
        day, value = _day(r[p.name]), _number(r[m.name])
        if day is None or value is None or day.isoformat() in partial or r[g.name] in (None, ""):
            continue
        series.setdefault(str(r[g.name]), {})[day] = value
    days = sorted({d for points in series.values() for d in points})
    if len(series) < 2 or len(days) < 2:
        return []
    grain = p.grain or "month"
    name = lambda day: period_label(day, grain, fiscal_start=logical.fiscal_start)   # noqa: E731
    word = grain.replace("fiscal_", "fiscal ").replace("_", " ")
    noun = _noun(g.label)
    found: list[str] = []

    leaders = [max(((member, points[d]) for member, points in series.items() if d in points),
                   key=lambda x: (x[1], x[0]))[0] for d in days]
    top = max(set(leaders), key=lambda member: (leaders.count(member), member))
    times = leaders.count(top)
    if times == len(days):
        found.append(f"{top} led in every {word}, of the {len(series)} {noun}.")
    elif times > len(days) / 2:
        others = sorted({member for member in leaders if member != top})
        found.append(f"{top} led in {times} of the {len(days)} {word}s; "
                     f"{_listed_names(others)} in the others.")

    first, last = days[0], days[-1]
    moves = [(member, ratio) for member, points in series.items() if first in points and last in points
             for ratio in [_pct(points[first], points[last])] if ratio is not None]
    if len(moves) >= 2:
        up = max(moves, key=lambda x: (x[1], x[0]))
        down = min(moves, key=lambda x: (x[1], x[0]))
        if up[1] > 0.005 and down[1] < -0.005:
            found.append(f"From {name(first)} to {name(last)}, {up[0]} rose most ({_signed_pct(up[1])}) and "
                         f"{down[0]} fell most ({_signed_pct(down[1])}).")
        elif up[1] > 0.005:
            found.append(f"From {name(first)} to {name(last)}, every one of them rose; {up[0]} most "
                         f"({_signed_pct(up[1])}).")
        elif down[1] < -0.005:
            found.append(f"From {name(first)} to {name(last)}, every one of them fell; {down[0]} most "
                         f"({_signed_pct(down[1])}).")
    return found


def _listed_names(names: list[str]) -> str:
    if len(names) <= 2:
        return " and ".join(names)
    return f"{', '.join(names[:2])} and {len(names) - 2} more"


# ── a listing ──────────────────────────────────────────────────────────────


def _listing(cols: _Columns, rows: list[dict]) -> list[str]:
    """What a list holds beyond its count: how its rows fall into a small grouping, and the range of a figure."""
    found: list[str] = []
    texts = [c for c in cols.columns if c.role in ("attribute", "member_code") and c.format in (None, "", "text")]
    for c in texts[1:]:
        counts: dict[str, int] = {}
        for r in rows:
            if r.get(c.name) not in (None, ""):
                counts[str(r[c.name])] = counts.get(str(r[c.name]), 0) + 1
        if 2 <= len(counts) <= 8 and sum(counts.values()) == len(rows):
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            shown = ", ".join(f"{k} ({n})" for k, n in ranked[:3])
            more = f" and {len(ranked) - 3} more" if len(ranked) > 3 else ""
            word = re.sub(r"\s+(name|description|desc|label|title)$", "", _lower(c.label))
            found.append(f"By {word}: {shown}{more}.")
            break
    for c in cols.columns:
        if c.role not in ("measure", "attribute") or c.format in (None, "", "text", "date"):
            continue
        values = [v for v in (_number(r.get(c.name)) for r in rows) if v is not None]
        if len(values) >= 3 and min(values) != max(values):
            found.append(f"{c.label} runs from {fmt(min(values), c.format)} to {fmt(max(values), c.format)}.")
            break
    return found


# ── what adds up ───────────────────────────────────────────────────────────


def _adds_up(o: OutMeasure | None, column: OutColumn) -> bool:
    """Do the groups' values add up to the whole (a sum or a count), so shares of it mean something?"""
    return o is not None and column.format not in ("percent", "percentage", "days") and adds_up(o)


def _pct(a: float, b: float) -> float | None:
    """The change from ``a`` to ``b`` as a fraction of ``a``; None when ``a`` is zero."""
    return (b - a) / abs(a) if a else None


def _signed_pct(ratio: float) -> str:
    return f"{ratio * 100:+.0f}%" if abs(ratio) >= 0.1 else f"{ratio * 100:+.1f}%"


# ── a breakdown or a ranking ───────────────────────────────────────────────


def _breakdown(logical: Logical, cols: _Columns, rows: list[dict], measures: list[OutColumn],
               out: dict[str, OutMeasure], g: OutColumn, unit: str | None, *, truncated: bool) -> list[str]:
    m = measures[0]
    named = [(str(r[g.name]), _number(r[m.name])) for r in rows if r[g.name] not in (None, "")]
    values = [(name, v) for name, v in named if v is not None]
    if len(values) < 3 or len({v for _, v in values}) == 1:
        return []                     # nothing to rank, or every group the same (the sentence says so)
    noun = _noun(g.label)
    show = lambda v: fmt(v, m.format, unit=unit)   # noqa: E731
    ranked = sorted(values, key=lambda x: (-x[1], x[0]))
    if len({show(v) for _, v in values}) == 1:
        # Different only past what is shown (38.0% everywhere): that is the finding.
        return [f"All {len(values)} {noun} are at {show(ranked[0][1])}."]
    # Ranked from the lowest by a measure: the top of the list is not what was asked about.
    upward = bool(logical.sort) and not logical.sort[0][1] and logical.sort[0][0] in {c.name for c in measures}
    lowest_first = upward and logical.sort[0][0] == m.name
    found: list[str] = []

    below = [(n, v) for n, v in values if v < 0]
    if below and _adds_up(out.get(m.name), m):
        total_below = sum(v for _, v in below)
        if len(below) == 1:
            found.append(f"{below[0][0]} is below zero, at {show(below[0][1])}.")
        else:
            found.append(f"{len(below)} {noun} are below zero, {show(total_below)} together.")

    share = next((c for c in cols.of("share") if c.measure == m.measure), None)
    limited = bool(logical.limit) or truncated
    if _adds_up(out.get(m.name), m) and not below and not upward:
        total = sum(v for _, v in values)
        named_rows = [r for r in rows if r[g.name] not in (None, "") and _number(r[m.name]) is not None]
        shares = ([_number(r[share.name]) or 0.0 for r in sorted(named_rows, key=lambda r: -(_number(r[m.name]) or 0.0))]
                  if share is not None else None)
        if shares is None and not limited and total > 0:
            shares = [v / total for _, v in ranked]
        if shares:
            whole = "of the grand total" if share is not None else "of the total"
            running, k = 0.0, 0
            for s in shares:
                running += s
                k += 1
                if running >= 0.8:
                    break
            if len(shares) >= 10 and running >= 0.8 and k <= len(shares) // 2:
                found.append(f"{k} of the {len(shares)} {noun} make {running:.0%} {whole}.")
            elif len(shares) >= 5:
                top3 = sum(shares[:3])
                found.append(f"The top 3 of the {len(shares)} {noun} shown make {top3:.0%} {whole}."
                             if limited else f"The top 3 of the {len(shares)} {noun} make {top3:.0%} {whole}.")

    if lowest_first and _adds_up(out.get(m.name), m) and not below and not limited and len(values) >= 5:
        total = sum(v for _, v in values)
        bottom = sum(v for _, v in ranked[-3:])
        if total > 0:
            found.append(f"The 3 lowest of the {len(values)} {noun} make {bottom / total:.0%} of the total.")
    (low_name, low), (next_name, next_low) = ranked[-1], ranked[-2]
    if lowest_first and low > 0 and next_low >= 1.5 * low:
        found.append(f"{low_name} is {1 - low / next_low:.0%} below the next lowest, {next_name} ({show(next_low)}).")

    (first, v1), (second, v2) = ranked[0], ranked[1]
    if v1 > 0 and v2 > 0 and not upward:
        if v1 >= 1.5 * v2:
            found.append(f"{first} is {v1 / v2:.1f} times the next, {second} ({show(v2)}).")
        elif (v1 - v2) / v1 <= 0.02 and _adds_up(out.get(m.name), m):   # averages close together: the range says it
            apart = "less than 1%" if (v1 - v2) / v1 < 0.01 else f"{(v1 - v2) / v1:.0%}"
            found.append(f"{first} and {second} are {apart} apart ({show(v1)} and {show(v2)}).")

    if not _adds_up(out.get(m.name), m) and len(values) >= 5:
        low_name, low = ranked[-1]
        middle = statistics.median(v for _, v in values)
        found.append(f"From {show(low)} ({low_name}) to {show(v1)} ({first}); half the {noun} are above "
                     f"{show(middle)}.")

    for c in measures[1:]:
        others = [(str(r[g.name]), _number(r[c.name])) for r in rows if r[g.name] not in (None, "")]
        others = [(n, v) for n, v in others if v is not None]
        if len(others) < 3 or len({fmt(v, c.format, unit=unit) for _, v in others}) == 1:
            continue                  # every group the same on it: no other leader to name
        lead, value = max(others, key=lambda x: (x[1], x[0]))
        if lead != first:
            most = "the most" if c.format == "days" else "the highest"
            found.append(f"{lead} has {most} {_lower(c.label)} ({fmt(value, c.format, unit=unit)}), "
                         f"though {first} leads on {_lower(m.label)}.")
            break
    return found


# ── a series ───────────────────────────────────────────────────────────────


def _series(logical: Logical, rows: list[dict], m: OutColumn, p: OutColumn, unit: str | None) -> list[str]:
    partial = {d.isoformat() for d in logical.partial}
    points = sorted(((day, v) for r in rows for day in [_day(r[p.name])] for v in [_number(r[m.name])]
                     if day is not None and v is not None and day.isoformat() not in partial),
                    key=lambda x: x[0])
    if len(points) < 3:
        return []
    grain = p.grain or "month"
    name = lambda day: period_label(day, grain, fiscal_start=logical.fiscal_start)   # noqa: E731
    show = lambda v: fmt(v, m.format, unit=unit)   # noqa: E731
    values = [v for _, v in points]
    if max(values) == min(values):
        return []                     # flat: the sentence says "in every month"
    word = grain.replace("fiscal_", "fiscal ").replace("_", " ")
    found: list[str] = []

    first, last = points[0], points[-1]
    whole = _pct(first[1], last[1])
    if whole is not None and abs(whole) >= 0.005:
        found.append(f"{name(last[0])} is {abs(whole):.0%} {'above' if whole > 0 else 'below'} "
                     f"{name(first[0])}." if abs(whole) >= 0.01 else
                     f"{name(last[0])} is within 1% of {name(first[0])}.")

    steps = [(points[i][0], r) for i in range(1, len(points))
             for r in [_pct(points[i - 1][1], points[i][1])] if r is not None]
    if steps:
        day, biggest = max(steps, key=lambda s: abs(s[1]))
        if abs(biggest) >= 0.05:
            found.append(f"The biggest move between two {word}s was {_signed_pct(biggest)}, into {name(day)}.")

    run = 0
    direction = 0
    for i in range(len(values) - 1, 0, -1):
        step = (values[i] > values[i - 1]) - (values[i] < values[i - 1])
        if step == 0 or (direction and step != direction):
            break
        direction, run = step, run + 1
    if run >= 3:
        found.append(f"{'Up' if direction > 0 else 'Down'} {run} {word}s in a row, up to {name(last[0])}.")

    average = sum(values) / len(values)
    latest = _pct(average, last[1])
    if latest is not None and abs(latest) >= 0.05 and len(points) >= 4:
        found.append(f"{name(last[0])} is {abs(latest):.0%} {'above' if latest > 0 else 'below'} the average of "
                     f"the {len(points)} {word}s ({show(average)}).")

    low_day, low = min(points, key=lambda x: (x[1], x[0]))
    if low_day not in (first[0], last[0]):
        found.append(f"The low point was {name(low_day)}, at {show(low)}.")
    return found


# ── a comparison by member ─────────────────────────────────────────────────


def _compared(logical: Logical, cols: _Columns, rows: list[dict], m: OutColumn, o: OutMeasure | None,
              g: OutColumn, unit: str | None) -> list[str]:
    change = next((c for c in cols.columns if c.role == "change" and c.measure == m.measure), None)
    pct = next((c for c in cols.columns if c.role == "pct_change" and c.measure == m.measure), None)
    prior = next((c for c in cols.columns if c.role == "prior" and c.measure == m.measure), None)
    if change is None:
        return []
    moves: list[tuple[str, float, float | None, float | None]] = []
    for r in rows:
        if r[g.name] in (None, ""):
            continue
        by = _number(r[change.name])
        if by is None:
            continue
        moves.append((str(r[g.name]), by, _number(r[pct.name]) if pct else None,
                      _number(r[prior.name]) if prior else None))
    if len(moves) < 3:
        return []
    noun = _noun(g.label)
    show = lambda v: fmt(abs(v), m.format, unit=unit)   # noqa: E731
    found: list[str] = []
    up = sum(1 for _, by, _, _ in moves if by > 0)
    down = sum(1 for _, by, _, _ in moves if by < 0)
    if up and down:                   # all one way is the sentence's own ("all 8 rose")
        found.append(f"{up} of the {len(moves)} {noun} rose and {down} fell.")

    net = sum(by for _, by, _, _ in moves)
    if _adds_up(o, m) and net:
        same = [x for x in moves if (x[1] > 0) == (net > 0)]
        if same:
            name, by, _, _ = max(same, key=lambda x: (abs(x[1]), x[0]))
            part = by / net
            rise = "rise" if net > 0 else "fall"
            if 0.4 <= part <= 1:
                found.append(f"{name} alone made {part:.0%} of the net {rise} ({show(by)} of {show(net)}).")
            elif part > 1:
                found.append(f"{name} {'rose' if net > 0 else 'fell'} {show(by)}, more than the net {rise} of "
                             f"{show(net)}: the other {noun} {'fell' if net > 0 else 'rose'} {show(net - by)} "
                             f"together.")

    rated = [(n, r) for n, _, r, before in moves if r is not None and before]
    if len(rated) >= 3:
        biggest_by = max(moves, key=lambda x: (abs(x[1]), x[0]))[0]
        name, ratio = max(rated, key=lambda x: (abs(x[1]), x[0]))
        if name != biggest_by and abs(ratio) >= 0.1:
            found.append(f"In percent, {name} moved most ({_signed_pct(ratio)}).")
    new = [n for n, by, _, before in moves if not before and by > 0]
    if new:
        found.append(f"{len(new)} {noun if len(new) > 1 else _singular(noun)} had none before: "
                     f"{', '.join(new[:3])}{' and more' if len(new) > 3 else ''}.")
    return found


def _singular(noun: str) -> str:
    from core2.bootstrap.names import singular

    return singular(noun)
