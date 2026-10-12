"""What an analyst checks next, from up to three small queries more: the facts the answer's own rows cannot hold.

A figure alone ("Net amount in Q1 2026: $543K") says nothing of whether it is good. The note now says, each from
one query of its own: what made a one-off period stand out (its largest member and the share it holds), how the
figure compares with the same period a year before, how it stands against its target where the data keeps one,
and which member moved it most since the period before. Each fact carries its figures, its words in English and
French and how it was worked out; a query that fails, or finds nothing to say, costs that fact and never the
answer.
"""

from __future__ import annotations

import calendar
import datetime as dt
import logging
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from core2.answer.drivers import Grouping, candidates
from core2.answer.facts import STANDS, WATCH, WHY, Fact, amount, percent, period
from core2.model.schema import SemanticModel
from core2.plan import analyses
from core2.plan.ir import Plan

log = logging.getLogger(__name__)

MOST = 3                  # queries one answer may ask for its note
FLAT = 0.01               # a change under 1% is "about the same"
MOSTLY = 0.5              # a member holding half of a whole is "mostly" it

Run = Callable[[Plan], tuple[Any, Any, Any]]       # a plan -> (its logical, its compiled query, its result)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    return None


def _rows(compiled: Any, result: Any) -> list[dict[str, Any]]:
    names = [str(n) for n in result.columns]
    return [dict(zip(names, row)) for row in result.rows]


def _col(compiled: Any, role: str) -> str | None:
    found = next((c for c in compiled.columns if c.role == role), None)
    return found.name if found is not None else None


def _end(start: dt.date, grain: str) -> dt.date | None:
    """The last day of the period that starts on ``start``."""
    if grain in ("month", "fiscal_month"):
        return start.replace(day=calendar.monthrange(start.year, start.month)[1])
    if grain in ("quarter", "fiscal_quarter"):
        month = start.month + 2
        year = start.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        return dt.date(year, month, calendar.monthrange(year, month)[1])
    if grain == "week":
        return start + dt.timedelta(days=6)
    if grain == "day":
        return start
    if grain in ("year", "fiscal_year"):
        return dt.date(start.year, 12, 31) if start.month == 1 else None
    return None


def extra_facts(plan: Plan, logical: Any, payload: dict[str, Any], model: SemanticModel, run: Run, *,
                allowed: set[str] | None = None, most: int = MOST) -> list[Fact]:
    """The facts the next queries give, the strongest kinds first, at most ``most`` queries."""
    if plan.kind != "query" or plan.shaped() or not logical.measures or logical.intent == "list" \
            or plan.intent in ("drivers", "forecast", "list") or not plan.measures or logical.people:
        return []
    measure = logical.measures[0]
    if measure.hidden or measure.format in ("percent", "percentage") and not plan.target:
        return []
    reading = payload.get("reading") or {}
    rows = (payload.get("data") or {}).get("rows") or []
    if not rows:
        return []
    asks: list[Callable[[], Fact | None]] = []
    one_off = (reading.get("one_offs") or [None])[0]
    period = next((g for g in logical.groups if g.kind == "period"), None)
    members = [g for g in logical.groups if g.kind == "attribute" and g.name != logical.unit_group]
    if one_off and period is not None and period.grain and not members:
        asks.append(lambda: _one_off_why(plan, logical, model, run, one_off, period.grain, allowed))
    single = not plan.group_by and not plan.time.grain
    bounded = logical.window.start is not None and logical.window.end is not None
    if bounded and logical.compare is None and not plan.target:
        asks.append(lambda: _against_target(plan, logical, model, run, allowed))
        if not any(f.get("kind") == "year_on_year" for f in reading.get("facts") or []):
            asks.append(lambda: _last_year(plan, logical, run))
        if single:
            asks.append(lambda: _what_moved_it(plan, logical, model, run, allowed))
    facts: list[Fact] = []
    for ask in asks[:most]:
        try:
            fact = ask()
        except Exception as exc:  # noqa: BLE001 - the note goes without this fact
            log.info("core2 note: one more query failed: %s", exc)
            continue
        if fact is not None:
            facts.append(fact)
    return facts


def _base(plan: Plan, **update: Any) -> Plan:
    """The answer's plan, one figure, no grouping or order of its own, with ``update`` (checked as any plan is)."""
    data = plan.model_dump(mode="json", exclude_none=True)
    time = {**{k: v for k, v in (data.get("time") or {}).items() if k not in ("grain", "compare")},
            **update.pop("time", {})}
    data.update({"measures": plan.measures[:1], "derived": [], "rates": [], "group_by": [], "sort": [],
                 "time": time, "follow_up": "new", **update})
    for key in ("limit", "chart"):
        data.pop(key, None)
    return Plan.model_validate(data)


def _first_grouping(plan: Plan, logical: Any, model: SemanticModel, allowed: set[str] | None) -> str | None:
    if not logical.parts:
        return None
    skip = {f.field for f in plan.filters} | set(plan.group_by)
    found = candidates(model, logical.parts[0].table, allowed=allowed, skip=skip | {s.split(".", 1)[0] for s in skip})
    return found[0] if found else None


def _word(model: SemanticModel, slug: str) -> str:
    name = model.entities[slug].business_name if slug in model.entities else \
        model.attributes[slug].business_name if slug in model.attributes else slug
    return Grouping(slug, name, []).word


def _one_off_why(plan: Plan, logical: Any, model: SemanticModel, run: Run, one_off: dict[str, Any], grain: str,
                 allowed: set[str] | None) -> Fact | None:
    """What made the one-off: the period alone, by the grouping the figure's table reaches first."""
    slug = _first_grouping(plan, logical, model, allowed)
    start = dt.date.fromisoformat(str(one_off["period"])[:10])
    end = _end(start, grain)
    if slug is None or end is None:
        return None
    window = {"kind": "between", "start": start.isoformat(), "end": end.isoformat()}
    probe = _base(plan, intent="breakdown", group_by=[slug], time={"window": window})
    lg, compiled, result = run(probe)
    rows = _rows(compiled, result)
    m = _col(compiled, "measure")
    member = _col(compiled, "attribute")
    if not rows or m is None or member is None:
        return None
    values = [(str(r[member]), v) for r in rows for v in [_number(r[m])] if v is not None and r[member] is not None]
    total = sum(v for _, v in values)
    if len(values) < 2 or not total:
        return None
    name, top = max(values, key=lambda x: x[1])
    share = top / total
    fmt, label = lg.measures[0].format, one_off["label"]
    label_fr = period(start, grain, lang="fr")
    word = _word(model, slug)
    things = f"{word}s" if not word.endswith("s") else word
    if share >= MOSTLY:
        en = (f"The one-off in {label} came mostly from {name}: {amount(top, fmt)} of {amount(total, fmt)} "
              f"({percent(share)}).")
        fr = (f"L'exception de {label_fr} vient surtout de {name} : {amount(top, fmt, lang='fr')} sur "
              f"{amount(total, fmt, lang='fr')} ({percent(share, lang='fr')}).")
    else:
        en = (f"The one-off in {label} was spread across {things}; the largest, {name}, had {amount(top, fmt)} "
              f"({percent(share)}).")
        fr = (f"L'exception de {label_fr} se répartit entre plusieurs {word} ; le plus grand, {name}, pèse "
              f"{amount(top, fmt, lang='fr')} ({percent(share, lang='fr')}).")
    return Fact("one_off_why", WHY, en, fr, [round(top, 2), round(total, 2), round(share, 4)], size=2.9,
                members=[name], query=compiled.sql,
                how=f"The one-off period alone, by {word}: its largest member and its share.",
                how_fr=f"La période exceptionnelle seule, par {word} : son plus grand membre et sa part.")


def _last_year(plan: Plan, logical: Any, run: Run) -> Fact | None:
    """The same figure for the same dates a year before."""
    if (logical.window.end - logical.window.start).days > 370:
        return None                                  # a span of more than a year has no "same period" before it
    probe = _base(plan, intent="compare", time={"compare": {"kind": "same_period_last_year"}})
    lg, compiled, result = run(probe)
    role = lg.parts[0].date.role if lg.parts and lg.parts[0].date else None
    if lg.compare is None or lg.compare.start is None or role is None or role.first is None \
            or lg.compare.start < role.first:
        return None                                  # no data a year before: nothing to compare with
    rows = _rows(compiled, result)
    now_col, before_col = _col(compiled, "measure"), _col(compiled, "prior")
    if len(rows) != 1 or now_col is None or before_col is None:
        return None
    now, before = _number(rows[0][now_col]), _number(rows[0][before_col])
    if now is None or not before:
        return None
    change = (now - before) / abs(before)
    fmt, label = lg.measures[0].format, lg.measures[0].label
    if abs(change) < FLAT:
        en = f"{label} is about the same as the same period a year before ({amount(before, fmt)})."
        fr = f"{label} est comparable à la même période un an plus tôt ({amount(before, fmt, lang='fr')})."
    else:
        en = (f"{label} is {'up' if change > 0 else 'down'} {percent(abs(change))} on the same period a year "
              f"before ({amount(before, fmt)}).")
        fr = (f"{label} est en {'hausse' if change > 0 else 'baisse'} de {percent(abs(change), lang='fr')} sur la "
              f"même période un an plus tôt ({amount(before, fmt, lang='fr')}).")
    return Fact("last_year", STANDS, en, fr, [round(now, 2), round(abs(change), 4), round(before, 2)],
                size=2.0 + min(abs(change) * 4, 1.2), query=compiled.sql,
                how="The same figure for the same dates a year before, from one more query.",
                how_fr="Le même chiffre pour les mêmes dates un an plus tôt, par une requête de plus.")


def _against_target(plan: Plan, logical: Any, model: SemanticModel, run: Run,
                    allowed: set[str] | None) -> Fact | None:
    """Where the data keeps a target for the figure: how far the period got."""
    may = (lambda t: t in allowed) if allowed is not None else (lambda t: True)
    pair = analyses.target_pair(model, may)
    figure = logical.measures[0].measure
    if pair is None or figure is None or pair[0].slug != figure.slug:
        return None
    actual, target = pair
    probe = _base(plan, intent="value", measures=[], target={"actual": actual.slug, "target": target.slug})
    lg, compiled, result = run(probe)
    rows = _rows(compiled, result)
    if len(rows) != 1:
        return None
    got, aim = _number(rows[0].get(actual.slug)), _number(rows[0].get(target.slug))
    if got is None or not aim:
        return None
    share = got / aim
    fmt, label = actual.format, logical.measures[0].label
    if share >= 1:
        en = (f"{label} is {percent(share - 1)} above its target: {amount(got, fmt)} against {amount(aim, fmt)}."
              if share - 1 >= FLAT else f"{label} is on its target: {amount(got, fmt)} against {amount(aim, fmt)}.")
        fr = (f"{label} est {percent(share - 1, lang='fr')} au-dessus de l'objectif : {amount(got, fmt, lang='fr')} "
              f"pour {amount(aim, fmt, lang='fr')}." if share - 1 >= FLAT else
              f"{label} est à l'objectif : {amount(got, fmt, lang='fr')} pour {amount(aim, fmt, lang='fr')}.")
    else:
        en = (f"{label} is at {percent(share)} of its target: {amount(got, fmt)} against {amount(aim, fmt)}, "
              f"{amount(aim - got, fmt)} short.")
        fr = (f"{label} est à {percent(share, lang='fr')} de l'objectif : {amount(got, fmt, lang='fr')} pour "
              f"{amount(aim, fmt, lang='fr')}, soit {amount(aim - got, fmt, lang='fr')} de moins.")
    return Fact("target", STANDS if share >= 1 else WATCH, en, fr, [round(got, 2), round(aim, 2), round(share, 4)],
                size=2.4 + min(abs(1 - share) * 4, 1.0), query=compiled.sql,
                how=f"The figure against {target.business_name.lower()} for the same period, from one more query.",
                how_fr=f"Le chiffre face à {target.business_name.lower()} sur la même période, par une requête de "
                       "plus.")


def _what_moved_it(plan: Plan, logical: Any, model: SemanticModel, run: Run,
                   allowed: set[str] | None) -> Fact | None:
    """Since the period before: the member that moved the figure most, in the direction it moved."""
    if logical.measures[0].measure is None or logical.measures[0].measure.additivity != "additive":
        return None                                  # a ratio's members do not add up to its change
    slug = _first_grouping(plan, logical, model, allowed)
    if slug is None:
        return None
    probe = _base(plan, intent="compare", group_by=[slug], time={"compare": {"kind": "previous_period"}})
    lg, compiled, result = run(probe)
    role = lg.parts[0].date.role if lg.parts and lg.parts[0].date else None
    if lg.compare is None or lg.compare.start is None or role is None or role.first is None \
            or lg.compare.start < role.first:
        return None
    rows = _rows(compiled, result)
    now_col, before_col, member = _col(compiled, "measure"), _col(compiled, "prior"), _col(compiled, "attribute")
    if not rows or now_col is None or before_col is None or member is None:
        return None
    moves = [(str(r[member]), (_number(r[now_col]) or 0.0) - (_number(r[before_col]) or 0.0)) for r in rows
             if r[member] is not None]
    before_total = sum(_number(r[before_col]) or 0.0 for r in rows)
    net = sum(c for _, c in moves)
    if len(moves) < 2 or not before_total or abs(net) / abs(before_total) < FLAT:
        return None
    name, top = max(moves, key=lambda x: x[1] if net > 0 else -x[1])
    if (top > 0) != (net > 0):
        return None
    fmt = lg.measures[0].format
    signed = lambda v, lang="en": ("+" if v > 0 else "−") + amount(abs(v), fmt, lang=lang)  # noqa: E731
    rise = net > 0
    if abs(top) >= MOSTLY * abs(net):
        en = (f"Most of the {'rise' if rise else 'fall'} since the period before came from {name} "
              f"({signed(top)} of {signed(net)}).")
        fr = (f"L'essentiel de la {'hausse' if rise else 'baisse'} depuis la période précédente vient de {name} "
              f"({signed(top, 'fr')} sur {signed(net, 'fr')}).")
    else:
        en = (f"The {'rise' if rise else 'fall'} since the period before ({signed(net)}) was spread out; "
              f"{name} moved most ({signed(top)}).")
        fr = (f"La {'hausse' if rise else 'baisse'} depuis la période précédente ({signed(net, 'fr')}) est "
              f"répartie ; {name} a le plus varié ({signed(top, 'fr')}).")
    word = _word(model, slug)
    return Fact("moved_by", WHY, en, fr, [round(top, 2), round(net, 2)], size=2.3 + min(abs(top / net), 1.0),
                members=[name], query=compiled.sql,
                how=f"Each {word}'s change since the period before, from one more query; the largest in the "
                    "direction of the whole.",
                how_fr=f"La variation de chaque {word} depuis la période précédente, par une requête de plus ; "
                       "la plus grande dans le sens de l'ensemble.")
