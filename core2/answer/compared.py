"""Last year in full beside this year so far: what a comparison of the year to date adds from one more query.

"What is my revenue comparison?" compares this year so far with the same days last year, month by month. An
analyst lays last year's whole year beside it: the months this year has not reached yet (drawn and listed as
last year's only, under the months they line up with), the whole of the month in progress, last year in full,
and what the rest of this year needs to match it. One query of last year's months gives all of it; a query that
fails costs these figures and never the answer.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from typing import Any

from core2.answer.builder import _day, _number, a_year_before, said
from core2.answer.facts import STANDS, WATCH, Fact, amount, period
from core2.plan.ir import Plan

log = logging.getLogger(__name__)

Run = Callable[[Plan], tuple[Any, Any, Any]]       # a plan -> (its logical, its compiled query, its result)


def _add_months(day: dt.date, months: int) -> dt.date:
    index = day.year * 12 + (day.month - 1) + months
    return dt.date(index // 12, index % 12 + 1, 1)


_MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre",
         "décembre"]


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def _tout(what: str) -> str:
    """ "Tout 2025", "Tout juin 2025", "Tout l'exercice ..."; "Toute" before a feminine word."""
    feminine = what.startswith(("la ", "l'année", "l’année"))
    return f"{'Toute' if feminine else 'Tout'} {what}"


def _a_year_on(day: dt.date) -> dt.date:
    try:
        return day.replace(year=day.year + 1)
    except ValueError:
        return day.replace(year=day.year + 1, day=28)


def last_year_in_full(plan: Plan, logical: Any, payload: dict[str, Any], run: Run) -> None:
    """Last year's whole year beside this year so far, in place in ``payload``: its remaining periods (table and
    chart), its total (a totals row and a figure under the lead), the whole of the period in progress, and what
    the rest of this year needs. Only for a year so far compared with the same days a year before."""
    compared = payload.get("compared")
    window = plan.time.window if plan.time else None
    if not compared or window is None or window.kind != "to_date" or window.unit not in ("year", "fiscal_year") \
            or not a_year_before(logical) or compared.get("grain") not in ("month", "quarter"):
        return
    c = logical.compare
    if c is None or c.start is None or c.end is None:
        return
    full_end = _a_year_on(c.start)
    if c.end >= full_end:
        return                              # this year's window is a whole year: nothing more of last year
    grain = compared["grain"]
    data = plan.model_dump(mode="json", exclude_none=True)
    time = {k: v for k, v in (data.get("time") or {}).items() if k not in ("compare", "window", "grain")}
    time.update({"window": {"kind": "between", "start": c.start.isoformat(),
                            "end": (full_end - dt.timedelta(days=1)).isoformat()}, "grain": grain})
    data.update({"intent": "trend", "measures": plan.measures[:1], "derived": [], "rates": [], "sort": [],
                 "group_by": [], "time": time, "follow_up": "new"})
    for key in ("limit", "chart"):
        data.pop(key, None)
    try:
        _lg, compiled, result = run(Plan.model_validate(data))
    except Exception as exc:  # noqa: BLE001 - the answer stands without last year in full
        log.info("core2 comparison: last year in full failed: %s", exc)
        return
    names = [str(n) for n in result.columns]
    at = {col.role: names.index(col.name) for col in compiled.columns if col.role in ("period", "measure")
          and col.name in names}
    if "period" not in at or "measure" not in at:
        return
    months: dict[dt.date, float] = {}
    for row in result.rows:
        day, value = _day(row[at["period"]]), _number(row[at["measure"]])
        if day is not None and value is not None:
            months[day] = months.get(day, 0.0) + value
    if not months:
        return
    shown = [d for r in (payload.get("data") or {}).get("rows") or [] for d in [_day(r.get(compared["period"]))] if d]
    if not shown:
        return
    last = max(shown)                                  # this year's last period on screen
    after = sorted(d for d in months if _add_months(d, 12) > last)
    full = sum(months.values())
    year = c.start.year if (c.start.month, c.start.day) == (1, 1) else None
    closing = full_end - dt.timedelta(days=1)
    # A calendar year by its number; a fiscal year by the month it closes ("the fiscal year to Jun 2026").
    year_en = str(year) if year else f"the fiscal year to {closing:%b %Y}"
    year_fr = str(year) if year else f"l'exercice clos fin {_MOIS[closing.month - 1]} {closing.year}"
    fmt = compared["format"]

    # Last year's periods after this year's last one: listed and drawn as last year's only.
    reference = []
    named: dict[str, dict[str, str]] = {}      # each later period on the chart, named as the period it holds
    sample = str(next(iter((payload.get("data") or {}).get("rows") or [{}])).get(compared["period"]) or "")
    for d in after:
        on = _add_months(d, 12)
        label = on.isoformat()[:len(sample)] if len(sample) in (7, 10) else on.isoformat()   # as the rows write it
        reference.append({compared["period"]: label, compared["column"]: None, compared["prior"]: months[d],
                          **({compared["change"]: None} if compared.get("change") else {}),
                          **({compared["pct"]: None} if compared.get("pct") else {})})
        named[label] = {"en": period(d, grain, long=True), "fr": period(d, grain, lang="fr", long=True)}
    data_block = payload.setdefault("data", {})
    if reference:
        data_block["reference_rows"] = reference
        chart = payload.get("chart") or {}
        x, ys = chart.get("x_key"), chart.get("y_keys") or []
        if chart and x == compared["period"] and compared["prior"] in ys:
            chart["rows"] = [*(chart.get("rows") or []),
                             *({x: r[compared["period"]], **{y: (r[compared["prior"]] if y == compared["prior"]
                                                                  else None) for y in ys}} for r in reference)]
            cmp = chart.setdefault("compare", {})
            # Last year's later periods are last year's: their bars and tooltips say so, and they are no place to
            # drill into this year. The quiet bars, all of last year's now, are named by the year.
            cmp["reference"] = named
            if year:
                cmp["prior_label"] = cmp["prior_label_fr"] = str(year)
            why = chart.get("why") or {}
            if why.get("en"):
                chart["why"] = {"en": f"{why['en']} The rest of {year_en} is drawn for reference.",
                                "fr": f"{why.get('fr') or why['en']} Le reste de {year_fr} est dessiné pour référence."}
    full_en, full_fr = f"{_capital(year_en)} in full", f"{_capital(year_fr)} en entier"
    data_block.setdefault("totals", []).append({"label": full_en, "label_fr": full_fr,
                                                "values": {compared["prior"]: full}})
    tile = {"key": "full", "label": full_en, "label_fr": full_fr, "value": full, "format": fmt}
    compared["tiles"].append({**tile, **said(tile)})
    compared["full"] = full

    # What the rest of this year needs; the whole of the period in progress last year.
    reading = payload.setdefault("reading", {"shape": "compared", "lead": {}, "one_offs": [], "facts": []})
    now = float(compared.get("now") or 0.0)
    this_en = str(year + 1) if year else "this fiscal year"
    this_fr = str(year + 1) if year else "cet exercice"
    facts: list[Fact] = []
    if full > now:
        match = f"{year_en}'s {amount(full, fmt)}" if year else f"all of {year_en} ({amount(full, fmt)})"
        facts.append(Fact("rest_of_year", WATCH,
                          f"To match {match}, the rest of {this_en} needs {amount(full - now, fmt)}.",
                          f"Pour égaler les {amount(full, fmt, lang='fr')} de {year_fr}, le reste de {this_fr} doit "
                          f"apporter {amount(full - now, fmt, lang='fr')}.",
                          [full, full - now], size=2.4,
                          how=f"All of {year_en} less {this_en} so far.",
                          how_fr=f"{_tout(year_fr)}, moins {this_fr} à ce jour."))
    else:
        facts.append(Fact("rest_of_year", STANDS,
                          f"{this_en.capitalize()} so far ({amount(now, fmt)}) is already above all of {year_en} "
                          f"({amount(full, fmt)}).",
                          f"{_capital(this_fr)} à ce jour ({amount(now, fmt, lang='fr')}) dépasse déjà "
                          f"{_tout(year_fr).lower()} ({amount(full, fmt, lang='fr')}).",
                          [now, full], size=2.8, how=f"{_capital(this_en)} so far against all of {year_en}.",
                          how_fr=f"{_capital(this_fr)} à ce jour contre {_tout(year_fr).lower()}."))
    if last in set(logical.partial):
        before = _add_months(last, -12)
        if before in months:
            facts.append(Fact("whole_month", WATCH,
                              f"All of {period(before, grain, long=True)} came to {amount(months[before], fmt)}.",
                              f"{period(before, grain, lang='fr', long=True)} entier : "
                              f"{amount(months[before], fmt, lang='fr')}.",
                              [months[before]], size=1.6,
                              how=f"The whole of {period(before, grain)}, a year before the {grain} in progress.",
                              how_fr=f"{_tout(period(before, grain, lang='fr'))}, un an avant la période en cours."))
    reading["facts"] = [*(reading.get("facts") or []), *(f.as_dict() for f in facts)]
