"""What the data of a chart looks like, measured before it is drawn, and the form that follows from it.

The builder (core2/answer/builder.py) picks a chart for the kind of answer: a line for periods, bars for members, a
pie for a share asked for. The data itself then decides what reads well: a line through three months, a pie of
one slice, forty thin bars, a donut of amounts below zero, a jagged line nobody can read a trend in, grouped bars
in dozens of colours. Each is measured here -- points, the periods missing, the largest member's share, how many
members make 80%, values below zero, how much a series swings -- and the form put right:

=========================================  ==========================================  ===========================
When the data shows                        Drawn                                       Instead of
=========================================  ==========================================  ===========================
One period only                            The number with its change (no chart)      A one-point line
Fewer than 4 periods, or gaps              Bars by period, the gaps left empty         A line through missing months
Up to 6 parts of a whole, largest < 70%    A donut                                     Bars
One member over 90% of the whole           Bars, that member named in the note         A pie with one slice
More than 12 members                       The largest 10 and Other, largest first     40 thin bars
Values above and below zero                Bars from a zero line                       A donut or an area
A series that swings                       A line with its 3-period average            A jagged line alone
Two groupings, most cells filled          A grid                                      Grouped bars in a dozen colours
More than 6 lines                          The largest 5 and Other                     A tangle of lines
=========================================  ==========================================  ===========================

The note under the chart says what decided it ("A line: 24 months, rising."), in English and French. A reader's
own pick of a form still wins (applied after this, by the builder); the forms the data rules out (a donut of
amounts below zero) are no longer offered. The same chart spec draws the chat's answer, a dashboard tile, and the
pictures of a scheduled email and its PDF, so the rules hold in all of them.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

FEW_PERIODS = 4          # fewer periods than this read as bars, not a line
COMPARED_BARS = 13       # a comparison by period up to this many periods: bars side by side; past it, two lines
MOST_PARTS = 6           # parts of a whole drawn as a donut
DONUT_LARGEST = 0.70     # ... while the largest is under this share
ONE_MEMBER = 0.90        # one member this much of the whole is named, never drawn as a slice
MOST_BARS = 12           # members drawn one bar each
TOP_BARS = 10            # past that, the largest drawn
SWINGS = 8               # points a series needs before its swings are measured
SWING = 0.25             # typical (median) move from one period to the next, against the series' median level
GRID_FILLED = 0.70       # share of the pairs of two groupings with a value, for a grid
GRID_SERIES = 6          # colours past which a filled cross is a grid (as the builder draws one)
AVERAGE_KEY = "__average"

_GRAIN = {"day": ("day", "days", "jour", "jours"), "week": ("week", "weeks", "semaine", "semaines"),
          "month": ("month", "months", "mois", "mois"), "quarter": ("quarter", "quarters", "trimestre", "trimestres"),
          "year": ("year", "years", "année", "années")}


@dataclass
class Shape:
    """The measures of a chart's data the form is chosen by."""
    points: int = 0                     # rows with a value
    slots: int = 0                      # rows drawn, periods missing included
    gaps: int = 0                       # periods with no value at all
    negatives: bool = False
    positives: bool = False
    zeros: int = 0
    largest: str = ""                   # the largest member
    largest_share: float | None = None  # its share of the whole, when the values are parts of one
    to_80: int | None = None            # members making 80% of the whole
    spread: float | None = None         # the largest value against the smallest above zero
    swing: float | None = None          # median move between periods, against the median level (a one-off
                                        # month moves it little: it is an outlier, not a swing)
    outliers: list[str] = field(default_factory=list)
    direction: str = ""                 # rising | falling | steady, for a series
    filled: float | None = None         # share of the pairs of two groupings with a value


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def measure(chart: dict[str, Any]) -> Shape:
    """The shape of a chart's data, from its own rows: what is drawn, before any form is chosen."""
    rows = list(chart.get("rows") or [])
    ys = [y for y in chart.get("y_keys") or [] if y != AVERAGE_KEY]
    x = str(chart.get("x_key") or "")
    out = Shape(slots=len(rows))
    if not rows or not ys:
        return out
    values = [[_num(r.get(y)) for y in ys] for r in rows]
    every = [v for row in values for v in row if v is not None]
    out.points = sum(1 for row in values if any(v is not None for v in row))
    out.gaps = len(rows) - out.points
    out.negatives = any(v < 0 for v in every)
    out.positives = any(v > 0 for v in every)
    out.zeros = sum(1 for v in every if v == 0)
    if chart.get("grouped_by"):
        out.filled = sum(1 for v in every if v) / max(1, len(rows) * len(ys))
    first = [row[0] for row in values]
    known = [v for v in first if v is not None]
    if len(ys) == 1 and known:
        ranked = sorted(((v, str(r.get(x))) for v, r in zip(first, rows) if v is not None), reverse=True)
        out.largest = ranked[0][1]
        total = sum(known)
        if total > 0 and not out.negatives:
            out.largest_share = ranked[0][0] / total
            running, n = 0.0, 0
            for v, _ in ranked:
                running, n = running + v, n + 1
                if running >= 0.8 * total:
                    break
            out.to_80 = n
        above = [v for v in known if v > 0]
        out.spread = max(above) / min(above) if above else None
        if len(known) >= 5:
            middle = statistics.median(known)
            mad = statistics.median(abs(v - middle) for v in known) or 0.0
            out.outliers = [name for v, name in ranked if mad and abs(v - middle) > 5 * mad][:3]
        if len(known) >= 2:
            level = statistics.median(abs(v) for v in known)
            moves = [abs(b - a) for a, b in zip(known, known[1:])]
            out.swing = statistics.median(moves) / level if level else None
            third = max(1, len(known) // 3)
            # The middle of each end, so one month far off the rest decides neither.
            start, end = statistics.median(known[:third]), statistics.median(known[-third:])
            change = (end - start) / abs(start) if start else (1.0 if end > 0 else -1.0 if end < 0 else 0.0)
            out.direction = "rising" if change > 0.05 else "falling" if change < -0.05 else "steady"
    return out


# ── the form the data asks for ───────────────────────────────────────────────


def _temporal(chart: dict[str, Any]) -> bool:
    x = (chart.get("chart_spec") or {}).get("x") or {}
    return x.get("role") == "temporal"


def _grain(chart: dict[str, Any], grain: str | None) -> tuple[str, str, str, str]:
    return _GRAIN.get((grain or "month").replace("fiscal_", ""), _GRAIN["month"])


def _percent(share: float) -> str:
    return f"{share * 100:.0f}%"


def _offer_without(chart: dict[str, Any], *kinds: str) -> None:
    for key in ("renderable_types", "allowed_types"):
        chart[key] = [k for k in chart.get(key) or [] if k not in kinds] or ["bar"]


def _set(chart: dict[str, Any], kind: str) -> None:
    chart["chart_type"] = chart["recommended_type"] = kind
    for key in ("renderable_types", "allowed_types"):
        if kind not in (chart.get(key) or []):
            chart[key] = [kind, *(chart.get(key) or [])]


def _say(chart: dict[str, Any], en: str, fr: str) -> None:
    chart["why"] = {"en": en, "fr": fr}


def _members_of_one_period(chart: dict[str, Any]) -> dict[str, Any] | None:
    """A line per member over one period: the members side by side as bars (a one-point line shows nothing)."""
    rows = chart.get("rows") or []
    member, measure = chart.get("grouped_by"), chart.get("grouped_measure")
    if len(rows) != 1 or not member or not measure:
        return None
    row = rows[0]
    series = [s for s in chart.get("y_keys") or [] if s != AVERAGE_KEY]
    bars = sorted(({member: s, measure: _num(row.get(s))} for s in series if _num(row.get(s)) is not None),
                  key=lambda r: -(r[measure] or 0.0))
    if len(bars) < 2:
        return None
    roles = chart.get("column_roles") or {}
    fmt = (roles.get(series[0]) or {}).get("format") or ""
    new_roles = {member: {"column": member, "label": str(chart.get("member_label") or member), "role": "dimension",
                          "format": "text"},
                 measure: {"column": measure, "label": str(chart.get("measure_label") or measure), "role": "measure",
                           "format": fmt}}
    return {**chart, "chart_type": "bar", "recommended_type": "bar", "x_key": member, "y_keys": [measure],
            "rows": bars, "other": None, "x_style": "", "column_roles": new_roles,
            "column_formats": {k: v["format"] for k, v in new_roles.items()},
            "renderable_types": ["bar"], "allowed_types": ["bar"], "grouped_by": None, "grouped_measure": None,
            "chart_spec": {"x": {"column": member, "role": "dimension"}, "column_roles": new_roles}}


def _average(rows: list[dict], y: str, n: int = 3) -> list[float | None]:
    out: list[float | None] = []
    for i in range(len(rows)):
        window = [_num(r.get(y)) for r in rows[max(0, i - n + 1):i + 1]]
        out.append(statistics.fmean(window) if i >= n - 1 and all(v is not None for v in window) else None)
    return out


def _span(label: str, lang: str = "en", *, bare: bool = False) -> str:
    """A compared period's label in a sentence: "of Jan–Sep 2026", "from 1 Jan 2026 to 14 Jun 2026"; in French
    "de janv.–sept. 2026", "du 1er janv. 2026 au 14 juin 2026" (with ``bare``, a span with no "of": "la période
    du ... au ...")."""
    if " – " in label:
        start, end = label.split(" – ", 1)
        if lang == "fr":
            return f"{'la période ' if bare else ''}du {start} au {end}"
        return f"{start} to {end}" if bare else f"from {start} to {end}"
    if bare:
        return label
    return f"de {label}" if lang == "fr" else f"of {label}"


def settle(chart: dict[str, Any] | None, *, grain: str | None = None, adds: bool = True,
           things: str = "members") -> dict[str, Any] | None:
    """The chart in the form its data asks for, with the note saying why; None when no chart says more than the
    number (one period only). ``grain``: the periods' unit; ``adds``: the measure adds up across members (an
    "Other" of the rest is a sum); ``things``: what the members are, in the plural ("stores")."""
    if chart is None:
        return None
    kind = str(chart.get("chart_type") or "")
    if kind in ("map", "scatter", "dumbbell") or chart.get("forecast_meta"):
        return chart
    shape = measure(chart)
    chart["shape"] = {"points": shape.points, "gaps": shape.gaps, "largest_share": shape.largest_share,
                      "to_80": shape.to_80, "negatives": shape.negatives, "swing": shape.swing,
                      "direction": shape.direction, "filled": shape.filled}
    one, many, un_fr, many_fr = _grain(chart, grain)
    temporal = _temporal(chart)
    ys = [y for y in chart.get("y_keys") or [] if y != AVERAGE_KEY]
    both_signs = shape.negatives and shape.positives

    if temporal and chart.get("grouped_by") and kind in ("line", "area", "bar"):
        # A line per member.
        if shape.points <= 1:
            bars = _members_of_one_period(chart)
            if bars is not None:
                _say(bars, f"Bars: one {one} only, its {things} side by side.",
                     "Des barres : une seule période, côte à côte.")
            return bars
        if shape.points < FEW_PERIODS and kind != "bar":
            _set(chart, "bar")
            _say(chart, f"Bars: {shape.points} {many} only, too few for lines.",
                 f"Des barres : {shape.points} {many_fr} seulement, trop peu pour des courbes.")
            return chart
        other = chart.get("other")
        lines = len(ys) - (1 if other else 0)
        if other:
            _say(chart, f"Lines: the {lines} largest {things} over {shape.points} {many}, the rest as Other.",
                 f"Des courbes : les {lines} plus grands sur {shape.points} {many_fr}, le reste dans Autres.")
        else:
            _say(chart, f"Lines: {lines} {things} over {shape.points} {many}.",
                 f"Des courbes : {lines} séries sur {shape.points} {many_fr}.")
        return chart

    if temporal:
        if shape.points <= 1:
            return None                   # the number and its change say it all
        compared = chart.get("compare") or {}
        # Side by side only where a period holds both figures: with no data the year before there is nothing to
        # set beside this year's, and the bars would claim a comparison.
        paired = any(_num(r.get(compared.get("prior"))) is not None and _num(r.get(compared.get("current")))
                     is not None for r in chart.get("rows") or [])
        if compared.get("prior") and paired and len(ys) == 2 and not both_signs and shape.points <= COMPARED_BARS \
                and kind in ("line", "area", "bar"):
            # This year's periods against the same ones before: bars side by side, the change in % under them.
            _set(chart, "bar")
            chart["renderable_types"] = ["bar", "line", *[k for k in chart.get("renderable_types") or []
                                                          if k not in ("bar", "line")]]
            compared["strip"] = True
            before, now = compared.get("prior_label") or "", compared.get("current_label") or ""
            avant = compared.get("prior_label_fr") or before
            maintenant = compared.get("current_label_fr") or now
            _say(chart, f"Bars side by side: each {one} {_span(now)} against {_span(before, bare=True)}, the change "
                        "in % under them.",
                 f"Des barres côte à côte : chaque {un_fr} {_span(maintenant, 'fr')} face à "
                 f"{_span(avant, 'fr', bare=True)}, la variation en % dessous.")
            return chart
        if both_signs:
            _set(chart, "bar")
            _offer_without(chart, "area", "pie", "donut")
            _say(chart, f"Bars from zero: some {many} are below zero.",
                 f"Des barres depuis zéro : certains {many_fr} sont sous zéro.")
            return chart
        if shape.gaps and kind in ("line", "area"):
            _set(chart, "bar")
            _say(chart, f"Bars: {shape.gaps} of the {shape.slots} {many} have no data, left empty.",
                 f"Des barres : {shape.gaps} des {shape.slots} {many_fr} n'ont pas de données, laissés vides.")
            return chart
        if shape.points < FEW_PERIODS and kind in ("line", "area"):
            _set(chart, "bar")
            _say(chart, f"Bars: {shape.points} {many} only, too few for a line.",
                 f"Des barres : {shape.points} {many_fr} seulement, trop peu pour une courbe.")
            return chart
        if kind == "line" and len(ys) == 1 and not chart.get("compare") and shape.points >= SWINGS \
                and (shape.swing or 0) >= SWING:
            rows = chart.get("rows") or []
            for r, v in zip(rows, _average(rows, ys[0])):
                r[AVERAGE_KEY] = v
            label, label_fr = f"3-{one} average", f"Moyenne sur 3 {many_fr}"
            chart["y_keys"] = [*ys, AVERAGE_KEY]
            chart["average"] = {"key": AVERAGE_KEY, "label": label, "label_fr": label_fr}
            fmt = ((chart.get("column_roles") or {}).get(ys[0]) or {}).get("format")
            chart.setdefault("column_roles", {})[AVERAGE_KEY] = {"column": AVERAGE_KEY, "label": label,
                                                                  "label_fr": label_fr, "role": "average",
                                                                  "format": fmt}
            chart.setdefault("column_formats", {})[AVERAGE_KEY] = fmt
            _say(chart, f"A line with its 3-{one} average: {shape.points} {many} that swing widely.",
                 f"Une courbe et sa moyenne sur 3 {many_fr} : {shape.points} {many_fr} qui varient fortement.")
            return chart
        if kind in ("line", "area"):
            way = {"rising": ("rising", "en hausse"), "falling": ("falling", "en baisse")}.get(
                shape.direction, ("steady", "stable"))
            _say(chart, f"A line: {shape.points} {many}, {way[0]}.",
                 f"Une courbe : {shape.points} {many_fr}, {way[1]}.")
        elif kind == "bar":
            _say(chart, f"Bars: {shape.points} {many}.", f"Des barres : {shape.points} {many_fr}.")
        return chart

    if chart.get("grouped_by") and kind in ("stacked", "bar"):
        # Two groupings: a grid once most pairs have a value and the colours would be many.
        if (shape.filled or 0) >= GRID_FILLED and len(ys) > GRID_SERIES:
            _set(chart, "heatmap")
            _say(chart, f"A grid: {len(chart.get('rows') or [])} by {len(ys)}, most pairs filled.",
                 f"Une grille : {len(chart.get('rows') or [])} × {len(ys)}, la plupart des cases remplies.")
        else:
            _say(chart, f"Bars split by colour: {len(chart.get('rows') or [])} {things}, {len(ys)} colours.",
                 f"Des barres par couleur : {len(chart.get('rows') or [])} barres, {len(ys)} couleurs.")
        return chart
    if kind == "heatmap":
        _say(chart, f"A grid: {len(chart.get('rows') or [])} by {len(ys)}.",
             f"Une grille : {len(chart.get('rows') or [])} × {len(ys)}.")
        return chart

    # Members: bars, a pie or a donut.
    if both_signs:
        if kind in ("pie", "donut"):
            _set(chart, "bar")
        _offer_without(chart, "pie", "donut", "area")
        _say(chart, f"Bars from zero: some {things} are below zero.",
             "Des barres depuis zéro : certains sont sous zéro.")
        return chart
    share = shape.largest_share
    if share is not None and share >= ONE_MEMBER and len(chart.get("rows") or []) > 1:
        if kind in ("pie", "donut"):
            _set(chart, "bar")
        _offer_without(chart, "pie", "donut")
        _say(chart, f"Bars: {shape.largest} is {_percent(share)} of the total.",
             f"Des barres : {shape.largest} fait {_percent(share)} du total.")
        return chart
    rows = chart.get("rows") or []
    whole = "donut" in (chart.get("renderable_types") or []) or (bool(chart.get("whole")) and adds)
    parts = whole and len(ys) == 1 and not chart.get("other") and 2 <= len(rows) <= MOST_PARTS \
        and share is not None and share < DONUT_LARGEST
    if parts and kind == "bar":       # a pie the question asked for (a share) stays one
        _set(chart, "donut")
        _say(chart, f"A donut: {len(rows)} parts of the whole, the largest {_percent(share)}.",
             f"Un anneau : {len(rows)} parts du total, la plus grande {_percent(share)}.")
        return chart
    if kind == "bar" and len(rows) > MOST_BARS and not chart.get("other") and chart.get("x_order") != "number" \
            and not chart.get("x_pair") and len(ys) >= 1:
        # Past a dozen members that do not add up (an average each): the largest drawn, said; the table keeps all.
        chart["rows"] = sorted(rows, key=lambda r: -abs(_num(r.get(ys[0])) or 0.0))[:TOP_BARS]     # largest first
        chart["chart_warnings"] = [*(chart.get("chart_warnings") or []),
                                   f"Showing the {TOP_BARS} largest of {len(rows)} {things}: every one is in the "
                                   "table."]
        _say(chart, f"Bars: the {TOP_BARS} largest of {len(rows)} {things}.",
             f"Des barres : les {TOP_BARS} plus grands sur {len(rows)}.")
        return chart
    if kind == "bar" and chart.get("other"):
        count = int((chart.get("other") or {}).get("count") or 0)
        drawn = len(rows) - 1
        _say(chart, f"Bars: the {drawn} largest of {drawn + count} {things}, the rest as Other.",
             f"Des barres : les {drawn} plus grands sur {drawn + count}, le reste dans Autres.")
    elif kind == "bar" and chart.get("shown_of"):
        _say(chart, f"Bars: the {len(rows)} largest of {chart['shown_of']} {things}.",
             f"Des barres : les {len(rows)} plus grands sur {chart['shown_of']}.")
    elif kind == "bar":
        _say(chart, f"Bars: {len(rows)} {things}, largest first." if chart.get("x_order") != "number" else
             f"Bars: {len(rows)} {things} in their own order.",
             f"Des barres : {len(rows)} éléments, du plus grand au plus petit." if chart.get("x_order") != "number"
             else f"Des barres : {len(rows)} éléments dans leur ordre.")
    elif kind in ("pie", "donut"):
        _say(chart, f"A {'donut' if kind == 'donut' else 'pie'}: {len(rows)} parts of the whole"
                    + (f", the largest {_percent(share)}." if share is not None else "."),
             f"{'Un anneau' if kind == 'donut' else 'Un camembert'} : {len(rows)} parts du total"
             + (f", la plus grande {_percent(share)}." if share is not None else "."))
    return chart


# ── dashboard tiles: worth a place? ──────────────────────────────────────────


WEAK = {
    "flat": {"en": "The same in every period, so a chart shows nothing a number would not.",
             "fr": "La même valeur à chaque période : un graphique n'apprend rien de plus qu'un chiffre."},
    "one_member": {"en": "Only one {thing} has any, so there is nothing to compare.",
                   "fr": "Un seul élément en a : rien à comparer."},
    "repeat": {"en": "It shows the same as “{other}”.",
               "fr": "Elle montre la même chose que « {other} »."},
}


def _rows_key(payload: dict[str, Any]) -> tuple:
    data = payload.get("data") or {}
    headers = list(data.get("headers") or [])
    return tuple(tuple(str(r.get(h)) for h in headers) for r in data.get("rows") or [])


def weak_tile(kind: str, payload: dict[str, Any], others: list[tuple[str, dict[str, Any]]],
              lang: str = "en") -> str:
    """Why a dashboard tile is not worth its place, or "" when it is: a trend that never moves, a breakdown with one
    member, a tile showing the same rows as another one kept before it."""
    data = payload.get("data") or {}
    rows = data.get("rows") or []
    key = "fr" if lang == "fr" else "en"
    if rows:
        mine = _rows_key(payload)
        for title, other in others:
            if mine and mine == _rows_key(other):
                return WEAK["repeat"][key].format(other=title)
    chart = payload.get("chart") or {}
    if kind == "trend" and chart and _temporal(chart):
        ys = [y for y in chart.get("y_keys") or [] if y != AVERAGE_KEY]
        values = [_num(r.get(y)) for r in chart.get("rows") or [] for y in ys]
        known = [v for v in values if v is not None]
        if len(known) >= 3 and max(known) - min(known) <= 1e-9 + 0.001 * max(abs(v) for v in known):
            return WEAK["flat"][key]
    if kind == "breakdown" and len(rows) == 1 and (data.get("headers") or []):
        labels = data.get("header_labels") or {}
        first = (data.get("headers") or [""])[0]
        thing = str(labels.get(first) or first).lower()
        return WEAK["one_member"][key].format(thing=thing)
    return ""


def tile_score(kind: str, payload: dict[str, Any]) -> float:
    """How much a built tile shows, to keep the best when there are more than fit: rows with a value, members to
    compare, a series that moves."""
    chart = payload.get("chart") or {}
    shape = chart.get("shape") or {}
    rows = len((payload.get("data") or {}).get("rows") or [])
    score = {"number": 3.0, "trend": 2.5, "breakdown": 2.0, "table": 1.0}.get(kind, 1.0)
    score += min(rows, 24) / 24
    if shape.get("swing"):
        score += min(float(shape["swing"]), 1.0) * 0.5
    if shape.get("largest_share") is not None:
        score += (1.0 - float(shape["largest_share"])) * 0.5
    return round(score, 3)
