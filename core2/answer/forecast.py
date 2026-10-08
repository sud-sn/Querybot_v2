"""What comes next: a measure's series projected forward, with a 95% range.

A forecast question ("forecast sales for the next 6 months", "where will stock be
by year end?") reads the series the way a trend question does, through the same
resolver, compiler and governance, then projects its complete periods forward. A
partial last period is shown but never fitted: a month half over is not a low
month.

* With two full seasons of history (24 months, 8 quarters, 104 weeks, 14 days)
  the projection is a trend line through the seasonally adjusted series plus each
  season's average lift (classical additive decomposition).
* With less, it is the trend line alone, from at least 6 complete periods.
* Fewer than 6 complete periods are not forecast: the answer says how many exist.

The range is the 95% prediction interval of the fitted line (the portal's chart
labels it so) and widens with the horizon. A measure that never went below zero
is not projected below it. Future periods are dated like the past ones and marked
as forecasts; the headline gives the last actual period and the forecast apart,
so a projection is never read as a result. Everything is deterministic: the same
history gives the same forecast.
"""

from __future__ import annotations

import datetime as dt
import math
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from core2.answer.builder import answer_badges, conditions_tail, fmt, frame, scoped_label
from core2.compile.compiler import compile_query
from core2.model.schema import SemanticModel
from core2.plan.ir import Plan, TimeSpec
from core2.resolve.resolver import Context, DaysBetween, ResolveError, resolve
from core2.resolve.time import add_units, label as period_label, unit_start
from core2.warehouse.runner import Warehouse

MIN_PERIODS = 6
DEFAULT_HORIZON = 3
SEASON = {"day": 7, "week": 52, "month": 12, "fiscal_month": 12, "quarter": 4, "fiscal_quarter": 4}
Z95 = 1.959964
_UNIT = {"fiscal_month": "month", "fiscal_quarter": "quarter", "fiscal_year": "year"}
_STYLE = {"day": "iso_date", "week": "iso_date", "month": "month_year_long", "quarter": "quarter", "year": "year"}


@dataclass
class Projection:
    values: list[float]          # the forecast, one per future period
    low: list[float]
    high: list[float]
    method: str                  # "trend" | "trend and seasonality"
    slope: float                 # per period, on the adjusted series
    r2: float


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _day(value: Any) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    return my - slope * mx, slope


def _median(xs: list[float]) -> float:
    ordered = sorted(xs)
    n = len(ordered)
    return (ordered[(n - 1) // 2] + ordered[n // 2]) / 2 if n else 0.0


def _seasonal(values: list[float], season: int) -> list[float]:
    """Each season's typical lift over a centred moving average (medians, so one odd year cannot set it)."""
    n = len(values)
    half = season // 2
    lifts: list[list[float]] = [[] for _ in range(season)]
    for i in range(half, n - half):
        window = values[i - half:i + half + 1]
        if season % 2:
            average = sum(window) / season
        else:   # a 2 x season moving average centres an even season
            average = (sum(window[1:-1]) + (window[0] + window[-1]) / 2) / season
        lifts[i % season].append(values[i] - average)
    typical = [_median(x) if x else 0.0 for x in lifts]
    centre = sum(typical) / season
    return [m - centre for m in typical]


@dataclass
class _Fit:
    intercept: float
    slope: float
    lifts: list[float] | None
    season: int | None

    def at(self, i: int) -> float:
        lift = self.lifts[i % self.season] if self.lifts and self.season else 0.0
        return self.intercept + self.slope * i + lift


def _fit(values: list[float], season: int | None) -> _Fit:
    lifts = _seasonal(values, season) if season and len(values) >= 2 * season else None
    adjusted = [v - lifts[i % season] for i, v in enumerate(values)] if lifts and season else list(values)
    intercept, slope = _ols([float(i) for i in range(len(values))], adjusted)
    return _Fit(intercept, slope, lifts, season if lifts else None)


def unusual(values: list[float], season: int | None) -> list[int]:
    """Periods far outside the fitted pattern (a robust z-score above 3.5): a spike, a bad load."""
    cleaned = list(values)
    flagged: list[int] = []
    for _ in range(2):
        fit = _fit(cleaned, season)
        residuals = [v - fit.at(i) for i, v in enumerate(cleaned)]
        centre = _median(residuals)
        spread = 1.4826 * _median([abs(r - centre) for r in residuals])
        if spread <= 0:
            break
        found = [i for i, r in enumerate(residuals) if abs(r - centre) / spread > 3.5 and i not in flagged]
        if not found:
            break
        for i in found:
            cleaned[i] = fit.at(i)
        flagged += found
    return sorted(flagged)


def project(values: list[float], season: int | None, horizon: int, *, skip: list[int] | None = None) -> Projection:
    """``horizon`` periods after ``values`` (equally spaced, oldest first); ``skip`` periods are refitted, not used."""
    n = len(values)
    cleaned = list(values)
    if skip:
        rough = _fit(values, season)
        for _ in range(2):
            for i in skip:
                cleaned[i] = rough.at(i)
            rough = _fit(cleaned, season)
    fit = _fit(cleaned, season)
    residuals = [v - fit.at(i) for i, v in enumerate(cleaned)]
    used = 2 + ((season - 1) if fit.lifts and season else 0) + len(skip or [])
    dof = max(1, n - used)
    sigma = math.sqrt(sum(r * r for r in residuals) / dof)
    xs = [float(i) for i in range(n)]
    mean_x = sum(xs) / n
    sxx = sum((x - mean_x) ** 2 for x in xs) or 1.0
    my = sum(cleaned) / n
    total = sum((v - my) ** 2 for v in cleaned)
    r2 = 1 - sum(r * r for r in residuals) / total if total else 0.0
    floor = 0.0 if min(values) >= 0 else None
    out, low, high = [], [], []
    for h in range(horizon):
        i = n + h
        point = fit.at(i)
        spread = Z95 * sigma * math.sqrt(1 + 1 / n + (i - mean_x) ** 2 / sxx)
        lo, hi = point - spread, point + spread
        if floor is not None:
            point, lo, hi = max(point, floor), max(lo, floor), max(hi, floor)
        out.append(point)
        low.append(lo)
        high.append(hi)
    return Projection(out, low, high, "trend and seasonality" if fit.lifts else "trend", fit.slope, max(0.0, r2))


MAX_GAP = 2      # periods with no rows inside the history before it is cut


def _steps(start: dt.date, end: dt.date, unit: str) -> int:
    """How many periods from ``start`` to ``end``."""
    if unit == "day":
        return (end - start).days
    n = 0
    while start < end:
        start, n = add_units(start, unit, 1), n + 1
    return n


def answer_forecast(question: str, plan: Plan, *, model: SemanticModel, warehouse: Warehouse, ctx: Context,
                    data_source: str = "", question_id: str = "", model_version: int = 0,
                    started: float | None = None) -> dict[str, Any]:
    started = started if started is not None else time.perf_counter()
    horizon = plan.forecast.periods if plan.forecast else DEFAULT_HORIZON
    grain = plan.time.grain or "month"
    keep = plan.measures[:1]
    derived = [] if keep else plan.derived[:1]
    shown = [d for d in plan.durations if d.measure]
    durations = [] if keep or derived else shown[:1]
    if not keep and not derived and not durations:
        raise ResolveError("unknown", "a forecast needs the measure to project",
                           sorted(m.slug for m in model.measures.values() if not m.hidden)[:12])
    history = plan.model_copy(update={
        "intent": "trend", "measures": keep, "derived": derived, "group_by": [], "via": {}, "sort": [],
        "durations": durations + [d for d in plan.durations if not d.measure],
        "limit": None, "forecast": None, "drivers": None,
        "time": TimeSpec(date=plan.time.date, grain=grain, window=plan.time.window, compare=None)})
    logical = resolve(history, model, ctx)
    compiled = compile_query(logical, model, warehouse.dialect)
    result = warehouse.query(compiled.sql, max_rows=compiled.row_cap)
    at = {name.casefold(): i for i, name in enumerate(result.columns)}
    period_col = next(c for c in compiled.columns if c.role == "period")
    measure_col = next(c for c in compiled.columns if c.role == "measure")
    m = logical.measures[0]
    series: dict[dt.date, float] = {}
    for row in result.rows:
        day = _day(row[at[period_col.name.casefold()]])
        value = _number(row[at[measure_col.name.casefold()]])
        if day is not None and value is not None:
            series[day] = value
    notes = list(logical.notes)
    if len(plan.measures) + len(plan.derived) + len(shown) > 1:
        notes.append("Forecast for the first measure asked about.")
    if plan.group_by:
        notes.append("Forecast for the total; a forecast per group is not available yet.")

    unit = _UNIT.get(grain, grain)
    fiscal = model.settings.fiscal_year_start_month if grain.startswith("fiscal") else None
    partial = set(logical.partial)
    complete = sorted(d for d in series if d not in partial)
    snapshot = bool(m.measure and m.measure.additivity == "semi_additive")
    label_of = lambda d: period_label(d, grain, fiscal_start=logical.fiscal_start)  # noqa: E731
    if complete:
        # The history is the latest unbroken stretch: a series that stops for longer than
        # MAX_GAP periods starts again after it (a long gap is no information, not zeros).
        run = [complete[-1]]
        for day in reversed(complete[:-1]):
            if _steps(day, run[0], unit) > MAX_GAP + 1:
                notes.append(f"Only the data since {label_of(run[0])} is used: before it, the series stops for "
                             f"{_steps(day, run[0], unit) - 1} {unit}s.")
                break
            run.insert(0, day)
        complete = run
    values: list[float] = []
    starts: list[dt.date] = []
    gaps = 0
    if complete:
        day = complete[0]
        last_value = 0.0
        while day <= complete[-1]:
            if day in series and day not in partial:
                last_value = series[day]
            else:
                gaps += 1           # no rows: nothing sold, or a level carried from the last snapshot
                last_value = last_value if snapshot else 0.0
            starts.append(day)
            values.append(last_value)
            day = add_units(day, unit, 1) if unit != "day" else day + dt.timedelta(days=1)
    if gaps:
        notes.append(f"{gaps} period{'s' if gaps > 1 else ''} with no rows "
                     f"{'kept the last level' if snapshot else 'counted as zero'}.")

    fmt_ = m.format
    if fmt_ == "number" and values and all(float(v).is_integer() for v in values):
        fmt_ = "integer"        # a count kept as a number: "about 47 a month", not 46.68
    table_format = {"currency": "currency", "percent": "percentage"}.get(fmt_, "number")
    headers = ["period", "actual", "forecast", "forecast_low", "forecast_high"]
    labels = {"period": {"week": "Week of", "day": "Day"}.get(grain, "Period"), "actual": "Actual",
              "forecast": "Forecast", "forecast_low": "Low (95%)", "forecast_high": "High (95%)"}
    formats = {"period": "text" if grain.startswith("fiscal") else "date", "actual": table_format,
               "forecast": table_format, "forecast_low": table_format, "forecast_high": table_format}
    display = {} if grain.startswith("fiscal") else {"period": {"type": "date", "style": _STYLE.get(grain, "iso_date")}}

    def period_cell(d: dt.date) -> str:
        return label_of(d) if grain.startswith("fiscal") else d.isoformat()

    records = [{"period": period_cell(d), "actual": series.get(d), "forecast": None, "forecast_low": None,
                "forecast_high": None} for d in sorted(series)]
    sqls = compiled.sql
    caveats: list[str] = []
    if partial & set(series):
        named = ", ".join(label_of(d) for d in sorted(partial & set(series)))
        caveats.append(f"{named} is only partly covered by the data: shown, but not used for the forecast.")

    if len(values) < MIN_PERIODS:
        headline = (f"Not enough history to forecast {m.label.lower()}: {len(values)} complete "
                    f"{unit}{'s' if len(values) != 1 else ''} in the data; at least {MIN_PERIODS} are needed.")
        return frame(question, headline=headline, caveats=caveats, chart=None, kpi=None, suggestions=[],
                     headers=headers[:2], labels={k: labels[k] for k in headers[:2]},
                     records=[{k: r[k] for k in headers[:2]} for r in records],
                     formats={k: formats[k] for k in headers[:2]}, display=display, sql=sqls,
                     row_count=len(result.rows), duration_ms=(time.perf_counter() - started) * 1000,
                     data_source=data_source, question_id=question_id, notes=notes, model_version=model_version)

    season = SEASON.get(grain)
    odd = unusual(values, season)
    fit = project(values, season, horizon, skip=odd)
    if odd:
        named = ", ".join(label_of(starts[i]) for i in odd[:4]) + (" and others" if len(odd) > 4 else "")
        caveats.append(f"{named} {'is' if len(odd) == 1 else 'are'} far outside the usual pattern and "
                       "did not shape the forecast.")
    future: list[dt.date] = []
    day = starts[-1]
    for _ in range(horizon):
        day = add_units(day, unit, 1) if unit != "day" else day + dt.timedelta(days=1)
        future.append(unit_start(day, unit, fiscal_start=fiscal) if unit != "day" else day)
    # A partial period already in the data is the first forecast period's place: its actual is shown,
    # and the forecast covers it too, so the reader sees what the whole period is expected to reach.
    for d, value, lo, hi in zip(future, fit.values, fit.low, fit.high):
        cell = period_cell(d)
        existing = next((r for r in records if r["period"] == cell), None)
        if existing is None:
            existing = {"period": cell, "actual": None}
            records.append(existing)
        existing.update({"forecast": round(value, 6), "forecast_low": round(lo, 6), "forecast_high": round(hi, 6)})

    last_actual, last_value = starts[-1], values[-1]
    first_future, last_future = future[0], future[-1]
    total = sum(fit.values)
    span = label_of(first_future) if horizon == 1 else f"{label_of(first_future)} to {label_of(last_future)}"
    per = {"day": "a day", "week": "a week", "month": "a month", "quarter": "a quarter", "year": "a year"}[unit]
    if horizon == 1:
        expected = f"{fmt(fit.values[0], fmt_)} (95% range {fmt(fit.low[0], fmt_)} to {fmt(fit.high[0], fmt_)})"
    elif snapshot or m.format == "percent" or (m.measure and m.measure.additivity == "non_additive") or (
            isinstance(m.expr, DaysBetween) and m.expr.agg != "sum"):     # an average of days does not add up
        expected = (f"from {fmt(fit.values[0], fmt_)} to {fmt(fit.values[-1], fmt_)} "
                    f"(95% range by {label_of(last_future)}: {fmt(fit.low[-1], fmt_)} to {fmt(fit.high[-1], fmt_)})")
    else:
        expected = (f"about {fmt(total / horizon, fmt_)} {per}, {fmt(total, fmt_)} in all (95% range by "
                    f"{label_of(last_future)}: {fmt(fit.low[-1], fmt_)} to {fmt(fit.high[-1], fmt_)})")
    how = ("the trend and the seasonal pattern" if fit.method == "trend and seasonality" else "the trend")
    headline = (f"{scoped_label(logical, m.label)} was {fmt(last_value, fmt_)} in {label_of(last_actual)}, the last complete "
                f"{unit}. Forecast for {span}, from {how} of {len(values)} {unit}s: {expected}.")
    notes.append(f"Forecast from {how} of {len(values)} complete {unit}s ({label_of(starts[0])} to "
                 f"{label_of(starts[-1])}); the range is the 95% prediction interval.")

    # A period with a forecast is drawn as one, a partial period too: its actual is not a whole period's.
    chart_rows = [{"period": r["period"],
                   "value": _number(r["forecast"] if r["forecast"] is not None else r["actual"]),
                   "is_forecast": r["forecast"] is not None,
                   "forecast_low": r["forecast_low"], "forecast_high": r["forecast_high"]} for r in records]
    roles = {"period": {"column": "period", "label": labels["period"], "role": "temporal"},
             "value": {"column": "value", "label": m.label, "role": "measure", "format": table_format}}
    chart = {"title": f"{m.label}: actual and forecast", "chart_type": "forecast", "x_key": "period",
             "y_keys": ["value"], "y_key": "value", "rows": chart_rows,
             "x_style": _STYLE.get(grain, "") if not grain.startswith("fiscal") else "",
             "column_roles": roles, "column_formats": {"period": formats["period"], "value": table_format},
             "renderable_types": ["forecast", "line"], "allowed_types": ["forecast", "line"],
             "recommended_type": "forecast", "chart_spec": {"x": {"column": "period", "role": "temporal"},
                                                            "column_roles": roles},
             "intent": "forecast", "grouped_by": None, "chart_warnings": [],
             "forecast_meta": {"method": fit.method, "slope": round(fit.slope, 6), "r2": round(fit.r2, 4),
                               "horizon": horizon, "interval": 0.95}}
    payload = frame(question, headline=headline, short_value=fmt(fit.values[0], fmt_),
                    comparison=f"forecast for {label_of(first_future)}", caveats=caveats, chart=chart, kpi=None,
                    badges=answer_badges(logical),
                    suggestions=[f"Why did {m.label.lower()} change in {label_of(last_actual)}{conditions_tail(logical)}?",
                                 f"{m.label} by {unit} for the last 12 {unit}s{conditions_tail(logical)}"],
                    headers=headers, labels=labels, records=records, formats=formats, display=display, sql=sqls,
                    row_count=len(result.rows), duration_ms=(time.perf_counter() - started) * 1000,
                    data_source=data_source, question_id=question_id, notes=notes, model_version=model_version)
    payload["forecast_meta"] = chart["forecast_meta"]
    return payload
