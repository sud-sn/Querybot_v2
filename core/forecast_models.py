"""Fit a series and say how uncertain the projection is.

The only module that touches statsmodels, and it treats it as OPTIONAL. That
follows the pattern ``requirements.txt`` already documents for
``presidio-analyzer``: named, commented, loaded through a lazy guarded
singleton, and degrading to the pure-Python path when absent. statsmodels pulls
scipy, pandas and patsy for one feature, and this repo otherwise has no ML
dependency at all, so the fallback is the supported configuration rather than a
courtesy.

A PROJECTION WITHOUT AN INTERVAL IS A CLAIM, NOT A FORECAST. That is the real
reason the dependency earns its place: "revenue will be 7.6M next month" and
"revenue will be between 6.1M and 9.1M next month" are different statements,
and only the second is honest about a fit that explains a tenth of the
movement. Every model here returns a band, including OLS, which gets a proper
t-based prediction interval computed in pure Python.

Nothing here decides WHETHER to forecast. core/forecast_gate.py does that, and
it runs first.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("querybot.forecast_models")

# None = not tried yet, False = unavailable. Same shape as
# core/masking.py::_get_presidio.
_statsmodels: Any = None

# Two-sided 95% Student t by degrees of freedom. A short series is exactly
# where the normal approximation is worst -- at n=6 the interval would be 20%
# too narrow -- and it is also exactly where this product operates, so the
# table is worth the twelve lines it costs. scipy would give this for free and
# is precisely the dependency being avoided on the pure-Python path.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
}


def _t_value(df: int) -> float:
    if df <= 0:
        return 12.706
    if df in _T95:
        return _T95[df]
    for key in sorted(_T95):
        if df < key:
            return _T95[key]
    return 1.960


@dataclass(frozen=True)
class Fit:
    """A fitted series and its projection."""

    model: str                     # the model that actually ran, after fallback
    predictions: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    r2: float | None = None
    backtest_mape: float | None = None
    fell_back_from: str = ""       # "" when the requested model ran


def _load_statsmodels():
    """Lazy, guarded. Returns (ExponentialSmoothing, SARIMAX) or (None, None)."""
    global _statsmodels
    if _statsmodels is not None:
        return _statsmodels or (None, None)
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        _statsmodels = (ExponentialSmoothing, SARIMAX)
        log.info("forecast: statsmodels available — ETS and SARIMAX enabled")
        return _statsmodels
    except Exception as exc:
        # Not just ImportError: a broken or mismatched binary wheel raises
        # other things, and a forecast must never fail because of that.
        _statsmodels = False
        log.info("forecast: statsmodels unavailable, using OLS — %s", exc)
        return None, None


def _residuals_of(fitted) -> list[float]:
    """statsmodels returns residuals as a numpy array.

    Written first as ``getattr(fitted, "resid", []) or []``, which raises
    "truth value of an array with more than one element is ambiguous" -- inside
    the try block, so both ETS and SARIMAX silently fell back to OLS on every
    call while reporting themselves available. An explicit None check is the
    only safe way to default a value that might be an array.
    """
    resid = getattr(fitted, "resid", None)
    if resid is None:
        return []
    return [float(r) for r in resid]


def _ols_coefficients(
    values: list[float], xs: list[float] | None = None,
) -> tuple[float, float, float]:
    """(slope, intercept, r2). x defaults to 0..n-1."""
    n = len(values)
    xs = list(xs) if xs else [float(i) for i in range(n)]
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    slope = sxy / sxx if sxx else 0.0
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in values)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, values))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot else 0.0
    return slope, intercept, r2


def _fit_ols(values: list[float], horizon: int, xs: list[float] | None = None) -> Fit:
    """Least squares with a real prediction interval.

    The interval widens with distance from the centre of the data, which is the
    whole point: the fourth projected period is genuinely less certain than the
    first, and a constant-width band would hide that.

    ``xs`` carries real elapsed time when the caller could parse the period
    labels. For an evenly spaced series that is an affine relabelling of the row
    index and the projection is identical -- it only changes the answer when the
    spacing is uneven, where using the index would treat a six-month gap and a
    one-month gap as the same step.
    """
    n = len(values)
    xs = list(xs) if xs else [float(i) for i in range(n)]
    slope, intercept, r2 = _ols_coefficients(values, xs)
    mean_x = sum(xs) / n
    sxx = sum((x - mean_x) ** 2 for x in xs) or 1.0
    df = max(n - 2, 1)
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, values)]
    sigma = math.sqrt(sum(r * r for r in residuals) / df)
    t = _t_value(df)

    gaps = [b - a for a, b in zip(xs, xs[1:])]
    step_size = sorted(gaps)[len(gaps) // 2] if gaps else 1.0
    predictions, lower, upper = [], [], []
    for step in range(1, horizon + 1):
        x0 = xs[-1] + step * step_size
        point = intercept + slope * x0
        se = sigma * math.sqrt(1.0 + 1.0 / n + ((x0 - mean_x) ** 2) / sxx)
        predictions.append(point)
        lower.append(point - t * se)
        upper.append(point + t * se)
    return Fit("ols", tuple(predictions), tuple(lower), tuple(upper), r2=r2)


def _fit_ets(values: list[float], horizon: int) -> Fit | None:
    ExponentialSmoothing, _ = _load_statsmodels()
    if ExponentialSmoothing is None:
        return None
    try:
        fitted = ExponentialSmoothing(
            values, trend="add", damped_trend=True, seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True)
        predictions = [float(v) for v in fitted.forecast(horizon)]
        residuals = _residuals_of(fitted)
    except Exception as exc:
        log.info("forecast: ETS did not fit, falling back — %s", exc)
        return None
    if not predictions or any(p != p for p in predictions):
        return None
    lower, upper = _interval_from_residuals(predictions, residuals, len(values))
    return Fit("ets", tuple(predictions), lower, upper, r2=_r2_from_residuals(values, residuals))


def _fit_sarimax(values: list[float], horizon: int, seasonal_period: int) -> Fit | None:
    _, SARIMAX = _load_statsmodels()
    if SARIMAX is None:
        return None
    try:
        fitted = SARIMAX(
            values, order=(1, 1, 1),
            seasonal_order=(1, 1, 1, seasonal_period),
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        forecast = fitted.get_forecast(steps=horizon)
        predictions = [float(v) for v in forecast.predicted_mean]
        # statsmodels gives a real confidence interval here rather than one
        # derived from residual spread -- the best band any of these produce.
        conf = forecast.conf_int(alpha=0.05)
        lower = tuple(float(row[0]) for row in conf)
        upper = tuple(float(row[1]) for row in conf)
        residuals = _residuals_of(fitted)
    except Exception as exc:
        log.info("forecast: SARIMAX did not fit, falling back — %s", exc)
        return None
    if not predictions or any(p != p for p in predictions):
        return None
    return Fit(
        "sarimax", tuple(predictions), lower, upper,
        r2=_r2_from_residuals(values, residuals),
    )


def _r2_from_residuals(values: list[float], residuals: list[float]) -> float | None:
    if not residuals or len(residuals) != len(values):
        return None
    mean_y = sum(values) / len(values)
    ss_tot = sum((y - mean_y) ** 2 for y in values)
    if not ss_tot:
        return None
    return 1.0 - (sum(r * r for r in residuals) / ss_tot)


def _interval_from_residuals(
    predictions: list[float], residuals: list[float], n: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Band from residual spread, widening with the square root of the horizon.

    Cruder than SARIMAX's own confidence interval, and deliberately not omitted
    when it cannot be computed exactly: a projection shown without a band reads
    as more certain than one shown with a wide band, so the failure mode of
    "no interval" is worse than the failure mode of "approximate interval".
    """
    if not residuals:
        spread = abs(predictions[0]) * 0.1 if predictions else 0.0
    else:
        df = max(len(residuals) - 1, 1)
        spread = math.sqrt(sum(r * r for r in residuals) / df)
    t = _t_value(max(n - 2, 1))
    lower = tuple(p - t * spread * math.sqrt(step) for step, p in enumerate(predictions, 1))
    upper = tuple(p + t * spread * math.sqrt(step) for step, p in enumerate(predictions, 1))
    return lower, upper


def _backtest_mape(values: list[float], model: str, seasonal_period: int) -> float | None:
    """Rolling-origin error: hold out the tail, fit on the rest, compare.

    In-sample R-squared says how well a line describes points it was fitted to.
    This says how well the model predicts points it has never seen, which is
    the question a forecast is actually making a claim about.
    """
    holdout = min(3, len(values) // 4)
    if holdout < 1 or len(values) - holdout < 4:
        return None
    train, actual = values[:-holdout], values[-holdout:]
    fit = fit_series(train, holdout, model=model, seasonal_period=seasonal_period,
                     with_backtest=False)
    if not fit or not fit.predictions:
        return None
    pairs = [(a, p) for a, p in zip(actual, fit.predictions) if a]
    if not pairs:
        return None
    return 100.0 * sum(abs((a - p) / a) for a, p in pairs) / len(pairs)


def fit_series(
    values: list[float],
    horizon: int,
    *,
    model: str = "ols",
    seasonal_period: int = 0,
    with_backtest: bool = True,
    xs: list[float] | None = None,
) -> Fit | None:
    """Fit and project, falling back down the ladder rather than failing.

    A missing library, a fit that will not converge, a wheel that imports but
    crashes -- none of those are reasons to refuse a user a forecast when a
    simpler model would answer honestly. They are recorded in ``fell_back_from``
    so the caller can say so.
    """
    values = [float(v) for v in values or []]
    horizon = max(1, int(horizon or 1))
    if len(values) < 2:
        return None

    requested = model
    fit: Fit | None = None
    if model == "sarimax" and seasonal_period >= 2:
        fit = _fit_sarimax(values, horizon, seasonal_period)
        if fit is None:
            model = "ets"
    if fit is None and model == "ets":
        fit = _fit_ets(values, horizon)
        if fit is None:
            model = "ols"
    if fit is None:
        # ETS and SARIMAX are evenly-spaced-index models by construction, so
        # only OLS can honour real elapsed time.
        fit = _fit_ols(values, horizon, xs)

    if fit and requested != fit.model:
        fit = Fit(
            fit.model, fit.predictions, fit.lower, fit.upper,
            r2=fit.r2, backtest_mape=fit.backtest_mape, fell_back_from=requested,
        )
    if fit and with_backtest:
        mape = _backtest_mape(values, fit.model, seasonal_period)
        fit = Fit(
            fit.model, fit.predictions, fit.lower, fit.upper,
            r2=fit.r2, backtest_mape=mape, fell_back_from=fit.fell_back_from,
        )
    return fit


def statsmodels_available() -> bool:
    """For diagnostics and tests; never gates a forecast on its own."""
    return _load_statsmodels()[0] is not None
