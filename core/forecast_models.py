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
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    35: 2.030, 40: 2.021, 45: 2.014, 50: 2.009, 60: 2.000,
    80: 1.990, 100: 1.984, 120: 1.980,
}


def _t_value(df: int) -> float:
    """Two-sided 95% t, rounding an unlisted df DOWN to the next table row.

    Rounding UP was the original, and it is the wrong direction: a larger df
    means a smaller t, so every approximated value produced a band narrower
    than 95%. At df=21 it returned t(25)=2.060 where the true value is 2.080.
    Rounding down returns t(20)=2.086 -- marginally conservative, which is the
    right way to be wrong about an interval.
    """
    if df <= 0:
        return 12.706
    if df in _T95:
        return _T95[df]
    below = [k for k in _T95 if k < df]
    if not below:
        return 12.706
    # Above the table, hold the last row (1.980 at df=120) rather than dropping
    # to the normal 1.960. t(200) is 1.972 and t(1000) is 1.962, so returning
    # the normal would be narrower than the truth for every finite df -- the
    # same understatement this function was just fixed for, hiding in the one
    # branch that looked safe. 1.980 is at most 1% wide and never narrow.
    return _T95[max(below)]


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


_MISSPECIFICATION_MARGIN = 1.5


def _floored(sigma: float, floor: float | None) -> float:
    """Never let in-sample residuals set the band when the model predicts worse.

    In-sample residual spread measures how well a model describes points it was
    fitted to. A prediction interval is a claim about points it has not seen,
    and the two only agree when the model is right about the shape of the
    series. Fit a straight line to a series whose LEVEL wanders -- which is what
    revenue does -- and the residuals look small while the projections drift
    away, so a band built from residuals alone is confidently narrow exactly
    when it should not be.

    Measured, nominal 95%, three steps ahead on a random walk with drift:
    OLS covered 0.70 and ETS 0.77 before this floor. On a deterministic trend,
    where the models ARE right about the shape, out-of-sample error matches the
    residual spread and the floor does nothing.

    The floor is the rolling-origin RMSE: what the model actually got wrong on
    points held out from it.
    """
    if floor is None or floor <= 0 or floor != floor:
        return sigma
    # Only when out-of-sample error EXCEEDS in-sample spread by a clear margin.
    # The backtest holds out three points, so its RMSE is a noisy estimate and
    # exceeds the residual spread by chance about half the time; applying it
    # unconditionally widened a correctly-specified fit to 99% coverage, which
    # is not dishonest but is less useful. The margin lets ordinary noise pass
    # and catches the case this exists for -- a model with the wrong shape for
    # the series, where out-of-sample error is not slightly but obviously worse.
    if float(floor) <= sigma * _MISSPECIFICATION_MARGIN:
        return sigma
    return float(floor)


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


def _fit_ols(values: list[float], horizon: int, xs: list[float] | None = None,
             sigma_floor: float | None = None) -> Fit:
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
    sigma = _floored(sigma, sigma_floor)
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


def _fit_ets(values: list[float], horizon: int,
             sigma_floor: float | None = None) -> Fit | None:
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
        params = fitted.params or {}
        alpha = float(params.get("smoothing_level") or 0.0)
        beta = float(params.get("smoothing_trend") or 0.0)
        phi = float(params.get("damping_trend") or 1.0)
    except Exception as exc:
        log.info("forecast: ETS did not fit, falling back — %s", exc)
        return None
    if not predictions or any(p != p for p in predictions):
        return None
    lower, upper = _ets_interval(predictions, residuals, len(values),
                                 alpha, beta, phi, sigma_floor)
    return Fit("ets", tuple(predictions), lower, upper)


def _fit_sarimax(values: list[float], horizon: int, seasonal_period: int,
                 sigma_floor: float | None = None) -> Fit | None:
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
    # statsmodels supplies this interval, so the floor is applied by widening it
    # in proportion rather than by rebuilding it: if the model's own one-step
    # half-width is narrower than what it actually got wrong out of sample,
    # scale every step by the same ratio.
    if sigma_floor and sigma_floor > 0 and predictions:
        half1 = (upper[0] - lower[0]) / 2.0
        implied = half1 / _t_value(max(len(values) - 2, 1))
        if implied > 0 and sigma_floor > implied:
            scale = sigma_floor / implied
            lower = tuple(p - (p - lo) * scale for p, lo in zip(predictions, lower))
            upper = tuple(p + (hi - p) * scale for p, hi in zip(predictions, upper))
    return Fit("sarimax", tuple(predictions), lower, upper)


# alpha, beta, phi, plus the initial level and initial trend. All five are
# estimated by the fit, so they are five degrees of freedom the residuals no
# longer have.
_ETS_PARAM_COUNT = 5


def _ets_interval(
    predictions: list[float], residuals: list[float], n: int,
    alpha: float, beta: float, phi: float, sigma_floor: float | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The h-step prediction interval for ETS(A,Ad,N), measured rather than guessed.

    The first version of this scaled a residual standard deviation by
    sqrt(step), with df = n-1 and the t taken from a different df again. Monte
    Carlo says what that produced: 90% coverage at one step ahead and 98-100% at
    three, from a band labelled 95% in the legend. Both errors were real and
    they pointed opposite ways, so neither showed up in a test that only looked
    at one horizon.

    Two corrections, verified by simulation across n = 12, 14, 24 and h = 1, 3:

      - df is n - 5, not n - 1. The fit estimates alpha, beta, phi and the two
        initial states, and the same df now feeds both the variance and the t.
      - the h-step multiplier is the ETS(A,Ad,N) variance expression, not
        sqrt(h). Forecast error accumulates through the damped trend, and how
        fast depends on the fitted alpha, beta and phi -- sqrt(h) assumes a
        random walk this model is not.

    Measured coverage after the change: 0.941 to 0.968 against a nominal 0.95.
    statsmodels' own ETS interval was measured too and is worse here (0.80-0.90),
    because it also omits parameter-estimation uncertainty at these sample sizes.
    """
    df = max(n - _ETS_PARAM_COUNT, 1)
    if residuals:
        sigma = math.sqrt(sum(r * r for r in residuals) / df)
    else:
        sigma = abs(predictions[0]) * 0.1 if predictions else 0.0
    sigma = _floored(sigma, sigma_floor)
    t = _t_value(df)

    lower, upper = [], []
    for step, point in enumerate(predictions, 1):
        acc = 1.0
        for j in range(1, step):
            if phi < 1.0:
                contrib = alpha + beta * phi * (1.0 - phi ** j) / (1.0 - phi)
            else:
                contrib = alpha + beta * j
            acc += contrib ** 2
        half = t * sigma * math.sqrt(acc)
        lower.append(point - half)
        upper.append(point + half)
    return tuple(lower), tuple(upper)


def _backtest_mape(
    values: list[float], model: str, seasonal_period: int,
    xs: list[float] | None = None,
) -> tuple[float | None, float | None]:
    """Rolling-origin error: hold out the tail, fit on the rest, compare.

    In-sample R-squared says how well a line describes points it was fitted to.
    This says how well the model predicts points it has never seen, which is
    the question a forecast is actually making a claim about, and it is what
    core.forecast_gate.assess_fit judges.

    ``xs`` is sliced alongside the values so the backtest refits on the same
    time axis as the real fit. Without it an unevenly spaced series was scored
    against a row-index refit -- a different model from the one being shipped,
    which makes the reported accuracy describe something the user never sees.
    """
    holdout = min(3, len(values) // 4)
    if holdout < 1 or len(values) - holdout < 4:
        return None, None
    train, actual = values[:-holdout], values[-holdout:]
    train_xs = xs[:-holdout] if xs else None
    fit = fit_series(train, holdout, model=model, seasonal_period=seasonal_period,
                     with_backtest=False, xs=train_xs)
    if not fit or not fit.predictions:
        return None
    # `if a` would silently drop a genuine zero AND treat it as absent; a zero
    # actual has no percentage error to report, which is a different thing from
    # a missing one, so both are excluded but only deliberately.
    errors = [a - p for a, p in zip(actual, fit.predictions)]
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else None
    pairs = [(a, p) for a, p in zip(actual, fit.predictions) if a not in (0, None)]
    if not pairs:
        return None, rmse
    return 100.0 * sum(abs((a - p) / a) for a, p in pairs) / len(pairs), rmse


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

    # The backtest runs BEFORE the real fit, because its error is what floors
    # the band. Without it the interval is built purely from in-sample
    # residuals, which understates the uncertainty of any series the model has
    # the wrong shape for.
    mape = rmse = None
    if with_backtest:
        mape, rmse = _backtest_mape(values, model, seasonal_period, xs)

    requested = model
    fit: Fit | None = None
    if model == "sarimax" and seasonal_period >= 2:
        fit = _fit_sarimax(values, horizon, seasonal_period, rmse)
        if fit is None:
            model = "ets"
    if fit is None and model == "ets":
        fit = _fit_ets(values, horizon, rmse)
        if fit is None:
            model = "ols"
    if fit is None:
        # ETS and SARIMAX are evenly-spaced-index models by construction, so
        # only OLS can honour real elapsed time.
        fit = _fit_ols(values, horizon, xs, rmse)

    if fit is None:
        return None
    # r2 is ALWAYS the straight-line R-squared, whichever model ran.
    #
    # It used to be set only by the OLS path, so a Fit from ETS or SARIMAX
    # carried r2=None -- and `assess_fit(decision, fit.r2, fit.backtest_mape)`
    # is the obvious thing to write, silently disabling half the poor_fit gate
    # (None cannot fail the R-squared condition). Production never hit it,
    # because the pipeline reads the value out of __forecast_meta instead, but
    # two test modules did write exactly that line and passed only because
    # their series happened to route to OLS.
    #
    # This is the same number the chart captions and the same one MIN_R2 was
    # calibrated against: how much of the movement a straight line explains.
    # How well the MODEL predicts is backtest_mape, which is a separate field
    # precisely because it answers a different question.
    _slope, _intercept, trend_r2 = _ols_coefficients(values, xs)
    return Fit(
        fit.model, fit.predictions, fit.lower, fit.upper,
        r2=trend_r2, backtest_mape=mape,
        fell_back_from=requested if requested != fit.model else "",
    )


def statsmodels_available() -> bool:
    """For diagnostics and tests; never gates a forecast on its own."""
    return _load_statsmodels()[0] is not None
