"""Decide whether a forecast is defensible before one is drawn.

``core.forecast`` will fit a line through anything. It needs two points, it fits
on row index rather than parsed dates, it does not require the period column to
be temporal, it does not check ordering, gaps or truncation, and it computes an
R-squared it then uses only as a chart caption.

The live consequence, on real revenue: six flat-ish monthly points, an
R-squared of 0.0964 -- a line explaining under a tenth of the variance -- and
three months projected forward under the confident heading "Trend:
+$43.9K/period". Nothing in the product said the projection was worthless.

This module is the part that says so. Every refusal names a threshold, because
"I could not forecast that" is only useful if it also says what would make it
possible.

Pure Python, no I/O, no LLM -- same posture as core.anomaly_detection. The
statistical model itself lives in core.forecast_models, which is the only place
that touches statsmodels and degrades to OLS when it is not installed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from core.i18n import grain_label as _grain, t as _t
from core.temporal_columns import (
    infer_series_grain,
    is_temporal_result_column,
    parse_period_label,
    seasonal_period_for_grain,
)

log = logging.getLogger("querybot.forecast_gate")

# Every threshold in one place, because each one is an argument about what
# "defensible" means and they should be arguable together.
MIN_POINTS = 6                 # below this, any fit is describing noise
MIN_CADENCE_CONSISTENCY = 0.9  # share of gaps that must match the modal gap
MAX_GAP_SHARE = 0.1            # share of expected periods that may be missing
MIN_VARIATION_CV = 0.001       # flat to this degree is not a trend, it is a constant
MIN_R2 = 0.5                   # ... but only refuses when the backtest ALSO fails
MAX_BACKTEST_MAPE = 30.0       # per cent
MAX_HORIZON = 12

_MASKED_MARKERS = {"[REDACTED]"}


@dataclass(frozen=True)
class ForecastDecision:
    allowed: bool
    reason_code: str = ""
    caveat: str = ""
    model: str = ""
    period_col: str = ""
    value_col: str = ""
    grain: str = ""
    seasonal_period: int = 0
    horizon: int = 0
    n_points: int = 0
    notes: tuple[str, ...] = ()          # non-fatal: clamped horizon, grain mismatch
    fit_quality: dict[str, Any] = field(default_factory=dict)


def _refuse(code: str, caveat: str, **extra) -> ForecastDecision:
    return ForecastDecision(allowed=False, reason_code=code, caveat=caveat, **extra)


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip().replace(",", "").replace("$", "").replace("%", ""))
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _is_masked(value: Any) -> bool:
    """Whether a cell was replaced by the compliance boundary.

    ``protect_rows`` runs before any analytics, so by the time a forecast sees a
    row the masked cells are already strings or None. ``core.forecast`` drops
    them silently, which means fitting over an unknown, unreported subset of
    the periods -- the one failure mode nobody could see from the answer.
    """
    if value is None:
        return False
    text = str(value).strip()
    return text in _MASKED_MARKERS or bool(
        text and _to_float(value) is None and not text.isspace()
    )


def _expected_periods(first, last, grain: str) -> int:
    days = (last - first).days
    per = {"day": 1, "week": 7, "month": 30.44, "quarter": 91.31, "year": 365.25}.get(grain, 0)
    return int(round(days / per)) + 1 if per else 0


def _mape(actual: list[float], predicted: list[float]) -> float:
    pairs = [(a, p) for a, p in zip(actual, predicted) if a not in (0, None)]
    if not pairs:
        return float("inf")
    return 100.0 * sum(abs((a - p) / a) for a, p in pairs) / len(pairs)


def evaluate_forecast_request(
    rows: list[dict],
    *,
    question: str = "",
    horizon: int = 3,
    truncated: bool = False,
    policy_allows_derived_visual: bool = True,
) -> ForecastDecision:
    """Whether these rows can carry a forecast, and which model should fit it.

    Conditions are evaluated in order and the FIRST failure wins, because the
    first thing wrong with a series is the thing worth telling someone about.
    """
    rows = list(rows or [])

    # 1. Compliance. A forecast is a derived visual of the same values a chart
    #    would draw, so it inherits the chart's aggregate-only decision rather
    #    than inventing a second, weaker rule.
    if not policy_allows_derived_visual:
        return _refuse("policy_blocked", _t("caveat.forecast.policy_blocked"))

    if not rows:
        return _refuse("no_rows", "")

    headers = list(rows[0].keys())

    # 2. Masking. Silently dropping masked cells means fitting over an unknown
    #    subset and reporting the result as if it covered everything.
    period_candidates = [
        column for column in headers
        if is_temporal_result_column(column, [row.get(column) for row in rows])
    ]
    numeric_candidates = [
        column for column in headers
        if column not in period_candidates
        and any(_to_float(row.get(column)) is not None for row in rows)
    ]
    if not period_candidates:
        return _refuse("no_temporal_axis", _t("caveat.forecast.no_temporal_axis"))
    if not numeric_candidates:
        # This refused with an empty caveat, and the pipeline only surfaces a
        # refusal that HAS one -- so a forecast question over a result with no
        # numeric column produced a plain table and total silence about why.
        # Every other refusal here names its reason; this one was written as
        # though it were too obvious to say, which is exactly when a user
        # assumes the product simply ignored them.
        return _refuse("no_measure", _t("caveat.forecast.no_measure"))

    period_col, value_col = period_candidates[0], numeric_candidates[0]

    if any(_is_masked(row.get(value_col)) for row in rows):
        return _refuse(
            "masked_series", _t("caveat.forecast.masked_series"),
            period_col=period_col, value_col=value_col,
        )

    # 3. Truncation. Four sibling modules already refuse rather than compute
    #    over a prefix; forecast was the one that did not.
    if truncated:
        return _refuse(
            "truncated_result", _t("caveat.forecast.truncated_result"),
            period_col=period_col, value_col=value_col,
        )

    # 4. One series, or the projection silently mixes several.
    breakdown_cols = [
        column for column in headers
        if column not in {period_col, value_col}
        and len({str(row.get(column)) for row in rows}) > 1
    ]
    if breakdown_cols:
        return _refuse(
            "multi_series",
            _t("caveat.forecast.multi_series", column=breakdown_cols[0]),
            period_col=period_col, value_col=value_col,
        )

    parsed = [(parse_period_label(row.get(period_col)), _to_float(row.get(value_col)))
              for row in rows]
    points = [(p, v) for p, v in parsed if p is not None and v is not None]
    if len(points) < 2:
        return _refuse(
            "too_short",
            _t("caveat.forecast.too_short", minimum=MIN_POINTS, count=len(points)),
            period_col=period_col, value_col=value_col, n_points=len(points),
        )

    # 5. Order. Refuse rather than silently sorting: the SQL's ORDER BY is the
    #    governed artefact, and re-sorting here would hide a planning defect.
    if any(points[i][0] > points[i + 1][0] for i in range(len(points) - 1)):
        return _refuse(
            "unordered_series", _t("caveat.forecast.unordered_series"),
            period_col=period_col, value_col=value_col, n_points=len(points),
        )

    grain, consistency = infer_series_grain([p for p, _ in points])
    if not grain or consistency < MIN_CADENCE_CONSISTENCY:
        return _refuse(
            "irregular_cadence", _t("caveat.forecast.irregular_cadence"),
            period_col=period_col, value_col=value_col, n_points=len(points),
        )

    expected = _expected_periods(points[0][0], points[-1][0], grain)
    if expected and (expected - len(points)) / expected > MAX_GAP_SHARE:
        missing = expected - len(points)
        return _refuse(
            "gaps_in_series",
            _t("caveat.forecast.gaps_in_series", missing=missing, expected=expected,
               periods=_grain(grain, expected)),
            period_col=period_col, value_col=value_col, grain=grain, n_points=len(points),
        )

    if len(points) < MIN_POINTS:
        return _refuse(
            "too_short",
            _t("caveat.forecast.too_short_grain", count=len(points),
               periods=_grain(grain, len(points)), minimum=MIN_POINTS),
            period_col=period_col, value_col=value_col, grain=grain, n_points=len(points),
        )

    values = [v for _, v in points]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
    stdev = math.sqrt(variance)
    if stdev == 0 or (mean and abs(stdev / mean) < MIN_VARIATION_CV):
        return _refuse(
            "constant_series", _t("caveat.forecast.constant_series"),
            period_col=period_col, value_col=value_col, grain=grain, n_points=len(points),
        )

    notes: list[str] = []
    capped = max(1, min(int(horizon or 3), MAX_HORIZON, math.ceil(len(points) / 2)))
    if capped < int(horizon or 3):
        notes.append(_t(
            "caveat.forecast.capped_horizon", capped=capped,
            periods=_grain(grain, capped), asked=int(horizon),
        ))

    from core.contextual_dates import requested_temporal_grain

    asked_grain = requested_temporal_grain(question or "")
    if asked_grain and asked_grain != grain:
        notes.append(_t(
            "caveat.forecast.grain_mismatch",
            asked_grain=_grain(asked_grain, 1), grain=_grain(grain, 1),
        ))

    seasonal_period = seasonal_period_for_grain(grain)
    model = _select_model(values, len(points), seasonal_period)
    return ForecastDecision(
        allowed=True, model=model, period_col=period_col, value_col=value_col,
        grain=grain, seasonal_period=seasonal_period if model == "sarimax" else 0,
        horizon=capped, n_points=len(points), notes=tuple(notes),
    )


def _autocorrelation(values: list[float], lag: int) -> float:
    if lag <= 0 or len(values) <= lag + 1:
        return 0.0
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    if len(diffs) <= lag:
        return 0.0
    mean = sum(diffs) / len(diffs)
    denominator = sum((d - mean) ** 2 for d in diffs)
    if denominator == 0:
        return 0.0
    numerator = sum(
        (diffs[i] - mean) * (diffs[i - lag] - mean) for i in range(lag, len(diffs))
    )
    return numerator / denominator


def _select_model(values: list[float], n: int, seasonal_period: int) -> str:
    """Pick the simplest model the history can actually support.

    A seasonal SARIMAX estimates seven parameters and needs at least two full
    cycles plus room to breathe; fitting one to eight monthly points does not
    find a seasonal pattern, it memorises the noise and projects it forward
    with great confidence. Hence the ladder, and hence OLS keeping the short
    series it is honest on.
    """
    if n < 12:
        return "ols"
    if (
        seasonal_period >= 2
        and n >= 2 * seasonal_period + 4
        and _autocorrelation(values, seasonal_period) >= 0.3
    ):
        return "sarimax"
    return "ets"


def assess_fit(
    decision: ForecastDecision, r2: float | None, backtest_mape: float | None,
) -> ForecastDecision:
    """Refuse after the fact when the model turned out not to describe the data.

    Both measures must fail. A seasonal series legitimately has poor linear
    R-squared while an ETS or SARIMAX fit predicts it well, so refusing on
    R-squared alone would throw away the cases the seasonal models exist for.

    This is where the 0.0964 goes. R-squared has always been computed here and
    never once consulted.
    """
    quality = {"r2": r2, "backtest_mape": backtest_mape}
    poor_r2 = r2 is not None and r2 < MIN_R2
    poor_backtest = backtest_mape is not None and backtest_mape > MAX_BACKTEST_MAPE
    if poor_r2 and poor_backtest:
        return ForecastDecision(
            allowed=False, reason_code="poor_fit",
            caveat=_t(
                "caveat.forecast.poor_fit",
                r2=f"{max(0.0, (r2 or 0.0)) * 100:.0f}",
                mape=f"{backtest_mape:.0f}",
            ),
            model=decision.model, period_col=decision.period_col,
            value_col=decision.value_col, grain=decision.grain,
            n_points=decision.n_points, notes=decision.notes, fit_quality=quality,
        )
    return ForecastDecision(
        allowed=True, model=decision.model, period_col=decision.period_col,
        value_col=decision.value_col, grain=decision.grain,
        seasonal_period=decision.seasonal_period, horizon=decision.horizon,
        n_points=decision.n_points, notes=decision.notes, fit_quality=quality,
    )
