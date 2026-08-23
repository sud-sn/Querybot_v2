"""
tests/test_forecast_models.py

The forecast gate decided WHETHER to project. This is about the projection
itself, and mostly about its interval.

A point estimate with no band is the failure this file exists to prevent. Live,
six flat months of revenue were projected forward as three bare numbers under
"Trend: +$43.9K/period". The projection was not even wrong -- a flat series
really does continue flat, and the backtest error was 0.3%. What was wrong was
that nothing on screen distinguished it from a forecast anyone should act on.

statsmodels is optional, so most of these run on the pure-Python OLS path, which
is the path every deployment has. The coverage test below is the one that would
catch a wrong interval formula: it does not check the arithmetic against a
restatement of itself, it draws hundreds of series from a known model and counts
how often the truth lands inside the band.
"""

from __future__ import annotations

import random

import pytest

from core.forecast import compute_forecast
from core.forecast_gate import assess_fit, evaluate_forecast_request
from core.forecast_models import _t_value, fit_series, statsmodels_available


def months(values, start=1, year=2026):
    return [{"PERIOD": f"{year}-{start + i:02d}", "REVENUE": v} for i, v in enumerate(values)]


class TestTheIntervalIsReal:
    def test_a_projection_always_carries_a_band(self):
        fit = fit_series([100, 112, 119, 131, 138, 152], 3)
        assert len(fit.lower) == len(fit.upper) == len(fit.predictions) == 3
        assert all(lo < p < hi for lo, p, hi in zip(fit.lower, fit.predictions, fit.upper))

    def test_the_band_widens_with_distance(self):
        """The third projected period is genuinely less certain than the first.
        A constant-width band would hide exactly that."""
        fit = fit_series([100, 118, 115, 140, 132, 161], 4)
        widths = [hi - lo for lo, hi in zip(fit.lower, fit.upper)]
        assert widths == sorted(widths)
        assert widths[-1] > widths[0]

    def test_a_noisier_series_gets_a_wider_band(self):
        tight = fit_series([100, 110, 120, 130, 140, 150], 3)
        noisy = fit_series([100, 180, 40, 210, 60, 150], 3)
        assert (noisy.upper[0] - noisy.lower[0]) > (tight.upper[0] - tight.lower[0]) * 3

    @staticmethod
    def _trend(rng, n, slope=4.0, sigma=25.0):
        """A deterministic line plus noise: the models' own assumption."""
        return [100 + slope * i + rng.gauss(0, sigma) for i in range(n)]

    @staticmethod
    def _random_walk(rng, n, drift=4.0, step=25.0):
        """A wandering LEVEL, which is what revenue actually looks like.

        Neither a straight line nor a damped trend is the right shape for this,
        and that is the point: a prediction interval has to be honest about a
        model being wrong, not only about it being noisy.
        """
        out, level = [], 100.0
        for _ in range(n):
            level += drift + rng.gauss(0, step)
            out.append(level)
        return out

    def _coverage(self, *, n, h, model, dgp=None, trials=300, seed=97):
        gen = dgp or self._trend
        rng = random.Random(seed)
        hits = 0
        for _ in range(trials):
            full = gen(rng, n + h)
            series, truth = full[:n], full[n + h - 1]
            fit = fit_series(series, h, model=model)
            if fit.lower[h - 1] <= truth <= fit.upper[h - 1]:
                hits += 1
        return hits / trials

    @pytest.mark.parametrize("n,h", [(12, 1), (12, 3), (18, 1), (18, 3)])
    def test_the_ols_band_covers_the_truth_95_percent_of_the_time(self, n, h):
        """The test that checks a formula against reality rather than against a
        restatement of itself: draw series from a known model, project, and
        count how often the truth lands in the band."""
        cov = self._coverage(n=n, h=h, model="ols")
        assert 0.91 <= cov <= 0.99, f"nominal 95% band covered {cov:.1%}"

    @pytest.mark.parametrize("n,h", [(12, 1), (14, 3), (24, 1), (24, 3)])
    def test_the_ets_band_covers_the_truth_95_percent_of_the_time(self, n, h):
        """ETS had NO coverage test, because the only one called fit_series
        without a model and `model` defaults to "ols". The band it shipped
        covered 88-90% at one step and 98-100% at three -- too narrow and too
        wide at once, from two errors pointing opposite ways, under a legend
        reading "95% interval".

        Both horizons are checked precisely because a single-horizon test is
        what let that through.
        """
        if not statsmodels_available():
            pytest.skip("statsmodels not installed; ETS falls back to OLS")
        cov = self._coverage(n=n, h=h, model="ets")
        assert 0.91 <= cov <= 0.99, f"nominal 95% ETS band covered {cov:.1%}"

    @pytest.mark.parametrize("model,n,h", [
        ("ols", 12, 1), ("ols", 18, 1), ("ets", 12, 1), ("ets", 18, 1),
    ])
    def test_a_wandering_level_is_still_covered_one_step_out(self, model, n, h):
        """A random walk is the wrong shape for both models, and one step ahead
        they still hold, because the out-of-sample floor catches it."""
        if model == "ets" and not statsmodels_available():
            pytest.skip("statsmodels not installed")
        cov = self._coverage(n=n, h=h, model=model, dgp=self._random_walk)
        assert cov >= 0.91, f"covered {cov:.1%}"

    @pytest.mark.parametrize("model,n", [("ols", 12), ("ols", 18), ("ets", 12), ("ets", 18)])
    def test_a_wandering_level_undercovers_further_out_and_this_is_known(self, model, n):
        """A LIMITATION, pinned so it cannot quietly get worse.

        Three steps ahead on a series whose level wanders, the band covers
        roughly 82-86% against a nominal 95%. Neither a straight line nor a
        damped trend can express a stochastic level, and no interval built from
        one of them fully accounts for being the wrong model; a differenced
        ARIMA would, and the gate only reaches for one when it detects
        seasonality.

        Measured before the out-of-sample floor was added: 0.70-0.82. The floor
        is what moved it, and the floor is what this guards. The upper bound is
        deliberately absent -- getting better here is not a failure.
        """
        if model == "ets" and not statsmodels_available():
            pytest.skip("statsmodels not installed")
        cov = self._coverage(n=n, h=3, model=model, dgp=self._random_walk)
        assert cov >= 0.78, (
            f"covered {cov:.1%}; the out-of-sample floor has stopped working"
        )

    def test_the_bound_is_tight_enough_to_catch_the_formulas_it_names(self):
        """A coverage bound is only worth having if a wrong formula fails it.

        The previous bound was 0.90, and the two errors actually shipped landed
        at 0.90 and 0.98 -- inside it. These are the real numbers measured
        before the fix, checked against the bound now in force.
        """
        # 300 trials puts the 95% Monte Carlo interval at roughly +/-0.025, so a
        # 0.91 floor is about three standard errors below nominal: loose enough
        # not to flake, tight enough to fail every number actually shipped.
        for wrong in (0.877, 0.895, 0.905, 1.000):
            assert not (0.91 <= wrong <= 0.99), f"{wrong} would have passed"

    @pytest.mark.parametrize("df,expected", [
        (1, 12.706), (5, 2.571), (10, 2.228), (30, 2.042), (100, 1.984),
    ])
    def test_short_series_use_the_student_t_not_1_96(self, df, expected):
        assert _t_value(df) == expected

    def test_a_very_long_series_sits_just_above_the_normal(self):
        """It holds the last table row rather than returning 1.960. The normal
        is the limit t approaches from ABOVE, so returning it exactly is
        narrower than the truth at every finite df."""
        assert 1.96 <= _t_value(5000) <= 1.99

    @pytest.mark.parametrize("df,true_t", [
        (21, 2.080), (26, 2.056), (33, 2.035), (61, 2.000), (200, 1.972),
    ])
    def test_an_unlisted_df_errs_wide_never_narrow(self, df, true_t):
        """The table interpolates by rounding the degrees of freedom DOWN.

        It rounded UP originally, which is the wrong direction: more df means a
        smaller t, so every approximated value produced a band narrower than
        the 95% it was labelled. df=21 returned t(25)=2.060 against a true
        2.080. An interval may be conservative; it may not overstate itself.
        """
        assert _t_value(df) >= true_t - 1e-9, "band would be narrower than 95%"
        assert _t_value(df) <= true_t * 1.05, "band absurdly over-wide"


class TestTheFallbackLadder:
    """A missing library is never a reason to refuse a user a forecast."""

    def test_an_unavailable_model_degrades_and_records_it(self):
        fit = fit_series([100 + i for i in range(30)], 3, model="sarimax", seasonal_period=12)
        if statsmodels_available():
            assert fit.model == "sarimax" and fit.fell_back_from == ""
        else:
            assert fit.model == "ols" and fit.fell_back_from == "sarimax"
        assert fit.predictions and fit.lower and fit.upper

    def test_ets_runs_when_present_and_degrades_when_not(self):
        """Written first as `assert fit.model in {"ets", "ols"}`, which passes
        whether or not ETS works -- and it did not: a numpy array defaulted with
        `or []` raised inside the try block, so both statsmodels models fell
        back to OLS on every call while reporting themselves available. An
        assertion that accepts either outcome tests nothing."""
        fit = fit_series([100 + i * 3 for i in range(14)], 3, model="ets")
        if statsmodels_available():
            assert fit.model == "ets" and fit.fell_back_from == ""
        else:
            assert fit.model == "ols" and fit.fell_back_from == "ets"
        assert len(fit.predictions) == 3

    def test_a_seasonal_series_is_fitted_seasonally_when_it_can_be(self):
        """The reason the dependency exists. A straight line through a sawtooth
        is wrong in a way no amount of arithmetic fixes."""
        seasonal = [100 + (i % 12) * 20 + i * 2 for i in range(30)]
        truth = [100 + (i % 12) * 20 + i * 2 for i in range(30, 34)]
        fit = fit_series(seasonal, 4, model="sarimax", seasonal_period=12)
        if not statsmodels_available():
            assert fit.model == "ols"
            return
        assert fit.model == "sarimax"
        ols = fit_series(seasonal, 4, model="ols")
        err = sum(abs(p - t) for p, t in zip(fit.predictions, truth))
        ols_err = sum(abs(p - t) for p, t in zip(ols.predictions, truth))
        assert err < ols_err / 2, f"sarimax {err:.0f} vs ols {ols_err:.0f}"

    def test_a_broken_import_does_not_raise(self, monkeypatch):
        import sys

        import core.forecast_models as fm

        monkeypatch.setattr(fm, "_statsmodels", None)
        monkeypatch.setitem(sys.modules, "statsmodels.tsa.holtwinters", None)
        fit = fit_series([1, 4, 9, 16, 25, 36], 2, model="ets")
        assert fit is not None and fit.model == "ols"

    def test_a_series_too_short_to_fit_returns_nothing(self):
        assert fit_series([5], 3) is None
        assert fit_series([], 3) is None


class TestTheBacktestIsOutOfSample:
    def test_a_clean_trend_backtests_well(self):
        fit = fit_series([100 + i * 10 for i in range(12)], 3)
        assert fit.backtest_mape is not None and fit.backtest_mape < 5.0

    def test_noise_backtests_badly(self):
        fit = fit_series([100, 900, 150, 40, 800, 90, 700, 30, 850, 60], 3)
        assert fit.backtest_mape is not None and fit.backtest_mape > 30.0

    def test_a_series_too_short_to_hold_anything_out_reports_nothing(self):
        assert fit_series([1, 2, 3, 4], 2).backtest_mape is None


class TestComputeForecastEmitsTheBand:
    def test_projected_rows_carry_low_and_high_and_historical_rows_do_not(self):
        rows = compute_forecast(months([100, 118, 115, 140, 132, 161]), "PERIOD", "REVENUE", 3)
        historical = [r for r in rows if not r["is_forecast"]]
        projected = [r for r in rows if r["is_forecast"]]
        assert len(historical) == 6 and len(projected) == 3
        assert all(r["forecast_low"] is None and r["forecast_high"] is None for r in historical)
        for r in projected:
            assert r["forecast_low"] < r["forecast_value"] < r["forecast_high"]

    def test_the_fit_metadata_travels_with_the_result(self):
        rows = compute_forecast(months([100, 118, 115, 140, 132, 161]), "PERIOD", "REVENUE", 3)
        meta = rows[0]["__forecast_meta"]
        assert meta["model"] == "ols"
        assert meta["n_points"] == 6 and meta["horizon"] == 3
        assert meta["interval_confidence"] == 0.95
        assert meta["r2"] is not None and meta["backtest_mape"] is not None

    def test_the_legacy_row_zero_keys_survive_for_one_release(self):
        """portal_chat.html read rows[0].__trend_slope. A tab opened before this
        shipped must keep captioning its chart."""
        rows = compute_forecast(months([100, 110, 120, 130, 140, 150]), "PERIOD", "REVENUE", 2)
        assert rows[0]["__trend_slope"] == pytest.approx(10.0)
        assert rows[0]["__trend_r2"] == pytest.approx(1.0)

    def test_an_unfittable_series_still_returns_annotated_rows(self):
        rows = compute_forecast([{"PERIOD": "2026-01", "REVENUE": 5}], "PERIOD", "REVENUE", 3)
        assert len(rows) == 1 and rows[0]["is_forecast"] is False


class TestItFitsOnDatesNotRowNumbers:
    """Row index treats a six-month gap and a one-month gap as the same step.

    The gate refuses uneven cadence, so this only changes the answer on the
    ungated path -- a direct caller, or a test. It is still the difference
    between a projection and a coincidence there.
    """

    def test_an_uneven_series_is_projected_from_elapsed_time(self):
        rows = [{"PERIOD": p, "REVENUE": v} for p, v in [
            ("2026-01", 100), ("2026-02", 110), ("2026-03", 120),
            ("2026-09", 180), ("2026-10", 190), ("2026-11", 200),
        ]]
        out = compute_forecast(rows, "PERIOD", "REVENUE", 1)
        assert out[0]["__forecast_meta"]["fitted_on"] == "period_dates"
        # 10 per month over the real calendar, not 20 per row.
        assert out[0]["__trend_slope"] == pytest.approx(10.0, abs=0.5)

    def test_an_evenly_spaced_series_is_unchanged_by_the_switch(self):
        """The safety property for the gated path, where cadence is already
        guaranteed regular: with no gaps the calendar index IS the row index, so
        both paths must produce the same projection to the last decimal.

        Checked by running both, not against expected numbers -- a hand-written
        constant here would only prove I could do the arithmetic twice.
        """
        values = [100, 118, 115, 140, 132, 161]
        by_date = compute_forecast(months(values), "PERIOD", "REVENUE", 3)
        by_index = compute_forecast(
            [{"PERIOD": f"row{i}", "REVENUE": v} for i, v in enumerate(values)],
            "PERIOD", "REVENUE", 3,
        )
        assert by_date[0]["__forecast_meta"]["fitted_on"] == "period_dates"
        assert by_index[0]["__forecast_meta"]["fitted_on"] == "row_index"
        assert ([r["forecast_value"] for r in by_date if r["is_forecast"]]
                == [r["forecast_value"] for r in by_index if r["is_forecast"]])
        assert ([r["forecast_low"] for r in by_date if r["is_forecast"]]
                == [r["forecast_low"] for r in by_index if r["is_forecast"]])
        assert by_date[0]["__trend_slope"] == by_index[0]["__trend_slope"]

    def test_unparseable_labels_fall_back_to_row_index(self):
        rows = [{"PERIOD": n, "REVENUE": 100 + i * 10}
                for i, n in enumerate(["a", "b", "c", "d", "e", "f"])]
        out = compute_forecast(rows, "PERIOD", "REVENUE", 2)
        assert out[0]["__forecast_meta"]["fitted_on"] == "row_index"
        assert out[0]["__trend_slope"] == pytest.approx(10.0)


class TestTheGateAndTheModelAgree:
    def test_a_flat_but_predictable_series_is_allowed(self):
        """The live case. R-squared 0.08 and a backtest under 1%: the trend is
        meaningless but the projection is accurate, and poor_fit needs both to
        fail. Refusing here would refuse a correct answer."""
        values = [7489655, 7280000, 7550000, 7400000, 7520000, 7480000]
        decision = evaluate_forecast_request(months(values))
        assert decision.allowed
        fit = fit_series(values, decision.horizon, model=decision.model)
        assert fit.r2 < 0.5 and fit.backtest_mape < 30.0
        assert assess_fit(decision, fit.r2, fit.backtest_mape).allowed

    def test_a_series_that_is_both_unexplained_and_unpredictable_is_refused(self):
        values = [100, 900, 150, 40, 800, 90, 700, 30, 850, 60]
        decision = evaluate_forecast_request(months(values))
        fit = fit_series(values, decision.horizon, model=decision.model)
        refused = assess_fit(decision, fit.r2, fit.backtest_mape)
        assert not refused.allowed and refused.reason_code == "poor_fit"

    def test_the_gates_chosen_model_is_what_gets_fitted(self):
        rows = months([100 + i * 5 for i in range(14)])
        decision = evaluate_forecast_request(rows)
        out = compute_forecast(rows, decision.period_col, decision.value_col,
                               decision.horizon, model=decision.model,
                               seasonal_period=decision.seasonal_period)
        meta = out[0]["__forecast_meta"]
        assert decision.model == "ets"
        assert meta["model"] == ("ets" if statsmodels_available() else "ols")


class TestThePayloadTheBrowserReceives:
    """The band is drawn from payload.rows and payload.forecast_meta. If either
    contract slips, the chart silently loses its interval."""

    def test_the_chart_payload_carries_the_band_and_the_meta(self):
        from core.chart import build_chart_payload

        rows = compute_forecast(months([100, 118, 115, 140, 132, 161]),
                                "PERIOD", "REVENUE", 3)
        payload = build_chart_payload(rows, chart_type="forecast", title="Revenue")
        assert payload["chart_type"] == "forecast"
        assert payload["forecast_meta"]["model"] == "ols"
        projected = [r for r in payload["rows"] if r.get("is_forecast")]
        assert projected and all(
            r["forecast_low"] is not None and r["forecast_high"] is not None
            for r in projected
        )

    def test_the_meta_block_is_not_left_sitting_in_a_row(self):
        from core.chart import build_chart_payload

        rows = compute_forecast(months([100, 118, 115, 140, 132, 161]),
                                "PERIOD", "REVENUE", 3)
        payload = build_chart_payload(rows, chart_type="forecast", title="Revenue")
        assert all("__forecast_meta" not in r for r in payload["rows"])

    def test_the_payload_is_json_serialisable(self):
        """A Decimal from the driver reached the wire once already and killed
        every forecast answer after the SQL had run."""
        import json
        from decimal import Decimal

        from core.chart import build_chart_payload

        rows = compute_forecast(
            [{"PERIOD": f"2026-0{i + 1}", "REVENUE": Decimal(str(100 + i * 9))}
             for i in range(6)],
            "PERIOD", "REVENUE", 3,
        )
        json.dumps(build_chart_payload(rows, chart_type="forecast", title="Revenue"))
