"""
tests/test_forecast_gate.py

core/forecast.py will fit a line through anything: two points are enough, the
fit is on row index rather than parsed dates, the period column need not be
temporal, and the R-squared it computes is used only to caption the chart.

Live, on real revenue: six flat-ish monthly points, R-squared 0.0964 -- a line
explaining under a tenth of the movement -- projected three months forward under
"Trend: +$43.9K/period". Nothing in the product said the projection was
worthless. These tests are the part that says so.

Every refusal has to name a threshold. "I could not forecast that" is only
useful if it also says what would make it possible.
"""

from __future__ import annotations

import pytest

from core.forecast_gate import (
    MIN_POINTS,
    assess_fit,
    evaluate_forecast_request,
)


def months(values, start=1, year=2026):
    return [{"PERIOD": f"{year}-{start + i:02d}", "REVENUE": v} for i, v in enumerate(values)]


class TestWhatIsRefusedAndWhy:
    def test_a_short_series_names_the_minimum(self):
        decision = evaluate_forecast_request(months([100, 110, 120, 130]))
        assert decision.reason_code == "too_short"
        assert str(MIN_POINTS) in decision.caveat and "4" in decision.caveat

    def test_a_gappy_series_is_refused(self):
        rows = [{"PERIOD": p, "REVENUE": v} for p, v in [
            ("2026-01", 1), ("2026-02", 2), ("2026-08", 3),
            ("2026-09", 4), ("2026-10", 5), ("2026-11", 6),
        ]]
        assert evaluate_forecast_request(rows).reason_code == "irregular_cadence"

    def test_an_unordered_series_is_refused_not_silently_sorted(self):
        """The SQL ORDER BY is the governed artefact. Re-sorting in Python would
        hide a planning defect behind a plausible chart."""
        assert evaluate_forecast_request(
            months([1, 2, 3, 4, 5, 6])[::-1]
        ).reason_code == "unordered_series"

    def test_a_flat_series_is_not_a_trend(self):
        assert evaluate_forecast_request(months([100] * 8)).reason_code == "constant_series"

    def test_a_non_temporal_axis_is_refused(self):
        """Today infer_forecast_cols falls back to text[0], so a customer-name
        column becomes the time axis and the projection is nonsense."""
        rows = [{"CUSTOMER": n, "REVENUE": i} for i, n in enumerate(
            ["Nova", "Maritime", "Pacific", "Atlantic", "Ontario", "Calgary"])]
        assert evaluate_forecast_request(rows).reason_code == "no_temporal_axis"

    def test_a_grouped_result_holds_several_series(self):
        rows = [{"PERIOD": f"2026-0{i % 3 + 1}", "REGION": r, "REVENUE": i}
                for i, r in enumerate(["W", "E", "W", "E", "W", "E"])]
        decision = evaluate_forecast_request(rows)
        assert decision.reason_code == "multi_series"
        assert "REGION" in decision.caveat

    def test_a_truncated_result_is_refused(self):
        """Four sibling modules already refuse rather than compute over a
        prefix. forecast was the one that did not."""
        assert evaluate_forecast_request(
            months(list(range(1, 9))), truncated=True,
        ).reason_code == "truncated_result"

    def test_a_masked_series_is_refused(self):
        """protect_rows runs before analytics, and core/forecast.py drops the
        masked cells silently -- fitting over an unknown, unreported subset."""
        rows = months([1, 2, 3, 4, 5, 6, 7, 8])
        rows[3]["REVENUE"] = "[REDACTED]"
        assert evaluate_forecast_request(rows).reason_code == "masked_series"

    def test_policy_can_block_it_outright(self):
        """A forecast is a derived visual of the same values a chart draws, so
        it inherits the chart aggregate-only decision."""
        assert evaluate_forecast_request(
            months(list(range(1, 9))), policy_allows_derived_visual=False,
        ).reason_code == "policy_blocked"

    def test_every_refusal_says_what_would_make_it_possible(self):
        for rows, kwargs in [
            (months([1, 2, 3]), {}),
            (months([100] * 8), {}),
            (months(list(range(1, 9))), {"truncated": True}),
        ]:
            decision = evaluate_forecast_request(rows, **kwargs)
            assert not decision.allowed
            assert decision.caveat and decision.caveat.startswith("I did not project")


class TestWhatIsAllowed:
    def test_a_clean_monthly_series_is_allowed(self):
        decision = evaluate_forecast_request(months([100 + i * 10 for i in range(8)]))
        assert decision.allowed
        assert decision.grain == "month"
        assert decision.period_col == "PERIOD" and decision.value_col == "REVENUE"

    def test_the_horizon_is_clamped_to_half_the_history(self):
        decision = evaluate_forecast_request(
            months([100 + i * 10 for i in range(8)]), horizon=12,
        )
        assert decision.allowed and decision.horizon == 4
        assert any("rather than 12" in note for note in decision.notes)

    def test_a_clamp_is_a_note_not_a_refusal(self):
        decision = evaluate_forecast_request(
            months([100 + i * 10 for i in range(8)]), horizon=99,
        )
        assert decision.allowed and decision.notes


class TestTheModelLadder:
    """A seasonal SARIMAX estimates seven parameters and needs two full cycles
    plus room. Fitting one to eight monthly points does not find a season -- it
    memorises the noise and projects it forward confidently."""

    @pytest.mark.parametrize("n,model", [(6, "ols"), (8, "ols"), (12, "ets"), (18, "ets")])
    def test_short_and_medium_series(self, n, model):
        rows = [{"PERIOD": f"{2020 + i // 12}-{i % 12 + 1:02d}", "REVENUE": 100 + i}
                for i in range(n)]
        assert evaluate_forecast_request(rows).model == model

    def test_sarimax_only_with_two_full_cycles_and_real_seasonality(self):
        rows = [{"PERIOD": f"{2020 + i // 12}-{i % 12 + 1:02d}",
                 "REVENUE": 100 + (i % 12) * 20 + i} for i in range(28)]
        assert evaluate_forecast_request(rows).model == "sarimax"

    def test_sarimax_is_never_fitted_to_a_short_series(self):
        for n in range(MIN_POINTS, 12):
            rows = [{"PERIOD": f"2026-{i + 1:02d}", "REVENUE": 100 + (i % 4) * 50}
                    for i in range(n)]
            assert evaluate_forecast_request(rows).model != "sarimax"


class TestTheFitIsFinallyConsulted:
    """R-squared has always been computed and never once used as a gate."""

    def _allowed(self):
        return evaluate_forecast_request(months([100 + i * 10 for i in range(9)]))

    def test_a_poor_line_and_a_poor_backtest_refuses(self):
        decision = assess_fit(self._allowed(), 0.0964, 45.0)
        assert decision.reason_code == "poor_fit"
        assert "10%" in decision.caveat and "45%" in decision.caveat

    def test_a_poor_line_with_a_good_backtest_is_allowed(self):
        """Both have to fail. A seasonal series legitimately has poor linear
        R-squared while an ETS or SARIMAX fit predicts it well -- refusing on
        R-squared alone would discard the cases those models exist for."""
        assert assess_fit(self._allowed(), 0.0964, 8.0).allowed

    def test_a_good_line_with_a_poor_backtest_is_allowed(self):
        assert assess_fit(self._allowed(), 0.91, 45.0).allowed

    def test_the_quality_numbers_are_carried_for_the_caller(self):
        decision = assess_fit(self._allowed(), 0.42, 12.0)
        assert decision.fit_quality == {"r2": 0.42, "backtest_mape": 12.0}


class TestItIsActuallyWired:
    """A gate nobody calls is the failure mode this repository keeps hitting."""

    def test_the_pipeline_asks_before_computing(self):
        from pathlib import Path

        pipeline = (Path(__file__).resolve().parents[1] / "core" / "query_pipeline.py").read_text(
            encoding="utf-8",
        )
        idx = pipeline.index('_post_intents.get("forecast")')
        block = pipeline[idx:idx + 2000]
        assert "evaluate_forecast_request" in block
        assert block.index("evaluate_forecast_request") < block.index("compute_forecast(")

    def test_a_refusal_reaches_the_answer(self):
        from pathlib import Path

        renderer = (Path(__file__).resolve().parents[1] / "core" / "result_renderer.py").read_text(
            encoding="utf-8",
        )
        assert 'confidence_context.get("forecast_caveats")' in renderer
        assert "coverage_caveats.extend" in renderer
