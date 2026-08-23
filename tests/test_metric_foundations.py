"""
tests/test_metric_foundations.py

Phase 1 of the agentic-metrics work: the three pieces that Features 1, 2 and 4
all sit on.

  core/temporal_columns.py   one strict "is this really a time axis" test
  core/metric_builder.py     `ratio` mode -- the X-per-Y shape the builder
                             could not express, so every "per" metric had to be
                             hand-written SQL
  core/analytical_request_plan.py   a formula field that was always None

The temporal tests matter more than they look. Five detectors in this codebase
answer "is this column temporal" and they disagree; a chart guessing wrong costs
an odd chart, but a FORECAST guessing wrong produces a confident, plausible,
wrong number about the future. This module is the strict one.
"""

from __future__ import annotations

from datetime import date

import pytest

from core.analytical_request_plan import compile_analytical_request_plan
from core.metric_builder import compile_metric_builder_config
from core.temporal_columns import (
    infer_series_grain,
    is_temporal_result_column,
    parse_period_label,
    seasonal_period_for_grain,
)


class TestOnlyARealTimeAxisIsTemporal:
    def test_a_surrogate_key_named_period_is_not_a_time_axis(self):
        """The trap the looser detectors fall into: CUSTOMER_PERIOD_KEY has a
        date-ish name and integer values, but 900001 is not a month of any year.
        Forecasting off it would project a customer id forward."""
        assert not is_temporal_result_column("CUSTOMER_PERIOD_KEY", [900001, 900002, 900003])

    def test_an_encoded_date_key_with_real_dates_is_a_time_axis(self):
        assert is_temporal_result_column("INVOICE_DT_DMS_KEY", [20260117, 20260218, 20260319])

    @pytest.mark.parametrize("col,values", [
        ("MONTH", ["2026-01", "2026-02"]),
        ("PERIOD", ["Q1 2026", "Q2 2026"]),
        ("INVOICE_DATE", [date(2026, 1, 1), date(2026, 2, 1)]),
        ("FISCAL_YEAR", ["2025", "2026"]),
    ])
    def test_real_periods_are_accepted(self, col, values):
        assert is_temporal_result_column(col, values)

    @pytest.mark.parametrize("col,values", [
        ("CUS_NM", ["Nova Scotia Service", "Maritime Equipment"]),
        ("PROFIT_CENTRE", ["Ontario West", "Calgary Yard"]),
        ("AMOUNT", [1000.0, 2000.0]),
    ])
    def test_business_columns_are_rejected(self, col, values):
        assert not is_temporal_result_column(col, values)

    def test_a_date_named_column_of_business_names_is_rejected(self):
        """A name alone is never enough -- the values have to corroborate."""
        assert not is_temporal_result_column("UPDATE_DAY_OWNER", ["Alice", "Bob"])


class TestPeriodLabelsParse:
    @pytest.mark.parametrize("label,expected", [
        ("2026-03", date(2026, 3, 1)),
        ("2026-03-17", date(2026, 3, 17)),
        (20260317, date(2026, 3, 17)),
        (202603, date(2026, 3, 1)),
        ("Q2 2026", date(2026, 4, 1)),
        ("2026-Q3", date(2026, 7, 1)),
        ("Mar 2026", date(2026, 3, 1)),
        ("March 2026", date(2026, 3, 1)),
        ("2026", date(2026, 1, 1)),
    ])
    def test_every_shape_the_product_emits(self, label, expected):
        assert parse_period_label(label) == expected

    @pytest.mark.parametrize("label", ["Nova Scotia", "", None, "Forecast +1", 900001])
    def test_a_non_period_parses_to_nothing(self, label):
        assert parse_period_label(label) is None


class TestCadenceComesFromTheData:
    @pytest.mark.parametrize("labels,grain,period", [
        (["2026-01", "2026-02", "2026-03", "2026-04"], "month", 12),
        (["Q1 2026", "Q2 2026", "Q3 2026"], "quarter", 4),
        (["2026-01-01", "2026-01-02", "2026-01-03"], "day", 7),
        (["2024", "2025", "2026"], "year", 0),
    ])
    def test_the_observed_gap_names_the_grain(self, labels, grain, period):
        found, consistency = infer_series_grain(labels)
        assert found == grain
        assert consistency == 1.0
        assert seasonal_period_for_grain(found) == period

    def test_a_gappy_series_scores_below_the_consistency_bar(self):
        """Four months with two missing: the modal gap is still a month, but it
        only holds for two of three intervals. The forecast gate refuses below
        90%, so this is what stops a projection over a broken series."""
        _, consistency = infer_series_grain(["2026-01", "2026-02", "2026-06", "2026-07"])
        assert consistency < 0.9

    def test_unparseable_labels_yield_no_grain(self):
        assert infer_series_grain(["Nova Scotia", "Maritime"]) == ("", 0.0)

    def test_a_single_point_has_no_cadence(self):
        assert infer_series_grain(["2026-01"]) == ("", 0.0)


class TestTheRatioMode:
    def test_revenue_per_active_customer_compiles(self):
        compiled = compile_metric_builder_config({
            "enabled": True, "mode": "ratio",
            "numerator": {"aggregation": "SUM", "measure": "IVC_AMT"},
            "denominator": {
                "aggregation": "COUNT_DISTINCT", "measure": "CUS_NO",
                "filters": [{"field": "STATUS_CD", "operator": "equals", "value": "ACTIVE"}],
            },
        })
        assert compiled.formula == (
            "SUM(IVC_AMT) * 1.0 / NULLIF("
            "COUNT(DISTINCT CASE WHEN STATUS_CD = 'ACTIVE' THEN CUS_NO END), 0)"
        )
        assert compiled.required_columns == ["IVC_AMT", "CUS_NO", "STATUS_CD"]

    def test_the_denominator_is_always_null_guarded(self):
        """An empty denominator is an ordinary period, not an error. Without
        NULLIF the user sees a database exception where they should see a
        blank."""
        compiled = compile_metric_builder_config({
            "enabled": True, "mode": "ratio",
            "numerator": {"measure": "AMOUNT"}, "denominator": {"measure": "ORDER_NO"},
        })
        assert "NULLIF(" in compiled.formula

    @pytest.mark.parametrize("injected", [
        "IVC_AMT); DROP TABLE X--",
        "SUM(IVC_AMT)",
        "(SELECT 1)",
        "AMOUNT FROM OTHER_TABLE",
    ])
    def test_sql_text_cannot_reach_the_formula_through_either_side(self, injected):
        """This is why the bot never authors SQL: it fills structured slots, and
        every slot is an identifier or nothing."""
        with pytest.raises(ValueError):
            compile_metric_builder_config({
                "enabled": True, "mode": "ratio",
                "numerator": {"measure": injected}, "denominator": {"measure": "CUS_NO"},
            })

    def test_a_missing_side_is_refused(self):
        with pytest.raises(ValueError):
            compile_metric_builder_config({
                "enabled": True, "mode": "ratio", "numerator": {"measure": "AMOUNT"},
            })

    def test_the_config_round_trips_as_ratio(self):
        import json

        compiled = compile_metric_builder_config({
            "enabled": True, "mode": "ratio",
            "numerator": {"measure": "AMOUNT"}, "denominator": {"measure": "CUS_NO"},
        })
        config = json.loads(compiled.config_json)
        assert config["mode"] == "ratio"
        assert config["numerator"]["measure"] == "AMOUNT"
        assert config["denominator"]["aggregation"] == "COUNT"

    @pytest.mark.parametrize("mode,config", [
        ("aggregate", {"enabled": True, "mode": "aggregate", "aggregation": "SUM", "measure": "AMOUNT"}),
    ])
    def test_the_existing_modes_are_untouched(self, mode, config):
        assert compile_metric_builder_config(config).formula == "SUM(AMOUNT)"


class TestTheCompiledPlanCarriesTheRealFormula:
    def test_a_metric_formula_is_no_longer_always_none(self):
        """`formula` read `m.get("formula") or m.get("sql_formula")`; the column
        is `sql_template`, so every compiled plan described its metrics with a
        null formula."""
        plan = compile_analytical_request_plan(
            "what is net revenue",
            {"fields": [], "joins": [], "source_scope": {"selected_fact": "DW.SALES_FACT"}},
            matched_metrics=[{
                "id": 1, "name": "Net Revenue",
                "sql_template": "SUM(AMOUNT) - SUM(DISCOUNT)",
                "base_table": "DW.SALES_FACT",
            }],
            analytical_intent_plan={"intent": "metric_query"},
        )
        assert plan["metrics"][0]["formula"] == "SUM(AMOUNT) - SUM(DISCOUNT)"


class TestAnExplicitBreakdownBeatsTheWindow:
    """Found live. "What is my revenue by month this year" returned ONE row —
    $44,430,302.60 labelled 2026-01-01 — where it should return one per month.

    requested_temporal_grain read the WINDOW's unit first and returned early, so
    for "this year" it answered "year" and the compiler bucketed by year. The
    explicit "by month" branch sat below that early return and was unreachable
    for every question that also named a period, which is most of them. A window
    and a breakdown are different things: "by month this year" asks for twelve
    numbers over a yearly window.
    """

    @pytest.mark.parametrize("question,grain", [
        ("what is my revenue by month this year", "month"),
        ("revenue by quarter this year", "quarter"),
        ("revenue by day last week", "day"),
        ("monthly revenue this year", "month"),
        ("show revenue by month for the last 2 years", "month"),
    ])
    def test_the_breakdown_wins(self, question, grain):
        from core.contextual_dates import requested_temporal_grain

        assert requested_temporal_grain(question) == grain

    @pytest.mark.parametrize("question,grain", [
        ("what is my revenue this year", "year"),
        ("revenue for the last 6 months", "month"),
        ("revenue for the last 7 days", "day"),
    ])
    def test_the_window_still_answers_when_no_breakdown_is_named(self, question, grain):
        """The window's unit remains the fallback, so nothing that worked before
        this changed starts behaving differently."""
        from core.contextual_dates import requested_temporal_grain

        assert requested_temporal_grain(question) == grain

    def test_a_question_with_neither_has_no_requested_grain(self):
        from core.contextual_dates import requested_temporal_grain

        assert requested_temporal_grain("what is my revenue") == ""
