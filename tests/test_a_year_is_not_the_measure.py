"""
tests/test_a_year_is_not_the_measure.py

"Net sales by year" reported the average of six calendar years.

An Infor M3 mart stores the period as an INTEGER: "net sales by year" comes back
as IVC_YR beside NET_SLS_AMT, and IVC_YR parses as a number. Every column
classifier in core/response_builder.py and core/insight.py counted it as numeric,
and the FIRST numeric column is the one the narrative describes. Measured:

    6 records — Invoice Yr ranges 2,020 to 2,025, avg 2,022.50.

The average of six years, offered as the finding. A year-over-year row was worse
for being shorter -- one row of integers produced

    1 record — Current Year ranges 2,025 to 2,025, avg 2,025.

And the second half of the defect is the part that cannot be read off those
sentences. Because the year was numeric, ``text_cols`` came back EMPTY: the
result had no label column at all, so it fell through to numeric_table mode and
produced no trend, no ranking and no chart. The year was simultaneously the
measure and the missing axis, and the whole analytical layer was dead for the
shape a distributor asks for most.

Three things had to be true together, so all three are asserted here:

  * a calendar period is not a measure -- name AND values must agree, and a
    measure suffix vetoes both, because ORD_QTY holding [2020, 1500, 3200] would
    otherwise be read as a period and the real measure would be stolen instead;
  * a period moves to the LABEL candidates rather than being dropped, and after
    the text columns, so a result carrying a warehouse name and a year still
    narrates the warehouse;
  * a bare four-digit year is a period LABEL, which no classifier read -- so even
    with the year as the axis, six years still ranked against each other like
    six warehouses instead of trending.

The existing ``is_temporal_result_column`` in the same module deliberately is NOT
reused: it accepts a column whose values merely look like periods when the name
says nothing, which is affordable for a forecast axis and is a wrong number here.
"""

from __future__ import annotations

import pytest

from core.insight import _looks_temporal as insight_looks_temporal
from core.insight import compute_data_brief
from core.response_builder import (
    _looks_temporal as answer_looks_temporal,
    _measure_and_label_cols,
    _numeric_cols,
    _text_cols,
    build_answer,
    summarize_result_context,
)
from core.temporal_columns import (
    is_calendar_period_column,
    is_temporal_result_column,
    labels_are_bare_years,
    period_columns,
)

BY_YEAR = [
    {"IVC_YR": 2020, "NET_SLS_AMT": 41_200_000.0},
    {"IVC_YR": 2021, "NET_SLS_AMT": 48_900_000.0},
    {"IVC_YR": 2022, "NET_SLS_AMT": 55_100_000.0},
    {"IVC_YR": 2023, "NET_SLS_AMT": 61_800_000.0},
    {"IVC_YR": 2024, "NET_SLS_AMT": 66_400_000.0},
    {"IVC_YR": 2025, "NET_SLS_AMT": 71_900_000.0},
]
YEAR_OVER_YEAR = [{
    "CUR_YR": 2025, "PRV_YR": 2024,
    "CUR_NET_SLS": 71_900_000.0, "PRV_NET_SLS": 66_400_000.0,
}]


class TestAPeriodIsRecognisedAndAMeasureIsNot:
    """Name AND values, with a measure suffix vetoing both. Each row below is a
    column an M3 mart actually returns."""

    @pytest.mark.parametrize("column,values", [
        ("IVC_YR", [2020, 2021, 2022, 2023, 2024, 2025]),
        ("CUR_YR", [2025]),
        ("PRV_YR", [2024]),
        ("DMS_YR", [2024, 2025]),
        ("FISCAL_YEAR", [2024, 2025]),
        ("IVC_PRD", [202401, 202402, 202403]),
        ("PRD_DMS_KEY", [202401, 202402]),
        ("DT_DMS_KEY", [20250401, 20250402]),
        ("IVC_MTH", [1, 2, 3, 11, 12]),
        ("IVC_QTR", [1, 2, 3, 4]),
        ("IVC_DT", ["2025-04-01", "2025-05-01"]),
    ])
    def test_the_period_is_a_period(self, column, values):
        assert is_calendar_period_column(column, values) is True

    @pytest.mark.parametrize("column,values,why", [
        ("NET_SLS_AMT", [41_200_000.0, 48_900_000.0], "no period token"),
        ("ORD_QTY", [2020, 1500, 3200], "values look like years, name does not"),
        # The cases the NAME gate exists for, and the only ones that can fail
        # when it is removed: every value is a perfectly plausible year, and the
        # column is still a count. Without the name gate these become the time
        # axis and the real measure is discarded.
        ("INVOICES", [2020, 2021, 2022], "a count that lands in the year range"),
        ("HEADCOUNT", [2020, 2100], "ditto, at the range's edges"),
        ("OPEN_ORDERS", [1995, 2005, 2015], "ditto"),
        ("CUS_NBR", [100234, 100235], "an identifier"),
        ("WHS_NBR", [1, 2, 3], "an identifier"),
        ("UNIT_PRC", [2024.50, 1999.99], "a price that looks like a year"),
        ("YR_TO_DT_AMT", [2024.50, 1999.99], "measure suffix vetoes the token"),
        ("DLV_DAY_CNT", [1, 2, 3], "a count of days, not a day"),
        ("MTH_SLS_TOTAL", [1, 2, 3], "a total, not a month"),
        ("MARGIN_PCT", [12.5, 14.0], "a percentage"),
        ("YR_AVG", [2020, 2021], "an average"),
        ("LEAD_TIME_DAY", [14, 21, 30], "a duration in days"),
        ("AGE_WK", [1, 2, 52], "a duration in weeks"),
        ("FISCAL_YR", ["north", "south"], "name only, values disagree"),
        ("IVC_MTH", [1, 2, 3, 44], "44 is not a month"),
        ("IVC_QTR", [1, 2, 3, 9], "9 is not a quarter"),
        ("IVC_YR", [900001, 900002], "not a year"),
        ("IVC_YR", [], "nothing to corroborate"),
    ])
    def test_the_measure_stays_the_measure(self, column, values, why):
        assert is_calendar_period_column(column, values) is False, why

    def test_a_duration_in_days_is_where_the_forecast_classifier_differs(self):
        """Stated as a difference, not a bug in either: is_temporal_result_column
        accepts values that merely look like periods, which is the right trade for
        refusing a forecast and the wrong one for choosing a measure."""
        assert is_temporal_result_column("ORD_QTY", [2020, 1500, 3200]) is True
        assert is_calendar_period_column("ORD_QTY", [2020, 1500, 3200]) is False

    def test_period_columns_keeps_the_given_order(self):
        rows = [{"IVC_YR": 2024, "IVC_MTH": 1, "NET_SLS_AMT": 1.0}]
        assert period_columns(rows, ["IVC_YR", "NET_SLS_AMT", "IVC_MTH"]) == \
            ["IVC_YR", "IVC_MTH"]

    def test_no_rows_means_no_periods(self):
        assert period_columns([], ["IVC_YR"]) == []


class TestTheSplitMovesThePeriodToTheAxis:

    def test_the_measure_is_the_amount_and_the_label_is_the_year(self):
        numeric = _numeric_cols(BY_YEAR)
        text = _text_cols(BY_YEAR, numeric)
        assert numeric == ["IVC_YR", "NET_SLS_AMT"], "ground truth: both numeric"
        assert text == [], "ground truth: no label column at all"
        measures, labels, periods = _measure_and_label_cols(BY_YEAR, numeric, text)
        assert measures == ["NET_SLS_AMT"]
        assert labels == ["IVC_YR"]
        assert periods == ["IVC_YR"]

    def test_a_text_dimension_still_comes_first(self):
        """A result grouped by warehouse AND year narrates the warehouse; the
        year is the period it repeats over."""
        rows = [{"WHS_DSC": "Montreal", "IVC_YR": 2024, "NET_SLS_AMT": 1.0}]
        numeric = _numeric_cols(rows)
        _measures, labels, _periods = _measure_and_label_cols(
            rows, numeric, _text_cols(rows, numeric))
        assert labels[0] == "WHS_DSC"
        assert "IVC_YR" in labels

    def test_a_result_of_only_periods_is_left_alone(self):
        """There is no measure to find, and taking the axis away too would leave
        nothing to say."""
        rows = [{"IVC_YR": 2024, "IVC_MTH": 1}, {"IVC_YR": 2025, "IVC_MTH": 2}]
        numeric = _numeric_cols(rows)
        measures, labels, periods = _measure_and_label_cols(
            rows, numeric, _text_cols(rows, numeric))
        assert measures == numeric
        assert periods == []
        assert labels == []

    def test_a_result_with_no_period_is_untouched(self):
        rows = [{"WHS_DSC": "Montreal", "NET_SLS_AMT": 1.0}]
        numeric = _numeric_cols(rows)
        text = _text_cols(rows, numeric)
        assert _measure_and_label_cols(rows, numeric, text) == (numeric, text, [])


class TestABareYearIsAPeriodLabel:

    @pytest.mark.parametrize("labels,expected", [
        (["2020", "2021", "2022"], True),
        ([2020, 2021, 2022], True),
        (["1999", "2000"], True),
        (["Depot 2019", "Depot 2020"], False),
        (["Montreal", "Toronto"], False),
        (["2020"], False),
        (["1500", "3200"], False),
        ([], False),
        (["2020", "Montreal"], False),
    ])
    def test_every_label_must_be_a_year(self, labels, expected):
        assert labels_are_bare_years(labels) is expected

    @pytest.mark.parametrize("labels", [
        ["2020", "2021", "2022"], ["2025-04", "2025-05"],
    ])
    def test_both_answer_classifiers_agree(self, labels):
        """Two copies of this classifier exist and the file's own docstring
        records four of them drifting. The year rule lives in one place."""
        assert answer_looks_temporal(labels) is True
        assert insight_looks_temporal(labels) is True

    @pytest.mark.parametrize("labels", [
        ["Montreal", "Toronto"], ["Depot 2019", "Depot 2020"],
    ])
    def test_and_agree_about_what_is_not_a_period(self, labels):
        assert answer_looks_temporal(labels) is False
        assert insight_looks_temporal(labels) is False


class TestTheAnswerDescribesTheSales:

    def test_the_year_series_becomes_a_trend(self):
        ctx = summarize_result_context(BY_YEAR, "net sales by year")
        assert ctx["mode"] == "time_series"
        assert ctx["value_col"] == "NET_SLS_AMT"
        assert ctx["label_col"] == "IVC_YR"
        assert ctx["period_cols"] == ["IVC_YR"]

    def test_and_the_numbers_are_the_sales_not_the_years(self):
        ctx = summarize_result_context(BY_YEAR, "net sales by year")
        assert ctx["min_value"] == 41_200_000.0
        assert ctx["max_value"] == 71_900_000.0
        assert 2020 not in (ctx["min_value"], ctx["max_value"], ctx["avg_value"])

    def test_the_trend_runs_from_the_first_year_to_the_last(self):
        ctx = summarize_result_context(BY_YEAR, "net sales by year")
        assert str(ctx["first_label"]) == "2020"
        assert str(ctx["last_label"]) == "2025"
        assert ctx["first_value"] == 41_200_000.0
        assert ctx["last_value"] == 71_900_000.0
        assert round(ctx["pct_change"], 1) == 74.5

    def test_it_is_chartable(self):
        """numeric_table mode is not chartable, so the defect also cost the
        chart."""
        assert summarize_result_context(BY_YEAR, "net sales by year")["chartable"]

    def test_the_headline_is_about_the_sales(self):
        answer = build_answer(BY_YEAR, "net sales by year")
        text = " ".join(str(value) for value in answer.values() if value)
        assert "2,022" not in text, text
        assert "2,020 to 2,025" not in text, text

    def test_a_year_over_year_row_measures_the_sales(self):
        ctx = summarize_result_context(YEAR_OVER_YEAR, "net sales this year vs last")
        assert ctx["value_col"] in {"CUR_NET_SLS", "PRV_NET_SLS"}
        assert ctx["value_col"] not in {"CUR_YR", "PRV_YR"}
        assert set(ctx["period_cols"]) == {"CUR_YR", "PRV_YR"}
        assert ctx["avg_value"] != 2025


class TestTheBriefMeasuresTheSales:

    def test_the_period_is_typed_as_a_period(self):
        brief = compute_data_brief(BY_YEAR, "net sales by year")
        assert brief["columns"]["IVC_YR"] == "period"
        assert brief["columns"]["NET_SLS_AMT"] == "numeric"

    def test_no_statistics_are_computed_about_calendar_years(self):
        """numeric_summaries feeds the callouts, the outlier detector and the
        narration prompt. A standard deviation of six years is not a finding."""
        brief = compute_data_brief(BY_YEAR, "net sales by year")
        assert list((brief.get("numeric_summaries") or {})) == ["NET_SLS_AMT"]

    def test_the_brief_sees_a_trend(self):
        brief = compute_data_brief(BY_YEAR, "net sales by year")
        assert brief["mode"] == "time_series"
        ts = brief.get("time_series") or {}
        assert ts.get("direction") == "increasing"
        assert ts.get("first_value") == 41_200_000.0
        assert ts.get("last_value") == 71_900_000.0


class TestAnOrdinaryResultIsUnchanged:
    """The classifiers are shared by every answer in the product; a result with
    no integer period must behave exactly as it did."""

    ROWS = [
        {"WHS_DSC": "Montreal", "NET_SLS_AMT": 40_000_000.0},
        {"WHS_DSC": "Toronto", "NET_SLS_AMT": 35_000_000.0},
        {"WHS_DSC": "Calgary", "NET_SLS_AMT": 25_000_000.0},
    ]
    MONTHLY = [
        {"IVC_MTH": "2025-04", "NET_SLS_AMT": 5_000_000.0},
        {"IVC_MTH": "2025-05", "NET_SLS_AMT": 6_000_000.0},
        {"IVC_MTH": "2025-06", "NET_SLS_AMT": 7_000_000.0},
    ]

    def test_a_text_ranking_still_ranks(self):
        ctx = summarize_result_context(self.ROWS, "net sales by warehouse")
        assert ctx["mode"] == "ranking"
        assert ctx["value_col"] == "NET_SLS_AMT"
        assert ctx["label_col"] == "WHS_DSC"
        assert ctx["period_cols"] == []

    def test_a_string_month_series_still_trends(self):
        ctx = summarize_result_context(self.MONTHLY, "net sales by month")
        assert ctx["mode"] == "time_series"
        assert ctx["value_col"] == "NET_SLS_AMT"
        assert ctx["period_cols"] == []

    def test_a_single_scalar_is_untouched(self):
        ctx = summarize_result_context([{"NET_SLS_AMT": 71_900_000.0}], "net sales")
        assert ctx["mode"] == "single_value"
        assert ctx["value"] == 71_900_000.0
