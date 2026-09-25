# -*- coding: utf-8 -*-
"""The chart drawn for an answer is the one the answer's shape calls for.

Found by running a matrix of result shapes a warehouse actually returns through
the real chooser (core/chart_spec.py, then core/chart.py's payload builder).
Each class below is one way it chose wrong:

  * a month key stored as a number (PRD_KEY = 202401) read as an item code, so
    "stock on hand by period" drew a bar per month code under a warning that
    the axis "looks like a technical identifier", instead of a line;
  * measures named the way a mart abbreviates them -- ON_HND_VAL, UNIT_PRC --
    read as identifiers, so a stock-value ranking got no chart at all and a
    price-versus-quantity question lost its price;
  * a stock value of 5000.5 read as the fifth month of the year 5000;
  * a week (2026-W05) or a fiscal period (FY25 P03) read as a category;
  * a pie drawn of margin percentages, whose slices add up to nothing, and a
    pie of two slices;
  * three regions over six months drawn as three overlapping filled areas;
  * a Monday-to-Sunday profile drawn as a trend line;
  * French wording ("évolution mensuelle", "répartition") chose as if the
    question had said nothing;
  * ten warehouses by six months -- a palette too small for ten series --
    drawn as one line zig-zagging between warehouses.

Every test calls the real functions on synthetic rows named the way a mart
names its columns. No customer data.
"""

from __future__ import annotations

from core.chart import build_chart_payload, detect_chart_type
from core.chart_spec import infer_chart_spec


def _spec(rows, question):
    return infer_chart_spec(rows, question=question)


def _role(spec, column):
    return spec["column_roles"][column]["role"]


def _periods(n=12, start=202401):
    return [start + i for i in range(n)]


class TestAPeriodKeyIsTheTimeAxis:

    def test_a_yyyymm_number_is_the_axis_of_a_trend(self):
        rows = [{"PRD_KEY": p, "ON_HND_QTY": 5000 + i * 37} for i, p in enumerate(_periods())]
        spec = _spec(rows, "stock on hand")
        assert _role(spec, "PRD_KEY") == "temporal"
        assert spec["x"]["column"] == "PRD_KEY"
        assert spec["intent"] == "trend"
        assert spec["recommended_type"] in {"area", "line"}
        assert not spec["warnings"]

    def test_a_yyyymm_string_is_the_axis_too(self):
        rows = [{"PRD_KEY": str(p), "ON_HND_QTY": 5000 + i} for i, p in enumerate(_periods())]
        payload = build_chart_payload(rows, detect_chart_type(rows, "stock on hand"),
                                      question="stock on hand")
        assert payload["x_key"] == "PRD_KEY"
        assert payload["chart_type"] in {"area", "line"}
        assert payload["y_keys"] == ["ON_HND_QTY"]

    def test_a_surrogate_key_that_only_looks_numeric_is_not_a_period(self):
        # 900001 parses as digits but is no month of any year: the shared rule's
        # guard, so a key named like a period does not become a time axis.
        rows = [{"PRD_KEY": 900001 + i, "ON_HND_QTY": 50 + i} for i in range(6)]
        assert _role(_spec(rows, "stock on hand"), "PRD_KEY") != "temporal"


class TestAMeasureNamedTheWayAMartNamesIt:

    def test_a_whole_dollar_stock_value_is_a_measure(self):
        rows = [{"ITM_DSC": f"Fitting {i}", "ON_HND_VAL": 1000.0 - i * 90} for i in range(8)]
        spec = _spec(rows, "stock value by product")
        assert _role(spec, "ON_HND_VAL") == "measure"
        assert spec["recommended_type"] == "bar"
        assert [y["column"] for y in spec["y"]] == ["ON_HND_VAL"]

    def test_a_unit_price_is_one_of_the_two_measures_of_a_scatter(self):
        rows = [{"ITM_NO": f"A{i}", "UNIT_PRC": 10.0 + i, "QTY": 100 - i * 3} for i in range(15)]
        spec = _spec(rows, "relationship between price and quantity")
        assert _role(spec, "UNIT_PRC") == "measure"
        assert spec["recommended_type"] == "scatter"
        assert {y["column"] for y in spec["y"]} == {"UNIT_PRC", "QTY"}

    def test_an_account_number_is_not_a_count(self):
        # ACCOUNT_NO carries "count" inside it; the key suffix says what it is.
        rows = [{"ACCOUNT_NO": 100001 + i, "NET_SLS_AMT": 500.0 + i * 7} for i in range(6)]
        spec = _spec(rows, "sales by account")
        assert _role(spec, "ACCOUNT_NO") == "identifier"
        assert [y["column"] for y in spec["y"]] == ["NET_SLS_AMT"]

    def test_a_revenue_or_a_weight_named_by_its_suffix_is_a_measure(self):
        # _REV and _WGT are measure suffixes in the naming convention and in no
        # word list: whole numbers under them read as codes before.
        rows = [{"CUS_NM": f"Customer {i}", "NET_REV": 1200 - i * 90, "SHP_WGT": 40 + i}
                for i in range(6)]
        spec = _spec(rows, "revenue by customer")
        assert _role(spec, "NET_REV") == "measure"
        assert _role(spec, "SHP_WGT") == "measure"
        assert spec["y"][0]["column"] == "NET_REV"

    def test_a_year_to_date_amount_is_an_amount_not_a_date(self):
        rows = [{"RGN_NM": r, "YR_TO_DT_AMT": v}
                for r, v in (("North", 1200.0), ("South", 800.0), ("West", 300.0))]
        spec = _spec(rows, "year to date sales by region")
        assert _role(spec, "YR_TO_DT_AMT") == "measure"
        assert spec["x"]["column"] == "RGN_NM"


class TestADecimalIsNeverADate:

    def test_a_stock_value_with_cents_is_not_a_month(self):
        rows = [{"PRD_KEY": p, "ON_HND_VAL": 5000.5 + i * 37} for i, p in enumerate(_periods())]
        spec = _spec(rows, "stock value by month")
        assert _role(spec, "ON_HND_VAL") == "measure"
        assert spec["x"]["column"] == "PRD_KEY"
        assert spec["recommended_type"] in {"area", "line"}

    def test_a_number_in_the_range_of_years_with_a_fraction_is_not_a_month(self):
        # 2020.5 as text reads "2020, month 5". As a NUMBER it is only ever a
        # quantity -- a weight here, with nothing in its name to say so.
        rows = [{"ITM_DSC": f"Pallet {i}", "WEIGHT": w}
                for i, w in enumerate((2020.5, 2011.5, 2004.5, 1995.5, 1987.5))]
        spec = _spec(rows, "weight by pallet")
        assert _role(spec, "WEIGHT") == "measure"
        assert spec["recommended_type"] == "bar"

    def test_a_decimal_written_as_text_is_not_a_date_either(self):
        rows = [{"ITM_DSC": f"Item {i}", "SCORE": f"{5000.5 + i * 37}"} for i in range(5)]
        assert _role(_spec(rows, "score by item"), "SCORE") == "measure"

    def test_a_dotted_year_and_month_written_as_text_is_still_a_date(self):
        rows = [{"MTH": f"2025.{m:02d}", "NET_SLS_AMT": 100.0 + m} for m in range(1, 7)]
        assert _role(_spec(rows, "net sales"), "MTH") == "temporal"


class TestWeeksAndFiscalPeriodsAreTime:

    def test_an_iso_week_label_is_a_time_axis(self):
        rows = [{"WK": f"2026-W{w:02d}", "NET_SLS_AMT": 100.0 + w} for w in range(1, 11)]
        spec = _spec(rows, "net sales by week")
        assert _role(spec, "WK") == "temporal"
        assert spec["intent"] == "trend"

    def test_a_fiscal_period_label_is_a_time_axis(self):
        rows = [{"FSC_PRD": f"FY25 P{p:02d}", "NET_SLS_AMT": 100.0 + p} for p in range(1, 13)]
        spec = _spec(rows, "net sales by fiscal period")
        assert _role(spec, "FSC_PRD") == "temporal"
        assert spec["recommended_type"] in {"area", "line"}


class TestAPieIsDrawnOnlyOfAWhole:

    MARGIN = [{"RGN_NM": r, "GRS_MRGN_PCT": v}
              for r, v in (("North", 31.2), ("South", 27.5), ("West", 22.1))]

    def test_a_rate_by_region_is_a_bar_without_a_pie_on_offer(self):
        spec = _spec(self.MARGIN, "gross margin percent by region")
        assert spec["recommended_type"] == "bar"
        assert "pie" not in spec["allowed_types"]
        # The reader asked for no pie, so nothing is said about one.
        assert not spec["warnings"]

    def test_a_pie_asked_for_by_name_is_refused_and_the_reader_told_why(self):
        spec = _spec(self.MARGIN, "gross margin percent by region in a pie chart")
        assert spec["recommended_type"] == "bar"
        assert any("add up" in w for w in spec["warnings"])

    def test_shares_that_make_up_the_whole_are_a_pie(self):
        rows = [{"CHNL_NM": c, "SLS_SHARE_PCT": v}
                for c, v in (("Retail", 50.0), ("Online", 30.0), ("Trade", 20.0))]
        spec = _spec(rows, "share of sales by channel")
        assert spec["recommended_type"] == "pie"

    def test_an_amount_split_three_ways_is_a_pie(self):
        rows = [{"CHNL_NM": c, "NET_SLS_AMT": v}
                for c, v in (("Retail", 500.0), ("Online", 300.0), ("Trade", 150.0))]
        assert _spec(rows, "share of net sales by channel")["recommended_type"] == "pie"

    def test_two_slices_are_never_a_pie(self):
        rows = [{"CHNL_NM": c, "NET_SLS_AMT": v} for c, v in (("Retail", 700.0), ("Online", 300.0))]
        spec = _spec(rows, "share of net sales by channel")
        assert spec["recommended_type"] == "bar"
        assert "pie" not in spec["allowed_types"]


class TestSeveralSeriesOverTimeAreLines:

    ROWS = [{"YR_MTH": f"2025-{m:02d}", "RGN_NM": r, "NET_SLS_AMT": 100.0 + m * 3 + j * 20}
            for m in range(1, 7) for j, r in enumerate(("North", "South", "West"))]

    def test_three_regions_over_six_months_are_three_lines(self):
        payload = build_chart_payload(self.ROWS, None, question="net sales by month and region")
        assert payload["chart_type"] == "line"
        assert payload["x_key"] == "YR_MTH"
        assert payload["y_keys"] == ["North", "South", "West"]

    def test_one_series_keeps_its_area(self):
        rows = [{"YR_MTH": f"2025-{m:02d}", "NET_SLS_AMT": 100.0 + m} for m in range(1, 13)]
        assert _spec(rows, "net sales by month")["recommended_type"] == "area"

    def test_an_area_the_reader_asked_for_is_kept(self):
        spec = _spec(self.ROWS, "net sales by month and region as an area chart")
        assert spec["recommended_type"] == "area"


class TestAWeekProfileIsCompared:

    def test_monday_to_sunday_is_bars_in_calendar_order(self):
        days = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
        rows = [{"DAY_NM": d, "NET_SLS_AMT": 100.0 + i} for i, d in enumerate(days)]
        payload = build_chart_payload(rows, None, question="net sales by day")
        assert payload["chart_type"] == "bar"
        assert payload["chart_spec"]["intent"] == "profile"
        assert [r["DAY_NM"] for r in payload["rows"]] == list(days)


class TestFrenchWordingChoosesTheSameChart:

    def test_a_french_monthly_evolution_is_a_trend(self):
        rows = [{"YR_MTH": f"2025-{m:02d}", "RGN_NM": r, "NET_SLS_AMT": 100.0 + m + j}
                for m in range(1, 7) for j, r in enumerate(("Nord", "Sud", "Ouest"))]
        spec = _spec(rows, "évolution des ventes par région")
        assert spec["intent"] == "trend"
        assert spec["x"]["column"] == "YR_MTH"

    def test_a_french_breakdown_is_a_pie(self):
        rows = [{"CHNL_NM": c, "NET_SLS_AMT": v}
                for c, v in (("Détail", 500.0), ("En ligne", 300.0), ("Gros", 150.0))]
        assert _spec(rows, "répartition des ventes par canal")["recommended_type"] == "pie"

    def test_a_french_ranking_asks_for_the_top_of_a_list(self):
        rows = [{"RGN_NM": f"R{i}", "NET_SLS_AMT": 100.0 - i} for i in range(60)]
        assert _spec(rows, "classement des régions par ventes")["intent"] == "ranking"


class TestAGridTurnsToFitThePalette:

    def test_ten_warehouses_by_six_months_are_six_series_of_a_warehouse_axis(self):
        rows = [{"PRD_KEY": p, "WHS_NM": f"W{j:02d}", "ON_HND_QTY": 500 + i * 9 + j * 40}
                for i, p in enumerate(_periods(6)) for j in range(10)]
        payload = build_chart_payload(rows, None, question="stock on hand by warehouse by month")
        assert payload["x_key"] == "WHS_NM"
        assert payload["grouped_by"] == "PRD_KEY"
        assert payload["chart_type"] == "bar"
        assert len(payload["y_keys"]) == 6
        assert len(payload["rows"]) == 10

    def test_two_years_by_twelve_months_are_a_line_per_year(self):
        rows = [{"SLS_YR": y, "SLS_MTH": m, "NET_SLS_AMT": 100.0 + m + (y - 2024) * 5}
                for y in (2024, 2025) for m in range(1, 13)]
        payload = build_chart_payload(rows, None, question="net sales by year and month")
        assert payload["x_key"] == "SLS_MTH"
        assert payload["grouped_by"] == "SLS_YR"
        assert payload["y_keys"] == ["2024", "2025"]
        assert payload["chart_type"] == "line"
