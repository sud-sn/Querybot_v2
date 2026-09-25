# -*- coding: utf-8 -*-
"""Two shapes a single chart could not draw honestly.

Measures in different units. "Net sales and gross margin % by region" drew the
margins on the sales axis: on a scale of a million dollars, 31% is a bar no one
can see. A second y-axis is no fix -- two scales on one plot invent a
correlation that is not in the data. Small multiples are: a panel per unit, the
categories shared, stacked for columns and lines and side by side for a
horizontal ranking.

A grid too wide for a series. Fifteen warehouses by twelve months cannot be
drawn as series in an eight-colour palette either way round, and before this
it was left to the table. A heatmap draws every cell: a row per warehouse, a
column per month, the measure as depth of one hue.

Every test calls the real chooser and payload builder, and executes the real
renderer for what is drawn. Synthetic rows named the way a mart names them.
"""

from __future__ import annotations

import json

import pytest

from core.chart import build_chart_payload
from core.chart_spec import infer_chart_spec
from core.schema_enrichment import display_label

REGIONS = ("North", "South", "West", "East")


def margin_rows():
    return [{"RGN_NM": r, "NET_SLS_AMT": s, "GRS_MRGN_PCT": m}
            for r, s, m in zip(REGIONS, (1_200_000.0, 800_000.0, 300_000.0, 650_000.0),
                               (31.2, 27.5, 22.1, 29.0))]


def three_unit_rows():
    # Money in millions, a price in single dollars, a quantity in thousands.
    return [{"RGN_NM": r, "NET_SLS_AMT": s, "AVG_UNIT_PRC_AMT": p, "ORD_QTY": q}
            for r, s, p, q in zip(REGIONS, (1_200_000.0, 800_000.0, 300_000.0, 650_000.0),
                                  (4.1, 3.9, 4.4, 4.0), (5200, 4100, 1300, 2900))]


class TestMeasuresInDifferentUnitsGetAPanelEach:

    def test_money_and_a_rate_are_two_panels(self):
        spec = infer_chart_spec(margin_rows(), question="net sales and gross margin percent by region")
        assert spec["facets"] == [["NET_SLS_AMT"], ["GRS_MRGN_PCT"]]

    def test_two_amounts_of_a_size_share_one_axis(self):
        rows = [{"RGN_NM": r, "NET_SLS_AMT": s, "COGS_AMT": s * 0.6}
                for r, s in zip(REGIONS, (1200.0, 800.0, 300.0, 650.0))]
        assert infer_chart_spec(rows, question="net sales and cost by region")["facets"] is None

    def test_two_amounts_fifty_times_apart_are_split(self):
        # A unit price beside a revenue total: one axis draws the price as zero.
        rows = [{"ITM_DSC": f"Item {i}", "NET_SLS_AMT": 90_000.0 + i * 1000, "AVG_UNIT_CST": 12.0 + i}
                for i in range(5)]
        spec = infer_chart_spec(rows, question="net sales and unit cost by item")
        assert spec["facets"] is not None and len(spec["facets"]) == 2

    def test_a_measure_at_zero_throughout_shares_the_axis(self):
        rows = [{"RGN_NM": r, "NET_SLS_AMT": s, "RTN_AMT": 0.0}
                for r, s in zip(REGIONS, (1200.0, 800.0, 300.0, 650.0))]
        assert infer_chart_spec(rows, question="net sales and returns by region")["facets"] is None

    def test_three_panels_at_most_and_the_measure_left_out_is_named(self):
        rows = [{"RGN_NM": r, "NET_SLS_AMT": s, "ORD_QTY": q, "ORD_LN_CNT": n, "AVG_UNIT_PRC_AMT": p}
                for r, s, q, n, p in zip(REGIONS, (1_200_000.0, 800_000.0, 300_000.0, 650_000.0),
                                         (5200, 4100, 1300, 2900), (21, 17, 6, 12), (4.1, 3.9, 4.4, 4.0))]
        question = "net sales, quantity, order lines and unit price by region"
        spec = infer_chart_spec(rows, question=question)
        # Four units: money in millions, a price in single dollars, a quantity
        # in thousands and a count of order lines 250 times smaller -- one
        # panel more than a card holds.
        assert len(spec["facets"]) == 3
        drawn = [col for group in spec["facets"] for col in group]
        (left_out,) = {"NET_SLS_AMT", "ORD_QTY", "ORD_LN_CNT", "AVG_UNIT_PRC_AMT"} - set(drawn)
        assert any(display_label(left_out) in warning for warning in spec["warnings"])
        assert build_chart_payload(rows, None, question=question)["y_keys"] == drawn

    def test_a_grouped_chart_has_one_measure_and_no_panels(self):
        rows = [{"YR_MTH": f"2025-{m:02d}", "RGN_NM": r, "NET_SLS_AMT": 100.0 + m}
                for m in range(1, 7) for r in REGIONS[:3]]
        assert infer_chart_spec(rows, question="net sales by month and region")["facets"] is None

    def test_the_payload_carries_the_panels(self):
        built = build_chart_payload(margin_rows(), None, question="net sales and gross margin percent by region")
        assert built["facets"] == [["NET_SLS_AMT"], ["GRS_MRGN_PCT"]]
        assert built["y_keys"] == ["NET_SLS_AMT", "GRS_MRGN_PCT"]


dukpy = pytest.importorskip("dukpy", reason="the renderer is JavaScript; only executing it shows what it draws")
from tests.test_chart_annotation_language import PAGES, _build, _run  # noqa: E402


def drawn(payload, expression, page="portal_chat.html"):
    return json.loads(_build(page, "en", payload, f"JSON.stringify({expression})"))


def rendered(payload, expression, width, height, grow=True):
    """`expression` over the option render() hands ECharts in a card of that size."""
    script = f"""
var drawnOption = null;
window.echarts = {{getInstanceByDom: function () {{ return null; }},
                  init: function () {{ return {{setOption: function (o) {{ drawnOption = o; }},
                                              resize: function () {{}}, dispose: function () {{}}}}; }}}};
var el = {{clientHeight: {height}, clientWidth: {width}, style: {{}}}};
QBCharts.render(el, {json.dumps(payload)}, {{grow: {json.dumps(grow)}}});
var opt = drawnOption;
JSON.stringify({expression});
"""
    return json.loads(_run("portal_chat.html", "en", script))


def grown(payload, clientHeight, grow=True):
    """The height render() gives a card of `clientHeight`, or None if unchanged."""
    script = f"""
window.echarts = {{getInstanceByDom: function () {{ return null; }},
                  init: function () {{ return {{setOption: function () {{}}, resize: function () {{}},
                                              dispose: function () {{}}}}; }}}};
var el = {{clientHeight: {clientHeight}, clientWidth: 640, style: {{}}}};
QBCharts.render(el, {json.dumps(payload)}, {{grow: {json.dumps(grow)}}});
JSON.stringify(el.style.height || null);
"""
    return json.loads(_run("portal_chat.html", "en", script))


class TestThePanelsAreDrawn:

    def payload(self, rows=None, question="net sales and gross margin percent by region"):
        return build_chart_payload(rows or margin_rows(), None, question=question)

    @pytest.mark.parametrize("page", PAGES)
    def test_each_unit_has_its_own_panel_and_axis(self, page):
        option = drawn(self.payload(), "{grids: opt.grid.length, y: opt.yAxis.map(function (a) { return a.name }),"
                       " placed: opt.series.map(function (s) { return [s.name, s.xAxisIndex, s.yAxisIndex] })}", page)
        assert option["grids"] == 2
        assert option["y"] == ["Net Sls Amount", "Gross Mrgn Percent"]
        assert option["placed"] == [["NET_SLS_AMT", 0, 0], ["GRS_MRGN_PCT", 1, 1]]

    def test_each_panel_reads_in_its_own_unit(self):
        payload = self.payload()
        assert _build("portal_chat.html", "en", payload, "opt.yAxis[0].axisLabel.formatter(1000000)") == "$1M"
        assert _build("portal_chat.html", "en", payload, "opt.yAxis[1].axisLabel.formatter(30)").endswith("%")

    def test_the_categories_are_named_once_under_the_last_panel(self):
        shown = drawn(self.payload(), "opt.xAxis.map(function (a) { return a.axisLabel.show })")
        assert shown == [False, True]

    def test_the_panels_share_one_pointer_and_one_tooltip(self):
        option = drawn(self.payload(), "{link: opt.axisPointer.link, trigger: opt.tooltip.trigger}")
        assert option == {"link": [{"xAxisIndex": "all"}], "trigger": "axis"}

    def test_a_panel_of_one_series_needs_no_legend(self):
        # Its title names it.
        assert drawn(self.payload(), "opt.legend === undefined") is True

    def test_a_category_sits_at_the_same_place_in_every_panel(self):
        # The value labels are drawn in one fixed margin, not measured into
        # each panel: "$1.2M" and "30%" would otherwise start the plots at
        # different places and North's bars would not line up.
        grids = drawn(self.payload(), "opt.grid")
        assert {g["left"] for g in grids} == {64} and not any(g["containLabel"] for g in grids)
        assert len({g["height"] for g in grids}) == 1

    @pytest.mark.parametrize("measures", [2, 3])
    def test_the_panels_fit_the_chart_with_room_for_the_names_under_them(self, measures):
        rows = three_unit_rows()
        if measures == 2:
            rows = [{k: v for k, v in r.items() if k != "ORD_QTY"} for r in rows]
        grids = drawn(self.payload(rows, "net sales, unit price and quantity by region"), "opt.grid")
        assert len(grids) == measures
        spans = [(float(g["top"].rstrip("%")), float(g["top"].rstrip("%")) + float(g["height"].rstrip("%")))
                 for g in grids]
        # One below the other, apart, and the last ending clear of the bottom
        # edge, where its category names are written.
        assert all(below[0] > above[1] for above, below in zip(spans, spans[1:]))
        assert spans[-1][1] <= 90

    def test_three_panels_get_the_height_to_be_read(self):
        rows = three_unit_rows()
        built = self.payload(rows, "net sales, unit price and quantity by region")
        assert len(built["facets"]) == 3
        assert grown(built, clientHeight=320) == f"{3 * 130 + 70}px"
        # A dashboard tile keeps its size.
        assert grown(built, clientHeight=320, grow=False) is None

    def test_side_by_side_panels_are_as_wide_as_each_other_and_apart(self):
        rows = [{"ITM_DSC": f"Stainless steel fitting {i} inch", "NET_SLS_AMT": 90_000.0 - i * 5000,
                 "GRS_MRGN_PCT": 20.0 + i} for i in range(10)]
        grids = drawn(self.payload(rows, "net sales and gross margin percent by product"), "opt.grid")
        lefts = [float(g["left"].rstrip("%")) for g in grids]
        widths = [float(g["width"].rstrip("%")) for g in grids]
        assert widths[0] == widths[1]
        assert lefts[1] > lefts[0] + widths[0]
        assert lefts[1] + widths[1] < 100

    def test_the_names_column_fits_a_narrow_card(self):
        # render() tells the layout the card's width: in a 360px card the
        # names get 30% of it, not the 180px a wide one gives them.
        rows = [{"ITM_DSC": f"Stainless steel fitting {i} inch", "NET_SLS_AMT": 90_000.0 - i * 5000,
                 "GRS_MRGN_PCT": 20.0 + i} for i in range(10)]
        built = self.payload(rows, "net sales and gross margin percent by product")
        assert rendered(built, "opt.yAxis[0].axisLabel.width", width=360, height=900) == 108

    def test_a_long_ranking_in_panels_grows_like_a_ranking(self):
        rows = [{"ITM_DSC": f"Stainless steel fitting {i} inch", "NET_SLS_AMT": 90_000.0 - i * 2000,
                 "GRS_MRGN_PCT": 20.0 + i % 7} for i in range(20)]
        built = self.payload(rows, "net sales and gross margin percent by product")
        assert len(built["facets"]) == 2
        assert grown(built, clientHeight=320) == f"{20 * 22 + 72 + 24}px"

    def test_a_panel_for_a_measure_no_longer_sent_is_dropped(self):
        # A dashboard tile saved before a column left the result: one panel is
        # left, and one panel is an ordinary chart.
        payload = dict(self.payload(), facets=[["NET_SLS_AMT"], ["GONE_PCT"]], y_keys=["NET_SLS_AMT"])
        option = drawn(payload, "{panels: Array.isArray(opt.grid), series: opt.series.length}")
        assert option == {"panels": False, "series": 1}

    def test_a_long_ranking_is_drawn_side_by_side(self):
        rows = [{"ITM_DSC": f"Stainless steel fitting {i} inch", "NET_SLS_AMT": 90_000.0 - i * 5000,
                 "GRS_MRGN_PCT": 20.0 + i} for i in range(10)]
        option = drawn(self.payload(rows, "net sales and gross margin percent by product"),
                       "{names: opt.yAxis.map(function (a) { return a.axisLabel.show }),"
                       " inverse: opt.yAxis[0].inverse, link: opt.axisPointer.link}")
        # The first panel carries the product names, and every panel ranks
        # top-down on the same categories.
        assert option["names"] == [True, False]
        assert option["inverse"] is True
        assert option["link"] == [{"yAxisIndex": "all"}]


def grid_rows(warehouses=15, months=12):
    return [{"PRD_KEY": 202401 + m, "WHS_NM": f"Warehouse {w:02d}", "ON_HND_QTY": 500 + m * 20 + w * 35}
            for m in range(months) for w in range(warehouses)]


class TestAGridTooWideForASeriesIsAHeatmap:

    def test_fifteen_warehouses_by_twelve_months(self):
        built = build_chart_payload(grid_rows(), None, question="stock on hand by warehouse by month")
        assert built["chart_type"] == "heatmap"
        assert built["x_key"] == "WHS_NM"                 # a row per warehouse
        assert built["y_keys"][0] == "202401"             # a column per month
        assert len(built["y_keys"]) == 12
        assert built["grouped_measure"] == "ON_HND_QTY"
        assert built["rows"][0]["WHS_NM"] == "Warehouse 00"

    def test_time_runs_along_the_top_whichever_column_came_first(self):
        rows = [{"WHS_NM": r["WHS_NM"], "PRD_KEY": r["PRD_KEY"], "ON_HND_QTY": r["ON_HND_QTY"]}
                for r in grid_rows()]
        built = build_chart_payload(rows, None, question="stock on hand by warehouse by month")
        assert built["x_key"] == "WHS_NM"

    @pytest.mark.parametrize("time_first", [True, False])
    def test_time_runs_along_the_top_even_as_the_longer_side(self, time_first):
        rows = grid_rows(warehouses=10, months=20)
        if not time_first:
            rows = [{"WHS_NM": r["WHS_NM"], "PRD_KEY": r["PRD_KEY"], "ON_HND_QTY": r["ON_HND_QTY"]} for r in rows]
        built = build_chart_payload(rows, None, question="stock on hand by warehouse by month")
        assert (built["chart_type"], built["x_key"], len(built["y_keys"])) == ("heatmap", "WHS_NM", 20)

    @pytest.mark.parametrize("products_first", [True, False])
    def test_without_time_the_longer_side_runs_down(self, products_first):
        rows = [{"ITM_DSC": f"Item {i:02d}", "WHS_NM": f"Warehouse {w:02d}", "ON_HND_QTY": 100 + i * 7 + w}
                for i in range(30) for w in range(10)]
        if not products_first:
            rows = [{"WHS_NM": r["WHS_NM"], "ITM_DSC": r["ITM_DSC"], "ON_HND_QTY": r["ON_HND_QTY"]} for r in rows]
        built = build_chart_payload(rows, None, question="stock on hand by item by warehouse")
        assert (built["chart_type"], built["x_key"], len(built["y_keys"])) == ("heatmap", "ITM_DSC", 10)

    def test_a_cell_counted_twice_is_not_a_grid(self):
        rows = grid_rows()
        rows.append(dict(rows[0], ON_HND_QTY=10))
        spec = infer_chart_spec(rows, question="stock on hand by warehouse by month")
        assert spec["recommended_type"] == "table"

    def test_a_grid_mostly_empty_is_not_a_heatmap(self):
        rows = [r for r in grid_rows() if (r["PRD_KEY"] + int(r["WHS_NM"][-2:])) % 3 == 0]
        spec = infer_chart_spec(rows, question="stock on hand by warehouse by month")
        assert spec["recommended_type"] == "table"

    def test_a_member_spelled_like_a_column_is_not_pivoted_over_it(self):
        # The pivot makes each member of the column dimension a key of the row;
        # one spelled like the row's own label column would overwrite it.
        rows = [{"SEGMENT": f"Segment {s:02d}", "CHANNEL": "SEGMENT" if c == 0 else f"Channel {c}",
                 "NET_SLS_AMT": 1000.0 + s * 10 + c} for s in range(12) for c in range(10)]
        spec = infer_chart_spec(rows, question="net sales by segment by channel")
        assert spec["recommended_type"] == "table"

    def test_a_grid_a_series_can_carry_is_not_a_heatmap(self):
        built = build_chart_payload(grid_rows(warehouses=4), None,
                                    question="stock on hand by warehouse by month")
        assert built["chart_type"] == "line"

    def test_a_chart_type_the_reader_named_is_not_overruled(self):
        spec = infer_chart_spec(grid_rows(), question="stock on hand by warehouse by month as a bar chart")
        assert spec["recommended_type"] != "heatmap"

    def test_every_month_is_a_column_past_the_palettes_length(self):
        # A heatmap's columns are cells on one ramp, not series in colours: the
        # palette's eight is no limit on them.
        built = build_chart_payload(grid_rows(), None, question="stock on hand by warehouse by month")
        assert drawn(built, "opt.xAxis.data.length") == 12
        assert drawn(built, "opt.series[0].data.length") == 15 * 12

    def test_the_months_read_as_months_and_the_cells_in_the_measures_format(self):
        built = build_chart_payload(grid_rows(), None, question="stock on hand by warehouse by month")
        # The year once, under its first month, as on a time axis.
        heads = drawn(built, "opt.xAxis.data.map(function (v, i) { return opt.xAxis.axisLabel.formatter(v, i) })")
        assert heads[:3] == ["Jan\n2024", "Feb", "Mar"]
        tip = _build("portal_chat.html", "en", built, "opt.tooltip.formatter({value:[2, 0, 1250], color:'#000'})")
        assert "Warehouse 00 · March 2024" in tip
        assert "1,250" in tip

    def test_a_cohort_is_named_as_its_month(self):
        payload = {"chart_type": "heatmap", "x_key": "cohort", "y_keys": [f"Month {k}" for k in range(10)],
                   "rows": [{"cohort": f"2025-{c:02d}", **{f"Month {k}": 90.0 - k * 5 for k in range(10 - c)}}
                            for c in range(1, 4)]}
        assert drawn(payload, "opt.xAxis.data.length") == 10
        assert drawn(payload, "opt.yAxis.data.map(function (v) { return opt.yAxis.axisLabel.formatter(v) })") == [
            "Jan 2025", "Feb 2025", "Mar 2025"]
        tip = _build("portal_chat.html", "en", payload, "opt.tooltip.formatter({data:[1, 0, 85]})")
        assert "January 2025 · Month 1" in tip

    def test_a_value_is_written_in_a_cell_only_where_it_fits(self):
        built = build_chart_payload(grid_rows(warehouses=16), None, question="stock on hand by warehouse by month")
        labelled = "opt.series[0].label.show"
        # A chat card grows to 22px a row: every cell carries its value.
        assert rendered(built, labelled, width=640, height=320) is True
        # A 420 x 260 dashboard tile keeps its size: sixteen rows of 11px
        # cannot hold "1.1K", and the colour and the tooltip carry the value.
        assert rendered(built, labelled, width=420, height=260, grow=False) is False

    def test_short_row_names_leave_room_for_the_values_in_a_tile(self):
        payload = {"chart_type": "heatmap", "x_key": "cohort", "y_keys": [f"Month {k}" for k in range(7)],
                   "rows": [{"cohort": f"2025-{c:02d}", **{f"Month {k}": 90.0 - k * 5 for k in range(8 - c)}}
                            for c in range(1, 7)]}
        assert rendered(payload, "opt.series[0].label.show", width=388, height=260, grow=False) is True

    def test_every_warehouse_gets_a_readable_row(self):
        # Fifteen rows need more height than a card's default: the chart grows.
        built = build_chart_payload(grid_rows(), None, question="stock on hand by warehouse by month")
        assert grown(built, clientHeight=320) == f"{15 * 22 + 72}px"
