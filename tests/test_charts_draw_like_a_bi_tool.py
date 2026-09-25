# -*- coding: utf-8 -*-
"""Charts drawn the way a BI tool draws them, by one renderer.

The chat and the dashboard drew their charts with two copies of an ECharts 5
option builder, and the copies had drifted apart. Screenshots of every chart
type showed what the reader got:

  * a city named "Marseille" read as March -- time was guessed from label text
    -- so a ranking of cities lost its cap and its horizontal layout;
  * a month key printed as "202401" on the axis;
  * a ranking of twenty cities showing every second name, and ranked bottom-up;
  * gradient bars, glowing markers and shadowed slices, "$812.0K" and "$-80.2K";
  * a legend key swallowed by the ring its series' markers carry.

Both pages now load one renderer, static/js/qb-charts.js, on Apache ECharts 6,
drawn as SVG. Every test here EXECUTES that renderer (the harness in
tests/test_chart_annotation_language.py builds the real option the page would
draw) and asserts on what it returns. Synthetic rows; no customer data.
"""

from __future__ import annotations

import json

import pytest

from core import i18n

pytest.importorskip("dukpy", reason="the renderer is JavaScript; only executing it shows what it draws")

from tests.test_chart_annotation_language import PAGES, _build, _run  # noqa: E402

CITIES = ["Marseille", "Mayenne", "Decatur", "Halifax", "Moncton", "Saint John",
          "Fredericton", "Truro", "Sydney", "Bathurst", "Miramichi", "Dieppe",
          "Amherst", "Yarmouth", "Kentville", "Antigonish", "Bedford", "Dartmouth",
          "Summerside", "Woodstock", "Sackville", "Riverview", "Oromocto", "Shediac"]
MONTHS = [202401 + i for i in range(12)]


def opt(payload, expression, page="portal_chat.html", lang="en"):
    return json.loads(_build(page, lang, payload, f"JSON.stringify({expression})"))


def ranking(labels, values=None, role="dimension", chart_type="bar", measure="AMT"):
    values = values or [float(100 - i) for i in range(len(labels))]
    return {"rows": [{"LBL": label, measure: value} for label, value in zip(labels, values)],
            "x_key": "LBL", "y_keys": [measure], "chart_type": chart_type,
            "chart_spec": {"x": {"column": "LBL", "role": role}}}


class TestTimeIsWhatTheSpecSaysItIs:

    @pytest.mark.parametrize("page", PAGES)
    def test_a_city_named_like_a_month_is_still_a_category(self, page):
        drawn = opt(ranking(CITIES), "{axis: opt.yAxis.data, inverse: opt.yAxis.inverse}", page)
        # Capped and turned horizontal like any ranking of 24 categories --
        # not kept whole as if "Marseille" were a month.
        assert len(drawn["axis"]) == 20
        assert drawn["axis"][0] == "Marseille"
        assert drawn["inverse"] is True

    @pytest.mark.parametrize("page", PAGES)
    def test_a_period_axis_is_never_capped_or_turned(self, page):
        labels = [str(202001 + (m // 12) * 100 + m % 12) for m in range(30)]
        drawn = opt(ranking(labels, role="temporal"), "{type: opt.xAxis.type, axis: opt.xAxis.data}", page)
        assert drawn["type"] == "category"
        assert drawn["axis"] == labels

    def test_a_chart_saved_before_the_spec_travelled_still_reads_its_periods(self):
        payload = ranking([f"{2024 + i // 12}-{i % 12 + 1:02d}" for i in range(24)])
        payload.pop("chart_spec")
        drawn = opt(payload, "{type: opt.xAxis.type, n: opt.xAxis.data.length}")
        assert drawn == {"type": "category", "n": 24}


class TestPeriodsReadAsDates:

    def axis(self, labels, lang="en", column="LBL"):
        payload = ranking([str(v) for v in labels], role="temporal", chart_type="line")
        payload["rows"] = [{column: r["LBL"], "AMT": r["AMT"]} for r in payload["rows"]]
        payload["x_key"] = column
        payload["chart_spec"]["x"]["column"] = column
        return lambda value, index: _build(
            "portal_chat.html", lang, payload,
            f"opt.xAxis.axisLabel.formatter({json.dumps(value)}, {index})")

    def test_a_month_key_is_a_month_and_the_year_is_written_once(self):
        fmt = self.axis(MONTHS)
        assert fmt("202401", 0) == "Jan\n2024"
        assert fmt("202402", 1) == "Feb"

    def test_the_year_is_written_again_where_it_changes(self):
        fmt = self.axis([202411, 202412, 202501, 202502])
        assert fmt("202501", 2) == "Jan\n2025"

    def test_in_french_the_months_are_french(self):
        fmt = self.axis(MONTHS, lang="fr")
        assert fmt("202401", 0) == i18n.MESSAGES["date.month.short.1"]["fr"] + "\n2024"

    def test_a_month_number_on_a_month_column_is_a_month_name(self):
        fmt = self.axis(list(range(1, 13)), column="SLS_MTH")
        assert fmt("1", 0) == "Jan"
        assert fmt("12", 11) == "Dec"

    def test_the_tooltip_names_the_whole_month(self):
        payload = ranking([str(v) for v in MONTHS], role="temporal", chart_type="line")
        html = _build("portal_chat.html", "en", payload,
                      "opt.tooltip.formatter([{axisValue:'202403', value:5, seriesName:'AMT', color:'#000'}])")
        assert "March 2024" in html

    def test_a_label_that_is_not_a_period_passes_through(self):
        days = ["Monday", "Tuesday", "Wednesday"]
        payload = ranking(days, role="temporal")
        assert _build("portal_chat.html", "en", payload,
                      "opt.xAxis.axisLabel.formatter('Monday', 0)") == "Monday"


class TestTheMarks:

    def test_a_bar_below_zero_rounds_its_own_end_and_labels_beyond_it(self):
        drawn = opt(ranking(["North", "South", "West"], [120.0, -80.0, 30.0], measure="PROFIT_AMT"),
                    "opt.series[0].data")
        assert drawn[0] == 120.0, "a bar that needs no styling stays a plain value"
        assert drawn[1]["itemStyle"]["borderRadius"] == [0, 0, 4, 4]
        assert drawn[1]["label"]["position"] == "bottom"

    def test_a_value_sits_at_the_tip_of_its_bar(self):
        # Not inside it: ink on a filled bar, at half its height, read as a
        # smudge on the dashboard.
        assert opt(ranking(["North", "South", "West"]), "opt.series[0].label.position") == "top"
        assert opt(ranking([f"Region number {n}" for n in range(12)]),
                   "opt.series[0].label.position") == "right"

    def test_the_axis_leaves_room_past_the_largest_labelled_bar(self):
        assert opt(ranking(["North", "South", "West"]), "opt.yAxis.boundaryGap") == [0, "8%"]
        assert opt(ranking([f"Region number {n}" for n in range(12)]),
                   "opt.xAxis.boundaryGap") == [0, "8%"]

    def test_a_horizontal_bar_below_zero_rounds_its_left_end(self):
        labels = [f"Region number {n}" for n in range(12)]
        values = [float(50 - n * 9) for n in range(12)]
        drawn = opt(ranking(labels, values, measure="PROFIT_AMT"), "opt.series[0].data")
        negative = next(item for item in drawn if isinstance(item, dict))
        assert negative["itemStyle"]["borderRadius"] == [4, 0, 0, 4]
        assert negative["label"]["position"] == "left"

    def test_a_variance_carries_its_direction_in_the_delta_colours_and_a_sign(self):
        payload = ranking(["North", "South", "West"], [120.0, -80.0, 30.0], measure="VARIANCE_AMT")
        colours = opt(payload, "opt.series[0].data.map(function (d) { return d.itemStyle.color })")
        assert colours == ["#070", "#a00", "#070"]   # the theme's good / bad
        label = _build("portal_chat.html", "en", payload,
                       "opt.series[0].label.formatter({value: 120})")
        assert label.startswith("+")

    def test_any_other_measure_keeps_one_colour_whatever_its_sign(self):
        payload = ranking(["North", "South", "West"], [120.0, -80.0, 30.0], measure="PROFIT_AMT")
        drawn = opt(payload, "[opt.series[0].itemStyle.color, opt.series[0].data[1].itemStyle.color]")
        assert drawn[0] == drawn[1]

    def test_lines_are_straight_segments(self):
        payload = ranking([str(v) for v in MONTHS], role="temporal", chart_type="line")
        assert opt(payload, "opt.series[0].smooth") is False

    def test_the_wash_belongs_to_an_area_and_not_to_a_line(self):
        payload = ranking([str(v) for v in MONTHS], role="temporal", chart_type="line")
        assert opt(payload, "opt.series[0].areaStyle === undefined") is True
        payload["chart_type"] = "area"
        assert opt(payload, "opt.series[0].areaStyle.opacity") == pytest.approx(0.10)

    def test_a_single_line_says_its_latest_value_at_its_end(self):
        payload = ranking([str(v) for v in MONTHS], role="temporal", chart_type="line")
        assert opt(payload, "opt.series[0].endLabel.show") is True

    def test_markers_give_way_on_a_long_series(self):
        short = ranking([str(v) for v in MONTHS], role="temporal", chart_type="line")
        long_ = ranking([f"2020-{1 + n % 12:02d}-{1 + n // 12:02d}" for n in range(40)],
                        role="temporal", chart_type="line")
        assert opt(short, "opt.series[0].showSymbol") is True
        assert opt(long_, "opt.series[0].showSymbol") is False


class TestTheLegendAndTheTooltip:

    TWO = {"rows": [{"R": r, "A": a, "B": b} for r, a, b in (("N", 1.0, 2.0), ("S", 3.0, 4.0))],
           "x_key": "R", "y_keys": ["A", "B"], "chart_type": "bar"}

    def test_one_series_needs_no_legend(self):
        assert opt(ranking(["N", "S"]), "opt.legend === undefined") is True

    def test_a_legend_key_is_not_swallowed_by_the_marker_ring(self):
        assert opt(self.TWO, "opt.legend.itemStyle.borderWidth") == 0

    def test_a_line_chart_keys_its_legend_with_a_line(self):
        line = dict(self.TWO, chart_type="line")
        assert opt(line, "opt.legend.icon").startswith("path://")

    def test_the_legend_swatch_is_the_series_colour(self):
        drawn = opt(self.TWO, "[opt.color[0], opt.series[0].itemStyle.color, opt.color[1], opt.series[1].itemStyle.color]")
        assert drawn[0] == drawn[1] and drawn[2] == drawn[3]

    def test_grouped_bars_compare_the_group_and_single_bars_their_mark(self):
        assert opt(self.TWO, "opt.tooltip.trigger") == "axis"
        assert opt(self.TWO, "opt.tooltip.axisPointer.type") == "shadow"
        assert opt(ranking(["N", "S"]), "opt.tooltip.trigger") == "item"

    def test_a_line_gets_a_crosshair(self):
        payload = ranking([str(v) for v in MONTHS], role="temporal", chart_type="line")
        assert opt(payload, "opt.tooltip.axisPointer.type") == "line"

    def test_a_category_name_is_escaped_in_the_tooltip(self):
        html = _build("portal_chat.html", "en", ranking(["<b>x</b>", "y"]),
                      "opt.tooltip.formatter({name:'<b>x</b>', value:1, color:'#000'})")
        assert "&lt;b&gt;x&lt;/b&gt;" in html
        assert "<b>x</b>" not in html


class TestTheNumbers:

    def fmt(self, value, lang="en"):
        payload = ranking(["N", "S"], measure="NET_SLS_AMT")
        payload["column_formats"] = {"NET_SLS_AMT": "currency"}
        return _build("portal_chat.html", lang, payload, f"opt.yAxis.axisLabel.formatter({value})")

    def test_a_whole_tier_carries_no_trailing_zero(self):
        assert self.fmt(812044) == "$812K"
        assert self.fmt(1200000) == "$1.2M"

    def test_the_sign_leads_the_currency_symbol(self):
        assert self.fmt(-80200) == "-$80.2K"

    def test_french_puts_the_symbol_after_and_the_sign_first(self):
        drawn = self.fmt(-80200, lang="fr")
        assert drawn.startswith("-") and drawn.endswith("$")
        assert "80,2" in drawn


class TestThePieAndTheDonut:

    FOUR = {"rows": [{"CH": c, "V": v} for c, v in (("Retail", 520.0), ("Online", 310.0),
                                                    ("Trade", 150.0), ("Other", 20.0))],
            "x_key": "CH", "y_keys": ["V"], "chart_type": "donut"}

    def test_slices_are_separated_by_the_surface_not_by_an_outline(self):
        drawn = opt(self.FOUR, "opt.series[0].itemStyle")
        assert drawn == {"borderColor": "#fff", "borderWidth": 2}

    def test_nothing_glows_on_hover(self):
        assert "shadowBlur" not in _build("portal_chat.html", "en", self.FOUR, "JSON.stringify(opt)")

    def test_the_donut_says_its_total_in_the_middle(self):
        texts = opt(self.FOUR, "opt.graphic[0].children.map(function (c) { return c.style.text })")
        assert texts == ["1K", "Total"]


class TestALongRankingIsReadable:

    def mount(self, options="undefined"):
        script = """
window.echarts = {getInstanceByDom: function () { return null; },
                  init: function () { return {setOption: function (o) { this.o = o; },
                                              resize: function () {}, dispose: function () {}}; }};
var el = {clientHeight: 320, style: {}};
var chart = QBCharts.render(el, PAYLOAD, OPTIONS);
JSON.stringify({height: el.style.height || null, interval: chart.o.yAxis.axisLabel.interval});
""".replace("PAYLOAD", json.dumps(ranking(CITIES[:20]))).replace("OPTIONS", options)
        return json.loads(_run("portal_chat.html", "en", script))

    def test_the_chart_grows_to_give_every_category_a_row(self):
        # Twenty categories at 22px each, plus the axis band.
        assert self.mount() == {"height": "512px", "interval": 0}

    def test_a_dashboard_card_keeps_its_size_and_names_every_other_row(self):
        assert self.mount("{grow: false}") == {"height": None, "interval": "auto"}

    def test_the_first_row_of_a_ranking_is_at_the_top(self):
        assert opt(ranking(CITIES), "opt.yAxis.inverse") is True


class TestAHeatmapOfAMeasure:

    def test_a_pivoted_measure_reads_in_its_own_format(self):
        payload = {"rows": [{"WHS": "Halifax", "202401": 1200.0, "202402": 900.0}],
                   "x_key": "WHS", "y_keys": ["202401", "202402"], "chart_type": "heatmap",
                   "grouped_measure": "ON_HND_VAL", "column_formats": {"ON_HND_VAL": "currency"}}
        html = _build("portal_chat.html", "en", payload,
                      "opt.tooltip.formatter({value:[0, 0, 1200], color:'#000'})")
        assert "$1,200.00" in html
        assert "Retention" not in html
