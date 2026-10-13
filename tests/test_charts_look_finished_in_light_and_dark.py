"""The charts' finished look (approved October 2026, from before-and-after pictures), in light and in dark.

- On the dark page every palette a reader may choose is drawn in its own dark steps: the same hues in the same
  order, stepped to read on the dark surface. The light steps there sat outside the lightness band, and the
  default's violet, at 2:1, all but vanished. Magnitude (a heatmap, a map) runs the other way on the dark page,
  the most brightest; and a value written inside a pale cell is in dark ink whatever the page's own ink is.
- Bars that each say their value drop the scale and its gridlines, which would only say the values again.
- Several lines carry no dot on every point; two to four are named at their ends where their ends lie apart.
- A single line stands on a wash, its latest point ringed on a halo of its own hue; its highest and latest
  values are in strong ink.
- Stacked bars total their parts in the tooltip; the tooltip is the page's own card.

Every test executes the renderer (static/js/qb-charts.js) with the palettes file (static/js/chart-palettes.js).
Invented numbers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("dukpy", reason="the renderer is JavaScript; only executing it shows what it draws")
import dukpy  # noqa: E402

from store.user_store import CHART_PREFS  # noqa: E402
from tests.test_chart_annotation_language import (  # noqa: E402
    _STATUS_STUB, _palette_literal, _renderer_for, _shell_preamble)
from tests.test_chart_spec import ChartPaletteValidationTests as Colour  # noqa: E402

DARK_SURFACE = "#141842"
PALETTES_JS = (Path(__file__).resolve().parents[1] / "static" / "js" / "chart-palettes.js").read_text(encoding="utf-8")


def _ramp(name: str) -> str:
    """A one-line array (a sequential ramp) as chart-palettes.js writes it."""
    return re.search(rf"window\.{name} = (\[[^\]]*\]);", PALETTES_JS).group(1)

LIGHT_THEME = """window.QB_CHART_THEME = function () {
  return {axis:'#888', split:'#eee', axisLine:'#ccc', tooltipBg:'#fff', tooltipText:'#0D0F2E', surface:'#FCFCFF',
          ink2:'#444', good:'#070', bad:'#a00', font:'sans-serif'};
};"""
DARK_THEME = """window.QB_CHART_THEME = function () {
  return {axis:'#A9AED6', split:'#262C66', axisLine:'#363F88', tooltipBg:'#181C4C', tooltipText:'#F4F5FF',
          surface:'%s', ink2:'#C9CCEB', good:'#3BD08A', bad:'#FF7A6E', font:'sans-serif'};
};""" % DARK_SURFACE


def _run(payload: dict, expression: str, *, dark: bool = False, page: str = "portal_chat.html", layout=None):
    body = "\n".join([
        _shell_preamble("en"), _STATUS_STUB, DARK_THEME if dark else LIGHT_THEME,
        *[f"window.{name} = " + _palette_literal(name) + ";" for name in ("QB_PALETTES", "QB_PALETTES_DARK")],
        "window.QB_SEQUENTIAL = " + _ramp("QB_SEQUENTIAL") + ";",
        "window.QB_SEQUENTIAL_DARK = " + _ramp("QB_SEQUENTIAL_DARK") + ";",
        _renderer_for(page),
        f"var opt = QBCharts.buildOption({json.dumps(payload)}, {json.dumps(layout)});",
        f"JSON.stringify({expression})",
    ])
    return json.loads(dukpy.evaljs(body))


def _palettes(name: str):
    literal = _ramp(name) if name.startswith("QB_SEQUENTIAL") else _palette_literal(name)
    return json.loads(dukpy.evaljs(f"var P = {literal}; JSON.stringify(P)"))


MONTHS = [f"2025-{m:02d}-01" for m in range(1, 13)]


def _line(series: dict[str, list[float]], **extra) -> dict:
    return {"chart_type": "line", "x_key": "period", "y_keys": list(series), "x_style": "month_year_long",
            "chart_spec": {"x": {"column": "period", "role": "temporal"}},
            "rows": [{"period": m, **{k: v[i] for k, v in series.items()}} for i, m in enumerate(MONTHS)], **extra}


def _ranking(count: int, **extra) -> dict:
    return {"chart_type": "bar", "x_key": "store", "y_keys": ["amt"],
            "rows": [{"store": f"Store {i:02d}", "amt": 1000.0 - 10 * i} for i in range(count)], **extra}


STACKED = {"chart_type": "stacked", "x_key": "region", "y_keys": ["Retail", "Online"],
           "rows": [{"region": r, "Retail": a, "Online": b} for r, a, b in (("North", 70.0, 30.0), ("West", 40.0, 25.0))]}
GRID = {"chart_type": "heatmap", "x_key": "region", "y_keys": ["Retail", "Online", "Trade"],
        "rows": [{"region": r, "Retail": a, "Online": b, "Trade": c}
                 for r, a, b, c in (("North", 900.0, 90.0, 100.0), ("West", 130.0, 240.0, 95.0))]}


# ── colour on the dark page ───────────────────────────────────────────────────


def _lightness(hexes):
    return [Colour._oklab_from_lin(Colour._lin(h))[0] for h in hexes]


def _contrast(a: str, b: str) -> float:
    def lum(h):
        r, g, b_ = Colour._lin(h)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b_
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("name", CHART_PREFS["palette"])
def test_every_palette_a_reader_may_choose_has_dark_steps_that_read_on_the_dark_page(name):
    light, dark = _palettes("QB_PALETTES")[name], _palettes("QB_PALETTES_DARK")[name]
    assert len(dark) == len(light) == 8, "a colour follows its series across the two modes"
    assert all(0.48 <= L <= 0.67 for L in _lightness(dark)), "the dark lightness band"
    assert all(_contrast(h, DARK_SURFACE) >= 3.0 for h in dark), "3:1 against the dark surface"
    pairs = list(zip(dark, dark[1:]))
    assert min(Colour._delta_e(a, b) for a, b in pairs) >= 15, "neighbours told apart with full colour vision"
    assert min(Colour._delta_e(a, b, kind) for a, b in pairs for kind in ("protan", "deutan")) >= 8, \
        "and by a colour-blind reader"


def test_the_default_violet_no_longer_vanishes_on_the_dark_page():
    light, dark = _palettes("QB_PALETTES")["default"], _palettes("QB_PALETTES_DARK")["default"]
    assert _contrast(light[6], DARK_SURFACE) < 2.5 < 3.0 <= _contrast(dark[6], DARK_SURFACE)


def test_a_chart_on_the_dark_page_is_drawn_in_the_dark_steps_of_the_palette_it_uses():
    dark = _palettes("QB_PALETTES_DARK")
    light = _palettes("QB_PALETTES")
    assert _run(_ranking(3), "opt.color") == light["default"]
    assert _run(_ranking(3), "opt.color", dark=True) == dark["default"]
    assert _run(_ranking(3, color_palette="ocean"), "opt.color", dark=True) == dark["ocean"]
    assert _run(_ranking(3), "[QBCharts.paletteFor('candy'), QBCharts.paletteFor('nothing such')]", dark=True) \
        == [dark["candy"], dark["default"]]


def test_magnitude_runs_brightest_on_the_dark_page_and_a_pale_cell_keeps_dark_ink():
    cells = _run(GRID, "opt.series[0].data.map(function (d) { return [d.value[2], d.label.color]; })", dark=True)
    by_value = dict((v, ink) for v, ink in cells)
    ramp = _palettes("QB_SEQUENTIAL_DARK")
    assert _run(GRID, "opt.visualMap.inRange.color", dark=True) == ramp
    assert _lightness([ramp[0]])[0] < _lightness([ramp[-1]])[0], "the least nearest the surface, the most brightest"
    assert by_value[900.0] == "#0D0F2E", "the largest cell is pale: its value in dark ink, not the page's light ink"
    assert by_value[90.0] == "#FFFFFF"
    assert _run(GRID, "opt.visualMap.inRange.color") == _palettes("QB_SEQUENTIAL"), "the light page keeps its ramp"


# ── bars ──────────────────────────────────────────────────────────────────────


def test_bars_that_each_say_their_value_drop_the_scale_and_its_gridlines():
    shown = "[opt.xAxis.axisLabel.show, opt.xAxis.splitLine.show, opt.series[0].label.show]"
    assert _run(_ranking(12), shown) == [False, False, True]
    column = {**_ranking(5), "rows": _ranking(5)["rows"]}
    assert _run(column, "[opt.yAxis.axisLabel.show, opt.yAxis.splitLine.show]") == [False, False]
    # Grouped bars do not say their values: the scale stays.
    grouped = {"chart_type": "bar", "x_key": "region", "y_keys": ["this_year", "last_year"],
               "column_formats": {"this_year": "currency", "last_year": "currency"},
               "rows": [{"region": r, "this_year": a, "last_year": b} for r, a, b in (("North", 7.0, 5.0), ("West", 4.0, 6.0))]}
    assert _run(grouped, "[opt.yAxis.axisLabel.show !== false, opt.yAxis.splitLine.show !== false, !!opt.series[0].label]") \
        == [True, True, False]


def test_the_bar_under_the_pointer_stays_and_the_others_step_back():
    assert _run(_ranking(5), "[opt.series[0].emphasis.focus, opt.series[0].blur.itemStyle.opacity]") == ["self", 0.4]


def test_a_rankings_names_read_in_secondary_ink():
    assert _run(_ranking(12), "opt.yAxis.axisLabel.color") == "#444"


def test_stacked_parts_stand_two_pixels_apart_and_the_tooltip_totals_them():
    assert _run(STACKED, "opt.series.map(function (s) { return s.itemStyle.borderWidth; })") == [2, 2]
    tip = _run(STACKED, "opt.tooltip.formatter([{axisValue: 'North', seriesName: 'Retail', value: 70, color: '#111'},"
                        " {axisValue: 'North', seriesName: 'Online', value: 30, color: '#222'}])")
    assert "Total" in tip and tip.index("Online") < tip.index("Total") and "100" in tip.split("Total")[1]


# ── lines ─────────────────────────────────────────────────────────────────────


def test_several_lines_carry_no_dot_on_every_point_and_are_named_at_their_ends_when_apart():
    apart = _line({"North": [float(10 + i) for i in range(12)], "South": [float(60 + i) for i in range(12)]})
    drawn = _run(apart, "opt.series.map(function (s) { return [s.showSymbol, s.endLabel && s.endLabel.show,"
                        " s.endLabel && s.endLabel.formatter()]; })")
    assert drawn == [[False, True, "North"], [False, True, "South"]]
    assert _run(apart, "opt.grid.right") > 16, "room at the right for the names"
    meeting = _line({"North": [float(10 + i) for i in range(12)], "South": [float(10.2 + i) for i in range(12)]})
    assert _run(meeting, "opt.series.map(function (s) { return !!(s.endLabel && s.endLabel.show); })") == \
        [False, False], "ends that meet are left to the legend and the tooltip, never nudged off their lines"
    five = _line({f"R{n}": [float(10 * n + i) for i in range(12)] for n in range(1, 6)})
    assert not any(_run(five, "opt.series.map(function (s) { return !!(s.endLabel && s.endLabel.show); })"))


def test_a_single_line_stands_on_a_wash_with_its_latest_point_ringed():
    values = [21.0, 18.0, 23.0, 22.0, 20.0, 23.0, 22.5, 20.5, 26.0, 24.0, 23.5, 24.5]
    marks = _run(_line({"amt": values}), "opt.series[0].markPoint.data")
    latest = [m for m in marks if not m.get("label", {}).get("show")]
    assert [m["coord"] for m in latest] == [[11, 24.5], [11, 24.5]]
    halo, dot = latest
    assert halo["symbolSize"] > dot["symbolSize"] == 10 and halo["itemStyle"]["color"].startswith("rgba(")
    assert dot["itemStyle"]["borderWidth"] == 2
    assert "colorStops" in _run(_line({"amt": values}), "opt.series[0].areaStyle.color")


def test_the_wash_and_the_halo_are_stronger_on_the_dark_page():
    line = _line({"amt": [float(v) for v in range(12)]})
    top = "opt.series[0].areaStyle.color.colorStops[0].color"
    light, dark = _run(line, top), _run(line, top, dark=True)
    assert float(light.rstrip(")").split(",")[-1]) < float(dark.rstrip(")").split(",")[-1])


def test_a_months_axis_names_every_month_where_there_is_room():
    assert _run(_line({"amt": [1.0] * 12}), "opt.xAxis.axisLabel.interval", layout={"width": 760, "height": 0}) == 0


def test_a_tiles_value_axis_has_four_steps():
    assert _run(_line({"amt": [1.0] * 12}, tile=True), "opt.yAxis.splitNumber", page="portal_dashboard.html") == 4
    assert _run(_line({"amt": [1.0] * 12}), "opt.yAxis.splitNumber === undefined") is True


# ── the tooltip ───────────────────────────────────────────────────────────────


def test_the_tooltip_is_the_pages_own_card_with_a_deeper_shadow_in_the_dark():
    light = _run(_ranking(3), "opt.tooltip.extraCssText")
    dark = _run(_ranking(3), "opt.tooltip.extraCssText", dark=True)
    assert "border-radius:10px" in light and light != dark and "rgba(0,0,0" in dark
