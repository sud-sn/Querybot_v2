"""A chart is drawn again in the page's colours when the reader changes them, and a line's values stay off it.

Seen in the chart audit, on the retail stress data:

* A reader who turned the page dark kept every chart already on it in the light theme's colours: axis names,
  values and legends in a grey too dim to read on the dark page, gridlines glaring over the data. The page told
  the charts (``qb-theme-change``) and nothing listened.
* A line of monthly sales with a one-off month had the months either side of it say their value right where the
  line climbs to it and falls from it: "$180.6K" crossed by the line.

Every test executes the renderer (static/js/qb-charts.js) as the page does. Invented figures.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("dukpy", reason="the renderer is JavaScript; only executing it shows what it draws")

import dukpy  # noqa: E402

from tests.test_chart_annotation_language import (  # noqa: E402
    _STATUS_STUB,
    _THEME_STUB,
    _build,
    _palette_literal,
    _renderer_for,
    _shell_preamble,
)


def _line(values, *, tile=False):
    payload = {"rows": [{"MONTH": f"2025-{m:02d}-01", "AMT": v} for m, v in enumerate(values, start=1)],
               "x_key": "MONTH", "y_keys": ["AMT"], "chart_type": "line",
               "chart_spec": {"x": {"column": "MONTH", "role": "time", "grain": "month"}}}
    if tile:
        payload["tile"] = True
    return payload


def _drawn(payload, expression, page="portal_chat.html"):
    return json.loads(_build(page, "en", payload, f"JSON.stringify({expression})"))


def _sides(values, page="portal_chat.html"):
    data = _drawn(_line(values), "opt.series[0].data", page)
    return [d["label"]["position"] if isinstance(d, dict) else "top" for d in data]


def _labels(values, page="portal_chat.html"):
    return _drawn(_line(values), "[" + ", ".join(
        f"opt.series[0].label.formatter({{value: {v}, dataIndex: {i}}})" for i, v in enumerate(values)) + "]", page)


# ── a line's values stay off the line ───────────────────────────────────────

ONE_OFF = [160.0, 150.0, 170.0, 185.0, 165.0, 175.0, 180.0, 1100.0, 182.0, 170.0, 215.0, 210.0]


@pytest.mark.parametrize("page", ["portal_chat.html", "portal_dashboard.html"])
def test_the_months_either_side_of_a_one_off_say_their_value_below_the_line(page):
    sides = _sides(ONE_OFF, page)
    assert sides[6] == "bottom" and sides[8] == "bottom", "the line climbs from July and falls into September"
    assert [s for i, s in enumerate(sides) if i not in (6, 8)] == ["top"] * 10
    # Every month still says its value: only where it says it moved.
    assert all(_labels(ONE_OFF, page))


def test_a_point_in_the_middle_of_a_climb_says_its_value_on_the_side_the_line_is_not():
    rising = [100.0, 400.0, 700.0, 1000.0, 1300.0]
    sides = _sides(rising)
    assert sides[1:4] == ["left"] * 3, "the line runs up to the right: the room is to the left of each point"
    falling = list(reversed(rising))
    assert _sides(falling)[1:4] == ["right"] * 3


def test_a_point_too_near_the_foot_for_a_label_below_leaves_its_value_to_the_tooltip():
    rising = [100.0, 400.0, 700.0, 1000.0, 1300.0]
    labels = _labels(rising)
    # $100 sits under a climb, and a label below it would sit on the axis's own labels.
    assert labels[0] == "" and all(labels[1:])


def test_a_gentle_line_keeps_every_value_above_its_point():
    gentle = [200.0, 210.0, 205.0, 220.0, 215.0, 230.0, 225.0, 240.0]
    assert _sides(gentle) == ["top"] * len(gentle)


# ── drawn again in the page's colours ───────────────────────────────────────

_PAGE = """
var handlers = {};
window.addEventListener = function (name, fn) { (handlers[name] = handlers[name] || []).push(fn); };
"""

_A_CHART_ON_THE_PAGE = """
var options = [];
var el = {clientWidth: 640, clientHeight: 320, style: {}};
var chart = {getDom: function () { return el; }, isDisposed: function () { return false; },
             setOption: function (o) { options.push(o); }};
var payload = %s;
QBCharts.update(chart, payload, {});
window.document = {querySelectorAll: function (sel) { return sel === '[_echarts_instance_]' ? [el] : []; }};
window.echarts = {getInstanceByDom: function (node) { return node === el ? chart : null; }};
window.QB_CHART_THEME = function () {
  return {axis:'#aab', split:'#334', axisLine:'#445', tooltipBg:'#112', tooltipText:'#fff', surface:'#0b0d1e',
          ink2:'#ccd', good:'#4c4', bad:'#f66', font:'sans-serif'};
};
"""


def _page_with_a_chart(payload, then):
    body = "\n".join([_shell_preamble("en"), _PAGE, _STATUS_STUB, _THEME_STUB,
                      "window.QB_PALETTES = " + _palette_literal("QB_PALETTES") + ";",
                      _renderer_for("portal_chat.html"), _A_CHART_ON_THE_PAGE % json.dumps(payload), then])
    return json.loads(dukpy.evaljs(body))


def test_a_chart_on_the_page_is_drawn_again_in_the_colours_the_page_turned_to():
    seen = _page_with_a_chart(_line(ONE_OFF), """
(handlers['qb-theme-change'] || []).forEach(function (fn) { fn({detail: {theme: 'dark'}}); });
JSON.stringify({drawn: options.length,
                before: options[0].yAxis.axisLabel.color, after: options[options.length - 1].yAxis.axisLabel.color,
                grid: options[options.length - 1].yAxis.splitLine.lineStyle.color});
""")
    assert seen["drawn"] == 2, "drawn once, then again when the page changed theme"
    assert seen["before"] == "#888" and seen["after"] == "#aab" and seen["grid"] == "#334"


def test_a_chart_is_drawn_again_with_what_it_last_showed():
    """A chart switched to bars, then the page turned dark: drawn dark as bars, not as the line it began as."""
    seen = _page_with_a_chart(_line(ONE_OFF), """
var bars = JSON.parse(JSON.stringify(payload)); bars.chart_type = 'bar';
QBCharts.update(chart, bars, {});
QBCharts.redrawAll();
JSON.stringify(options.map(function (o) { return o.series[0].type; }));
""")
    assert seen == ["line", "bar", "bar"]


# ── a scatter's names stay inside the chart ─────────────────────────────────

def test_a_named_dot_at_the_right_of_a_scatter_is_named_on_its_left():
    """The dots named are the furthest out on the first measure, so at the right edge: "Riverside Sto" was cut off
    by the chart's edge. Named on their left there; a dot in the left half keeps its name on its right."""
    stores = [("Harbour Store", 550, 460.0), ("Hilltop Store", 480, 300.0), ("Cove Store", 120, 900.0),
              ("Mill Store", 260, 180.0), ("Quay Store", 300, 200.0)]
    payload = {"rows": [{"STORE": s, "ORDERS": n, "AMT": a} for s, n, a in stores],
               "x_key": "STORE", "y_keys": ["ORDERS", "AMT"], "chart_type": "scatter"}
    data = _drawn(payload, "opt.series[0].data")
    sides = {d["value"][2]: (d.get("label") or {}).get("position") for d in data}
    assert sides["Harbour Store"] == "left" and sides["Hilltop Store"] == "left"
    assert sides["Quay Store"] == "right", "the third furthest out, left of the middle: named on its right"
    assert sides["Cove Store"] is None and sides["Mill Store"] is None
