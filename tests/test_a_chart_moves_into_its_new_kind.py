"""A chart moves into the kind it is switched to: bars grow into slices, a line into bars, in the same chart.

Asked for (October 2026, the interactions): smooth transitions, and the chat's charts on the same component as the
dashboards'. A chart already drawn in an element is drawn again in place (QBCharts.draw -> update), its marks set to
move into the next option (ECharts' universalTransition); the chat's type switch, its "without one-offs" button and
its expand all go through it. A reader who asked for no motion sees the new chart at once.

Invented data only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

dukpy = pytest.importorskip("dukpy")

from tests.js_lift import function as lift  # noqa: E402
from tests.test_chart_annotation_language import _run  # noqa: E402
from tests.test_charts_animate_where_they_are_seen import CLOCK, bars, trend  # noqa: E402

CHAT = (Path(__file__).resolve().parents[1] / "portal" / "templates" / "portal_chat.html").read_text(encoding="utf-8")


def _as(payload, kind):
    return {**payload, "chart_type": kind}


def test_a_chart_on_screen_takes_its_new_kind_in_the_same_chart():
    script = CLOCK + f"""
var el = {{clientWidth: 600, clientHeight: 300, style: {{}}}};
QBCharts.draw(el, {json.dumps(bars())});
tick(); tick(); tick();
existing = charts[0];
var again = QBCharts.draw(el, {json.dumps(_as(bars(), "pie"))});
var last = charts[0].options[charts[0].options.length - 1];
JSON.stringify({{made: charts.length, same: again === charts[0], disposed: charts[0].disposed,
                calls: charts[0].calls, kind: last.series[0].type, moves: last.series[0].universalTransition}});
"""
    out = json.loads(_run("portal_chat.html", "en", script))
    assert out["made"] == 1 and out["same"] and not out["disposed"], "one chart, kept"
    assert out["calls"] == ["resize", "setOption", "setOption"], "drawn at once, no wait for a box it already has"
    assert out["kind"] == "pie" and out["moves"] == {"enabled": True}


def test_a_chart_not_yet_drawn_is_drawn_afresh():
    script = CLOCK + f"""
var el = {{clientWidth: 600, clientHeight: 300, style: {{}}}};
QBCharts.draw(el, {json.dumps(bars())});
tick(); tick(); tick();
JSON.stringify({{made: charts.length, drawn: charts[0].options.length}});
"""
    assert json.loads(_run("portal_chat.html", "en", script)) == {"made": 1, "drawn": 1}


def test_a_box_that_changes_with_the_kind_is_taken_before_the_move():
    # A pie is drawn taller than bars in the chat. Sized after the move, by the observer, the chart would be laid
    # out again at once -- resize() draws without animation -- and the move cut short.
    script = CLOCK + f"""
var el = {{clientWidth: 600, clientHeight: 200, style: {{}}}};
QBCharts.draw(el, {json.dumps(bars())});
tick(); tick(); tick();
existing = charts[0];
el.clientHeight = 280;
QBCharts.draw(el, {json.dumps(_as(bars(), "pie"))});
JSON.stringify(charts[0].calls);
"""
    assert json.loads(_run("portal_chat.html", "en", script)) == ["resize", "setOption", "resize", "setOption"]


@pytest.mark.parametrize("kind", ["bar", "line", "pie", "donut", "scatter"])
def test_marks_move_from_one_kind_to_the_next(kind):
    payload = _as(trend() if kind in ("line", "scatter") else bars(), kind)
    script = f"""var opt = QBCharts.buildOption({json.dumps(payload)});
JSON.stringify(opt.series.filter(function (s) {{ return ['bar', 'line', 'pie', 'scatter'].indexOf(s.type) >= 0; }})
  .map(function (s) {{ return s.universalTransition; }}));"""
    moves = json.loads(_run("portal_chat.html", "en", script))
    assert moves and all(m == {"enabled": True} for m in moves), (kind, moves)


@pytest.mark.parametrize("asked", [
    "window.QB_CHART_PREFS = {motion: 'off'};",
    "window.matchMedia = function (q) { return {matches: q.indexOf('reduce') >= 0}; };",
])
def test_a_reader_who_asked_for_no_motion_sees_the_new_kind_at_once(asked):
    script = asked + f"""
var opt = QBCharts.buildOption({json.dumps(_as(bars(), "pie"))});
JSON.stringify({{animation: opt.animation, moves: opt.series.map(function (s) {{ return s.universalTransition || null; }})}});"""
    out = json.loads(_run("portal_chat.html", "en", script))
    assert out["animation"] is False and set(out["moves"]) == {None}


def _chat(body: str) -> dict:
    script = (
        "var sent = [], drawn = [], rendered = 0;\n"
        "var el = {};\n"
        "var live = null;\n"
        "function makeChart() { var c = {handlers: [], on: function (event, fn) { if (event === 'click') this.handlers.push(fn); },"
        " getDom: function () { return el; }}; return c; }\n"
        "var window = {echarts: {getInstanceByDom: function () { return live; }}, ResizeObserver: function () {},"
        " _chartPayloads: {}, QBCharts: {\n"
        "  draw: function (target, payload, opts) { target._qbDrawn = {payload: payload, opts: opts};"
        " drawn.push([payload.chart_type, Boolean(opts.drill), live ? 'moved' : 'new']); live = live || makeChart(); return live; },\n"
        "  render: function () { rendered += 1; return makeChart(); }}};\n"
        "function t(id, vars) { return 'Break this down for ' + vars.label; }\n"
        "function escHtml(s) { return s; }\n"
        "function sendChartDrill(text) { sent.push(text); }\n"
        "function _openDrillMenu() { sent.push('menu'); }\n"
        + lift(CHAT, "function _chartDrillable(payload)") + "\n"
        + lift(CHAT, "function _wireChartDrillClick(chart, payload)") + "\n"
        + lift(CHAT, "function renderChartInto(chartEl, payload)") + "\n"
        + "function click() { live.handlers.forEach(function (fn) { fn({componentType: 'series', name: 'North'}); }); }\n"
        + body
    )
    return json.loads(dukpy.evaljs(script))


def test_the_chat_switches_a_kind_in_the_same_chart():
    out = _chat("""
renderChartInto(el, {chart_type: 'bar'});
renderChartInto(el, {chart_type: 'pie'});
renderChartInto(el, {chart_type: 'line'});
JSON.stringify({drawn: drawn, rendered: rendered, bound: live.handlers.length});
""")
    assert out["drawn"] == [["bar", True, "new"], ["pie", False, "moved"], ["line", True, "moved"]]
    assert out["rendered"] == 0, "never drawn again from nothing"
    assert out["bound"] == 1, "the click is bound once to the one chart"


def test_a_click_answers_for_the_kind_the_chart_shows_now():
    out = _chat("""
var said = [];
renderChartInto(el, {chart_type: 'bar'});
click(); said.push(sent.splice(0));
renderChartInto(el, {chart_type: 'pie'});
click(); said.push(sent.splice(0));
renderChartInto(el, {chart_type: 'bar'});
click(); said.push(sent.splice(0));
JSON.stringify(said);
""")
    assert out == [["Break this down for North"], [], ["Break this down for North"]]


def test_a_pie_turned_into_bars_answers_a_click():
    out = _chat("""
renderChartInto(el, {chart_type: 'pie'});
click();
var before = sent.splice(0);
renderChartInto(el, {chart_type: 'bar'});
click();
JSON.stringify([before, sent]);
""")
    assert out == [[], ["Break this down for North"]]
