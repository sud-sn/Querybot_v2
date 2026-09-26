# -*- coding: utf-8 -*-
"""A chart that animates where the reader can see it, and says what a click does.

The entrance was there and nobody saw it. The chat draws an answer's chart
into a pane that slides open from nothing: the chart was drawn at width 0,
played its entrance there, and snapped to its size as the pane opened --
sampled frame by frame in Chromium, the bars stood at full height 36ms after
they appeared. A chart is now drawn once its box has held one size from one
frame to the next, and its entrance is long enough to be seen: 600ms, the
bars rising one after another. A reader who asked the system for less motion
gets none.

A click on a bar or a point in the chat asks for that category's breakdown,
and nothing said so. The tooltip does now, exactly where a click is answered.

A dashboard card redrawn with a new type or palette is drawn for its own box,
as its first drawing was -- the redraw used a wide card's layout, which
clipped a narrow card's names. A type chosen in a card's expanded view now
reaches the card behind it, and the card's badge names the type in the
reader's language instead of the raw type code.

Every test executes the real renderer (static/js/qb-charts.js) or the page's
own functions, with ECharts, the frame clock and the DOM stubbed at their
boundaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.chart import build_chart_payload

dukpy = pytest.importorskip("dukpy", reason="the renderer is JavaScript; only executing it shows what it does")
from tests.js_lift import function as lift  # noqa: E402
from tests.test_chart_annotation_language import _run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHAT = (ROOT / "portal" / "templates" / "portal_chat.html").read_text(encoding="utf-8")
DASHBOARD = (ROOT / "portal" / "templates" / "portal_dashboard.html").read_text(encoding="utf-8")

REGIONS = ("North", "South", "West", "East")


def ranking_in_panels():
    rows = [{"ITM_DSC": f"Stainless steel fitting {i} inch", "NET_SLS_AMT": 90_000.0 - i * 5000,
             "GRS_MRGN_PCT": 20.0 + i} for i in range(10)]
    return build_chart_payload(rows, None, question="net sales and gross margin percent by product")


def bars():
    rows = [{"RGN_NM": r, "NET_SLS_AMT": v} for r, v in zip(REGIONS, (1200.0, 800.0, 300.0, 650.0))]
    return build_chart_payload(rows, None, question="net sales by region")


def grouped_bars():
    rows = [{"RGN_NM": r, "NET_SLS_AMT": s, "COGS_AMT": s * 0.6} for r, s in zip(REGIONS, (1200.0, 800.0, 300.0, 650.0))]
    return build_chart_payload(rows, None, question="net sales and cost by region")


def trend():
    rows = [{"YR_MTH": f"2025-{m:02d}", "NET_SLS_AMT": 1000.0 + m * 40} for m in range(1, 13)]
    return build_chart_payload(rows, None, question="net sales by month")


def drawn(payload, expression, lang="en", layout="undefined"):
    """`expression` over the option the renderer builds for `payload`."""
    script = (f"var opt = QBCharts.buildOption({json.dumps(payload)}, {layout});\n"
              f"JSON.stringify({expression});")
    return json.loads(_run("portal_chat.html", lang, script))


# A frame clock and an ECharts that record what was drawn and when.
CLOCK = """
var frames = [], charts = [];
window.requestAnimationFrame = function (fn) { frames.push(fn); };
function tick() { var due = frames.splice(0); due.forEach(function (fn) { fn(); }); }
function stubChart(el) {
  var chart = {options: [], calls: [], disposed: false, size: el ? [el.clientWidth, el.clientHeight] : [0, 0],
               getWidth: function () { return this.size[0]; }, getHeight: function () { return this.size[1]; },
               getDom: function () { return this.el; },
               setOption: function (o, notMerge) { this.options.push(o); this.calls.push('setOption'); },
               resize: function () { this.calls.push('resize'); if (this.el) this.size = [this.el.clientWidth, this.el.clientHeight]; },
               on: function () {},
               dispose: function () { this.disposed = true; },
               isDisposed: function () { return this.disposed; }};
  chart.el = el;
  charts.push(chart);
  return chart;
}
var existing = null, boxChanged = null;
window.echarts = {getInstanceByDom: function () { return existing; },
                  init: function (el) { return stubChart(el); }};
window.ResizeObserver = function (callback) {
  this.observe = function () { boxChanged = callback; };
  this.disconnect = function () {};
};
"""


class TestTheEntranceIsSeen:

    def test_a_chart_in_an_opening_pane_is_drawn_once_the_pane_has_opened(self):
        # The widths the answer pane reported, frame by frame, in Chromium --
        # closed for a frame or two, then opening.
        script = CLOCK + f"""
var el = {{clientWidth: 0, clientHeight: 494, style: {{}}}};
QBCharts.render(el, {json.dumps(ranking_in_panels())});
var log = [];
[0, 0, 383, 452, 505, 505].forEach(function (w) {{ el.clientWidth = w; tick(); log.push(charts[0].options.length); }});
JSON.stringify({{log: log, calls: charts[0].calls, names: charts[0].options[0].yAxis[0].axisLabel.width}});
"""
        out = json.loads(_run("portal_chat.html", "en", script))
        # Nothing while the pane is shut or opening -- a width of nothing held
        # for two frames is not a size; drawn once, when it has held a width.
        assert out["log"] == [0, 0, 0, 0, 0, 1]
        # Sized to the box before it is drawn: the instance was made when the
        # box was nothing wide.
        assert out["calls"] == ["resize", "setOption"]
        # And laid out for the width it settled at, not the zero it began at:
        # the ranking's names get 30% of 505px.
        assert out["names"] == round(505 * 0.3)

    def test_a_box_that_did_not_change_size_does_not_cut_the_entrance_short(self):
        # In the chat the observer fired 65ms after the chart was drawn, for a
        # box still 619 x 494: resize() redraws without animation, and the
        # bars jumped to their full height in one frame.
        script = CLOCK + f"""
var el = {{clientWidth: 619, clientHeight: 494, style: {{}}}};
QBCharts.render(el, {json.dumps(bars())});
tick(); tick();
var drawnAt = charts[0].calls.length;
boxChanged();                                   // the same box
var afterSame = charts[0].calls.slice(drawnAt);
el.clientWidth = 700;
boxChanged();                                   // a wider one
JSON.stringify({{same: afterSame, wider: charts[0].calls.slice(drawnAt)}});
"""
        out = json.loads(_run("portal_chat.html", "en", script))
        assert out["same"] == []
        assert out["wider"] == ["resize"]

    def test_a_box_that_never_settles_is_drawn_within_a_second(self):
        script = CLOCK + f"""
var el = {{clientWidth: 100, clientHeight: 300, style: {{}}}};
QBCharts.render(el, {json.dumps(bars())});
var first = null;
for (var i = 1; i <= 80; i++) {{ el.clientWidth = 100 + i; tick(); if (first === null && charts[0].options.length) first = i; }}
JSON.stringify({{first: first, times: charts[0].options.length}});
"""
        out = json.loads(_run("portal_chat.html", "en", script))
        assert out["times"] == 1
        assert 50 <= out["first"] <= 65          # about a second of frames

    def test_a_chart_redrawn_in_its_own_box_is_drawn_at_once(self):
        # A new type for the chart already in the card: the box is laid out,
        # and waiting would blank the chart for a frame.
        script = CLOCK + f"""
existing = stubChart();
var el = {{clientWidth: 600, clientHeight: 320, style: {{}}}};
QBCharts.render(el, {json.dumps(bars())});
JSON.stringify({{old: existing.disposed, drawnNow: charts[1].options.length}});
"""
        assert json.loads(_run("portal_chat.html", "en", script)) == {"old": True, "drawnNow": 1}

    def test_a_chart_replaced_before_it_was_drawn_is_never_drawn(self):
        script = CLOCK + f"""
var el = {{clientWidth: 600, clientHeight: 320, style: {{}}}};
QBCharts.render(el, {json.dumps(bars())});          // waits for its box
existing = charts[0];
QBCharts.render(el, {json.dumps(trend())});         // replaced before it settled
tick(); tick(); tick();
JSON.stringify(charts.map(function (c) {{ return c.options.length; }}));
"""
        assert json.loads(_run("portal_chat.html", "en", script)) == [0, 1]


class TestAResizeOnlyForABoxThatChanged:
    """resize() redraws without animation. The pages called it on every chart
    after any layout event -- the dashboard 120ms after its grid was laid out,
    125ms into every tile's entrance -- and each call cut an entrance short."""

    CHARTS = """
var resizes = [];
function sized(id, w, h, boxW, boxH) {
  var el = {id: id, clientWidth: boxW, clientHeight: boxH};
  return {el: el, getDom: function () { return el; }, getWidth: function () { return w; },
          getHeight: function () { return h; }, resize: function () { resizes.push(id); },
          isDisposed: function () { return false; }};
}
"""

    def test_fit_resizes_only_a_box_that_changed(self):
        script = self.CHARTS + """
var same = sized('same', 600, 320, 600, 320), wider = sized('wider', 600, 320, 640, 320);
var gone = sized('gone', 1, 1, 600, 320); gone.isDisposed = function () { return true; };
[same, wider, gone, null].forEach(function (c) { QBCharts.fit(c); });
JSON.stringify(resizes);
"""
        assert json.loads(_run("portal_chat.html", "en", script)) == ["wider"]

    def test_the_dashboard_resizes_only_the_tiles_whose_box_changed(self):
        script = self.CHARTS + """
var charts = [sized('a', 514, 300, 514, 300), sized('b', 514, 300, 700, 300)];
var nodes = [charts[0].el, charts[1].el, {id: 'no-chart'}];
var echarts = window.echarts = {getInstanceByDom: function (n) {
  return charts.filter(function (c) { return c.el === n; })[0] || null; }};
var document = {querySelectorAll: function () { return nodes; }};
""" + lift(DASHBOARD, "function resizeDashboardCharts()") + """
resizeDashboardCharts();
JSON.stringify(resizes);
"""
        assert json.loads(_run("portal_dashboard.html", "en", script)) == ["b"]

    def test_the_chat_fits_a_chart_when_its_tab_is_shown(self):
        # Drawn while its tab was hidden, the chart has no size; shown, it
        # takes the box. A tab shown again without a change of box is left be.
        script = self.CHARTS + """
var hidden = sized('artifact-chart-2', 0, 0, 619, 494);
var shown = sized('artifact-chart-3', 619, 494, 619, 494);
var current = null;
var echarts = window.echarts = {getInstanceByDom: function () { return current; }};
var document = {getElementById: function () { return current && current.el; }};
var setTimeout = function (fn) { fn(); };
var _activeArtifactChartId = 'artifact-chart';
""" + lift(CHAT, "function _artifactTabShown(tabName)") + """
current = hidden; _artifactTabShown('visual');
current = shown; _artifactTabShown('visual');
current = hidden; _artifactTabShown('query');
JSON.stringify(resizes);
"""
        assert json.loads(_run("portal_chat.html", "en", script)) == ["artifact-chart-2"]


class TestTheMotion:

    def test_the_entrance_is_long_enough_to_see_and_a_change_is_quicker(self):
        motion = drawn(bars(), "{on: opt.animation, entrance: opt.animationDuration, change: opt.animationDurationUpdate}")
        assert motion["on"] is True
        # Was 300ms, over before the pane had finished opening.
        assert 400 <= motion["entrance"] <= 1000
        assert motion["change"] < motion["entrance"]

    def test_bars_rise_one_after_another(self):
        delays = drawn(bars(), "[0, 1, 2, 3, 200].map(function (i) { return opt.series[0].animationDelay(i) })")
        assert delays[0] == 0 and delays[0] < delays[1] < delays[2] < delays[3]
        # A long ranking's last bar does not wait out the whole entrance.
        entrance = drawn(bars(), "opt.animationDuration")
        assert delays[4] <= entrance / 2

    def test_panel_bars_rise_the_same_way(self):
        rows = [{"RGN_NM": r, "NET_SLS_AMT": s, "GRS_MRGN_PCT": m}
                for r, s, m in zip(REGIONS, (1_200_000.0, 800_000.0, 300_000.0, 650_000.0), (31.2, 27.5, 22.1, 29.0))]
        panels = build_chart_payload(rows, None, question="net sales and gross margin percent by region")
        assert drawn(panels, "opt.series.map(function (s) { return s.animationDelay(2) > s.animationDelay(0) })") \
            == [True, True]

    def test_a_reader_who_asked_for_less_motion_sees_none(self):
        asked = """window.matchMedia = function (q) { return {matches: q.indexOf('reduce') >= 0}; };
var opt = QBCharts.buildOption(PAYLOAD);
JSON.stringify(opt.animation);"""
        assert json.loads(_run("portal_chat.html", "en", asked.replace("PAYLOAD", json.dumps(bars())))) is False
        broken = """window.matchMedia = function () { throw new Error('no media queries here'); };
var opt = QBCharts.buildOption(PAYLOAD);
JSON.stringify(opt.animation);"""
        assert json.loads(_run("portal_chat.html", "en", broken.replace("PAYLOAD", json.dumps(bars())))) is True


class TestAClickIsAnnounced:

    HINT = {"en": "Click to break this down", "fr": "Cliquez pour ventiler"}

    @pytest.mark.parametrize("lang", ["en", "fr"])
    def test_a_bar_says_a_click_breaks_it_down(self, lang):
        tip = drawn(bars(), "opt.tooltip.formatter({name: 'North', value: 1200, color: '#000'})",
                    lang=lang, layout="{drill: true}")
        assert self.HINT[lang] in tip

    @pytest.mark.parametrize("payload", [grouped_bars, trend, ranking_in_panels], ids=lambda f: f.__name__)
    def test_every_chart_a_click_drills_says_so(self, payload):
        built = payload()
        point = [{"axisValue": built["rows"][0][built["x_key"]], "seriesName": built["y_keys"][0],
                  "value": 1, "color": "#000"}]
        tip = drawn(built, f"opt.tooltip.formatter({json.dumps(point)})", layout="{drill: true}")
        assert self.HINT["en"] in tip
        # Where the page does not answer a click, the tooltip does not offer one.
        assert self.HINT["en"] not in drawn(built, f"opt.tooltip.formatter({json.dumps(point)})")

    def test_a_pie_does_not_offer_a_click_it_cannot_answer(self):
        rows = [{"CHNL_NM": c, "NET_SLS_AMT": v} for c, v in (("Retail", 520.0), ("Online", 310.0), ("Trade", 152.0))]
        pie = build_chart_payload(rows, None, question="share of net sales by channel")
        assert pie["chart_type"] == "pie"
        tip = drawn(pie, "opt.tooltip.formatter({name: 'Retail', value: 520, percent: 52.9, color: '#000'})",
                    layout="{drill: true}")
        assert self.HINT["en"] not in tip

    def test_the_chat_offers_the_click_exactly_where_it_answers_one(self):
        script = (
            "var window = {echarts: {}, ResizeObserver: function () {}, _chartPayloads: {}};\n"
            "var asked = null, wired = false;\n"
            "window.QBCharts = {render: function (el, payload, opts) { asked = opts;"
            " return {on: function (event) { if (event === 'click') wired = true; }}; }};\n"
            "function t(id) { return id; }\nfunction escHtml(s) { return s; }\n"
            "function sendChartDrill() {}\n"
            + lift(CHAT, "function _chartDrillable(payload)") + "\n"
            + lift(CHAT, "function _wireChartDrillClick(chart, payload)") + "\n"
            + lift(CHAT, "function renderChartInto(chartEl, payload)") + "\n"
            + """
var out = {};
['bar', 'line', 'area', 'pie', 'donut', 'scatter', 'heatmap'].forEach(function (type) {
  asked = null; wired = false;
  renderChartInto({}, {chart_type: type});
  out[type] = [Boolean(asked && asked.drill), wired];
});
JSON.stringify(out);
""")
        out = json.loads(dukpy.evaljs(script))
        for kind, (hinted, wired) in out.items():
            assert hinted == wired, kind
        assert out["bar"] == [True, True] and out["pie"] == [False, False]


def heatmap_rows(warehouses=16, months=12):
    return [{"PRD_KEY": 202401 + m, "WHS_NM": f"Warehouse {w:02d}", "ON_HND_QTY": 500 + m * 20 + w * 35}
            for m in range(months) for w in range(warehouses)]


class TestARedrawKeepsItsBox:

    def test_update_draws_for_the_box_the_chart_is_in(self):
        heatmap = build_chart_payload(heatmap_rows(), None, question="stock on hand by warehouse by month")
        script = f"""
var calls = [];
var tile = {{clientWidth: 420, clientHeight: 260, style: {{}}}};
var chart = {{getDom: function () {{ return tile; }},
             setOption: function (o, notMerge) {{ calls.push({{cells: o.series[0].label.show, notMerge: notMerge}}); }}}};
QBCharts.update(chart, {json.dumps(heatmap)}, {{grow: false}});
tile.clientWidth = 900; tile.clientHeight = 700;
QBCharts.update(chart, {json.dumps(heatmap)}, {{grow: false}});
JSON.stringify(calls);
"""
        # A 420 x 260 tile has no room for a value in each of 192 cells; the
        # same chart given 900 x 700 does. Redrawn in place (notMerge), so the
        # instance -- and its click handler -- is kept.
        assert json.loads(_run("portal_dashboard.html", "en", script)) == [
            {"cells": False, "notMerge": True}, {"cells": True, "notMerge": True}]

    def _dashboard(self, lang, body):
        script = (
            "var updates = [], saved = [], badge = {textContent: 'BAR'};\n"
            "function button(kind, key) { return {dataset: kind === 'type' ? {type: key} : {palette: key},"
            " active: false, classList: {toggle: function (cls, on) { this.owner.active = on; }}}; }\n"
            "var typeButtons = ['bar', 'line'].map(function (k) { var b = button('type', k); b.classList.owner = b; return b; });\n"
            "var swatches = ['default', 'warm'].map(function (k) { var b = button('pal', k); b.classList.owner = b; return b; });\n"
            "var node = {id: 'dashboard-chart-7'}, inst = {id: 'inst-7'};\n"
            "var echarts = window.echarts = {getInstanceByDom: function (n) { return n === node ? inst : null; }};\n"
            "window.QBCharts.update = function (chart, payload, opts) { updates.push([chart.id, payload, opts]); };\n"
            "window._dashPayloads = {7: {chart_type: 'bar', color_palette: 'default', rows: [{a: 1}]}};\n"
            "var document = {getElementById: function (id) {"
            " return id === 'dashboard-chart-7' ? node : (id === 'chart-type-badge-7' ? badge : null); },"
            " querySelector: function () { return {querySelectorAll: function (sel) {"
            " return sel === '.dctt-btn' ? typeButtons : swatches; }}; }};\n"
            "function _updateChart(id, fields) { saved.push([id, fields]); }\n"
            + lift(DASHBOARD, "function t(id, vars)") + "\n"
            + lift(DASHBOARD, "function _redrawCard(chartId)") + "\n"
            + lift(DASHBOARD, "function _setTypeBadge(chartId, type)") + "\n"
            + lift(DASHBOARD, "function _syncCardFromModal(chartId, fields)") + "\n"
            + body)
        return json.loads(_run("portal_dashboard.html", lang, script))

    @pytest.mark.parametrize("lang,label", [("en", "LINE"), ("fr", "COURBE")])
    def test_a_type_chosen_in_the_expanded_view_reaches_the_card(self, lang, label):
        out = self._dashboard(lang, """
_syncCardFromModal(7, {chart_type: 'line'});
JSON.stringify({updates: updates, badge: badge.textContent, saved: saved,
                active: typeButtons.filter(function (b) { return b.active; }).map(function (b) { return b.dataset.type; })});
""")
        # The card's own chart is redrawn in the new type, for its own box.
        assert len(out["updates"]) == 1
        chart, payload, opts = out["updates"][0]
        assert (chart, payload["chart_type"], opts) == ("inst-7", "line", {"grow": False})
        # Its badge names the type in the reader's language, not "LINE" in a
        # French interface; its buttons agree; the choice is saved.
        assert out["badge"] == label
        assert out["active"] == ["line"]
        assert out["saved"] == [[7, {"chart_type": "line"}]]

    def test_a_palette_chosen_in_the_expanded_view_reaches_the_card(self):
        out = self._dashboard("en", """
_syncCardFromModal(7, {color_palette: 'warm'});
JSON.stringify({updates: updates, badge: badge.textContent,
                active: swatches.filter(function (b) { return b.active; }).map(function (b) { return b.dataset.palette; })});
""")
        assert out["updates"][0][1]["color_palette"] == "warm"
        assert out["updates"][0][1]["chart_type"] == "bar"        # the type is kept
        assert out["badge"] == "BAR"                              # and so is its badge
        assert out["active"] == ["warm"]

    # The two places a reader picks a type: the card's own buttons, and the
    # buttons in its expanded view. Executed through the page's own builders,
    # mountDashChartControls and openChartModal, with the reader's click
    # delivered to the handler each one wires.
    CONTROLS = """
function stubButton(data) {
  var b = {dataset: data, handlers: {}, active: false,
           classList: {toggle: function (cls, on) { b.active = on; }},
           addEventListener: function (ev, fn) { b.handlers[ev] = fn; }};
  return b;
}
function row() {
  var types = ['bar', 'line'].map(function (k) { return stubButton({type: k}); });
  var swatches = ['default', 'warm'].map(function (k) { return stubButton({palette: k}); });
  return {dataset: {chartId: '7'}, innerHTML: '', types: types, swatches: swatches,
          querySelectorAll: function (sel) { return sel === '.dctt-btn' ? types : swatches; }};
}
var DASH_TYPES = ['bar', 'line', 'area', 'pie', 'donut', 'scatter'];
var _PALETTES = {default: ['#111', '#222', '#333'], warm: ['#444', '#555', '#666']};
var requestAnimationFrame = function (fn) { fn(); };
"""

    def _pick(self, lang, where):
        payload = dict(bars(), chart_id=7)
        body = self.CONTROLS + f"""
window._dashPayloads[7] = {json.dumps(payload)};
var cardRow = row(), modalRow = row();
var modal = {{style: {{}}, addEventListener: function () {{}}}};
var modalChart = {{id: 'modal-chart', resize: function () {{}}}};
var elements = {{'dashboard-chart-7': node, 'chart-type-badge-7': badge, 'chart-modal': modal,
                'chart-modal-title': {{textContent: ''}}, 'chart-modal-ctrl': modalRow,
                'chart-modal-canvas': {{clientWidth: 900, clientHeight: 600, style: {{}}}}}};
document.getElementById = function (id) {{ return elements[id] || null; }};
document.querySelector = function () {{ return cardRow; }};
window.QBCharts.render = function () {{ return modalChart; }};
if ({json.dumps(where)} === 'card') {{
  mountDashChartControls(cardRow);
  cardRow.types[1].handlers.click();
}} else {{
  openChartModal({{dataset: {{chartId: '7'}}, querySelector: function (sel) {{
    return sel === '[data-chart]' ? node : {{textContent: 'Net sales'}}; }}}});
  modalRow.types[1].handlers.click();
}}
JSON.stringify({{updates: updates.map(function (u) {{ return [u[0], u[1].chart_type, u[2]]; }}),
                badge: badge.textContent, saved: saved,
                cardActive: cardRow.types.filter(function (b) {{ return b.active; }}).map(function (b) {{ return b.dataset.type; }})}});
"""
        return self._dashboard(lang, lift(DASHBOARD, "function escHtmlDash(value)") + "\n"
                               + lift(DASHBOARD, "function mountDashChartControls(ctrlRow)") + "\n"
                               + lift(DASHBOARD, "function openChartModal(cardEl)") + "\n" + body)

    @pytest.mark.parametrize("lang,label", [("en", "LINE"), ("fr", "COURBE")])
    def test_a_type_picked_on_the_card_redraws_it_in_that_type(self, lang, label):
        out = self._pick(lang, "card")
        assert out["updates"] == [["inst-7", "line", {"grow": False}]]
        assert out["badge"] == label
        assert out["saved"] == [[7, {"chart_type": "line"}]]

    @pytest.mark.parametrize("lang,label", [("en", "LINE"), ("fr", "COURBE")])
    def test_a_type_picked_in_the_expanded_view_redraws_the_card_behind_it(self, lang, label):
        out = self._pick(lang, "modal")
        # The expanded chart, then the card behind it -- which kept its old
        # type before, under a badge written to an attribute nothing reads.
        assert out["updates"] == [["modal-chart", "line", {"grow": False}], ["inst-7", "line", {"grow": False}]]
        assert out["badge"] == label
        assert out["cardActive"] == ["line"]
        assert out["saved"] == [[7, {"chart_type": "line"}]]

    def test_the_expanded_chart_is_sized_to_the_room_it_has(self):
        assert expanded_chart_resizes() == 1

    def test_a_card_with_no_chart_yet_is_left_alone(self):
        out = self._dashboard("en", """
echarts.getInstanceByDom = function () { return null; };
_redrawCard(7);
JSON.stringify(updates.length);
""")
        assert out == 0


def expanded_chart_resizes() -> int:
    """How many times the expanded view's chart is resized as it opens.

    The chart is drawn as the modal appears, before the modal has its full
    size; once laid out, it must be sized to the room it has. Executed
    through the page's own openChartModal, the chart in a box it has not been
    sized to yet.
    """
    test = TestARedrawKeepsItsBox()
    script = test.CONTROLS + f"""
window._dashPayloads[7] = {json.dumps(dict(bars(), chart_id=7))};
var resizes = 0, modalRow = row();
var canvas = {{clientWidth: 900, clientHeight: 600, style: {{}}}};
var modalChart = {{id: 'modal-chart', getDom: function () {{ return canvas; }},
                  getWidth: function () {{ return 300; }}, getHeight: function () {{ return 200; }},
                  resize: function () {{ resizes += 1; }}}};
var elements = {{'chart-modal': {{style: {{}}, addEventListener: function () {{}}}}, 'chart-modal-title': {{textContent: ''}},
                'chart-modal-ctrl': modalRow, 'chart-modal-canvas': canvas}};
document.getElementById = function (id) {{ return elements[id] || null; }};
window.QBCharts.render = function () {{ return modalChart; }};
openChartModal({{dataset: {{chartId: '7'}}, querySelector: function (sel) {{
  return sel === '[data-chart]' ? node : {{textContent: 'Net sales'}}; }}}});
JSON.stringify(resizes);
"""
    return test._dashboard("en", lift(DASHBOARD, "function escHtmlDash(value)") + "\n"
                           + lift(DASHBOARD, "function openChartModal(cardEl)") + "\n" + script)
