# -*- coding: utf-8 -*-
"""
tests/test_chart_annotation_language.py

The words drawn onto a chart, in the reader's language.

A chart label is the least forgiving copy in the product. Body text sits in a
paragraph the reader takes as a whole, and one English word inside a French
sentence at least has a sentence around it. "Drop" baked into a canvas label
sits alone, four characters wide, beside a number -- there is nothing next to
it to say what language the page was meant to be in, and it cannot be selected,
searched or inspected. The same goes for "23.4%" on a French page: the reader
reads the dot as nothing at all and the comma, when it appears elsewhere, as
the decimal point.

Both pages build these charts from their own copy of the code, so every
assertion here runs against BOTH portal_chat.html and portal_dashboard.html.
The two drifted before -- the chat page translated the annotation's tooltip
name and left the on-canvas label English, which is exactly the half-done state
reading the diff would have called finished.

Nothing here asserts on source text: the real functions are lifted out of the
real templates and EXECUTED, and the assertions are on what they return.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import i18n
from tests.js_lift import function as lift

dukpy = pytest.importorskip(
    "dukpy",
    reason="a JavaScript engine is required to EXECUTE the chart label code; "
           "reading it and assuming what it draws is the drift this file "
           "exists to catch",
)

ROOT = Path(__file__).resolve().parents[1]
SHELL = (ROOT / "portal" / "templates" / "portal_base.html").read_text(encoding="utf-8")
PAGES = ("portal_chat.html", "portal_dashboard.html")

# The theme is a DOM boundary (getComputedStyle on the document root), not the
# code under test. Stubbed with the shape the real one returns; colours are
# irrelevant to which language a word is in.
_STATUS_STUB = """
window.QB_CHART_STATUS = function () {
  return {color:{drop:'#a00',gain:'#0a0'}, label:{drop:'#fff',gain:'#fff'},
          background:{drop:'#111',gain:'#111'}, border:{drop:'#222',gain:'#222'},
          ring:'#333', shadow:{drop:'#444',gain:'#555'}};
};
"""


def _page(name: str) -> str:
    return (ROOT / "portal" / "templates" / name).read_text(encoding="utf-8")


def _shell_preamble(lang: str) -> str:
    """The real shell helpers, in scope, for one language."""
    return f"""
var window = {{}};
window.QB_LANG = {json.dumps(lang)};
window.QB_I18N = {json.dumps(i18n.catalogue_for(lang))};
window.QB_NUM = {json.dumps(i18n.number_format(lang))};
{lift(SHELL, "window.qbT = function (id, vars)")}
{lift(SHELL, "window.qbNum = function (value, options)")}
{lift(SHELL, "window.qbPct = function (value, digits)")}
var I18N = window.QB_I18N;
"""


def _run(page: str, lang: str, functions: tuple[str, ...], expression: str):
    src = _page(page)
    lifted = "\n".join(lift(src, sig) for sig in functions)
    return dukpy.evaljs(
        _shell_preamble(lang)
        + lift(src, "function t(id, vars)")
        + _STATUS_STUB
        + lifted
        + "\n"
        + expression
    )


ANNOTATED = {
    "annotations": {
        "biggest_period_drop": {"period": "Mar", "pct_change": -23.45},
        "biggest_period_gain": {"period": "Jan", "pct_change": 12.3},
    }
}


def _mark_points(page: str, lang: str):
    expression = (
        "JSON.stringify(_buildAnnotationMarkPoints("
        f"{json.dumps(ANNOTATED)}, ['Jan','Feb','Mar'], [10,20,30]"
        ").map(function (p) {"
        "  return {name: p.name, formatter: p.label.formatter, value: p.value};"
        "}))"
    )
    return json.loads(
        _run(page, lang, ("function _buildAnnotationMarkPoints(payload",), expression)
    )


class TestTheAnnotationLabelIsDrawnInTheReadersLanguage:
    """The ↓ -23,4 % Baisse that ends up on the canvas."""

    @pytest.mark.parametrize("page", PAGES)
    def test_the_english_label_carries_the_english_words(self, page):
        drop, gain = _mark_points(page, "en")
        assert drop["formatter"] == "↓ -23.4% Drop", page
        assert gain["formatter"] == "↑ +12.3% Gain", page

    @pytest.mark.parametrize("page", PAGES)
    def test_the_french_label_carries_the_french_words(self, page):
        # The expected percentage comes from core.i18n rather than being typed
        # here: French separates the sign with U+00A0 and groups with U+202F,
        # and a hand-typed expectation quietly uses an ordinary space, then
        # fails for a reason that has nothing to do with the product.
        drop, gain = _mark_points(page, "fr")
        down = i18n.MESSAGES["ui.chart.drop"]["fr"]
        up = i18n.MESSAGES["ui.chart.gain"]["fr"]
        assert drop["formatter"] == (
            "\u2193 " + i18n.format_percent(-23.45, 1, lang="fr") + " " + down), page
        assert gain["formatter"] == (
            "\u2191 +" + i18n.format_percent(12.3, 1, lang="fr") + " " + up), page

    @pytest.mark.parametrize("page", PAGES)
    def test_no_english_word_survives_into_the_french_label(self, page):
        # The state this file was written for: the tooltip name translated,
        # the drawn label left in English.
        for point in _mark_points(page, "fr"):
            assert "Drop" not in point["formatter"], (page, point)
            assert "Gain" not in point["formatter"], (page, point)
            assert "Biggest" not in point["name"], (page, point)

    @pytest.mark.parametrize("page", PAGES)
    def test_the_tooltip_name_is_translated_too(self, page):
        drop, gain = _mark_points(page, "fr")
        assert drop["name"] == i18n.MESSAGES["ui.chat.chart.biggest_drop"]["fr"], page
        assert gain["name"] == i18n.MESSAGES["ui.chat.chart.biggest_gain"]["fr"], page

    @pytest.mark.parametrize("page", PAGES)
    def test_the_percentage_is_written_in_french_notation(self, page):
        drop, gain = _mark_points(page, "fr")
        assert drop["value"] == i18n.format_percent(-23.45, 1, lang="fr"), page
        assert gain["value"] == "+" + i18n.format_percent(12.3, 1, lang="fr"), page
        # Spelled out once, so a reader of this file can see what that means:
        # a comma for the decimal point, and a no-break space before the sign.
        assert drop["value"] == "-23,4\u00a0%", (page, drop["value"])
        assert "." not in drop["value"], (page, drop["value"])

    @pytest.mark.parametrize("page", PAGES)
    def test_both_pages_draw_the_same_label(self, page):
        # They are separate copies of this code. A fix applied to one and not
        # the other is the failure mode; comparing them is the check.
        assert _mark_points("portal_chat.html", "fr") == \
               _mark_points("portal_dashboard.html", "fr")
        assert _mark_points("portal_chat.html", "en") == \
               _mark_points("portal_dashboard.html", "en")

    @pytest.mark.parametrize("page", PAGES)
    def test_an_unannotated_payload_draws_nothing(self, page):
        expression = ("JSON.stringify(_buildAnnotationMarkPoints("
                      "{}, ['Jan'], [1]))")
        assert json.loads(_run(
            page, "fr", ("function _buildAnnotationMarkPoints(payload",), expression
        )) == []

    @pytest.mark.parametrize("page", PAGES)
    def test_a_period_the_chart_does_not_plot_is_skipped(self, page):
        # The annotation names a period; if it is not on the axis there is no
        # x-position to anchor to and drawing it would put the label on the
        # wrong bar.
        payload = {"annotations": {"biggest_period_drop":
                                   {"period": "Dec", "pct_change": -5.0}}}
        expression = ("JSON.stringify(_buildAnnotationMarkPoints("
                      f"{json.dumps(payload)}, ['Jan','Feb'], [1,2]))")
        assert json.loads(_run(
            page, "fr", ("function _buildAnnotationMarkPoints(payload",), expression
        )) == []


class TestTheColumnLabelFallbackIsTranslated:
    """Axis titles when the payload names no column."""

    def _label(self, page, lang, col, fallback=None):
        arg = "null" if fallback is None else json.dumps(fallback)
        expression = (f"_chartColumnLabel({{}}, {json.dumps(col)}, "
                      f"{arg} === null ? undefined : {arg})")
        return _run(page, lang,
                    ("function _chartNormKey(", "function _chartColumnLabel(payload"),
                    expression)

    @pytest.mark.parametrize("page", PAGES)
    def test_a_missing_column_falls_back_to_the_french_word(self, page):
        assert self._label(page, "fr", "") == i18n.MESSAGES["ui.chart.value"]["fr"]
        assert self._label(page, "en", "") == "Value"

    @pytest.mark.parametrize("page", PAGES)
    def test_a_supplied_fallback_is_used_verbatim(self, page):
        # Not title-cased word by word: "Non Précisé" is not how French
        # writes a phrase, and the prettifier is for identifiers.
        assert self._label(page, "fr", "", "part du total") == "part du total"

    @pytest.mark.parametrize("page", PAGES)
    def test_a_real_column_name_is_still_prettified(self, page):
        # The fallback change must not cost the identifier formatting that
        # every axis with a real column depends on.
        assert self._label(page, "fr", "total_revenue") == "Total Revenue"
        assert self._label(page, "en", "net-sales") == "Net Sales"


class TestTheBrowserWritesPercentagesLikePython:
    """
    window.qbPct against core.i18n.format_percent.

    One rule, two implementations, because the browser builds labels the server
    never sees. Two implementations of one rule is the shape that drifts
    silently -- a French chart reading "23.4%" looks like ordinary output. So
    both are EXECUTED and compared.
    """

    # Includes the values that separate a REAL tie from one that only looks
    # like a tie in decimal. 0.45 is stored just ABOVE 0.45, so it rounds up
    # to 0.5; a tie test that reads only one extra digit sees the "5", calls
    # it a tie, finds an even 4 before it and truncates to 0.4 -- the right
    # answer arrived at from the wrong place is what makes that bug survive.
    VALUES = (0, 1, -1, 0.5, 1.5, 2.5, 3.5, 12.3, -23.45, 99.99, 100, 1234.5,
              -0.04, 0.45, 0.65, 0.85, 0.25, 0.25000001, 2.6750001, 0.05)

    @pytest.mark.parametrize("lang", ("en", "fr"))
    @pytest.mark.parametrize("digits", (0, 1, 2))
    def test_every_value_matches_python(self, lang, digits):
        expression = (
            "JSON.stringify(" + json.dumps(list(self.VALUES)) +
            f".map(function (v) {{ return window.qbPct(v, {digits}); }}))"
        )
        got = json.loads(dukpy.evaljs(_shell_preamble(lang) + expression))
        want = [i18n.format_percent(v, digits, lang=lang) for v in self.VALUES]
        assert got == want, (lang, digits)

    def test_the_default_is_one_digit(self):
        assert dukpy.evaljs(_shell_preamble("fr") + "window.qbPct(12.34)") == \
            i18n.format_percent(12.34, 1, lang="fr")

    def test_french_uses_a_comma_and_a_gap_and_english_does_not(self):
        # Written out rather than derived, so this file states the contract
        # somewhere instead of only comparing two implementations of it.
        # 23.45 sits just below its own decimal literal in binary, so both
        # languages round it down -- a fact about floats, not about French.
        assert dukpy.evaljs(_shell_preamble("fr") + "window.qbPct(23.45, 1)") == "23,4\u00a0%"
        assert dukpy.evaljs(_shell_preamble("en") + "window.qbPct(23.45, 1)") == "23.4%"
        assert dukpy.evaljs(_shell_preamble("fr") + "window.qbPct(1234.5, 0)") \
            == "1\u202f234\u00a0%"

    def test_a_value_that_only_looks_like_a_tie_is_not_treated_as_one(self):
        # 0.45 is stored as 0.4500000000000000111..., so it is above the tie
        # and rounds up. Deciding tie-ness from the one digit after the cut
        # gets this wrong in the quietest possible way.
        assert dukpy.evaljs(_shell_preamble("en") + "window.qbNum(0.45, {min:1, max:1})") == "0.5"
        assert dukpy.evaljs(_shell_preamble("en") + "window.qbNum(0.25000001, {min:1, max:1})") == "0.3"
        # And 23.45, stored just BELOW, rounds down for the same reason.
        assert dukpy.evaljs(_shell_preamble("en") + "window.qbNum(23.45, {min:1, max:1})") == "23.4"

    def test_a_tie_rounds_the_way_the_server_rounds_it(self):
        # Pre-existing drift this file found: toFixed rounds a tie away from
        # zero and Python rounds it to even, so a 2.5% share read "3%" on the
        # chart and "2%" in the sentence under it, on the same page.
        for value, expected in ((0.5, "0%"), (1.5, "2%"), (2.5, "2%"), (3.5, "4%")):
            got = dukpy.evaljs(_shell_preamble("en") + f"window.qbPct({value}, 0)")
            assert got == expected, (value, got)
            assert got == i18n.format_percent(value, 0, lang="en"), value


class TestTheCatalogueCoversTheChartWords:

    # Chart words that are genuinely the same in French. Listed here rather
    # than pattern-matched, so the next copy-pasted "translation" still fails.
    IDENTICAL_BY_DESIGN = {
        # A box plot's whiskers. French abbreviates maximum and minimum the
        # same way; its median ("Méd") and its mean ("Moyenne") do not, which
        # is why only these two are here.
        "ui.chart.box.max",
        "ui.chart.box.min",
    }

    def test_every_chart_id_has_a_distinct_french_form(self):
        chart_ids = [k for k in i18n.MESSAGES if k.startswith("ui.chart.")]
        assert chart_ids, "the ui.chart.* namespace is empty"
        identical = set()
        for msg_id in chart_ids:
            entry = i18n.MESSAGES[msg_id]
            assert entry.get("fr"), msg_id
            if entry["en"] == entry["fr"]:
                identical.add(msg_id)
        assert identical == self.IDENTICAL_BY_DESIGN, (
            f"reads the same in both languages: "
            f"{sorted(identical - self.IDENTICAL_BY_DESIGN)}; "
            f"now translated, drop from the allowlist: "
            f"{sorted(self.IDENTICAL_BY_DESIGN - identical)}"
        )

    def test_the_ids_the_pages_ask_for_all_exist(self):
        # A t() call for an id that is not in the catalogue renders the id
        # itself onto the chart -- "ui.chart.drop" drawn on a canvas.
        import re
        asked = set()
        for page in PAGES:
            asked.update(re.findall(r"t\(\s*'(ui\.chart\.[a-z_.]+)'", _page(page)))
            asked.update(re.findall(r"'(ui\.chart\.[a-z_.]+)'\s*:", _page(page)))
            asked.update(re.findall(r"\?\s*'(ui\.chart\.[a-z_.]+)'", _page(page)))
            asked.update(re.findall(r":\s*'(ui\.chart\.[a-z_.]+)'", _page(page)))
        assert asked, "no chart ids are asked for by either page"
        missing = sorted(asked - set(i18n.MESSAGES))
        assert not missing, missing


# ── The whole option builder ──────────────────────────────────────────────────
#
# The pie, histogram and cartesian branches build their labels from locals
# (xLabel, yLabel) that nothing else exposes, and their formatters are
# closures on the option object. Both are only observable by building a real
# option and CALLING the formatter it returns -- which is what happens below.
#
# It costs a dependency list per page. That is the price of testing the code
# that runs instead of the code that is easy to reach, and it caught a fallback
# that a scan for English words would have had to name a literal to find.

_ECHARTS_STUB = """
var echarts = { graphic: { LinearGradient: function (x, y, x2, y2, stops) {
  this.type = 'linear'; this.colorStops = stops;
} } };
window.QB_CHART_THEME = function () {
  return {axis:'#888', split:'#eee', axisLine:'#ccc', tooltipBg:'#fff',
          tooltipText:'#000', surface:'#fff'};
};
window.QB_HEATMAP_RAMP = ['#fff', '#000'];
"""

_PALETTES_JS = (ROOT / "static" / "js" / "chart-palettes.js").read_text(encoding="utf-8")


def _palette_literal(name: str) -> str:
    from tests.js_lift import const_block
    block = const_block(
        _PALETTES_JS.replace(f"window.{name} =", f"const {name} ="), name)
    return block[block.index("{"):]


# The functions each page's option builder closes over, in dependency order.
_BUILDER_DEPS = {
    "portal_chat.html": (
        "function escHtml(str)",
        "function _qbMoney(body, symbol)",
        "function _fmtNum(v)",
        "function _chartNormKey(v)",
        "function _chartFormatFor(payload, col)",
        "function _chartColumnLabel(payload, col, fallback)",
        "function _fmtChartValue(v, fmt, compact",
        "function _chartNumber(v)",
        "function _buildAnnotationMarkPoints(payload",
        "function buildChartOption(payload)",
    ),
    "portal_dashboard.html": (
        # Two of these were missing, and it was invisible: every payload the
        # file happened to build for the dashboard took the pie or funnel
        # branch, which does not reach them. A bar or a line raised
        # "ReferenceError: isTemporalLabel is not defined" -- so the promise in
        # this module's docstring, that every assertion runs against BOTH
        # pages, held only for the branches nobody had written a cartesian
        # test for. TestBothPagesCanDrawACartesianChart below is the guard.
        "function isTemporalLabel(v)",
        "function truncateLabel(l, n=16)",
        "function escHtmlDash(value)",
        "function _fmtNum(v)",
        "function _qbMoney(body, symbol)",
        "function _chartNormKey(v)",
        "function _chartFormatFor(payload, col)",
        "function _chartColumnLabel(payload, col, fallback)",
        "function _chartEscHtml(value)",
        "function _fmtChartValue(v, fmt, compact",
        "function _chartNumber(v)",
        "function renderChartWarnings(payload)",
        "function _buildAnnotationMarkPoints(payload",
        "function buildDashboardOption(payload)",
    ),
}
_BUILDER_ENTRY = {
    "portal_chat.html": "buildChartOption",
    "portal_dashboard.html": "buildDashboardOption",
}


def _build(page: str, lang: str, payload: dict, expression: str):
    """Build a real chart option for `payload`, then evaluate `expression`."""
    src = _page(page)
    body = "\n".join([
        _shell_preamble(lang),
        lift(src, "function t(id, vars)"),
        _STATUS_STUB,
        _ECHARTS_STUB,
        "window.QB_PALETTES = " + _palette_literal("QB_PALETTES") + ";",
        "window.QB_PALETTE_GRADIENTS = "
        + _palette_literal("QB_PALETTE_GRADIENTS") + ";",
        "var _PALETTES = window.QB_PALETTES;",
        "var _PAL_GRAD = window.QB_PALETTE_GRADIENTS;",
        # The dashboard binds the theme accessor to a module-level name.
        "var chartThemeTokens = window.QB_CHART_THEME;",
        *[lift(src, sig) for sig in _BUILDER_DEPS[page]],
        f"var opt = {_BUILDER_ENTRY[page]}({json.dumps(payload)});",
        expression,
    ])
    return dukpy.evaljs(body)


PIE = {"rows": [{"m": "Jan", "v": 10}, {"m": None, "v": 30}],
       "x_key": "m", "y_keys": ["v"], "chart_type": "pie"}
PIE_NO_X = {"rows": [{"v": 10}, {"v": 30}],
            "x_key": None, "y_keys": ["v"], "chart_type": "pie"}


BAR = {"rows": [{"WHS_NM": "Halifax", "REVENUE_AMT": 10.0},
                {"WHS_NM": "Calgary", "REVENUE_AMT": 20.0}],
       "x_key": "WHS_NM", "y_keys": ["REVENUE_AMT"], "chart_type": "bar"}


class TestBothPagesCanDrawACartesianChartAtAll:
    """The guard for this module's own promise.

    Its docstring says every assertion here runs against BOTH pages. That held
    only for the branches the fixtures happened to reach: every payload in the
    file was a pie, a funnel or an annotation set, and none of those touches
    the dashboard's cartesian code. Two helpers it needs were missing from
    _BUILDER_DEPS, so a bar or a line raised

        ReferenceError: isTemporalLabel is not defined

    -- and nothing failed, because nothing asked. A harness gap does not
    announce itself; it just quietly narrows what the suite covers.

    These two tests are cheap and they fail loudly the moment either page's
    cartesian branch stops being executable here.
    """

    @pytest.mark.parametrize("page", PAGES)
    def test_a_bar_chart_builds(self, page):
        drawn = json.loads(_build(
            page, "en", BAR,
            "JSON.stringify({n: opt.series.length,"
            " axis: (opt.xAxis.data || opt.xAxis[0].data)})"))
        assert drawn["n"] == 1, page
        assert drawn["axis"] == ["Halifax", "Calgary"], page

    @pytest.mark.parametrize("page", PAGES)
    def test_a_line_chart_builds(self, page):
        # A second cartesian type, because the two pages branch on the type
        # name and a helper reached only by one of them would slip through.
        line = {**BAR, "chart_type": "line"}
        assert json.loads(_build(
            page, "en", line, "JSON.stringify(opt.series.length)")) == 1, page


class TestThePieSliceIsLabelledInTheReadersLanguage:
    """Built by the real builder; the formatter it returns is then called."""

    @pytest.mark.parametrize("page", PAGES)
    def test_a_row_with_no_category_says_so_in_french(self, page):
        data = json.loads(_build(page, "fr", PIE, "JSON.stringify(opt.series[0].data)"))
        assert data[1]["name"] == i18n.MESSAGES["ui.chart.unspecified"]["fr"], page
        english = json.loads(_build(page, "en", PIE, "JSON.stringify(opt.series[0].data)"))
        assert english[1]["name"] == "Unspecified", page

    @pytest.mark.parametrize("page", PAGES)
    def test_the_tooltip_says_share_of_total_in_french(self, page):
        html = _build(page, "fr", PIE,
                      "opt.tooltip.formatter({name:'Jan', value:30, percent:75})")
        assert i18n.MESSAGES["ui.chart.share_of_total"]["fr"] in html, html
        assert "Share of total" not in html, html

    @pytest.mark.parametrize("page", PAGES)
    def test_every_percentage_on_the_pie_is_in_french_notation(self, page):
        for expression in (
            "opt.tooltip.formatter({name:'Jan', value:30, percent:75.5})",
            "opt.series[0].label.formatter({name:'Jan', value:30, percent:75.5})",
            "opt.legend.formatter('Jan')",
        ):
            drawn = _build(page, "fr", PIE, expression)
            assert "75.5%" not in drawn, (page, expression, drawn)
            assert "." not in drawn.split(">")[-1], (page, expression, drawn)

    @pytest.mark.parametrize("page", PAGES)
    def test_the_axis_falls_back_to_the_french_word_for_category(self, page):
        # The fallback is only reachable through the builder: it is a local
        # the option exposes nowhere except inside this tooltip's closure.
        html = _build(page, "fr", PIE_NO_X,
                      "opt.tooltip.formatter({name:'x', value:30, percent:75})")
        assert i18n.MESSAGES["ui.chart.category"]["fr"] in html, html
        assert "Category" not in html, html

    def test_the_funnel_names_its_own_parts_in_french(self):
        # Shaped the way core.post_process.compute_funnel writes it: the
        # conversion and drop-off columns are what the tooltip reads, and a
        # fixture without them exercises a tooltip with nothing to say.
        payload = {"rows": [{"stage": "A", "n": 100, "conversion_rate": None,
                             "drop_off": None},
                            {"stage": "B", "n": 40, "conversion_rate": 40.0,
                             "drop_off": 60}],
                   "x_key": "stage", "y_keys": ["n"], "chart_type": "funnel"}
        html = _build("portal_chat.html", "fr", payload,
                      "opt.tooltip.formatter({name:'B', dataIndex:1, value:40})")
        for msg_id in ("ui.chart.count", "ui.chart.dropoff"):
            assert i18n.MESSAGES[msg_id]["fr"] in html, (msg_id, html)
        assert "Drop-off" not in html, html
        assert "from prev" not in html, html
        # The slice label is drawn on the canvas, so its percentage matters.
        label = _build("portal_chat.html", "fr", payload,
                       "opt.series[0].label.formatter({name:'B', value:40.5})")
        assert "40,5\u00a0%" in label, label

    def test_the_treemap_percentage_is_in_french_notation(self):
        payload = {"rows": [{"c": "A", "v": 60}, {"c": "B", "v": 40}],
                   "x_key": "c", "y_keys": ["v"], "chart_type": "treemap"}
        label = _build("portal_chat.html", "fr", payload,
                       "opt.series[0].label.formatter({name:'A', value:60})")
        assert "60,0\u00a0%" in label, label
        assert "60.0%" not in label, label

    def test_the_cohort_heatmap_speaks_french(self):
        payload = {"rows": [{"cohort": "2024-01", "Month 0": 100.0, "Month 1": 80.5,
                             "__abs_Month 1": 1234}],
                   "x_key": "cohort", "y_keys": ["Month 0", "Month 1"],
                   "chart_type": "heatmap"}
        html = _build("portal_chat.html", "fr", payload,
                      "opt.tooltip.formatter({data:[1, 0, 80.5]})")
        assert i18n.MESSAGES["ui.chart.retention"]["fr"] in html, html
        assert i18n.MESSAGES["ui.chart.users"]["fr"] in html, html
        assert "Retention" not in html and "Users" not in html, html
        assert "80,5\u00a0%" in html, html
        # The absolute count is grouped for the reader too.
        assert "1\u202f234" in html, html
        # The cell labels and the colour scale are drawn on the canvas.
        # 80.5 at zero digits is an exact tie, and both sides round it to the
        # even 80 -- the browser only because window.qbNum was taught to.
        assert i18n.format_percent(80.5, 0, lang="fr") == "80\u00a0%"
        assert _build("portal_chat.html", "fr", payload,
                      "opt.series[0].label.formatter({data:[1,0,80.5]})") == "80\u00a0%"
        assert _build("portal_chat.html", "fr", payload,
                      "opt.visualMap.formatter(80.5)") == "80\u00a0%"

    def test_a_cohort_cell_with_no_value_says_so_in_french(self):
        payload = {"rows": [{"cohort": "2024-01", "Month 0": 100.0, "Month 1": None}],
                   "x_key": "cohort", "y_keys": ["Month 0", "Month 1"],
                   "chart_type": "heatmap"}
        html = _build("portal_chat.html", "fr", payload,
                      "opt.tooltip.formatter({data:[1, 0, null]})")
        assert i18n.MESSAGES["ui.chart.not_available"]["fr"] in html, html
        assert "N/A" not in html, html

    def test_the_histogram_counts_in_french(self):
        # Chat page only -- the dashboard has no histogram branch.
        payload = {"rows": [{"v": 1}, {"v": 2}, {"v": 3}],
                   "x_key": None, "y_keys": ["v"], "chart_type": "histogram"}
        assert _build("portal_chat.html", "fr", payload, "opt.yAxis.name") == \
            i18n.MESSAGES["ui.chart.count"]["fr"]
        assert _build("portal_chat.html", "en", payload, "opt.yAxis.name") == "Count"


class TestNoChartLabelGoesBackToEnglishNotation:
    """
    A source scan, deliberately, and the only one in this file.

    Everything above executes: the annotation, the pie, the funnel, the
    cohort heatmap, the treemap and the histogram are all built by the real
    builder and their real formatters are called. This scan is the backstop
    for the branches no test has reached yet -- waterfall, scatter, and
    whatever chart type is added next -- because the failure it catches is one
    a reviewer cannot see either: `n.toFixed(1) + '%'` looks like ordinary
    number formatting and is English notation on every page in the product.

    It checks an invariant with a single readable shape, not a pinned
    initialiser, so it fails when the invariant breaks rather than when the
    code around it moves.
    """

    @pytest.mark.parametrize("page", PAGES)
    def test_no_percentage_is_written_with_tofixed(self, page):
        offenders = [
            f"{n}: {line.strip()[:90]}"
            for n, line in enumerate(_page(page).splitlines(), 1)
            if "toFixed" in line and "%" in line
        ]
        assert not offenders, (
            f"{page} writes a percentage with toFixed, which is English "
            f"notation on every page: " + "; ".join(offenders)
        )


# ── The caption that says the chart is a subset ──────────────────────────────
# Two defects, seen together on one live chart: the caption was an English
# template literal in the page, and it sat on the same row as the legend, so
# "Showing the 20 largest of 24" printed straight through the series names.

def _wide(series: int = 1) -> dict:
    """24 categories -- past CATEGORY_CAP -- with `series` measures each."""
    keys = [f"v{i}" for i in range(series)]
    rows = [{"cat": f"Warehouse {n:02d}", **{k: n * (i + 1) for i, k in enumerate(keys)}}
            for n in range(24)]
    return {"rows": rows, "x_key": "cat", "y_keys": keys, "chart_type": "bar"}


class TestTheSubsetCaptionIsInTheReadersLanguage:

    def test_the_caption_says_so_in_english(self):
        text = _build("portal_chat.html", "en", _wide(), "opt.title.subtext")
        assert text == "Showing the 20 largest of 24"

    def test_the_caption_says_so_in_french(self):
        text = _build("portal_chat.html", "fr", _wide(), "opt.title.subtext")
        assert text == "Affichage des 20 plus grands sur 24"

    def test_a_chart_showing_everything_carries_no_caption(self):
        payload = {"rows": [{"cat": "A", "v": 1}, {"cat": "B", "v": 2}],
                   "x_key": "cat", "y_keys": ["v"], "chart_type": "bar"}
        assert _build("portal_chat.html", "en", payload,
                      "opt.title === undefined ? 'none' : 'present'") == "none"

    def test_dropped_series_are_named_in_french_too(self):
        # Nine measures against an eight-slot palette: one series is dropped
        # rather than drawn in a colour another series already has.
        text = _build("portal_chat.html", "fr", _wide(series=9), "opt.title.subtext")
        assert "séries supplémentaires non affichées" in text


class TestTheCaptionAndTheLegendDoNotOverlap:
    """They both sat on the top row and drew through each other. The forecast
    chart already stacked them; every other type did not."""

    def test_the_caption_drops_below_a_legend(self):
        top = _build("portal_chat.html", "en", _wide(series=3), "String(opt.title.top)")
        legend = _build("portal_chat.html", "en", _wide(series=3),
                        "String(opt.legend.top)")
        assert int(top) > int(legend), (
            f"caption at {top} is not below the legend at {legend}"
        )

    def test_the_caption_stays_at_the_top_when_there_is_no_legend(self):
        assert _build("portal_chat.html", "en", _wide(), "String(opt.title.top)") == "2"

    def test_the_plot_area_clears_both(self):
        # The grid has to start below whatever is stacked above it, or the
        # caption prints over the topmost bars instead of over the legend.
        stacked = int(_build("portal_chat.html", "en", _wide(series=3),
                             "String(opt.grid.top)"))
        caption = int(_build("portal_chat.html", "en", _wide(series=3),
                             "String(opt.title.top)"))
        assert stacked > caption + 11, (
            f"grid starts at {stacked}, caption sits at {caption}"
        )

    def test_a_chart_with_no_caption_does_not_reserve_the_space(self):
        payload = {"rows": [{"cat": f"W{n}", "a": n, "b": n * 2} for n in range(5)],
                   "x_key": "cat", "y_keys": ["a", "b"], "chart_type": "bar"}
        with_caption = int(_build("portal_chat.html", "en", _wide(series=2),
                                  "String(opt.grid.top)"))
        without = int(_build("portal_chat.html", "en", payload, "String(opt.grid.top)"))
        assert without < with_caption
