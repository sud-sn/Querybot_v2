"""A reader chooses how their charts are drawn, on their own Settings page.

Asked for: lines that curve rather than run straight, values written on line charts, motion as charts appear,
and a choice of colours kept per reader -- "not the full spec, just palettes of the best colour". The Settings
link opened the change-password form; it opens Settings now, with the password a link away.

* the colours offered are the palettes that pass the colour-vision checks (static/js/chart-palettes.js), and
  a palette saved with a chart still wins over the reader's;
* a smooth line is a monotone curve, which never bulges past a point it passes through;
* a single line of a dozen points or fewer says each value; more points, or several lines, only where each
  ends;
* "No animation" stills every chart, as the system's reduce-motion setting does; a KPI's number counts up to
  itself otherwise, ending on exactly its text;
* what is stored is only ever one of the offered values.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from tests.test_a_new_core_answer_can_be_pinned import _reader, fresh_store  # noqa: F401
from tests.test_chart_annotation_language import _build
from tests.test_dashboard_page import _render

LINE = {"chart_type": "line", "x_key": "m", "y_keys": ["v"],
        "rows": [{"m": f"2026-{i:02d}-01", "v": 10.0 + i * (3 if i % 2 else -1)} for i in range(1, 7)],
        "column_roles": {"m": {"role": "temporal"}, "v": {"role": "measure", "format": "currency"}},
        "chart_spec": {"x": {"column": "m", "role": "temporal"}}}
BARS = {"chart_type": "bar", "x_key": "s", "y_keys": ["v"], "rows": [{"s": "A", "v": 3.0}, {"s": "B", "v": 2.0}]}


def _with(prefs: dict, payload: dict, expression: str):
    """The option the chat page draws for ``payload`` when the reader chose ``prefs``."""
    return json.loads(_build("portal_chat.html", "en", payload,
                             f"window.QB_CHART_PREFS = {json.dumps(prefs)}; var o = QBCharts.buildOption("
                             f"{json.dumps(payload)}); JSON.stringify({expression})"))


# ── what is kept ────────────────────────────────────────────────────────────


def test_only_an_offered_value_is_ever_kept(fresh_store):  # noqa: F811
    user = _reader(fresh_store)
    kept = fresh_store.set_chart_prefs(user["id"], {"palette": "ocean", "line": "wavy", "values": "<script>",
                                                    "motion": "off", "extra": "x"})
    assert kept == {"palette": "ocean", "line": "straight", "values": "show", "motion": "off"}
    assert fresh_store.chart_prefs(fresh_store.get_user(user["id"])) == kept


def test_the_palettes_offered_are_the_ones_that_pass_the_colour_checks():
    import store

    # Each was run through the dataviz validator (lightness band, chroma floor, colour-blind separation of
    # neighbours, normal-vision floor); "sunset" and "forest" fail it and are never offered.
    assert store.CHART_PREFS["palette"] == ("default", "ocean", "candy")


def test_the_settings_page_saves_and_every_page_carries_the_choice(fresh_store):  # noqa: F811
    from portal import routes

    user = _reader(fresh_store)
    request = MagicMock()

    async def form():
        return {"palette": "candy", "line": "smooth", "values": "hide", "motion": "on"}
    request.form = form
    with patch.object(routes, "_get_portal_user", return_value=user):
        response = asyncio.run(routes.settings_submit(request))
    assert response.status_code == 303 and response.headers["location"] == "/portal/settings?saved=1"
    stored = fresh_store.get_user(user["id"])
    assert routes.templates.env.globals["chart_prefs"](stored) == {
        "palette": "candy", "line": "smooth", "values": "hide", "motion": "on"}
    markup = _render([])
    assert "window.QB_CHART_PREFS = {" in markup
    assert "document.documentElement.dataset.motion = window.QB_CHART_PREFS.motion" in markup
    from pathlib import Path
    css = (Path(__file__).resolve().parents[1] / "static" / "css" / "portal.css").read_text(encoding="utf-8")
    assert 'html:not([data-motion="off"]) .kpi-tile' in css, "no animation stills the KPI tiles too"
    assert 'href="/portal/settings"' in markup and 'href="/portal/change-password" title' not in markup


# ── what the charts do with it ──────────────────────────────────────────────


def test_the_readers_palette_colours_their_charts_and_a_saved_one_wins():
    default = _with({}, BARS, "o.color")
    ocean = _with({"palette": "ocean"}, BARS, "o.color")
    assert ocean != default and ocean[0] == "#1d4ed8"
    assert _with({"palette": "ocean"}, {**BARS, "color_palette": "candy"}, "o.color")[0] == "#7c3aed"


def test_lines_run_straight_unless_the_reader_chose_curves():
    assert _with({}, LINE, "[o.series[0].smooth, o.series[0].smoothMonotone || null]") == [False, None]
    assert _with({"line": "smooth"}, LINE, "[o.series[0].smooth, o.series[0].smoothMonotone]") == [0.35, "x"]


def test_a_short_single_line_says_each_value_unless_the_reader_chose_not():
    shown = _with({}, LINE, "[!!(o.series[0].label && o.series[0].label.show), !!o.series[0].endLabel]")
    assert shown == [True, False], "every point's value, so no second label at its end"
    hidden = _with({"values": "hide"}, LINE, "[!!(o.series[0].label && o.series[0].label.show), "
                                             "!!(o.series[0].endLabel && o.series[0].endLabel.show)]")
    assert hidden == [False, True]
    long = {**LINE, "rows": [{"m": f"2025-{1 + i % 12:02d}-01" if i < 12 else f"2026-{i - 11:02d}-01", "v": float(i)}
                             for i in range(18)]}
    assert _with({}, long, "!!(o.series[0].label && o.series[0].label.show)") is False


def test_no_animation_stills_every_chart():
    assert _with({}, BARS, "o.animation") is True
    assert _with({"motion": "off"}, BARS, "o.animation") is False


@pytest.mark.parametrize("lang, text, middle", [
    ("en", "$185,933.03", "$162,691.40"), ("fr", "185 933,03 $", "162 691,40 $"), ("en", "38.0%", "33.3%")])
def test_a_kpi_counts_up_to_exactly_its_own_text(lang, text, middle):
    script = (f"window.QB_LANG = {json.dumps(lang)}; var frames = []; window.performance = {{now: function () "
              f"{{ return 0; }}}}; window.requestAnimationFrame = function (f) {{ frames.push(f); }};"
              f"var el = {{textContent: {json.dumps(text)}, dataset: {{}}}}; QBCharts.countUp(el, 700);"
              f"frames.shift()(350); var half = el.textContent; while (frames.length) frames.shift()(700);"
              f"JSON.stringify([half, el.textContent])")
    assert json.loads(_build("portal_chat.html", lang, BARS, script)) == [middle, text]
