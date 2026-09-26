"""
tests/test_chart_robustness.py

buildChartOption never threw on any data shape thrown at it — zero rows, all
nulls, 500 categories, twelve series, non-numeric values, unicode labels. What
it did instead was draw something misleading, which is worse than an error
because nobody investigates it.

Measured in a browser against the real function, before this change:

  1.2e12                 formatted as "1200B", not "1.2T"
  0.00012                formatted as "0", so an axis of small rates read as zeros
  Infinity               formatted as "InfinityB"
  500 categories         drew 500 bars into a fixed grid: 0.6px per row, no zoom
  40 pie slices          drew all 40, labels off, legend overflowing
  12 series              drew 12 with 8 unique colours — series 9-12 repeated
                         1-4, and the legend carried duplicate swatches
  0 rows                 drew empty axes with no explanation

SCOPE. The renderer is static/js/qb-charts.js, shared by the chat and the
dashboard. Every test below EXECUTES it (under dukpy, through the harness in
tests/test_chart_annotation_language.py) and asserts on the option it builds;
the browser-level behaviour -- a capped ranking readable at its drawn size, a
long series moving under its zoom control -- was checked in Chromium against
real ECharts instances.
"""

from __future__ import annotations

import re
from pathlib import Path

import json

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHAT = ROOT / "portal" / "templates" / "portal_chat.html"


dukpy = pytest.importorskip("dukpy")


def _compact(value) -> str:
    from tests.test_chart_annotation_language import _run
    return _run("portal_chat.html", "en", f"QBCharts.compactNumber({value})")


def _option(payload: dict, expression: str):
    from tests.test_chart_annotation_language import _build
    return json.loads(_build("portal_chat.html", "en", payload,
                             f"JSON.stringify({expression})"))


def test_the_magnitude_ladder_reaches_trillions():
    # Without a trillion tier 1.2e12 formatted as "1200B", which reads as a
    # mistake on an axis.
    assert _compact("1.2e12") == "1.2T"
    assert _compact("3.4e9") == "3.4B"
    assert _compact("812044") == "812K", "a whole tier carries no trailing .0"


def test_small_magnitudes_do_not_collapse_to_zero():
    # A fixed decimal format rendered every value below 0.005 as "0", so a
    # chart of rates drew an axis of zeros.
    assert _compact("0.00012") == "0.00012"
    assert _compact("0.0045") == "0.0045"


def test_non_finite_numbers_do_not_enter_the_ladder():
    # isNaN() alone let Infinity through to the ladder: "InfinityB".
    assert _compact("Infinity") == "Infinity"


def test_a_dense_categorical_chart_is_capped_to_what_can_be_read():
    rows = [{"CTY": f"City {n:02d}", "AMT": float(100 - n)} for n in range(30)]
    drawn = _option({"rows": rows, "x_key": "CTY", "y_keys": ["AMT"], "chart_type": "bar"},
                    "{axis: opt.yAxis.data, inverse: opt.yAxis.inverse, interval: opt.yAxis.axisLabel.interval}")
    # Twenty categories, drawn horizontally, largest first, every one named.
    assert len(drawn["axis"]) == 20
    assert drawn["axis"][0] == "City 00"
    assert drawn["inverse"] is True
    assert drawn["interval"] == 0


def test_a_time_series_is_never_reordered_or_dropped():
    """Ordering carries the meaning. Taking the largest values out of a time
    series silently rewrites the shape of the line."""
    labels = [f"{2016 + m // 12}-{m % 12 + 1:02d}" for m in range(120)]
    payload = {"rows": [{"MTH": label, "AMT": float((m * 37) % 90)} for m, label in enumerate(labels)],
               "x_key": "MTH", "y_keys": ["AMT"], "chart_type": "line",
               "chart_spec": {"x": {"column": "MTH", "role": "temporal"}}}
    drawn = _option(payload, "{axis: opt.xAxis.data, zoom: !!opt.dataZoom}")
    assert drawn["axis"] == labels
    assert drawn["zoom"], "a long series lost its way to move through the data"


def test_a_pie_keeps_its_total_when_capped():
    """A pie is a part-to-whole claim. Dropping the tail would leave the slices
    adding up to a different total than the answer states.

    Was two substring checks over a source window -- "Other (" and "reduce" --
    which is the wrong instrument twice over: it passed if the words appeared
    in a comment, and it FAILED when "Other (N)" moved into the message
    catalogue, which changed nothing about whether the total survives. Executed
    against the real builder now, so it fails when the arithmetic is wrong and
    not when the wording changes.
    """
    dukpy = pytest.importorskip("dukpy")
    from tests.test_chart_annotation_language import _build

    rows = [{"CAT": f"c{i:02d}", "AMT": float(i + 1)} for i in range(30)]
    payload = {"rows": rows, "x_key": "CAT", "y_keys": ["AMT"],
               "chart_type": "pie"}
    drawn = json.loads(_build("portal_chat.html", "en", payload,
                              "JSON.stringify(opt.series[0].data)"))
    assert len(drawn) < len(rows), "the pie was not capped at all"
    assert sum(item["value"] for item in drawn) == sum(r["AMT"] for r in rows), (
        "the capped pie no longer adds up to the answer's total")


def test_series_are_capped_to_the_validated_palette_length():
    rows = [{"cat": f"W{n}", **{f"v{i}": n + i for i in range(9)}} for n in range(3)]
    drawn = _option({"rows": rows, "x_key": "cat", "y_keys": [f"v{i}" for i in range(9)],
                     "chart_type": "bar"},
                    "opt.series.map(function (s) { return s.itemStyle.color })")
    # Eight series, eight different colours: a ninth would repeat one.
    assert len(drawn) == 8
    assert len(set(drawn)) == 8


def test_the_reader_is_told_when_a_chart_is_a_subset():
    """A chart that silently shows part of the data is read as all of it.

    This asserted on the page's source, so it broke the day the caption moved
    into the message catalogue and would have passed for a caption that
    resolved to nothing. It runs the real builder now: what matters is that a
    truncated chart carries a caption saying so, on both the cartesian and the
    pie branches.
    """
    from tests.test_chart_annotation_language import _build

    wide = {"rows": [{"cat": f"W{n:02d}", "v": n} for n in range(24)],
            "x_key": "cat", "y_keys": ["v"], "chart_type": "bar"}
    assert "24" in _build("portal_chat.html", "en", wide, "opt.title.subtext")

    pie = dict(wide, chart_type="pie")
    assert "24" in _build("portal_chat.html", "en", pie, "opt.title.subtext"), (
        "the pie branch shows a subset silently"
    )

    many = {"rows": [{"cat": f"W{n:02d}",
                      **{f"v{i}": n + i for i in range(9)}} for n in range(3)],
            "x_key": "cat", "y_keys": [f"v{i}" for i in range(9)],
            "chart_type": "bar"}
    assert "series" in _build("portal_chat.html", "en", many, "opt.title.subtext")


def test_a_chart_with_nothing_to_draw_says_so():
    # An empty frame reads as a broken chart rather than as an empty result.
    empty = _option({"rows": [], "x_key": "cat", "y_keys": ["v"], "chart_type": "bar"},
                    "opt.graphic.style.text")
    assert empty == "No rows to plot"
    no_value = _option({"rows": [{"cat": "A"}], "x_key": "cat", "y_keys": [], "chart_type": "bar"},
                       "opt.graphic.style.text")
    assert no_value == "No value column to plot"


def test_chart_furniture_comes_from_the_theme():
    """Axis, gridlines and tooltip were a Tailwind slate set, so every chart's
    frame sat in a different neutral family from the product around it. The
    harness's theme stub stands in for the tokens; the drawn chrome is it."""
    rows = [{"cat": "A", "v": 1.0}, {"cat": "B", "v": 2.0}]
    drawn = _option({"rows": rows, "x_key": "cat", "y_keys": ["v"], "chart_type": "bar"},
                    "{label: opt.yAxis.axisLabel.color, grid: opt.yAxis.splitLine.lineStyle.color,"
                    " all: JSON.stringify(opt)}")
    assert drawn["label"] == "#888" and drawn["grid"] == "#eee"
    for stale in ("#94a3b8", "#64748b", "rgba(148,163,184,.14)", "#F2F4F7", "#94A3B8", "#334155"):
        assert stale not in drawn["all"], f"chart chrome hardcodes {stale} again"


def test_both_portal_pages_share_one_chart_theme():
    from tests.test_chart_annotation_language import PAGES, _page, _renderer_for
    for page in PAGES:
        _renderer_for(page)   # asserts the page loads the shared renderer
        assert "src=\"{{ asset('js/chart-palettes.js') }}\"" in _page(page), page
