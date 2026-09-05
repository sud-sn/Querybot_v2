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

SCOPE. The renderer is JavaScript and the suite has no JS runtime, so the
behaviour was verified in a browser driving the real buildChartOption and real
ECharts instances: 500 rows capped to 20 at 12.6px per row (readable against an
11px label), pie capped to 12 plus an "Other (28)" slice with the total
preserved, 120 monthly points kept in chronological order with a zoom control,
and distinct empty-state messages. What is pinned here is the source-level
intent, so the thresholds and the reasoning cannot quietly drift back.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHAT = ROOT / "portal" / "templates" / "portal_chat.html"


@pytest.fixture(scope="module")
def source() -> str:
    return CHAT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def fmt_num(source: str) -> str:
    start = source.index("function _fmtNum(")
    return source[start: source.index("\n}", start)]


def test_the_magnitude_ladder_reaches_trillions(fmt_num):
    assert "1e12" in fmt_num and "'T'" in fmt_num, (
        "without a trillion tier a total of 1.2e12 formats as '1200B', which "
        "reads as a mistake on an axis"
    )
    for tier in ("1e9", "1e6", "1e3"):
        assert tier in fmt_num, f"the {tier} tier disappeared"


def test_small_magnitudes_do_not_collapse_to_zero(fmt_num):
    assert "toPrecision" in fmt_num, (
        "a fixed decimal format renders every value below 0.005 as '0', so a "
        "chart of rates or conversion fractions draws an axis of zeros"
    )
    assert "abs < 0.01" in fmt_num


def test_non_finite_numbers_do_not_enter_the_ladder(fmt_num):
    assert "Number.isFinite" in fmt_num, (
        "isNaN() alone lets Infinity through to the magnitude ladder, where it "
        "formats as 'InfinityB'"
    )


def test_a_dense_categorical_chart_is_capped_to_what_can_be_read(source):
    cap = re.search(r"const CATEGORY_CAP = (\d+);", source)
    assert cap, "the category cap is gone"
    value = int(cap.group(1))
    # A dense categorical chart renders horizontally, so each category needs a
    # readable row. At the ~300px these render at, the budget is height/cap.
    assert 300 / value >= 12, (
        f"a cap of {value} leaves {300/value:.1f}px per row against an 11px "
        f"label; the bars become unnameable"
    )


def test_a_time_series_is_never_reordered_or_dropped(source):
    """Ordering carries the meaning. Taking the largest values out of a time
    series silently rewrites the shape of the line."""
    assert re.search(r"const ordered = .*?type === 'line'.*?temporal", source, re.S), (
        "the ordered-series guard is gone; a line chart can now be re-ranked"
    )
    block = source[source.index("const CATEGORY_CAP"): source.index("const horizontal =")]
    assert "!ordered" in block, "the cap no longer excludes ordered series"
    assert "dataZoom" in source, "long series lost their way to move through the data"


def test_a_pie_keeps_its_total_when_capped(source):
    """A pie is a part-to-whole claim. Dropping the tail would leave the slices
    adding up to a different total than the answer states."""
    block = source[source.index("const PIE_CAP"): source.index("const horizontal =")]
    assert "Other (" in block, "the pie tail is dropped rather than grouped"
    assert "reduce" in block, "the grouped slice is not summed from the tail"


def test_series_are_capped_to_the_validated_palette_length(source):
    assert "const seriesCap   = colors.length;" in source, (
        "series are no longer bounded by the palette, so `i % colors.length` "
        "will repeat colours and the legend will carry duplicate swatches"
    )
    assert "droppedSeries" in source


def test_the_reader_is_told_when_a_chart_is_a_subset(source):
    """A chart that silently shows part of the data is read as all of it."""
    assert "capNotice" in source
    for path in ("title: capNotice,",):
        assert source.count(path) == 2, (
            "the cap notice must reach both the pie branch and the cartesian "
            "base, or one of them shows a subset silently"
        )
    assert "more series not shown" in source


def test_a_chart_with_nothing_to_draw_says_so(source):
    # Both notices are catalogue ids in the source now, so this asserts they
    # still reach the browser -- a source assertion would pass for a page that
    # resolves the id to nothing, which is the empty grid this test is about.
    from tests.chat_render import render as _render_chat, catalogue as _catalogue
    _shipped = _catalogue(_render_chat(lang="en")).values()
    assert "No rows to plot" in _shipped and "No value column to plot" in _shipped, (
        "an empty chart renders as an empty grid, which reads as broken rather "
        "than as an empty result"
    )
    guard = source.index("if (!rows.length || !yKey)")
    base = source.index("// ── Shared cartesian base")
    assert guard < base, "the empty guard runs after the chart is already built"


def test_chart_furniture_comes_from_the_theme(source):
    """Axis, gridlines and tooltip were a Tailwind slate set, so every chart's
    frame sat in a different neutral family from the product around it."""
    assert "window.QB_CHART_THEME()" in source
    for stale in ("'#94a3b8'", "'#64748b'", "rgba(148,163,184,.14)", "'#F2F4F7'"):
        assert stale not in source, f"chart chrome hardcodes {stale} again"


def test_both_portal_pages_share_one_chart_theme():
    dash = (ROOT / "portal" / "templates" / "portal_dashboard.html").read_text(encoding="utf-8")
    assert "window.QB_CHART_THEME" in dash, "the dashboard has its own chrome again"
    assert "'#94A3B8'" not in dash and "'#334155'" not in dash
