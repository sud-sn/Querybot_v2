# -*- coding: utf-8 -*-
"""A legend wider than its chart wraps onto another row instead of paging.

On a phone a chart of two long-named series showed one name and a "1/2" pager with arrows: the reader had to tap
through to learn what the second line was. The legend now wraps (up to three rows), and the plot and the caption under
the legend move down by the rows it took. A legend that fits, or one too long for three rows, is left as it was.

Every test executes the renderer (static/js/qb-charts.js) and asserts on the option it would draw. Synthetic rows.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("dukpy", reason="the renderer is JavaScript; only executing it shows what it draws")

from tests.test_chart_annotation_language import _build  # noqa: E402

TWO_LINES = {
    "rows": [{"MONTH": f"2026-0{m}", "NET_AMOUNT_EAST_AND_NORTH": 100.0 + m, "NET_AMOUNT_WEST_AND_SOUTH": 90.0 + m}
             for m in range(1, 7)],
    "x_key": "MONTH", "y_keys": ["NET_AMOUNT_EAST_AND_NORTH", "NET_AMOUNT_WEST_AND_SOUTH"], "chart_type": "line",
    "chart_spec": {"x": {"column": "MONTH", "role": "time", "grain": "month"}},
}


def _drawn(width: int) -> dict:
    expression = (f"var before = JSON.parse(JSON.stringify(opt)); QBCharts.wrapLegend(opt, {width});"
                  "JSON.stringify({type: opt.legend.type, top: [].concat(opt.grid)[0].top,"
                  " was: [].concat(before.grid)[0].top, typeBefore: before.legend.type})")
    return json.loads(_build("portal_chat.html", "en", TWO_LINES, expression))


def test_on_a_phone_the_legend_wraps_and_the_plot_moves_down_for_it():
    drawn = _drawn(300)
    assert drawn["typeBefore"] == "scroll"
    assert drawn["type"] == "plain"
    assert drawn["top"] > drawn["was"]


def test_a_legend_that_fits_is_left_as_it_was():
    drawn = _drawn(1200)
    assert drawn["type"] == "scroll" and drawn["top"] == drawn["was"]
