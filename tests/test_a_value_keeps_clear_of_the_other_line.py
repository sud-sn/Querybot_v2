"""A value written on a line chart keeps clear of the other line drawn with it.

Found in the portal (October 2026), on a dashboard tile of a store's months against the year before: the dashed
line of the year before ran straight through a month's value ("$14.1K" struck through), and where the line fell
through a point, the value set beside it had the line run under its first figure, read as a minus ("-$11.7K").

* a value the other line runs through takes the other side of its point, where that side is clear of every line;
* where no side is, it sits on a small plate of the surface, which breaks the other line where they meet;
* a value set beside its point is lifted into the corner its line leaves free;
* a line drawn alone is drawn as it was.

Invented data only.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("dukpy", reason="the renderer is JavaScript; only executing it shows what it draws")
from tests.test_chart_annotation_language import _build  # noqa: E402

MONTHS = [f"2025-{m:02d}-01" for m in range(1, 13)]


def _chart(now, before=None):
    rows = [{"m": m, "now": float(a), **({"before": float(b)} if before else {})}
            for m, a, b in zip(MONTHS, now, before or now)]
    payload = {"chart_type": "line", "x_key": "m", "y_keys": ["now", "before"] if before else ["now"], "rows": rows,
               "column_formats": {"now": "currency", "before": "currency"}, "x_style": "month_year_long",
               "chart_spec": {"x": {"column": "m", "role": "temporal"}}}
    if before:
        payload["compare"] = {"prior": "before", "current": "now", "prior_label": "2024", "current_label": "2025"}
    return payload


def _labels(payload):
    """For each point of the line: where its value sits, whether on a plate, and how far it is lifted."""
    return json.loads(_build("portal_chat.html", "en", payload, """JSON.stringify(opt.series[0].data.map(function (d, i) {
        var label = (d && typeof d === 'object' && d.label) || {};
        var layout = typeof opt.series[0].labelLayout === 'function' ? opt.series[0].labelLayout({dataIndex: i}) : {};
        return [label.position || 'top', !!label.backgroundColor, layout.dy || 0];
    }))"""))


# A year that climbs gently, and the year before a little above it all along: every value above its point would
# sit on the dashed line.
GENTLE = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
JUST_ABOVE = [v + 3 for v in GENTLE]


def test_a_value_the_other_line_runs_through_takes_the_free_side():
    placed = _labels(_chart(GENTLE, JUST_ABOVE))
    middle = placed[2:10]
    assert all(side == "bottom" and not plate for side, plate, _ in middle), placed
    alone = _labels(_chart(GENTLE))
    assert all(side == "top" and not plate for side, plate, _ in alone[2:10]), "a line alone is drawn as it was"


def test_where_no_side_is_free_the_value_sits_on_a_plate():
    # A peak: its own line falls away on both sides of it below, and the year before runs just above it.
    now = [40, 40, 40, 40, 40, 70, 40, 40, 40, 40, 40, 40]
    before = [40, 40, 40, 40, 60, 74, 60, 40, 40, 40, 40, 40]
    side, plate, _ = _labels(_chart(now, before))[5]
    assert (side, plate) == ("top", True)
    plates = json.loads(_build("portal_chat.html", "en", _chart(now, before),
                               "JSON.stringify(opt.series[0].data[5].label)"))
    assert plates["backgroundColor"] and plates["padding"], "a plate of the surface, a little wider than the value"


def test_a_value_beside_its_point_is_lifted_off_its_own_line():
    # The line falls steeply through September: its value goes beside the point, lifted above the way down.
    now = [40, 42, 44, 46, 48, 50, 52, 80, 55, 30, 32, 34]
    placed = _labels(_chart(now))
    sides = [(i, side, dy) for i, (side, _, dy) in enumerate(placed) if side in ("left", "right")]
    assert sides and all(dy < 0 for _, _, dy in sides), placed
    assert all(dy == 0 for side, _, dy in placed if side in ("top", "bottom"))
