"""Names that share a long start stay told apart, and a line's lowest point says its value once.

Seen on a dashboard built from a request, on a pharmacy's data: three bars all labelled "Cedar Point
Compounding …" (the names differ only in their last word, which the axis cut off), and a line whose lowest
month read "$18.5K" and "Low $18.5K" drawn over each other. Each name is now its first words and the part that
tells it apart ("Cedar Point… Phoenix"), and the lowest point keeps only its own label. Every test executes the
renderer (static/js/qb-charts.js) as the page does. Synthetic names; no customer data.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("dukpy", reason="the renderer is JavaScript; only executing it shows what it draws")

from tests.test_chart_annotation_language import _build  # noqa: E402

SITES = ["Harbour Light Compounding Pharmacy - Phoenix", "Harbour Light Compounding Pharmacy - Orlando",
         "Harbour Light Compounding Pharmacy - Dallas"]


def opt(payload, expression, page="portal_chat.html"):
    return json.loads(_build(page, "en", payload, f"JSON.stringify({expression})"))


def ranking(labels):
    return {"rows": [{"LBL": label, "AMT": float(300 - 10 * i)} for i, label in enumerate(labels)],
            "x_key": "LBL", "y_keys": ["AMT"], "chart_type": "bar",
            "chart_spec": {"x": {"column": "LBL", "role": "dimension"}}}


@pytest.mark.parametrize("page", ["portal_chat.html", "portal_dashboard.html"])
def test_names_sharing_a_long_start_show_the_part_that_tells_them_apart(page):
    shown = opt(ranking(SITES), "[" + ", ".join(f"opt.yAxis.axisLabel.formatter({json.dumps(s)})" for s in SITES) + "]",
                page)
    assert shown == ["Harbour Light… Phoenix", "Harbour Light… Orlando", "Harbour Light… Dallas"]


def test_short_names_and_names_that_differ_early_are_left_as_they_are():
    names = ["North", "South", "East"]
    assert opt(ranking(names), "opt.yAxis.axisLabel.formatter('North')") == "North"
    mixed = ["Ketamine 10% / Gabapentin 6% Cream", "Baclofen 2% / Cyclobenzaprine 1% Gel"]
    assert opt(ranking(mixed), f"opt.yAxis.axisLabel.formatter({json.dumps(mixed[0])})") == mixed[0]


def _trend(values):
    return {"rows": [{"MONTH": f"2026-{m:02d}-01", "AMT": v} for m, v in enumerate(values, start=1)],
            "x_key": "MONTH", "y_keys": ["AMT"], "chart_type": "line", "tile": True,
            "chart_spec": {"x": {"column": "MONTH", "role": "time", "grain": "month"}}}


def _labels(values):
    return opt(_trend(values), "[" + ", ".join(
        f"opt.series[0].label.formatter({{value: {v}, dataIndex: {i}}})" for i, v in enumerate(values)) + "]",
        "portal_dashboard.html")


def test_the_lowest_point_says_its_value_once():
    values = [21.1, 18.5, 23.2, 21.7, 20.1, 23.2, 22.9, 20.2, 24.4, 24.1, 23.6, 24.7]
    labels = _labels(values)
    # The low says "Low $18.5" in its own label, never "$18.5" under it as well; its neighbours sit well above it
    # (a fifth of the chart's height and more), their labels clear of its own, and keep their values.
    assert labels[1] == "" and all(labels[i] for i in range(len(values)) if i != 1)
    low = opt(_trend(values), "opt.series[0].markPoint.data.filter(p => p.label.show).map(p => p.label.formatter())",
              "portal_dashboard.html")
    assert low == ["Low $18.5"]


@pytest.mark.parametrize("low_at, scale, quiet", [(1, 1, {1, 2}), (5, 1, {4, 5, 6}), (10, 1, {9, 10}),
                                                  (1, 4300, {1, 2, 3})])
def test_the_low_label_never_lies_over_a_neighbours_label(low_at, scale, quiet):
    """"Low $157.7K" was drawn over the next month's "$167.8K", and once over the month after that too: every
    neighbour within the low label's reach (measured: about 7px a character against the room between two
    months) and about as low as it gives way -- on the side it runs to, both sides when it is centred. A short low
    label reaches the next month; "Low $124.7K" the two after it."""
    values = [(30.0 + (i % 3) * 0.2) * scale for i in range(12)]
    values[low_at] = 29.0 * scale
    assert {i for i, label in enumerate(_labels(values)) if not label} == quiet


def test_a_neighbour_well_above_the_low_keeps_its_value():
    values = [30.0 + (i % 3) for i in range(12)]
    values[5] = 10.0
    assert {i for i, label in enumerate(_labels(values)) if not label} == {5}


def test_the_highest_and_the_latest_values_are_said_in_strong_ink():
    values = [31.1, 28.5, 33.2, 31.7, 30.1, 33.2, 32.9, 30.2, 36.4, 34.1, 33.6, 34.7]
    labels = _labels(values)
    assert labels[8] == "{key|$36.4}" and labels[11] == "{key|$34.7}" and labels[10] == "$33.6"
    rich = opt(_trend(values), "opt.series[0].label.rich.key", "portal_dashboard.html")
    assert rich["fontWeight"] == 600 and rich["textBorderWidth"] == 3
