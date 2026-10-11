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


def test_the_lowest_point_says_its_value_once():
    values = [21.1, 18.5, 23.2, 21.7, 20.1, 23.2, 22.9, 20.2, 24.4, 24.1, 23.6, 24.7]
    labels = opt(_trend(values), "[" + ", ".join(
        f"opt.series[0].label.formatter({{value: {v}, dataIndex: {i}}})" for i, v in enumerate(values)) + "]",
        "portal_dashboard.html")
    assert labels[1] == "" and all(labels[i] for i in range(len(values)) if i != 1)
    low = opt(_trend(values), "opt.series[0].markPoint.data.map(p => p.label.formatter())", "portal_dashboard.html")
    assert low == ["Low $18.5"]
