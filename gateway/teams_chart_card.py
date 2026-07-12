"""
gateway/teams_chart_card.py

Native Adaptive Card chart builder for Microsoft Teams.

Maps the interactive chart payload produced by core/chart.py's
build_chart_payload — the exact same payload the web portal's ECharts
renderer consumes — onto Teams' native Adaptive Card v1.5 chart elements
(Chart.VerticalBar, Chart.VerticalBar.Grouped, Chart.HorizontalBar,
Chart.Line, Chart.Pie, Chart.Donut), so Teams users get a real,
Teams-rendered chart with hover tooltips and legends instead of a flat
matplotlib PNG.

Element schemas verified against Microsoft's documentation
(learn.microsoft.com "Charts in Adaptive Cards - Teams"):
  Chart.VerticalBar / Chart.HorizontalBar : data = [{x, y}]
  Chart.VerticalBar.Grouped / Chart.Line  : data = [{legend, values: [{x, y}]}]
  Chart.Pie / Chart.Donut                 : data = [{legend, value}]
Card version must be "1.5". Optional per-element props used here:
title, xAxisTitle, yAxisTitle, colorSet.

Types with no native equivalent (scatter, heatmap, waterfall, funnel,
histogram, boxplot, treemap) return None — the caller
(TeamsAdapter.send_chart) falls back to the existing matplotlib PNG path.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("querybot.teams_chart_card")

# Teams caps card payload size; a chart with hundreds of points is
# unreadable at card width anyway. Mirrors the pipeline's own row caps.
_MAX_POINTS = 50

# Portal chart types that map onto a native Teams chart element.
_NATIVE_TYPES = frozenset({"bar", "line", "area", "forecast", "pie", "donut"})

# Portal heuristic mirrored from portal_chat.html's buildChartOption:
# many/long category labels read better as a horizontal bar chart.
_HORIZONTAL_LABEL_COUNT = 9
_HORIZONTAL_LABEL_LEN = 14


def _display_label(column: str) -> str:
    spaced = str(column or "").replace("_", " ").strip()
    return " ".join(
        part.capitalize() if not part.isupper() else part
        for part in spaced.split()
    )


def _to_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        n = float(value)
        return n if n == n else None   # NaN guard
    except (TypeError, ValueError):
        return None


def is_native_teams_chart_type(chart_type: str) -> bool:
    return (chart_type or "").lower().strip() in _NATIVE_TYPES


def build_teams_chart_card(chart_payload: dict) -> dict | None:
    """
    Build an Adaptive Card v1.5 with a native chart element from the
    interactive chart payload. Returns None when the payload can't be
    represented natively (unmappable type, no rows, no usable axes) —
    the caller then uses the matplotlib PNG path instead.
    """
    payload = chart_payload or {}
    chart_type = (payload.get("chart_type") or "").lower().strip()
    if chart_type not in _NATIVE_TYPES:
        return None

    rows = payload.get("rows") or []
    x_key = payload.get("x_key") or ""
    y_keys = [k for k in (payload.get("y_keys") or []) if k]
    if not rows or not x_key or not y_keys:
        return None

    rows = rows[:_MAX_POINTS]
    title = str(payload.get("title") or payload.get("question") or "Results").strip()

    if chart_type in {"pie", "donut"}:
        element = _pie_element(chart_type, rows, x_key, y_keys[0], title)
    elif chart_type in {"line", "area", "forecast"}:
        element = _line_element(rows, x_key, y_keys, title)
    elif len(y_keys) > 1:
        element = _grouped_bar_element(rows, x_key, y_keys, title)
    else:
        element = _bar_element(rows, x_key, y_keys[0], title)

    if element is None:
        return None

    return {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.5",
        "body": [
            {
                "type": "TextBlock",
                "text": title,
                "weight": "bolder",
                "size": "medium",
                "wrap": True,
            },
            element,
        ],
    }


def _pie_element(chart_type: str, rows: list[dict], x_key: str, y_key: str, title: str) -> dict | None:
    data = []
    for r in rows:
        value = _to_number(r.get(y_key))
        legend = str(r.get(x_key) or "").strip()
        # Pie/Donut slices must be positive to render meaningfully.
        if legend and value is not None and value > 0:
            data.append({"legend": legend, "value": value})
    if not data:
        return None
    return {
        "type": "Chart.Donut" if chart_type == "donut" else "Chart.Pie",
        "title": title,
        "colorSet": "categorical",
        "data": data,
    }


def _line_element(rows: list[dict], x_key: str, y_keys: list[str], title: str) -> dict | None:
    data = []
    for y_key in y_keys:
        values = []
        for r in rows:
            y = _to_number(r.get(y_key))
            x = str(r.get(x_key) or "").strip()
            if x and y is not None:
                values.append({"x": x, "y": y})
        if values:
            data.append({"legend": _display_label(y_key), "values": values})
    if not data:
        return None
    return {
        "type": "Chart.Line",
        "title": title,
        "xAxisTitle": _display_label(x_key),
        "yAxisTitle": _display_label(y_keys[0]) if len(y_keys) == 1 else "",
        "colorSet": "categorical",
        "data": data,
    }


def _grouped_bar_element(rows: list[dict], x_key: str, y_keys: list[str], title: str) -> dict | None:
    data = []
    for y_key in y_keys:
        values = []
        for r in rows:
            y = _to_number(r.get(y_key))
            x = str(r.get(x_key) or "").strip()
            if x and y is not None:
                values.append({"x": x, "y": y})
        if values:
            data.append({"legend": _display_label(y_key), "values": values})
    if not data:
        return None
    return {
        "type": "Chart.VerticalBar.Grouped",
        "title": title,
        "xAxisTitle": _display_label(x_key),
        "colorSet": "categorical",
        "data": data,
    }


def _bar_element(rows: list[dict], x_key: str, y_key: str, title: str) -> dict | None:
    data = []
    for r in rows:
        y = _to_number(r.get(y_key))
        x = str(r.get(x_key) or "").strip()
        if x and y is not None:
            data.append({"x": x, "y": y})
    if not data:
        return None

    labels = [d["x"] for d in data]
    horizontal = (
        len(labels) > _HORIZONTAL_LABEL_COUNT
        or any(len(lbl) > _HORIZONTAL_LABEL_LEN for lbl in labels)
    )
    return {
        "type": "Chart.HorizontalBar" if horizontal else "Chart.VerticalBar",
        "title": title,
        "xAxisTitle": _display_label(x_key),
        "yAxisTitle": _display_label(y_key),
        "colorSet": "categorical",
        "data": data,
    }
