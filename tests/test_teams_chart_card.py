"""
tests/test_teams_chart_card.py

Native Adaptive Card charts for Teams (gateway/teams_chart_card.py +
TeamsAdapter.send_chart wiring).

The portal renders interactive ECharts from the chart payload built by
core/chart.py::build_chart_payload; Teams used to flatten that same payload
into a matplotlib PNG. Teams supports native chart elements in Adaptive
Cards v1.5 (Chart.VerticalBar[.Grouped], Chart.HorizontalBar, Chart.Line,
Chart.Pie, Chart.Donut — schemas verified against Microsoft's "Charts in
Adaptive Cards" documentation), so the adapter now prefers a native chart
card and falls back to the PNG path for unmappable types, POST failures,
or when TEAMS_NATIVE_CHARTS=0.

Pure-adapter tests: no DB, no network — all HTTP mocked, following the
conventions of tests/test_teams_parity.py.
"""

import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.base import PlatformEvent
from gateway.teams_adapter import TeamsAdapter
from gateway.teams_chart_card import build_teams_chart_card, is_native_teams_chart_type


def _run(coro):
    return asyncio.run(coro)


def _make_adapter():
    return TeamsAdapter({
        "app_id":       "fake-app-id",
        "app_password": "fake-password",
        "tenant_id":    "common",
    })


def _make_event() -> PlatformEvent:
    return PlatformEvent(
        account_id="tenant-xyz",
        user_id="user-abc",
        channel_id=json.dumps({
            "service_url":     "https://smba.trafficmanager.net/amer/",
            "conversation_id": "conv-1",
            "activity_id":     "act-1",
        }),
        text="ignored in these tests",
        platform="teams",
        raw={},
    )


def _payload(chart_type="bar", y_keys=None, rows=None, title="Revenue by warehouse"):
    return {
        "chart_type": chart_type,
        "title": title,
        "x_key": "WHS_NM",
        "y_keys": y_keys or ["REVENUE"],
        "rows": rows if rows is not None else [
            {"WHS_NM": "North", "REVENUE": 1000.5, "COST": 400.0},
            {"WHS_NM": "South", "REVENUE": 800.25, "COST": 350.0},
        ],
    }


# ──────────────────────────────────────────────────────────────────────────
# Card builder — type mapping
# ──────────────────────────────────────────────────────────────────────────
class BuilderTypeMappingTests(unittest.TestCase):

    def _element(self, card):
        self.assertIsNotNone(card)
        return card["body"][-1]

    def test_single_series_bar_maps_to_vertical_bar(self):
        el = self._element(build_teams_chart_card(_payload("bar")))
        self.assertEqual(el["type"], "Chart.VerticalBar")
        self.assertEqual(el["data"], [{"x": "North", "y": 1000.5}, {"x": "South", "y": 800.25}])

    def test_multi_series_bar_maps_to_grouped(self):
        el = self._element(build_teams_chart_card(_payload("bar", y_keys=["REVENUE", "COST"])))
        self.assertEqual(el["type"], "Chart.VerticalBar.Grouped")
        self.assertEqual(len(el["data"]), 2)
        self.assertEqual(el["data"][0]["legend"], "REVENUE")
        self.assertEqual(el["data"][0]["values"][0], {"x": "North", "y": 1000.5})
        self.assertEqual(el["data"][1]["legend"], "COST")

    def test_many_labels_switch_to_horizontal_bar(self):
        rows = [{"WHS_NM": f"W{i}", "REVENUE": float(i)} for i in range(1, 12)]
        el = self._element(build_teams_chart_card(_payload("bar", rows=rows)))
        self.assertEqual(el["type"], "Chart.HorizontalBar")

    def test_long_label_switches_to_horizontal_bar(self):
        rows = [
            {"WHS_NM": "NOBLE 980 PORT KELLS DC", "REVENUE": 10.0},
            {"WHS_NM": "EAST", "REVENUE": 5.0},
        ]
        el = self._element(build_teams_chart_card(_payload("bar", rows=rows)))
        self.assertEqual(el["type"], "Chart.HorizontalBar")

    def test_line_area_forecast_map_to_chart_line(self):
        for t in ("line", "area", "forecast"):
            el = self._element(build_teams_chart_card(_payload(t)))
            self.assertEqual(el["type"], "Chart.Line", t)
            self.assertEqual(el["data"][0]["legend"], "REVENUE")
            self.assertEqual(el["data"][0]["values"][0], {"x": "North", "y": 1000.5})

    def test_pie_and_donut_data_shape(self):
        for t, expected in (("pie", "Chart.Pie"), ("donut", "Chart.Donut")):
            el = self._element(build_teams_chart_card(_payload(t)))
            self.assertEqual(el["type"], expected, t)
            self.assertEqual(el["data"][0], {"legend": "North", "value": 1000.5})

    def test_pie_drops_non_positive_slices(self):
        rows = [
            {"WHS_NM": "North", "REVENUE": 100.0},
            {"WHS_NM": "South", "REVENUE": -50.0},
            {"WHS_NM": "East", "REVENUE": 0.0},
        ]
        el = self._element(build_teams_chart_card(_payload("pie", rows=rows)))
        self.assertEqual(len(el["data"]), 1)
        self.assertEqual(el["data"][0]["legend"], "North")

    def test_unmappable_types_return_none(self):
        for t in ("scatter", "heatmap", "waterfall", "funnel", "histogram", "boxplot", "treemap"):
            self.assertIsNone(build_teams_chart_card(_payload(t)), t)
            self.assertFalse(is_native_teams_chart_type(t), t)

    def test_native_type_predicate(self):
        for t in ("bar", "line", "area", "forecast", "pie", "donut"):
            self.assertTrue(is_native_teams_chart_type(t), t)


# ──────────────────────────────────────────────────────────────────────────
# Card builder — hygiene / edge cases
# ──────────────────────────────────────────────────────────────────────────
class BuilderEdgeCaseTests(unittest.TestCase):

    def test_empty_rows_returns_none(self):
        self.assertIsNone(build_teams_chart_card(_payload("bar", rows=[])))

    def test_missing_x_key_or_y_keys_returns_none(self):
        p = _payload("bar")
        p["x_key"] = ""
        self.assertIsNone(build_teams_chart_card(p))
        p = _payload("bar")
        p["y_keys"] = []
        self.assertIsNone(build_teams_chart_card(p))

    def test_none_payload_returns_none(self):
        self.assertIsNone(build_teams_chart_card(None))
        self.assertIsNone(build_teams_chart_card({}))

    def test_non_numeric_values_skipped(self):
        rows = [
            {"WHS_NM": "North", "REVENUE": "not-a-number"},
            {"WHS_NM": "South", "REVENUE": 800.25},
        ]
        card = build_teams_chart_card(_payload("bar", rows=rows))
        self.assertEqual(card["body"][-1]["data"], [{"x": "South", "y": 800.25}])

    def test_all_values_non_numeric_returns_none(self):
        rows = [{"WHS_NM": "North", "REVENUE": "x"}]
        self.assertIsNone(build_teams_chart_card(_payload("bar", rows=rows)))

    def test_row_cap_enforced(self):
        rows = [{"WHS_NM": f"W{i}", "REVENUE": float(i)} for i in range(200)]
        card = build_teams_chart_card(_payload("bar", rows=rows))
        self.assertEqual(len(card["body"][-1]["data"]), 50)

    def test_card_version_and_schema(self):
        card = build_teams_chart_card(_payload("bar"))
        self.assertEqual(card["version"], "1.5")
        self.assertEqual(card["$schema"], "http://adaptivecards.io/schemas/adaptive-card.json")
        self.assertEqual(card["type"], "AdaptiveCard")

    def test_sample_card_full_structure(self):
        # Full-structure snapshot: paste this JSON into the Adaptive Cards
        # designer (Teams host) for visual confirmation without a live tenant.
        card = build_teams_chart_card(_payload("bar"))
        self.assertEqual(card, {
            "type": "AdaptiveCard",
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "version": "1.5",
            "body": [
                {
                    "type": "TextBlock",
                    "text": "Revenue by warehouse",
                    "weight": "bolder",
                    "size": "medium",
                    "wrap": True,
                },
                {
                    "type": "Chart.VerticalBar",
                    "title": "Revenue by warehouse",
                    "xAxisTitle": "WHS NM",
                    "yAxisTitle": "REVENUE",
                    "colorSet": "categorical",
                    "data": [
                        {"x": "North", "y": 1000.5},
                        {"x": "South", "y": 800.25},
                    ],
                },
            ],
        })


# ──────────────────────────────────────────────────────────────────────────
# Adapter wiring — native card POST + PNG fallback
# ──────────────────────────────────────────────────────────────────────────
class SendChartNativeCardTests(unittest.TestCase):

    def _post_capture(self, status_code=200):
        captured = {}

        async def fake_post(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            class FakeResp:
                pass
            FakeResp.status_code = status_code
            FakeResp.text = "err" if status_code >= 400 else "ok"
            return FakeResp()

        return captured, fake_post

    def test_mappable_payload_sends_native_chart_card(self):
        adapter = _make_adapter()
        adapter.upload_file = AsyncMock()
        captured, fake_post = self._post_capture(200)

        with patch.dict(os.environ, {"TEAMS_NATIVE_CHARTS": "1"}), \
             patch.object(adapter, "_get_token", new=AsyncMock(return_value="t")), \
             patch("gateway.teams_adapter.httpx.AsyncClient") as FakeClient:
            instance = FakeClient.return_value.__aenter__.return_value
            instance.post = AsyncMock(side_effect=fake_post)
            _run(adapter.send_chart(_make_event(), _payload("bar")))

        activity = captured["json"]
        att = activity["attachments"][0]
        self.assertEqual(att["contentType"], "application/vnd.microsoft.card.adaptive")
        self.assertEqual(att["content"]["version"], "1.5")
        self.assertEqual(att["content"]["body"][-1]["type"], "Chart.VerticalBar")
        # Native card succeeded — the PNG path must not also fire.
        adapter.upload_file.assert_not_called()

    def test_post_failure_falls_back_to_png(self):
        adapter = _make_adapter()
        adapter.upload_file = AsyncMock()
        captured, fake_post = self._post_capture(500)

        with patch.dict(os.environ, {"TEAMS_NATIVE_CHARTS": "1"}), \
             patch.object(adapter, "_get_token", new=AsyncMock(return_value="t")), \
             patch("gateway.teams_adapter.httpx.AsyncClient") as FakeClient, \
             patch("core.chart.generate_chart", return_value=b"fake-png"):
            instance = FakeClient.return_value.__aenter__.return_value
            instance.post = AsyncMock(side_effect=fake_post)
            _run(adapter.send_chart(_make_event(), _payload("bar")))

        self.assertIn("json", captured)  # native card WAS attempted
        adapter.upload_file.assert_called_once()
        args, _ = adapter.upload_file.call_args
        self.assertEqual(args[1], b"fake-png")

    def test_kill_switch_forces_png_without_post(self):
        adapter = _make_adapter()
        adapter.upload_file = AsyncMock()
        adapter._post_chart_card = AsyncMock()

        with patch.dict(os.environ, {"TEAMS_NATIVE_CHARTS": "0"}), \
             patch("core.chart.generate_chart", return_value=b"fake-png"):
            _run(adapter.send_chart(_make_event(), _payload("bar")))

        adapter._post_chart_card.assert_not_called()
        adapter.upload_file.assert_called_once()

    def test_unmappable_type_goes_straight_to_png_without_post(self):
        adapter = _make_adapter()
        adapter.upload_file = AsyncMock()
        adapter._post_chart_card = AsyncMock()

        with patch.dict(os.environ, {"TEAMS_NATIVE_CHARTS": "1"}), \
             patch("core.chart.generate_chart", return_value=b"fake-png"):
            _run(adapter.send_chart(_make_event(), _payload("scatter")))

        adapter._post_chart_card.assert_not_called()
        adapter.upload_file.assert_called_once()

    def test_empty_rows_still_skips_everything(self):
        adapter = _make_adapter()
        adapter.upload_file = AsyncMock()
        adapter._post_chart_card = AsyncMock()
        _run(adapter.send_chart(_make_event(), {"rows": [], "chart_type": "bar", "title": "x"}))
        adapter._post_chart_card.assert_not_called()
        adapter.upload_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
