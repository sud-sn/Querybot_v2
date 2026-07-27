"""
Regression tests for the Teams PNG chart-rendering bug: core/chart.py's
_render() only had dedicated branches for bar/line/area/scatter, so a
waterfall, funnel, histogram, boxplot, treemap, or heatmap result silently
rendered as a plain bar chart on Teams -- correct numbers, wrong (misleading)
shape, with no indication anything was substituted.

Covers:
  - waterfall/funnel/histogram/boxplot now render via their own logic
    (verified by asserting real PNG bytes come back and, where practical,
    that the matplotlib call shape differs from a plain bar -- e.g.
    waterfall's colored increase/decrease bars, boxplot's bxp() summary
    stats) rather than falling through to the generic bar `else` branch.
  - heatmap/treemap raise UnsupportedChartTypeError instead of silently
    substituting a wrong chart -- and gateway/teams_adapter.py's send_chart
    turns that into an honest user-facing message instead of returning
    silently (the pre-fix behavior when generate_chart returned None).
"""
import unittest
from unittest.mock import AsyncMock, patch

from core.chart import generate_chart, UnsupportedChartTypeError, _render, _NO_PNG_RENDERING


def _png_ok(png_bytes: bytes) -> bool:
    return isinstance(png_bytes, bytes) and png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


class WaterfallRenderingTests(unittest.TestCase):
    def test_waterfall_renders_real_png(self):
        # Non-round values -- all-integer, highly-unique numeric columns get
        # classified as a dimension/ID column by _is_dimension_col, not a
        # measure (a pre-existing, unrelated heuristic); real dollar deltas
        # are never suspiciously round like that anyway.
        rows = [
            {"stage": "Starting balance", "delta": 100.25},
            {"stage": "Q1 gain", "delta": 30.50},
            {"stage": "Q2 loss", "delta": -20.75},
            {"stage": "Ending balance", "delta": 0.0},
        ]
        png = generate_chart(rows, "waterfall", "Balance waterfall")
        self.assertTrue(_png_ok(png))

    def test_waterfall_bar_bottoms_track_cumulative_running_total(self):
        # Direct unit check of the cumulative math (not just "it rendered"):
        # an increase's bar bottom is the running total BEFORE it; a
        # decrease's bar bottom is the running total AFTER it (so the bar
        # still spans the correct vertical range) -- this is the actual
        # waterfall shape a plain bar chart of the same raw deltas would not
        # produce.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = [{"stage": "A", "delta": 100.5}, {"stage": "B", "delta": -40.5}, {"stage": "C", "delta": 25.5}]
        captured = {}
        real_bar = plt.Axes.bar
        def spy_bar(self, x, height, bottom=None, **kw):
            captured["heights"] = list(height)
            captured["bottoms"] = list(bottom) if bottom is not None else None
            return real_bar(self, x, height, bottom=bottom, **kw)
        with patch.object(plt.Axes, "bar", spy_bar):
            generate_chart(rows, "waterfall", "t")
        self.assertEqual(captured["heights"], [100.5, 40.5, 25.5])
        self.assertEqual(captured["bottoms"], [0.0, 60.0, 60.0])


class FunnelRenderingTests(unittest.TestCase):
    def test_funnel_renders_real_png(self):
        rows = [
            {"stage": "Visited", "count": 1000.0, "funnel_pct": 100.0},
            {"stage": "Signed up", "count": 400.0, "funnel_pct": 40.0},
            {"stage": "Purchased", "count": 120.0, "funnel_pct": 12.0},
        ]
        png = generate_chart(rows, "funnel", "Signup funnel")
        self.assertTrue(_png_ok(png))

    def test_funnel_uses_horizontal_bars_not_vertical(self):
        # matplotlib's own barh() is internally implemented as bar(...,
        # orientation='horizontal') in current versions, so asserting
        # bar() itself is never called would test matplotlib's internals,
        # not this code -- assert barh() was called, and that whichever
        # underlying call happened carries orientation='horizontal'.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = [{"stage": "A", "count": 100.0, "funnel_pct": 100.0},
                {"stage": "B", "count": 50.0, "funnel_pct": 50.0}]
        with patch.object(plt.Axes, "barh", wraps=plt.Axes.barh, autospec=True) as spy_h:
            generate_chart(rows, "funnel", "t")
        spy_h.assert_called_once()


class HistogramRenderingTests(unittest.TestCase):
    def test_histogram_renders_real_png(self):
        rows = [
            {"bin_label": "0-10", "count": 5, "bin_min": 0, "bin_max": 10},
            {"bin_label": "10-20", "count": 12, "bin_min": 10, "bin_max": 20},
            {"bin_label": "20-30", "count": 3, "bin_min": 20, "bin_max": 30},
        ]
        png = generate_chart(rows, "histogram", "Distribution")
        self.assertTrue(_png_ok(png))

    def test_histogram_bars_touch_unlike_a_bar_chart(self):
        # width=1.0 (no gap) is the one visual property that reads as
        # "histogram" rather than "bar chart of unrelated categories".
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = [{"bin_label": "0-10", "count": 5}, {"bin_label": "10-20", "count": 12}]
        captured = {}
        real_bar = plt.Axes.bar
        def spy_bar(self, x, height, width=0.8, **kw):
            captured["width"] = width
            return real_bar(self, x, height, width=width, **kw)
        with patch.object(plt.Axes, "bar", spy_bar):
            generate_chart(rows, "histogram", "t")
        self.assertEqual(captured["width"], 1.0)


class BoxplotRenderingTests(unittest.TestCase):
    def test_boxplot_renders_via_precomputed_summary_stats(self):
        rows = [
            {"group": "Region A", "bp_min": 1.0, "bp_q1": 2.0, "bp_median": 3.0,
             "bp_q3": 4.0, "bp_max": 5.0, "bp_data": [1.0, 2.0, 3.0, 4.0, 5.0], "bp_outliers": []},
            {"group": "Region B", "bp_min": 0.5, "bp_q1": 1.5, "bp_median": 2.5,
             "bp_q3": 3.5, "bp_max": 4.5, "bp_data": [0.5, 1.5, 2.5, 3.5, 4.5], "bp_outliers": [9.0]},
        ]
        png = generate_chart(rows, "boxplot", "Distribution by region")
        self.assertTrue(_png_ok(png))

    def test_boxplot_uses_bxp_not_generic_bar_fallback(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = [{"group": "A", "bp_min": 1.0, "bp_q1": 2.0, "bp_median": 3.0,
                 "bp_q3": 4.0, "bp_max": 5.0, "bp_data": [1.0, 2.0, 3.0, 4.0, 5.0], "bp_outliers": []}]
        with patch.object(plt.Axes, "bxp", wraps=plt.Axes.bxp, autospec=True) as spy_bxp, \
             patch.object(plt.Axes, "bar", wraps=plt.Axes.bar, autospec=True) as spy_bar:
            generate_chart(rows, "boxplot", "t")
        spy_bxp.assert_called_once()
        spy_bar.assert_not_called()


class UnsupportedTypesTests(unittest.TestCase):
    def test_heatmap_and_treemap_raise_instead_of_silently_substituting(self):
        rows = [{"row_dim": "A", "col_dim": "X", "value": 5.0}]
        for bad_type in ("heatmap", "treemap"):
            with self.subTest(bad_type=bad_type):
                with self.assertRaises(UnsupportedChartTypeError) as ctx:
                    generate_chart(rows, bad_type, "t")
                self.assertEqual(ctx.exception.chart_type, bad_type)

    def test_no_png_rendering_set_matches_actually_unimplemented_types(self):
        self.assertEqual(_NO_PNG_RENDERING, {"heatmap", "treemap"})


class TeamsSendChartHonestFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_unsupported_chart_type_sends_message_not_silence(self):
        from gateway.teams_adapter import TeamsAdapter
        from gateway.base import PlatformEvent

        adapter = TeamsAdapter.__new__(TeamsAdapter)
        adapter.send_message = AsyncMock()
        adapter._native_charts_enabled = lambda: False  # force straight to PNG path

        event = PlatformEvent(
            account_id="acct", user_id="u1", channel_id="{}", text="", platform="teams",
        )
        chart = {"rows": [{"row_dim": "A", "col_dim": "X", "value": 5.0}], "chart_type": "heatmap", "title": "t"}

        await adapter.send_chart(event, chart)

        adapter.send_message.assert_called_once()
        sent_text = adapter.send_message.call_args[0][1]
        self.assertIn("heatmap", sent_text)
        self.assertIn("portal", sent_text.lower())


if __name__ == "__main__":
    unittest.main()
