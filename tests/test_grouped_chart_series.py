# -*- coding: utf-8 -*-
"""tests/test_grouped_chart_series.py

L5 · A second dimension becomes a series, not a longer axis.

"Compare revenue by warehouse for the last three months" returns a row per
warehouse PER MONTH. The chart stack plotted it as ONE series, so the category
axis carried each warehouse once per month and the line walked between rows
belonging to different months. The reader saw a single jagged series that was
really three flat ones interleaved.

The earlier attempt at this -- core/chart.py::_category_axis, shipped in
a8f4842 -- de-duplicated the AXIS. That was the wrong layer twice over: it
answers the wrong question (a repeated axis is a symptom; the missing series is
the defect), and it is only reachable when the spec leaves x unpinned, which it
never does. Its own test, test_this_is_only_the_fallback, says so.

The fix is upstream of both. core/chart_spec.py decides WHICH column splits the
rows -- it is the only layer that knows every column's role -- and core/chart.py
pivots the rows before the projection at build_chart_payload, which is the last
place the grouping column's VALUES still exist. What comes out is the shape both
templates already draw: one series per key in the row dict. So this needs no new
renderer branch, no new payload concept and no new chart type.

Every test executes the real code. The Python half calls infer_chart_spec and
build_chart_payload; the JavaScript half lifts buildChartOption and
buildDashboardOption out of the real templates and runs them under dukpy, on
both pages, because the two are separate copies that have drifted before.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.chart import build_chart_payload  # noqa: E402
from core.chart_spec import infer_chart_spec  # noqa: E402

PERIODS = ("2026-03", "2026-04", "2026-05", "2026-06")
WAREHOUSES = (("Halifax", 1000.0), ("Calgary", 2000.0), ("Moncton", 3000.0))
GROUPED_Q = "Compare revenue by warehouse for the last 3 months"


def grid(measure: str = "REVENUE_AMT") -> list[dict]:
    """A row per warehouse per period — the shape the live question returns."""
    return [{"WHS_NM": name, "PERIOD": period, measure: value}
            for period in PERIODS for name, value in WAREHOUSES]


def flat(measure: str = "REVENUE_AMT") -> list[dict]:
    return [{"WHS_NM": name, measure: value} for name, value in WAREHOUSES]


def payload(rows, question=GROUPED_Q, chart_type=None, **kw):
    return build_chart_payload(rows, chart_type, question=question,
                               column_formats={"REVENUE_AMT": "currency"}, **kw)


class TestTheGroupingColumnIsFound(unittest.TestCase):
    """core/chart_spec.py::_series_dimension — the choice, before any drawing."""

    def series_of(self, rows, question=GROUPED_Q):
        spec = infer_chart_spec(rows, question=question)
        return (spec.get("series") or {}).get("column")

    def test_the_calendar_becomes_the_series_when_the_warehouse_is_the_axis(self):
        self.assertEqual(self.series_of(grid()), "PERIOD")

    def test_and_the_warehouse_when_the_calendar_is_the_axis(self):
        # The same grid, asked the other way round. Whichever column the spec
        # pins as x, the OTHER one becomes the series -- so the reader gets the
        # chart their question implies rather than a fixed choice.
        spec = infer_chart_spec(grid(), question="revenue by warehouse by month")
        self.assertEqual(spec["x"]["column"], "PERIOD")
        self.assertEqual(spec["series"]["column"], "WHS_NM")

    def test_a_one_dimensional_result_has_no_series(self):
        # The gate that makes every existing chart byte-identical.
        self.assertIsNone(self.series_of(flat(), "revenue by warehouse"))

    def test_a_column_fixed_per_category_is_not_a_series(self):
        # REGION is determined by the warehouse, so it does not explain why the
        # warehouse repeats. Grouping by it would draw three series that are
        # each mostly gaps.
        region = {"Halifax": "East", "Calgary": "West", "Moncton": "East"}
        rows = [{"WHS_NM": w, "REGION": region[w], "PERIOD": p, "REVENUE_AMT": v}
                for p in PERIODS for w, v in WAREHOUSES]
        self.assertEqual(self.series_of(rows), "PERIOD")

    def test_a_third_dimension_gets_no_series_at_all(self):
        # Warehouse x period x channel. A cell appears twice, and no chart with
        # two axes can show three dimensions -- so it draws as it did before
        # rather than picking one arbitrarily.
        rows = [{"WHS_NM": w, "PERIOD": p, "CHANNEL": c, "REVENUE_AMT": v}
                for p in PERIODS for w, v in WAREHOUSES for c in ("Web", "Trade")]
        self.assertIsNone(self.series_of(rows))

    def test_a_constant_column_splits_nothing(self):
        rows = [{"WHS_NM": w, "PERIOD": p, "CO": "ACME", "REVENUE_AMT": v}
                for p in PERIODS for w, v in WAREHOUSES]
        self.assertEqual(self.series_of(rows), "PERIOD")

    def test_more_groups_than_the_palette_is_refused(self):
        # Nine periods against the eight-colour validated palette. A ninth
        # series either repeats a colour or is silently dropped by one page and
        # not the other.
        nine = [f"2026-{m:02d}" for m in range(1, 10)]
        rows = [{"WHS_NM": w, "PERIOD": p, "REVENUE_AMT": v}
                for p in nine for w, v in WAREHOUSES]
        self.assertIsNone(self.series_of(rows))

    def test_eight_groups_is_still_drawn(self):
        # Guards the guard: an off-by-one at the cap would silently withdraw
        # the feature for the common twelve-week case.
        eight = [f"2026-{m:02d}" for m in range(1, 9)]
        rows = [{"WHS_NM": w, "PERIOD": p, "REVENUE_AMT": v}
                for p in eight for w, v in WAREHOUSES]
        self.assertEqual(self.series_of(rows), "PERIOD")

    def test_a_group_value_that_collides_with_a_column_name_is_refused(self):
        # The pivot uses group values as row keys, so a period literally named
        # REVENUE_AMT would overwrite the measure.
        rows = [{"WHS_NM": w, "PERIOD": p, "REVENUE_AMT": v}
                for p in ("REVENUE_AMT", "2026-04") for w, v in WAREHOUSES]
        self.assertIsNone(self.series_of(rows))

    def test_a_one_dimensional_result_with_a_spare_dimension_still_has_none(self):
        # The distinct-x gate is the headline rule, but it must also hold when
        # there IS another categorical column sitting there to be chosen.
        region = {"Halifax": "East", "Calgary": "West", "Moncton": "East"}
        rows = [{"WHS_NM": w, "REGION": region[w], "REVENUE_AMT": v}
                for w, v in WAREHOUSES]
        self.assertIsNone(self.series_of(rows, "revenue by warehouse"))

    def test_a_repeated_cell_is_refused_even_with_one_candidate(self):
        # The grid delivered twice: PERIOD is the only candidate and it still
        # does not account for the rows, because each cell appears twice. Some
        # dimension the result does not carry is doing that, and a chart cannot
        # draw what it cannot see.
        self.assertIsNone(self.series_of(grid() + grid()))

    def test_two_candidates_that_both_fit_are_an_ambiguity_not_a_choice(self):
        # Warehouse x period, with a channel that also forms a clean grid
        # against the warehouse. Either could be the series and nothing here
        # can say which the reader meant, so neither is picked.
        rows = [{"WHS_NM": w, "PERIOD": p, "CHANNEL": c, "REVENUE_AMT": v}
                for (p, c) in (("2026-03", "Web"), ("2026-04", "Trade"))
                for w, v in WAREHOUSES]
        self.assertIsNone(self.series_of(rows))

    def test_a_grid_too_sparse_to_be_a_grid_is_refused(self):
        # Four warehouses, four periods, seven cells filled. Four series of one
        # or two points each is a scatter of dots wearing a legend.
        cells = [("Halifax", "2026-03"), ("Halifax", "2026-04"),
                 ("Calgary", "2026-05"), ("Moncton", "2026-06"),
                 ("Sydney", "2026-03"), ("Sydney", "2026-06"),
                 ("Calgary", "2026-03")]
        rows = [{"WHS_NM": w, "PERIOD": p, "REVENUE_AMT": 10.0} for w, p in cells]
        self.assertIsNone(self.series_of(rows))

    def test_a_nearly_empty_cross_product_is_refused(self):
        # Twelve customers in three segments, one row each: the segments would
        # be three series that are two-thirds gaps.
        rows = [{"SEGMENT": f"S{i % 3}", "CUST_NM": f"C{i}", "REVENUE_AMT": 10.0}
                for i in range(12)]
        self.assertIsNone(self.series_of(rows, "revenue by segment"))


class TestThePivot(unittest.TestCase):
    """core/chart.py::build_chart_payload — the rows the browser receives."""

    def test_one_row_per_category_with_a_key_per_group(self):
        built = payload(grid())
        self.assertEqual(built["x_key"], "WHS_NM")
        self.assertEqual(built["y_keys"], list(PERIODS))
        self.assertEqual(len(built["rows"]), len(WAREHOUSES))
        self.assertEqual(built["rows"][0],
                         {"WHS_NM": "Halifax", **{p: 1000.0 for p in PERIODS}})

    def test_the_payload_says_what_it_grouped_by(self):
        built = payload(grid())
        self.assertEqual(built["grouped_by"], "PERIOD")
        self.assertEqual(built["grouped_measure"], "REVENUE_AMT")

    def test_a_one_dimensional_payload_is_untouched(self):
        built = payload(flat(), "revenue by warehouse")
        self.assertIsNone(built["grouped_by"])
        self.assertEqual(built["y_keys"], ["REVENUE_AMT"])
        self.assertEqual(len(built["rows"]), len(WAREHOUSES))

    def test_a_missing_cell_is_a_gap_and_never_a_zero(self):
        # A warehouse that opened in the second month. Zero is a claim, and
        # nobody made it.
        rows = [r for r in grid()
                if not (r["WHS_NM"] == "Moncton" and r["PERIOD"] == PERIODS[0])]
        built = payload(rows)
        moncton = next(r for r in built["rows"] if r["WHS_NM"] == "Moncton")
        self.assertIsNone(moncton[PERIODS[0]])
        self.assertEqual(moncton[PERIODS[1]], 3000.0)

    def test_a_null_measure_stays_null_and_does_not_become_zero(self):
        # A row the result DID return, whose measure is null. Coercing it to
        # zero draws a bar claiming the warehouse sold nothing that month,
        # which is a different statement from "not reported".
        rows = grid()
        rows[0]["REVENUE_AMT"] = None
        built = payload(rows)
        first = next(r for r in built["rows"] if r["WHS_NM"] == rows[0]["WHS_NM"])
        self.assertIsNone(first[rows[0]["PERIOD"]])

    def test_the_group_keys_carry_the_measures_number_format(self):
        # The browser reads column_roles for the tooltip's format. The keys are
        # now group VALUES, which no column describes.
        built = payload(grid())
        self.assertEqual(built["column_roles"]["2026-03"]["format"], "currency")
        self.assertEqual(built["column_roles"]["2026-03"]["label"], "2026-03")

    def test_a_real_columns_label_is_never_rewritten(self):
        # The registration above is keyed on group values. Applied when the
        # pivot did NOT run, it would overwrite a real column's display label
        # with its raw name -- L4, reintroduced through the fix for L5.
        built = payload(flat(), "revenue by warehouse")
        self.assertEqual(built["column_roles"]["REVENUE_AMT"]["label"],
                         "REVENUE AMT")

    def test_the_totals_survive_the_pivot(self):
        # The invariant: reshaping must not change what the chart adds up to.
        built = payload(grid())
        drawn = sum(value for row in built["rows"] for key, value in row.items()
                    if key in built["y_keys"] and value is not None)
        self.assertEqual(drawn, sum(r["REVENUE_AMT"] for r in grid()))

    def test_annotations_are_withheld_from_a_grouped_chart(self):
        # A mark point names one period and anchors to one series. With four
        # series it would land on whichever came first.
        annotations = {"biggest_period_gain": {"to_period": "2026-04",
                                               "pct_change": 12.0}}
        self.assertIsNone(payload(grid(), annotations=annotations)
                          .get("annotations"))
        one_d = payload(
            [{"PERIOD": p, "REVENUE_AMT": v}
             for p, v in zip(PERIODS, (10.0, 20.0, 30.0, 40.0))],
            "monthly revenue trend", annotations=annotations)
        self.assertIsNotNone(one_d.get("annotations"))

    def test_they_are_withheld_when_the_calendar_is_the_axis_too(self):
        # The orientation where it actually matters: the annotation names a
        # period, the axis IS periods, so the mark point finds its anchor and
        # lands on whichever of the four warehouse series happens to be first.
        annotations = {"biggest_period_gain": {"to_period": "2026-04",
                                               "pct_change": 12.0}}
        built = payload(grid(), "revenue by warehouse by month",
                        annotations=annotations)
        self.assertEqual(built["grouped_by"], "WHS_NM")
        self.assertIn("2026-04", built["rows"][1].values())
        self.assertIsNone(built.get("annotations"))


class TestTheControlsStayHonest(unittest.TestCase):
    """A chart type the reader can pick has to be one the data can draw."""

    def test_scatter_is_withdrawn_when_the_series_slot_takes_the_measure(self):
        # A grouped chart draws ONE measure, and a scatter needs two. Left in
        # `allowed`, the button renders an empty chart.
        rows = [{"WHS_NM": w, "PERIOD": p, "REVENUE_AMT": v, "COST_AMT": v * 0.6}
                for p in PERIODS for w, v in WAREHOUSES]
        spec = infer_chart_spec(
            rows, question="compare revenue and cost by warehouse over 3 months")
        self.assertEqual(spec["series"]["column"], "PERIOD")
        self.assertNotIn("scatter", spec["allowed_types"])
        self.assertEqual([c["column"] for c in spec["y"]], ["REVENUE_AMT"])

    def test_the_reader_is_told_which_measure_was_set_aside(self):
        rows = [{"WHS_NM": w, "PERIOD": p, "REVENUE_AMT": v, "COST_AMT": v * 0.6}
                for p in PERIODS for w, v in WAREHOUSES]
        spec = infer_chart_spec(
            rows, question="compare revenue and cost by warehouse over 3 months")
        self.assertTrue(any("COST" in w for w in spec["warnings"]), spec["warnings"])

    def test_a_scatter_on_a_one_dimensional_result_is_untouched(self):
        rows = [{"WHS_NM": w, "REVENUE_AMT": v, "COST_AMT": v * 0.6}
                for w, v in WAREHOUSES]
        spec = infer_chart_spec(rows, question="revenue against cost by warehouse")
        self.assertIn("scatter", spec["allowed_types"])

    def test_a_pie_of_a_measure_that_cannot_be_totalled_becomes_a_bar(self):
        # The renderer totals rows sharing a slice name -- it has to, or a
        # category is drawn twice. A total is only a whole when the measure
        # adds up, and a margin percentage does not. Commit 4452268 fixed the
        # drawing and said this choice belongs upstream; this is upstream.
        rows = [{"WHS_NM": w, "PERIOD": p, "GRS_MARG_PCT": v / 100}
                for p in PERIODS[:2] for w, v in WAREHOUSES]
        spec = infer_chart_spec(
            rows, question="what is the share of gross margin percent by warehouse")
        self.assertEqual(spec["recommended_type"], "bar")
        self.assertNotIn("pie", spec["allowed_types"])
        self.assertTrue(spec["warnings"])

    def test_a_pie_of_an_additive_measure_over_repeats_is_still_offered(self):
        # Refusing is about the measure, not about repetition. Revenue sums.
        rows = [{"WHS_NM": w, "PERIOD": p, "REVENUE_AMT": v}
                for p in PERIODS[:2] for w, v in WAREHOUSES]
        spec = infer_chart_spec(
            rows, question="what is the share of revenue by warehouse")
        self.assertIn(spec["recommended_type"], {"pie", "donut", "bar"})
        if spec["recommended_type"] in {"pie", "donut"}:
            self.assertIn("pie", spec["allowed_types"])


dukpy = pytest.importorskip(
    "dukpy",
    reason="a JavaScript engine is required to EXECUTE the two chart builders; "
           "the whole claim of this design is that the pivoted payload needs no "
           "renderer change, and only running them can show that",
)
from tests.test_chart_annotation_language import _build, PAGES  # noqa: E402


class TestBothPagesDrawItWithoutAChange:
    """The design's central claim, executed on both real templates."""

    def option(self, page, expression, rows=None, question=GROUPED_Q):
        return json.loads(_build(page, "en", payload(rows or grid(), question),
                                 f"JSON.stringify({expression})"))

    @pytest.mark.parametrize("page", PAGES)
    def test_one_series_per_period(self, page):
        drawn = self.option(page, "opt.series.map(function (s) { return s.name })")
        assert drawn == list(PERIODS), page

    @pytest.mark.parametrize("page", PAGES)
    def test_each_warehouse_appears_on_the_axis_exactly_once(self, page):
        axis = self.option(
            page, "(Array.isArray(opt.xAxis) ? opt.xAxis[0] : opt.xAxis).data")
        assert axis == [name for name, _ in WAREHOUSES], page

    @pytest.mark.parametrize("page", PAGES)
    def test_a_series_carries_one_value_per_warehouse(self, page):
        first = self.option(page, "opt.series[0].data")
        assert first == [1000, 2000, 3000], page

    @pytest.mark.parametrize("page", PAGES)
    def test_the_legend_names_the_periods(self, page):
        legend = self.option(page, "(opt.legend && opt.legend.data) || []")
        assert legend == list(PERIODS), page

    @pytest.mark.parametrize("page", PAGES)
    def test_the_tooltip_compares_the_whole_group(self, page):
        # 'item' would show the one segment under the cursor on a chart whose
        # entire purpose is comparing the series side by side.
        assert self.option(page, "opt.tooltip.trigger") == "axis", page

    @pytest.mark.parametrize("page", PAGES)
    def test_a_one_dimensional_chart_still_draws_one_series(self, page):
        names = self.option(page, "opt.series.length",
                            rows=flat(), question="revenue by warehouse")
        assert names == 1, page


class TestTheCategoryCapRanksTheWholeBar:
    """Found while checking what the pivot would do to the density cap.

    The cap ranked categories by yKeys[0] alone, so a category whose magnitude
    sits in a later series was dropped -- and the caption still read "Showing
    the N largest". Executed against the shipped code, the single largest value
    in the whole chart was the one thrown away. A grouped chart makes that the
    ordinary case: the categories would be ranked by whichever period came
    first.
    """

    WIDE = {"rows": [{"C": f"c{i:02d}", "S1": 100 - i, "S2": 1.0} for i in range(21)]
                    + [{"C": "BIG", "S1": 0.0, "S2": 999999.0}],
            "x_key": "C", "y_keys": ["S1", "S2"], "chart_type": "bar"}

    def categories(self, payload_):
        return json.loads(_build(
            "portal_chat.html", "en", payload_,
            "JSON.stringify((Array.isArray(opt.xAxis) ? opt.xAxis[0] : opt.xAxis).data"
            " || (Array.isArray(opt.yAxis) ? opt.yAxis[0] : opt.yAxis).data)"))

    def test_the_largest_bar_in_the_chart_is_kept(self):
        assert "BIG" in self.categories(self.WIDE)

    def test_the_cap_itself_still_applies(self):
        # Not "keep everything": the caption promises twenty.
        assert len(self.categories(self.WIDE)) == 20

    def test_a_single_measure_chart_ranks_exactly_as_before(self):
        single = {**self.WIDE, "y_keys": ["S1"],
                  "rows": [{"C": f"c{i:02d}", "S1": 100 - i} for i in range(25)]}
        assert self.categories(single)[0] == "c00"


if __name__ == "__main__":
    unittest.main()
