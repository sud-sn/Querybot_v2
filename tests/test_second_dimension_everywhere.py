# -*- coding: utf-8 -*-
"""tests/test_second_dimension_everywhere.py

The same defect in four more places, found by sweeping for it after fixing two.

"Compare revenue by warehouse for the last 3 months" returns a row per
warehouse PER MONTH. Every consumer that reads those rows as though each
warehouse appeared once produces a confident, wrong number:

  core/insight.py            called perfectly flat warehouses "decreasing" and
                             named a MONTH as the leading category at 9.4% --
                             and this is the brief that feeds KEY INSIGHTS, the
                             chart annotations and the narration prompt, so the
                             earlier fixes never reached the live path.
  core/response_builder.py   build_answer led with "2026-05 closed at 900,000":
                             a month's name against one warehouse's row.
  core/analysis_evidence.py  concentration_findings divided one row by a total
                             summed over every period, dropping a genuine 37.5%
                             leader below the threshold so the finding vanished
                             rather than merely being wrong.
  portal_chat.html           the pie drew Halifax twice at 30% each against a
                             true share of 60%.

One rule underneath all four, in core.analysis_contract.collapse_rows_by_label:
merge to one row per label, and refuse when the measure may not be summed.

Every test runs the real function on the live shape.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.analysis_contract import collapse_rows_by_label  # noqa: E402

PERIODS = ("2026-03", "2026-04", "2026-05", "2026-06")
WAREHOUSES = (("Halifax", 1_200_000.0), ("Calgary", 1_100_000.0),
              ("Vancouver", 900_000.0))
QUESTION = "compare revenue by warehouse for the last 3 months"


def grid(value_col: str = "REVENUE_AMT") -> list[dict]:
    """A row per warehouse per period, every warehouse flat across the quarter."""
    return [{"PERIOD": period, "WHS_NM": name, value_col: value}
            for period in PERIODS for name, value in WAREHOUSES]


def flat(value_col: str = "REVENUE_AMT") -> list[dict]:
    """The same totals, already one row per warehouse."""
    return [{"WHS_NM": name, value_col: value * len(PERIODS)}
            for name, value in WAREHOUSES]


def coded_grid(value_col: str = "REVENUE_AMT") -> list[dict]:
    """The same grid with the warehouse as a numeric code.

    The period is then the only TEXT column, so the narrative has nothing else
    to speak about and ``_narrative_label_column`` hands back the calendar --
    repeating labels and all. This is the shape where the distinct-label guard
    is the only thing standing between the reader and a trend built out of
    unrelated rows, and it is an ordinary result: a warehouse, a product or a
    customer identified by its code rather than its name.
    """
    return [{"PERIOD": period, value_col: value, "WHS_ID": code}
            for period in PERIODS
            for code, (_, value) in zip((101, 102, 103), WAREHOUSES)]


class TestTheRuleItself(unittest.TestCase):

    def test_repeats_are_summed_when_the_measure_adds_up(self):
        self.assertEqual(
            dict(collapse_rows_by_label(grid(), "WHS_NM", "REVENUE_AMT")),
            {"Halifax": 4_800_000.0, "Calgary": 4_400_000.0,
             "Vancouver": 3_600_000.0})

    def test_a_percentage_is_refused_rather_than_summed(self):
        self.assertIsNone(
            collapse_rows_by_label(grid("GRS_MARG_PCT"), "WHS_NM", "GRS_MARG_PCT"))

    def test_a_balance_is_refused_because_the_axis_collapsed_is_time(self):
        # Semi-additive means "may be summed across warehouses, never across
        # months", and months are exactly what this merges.
        self.assertIsNone(
            collapse_rows_by_label(grid("BAL_VAL_AMT"), "WHS_NM", "BAL_VAL_AMT"))

    def test_an_unrecognised_measure_is_not_guessed_at(self):
        self.assertIsNone(
            collapse_rows_by_label(grid("SOME_COLUMN"), "WHS_NM", "SOME_COLUMN"))

    def test_one_row_per_label_is_returned_untouched_whatever_the_measure(self):
        # Nothing to add, so the additivity rule never applies -- a percentage
        # with one row per warehouse still ranks honestly.
        pairs = collapse_rows_by_label(flat("GRS_MARG_PCT"), "WHS_NM", "GRS_MARG_PCT")
        self.assertEqual([name for name, _ in pairs],
                         [name for name, _ in WAREHOUSES])

    def test_the_measure_name_can_be_given_separately(self):
        # For a caller that has projected its rows onto working keys and no
        # longer carries the column the rule needs to judge.
        rows = [{"_l": name, "_v": value} for name, value in WAREHOUSES] * 2
        self.assertIsNone(
            collapse_rows_by_label(rows, "_l", "_v", measure_name="GRS_MARG_PCT"))
        self.assertIsNotNone(
            collapse_rows_by_label(rows, "_l", "_v", measure_name="REVENUE_AMT"))

    def test_unreadable_values_are_skipped_not_fatal(self):
        rows = [{"WHS_NM": "A", "REVENUE_AMT": "1,200"},
                {"WHS_NM": "B", "REVENUE_AMT": None},
                {"WHS_NM": "C", "REVENUE_AMT": "n/a"}]
        self.assertEqual(collapse_rows_by_label(rows, "WHS_NM", "REVENUE_AMT"),
                         [("A", 1200.0)])

    def test_nothing_in_nothing_out(self):
        self.assertEqual(collapse_rows_by_label([], "WHS_NM", "REVENUE_AMT"), [])


class TestTheDataBrief(unittest.TestCase):
    """core/insight.py — the brief the live answer path actually uses."""

    def brief(self, value_col="REVENUE_AMT", rows=None):
        from core.insight import compute_data_brief

        return compute_data_brief(rows if rows is not None else grid(value_col),
                                  QUESTION)

    def test_flat_warehouses_are_not_a_falling_trend(self):
        brief = self.brief()
        self.assertEqual(brief.get("mode"), "ranking")
        self.assertIsNone(brief.get("time_series"))

    def test_the_breakdown_is_by_warehouse_not_by_month(self):
        cat = self.brief()["category_breakdown"]
        self.assertEqual(cat["label_column"], "WHS_NM")
        self.assertEqual(cat["category_count"], 3)

    def test_the_leader_is_a_warehouse_at_its_real_share(self):
        cat = self.brief()["category_breakdown"]
        self.assertEqual(cat["top_5"][0]["label"], "Halifax")
        self.assertEqual(cat["top_5"][0]["value"], 4_800_000.0)
        self.assertEqual(cat["leader_share_pct"], 37.5)

    def test_the_gap_is_between_two_different_warehouses(self):
        cat = self.brief()["category_breakdown"]
        self.assertEqual(cat["leader_vs_runner_up_gap"], 400_000.0)

    def test_the_grouped_and_the_flat_form_agree(self):
        # The invariant that matters: the same totals must read the same way
        # whether the warehouse rows arrive pre-aggregated or per month.
        grouped = self.brief()["category_breakdown"]
        already = self.brief(rows=flat())["category_breakdown"]
        for key in ("category_count", "leader_share_pct", "leader_vs_runner_up_gap"):
            self.assertEqual(grouped[key], already[key], key)

    def test_no_leader_is_invented_for_a_percentage(self):
        cat = self.brief("GRS_MARG_PCT")["category_breakdown"]
        self.assertEqual(cat["top_5"], [])
        self.assertIsNone(cat.get("leader_share_pct"))

    def test_a_repeating_calendar_is_not_a_series_even_when_it_is_the_label(self):
        # Every month totals exactly 3,200,000. Read as a flat list the twelve
        # rows walk 1.2M, 1.1M, 0.9M four times over, and a reader that trusts
        # the first and last of them calls a perfectly level quarter a fall.
        brief = self.brief(rows=coded_grid())
        self.assertEqual(brief.get("mode"), "ranking")
        self.assertIsNone(brief.get("time_series"))
        self.assertEqual(brief["category_breakdown"]["label_column"], "PERIOD")
        self.assertEqual(
            [item["value"] for item in brief["category_breakdown"]["top_5"]],
            [3_200_000.0] * 4)

    def test_a_genuine_series_is_still_a_series(self):
        rows = [{"PERIOD": p, "REVENUE_AMT": v}
                for p, v in zip(PERIODS, (100.0, 150.0, 200.0, 300.0))]
        brief = self.brief(rows=rows)
        self.assertEqual(brief["mode"], "time_series")
        self.assertEqual(brief["time_series"]["direction"], "increasing")


class TestTheHeadlineCard(unittest.TestCase):
    """core/response_builder.build_answer — what the reader sees first."""

    def answer(self, value_col="REVENUE_AMT", rows=None):
        from core.response_builder import build_answer

        return build_answer(rows if rows is not None else grid(value_col), QUESTION)

    def test_the_headline_names_a_warehouse_not_a_month(self):
        headline = self.answer()["headline"]
        self.assertIn("Halifax", headline)
        for period in PERIODS:
            self.assertNotIn(period, headline)

    def test_the_headline_carries_the_warehouses_whole_total(self):
        self.assertIn("4,800,000", self.answer()["headline"])

    def test_the_gap_is_to_a_different_warehouse(self):
        # Was a month-over-month step for the same warehouse, printed as though
        # it were the distance to a competitor.
        self.assertIn("400,000", self.answer()["comparison"])

    def test_a_percentage_gets_no_invented_leader(self):
        answer = self.answer("GRS_MARG_PCT")
        self.assertNotIn("leads", answer["headline"])

    def test_a_true_series_still_closes_on_its_last_period(self):
        rows = [{"PERIOD": p, "REVENUE_AMT": v}
                for p, v in zip(PERIODS, (100.0, 150.0, 200.0, 300.0))]
        self.assertIn("2026-06", self.answer(rows=rows)["headline"])

    def test_no_fall_is_announced_over_a_repeating_calendar(self):
        # The other half of the same guard. Without it the card takes the first
        # and last ROWS -- one warehouse in March against another in June --
        # and reports "2026-06 closed at 900,000", down 25%, for a quarter in
        # which nothing moved at all.
        answer = self.answer(rows=coded_grid())
        self.assertNotIn("closed at", answer["headline"])
        self.assertNotIn("900,000", answer["headline"])
        self.assertIn("3,200,000", answer["headline"])

    def test_a_single_dimension_ranking_is_untouched(self):
        answer = self.answer(rows=flat())
        self.assertIn("Halifax", answer["headline"])
        self.assertIn("400,000", answer["comparison"])


class TestTheConcentrationDetector(unittest.TestCase):
    """core/analysis_evidence.concentration_findings — the finding vanished."""

    def leader(self, value_col="REVENUE_AMT", rows=None):
        from core.analysis_evidence import CONCENTRATION_LEADER, concentration_findings

        found = concentration_findings(
            rows if rows is not None else grid(value_col), value_col, "WHS_NM")
        return next((f for f in found if f.kind == CONCENTRATION_LEADER), None)

    def test_the_leader_is_reported_at_its_real_share(self):
        leader = self.leader()
        self.assertIsNotNone(leader, "the leader disappeared entirely")
        self.assertEqual(leader.numbers["share"], 37.5)
        self.assertEqual(leader.numbers["value"], 4_800_000.0)
        self.assertEqual(leader.labels["leader"], "Halifax")

    def test_the_grouped_and_the_flat_form_agree(self):
        grouped, already = self.leader(), self.leader(rows=flat())
        self.assertEqual(grouped.numbers, already.numbers)
        self.assertEqual(grouped.labels, already.labels)

    def test_a_percentage_reports_nothing_rather_than_a_share_of_nothing(self):
        from core.analysis_evidence import concentration_findings

        self.assertEqual(
            concentration_findings(grid("GRS_MARG_PCT"), "GRS_MARG_PCT", "WHS_NM"), [])

    def test_a_below_threshold_leader_is_still_withheld(self):
        # The guard restores real shares; it must not start reporting leaders
        # the threshold exists to suppress.
        rows = [{"WHS_NM": f"W{n}", "REVENUE_AMT": 100.0} for n in range(10)]
        self.assertIsNone(self.leader(rows=rows))


class TestThePieDrawsEachCategoryOnce(unittest.TestCase):
    """portal_chat.html — the same defect, in the renderer.

    Built with ``rows.map``, a grid gave the pie one slice per ROW, so Halifax
    was drawn once per month. ECharts happily draws four "Halifax" slices; the
    legend then keys its percentages by NAME, so the label read whichever row
    came last -- 30% beside a share that is really 60%.

    The real ``buildChartOption`` is lifted out of the real template and
    EXECUTED here. Nothing in this class reads the template as text.
    """

    @classmethod
    def setUpClass(cls):
        import pytest

        pytest.importorskip(
            "dukpy",
            reason="a JavaScript engine is required to EXECUTE the pie builder; "
                   "reading it and assuming what it draws is the defect this "
                   "class exists to catch")
        from tests.test_chart_annotation_language import _build

        cls._build = staticmethod(_build)

    #: A grid, in the shape the chart payload actually carries.
    PAYLOAD = {"rows": [{"WHS_NM": name, "REVENUE_AMT": value}
                        for _ in PERIODS for name, value in WAREHOUSES],
               "x_key": "WHS_NM", "y_keys": ["REVENUE_AMT"],
               "chart_type": "pie"}

    def slices(self, payload=None):
        import json

        return json.loads(self._build(
            "portal_chat.html", "en", payload or self.PAYLOAD,
            "JSON.stringify(opt.series[0].data)"))

    def test_each_warehouse_gets_exactly_one_slice(self):
        names = [item["name"] for item in self.slices()]
        self.assertEqual(sorted(names), ["Calgary", "Halifax", "Vancouver"])

    def test_the_slice_carries_the_warehouses_whole_total(self):
        by_name = {item["name"]: item["value"] for item in self.slices()}
        self.assertEqual(by_name["Halifax"], 4_800_000.0)
        self.assertEqual(by_name["Vancouver"], 3_600_000.0)

    def test_the_legend_reports_the_real_share(self):
        # The number the reader sees. 4.8M of 12.8M is 37.5%; drawn per row it
        # read 9.4%, the share of a single month.
        drawn = self._build("portal_chat.html", "en", self.PAYLOAD,
                            "opt.legend.formatter('Halifax')")
        self.assertIn("37.5%", drawn)
        self.assertNotIn("9.4%", drawn)

    def test_the_slices_add_up_to_the_whole(self):
        # A pie is a part-to-whole picture, so this is the invariant that
        # matters more than any single slice.
        self.assertEqual(sum(item["value"] for item in self.slices()),
                         sum(value for _, value in WAREHOUSES) * len(PERIODS))

    def test_a_result_that_was_already_one_row_per_category_is_unchanged(self):
        one_each = {**self.PAYLOAD,
                    "rows": [{"WHS_NM": name, "REVENUE_AMT": value}
                             for name, value in WAREHOUSES]}
        self.assertEqual(
            [(item["name"], item["value"]) for item in self.slices(one_each)],
            [(name, value) for name, value in WAREHOUSES])

    def test_a_row_with_no_category_still_gets_its_own_slice(self):
        # Grouping must not swallow the unlabelled rows into a neighbour: they
        # share one bucket of their own, which is what the reader is told.
        blanks = {**self.PAYLOAD,
                  "rows": [{"WHS_NM": "Halifax", "REVENUE_AMT": 10.0},
                           {"WHS_NM": None, "REVENUE_AMT": 30.0},
                           {"WHS_NM": None, "REVENUE_AMT": 20.0}]}
        by_name = {item["name"]: item["value"] for item in self.slices(blanks)}
        self.assertEqual(by_name["Unspecified"], 50.0)
        self.assertEqual(by_name["Halifax"], 10.0)


if __name__ == "__main__":
    unittest.main()
