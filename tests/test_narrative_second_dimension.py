# -*- coding: utf-8 -*-
"""tests/test_narrative_second_dimension.py

"Compare revenue by warehouse for the last 3 months" answered with a time
series and never said the word warehouse.

Found on a live workspace. The card led with "2026-06 closed at 1,242,800.62"
and reported "+1,437.3% from 2026-03 to 2026-06" in KEY INSIGHTS. The chart
underneath had warehouses down its axis, so the SQL had grouped correctly --
the prose was describing an axis the reader had not asked about, and building a
series out of rows that were not one.

Three things went wrong in one place, and they are the same mistake seen from
three sides:

  1. The label column was the calendar, which repeats once per warehouse.
  2. Repeating labels were still read as a series, so first and last were two
     different warehouses a quarter apart.
  3. Ranking the raw rows made the leader and the runner-up the same warehouse
     in two different months, with its share divided by a total counted once
     per period.

Every test runs the real `summarize_result_context`, which is what the answer
card is built from.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.response_builder import summarize_result_context  # noqa: E402

PERIODS = ("2026-03", "2026-04", "2026-05")
WAREHOUSES = (("Halifax Branch Store", 1_200_000.0),
              ("Calgary Distribution", 1_100_000.0),
              ("Vancouver Distribution", 900_000.0))

QUESTION = "compare revenue by warehouse for the last 3 months"


def grid(value_col: str = "REVENUE_AMT", *, flat: bool = True) -> list[dict]:
    """A row per warehouse per period — the live shape, calendar column first."""
    rows = []
    for index, period in enumerate(PERIODS):
        for name, value in WAREHOUSES:
            rows.append({"PERIOD": period, "WHS_NM": name,
                         value_col: value if flat else value * (index + 1)})
    return rows


def series(value_col: str = "REVENUE_AMT") -> list[dict]:
    return [{"PERIOD": p, value_col: v}
            for p, v in zip(PERIODS, (100.0, 150.0, 300.0))]


class TheProseNamesWhatWasAskedAbout(unittest.TestCase):

    def test_the_label_column_is_the_warehouse_not_the_calendar(self):
        self.assertEqual(summarize_result_context(grid(), QUESTION)["label_col"],
                         "WHS_NM")

    def test_the_result_is_not_described_as_a_time_series(self):
        self.assertEqual(summarize_result_context(grid(), QUESTION)["mode"],
                         "ranking")

    def test_no_percentage_change_is_computed_between_two_warehouses(self):
        # pct_change is only set on the time-series path; its presence here is
        # the +1,437.3% that started this.
        ctx = summarize_result_context(grid(), QUESTION)
        self.assertIsNone(ctx.get("pct_change"))
        self.assertIsNone(ctx.get("first_label"))

    def test_the_categories_counted_are_warehouses_not_rows(self):
        ctx = summarize_result_context(grid(), QUESTION)
        self.assertEqual(ctx["distribution_stats"]["category_count"], 3)

    def test_a_calendar_that_does_not_repeat_is_still_a_series(self):
        # The guard must not fire on the ordinary case sitting next to it.
        ctx = summarize_result_context(series(), "revenue over the last 3 months")
        self.assertEqual(ctx["mode"], "time_series")
        self.assertEqual(ctx["pct_change"], 200.0)

    def test_a_lone_repeating_calendar_is_still_not_a_series(self):
        # The second guard, reached when there is no other column to prefer:
        # a repeating calendar is the only text column, so it stays the label
        # rather than leaving the narrative with nothing to speak about — but
        # it must not be read as a series, because two rows sharing 2026-03
        # are two members of a dimension this result does not name.
        rows = [{"PERIOD": "2026-03", "REVENUE_AMT": 10.0},
                {"PERIOD": "2026-03", "REVENUE_AMT": 20.0},
                {"PERIOD": "2026-04", "REVENUE_AMT": 30.0},
                {"PERIOD": "2026-04", "REVENUE_AMT": 40.0}]
        ctx = summarize_result_context(rows, "revenue by period")
        self.assertEqual(ctx["label_col"], "PERIOD")
        self.assertEqual(ctx["mode"], "ranking")
        self.assertIsNone(ctx.get("pct_change"))


class OneRowPerLabelBeforeAnybodyLeads(unittest.TestCase):

    def leader(self, value_col="REVENUE_AMT"):
        ctx = summarize_result_context(grid(value_col), QUESTION)
        return ctx.get("comparison_stats") or {}

    def test_the_leader_and_the_runner_up_are_different_warehouses(self):
        stats = self.leader()
        self.assertEqual(stats["leader"], "Halifax Branch Store")
        self.assertEqual(stats["runner_up"], "Calgary Distribution")
        self.assertNotEqual(stats["leader"], stats["runner_up"])

    def test_the_share_is_of_the_whole_not_of_one_period(self):
        # Halifax holds 3.6M of 9.6M across the three months.
        self.assertEqual(self.leader()["leader_share_pct"], 37.5)

    def test_each_warehouse_appears_once_in_the_ranking(self):
        items = summarize_result_context(grid(), QUESTION)["top_items"]
        names = [item["label"] for item in items]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), 3)

    def test_a_percentage_measure_is_never_summed_into_a_leader(self):
        # Adding a margin percentage across three months is arithmetic on
        # nothing. No leader is better than a plausible wrong one.
        self.assertEqual(self.leader("GRS_MARG_PCT"), {})

    def test_a_balance_is_not_summed_across_time_either(self):
        # Semi-additive: a stock balance may be summed across warehouses and
        # never across months, and this collapses the months.
        self.assertEqual(self.leader("BAL_VAL_AMT"), {})

    def test_an_unrecognised_measure_is_not_guessed_at(self):
        self.assertEqual(self.leader("SOME_UNTYPED_COLUMN"), {})

    def test_a_single_dimension_ranking_is_untouched(self):
        rows = [{"WHS_NM": name, "REVENUE_AMT": value} for name, value in WAREHOUSES]
        stats = summarize_result_context(rows, "revenue by warehouse")["comparison_stats"]
        self.assertEqual(stats["leader"], "Halifax Branch Store")
        self.assertEqual(stats["runner_up"], "Calgary Distribution")
        self.assertEqual(stats["gap"], 100000.0)

    def test_a_percentage_over_one_row_per_label_still_ranks(self):
        # The refusal is about COLLAPSING rows, not about the measure: with one
        # row per warehouse there is nothing to sum and the ranking is honest.
        rows = [{"WHS_NM": name, "GRS_MARG_PCT": value}
                for name, value in (("Halifax", 31.0), ("Calgary", 22.0))]
        stats = summarize_result_context(rows, "margin by warehouse")["comparison_stats"]
        self.assertEqual(stats["leader"], "Halifax")


class TheProseUsesTheBusinessNameNotTheSpelling(unittest.TestCase):
    """"across 6 whs nm", "Break down by Sup", "Current Revenue".

    _display_label was replace("_", " ").title(), so every place the narrative
    named a column it printed the warehouse's spelling at a reader who never
    chose it. core.schema_enrichment resolves these against the tenant's active
    vocabulary -- the same expansion that names the semantic model's measures
    -- and nothing in this layer was asking it.
    """

    def test_a_warehouse_code_reads_as_words(self):
        from core.response_builder import _display_label

        self.assertEqual(_display_label("WHS_NM"), "Warehouse Name")
        self.assertEqual(_display_label("BAL_VAL_AMT"), "Balance Value Amount")
        self.assertEqual(_display_label("SUP_NM"), "Supplier Name")

    def test_a_column_already_in_words_is_unchanged(self):
        from core.response_builder import _display_label

        self.assertEqual(_display_label("CURRENT_REVENUE"), "Current Revenue")
        self.assertEqual(_display_label("REVENUE"), "Revenue")

    def test_an_infrastructure_column_is_not_given_a_business_name(self):
        # The expansion answers "data platform field: ..." for these, which is
        # a description rather than a label.
        from core.response_builder import _display_label

        self.assertEqual(_display_label("AZ_UPD_TS"), "Az Upd Ts")

    def test_nothing_in_makes_nothing_out(self):
        from core.response_builder import _display_label

        self.assertEqual(_display_label(""), "")
        self.assertEqual(_display_label(None), "")

    def test_a_broken_vocabulary_costs_a_label_not_an_answer(self):
        from unittest.mock import patch

        import core.schema_enrichment as se
        from core.response_builder import _display_label

        with patch.object(se, "enrich_columns", side_effect=RuntimeError("boom")):
            self.assertEqual(_display_label("WHS_NM"), "Whs Nm")

    def test_the_sentence_a_reader_sees_says_warehouse(self):
        # Through the real summary builder, on the live shape.
        import core.response_builder as rb

        rows = grid()
        ctx = rb.summarize_result_context(rows, QUESTION)
        brief = {"mode": ctx["mode"], "category_breakdown": {
            "top_5": ctx["top_items"],
            "leader_share_pct": ctx["comparison_stats"]["leader_share_pct"],
            "label_column": ctx["label_col"], "category_count": 3}}
        sentence = rb._build_insight_summary(rows, ctx, brief)
        self.assertIn("warehouse names", sentence)
        self.assertNotIn("whs", sentence.lower())

    def test_the_counted_label_is_plural(self):
        # It follows a count. "across 3 warehouse name" only became visibly
        # wrong once the label stopped being an abbreviation.
        import core.response_builder as rb

        rows = grid()
        ctx = rb.summarize_result_context(rows, QUESTION)
        brief = {"mode": ctx["mode"], "category_breakdown": {
            "top_5": ctx["top_items"],
            "leader_share_pct": ctx["comparison_stats"]["leader_share_pct"],
            "label_column": ctx["label_col"], "category_count": 3}}
        self.assertIn("3 warehouse names", rb._build_insight_summary(rows, ctx, brief))


if __name__ == "__main__":
    unittest.main()
