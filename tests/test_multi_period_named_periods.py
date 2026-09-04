"""
tests/test_multi_period_named_periods.py

`detect_multi_period_intent` was computed on every question and consumed by
nothing. Fourteen of the fifteen keys in `detect_analytical_intents` have an
`if _intents.get(...)` branch in the pipeline that appends a SQL hint;
`multi_period` had none, and the string does not appear in
core/query_pipeline.py at all. So a question naming two years produced a
one-year query and an answer explaining a change it never fetched.

Before wiring it, the detector had to be made trustworthy, because what it
returned was not. Executed against the unguarded version:

    list SKUs 2001 2002 2003                      -> 3 "periods"
    show items priced 2020, 2030 and 2040         -> 3 "periods"
    compare warehouse 2024 stock to warehouse 2025 -> 2 "periods"

Those are part numbers. A bare four-digit integer is the weakest period signal
in the language and on an ERP schema it is usually not a period at all.

The second defect is subtler. "last N years" derives its labels from
date.today(), and the obvious guard -- "is this label actually in the
question?" -- fails open on

    compare revenue for the last 2 years for warehouse 2025 and warehouse 2024

where the clock-derived labels DO appear, as warehouse IDs. Provenance has to
be recorded where the labels are made, not reconstructed afterwards.

Every test here calls the real function and asserts on what it returns.
"""

import unittest

from core.multi_period import (
    MultiPeriodIntent,
    detect_multi_period_intent,
    extract_period_specs,
)

TARGET = ("Compare 2025 against 2024 by revenue category which grew, "
          "which shrank, and what did each contribute to the overall change?")


def _labels(intent: MultiPeriodIntent | None) -> list[str] | None:
    return None if intent is None else [s.label for s in intent.period_specs]


class TheDetectorFindsRealPeriods(unittest.TestCase):

    def test_the_target_question(self):
        intent = detect_multi_period_intent(TARGET)
        self.assertIsNotNone(intent)
        self.assertEqual(_labels(intent), ["2025", "2024"])
        self.assertEqual(intent.compare_count, 2)
        self.assertEqual(intent.grain, "yearly")
        self.assertEqual(intent.source, "named")

    def test_three_quarters(self):
        intent = detect_multi_period_intent(
            "Compare Q1 2024, Q1 2023, and Q1 2022 revenue by region")
        self.assertEqual(_labels(intent), ["Q1 2024", "Q1 2023", "Q1 2022"])
        self.assertEqual(intent.grain, "quarterly")
        self.assertEqual(intent.source, "named")

    def test_a_comma_separated_year_list(self):
        intent = detect_multi_period_intent(
            "what did each region contribute to revenue in 2023, 2024 and 2025")
        self.assertEqual(_labels(intent), ["2023", "2024", "2025"])

    def test_a_cue_word_carries_a_year_that_opens_no_list(self):
        self.assertEqual(
            [s.label for s in extract_period_specs("revenue in 2024 by region")],
            ["2024"],
        )


class AnIntegerIsNotAPeriod(unittest.TestCase):
    """Each of these returned periods before the guard existed."""

    def test_part_numbers_in_a_list(self):
        self.assertIsNone(detect_multi_period_intent("list SKUs 2001 2002 2003"))

    def test_prices(self):
        self.assertIsNone(
            detect_multi_period_intent("show items priced 2020, 2030 and 2040"))

    def test_warehouse_ids_behind_a_compare(self):
        """The comparison word is real; the numbers are not periods."""
        self.assertIsNone(detect_multi_period_intent(
            "compare warehouse 2024 stock to warehouse 2025 stock"))

    def test_a_year_out_of_the_plausible_range_is_not_a_period(self):
        self.assertEqual(
            [s.label for s in extract_period_specs(
                "compare revenue between 2020 and 2024 for warehouse 2050")],
            ["2020", "2024"],
        )

    def test_a_separating_comma_only_counts_after_a_period_is_named(self):
        """"priced 2020, 2030" is a list of prices; "in 2023, 2024" is a list
        of periods. The comma alone cannot tell them apart, so it only
        separates once something before it was accepted."""
        self.assertEqual(extract_period_specs("items priced 2020, 2030"), [])
        self.assertEqual(
            [s.label for s in extract_period_specs("revenue in 2020, 2030")],
            ["2020", "2030"],
        )


class ClockDerivedLabelsAreMarkedAtSource(unittest.TestCase):

    def test_last_n_years_is_flagged_relative(self):
        intent = detect_multi_period_intent("compare revenue for the last 2 years")
        self.assertIsNotNone(intent)
        self.assertEqual(intent.source, "relative")

    def test_the_substring_guard_would_have_failed_open(self):
        """The labels the clock produced are present in this question as
        warehouse IDs, so 'is the label in the question?' says yes and is
        wrong. Provenance is the only thing that answers correctly."""
        question = ("compare revenue for the last 2 years "
                    "for warehouse 2025 and warehouse 2024")
        intent = detect_multi_period_intent(question)
        self.assertIsNotNone(intent)
        self.assertEqual(intent.source, "relative")
        # The trap: every clock-derived label is present in the question text,
        # so a containment check would have called these user-named.
        for label in _labels(intent):
            self.assertIn(label, question)

    def test_named_periods_are_flagged_named(self):
        self.assertEqual(detect_multi_period_intent(TARGET).source, "named")


class TheNExecutionHalfIsGone(unittest.TestCase):
    """Five symbols in multi_period, two in contribution_analysis and one in
    insight had zero production callers. Leaving a half-built parallel design
    in the same file as the fix is how the original defect happened.

    An executed namespace assertion, not a source scan: these names must not
    resolve at all.
    """

    REMOVED = {
        "core.multi_period": [
            "PeriodResult", "MultiPeriodResult", "merge_multi_period_results",
            "build_multi_period_chart_payload", "build_multi_period_rewrite_prompt",
        ],
        "core.contribution_analysis": ["ContribSummary", "build_contribution_summary"],
        "core.insight": ["detect_comparison_intent"],
    }

    def test_the_dead_symbols_do_not_resolve(self):
        import importlib
        for module_name, names in self.REMOVED.items():
            module = importlib.import_module(module_name)
            for name in names:
                with self.subTest(symbol=f"{module_name}.{name}"):
                    self.assertFalse(
                        hasattr(module, name),
                        f"{name} was deleted but still resolves",
                    )

    def test_what_survived_is_still_wired(self):
        """build_contribution_sql_hint IS consumed by the pipeline and must
        not have been swept up with its dead neighbours."""
        from core.contribution_analysis import (
            build_contribution_sql_hint, compute_contribution,
            detect_contribution_intent,
        )
        self.assertTrue(callable(build_contribution_sql_hint))
        self.assertTrue(callable(compute_contribution))
        self.assertTrue(detect_contribution_intent(TARGET))


if __name__ == "__main__":
    unittest.main()
