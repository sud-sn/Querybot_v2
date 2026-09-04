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
from datetime import date

from core.multi_period import (
    MultiPeriodIntent,
    build_period_plan,
    detect_multi_period_intent,
    extract_period_specs,
    period_alias_suffix,
    period_bounds,
    period_parts,
    period_sort_key,
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


class ThePeriodVocabulary(unittest.TestCase):
    """The detector returns question order -- the target question asks for 2025
    before 2024 -- so anything that computes a change must sort first."""

    def test_labels_parse_into_calendar_parts(self):
        self.assertEqual(period_parts("2024"), (2024, 0, 0))
        self.assertEqual(period_parts("Q1 2024"), (2024, 1, 1))
        self.assertEqual(period_parts("Jan 2024"), (2024, 2, 1))
        self.assertEqual(period_parts("January 2024"), (2024, 2, 1))
        self.assertIsNone(period_parts("banana"))
        self.assertIsNone(period_parts(""))

    def test_sorting_is_chronological_across_grains(self):
        self.assertEqual(
            sorted(["2025", "2024", "2023"], key=period_sort_key),
            ["2023", "2024", "2025"],
        )
        self.assertEqual(
            sorted(["Q1 2024", "Q4 2023", "Q2 2024"], key=period_sort_key),
            ["Q4 2023", "Q1 2024", "Q2 2024"],
        )

    def test_bounds_are_half_open(self):
        """A closed upper bound silently drops the last day's rows on a
        datetime column, and this is the fallback path with the least other
        checking around it."""
        self.assertEqual(period_bounds("2024"), (date(2024, 1, 1), date(2025, 1, 1)))
        self.assertEqual(period_bounds("Q4 2024"), (date(2024, 10, 1), date(2025, 1, 1)))

    def test_december_rolls_into_the_next_year(self):
        self.assertEqual(period_bounds("Dec 2024"), (date(2024, 12, 1), date(2025, 1, 1)))

    def test_aliases_are_column_name_safe(self):
        self.assertEqual(period_alias_suffix("Q1 2024"), "Q1_2024")
        self.assertEqual(period_alias_suffix("Jan 2024"), "JAN_2024")
        self.assertEqual(period_alias_suffix("2024"), "2024")


def _semantic_plan(calendar_attributes, *, key_type="surrogate_fk",
                   role_alias="invoice_date"):
    """A governed date policy shaped like the one contextual_dates emits.

    Obviously synthetic table names -- no tenant schema is referenced anywhere
    in this file.
    """
    return {"date_key_policies": [{
        "table": "DBO.F_SALES", "column": "DATE_KEY",
        "date_value_table": "DBO.D_DATE", "date_value_column": "FULL_DATE",
        "date_key_type": key_type, "role_alias": role_alias,
        "calendar_attributes": dict(calendar_attributes),
    }]}


ATTRS = {"year": "CALENDAR_YEAR", "quarter": "CAL_QUARTER",
         "month_number": "MONTH_NUM"}


class ThePlanCompilesGovernedPredicates(unittest.TestCase):

    def test_yearly_uses_the_calendar_dimension_attribute(self):
        plan = build_period_plan(
            detect_multi_period_intent(TARGET), TARGET, _semantic_plan(ATTRS), "azure_sql")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.labels, ["2024", "2025"])     # sorted, not asked-order
        self.assertEqual(plan.aliases, ["2024", "2025"])
        self.assertEqual(plan.predicates, [
            "invoice_date.[CALENDAR_YEAR] = 2024",
            "invoice_date.[CALENDAR_YEAR] = 2025",
        ])

    def test_the_dialect_changes_the_quoting(self):
        plan = build_period_plan(
            detect_multi_period_intent(TARGET), TARGET, _semantic_plan(ATTRS), "snowflake")
        self.assertEqual(plan.predicates[0], 'invoice_date."CALENDAR_YEAR" = 2024')

    def test_quarterly_constrains_both_attributes(self):
        question = "Compare Q1 2024, Q1 2023, and Q1 2022 revenue by region"
        plan = build_period_plan(
            detect_multi_period_intent(question), question, _semantic_plan(ATTRS), "azure_sql")
        self.assertEqual(plan.labels, ["Q1 2022", "Q1 2023", "Q1 2024"])
        self.assertEqual(
            plan.predicates[0],
            "invoice_date.[CALENDAR_YEAR] = 2022 AND invoice_date.[CAL_QUARTER] = 1")

    def test_monthly_falls_back_to_a_year_month_integer(self):
        question = "compare revenue in Jan 2024 against Feb 2024 by product"
        plan = build_period_plan(
            detect_multi_period_intent(question), question,
            _semantic_plan({"year_month": "YEAR_MONTH"}), "azure_sql")
        self.assertEqual(plan.predicates,
                         ["invoice_date.[YEAR_MONTH] = 202401",
                          "invoice_date.[YEAR_MONTH] = 202402"])

    def test_without_attributes_it_uses_a_half_open_range_on_the_alias(self):
        """Alias-qualified, matching how core/pipeline_helpers builds a date
        reference. Naming the physical table produces SQL that will not
        resolve, because the dimension is joined under the role alias."""
        plan = build_period_plan(
            detect_multi_period_intent(TARGET), TARGET, _semantic_plan({}), "azure_sql")
        self.assertEqual(
            plan.predicates[0],
            "invoice_date.[FULL_DATE] >= '2024-01-01' "
            "AND invoice_date.[FULL_DATE] < '2025-01-01'")

    def test_an_integer_date_key_keeps_its_conversion(self):
        plan = build_period_plan(
            detect_multi_period_intent(TARGET), TARGET,
            _semantic_plan({}, key_type="yyyymmdd_integer"), "azure_sql")
        self.assertIn("TRY_CONVERT(date, CONVERT(varchar(8), invoice_date.[FULL_DATE]), 112)",
                      plan.predicates[0])

    def test_no_predicate_ever_wraps_the_date_in_year_or_datepart(self):
        """Those wrappers are what the surrogate_date_conversion rules in
        core/validator.py refuse, so emitting them here would produce SQL the
        product then rejects."""
        for attrs in ({}, ATTRS, {"year_month": "YEAR_MONTH"}):
            for db in ("azure_sql", "snowflake", "oracle"):
                plan = build_period_plan(
                    detect_multi_period_intent(TARGET), TARGET,
                    _semantic_plan(attrs), db)
                if plan is None:
                    continue
                for predicate in plan.predicates:
                    with self.subTest(attrs=bool(attrs), db=db):
                        for banned in ("YEAR(", "DATEPART", "FORMAT(", "CONVERT(varchar(4)"):
                            self.assertNotIn(banned, predicate)


class ThePlanRefusesRatherThanGuesses(unittest.TestCase):
    """Returning None means "behave exactly as before". There is no partial
    mode: a plan naming some periods and guessing others produces arithmetic
    across incomparable columns."""

    def _refuses(self, question, plan=None):
        return build_period_plan(
            detect_multi_period_intent(question), question,
            _semantic_plan(ATTRS) if plan is None else plan, "azure_sql")

    def test_no_governed_date_field_means_no_plan(self):
        self.assertIsNone(self._refuses(TARGET, {"date_key_policies": []}))

    def test_clock_derived_periods_are_refused(self):
        self.assertIsNone(self._refuses("compare revenue for the last 2 years"))

    def test_a_question_with_no_comparison_word_is_refused(self):
        """"revenue in 2023, 2024 and 2025" names periods but asks for no
        comparison, so it keeps today's behaviour exactly."""
        self.assertIsNone(self._refuses("revenue in 2023, 2024 and 2025 by product"))

    def test_part_numbers_are_refused(self):
        self.assertIsNone(
            self._refuses("compare warehouse 2024 stock to warehouse 2025 stock"))

    def test_mixed_grains_are_refused(self):
        self.assertIsNone(self._refuses("compare Q1 2024 against 2023 by region"))

    def test_more_periods_than_the_chart_can_draw_are_refused(self):
        """Past the palette length portal_chat.html silently drops series, so
        rendering fewer series than were asked for would repeat the exact
        silent incompleteness this change removes."""
        question = ("compare revenue in 2018, 2019, 2020, 2021, 2022, 2023 "
                    "and 2024 by product")
        self.assertIsNone(self._refuses(question))


class TheDateGateOpensForNamedComparisons(unittest.TestCase):
    """The root cause. question_has_temporal_intent returns False for the
    target question, so resolve_contextual_date_binding returned
    {"status": "none"}, no date role bound, and the field plan carried no date
    field -- leaving the model to guess a date column with no governed join."""

    ROLE = {
        "name": "Invoice date", "business_role": "invoice_date",
        "fact_table": "DBO.F_SALES", "fact_column": "DATE_KEY",
        "dimension_table": "DBO.D_DATE", "dimension_key": "DATE_KEY",
        "date_value_column": "FULL_DATE", "date_key_type": "surrogate_fk",
        "is_default": True, "status": "approved", "confidence": 90,
    }

    def _resolve(self, question):
        from core.contextual_dates import resolve_contextual_date_binding
        return resolve_contextual_date_binding(
            question, matched_metrics=[], bindings=[], date_roles=[self.ROLE])

    def test_the_pre_state_is_still_true(self):
        """Guards the premise: if question_has_temporal_intent ever learns to
        see this question, the gate widening becomes redundant and this test
        says so instead of silently passing."""
        from core.date_roles import question_has_temporal_intent
        self.assertFalse(question_has_temporal_intent(TARGET))

    def test_the_target_question_now_binds_a_date_role(self):
        resolved = self._resolve(TARGET)
        self.assertEqual(resolved.get("status"), "selected")
        self.assertEqual(resolved["binding"].get("fact_table"), "DBO.F_SALES")
        self.assertEqual(resolved["binding"].get("fact_column"), "DATE_KEY")

    def test_the_widening_is_narrow(self):
        for question in ("list SKUs 2001 2002 2003",
                         "total revenue by warehouse",
                         "revenue in 2024 by region"):
            with self.subTest(question=question):
                self.assertEqual(self._resolve(question).get("status"), "none")

    def test_ordinary_temporal_questions_are_untouched(self):
        self.assertEqual(self._resolve("revenue last month by region").get("status"),
                         "selected")


class TheCalendarAttributeFormatterWasExtractedIntact(unittest.TestCase):
    """format_calendar_attribute_ref was lifted out of an inner closure so the
    period compiler quotes attributes from the same source of truth as the
    period bucket. The bucket's behaviour must not have moved."""

    ATTRS = {"year": "CALENDAR_YEAR", "month_number": "MONTH_NUM",
             "quarter": "CAL_QUARTER", "year_month": "YEAR_MONTH"}

    def test_the_period_bucket_still_renders_the_same(self):
        from core.contextual_dates import format_period_bucket_expression as bucket
        cases = [
            (("inv.FULL_DATE", "year", "azure_sql"), "invoice_date", self.ATTRS,
             "invoice_date.[CALENDAR_YEAR]"),
            (("inv.FULL_DATE", "year", "azure_sql"), "invoice_date", {},
             "DATEFROMPARTS(YEAR(inv.FULL_DATE), 1, 1)"),
            (("inv.FULL_DATE", "year", "snowflake"), "invoice_date", self.ATTRS,
             'invoice_date."CALENDAR_YEAR"'),
            (("inv.FULL_DATE", "month", "azure_sql"), "invoice_date", self.ATTRS,
             "invoice_date.[YEAR_MONTH]"),
        ]
        for (ref, grain, db), alias, attrs, expected in cases:
            with self.subTest(grain=grain, db=db, attrs=bool(attrs)):
                self.assertEqual(
                    bucket(ref, grain, db, role_alias=alias, calendar_attributes=attrs),
                    expected)

    def test_a_missing_attribute_is_empty_not_a_guess(self):
        from core.contextual_dates import format_calendar_attribute_ref as ref
        self.assertEqual(ref("invoice_date", self.ATTRS, "week_number", "azure_sql"), "")
        self.assertEqual(ref("invoice_date", None, "year", "azure_sql"), "")


if __name__ == "__main__":
    unittest.main()
