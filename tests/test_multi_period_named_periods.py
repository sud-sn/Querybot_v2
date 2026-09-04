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
    PeriodPlan,
    annotate_period_change,
    build_multi_period_sql_hint,
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


class TheHintIsGroundedAndPromptSafe(unittest.TestCase):
    """The hint is the only thing that makes the model pivot, and it sits after
    _KB_SECTION_MARKER -- the last boundary _filter_sql_rules_for_compiled_plan
    knows about. A rule marker in this text makes that filter delete from the
    match to the end of the prompt, taking the field plan and the schema with
    it, so the text is constrained as tightly as the SQL it prescribes."""

    def _plan(self, question=TARGET, attrs=None, db="azure_sql"):
        return build_period_plan(
            detect_multi_period_intent(question), question,
            _semantic_plan(ATTRS if attrs is None else attrs), db)

    def _hint(self, **kw):
        return build_multi_period_sql_hint(self._plan(**kw), kw.get("db", "azure_sql"))

    def test_hint_text_is_grounded_and_prompt_safe(self):
        from core.llm import _OPTIONAL_SQL_RULE_MARKERS

        hint = self._hint()
        plan = self._plan()

        # Grounded: the governed predicates appear verbatim, so the model is
        # never left to invent a period expression.
        for predicate in plan.predicates:
            self.assertIn(predicate, hint)
        self.assertIn("2024", hint)
        self.assertIn("2025", hint)

        # A star anywhere -- the CTE included -- is refused by
        # _production_shape_errors, which walks every Select scope.
        self.assertNotIn("SELECT *", hint)
        self.assertNotIn("select *", hint.lower())

        for line in hint.split("\n"):
            self.assertFalse(line.startswith("- "), line)
        for feature, marker in _OPTIONAL_SQL_RULE_MARKERS.items():
            with self.subTest(rule=feature):
                self.assertNotIn(marker, hint)

    def test_no_plan_means_no_prose(self):
        """Forbidding YEAR()/DATEPART() while naming no sanctioned alternative
        is an unfollowable instruction, so a withheld plan emits nothing."""
        self.assertEqual(build_multi_period_sql_hint(None, "azure_sql"), "")
        self.assertEqual(
            build_multi_period_sql_hint(
                self._plan("compare warehouse 2024 stock to warehouse 2025 stock"),
                "azure_sql"),
            "")

    def test_the_hint_carries_the_dialect_of_the_plan(self):
        self.assertIn('invoice_date."CALENDAR_YEAR" = 2024', self._hint(db="snowflake"))

    def test_period_columns_keep_the_measure_name(self):
        """A generic P_2024 with whole-number values is classified 'identifier'
        by core/chart_spec.py and the chart loses the series entirely."""
        hint = self._hint()
        self.assertIn("<MEASURE>_2024", hint)
        self.assertIn("<MEASURE>_2025", hint)
        self.assertNotIn("P_2024", hint)

    def test_hint_survives_the_rule_filter(self):
        """Executed both ways: the real hint leaves the prompt tail intact, and
        the same prompt with one marker added to the hint loses it. Without the
        second half this test passes on any string at all."""
        import core.llm as llm

        hint = self._hint()
        semantic_plan = _compiled_request_plan()
        prompt = llm.build_sql_system_prompt(
            "azure_sql",
            "DBO.F_SALES: sales facts.\n\n---\n\n" + hint,
            semantic_plan=semantic_plan,
            question=TARGET,
        )
        self.assertIn(hint, prompt)
        self.assertIn("FIELD PLAN RULE:", prompt)

        filtered = llm._filter_sql_rules_for_compiled_plan(prompt, semantic_plan, None)
        self.assertIn(hint, filtered)
        self.assertIn("FIELD PLAN RULE:", filtered)

        poisoned_prompt = llm.build_sql_system_prompt(
            "azure_sql",
            "DBO.F_SALES: sales facts.\n\n---\n\n" + hint
            + "\n- RANKING RULE: a marker inside a hint.\n",
            semantic_plan=semantic_plan,
            question=TARGET,
        )
        poisoned = llm._filter_sql_rules_for_compiled_plan(
            poisoned_prompt, semantic_plan, None)
        self.assertNotIn(
            "FIELD PLAN RULE:", poisoned,
            "the control did not fail, so the real assertion above proves nothing",
        )


# Obviously synthetic schema -- no tenant tables anywhere in this file.
_HINT_TABLE_COLUMNS = {
    "DBO.F_SALES": {"NET_AMOUNT": "decimal", "CAT_KEY": "int", "DATE_KEY": "int"},
    "DBO.D_CATEGORY": {"CAT_KEY": "int", "REVENUE_CATEGORY": "varchar"},
    "DBO.D_DATE": {"DATE_KEY": "int", "FULL_DATE": "date", "CALENDAR_YEAR": "int",
                   "CAL_QUARTER": "int", "MONTH_NUM": "int"},
}

_HINT_FROM = (
    "DBO.F_SALES f "
    "JOIN DBO.D_CATEGORY c ON c.CAT_KEY = f.CAT_KEY "
    "JOIN DBO.D_DATE invoice_date ON invoice_date.DATE_KEY = f.DATE_KEY"
)


def _compiled_request_plan():
    """A semantic plan trustworthy enough for _compiled_sql_rule_features to
    return a set. With None it keeps the legacy full prompt and the rule filter
    is a no-op, which would make the filter test vacuous."""
    policy = _semantic_plan(ATTRS)["date_key_policies"][0]
    return {
        "enabled": True,
        "date_key_policies": [policy],
        "fields": [{"term": "revenue", "table": "DBO.F_SALES", "column": "NET_AMOUNT",
                    "role": "measure", "enforcement": "required"}],
        "analytical_request_plan": {
            "status": "compiled", "question": TARGET, "intent": "comparison",
            "dimensions": [{"name": "REVENUE_CATEGORY"}],
            "measures": [{"name": "NET_AMOUNT"}],
            "source_facts": ["DBO.F_SALES"],
        },
    }


def _sql_the_hint_prescribes(hint: str) -> str:
    """The hint's own SQL sketch with its placeholders filled in.

    Lifted out of the returned string rather than retyped, so the statement
    validated below is the one the model is actually being shown.
    """
    start = hint.index("WITH period_totals")
    end = hint.index("\n\n", start)
    return (
        hint[start:end]
        .replace("<the fact table and its approved joins>", _HINT_FROM)
        .replace("<MEASURE>", "NET_AMOUNT")
        .replace("<measure>", "f.NET_AMOUNT")
        .replace("<category>", "REVENUE_CATEGORY")
    )


class ThePrescribedShapeIsAcceptedAsGenerated(unittest.TestCase):
    """The hint is only worth sending if the SQL it describes survives the
    validators the product then applies to it. Composition mode is on, because
    the analysis contract is hint index 0 and its 'one row per requested
    business category' is what periods-as-columns satisfies and
    periods-as-rows fights."""

    def _sql(self):
        plan = build_period_plan(
            detect_multi_period_intent(TARGET), TARGET, _semantic_plan(ATTRS), "azure_sql")
        return _sql_the_hint_prescribes(build_multi_period_sql_hint(plan, "azure_sql"))

    def test_prescribed_shape_validates(self):
        from core.validator import validate_sql_detailed

        result = validate_sql_detailed(
            self._sql(),
            set(_HINT_TABLE_COLUMNS),
            "azure_sql",
            set(_HINT_TABLE_COLUMNS),
            _HINT_TABLE_COLUMNS,
            {"production_sql": True,
             "analysis_contract": {"enabled": True, "mode": "composition"}},
        )
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.code, "ok")

    def test_the_lineage_is_aggregate_only_and_starless(self):
        """Pins the compliance read of the same statement: no star in any
        scope, and every emitted measure traces to an aggregate rather than to
        a raw row value."""
        from core.compliance.sql_guard import analyze_sql

        analysis = analyze_sql(self._sql(), "azure_sql")
        self.assertFalse(analysis.has_star)
        for column in ("NET_AMOUNT_2024", "NET_AMOUNT_2025",
                       "CHANGE_ABS", "CHANGE_PCT", "SHARE_OF_CHANGE_PCT"):
            with self.subTest(column=column):
                self.assertIn(column, analysis.aggregate_outputs)


def _wide(pairs, label_key="REVENUE_CATEGORY",
          columns=("NET_AMOUNT_2024", "NET_AMOUNT_2025")):
    """Rows in the shape the hint asks the model for."""
    return [dict(zip((label_key,) + tuple(columns), row)) for row in pairs]


_WIDE_PLAN = PeriodPlan(
    labels=["2024", "2025"], aliases=["2024", "2025"],
    predicates=["invoice_date.[CALENDAR_YEAR] = 2024",
                "invoice_date.[CALENDAR_YEAR] = 2025"],
    grain="yearly", date_field={},
)


def _fresh_plan(**overrides):
    """A plan per test: `warnings` is the mutable channel annotate writes to,
    so a shared instance would leak state between cases."""
    fields = {"labels": ["2024", "2025"], "aliases": ["2024", "2025"],
              "predicates": ["a", "b"], "grain": "yearly", "date_field": {}}
    fields.update(overrides)
    return PeriodPlan(**fields)


class TheChangeIsRecomputedFromTheReturnedColumns(unittest.TestCase):
    """The half of the fix that does not depend on the model.

    It also disarms a proven wrong-answer path: infer_numeric_col's metric-word
    tie-break picks the EARLIER column, so on a NET_AMOUNT_2024 /
    NET_AMOUNT_2025 shape compute_contribution would have shipped a
    share-of-2024 number under the label contribution_pct.
    """

    def test_compliant_wide_rows_get_the_change_and_its_share(self):
        rows = _wide([("Pumps", 1000, 1600),
                      ("Valves", 900, 700),
                      ("Seals", 500, 600)])
        plan = _fresh_plan()
        out = annotate_period_change(rows, plan)

        self.assertEqual([r["CHANGE_ABS"] for r in out], [600.0, -200.0, 100.0])
        for row in out:
            self.assertAlmostEqual(
                row["CHANGE_ABS"],
                row["NET_AMOUNT_2025"] - row["NET_AMOUNT_2024"], places=6)
        self.assertAlmostEqual(
            sum(r["SHARE_OF_CHANGE_PCT"] for r in out), 100.0, places=1)
        self.assertEqual(plan.warnings, [])

    def test_the_input_rows_are_never_mutated(self):
        rows = _wide([("Pumps", 1000, 1600)])
        annotate_period_change(rows, _fresh_plan())
        self.assertEqual(rows, [{"REVENUE_CATEGORY": "Pumps",
                                 "NET_AMOUNT_2024": 1000,
                                 "NET_AMOUNT_2025": 1600}])

    def test_a_single_period_result_returns_none(self):
        """The signal that the hint did not land. The caller turns it into a
        coverage caveat instead of a confident single-period answer."""
        rows = [{"REVENUE_CATEGORY": "Pumps", "NET_AMOUNT": 1600},
                {"REVENUE_CATEGORY": "Valves", "NET_AMOUNT": 700}]
        self.assertIsNone(annotate_period_change(rows, _fresh_plan()))

    def test_only_one_of_the_two_periods_came_back(self):
        rows = _wide([("Pumps", 1000, 1600)], columns=("NET_AMOUNT_2025", "OTHER"))
        self.assertIsNone(annotate_period_change(rows, _fresh_plan()))

    def test_near_cancelling_movements_suppress_the_share(self):
        """Gains and losses that cancel make a share of the NET meaningless:
        five categories moving a million each and netting twenty thousand would
        report shares in the thousands of percent. This is the
        requires_positive_total guard core/analysis_contract.py declares and
        nothing read until now."""
        rows = _wide([("Pumps", 1000, 2000),
                      ("Valves", 2000, 1000),
                      ("Seals", 500, 520)])
        plan = _fresh_plan()
        out = annotate_period_change(rows, plan)

        self.assertTrue(all(r["SHARE_OF_CHANGE_PCT"] is None for r in out))
        self.assertTrue(plan.warnings)
        # The per-category changes themselves are still real and still shown.
        self.assertEqual([r["CHANGE_ABS"] for r in out], [1000.0, -1000.0, 20.0])

    def test_a_masked_cell_is_carried_through_rather_than_zeroed(self):
        """protect_rows replaces a masked value with a string. Coercing it to
        zero would report a masked category as flat and would drag every other
        category's share off."""
        rows = _wide([("Pumps", 1000, 1600),
                      ("Redacted", "[REDACTED]", "[REDACTED]"),
                      ("Valves", 900, 700),
                      ("Seals", 500, 600)])
        plan = _fresh_plan()
        out = annotate_period_change(rows, plan)

        masked = out[1]
        self.assertIsNone(masked["CHANGE_ABS"])
        self.assertIsNone(masked["CHANGE_PCT"])
        self.assertIsNone(masked["SHARE_OF_CHANGE_PCT"])
        self.assertAlmostEqual(
            sum(r["SHARE_OF_CHANGE_PCT"] for r in out
                if r["SHARE_OF_CHANGE_PCT"] is not None),
            100.0, places=1)

    def test_a_truncated_result_derives_nothing(self):
        """core/compliance/governed_query.py tells consumers that aggregate
        across rows to refuse on a truncated prefix. A share over a 200-row
        head is exactly that."""
        rows = _wide([("Pumps", 1000, 1600), ("Valves", 900, 700)])
        plan = _fresh_plan()
        out = annotate_period_change(rows, plan, truncated=True)

        self.assertEqual(len(out), 2)
        for row in out:
            self.assertNotIn("CHANGE_ABS", row)
            self.assertNotIn("CHANGE_PCT", row)
            self.assertNotIn("SHARE_OF_CHANGE_PCT", row)
        self.assertTrue(plan.warnings)

    def test_erp_part_columns_are_not_hijacked(self):
        """Exact alias match, not a regex over four-digit suffixes. P_CODE and
        P_QTY are the columns this guard exists for."""
        rows = [{"P_CODE": "A-1", "P_QTY": 12, "QTY": 3},
                {"P_CODE": "A-2", "P_QTY": 40, "QTY": 9}]
        self.assertIsNone(annotate_period_change(
            rows, _fresh_plan(aliases=["NET_AMOUNT_2024", "NET_AMOUNT_2025"])))
        self.assertIsNone(annotate_period_change(rows, _fresh_plan()))

    def test_two_different_measures_are_not_treated_as_one(self):
        """NET_AMOUNT_2024 beside ORDER_COUNT_2025 is two measures, not one
        measure in two periods, and subtracting them is nonsense."""
        rows = [{"REVENUE_CATEGORY": "Pumps",
                 "NET_AMOUNT_2024": 1000, "ORDER_COUNT_2025": 4}]
        self.assertIsNone(annotate_period_change(rows, _fresh_plan()))

    def test_a_zero_baseline_leaves_the_percent_empty_not_infinite(self):
        rows = _wide([("Pumps", 0, 1600), ("Valves", 900, 700)])
        out = annotate_period_change(rows, _fresh_plan())
        self.assertIsNone(out[0]["CHANGE_PCT"])
        self.assertEqual(out[0]["CHANGE_ABS"], 1600.0)

    def test_quarterly_aliases_match_their_own_columns(self):
        rows = [{"REGION": "North", "REVENUE_Q1_2023": 100, "REVENUE_Q1_2024": 150}]
        plan = _fresh_plan(labels=["Q1 2023", "Q1 2024"],
                           aliases=["Q1_2023", "Q1_2024"], grain="quarterly")
        out = annotate_period_change(rows, plan)
        self.assertEqual(out[0]["CHANGE_ABS"], 50.0)


class AMissedPivotStillAnswersTheContributionQuestion(unittest.TestCase):
    """Design 1 gated the hint on the plan and the post-processor on the raw
    intent, and shipped a confident single-period answer when the model ignored
    the hint. Both halves are fixed here, and both are executed: the real
    post-processing block is compiled out of core/query_pipeline.py and run.
    """

    START = "            # First, because the contribution branch below is gated"
    END = '            if _post_intents.get("anomaly")'

    def _block(self):
        import textwrap
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1]
                  / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        block = textwrap.dedent(source[source.index(self.START):source.index(self.END)])
        for marker in ("annotate_period_change", "_mp_rows_applied",
                       "compute_contribution", "_mp_caveats"):
            assert marker in block, f"stale or truncated read: {marker!r} missing"
        return block

    def _run(self, rows, plan, truncated=False):
        import core.query_pipeline as qp

        logged: dict[str, list[str]] = {}

        class _Log:
            def info(self, msg, *a):
                logged.setdefault("info", []).append(msg % a if a else msg)

            def warning(self, msg, *a, **kw):
                logged.setdefault("warning", []).append(msg % a if a else msg)

        class _Event:
            pass

        event = _Event()
        event._multi_period_plan = plan
        env = {
            **vars(qp),
            "rows": rows,
            "event": event,
            "_post_intents": {"contribution": True, "multi_period": object()},
            "_rows_truncated": truncated,
            "_mp_caveats": [],
            "_mp_rows_applied": False,
            "log": _Log(),
        }
        exec(compile(self._block(), "<multi-period-block>", "exec"), env)
        return env, logged

    def test_contribution_survives_a_multi_period_miss(self):
        """The model returned one period. contribution_pct must NOT be lost --
        the question literally asks what each category contributed -- and the
        reader must be told the answer covers less than was asked for."""
        rows = [{"REVENUE_CATEGORY": "Pumps", "NET_AMOUNT": 600},
                {"REVENUE_CATEGORY": "Valves", "NET_AMOUNT": 400}]
        env, logged = self._run(rows, _fresh_plan())

        self.assertFalse(env["_mp_rows_applied"])
        self.assertEqual([r["contribution_pct"] for r in env["rows"]], [60.0, 40.0])
        self.assertTrue(env["_mp_caveats"])
        self.assertIn("2024", env["_mp_caveats"][0])
        self.assertIn("2025", env["_mp_caveats"][0])
        self.assertTrue(logged.get("warning"))

    def test_a_landed_pivot_suppresses_the_contribution_column(self):
        """compute_contribution's SUM(...) OVER () has no PARTITION BY, so on a
        widened result it would be each category's share of the COMBINED 2024
        and 2025 total -- neither a per-period mix nor a share of the change."""
        rows = _wide([("Pumps", 1000, 1600), ("Valves", 900, 700)])
        env, logged = self._run(rows, _fresh_plan())

        self.assertTrue(env["_mp_rows_applied"])
        self.assertNotIn("contribution_pct", env["rows"][0])
        self.assertEqual(env["rows"][0]["CHANGE_ABS"], 600.0)
        self.assertEqual(env["_mp_caveats"], [])
        self.assertTrue(any("change contribution added" in m
                            for m in logged.get("info", [])))

    def test_an_ordinary_contribution_question_is_untouched(self):
        """No plan stashed: "what did each region contribute to revenue in
        2023, 2024 and 2025" has a truthy multi_period intent, no comparison
        word, and must keep behaving exactly as it did."""
        rows = [{"REGION": "North", "REVENUE": 750},
                {"REGION": "South", "REVENUE": 250}]
        env, _ = self._run(rows, None)

        self.assertFalse(env["_mp_rows_applied"])
        self.assertEqual([r["contribution_pct"] for r in env["rows"]], [75.0, 25.0])
        self.assertEqual(env["_mp_caveats"], [])

    def test_a_truncated_pivot_says_so_and_keeps_the_periods(self):
        rows = _wide([("Pumps", 1000, 1600), ("Valves", 900, 700)])
        env, _ = self._run(rows, _fresh_plan(), truncated=True)

        self.assertTrue(env["_mp_rows_applied"])
        self.assertNotIn("CHANGE_ABS", env["rows"][0])
        self.assertNotIn("contribution_pct", env["rows"][0])
        self.assertTrue(env["_mp_caveats"])


class TheCaveatsReachTheRenderedCard(unittest.TestCase):
    """A guard that fires has to become a sentence the reader sees. The
    renderer collects the pipeline's multi_period_caveats on the same channel
    as the forecast caveats, and the rendered payload is what this asserts on
    -- _send_results is executed, not read."""

    ROWS = [{"REVENUE_CATEGORY": "Pumps",
             "NET_AMOUNT_2024": 1000, "NET_AMOUNT_2025": 1600}]
    NOTE = ("This answer covers only part of the periods you asked about "
            "(2024, 2025); the query returned a single period.")

    def _render(self, confidence_context):
        import asyncio

        payload: dict = {}

        class _Adapter:
            async def send_assistant_response(self, event, response):
                payload.update(response)

            async def send_message(self, event, text):
                payload["text"] = text

        class _Event:
            platform = "portal"
            schema_hint = ""

        import core.result_renderer as rr
        asyncio.run(rr._send_results(
            _Event(), _Adapter(), "compare 2025 against 2024 by revenue category",
            list(self.ROWS), "SELECT 1", 10, None, "acct",
            {"db_type": "azure_sql", "id": 1},
            confidence_context=confidence_context,
            cache_result=False,
        ))
        return payload

    def test_multi_period_caveats_become_coverage_caveats(self):
        payload = self._render({"multi_period_caveats": [self.NOTE]})
        self.assertIn(self.NOTE, payload.get("coverage_caveats") or [])

    def test_an_answer_with_no_caveats_gains_none(self):
        """The control. Without it the assertion above would pass on a
        renderer that appended that sentence to every answer."""
        payload = self._render({})
        self.assertNotIn("coverage_caveats", payload)


class ThePeriodColumnsCarryADisplayFormat(unittest.TestCase):
    """core/chart_spec.py demotes a whole-number column whose name matches none
    of its currency/percent/count patterns to an identifier, and the chart then
    loses both period series. The pipeline publishes a format for exactly the
    columns the plan owns, on the explicit_formats channel that already
    exists."""

    ROWS = [{"REGION": "North", "TONNAGE_2024": 3400, "TONNAGE_2025": 3800}]

    def _formats(self, metrics):
        from core.multi_period import period_columns_for_plan
        from core.query_pipeline import _multi_period_column_formats

        return _multi_period_column_formats(
            period_columns_for_plan(self.ROWS, _fresh_plan()), metrics)

    def test_a_generic_measure_gets_number(self):
        self.assertEqual(self._formats([]),
                         {"TONNAGE_2024": "number", "TONNAGE_2025": "number"})

    def test_one_matched_metric_lends_its_own_format(self):
        self.assertEqual(
            self._formats([{"name": "Net revenue", "result_format": "currency"}]),
            {"TONNAGE_2024": "currency", "TONNAGE_2025": "currency"})

    def test_several_metrics_are_ambiguous_so_number_wins(self):
        self.assertEqual(
            self._formats([{"result_format": "currency"}, {"result_format": "percentage"}]),
            {"TONNAGE_2024": "number", "TONNAGE_2025": "number"})

    def test_no_period_columns_publishes_nothing(self):
        from core.query_pipeline import _multi_period_column_formats
        self.assertEqual(_multi_period_column_formats({}, []), {})


if __name__ == "__main__":
    unittest.main()
