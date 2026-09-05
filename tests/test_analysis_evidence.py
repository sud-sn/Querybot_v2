"""The evidence engine — core/analysis_evidence.py.

Every test here executes the real detector or the real entry point and asserts
on the returned Finding objects. Nothing asserts on source text, and no test
hands in the value it is checking the code can compute.

The suite is organised by what would break if the engine regressed:

  * detectors — each one, alone, on data shaped to trigger it and on data
    shaped to prove it stays quiet;
  * ranking — materiality ordering, the family/column de-duplication, and the
    per-family cap;
  * redundancy rules — the three cases where two true findings say one thing;
  * governance — labels are separable from numbers, and withholding them
    leaves the aggregates intact;
  * the entry point — column classification, bounds, and failing open.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.analysis_evidence import (  # noqa: E402
    BELOW_AVERAGE_CLUSTER,
    COMOVEMENT,
    CONCENTRATION_LEADER,
    CONCENTRATION_PARETO,
    CORRELATION,
    LONG_TAIL,
    MAX_PER_FAMILY,
    OUTLIERS,
    RANGE_SPAN,
    SKEW_RIGHT,
    SPREAD_HIGH,
    SPREAD_LOW,
    TREND_DOWN,
    TREND_FLAT,
    TREND_REVERSAL,
    TREND_UP,
    AnalysisEvidence,
    Finding,
    build_evidence,
    classify_columns,
    concentration_findings,
    family_of,
    find_temporal_column,
    outlier_findings,
    rank_findings,
    relationship_findings,
    spread_findings,
    trend_findings,
)


def _kinds(findings):
    return [f.kind for f in findings]


def _series(values, col="REVENUE", period="MONTH"):
    return [
        {period: f"2026-{i:02d}", col: v}
        for i, v in enumerate(values, start=1)
    ]


def _categories(pairs, label="CUSTOMER", col="REVENUE"):
    return [{label: name, col: value} for name, value in pairs]


# ══════════════════════════════════════════════════════════════════════════════
# Trend
# ══════════════════════════════════════════════════════════════════════════════

class TestTrend(unittest.TestCase):

    def test_a_rising_series_is_reported_as_rising_with_its_size(self):
        found = trend_findings(_series([100, 110, 130, 200]), "REVENUE", "MONTH")
        up = [f for f in found if f.kind == TREND_UP]
        self.assertEqual(len(up), 1)
        self.assertEqual(up[0].numbers["pct"], 100.0)
        self.assertEqual(up[0].numbers["first"], 100.0)
        self.assertEqual(up[0].numbers["last"], 200.0)

    def test_a_falling_series_is_reported_as_falling(self):
        found = trend_findings(_series([200, 180, 150, 100]), "REVENUE", "MONTH")
        self.assertIn(TREND_DOWN, _kinds(found))
        self.assertNotIn(TREND_UP, _kinds(found))

    def test_drift_inside_the_flat_band_is_flat_not_directional(self):
        # 100 -> 102 is a 2% rise; below FLAT_TREND_PCT it must not be called
        # an upward trend, or the summary contradicts the "holding steady"
        # wording the renderer uses on the same series.
        found = trend_findings(_series([100, 101, 100, 102]), "REVENUE", "MONTH")
        self.assertIn(TREND_FLAT, _kinds(found))
        self.assertNotIn(TREND_UP, _kinds(found))

    def test_a_series_that_turns_reports_the_reversal_with_peak_and_trough(self):
        found = trend_findings(_series([100, 200, 300, 120, 90]), "REVENUE", "MONTH")
        turns = [f for f in found if f.kind == TREND_REVERSAL]
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].numbers["turns"], 1.0)
        self.assertEqual(turns[0].numbers["peak"], 300.0)
        self.assertEqual(turns[0].numbers["trough"], 90.0)

    def test_noise_around_zero_is_not_counted_as_turns(self):
        # Alternating by well under the flat band on a large scale. A naive
        # sign-change count reads this as four reversals.
        found = trend_findings(
            _series([1000, 1001, 1000, 1001, 1000, 1001]), "REVENUE", "MONTH")
        self.assertNotIn(TREND_REVERSAL, _kinds(found))

    def test_a_reversal_outranks_the_direction_it_ends_on(self):
        found = rank_findings(
            trend_findings(_series([100, 400, 300, 150]), "REVENUE", "MONTH"))
        self.assertEqual(found[0].kind, TREND_REVERSAL)

    def test_a_single_point_yields_nothing(self):
        self.assertEqual(trend_findings(_series([100]), "REVENUE", "MONTH"), [])

    def test_a_series_starting_at_zero_does_not_divide_by_it(self):
        found = trend_findings(_series([0, 50, 90]), "REVENUE", "MONTH")
        self.assertNotIn(TREND_UP, _kinds(found))
        self.assertNotIn(TREND_DOWN, _kinds(found))


# ══════════════════════════════════════════════════════════════════════════════
# Concentration
# ══════════════════════════════════════════════════════════════════════════════

class TestConcentration(unittest.TestCase):

    LEADER_DOMINATED = [
        ("EMCO", 6200), ("Acme", 900), ("Borel", 700), ("Duval", 500),
        ("Fabre", 400), ("Gide", 300), ("Hugo", 200), ("Ivry", 100),
        ("Jarry", 50), ("Kern", 20),
    ]
    # No row above the leader threshold, but the top fifth still holds most
    # of the total — the shape Pareto exists to describe.
    TRUE_PARETO = [
        ("a", 34), ("b", 33), ("c", 5), ("d", 5), ("e", 5),
        ("f", 3), ("g", 3), ("h", 4), ("i", 4), ("j", 4),
    ]

    def test_a_dominant_group_is_reported_with_its_share_and_name(self):
        found = concentration_findings(
            _categories(self.LEADER_DOMINATED), "REVENUE", "CUSTOMER")
        leader = [f for f in found if f.kind == CONCENTRATION_LEADER]
        self.assertEqual(len(leader), 1)
        self.assertEqual(leader[0].labels["leader"], "EMCO")
        self.assertAlmostEqual(leader[0].numbers["share"], 66.2, places=1)
        self.assertEqual(leader[0].numbers["value"], 6200.0)

    def test_an_even_spread_reports_no_leader(self):
        even = _categories([(f"c{i}", 100) for i in range(10)])
        self.assertNotIn(CONCENTRATION_LEADER,
                         _kinds(concentration_findings(even, "REVENUE", "CUSTOMER")))

    def test_pareto_is_reported_when_no_single_row_dominates(self):
        found = concentration_findings(
            _categories(self.TRUE_PARETO), "REVENUE", "CUSTOMER")
        pareto = [f for f in found if f.kind == CONCENTRATION_PARETO]
        self.assertEqual(len(pareto), 1)
        self.assertEqual(pareto[0].numbers["share"], 67.0)
        self.assertEqual(pareto[0].numbers["n_top"], 2.0)
        self.assertEqual(pareto[0].numbers["n_total"], 10.0)

    def test_pareto_is_suppressed_when_it_only_restates_the_leader(self):
        # The regression this guards: leader 66.2% and top-2 75.8% are one
        # fact, and reporting both spends two sentences on one number.
        found = concentration_findings(
            _categories(self.LEADER_DOMINATED), "REVENUE", "CUSTOMER")
        self.assertIn(CONCENTRATION_LEADER, _kinds(found))
        self.assertNotIn(CONCENTRATION_PARETO, _kinds(found))

    def test_a_negligible_tail_is_reported_with_how_many_rows_it_holds(self):
        found = concentration_findings(
            _categories([("a", 30), ("b", 29), ("c", 28), ("d", 27),
                         ("e", 3), ("f", 2), ("g", 2), ("h", 2),
                         ("i", 1), ("j", 1)]),
            "REVENUE", "CUSTOMER")
        tail = [f for f in found if f.kind == LONG_TAIL]
        self.assertEqual(len(tail), 1)
        self.assertEqual(tail[0].numbers["n_bottom"], 5.0)
        self.assertLess(tail[0].numbers["share"], 10.0)

    def test_negative_and_zero_values_do_not_break_the_share_arithmetic(self):
        found = concentration_findings(
            _categories([("a", -5), ("b", 0), ("c", 100), ("d", 3)]),
            "REVENUE", "CUSTOMER")
        for finding in found:
            self.assertLessEqual(finding.numbers.get("share", 0.0), 100.0)
            self.assertGreaterEqual(finding.numbers.get("share", 0.0), 0.0)

    def test_one_row_yields_nothing(self):
        self.assertEqual(
            concentration_findings(_categories([("a", 5)]), "REVENUE", "CUSTOMER"), [])


# ══════════════════════════════════════════════════════════════════════════════
# Spread
# ══════════════════════════════════════════════════════════════════════════════

class TestSpread(unittest.TestCase):

    def test_a_wildly_varying_column_reports_high_spread(self):
        rows = [{"V": v} for v in [1, 500, 3, 900, 2, 700]]
        found = spread_findings(rows, "V")
        high = [f for f in found if f.kind == SPREAD_HIGH]
        self.assertEqual(len(high), 1)
        self.assertGreater(high[0].numbers["cv"], 0.6)

    def test_a_uniform_column_reports_low_spread(self):
        rows = [{"V": v} for v in [100, 101, 100, 99, 100, 101]]
        self.assertIn(SPREAD_LOW, _kinds(spread_findings(rows, "V")))

    def test_high_and_low_spread_are_never_both_reported(self):
        for values in ([1, 500, 3, 900], [100, 101, 100, 99], [10, 20, 30, 40]):
            kinds = _kinds(spread_findings([{"V": v} for v in values], "V"))
            self.assertFalse(SPREAD_HIGH in kinds and SPREAD_LOW in kinds, values)

    def test_a_long_right_tail_is_reported_with_the_mean_median_ratio(self):
        rows = [{"V": v} for v in [1, 1, 1, 1, 1, 1, 1, 1, 100]]
        skew = [f for f in spread_findings(rows, "V") if f.kind == SKEW_RIGHT]
        self.assertEqual(len(skew), 1)
        self.assertEqual(skew[0].numbers["median"], 1.0)
        self.assertGreater(skew[0].numbers["ratio"], 1.5)

    def test_a_cluster_far_below_average_is_counted(self):
        rows = [{"V": v} for v in [1, 1, 1, 100, 100, 100, 100, 100]]
        cluster = [f for f in spread_findings(rows, "V")
                   if f.kind == BELOW_AVERAGE_CLUSTER]
        self.assertEqual(len(cluster), 1)
        self.assertEqual(cluster[0].numbers["count"], 3.0)
        self.assertEqual(cluster[0].numbers["n_total"], 8.0)

    def test_the_range_reports_the_multiple_between_smallest_and_largest(self):
        rows = [{"V": v} for v in [10, 20, 30, 110]]
        span = [f for f in spread_findings(rows, "V") if f.kind == RANGE_SPAN]
        self.assertEqual(len(span), 1)
        self.assertEqual(span[0].numbers["low"], 10.0)
        self.assertEqual(span[0].numbers["high"], 110.0)
        self.assertEqual(span[0].numbers["multiple"], 11.0)

    def test_a_range_crossing_zero_reports_no_meaningless_multiple(self):
        rows = [{"V": v} for v in [-50, -10, 5, 40]]
        for span in spread_findings(rows, "V"):
            if span.kind == RANGE_SPAN:
                self.assertNotIn("multiple", span.numbers)


# ══════════════════════════════════════════════════════════════════════════════
# Outliers and relationships — delegated detectors
# ══════════════════════════════════════════════════════════════════════════════

class TestOutliers(unittest.TestCase):

    def test_a_clear_outlier_is_counted_against_the_row_total(self):
        rows = [{"V": v} for v in [10, 11, 10, 12, 11, 10, 11, 900]]
        found = outlier_findings(rows, "V")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].kind, OUTLIERS)
        self.assertEqual(found[0].numbers["count"], 1.0)
        self.assertEqual(found[0].numbers["n_total"], 8.0)

    def test_an_even_column_reports_no_outliers(self):
        rows = [{"V": v} for v in [10, 11, 10, 12, 11, 10, 11, 12]]
        self.assertEqual(outlier_findings(rows, "V"), [])

    def test_a_handful_of_outliers_outranks_half_the_rows_being_flagged(self):
        # Materiality peaks near a tenth of the rows: when most rows are
        # "anomalous" the threshold is the story, not the data.
        few = outlier_findings([{"V": v} for v in
                                [10, 10, 10, 10, 10, 10, 10, 10, 10, 900]], "V")
        self.assertTrue(few)
        many = outlier_findings([{"V": v} for v in
                                 [1, 900, 2, 950, 3, 1000, 4, 890]], "V")
        if many:
            self.assertGreater(few[0].materiality, many[0].materiality)

    def test_too_few_rows_to_judge_yields_nothing(self):
        self.assertEqual(outlier_findings([{"V": 1}, {"V": 900}], "V"), [])


class TestRelationships(unittest.TestCase):

    def test_a_strong_correlation_is_reported_with_r(self):
        rows = [{"X": i, "Y": i * 3.0} for i in range(1, 12)]
        found = relationship_findings(rows, "X", "Y")
        corr = [f for f in found if f.kind == CORRELATION]
        self.assertEqual(len(corr), 1)
        self.assertEqual(corr[0].numbers["r"], 1.0)
        self.assertEqual(corr[0].numbers["n"], 11.0)

    def test_a_strong_negative_correlation_keeps_its_sign(self):
        rows = [{"X": i, "Y": -i * 3.0} for i in range(1, 12)]
        corr = [f for f in relationship_findings(rows, "X", "Y")
                if f.kind == CORRELATION]
        self.assertEqual(len(corr), 1)
        self.assertLess(corr[0].numbers["r"], 0)

    def test_unrelated_columns_report_neither_correlation_nor_comovement(self):
        rows = [{"X": x, "Y": y} for x, y in
                [(1, 9), (2, 1), (3, 8), (4, 2), (5, 7), (6, 3), (7, 6), (8, 4)]]
        self.assertEqual(relationship_findings(rows, "X", "Y"), [])

    def test_a_non_linear_pairing_pearson_misses_is_caught_as_comovement(self):
        # Both columns sit above or below their own mean together on most
        # rows, but the relationship is not linear enough for |r| >= 0.5.
        rows = [{"X": x, "Y": y} for x, y in
                [(1, 1), (2, 1), (3, 2), (4, 2), (5, 90),
                 (6, 95), (7, 99), (8, 100), (9, 100), (10, 100)]]
        kinds = _kinds(relationship_findings(rows, "X", "Y"))
        self.assertTrue(CORRELATION in kinds or COMOVEMENT in kinds)


# ══════════════════════════════════════════════════════════════════════════════
# Ranking
# ══════════════════════════════════════════════════════════════════════════════

class TestRanking(unittest.TestCase):

    def test_findings_come_back_strongest_first(self):
        ranked = rank_findings([
            Finding(kind=SPREAD_LOW, columns=("A",), materiality=0.1),
            Finding(kind=CONCENTRATION_LEADER, columns=("B",), materiality=0.9),
            Finding(kind=OUTLIERS, columns=("C",), materiality=0.5),
        ])
        self.assertEqual([f.materiality for f in ranked], [0.9, 0.5, 0.1])

    def test_one_column_contributes_one_finding_per_family(self):
        ranked = rank_findings([
            Finding(kind=SPREAD_HIGH, columns=("V",), materiality=0.5),
            Finding(kind=SKEW_RIGHT, columns=("V",), materiality=0.4),
            Finding(kind=RANGE_SPAN, columns=("V",), materiality=0.3),
        ])
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].kind, SPREAD_HIGH)

    def test_two_columns_may_each_contribute_to_the_same_family(self):
        ranked = rank_findings([
            Finding(kind=SPREAD_HIGH, columns=("A",), materiality=0.5),
            Finding(kind=SPREAD_HIGH, columns=("B",), materiality=0.4),
        ])
        self.assertEqual(len(ranked), 2)

    def test_a_family_is_capped_so_four_measures_cannot_fill_the_summary(self):
        ranked = rank_findings([
            Finding(kind=RANGE_SPAN, columns=(c,), materiality=0.3)
            for c in ("A", "B", "C", "D", "E")
        ])
        self.assertEqual(len(ranked), MAX_PER_FAMILY)

    def test_ranking_is_stable_for_equal_materiality(self):
        findings = [
            Finding(kind=OUTLIERS, columns=("B",), materiality=0.4),
            Finding(kind=OUTLIERS, columns=("A",), materiality=0.4),
        ]
        first = [f.primary_column for f in rank_findings(findings)]
        second = [f.primary_column for f in rank_findings(list(reversed(findings)))]
        self.assertEqual(first, second)

    def test_kinds_that_say_the_same_sort_of_thing_share_a_family(self):
        # De-duplication is per family, so a kind landing in the wrong family
        # (or in none, where family_of falls back to the kind itself) silently
        # stops de-duplicating. Assert the groupings the ranker relies on.
        self.assertEqual(
            {family_of(k) for k in (TREND_UP, TREND_DOWN, TREND_FLAT, TREND_REVERSAL)},
            {"trend"})
        self.assertEqual(
            {family_of(k) for k in
             (CONCENTRATION_LEADER, CONCENTRATION_PARETO, LONG_TAIL)},
            {"concentration"})
        self.assertEqual(
            {family_of(k) for k in
             (SPREAD_HIGH, SPREAD_LOW, SKEW_RIGHT, BELOW_AVERAGE_CLUSTER, RANGE_SPAN)},
            {"spread"})
        self.assertEqual({family_of(CORRELATION), family_of(COMOVEMENT)},
                         {"relationship"})


# ══════════════════════════════════════════════════════════════════════════════
# Governance — labels are separable from aggregates
# ══════════════════════════════════════════════════════════════════════════════

class TestLabelsAreGoverned(unittest.TestCase):

    ROWS = _categories([
        ("Ospedale San Raffaele", 6200), ("Acme", 900), ("Borel", 700),
        ("Duval", 500), ("Fabre", 400), ("Gide", 300),
    ])

    def test_a_data_derived_name_travels_in_labels_never_in_numbers(self):
        evidence = build_evidence(self.ROWS)
        leader = evidence.of_kind(CONCENTRATION_LEADER)
        self.assertEqual(len(leader), 1)
        self.assertEqual(leader[0].labels["leader"], "Ospedale San Raffaele")
        for value in leader[0].numbers.values():
            self.assertIsInstance(value, float)

    def test_withholding_labels_keeps_the_findings_and_their_numbers(self):
        with_labels = build_evidence(self.ROWS, include_labels=True)
        without = build_evidence(self.ROWS, include_labels=False)
        self.assertEqual(_kinds(with_labels.findings), _kinds(without.findings))
        self.assertEqual(
            [f.numbers for f in with_labels.findings],
            [f.numbers for f in without.findings],
        )

    def test_withholding_labels_leaves_no_data_string_anywhere_in_the_evidence(self):
        evidence = build_evidence(self.ROWS, include_labels=False)
        self.assertFalse(evidence.labels_included)
        for finding in evidence.findings:
            self.assertEqual(finding.labels, {})
            self.assertFalse(finding.needs_label)
        # The whole object, rendered, must not carry a customer's name.
        self.assertNotIn("Ospedale", repr(evidence))
        self.assertNotIn("Raffaele", repr(evidence))

    def test_the_label_bearing_finding_declares_that_it_needs_one(self):
        # The phrasing layer chooses a label-free sentence off this flag.
        evidence = build_evidence(self.ROWS, include_labels=True)
        leader = evidence.of_kind(CONCENTRATION_LEADER)[0]
        self.assertTrue(leader.needs_label)
        spread = [f for f in evidence.findings if f.family == "spread"]
        for finding in spread:
            self.assertFalse(finding.needs_label)


# ══════════════════════════════════════════════════════════════════════════════
# The entry point
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildEvidence(unittest.TestCase):

    def test_columns_are_split_into_measures_and_labels(self):
        rows = [{"CUSTOMER": "a", "REGION": "x", "REVENUE": 10, "UNITS": 2}]
        numeric, labels = classify_columns(rows)
        self.assertEqual(set(numeric), {"REVENUE", "UNITS"})
        self.assertEqual(set(labels), {"CUSTOMER", "REGION"})

    def test_a_mostly_numeric_column_with_a_gap_is_still_a_measure(self):
        rows = [{"V": 1}, {"V": 2}, {"V": None}, {"V": 4}]
        numeric, _ = classify_columns(rows)
        self.assertIn("V", numeric)

    def test_a_period_column_is_found_and_named(self):
        rows = _series([1, 2, 3])
        _, labels = classify_columns(rows)
        self.assertEqual(find_temporal_column(rows, labels), "MONTH")

    def test_a_business_name_that_contains_a_month_is_not_a_period_column(self):
        # "Nova", "Maritime", "Mayfield", "Septic" each contain a month
        # abbreviation. A result grouped by these is not a time series, and
        # calling it one produced trend language about a categorical result.
        rows = [{"BRANCH": n, "V": i} for i, n in
                enumerate(["Nova", "Maritime", "Mayfield", "Septic", "Junction"], 1)]
        _, labels = classify_columns(rows)
        self.assertEqual(find_temporal_column(rows, labels), "")

    def test_an_empty_result_yields_empty_evidence_not_an_error(self):
        evidence = build_evidence([])
        self.assertIsInstance(evidence, AnalysisEvidence)
        self.assertEqual(evidence.findings, ())
        self.assertEqual(evidence.row_count, 0)

    def test_the_row_count_and_column_lists_are_reported(self):
        rows = _categories([("a", 5), ("b", 6), ("c", 7)])
        evidence = build_evidence(rows)
        self.assertEqual(evidence.row_count, 3)
        self.assertEqual(evidence.numeric_columns, ("REVENUE",))
        self.assertEqual(evidence.label_columns, ("CUSTOMER",))

    def test_a_wide_result_bounds_how_many_measures_are_analysed(self):
        rows = [{"R": f"r{i}", **{f"M{j}": i * (j + 1) for j in range(8)}}
                for i in range(1, 15)]
        evidence = build_evidence(rows, max_numeric_columns=2)
        analysed = {c for f in evidence.findings for c in f.columns}
        self.assertFalse(analysed & {"M2", "M3", "M4", "M5", "M6", "M7"})

    def test_every_number_asserted_is_collected_for_the_prose_check(self):
        # core.analysis_narrative rejects a model's paragraph when it contains
        # a figure this list does not.
        evidence = build_evidence(
            _series([100, 120, 140, 160, 900, 80]))
        self.assertTrue(evidence.findings)
        numbers = evidence.all_numbers()
        for finding in evidence.findings:
            for value in finding.numbers.values():
                self.assertIn(value, numbers)

    def test_a_failing_detector_costs_its_findings_not_the_answer(self):
        import core.analysis_evidence as module

        rows = _series([100, 200, 300, 120, 90])
        healthy = build_evidence(rows)
        self.assertTrue(healthy.findings)

        original = module.spread_findings
        module.spread_findings = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("detector exploded"))
        try:
            degraded = build_evidence(rows)
        finally:
            module.spread_findings = original

        self.assertIsInstance(degraded, AnalysisEvidence)
        self.assertTrue(degraded.findings)
        self.assertNotIn("spread", {f.family for f in degraded.findings})

    def test_the_dominant_leaders_own_row_is_not_also_reported_as_an_outlier(self):
        rows = _categories([
            ("EMCO", 6200), ("Acme", 90), ("Borel", 70), ("Duval", 50),
            ("Fabre", 40), ("Gide", 30), ("Hugo", 20), ("Ivry", 10),
        ])
        evidence = build_evidence(rows)
        self.assertIn(CONCENTRATION_LEADER, _kinds(evidence.findings))
        self.assertNotIn(OUTLIERS, _kinds(evidence.findings))

    def test_outliers_are_still_reported_when_no_group_dominates(self):
        # Twenty even rows plus one high one: the high row is a genuine
        # outlier and, at well under a third of the total, not a leader. The
        # suppression rule must not reach this case.
        rows = _categories(
            [(f"c{i}", 100) for i in range(20)] + [("spike", 400)])
        kinds = _kinds(build_evidence(rows).findings)
        self.assertNotIn(CONCENTRATION_LEADER, kinds)
        self.assertIn(OUTLIERS, kinds)

    def test_a_realistic_time_series_answers_with_the_trend_first(self):
        # The shape of an actual monthly revenue answer: a clear rise, no
        # dominant period, mild variation. The trend is the story.
        rows = _series([100, 108, 121, 130, 142, 155, 170, 188])
        evidence = build_evidence(rows)
        self.assertTrue(evidence.findings)
        self.assertEqual(evidence.findings[0].kind, TREND_UP)
        self.assertEqual(evidence.findings[0].numbers["last"], 188.0)


if __name__ == "__main__":
    unittest.main()
