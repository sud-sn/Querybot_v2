"""
tests/test_quality_scorer.py

Unit tests for core/quality_scorer.py
All tests are pure-function / no I/O.
"""

import unittest
from core.quality_scorer import (
    score_trace,
    classify_score,
    net_feedback_delta,
    W_SQL_VALIDATION,
    W_EXECUTION_SUCCESS,
    W_METRIC_COMPLIANCE,
    W_SCHEMA_ACL_COMPLIANCE,
    W_ENTITY_GRAPH_COMPLIANCE,
    W_NO_REPAIR,
    W_SUCCESSFUL_REPAIR,
    W_USABLE_RESULT,
    FEEDBACK_POSITIVE,
    FEEDBACK_NEGATIVE,
    THRESHOLD_POSITIVE,
    THRESHOLD_REVIEW_LO,
)


class TestScoreTracePerfectRun(unittest.TestCase):
    """A query that passed everything with no repair gets the max raw score."""

    def setUp(self):
        self.score, self.ev = score_trace(
            validation_passed=True,
            execution_success=True,
            had_repair=False,
            repair_succeeded=False,
            row_count=10,
        )

    def test_max_score_is_100(self):
        expected = (
            W_SQL_VALIDATION + W_EXECUTION_SUCCESS +
            W_METRIC_COMPLIANCE + W_SCHEMA_ACL_COMPLIANCE +
            W_ENTITY_GRAPH_COMPLIANCE + W_NO_REPAIR + W_USABLE_RESULT
        )
        self.assertEqual(self.score, expected)

    def test_evidence_keys_present(self):
        for key in (
            "sql_validation", "execution_success",
            "metric_compliance", "schema_acl_compliance",
            "entity_graph_compliance", "repair",
            "usable_result", "technical_score", "final_score",
        ):
            self.assertIn(key, self.ev)

    def test_no_repair_label(self):
        self.assertIn("no_repair", self.ev["repair"])

    def test_final_equals_technical_when_no_feedback(self):
        self.assertEqual(self.ev["technical_score"], self.ev["final_score"])


class TestScoreTraceValidationFail(unittest.TestCase):
    """Failed SQL validation should zero out the sql_validation dimension."""

    def setUp(self):
        self.score, self.ev = score_trace(
            validation_passed=False,
            execution_success=True,
            had_repair=False,
            repair_succeeded=False,
            row_count=5,
        )

    def test_sql_validation_pts_zero(self):
        self.assertEqual(self.ev["sql_validation"], 0)

    def test_score_below_max(self):
        self.assertLess(self.score, 100)


class TestScoreTraceExecutionFail(unittest.TestCase):
    """Failed execution zeroes the execution dimension and usable_result."""

    def setUp(self):
        self.score, self.ev = score_trace(
            validation_passed=True,
            execution_success=False,
            had_repair=False,
            repair_succeeded=False,
            row_count=0,
        )

    def test_execution_pts_zero(self):
        self.assertEqual(self.ev["execution_success"], 0)

    def test_usable_result_zero(self):
        self.assertEqual(self.ev["usable_result"], 0)


class TestScoreTraceRepair(unittest.TestCase):
    """Repair path: had_repair=True, repair_succeeded controls sub-dimension."""

    def test_repair_succeeded_gives_w_successful_repair(self):
        _, ev = score_trace(
            validation_passed=True,
            execution_success=True,
            had_repair=True,
            repair_succeeded=True,
            row_count=3,
        )
        self.assertIn("repair_succeeded", ev["repair"])

    def test_repair_failed_gives_zero(self):
        _, ev = score_trace(
            validation_passed=True,
            execution_success=True,
            had_repair=True,
            repair_succeeded=False,
            row_count=3,
        )
        self.assertIn("repair_failed", ev["repair"])

    def test_no_repair_beats_repair_succeeded(self):
        score_clean, _ = score_trace(
            validation_passed=True, execution_success=True,
            had_repair=False, repair_succeeded=False, row_count=5,
        )
        score_repaired, _ = score_trace(
            validation_passed=True, execution_success=True,
            had_repair=True, repair_succeeded=True, row_count=5,
        )
        self.assertGreater(score_clean, score_repaired)

    def test_no_and_successful_repair_mutually_exclusive(self):
        """Scores must differ — W_NO_REPAIR != W_SUCCESSFUL_REPAIR."""
        self.assertNotEqual(W_NO_REPAIR, W_SUCCESSFUL_REPAIR)


class TestScoreTraceZeroRows(unittest.TestCase):
    """Zero rows: no penalty, no bonus, with an evidence note."""

    def setUp(self):
        self.score_zero, self.ev_zero = score_trace(
            validation_passed=True,
            execution_success=True,
            had_repair=False,
            repair_succeeded=False,
            row_count=0,
        )
        self.score_rows, _ = score_trace(
            validation_passed=True,
            execution_success=True,
            had_repair=False,
            repair_succeeded=False,
            row_count=5,
        )

    def test_zero_rows_has_note(self):
        self.assertIn("zero_rows_note", self.ev_zero)

    def test_zero_rows_lower_than_nonzero(self):
        self.assertLess(self.score_zero, self.score_rows)

    def test_usable_result_zero_for_zero_rows(self):
        self.assertEqual(self.ev_zero["usable_result"], 0)


class TestScoreTracePartialCompliance(unittest.TestCase):
    """Partial compliance floats (0.0–1.0) produce prorated points."""

    def test_50_pct_metric_compliance(self):
        _, ev = score_trace(
            validation_passed=True, execution_success=True,
            had_repair=False, repair_succeeded=False,
            metric_compliance=0.5, row_count=1,
        )
        expected = round(W_METRIC_COMPLIANCE * 0.5)
        self.assertEqual(ev["metric_compliance"], expected)

    def test_zero_compliance_gives_zero_pts(self):
        _, ev = score_trace(
            validation_passed=True, execution_success=True,
            had_repair=False, repair_succeeded=False,
            metric_compliance=0.0, schema_acl_compliance=0.0,
            entity_graph_compliance=0.0, row_count=1,
        )
        self.assertEqual(ev["metric_compliance"], 0)
        self.assertEqual(ev["schema_acl_compliance"], 0)
        self.assertEqual(ev["entity_graph_compliance"], 0)

    def test_compliance_clamped_above_1(self):
        _, ev = score_trace(
            validation_passed=True, execution_success=True,
            had_repair=False, repair_succeeded=False,
            metric_compliance=1.5, row_count=1,
        )
        # Should be clamped to 1.0 → W_METRIC_COMPLIANCE points
        self.assertEqual(ev["metric_compliance"], W_METRIC_COMPLIANCE)

    def test_compliance_clamped_below_0(self):
        _, ev = score_trace(
            validation_passed=True, execution_success=True,
            had_repair=False, repair_succeeded=False,
            metric_compliance=-0.5, row_count=1,
        )
        self.assertEqual(ev["metric_compliance"], 0)

    def test_partial_compliance_note_when_below_100(self):
        _, ev = score_trace(
            validation_passed=True, execution_success=True,
            had_repair=False, repair_succeeded=False,
            metric_compliance=0.7, row_count=1,
        )
        self.assertIn("metric_compliance_pct", ev)
        self.assertEqual(ev["metric_compliance_pct"], 70)

    def test_no_partial_note_at_100_pct(self):
        _, ev = score_trace(
            validation_passed=True, execution_success=True,
            had_repair=False, repair_succeeded=False,
            metric_compliance=1.0, row_count=1,
        )
        self.assertNotIn("metric_compliance_pct", ev)


class TestScoreTraceFeedbackDelta(unittest.TestCase):
    """Feedback delta is applied on top of the technical score."""

    def test_positive_delta_raises_score(self):
        base, _ = score_trace(
            validation_passed=True, execution_success=True,
            had_repair=False, repair_succeeded=False, row_count=5,
        )
        with_pos, ev = score_trace(
            validation_passed=True, execution_success=True,
            had_repair=False, repair_succeeded=False, row_count=5,
            feedback_delta=FEEDBACK_POSITIVE,
        )
        self.assertEqual(with_pos, base + FEEDBACK_POSITIVE)
        self.assertIn("feedback_delta", ev)

    def test_negative_delta_lowers_score(self):
        base, _ = score_trace(
            validation_passed=True, execution_success=True,
            had_repair=False, repair_succeeded=False, row_count=5,
        )
        with_neg, _ = score_trace(
            validation_passed=True, execution_success=True,
            had_repair=False, repair_succeeded=False, row_count=5,
            feedback_delta=FEEDBACK_NEGATIVE,
        )
        self.assertEqual(with_neg, base + FEEDBACK_NEGATIVE)

    def test_score_never_below_zero(self):
        score, _ = score_trace(
            validation_passed=False, execution_success=False,
            had_repair=True, repair_succeeded=False, row_count=0,
            feedback_delta=-9999,
        )
        self.assertGreaterEqual(score, 0)

    def test_zero_delta_not_in_evidence(self):
        _, ev = score_trace(
            validation_passed=True, execution_success=True,
            had_repair=False, repair_succeeded=False, row_count=5,
            feedback_delta=0,
        )
        # feedback_delta key only added when non-zero
        self.assertNotIn("feedback_delta", ev)


class TestClassifyScore(unittest.TestCase):
    """classify_score maps score + negative-feedback flag to category."""

    def test_85_is_positive(self):
        self.assertEqual(classify_score(THRESHOLD_POSITIVE), "positive")

    def test_above_85_is_positive(self):
        self.assertEqual(classify_score(95), "positive")

    def test_84_is_review(self):
        self.assertEqual(classify_score(84), "review")

    def test_60_is_review(self):
        self.assertEqual(classify_score(THRESHOLD_REVIEW_LO), "review")

    def test_59_is_negative(self):
        self.assertEqual(classify_score(59), "negative")

    def test_zero_is_negative(self):
        self.assertEqual(classify_score(0), "negative")

    def test_net_negative_feedback_overrides_to_negative(self):
        # Even a score of 90 becomes "negative" if net feedback is negative
        self.assertEqual(classify_score(90, has_net_negative_feedback=True), "negative")

    def test_net_negative_false_does_not_override(self):
        self.assertEqual(classify_score(90, has_net_negative_feedback=False), "positive")

    def test_threshold_boundary_review_lo(self):
        self.assertEqual(classify_score(THRESHOLD_REVIEW_LO - 1), "negative")
        self.assertEqual(classify_score(THRESHOLD_REVIEW_LO), "review")

    def test_threshold_boundary_positive(self):
        self.assertEqual(classify_score(THRESHOLD_POSITIVE - 1), "review")
        self.assertEqual(classify_score(THRESHOLD_POSITIVE), "positive")


class TestNetFeedbackDelta(unittest.TestCase):
    """net_feedback_delta converts vote counts to a fixed delta."""

    def test_more_ups_returns_positive(self):
        self.assertEqual(net_feedback_delta(5, 2), FEEDBACK_POSITIVE)

    def test_more_downs_returns_negative(self):
        self.assertEqual(net_feedback_delta(1, 4), FEEDBACK_NEGATIVE)

    def test_equal_votes_returns_zero(self):
        self.assertEqual(net_feedback_delta(3, 3), 0)

    def test_no_votes_returns_zero(self):
        self.assertEqual(net_feedback_delta(0, 0), 0)

    def test_one_up_zero_down_is_positive(self):
        self.assertEqual(net_feedback_delta(1, 0), FEEDBACK_POSITIVE)

    def test_zero_up_one_down_is_negative(self):
        self.assertEqual(net_feedback_delta(0, 1), FEEDBACK_NEGATIVE)

    def test_large_counts_only_direction_matters(self):
        """100 ups vs 1 down → same result as 2 ups vs 1 down (direction wins)."""
        self.assertEqual(
            net_feedback_delta(100, 1),
            net_feedback_delta(2, 1),
        )

    def test_feedback_positive_constant_is_positive(self):
        self.assertGreater(FEEDBACK_POSITIVE, 0)

    def test_feedback_negative_constant_is_negative(self):
        self.assertLess(FEEDBACK_NEGATIVE, 0)


class TestWeightConstants(unittest.TestCase):
    """Sanity-check the weight constants add up to exactly 100."""

    def test_weights_sum_to_100(self):
        total = (
            W_SQL_VALIDATION + W_EXECUTION_SUCCESS +
            W_METRIC_COMPLIANCE + W_SCHEMA_ACL_COMPLIANCE +
            W_ENTITY_GRAPH_COMPLIANCE + W_NO_REPAIR + W_USABLE_RESULT
        )
        self.assertEqual(total, 100)

    def test_repair_variants_both_below_no_repair(self):
        self.assertLess(W_SUCCESSFUL_REPAIR, W_NO_REPAIR)


if __name__ == "__main__":
    unittest.main()
