"""Candidate selection — core/candidate_selection.py.

Every test executes the real selector. The properties asserted hardest are
the two that decide whether this makes the product better or worse:

  * the verifier chooses, and agreement only breaks ties it already ranked
    equally — because the research is explicit that the most-agreed answer is
    often not the right one; and
  * when nothing verifies, the selector returns nothing and a reason, because
    presenting the least-bad member of a set that all disagree is exactly how
    a confident wrong number reaches a user.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.candidate_selection import (  # noqa: E402
    AGREEMENT_TOLERANCE,
    DEFAULT_CANDIDATE_COUNT,
    MAX_CANDIDATE_COUNT,
    VERIFIED_SCORE,
    Candidate,
    agreeing_group,
    ambiguity_of,
    candidate_count_for,
    rank,
    select,
    summarise,
)


def _candidate(source, *, score=90, value=100.0, rows=None, validated=True,
               executed=True, error="", validation_reason=""):
    if rows is None:
        rows = [{"REVENUE": value}] if value is not None else []
    return Candidate(
        sql=f"SELECT /* {source} */ 1", source=source,
        validated=validated, validation_reason=validation_reason,
        executed=executed, error=error, rows=rows,
        verification={"score": score},
    )


# ══════════════════════════════════════════════════════════════════════════════
# When to spend the extra queries
# ══════════════════════════════════════════════════════════════════════════════

class TestAmbiguityGating(unittest.TestCase):

    def test_a_determined_plan_gets_one_candidate(self):
        # The common case, and the reason this does not triple the cost of
        # the product.
        self.assertEqual(ambiguity_of({}), [])
        self.assertEqual(candidate_count_for({}), 1)
        self.assertEqual(candidate_count_for(None), 1)

    def test_a_plan_with_one_reachable_fact_is_not_ambiguous(self):
        self.assertEqual(ambiguity_of({"considered_facts": ["S.F_SALES"]}), [])
        self.assertEqual(candidate_count_for({"considered_facts": ["S.F_SALES"]}), 1)

    def test_two_facts_are_ambiguous(self):
        plan = {"considered_facts": ["S.F_SALES", "S.F_ORDERS"]}
        self.assertEqual(ambiguity_of(plan), ["fact"])
        self.assertGreater(candidate_count_for(plan), 1)

    def test_duplicate_values_are_not_two_choices(self):
        self.assertEqual(ambiguity_of({"considered_facts": ["A", "A"]}), [])

    def test_every_source_of_ambiguity_is_reported_by_name(self):
        reasons = ambiguity_of({
            "considered_facts": ["S.F_SALES", "S.F_ORDERS"],
            "considered_date_roles": ["ORDER_DATE", "SHIP_DATE"],
            "considered_metrics": ["Net Revenue", "Gross Revenue"],
            "candidate_grains": ["day", "month"],
            "candidate_join_paths": ["a", "b"],
        })
        self.assertEqual(set(reasons),
                         {"fact", "date_role", "metric", "grain", "join_path"})

    def test_the_compilers_own_flag_outranks_the_heuristics(self):
        reasons = ambiguity_of({"ambiguous": True, "ambiguity_reason": "two_facts"})
        self.assertEqual(reasons, ["two_facts"])

    def test_the_count_is_capped(self):
        many = {key: ["a", "b"] for key in (
            "considered_facts", "considered_date_roles", "considered_metrics",
            "candidate_grains", "candidate_join_paths")}
        self.assertLessEqual(candidate_count_for(many, cap=99), MAX_CANDIDATE_COUNT)
        self.assertLessEqual(candidate_count_for(many), DEFAULT_CANDIDATE_COUNT)


# ══════════════════════════════════════════════════════════════════════════════
# A candidate's own state
# ══════════════════════════════════════════════════════════════════════════════

class TestCandidateState(unittest.TestCase):

    def test_a_candidate_that_failed_validation_is_not_usable(self):
        self.assertFalse(_candidate("a", validated=False).usable)

    def test_a_candidate_that_failed_execution_is_not_usable(self):
        self.assertFalse(_candidate("a", error="ORA-00942").usable)
        self.assertFalse(_candidate("a", executed=False).usable)

    def test_a_usable_candidate_below_the_bar_is_not_verified(self):
        self.assertTrue(_candidate("a", score=VERIFIED_SCORE - 1).usable)
        self.assertFalse(_candidate("a", score=VERIFIED_SCORE - 1).verified)
        self.assertTrue(_candidate("a", score=VERIFIED_SCORE).verified)

    def test_the_headline_is_the_first_numeric_cell(self):
        self.assertEqual(
            Candidate(sql="", rows=[{"REGION": "West", "REVENUE": 1250.0}]).headline(),
            1250.0)

    def test_a_result_with_no_number_has_no_headline(self):
        self.assertIsNone(Candidate(sql="", rows=[{"REGION": "West"}]).headline())
        self.assertIsNone(Candidate(sql="", rows=[]).headline())

    def test_a_formatted_number_is_still_a_number(self):
        self.assertEqual(
            Candidate(sql="", rows=[{"REVENUE": "1,250"}]).headline(), 1250.0)

    def test_a_missing_verification_scores_zero_rather_than_raising(self):
        self.assertEqual(Candidate(sql="", verification={}).score, 0)
        self.assertEqual(Candidate(sql="", verification={"score": "n/a"}).score, 0)


# ══════════════════════════════════════════════════════════════════════════════
# Ranking and selection
# ══════════════════════════════════════════════════════════════════════════════

class TestRanking(unittest.TestCase):

    def test_unusable_candidates_are_dropped_before_ranking(self):
        ranked = rank([
            _candidate("bad", validated=False),
            _candidate("broken", error="timeout"),
            _candidate("good"),
        ])
        self.assertEqual([c.source for c in ranked], ["good"])

    def test_the_verifier_decides_the_order(self):
        ranked = rank([
            _candidate("low", score=40),
            _candidate("high", score=95),
            _candidate("mid", score=70),
        ])
        self.assertEqual([c.source for c in ranked], ["high", "mid", "low"])

    def test_an_empty_result_ranks_below_one_with_rows_at_equal_score(self):
        ranked = rank([
            _candidate("empty", score=80, rows=[]),
            _candidate("rows", score=80),
        ])
        self.assertEqual(ranked[0].source, "rows")

    def test_ranking_is_stable_regardless_of_input_order(self):
        candidates = [_candidate("a", score=80), _candidate("b", score=80)]
        self.assertEqual([c.source for c in rank(candidates)],
                         [c.source for c in rank(list(reversed(candidates)))])


class TestSelection(unittest.TestCase):

    def test_the_verifier_chooses_not_the_majority(self):
        # The property the research is clearest about: two agreeing
        # candidates the verifier dislikes must not outvote one it accepts.
        chosen, reason = select([
            _candidate("agree_1", score=30, value=500.0),
            _candidate("agree_2", score=30, value=500.0),
            _candidate("verified", score=95, value=1250.0),
        ])
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.source, "verified")
        self.assertEqual(reason, "single_verified_candidate")

    def test_agreement_breaks_a_tie_the_verifier_could_not(self):
        chosen, reason = select([
            _candidate("alone", score=90, value=999.0),
            _candidate("agree_1", score=90, value=100.0),
            _candidate("agree_2", score=90, value=100.0),
        ])
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.headline(), 100.0)
        self.assertIn("agreement_of_2", reason)

    def test_verified_candidates_that_disagree_produce_no_answer(self):
        # Two queries the verifier likes equally, returning different
        # numbers, means the question was ambiguous in a way the plan did not
        # capture. Answering with either is a coin toss presented as fact.
        chosen, reason = select([
            _candidate("a", score=90, value=100.0),
            _candidate("b", score=90, value=880.0),
        ])
        self.assertIsNone(chosen)
        self.assertEqual(reason, "verified_candidates_disagree")

    def test_nothing_verified_among_several_produces_no_answer(self):
        chosen, reason = select([
            _candidate("a", score=40, value=100.0),
            _candidate("b", score=30, value=200.0),
        ])
        self.assertIsNone(chosen)
        self.assertEqual(reason, "no_candidate_verified")

    def test_a_single_unverified_candidate_is_still_returned(self):
        # A determined plan produces one candidate, and refusing it would
        # change what the product does for every ordinary question. The
        # reason records that the verifier was not satisfied.
        chosen, reason = select([_candidate("only", score=20)])
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.source, "only")
        self.assertEqual(reason, "single_candidate_unverified")

    def test_the_highest_score_wins_when_it_is_unique(self):
        chosen, reason = select([
            _candidate("best", score=95, value=100.0),
            _candidate("good", score=80, value=200.0),
        ])
        self.assertEqual(chosen.source, "best")
        self.assertEqual(reason, "highest_verification_score")

    def test_no_candidates_at_all_is_reported_not_crashed(self):
        self.assertEqual(select([]), (None, "no_candidates"))

    def test_an_execution_failure_is_reported_with_its_error(self):
        chosen, reason = select([_candidate("a", error="ORA-00942: table not found")])
        self.assertIsNone(chosen)
        self.assertIn("ORA-00942", reason)

    def test_a_validation_failure_is_reported_with_its_reason(self):
        chosen, reason = select([
            _candidate("a", validated=False, validation_reason="raw fact-to-fact join"),
        ])
        self.assertIsNone(chosen)
        self.assertIn("fact-to-fact", reason)

    def test_candidates_with_no_number_to_compare_still_select(self):
        # A list of names has no headline figure. The verifier is the whole
        # decision there, and it must not fall through to "disagree".
        chosen, reason = select([
            _candidate("a", score=95, rows=[{"NAME": "West"}]),
            _candidate("b", score=95, rows=[{"NAME": "East"}]),
        ])
        self.assertIsNotNone(chosen)
        self.assertEqual(reason, "highest_verification_score")


class TestAgreement(unittest.TestCase):

    def test_identical_figures_agree(self):
        group = agreeing_group([_candidate("a"), _candidate("b"), _candidate("c")])
        self.assertEqual(len(group), 3)

    def test_a_rounding_difference_still_agrees(self):
        group = agreeing_group([
            _candidate("a", value=1000.0),
            _candidate("b", value=1000.0 * (1 + AGREEMENT_TOLERANCE / 2)),
        ])
        self.assertEqual(len(group), 2)

    def test_a_materially_different_figure_does_not_agree(self):
        group = agreeing_group([
            _candidate("a", value=1000.0),
            _candidate("b", value=1100.0),
        ])
        self.assertEqual(len(group), 1)

    def test_the_tolerance_is_tight_enough_to_mean_something(self):
        # A loose band would let two materially different answers vote for
        # each other, which is worse than no agreement rule at all.
        self.assertLess(AGREEMENT_TOLERANCE, 0.02)

    def test_the_largest_agreeing_set_wins(self):
        group = agreeing_group([
            _candidate("a", value=100.0),
            _candidate("b", value=100.0),
            _candidate("c", value=100.0),
            _candidate("d", value=900.0),
            _candidate("e", value=900.0),
        ])
        self.assertEqual(len(group), 3)

    def test_zero_is_compared_without_dividing_by_it(self):
        group = agreeing_group([_candidate("a", value=0.0), _candidate("b", value=0.0)])
        self.assertEqual(len(group), 2)


class TestTheRecord(unittest.TestCase):

    def test_the_summary_says_what_was_tried_and_why_one_won(self):
        candidates = [
            _candidate("primary", score=95, value=100.0),
            _candidate("variant_grain", score=60, value=200.0),
            _candidate("variant_join", validated=False,
                       validation_reason="no approved path"),
        ]
        chosen, reason = select(candidates)
        record = summarise(candidates, chosen, reason)
        self.assertEqual(record["candidates"], 3)
        self.assertEqual(record["usable"], 2)
        self.assertEqual(record["verified"], 1)
        self.assertEqual(record["chosen_source"], "primary")
        self.assertEqual(record["chosen_score"], 95)
        self.assertEqual(record["reason"], reason)
        self.assertEqual(len(record["scores"]), 3)

    def test_the_summary_records_a_refusal_too(self):
        candidates = [
            _candidate("a", score=90, value=100.0),
            _candidate("b", score=90, value=880.0),
        ]
        chosen, reason = select(candidates)
        record = summarise(candidates, chosen, reason)
        self.assertEqual(record["chosen_source"], "")
        self.assertEqual(record["reason"], "verified_candidates_disagree")

    def test_the_summary_carries_no_row_value(self):
        # It goes into the trace, which is durable. Scores, counts and errors
        # only — the rows themselves stay in the governed result.
        candidates = [_candidate("a", rows=[{"CUSTOMER": "Ospedale", "V": 1}])]
        chosen, reason = select(candidates)
        rendered = repr(summarise(candidates, chosen, reason))
        self.assertNotIn("Ospedale", rendered)


if __name__ == "__main__":
    unittest.main()


# ══════════════════════════════════════════════════════════════════════════════
# Against real compiled plans
# ══════════════════════════════════════════════════════════════════════════════

class TestAgainstRealPlans(unittest.TestCase):
    """``ambiguity_of`` must read what the compiler actually emits.

    The first version of this module read invented ``candidate_*`` keys, so on
    every real plan it returned "not ambiguous" -- a gate that is always shut
    is indistinguishable from one that is not wired at all. These tests build
    plans with the real compiler.
    """

    @staticmethod
    def _plan(semantic_plan, **kwargs):
        from core.analytical_request_plan import compile_analytical_request_plan
        return compile_analytical_request_plan(
            kwargs.pop("question", "revenue by month"),
            semantic_plan,
            matched_metrics=kwargs.pop("matched_metrics", []),
            analytical_intent_plan=kwargs.pop(
                "analytical_intent_plan", {"intent": "trend"}),
            **kwargs,
        )

    def test_the_compiler_records_the_facts_it_chose_between(self):
        plan = self._plan({
            "source_scope": {
                "selected_fact": "S.F_SALES",
                "selected_facts": ["S.F_SALES", "S.F_ORDERS"],
            },
        })
        self.assertEqual(plan["source_fact"], "S.F_SALES")
        self.assertEqual(plan["considered_facts"], ["S.F_ORDERS", "S.F_SALES"])

    def test_the_compiler_records_the_date_roles_it_chose_between(self):
        plan = self._plan({
            "source_scope": {"selected_fact": "S.F_SALES"},
            "temporal_policies": [
                {"date_role": "ORDER_DATE"}, {"date_role": "SHIP_DATE"}],
        })
        self.assertEqual(plan["considered_date_roles"], ["ORDER_DATE", "SHIP_DATE"])

    def test_a_plan_with_one_fact_and_one_date_role_is_not_ambiguous(self):
        plan = self._plan({
            "source_scope": {
                "selected_fact": "S.F_SALES", "selected_facts": ["S.F_SALES"]},
            "temporal_policies": [{"date_role": "ORDER_DATE"}],
        })
        self.assertEqual(ambiguity_of(plan), [])
        self.assertEqual(candidate_count_for(plan), 1)

    def test_a_plan_that_arbitrated_between_two_facts_is_ambiguous(self):
        plan = self._plan({
            "source_scope": {
                "selected_fact": "S.F_SALES",
                "selected_facts": ["S.F_SALES", "S.F_ORDERS"],
            },
        })
        self.assertIn("fact", ambiguity_of(plan))
        self.assertGreater(candidate_count_for(plan), 1)

    def test_two_open_decisions_are_both_named(self):
        plan = self._plan({
            "source_scope": {
                "selected_fact": "S.F_SALES",
                "selected_facts": ["S.F_SALES", "S.F_ORDERS"],
            },
            "temporal_policies": [
                {"date_role": "ORDER_DATE"}, {"date_role": "SHIP_DATE"}],
        })
        self.assertEqual(set(ambiguity_of(plan)), {"fact", "date_role"})
        self.assertEqual(candidate_count_for(plan), 3)

    def test_two_matched_metrics_count_as_a_choice(self):
        plan = self._plan(
            {"source_scope": {"selected_fact": "S.F_SALES"}},
            matched_metrics=[
                {"id": 1, "name": "Net Revenue", "base_table": "S.F_SALES"},
                {"id": 2, "name": "Gross Revenue", "base_table": "S.F_SALES"},
            ],
        )
        self.assertEqual(plan["considered_metrics"],
                         ["Gross Revenue", "Net Revenue"])
        self.assertIn("metric", ambiguity_of(plan))

    def test_an_empty_plan_does_not_raise(self):
        plan = self._plan({})
        self.assertEqual(ambiguity_of(plan), [])


class TestTheAmbiguityIsRecordedInTheTrace(unittest.TestCase):
    """The pipeline must compute the ambiguity, not merely be able to.

    A gate whose value nothing records is a gate nobody can evaluate, and the
    decision it exists to inform -- whether k-candidate generation earns its
    cost -- needs the rate first.
    """

    def test_the_pipeline_records_ambiguity_on_the_plan_step(self):
        import inspect

        import core.query_pipeline as qp

        source = inspect.getsource(qp._handle_query_impl)
        marker = "from core.candidate_selection import ambiguity_of"
        self.assertIn(marker, source)
        # The call must sit between the plan being compiled and the trace step
        # that reports it -- an ambiguity computed after the step is recorded
        # reaches nothing.
        compiled_at = source.index("_semantic_plan[\"analytical_request_plan\"]")
        computed_at = source.index(marker)
        traced_at = source.index('"analytical_request_plan",', computed_at)
        self.assertLess(compiled_at, computed_at)
        self.assertLess(computed_at, traced_at)
        self.assertIn('"ambiguity": _plan_ambiguity,', source)
