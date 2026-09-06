"""Asking the same question a second way — core/candidate_generation.py,
core.candidate_selection.verify_and_select, and their wiring.

The plan already recorded what the compiler chose BETWEEN. This is what turns
that into work: one extra generation per open decision, pinned to the
alternative that was discarded, and the verifier picking between the results.

Two properties decide whether this makes the product better or worse:

  * a determined plan generates exactly one query, so the common case costs
    what it always cost; and
  * two verified candidates returning different numbers produce no selection
    and a warning the user sees — the failure this whole mechanism exists to
    surface is an answer that looked confident and was a coin toss.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.candidate_generation import (  # noqa: E402
    PRIMARY,
    CandidateSpec,
    specs_for,
    variant_user_message,
)
from core.candidate_selection import candidate_count_for, verify_and_select  # noqa: E402
from core.sql_attempt import Attempt  # noqa: E402

AMBIGUOUS = {
    "source_fact": "S.F_SALES",
    "considered_facts": ["S.F_SALES", "S.F_ORDERS"],
    "temporal_operations": [{"date_role": "ORDER_DATE"}],
    "considered_date_roles": ["ORDER_DATE", "SHIP_DATE"],
}
DETERMINED = {
    "source_fact": "S.F_SALES",
    "considered_facts": ["S.F_SALES"],
    "temporal_operations": [{"date_role": "ORDER_DATE"}],
    "considered_date_roles": ["ORDER_DATE"],
}


def _attempt(source, *, rows, ok=True, error=""):
    return Attempt(sql=f"SELECT /* {source} */ 1", source=source, ok=ok,
                   rows=rows, exec_error=error or None)


# ══════════════════════════════════════════════════════════════════════════════
# What gets generated
# ══════════════════════════════════════════════════════════════════════════════

class TestSpecs(unittest.TestCase):

    def test_a_determined_plan_generates_exactly_one_query(self):
        # The common case, and the reason this does not multiply the cost of
        # the product.
        specs = specs_for(DETERMINED, limit=candidate_count_for(DETERMINED))
        self.assertEqual(specs, [PRIMARY])
        self.assertTrue(specs[0].is_primary)

    def test_an_empty_plan_generates_one_query(self):
        self.assertEqual(specs_for({}), [PRIMARY])
        self.assertEqual(specs_for(None), [PRIMARY])

    def test_each_discarded_alternative_becomes_a_variant(self):
        specs = specs_for(AMBIGUOUS, limit=candidate_count_for(AMBIGUOUS))
        self.assertEqual(specs[0], PRIMARY)
        self.assertEqual(
            {s.choice for s in specs[1:]}, {"SHIP_DATE", "S.F_ORDERS"})

    def test_the_chosen_value_is_never_regenerated_as_a_variant(self):
        # The primary already asked that way. A variant pinned to the same
        # choice is a second identical query that always "agrees".
        specs = specs_for(AMBIGUOUS, limit=5)
        self.assertNotIn("ORDER_DATE", [s.choice for s in specs[1:]])
        self.assertNotIn("S.F_SALES", [s.choice for s in specs[1:]])

    def test_the_date_role_is_varied_before_the_fact(self):
        # A different business date changes the answer more often than a
        # different source table, so it is the first alternative spent on.
        specs = specs_for(AMBIGUOUS, limit=2)
        self.assertEqual(specs[1].dimension, "date_role")

    def test_duplicates_and_blanks_do_not_become_variants(self):
        specs = specs_for({
            "source_fact": "S.F",
            "considered_facts": ["S.F", "S.F", "  ", "s.f", "S.OTHER"],
        }, limit=5)
        self.assertEqual([s.choice for s in specs[1:]], ["S.OTHER"])

    def test_the_number_of_variants_respects_the_limit(self):
        self.assertEqual(len(specs_for(AMBIGUOUS, limit=2)), 2)
        self.assertEqual(len(specs_for(AMBIGUOUS, limit=1)), 1)

    def test_a_variant_names_what_it_is_testing_for_the_trace(self):
        spec = specs_for(AMBIGUOUS, limit=2)[1]
        self.assertIn("date_role", spec.source)
        self.assertIn("SHIP_DATE", spec.source)


class TestTheVariantPrompt(unittest.TestCase):

    def test_the_primary_asks_the_question_unchanged(self):
        self.assertEqual(variant_user_message("revenue by month", PRIMARY),
                         "revenue by month")

    def test_a_variant_is_the_same_question_plus_one_constraint(self):
        # A variant built from a different prompt would differ in ways the
        # verifier cannot attribute, and the whole basis for comparing two
        # candidates is that the only difference is the decision under test.
        spec = specs_for(AMBIGUOUS, limit=2)[1]
        message = variant_user_message("revenue by month", spec)
        self.assertTrue(message.startswith("revenue by month"))
        self.assertIn("SHIP_DATE", message)

    def test_the_directive_tells_the_model_what_to_pin(self):
        for spec in specs_for(AMBIGUOUS, limit=3)[1:]:
            self.assertIn("OVERRIDE", spec.directive)
            self.assertIn(spec.choice, spec.directive)

    def test_a_metric_alternative_is_offered_too(self):
        plan = {
            "metrics": [{"name": "Net Revenue"}],
            "considered_metrics": ["Net Revenue", "Gross Revenue"],
        }
        specs = specs_for(plan, limit=3)
        self.assertEqual([s.choice for s in specs[1:]], ["Gross Revenue"])
        self.assertIn("METRIC OVERRIDE", specs[1].directive)


# ══════════════════════════════════════════════════════════════════════════════
# Choosing between what came back
# ══════════════════════════════════════════════════════════════════════════════

class TestVerifyAndSelect(unittest.TestCase):

    def test_two_agreeing_candidates_produce_an_answer(self):
        rows = [{"MONTH": "2026-01", "REVENUE": 1000.0}]
        chosen, record = verify_and_select([
            _attempt("primary", rows=rows),
            _attempt("variant:date_role=SHIP_DATE", rows=list(rows)),
        ])
        self.assertIsNotNone(chosen)
        self.assertEqual(record["candidates"], 2)
        self.assertIn("agreement", record["reason"])

    def test_two_disagreeing_candidates_produce_no_selection(self):
        # The failure this whole mechanism exists to surface.
        chosen, record = verify_and_select([
            _attempt("primary", rows=[{"MONTH": "2026-01", "REVENUE": 1000.0}]),
            _attempt("variant:date_role=SHIP_DATE",
                     rows=[{"MONTH": "2026-01", "REVENUE": 8800.0}]),
        ])
        self.assertEqual(record["reason"], "verified_candidates_disagree")
        self.assertEqual(record["chosen_source"], "")
        # The primary still comes back, so the pipeline's existing failure
        # path has something to explain -- but the record says it was not
        # agreed, and the caller has to surface that.
        self.assertEqual(chosen.source, "primary")

    def test_a_variant_that_failed_never_outvotes_a_working_primary(self):
        chosen, record = verify_and_select([
            _attempt("primary", rows=[{"MONTH": "2026-01", "REVENUE": 1000.0}]),
            _attempt("variant:fact=S.F_ORDERS", rows=None, error="ORA-00942"),
        ])
        self.assertEqual(chosen.source, "primary")
        self.assertEqual(record["usable"], 1)

    def test_an_unexecuted_attempt_is_not_verified(self):
        # verify_result_shape is never handed rows that do not exist.
        with patch("core.result_verifier.verify_result_shape") as verify:
            verify_and_select([_attempt("primary", rows=None, ok=False)])
        verify.assert_not_called()

    def test_a_verifier_failure_costs_that_candidate_its_score_not_the_run(self):
        with (
            patch("core.result_verifier.verify_result_shape",
                  side_effect=RuntimeError("verifier exploded")),
            self.assertLogs("querybot.candidate_selection", level="WARNING"),
        ):
            chosen, record = verify_and_select([
                _attempt("primary", rows=[{"A": 1}]),
            ])
        self.assertIsNotNone(chosen)
        self.assertEqual(record["scores"][0]["score"], 0)

    def test_the_plans_reach_the_verifier(self):
        with patch("core.result_verifier.verify_result_shape",
                   return_value={"status": "pass", "score": 90}) as verify:
            verify_and_select(
                [_attempt("primary", rows=[{"A": 1}])],
                analytical_plan={"intent": "trend"},
                resolution_plan={"metrics": [{"name": "Net Revenue"}]},
                request_plan={"status": "compiled"},
            )
        kwargs = verify.call_args.kwargs
        self.assertEqual(kwargs["analytical_plan"], {"intent": "trend"})
        self.assertEqual(kwargs["resolution_plan"], {"metrics": [{"name": "Net Revenue"}]})
        self.assertEqual(kwargs["request_plan"], {"status": "compiled"})

    def test_the_returned_attempt_is_the_one_that_was_chosen(self):
        # Not a Candidate copy: the pipeline reads rows, sql, truncation and
        # timing off the Attempt, and handing back the wrong object would
        # answer with one query's rows under another query's SQL.
        primary = _attempt("primary", rows=[{"A": 1.0}])
        better = _attempt("variant:fact=S.F_ORDERS", rows=[{"A": 1.0}])
        with patch("core.result_verifier.verify_result_shape",
                   side_effect=[{"status": "warning", "score": 40},
                                {"status": "pass", "score": 95}]):
            chosen, _record = verify_and_select([primary, better])
        self.assertIs(chosen, better)

    def test_no_attempts_at_all_is_reported_not_crashed(self):
        chosen, record = verify_and_select([])
        self.assertIsNone(chosen)
        self.assertEqual(record["reason"], "no_candidates")


# ══════════════════════════════════════════════════════════════════════════════
# What the user is told
# ══════════════════════════════════════════════════════════════════════════════

class TestConfidenceReflectsTheSelection(unittest.TestCase):

    def _confidence(self, reason, **overrides):
        from core.answer_confidence import build_answer_confidence
        fields = dict(validation_code="ok", row_count=10, has_semantic_plan=True)
        fields.update(overrides)
        return build_answer_confidence(
            candidate_selection={"reason": reason}, **fields)

    def test_a_disagreement_drops_the_answer_below_medium_and_warns(self):
        # An answer that was a coin toss must not be presented at the same
        # confidence as one nothing disputed.
        confident = self._confidence("")
        disputed = self._confidence("verified_candidates_disagree")
        self.assertLess(disputed["score"], confident["score"])
        self.assertLessEqual(disputed["score"], 49)
        self.assertTrue(any("second way" in w for w in disputed["warnings"]))

    def test_the_warning_says_what_to_check(self):
        warnings = self._confidence("verified_candidates_disagree")["warnings"]
        joined = " ".join(warnings)
        self.assertIn("business", joined)
        self.assertIn("date", joined)

    def test_nothing_verifying_is_a_smaller_penalty_than_a_disagreement(self):
        # Two answers that disagree is worse evidence than none matching.
        unverified = self._confidence("no_candidate_verified")
        disputed = self._confidence("verified_candidates_disagree")
        self.assertLess(disputed["score"], unverified["score"])

    def test_agreement_raises_confidence(self):
        # Scored from a base that is not already at the ceiling -- a clean
        # answer caps at 100, where a +5 is invisible and the assertion would
        # pass whether or not agreement counted for anything.
        agreed = self._confidence("agreement_of_2_candidates", weak_retrieval=True)
        plain = self._confidence("", weak_retrieval=True)
        self.assertGreater(agreed["score"], plain["score"])
        self.assertTrue(any("same figure" in r for r in agreed["reasons"]))

    def test_a_single_candidate_changes_nothing(self):
        self.assertEqual(self._confidence("single_candidate_unverified")["score"],
                         self._confidence("")["score"])


class TestThePipelineRunsIt(unittest.TestCase):
    """The loop has to be reached, or none of the above matters."""

    @staticmethod
    def _source():
        import inspect

        import core.query_pipeline as qp
        return inspect.getsource(qp._handle_query_impl)

    def test_variants_are_generated_and_attempted_before_the_verdict(self):
        source = self._source()
        generated = source.index("_variant_sql = await _generate_variant_sql(_spec)")
        attempted = source.index("_candidate_attempts.append(await _run_one(")
        chosen = source.index("_attempt, _candidate_selection = _choose_candidate(")
        adopted = source.index("sql, ok, reason, code = _attempt.sql")
        self.assertLess(generated, attempted)
        self.assertLess(attempted, chosen)
        self.assertLess(chosen, adopted)

    def test_a_reused_or_compiled_plan_is_never_second_guessed(self):
        # Neither is a guess, so there is nothing to spend the extra
        # generation on.
        self.assertIn("if not (_reused_plan or _compiled_governed_sql):", self._source())

    def test_the_selection_reaches_answer_confidence(self):
        source = self._source()
        self.assertIn('_confidence_context["candidate_selection"] = _candidate_selection',
                      source)

    def test_the_renderer_reads_what_the_pipeline_wrote(self):
        """Write API to read API, on the identity key that spans them.

        The pipeline writes confidence_context["candidate_selection"] and the
        renderer has to read that exact key. A mismatch here is the quietest
        bug in this codebase: every lookup misses, no exception fires, and the
        answer is presented at full confidence with the disagreement dropped.
        """
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        import core.result_renderer as rr

        adapter = MagicMock()
        adapter.send_message = AsyncMock()
        adapter.send_result = AsyncMock()
        adapter.cache_result = None
        record = {"reason": "verified_candidates_disagree", "chosen_source": ""}

        with patch.object(rr, "build_answer_confidence",
                          wraps=rr.build_answer_confidence) as confidence:
            try:
                asyncio.run(rr._send_results(
                    MagicMock(), adapter, "revenue by month",
                    [{"MONTH": "2026-01", "REVENUE": 1000.0}],
                    "SELECT 1", 12, None, "acct",
                    {"db_type": "azure_sql", "credentials": {}},
                    confidence_context={"candidate_selection": record},
                    cache_result=False,
                ))
            except Exception:
                # The renderer does far more than build confidence; what it
                # does after is not this test's business, and it has already
                # happened by the time anything else can fail.
                pass

        confidence.assert_called()
        self.assertEqual(
            confidence.call_args.kwargs.get("candidate_selection"), record)

    def test_a_variant_gets_the_same_cleanup_as_the_primary(self):
        # Two candidates that differ in fence stripping, the DISTINCT safety
        # net or dialect normalisation differ in ways the verifier cannot
        # attribute.
        source = self._source()
        self.assertEqual(source.count("clean_generated_sql("), 2)


if __name__ == "__main__":
    unittest.main()
