"""
tests/test_execution_correction.py

B3 — one correction pass, guided by what actually happened when the SQL ran.

Two things the product could already see and did not act on. The database's
own diagnosis of a failure: `sanitize_db_error` has an ordered matcher that
reads "Invalid object name" as "a table this query needs does not exist", and
the repair prompt was handed the driver's sentence instead. And the verifier's
complaint: `core.result_verifier` knew a "trend by month" came back with one
row and no date column, but its report was computed at the very end of the
turn, to score confidence — a footnote on an answer nobody corrected.

The property that makes this safe to ship is that a correction is
NON-DESTRUCTIVE. A shape complaint means the first attempt returned usable
rows; a correction that fails must not turn a slightly-wrong answer into no
answer. Most of this file is about that.

`run_correction` takes its four dependencies as arguments so it can be
executed here. That is not test scaffolding — it is the same move candidate
selection made, and for the same reason: a 6,400-line function cannot be
called from a test, and the checks on code inside one decay into source scans
that never run the thing they are named for.
"""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass, field

from core.execution_correction import (
    MAX_COMPLAINTS,
    CorrectionDecision,
    correction_prompt,
    diagnose_execution_error,
    is_improvement,
    needs_shape_correction,
    run_correction,
)
from core.result_verifier import verify_result_shape


@dataclass
class FakeAttempt:
    """Shaped like core.sql_attempt.Attempt, which is what the caller passes."""

    sql: str = "SELECT 1"
    source: str = "primary"
    ok: bool = True
    reason: str = "OK"
    code: str = "ok"
    rows: list | None = field(default_factory=lambda: [{"revenue": 100}])
    exec_error: str | None = None
    truncated: bool = False
    policy_denied: object = None


# Real verifier reports, produced by the real verifier, so a change to what it
# complains about shows up here rather than being papered over by a dict I
# typed. A trend question answered with one row and no date column is the
# canonical shape failure.
#
# The trend checks read the ANALYTICAL plan, not the compiled request plan.
# A first draft of this file passed the request plan, and every fixture below
# verified clean — which made every "the correction fired" test pass for the
# wrong reason and every "it did not fire" test pass for no reason at all.
# That is what TestTheFixtureIsRealBeforeAnythingElse is for.
TREND_PLAN = {"intent": "trend"}


def report_for(rows, plan=TREND_PLAN):
    return verify_result_shape(rows, analytical_plan=plan)


ONE_ROW = [{"revenue": 100}]
GOOD_TREND = [{"month": "2024-01", "revenue": 100},
              {"month": "2024-02", "revenue": 120},
              {"month": "2024-03", "revenue": 140}]


class TestTheFixtureIsRealBeforeAnythingElse(unittest.TestCase):
    """If these two are not what they claim, every test below proves nothing."""

    def test_the_bad_shape_really_does_fail_verification(self):
        errors = report_for(ONE_ROW).get("errors") or []
        self.assertTrue(errors, "the 'bad' fixture verifies clean")

    def test_the_good_shape_really_does_pass(self):
        self.assertEqual(report_for(GOOD_TREND).get("errors") or [], [])


class TestWhenAPassIsWorthSpending(unittest.TestCase):

    def test_a_shape_the_verifier_objects_to_is_corrected(self):
        decision = needs_shape_correction(
            ok=True, exec_error=None, rows=ONE_ROW, verification=report_for(ONE_ROW))
        self.assertTrue(decision.should_correct)
        self.assertEqual(decision.kind, "result_shape")
        self.assertTrue(decision.complaints)

    def test_a_shape_the_verifier_accepts_is_left_alone(self):
        decision = needs_shape_correction(
            ok=True, exec_error=None, rows=GOOD_TREND,
            verification=report_for(GOOD_TREND))
        self.assertFalse(decision.should_correct)

    def test_warnings_alone_are_not_grounds(self):
        # The verifier separates them from errors deliberately: "no column is
        # labelled with the approved metric name" is a note for the reader,
        # not a defect worth a model call and a second execution.
        report = {"status": "ok", "score": 80, "errors": [],
                  "warnings": ["Numeric output was returned, but no column is "
                               "labelled with the approved metric name."]}
        self.assertFalse(needs_shape_correction(
            ok=True, exec_error=None, rows=ONE_ROW, verification=report).should_correct)

    def test_a_failed_attempt_is_the_repair_ladders_business(self):
        # Stacking a correction on an attempt that produced nothing is how one
        # pass becomes four.
        report = report_for(ONE_ROW)
        for kwargs in ({"ok": False}, {"exec_error": "boom"}, {"rows": []},
                       {"rows": None}):
            with self.subTest(**kwargs):
                base = {"ok": True, "exec_error": None, "rows": ONE_ROW,
                        "verification": report}
                base.update(kwargs)
                self.assertFalse(needs_shape_correction(**base).should_correct)

    def test_an_empty_result_is_left_to_the_zero_row_paths(self):
        # Those know things this does not, such as whether a date filter was
        # involved, and they already retry.
        empty_report = verify_result_shape([], analytical_plan=TREND_PLAN)
        self.assertEqual(empty_report["status"], "empty")
        self.assertFalse(needs_shape_correction(
            ok=True, exec_error=None, rows=[{"a": 1}],
            verification=empty_report).should_correct)

    def test_one_pass_only(self):
        self.assertFalse(needs_shape_correction(
            ok=True, exec_error=None, rows=ONE_ROW,
            verification=report_for(ONE_ROW), already_corrected=True).should_correct)

    def test_the_complaint_list_is_bounded(self):
        report = {"status": "ok", "score": 10,
                  "errors": [f"complaint {i}" for i in range(20)]}
        decision = needs_shape_correction(
            ok=True, exec_error=None, rows=ONE_ROW, verification=report)
        self.assertEqual(len(decision.complaints), MAX_COMPLAINTS)

    def test_duplicate_complaints_are_collapsed(self):
        report = {"status": "ok", "score": 10, "errors": ["same", "same", "other"]}
        self.assertEqual(
            needs_shape_correction(ok=True, exec_error=None, rows=ONE_ROW,
                                   verification=report).complaints,
            ("same", "other"))


class TestTheDatabasesOwnDiagnosis(unittest.TestCase):

    def test_a_known_error_carries_its_reading_and_its_remedy(self):
        decision = diagnose_execution_error(
            "[Microsoft][ODBC Driver 17] Invalid object name 'SALES.ORDERZ'.")
        self.assertTrue(decision.should_correct)
        self.assertTrue(decision.diagnosis)
        self.assertTrue(decision.next_step)
        # The reading, not the driver's sentence.
        self.assertNotIn("ODBC", decision.diagnosis)

    def test_an_unknown_error_still_produces_something_to_say(self):
        decision = diagnose_execution_error("something nobody has a matcher for")
        self.assertTrue(decision.should_correct)
        self.assertTrue(decision.diagnosis)

    def test_no_error_is_no_decision(self):
        for value in ("", "   ", None):
            with self.subTest(value=value):
                self.assertFalse(diagnose_execution_error(value).should_correct)


class TestWhatTheModelIsAsked(unittest.TestCase):

    def _prompt(self):
        decision = needs_shape_correction(
            ok=True, exec_error=None, rows=ONE_ROW, verification=report_for(ONE_ROW))
        return correction_prompt("revenue trend by month",
                                 "SELECT SUM(revenue) FROM sales", decision)

    def test_it_carries_the_question_the_sql_and_the_complaint(self):
        prompt = self._prompt()
        self.assertIn("revenue trend by month", prompt)
        self.assertIn("SELECT SUM(revenue) FROM sales", prompt)
        self.assertIn("trend", prompt.casefold())

    def test_it_asks_for_a_minimal_change(self):
        # A repair that rewrites the measure or the date range to satisfy a
        # shape complaint answers a different question.
        prompt = self._prompt()
        self.assertIn("Change only what", prompt)
        self.assertIn("same measure", prompt)

    def test_it_returns_sql_and_not_prose(self):
        self.assertIn("Return only the corrected SQL", self._prompt())

    def test_an_execution_failure_says_so_rather_than_talking_about_shape(self):
        decision = diagnose_execution_error("Invalid object name 'X'.")
        prompt = correction_prompt("q", "SELECT 1", decision,
                                   scrubbed_error="Invalid object name '[value]'.")
        self.assertIn("failed when the database ran it", prompt)
        self.assertIn("Invalid object name '[value]'.", prompt)
        self.assertIn(decision.diagnosis, prompt)
        self.assertIn(decision.next_step, prompt)

    def test_no_row_value_reaches_the_prompt(self):
        # The verifier states shapes and counts, never data. A prompt built
        # from its report must not become the one place a value escapes.
        rows = [{"customer": "Priya Raghunathan", "revenue": 41234.5}]
        decision = needs_shape_correction(
            ok=True, exec_error=None, rows=rows, verification=report_for(rows))
        prompt = correction_prompt("q", "SELECT 1", decision)
        self.assertNotIn("Priya", prompt)
        self.assertNotIn("41234", prompt)


class TestOnlyAStrictlyBetterResultReplacesTheAnswer(unittest.TestCase):

    BAD = {"errors": ["a", "b"], "score": 40}
    BETTER = {"errors": ["a"], "score": 60}
    CLEAN = {"errors": [], "score": 90}

    def test_fewer_failures_and_no_regression_is_adopted(self):
        adopted, why = is_improvement(self.BAD, self.CLEAN, after_rows=ONE_ROW)
        self.assertTrue(adopted)
        self.assertIn("resolved", why)

    def test_partly_better_is_still_better(self):
        self.assertTrue(is_improvement(self.BAD, self.BETTER, after_rows=ONE_ROW)[0])

    def test_the_same_failures_are_not_an_improvement(self):
        adopted, why = is_improvement(self.BAD, dict(self.BAD), after_rows=ONE_ROW)
        self.assertFalse(adopted)
        self.assertIn("did not resolve", why)

    def test_new_failures_are_refused(self):
        adopted, why = is_improvement(self.BETTER, self.BAD, after_rows=ONE_ROW)
        self.assertFalse(adopted)
        self.assertIn("introduced", why)

    def test_no_rows_is_refused_however_clean_the_report(self):
        # An empty result verifies as "empty", which has no shape failures at
        # all -- so a score-and-error comparison alone would happily replace a
        # real answer with nothing.
        adopted, why = is_improvement(self.BAD, self.CLEAN, after_rows=[])
        self.assertFalse(adopted)
        self.assertIn("no rows", why)

    def test_a_fix_that_costs_more_elsewhere_is_refused(self):
        adopted, why = is_improvement({"errors": ["a", "b"], "score": 80},
                                      {"errors": ["a"], "score": 30},
                                      after_rows=ONE_ROW)
        self.assertFalse(adopted)
        self.assertIn("cost more", why)

    def test_score_alone_never_decides(self):
        # The first version of this used the score, and a correction could
        # raise it while leaving every complaint standing.
        self.assertFalse(is_improvement({"errors": ["a"], "score": 10},
                                        {"errors": ["a"], "score": 99},
                                        after_rows=ONE_ROW)[0])


class TestTheWholePassEndToEnd(unittest.IsolatedAsyncioTestCase):
    """run_correction, executed, with real verifier reports on both sides."""

    def setUp(self):
        self.generated = []
        self.executed = []
        self.traced = []

    def _verify(self, rows):
        return report_for(rows)

    async def _generate(self, prompt):
        self.generated.append(prompt)
        return "SELECT month, SUM(revenue) revenue FROM sales GROUP BY month"

    def _run(self, attempt, generate=None, execute=None, **kwargs):
        async def _execute(sql):
            self.executed.append(sql)
            return FakeAttempt(sql=sql, rows=GOOD_TREND, source="shape_correction")

        return asyncio.run(run_correction(
            attempt, question="revenue trend by month",
            verify=self._verify,
            generate=generate or self._generate,
            execute=execute or _execute,
            on_trace=self.traced.append,
            **kwargs))

    def test_a_wrong_shape_is_corrected_and_adopted(self):
        used, record = self._run(FakeAttempt(rows=ONE_ROW))
        self.assertEqual(used.rows, GOOD_TREND)
        self.assertEqual(used.source, "shape_correction")
        self.assertTrue(record["adopted"])
        self.assertEqual(record["errors_after"], 0)
        self.assertLess(record["errors_after"], record["errors_before"])

    def test_the_question_reaches_the_prompt(self):
        # Attempt carries the SQL and what happened to it, not what was asked.
        # Reading the question off it would have sent an empty question in a
        # prompt whose whole job is "make this answer the question".
        self._run(FakeAttempt(rows=ONE_ROW))
        self.assertIn("revenue trend by month", self.generated[0])

    def test_a_right_shape_costs_nothing(self):
        used, record = self._run(FakeAttempt(rows=GOOD_TREND))
        self.assertEqual(record, {})
        self.assertEqual(self.generated, [])
        self.assertEqual(self.executed, [])

    def test_a_correction_that_does_not_help_is_discarded(self):
        original = FakeAttempt(rows=ONE_ROW)

        async def _execute(sql):
            return FakeAttempt(sql=sql, rows=[{"revenue": 999}])

        used, record = self._run(original, execute=_execute)
        self.assertIs(used, original)
        self.assertFalse(record["adopted"])
        self.assertIn("did not resolve", record["reason"])

    def test_a_correction_that_returns_nothing_is_discarded(self):
        original = FakeAttempt(rows=ONE_ROW)

        async def _execute(sql):
            return FakeAttempt(sql=sql, rows=[])

        used, record = self._run(original, execute=_execute)
        self.assertIs(used, original)
        self.assertFalse(record["adopted"])

    def test_a_correction_that_fails_to_validate_is_discarded(self):
        original = FakeAttempt(rows=ONE_ROW)

        async def _execute(sql):
            return FakeAttempt(sql=sql, ok=False, reason="unknown column",
                               code="unknown_column", rows=None)

        used, record = self._run(original, execute=_execute)
        self.assertIs(used, original)
        self.assertEqual(used.rows, ONE_ROW)
        self.assertFalse(record["adopted"])

    def test_a_generation_failure_costs_nothing_but_is_recorded(self):
        original = FakeAttempt(rows=ONE_ROW)

        async def _generate(prompt):
            raise RuntimeError("provider down")

        used, record = self._run(original, generate=_generate)
        self.assertIs(used, original)
        self.assertFalse(record["attempted"])
        self.assertEqual(record["reason"], "generation_failed")
        self.assertEqual(self.executed, [])

    def test_a_refusal_to_generate_costs_nothing(self):
        original = FakeAttempt(rows=ONE_ROW)

        async def _generate(prompt):
            return ""

        used, record = self._run(original, generate=_generate)
        self.assertIs(used, original)
        self.assertEqual(record["reason"], "no_usable_sql")
        self.assertEqual(self.executed, [])

    def test_an_execution_that_raises_costs_nothing(self):
        original = FakeAttempt(rows=ONE_ROW)

        async def _execute(sql):
            raise RuntimeError("executor exploded")

        used, record = self._run(original, execute=_execute)
        self.assertIs(used, original)
        self.assertEqual(record["reason"], "execution_failed")

    def test_a_policy_denied_attempt_is_never_corrected(self):
        # A refusal is the answer. Rewriting until it stops being refused is
        # the shape of getting around governance rather than working with it.
        denied = FakeAttempt(rows=ONE_ROW, policy_denied=object())
        used, record = self._run(denied)
        self.assertIs(used, denied)
        self.assertEqual(record, {})
        self.assertEqual(self.generated, [])

    def test_exactly_one_generation_and_one_execution(self):
        self._run(FakeAttempt(rows=ONE_ROW))
        self.assertEqual(len(self.generated), 1)
        self.assertEqual(len(self.executed), 1)

    def test_a_second_pass_is_refused(self):
        used, record = self._run(FakeAttempt(rows=ONE_ROW), already_corrected=True)
        self.assertEqual(record, {})
        self.assertEqual(self.generated, [])

    def test_the_decision_is_traced_either_way(self):
        self._run(FakeAttempt(rows=ONE_ROW))
        self.assertEqual(len(self.traced), 1)
        self.assertTrue(self.traced[0]["adopted"])

        self.traced.clear()

        async def _execute(sql):
            return FakeAttempt(sql=sql, rows=[{"revenue": 1}])

        self._run(FakeAttempt(rows=ONE_ROW), execute=_execute)
        self.assertEqual(len(self.traced), 1)
        self.assertFalse(self.traced[0]["adopted"])
        self.assertTrue(self.traced[0]["complaints"])

    def test_a_verifier_that_explodes_does_not_cost_the_answer(self):
        original = FakeAttempt(rows=ONE_ROW)

        def _verify(rows):
            raise RuntimeError("verifier exploded")

        used, record = asyncio.run(run_correction(
            original, question="q", verify=_verify,
            generate=self._generate,
            execute=lambda sql: None,
            on_trace=self.traced.append))
        self.assertIs(used, original)
        self.assertEqual(record, {})

    def test_a_trace_callback_that_explodes_does_not_cost_the_answer(self):
        def _boom(record):
            raise RuntimeError("trace exploded")

        used, _ = asyncio.run(run_correction(
            FakeAttempt(rows=ONE_ROW), question="q", verify=self._verify,
            generate=self._generate,
            execute=lambda sql: _as_coro(FakeAttempt(sql=sql, rows=GOOD_TREND)),
            on_trace=_boom))
        self.assertEqual(used.rows, GOOD_TREND)


async def _as_coro(value):
    return value


class TestItIsWiredIntoThePipeline(unittest.TestCase):
    """
    The wrapper exists, is called, and hands over what run_correction needs.

    Deliberately a source check and the only one here: everything above
    executes. What it covers is placement inside a 6,400-line function, which
    is exactly where an insertion lands in the wrong body — this branch has
    already shipped a fix that sat between an `except` and a `return`.
    """

    def _source(self):
        import inspect

        import core.query_pipeline as pipeline
        return inspect.getsource(pipeline._handle_query_impl)

    def test_the_wrapper_is_inside_the_pipeline_function(self):
        source = self._source()
        self.assertIn("async def _correct_result_shape(attempt):", source)
        self.assertIn("from core.execution_correction import run_correction", source)

    def test_the_wrapper_is_actually_called(self):
        self.assertIn("_attempt, _shape_correction = await _correct_result_shape",
                      self._source())

    def test_the_question_is_handed_over(self):
        # The bug this exists for: run_correction reads the question from its
        # argument, and an Attempt has no question on it.
        #
        # Asserted with the line that follows it. "question=question," alone
        # appears elsewhere in this function, so the bare substring is
        # satisfied by an unrelated call and says nothing about this one.
        self.assertIn(
            "            question=question,\n"
            "            verify=lambda rows: verify_result_shape(",
            self._source())

    def test_the_correction_runs_through_the_governed_executor(self):
        # Not a direct execute: the same _run_one every other candidate uses,
        # which validates, repairs and then calls execute_governed_query.
        self.assertIn('execute=lambda candidate: _run_one(candidate, "shape_correction")',
                      self._source())

    def test_the_result_reaches_the_confidence_context(self):
        source = self._source()
        self.assertIn('_confidence_context["shape_correction"] = _shape_correction',
                      source)

    def test_the_execution_error_prompt_carries_the_diagnosis(self):
        source = self._source()
        self.assertIn("diagnose_execution_error(exec_error)", source)
        # In the prompt, not merely assigned above it. Building the note and
        # then not interpolating it is a whole feature that costs a matcher
        # lookup per failure and changes nothing the model sees.
        self.assertIn('f"Error: {scrub_error_for_llm(exec_error)}\\n"\n'
                      '                f"{_diagnosis_note}"', source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
