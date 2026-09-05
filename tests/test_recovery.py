# -*- coding: utf-8 -*-
"""Bounded recovery, as a trace reader sees it — core/recovery.py.

The pipeline already recovers within a budget of two: a repair that exposes a
DIFFERENT failure code gets one more attempt, and the same code twice is
treated as a non-progress loop. What was missing is how the run reads
afterwards — every attempt is written with the status it had at the time, so
a run that failed twice and then succeeded looks like two defects and an
answer, and an operator scanning traces finds problems that were solved.

Two properties are asserted hardest:

  * a failure AFTER the last success is never relabelled — that is the run's
    actual outcome, and tidying it away is the opposite of the point; and
  * the original status survives on every step, because the audit record has
    to say what happened and a report that rewrites failures into successes
    is worth less than no report.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.i18n import MESSAGES  # noqa: E402
from core.recovery import (  # noqa: E402
    ATTEMPT_STEPS,
    FAILED_STATUSES,
    MAX_REPLANS,
    SUPERSEDED,
    SUPERSEDABLE_STEPS,
    RecoverySummary,
    annotate_trace,
    describe,
    mark_superseded,
    summarise,
)


def _step(name, status="success"):
    return {"step_name": name, "status": status, "step_order": 0}


CLEAN = [_step("llm_generate_sql"), _step("validate_sql")]
REPAIRED = [
    _step("llm_generate_sql"),
    _step("validate_sql", "error"),
    _step("sql_repair"),
    _step("validate_sql"),
]
BROKEN = [_step("llm_generate_sql"), _step("validate_sql", "error")]


class TestSuperseding(unittest.TestCase):

    def test_a_failure_that_a_later_attempt_corrected_is_relabelled(self):
        marked = mark_superseded(REPAIRED)
        failed = marked[1]
        self.assertTrue(failed["superseded"])
        self.assertEqual(failed["status"], SUPERSEDED)

    def test_the_original_status_survives_on_every_step(self):
        # The audit record must say what happened. A report that quietly
        # rewrites failures into successes is worth less than no report.
        marked = mark_superseded(REPAIRED)
        self.assertEqual(marked[1]["original_status"], "error")
        self.assertEqual(marked[0]["original_status"], "success")

    def test_a_failure_after_the_last_success_stays_a_failure(self):
        # That is the run's actual outcome. Relabelling it would turn a
        # broken run into a tidy one.
        marked = mark_superseded(BROKEN)
        self.assertFalse(marked[1]["superseded"])
        self.assertEqual(marked[1]["status"], "error")

    def test_a_failure_after_a_success_that_is_then_final_stays_a_failure(self):
        steps = REPAIRED + [_step("validate_sql", "error")]
        marked = mark_superseded(steps)
        self.assertTrue(marked[1]["superseded"])
        self.assertFalse(marked[-1]["superseded"])

    def test_a_clean_run_is_left_alone(self):
        marked = mark_superseded(CLEAN)
        self.assertFalse(any(s["superseded"] for s in marked))
        self.assertEqual([s["status"] for s in marked], ["success", "success"])

    def test_steps_outside_the_query_path_are_never_relabelled(self):
        # A failed KB retrieval is not corrected by a later SQL success.
        steps = [
            _step("retrieve_kb", "error"),
            _step("llm_generate_sql"),
            _step("validate_sql"),
        ]
        marked = mark_superseded(steps)
        self.assertFalse(marked[0]["superseded"])
        self.assertEqual(marked[0]["status"], "error")

    def test_the_input_is_not_mutated(self):
        original = [dict(s) for s in REPAIRED]
        mark_superseded(REPAIRED)
        self.assertEqual(REPAIRED, original)

    def test_an_empty_trace_is_handled(self):
        self.assertEqual(mark_superseded([]), [])


class TestTheBudget(unittest.TestCase):

    def test_a_normal_repaired_run_is_within_budget(self):
        # The regression this guards: counting validate_sql as an attempt
        # made an ordinary one-repair run read as twice over budget, because
        # validation runs once per attempt.
        self.assertTrue(summarise(REPAIRED).within_budget)
        self.assertEqual(summarise(REPAIRED).attempts, 2)

    def test_validation_steps_do_not_count_as_attempts(self):
        self.assertNotIn("validate_sql", ATTEMPT_STEPS)
        self.assertIn("validate_sql", SUPERSEDABLE_STEPS)

    def test_a_run_beyond_the_budget_is_reported(self):
        steps = REPAIRED + [_step("sql_repair"), _step("sql_repair"),
                            _step("validate_sql")]
        summary = summarise(steps)
        self.assertGreater(summary.attempts, MAX_REPLANS + 1)
        self.assertFalse(summary.within_budget)

    def test_the_budget_here_matches_the_one_the_pipeline_enforces(self):
        # Stated in two places, so they are asserted equal: a reader of a
        # trace and the code that enforces the limit must not drift apart.
        import inspect

        import core.query_pipeline as qp

        source = inspect.getsource(qp._handle_query_impl)
        self.assertIn(f"max_attempts={MAX_REPLANS},", source)

    def test_every_attempt_step_can_also_be_superseded(self):
        self.assertTrue(ATTEMPT_STEPS <= SUPERSEDABLE_STEPS)


class TestTheSummary(unittest.TestCase):

    def test_a_clean_run_says_so(self):
        summary = summarise(CLEAN)
        self.assertTrue(summary.clean)
        self.assertEqual(summary.attempts, 1)
        self.assertEqual(summary.superseded, 0)

    def test_a_run_that_failed_outright_is_not_clean(self):
        # One attempt, nothing superseded -- and a failure nobody fixed.
        # Counting that as clean told the reader nothing about the one run
        # that most needed explaining.
        summary = summarise(BROKEN)
        self.assertEqual(summary.attempts, 1)
        self.assertEqual(summary.superseded, 0)
        self.assertFalse(summary.clean)

    def test_a_corrected_run_reports_recovery(self):
        summary = summarise(REPAIRED)
        self.assertTrue(summary.recovered)
        self.assertFalse(summary.clean)
        self.assertEqual(summary.unresolved, ())

    def test_an_uncorrected_failure_is_named(self):
        # The steps an operator should actually look at.
        summary = summarise(BROKEN)
        self.assertFalse(summary.recovered)
        self.assertEqual(summary.unresolved, ("validate_sql",))

    def test_a_run_with_both_reports_the_unresolved_one(self):
        steps = REPAIRED + [_step("validate_sql", "error")]
        summary = summarise(steps)
        self.assertFalse(summary.recovered)
        self.assertIn("validate_sql", summary.unresolved)

    def test_every_failed_status_counts(self):
        for status in FAILED_STATUSES:
            steps = [_step("llm_generate_sql"), _step("validate_sql", status)]
            self.assertTrue(summarise(steps).unresolved, status)


class TestWhatAReaderIsTold(unittest.TestCase):

    def test_a_clean_run_says_nothing(self):
        self.assertEqual(describe(summarise(CLEAN)), "")
        self.assertEqual(describe(RecoverySummary()), "")

    def test_a_corrected_run_says_it_was_corrected(self):
        line = describe(summarise(REPAIRED))
        self.assertIn("corrected", line)
        self.assertIn("2", line)

    def test_an_uncorrected_failure_names_the_step(self):
        line = describe(summarise(BROKEN))
        self.assertIn("validate_sql", line)
        self.assertIn("not corrected", line)

    def test_both_languages_say_it_differently(self):
        for steps in (REPAIRED, BROKEN):
            english = describe(summarise(steps), lang="en")
            french = describe(summarise(steps), lang="fr")
            self.assertTrue(english and french)
            self.assertNotEqual(english, french)

    def test_every_recovery_id_exists_in_both_languages(self):
        ids = [k for k in MESSAGES if k.startswith("recovery.")]
        self.assertGreaterEqual(len(ids), 3)
        for msg_id in ids:
            self.assertTrue(MESSAGES[msg_id].get("en"), msg_id)
            self.assertTrue(MESSAGES[msg_id].get("fr"), msg_id)
            self.assertNotEqual(MESSAGES[msg_id]["en"], MESSAGES[msg_id]["fr"])


class TestAnnotatingATrace(unittest.TestCase):

    def test_a_trace_comes_back_with_relabelled_steps_and_a_summary(self):
        annotated = annotate_trace({"id": 1, "steps": REPAIRED})
        self.assertTrue(annotated["steps"][1]["superseded"])
        self.assertTrue(annotated["recovery"]["recovered"])
        self.assertEqual(annotated["recovery"]["attempts"], 2)

    def test_a_trace_with_no_steps_is_returned_unharmed(self):
        self.assertEqual(annotate_trace({"id": 1})["recovery"]["attempts"], 0)

    def test_nothing_is_returned_for_nothing(self):
        self.assertIsNone(annotate_trace(None))

    def test_a_failure_costs_the_annotation_not_the_trace(self):
        # A trace view that 500s because of a presentation nicety is worse
        # than one showing raw statuses.
        from unittest.mock import patch

        import core.recovery as module

        with patch.object(module, "mark_superseded",
                          side_effect=RuntimeError("boom")):
            trace = annotate_trace({"id": 1, "steps": REPAIRED})
        self.assertEqual(trace["id"], 1)
        self.assertEqual(trace["steps"], REPAIRED)


class TestTheStoreAnnotatesOnRead(unittest.TestCase):
    """Write API to read API: a trace written with a failed attempt and a
    later success must come back out of the store already relabelled."""

    def setUp(self):
        import os
        import tempfile
        import uuid

        import store
        self._dir = tempfile.mkdtemp(prefix="qb-recovery-")
        self._saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(self._dir, "t.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        store.init_db()
        self.account_id = f"acct-rec-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

    def tearDown(self):
        import os
        import shutil

        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._dir, ignore_errors=True)

    def test_a_corrected_run_reads_as_a_correction(self):
        import store
        from store.trace_store import (
            create_answer_trace, get_answer_trace, log_answer_trace_step,
        )

        trace_id = create_answer_trace(
            account_id=self.account_id, question_id="q1",
            question_text="revenue by month", request_source="portal",
        )
        for name, status in (("llm_generate_sql", "success"),
                             ("validate_sql", "error"),
                             ("sql_repair", "success"),
                             ("validate_sql", "success")):
            log_answer_trace_step(trace_id, step_name=name, status=status)

        # Nothing handed in below this line.
        trace = get_answer_trace(trace_id)
        statuses = [s["status"] for s in trace["steps"]]
        self.assertIn(SUPERSEDED, statuses)
        self.assertEqual(trace["recovery"]["superseded"], 1)
        self.assertTrue(trace["recovery"]["recovered"])
        # And the audit record still says what happened.
        self.assertIn("error", [s["original_status"] for s in trace["steps"]])
        self.assertTrue(store.get_client(self.account_id))

    def test_a_run_that_never_recovered_still_reads_as_failed(self):
        from store.trace_store import (
            create_answer_trace, get_answer_trace, log_answer_trace_step,
        )

        trace_id = create_answer_trace(
            account_id=self.account_id, question_id="q2",
            question_text="revenue by month", request_source="portal",
        )
        for name, status in (("llm_generate_sql", "success"),
                             ("validate_sql", "error")):
            log_answer_trace_step(trace_id, step_name=name, status=status)

        trace = get_answer_trace(trace_id)
        self.assertNotIn(SUPERSEDED, [s["status"] for s in trace["steps"]])
        self.assertFalse(trace["recovery"]["recovered"])
        self.assertIn("validate_sql", trace["recovery"]["unresolved"])


if __name__ == "__main__":
    unittest.main()
