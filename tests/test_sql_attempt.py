"""One try at answering a question — core/sql_attempt.py.

This sequence was inline in a 4,000-line function, which is why the tests that
covered it were source scans: it could not be called. Every test here calls it,
with the validator, the repairs and the warehouse mocked at their boundaries
and nothing else.

The properties asserted hardest are the ones the pipeline reads back as state:
a timeout is not an execution error (a repair cannot make the database faster),
a policy denial is not a failure of the SQL, and a repair that produces
differently-invalid SQL is not progress.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.sql_attempt import (  # noqa: E402
    Attempt,
    ValidationScope,
    execute,
    run_attempt,
    validate_with_repairs,
)

SCOPE = ValidationScope(
    known_tables={"S.F", "S.D"}, db_type="azure_sql",
    allowed_tables={"S.F", "S.D"},
    table_columns={"S.F": {"AMT": "decimal"}, "S.D": {"NAME": "varchar"}},
)


def _rows(rows, sql="SELECT 1", truncated=False):
    return SimpleNamespace(rows=rows, sql=sql, truncated=truncated)


class TestValidation(unittest.TestCase):

    def test_valid_sql_comes_back_valid_with_no_repairs(self):
        with patch("core.validator.validate_sql", return_value=(True, "OK", "ok")):
            attempt = validate_with_repairs("SELECT 1", SCOPE)
        self.assertTrue(attempt.ok)
        self.assertEqual(attempt.code, "ok")
        self.assertEqual(attempt.repairs, ())
        self.assertEqual(attempt.sql, "SELECT 1")

    def test_an_unrepairable_failure_is_reported_with_its_code(self):
        with patch("core.validator.validate_sql",
                   return_value=(False, "no approved join", "graph_plan_mismatch")):
            with patch("core.pipeline_helpers."
                       "attempt_governed_temporal_metric_repair", return_value=""):
                attempt = validate_with_repairs("SELECT 1", SCOPE)
        self.assertFalse(attempt.ok)
        self.assertEqual(attempt.code, "graph_plan_mismatch")
        self.assertEqual(attempt.reason, "no approved join")

    def test_a_temporal_repair_is_tried_before_a_field_plan_one(self):
        # Ordering matters: a governed period comparison recompiled from an
        # approved metric and date role is a better fix than swapping a
        # display field, and both trigger on field_plan_mismatch.
        order = []
        with (
            patch("core.validator.validate_sql",
                  return_value=(False, "x", "field_plan_mismatch")),
            patch("core.pipeline_helpers.attempt_governed_temporal_metric_repair",
                  side_effect=lambda *a: (order.append("temporal"), "SELECT 9")[1]),
            patch("core.pipeline_helpers.attempt_field_plan_repair",
                  side_effect=lambda *a: (order.append("field"), "SELECT 8")[1]),
        ):
            attempt = validate_with_repairs("SELECT 1", SCOPE)
        self.assertEqual(order, ["temporal"])
        self.assertEqual(attempt.sql, "SELECT 9")
        self.assertEqual(attempt.repairs, ("governed_temporal_repair",))

    def test_a_temporal_repair_is_not_tried_for_an_unrelated_code(self):
        with (
            patch("core.validator.validate_sql",
                  return_value=(False, "x", "access_denied")),
            patch("core.pipeline_helpers.attempt_governed_temporal_metric_repair"
                  ) as temporal,
        ):
            validate_with_repairs("SELECT 1", SCOPE)
        temporal.assert_not_called()

    def test_a_column_repair_that_is_still_invalid_is_not_accepted(self):
        # Differently-invalid SQL is not progress, and accepting it would
        # spend the LLM repair budget on a query the deterministic path had
        # already made worse.
        bad = SimpleNamespace(ok=False, errors=[{"code": "unknown_column",
                                                 "table": "S.F", "column": "X",
                                                 "suggestions": ["AMT"],
                                                 "candidate_tables": []}])
        still_bad = SimpleNamespace(ok=False, errors=[])
        with (
            patch("core.validator.validate_sql",
                  return_value=(False, "Column X is unknown.", "unknown_column")),
            patch("core.validator.validate_sql_detailed",
                  side_effect=[bad, still_bad]),
            patch("core.validator.repair_unambiguous_unknown_columns",
                  return_value="SELECT BROKEN FROM S.F"),
        ):
            attempt = validate_with_repairs("SELECT X FROM S.F", SCOPE)
        self.assertFalse(attempt.ok)
        self.assertEqual(attempt.sql, "SELECT X FROM S.F")
        self.assertEqual(attempt.repairs, ())

    def test_a_repair_callback_receives_what_changed_and_how_long_it_took(self):
        seen = []
        with (
            patch("core.validator.validate_sql",
                  return_value=(False, "x", "field_plan_mismatch")),
            patch("core.pipeline_helpers.attempt_governed_temporal_metric_repair",
                  return_value=""),
            patch("core.pipeline_helpers.attempt_field_plan_repair",
                  return_value="SELECT 2"),
        ):
            validate_with_repairs("SELECT 1", SCOPE,
                                  on_repair=lambda *a: seen.append(a))
        self.assertEqual(len(seen), 1)
        kind, before, after, metadata, duration_ms = seen[0]
        self.assertEqual((kind, before, after), ("field_plan_repair", "SELECT 1", "SELECT 2"))
        self.assertEqual(metadata["mode"], "deterministic")
        self.assertIsInstance(duration_ms, int)


class TestExecution(unittest.TestCase):

    def _run(self, attempt, executor, timeout=5):
        return asyncio.run(execute(
            attempt, executor=executor, semantic_context=None,
            timeout=timeout, timeout_message="timed out",
        ))

    def test_rows_and_the_governed_sql_come_back(self):
        # execute_governed_query can rewrite the SQL it ran (row caps, policy
        # rewrites), and the attempt must carry what actually executed.
        done = self._run(
            Attempt(sql="SELECT 1", ok=True),
            lambda sql, semantic: _rows([{"A": 1}], sql="SELECT TOP 200 1"))
        self.assertEqual(done.rows, [{"A": 1}])
        self.assertEqual(done.sql, "SELECT TOP 200 1")
        self.assertTrue(done.executed)
        self.assertEqual(done.row_count, 1)

    def test_truncation_is_carried_through(self):
        done = self._run(Attempt(sql="SELECT 1", ok=True),
                         lambda sql, semantic: _rows([{"A": 1}], truncated=True))
        self.assertTrue(done.truncated)

    def test_an_invalid_attempt_is_never_executed(self):
        calls = []
        done = self._run(Attempt(sql="SELECT 1", ok=False, code="parse"),
                         lambda sql, semantic: calls.append(sql))
        self.assertEqual(calls, [])
        self.assertIsNone(done.rows)

    def test_a_timeout_is_flagged_as_a_timeout_not_an_ordinary_error(self):
        # A repair cannot make the database faster. The pipeline suppresses
        # the retry on this specifically, and conflating it with an execution
        # error spends another full timeout window on a rewrite.
        import time as _time

        def _slow(sql, semantic):
            _time.sleep(0.5)
            return _rows([])

        done = self._run(Attempt(sql="SELECT 1", ok=True), _slow, timeout=0.01)
        self.assertTrue(done.timed_out)
        self.assertEqual(done.exec_error, "timed out")
        self.assertIsNone(done.rows)

    def test_an_execution_failure_is_recorded_not_raised(self):
        def _boom(sql, semantic):
            raise RuntimeError("ORA-00942: table or view does not exist")

        done = self._run(Attempt(sql="SELECT 1", ok=True), _boom)
        self.assertIn("ORA-00942", done.exec_error)
        self.assertFalse(done.timed_out)
        self.assertIsNone(done.rows)

    def test_a_policy_denial_is_not_a_failure_of_the_sql(self):
        # It carries no exec_error: nothing about the query is wrong, so the
        # repair path must not treat it as something to rewrite.
        from core.compliance.governed_query import PolicyDeniedError

        decision = SimpleNamespace(
            explanation="Blocked by regulated data policy.",
            reason_code="policy_denied", audit_id="aud-1")

        def _denied(sql, semantic):
            raise PolicyDeniedError(decision)

        done = self._run(Attempt(sql="SELECT 1", ok=True), _denied)
        self.assertIsNone(done.exec_error)
        self.assertFalse(done.ok)
        self.assertEqual(done.code, "policy_denied")
        self.assertIs(done.policy_denied, decision)

    def test_the_semantic_context_reaches_the_executor(self):
        seen = {}
        scope = ValidationScope(known_tables={"S.F"}, db_type="azure_sql",
                                semantic_context={"metric": "Net Revenue"})
        with patch("core.validator.validate_sql", return_value=(True, "OK", "ok")):
            asyncio.run(run_attempt(
                "SELECT 1", scope,
                executor=lambda sql, semantic: seen.setdefault("semantic", semantic)
                or _rows([]),
                timeout=5, timeout_message="t",
            ))
        self.assertEqual(seen["semantic"], {"metric": "Net Revenue"})


class TestRunAttempt(unittest.TestCase):

    def _run(self, sql="SELECT 1", *, valid=True, **kwargs):
        with patch("core.validator.validate_sql",
                   return_value=(True, "OK", "ok") if valid
                   else (False, "bad", "parse")):
            return asyncio.run(run_attempt(
                sql, SCOPE,
                executor=kwargs.pop("executor",
                                    lambda s, sem: _rows([{"A": 1}], sql=s)),
                timeout=5, timeout_message="timed out", **kwargs,
            ))

    def test_a_valid_query_validates_and_executes_in_one_call(self):
        attempt = self._run()
        self.assertTrue(attempt.ok)
        self.assertEqual(attempt.rows, [{"A": 1}])
        self.assertEqual(attempt.source, "primary")

    def test_the_source_label_is_carried(self):
        self.assertEqual(self._run(source="variant:grain").source, "variant:grain")

    def test_the_validated_callback_fires_before_execution(self):
        order = []
        self._run(
            on_validated=lambda a: order.append(("validated", a.ok)),
            executor=lambda s, sem: (order.append(("executed", True)), _rows([]))[1],
        )
        self.assertEqual([o[0] for o in order], ["validated", "executed"])

    def test_the_executing_callback_does_not_fire_for_an_invalid_query(self):
        fired = []
        self._run(valid=False, on_executing=lambda: fired.append(True))
        self.assertEqual(fired, [])

    def test_an_async_executing_callback_is_awaited(self):
        fired = []

        async def _announce():
            fired.append("announced")

        self._run(on_executing=_announce)
        self.assertEqual(fired, ["announced"])

    def test_a_broken_callback_costs_its_own_signal_not_the_answer(self):
        # Observability must never cost the answer -- but it must be visible
        # when it breaks, or a trace that silently stopped recording looks
        # identical to a run that had nothing to record.
        with self.assertLogs("querybot.sql_attempt", level="WARNING") as logged:
            attempt = self._run(
                on_validated=lambda a: (_ for _ in ()).throw(RuntimeError("no")))
        self.assertTrue(attempt.ok)
        self.assertEqual(attempt.rows, [{"A": 1}])
        self.assertTrue(any("on_validated" in line for line in logged.output))


if __name__ == "__main__":
    unittest.main()
