"""
tests/test_examples_validation_timeout.py

Regression tests for two related bugs in the Stage-2 example-validation step
(runs after every KB build):

1. core/examples.py._open_connection did not set any driver-level query
   timeout, so a single slow validation pattern (full scan, lock wait) would
   hang the whole sequential ~200-pattern batch indefinitely with nothing
   further logged.
2. Executor-thread cancellation cannot terminate a pyodbc call blocked inside
   native ODBC code on Windows. Validation therefore runs in a supervised
   child process with a hard timeout and a stop path that can terminate and
   reap the worker without taking down the QueryBot service.
"""
import asyncio
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.examples as examples
import core.dispatcher as dispatcher


def _successful_process_worker(
    result_queue, stop_event, account_id, kb_dir, credentials, db_type, chroma_dir
):
    """Pickle-safe worker used by spawn-based process supervision tests."""
    result_queue.put({
        "kind": "progress",
        "payload": {
            "phase": "validating_sql",
            "current": 1,
            "total": 1,
            "validated": 1,
            "failed": 0,
        },
    })
    result_queue.put({
        "kind": "result",
        "status": "completed",
        "total": 1,
        "validated": 1,
        "failed": 0,
    })


def _hung_process_worker(
    result_queue, stop_event, account_id, kb_dir, credentials, db_type, chroma_dir
):
    """Simulate native ODBC code that ignores both events and cancellation."""
    import time
    time.sleep(60)


def _exploding_process_worker(
    result_queue, stop_event, account_id, kb_dir, credentials, db_type, chroma_dir
):
    raise RuntimeError("db exploded")


class OpenConnectionTimeoutTests(unittest.TestCase):
    def test_azure_sql_sets_pyodbc_timeout(self):
        fake_conn = MagicMock()
        with patch("core.schema._az_connect", return_value=fake_conn):
            conn = examples._open_connection({}, "azure_sql")
        self.assertIs(conn, fake_conn)
        self.assertEqual(conn.timeout, examples._QUERY_TIMEOUT_SECONDS)


class CompileOnlyValidationTests(unittest.TestCase):
    def test_azure_sql_describes_result_without_executing_source_query(self):
        cursor = MagicMock()
        conn = MagicMock()
        conn.cursor.return_value = cursor

        examples._execute_on_connection(
            conn, "azure_sql", "SELECT SUM(amount) AS revenue FROM huge_fact"
        )

        command, statement = cursor.execute.call_args.args
        self.assertIn("sp_describe_first_result_set", command)
        self.assertEqual(statement, "SELECT SUM(amount) AS revenue FROM huge_fact")
        self.assertNotEqual(command, statement)

    def test_snowflake_uses_explain_not_query_execution(self):
        import types

        cursor = MagicMock()
        conn = MagicMock()
        conn.cursor.return_value = cursor
        connector_module = types.ModuleType("snowflake.connector")
        connector_module.DictCursor = object
        snowflake_module = types.ModuleType("snowflake")
        snowflake_module.connector = connector_module

        with patch.dict(sys.modules, {
            "snowflake": snowflake_module,
            "snowflake.connector": connector_module,
        }):
            examples._execute_on_connection(
                conn, "snowflake", "SELECT SUM(amount) FROM huge_fact"
            )

        command = cursor.execute.call_args.args[0]
        self.assertTrue(command.startswith("EXPLAIN USING TEXT SELECT"))

    def test_oracle_uses_cursor_parse_without_execute(self):
        cursor = MagicMock()
        conn = MagicMock()
        conn.cursor.return_value = cursor

        examples._execute_on_connection(
            conn, "oracle", "SELECT SUM(amount) FROM huge_fact"
        )

        cursor.parse.assert_called_once_with("SELECT SUM(amount) FROM huge_fact")
        cursor.execute.assert_not_called()

    def test_oracle_sets_call_timeout_in_milliseconds(self):
        fake_conn = MagicMock()
        with patch("core.schema._ora_connect", return_value=fake_conn):
            conn = examples._open_connection({}, "oracle")
        self.assertIs(conn, fake_conn)
        self.assertEqual(conn.call_timeout, examples._QUERY_TIMEOUT_SECONDS * 1000)


class BatchProgressAndStopTests(unittest.TestCase):
    def test_stop_event_is_checked_between_queries_and_skips_embedding(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "orders_queries.md").write_text(
                "Q: first query\nSQL: SELECT * FROM first_table\n"
                "Q: second query\nSQL: SELECT * FROM second_table\n",
                encoding="utf-8",
            )
            stop_event = threading.Event()
            progress = []
            fake_conn = MagicMock()

            def _capture(payload):
                progress.append(payload)
                if payload.get("current") == 1:
                    stop_event.set()

            with patch("core.examples._open_connection", return_value=fake_conn), \
                 patch("core.examples._execute_on_connection") as execute, \
                 patch("store.save_validated_example") as save:
                validated = examples.validate_and_store_examples(
                    "acct1",
                    tmp,
                    {},
                    "azure_sql",
                    "acct1",
                    stop_event=stop_event,
                    progress_callback=_capture,
                )

            self.assertEqual(validated, 1)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(save.call_count, 1)
            self.assertEqual([item["current"] for item in progress], [0, 1])
            fake_conn.close.assert_called_once()

    def test_snowflake_sets_statement_timeout_via_alter_session(self):
        fake_cursor = MagicMock()
        fake_conn = MagicMock()
        fake_conn.cursor.return_value = fake_cursor
        with patch("core.schema._sf_connect", return_value=fake_conn):
            conn = examples._open_connection({}, "snowflake")
        self.assertIs(conn, fake_conn)
        executed_sql = fake_cursor.execute.call_args[0][0]
        self.assertIn("STATEMENT_TIMEOUT_IN_SECONDS", executed_sql)
        self.assertIn(str(examples._QUERY_TIMEOUT_SECONDS), executed_sql)

    def test_timeout_setting_failure_does_not_prevent_connection_reuse(self):
        # A driver that rejects the timeout attribute must not break validation
        # entirely — the connection is still usable, just without the guard.
        class _NoTimeoutConn:
            @property
            def timeout(self):
                raise AttributeError("no timeout support")

            @timeout.setter
            def timeout(self, value):
                raise AttributeError("no timeout support")

        fake_conn = _NoTimeoutConn()
        with patch("core.schema._az_connect", return_value=fake_conn):
            conn = examples._open_connection({}, "azure_sql")
        self.assertIs(conn, fake_conn)


class RunExampleValidationProcessTests(unittest.TestCase):
    _DB_CFG = {"credentials": {}, "db_type": "azure_sql"}

    def test_successful_worker_reports_progress_without_blocking_event_loop(self):
        progress = []

        async def _capture(payload):
            progress.append(payload)

        result = asyncio.run(dispatcher._run_validation_process(
            "acct1",
            "kb_dir",
            "chroma_dir",
            self._DB_CFG,
            progress_callback=_capture,
            timeout_seconds=5,
            worker_target=_successful_process_worker,
        ))

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["validated"], 1)
        self.assertEqual(progress[0]["step"], "Validating generated SQL examples (1/1)")

    def test_hung_native_worker_is_force_terminated_at_hard_timeout(self):
        import time

        started = time.monotonic()
        result = asyncio.run(dispatcher._run_validation_process(
            "acct1",
            "kb_dir",
            "chroma_dir",
            self._DB_CFG,
            timeout_seconds=0.25,
            worker_target=_hung_process_worker,
        ))
        elapsed = time.monotonic() - started

        self.assertEqual(result["status"], "timeout")
        self.assertLess(elapsed, 5)

    def test_stop_event_force_terminates_worker_that_ignores_cancellation(self):
        async def _run():
            stop_event = asyncio.Event()

            async def _request_stop():
                await asyncio.sleep(0.15)
                stop_event.set()

            stopper = asyncio.create_task(_request_stop())
            try:
                return await dispatcher._run_validation_process(
                    "acct1",
                    "kb_dir",
                    "chroma_dir",
                    self._DB_CFG,
                    stop_event=stop_event,
                    timeout_seconds=10,
                    stop_grace_seconds=0.1,
                    worker_target=_hung_process_worker,
                )
            finally:
                await stopper

        result = asyncio.run(_run())
        self.assertEqual(result["status"], "stopped")

    def test_worker_crash_is_caught_not_raised(self):
        result = asyncio.run(dispatcher._run_validation_process(
            "acct1",
            "kb_dir",
            "chroma_dir",
            self._DB_CFG,
            timeout_seconds=5,
            worker_target=_exploding_process_worker,
        ))
        self.assertEqual(result["status"], "error")
        self.assertIn("exited with code", result["error"])

    def test_timeout_preserves_real_query_total_for_admin_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "orders_queries.md").write_text(
                "Q: revenue\nSQL: SELECT SUM(amount) FROM orders\n",
                encoding="utf-8",
            )
            worker_result = {
                "status": "timeout",
                "total": 0,
                "validated": 0,
                "failed": 0,
            }
            with patch(
                "core.dispatcher._run_validation_process",
                new=AsyncMock(return_value=worker_result),
            ):
                result = asyncio.run(dispatcher._run_example_validation(
                    "acct1",
                    tmp,
                    "acct1",
                    self._DB_CFG,
                    timeout_seconds=0.1,
                ))

        self.assertEqual(result, {
            "status": "timeout",
            "total": 1,
            "validated": 0,
            "failed": 1,
        })

    def test_completed_worker_activates_only_when_every_pattern_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "orders_queries.md").write_text(
                "Q: revenue\nSQL: SELECT SUM(amount) FROM orders\n",
                encoding="utf-8",
            )
            worker_result = {
                "status": "completed",
                "total": 1,
                "validated": 1,
                "failed": 0,
            }
            with patch(
                "core.dispatcher._run_validation_process",
                new=AsyncMock(return_value=worker_result),
            ):
                result = asyncio.run(dispatcher._run_example_validation(
                    "acct1", tmp, "acct1", self._DB_CFG
                ))

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["validated"], 1)
        self.assertEqual(result["failed"], 0)


class StaleNullDiagnosticExampleFilterTests(unittest.TestCase):
    """
    A harvested/approved few-shot example carrying the old MatchedRows/
    NonNullMetricRows diagnostic shape on a plain time-bounded aggregate
    (from before core/validator.py's null_aggregate_diagnostic guard was
    narrowed to identity/category filters only) keeps re-teaching that
    deprecated pattern to the LLM every time it's retrieved — a concrete
    Q/SQL example in the prompt tends to win over even an explicit prose
    rule telling the model not to reproduce it. format_examples_for_prompt
    must drop such examples rather than inject them.
    """

    _STALE_SQL = (
        "SELECT COUNT_BIG(*) AS MatchedRows, COUNT(SOP_CUS_IVC_LIN_AMT) AS NonNullMetricRows, "
        "COALESCE(SUM(SOP_CUS_IVC_LIN_AMT), 0) AS TOTAL_SALES "
        "FROM EMCODW_DEV.EMDW_DMART.CUS_ORD_IVC_FCT "
        "WHERE CUS_ORD_DT_DMS_KEY >= 20260101"
    )
    _LEGIT_SQL = (
        "SELECT COUNT_BIG(*) AS MatchedRows, COUNT(CUS_IVC_LIN_AMT) AS NonNullMetricRows, "
        "COALESCE(SUM(CUS_IVC_LIN_AMT), 0) AS Revenue "
        "FROM CUS_ORD_IVC_FCT WHERE CUS_DMS_KEY = 123"
    )

    def test_stale_date_only_diagnostic_example_dropped(self):
        out = examples.format_examples_for_prompt(
            [{"question": "what is my sales for the last 7 days", "sql": self._STALE_SQL}]
        )
        self.assertEqual(out, "")

    def test_legit_identity_filter_diagnostic_example_kept(self):
        out = examples.format_examples_for_prompt(
            [{"question": "revenue for customer 123", "sql": self._LEGIT_SQL}]
        )
        self.assertIn("revenue for customer 123", out)
        self.assertIn("MatchedRows", out)

    def test_example_without_diagnostic_pattern_unaffected(self):
        out = examples.format_examples_for_prompt(
            [{"question": "total orders", "sql": "SELECT COUNT(*) AS TotalOrders FROM ORDERS"}]
        )
        self.assertIn("total orders", out)

    def test_mixed_list_drops_only_the_stale_one(self):
        out = examples.format_examples_for_prompt([
            {"question": "what is my sales for the last 7 days", "sql": self._STALE_SQL},
            {"question": "total orders", "sql": "SELECT COUNT(*) AS TotalOrders FROM ORDERS"},
        ])
        self.assertNotIn("last 7 days", out)
        self.assertIn("total orders", out)

    def test_all_examples_stale_yields_empty_string(self):
        out = examples.format_examples_for_prompt(
            [{"question": "what is my sales for the last 7 days", "sql": self._STALE_SQL}]
        )
        self.assertEqual(out, "")  # not just an empty examples list — no header either

    def test_is_stale_helper_direct(self):
        self.assertTrue(examples._is_stale_null_diagnostic_example(self._STALE_SQL))
        self.assertFalse(examples._is_stale_null_diagnostic_example(self._LEGIT_SQL))
        self.assertFalse(examples._is_stale_null_diagnostic_example("SELECT COUNT(*) FROM T"))
        self.assertFalse(examples._is_stale_null_diagnostic_example(""))


if __name__ == "__main__":
    unittest.main()
