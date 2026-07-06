"""
tests/test_examples_validation_timeout.py

Regression tests for two related bugs in the Stage-2 example-validation step
(runs after every KB build):

1. core/examples.py._open_connection did not set any driver-level query
   timeout, so a single slow validation pattern (full scan, lock wait) would
   hang the whole sequential ~200-pattern batch indefinitely with nothing
   further logged.
2. core/dispatcher.py._run_example_validation is `async def` but called the
   blocking validate_and_store_examples() directly (no run_in_executor) —
   since this runs on the shared asyncio event loop, a slow/hung validation
   batch froze every other request the app was serving, not just the KB
   build in progress.
"""
import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.examples as examples
import core.dispatcher as dispatcher


class OpenConnectionTimeoutTests(unittest.TestCase):
    def test_azure_sql_sets_pyodbc_timeout(self):
        fake_conn = MagicMock()
        with patch("core.schema._az_connect", return_value=fake_conn):
            conn = examples._open_connection({}, "azure_sql")
        self.assertIs(conn, fake_conn)
        self.assertEqual(conn.timeout, examples._QUERY_TIMEOUT_SECONDS)

    def test_oracle_sets_call_timeout_in_milliseconds(self):
        fake_conn = MagicMock()
        with patch("core.schema._ora_connect", return_value=fake_conn):
            conn = examples._open_connection({}, "oracle")
        self.assertIs(conn, fake_conn)
        self.assertEqual(conn.call_timeout, examples._QUERY_TIMEOUT_SECONDS * 1000)

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


class RunExampleValidationExecutorTests(unittest.TestCase):
    def test_validation_runs_off_the_event_loop(self):
        """The blocking validator must run via run_in_executor, not be awaited
        directly on the event loop — otherwise it blocks all concurrent
        requests, not just this KB build's background task."""
        calls = []

        def _fake_validate(account_id, queries_dir, credentials, db_type, chroma_dir):
            import threading
            calls.append(threading.current_thread() is threading.main_thread())
            return 5

        with patch("core.examples.validate_and_store_examples", _fake_validate):
            asyncio.run(dispatcher._run_example_validation(
                "acct1", "kb_dir", "chroma_dir",
                {"credentials": {}, "db_type": "azure_sql"},
            ))
        self.assertEqual(calls, [False])  # ran on an executor thread, not main

    def test_overall_timeout_is_bounded_not_infinite(self):
        import time

        def _slow_validate(*args, **kwargs):
            time.sleep(0.2)
            return 0

        with patch("core.examples.validate_and_store_examples", _slow_validate), \
             patch("asyncio.wait_for", wraps=asyncio.wait_for) as spy:
            asyncio.run(dispatcher._run_example_validation(
                "acct1", "kb_dir", "chroma_dir",
                {"credentials": {}, "db_type": "azure_sql"},
            ))
        # Confirm the call is wrapped with SOME finite timeout, not left to
        # run unbounded on the event loop.
        self.assertTrue(spy.called)
        _, kwargs = spy.call_args
        self.assertIsNotNone(kwargs.get("timeout"))
        self.assertGreater(kwargs["timeout"], 0)

    def test_timeout_error_is_caught_and_logged_not_raised(self):
        async def _never_returns(*a, **k):
            raise asyncio.TimeoutError()

        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            try:
                asyncio.run(dispatcher._run_example_validation(
                    "acct1", "kb_dir", "chroma_dir",
                    {"credentials": {}, "db_type": "azure_sql"},
                ))
            except asyncio.TimeoutError:
                self.fail("_run_example_validation must catch its own timeout, not propagate it")

    def test_exception_from_validator_is_caught_not_raised(self):
        def _boom(*args, **kwargs):
            raise RuntimeError("db exploded")

        with patch("core.examples.validate_and_store_examples", _boom):
            try:
                asyncio.run(dispatcher._run_example_validation(
                    "acct1", "kb_dir", "chroma_dir",
                    {"credentials": {}, "db_type": "azure_sql"},
                ))
            except RuntimeError:
                self.fail("_run_example_validation must not propagate validator exceptions")


if __name__ == "__main__":
    unittest.main()
