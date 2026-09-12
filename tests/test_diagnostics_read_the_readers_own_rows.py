# -*- coding: utf-8 -*-
"""tests/test_diagnostics_read_the_readers_own_rows.py

A reader restricted to EAST region got a coverage verdict computed over the
whole country.

core/date_coverage.py's check_date_coverage and core/pipeline_helpers.py's
_count_tables_for_zero_row are diagnostics that run ALONGSIDE a governed main
answer -- a freshness caveat, an "is this table really empty" RCA hint. Both
read the warehouse directly through core.schema.run_query, which injects no
row policy: a user whose row policy restricts them to a subset of a shared
fact table got a coverage gap, an anchor date, or a zero-row RCA hint computed
over rows they are not authorized to see.

This is the exact defect class already fixed for core/alert_engine.py's
date-anchor probe (tests/test_anchor_is_scoped_to_the_reader.py) -- fixed
there, and, until this file, NOT applied to these two diagnostics.

The fix is additive, not a rewrite: both functions gained an optional `run`
parameter -- a governed reader (core.compliance.governed_query
.governed_reader_for_user, or the caller's own already-built governed
executor) -- and use it in place of the raw run_query call when supplied. The
ungoverned path remains as a LOUD fallback (a log.warning naming the fact
table) for callers with no live PolicyContext, rather than a hard failure.

Three things are proved here:
  1. governed_reader_for_user itself, in isolation.
  2. Each diagnostic function, given a governed reader, actually calls it
     rather than the ungoverned path -- and is still silent (no warning)
     when it does.
  3. The two PRODUCTION call sites (core/result_renderer.py,
     core/query_pipeline.py) actually construct and pass one, rather than
     leaving the fix as an unused capability. query_pipeline's needs a
     warehouse, a websocket and a model to run end to end, so its wiring is
     read as a syntax tree -- the documented exception this test suite uses
     elsewhere for exactly this shape of dependency.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class _FakeGovernedResult:
    """The one attribute callers read off a GovernedQueryResult -- not a
    mock of the function under test, a stand-in for its return shape."""

    def __init__(self, rows):
        self.rows = rows


class GovernedReaderForUserTests(unittest.TestCase):
    """The shared helper, in isolation."""

    def test_no_account_id_returns_none(self):
        from core.compliance.governed_query import governed_reader_for_user

        self.assertIsNone(
            governed_reader_for_user("", {"id": 7}, channel="portal"))

    def test_no_user_returns_none(self):
        from core.compliance.governed_query import governed_reader_for_user

        self.assertIsNone(
            governed_reader_for_user("acct-1", None, channel="portal"))

    def test_a_real_user_returns_a_working_reader(self):
        """`governed_reader_for_user` does a bare `import store` internally
        (matching core/alert_engine.py's `_owner_execution`, which this
        mirrors) -- that statement resolves via sys.modules, not via a
        `governed_query.store` attribute, so patching THAT attribute (a first
        version of this test did) patches nothing a caller would ever see.
        Patched at the real source instead: store.get_client_state and
        store.get_allowed_tables, which `import store; store.x(...)` reads
        regardless of which module the import happens inside."""
        from core.compliance import governed_query

        with patch("store.get_client_state", return_value={"schema_dir": "x"}), \
             patch("store.get_allowed_tables", return_value={"F_ORDERS"}), \
             patch("core.compliance.policy_engine.resolve_context",
                   return_value="the-context") as mock_resolve, \
             patch("core.schema.load_known_tables", return_value={"F_ORDERS"}), \
             patch("core.schema.load_schema_columns", return_value={}), \
             patch.object(governed_query, "execute_governed_query") as mock_exec:
            mock_exec.return_value = _FakeGovernedResult([{"N": 1}])

            run = governed_query.governed_reader_for_user(
                "acct-1", {"id": 7, "role": "analyst"}, channel="portal",
                db_cfg={"credentials": {"user": "x"}, "db_type": "snowflake"},
            )
            self.assertIsNotNone(run)
            result = run("SELECT 1 AS N")
        self.assertEqual(result.rows, [{"N": 1}])
        mock_exec.assert_called_once()
        _, kwargs = mock_exec.call_args
        self.assertEqual(kwargs["context"], "the-context")
        self.assertEqual(kwargs["known_tables"], {"F_ORDERS"})
        self.assertEqual(kwargs["allowed_tables"], {"F_ORDERS"})
        self.assertEqual(mock_resolve.call_args.kwargs["action"], "query_execution")

    def test_the_context_is_built_for_query_execution_not_something_broader(self):
        """A diagnostic reading under a wider action than the main answer
        itself uses would be a governance escalation, not a fix."""
        from core.compliance import governed_query

        with patch("store.get_client_state", return_value={}), \
             patch("store.get_allowed_tables", return_value=set()), \
             patch("core.compliance.policy_engine.resolve_context",
                   return_value=None) as mock_resolve, \
             patch("core.schema.load_known_tables", return_value=set()), \
             patch("core.schema.load_schema_columns", return_value={}), \
             patch.object(governed_query, "execute_governed_query"):
            governed_query.governed_reader_for_user(
                "acct-1", {"id": 7}, channel="portal")("SELECT 1")
        self.assertEqual(mock_resolve.call_args.kwargs["action"], "query_execution")


class CheckDateCoverageUsesTheGovernedReaderTests(unittest.TestCase):
    """core/date_coverage.py, given a governed reader, actually uses it."""

    def setUp(self):
        self.db_cfg = {"credentials": {}}
        self.policy = {
            "amount": 7, "unit": "day",
            "fact_table": "DBO.F_ORDERS", "fact_column": "ORDER_DATE",
            "date_table": "DBO.F_ORDERS", "date_column": "ORDER_DATE",
            "date_key_type": "native",
        }

    def test_the_governed_reader_is_called_instead_of_run_query(self):
        from core.date_coverage import check_date_coverage

        run = MagicMock(side_effect=[
            _FakeGovernedResult([{"AnchorDate": "2026-07-20"}]),
            _FakeGovernedResult([{"DaysWithData": 7}]),
        ])
        with patch(
            "core.contextual_dates.format_required_anchor",
            return_value="(SELECT MAX(ORDER_DATE) FROM DBO.F_ORDERS)",
        ), patch("core.date_coverage.run_query") as mock_run_query:
            gap = check_date_coverage(
                self.db_cfg, self.policy, "azure_sql", run=run)
        self.assertIsNone(gap)  # full coverage -- no gap to report
        self.assertGreaterEqual(run.call_count, 2)
        mock_run_query.assert_not_called()

    def test_a_real_gap_is_still_detected_through_the_governed_reader(self):
        """No regression: the actual coverage LOGIC must still work when
        reads are routed through `run` instead of run_query."""
        from core.date_coverage import check_date_coverage

        run = MagicMock(side_effect=[
            _FakeGovernedResult([{"AnchorDate": "2026-07-20"}]),
            _FakeGovernedResult([{"DaysWithData": 3}]),  # only 3 of 7 days
        ])
        with patch(
            "core.contextual_dates.format_required_anchor",
            return_value="(SELECT MAX(ORDER_DATE) FROM DBO.F_ORDERS)",
        ):
            gap = check_date_coverage(
                self.db_cfg, self.policy, "azure_sql", run=run)
        self.assertIsNotNone(gap)
        self.assertEqual(gap.actual_days, 3)
        self.assertEqual(gap.requested_days, 7)

    def test_no_reader_supplied_still_works_but_logs_a_warning(self):
        """The documented fallback: best-effort ungoverned reads for a
        caller with no live PolicyContext, loudly flagged rather than
        silent."""
        from core.date_coverage import check_date_coverage

        with patch(
            "core.contextual_dates.format_required_anchor",
            return_value="(SELECT MAX(ORDER_DATE) FROM DBO.F_ORDERS)",
        ), patch("core.date_coverage.run_query", side_effect=[
            [{"AnchorDate": "2026-07-20"}], [{"DaysWithData": 7}],
        ]), self.assertLogs("querybot.date_coverage", level="WARNING") as captured:
            gap = check_date_coverage(self.db_cfg, self.policy, "azure_sql")
        self.assertIsNone(gap)
        self.assertIn("DBO.F_ORDERS", " ".join(captured.output))

    def test_a_reader_supplied_logs_no_such_warning(self):
        from core.date_coverage import check_date_coverage
        import logging

        run = MagicMock(side_effect=[
            _FakeGovernedResult([{"AnchorDate": "2026-07-20"}]),
            _FakeGovernedResult([{"DaysWithData": 7}]),
        ])
        logger = logging.getLogger("querybot.date_coverage")
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.WARNING)
        try:
            with patch(
                "core.contextual_dates.format_required_anchor",
                return_value="(SELECT MAX(ORDER_DATE) FROM DBO.F_ORDERS)",
            ):
                check_date_coverage(
                    self.db_cfg, self.policy, "azure_sql", run=run)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)
        self.assertEqual(records, [])


class CountTablesForZeroRowUsesTheGovernedReaderTests(unittest.TestCase):

    def test_the_governed_reader_is_called_instead_of_run_query(self):
        from core.pipeline_helpers import _count_tables_for_zero_row

        run = MagicMock(return_value=_FakeGovernedResult([{"RowCount": 0}]))
        with patch("core.pipeline_helpers.run_query") as mock_run_query:
            counts = _count_tables_for_zero_row(
                {"db_type": "azure_sql"}, ["DBO.F_ORDERS"], run=run)
        self.assertEqual(counts, {"DBO.F_ORDERS": 0})
        run.assert_called_once()
        mock_run_query.assert_not_called()

    def test_a_governance_refusal_reports_unknown_not_a_crash(self):
        """A table this reader may not see at all: reported the same way
        every other failure already is -- None, "could not determine" --
        never surfaced as an exception to the caller."""
        from core.pipeline_helpers import _count_tables_for_zero_row

        def _refuse(sql):
            raise PermissionError("access_denied")

        counts = _count_tables_for_zero_row(
            {"db_type": "azure_sql"}, ["DBO.RESTRICTED"], run=_refuse)
        self.assertEqual(counts, {"DBO.RESTRICTED": None})

    def test_no_reader_supplied_still_works_but_logs_a_warning(self):
        from core.pipeline_helpers import _count_tables_for_zero_row

        with patch("core.pipeline_helpers.run_query",
                   return_value=[{"RowCount": 5}]), \
             self.assertLogs("querybot", level="WARNING") as captured:
            counts = _count_tables_for_zero_row(
                {"db_type": "azure_sql"}, ["DBO.F_ORDERS"])
        self.assertEqual(counts, {"DBO.F_ORDERS": 5})
        self.assertIn("DBO.F_ORDERS", " ".join(captured.output))

    def test_a_reader_supplied_logs_no_such_warning(self):
        import logging

        from core.pipeline_helpers import _count_tables_for_zero_row

        run = MagicMock(return_value=_FakeGovernedResult([{"RowCount": 1}]))
        # core/pipeline_helpers.py logs under the bare "querybot" logger, not
        # a per-module child -- confirmed by reading log.warning's own output
        # in the sibling test above, not assumed.
        logger = logging.getLogger("querybot")
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.WARNING)
        try:
            _count_tables_for_zero_row(
                {"db_type": "azure_sql"}, ["DBO.F_ORDERS"], run=run)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)
        self.assertEqual(records, [])


class ProductionWiringTests(unittest.TestCase):
    """The fix only matters if the two real call sites actually pass a
    governed reader, not merely offer the capability."""

    def test_result_renderer_builds_a_reader_scoped_to_the_sending_user(self):
        import ast
        import inspect

        import core.result_renderer as rr

        tree = ast.parse(inspect.getsource(rr._send_results))
        called = {
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        } | {
            node.func.attr for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("governed_reader_for_user", called)
        # And it's built from THIS request's own account_id/portal_user, not
        # a placeholder -- passing "" for account_id still constructs a
        # (uselessly None) reader without raising, so only checking that the
        # call exists would miss it scoping to the wrong (or no) account.
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", "") == "governed_reader_for_user"):
                arg_names = [getattr(a, "id", "") for a in node.args]
                self.assertEqual(arg_names[:2], ["account_id", "portal_user"])
                break
        else:
            self.fail("governed_reader_for_user call site not found")
        # And the result is actually threaded into check_date_coverage, not
        # just built and left unused.
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", "") == "check_date_coverage"):
                kw_names = {kw.arg for kw in node.keywords}
                self.assertIn("run", kw_names)
                return
        self.fail("check_date_coverage call site not found")

    def test_query_pipeline_passes_the_same_governed_executor_the_main_answer_uses(self):
        """core/query_pipeline.py's _handle_query_impl needs a warehouse, a
        websocket and a model to execute for real, so its branch is read as
        a syntax tree -- the documented exception this suite uses elsewhere
        (e.g. tests/test_one_date_per_fact.py) for exactly this shape of
        dependency."""
        import ast
        import inspect

        import core.query_pipeline as qp

        tree = ast.parse(inspect.getsource(qp._handle_query_impl))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "to_thread"
                    and any(
                        getattr(a, "id", "") == "_count_tables_for_zero_row"
                        for a in node.args
                    )):
                kwargs = {kw.arg: getattr(kw.value, "id", "") for kw in node.keywords}
                self.assertEqual(kwargs.get("run"), "_execute_with_policy")
                return
        self.fail("_count_tables_for_zero_row call site not found")


if __name__ == "__main__":
    unittest.main()
