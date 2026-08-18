"""
tests/test_retry_timeout_reporting.py

Catalogue check F5 — "a timeout is reported as a timeout, never repaired into a
different error."

The pipeline decides `_timed_out` once, from the FIRST execution, because that
decision has to be made early: it suppresses the repair loop, since rewriting
valid SQL cannot make the database faster.

But `exec_error` is reassigned afterwards by the retry and progressive-repair
executions, and either of those can itself time out. The final report read the
stale flag, so a query that ran to the statement timeout on its second attempt
was described as a generic execution failure — losing the one diagnosis that
matters and dropping the index guidance built for exactly this case.
"""

from __future__ import annotations

import ast
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_tmpdir = tempfile.mkdtemp(prefix="querybot_timeout_")
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_tmpdir) / "test_key"))

from core.failure_messages import (  # noqa: E402
    build_query_timeout_guidance,
    is_query_timeout,
)

SOURCE = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")


class TestEveryTimeoutSpellingIsRecognised(unittest.TestCase):
    """The retry paths raise their own strings rather than the driver's."""

    def test_the_retry_and_progressive_repair_messages_are_timeouts(self):
        for message in (
            "Retry query timed out after 3 minutes.",
            "Progressive repair query timed out after 3 minutes.",
            "Timeout expired.  The timeout period elapsed prior to completion",
        ):
            with self.subTest(message=message):
                self.assertTrue(is_query_timeout(message))

    def test_a_login_timeout_is_not_a_statement_timeout(self):
        # Connectivity problems are genuinely worth another attempt.
        self.assertFalse(is_query_timeout("Login timeout expired"))


class TestTheFlagIsRederivedBeforeReporting(unittest.TestCase):

    def _assignments(self):
        tree = ast.parse(SOURCE)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == "_handle_query_impl"
        )
        out = {"timed_out": [], "exec_error": [], "report": None}
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "_timed_out":
                        out["timed_out"].append(node.lineno)
                    if isinstance(tgt, ast.Name) and tgt.id == "exec_error":
                        out["exec_error"].append(node.lineno)
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "_log_q":
                    if out["report"] is None or node.lineno > out["report"]:
                        out["report"] = node.lineno
        return out

    def test_exec_error_really_is_reassigned_after_the_first_decision(self):
        a = self._assignments()
        first = min(a["timed_out"])
        self.assertTrue(
            [ln for ln in a["exec_error"] if ln > first],
            "premise no longer holds — exec_error is never reassigned",
        )

    def test_the_flag_is_recomputed_after_the_last_exec_error_assignment(self):
        a = self._assignments()
        last_exec_error = max(a["exec_error"])
        self.assertTrue(
            [ln for ln in a["timed_out"] if ln > last_exec_error],
            "_timed_out is never re-derived after the retry paths reassign "
            "exec_error, so a retry timeout reports as a generic failure",
        )


class TestTheGuidanceStaysActionable(unittest.TestCase):

    def test_it_names_the_governed_column_rather_than_blaming_the_network(self):
        plan = {
            "temporal_policies": [{
                "fact_table": "EMDW_DMART.CUS_ORD_IVC_FCT",
                "fact_column": "CUS_IVC_DT_DMS_KEY",
                "business_role": "Invoice Date",
            }],
        }
        guidance = build_query_timeout_guidance(plan, timeout_seconds=300)
        blob = f"{guidance.get('reason', '')} {guidance.get('next_step', '')}"
        self.assertIn("CUS_IVC_DT_DMS_KEY", blob)
        self.assertNotIn("could not be reached", blob.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
