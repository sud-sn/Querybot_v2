"""
tests/test_llm_call_bounds.py

Every LLM call was unbounded in wall-clock time.

core/llm.py constructed all three provider clients with no timeout and no
explicit retry policy, so each inherited the vendor SDK default of 600 s per
request with 2 automatic retries -- roughly 30 minutes for a single call. The
SQL path issues up to four sequential calls (generate, recovery, retry,
progressive repair) and the pipeline bounds none of them. That is the most
likely explanation for the 800 s question observed on the EMCO server, and it
is indistinguishable from a hung product to the person waiting.

The fix sets an explicit timeout and retry count on all three clients, both
env-overridable and clamped, and folds them into the client cache key so a
changed setting cannot silently reuse an old client.

Every assertion here fails against the pre-fix code.
"""

import ast
import os
import unittest
from pathlib import Path

from core.llm import _llm_max_retries, _llm_timeout_seconds

REPO = Path(__file__).resolve().parents[1]

class LlmCallsAreBounded(unittest.TestCase):
    """No timeout and no explicit retry meant ~30 min per call, worst case."""

    def setUp(self):
        for var in ("QUERYBOT_LLM_TIMEOUT_SECONDS", "QUERYBOT_LLM_MAX_RETRIES"):
            self.addCleanup(os.environ.pop, var, None)
            os.environ.pop(var, None)

    def test_default_timeout_is_bounded_and_well_under_the_sdk_default(self):
        self.assertLessEqual(
            _llm_timeout_seconds(),
            300,
            "must be materially tighter than the vendor SDK 600 s default",
        )
        self.assertGreaterEqual(
            _llm_timeout_seconds(), 60, "must leave room for a real completion"
        )

    def test_retries_are_explicit_and_small(self):
        self.assertLessEqual(_llm_max_retries(), 2)

    def test_worst_case_wall_clock_is_bounded(self):
        worst = _llm_timeout_seconds() * (1 + _llm_max_retries())
        self.assertLess(
            worst,
            600,
            "one LLM call must not be able to outlast a user by itself",
        )

    def test_env_overrides_apply_and_are_clamped(self):
        os.environ["QUERYBOT_LLM_TIMEOUT_SECONDS"] = "45"
        self.assertEqual(_llm_timeout_seconds(), 45.0)
        os.environ["QUERYBOT_LLM_TIMEOUT_SECONDS"] = "99999"
        self.assertEqual(_llm_timeout_seconds(), 600.0)
        os.environ["QUERYBOT_LLM_TIMEOUT_SECONDS"] = "not-a-number"
        self.assertEqual(_llm_timeout_seconds(), 180.0)

    def test_openai_client_actually_carries_the_timeout(self):
        from core.llm import _get_openai_client

        client = _get_openai_client("sk-test-not-a-real-key")
        self.assertLessEqual(float(client.timeout), 300.0)
        self.assertLessEqual(int(client.max_retries), 2)

    def test_azure_client_actually_carries_the_timeout(self):
        from core.llm import _get_azure_client

        client = _get_azure_client(
            "k", "https://example.openai.azure.com", "2024-02-01"
        )
        self.assertLessEqual(float(client.timeout), 300.0)
        self.assertLessEqual(int(client.max_retries), 2)

    def test_anthropic_client_is_constructed_with_both_bounds(self):
        """The anthropic SDK is not installed here, so assert the wiring."""
        src = (REPO / "core" / "llm.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_get_anthropic_client"
        )
        calls = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and getattr(n.func, "attr", "") == "AsyncAnthropic"
        ]
        self.assertTrue(calls, "expected an AsyncAnthropic construction")
        kwargs = {k.arg for k in calls[0].keywords}
        self.assertIn("timeout", kwargs)
        self.assertIn("max_retries", kwargs)


if __name__ == "__main__":
    unittest.main()
