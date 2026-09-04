"""
Phase 2 (agent roadmap): zero_row_fresh retry -- a freshly generated (not
reused-cache) query with an active date-role temporal filter that returns
zero rows gets exactly one retry attempt with explicit "try a different
table/join/anchor" guidance, mirroring the already-proven reused_plan_empty
repair path. Deliberately scoped to date-filtered questions only (per
explicit user decision) -- a non-date-filtered zero-row result (e.g. "how
many orders did X cancel today") is very often a legitimate answer and must
NOT be retried away into a fabricated non-zero result.

core/query_pipeline.py::handle_query is a large, heavily side-effecting
function with no existing direct-invocation test convention for this
specific retry machinery -- this file follows the established source-scan
convention already used for its sibling checks.
"""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ZeroRowFreshRetryWiringTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")

    def test_zero_row_fresh_is_a_retryable_code(self):
        """Raised by the pipeline rather than the validator, but repairable."""
        from core.validator import REPAIRABLE_REASON_CODES

        self.assertIn("zero_row_fresh", REPAIRABLE_REASON_CODES)

    def test_detection_requires_not_reused_plan(self):
        start = self.source.index('last_code = "zero_row_fresh"')
        block = self.source[max(0, start - 400):start]
        self.assertIn("not _reused_plan", block)

    def test_detection_requires_temporal_policies_present(self):
        start = self.source.index('last_code = "zero_row_fresh"')
        block = self.source[max(0, start - 400):start]
        self.assertIn('_semantic_plan or {}).get("temporal_policies")', block)

    def test_detection_requires_zero_rows_and_no_exec_error(self):
        start = self.source.index('last_code = "zero_row_fresh"')
        block = self.source[max(0, start - 400):start]
        self.assertIn("exec_error is None", block)
        self.assertIn("len(rows) == 0", block)

    def test_zero_row_fresh_check_is_an_elif_of_reused_plan_empty(self):
        # Must be an elif (mutually exclusive), never a second independent
        # if -- the two detection blocks must never both fire for the same
        # result.
        reused_if_pos = self.source.index(
            "if _reused_plan and ok and exec_error is None and rows is not None and len(rows) == 0:"
        )
        fresh_pos = self.source.index('last_code = "zero_row_fresh"')
        between = self.source[reused_if_pos:fresh_pos]
        self.assertIn("\n    elif (", between)
        self.assertIn("not _reused_plan", between)

    def _zero_row_branch(self) -> str:
        """The whole branch, bounded by real code rather than a byte count.

        This used to slice a fixed 800 characters from the `elif`, which meant
        a comment added inside the branch could push the assertions' target out
        of the window and fail a test about text that had not changed.
        """
        start = self.source.index('elif last_code == "zero_row_fresh":')
        end = self.source.index("retry_user = (", start)
        return self.source[start:end]

    def test_repair_note_branch_reuses_governed_date_anchor_lines(self):
        self.assertIn("_governed_date_anchor_repair_lines(_semantic_plan or {})",
                      self._zero_row_branch())

    def test_repair_note_tells_llm_to_try_a_different_approach(self):
        block = self._zero_row_branch()
        self.assertIn("Try a different table, join path, or date anchor", block)
        self.assertIn("do not repeat the", block)
        self.assertIn("same restrictive filter", block)

    def test_the_note_also_carries_the_proven_wrong_filter_value(self):
        """The value index can prove a WHERE literal absent and name the real
        values; that proof used to be spent only on the user-facing apology."""
        block = self._zero_row_branch()
        self.assertIn("_unmatched_literal_repair_lines(sql, account_id)", block)
        self.assertIn("+ _literal_lines", block)


if __name__ == "__main__":
    unittest.main()
