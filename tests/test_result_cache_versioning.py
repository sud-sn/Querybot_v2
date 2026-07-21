"""Sprint 3e: version-aware DuckDB follow-up cache.

learning_candidate and eval_run already carry a contract_version column
(see store/db.py migrations) so the learning/eval side of "staleness-aware
retrieval" was already covered by earlier work. The one real gap was
core/result_cache.py: a cached result from a follow-up-question session had
no idea which semantic contract was live when it was produced, so a stale
row set (built under a metric/field/join definition an admin has since
changed) could silently answer a follow-up question under the OLD meaning.

This closes it two ways:
  1. core/result_cache.py: ResultCache.get_contract_version() reads the
     contract_version stamped into a cache entry's existing metadata dict
     (no schema change - store()/derive_snapshot() already accepted
     metadata=).
  2. core/query_pipeline.py: before routing a follow-up to the cache, the
     cached entry's stamped version is compared against the live contract
     version; a mismatch clears the cache instead of answering from it.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.result_cache import ResultCache


class ContractVersionAccessorTests(unittest.TestCase):
    def setUp(self):
        self.cache = ResultCache(max_sessions=4)
        self.session = "acct1:user1"

    def test_version_stamped_at_store_is_readable_back(self):
        self.cache.store(
            self.session, [{"a": 1}], "q", "SELECT 1",
            metadata={"contract_version": "v1"},
        )
        self.assertEqual(self.cache.get_contract_version(self.session), "v1")

    def test_missing_metadata_returns_empty_not_error(self):
        self.cache.store(self.session, [{"a": 1}], "q", "SELECT 1")
        self.assertEqual(self.cache.get_contract_version(self.session), "")

    def test_missing_session_returns_empty(self):
        self.assertEqual(self.cache.get_contract_version("no-such-session"), "")

    def test_version_readable_by_result_id(self):
        rid = self.cache.store(
            self.session, [{"a": 1}], "q", "SELECT 1",
            metadata={"contract_version": "v7"}, result_id="r1",
        )
        self.assertEqual(
            self.cache.get_contract_version(self.session, result_id=rid), "v7",
        )

    def test_derived_snapshot_preserves_no_version_by_default(self):
        # derive_snapshot() doesn't inherit the source's metadata dict
        # unless the caller passes it explicitly - confirms current
        # behavior so a future change here is a deliberate decision, not
        # an accident.
        self.cache.store(
            self.session, [{"a": 1}], "q", "SELECT 1",
            metadata={"contract_version": "v1"}, result_id="src",
        )
        self.cache.derive_snapshot(
            self.session, "src", [{"a": 2}], question="q2", operation="sort",
        )
        self.assertEqual(self.cache.get_contract_version(self.session), "")


class StalenessDecisionTests(unittest.TestCase):
    """Mirrors the exact comparison core/query_pipeline.py's new block
    performs (both non-empty AND different => stale). Testing the pure
    decision here since the surrounding ~1,100-line handle_query gets
    static source assertions, matching this codebase's established
    pattern (see ResolutionPlanWiringTests, QuestionScrubWiringTests)."""

    @staticmethod
    def _is_stale(cached: str, current: str) -> bool:
        return bool(cached and current and cached != current)

    def test_same_version_not_stale(self):
        self.assertFalse(self._is_stale("v1", "v1"))

    def test_different_versions_are_stale(self):
        self.assertTrue(self._is_stale("v1", "v2"))

    def test_unknown_cached_version_never_stale(self):
        # Pre-Sprint-3e caches and accounts with no compiled contract must
        # not be punished for a version they never had.
        self.assertFalse(self._is_stale("", "v2"))

    def test_unknown_current_version_never_stale(self):
        self.assertFalse(self._is_stale("v1", ""))

    def test_both_unknown_not_stale(self):
        self.assertFalse(self._is_stale("", ""))


class ResultCacheVersioningWiringTests(unittest.TestCase):
    """Static source assertions for the ~1,100-line handle_query, matching
    this codebase's established pattern for that function."""

    def setUp(self):
        self.src = (ROOT / "core/query_pipeline.py").read_text(encoding="utf-8")

    def test_contract_version_fetched_before_cache_read_check(self):
        version_pos = self.src.index("_load_contract_early")
        cache_read_pos = self.src.index('action="cache_read"')
        self.assertLess(version_pos, cache_read_pos)

    def test_stale_cache_is_cleared_not_silently_used(self):
        anchor = "_cached_contract_version = result_cache.get_contract_version(_session_id)"
        self.assertIn(anchor, self.src)
        block = self.src[self.src.index(anchor):self.src.index(anchor) + 800]
        self.assertIn("result_cache.clear(_session_id)", block)
        self.assertIn('"result_cache_stale"', block)

    def test_send_results_call_sites_pass_contract_version(self):
        # All three _send_results() call sites (governed-cache route,
        # metric-registry route, main LLM-SQL route) must stamp the result
        # they cache with the version that was live when it was produced.
        for anchor, next_call in (
            (
                "await _send_results(\n                event,\n                adapter,\n                question,\n                _cache_rows,",
                "await _send_results(event, adapter, question, rows, sql_from_metric,",
            ),
            (
                "await _send_results(event, adapter, question, rows, sql_from_metric,",
                "await _send_results(event, adapter, question, rows, sql, duration_ms,",
            ),
            (
                "await _send_results(event, adapter, question, rows, sql, duration_ms,",
                None,
            ),
        ):
            start = self.src.index(anchor)
            end = self.src.index(next_call, start) if next_call else start + 600
            call_block = self.src[start:end]
            self.assertIn(
                "contract_version=_contract_version", call_block,
                f"call site starting {anchor!r} missing contract_version kwarg",
            )

    def test_web_adapter_cache_result_accepts_and_forwards_contract_version(self):
        src = (ROOT / "gateway/web_adapter.py").read_text(encoding="utf-8")
        self.assertIn("contract_version: str = \"\"", src)
        self.assertIn('"contract_version": contract_version', src)


if __name__ == "__main__":
    unittest.main()
