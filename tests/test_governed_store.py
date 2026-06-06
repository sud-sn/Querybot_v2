"""
tests/test_governed_store.py

Unit tests for the governed example store (Day 5-7).

Covers:
  core/governed_store.py  -- upsert, delete, retrieve, count, backfill
  store/learning_store.py -- _fire_governed_upsert / _fire_governed_delete hooks
  core/examples.py        -- dual-collection retrieve_similar_examples

Strategy: all Qdrant and embedding calls are mocked.  No network, no GPU.
"""

import hashlib
import unittest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_qdrant_client(
    collections: list[str] | None = None,
    count: int = 0,
    search_results: list | None = None,
):
    """
    Build a MagicMock Qdrant client with pre-configured return values.

    collections: list of collection names that 'exist' (default: ["querybot_governed"])
    count:       what .count() returns
    search_results: list of mock ScoredPoint objects returned by query_points
    """
    if collections is None:
        collections = ["querybot_governed"]

    client = MagicMock()

    # get_collections()
    coll_objects = []
    for name in collections:
        c = MagicMock()
        c.name = name
        coll_objects.append(c)
    client.get_collections.return_value.collections = coll_objects

    # count()
    count_result = MagicMock()
    count_result.count = count
    client.count.return_value = count_result

    # query_points()
    if search_results is None:
        search_results = []
    client.query_points.return_value.points = search_results

    return client


def _make_hit(question: str, sql: str, source: str = "auto", score: float = 0.9):
    """Build a mock Qdrant ScoredPoint."""
    hit = MagicMock()
    hit.score = score
    hit.payload = {
        "question":     question,
        "sql":          sql,
        "source":       source,
        "final_score":  85,
        "account_id":   "acct1",
        "candidate_id": "abc123def456",
        "doc_type":     "governed_example",
    }
    return hit


_FAKE_VECTOR = [0.1] * 384


# ---------------------------------------------------------------------------
# Helpers to compute expected point IDs
# ---------------------------------------------------------------------------

def _expected_point_id(candidate_id: str) -> str:
    digest = hashlib.md5(f"governed::{candidate_id}".encode()).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


# ===========================================================================
# core/governed_store -- upsert_governed_example
# ===========================================================================

class TestUpsertGovernedExample(unittest.TestCase):

    def _call(self, **kwargs):
        from core.governed_store import upsert_governed_example
        return upsert_governed_example(**kwargs)

    def _patches(self, client=None, existing=None):
        """Return context-manager patches for qdrant + embed."""
        if client is None:
            client = _make_qdrant_client(
                collections=existing if existing is not None else ["querybot_governed"]
            )
        return (
            patch("core.governed_store._qdrant", return_value=client),
            patch("core.governed_store._embed", return_value=[_FAKE_VECTOR]),
        )

    def test_returns_non_empty_point_id(self):
        client = _make_qdrant_client()
        with patch("core.governed_store._qdrant", return_value=client), \
             patch("core.governed_store._embed", return_value=[_FAKE_VECTOR]):
            pid = self._call(
                candidate_id="cand1",
                account_id="acct1",
                question="What is revenue?",
                sql="SELECT SUM(revenue) FROM sales",
            )
        self.assertTrue(len(pid) > 0)

    def test_point_id_is_uuid_format(self):
        client = _make_qdrant_client()
        with patch("core.governed_store._qdrant", return_value=client), \
             patch("core.governed_store._embed", return_value=[_FAKE_VECTOR]):
            pid = self._call(
                candidate_id="cand1",
                account_id="acct1",
                question="What is revenue?",
                sql="SELECT SUM(revenue) FROM sales",
            )
        parts = pid.split("-")
        self.assertEqual(len(parts), 5, "Expected UUID format: 8-4-4-4-12")

    def test_same_candidate_id_produces_same_point_id(self):
        client = _make_qdrant_client()
        with patch("core.governed_store._qdrant", return_value=client), \
             patch("core.governed_store._embed", return_value=[_FAKE_VECTOR]):
            pid1 = self._call(
                candidate_id="cand_abc",
                account_id="acct1",
                question="Q1",
                sql="SELECT 1",
            )
            pid2 = self._call(
                candidate_id="cand_abc",
                account_id="acct1",
                question="Q1 updated",
                sql="SELECT 2",
            )
        self.assertEqual(pid1, pid2, "Deterministic ID must not change on re-upsert")

    def test_different_candidate_ids_produce_different_point_ids(self):
        client = _make_qdrant_client()
        with patch("core.governed_store._qdrant", return_value=client), \
             patch("core.governed_store._embed", return_value=[_FAKE_VECTOR]):
            pid1 = self._call(
                candidate_id="cand_a", account_id="acct1",
                question="Q", sql="SELECT 1",
            )
            pid2 = self._call(
                candidate_id="cand_b", account_id="acct1",
                question="Q", sql="SELECT 1",
            )
        self.assertNotEqual(pid1, pid2)

    def test_empty_question_skips_upsert_returns_empty(self):
        client = _make_qdrant_client()
        with patch("core.governed_store._qdrant", return_value=client), \
             patch("core.governed_store._embed", return_value=[_FAKE_VECTOR]):
            pid = self._call(
                candidate_id="c1", account_id="acct1",
                question="", sql="SELECT 1",
            )
        self.assertEqual(pid, "")
        client.upsert.assert_not_called()

    def test_empty_sql_skips_upsert_returns_empty(self):
        client = _make_qdrant_client()
        with patch("core.governed_store._qdrant", return_value=client), \
             patch("core.governed_store._embed", return_value=[_FAKE_VECTOR]):
            pid = self._call(
                candidate_id="c1", account_id="acct1",
                question="What is revenue?", sql="",
            )
        self.assertEqual(pid, "")
        client.upsert.assert_not_called()

    def test_whitespace_only_question_skips_upsert(self):
        client = _make_qdrant_client()
        with patch("core.governed_store._qdrant", return_value=client), \
             patch("core.governed_store._embed", return_value=[_FAKE_VECTOR]):
            pid = self._call(
                candidate_id="c1", account_id="acct1",
                question="   ", sql="SELECT 1",
            )
        self.assertEqual(pid, "")

    def test_creates_collection_if_not_exists(self):
        client = _make_qdrant_client(collections=[])   # empty collection list
        with patch("core.governed_store._qdrant", return_value=client), \
             patch("core.governed_store._embed", return_value=[_FAKE_VECTOR]):
            self._call(
                candidate_id="c1", account_id="acct1",
                question="Q", sql="SELECT 1",
            )
        client.create_collection.assert_called_once()

    def test_skips_collection_creation_if_already_exists(self):
        client = _make_qdrant_client(collections=["querybot_governed"])
        with patch("core.governed_store._qdrant", return_value=client), \
             patch("core.governed_store._embed", return_value=[_FAKE_VECTOR]):
            self._call(
                candidate_id="c1", account_id="acct1",
                question="Q", sql="SELECT 1",
            )
        client.create_collection.assert_not_called()

    def test_payload_contains_all_required_fields(self):
        client = _make_qdrant_client()
        captured_point = []

        def _capture_upsert(**kwargs):
            captured_point.extend(kwargs.get("points", []))

        client.upsert.side_effect = _capture_upsert

        with patch("core.governed_store._qdrant", return_value=client), \
             patch("core.governed_store._embed", return_value=[_FAKE_VECTOR]):
            self._call(
                candidate_id="c1",
                account_id="acct1",
                question="What is revenue?",
                sql="SELECT SUM(rev) FROM sales",
                source="admin_correction",
                final_score=90,
            )

        self.assertEqual(len(captured_point), 1)
        p = captured_point[0]
        payload = p.payload
        for field in ("candidate_id", "account_id", "question", "sql",
                      "source", "final_score", "doc_type", "ts"):
            self.assertIn(field, payload, f"Missing payload field: {field}")
        self.assertEqual(payload["doc_type"], "governed_example")
        self.assertEqual(payload["source"], "admin_correction")
        self.assertEqual(payload["final_score"], 90)

    def test_upsert_called_exactly_once_per_call(self):
        client = _make_qdrant_client()
        with patch("core.governed_store._qdrant", return_value=client), \
             patch("core.governed_store._embed", return_value=[_FAKE_VECTOR]):
            self._call(
                candidate_id="c1", account_id="acct1",
                question="Q", sql="SELECT 1",
            )
        self.assertEqual(client.upsert.call_count, 1)

    def test_uses_correct_collection_name(self):
        client = _make_qdrant_client()
        with patch("core.governed_store._qdrant", return_value=client), \
             patch("core.governed_store._embed", return_value=[_FAKE_VECTOR]):
            self._call(
                candidate_id="c1", account_id="acct1",
                question="Q", sql="SELECT 1",
            )
        call_kwargs = client.upsert.call_args
        self.assertEqual(
            call_kwargs.kwargs.get("collection_name") or call_kwargs.args[0],
            "querybot_governed",
        )


# ===========================================================================
# core/governed_store -- delete_governed_example
# ===========================================================================

class TestDeleteGovernedExample(unittest.TestCase):

    def _call(self, candidate_id, account_id="acct1"):
        from core.governed_store import delete_governed_example
        return delete_governed_example(candidate_id, account_id)

    def test_delete_called_with_correct_point_id(self):
        client = _make_qdrant_client()
        with patch("core.governed_store._qdrant", return_value=client):
            self._call("cand_xyz")

        client.delete.assert_called_once()
        # Verify the point ID matches the deterministic formula
        expected_pid = _expected_point_id("cand_xyz")
        args = client.delete.call_args
        points_selector = args.kwargs.get("points_selector") or args.args[1]
        # PointIdsList.points contains the IDs
        self.assertIn(expected_pid, points_selector.points)

    def test_delete_on_correct_collection(self):
        client = _make_qdrant_client()
        with patch("core.governed_store._qdrant", return_value=client):
            self._call("cand_xyz")

        args = client.delete.call_args
        coll = args.kwargs.get("collection_name") or args.args[0]
        self.assertEqual(coll, "querybot_governed")

    def test_delete_noop_on_qdrant_error(self):
        """delete_governed_example must not raise when Qdrant is unavailable."""
        client = _make_qdrant_client()
        client.delete.side_effect = ConnectionError("Qdrant down")
        with patch("core.governed_store._qdrant", return_value=client):
            try:
                self._call("cand_xyz")   # must not raise
            except Exception as exc:
                self.fail(f"delete_governed_example raised: {exc}")

    def test_point_id_is_deterministic_from_candidate_id(self):
        """The point ID used in delete must match the one from upsert."""
        from core.governed_store import _governed_point_id
        pid_a = _governed_point_id("candidate_001")
        pid_b = _governed_point_id("candidate_001")
        self.assertEqual(pid_a, pid_b)


# ===========================================================================
# core/governed_store -- retrieve_governed_examples
# ===========================================================================

class TestRetrieveGovernedExamples(unittest.TestCase):

    def _call(self, account_id="acct1", question="revenue?", n=3):
        from core.governed_store import retrieve_governed_examples
        return retrieve_governed_examples(account_id, question, n=n)

    def test_returns_empty_when_collection_does_not_exist(self):
        client = _make_qdrant_client(collections=[])   # governed not in list
        with patch("core.governed_store._qdrant", return_value=client):
            result = self._call()
        self.assertEqual(result, [])

    def test_returns_empty_when_count_is_zero(self):
        client = _make_qdrant_client(collections=["querybot_governed"], count=0)
        with patch("core.governed_store._qdrant", return_value=client):
            result = self._call()
        self.assertEqual(result, [])
        client.query_points.assert_not_called()

    def test_returns_empty_on_qdrant_unavailable(self):
        client = MagicMock()
        client.get_collections.side_effect = ConnectionError("Qdrant down")
        with patch("core.governed_store._qdrant", return_value=client):
            result = self._call()   # must not raise
        self.assertEqual(result, [])

    def test_returns_empty_on_query_points_error(self):
        client = _make_qdrant_client(count=5)
        client.query_points.side_effect = RuntimeError("index error")
        with patch("core.governed_store._qdrant", return_value=client), \
             patch("core.governed_store._embed", return_value=[_FAKE_VECTOR]):
            result = self._call()
        self.assertEqual(result, [])

    def test_returns_results_with_correct_keys(self):
        hits = [_make_hit("What is revenue?", "SELECT SUM(rev) FROM sales")]
        client = _make_qdrant_client(count=1, search_results=hits)
        with patch("core.governed_store._qdrant", return_value=client), \
             patch("core.governed_store._embed", return_value=[_FAKE_VECTOR]):
            result = self._call()

        self.assertEqual(len(result), 1)
        for key in ("question", "sql", "table", "source"):
            self.assertIn(key, result[0], f"Missing key: {key}")

    def test_table_field_is_empty_string(self):
        hits = [_make_hit("Q", "SELECT 1")]
        client = _make_qdrant_client(count=1, search_results=hits)
        with patch("core.governed_store._qdrant", return_value=client), \
             patch("core.governed_store._embed", return_value=[_FAKE_VECTOR]):
            result = self._call()
        self.assertEqual(result[0]["table"], "")

    def test_skips_hits_with_empty_sql(self):
        hits = [
            _make_hit("Q1", "SELECT 1"),
            _make_hit("Q2", ""),       # missing sql
        ]
        client = _make_qdrant_client(count=2, search_results=hits)
        with patch("core.governed_store._qdrant", return_value=client), \
             patch("core.governed_store._embed", return_value=[_FAKE_VECTOR]):
            result = self._call()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["question"], "Q1")

    def test_skips_hits_with_empty_question(self):
        hits = [
            _make_hit("", "SELECT 1"),   # missing question
            _make_hit("Q2", "SELECT 2"),
        ]
        client = _make_qdrant_client(count=2, search_results=hits)
        with patch("core.governed_store._qdrant", return_value=client), \
             patch("core.governed_store._embed", return_value=[_FAKE_VECTOR]):
            result = self._call()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["question"], "Q2")

    def test_respects_n_limit(self):
        hits = [_make_hit(f"Q{i}", f"SELECT {i}") for i in range(5)]
        client = _make_qdrant_client(count=5, search_results=hits)
        with patch("core.governed_store._qdrant", return_value=client), \
             patch("core.governed_store._embed", return_value=[_FAKE_VECTOR]):
            result = self._call(n=2)
        # Qdrant limit is passed as `limit=n`; mock returns all 5 but result set should match
        # what query_points was called with
        call_kwargs = client.query_points.call_args.kwargs
        self.assertEqual(call_kwargs.get("limit"), 2)


# ===========================================================================
# core/governed_store -- get_governed_count
# ===========================================================================

class TestGetGovernedCount(unittest.TestCase):

    def _call(self, account_id="acct1"):
        from core.governed_store import get_governed_count
        return get_governed_count(account_id)

    def test_returns_zero_when_collection_missing(self):
        client = _make_qdrant_client(collections=[])
        with patch("core.governed_store._qdrant", return_value=client):
            result = self._call()
        self.assertEqual(result, 0)

    def test_returns_correct_count(self):
        client = _make_qdrant_client(count=7)
        with patch("core.governed_store._qdrant", return_value=client):
            result = self._call()
        self.assertEqual(result, 7)

    def test_returns_zero_on_exception(self):
        client = MagicMock()
        client.get_collections.side_effect = ConnectionError("gone")
        with patch("core.governed_store._qdrant", return_value=client):
            result = self._call()   # must not raise
        self.assertEqual(result, 0)


# ===========================================================================
# core/governed_store -- backfill_approved_candidates
# ===========================================================================

class TestBackfillApprovedCandidates(unittest.TestCase):

    def _call(self, account_id="acct1"):
        from core.governed_store import backfill_approved_candidates
        return backfill_approved_candidates(account_id)

    def test_returns_zero_if_no_approved_candidates(self):
        with patch("store.learning_store.list_candidates", return_value=[]):
            result = self._call()
        self.assertEqual(result, 0)

    def test_upserts_each_approved_candidate(self):
        candidates = [
            {"candidate_id": "c1", "account_id": "acct1", "question_text": "Q1",
             "sql_text": "SELECT 1", "corrected_sql": "", "source": "auto", "final_score": 88},
            {"candidate_id": "c2", "account_id": "acct1", "question_text": "Q2",
             "sql_text": "SELECT 2", "corrected_sql": "", "source": "auto", "final_score": 91},
        ]
        upsert_calls = []

        def _fake_upsert(**kwargs):
            upsert_calls.append(kwargs["candidate_id"])
            return "fake-uuid"

        with (
            patch("store.learning_store.list_candidates", return_value=candidates),
            patch("core.governed_store.upsert_governed_example", side_effect=_fake_upsert),
        ):
            count = self._call()

        self.assertEqual(count, 2)
        self.assertIn("c1", upsert_calls)
        self.assertIn("c2", upsert_calls)

    def test_prefers_corrected_sql_over_sql_text(self):
        candidates = [{
            "candidate_id": "c1", "account_id": "acct1", "question_text": "Q1",
            "sql_text": "SELECT 1",
            "corrected_sql": "SELECT 1 corrected",
            "source": "admin_correction", "final_score": 85,
        }]
        captured = {}

        def _fake_upsert(**kwargs):
            captured.update(kwargs)
            return "pid"

        with (
            patch("store.learning_store.list_candidates", return_value=candidates),
            patch("core.governed_store.upsert_governed_example", side_effect=_fake_upsert),
        ):
            self._call()

        self.assertEqual(captured.get("sql"), "SELECT 1 corrected")

    def test_skips_candidates_with_no_sql(self):
        candidates = [{
            "candidate_id": "c1", "account_id": "acct1", "question_text": "Q1",
            "sql_text": "", "corrected_sql": "", "source": "auto", "final_score": 70,
        }]
        upsert_calls = []

        with (
            patch("store.learning_store.list_candidates", return_value=candidates),
            patch("core.governed_store.upsert_governed_example", side_effect=lambda **kw: upsert_calls.append(kw)),
        ):
            count = self._call()

        self.assertEqual(count, 0)
        self.assertEqual(len(upsert_calls), 0)

    def test_handles_individual_upsert_failures_gracefully(self):
        candidates = [
            {"candidate_id": "c1", "account_id": "acct1", "question_text": "Q1",
             "sql_text": "SELECT 1", "corrected_sql": "", "source": "auto", "final_score": 88},
            {"candidate_id": "c2", "account_id": "acct1", "question_text": "Q2",
             "sql_text": "SELECT 2", "corrected_sql": "", "source": "auto", "final_score": 90},
        ]
        call_count = [0]

        def _flaky_upsert(**kwargs):
            call_count[0] += 1
            if kwargs["candidate_id"] == "c1":
                raise RuntimeError("vector store error")
            return "pid"

        with (
            patch("store.learning_store.list_candidates", return_value=candidates),
            patch("core.governed_store.upsert_governed_example", side_effect=_flaky_upsert),
        ):
            count = self._call()   # must not raise

        # c1 failed, c2 succeeded -> count=1
        self.assertEqual(count, 1)


# ===========================================================================
# store/learning_store -- governed hooks in update_candidate_status
# ===========================================================================

class TestUpdateCandidateStatusGovernedHooks(unittest.TestCase):
    """
    Verify that update_candidate_status fires the correct governed-collection
    hooks after the DB write, and that those hooks are best-effort (failures
    never propagate to the caller).
    """

    def test_approve_fires_upsert_hook(self):
        fired = [False]

        def _mock_upsert(candidate_id):
            fired[0] = True

        with (
            patch("store.learning_store.get_db"),
            patch("store.learning_store.get_candidate", return_value=None),
            patch("store.learning_store._fire_governed_upsert", side_effect=_mock_upsert),
            patch("store.learning_store._fire_governed_delete"),
        ):
            from store.learning_store import update_candidate_status
            update_candidate_status("cand1", "approved", reviewer_id="admin")

        self.assertTrue(fired[0], "Approve must fire _fire_governed_upsert")

    def test_reject_does_not_fire_any_governed_hook(self):
        upsert_called = [False]
        delete_called = [False]

        with (
            patch("store.learning_store.get_db"),
            patch("store.learning_store._fire_governed_upsert",
                  side_effect=lambda *a, **kw: upsert_called.__setitem__(0, True)),
            patch("store.learning_store._fire_governed_delete",
                  side_effect=lambda *a, **kw: delete_called.__setitem__(0, True)),
        ):
            from store.learning_store import update_candidate_status
            update_candidate_status("cand1", "rejected")

        self.assertFalse(upsert_called[0], "Reject must NOT fire upsert hook")
        self.assertFalse(delete_called[0], "Reject must NOT fire delete hook")

    def test_revoke_fires_delete_hook(self):
        fired = [False]

        def _mock_delete(candidate_id):
            fired[0] = True

        with (
            patch("store.learning_store.get_db"),
            patch("store.learning_store._fire_governed_upsert"),
            patch("store.learning_store._fire_governed_delete", side_effect=_mock_delete),
        ):
            from store.learning_store import update_candidate_status
            update_candidate_status("cand1", "revoked")

        self.assertTrue(fired[0], "Revoke must fire _fire_governed_delete")

    def test_approve_hook_failure_does_not_propagate(self):
        """_fire_governed_upsert raising must not cause update_candidate_status to raise."""
        with (
            patch("store.learning_store.get_db"),
            patch("store.learning_store._fire_governed_upsert",
                  side_effect=RuntimeError("qdrant down")),
        ):
            from store.learning_store import update_candidate_status
            try:
                update_candidate_status("cand1", "approved")
            except RuntimeError:
                self.fail("Governed hook failure must not propagate from update_candidate_status")

    def test_known_failure_status_no_governed_hook(self):
        upsert_called = [False]
        delete_called = [False]

        with (
            patch("store.learning_store.get_db"),
            patch("store.learning_store._fire_governed_upsert",
                  side_effect=lambda *a: upsert_called.__setitem__(0, True)),
            patch("store.learning_store._fire_governed_delete",
                  side_effect=lambda *a: delete_called.__setitem__(0, True)),
        ):
            from store.learning_store import update_candidate_status
            update_candidate_status("cand1", "known_failure")

        self.assertFalse(upsert_called[0])
        self.assertFalse(delete_called[0])

    def test_invalid_status_raises_value_error(self):
        with patch("store.learning_store.get_db"):
            from store.learning_store import update_candidate_status
            with self.assertRaises(ValueError):
                update_candidate_status("cand1", "explode")


# ===========================================================================
# store/learning_store -- _fire_governed_upsert internals
# ===========================================================================

class TestFireGovernedUpsert(unittest.TestCase):

    def _call(self, candidate_id="cand1"):
        from store.learning_store import _fire_governed_upsert
        return _fire_governed_upsert(candidate_id)

    def test_noop_when_candidate_not_found(self):
        """_fire_governed_upsert is a no-op (no error) when candidate doesn't exist."""
        upsert_called = [False]

        def _fake_upsert(**kw):
            upsert_called[0] = True
            return "pid"

        with (
            patch("store.learning_store.get_candidate", return_value=None),
            patch("core.governed_store.upsert_governed_example", side_effect=_fake_upsert),
        ):
            self._call()   # must not raise

        self.assertFalse(upsert_called[0])

    def test_uses_corrected_sql_when_available(self):
        candidate = {
            "candidate_id": "cand1", "account_id": "acct1",
            "question_text": "Q1",
            "sql_text": "SELECT 1 (bad)",
            "corrected_sql": "SELECT 1 (fixed)",
            "source": "admin_correction", "final_score": 85,
        }
        captured = {}

        def _fake_upsert(**kw):
            captured.update(kw)
            return "pid"

        with (
            patch("store.learning_store.get_candidate", return_value=candidate),
            patch("store.learning_store.get_db"),
            patch("core.governed_store.upsert_governed_example", side_effect=_fake_upsert),
        ):
            self._call()

        self.assertEqual(captured.get("sql"), "SELECT 1 (fixed)")

    def test_falls_back_to_sql_text_when_corrected_empty(self):
        candidate = {
            "candidate_id": "cand1", "account_id": "acct1",
            "question_text": "Q1",
            "sql_text": "SELECT SUM(rev)",
            "corrected_sql": "",
            "source": "auto", "final_score": 88,
        }
        captured = {}

        def _fake_upsert(**kw):
            captured.update(kw)
            return "pid"

        with (
            patch("store.learning_store.get_candidate", return_value=candidate),
            patch("store.learning_store.get_db"),
            patch("core.governed_store.upsert_governed_example", side_effect=_fake_upsert),
        ):
            self._call()

        self.assertEqual(captured.get("sql"), "SELECT SUM(rev)")

    def test_writes_qdrant_id_back_to_db(self):
        candidate = {
            "candidate_id": "cand1", "account_id": "acct1",
            "question_text": "Q1", "sql_text": "SELECT 1",
            "corrected_sql": "", "source": "auto", "final_score": 90,
        }
        executed_sqls = []
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = lambda sql, *args: executed_sqls.append(sql)
        from contextlib import contextmanager

        @contextmanager
        def _fake_db():
            yield mock_conn

        with (
            patch("store.learning_store.get_candidate", return_value=candidate),
            patch("store.learning_store.get_db", side_effect=_fake_db),
            patch("core.governed_store.upsert_governed_example", return_value="the-qdrant-id"),
        ):
            self._call()

        # At least one UPDATE with qdrant_id should have been executed
        update_calls = [s for s in executed_sqls if "qdrant_id" in s.lower()]
        self.assertTrue(len(update_calls) > 0, "Should write qdrant_id back to DB")

    def test_noop_when_sql_is_empty(self):
        candidate = {
            "candidate_id": "cand1", "account_id": "acct1",
            "question_text": "Q1", "sql_text": "",
            "corrected_sql": "", "source": "auto", "final_score": 0,
        }
        upsert_called = [False]

        with (
            patch("store.learning_store.get_candidate", return_value=candidate),
            patch("core.governed_store.upsert_governed_example",
                  side_effect=lambda **kw: upsert_called.__setitem__(0, True)),
        ):
            self._call()

        self.assertFalse(upsert_called[0])

    def test_exception_does_not_propagate(self):
        with (
            patch("store.learning_store.get_candidate",
                  side_effect=RuntimeError("db error")),
        ):
            try:
                self._call()
            except Exception as exc:
                self.fail(f"_fire_governed_upsert must not raise: {exc}")


# ===========================================================================
# core/examples.py -- dual-collection retrieve_similar_examples
# ===========================================================================

class TestDualRetrieveSimilarExamples(unittest.TestCase):

    def _call(self, question="revenue?", account_id="acct1", n=3):
        from core.examples import retrieve_similar_examples
        return retrieve_similar_examples(question, account_id, n=n)

    _LEGACY = [
        {"question": "Total revenue?", "sql": "SELECT SUM(rev)", "table": "sales"},
    ]
    _GOVERNED = [
        {"question": "Revenue this quarter?", "sql": "SELECT SUM(rev) WHERE qtr=...", "table": "", "source": "admin_correction"},
    ]

    def test_returns_governed_first(self):
        with (
            patch("core.vector_store.retrieve_similar_examples", return_value=self._LEGACY),
            patch("core.governed_store.retrieve_governed_examples", return_value=self._GOVERNED),
        ):
            result = self._call()

        self.assertTrue(len(result) >= 1)
        # First result should be from governed
        self.assertIn(result[0]["question"], [g["question"] for g in self._GOVERNED])

    def test_legacy_fills_remaining_slots(self):
        governed = [{"question": "G1", "sql": "SELECT 1", "table": "", "source": "auto"}]
        legacy   = [
            {"question": "L1", "sql": "SELECT 2", "table": "t"},
            {"question": "L2", "sql": "SELECT 3", "table": "t"},
        ]
        with (
            patch("core.vector_store.retrieve_similar_examples", return_value=legacy),
            patch("core.governed_store.retrieve_governed_examples", return_value=governed),
        ):
            result = self._call(n=3)

        questions = [r["question"] for r in result]
        self.assertIn("G1", questions)
        self.assertIn("L1", questions)
        self.assertIn("L2", questions)

    def test_deduplicates_same_question(self):
        """Same question in both collections should appear only once."""
        same_q = "What is total revenue?"
        governed = [{"question": same_q, "sql": "SELECT SUM(rev) v2", "table": "", "source": "auto"}]
        legacy   = [{"question": same_q, "sql": "SELECT SUM(rev) v1", "table": "t"}]
        with (
            patch("core.vector_store.retrieve_similar_examples", return_value=legacy),
            patch("core.governed_store.retrieve_governed_examples", return_value=governed),
        ):
            result = self._call(n=3)

        occurrences = sum(1 for r in result if r["question"].lower() == same_q.lower())
        self.assertEqual(occurrences, 1, "Same question should appear at most once")

    def test_governed_version_wins_dedup(self):
        """When same question in both, governed example is kept (appears first)."""
        same_q = "What is revenue?"
        governed = [{"question": same_q, "sql": "SELECT SUM(rev) CORRECTED", "table": "", "source": "admin_correction"}]
        legacy   = [{"question": same_q, "sql": "SELECT SUM(rev) ORIGINAL", "table": "t"}]
        with (
            patch("core.vector_store.retrieve_similar_examples", return_value=legacy),
            patch("core.governed_store.retrieve_governed_examples", return_value=governed),
        ):
            result = self._call(n=3)

        matched = [r for r in result if r["question"].lower() == same_q.lower()]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["sql"], "SELECT SUM(rev) CORRECTED")

    def test_returns_only_legacy_when_governed_empty(self):
        with (
            patch("core.vector_store.retrieve_similar_examples", return_value=self._LEGACY),
            patch("core.governed_store.retrieve_governed_examples", return_value=[]),
        ):
            result = self._call()

        self.assertEqual(result, self._LEGACY[:3])

    def test_returns_only_governed_when_legacy_empty(self):
        with (
            patch("core.vector_store.retrieve_similar_examples", return_value=[]),
            patch("core.governed_store.retrieve_governed_examples", return_value=self._GOVERNED),
        ):
            result = self._call()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["question"], self._GOVERNED[0]["question"])

    def test_returns_empty_when_both_empty(self):
        with (
            patch("core.vector_store.retrieve_similar_examples", return_value=[]),
            patch("core.governed_store.retrieve_governed_examples", return_value=[]),
        ):
            result = self._call()

        self.assertEqual(result, [])

    def test_governed_failure_falls_back_to_legacy(self):
        with (
            patch("core.vector_store.retrieve_similar_examples", return_value=self._LEGACY),
            patch("core.governed_store.retrieve_governed_examples",
                  side_effect=ConnectionError("Qdrant down")),
        ):
            result = self._call()

        self.assertEqual(result, self._LEGACY[:3])

    def test_legacy_failure_falls_back_to_governed(self):
        with (
            patch("core.vector_store.retrieve_similar_examples",
                  side_effect=RuntimeError("vector store error")),
            patch("core.governed_store.retrieve_governed_examples", return_value=self._GOVERNED),
        ):
            result = self._call()

        self.assertEqual(len(result), 1)

    def test_respects_n_limit(self):
        governed = [{"question": f"G{i}", "sql": f"SELECT {i}", "table": "", "source": "auto"} for i in range(5)]
        legacy   = [{"question": f"L{i}", "sql": f"SELECT {i}", "table": "t"} for i in range(5)]
        with (
            patch("core.vector_store.retrieve_similar_examples", return_value=legacy),
            patch("core.governed_store.retrieve_governed_examples", return_value=governed),
        ):
            result = self._call(n=3)

        self.assertLessEqual(len(result), 3)

    def test_legacy_path_string_normalised_to_account_id(self):
        """A filesystem-style path 'clients/my_tenant' -> parts[1] = 'my_tenant'."""
        captured = {}

        def _fake_legacy(account_id, question, n=3, allowed_tables=None):
            captured["account_id"] = account_id
            return []

        with (
            patch("core.vector_store.retrieve_similar_examples", side_effect=_fake_legacy),
            patch("core.governed_store.retrieve_governed_examples", return_value=[]),
        ):
            from core.examples import retrieve_similar_examples
            # Path("clients/my_tenant").parts = ('clients', 'my_tenant') -> parts[1]='my_tenant'
            retrieve_similar_examples("revenue?", "clients/my_tenant", n=2)

        self.assertEqual(captured.get("account_id"), "my_tenant")


if __name__ == "__main__":
    unittest.main()
