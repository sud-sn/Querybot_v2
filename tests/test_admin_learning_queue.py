"""
tests/test_admin_learning_queue.py

Unit tests for the Admin Learning Queue routes:
  GET  /admin/clients/{account_id}/learning-queue
  POST /admin/clients/{account_id}/learning-queue/{candidate_id}/review
  POST /admin/clients/{account_id}/learning-queue/{candidate_id}/correct-sql
  GET  /admin/api/clients/{account_id}/learning-queue/count

Strategy: direct unit tests against the async route handlers using mocked
dependencies — no FastAPI TestClient, no SQLite, no network.
"""

import asyncio
import json
import unittest
from unittest.mock import MagicMock, patch, AsyncMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arun(coro):
    """Run a coroutine in a fresh event loop (works on Python 3.10-3.14)."""
    return asyncio.run(coro)


def _make_get_request(query_params: dict | None = None, authed: bool = True):
    """Build a minimal mock FastAPI GET Request."""
    req = MagicMock()
    req.query_params = query_params or {}
    req.session = {"admin_id": "admin_user_1"}
    return req, authed


def _make_post_request(form_data: dict | None = None, authed: bool = True):
    """Build a minimal mock FastAPI POST Request with form data."""
    req = MagicMock()
    req.query_params = {}
    req.session = {"admin_id": "admin_user_1"}

    async def _form():
        return form_data or {}

    req.form = _form
    return req, authed


_MOCK_CLIENT = {
    "account_id": "acct1",
    "client_name": "Test Corp",
    "state": "READY",
    "enable_feedback_collection": 1,
}

_MOCK_CLIENT_FLAG_OFF = {
    "account_id": "acct1",
    "client_name": "Test Corp",
    "state": "READY",
    "enable_feedback_collection": 0,
}

_MOCK_CANDIDATES = [
    {
        "candidate_id": "abc123def456",
        "account_id": "acct1",
        "question_text": "What is total revenue?",
        "sql_text": "SELECT SUM(revenue) FROM sales",
        "final_score": 90,
        "candidate_type": "positive",
        "status": "pending_review",
        "source": "live_query",
        "positive_vote_count": 2,
        "negative_vote_count": 0,
        "created_at": "2026-06-01T10:00:00",
        "promoted_at": None,
        "reviewed_at": None,
        "reviewer_note": "",
        "corrected_sql": None,
        "evidence": '{"validation_passed": true}',
    },
    {
        "candidate_id": "xyz789uvw012",
        "account_id": "acct1",
        "question_text": "Monthly active users?",
        "sql_text": "SELECT COUNT(DISTINCT user_id) FROM sessions",
        "final_score": 55,
        "candidate_type": "negative",
        "status": "pending_review",
        "source": "live_query",
        "positive_vote_count": 0,
        "negative_vote_count": 3,
        "created_at": "2026-06-02T09:00:00",
        "promoted_at": None,
        "reviewed_at": None,
        "reviewer_note": "",
        "corrected_sql": None,
        "evidence": None,
    },
]


# ---------------------------------------------------------------------------
# GET /admin/clients/{account_id}/learning-queue
# ---------------------------------------------------------------------------

class TestAdminLearningQueueGet(unittest.TestCase):

    def _call(self, request, account_id="acct1"):
        from admin.routes import admin_learning_queue
        return _arun(admin_learning_queue(request, account_id))

    def test_unauthed_redirects_to_login(self):
        req, _ = _make_get_request()
        with patch("admin.routes._is_auth", return_value=False):
            result = self._call(req)
        self.assertEqual(result.status_code, 303)
        self.assertIn("/admin/login", result.headers["location"])

    def test_unknown_client_redirects(self):
        req, _ = _make_get_request()
        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=None),
        ):
            result = self._call(req)
        self.assertEqual(result.status_code, 303)
        self.assertIn("/admin/clients", result.headers["location"])

    def test_happy_path_returns_200(self):
        req, _ = _make_get_request({"status": "pending_review"})
        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.list_candidates", return_value=_MOCK_CANDIDATES),
            patch("admin.routes._resp") as mock_resp,
        ):
            mock_resp.return_value = MagicMock(status_code=200)
            result = self._call(req)
        self.assertEqual(result.status_code, 200)

    def test_template_context_contains_expected_keys(self):
        req, _ = _make_get_request({"status": "all"})
        ctx_captured = {}

        def _fake_resp(request, name, ctx=None):
            ctx_captured.update(ctx or {})
            r = MagicMock()
            r.status_code = 200
            return r

        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.list_candidates", return_value=_MOCK_CANDIDATES),
            patch("admin.routes._resp", side_effect=_fake_resp),
        ):
            self._call(req)

        for key in ("client", "candidates", "status_filter", "stats", "flag_enabled"):
            self.assertIn(key, ctx_captured, f"Missing key: {key}")

    def test_flag_off_sets_flag_enabled_false(self):
        req, _ = _make_get_request()
        ctx_captured = {}

        def _fake_resp(request, name, ctx=None):
            ctx_captured.update(ctx or {})
            r = MagicMock()
            r.status_code = 200
            return r

        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT_FLAG_OFF),
            patch("store.learning_store.list_candidates", return_value=[]),
            patch("admin.routes._resp", side_effect=_fake_resp),
        ):
            self._call(req)

        self.assertFalse(ctx_captured.get("flag_enabled"))

    def test_flag_on_sets_flag_enabled_true(self):
        req, _ = _make_get_request()
        ctx_captured = {}

        def _fake_resp(request, name, ctx=None):
            ctx_captured.update(ctx or {})
            r = MagicMock()
            r.status_code = 200
            return r

        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.list_candidates", return_value=[]),
            patch("admin.routes._resp", side_effect=_fake_resp),
        ):
            self._call(req)

        self.assertTrue(ctx_captured.get("flag_enabled"))

    def test_status_filter_all_passes_none_to_list_candidates(self):
        req, _ = _make_get_request({"status": "all"})
        calls = []

        def _fake_list(account_id, status=None, limit=50, offset=0):
            calls.append({"account_id": account_id, "status": status})
            return []

        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.list_candidates", side_effect=_fake_list),
            patch("admin.routes._resp", return_value=MagicMock(status_code=200)),
        ):
            self._call(req)

        # The first call (main page load, limit=100) should have status=None
        main_calls = [c for c in calls if c.get("status") is None]
        self.assertTrue(len(main_calls) > 0, "Expected at least one call with status=None for 'all' filter")

    def test_evidence_json_string_parsed_to_dict(self):
        req, _ = _make_get_request()
        ctx_captured = {}

        def _fake_resp(request, name, ctx=None):
            ctx_captured.update(ctx or {})
            r = MagicMock()
            r.status_code = 200
            return r

        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.list_candidates", return_value=list(_MOCK_CANDIDATES)),
            patch("admin.routes._resp", side_effect=_fake_resp),
        ):
            self._call(req)

        candidates = ctx_captured.get("candidates", [])
        if candidates:
            # The first candidate has evidence as a JSON string — should be parsed
            ev = candidates[0].get("evidence")
            self.assertIsInstance(ev, dict, "Evidence should be parsed from JSON string to dict")

    def test_evidence_none_becomes_empty_dict(self):
        req, _ = _make_get_request()
        ctx_captured = {}

        def _fake_resp(request, name, ctx=None):
            ctx_captured.update(ctx or {})
            r = MagicMock()
            r.status_code = 200
            return r

        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.list_candidates", return_value=list(_MOCK_CANDIDATES)),
            patch("admin.routes._resp", side_effect=_fake_resp),
        ):
            self._call(req)

        candidates = ctx_captured.get("candidates", [])
        second = next((c for c in candidates if c.get("candidate_id") == "xyz789uvw012"), None)
        if second:
            self.assertIsInstance(second.get("evidence"), dict)

    def test_default_status_filter_is_pending_review(self):
        req, _ = _make_get_request({})   # no "status" key
        ctx_captured = {}

        def _fake_resp(request, name, ctx=None):
            ctx_captured.update(ctx or {})
            r = MagicMock()
            r.status_code = 200
            return r

        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.list_candidates", return_value=[]),
            patch("admin.routes._resp", side_effect=_fake_resp),
        ):
            self._call(req)

        self.assertEqual(ctx_captured.get("status_filter"), "pending_review")


# ---------------------------------------------------------------------------
# POST /admin/clients/{account_id}/learning-queue/{candidate_id}/review
# ---------------------------------------------------------------------------

class TestAdminLearningQueueReview(unittest.TestCase):

    def _call(self, request, account_id="acct1", candidate_id="abc123def456"):
        from admin.routes import admin_learning_queue_review
        return _arun(admin_learning_queue_review(request, account_id, candidate_id))

    def test_unauthed_redirects_to_login(self):
        req, _ = _make_post_request({"action": "approve"})
        with patch("admin.routes._is_auth", return_value=False):
            result = self._call(req)
        self.assertEqual(result.status_code, 303)
        self.assertIn("/admin/login", result.headers["location"])

    def test_unknown_client_redirects(self):
        req, _ = _make_post_request({"action": "approve"})
        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=None),
        ):
            result = self._call(req)
        self.assertEqual(result.status_code, 303)
        self.assertIn("/admin/clients", result.headers["location"])

    def test_approve_calls_update_candidate_status(self):
        req, _ = _make_post_request({"action": "approve"})
        calls = {}

        def _fake_update(candidate_id, status, reviewer_id="", reviewer_note=""):
            calls["candidate_id"] = candidate_id
            calls["status"] = status

        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.update_candidate_status", side_effect=_fake_update),
        ):
            result = self._call(req, candidate_id="abc123def456")

        self.assertEqual(result.status_code, 303)
        self.assertEqual(calls.get("candidate_id"), "abc123def456")
        self.assertEqual(calls.get("status"), "approved")

    def test_reject_calls_update_with_rejected(self):
        req, _ = _make_post_request({"action": "reject"})
        calls = {}

        def _fake_update(candidate_id, status, reviewer_id="", reviewer_note=""):
            calls["status"] = status

        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.update_candidate_status", side_effect=_fake_update),
        ):
            self._call(req)

        self.assertEqual(calls.get("status"), "rejected")

    def test_known_failure_calls_update_with_known_failure(self):
        req, _ = _make_post_request({"action": "known_failure"})
        calls = {}

        def _fake_update(candidate_id, status, reviewer_id="", reviewer_note=""):
            calls["status"] = status

        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.update_candidate_status", side_effect=_fake_update),
        ):
            self._call(req)

        self.assertEqual(calls.get("status"), "known_failure")

    def test_invalid_action_redirects_without_calling_store(self):
        req, _ = _make_post_request({"action": "explode"})
        called = [False]

        def _fake_update(*a, **kw):
            called[0] = True

        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.update_candidate_status", side_effect=_fake_update),
        ):
            result = self._call(req)

        self.assertFalse(called[0], "Store should NOT be called for invalid action")
        self.assertEqual(result.status_code, 303)
        self.assertIn("learning-queue", result.headers["location"])

    def test_review_redirects_back_to_queue(self):
        req, _ = _make_post_request({"action": "approve"})
        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.update_candidate_status"),
        ):
            result = self._call(req)

        self.assertEqual(result.status_code, 303)
        self.assertIn("learning-queue", result.headers["location"])

    def test_store_error_redirects_with_error_message(self):
        req, _ = _make_post_request({"action": "approve"})
        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.update_candidate_status", side_effect=Exception("db gone")),
        ):
            result = self._call(req)

        self.assertEqual(result.status_code, 303)
        loc = result.headers["location"]
        self.assertIn("learning-queue", loc)
        self.assertIn("Error", loc)


# ---------------------------------------------------------------------------
# POST /admin/clients/{account_id}/learning-queue/{candidate_id}/correct-sql
# ---------------------------------------------------------------------------

class TestAdminLearningQueueCorrectSql(unittest.TestCase):

    def _call(self, request, account_id="acct1", candidate_id="abc123def456"):
        from admin.routes import admin_learning_queue_correct_sql
        return _arun(admin_learning_queue_correct_sql(request, account_id, candidate_id))

    def test_unauthed_redirects_to_login(self):
        req, _ = _make_post_request({"corrected_sql": "SELECT 1"})
        with patch("admin.routes._is_auth", return_value=False):
            result = self._call(req)
        self.assertEqual(result.status_code, 303)
        self.assertIn("/admin/login", result.headers["location"])

    def test_unknown_client_redirects(self):
        req, _ = _make_post_request({"corrected_sql": "SELECT 1"})
        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=None),
        ):
            result = self._call(req)
        self.assertEqual(result.status_code, 303)
        self.assertIn("/admin/clients", result.headers["location"])

    def test_empty_sql_redirects_without_calling_store(self):
        req, _ = _make_post_request({"corrected_sql": ""})
        called = [False]

        def _fake_set(*a, **kw):
            called[0] = True

        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.set_candidate_corrected_sql", side_effect=_fake_set),
        ):
            result = self._call(req)

        self.assertFalse(called[0], "set_candidate_corrected_sql should NOT be called for empty SQL")
        self.assertEqual(result.status_code, 303)

    def test_valid_sql_calls_set_candidate_corrected_sql(self):
        req, _ = _make_post_request({"corrected_sql": "SELECT SUM(amount) FROM orders"})
        calls = {}

        def _fake_set(candidate_id, corrected_sql, reviewer_id=""):
            calls["candidate_id"] = candidate_id
            calls["corrected_sql"] = corrected_sql

        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.set_candidate_corrected_sql", side_effect=_fake_set),
        ):
            result = self._call(req, candidate_id="abc123def456")

        self.assertEqual(result.status_code, 303)
        self.assertEqual(calls.get("candidate_id"), "abc123def456")
        self.assertEqual(calls.get("corrected_sql"), "SELECT SUM(amount) FROM orders")

    def test_correct_sql_redirects_back_to_queue(self):
        req, _ = _make_post_request({"corrected_sql": "SELECT 1"})
        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.set_candidate_corrected_sql"),
        ):
            result = self._call(req)

        self.assertEqual(result.status_code, 303)
        self.assertIn("learning-queue", result.headers["location"])

    def test_store_error_redirects_with_error_message(self):
        req, _ = _make_post_request({"corrected_sql": "SELECT 1"})
        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.set_candidate_corrected_sql",
                  side_effect=Exception("integrity error")),
        ):
            result = self._call(req)

        self.assertEqual(result.status_code, 303)
        loc = result.headers["location"]
        self.assertIn("learning-queue", loc)
        self.assertIn("Error", loc)

    def test_reviewer_note_passed_to_update_status(self):
        req, _ = _make_post_request({
            "corrected_sql": "SELECT 1",
            "reviewer_note": "Fixed join condition",
        })
        update_calls = {}

        def _fake_update(candidate_id, status, reviewer_id="", reviewer_note=""):
            update_calls["reviewer_note"] = reviewer_note

        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.set_candidate_corrected_sql"),
            patch("store.learning_store.update_candidate_status", side_effect=_fake_update),
        ):
            self._call(req)

        self.assertEqual(update_calls.get("reviewer_note"), "Fixed join condition")

    def test_whitespace_only_sql_treated_as_empty(self):
        req, _ = _make_post_request({"corrected_sql": "   \t  "})
        called = [False]

        def _fake_set(*a, **kw):
            called[0] = True

        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.set_candidate_corrected_sql", side_effect=_fake_set),
        ):
            self._call(req)

        self.assertFalse(called[0], "Whitespace-only SQL should be treated as empty")


# ---------------------------------------------------------------------------
# GET /admin/api/clients/{account_id}/learning-queue/count
# ---------------------------------------------------------------------------

class TestAdminLearningQueueCount(unittest.TestCase):

    def _call(self, request, account_id="acct1"):
        from admin.routes import admin_learning_queue_count
        return _arun(admin_learning_queue_count(request, account_id))

    def test_unauthed_raises_401(self):
        req, _ = _make_get_request()
        with patch("admin.routes._is_auth", return_value=False):
            from fastapi import HTTPException
            with self.assertRaises(HTTPException) as ctx:
                self._call(req)
            self.assertEqual(ctx.exception.status_code, 401)

    def test_unknown_client_raises_404(self):
        req, _ = _make_get_request()
        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=None),
        ):
            from fastapi import HTTPException
            with self.assertRaises(HTTPException) as ctx:
                self._call(req)
            self.assertEqual(ctx.exception.status_code, 404)

    def test_returns_pending_count_in_json(self):
        req, _ = _make_get_request()
        pending_candidates = [
            {"candidate_id": "c1", "status": "pending_review"},
            {"candidate_id": "c2", "status": "pending_review"},
        ]
        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.list_candidates", return_value=pending_candidates),
        ):
            result = self._call(req)

        self.assertEqual(result.status_code, 200)
        data = json.loads(result.body)
        self.assertEqual(data["pending_review"], 2)
        self.assertEqual(data["account_id"], "acct1")

    def test_zero_pending_returns_zero(self):
        req, _ = _make_get_request()
        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.list_candidates", return_value=[]),
        ):
            result = self._call(req)

        data = json.loads(result.body)
        self.assertEqual(data["pending_review"], 0)

    def test_store_error_raises_500(self):
        req, _ = _make_get_request()
        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.list_candidates",
                  side_effect=RuntimeError("connection lost")),
        ):
            from fastapi import HTTPException
            with self.assertRaises(HTTPException) as ctx:
                self._call(req)
            self.assertEqual(ctx.exception.status_code, 500)

    def test_count_queries_pending_review_status(self):
        req, _ = _make_get_request()
        calls = []

        def _fake_list(account_id, status=None, limit=50, offset=0):
            calls.append(status)
            return []

        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client", return_value=_MOCK_CLIENT),
            patch("store.learning_store.list_candidates", side_effect=_fake_list),
        ):
            self._call(req)

        self.assertIn("pending_review", calls,
                      "list_candidates must be called with status='pending_review'")


if __name__ == "__main__":
    unittest.main()
