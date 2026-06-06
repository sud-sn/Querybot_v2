"""
tests/test_portal_feedback_api.py

Tests for POST /portal/api/answers/{question_id}/feedback

Strategy: direct unit tests against the async route handler using mocked
dependencies — no FastAPI TestClient, no SQLite, no network.  Pure function-
level mocking keeps the suite fast and free of production DB side-effects.
"""

import asyncio
import json
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arun(coro):
    """Run a coroutine in a fresh event loop (works on Python 3.10–3.14)."""
    return asyncio.run(coro)


def _make_request(body: dict, has_cookie: bool = True):
    """Build a minimal mock FastAPI Request with a fake session cookie."""
    req = MagicMock()
    req.cookies = {"qb_portal_session": "tok.sig"} if has_cookie else {}

    async def _json():
        return body

    req.json = _json
    return req


def _make_bad_json_request():
    """Request whose .json() raises (simulates malformed body)."""
    req = MagicMock()
    req.cookies = {"qb_portal_session": "tok.sig"}

    async def _json():
        raise ValueError("not json")

    req.json = _json
    return req


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_GOOD_USER = {
    "id": 1,
    "account_id": "acct1",
    "name": "Tester",
    "email": "t@t.com",
    "role": "analyst",
}
_CLIENT_ON  = {"enable_feedback_collection": 1}
_CLIENT_OFF = {"enable_feedback_collection": 0}


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestPortalFeedbackApi(unittest.TestCase):

    def _call(self, request, question_id="q1"):
        """Import and call the handler; returns the JSONResponse."""
        from portal.routes import portal_answer_feedback
        return _arun(portal_answer_feedback(request, question_id))

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_thumbs_up_returns_200_and_ok(self):
        req = _make_request({"rating": 1})
        with (
            patch("portal.routes._get_portal_user", return_value=_GOOD_USER),
            patch("portal.routes.store.get_client",  return_value=_CLIENT_ON),
            patch("store.learning_store.save_feedback", return_value={}),
        ):
            result = self._call(req)

        self.assertEqual(result.status_code, 200)
        data = json.loads(result.body)
        self.assertTrue(data["ok"])
        self.assertEqual(data["rating"], 1)

    def test_thumbs_down_returns_200_and_ok(self):
        req = _make_request({
            "rating": -1,
            "reason_code": "wrong_metric",
            "comment": "The numbers are off",
        })
        with (
            patch("portal.routes._get_portal_user", return_value=_GOOD_USER),
            patch("portal.routes.store.get_client",  return_value=_CLIENT_ON),
            patch("store.learning_store.save_feedback", return_value={}),
        ):
            result = self._call(req)

        self.assertEqual(result.status_code, 200)
        data = json.loads(result.body)
        self.assertTrue(data["ok"])
        self.assertEqual(data["rating"], -1)

    def test_positive_message_for_thumbs_up(self):
        req = _make_request({"rating": 1})
        with (
            patch("portal.routes._get_portal_user", return_value=_GOOD_USER),
            patch("portal.routes.store.get_client",  return_value=_CLIENT_ON),
            patch("store.learning_store.save_feedback", return_value={}),
        ):
            data = json.loads(self._call(req).body)

        self.assertIn("helpful", data["message"].lower())

    def test_correction_message_for_thumbs_down(self):
        req = _make_request({"rating": -1})
        with (
            patch("portal.routes._get_portal_user", return_value=_GOOD_USER),
            patch("portal.routes.store.get_client",  return_value=_CLIENT_ON),
            patch("store.learning_store.save_feedback", return_value={}),
        ):
            data = json.loads(self._call(req).body)

        self.assertIn("improve", data["message"].lower())

    # ── Auth / feature-flag guards ─────────────────────────────────────────────

    def test_unauthenticated_raises_401(self):
        req = _make_request({"rating": 1})
        with patch("portal.routes._get_portal_user", return_value=None):
            from fastapi import HTTPException
            with self.assertRaises(HTTPException) as ctx:
                self._call(req)
            self.assertEqual(ctx.exception.status_code, 401)

    def test_flag_off_raises_403(self):
        req = _make_request({"rating": 1})
        with (
            patch("portal.routes._get_portal_user", return_value=_GOOD_USER),
            patch("portal.routes.store.get_client",  return_value=_CLIENT_OFF),
        ):
            from fastapi import HTTPException
            with self.assertRaises(HTTPException) as ctx:
                self._call(req)
            self.assertEqual(ctx.exception.status_code, 403)

    def test_none_client_treated_as_flag_off(self):
        """store.get_client returning None → fallback {} → flag 0 → 403."""
        req = _make_request({"rating": 1})
        with (
            patch("portal.routes._get_portal_user", return_value=_GOOD_USER),
            patch("portal.routes.store.get_client",  return_value=None),
        ):
            from fastapi import HTTPException
            with self.assertRaises(HTTPException) as ctx:
                self._call(req)
            self.assertEqual(ctx.exception.status_code, 403)

    # ── Input validation ───────────────────────────────────────────────────────

    def test_rating_zero_raises_422(self):
        req = _make_request({"rating": 0})
        with (
            patch("portal.routes._get_portal_user", return_value=_GOOD_USER),
            patch("portal.routes.store.get_client",  return_value=_CLIENT_ON),
        ):
            from fastapi import HTTPException
            with self.assertRaises(HTTPException) as ctx:
                self._call(req)
            self.assertEqual(ctx.exception.status_code, 422)

    def test_rating_string_raises_422(self):
        req = _make_request({"rating": "up"})
        with (
            patch("portal.routes._get_portal_user", return_value=_GOOD_USER),
            patch("portal.routes.store.get_client",  return_value=_CLIENT_ON),
        ):
            from fastapi import HTTPException
            with self.assertRaises(HTTPException) as ctx:
                self._call(req)
            self.assertEqual(ctx.exception.status_code, 422)

    def test_missing_rating_raises_422(self):
        req = _make_request({})
        with (
            patch("portal.routes._get_portal_user", return_value=_GOOD_USER),
            patch("portal.routes.store.get_client",  return_value=_CLIENT_ON),
        ):
            from fastapi import HTTPException
            with self.assertRaises(HTTPException) as ctx:
                self._call(req)
            self.assertEqual(ctx.exception.status_code, 422)

    def test_malformed_json_raises_400(self):
        req = _make_bad_json_request()
        with (
            patch("portal.routes._get_portal_user", return_value=_GOOD_USER),
            patch("portal.routes.store.get_client",  return_value=_CLIENT_ON),
        ):
            from fastapi import HTTPException
            with self.assertRaises(HTTPException) as ctx:
                self._call(req)
            self.assertEqual(ctx.exception.status_code, 400)

    # ── Reason-code normalisation ──────────────────────────────────────────────

    def test_unknown_reason_code_normalised_to_other(self):
        req = _make_request({"rating": -1, "reason_code": "totally_invented"})
        captured = {}

        def _save(**kw):
            captured.update(kw)
            return {}

        with (
            patch("portal.routes._get_portal_user", return_value=_GOOD_USER),
            patch("portal.routes.store.get_client",  return_value=_CLIENT_ON),
            patch("store.learning_store.save_feedback", side_effect=_save),
        ):
            self._call(req)

        self.assertEqual(captured.get("reason_code"), "other")

    def test_valid_reason_codes_pass_through(self):
        for code in (
            "wrong_metric", "wrong_dimension", "wrong_filter",
            "wrong_join", "wrong_data", "incomplete", "confusing",
            "expected_data_missing", "other",
        ):
            req = _make_request({"rating": -1, "reason_code": code})
            captured = {}

            def _save(**kw):
                captured.update(kw)
                return {}

            with (
                patch("portal.routes._get_portal_user", return_value=_GOOD_USER),
                patch("portal.routes.store.get_client",  return_value=_CLIENT_ON),
                patch("store.learning_store.save_feedback", side_effect=_save),
            ):
                self._call(req, question_id=f"q_{code}")

            self.assertEqual(captured.get("reason_code"), code, f"code={code!r}")

    # ── save_feedback call signature ───────────────────────────────────────────

    def test_save_feedback_receives_correct_kwargs(self):
        req = _make_request({
            "rating":        -1,
            "reason_code":   "wrong_filter",
            "comment":       "Wrong period",
            "question_text": "Revenue this quarter?",
            "sql_text":      "SELECT SUM(rev)",
        })
        captured = {}

        def _save(**kw):
            captured.update(kw)
            return {}

        with (
            patch("portal.routes._get_portal_user", return_value=_GOOD_USER),
            patch("portal.routes.store.get_client",  return_value=_CLIENT_ON),
            patch("store.learning_store.save_feedback", side_effect=_save),
        ):
            self._call(req, question_id="qXYZ")

        self.assertEqual(captured["question_id"],   "qXYZ")
        self.assertEqual(captured["user_id"],         1)
        self.assertEqual(captured["account_id"],      "acct1")
        self.assertEqual(captured["rating"],          -1)
        self.assertEqual(captured["reason_code"],     "wrong_filter")
        self.assertEqual(captured["comment"],         "Wrong period")
        self.assertEqual(captured["question_text"],   "Revenue this quarter?")
        self.assertEqual(captured["sql_text"],        "SELECT SUM(rev)")

    # ── Content length truncation ──────────────────────────────────────────────

    def test_comment_truncated_at_1000_chars(self):
        req = _make_request({"rating": -1, "comment": "c" * 2500})
        captured = {}

        def _save(**kw):
            captured.update(kw)
            return {}

        with (
            patch("portal.routes._get_portal_user", return_value=_GOOD_USER),
            patch("portal.routes.store.get_client",  return_value=_CLIENT_ON),
            patch("store.learning_store.save_feedback", side_effect=_save),
        ):
            self._call(req)

        self.assertLessEqual(len(captured.get("comment", "")), 1000)

    def test_question_text_truncated_at_500_chars(self):
        req = _make_request({"rating": 1, "question_text": "q" * 900})
        captured = {}

        def _save(**kw):
            captured.update(kw)
            return {}

        with (
            patch("portal.routes._get_portal_user", return_value=_GOOD_USER),
            patch("portal.routes.store.get_client",  return_value=_CLIENT_ON),
            patch("store.learning_store.save_feedback", side_effect=_save),
        ):
            self._call(req)

        self.assertLessEqual(len(captured.get("question_text", "")), 500)

    def test_sql_text_truncated_at_4000_chars(self):
        req = _make_request({"rating": 1, "sql_text": "S" * 5000})
        captured = {}

        def _save(**kw):
            captured.update(kw)
            return {}

        with (
            patch("portal.routes._get_portal_user", return_value=_GOOD_USER),
            patch("portal.routes.store.get_client",  return_value=_CLIENT_ON),
            patch("store.learning_store.save_feedback", side_effect=_save),
        ):
            self._call(req)

        self.assertLessEqual(len(captured.get("sql_text", "")), 4000)

    # ── Store error → 500 ─────────────────────────────────────────────────────

    def test_unexpected_store_error_raises_500(self):
        req = _make_request({"rating": 1})
        with (
            patch("portal.routes._get_portal_user", return_value=_GOOD_USER),
            patch("portal.routes.store.get_client",  return_value=_CLIENT_ON),
            patch("store.learning_store.save_feedback",
                  side_effect=RuntimeError("db gone")),
        ):
            from fastapi import HTTPException
            with self.assertRaises(HTTPException) as ctx:
                self._call(req)
            self.assertEqual(ctx.exception.status_code, 500)


if __name__ == "__main__":
    unittest.main()
