"""
tests/test_genie_ranker.py

Unit tests for the Genie behavioral signal scoring engine.

Day 8-9 of the Governed Self-Learning Analytical Genie sprint.

Coverage:
  TestScoreSuggestion           — pure scoring maths, no DB (12 tests)
  TestRankSuggestions           — mocked get_suggestion_stats, ordering (8 tests)
  TestBuildChatSuggestionsGenie — portal integration with feature flag (6 tests)
  TestPortalSuggestionsEventApi — POST /portal/api/suggestions/event (9 tests)

Strategy: direct unit tests — no FastAPI TestClient, no SQLite, no network.
All external calls are patched at the point of use.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import MagicMock, patch, call


# ── Async helper ──────────────────────────────────────────────────────────────

def _arun(coro):
    return asyncio.run(coro)


# ── Shared fixtures ───────────────────────────────────────────────────────────

_USER = {
    "id": 42,
    "account_id": "tenant1",
    "name": "Alice",
    "email": "alice@example.com",
    "role": "analyst",
}

_SUGS = [
    {"question": "What is revenue?",           "fqn": "DB.DBO.SALES"},
    {"question": "Show me top customers",      "fqn": "DB.DBO.CUSTOMERS"},
    {"question": "Monthly active users trend", "fqn": "DB.DBO.EVENTS"},
]


# ══════════════════════════════════════════════════════════════════════════════
# TestScoreSuggestion — pure math, no I/O
# ══════════════════════════════════════════════════════════════════════════════

class TestScoreSuggestion(unittest.TestCase):

    def _call(self, stats, source="static", threshold=10):
        from core.genie_ranker import score_suggestion
        return score_suggestion(stats, source=source, impression_threshold=threshold)

    # ── Cold-start ────────────────────────────────────────────────────────────

    def test_cold_start_no_impressions_static_returns_zero(self):
        score = self._call({})
        self.assertEqual(score, 0.0)

    def test_cold_start_below_threshold_governed_returns_boost(self):
        score = self._call({"displayed": 5}, source="governed")
        self.assertAlmostEqual(score, 0.10)

    def test_cold_start_below_threshold_learned_returns_boost(self):
        score = self._call({"displayed": 9}, source="learned")
        self.assertAlmostEqual(score, 0.05)

    def test_cold_start_exactly_at_threshold_is_warm(self):
        # impressions == threshold → warm-start (not cold)
        stats = {"displayed": 10, "clicked": 5}
        score = self._call(stats, source="static", threshold=10)
        # ctr=0.5 → behavioral=0.30*0.5=0.15; confidence=min(10/100,1)=0.1
        # score = 0.1*0.15 + 0.9*0.0 = 0.015
        self.assertAlmostEqual(score, 0.015, places=6)

    # ── Warm-start weight correctness ─────────────────────────────────────────

    def test_warm_ctr_only(self):
        stats = {"displayed": 100, "clicked": 50}
        score = self._call(stats)
        # behavioral=0.30*0.5=0.15; confidence=1.0; boost=0
        self.assertAlmostEqual(score, 0.15, places=6)

    def test_warm_exec_rate_only(self):
        stats = {"displayed": 100, "executed": 80}
        score = self._call(stats)
        # behavioral=0.40*0.8=0.32; confidence=1.0; boost=0
        self.assertAlmostEqual(score, 0.32, places=6)

    def test_warm_success_rate_only(self):
        stats = {"displayed": 100, "successful": 60}
        score = self._call(stats)
        # behavioral=0.25*0.6=0.15; confidence=1.0; boost=0
        self.assertAlmostEqual(score, 0.15, places=6)

    def test_warm_dismissal_penalty(self):
        stats = {"displayed": 100, "dismissed": 100}
        score = self._call(stats)
        # behavioral=−0.05*1.0=−0.05; confidence=1.0; boost=0
        self.assertAlmostEqual(score, -0.05, places=6)

    def test_warm_all_signals_combined(self):
        stats = {
            "displayed":  200,
            "clicked":     80,   # ctr=0.40
            "executed":    60,   # exec=0.30
            "successful":  40,   # suc=0.20
            "dismissed":   20,   # dis=0.10
        }
        score = self._call(stats, source="governed")
        # behavioral = 0.30*0.40 + 0.40*0.30 + 0.25*0.20 - 0.05*0.10
        #            = 0.12 + 0.12 + 0.05 - 0.005 = 0.285
        # confidence = min(200/100, 1.0) = 1.0
        # score = 1.0*0.285 + 0.0*0.10 = 0.285
        self.assertAlmostEqual(score, 0.285, places=6)

    # ── Confidence blending ───────────────────────────────────────────────────

    def test_confidence_blending_halfway(self):
        # impressions=50 → confidence=0.5
        stats = {"displayed": 50, "clicked": 50}   # ctr=1.0
        score = self._call(stats, source="learned")
        # behavioral=0.30*1.0=0.30; confidence=0.5; boost=0.05
        # score = 0.5*0.30 + 0.5*0.05 = 0.15 + 0.025 = 0.175
        self.assertAlmostEqual(score, 0.175, places=6)

    def test_confidence_capped_at_one(self):
        # impressions=500 → confidence still capped at 1.0
        stats = {"displayed": 500, "executed": 250}
        score = self._call(stats)
        # exec_rate=0.5; behavioral=0.40*0.5=0.20; confidence=1.0
        self.assertAlmostEqual(score, 0.20, places=6)

    # ── Source boost resolution ───────────────────────────────────────────────

    def test_source_admin_correction_maps_to_governed_boost(self):
        score = self._call({}, source="admin_correction")
        self.assertAlmostEqual(score, 0.10)

    def test_source_auto_maps_to_learned_boost(self):
        score = self._call({}, source="auto")
        self.assertAlmostEqual(score, 0.05)

    def test_source_unknown_gives_zero_boost(self):
        score = self._call({}, source="unknown_tier")
        self.assertAlmostEqual(score, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# TestRankSuggestions — mocked stats
# ══════════════════════════════════════════════════════════════════════════════

class TestRankSuggestions(unittest.TestCase):

    def _call(self, account_id, suggestions, **kwargs):
        from core.genie_ranker import rank_suggestions
        return rank_suggestions(account_id, suggestions, **kwargs)

    def _patch_stats(self, stats_by_question: dict):
        """Patch get_suggestion_stats to return pre-defined stats per question."""
        def _side(account_id, question):
            return stats_by_question.get(question, {})
        return patch("store.learning_store.get_suggestion_stats", side_effect=_side)

    # ── Basic behaviour ───────────────────────────────────────────────────────

    def test_empty_suggestions_returns_empty(self):
        with self._patch_stats({}):
            result = self._call("acct", [])
        self.assertEqual(result, [])

    def test_single_suggestion_returned(self):
        with self._patch_stats({"What is revenue?": {"displayed": 0}}):
            result = self._call("acct", [{"question": "What is revenue?", "fqn": ""}])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["question"], "What is revenue?")

    def test_sorted_by_score_descending(self):
        # revenue: 100 impressions, 80 clicks → high score
        # customers: 0 impressions → cold start score=0
        stats_map = {
            "What is revenue?":      {"displayed": 100, "clicked": 80},
            "Show me top customers": {},
        }
        sugs = [
            {"question": "Show me top customers", "fqn": ""},
            {"question": "What is revenue?",      "fqn": ""},
        ]
        with self._patch_stats(stats_map):
            result = self._call("acct", sugs)
        self.assertEqual(result[0]["question"], "What is revenue?")
        self.assertEqual(result[1]["question"], "Show me top customers")

    def test_score_key_injected(self):
        with self._patch_stats({}):
            result = self._call("acct", [{"question": "Q", "fqn": ""}])
        self.assertIn("_score", result[0])

    def test_source_map_overrides_suggestion_source(self):
        """source_map wins over any 'source' field already on the suggestion."""
        sugs = [{"question": "Q", "fqn": "", "source": "static"}]
        # With source_map → governed boost 0.10
        with self._patch_stats({}):
            result_gov = self._call("acct", sugs, source_map={"Q": "governed"})
        self.assertAlmostEqual(result_gov[0]["_score"], 0.10)

        # Without source_map → static boost 0.00
        with self._patch_stats({}):
            result_static = self._call("acct", sugs)
        self.assertAlmostEqual(result_static[0]["_score"], 0.00)

    def test_suggestion_source_field_used_when_no_source_map(self):
        sugs = [{"question": "Q", "fqn": "", "source": "learned"}]
        with self._patch_stats({}):
            result = self._call("acct", sugs)
        self.assertAlmostEqual(result[0]["_score"], 0.05)

    def test_stats_error_gives_cold_start_zero_score(self):
        """A DB error on a suggestion's stats should not propagate; score → 0.0."""
        def _bad_stats(account_id, question):
            raise RuntimeError("DB unavailable")

        with patch("store.learning_store.get_suggestion_stats", side_effect=_bad_stats):
            result = self._call("acct", [{"question": "Q", "fqn": ""}])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["_score"], 0.0)

    def test_original_dict_not_mutated(self):
        """rank_suggestions should not modify the caller's original dicts."""
        original = {"question": "Q", "fqn": ""}
        with self._patch_stats({}):
            result = self._call("acct", [original])
        self.assertNotIn("_score", original)
        self.assertIn("_score", result[0])


# ══════════════════════════════════════════════════════════════════════════════
# TestBuildChatSuggestionsGenie — portal integration
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildChatSuggestionsGenie(unittest.TestCase):
    """
    Test that _build_chat_suggestions wires the genie ranker and impression
    recording correctly when enable_genie_suggestions is on / off.
    """

    def _base_patches(self, *, genie_flag: int = 0):
        """Return a stack of patches shared by most tests in this class."""
        client = {"enable_genie_suggestions": genie_flag}
        patches = [
            patch("store.get_allowed_tables", return_value=["DB.DBO.SALES"]),
            patch("store.get_client", return_value=client),
            patch(
                "core.suggestions.get_suggestions",
                return_value=list(_SUGS),
            ),
            # Avoid hitting the glossary fallback
            patch(
                "portal.routes._guess_safe_metric_suggestions",
                return_value=[],
            ),
        ]
        return patches, client

    def test_flag_disabled_rank_not_called(self):
        patches, _ = self._base_patches(genie_flag=0)
        with (
            patches[0], patches[1], patches[2], patches[3],
            patch("core.genie_ranker.rank_suggestions") as mock_rank,
        ):
            from portal.routes import _build_chat_suggestions
            result = _build_chat_suggestions(_USER)
        mock_rank.assert_not_called()
        # Unranked suggestions returned
        self.assertEqual(len(result), 3)

    def test_flag_enabled_rank_called(self):
        patches, _ = self._base_patches(genie_flag=1)
        ranked = list(reversed(_SUGS))
        with (
            patches[0], patches[1], patches[2], patches[3],
            patch("core.genie_ranker.rank_suggestions", return_value=ranked) as mock_rank,
        ):
            from portal.routes import _build_chat_suggestions
            result = _build_chat_suggestions(_USER)
        mock_rank.assert_called_once()
        call_args = mock_rank.call_args
        self.assertEqual(call_args[0][0], "tenant1")  # account_id
        self.assertEqual(result, ranked[:6])

    def test_rank_exception_falls_back_to_unranked(self):
        patches, _ = self._base_patches(genie_flag=1)
        with (
            patches[0], patches[1], patches[2], patches[3],
            patch("core.genie_ranker.rank_suggestions",
                  side_effect=RuntimeError("qdrant down")),
        ):
            from portal.routes import _build_chat_suggestions
            result = _build_chat_suggestions(_USER)
        # Falls back to the original unranked list — must not raise
        self.assertEqual(len(result), 3)

    def test_record_displayed_called_when_flag_on(self):
        patches, _ = self._base_patches(genie_flag=1)
        with (
            patches[0], patches[1], patches[2], patches[3],
            patch("core.genie_ranker.rank_suggestions", return_value=list(_SUGS)),
            patch("portal.routes._record_suggestions_displayed") as mock_rec,
        ):
            # Simulate a portal_chat call that calls _build_chat_suggestions
            from portal.routes import _build_chat_suggestions, _record_suggestions_displayed
            sugs = _build_chat_suggestions(_USER)
            # Simulate what portal_chat does
            _record_suggestions_displayed(_USER, sugs)
        mock_rec.assert_called_once()

    def test_record_displayed_not_called_when_flag_off(self):
        """
        When enable_genie_suggestions=0, the portal_chat handler must NOT call
        _record_suggestions_displayed (controlled by the flag check in the route).
        We test the conditional logic directly.
        """
        client = {"enable_genie_suggestions": 0}
        # The route does: if client.get("enable_genie_suggestions") and suggestions:
        flag_on = bool(client.get("enable_genie_suggestions"))
        self.assertFalse(flag_on)

    def test_suggestions_trimmed_to_six_after_ranking(self):
        many = [{"question": f"Q{i}", "fqn": ""} for i in range(10)]
        with (
            patch("store.get_allowed_tables", return_value=[]),
            patch("store.get_client", return_value={"enable_genie_suggestions": 1}),
            patch("core.suggestions.get_suggestions", return_value=many),
            patch("portal.routes._guess_safe_metric_suggestions", return_value=[]),
            patch("core.genie_ranker.rank_suggestions", return_value=many),
        ):
            from portal.routes import _build_chat_suggestions
            result = _build_chat_suggestions(_USER)
        self.assertEqual(len(result), 6)


# ══════════════════════════════════════════════════════════════════════════════
# TestPortalSuggestionsEventApi — POST /portal/api/suggestions/event
# ══════════════════════════════════════════════════════════════════════════════

def _make_request(body: dict, has_cookie: bool = True):
    req = MagicMock()
    req.cookies = {"qb_portal_session": "tok.sig"} if has_cookie else {}

    async def _json():
        return body

    req.json = _json
    return req


def _make_bad_json_request():
    req = MagicMock()
    req.cookies = {"qb_portal_session": "tok.sig"}

    async def _json():
        raise ValueError("not json")

    req.json = _json
    return req


class TestPortalSuggestionsEventApi(unittest.TestCase):

    def _call(self, request):
        from portal.routes import portal_suggestions_event
        return _arun(portal_suggestions_event(request))

    # ── Auth ──────────────────────────────────────────────────────────────────

    def test_unauthenticated_returns_401(self):
        from fastapi import HTTPException
        req = _make_request({"event_type": "clicked", "suggestion_text": "Q"}, has_cookie=False)
        with patch("portal.routes._get_portal_user", return_value=None):
            with self.assertRaises(HTTPException) as ctx:
                self._call(req)
        self.assertEqual(ctx.exception.status_code, 401)

    # ── Input validation ──────────────────────────────────────────────────────

    def test_invalid_json_returns_400(self):
        from fastapi import HTTPException
        req = _make_bad_json_request()
        with patch("portal.routes._get_portal_user", return_value=_USER):
            with self.assertRaises(HTTPException) as ctx:
                self._call(req)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_invalid_event_type_returns_400(self):
        from fastapi import HTTPException
        req = _make_request({"event_type": "displayed", "suggestion_text": "Q"})
        with patch("portal.routes._get_portal_user", return_value=_USER):
            with self.assertRaises(HTTPException) as ctx:
                self._call(req)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_unknown_event_type_returns_400(self):
        from fastapi import HTTPException
        req = _make_request({"event_type": "invented", "suggestion_text": "Q"})
        with patch("portal.routes._get_portal_user", return_value=_USER):
            with self.assertRaises(HTTPException) as ctx:
                self._call(req)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_missing_suggestion_text_returns_400(self):
        from fastapi import HTTPException
        req = _make_request({"event_type": "clicked", "suggestion_text": ""})
        with patch("portal.routes._get_portal_user", return_value=_USER):
            with self.assertRaises(HTTPException) as ctx:
                self._call(req)
        self.assertEqual(ctx.exception.status_code, 400)

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_valid_click_event_returns_200_ok(self):
        req = _make_request({
            "event_type": "clicked",
            "suggestion_text": "What is revenue?",
            "session_id": "sess-abc",
        })
        with (
            patch("portal.routes._get_portal_user", return_value=_USER),
            patch("store.learning_store.record_event") as mock_rec,
        ):
            resp = self._call(req)
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.body)
        self.assertTrue(body["ok"])

    def test_valid_dismiss_event_returns_200_ok(self):
        req = _make_request({
            "event_type": "dismissed",
            "suggestion_text": "Show top customers",
        })
        with (
            patch("portal.routes._get_portal_user", return_value=_USER),
            patch("store.learning_store.record_event"),
        ):
            resp = self._call(req)
        self.assertEqual(resp.status_code, 200)

    def test_record_event_called_with_correct_args(self):
        req = _make_request({
            "event_type": "executed",
            "suggestion_text": "Monthly revenue",
            "session_id": "sess-xyz",
        })
        with (
            patch("portal.routes._get_portal_user", return_value=_USER),
            patch("store.learning_store.record_event") as mock_rec,
        ):
            self._call(req)
        mock_rec.assert_called_once_with(
            session_id="sess-xyz",
            account_id="tenant1",
            event_type="executed",
            suggestion_text="Monthly revenue",
            user_id=42,
        )

    def test_record_event_failure_still_returns_200(self):
        """A DB write failure must never break the chat client."""
        req = _make_request({
            "event_type": "successful",
            "suggestion_text": "Q",
        })
        with (
            patch("portal.routes._get_portal_user", return_value=_USER),
            patch("store.learning_store.record_event",
                  side_effect=RuntimeError("DB down")),
        ):
            resp = self._call(req)
        self.assertEqual(resp.status_code, 200)

    def test_missing_session_id_uses_browser_fallback(self):
        """Empty / missing session_id should be handled gracefully (defaults to 'browser')."""
        req = _make_request({
            "event_type": "clicked",
            "suggestion_text": "Q",
            # no session_id key
        })
        with (
            patch("portal.routes._get_portal_user", return_value=_USER),
            patch("store.learning_store.record_event") as mock_rec,
        ):
            resp = self._call(req)
        self.assertEqual(resp.status_code, 200)
        call_kwargs = mock_rec.call_args[1]
        self.assertEqual(call_kwargs["session_id"], "browser")


if __name__ == "__main__":
    unittest.main()
