"""
tests/test_learning_store.py

Integration tests for store/learning_store.py.

Uses an in-memory SQLite database via a monkeypatched get_db() so that:
  - Tests are fully isolated (no disk side-effects, no cross-test pollution).
  - All schema migrations are applied at fixture time, matching production.
"""

import json
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import sqlite3


# ---------------------------------------------------------------------------
# In-memory DB fixture
# ---------------------------------------------------------------------------

@contextmanager
def _memory_db_ctx(conn):
    """Context-manager wrapper around a SQLite connection (no-op on exit)."""
    try:
        yield conn
    finally:
        pass   # in-memory — nothing to close per-call


def _make_memory_db():
    """
    Create an in-memory SQLite connection with the full QueryBot schema applied,
    including all idempotent column migrations (v30 flags, approval_status, etc.).

    Strategy:
      1. Build the connection.
      2. Patch get_db to return it.
      3. Call _run_migrations() through that patch so the internal `with get_db()`
         call lands on our in-memory DB.
    """
    from store.db import _SCHEMA

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)

    # Apply all column migrations through the production code path so the
    # in-memory DB gets all added columns (v30 feature flags, etc.)
    with patch("store.db.get_db", return_value=_memory_db_ctx(conn)):
        from store.db import _run_migrations
        _run_migrations()

    # Insert the minimal seed rows our tests depend on via FK constraints.
    conn.execute(
        "INSERT OR IGNORE INTO client (account_id, client_name, platform_type) "
        "VALUES ('test_acct', 'Test', 'web')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO portal_user "
        "(id, account_id, email, password_hash, name, role) "
        "VALUES (1, 'test_acct', 'user@test.com', 'x', 'Tester', 'analyst')"
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Test base with patched get_db
# ---------------------------------------------------------------------------

class _LearningStoreBase(unittest.TestCase):
    """
    Patch get_db in both the db module and the learning_store module.

    learning_store.py does `from store.db import get_db`, so patching
    store.db.get_db alone does not intercept calls already imported into
    learning_store.  We must also patch the name where it is used.
    """

    def setUp(self):
        self._conn = _make_memory_db()
        # side_effect: fresh context-manager generator on every call
        _factory = lambda: _memory_db_ctx(self._conn)   # noqa: E731

        self._patcher_db = patch("store.db.get_db", side_effect=_factory)
        self._patcher_ls = patch("store.learning_store.get_db", side_effect=_factory)
        self._patcher_db.start()
        self._patcher_ls.start()

    def tearDown(self):
        self._patcher_ls.stop()
        self._patcher_db.stop()
        self._conn.close()


# ---------------------------------------------------------------------------
# save_feedback
# ---------------------------------------------------------------------------

class TestSaveFeedback(_LearningStoreBase):

    def test_save_returns_dict_with_expected_keys(self):
        from store.learning_store import save_feedback
        result = save_feedback("q1", 1, "test_acct", 1,
                               reason_code="wrong_metric",
                               comment="nope",
                               question_text="?", sql_text="SELECT 1",
                               schema_scope="")
        for key in ("question_id", "user_id", "rating", "reason_code"):
            self.assertIn(key, result)

    def test_rating_stored_correctly(self):
        from store.learning_store import save_feedback, get_feedback
        save_feedback("q2", 1, "test_acct", -1,
                      reason_code="wrong_filter", comment="",
                      question_text="q", sql_text="s", schema_scope="")
        row = get_feedback("q2", 1)
        self.assertEqual(row["rating"], -1)

    def test_upsert_updates_rating(self):
        from store.learning_store import save_feedback, get_feedback
        save_feedback("q3", 1, "test_acct", 1,
                      reason_code="other", comment="",
                      question_text="q", sql_text="s", schema_scope="")
        save_feedback("q3", 1, "test_acct", -1,
                      reason_code="wrong_data", comment="",
                      question_text="q", sql_text="s", schema_scope="")
        row = get_feedback("q3", 1)
        self.assertEqual(row["rating"], -1)

    def test_invalid_rating_raises(self):
        from store.learning_store import save_feedback
        with self.assertRaises(ValueError):
            save_feedback("q4", 1, "test_acct", 0,
                          reason_code="other", comment="",
                          question_text="q", sql_text="s", schema_scope="")

    def test_invalid_reason_code_normalised_to_other(self):
        from store.learning_store import save_feedback, get_feedback
        save_feedback("q5", 1, "test_acct", 1,
                      reason_code="completely_made_up_code",
                      comment="", question_text="q", sql_text="s", schema_scope="")
        row = get_feedback("q5", 1)
        self.assertEqual(row["reason_code"], "other")

    def test_different_users_independent_rows(self):
        from store.learning_store import save_feedback, list_feedback
        # Add a second portal user
        self._conn.execute(
            "INSERT OR IGNORE INTO portal_user "
            "(id, account_id, email, password_hash, name, role) "
            "VALUES (2, 'test_acct', 'u2@test.com', 'x', 'User2', 'analyst')"
        )
        self._conn.commit()
        save_feedback("q6", 1, "test_acct", 1,
                      reason_code="other", comment="",
                      question_text="q", sql_text="s", schema_scope="")
        save_feedback("q6", 2, "test_acct", -1,
                      reason_code="other", comment="",
                      question_text="q", sql_text="s", schema_scope="")
        rows = list_feedback("q6")
        self.assertEqual(len(rows), 2)
        ratings = {r["user_id"]: r["rating"] for r in rows}
        self.assertEqual(ratings[1], 1)
        self.assertEqual(ratings[2], -1)


class TestGetFeedback(_LearningStoreBase):

    def test_returns_none_for_missing(self):
        from store.learning_store import get_feedback
        self.assertIsNone(get_feedback("no_such_q", 99))

    def test_returns_dict_after_save(self):
        from store.learning_store import save_feedback, get_feedback
        save_feedback("qx", 1, "test_acct", 1,
                      reason_code="other", comment="",
                      question_text="q", sql_text="s", schema_scope="")
        row = get_feedback("qx", 1)
        self.assertIsNotNone(row)
        self.assertEqual(row["question_id"], "qx")


class TestListFeedback(_LearningStoreBase):

    def test_empty_when_none(self):
        from store.learning_store import list_feedback
        self.assertEqual(list_feedback("nothing"), [])

    def test_returns_all_votes(self):
        from store.learning_store import save_feedback, list_feedback
        self._conn.execute(
            "INSERT OR IGNORE INTO portal_user "
            "(id, account_id, email, password_hash, name, role) "
            "VALUES (3, 'test_acct', 'u3@test.com', 'x', 'User3', 'analyst')"
        )
        self._conn.commit()
        save_feedback("qlist", 1, "test_acct", 1,
                      reason_code="other", comment="",
                      question_text="q", sql_text="s", schema_scope="")
        save_feedback("qlist", 3, "test_acct", -1,
                      reason_code="other", comment="",
                      question_text="q", sql_text="s", schema_scope="")
        rows = list_feedback("qlist")
        self.assertEqual(len(rows), 2)


# ---------------------------------------------------------------------------
# create_candidate
# ---------------------------------------------------------------------------

class TestCreateCandidate(_LearningStoreBase):

    def _create(self, score=75, **kwargs):
        from store.learning_store import create_candidate
        defaults = dict(
            origin_question_id="orig_q",
            account_id="test_acct",
            question_text="What is revenue?",
            sql_text="SELECT SUM(revenue) FROM sales",
            technical_score=score,
            evidence={"sql_validation": 25},
        )
        defaults.update(kwargs)
        return create_candidate(**defaults)

    def test_returns_dict_with_candidate_id(self):
        c = self._create()
        self.assertIn("candidate_id", c)
        self.assertTrue(len(c["candidate_id"]) == 12)

    def test_initial_final_score_equals_technical_score(self):
        c = self._create(score=70)
        self.assertEqual(c["final_score"], 70)
        self.assertEqual(c["technical_score"], 70)

    def test_initial_status_pending_review(self):
        c = self._create()
        self.assertEqual(c["status"], "pending_review")

    def test_high_score_classified_positive(self):
        c = self._create(score=90)
        self.assertEqual(c["candidate_type"], "positive")

    def test_mid_score_classified_review(self):
        c = self._create(score=70)
        self.assertEqual(c["candidate_type"], "review")

    def test_low_score_classified_negative(self):
        c = self._create(score=40)
        self.assertEqual(c["candidate_type"], "negative")

    def test_evidence_stored_as_json(self):
        c = self._create(evidence={"test": 42})
        ev = json.loads(c["evidence"])
        self.assertEqual(ev["test"], 42)

    def test_source_defaults_to_auto(self):
        c = self._create()
        self.assertEqual(c["source"], "auto")

    def test_explicit_source(self):
        c = self._create(source="pre_governed")
        self.assertEqual(c["source"], "pre_governed")


class TestGetCandidate(_LearningStoreBase):

    def test_returns_none_for_unknown(self):
        from store.learning_store import get_candidate
        self.assertIsNone(get_candidate("notexist"))

    def test_roundtrip(self):
        from store.learning_store import create_candidate, get_candidate
        c = create_candidate("q", "test_acct", "Q", "SQL", 80, {})
        fetched = get_candidate(c["candidate_id"])
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["candidate_id"], c["candidate_id"])


class TestListCandidates(_LearningStoreBase):

    _seq = 0  # class-level counter for unique origin_question_id per call

    def _make(self, score, status_override=None):
        from store.learning_store import create_candidate, update_candidate_status
        TestListCandidates._seq += 1
        qid = f"q_{TestListCandidates._seq}"
        c = create_candidate(qid, "test_acct", "Q", "SQL", score, {})
        if status_override:
            update_candidate_status(c["candidate_id"], status_override)
        return c

    def test_returns_all_for_account(self):
        from store.learning_store import list_candidates
        self._make(80)
        self._make(60)
        rows = list_candidates("test_acct")
        self.assertGreaterEqual(len(rows), 2)

    def test_status_filter(self):
        from store.learning_store import list_candidates
        self._make(90, status_override="approved")
        self._make(55)
        approved = list_candidates("test_acct", status="approved")
        self.assertTrue(all(r["status"] == "approved" for r in approved))

    def test_empty_for_unknown_account(self):
        from store.learning_store import list_candidates
        self.assertEqual(list_candidates("no_such_account"), [])

    def test_limit_respected(self):
        from store.learning_store import list_candidates
        for _ in range(5):
            self._make(70)
        rows = list_candidates("test_acct", limit=2)
        self.assertLessEqual(len(rows), 2)


# ---------------------------------------------------------------------------
# update_candidate_score
# ---------------------------------------------------------------------------

class TestUpdateCandidateScore(_LearningStoreBase):

    def test_score_updated(self):
        from store.learning_store import create_candidate, update_candidate_score, get_candidate
        c = create_candidate("q", "test_acct", "Q", "SQL", 70, {})
        update_candidate_score(c["candidate_id"], final_score=90,
                               feedback_delta=10, evidence={"ok": True})
        fetched = get_candidate(c["candidate_id"])
        self.assertEqual(fetched["final_score"], 90)

    def test_approved_drops_to_pending_when_negative(self):
        from store.learning_store import (
            create_candidate, update_candidate_status,
            update_candidate_score, get_candidate,
        )
        c = create_candidate("q", "test_acct", "Q", "SQL", 90, {})
        update_candidate_status(c["candidate_id"], "approved")
        # Simulate strong negative feedback driving score to 20
        update_candidate_score(c["candidate_id"], final_score=20,
                               feedback_delta=-30, evidence={},
                               positive_votes=0, negative_votes=3)
        fetched = get_candidate(c["candidate_id"])
        self.assertEqual(fetched["status"], "pending_review")

    def test_no_op_for_unknown_candidate(self):
        from store.learning_store import update_candidate_score
        # Should not raise
        update_candidate_score("does_not_exist", 80, 0, {})


# ---------------------------------------------------------------------------
# update_candidate_status
# ---------------------------------------------------------------------------

class TestUpdateCandidateStatus(_LearningStoreBase):

    def test_valid_status_transitions(self):
        from store.learning_store import (
            create_candidate, update_candidate_status, get_candidate,
        )
        c = create_candidate("q", "test_acct", "Q", "SQL", 80, {})
        for status in ("approved", "rejected", "known_failure", "revoked"):
            update_candidate_status(c["candidate_id"], status)
            fetched = get_candidate(c["candidate_id"])
            self.assertEqual(fetched["status"], status)

    def test_invalid_status_raises(self):
        from store.learning_store import create_candidate, update_candidate_status
        c = create_candidate("q", "test_acct", "Q", "SQL", 80, {})
        with self.assertRaises(ValueError):
            update_candidate_status(c["candidate_id"], "super_approved")

    def test_approved_sets_promoted_at(self):
        from store.learning_store import (
            create_candidate, update_candidate_status, get_candidate,
        )
        c = create_candidate("q", "test_acct", "Q", "SQL", 90, {})
        update_candidate_status(c["candidate_id"], "approved")
        fetched = get_candidate(c["candidate_id"])
        self.assertIsNotNone(fetched["promoted_at"])

    def test_non_approved_does_not_set_promoted_at(self):
        from store.learning_store import (
            create_candidate, update_candidate_status, get_candidate,
        )
        c = create_candidate("q", "test_acct", "Q", "SQL", 40, {})
        update_candidate_status(c["candidate_id"], "rejected")
        fetched = get_candidate(c["candidate_id"])
        self.assertIsNone(fetched["promoted_at"])


# ---------------------------------------------------------------------------
# set_candidate_corrected_sql
# ---------------------------------------------------------------------------

class TestSetCandidateCorrectedSql(_LearningStoreBase):

    def test_corrected_sql_stored(self):
        from store.learning_store import (
            create_candidate, set_candidate_corrected_sql, get_candidate,
        )
        c = create_candidate("q", "test_acct", "Q", "SQL", 50, {})
        set_candidate_corrected_sql(c["candidate_id"], "SELECT 2", "admin_1")
        fetched = get_candidate(c["candidate_id"])
        self.assertEqual(fetched["corrected_sql"], "SELECT 2")

    def test_score_set_to_85(self):
        from store.learning_store import (
            create_candidate, set_candidate_corrected_sql, get_candidate,
        )
        c = create_candidate("q", "test_acct", "Q", "SQL", 40, {})
        set_candidate_corrected_sql(c["candidate_id"], "SELECT 2")
        fetched = get_candidate(c["candidate_id"])
        self.assertEqual(fetched["technical_score"], 85)
        self.assertEqual(fetched["final_score"], 85)

    def test_source_becomes_admin_correction(self):
        from store.learning_store import (
            create_candidate, set_candidate_corrected_sql, get_candidate,
        )
        c = create_candidate("q", "test_acct", "Q", "SQL", 40, {})
        set_candidate_corrected_sql(c["candidate_id"], "SELECT 2")
        fetched = get_candidate(c["candidate_id"])
        self.assertEqual(fetched["source"], "admin_correction")

    def test_candidate_type_becomes_positive(self):
        from store.learning_store import (
            create_candidate, set_candidate_corrected_sql, get_candidate,
        )
        c = create_candidate("q", "test_acct", "Q", "SQL", 10, {})
        set_candidate_corrected_sql(c["candidate_id"], "SELECT 2")
        fetched = get_candidate(c["candidate_id"])
        self.assertEqual(fetched["candidate_type"], "positive")


# ---------------------------------------------------------------------------
# recompute_candidate_score_from_feedback
# ---------------------------------------------------------------------------

class TestRecomputeCandidateScore(_LearningStoreBase):

    def _setup_candidate_with_votes(self, pos_votes, neg_votes):
        """
        Create a candidate linked to a question, then add feedback votes.
        Returns the candidate dict.
        """
        from store.learning_store import create_candidate, save_feedback

        c = create_candidate("qr1", "test_acct", "Q", "SQL", 75, {})

        # Add extra portal users if needed
        for uid in range(1, pos_votes + neg_votes + 1):
            self._conn.execute(
                "INSERT OR IGNORE INTO portal_user "
                "(id, account_id, email, password_hash, name, role) "
                "VALUES (?, 'test_acct', ?, 'x', ?, 'analyst')",
                (uid + 10, f"u{uid+10}@test.com", f"U{uid+10}"),
            )
        self._conn.commit()

        for uid in range(pos_votes):
            save_feedback("qr1", uid + 11, "test_acct", 1,
                          reason_code="other", comment="",
                          question_text="Q", sql_text="SQL", schema_scope="")
        for uid in range(neg_votes):
            save_feedback("qr1", uid + 11 + pos_votes, "test_acct", -1,
                          reason_code="other", comment="",
                          question_text="Q", sql_text="SQL", schema_scope="")
        return c

    def test_positive_votes_raise_final_score(self):
        from store.learning_store import get_candidate
        c = self._setup_candidate_with_votes(pos_votes=3, neg_votes=0)
        fetched = get_candidate(c["candidate_id"])
        self.assertGreater(fetched["final_score"], c["technical_score"])

    def test_negative_votes_lower_final_score(self):
        from store.learning_store import create_candidate, save_feedback, get_candidate
        c = create_candidate("qr2", "test_acct", "Q2", "SQL", 80, {})
        self._conn.execute(
            "INSERT OR IGNORE INTO portal_user "
            "(id, account_id, email, password_hash, name, role) "
            "VALUES (20, 'test_acct', 'neg@test.com', 'x', 'Neg', 'analyst')"
        )
        self._conn.commit()
        save_feedback("qr2", 20, "test_acct", -1,
                      reason_code="wrong_data", comment="",
                      question_text="Q2", sql_text="SQL", schema_scope="")
        fetched = get_candidate(c["candidate_id"])
        self.assertLess(fetched["final_score"], c["technical_score"])

    def test_noop_when_no_candidate(self):
        from store.learning_store import recompute_candidate_score_from_feedback
        # Should not raise
        recompute_candidate_score_from_feedback("nonexistent_question", "test_acct")


# ---------------------------------------------------------------------------
# record_event / get_suggestion_stats
# ---------------------------------------------------------------------------

class TestRecommendationEvents(_LearningStoreBase):

    def test_record_displayed_event(self):
        from store.learning_store import record_event, get_suggestion_stats
        record_event("sess1", "test_acct", "displayed", "Show top customers",
                     user_id=1, suggestion_source="genie")
        stats = get_suggestion_stats("test_acct", "Show top customers")
        self.assertEqual(stats.get("displayed", 0), 1)

    def test_record_clicked_event(self):
        from store.learning_store import record_event, get_suggestion_stats
        record_event("sess2", "test_acct", "clicked", "Monthly revenue",
                     suggestion_source="learned")
        stats = get_suggestion_stats("test_acct", "Monthly revenue")
        self.assertEqual(stats.get("clicked", 0), 1)

    def test_unknown_event_type_silently_dropped(self):
        from store.learning_store import record_event, get_suggestion_stats
        record_event("sess3", "test_acct", "unknown_type", "Anything")
        stats = get_suggestion_stats("test_acct", "Anything")
        self.assertEqual(stats, {})

    def test_stats_empty_for_unseen_suggestion(self):
        from store.learning_store import get_suggestion_stats
        stats = get_suggestion_stats("test_acct", "No such suggestion ever")
        self.assertEqual(stats, {})

    def test_multiple_events_counted_correctly(self):
        from store.learning_store import record_event, get_suggestion_stats
        for _ in range(3):
            record_event("s", "test_acct", "displayed", "Top products")
        record_event("s", "test_acct", "clicked", "Top products")
        stats = get_suggestion_stats("test_acct", "Top products")
        self.assertEqual(stats["displayed"], 3)
        self.assertEqual(stats["clicked"], 1)

    def test_suggestion_truncated_to_500_chars(self):
        from store.learning_store import record_event
        long_text = "x" * 600
        # Should not raise
        record_event("s", "test_acct", "displayed", long_text)


# ---------------------------------------------------------------------------
# VALID_REASON_CODES set
# ---------------------------------------------------------------------------

class TestValidReasonCodes(unittest.TestCase):

    def test_known_codes_present(self):
        from store.learning_store import VALID_REASON_CODES
        for code in ("wrong_metric", "wrong_dimension", "wrong_filter",
                     "wrong_join", "wrong_data", "incomplete",
                     "confusing", "expected_data_missing", "other"):
            self.assertIn(code, VALID_REASON_CODES)

    def test_other_always_valid(self):
        from store.learning_store import VALID_REASON_CODES
        self.assertIn("other", VALID_REASON_CODES)


if __name__ == "__main__":
    unittest.main()
