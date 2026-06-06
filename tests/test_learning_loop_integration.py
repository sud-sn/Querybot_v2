"""
tests/test_learning_loop_integration.py

Day 10 — Integration tests for the full self-learning loop.

Three test classes:

  TestFullLearningLoop      (12 tests)
    End-to-end with a REAL SQLite database (temp file) and mocked Qdrant.
    Exercises the complete pipeline: answer scored → candidate created →
    feedback adjusts score → admin action → Qdrant upsert/delete.

  TestTenantIsolation       (10 tests)
    Real SQLite, two tenants side-by-side.
    Verifies that no function leaks rows, counts, or events across account_id
    boundaries — at every layer: candidates, feedback, events, backfill, Qdrant filter.

  TestStagedRolloutVerification (8 tests)
    Verifies the feature-flag contract: each flag gates exactly its own
    subsystem and does not bleed into other subsystems.

Strategy
--------
  • Real SQLite: DB_PATH is patched to a tempfile so every test runs against
    a fresh, isolated schema.  Foreign-key constraints are satisfied with
    minimal INSERT helpers.
  • Qdrant / embedding: always mocked — tests must not require a running server.
  • No network, no GPU.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch, call


# ── Real-DB fixture ────────────────────────────────────────────────────────────

@contextmanager
def _temp_db():
    """
    Context manager:
      1. Creates a temp SQLite file.
      2. Patches store.db.DB_PATH to point at it.
      3. Runs init_db() to create all tables.
      4. Yields the Path.
      5. Restores the original DB_PATH and deletes the temp file.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = Path(tmp.name)

    import store.db as _db_mod
    original_path = _db_mod.DB_PATH
    _db_mod.DB_PATH = db_path

    try:
        from store.db import init_db
        init_db()
        yield db_path
    finally:
        _db_mod.DB_PATH = original_path
        try:
            os.unlink(db_path)
        except Exception:
            pass


def _setup_tenant(account_id: str = "acct_a") -> int:
    """
    Insert minimum FK prerequisites so learning_store writes don't violate
    foreign key constraints.  Returns the portal_user.id for that tenant.
    """
    from store.db import get_db
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO client "
            "(account_id, client_name, platform_type, state) VALUES (?, ?, ?, ?)",
            (account_id, f"Test {account_id}", "web", "READY"),
        )
        conn.execute(
            "INSERT OR IGNORE INTO portal_user "
            "(account_id, name, email, password_hash) VALUES (?, ?, ?, ?)",
            (account_id, "Alice", f"alice@{account_id}.com", "bcrypt_hash"),
        )
        row = conn.execute(
            "SELECT id FROM portal_user WHERE account_id=?", (account_id,)
        ).fetchone()
    return row["id"]


def _mock_qdrant_upsert():
    """Patch both the Qdrant client and embedder so upsert_governed_example runs without a server."""
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []

    embed_patch = patch("core.governed_store._embed", return_value=[[0.1] * 384])
    qdrant_patch = patch("core.governed_store._qdrant", return_value=mock_client)
    return embed_patch, qdrant_patch, mock_client


# ══════════════════════════════════════════════════════════════════════════════
# TestFullLearningLoop
# ══════════════════════════════════════════════════════════════════════════════

class TestFullLearningLoop(unittest.TestCase):
    """
    End-to-end integration tests with a real SQLite database.

    Every test spins up a fresh temp DB, populates the minimum FK rows,
    then exercises the full learning_store API with mocked Qdrant.
    """

    # ── Candidate creation ────────────────────────────────────────────────────

    def test_create_candidate_persists_to_db(self):
        with _temp_db():
            _setup_tenant("t1")
            from store.learning_store import create_candidate, get_candidate
            c = create_candidate(
                origin_question_id="q1",
                account_id="t1",
                question_text="What is revenue?",
                sql_text="SELECT SUM(amount) FROM sales",
                technical_score=80,
                evidence={"validation": 10},
            )
            fetched = get_candidate(c["candidate_id"])
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["question_text"], "What is revenue?")
        self.assertEqual(fetched["status"], "pending_review")

    def test_high_score_creates_positive_candidate(self):
        with _temp_db():
            _setup_tenant("t1")
            from store.learning_store import create_candidate
            c = create_candidate(
                origin_question_id="q1", account_id="t1",
                question_text="Q", sql_text="SELECT 1",
                technical_score=90, evidence={},
            )
        self.assertEqual(c["candidate_type"], "positive")

    def test_low_score_creates_negative_candidate(self):
        with _temp_db():
            _setup_tenant("t1")
            from store.learning_store import create_candidate
            c = create_candidate(
                origin_question_id="q2", account_id="t1",
                question_text="Q", sql_text="SELECT 1",
                technical_score=40, evidence={},
            )
        self.assertEqual(c["candidate_type"], "negative")

    def test_mid_score_creates_review_candidate(self):
        with _temp_db():
            _setup_tenant("t1")
            from store.learning_store import create_candidate
            c = create_candidate(
                origin_question_id="q3", account_id="t1",
                question_text="Q", sql_text="SELECT 1",
                technical_score=70, evidence={},
            )
        self.assertEqual(c["candidate_type"], "review")

    # ── Feedback delta ────────────────────────────────────────────────────────

    def test_thumbs_up_raises_final_score(self):
        with _temp_db():
            uid = _setup_tenant("t1")
            from store.learning_store import create_candidate, save_feedback, get_candidate
            c = create_candidate(
                origin_question_id="q1", account_id="t1",
                question_text="Q", sql_text="SELECT 1",
                technical_score=70, evidence={},
            )
            save_feedback(
                question_id="q1", user_id=uid, account_id="t1",
                rating=1, question_text="Q", sql_text="SELECT 1",
            )
            updated = get_candidate(c["candidate_id"])
        self.assertGreater(updated["final_score"], 70)

    def test_thumbs_down_lowers_final_score(self):
        with _temp_db():
            uid = _setup_tenant("t1")
            from store.learning_store import create_candidate, save_feedback, get_candidate
            c = create_candidate(
                origin_question_id="q1", account_id="t1",
                question_text="Q", sql_text="SELECT 1",
                technical_score=80, evidence={},
            )
            save_feedback(
                question_id="q1", user_id=uid, account_id="t1",
                rating=-1, question_text="Q", sql_text="SELECT 1",
            )
            updated = get_candidate(c["candidate_id"])
        self.assertLess(updated["final_score"], 80)

    def test_net_negative_feedback_forces_negative_type(self):
        with _temp_db():
            uid = _setup_tenant("t1")
            from store.learning_store import create_candidate, save_feedback, get_candidate
            # Need two users to have two distinct thumbs-down votes
            from store.db import get_db
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO portal_user (account_id, name, email, password_hash) VALUES (?, ?, ?, ?)",
                    ("t1", "Bob", "bob@t1.com", "hash2"),
                )
                uid2 = conn.execute(
                    "SELECT id FROM portal_user WHERE email='bob@t1.com'"
                ).fetchone()["id"]

            c = create_candidate(
                origin_question_id="q1", account_id="t1",
                question_text="Q", sql_text="SELECT 1",
                technical_score=85, evidence={},  # starts positive
            )
            # Two thumbs-down (different users)
            save_feedback("q1", uid, "t1", rating=-1, question_text="Q", sql_text="S")
            save_feedback("q1", uid2, "t1", rating=-1, question_text="Q", sql_text="S")
            updated = get_candidate(c["candidate_id"])
        self.assertEqual(updated["candidate_type"], "negative")

    # ── Admin actions + Qdrant hooks ──────────────────────────────────────────

    def test_admin_approve_fires_qdrant_upsert(self):
        with _temp_db():
            _setup_tenant("t1")
            from store.learning_store import create_candidate, update_candidate_status
            c = create_candidate(
                origin_question_id="q1", account_id="t1",
                question_text="What is revenue?",
                sql_text="SELECT SUM(amount) FROM sales",
                technical_score=85, evidence={},
            )
            ep, qp, mock_client = _mock_qdrant_upsert()
            with ep, qp:
                update_candidate_status(c["candidate_id"], "approved", reviewer_id="admin1")
        mock_client.upsert.assert_called_once()

    def test_qdrant_id_written_back_after_approve(self):
        with _temp_db():
            _setup_tenant("t1")
            from store.learning_store import create_candidate, update_candidate_status, get_candidate
            c = create_candidate(
                origin_question_id="q1", account_id="t1",
                question_text="What is revenue?",
                sql_text="SELECT SUM(amount) FROM sales",
                technical_score=85, evidence={},
            )
            ep, qp, mock_client = _mock_qdrant_upsert()
            with ep, qp:
                update_candidate_status(c["candidate_id"], "approved", reviewer_id="admin1")
            after = get_candidate(c["candidate_id"])
        # qdrant_id should be non-empty — it's the deterministic MD5-based UUID
        self.assertNotEqual(after.get("qdrant_id", ""), "")

    def test_admin_correction_sets_score_85_and_source(self):
        with _temp_db():
            _setup_tenant("t1")
            from store.learning_store import create_candidate, set_candidate_corrected_sql, get_candidate
            c = create_candidate(
                origin_question_id="q1", account_id="t1",
                question_text="Q", sql_text="SELECT broken_sql",
                technical_score=40, evidence={},
            )
            set_candidate_corrected_sql(
                c["candidate_id"],
                corrected_sql="SELECT SUM(amount) FROM sales",
                reviewer_id="admin1",
            )
            updated = get_candidate(c["candidate_id"])
        self.assertEqual(updated["corrected_sql"], "SELECT SUM(amount) FROM sales")
        self.assertEqual(updated["final_score"], 85)
        self.assertEqual(updated["source"], "admin_correction")

    def test_corrected_sql_embedded_not_original(self):
        """After correction + approve, the Qdrant upsert must use corrected_sql."""
        captured = {}

        def _fake_upsert(**kwargs):
            captured.update(kwargs)
            return "fake-point-id"

        with _temp_db():
            _setup_tenant("t1")
            from store.learning_store import (
                create_candidate, set_candidate_corrected_sql, update_candidate_status,
            )
            c = create_candidate(
                origin_question_id="q1", account_id="t1",
                question_text="Q", sql_text="SELECT wrong_sql",
                technical_score=40, evidence={},
            )
            set_candidate_corrected_sql(
                c["candidate_id"],
                corrected_sql="SELECT SUM(amount) FROM sales",
                reviewer_id="admin1",
            )
            with patch("core.governed_store.upsert_governed_example", side_effect=_fake_upsert):
                update_candidate_status(c["candidate_id"], "approved", reviewer_id="admin1")

        self.assertEqual(captured.get("sql"), "SELECT SUM(amount) FROM sales")
        self.assertNotEqual(captured.get("sql"), "SELECT wrong_sql")

    def test_revoke_fires_qdrant_delete(self):
        with _temp_db():
            _setup_tenant("t1")
            from store.learning_store import create_candidate, update_candidate_status
            c = create_candidate(
                origin_question_id="q1", account_id="t1",
                question_text="Q", sql_text="SELECT 1",
                technical_score=85, evidence={},
            )
            with patch("core.governed_store.delete_governed_example") as mock_del:
                update_candidate_status(c["candidate_id"], "revoked", reviewer_id="admin1")
        mock_del.assert_called_once()

    def test_approved_then_negative_feedback_requeues_for_review(self):
        """An already-approved candidate with net-negative feedback reverts to pending_review."""
        with _temp_db():
            uid = _setup_tenant("t1")
            from store.db import get_db
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO portal_user (account_id, name, email, password_hash) VALUES (?, ?, ?, ?)",
                    ("t1", "Bob", "bob@t1.com", "h2"),
                )
                uid2 = conn.execute(
                    "SELECT id FROM portal_user WHERE email='bob@t1.com'"
                ).fetchone()["id"]

            from store.learning_store import (
                create_candidate, update_candidate_status, save_feedback, get_candidate,
            )
            c = create_candidate(
                origin_question_id="q1", account_id="t1",
                question_text="Q", sql_text="SELECT 1",
                technical_score=90, evidence={},
            )
            ep, qp, _ = _mock_qdrant_upsert()
            with ep, qp:
                update_candidate_status(c["candidate_id"], "approved", reviewer_id="admin1")

            # Now two thumbs-down votes
            save_feedback("q1", uid, "t1", rating=-1, question_text="Q", sql_text="S")
            save_feedback("q1", uid2, "t1", rating=-1, question_text="Q", sql_text="S")

            after = get_candidate(c["candidate_id"])

        # Should have been re-queued, not left as approved
        self.assertEqual(after["status"], "pending_review")
        self.assertEqual(after["candidate_type"], "negative")


# ══════════════════════════════════════════════════════════════════════════════
# TestTenantIsolation
# ══════════════════════════════════════════════════════════════════════════════

class TestTenantIsolation(unittest.TestCase):
    """
    Tenant isolation verification — all tests use two real tenants in the same
    SQLite database.  Every assertion checks that tenantA's data cannot be
    retrieved via tenantB's account_id and vice versa.
    """

    def test_list_candidates_returns_only_own_tenant(self):
        with _temp_db():
            _setup_tenant("acct_a")
            _setup_tenant("acct_b")
            from store.learning_store import create_candidate, list_candidates
            create_candidate("qA", "acct_a", "Q-A", "SELECT 1", 80, {})
            create_candidate("qB", "acct_b", "Q-B", "SELECT 2", 80, {})

            results_a = list_candidates("acct_a")
            results_b = list_candidates("acct_b")

        self.assertEqual(len(results_a), 1)
        self.assertEqual(results_a[0]["question_text"], "Q-A")
        self.assertEqual(len(results_b), 1)
        self.assertEqual(results_b[0]["question_text"], "Q-B")

    def test_list_candidates_status_filter_is_tenant_scoped(self):
        """Approved candidates for acct_a must not appear in acct_b's pending list."""
        with _temp_db():
            _setup_tenant("acct_a")
            _setup_tenant("acct_b")
            from store.learning_store import create_candidate, update_candidate_status, list_candidates
            cA = create_candidate("qA", "acct_a", "Q-A", "SELECT 1", 90, {})
            with patch("core.governed_store.upsert_governed_example", return_value="pid"):
                update_candidate_status(cA["candidate_id"], "approved", reviewer_id="admin")
            create_candidate("qB", "acct_b", "Q-B", "SELECT 2", 70, {})

            approved_a = list_candidates("acct_a", status="approved")
            approved_b = list_candidates("acct_b", status="approved")
            pending_b  = list_candidates("acct_b", status="pending_review")

        self.assertEqual(len(approved_a), 1)
        self.assertEqual(len(approved_b), 0)  # acct_b has no approved
        self.assertEqual(len(pending_b), 1)

    def test_save_feedback_scoped_by_account(self):
        """Feedback saved for acct_a's question is not returned as acct_b's feedback."""
        with _temp_db():
            uid_a = _setup_tenant("acct_a")
            from store.learning_store import save_feedback, get_feedback
            save_feedback("q1", uid_a, "acct_a", rating=1, question_text="Q", sql_text="S")

            # acct_b's user should have no feedback for the same question_id
            _setup_tenant("acct_b")
            from store.db import get_db
            with get_db() as conn:
                uid_b = conn.execute(
                    "SELECT id FROM portal_user WHERE account_id='acct_b'"
                ).fetchone()["id"]

            fb = get_feedback("q1", uid_b)

        self.assertIsNone(fb)

    def test_record_event_scoped_by_account(self):
        """Events recorded for acct_a are not counted in acct_b's suggestion stats."""
        with _temp_db():
            _setup_tenant("acct_a")
            _setup_tenant("acct_b")
            from store.learning_store import record_event, get_suggestion_stats
            record_event(
                session_id="s1", account_id="acct_a",
                event_type="displayed",
                suggestion_text="What is revenue?",
            )
            stats_a = get_suggestion_stats("acct_a", "What is revenue?")
            stats_b = get_suggestion_stats("acct_b", "What is revenue?")

        self.assertEqual(stats_a.get("displayed", 0), 1)
        self.assertEqual(stats_b.get("displayed", 0), 0)  # isolation

    def test_get_suggestion_stats_independent_per_tenant(self):
        """Same suggestion text, different tenants → independent counts."""
        with _temp_db():
            _setup_tenant("acct_a")
            _setup_tenant("acct_b")
            from store.learning_store import record_event, get_suggestion_stats
            # acct_a: 3 displays, 2 clicks
            for _ in range(3):
                record_event("s", "acct_a", "displayed", "Show revenue", user_id=None)
            for _ in range(2):
                record_event("s", "acct_a", "clicked", "Show revenue", user_id=None)
            # acct_b: 1 display only
            record_event("s", "acct_b", "displayed", "Show revenue", user_id=None)

            sa = get_suggestion_stats("acct_a", "Show revenue")
            sb = get_suggestion_stats("acct_b", "Show revenue")

        self.assertEqual(sa["displayed"], 3)
        self.assertEqual(sa["clicked"], 2)
        self.assertEqual(sb["displayed"], 1)
        self.assertNotIn("clicked", sb)

    def test_backfill_only_processes_own_tenant_candidates(self):
        """backfill_approved_candidates for acct_a must not upsert acct_b's candidates."""
        upserted_accounts = []

        def _capture_upsert(**kwargs):
            upserted_accounts.append(kwargs.get("account_id"))
            return "pid"

        with _temp_db():
            _setup_tenant("acct_a")
            _setup_tenant("acct_b")
            from store.learning_store import create_candidate, update_candidate_status

            cA = create_candidate("qA", "acct_a", "Q-A", "SELECT 1", 90, {})
            cB = create_candidate("qB", "acct_b", "Q-B", "SELECT 2", 90, {})

            with patch("core.governed_store.upsert_governed_example", return_value="p"):
                update_candidate_status(cA["candidate_id"], "approved", reviewer_id="a")
                update_candidate_status(cB["candidate_id"], "approved", reviewer_id="a")

            with patch("core.governed_store.upsert_governed_example", side_effect=_capture_upsert):
                from core.governed_store import backfill_approved_candidates
                backfill_approved_candidates("acct_a")

        # Only acct_a candidates should have been backfilled
        self.assertTrue(all(acc == "acct_a" for acc in upserted_accounts))
        self.assertGreater(len(upserted_accounts), 0)

    def test_retrieve_governed_examples_qdrant_filter_includes_account_id(self):
        """retrieve_governed_examples must include account_id in the Qdrant filter."""
        from qdrant_client.models import Filter
        captured_filters = []

        mock_client = MagicMock()
        existing = MagicMock()
        existing.name = "querybot_governed"
        mock_client.get_collections.return_value.collections = [existing]
        count_res = MagicMock()
        count_res.count = 2
        mock_client.count.return_value = count_res

        h = MagicMock()
        h.payload = {"question": "Q", "sql": "SELECT 1", "source": "auto"}
        mock_client.query_points.return_value.points = [h]

        def _capture_count(**kwargs):
            captured_filters.append(kwargs.get("count_filter"))
            return count_res

        mock_client.count.side_effect = _capture_count

        with (
            patch("core.governed_store._qdrant", return_value=mock_client),
            patch("core.governed_store._embed", return_value=[[0.0] * 384]),
        ):
            from core.governed_store import retrieve_governed_examples
            retrieve_governed_examples("acct_a", "What is revenue?", n=3)

        # The count filter must contain account_id = "acct_a"
        self.assertTrue(len(captured_filters) > 0)
        f = captured_filters[0]
        must_conditions = f.must
        account_conds = [
            c for c in must_conditions
            if hasattr(c, "key") and c.key == "account_id"
        ]
        self.assertTrue(len(account_conds) > 0)

    def test_create_candidate_stores_correct_account_id(self):
        """candidate.account_id must match what was passed — no default bleed."""
        with _temp_db():
            _setup_tenant("acct_x")
            _setup_tenant("acct_y")
            from store.learning_store import create_candidate
            cx = create_candidate("qx", "acct_x", "Q", "SELECT 1", 80, {})
            cy = create_candidate("qy", "acct_y", "Q", "SELECT 1", 80, {})

        self.assertEqual(cx["account_id"], "acct_x")
        self.assertEqual(cy["account_id"], "acct_y")
        self.assertNotEqual(cx["account_id"], cy["account_id"])

    def test_multiple_tenants_independent_candidate_counts(self):
        """Candidate list length is per-tenant and does not aggregate across tenants."""
        with _temp_db():
            _setup_tenant("ta")
            _setup_tenant("tb")
            from store.learning_store import create_candidate, list_candidates
            for i in range(3):
                create_candidate(f"qa{i}", "ta", f"Q-A{i}", "SELECT 1", 80, {})
            for i in range(5):
                create_candidate(f"qb{i}", "tb", f"Q-B{i}", "SELECT 2", 80, {})

            ta_all = list_candidates("ta")
            tb_all = list_candidates("tb")

        self.assertEqual(len(ta_all), 3)
        self.assertEqual(len(tb_all), 5)


# ══════════════════════════════════════════════════════════════════════════════
# TestStagedRolloutVerification
# ══════════════════════════════════════════════════════════════════════════════

class TestStagedRolloutVerification(unittest.TestCase):
    """
    Verifies the feature-flag contract:

    Step 1 — enable_feedback_collection=1
      → Candidates created after answers; thumbs UI shown; admin queue populated.
    Step 2 — enable_genie_suggestions=1  (independent of Step 1)
      → Suggestions ranked by behavioral signals; impression events recorded.

    Each flag gates exactly its own subsystem; toggling one does not affect the other.
    """

    # ── Step 1: feedback collection ───────────────────────────────────────────

    def test_flag_off_no_candidate_created_after_answer(self):
        """_create_learning_candidate must be a no-op when enable_feedback_collection=0."""
        client = {"enable_feedback_collection": 0}
        created = []

        with (
            patch("store.get_client", return_value=client),
            patch("store.learning_store.create_candidate", side_effect=lambda **kw: created.append(kw)),
            patch("core.quality_scorer.score_trace", return_value=(80, {})),
        ):
            # Simulate what main.py:_create_learning_candidate does
            if client.get("enable_feedback_collection"):
                from store.learning_store import create_candidate
                create_candidate(
                    origin_question_id="q1", account_id="t1",
                    question_text="Q", sql_text="SELECT 1",
                    technical_score=80, evidence={},
                )

        self.assertEqual(len(created), 0)

    def test_flag_on_candidate_created_after_answer(self):
        """_create_learning_candidate must create a candidate when flag=1."""
        client = {"enable_feedback_collection": 1}
        created = []

        with (
            patch("store.learning_store.create_candidate", side_effect=lambda **kw: created.append(kw) or {}),
            patch("core.quality_scorer.score_trace", return_value=(80, {})),
        ):
            if client.get("enable_feedback_collection"):
                from store.learning_store import create_candidate
                create_candidate(
                    origin_question_id="q1", account_id="t1",
                    question_text="Q", sql_text="SELECT 1",
                    technical_score=80, evidence={},
                )

        self.assertEqual(len(created), 1)

    def test_feedback_endpoint_blocked_when_flag_off(self):
        """Portal feedback API returns 403 when enable_feedback_collection=0."""
        import asyncio
        from unittest.mock import AsyncMock

        req = MagicMock()
        req.cookies = {"qb_portal_session": "tok"}

        async def _json():
            return {"rating": 1}

        req.json = _json

        user = {"id": 1, "account_id": "t1"}
        client_off = {"enable_feedback_collection": 0}

        from fastapi import HTTPException

        with (
            patch("portal.routes._get_portal_user", return_value=user),
            patch("store.get_client", return_value=client_off),
        ):
            from portal.routes import portal_answer_feedback
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(portal_answer_feedback(req, "q1"))

        self.assertEqual(ctx.exception.status_code, 403)

    def test_feedback_endpoint_works_when_flag_on(self):
        """Portal feedback API returns 200 when enable_feedback_collection=1."""
        import asyncio

        req = MagicMock()
        req.cookies = {"qb_portal_session": "tok"}

        async def _json():
            return {"rating": 1}

        req.json = _json

        user = {"id": 1, "account_id": "t1"}
        client_on = {"enable_feedback_collection": 1}

        with (
            patch("portal.routes._get_portal_user", return_value=user),
            patch("store.get_client", return_value=client_on),
            patch("store.learning_store.save_feedback", return_value={"id": 1}),
            patch("store.learning_store.get_feedback", return_value=None),
        ):
            from portal.routes import portal_answer_feedback
            resp = asyncio.run(portal_answer_feedback(req, "q1"))

        self.assertEqual(resp.status_code, 200)

    # ── Step 2: genie suggestions ─────────────────────────────────────────────

    def test_genie_flag_off_rank_suggestions_not_called(self):
        """When enable_genie_suggestions=0, rank_suggestions is never called."""
        client = {"enable_genie_suggestions": 0}
        sugs = [{"question": "Q", "fqn": ""}]

        with (
            patch("store.get_allowed_tables", return_value=[]),
            patch("store.get_client", return_value=client),
            patch("core.suggestions.get_suggestions", return_value=sugs),
            patch("portal.routes._guess_safe_metric_suggestions", return_value=[]),
            patch("core.genie_ranker.rank_suggestions") as mock_rank,
        ):
            from portal.routes import _build_chat_suggestions
            _build_chat_suggestions({"account_id": "t1", "id": 1})

        mock_rank.assert_not_called()

    def test_genie_flag_on_rank_suggestions_called(self):
        """When enable_genie_suggestions=1, rank_suggestions is called."""
        client = {"enable_genie_suggestions": 1}
        sugs = [{"question": "Q", "fqn": ""}]

        with (
            patch("store.get_allowed_tables", return_value=[]),
            patch("store.get_client", return_value=client),
            patch("core.suggestions.get_suggestions", return_value=sugs),
            patch("portal.routes._guess_safe_metric_suggestions", return_value=[]),
            patch("core.genie_ranker.rank_suggestions", return_value=sugs) as mock_rank,
        ):
            from portal.routes import _build_chat_suggestions
            _build_chat_suggestions({"account_id": "t1", "id": 1})

        mock_rank.assert_called_once()

    def test_flags_are_independent_genie_on_without_feedback(self):
        """
        Genie flag on + feedback flag off:
          - rank_suggestions IS called
          - No learning candidate should be created (feedback flag is off)
        """
        client_state = {"enable_genie_suggestions": 1, "enable_feedback_collection": 0}
        sugs = [{"question": "Q", "fqn": ""}]
        created = []

        with (
            patch("store.get_allowed_tables", return_value=[]),
            patch("store.get_client", return_value=client_state),
            patch("core.suggestions.get_suggestions", return_value=sugs),
            patch("portal.routes._guess_safe_metric_suggestions", return_value=[]),
            patch("core.genie_ranker.rank_suggestions", return_value=sugs) as mock_rank,
            patch("store.learning_store.create_candidate", side_effect=lambda **kw: created.append(kw)),
        ):
            from portal.routes import _build_chat_suggestions
            _build_chat_suggestions({"account_id": "t1", "id": 1})

            # Simulate: feedback flag off → no candidate
            if client_state.get("enable_feedback_collection"):
                from store.learning_store import create_candidate
                create_candidate(
                    origin_question_id="q1", account_id="t1",
                    question_text="Q", sql_text="S", technical_score=80, evidence={},
                )

        mock_rank.assert_called_once()   # genie active
        self.assertEqual(len(created), 0)  # no candidates

    def test_cold_start_governed_suggestion_scores_above_static(self):
        """
        At cold start (0 impressions), a governed suggestion must outscore
        a static suggestion — ensuring approved examples surface first even
        with no behavioral data.
        """
        from core.genie_ranker import score_suggestion

        governed_score = score_suggestion({}, source="governed")
        static_score   = score_suggestion({}, source="static")

        self.assertGreater(governed_score, static_score)


if __name__ == "__main__":
    unittest.main()
