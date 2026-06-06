"""
store/learning_store.py

CRUD layer for the self-learning loop tables:
  answer_feedback, learning_candidate, recommendation_event.

Design principles
─────────────────
• Every write is idempotent where the schema allows (UNIQUE constraints,
  INSERT OR REPLACE, ON CONFLICT DO UPDATE).
• No raw result-row values are accepted or stored by any function here.
• Re-scoring on feedback update: save_feedback() re-computes the linked
  learning_candidate score so the admin queue always reflects current sentiment.
• All functions are synchronous — they're called either in background tasks
  or in lightweight API handlers, never in the hot query path.

Public API
──────────
  Feedback
    save_feedback(question_id, user_id, account_id, …) → dict
    get_feedback(question_id, user_id) → dict | None
    list_feedback(question_id) → list[dict]

  Candidates
    create_candidate(origin_question_id, account_id, …) → dict
    get_candidate(candidate_id) → dict | None
    list_candidates(account_id, status=None, limit=50) → list[dict]
    update_candidate_score(candidate_id, score, delta, evidence) → None
    update_candidate_status(candidate_id, status, reviewer_id, note) → None
    set_candidate_corrected_sql(candidate_id, corrected_sql, reviewer_id) → None
    recompute_candidate_score_from_feedback(question_id) → None

  Recommendation events
    record_event(session_id, user_id, account_id, event_type, …) → None
    get_suggestion_stats(account_id, suggestion_text) → dict
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from store.db import get_db

log = logging.getLogger("querybot.learning_store")


# ══════════════════════════════════════════════════════════════════════════════
# Answer feedback
# ══════════════════════════════════════════════════════════════════════════════

VALID_REASON_CODES = frozenset({
    "wrong_metric", "wrong_dimension", "wrong_filter", "wrong_join",
    "wrong_data", "incomplete", "confusing", "expected_data_missing", "other",
})


def save_feedback(
    question_id: str,
    user_id: int,
    account_id: str,
    rating: int,                    # 1 = up, -1 = down
    *,
    reason_code: str = "other",
    comment: str = "",
    question_text: str = "",
    sql_text: str = "",
    schema_scope: str = "",
) -> dict[str, Any]:
    """
    Upsert a feedback row for (question_id, user_id).

    An existing row is updated (rating + reason_code + comment) so a user
    can change their mind; updated_at is refreshed.

    After saving, recomputes the linked learning_candidate score so the
    admin queue reflects the current sentiment immediately (B6 = Option A).

    Returns the saved feedback as a dict.
    """
    if rating not in (1, -1):
        raise ValueError(f"rating must be 1 or -1, got {rating!r}")
    if reason_code not in VALID_REASON_CODES:
        reason_code = "other"

    now = time.strftime("%Y-%m-%dT%H:%M:%S")

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO answer_feedback
                (question_id, user_id, account_id, schema_scope, rating,
                 reason_code, comment, question_text, sql_text,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(question_id, user_id) DO UPDATE SET
                rating        = excluded.rating,
                reason_code   = excluded.reason_code,
                comment       = excluded.comment,
                updated_at    = excluded.updated_at
            """,
            (question_id, user_id, account_id, schema_scope, rating,
             reason_code, comment, question_text, sql_text, now, now),
        )
        row = conn.execute(
            "SELECT * FROM answer_feedback WHERE question_id=? AND user_id=?",
            (question_id, user_id),
        ).fetchone()
        result = dict(row) if row else {}

    # Re-score the linked candidate to reflect the new feedback state.
    # Fire-and-forget — any error is logged but never surfaced to the caller.
    try:
        recompute_candidate_score_from_feedback(question_id, account_id)
    except Exception as exc:
        log.warning(
            "learning_store: re-score after feedback failed for %s: %s",
            question_id, exc,
        )

    return result


def get_feedback(question_id: str, user_id: int) -> dict | None:
    """Return one user's feedback for a given question, or None."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM answer_feedback WHERE question_id=? AND user_id=?",
            (question_id, user_id),
        ).fetchone()
    return dict(row) if row else None


def list_feedback(question_id: str) -> list[dict]:
    """Return all feedback rows for a question (all users)."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM answer_feedback WHERE question_id=? ORDER BY created_at DESC",
            (question_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# Learning candidates
# ══════════════════════════════════════════════════════════════════════════════

def create_candidate(
    origin_question_id: str,
    account_id: str,
    question_text: str,
    sql_text: str,
    technical_score: int,
    evidence: dict,
    *,
    schema_scope: str = "",
    semantic_model_version: str = "",
    metric_version: str = "",
    schema_version: str = "",
    candidate_type: str = "review",
    source: str = "auto",
) -> dict[str, Any]:
    """
    Insert a new learning_candidate row.

    candidate_id is auto-generated (12-char hex UUID fragment).
    final_score = technical_score (no feedback yet).

    Returns the full candidate dict.
    """
    from core.quality_scorer import classify_score

    cid  = uuid.uuid4().hex[:12]
    now  = time.strftime("%Y-%m-%dT%H:%M:%S")
    ctype = classify_score(technical_score) if candidate_type == "review" else candidate_type

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO learning_candidate (
                candidate_id, origin_question_id, account_id, schema_scope,
                candidate_type, question_text, sql_text,
                technical_score, feedback_delta, final_score,
                evidence, status, source,
                semantic_model_version, metric_version, schema_version,
                created_at, updated_at
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, 0, ?,
                ?, 'pending_review', ?,
                ?, ?, ?,
                ?, ?
            )
            """,
            (
                cid, origin_question_id, account_id, schema_scope,
                ctype, question_text, sql_text,
                technical_score, technical_score,
                json.dumps(evidence, default=str), source,
                semantic_model_version, metric_version, schema_version,
                now, now,
            ),
        )
        row = conn.execute(
            "SELECT * FROM learning_candidate WHERE candidate_id=?", (cid,)
        ).fetchone()

    log.info(
        "learning_store: created candidate %s (type=%s score=%d) for question %s",
        cid, ctype, technical_score, origin_question_id[:16],
    )
    return dict(row) if row else {}


def get_candidate(candidate_id: str) -> dict | None:
    """Return a candidate by its candidate_id, or None."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM learning_candidate WHERE candidate_id=?",
            (candidate_id,),
        ).fetchone()
    return dict(row) if row else None


def list_candidates(
    account_id: str,
    status: str | None = None,
    candidate_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """
    Return candidates for an account, newest first.

    Optional filters: status, candidate_type.
    """
    params: list[Any] = [account_id]
    clauses = ["account_id = ?"]

    if status:
        clauses.append("status = ?")
        params.append(status)
    if candidate_type:
        clauses.append("candidate_type = ?")
        params.append(candidate_type)

    params += [limit, offset]
    sql = (
        f"SELECT * FROM learning_candidate WHERE {' AND '.join(clauses)} "
        f"ORDER BY created_at DESC LIMIT ? OFFSET ?"
    )
    with get_db() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def update_candidate_score(
    candidate_id: str,
    final_score: int,
    feedback_delta: int,
    evidence: dict,
    positive_votes: int = 0,
    negative_votes: int = 0,
) -> None:
    """
    Update the score fields on an existing candidate.

    Also reclassifies candidate_type based on the new score.
    If the candidate is already approved and the new score is "negative",
    flags it with status = 'pending_review' for re-evaluation rather than
    auto-revoking.
    """
    from core.quality_scorer import classify_score

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    has_neg = negative_votes > positive_votes
    new_type = classify_score(final_score, has_net_negative_feedback=has_neg)

    with get_db() as conn:
        current = conn.execute(
            "SELECT status, candidate_type FROM learning_candidate WHERE candidate_id=?",
            (candidate_id,),
        ).fetchone()
        if not current:
            return

        # If already approved and now negative → re-queue for review, don't auto-revoke
        current_status = current["status"]
        new_status = current_status
        if new_type == "negative" and current_status == "approved":
            new_status = "pending_review"
            log.warning(
                "learning_store: approved candidate %s re-queued — "
                "score dropped to %d (negative feedback)",
                candidate_id, final_score,
            )

        conn.execute(
            """
            UPDATE learning_candidate SET
                final_score         = ?,
                feedback_delta      = ?,
                evidence            = ?,
                candidate_type      = ?,
                positive_vote_count = ?,
                negative_vote_count = ?,
                status              = ?,
                updated_at          = ?
            WHERE candidate_id = ?
            """,
            (
                final_score, feedback_delta,
                json.dumps(evidence, default=str),
                new_type, positive_votes, negative_votes,
                new_status, now, candidate_id,
            ),
        )


def update_candidate_status(
    candidate_id: str,
    status: str,
    reviewer_id: str = "",
    reviewer_note: str = "",
) -> None:
    """
    Set the admin-review status on a candidate.

    Valid statuses: pending_review | approved | rejected | known_failure | revoked
    Updates reviewed_at and promoted_at (on approval) automatically.
    """
    valid = {"pending_review", "approved", "rejected", "known_failure", "revoked"}
    if status not in valid:
        raise ValueError(f"Invalid status {status!r}. Must be one of {valid}")

    now = time.strftime("%Y-%m-%dT%H:%M:%S")

    with get_db() as conn:
        promoted_at_sql = ", promoted_at = ?" if status == "approved" else ""
        params: list[Any] = [
            status, reviewer_id, reviewer_note, now,
        ]
        if status == "approved":
            params.append(now)   # promoted_at
        params.append(candidate_id)

        conn.execute(
            f"""
            UPDATE learning_candidate SET
                status        = ?,
                reviewer_id   = ?,
                reviewer_note = ?,
                reviewed_at   = ?
                {promoted_at_sql}
            WHERE candidate_id = ?
            """,
            params,
        )
    log.info(
        "learning_store: candidate %s status → %s (reviewer=%s)",
        candidate_id, status, reviewer_id or "system",
    )


def set_candidate_corrected_sql(
    candidate_id: str,
    corrected_sql: str,
    reviewer_id: str = "",
) -> None:
    """
    Store admin-entered corrected SQL.

    When corrected SQL is set, the candidate is automatically scored at 85
    with source='admin_correction' (B5 = trust-by-authority).
    The candidate type is reclassified as 'positive'.
    """
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    evidence = {
        "note": "admin_correction",
        "technical_score": 85,
        "final_score": 85,
        "source": "admin_correction",
    }
    with get_db() as conn:
        conn.execute(
            """
            UPDATE learning_candidate SET
                corrected_sql   = ?,
                source          = 'admin_correction',
                candidate_type  = 'positive',
                technical_score = 85,
                final_score     = 85,
                evidence        = ?,
                reviewer_id     = ?,
                updated_at      = ?
            WHERE candidate_id = ?
            """,
            (corrected_sql, json.dumps(evidence), reviewer_id, now, candidate_id),
        )
    log.info(
        "learning_store: corrected SQL set on candidate %s by %s",
        candidate_id, reviewer_id or "admin",
    )


def recompute_candidate_score_from_feedback(
    question_id: str,
    account_id: str,
) -> None:
    """
    Re-score the learning_candidate linked to question_id after a feedback change.

    Fetches current vote counts, recomputes the net delta, and calls
    update_candidate_score().  No-op if no candidate exists for this question.
    """
    from core.quality_scorer import net_feedback_delta

    with get_db() as conn:
        candidate = conn.execute(
            """
            SELECT candidate_id, technical_score, evidence
            FROM   learning_candidate
            WHERE  origin_question_id = ? AND account_id = ?
            LIMIT 1
            """,
            (question_id, account_id),
        ).fetchone()
        if not candidate:
            return

        votes = conn.execute(
            """
            SELECT
                SUM(CASE WHEN rating =  1 THEN 1 ELSE 0 END) AS pos,
                SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) AS neg
            FROM answer_feedback
            WHERE question_id = ?
            """,
            (question_id,),
        ).fetchone()

    pos = (votes["pos"] or 0) if votes else 0
    neg = (votes["neg"] or 0) if votes else 0
    delta = net_feedback_delta(pos, neg)

    base_score = candidate["technical_score"]
    final = max(0, base_score + delta)

    try:
        evidence = json.loads(candidate["evidence"] or "{}")
    except (ValueError, TypeError):
        evidence = {}
    evidence["feedback_delta"] = delta
    evidence["final_score"] = final
    evidence["positive_votes"] = pos
    evidence["negative_votes"] = neg

    update_candidate_score(
        candidate["candidate_id"],
        final_score=final,
        feedback_delta=delta,
        evidence=evidence,
        positive_votes=pos,
        negative_votes=neg,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Recommendation events
# ══════════════════════════════════════════════════════════════════════════════

def record_event(
    session_id: str,
    account_id: str,
    event_type: str,
    suggestion_text: str,
    *,
    user_id: int | None = None,
    suggestion_source: str = "static",
    result_question_id: str = "",
) -> None:
    """
    Append one recommendation event row.

    event_type must be: displayed | clicked | executed | successful | dismissed
    Fire-and-forget — any DB error is logged but never raised.
    """
    valid_events = {"displayed", "clicked", "executed", "successful", "dismissed"}
    if event_type not in valid_events:
        log.warning("record_event: unknown event_type %r — skipped", event_type)
        return

    try:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO recommendation_event
                    (session_id, user_id, account_id, event_type,
                     suggestion_text, suggestion_source, result_question_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, user_id, account_id, event_type,
                 suggestion_text[:500], suggestion_source, result_question_id),
            )
    except Exception as exc:
        log.warning("record_event failed: %s", exc)


def get_suggestion_stats(account_id: str, suggestion_text: str) -> dict[str, int]:
    """
    Return click/execution/dismissal counts for a suggestion text.

    Used by the learned suggestion ranker to apply behavioral signal weights.
    """
    try:
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT event_type, COUNT(*) AS cnt
                FROM   recommendation_event
                WHERE  account_id = ? AND suggestion_text = ?
                GROUP BY event_type
                """,
                (account_id, suggestion_text),
            ).fetchall()
    except Exception:
        return {}

    return {r["event_type"]: r["cnt"] for r in rows}
