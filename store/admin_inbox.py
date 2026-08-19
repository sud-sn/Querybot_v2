"""Cross-client counts for the admin console's inbox.

The console could tell an operator how many clients existed and what they had
cost, but not that anyone was waiting on them. Finding that out meant opening
each client and walking its tabs: access requests under Access, flagged answers
under Quality, unresolved conflicts under Model Health. Nothing aggregated, so
with more than a handful of clients the honest answer to "what needs me today"
was that nobody knew.

One grouped query per signal, keyed by account. Not a per-client call in a loop:
the existing count endpoints each serve a single account, and
``list_candidates(account_id, limit=9999)`` pulls whole rows through
``SELECT *`` just to take a length, which is fine for one badge and wrong for a
sweep across every client.

``semantic_feedback_pending_summary`` in semantic_feedback.py already did it
this way; these follow it.
"""

from __future__ import annotations

from .db import get_db


def _counts(sql: str, params: tuple = ()) -> dict[str, int]:
    """Run a two-column (account_id, count) query into a plain dict."""
    with get_db() as conn:
        rows = conn.execute(sql, params).fetchall()
    return {str(r[0]): int(r[1]) for r in rows if r[0]}


def pending_access_by_account() -> dict[str, int]:
    """People who asked for access and are still waiting.

    Ranked first in the inbox: a human is blocked, and unlike the other signals
    the delay is visible to them.
    """
    return _counts(
        """SELECT account_id, COUNT(*) FROM pending_platform_user
           WHERE status = 'pending' GROUP BY account_id"""
    )


def flagged_answers_by_account() -> dict[str, int]:
    """Answers a user marked wrong, awaiting review.

    These are the corrections the learning loop cannot apply until someone looks,
    so they age into stale wrong answers rather than resolving on their own.
    """
    return _counts(
        """SELECT account_id, COUNT(*) FROM learning_candidate
           WHERE status = 'pending_review' GROUP BY account_id"""
    )


def open_conflicts_by_account(severity: str = "") -> dict[str, int]:
    """Unresolved semantic-model conflicts.

    Pass a severity to count only that band; ERROR is the one worth surfacing on
    its own, since a WARNING backlog is normal and an ERROR is a model the
    compiler will not trust.
    """
    if severity:
        return _counts(
            """SELECT account_id, COUNT(*) FROM semantic_conflict
               WHERE status = 'open' AND severity = ? GROUP BY account_id""",
            (severity,),
        )
    return _counts(
        """SELECT account_id, COUNT(*) FROM semantic_conflict
           WHERE status = 'open' GROUP BY account_id"""
    )
