"""A metric someone asked for, that nobody has approved yet.

Metrics are the highest-authority layer in the semantic contract: saving one
triggers a synchronous recompile and every conflict detector, and from then on
it can change the answer to questions nobody has asked yet. So neither chat
surface writes one. They write a proposal here, and a human accept route is the
only thing that turns it into a metric — the same rule the entity-graph chat
already states in its docstring.

Modelled on ``store.config_store``'s graph_change_proposal functions, in its own
table for the reason given in the DDL.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from store.db import get_db

log = logging.getLogger("querybot.metric_proposal_store")

_JSON_COLUMNS = ("before_json", "payload_json", "validation_json", "dryrun_json")
_REVIEW_STATUSES = {"accepted", "rejected"}

# The fields whose drift makes a pending update proposal stale. A proposal says
# "change it from THIS to that"; if the live metric no longer matches THIS, the
# reviewer would be approving a diff against something that no longer exists.
GUARDED_METRIC_FIELDS = (
    "name", "sql_template", "formula_type", "required_columns",
    "base_table", "base_entity",
)


def _row_to_dict(row) -> dict[str, Any]:
    record = dict(row)
    for column in _JSON_COLUMNS:
        raw = record.get(column)
        try:
            record[column.removesuffix("_json")] = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            record[column.removesuffix("_json")] = {}
    return record


def create_metric_proposal(
    account_id: str,
    *,
    payload: dict,
    action: str = "create_metric",
    target_metric_id: int | None = None,
    before: dict | None = None,
    confidence_score: int = 0,
    generated_by: str = "portal_chat",
    requested_by_user_id: int | None = None,
    request_text: str = "",
    source_question: str = "",
    validation: dict | None = None,
    dryrun: dict | None = None,
    reason: str = "",
) -> int:
    """Record a request for a metric. Changes nothing that is live."""
    if action not in {"create_metric", "update_metric"}:
        raise ValueError(f"Unsupported metric proposal action: {action}")
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO metric_proposal
                (account_id, action, target_metric_id, before_json, payload_json,
                 status, confidence_score, generated_by, requested_by_user_id,
                 request_text, source_question, validation_json, dryrun_json, reason)
            VALUES (?,?,?,?,?,'pending',?,?,?,?,?,?,?,?)
            """,
            (
                account_id, action,
                int(target_metric_id) if target_metric_id else None,
                json.dumps(before or {}, default=str),
                json.dumps(payload or {}, default=str),
                max(0, min(100, int(confidence_score or 0))),
                generated_by,
                int(requested_by_user_id) if requested_by_user_id else None,
                str(request_text or "")[:500],
                str(source_question or "")[:500],
                json.dumps(validation or {}, default=str),
                json.dumps(dryrun or {}, default=str),
                reason,
            ),
        )
        return int(cur.lastrowid)


def list_metric_proposals(account_id: str, status: str = "") -> list[dict[str, Any]]:
    sql = "SELECT * FROM metric_proposal WHERE account_id=?"
    params: list[Any] = [account_id]
    if status:
        sql += " AND status=?"
        params.append(status)
    sql += " ORDER BY created_at DESC, id DESC"
    with get_db() as conn:
        return [_row_to_dict(row) for row in conn.execute(sql, params).fetchall()]


def get_metric_proposal(account_id: str, proposal_id: int) -> dict[str, Any] | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM metric_proposal WHERE id=? AND account_id=?",
            (int(proposal_id), account_id),
        ).fetchone()
        return _row_to_dict(row) if row else None


def review_metric_proposal(
    account_id: str,
    proposal_id: int,
    status: str,
    *,
    reviewed_by: str = "admin",
    review_note: str = "",
) -> bool:
    """Mark a proposal reviewed. Guarded on status='pending' so a double-submit
    cannot apply the same proposal twice."""
    if status not in _REVIEW_STATUSES:
        raise ValueError("Unsupported metric proposal review status.")
    with get_db() as conn:
        cur = conn.execute(
            """
            UPDATE metric_proposal
               SET status=?, reviewed_by=?, reviewed_at=datetime('now'), review_note=?
             WHERE id=? AND account_id=? AND status='pending'
            """,
            (status, reviewed_by, str(review_note or "")[:500], int(proposal_id), account_id),
        )
        return cur.rowcount > 0


def count_pending_metric_proposals(account_id: str) -> int:
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM metric_proposal WHERE account_id=? AND status='pending'",
            (account_id,),
        ).fetchone()
        return int(row[0] if row else 0)


def metric_has_drifted(live_metric: dict | None, before: dict | None) -> tuple[bool, list[str]]:
    """Whether the metric changed under a pending update proposal.

    Returns ``(drifted, fields)``. A proposal describes a change from a specific
    starting point; if someone edited the metric in between, accepting it would
    silently overwrite their edit with a diff computed against a version that no
    longer exists. The graph accept route answers this with a 409 and so does
    ours.
    """
    if not before:
        return False, []
    if not live_metric:
        return True, ["metric no longer exists"]
    changed = [
        field for field in GUARDED_METRIC_FIELDS
        if str(live_metric.get(field) or "") != str(before.get(field) or "")
    ]
    return bool(changed), changed
