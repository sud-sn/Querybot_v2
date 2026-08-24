"""Metric logic that is live for one person, in one thread, for a few hours.

The first of the two tracks in conversational metric authoring: a user
describes a calculation, it runs and answers their question straight away, and
it stays available for follow-ups in that thread. It is never written to
metric_registry and so never changes anybody else's answers -- becoming shared
takes a metric_proposal and a human.

Everything here is scoped to (account_id, session_id, portal_user_id) and
expires. The ACL is re-checked on every read rather than trusted from write
time, because a thread can outlive a permission.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from store.db import get_db

log = logging.getLogger("querybot.adhoc_metric_store")

# Long enough to survive a lunch break, short enough that a definition someone
# has forgotten making does not keep steering their answers. conversation_state
# uses 30 minutes, which is right for routing metadata and much too short for a
# definition the user is actively working with.
_DEFAULT_TTL_SECONDS = 4 * 60 * 60


def _ttl_seconds() -> int:
    try:
        return max(60, int(os.getenv("ADHOC_METRIC_TTL_SECONDS", _DEFAULT_TTL_SECONDS)))
    except (TypeError, ValueError):
        return _DEFAULT_TTL_SECONDS


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _normalise_table(value: Any) -> str:
    return str(value or "").strip().strip('[]"`').upper()


def save_session_metric_draft(
    account_id: str,
    session_id: str,
    portal_user_id: int,
    draft: dict[str, Any],
    *,
    source_tables: list[str] | None = None,
    validation: dict | None = None,
    dryrun: dict | None = None,
    confidence: float = 0.0,
    source_question: str = "",
    origin: str = "portal_chat",
) -> int:
    """Store a draft and supersede any earlier one in the same thread.

    One active draft per thread on purpose: two competing definitions of the
    same business idea, both silently steering the same follow-up question, is
    worse than losing the first one.
    """
    expires_at = (_now() + timedelta(seconds=_ttl_seconds())).isoformat()
    with get_db() as conn:
        conn.execute(
            """UPDATE session_metric_draft SET status='superseded'
                WHERE account_id=? AND session_id=? AND portal_user_id=? AND status='active'""",
            (account_id, str(session_id or ""), int(portal_user_id)),
        )
        cur = conn.execute(
            """
            INSERT INTO session_metric_draft
                (account_id, session_id, portal_user_id, name, synonyms, description,
                 sql_template, formula_type, result_format, base_table, required_columns,
                 allowed_dimensions, default_time_column, metric_builder_config,
                 source_tables, validation_json, dryrun_json, confidence,
                 source_question, origin, status, expires_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'active',?)
            """,
            (
                account_id, str(session_id or ""), int(portal_user_id),
                str(draft.get("name") or "").strip(),
                str(draft.get("synonyms") or "").strip(),
                str(draft.get("description") or "").strip(),
                str(draft.get("sql_template") or "").strip(),
                str(draft.get("formula_type") or "expression").strip(),
                str(draft.get("result_format") or "number").strip(),
                str(draft.get("base_table") or "").strip(),
                str(draft.get("required_columns") or "").strip(),
                str(draft.get("allowed_dimensions") or "").strip(),
                str(draft.get("default_time_column") or "").strip(),
                str(draft.get("metric_builder_config") or "").strip(),
                json.dumps([_normalise_table(t) for t in (source_tables or []) if t]),
                json.dumps(validation or {}, default=str),
                json.dumps(dryrun or {}, default=str),
                float(confidence or 0.0),
                str(source_question or "")[:500],
                origin, expires_at,
            ),
        )
        return int(cur.lastrowid)


def _row_to_metric(row) -> dict[str, Any]:
    """Shape a draft exactly like a metric_registry row, plus two markers.

    Identical shape is what lets it flow through resolve_metric_scope,
    _format_metric_formula_context and the validator with no special cases. The
    markers are how the few places that must know tell it apart.
    """
    record = dict(row)
    try:
        source_tables = json.loads(record.get("source_tables") or "[]")
    except (TypeError, ValueError):
        source_tables = []
    return {
        "id": None,                       # never a metric_registry id
        "name": record.get("name") or "",
        "synonyms": record.get("synonyms") or "",
        "description": record.get("description") or "",
        "sql_template": record.get("sql_template") or "",
        "formula_type": record.get("formula_type") or "expression",
        "result_format": record.get("result_format") or "number",
        "base_table": record.get("base_table") or "",
        "required_columns": record.get("required_columns") or "",
        "allowed_dimensions": record.get("allowed_dimensions") or "",
        "default_time_column": record.get("default_time_column") or "",
        "metric_builder_config": record.get("metric_builder_config") or "",
        "is_active": 1,
        "metric_status": "session_draft",
        # Markers. `_adhoc` keeps it out of usage counting and anything else
        # that assumes a registry row; `_pinned_thread_metric` keeps it in the
        # plan on a follow-up turn that does not repeat its name.
        "_adhoc": True,
        "_pinned_thread_metric": True,
        "_draft_id": int(record.get("id") or 0),
        "_source_tables": source_tables,
        "_resolved_source_tables": source_tables,
        "_confidence": float(record.get("confidence") or 0.0),
    }


def _tables_still_permitted(source_tables, allowed_tables) -> bool:
    """Does this draft still only touch tables the user may see?

    One rule, two readers. The listing read has always applied it; the read
    used by the promotion frame did not, so a user whose grant was revoked
    mid-thread could still turn that draft into a proposal naming a table they
    can no longer query. A draft is a live reference to data, and an ACL that
    only applies on the path someone remembered is not an ACL.
    """
    permitted = {_normalise_table(t) for t in allowed_tables or []}
    required = {_normalise_table(t) for t in source_tables or [] if t}
    if not required:
        return True
    # Compare on the bare table name too: an ACL may be stored unqualified
    # while a draft records a fully-qualified name.
    bare = {p.split(".")[-1] for p in permitted}
    return all(
        table in permitted or table.split(".")[-1] in bare
        for table in required
    )


def active_session_metrics(
    account_id: str,
    session_id: str,
    allowed_tables: set[str] | None,
) -> list[dict[str, Any]]:
    """Drafts still live in this thread, filtered by the caller's current ACL.

    ``allowed_tables`` is ``None`` for an unrestricted user (that is what
    ``store.get_allowed_tables`` returns for an admin), and a set of permitted
    table names otherwise. A draft whose source tables are no longer all
    permitted is dropped -- checked here rather than at write time, because a
    four-hour thread can outlive a permission.
    """
    if not session_id:
        return []
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM session_metric_draft
                WHERE account_id=? AND session_id=? AND status='active'
                  AND expires_at > ?
                ORDER BY id DESC""",
            (account_id, str(session_id), _now().isoformat()),
        ).fetchall()

    metrics: list[dict[str, Any]] = []
    for row in rows:
        metric = _row_to_metric(row)
        if allowed_tables is not None and not _tables_still_permitted(
            metric["_source_tables"], allowed_tables
        ):
            log.info(
                "Session metric draft %s dropped: its tables are no longer permitted",
                metric["_draft_id"],
            )
            continue
        metrics.append(metric)
    return metrics


def get_session_metric_draft(
    account_id: str, draft_id: int, allowed_tables=None,
) -> dict[str, Any] | None:
    """Read one draft.

    ``allowed_tables`` is optional only so existing non-user-facing callers
    keep working; every caller acting for a USER must pass it. Promotion is the
    one path that turns a draft into a durable artifact, and it was the path
    that never re-checked the ACL.
    """
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM session_metric_draft WHERE id=? AND account_id=?",
            (int(draft_id), account_id),
        ).fetchone()
    if not row:
        return None
    record = dict(row)
    for column in ("validation_json", "dryrun_json"):
        try:
            record[column.removesuffix("_json")] = json.loads(record.get(column) or "{}")
        except (TypeError, ValueError):
            record[column.removesuffix("_json")] = {}
    try:
        record["source_tables_list"] = json.loads(record.get("source_tables") or "[]")
    except (TypeError, ValueError):
        record["source_tables_list"] = []
    if allowed_tables is not None and not _tables_still_permitted(
        record["source_tables_list"], allowed_tables
    ):
        log.info(
            "Session metric draft %s withheld: its tables are no longer permitted",
            draft_id,
        )
        return None
    return record


def mark_draft_promoted(account_id: str, draft_id: int, proposal_id: int) -> bool:
    with get_db() as conn:
        cur = conn.execute(
            """UPDATE session_metric_draft SET status='promoted', proposal_id=?
                WHERE id=? AND account_id=? AND status='active'""",
            (int(proposal_id), int(draft_id), account_id),
        )
        return cur.rowcount > 0


def discard_session_metric_draft(account_id: str, draft_id: int, portal_user_id: int) -> bool:
    with get_db() as conn:
        cur = conn.execute(
            """UPDATE session_metric_draft SET status='withdrawn'
                WHERE id=? AND account_id=? AND portal_user_id=? AND status='active'""",
            (int(draft_id), account_id, int(portal_user_id)),
        )
        return cur.rowcount > 0


def purge_expired_drafts() -> int:
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM session_metric_draft WHERE expires_at <= ?", (_now().isoformat(),),
        )
        return cur.rowcount
