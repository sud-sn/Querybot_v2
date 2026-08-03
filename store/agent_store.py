"""Tenant-scoped persistence for the bounded portal agent runtime."""

from __future__ import annotations

import json
import re
from typing import Any
from uuid import uuid4

from store.db import get_db


_THREAD_KEY_RE = re.compile(r"[^A-Za-z0-9_-]")
_RUN_STATUSES = {
    "queued", "planning", "running", "reviewing", "waiting_for_user",
    "completed", "cancelled", "failed", "blocked", "budget_exceeded",
}
_STEP_STATUSES = {"pending", "running", "completed", "cancelled", "failed", "blocked", "waiting_for_user"}


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, default=str, separators=(",", ":"))
    except Exception:
        return "{}" if isinstance(value, dict) else "[]"


def _row_dict(row: Any) -> dict | None:
    return dict(row) if row is not None else None


def _thread_key(value: str) -> str:
    cleaned = _THREAD_KEY_RE.sub("", str(value or ""))[:80]
    return cleaned or "default"


def ensure_agent_thread(
    *,
    account_id: str,
    portal_user_id: int,
    external_thread_id: str,
    title: str = "",
) -> dict:
    """Return the caller-owned thread, creating it if needed.

    The composite owner key prevents a thread id supplied by one tenant or user
    from resolving another tenant's conversation.
    """
    external_thread_id = _thread_key(external_thread_id)
    with get_db() as conn:
        row = conn.execute(
            """SELECT * FROM agent_thread
               WHERE account_id=? AND portal_user_id=? AND external_thread_id=?""",
            (account_id, int(portal_user_id), external_thread_id),
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE agent_thread SET updated_at=datetime('now') WHERE id=? AND account_id=? AND portal_user_id=?",
                (row["id"], account_id, int(portal_user_id)),
            )
            return dict(row)

        thread_id = uuid4().hex
        conn.execute(
            """INSERT INTO agent_thread
               (id, account_id, portal_user_id, external_thread_id, title, status)
               VALUES (?, ?, ?, ?, ?, 'active')""",
            (
                thread_id,
                account_id,
                int(portal_user_id),
                external_thread_id,
                str(title or "")[:160],
            ),
        )
        row = conn.execute(
            "SELECT * FROM agent_thread WHERE id=? AND account_id=? AND portal_user_id=?",
            (thread_id, account_id, int(portal_user_id)),
        ).fetchone()
        return dict(row)


def create_agent_run(
    *,
    thread_id: str,
    account_id: str,
    portal_user_id: int,
    objective_sanitized: str,
    selected_schema: str = "",
    purpose_id: str = "",
    allowed_tables_snapshot: Any = None,
    policy_version: int = 0,
    contract_version: str = "",
    max_tool_calls: int = 5,
) -> dict:
    run_id = uuid4().hex
    with get_db() as conn:
        owner = conn.execute(
            "SELECT id FROM agent_thread WHERE id=? AND account_id=? AND portal_user_id=?",
            (thread_id, account_id, int(portal_user_id)),
        ).fetchone()
        if not owner:
            raise PermissionError("Agent thread does not belong to this user and workspace.")
        conn.execute(
            """INSERT INTO agent_run
               (id, thread_id, account_id, portal_user_id, objective_sanitized,
                status, current_stage, max_tool_calls, selected_schema, purpose_id,
                allowed_tables_snapshot, policy_version, contract_version, started_at)
               VALUES (?, ?, ?, ?, ?, 'running', 'understanding', ?, ?, ?, ?, ?, ?, datetime('now'))""",
            (
                run_id,
                thread_id,
                account_id,
                int(portal_user_id),
                str(objective_sanitized or "")[:4000],
                max(1, min(20, int(max_tool_calls or 5))),
                str(selected_schema or "")[:128],
                str(purpose_id or "")[:128],
                _json(allowed_tables_snapshot or []),
                max(0, int(policy_version or 0)),
                str(contract_version or "")[:128],
            ),
        )
        return dict(conn.execute(
            "SELECT * FROM agent_run WHERE id=? AND account_id=? AND portal_user_id=?",
            (run_id, account_id, int(portal_user_id)),
        ).fetchone())


def get_agent_run(*, account_id: str, portal_user_id: int, run_id: str) -> dict | None:
    with get_db() as conn:
        return _row_dict(conn.execute(
            "SELECT * FROM agent_run WHERE id=? AND account_id=? AND portal_user_id=?",
            (run_id, account_id, int(portal_user_id)),
        ).fetchone())


def get_waiting_agent_run(
    *, account_id: str, portal_user_id: int, external_thread_id: str,
) -> dict | None:
    with get_db() as conn:
        return _row_dict(conn.execute(
            """SELECT r.* FROM agent_run r
               JOIN agent_thread t ON t.id=r.thread_id
               WHERE r.account_id=? AND r.portal_user_id=?
                 AND t.external_thread_id=? AND r.status='waiting_for_user'
               ORDER BY r.created_at DESC LIMIT 1""",
            (account_id, int(portal_user_id), _thread_key(external_thread_id)),
        ).fetchone())


def add_agent_message(
    *,
    thread_id: str,
    run_id: str | None,
    account_id: str,
    portal_user_id: int,
    role: str,
    content_sanitized: str,
    message_type: str = "text",
    metadata: Any = None,
) -> int:
    role = str(role or "").lower()
    if role not in {"user", "assistant", "system", "tool"}:
        raise ValueError("Unsupported agent message role.")
    with get_db() as conn:
        owner = conn.execute(
            "SELECT id FROM agent_thread WHERE id=? AND account_id=? AND portal_user_id=?",
            (thread_id, account_id, int(portal_user_id)),
        ).fetchone()
        if not owner:
            raise PermissionError("Agent thread does not belong to this user and workspace.")
        if run_id:
            run = conn.execute(
                "SELECT id FROM agent_run WHERE id=? AND thread_id=? AND account_id=? AND portal_user_id=?",
                (run_id, thread_id, account_id, int(portal_user_id)),
            ).fetchone()
            if not run:
                raise PermissionError("Agent run does not belong to this user and thread.")
        cur = conn.execute(
            """INSERT INTO agent_message
               (thread_id, run_id, account_id, portal_user_id, role, message_type,
                content_sanitized, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                thread_id,
                run_id,
                account_id,
                int(portal_user_id),
                role,
                str(message_type or "text")[:64],
                str(content_sanitized or "")[:4000],
                _json(metadata or {}),
            ),
        )
        return int(cur.lastrowid)


def create_agent_step(
    *,
    run_id: str,
    account_id: str,
    portal_user_id: int,
    step_index: int,
    tool_name: str,
    label: str = "",
    detail: str = "",
    metadata: Any = None,
) -> int:
    with get_db() as conn:
        run = conn.execute(
            "SELECT max_tool_calls, tool_calls_used FROM agent_run WHERE id=? AND account_id=? AND portal_user_id=?",
            (run_id, account_id, int(portal_user_id)),
        ).fetchone()
        if not run:
            raise PermissionError("Agent run does not belong to this user and workspace.")
        if int(run["tool_calls_used"] or 0) >= int(run["max_tool_calls"] or 0):
            raise RuntimeError("Agent tool-call budget exceeded.")
        cur = conn.execute(
            """INSERT INTO agent_step
               (run_id, account_id, portal_user_id, step_index, tool_name, label,
                detail, status, metadata_json, started_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, datetime('now'))""",
            (
                run_id,
                account_id,
                int(portal_user_id),
                int(step_index),
                str(tool_name or "")[:96],
                str(label or "")[:200],
                str(detail or "")[:500],
                _json(metadata or {}),
            ),
        )
        conn.execute(
            """UPDATE agent_run SET tool_calls_used=tool_calls_used+1,
               current_stage=?, updated_at=datetime('now')
               WHERE id=? AND account_id=? AND portal_user_id=?""",
            (str(tool_name or "")[:96], run_id, account_id, int(portal_user_id)),
        )
        return int(cur.lastrowid)


def update_agent_step(
    *,
    run_id: str,
    account_id: str,
    portal_user_id: int,
    step_index: int,
    status: str | None = None,
    label: str | None = None,
    detail: str | None = None,
    answer_trace_id: int | None = None,
    policy_reason_code: str | None = None,
    metadata: Any = None,
) -> None:
    assignments: list[str] = ["updated_at=datetime('now')"]
    params: list[Any] = []
    if status is not None:
        if status not in _STEP_STATUSES:
            raise ValueError("Unsupported agent step status.")
        assignments.append("status=?")
        params.append(status)
        if status in {"completed", "cancelled", "failed", "blocked"}:
            assignments.append("completed_at=datetime('now')")
        else:
            assignments.append("completed_at=NULL")
    for key, value, limit in (
        ("label", label, 200),
        ("detail", detail, 500),
        ("policy_reason_code", policy_reason_code, 128),
    ):
        if value is not None:
            assignments.append(f"{key}=?")
            params.append(str(value)[:limit])
    if answer_trace_id is not None:
        assignments.append("answer_trace_id=?")
        params.append(int(answer_trace_id))
    if metadata is not None:
        assignments.append("metadata_json=?")
        params.append(_json(metadata))
    params.extend((run_id, account_id, int(portal_user_id), int(step_index)))
    with get_db() as conn:
        conn.execute(
            f"UPDATE agent_step SET {', '.join(assignments)} "
            "WHERE run_id=? AND account_id=? AND portal_user_id=? AND step_index=?",
            params,
        )


def update_agent_run(
    *,
    run_id: str,
    account_id: str,
    portal_user_id: int,
    status: str | None = None,
    current_stage: str | None = None,
    answer_trace_id: int | None = None,
    outcome_type: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    policy_version: int | None = None,
    contract_version: str | None = None,
) -> None:
    assignments: list[str] = ["updated_at=datetime('now')"]
    params: list[Any] = []
    if status is not None:
        if status not in _RUN_STATUSES:
            raise ValueError("Unsupported agent run status.")
        assignments.append("status=?")
        params.append(status)
        if status in {"completed", "cancelled", "failed", "blocked", "budget_exceeded"}:
            assignments.append("completed_at=datetime('now')")
        else:
            assignments.append("completed_at=NULL")
    for key, value, limit in (
        ("current_stage", current_stage, 96),
        ("outcome_type", outcome_type, 96),
        ("error_code", error_code, 128),
        ("error_message", error_message, 1000),
        ("contract_version", contract_version, 128),
    ):
        if value is not None:
            assignments.append(f"{key}=?")
            params.append(str(value)[:limit])
    if answer_trace_id is not None:
        assignments.append("answer_trace_id=?")
        params.append(int(answer_trace_id))
    if policy_version is not None:
        assignments.append("policy_version=?")
        params.append(max(0, int(policy_version)))
    params.extend((run_id, account_id, int(portal_user_id)))
    with get_db() as conn:
        cur = conn.execute(
            f"UPDATE agent_run SET {', '.join(assignments)} WHERE id=? AND account_id=? AND portal_user_id=?",
            params,
        )
        if cur.rowcount == 0:
            raise PermissionError("Agent run does not belong to this user and workspace.")


def list_agent_steps(*, account_id: str, portal_user_id: int, run_id: str) -> list[dict]:
    with get_db() as conn:
        return [dict(row) for row in conn.execute(
            """SELECT * FROM agent_step WHERE run_id=? AND account_id=? AND portal_user_id=?
               ORDER BY step_index""",
            (run_id, account_id, int(portal_user_id)),
        ).fetchall()]


def create_agent_subtask(
    *,
    parent_run_id: str,
    account_id: str,
    portal_user_id: int,
    objective_sanitized: str,
    tool_name: str,
    metadata: Any = None,
) -> dict:
    subtask_id = uuid4().hex
    with get_db() as conn:
        parent = conn.execute(
            """SELECT id FROM agent_run
                WHERE id=? AND account_id=? AND portal_user_id=?""",
            (parent_run_id, account_id, int(portal_user_id)),
        ).fetchone()
        if not parent:
            raise PermissionError("Parent agent run does not belong to this user and workspace.")
        conn.execute(
            """INSERT INTO agent_subtask
                   (id, parent_run_id, account_id, portal_user_id,
                    objective_sanitized, tool_name, status, metadata_json, started_at)
               VALUES (?, ?, ?, ?, ?, ?, 'running', ?, datetime('now'))""",
            (
                subtask_id, parent_run_id, account_id, int(portal_user_id),
                str(objective_sanitized or "")[:1000], str(tool_name or "")[:96],
                _json(metadata or {}),
            ),
        )
        row = conn.execute(
            "SELECT * FROM agent_subtask WHERE id=?", (subtask_id,)
        ).fetchone()
    return dict(row)


def finish_agent_subtask(
    *,
    subtask_id: str,
    parent_run_id: str,
    account_id: str,
    portal_user_id: int,
    status: str,
    result_summary: str = "",
    metadata: Any = None,
) -> None:
    if status not in {"completed", "failed", "blocked", "cancelled"}:
        raise ValueError("Unsupported agent subtask status.")
    assignments = [
        "status=?", "result_summary=?", "completed_at=datetime('now')",
        "updated_at=datetime('now')",
    ]
    params: list[Any] = [status, str(result_summary or "")[:2000]]
    if metadata is not None:
        assignments.append("metadata_json=?")
        params.append(_json(metadata))
    params.extend((subtask_id, parent_run_id, account_id, int(portal_user_id)))
    with get_db() as conn:
        cur = conn.execute(
            f"UPDATE agent_subtask SET {', '.join(assignments)} "
            "WHERE id=? AND parent_run_id=? AND account_id=? AND portal_user_id=?",
            params,
        )
        if cur.rowcount == 0:
            raise PermissionError("Agent subtask does not belong to this run.")


def list_agent_subtasks(
    *, account_id: str, portal_user_id: int, run_id: str
) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM agent_subtask
                WHERE parent_run_id=? AND account_id=? AND portal_user_id=?
                ORDER BY created_at, id""",
            (run_id, account_id, int(portal_user_id)),
        ).fetchall()
    return [dict(row) for row in rows]


def list_agent_subtasks_for_account(account_id: str, limit: int = 100) -> list[dict]:
    """Admin-facing analysis activity; contains metadata, never result rows."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT s.*, u.name AS user_name, u.email AS user_email,
                      r.objective_sanitized AS run_objective, r.created_at AS run_created_at
                 FROM agent_subtask s
                 JOIN agent_run r ON r.id=s.parent_run_id AND r.account_id=s.account_id
                 LEFT JOIN portal_user u ON u.id=s.portal_user_id
                WHERE s.account_id=? AND s.tool_name LIKE 'analysis_%'
                ORDER BY s.created_at DESC, s.id DESC LIMIT ?""",
            (account_id, max(1, min(int(limit or 100), 500))),
        ).fetchall()
    output = []
    for raw in rows:
        item = dict(raw)
        try:
            item["metadata"] = json.loads(item.get("metadata_json") or "{}")
        except (TypeError, ValueError):
            item["metadata"] = {}
        output.append(item)
    return output
