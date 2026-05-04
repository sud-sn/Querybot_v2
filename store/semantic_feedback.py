"""Pending user feedback for Semantic Layer field metadata."""

from __future__ import annotations

from store.db import get_db


def save_semantic_field_feedback(
    *,
    account_id: str,
    portal_user_id: int,
    table_fqn: str,
    schema_name: str,
    table_name: str,
    column_name: str,
    current_meaning: str = "",
    current_use_case: str = "",
    suggested_meaning: str = "",
    suggested_use_case: str = "",
    user_comment: str = "",
    confidence_score: int = 0,
) -> int:
    """Create a pending admin-review item for a semantic field correction."""
    with get_db() as conn:
        _ensure_table(conn)
        cur = conn.execute(
            """
            INSERT INTO semantic_field_feedback (
                account_id, portal_user_id, table_fqn, schema_name, table_name,
                column_name, current_meaning, current_use_case,
                suggested_meaning, suggested_use_case, user_comment,
                confidence_score, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """,
            (
                account_id,
                portal_user_id,
                table_fqn.upper(),
                schema_name,
                table_name,
                column_name,
                current_meaning,
                current_use_case,
                suggested_meaning,
                suggested_use_case,
                user_comment,
                int(confidence_score or 0),
            ),
        )
        return int(cur.lastrowid)


def list_semantic_field_feedback(
    account_id: str,
    *,
    status: str | None = None,
    limit: int = 200,
) -> list[dict]:
    with get_db() as conn:
        _ensure_table(conn)
        params: list = [account_id]
        where = "f.account_id = ?"
        if status:
            where += " AND f.status = ?"
            params.append(status)
        params.append(limit)
        rows = conn.execute(
            f"""
            SELECT f.*, u.name AS submitted_by, u.email AS submitted_email
            FROM semantic_field_feedback f
            LEFT JOIN portal_user u ON u.id = f.portal_user_id
            WHERE {where}
            ORDER BY f.created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]


def count_semantic_field_feedback(account_id: str, status: str = "pending") -> int:
    with get_db() as conn:
        _ensure_table(conn)
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM semantic_field_feedback WHERE account_id=? AND status=?",
            (account_id, status),
        ).fetchone()
        return int(row["n"] if row else 0)


def semantic_feedback_pending_summary() -> dict:
    """Return pending Semantic Layer feedback counts grouped by client."""
    with get_db() as conn:
        _ensure_table(conn)
        rows = conn.execute(
            """
            SELECT
                f.account_id,
                COALESCE(c.client_name, f.account_id) AS client_name,
                COUNT(*) AS pending,
                MAX(f.created_at) AS last_created_at
            FROM semantic_field_feedback f
            LEFT JOIN client c ON c.account_id = f.account_id
            WHERE f.status = 'pending'
            GROUP BY f.account_id, COALESCE(c.client_name, f.account_id)
            ORDER BY pending DESC, last_created_at DESC
            """
        ).fetchall()
    clients = [dict(r) for r in rows]
    return {
        "total": sum(int(c.get("pending") or 0) for c in clients),
        "clients": clients,
    }


def review_semantic_field_feedback(
    feedback_id: int,
    account_id: str,
    *,
    status: str,
    admin_note: str = "",
) -> bool:
    """Approve or reject one user-submitted Semantic Layer correction."""
    if status not in {"approved", "rejected"}:
        raise ValueError("status must be approved or rejected")

    with get_db() as conn:
        _ensure_table(conn)
        row = conn.execute(
            """
            SELECT id, table_fqn, column_name
            FROM semantic_field_feedback
            WHERE id = ? AND account_id = ?
            """,
            (feedback_id, account_id),
        ).fetchone()
        if not row:
            return False

        conn.execute(
            """
            UPDATE semantic_field_feedback
               SET status = ?,
                   admin_note = ?,
                   reviewed_at = datetime('now')
             WHERE id = ? AND account_id = ?
            """,
            (status, admin_note.strip(), feedback_id, account_id),
        )

        if status == "approved":
            conn.execute(
                """
                UPDATE semantic_field_feedback
                   SET status = 'rejected',
                       admin_note = CASE
                           WHEN COALESCE(admin_note, '') = ''
                           THEN 'Superseded by an approved edit for this field.'
                           ELSE admin_note
                       END,
                       reviewed_at = datetime('now')
                 WHERE account_id = ?
                   AND table_fqn = ?
                   AND UPPER(column_name) = UPPER(?)
                   AND status = 'pending'
                   AND id <> ?
                """,
                (account_id, row["table_fqn"], row["column_name"], feedback_id),
            )
        return True


def list_recent_reviewed_semantic_feedback(
    account_id: str,
    portal_user_id: int | None = None,
    *,
    limit: int = 50,
) -> list[dict]:
    """Latest approved/rejected Semantic Layer feedback visible to a portal user."""
    params: list = [account_id]
    where = "account_id = ? AND status IN ('approved','rejected')"
    if portal_user_id is not None:
        where += " AND portal_user_id = ?"
        params.append(int(portal_user_id))
    params.append(int(limit))
    with get_db() as conn:
        _ensure_table(conn)
        rows = conn.execute(
            f"""
            SELECT *
              FROM semantic_field_feedback
             WHERE {where}
             ORDER BY reviewed_at DESC, id DESC
             LIMIT ?
            """,
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def semantic_feedback_maps(account_id: str) -> tuple[dict[tuple[str, str], dict], set[tuple[str, str]]]:
    """
    Return latest approved feedback and fields with pending feedback.

    Keys are (TABLE_FQN, COLUMN_NAME_UPPER).
    """
    approved: dict[tuple[str, str], dict] = {}
    pending: set[tuple[str, str]] = set()
    rows = list_semantic_field_feedback(account_id, limit=1000)
    for row in rows:
        key = ((row.get("table_fqn") or "").upper(), (row.get("column_name") or "").upper())
        if not key[0] or not key[1]:
            continue
        if row.get("status") == "pending":
            pending.add(key)
        elif row.get("status") == "approved" and key not in approved:
            approved[key] = row
    return approved, pending


def _ensure_table(conn) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS semantic_field_feedback (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id           TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            portal_user_id       INTEGER REFERENCES portal_user(id) ON DELETE SET NULL,
            table_fqn            TEXT    NOT NULL,
            schema_name          TEXT    DEFAULT '',
            table_name           TEXT    NOT NULL,
            column_name          TEXT    NOT NULL,
            current_meaning      TEXT    DEFAULT '',
            current_use_case     TEXT    DEFAULT '',
            suggested_meaning    TEXT    DEFAULT '',
            suggested_use_case   TEXT    DEFAULT '',
            user_comment         TEXT    DEFAULT '',
            confidence_score     INTEGER DEFAULT 0,
            status               TEXT    NOT NULL DEFAULT 'pending'
                                 CHECK(status IN ('pending','approved','rejected')),
            admin_note           TEXT    DEFAULT '',
            created_at           TEXT    DEFAULT (datetime('now')),
            reviewed_at          TEXT    DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_semantic_feedback_account_status
            ON semantic_field_feedback(account_id, status, created_at);
        CREATE INDEX IF NOT EXISTS idx_semantic_feedback_field
            ON semantic_field_feedback(account_id, table_fqn, column_name, status);
        """
    )
