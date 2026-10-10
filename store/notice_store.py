"""Notices kept for portal readers (store/db.py, portal_notice): what core/notices.py tells someone.

A notice is kept whether or not its reader is online; the Notifications page lists them, newest first, and
reading the page marks them read. Each reader keeps their latest ``KEPT`` notices.
"""

from __future__ import annotations

from .db import get_db

KEPT = 200


def add_notice(account_id: str, user_id: int, kind: str, title: str, body: str = "", link: str = "") -> int:
    """Keep a notice for one reader; the oldest beyond ``KEPT`` go. A link is the portal's own (a path)."""
    link = str(link or "")
    if not link.startswith("/portal/"):
        link = ""                       # never a link out of the portal
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO portal_notice (account_id, user_id, kind, title, body, link)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (account_id, int(user_id), str(kind)[:40], str(title)[:300], str(body or "")[:2000], link[:500]),
        )
        notice_id = int(cur.lastrowid)
        conn.execute(
            """DELETE FROM portal_notice WHERE account_id=? AND user_id=? AND id NOT IN (
                   SELECT id FROM portal_notice WHERE account_id=? AND user_id=? ORDER BY id DESC LIMIT ?)""",
            (account_id, int(user_id), account_id, int(user_id), KEPT),
        )
    return notice_id


def list_notices(account_id: str, user_id: int, limit: int = 50) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, kind, title, body, link, created_at, read_at FROM portal_notice
                WHERE account_id=? AND user_id=? ORDER BY id DESC LIMIT ?""",
            (account_id, int(user_id), int(limit)),
        ).fetchall()
    return [dict(r) for r in rows]


def unread_notice_count(account_id: str, user_id: int) -> int:
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM portal_notice WHERE account_id=? AND user_id=? AND read_at IS NULL",
            (account_id, int(user_id)),
        ).fetchone()
    return int(row["n"] or 0) if row else 0


def mark_notices_read(account_id: str, user_id: int) -> int:
    with get_db() as conn:
        cur = conn.execute(
            """UPDATE portal_notice SET read_at=datetime('now')
                WHERE account_id=? AND user_id=? AND read_at IS NULL""",
            (account_id, int(user_id)),
        )
    return cur.rowcount or 0
