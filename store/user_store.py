"""
store/user_store.py

CRUD for portal users, groups, table access, registration tokens, and pinned charts.
All password handling uses SHA-256 (same as admin panel).
"""

import hashlib
import hmac
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from store.db import get_db

log = logging.getLogger("querybot.user_store")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _hash_pw(pw: str) -> str:
    salt = secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200000)
    return f"pbkdf2_sha256$200000${salt}${derived.hex()}"


def _verify_pw(stored_hash: str, password: str) -> bool:
    if not stored_hash:
        return False
    if stored_hash.startswith("pbkdf2_sha256$"):
        try:
            _, rounds, salt, expected = stored_hash.split("$", 3)
            derived = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(rounds))
            return hmac.compare_digest(expected, derived.hex())
        except Exception:
            return False
    legacy = hashlib.sha256(password.encode()).hexdigest()
    return hmac.compare_digest(stored_hash, legacy)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _expiry(hours: int = 48) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")


def generate_temp_password() -> str:
    """Generate a readable 10-char temp password."""
    chars = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(chars) for _ in range(10))


# ══════════════════════════════════════════════════════════════════════════════
# Groups
# ══════════════════════════════════════════════════════════════════════════════

def create_group(account_id: str, name: str, description: str = "") -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO user_group (account_id, name, description) VALUES (?,?,?)",
            (account_id, name, description)
        )
        gid = cur.lastrowid
    log.info("Created group '%s' (id=%d) for client %s", name, gid, account_id)
    return gid


def list_groups(account_id: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT g.*, COUNT(u.id) AS member_count FROM user_group g "
            "LEFT JOIN portal_user u ON u.group_id = g.id "
            "WHERE g.account_id = ? GROUP BY g.id ORDER BY g.name",
            (account_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_group(group_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM user_group WHERE id=?", (group_id,)).fetchone()
    return dict(row) if row else None


def update_group(group_id: int, name: str, description: str) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE user_group SET name=?, description=? WHERE id=?",
            (name, description, group_id)
        )


def delete_group(group_id: int) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM user_group WHERE id=?", (group_id,))


# ── Group table access ────────────────────────────────────────────────────────

def set_group_tables(group_id: int, account_id: str, tables: list[str]) -> None:
    """Replace the entire table access list for a group."""
    with get_db() as conn:
        conn.execute("DELETE FROM group_table_access WHERE group_id=?", (group_id,))
        for t in tables:
            conn.execute(
                "INSERT OR IGNORE INTO group_table_access (group_id, account_id, table_name) VALUES (?,?,?)",
                (group_id, account_id, t.upper())
            )
    log.info("Set %d tables for group %d", len(tables), group_id)


def get_group_tables(group_id: int) -> list[str]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT table_name FROM group_table_access WHERE group_id=? ORDER BY table_name",
            (group_id,)
        ).fetchall()
    return [r["table_name"] for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# Portal users
# ══════════════════════════════════════════════════════════════════════════════

def create_user(
    account_id: str,
    name: str,
    email: str,
    group_id: Optional[int] = None,
    role: str = "analyst",
    password: Optional[str] = None,
) -> tuple[int, str]:
    """
    Create a portal user.

    If `password` is supplied: that password is set and is_temp_pw=0 — the
    user lands directly on the chat page after login, no forced change.

    If `password` is not supplied: a random temp password is generated,
    is_temp_pw=1 — the user must change it on first login.

    Returns (user_id, plain_text_password). The plain password is shown ONCE
    in the admin UI — never stored.
    """
    if password and password.strip():
        plain   = password.strip()
        is_temp = 0
    else:
        plain   = generate_temp_password()
        is_temp = 1
    pw_hash = _hash_pw(plain)
    with get_db() as conn:
        cur = conn.execute("""
            INSERT INTO portal_user
                (account_id, group_id, name, email, password_hash, role, is_temp_pw)
            VALUES (?,?,?,?,?,?,?)
        """, (account_id, group_id, name, email.lower().strip(), pw_hash, role, is_temp))
        uid = cur.lastrowid
    log.info("Created user '%s' (id=%d) for client %s (temp_pw=%s)",
             email, uid, account_id, bool(is_temp))
    return uid, plain


def get_user(user_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT u.*, g.name AS group_name FROM portal_user u "
            "LEFT JOIN user_group g ON g.id = u.group_id "
            "WHERE u.id=?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_email(account_id: str, email: str) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT u.*, g.name AS group_name FROM portal_user u "
            "LEFT JOIN user_group g ON g.id = u.group_id "
            "WHERE u.account_id=? AND u.email=? AND u.is_active=1",
            (account_id, email.lower().strip())
        ).fetchone()
    return dict(row) if row else None


def get_user_by_zoom_id(zoom_user_id: str) -> Optional[dict]:
    """Look up portal user by their Zoom userId from webhook payload."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT u.*, g.name AS group_name FROM portal_user u "
            "LEFT JOIN user_group g ON g.id = u.group_id "
            "WHERE u.zoom_user_id=? AND u.is_active=1",
            (zoom_user_id,)
        ).fetchone()
    return dict(row) if row else None


def list_users(account_id: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT u.*, g.name AS group_name FROM portal_user u "
            "LEFT JOIN user_group g ON g.id = u.group_id "
            "WHERE u.account_id=? ORDER BY u.name",
            (account_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def update_user(
    user_id: int,
    name: Optional[str] = None,
    group_id: Optional[int] = None,
    role: Optional[str] = None,
    is_active: Optional[int] = None,
) -> None:
    fields, params = [], []
    if name is not None:
        fields.append("name=?"); params.append(name)
    if group_id is not None:
        fields.append("group_id=?"); params.append(group_id)
    if role is not None:
        fields.append("role=?"); params.append(role)
    if is_active is not None:
        fields.append("is_active=?"); params.append(is_active)
    if not fields:
        return
    fields.append("updated_at=datetime('now')")
    params.append(user_id)
    with get_db() as conn:
        conn.execute(f"UPDATE portal_user SET {','.join(fields)} WHERE id=?", params)


def change_password(user_id: int, new_password: str, is_temp: bool = False) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE portal_user SET password_hash=?, is_temp_pw=?, updated_at=datetime('now') WHERE id=?",
            (_hash_pw(new_password), 1 if is_temp else 0, user_id)
        )


def reset_user_password(user_id: int) -> str:
    """Admin resets a user's password. Returns new temp password."""
    temp_pw = generate_temp_password()
    change_password(user_id, temp_pw, is_temp=True)
    return temp_pw


def link_zoom_user(user_id: int, zoom_user_id: str) -> None:
    """Link a portal user account to their Zoom userId."""
    with get_db() as conn:
        conn.execute(
            "UPDATE portal_user SET zoom_user_id=?, updated_at=datetime('now') WHERE id=?",
            (zoom_user_id, user_id)
        )
    log.info("Linked portal user %d to Zoom userId %s", user_id, zoom_user_id)


def verify_password(user: dict, password: str) -> bool:
    return _verify_pw(user.get("password_hash", ""), password)


def delete_user(user_id: int) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM portal_user WHERE id=?", (user_id,))


# ── Individual table overrides ────────────────────────────────────────────────

def set_user_extra_tables(user_id: int, account_id: str, tables: list[str]) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM user_table_access WHERE user_id=?", (user_id,))
        for t in tables:
            conn.execute(
                "INSERT OR IGNORE INTO user_table_access (user_id, account_id, table_name) VALUES (?,?,?)",
                (user_id, account_id, t.upper())
            )


def get_user_extra_tables(user_id: int) -> list[str]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT table_name FROM user_table_access WHERE user_id=? ORDER BY table_name",
            (user_id,)
        ).fetchall()
    return [r["table_name"] for r in rows]


def get_allowed_tables(user: dict) -> Optional[set[str]]:
    """
    Return the full set of tables this user can query.
    Returns None if user is admin role (all tables allowed).
    Combines group tables + individual overrides.
    """
    if user.get("role") == "admin":
        return None  # None means unrestricted

    tables: set[str] = set()

    # Group tables
    if user.get("group_id"):
        for t in get_group_tables(user["group_id"]):
            tables.add(t.upper())

    # Individual overrides
    for t in get_user_extra_tables(user["id"]):
        tables.add(t.upper())

    return tables if tables else set()  # empty set = no access


# ══════════════════════════════════════════════════════════════════════════════
# Registration tokens
# ══════════════════════════════════════════════════════════════════════════════

def create_registration_token(account_id: str, zoom_user_id: str) -> str:
    """Create a one-time registration token valid for 48 hours."""
    token = secrets.token_urlsafe(32)
    with get_db() as conn:
        # Invalidate any existing unused tokens for this Zoom user
        conn.execute(
            "UPDATE registration_token SET used=1 WHERE account_id=? AND zoom_user_id=? AND used=0",
            (account_id, zoom_user_id)
        )
        conn.execute(
            "INSERT INTO registration_token (token, account_id, zoom_user_id, expires_at) VALUES (?,?,?,?)",
            (token, account_id, zoom_user_id, _expiry(48))
        )
    return token


def consume_registration_token(token: str) -> Optional[dict]:
    """Validate and consume a registration token. Returns token data or None."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM registration_token WHERE token=? AND used=0",
            (token,)
        ).fetchone()
        if not row:
            return None
        row = dict(row)
        # Check expiry
        expiry = datetime.strptime(row["expires_at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > expiry:
            return None
        # Mark used
        conn.execute("UPDATE registration_token SET used=1 WHERE token=?", (token,))
    return row


# ══════════════════════════════════════════════════════════════════════════════
# Pending platform users (admin-approval flow)
# ══════════════════════════════════════════════════════════════════════════════

def upsert_pending_user(
    account_id: str,
    platform_type: str,
    platform_user_id: str,
    display_name: str,
    conversation_ref: str,
) -> tuple[bool, dict]:
    """
    Insert a pending platform user row, or update the conversation_ref if
    the row already exists (so repeated messages keep the ref fresh).

    Returns (is_new: bool, row: dict).
    is_new=True  → first time this user messaged; caller should reply to them.
    is_new=False → already pending/approved/rejected; caller stays silent.
    """
    import json as _json
    with get_db() as conn:
        existing = conn.execute(
            "SELECT * FROM pending_platform_user WHERE account_id=? AND platform_user_id=?",
            (account_id, platform_user_id),
        ).fetchone()
        if existing:
            # Update conversation_ref so proactive message always goes to latest conv
            conn.execute(
                "UPDATE pending_platform_user SET conversation_ref=?, display_name=? "
                "WHERE account_id=? AND platform_user_id=?",
                (conversation_ref, display_name or dict(existing).get("display_name", ""),
                 account_id, platform_user_id),
            )
            row = conn.execute(
                "SELECT * FROM pending_platform_user WHERE account_id=? AND platform_user_id=?",
                (account_id, platform_user_id),
            ).fetchone()
            return False, dict(row)
        conn.execute(
            """INSERT INTO pending_platform_user
               (account_id, platform_type, platform_user_id, display_name, conversation_ref)
               VALUES (?,?,?,?,?)""",
            (account_id, platform_type, platform_user_id, display_name, conversation_ref),
        )
        row = conn.execute(
            "SELECT * FROM pending_platform_user WHERE account_id=? AND platform_user_id=?",
            (account_id, platform_user_id),
        ).fetchone()
    return True, dict(row)


def list_pending_users(account_id: str, status: str = "pending") -> list[dict]:
    """Return pending platform users for this account filtered by status."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM pending_platform_user WHERE account_id=? AND status=? "
            "ORDER BY created_at DESC",
            (account_id, status),
        ).fetchall()
    return [dict(r) for r in rows]


def get_pending_user_count(account_id: str) -> int:
    """Return count of users with status='pending' for dashboard badge."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM pending_platform_user WHERE account_id=? AND status='pending'",
            (account_id,),
        ).fetchone()
    return row[0] if row else 0


def approve_pending_user(
    pending_id: int,
    account_id: str,
    group_id: Optional[int],
    reviewer_id: str = "admin",
    reviewer_note: str = "",
) -> Optional[dict]:
    """
    Approve a pending platform user:
      1. Create a portal_user row (no password — they auth via platform).
      2. Link the platform_user_id as zoom_user_id on the new portal_user.
      3. Mark the pending row approved + set portal_user_id FK.

    Returns the new portal_user dict, or None if pending row not found.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        pending = conn.execute(
            "SELECT * FROM pending_platform_user WHERE id=? AND account_id=?",
            (pending_id, account_id),
        ).fetchone()
        if not pending:
            return None
        pending = dict(pending)

        # Create the portal_user.  A placeholder password hash is used (unusable
        # hash — they will never log in via the portal, only via Teams).
        import hashlib as _hashlib
        _placeholder_hash = _hashlib.sha256(b"__platform_user__").hexdigest()
        conn.execute(
            """INSERT OR IGNORE INTO portal_user
               (account_id, group_id, name, email, password_hash,
                zoom_user_id, role, is_temp_pw, is_active, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,0,1,?,?)""",
            (
                account_id,
                group_id,
                pending["display_name"] or pending["platform_user_id"],
                f"{pending['platform_user_id']}@platform.internal",
                _placeholder_hash,
                pending["platform_user_id"],
                "analyst",
                now, now,
            ),
        )
        portal_user_row = conn.execute(
            "SELECT * FROM portal_user WHERE account_id=? AND zoom_user_id=?",
            (account_id, pending["platform_user_id"]),
        ).fetchone()
        portal_user_id = portal_user_row["id"] if portal_user_row else None

        # Update group_id if we just hit the IGNORE path (user existed)
        if portal_user_row and group_id:
            conn.execute(
                "UPDATE portal_user SET group_id=?, updated_at=? WHERE id=?",
                (group_id, now, portal_user_id),
            )

        conn.execute(
            """UPDATE pending_platform_user SET
               status='approved', portal_user_id=?, reviewer_id=?,
               reviewer_note=?, reviewed_at=?
               WHERE id=?""",
            (portal_user_id, reviewer_id, reviewer_note, now, pending_id),
        )
    return dict(portal_user_row) if portal_user_row else None


def reject_pending_user(
    pending_id: int,
    account_id: str,
    reviewer_id: str = "admin",
    reviewer_note: str = "",
) -> bool:
    """Mark a pending user as rejected. Returns True if the row existed."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM pending_platform_user WHERE id=? AND account_id=?",
            (pending_id, account_id),
        ).fetchone()
        if not existing:
            return False
        conn.execute(
            """UPDATE pending_platform_user SET
               status='rejected', reviewer_id=?, reviewer_note=?, reviewed_at=?
               WHERE id=?""",
            (reviewer_id, reviewer_note, now, pending_id),
        )
    return True


def get_pending_user(pending_id: int, account_id: str) -> Optional[dict]:
    """Return a single pending_platform_user row, or None."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM pending_platform_user WHERE id=? AND account_id=?",
            (pending_id, account_id),
        ).fetchone()
    return dict(row) if row else None


# ══════════════════════════════════════════════════════════════════════════════
# Pinned charts
# ══════════════════════════════════════════════════════════════════════════════

def pin_chart(
    user_id: int,
    account_id: str,
    title: str,
    question: str,
    sql_query: str,
    chart_type: str,
    db_config_id: int,
    color_palette: str = "default",
) -> int:
    with get_db() as conn:
        # Get next position
        row = conn.execute(
            "SELECT COALESCE(MAX(position),0)+1 AS next FROM pinned_chart WHERE user_id=?",
            (user_id,)
        ).fetchone()
        pos = row["next"] if row else 1
        grid_w = 6
        grid_h = 5
        grid_x = ((pos - 1) % 2) * grid_w
        grid_y = ((pos - 1) // 2) * grid_h
        cur = conn.execute("""
            INSERT INTO pinned_chart
                (user_id, account_id, title, question, sql_query, chart_type,
                 db_config_id, position, color_palette, grid_x, grid_y, grid_w, grid_h)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (user_id, account_id, title, question, sql_query, chart_type,
               db_config_id, pos, color_palette, grid_x, grid_y, grid_w, grid_h))
        cid = cur.lastrowid
    log.info("Pinned chart %d for user %d", cid, user_id)
    return cid


def list_pinned_charts(user_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM pinned_chart WHERE user_id=? ORDER BY position, id",
            (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def delete_pinned_chart(chart_id: int, user_id: int) -> None:
    with get_db() as conn:
        conn.execute(
            "DELETE FROM pinned_chart WHERE id=? AND user_id=?",
            (chart_id, user_id)
        )


def update_pinned_chart_title(chart_id: int, user_id: int, title: str) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE pinned_chart SET title=? WHERE id=? AND user_id=?",
            (title, chart_id, user_id)
        )


def update_pinned_chart(
    chart_id: int,
    user_id: int,
    title: str | None = None,
    chart_type: str | None = None,
    color_palette: str | None = None,
) -> None:
    """Update title, chart_type, and/or color_palette for a pinned chart."""
    fields: list[str] = []
    values: list = []
    if title is not None:
        fields.append("title=?"); values.append(title.strip()[:120])
    if chart_type is not None:
        fields.append("chart_type=?"); values.append(chart_type.strip())
    if color_palette is not None:
        fields.append("color_palette=?"); values.append(color_palette.strip())
    if not fields:
        return
    values.extend([chart_id, user_id])
    with get_db() as conn:
        conn.execute(
            f"UPDATE pinned_chart SET {', '.join(fields)} WHERE id=? AND user_id=?",
            values,
        )


def update_pinned_chart_layouts(user_id: int, layouts: list[dict]) -> None:
    """Persist dashboard card positions/sizes for a portal user."""
    if not layouts:
        return

    rows: list[tuple[int, int, int, int, int, int]] = []
    for idx, item in enumerate(layouts):
        try:
            chart_id = int(item.get("chart_id") or item.get("id") or 0)
            if not chart_id:
                continue
            x = max(0, min(11, int(item.get("x") or 0)))
            y = max(0, int(item.get("y") or 0))
            w = max(2, min(12, int(item.get("w") or 6)))
            h = max(3, min(12, int(item.get("h") or 5)))
            position = max(1, int(item.get("position") or idx + 1))
        except Exception:
            continue
        rows.append((x, y, w, h, position, chart_id))

    if not rows:
        return

    with get_db() as conn:
        for x, y, w, h, position, chart_id in rows:
            conn.execute(
                """
                UPDATE pinned_chart
                   SET grid_x=?, grid_y=?, grid_w=?, grid_h=?, position=?
                 WHERE id=? AND user_id=?
                """,
                (x, y, w, h, position, chart_id, user_id),
            )


def update_chart_refreshed(chart_id: int) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE pinned_chart SET last_refreshed=datetime('now') WHERE id=?",
            (chart_id,)
        )
