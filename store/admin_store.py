"""
store/admin_store.py

Named admin accounts for the admin console, the workspaces each one may open,
and the log of what admins did.

The console had one shared password and no names: anyone who knew it could do
everything, and nothing recorded who did what. Each admin now signs in as
themselves with one of three roles:

    super_admin      every page: workspaces, databases, Learn, compliance,
                     system settings, other admins
    workspace_admin  the people, groups and table access of the workspaces it
                     is given, plus everything a data steward does there
    data_steward     metrics, dates, relationships, the knowledge base and the
                     readers' requests, in the workspaces it is given

A super admin is not tied to workspaces; the other two see only theirs
(admin_workspace). The shared password stays as the emergency sign-in
(admin/credentials.py): it signs in as a super admin with no name, every use is
logged, and a super admin can turn it off.

An invitation is a one-time link: its token is stored hashed, expires, and is
spent when the invited admin chooses a password.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from store import passwords
from store.db import get_db

SUPER_ADMIN = "super_admin"
WORKSPACE_ADMIN = "workspace_admin"
DATA_STEWARD = "data_steward"
ADMIN_ROLES = (SUPER_ADMIN, WORKSPACE_ADMIN, DATA_STEWARD)
ADMIN_ROLE_LABELS = {SUPER_ADMIN: "Super admin", WORKSPACE_ADMIN: "Workspace admin", DATA_STEWARD: "Data steward"}
# How far each role reaches: a page open to a role is open to every role above it.
ADMIN_ROLE_RANK = {DATA_STEWARD: 1, WORKSPACE_ADMIN: 2, SUPER_ADMIN: 3}

INVITE_HOURS = 72
MIN_PASSWORD = 12


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def normalise_email(email: str) -> str:
    return str(email or "").strip().lower()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _row(row) -> Optional[dict]:
    if row is None:
        return None
    admin = dict(row)
    admin["role_label"] = ADMIN_ROLE_LABELS.get(admin.get("role"), admin.get("role"))
    return admin


def any_admin_accounts() -> bool:
    """Whether anyone has a named admin account yet (active or not)."""
    with get_db() as conn:
        return conn.execute("SELECT 1 FROM admin_account LIMIT 1").fetchone() is not None


def count_active_super_admins(*, with_password: bool = True) -> int:
    with get_db() as conn:
        sql = "SELECT COUNT(*) AS n FROM admin_account WHERE role=? AND is_active=1"
        if with_password:
            sql += " AND password_hash<>''"
        return int(conn.execute(sql, (SUPER_ADMIN,)).fetchone()["n"])


def create_admin(email: str, name: str, role: str, *, password: Optional[str] = None,
                 workspaces: Optional[list[str]] = None, created_by: str = "") -> int:
    """A new admin account. With ``password`` it can sign in at once; without, it waits for its invitation."""
    email = normalise_email(email)
    if role not in ADMIN_ROLES:
        raise ValueError(f"Unknown admin role: {role!r}")
    if not email or "@" not in email:
        raise ValueError("An admin needs an email address.")
    pw_hash = passwords.hash_password(password) if password else ""
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO admin_account (email, name, role, password_hash, created_by) VALUES (?,?,?,?,?)",
            (email, str(name or "").strip()[:120], role, pw_hash, str(created_by or "")[:120]),
        )
        admin_id = int(cur.lastrowid)
        for account_id in sorted(set(workspaces or [])) if role != SUPER_ADMIN else []:
            conn.execute("INSERT INTO admin_workspace (admin_id, account_id) VALUES (?,?)", (admin_id, account_id))
    return admin_id


def get_admin(admin_id: int) -> Optional[dict]:
    with get_db() as conn:
        return _row(conn.execute("SELECT * FROM admin_account WHERE id=?", (int(admin_id),)).fetchone())


def get_admin_by_email(email: str) -> Optional[dict]:
    with get_db() as conn:
        return _row(conn.execute("SELECT * FROM admin_account WHERE email=?", (normalise_email(email),)).fetchone())


def list_admins(*, account_id: Optional[str] = None) -> list[dict]:
    """Every admin, or the admins who can open ``account_id`` (its own admins and every super admin)."""
    with get_db() as conn:
        if account_id is None:
            rows = conn.execute("SELECT * FROM admin_account ORDER BY name, email").fetchall()
        else:
            rows = conn.execute(
                "SELECT a.* FROM admin_account a WHERE a.role=? OR EXISTS ("
                " SELECT 1 FROM admin_workspace w WHERE w.admin_id=a.id AND w.account_id=?)"
                " ORDER BY a.name, a.email", (SUPER_ADMIN, account_id)).fetchall()
        admins = [_row(r) for r in rows]
        spaces: dict[int, list[str]] = {}
        for r in conn.execute("SELECT admin_id, account_id FROM admin_workspace ORDER BY account_id").fetchall():
            spaces.setdefault(int(r["admin_id"]), []).append(r["account_id"])
    for admin in admins:
        admin["workspaces"] = spaces.get(int(admin["id"]), [])
    return admins


def admin_workspaces(admin_id: int) -> list[str]:
    with get_db() as conn:
        return [r["account_id"] for r in conn.execute(
            "SELECT account_id FROM admin_workspace WHERE admin_id=? ORDER BY account_id", (int(admin_id),)).fetchall()]


def set_admin_workspaces(admin_id: int, workspaces: list[str]) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM admin_workspace WHERE admin_id=?", (int(admin_id),))
        for account_id in sorted(set(workspaces)):
            conn.execute("INSERT INTO admin_workspace (admin_id, account_id) VALUES (?,?)", (int(admin_id), account_id))


def add_admin_workspace(admin_id: int, account_id: str) -> None:
    with get_db() as conn:
        if conn.execute("SELECT 1 FROM admin_workspace WHERE admin_id=? AND account_id=?",
                        (int(admin_id), account_id)).fetchone() is None:
            conn.execute("INSERT INTO admin_workspace (admin_id, account_id) VALUES (?,?)", (int(admin_id), account_id))


def remove_admin_workspace(admin_id: int, account_id: str) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM admin_workspace WHERE admin_id=? AND account_id=?", (int(admin_id), account_id))


def update_admin(admin_id: int, *, name: Optional[str] = None, role: Optional[str] = None,
                 is_active: Optional[bool] = None) -> None:
    """Change an admin. A new role or a deactivation ends their sessions; a super admin needs no workspaces."""
    fields, params = [], []
    if name is not None:
        fields.append("name=?"); params.append(str(name).strip()[:120])
    if role is not None:
        if role not in ADMIN_ROLES:
            raise ValueError(f"Unknown admin role: {role!r}")
        fields.append("role=?"); params.append(role)
        fields.append("session_version=session_version+1")
    if is_active is not None:
        fields.append("is_active=?"); params.append(1 if is_active else 0)
        if not is_active:
            fields.append("session_version=session_version+1")
    if not fields:
        return
    fields.append("updated_at=datetime('now')")
    params.append(int(admin_id))
    with get_db() as conn:
        conn.execute(f"UPDATE admin_account SET {','.join(fields)} WHERE id=?", params)


def set_admin_password(admin_id: int, password: str) -> None:
    """A new password; every session the admin had ends, and an open invitation is spent."""
    with get_db() as conn:
        conn.execute(
            "UPDATE admin_account SET password_hash=?, invite_token_hash='', invite_expires_at=NULL, "
            "session_version=session_version+1, updated_at=datetime('now') WHERE id=?",
            (passwords.hash_password(password), int(admin_id)))


def verify_admin_password(admin: dict, password: str) -> bool:
    stored = str(admin.get("password_hash") or "")
    return bool(stored) and passwords.verify_password(stored, password)


def upgrade_admin_hash(admin: dict, password: str) -> None:
    stored = str(admin.get("password_hash") or "")
    if stored and passwords.needs_rehash(stored) and passwords.verify_password(stored, password):
        with get_db() as conn:
            conn.execute("UPDATE admin_account SET password_hash=? WHERE id=?",
                         (passwords.hash_password(password), int(admin["id"])))


def mark_admin_signed_in(admin_id: int) -> None:
    with get_db() as conn:
        conn.execute("UPDATE admin_account SET last_sign_in_at=? WHERE id=?", (_stamp(_now()), int(admin_id)))


def issue_admin_invite(admin_id: int, *, hours: int = INVITE_HOURS) -> str:
    """A one-time link token for the admin to choose a password; the old link stops working."""
    token = secrets.token_urlsafe(32)
    with get_db() as conn:
        conn.execute("UPDATE admin_account SET invite_token_hash=?, invite_expires_at=? WHERE id=?",
                     (_token_hash(token), _stamp(_now() + timedelta(hours=hours)), int(admin_id)))
    return token


def admin_for_invite(token: str) -> Optional[dict]:
    """The active admin a live invitation is for, or None (unknown, spent or expired)."""
    if not token:
        return None
    with get_db() as conn:
        row = conn.execute("SELECT * FROM admin_account WHERE invite_token_hash=? AND is_active=1",
                           (_token_hash(token),)).fetchone()
    admin = _row(row)
    if admin is None or not admin.get("invite_expires_at"):
        return None
    try:
        expires = datetime.strptime(str(admin["invite_expires_at"])[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return admin if expires > _now() else None


def delete_admin(admin_id: int) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM admin_workspace WHERE admin_id=?", (int(admin_id),))
        conn.execute("DELETE FROM admin_account WHERE id=?", (int(admin_id),))


# ── What admins did ───────────────────────────────────────────────────────────

def record_admin_event(*, admin_id: Optional[int], admin_email: str, action: str, method: str = "",
                       path: str = "", account_id: str = "", status: int = 0, detail: str = "",
                       ip: str = "") -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT INTO admin_event (admin_id, admin_email, action, method, path, account_id, status, detail, ip) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (admin_id, str(admin_email or "")[:200], str(action)[:40], str(method or "")[:10], str(path or "")[:400],
             str(account_id or "")[:120], int(status or 0), str(detail or "")[:500], str(ip or "")[:64]))


def list_admin_events(*, account_id: Optional[str] = None, admin_id: Optional[int] = None,
                      limit: int = 50) -> list[dict]:
    clauses, params = [], []
    if account_id is not None:
        clauses.append("account_id=?"); params.append(account_id)
    if admin_id is not None:
        clauses.append("admin_id=?"); params.append(int(admin_id))
    where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
    with get_db() as conn:
        rows = conn.execute(f"SELECT * FROM admin_event {where}ORDER BY id DESC LIMIT ?",
                            (*params, int(limit))).fetchall()
    return [dict(r) for r in rows]
