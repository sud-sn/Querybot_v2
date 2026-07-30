"""
store/report_store.py

CRUD for named, multi-metric reports (core/report_engine.py's data layer).
A report is an admin-defined group of metric_registry rows a user can ask
about live in chat or subscribe to on a schedule. Tables (report,
report_metric, report_subscription) are created centrally by
store/db.py's init_db() — no local schema management needed here, matching
the style of simpler CRUD modules like date_context_store.py's siblings
that don't own their own table.
"""

from __future__ import annotations

from store.db import get_db


# ══════════════════════════════════════════════════════════════════════════════
# Reports
# ══════════════════════════════════════════════════════════════════════════════

def create_report(
    account_id: str,
    name: str,
    description: str = "",
    *,
    is_default: bool = False,
    created_by_user_id: int | None = None,
) -> dict:
    """Create a new report. Raises sqlite3.IntegrityError if the name is
    already taken for this account (UNIQUE(account_id, name)).

    created_by_user_id is None for admin-created reports (unchanged default)
    or a portal_user id for self-service reports -- attribution only, shown
    to admins; it does not restrict who can ask for/subscribe to the report."""
    name = (name or "").strip()
    with get_db() as conn:
        if is_default:
            conn.execute(
                "UPDATE report SET is_default=0 WHERE account_id=?", (account_id,)
            )
        cur = conn.execute(
            """INSERT INTO report (account_id, name, description, is_default, created_by_user_id)
               VALUES (?, ?, ?, ?, ?)""",
            (account_id, name, description or "", 1 if is_default else 0, created_by_user_id),
        )
        report_id = int(cur.lastrowid)
        row = conn.execute("SELECT * FROM report WHERE id=?", (report_id,)).fetchone()
    return dict(row)


def list_reports(account_id: str, active_only: bool = True) -> list[dict]:
    with get_db() as conn:
        if active_only:
            rows = conn.execute(
                "SELECT * FROM report WHERE account_id=? AND is_active=1 ORDER BY name",
                (account_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM report WHERE account_id=? ORDER BY is_active DESC, name",
                (account_id,),
            ).fetchall()
    return [dict(r) for r in rows]


def get_report(report_id: int, account_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM report WHERE id=? AND account_id=?", (report_id, account_id)
        ).fetchone()
    return dict(row) if row else None


def get_report_by_name(account_id: str, name: str) -> dict | None:
    """
    Resolve a user-typed report name to a report row. Case-insensitive exact
    match first (the common case — most accounts name reports plainly);
    falls back to a substring match so "daily" matches "Daily Ops Report".
    Returns None on no match OR on an ambiguous substring match (caller
    should list report names and ask, not guess).
    """
    name = (name or "").strip().lower()
    if not name:
        return None
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM report WHERE account_id=? AND is_active=1 AND lower(name)=?",
            (account_id, name),
        ).fetchone()
        if row:
            return dict(row)
        rows = conn.execute(
            "SELECT * FROM report WHERE account_id=? AND is_active=1 AND lower(name) LIKE ?",
            (account_id, f"%{name}%"),
        ).fetchall()
    if len(rows) == 1:
        return dict(rows[0])
    return None


def update_report(report_id: int, account_id: str, updates: dict) -> None:
    allowed = {"name", "description", "is_default", "is_active"}
    fields = {k: v for k, v in (updates or {}).items() if k in allowed}
    if not fields:
        return
    with get_db() as conn:
        if fields.get("is_default"):
            conn.execute(
                "UPDATE report SET is_default=0 WHERE account_id=?", (account_id,)
            )
        set_clause = ", ".join(f"{k}=?" for k in fields)
        conn.execute(
            f"UPDATE report SET {set_clause}, updated_at=datetime('now') "
            f"WHERE id=? AND account_id=?",
            (*fields.values(), report_id, account_id),
        )


def delete_report(report_id: int, account_id: str) -> bool:
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM report WHERE id=? AND account_id=?", (report_id, account_id)
        )
        return bool(cur.rowcount)


def update_own_report(report_id: int, account_id: str, user_id: int, updates: dict) -> bool:
    """Self-service edit, scoped to reports the caller created. Only
    name/description are editable this way -- is_default/is_active are
    account-wide policy knobs reserved for admin's own update_report."""
    allowed = {"name", "description"}
    fields = {k: v for k, v in (updates or {}).items() if k in allowed}
    if not fields:
        return False
    with get_db() as conn:
        set_clause = ", ".join(f"{k}=?" for k in fields)
        cur = conn.execute(
            f"UPDATE report SET {set_clause}, updated_at=datetime('now') "
            f"WHERE id=? AND account_id=? AND created_by_user_id=?",
            (*fields.values(), report_id, account_id, user_id),
        )
        return bool(cur.rowcount)


def delete_own_report(report_id: int, account_id: str, user_id: int) -> bool:
    """user_id-scoped so a user can only delete a report they created."""
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM report WHERE id=? AND account_id=? AND created_by_user_id=?",
            (report_id, account_id, user_id),
        )
        return bool(cur.rowcount)


# ══════════════════════════════════════════════════════════════════════════════
# Report ↔ metric membership
# ══════════════════════════════════════════════════════════════════════════════

def add_metric_to_report(report_id: int, metric_id: int, sort_order: int = 0) -> None:
    with get_db() as conn:
        conn.execute(
            """INSERT INTO report_metric (report_id, metric_id, sort_order)
               VALUES (?, ?, ?)
               ON CONFLICT(report_id, metric_id) DO UPDATE SET sort_order=excluded.sort_order""",
            (report_id, metric_id, sort_order),
        )


def remove_metric_from_report(report_id: int, metric_id: int) -> None:
    with get_db() as conn:
        conn.execute(
            "DELETE FROM report_metric WHERE report_id=? AND metric_id=?",
            (report_id, metric_id),
        )


def list_report_metrics(report_id: int) -> list[dict]:
    """Metrics assigned to a report, joined to metric_registry for the
    fields report_engine needs to run each one (name, base_table,
    sql_template, formula_type)."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT m.*, rm.sort_order
                 FROM report_metric rm
                 JOIN metric_registry m ON m.id = rm.metric_id
                WHERE rm.report_id=?
                ORDER BY rm.sort_order, m.name""",
            (report_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# Subscriptions
# ══════════════════════════════════════════════════════════════════════════════

def create_subscription(
    account_id: str,
    user_id: int,
    report_id: int,
    *,
    cadence: str = "daily",
    day_of_week: int = 0,
    hour: int = 8,
) -> dict:
    cadence = cadence if cadence in ("daily", "weekly") else "daily"
    hour = max(0, min(23, int(hour)))
    day_of_week = max(0, min(6, int(day_of_week)))
    with get_db() as conn:
        conn.execute(
            """INSERT INTO report_subscription
                   (account_id, user_id, report_id, cadence, day_of_week, hour, status)
               VALUES (?, ?, ?, ?, ?, ?, 'active')
               ON CONFLICT(user_id, report_id) DO UPDATE SET
                   cadence=excluded.cadence, day_of_week=excluded.day_of_week,
                   hour=excluded.hour, status='active'""",
            (account_id, user_id, report_id, cadence, day_of_week, hour),
        )
        row = conn.execute(
            "SELECT * FROM report_subscription WHERE user_id=? AND report_id=?",
            (user_id, report_id),
        ).fetchone()
    return dict(row)


def list_subscriptions(account_id: str | None = None, user_id: int | None = None) -> list[dict]:
    where = ["1=1"]
    params: list = []
    if account_id:
        where.append("account_id=?")
        params.append(account_id)
    if user_id is not None:
        where.append("user_id=?")
        params.append(user_id)
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT * FROM report_subscription WHERE {' AND '.join(where)} ORDER BY id",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def update_subscription(subscription_id: int, updates: dict) -> None:
    allowed = {"cadence", "day_of_week", "hour", "status", "last_sent"}
    fields = {k: v for k, v in (updates or {}).items() if k in allowed}
    if not fields:
        return
    with get_db() as conn:
        set_clause = ", ".join(f"{k}=?" for k in fields)
        conn.execute(
            f"UPDATE report_subscription SET {set_clause} WHERE id=?",
            (*fields.values(), subscription_id),
        )


def delete_subscription(subscription_id: int, user_id: int) -> bool:
    """user_id-scoped so a user can only unsubscribe their own subscription."""
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM report_subscription WHERE id=? AND user_id=?",
            (subscription_id, user_id),
        )
        return bool(cur.rowcount)
