"""
store/schedule_store.py

Dashboards emailed on a schedule (core/dashboard_schedule.py sends them).

    dashboard_schedule            one per schedule: the dashboard, its owner, when (cadence, day, time, time zone),
                                  which tabs, the filters it is drawn under, the PDF, a subject and a note
    dashboard_schedule_recipient  who it goes to: people, and groups (their members when it is sent)
    dashboard_schedule_run        each time it went: when it was due, how many it was sent to, and why not

A run is claimed by inserting it (schedule, due time) before anything is sent: a second scheduler, or the same
one after a restart, finds the row and sends nothing twice. Each email of a run carries its own dedupe key in
the outbox, so a run taken up again after a crash sends only to those it had not reached.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from store.db import get_db

CADENCES = ("daily", "weekdays", "weekly", "monthly")
RUNNING, SENT, PARTIAL, FAILED, NOTHING = "running", "sent", "partial", "failed", "nothing"
MAX_SUBJECTS = 50

_FIELDS = ("cadence", "weekday", "month_day", "send_time", "timezone", "tabs_json", "view_json", "attach_pdf",
           "include_owner", "subject", "message")


def _now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def _row(row) -> Optional[dict]:
    if row is None:
        return None
    schedule = dict(row)
    for key, empty in (("tabs_json", []), ("view_json", {})):
        try:
            value = json.loads(schedule.get(key) or "")
        except (TypeError, ValueError):
            value = empty
        schedule[key[:-5]] = value if isinstance(value, type(empty)) else empty
    return schedule


def create_schedule(account_id: str, dashboard_id: int, owner_id: int, *, next_run_at: str,
                    recipients: list[tuple[str, int]], **fields) -> int:
    values = {key: fields[key] for key in _FIELDS if key in fields}
    columns = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    with get_db() as conn:
        cur = conn.execute(
            f"INSERT INTO dashboard_schedule (account_id, dashboard_id, owner_id, next_run_at"
            f"{', ' + columns if columns else ''}) VALUES (?,?,?,?{', ' + marks if marks else ''})",
            (str(account_id), int(dashboard_id), int(owner_id), next_run_at, *values.values()))
        schedule_id = int(cur.lastrowid)
        _put_recipients(conn, schedule_id, recipients)
    return schedule_id


def update_schedule(schedule_id: int, *, next_run_at: Optional[str] = None,
                    recipients: Optional[list[tuple[str, int]]] = None, **fields) -> None:
    values = {key: fields[key] for key in _FIELDS if key in fields}
    if next_run_at is not None:
        values["next_run_at"] = next_run_at
    with get_db() as conn:
        if values:
            sets = ", ".join(f"{key}=?" for key in values)
            conn.execute(f"UPDATE dashboard_schedule SET {sets}, updated_at=datetime('now') WHERE id=?",
                         (*values.values(), int(schedule_id)))
        if recipients is not None:
            conn.execute("DELETE FROM dashboard_schedule_recipient WHERE schedule_id=?", (int(schedule_id),))
            _put_recipients(conn, int(schedule_id), recipients)


def _put_recipients(conn, schedule_id: int, recipients: list[tuple[str, int]]) -> None:
    for subject_type, subject_id in dict.fromkeys(recipients):
        conn.execute("INSERT INTO dashboard_schedule_recipient (schedule_id, subject_type, subject_id) VALUES (?,?,?)",
                     (int(schedule_id), str(subject_type), int(subject_id)))


def get_schedule(schedule_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM dashboard_schedule WHERE id=?", (int(schedule_id),)).fetchone()
    return _row(row)


def list_schedules(account_id: str, dashboard_id: int, owner_id: int) -> list[dict]:
    """The owner's schedules of one dashboard, newest first, each with who it goes to."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM dashboard_schedule WHERE account_id=? AND dashboard_id=? AND owner_id=? ORDER BY id DESC",
            (str(account_id), int(dashboard_id), int(owner_id))).fetchall()
    schedules = [_row(r) for r in rows]
    for schedule in schedules:
        schedule["recipients"] = schedule_recipients(int(schedule["id"]))
    return schedules


def schedule_recipients(schedule_id: int) -> list[dict]:
    """Who a schedule names, as the Share dialog shows people: [{"type", "id", "name", "group", "members"}] -- a
    person by name and group, a group by name and size, never an email. Someone gone since is left out."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT r.subject_type, r.subject_id,
                      CASE WHEN r.subject_type='user' THEN u.name ELSE g.name END AS name,
                      ug.name AS group_name,
                      (SELECT COUNT(*) FROM portal_user m WHERE r.subject_type='group' AND m.group_id=r.subject_id
                                                            AND m.is_active=1) AS members
                 FROM dashboard_schedule_recipient r
                 JOIN dashboard_schedule s ON s.id=r.schedule_id
                 LEFT JOIN portal_user u ON r.subject_type='user' AND u.id=r.subject_id AND u.account_id=s.account_id
                                        AND u.is_active=1
                 LEFT JOIN user_group ug ON ug.id=u.group_id
                 LEFT JOIN user_group g ON r.subject_type='group' AND g.id=r.subject_id AND g.account_id=s.account_id
                WHERE r.schedule_id=?
                ORDER BY r.subject_type DESC, name""", (int(schedule_id),)).fetchall()
    return [{"type": r["subject_type"], "id": int(r["subject_id"]), "name": r["name"], "group": r["group_name"] or "",
             "members": int(r["members"] or 0)} for r in rows if r["name"]]


def schedule_people(schedule: dict) -> list[dict]:
    """Everyone a schedule reaches now, once each: the people it names, the active members of its groups, and its
    owner when they asked for a copy. Active accounts of its workspace only."""
    account_id = str(schedule["account_id"])
    with get_db() as conn:
        rows = conn.execute(
            """SELECT DISTINCT u.* FROM portal_user u
                WHERE u.account_id=? AND u.is_active=1 AND (
                      u.id IN (SELECT subject_id FROM dashboard_schedule_recipient
                                WHERE schedule_id=? AND subject_type='user')
                   OR u.group_id IN (SELECT subject_id FROM dashboard_schedule_recipient
                                      WHERE schedule_id=? AND subject_type='group')
                   OR (u.id=? AND ?=1))
                ORDER BY u.id""",
            (account_id, int(schedule["id"]), int(schedule["id"]), int(schedule["owner_id"]),
             1 if int(schedule.get("include_owner") or 0) else 0)).fetchall()
    return [dict(r) for r in rows]


def delete_schedule(schedule_id: int) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM dashboard_schedule_recipient WHERE schedule_id=?", (int(schedule_id),))
        conn.execute("DELETE FROM dashboard_schedule_run WHERE schedule_id=?", (int(schedule_id),))
        conn.execute("DELETE FROM dashboard_schedule WHERE id=?", (int(schedule_id),))


def set_schedule_active(schedule_id: int, active: bool, *, reason: str = "", next_run_at: Optional[str] = None) -> None:
    """Pause a schedule (``reason`` says why, for its owner), or resume it from ``next_run_at``."""
    with get_db() as conn:
        conn.execute(
            "UPDATE dashboard_schedule SET is_active=?, paused_reason=?, next_run_at=COALESCE(?, next_run_at), "
            "updated_at=datetime('now') WHERE id=?",
            (1 if active else 0, "" if active else str(reason)[:300], next_run_at, int(schedule_id)))


def due_schedules(now: str, limit: int = 5) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM dashboard_schedule WHERE is_active=1 AND next_run_at IS NOT NULL AND next_run_at<=? "
            "ORDER BY next_run_at, id LIMIT ?", (now, int(limit))).fetchall()
    return [_row(r) for r in rows]


def claim_run(schedule_id: int, due_at: str, *, kind: str = "scheduled") -> Optional[int]:
    """A run of a schedule for the time it was due; None when it was claimed already."""
    with get_db() as conn:
        if conn.execute("SELECT 1 FROM dashboard_schedule_run WHERE schedule_id=? AND due_at=?",
                        (int(schedule_id), due_at)).fetchone():
            return None
        # A savepoint: on PostgreSQL a refused insert would otherwise spoil the rest of the transaction.
        conn.execute("SAVEPOINT sp_claim_run")
        try:
            cur = conn.execute("INSERT INTO dashboard_schedule_run (schedule_id, due_at, kind, started_at) "
                               "VALUES (?,?,?,?)", (int(schedule_id), due_at, kind, _now()))
        except Exception:  # noqa: BLE001 - the unique key: another scheduler claimed it between the two
            conn.execute("ROLLBACK TO SAVEPOINT sp_claim_run")
            return None
        conn.execute("RELEASE SAVEPOINT sp_claim_run")
        return int(cur.lastrowid)


def advance_schedule(schedule_id: int, *, next_run_at: Optional[str], last_run_at: str) -> None:
    with get_db() as conn:
        conn.execute("UPDATE dashboard_schedule SET next_run_at=?, last_run_at=? WHERE id=?",
                     (next_run_at, last_run_at, int(schedule_id)))


def finish_run(run_id: int, *, status: str, recipients: int, queued: int, skipped: int, detail: str = "") -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE dashboard_schedule_run SET status=?, recipients=?, queued=?, skipped=?, detail=?, finished_at=? "
            "WHERE id=?", (status, int(recipients), int(queued), int(skipped), str(detail)[:1000], _now(), int(run_id)))
        row = conn.execute("SELECT schedule_id, kind FROM dashboard_schedule_run WHERE id=?", (int(run_id),)).fetchone()
        if row and row["kind"] != "test":
            conn.execute("UPDATE dashboard_schedule SET last_status=?, last_detail=? WHERE id=?",
                         (status, str(detail)[:300], int(row["schedule_id"])))


def get_run(run_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM dashboard_schedule_run WHERE id=?", (int(run_id),)).fetchone()
    return dict(row) if row else None


def stale_runs(before: str) -> list[dict]:
    """Runs still marked running that started before ``before``: the server stopped in the middle of one."""
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM dashboard_schedule_run WHERE status=? AND started_at<? ORDER BY id",
                            (RUNNING, before)).fetchall()
    return [dict(r) for r in rows]


def restart_run(run_id: int) -> bool:
    """Take up a stopped run again (once per sweep): its start moves to now."""
    with get_db() as conn:
        cur = conn.execute("UPDATE dashboard_schedule_run SET started_at=? WHERE id=? AND status=?",
                           (_now(), int(run_id), RUNNING))
    return bool(cur.rowcount)


def list_runs(schedule_id: int, limit: int = 10) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM dashboard_schedule_run WHERE schedule_id=? ORDER BY id DESC LIMIT ?",
                            (int(schedule_id), int(limit))).fetchall()
    return [dict(r) for r in rows]
