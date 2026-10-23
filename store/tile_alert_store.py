"""
store/tile_alert_store.py

Alerts on a dashboard's number tiles (core/tile_alerts.py checks them).

    tile_alert             one per alert: the tile, who made it, the condition (above or below a value; a rise, a
                           fall or a move of more than N% on the period before), how often it is checked, and
                           whether it also goes by email
    tile_alert_recipient   who else it tells: people and groups (the maker is always told)
    tile_alert_state       for each person it tells: whether the line is crossed for them now, and the last value
                           they were shown -- each person's own, as their own access draws the tile
    tile_alert_event       each time it fired or came back, for whom, at what value

A check is claimed by moving the alert's next check forward from the value it was read at, so two schedulers never
check one alert at once.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from store.db import get_db

CONDITIONS = ("above", "below", "rise", "fall", "move")
OK, FIRING = "ok", "firing"
FIRED, CLEARED = "fired", "cleared"
EVERY = (60, 1440)                  # checked every hour, or every day
MAX_SUBJECTS = 50


def _now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def create_alert(account_id: str, dashboard_id: int, chart_id: int, owner_id: int, *, condition: str,
                 threshold: float, check_minutes: int, by_email: bool, next_check_at: str, value_format: str = "",
                 recipients: list[tuple[str, int]] = ()) -> int:
    """``value_format``: how the tile writes its number (currency, percent...), to write the line the same way."""
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO tile_alert (account_id, dashboard_id, chart_id, owner_id, condition, threshold, "
            "check_minutes, by_email, next_check_at, value_format) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (str(account_id), int(dashboard_id), int(chart_id), int(owner_id), condition, float(threshold),
             int(check_minutes), 1 if by_email else 0, next_check_at, str(value_format or "")))
        alert_id = int(cur.lastrowid)
        for subject_type, subject_id in dict.fromkeys(recipients):
            conn.execute("INSERT INTO tile_alert_recipient (alert_id, subject_type, subject_id) VALUES (?,?,?)",
                         (alert_id, str(subject_type), int(subject_id)))
    return alert_id


def get_alert(alert_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM tile_alert WHERE id=?", (int(alert_id),)).fetchone()
    return dict(row) if row else None


def tile_title(alert_id: int) -> str:
    """The name of the tile an alert is on ("" when the alert or its tile is gone)."""
    with get_db() as conn:
        row = conn.execute("SELECT c.title FROM tile_alert a JOIN pinned_chart c ON c.id=a.chart_id WHERE a.id=?",
                           (int(alert_id),)).fetchone()
    return str(row["title"] or "") if row else ""


def alert_recipients(alert_id: int) -> list[dict]:
    """Who else an alert tells, as Share shows people: a name and a group, a group's name and size; never an email."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT r.subject_type, r.subject_id,
                      CASE WHEN r.subject_type='user' THEN u.name ELSE g.name END AS name
                 FROM tile_alert_recipient r JOIN tile_alert a ON a.id=r.alert_id
                 LEFT JOIN portal_user u ON r.subject_type='user' AND u.id=r.subject_id AND u.account_id=a.account_id
                                        AND u.is_active=1
                 LEFT JOIN user_group g ON r.subject_type='group' AND g.id=r.subject_id AND g.account_id=a.account_id
                WHERE r.alert_id=? ORDER BY r.subject_type DESC, name""", (int(alert_id),)).fetchall()
    return [{"type": r["subject_type"], "id": int(r["subject_id"]), "name": r["name"]} for r in rows if r["name"]]


def alert_people(alert: dict) -> list[dict]:
    """Everyone an alert tells now, once each: its maker, the people it names and the active members of its groups;
    active accounts of its workspace only."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT DISTINCT u.* FROM portal_user u
                WHERE u.account_id=? AND u.is_active=1 AND (
                      u.id=?
                   OR u.id IN (SELECT subject_id FROM tile_alert_recipient WHERE alert_id=? AND subject_type='user')
                   OR u.id IN (SELECT gm.user_id FROM user_group_member gm
                                WHERE gm.group_id IN (SELECT subject_id FROM tile_alert_recipient
                                                       WHERE alert_id=? AND subject_type='group')))
                ORDER BY u.id""",
            (str(alert["account_id"]), int(alert["owner_id"]), int(alert["id"]), int(alert["id"]))).fetchall()
    return [dict(r) for r in rows]


def alerts_for_tile(dashboard_id: int, chart_id: int, user_id: int, account_id: str) -> list[dict]:
    """The alerts on one tile that tell ``user_id``: the ones they made, then the ones others made for them."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT a.*, o.name AS owner_name, s.state, s.value AS last_value, s.changed_at
                 FROM tile_alert a JOIN portal_user o ON o.id=a.owner_id
                 LEFT JOIN tile_alert_state s ON s.alert_id=a.id AND s.user_id=?
                WHERE a.account_id=? AND a.dashboard_id=? AND a.chart_id=?
                  AND (a.owner_id=?
                       OR EXISTS (SELECT 1 FROM tile_alert_recipient r WHERE r.alert_id=a.id
                                     AND ((r.subject_type='user' AND r.subject_id=?)
                                       OR (r.subject_type='group' AND r.subject_id IN (SELECT group_id
                                                FROM user_group_member WHERE user_id=?)))))
                ORDER BY CASE WHEN a.owner_id=? THEN 0 ELSE 1 END, a.id DESC""",
            (int(user_id), str(account_id), int(dashboard_id), int(chart_id), int(user_id), int(user_id), int(user_id),
             int(user_id))).fetchall()
    return [dict(r) for r in rows]


def tile_marks(dashboard_id: int, user_id: int, account_id: str) -> dict[int, dict]:
    """For a dashboard's tiles, what its bell shows ``user_id``: {chart id: {"on": True, "firing": bool}}."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT a.chart_id, MAX(CASE WHEN s.state='firing' AND a.is_active=1 THEN 1 ELSE 0 END) AS firing
                 FROM tile_alert a
                 LEFT JOIN tile_alert_state s ON s.alert_id=a.id AND s.user_id=?
                WHERE a.account_id=? AND a.dashboard_id=?
                  AND (a.owner_id=?
                       OR EXISTS (SELECT 1 FROM tile_alert_recipient r WHERE r.alert_id=a.id
                                     AND ((r.subject_type='user' AND r.subject_id=?)
                                       OR (r.subject_type='group' AND r.subject_id IN (SELECT group_id
                                                FROM user_group_member WHERE user_id=?)))))
                GROUP BY a.chart_id""",
            (int(user_id), str(account_id), int(dashboard_id), int(user_id), int(user_id), int(user_id))).fetchall()
    return {int(r["chart_id"]): {"on": True, "firing": bool(r["firing"])} for r in rows}


def set_alert_active(alert_id: int, active: bool, *, reason: str = "", next_check_at: Optional[str] = None) -> None:
    with get_db() as conn:
        conn.execute("UPDATE tile_alert SET is_active=?, paused_reason=?, next_check_at=COALESCE(?, next_check_at) "
                     "WHERE id=?", (1 if active else 0, "" if active else str(reason)[:300], next_check_at,
                                    int(alert_id)))


def delete_alert(alert_id: int) -> None:
    with get_db() as conn:
        for table in ("tile_alert_recipient", "tile_alert_state", "tile_alert_event"):
            conn.execute(f"DELETE FROM {table} WHERE alert_id=?", (int(alert_id),))
        conn.execute("DELETE FROM tile_alert WHERE id=?", (int(alert_id),))


def due_alerts(now: str, limit: int = 20) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM tile_alert WHERE is_active=1 AND next_check_at<=? ORDER BY next_check_at, id "
                            "LIMIT ?", (now, int(limit))).fetchall()
    return [dict(r) for r in rows]


def claim_check(alert_id: int, seen_next: str, next_check_at: str) -> bool:
    """Take the check that was due at ``seen_next``: False when another scheduler took it first."""
    with get_db() as conn:
        cur = conn.execute("UPDATE tile_alert SET next_check_at=?, last_checked_at=? WHERE id=? AND next_check_at=?",
                           (next_check_at, _now(), int(alert_id), seen_next))
    return cur.rowcount == 1


def get_state(alert_id: int, user_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM tile_alert_state WHERE alert_id=? AND user_id=?",
                           (int(alert_id), int(user_id))).fetchone()
    return dict(row) if row else None


def put_state(alert_id: int, user_id: int, *, state: str, value: Optional[float], changed: bool) -> None:
    now = _now()
    with get_db() as conn:
        found = conn.execute("SELECT 1 FROM tile_alert_state WHERE alert_id=? AND user_id=?",
                             (int(alert_id), int(user_id))).fetchone()
        if found:
            conn.execute("UPDATE tile_alert_state SET state=?, value=?, checked_at=?"
                         + (", changed_at=?" if changed else "") + " WHERE alert_id=? AND user_id=?",
                         (state, value, now, *((now,) if changed else ()), int(alert_id), int(user_id)))
        else:
            conn.execute("INSERT INTO tile_alert_state (alert_id, user_id, state, value, checked_at, changed_at) "
                         "VALUES (?,?,?,?,?,?)", (int(alert_id), int(user_id), state, value, now, now))


def add_event(alert_id: int, user_id: int, kind: str, *, value: Optional[float], display: str) -> int:
    with get_db() as conn:
        cur = conn.execute("INSERT INTO tile_alert_event (alert_id, user_id, kind, value, display, at) "
                           "VALUES (?,?,?,?,?,?)", (int(alert_id), int(user_id), kind, value, str(display)[:200], _now()))
    return int(cur.lastrowid)


def recent_events(dashboard_id: int, chart_id: int, user_id: int, limit: int = 8) -> list[dict]:
    """What a tile's alerts told ``user_id`` lately, newest first."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT e.*, a.condition, a.threshold FROM tile_alert_event e JOIN tile_alert a ON a.id=e.alert_id
                WHERE a.dashboard_id=? AND a.chart_id=? AND e.user_id=? ORDER BY e.id DESC LIMIT ?""",
            (int(dashboard_id), int(chart_id), int(user_id), int(limit))).fetchall()
    return [dict(r) for r in rows]
