"""
store/profile_store.py

A reader's own profile, their photo, and where they signed in from (portal Settings › Profile and Security).

    portal_profile   job title, time zone (the one schedules start from), start page (where signing in lands), and
                     a photo revision that changes with every new photo, so a browser never shows the old one
    portal_sign_in   each sign-in: when, from which address and browser; the last SIGN_INS are kept
    portal_standing  the reader's standing conditions for every conversation in a workspace ("always use the last
                     2 months", core2/standing.py), listed on the Profile page

The photo is a 256-pixel square PNG beside the database (<db dir>/profile_photos/<user id>.png), re-drawn from what
was uploaded, so nothing of the original file (its metadata, a script in it) is kept.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from store.db import get_db

SIGN_INS = 20
START_PAGES = ("", "chat", "dashboards")          # "" is the workspace's own; "dashboard:<id>" one dashboard


def photos_root() -> Path:
    db_path = Path(os.getenv("QUERYBOT_DB_PATH") or os.getenv("DB_PATH") or "data/querybot.db")
    return db_path.parent / "profile_photos"


def get_profile(user_id: int) -> dict:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM portal_profile WHERE user_id=?", (int(user_id),)).fetchone()
    found = dict(row) if row else {}
    return {"job_title": found.get("job_title") or "", "timezone": found.get("timezone") or "",
            "start_page": found.get("start_page") or "", "photo_rev": int(found.get("photo_rev") or 0)}


def save_profile(user_id: int, **fields) -> None:
    keep = {k: str(v or "")[:200] for k, v in fields.items() if k in ("job_title", "timezone", "start_page")}
    if not keep:
        return
    with get_db() as conn:
        if not conn.execute("SELECT 1 FROM portal_profile WHERE user_id=?", (int(user_id),)).fetchone():
            conn.execute("INSERT INTO portal_profile (user_id) VALUES (?)", (int(user_id),))
        sets = ", ".join(f"{k}=?" for k in keep)
        conn.execute(f"UPDATE portal_profile SET {sets}, updated_at=datetime('now') WHERE user_id=?",
                     (*keep.values(), int(user_id)))


def photo_path(user_id: int) -> Path:
    return photos_root() / f"{int(user_id)}.png"


def save_photo(user_id: int, png: bytes) -> int:
    """Keep a reader's photo (a PNG already re-drawn); its new revision."""
    root = photos_root()
    root.mkdir(parents=True, exist_ok=True)
    photo_path(user_id).write_bytes(png)
    with get_db() as conn:
        if not conn.execute("SELECT 1 FROM portal_profile WHERE user_id=?", (int(user_id),)).fetchone():
            conn.execute("INSERT INTO portal_profile (user_id) VALUES (?)", (int(user_id),))
        conn.execute("UPDATE portal_profile SET photo_rev=photo_rev+1, updated_at=datetime('now') WHERE user_id=?",
                     (int(user_id),))
    return get_profile(user_id)["photo_rev"]


def delete_photo(user_id: int) -> None:
    try:
        photo_path(user_id).unlink()
    except FileNotFoundError:
        pass
    with get_db() as conn:
        conn.execute("UPDATE portal_profile SET photo_rev=0, updated_at=datetime('now') WHERE user_id=?",
                     (int(user_id),))


def record_sign_in(user_id: int, account_id: str, *, ip: str = "", agent: str = "") -> None:
    with get_db() as conn:
        conn.execute("INSERT INTO portal_sign_in (user_id, account_id, ip, agent) VALUES (?,?,?,?)",
                     (int(user_id), str(account_id), str(ip or "")[:64], str(agent or "")[:300]))
        conn.execute("DELETE FROM portal_sign_in WHERE user_id=? AND id NOT IN (SELECT id FROM portal_sign_in "
                     "WHERE user_id=? ORDER BY id DESC LIMIT ?)", (int(user_id), int(user_id), SIGN_INS))


def recent_sign_ins(user_id: int, limit: int = 10) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM portal_sign_in WHERE user_id=? ORDER BY id DESC LIMIT ?",
                            (int(user_id), int(limit))).fetchall()
    return [dict(r) for r in rows]


STANDING_MOST = 12


def standing_conditions(user_id: int, account_id: str) -> list[dict]:
    """The reader's standing conditions in a workspace, oldest first (core2/standing.py Standing.data)."""
    import json

    with get_db() as conn:
        rows = conn.execute("SELECT data FROM portal_standing WHERE user_id=? AND account_id=? ORDER BY position, id",
                            (int(user_id), str(account_id))).fetchall()
    out = []
    for row in rows:
        try:
            out.append(json.loads(row["data"]))
        except (TypeError, ValueError):
            continue
    return out


def keep_standing(user_id: int, account_id: str, data: dict) -> None:
    """Keep one standing condition; a new period replaces the period kept, a field's condition the one on it."""
    import json

    kept = standing_conditions(user_id, account_id)
    fields = {f.get("field") for f in data.get("filters") or []}
    replaced = [k["id"] for k in kept if k.get("id") == data["id"]
                or data.get("window") and k.get("window")
                or fields and {f.get("field") for f in k.get("filters") or []} == fields]
    with get_db() as conn:
        for old in replaced:
            conn.execute("DELETE FROM portal_standing WHERE user_id=? AND account_id=? AND id=?",
                         (int(user_id), str(account_id), old))
        if len(kept) - len(replaced) >= STANDING_MOST:
            return
        last = conn.execute("SELECT MAX(position) AS last FROM portal_standing WHERE user_id=? AND account_id=?",
                            (int(user_id), str(account_id))).fetchone()
        position = int((last["last"] if last is not None else 0) or 0) + 1
        conn.execute("INSERT INTO portal_standing (user_id, account_id, id, data, position) VALUES (?, ?, ?, ?, ?)",
                     (int(user_id), str(account_id), str(data["id"]), json.dumps(data), position))


def drop_standing(user_id: int, account_id: str, standing_id: str) -> bool:
    with get_db() as conn:
        done = conn.execute("DELETE FROM portal_standing WHERE user_id=? AND account_id=? AND id=?",
                            (int(user_id), str(account_id), str(standing_id)))
    return bool(done.rowcount)


def end_other_sessions(user_id: int) -> None:
    """Every session of this reader ends (the one in hand is issued again by the route)."""
    with get_db() as conn:
        conn.execute("UPDATE portal_user SET session_version=COALESCE(session_version, 1)+1, updated_at=datetime('now') "
                     "WHERE id=?", (int(user_id),))


def forget_people(conn, user_ids: list[int]) -> None:
    """A deleted person's profile, sign-ins and photo go with them (store/user_store.py delete_user, and a deleted
    workspace's people)."""
    for user_id in user_ids:
        conn.execute("DELETE FROM portal_profile WHERE user_id=?", (int(user_id),))
        conn.execute("DELETE FROM portal_sign_in WHERE user_id=?", (int(user_id),))
        conn.execute("DELETE FROM portal_standing WHERE user_id=?", (int(user_id),))
        try:
            photo_path(int(user_id)).unlink()
        except FileNotFoundError:
            pass
