"""
store/email_store.py

The email the product sends: its outbox (every message, queued, sent, failed or skipped, and why), the files
attached to one, and the addresses that asked to stop a kind of email.

The outbox is the delivery log too: a row is never deleted when it is sent. A message is claimed by one sender
before it goes (``claim_email``), so two schedulers never send it twice, and a ``dedupe_key`` keeps a message
that is queued twice (a scheduled report retried after a restart) to one row.

Attachments are files beside the database (<db dir>/email_attachments/<outbox id>/), not rows: a PDF of a
dashboard is a megabyte or two, and SQLite and PostgreSQL both store a path more cheaply than a blob.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from store.db import get_db

QUEUED, SENDING, SENT, FAILED, SKIPPED = "queued", "sending", "sent", "failed", "skipped"
STATUSES = (QUEUED, SENDING, SENT, FAILED, SKIPPED)


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def attachments_root() -> Path:
    db_path = Path(os.getenv("QUERYBOT_DB_PATH") or os.getenv("DB_PATH") or "data/querybot.db")
    return db_path.parent / "email_attachments"


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", str(name or "attachment")).strip(" .") or "attachment"
    return cleaned[:120]


def queue_email(*, kind: str, to_address: str, subject: str, body_text: str, body_html: str = "",
                account_id: str = "", user_id: Optional[int] = None, attachments: tuple = (),
                dedupe_key: Optional[str] = None, status: str = QUEUED, reason: str = "",
                ref: str = "", unsubscribe_url: str = "") -> Optional[int]:
    """A message in the outbox; None when ``dedupe_key`` is queued already.

    ``attachments``: (filename, content_type, bytes), or (filename, content_type, bytes, content_id) for an image
    the HTML shows in its body (``<img src="cid:content_id">``). ``status`` SKIPPED records a message that was not
    sent, and ``reason`` says why, so the delivery log shows it.
    """
    with get_db() as conn:
        if dedupe_key and conn.execute("SELECT 1 FROM email_outbox WHERE dedupe_key=?", (dedupe_key,)).fetchone():
            return None
        cur = conn.execute(
            "INSERT INTO email_outbox (account_id, kind, ref, to_address, user_id, subject, body_text, body_html, "
            "status, last_error, next_attempt_at, dedupe_key, attachments_json, unsubscribe_url) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(account_id or ""), str(kind)[:40], str(ref or "")[:120], str(to_address).strip()[:320],
             user_id, str(subject)[:300], body_text, body_html or "", status, str(reason or "")[:500],
             _stamp(_now()), dedupe_key, "[]", str(unsubscribe_url or "")[:1000]))
        email_id = int(cur.lastrowid)
    if attachments:
        folder = attachments_root() / str(email_id)
        folder.mkdir(parents=True, exist_ok=True)
        kept = []
        for filename, content_type, data, *inline in attachments:
            name = _safe_name(filename)
            (folder / name).write_bytes(data)
            item = {"name": name, "content_type": str(content_type or "application/octet-stream"), "size": len(data)}
            if inline and inline[0]:
                item["cid"] = re.sub(r"[^A-Za-z0-9._@-]+", "-", str(inline[0]))[:120]
            kept.append(item)
        with get_db() as conn:
            conn.execute("UPDATE email_outbox SET attachments_json=? WHERE id=?", (json.dumps(kept), email_id))
    return email_id


def get_email(email_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM email_outbox WHERE id=?", (int(email_id),)).fetchone()
    return _row(row)


def _row(row) -> Optional[dict]:
    if row is None:
        return None
    email = dict(row)
    try:
        email["attachments"] = json.loads(email.get("attachments_json") or "[]")
    except ValueError:
        email["attachments"] = []
    return email


def email_attachments(email: dict) -> list[tuple]:
    """The files attached to a message, read back: (filename, content_type, bytes), with the content id after them
    for an image shown in the body. A missing file is left out."""
    folder = attachments_root() / str(email["id"])
    found = []
    for item in email.get("attachments") or []:
        path = folder / str(item.get("name") or "")
        if path.is_file():
            kept = (path.name, str(item.get("content_type") or "application/octet-stream"), path.read_bytes())
            found.append(kept + (str(item["cid"]),) if item.get("cid") else kept)
    return found


def due_emails(limit: int = 20, *, now: Optional[datetime] = None) -> list[dict]:
    """Messages waiting to go whose time has come, oldest first."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM email_outbox WHERE status=? AND (next_attempt_at IS NULL OR next_attempt_at<=?) "
            "ORDER BY id LIMIT ?", (QUEUED, _stamp(now or _now()), int(limit))).fetchall()
    return [_row(r) for r in rows]


def claim_email(email_id: int) -> bool:
    """This sender takes the message: False when another one took it first."""
    with get_db() as conn:
        cur = conn.execute("UPDATE email_outbox SET status=?, attempts=attempts+1, claimed_at=? WHERE id=? AND status=?",
                           (SENDING, _stamp(_now()), int(email_id), QUEUED))
        return int(getattr(cur, "rowcount", 0) or 0) == 1


def mark_sent(email_id: int, *, provider: str, provider_id: str = "") -> None:
    with get_db() as conn:
        conn.execute("UPDATE email_outbox SET status=?, sent_at=?, provider=?, provider_message_id=?, last_error='' "
                     "WHERE id=?", (SENT, _stamp(_now()), provider, str(provider_id or "")[:200], int(email_id)))


def mark_retry(email_id: int, *, error: str, after: timedelta, provider: str = "") -> None:
    with get_db() as conn:
        conn.execute("UPDATE email_outbox SET status=?, last_error=?, next_attempt_at=?, provider=? WHERE id=?",
                     (QUEUED, str(error)[:500], _stamp(_now() + after), provider, int(email_id)))


def mark_failed(email_id: int, *, error: str, provider: str = "") -> None:
    with get_db() as conn:
        conn.execute("UPDATE email_outbox SET status=?, last_error=?, provider=? WHERE id=?",
                     (FAILED, str(error)[:500], provider, int(email_id)))


def requeue_email(email_id: int) -> bool:
    """A failed or skipped message tried again now (an admin's Retry)."""
    with get_db() as conn:
        cur = conn.execute("UPDATE email_outbox SET status=?, attempts=0, next_attempt_at=?, last_error='' "
                           "WHERE id=? AND status IN (?,?)", (QUEUED, _stamp(_now()), int(email_id), FAILED, SKIPPED))
        return int(getattr(cur, "rowcount", 0) or 0) == 1


def release_stuck_emails(older_than: timedelta = timedelta(minutes=15)) -> int:
    """Messages left 'sending' by a sender that stopped (a restart mid-send) go back in the queue."""
    with get_db() as conn:
        cur = conn.execute("UPDATE email_outbox SET status=? WHERE status=? AND claimed_at<?",
                           (QUEUED, SENDING, _stamp(_now() - older_than)))
        return int(getattr(cur, "rowcount", 0) or 0)


def list_emails(*, status: Optional[str] = None, kind: Optional[str] = None, account_id: Optional[str] = None,
                limit: int = 100) -> list[dict]:
    clauses, params = [], []
    if status:
        clauses.append("status=?"); params.append(status)
    if kind:
        clauses.append("kind=?"); params.append(kind)
    if account_id is not None:
        clauses.append("account_id=?"); params.append(account_id)
    where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
    with get_db() as conn:
        rows = conn.execute(f"SELECT * FROM email_outbox {where}ORDER BY id DESC LIMIT ?", (*params, int(limit))).fetchall()
    return [_row(r) for r in rows]


def email_counts() -> dict[str, int]:
    with get_db() as conn:
        rows = conn.execute("SELECT status, COUNT(*) AS n FROM email_outbox GROUP BY status").fetchall()
    counts = {status: 0 for status in STATUSES}
    counts.update({r["status"]: int(r["n"]) for r in rows})
    return counts


def delete_email_attachments(email_id: int) -> None:
    shutil.rmtree(attachments_root() / str(int(email_id)), ignore_errors=True)


# ── Opting out ────────────────────────────────────────────────────────────────

def opt_out(address: str, kind: str, ref: str = "") -> None:
    address = str(address or "").strip().lower()
    with get_db() as conn:
        if conn.execute("SELECT 1 FROM email_opt_out WHERE address=? AND kind=? AND ref=?",
                        (address, kind, ref)).fetchone() is None:
            conn.execute("INSERT INTO email_opt_out (address, kind, ref) VALUES (?,?,?)", (address, kind, ref))


def opted_out(address: str, kind: str, ref: str = "") -> bool:
    """Whether ``address`` asked to stop this kind of email: this one (``ref``), or every one of the kind."""
    address = str(address or "").strip().lower()
    with get_db() as conn:
        return conn.execute("SELECT 1 FROM email_opt_out WHERE address=? AND kind=? AND (ref=? OR ref='')",
                            (address, kind, ref)).fetchone() is not None


def opt_back_in(address: str, kind: str, ref: str = "") -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM email_opt_out WHERE address=? AND kind=? AND ref=?",
                     (str(address or "").strip().lower(), kind, ref))
