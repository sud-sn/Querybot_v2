"""
store/sign_in_throttle.py

How often a sign-in has failed lately, per identity, and how long the next
attempt has to wait.

Neither console limited attempts at all, so a password could be guessed as
fast as the server answered. Now five failures within fifteen minutes are
free -- people mistype -- and each one after that doubles the wait, from a
minute up to fifteen. While a wait runs the password is not checked, so a
guesser gets five tries and then one a minute or fewer, and someone who has
simply forgotten their password is never shut out for long. A success clears
the count, and so does any change of the password (an admin's reset included).

An identity is a scope and the names that sign in: the workspace and email
for the portal, the client address for the admin console, which has one
password and no user name. It is stored hashed, because people type
passwords into the email field.
"""

from __future__ import annotations

import hashlib
import math
import time

from store.db import get_db

FREE_FAILURES = 5
WINDOW_SECONDS = 15 * 60
FIRST_WAIT_SECONDS = 60
MAX_WAIT_SECONDS = 15 * 60


def sign_in_identity(scope: str, *parts: object) -> str:
    """The stored form of who is signing in: a hash, never the typed text."""
    raw = "\x1f".join([scope, *(str(p or "").strip().lower() for p in parts)])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sign_in_wait_seconds(identity: str, now: float | None = None) -> int:
    """Seconds before this identity may try again; 0 when it may try now."""
    now = time.time() if now is None else now
    with get_db() as conn:
        row = conn.execute(
            "SELECT locked_until FROM sign_in_attempt WHERE identity = ?", (identity,)
        ).fetchone()
    if not row or not row["locked_until"]:
        return 0
    return max(0, math.ceil(float(row["locked_until"]) - now))


def record_sign_in_failure(identity: str, scope: str, now: float | None = None) -> int:
    """Count a failed attempt. Returns how long the next attempt must now wait."""
    now = time.time() if now is None else now
    with get_db() as conn:
        row = conn.execute(
            "SELECT failures, last_failed_at FROM sign_in_attempt WHERE identity = ?", (identity,)
        ).fetchone()
        recent = bool(row) and now - float(row["last_failed_at"] or 0) < WINDOW_SECONDS
        failures = (int(row["failures"] or 0) if recent else 0) + 1
        wait = 0
        if failures > FREE_FAILURES:
            wait = min(FIRST_WAIT_SECONDS * 2 ** (failures - FREE_FAILURES - 1), MAX_WAIT_SECONDS)
        conn.execute(
            "INSERT INTO sign_in_attempt (identity, scope, failures, last_failed_at, locked_until) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(identity) DO UPDATE SET scope = excluded.scope, failures = excluded.failures, "
            "last_failed_at = excluded.last_failed_at, locked_until = excluded.locked_until",
            (identity, scope, failures, int(now), int(now) + wait if wait else None),
        )
    return wait


def clear_sign_in_failures(identity: str) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM sign_in_attempt WHERE identity = ?", (identity,))


def clear_sign_in_scope(scope: str) -> None:
    """Forget every count in a scope: how the server's reset unlocks the admin console."""
    with get_db() as conn:
        conn.execute("DELETE FROM sign_in_attempt WHERE scope = ?", (scope,))
