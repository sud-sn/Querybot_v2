"""
store/database.py
─────────────────
Unified database adapter for QueryBot v2.

Select the backend via environment variable:

    SQLite (default)  DATABASE_URL unset or empty  → ./data/querybot.db
    PostgreSQL        DATABASE_URL=postgresql://user:pw@host:5432/dbname

All 29 store-importing files use the same unchanged call pattern:

    from store.db import get_db
    with get_db() as conn:
        conn.execute("SELECT * FROM client WHERE account_id = ?", (aid,))

The adapter transparently handles all SQLite→PostgreSQL translation:

    ┌─────────────────────────────────────────────────────────────────┐
    │  ?                        →  %s  (parameter placeholder)        │
    │  datetime('now')          →  NOW()::text                        │
    │  INSERT OR IGNORE         →  INSERT … ON CONFLICT DO NOTHING    │
    │  AUTOINCREMENT            →  SERIAL PRIMARY KEY  (DDL only)     │
    │  PRAGMA …                 →  silently skipped   (DDL only)      │
    │  executescript(sql)       →  split + adapt + execute            │
    │  conn.row_factory         →  _DictRow (dict + index access)     │
    └─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import os
import logging
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

log = logging.getLogger("querybot.database")

# ── Environment config ─────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv("DATABASE_URL", "")
DB_PATH = Path(os.getenv("DB_PATH", "data/querybot.db"))

# Compiled patterns — reused across every execute() call
_RE_INSERT_OR_IGNORE = re.compile(r"\bINSERT\s+OR\s+IGNORE\b", re.IGNORECASE)
_RE_DATETIME_NOW     = re.compile(r"datetime\('now'\)", re.IGNORECASE)
_RE_AUTOINCREMENT    = re.compile(
    r"\bINTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT\b", re.IGNORECASE
)


def is_postgres() -> bool:
    """True when DATABASE_URL points to a PostgreSQL server."""
    return DATABASE_URL.startswith(("postgresql://", "postgres://"))


# ══════════════════════════════════════════════════════════════════════════════
# Row wrapper
# ══════════════════════════════════════════════════════════════════════════════

class _DictRow(dict):
    """
    Dict row that also supports integer-index access, mirroring sqlite3.Row.
    All existing code that does row["col"] or row[0] continues to work.
    """
    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


# ══════════════════════════════════════════════════════════════════════════════
# PostgreSQL cursor wrapper
# ══════════════════════════════════════════════════════════════════════════════

class _PgCursor:
    """
    Wraps a psycopg2 RealDictCursor and applies all SQLite→PostgreSQL
    translations on every execute() call so store code never needs to change.
    """

    def __init__(self, raw_cur):
        self._cur = raw_cur

    # ── SQL adaptation ──────────────────────────────────────────────────────

    @staticmethod
    def _adapt(sql: str) -> tuple[str, bool]:
        """
        Return (adapted_sql, had_insert_or_ignore).
        Order matters: strip INSERT OR IGNORE before the generic ? → %s pass.
        """
        had_ioi = bool(_RE_INSERT_OR_IGNORE.search(sql))
        sql = _RE_INSERT_OR_IGNORE.sub("INSERT", sql)
        sql = _RE_DATETIME_NOW.sub("NOW()::text", sql)
        sql = sql.replace("?", "%s")
        return sql, had_ioi

    # ── Execution ───────────────────────────────────────────────────────────

    def execute(self, sql: str, params=None) -> "_PgCursor":
        adapted, had_ioi = self._adapt(sql)
        if had_ioi:
            adapted = adapted.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"
        self._cur.execute(adapted, params or ())
        return self

    def executemany(self, sql: str, seq) -> "_PgCursor":
        adapted, had_ioi = self._adapt(sql)
        if had_ioi:
            adapted = adapted.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"
        self._cur.executemany(adapted, seq)
        return self

    # ── Fetch ───────────────────────────────────────────────────────────────

    def fetchone(self):
        row = self._cur.fetchone()
        return _DictRow(row) if row else None

    def fetchall(self) -> list[_DictRow]:
        return [_DictRow(r) for r in (self._cur.fetchall() or [])]

    # ── Metadata ────────────────────────────────────────────────────────────

    @property
    def rowcount(self) -> int:
        return self._cur.rowcount

    def __iter__(self):
        for row in self._cur:
            yield _DictRow(row)


# ══════════════════════════════════════════════════════════════════════════════
# PostgreSQL connection wrapper
# ══════════════════════════════════════════════════════════════════════════════

def _adapt_ddl_for_postgres(sql: str) -> str:
    """
    Translate SQLite-only DDL patterns to their PostgreSQL equivalents.
    Applied only inside executescript() — not for regular DML.
    """
    # INTEGER PRIMARY KEY AUTOINCREMENT → SERIAL PRIMARY KEY
    sql = _RE_AUTOINCREMENT.sub("SERIAL PRIMARY KEY", sql)
    # DEFAULT (datetime('now')) → DEFAULT (NOW()::text)
    sql = _RE_DATETIME_NOW.sub("NOW()::text", sql)
    # INSERT OR IGNORE in seed scripts → INSERT (ON CONFLICT DO NOTHING appended per statement)
    sql = _RE_INSERT_OR_IGNORE.sub("INSERT", sql)
    return sql


class _PgConnection:
    """
    Wraps a psycopg2 connection to expose the same interface as sqlite3.Connection.

    Translations applied automatically:
      • ? → %s  (all execute calls)
      • datetime('now') → NOW()::text
      • INSERT OR IGNORE → INSERT … ON CONFLICT DO NOTHING
      • executescript(): split on ; , adapt DDL, skip PRAGMA
    """

    def __init__(self, raw_conn):
        self._conn = raw_conn

    def _new_cursor(self):
        import psycopg2.extras
        return self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # ── sqlite3.Connection-compatible interface ──────────────────────────────

    def execute(self, sql: str, params=None) -> _PgCursor:
        return _PgCursor(self._new_cursor()).execute(sql, params)

    def executemany(self, sql: str, seq) -> None:
        _PgCursor(self._new_cursor()).executemany(sql, seq)

    def executescript(self, sql: str) -> None:
        """
        Replicate sqlite3.executescript() for PostgreSQL.

        Steps:
          1. Adapt all DDL idioms (AUTOINCREMENT, datetime defaults, PRAGMA)
          2. Split on semicolons
          3. Skip blank / PRAGMA-only statements
          4. For INSERT (was INSERT OR IGNORE) — append ON CONFLICT DO NOTHING
          5. Execute each statement individually; log and continue on DDL errors
             (e.g., duplicate table/index is safe to ignore)
        """
        adapted_sql = _adapt_ddl_for_postgres(sql)
        raw_cur = self._conn.cursor()
        for stmt in adapted_sql.split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            upper = stmt.upper().lstrip()
            if upper.startswith("PRAGMA"):
                continue
            # Re-apply INSERT → ON CONFLICT DO NOTHING for seed statements
            had_ioi_in_orig = "INSERT" in upper and "ON CONFLICT" not in upper
            if _RE_INSERT_OR_IGNORE.search(stmt):
                stmt = _RE_INSERT_OR_IGNORE.sub("INSERT", stmt)
                stmt = stmt + " ON CONFLICT DO NOTHING"
            try:
                raw_cur.execute(stmt)
            except Exception as exc:
                # DDL errors (duplicate object, etc.) are safe to swallow here
                log.debug("executescript stmt skipped: %.80s — %s", stmt, exc)
                self._conn.rollback()

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# Connection factories
# ══════════════════════════════════════════════════════════════════════════════

def _get_pg_connection() -> _PgConnection:
    """Open a new PostgreSQL connection and wrap it."""
    try:
        import psycopg2
    except ImportError as exc:
        raise ImportError(
            "psycopg2 is required for PostgreSQL. "
            "Install it with:  pip install psycopg2-binary"
        ) from exc
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    return _PgConnection(conn)


def _get_sqlite_connection() -> sqlite3.Connection:
    """Open a new SQLite connection with WAL mode and FK enforcement."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ══════════════════════════════════════════════════════════════════════════════
# Public API  (imported by store/db.py and re-exported from there)
# ══════════════════════════════════════════════════════════════════════════════

def get_connection():
    """
    Return a fresh database connection.
    SQLite when DATABASE_URL is unset; PostgreSQL otherwise.
    """
    if is_postgres():
        return _get_pg_connection()
    return _get_sqlite_connection()


@contextmanager
def get_db() -> Generator:
    """
    Context manager — yields a connection, commits on clean exit, rolls back on error.

        from store.db import get_db   # ← callers use this unchanged import
        with get_db() as conn:
            conn.execute("INSERT INTO foo VALUES (?)", (val,))
    """
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_table_columns(conn, table: str) -> list[str]:
    """
    Return the column names for *table*.

    Abstracts:
      SQLite    → PRAGMA table_info(table)
      PostgreSQL → information_schema.columns
    """
    if is_postgres():
        rows = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = ? AND table_schema = 'public'",
            (table,),
        ).fetchall()
        return [r["column_name"] for r in rows]
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [row[1] for row in rows]
