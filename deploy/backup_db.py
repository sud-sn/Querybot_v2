"""Copies QueryBot's SQLite database to <database>.bak-<time>, consistently, even while QueryBot runs.

    python deploy/backup_db.py            # prints the copy's path, or nothing when there is no SQLite database

Run it from the QueryBot folder. Where the database is comes from QUERYBOT_DB_PATH, from .env in that folder or
the environment, as QueryBot itself reads it.

The database runs in WAL mode: recent writes sit in querybot.db-wal until they are folded in, so copying the
.db file alone can miss them (or catch it half written). SQLite's backup reads both. A server on PostgreSQL
(DATABASE_URL set, or data/pg_url) has no SQLite database to copy; back it up with pg_dump.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path


def database_path() -> Path | None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(".env", override=False)
    if (os.getenv("DATABASE_URL") or "").strip():
        return None
    # PostgreSQL set on the admin's System page is kept in pg_url beside the database (store/database.py).
    pg_url = Path(os.getenv("QUERYBOT_DB_PATH") or "data/querybot.db").parent / "pg_url"
    if pg_url.is_file() and pg_url.read_text(encoding="utf-8").strip():
        return None
    path = Path(os.getenv("QUERYBOT_DB_PATH") or os.getenv("DB_PATH") or "data/querybot.db")
    return path if path.is_file() else None


def backup(path: Path) -> Path:
    target = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    source = sqlite3.connect(path)
    try:
        copy = sqlite3.connect(target)
        try:
            source.backup(copy)
        finally:
            copy.close()
    finally:
        source.close()
    return target


def main() -> int:
    path = database_path()
    if path is not None:
        print(backup(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
