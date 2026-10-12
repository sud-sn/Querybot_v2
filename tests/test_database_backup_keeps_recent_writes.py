"""The update scripts' database copy (deploy/backup_db.py). The store runs in WAL mode, so the latest writes can
sit in querybot.db-wal while QueryBot runs: a copy of the .db file alone loses them, SQLite's backup does not."""

from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import backup_db  # noqa: E402


def _open_with_unfolded_writes(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")       # keep every write in the -wal file, as a busy server may
    conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)")
    conn.executemany("INSERT INTO notes (body) VALUES (?)", [(f"note {i}",) for i in range(50)])
    conn.commit()
    return conn


def test_the_copy_has_the_writes_still_in_the_wal_file(tmp_path):
    db = tmp_path / "querybot.db"
    live = _open_with_unfolded_writes(db)               # QueryBot still running
    try:
        assert (tmp_path / "querybot.db-wal").stat().st_size > 0
        plain = tmp_path / "plain-copy.db"
        shutil.copyfile(db, plain)                      # what a file copy would have kept
        assert "notes" not in {r[0] for r in sqlite3.connect(plain).execute("SELECT name FROM sqlite_master")}

        copy = backup_db.backup(db)
        assert copy.name.startswith("querybot.db.bak-")
        assert sqlite3.connect(copy).execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 50
    finally:
        live.close()


def test_it_follows_the_database_setting_and_skips_postgres(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DB_PATH", raising=False)
    monkeypatch.delenv("QUERYBOT_DB_PATH", raising=False)
    assert backup_db.database_path() is None            # no database yet: nothing to copy, not an error

    other = tmp_path / "store" / "qb.db"
    other.parent.mkdir()
    sqlite3.connect(other).close()
    (tmp_path / ".env").write_text(f"QUERYBOT_DB_PATH={other}\n", encoding="utf-8")
    assert backup_db.database_path() == other           # read from .env, as QueryBot reads it

    monkeypatch.setenv("DATABASE_URL", "postgresql://qb@localhost/qb")
    assert backup_db.database_path() is None            # PostgreSQL is backed up with pg_dump


def test_it_skips_postgres_set_on_the_admin_page(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("DATABASE_URL", "DB_PATH", "QUERYBOT_DB_PATH"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "data").mkdir()
    sqlite3.connect(tmp_path / "data" / "querybot.db").close()   # the SQLite file left from before
    assert backup_db.database_path() == Path("data/querybot.db")
    (tmp_path / "data" / "pg_url").write_text("postgresql://qb@localhost/qb\n", encoding="utf-8")
    assert backup_db.database_path() is None
