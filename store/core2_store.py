"""Persistence for core2: semantic model versions, admin overrides, build runs.

A workspace's model is identified by (account_id, db_config_id); every build
writes a new version and the last ``KEEP_VERSIONS`` are kept. Overrides are one
row per decision (object key + field) and survive every rebuild. Tables are
created in store/db.py::_ensure_core2_tables.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import uuid
from typing import Any

from store.db import get_db

KEEP_VERSIONS = 10


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def save_core2_model(account_id: str, db_config_id: int | None, model_json: str, *,
                     source_hash: str = "", stats: dict[str, Any] | None = None) -> int:
    """Store a new model version and return its number."""
    db_id = int(db_config_id or 0)
    with get_db() as conn:
        row = conn.execute("SELECT MAX(version) AS v FROM core2_model WHERE account_id = ? AND db_config_id = ?",
                           (account_id, db_id)).fetchone()
        version = int((row["v"] if row else 0) or 0) + 1
        conn.execute(
            """INSERT INTO core2_model(account_id, db_config_id, version, built_at, source_hash, model_json, stats_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (account_id, db_id, version, _now(), source_hash, model_json, json.dumps(stats or {}, default=str)))
        conn.execute("DELETE FROM core2_model WHERE account_id = ? AND db_config_id = ? AND version <= ?",
                     (account_id, db_id, version - KEEP_VERSIONS))
    return version


def load_core2_model(account_id: str, db_config_id: int | None, version: int | None = None) -> dict[str, Any] | None:
    """``{version, built_at, source_hash, model_json, stats}`` for the latest (or a given) version."""
    db_id = int(db_config_id or 0)
    with get_db() as conn:
        if version is None:
            row = conn.execute(
                """SELECT * FROM core2_model WHERE account_id = ? AND db_config_id = ?
                   ORDER BY version DESC LIMIT 1""", (account_id, db_id)).fetchone()
        else:
            row = conn.execute("SELECT * FROM core2_model WHERE account_id = ? AND db_config_id = ? AND version = ?",
                               (account_id, db_id, int(version))).fetchone()
    if not row:
        return None
    try:
        stats = json.loads(row["stats_json"] or "{}")
    except (TypeError, ValueError):
        stats = {}
    return {"version": int(row["version"]), "built_at": row["built_at"], "source_hash": row["source_hash"],
            "model_json": row["model_json"], "stats": stats}


def list_core2_model_versions(account_id: str, db_config_id: int | None) -> list[dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT version, built_at, source_hash, stats_json FROM core2_model
               WHERE account_id = ? AND db_config_id = ? ORDER BY version DESC""",
            (account_id, int(db_config_id or 0))).fetchall()
    return [{"version": int(r["version"]), "built_at": r["built_at"], "source_hash": r["source_hash"]} for r in rows]


def set_core2_override(account_id: str, db_config_id: int | None, object_key: str, field: str, value: Any, *,
                       author: str = "admin", note: str = "") -> None:
    with get_db() as conn:
        conn.execute(
            """INSERT INTO core2_override(account_id, db_config_id, object_key, field, value_json, author, note, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(account_id, db_config_id, object_key, field)
               DO UPDATE SET value_json = excluded.value_json, author = excluded.author,
                             note = excluded.note, updated_at = excluded.updated_at""",
            (account_id, int(db_config_id or 0), object_key, field, json.dumps(value, default=str), author, note,
             _now()))


def delete_core2_override(account_id: str, db_config_id: int | None, object_key: str, field: str) -> None:
    with get_db() as conn:
        conn.execute("""DELETE FROM core2_override
                        WHERE account_id = ? AND db_config_id = ? AND object_key = ? AND field = ?""",
                     (account_id, int(db_config_id or 0), object_key, field))


def list_core2_overrides(account_id: str, db_config_id: int | None) -> list[dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT object_key, field, value_json, author, note, updated_at FROM core2_override
               WHERE account_id = ? AND db_config_id = ? ORDER BY object_key, field""",
            (account_id, int(db_config_id or 0))).fetchall()
    out = []
    for r in rows:
        try:
            value = json.loads(r["value_json"])
        except (TypeError, ValueError):
            continue
        out.append({"object_key": r["object_key"], "field": r["field"], "value": value, "author": r["author"],
                    "note": r["note"], "updated_at": r["updated_at"]})
    return out


_RUNNER_ENV = "QUERYBOT_CORE2_RUNNER"


def core2_runner() -> str:
    """This service process. A Learn runs inside the process that started it, so one started by
    another process (before a restart) will never finish. Kept in the process's environment,
    which a restart starts afresh and a re-imported module does not."""
    runner = os.environ.get(_RUNNER_ENV)
    if not runner:
        runner = os.environ[_RUNNER_ENV] = uuid.uuid4().hex
    return runner


def start_core2_build(account_id: str, db_config_id: int | None) -> str:
    started = _now()
    with get_db() as conn:
        conn.execute("""INSERT OR IGNORE INTO core2_build(account_id, db_config_id, started_at, status, runner)
                        VALUES (?, ?, ?, 'running', ?)""", (account_id, int(db_config_id or 0), started, core2_runner()))
    return started


def finish_core2_build(account_id: str, db_config_id: int | None, started_at: str, *, status: str,
                       message: str = "", version: int = 0) -> None:
    with get_db() as conn:
        conn.execute("""UPDATE core2_build SET finished_at = ?, status = ?, message = ?, version = ?
                        WHERE account_id = ? AND db_config_id = ? AND started_at = ?""",
                     (_now(), status, message[:2000], int(version), account_id, int(db_config_id or 0), started_at))


def add_core2_build_line(account_id: str, db_config_id: int | None, started_at: str, line: str) -> None:
    """One step of a Learn as it happens, timed: what the learned page shows while it runs."""
    text = " ".join(str(line).split())[:600]
    with get_db() as conn:
        conn.execute("""UPDATE core2_build SET log = log || ? WHERE account_id = ? AND db_config_id = ?
                        AND started_at = ?""", (f"{_now()[11:]} {text}\n", account_id, int(db_config_id or 0),
                                                started_at))


def set_core2_import_report(account_id: str, db_config_id: int | None, report: dict[str, Any]) -> None:
    """What the latest import brought over from today's setup, and what it could not (shown on the page)."""
    with get_db() as conn:
        row = conn.execute("""SELECT started_at FROM core2_build WHERE account_id = ? AND db_config_id = ?
                              ORDER BY started_at DESC LIMIT 1""", (account_id, int(db_config_id or 0))).fetchone()
        if row is None:
            return
        conn.execute("""UPDATE core2_build SET import_report = ? WHERE account_id = ? AND db_config_id = ?
                        AND started_at = ?""", (json.dumps(report, default=str), account_id,
                                               int(db_config_id or 0), row["started_at"]))


def latest_core2_build(account_id: str, db_config_id: int | None) -> dict[str, Any] | None:
    with get_db() as conn:
        row = conn.execute("""SELECT * FROM core2_build WHERE account_id = ? AND db_config_id = ?
                              ORDER BY started_at DESC LIMIT 1""", (account_id, int(db_config_id or 0))).fetchone()
    return dict(row) if row else None


# ── the switch, and what the new core answered ─────────────────────────────

ENGINES = ("legacy", "compare", "core2")


def get_query_engine(account_id: str) -> str:
    """Which core answers the workspace's portal questions: legacy (default), compare or core2."""
    try:
        with get_db() as conn:
            row = conn.execute("SELECT query_engine FROM client WHERE account_id=?", (account_id,)).fetchone()
    except Exception:
        return "legacy"
    value = str((row[0] if row else "") or "legacy")
    return value if value in ENGINES else "legacy"


def set_query_engine(account_id: str, engine: str) -> None:
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}")
    with get_db() as conn:
        conn.execute("UPDATE client SET query_engine=?, updated_at=datetime('now') WHERE account_id=?",
                     (engine, account_id))


def log_core2_answer(account_id: str, *, user_id: str, mode: str, question: str, status: str, headline: str,
                     sql: str, row_count: int, plan: dict[str, Any] | None, duration_ms: int,
                     model_version: int, question_id: str = "") -> None:
    with get_db() as conn:
        conn.execute(
            """INSERT INTO core2_answer (account_id, user_id, mode, question, status, headline, sql, row_count,
                                         plan_json, duration_ms, model_version, question_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (account_id, user_id, mode, question[:2000], status, headline[:2000], sql, row_count,
             json.dumps(plan or {}), duration_ms, model_version, question_id))


def list_core2_answers(account_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """The new core's recent answers, with the reader's thumbs on each when they gave one."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT a.id, a.user_id, a.mode, a.question, a.status, a.headline, a.sql, a.row_count, a.plan_json,
                      a.duration_ms, a.model_version, a.created_at, a.question_id,
                      (SELECT f.rating FROM answer_feedback f WHERE a.question_id <> '' AND f.question_id = a.question_id
                        ORDER BY f.updated_at DESC LIMIT 1) AS rating
               FROM core2_answer a WHERE a.account_id=? ORDER BY a.id DESC LIMIT ?""", (account_id, limit)).fetchall()
    keys = ("id", "user_id", "mode", "question", "status", "headline", "sql", "row_count", "plan_json",
            "duration_ms", "model_version", "created_at", "question_id", "rating")
    return [dict(zip(keys, r)) for r in rows]
