"""Persistence for core2: semantic model versions, admin overrides, build runs, readers' requests.

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


def core2_model_stamp(account_id: str, db_config_id: int | None) -> tuple | None:
    """What the model questions are answered with is made of, cheaply: the latest version (and when it was
    built) and every admin decision on it. None before the first Learn. Equal stamps, equal models."""
    import hashlib

    db_id = int(db_config_id or 0)
    with get_db() as conn:
        latest = conn.execute("""SELECT version, built_at FROM core2_model WHERE account_id = ? AND db_config_id = ?
                                 ORDER BY version DESC LIMIT 1""", (account_id, db_id)).fetchone()
        if not latest:
            return None
        decisions = conn.execute("""SELECT object_key, field, value_json FROM core2_override
                                    WHERE account_id = ? AND db_config_id = ? ORDER BY object_key, field""",
                                 (account_id, db_id)).fetchall()
    digest = hashlib.sha256()
    for row in decisions:
        digest.update(json.dumps([row["object_key"], row["field"], row["value_json"]]).encode())
    return int(latest["version"]), str(latest["built_at"] or ""), len(decisions), digest.hexdigest()


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


# A Learn a restart stopped is picked up within this many hours; after that, its tables are read again.
RESUME_HOURS = 24


def keep_core2_learn_table(account_id: str, db_config_id: int | None, started_at: str, table_key: str,
                           fingerprint: str, profile_json: str) -> None:
    """One table the Learn started at ``started_at`` has read: kept until that Learn finishes."""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO core2_learn_table(account_id, db_config_id, table_key, started_at, fingerprint,
                                            profile_json, read_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(account_id, db_config_id, table_key)
               DO UPDATE SET started_at = excluded.started_at, fingerprint = excluded.fingerprint,
                             profile_json = excluded.profile_json, read_at = excluded.read_at""",
            (account_id, int(db_config_id or 0), table_key, started_at, fingerprint, profile_json, _now()))


def resume_core2_learn(account_id: str, db_config_id: int | None, started_at: str) -> dict[str, tuple[str, str]]:
    """The tables the Learn before this one read, when it did not finish: ``{table_key: (fingerprint, profile)}``.

    Handed to the Learn started at ``started_at`` only when the one before it was stopped (by a restart, or a
    failure) and read them in the last RESUME_HOURS: a Learn after a finished one reads everything again, as
    an admin who learns again expects. Whatever else was kept is removed.
    """
    db_id = int(db_config_id or 0)
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=RESUME_HOURS)).strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        before = conn.execute("""SELECT started_at, status FROM core2_build
                                 WHERE account_id = ? AND db_config_id = ? AND started_at < ?
                                 ORDER BY started_at DESC LIMIT 1""", (account_id, db_id, started_at)).fetchone()
        if before is not None and before["status"] != "done":
            conn.execute("""UPDATE core2_learn_table SET started_at = ?
                            WHERE account_id = ? AND db_config_id = ? AND started_at = ? AND read_at >= ?""",
                         (started_at, account_id, db_id, before["started_at"], cutoff))
        conn.execute("DELETE FROM core2_learn_table WHERE account_id = ? AND db_config_id = ? AND started_at <> ?",
                     (account_id, db_id, started_at))
        rows = conn.execute("""SELECT table_key, fingerprint, profile_json FROM core2_learn_table
                               WHERE account_id = ? AND db_config_id = ? AND started_at = ?""",
                            (account_id, db_id, started_at)).fetchall()
    return {r["table_key"]: (r["fingerprint"], r["profile_json"]) for r in rows}


def forget_core2_learn(account_id: str, db_config_id: int | None) -> None:
    """A finished Learn's tables are in its model: the copies kept while it ran go."""
    with get_db() as conn:
        conn.execute("DELETE FROM core2_learn_table WHERE account_id = ? AND db_config_id = ?",
                     (account_id, int(db_config_id or 0)))


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


# ── what the new core answered ──────────────────────────────────────────────

# The new core answers every workspace, on every channel. The old engine and the side-by-side comparison are gone
# from the product; a value stored in client.query_engine before is no longer read.
ENGINES = ("core2",)


def get_query_engine(account_id: str) -> str:
    """Which core answers the workspace's questions: the new core, always."""
    return "core2"


def set_query_engine(account_id: str, engine: str) -> None:
    """Kept for callers of the old switch: only the new core can be chosen."""
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}: the new core answers every workspace")


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


def get_core2_answer_plan(account_id: str, question_id: str) -> dict[str, Any] | None:
    """The plan a new-core answer was given from, by its question id (None when none was kept)."""
    if not question_id:
        return None
    with get_db() as conn:
        row = conn.execute("SELECT plan_json FROM core2_answer WHERE account_id=? AND question_id=? "
                           "ORDER BY id DESC LIMIT 1", (account_id, question_id)).fetchone()
    try:
        plan = json.loads(row["plan_json"] or "{}") if row else None
    except (TypeError, ValueError):
        return None
    return plan if isinstance(plan, dict) and plan else None


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


def core2_measure_uses(account_id: str, since: str) -> dict[str, int]:
    """How many answered questions asked for each measure (by its slug) since ``since``."""
    with get_db() as conn:
        rows = conn.execute("""SELECT plan_json FROM core2_answer
                               WHERE account_id = ? AND created_at >= ? AND status = 'answered'""",
                            (account_id, since)).fetchall()
    counts: dict[str, int] = {}
    for row in rows:
        try:
            plan = json.loads(row["plan_json"] or "{}")
        except (TypeError, ValueError):
            continue
        slugs = set(plan.get("measures") or []) | {s for d in plan.get("derived") or [] for s in d.get("measures") or []}
        for slug in slugs:
            counts[str(slug)] = counts.get(str(slug), 0) + 1
    return counts


# ── readers' requests ────────────────────────────────────────────────────────
# A change a reader asked for (what a metric means, the other names people use, the date it is
# counted by), or a metric they want added. An admin accepts it, which writes it as admin decisions
# (``applied`` keeps what each decision was before, so it can be undone), or rejects it with a reason
# the reader is shown.

REQUEST_STATUSES = ("waiting", "accepted", "rejected", "withdrawn", "undone")


def _request(row: Any) -> dict[str, Any]:
    out = dict(row)
    for key, empty in (("changes_json", []), ("applied_json", []), ("definition_json", None)):
        try:
            out[key.removesuffix("_json")] = json.loads(row[key]) if row[key] else empty
        except (TypeError, ValueError):
            out[key.removesuffix("_json")] = empty
        out.pop(key, None)
    return out


def add_core2_request(account_id: str, db_config_id: int | None, *, kind: str, user_id: str, user_name: str,
                      target_kind: str = "", target_key: str = "", target_name: str = "",
                      changes: list[dict[str, Any]] | None = None, note: str = "", example: str = "",
                      definition: dict[str, Any] | None = None, source: str = "page", question_id: str = "") -> int:
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO core2_request(account_id, db_config_id, kind, target_kind, target_key, target_name,
                                         changes_json, note, example, definition_json, source, question_id,
                                         user_id, user_name, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (account_id, int(db_config_id or 0), kind, target_kind, target_key, target_name,
             json.dumps(changes or [], default=str), note, example,
             json.dumps(definition, default=str) if definition else "", source, question_id, str(user_id),
             user_name, _now()))
        return int(cur.lastrowid)


def get_core2_request(account_id: str, request_id: int) -> dict[str, Any] | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM core2_request WHERE account_id = ? AND id = ?",
                           (account_id, int(request_id))).fetchone()
    return _request(row) if row else None


def list_core2_requests(account_id: str, *, status: str | None = None, user_id: str | None = None,
                        limit: int = 200) -> list[dict[str, Any]]:
    where, params = ["account_id = ?"], [account_id]
    if status:
        where.append("status = ?")
        params.append(status)
    if user_id is not None:
        where.append("user_id = ?")
        params.append(str(user_id))
    with get_db() as conn:
        rows = conn.execute(f"SELECT * FROM core2_request WHERE {' AND '.join(where)} "
                            "ORDER BY created_at DESC, id DESC LIMIT ?", (*params, int(limit))).fetchall()
    return [_request(r) for r in rows]


def count_core2_requests(account_id: str) -> dict[str, int]:
    """How many requests are in each state."""
    with get_db() as conn:
        rows = conn.execute("SELECT status, COUNT(*) AS n FROM core2_request WHERE account_id = ? GROUP BY status",
                            (account_id,)).fetchall()
    return {**{s: 0 for s in REQUEST_STATUSES}, **{r["status"]: int(r["n"]) for r in rows}}


def waiting_core2_requests() -> dict[str, int]:
    """Requests waiting for an admin, per workspace (the admin inbox)."""
    with get_db() as conn:
        rows = conn.execute("SELECT account_id, COUNT(*) AS n FROM core2_request WHERE status = 'waiting' "
                            "GROUP BY account_id").fetchall()
    return {r["account_id"]: int(r["n"]) for r in rows}


def decide_core2_request(account_id: str, request_id: int, *, status: str, decided_by: str, reason: str = "",
                         applied: list[dict[str, Any]] | None = None, changes: list[dict[str, Any]] | None = None,
                         expect: tuple[str, ...] = ("waiting",)) -> bool:
    """Move a request on: only from a state in ``expect``, so two admins cannot both decide it."""
    if status not in REQUEST_STATUSES:
        raise ValueError(f"no such request status: {status!r}")
    sets = ["status = ?", "decided_by = ?", "reason = ?", "decided_at = ?"]
    params: list[Any] = [status, decided_by, reason, _now()]
    if applied is not None:
        sets.append("applied_json = ?")
        params.append(json.dumps(applied, default=str))
    if changes is not None:
        sets.append("changes_json = ?")
        params.append(json.dumps(changes, default=str))
    marks = ", ".join("?" for _ in expect)
    with get_db() as conn:
        cur = conn.execute(f"UPDATE core2_request SET {', '.join(sets)} WHERE account_id = ? AND id = ? "
                           f"AND status IN ({marks})", (*params, account_id, int(request_id), *expect))
        return cur.rowcount == 1


def get_core2_override(account_id: str, db_config_id: int | None, object_key: str, field: str) -> dict[str, Any] | None:
    """One decision as stored, or None: what an accepted request replaced, so it can be put back."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT value_json, author, note FROM core2_override
               WHERE account_id = ? AND db_config_id = ? AND object_key = ? AND field = ?""",
            (account_id, int(db_config_id or 0), object_key, field)).fetchone()
    if not row:
        return None
    try:
        return {"value": json.loads(row["value_json"]), "author": row["author"], "note": row["note"]}
    except (TypeError, ValueError):
        return None


def save_core2_question_map(account_id: str, db_config_id: int | None, stamp: str, map_json: str) -> None:
    """Keep a workspace's question map (core2/question_map.py), replacing the one before."""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO core2_question_map(account_id, db_config_id, stamp, built_at, map_json)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(account_id, db_config_id) DO UPDATE SET stamp = excluded.stamp,
                   built_at = excluded.built_at, map_json = excluded.map_json""",
            (account_id, int(db_config_id or 0), stamp, _now(), map_json))


def load_core2_question_map(account_id: str, db_config_id: int | None) -> dict[str, Any] | None:
    """``{stamp, built_at, map_json}`` of a workspace's question map, or None before the first."""
    with get_db() as conn:
        row = conn.execute("SELECT stamp, built_at, map_json FROM core2_question_map WHERE account_id = ? AND "
                           "db_config_id = ?", (account_id, int(db_config_id or 0))).fetchone()
    return {"stamp": row["stamp"], "built_at": row["built_at"], "map_json": row["map_json"]} if row else None


# ── a reader's start page (core2/suggest.py) ────────────────────────────────


def list_core2_answers_of(account_id: str, user_id: str, since: str, limit: int = 400) -> list[dict[str, Any]]:
    """One reader's new-core questions since ``since`` (newest first): what they asked and the plan it was answered
    with. Read by code for their suggestions; never sent to the AI."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT question, status, plan_json, created_at FROM core2_answer
               WHERE account_id = ? AND user_id = ? AND created_at >= ? ORDER BY id DESC LIMIT ?""",
            (account_id, str(user_id), since, int(limit))).fetchall()
    out = []
    for r in rows:
        try:
            plan = json.loads(r["plan_json"] or "{}")
        except (TypeError, ValueError):
            plan = {}
        out.append({"question": r["question"], "status": r["status"], "plan": plan if isinstance(plan, dict) else {},
                    "created_at": r["created_at"]})
    return out


def load_core2_start_page(account_id: str, user_id: str) -> dict[str, Any] | None:
    with get_db() as conn:
        row = conn.execute("""SELECT day, stamp, page_json, seen_at, updated_at FROM core2_start_page
                              WHERE account_id = ? AND user_id = ?""", (account_id, str(user_id))).fetchone()
    if not row:
        return None
    try:
        page = json.loads(row["page_json"] or "{}")
    except (TypeError, ValueError):
        page = {}
    return {"day": row["day"], "stamp": row["stamp"], "page": page if isinstance(page, dict) else {},
            "seen_at": row["seen_at"], "updated_at": row["updated_at"]}


def save_core2_start_page(account_id: str, user_id: str, *, day: str, stamp: str, page: dict[str, Any],
                          seen_at: str) -> None:
    with get_db() as conn:
        conn.execute(
            """INSERT INTO core2_start_page (account_id, user_id, day, stamp, page_json, seen_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(account_id, user_id) DO UPDATE SET day = excluded.day, stamp = excluded.stamp,
                   page_json = excluded.page_json, seen_at = excluded.seen_at, updated_at = excluded.updated_at""",
            (account_id, str(user_id), day, stamp, json.dumps(page, default=str), seen_at, _now()))


def core2_suggest_kinds(account_id: str, user_id: str) -> dict[str, tuple[int, int]]:
    """``{kind: (times offered, times asked)}`` of a reader's suggestions."""
    with get_db() as conn:
        rows = conn.execute("SELECT kind, shown, clicked FROM core2_suggest_kind WHERE account_id = ? AND user_id = ?",
                            (account_id, str(user_id))).fetchall()
    return {r["kind"]: (int(r["shown"]), int(r["clicked"])) for r in rows}


def count_core2_suggest_kind(account_id: str, user_id: str, kind: str, *, shown: int = 0, clicked: int = 0) -> None:
    with get_db() as conn:
        conn.execute(
            """INSERT INTO core2_suggest_kind (account_id, user_id, kind, shown, clicked) VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(account_id, user_id, kind) DO UPDATE SET shown = shown + excluded.shown,
                   clicked = clicked + excluded.clicked""",
            (account_id, str(user_id), kind, int(shown), int(clicked)))


def core2_metrics_accepted_since(account_id: str, db_config_id: int | None, since: str) -> list[str]:
    """The measures (by key) an admin accepted or wrote after ``since``, newest first."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT object_key, field, value_json FROM core2_override
               WHERE account_id = ? AND db_config_id = ? AND object_key LIKE 'measure:%' AND updated_at > ?
                     AND field IN ('status', 'define') ORDER BY updated_at DESC""",
            (account_id, int(db_config_id or 0), since)).fetchall()
    out: list[str] = []
    for r in rows:
        if r["field"] == "status" and str(r["value_json"]).strip('"') != "approved":
            continue
        key = str(r["object_key"]).partition(":")[2]
        if key and key not in out:
            out.append(key)
    return out


def add_core2_setup_check(account_id: str, db_config_id: int | None, *, file_name: str, uploaded_by: str,
                          items: list[dict[str, Any]]) -> int:
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO core2_setup_check (account_id, db_config_id, file_name, uploaded_by, items_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (account_id, int(db_config_id or 0), file_name, uploaded_by, json.dumps(items, default=str), _now()))
        return int(cur.lastrowid)


def get_core2_setup_check(account_id: str, check_id: int) -> dict[str, Any] | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM core2_setup_check WHERE account_id = ? AND id = ?",
                           (account_id, int(check_id))).fetchone()
    if not row:
        return None
    out = dict(row)
    out["items"] = json.loads(out.pop("items_json") or "[]")
    out["summary"] = json.loads(out.pop("summary_json") or "{}")
    return out


def close_core2_setup_check(account_id: str, check_id: int, *, status: str, summary: dict[str, Any],
                            expect: str = "waiting") -> bool:
    """Move a setup check on from ``expect`` (waiting: to applying or discarded; applying: to applied or failed),
    once only, so two admins never apply the same one twice."""
    with get_db() as conn:
        cur = conn.execute(
            """UPDATE core2_setup_check SET status = ?, summary_json = ?, applied_at = ?
               WHERE account_id = ? AND id = ? AND status = ?""",
            (status, json.dumps(summary, default=str), _now(), account_id, int(check_id), expect))
        return cur.rowcount == 1


def list_core2_setup_checks(account_id: str, limit: int = 5) -> list[dict[str, Any]]:
    """The latest setup checks uploaded for a workspace, without their items."""
    with get_db() as conn:
        rows = conn.execute("""SELECT id, file_name, uploaded_by, status, summary_json, created_at, applied_at
                               FROM core2_setup_check WHERE account_id = ? ORDER BY id DESC LIMIT ?""",
                            (account_id, int(limit))).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["summary"] = json.loads(item.pop("summary_json") or "{}")
        out.append(item)
    return out


def add_core2_test_question(account_id: str, db_config_id: int | None, *, question: str, expected: str = "",
                            report: str = "", period: str = "", note: str = "", setup_check_id: int = 0) -> int:
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO core2_test_question (account_id, db_config_id, question, expected, report, period, note,
                                                setup_check_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (account_id, int(db_config_id or 0), question, expected, report, period, note, int(setup_check_id),
             _now()))
        return int(cur.lastrowid)


def list_core2_test_questions(account_id: str, db_config_id: int | None) -> list[dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute("""SELECT * FROM core2_test_question WHERE account_id = ? AND db_config_id = ?
                               ORDER BY id""", (account_id, int(db_config_id or 0))).fetchall()
    return [dict(r) for r in rows]


def delete_core2_test_question(account_id: str, db_config_id: int | None, question_id: int) -> bool:
    """A test question taken off the answers check, with its results."""
    with get_db() as conn:
        cur = conn.execute("DELETE FROM core2_test_question WHERE account_id = ? AND db_config_id = ? AND id = ?",
                           (account_id, int(db_config_id or 0), int(question_id)))
        conn.execute("""DELETE FROM core2_answer_check_result WHERE account_id = ? AND db_config_id = ?
                        AND test_question_id = ?""", (account_id, int(db_config_id or 0), int(question_id)))
        return cur.rowcount > 0


# A run its process stopped minding (a restart, a crash) is let go after this long, and another may start.
ANSWER_CHECK_STALE_MINUTES = 120


def release_stopped_core2_answer_checks(account_id: str, db_config_id: int | None) -> None:
    """A run another process was minding before it stopped (a restart), or one running for hours, is marked
    stopped: read as running, it would hold the page and every next run for good."""
    stale = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=ANSWER_CHECK_STALE_MINUTES)
             ).strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        conn.execute("""UPDATE core2_answer_check SET status = 'failed', finished_at = ?,
                        message = 'Stopped before it finished (the server restarted)'
                        WHERE account_id = ? AND db_config_id = ? AND status = 'running'
                        AND (runner != ? OR started_at < ?)""",
                     (_now(), account_id, int(db_config_id or 0), core2_runner(), stale))


def claim_core2_answer_check(account_id: str, db_config_id: int | None, *, mode: str, reason: str, stamp: Any,
                             total: int, started_by: str = "", whole: bool = True) -> int | None:
    """Start a run of the answers check, or None while one runs for the workspace (a stopped one let go first).
    ``whole``: every question, not only some (one asked again alone, new ones)."""
    db_id = int(db_config_id or 0)
    now = _now()
    release_stopped_core2_answer_checks(account_id, db_id)
    try:
        with get_db() as conn:
            cur = conn.execute(
                """INSERT INTO core2_answer_check (account_id, db_config_id, mode, reason, whole, stamp_json, status,
                                                   total, started_by, runner, started_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?)""",
                (account_id, db_id, mode, reason, 1 if whole else 0, json.dumps(stamp, default=str), int(total),
                 started_by, core2_runner(), now))
            return int(cur.lastrowid)
    except Exception as exc:  # noqa: BLE001 - the unique index says one runs already
        if "unique" in str(exc).lower():
            return None
        raise


def update_core2_answer_check(check_id: int, *, done_count: int | None = None, status: str | None = None,
                              message: str | None = None) -> None:
    sets, args = [], []
    if done_count is not None:
        sets.append("done_count = ?")
        args.append(int(done_count))
    if status is not None:
        sets += ["status = ?", "finished_at = ?"]
        args += [status, _now() if status != "running" else ""]
    if message is not None:
        sets.append("message = ?")
        args.append(message[:1000])
    if not sets:
        return
    with get_db() as conn:
        # Only a run still running: one let go as stopped stays stopped.
        conn.execute(f"UPDATE core2_answer_check SET {', '.join(sets)} WHERE id = ? AND status = 'running'",
                     (*args, int(check_id)))


def _check_row(row: Any) -> dict[str, Any]:
    item = dict(row)
    item["stamp"] = json.loads(item.pop("stamp_json") or "[]")
    return item


def latest_core2_answer_check(account_id: str, db_config_id: int | None, *, finished: bool = False,
                              whole: bool = False) -> dict[str, Any] | None:
    """The workspace's latest run of the answers check: the latest finished one, with ``finished``; the latest
    of every question, with ``whole``."""
    where = ("AND status = 'done' " if finished else "") + ("AND whole = 1" if whole else "")
    with get_db() as conn:
        row = conn.execute(f"""SELECT * FROM core2_answer_check WHERE account_id = ? AND db_config_id = ?
                               {where} ORDER BY id DESC LIMIT 1""",
                           (account_id, int(db_config_id or 0))).fetchone()
    return _check_row(row) if row else None


def add_core2_answer_result(check_id: int, account_id: str, db_config_id: int | None, test_question_id: int, *,
                            verdict: str, detail: dict[str, Any], plan: Any = None, headline: str = "",
                            asked: bool = False) -> None:
    with get_db() as conn:
        conn.execute(
            """INSERT INTO core2_answer_check_result (check_id, account_id, db_config_id, test_question_id, verdict,
                                                      asked, headline, detail_json, plan_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (int(check_id), account_id, int(db_config_id or 0), int(test_question_id), verdict, 1 if asked else 0,
             headline[:600], json.dumps(detail, default=str),
             json.dumps(plan, default=str) if plan is not None else "", _now()))


def core2_answer_results(account_id: str, db_config_id: int | None, *, per_question: int = 2
                         ) -> dict[int, list[dict[str, Any]]]:
    """Each test question's latest results, newest first (its latest and the one before, by default)."""
    with get_db() as conn:
        rows = conn.execute("""SELECT * FROM core2_answer_check_result WHERE account_id = ? AND db_config_id = ?
                               ORDER BY test_question_id, id DESC""", (account_id, int(db_config_id or 0))).fetchall()
    out: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        kept = out.setdefault(int(row["test_question_id"]), [])
        if len(kept) < per_question:
            item = dict(row)
            item["detail"] = json.loads(item.pop("detail_json") or "{}")
            text = item.pop("plan_json") or ""
            item["plan"] = json.loads(text) if text else None
            kept.append(item)
    return out


# Each question keeps its latest results, and the workspace its latest runs: enough for "was" and the history.
ANSWER_RESULTS_KEPT = 5
ANSWER_CHECKS_KEPT = 50


def prune_core2_answer_checks(account_id: str, db_config_id: int | None) -> None:
    """Results of questions taken off, results past the latest few of each question, and runs past the latest
    ones (never the one running) go."""
    db_id = int(db_config_id or 0)
    with get_db() as conn:
        conn.execute("""DELETE FROM core2_answer_check_result WHERE account_id = ? AND db_config_id = ?
                        AND test_question_id NOT IN (SELECT id FROM core2_test_question WHERE account_id = ?
                                                     AND db_config_id = ?)""", (account_id, db_id, account_id, db_id))
        conn.execute("""DELETE FROM core2_answer_check_result WHERE account_id = ? AND db_config_id = ? AND id NOT IN (
                            SELECT id FROM (SELECT id, ROW_NUMBER() OVER (PARTITION BY test_question_id ORDER BY id DESC)
                                            AS n FROM core2_answer_check_result WHERE account_id = ? AND db_config_id = ?)
                            AS ranked WHERE n <= ?)""", (account_id, db_id, account_id, db_id, ANSWER_RESULTS_KEPT))
        conn.execute("""DELETE FROM core2_answer_check WHERE account_id = ? AND db_config_id = ? AND status != 'running'
                        AND id NOT IN (SELECT id FROM (SELECT id FROM core2_answer_check WHERE account_id = ?
                                       AND db_config_id = ? ORDER BY id DESC LIMIT ?) AS kept)""",
                     (account_id, db_id, account_id, db_id, ANSWER_CHECKS_KEPT))


def core2_test_question_workspaces() -> list[tuple[str, int]]:
    """Every workspace with test questions to check its answers against."""
    with get_db() as conn:
        rows = conn.execute("SELECT DISTINCT account_id, db_config_id FROM core2_test_question").fetchall()
    return [(str(r["account_id"]), int(r["db_config_id"])) for r in rows]
