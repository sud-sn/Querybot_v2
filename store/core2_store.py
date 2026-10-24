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
                     model_version: int, question_id: str = "", verifiable: bool = False) -> None:
    """One new-core answer. ``verifiable``: read from the question's own words alone (core2/service.py), so an admin
    may verify it."""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO core2_answer (account_id, user_id, mode, question, status, headline, sql, row_count,
                                         plan_json, duration_ms, model_version, question_id, verifiable)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (account_id, user_id, mode, question[:2000], status, headline[:2000], sql, row_count,
             json.dumps(plan or {}), duration_ms, model_version, question_id, 1 if verifiable else 0))


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


# ── Verified answers (core2/verified.py) ──────────────────────────────────────

def _verified_row(row) -> dict[str, Any]:
    try:
        plan = json.loads(row["plan_json"] or "{}")
    except (TypeError, ValueError):
        plan = {}
    return {"id": int(row["id"]), "question": row["question"], "plan": plan if isinstance(plan, dict) else {},
            "headline": row["headline"] or "", "by": row["verified_by"], "at": row["verified_at"],
            "uses": int(row["uses"] or 0),
            "last_used_at": row["last_used_at"], "needs_look": bool(row["needs_look_at"])}


def verify_answer(account_id: str, question: str, plan: dict[str, Any], *, by: str, headline: str = "") -> int:
    """Keep ``plan`` as the checked answer to ``question``; the same question verified again takes the new plan
    (and is looked at again) and keeps the words it was first verified in. ``headline`` is what the answer said
    when it was verified, for the admin's page only: a verified question is answered on the data of the day it is
    asked. Its id."""
    from core2.verified import key

    words = key(question)
    if not words:
        raise ValueError("a question with no words cannot be verified")
    with get_db() as conn:
        conn.execute(
            """INSERT INTO core2_verified (account_id, question, question_key, plan_json, headline, verified_by,
                                           verified_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(account_id, question_key) DO UPDATE SET plan_json=excluded.plan_json,
                   headline=excluded.headline, verified_by=excluded.verified_by,
                   verified_at=excluded.verified_at, needs_look_at=NULL""",
            (account_id, str(question).strip()[:2000], words, json.dumps(plan or {}), str(headline or "")[:500],
             str(by or "")[:200], _now()))
        row = conn.execute("SELECT id FROM core2_verified WHERE account_id=? AND question_key=?",
                           (account_id, words)).fetchone()
    return int(row["id"])


def list_verified(account_id: str) -> list[dict[str, Any]]:
    """A workspace's verified answers, those to look at first, then the most used."""
    with get_db() as conn:
        rows = conn.execute("""SELECT * FROM core2_verified WHERE account_id=?
                               ORDER BY needs_look_at IS NULL, uses DESC, id DESC""", (account_id,)).fetchall()
    return [_verified_row(r) for r in rows]


def find_verified(account_id: str, question: str) -> dict[str, Any] | None:
    """The verified answer to the same question, in its words however written; None when there is none."""
    from core2.verified import key

    words = key(question)
    if not words:
        return None
    with get_db() as conn:
        row = conn.execute("SELECT * FROM core2_verified WHERE account_id=? AND question_key=?",
                           (account_id, words)).fetchone()
    return _verified_row(row) if row else None


def remove_verified(account_id: str, verified_id: int) -> bool:
    with get_db() as conn:
        cur = conn.execute("DELETE FROM core2_verified WHERE account_id=? AND id=?", (account_id, int(verified_id)))
    return cur.rowcount > 0


def verified_seen(account_id: str, verified_id: int, answered: bool) -> None:
    """A verified answer given again (counted), or set aside because the data no longer answers its plan (it
    needs a look)."""
    with get_db() as conn:
        if answered:
            conn.execute("UPDATE core2_verified SET uses=uses+1, last_used_at=? WHERE account_id=? AND id=?",
                         (_now(), account_id, int(verified_id)))
        else:
            conn.execute("UPDATE core2_verified SET needs_look_at=? WHERE account_id=? AND id=?",
                         (_now(), account_id, int(verified_id)))


def core2_answered(account_id: str, question_id: str) -> dict[str, Any] | None:
    """The answer a question id names in this workspace that an admin may verify -- its words and plan -- or None:
    a refusal, a question back, another workspace's, or one not read from the question's own words alone (a
    follow-up, a chip, a reader's standing condition, a date fixed for "since January")."""
    if not question_id:
        return None
    with get_db() as conn:
        row = conn.execute("""SELECT question, headline, plan_json FROM core2_answer
                              WHERE account_id=? AND question_id=? AND status='answered' AND verifiable=1
                              ORDER BY id DESC LIMIT 1""", (account_id, question_id)).fetchone()
    if not row:
        return None
    try:
        plan = json.loads(row["plan_json"] or "{}")
    except (TypeError, ValueError):
        return None
    return {"question": row["question"], "headline": row["headline"] or "", "plan": plan} \
        if isinstance(plan, dict) and plan else None


def latest_answer_to(account_id: str, question: str, *, scan: int = 2000) -> str:
    """The question id of the latest answer to the same question (its words however written), or "": for a
    verified answer the data no longer answers, the fresh reading an admin can verify in its place."""
    from core2.verified import key

    words = key(question)
    if not words:
        return ""
    with get_db() as conn:
        rows = conn.execute("""SELECT question, question_id FROM core2_answer
                               WHERE account_id=? AND status='answered' AND verifiable=1 AND question_id<>''
                               ORDER BY id DESC LIMIT ?""", (account_id, int(scan))).fetchall()
    return next((r["question_id"] for r in rows if key(r["question"]) == words), "")


def verified_candidates(account_id: str, *, days: int = 30, least: int = 2, most: int = 10) -> list[dict[str, Any]]:
    """The questions answered most in the last ``days`` (``least`` times or more) that nobody has verified: each
    with how often, and the latest answer to verify it from (its id and what it said)."""
    from core2.verified import key

    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        rows = conn.execute("""SELECT question, question_id, headline FROM core2_answer
                               WHERE account_id=? AND status='answered' AND verifiable=1 AND created_at>=?
                               AND question_id<>'' ORDER BY id DESC LIMIT 5000""", (account_id, since)).fetchall()
        kept = {r["question_key"] for r in conn.execute(
            "SELECT question_key FROM core2_verified WHERE account_id=?", (account_id,)).fetchall()}
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        words = key(row["question"])
        if not words or words in kept:
            continue
        entry = seen.setdefault(words, {"question": row["question"], "question_id": row["question_id"],
                                        "headline": row["headline"] or "", "times": 0})
        entry["times"] += 1
    offered = [e for e in seen.values() if e["times"] >= least]
    return sorted(offered, key=lambda e: -e["times"])[:most]


# ── Precomputed copies (core2/precompute.py) ────────────────────────────────

def _copy_row(row: Any) -> dict[str, Any]:
    item = dict(row)
    try:
        item["fields"] = json.loads(item.pop("fields_json") or "[]")
    except ValueError:
        item["fields"] = []
    return item


def add_copy(account_id: str, db_config_id: int, *, metric: str, fields: list[str], date_role: str = "",
             grain: str = "day", every_hours: int = 6, by: str = "") -> int:
    """Keep a copy of ``metric`` by ``fields`` and by ``grain``, made again every ``every_hours``; its id."""
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO core2_copies (account_id, db_config_id, metric, fields_json, date_role, grain, every_hours,
                                         added_by, added_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (account_id, int(db_config_id or 0), metric, json.dumps(list(fields)), date_role or "", grain,
             int(every_hours), str(by or "")[:200], _now()))
        return int(cur.lastrowid)


def list_copies(account_id: str, db_config_id: int | None = None) -> list[dict[str, Any]]:
    """A workspace's copies (of its database, when given), the newest first."""
    with get_db() as conn:
        if db_config_id is None:
            rows = conn.execute("SELECT * FROM core2_copies WHERE account_id=? ORDER BY id DESC",
                                (account_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM core2_copies WHERE account_id=? AND db_config_id=? ORDER BY id DESC",
                                (account_id, int(db_config_id or 0))).fetchall()
    return [_copy_row(r) for r in rows]


def all_copies() -> list[dict[str, Any]]:
    """Every workspace's copies (for the scheduler)."""
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM core2_copies ORDER BY id").fetchall()
    return [_copy_row(r) for r in rows]


def get_copy(account_id: str, copy_id: int) -> dict[str, Any] | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM core2_copies WHERE account_id=? AND id=?",
                           (account_id, int(copy_id))).fetchone()
    return _copy_row(row) if row else None


def remove_copy(account_id: str, copy_id: int) -> dict[str, Any] | None:
    """The copy taken out (its row, so its file can be deleted), or None when there was none."""
    found = get_copy(account_id, copy_id)
    if found is None:
        return None
    with get_db() as conn:
        conn.execute("DELETE FROM core2_copies WHERE account_id=? AND id=?", (account_id, int(copy_id)))
    return found


def set_copy_schedule(account_id: str, copy_id: int, every_hours: int) -> bool:
    with get_db() as conn:
        cur = conn.execute("UPDATE core2_copies SET every_hours=? WHERE account_id=? AND id=?",
                           (int(every_hours), account_id, int(copy_id)))
    return cur.rowcount > 0


def copy_making(account_id: str, copy_id: int, *, stale_hours: float = 2.0) -> bool:
    """Mark the copy as being made by this process; False when a making of it started here is under way (one
    started more than ``stale_hours`` ago, or by a process since stopped, is taken over)."""
    stale = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=stale_hours)).strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        cur = conn.execute(
            """UPDATE core2_copies SET status='making', started_at=?, runner=?
               WHERE account_id=? AND id=? AND (status!='making' OR runner!=? OR started_at IS NULL
                                                OR started_at<?)""",
            (_now(), core2_runner(), account_id, int(copy_id), core2_runner(), stale))
    return cur.rowcount > 0


def copy_made(account_id: str, copy_id: int, *, rows: int, took_ms: int, size_bytes: int, file: str,
              fingerprint: str, model_version: int) -> str:
    """The copy is ready; the file of its last making (to delete), or ""."""
    before = get_copy(account_id, copy_id) or {}
    now = _now()
    with get_db() as conn:
        conn.execute(
            """UPDATE core2_copies SET status='ready', made_at=?, tried_at=?, rows_kept=?, took_ms=?, size_bytes=?,
                   file=?, fingerprint=?, model_version=?, error='', runner=''
               WHERE account_id=? AND id=?""",
            (now, now, int(rows), int(took_ms), int(size_bytes), file, fingerprint, int(model_version),
             account_id, int(copy_id)))
    old = str(before.get("file") or "")
    return old if old and old != file else ""


def copy_failed(account_id: str, copy_id: int, error: str) -> None:
    """The copy could not be made: why, for the admin. A copy made before stays as it was (used while fresh)."""
    with get_db() as conn:
        conn.execute(
            """UPDATE core2_copies SET status=CASE WHEN file!='' THEN 'ready' ELSE 'failed' END, error=?, tried_at=?,
                   runner=''
               WHERE account_id=? AND id=?""", (str(error or "")[:500], _now(), account_id, int(copy_id)))


def copy_seen(account_id: str, copy_id: int) -> None:
    """A query answered from the copy (counted on the admin page)."""
    with get_db() as conn:
        conn.execute("UPDATE core2_copies SET served=served+1, last_served_at=? WHERE account_id=? AND id=?",
                     (_now(), account_id, int(copy_id)))
