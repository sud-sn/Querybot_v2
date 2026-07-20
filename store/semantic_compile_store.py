"""Persistence for governed semantic compilation and conflict review."""

from __future__ import annotations

import json
import uuid
from typing import Any

from store.db import get_db


def _row(row) -> dict[str, Any]:
    return dict(row) if row else {}


def get_semantic_compiler_state(account_id: str) -> dict[str, Any]:
    with get_db() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO semantic_compiler_state(account_id)
               VALUES (?)""",
            (account_id,),
        )
        return _row(conn.execute(
            "SELECT * FROM semantic_compiler_state WHERE account_id = ?",
            (account_id,),
        ).fetchone())


def set_semantic_compiler_mode(
    account_id: str, mode: str, *, publish_mode: str | None = None,
) -> dict[str, Any]:
    if mode not in {"off", "shadow", "enforce"}:
        raise ValueError("mode must be off, shadow, or enforce")
    get_semantic_compiler_state(account_id)
    with get_db() as conn:
        if publish_mode is None:
            conn.execute(
                """UPDATE semantic_compiler_state
                   SET mode = ?, updated_at = datetime('now') WHERE account_id = ?""",
                (mode, account_id),
            )
        else:
            if publish_mode not in {"auto_publish_clean", "explicit_publish"}:
                raise ValueError("invalid publish_mode")
            conn.execute(
                """UPDATE semantic_compiler_state
                   SET mode = ?, publish_mode = ?, updated_at = datetime('now')
                   WHERE account_id = ?""",
                (mode, publish_mode, account_id),
            )
    return get_semantic_compiler_state(account_id)


def create_semantic_compile_run(
    account_id: str, *, trigger: str, initiated_by: str, mode: str,
    base_version: str,
) -> str:
    run_id = uuid.uuid4().hex
    with get_db() as conn:
        conn.execute(
            """INSERT INTO semantic_compile_run(
                   run_id, account_id, trigger_label, initiated_by, mode,
                   base_version, status
               ) VALUES (?, ?, ?, ?, ?, ?, 'running')""",
            (run_id, account_id, trigger, initiated_by, mode, base_version),
        )
    return run_id


def finish_semantic_compile_run(
    run_id: str, *, status: str, draft_version: str = "",
    published_version: str = "", error_count: int = 0,
    warning_count: int = 0, info_count: int = 0,
    source_fingerprints: dict | None = None, message: str = "",
) -> None:
    with get_db() as conn:
        conn.execute(
            """UPDATE semantic_compile_run
               SET status = ?, draft_version = ?, published_version = ?,
                   error_count = ?, warning_count = ?, info_count = ?,
                   source_fingerprints_json = ?, message = ?,
                   completed_at = datetime('now')
               WHERE run_id = ?""",
            (
                status, draft_version, published_version, error_count,
                warning_count, info_count,
                json.dumps(source_fingerprints or {}, sort_keys=True), message, run_id,
            ),
        )


def save_semantic_contract_version(
    account_id: str, contract: dict, *, status: str = "draft",
    compile_run_id: str = "", created_by: str = "",
) -> str:
    version = str((contract.get("meta") or {}).get("contract_version") or "")
    if not version:
        raise ValueError("contract has no contract_version")
    sources = (contract.get("meta") or {}).get("sources") or {}
    with get_db() as conn:
        conn.execute(
            """INSERT INTO semantic_contract_version(
                   account_id, version, status, contract_json,
                   source_fingerprints_json, compile_run_id, created_by
               ) VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(account_id, version) DO UPDATE SET
                   contract_json = excluded.contract_json,
                   source_fingerprints_json = excluded.source_fingerprints_json,
                   compile_run_id = excluded.compile_run_id,
                   created_by = excluded.created_by""",
            (
                account_id, version, status,
                json.dumps(contract, sort_keys=True),
                json.dumps(sources, sort_keys=True), compile_run_id, created_by,
            ),
        )
    return version


def publish_semantic_contract_version(
    account_id: str, version: str, *, baseline: bool = False,
) -> None:
    get_semantic_compiler_state(account_id)
    with get_db() as conn:
        conn.execute(
            """UPDATE semantic_contract_version
               SET status = 'superseded'
               WHERE account_id = ? AND status = 'active' AND version <> ?""",
            (account_id, version),
        )
        conn.execute(
            """UPDATE semantic_contract_version
               SET status = 'active', published_at = datetime('now')
               WHERE account_id = ? AND version = ?""",
            (account_id, version),
        )
        if baseline:
            conn.execute(
                """UPDATE semantic_compiler_state
                   SET baseline_version = ?, active_version = ?, draft_version = '',
                       updated_at = datetime('now') WHERE account_id = ?""",
                (version, version, account_id),
            )
        else:
            conn.execute(
                """UPDATE semantic_compiler_state
                   SET active_version = ?, draft_version = '',
                       updated_at = datetime('now') WHERE account_id = ?""",
                (version, account_id),
            )


def set_semantic_draft_version(account_id: str, version: str) -> None:
    get_semantic_compiler_state(account_id)
    with get_db() as conn:
        conn.execute(
            """UPDATE semantic_compiler_state
               SET draft_version = ?, updated_at = datetime('now')
               WHERE account_id = ?""",
            (version, account_id),
        )


def save_semantic_conflicts(
    run_id: str, account_id: str, conflicts: list[dict[str, Any]],
) -> None:
    with get_db() as conn:
        for conflict in conflicts:
            key = str(conflict.get("conflict_key") or uuid.uuid4().hex)
            conn.execute(
                """INSERT OR IGNORE INTO semantic_conflict(
                       conflict_id, compile_run_id, account_id, conflict_key,
                       code, severity, object_type, object_id, schema_name,
                       table_name, origin, message, evidence_json, suggestions_json
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    uuid.uuid4().hex, run_id, account_id, key,
                    conflict.get("code") or "semantic_conflict",
                    conflict.get("severity") or "ERROR",
                    conflict.get("object_type") or "",
                    conflict.get("object_id") or "",
                    conflict.get("schema_name") or "",
                    conflict.get("table_name") or "",
                    conflict.get("origin") or "compiler",
                    conflict.get("message") or "Semantic conflict detected.",
                    json.dumps(conflict.get("evidence") or {}, sort_keys=True),
                    json.dumps(conflict.get("suggestions") or [], sort_keys=True),
                ),
            )


def reconcile_semantic_conflicts(
    account_id: str, current_conflict_keys: set[str], *, run_id: str,
) -> None:
    """Resolve open conflicts that disappeared in the newest completed compile."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT conflict_id, conflict_key FROM semantic_conflict
               WHERE account_id = ? AND status = 'open' AND compile_run_id <> ?""",
            (account_id, run_id),
        ).fetchall()
        resolved_ids = [
            row["conflict_id"] for row in rows
            if row["conflict_key"] not in current_conflict_keys
        ]
        for conflict_id in resolved_ids:
            conn.execute(
                """UPDATE semantic_conflict
                   SET status = 'resolved',
                       resolution_note = 'No longer present in the latest compile',
                       resolved_by = 'semantic_compiler',
                       resolved_at = datetime('now')
                   WHERE conflict_id = ?""",
                (conflict_id,),
            )


def list_semantic_conflicts(
    account_id: str, *, status: str = "open", severity: str = "", limit: int = 50,
) -> list[dict[str, Any]]:
    where = "account_id = ?"
    params: list[Any] = [account_id]
    if status:
        where += " AND status = ?"
        params.append(status)
    if severity:
        where += " AND severity = ?"
        params.append(severity)
    params.append(max(1, min(int(limit), 500)))
    with get_db() as conn:
        rows = conn.execute(
            f"""SELECT * FROM semantic_conflict WHERE {where}
                ORDER BY
                    CASE severity WHEN 'ERROR' THEN 0 WHEN 'WARNING' THEN 1
                                  WHEN 'INFO' THEN 2 ELSE 3 END,
                    created_at DESC
                LIMIT ?""",
            tuple(params),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        for source, target, default in (
            ("evidence_json", "evidence", {}),
            ("suggestions_json", "suggestions", []),
        ):
            try:
                item[target] = json.loads(item.get(source) or "")
            except Exception:
                item[target] = default
        result.append(item)
    return result


def get_semantic_conflict(account_id: str, conflict_id: str) -> dict[str, Any]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM semantic_conflict WHERE account_id = ? AND conflict_id = ?",
            (account_id, conflict_id),
        ).fetchone()
    item = _row(row)
    if not item:
        return {}
    for source, target, default in (
        ("evidence_json", "evidence", {}),
        ("suggestions_json", "suggestions", []),
    ):
        try:
            item[target] = json.loads(item.get(source) or "")
        except Exception:
            item[target] = default
    return item


_CONFLICT_ACTIONS = {
    "resolve": "resolved",
    "acknowledge": "acknowledged",
    "dismiss": "dismissed",
}


def resolve_semantic_conflict(
    account_id: str, conflict_id: str, *, action: str, note: str = "", resolved_by: str = "",
) -> dict[str, Any]:
    """Manually transition one conflict's review status (Sprint 5's conflict
    inbox). Distinct from reconcile_semantic_conflicts(), which the compiler
    itself calls when a conflict's underlying cause disappears on its own —
    this is the admin saying "I looked at this," not the compiler detecting
    it's moot. Scoped to account_id so one tenant can never touch another's
    conflict by guessing an id."""
    status = _CONFLICT_ACTIONS.get(action)
    if not status:
        raise ValueError(f"action must be one of {sorted(_CONFLICT_ACTIONS)}")
    with get_db() as conn:
        cur = conn.execute(
            """UPDATE semantic_conflict
               SET status = ?, resolution_note = ?, resolved_by = ?,
                   resolved_at = datetime('now')
               WHERE account_id = ? AND conflict_id = ? AND status = 'open'""",
            (status, note, resolved_by, account_id, conflict_id),
        )
        if cur.rowcount == 0:
            existing = conn.execute(
                "SELECT 1 FROM semantic_conflict WHERE account_id = ? AND conflict_id = ?",
                (account_id, conflict_id),
            ).fetchone()
            if not existing:
                raise LookupError("Conflict not found for this account.")
            raise ValueError("This conflict is no longer open — it may already have been resolved.")
    return get_semantic_conflict(account_id, conflict_id)


def list_semantic_contract_versions(account_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
    """Version history for Sprint 5's diff/rollback UI. Excludes contract_json
    (large) - callers needing the full body call get_semantic_contract_version."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT account_id, version, status, compile_run_id, created_by,
                      created_at, published_at
               FROM semantic_contract_version
               WHERE account_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (account_id, max(1, min(int(limit), 200))),
        ).fetchall()
    return [_row(row) for row in rows]


def get_semantic_contract_version(account_id: str, version: str) -> dict[str, Any]:
    """Full stored contract body for one version, for Sprint 5's diff view
    and for publish_semantic_contract_version-based rollback."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT * FROM semantic_contract_version
               WHERE account_id = ? AND version = ?""",
            (account_id, version),
        ).fetchone()
    item = _row(row)
    if not item:
        return {}
    try:
        item["contract"] = json.loads(item.get("contract_json") or "{}")
    except Exception:
        item["contract"] = {}
    return item


def get_semantic_compiler_summary(account_id: str) -> dict[str, Any]:
    state = get_semantic_compiler_state(account_id)
    with get_db() as conn:
        latest = _row(conn.execute(
            """SELECT * FROM semantic_compile_run WHERE account_id = ?
               ORDER BY started_at DESC LIMIT 1""",
            (account_id,),
        ).fetchone())
        counts = _row(conn.execute(
            """SELECT
                   SUM(CASE WHEN status = 'open' AND severity = 'ERROR' THEN 1 ELSE 0 END) AS errors,
                   SUM(CASE WHEN status = 'open' AND severity = 'WARNING' THEN 1 ELSE 0 END) AS warnings,
                   SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) AS open_total
               FROM semantic_conflict WHERE account_id = ?""",
            (account_id,),
        ).fetchone())
    return {
        "state": state,
        "latest_run": latest,
        "counts": {key: int(value or 0) for key, value in counts.items()},
        "conflicts": list_semantic_conflicts(account_id, limit=8),
    }
