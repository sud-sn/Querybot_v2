"""Persistence helpers for golden-question evaluation runs."""

from __future__ import annotations

import json
from typing import Any

from store.db import get_db


def save_eval_run(
    *,
    account_id: str,
    schema_name: str = "",
    case_file: str = "",
    total_cases: int = 0,
    passed_cases: int = 0,
    avg_score: float = 0.0,
    status: str = "completed",
    report_path: str = "",
) -> int:
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO eval_run
                (account_id, schema_name, case_file, total_cases, passed_cases,
                 avg_score, status, report_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                account_id, schema_name or "", case_file or "", int(total_cases or 0),
                int(passed_cases or 0), float(avg_score or 0.0), status or "completed",
                report_path or "",
            ),
        )
        return int(cur.lastrowid)


def save_eval_case_result(
    eval_run_id: int,
    *,
    case_id: str,
    question: str,
    score: float,
    passed: bool,
    generated_sql: str = "",
    validation_status: str = "",
    validation_error: str = "",
    execution_status: str = "",
    row_count: int = 0,
    failures: list[str] | None = None,
) -> int:
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO eval_case_result
                (eval_run_id, case_id, question, score, passed, generated_sql,
                 validation_status, validation_error, execution_status, row_count,
                 failures_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(eval_run_id), case_id or "", question or "", float(score or 0.0),
                1 if passed else 0, generated_sql or "", validation_status or "",
                validation_error or "", execution_status or "", int(row_count or 0),
                json.dumps(failures or [], ensure_ascii=True),
            ),
        )
        return int(cur.lastrowid)


def list_eval_runs(account_id: str, limit: int = 20) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM eval_run
             WHERE account_id=?
             ORDER BY created_at DESC, id DESC
             LIMIT ?
            """,
            (account_id, int(limit)),
        ).fetchall()
    return [dict(r) for r in rows]


def get_eval_run(eval_run_id: int) -> dict | None:
    with get_db() as conn:
        run = conn.execute("SELECT * FROM eval_run WHERE id=?", (int(eval_run_id),)).fetchone()
        if not run:
            return None
        cases = conn.execute(
            "SELECT * FROM eval_case_result WHERE eval_run_id=? ORDER BY id",
            (int(eval_run_id),),
        ).fetchall()
    result = dict(run)
    result["cases"] = [dict(c) for c in cases]
    for c in result["cases"]:
        try:
            c["failures"] = json.loads(c.get("failures_json") or "[]")
        except Exception:
            c["failures"] = []
    return result


def latest_eval_run(account_id: str) -> dict | None:
    rows = list_eval_runs(account_id, limit=1)
    return rows[0] if rows else None
