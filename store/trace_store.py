"""Persistence helpers for answer-level observability traces."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from store.db import get_db


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, default=str)
    except Exception:
        return json.dumps(str(value), ensure_ascii=True)


def _snippet_refs(chunks: list[str] | None) -> list[dict]:
    refs: list[dict] = []
    for idx, chunk in enumerate(chunks or [], 1):
        text = str(chunk or "")
        refs.append({
            "rank": idx,
            "snippet_hash": hashlib.sha256(text[:2000].encode("utf-8", "ignore")).hexdigest()[:16],
            "preview": " ".join(text.split())[:180],
        })
    return refs


def create_answer_trace(
    *,
    account_id: str,
    question_id: str,
    question_text: str,
    portal_user_id: int | None = None,
    platform_user_id: str = "",
    session_id: str = "",
    parent_question_id: str = "",
    request_source: str = "",
    route: str = "",
    selected_schema: str = "",
    allowed_tables_snapshot: Any = None,
) -> int:
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO answer_trace
                (account_id, portal_user_id, platform_user_id, session_id,
                 question_id, parent_question_id, question_text_sanitized,
                 request_source, route, selected_schema, allowed_tables_snapshot)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                account_id, portal_user_id, platform_user_id or "", session_id or "",
                question_id or "", parent_question_id or "", question_text or "",
                request_source or "", route or "", selected_schema or "",
                _json(allowed_tables_snapshot or []),
            ),
        )
        return int(cur.lastrowid)


def update_answer_trace(trace_id: int | None, **fields: Any) -> None:
    if not trace_id or not fields:
        return
    allowed = {
        "route", "selected_schema", "allowed_tables_snapshot", "retrieved_kb_chunk_ids",
        "retrieved_kb_scores", "llm_provider", "llm_model", "prompt_tokens",
        "completion_tokens", "generated_sql", "sql_validation_status",
        "sql_validation_error", "db_type", "query_row_count", "query_duration_ms",
        "answer_type", "final_answer_summary", "error_message", "status",
        "result_rows", "policy_version_at_query", "contract_version",
    }
    assignments, params = [], []
    for key, value in fields.items():
        if key not in allowed:
            continue
        if key in {
            "allowed_tables_snapshot", "retrieved_kb_chunk_ids",
            "retrieved_kb_scores", "result_rows",
        }:
            value = _json(value)
        assignments.append(f"{key}=?")
        params.append(value)
    if not assignments:
        return
    params.append(trace_id)
    with get_db() as conn:
        conn.execute(
            f"UPDATE answer_trace SET {', '.join(assignments)} WHERE id=?",
            params,
        )


def log_answer_trace_step(
    trace_id: int | None,
    *,
    step_name: str,
    input_summary: Any = "",
    output_summary: Any = "",
    duration_ms: int = 0,
    status: str = "success",
    metadata: Any = None,
) -> None:
    if not trace_id:
        return
    with get_db() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(step_order), 0) + 1 FROM answer_trace_step WHERE trace_id=?",
            (trace_id,),
        ).fetchone()
        step_order = int(row[0] or 1)
        conn.execute(
            """
            INSERT INTO answer_trace_step
                (trace_id, step_order, step_name, input_summary, output_summary,
                 duration_ms, status, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace_id, step_order, step_name,
                input_summary if isinstance(input_summary, str) else _json(input_summary),
                output_summary if isinstance(output_summary, str) else _json(output_summary),
                int(duration_ms or 0), status or "success", _json(metadata or {}),
            ),
        )


def finish_answer_trace(
    trace_id: int | None,
    *,
    status: str,
    answer_type: str = "",
    final_answer_summary: str = "",
    error_message: str = "",
    row_count: int | None = None,
    duration_ms: int | None = None,
) -> None:
    fields: dict[str, Any] = {"status": status}
    if answer_type:
        fields["answer_type"] = answer_type
    if final_answer_summary:
        fields["final_answer_summary"] = final_answer_summary[:500]
    if error_message:
        fields["error_message"] = error_message[:1000]
    if row_count is not None:
        fields["query_row_count"] = int(row_count)
    if duration_ms is not None:
        fields["query_duration_ms"] = int(duration_ms)
    update_answer_trace(trace_id, **fields)


def kb_chunk_refs(chunks: list[str] | None) -> list[dict]:
    return _snippet_refs(chunks)


# Cross-encoder sigmoid scores below this are "borderline" relevance — the
# doc was kept, but it matched the question weakly. High borderline rates
# on a heavily-retrieved table = the KB doc's wording doesn't match how
# users actually ask, which is exactly what an admin edit fixes.
_KB_BORDERLINE_SCORE = 0.3

# answer_type values that count as "the question failed after this table
# was retrieved" for doc-quality attribution.
_FAILED_ANSWER_TYPES = {"error", "timeout", "policy_denied", "cannot_generate"}


def get_kb_doc_quality(
    account_id: str,
    days: int = 30,
    min_retrievals: int = 2,
    limit: int = 15,
) -> list[dict]:
    """
    Rank KB table docs by "most retrieved, least answerable" so admins edit
    the docs that matter first.

    Aggregates answer_trace.retrieved_kb_scores (per-table retrieval
    telemetry) against each trace's outcome. Per table:
      retrieved     — questions where this table entered the candidate set
      used_in_sql   — of those, how often the final SQL actually used it
      failed        — of those, how often the question errored out
      floored       — dropped by the relevance floor despite being retrieved
      borderline    — kept, but best score below the borderline threshold
      avg_score     — mean best cross-encoder score across retrievals

    attention_score = retrieved x max(failure rate, unused rate, borderline
    rate) — a table that's pulled into many prompts but rarely ends up in
    working SQL (or matches only weakly) floats to the top. Sorted
    descending; only tables above min_retrievals are ranked.
    """
    import re
    from datetime import datetime, timedelta, timezone

    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=int(days))
    ).strftime("%Y-%m-%d %H:%M:%S")

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT retrieved_kb_scores, generated_sql, status, answer_type
              FROM answer_trace
             WHERE account_id=? AND created_at >= ?
               AND retrieved_kb_scores NOT IN ('', '[]')
            """,
            (account_id, cutoff),
        ).fetchall()

    agg: dict[str, dict] = {}
    for row in rows:
        try:
            stats = json.loads(row["retrieved_kb_scores"]) or []
        except Exception:
            continue
        if not isinstance(stats, list):
            continue
        sql_upper = (row["generated_sql"] or "").upper()
        failed = (
            (row["status"] or "") == "error"
            or (row["answer_type"] or "") in _FAILED_ANSWER_TYPES
        )
        for s in stats:
            if not isinstance(s, dict):
                continue
            fqn = str(s.get("fqn") or "").upper()
            if not fqn:
                continue
            e = agg.setdefault(fqn, {
                "fqn": fqn, "retrieved": 0, "used_in_sql": 0,
                "failed": 0, "floored": 0, "borderline": 0,
                "_score_sum": 0.0, "_score_n": 0,
            })
            e["retrieved"] += 1
            kept = bool(s.get("kept", True))
            if not kept:
                e["floored"] += 1
            score = s.get("best_score")
            if isinstance(score, (int, float)):
                e["_score_sum"] += float(score)
                e["_score_n"] += 1
                if kept and float(score) < _KB_BORDERLINE_SCORE:
                    e["borderline"] += 1
            bare = fqn.split(".")[-1]
            if bare and re.search(rf"\b{re.escape(bare)}\b", sql_upper):
                e["used_in_sql"] += 1
            if failed:
                e["failed"] += 1

    ranked: list[dict] = []
    for e in agg.values():
        n = e["retrieved"]
        if n < min_retrievals:
            continue
        e["avg_score"] = (
            round(e["_score_sum"] / e["_score_n"], 3) if e["_score_n"] else None
        )
        e["used_rate"] = round(e["used_in_sql"] / n, 2)
        e["failure_rate"] = round(e["failed"] / n, 2)
        e["attention_score"] = round(
            n * max(e["failure_rate"], 1 - e["used_rate"], e["borderline"] / n), 1
        )
        del e["_score_sum"], e["_score_n"]
        ranked.append(e)

    ranked.sort(key=lambda x: (-x["attention_score"], -x["retrieved"]))
    return ranked[:limit]


def store_protected_result_rows(
    account_id: str,
    question_id: str,
    rows: list[dict],
    *,
    policy_version: int = 0,
) -> None:
    """Persist only post-policy rows for later authorized export."""
    with get_db() as conn:
        conn.execute(
            """
            UPDATE answer_trace
            SET result_rows=?, policy_version_at_query=?
            WHERE account_id=? AND question_id=?
            """,
            (_json(rows), int(policy_version), account_id, question_id),
        )


def get_answer_trace(trace_id: int) -> dict | None:
    with get_db() as conn:
        trace = conn.execute("SELECT * FROM answer_trace WHERE id=?", (trace_id,)).fetchone()
        if not trace:
            return None
        steps = conn.execute(
            "SELECT * FROM answer_trace_step WHERE trace_id=? ORDER BY step_order",
            (trace_id,),
        ).fetchall()
    result = dict(trace)
    result["steps"] = [dict(s) for s in steps]
    return result


def get_answer_trace_by_question_id(account_id: str, question_id: str) -> dict | None:
    """Resolve a query-log row's question_id to its answer_trace, so the admin
    audit table can link straight to the query-duration breakdown. Both tables
    already carry question_id — no schema change needed. Picks the most recent
    trace when duplicates exist (e.g. a retried/re-asked question)."""
    if not question_id:
        return None
    with get_db() as conn:
        row = conn.execute(
            "SELECT id FROM answer_trace WHERE account_id=? AND question_id=? "
            "ORDER BY id DESC LIMIT 1",
            (account_id, question_id),
        ).fetchone()
    if not row:
        return None
    return get_answer_trace(int(row[0]))


def list_answer_traces(account_id: str, limit: int = 50) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM answer_trace
             WHERE account_id=?
             ORDER BY created_at DESC, id DESC
             LIMIT ?
            """,
            (account_id, int(limit)),
        ).fetchall()
    return [dict(r) for r in rows]
