"""
core/pipeline_trace.py
──────────────────────
Observability helpers extracted from main.py.

Covers:
  • _log_q                    — write a query_log row
  • _trace_create/update/step/finish — answer_trace lifecycle wrappers
  • _create_learning_candidate — score + enqueue a learning candidate
  • _create_pin_token          — store pin data and return a short token
"""

from __future__ import annotations

import logging
import secrets

import store
from store.db import get_db

log = logging.getLogger("querybot")


# ── Query log ─────────────────────────────────────────────────────────────────

def _log_q(account_id, question, sql, rows, success, error,
           provider, model, tok_in, tok_out, dur_ms,
           portal_user_id=None, zoom_user_id="",
           question_id="", parent_question_id=""):
    store.log_query(
        account_id=account_id, question=question, sql_generated=sql,
        row_count=rows, success=success, error_msg=error,
        llm_provider=provider, llm_model=model,
        tokens_in=tok_in, tokens_out=tok_out, duration_ms=dur_ms,
        portal_user_id=portal_user_id, zoom_user_id=zoom_user_id or "",
        question_id=question_id or "", parent_question_id=parent_question_id or "",
    )


# ── Answer trace ──────────────────────────────────────────────────────────────

def _trace_create(**kwargs) -> int | None:
    try:
        if "question" in kwargs and "question_text" not in kwargs:
            kwargs["question_text"] = kwargs.pop("question")
        return store.create_answer_trace(**kwargs)
    except Exception as exc:
        log.debug("answer trace create failed: %s", exc)
        return None


def _trace_update(trace_id: int | None, **fields) -> None:
    try:
        store.update_answer_trace(trace_id, **fields)
    except Exception as exc:
        log.debug("answer trace update failed: %s", exc)


def _trace_step(trace_id: int | None, step_name: str, **kwargs) -> None:
    try:
        store.log_answer_trace_step(trace_id, step_name=step_name, **kwargs)
    except Exception as exc:
        log.debug("answer trace step failed: %s", exc)


def _trace_finish(trace_id: int | None, **kwargs) -> None:
    try:
        store.finish_answer_trace(trace_id, **kwargs)
    except Exception as exc:
        log.debug("answer trace finish failed: %s", exc)


# ── Query duration breakdown (Snowflake-style phase bars) ───────────────────

# step_name -> display bucket. Steps not listed here (receive_question,
# resolve_user_permissions, route, semantic_model_context, semantic_field_plan,
# semantic_model_plan, value_resolution, regulated_*) are near-instant
# deterministic Python with no LLM/DB round-trip — they're not bucketed
# individually, they fall into the computed "Other" remainder instead.
_DURATION_BUCKETS: dict[str, str] = {
    "retrieve_kb": "KB retrieval",
    "retrieve_examples": "KB retrieval",
    "llm_generate_sql": "SQL generation",
    "validate_sql": "Validation",
    "field_plan_repair": "Validation",
    "execute_sql": "Execution",
}
_BUCKET_ORDER = ["KB retrieval", "SQL generation", "Validation", "Execution"]


def compute_duration_breakdown(steps: list[dict], total_ms: int) -> list[dict]:
    """Roll up per-step durations into the four named phase buckets plus an
    'Other' remainder, for the query-duration bar panel on the Traces page.

    Sums by step_name (not call site), so a retried step — a second
    llm_generate_sql/validate_sql/execute_sql row — naturally adds onto the
    same bucket rather than needing special-case retry handling. Computed
    fresh from the raw steps every time; nothing is persisted separately.
    """
    totals = {bucket: 0 for bucket in _BUCKET_ORDER}
    for step in steps or []:
        bucket = _DURATION_BUCKETS.get(step.get("step_name") or "")
        if bucket:
            totals[bucket] += int(step.get("duration_ms") or 0)

    total_ms = max(0, int(total_ms or 0))
    bucketed_ms = sum(totals.values())
    other_ms = max(0, total_ms - bucketed_ms)

    # Percent of total for bar widths; guard divide-by-zero when nothing timed.
    denom = total_ms if total_ms > 0 else (bucketed_ms or 1)
    rows = [
        {"label": label, "duration_ms": totals[label], "pct": round(totals[label] / denom * 100, 1)}
        for label in _BUCKET_ORDER
    ]
    rows.append({"label": "Other", "duration_ms": other_ms, "pct": round(other_ms / denom * 100, 1)})
    rows.append({"label": "Total", "duration_ms": total_ms, "pct": 100.0})
    return rows


# ── Learning pipeline ─────────────────────────────────────────────────────────

def compute_learning_versions(account_id: str, kb_dir: str = "", schema_dir: str = "") -> dict:
    """
    Cheap content fingerprints of the three things that can make a stored
    learning example stale: the semantic model, the discovered schema, and
    the metric registry. Stamped onto every learning_candidate at creation
    so retrieval can prefer examples born under the CURRENT model over ones
    from before a KB rebuild / metric edit. Best-effort — a missing file or
    DB hiccup yields "" for that fingerprint, never an exception.
    """
    import hashlib
    from pathlib import Path

    def _file_fingerprint(path: Path) -> str:
        try:
            if path.is_file():
                return hashlib.md5(path.read_bytes()).hexdigest()[:12]
        except Exception:
            pass
        return ""

    versions = {"semantic_model_version": "", "schema_version": "", "metric_version": ""}
    try:
        if kb_dir:
            from core.semantic_model import semantic_model_fingerprint
            versions["semantic_model_version"] = semantic_model_fingerprint(kb_dir)
        if schema_dir:
            versions["schema_version"] = _file_fingerprint(Path(schema_dir) / "_schema.json")
    except Exception as exc:
        log.debug("compute_learning_versions file fingerprints failed: %s", exc)
    try:
        with get_db() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM metric_version WHERE account_id=?",
                (account_id,),
            ).fetchone()
        versions["metric_version"] = str(int(row[0] or 0))
    except Exception as exc:
        log.debug("compute_learning_versions metric fingerprint failed: %s", exc)
    return versions


def _create_learning_candidate(
    account_id: str,
    question_id: str,
    question: str,
    sql: str,
    validation_passed: bool,
    had_repair: bool,
    repair_succeeded: bool,
    row_count: int,
    confidence_ctx: dict,
    schema_scope: str = "",
    kb_dir: str = "",
    schema_dir: str = "",
) -> None:
    """
    Score this trace and create a learning_candidate row.

    Called only when client.enable_feedback_collection = 1.
    Invoked directly after the response is sent — latency impact is invisible.
    Never raises — all errors are logged and swallowed so a failure here
    never affects the user-facing response.
    """
    try:
        from core.quality_scorer import score_trace
        from store.learning_store import create_candidate

        # If we reached this point the DB execution succeeded (rows were sent).
        execution_success = True

        # Compliance signals from the confidence context (best-effort, default 1.0)
        metric_compliance  = float(confidence_ctx.get("semantic_score", 100)) / 100.0
        schema_compliance  = 1.0   # ACL was enforced upstream; if we got here, it passed
        eg_compliance      = float(confidence_ctx.get("graph_score", 100)) / 100.0

        score, evidence = score_trace(
            validation_passed        = validation_passed,
            execution_success        = execution_success,
            had_repair               = had_repair,
            repair_succeeded         = repair_succeeded,
            metric_compliance        = min(1.0, max(0.0, metric_compliance)),
            schema_acl_compliance    = schema_compliance,
            entity_graph_compliance  = min(1.0, max(0.0, eg_compliance)),
            row_count                = row_count,
            feedback_delta           = 0,   # no feedback yet at creation time
        )

        versions = compute_learning_versions(account_id, kb_dir=kb_dir, schema_dir=schema_dir)

        create_candidate(
            origin_question_id = question_id,
            account_id         = account_id,
            question_text      = question,
            sql_text           = sql,
            technical_score    = score,
            evidence           = evidence,
            schema_scope       = schema_scope,
            semantic_model_version = versions["semantic_model_version"],
            metric_version         = versions["metric_version"],
            schema_version         = versions["schema_version"],
        )
    except Exception as exc:
        log.debug("_create_learning_candidate failed (non-fatal): %s", exc)


# ── Pin token ─────────────────────────────────────────────────────────────────

def _create_pin_token(
    user_id: int, account_id: str, question: str,
    sql_query: str, chart_type: str, db_config_id: int,
) -> str:
    """
    Store pending pin data server-side and return a short token.
    The token is passed in the URL — SQL never goes through Zoom markdown.
    Token expires after 30 minutes (user must click the pin link promptly).
    """
    token = secrets.token_urlsafe(16)
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pin_token (
                token        TEXT PRIMARY KEY,
                user_id      INTEGER NOT NULL,
                account_id   TEXT NOT NULL,
                question     TEXT NOT NULL,
                sql_query    TEXT NOT NULL,
                chart_type   TEXT NOT NULL,
                db_config_id INTEGER NOT NULL,
                expires_at   TEXT NOT NULL,
                created_at   TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            INSERT INTO pin_token
                (token, user_id, account_id, question, sql_query,
                 chart_type, db_config_id, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', '+30 minutes'))
        """, (token, user_id, account_id, question, sql_query,
              chart_type, db_config_id))
    return token
