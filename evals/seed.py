"""Auto-seed per-client golden-question suites from real usage.

Most clients have no golden_questions.yaml, so the eval gate protects
nothing. This module harvests the client's most-asked SUCCESSFUL questions
from answer_trace (validated SQL included, so the cases score offline —
no LLM call needed on eval runs) and writes/merges them into
evals/clients/<account_id>/<schema>/golden_questions.yaml.

Merge rules: existing cases are never modified or removed — hand-edited
suites stay authoritative; new auto cases are appended by question hash.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

log = logging.getLogger("querybot.evals.seed")

_TABLE_REF_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+([A-Za-z_\[\]\"][\w.$\[\]\"]*)",
    re.IGNORECASE,
)

_MAX_CASES_PER_SCHEMA = 40


def _norm_question(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())


def _case_id(question: str) -> str:
    return "auto_" + hashlib.md5(_norm_question(question).encode("utf-8")).hexdigest()[:10]


def extract_tables_from_sql(sql: str) -> list[str]:
    """FROM/JOIN table references, cleaned of quoting/brackets and aliases.
    Substring assertions only — no need for full sqlglot parsing here."""
    tables: list[str] = []
    for raw in _TABLE_REF_RE.findall(sql or ""):
        name = raw.strip().strip('"').replace("[", "").replace("]", "")
        # Skip derived tables / CTE openers the regex may catch
        if not name or name.upper() in {"SELECT", "("}:
            continue
        upper = name.upper()
        if upper not in tables:
            tables.append(upper)
    return tables


def harvest_golden_cases(account_id: str, top_n: int = 20) -> dict[str, list[dict]]:
    """Return {schema_name: [case, ...]} from the client's successful answers.

    Ranked by how often the (normalized) question was asked — the questions
    users actually rely on are the ones a regression hurts most.
    """
    from store.db import get_db

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT question_text_sanitized AS question,
                   generated_sql,
                   COALESCE(selected_schema, '') AS schema_name,
                   COUNT(*) AS freq,
                   MAX(created_at) AS last_asked
              FROM answer_trace
             WHERE account_id = ?
               AND status = 'success'
               AND COALESCE(generated_sql, '') != ''
               AND COALESCE(query_row_count, 0) > 0
               AND route IN ('normal_sql', 'metric_registry')
             GROUP BY LOWER(TRIM(question_text_sanitized)), schema_name
             ORDER BY freq DESC, last_asked DESC
             LIMIT ?
            """,
            (account_id, int(top_n) * 3),  # headroom for dedupe/skips
        ).fetchall()

    by_schema: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for r in rows:
        row = dict(r)
        question = (row.get("question") or "").strip()
        sql = (row.get("generated_sql") or "").strip()
        if not question or not sql:
            continue
        key = _norm_question(question)
        if key in seen:
            continue
        seen.add(key)
        schema = (row.get("schema_name") or "").strip() or "default"
        tables = extract_tables_from_sql(sql)
        case = {
            "id": _case_id(question),
            "question": question,
            # Offline scoring: the validated SQL that actually answered this
            # question in production. Eval runs re-validate + re-assert it
            # against the CURRENT semantic state without any LLM call.
            "generated_sql": sql,
            "expected_tables": tables,
            "min_score": 0.85,
        }
        bucket = by_schema.setdefault(schema, [])
        if len(bucket) < top_n:
            bucket.append(case)
    return by_schema


def seed_golden_suite(account_id: str, top_n: int = 20) -> dict:
    """Harvest and merge into golden_questions.yaml per schema.

    Returns {"files": [...], "added": int, "skipped_existing": int}.
    """
    try:
        import yaml
    except Exception as exc:   # pragma: no cover
        raise RuntimeError("Golden suite seeding requires PyYAML.") from exc

    harvested = harvest_golden_cases(account_id, top_n=top_n)
    summary = {"files": [], "added": 0, "skipped_existing": 0}

    for schema, new_cases in harvested.items():
        target = Path("evals") / "clients" / account_id / schema / "golden_questions.yaml"
        existing_cases: list[dict] = []
        if target.exists():
            try:
                data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
                existing_cases = list(
                    data.get("cases", data) if isinstance(data, dict) else data
                ) or []
            except Exception as exc:
                log.warning("Could not parse %s — leaving it untouched: %s", target, exc)
                continue

        existing_ids = {str(c.get("id") or "") for c in existing_cases}
        existing_questions = {_norm_question(str(c.get("question") or "")) for c in existing_cases}

        added_here = 0
        for case in new_cases:
            if len(existing_cases) + added_here >= _MAX_CASES_PER_SCHEMA:
                break
            if case["id"] in existing_ids or _norm_question(case["question"]) in existing_questions:
                summary["skipped_existing"] += 1
                continue
            existing_cases.append(case)
            added_here += 1

        if not added_here:
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump({"cases": existing_cases}, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        summary["files"].append(str(target))
        summary["added"] += added_here
        log.info("Golden suite seeded: %s (+%d cases)", target, added_here)

    return summary
