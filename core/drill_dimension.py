"""
core/drill_dimension.py

Drill-by-dimension engine for the "Break down by X" chips (Sprint C).

Design principles
─────────────────
• Finding drill candidates is a pure function of the semantic plan and the
  current result column set — no DB or LLM needed.
• SQL rewrite is delegated to the LLM with a deliberately narrow prompt: add
  ONE dimension column + its JOIN.  Everything else must be preserved.
• The response is a full ``assistant_response`` (table result with chips), not
  an ``assistant_analysis`` card, because the user is asking for new data.
• Every failure exits with a clear fallback message — never a Python exception
  propagated to the WebSocket.

Entry point
───────────
  await generate_drill_by_dimension(
      dim_name=...,
      rows=..., question=..., original_sql=...,
      semantic_plan=..., db_cfg=..., known_tables=...,
      provider=..., model=..., api_key=...,
      ...
  )
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger("querybot.drill_dimension")


# ══════════════════════════════════════════════════════════════════════════════
# Candidate lookup (pure, no side effects)
# ══════════════════════════════════════════════════════════════════════════════

def find_drill_candidate(
    dim_name: str,
    semantic_plan: dict,
) -> dict | None:
    """
    Return the dimension metadata for *dim_name* from
    ``semantic_plan["available_dimensions"]``, or ``None`` if not found.

    Matching is case-insensitive on the dimension name.
    """
    target = (dim_name or "").strip().lower()
    for dim in (semantic_plan.get("available_dimensions") or []):
        if (dim.get("name") or "").strip().lower() == target:
            return dim
    return None


# ══════════════════════════════════════════════════════════════════════════════
# LLM prompt builder (pure, no side effects)
# ══════════════════════════════════════════════════════════════════════════════

def build_drill_sql_prompt(
    original_sql: str,
    display_table: str,
    display_col: str,
    source_table: str,
    source_key: str,
    display_key: str,
) -> tuple[str, str]:
    """
    Build the (system, user) prompt pair for adding ONE dimension to a SQL query.

    The LLM's only task is:
      1. Add ``display_col`` to SELECT and GROUP BY.
      2. Add a LEFT JOIN to ``display_table`` ON the given condition (if absent).
      3. Leave all other SELECT columns, WHERE conditions, metrics, and JOINs
         completely unchanged.

    Returns CANNOT_REWRITE if the change cannot be made safely.
    """
    system = (
        "You are a SQL dimension-adder. Your ONLY task is to add one dimension "
        "column to an existing query.\n\n"
        "RULES:\n"
        f"1. Add {display_col} to the SELECT column list.\n"
        f"2. Add {display_col} to the GROUP BY clause.\n"
        f"3. If {display_table} is not already joined: add "
        f"LEFT JOIN {display_table} ON "
        f"{source_table}.{source_key} = {display_table}.{display_key}\n"
        "4. Do NOT change existing SELECT columns, WHERE conditions, "
        "aggregations, metrics, or other JOINs.\n"
        "5. Use a short alias for the new table if the SQL style uses aliases "
        "(e.g. 'd' for a dimension table). Be consistent with existing style.\n"
        "6. Return ONLY the modified SQL — no explanation, no markdown fences.\n"
        "7. If the change cannot be made safely, return exactly: CANNOT_REWRITE"
    )
    user = (
        f"Original SQL:\n{original_sql}\n\n"
        f"Dimension to add:\n"
        f"  Display column : {display_col}\n"
        f"  From table     : {display_table}\n"
        f"  Join condition : {source_table}.{source_key} = {display_table}.{display_key}\n\n"
        "Add this dimension and return the complete modified SQL."
    )
    return system, user


# ══════════════════════════════════════════════════════════════════════════════
# SQL cleanup helper
# ══════════════════════════════════════════════════════════════════════════════

def _clean_sql(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        s = s.rsplit("```", 1)[0].strip()
    return s


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

async def generate_drill_by_dimension(
    *,
    dim_name: str,
    rows: list[dict],
    question: str,
    original_sql: str,
    semantic_plan: dict,
    db_cfg: dict,
    known_tables: set[str] | None = None,
    provider: str,
    model: str,
    api_key: str,
    display_context: dict | None = None,
    **extra_kwargs,
) -> dict:
    """
    Rewrite the original SQL to include a new dimension, execute it, and return
    a full ``assistant_response`` (table result with chips).

    Pipeline
    ────────
    1. Resolve the dimension metadata from the semantic plan
    2. LLM rewrites the SQL to add the display column + JOIN
    3. Validate the new SQL
    4. Execute against the live DB
    5. Build and return a full assistant_response

    All failure paths return a graceful ``assistant_error`` dict — never raise.
    """
    from core.llm import llm_complete
    from core.response_builder import build_assistant_response
    from core.schema import run_query
    from core.validator import validate_sql

    def _fallback(reason: str, suggestion: str = "") -> dict:
        return {
            "type": "assistant_error",
            "action": "drill_dim",
            "title": f"Break down by {dim_name}",
            "content": reason,
            "suggestion": (
                suggestion or
                f"Try asking: \"Show [metric] broken down by {dim_name}\""
            ),
        }

    # ── Step 1: Resolve dimension metadata ──────────────────────────────────
    dim = find_drill_candidate(dim_name, semantic_plan)
    if not dim:
        return _fallback(
            f"Dimension '{dim_name}' not found in the semantic model.",
            f"Try asking: \"Break down by {dim_name}\" in a new question.",
        )

    display_table  = dim["display_table"]
    display_col    = dim["display_column"]
    source_table   = dim["source_table"]
    source_key     = dim["source_key_column"]
    display_key    = dim.get("display_key") or source_key

    # ── Step 2: LLM rewrites the SQL ────────────────────────────────────────
    sys_p, usr_p = build_drill_sql_prompt(
        original_sql, display_table, display_col,
        source_table, source_key, display_key,
    )

    try:
        raw_sql, _, _ = await llm_complete(
            sys_p, usr_p,
            provider, model, api_key,
            max_tokens=700,
            temperature=0.0,
            **extra_kwargs,
        )
    except Exception as exc:
        log.warning("drill_dim: LLM rewrite failed: %s", exc)
        return _fallback(
            "An error occurred while preparing the drill-down query.",
            f"Try asking: \"Show [metric] by {dim_name}\" directly.",
        )

    drill_sql = _clean_sql(raw_sql)
    if not drill_sql or "CANNOT_REWRITE" in drill_sql.upper():
        return _fallback(
            f"Could not safely add the '{dim_name}' dimension to the current query.",
            f"Try asking: \"Show [metric] broken down by {dim_name}\".",
        )

    # ── Step 3: Validate ────────────────────────────────────────────────────
    try:
        ok, reason, _code = validate_sql(
            drill_sql,
            known_tables or set(),
            db_cfg.get("db_type", "azure_sql"),
        )
        if not ok:
            log.warning("drill_dim: validation failed: %s", reason)
            return _fallback(
                "The rewritten query failed validation.",
                f"Try asking: \"Show [metric] by {dim_name}\" directly.",
            )
    except Exception as exc:
        log.warning("drill_dim: validation error: %s", exc)
        return _fallback("Validation error while preparing the drill-down query.")

    # ── Step 4: Execute ──────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        drill_rows = run_query(
            db_cfg.get("credentials") or db_cfg,
            db_cfg.get("db_type", "azure_sql"),
            drill_sql,
        )
    except Exception as exc:
        log.warning("drill_dim: DB execution failed: %s", exc)
        return _fallback(
            f"The drill-down query failed to execute: {str(exc)[:120]}",
        )
    duration_ms = int((time.monotonic() - t0) * 1000)

    if not drill_rows:
        return _fallback(
            f"The '{dim_name}' breakdown returned no data.",
            "The dimension may not have data for the current filter period.",
        )

    # ── Step 5: Build full assistant_response ───────────────────────────────
    drill_question = f"{question} — broken down by {dim_name}"
    response = build_assistant_response(
        question=drill_question,
        rows=drill_rows,
        sql=drill_sql,
        duration_ms=duration_ms,
        data_source=db_cfg.get("db_type", ""),
        display_context=display_context,
        semantic_plan=semantic_plan,   # pass through so chips re-evaluate
    )
    # Tag the response so the UI can render it as a drill result
    response["drill_from"] = question
    response["drill_dimension"] = dim_name
    return response
