"""
core/pipeline_helpers.py
────────────────────────
Stateless query-pipeline utilities extracted from main.py.

Covers:
  • _looks_like_new_query          — detect new vs refinement intent
  • _extract_kb_synonym_injection  — build column-synonym hint block from KB
  • _send_live_stage               — push streaming status to chat adapter
  • _sql_preview                   — truncated SQL for display
  • _quote_table_for_count         — dialect-safe table quoting
  • _count_tables_for_zero_row     — best-effort row counts for RCA
  • _zero_row_rca_hints            — intent-based hints for empty results
  • _build_zero_row_message        — full user-facing zero-row response
  • _format_metric_formula_context — admin-approved metric instructions block
  • _extract_metric_formula_tables — pull table names from formula expressions
"""

from __future__ import annotations

import logging
import re

from core.schema import run_query
from core.query_semantics import analyze_query_intent
from core.answer_confidence import build_answer_confidence
from core.answer_formatter import format_zero_row_business_response
from core.answer_rca import build_business_rca, extract_sql_tables

log = logging.getLogger("querybot")


# ── Intent detection ──────────────────────────────────────────────────────────

def _looks_like_new_query(text: str, original_q: str = "") -> bool:
    """
    Return True only when the user's message is clearly a brand-new question
    unrelated to any pending clarification.

    Key fix: a message ending with "?" that shares significant vocabulary
    with the original question is a REFINEMENT, not a new query.
    Example: original="find employees by department", reply="by late status?"
    → shares "by", "department"-adjacent context → treat as refinement.
    """
    msg = (text or "").strip().lower()
    if not msg:
        return False

    # If we have an original question to compare against, use word-overlap
    # to detect refinements before checking for new-query signals.
    if original_q:
        orig_words = {
            w for w in original_q.lower().split()
            if len(w) >= 4  # ignore short words like "is", "by", "the"
        }
        reply_words = {w for w in msg.split() if len(w) >= 4}
        overlap = orig_words & reply_words
        # If reply shares 2+ meaningful words with the original question,
        # it is almost certainly a refinement — not a new query
        if len(overlap) >= 2:
            return False

    # Short one/two word clarification answers are never new queries
    if len(msg.split()) <= 2:
        return False

    starters = (
        "show", "what", "which", "how", "compare", "list", "give",
        "break down", "breakdown", "analyze", "analyse", "explain",
        "why", "trend", "count", "total", "find", "get",
    )
    if any(msg.startswith(prefix) for prefix in starters):
        return True

    # Only treat "?" as a new-query signal when the message is long
    # (short messages ending in "?" are often just emphasis: "attrition status?")
    if "?" in msg and len(msg.split()) >= 5:
        return True

    return len(msg.split()) >= 8  # long messages with no starter are new queries


# ── KB synonym injection ──────────────────────────────────────────────────────

def _extract_kb_synonym_injection(context: str) -> str:
    """
    Scan retrieved KB text for ## Business Synonyms and ## Key Metrics sections
    and build a compact 'Plain-English term → exact column' injection block.

    This fires at every query — even follow-ups — because it works directly from
    the already-retrieved KB chunks without needing the glossary DB to be populated.
    It is the last-resort guard against the LLM inventing CamelCase column names.
    """
    synonym_rows: list[tuple[str, str]] = []   # (plain-english terms, column)
    metric_rows:  list[tuple[str, str]] = []   # (metric name, column)

    in_synonyms = False
    in_metrics  = False

    for line in context.splitlines():
        stripped = line.strip()

        # KB chunk separator — reset section state
        if stripped == "---":
            in_synonyms = False
            in_metrics  = False
            continue

        # Section detection
        if stripped.startswith("## ") or stripped.startswith("### "):
            header = stripped.lstrip("#").strip().lower()
            in_synonyms = header.startswith("business synonym")
            in_metrics  = header.startswith("key metric")
            continue

        # Business Synonyms table rows: | Plain English | Column | Notes |
        if in_synonyms and stripped.startswith("|") and "---" not in stripped:
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if len(cells) >= 2:
                eng  = cells[0].strip("`").strip()
                col  = cells[1].strip("`").strip()
                if col and eng and eng.lower() not in ("plain english", "column", ""):
                    synonym_rows.append((eng, col))

        # Key Metrics lines: - **Metric name**: `COLUMN_NAME` — ...
        if in_metrics and stripped.startswith("-"):
            m = re.match(
                r"-\s*\*\*([^*]+)\*\*\s*:?\s*`?([A-Za-z_][A-Za-z0-9_]*)`?",
                stripped,
            )
            if m:
                metric_name = m.group(1).strip()
                col_name    = m.group(2).strip()
                if metric_name and col_name:
                    metric_rows.append((metric_name, col_name))

    if not synonym_rows and not metric_rows:
        return ""

    lines = [
        "COLUMN SYNONYM MAP (authoritative — use EXACT column names shown here):",
        "When the user's question mentions any of the plain-English terms below, "
        "use the exact column name on the right. Do NOT invent CamelCase variants.",
    ]
    seen_cols: set[str] = set()
    for eng, col in synonym_rows[:25]:
        if col.upper() not in seen_cols:
            lines.append(f"  • '{eng}' → exact column: {col}")
            seen_cols.add(col.upper())
    for metric, col in metric_rows[:15]:
        if col.upper() not in seen_cols:
            lines.append(f"  • '{metric}' → exact column: {col}")
            seen_cols.add(col.upper())

    return "\n".join(lines) + "\n"


# ── Live streaming status ─────────────────────────────────────────────────────

async def _send_live_stage(adapter, event, stage: str, label: str, detail: str = "") -> None:
    sender = getattr(adapter, "send_status", None)
    if callable(sender):
        try:
            await sender(event, stage, label, detail)
        except Exception as e:
            log.debug("Live status send failed: %s", e)


# ── SQL helpers ───────────────────────────────────────────────────────────────

def _sql_preview(sql: str, limit: int = 1200) -> str:
    sql = (sql or "").strip()
    return sql[:limit] + "..." if len(sql) > limit else sql


def _quote_table_for_count(table: str, db_type: str) -> str:
    parts = [p.strip().strip("[]").strip('"').strip("`") for p in str(table or "").split(".") if p.strip()]
    if not parts:
        return ""
    if db_type == "azure_sql":
        return ".".join(f"[{p}]" for p in parts)
    if db_type in {"snowflake", "oracle"}:
        return ".".join(f'"{p}"' for p in parts)
    return ".".join(parts)


# ── Zero-row RCA helpers ──────────────────────────────────────────────────────

def _count_tables_for_zero_row(db_cfg: dict, tables: list[str]) -> dict[str, int | None]:
    """Best-effort table row counts for business RCA after a zero-row answer."""
    db_type = str((db_cfg or {}).get("db_type") or "azure_sql")
    credentials = (db_cfg or {}).get("credentials") or {}
    counts: dict[str, int | None] = {}
    for table in tables[:6]:
        quoted = _quote_table_for_count(table, db_type)
        if not quoted or table in counts:
            continue
        count_expr = "COUNT_BIG(1)" if db_type == "azure_sql" else "COUNT(*)"
        try:
            rows = run_query(credentials, db_type, f"SELECT {count_expr} AS RowCount FROM {quoted}", max_rows=1)
            first = rows[0] if rows else {}
            value = next(iter(first.values())) if first else None
            counts[table] = int(value) if value is not None else None
        except Exception as exc:
            log.debug("Zero-row table count skipped for %s: %s", table, exc)
            counts[table] = None
    return counts


def _zero_row_rca_hints(question: str, graph_ctx: dict | None = None) -> str:
    intent = analyze_query_intent(question)
    hints: list[str] = []
    if intent.get("wants_having_filter"):
        hints.append("HAVING threshold may be too high.")
    if intent.get("wants_missing_records") or (graph_ctx or {}).get("anti_join"):
        hints.append("The anti-join may have found no missing records.")
    if intent.get("wants_named_period") or intent.get("wants_time_series") or intent.get("wants_mom_qoq"):
        hints.append("A date filter or date-key conversion may exclude all rows.")
    if (graph_ctx or {}).get("enabled"):
        hints.append("The graph join path may be too restrictive for the selected tables.")
    if not hints:
        hints.append("Try broadening the filter or checking category values used in the question.")
    return "\n".join(f"- {h}" for h in hints)


def _build_zero_row_message(
    question: str,
    sql: str,
    graph_ctx: dict | None,
    validation_code: str,
    retry_count: int,
    tables_used: list[str] | None = None,
    empty_tables: list[str] | None = None,
    semantic_plan: dict | None = None,
) -> str:
    tables = tables_used or extract_sql_tables(sql)
    empty = empty_tables or []
    confidence = build_answer_confidence(
        validation_code=validation_code or "ok",
        row_count=0,
        retry_count=retry_count,
        has_semantic_plan=bool((semantic_plan or {}).get("enabled")),
        has_graph_context=bool((graph_ctx or {}).get("enabled") or (graph_ctx or {}).get("detected")),
        tables_used=tables,
        empty_tables=empty,
    )
    rca = build_business_rca(
        question=question,
        row_count=0,
        tables_used=tables,
        empty_tables=empty,
        validation_code=validation_code or "ok",
        retry_count=retry_count,
        graph_context=graph_ctx,
        semantic_plan=semantic_plan,
    )
    return format_zero_row_business_response(
        confidence=confidence,
        rca=rca,
        sql=sql,
        sql_preview_fn=_sql_preview,
    )


# ── Metric formula helpers ────────────────────────────────────────────────────

def _format_metric_formula_context(metrics: list[dict], account_id: str = "") -> str:
    if not metrics:
        return ""

    blocks = [
        "=" * 60,
        "APPROVED METRIC FORMULAS — READ THIS FIRST",
        "=" * 60,
        "These metric formulas are ADMIN-APPROVED and take ABSOLUTE PRECEDENCE.",
        "They OVERRIDE any column or formula documented in the Knowledge Base below.",
        "For formula expressions: use the EXACT sql_template in EVERY SELECT expression",
        "(including inside CTEs). The formula columns MUST appear in the SELECT clause.",
        "NEVER substitute a similar-sounding column from the KB for an approved formula.",
    ]
    for idx, metric in enumerate(metrics, start=1):
        formula_type = (metric.get("formula_type") or "query").lower()
        kind = "formula expression" if formula_type == "expression" else "trusted SQL query/template"
        lines = [
            f"{idx}. Metric: {metric.get('name', '')}",
            f"   Type: {kind}",
            f"   Result format: {metric.get('result_format') or 'number'}",
            f"   Synonyms: {metric.get('synonyms') or '(none)'}",
        ]
        if metric.get("description"):
            lines.append(f"   Business meaning: {metric.get('description')}")
        req_cols = (metric.get("required_columns") or "").strip()
        if req_cols:
            lines.append(f"   Required columns (MUST appear in SELECT): {req_cols}")
        if metric.get("allowed_dimensions"):
            lines.append(f"   Safe dimensions: {metric.get('allowed_dimensions')}")
        if metric.get("grain"):
            lines.append(f"   Grain: {metric.get('grain')}")
        if metric.get("example_questions"):
            lines.append(f"   Example questions: {metric.get('example_questions')}")
        if metric.get("default_time_column"):
            lines.append(f"   Default time column: {metric.get('default_time_column')} — use this column when grouping by date/period")
        sql_tpl = (metric.get("sql_template") or "").strip()
        if account_id and "${" in sql_tpl:
            try:
                from store.config_store import resolve_metric_refs
                sql_tpl = resolve_metric_refs(account_id, sql_tpl)
            except Exception:
                pass  # use raw formula if resolution fails
        lines.append(f"   EXACT formula to use in SELECT: {sql_tpl}")
        if formula_type == "expression" and req_cols:
            lines.append(
                f"   *** WARNING: The Knowledge Base may document similar columns. "
                f"You MUST use the formula above — not any other column. ***"
            )
        blocks.append("\n".join(lines))
    blocks.append("=" * 60)
    return "\n\n".join(blocks)


def _extract_metric_formula_tables(metrics: list[dict]) -> set[str]:
    """
    Extract bare table names from formula expressions so the table-coverage
    guarantee can fetch KB docs for them even when they are not in the graph.

    Handles TABLE.COLUMN and SCHEMA.TABLE.COLUMN patterns.
    """
    tables: set[str] = set()
    for metric in metrics:
        if (metric.get("formula_type") or "query").lower() != "expression":
            continue
        sql = metric.get("sql_template") or ""
        # Match WORD.WORD patterns (TABLE.COLUMN or SCHEMA.TABLE)
        for match in re.finditer(r'\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b', sql):
            # The first part is the table (or schema). Collect both so the
            # gap-fill can try each variant (bare table, schema.table).
            tables.add(match.group(1).upper())
            tables.add(f"{match.group(1).upper()}.{match.group(2).upper()}")
    return tables
