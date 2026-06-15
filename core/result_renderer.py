"""
core/result_renderer.py
───────────────────────
Result formatting and dispatch helpers extracted from main.py.

Covers:
  • _send_results               — send rows to the chat platform (shared by all query paths)
  • _format_value               — smart cell value formatter
  • _generate_result_narration  — one-sentence LLM insight for drill-downs
  • _build_cannot_generate_hint — column-aware "you can ask..." hint
  • _result_has_identifiers     — detect pure-aggregate vs identifier-bearing results
  • _build_followup_sql_context — context block for production-DB follow-up
  • _build_aggregate_drilldown_context — context block for aggregate → detail drills
  • _build_drilldown_context    — entity-value context for "these patients" style follow-ups
  • _sanitize_rows              — coerce non-JSON-serializable cell types
  • _inject_distinct_if_needed  — safety net: add SELECT DISTINCT on list queries
  • _rows_to_table              — plain-text table formatter

Bug fixed during extraction:
  _send_results used two free variables from handle_query's local scope
  (_semantic_plan and audit_request_id) that were inaccessible at module level.
  Replaced with:
    • confidence_context.get("semantic_plan")  (already passed via confidence_context)
    • question_id                              (already a parameter of _send_results)
"""

from __future__ import annotations

import decimal as _decimal
import datetime as _datetime
import logging
import re

from core.llm import llm_complete, resolve_provider
from core.chart import detect_chart_type, build_chart_payload
from core.response_builder import build_assistant_response, build_column_formats, detect_null_metric_issue
from core.insight import generate_followup_suggestions
from core.answer_confidence import build_answer_confidence
from core.answer_formatter import format_success_confidence_text
from core.answer_rca import extract_sql_tables
from core.pipeline_context import get_portal_base
from core.pipeline_helpers import _send_live_stage
from core.pipeline_trace import _create_pin_token

log = logging.getLogger("querybot")


# ── Cell value formatter ──────────────────────────────────────────────────────

# Keywords that indicate a column holds a percentage value.
_PCT_KEYWORDS = {
    "pct", "percent", "percentage", "rate", "ratio", "margin", "share",
    "utilization", "utilisation", "efficiency", "coverage", "yield",
    "variance", "growth", "change", "discount_pct", "discount_percent",
    "fill_rate", "hit_rate", "return_rate", "churn_rate", "conversion",
}

# Keywords that indicate a column holds a currency / money value.
_CURRENCY_KEYWORDS = {
    "amount", "amt", "cost", "price", "revenue", "salary", "wage",
    "budget", "spend", "fee", "charge", "total", "value", "val",
    "payment", "pay", "income", "profit", "loss", "expense", "expenditure",
    "sales", "billing", "billed", "invoice", "invoiced", "gross", "net",
    "earnings", "commission", "bonus", "rebate", "credit", "debit",
    "balance", "liability", "asset", "tax", "vat", "gst",
}


def _detect_column_format(col_name: str) -> str:
    """
    Infer display format from a column name.

    Returns one of: "currency" | "percent" | "number"

    Strategy: tokenise on underscores/spaces/camelCase, then check each
    token against keyword sets.  Percentage wins over currency when both
    match (e.g. MARGIN_PCT → percent, not currency).
    """
    import re
    # Split camelCase and underscores into lower-case tokens
    raw = col_name.lower()
    tokens = set(re.split(r"[_\s]+", raw))
    # Also add the full name without separators (e.g. "totalamt" → check "amt")
    tokens.add(raw.replace("_", "").replace(" ", ""))

    # Percentage check first (takes priority)
    if tokens & _PCT_KEYWORDS:
        return "percent"
    # Suffix-based fast checks common in ERP column codes
    if raw.endswith(("pct", "pc", "_pct", "_rate", "_ratio", "_percent")):
        return "percent"
    if raw.endswith(("amt", "amount", "cost", "price", "rev", "sal")):
        return "currency"
    # Token-based currency check
    if tokens & _CURRENCY_KEYWORDS:
        return "currency"
    return "number"


def _format_value(val, col_name: str = "") -> str:
    """
    Format a single cell value for clean display.

    When col_name is supplied the formatter infers whether the value
    represents a currency amount or a percentage and decorates accordingly:
      - Currency  → $1,234.56
      - Percent   → 45.23%
      - Number    → 1,973,331 / 538.42  (existing behaviour)
      - None      → —
      - String    → as-is
    """
    if val is None:
        return "—"

    fmt = _detect_column_format(col_name) if col_name else "number"

    if isinstance(val, (int, float)):
        num = float(val)

        if fmt == "currency":
            return f"${num:,.2f}"

        if fmt == "percent":
            return f"{num:.2f}%"

        # Plain number — keep existing smart behaviour
        if isinstance(val, float):
            if val == int(val):
                return f"{int(val):,}"
            rounded = round(val, 4)
            parts = f"{rounded:,.4f}".split(".")
            decimal = parts[1].rstrip("0")
            return f"{parts[0]}.{decimal}" if decimal else parts[0]
        if val > 999:
            return f"{val:,}"

    return str(val)


def _rows_to_table(rows) -> str:
    """Format query results as a clean text table with smart value formatting."""
    if not rows:
        return "(no results)"
    headers = list(rows[0].keys())
    formatted = [{h: _format_value(r.get(h), h) for h in headers} for r in rows]
    widths = {
        h: max(len(str(h)), max(len(f[h]) for f in formatted))
        for h in headers
    }
    head = " | ".join(str(h).ljust(widths[h]) for h in headers)
    sep  = "-+-".join("-" * widths[h] for h in headers)
    body = "\n".join(
        " | ".join(f[h].ljust(widths[h]) for h in headers)
        for f in formatted
    )
    return f"{head}\n{sep}\n{body}"


# ── Row sanitisation ──────────────────────────────────────────────────────────

def _sanitize_rows(rows: list[dict]) -> list[dict]:
    """
    Convert non-JSON-serializable cell values to safe Python primitives.

    Covers the types that database drivers and DuckDB commonly return:
      decimal.Decimal  → float  (DB numeric/money columns)
      datetime / date  → ISO string
      bytes            → hex string
      anything else    → str()

    Safe to call on rows that are already clean — passes through
    int/float/str/None unchanged.
    """
    def _safe(v):
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        if isinstance(v, _decimal.Decimal):
            return float(v)
        if isinstance(v, (_datetime.datetime, _datetime.date)):
            return v.isoformat()
        if isinstance(v, bytes):
            return v.hex()
        return str(v)

    return [{k: _safe(val) for k, val in row.items()} for row in (rows or [])]


# ── SQL helpers ───────────────────────────────────────────────────────────────

def _inject_distinct_if_needed(sql: str, question: str) -> str:
    """
    Safety net: if the LLM forgot SELECT DISTINCT on a list-entity query, add it.

    Fires only when ALL of:
      1. The question is a "list entities" question (list/show/who/which/find/get/what)
      2. The SQL has no aggregate functions (SUM/COUNT/AVG/MIN/MAX) in the SELECT clause
      3. The SQL has no GROUP BY clause
      4. SELECT DISTINCT is not already present

    This prevents duplicate rows when a dimension table is joined to a fact table
    without deduplication (e.g. prescribers × fill records = many rows per prescriber).
    """
    _LIST_Q_RE = re.compile(
        r"^\s*(list|show|who|which|find|get|what\s+(are|is)\s+the|"
        r"give\s+me|display|return|fetch|identify)",
        re.IGNORECASE,
    )
    if not _LIST_Q_RE.match(question.strip()):
        return sql

    sql_upper = sql.upper()
    if "SELECT DISTINCT" in sql_upper:
        return sql

    _AGG_RE = re.compile(
        r"\b(SUM|COUNT|AVG|MIN|MAX|STDEV|VARIANCE|PERCENTILE)\s*\(",
        re.IGNORECASE,
    )
    if _AGG_RE.search(sql):
        return sql

    if "GROUP BY" in sql_upper:
        return sql

    patched = re.sub(r"(?i)^(\s*SELECT\s+)", r"\1DISTINCT ", sql, count=1)
    if patched != sql:
        log.info("DISTINCT injected into list-entity SQL for question: %r", question[:80])
    return patched


# ── LLM narration ─────────────────────────────────────────────────────────────

async def _generate_result_narration(
    question: str,
    rows: list[dict],
    currency_cols: list[str],
    client: dict,
) -> str:
    """
    Generate a 1-2 sentence plain-English insight from a drill-down result.
    Uses a minimal 150-token LLM call.  Returns "" on any failure so callers
    can silently skip it.
    """
    if not rows:
        return ""
    try:
        col_names = list(rows[0].keys())
        preview_rows = rows[:5]
        row_lines = []
        for r in preview_rows:
            parts = []
            for k, v in r.items():
                if k in currency_cols and v is not None:
                    try:
                        parts.append(f"{k}=${float(v):,.2f}")
                    except (ValueError, TypeError):
                        parts.append(f"{k}={v}")
                else:
                    parts.append(f"{k}={v}")
            row_lines.append("  " + ", ".join(parts))
        total = len(rows)
        summary = (
            f"Question: {question}\n"
            f"Result ({total} row{'s' if total != 1 else ''}):\n"
            + "\n".join(row_lines)
            + (f"\n  ...and {total - 5} more rows" if total > 5 else "")
        )
        provider, model, api_key, az_kwargs = resolve_provider(client, purpose="query")
        narration, _, _ = await llm_complete(
            system=(
                "You are a data analyst. Given a question and a query result, "
                "write exactly ONE concise sentence (max 30 words) summarising "
                "the key insight or direct answer. Be specific — include the top "
                "value or most notable number. No padding, no 'The data shows'."
            ),
            user=summary,
            provider=provider,
            model=model,
            api_key=api_key,
            max_tokens=80,
            temperature=0.2,
            **az_kwargs,
        )
        return (narration or "").strip()
    except Exception as _exc:
        log.debug("result_chat narration skipped: %s", _exc)
        return ""


# ── Drilldown context builders ────────────────────────────────────────────────

def _build_cannot_generate_hint(
    schema: list[dict],
    stats: dict,
    prev_rows: list[dict] | None = None,
    prev_question: str = "",
) -> str:
    """
    Build a column-aware suggestion message when DuckDB AND the production
    DB fallback both fail to answer.  Lists what the user CAN ask based on
    the available columns, and adds a rephrasing tip with explicit values
    when the question likely references entities from the previous result.
    """
    if not schema:
        return "Try asking a fresh question in the main chat."

    numeric_cols = [
        c["name"] for c in stats.get("columns", [])
        if c.get("min") is not None
    ]
    text_cols = [
        c["name"] for c in stats.get("columns", [])
        if c.get("sample_values")
    ]
    currency_cols = [
        c["name"] for c in stats.get("columns", [])
        if c.get("is_currency")
    ]

    suggestions = []
    if numeric_cols:
        col = numeric_cols[0]
        prefix = "$" if col in currency_cols else ""
        suggestions += [
            f"'what is the average {col.lower().replace('_', ' ')}'",
            f"'show rows where {col.lower().replace('_', ' ')} is above average'",
            f"'rank by {col.lower().replace('_', ' ')}'",
        ]
    if text_cols:
        col = text_cols[0]
        label = col.lower().replace("_", " ")
        suggestions.append(f"'filter by {label}'")

    col_summary = ", ".join(f"`{c['name']}`" for c in schema)

    if not text_cols and numeric_cols:
        num_col = numeric_cols[0].lower().replace("_", " ")
        if prev_rows and len(prev_rows) == 1:
            val = list(prev_rows[0].values())[0] if prev_rows[0] else ""
            return (
                f"This result shows a summary total ({col_summary} = **{val}**). "
                f"There are no patient or record identifiers here to drill into.\n\n"
                f"To list the actual records, ask a **fresh question in the main chat** — for example:\n"
                f"  • *'List all prescriptions with their patient details'*\n"
                f"  • *'Show me the prescriptions that make up this total'*\n"
                f"  • *'List patients with prescription counts'*"
            )
        return (
            f"The current result only has summary columns: {col_summary}.\n"
            "Questions you can ask here:\n"
            + "\n".join(f"  • {s}" for s in suggestions[:3])
            + "\n\nTo see record-level details, ask a fresh question in the main chat."
        )

    hint = (
        f"The current result only has these columns: {col_summary}.\n"
        "Questions you can ask:\n"
        + "\n".join(f"  • {s}" for s in suggestions[:4])
    )
    if prev_rows and text_cols:
        key_col = text_cols[0]
        values  = list(dict.fromkeys(
            str(r[key_col]) for r in prev_rows if r.get(key_col) is not None
        ))[:5]
        if values:
            names = ", ".join(values)
            hint += (
                f"\n\nFor anything else, ask a fresh question in the main chat — "
                f"for example, name them explicitly:\n"
                f"  *'... for {names}'*"
            )
    else:
        hint += "\n\nFor anything else, ask a fresh question in the main chat."
    return hint


def _result_has_identifiers(schema: list[dict]) -> bool:
    """
    Return True if the cached result contains at least one text/categorical
    column that could serve as an entity identifier for a drill-down query.
    """
    _TEXT_TYPES = {
        "TEXT", "VARCHAR", "STRING", "NVARCHAR", "CHAR",
        "CHARACTER VARYING", "NCHAR",
    }
    return any(
        s.get("type", "TEXT").upper().split("(")[0].strip() in _TEXT_TYPES
        for s in (schema or [])
    )


def _build_followup_sql_context(
    original_sql: str,
    original_question: str,
    follow_up_question: str,
    prev_rows: list[dict],
    schema: list[dict],
    has_identifiers: bool,
) -> str:
    """
    Unified context block injected into the production-DB fallback prompt for
    ALL result-chat follow-up questions.
    """
    if not original_sql and not prev_rows:
        return ""

    lines = [
        "FOLLOW-UP QUERY CONTEXT:",
        "The user is asking a follow-up question about a previously returned result.",
    ]
    if original_question:
        lines.append(f'The original question was: "{original_question}"')
    if original_sql:
        lines += [
            "",
            "The SQL that produced the previous result was:",
            original_sql.strip(),
            "",
            "You can answer the follow-up using any of these approaches:",
            "  • Subquery  — SELECT ... FROM (<original SQL>) sub WHERE sub.col = ...",
            "  • CTE       — WITH base AS (<original SQL>) SELECT ... FROM base WHERE ...",
            "  • Rewrite   — Keep the same FROM/WHERE conditions, change SELECT columns.",
            "  • JOIN      — JOIN additional tables onto the result using the KB schema.",
            "  The choice depends on what the follow-up question needs.",
        ]

    if not has_identifiers and original_sql:
        lines += [
            "",
            "NOTE: The previous result was a summary aggregate (COUNT/SUM/AVG).",
            "To list detail records, change SELECT from the aggregate function to the",
            "specific columns requested, keeping FROM and WHERE from the original SQL.",
            "If unsure which columns to use, SELECT TOP 20 * (or LIMIT 20) from the",
            "same table — a broad result is far better than CANNOT_GENERATE.",
        ]
    elif has_identifiers and prev_rows and schema:
        _TEXT_TYPES = {"TEXT", "VARCHAR", "STRING", "NVARCHAR", "CHAR", "CHARACTER VARYING", "NCHAR"}
        value_groups: list[tuple[str, list[str]]] = []
        for col_info in schema:
            col   = col_info["name"]
            dtype = col_info.get("type", "TEXT").upper().split("(")[0].strip()
            if dtype not in _TEXT_TYPES:
                continue
            vals = list(dict.fromkeys(
                str(r[col]) for r in prev_rows if r.get(col) is not None
            ))[:15]
            if vals:
                value_groups.append((col, vals))
            if len(value_groups) >= 3:
                break
        if value_groups:
            lines += ["", "Alternatively, the result contained these specific values:"]
            for col, vals in value_groups:
                in_list = ", ".join(f"'{v}'" for v in vals)
                lines.append(f"  • column '{col}': {in_list}")
            lines.append(
                "You may use these in WHERE col IN (...) instead of a subquery if simpler."
            )

    lines += [
        "",
        "CRITICAL: Use ONLY column and table names that appear in the Knowledge Base above.",
        "Do NOT use SELECT-alias names from the previous result as real column names.",
        "Only return CANNOT_GENERATE if the question requires data that is genuinely",
        "unavailable in the database schema shown in the Knowledge Base.",
    ]
    return "\n".join(lines)


def _build_aggregate_drilldown_context(
    original_sql: str,
    original_question: str,
    follow_up_question: str,
) -> str:
    """
    Build a context block for the production-DB fallback when the previous
    result was a pure aggregate (COUNT/SUM/AVG — no identifier columns).
    """
    if not original_sql:
        return ""

    lines = [
        "AGGREGATE DRILL-DOWN REQUEST:",
        "The user previously ran an aggregate query (COUNT/SUM/AVG) and now",
        "wants to see the underlying detail records that make up that total.",
        "",
        f'The original question was: "{original_question}"',
        "",
        "The original aggregate SQL was:",
        original_sql.strip(),
        "",
        "YOUR TASK:",
        "Rewrite that SQL as a detail query that:",
        "  1. Keeps the exact same FROM clause and WHERE conditions (if any).",
        "  2. Removes the aggregate function (COUNT, SUM, AVG, etc.) from SELECT.",
        "  3. Tries to select the specific columns the user is asking for (e.g. patient",
        "     name, prescription details) based on the Knowledge Base schema.",
        "  4. Adds a row limit (TOP 20 / LIMIT 20) unless the user specified a number.",
        "",
        "FALLBACK RULE (important):",
        "  If you are unsure which exact columns to select, write a SELECT TOP 20 * (or",
        "  LIMIT 20) from the same table(s) and WHERE conditions in the original SQL.",
        "  A generic SELECT * that returns all columns is FAR BETTER than CANNOT_GENERATE.",
        "  Only return CANNOT_GENERATE if you cannot even identify the table to query.",
        "",
        "CRITICAL: Use ONLY column names / table names that appear in the Knowledge Base.",
        "Do NOT use the aggregate alias (e.g. TOTAL_PRESCRIPTIONS) as a column name —",
        "that was a SELECT alias, not a real column in the database.",
    ]
    return "\n".join(lines)


def _build_drilldown_context(
    prev_question: str,
    prev_rows: list[dict],
    schema: list[dict],
) -> str:
    """
    Build a context block injected into the production-DB fallback prompt
    so the LLM can answer follow-up questions that reference entities from
    the previous result ("these top 5 prescribers", "the ones shown", etc.).
    """
    if not prev_rows or not schema:
        return ""

    parts: list[str] = []
    if prev_question:
        parts.append(
            "The user is following up on a previous database result.\n"
            f'The previous question was: "{prev_question}"'
        )
    else:
        parts.append("The user is following up on a previous database result.")

    col_names = ", ".join(s["name"] for s in schema)
    parts.append(
        f"That result returned {len(prev_rows)} row(s) with columns: {col_names}."
    )

    value_groups: list[tuple[str, list[str]]] = []
    for col_info in schema:
        col   = col_info["name"]
        dtype = col_info.get("type", "TEXT").upper()
        if dtype not in ("TEXT", "VARCHAR", "STRING", "NVARCHAR", "CHAR"):
            continue
        values = list(dict.fromkeys(
            str(r[col]) for r in prev_rows if r.get(col) is not None
        ))[:15]
        if values:
            value_groups.append((col, values))
        if len(value_groups) >= 3:
            break

    if value_groups:
        hint_lines: list[str] = []
        for col, values in value_groups:
            in_list = ", ".join(f"'{v}'" for v in values)
            hint_lines.append(
                f"  - The result column '{col}' contained these specific values: {in_list}"
            )
        parts.append(
            "The follow-up question references items from that result.\n"
            "Use the values below to build a WHERE ... IN (...) filter, but identify "
            "the correct production table and column name from the KB schema context — "
            "do NOT use the result column name directly as a database column:\n"
            + "\n".join(hint_lines)
        )

    return "\n".join(parts)


# ── Main result sender ────────────────────────────────────────────────────────

async def _send_results(event, adapter, question, rows, sql, duration_ms,
                        portal_user, account_id, db_cfg,
                        rag_context: str = "", question_id: str | None = None,
                        confidence_context: dict | None = None,
                        display_context: dict | None = None):
    """Send formatted results to the chat platform. Shared by LLM and metric registry paths."""
    column_formats = build_column_formats(rows, display_context=display_context)
    # Cache result on adapter for insight follow-ups (WebSocket sessions).
    # Pass question_id so drilldowns can link back to this original question.
    cache_fn = getattr(adapter, "cache_result", None)
    if callable(cache_fn):
        cache_fn(
            rows, question, sql, db_cfg, rag_context,
            question_id=question_id,
            column_formats=column_formats,
            data_brief=confidence_context.get("data_brief") if confidence_context else None,
            semantic_plan=confidence_context.get("semantic_plan") if confidence_context else None,
        )

    table_text = _rows_to_table(rows)
    row_word   = "row" if len(rows) == 1 else "rows"
    dur_label  = f"{duration_ms}ms" if duration_ms < 1000 else f"{duration_ms/1000:.1f}s"
    _has_confidence_context = bool(confidence_context)
    confidence_context = confidence_context or {}
    null_metric_issue = detect_null_metric_issue(rows)
    confidence = build_answer_confidence(
        validation_code=confidence_context.get("validation_code") or "ok",
        row_count=len(rows),
        retry_count=int(confidence_context.get("retry_count") or 0),
        has_semantic_plan=bool(confidence_context.get("has_semantic_plan")),
        has_graph_context=bool(confidence_context.get("has_graph_context")),
        tables_used=confidence_context.get("tables_used") or extract_sql_tables(sql, db_cfg.get("db_type", "azure_sql")),
        empty_tables=confidence_context.get("empty_tables") or [],
        null_metric_issue=bool(null_metric_issue),
    )

    chart_type = detect_chart_type(rows, question=question)
    pin_token = None
    chart_payload = None
    if chart_type and portal_user:
        pin_token = _create_pin_token(
            user_id=portal_user["id"],
            account_id=account_id,
            question=question,
            sql_query=sql,
            chart_type=chart_type,
            db_config_id=db_cfg["id"],
        )
    if chart_type:
        chart_payload = build_chart_payload(
            rows,
            chart_type,
            title=question,
            question=question,
            column_formats=column_formats,
        )
        if chart_payload:
            if pin_token:
                chart_payload["pin_token"] = pin_token
            chart_payload["question"] = question
            chart_payload["duration_label"] = dur_label
            chart_payload["row_count"] = len(rows)

    rich_sender = getattr(adapter, "send_assistant_response", None)
    if callable(rich_sender):
        if chart_payload:
            await _send_live_stage(adapter, event, "chart_ready", "Building chart", "Rendering an interactive chart for this answer.")
        response_payload = build_assistant_response(
            question=question,
            rows=rows,
            sql=sql,
            duration_ms=duration_ms,
            chart=chart_payload,
            data_source=str(db_cfg.get("db_type", "")),
            confidence=confidence,
            display_context=display_context,
            column_formats=column_formats,
            semantic_plan=confidence_context.get("semantic_plan") or None,
            question_id=question_id,   # exposed in trust block for feedback API
        )
        # ── Result-aware follow-up suggestions (web portal only) ──────────
        selected_schema = (getattr(event, "schema_hint", "") or "").strip().upper()
        response_payload.setdefault("trust", {})["schema"] = (
            f"{selected_schema} schema" if selected_schema else "All allowed schemas"
        )
        response_payload["trust"]["schema_mode"] = "selected" if selected_schema else "all"
        # Generate suggestions from the statistical brief already computed
        # inside build_assistant_response — no extra DB call or raw row exposure.
        # Uses a lightweight 160-token LLM call; failures are silent.
        if portal_user and rows and len(rows[0]) >= 2 if rows else False:
            brief         = response_payload.get("data_brief") or {}
            result_scope  = response_payload.get("result_scope") or {}
            audit_rid     = response_payload.get("analysis_contract", {}).get("request_id", "") or question_id or ""
            try:
                suggestions = await generate_followup_suggestions(
                    brief=brief,
                    question=question,
                    result_scope=result_scope,
                    db_cfg=db_cfg,
                    account_id=account_id,
                    audit_enabled=True,
                    audit_request_id=str(audit_rid),
                )
                if suggestions:
                    response_payload["follow_up_suggestions"] = suggestions
            except Exception as _fex:
                log.debug("Follow-up suggestions skipped: %s", _fex)
        await rich_sender(event, response_payload)
        return

    # Only show the confidence block when a full confidence context was provided
    # (i.e. LLM-path queries). Metric registry and duck-db paths don't compute
    # query-level confidence signals, so we omit the block rather than show
    # misleading default values.
    conf_text = format_success_confidence_text(confidence) + "\n\n" if _has_confidence_context else ""
    if len(rows) == 1 and len(rows[0]) == 1:
        col_name = list(rows[0].keys())[0]
        value    = _format_value(rows[0][col_name], col_name)
        greeting = f"*{portal_user['name']}* — " if portal_user else ""
        reply = (
            f"{greeting}*{question}*\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"  {col_name}\n"
            f"  *{value}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"{conf_text}"
            f"_{dur_label} · {col_name.replace('_', ' ')}_"
        )
    else:
        greeting = f"*{portal_user['name']}*\n" if portal_user else ""
        reply = (
            f"{greeting}*{question}*\n\n"
            f"*{len(rows)} {row_word}*\n"
            f"{'─' * 40}\n"
            f"{table_text}\n"
            f"{'─' * 40}\n"
            f"{conf_text}"
            f"_{dur_label}_"
        )

    await adapter.send_message(event, reply)

    chart_sender = getattr(adapter, "send_chart", None)
    if chart_payload and callable(chart_sender):
        await _send_live_stage(adapter, event, "chart_ready", "Building chart", "Rendering an interactive chart for this answer.")
        await chart_sender(event, chart_payload)
        return

    if chart_payload and portal_user:
        pin_url = f"{get_portal_base()}/portal/pin-confirm?token={pin_token}"
        await adapter.send_message(event, f"Chart ready — [Pin to my dashboard]({pin_url})")
