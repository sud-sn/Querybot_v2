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
import json
import re
from typing import Callable

from core.schema import run_query
from core.query_semantics import analyze_query_intent
from core.answer_confidence import build_answer_confidence
from core.answer_formatter import format_zero_row_business_response
from core.answer_rca import build_business_rca, extract_sql_tables

log = logging.getLogger("querybot")


def allow_progressive_sql_repair(
    seen_reason_codes: set[str] | list[str] | tuple[str, ...],
    candidate_reason_code: str,
    attempts_used: int,
    *,
    max_attempts: int = 2,
) -> bool:
    """Admit another LLM SQL repair only when it is making progress.

    A different validator reason means the previous repair cleared one layer
    and exposed another.  Repeating the same reason is a non-progress loop and
    is stopped immediately.  The hard attempt cap prevents runaway cost.
    """
    code = str(candidate_reason_code or "").strip().casefold()
    seen = {str(value or "").strip().casefold() for value in seen_reason_codes}
    return bool(code and attempts_used < max_attempts and code not in seen)


# ── Prompt-context size control ───────────────────────────────────────────────
# The assembled SQL-generation context previously had NO size limit anywhere:
# 7 full reassembled table docs + up to 6 gap-fill docs + examples + semantic
# blocks could grow the prompt unboundedly (quality degradation, cost spikes,
# context-window overflow on wide schemas). Two layers of control:
#   _clamp_kb_doc     — per-doc: drop droppable sections from one KB doc
#   _clamp_prompt_context — final hard cap on the fully assembled string
import os as _os

_PER_DOC_CHAR_CAP = int(_os.getenv("QUERYBOT_KB_DOC_CHAR_CAP", "9000"))
_PROMPT_CONTEXT_CHAR_CAP = int(_os.getenv("QUERYBOT_PROMPT_CONTEXT_CHAR_CAP", "120000"))

# Sections safe to drop from an oversized KB doc. Columns and Join Keys are
# never dropped — they are what SQL generation actually needs.
#
# ORDER MATTERS, and it is by value, not by position in the file. The previous
# version walked backwards from the tail, which made "## Business Synonyms" the
# first thing removed simply because the mandated KB format puts it last. That
# is the one section mapping plain-English terms to exact columns, and the only
# one carrying the "could be confused with a generic name" warnings — so the
# disambiguation material was discarded first, on precisely the biggest,
# most column-dense tables where disambiguation matters most. It also silently
# emptied the downstream COLUMN SYNONYM MAP, which is built by re-reading this
# already-clamped text.
_DROP_ORDER = (
    re.compile(r"(?i)^##\s*sample\s+data\b"),
    re.compile(r"(?i)^##\s*(query\s+patterns|patterns)\b"),
    re.compile(r"(?i)^##\s*overview\b"),
    # Last resort among the droppable sections.
    re.compile(r"(?i)^##\s*business\s+synonyms\b"),
)

# Kept for callers/tests that ask "is this section droppable at all?"
_DROPPABLE_SECTION_RE = re.compile(
    r"(?i)^##\s*(business\s+synonyms|sample\s+data|query\s+patterns|patterns|overview)\b"
)


def _clamp_kb_doc(doc: str, cap: int = 0) -> str:
    """Trim one KB doc to *cap* chars by removing droppable sections in order of
    increasing value (sample data first, business synonyms last), never Columns
    or Join Keys. Falls back to a hard tail-truncate only if still over after
    every droppable section has gone."""
    cap = cap or _PER_DOC_CHAR_CAP
    if len(doc) <= cap:
        return doc

    # Split into header + "## " sections, preserving order.
    parts = re.split(r"(?m)^(?=## )", doc)
    kept = list(parts)
    dropped: list[str] = []
    for pattern in _DROP_ORDER:
        if len("".join(kept)) <= cap:
            break
        for idx in range(len(kept) - 1, 0, -1):
            if len("".join(kept)) <= cap:
                break
            if pattern.match(kept[idx]):
                dropped.append(kept[idx].splitlines()[0].strip())
                kept.pop(idx)
    clamped = "".join(kept)
    if len(clamped) > cap:
        # Only Columns / Join Keys are left and they still do not fit. The cut
        # lands inside material this function promises to preserve, so say so
        # rather than letting a table lose its column list without a trace.
        log.warning(
            "KB doc still %d chars over the %d cap after dropping %s — "
            "hard-truncating, which cuts into Columns or Join Keys",
            len(clamped) - cap, cap, ", ".join(dropped) or "nothing droppable",
        )
        clamped = clamped[:cap] + "\n[... truncated for prompt size]"
    elif dropped:
        log.debug("KB doc clamped to %d chars; dropped %s", cap, ", ".join(dropped))
    return clamped


def _clamp_prompt_context(context: str, cap: int = 0) -> str:
    """Final hard cap on the assembled prompt context. Priority blocks
    (semantic model context, metric formulas) are PREPENDED upstream, so a
    tail truncation always sacrifices the lowest-priority material (the last
    KB docs / hints), never the deterministic guidance at the head."""
    cap = cap or _PROMPT_CONTEXT_CHAR_CAP
    if len(context) <= cap:
        return context
    log.warning(
        "Prompt context clamped: %d chars -> %d cap (tail truncated)",
        len(context), cap,
    )
    return context[:cap] + "\n\n[... additional context truncated for prompt size]"


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
    # (plain-english term, column, owning table FQN)
    synonym_rows: list[tuple[str, str, str]] = []
    metric_rows:  list[tuple[str, str, str]] = []

    in_synonyms = False
    in_metrics  = False
    # Every reassembled KB doc opens with a single-# FQN header (see
    # core/vector_store.py _reconstruct_full_doc). Without tracking it, a row
    # harvested from one table's synonyms section was emitted as a bare column
    # name and could be applied to a different table entirely.
    current_table = ""

    for line in context.splitlines():
        stripped = line.strip()

        # KB chunk separator — reset section state
        if stripped == "---":
            in_synonyms = False
            in_metrics  = False
            current_table = ""
            continue

        # Doc header: "# DW.F_SALES"
        if stripped.startswith("# "):
            current_table = stripped[2:].strip().strip("`")
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
                    synonym_rows.append((eng, col, current_table))

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
                    metric_rows.append((metric_name, col_name, current_table))

    if not synonym_rows and not metric_rows:
        return ""

    lines = [
        "COLUMN SYNONYM MAP (use EXACT column names shown here — never invent "
        "CamelCase variants):",
        "Each mapping belongs to the table it is listed under. Apply a mapping "
        "only to that table.",
    ]
    lines.extend(_synonym_map_lines(synonym_rows[:25], metric_rows[:15]))
    return "\n".join(lines) + "\n"


def _synonym_map_lines(
    synonym_rows: list[tuple[str, str, str]],
    metric_rows: list[tuple[str, str, str]],
) -> list[str]:
    """Render harvested term→column rows without contradicting ourselves.

    Two tables both documenting "revenue" is normal and not a conflict — one
    means F_SALES.NET_AMOUNT, the other F_ORDERS.ORDER_VALUE. The previous
    block dropped the owning table, deduplicated on the COLUMN, and presented
    whatever survived as "authoritative", so the model received two unqualified
    and contradictory instructions for the same business word and picked
    between them arbitrarily. It could also apply a column from one table to
    another, which is either an invalid-column error or, worse, a valid column
    that means something else.

    Keying on the column also silently discarded a second TERM for the same
    column, losing a phrasing users actually type.
    """
    by_term: dict[str, list[tuple[str, str]]] = {}
    order: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for term, column, table in [*synonym_rows, *metric_rows]:
        key = (term.lower(), table.upper(), column.upper())
        if key in seen:
            continue
        seen.add(key)
        if term.lower() not in by_term:
            by_term[term.lower()] = []
            order.append(term.lower())
        by_term[term.lower()].append((column, table))

    def qualified(column: str, table: str) -> str:
        return f"{table}.{column}" if table else column

    lines: list[str] = []
    for term_key in order:
        entries = by_term[term_key]
        display = term_key
        if len(entries) == 1:
            column, table = entries[0]
            lines.append(f"  • '{display}' → exact column: {qualified(column, table)}")
            continue
        listed = ", ".join(qualified(column, table) for column, table in entries)
        lines.append(
            f"  • '{display}' is documented on more than one table — use the one "
            f"belonging to the table the query selects FROM: {listed}. If the "
            f"question does not make the table clear, do not guess."
        )
    return lines


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
    account_id: str = "",
) -> str:
    tables = tables_used or extract_sql_tables(sql)
    empty = empty_tables or []
    # A WHERE literal that matches nothing in the value index is the most
    # actionable zero-row explanation — the user gets the closest real values
    # instead of a generic "no matching records".
    unmatched_literals: list[dict] = []
    if account_id:
        try:
            from core.value_resolver import find_unmatched_literals
            unmatched_literals = find_unmatched_literals(sql, account_id)
        except Exception as exc:
            log.debug("Unmatched-literal check skipped: %s", exc)
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
        unmatched_literals=unmatched_literals,
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
        builder_config = metric.get("metric_builder_config") or ""
        if builder_config:
            try:
                cfg = json.loads(builder_config)
            except Exception:
                cfg = {}
            if isinstance(cfg, dict) and cfg.get("mode") == "row_calculated":
                lines.append("   Row-level metric: calculate the expression per source row, then aggregate it at the user's requested grain.")
                joins = cfg.get("required_joins") or []
                if joins:
                    lines.append("   Required row-expression joins:")
                    for join in joins:
                        if not isinstance(join, dict):
                            continue
                        alias = join.get("alias") or "(choose alias)"
                        table = join.get("table") or join.get("to_table") or ""
                        from_col = join.get("from_column") or ""
                        to_col = join.get("to_column") or ""
                        role = join.get("role") or ""
                        lines.append(
                            f"     - Join {table} AS {alias} ON fact.{from_col} = {alias}.{to_col}"
                            + (f" for {role}" if role else "")
                        )
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
        builder_config = metric.get("metric_builder_config") or ""
        if builder_config:
            try:
                cfg = json.loads(builder_config)
            except Exception:
                cfg = {}
            if isinstance(cfg, dict):
                for join in cfg.get("required_joins") or []:
                    if isinstance(join, dict):
                        table = (join.get("table") or join.get("to_table") or "").strip()
                        if table:
                            tables.add(table.upper())
        # Match WORD.WORD patterns (TABLE.COLUMN or SCHEMA.TABLE)
        for match in re.finditer(r'\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b', sql):
            # The first part is the table (or schema). Collect both so the
            # gap-fill can try each variant (bare table, schema.table).
            tables.add(match.group(1).upper())
            tables.add(f"{match.group(1).upper()}.{match.group(2).upper()}")
    return tables


def _build_row_metric_join_sql(
    metrics: list[dict],
    db_type: str,
    existing_skeleton: str,
) -> str:
    """
    For row-calculated metrics, build LEFT JOIN clauses using the metric-defined
    aliases (e.g. due_dt, pay_dt) and return them ready to append to the graph
    join skeleton. Returns "" when there is nothing to inject or the anchor alias
    cannot be determined.

    The anchor alias is parsed from the FROM line of the existing skeleton so the
    ON clause references the correct fact-table alias (e.g. inv, pre).
    """
    from core.graph_resolver import _quote_table, _quote_col

    anchor_alias = ""
    for line in (existing_skeleton or "").splitlines():
        if line.strip().upper().startswith("FROM "):
            parts = line.strip().split()
            if len(parts) >= 3:
                anchor_alias = parts[-1]
            break

    if not anchor_alias:
        return ""

    seen: set[str] = set()
    lines: list[str] = []

    for metric in metrics:
        builder_config = metric.get("metric_builder_config") or ""
        if not builder_config:
            continue
        try:
            cfg = json.loads(builder_config)
        except Exception:
            continue
        if not isinstance(cfg, dict) or cfg.get("mode") not in ("row_calculated", "date_gap"):
            continue

        for join in cfg.get("required_joins") or []:
            if not isinstance(join, dict):
                continue
            alias = (join.get("alias") or "").strip()
            table = (join.get("table") or "").strip()
            from_col = (join.get("from_column") or "").strip()
            to_col = (join.get("to_column") or "").strip()

            if not alias or not table or not from_col or not to_col:
                continue

            key = f"{alias}:{table.upper()}:{from_col.upper()}:{to_col.upper()}"
            if key in seen:
                continue
            seen.add(key)

            tbl_sql = _quote_table(table, "", db_type)
            from_sql = _quote_col(from_col, db_type)
            to_sql = _quote_col(to_col, db_type)

            on_clause = f"{anchor_alias}.{from_sql} = {alias}.{to_sql}"
            # Sentinel key values (0, 777…) mean "no date" — excluding them in
            # the LEFT JOIN makes the dimension date read as NULL, matching the
            # ISBLANK guard of the equivalent DAX measure.
            invalid_keys = [
                str(int(k)) for k in (join.get("invalid_keys") or [])
                if str(k).strip().lstrip("-").isdigit()
            ]
            if invalid_keys:
                on_clause += f" AND {anchor_alias}.{from_sql} NOT IN ({', '.join(invalid_keys)})"

            lines.append(f"LEFT  JOIN {tbl_sql} {alias} ON {on_clause}")

    return "\n".join(lines)


# ── Deterministic field-plan repair ──────────────────────────────────────────

_REPAIR_DIALECT = {"azure_sql": "tsql", "oracle": "oracle", "snowflake": "snowflake"}


def _same_physical_table(left: object, right: object) -> bool:
    """Compare qualified table names without making database qualification mandatory."""
    left_parts = [part for part in re.split(r"[.\[\]`\"]", str(left or "").upper()) if part]
    right_parts = [part for part in re.split(r"[.\[\]`\"]", str(right or "").upper()) if part]
    if not left_parts or not right_parts:
        return False
    if left_parts == right_parts or left_parts[-2:] == right_parts[-2:]:
        return True
    # A bare table name may legitimately be compared with a discovered
    # schema-qualified name. Two differently qualified schemas must not be
    # treated as the same table merely because their final identifier matches.
    return (
        (len(left_parts) == 1 or len(right_parts) == 1)
        and left_parts[-1] == right_parts[-1]
    )


def _join_condition_pairs(join: dict) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for condition in join.get("conditions") or []:
        if isinstance(condition, dict):
            left = condition.get("from_column") or condition.get("left_column")
            right = condition.get("to_column") or condition.get("right_column")
        elif isinstance(condition, (list, tuple)) and len(condition) >= 2:
            left, right = condition[0], condition[1]
        else:
            continue
        left_text = str(left or "").strip().strip("[]\"`")
        right_text = str(right or "").strip().strip("[]\"`")
        if left_text and right_text:
            pairs.append((left_text, right_text))
    return pairs


def _required_join_path(
    source: str,
    target: str,
    joins: list[dict],
    *,
    forbidden_facts: set[str] | None = None,
) -> list[dict]:
    """Return the shortest governed path, preserving each edge's orientation."""
    if _same_physical_table(source, target):
        return []
    graph: dict[str, list[dict]] = {}
    names: dict[str, str] = {}

    def key(table: object) -> str:
        text = str(table or "").strip()
        parts = [part for part in re.split(r"[.\[\]`\"]", text.upper()) if part]
        return ".".join(parts[-2:] if len(parts) >= 2 else parts)

    forbidden = {key(table) for table in (forbidden_facts or set())}
    for join in joins:
        if str(join.get("enforcement") or "required").lower() == "optional":
            continue
        left = str(join.get("from") or join.get("from_table") or "").strip()
        right = str(join.get("to") or join.get("to_table") or "").strip()
        conditions = _join_condition_pairs(join)
        if not left or not right or not conditions:
            continue
        left_key, right_key = key(left), key(right)
        names[left_key], names[right_key] = left, right
        graph.setdefault(left_key, []).append({
            "from": left_key, "to": right_key, "conditions": conditions,
        })
        graph.setdefault(right_key, []).append({
            "from": right_key, "to": left_key,
            "conditions": [(right_col, left_col) for left_col, right_col in conditions],
        })

    start, finish = key(source), key(target)
    queue: list[tuple[str, list[dict]]] = [(start, [])]
    seen = {start}
    while queue:
        table, path = queue.pop(0)
        for edge in graph.get(table, []):
            nxt = edge["to"]
            if nxt in seen or (nxt in forbidden and nxt != finish):
                continue
            next_path = path + [{
                "from": names.get(edge["from"], edge["from"]),
                "to": names.get(edge["to"], edge["to"]),
                "conditions": list(edge["conditions"]),
            }]
            if nxt == finish:
                return next_path
            seen.add(nxt)
            queue.append((nxt, next_path))
    return []


def _compile_governed_grouped_request_sql(
    db_type: str,
    known_tables: set[str],
    allowed_tables: set[str] | None,
    table_columns: dict[str, dict[str, str]] | None,
    semantic_context: dict | None,
) -> str:
    """Compile one-fact grouped/ranked requests from exact governed bindings.

    This path intentionally refuses to infer a table, column, relationship, or
    aggregation. It exists for cases where the semantic compiler has already
    resolved those decisions but free-form generation can still drift onto an
    unrelated metric or fact table.
    """
    from core.contextual_dates import format_period_bucket_expression
    from core.validator import validate_sql_detailed

    context = semantic_context or {}
    plan = context.get("semantic_plan") or {}
    request = context.get("analytical_request_plan") or {}
    question = str(context.get("question") or "")
    if str(request.get("status") or "") != "compiled":
        return ""
    source_facts = [str(table) for table in (request.get("source_facts") or []) if table]
    if len(source_facts) != 1 or request.get("subrequests"):
        return ""
    fact_table = str(request.get("source_fact") or source_facts[0])
    if not fact_table:
        return ""

    policies = list(plan.get("temporal_policies") or [])
    if len(policies) > 1:
        return ""
    policy = policies[0] if policies else {}
    if policy and str(policy.get("kind") or "") not in {
        "last_n", "latest_n_observed", "today", "yesterday",
    }:
        return ""

    metrics = [
        metric for metric in (context.get("metric_formulas") or [])
        if str(metric.get("formula_type") or "query").lower() == "expression"
        and str(metric.get("sql_template") or metric.get("formula") or "").strip()
    ]
    metric_specs: list[tuple[str, str]] = []
    for metric in metrics:
        sources = list(
            metric.get("_resolved_source_tables")
            or metric.get("source_tables")
            or ([metric.get("base_table")] if metric.get("base_table") else [])
        )
        if sources and not any(_same_physical_table(source, fact_table) for source in sources):
            return ""
        formula = str(metric.get("sql_template") or metric.get("formula") or "").strip().rstrip(";")
        if not formula or re.search(r"\b(?:SELECT|FROM|WITH)\b|;", formula, re.I):
            return ""
        alias = re.sub(r"[^A-Za-z0-9_]", "_", str(metric.get("name") or "metric")).upper()
        metric_specs.append((alias or "METRIC", formula))

    derived = request.get("derived_measure") or {}
    if not metric_specs and derived.get("semantics") == "count_distinct_business_identifier":
        target_table = str(derived.get("target_table") or "")
        target_column = str(derived.get("target_column") or "")
        if not target_column or not _same_physical_table(target_table, fact_table):
            return ""
        metric_specs.append((
            re.sub(r"[^A-Za-z0-9_]", "_", str(derived.get("business_entity") or "event")).upper()
            + "_COUNT",
            f"COUNT(DISTINCT fact_rows.{{qcol:{target_column}}})",
        ))
    if not metric_specs or len(metric_specs) > 3:
        return ""

    def qcol(name: str) -> str:
        clean = str(name or "").strip().strip("[]\"`")
        if db_type == "azure_sql":
            return f"[{clean}]"
        if db_type in {"snowflake", "oracle"}:
            return f'"{clean}"'
        return clean

    quoted_metric_specs: list[tuple[str, str]] = []
    derived_target_column = str(derived.get("target_column") or "")
    for alias, formula in metric_specs:
        if "{qcol:" in formula:
            formula = formula.replace(
                f"{{qcol:{derived_target_column}}}", qcol(derived_target_column)
            )
        quoted_metric_specs.append((alias, formula))
    metric_specs = quoted_metric_specs

    required_fields = [
        field for field in (plan.get("fields") or [])
        if str(field.get("enforcement") or "required").lower() != "optional"
    ]
    dimensions = []
    for field in required_fields:
        role = str(field.get("role") or "").lower()
        if role in {"date_dimension", "contextual_date", "measure", "measure_candidate"}:
            continue
        table = str(field.get("table") or "")
        column = str(field.get("column") or "")
        if not table or not column:
            continue
        if field.get("display_required") or not _same_physical_table(table, fact_table):
            dimensions.append(field)
        elif role in {"dimension", "display_dimension", "attribute"}:
            column_types = next(
                (cols for name, cols in (table_columns or {}).items() if _same_physical_table(name, table)),
                {},
            )
            dtype = str(column_types.get(column) or column_types.get(column.upper()) or "").lower()
            if any(token in dtype for token in ("char", "text", "string")):
                dimensions.append(field)
    unique_dimensions: list[dict] = []
    for field in dimensions:
        if not any(
            _same_physical_table(field.get("table"), existing.get("table"))
            and str(field.get("column") or "").upper() == str(existing.get("column") or "").upper()
            for existing in unique_dimensions
        ):
            unique_dimensions.append(field)
    if len(unique_dimensions) > 1:
        return ""
    # Preserve the established scalar/single-series compiler byte-for-byte.
    # This helper owns only the new cases: a governed dimension, a governed
    # derived event count, or multiple same-fact metrics.
    if (
        not unique_dimensions
        and len(metric_specs) == 1
        and derived.get("semantics") != "count_distinct_business_identifier"
    ):
        return ""

    intent = str(request.get("intent") or "").lower()
    top_n = request.get("top_n") or ((context.get("top_n") or {}).get("limit"))
    ranking = intent == "ranking" or bool(top_n)
    if ranking and not unique_dimensions:
        return ""
    if intent in {"comparison", "distribution", "causal_analysis"}:
        return ""

    fact_sql = _quote_table_for_count(fact_table, db_type)
    from_lines = [f"{fact_sql} AS fact_rows"]
    alias_by_table: dict[str, str] = {fact_table: "fact_rows"}

    def alias_for(table: str) -> str:
        for known, alias in alias_by_table.items():
            if _same_physical_table(known, table):
                return alias
        return ""

    requested_targets: list[tuple[str, str]] = []
    if unique_dimensions:
        requested_targets.append((str(unique_dimensions[0]["table"]), "business_dimension"))
    date_target = str(policy.get("dimension_table") or "") if policy else ""
    date_alias = re.sub(r"[^A-Za-z0-9_]", "_", str(policy.get("role_alias") or "business_date"))
    if date_target:
        requested_targets.append((date_target, date_alias))

    all_source_facts = {str(table) for table in source_facts}
    joins = list(plan.get("joins") or [])
    for target, preferred_alias in requested_targets:
        if alias_for(target):
            continue
        path = _required_join_path(
            fact_table, target, joins,
            forbidden_facts={table for table in all_source_facts if not _same_physical_table(table, fact_table)},
        )
        if not path:
            return ""
        for edge in path:
            left_alias = alias_for(str(edge["from"]))
            if not left_alias:
                return ""
            right_table = str(edge["to"])
            right_alias = alias_for(right_table)
            if right_alias:
                continue
            right_alias = preferred_alias if _same_physical_table(right_table, target) else f"join_{len(alias_by_table)}"
            if right_alias in alias_by_table.values():
                right_alias = f"join_{len(alias_by_table)}"
            conditions = " AND ".join(
                f"{left_alias}.{qcol(left_col)} = {right_alias}.{qcol(right_col)}"
                for left_col, right_col in edge["conditions"]
            )
            from_lines.append(
                f"JOIN {_quote_table_for_count(right_table, db_type)} AS {right_alias} ON {conditions}"
            )
            alias_by_table[right_table] = right_alias

    dimension_ref = ""
    dimension_alias = ""
    if unique_dimensions:
        dimension = unique_dimensions[0]
        dimension_alias = re.sub(
            r"[^A-Za-z0-9_]", "_", str(dimension.get("term") or dimension.get("column") or "dimension")
        ).upper()
        table_alias = alias_for(str(dimension["table"]))
        if not table_alias:
            return ""
        dimension_ref = f"{table_alias}.{qcol(str(dimension['column']))}"

    date_ref = ""
    if policy:
        fact_column = str(policy.get("fact_column") or policy.get("anchor_column") or "")
        date_column = str(policy.get("date_column") or "")
        key_type = str(policy.get("date_key_type") or "")
        if not fact_column or not date_column:
            return ""
        if key_type == "surrogate_fk":
            if not date_target or not str(policy.get("dimension_key") or ""):
                return ""
            resolved_date_alias = alias_for(date_target)
            if not resolved_date_alias:
                return ""
            date_ref = f"{resolved_date_alias}.{qcol(date_column)}"
        else:
            date_ref = f"fact_rows.{qcol(date_column)}"

    from_sql = "\n    ".join(from_lines)
    where_parts: list[str] = []
    anchor_sql = ""
    if policy:
        try:
            amount = int(policy.get("amount"))
        except (TypeError, ValueError):
            return ""
        unit = str(policy.get("unit") or "").lower()
        kind = str(policy.get("kind") or "")
        if unit not in {"day", "week", "month", "quarter", "year"}:
            return ""
        if kind in {"last_n", "latest_n_observed"} and amount <= 0:
            return ""
        if kind == "latest_n_observed":
            if unit != "day":
                # Selecting latest observed weeks/months requires a governed
                # period bucket before limiting. Keep that less common shape
                # on the validated planner path for now.
                return ""
            if db_type == "azure_sql":
                observed_limit = f"TOP ({amount}) "
                observed_suffix = ""
            elif db_type == "snowflake":
                observed_limit = ""
                observed_suffix = f"\n    LIMIT {amount}"
            elif db_type == "oracle":
                observed_limit = ""
                observed_suffix = f"\n    FETCH FIRST {amount} ROWS ONLY"
            else:
                return ""
            anchor_sql = f"""WITH observed_periods AS (
    SELECT DISTINCT {observed_limit}{date_ref} AS observed_business_date
    FROM {from_sql}
    WHERE {date_ref} IS NOT NULL
    ORDER BY {date_ref} DESC{observed_suffix}
)
"""
            where_parts.append(
                f"{date_ref} IN (SELECT observed_business_date FROM observed_periods)"
            )
        else:
            anchor_sql = f"WITH anchor AS (\n    SELECT MAX({date_ref}) AS max_business_date\n    FROM {from_sql}\n)\n"

        recipe = request.get("analytical_recipe") or {}
        if recipe.get("kind") == "period_over_period_entity_change":
            if (
                kind != "last_n"
                or not dimension_ref
                or derived.get("semantics") != "count_distinct_business_identifier"
                or db_type not in {"azure_sql", "snowflake"}
            ):
                return ""
            target_column = str(derived.get("target_column") or "")
            if not target_column:
                return ""
            current_start = f"DATEADD({unit}, -{amount}, anchor.max_business_date)"
            prior_start = f"DATEADD({unit}, -{amount * 2}, anchor.max_business_date)"
            target_ref = f"fact_rows.{qcol(target_column)}"
            direction = str(recipe.get("direction") or "compare").lower()
            direction_filter = ""
            order_direction = "DESC"
            if direction == "decrease":
                direction_filter = "\nWHERE CURRENT_PERIOD_COUNT < PRIOR_PERIOD_COUNT"
                order_direction = "ASC"
            elif direction == "increase":
                direction_filter = "\nWHERE CURRENT_PERIOD_COUNT > PRIOR_PERIOD_COUNT"
            change_anchor_sql = anchor_sql.rstrip() + ",\n"
            compiled_change = f"""{change_anchor_sql}period_counts AS (
    SELECT
        {dimension_ref} AS ENTITY,
        COUNT(DISTINCT CASE
            WHEN {date_ref} > {current_start}
             AND {date_ref} <= anchor.max_business_date
            THEN {target_ref} END) AS CURRENT_PERIOD_COUNT,
        COUNT(DISTINCT CASE
            WHEN {date_ref} > {prior_start}
             AND {date_ref} <= {current_start}
            THEN {target_ref} END) AS PRIOR_PERIOD_COUNT
    FROM {from_sql}
    CROSS JOIN anchor
    WHERE {date_ref} > {prior_start}
      AND {date_ref} <= anchor.max_business_date
    GROUP BY {dimension_ref}
)
SELECT
    ENTITY,
    CURRENT_PERIOD_COUNT,
    PRIOR_PERIOD_COUNT,
    CURRENT_PERIOD_COUNT - PRIOR_PERIOD_COUNT AS ABSOLUTE_CHANGE,
    CASE
        WHEN PRIOR_PERIOD_COUNT = 0 THEN NULL
        ELSE 100.0 * (CURRENT_PERIOD_COUNT - PRIOR_PERIOD_COUNT) / PRIOR_PERIOD_COUNT
    END AS PERCENTAGE_CHANGE
FROM period_counts{direction_filter}
ORDER BY ABSOLUTE_CHANGE {order_direction}"""
            result = validate_sql_detailed(
                compiled_change,
                known_tables,
                db_type,
                allowed_tables,
                table_columns or {},
                context,
            )
            if not result.ok:
                log.debug(
                    "Governed entity-change compiler did not validate: %s (%s)",
                    result.reason,
                    result.code,
                )
                return ""
            return compiled_change

        if kind == "latest_n_observed":
            pass
        elif kind == "last_n":
            if db_type in {"azure_sql", "snowflake"}:
                start = f"DATEADD({unit}, -{amount}, anchor.max_business_date)"
            elif db_type == "oracle":
                if unit == "day":
                    start = f"anchor.max_business_date - {amount}"
                elif unit == "week":
                    start = f"anchor.max_business_date - {amount * 7}"
                else:
                    start = f"ADD_MONTHS(anchor.max_business_date, -{amount * {'month': 1, 'quarter': 3, 'year': 12}[unit]})"
            else:
                return ""
            where_parts.extend([f"{date_ref} > {start}", f"{date_ref} <= anchor.max_business_date"])
        else:
            if db_type in {"azure_sql", "snowflake"}:
                selected = "CAST(anchor.max_business_date AS date)"
                if kind == "yesterday":
                    selected = "DATEADD(day, -1, CAST(anchor.max_business_date AS date))"
                where_parts.append(f"CAST({date_ref} AS date) = {selected}")
            elif db_type == "oracle":
                selected = "TRUNC(anchor.max_business_date)" + (" - 1" if kind == "yesterday" else "")
                where_parts.append(f"TRUNC({date_ref}) = {selected}")
            else:
                return ""

    is_trend = bool(
        intent == "trend"
        or str(request.get("output_shape") or "").lower() == "time_series"
        or re.search(r"\b(?:trend|over\s+time|by\s+(?:day|week|month|quarter|year))\b", question, re.I)
    )
    select_parts: list[str] = []
    group_parts: list[str] = []
    if is_trend:
        if not date_ref:
            return ""
        grain = str(policy.get("requested_grain") or policy.get("unit") or "day").lower()
        if grain not in {"day", "week", "month", "quarter", "year"}:
            grain = "day"
        bucket = format_period_bucket_expression(
            date_ref, grain, db_type,
            role_alias=date_alias,
            calendar_attributes=policy.get("calendar_attributes") or {},
        )
        if not bucket:
            return ""
        select_parts.append(f"{bucket} AS PERIOD")
        group_parts.append(bucket)
    if dimension_ref:
        select_parts.append(f"{dimension_ref} AS {dimension_alias}")
        group_parts.append(dimension_ref)
    select_parts.extend(f"{formula} AS {alias}" for alias, formula in metric_specs)

    top_clause = ""
    try:
        limit = int(top_n or 0)
    except (TypeError, ValueError):
        limit = 0
    if ranking and limit > 0:
        if db_type == "azure_sql":
            top_clause = f"TOP ({limit}) "
        elif db_type not in {"snowflake", "oracle"}:
            return ""
    cross_anchor = (
        "\nCROSS JOIN anchor"
        if policy and str(policy.get("kind") or "") != "latest_n_observed"
        else ""
    )
    where_sql = "\nWHERE " + "\n  AND ".join(where_parts) if where_parts else ""
    group_sql = "\nGROUP BY " + ", ".join(group_parts) if group_parts else ""
    order_sql = ""
    if ranking:
        order_sql = f"\nORDER BY {metric_specs[0][0]} DESC"
    elif is_trend:
        order_sql = "\nORDER BY PERIOD"
    limit_sql = f"\nFETCH FIRST {limit} ROWS ONLY" if ranking and limit > 0 and db_type == "oracle" else ""
    if ranking and limit > 0 and db_type == "snowflake":
        limit_sql = f"\nLIMIT {limit}"

    compiled = (
        f"{anchor_sql}SELECT\n    {top_clause}" + ",\n    ".join(select_parts)
        + f"\nFROM {from_sql}{cross_anchor}{where_sql}{group_sql}{order_sql}{limit_sql}"
    )
    result = validate_sql_detailed(
        compiled, known_tables, db_type, allowed_tables, table_columns or {}, context,
    )
    if not result.ok:
        log.debug(
            "Governed grouped request compiler did not validate: %s (%s)",
            result.reason, result.code,
        )
        return ""
    return compiled


def build_seekable_date_window(
    *,
    db_type: str,
    fact_sql: str,
    fact_key: str,
    dim_sql: str,
    dim_key: str,
    date_col: str,
    quote: "Callable[[str], str]",
) -> tuple[str, str, str]:
    """Build an anchor and a key list a star-schema fact can SEEK on.

    The obvious shape for "latest available business date" is
    ``MAX(dim.date)`` over ``fact JOIN dim``, and the obvious window filter is
    ``dim.date BETWEEN ... AND ...``. Both read the whole fact:

      * The anchor hash-joins every fact row to the date dimension purely to
        take one MAX, with no predicate to restrict either side.
      * Filtering on a *dimension* column leaves the optimiser no way to use an
        index on the fact's own date key, so a two-day window scans years of
        invoices and then throws almost all of them away.

    On a production fact this is the difference between an index seek and a
    full scan — the live symptom was a 120 s ODBC statement timeout on
    "revenue for the last 2 days".

    This returns three fragments that keep the same semantics while making the
    fact access seekable:

      ``anchor_sql``  MAX(date) taken over the DIMENSION, restricted by an
                      EXISTS against the fact key. The semi-join can be
                      satisfied from the fact's key index alone rather than
                      materialising every row, and it still means "the latest
                      date that actually has fact rows".
      ``window_cte``  the date keys inside the requested window — a handful of
                      rows, computed entirely from the small dimension.
      ``key_filter``  a predicate on the FACT's own key column, which is what
                      turns the scan into a seek.

    The caller keeps its physical ``fact JOIN dim`` in the main query, so the
    governed entity-graph edge is still present for the validator and any
    calendar attributes stay addressable.
    """
    return (
        (
            f"SELECT MAX(anchor_date.{quote(date_col)}) AS max_business_date\n"
            f"    FROM {dim_sql} AS anchor_date\n"
            f"    WHERE EXISTS (\n"
            f"        SELECT 1 FROM {fact_sql} AS anchor_fact\n"
            f"        WHERE anchor_fact.{quote(fact_key)} = anchor_date.{quote(dim_key)}\n"
            f"    )"
        ),
        (
            f"SELECT window_date.{quote(dim_key)} AS business_date_key\n"
            f"    FROM {dim_sql} AS window_date{{anchor_join}}\n"
            f"    WHERE {{window_predicate}}"
        ),
        (
            f"fact_rows.{quote(fact_key)} IN "
            f"(SELECT business_date_key FROM window_keys)"
        ),
    )


# Calendar-period windows the compiler can express as an exact date range off
# the governed anchor. "this_week" is deliberately absent: SQL Server's
# DATEDIFF(week, …) week boundary is not the same as Oracle's TRUNC(d,'IW'),
# and a silently different first-day-of-week would move the answer. It keeps
# falling back to free-form generation until the boundary is stated somewhere
# governed rather than guessed per dialect.
_CALENDAR_PERIOD_KINDS = {
    "this_month", "this_quarter", "this_year",
    "previous_month", "previous_quarter", "previous_year",
}
_COMPILABLE_WINDOW_KINDS = {
    "last_n", "latest_n_observed", "today", "yesterday",
} | _CALENDAR_PERIOD_KINDS

# SQL Server / Snowflake / Oracle truncation to the start of a calendar period.
_PERIOD_UNIT = {"month": "month", "quarter": "quarter", "year": "year"}
_ORACLE_TRUNC_FMT = {"month": "MM", "quarter": "Q", "year": "YYYY"}


def _period_start(dialect: str, expr: str, unit: str) -> str:
    """First day of the calendar period containing `expr`."""
    if dialect in {"azure_sql", "snowflake"}:
        # DATEDIFF from the zero date then back again — the portable T-SQL
        # truncation idiom, and Snowflake accepts the same shape.
        return f"DATEADD({unit}, DATEDIFF({unit}, 0, {expr}), 0)"
    if dialect == "oracle":
        return f"TRUNC({expr}, '{_ORACLE_TRUNC_FMT[unit]}')"
    return ""


def _shift_period(dialect: str, expr: str, unit: str, periods: int) -> str:
    if dialect in {"azure_sql", "snowflake"}:
        return f"DATEADD({unit}, {periods}, {expr})"
    if dialect == "oracle":
        months = periods * {"month": 1, "quarter": 3, "year": 12}[unit]
        return f"ADD_MONTHS({expr}, {months})"
    return ""


def calendar_period_bounds(dialect: str, anchor_expr: str, kind: str) -> tuple[str, str]:
    """(start, end) SQL expressions for a calendar-period window.

    Both bounds are inclusive and derived from the governed business-date
    anchor, never from the database clock — "this month" means the month the
    DATA is in, and it ends at the anchor rather than at a future month-end
    the warehouse has no rows for.
    """
    if kind not in _CALENDAR_PERIOD_KINDS:
        return "", ""
    unit = _PERIOD_UNIT[kind.split("_", 1)[1]]
    current_start = _period_start(dialect, anchor_expr, unit)
    if not current_start:
        return "", ""
    if kind.startswith("this_"):
        # Ends at the anchor: the current period is only partly loaded, and
        # extending to its calendar end would claim coverage the data lacks.
        return current_start, anchor_expr
    previous_start = _period_start(
        dialect, _shift_period(dialect, anchor_expr, unit, -1), unit,
    )
    if not previous_start:
        return "", ""
    # The day before the current period begins — the previous period's last day
    # regardless of its length, so February and leap years need no special case.
    previous_end = (
        f"DATEADD(day, -1, {current_start})"
        if dialect in {"azure_sql", "snowflake"}
        else f"{current_start} - 1"
    )
    return previous_start, previous_end


def compile_governed_temporal_metric_sql(
    db_type: str,
    known_tables: set[str],
    allowed_tables: set[str] | None,
    table_columns: dict[str, dict[str, str]] | None,
    semantic_context: dict | None,
) -> str:
    """Compile a simple governed metric/window request without an LLM.

    The metric registry and an approved Date Role already contain every
    physical decision needed for requests such as ``total revenue for the
    last 6 months`` and ``revenue trend for the last 7 days``.  Sending that
    fully-resolved case back through free-form SQL generation introduces
    needless failure modes (wrong date key, calendar-table anchoring,
    truncated CTEs and redundant aggregate-of-aggregate wrappers).

    The scalar branch remains intentionally narrow. Governed grouped/ranked,
    exact event-count and same-fact multi-metric requests are dispatched to a
    separate conservative compiler above. Anything requiring an unresolved
    relationship, distribution or multiple facts remains on the normal
    analytical planner path.
    """
    from core.contextual_dates import format_period_bucket_expression
    from core.validator import validate_sql_detailed

    grouped = _compile_governed_grouped_request_sql(
        db_type,
        known_tables,
        allowed_tables,
        table_columns,
        semantic_context,
    )
    if grouped:
        return grouped

    context = semantic_context or {}
    plan = context.get("semantic_plan") or {}
    request_plan = context.get("analytical_request_plan") or {}
    question = str(context.get("question") or "")
    policies = list(plan.get("temporal_policies") or [])
    metrics = [
        metric for metric in (context.get("metric_formulas") or [])
        if str(metric.get("formula_type") or "query").lower() == "expression"
        and str(metric.get("sql_template") or metric.get("formula") or "").strip()
    ]
    if len(policies) != 1 or len(metrics) != 1:
        return ""
    if request_plan and str(request_plan.get("status") or "") != "compiled":
        return ""
    if request_plan.get("subrequests") or len(request_plan.get("source_facts") or []) > 1:
        return ""
    if re.search(
        r"\b(?:compare|comparison|versus|vs\.?|difference|change|rank|top|bottom|"
        r"distribution|share|percentile|correlation|why|forecast)\b",
        question,
        re.I,
    ):
        return ""

    # A requested non-date display field changes the result grain.  It must be
    # handled by the full semantic join planner, not this scalar/time-series
    # compiler.
    if any(
        field.get("display_required") and field.get("enforcement") != "optional"
        for field in (plan.get("fields") or [])
    ):
        return ""

    policy = policies[0]
    window_kind = str(policy.get("kind") or "")
    if window_kind not in _COMPILABLE_WINDOW_KINDS:
        return ""
    try:
        amount = int(policy.get("amount"))
    except (TypeError, ValueError):
        return ""
    unit = str(policy.get("unit") or "").lower()
    if unit not in {"day", "week", "month", "quarter", "year"}:
        return ""
    if window_kind in {"last_n", "latest_n_observed"} and amount <= 0:
        return ""

    fact_table = str(policy.get("fact_table") or policy.get("anchor_table") or "")
    fact_column = str(policy.get("fact_column") or policy.get("anchor_column") or "")
    dimension_table = str(policy.get("dimension_table") or "")
    dimension_key = str(policy.get("dimension_key") or "")
    date_column = str(policy.get("date_column") or "")
    date_key_type = str(policy.get("date_key_type") or "")
    role_alias = re.sub(
        r"[^A-Za-z0-9_]", "_", str(policy.get("role_alias") or "business_date")
    )
    if not fact_table or not fact_column or not date_column:
        return ""
    if date_key_type == "surrogate_fk" and not (dimension_table and dimension_key):
        return ""

    metric = metrics[0]
    formula = str(metric.get("sql_template") or metric.get("formula") or "").strip().rstrip(";")
    if not formula or re.search(r"\b(?:SELECT|FROM|WITH)\b|;", formula, re.I):
        return ""

    metric_sources = list(
        metric.get("_resolved_source_tables")
        or metric.get("source_tables")
        or ([metric.get("base_table")] if metric.get("base_table") else [])
    )
    if metric_sources and not any(
        str(source).split(".")[-1].upper() == fact_table.split(".")[-1].upper()
        for source in metric_sources
    ):
        return ""

    def qcol(name: str) -> str:
        clean = str(name or "").strip().strip("[]\"`")
        if db_type == "azure_sql":
            return f"[{clean}]"
        if db_type in {"snowflake", "oracle"}:
            return f'"{clean}"'
        return clean

    fact_sql = _quote_table_for_count(fact_table, db_type)
    if date_key_type == "surrogate_fk":
        dim_sql = _quote_table_for_count(dimension_table, db_type)
        from_sql = (
            f"{fact_sql} AS fact_rows\n"
            f"    LEFT JOIN {dim_sql} AS {role_alias}\n"
            f"      ON fact_rows.{qcol(fact_column)} = {role_alias}.{qcol(dimension_key)}"
        )
        date_ref = f"{role_alias}.{qcol(date_column)}"
    else:
        from_sql = f"{fact_sql} AS fact_rows"
        date_ref = f"fact_rows.{qcol(date_column)}"

    dialect = str(db_type or "").lower()

    # A surrogate-key star gets a seekable shape: the window is resolved to date
    # KEYS off the small dimension, and the fact is filtered on its own key.
    # Filtering the dimension's date column instead leaves the optimiser no way
    # to use the fact's date-key index, which is what made a two-day revenue
    # question scan the whole invoice fact and hit the statement timeout.
    seekable = date_key_type == "surrogate_fk"
    anchor_body = window_keys_body = key_filter = ""
    # A pre-resolved anchor lets the window become two literals, so nothing has
    # to read the fact just to discover its newest date. Only accepted when it
    # was probed for THIS fact and date key — see core.date_anchor.
    from core.date_anchor import anchor_for_policy

    resolved_anchor = anchor_for_policy(
        context.get("resolved_date_anchor"), policy,
    )
    literal_window_predicate = ""
    if seekable:
        anchor_body, window_keys_body, key_filter = build_seekable_date_window(
            db_type=db_type,
            fact_sql=fact_sql,
            fact_key=fact_column,
            dim_sql=_quote_table_for_count(dimension_table, db_type),
            dim_key=dimension_key,
            date_col=date_column,
            quote=qcol,
        )
        # Inside window_keys the dimension is aliased window_date, so the window
        # predicate must be expressed against that alias, not the display join.
        window_date_ref = f"window_date.{qcol(date_column)}"
    else:
        window_date_ref = date_ref

    observed_period_cte = ""
    if window_kind == "latest_n_observed":
        if unit != "day":
            return ""
        if dialect == "azure_sql":
            observed_limit = f"TOP ({amount}) "
            observed_suffix = ""
        elif dialect == "snowflake":
            observed_limit = ""
            observed_suffix = f"\n    LIMIT {amount}"
        elif dialect == "oracle":
            observed_limit = ""
            observed_suffix = f"\n    FETCH FIRST {amount} ROWS ONLY"
        else:
            return ""
        # Deliberately NOT re-shaped for seeks. `observed_period_shape` in the
        # validator requires the observed days to be selected from the FACT
        # ("TOP N DISTINCT <date> ... from <fact>"), so moving them onto the
        # dimension would need that governed contract relaxed too. This window
        # keeps its established, validated shape; `last_n`, `today` and
        # `yesterday` below carry the seek rewrite.
        observed_period_cte = f"""WITH observed_periods AS (
    SELECT DISTINCT {observed_limit}{date_ref} AS observed_business_date
    FROM {from_sql}
    WHERE {date_ref} IS NOT NULL
    ORDER BY {date_ref} DESC{observed_suffix}
)
"""
        where_sql = f"{date_ref} IN (SELECT observed_business_date FROM observed_periods)"
    elif window_kind == "last_n":
        if dialect in {"azure_sql", "snowflake"}:
            start_expr = f"DATEADD({unit}, -{amount}, anchor.max_business_date)"
        elif dialect == "oracle":
            if unit == "day":
                start_expr = f"anchor.max_business_date - {amount}"
            elif unit == "week":
                start_expr = f"anchor.max_business_date - {amount * 7}"
            else:
                months = amount * {"month": 1, "quarter": 3, "year": 12}[unit]
                start_expr = f"ADD_MONTHS(anchor.max_business_date, -{months})"
        else:
            return ""
        window_predicate = (
            f"{window_date_ref} > {start_expr}\n"
            f"      AND {window_date_ref} <= anchor.max_business_date"
        )
        if resolved_anchor.get("value"):
            _anchor_literal = f"CAST('{resolved_anchor['value']}' AS date)"
            if dialect in {"azure_sql", "snowflake"}:
                _start_literal = f"DATEADD({unit}, -{amount}, {_anchor_literal})"
            elif dialect == "oracle":
                if unit == "day":
                    _start_literal = f"{_anchor_literal} - {amount}"
                elif unit == "week":
                    _start_literal = f"{_anchor_literal} - {amount * 7}"
                else:
                    _months = amount * {"month": 1, "quarter": 3, "year": 12}[unit]
                    _start_literal = f"ADD_MONTHS({_anchor_literal}, -{_months})"
            else:
                _start_literal = ""
            if _start_literal:
                literal_window_predicate = (
                    f"{window_date_ref} > {_start_literal}\n"
                    f"      AND {window_date_ref} <= {_anchor_literal}"
                )
        where_sql = key_filter if seekable else (
            f"{date_ref} > {start_expr}\n"
            f"  AND {date_ref} <= anchor.max_business_date"
        )
    elif window_kind in _CALENDAR_PERIOD_KINDS:
        # "this month" / "last quarter" and friends. These were rejected
        # outright, so every calendar-period question left the governed
        # contract for free-form SQL — the one shape most likely to get the
        # boundary subtly wrong, and the reason those answers needed repair
        # retries that then capped their confidence.
        _start, _end = calendar_period_bounds(
            dialect, "anchor.max_business_date", window_kind,
        )
        if not _start or not _end:
            return ""
        window_predicate = (
            f"{window_date_ref} >= {_start}\n"
            f"      AND {window_date_ref} <= {_end}"
        )
        where_sql = key_filter if seekable else (
            f"{date_ref} >= {_start}\n"
            f"  AND {date_ref} <= {_end}"
        )
        if resolved_anchor.get("value"):
            _anchor_literal = f"CAST('{resolved_anchor['value']}' AS date)"
            _lit_start, _lit_end = calendar_period_bounds(
                dialect, _anchor_literal, window_kind,
            )
            if _lit_start and _lit_end:
                literal_window_predicate = (
                    f"{window_date_ref} >= {_lit_start}\n"
                    f"      AND {window_date_ref} <= {_lit_end}"
                )
    else:
        if dialect == "azure_sql":
            selected_day = (
                "CAST(anchor.max_business_date AS date)"
                if window_kind == "today"
                else "DATEADD(day, -1, CAST(anchor.max_business_date AS date))"
            )
            date_day = f"CAST({date_ref} AS date)"
            window_day = f"CAST({window_date_ref} AS date)"
        elif dialect == "snowflake":
            selected_day = (
                "CAST(anchor.max_business_date AS date)"
                if window_kind == "today"
                else "DATEADD(day, -1, CAST(anchor.max_business_date AS date))"
            )
            date_day = f"CAST({date_ref} AS date)"
            window_day = f"CAST({window_date_ref} AS date)"
        elif dialect == "oracle":
            selected_day = (
                "TRUNC(anchor.max_business_date)"
                if window_kind == "today"
                else "TRUNC(anchor.max_business_date) - 1"
            )
            date_day = f"TRUNC({date_ref})"
            window_day = f"TRUNC({window_date_ref})"
        else:
            return ""
        window_predicate = f"{window_day} = {selected_day}"
        where_sql = key_filter if seekable else f"{date_day} = {selected_day}"
        if resolved_anchor.get("value"):
            _anchor_literal = f"CAST('{resolved_anchor['value']}' AS date)"
            if dialect == "oracle":
                _selected_literal = (
                    _anchor_literal if window_kind == "today"
                    else f"{_anchor_literal} - 1"
                )
            else:
                _selected_literal = (
                    _anchor_literal if window_kind == "today"
                    else f"DATEADD(day, -1, {_anchor_literal})"
                )
            literal_window_predicate = f"{window_day} = {_selected_literal}"

    metric_alias = (
        re.sub(r"[^A-Za-z0-9_]", "_", str(metric.get("name") or "metric")).upper()
        or "METRIC"
    )
    is_trend = bool(
        str(request_plan.get("intent") or "").lower() == "trend"
        or str(request_plan.get("output_shape") or "").lower() == "time_series"
        or re.search(r"\b(?:trend|over\s+time|by\s+(?:day|week|month|quarter|year))\b", question, re.I)
    )
    select_sql = f"{formula} AS {metric_alias}"
    group_sql = ""
    order_sql = ""
    if is_trend:
        grain = str(policy.get("requested_grain") or unit).lower()
        if grain not in {"day", "week", "month", "quarter", "year"}:
            grain = unit
        bucket = format_period_bucket_expression(
            date_ref,
            grain,
            db_type,
            role_alias=role_alias,
            calendar_attributes=policy.get("calendar_attributes") or {},
        )
        if not bucket:
            return ""
        select_sql = f"{bucket} AS PERIOD,\n    {formula} AS {metric_alias}"
        group_sql = f"\nGROUP BY {bucket}"
        order_sql = "\nORDER BY PERIOD"

    if window_kind == "latest_n_observed":
        compiled = f"""{observed_period_cte}SELECT
    {select_sql}
FROM {from_sql}
WHERE {where_sql}{group_sql}{order_sql}"""
    elif seekable and resolved_anchor.get("value"):
        # The anchor was already resolved from the fact's own rows once for this
        # account+fact+key and cached, so the query does not have to ask the
        # database "what is the newest date here?" all over again. Nothing reads
        # the fact to compute an anchor: the window is two literals and the fact
        # is reached through its own key. This is the shape that survives a large
        # fact with no index on the date key.
        window_keys_sql = window_keys_body.replace(
            "{anchor_join}", "",
        ).replace("{window_predicate}", literal_window_predicate)
        compiled = f"""WITH window_keys AS (
    {window_keys_sql}
)
SELECT
    {select_sql}
FROM {from_sql}
WHERE {where_sql}{group_sql}{order_sql}"""
    elif seekable:
        # anchor and window_keys are both computed off the small date dimension.
        # The main query keeps its physical fact-to-dimension join (so the
        # governed graph edge is present and calendar attributes stay
        # addressable) but reaches the fact through its own key, which is what
        # makes this an index seek instead of a full scan.
        window_keys_sql = window_keys_body.replace(
            "{anchor_join}", "\n    CROSS JOIN anchor",
        ).replace("{window_predicate}", window_predicate)
        compiled = f"""WITH anchor AS (
    {anchor_body}
),
window_keys AS (
    {window_keys_sql}
)
SELECT
    {select_sql}
FROM {from_sql}
WHERE {where_sql}{group_sql}{order_sql}"""
    elif resolved_anchor.get("value") and literal_window_predicate:
        # A fact-native date with a resolved anchor is the cheapest shape there
        # is: the window is two literals against the fact's own date column, so
        # there is no anchor CTE, no window_keys CTE and no dimension to reach.
        # This branch previously required `seekable`, so for a native-date role
        # the literal window was computed and then silently discarded and the
        # full-scan anchor came back — the cache bought nothing for exactly the
        # configuration it helps most.
        compiled = f"""SELECT
    {select_sql}
FROM {from_sql}
WHERE {literal_window_predicate}{group_sql}{order_sql}"""
    else:
        compiled = f"""WITH anchor AS (
    SELECT MAX({date_ref}) AS max_business_date
    FROM {from_sql}
)
SELECT
    {select_sql}
FROM {from_sql}
CROSS JOIN anchor
WHERE {where_sql}{group_sql}{order_sql}"""

    result = validate_sql_detailed(
        compiled,
        known_tables,
        db_type,
        allowed_tables,
        table_columns or {},
        semantic_context,
    )
    if not result.ok:
        log.debug(
            "Governed rolling metric compiler did not validate: %s (%s)",
            result.reason,
            result.code,
        )
        return ""
    return compiled


def attempt_governed_temporal_metric_repair(
    sql: str,
    db_type: str,
    known_tables: set[str],
    allowed_tables: set[str] | None,
    table_columns: dict[str, dict[str, str]] | None,
    semantic_context: dict | None,
) -> str:
    """Compile a standard governed period comparison from executable metadata.

    This is deliberately conservative: one approved expression metric, one
    selected Date Role, and a current-vs-previous period request.  Those inputs
    fully determine the SQL, so retrying an LLM that already ignored the role
    is unnecessary and unsafe.  More complex dimensional comparisons continue
    through the normal planner/retry path.
    """
    from core.contextual_dates import format_period_bucket_expression
    from core.validator import validate_sql_detailed

    context = semantic_context or {}
    plan = context.get("semantic_plan") or {}
    question = str(context.get("question") or "")
    policies = list(plan.get("temporal_policies") or [])
    metrics = [
        metric for metric in (context.get("metric_formulas") or [])
        if str(metric.get("formula_type") or "query").lower() == "expression"
        and str(metric.get("sql_template") or "").strip()
    ]
    if len(policies) != 1 or len(metrics) != 1:
        return ""
    if not re.search(r"\b(?:compare|comparison|versus|vs\.?|difference|change)\b", question, re.I):
        return ""
    if not re.search(r"\b(?:this|current)\s+(?:week|month|quarter|year)\b", question, re.I):
        return ""
    if not re.search(r"\b(?:last|previous|prior)\s+(?:week|month|quarter|year)\b", question, re.I):
        return ""

    # A required non-date display dimension changes the requested grain and
    # cannot be safely synthesized by this narrow compiler.
    if any(
        field.get("display_required") and field.get("enforcement") != "optional"
        for field in (plan.get("fields") or [])
    ):
        return ""

    policy = policies[0]
    fact_table = str(policy.get("fact_table") or policy.get("anchor_table") or "")
    fact_column = str(policy.get("fact_column") or policy.get("anchor_column") or "")
    dimension_table = str(policy.get("dimension_table") or "")
    dimension_key = str(policy.get("dimension_key") or "")
    date_column = str(policy.get("date_column") or "")
    date_key_type = str(policy.get("date_key_type") or "")
    role_alias = re.sub(r"[^A-Za-z0-9_]", "_", str(policy.get("role_alias") or "business_date"))
    if not fact_table or not fact_column or not date_column:
        return ""
    if date_key_type == "surrogate_fk" and not (dimension_table and dimension_key):
        return ""

    metric = metrics[0]
    formula = str(metric.get("sql_template") or "").strip().rstrip(";")
    # Query/template metrics and multi-statement fragments are intentionally
    # outside this compiler's authority.
    if not formula or re.search(r"\b(?:SELECT|FROM|WITH)\b|;", formula, re.I):
        return ""

    grain_match = re.search(r"\b(?:this|current)\s+(week|month|quarter|year)\b", question, re.I)
    grain = str(policy.get("requested_grain") or (grain_match.group(1) if grain_match else "month")).lower()
    if grain not in {"week", "month", "quarter", "year"}:
        return ""

    def qcol(name: str) -> str:
        clean = str(name or "").strip().strip("[]\"`")
        if db_type == "azure_sql":
            return f"[{clean}]"
        if db_type in {"snowflake", "oracle"}:
            return f'"{clean}"'
        return clean

    fact_sql = _quote_table_for_count(fact_table, db_type)
    dim_sql = _quote_table_for_count(dimension_table, db_type) if dimension_table else ""
    if date_key_type == "surrogate_fk":
        from_sql = (
            f"{fact_sql} AS fact_rows\n"
            f"    LEFT JOIN {dim_sql} AS {role_alias}\n"
            f"      ON fact_rows.{qcol(fact_column)} = {role_alias}.{qcol(dimension_key)}"
        )
        date_ref = f"{role_alias}.{qcol(date_column)}"
    else:
        from_sql = f"{fact_sql} AS fact_rows"
        date_ref = f"fact_rows.{qcol(date_column)}"

    bucket = format_period_bucket_expression(
        date_ref,
        grain,
        db_type,
        role_alias=role_alias,
        calendar_attributes=policy.get("calendar_attributes") or {},
    )
    dialect = str(db_type or "azure_sql").lower()
    if dialect == "azure_sql":
        base_bucket = {
            "week": "DATEADD(day, 1 - DATEPART(weekday, anchor.max_business_date), CAST(anchor.max_business_date AS date))",
            "month": "DATEFROMPARTS(YEAR(anchor.max_business_date), MONTH(anchor.max_business_date), 1)",
            "quarter": "DATEFROMPARTS(YEAR(anchor.max_business_date), ((DATEPART(quarter, anchor.max_business_date) - 1) * 3) + 1, 1)",
            "year": "DATEFROMPARTS(YEAR(anchor.max_business_date), 1, 1)",
        }[grain]
        start_expr = f"DATEADD({grain}, -1, {base_bucket})"
        end_expr = f"DATEADD({grain}, 1, {base_bucket})"
    elif dialect == "snowflake":
        base_bucket = f"DATE_TRUNC('{grain}', anchor.max_business_date)"
        start_expr = f"DATEADD({grain}, -1, {base_bucket})"
        end_expr = f"DATEADD({grain}, 1, {base_bucket})"
    elif dialect == "oracle":
        trunc_fmt = {"week": "IW", "month": "MM", "quarter": "Q", "year": "YYYY"}[grain]
        base_bucket = f"TRUNC(anchor.max_business_date, '{trunc_fmt}')"
        months = {"month": 1, "quarter": 3, "year": 12}.get(grain)
        if months:
            start_expr = f"ADD_MONTHS({base_bucket}, -{months})"
            end_expr = f"ADD_MONTHS({base_bucket}, {months})"
        else:
            start_expr = f"{base_bucket} - 7"
            end_expr = f"{base_bucket} + 7"
    else:
        return ""

    metric_alias = re.sub(r"[^A-Za-z0-9_]", "_", str(metric.get("name") or "metric")).upper() or "METRIC"
    compiled = f"""WITH anchor AS (
    SELECT MAX({date_ref}) AS max_business_date
    FROM {from_sql}
),
period_totals AS (
    SELECT
        {bucket} AS period_start,
        {formula} AS {metric_alias}
    FROM {from_sql}
    CROSS JOIN anchor
    WHERE {date_ref} >= {start_expr}
      AND {date_ref} < {end_expr}
    GROUP BY {bucket}
),
period_comparison AS (
    SELECT
        period_start,
        {metric_alias},
        LAG({metric_alias}) OVER (ORDER BY period_start) AS previous_{metric_alias}
    FROM period_totals
)
SELECT
    period_start,
    {metric_alias},
    previous_{metric_alias},
    {metric_alias} - previous_{metric_alias} AS absolute_change,
    100.0 * ({metric_alias} - previous_{metric_alias}) / NULLIF(previous_{metric_alias}, 0) AS percent_change
FROM period_comparison
ORDER BY period_start"""

    result = validate_sql_detailed(
        compiled,
        known_tables,
        db_type,
        allowed_tables,
        table_columns or {},
        semantic_context,
    )
    if not result.ok:
        log.debug("Governed temporal compiler did not validate: %s (%s)", result.reason, result.code)
        return ""
    return compiled


def attempt_field_plan_repair(
    sql: str,
    db_type: str,
    known_tables: set[str],
    allowed_tables: set[str] | None,
    table_columns: dict[str, dict[str, str]] | None,
    semantic_context: dict | None,
) -> str:
    """
    Deterministically repair a field_plan_mismatch failure without an LLM call.

    The most common plan failure is fully mechanical: the plan requires a
    display dimension (e.g. customer → CUS_DMS.CUS_NM) but the LLM grouped by
    the surrogate key (CUS_DMS_KEY) instead. Everything needed to fix that is
    already in the plan — the display table, display column, and join key — so
    rewrite the SQL directly: add the missing dimension join and swap the key
    column for the display column in SELECT / GROUP BY / ORDER BY.

    Returns the repaired SQL when the rewrite re-validates cleanly, else "".
    Three mechanical cases are attempted — display-field mismatches (and
    their join edges), superseded-column swaps, and missing required MEASURE
    swaps when the SQL used exactly one non-plan sibling measure on the same
    table (e.g. plan requires CUS_IVC_LIN_AMT, the LLM summed
    SOP_CUS_IVC_LIN_AMT). Anything else bails out to the normal LLM retry.
    """
    try:
        import sqlglot
        from sqlglot import exp as sg_exp
    except ImportError:
        return ""
    from core.validator import validate_sql_detailed, _table_matches

    plan = (semantic_context or {}).get("semantic_plan") or {}
    if not plan.get("enabled") or not plan.get("fields"):
        return ""

    result = validate_sql_detailed(
        sql, known_tables, db_type, allowed_tables, table_columns, semantic_context
    )
    if result.ok or result.code != "field_plan_mismatch":
        return ""

    plan_fields = {
        ((f.get("table") or "").upper(), (f.get("column") or "").upper()): f
        for f in plan.get("fields") or []
    }
    # Superseded-column violations first: swapping the old rival for the
    # admin-approved column also satisfies the "missing required approved
    # field" error the same SQL usually raises alongside it.
    avoided_swaps: list[dict] = [
        err for err in result.errors
        if err.get("code") == "field_plan_mismatch"
        and err.get("avoided_column") and err.get("use_instead_column")
    ]
    for err in avoided_swaps:
        # Only same-table swaps are mechanical; cross-table redirection
        # changes join shape — leave to the LLM retry, which now carries
        # the explicit supersession message.
        if not _table_matches(
            err.get("use_instead_table") or "", err.get("avoided_table") or ""
        ):
            return ""

    missing_display: list[dict] = []
    missing_measures: list[dict] = []
    display_tables: set[str] = set()
    for err in result.errors:
        if err.get("code") == "field_plan_mismatch":
            if err.get("avoided_column") and err.get("use_instead_column"):
                continue    # already collected above
            if any(
                (err.get("column") or "").upper() == (a.get("use_instead_column") or "").upper()
                and _table_matches(err.get("table") or "", a.get("use_instead_table") or "")
                for a in avoided_swaps
            ):
                continue    # the swap below will introduce this required column
            f = plan_fields.get(
                ((err.get("table") or "").upper(), (err.get("column") or "").upper())
            )
            # Plan table may differ from the validator-resolved table key —
            # fall back to matching by column + display flag.
            if f is None:
                col_u = (err.get("column") or "").upper()
                f = next(
                    (
                        pf for (pt, pc), pf in plan_fields.items()
                        if pc == col_u and _table_matches(pt, err.get("table") or "")
                    ),
                    None,
                )
            if f is None:
                return ""
            if f.get("display_required") and f.get("source_key_column"):
                missing_display.append(f)
                display_tables.add((f.get("table") or "").upper())
            elif str(f.get("role") or "").lower() == "measure":
                missing_measures.append(f)
            else:
                return ""
        elif err.get("code") == "field_plan_join_missing":
            # Repairable only when the missing join reaches a display table we
            # are about to add anyway.
            if (err.get("right_table") or "").upper() not in display_tables and (
                (err.get("left_table") or "").upper() not in display_tables
            ):
                return ""
        else:
            return ""
    if not missing_display and not avoided_swaps and not missing_measures:
        return ""

    dialect = _REPAIR_DIALECT.get(db_type)
    try:
        tree = sqlglot.parse_one(sql, dialect=dialect)
    except Exception:
        return ""

    def _node_matches_table(node, table_name: str) -> bool:
        parts = [p for p in [
            str(node.catalog or ""), str(node.db or ""), str(node.name or "")
        ] if p]
        return bool(parts) and _table_matches(".".join(parts), table_name)

    def _has_ancestor(node, kinds: tuple[type, ...]) -> bool:
        parent = getattr(node, "parent", None)
        while parent is not None:
            if isinstance(parent, kinds):
                return True
            parent = getattr(parent, "parent", None)
        return False

    def _inside_join_or_filter(node) -> bool:
        """Return True for structural predicates that repair must never edit."""
        return _has_ancestor(node, (sg_exp.Join, sg_exp.Where))

    def _output_or_group_column(node) -> bool:
        """Allow display/supersession swaps only in presentation contexts.

        A deterministic field repair may change a selected/grouped business
        field, but never a relationship predicate or row filter.  ORDER BY is
        intentionally included because it commonly repeats the selected
        display expression.
        """
        if _inside_join_or_filter(node):
            return False
        return _has_ancestor(node, (sg_exp.Select, sg_exp.Group, sg_exp.Order, sg_exp.Having))

    def _aggregate_measure_column(node) -> bool:
        """Only aggregate arguments are eligible for a measure substitution."""
        if _inside_join_or_filter(node) or _has_ancestor(node, (sg_exp.Group,)):
            return False
        return _has_ancestor(node, (sg_exp.AggFunc,)) and _has_ancestor(
            node, (sg_exp.Select, sg_exp.Having)
        )

    changed = False

    # ── Superseded-column swaps (avoided → admin-approved, same table) ────────
    for err in avoided_swaps:
        avoid_table = err.get("avoided_table") or ""
        avoid_col = (err.get("avoided_column") or "").upper()
        new_col = (err.get("use_instead_column") or "").upper()
        if not avoid_col or not new_col:
            return ""
        aliases = {
            str(t.alias_or_name or t.name or "").upper()
            for t in tree.find_all(sg_exp.Table)
            if _node_matches_table(t, avoid_table)
        }
        replaced_here = False
        for col_node in list(tree.find_all(sg_exp.Column)):
            if (col_node.name or "").upper() != avoid_col:
                continue
            if not _output_or_group_column(col_node):
                continue
            tbl_ref = (col_node.table or "").upper()
            if tbl_ref and aliases and tbl_ref not in aliases:
                continue
            replacement = (
                sg_exp.column(new_col, table=col_node.table)
                if col_node.table else sg_exp.column(new_col)
            )
            col_node.replace(replacement)
            replaced_here = True
        if not replaced_here:
            return ""
        changed = True

    # ── Missing required MEASURE swaps ────────────────────────────────────────
    # The plan required a measure column but the LLM aggregated a sibling on
    # the same table (defect: 'sales amount' plan-bound to CUS_IVC_LIN_AMT,
    # SQL summed SOP_CUS_IVC_LIN_AMT). Mechanical only when exactly ONE
    # non-plan measure-shaped column from that table appears in the SQL —
    # any ambiguity goes to the LLM retry with the repair note instead.
    for field in missing_measures:
        req_table = field.get("table") or ""
        req_col = (field.get("column") or "").upper()
        table_cols_typed: dict[str, str] = {}
        for tk, cols in (table_columns or {}).items():
            if _table_matches(tk, req_table):
                table_cols_typed = {
                    str(c).upper(): str(t or "") for c, t in (cols or {}).items()
                }
                break
        if req_col not in table_cols_typed:
            return ""
        aliases = {
            str(t.alias_or_name or t.name or "").upper()
            for t in tree.find_all(sg_exp.Table)
            if _node_matches_table(t, req_table)
        }
        if not aliases:
            return ""
        plan_cols_for_table = {
            pc for (pt, pc) in plan_fields if _table_matches(pt, req_table)
        }

        def _measure_shaped(name: str, ctype: str) -> bool:
            # Surrogate/business/date keys are identifiers even when physically
            # stored as integers.  Treating *_SK as a measure is what changed
            # ORDER_SK = ORDER_SK into ORDER_SK = SHIPPED_AMOUNT in production.
            if name.endswith((
                "_SK", "_KEY", "_FK", "_ID", "_CD", "_CODE", "_NUM",
                "_NO", "_NBR", "_YYYYMM", "_YYYYMMDD", "_DMS_KEY",
            )):
                return False
            if any(token in name for token in ("DATE_KEY", "DATE_SK", "DT_DMS_KEY")):
                return False
            base = ctype.lower().split("(")[0].strip()
            return any(t in base for t in (
                "decimal", "numeric", "money", "float", "real", "int", "number",
            ))

        candidates: set[str] = set()
        for col_node in tree.find_all(sg_exp.Column):
            if not _aggregate_measure_column(col_node):
                continue
            name = (col_node.name or "").upper()
            tbl_ref = (col_node.table or "").upper()
            if tbl_ref and tbl_ref not in aliases:
                continue
            if not name or name == req_col or name in plan_cols_for_table:
                continue
            ctype = table_cols_typed.get(name)
            if ctype is None or not _measure_shaped(name, ctype):
                continue
            candidates.add(name)
        if len(candidates) != 1:
            return ""
        wrong_col = candidates.pop()
        replaced_measure = False
        for col_node in list(tree.find_all(sg_exp.Column)):
            if (col_node.name or "").upper() != wrong_col:
                continue
            if not _aggregate_measure_column(col_node):
                continue
            tbl_ref = (col_node.table or "").upper()
            if tbl_ref and tbl_ref not in aliases:
                continue
            col_node.replace(
                sg_exp.column(req_col, table=col_node.table)
                if col_node.table else sg_exp.column(req_col)
            )
            replaced_measure = True
        if not replaced_measure:
            return ""
        changed = True

    for field in missing_display:
        display_table = field.get("table") or ""
        display_col = (field.get("column") or "").upper()
        key_col = (field.get("source_key_column") or "").upper()
        source_table = field.get("source_key_table") or field.get("source_table") or ""

        # Locate the SELECT whose scope contains the source (fact) table.
        target_select = None
        source_node = None
        for select_node in tree.find_all(sg_exp.Select):
            for tbl in select_node.find_all(sg_exp.Table):
                if source_table and _node_matches_table(tbl, source_table):
                    target_select, source_node = select_node, tbl
                    break
                for tk, cols in (table_columns or {}).items():
                    if _node_matches_table(tbl, tk) and key_col in {
                        str(c).upper() for c in (cols or {})
                    }:
                        target_select, source_node = select_node, tbl
                        break
                if source_node is not None:
                    break
            if source_node is not None:
                break
        if target_select is None or source_node is None:
            return ""
        src_alias = source_node.alias_or_name or source_node.name

        # Is the display table already joined in this SELECT?
        disp_alias = ""
        for tbl in target_select.find_all(sg_exp.Table):
            if _node_matches_table(tbl, display_table):
                disp_alias = tbl.alias_or_name or tbl.name
                break
        if not disp_alias:
            bare = display_table.split(".")[-1]
            base_alias = re.sub(r"[^a-z]", "", bare.lower())[:3] or "d"
            existing = {
                (t.alias_or_name or t.name or "").lower()
                for t in target_select.find_all(sg_exp.Table)
            }
            disp_alias = base_alias
            n = 2
            while disp_alias in existing:
                disp_alias = f"{base_alias}{n}"
                n += 1
            join_frag = (
                f"SELECT 1 FROM t JOIN {display_table} AS {disp_alias} "
                f"ON {src_alias}.{key_col} = {disp_alias}.{key_col}"
            )
            try:
                join_expr = sqlglot.parse_one(join_frag, dialect=dialect).find(sg_exp.Join)
            except Exception:
                return ""
            if join_expr is None:
                return ""
            target_select.append("joins", join_expr)
            changed = True

        # Swap the surrogate key for the display column in projection,
        # GROUP BY, and ORDER BY (never in JOIN ON / WHERE).
        swap_scopes = list(target_select.expressions)
        for arg in ("group", "order"):
            scope = target_select.args.get(arg)
            if scope is not None:
                swap_scopes.append(scope)
        replaced_here = False
        for scope in swap_scopes:
            for col_node in list(scope.find_all(sg_exp.Column)):
                if (col_node.name or "").upper() != key_col:
                    continue
                tbl_ref = (col_node.table or "")
                if tbl_ref and tbl_ref.upper() != str(src_alias).upper():
                    continue
                col_node.replace(sg_exp.column(display_col, table=disp_alias))
                replaced_here = True
        if not replaced_here:
            # Key column not projected/grouped at all — adding the display
            # column would change the query grain; leave that to the LLM.
            return ""
        changed = True

    if not changed:
        return ""

    try:
        repaired = tree.sql(dialect=dialect)
    except Exception:
        return ""

    recheck = validate_sql_detailed(
        repaired, known_tables, db_type, allowed_tables, table_columns, semantic_context
    )
    if not recheck.ok:
        return ""
    log.info("Deterministic field-plan repair applied (no LLM retry needed)")
    return repaired


# ── Cross-channel reuse staleness guard ───────────────────────────────────────

def reused_plan_is_stale_for_graph(sql: str, graph_ctx: dict | None, db_type: str) -> bool:
    """True when a cached SQL plan's referenced tables no longer match what
    the CURRENT (freshly re-run) entity-graph resolution says this question
    needs.

    store.find_reusable_validated_sql_plan() only checks question text +
    schema + tables + contract_version — none of which change when the
    resolution/validation CODE changes (only compiled semantic DATA bumps
    contract_version). A plan cached before a resolver/validator fix shipped
    (e.g. a fan-out join a since-tightened detect_entities() would no longer
    produce) can otherwise be reused indefinitely, silently bypassing the fix
    for that exact question forever.

    Conservative by design: only rejects reuse when there's a real, non-empty
    current detection to compare against, and only flags a table as "extra"
    when it's a table the graph actually KNOWS about (a real entity) that
    just isn't in the current detected set — never a CTE alias or subquery
    name from the cached SQL, which extract_sql_tables does not distinguish
    from real tables. Any error leaves existing reuse behavior unchanged.
    """
    graph_ctx = graph_ctx or {}
    if graph_ctx.get("resolution_error"):
        # Resolution raised rather than reporting "no graph", so there is no
        # current detection to compare a cached plan against. Reading that as
        # "nothing to object to" is how a plan whose joins the current resolver
        # would reject gets reused precisely when governance is down. Fresh
        # generation still runs, and still has to pass the validator.
        return True
    if graph_ctx.get("review_only"):
        # No unreviewed edge may become executable indirectly through an old
        # cached plan. Fresh generation can still use the governed KB and
        # semantic model, but it must do so without inheriting historical graph
        # joins that the current resolver deliberately excluded.
        return True
    if not graph_ctx.get("enabled"):
        return False
    detected = set(graph_ctx.get("detected") or [])
    if not detected:
        return False

    entities = graph_ctx.get("entities") or []
    entity_tables = {
        (ent.get("table_name") or "").strip().upper()
        for ent in entities
        if ent.get("table_name")
    }
    expected_tables = {
        (ent.get("table_name") or "").strip().upper()
        for ent in entities
        if ent.get("entity_name") in detected and ent.get("table_name")
    }
    if not expected_tables:
        return False

    try:
        actual_tables = {
            str(t).split(".")[-1].strip().upper()
            for t in extract_sql_tables(sql, db_type)
        }
    except Exception:
        return False

    # Only tables the graph recognizes as real entities count — this excludes
    # CTE names and other non-table identifiers extract_sql_tables may return.
    extra = (actual_tables & entity_tables) - expected_tables
    return bool(extra)


def reused_plan_semantic_staleness_code(
    sql: str,
    known_tables: set[str],
    db_type: str,
    allowed_tables: set[str] | None = None,
    table_columns: dict[str, dict[str, str]] | None = None,
    semantic_context: dict | None = None,
    expected_temporal_window: dict | None = None,
) -> str:
    """Return the current-contract validation code for an obsolete cached plan.

    Reusable plans are indexed by the visible question text.  A result follow-up
    such as ``provide the trend for each day`` can keep that text while inheriting
    a new structured temporal window from the preceding turn.  The cache lookup
    therefore cannot, by itself, tell an unbounded historical daily plan from the
    newly requested five-day plan.

    Revalidate the candidate against the *current* semantic contract before it is
    selected.  This covers temporal policies, approved metric/date mappings,
    source-fact constraints, field plans, and join governance without encoding a
    client, schema, date column, or interval in the reuse layer.  Any validation
    failure means the candidate is stale for this request and fresh compilation
    must continue.  Validation exceptions also fail closed because accepting an
    unchecked cached plan would bypass the normal validator later in the pipeline.
    """
    if not str(sql or "").strip():
        return "empty_reused_plan"
    # The effective rolling window is request state, not an incidental detail
    # of the cached SQL.  Clarification and result-follow-up turns can retain
    # that state even when an upstream semantic-plan merge is incomplete.  In
    # that case ordinary validation has no temporal policy to enforce and an
    # old, unbounded daily query would otherwise look valid.  Never reuse a
    # candidate until the current request has compiled its window into the
    # semantic contract.  Fresh generation can then either build the governed
    # contract or fail closed; cache reuse must not hide the missing contract.
    temporal_window = dict(expected_temporal_window or {})
    semantic_plan = (semantic_context or {}).get("semantic_plan") or {}
    if temporal_window and not list(semantic_plan.get("temporal_policies") or []):
        return "current_temporal_policy_missing"
    try:
        from core.validator import validate_sql_detailed

        result = validate_sql_detailed(
            sql,
            known_tables,
            db_type,
            allowed_tables,
            table_columns,
            semantic_context,
        )
    except Exception:
        log.exception("Reusable SQL plan current-contract validation failed")
        return "reuse_validation_error"
    if result.ok:
        return ""
    return str(result.code or "current_contract_mismatch")
