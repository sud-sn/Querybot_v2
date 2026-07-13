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

from core.schema import run_query
from core.query_semantics import analyze_query_intent
from core.answer_confidence import build_answer_confidence
from core.answer_formatter import format_zero_row_business_response
from core.answer_rca import build_business_rca, extract_sql_tables

log = logging.getLogger("querybot")


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

# Sections safe to drop from an oversized KB doc, in drop order. Columns and
# Join Keys are never dropped — they are what SQL generation actually needs.
_DROPPABLE_SECTION_RE = re.compile(
    r"(?i)^##\s*(business\s+synonyms|sample\s+data|query\s+patterns|patterns|overview)\b"
)


def _clamp_kb_doc(doc: str, cap: int = 0) -> str:
    """Trim one KB doc to *cap* chars by removing droppable sections from the
    end first (synonyms/sample-data/patterns/overview), never Columns or Join
    Keys. Falls back to a hard tail-truncate only if still over after that."""
    cap = cap or _PER_DOC_CHAR_CAP
    if len(doc) <= cap:
        return doc

    # Split into header + "## " sections, preserving order.
    parts = re.split(r"(?m)^(?=## )", doc)
    kept = list(parts)
    # Remove droppable sections from the tail end first.
    for idx in range(len(kept) - 1, 0, -1):
        if len("".join(kept)) <= cap:
            break
        if _DROPPABLE_SECTION_RE.match(kept[idx]):
            kept.pop(idx)
    clamped = "".join(kept)
    if len(clamped) > cap:
        clamped = clamped[:cap] + "\n[... truncated for prompt size]"
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
            if name.endswith(("_KEY", "_ID", "_CD", "_NUM", "_NO")):
                return False
            base = ctype.lower().split("(")[0].strip()
            return any(t in base for t in (
                "decimal", "numeric", "money", "float", "real", "int", "number",
            ))

        candidates: set[str] = set()
        for col_node in tree.find_all(sg_exp.Column):
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
        for col_node in list(tree.find_all(sg_exp.Column)):
            if (col_node.name or "").upper() != wrong_col:
                continue
            tbl_ref = (col_node.table or "").upper()
            if tbl_ref and tbl_ref not in aliases:
                continue
            col_node.replace(
                sg_exp.column(req_col, table=col_node.table)
                if col_node.table else sg_exp.column(req_col)
            )
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
