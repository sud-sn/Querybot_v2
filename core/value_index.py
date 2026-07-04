"""
core/value_index.py

Per-client index of filterable column values for literal grounding.

Schema discovery only captures distinct values for categorical-looking columns
and silently drops anything with more than ~30 distinct values — so customer
names, item descriptions, and other high-cardinality filter columns have zero
value representation in the KB. When a user asks "sales for Emco corp" the LLM
must guess the WHERE literal; the data may say 'EMCO Corporation' and the
query returns zero rows with no explanation.

This module builds a SQLite value index at discovery time
(clients/{account_id}/value_index.sqlite) and provides millisecond lookups so
the query pipeline can resolve user-typed literals to exact database values
BEFORE the LLM writes SQL (core/value_resolver.py) and explain unmatched
literals after a zero-row result (core/answer_rca.py).

Privacy layers (values from masked/PII columns must never be indexed):
  1. Columns listed in the table's `masked_fields` (and tables with
     mask_mode == "all") from _schema.json are skipped.
  2. core.masking.detect_sensitive_columns name-pattern hits are skipped.
  3. Harvested values are scanned with core.masking's value-level PII check;
     a column whose values look like PII is dropped entirely.
  4. Values longer than 200 chars or containing newlines are rejected
     (free-text content, and a prompt-injection guard — indexed values are
     later quoted into LLM prompts).

The index lives beside column_context.json, so the existing client-reset
flow (which deletes clients/{account_id}/ wholesale) cleans it up for free.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("querybot.value_index")

_DEFAULT_BASE_DIR = "clients"
_INDEX_FILENAME = "value_index.sqlite"

# Value hygiene: anything longer or multi-line is free text, not a filter value.
_MAX_VALUE_LEN = 200

# Fuzzy thresholds — aligned with core/clarification.py's typo resolver.
FUZZY_VERIFIED = 0.87
FUZZY_CANDIDATE = 0.75

# Suffix roles from core/naming_convention.py that mark filterable display text.
_FILTERABLE_ROLES = {"display", "code"}


# ── Paths / normalization ─────────────────────────────────────────────────────

def _index_path(account_id: str, base_dir: str = _DEFAULT_BASE_DIR) -> Path:
    return Path(base_dir) / (account_id or "") / _INDEX_FILENAME


def normalize_value(value: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Shared by the
    builder, the question-time resolver, and the zero-row RCA matcher so all
    three agree on what 'the same value' means."""
    text = (value or "").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def value_index_enabled(state_data: dict | None) -> bool:
    """Per-client opt-out via client state_data; default ON."""
    if not isinstance(state_data, dict):
        return True
    flag = state_data.get("value_index_enabled")
    if flag is None:
        return True
    return bool(flag) and str(flag).lower() not in ("0", "false", "no", "off")


# ── Column selection ──────────────────────────────────────────────────────────

def _is_dimension_table(bare_table: str) -> bool:
    upper = (bare_table or "").upper()
    return upper.endswith(("_DMS", "_DIM")) or upper.startswith(("DIM_", "DMS_", "D_"))


def _is_string_type(col_type: str) -> bool:
    from core.schema import _CATEGORICAL_TYPES
    base = (col_type or "").lower().split("(")[0].strip()
    return any(t in base for t in _CATEGORICAL_TYPES)


def select_filterable_columns(schema: dict, vocab=None) -> list[dict]:
    """
    Choose columns worth value-indexing from a normalized _schema.json dict.

    Included: string-typed columns whose naming role is display/code
    (_NM/_NAME/_DSC/_DESC/_CD/_CODE …) on dimension-classified tables, plus
    categorical columns discovery already scans (so the RCA matcher covers
    them uniformly). Excluded: every masking signal (see module docstring).
    """
    from core.masking import detect_sensitive_columns
    from core.naming_convention import match_column_suffix
    from core.schema import _is_categorical

    selected: list[dict] = []
    for fqn, meta in (schema or {}).items():
        if str(fqn).startswith("__") or not isinstance(meta, dict):
            continue
        if (meta.get("mask_mode") or "") == "all":
            continue
        columns = meta.get("columns") or []
        col_defs = [c for c in columns if isinstance(c, dict) and c.get("name")]
        masked = {str(f).upper() for f in (meta.get("masked_fields") or [])}
        sensitive = {str(c).upper() for c in detect_sensitive_columns(col_defs)}

        parts = str(fqn).split(".")
        bare_table = parts[-1]
        for col in col_defs:
            name = str(col.get("name") or "")
            ctype = str(col.get("type") or "")
            upper = name.upper()
            if upper in masked or upper in sensitive:
                continue
            if not _is_string_type(ctype):
                continue

            rule = match_column_suffix(name)
            is_display = bool(rule and rule.role in _FILTERABLE_ROLES) and _is_dimension_table(bare_table)
            is_cat = _is_categorical(name, ctype)
            if not (is_display or is_cat):
                continue

            business_name = ""
            try:
                from core.schema_enrichment import enrich_columns
                enriched = enrich_columns([name], vocab=vocab)
                if enriched:
                    business_name = enriched[0].expanded_name
            except Exception:
                pass
            selected.append({
                "table_fqn": str(fqn),
                "column": name,
                "type": ctype,
                "business_name": business_name,
                "database": str(meta.get("database") or ""),
                "schema": str(meta.get("schema") or ""),
                "table": bare_table,
            })
    return selected


# ── DISTINCT harvesting ───────────────────────────────────────────────────────

def _distinct_sql(db_type: str, database: str, schema: str, table: str, column: str, cap: int) -> str:
    if db_type == "azure_sql":
        tbl = f"[{schema}].[{table}]" if schema else f"[{table}]"
        return (
            f"SELECT DISTINCT TOP {cap + 1} [{column}] FROM {tbl} "
            f"WHERE [{column}] IS NOT NULL"
        )
    if db_type == "oracle":
        tbl = f'"{schema.upper()}"."{table.upper()}"' if schema else f'"{table.upper()}"'
        return (
            f'SELECT * FROM (SELECT DISTINCT "{column.upper()}" FROM {tbl} '
            f'WHERE "{column.upper()}" IS NOT NULL) WHERE ROWNUM <= {cap + 1}'
        )
    # snowflake / default
    parts = [p for p in (database, schema, table) if p]
    tbl = ".".join(f'"{p}"' for p in parts)
    return (
        f'SELECT DISTINCT "{column}" FROM {tbl} '
        f'WHERE "{column}" IS NOT NULL LIMIT {cap + 1}'
    )


def _values_look_like_pii(column: str, col_type: str, values: list[str]) -> bool:
    """Value-level PII gate reusing core.masking's scanner."""
    from core.masking import scan_values_for_pii
    rows = [{column: v} for v in values[:200]]
    col_defs = [{"name": column, "type": col_type or "varchar"}]
    try:
        return bool(scan_values_for_pii(rows, col_defs))
    except Exception:
        return False


def build_value_index(
    account_id: str,
    credentials: dict,
    db_type: str,
    schema_dir: str,
    *,
    per_column_cap: int = 5000,
    vocab=None,
    run_query_fn: Callable[..., list[dict]] | None = None,
    base_dir: str = _DEFAULT_BASE_DIR,
) -> dict:
    """
    Harvest distinct values for filterable columns into
    clients/{account_id}/value_index.sqlite. Atomic: written to a .tmp file
    and os.replace'd, so readers never see a half-built index.

    run_query_fn is injectable for tests; defaults to core.schema.run_query.
    Returns build stats (also persisted in the index's meta table).
    """
    from core.schema import load_schema_json

    if run_query_fn is None:
        from core.schema import run_query as run_query_fn  # type: ignore[no-redef]

    schema = load_schema_json(schema_dir)
    columns = select_filterable_columns(schema, vocab=vocab)

    final_path = _index_path(account_id, base_dir)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(".sqlite.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    stats: dict[str, Any] = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_column_cap": per_column_cap,
        "columns_considered": len(columns),
        "columns_indexed": 0,
        "values_indexed": 0,
        "columns_skipped_pii": 0,
        "columns_failed": 0,
        "truncated_columns": [],
    }

    conn = sqlite3.connect(tmp_path)
    try:
        conn.executescript(
            """
            CREATE TABLE column_value (
              table_fqn TEXT NOT NULL, column_name TEXT NOT NULL,
              business_name TEXT NOT NULL DEFAULT '',
              value TEXT NOT NULL, value_norm TEXT NOT NULL);
            CREATE INDEX ix_cv_norm ON column_value(value_norm);
            CREATE INDEX ix_cv_col ON column_value(table_fqn, column_name);
            CREATE TABLE value_index_meta (key TEXT PRIMARY KEY, value TEXT);
            """
        )
        for col in columns:
            sql = _distinct_sql(
                db_type, col["database"], col["schema"], col["table"],
                col["column"], per_column_cap,
            )
            try:
                rows = run_query_fn(credentials, db_type, sql, max_rows=per_column_cap + 1)
            except Exception as exc:
                stats["columns_failed"] += 1
                log.debug("Value index: DISTINCT failed for %s.%s: %s",
                          col["table_fqn"], col["column"], exc)
                continue

            values: list[str] = []
            for row in rows:
                raw = next(iter(row.values())) if isinstance(row, dict) and row else None
                if raw is None:
                    continue
                text = str(raw).strip()
                if not text or len(text) > _MAX_VALUE_LEN or "\n" in text or "\r" in text:
                    continue
                values.append(text)

            if not values:
                continue
            if _values_look_like_pii(col["column"], col.get("type", ""), values):
                stats["columns_skipped_pii"] += 1
                log.info("Value index: skipping %s.%s — values look like PII",
                         col["table_fqn"], col["column"])
                continue
            if len(values) > per_column_cap:
                values = values[:per_column_cap]
                stats["truncated_columns"].append(f"{col['table_fqn']}.{col['column']}")

            conn.executemany(
                "INSERT INTO column_value (table_fqn, column_name, business_name, value, value_norm) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (col["table_fqn"], col["column"], col["business_name"], v, normalize_value(v))
                    for v in values
                ],
            )
            stats["columns_indexed"] += 1
            stats["values_indexed"] += len(values)

        conn.executemany(
            "INSERT INTO value_index_meta (key, value) VALUES (?, ?)",
            [(k, json.dumps(v)) for k, v in stats.items()],
        )
        conn.commit()
    finally:
        conn.close()

    os.replace(tmp_path, final_path)
    log.info(
        "Value index built for %s: %d columns, %d values (%d PII-skipped, %d failed)",
        account_id, stats["columns_indexed"], stats["values_indexed"],
        stats["columns_skipped_pii"], stats["columns_failed"],
    )
    return stats


# ── Lookup API ────────────────────────────────────────────────────────────────

def index_exists(account_id: str, base_dir: str = _DEFAULT_BASE_DIR) -> bool:
    return _index_path(account_id, base_dir).is_file()


def load_index_stats(account_id: str, base_dir: str = _DEFAULT_BASE_DIR) -> dict:
    path = _index_path(account_id, base_dir)
    if not path.is_file():
        return {}
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT key, value FROM value_index_meta").fetchall()
        finally:
            conn.close()
        return {k: json.loads(v) for k, v in rows}
    except Exception:
        return {}


def _open_ro(account_id: str, base_dir: str) -> sqlite3.Connection | None:
    path = _index_path(account_id, base_dir)
    if not path.is_file():
        return None
    try:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except Exception:
        return None


def _table_allowed(table_fqn: str, allowed_tables: set[str] | None) -> bool:
    if allowed_tables is None:
        return True
    upper = table_fqn.upper()
    parts = upper.split(".")
    variants = {upper, parts[-1]}
    if len(parts) >= 2:
        variants.add(".".join(parts[-2:]))
    allowed_upper = {str(t).upper() for t in allowed_tables}
    return bool(variants & allowed_upper)


def lookup_exact(
    account_id: str,
    phrase: str,
    allowed_tables: set[str] | None = None,
    base_dir: str = _DEFAULT_BASE_DIR,
) -> list[dict]:
    """Case-insensitive exact match, then normalized match. Returns
    [{table_fqn, column, business_name, value, method}]."""
    conn = _open_ro(account_id, base_dir)
    if conn is None:
        return []
    try:
        out: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for method, sql, arg in (
            ("exact", "SELECT table_fqn, column_name, business_name, value FROM column_value "
                      "WHERE value = ? COLLATE NOCASE LIMIT 50", (phrase or "").strip()),
            ("normalized", "SELECT table_fqn, column_name, business_name, value FROM column_value "
                           "WHERE value_norm = ? LIMIT 50", normalize_value(phrase)),
        ):
            if not arg:
                continue
            for tf, cn, bn, val in conn.execute(sql, (arg,)).fetchall():
                key = (tf, cn, val)
                if key in seen or not _table_allowed(tf, allowed_tables):
                    continue
                seen.add(key)
                out.append({"table_fqn": tf, "column": cn, "business_name": bn,
                            "value": val, "method": method, "score": 1.0})
            if out:
                break   # exact hits are authoritative; skip weaker tier
        return out
    finally:
        conn.close()


def lookup_fuzzy(
    account_id: str,
    phrase: str,
    allowed_tables: set[str] | None = None,
    limit: int = 5,
    base_dir: str = _DEFAULT_BASE_DIR,
    min_score: float = FUZZY_CANDIDATE,
) -> list[dict]:
    """
    Fuzzy match: SQL LIKE prefilter (first token prefix OR longest-token
    containment, capped at 300 candidates), then difflib.SequenceMatcher
    scoring against the normalized phrase. Returns matches with
    score >= min_score (default FUZZY_CANDIDATE for prompt injection; the
    zero-row RCA passes a looser floor since 'closest values' suggestions
    only inform the user, they never rewrite a query).
    """
    norm = normalize_value(phrase)
    if len(norm) < 3:
        return []
    conn = _open_ro(account_id, base_dir)
    if conn is None:
        return []
    try:
        tokens = norm.split()
        longest = max(tokens, key=len)
        first = tokens[0]
        # Three probes: first-token prefix (fast path), longest-token
        # containment, and a 2-char prefix so first-syllable typos still
        # reach the scorer ("emko corp" must find "emco corporation").
        # Short rows first keeps the most comparable candidates inside the
        # LIMIT when a 2-char prefix is common.
        candidates = conn.execute(
            "SELECT table_fqn, column_name, business_name, value, value_norm FROM column_value "
            "WHERE value_norm LIKE ? OR value_norm LIKE ? OR value_norm LIKE ? "
            "ORDER BY length(value_norm) LIMIT 300",
            (f"{first}%", f"%{longest}%", f"{norm[:2]}%"),
        ).fetchall()
    finally:
        conn.close()

    scored: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for tf, cn, bn, val, vn in candidates:
        key = (tf, cn, val)
        if key in seen or not _table_allowed(tf, allowed_tables):
            continue
        seen.add(key)
        score = SequenceMatcher(None, norm, vn).ratio()
        # Containment bonus: "emco" inside "emco corporation" is a strong
        # signal SequenceMatcher under-scores for length-mismatched strings.
        if norm and norm in vn:
            score = max(score, 0.60 + 0.40 * (len(norm) / max(len(vn), 1)))
        if score >= min_score:
            scored.append({"table_fqn": tf, "column": cn, "business_name": bn,
                           "value": val, "method": "fuzzy", "score": round(score, 4)})
    scored.sort(key=lambda m: m["score"], reverse=True)
    return scored[:limit]
