"""
core/result_cache.py

Session-keyed in-memory result store for Tier 2 analytical queries.

When a user asks a follow-up question like "who is below average?" or
"show the ratio of charges to fills", this module re-runs that query
against the already-fetched result rows using DuckDB in-memory — no
round-trip to the production database, no LIMIT constraints, full SQL
analytic functions (MEDIAN, STDDEV, PERCENTILE_CONT, window functions).

Design:
  - One DuckDB connection created fresh per query — no shared state
  - Rows are stored as plain list[dict] keyed by session_id
  - LRU eviction: max 200 sessions, 10-minute TTL
  - Falls back to a lightweight pure-Python executor if DuckDB is not
    installed (handles simple SELECT * WHERE / ORDER BY / LIMIT)

Public API
----------
result_cache  — module-level singleton
    .store(session_id, rows, question, sql)  → None
    .query(session_id, sql)                  → list[dict]
    .get_schema(session_id)                  → list[dict]  [{name, type}, ...]
    .has_result(session_id)                  → bool
    .clear(session_id)                       → None
"""

from __future__ import annotations

import logging
import re
import time
from collections import OrderedDict
from typing import Any

from core.duckdb_sql_validator import ensure_duckdb_result_sql

# ── Currency column name heuristic ────────────────────────────────────────────
# Matches column names that almost certainly represent monetary values.
# Used to auto-detect which columns should be prefixed with $ in the UI.
_CURRENCY_NAME_RE = re.compile(
    r"\b(revenue|amount|cost|price|total|sales|charge|fee|payment|spend|"
    r"value|income|profit|loss|margin|earning|billing|invoice|budget|"
    r"gross|net|balance|credit|debit|cash|dollar|usd|gbp|eur|salary|"
    r"wage|commission|rebate|discount|tax|surcharge|reimbursement)\b",
    re.IGNORECASE,
)
_RESULT_FORMATS = {"number", "currency", "percentage", "date", "text"}

log = logging.getLogger("querybot.result_cache")

_MAX_SESSIONS  = 200
_TTL_SECONDS   = 600      # 10 minutes


def _normalise_result_format(value: Any) -> str:
    fmt = str(value or "number").strip().lower()
    return fmt if fmt in _RESULT_FORMATS else "number"


def _normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _normalise_column_formats(rows: list[dict], column_formats: dict | None) -> dict[str, str]:
    if not rows or not column_formats:
        return {}
    headers = list(rows[0].keys())
    by_norm = {_normalise_key(h): h for h in headers}
    cleaned: dict[str, str] = {}
    for raw_col, raw_fmt in column_formats.items():
        header = by_norm.get(_normalise_key(raw_col))
        fmt = _normalise_result_format(raw_fmt)
        if header and fmt != "number":
            cleaned[header] = fmt
    return cleaned


# ── DuckDB type inference ─────────────────────────────────────────────────────

def _infer_duckdb_type(values: list[Any]) -> str:
    """Return a DuckDB column type from a sample of row values."""
    non_null = [v for v in values if v is not None]
    if not non_null:
        return "TEXT"
    sample = non_null[0]
    if isinstance(sample, bool):
        return "BOOLEAN"
    if isinstance(sample, int):
        return "BIGINT"
    if isinstance(sample, float):
        return "DOUBLE"
    # Try numeric string
    try:
        float(str(sample).replace(",", ""))
        # Check all samples are numeric
        if all(_is_numeric_str(v) for v in non_null[:10]):
            # Integer or float?
            if all(str(v).replace(",", "").isdigit() for v in non_null[:10]):
                return "BIGINT"
            return "DOUBLE"
    except (ValueError, TypeError):
        pass
    return "TEXT"


def _is_numeric_str(v: Any) -> bool:
    try:
        float(str(v).replace(",", ""))
        return True
    except (ValueError, TypeError):
        return False


def _schema_from_rows(rows: list[dict]) -> list[dict]:
    """Return [{name, type, duckdb_type}, ...] inferred from the first row."""
    if not rows:
        return []
    schema = []
    for col in rows[0].keys():
        sample_values = [r.get(col) for r in rows[:20]]
        dtype = _infer_duckdb_type(sample_values)
        schema.append({"name": col, "type": dtype})
    return schema


# ── Python fallback executor ──────────────────────────────────────────────────

def _python_fallback_query(rows: list[dict], sql: str) -> list[dict]:
    """
    Very basic in-memory query engine — used when DuckDB is not installed.
    Supports: SELECT * FROM result [WHERE col OP val] [ORDER BY col [DESC]] [LIMIT n]
    """
    import re as _re
    result = list(rows)

    # WHERE clause
    where_m = _re.search(r"WHERE\s+(.+?)(?:ORDER|LIMIT|$)", sql, _re.IGNORECASE | _re.DOTALL)
    if where_m:
        cond = where_m.group(1).strip()
        # Support: col < val, col > val, col = val, col <= val, col >= val
        cm = _re.match(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|<>|!=|<|>|=)\s*(.+)$", cond
        )
        if cm:
            c, op, val_str = cm.group(1).strip(), cm.group(2), cm.group(3).strip().strip("'\"")
            try:
                num_val = float(val_str.replace(",", ""))
                ops = {"<": float.__lt__, ">": float.__gt__, "<=": float.__le__,
                       ">=": float.__ge__, "=": float.__eq__, "!=": float.__ne__, "<>": float.__ne__}
                if op in ops:
                    fn = ops[op]
                    result = [r for r in result if _is_numeric_str(r.get(c, ""))
                              and fn(float(str(r.get(c, 0)).replace(",", "")), num_val)]
            except (ValueError, TypeError):
                pass

    # ORDER BY
    order_m = _re.search(r"ORDER\s+BY\s+([A-Za-z_][A-Za-z0-9_]*)\s*(DESC|ASC)?", sql, _re.IGNORECASE)
    if order_m:
        col_o = order_m.group(1)
        desc  = (order_m.group(2) or "").upper() == "DESC"
        result.sort(key=lambda r: (float(str(r.get(col_o, 0)).replace(",", ""))
                                   if _is_numeric_str(r.get(col_o, "")) else str(r.get(col_o, ""))),
                    reverse=desc)

    # LIMIT
    limit_m = _re.search(r"LIMIT\s+(\d+)", sql, _re.IGNORECASE)
    if limit_m:
        result = result[:int(limit_m.group(1))]

    return _project_fallback_rows(result, sql)


def _split_select_items(select_text: str) -> list[str]:
    items: list[str] = []
    buf: list[str] = []
    depth = 0
    quote = ""
    for ch in select_text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in ("'", '"', "`"):
            quote = ch
            buf.append(ch)
            continue
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        if ch == "," and depth == 0:
            item = "".join(buf).strip()
            if item:
                items.append(item)
            buf = []
        else:
            buf.append(ch)
    item = "".join(buf).strip()
    if item:
        items.append(item)
    return items


def _split_concat(expr: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    quote = ""
    i = 0
    while i < len(expr):
        ch = expr[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in ("'", '"', "`"):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        if depth == 0 and expr[i:i + 2] == "||":
            parts.append("".join(buf).strip())
            buf = []
            i += 2
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _extract_alias(expr: str) -> tuple[str, str | None]:
    m = re.match(r"(.+?)\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", expr, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return expr.strip(), None


def _fallback_row_numbers(rows: list[dict], order_col: str | None) -> dict[int, int]:
    indexed = list(enumerate(rows))
    if order_col:
        indexed.sort(key=lambda pair: (
            float(str(pair[1].get(order_col, 0)).replace(",", ""))
            if _is_numeric_str(pair[1].get(order_col, "")) else str(pair[1].get(order_col, ""))
        ))
    return {idx: pos + 1 for pos, (idx, _) in enumerate(indexed)}


def _eval_fallback_expr(expr: str, row: dict, row_number: int) -> Any:
    expr = expr.strip()
    if expr == "*":
        return row
    if "||" in expr:
        return "".join(str(_eval_fallback_expr(part, row, row_number) or "") for part in _split_concat(expr))
    if (expr.startswith("'") and expr.endswith("'")) or (expr.startswith('"') and expr.endswith('"')):
        return expr[1:-1]
    if re.search(r"\bROW_NUMBER\s*\(\s*\)\s*OVER\s*\(", expr, re.IGNORECASE):
        offset = 0
        off_m = re.search(r"(\d+)\s*\+\s*ROW_NUMBER", expr, re.IGNORECASE)
        if off_m:
            offset = int(off_m.group(1))
        if re.search(r"\b(CHR|CHAR)\s*\(", expr, re.IGNORECASE):
            try:
                return chr(offset + row_number)
            except (ValueError, TypeError):
                return ""
        return row_number
    cast_m = re.match(r"CAST\s*\((.+?)\s+AS\s+[A-Za-z0-9_()]+\s*\)$", expr, re.IGNORECASE | re.DOTALL)
    if cast_m:
        return _eval_fallback_expr(cast_m.group(1), row, row_number)
    col = expr.strip().strip('"').strip("`").strip("[]")
    return row.get(col)


def _project_fallback_rows(rows: list[dict], sql: str) -> list[dict]:
    m = re.search(r"^\s*SELECT\s+(.+?)\s+FROM\s+result\b", sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return rows
    select_text = m.group(1).strip()
    if select_text == "*":
        return rows

    rn_order = None
    rn_m = re.search(r"ROW_NUMBER\s*\(\s*\)\s*OVER\s*\(\s*ORDER\s+BY\s+([A-Za-z_][A-Za-z0-9_]*)", select_text, re.IGNORECASE)
    if rn_m:
        rn_order = rn_m.group(1)
    row_numbers = _fallback_row_numbers(rows, rn_order)

    projected: list[dict] = []
    for idx, row in enumerate(rows):
        out: dict[str, Any] = {}
        for raw_item in _split_select_items(select_text):
            expr, alias = _extract_alias(raw_item)
            if expr == "*":
                out.update(row)
                continue
            value = _eval_fallback_expr(expr, row, row_numbers.get(idx, idx + 1))
            if isinstance(value, dict):
                out.update(value)
                continue
            col_name = alias or expr.strip().strip('"').strip("`").strip("[]")
            out[col_name] = value
        projected.append(out)
    return projected


# ══════════════════════════════════════════════════════════════════════════════
# Cache entry
# ══════════════════════════════════════════════════════════════════════════════

class _CacheEntry:
    __slots__ = ("rows", "schema", "question", "sql", "column_formats", "stored_at")

    def __init__(
        self,
        rows: list[dict],
        question: str,
        sql: str,
        column_formats: dict | None = None,
    ):
        self.rows           = rows
        self.schema         = _schema_from_rows(rows)
        self.question       = question
        self.sql            = sql
        self.column_formats = _normalise_column_formats(rows, column_formats)
        self.stored_at      = time.monotonic()

    def is_expired(self) -> bool:
        return (time.monotonic() - self.stored_at) > _TTL_SECONDS


# ══════════════════════════════════════════════════════════════════════════════
# Public cache class
# ══════════════════════════════════════════════════════════════════════════════

class ResultCache:
    """Thread-safe LRU in-memory result cache backed by DuckDB per-query."""

    def __init__(self, max_sessions: int = _MAX_SESSIONS):
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._max   = max_sessions

    # ── Write ─────────────────────────────────────────────────────────────────

    def store(
        self,
        session_id: str,
        rows: list[dict],
        question: str = "",
        sql: str = "",
        column_formats: dict | None = None,
    ) -> None:
        """Cache `rows` under `session_id`.  Evicts oldest entry when full."""
        if not session_id or not rows:
            return
        self._evict_expired()
        if session_id in self._store:
            self._store.move_to_end(session_id)
        elif len(self._store) >= self._max:
            self._store.popitem(last=False)   # evict LRU
        self._store[session_id] = _CacheEntry(rows, question, sql, column_formats)
        log.debug("Cached %d rows for session %s", len(rows), session_id[:12])

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, session_id: str, sql: str) -> list[dict]:
        """
        Run `sql` against the cached result for `session_id`.

        The virtual table name is always `result`.  DuckDB is used when
        available; falls back to the Python mini-executor otherwise.

        Returns [] if the session is unknown or expired.
        """
        entry = self._get(session_id)
        if entry is None:
            return []

        try:
            safe_sql = ensure_duckdb_result_sql(sql)
        except ValueError as exc:
            log.warning("Rejected DuckDB result-cache SQL: %s", exc)
            return []

        try:
            import duckdb
            return self._duckdb_query(entry.rows, entry.schema, safe_sql)
        except ImportError:
            log.debug("DuckDB not installed — using Python fallback")
            return _python_fallback_query(entry.rows, safe_sql)
        except Exception as exc:
            log.warning("DuckDB query failed: %s — trying fallback", exc)
            try:
                return _python_fallback_query(entry.rows, safe_sql)
            except Exception as fb_exc:
                log.warning("Python fallback also failed: %s", fb_exc)
                return []

    def _duckdb_query(
        self, rows: list[dict], schema: list[dict], safe_sql: str
    ) -> list[dict]:
        import duckdb

        conn = duckdb.connect(":memory:")
        try:
            # Build CREATE TABLE from inferred schema
            col_defs = ", ".join(
                f'"{s["name"]}" {s["type"]}' for s in schema
            )
            conn.execute(f"CREATE TABLE result ({col_defs})")

            # Insert rows via parameter binding (safe, no SQL injection)
            if rows:
                col_names = [s["name"] for s in schema]
                placeholders = ", ".join(["?"] * len(col_names))
                cols_str = ", ".join(f'"{c}"' for c in col_names)
                batch = [
                    [_coerce(r.get(c), s["type"]) for c, s in zip(col_names, schema)]
                    for r in rows
                ]
                conn.executemany(
                    f"INSERT INTO result ({cols_str}) VALUES ({placeholders})", batch
                )

            result = conn.execute(safe_sql).fetchall()
            col_names_out = [desc[0] for desc in conn.description]
            return [dict(zip(col_names_out, row)) for row in result]
        finally:
            conn.close()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_schema(self, session_id: str) -> list[dict]:
        """Return [{name, type}, ...] for the cached result, or []."""
        entry = self._get(session_id)
        return entry.schema if entry else []

    def get_stats(self, session_id: str) -> dict:
        """
        Return a statistical summary of the cached result for DuckDB prompt
        injection.  Gives the LLM concrete knowledge of the data shape so it
        writes correct GROUP BY clauses and meaningful aggregates.

        Returns:
            {
                "row_count": int,
                "columns": [
                    {
                        "name": str, "type": str,
                        "min": float|None, "max": float|None, "avg": float|None,
                        "unique_count": int, "null_count": int,
                        "sample_values": list[str],   # top 5 for TEXT cols
                    }, ...
                ]
            }
        Returns {} when session is unknown or expired.
        """
        entry = self._get(session_id)
        if not entry or not entry.rows:
            return {}

        rows = entry.rows
        col_stats: list[dict] = []

        for col_info in entry.schema:
            col   = col_info["name"]
            dtype = col_info["type"]
            values     = [r.get(col) for r in rows]
            non_null   = [v for v in values if v is not None]
            null_count = len(values) - len(non_null)

            stat: dict = {
                "name":         col,
                "type":         dtype,
                "null_count":   null_count,
                "unique_count": len({str(v) for v in non_null}),
                "sample_values": [],
            }

            is_numeric = dtype in ("DOUBLE", "REAL", "FLOAT", "BIGINT", "INTEGER")
            if is_numeric:
                try:
                    nums = [float(v) for v in non_null]
                    if nums:
                        stat["min"] = round(min(nums), 2)
                        stat["max"] = round(max(nums), 2)
                        stat["avg"] = round(sum(nums) / len(nums), 2)
                except (ValueError, TypeError):
                    pass
            else:
                # Categorical — list first 5 distinct values (insertion order)
                seen: dict = {}
                for v in non_null:
                    seen[str(v)] = None
                    if len(seen) >= 5:
                        break
                stat["sample_values"] = list(seen.keys())

            explicit_format = entry.column_formats.get(col)
            if explicit_format:
                stat["format"] = explicit_format

            # Currency detection: explicit metric format or name heuristic + numeric type
            stat["is_currency"] = bool(
                explicit_format == "currency"
                or (is_numeric and _CURRENCY_NAME_RE.search(col))
            )

            col_stats.append(stat)

        return {"row_count": len(rows), "columns": col_stats}

    def get_currency_columns(self, session_id: str) -> list[str]:
        """
        Return column names from the cached result that are auto-detected as
        currency/monetary values.  Used to apply $ formatting and inform the
        LLM that these are dollar amounts.
        """
        stats = self.get_stats(session_id)
        return [
            c["name"] for c in stats.get("columns", [])
            if c.get("is_currency")
        ]

    def get_column_formats(self, session_id: str) -> dict[str, str]:
        """
        Return display formats for cached result columns.

        Explicit metric formats are preserved. Currency name heuristics are
        added as a fallback for older cached results and non-metric queries.
        """
        entry = self._get(session_id)
        if entry is None:
            return {}
        formats = dict(entry.column_formats)
        for col in self.get_currency_columns(session_id):
            formats.setdefault(col, "currency")
        return formats

    def has_result(self, session_id: str) -> bool:
        entry = self._get(session_id)
        return entry is not None

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def _get(self, session_id: str) -> "_CacheEntry | None":
        entry = self._store.get(session_id)
        if entry is None:
            return None
        if entry.is_expired():
            del self._store[session_id]
            return None
        self._store.move_to_end(session_id)   # LRU refresh
        return entry

    def _evict_expired(self) -> None:
        expired = [k for k, v in self._store.items() if v.is_expired()]
        for k in expired:
            del self._store[k]


def _coerce(value: Any, dtype: str) -> Any:
    """Coerce a Python value to the expected DuckDB column type."""
    if value is None:
        return None
    if dtype in ("BIGINT", "INTEGER"):
        try:
            return int(float(str(value).replace(",", "")))
        except (ValueError, TypeError):
            return None
    if dtype in ("DOUBLE", "REAL", "FLOAT"):
        try:
            return float(str(value).replace(",", ""))
        except (ValueError, TypeError):
            return None
    if dtype == "BOOLEAN":
        return bool(value)
    return str(value)


# Module-level singleton — import and use directly
result_cache = ResultCache()
