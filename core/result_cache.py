"""
core/result_cache.py

Session-scoped, versioned in-memory result store for analytical follow-ups.

When a user asks a follow-up question like "who is below average?" or
"show the ratio of charges to fills", this module re-runs that query
against the already-fetched result rows using DuckDB in-memory — no
round-trip to the production database, no LIMIT constraints, full SQL
analytic functions (MEDIAN, STDDEV, PERCENTILE_CONT, window functions).

Design:
  - Every source result and local transform is an immutable snapshot.
  - Snapshot IDs are scoped to one authenticated chat session.
  - One DuckDB connection is created per local query; values are bound as
    parameters instead of being interpolated into generated SQL.
  - LRU eviction: max 200 sessions, 10-minute TTL by default.
  - A lightweight Python fallback supports the narrow deterministic command
    surface when DuckDB is unavailable.

Public API
----------
result_cache — module-level singleton
    .store(session_id, rows, question, sql) → result_id
    .derive_snapshot(session_id, parent_result_id, rows, ...) → snapshot
    .query(session_id, sql, result_id=..., parameters=...) → list[dict]
    .get_snapshot(session_id, result_id=...) → snapshot
    .restore_parent(session_id, result_id=...) → snapshot
    .get_schema(session_id, result_id=...) → list[dict]
    .has_result(session_id, result_id=...) → bool
    .clear(session_id) → None
"""

from __future__ import annotations

import logging
import hashlib
import hmac
import json
import re
import secrets
import threading
import time
import uuid
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
_ROW_TOKEN_SECRET = secrets.token_bytes(32)


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
        # Allow explicit "number" through — it lets admins override currency heuristics
        if header:
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
    # Try numeric string — sample up to 50 rows to avoid truncating floats that
    # appear later in the set, and handle negative numbers (.isdigit() is False for '-5').
    try:
        float(str(sample).replace(",", ""))
        probe = non_null[:50]
        if all(_is_numeric_str(v) for v in probe):
            # Integer check: strip optional leading '-' before testing digits
            if all(str(v).replace(",", "").lstrip("-").isdigit() for v in probe):
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
        # Compound AND/OR predicates are not supported by this fallback parser.
        # Return all rows un-filtered rather than silently returning wrong rows.
        if _re.search(r"\b(?:AND|OR)\b", cond, _re.IGNORECASE):
            log.debug("Python fallback: compound WHERE clause not supported — skipping filter")
            return _project_fallback_rows(result, sql)
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
    __slots__ = (
        "rows", "schema", "question", "sql", "column_formats", "stored_at",
        "row_token_nonce", "session_id", "result_id", "parent_result_id",
        "operation", "created_at", "metadata",
    )

    def __init__(
        self,
        rows: list[dict],
        question: str,
        sql: str,
        column_formats: dict | None = None,
        *,
        session_id: str = "",
        result_id: str = "",
        parent_result_id: str = "",
        operation: str = "source_query",
        schema: list[dict] | None = None,
        metadata: dict | None = None,
    ):
        self.rows           = rows
        self.schema         = list(schema or _schema_from_rows(rows))
        self.question       = question
        self.sql            = sql
        self.column_formats = _normalise_column_formats(rows, column_formats)
        self.stored_at      = time.monotonic()
        self.row_token_nonce = secrets.token_hex(16)
        self.session_id = session_id
        self.result_id = result_id or uuid.uuid4().hex
        self.parent_result_id = parent_result_id
        self.operation = operation
        self.created_at = time.time()
        self.metadata = dict(metadata or {})

    def is_expired(self) -> bool:
        return (time.monotonic() - self.stored_at) > _TTL_SECONDS


# ══════════════════════════════════════════════════════════════════════════════
# Public cache class
# ══════════════════════════════════════════════════════════════════════════════

class ResultCache:
    """Thread-safe LRU in-memory result cache backed by DuckDB per-query."""

    def __init__(self, max_sessions: int = _MAX_SESSIONS):
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._snapshots: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._session_snapshots: dict[str, list[str]] = {}
        self._max   = max_sessions
        self._max_snapshots = max_sessions * 8
        # RLock so the same thread can re-acquire (e.g. store→_evict_expired→_get)
        self._lock  = threading.RLock()

    # ── Write ─────────────────────────────────────────────────────────────────

    def store(
        self,
        session_id: str,
        rows: list[dict],
        question: str = "",
        sql: str = "",
        column_formats: dict | None = None,
        *,
        result_id: str | None = None,
        parent_result_id: str = "",
        operation: str = "source_query",
        metadata: dict | None = None,
    ) -> str:
        """Cache rows and return a versioned, session-scoped result id.

        Existing callers still resolve the newest snapshot by ``session_id``.
        New result-command callers can address an immutable source snapshot by
        ``result_id`` and derive a new child without mutating the source.
        """
        if not session_id or not rows:
            return ""
        with self._lock:
            self._evict_expired()
            snapshot_id = str(result_id or uuid.uuid4().hex).strip() or uuid.uuid4().hex
            existing = self._snapshots.get(snapshot_id)
            if existing is not None and existing.session_id != session_id:
                snapshot_id = uuid.uuid4().hex
            entry = _CacheEntry(
                list(rows), question, sql, column_formats,
                session_id=session_id,
                result_id=snapshot_id,
                parent_result_id=parent_result_id,
                operation=operation,
                metadata=metadata,
            )
            self._register_snapshot(entry)
        log.debug("Cached %d rows for session %s", len(rows), session_id[:12])
        return entry.result_id

    def derive_snapshot(
        self,
        session_id: str,
        source_result_id: str,
        rows: list[dict],
        *,
        question: str,
        operation: str,
        sql: str = "",
        metadata: dict | None = None,
        schema: list[dict] | None = None,
        column_formats: dict | None = None,
    ) -> dict:
        """Create a child snapshot while preserving the original result."""
        with self._lock:
            source = self._get(session_id, source_result_id)
            if source is None:
                raise LookupError("The source result has expired. Run the query again.")
            source_columns = [item.get("name") for item in source.schema]
            result_columns = list(rows[0].keys()) if rows else source_columns
            resolved_schema = (
                list(schema)
                if schema is not None
                else list(source.schema)
                if result_columns == source_columns
                else _schema_from_rows(rows)
            )
            resolved_formats = (
                dict(source.column_formats)
                if column_formats is None
                else dict(column_formats)
            )
            child = _CacheEntry(
                list(rows),
                question or source.question,
                sql or source.sql,
                resolved_formats,
                session_id=session_id,
                result_id=uuid.uuid4().hex,
                parent_result_id=source.result_id,
                operation=operation,
                schema=resolved_schema,
                metadata=metadata,
            )
            self._register_snapshot(child)
            return self._snapshot_payload(child)

    def get_snapshot(self, session_id: str, result_id: str | None = None) -> dict:
        with self._lock:
            entry = self._get(session_id, result_id)
            if entry is None:
                return {}
            return self._snapshot_payload(entry)

    def restore_parent(self, session_id: str, result_id: str | None = None) -> dict:
        """Make the parent snapshot current and return it without copying rows."""
        with self._lock:
            current = self._get(session_id, result_id)
            if current is None:
                raise LookupError("The result has expired. Run the query again.")
            if not current.parent_result_id:
                raise ValueError("This result has no previous version to restore.")
            parent = self._get(session_id, current.parent_result_id)
            if parent is None:
                raise LookupError("The previous result version has expired.")
            self._store[session_id] = parent
            self._store.move_to_end(session_id)
            return self._snapshot_payload(parent)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        session_id: str,
        sql: str,
        *,
        result_id: str | None = None,
        parameters: list[Any] | tuple[Any, ...] | None = None,
    ) -> list[dict]:
        """
        Run `sql` against the cached result for `session_id`.

        The virtual table name is always `result`.  DuckDB is used when
        available; falls back to the Python mini-executor otherwise.

        Returns [] if the session is unknown or expired.
        """
        # Hold the lock only long enough to fetch the entry reference.
        # The DuckDB query itself runs without the lock — entry.rows is
        # immutable once stored, so no corruption risk during the query.
        with self._lock:
            entry = self._get(session_id, result_id)
        if entry is None:
            return []

        try:
            safe_sql = ensure_duckdb_result_sql(sql)
        except ValueError as exc:
            log.warning("Rejected DuckDB result-cache SQL: %s", exc)
            return []

        try:
            import duckdb
            return self._duckdb_query(
                entry.rows, entry.schema, safe_sql, parameters=parameters,
            )
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

    @staticmethod
    def _row_token(entry: _CacheEntry, index: int, row: dict) -> str:
        """Return an opaque handle for one cached row.

        The handle is scoped to the current cache generation. It contains no
        row value and becomes invalid after any exclusion or cache refresh.
        """
        canonical = json.dumps(row, sort_keys=True, default=str, separators=(",", ":"))
        row_hash = hashlib.sha256(canonical.encode("utf-8", errors="ignore")).hexdigest()
        message = f"{entry.row_token_nonce}:{index}:{row_hash}".encode("utf-8")
        return hmac.new(_ROW_TOKEN_SECRET, message, hashlib.sha256).hexdigest()[:32]

    def get_row_tokens(
        self, session_id: str, limit: int | None = None,
        *, result_id: str | None = None,
    ) -> list[str]:
        """Return opaque, ordered handles aligned with the cached rows."""
        with self._lock:
            entry = self._get(session_id, result_id)
            if entry is None:
                return []
            rows = entry.rows if limit is None else entry.rows[: max(0, int(limit))]
            return [self._row_token(entry, index, row) for index, row in enumerate(rows)]

    def exclude_rows(
        self, session_id: str, row_tokens: list[str],
        *, result_id: str | None = None,
    ) -> dict:
        """Remove selected rows from a session result without invoking an LLM.

        Only opaque handles issued for the current cache generation are
        accepted. The returned rows remain inside the existing governed result
        boundary; callers may safely render them back to the same user.
        """
        requested = {
            str(token).strip()
            for token in (row_tokens or [])
            if isinstance(token, str) and str(token).strip()
        }
        if not requested:
            raise ValueError("Select at least one result row to exclude.")

        with self._lock:
            entry = self._get(session_id, result_id)
            if entry is None:
                raise LookupError("The result has expired. Run the query again.")

            token_to_index = {
                self._row_token(entry, index, row): index
                for index, row in enumerate(entry.rows)
            }
            unknown = requested - token_to_index.keys()
            if unknown:
                raise ValueError("One or more selected rows are stale. Refresh the result and retry.")

            excluded_indexes = {token_to_index[token] for token in requested}
            before = len(entry.rows)
            entry.rows = [
                row for index, row in enumerate(entry.rows)
                if index not in excluded_indexes
            ]
            # Preserve the original schema even when every row is excluded.
            entry.stored_at = time.monotonic()
            entry.row_token_nonce = secrets.token_hex(16)
            self._store.move_to_end(session_id)
            after = len(entry.rows)
            new_tokens = [
                self._row_token(entry, index, row)
                for index, row in enumerate(entry.rows)
            ]

            return {
                "rows": list(entry.rows),
                "row_tokens": new_tokens,
                "rows_before": before,
                "rows_after": after,
                "excluded_count": before - after,
                "question": entry.question,
                "sql": entry.sql,
                "column_formats": dict(entry.column_formats),
            }

    def _duckdb_query(
        self, rows: list[dict], schema: list[dict], safe_sql: str,
        *, parameters: list[Any] | tuple[Any, ...] | None = None,
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

            cursor = conn.execute(safe_sql, list(parameters or []))
            result = cursor.fetchall()
            col_names_out = [desc[0] for desc in cursor.description]
            return [dict(zip(col_names_out, row)) for row in result]
        finally:
            conn.close()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_schema(
        self, session_id: str, *, result_id: str | None = None,
    ) -> list[dict]:
        """Return [{name, type}, ...] for the cached result, or []."""
        with self._lock:
            entry = self._get(session_id, result_id)
        return list(entry.schema) if entry else []

    def get_sql(self, session_id: str, *, result_id: str | None = None) -> str:
        """Return the original SQL that produced the cached result, or ""."""
        with self._lock:
            entry = self._get(session_id, result_id)
        return entry.sql if entry else ""

    def get_stats(
        self, session_id: str, *, result_id: str | None = None,
    ) -> dict:
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
        with self._lock:
            entry = self._get(session_id, result_id)
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

    def get_currency_columns(
        self, session_id: str, *, result_id: str | None = None,
    ) -> list[str]:
        """
        Return column names from the cached result that are auto-detected as
        currency/monetary values.  Used to apply $ formatting and inform the
        LLM that these are dollar amounts.
        """
        stats = self.get_stats(session_id, result_id=result_id)
        return [
            c["name"] for c in stats.get("columns", [])
            if c.get("is_currency")
        ]

    def get_column_formats(
        self, session_id: str, *, result_id: str | None = None,
    ) -> dict[str, str]:
        """
        Return display formats for cached result columns.

        Explicit metric formats are preserved. Currency name heuristics are
        added as a fallback for older cached results and non-metric queries.
        """
        with self._lock:
            entry = self._get(session_id, result_id)
        if entry is None:
            return {}
        formats = dict(entry.column_formats)
        for col in self.get_currency_columns(session_id, result_id=result_id):
            formats.setdefault(col, "currency")
        return formats

    def has_result(self, session_id: str, *, result_id: str | None = None) -> bool:
        with self._lock:
            entry = self._get(session_id, result_id)
        return entry is not None

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._clear_session(session_id)

    def clear_account(self, account_id: str) -> int:
        """Remove every cached session belonging to a tenant."""
        prefix = f"{account_id}:"
        with self._lock:
            keys = [key for key in self._store if key == account_id or key.startswith(prefix)]
            for key in keys:
                self._clear_session(key)
        return len(keys)

    def _get(
        self, session_id: str, result_id: str | None = None,
    ) -> "_CacheEntry | None":
        """Must be called with self._lock held."""
        entry = (
            self._snapshots.get(str(result_id))
            if result_id
            else self._store.get(session_id)
        )
        if entry is None:
            return None
        if entry.session_id != session_id:
            return None
        if entry.is_expired():
            self._remove_snapshot(entry.result_id)
            return None
        self._snapshots.move_to_end(entry.result_id)
        if not result_id and session_id in self._store:
            self._store.move_to_end(session_id)
        return entry

    def _evict_expired(self) -> None:
        """Must be called with self._lock held."""
        expired = [key for key, entry in self._snapshots.items() if entry.is_expired()]
        for result_id in expired:
            self._remove_snapshot(result_id)

    def _register_snapshot(self, entry: _CacheEntry) -> None:
        """Register a snapshot and make it the latest for its session."""
        if entry.session_id not in self._store and len(self._store) >= self._max:
            oldest_session = next(iter(self._store))
            self._clear_session(oldest_session)

        self._snapshots[entry.result_id] = entry
        self._snapshots.move_to_end(entry.result_id)
        history = self._session_snapshots.setdefault(entry.session_id, [])
        if entry.result_id not in history:
            history.append(entry.result_id)
        self._store[entry.session_id] = entry
        self._store.move_to_end(entry.session_id)

        while len(self._snapshots) > self._max_snapshots:
            oldest_result_id = next(iter(self._snapshots))
            self._remove_snapshot(oldest_result_id)

    def _remove_snapshot(self, result_id: str) -> None:
        entry = self._snapshots.pop(result_id, None)
        if entry is None:
            return
        history = self._session_snapshots.get(entry.session_id, [])
        if result_id in history:
            history.remove(result_id)
        if not history:
            self._session_snapshots.pop(entry.session_id, None)
            self._store.pop(entry.session_id, None)
            return
        latest = self._store.get(entry.session_id)
        if latest is entry:
            replacement = self._snapshots.get(history[-1])
            if replacement is None:
                self._store.pop(entry.session_id, None)
            else:
                self._store[entry.session_id] = replacement

    def _clear_session(self, session_id: str) -> None:
        for result_id in list(self._session_snapshots.get(session_id, [])):
            self._snapshots.pop(result_id, None)
        self._session_snapshots.pop(session_id, None)
        self._store.pop(session_id, None)

    @staticmethod
    def _snapshot_payload(entry: _CacheEntry) -> dict:
        return {
            "result_id": entry.result_id,
            "parent_result_id": entry.parent_result_id,
            "operation": entry.operation,
            "rows": list(entry.rows),
            "row_count": len(entry.rows),
            "question": entry.question,
            "sql": entry.sql,
            "schema": list(entry.schema),
            "column_formats": dict(entry.column_formats),
            "created_at": entry.created_at,
            "metadata": dict(entry.metadata),
        }


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
