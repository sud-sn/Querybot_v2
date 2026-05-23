"""
Safety validator for LLM-generated DuckDB SQL over the in-memory result table.

The result-chat path must be more restrictive than normal SQL generation:
queries run locally against a single virtual table named ``result`` and should
never access files, extensions, pragmas, attachments, or mutate state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_FORBIDDEN_RE = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|MERGE|REPLACE|"
    r"COPY|EXPORT|INSTALL|LOAD|ATTACH|DETACH|PRAGMA|CALL|SET|RESET|"
    r"VACUUM|CHECKPOINT|IMPORT|DESCRIBE|SHOW"
    r")\b|"
    r"\b(read_csv|read_csv_auto|read_parquet|read_json|read_json_auto|"
    r"read_ndjson|read_blob|httpfs|sqlite_scan|postgres_scan)\s*\(",
    re.IGNORECASE,
)

_DANGEROUS_COMMENT_RE = re.compile(r"(--|/\*)")
_FROM_JOIN_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+((?:\"[^\"]+\"|'[^']+'|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w$]*)(?:\s*\.\s*(?:\"[^\"]+\"|'[^']+'|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w$]*))*)",
    re.IGNORECASE,
)
_CTE_RE = re.compile(r"(?:WITH|,)\s+([A-Za-z_][\w$]*)\s+AS\s*\(", re.IGNORECASE)


@dataclass(frozen=True)
class DuckDBValidationResult:
    ok: bool
    reason: str = "OK"
    code: str = "ok"


def _strip_trailing_semicolon(sql: str) -> str:
    sql = (sql or "").strip()
    return sql[:-1].strip() if sql.endswith(";") else sql


def _has_multiple_statements(sql: str) -> bool:
    body = _strip_trailing_semicolon(sql)
    return ";" in body


def _normalize_identifier(name: str) -> str:
    value = (name or "").strip()
    if value[:1] in ("\"", "'", "`") and value[-1:] == value[:1]:
        value = value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return value.strip().lower()


def _base_table_name(ref: str) -> str:
    parts = re.split(r"\s*\.\s*", ref.strip())
    return _normalize_identifier(parts[-1] if parts else ref)


def validate_duckdb_result_sql(sql: str) -> DuckDBValidationResult:
    """Return whether ``sql`` is safe to execute against virtual table result."""
    raw = (sql or "").strip()
    if not raw:
        return DuckDBValidationResult(False, "Empty DuckDB SQL", "empty")

    if _DANGEROUS_COMMENT_RE.search(raw):
        return DuckDBValidationResult(False, "DuckDB SQL comments are not allowed", "comment")

    if _has_multiple_statements(raw):
        return DuckDBValidationResult(False, "DuckDB SQL must contain one statement", "multi_statement")

    body = _strip_trailing_semicolon(raw)
    if not re.match(r"^\s*(SELECT|WITH)\b", body, re.IGNORECASE):
        return DuckDBValidationResult(False, "DuckDB SQL must be SELECT or WITH", "not_select")

    if _FORBIDDEN_RE.search(body):
        return DuckDBValidationResult(False, "DuckDB SQL contains a forbidden operation", "forbidden")

    cte_names = {_normalize_identifier(m.group(1)) for m in _CTE_RE.finditer(body)}
    table_refs = [_base_table_name(m.group(1)) for m in _FROM_JOIN_RE.finditer(body)]
    if not table_refs:
        return DuckDBValidationResult(False, "DuckDB SQL must read from result", "no_result_table")

    invalid = [t for t in table_refs if t != "result" and t not in cte_names]
    if invalid:
        return DuckDBValidationResult(
            False,
            "DuckDB SQL may only reference the result table",
            "invalid_table",
        )

    return DuckDBValidationResult(True)


def ensure_duckdb_result_sql(sql: str) -> str:
    """Raise ValueError unless ``sql`` passes ``validate_duckdb_result_sql``."""
    verdict = validate_duckdb_result_sql(sql)
    if not verdict.ok:
        raise ValueError(f"{verdict.code}: {verdict.reason}")
    return _strip_trailing_semicolon(sql)
