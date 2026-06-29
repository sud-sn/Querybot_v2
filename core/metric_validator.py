"""
core/metric_validator.py

Backend metric formula validator for the Semantic Layer (Sprint 1).

Validates metric SQL templates and formula expressions before they are
saved as 'validated' or 'published'. Returns a structured ValidationResult
with errors and warnings so the UI can show inline feedback.

Checks performed
────────────────
  1. Presence — formula must not be empty
  2. Dangerous SQL — blocks subqueries, DDL, DML
  3. Function allowlist — only safe functions for the account's DB dialect
  4. Aggregate rule — expression-type metrics must contain an aggregate
  5. Column existence — required_columns checked against discovered schema
  6. base_table / base_entity — warns when missing (needed for Sprint 6 compiler)
  7. formula_ast extraction — lightweight structural parse stored on the metric

Integration points
──────────────────
  store/config_store.py  — save_metric(), update_metric() call validate_metric()
  admin/routes.py        — /metrics/validate endpoint (live JSON feedback)
"""

from __future__ import annotations

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("querybot.metric_validator")

# ══════════════════════════════════════════════════════════════════════════════
# Allowlisted safe functions per DB dialect
# ══════════════════════════════════════════════════════════════════════════════

_AGGREGATE_FUNCTIONS: set[str] = {
    "SUM", "COUNT", "AVG", "MIN", "MAX",
    "COUNT_BIG", "STDEV", "STDEVP", "VAR", "VARP",         # Azure SQL
    "STDDEV", "STDDEV_POP", "VAR_POP", "VAR_SAMP",          # Snowflake
    "MEDIAN", "APPROX_COUNT_DISTINCT", "COUNT_IF", "LISTAGG", # Snowflake
    "VARIANCE",                                               # Oracle
}

_SAFE_FUNCTIONS: dict[str, set[str]] = {
    "azure_sql": {
        # Aggregates
        "SUM", "COUNT", "COUNT_BIG", "AVG", "MIN", "MAX",
        "STDEV", "STDEVP", "VAR", "VARP",
        # Math
        "ABS", "CEILING", "FLOOR", "ROUND", "POWER", "SQRT",
        "LOG", "LOG10", "EXP", "SIGN", "RAND", "PI",
        # String
        "LEN", "UPPER", "LOWER", "LTRIM", "RTRIM", "TRIM",
        "SUBSTRING", "LEFT", "RIGHT", "CHARINDEX", "REPLACE",
        "CONCAT", "ISNULL", "NULLIF", "COALESCE", "STRING_AGG",
        # Date
        "YEAR", "MONTH", "DAY", "DATEADD", "DATEDIFF", "GETDATE",
        "GETUTCDATE", "EOMONTH", "DATEPART", "DATEFROMPARTS",
        "DATETIMEFROMPARTS", "FORMAT", "ISDATE",
        # Conditional / type
        "CASE", "IIF", "CHOOSE", "CAST", "CONVERT",
        "TRY_CAST", "TRY_CONVERT", "TRY_PARSE",
        # Window (allowed in Full SQL metrics)
        "ROW_NUMBER", "RANK", "DENSE_RANK", "NTILE",
        "LAG", "LEAD", "FIRST_VALUE", "LAST_VALUE",
        "SUM", "AVG", "COUNT", "MIN", "MAX",  # window variants
    },
    "snowflake": {
        # Aggregates
        "SUM", "COUNT", "AVG", "MIN", "MAX",
        "STDDEV", "STDDEV_POP", "VAR_POP", "VAR_SAMP",
        "MEDIAN", "APPROX_COUNT_DISTINCT", "COUNT_IF",
        "LISTAGG", "ARRAY_AGG", "OBJECT_AGG", "BOOLOR_AGG",
        # Math
        "ABS", "CEIL", "FLOOR", "ROUND", "POWER", "SQRT",
        "LOG", "LN", "EXP", "SIGN", "MOD", "TRUNC",
        # String
        "LENGTH", "UPPER", "LOWER", "LTRIM", "RTRIM", "TRIM",
        "SUBSTR", "SUBSTRING", "LEFT", "RIGHT", "POSITION",
        "REPLACE", "CONCAT", "NVL", "NVL2", "NULLIF",
        "COALESCE", "ZEROIFNULL", "IFF", "IFNULL",
        "SPLIT_PART", "REGEXP_REPLACE", "REGEXP_COUNT",
        # Date
        "YEAR", "MONTH", "DAY", "DATEADD", "DATEDIFF",
        "CURRENT_DATE", "CURRENT_TIMESTAMP", "SYSDATE",
        "DATE_TRUNC", "DATE_PART", "TO_DATE", "TO_TIMESTAMP",
        "EXTRACT", "LAST_DAY", "NEXT_DAY",
        # Conditional / type
        "CASE", "IFF", "DECODE", "CAST", "TRY_CAST",
        "TO_NUMBER", "TO_DECIMAL", "TO_CHAR", "TO_BOOLEAN",
        # Window
        "ROW_NUMBER", "RANK", "DENSE_RANK", "NTILE",
        "LAG", "LEAD", "FIRST_VALUE", "LAST_VALUE",
        "RATIO_TO_REPORT", "PERCENT_RANK", "CUME_DIST",
    },
    "oracle": {
        # Aggregates
        "SUM", "COUNT", "AVG", "MIN", "MAX",
        "STDDEV", "STDDEV_POP", "VARIANCE", "VAR_POP",
        "MEDIAN", "LISTAGG", "WM_CONCAT",
        # Math
        "ABS", "CEIL", "FLOOR", "ROUND", "POWER", "SQRT",
        "LOG", "LN", "EXP", "SIGN", "MOD", "TRUNC",
        # String
        "LENGTH", "UPPER", "LOWER", "LTRIM", "RTRIM", "TRIM",
        "SUBSTR", "INSTR", "REPLACE", "CONCAT", "LPAD", "RPAD",
        "NVL", "NVL2", "NULLIF", "COALESCE", "DECODE",
        "REGEXP_REPLACE", "REGEXP_COUNT",
        # Date
        "MONTHS_BETWEEN", "ADD_MONTHS", "SYSDATE", "SYSTIMESTAMP",
        "TO_DATE", "TO_TIMESTAMP", "EXTRACT", "TRUNC", "LAST_DAY",
        "NEXT_DAY", "ROUND",
        # Conditional / type
        "CASE", "DECODE", "CAST", "TO_NUMBER", "TO_CHAR", "TO_DATE",
        "TO_BINARY_FLOAT", "TO_BINARY_DOUBLE",
        # Window
        "ROW_NUMBER", "RANK", "DENSE_RANK", "NTILE",
        "LAG", "LEAD", "FIRST_VALUE", "LAST_VALUE",
        "RATIO_TO_REPORT", "PERCENT_RANK", "CUME_DIST",
    },
}
# Generic fallback used when db_type is unknown
_SAFE_FUNCTIONS["generic"] = (
    _SAFE_FUNCTIONS["azure_sql"]
    | _SAFE_FUNCTIONS["snowflake"]
    | _SAFE_FUNCTIONS["oracle"]
)

# ══════════════════════════════════════════════════════════════════════════════
# Dangerous SQL patterns
# ══════════════════════════════════════════════════════════════════════════════

# DDL keywords that must never appear in a metric formula
_DDL_PATTERN = re.compile(
    r"\b(CREATE|ALTER|DROP|TRUNCATE|RENAME|COMMENT|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

# DML keywords that must never appear in a metric formula
_DML_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|UPSERT|REPLACE\s+INTO|LOAD\s+DATA)\b",
    re.IGNORECASE,
)

# Subquery detection — SELECT … FROM inside the formula
_SUBQUERY_PATTERN = re.compile(
    r"\bSELECT\b.+\bFROM\b",
    re.IGNORECASE | re.DOTALL,
)

# Script/proc keywords
_SCRIPT_PATTERN = re.compile(
    r"\b(EXEC|EXECUTE|SP_|XP_|DECLARE|BEGIN|COMMIT|ROLLBACK|PRAGMA|CALL)\b",
    re.IGNORECASE,
)

# Comment injection
_COMMENT_PATTERN = re.compile(r"(--|/\*|\*/)", re.IGNORECASE)

# Function call extractor — finds WORD( patterns
_FUNCTION_CALL_RE = re.compile(r"\b([A-Z_][A-Z0-9_]*)\s*\(", re.IGNORECASE)

# Column reference extractor — simple identifier tokens not followed by (
_IDENTIFIER_RE = re.compile(
    r"\b([A-Z_][A-Z0-9_#@$]*)\b(?!\s*\()",
    re.IGNORECASE,
)

# SQL keywords to exclude from column lists
_SQL_KEYWORDS: set[str] = {
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "IS", "NULL",
    "AS", "ON", "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "OUTER",
    "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "TOP", "DISTINCT",
    "OVER", "PARTITION", "ROWS", "RANGE", "BETWEEN", "THEN", "ELSE",
    "END", "WHEN", "CASE", "CAST", "CONVERT", "LIKE", "WITH", "NOLOCK",
    "ASC", "DESC", "ALL", "SOME", "ANY", "EXISTS", "UNION",
    "INTERSECT", "EXCEPT", "ROWNUM", "DUAL",
}


# ══════════════════════════════════════════════════════════════════════════════
# Validation result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    formula_ast: dict = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> dict:
        return {
            "valid":       self.valid,
            "errors":      self.errors,
            "warnings":    self.warnings,
            "formula_ast": self.formula_ast,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def validate_metric(
    metric: dict,
    *,
    db_type: str = "azure_sql",
    schema_columns: Optional[dict[str, list[str]]] = None,
) -> ValidationResult:
    """
    Validate a metric definition dict.

    Parameters
    ──────────
    metric         — metric dict (same shape as metric_registry row)
    db_type        — 'azure_sql' | 'snowflake' | 'oracle'  (for function allowlist)
    schema_columns — optional {table_fqn: [col, ...]} for column existence checks

    Returns
    ───────
    ValidationResult with .valid, .errors, .warnings, .formula_ast
    """
    result = ValidationResult()
    formula_type = (metric.get("formula_type") or "query").lower().strip()
    formula      = (metric.get("sql_template") or "").strip()

    # ── 1. Presence ──────────────────────────────────────────────────────────
    if not formula:
        result.add_error("Formula/SQL is required.")
        result.formula_ast = _empty_ast(formula_type)
        return result

    # ── 2. Dangerous SQL ─────────────────────────────────────────────────────
    if _COMMENT_PATTERN.search(formula):
        result.add_error(
            "SQL comments (-- or /* */) are not allowed in metric formulas."
        )
    if _DDL_PATTERN.search(formula):
        m = _DDL_PATTERN.search(formula)
        result.add_error(
            f"DDL keyword '{m.group(0).upper()}' is not allowed in metric formulas."
        )
    if _DML_PATTERN.search(formula):
        m = _DML_PATTERN.search(formula)
        result.add_error(
            f"DML keyword '{m.group(0).upper()}' is not allowed in metric formulas."
        )
    if _SCRIPT_PATTERN.search(formula):
        m = _SCRIPT_PATTERN.search(formula)
        result.add_error(
            f"Keyword '{m.group(0).upper()}' is not allowed — "
            "stored procedures and scripts cannot be used in metrics."
        )

    # Subquery check — only applies to expression-type metrics;
    # full SQL (query type) may legitimately contain SELECT … FROM.
    if formula_type == "expression" and _SUBQUERY_PATTERN.search(formula):
        result.add_error(
            "Subqueries (SELECT … FROM …) are not allowed in expression-type metrics. "
            "Use formula_type='query' for full SQL definitions."
        )

    # ── 3. Extract functions and build formula_ast ───────────────────────────
    # Strip ${MetricName} refs before syntactic analysis — they expand to valid
    # aggregates at runtime and should not be treated as unknown identifiers.
    _METRIC_REF_STRIP = re.compile(r'\$\{[^}]+\}')
    has_metric_refs   = bool(_METRIC_REF_STRIP.search(formula))
    formula_stripped  = _METRIC_REF_STRIP.sub('SUM(1)', formula) if has_metric_refs else formula

    functions_used    = _extract_functions(formula_stripped)
    columns_referenced = _extract_columns(formula_stripped)
    aggregate_fns     = functions_used & _AGGREGATE_FUNCTIONS
    has_subquery      = bool(_SUBQUERY_PATTERN.search(formula_stripped))

    result.formula_ast = {
        "type":                 formula_type,
        "functions_used":       sorted(functions_used),
        "aggregate_functions":  sorted(aggregate_fns),
        "columns_referenced":   sorted(columns_referenced),
        "has_subquery":         has_subquery,
        "has_dangerous_keyword": not result.valid,
        "has_metric_refs":      has_metric_refs,
    }

    # Stop further checks if we already have fatal errors
    if not result.valid:
        return result

    # ── 4. Function allowlist ────────────────────────────────────────────────
    allowed = _SAFE_FUNCTIONS.get(db_type, _SAFE_FUNCTIONS["generic"])
    unknown_fns = [
        fn for fn in functions_used
        if fn.upper() not in allowed and fn.upper() not in _SQL_KEYWORDS
    ]
    for fn in sorted(unknown_fns):
        result.add_error(
            f"Function '{fn}' is not on the allowlist for {db_type}. "
            "Use only approved aggregate, math, string, date, and conditional functions."
        )

    # ── 5. Aggregate rule for expression-type metrics ────────────────────────
    # Skip when metric refs are present — they expand to aggregates at runtime.
    if formula_type == "expression" and not aggregate_fns and not has_metric_refs:
        result.add_error(
            "Expression-type metrics must contain at least one aggregate function "
            f"(e.g. SUM, COUNT, AVG). Found none in: {formula[:120]}"
        )

    # ── 6. Column existence check ────────────────────────────────────────────
    required_columns_raw = (metric.get("required_columns") or "").strip()
    base_table           = (metric.get("base_table") or "").strip()

    if required_columns_raw and schema_columns and base_table:
        required_cols = [
            c.strip() for c in required_columns_raw.split(",") if c.strip()
        ]
        # Find best matching table entry (case-insensitive suffix match)
        table_cols = _resolve_table_columns(base_table, schema_columns)
        if table_cols is not None:
            table_cols_upper = {c.upper() for c in table_cols}
            for col in required_cols:
                if col.upper() not in table_cols_upper:
                    result.add_error(
                        f"Required column '{col}' does not exist in table '{base_table}'. "
                        "Check schema discovery has run and the column name is correct."
                    )
        else:
            result.add_warning(
                f"Cannot verify required columns — table '{base_table}' was not found "
                "in the discovered schema. Run schema discovery first."
            )
    elif required_columns_raw and not base_table:
        result.add_warning(
            "Required columns are specified but no base_table is set. "
            "Column existence cannot be verified without a base table."
        )

    # ── 7. base_table / base_entity advisory ─────────────────────────────────
    if not base_table:
        result.add_warning(
            "No base table set. Providing a base table improves validation accuracy "
            "and is required for the deterministic SQL compiler (Sprint 6)."
        )
    if not (metric.get("base_entity") or "").strip():
        result.add_warning(
            "No base entity set. Linking to an entity graph entity enables "
            "dimension pickers and automatic join path resolution."
        )

    return result


def derive_metric_status(result: ValidationResult, formula_type: str) -> str:
    """
    Return the metric_status string to persist based on validation outcome.

    draft       — validation failed; metric cannot be used
    validated   — validation passed; metric is ready to publish
    published   — caller explicitly promotes (not set here automatically)
    """
    if not result.valid:
        return "draft"
    return "validated"


def load_schema_columns(account_id: str) -> Optional[dict[str, list[str]]]:
    """
    Load the discovered schema column map for an account.
    Returns {table_fqn: [col_name, ...]} or None if not yet discovered.
    """
    import json as _json
    from pathlib import Path as _Path
    try:
        import store
        state      = store.get_client_state(account_id)
        schema_dir = (state or {}).get("schema_dir") or ""
        if not schema_dir:
            return None
        schema_path = _Path(schema_dir) / "_schema.json"
        if not schema_path.exists():
            return None
        master = _json.loads(schema_path.read_text(encoding="utf-8"))
        result: dict[str, list[str]] = {}
        for fqn, tbl_data in master.items():
            if isinstance(tbl_data, dict):
                cols = [
                    c.get("name", "") if isinstance(c, dict) else str(c)
                    for c in (tbl_data.get("columns") or [])
                ]
                result[fqn] = [c for c in cols if c]
        return result or None
    except Exception as exc:
        log.debug("load_schema_columns failed for %s: %s", account_id, exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _extract_functions(formula: str) -> set[str]:
    """Return the set of function names called in the formula (upper-cased)."""
    return {m.group(1).upper() for m in _FUNCTION_CALL_RE.finditer(formula)}


def _extract_columns(formula: str) -> set[str]:
    """
    Return likely column name references — identifiers that are NOT
    SQL keywords, NOT function names (not followed by '('), and
    NOT purely numeric.
    """
    function_names = _extract_functions(formula)
    columns: set[str] = set()
    for m in _IDENTIFIER_RE.finditer(formula):
        token = m.group(1)
        if (
            token.upper() not in _SQL_KEYWORDS
            and token.upper() not in function_names
            and not token.isdigit()
            and len(token) > 1
        ):
            columns.add(token)
    return columns


def _resolve_table_columns(
    base_table: str,
    schema_columns: dict[str, list[str]],
) -> Optional[list[str]]:
    """
    Find the column list for base_table in schema_columns.
    Tries exact match first, then case-insensitive suffix match.
    """
    # Exact match
    if base_table in schema_columns:
        return schema_columns[base_table]
    # Case-insensitive full match
    base_upper = base_table.upper()
    for fqn, cols in schema_columns.items():
        if fqn.upper() == base_upper:
            return cols
    # Suffix match — "dbo.ORDERS" matches key "mydb.dbo.ORDERS"
    for fqn, cols in schema_columns.items():
        parts = fqn.upper().split(".")
        if parts and parts[-1] == base_upper.split(".")[-1]:
            return cols
    return None


def _empty_ast(formula_type: str) -> dict:
    return {
        "type":                 formula_type,
        "functions_used":       [],
        "aggregate_functions":  [],
        "columns_referenced":   [],
        "has_subquery":         False,
        "has_dangerous_keyword": False,
    }
