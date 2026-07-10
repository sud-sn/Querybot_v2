"""
core/validator.py

SQL safety and schema grounding.

Layers:
  1. DDL/DML block: reject destructive statements.
  2. AST parse: sqlglot with the correct SQL dialect.
  3. Table validation: CTE-aware table existence and ACL checks.
  4. Column validation: alias-aware exact column checks when schema metadata is supplied.
  5. Date-key guardrails: reject FORMAT(int_yyyymmdd_key, 'yyyy-MM') style bugs.
  6. ORDER BY alias drift check.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import get_close_matches

log = logging.getLogger("querybot.validator")

try:
    import sqlglot
    from sqlglot import exp as sg_exp
    _HAS_SQLGLOT = True
except ImportError:
    _HAS_SQLGLOT = False
    log.warning("sqlglot not installed - structural SQL validation disabled")


@dataclass
class SqlValidationResult:
    ok: bool
    reason: str
    code: str
    errors: list[dict] = field(default_factory=list)


_DDL_DML = re.compile(
    r"""\b(
        CREATE | DROP    | ALTER   | INSERT  | UPDATE  | DELETE  |
        TRUNCATE | MERGE | GRANT   | REVOKE  | EXECUTE | EXEC    |
        COPY | PUT | GET | UNLOAD  | LOAD    | BULK    | OPENROWSET |
        CALL | PROCEDURE | FUNCTION | TRIGGER
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

_DIALECT: dict[str, str] = {
    "snowflake": "snowflake",
    "oracle": "oracle",
    "azure_sql": "tsql",
}


_SQLSERVER_BAD_DATEPART_CAST_RE = re.compile(
    r"CAST\s*\(\s*"
    r"(?P<part>YEAR|MONTH|DAY)\s*\(\s*"
    r"(?P<inner>"
    r"TRY_CONVERT\s*\(\s*"
    r"(?P<date_type>DATE|DATETIME|DATETIME2|SMALLDATETIME)\s*,\s*"
    r"CONVERT\s*\(\s*"
    r"(?P<char_type>N?VARCHAR|N?CHAR)\s*\(\s*8\s*\)\s*,\s*"
    r"(?P<date_col>(?:\[[^\]]+\]|[A-Za-z_][\w$]*)(?:\s*\.\s*(?:\[[^\]]+\]|[A-Za-z_][\w$]*))?)"
    r"\s*\)\s*,\s*112\s*\)"
    r")\s*\)\s*\)\s*AS\s+(?P<cast_type>INT|INTEGER|BIGINT)\s*\)",
    re.IGNORECASE,
)


def normalize_generated_sql(sql: str, db_type: str = "snowflake") -> str:
    """
    Apply deterministic, semantics-preserving cleanup to generated SQL before
    validation/execution.

    The main production case is Azure SQL date-key grouping. LLMs sometimes
    write this malformed expression:

        CAST(YEAR(TRY_CONVERT(date, CONVERT(varchar(8), col), 112))) AS INT)

    SQL Server expects the target type inside CAST(... AS type), so normalize it
    to:

        CAST(YEAR(TRY_CONVERT(date, CONVERT(varchar(8), col), 112)) AS INT)

    This is intentionally narrow: it only fixes misplaced CAST type syntax
    around YEAR/MONTH/DAY over a YYYYMMDD TRY_CONVERT pattern.
    """
    if not sql or (db_type or "").lower() != "azure_sql":
        return sql

    def _replace(match: re.Match) -> str:
        part = match.group("part").upper()
        date_type = match.group("date_type").lower()
        char_type = match.group("char_type").lower()
        date_col = re.sub(r"\s+", "", match.group("date_col"))
        cast_type = match.group("cast_type").upper()
        return (
            f"CAST({part}(TRY_CONVERT({date_type}, "
            f"CONVERT({char_type}(8), {date_col}), 112)) AS {cast_type})"
        )

    previous = None
    normalized = sql
    # Re-run until stable in case the same bad expression appears in SELECT,
    # GROUP BY, ORDER BY, and CTE filters.
    while previous != normalized:
        previous = normalized
        normalized = _SQLSERVER_BAD_DATEPART_CAST_RE.sub(_replace, normalized)
    return normalized


def _strip_literals_and_comments(sql: str) -> str:
    """
    Blank out string literals and comments so the DDL/DML keyword scan only
    sees executable SQL. A filter value like ACTION_TYPE = 'DELETE' is data,
    not an operation — without this, audit/log-table queries whose values
    happen to be words like DELETE, UPDATE, or EXEC are falsely rejected.
    Single-quoted literals handle '' escaping; identifiers in [] / "" are kept
    (a DDL keyword cannot hide there as an executable statement).
    """
    text = sql or ""
    text = re.sub(r"'(?:[^']|'')*'", "''", text)
    text = re.sub(r"--[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text


def _normalize_identifier(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", (name or "").upper())


def _strip_identifier(identifier: str) -> str:
    return (identifier or "").strip().strip("[]\"`").upper()


def _column_suggestions(column: str, candidates: set[str]) -> list[str]:
    if not column or not candidates:
        return []
    norm = _normalize_identifier(column)
    exact_norm = sorted(c for c in candidates if _normalize_identifier(c) == norm)
    if exact_norm:
        return exact_norm[:5]
    norm_to_original = {_normalize_identifier(c): c for c in candidates}
    matches = get_close_matches(norm, list(norm_to_original), n=5, cutoff=0.72)
    return [norm_to_original[m] for m in matches]


def _tables_with_column(column: str, table_columns: dict[str, dict[str, str]]) -> list[str]:
    matches: list[str] = []
    seen_bare: set[str] = set()
    col_upper = (column or "").upper()
    col_norm = _normalize_identifier(column)
    for table_key, cols in table_columns.items():
        if col_upper not in cols and not any(_normalize_identifier(c) == col_norm for c in cols):
            continue
        parts = table_key.upper().split(".")
        bare = parts[-1]
        # Prefer qualified entries when available; suppress duplicate bare variants.
        if len(parts) == 1 and bare in seen_bare:
            continue
        seen_bare.add(bare)
        matches.append(table_key)
    return sorted(matches)[:8]


def _expand_table_set(tables: set[str] | None) -> set[str] | None:
    if tables is None:
        return None
    expanded: set[str] = set()
    for t in tables:
        upper = str(t).upper()
        expanded.add(upper)
        parts = upper.split(".")
        if len(parts) >= 2:
            expanded.add(parts[-1])
            expanded.add(".".join(parts[-2:]))
    return expanded


def _table_variants(node) -> list[str]:
    parts: list[str] = []
    if getattr(node, "catalog", ""):
        parts.append(str(node.catalog))
    if getattr(node, "db", ""):
        parts.append(str(node.db))
    if getattr(node, "name", ""):
        parts.append(str(node.name))
    if not parts:
        return []
    variants = [".".join(parts).upper()]
    if len(parts) >= 2:
        variants.append(".".join(parts[-2:]).upper())
    variants.append(parts[-1].upper())
    return list(dict.fromkeys(v for v in variants if v))


def _pick_table_key(variants: list[str], table_columns: dict[str, dict[str, str]]) -> str:
    for variant in variants:
        if variant in table_columns:
            return variant
    return variants[-1] if variants else ""


def _table_matches(left: str, right: str) -> bool:
    left_u = (left or "").upper()
    right_u = (right or "").upper()
    if left_u == right_u:
        return True
    left_parts = left_u.split(".")
    right_parts = right_u.split(".")
    if left_parts[-1:] == right_parts[-1:]:
        return True
    if len(left_parts) >= 2 and len(right_parts) >= 2:
        return ".".join(left_parts[-2:]) == ".".join(right_parts[-2:])
    return False


def _find_table_with_column(table_columns: dict[str, dict[str, str]], table: str, column: str) -> str:
    col_u = (column or "").upper()
    for table_key, cols in table_columns.items():
        if _table_matches(table_key, table) and col_u in cols:
            return table_key
    return (table or "").upper()


def _format_plan_field(field: dict) -> str:
    return f"{field.get('term') or field.get('column')}: {field.get('table')}.{field.get('column')}"


_SQL_IDENTIFIER_STOPWORDS = {
    "AS",
    "AVG",
    "CASE",
    "CAST",
    "COALESCE",
    "CONVERT",
    "COUNT",
    "COUNT_BIG",
    "DATE",
    "DECIMAL",
    "DISTINCT",
    "ELSE",
    "END",
    "FLOAT",
    "FROM",
    "GROUP",
    "ISNULL",
    "MAX",
    "MIN",
    "NULL",
    "NULLIF",
    "NUMERIC",
    "NVL",
    "ROUND",
    "SELECT",
    "SUM",
    "THEN",
    "TRY_CONVERT",
    "VARCHAR",
    "WHEN",
}


def _strip_column_reference(identifier: str) -> str:
    """Return the final column part from TABLE.COLUMN / [TABLE].[COLUMN]."""
    value = (identifier or "").strip()
    if not value:
        return ""
    parts = [p for p in re.split(r"\s*\.\s*", value) if p]
    return _strip_identifier(parts[-1] if parts else value)


def _split_required_columns(raw: str) -> list[str]:
    columns: list[str] = []
    for part in re.split(r"[,;\n]+", raw or ""):
        col = _strip_column_reference(part)
        if col and re.match(r"^[A-Z_][A-Z0-9_]*$", col):
            columns.append(col)
    return columns


def _extract_formula_columns(metric: dict, known_columns: set[str]) -> set[str]:
    """
    Extract source columns required by an approved formula metric.

    This is deliberately schema-backed: only identifiers that are known columns
    are enforced, so metric names, aliases, and SQL functions do not become
    false requirements.
    """
    columns: set[str] = set()
    for col in _split_required_columns(metric.get("required_columns") or ""):
        if not known_columns or col in known_columns:
            columns.add(col)

    formula = metric.get("sql_template") or ""
    for match in re.finditer(r"\b(?:SUM|AVG|COUNT|MIN|MAX)\s*\((.*?)\)", formula, re.IGNORECASE | re.DOTALL):
        inner = match.group(1)
        for identifier in re.findall(r"(?:\[?[A-Za-z_][A-Za-z0-9_]*\]?\s*\.\s*)?\[?[A-Za-z_][A-Za-z0-9_]*\]?", inner):
            col = _strip_column_reference(identifier)
            if col and col not in _SQL_IDENTIFIER_STOPWORDS and (not known_columns or col in known_columns):
                columns.add(col)

    if not columns:
        for identifier in re.findall(r"(?:\[?[A-Za-z_][A-Za-z0-9_]*\]?\s*\.\s*)?\[?[A-Za-z_][A-Za-z0-9_]*\]?", formula):
            col = _strip_column_reference(identifier)
            if col and col not in _SQL_IDENTIFIER_STOPWORDS and (not known_columns or col in known_columns):
                columns.add(col)
    return columns


def _metric_phrases(metric: dict) -> set[str]:
    phrases = {(metric.get("name") or "").strip().lower()}
    for syn in (metric.get("synonyms") or "").split(","):
        syn = syn.strip().lower()
        if syn:
            phrases.add(syn)
    return {p for p in phrases if p}


def _metric_mentioned(metric: dict, question: str) -> bool:
    q = re.sub(r"[^a-z0-9]+", " ", (question or "").lower()).strip()
    if not q:
        return False
    for phrase in _metric_phrases(metric):
        phrase_norm = re.sub(r"[^a-z0-9]+", " ", phrase).strip()
        if phrase_norm and re.search(rf"(?<![a-z0-9]){re.escape(phrase_norm)}(?![a-z0-9])", q):
            return True
    return False


def _find_metric_formula_errors(sql: str, tree, semantic_context: dict | None, table_columns: dict[str, dict[str, str]]) -> list[dict]:
    metrics = (semantic_context or {}).get("metric_formulas") or []
    if not metrics:
        return []

    question = (semantic_context or {}).get("question") or ""
    known_columns = {str(c).upper() for cols in table_columns.values() for c in (cols or {})}

    # Restrict column tracking to SELECT expressions only (not WHERE / GROUP BY /
    # ORDER BY), so a column used only as a filter predicate does not satisfy the
    # formula-enforcement check — the formula must appear in the projection.
    #
    # IMPORTANT: scan ALL Select nodes, not just the first one. CTE queries place
    # the approved metric formula inside a CTE's inner SELECT; tree.find() would
    # only return the outer SELECT which never contains the formula columns.
    used_columns: set[str] = set()
    for select_node in tree.find_all(sg_exp.Select):
        for expr in select_node.expressions:
            for col_node in expr.find_all(sg_exp.Column):
                col_name = (col_node.name or "").upper()
                if col_name and col_name != "*":
                    used_columns.add(col_name)

    errors: list[dict] = []
    for metric in metrics:
        if (metric.get("formula_type") or "query").lower() != "expression":
            continue
        if not _metric_mentioned(metric, question):
            continue
        required = _extract_formula_columns(metric, known_columns)
        if not required:
            continue
        missing = sorted(required - used_columns)
        if not missing:
            continue
        errors.append({
            "code": "metric_formula_mismatch",
            "message": (
                f"SQL did not use approved metric formula for {metric.get('name') or 'matched metric'}. "
                f"Missing required formula column(s): {', '.join(missing)}."
            ),
            "metric": metric.get("name", ""),
            "formula": (metric.get("sql_template") or "").strip(),
            "required_columns": sorted(required),
            "missing_columns": missing,
            "used_columns": sorted(used_columns),
        })
    return errors


def _is_numeric_date_key(col_name: str, col_type: str = "") -> bool:
    name = (col_name or "").upper()
    ctype = (col_type or "").upper()
    if name.endswith("_DT_DMS_KEY") or name.endswith("_DATE_DMS_KEY"):
        return not any(token in ctype for token in ("DATE", "TIME"))
    return False


def _find_date_key_format_errors(sql: str, table_columns: dict[str, dict[str, str]]) -> list[dict]:
    errors: list[dict] = []
    all_cols: dict[str, str] = {}
    for cols in table_columns.values():
        all_cols.update(cols)

    pattern = re.compile(
        r"\bFORMAT\s*\(\s*(?:(?P<alias>\[?[A-Za-z_][A-Za-z0-9_]*\]?)\s*\.\s*)?"
        r"(?P<col>\[?[A-Za-z_][A-Za-z0-9_]*\]?)\s*,\s*'yyyy-MM'",
        re.IGNORECASE,
    )
    for match in pattern.finditer(sql or ""):
        col = _strip_identifier(match.group("col"))
        col_type = all_cols.get(col, "")
        if _is_numeric_date_key(col, col_type):
            errors.append({
                "code": "date_key_format",
                "message": (
                    f"FORMAT() was applied directly to numeric date key {col}. "
                    "Convert YYYYMMDD keys to date first."
                ),
                "column": col,
                "alias": _strip_identifier(match.group("alias") or ""),
                "suggestions": [
                    f"TRY_CONVERT(date, CONVERT(varchar(8), alias.{col}), 112)",
                    f"FORMAT(TRY_CONVERT(date, CONVERT(varchar(8), alias.{col}), 112), 'yyyy-MM')",
                ],
            })
    return errors


_DATE_FILTER_COLUMN_RE = re.compile(
    r"(?i)(_DT_DMS_KEY$|_DATE_DMS_KEY$|_DT$|_DATE$|^DT_|^DATE_|"
    r"_YR$|^YR$|_YEAR$|^YEAR$|_MTH$|^MTH$|_MONTH$|^MONTH$|"
    r"_QTR$|^QTR$|_QUARTER$|^QUARTER$|_WK$|^WK$|_WEEK$|^WEEK$)"
)


def _where_has_identity_filter(where) -> bool:
    """True if a parsed WHERE clause has an equality/IN condition on a
    non-date-like column — an identity/category lookup ("customer_id = 123",
    "status IN (...)") as opposed to a pure time-range filter. Shared by the
    validator (below) and by example-filtering (core/examples.py), so a
    stored few-shot example is held to the exact same rule a freshly
    generated query is.
    """
    for cond in where.find_all(sg_exp.EQ, sg_exp.In):
        col_node = cond.this if isinstance(cond.this, sg_exp.Column) else None
        if col_node is None:
            continue
        if not _DATE_FILTER_COLUMN_RE.search(col_node.name or ""):
            return True
    return False


def has_identity_filter(sql: str) -> bool:
    """Public wrapper: parse *sql* and report whether its WHERE clause has
    an identity/category filter (see _where_has_identity_filter). False for
    unparseable SQL, no WHERE clause, or a pure date/time-range filter."""
    if not _HAS_SQLGLOT or not sql:
        return False
    try:
        tree = sqlglot.parse_one(sql)
    except Exception:
        return False
    where = tree.find(sg_exp.Where)
    if where is None:
        return False
    return _where_has_identity_filter(where)


def _find_null_aggregate_diagnostic_errors(sql: str, tree) -> list[dict]:
    """
    Guard filtered single-row SUM queries from returning a misleading NULL.

    For questions like "revenue for customer X", SQL Server returns one row with
    SUM(col)=NULL when rows match but every metric value is NULL.  Require the
    query to carry enough diagnostics for the answer layer to explain that case.

    This must NOT fire on a plain time-bounded aggregate ("sales for the last
    7 days", "revenue this month") — WHERE order_date >= X has no "does this
    entity exist" ambiguity the way WHERE customer_id = 123 does, so forcing
    MatchedRows/NonNullMetricRows onto it only adds noise, breaks the
    single-value answer shape, and costs an unnecessary repair retry. Only
    trigger when the WHERE clause has an equality/IN condition on a
    non-date-like column — the actual identity/category-lookup case this
    guard was written for.
    """
    where = tree.find(sg_exp.Where)
    if not sql or where is None or tree.find(sg_exp.Group) is not None:
        return []

    if not _where_has_identity_filter(where):
        return []

    select = tree.find(sg_exp.Select)
    if select is None:
        return []

    sql_text = sql or ""
    sum_pattern = re.compile(
        r"\bSUM\s*\(\s*(?:(?P<alias>\[?[A-Za-z_][A-Za-z0-9_]*\]?)\s*\.\s*)?"
        r"(?P<col>\[?[A-Za-z_][A-Za-z0-9_]*\]?)\s*\)",
        re.IGNORECASE,
    )
    sum_cols: list[tuple[str, str, bool]] = []
    for match in sum_pattern.finditer(sql_text):
        col = _strip_identifier(match.group("col"))
        prefix = sql_text[max(0, match.start() - 40):match.start()]
        wrapped = bool(re.search(r"\b(COALESCE|ISNULL|NVL|IFNULL)\s*\(\s*$", prefix, re.IGNORECASE))
        if col:
            sum_cols.append((_strip_identifier(match.group("alias") or ""), col, wrapped))

    if not sum_cols:
        return []

    has_matched_count = bool(
        re.search(r"\bCOUNT(?:_BIG)?\s*\(\s*\*\s*\)", sql_text, re.IGNORECASE)
    )

    errors: list[dict] = []
    for alias, col, wrapped in sum_cols:
        qualified = (
            rf"(?:{re.escape(alias)}\s*\.\s*)?" if alias else r"(?:\[?[A-Za-z_][A-Za-z0-9_]*\]?\s*\.\s*)?"
        )
        has_non_null_count = bool(
            re.search(
                rf"\bCOUNT(?:_BIG)?\s*\(\s*{qualified}\[?{re.escape(col)}\]?\s*\)",
                sql_text,
                re.IGNORECASE,
            )
        )
        if has_matched_count and has_non_null_count and wrapped:
            continue
        errors.append({
            "code": "null_aggregate_diagnostic",
            "message": (
                f"Filtered SUM on {col} must include matched-row count, non-null metric count, "
                "and a null-safe SUM so the answer can distinguish zero from missing data."
            ),
            "column": col,
            "requires_matched_count": not has_matched_count,
            "requires_non_null_count": not has_non_null_count,
            "requires_null_safe_sum": not wrapped,
            "suggestions": [
                "COUNT_BIG(*) AS [MatchedRows]",
                f"COUNT({alias + '.' if alias else ''}{col}) AS [NonNull{col}Rows]",
                f"COALESCE(SUM({alias + '.' if alias else ''}{col}), 0) AS [MetricValue]",
            ],
        })
    return errors


def validate_sql_detailed(
    sql: str,
    known_tables: set[str],
    db_type: str = "snowflake",
    allowed_tables: set[str] | None = None,
    table_columns: dict[str, dict[str, str]] | None = None,
    semantic_context: dict | None = None,
) -> SqlValidationResult:
    """Return structured validation status and errors."""
    sql = normalize_generated_sql(sql, db_type)

    if sql.strip() == "CANNOT_GENERATE":
        return SqlValidationResult(
            False,
            "I was unable to generate a SQL query for that question using the available tables. Please rephrase or ask a different question.",
            "cannot_generate",
        )

    m = _DDL_DML.search(_strip_literals_and_comments(sql))
    if m:
        op = m.group(1).upper()
        return SqlValidationResult(
            False,
            f"The operation *{op}* is not permitted. Only SELECT queries are allowed.",
            "ddl",
        )

    if not _HAS_SQLGLOT:
        log.debug("sqlglot unavailable - skipping structural validation")
        return SqlValidationResult(True, "OK", "ok")

    dialect = _DIALECT.get(db_type, "snowflake")
    tree = None
    for d in [dialect, None]:
        try:
            tree = sqlglot.parse_one(sql, dialect=d)
            break
        except Exception:
            continue
    if tree is None:
        return SqlValidationResult(False, "SQL could not be parsed - please check for syntax errors.", "parse")

    known_upper = _expand_table_set(known_tables) or set()
    allowed_upper = _expand_table_set(allowed_tables)
    table_columns = {
        str(k).upper(): {str(ck).upper(): str(cv) for ck, cv in (v or {}).items()}
        for k, v in (table_columns or {}).items()
    }

    cte_aliases: set[str] = set()
    for cte in tree.find_all(sg_exp.CTE):
        if cte.alias:
            cte_aliases.add(cte.alias.upper())

    unknown: list[str] = []
    denied: list[str] = []
    alias_to_table: dict[str, str] = {}
    base_table_keys: list[str] = []
    distinct_base_tables: set[str] = set()

    for node in tree.find_all(sg_exp.Table):
        variants = _table_variants(node)
        if not variants:
            continue
        bare = variants[-1]
        if bare in cte_aliases:
            continue
        if not any(v in known_upper for v in variants):
            unknown.append(variants[0])
            continue
        if allowed_upper is not None and not any(v in allowed_upper for v in variants):
            denied.append(variants[0])
            continue

        distinct_base_tables.add(bare)
        table_key = _pick_table_key(variants, table_columns)
        if table_key:
            base_table_keys.append(table_key)
        alias = (node.alias_or_name or node.name or "").upper()
        if alias:
            alias_to_table[alias] = table_key
        for variant in variants:
            alias_to_table[variant] = table_key

    if denied:
        tables_list = ", ".join(sorted(set(denied)))
        return SqlValidationResult(
            False,
            (
                f"*You don't have access to the following table(s):* {tables_list}\n\n"
                "Please contact your administrator to request access, or rephrase your question using only the tables assigned to your group."
            ),
            "access_denied",
        )

    if unknown:
        return SqlValidationResult(
            False,
            (
                f"Table(s) not found in the connected database: *{', '.join(sorted(set(unknown)))}*\n\n"
                "Please rephrase your question using only the available tables."
            ),
            "unknown_table",
        )

    intent = (semantic_context or {}).get("intent") or {}
    if intent.get("wants_missing_records"):
        left_join_exists = any(
            (join.args.get("side") or "").upper() == "LEFT"
            for join in tree.find_all(sg_exp.Join)
        )
        null_test_exists = any(
            isinstance(is_node.expression, sg_exp.Null)
            for is_node in tree.find_all(sg_exp.Is)
        )
        if not left_join_exists or not null_test_exists:
            error = {
                "code": "anti_join_shape",
                "message": (
                    "Missing-record questions must use a source/parent table LEFT JOINed "
                    "to the missing-side table, with a right-side key IS NULL predicate. "
                    "A single-table NULL filter on the missing-side table is not enough."
                ),
                "requires_left_join": True,
                "requires_is_null": True,
            }
            return SqlValidationResult(
                False,
                (
                    "Anti-join shape is required for this missing-record question.\n\n"
                    "Use the source table containing the records to list, LEFT JOIN the "
                    "possibly-missing table, and filter with right_side_key IS NULL. "
                    "Do not answer this by querying only the missing-side table or only "
                    "checking a measure column for NULL."
                ),
                "anti_join_shape",
                [error],
            )

    select_aliases: set[str] = set()
    select_column_names: set[str] = set()
    for alias_node in tree.find_all(sg_exp.Alias):
        if alias_node.alias:
            select_aliases.add(alias_node.alias.upper())
    select = tree.find(sg_exp.Select)
    if select is not None:
        for expr in select.expressions:
            target = expr.this if isinstance(expr, sg_exp.Alias) else expr
            if isinstance(target, sg_exp.Column) and target.name:
                select_column_names.add(target.name.upper())

    if table_columns:
        date_errors = _find_date_key_format_errors(sql, table_columns)
        if date_errors:
            return SqlValidationResult(
                False,
                "\n".join(e["message"] for e in date_errors),
                "date_key_format",
                date_errors,
            )

        unique_base_keys = [k for k in dict.fromkeys(base_table_keys) if k in table_columns]
        # A bare column matching NO base table is only provably invalid when
        # every distinct table in the query has populated column metadata —
        # if any base table's columns are unknown (missing from table_columns
        # or an empty entry), the column might legitimately live there. The
        # same condition guards the single-table attribution below: with
        # partial metadata, "one known table" does not mean "one table".
        tables_fully_known = (
            bool(unique_base_keys)
            and len(unique_base_keys) == len(distinct_base_tables)
            and all(table_columns.get(k) for k in unique_base_keys)
        )
        unknown_cols_by_key: dict[tuple[str, str], dict] = {}
        for col_node in tree.find_all(sg_exp.Column):
            col_name = (col_node.name or "").upper()
            if not col_name or col_name == "*":
                continue
            table_ref = (col_node.table or "").upper()
            if table_ref in cte_aliases:
                continue

            if table_ref:
                table_key = alias_to_table.get(table_ref)
                if not table_key or table_key not in table_columns:
                    continue
            else:
                if col_name in select_aliases:
                    continue
                if len(unique_base_keys) == 1 and len(distinct_base_tables) == 1:
                    table_key = unique_base_keys[0]
                else:
                    # Unqualified column in a multi-table query: find the first base table
                    # that actually contains this column so we can track it correctly.
                    table_key = next(
                        (k for k in unique_base_keys if col_name in {str(c).upper() for c in (table_columns.get(k) or {})}),
                        "",
                    )
                if not table_key:
                    # No base table has this column. Previously a silent skip,
                    # which let hallucinated calendar columns through — e.g.
                    # "previous year cost amount" produced SQL referencing YR
                    # directly on the fact table when YR only exists on the
                    # DT_DMS date dimension. Flag it (with the tables that DO
                    # carry the column, so the retry knows to join them) when
                    # every table in the query has known columns.
                    if tables_fully_known:
                        key = ("", col_name)
                        if key not in unknown_cols_by_key:
                            candidate_tables = _tables_with_column(col_name, table_columns)
                            unknown_cols_by_key[key] = {
                                "code": "unknown_column",
                                "message": (
                                    f"Column {col_name} was not found on any table referenced in this query."
                                ),
                                "table": "",
                                "column": col_name,
                                "alias": "",
                                "suggestions": [],
                                "candidate_tables": candidate_tables,
                            }
                    continue

            cols = set(table_columns.get(table_key, {}))
            if cols and col_name not in cols:
                key = (table_key, col_name)
                if key in unknown_cols_by_key:
                    continue
                candidate_tables = _tables_with_column(col_name, table_columns)
                unknown_cols_by_key[key] = {
                    "code": "unknown_column",
                    "message": f"Column {col_name} was not found on table {table_key}.",
                    "table": table_key,
                    "column": col_name,
                    "alias": table_ref,
                    "suggestions": _column_suggestions(col_name, cols),
                    "candidate_tables": candidate_tables,
                }

        unknown_cols = list(unknown_cols_by_key.values())

        if unknown_cols:
            parts = []
            for err in unknown_cols[:5]:
                suggestions = err.get("suggestions") or []
                suffix = f" Suggestions: {', '.join(suggestions)}." if suggestions else ""
                candidate_tables = err.get("candidate_tables") or []
                if candidate_tables:
                    suffix += f" Exact column exists on: {', '.join(candidate_tables)} — join that table to use it."
                location = f"on {err['table']}" if err.get("table") else "on any table in this query"
                parts.append(f"{err['column']} {location}.{suffix}")
            return SqlValidationResult(
                False,
                (
                    "Column(s) not found in the connected database schema: "
                    + " ".join(parts)
                    + "\n\nUse exact column names from the Knowledge Base; do not remove underscores or invent aliases as source columns. "
                    "If a requested column exists on another table, change the source table or join to that table instead of reusing the same invalid table alias."
                ),
                "unknown_column",
                unknown_cols,
            )

        metric_formula_errors = _find_metric_formula_errors(sql, tree, semantic_context, table_columns)
        if metric_formula_errors:
            return SqlValidationResult(
                False,
                (
                    "Generated SQL ignored an approved metric formula. "
                    + " ".join(e["message"] for e in metric_formula_errors[:3])
                    + "\n\nUse the approved metric calculation exactly; do not replace it with a nearby semantic or knowledge-base column."
                ),
                "metric_formula_mismatch",
                metric_formula_errors,
            )

        null_agg_errors = _find_null_aggregate_diagnostic_errors(sql, tree)
        if null_agg_errors:
            return SqlValidationResult(
                False,
                (
                    "Filtered aggregate queries must be null-aware. Include matched-row "
                    "count, non-null metric count, and COALESCE/ISNULL around SUM() so "
                    "the answer can explain when records exist but metric values are missing."
                ),
                "null_aggregate_diagnostic",
                null_agg_errors,
            )

        field_plan = (semantic_context or {}).get("semantic_plan") or {}
        if field_plan.get("enabled") and field_plan.get("fields"):
            used_columns: set[tuple[str, str]] = set()
            for col_node in tree.find_all(sg_exp.Column):
                col_name = (col_node.name or "").upper()
                if not col_name or col_name == "*":
                    continue
                table_ref = (col_node.table or "").upper()
                if table_ref in cte_aliases:
                    continue
                table_key = ""
                if table_ref:
                    table_key = alias_to_table.get(table_ref, "")
                elif len(unique_base_keys) == 1:
                    table_key = unique_base_keys[0]
                else:
                    # Unqualified column in a multi-table query — search all base
                    # tables to find which one owns it (mirrors the unknown-column
                    # scanner at line ~598 to avoid false plan-mismatch errors).
                    table_key = next(
                        (k for k in unique_base_keys if col_name in {str(c).upper() for c in (table_columns.get(k) or {})}),
                        "",
                    )
                if table_key:
                    used_columns.add((table_key, col_name))

            sql_has_group_by = tree.find(sg_exp.Group) is not None

            # Collect column names that appear in any approved metric formula —
            # if the plan field is satisfied by a metric, skip the raw-column check.
            _metric_formula_cols: set[str] = set()
            for _mf in (semantic_context or {}).get("metric_formulas") or []:
                _tpl = (_mf.get("sql_template") or "").upper()
                for _m in re.finditer(r'\b([A-Z][A-Z0-9_]*)\b', _tpl):
                    _metric_formula_cols.add(_m.group(1))

            missing_plan_fields: list[dict] = []
            for field in field_plan.get("fields") or []:
                # Optional fields (e.g. date-role dimension keys) are hints, not
                # hard requirements — the LLM may satisfy the same intent another
                # valid way (e.g. deriving a period directly from a YYYYMMDD fact
                # key instead of joining the date dimension). Mirrors the
                # enforcement=="optional" skip already applied to join edges below.
                if field.get("enforcement") == "optional":
                    continue
                # Dimension display fields (e.g. CUS_NM) are only required when
                # the SQL groups results.  Pure aggregates don't need the join.
                if field.get("display_required") and not sql_has_group_by:
                    continue
                plan_col = (field.get("column") or "").upper()
                plan_table = _find_table_with_column(table_columns, field.get("table") or "", plan_col)
                if not plan_col or not plan_table:
                    continue
                # If the column is referenced by an approved metric formula,
                # the formula is the authoritative source — skip this check.
                if plan_col in _metric_formula_cols:
                    continue
                if not any(_table_matches(used_table, plan_table) and used_col == plan_col for used_table, used_col in used_columns):
                    missing_plan_fields.append({
                        "code": "field_plan_mismatch",
                        "message": f"SQL did not use required semantic field {_format_plan_field(field)}.",
                        "table": plan_table,
                        "column": plan_col,
                        "term": field.get("term", ""),
                    })

            # Superseded columns: an admin-approved mapping actively forbids
            # the old rival column for the matched business term.  Presence of
            # the approved field alone is not enough — SQL selecting BOTH (or
            # only the rival) silently answers with the wrong data.
            for avoid in field_plan.get("avoid_columns") or []:
                avoid_col = (avoid.get("column") or "").upper()
                avoid_table = _find_table_with_column(table_columns, avoid.get("table") or "", avoid_col)
                if not avoid_col or not avoid_table:
                    continue
                if avoid_col in _metric_formula_cols:
                    continue    # an approved metric formula owns this column
                if any(_table_matches(used_table, avoid_table) and used_col == avoid_col for used_table, used_col in used_columns):
                    term = str(avoid.get("term") or "this term")
                    use_instead = f"{avoid.get('use_instead_table')}.{avoid.get('use_instead_column')}"
                    missing_plan_fields.append({
                        "code": "field_plan_mismatch",
                        "message": (
                            f"SQL used {avoid_table}.{avoid_col}, but the admin-approved source "
                            f"for '{term}' is {use_instead}."
                        ),
                        "table": avoid_table,
                        "column": avoid_col,
                        "term": term,
                        "avoided_column": avoid_col,
                        "avoided_table": avoid_table,
                        "use_instead_table": avoid.get("use_instead_table", ""),
                        "use_instead_column": avoid.get("use_instead_column", ""),
                    })

            # Build a set of all (left_col, right_col) pairs that appear in any
            # JOIN ... ON equality condition via AST — more precise than raw substring
            # matching which would false-pass if a column name appears only in SELECT.
            join_eq_pairs: set[tuple[str, str]] = set()
            for join_node in tree.find_all(sg_exp.Join):
                for eq_node in join_node.find_all(sg_exp.EQ):
                    left_expr = eq_node.left
                    right_expr = eq_node.right
                    if isinstance(left_expr, sg_exp.Column) and isinstance(right_expr, sg_exp.Column):
                        lc = (left_expr.name or "").upper()
                        rc = (right_expr.name or "").upper()
                        if lc and rc:
                            join_eq_pairs.add((lc, rc))
                            join_eq_pairs.add((rc, lc))  # symmetric

            required_join_errors: list[dict] = []
            for edge in field_plan.get("joins") or []:
                if edge.get("enforcement") == "optional":
                    continue
                for left_col, right_col in edge.get("conditions") or []:
                    left_col_u = str(left_col).upper()
                    right_col_u = str(right_col).upper()
                    if (left_col_u, right_col_u) not in join_eq_pairs:
                        required_join_errors.append({
                            "code": "field_plan_join_missing",
                            "message": (
                                f"Required semantic join condition was not visible: "
                                f"{edge.get('from')}.{left_col_u} = {edge.get('to')}.{right_col_u}."
                            ),
                            "left_table": edge.get("from", ""),
                            "right_table": edge.get("to", ""),
                            "left_column": left_col_u,
                            "right_column": right_col_u,
                        })
                        break

            if missing_plan_fields or required_join_errors:
                errors = missing_plan_fields + required_join_errors
                expected = "; ".join(e["message"] for e in errors[:5])
                return SqlValidationResult(
                    False,
                    (
                        "Generated SQL does not follow the semantic field-source plan. "
                        + expected
                        + "\n\nUse the exact table.column mappings and required joins from the semantic field-source plan."
                    ),
                    "field_plan_mismatch",
                    errors,
                )

    if select_aliases:
        bad_order: set[str] = set()
        for ordered_node in tree.find_all(sg_exp.Ordered):
            col_expr = ordered_node.this
            if isinstance(col_expr, sg_exp.Column) and not col_expr.table:
                col_name = col_expr.name
                if col_name and col_name.upper() not in select_aliases and col_name.upper() not in select_column_names:
                    bad_order.add(col_name)
        if bad_order:
            alias_list = ", ".join(sorted(bad_order))
            log.warning("ORDER BY alias drift detected: %s not in SELECT aliases %s", alias_list, select_aliases)
            return SqlValidationResult(
                False,
                (
                    f"ORDER BY references column(s) not defined as a SELECT alias: *{alias_list}*\n\n"
                    "Make sure every ORDER BY name matches the exact alias used in SELECT."
                ),
                "order_alias_mismatch",
            )

    return SqlValidationResult(True, "OK", "ok")


def validate_sql(
    sql: str,
    known_tables: set[str],
    db_type: str = "snowflake",
    allowed_tables: set[str] | None = None,
    table_columns: dict[str, dict[str, str]] | None = None,
    semantic_context: dict | None = None,
) -> tuple[bool, str, str]:
    """
    Backward-compatible tuple API.

    Returns (is_valid, reason, code). For structured error details, call
    validate_sql_detailed().
    """
    result = validate_sql_detailed(sql, known_tables, db_type, allowed_tables, table_columns, semantic_context)
    return result.ok, result.reason, result.code


def load_known_tables(schema_dir: str) -> set[str]:
    """Convenience re-export of core.schema.load_known_tables."""
    from core.schema import load_known_tables as _load
    return _load(schema_dir)
