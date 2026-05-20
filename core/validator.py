"""
core/validator.py

Three-layer SQL safety net.

Layer 1 — DDL/DML block      : regex, immediate rejection on any destructive keyword
Layer 2 — AST parse           : sqlglot with correct dialect per DB type
Layer 3 — CTE-aware table check: real base tables checked against _schema.json;
                                  CTE aliases excluded to avoid false positives
"""

import re
import logging

log = logging.getLogger("querybot.validator")

try:
    import sqlglot
    from sqlglot import exp as sg_exp
    _HAS_SQLGLOT = True
except ImportError:
    _HAS_SQLGLOT = False
    log.warning("sqlglot not installed — structural SQL validation disabled")

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
    "oracle":    "oracle",
    "azure_sql": "tsql",
}


def validate_sql(
    sql: str,
    known_tables: set[str],
    db_type: str = "snowflake",
    allowed_tables: set[str] | None = None,
) -> tuple[bool, str, str]:
    """
    Returns (is_valid, reason, code).

    code values:
      "ok"             — SQL passed all layers
      "cannot_generate" — LLM could not produce SQL (sentinel)
      "ddl"            — non-SELECT keyword detected
      "parse"          — SQL could not be parsed
      "unknown_table"  — references a table that doesn't exist in the DB
      "access_denied"  — references a real table the user isn't allowed to query

    known_tables  : every base table that exists in the connected DB (uppercase).
    allowed_tables: user's permitted subset (uppercase). None = admin / unrestricted.
    """
    if sql.strip() == "CANNOT_GENERATE":
        return False, (
            "I was unable to generate a SQL query for that question "
            "using the available tables. Please rephrase or ask a different question."
        ), "cannot_generate"

    # Layer 1: DDL/DML block
    m = _DDL_DML.search(sql)
    if m:
        op = m.group(1).upper()
        return False, (
            f"The operation *{op}* is not permitted. "
            "Only SELECT queries are allowed."
        ), "ddl"

    # Layers 2 + 3: structural validation
    if not _HAS_SQLGLOT:
        log.debug("sqlglot unavailable — skipping structural validation")
        return True, "OK", "ok"

    dialect = _DIALECT.get(db_type, "snowflake")

    tree = None
    for d in [dialect, None]:
        try:
            tree = sqlglot.parse_one(sql, dialect=d)
            break
        except Exception:
            continue

    if tree is None:
        return False, "SQL could not be parsed — please check for syntax errors.", "parse"

    cte_aliases: set[str] = set()
    for cte in tree.find_all(sg_exp.CTE):
        if cte.alias:
            cte_aliases.add(cte.alias.upper())

    # Expand allowed_tables to include all name variants (bare, 2-part, FQN)
    # so the ACL check passes regardless of how the LLM qualified the table.
    # E.g. if allowed = {"MYDB.HR.EMPLOYEES"}, the check
    # must also pass for bare "EMPLOYEES" from sqlglot's node.name.
    def _expand_table_set(tables: set[str] | None) -> set[str] | None:
        if tables is None:
            return None
        expanded: set[str] = set()
        for t in tables:
            upper = t.upper()
            expanded.add(upper)
            parts = upper.split(".")
            if len(parts) >= 2:
                expanded.add(parts[-1])           # bare name
                expanded.add(".".join(parts[-2:])) # schema.table
        return expanded

    unknown:  list[str] = []
    denied:   list[str] = []
    allowed_upper = _expand_table_set(allowed_tables)

    for node in tree.find_all(sg_exp.Table):
        ref = node.name
        if not ref:
            continue
        ref_upper = ref.upper()
        if ref_upper in cte_aliases:
            continue
        if ref_upper not in known_tables:
            unknown.append(ref)
            continue
        if allowed_upper is not None and ref_upper not in allowed_upper:
            denied.append(ref)

    # Access denial takes precedence over unknown — it is a more
    # actionable error for the user, and also prevents information
    # leakage ("which tables exist in the DB").
    if denied:
        tables_list = ", ".join(sorted(set(denied)))
        return False, (
            f"🔒 *You don't have access to the following table(s):* {tables_list}\n\n"
            "Please contact your administrator to request access, or rephrase "
            "your question using only the tables assigned to your group."
        ), "access_denied"

    if unknown:
        return False, (
            f"Table(s) not found in the connected database: "
            f"*{', '.join(sorted(set(unknown)))}*\n\n"
            "Please rephrase your question using only the available tables."
        ), "unknown_table"

    # ── Layer 4: ORDER BY alias drift check ──────────────────────────────────
    # Collect SELECT-level aliases.  Only run when aliases are present so we
    # don't false-positive on queries that ORDER BY bare column names.
    select_aliases: set[str] = set()
    for alias_node in tree.find_all(sg_exp.Alias):
        if alias_node.alias:
            select_aliases.add(alias_node.alias.upper())

    if select_aliases:
        bad_order: list[str] = []
        for ordered_node in tree.find_all(sg_exp.Ordered):
            col_expr = ordered_node.this
            # Only check unqualified column references (no table prefix).
            # Qualified refs like f.AMOUNT are real column names, not aliases.
            if isinstance(col_expr, sg_exp.Column) and not col_expr.table:
                col_name = col_expr.name
                if col_name and col_name.upper() not in select_aliases:
                    bad_order.append(col_name)

        if bad_order:
            alias_list = ", ".join(sorted(set(bad_order)))
            log.warning(
                "ORDER BY alias drift detected: %s not in SELECT aliases %s",
                alias_list, select_aliases,
            )
            return False, (
                f"ORDER BY references column(s) not defined as a SELECT alias: "
                f"*{alias_list}*\n\n"
                "Make sure every ORDER BY name matches the exact alias used in SELECT."
            ), "order_alias_mismatch"

    return True, "OK", "ok"


def load_known_tables(schema_dir: str) -> set[str]:
    """Load set of uppercase table names from _schema.json.
    
    NOTE: This is a convenience re-export. The authoritative implementation
    lives in core.schema.load_known_tables. Both are kept in sync.
    """
    from core.schema import load_known_tables as _load
    return _load(schema_dir)
