"""What differs between Snowflake, Azure SQL, Oracle and DuckDB, in one place.

The compiler builds sqlglot expressions and prints them per dialect. Where
sqlglot does not print a construct faithfully for a dialect, the construct is a
fragment here, parsed from that dialect's own spelling so it prints back
unchanged (sqlglot 30 prints Oracle month truncation as ``TIMESTAMP_TRUNC``,
which Oracle rejects; T-SQL weeks as ``DATETRUNC(WEEK, ...)``, which depends on
the server's ``DATEFIRST``; and ``APPROX_COUNT_DISTINCT`` as ``APPROX_DISTINCT``
for T-SQL and Oracle).

Bootstrap profiling SQL is written as text from the same fragments, with every
identifier quoted by :func:`quote`.
"""

from __future__ import annotations

import datetime as dt
import functools
import re

import sqlglot
from sqlglot import exp

DIALECTS = ("snowflake", "tsql", "oracle", "duckdb")
_BY_DB_TYPE = {"snowflake": "snowflake", "azure_sql": "tsql", "oracle": "oracle", "duckdb": "duckdb"}


def for_db_type(db_type: str) -> str:
    try:
        return _BY_DB_TYPE[db_type]
    except KeyError:
        raise ValueError(f"no SQL dialect for database type {db_type!r}") from None


# Words that must be quoted when used as identifiers in any of the four dialects.
# A superset is harmless (an unneeded quote changes nothing); a missing word breaks
# a query, so the list errs long.
RESERVED = frozenset("""
ACCESS ADD ALL ALTER AND ANY AS ASC AUDIT BETWEEN BREAK BROWSE BULK BY CASCADE CASE CAST CHECK CHECKPOINT
CLOSE CLUSTER CLUSTERED COALESCE COLLATE COLUMN COMMENT COMMIT COMPRESS COMPUTE CONNECT CONNECTION CONSTRAINT
CONTAINS CONTINUE CONVERT CREATE CROSS CURRENT CURRENT_DATE CURRENT_TIME CURRENT_TIMESTAMP CURRENT_USER CURSOR
DATABASE DATE DBCC DEALLOCATE DECIMAL DECLARE DEFAULT DELETE DENY DESC DISK DISTINCT DISTRIBUTED DOUBLE DROP
DUMP ELSE END ERRLVL ESCAPE EXCEPT EXCLUSIVE EXEC EXECUTE EXISTS EXIT EXTERNAL FETCH FILE FILLFACTOR FLOAT FOLLOWING
FOR FOREIGN FREETEXT FROM FULL FUNCTION GET GOTO GRANT GROUP HAVING HOLDLOCK IDENTIFIED IDENTITY IF ILIKE IMMEDIATE
IN INCREMENT INDEX INITIAL INNER INSERT INTEGER INTERSECT INTERVAL INTO IS ISSUE JOIN KEY KILL LATERAL LEFT LEVEL
LIKE LIMIT LINENO LOAD LOCK LONG MAXEXTENTS MERGE MINUS MLSLABEL MODE MODIFY NATIONAL NATURAL NOAUDIT NOCHECK
NOCOMPRESS NONCLUSTERED NOT NOWAIT NULL NULLIF NUMBER OF OFF OFFLINE OFFSET OFFSETS ON ONLINE OPEN OPTION OR
ORDER OUTER OVER PCTFREE PERCENT PIVOT PLAN PRECISION PRIMARY PRINT PRIOR PRIVILEGES PROC PROCEDURE PUBLIC PUT
QUALIFY RAISERROR RAW READ READTEXT REAL RECONFIGURE REFERENCES REGEXP RENAME REPLICATION RESOURCE RESTORE
RESTRICT RETURN REVERT REVOKE RIGHT RLIKE ROLLBACK ROW ROWCOUNT ROWGUIDCOL ROWID ROWNUM ROWS RULE SAMPLE SAVE
SCHEMA SELECT SESSION SESSION_USER SET SETUSER SHARE SHUTDOWN SIZE SMALLINT SOME START STATISTICS SUCCESSFUL
SYNONYM SYSDATE SYSTEM_USER TABLE TABLESAMPLE TEXTSIZE THEN TO TOP TRAN TRANSACTION TRIGGER TRUE TRUNCATE
TRY_CONVERT UID UNION UNIQUE UNPIVOT UPDATE UPDATETEXT USE USER USING VALIDATE VALUE VALUES VARCHAR VARCHAR2 VARYING
VIEW WAITFOR WHEN WHENEVER WHERE WHILE WINDOW WITH WITHIN WRITETEXT YEAR MONTH DAY HOUR MINUTE SECOND
""".split())

_SIMPLE_UPPER = re.compile(r"[A-Z_][A-Z0-9_$#]*")
_SIMPLE_ANY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def needs_quote(name: str, dialect: str) -> bool:
    """Whether an identifier, in its stored spelling, must be quoted to mean itself."""
    if name.upper() in RESERVED:
        return True
    if dialect in ("snowflake", "oracle"):
        # Unquoted names fold to upper case: anything else was created quoted.
        return not _SIMPLE_UPPER.fullmatch(name)
    return not _SIMPLE_ANY.fullmatch(name)


def ident(name: str, dialect: str) -> exp.Identifier:
    return exp.to_identifier(name, quoted=needs_quote(name, dialect))


def quote(name: str, dialect: str) -> str:
    """An identifier as SQL text for ``dialect``, quoted only when it has to be."""
    return ident(name, dialect).sql(dialect=dialect)


def table_sql(database: str, schema: str, name: str, dialect: str) -> str:
    return ".".join(quote(part, dialect) for part in (database, schema, name) if part)


def table_expr(database: str, schema: str, name: str, dialect: str, alias: str | None = None) -> exp.Table:
    table = exp.Table(this=ident(name, dialect),
                      db=ident(schema, dialect) if schema else None,
                      catalog=ident(database, dialect) if database and schema else None)
    if alias:
        table.set("alias", exp.TableAlias(this=ident(alias, dialect)))
    return table


# ── arithmetic that keeps its meaning ──────────────────────────────────────
# sqlglot prints a tree as written: Mod(Add(a, b), c) comes out as "a + b % c".
# Every operand that is itself an operation is parenthesised here, so an
# expression built in code always means what its tree says.


def _wrap(value: exp.Expression) -> exp.Expression:
    if isinstance(value, (exp.Binary, exp.Unary)) and not isinstance(value, exp.Paren):
        return exp.Paren(this=value)
    return value


def _number(value: exp.Expression | int | float) -> exp.Expression:
    return exp.Literal.number(value) if isinstance(value, (int, float)) else value


def add(a: exp.Expression | int | float, b: exp.Expression | int | float) -> exp.Expression:
    return exp.Add(this=_wrap(_number(a)), expression=_wrap(_number(b)))


def sub(a: exp.Expression | int | float, b: exp.Expression | int | float) -> exp.Expression:
    return exp.Sub(this=_wrap(_number(a)), expression=_wrap(_number(b)))


def mul(a: exp.Expression | int | float, b: exp.Expression | int | float) -> exp.Expression:
    return exp.Mul(this=_wrap(_number(a)), expression=_wrap(_number(b)))


def div(a: exp.Expression | int | float, b: exp.Expression | int | float) -> exp.Expression:
    return exp.Div(this=_wrap(_number(a)), expression=_wrap(_number(b)))


def mod(a: exp.Expression | int | float, b: exp.Expression | int | float) -> exp.Expression:
    return exp.Mod(this=_wrap(_number(a)), expression=_wrap(_number(b)))


# ── fragments ──────────────────────────────────────────────────────────────

_PERIOD_START = {
    "snowflake": {
        "day": "CAST(__X__ AS DATE)",
        "week": "DATEADD(DAY, 1 - DAYOFWEEKISO(__X__), CAST(__X__ AS DATE))",
        "month": "DATE_TRUNC('MONTH', CAST(__X__ AS DATE))",
        "quarter": "DATE_TRUNC('QUARTER', CAST(__X__ AS DATE))",
        "year": "DATE_TRUNC('YEAR', CAST(__X__ AS DATE))",
    },
    "tsql": {
        "day": "CAST(__X__ AS DATE)",
        # Monday whatever the server's DATEFIRST: (weekday + @@DATEFIRST + 5) % 7 is 0 on a Monday.
        "week": "DATEADD(DAY, -((DATEPART(WEEKDAY, __X__) + @@DATEFIRST + 5) % 7), CAST(__X__ AS DATE))",
        "month": "DATEFROMPARTS(YEAR(__X__), MONTH(__X__), 1)",
        "quarter": "DATEFROMPARTS(YEAR(__X__), (DATEPART(QUARTER, __X__) - 1) * 3 + 1, 1)",
        "year": "DATEFROMPARTS(YEAR(__X__), 1, 1)",
    },
    "oracle": {
        "day": "TRUNC(__X__)",
        "week": "TRUNC(__X__, 'IW')",
        "month": "TRUNC(__X__, 'MM')",
        "quarter": "TRUNC(__X__, 'Q')",
        "year": "TRUNC(__X__, 'YYYY')",
    },
    "duckdb": {
        "day": "CAST(__X__ AS DATE)",
        "week": "CAST(DATE_TRUNC('WEEK', __X__) AS DATE)",
        "month": "CAST(DATE_TRUNC('MONTH', __X__) AS DATE)",
        "quarter": "CAST(DATE_TRUNC('QUARTER', __X__) AS DATE)",
        "year": "CAST(DATE_TRUNC('YEAR', __X__) AS DATE)",
    },
}

GRAINS = ("day", "week", "month", "quarter", "year")


@functools.lru_cache(maxsize=None)
def _template(text: str, dialect: str) -> exp.Expression:
    return sqlglot.parse_one(f"SELECT {text}", read=dialect).expressions[0]


def _fill(text: str, dialect: str, value: exp.Expression) -> exp.Expression:
    def swap(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Column) and node.name == "__X__" and not node.table:
            return value.copy()
        return node

    return _template(text, dialect).copy().transform(swap)


def period_start(value: exp.Expression, grain: str, dialect: str) -> exp.Expression:
    """The first day of the ``grain`` period holding ``value``, as a DATE. Weeks start on Monday."""
    try:
        text = _PERIOD_START[dialect][grain]
    except KeyError:
        raise ValueError(f"no {grain!r} period in {dialect!r}") from None
    return _fill(text, dialect, value)


_DATE_PARTS = {
    "snowflake": {"year": "YEAR(__X__)", "month": "MONTH(__X__)", "day": "DAY(__X__)", "quarter": "QUARTER(__X__)",
                  "isodow": "DAYOFWEEKISO(__X__)", "isoweek": "WEEKISO(__X__)"},
    "tsql": {"year": "YEAR(__X__)", "month": "MONTH(__X__)", "day": "DAY(__X__)",
             "quarter": "DATEPART(QUARTER, __X__)",
             "isodow": "((DATEPART(WEEKDAY, __X__) + @@DATEFIRST + 5) % 7) + 1",
             "isoweek": "DATEPART(ISO_WEEK, __X__)"},
    "oracle": {"year": "EXTRACT(YEAR FROM __X__)", "month": "EXTRACT(MONTH FROM __X__)",
               "day": "EXTRACT(DAY FROM __X__)", "quarter": "TO_NUMBER(TO_CHAR(__X__, 'Q'))",
               "isodow": "(TRUNC(__X__) - TRUNC(__X__, 'IW')) + 1", "isoweek": "TO_NUMBER(TO_CHAR(__X__, 'IW'))"},
    "duckdb": {"year": "YEAR(__X__)", "month": "MONTH(__X__)", "day": "DAY(__X__)", "quarter": "QUARTER(__X__)",
               "isodow": "ISODOW(__X__)", "isoweek": "WEEKOFYEAR(__X__)"},
}
DATE_PARTS = ("year", "month", "day", "quarter", "isodow", "isoweek")


def date_part(value: exp.Expression, part: str, dialect: str) -> exp.Expression:
    """A number from a date: year, month, day, quarter, ISO day of week (Monday 1), ISO week."""
    try:
        text = _DATE_PARTS[dialect][part]
    except KeyError:
        raise ValueError(f"no {part!r} date part in {dialect!r}") from None
    return _fill(text, dialect, value)


def date_literal(day: dt.date, dialect: str) -> exp.Expression:
    text = day.isoformat()
    if dialect == "oracle":
        # Independent of NLS_DATE_FORMAT.
        return exp.Anonymous(this="TO_DATE", expressions=[exp.Literal.string(text), exp.Literal.string("YYYY-MM-DD")])
    return exp.Cast(this=exp.Literal.string(text), to=exp.DataType.build("DATE"))


def approx_distinct(value: exp.Expression, dialect: str) -> exp.Expression:
    return exp.Anonymous(this="APPROX_COUNT_DISTINCT", expressions=[value])


def text_length(value: exp.Expression, dialect: str) -> exp.Expression:
    return exp.Anonymous(this="LEN" if dialect == "tsql" else "LENGTH", expressions=[value])


def as_text(value: exp.Expression, dialect: str) -> exp.Expression:
    """A value as text for profiling output (codes and names, never dates)."""
    if dialect == "oracle":
        return exp.Anonymous(this="TO_CHAR", expressions=[value])
    target = {"tsql": "NVARCHAR(400)", "snowflake": "VARCHAR", "duckdb": "VARCHAR"}[dialect]
    return exp.Cast(this=value, to=exp.DataType.build(target, dialect=dialect))


def day_start(value: exp.Expression, dialect: str) -> exp.Expression:
    """Midnight of a timestamp's day, in the timestamp's own type (for time-of-day tests)."""
    text = {"snowflake": "DATE_TRUNC('DAY', __X__)", "tsql": "CAST(CAST(__X__ AS DATE) AS DATETIME2)",
            "oracle": "TRUNC(__X__)", "duckdb": "DATE_TRUNC('DAY', __X__)"}[dialect]
    return _fill(text, dialect, value)


def sample_clause(dialect: str, *, rows: int, total_rows: int) -> str:
    """Text placed after a table reference to profile a sample of about ``rows`` rows."""
    percent = max(0.01, min(100.0, 100.0 * rows / max(total_rows, 1)))
    if dialect == "snowflake":
        return f"SAMPLE ({rows} ROWS)"
    if dialect == "tsql":
        return f"TABLESAMPLE ({percent:.4f} PERCENT)"
    if dialect == "oracle":
        return f"SAMPLE ({percent:.4f})"
    return f"USING SAMPLE {rows} ROWS"


def render(expression: exp.Expression, dialect: str) -> str:
    """Print an expression for ``dialect`` and make sure the dialect can read it back."""
    sql = expression.sql(dialect=dialect)
    sqlglot.parse_one(sql, read=dialect)
    return sql
