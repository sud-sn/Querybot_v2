"""What differs between Snowflake, Azure SQL, Oracle, PostgreSQL and DuckDB, in one place.

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

DIALECTS = ("snowflake", "tsql", "oracle", "postgres", "duckdb")
_BY_DB_TYPE = {"snowflake": "snowflake", "azure_sql": "tsql", "oracle": "oracle", "postgres": "postgres",
               "postgresql": "postgres", "duckdb": "duckdb", "excel": "duckdb"}


def for_db_type(db_type: str) -> str:
    try:
        return _BY_DB_TYPE[db_type]
    except KeyError:
        raise ValueError(f"no SQL dialect for database type {db_type!r}") from None


# Words that must be quoted when used as identifiers in any of the five dialects.
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
ANALYSE ANALYZE ARRAY ASYMMETRIC BOTH COLLATION CONCURRENTLY DEFERRABLE DO FREEZE INITIALLY ISNULL LEADING
LOCALTIME LOCALTIMESTAMP NOTNULL ONLY OVERLAPS PLACING RETURNING SIMILAR SYMMETRIC TRAILING VARIADIC VERBOSE
""".split())

_SIMPLE_UPPER = re.compile(r"[A-Z_][A-Z0-9_$#]*")
_SIMPLE_ANY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SIMPLE_LOWER = re.compile(r"[a-z_][a-z0-9_$]*")


def needs_quote(name: str, dialect: str) -> bool:
    """Whether an identifier, in its stored spelling, must be quoted to mean itself."""
    if name.upper() in RESERVED:
        return True
    if dialect in ("snowflake", "oracle"):
        # Unquoted names fold to upper case: anything else was created quoted.
        return not _SIMPLE_UPPER.fullmatch(name)
    if dialect == "postgres":
        # Unquoted names fold to lower case: a name with a capital was created quoted.
        return not _SIMPLE_LOWER.fullmatch(name)
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


def total(value: exp.Expression, data_type: str, dialect: str) -> exp.Expression:
    """SUM of a column that cannot overflow.

    Azure SQL sums an int column as an int and stops the query past
    2,147,483,647 (error 8115, "arithmetic overflow"): a stock quantity summed
    over a large table gets there. Its whole numbers are summed as a bigint.
    """
    if dialect == "tsql" and data_type == "integer":
        value = exp.Cast(this=value, to=exp.DataType.build("BIGINT"))
    return exp.Sum(this=value)


class FloorDiv(exp.Expression, exp.Func):
    """a / b rounded down to a whole number: printed per dialect (see :func:`floor_div`)."""

    arg_types = {"this": True, "expression": True}


def _floor_div_sql(self, node: FloorDiv) -> str:
    return self.sql(exp.Floor(this=exp.Div(this=node.this, expression=node.expression, typed=True)))


def _floor_div_postgres(self, node: FloorDiv) -> str:
    # FLOOR returns a double precision, whose remainder PostgreSQL refuses ("operator does not exist:
    # double precision % integer"): the whole number is cast back to one.
    return self.sql(exp.Cast(this=exp.Floor(this=exp.Div(this=node.this, expression=node.expression, typed=True)),
                             to=exp.DataType.build("BIGINT")))


def _register_floor_div() -> None:
    from sqlglot import generator as generator_module
    from sqlglot.dialects.dialect import Dialect
    from sqlglot.generator import Generator

    Generator.TRANSFORMS[FloorDiv] = _floor_div_sql
    for name in DIALECTS:
        Dialect.get_or_raise(name).generator_class.TRANSFORMS[FloorDiv] = (
            _floor_div_postgres if name == "postgres" else _floor_div_sql)
    # sqlglot (30.x) keeps each dialect's table of how to write each expression from the first SQL it writes
    # in that dialect. A server that wrote any SQL before this module was imported (the question pipeline
    # checking a first question) kept a table without FloorDiv, and every later key read wrote FLOOR_DIV(),
    # a function no database has: Learn then left out every table with a date key. The tables are rebuilt.
    getattr(generator_module, "_DISPATCH_CACHE", {}).clear()


_register_floor_div()


def floor_div(a: exp.Expression | int | float, b: exp.Expression | int | float) -> exp.Expression:
    """a / b rounded down, as a whole number: a key's leading digits (20260131 -> 202601).

    div() is a true division, which sqlglot writes for Azure SQL as
    CAST(a AS FLOAT) / b; Azure SQL then refuses the remainder of it (error 402,
    "float and int are incompatible in the modulo operator"), so the month of a
    yyyymmdd key could not be read there. FLOOR keeps an integer an integer on the
    others; PostgreSQL's FLOOR gives a double precision, refused there the same way,
    and is cast back to a whole number.
    """
    return FloorDiv(this=_wrap(_number(a)), expression=_wrap(_number(b)))


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
    "postgres": {
        "day": "CAST(__X__ AS DATE)",
        "week": "CAST(DATE_TRUNC('WEEK', __X__) AS DATE)",   # ISO weeks: Monday
        "month": "CAST(DATE_TRUNC('MONTH', __X__) AS DATE)",
        "quarter": "CAST(DATE_TRUNC('QUARTER', __X__) AS DATE)",
        "year": "CAST(DATE_TRUNC('YEAR', __X__) AS DATE)",
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


def _fill(text: str, dialect: str, value: exp.Expression, n: exp.Expression | None = None) -> exp.Expression:
    def swap(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Column) and not node.table and node.name in ("__X__", "__N__"):
            return (value if node.name == "__X__" else n or value).copy()
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
                  "isodow": "DAYOFWEEKISO(__X__)", "isoweek": "WEEKISO(__X__)", "hour": "HOUR(__X__)",
                  "minute": "MINUTE(__X__)"},
    "tsql": {"year": "YEAR(__X__)", "month": "MONTH(__X__)", "day": "DAY(__X__)",
             "quarter": "DATEPART(QUARTER, __X__)",
             "isodow": "((DATEPART(WEEKDAY, __X__) + @@DATEFIRST + 5) % 7) + 1",
             "isoweek": "DATEPART(ISO_WEEK, __X__)", "hour": "DATEPART(HOUR, __X__)",
             "minute": "DATEPART(MINUTE, __X__)"},
    "oracle": {"year": "EXTRACT(YEAR FROM __X__)", "month": "EXTRACT(MONTH FROM __X__)",
               "day": "EXTRACT(DAY FROM __X__)", "quarter": "TO_NUMBER(TO_CHAR(__X__, 'Q'))",
               "isodow": "(TRUNC(__X__) - TRUNC(__X__, 'IW')) + 1", "isoweek": "TO_NUMBER(TO_CHAR(__X__, 'IW'))",
               "hour": "TO_NUMBER(TO_CHAR(__X__, 'HH24'))", "minute": "TO_NUMBER(TO_CHAR(__X__, 'MI'))"},
    # EXTRACT gives a numeric: a whole number here, as the others give.
    "postgres": {part: f"CAST(EXTRACT({field} FROM __X__) AS INTEGER)" for part, field in (
        ("year", "YEAR"), ("month", "MONTH"), ("day", "DAY"), ("quarter", "QUARTER"), ("isodow", "ISODOW"),
        ("isoweek", "WEEK"), ("hour", "HOUR"), ("minute", "MINUTE"))},
    "duckdb": {"year": "YEAR(__X__)", "month": "MONTH(__X__)", "day": "DAY(__X__)", "quarter": "QUARTER(__X__)",
               "isodow": "ISODOW(__X__)", "isoweek": "WEEKOFYEAR(__X__)", "hour": "HOUR(__X__)",
               "minute": "MINUTE(__X__)"},
}
DATE_PARTS = ("year", "month", "day", "quarter", "isodow", "isoweek", "hour", "minute")


def date_part(value: exp.Expression, part: str, dialect: str) -> exp.Expression:
    """A number from a date: year, month, day, quarter, ISO day of week (Monday 1), ISO week, hour, minute."""
    try:
        text = _DATE_PARTS[dialect][part]
    except KeyError:
        raise ValueError(f"no {part!r} date part in {dialect!r}") from None
    return _fill(text, dialect, value)


_FROM_NUMBER = {
    "tsql": {"yyyymmdd": "TRY_CONVERT(DATE, CONVERT(VARCHAR(8), CAST(__X__ AS BIGINT)), 112)",
             "yyyymm": "TRY_CONVERT(DATE, CONVERT(VARCHAR(6), CAST(__X__ AS BIGINT)) + '01', 112)"},
    "oracle": {fmt.lower(): f"CASE WHEN VALIDATE_CONVERSION(TO_CHAR(TRUNC(__X__)) AS DATE, '{fmt}') = 1 "
                            f"THEN TO_DATE(TO_CHAR(TRUNC(__X__)), '{fmt}') END" for fmt in ("YYYYMMDD", "YYYYMM")},
    # PostgreSQL has no conversion that returns NULL, and TO_DATE stops the query on a month 00 or a
    # 31 February: each part is checked first, the day against its month's length in a CASE of its own,
    # so MAKE_DATE only ever sees a real date.
    "postgres": {
        "yyyymmdd": "CASE WHEN CAST(__X__ AS BIGINT) BETWEEN 10000101 AND 99991231 "
                    "AND MOD(CAST(__X__ AS BIGINT) / 100, 100) BETWEEN 1 AND 12 "
                    "THEN CASE WHEN MOD(CAST(__X__ AS BIGINT), 100) BETWEEN 1 AND EXTRACT(DAY FROM "
                    "MAKE_DATE(CAST(CAST(__X__ AS BIGINT) / 10000 AS INT), CAST(MOD(CAST(__X__ AS BIGINT) / 100, 100) "
                    "AS INT), 1) + INTERVAL '1 MONTH' - INTERVAL '1 DAY') "
                    "THEN MAKE_DATE(CAST(CAST(__X__ AS BIGINT) / 10000 AS INT), CAST(MOD(CAST(__X__ AS BIGINT) / 100, "
                    "100) AS INT), CAST(MOD(CAST(__X__ AS BIGINT), 100) AS INT)) END END",
        "yyyymm": "CASE WHEN CAST(__X__ AS BIGINT) BETWEEN 100001 AND 999912 "
                  "AND MOD(CAST(__X__ AS BIGINT), 100) BETWEEN 1 AND 12 "
                  "THEN MAKE_DATE(CAST(CAST(__X__ AS BIGINT) / 100 AS INT), CAST(MOD(CAST(__X__ AS BIGINT), 100) AS INT), "
                  "1) END",
    },
}


def from_number(value: exp.Expression, shape: str, dialect: str) -> exp.Expression:
    """A yyyymmdd or yyyymm number as a DATE (the month's first day), NULL when it is no date.

    Placeholder keys (0, -1, 19000101, month 00) become NULL instead of failing the
    query, whatever order the warehouse evaluates filters and expressions in.
    """
    if shape not in ("yyyymmdd", "yyyymm"):
        raise ValueError(f"no date shape {shape!r}")
    if dialect in _FROM_NUMBER:
        return _fill(_FROM_NUMBER[dialect][shape], dialect, value)
    whole = exp.Cast(this=value.copy(), to=exp.DataType.build("BIGINT"))
    if dialect == "snowflake":
        return exp.Anonymous(this="TRY_TO_DATE", expressions=[
            exp.Anonymous(this="TO_VARCHAR", expressions=[whole]), exp.Literal.string(shape.upper())])
    if dialect == "duckdb":
        text = exp.Cast(this=whole, to=exp.DataType.build("VARCHAR"))
        fmt = "%Y%m%d" if shape == "yyyymmdd" else "%Y%m"
        return exp.Cast(this=exp.Anonymous(this="TRY_STRPTIME", expressions=[text, exp.Literal.string(fmt)]),
                        to=exp.DataType.build("DATE"))
    raise ValueError(f"no dialect {dialect!r}")


_ADD_MONTHS = {"snowflake": "DATEADD(MONTH, __N__, __X__)", "tsql": "DATEADD(MONTH, __N__, __X__)",
               "oracle": "ADD_MONTHS(__X__, __N__)", "postgres": "CAST(__X__ + __N__ * INTERVAL '1 MONTH' AS DATE)",
               "duckdb": "CAST(__X__ + TO_MONTHS(__N__) AS DATE)"}


def add_months(value: exp.Expression, months: exp.Expression | int, dialect: str) -> exp.Expression:
    """A DATE moved by a whole number of months (``months`` may be an expression)."""
    return _fill(_ADD_MONTHS[dialect], dialect, value, _number(months))


_ADD_DAYS = {"snowflake": "DATEADD(DAY, __N__, __X__)", "tsql": "DATEADD(DAY, __N__, __X__)",
             "oracle": "(__X__ + __N__)", "postgres": "CAST(__X__ + __N__ * INTERVAL '1 DAY' AS DATE)",
             "duckdb": "CAST(__X__ + TO_DAYS(__N__) AS DATE)"}


def add_days(value: exp.Expression, days: exp.Expression | int, dialect: str) -> exp.Expression:
    """A DATE moved by a whole number of days (``days`` may be an expression)."""
    return _fill(_ADD_DAYS[dialect], dialect, value, _number(days))


_DAYS_BETWEEN = {"snowflake": "DATEDIFF(DAY, __X__, __N__)", "tsql": "DATEDIFF(DAY, __X__, __N__)",
                 "oracle": "(TRUNC(__N__) - TRUNC(__X__))",
                 "postgres": "(CAST(__N__ AS DATE) - CAST(__X__ AS DATE))",
                 "duckdb": "DATE_DIFF('day', CAST(__X__ AS DATE), CAST(__N__ AS DATE))"}


def days_between(start: exp.Expression, end: exp.Expression, dialect: str) -> exp.Expression:
    """Whole calendar days from ``start`` to ``end`` (negative when ``end`` is earlier), dates or timestamps."""
    return _fill(_DAYS_BETWEEN[dialect], dialect, start, end)


def date_literal(day: dt.date, dialect: str) -> exp.Expression:
    text = day.isoformat()
    if dialect == "oracle":
        # Independent of NLS_DATE_FORMAT.
        return exp.Anonymous(this="TO_DATE", expressions=[exp.Literal.string(text), exp.Literal.string("YYYY-MM-DD")])
    return exp.Cast(this=exp.Literal.string(text), to=exp.DataType.build("DATE"))


def approx_distinct(value: exp.Expression, dialect: str) -> exp.Expression:
    if dialect == "postgres":   # no approximate count: the exact one
        return exp.Count(this=exp.Distinct(expressions=[value]))
    return exp.Anonymous(this="APPROX_COUNT_DISTINCT", expressions=[value])


def text_length(value: exp.Expression, dialect: str) -> exp.Expression:
    return exp.Anonymous(this="LEN" if dialect == "tsql" else "LENGTH", expressions=[value])


def as_text(value: exp.Expression, dialect: str) -> exp.Expression:
    """A value as text for profiling output (codes and names, never dates)."""
    if dialect == "oracle":
        return exp.Anonymous(this="TO_CHAR", expressions=[value])
    target = {"tsql": "NVARCHAR(400)", "snowflake": "VARCHAR", "postgres": "VARCHAR", "duckdb": "VARCHAR"}[dialect]
    return exp.Cast(this=value, to=exp.DataType.build(target, dialect=dialect))


def listed(value: exp.Expression, dialect: str) -> exp.Expression:
    """The values of a column among a group's rows, each once and in order, joined by commas ("Dish Soap, Paper
    Towels"): STRING_AGG or LISTAGG. T-SQL takes no DISTINCT there; its repeats, next to each other in that order,
    are taken out once the rows are read (core2/answer/builder.py). Oracle's list stops at its 4,000 bytes with
    "..." rather than fail the question (ORA-01489)."""
    if dialect == "tsql":
        inner: exp.Expression = exp.Cast(this=value.copy(), to=exp.DataType.build("NVARCHAR(MAX)", dialect=dialect))
        order: exp.Expression = value.copy()             # in the column's own order, never by a MAX text
    else:
        order = as_text(value, dialect)
        inner = exp.Distinct(expressions=[order.copy()])
    overflow = exp.OverflowTruncateBehavior(this=exp.Literal.string("..."), with_count=False) \
        if dialect == "oracle" else None
    return exp.GroupConcat(this=exp.Order(this=inner, expressions=[exp.Ordered(this=order, nulls_first=False)]),
                           separator=exp.Literal.string(", "), on_overflow=overflow)


def day_start(value: exp.Expression, dialect: str) -> exp.Expression:
    """Midnight of a timestamp's day, in the timestamp's own type (for time-of-day tests)."""
    text = {"snowflake": "DATE_TRUNC('DAY', __X__)", "tsql": "CAST(CAST(__X__ AS DATE) AS DATETIME2)",
            "oracle": "TRUNC(__X__)", "postgres": "DATE_TRUNC('DAY', __X__)", "duckdb": "DATE_TRUNC('DAY', __X__)"}[dialect]
    return _fill(text, dialect, value)


# Text that holds dates written year first ("2024-03-01", "2024-03-01 15:40:46", "2024-03-01T15:40:46"), as the
# database keeps them after a load from a file or another database. Read as a date, a value that is not one is no
# date (NULL), never an error that stops the question.
_TEXT_DATE = {
    "date": {"snowflake": "TRY_TO_DATE(LEFT(__X__, 10))", "tsql": "TRY_CONVERT(DATE, LEFT(__X__, 10))",
             "oracle": "CASE WHEN REGEXP_LIKE(__X__, '^[0-9]{4}-[0-9]{2}-[0-9]{2}') "
                       "THEN TO_DATE(SUBSTR(__X__, 1, 10), 'YYYY-MM-DD') END",
             "postgres": "CASE WHEN __X__ ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' THEN CAST(LEFT(__X__, 10) AS DATE) END",
             "duckdb": "TRY_CAST(LEFT(__X__, 10) AS DATE)"},
    "timestamp": {"snowflake": "TRY_TO_TIMESTAMP(__X__)", "tsql": "TRY_CONVERT(DATETIME2, __X__)",
                  "oracle": "CASE WHEN REGEXP_LIKE(__X__, '^[0-9]{4}-[0-9]{2}-[0-9]{2}') "
                            "THEN TO_TIMESTAMP(REPLACE(SUBSTR(__X__, 1, 19), 'T', ' '), 'YYYY-MM-DD HH24:MI:SS') END",
                  "postgres": "CASE WHEN __X__ ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' THEN CAST(__X__ AS TIMESTAMP) END",
                  "duckdb": "TRY_CAST(__X__ AS TIMESTAMP)"},
}


def text_date(value: exp.Expression, kind: str, dialect: str) -> exp.Expression:
    """A text column holding dates, read as a ``kind`` ("date" or "timestamp")."""
    return _fill(_TEXT_DATE["timestamp" if kind == "timestamp" else "date"][dialect], dialect, value)


def sample_clause(dialect: str, *, rows: int, total_rows: int) -> str:
    """Text placed after a table reference to profile a sample of about ``rows`` rows."""
    percent = max(0.01, min(100.0, 100.0 * rows / max(total_rows, 1)))
    if dialect == "snowflake":
        return f"SAMPLE ({rows} ROWS)"
    if dialect == "tsql":
        return f"TABLESAMPLE ({percent:.4f} PERCENT)"
    if dialect == "oracle":
        return f"SAMPLE ({percent:.4f})"
    if dialect == "postgres":
        # Whole pages, as Azure SQL's: reads a fraction of a large table rather than all of it.
        return f"TABLESAMPLE SYSTEM ({percent:.4f})"
    # DuckDB's sample of a table, written after the table. Its "USING SAMPLE" belongs at the end of the query,
    # after GROUP BY: after the table, a profile's "GROUP BY column" was a syntax error on every table of over a
    # million rows, and the common values of its columns were never read.
    return f"TABLESAMPLE reservoir({rows} ROWS)"


def first_rows(name: str, dialect: str, *, rows: int) -> str:
    """The first ``rows`` rows of a table or view, as a parenthesised query to alias: where a sample is refused.

    Azure SQL samples only tables (TABLESAMPLE on a view is an error), and a
    warehouse may refuse its sampling clause on an object for its own reasons;
    the first rows it returns are a rougher sample, but one every warehouse reads.
    """
    if dialect == "tsql":
        return f"(SELECT TOP ({rows}) * FROM {name})"
    if dialect == "oracle":
        return f"(SELECT * FROM {name} FETCH FIRST {rows} ROWS ONLY)"
    return f"(SELECT * FROM {name} LIMIT {rows})"


def aliased(source: str, alias: str, dialect: str) -> str:
    """A parenthesised query as a FROM source: Azure SQL requires the alias, Oracle refuses AS before it."""
    return f"{source} {alias}" if dialect == "oracle" else f"{source} AS {alias}"


_COLLATED = ("CHAR", "VARCHAR", "NCHAR", "NVARCHAR")


def same_text(a: exp.Expression, b: exp.Expression, dialect: str, raw_types: tuple[str, str] = ("", "")) -> exp.Expr:
    """a = b for two text columns, whatever collation each table was created with.

    Azure SQL refuses to compare text of two different collations (error 468,
    "Cannot resolve the collation conflict"); comparing both in the database's
    own collation always works. Only character columns have a collation: a
    uniqueidentifier refuses one (error 447).
    """
    bases = {raw.strip().upper().split("(")[0].strip() for raw in raw_types}
    if dialect == "tsql" and bases <= set(_COLLATED):
        a = exp.Collate(this=a, expression=exp.Var(this="DATABASE_DEFAULT"))
        b = exp.Collate(this=b, expression=exp.Var(this="DATABASE_DEFAULT"))
    return exp.EQ(this=a, expression=b)


_UNPADDED = {"snowflake": "LTRIM(TRIM(__X__), '0')", "duckdb": "LTRIM(TRIM(__X__), '0')",
             "oracle": "LTRIM(TRIM(__X__), '0')", "postgres": "LTRIM(TRIM(__X__), '0')",
             "tsql": "SUBSTRING(LTRIM(RTRIM(__X__)), PATINDEX('%[^0]%', LTRIM(RTRIM(__X__)) + '.'), 4000)"}


def unpadded(text: exp.Expression, dialect: str) -> exp.Expression:
    """A code kept as text read without its leading zeros ('0007' as '7'), as same_number compares it."""
    return _fill(_UNPADDED[dialect], dialect, text)


def same_number(text: exp.Expression, number: exp.Expression, dialect: str) -> exp.Expr:
    """A code kept as text ('0007') against a whole-number key (7).

    Both are compared as text without leading zeros: a code that is no number then
    matches nothing, where a cast would stop the whole query, and no warehouse has a
    safe cast that all of them read alike.
    """
    left, right = unpadded(text, dialect), unpadded(as_text(number, dialect), dialect)
    if dialect == "tsql":
        left = exp.Collate(this=left, expression=exp.Var(this="DATABASE_DEFAULT"))
        right = exp.Collate(this=right, expression=exp.Var(this="DATABASE_DEFAULT"))
    return exp.EQ(this=left, expression=right)


def render(expression: exp.Expression, dialect: str) -> str:
    """Print an expression for ``dialect`` and make sure the dialect can read it back."""
    sql = expression.sql(dialect=dialect)
    sqlglot.parse_one(sql, read=dialect)
    return sql
