"""
core/llm.py

LLM provider abstraction — Anthropic, OpenAI, Azure OpenAI.

v8 prompt changes:
  - build_kb_system_prompt:  DataPilot-style format (Overview / Key Metrics /
    Always Exclude / Columns / Query Patterns / Synonyms).
    NEEDS CONTEXT flag for ambiguous columns. Distinct values MUST be used.
    Generic — no domain-specific examples.
  - build_kb_query_prompt:   NEW — Stage 2 call that generates natural-language
    question → SQL pattern document from the actual KB content.
  - build_sql_system_prompt: MAX(date) rule for relative time queries.
    DDL detection moved here as a pre-check with a human-friendly message.
    All 3 DB types updated.
"""

import logging
from typing import Literal

from core.llm_audit import record_llm_call

log = logging.getLogger("querybot.llm")

Provider = Literal["anthropic", "openai", "azure_openai"]

# ── SQL syntax rules per DB type ──────────────────────────────────────────────

_SQL_SYNTAX: dict[str, str] = {
    "snowflake": (
        "- Row limit: If the user states a number (top 10, show 5), use that number in LIMIT. "
        "If no number is stated, default to LIMIT 20.\n"
        "- Date functions: DATE_TRUNC, DATEADD, CURRENT_DATE, DATEDIFF\n"
        "- Conditionals: IFF(), COALESCE()\n"
        "- Schema-qualify tables if needed: DATABASE.SCHEMA.TABLE\n"
        "- Split name concat: FIRST_NAME || ' ' || LAST_NAME AS FULL_NAME\n"
        "- CRITICAL TIME RULE: When the question uses relative time (last month, last week, "
        "this year, yesterday, recent), NEVER use CURRENT_DATE as the reference point. "
        "The database may be historical. Always anchor to the latest date in the data:\n"
        "  Last month:  WHERE DateCol >= DATEADD('month', -1, (SELECT MAX(DateCol) FROM TableName))\n"
        "  Last week:   WHERE DateCol >= DATEADD('week',  -1, (SELECT MAX(DateCol) FROM TableName))\n"
        "  This year:   WHERE YEAR(DateCol) = YEAR((SELECT MAX(DateCol) FROM TableName))\n"
        "- DATE BUCKETING (group by time period):\n"
        "  By month:   SELECT DATE_TRUNC('month', date_col) AS PERIOD, SUM(metric) AS TOTAL"
        " FROM tbl GROUP BY 1 ORDER BY 1\n"
        "  By quarter: SELECT DATE_TRUNC('quarter', date_col) AS PERIOD, SUM(metric) AS TOTAL"
        " FROM tbl GROUP BY 1 ORDER BY 1\n"
        "  By week:    SELECT DATE_TRUNC('week', date_col) AS PERIOD, SUM(metric) AS TOTAL"
        " FROM tbl GROUP BY 1 ORDER BY 1\n"
        "  By year:    SELECT DATE_TRUNC('year', date_col) AS PERIOD, SUM(metric) AS TOTAL"
        " FROM tbl GROUP BY 1 ORDER BY 1\n"
        "- NAMED PERIOD FILTERS:\n"
        "  Q1: WHERE QUARTER(date_col)=1  |  Q2: QUARTER=2  |  Q3: QUARTER=3  |  Q4: QUARTER=4\n"
        "  H1: WHERE MONTH(date_col) BETWEEN 1 AND 6\n"
        "  H2: WHERE MONTH(date_col) BETWEEN 7 AND 12\n"
        "  Last N months: WHERE date_col >= DATEADD('month',-N,(SELECT MAX(date_col) FROM tbl))\n"
        "  Last N weeks:  WHERE date_col >= DATEADD('week',-N,(SELECT MAX(date_col) FROM tbl))\n"
        "  Specific month by name: Jan=1, Feb=2, Mar=3, Apr=4, May=5, Jun=6,"
        " Jul=7, Aug=8, Sep=9, Oct=10, Nov=11, Dec=12 → WHERE MONTH(date_col)=N\n"
        "- PERCENTILE/MEDIAN:\n"
        "  Median:        MEDIAN(metric) OVER () AS MEDIAN_VAL  (or within GROUP BY context)\n"
        "  Nth percentile: PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY metric) AS P90\n"
    ),
    "oracle": (
        "- Row limit: If the user states a number (top 10, show 5), use FETCH FIRST N ROWS ONLY "
        "with that number. Default to FETCH FIRST 20 ROWS ONLY. NEVER use LIMIT.\n"
        "- Date functions: TRUNC(date,'MM'), SYSDATE, ADD_MONTHS, MONTHS_BETWEEN\n"
        "- Null handling: NVL() or COALESCE()\n"
        "- Schema-qualify tables: OWNER.TABLE_NAME\n"
        "- Split name concat: FIRST_NAME || ' ' || LAST_NAME AS FULL_NAME\n"
        "- CRITICAL TIME RULE: When the question uses relative time (last month, last week, "
        "this year, yesterday, recent), NEVER use SYSDATE as the reference point. "
        "The database may be historical. Always anchor to the latest date in the data:\n"
        "  Last month:  WHERE DateCol >= ADD_MONTHS((SELECT MAX(DateCol) FROM TableName), -1)\n"
        "  Last week:   WHERE DateCol >= (SELECT MAX(DateCol) FROM TableName) - 7\n"
        "  This year:   WHERE TRUNC(DateCol,'YYYY') = TRUNC((SELECT MAX(DateCol) FROM TableName),'YYYY')\n"
        "- DATE BUCKETING (group by time period):\n"
        "  By month:   SELECT TRUNC(date_col,'MM') AS PERIOD, SUM(metric) AS TOTAL"
        " FROM tbl GROUP BY TRUNC(date_col,'MM') ORDER BY 1\n"
        "  By quarter: SELECT TRUNC(date_col,'Q') AS PERIOD, SUM(metric) AS TOTAL"
        " FROM tbl GROUP BY TRUNC(date_col,'Q') ORDER BY 1\n"
        "  By week:    SELECT TRUNC(date_col,'IW') AS PERIOD, SUM(metric) AS TOTAL"
        " FROM tbl GROUP BY TRUNC(date_col,'IW') ORDER BY 1\n"
        "  By year:    SELECT TRUNC(date_col,'YYYY') AS PERIOD, SUM(metric) AS TOTAL"
        " FROM tbl GROUP BY TRUNC(date_col,'YYYY') ORDER BY 1\n"
        "- NAMED PERIOD FILTERS:\n"
        "  Q1: WHERE EXTRACT(MONTH FROM date_col) BETWEEN 1 AND 3\n"
        "  Q2: WHERE EXTRACT(MONTH FROM date_col) BETWEEN 4 AND 6\n"
        "  Q3: WHERE EXTRACT(MONTH FROM date_col) BETWEEN 7 AND 9\n"
        "  Q4: WHERE EXTRACT(MONTH FROM date_col) BETWEEN 10 AND 12\n"
        "  H1: WHERE EXTRACT(MONTH FROM date_col) BETWEEN 1 AND 6\n"
        "  H2: WHERE EXTRACT(MONTH FROM date_col) BETWEEN 7 AND 12\n"
        "  Last N months: WHERE date_col >= ADD_MONTHS((SELECT MAX(date_col) FROM tbl),-N)\n"
        "  Specific month: WHERE EXTRACT(MONTH FROM date_col)=N\n"
        "- PERCENTILE/MEDIAN:\n"
        "  Median:         MEDIAN(metric) AS MEDIAN_VAL\n"
        "  Nth percentile: PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY metric) AS P90\n"
        "  Never use AVG() as a substitute for MEDIAN.\n"
    ),
    "azure_sql": (
        "- Row limit: If the user states a number (top 10, show 5), use SELECT TOP N with that "
        "number. Default to SELECT TOP 20. TOP goes immediately after SELECT. NEVER use LIMIT.\n"
        "- Date functions: GETDATE(), DATEADD, DATEDIFF, FORMAT, CONVERT, YEAR(), MONTH(), "
        "DATEPART(QUARTER,...), DATEPART(WEEK,...), DATEPART(ISO_WEEK,...)\n"
        "- TABLE NAMING RULE (CRITICAL): Azure SQL Database only supports TWO-part table names. "
        "Always write tables as [SCHEMA].[TABLE_NAME]. "
        "The Knowledge Base shows 'SQL table name: [SCHEMA].[TABLE]' for each table — "
        "use that exact two-part format. "
        "NEVER use three-part names like [DATABASE].[SCHEMA].[TABLE] — Azure SQL rejects them with error 40515.\n"
        "- BIT COLUMN RULE: SQL Server has no TRUE/FALSE literals. For BIT or boolean columns "
        "ALWAYS use 1 (true / yes / active / included) or 0 (false / no / inactive / excluded) "
        "in WHERE clauses. NEVER write True, False, 'True', or 'False' — SQL Server treats "
        "those as column names and raises error 207 (invalid column name).\n"
        "- Null handling: ISNULL() or COALESCE()\n"
        "- Split name concat: CONCAT(FIRST_NAME, ' ', LAST_NAME) AS FULL_NAME\n"
        "- CRITICAL TIME RULE: When the question uses relative time (last month, last week, "
        "this year, yesterday, recent), NEVER use GETDATE() as the reference point. "
        "The database may be historical. Always anchor to the latest date in the data:\n"
        "  Last month:  WHERE DateCol >= DATEADD(month,-1,(SELECT MAX(DateCol) FROM [schema].[TableName]))\n"
        "  Last week:   WHERE DateCol >= DATEADD(week,-1,(SELECT MAX(DateCol) FROM [schema].[TableName]))\n"
        "  This year:   WHERE YEAR(DateCol) = YEAR((SELECT MAX(DateCol) FROM [schema].[TableName]))\n"
        "- DATE BUCKETING (group by time period) — ORDER BY the raw expression, not the alias:\n"
        "  By month:   SELECT FORMAT(date_col,'yyyy-MM') AS PERIOD, SUM(metric) AS TOTAL"
        " FROM [sch].[tbl] GROUP BY FORMAT(date_col,'yyyy-MM') ORDER BY FORMAT(date_col,'yyyy-MM')\n"
        "  By quarter: SELECT CONCAT(YEAR(date_col),'-Q',DATEPART(QUARTER,date_col)) AS PERIOD,"
        " SUM(metric) AS TOTAL FROM [sch].[tbl]"
        " GROUP BY YEAR(date_col),DATEPART(QUARTER,date_col)"
        " ORDER BY YEAR(date_col),DATEPART(QUARTER,date_col)\n"
        "  By week:    SELECT CONCAT(YEAR(date_col),'-W',DATEPART(ISO_WEEK,date_col)) AS PERIOD,"
        " SUM(metric) AS TOTAL FROM [sch].[tbl]"
        " GROUP BY YEAR(date_col),DATEPART(ISO_WEEK,date_col)"
        " ORDER BY YEAR(date_col),DATEPART(ISO_WEEK,date_col)\n"
        "  By year:    SELECT YEAR(date_col) AS PERIOD, SUM(metric) AS TOTAL"
        " FROM [sch].[tbl] GROUP BY YEAR(date_col) ORDER BY YEAR(date_col)\n"
        "- NAMED PERIOD FILTERS:\n"
        "  Q1: WHERE DATEPART(QUARTER,date_col)=1  |  Q2/Q3/Q4: DATEPART(QUARTER,date_col)=N\n"
        "  H1: WHERE DATEPART(MONTH,date_col) BETWEEN 1 AND 6\n"
        "  H2: WHERE DATEPART(MONTH,date_col) BETWEEN 7 AND 12\n"
        "  Last N months: WHERE date_col >= DATEADD(month,-N,(SELECT MAX(date_col) FROM [sch].[tbl]))\n"
        "  Last N weeks:  WHERE date_col >= DATEADD(week,-N,(SELECT MAX(date_col) FROM [sch].[tbl]))\n"
        "  Specific month by name: Jan=1,Feb=2,Mar=3,Apr=4,May=5,Jun=6,"
        "Jul=7,Aug=8,Sep=9,Oct=10,Nov=11,Dec=12 → WHERE MONTH(date_col)=N\n"
        "- PERCENTILE/MEDIAN: Azure SQL has NO MEDIAN() function — never use it.\n"
        "  Median:         SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY metric) OVER () AS MEDIAN_VAL\n"
        "  Nth percentile: SELECT PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY metric) OVER () AS P90\n"
        "  Never use AVG() as a substitute for median.\n"
    ),
}

_DB_LABELS: dict[str, str] = {
    "snowflake": "Snowflake",
    "oracle":    "Oracle",
    "azure_sql": "Azure SQL",
}

# Operations that must never be executed — user gets a friendly message
_DDL_KEYWORDS = {
    "CREATE", "DROP", "ALTER", "TRUNCATE", "INSERT", "UPDATE", "DELETE",
    "MERGE", "GRANT", "REVOKE", "EXEC", "EXECUTE", "CALL",
    "BULK", "COPY", "PUT", "GET", "UNLOAD", "LOAD",
}


def is_ddl_attempt(text: str) -> bool:
    """Return True if the user's raw message looks like a DDL/DML attempt."""
    import re
    first_word = re.split(r"\s+", text.strip().upper())[0] if text.strip() else ""
    return first_word in _DDL_KEYWORDS


_DDL_USER_MESSAGE = (
    "🔒 *That operation is not permitted.*\n\n"
    "QueryBot is a read-only analytics assistant — it can only run SELECT queries "
    "to retrieve and analyse data.\n\n"
    "Operations that modify data (CREATE, DROP, INSERT, UPDATE, DELETE, etc.) "
    "are blocked for security. If you need to make database changes, please use "
    "your database administration tool directly."
)


# ══════════════════════════════════════════════════════════════════════════════
# Prompt builders
# ══════════════════════════════════════════════════════════════════════════════

def build_sql_system_prompt(
    db_type: str,
    table_context: str,
    conversation_history: list | None = None,
    graph_context: dict | None = None,
    semantic_plan: dict | None = None,
) -> str:
    """System prompt for SQL generation — used on every user query.

    graph_context: dict from graph_resolver.resolve_for_question().
    conversation_history: list of {question, sql, columns, row_count} dicts.
    Injected as session context to resolve follow-up references.
    """
    label  = _DB_LABELS.get(db_type, db_type)
    syntax = _SQL_SYNTAX.get(db_type, "- Use standard ANSI SQL\n")

    # Dialect-correct pattern for the CROSS-TABLE QUERY RULE example.
    # This prevents the generic LIMIT 20 example from overriding the per-dialect
    # "NEVER use LIMIT" rule for Azure SQL / Oracle.
    if db_type == "azure_sql":
        _xjoin_pattern = (
            "Pattern: SELECT TOP 20 d.NAME_COL, SUM(f.METRIC_COL) FROM FACT f "
            "JOIN DIM d ON f.FK = d.PK GROUP BY d.NAME_COL ORDER BY 2 DESC"
        )
        _mom_pattern = (
            "WITH monthly AS (\n"
            "  SELECT FORMAT(TRY_CONVERT(date, CONVERT(varchar(8), int_yyyymmdd_key), 112),'yyyy-MM') AS PERIOD, <approved_formula> AS METRIC\n"
            "  FROM [schema].[table]\n"
            "  WHERE int_yyyymmdd_key > 0\n"
            "  GROUP BY FORMAT(TRY_CONVERT(date, CONVERT(varchar(8), int_yyyymmdd_key), 112),'yyyy-MM')\n"
            ")\n"
            "SELECT PERIOD, METRIC,\n"
            "       LAG(METRIC) OVER (ORDER BY PERIOD) AS PREV_METRIC,\n"
            "       METRIC - LAG(METRIC) OVER (ORDER BY PERIOD) AS DIFF,\n"
            "       ROUND((METRIC - LAG(METRIC) OVER (ORDER BY PERIOD))*100.0\n"
            "             /NULLIF(LAG(METRIC) OVER (ORDER BY PERIOD),0),2) AS PCT_CHANGE\n"
            "FROM monthly ORDER BY PERIOD"
        )
        _top_n_per_group_pattern = (
            "WITH ranked AS (\n"
            "  SELECT *, ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY metric_col DESC) AS rn\n"
            "  FROM [schema].[table]\n"
            ")\n"
            "SELECT * FROM ranked WHERE rn <= N"
        )
        _pct_total_pattern = (
            "SELECT group_col,\n"
            "       SUM(metric) AS TOTAL,\n"
            "       ROUND(SUM(metric)*100.0/SUM(SUM(metric)) OVER (),2) AS PCT_OF_TOTAL\n"
            "FROM [schema].[table]\n"
            "GROUP BY group_col ORDER BY TOTAL DESC"
        )
        _antijoin_pattern = (
            "SELECT a.key_col, a.name_col\n"
            "FROM [schema].[parent] a\n"
            "LEFT JOIN [schema].[child] b ON a.key_col = b.fk_col\n"
            "    AND <optional time filter>\n"
            "WHERE b.fk_col IS NULL"
        )
        _running_total_pattern = (
            "SELECT date_col,\n"
            "       SUM(metric) AS PERIOD_TOTAL,\n"
            "       SUM(SUM(metric)) OVER (ORDER BY date_col ROWS UNBOUNDED PRECEDING) AS RUNNING_TOTAL\n"
            "FROM [schema].[table]\n"
            "GROUP BY date_col ORDER BY date_col"
        )
        _moving_avg_pattern = (
            "SELECT date_col, metric,\n"
            "       AVG(CAST(metric AS FLOAT)) OVER\n"
            "         (ORDER BY date_col ROWS BETWEEN N-1 PRECEDING AND CURRENT ROW) AS ROLLING_AVG\n"
            "FROM [schema].[table] ORDER BY date_col"
        )
    elif db_type == "oracle":
        _xjoin_pattern = (
            "Pattern: SELECT d.NAME_COL, SUM(f.METRIC_COL) FROM FACT f "
            "JOIN DIM d ON f.FK = d.PK GROUP BY d.NAME_COL ORDER BY 2 DESC "
            "FETCH FIRST 20 ROWS ONLY"
        )
        _mom_pattern = (
            "WITH monthly AS (\n"
            "  SELECT TRUNC(date_col,'MM') AS PERIOD, <approved_formula> AS METRIC\n"
            "  FROM schema.table GROUP BY TRUNC(date_col,'MM')\n"
            ")\n"
            "SELECT PERIOD, METRIC,\n"
            "       LAG(METRIC) OVER (ORDER BY PERIOD) AS PREV_METRIC,\n"
            "       METRIC - LAG(METRIC) OVER (ORDER BY PERIOD) AS DIFF,\n"
            "       ROUND((METRIC - LAG(METRIC) OVER (ORDER BY PERIOD))*100\n"
            "             /NULLIF(LAG(METRIC) OVER (ORDER BY PERIOD),0),2) AS PCT_CHANGE\n"
            "FROM monthly ORDER BY PERIOD"
        )
        _top_n_per_group_pattern = (
            "WITH ranked AS (\n"
            "  SELECT t.*, ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY metric_col DESC) AS rn\n"
            "  FROM schema.table t\n"
            ")\n"
            "SELECT * FROM ranked WHERE rn <= N"
        )
        _pct_total_pattern = (
            "SELECT group_col,\n"
            "       SUM(metric) AS TOTAL,\n"
            "       ROUND(SUM(metric)*100/SUM(SUM(metric)) OVER (),2) AS PCT_OF_TOTAL\n"
            "FROM schema.table\n"
            "GROUP BY group_col ORDER BY TOTAL DESC"
        )
        _antijoin_pattern = (
            "SELECT a.key_col, a.name_col\n"
            "FROM schema.parent a\n"
            "LEFT JOIN schema.child b ON a.key_col = b.fk_col\n"
            "WHERE b.fk_col IS NULL"
        )
        _running_total_pattern = (
            "SELECT date_col,\n"
            "       SUM(metric) AS PERIOD_TOTAL,\n"
            "       SUM(SUM(metric)) OVER (ORDER BY date_col ROWS UNBOUNDED PRECEDING) AS RUNNING_TOTAL\n"
            "FROM schema.table\n"
            "GROUP BY date_col ORDER BY date_col"
        )
        _moving_avg_pattern = (
            "SELECT date_col, metric,\n"
            "       AVG(metric) OVER\n"
            "         (ORDER BY date_col ROWS BETWEEN N-1 PRECEDING AND CURRENT ROW) AS ROLLING_AVG\n"
            "FROM schema.table ORDER BY date_col"
        )
    else:  # snowflake and ANSI
        _xjoin_pattern = (
            "Pattern: SELECT d.NAME_COL, SUM(f.METRIC_COL) FROM FACT f "
            "JOIN DIM d ON f.FK = d.PK GROUP BY d.NAME_COL ORDER BY 2 DESC LIMIT 20"
        )
        _mom_pattern = (
            "WITH monthly AS (\n"
            "  SELECT DATE_TRUNC('month', date_col) AS PERIOD, <approved_formula> AS METRIC\n"
            "  FROM db.schema.table GROUP BY 1\n"
            ")\n"
            "SELECT PERIOD, METRIC,\n"
            "       LAG(METRIC) OVER (ORDER BY PERIOD) AS PREV_METRIC,\n"
            "       METRIC - LAG(METRIC) OVER (ORDER BY PERIOD) AS DIFF,\n"
            "       ROUND((METRIC - LAG(METRIC) OVER (ORDER BY PERIOD))*100.0\n"
            "             /NULLIF(LAG(METRIC) OVER (ORDER BY PERIOD),0),2) AS PCT_CHANGE\n"
            "FROM monthly ORDER BY PERIOD"
        )
        _top_n_per_group_pattern = (
            "WITH ranked AS (\n"
            "  SELECT *, ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY metric_col DESC) AS rn\n"
            "  FROM db.schema.table\n"
            ")\n"
            "SELECT * FROM ranked WHERE rn <= N LIMIT 200"
        )
        _pct_total_pattern = (
            "SELECT group_col,\n"
            "       SUM(metric) AS TOTAL,\n"
            "       ROUND(SUM(metric)*100.0/SUM(SUM(metric)) OVER (),2) AS PCT_OF_TOTAL\n"
            "FROM db.schema.table\n"
            "GROUP BY group_col ORDER BY TOTAL DESC LIMIT 20"
        )
        _antijoin_pattern = (
            "SELECT a.key_col, a.name_col\n"
            "FROM db.schema.parent a\n"
            "LEFT JOIN db.schema.child b ON a.key_col = b.fk_col\n"
            "WHERE b.fk_col IS NULL\n"
            "LIMIT 20"
        )
        _running_total_pattern = (
            "SELECT date_col,\n"
            "       SUM(metric) AS PERIOD_TOTAL,\n"
            "       SUM(SUM(metric)) OVER (ORDER BY date_col ROWS UNBOUNDED PRECEDING) AS RUNNING_TOTAL\n"
            "FROM db.schema.table\n"
            "GROUP BY date_col ORDER BY date_col"
        )
        _moving_avg_pattern = (
            "SELECT date_col, metric,\n"
            "       AVG(metric) OVER\n"
            "         (ORDER BY date_col ROWS BETWEEN N-1 PRECEDING AND CURRENT ROW) AS ROLLING_AVG\n"
            "FROM db.schema.table ORDER BY date_col LIMIT 200"
        )

    base = (
        f"You are a {label} SQL expert. "
        "Convert the user's plain-English question into a valid SQL SELECT query.\n\n"
        "STRICT RULES:\n"
        "- Use ONLY the tables and columns described in the Knowledge Base below. "
        "Never invent, assume, or guess column names.\n"
        "- COLUMN NAME HALLUCINATION IS FORBIDDEN: If you need a column for a business "
        "concept (e.g. 'total revenue', 'charge amount'), look it up in the "
        "'COLUMN SYNONYM MAP', 'BUSINESS TERM DEFINITIONS', 'Business Synonyms', "
        "or 'Key Metrics' sections. If not found there, check the 'Session context' "
        "for columns returned in previous turns. Never invent CamelCase or "
        "concatenated variants (e.g. TotalRevenueUSD, ChargeAmount) that are not "
        "explicitly listed in the Knowledge Base.\n"
        "- If a column name is flagged [NEEDS CONTEXT] in the KB, do not use it — "
        "reply with CANNOT_GENERATE instead.\n"
        f"{syntax}"
        "- Return ONLY the raw SQL query. No markdown fences, no explanation, no comments.\n"
        "- If the question cannot be answered from the available tables and columns, "
        "reply with exactly: CANNOT_GENERATE\n"
        "- Never generate CREATE, DROP, ALTER, INSERT, UPDATE, DELETE, TRUNCATE, "
        "MERGE, GRANT, REVOKE or any data-modifying statement.\n"
        "- NAME CONCATENATION RULE: When a user asks for a person's name and the "
        "table has separate first/last name columns (FIRST_NAME/LAST_NAME, FNAME/LNAME, "
        "GIVEN_NAME/SURNAME, FORENAME/FAMILY_NAME or similar) but no combined column, "
        "always concatenate them using the dialect syntax shown above. Never return "
        "split name columns separately when the user asked for a name.\n"
        "- CROSS-TABLE QUERY RULE: When a question asks for a metric (count, total, sum, "
        "average, amount) BY or PER a dimension (name, category, region, type, department) "
        "you MUST write a JOIN. Metric columns live in FACT tables. Grouping columns live "
        "in DIMENSION tables. Use the Join Keys from the Knowledge Base. "
        f"{_xjoin_pattern}\n"
        "- ORDER BY ALIAS RULE: When you define a column alias in SELECT (e.g. "
        "SELECT SUM(col) AS TOTAL_COST), you MUST use that EXACT alias in ORDER BY "
        "(ORDER BY TOTAL_COST DESC). Never add, remove, or change underscores, spaces, "
        "or any characters in the alias. If you ORDER BY a name that was not defined "
        "in the SELECT clause the query will fail at runtime.\n"
        "- CORRELATION / SCATTER RULE: When the user asks whether two numeric metrics "
        "are correlated, asks 'are X and Y related', or asks to 'show X vs Y', do NOT "
        "attempt to compute a Pearson / statistical correlation coefficient — SQL has no "
        "built-in CORR() on all platforms. Instead generate a SELECT that returns BOTH "
        "numeric columns (and optionally a label column) so the result can be visualised "
        "as a scatter chart. Example: SELECT label_col, numeric_col_1, numeric_col_2 "
        "FROM table ORDER BY numeric_col_1 DESC\n"
        "- APPROVED METRIC FORMULA RULE: If the context includes 'APPROVED METRIC FORMULAS' "
        "and the user asks for that metric or any synonym, the approved calculation MUST be "
        "used in EVERY SELECT expression — including inside CTEs, subqueries, and comparison "
        "queries. The approved formula OVERRIDES any column name found in the KB schema docs "
        "for that metric. For by/per/grouped-by questions, put the approved formula in the "
        "SELECT list and group by the requested dimension. For percentage/rate formulas, do "
        "not average row-level values unless the formula explicitly uses AVG(). "
        "Never substitute a 'similar-sounding' column from the KB for an approved formula.\n"
        + (
            "- AZURE SQL DATE-KEY RULE: Columns ending in _DT_DMS_KEY or _DATE_DMS_KEY are "
            "integer YYYYMMDD keys, not real date columns. Do NOT call FORMAT(), YEAR(), MONTH(), "
            "DATEPART(), or LAG ordering directly on the integer key. First convert with "
            "TRY_CONVERT(date, CONVERT(varchar(8), alias.DATE_KEY_COL), 112), and filter out "
            "invalid zero keys with alias.DATE_KEY_COL > 0. For month buckets use "
            "FORMAT(TRY_CONVERT(date, CONVERT(varchar(8), alias.DATE_KEY_COL), 112), 'yyyy-MM').\n"
            if ("_DT_DMS_KEY" in table_context.upper() or "_DATE_DMS_KEY" in table_context.upper())
            else ""
        )
        + "- YEAR-OVER-YEAR / PERIOD COMPARISON RULE: When the user asks to compare a metric "
        "'last year vs year before', 'prior year', 'year over year', 'how did X change', "
        "'compared to last year', or 'vs previous year':\n"
        "  1. Always CAST the year/period column to INT: CAST(year_col AS INT) AS YR\n"
        "  2. Build two CTEs — one per year — using the approved metric formula for all "
        "aggregations. Use the 2 most recent years in the data:\n"
        "     curr year = (SELECT MAX(CAST(year_col AS INT)) FROM table)\n"
        "     prev year = curr year - 1\n"
        "  3. LEFT JOIN curr to prev so the result always shows current year even when "
        "prior-year data is absent.\n"
        "  4. Always include: difference column (curr_metric - prev_metric) and "
        "pct_change column ROUND((curr - prev) * 100.0 / NULLIF(prev, 0), 2).\n"
        "  5. Use NULLIF(prev, 0) on every division to guard against divide-by-zero.\n"
        "  Azure SQL pattern:\n"
        "    WITH base AS (\n"
        "      SELECT CAST(yr_col AS INT) AS YR,\n"
        "             <approved_formula> AS METRIC\n"
        "      FROM [schema].[table]\n"
        "      GROUP BY CAST(yr_col AS INT)\n"
        "    )\n"
        "    SELECT c.YR AS CURRENT_YEAR, c.METRIC AS CURRENT_METRIC,\n"
        "           p.YR AS PREV_YEAR,    p.METRIC AS PREV_METRIC,\n"
        "           c.METRIC - p.METRIC   AS DIFFERENCE,\n"
        "           ROUND((c.METRIC - p.METRIC)*100.0/NULLIF(p.METRIC,0),2) AS PCT_CHANGE\n"
        "    FROM base c\n"
        "    LEFT JOIN base p ON p.YR = c.YR - 1\n"
        "    WHERE c.YR = (SELECT MAX(YR) FROM base);\n"
        "  Snowflake / Oracle: same pattern with dialect date functions.\n"
        "  NEVER use MAX(col)-1 in a WHERE clause for year anchoring — use the CTE "
        "approach above so the anchor is derived once and reused cleanly.\n"
        "- DISTINCT ENTITY RULE: When the question asks to LIST, SHOW, FIND, GET, "
        "or WHO — referring to individual entities (prescribers, patients, doctors, "
        "customers, products, drugs, items, employees) rather than aggregating metrics — "
        "ALWAYS use SELECT DISTINCT on the entity name/identifier column. "
        "Joining a dimension table (e.g. DIM_PRESCRIBER) to a fact table (e.g. FACT_RXFILL) "
        "without DISTINCT returns one row per transaction row, not one row per entity. "
        "This causes duplicate names in the result. "
        "Rule: if there is no GROUP BY and no aggregate function (SUM/COUNT/AVG/MIN/MAX) "
        "in the SELECT clause, write SELECT DISTINCT.\n"
        "  Examples requiring DISTINCT:\n"
        "    'list prescribers who have not prescribed...' → SELECT DISTINCT p.NAME ...\n"
        "    'show patients that have...'                  → SELECT DISTINCT p.NAME ...\n"
        "    'which drugs appear in...'                    → SELECT DISTINCT d.DRUG_NAME ...\n"
        "    'who prescribed the top 5 formulas'           → SELECT DISTINCT pr.FULL_NAME ...\n"
        "    'find customers without any orders'           → SELECT DISTINCT c.CUSTOMER_NAME ...\n\n"
        "- HAVING RULE: When the user asks for groups/categories that meet a threshold "
        "(e.g. 'departments with more than 10 employees', 'products with total sales over 5000', "
        "'months with more than 100 transactions'), ALWAYS use HAVING to filter on the aggregate — "
        "NEVER use WHERE to filter on an aggregate expression. WHERE filters individual rows before "
        "grouping; HAVING filters the grouped result.\n"
        "  Correct:   SELECT dept, COUNT(*) AS CNT FROM tbl GROUP BY dept HAVING COUNT(*) > 10\n"
        "  Incorrect: SELECT dept, COUNT(*) AS CNT FROM tbl WHERE COUNT(*) > 10 GROUP BY dept\n"
        "  With alias: GROUP BY dept HAVING COUNT(*) > 10  (never HAVING CNT > 10 unless the DB "
        "explicitly supports alias in HAVING — assume it does not)\n\n"
        "- TOP-N PER GROUP RULE: When the user asks for 'top N per category', 'best X in each Y', "
        "'highest Z for every W', use a window function CTE with ROW_NUMBER() PARTITION BY:\n"
        f"{_top_n_per_group_pattern}\n"
        "  Replace group_col with the grouping dimension, metric_col with the ranking metric, "
        "and N with the requested count. Use RANK() instead of ROW_NUMBER() only when the user "
        "explicitly says 'ties should count equally'.\n\n"
        "- PERCENTAGE OF TOTAL RULE: When the user asks for share, contribution, proportion, "
        "'what percent of total', '% breakdown', or 'how much does X contribute', use a window "
        "SUM() OVER () to calculate the grand total inline — do NOT use a subquery or CTE just "
        "for the denominator:\n"
        f"{_pct_total_pattern}\n"
        "  Always ROUND to 2 decimal places. Always guard against divide-by-zero with "
        "NULLIF(SUM(SUM(metric)) OVER (), 0) if data may be empty.\n\n"
        "- ANTI-JOIN RULE: When the user asks for records that have NO matching rows in another "
        "table ('employees with no absences', 'customers without orders', 'products never sold', "
        "'items missing from', 'not in'), ALWAYS use a LEFT JOIN … WHERE right.key IS NULL "
        "pattern — do NOT use NOT IN (which fails with NULLs) or NOT EXISTS (subquery):\n"
        f"{_antijoin_pattern}\n"
        "  The optional time filter on the JOIN (not in WHERE) ensures the anti-join is scoped "
        "to the correct period without excluding parent rows that have records in other periods. "
        "The FROM table must be the source/parent table containing the records to list; the "
        "missing-side table must be on the RIGHT side of the LEFT JOIN. Never answer a "
        "missing-data question by querying only the missing-side table with WHERE measure IS NULL.\n\n"
        "- FACT-TO-FACT JOIN RULE: When a question combines measures from multiple fact tables "
        "(for example on-hand inventory, purchase receipts, and replacement cost), aggregate each "
        "fact table in its own CTE to the shared join grain first, then join those CTEs. Do NOT "
        "join raw fact rows and then SUM measures unless the Knowledge Base proves the join is "
        "one-to-one. This prevents duplicated totals from many-to-many joins.\n\n"
        "- CONDITIONAL AGGREGATION RULE: When the user asks to 'split by status', 'show count "
        "for each type side by side', 'pivot by category', or compares two groups in the same row "
        "(e.g. 'active vs inactive count', 'male vs female headcount'), use SUM(CASE WHEN) "
        "conditional aggregation in a single query — do NOT use multiple subqueries or UNIONs:\n"
        "  SELECT\n"
        "    SUM(CASE WHEN status_col = 'Active'   THEN 1 ELSE 0 END) AS ACTIVE_COUNT,\n"
        "    SUM(CASE WHEN status_col = 'Inactive' THEN 1 ELSE 0 END) AS INACTIVE_COUNT,\n"
        "    COUNT(*) AS TOTAL\n"
        "  FROM [schema].[table]\n"
        "  Use the exact distinct values from the KB for CASE WHEN conditions.\n\n"
        "- MONTH-OVER-MONTH / QUARTER-OVER-QUARTER RULE: When the user asks for MoM trend, "
        "'how did X change each month', 'monthly growth', 'QoQ comparison', or 'quarter over "
        "quarter', use a CTE with DATE_TRUNC/FORMAT for bucketing combined with LAG() OVER "
        "(ORDER BY PERIOD) to compute the prior-period value:\n"
        f"{_mom_pattern}\n"
        "  For quarterly: replace the month bucket function with the quarter bucket from the "
        "DATE BUCKETING rules above. Always include: period value, prior period value, "
        "absolute difference, and PCT_CHANGE rounded to 2 decimal places.\n\n"
        "- RUNNING TOTAL RULE: When the user asks for 'cumulative', 'running total', "
        "'year-to-date total', 'cumulative sum', or 'total so far', use a nested window aggregate "
        "SUM(SUM(metric)) OVER (ORDER BY date_col ROWS UNBOUNDED PRECEDING):\n"
        f"{_running_total_pattern}\n"
        "  The inner SUM() aggregates the group; the outer SUM() OVER () accumulates across groups. "
        "Always ORDER BY the period column both inside and outside the window.\n\n"
        "- MOVING AVERAGE RULE: When the user asks for a rolling average, smoothed trend, "
        "'N-period moving average', or 'trailing average', use AVG() OVER with a ROWS BETWEEN "
        "N-1 PRECEDING AND CURRENT ROW window:\n"
        f"{_moving_avg_pattern}\n"
        "  Replace N with the window size stated by the user (default 3 if unspecified). "
        "Cast integer metrics to FLOAT/DECIMAL to avoid integer division truncation "
        "(Azure SQL: CAST(metric AS FLOAT); Snowflake/Oracle: metric directly as AVG handles it).\n\n"
        "- NULL-SAFE JOIN RULE: When writing a JOIN or LEFT JOIN that might produce NULLs for "
        "numeric metrics on the right side (i.e. left join where right rows may be absent), "
        "always wrap the right-side numeric columns in COALESCE(col, 0) or ISNULL(col, 0) "
        "(Azure SQL) / NVL(col, 0) (Oracle) / COALESCE(col, 0) (Snowflake) in the SELECT list "
        "so missing values appear as 0 rather than NULL in the result.\n\n"
        "- NULL-AWARE FILTERED AGGREGATE RULE: When the user asks for a single metric for a "
        "specific key/entity/filter (for example revenue for customer 123, cost for item X, "
        "quantity for warehouse Y), include diagnostics so the answer can distinguish true zero "
        "from missing data. Return COUNT_BIG(*) AS [MatchedRows] on Azure SQL "
        "(COUNT(*) AS MatchedRows on other DBs), COUNT(metric_col) AS [NonNullMetricRows], "
        "and COALESCE(SUM(metric_col), 0) AS [MetricName]. Do not return bare SUM(metric_col) "
        "for filtered single-metric questions because SUM returns NULL when all matched metric "
        "values are NULL.\n\n"
        "- RANKING RULE: When the user asks to rank, score, order entities by a metric "
        "('rank employees by sales', 'show sales rep ranking', 'ordered by performance', "
        "'leaderboard'), add a RANK() or DENSE_RANK() window column alongside the metric:\n"
        "  SELECT entity_col,\n"
        "         SUM(metric_col) AS TOTAL,\n"
        "         RANK() OVER (ORDER BY SUM(metric_col) DESC) AS RANK\n"
        "  FROM [schema].[table]\n"
        "  GROUP BY entity_col\n"
        "  ORDER BY RANK\n"
        "  Use DENSE_RANK() when the user explicitly asks for no gaps in rank numbers after ties. "
        "Always ORDER BY the rank column in the outer query.\n\n"
        f"Knowledge Base — available tables and their business context:\n{table_context}"
    )
    if conversation_history:
        history_lines = []
        for i, turn in enumerate(conversation_history, 1):
            q    = str(turn.get("question", ""))[:120]
            sql  = str(turn.get("sql",      ""))[:300]
            cols = ", ".join(str(c) for c in (turn.get("columns") or []))
            history_lines.append(
                f"Turn {i}:\n"
                f"  Question: {q}\n"
                f"  Columns returned: {cols}\n"
                f"  SQL used: {sql}"
            )
        base = base + (
            "\n\n## Session context (previous turns this conversation)\n"
            "Use these to resolve follow-up references such as \'top 5\', "
            "\'filter to X\', \'same metric for Y\', \'break that down by Z\'.\n"
            "COLUMN REUSE RULE: If the user's new question refers to a metric or "
            "concept (e.g. \'total revenue\', \'total charge\', \'spend\') and a "
            "previous turn already returned a column for that concept, reuse that "
            "EXACT column name. Do NOT invent a new column name for the same concept.\n"
            "Do NOT copy previous SQL verbatim — generate fresh SQL for the "
            "NEW question informed by this context.\n\n"
            + "\n\n".join(history_lines)
        )
    # Inject pre-built JOIN skeleton from entity graph when available.
    # The LLM must use this skeleton and must NOT change table names or JOINs.
    if graph_context and graph_context.get("enabled") and graph_context.get("join_skeleton"):
        skeleton  = graph_context["join_skeleton"]
        detected  = ", ".join(graph_context.get("detected", []))
        has_where = "WHERE " in skeleton.upper()
        where_instruction = (
            "The skeleton already contains a WHERE clause with static entity filters "
            "that MUST always be applied — they enforce business rules (active records, "
            "valid statuses, etc.). Add any question-driven conditions using AND after "
            "the existing WHERE clause — do NOT write a second WHERE keyword."
            if has_where else
            "Use this skeleton as the FROM/JOIN block for the base query. If the answer "
            "requires CTEs for YoY, MoM, running totals, percentage-of-total, or top-N per "
            "group, put this exact skeleton inside the base CTE."
        )
        anti_join_instruction = ""
        if graph_context.get("anti_join"):
            anti_join_instruction = (
                "\nANTI-JOIN GRAPH MODE: the question asks for missing or unmatched records. "
                "Keep the LEFT JOINs from the skeleton and add a WHERE right_side_key IS NULL "
                "predicate for the missing target table. Do not convert these joins back to INNER JOIN."
            )
        base = base + (
            "\n\n## Entity graph — pre-resolved JOIN structure\n"
            "The following FROM + JOIN structure has been resolved deterministically "
            "from the business entity graph. You MUST use this exact structure as shown. "
            "Do NOT change table names, aliases, JOIN/ON conditions, or any static "
            "filters already present in the ON or WHERE clauses — those are permanent "
            "business rules set by the admin.\n"
            "Detected entities: " + detected + "\n\n"
            "```sql\n" + skeleton + "\n```\n\n"
            + where_instruction + " Do not add or remove JOINs." + anti_join_instruction
        )
    if semantic_plan and semantic_plan.get("enabled") and semantic_plan.get("fields"):
        try:
            from core.semantic_planner import format_semantic_field_plan
            plan_text = format_semantic_field_plan(semantic_plan, db_type)
        except Exception:
            plan_text = ""
        if plan_text:
            # When the entity graph is also active, make priority explicit so the
            # LLM doesn't treat the two instruction sets as contradictory.
            graph_priority_note = ""
            if graph_context and graph_context.get("detected"):
                graph_priority_note = (
                    "NOTE: When the entity graph join skeleton (above) and the semantic "
                    "field plan conflict, the entity graph ON conditions take priority for "
                    "table aliases and join predicates. Use the semantic plan for column "
                    "selection only — do not introduce new joins or change aliases from the skeleton.\n\n"
                )
            base = base + (
                "\n\n" + plan_text + "\n\n"
                + graph_priority_note
                + "FIELD PLAN RULE: the mapped fields above are deterministic schema-derived "
                "bindings. Use the listed table.column pairs exactly. If the query requires "
                "CTEs, place the listed joins and source fields in the base CTE. If you cannot "
                "use the plan with the available schema, return CANNOT_GENERATE."
            )
    return base


def build_kb_system_prompt(erp_hints: str = "", naming_hints: str = "") -> str:
    """
    Stage 1 KB generation system prompt.
    DataPilot-style format. Generic — works for any database domain.
    Requires distinct values to be used. Flags ambiguous columns.

    erp_hints:    optional pre-formatted hint block from core.erp_column_dict.
                  Handles specific cryptic ERP short-code column names (ORNO, DIVI…).
    naming_hints: optional pre-formatted hint block from core.naming_convention.
                  Handles structural suffix/prefix patterns (_DMS_KEY, _AMT, _PCT,
                  AZ_ audit columns, table type). Complements erp_hints.
    """
    erp_block = ""
    if erp_hints:
        erp_block = (
            "ERP COLUMN REFERENCE — MANDATORY:\n"
            "The following column names are ERP short codes with known business meanings.\n"
            "You MUST use these translations in the ## Columns and ## Business Synonyms sections.\n"
            "Do NOT write [NEEDS CONTEXT] for any column listed here.\n\n"
            f"{erp_hints}\n\n"
        )

    naming_block = ""
    if naming_hints:
        naming_block = (
            "NAMING CONVENTION REFERENCE — MANDATORY:\n"
            "The following rules describe the structural role of columns in this table based on\n"
            "their suffix/prefix patterns. Apply these rules in EVERY section of the KB:\n"
            "  - ## Columns: document the role, aggregation rule, and SQL guidance for each column.\n"
            "  - ## Key Metrics: only list _AMT/_QTY/_CST/_PFT/_REV columns as additive measures.\n"
            "    Never list _PCT/_RATE/_RATIO columns as directly-summable metrics.\n"
            "  - ## Business Synonyms: map _DSC/_NM columns as the display label.\n"
            "    Map _DMS_KEY columns as 'surrogate key — always JOIN, never display'.\n"
            "  - ## Always Exclude: list AZ_/ETL_/DW_ columns as 'never use in business queries'.\n"
            "  - ## Common Query Patterns: show the JOIN pattern for every _DMS_KEY column;\n"
            "    show recalculation (not SUM) for every _PCT/_RATE column.\n\n"
            f"{naming_hints}\n\n"
        )

    return (
        f"{erp_block}"
        f"{naming_block}"
        "You are a senior data analyst writing a Knowledge Base document for an AI SQL generator. "
        "The document will be used at query time to produce accurate SQL — "
        "write it for the SQL generator, not for a human reader.\n\n"

        "CRITICAL RULES:\n"
        "1. Use ONLY the column names that appear in the schema provided. "
        "Never invent column names.\n"
        "2. If the schema includes a 'Distinct Values' column for a field, "
        "you MUST list those exact values and use them in examples. "
        "Do not guess or invent values.\n"
        "3. If a column has no distinct values and its business meaning is unclear "
        "(e.g. numeric scores, rates, thresholds with no obvious interpretation), "
        "mark it: [NEEDS CONTEXT] — business rule unknown, do not use in filters.\n"
        "4. If sample data contains placeholder values (SAMPLE_, VAL_, etc.), "
        "ignore them. Use only Distinct Values from the schema.\n"
        "5. No domain-specific assumptions. Write for any industry.\n"
        "6. If the user prompt includes a SCHEMA INTELLIGENCE block, treat it as "
        "deterministic grounding evidence. Use its roles, expanded names, default "
        "filter candidates, date-key warnings, and join aliases before guessing from "
        "raw column names. Low-confidence fields must stay marked as candidates or "
        "[NEEDS ADMIN CONTEXT].\n\n"

        "DOCUMENT FORMAT — produce all 8 sections for every table:\n\n"

        "## Overview\n"
        "Two sentences: (1) what this table represents and what it is used for; "
        "(2) if a DATA COVERAGE block is provided in the user prompt, include the "
        "date range and whether the data is live or archived.\n\n"

        "## Key Metrics\n"
        "List every measurable business concept this table answers. "
        "For each metric: business name → exact column name → filter condition if applicable.\n"
        "Format: - **Metric name**: `COLUMN_NAME` — Filter: `WHERE COLUMN = 'value'`\n"
        "Use the actual distinct values from the schema for the filter conditions.\n"
        "IMPORTANT: For any column that holds a monetary amount, quantity, rate, "
        "weight, duration, score or any other measurable quantity, add an explicit "
        "anchor line: 'PRIMARY MEASURE FOR <business term>: use COLUMN_NAME.' "
        "Derive <business term> from the business description and column name — "
        "for a pharmacy that might be revenue/charges, for an HR system it might be "
        "salary/hours, for logistics it might be weight/volume. This anchors the SQL "
        "generator so it uses the correct column for the tenant's actual domain.\n\n"

        "## Always Exclude\n"
        "Standard WHERE conditions that should ALWAYS be applied in every query.\n"
        "MANDATORY: If the user prompt contains a DATA QUALITY HINTS block, "
        "every column listed there MUST appear as an IS NOT NULL condition in this section — "
        "the null rates were measured from actual data.\n"
        "Also include: active-record flags, non-null date anchors, soft-delete columns.\n"
        "If none apply, write: None identified.\n\n"

        "## Columns\n"
        "For EVERY column in the schema:\n"
        "- `COLUMN_NAME` (TYPE): business meaning. "
        "If distinct values exist, list them: values are 'A', 'B', 'C'. "
        "If ambiguous, write [NEEDS CONTEXT].\n"
        "SPLIT NAME RULE: If the table has separate first/last name columns (FIRST_NAME/LAST_NAME, FNAME/LNAME, GIVEN_NAME/SURNAME or similar) but no combined full name column, document both columns and add the note: [SPLIT NAME - always concatenate for full name queries].\n\n"

        "## Aggregation Rules\n"
        "Tag EVERY numeric column with its aggregation type. "
        "The SQL generator reads this section literally — a wrong tag causes incorrect SQL.\n"
        "Types:\n"
        "- ADDITIVE: safe to SUM() across any dimension and any time period "
        "(e.g. invoice amounts, order quantities, transaction counts, costs, revenues).\n"
        "- NON-ADDITIVE: NEVER SUM() — always compute from component columns "
        "(e.g. rates, ratios, percentages, averages, unit prices — re-derive as "
        "SUM(numerator)/SUM(denominator)).\n"
        "- SEMI-ADDITIVE: SUM() valid within a single snapshot date only — "
        "do NOT sum across different dates "
        "(e.g. account balances, inventory on hand, headcount).\n"
        "Format: - `COLUMN_NAME`: ADDITIVE | NON-ADDITIVE | SEMI-ADDITIVE — [one-line rule]\n"
        "Example: - `INVOICE_AMT`: ADDITIVE — sum across customers, products, and periods\n"
        "         - `MARGIN_PCT`: NON-ADDITIVE — compute as SUM(margin)/SUM(revenue)\n"
        "         - `STOCK_QTY`: SEMI-ADDITIVE — sum within a date, not across dates\n"
        "If the table has no numeric columns, write: No numeric measures in this table.\n\n"

        "## Common Query Patterns\n"
        "Write 6-10 specific business questions and the EXACT SQL pattern for each. "
        "Use real column names and real distinct values from the schema.\n"
        "SCALE RULE: If the user prompt contains a ROW COUNT / SCALE block showing "
        ">100K rows, EVERY SELECT must include TOP N (T-SQL) or LIMIT N. "
        "Default to TOP 20 unless the question implies a full result set.\n"
        "SPLIT NAME PATTERNS: If the table has separate first/last name columns, "
        "every name-related query must show the concatenation: "
        "FIRST_NAME || ' ' || LAST_NAME AS FULL_NAME (or dialect equivalent).\n"
        "Format:\n"
        "Q: [natural language question]\n"
        "SQL: SELECT ... FROM [table] WHERE ...\n\n"

        "## Join Keys\n"
        "List every column that links to another table. "
        "Specify: `COLUMN_NAME` → links to [other table].[column].\n\n"

        "## Business Synonyms\n"
        "Map common plain-English terms to exact column names.\n"
        "Format: | Plain English | Column | Notes |\n"
        "If any column could be confused with a generic name from other databases "
        "(like TOTAL_AMOUNT, STATUS, ID), add a WARNING: note.\n"
        "CRITICAL — for tables containing PEOPLE (employees, customers, users, "
        "doctors, patients, agents, staff, vendors, suppliers, contacts, members): "
        "you MUST add a dedicated row for every informal term a business user might say. "
        "Examples: a PRESCRIBER table → also map: doctor, physician, clinician, provider, "
        "healthcare provider, HCP, medical professional, prescribing doctor. "
        "A STAFF table → also map: employee, worker, team member, agent, rep. "
        "A PATIENT table → also map: customer, client, member, beneficiary. "
        "A SUPPLIER table → also map: vendor, provider, partner, contractor. "
        "Use the table name and column names to infer the domain — even if sample data "
        "contains placeholder values (SAMPLE_), reason from the column names alone.\n\n"

        "Return only the Markdown document. All 8 sections are mandatory."
    )


_KB_DIALECT_RULES: dict[str, str] = {
    "azure_sql": (
        "SQL DIALECT: Azure SQL / T-SQL\n"
        "• Use TOP N (not LIMIT): SELECT TOP 10 ...\n"
        "• Date casting: CONVERT(date, column) or CAST(column AS date)\n"
        "• Date arithmetic: DATEDIFF(day|month|year, start, end)\n"
        "• Number formatting: FORMAT(value, 'N2')\n"
        "• Current date anchor: prefer MAX(date_col) over GETDATE()\n"
        "• DO NOT use DATE_TRUNC, TO_CHAR, LIMIT, DATE_ADD, or PostgreSQL syntax"
    ),
    "postgres": (
        "SQL DIALECT: PostgreSQL\n"
        "• Use LIMIT N (not TOP N)\n"
        "• Date truncation: DATE_TRUNC('month', column)\n"
        "• Number formatting: TO_CHAR(value, '999,999.99')\n"
        "• Date arithmetic: column - INTERVAL '1 month'\n"
        "• Current date anchor: prefer MAX(date_col) over CURRENT_DATE\n"
        "• DO NOT use TOP, CONVERT(date,...), DATEDIFF, FORMAT, or T-SQL syntax"
    ),
    "snowflake": (
        "SQL DIALECT: Snowflake\n"
        "• Use LIMIT N (not TOP N)\n"
        "• Date truncation: DATE_TRUNC('month', column)\n"
        "• Date arithmetic: DATEDIFF(day, start, end)\n"
        "• Number formatting: TO_CHAR(value)\n"
        "• Current date anchor: prefer MAX(date_col) over CURRENT_DATE\n"
        "• DO NOT use TOP, GETDATE(), or T-SQL specific syntax"
    ),
    "oracle": (
        "SQL DIALECT: Oracle\n"
        "• Use FETCH FIRST N ROWS ONLY for row limiting\n"
        "• Date truncation: TRUNC(column, 'MM')\n"
        "• Date arithmetic: ADD_MONTHS, MONTHS_BETWEEN\n"
        "• Number formatting: TO_CHAR(value, '999,999.99')\n"
        "• Current date anchor: prefer MAX(date_col) over SYSDATE\n"
        "• DO NOT use TOP, LIMIT, GETDATE(), or T-SQL syntax"
    ),
}


def build_kb_query_prompt(
    table_name: str,
    kb_content: str,
    business_desc: str,
    related_tables: str = "",
    db_type: str = "azure_sql",
    entity_type: str = "unknown",
    confirmed_joins: str = "",
) -> str:
    """
    Stage 2 KB generation prompt.
    Takes the Stage 1 KB output and generates a question-to-SQL translation
    document from the actual data and real column values in the KB.

    entity_type: 'fact', 'dimension', 'date_role', 'bridge', or 'unknown'.
                 Fact tables get more Q&A pairs and more cross-table join patterns.
    confirmed_joins: newline-separated admin-confirmed join paths (FROM_TABLE.col = TO_TABLE.col).
                     These take priority over auto-discovered paths.
    """
    related_block = ""
    if related_tables:
        related_block = (
            f"## Related Tables and Join Paths\n"
            f"The following join paths are EXACT and pre-verified. "
            f"Use them when generating cross-table Q&A pairs. "
            f"Do NOT invent join columns — only use the paths listed here.\n\n"
            f"{related_tables}\n\n"
        )

    confirmed_block = ""
    if confirmed_joins:
        confirmed_block = (
            "## Admin-Confirmed Join Paths (AUTHORITATIVE — highest priority)\n"
            "These join paths have been reviewed and confirmed by the database administrator. "
            "ALWAYS prefer these exact column names over the auto-discovered paths above "
            "when generating JOIN queries for these table pairs.\n\n"
            f"{confirmed_joins}\n\n"
        )

    dialect_rules = _KB_DIALECT_RULES.get(db_type, _KB_DIALECT_RULES["azure_sql"])

    is_fact = entity_type == "fact"
    min_pairs = 20 if is_fact else 10
    cross_min = 5 if is_fact else 3
    fact_note = (
        f" This is a FACT table — generate at least {min_pairs} pairs with "
        f"at least {cross_min} cross-dimension JOIN patterns."
        if is_fact else ""
    )

    return (
        f"You have been given the Knowledge Base document for table: {table_name}\n\n"
        f"Business context: {business_desc}\n\n"
        f"{related_block}"
        f"{confirmed_block}"
        f"Knowledge Base content:\n{kb_content}\n\n"
        f"## SQL Syntax Requirements\n"
        f"{dialect_rules}\n\n"
        f"Your task: Generate a QUERY TRANSLATION document that maps natural-language "
        f"business questions to exact SQL patterns for this table.{fact_note}\n\n"
        "RULES:\n"
        f"1. Generate at least {min_pairs} question-SQL pairs.\n"
        "2. Use ONLY column names and values that appear in the Knowledge Base above. "
        "Never invent column names or values.\n"
        "3. Apply the SQL dialect rules above — every SQL statement must use the correct "
        "syntax for this database type.\n"
        "4. Cover these question types:\n"
        "   - Counting/aggregation (how many, total, sum)\n"
        "   - Ranking (top N, highest, lowest, most, least)\n"
        "   - Filtering by status/category using actual distinct values from the KB\n"
        "   - Time-based filtering using MAX(date_column) as the reference, not system date\n"
        "   - Cross-dimension analysis (by department, by category, by type)\n"
        "   - Trend questions (this period vs last period)\n"
        f"   - IMPORTANT: Write at least {cross_min} cross-table patterns using JOIN to related "
        "tables — e.g. revenue by customer name, orders by region, transactions by product "
        "category, sales by employee. These cross-table patterns are the most common "
        "business questions.\n"
        "5. For every time-relative question (last month, last week, recent), "
        "use MAX(date_column) as the date anchor, not GETDATE()/CURRENT_DATE/SYSDATE.\n"
        "6. Write questions the way a non-technical business user would actually ask them.\n"
        "7. If Admin-Confirmed Join Paths are provided above, use those exact column names "
        "for JOIN conditions — never substitute other columns for those table pairs.\n"
        "8. If Related Tables and Join Paths are provided, generate at least "
        f"{cross_min} Q&A pairs that JOIN to those related tables using ONLY the exact join "
        "columns shown — never guess or substitute other columns.\n\n"
        "FORMAT — use this exact structure for each pair:\n"
        "Q: [natural language question a business user would ask]\n"
        "SQL: [complete, runnable SQL using real column names]\n\n"
        "Return only the question-SQL pairs, no other text."
    )


def build_biz_vocab_prompt(
    table_names: list[str],
    column_reference: str,
    business_desc: str,
) -> str:
    """
    Business vocabulary KB prompt.
    Grounds the LLM to real column names across all tables.
    Generic — no domain assumptions.
    """
    return (
        f"Business description:\n{business_desc}\n\n"
        f"Tables in this database:\n{', '.join(table_names)}\n\n"
        f"EXACT column names per table "
        f"(you MUST use ONLY these — never invent column names):\n"
        f"{column_reference}\n\n"
        "Generate a Business Vocabulary document that maps plain-English business terms "
        "to the exact table names and column names listed above. "
        "Include: key business entities, common metrics, synonyms, and how business "
        "language maps to the actual column names. "
        "Every column or table name you reference must come from the list above. "
        "Do not use any name not in that list."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Core completion function
# ══════════════════════════════════════════════════════════════════════════════

async def llm_complete(
    system: str,
    user: str,
    provider: Provider,
    model: str,
    api_key: str,
    max_tokens: int = 1024,
    azure_endpoint: str = "",
    azure_api_version: str = "2024-02-01",
    temperature: float = 0.7,
) -> tuple[str, int, int]:
    try:
        if provider == "anthropic":
            result = await _anthropic_complete(system, user, model, api_key, max_tokens, temperature)
        elif provider == "openai":
            result = await _openai_complete(system, user, model, api_key, max_tokens, temperature)
        elif provider == "azure_openai":
            result = await _azure_openai_complete(
                system, user, model, api_key, max_tokens, azure_endpoint, azure_api_version, temperature
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider!r}")
    except Exception as exc:
        record_llm_call(
            llm_provider=provider,
            llm_model=model,
            system=system,
            user=user,
            status="error",
            error_msg=str(exc),
        )
        raise

    record_llm_call(
        llm_provider=provider,
        llm_model=model,
        system=system,
        user=user,
        status="success",
    )
    return result


async def _anthropic_complete(system, user, model, api_key, max_tokens, temperature=0.7):
    import anthropic as _ant
    client = _ant.AsyncAnthropic(api_key=api_key)
    try:
        resp = await client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            temperature=temperature,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text.strip(), resp.usage.input_tokens, resp.usage.output_tokens
    except Exception as e:
        log.error("Anthropic API error: %s", e)
        raise RuntimeError(f"Anthropic API error: {e}") from e
    finally:
        await client.close()


async def _openai_complete(system, user, model, api_key, max_tokens, temperature=0.7):
    import openai as _oai
    client = _oai.AsyncOpenAI(api_key=api_key)
    try:
        resp = await client.chat.completions.create(
            model=model, max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
        )
        text = (resp.choices[0].message.content or "").strip()
        return text, resp.usage.prompt_tokens, resp.usage.completion_tokens
    except Exception as e:
        log.error("OpenAI API error: %s", e)
        raise RuntimeError(f"OpenAI API error: {e}") from e
    finally:
        await client.close()


async def _azure_openai_complete(system, user, model, api_key, max_tokens, endpoint, api_version, temperature=0.7):
    import openai as _oai
    if not endpoint:
        raise RuntimeError(
            "Azure OpenAI endpoint not configured. "
            "Go to Admin → System and enter your endpoint URL."
        )
    client = _oai.AsyncAzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint.rstrip("/"),
        api_version=api_version or "2024-02-01",
    )
    try:
        resp = await client.chat.completions.create(
            model=model, max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
        )
        text = (resp.choices[0].message.content or "").strip()
        return text, resp.usage.prompt_tokens, resp.usage.completion_tokens
    except Exception as e:
        log.error("Azure OpenAI error: %s", e)
        raise RuntimeError(
            f"Azure OpenAI error: {e}\n\n"
            "Check your endpoint URL, API key, and deployment name in Admin → System."
        ) from e
    finally:
        await client.close()


# ══════════════════════════════════════════════════════════════════════════════
# Provider resolution
# ══════════════════════════════════════════════════════════════════════════════

def resolve_provider(client: dict, purpose: str = "query") -> tuple[str, str, str, dict]:
    import store
    sys_cfg = store.get_all_system()

    provider = (
        client.get("llm_provider")
        or sys_cfg.get("default_llm_provider")
        or "anthropic"
    )

    if purpose == "kb":
        model = sys_cfg.get("kb_llm_model") or _default_model(provider, "high")
    else:
        model = (
            client.get("llm_model")
            or sys_cfg.get("default_llm_model")
            or _default_model(provider, "fast")
        )

    extra_kwargs: dict = {}
    if provider == "anthropic":
        api_key = sys_cfg.get("anthropic_api_key", "")
    elif provider == "openai":
        api_key = sys_cfg.get("openai_api_key", "")
    elif provider == "azure_openai":
        api_key  = sys_cfg.get("azure_openai_api_key", "")
        endpoint = sys_cfg.get("azure_openai_endpoint", "")
        version  = sys_cfg.get("azure_openai_api_version", "2024-02-01")
        if not endpoint:
            raise RuntimeError(
                "Azure OpenAI endpoint not configured. "
                "Go to Admin → System → Azure OpenAI settings."
            )
        extra_kwargs = {"azure_endpoint": endpoint, "azure_api_version": version}
    else:
        raise ValueError(f"Unknown provider: {provider!r}")

    if not api_key:
        raise RuntimeError(
            f"No API key configured for provider '{provider}'. "
            "Go to Admin → System and add your API key."
        )

    return provider, model, api_key, extra_kwargs


def _default_model(provider: str, quality: str) -> str:
    defaults = {
        "anthropic":    {"fast": "claude-sonnet-4-6", "high": "claude-opus-4-5"},
        "openai":       {"fast": "gpt-4o-mini",       "high": "gpt-4o"},
        "azure_openai": {"fast": "gpt-4o-mini",       "high": "gpt-4o"},
    }
    return defaults.get(provider, {}).get(quality, "gpt-4o-mini")
