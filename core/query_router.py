"""
core/query_router.py

Decides whether a user question should be answered by the in-memory
result cache (Tier 2 DuckDB) rather than making a fresh round-trip to
the production database.

Route to cache when ALL of:
  1. A cached result exists for this session
  2. The question clearly refers to the previously-returned data
     (keywords: "above average", "below average", "from these results", etc.)
  3. The question is NOT asking for a completely new topic

Also builds the DuckDB-specific system prompt used for SQL generation
against the virtual `result` table.
"""

from __future__ import annotations

import re

# ── Pattern matching ──────────────────────────────────────────────────────────

# Strong signals: user is explicitly referring to the current result set
_RESULT_REF_RE = re.compile(
    r"\b("
    r"above average|below average|above the average|below the mean|above the mean|"
    r"above median|below median|"
    r"outlier|outliers|anomal|"
    r"from (these|this) results?|in (these|this) results?|"
    r"within (these|this)|among (these|this)|"
    r"of (these|this) rows?|"
    r"top \d+ of (these|this)|bottom \d+ of (these|this)|"
    r"filter (these|this)|narrow (these|this) down|"
    r"rank (these|this)|ranked|percentile|"
    r"ratio of|ratio between|"
    r"std\s*dev|standard deviation|variance|"
    r"cumulative|running total|running sum|"
    r"window|rolling|moving average|"
    r"who (is|are) the (highest|lowest|best|worst) (here|in this|of these)|"
    r"compare (these|this) results?"
    r")\b",
    re.IGNORECASE,
)

# Medium signals: analytical operations on "top N / bottom N" of the result,
# or explicit aggregate requests that make no sense without a prior result
_ANALYTIC_TOPN_RE = re.compile(
    r"\b("
    r"(avg|average|mean|sum|total|min|max|minimum|maximum|count)\s+of\s+(top|bottom)\s+\d+|"
    r"(top|bottom)\s+\d+\s+(avg|average|mean|sum|total|min|max|by|with|having)|"
    r"(avg|average|mean|median)\s+of\s+(these|this|the\s+\w+)|"
    r"(highest|lowest|best|worst)\s+\d+\s+\w+|"
    r"which\s+(one|ones|item|items|row|rows)\s+(is|are)\s+(the\s+)?(highest|lowest|best|worst)|"
    r"sort\s+(these|this|them|by)|order\s+(these|this|them)\s+by|"
    r"show\s+(only|me)?\s*(top|bottom)\s+\d+"
    r")\b",
    re.IGNORECASE,
)

# Weaker signals — only route if combined with a result-reference context word
_ANALYTIC_RE = re.compile(
    r"\b(average|mean|median|sum|total|minimum|maximum|percentile|rank|sort)\b",
    re.IGNORECASE,
)

_THESE_RE = re.compile(
    r"\b(these|this result|the result|the data|returned|shown|listed)\b",
    re.IGNORECASE,
)

# Prefix set by the "Ask about these results" button in the UI
_FROM_RESULTS_PREFIX_RE = re.compile(
    r"^from (these|this) results?,?\s*",
    re.IGNORECASE,
)


def should_route_to_result_cache(
    question: str,
    has_cached_result: bool,
    cached_col_names: list[str] | None = None,
) -> bool:
    """
    Return True when the question should be answered from the cached result
    rather than hitting the production database.

    Conservative: only routes when there is a clear reference to the current
    result set. Ambiguous questions always go to the DB (safe default).

    cached_col_names: column names from the last result. When provided,
    routing also fires if the question mentions an analytic operation
    AND one of those column names appears in the question.
    """
    if not has_cached_result:
        return False

    q = question.strip()

    # Explicit "From these results, ..." prefix (set by the Ask button)
    if _FROM_RESULTS_PREFIX_RE.match(q):
        return True

    # Strong match — always route
    if _RESULT_REF_RE.search(q):
        return True

    # Medium match — "avg of top 5 formulas", "top 3 by revenue", etc.
    if _ANALYTIC_TOPN_RE.search(q):
        return True

    # Weak match — route only when user also references "these / this result"
    if _ANALYTIC_RE.search(q) and _THESE_RE.search(q):
        return True

    # Column-name match — analytic question that names a column from the result
    # e.g. "what is the avg of TOTAL_REVENUE?" when TOTAL_REVENUE is in the cache
    if cached_col_names and _ANALYTIC_RE.search(q):
        q_lower = q.lower()
        for col in cached_col_names:
            # Match the column name or a natural-language version of it
            col_lower = col.lower().replace("_", " ")
            if col_lower in q_lower or col.lower() in q_lower:
                return True

    return False


# ── DuckDB system prompt ──────────────────────────────────────────────────────

def build_duckdb_system_prompt(schema: list[dict], db_type: str = "azure_sql") -> str:
    """
    Build the SQL generation system prompt for in-memory DuckDB queries.

    DuckDB supports the full analytics function set (MEDIAN, STDDEV_POP,
    PERCENTILE_CONT, CORR, window functions) that production databases may
    not, so this prompt is intentionally more permissive than the main one.

    `schema` is a list of {name, type} dicts from result_cache.get_schema().
    """
    col_lines = "\n".join(
        f"  - {s['name']}  ({s['type']})" for s in schema
    )
    return (
        "You are a DuckDB SQL expert. Convert the user's plain-English question "
        "into a valid DuckDB SELECT query against the in-memory table called 'result'.\n\n"
        "The 'result' table has these columns:\n"
        f"{col_lines}\n\n"
        "RULES:\n"
        "- Use ONLY these column names — never invent new ones\n"
        "- Table name is always: result\n"
        "- DuckDB supports: MEDIAN(), STDDEV_POP(), STDDEV_SAMP(), PERCENTILE_CONT(x) "
        "WITHIN GROUP (ORDER BY col), CORR(col1, col2), AVG(), SUM(), window functions "
        "(ROW_NUMBER(), RANK(), NTILE(), LAG(), LEAD()), QUALIFY clause\n"
        "- For 'above/below average': use AVG() in a subquery or window: "
        "WHERE col > (SELECT AVG(col) FROM result)\n"
        "- For 'percentile rank': use PERCENT_RANK() OVER (ORDER BY col)\n"
        "- For 'outliers': WHERE col > (SELECT AVG(col) + 2 * STDDEV_POP(col) FROM result)\n"
        "- For 'ratio': SELECT col_a / NULLIF(col_b, 0) AS ratio\n"
        "- For 'running total': SELECT col, SUM(metric) OVER (ORDER BY col) AS running_total\n"
        "- Row limit: default to no LIMIT unless the user specifies a number\n"
        "- Return ONLY the raw SQL. No markdown fences. No explanation.\n"
        "- If the question cannot be answered from this table, return exactly: CANNOT_GENERATE\n"
    )
