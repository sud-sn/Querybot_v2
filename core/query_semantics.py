from __future__ import annotations

import re


def _has_any(text: str, phrases: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in phrases)


def analyze_query_intent(question: str) -> dict[str, bool]:
    """
    Classify broad analytics intent from plain-English phrasing.

    This is generic language understanding only. It deliberately avoids any
    client-specific business rules so it is safe to reuse in SQL grounding and
    clarification prompts.
    """
    q = (question or "").strip().lower()
    return {
        "has_employee_scope": bool(re.search(r"\b(employee|employees|staff|workforce|headcount)\b", q)),
        "wants_distinct_count": bool(re.search(r"\b(unique|distinct|deduplicated)\b", q)),
        "wants_grouping": bool(re.search(r"\b(by|per|grouped by|breakdown|split by|each|for each|based on)\b", q)),
        "wants_status_filter": bool(re.search(r"\b(status|marked as|who are|where|with)\b", q)),
        "wants_names": bool(re.search(r"\b(name|names|full name|fullname)\b", q)),
        "wants_time_series": bool(re.search(r"\b(trend|over time|by month|by week|by year|monthly|weekly|daily)\b", q)),
        "wants_comparison": bool(re.search(r"\b(compare|comparison|versus|vs|difference|gap)\b", q)),
        "wants_yoy": bool(re.search(
            r"\b(last year.{0,20}(compared|vs|versus|against|than).{0,20}(year before|prior year|previous year)"
            r"|year.{0,10}over.{0,10}year"
            r"|yoy"
            r"|(compared|vs|versus).{0,20}(last year|prior year|previous year|year before)"
            r"|(last year|this year|prior year|previous year).{0,20}(compared|vs|versus|difference|change|growth))\b",
            q,
        )),
        "wants_having_filter": bool(re.search(
            r"\b(more than|greater than|at least|over|above|less than|fewer than|under|below|exceeds?|threshold)"
            r".{0,30}\b(count|total|sum|average|avg|number of|amount)\b"
            r"|\b(count|total|sum|average|avg|number of|amount).{0,30}"
            r"(more than|greater than|at least|over|above|less than|fewer than|under|below)\b",
            q,
        )),
        "wants_top_per_group": bool(re.search(
            r"\b(top|best|highest|lowest|worst|bottom).{0,20}(per|in each|for each|by|within|across each)\b"
            r"|\b(per|in each|for each|within each).{0,20}(top|best|highest|lowest|worst|bottom)\b",
            q,
        )),
        "wants_share": bool(re.search(
            r"\b(percent(age)?|proportion|share|contribution|breakdown|what.{0,10}(percent|share|part)"
            r"|how much.{0,10}(contribut|make up|account))\b",
            q,
        )),
        "wants_missing_records": bool(re.search(
            r"\b(not in|no |never|without|missing|absent|never had|have no|lack|don.t have|do not have|zero)\b"
            r".{0,30}\b(record|order|transaction|sale|attendance|entry|match|result|absenc)\b"
            r"|\b(record|order|transaction|sale|attendance|entry|match|result|absenc).{0,30}"
            r"\b(not in|never|without|missing|absent|never had|have no)\b"
            r"|\b(employees?|customers?|products?|items?).{0,40}\b(no|never|without|not).{0,30}"
            r"\b(absence|order|sale|record|transaction|visit|attendance)\b",
            q,
        )),
        "wants_conditional_split": bool(re.search(
            r"\b(active.{0,10}(vs|versus|and|compare).{0,10}(inactive|terminated|former)"
            r"|male.{0,10}(vs|versus|and).{0,10}female"
            r"|split by|side by side|pivot|breakdown by status|count (for each|per) (status|type|category|group))\b",
            q,
        )),
        "wants_mom_qoq": bool(re.search(
            r"\b(month.{0,10}over.{0,10}month"
            r"|mom"
            r"|quarter.{0,10}over.{0,10}quarter"
            r"|qoq"
            r"|monthly (change|growth|trend|comparison)"
            r"|quarterly (change|growth|trend|comparison)"
            r"|how.{0,20}changed.{0,20}(each month|monthly|each quarter|quarterly)"
            r"|(each month|each quarter|by month|by quarter).{0,30}(change|growth|trend|compare))\b",
            q,
        )),
        "wants_cumulative": bool(re.search(
            r"\b(cumulative|running total|year.{0,5}to.{0,5}date|ytd|cumulative sum|total so far"
            r"|running sum|accumulated|progressive total)\b",
            q,
        )),
        "wants_rolling": bool(re.search(
            r"\b(rolling (average|avg|mean)|moving (average|avg|mean)"
            r"|trailing (average|avg)|smoothed|n.period average"
            r"|\d+.?(day|week|month).?(rolling|moving|trailing))\b",
            q,
        )),
        "wants_named_period": bool(re.search(
            r"\b(q[1-4]|quarter [1-4]|first quarter|second quarter|third quarter|fourth quarter"
            r"|h[12]|first half|second half|january|february|march|april|may|june|july|august"
            r"|september|october|november|december|last \d+ (month|week|day)s?)\b",
            q,
        )),
        "wants_ranking": bool(re.search(
            r"\b(rank(ed|ing)?|leaderboard|top performer|score board|ordered by performance"
            r"|best performing|worst performing|by performance|ranked list)\b",
            q,
        )),
    }


def summarize_query_intent(question: str) -> str:
    intent = analyze_query_intent(question)
    labels: list[str] = []
    if intent["wants_distinct_count"] and intent["has_employee_scope"]:
        labels.append("distinct employee counting")
    elif intent["has_employee_scope"]:
        labels.append("employee-focused query")
    if intent["wants_grouping"]:
        labels.append("grouped breakdown")
    if intent["wants_status_filter"]:
        labels.append("categorical status/value filtering")
    if intent["wants_names"]:
        labels.append("name lookup")
    if intent["wants_time_series"]:
        labels.append("time-series analysis")
    if intent["wants_comparison"]:
        labels.append("comparison framing")
    if intent["wants_yoy"]:
        labels.append("year-over-year comparison")
    if intent["wants_having_filter"]:
        labels.append("aggregate threshold filter (HAVING)")
    if intent["wants_top_per_group"]:
        labels.append("top-N per group (window function)")
    if intent["wants_share"]:
        labels.append("percentage / share of total")
    if intent["wants_missing_records"]:
        labels.append("anti-join / missing records")
    if intent["wants_conditional_split"]:
        labels.append("conditional aggregation / pivot")
    if intent["wants_mom_qoq"]:
        labels.append("month-over-month / quarter-over-quarter trend")
    if intent["wants_cumulative"]:
        labels.append("cumulative / running total")
    if intent["wants_rolling"]:
        labels.append("rolling / moving average")
    if intent["wants_named_period"]:
        labels.append("named period filter (Q/H/month)")
    if intent["wants_ranking"]:
        labels.append("ranking / leaderboard")
    return ", ".join(labels)


def build_generic_query_hints(question: str) -> str:
    """
    Return safe, cross-client guidance for common analytics phrasing.

    This is intentionally generic language understanding, not a client-specific
    semantic registry. It helps the SQL model interpret ordinary requests such
    as "unique employee count" and lightly misspelled filter values.
    """
    q = (question or "").strip().lower()
    if not q:
        return ""

    intent = analyze_query_intent(question)
    hints: list[str] = [
        "GENERIC QUERY INTERPRETATION RULES:",
        "- Exact schema-backed categorical values from the provided context are authoritative and must be preserved exactly, even when they look misspelled. Only normalize a value when that exact literal is absent from schema or business context.",
    ]

    if intent["wants_distinct_count"] and intent["has_employee_scope"]:
        hints.append(
            "- When the user asks for a unique employee count or distinct employee total, use COUNT(DISTINCT stable employee key). Prefer EMPLOYEE_ID, EMPLOYEE_NUMBER, PERSON_ID, PERSON_NUMBER, STAFF_ID, or USER_ID over employee names when such keys exist."
        )

    if intent["has_employee_scope"] and intent["wants_grouping"]:
        hints.append(
            "- When the user asks for employees by a category such as department, group by that category and count distinct employees rather than counting raw attendance or event rows unless the question explicitly asks for record volume."
        )

    if intent["wants_status_filter"]:
        hints.append(
            "- Phrases like 'marked as', 'who are', or 'with status' usually mean a filter on a categorical status or value column, not a different metric."
        )

    if intent["wants_names"] and intent["has_employee_scope"]:
        hints.append(
            "- If the user asks for employee names, return names after applying the requested filters; do not convert the request into an aggregate unless they explicitly ask for a count or ranking."
        )

    if intent["wants_yoy"]:
        hints.append(
            "- YEAR-OVER-YEAR COMPARISON DETECTED: The user wants to compare a metric across two "
            "consecutive years. Follow the YEAR-OVER-YEAR / PERIOD COMPARISON RULE in the system "
            "prompt exactly:\n"
            "  • Use a CTE to compute per-year aggregates; CAST the year column to INT.\n"
            "  • Derive the anchor year as MAX(CAST(year_col AS INT)) from the data — do NOT "
            "use GETDATE()/CURRENT_DATE/SYSDATE as the anchor and do NOT write MAX(col)-1 in a "
            "WHERE clause.\n"
            "  • LEFT JOIN the CTE to itself on prev_year = curr_year - 1.\n"
            "  • Always output: current year value, previous year value, absolute difference, "
            "and percentage change rounded to 2 decimal places.\n"
            "  • If approved metric formulas are present, use them inside the CTE aggregation — "
            "never substitute KB column names for approved formulas in a YoY query."
        )

    if intent["wants_having_filter"]:
        hints.append(
            "- HAVING FILTER DETECTED: The user wants to filter groups by an aggregate threshold "
            "(e.g. 'more than N', 'at least N'). Apply the HAVING RULE: filter on the aggregate "
            "in HAVING, not in WHERE. Use COUNT(*)/SUM(col) etc. directly in HAVING — do not "
            "reference an alias."
        )

    if intent["wants_top_per_group"]:
        hints.append(
            "- TOP-N PER GROUP DETECTED: The user wants the best/top/highest N records within "
            "each group. Apply the TOP-N PER GROUP RULE: use ROW_NUMBER() OVER (PARTITION BY "
            "group_col ORDER BY metric DESC) in a CTE, then filter WHERE rn <= N."
        )

    if intent["wants_share"]:
        hints.append(
            "- PERCENTAGE/SHARE DETECTED: The user wants to see proportions or contributions. "
            "Apply the PERCENTAGE OF TOTAL RULE: use SUM(metric)*100.0/SUM(SUM(metric)) OVER () "
            "in a single query with GROUP BY — do not use a separate subquery for the denominator."
        )

    if intent["wants_missing_records"]:
        hints.append(
            "- ANTI-JOIN / MISSING RECORDS DETECTED: The user wants records with no matching "
            "rows in another table. Apply the ANTI-JOIN RULE: use LEFT JOIN … WHERE right_key "
            "IS NULL. Do NOT use NOT IN (fails with NULLs) or NOT EXISTS."
        )

    if intent["wants_conditional_split"]:
        hints.append(
            "- CONDITIONAL SPLIT DETECTED: The user wants side-by-side counts for different "
            "categories (e.g. active vs inactive, male vs female). Apply the CONDITIONAL "
            "AGGREGATION RULE: use SUM(CASE WHEN status = 'X' THEN 1 ELSE 0 END) in a single "
            "query — do not use multiple subqueries or UNION."
        )

    if intent["wants_mom_qoq"]:
        hints.append(
            "- MONTH-OVER-MONTH / QUARTER-OVER-QUARTER DETECTED: The user wants period-over-period "
            "change. Apply the MoM/QoQ RULE: build a CTE with period buckets using DATE_TRUNC/"
            "FORMAT/TRUNC (dialect-appropriate), then use LAG() OVER (ORDER BY PERIOD) to get "
            "the prior period value. Always output: period, current value, prior value, "
            "difference, and PCT_CHANGE rounded to 2 decimal places."
        )

    if intent["wants_cumulative"]:
        hints.append(
            "- CUMULATIVE / RUNNING TOTAL DETECTED: The user wants a running/cumulative sum. "
            "Apply the RUNNING TOTAL RULE: use SUM(SUM(metric)) OVER (ORDER BY date_col ROWS "
            "UNBOUNDED PRECEDING) — nested aggregate window. GROUP BY the period first, then "
            "accumulate with the window function."
        )

    if intent["wants_rolling"]:
        hints.append(
            "- ROLLING / MOVING AVERAGE DETECTED: The user wants a smoothed average over a "
            "sliding window. Apply the MOVING AVERAGE RULE: AVG(metric) OVER (ORDER BY date_col "
            "ROWS BETWEEN N-1 PRECEDING AND CURRENT ROW). Default window = 3 periods if the "
            "user didn't specify. Cast integer columns to FLOAT for Azure SQL."
        )

    if intent["wants_named_period"]:
        hints.append(
            "- NAMED PERIOD FILTER DETECTED: The user referred to a specific quarter (Q1-Q4), "
            "half (H1/H2), month name, or 'last N months/weeks'. Apply the NAMED PERIOD FILTERS "
            "from the SQL syntax rules — use DATEPART/EXTRACT/QUARTER/MONTH with the correct "
            "integer mapping. Do NOT use GETDATE()/SYSDATE — anchor to MAX(date_col) in the data."
        )

    if intent["wants_ranking"]:
        hints.append(
            "- RANKING DETECTED: The user wants entities ordered/ranked by a metric. Apply the "
            "RANKING RULE: include RANK() OVER (ORDER BY SUM(metric) DESC) AS RANK alongside the "
            "aggregate in the SELECT list. Use DENSE_RANK() only if the user mentions 'no gaps'. "
            "Always ORDER BY the rank column."
        )

    summary = summarize_query_intent(question)
    if summary:
        hints.append(f"- Query-intent summary: {summary}.")

    if len(hints) == 1:
        return ""
    return "\n".join(hints) + "\n"
