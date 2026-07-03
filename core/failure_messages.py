"""
core/failure_messages.py

Translate hard failures (validator rejections, database errors) into the same
business-readable {headline / most_likely_reason / suggested_next_step /
technical_notes} shape core/answer_rca.build_business_rca uses for zero rows —
so every failure the user sees leads with plain language and a concrete next
step, with the technical detail kept but demoted.

Raw error text is NEVER lost: callers keep logging the original string to
query_log / answer_trace; this module only shapes what reaches the chat.
"""

from __future__ import annotations

import re
from typing import Any

_MAX_TECH_CHARS = 300


# ── DB error sanitizer ────────────────────────────────────────────────────────

# Ordered: first match wins. Matchers run against the CLEANED error text
# (driver prefixes stripped), case-insensitive.
_DB_ERROR_MAP: list[tuple[str, str, str]] = [
    (
        r"login timeout|HYT00",
        "The database did not respond in time.",
        "Try again in a minute; if it keeps happening, ask your administrator to check that the database is running and reachable.",
    ),
    (
        r"login failed|\b18456\b|\b28000\b",
        "QueryBot could not sign in to the database.",
        "Ask your administrator to verify the database credentials in the connection settings.",
    ),
    (
        r"invalid object name|\b208\b.*object|table or view does not exist|\bORA-00942\b",
        "A table this query needs does not exist in the database.",
        "Ask your administrator to re-run schema discovery so QueryBot's table list matches the database.",
    ),
    (
        r"invalid column name|\bORA-00904\b",
        "A column this query used does not exist in the database.",
        "Rephrase using a field shown in a previous answer, or ask your administrator to rebuild the knowledge base.",
    ),
    (
        r"multi-part identifier .* could not be bound|\b4104\b",
        "The query referenced a table that was not joined in.",
        "Try asking the question again in different words; if it persists, ask your administrator to check the metric's join configuration.",
    ),
    (
        r"is invalid in the select list because it is not contained in either an aggregate|\b8120\b|not a GROUP BY expression|\bORA-00979\b",
        "The query mixed grouped and ungrouped columns in a way the database rejects.",
        "Try asking the question again — often rephrasing with an explicit breakdown (e.g. 'by customer') fixes this.",
    ),
    (
        r"permission was denied|\b229\b.*denied|\b297\b|insufficient privileges|\bORA-01031\b",
        "The database account does not have permission to read this data.",
        "Ask your administrator to grant read access to the table mentioned in the technical details.",
    ),
    (
        r"incorrect syntax|syntax error|\bORA-00933\b|\bORA-00936\b",
        "The generated query had a syntax error.",
        "Rephrase the question more simply — one metric and one breakdown at a time usually works best.",
    ),
    (
        r"conversion failed|error converting data type|\b241\b|\b245\b|\b8114\b|\bORA-01722\b|\bORA-01861\b",
        "A date or number in the query did not match the column's format.",
        "Try stating the date or number differently (for example 'in March 2025' instead of a raw date).",
    ),
    (
        r"divide by zero|\b8134\b|\bORA-01476\b",
        "The calculation divided by zero for this data.",
        "Ask your administrator to add a divide-by-zero guard (NULLIF) to this metric's formula.",
    ),
    (
        r"timed out|timeout expired|query timeout",
        "The query took too long and was stopped.",
        "Try narrowing the question — add a date range or a specific customer/product filter.",
    ),
    (
        r"communication link|\b08S01\b|connection (?:was )?(?:closed|reset|broken)|TCP Provider",
        "The connection to the database was interrupted.",
        "Try again — this is usually temporary. If it persists, ask your administrator to check network access to the database.",
    ),
    (
        r"deadlock|\b1205\b",
        "The database was busy and cancelled this query to resolve a conflict.",
        "Try again in a moment.",
    ),
]


def _clean_db_error(raw: str) -> str:
    """Strip driver noise: pyodbc tuple wrapping and [Vendor][Driver] prefixes."""
    text = (raw or "").strip()
    # pyodbc renders errors as ('42S02', "[Microsoft]...message. (208) (SQLExecDirectW)")
    m = re.match(r"^\(\s*'[^']*'\s*,\s*[\"'](.*)[\"']\s*\)$", text, re.DOTALL)
    if m:
        text = m.group(1)
    # Drop every leading [ ... ] bracket group (driver/vendor/server chain).
    text = re.sub(r"^(?:\[[^\]]*\]\s*)+", "", text).strip()
    # Drop trailing pyodbc call-site markers like (SQLExecDirectW)
    text = re.sub(r"\s*\(SQL[A-Za-z]+W?\)\s*$", "", text).strip()
    return text


def sanitize_db_error(raw: str) -> dict[str, str]:
    """
    Return {plain_reason, next_step, cleaned} for a raw driver/database error.

    cleaned = the original message minus driver prefixes — still technical,
    kept for the technical-details section. plain_reason/next_step come from
    the ordered matcher table; unknown errors fall back to the first sentence
    of the cleaned text so the user is never shown bracket soup.
    """
    cleaned = _clean_db_error(raw)
    probe = f"{raw or ''} || {cleaned}"
    for pattern, plain_reason, next_step in _DB_ERROR_MAP:
        if re.search(pattern, probe, re.IGNORECASE):
            return {"plain_reason": plain_reason, "next_step": next_step, "cleaned": cleaned}
    first_sentence = re.split(r"(?<=[.!?])\s", cleaned, maxsplit=1)[0][:200].strip()
    return {
        "plain_reason": first_sentence or "The database reported an unexpected error.",
        "next_step": "Try rephrasing the question; if it keeps failing, share the technical details with your administrator.",
        "cleaned": cleaned,
    }


# ── Validator-code translations ───────────────────────────────────────────────

_VALIDATION_REASONS: dict[str, str] = {
    "field_plan_mismatch": "The generated query did not use the approved business field mapping for one of the terms in your question.",
    "unknown_column": "The generated query used a column that does not exist in your data.",
    "unknown_table": "The generated query used a table that does not exist in your data.",
    "access_denied": "Your account does not have access to one of the tables this question needs.",
    "anti_join_shape": "This question asks about missing records, and the generated query did not check for them correctly.",
    "date_key_format": "The generated query used a date key column incorrectly.",
    "metric_formula_mismatch": "The generated query did not use the approved formula for a metric mentioned in your question.",
    "null_aggregate_diagnostic": "The generated query could not distinguish 'zero' from 'no data' for this metric.",
    "order_alias_mismatch": "The generated query sorted by a column it did not select.",
    "parse": "The generated query was not valid SQL.",
    "ddl": "The generated query tried an operation that is not allowed — only read-only questions are supported.",
    "cannot_generate": "I could not turn this question into a query using the available data.",
}

_VALIDATION_NEXT_STEPS: dict[str, str] = {
    "access_denied": "Ask your administrator to grant your group access to the table named in the technical details.",
    "cannot_generate": "Try naming the metric and the breakdown explicitly (e.g. 'total revenue by customer'), or add a time range.",
}
_DEFAULT_VALIDATION_NEXT_STEP = (
    "Try naming the metric and the breakdown explicitly (e.g. 'total revenue by customer'). "
    "If it keeps failing, ask your administrator to review the field mapping for this term."
)


def translate_failure(
    *,
    kind: str,
    code: str = "",
    reason: str = "",
    exception_text: str = "",
    sql: str = "",
    question: str = "",
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a business-readable failure RCA. Same dict shape as
    core/answer_rca.build_business_rca: {headline, most_likely_reason,
    suggested_next_step, technical_notes}. Never raises.
    """
    try:
        if kind == "execution":
            info = sanitize_db_error(exception_text or reason)
            technical = []
            if info["cleaned"]:
                technical.append(f"Database error: {info['cleaned'][:_MAX_TECH_CHARS]}")
            return {
                "headline": "I could not run this query against your database.",
                "most_likely_reason": info["plain_reason"],
                "suggested_next_step": info["next_step"],
                "technical_notes": technical,
            }

        if kind == "validation":
            code_key = (code or "").strip().lower()
            plain = _VALIDATION_REASONS.get(
                code_key,
                "The generated query did not pass QueryBot's safety and accuracy checks.",
            )
            technical = []
            if code_key:
                technical.append(f"Validation: {code_key}")
            if reason:
                technical.append((reason or "").strip()[:_MAX_TECH_CHARS])
            return {
                "headline": "I could not build a trusted query for this question.",
                "most_likely_reason": plain,
                "suggested_next_step": _VALIDATION_NEXT_STEPS.get(code_key, _DEFAULT_VALIDATION_NEXT_STEP),
                "technical_notes": technical,
            }

        # Unknown kind — generic but safe.
        technical = [t for t in [(reason or exception_text or "").strip()[:_MAX_TECH_CHARS]] if t]
        return {
            "headline": "I could not answer this question.",
            "most_likely_reason": "Something went wrong while preparing or running the query.",
            "suggested_next_step": "Try rephrasing the question; if it keeps failing, share the technical details with your administrator.",
            "technical_notes": technical,
        }
    except Exception:
        return {
            "headline": "I could not answer this question.",
            "most_likely_reason": "Something went wrong while preparing or running the query.",
            "suggested_next_step": "Try rephrasing the question, or contact your administrator.",
            "technical_notes": [],
        }
