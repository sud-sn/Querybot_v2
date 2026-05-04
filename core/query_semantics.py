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

    summary = summarize_query_intent(question)
    if summary:
        hints.append(f"- Query-intent summary: {summary}.")

    if len(hints) == 1:
        return ""
    return "\n".join(hints) + "\n"
