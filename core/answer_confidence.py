from __future__ import annotations

from typing import Any


def _level(score: int) -> str:
    if score >= 80:
        return "high"
    if score >= 50:
        return "medium"
    return "low"


def _label(level: str) -> str:
    return {
        "high": "High confidence",
        "medium": "Medium confidence",
        "low": "Low confidence",
    }.get(level, "Medium confidence")


def build_answer_confidence(
    *,
    validation_code: str = "ok",
    row_count: int | None = None,
    retry_count: int = 0,
    has_semantic_plan: bool = False,
    has_graph_context: bool = False,
    tables_used: list[str] | None = None,
    empty_tables: list[str] | None = None,
    null_metric_issue: bool = False,
) -> dict[str, Any]:
    """
    Convert technical query signals into a compact business-facing confidence score.

    The score is intentionally simple and deterministic. It is not a truth
    guarantee; it tells the user how much friction the answer encountered.
    """
    validation = (validation_code or "ok").lower()
    rows = 0 if row_count is None else max(int(row_count), 0)
    retries = max(int(retry_count or 0), 0)
    used_tables = [str(t) for t in (tables_used or []) if str(t).strip()]
    empty = [str(t) for t in (empty_tables or []) if str(t).strip()]

    score = 70
    reasons: list[str] = []
    warnings: list[str] = []

    if validation in {"ok", "pass", "trusted_metric"}:
        score += 15
        reasons.append("SQL passed schema validation.")
    else:
        # validation is always a non-empty string here (normalised above)
        score -= 25
        warnings.append("SQL needed validation attention before it could be trusted.")

    if retries:
        score -= min(20, 10 * retries)
        warnings.append("The SQL needed a repair retry before execution.")
    else:
        reasons.append("No SQL repair retry was needed.")

    if row_count is not None:
        if rows > 0:
            score += 10
            reasons.append(f"The query returned {rows} row{'s' if rows != 1 else ''}.")
        else:
            score -= 20
            warnings.append("The query ran successfully but returned no rows.")

    if empty:
        score -= 35
        listed = ", ".join(empty[:3])
        suffix = "..." if len(empty) > 3 else ""
        warnings.append(f"One table used by the query has no records: {listed}{suffix}.")
    elif used_tables:
        reasons.append("The answer used known database tables.")

    if null_metric_issue:
        score -= 25
        warnings.append("Records matched the filter, but the requested metric values were null or missing.")

    if has_semantic_plan:
        score += 5
        reasons.append("Business terms were mapped through the semantic layer.")

    if has_graph_context:
        score += 5
        reasons.append("Configured entity relationships were used.")

    score = max(0, min(100, score))
    level = _level(score)
    return {
        "score": score,
        "level": level,
        "label": _label(level),
        "reasons": reasons[:5],
        "warnings": warnings[:5],
    }
