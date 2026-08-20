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
    derived_metric_gap: str = "",
    weak_retrieval: bool = False,
    zero_match_result: bool = False,
    graph_scope: str = "",
    graph_resolution_failed: bool = False,
    fanout_risk: bool = False,
    result_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Convert technical query signals into a compact business-facing confidence score.

    The score is intentionally simple and deterministic. It is not a truth
    guarantee; it tells the user how much friction the answer encountered.

    zero_match_result: True for a single-row diagnostic aggregate whose own
    match-count column is zero (see response_builder.detect_zero_match_result)
    -- a real physical row exists, but it represents no matching data, not a
    successful single-value answer. Scored as if row_count were 0 regardless
    of the physical count passed in, and mutually exclusive with
    null_metric_issue (a real, non-zero match count with a missing metric).
    """
    validation = (validation_code or "ok").lower()
    rows = 0 if row_count is None else max(int(row_count), 0)
    if zero_match_result:
        rows = 0
        null_metric_issue = False
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

    if derived_metric_gap:
        score -= 25
        warnings.append(
            f"'{derived_metric_gap}' looks like a calculated business metric with no "
            "approved formula — this result may total a raw column instead of the real "
            "calculation. Ask your administrator to define the formula in the Metric "
            "Registry or Business Terms."
        )

    if weak_retrieval:
        score -= 20
        warnings.append(
            "The question matched the knowledge base only weakly — the answer may "
            "use the wrong table. Naming the metric or table explicitly usually fixes this."
        )

    if has_semantic_plan:
        score += 5
        reasons.append("Business terms were mapped through the semantic layer.")

    if has_graph_context:
        if str(graph_scope or "").lower() == "suggested_fallback":
            score -= 35
            warnings.append(
                "The query used unreviewed relationship suggestions; its joins require administrator review."
            )
        else:
            score += 5
            reasons.append("Configured entity relationships were used.")
    elif graph_resolution_failed and len(set(used_tables)) > 1:
        # Entity-graph resolution raised instead of returning "no graph", so
        # nothing checked the joins in this SQL against the approved ones. A
        # single-table answer has no joins to check and is unaffected; a
        # multi-table one is the model's own join plan, executed. Without this
        # the answer scored identically to a query that needed no governance.
        score -= 35
        warnings.append(
            "Relationship checks could not run for this question, so the joins "
            "between tables in this answer were not verified against your "
            "approved relationships. Treat the result as unconfirmed."
        )

    if fanout_risk:
        score -= 35
        warnings.append(
            "One or more relationships can multiply the requested result grain."
        )

    verification = result_verification or {}
    verification_status = str(verification.get("status") or "").lower()
    if verification_status == "pass":
        score += 5
        reasons.append("The returned result shape matched the analytical request.")
    elif verification_status in {"warning", "fail"}:
        score -= 10 if verification_status == "warning" else 30
        details = list(verification.get("errors") or []) + list(
            verification.get("warnings") or []
        )
        warnings.append(
            str(details[0])
            if details
            else "The returned result shape did not fully match the analytical request."
        )
        if verification_status == "fail":
            # A safe, executable query can still answer the wrong shape.  Do
            # not present that result as medium/high confidence merely because
            # schema validation passed.
            score = min(score, 49)

    # A repaired query may be usable, but compilation and execution alone do
    # not prove business correctness. Never present a repaired result as high
    # confidence until it is covered by deterministic result assertions.
    if retries:
        score = min(score, 75)

    score = max(0, min(100, score))
    level = _level(score)
    return {
        "score": score,
        "level": level,
        "label": _label(level),
        "reasons": reasons[:5],
        "warnings": warnings[:5],
    }
