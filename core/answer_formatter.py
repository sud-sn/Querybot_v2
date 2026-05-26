from __future__ import annotations

from typing import Any


def _bullet_lines(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items if item)


def format_zero_row_business_response(
    *,
    confidence: dict[str, Any],
    rca: dict[str, Any],
    sql: str,
    sql_preview_fn,
) -> str:
    warnings = confidence.get("warnings") or []
    reasons = confidence.get("reasons") or []
    why = warnings or reasons
    technical = rca.get("technical_notes") or []

    parts = [
        rca.get("headline") or "I could not find matching records for this question.",
        "",
        f"Confidence: {confidence.get('label', 'Medium confidence')} ({confidence.get('score', 0)}/100)",
        "",
        "Most likely reason:",
        rca.get("most_likely_reason") or "The query returned no rows for the selected data.",
        "",
        "Suggested next step:",
        rca.get("suggested_next_step") or "Check the filters, selected schema, or field mapping.",
    ]
    if why:
        parts.extend(["", "Why:", _bullet_lines(why[:4])])
    if technical:
        parts.extend(["", "Technical details:", _bullet_lines(technical[:6])])
    parts.extend(["", f"SQL tried:\n```sql\n{sql_preview_fn(sql)}\n```"])
    return "\n".join(parts)


def format_success_confidence_text(confidence: dict[str, Any]) -> str:
    reasons = confidence.get("reasons") or []
    warnings = confidence.get("warnings") or []
    lines = [
        f"Confidence: {confidence.get('label', 'Medium confidence')} ({confidence.get('score', 0)}/100)"
    ]
    if reasons:
        lines.append("Why:")
        lines.extend(f"- {reason}" for reason in reasons[:4])
    if warnings:
        lines.append("Watch-outs:")
        lines.extend(f"- {warning}" for warning in warnings[:3])
    return "\n".join(lines)
