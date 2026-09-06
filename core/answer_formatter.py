"""The plain-text diagnostic blob the portal parses back into a card.

The section labels here — "Most likely reason:", "Suggested next step:",
"Why:", "Technical details:", "SQL tried:" — are WIRE FORMAT, not copy. The
portal never displays them: it extracts each section by label and renders its
own translated heading above the value. So they must stay in English and
unchanged, and the values between them are what a reader actually sees.

``Kind:`` is the same kind of marker and exists for the same reason. The card's
kicker used to be chosen by regex-matching the English headline, which made the
headline wire format too — translating it silently downgraded every French
failure to the "no rows" kicker. The kind is stated now, so the prose above it
is free to be prose.
"""

from __future__ import annotations

from typing import Any

# Values the portal maps to a kicker. Machine tokens, never shown.
KIND_EXECUTION = "execution"
KIND_VALIDATION = "validation"
KIND_EMPTY = "empty"


def _kind_line(kind: str) -> list[str]:
    return ["", f"Kind: {kind}"]


def _t(msg_id: str, **kw) -> str:
    from core.i18n import t

    return t(msg_id, **kw)


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
        rca.get("headline") or _t("fail.zero_row.headline"),
        *_kind_line(KIND_EMPTY),
        "",
        # "Confidence:" is a wire label like the rest -- the portal extracts
        # by it and renders t('ui.chat.diag.confidence') above the value. Only
        # the VALUE is translated, and build_answer_confidence already does it.
        f"Confidence: {confidence.get('label') or _t('confidence.level.medium')} "
        f"({confidence.get('score', 0)}/100)",
        "",
        "Most likely reason:",
        rca.get("most_likely_reason") or _t("fail.zero_row.reason"),
        "",
        "Suggested next step:",
        rca.get("suggested_next_step") or _t("fail.zero_row.next_step"),
    ]
    if why:
        parts.extend(["", "Why:", _bullet_lines(why[:4])])
    if technical:
        parts.extend(["", "Technical details:", _bullet_lines(technical[:6])])
    parts.extend(["", f"SQL tried:\n```sql\n{sql_preview_fn(sql)}\n```"])
    return "\n".join(parts)


def format_failure_business_response(
    *,
    rca: dict[str, Any],
    sql: str = "",
    sql_preview_fn=None,
) -> str:
    """
    Business-readable hard-failure message (validator rejection, DB error).

    Uses the exact section labels of format_zero_row_business_response —
    "Most likely reason:", "Suggested next step:", "Technical details:" —
    which the portal's diagnostic-card renderer already parses, so failures
    get the same styled card with the technical detail demoted. Plain text
    degrades cleanly on Teams/Zoom.
    """
    technical = rca.get("technical_notes") or []
    parts = [
        rca.get("headline") or _t("fail.generic.headline"),
        *_kind_line(str(rca.get("kind") or KIND_VALIDATION)),
        "",
        "Most likely reason:",
        rca.get("most_likely_reason") or _t("fail.generic.reason"),
        "",
        "Suggested next step:",
        rca.get("suggested_next_step") or _t("fail.generic.next_step"),
    ]
    if technical:
        parts.extend(["", "Technical details:", _bullet_lines(technical[:6])])
    if sql:
        preview = sql_preview_fn(sql) if sql_preview_fn else sql[:1200]
        parts.extend(["", f"SQL tried:\n```sql\n{preview}\n```"])
    return "\n".join(parts)


def format_success_confidence_text(confidence: dict[str, Any]) -> str:
    reasons = confidence.get("reasons") or []
    warnings = confidence.get("warnings") or []
    lines = [
        f"Confidence: {confidence.get('label') or _t('confidence.level.medium')} "
        f"({confidence.get('score', 0)}/100)"
    ]
    if reasons:
        lines.append("Why:")
        lines.extend(f"- {reason}" for reason in reasons[:4])
    if warnings:
        lines.append("Watch-outs:")
        lines.extend(f"- {warning}" for warning in warnings[:3])
    return "\n".join(lines)
