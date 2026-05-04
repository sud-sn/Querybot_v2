from __future__ import annotations

import re
from typing import Any


_SEMANTIC_FAMILIES: dict[str, dict[str, Any]] = {
    "attendance": {
        "synonyms": [
            "attendance", "absence", "absent", "absenteeism", "late",
            "lateness", "leave", "present",
        ],
        "business_meaning": "Measures workforce presence, attendance state, or absence-related activity.",
        "why_it_matters": "This can affect staffing capacity, schedule reliability, and workforce continuity.",
        "safe_next_steps": ["compare by department", "compare by manager", "compare by month"],
    },
    "attrition": {
        "synonyms": ["attrition", "resignation", "turnover", "exit", "termination"],
        "business_meaning": "Represents employee exits or workforce loss over a period.",
        "why_it_matters": "This can affect hiring demand, retention planning, and workforce stability.",
        "safe_next_steps": ["compare by department", "compare by month", "compare by tenure band"],
    },
    "nationality_count": {
        "synonyms": ["nationality", "citizenship", "employee nationality"],
        "business_meaning": "Represents the distribution of employees across nationality groups.",
        "why_it_matters": "This may matter for workforce diversity, localization targets, and hiring mix analysis.",
        "safe_next_steps": ["compare top groups", "analyze concentration", "compare by department"],
    },
    "headcount": {
        "synonyms": ["employee", "employees", "headcount", "staff", "workforce"],
        "business_meaning": "Represents the size or distribution of the workforce.",
        "why_it_matters": "This is useful for workforce planning, organization design, and staffing analysis.",
        "safe_next_steps": ["compare by department", "compare by location", "compare over time"],
    },
    "revenue": {
        "synonyms": ["revenue", "sales", "income", "billing", "charges"],
        "business_meaning": "Represents earned commercial value across customers, products, or periods.",
        "why_it_matters": "This helps track business performance, demand mix, and concentration risk.",
        "safe_next_steps": ["compare top segments", "compare by month", "analyze concentration"],
    },
    "count": {
        "synonyms": ["count", "number", "total", "how many", "volume"],
        "business_meaning": "Represents the volume or frequency of records in the selected scope.",
        "why_it_matters": "This helps quantify scale before investigating patterns or segment differences.",
        "safe_next_steps": ["break down by category", "compare top groups", "compare over time"],
    },
}


def _tokenize(*parts: str) -> set[str]:
    tokens: set[str] = set()
    for part in parts:
        for token in re.split(r"[^a-z0-9_]+", (part or "").lower()):
            token = token.strip("_ ")
            if token:
                tokens.add(token)
    return tokens


def detect_metric_semantics(
    question: str,
    *,
    context: dict | None = None,
    business_context: str = "",
) -> dict[str, Any]:
    """
    Return a safe semantic description for the metric/question.

    This is intentionally descriptive rather than client-specific. It can be
    shared with the LLM without leaking raw rows or internal rules.
    """
    ctx = context or {}
    tokens = _tokenize(
        question,
        business_context[:500],
        ctx.get("label_col", ""),
        ctx.get("value_col", ""),
        " ".join(ctx.get("numeric_cols") or []),
        " ".join(ctx.get("text_cols") or []),
    )

    best_key = "count"
    best_score = 0
    for key, family in _SEMANTIC_FAMILIES.items():
        synonyms = {_s.lower() for _s in family.get("synonyms", [])}
        score = 0
        for synonym in synonyms:
            synonym_tokens = _tokenize(synonym)
            if synonym_tokens and synonym_tokens <= tokens:
                score += max(2, len(synonym_tokens))
            elif synonym in " ".join(sorted(tokens)):
                score += 1
        if score > best_score:
            best_key = key
            best_score = score

    selected = _SEMANTIC_FAMILIES[best_key]
    return {
        "key": best_key,
        "business_meaning": selected["business_meaning"],
        "why_it_matters": selected["why_it_matters"],
        "safe_next_steps": list(selected.get("safe_next_steps", [])),
    }
