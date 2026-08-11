"""Resolve the exact governed identifier for business-event counts.

The language layer can determine that "total orders" means an event count,
but COUNT(DISTINCT ...) is only correct when the expression inside it is the
identifier for the requested event at the intended grain.  This module turns
tenant semantic metadata into that exact, validator-ready target without
embedding client, ERP, table, or column names in runtime rules.
"""

from __future__ import annotations

import re
from typing import Any


_VALUE_WORDS = {
    "amount", "amt", "cost", "discount", "margin", "price", "profit", "qty",
    "quantity", "revenue", "sales", "units", "value",
}
_DISPLAY_WORDS = {"description", "desc", "dsc", "label", "name", "nm", "text"}
_LINE_WORDS = {"detail", "item", "line", "position", "row", "sequence"}
_IDENTIFIER_WORDS = {
    "business", "code", "id", "identifier", "no", "num", "number", "reference",
}
_SURROGATE_SUFFIXES = ("_DMS_KEY", "_KEY", "_SK", "_FK")


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _table_name(table: dict[str, Any]) -> str:
    return str(table.get("qualified_name") or table.get("table") or "").strip()


def _same_table(left: Any, right: Any) -> bool:
    left_parts = str(left or "").strip().strip("[]\"`").upper().split(".")
    right_parts = str(right or "").strip().strip("[]\"`").upper().split(".")
    if len(left_parts) >= 2 and len(right_parts) >= 2:
        return left_parts[-2:] == right_parts[-2:]
    return bool(left_parts and right_parts and left_parts[-1] == right_parts[-1])


def _entity_forms(entity: str) -> set[str]:
    base = _norm(entity)
    if not base:
        return set()
    forms = {base}
    if base.endswith("y"):
        forms.add(base[:-1] + "ies")
    elif base.endswith("s"):
        forms.add(base[:-1])
    else:
        forms.add(base + "s")
    return forms


def _field_text(field: dict[str, Any]) -> str:
    values: list[Any] = []
    for key in (
        "column", "expanded_name", "business_name", "display_name",
        "approved_meaning", "business_meaning", "description",
        "approved_use_case",
    ):
        values.append(field.get(key))
    for key in (
        "business_candidates", "synonyms", "approved_synonyms", "aliases",
    ):
        raw = field.get(key) or []
        values.extend(raw if isinstance(raw, (list, tuple, set)) else [raw])
    return _norm(" ".join(str(value or "") for value in values))


def _business_label(field: dict[str, Any], entity: str) -> str:
    expanded = str(
        field.get("business_name")
        or field.get("display_name")
        or field.get("expanded_name")
        or ""
    ).strip()
    expanded_words = set(_norm(expanded).split())
    physical_key_label = bool(
        expanded_words & {"sk", "fk"}
        or (expanded_words & {"key"} and not expanded_words & {"number", "reference", "identifier"})
    )
    if expanded and not re.fullmatch(r"[A-Z0-9_]+", expanded) and not physical_key_label:
        return expanded
    meaning = str(field.get("approved_meaning") or "").strip()
    if meaning:
        first = re.split(r"[.;]", meaning, maxsplit=1)[0].strip()
        if first:
            return first[:90]
    return f"{str(entity or 'Business event').strip().title()} identifier"


def _business_meaning(field: dict[str, Any], table: dict[str, Any], *, line_level: bool) -> str:
    meaning = str(
        field.get("approved_meaning")
        or field.get("business_meaning")
        or field.get("description")
        or field.get("approved_use_case")
        or ""
    ).strip()
    if meaning:
        meaning = re.split(r"\n", meaning, maxsplit=1)[0].strip()
    grain = str(table.get("grain") or "").strip()
    if line_level:
        grain_hint = "one value may represent an event line rather than the whole event"
    elif grain:
        grain_hint = grain
    else:
        grain_hint = "intended to identify one business event"
    return (meaning or grain_hint)[:180]


def _candidate(
    table: dict[str, Any],
    field: dict[str, Any],
    entity: str,
) -> dict[str, Any] | None:
    column = str(field.get("column") or "").strip()
    if not column:
        return None
    text = _field_text(field)
    words = set(text.split())
    identity_values: list[Any] = [
        column,
        field.get("business_name"),
        field.get("display_name"),
        field.get("expanded_name"),
    ]
    for key in ("business_candidates", "synonyms", "approved_synonyms", "aliases"):
        raw = field.get(key) or []
        identity_values.extend(raw if isinstance(raw, (list, tuple, set)) else [raw])
    identity_text = _norm(" ".join(str(value or "") for value in identity_values))
    identity_words = set(identity_text.split())
    forms = _entity_forms(entity)
    entity_match = any(
        re.search(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])", text)
        for form in forms
    )
    role = _norm(field.get("role") or "")
    aggregation = _norm(field.get("aggregation") or "")
    naming_role = _norm(field.get("naming_role") or "")

    # Words such as "sales" can describe the business event (Sales Order
    # Number) as well as a measure. Identifier evidence takes precedence unless
    # the semantic role explicitly marks the field as a measure.
    if role in {"measure", "measure candidate"} or (
        words & _VALUE_WORDS and not words & _IDENTIFIER_WORDS
    ):
        return None
    if words & _DISPLAY_WORDS and not (words & _IDENTIFIER_WORDS):
        return None
    identifier_evidence = bool(
        role in {"identifier", "dimension key", "key"}
        or aggregation == "identifier"
        or naming_role in {"business number", "identifier", "code", "surrogate fk"}
        or words & _IDENTIFIER_WORDS
        or column.upper().endswith(("_ID", "_NO", "_NUM", "_NUMBER", "_CODE", "_CD"))
        or column.upper().endswith(_SURROGATE_SUFFIXES)
    )
    # The owning fact is only supporting evidence. It must never turn an
    # unrelated key (for example CUSTOMER_SK on an order fact) into the event
    # identifier; the field's governed metadata must name the entity.
    if not identifier_evidence or not entity_match:
        return None

    score = 0
    evidence: list[str] = []
    if entity_match:
        score += 42
        evidence.append("field meaning matches requested entity")
    if role == "identifier":
        score += 26
        evidence.append("semantic role is identifier")
    elif role in {"dimension key", "key"}:
        score += 10
        evidence.append("semantic role is key")
    if aggregation == "identifier":
        score += 18
        evidence.append("aggregation policy is identifier")
    if words & _IDENTIFIER_WORDS:
        score += 16
        evidence.append("business-number semantics")
    if str(field.get("status") or "").casefold() == "approved":
        score += 15
        evidence.append("admin approved")
    try:
        score += min(10, max(0, int(field.get("confidence") or 0)) // 10)
    except (TypeError, ValueError):
        pass

    # Long descriptions often say that a valid header number is "shared by
    # every line". Use identity labels, rather than prose, to classify grain.
    line_level = bool(
        identity_words & _LINE_WORDS
        or re.search(r"(?:^|_)(?:LIN|LINE|ROW|POS)(?:_|$)", column.upper())
    )
    if line_level:
        score -= 38
        evidence.append("line-level identifier penalty")
    surrogate = column.upper().endswith(_SURROGATE_SUFFIXES)
    if surrogate:
        score -= 24
        evidence.append("surrogate-key penalty")

    return {
        "table": _table_name(table),
        "column": column,
        "business_name": _business_label(field, entity),
        "business_meaning": _business_meaning(field, table, line_level=line_level),
        "score": score,
        "confidence": max(0, min(100, score)),
        "line_level": line_level,
        "surrogate": surrogate,
        "status": str(field.get("status") or "generated"),
        "evidence": evidence,
    }


def resolve_count_target(
    entity: str,
    model: dict[str, Any] | None,
    *,
    source_scope: dict[str, Any] | None = None,
    confirmed_option: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve an exact count field or return business-facing ambiguity evidence."""
    entity = _norm(entity)
    if not entity:
        return {"status": "not_applicable", "entity": "", "candidates": []}

    scope = source_scope or {}
    selected_fact = str(scope.get("selected_fact") or "").strip()
    selected_facts = [
        str(value).strip() for value in (scope.get("selected_facts") or []) if str(value).strip()
    ]
    allowed = set(selected_facts or ([selected_fact] if selected_fact else []))
    fact_tables = [
        table for table in ((model or {}).get("tables") or [])
        if isinstance(table, dict)
        and str(table.get("type") or "").casefold() == "fact"
        and _table_name(table)
        and (not allowed or any(_same_table(_table_name(table), item) for item in allowed))
    ]

    candidates = [
        candidate
        for table in fact_tables
        for field in (table.get("fields") or [])
        if isinstance(field, dict)
        for candidate in [_candidate(table, field, entity)]
        if candidate is not None
    ]
    candidates.sort(key=lambda item: (-int(item["score"]), item["business_name"].casefold(), item["column"]))

    confirmed = confirmed_option or {}
    confirmed_table = str(confirmed.get("target_table") or "")
    confirmed_column = str(confirmed.get("target_column") or "")
    if confirmed_table and confirmed_column:
        selected = next((
            item for item in candidates
            if _same_table(item["table"], confirmed_table)
            and item["column"].upper() == confirmed_column.upper()
        ), None)
        if selected:
            return {
                "status": "selected", "entity": entity, "selected": selected,
                "candidates": candidates[:6], "reason": "user confirmed business meaning",
            }

    if not candidates:
        return {
            "status": "missing", "entity": entity, "selected": {}, "candidates": [],
            "reason": "no governed identifier candidate on the selected event fact",
        }

    top = candidates[0]
    runner_score = int(candidates[1]["score"]) if len(candidates) > 1 else -999
    confident = int(top["score"]) >= 70 and int(top["score"]) - runner_score >= 12
    if confident:
        return {
            "status": "selected", "entity": entity, "selected": top,
            "candidates": candidates[:6], "reason": "governed identifier evidence",
        }
    return {
        "status": "ambiguous", "entity": entity, "selected": {},
        "candidates": candidates[:6],
        "reason": "identifier meaning or grain requires user confirmation",
    }


def count_target_clarification_options(resolution: dict[str, Any]) -> list[dict[str, Any]]:
    """Return UI choices with business labels and hidden exact physical targets."""
    options: list[dict[str, Any]] = []
    used: set[str] = set()
    for candidate in (resolution or {}).get("candidates") or []:
        name = str(candidate.get("business_name") or "Business identifier").strip()
        meaning = str(candidate.get("business_meaning") or "").strip()
        label = f"{name} — {meaning}" if meaning else name
        label = label[:150]
        if label.casefold() in used:
            continue
        used.add(label.casefold())
        options.append({
            "id": f"count-target-{len(options) + 1}",
            "label": label,
            "value": name,
            "target_table": candidate.get("table"),
            "target_column": candidate.get("column"),
            "business_name": name,
            "business_meaning": meaning,
        })
        if len(options) >= 4:
            break
    return options
