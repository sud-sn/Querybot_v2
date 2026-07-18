"""Resolve governed date roles for metrics and business contexts."""

from __future__ import annotations

import re
from typing import Any

from core.date_roles import (
    normalize_date_key_type,
    normalize_date_role_text,
    question_has_temporal_intent,
)


def _terms(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        raw = [str(item) for item in value]
    else:
        raw = re.split(r"[,;\n|]+", str(value or ""))
    seen: set[str] = set()
    result: list[str] = []
    for item in raw:
        normalized = normalize_date_role_text(item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _table_identity(value: Any) -> tuple[str, str]:
    full = str(value or "").strip().strip("[]`").upper()
    return full, full.split(".")[-1] if full else ""


def _same_table(left: Any, right: Any) -> bool:
    left_full, left_bare = _table_identity(left)
    right_full, right_bare = _table_identity(right)
    return bool(left_full and right_full and (left_full == right_full or left_bare == right_bare))


def _binding_score(question: str, binding: dict) -> int:
    q = normalize_date_role_text(question)
    q_tokens = set(q.split())
    best = 0
    for phrase in _terms([binding.get("context_name", ""), *_terms(binding.get("aliases", ""))]):
        if phrase in q:
            best = max(best, 100 + len(phrase.split()) * 8)
            continue
        tokens = set(phrase.split())
        if tokens and tokens <= q_tokens:
            best = max(best, 70 + len(tokens) * 6)
        elif tokens:
            best = max(best, len(tokens & q_tokens) * 8)
    return best + int(binding.get("priority") or 0)


def _explicit_role_matches(question: str, date_roles: list[dict]) -> list[dict]:
    q = normalize_date_role_text(question)
    matches: list[tuple[int, int, str, dict]] = []
    for role in date_roles or []:
        if str(role.get("status") or "") != "approved":
            continue
        phrases = _terms([
            role.get("name", ""),
            str(role.get("business_role") or "").replace("_", " "),
            *_terms(role.get("synonyms", [])),
        ])
        role_matches = []
        for phrase in phrases:
            start = q.find(phrase) if phrase else -1
            if start >= 0:
                role_matches.append((start, start + len(phrase), phrase))
        if role_matches:
            start, end, phrase = max(role_matches, key=lambda item: (len(item[2]), -item[0]))
            matches.append((start, end, phrase, role))
    if not matches:
        return []

    # "cancelled order date" also contains "order date". Keep the most
    # specific role for overlapping text, while preserving separate phrases
    # such as "booked date and order date" as two intentional roles.
    selected: list[dict] = []
    for start, end, phrase, role in matches:
        shadowed = any(
            other_start <= start and other_end >= end and len(other_phrase) > len(phrase)
            for other_start, other_end, other_phrase, _other_role in matches
        )
        if not shadowed:
            selected.append({**role, "_matched_phrase": phrase})
    return selected


def _role_as_binding(role: dict, *, source: str) -> dict:
    return {
        "id": 0,
        "metric_id": 0,
        "metric_name": "",
        "context_name": role.get("name") or role.get("business_role") or "Date",
        "aliases": ", ".join(_terms(role.get("synonyms", []))),
        "date_role": role.get("business_role") or "business_date",
        "fact_table": role.get("fact_table") or "",
        "fact_column": role.get("fact_column") or "",
        "dimension_table": role.get("dimension_table") or "",
        "dimension_key": role.get("dimension_key") or "",
        "date_value_column": role.get("date_value_column") or "",
        "date_key_type": normalize_date_key_type(
            role.get("date_key_type") or "surrogate_fk"
        ),
        "is_default": 0,
        "priority": int(role.get("confidence") or 0),
        "resolution_source": source,
    }


def resolve_contextual_date_binding(
    question: str,
    *,
    matched_metrics: list[dict] | None,
    bindings: list[dict] | None,
    date_roles: list[dict] | None,
) -> dict:
    """Resolve a date binding without letting an LLM guess.

    Precedence is explicit role wording, context aliases, then one configured
    default. Ambiguous generic temporal questions are returned to the caller so
    it can ask the user to choose.
    """
    if not question_has_temporal_intent(question):
        return {"status": "none", "reason": "no temporal intent"}

    metrics = matched_metrics or []
    metric_tables = {
        _table_identity(metric.get("base_table"))[0]
        for metric in metrics if metric.get("base_table")
    }
    candidates = list(bindings or [])

    explicit = _explicit_role_matches(question, list(date_roles or []))
    if metric_tables:
        scoped = [
            role for role in explicit
            if any(_same_table(role.get("fact_table"), table) for table in metric_tables)
        ]
        explicit = scoped or explicit
    if len(explicit) == 1:
        return {
            "status": "selected",
            "binding": _role_as_binding(explicit[0], source="explicit_date_role"),
            "reason": "explicit date role in question",
        }
    if len(explicit) > 1:
        distinct_columns = {
            (_table_identity(role.get("fact_table"))[0], str(role.get("fact_column") or "").upper())
            for role in explicit
        }
        distinct_phrases = {
            str(role.get("_matched_phrase") or "") for role in explicit
        }
        if len(distinct_columns) == len(explicit) and len(distinct_phrases) == len(explicit):
            return {
                "status": "selected_many",
                "bindings": [
                    _role_as_binding(role, source="explicit_date_role")
                    for role in explicit
                ],
                "reason": "multiple explicit date roles in question",
            }
        return {
            "status": "ambiguous",
            "reason": "multiple approved date roles match the question",
            "options": [_role_as_binding(role, source="explicit_date_role") for role in explicit],
        }

    scored = [(_binding_score(question, item), item) for item in candidates]
    scored = [(score, item) for score, item in scored if score > int(item.get("priority") or 0)]
    if scored:
        top_score = max(score for score, _item in scored)
        winners = [dict(item) for score, item in scored if score == top_score]
        if len(winners) == 1:
            winners[0]["resolution_source"] = "business_context"
            return {"status": "selected", "binding": winners[0], "reason": "business context match"}
        return {
            "status": "ambiguous",
            "reason": "multiple business date contexts match the question",
            "options": winners,
        }

    defaults = [dict(item) for item in candidates if int(item.get("is_default") or 0)]
    if len(defaults) == 1:
        defaults[0]["resolution_source"] = "metric_default"
        return {"status": "selected", "binding": defaults[0], "reason": "metric default date context"}
    if len(defaults) > 1:
        return {"status": "ambiguous", "reason": "multiple metric defaults", "options": defaults}

    if len(candidates) > 1:
        return {
            "status": "ambiguous",
            "reason": "metric has multiple date contexts and no default",
            "options": [dict(item) for item in candidates],
        }
    if len(candidates) == 1:
        only = dict(candidates[0])
        only["resolution_source"] = "single_metric_context"
        return {"status": "selected", "binding": only, "reason": "only configured date context"}
    return {"status": "none", "reason": "no governed date context"}


def build_contextual_date_plan(binding: dict) -> dict:
    """Compile a selected binding into validator-enforced semantic fields."""
    fact_table = str(binding.get("fact_table") or "")
    fact_column = str(binding.get("fact_column") or "")
    dimension_table = str(binding.get("dimension_table") or "")
    dimension_key = str(binding.get("dimension_key") or "")
    date_value_column = str(binding.get("date_value_column") or "")
    if not all((fact_table, fact_column, dimension_table, dimension_key, date_value_column)):
        return {"enabled": False, "fields": [], "joins": [], "required_tables": [], "reason": "incomplete date context"}

    label = str(binding.get("context_name") or binding.get("date_role") or "Business date")
    date_key_type = normalize_date_key_type(
        binding.get("date_key_type") or "surrogate_fk"
    )
    alias_base = re.sub(r"[^a-z0-9]+", "_", normalize_date_role_text(label)).strip("_")
    role_alias = alias_base or "business_date"
    if not role_alias.endswith("date"):
        role_alias = f"{role_alias}_date"
    return {
        "enabled": True,
        "fields": [
            {
                "term": label,
                "table": dimension_table,
                "column": date_value_column,
                "role": "contextual_date",
                # Unlike an ordinary display dimension, the date value must
                # also appear in non-grouped temporal filters (for example
                # "revenue yesterday"). Do not let aggregate-only queries
                # satisfy the plan with the surrogate-key JOIN alone.
                "display_required": False,
                "source_table": fact_table,
                "source_key_column": fact_column,
                "confidence": 100,
                "source": "approved_metric_date_context",
                "enforcement": "required",
                "date_key_type": date_key_type,
                "role_alias": role_alias,
            }
        ],
        "joins": [
            {
                "from": fact_table,
                "to": dimension_table,
                "conditions": [(fact_column, dimension_key)],
                "source": "approved_metric_date_context",
                "enforcement": "required",
                "role_playing": True,
                "preserve_all": True,
                "role_alias": role_alias,
                "business_role": label,
            }
        ],
        "date_key_policies": [{
            "table": fact_table,
            "column": fact_column,
            "date_key_type": date_key_type,
            "date_value_table": dimension_table,
            "date_value_column": date_value_column,
            "role_alias": role_alias,
            "business_role": label,
        }],
        "required_tables": [fact_table, dimension_table],
        "reason": f"governed date context: {label}",
        "resolved_date_context": dict(binding),
    }


def build_contextual_date_plan_many(bindings: list[dict]) -> dict:
    """Compile multiple explicit role-playing dates into one exact plan."""
    plans = [build_contextual_date_plan(binding) for binding in bindings or []]
    plans = [plan for plan in plans if plan.get("enabled")]
    if not plans:
        return {"enabled": False, "fields": [], "joins": [], "required_tables": []}

    # Alias collisions are possible when admins use similar labels. Make every
    # role alias deterministic and unique without changing physical tables.
    used_aliases: set[str] = set()
    for index, plan in enumerate(plans, start=1):
        edge = plan["joins"][0]
        alias = str(edge.get("role_alias") or f"business_date_{index}")
        base = alias
        suffix = 2
        while alias in used_aliases:
            alias = f"{base}_{suffix}"
            suffix += 1
        used_aliases.add(alias)
        edge["role_alias"] = alias
        plan["fields"][0]["role_alias"] = alias
        plan["date_key_policies"][0]["role_alias"] = alias

    return {
        "enabled": True,
        "fields": [field for plan in plans for field in plan.get("fields", [])],
        "joins": [edge for plan in plans for edge in plan.get("joins", [])],
        "date_key_policies": [
            policy for plan in plans for policy in plan.get("date_key_policies", [])
        ],
        "required_tables": sorted({
            table for plan in plans for table in plan.get("required_tables", []) if table
        }),
        "reason": "governed role-playing date contexts: " + ", ".join(
            str(binding.get("context_name") or binding.get("date_role") or "Business date")
            for binding in bindings
        ),
        "resolved_date_contexts": [dict(binding) for binding in bindings],
    }
