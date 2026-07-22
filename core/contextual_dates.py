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


def _role_is_complete(role: dict) -> bool:
    """Return whether a discovered date role is safe enough to compile."""
    if not role.get("fact_table") or not role.get("fact_column"):
        return False
    key_type = normalize_date_key_type(role.get("date_key_type") or "surrogate_fk")
    if key_type != "surrogate_fk":
        return True
    return all(
        role.get(key)
        for key in ("dimension_table", "dimension_key", "date_value_column")
    )


def _explicit_role_matches(
    question: str,
    date_roles: list[dict],
    *,
    statuses: set[str] | None = None,
    minimum_confidence: int = 0,
    require_complete: bool = False,
) -> list[dict]:
    q = normalize_date_role_text(question)
    allowed_statuses = {item.casefold() for item in (statuses or {"approved"})}
    matches: list[tuple[int, int, str, dict]] = []
    for role in date_roles or []:
        if str(role.get("status") or "").casefold() not in allowed_statuses:
            continue
        if int(role.get("confidence") or 0) < minimum_confidence:
            continue
        if require_complete and not _role_is_complete(role):
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
                continue

            # Business users commonly replace the word "date" with the
            # requested grain: "booked month", "invoice year", "ship week".
            # Treat those as explicit references to Booked Date, Invoice Date,
            # and Ship Date rather than falling through to a fact default.
            tokens = phrase.split()
            if len(tokens) >= 2 and tokens[-1] in {
                "date", "day", "month", "week", "quarter", "year",
            }:
                stem = " ".join(tokens[:-1]).strip()
                if stem:
                    for grain in ("date", "day", "month", "week", "quarter", "year"):
                        variant = f"{stem} {grain}"
                        start = q.find(variant)
                        if start >= 0:
                            role_matches.append((start, start + len(variant), variant))
                    # Event wording often names the business date implicitly:
                    # "ordered revenue", "booked sales", "invoiced amount".
                    # Keep this word-boundary based so "order" does not match
                    # unrelated text such as "reorder".
                    event_variants = {stem}
                    if " " not in stem:
                        event_variants.update({f"{stem}ed", f"{stem}d", f"{stem}ing"})
                    for variant in sorted(event_variants, key=len, reverse=True):
                        event_match = re.search(rf"\b{re.escape(variant)}\b", q)
                        if event_match:
                            role_matches.append(
                                (event_match.start(), event_match.end(), variant)
                            )
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


def _governed_explicit_role_matches(
    question: str, date_roles: list[dict]
) -> list[dict]:
    """Resolve explicit roles with approval-first governance.

    A generated role is only a fallback when it is physically complete,
    high-confidence, and explicitly named by the user. It is never used as a
    generic/default date and can never override an approved role.
    """
    approved = _explicit_role_matches(
        question,
        date_roles,
        statuses={"approved"},
    )
    if approved:
        return [{**role, "_selection_status": "approved"} for role in approved]
    generated = _explicit_role_matches(
        question,
        date_roles,
        statuses={"generated"},
        minimum_confidence=95,
        require_complete=True,
    )
    return [{**role, "_selection_status": "generated"} for role in generated]


def find_explicit_date_roles(question: str, date_roles: list[dict] | None) -> list[dict]:
    """Return governed roles explicitly named by the user.

    Approved roles always win. A complete generated role with at least 95%
    confidence may be used as an exact-name fallback so a discovered surrogate
    relationship is not silently bypassed before an admin reviews it.
    """
    return _governed_explicit_role_matches(question, list(date_roles or []))


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
        "is_default": int(bool(role.get("is_default"))),
        "priority": int(role.get("confidence") or 0),
        "resolution_source": source,
        "governance_status": (
            role.get("_selection_status") or role.get("status") or ""
        ),
    }


def resolve_contextual_date_binding(
    question: str,
    *,
    matched_metrics: list[dict] | None,
    bindings: list[dict] | None,
    date_roles: list[dict] | None,
    required_fact_tables: set[str] | None = None,
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
    fact_scope = {
        _table_identity(table)[0]
        for table in (required_fact_tables or set()) if table
    } | metric_tables
    candidates = list(bindings or [])

    explicit = _governed_explicit_role_matches(question, list(date_roles or []))
    if fact_scope:
        scoped = [
            role for role in explicit
            if any(_same_table(role.get("fact_table"), table) for table in fact_scope)
        ]
        explicit = scoped or explicit
    if len(explicit) == 1:
        explicit_source = (
            "explicit_date_role"
            if explicit[0].get("_selection_status") == "approved"
            else "explicit_generated_date_role"
        )
        return {
            "status": "selected",
            "binding": _role_as_binding(explicit[0], source=explicit_source),
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
                    _role_as_binding(
                        role,
                        source=(
                            "explicit_date_role"
                            if role.get("_selection_status") == "approved"
                            else "explicit_generated_date_role"
                        ),
                    )
                    for role in explicit
                ],
                "reason": "multiple explicit date roles in question",
            }
        return {
            "status": "ambiguous",
            "reason": "multiple governed date roles match the question",
            "options": [
                _role_as_binding(
                    role,
                    source=(
                        "explicit_date_role"
                        if role.get("_selection_status") == "approved"
                        else "explicit_generated_date_role"
                    ),
                )
                for role in explicit
            ],
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

    # Fact defaults are intentionally considered only after explicit role and
    # metric-context resolution. Scope them to the facts already implied by
    # the question/metric so a default on Inventory cannot contaminate a Sales
    # query merely because both facts exist in the same schema.
    approved_roles = [
        role for role in (date_roles or [])
        if str(role.get("status") or "") == "approved"
        and bool(role.get("is_default"))
    ]
    if required_fact_tables is not None:
        approved_roles = [
            role for role in approved_roles
            if any(_same_table(role.get("fact_table"), table) for table in fact_scope)
        ]
    if len(approved_roles) == 1:
        return {
            "status": "selected",
            "binding": _role_as_binding(
                approved_roles[0], source="fact_default_date_role"
            ),
            "reason": "default date role for resolved fact",
        }
    if len(approved_roles) > 1:
        return {
            "status": "ambiguous",
            "reason": "multiple resolved facts have default date roles",
            "options": [
                _role_as_binding(role, source="fact_default_date_role")
                for role in approved_roles
            ],
        }

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


def detect_temporal_window(question: str) -> dict:
    """Detect relative calendar wording that must use a data-relative anchor."""
    q = normalize_date_role_text(question)
    patterns = (
        (r"\btoday\b", "today", 0, "day"),
        (r"\byesterday\b", "yesterday", 1, "day"),
        (r"\b(?:this|current)\s+week\b", "this_week", 0, "week"),
        (r"\b(?:this|current)\s+month\b", "this_month", 0, "month"),
        (r"\b(?:this|current)\s+quarter\b", "this_quarter", 0, "quarter"),
        (r"\b(?:this|current)\s+year\b", "this_year", 0, "year"),
        (r"\b(?:previous|prior|last)\s+month\b", "previous_month", 1, "month"),
        (r"\b(?:previous|prior|last)\s+quarter\b", "previous_quarter", 1, "quarter"),
        (r"\b(?:previous|prior|last)\s+year\b", "previous_year", 1, "year"),
    )
    for pattern, kind, amount, unit in patterns:
        if re.search(pattern, q):
            return {
                "kind": kind,
                "amount": amount,
                "unit": unit,
                "anchor_policy": "latest_available",
            }
    rolling = re.search(
        r"\b(?:last|past|previous)\s+(\d+)\s+(day|week|month|quarter|year)s?\b",
        q,
    )
    if rolling:
        return {
            "kind": "last_n",
            "amount": int(rolling.group(1)),
            "unit": rolling.group(2),
            "anchor_policy": "latest_available",
        }
    return {}


def build_contextual_date_plan(binding: dict, question: str = "") -> dict:
    """Compile a selected binding into validator-enforced semantic fields."""
    fact_table = str(binding.get("fact_table") or "")
    fact_column = str(binding.get("fact_column") or "")
    dimension_table = str(binding.get("dimension_table") or "")
    dimension_key = str(binding.get("dimension_key") or "")
    date_value_column = str(binding.get("date_value_column") or "")
    date_key_type = normalize_date_key_type(
        binding.get("date_key_type") or "surrogate_fk"
    )
    if not fact_table or not fact_column:
        return {"enabled": False, "fields": [], "joins": [], "required_tables": [], "reason": "incomplete date context"}
    if date_key_type == "surrogate_fk" and not all(
        (dimension_table, dimension_key, date_value_column)
    ):
        return {"enabled": False, "fields": [], "joins": [], "required_tables": [], "reason": "incomplete surrogate date context"}
    if date_key_type != "surrogate_fk":
        dimension_table = ""
        dimension_key = ""
        date_value_column = fact_column

    label = str(binding.get("context_name") or binding.get("date_role") or "Business date")
    alias_base = re.sub(r"[^a-z0-9]+", "_", normalize_date_role_text(label)).strip("_")
    role_alias = alias_base or "business_date"
    if not role_alias.endswith("date"):
        role_alias = f"{role_alias}_date"
    field_table = dimension_table or fact_table
    joins = []
    if dimension_table:
        joins.append({
            "from": fact_table,
            "to": dimension_table,
            "conditions": [(fact_column, dimension_key)],
            "source": "approved_metric_date_context",
            "enforcement": "required",
            "role_playing": True,
            "preserve_all": True,
            "role_alias": role_alias,
            "business_role": label,
        })
    plan = {
        "enabled": True,
        "fields": [
            {
                "term": label,
                "table": field_table,
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
        "joins": joins,
        "date_key_policies": [{
            "table": fact_table,
            "column": fact_column,
            "date_key_type": date_key_type,
            "date_value_table": field_table,
            "date_value_column": date_value_column,
            "role_alias": role_alias,
            "business_role": label,
            "governance_status": binding.get("governance_status") or "",
            "resolution_source": binding.get("resolution_source") or "",
        }],
        "required_tables": [table for table in (fact_table, dimension_table) if table],
        "reason": f"governed date context: {label}",
        "resolved_date_context": dict(binding),
    }
    window = detect_temporal_window(question)
    if window:
        plan["temporal_policies"] = [{
            **window,
            "fact_table": fact_table,
            "fact_column": fact_column,
            "date_table": field_table,
            "date_column": date_value_column,
            "dimension_table": dimension_table,
            "dimension_key": dimension_key,
            "role_alias": role_alias,
            "date_key_type": date_key_type,
            "business_role": label,
            "governance_status": binding.get("governance_status") or "",
            "resolution_source": binding.get("resolution_source") or "",
        }]
    return plan


def build_contextual_date_plan_many(bindings: list[dict], question: str = "") -> dict:
    """Compile multiple explicit role-playing dates into one exact plan."""
    plans = [build_contextual_date_plan(binding, question) for binding in bindings or []]
    plans = [plan for plan in plans if plan.get("enabled")]
    if not plans:
        return {"enabled": False, "fields": [], "joins": [], "required_tables": []}

    # Alias collisions are possible when admins use similar labels. Make every
    # role alias deterministic and unique without changing physical tables.
    used_aliases: set[str] = set()
    for index, plan in enumerate(plans, start=1):
        edge = (plan.get("joins") or [{}])[0]
        alias = str(
            edge.get("role_alias")
            or plan["fields"][0].get("role_alias")
            or f"business_date_{index}"
        )
        base = alias
        suffix = 2
        while alias in used_aliases:
            alias = f"{base}_{suffix}"
            suffix += 1
        used_aliases.add(alias)
        if plan.get("joins"):
            plan["joins"][0]["role_alias"] = alias
        plan["fields"][0]["role_alias"] = alias
        plan["date_key_policies"][0]["role_alias"] = alias
        for policy in plan.get("temporal_policies") or []:
            policy["role_alias"] = alias

    return {
        "enabled": True,
        "fields": [field for plan in plans for field in plan.get("fields", [])],
        "joins": [edge for plan in plans for edge in plan.get("joins", [])],
        "date_key_policies": [
            policy for plan in plans for policy in plan.get("date_key_policies", [])
        ],
        "temporal_policies": [
            policy for plan in plans for policy in plan.get("temporal_policies", [])
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
