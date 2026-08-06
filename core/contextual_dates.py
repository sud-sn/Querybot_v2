"""Resolve governed date roles for metrics and business contexts."""

from __future__ import annotations

import re
from typing import Any

from core.date_roles import (
    date_key_temporal_grain,
    normalize_date_key_type,
    normalize_date_role_text,
    question_has_temporal_intent,
)


_GRAIN_ORDER = {"day": 1, "week": 2, "month": 3, "quarter": 4, "year": 5}


def requested_temporal_grain(question: str) -> str:
    """Return the finest grain explicitly requested by the user."""
    window = detect_temporal_window(question)
    unit = str(window.get("unit") or "").lower()
    if unit in _GRAIN_ORDER:
        return unit
    q = normalize_date_role_text(question)
    for grain, words in (
        ("day", ("daily", "by day", "each day")),
        ("week", ("weekly", "by week", "each week")),
        ("month", ("monthly", "by month", "each month")),
        ("quarter", ("quarterly", "by quarter", "each quarter")),
        ("year", ("yearly", "annual", "by year", "each year")),
    ):
        if any(word in q for word in words):
            return grain
    return ""


def question_has_snapshot_intent(question: str) -> bool:
    """Detect semi-additive stock/balance questions that imply latest data.

    This is intentionally narrow: revenue/orders/sales wording is excluded so
    an operational measure is not silently converted into a snapshot query.
    """
    q = normalize_date_role_text(question)
    if not q or re.search(r"\b(?:revenue|sales|sold|orders?|invoices?)\b", q):
        return False
    return bool(re.search(r"\b(?:inventory|stock|on hand|balance|warehouse balance)\b", q))


def _role_temporal_grain(role: dict) -> str:
    return str(role.get("temporal_grain") or "").lower() or date_key_temporal_grain(
        str(role.get("date_key_type") or "")
    )


def _grain_is_compatible(source_grain: str, requested_grain: str) -> bool:
    if not requested_grain or not source_grain:
        return True
    return _GRAIN_ORDER.get(source_grain, 99) <= _GRAIN_ORDER.get(requested_grain, 0)


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
        # quality 2 = the role/grain was named explicitly ("booked month");
        # quality 1 = event wording only ("ordered revenue", "fill count").
        # Event wording is useful when it is the only date-role signal, but it
        # must not add a second role beside an explicit role merely because a
        # fact/entity noun appears elsewhere in the question.
        role_matches: list[tuple[int, int, str, int]] = []
        for phrase in phrases:
            start = q.find(phrase) if phrase else -1
            if start >= 0:
                role_matches.append((start, start + len(phrase), phrase, 2))
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
                            role_matches.append((start, start + len(variant), variant, 2))
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
                                (event_match.start(), event_match.end(), variant, 1)
                            )
        if role_matches:
            start, end, phrase, quality = max(
                role_matches,
                key=lambda item: (item[3], len(item[2]), -item[0]),
            )
            matches.append((start, end, phrase, quality, role))
    if not matches:
        return []

    # An explicit role/grain outranks incidental event wording globally. Live
    # example: "completed-fill net revenue by booked month" explicitly asks
    # for Booked Date; the noun "fill" describes the metric population and
    # must not also require Fill Date.
    if any(quality == 2 for _start, _end, _phrase, quality, _role in matches):
        matches = [item for item in matches if item[3] == 2]

    # "cancelled order date" also contains "order date". Keep the most
    # specific role for overlapping text, while preserving separate phrases
    # such as "booked date and order date" as two intentional roles.
    selected: list[dict] = []
    for start, end, phrase, _quality, role in matches:
        shadowed = any(
            other_start <= start and other_end >= end and len(other_phrase) > len(phrase)
            for other_start, other_end, other_phrase, _other_quality, _other_role in matches
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
        "temporal_grain": _role_temporal_grain(role),
        "inference_source": role.get("inference_source") or "",
        "inferred_fallback": bool(
            role.get("inference_source")
            and normalize_date_key_type(role.get("date_key_type"))
            in {"yyyymmdd_integer", "yyyymm_integer"}
            and str(role.get("status") or "").casefold() != "approved"
        ),
        "is_default": int(bool(role.get("is_default"))),
        "priority": int(role.get("confidence") or 0),
        "resolution_source": source,
        "governance_status": (
            role.get("_selection_status") or role.get("status") or ""
        ),
    }


def _relevant_discovered_date_roles(
    question: str,
    *,
    matched_metrics: list[dict] | None,
    date_roles: list[dict] | None,
    required_fact_tables: set[str] | None,
) -> list[dict]:
    """Rank physically complete date roles for a scoped clarification.

    This fallback is deliberately unavailable without a resolved metric/fact
    scope.  A production warehouse can contain dozens of business dates; a
    global list would be noisy and could expose dates from an unrelated
    business process.  Within the resolved fact, approved roles rank first,
    followed by the metric's configured time column and discovery confidence.
    """
    metrics = list(matched_metrics or [])
    fact_scope = {
        _table_identity(table)[0]
        for table in (required_fact_tables or set())
        if table
    }
    metric_tables = {
        _table_identity(metric.get("base_table"))[0]
        for metric in metrics
        if metric.get("base_table")
    }
    fact_scope |= metric_tables
    if not fact_scope:
        return []

    configured_time_columns = {
        str(metric.get("default_time_column") or "").strip().strip("[]`").upper().split(".")[-1]
        for metric in metrics
        if metric.get("default_time_column")
    }
    q_tokens = set(normalize_date_role_text(question).split())
    ranked: list[tuple[int, str, str, dict]] = []
    seen: set[tuple[str, str]] = set()
    for role in date_roles or []:
        status = str(role.get("status") or "").casefold()
        if status not in {"approved", "generated", "needs_review"}:
            continue
        if not _role_is_complete(role):
            continue
        role_table = _table_identity(role.get("fact_table"))[0]
        if not any(_same_table(role_table, table) for table in fact_scope):
            continue
        role_column = str(role.get("fact_column") or "").strip().upper()
        identity = (role_table, role_column)
        if identity in seen:
            continue
        seen.add(identity)

        score = max(0, min(100, int(role.get("confidence") or 0)))
        if status == "approved":
            score += 80
        if role_column.split(".")[-1] in configured_time_columns:
            score += 160
        if any(_same_table(role_table, table) for table in metric_tables):
            score += 60
        role_terms = set(
            normalize_date_role_text(
                " ".join([
                    str(role.get("name") or ""),
                    str(role.get("business_role") or "").replace("_", " "),
                    " ".join(_terms(role.get("synonyms", []))),
                ])
            ).split()
        )
        # Generic temporal words do not make every date equally relevant, but
        # an event word such as "invoiced" or "delivered" is useful evidence.
        score += min(30, len((role_terms & q_tokens) - {"date", "day", "today"}) * 10)
        ranked.append((score, role_table, role_column, role))

    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [dict(item[3]) for item in ranked]


def resolve_contextual_date_binding(
    question: str,
    *,
    matched_metrics: list[dict] | None,
    bindings: list[dict] | None,
    date_roles: list[dict] | None,
    required_fact_tables: set[str] | None = None,
    confirmed_date_role: dict | None = None,
    remembered_date_role: dict | None = None,
) -> dict:
    """Resolve a date binding without letting an LLM guess.

    Precedence is explicit role wording, context aliases, then one configured
    default. Ambiguous generic temporal questions are returned to the caller so
    it can ask the user to choose.
    """
    roles = list(date_roles or [])
    explicit = _governed_explicit_role_matches(question, roles)

    # An approved business synonym is temporal intent even when the phrase is
    # abbreviated (for example, "inv dt") and contains none of the generic
    # date words recognized by question_has_temporal_intent().
    if (
        not question_has_temporal_intent(question)
        and not question_has_snapshot_intent(question)
        and not explicit
    ):
        return {"status": "none", "reason": "no temporal intent"}

    requested_grain = requested_temporal_grain(question)

    # A clarification option is created by the server from the semantic
    # contract and copied onto the retry event by the dispatcher.  Preserve
    # that exact physical identity instead of trying to rediscover it from the
    # option label (which can collide across facts).
    if confirmed_date_role and _role_is_complete(confirmed_date_role):
        if confirmed_date_role.get("context_name"):
            selected = dict(confirmed_date_role)
            selected["resolution_source"] = "user_confirmed_date_role"
        else:
            selected = _role_as_binding(
                confirmed_date_role,
                source="user_confirmed_date_role",
            )
        return {
            "status": "selected",
            "binding": selected,
            "reason": "date role confirmed by the user",
        }

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

    if fact_scope:
        scoped = [
            role for role in explicit
            if any(_same_table(role.get("fact_table"), table) for table in fact_scope)
        ]
        explicit = scoped or explicit
    if len(explicit) == 1:
        explicit_grain = _role_temporal_grain(explicit[0])
        if not _grain_is_compatible(explicit_grain, requested_grain):
            return {
                "status": "unsupported_grain",
                "reason": "the selected business date is coarser than the requested period",
                "requested_grain": requested_grain,
                "available_grain": explicit_grain,
                "options": [_role_as_binding(explicit[0], source="explicit_date_role")],
            }
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

    # Ana-style thread continuity: a date explicitly confirmed by the user on
    # an earlier turn may be reused for the same metric/fact. This sits below
    # current-turn wording and governed defaults, and above another discovery
    # clarification. It is a session assumption, never an ontology approval.
    if remembered_date_role and _role_is_complete(remembered_date_role):
        remembered = dict(remembered_date_role)
        if fact_scope and not any(
            _same_table(remembered.get("fact_table"), table)
            for table in fact_scope
        ):
            remembered = {}
        if remembered:
            remembered["resolution_source"] = "thread_date_preference"
            return {
                "status": "selected",
                "binding": remembered,
                "reason": "date role previously confirmed in this thread",
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

    discovered = _relevant_discovered_date_roles(
        question,
        matched_metrics=metrics,
        date_roles=roles,
        required_fact_tables=required_fact_tables,
    )
    if discovered:
        if question_has_snapshot_intent(question) and not requested_grain:
            known_grains = [
                _role_temporal_grain(role)
                for role in discovered
                if _role_temporal_grain(role)
            ]
            if known_grains:
                finest = min(known_grains, key=lambda grain: _GRAIN_ORDER.get(grain, 99))
                discovered = [
                    role for role in discovered
                    if _role_temporal_grain(role) == finest
                ]
        compatible = [
            role for role in discovered
            if _grain_is_compatible(_role_temporal_grain(role), requested_grain)
        ]
        if compatible:
            discovered = compatible
        elif requested_grain:
            return {
                "status": "unsupported_grain",
                "reason": "the resolved fact has no date at the requested grain",
                "requested_grain": requested_grain,
                "available_grain": min(
                    (_role_temporal_grain(role) for role in discovered if _role_temporal_grain(role)),
                    key=lambda grain: _GRAIN_ORDER.get(grain, 99),
                    default="",
                ),
                "options": [
                    _role_as_binding(role, source="discovered_date_role")
                    for role in discovered[:4]
                ],
            }
        approved = [
            role for role in discovered
            if str(role.get("status") or "").casefold() == "approved"
        ]
        # One approved role on the resolved fact is governed and unambiguous;
        # it is safe to use even if an administrator did not explicitly mark
        # it as the default. Multiple approved roles still need a choice.
        if len(discovered) == 1 and len(approved) == 1:
            return {
                "status": "selected",
                "binding": _role_as_binding(
                    approved[0], source="single_approved_date_role"
                ),
                "reason": "only approved date role for the resolved fact",
            }
        # Deterministic direct integer encodings are safe scoped fallbacks:
        # the resolved fact owns the field, the physical representation is
        # known, and there is only one compatible candidate. This is not an
        # approval and is disclosed as inference in the returned provenance.
        if len(discovered) == 1:
            only = discovered[0]
            key_type = normalize_date_key_type(only.get("date_key_type"))
            if (
                str(only.get("status") or "").casefold() == "generated"
                and int(only.get("confidence") or 0) >= 95
                and key_type in {"yyyymmdd_integer", "yyyymm_integer"}
                and only.get("inference_source")
            ):
                binding = _role_as_binding(only, source="inferred_encoded_fact_date")
                binding["inferred_fallback"] = True
                return {
                    "status": "selected",
                    "binding": binding,
                    "reason": "only deterministic encoded date on the resolved fact",
                }
        all_options = [
            _role_as_binding(role, source="discovered_date_role")
            for role in discovered
        ]
        return {
            "status": "ambiguous",
            "reason": (
                "the resolved fact has business dates but no unambiguous "
                "approved/default date role"
            ),
            # Four choices remain scannable on mobile. The complete scoped set
            # is retained for matching a date typed into the custom field.
            "options": all_options[:4],
            "all_options": all_options,
            "allow_free_text": True,
        }
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
        r"\b(?:last|past|previous|latest)\s+(\d+)\s+(day|week|month|quarter|year)s?\b",
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
    temporal_grain = str(binding.get("temporal_grain") or "").lower() or date_key_temporal_grain(
        date_key_type
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
                "temporal_grain": temporal_grain,
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
            "temporal_grain": temporal_grain,
            "inference_source": binding.get("inference_source") or "",
        }],
        "required_tables": [table for table in (fact_table, dimension_table) if table],
        "reason": f"governed date context: {label}",
        "resolved_date_context": dict(binding),
        "date_disclosures": [{
            "label": label,
            "table": fact_table,
            "column": fact_column,
            "date_key_type": date_key_type,
            "temporal_grain": temporal_grain,
            "resolution_source": binding.get("resolution_source") or "",
            "inference_source": binding.get("inference_source") or "",
            "inferred": bool(binding.get("inferred_fallback")),
        }],
    }
    window = detect_temporal_window(question)
    if not window and question_has_snapshot_intent(question):
        window = {
            "kind": "latest_snapshot",
            "amount": 1,
            "unit": temporal_grain or "period",
            "anchor_policy": "latest_available",
            "implicit": True,
        }
    if window:
        plan["temporal_policies"] = [{
            **window,
            "fact_table": fact_table,
            "fact_column": fact_column,
            "date_table": field_table,
            "date_column": date_value_column,
            "dimension_table": dimension_table,
            "dimension_key": dimension_key,
            # The anchor for a relative period must always be derived from
            # the governed FACT rows, never the raw dimension -- date_table/
            # date_column above point at the dimension for surrogate_fk
            # roles (needed for the filter clause, which needs a real
            # calendar value), but anchor_table/anchor_column always point
            # at the fact table itself. format_required_anchor() is the
            # single place that turns these into the anchor subquery text;
            # nothing else should build that string by hand.
            "anchor_table": fact_table,
            "anchor_column": fact_column,
            "role_alias": role_alias,
            "date_key_type": date_key_type,
            "business_role": label,
            "governance_status": binding.get("governance_status") or "",
            "resolution_source": binding.get("resolution_source") or "",
            "temporal_grain": temporal_grain,
            "inference_source": binding.get("inference_source") or "",
        }]
    return plan


def format_date_value_expression(
    table: str,
    column: str,
    date_key_type: str,
    db_type: str = "azure_sql",
) -> str:
    """Render one governed physical date as a nullable calendar expression."""
    ref = f"{table}.{column}" if table else column
    key_type = normalize_date_key_type(date_key_type)
    dialect = str(db_type or "azure_sql").lower()
    if key_type == "yyyymmdd_integer":
        if dialect in {"azure_sql", "sqlserver", "mssql"}:
            return f"TRY_CONVERT(date, CONVERT(varchar(8), {ref}), 112)"
        if dialect == "snowflake":
            return f"TRY_TO_DATE(TO_VARCHAR({ref}), 'YYYYMMDD')"
        if dialect in {"postgres", "postgresql"}:
            return f"TO_DATE(CAST({ref} AS text), 'YYYYMMDD')"
        if dialect in {"mysql", "mariadb"}:
            return f"STR_TO_DATE(CAST({ref} AS CHAR), '%Y%m%d')"
    if key_type == "yyyymm_integer":
        if dialect in {"azure_sql", "sqlserver", "mssql"}:
            return f"TRY_CONVERT(date, CONVERT(varchar(6), {ref}) + '01', 112)"
        if dialect == "snowflake":
            return f"TRY_TO_DATE(TO_VARCHAR({ref}), 'YYYYMM')"
        if dialect in {"postgres", "postgresql"}:
            return f"TO_DATE(CAST({ref} AS text), 'YYYYMM')"
        if dialect in {"mysql", "mariadb"}:
            return f"STR_TO_DATE(CAST({ref} AS CHAR), '%Y%m')"
    return ref


def format_required_anchor(policy: dict, db_type: str = "azure_sql") -> str:
    """
    Build the "copy this exact subquery" anchor text for a temporal policy.

    Native date (date_key_type != surrogate_fk): MAX over the fact table's
    own column -- mirrors core/report_engine.py::_apply_latest_date_filter's
    already-correct pattern.

    Surrogate FK (a date stored as an ID joining to a date dimension): MAX
    must still be scoped to fact rows that actually exist, not the raw
    dimension -- an unrestricted dimension table commonly carries future
    calendar rows with no matching fact data, which silently anchors to a
    date with zero rows. JOINing to the fact table before taking MAX keeps
    the result a real calendar value while excluding rows with no match.
    """
    fact_table = str(policy.get("anchor_table") or policy.get("fact_table") or "")
    fact_column = str(policy.get("anchor_column") or policy.get("fact_column") or "")
    if str(policy.get("date_key_type") or "") != "surrogate_fk":
        date_table = str(policy.get("date_table") or fact_table)
        date_column = str(policy.get("date_column") or fact_column)
        date_expression = format_date_value_expression(
            "", date_column, str(policy.get("date_key_type") or ""), db_type
        )
        return f"(SELECT MAX({date_expression}) FROM {date_table})"

    dimension_table = str(policy.get("dimension_table") or policy.get("date_table") or "")
    dimension_key = str(policy.get("dimension_key") or "")
    date_value_column = str(policy.get("date_column") or "")
    if not (fact_table and fact_column and dimension_table and dimension_key and date_value_column):
        # Incomplete policy -- fall back to the plain (still fact-scoped
        # where possible) form rather than raising; callers already treat a
        # missing/blank anchor as "nothing to render."
        date_table = dimension_table or fact_table
        date_column = date_value_column or fact_column
        return f"(SELECT MAX({date_column}) FROM {date_table})" if date_table and date_column else ""

    return (
        f"(SELECT MAX({dimension_table}.{date_value_column}) FROM {dimension_table} "
        f"JOIN {fact_table} ON {fact_table}.{fact_column} = {dimension_table}.{dimension_key})"
    )


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
        "date_disclosures": [
            item for plan in plans for item in plan.get("date_disclosures", [])
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
