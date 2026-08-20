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
# "_DMS_KEY" is not listed: it is Infor M3's spelling of "_KEY", which is
# already here, so naming it bought nothing except the impression that this
# tuple has to learn every ERP's dimension-key convention.
_SURROGATE_SUFFIXES = ("_KEY", "_SK", "_FK")
# Words that qualify an identifier without changing which entity it identifies.
_KEY_WORDS = _IDENTIFIER_WORDS | {"fk", "key", "pk", "sk", "surrogate"}
_MASTER_TABLE_TYPES = {"dimension", "master", "reference", "lookup"}
# Preference among identifiers that all name the same master-table population.
# Every one of them counts the same rows, so this only has to be stable.
_IDENTIFIER_WORD_RANK = {
    "no": 0, "number": 0, "id": 1, "identifier": 1, "reference": 1,
    "code": 2, "key": 3, "sk": 4, "fk": 4, "pk": 4,
}


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


def _table_owns_entity(table: dict[str, Any], entity: str) -> bool:
    """Return whether the fact itself represents the requested event.

    Related facts routinely carry another event's surrogate key (for example,
    a shipment fact carrying ``ORDER_SK``).  Those keys are valid join paths,
    but they are not evidence that the related fact owns the event being
    counted.  Restrict ownership evidence to table identity metadata and the
    physical table name; descriptions and grain prose often mention related
    entities and would recreate the same false positive.
    """
    forms = _entity_forms(entity)
    if not forms:
        return False

    physical = _table_name(table).split(".")[-1]
    identity_values = [
        physical,
        table.get("entity"),
        table.get("business_entity"),
        table.get("business_name"),
        table.get("display_name"),
        table.get("subject"),
        table.get("subject_area"),
    ]
    identity_text = _norm(" ".join(str(value or "") for value in identity_values))
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])", identity_text)
        for form in forms
    )


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
    if surrogate and not _table_owns_entity(table, entity):
        # A foreign surrogate on a related fact is a join key, not the
        # governed identifier of the event being counted.  Fail closed rather
        # than silently switching the source fact and its Date Role.
        return None
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


def _singular_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "sses", "xes", "zes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _entity_tokens(value: Any) -> tuple[str, ...]:
    return tuple(_singular_token(token) for token in _norm(value).split() if token)


def _identifies_exactly(field: dict[str, Any], entity_tokens: tuple[str, ...]) -> str:
    """Return the identifier word by which a field names exactly this entity.

    A master table's own key reads as the entity plus an identifier word --
    "customer number", "supplier id".  A neighbouring master table's key reads
    as the entity plus a qualifier -- "customer type code" identifies a type of
    customer, not a customer.  Requiring the remainder to be exactly the entity
    is what keeps a population count off the qualifier's table; anything less
    exact resolves to no candidate and leaves the question as it was.
    """
    if not entity_tokens:
        return ""
    labels: list[Any] = [
        field.get("expanded_name"), field.get("business_name"),
        field.get("display_name"), field.get("column"),
    ]
    # An approved synonym is the admin saying in so many words what this field
    # identifies, which is stronger evidence than any expansion of the physical
    # name -- and often the only evidence, since abbreviation packs do not
    # cover every tenant's shorthand (SUP_NO expands to "sup no", not
    # "supplier no", and would otherwise never match "supplier").
    for key in ("business_candidates", "synonyms", "approved_synonyms", "aliases"):
        raw = field.get(key) or []
        labels.extend(raw if isinstance(raw, (list, tuple, set)) else [raw])
    suffix_word = _column_identifier_word(field.get("column"))
    for label in labels:
        tokens = _entity_tokens(label)
        identifier_word = ""
        while tokens and tokens[-1] in _KEY_WORDS:
            identifier_word = identifier_word or tokens[-1]
            tokens = tokens[:-1]
        if tokens != entity_tokens:
            continue
        # A label may name the entity without repeating the identifier word --
        # the alias "supplier" on SUP_NO. The physical suffix then says which
        # kind of identifier it is. A label with neither is a plain attribute.
        if identifier_word or suffix_word:
            return identifier_word or suffix_word
    return ""


_COLUMN_IDENTIFIER_SUFFIXES = (
    ("_NUMBER", "number"), ("_NUM", "number"), ("_NO", "no"),
    ("_IDENTIFIER", "identifier"), ("_ID", "id"),
    ("_CODE", "code"), ("_CD", "code"),
    ("_KEY", "key"), ("_SK", "sk"), ("_PK", "pk"),
)


def _column_identifier_word(column: Any) -> str:
    upper = str(column or "").strip().upper()
    for suffix, word in _COLUMN_IDENTIFIER_SUFFIXES:
        if upper.endswith(suffix) and len(upper) > len(suffix):
            return word
    return ""


def _population_candidate(
    table: dict[str, Any],
    field: dict[str, Any],
    entity_tokens: tuple[str, ...],
) -> dict[str, Any] | None:
    column = str(field.get("column") or "").strip()
    if not column:
        return None
    identifier_word = _identifies_exactly(field, entity_tokens)
    if not identifier_word:
        return None
    if _norm(field.get("role")) in {"measure", "measure candidate"}:
        return None

    entity = " ".join(entity_tokens)
    evidence = [f"master table identifier names exactly one {entity}"]
    score = 65
    if _norm(field.get("role")) in {"identifier", "dimension key", "key", "primary key"}:
        score += 10
        evidence.append("semantic role is identifier")
    if _norm(field.get("aggregation")) == "identifier":
        score += 6
        evidence.append("aggregation policy is identifier")
    if str(field.get("status") or "").casefold() == "approved":
        score += 15
        evidence.append("admin approved")
    surrogate = column.upper().endswith(_SURROGATE_SUFFIXES)
    if surrogate:
        # A version key counts rows, not members: on a slowly changing master
        # the business identifier is the one that survives a re-versioned row.
        score -= 20
        evidence.append("surrogate-key penalty")
    return {
        "table": _table_name(table),
        "column": column,
        "business_name": _business_label(field, entity),
        "business_meaning": _business_meaning(field, table, line_level=False),
        "score": score,
        "confidence": max(0, min(100, score)),
        "line_level": False,
        "surrogate": surrogate,
        "source_kind": "master",
        "identifier_rank": _IDENTIFIER_WORD_RANK.get(identifier_word, 5),
        "status": str(field.get("status") or "generated"),
        "evidence": evidence,
    }


def resolve_population_count_target(
    entity: str,
    model: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve "how many <entity> are there" against the entity's master table.

    Deliberately separate from the business-event resolver: a population is
    counted from the table that defines it, so a member with no activity is
    still a member.  Counting it from a fact answers a different question --
    how many members have activity on that fact.

    Every candidate returned identifies one row of the same master table, so
    there is nothing for a user to disambiguate; when nothing matches exactly
    this returns ``missing`` and the caller leaves the question alone.
    """
    entity_tokens = _entity_tokens(entity)
    if not entity_tokens:
        return {"status": "not_applicable", "entity": "", "candidates": []}

    candidates = [
        candidate
        for table in ((model or {}).get("tables") or [])
        if isinstance(table, dict)
        and str(table.get("type") or "").casefold() in _MASTER_TABLE_TYPES
        and _table_name(table)
        for field in (table.get("fields") or [])
        if isinstance(field, dict)
        for candidate in [_population_candidate(table, field, entity_tokens)]
        if candidate is not None
    ]
    entity_text = " ".join(entity_tokens)
    if not candidates:
        return {
            "status": "missing", "entity": entity_text, "selected": {}, "candidates": [],
            "reason": "no master table defines this population",
        }

    tables = {_table_name_key(candidate["table"]) for candidate in candidates}
    if len(tables) > 1:
        # Two master tables both claim to define the population.  Counting
        # either would be a guess about which is authoritative.
        return {
            "status": "ambiguous", "entity": entity_text, "selected": {},
            "candidates": sorted(candidates, key=_population_order)[:6],
            "reason": f"{len(tables)} master tables define this population",
        }

    candidates.sort(key=_population_order)
    return {
        "status": "selected", "entity": entity_text, "selected": candidates[0],
        "candidates": candidates[:6],
        "reason": "master table identifier for the whole population",
    }


def _table_name_key(table: Any) -> str:
    return str(table or "").strip().strip("[]\"`").upper().split(".")[-1]


def _population_order(candidate: dict[str, Any]) -> tuple:
    return (
        -int(candidate["score"]),
        int(candidate.get("identifier_rank") or 5),
        str(candidate["column"]).upper(),
    )


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
