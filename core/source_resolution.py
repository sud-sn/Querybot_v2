"""Dataset-neutral business source and fact resolution.

Source vocabulary is derived from each tenant's compiled semantic model and
active terminology pack.  Core runtime code therefore does not need to know
about M3, SAP, EMCO, or any other product/client naming convention.
"""

from __future__ import annotations

import re
from typing import Any


_GENERIC_TABLE_WORDS = {
    "data", "dataset", "table", "fact", "facts", "view", "history",
    "transaction", "transactions", "record", "records",
}
_GENERIC_MEASURE_SUFFIXES = {
    "amount", "amt", "value", "quantity", "qty", "count", "number",
    "total", "metric", "measure",
}
_GRAIN_WORDS = {
    "daily": "day", "day": "day", "monthly": "month", "month": "month",
    "quarterly": "quarter", "quarter": "quarter", "yearly": "year",
    "annual": "year", "year": "year",
}
_ATOMIC_ALIAS_STOP_WORDS = _GENERIC_TABLE_WORDS | {
    "annual", "balance", "daily", "day", "detail", "details", "dimension",
    "event", "fct", "header", "hist", "historical", "line", "lines", "month",
    "monthly", "period", "quarter", "quarterly", "snapshot", "summary", "year",
    "yearly", "one", "row", "rows", "per",
    "business", "code", "id", "identifier", "key", "no", "num", "number",
    "reference", "surrogate",
}
_SURROGATE_FIELD_SUFFIXES = ("_DMS_KEY", "_KEY", "_SK", "_FK")


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _contains_phrase(question: str, phrase: str) -> bool:
    phrase_n = _norm(phrase)
    return bool(phrase_n and re.search(
        rf"(?<![a-z0-9]){re.escape(phrase_n)}(?![a-z0-9])", _norm(question)
    ))


def _table_identity(table: dict[str, Any]) -> str:
    return str(table.get("qualified_name") or table.get("table") or "").upper()


def _table_bare(table_name: str) -> str:
    return str(table_name or "").upper().split(".")[-1]


def _expanded_identifier(value: str, vocab=None) -> str:
    try:
        from core.identifier_intelligence import tokenize_identifier
        pieces = tokenize_identifier(str(value or ""), vocab=vocab)
    except Exception:
        pieces = re.split(r"[_\W]+", str(value or ""))
    abbreviations = getattr(vocab, "abbreviations", {}) if vocab is not None else {}
    return _norm(" ".join(
        str(abbreviations.get(str(piece).upper(), piece)).lower()
        for piece in pieces if piece
    ))


def _measure_aliases(field: dict[str, Any], vocab=None) -> set[str]:
    aliases = {
        _norm(field.get("column") or ""),
        _norm(field.get("expanded_name") or ""),
        _norm(field.get("approved_meaning") or ""),
        _expanded_identifier(field.get("column") or "", vocab=vocab),
    }
    aliases.update(_norm(v) for v in (field.get("business_candidates") or []) if v)
    aliases.discard("")
    derived: set[str] = set()
    for alias in aliases:
        words = alias.split()
        while len(words) > 1 and words[-1] in _GENERIC_MEASURE_SUFFIXES:
            words = words[:-1]
            if words:
                derived.add(" ".join(words))
    return aliases | derived


def _table_aliases(table: dict[str, Any], vocab=None) -> set[str]:
    identity = _table_identity(table)
    bare = _table_bare(identity)
    aliases = {
        _norm(bare), _expanded_identifier(bare, vocab=vocab),
        _norm(table.get("entity") or ""), _norm(table.get("grain") or ""),
        _norm(table.get("fact_type") or ""),
    }
    table_dict = getattr(vocab, "table_dict", {}) if vocab is not None else {}
    entry = table_dict.get(bare, {}) if isinstance(table_dict, dict) else {}
    if isinstance(entry, dict):
        aliases.add(_norm(entry.get("label") or ""))
        aliases.update(_norm(v) for v in (entry.get("synonyms") or []) if v)

    # Some production models have weak/missing table-level entity metadata but
    # strong approved business identifiers (for example "Order Number"). Use
    # those identifiers to recover the event noun for source selection. Foreign
    # and surrogate keys are deliberately excluded: CUSTOMER_SK on an order
    # fact describes a relationship, not the fact's business event.
    for field in table.get("fields", []) or []:
        status = _norm(field.get("status") or field.get("review_status") or "")
        if status and status not in {"approved", "confirmed", "validated"}:
            continue
        column = str(field.get("column") or "").upper()
        if column.endswith(_SURROGATE_FIELD_SUFFIXES):
            continue
        role = _norm(field.get("role") or "")
        aggregation = _norm(field.get("aggregation") or "")
        naming_role = _norm(field.get("naming_role") or "")
        identifier_like = bool(
            role == "identifier"
            or aggregation == "identifier"
            or naming_role in {"business number", "identifier", "code"}
            or column.endswith(("_NUMBER", "_NUM", "_NO"))
        )
        if not identifier_like:
            continue
        values = [
            field.get("business_name"), field.get("display_name"),
            field.get("expanded_name"), field.get("approved_meaning"),
        ]
        raw_candidates = field.get("business_candidates") or []
        values.extend(
            raw_candidates
            if isinstance(raw_candidates, (list, tuple, set))
            else [raw_candidates]
        )
        for value in values:
            normalized = _norm(value or "")
            words = [word for word in normalized.split() if word not in _ATOMIC_ALIAS_STOP_WORDS]
            if words:
                aliases.add(" ".join(words))
    aliases = {a for a in aliases if a and a not in _GENERIC_TABLE_WORDS}
    # Tenant models often describe an event fact as "customer sales order" or
    # expose an ERP name such as CUS_ORD_IVC_FCT, while users simply say
    # "orders". Add meaningful atomic nouns from the tenant's own metadata.
    # Alias frequency below prevents a shared noun from silently selecting one
    # of several facts; shared matches become clarification candidates.
    atomic: set[str] = set()
    for alias in aliases:
        for token in alias.split():
            if len(token) >= 3 and token not in _ATOMIC_ALIAS_STOP_WORDS and not token.isdigit():
                atomic.add(token)
                if token.endswith("y") and len(token) > 3:
                    atomic.add(token[:-1] + "ies")
                elif not token.endswith("s"):
                    atomic.add(token + "s")
    return aliases | atomic


def _requested_grain(question: str) -> str:
    q = _norm(question)
    for word, grain in _GRAIN_WORDS.items():
        if re.search(rf"\b{word}\b", q):
            return grain
    return ""


def _requested_grains(question: str) -> set[str]:
    q = _norm(question)
    return {
        grain for word, grain in _GRAIN_WORDS.items()
        if re.search(rf"\b{word}\b", q)
    }


def _same_table(left: str, right: str) -> bool:
    lp = str(left or "").upper().strip().strip("[]\"`").split(".")
    rp = str(right or "").upper().strip().strip("[]\"`").split(".")
    if len(lp) >= 2 and len(rp) >= 2:
        return lp[-2:] == rp[-2:]
    return bool(lp and rp and lp[-1] == rp[-1])


def _grain_matches(requested: str, table: dict[str, Any]) -> bool:
    if not requested:
        return False
    values = _norm(" ".join([
        str(table.get("grain") or ""), str(table.get("fact_type") or ""),
        _table_bare(_table_identity(table)),
    ]))
    variant = {"day": "daily", "month": "monthly", "quarter": "quarterly", "year": "yearly"}.get(requested, "")
    return requested in values or bool(variant and variant in values)


def _business_source_label(table: dict[str, Any], vocab=None) -> str:
    """Return a user-facing source name without exposing a physical FQN."""
    identity = _table_identity(table)
    bare = _table_bare(identity)
    table_dict = getattr(vocab, "table_dict", {}) if vocab is not None else {}
    entry = table_dict.get(bare, {}) if isinstance(table_dict, dict) else {}
    if isinstance(entry, dict) and str(entry.get("label") or "").strip():
        return str(entry["label"]).strip()

    entity = str(table.get("entity") or "").strip()
    grain_text = _norm(" ".join([
        str(table.get("grain") or ""), str(table.get("fact_type") or ""), bare,
    ]))
    cadence = next(
        (label for token, label in (
            ("daily", "Daily"), (" day", "Daily"),
            ("monthly", "Monthly"), (" month", "Monthly"),
            ("quarterly", "Quarterly"), (" quarter", "Quarterly"),
            ("yearly", "Yearly"), (" annual", "Yearly"),
        ) if token in f" {grain_text}"),
        "",
    )
    if entity:
        label = " ".join(part for part in (cadence, entity) if part)
        return label.strip().title()
    expanded = _expanded_identifier(bare, vocab=vocab)
    return (expanded or bare.replace("_", " ")).strip().title()


def resolve_source_scope(
    question: str,
    model: dict[str, Any] | None,
    *,
    vocab=None,
    selected_schema: str = "",
    authoritative_fact_tables: set[str] | None = None,
) -> dict[str, Any]:
    """Resolve a source fact before resolving fields.

    A fact is selected only with strong, separated evidence. Close candidates
    remain ambiguous; callers must not silently choose one and create a
    multi-fact join.
    """
    selected_schema = str(selected_schema or "").upper().strip()
    tables: list[dict[str, Any]] = []
    for table in (model or {}).get("tables", []) or []:
        if str(table.get("type") or "").lower() != "fact":
            continue
        schema = str(table.get("schema") or "").upper()
        if selected_schema and schema and schema != selected_schema:
            continue
        if _table_identity(table):
            tables.append(table)
    if not tables:
        return {"status": "none", "selected_fact": "", "candidates": [], "reason": "no classified facts"}

    # Shared entity labels (for example two facts both called "inventory
    # snapshot") describe a subject area, not an explicit physical source.
    # Count aliases tenant-locally so only a unique alias can lock a source.
    alias_counts: dict[str, int] = {}
    table_aliases: dict[str, set[str]] = {}
    for table in tables:
        identity = _table_identity(table)
        aliases = _table_aliases(table, vocab=vocab)
        table_aliases[identity] = aliases
        for alias in aliases:
            alias_counts[alias] = alias_counts.get(alias, 0) + 1

    requested_grain = _requested_grain(question)
    requested_grains = _requested_grains(question)
    authoritative_fact_tables = {
        str(value or "").upper() for value in (authoritative_fact_tables or set()) if value
    }
    scored: list[dict[str, Any]] = []
    for table in tables:
        score = 0
        evidence: list[str] = []
        matches = sorted(
            (
                a for a in table_aliases.get(_table_identity(table), set())
                if alias_counts.get(a, 0) == 1 and _contains_phrase(question, a)
            ),
            key=len, reverse=True,
        )
        if matches:
            best = matches[0]
            meaningful = [w for w in best.split() if w not in _GENERIC_TABLE_WORDS]
            if meaningful:
                score += 5 + min(6, len(meaningful) * 2)
                evidence_kind = "source" if len(meaningful) > 1 else "subject"
                evidence.append(f"{evidence_kind}:{best}")
        shared_matches = sorted(
            (
                a for a in table_aliases.get(_table_identity(table), set())
                if alias_counts.get(a, 0) > 1 and _contains_phrase(question, a)
            ),
            key=len,
            reverse=True,
        )
        if shared_matches:
            score += 3
            evidence.append(f"shared_source:{shared_matches[0]}")

        metric_bound = any(
            _same_table(_table_identity(table), source)
            for source in authoritative_fact_tables
        )
        if metric_bound:
            # An approved metric's physical source is stronger evidence than a
            # cadence word.  This prevents "revenue by invoice month" from
            # selecting a monthly inventory fact merely because it says month.
            score += 30
            evidence.append("approved_metric_source")

        measure_hits: set[str] = set()
        for field in table.get("fields", []) or []:
            if str(field.get("role") or "").lower() not in {"measure", "measure_candidate"}:
                continue
            hit = next((a for a in _measure_aliases(field, vocab=vocab) if _contains_phrase(question, a)), "")
            if hit:
                measure_hits.add(hit)
        for measure in table.get("measures", []) or []:
            values = [measure.get("name"), measure.get("column"), *(measure.get("synonyms") or [])]
            hit = next((_norm(v) for v in values if v and _contains_phrase(question, _norm(v))), "")
            if hit:
                measure_hits.add(hit)
        if measure_hits:
            score += 5 + min(4, len(measure_hits))
            evidence.append("measure:" + ",".join(sorted(measure_hits)[:3]))

        for grain in requested_grains or ({requested_grain} if requested_grain else set()):
            if _grain_matches(grain, table):
                score += 5
                evidence.append(f"grain:{grain}")
        if score:
            scored.append({
                "table": _table_identity(table), "score": score,
                "evidence": evidence, "entity": str(table.get("entity") or ""),
                "grain": str(table.get("grain") or ""),
                "fact_type": str(table.get("fact_type") or ""),
                "label": _business_source_label(table, vocab=vocab),
            })

    scored.sort(key=lambda item: (-int(item["score"]), item["table"]))
    if not scored:
        return {"status": "none", "selected_fact": "", "candidates": [], "reason": "no source evidence"}
    compound_language = bool(re.search(r"\b(?:compare|versus|vs\.?|with)\b", _norm(question)))
    if len(requested_grains) > 1 and compound_language:
        selected_by_grain: list[dict[str, Any]] = []
        for grain in sorted(requested_grains):
            matches = [item for item in scored if f"grain:{grain}" in item["evidence"]]
            if matches:
                selected_by_grain.append(matches[0])
        unique = list({item["table"]: item for item in selected_by_grain}.values())
        if len(unique) > 1:
            return {
                "status": "compound",
                "selected_fact": "",
                "selected_facts": [item["table"] for item in unique],
                "candidates": unique,
                "requested_grains": sorted(requested_grains),
                "reason": "multiple requested grains require isolated fact subplans",
            }
    # An explicit source phrase always wins over metric metadata.  Otherwise a
    # single approved metric source is authoritative.  Both rules are derived
    # from the tenant model/registry, never from client-specific names.
    explicit = [item for item in scored if any(e.startswith("source:") for e in item["evidence"])]
    if explicit:
        explicit.sort(key=lambda item: (-int(item["score"]), item["table"]))
        top = explicit[0]
        return {
            "status": "selected", "selected_fact": top["table"],
            "candidates": explicit[:6], "requested_grain": requested_grain,
            "requested_grains": sorted(requested_grains),
            "reason": "explicit tenant source terminology",
        }

    metric_bound = [item for item in scored if "approved_metric_source" in item["evidence"]]
    if len(metric_bound) == 1:
        top = metric_bound[0]
        return {
            "status": "selected", "selected_fact": top["table"],
            "candidates": metric_bound, "requested_grain": requested_grain,
            "requested_grains": sorted(requested_grains),
            "reason": "approved metric source binding",
        }

    top = scored[0]
    runner_score = int(scored[1]["score"]) if len(scored) > 1 else 0
    selected = int(top["score"]) >= 7 and int(top["score"]) - runner_score >= 2
    return {
        "status": "selected" if selected else "ambiguous",
        "selected_fact": top["table"] if selected else "",
        "candidates": scored[:6], "requested_grain": requested_grain,
        "requested_grains": sorted(requested_grains),
        "reason": "tenant semantic model and terminology evidence",
    }


def source_clarification_options(scope: dict[str, Any]) -> list[dict[str, str]]:
    """Build business-facing choices backed by exact tenant table identities."""
    options: list[dict[str, str]] = []
    used_labels: set[str] = set()
    for candidate in (scope or {}).get("candidates", [])[:4]:
        table = str(candidate.get("table") or "").strip()
        label = str(candidate.get("label") or candidate.get("entity") or "").strip()
        if not table or not label:
            continue
        if label.lower() in used_labels:
            cadence = str(candidate.get("grain") or candidate.get("fact_type") or "").strip()
            label = f"{label} ({cadence})" if cadence else f"{label} ({table.split('.')[-1]})"
        used_labels.add(label.lower())
        options.append({"id": f"source_{len(options) + 1}", "label": label, "value": table})
    return options


def format_source_scope(scope: dict[str, Any]) -> str:
    if not scope or scope.get("status") == "none":
        return ""
    selected = str(scope.get("selected_fact") or "")
    if selected:
        return (
            "## Authoritative source scope\n"
            f"Use {selected} as the single measure fact for this request. "
            "Do not substitute or join another fact table. Dimension joins are allowed only through governed relationships."
        )
    selected_facts = [str(value) for value in scope.get("selected_facts") or [] if value]
    if selected_facts:
        return (
            "## Authoritative compound source scope\n"
            "Build one independently aggregated subquery for each fact: "
            + ", ".join(selected_facts)
            + ". Combine only their aggregated outputs through governed shared dimensions; "
              "never join physical fact rows directly."
        )
    candidates = ", ".join(c.get("table", "") for c in (scope.get("candidates") or [])[:3])
    return f"## Unresolved source scope\nDo not guess between these candidate facts: {candidates}."
