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
    return {a for a in aliases if a and a not in _GENERIC_TABLE_WORDS}


def _requested_grain(question: str) -> str:
    q = _norm(question)
    for word, grain in _GRAIN_WORDS.items():
        if re.search(rf"\b{word}\b", q):
            return grain
    return ""


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

    requested_grain = _requested_grain(question)
    scored: list[dict[str, Any]] = []
    for table in tables:
        score = 0
        evidence: list[str] = []
        matches = sorted(
            (a for a in _table_aliases(table, vocab=vocab) if _contains_phrase(question, a)),
            key=len, reverse=True,
        )
        if matches:
            best = matches[0]
            meaningful = [w for w in best.split() if w not in _GENERIC_TABLE_WORDS]
            if meaningful:
                score += 4 + min(6, len(meaningful) * 2)
                evidence.append(f"source:{best}")

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

        if _grain_matches(requested_grain, table):
            score += 5
            evidence.append(f"grain:{requested_grain}")
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
    top = scored[0]
    runner_score = int(scored[1]["score"]) if len(scored) > 1 else 0
    selected = int(top["score"]) >= 7 and int(top["score"]) - runner_score >= 2
    return {
        "status": "selected" if selected else "ambiguous",
        "selected_fact": top["table"] if selected else "",
        "candidates": scored[:6], "requested_grain": requested_grain,
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
    candidates = ", ".join(c.get("table", "") for c in (scope.get("candidates") or [])[:3])
    return f"## Unresolved source scope\nDo not guess between these candidate facts: {candidates}."
