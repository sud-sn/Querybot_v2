"""Shared identifier understanding for KB build and runtime retrieval.

The physical identifiers exposed by ERP/CRM/ETL systems are often not natural
language: ``ORD_DT``, ``CustAccount``, ``ORDERDATE`` and ``TRANDATE`` can all
represent business concepts a user phrases differently.  This module provides
one deterministic normalizer for both sides of the system:

* schema/K​​B build: physical identifier -> evidence-backed meaning + aliases;
* question retrieval: user shorthand -> the same governed terminology.

The implementation deliberately abstains when a compact token cannot be
expanded from an approved/built-in vocabulary.  An unknown code must remain a
review item rather than become a confident invented definition.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable


_GENERIC_WORDS: dict[str, str] = {
    "ACCOUNT": "account", "AMOUNT": "amount", "BOOKED": "booked",
    "CALENDAR": "calendar", "CANCELLED": "cancelled", "CODE": "code",
    "CONFIRMED": "confirmed", "COST": "cost", "CREATED": "created",
    "CREATION": "creation", "CUSTOMER": "customer", "DATE": "date",
    "DELIVERY": "delivery", "DESCRIPTION": "description", "DISCOUNT": "discount",
    "DUE": "due", "EMPLOYEE": "employee", "ENTITY": "entity", "EXTERNAL": "external",
    "FLAG": "flag", "GROSS": "gross", "GROUP": "group", "IDENTIFIER": "identifier",
    "INTERNAL": "internal", "INVOICE": "invoice", "ITEM": "item", "KEY": "key",
    "LINE": "line", "LOCATION": "location", "MODIFIED": "modified", "MONTH": "month",
    "NAME": "name", "NET": "net", "NUMBER": "number", "ORDER": "order",
    "PAYMENT": "payment", "PERCENT": "percent", "PLANNED": "planned",
    "POSTING": "posting", "PRICE": "price", "PRODUCT": "product",
    "PURCHASE": "purchase", "QUANTITY": "quantity", "RATE": "rate",
    "RECEIPT": "receipt", "REQUESTED": "requested", "REVENUE": "revenue",
    "SALES": "sales", "SHIP": "ship", "SHIPPING": "shipping", "STATUS": "status",
    "SUPPLIER": "supplier", "TIMESTAMP": "timestamp", "TOTAL": "total",
    "TRANSACTION": "transaction", "TYPE": "type", "UPDATED": "updated",
    "VALUE": "value", "VENDOR": "vendor", "WAREHOUSE": "warehouse", "YEAR": "year",
}

_TECHNICAL_ALIAS_WORDS = {
    "dimension", "fact", "foreign key", "key", "surrogate key",
}


@dataclass(frozen=True)
class IdentifierAnalysis:
    physical_name: str
    style: str
    tokens: tuple[str, ...]
    expanded_tokens: tuple[str, ...]
    expanded_name: str
    aliases: tuple[str, ...]
    confidence: int
    evidence: tuple[str, ...]
    unresolved_tokens: tuple[str, ...]


def _active_vocab(vocab=None):
    if vocab is not None:
        return vocab
    from core.vocab_packs import get_active_vocab
    return get_active_vocab()


def _clean(value: str) -> str:
    return str(value or "").strip().strip('"`[]')


def resolve_governed_column_code(identifier: str, vocab=None) -> tuple[str, str, str]:
    """Return ``(governed_code, record_prefix, canonical_table)``.

    Exact physical codes always win. A record prefix is removed only when the
    active terminology pack explicitly owns that two-character prefix *and*
    the remaining suffix is an exact governed dictionary code. This makes M3
    fields such as ``OBORNO`` understandable without guessing at arbitrary
    compact identifiers in other ERP/CRM datasets.
    """
    raw = _clean(identifier)
    compact = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    v = _active_vocab(vocab)
    if compact in (v.column_dict or {}):
        return compact, "", ""
    prefixes = getattr(v, "record_prefixes", {}) or {}
    if len(compact) > 2 and compact[:2] in prefixes:
        suffix = compact[2:]
        if suffix in (v.column_dict or {}):
            return suffix, compact[:2], str(prefixes[compact[:2]] or "")
    return compact, "", ""


def _dedupe(values: Iterable[str], *, limit: int = 24) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = re.sub(r"\s+", " ", str(value or "").strip().lower())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
        if len(result) >= limit:
            break
    return tuple(result)


def detect_identifier_style(identifier: str) -> str:
    raw = _clean(identifier)
    if not raw:
        return "unknown"
    if re.search(r"[-.\s]", raw):
        return "delimited"
    if "_" in raw:
        if raw.upper() == raw:
            return "upper_snake"
        if raw.lower() == raw:
            return "lower_snake"
        return "mixed_snake"
    if re.search(r"[a-z][A-Z]", raw):
        return "camel_case" if raw[:1].islower() else "pascal_case"
    if raw.isupper() and re.fullmatch(r"[A-Z0-9]+", raw):
        return "upper_compact" if len(raw) > 8 else "upper_abbreviated"
    if raw.islower() and re.fullmatch(r"[a-z0-9]+", raw):
        return "lower_compact"
    return "mixed"


def _split_case_and_boundaries(raw: str) -> list[str]:
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", raw)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])([0-9])", r"\1 \2", text)
    text = re.sub(r"([0-9])([A-Za-z])", r"\1 \2", text)
    return [piece.upper() for piece in re.split(r"[^A-Za-z0-9]+", text) if piece]


def _expansion_lexicon(vocab) -> dict[str, str]:
    lexicon = dict(_GENERIC_WORDS)
    lexicon.update({str(k).upper(): str(v).lower() for k, v in (vocab.abbreviations or {}).items()})
    return lexicon


def _segment_compact(token: str, lexicon: dict[str, str]) -> list[str]:
    """Split an all-caps compact token only when every piece is governed.

    Dynamic programming favours longer vocabulary entries.  Returning the raw
    token is the abstention path; unknown fragments are never guessed.
    """
    token = token.upper()
    if len(token) < 4 or not token.isalnum():
        return [token]
    keys = [key for key in lexicon if 2 <= len(key) < len(token) and key.isalnum()]
    if not keys:
        return [token]
    memo: dict[int, tuple[int, list[str]] | None] = {}

    def solve(index: int) -> tuple[int, list[str]] | None:
        if index == len(token):
            return 0, []
        if index in memo:
            return memo[index]
        best: tuple[int, list[str]] | None = None
        for key in keys:
            if not token.startswith(key, index):
                continue
            tail = solve(index + len(key))
            if tail is None:
                continue
            # Longer governed pieces are stronger than many tiny pieces.
            candidate = (tail[0] + len(key) * len(key), [key, *tail[1]])
            if best is None or candidate[0] > best[0] or (
                candidate[0] == best[0] and len(candidate[1]) < len(best[1])
            ):
                best = candidate
        memo[index] = best
        return best

    result = solve(0)
    if result and len(result[1]) >= 2:
        return result[1]
    return [token]


def tokenize_identifier(identifier: str, vocab=None) -> list[str]:
    raw = _clean(identifier)
    if not raw:
        return []
    v = _active_vocab(vocab)
    lexicon = _expansion_lexicon(v)
    initial = _split_case_and_boundaries(raw)
    result: list[str] = []
    for token in initial:
        if token in lexicon or len(initial) > 1:
            result.append(token)
        else:
            result.extend(_segment_compact(token, lexicon))
    return result


def canonical_identifier(identifier: str, vocab=None) -> str:
    return "_".join(tokenize_identifier(identifier, vocab=vocab))


def analyze_identifier(identifier: str, vocab=None) -> IdentifierAnalysis:
    raw = _clean(identifier)
    v = _active_vocab(vocab)
    style = detect_identifier_style(raw)
    compact = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    governed_code, record_prefix, record_table = resolve_governed_column_code(raw, vocab=v)
    dictionary_entry = (v.column_dict or {}).get(governed_code)
    tokens = tokenize_identifier(raw, vocab=v)
    lexicon = _expansion_lexicon(v)
    evidence: list[str] = [f"identifier style={style}"]
    unresolved: list[str] = []
    expanded_tokens: list[str] = []

    if dictionary_entry:
        label, dictionary_synonyms = dictionary_entry
        expanded_name = str(label).strip().lower()
        if record_prefix:
            evidence.append(
                f"terminology-pack record prefix {record_prefix}={record_table or 'known ERP file'}"
            )
            evidence.append(f"governed field code={governed_code}")
        else:
            evidence.append("terminology-pack exact column match")
        # Token expansions still feed physical shorthand variants, but the
        # governed dictionary label remains the authoritative meaning.
        for token in tokens or [governed_code]:
            expanded_tokens.append(lexicon.get(token, token.lower()))
        confidence = 95
    else:
        dictionary_synonyms = []
        expanded_count = 0
        for token in tokens:
            expansion = lexicon.get(token)
            if expansion:
                expanded_tokens.append(expansion)
                if expansion != token.lower():
                    expanded_count += 1
                    evidence.append(f"abbreviation {token}={expansion}")
            else:
                expanded_tokens.append(token.lower())
                # Full readable words are useful evidence even when no pack is
                # needed; compact codes are not.
                if token not in _GENERIC_WORDS and len(token) <= 6:
                    unresolved.append(token)
        expanded_name = " ".join(expanded_tokens).strip()
        if expanded_count and not unresolved:
            confidence = 82
        elif expanded_count:
            confidence = 68
        elif len(tokens) >= 2 and style in {"upper_compact", "lower_compact", "upper_abbreviated"} and all(
            token in lexicon for token in tokens
        ):
            confidence = 76
            evidence.append("governed compact-token segmentation")
        elif style in {"camel_case", "pascal_case", "upper_snake", "lower_snake", "mixed_snake"} and all(
            len(token) > 3 or token in _GENERIC_WORDS for token in tokens
        ):
            confidence = 72
            evidence.append("readable identifier tokens")
        else:
            confidence = 35
            if compact:
                evidence.append("no governed expansion found")

    raw_phrase = " ".join(token.lower() for token in tokens)
    alias_values: list[str] = [raw_phrase, expanded_name, *dictionary_synonyms]
    if dictionary_entry:
        # M3 labels often qualify the business term in parentheses, for
        # example "Ordered Quantity (Alternate Unit)". Users normally ask for
        # "ordered quantity"; retain both forms without reducing the alias to
        # a dangerously broad bare word such as "quantity".
        unqualified_label = re.sub(r"\s*\([^)]*\)\s*", " ", str(dictionary_entry[0])).strip()
        if unqualified_label:
            alias_values.append(unqualified_label)
    if tokens and len(tokens) <= 5:
        for index, token in enumerate(tokens):
            expansion = lexicon.get(token, token.lower())
            if expansion == token.lower():
                continue
            mixed = [part.lower() for part in tokens]
            mixed[index] = expansion
            alias_values.append(" ".join(mixed))
    # Avoid emitting a bare structural word as a user-facing alias. Exact
    # physical phrases remain available for technical questions.
    aliases = tuple(
        alias for alias in _dedupe(alias_values)
        if alias not in _TECHNICAL_ALIAS_WORDS or alias == raw_phrase
    )
    return IdentifierAnalysis(
        physical_name=raw,
        style=style,
        tokens=tuple(tokens),
        expanded_tokens=tuple(expanded_tokens),
        expanded_name=expanded_name or raw.lower(),
        aliases=aliases,
        confidence=confidence,
        evidence=tuple(_dedupe(evidence)),
        unresolved_tokens=tuple(_dedupe(unresolved)),
    )


def _pack_match_profile(pack_id: str, pack: dict[str, Any], columns: list[str], tables: list[str], ownership: dict[str, int]) -> dict[str, Any]:
    column_dict = {str(value).upper() for value in (pack.get("column_dict") or {})}
    table_dict = {str(value).upper() for value in (pack.get("table_dict") or {})}
    col_values = {re.sub(r"[^A-Za-z0-9]", "", value).upper() for value in columns if value}
    table_values = {re.sub(r"[^A-Za-z0-9]", "", value.split(".")[-1]).upper() for value in tables if value}
    col_pattern_values = {re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper() for value in columns if value}
    table_pattern_values = {re.sub(r"[^A-Za-z0-9]+", "_", value.split(".")[-1]).strip("_").upper() for value in tables if value}
    exact_columns = sorted(col_values & column_dict)
    exact_tables = sorted(table_values & table_dict)
    unique_columns = [value for value in exact_columns if ownership.get("C:" + value, 0) == 1]
    unique_tables = [value for value in exact_tables if ownership.get("T:" + value, 0) == 1]

    record_prefixes = {
        str(value).upper() for value in (pack.get("record_prefixes") or {})
        if len(str(value).strip()) == 2
    }
    prefixed_columns = sorted({
        value for value in col_values
        if len(value) > 2 and value[:2] in record_prefixes and value[2:] in column_dict
    })

    date_matches: list[str] = []
    for item in pack.get("date_role_patterns") or []:
        try:
            pattern = re.compile(str(item.get("pattern") or ""))
        except re.error:
            continue
        date_matches.extend(value for value in col_pattern_values if pattern.search(value))

    table_pattern_matches: list[str] = []
    classification = pack.get("table_classification") or {}
    for raw_pattern in (
        list(classification.get("fact_patterns") or [])
        + list(classification.get("dimension_patterns") or [])
        + list(classification.get("bridge_patterns") or [])
    ):
        try:
            pattern = re.compile(str(raw_pattern))
        except re.error:
            continue
        table_pattern_matches.extend(value for value in table_pattern_values if pattern.search(value))

    unique_evidence = len(unique_columns) + len(unique_tables) + len(prefixed_columns)
    score = (
        len(unique_columns) * 5
        + (len(exact_columns) - len(unique_columns))
        + len(unique_tables) * 10
        + (len(exact_tables) - len(unique_tables)) * 2
        + len(prefixed_columns) * 5
        + len(set(date_matches)) * 2
        + len(set(table_pattern_matches)) * 2
    )
    confidence = round((1.0 - math.exp(-score / 14.0)) * 100, 1) if score else 0.0
    return {
        "pack_id": pack_id,
        "erp_name": str(pack.get("erp_name") or pack_id),
        "confidence": confidence,
        "score": score,
        "unique_evidence": unique_evidence,
        "exact_column_matches": exact_columns[:20],
        "exact_table_matches": exact_tables[:12],
        "record_prefixed_column_matches": prefixed_columns[:20],
        "date_pattern_matches": sorted(set(date_matches))[:12],
        "table_pattern_matches": sorted(set(table_pattern_matches))[:12],
    }


def detect_naming_profile(column_names: Iterable[str], table_names: Iterable[str] = ()) -> dict[str, Any]:
    """Classify schema naming and recommend a terminology pack.

    Only a high-confidence, uniquely supported winner is marked for automatic
    application. Everything else remains a reviewable recommendation.
    """
    columns = [str(value) for value in column_names if str(value).strip()]
    tables = [str(value) for value in table_names if str(value).strip()]
    style_counts: dict[str, int] = {}
    for value in [*columns, *[table.split(".")[-1] for table in tables]]:
        style = detect_identifier_style(value)
        style_counts[style] = style_counts.get(style, 0) + 1
    total = sum(style_counts.values())
    ordered_styles = sorted(style_counts.items(), key=lambda item: (-item[1], item[0]))
    primary_style = ordered_styles[0][0] if ordered_styles else "unknown"
    style_confidence = round(ordered_styles[0][1] * 100.0 / total, 1) if ordered_styles and total else 0.0
    if len(ordered_styles) > 1 and style_confidence < 65:
        primary_style = "mixed"

    from core.vocab_packs import list_available_packs, load_pack

    complete: list[tuple[str, dict[str, Any]]] = []
    for manifest in list_available_packs():
        if str(manifest.get("status") or "complete") == "stub":
            continue
        pack_id = str(manifest.get("pack_id") or "")
        pack = load_pack(pack_id)
        if pack:
            complete.append((pack_id, pack))
    ownership: dict[str, int] = {}
    for _, pack in complete:
        for value in pack.get("column_dict") or {}:
            key = "C:" + str(value).upper()
            ownership[key] = ownership.get(key, 0) + 1
        for value in pack.get("table_dict") or {}:
            key = "T:" + str(value).upper()
            ownership[key] = ownership.get(key, 0) + 1
    recommendations = [
        _pack_match_profile(pack_id, pack, columns, tables, ownership)
        for pack_id, pack in complete
    ]
    recommendations.sort(key=lambda item: (-item["score"], item["pack_id"]))
    recommendations = [item for item in recommendations if item["score"] > 0]

    auto_applied: list[str] = []
    if recommendations:
        top = recommendations[0]
        runner_up = recommendations[1]["confidence"] if len(recommendations) > 1 else 0.0
        if (
            top["confidence"] >= 82.0
            and top["unique_evidence"] >= 4
            and top["confidence"] - runner_up >= 10.0
        ):
            auto_applied.append(top["pack_id"])

    from core.vocab_packs import _clone_builtin, _merge_pack
    base_vocab = _clone_builtin()
    for pack_id in auto_applied:
        pack = load_pack(pack_id)
        if pack:
            _merge_pack(base_vocab, pack, f"auto-detected:{pack_id}")
    analyses = [analyze_identifier(value, vocab=base_vocab) for value in columns]
    covered = sum(item.confidence >= 70 for item in analyses)
    unresolved = [
        {
            "column": item.physical_name,
            "confidence": item.confidence,
            "tokens": list(item.tokens),
            "unresolved_tokens": list(item.unresolved_tokens),
        }
        for item in analyses if item.confidence < 70
    ]
    return {
        "version": 1,
        "primary_style": primary_style,
        "style_confidence": style_confidence,
        "style_counts": dict(ordered_styles),
        "field_count": len(columns),
        "vocabulary_coverage_pct": round(covered * 100.0 / len(columns), 1) if columns else 0.0,
        "unresolved_count": len(unresolved),
        "unresolved_fields": unresolved[:50],
        "pack_recommendations": recommendations[:5],
        "auto_applied_packs": auto_applied,
    }


def expand_question_for_retrieval(question: str, vocab=None, *, max_aliases: int = 12) -> str:
    """Append governed terminology expansions without rewriting user intent."""
    original = str(question or "").strip()
    if not original:
        return original
    v = _active_vocab(vocab)
    abbreviations = {
        str(key).lower(): str(value).strip().lower()
        for key, value in (v.planner_abbreviations or {}).items()
        if str(value).strip()
    }
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]*", original)
    aliases: list[str] = []
    expanded_words: list[str] = []
    expanded_flags: list[bool] = []
    changed = False
    for word in words:
        if "_" in word or re.search(r"[a-z][A-Z]", word):
            analysis = analyze_identifier(word, vocab=v)
            aliases.extend(analysis.aliases)
        normalized = word.lower()
        expansion = abbreviations.get(normalized)
        if expansion and expansion != normalized:
            expanded_words.append(expansion)
            expanded_flags.append(True)
            changed = True
        else:
            expanded_words.append(normalized)
            expanded_flags.append(False)
    if changed:
        aliases.append(" ".join(expanded_words))
        for index in range(len(expanded_words) - 1):
            if not (expanded_flags[index] and expanded_flags[index + 1]):
                continue
            phrase = f"{expanded_words[index]} {expanded_words[index + 1]}"
            if phrase not in original.lower():
                aliases.append(phrase)
    governed = _dedupe(aliases, limit=max_aliases)
    if not governed:
        return original
    return original + "\nGoverned terminology aliases: " + "; ".join(governed)
