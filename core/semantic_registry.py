from __future__ import annotations

from typing import Optional


def normalize_option(option: dict, *, idx: int = 1, assume_legacy_valid: bool = True) -> dict:
    item = dict(option or {})
    item["id"] = str(item.get("id") or f"opt{idx}")
    label = str(item.get("label") or item.get("value") or "").strip()
    value = str(item.get("value") or label).strip()
    item["label"] = label
    item["value"] = value

    valid = item.get("valid")
    if valid is None:
        valid = assume_legacy_valid
    item["valid"] = bool(valid)
    return item


def validated_options(options: list[dict], *, assume_legacy_valid: bool = True) -> list[dict]:
    validated: list[dict] = []
    for idx, option in enumerate(options or [], start=1):
        if not isinstance(option, dict):
            continue
        normalized = normalize_option(
            option,
            idx=idx,
            assume_legacy_valid=assume_legacy_valid,
        )
        if not normalized.get("label") or not normalized.get("valid"):
            continue
        validated.append(normalized)
    return validated


def validated_term_options(term: Optional[dict]) -> list[dict]:
    if not isinstance(term, dict):
        return []
    return validated_options(term.get("clarification_options") or [])


def find_registry_clarification(
    account_id: str,
    question: str,
    *,
    allowed_tables: Optional[set[str]] = None,
) -> Optional[dict]:
    """
    Resolve a clarification prompt only from approved business-semantic options.

    Returns a dict with the matched term and validated options, or None when no
    approved clarification set is available.
    """
    import store

    term = store.find_ambiguous_term(account_id, question, allowed_tables)
    if not term:
        return None

    options = validated_term_options(term)
    if len(options) < 2:
        return None

    return {
        "term": term,
        "options": options,
    }
