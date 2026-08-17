"""Canonical identities for governed entity-graph relationships.

Relationship keys preserve independent business roles. Physical signatures are
stricter: they identify only edges with the same role-aware endpoints and the
same column pairs. SQL equality is symmetric, so a legacy edge recorded in the
opposite direction still receives the same signature. This lets schema sync
retire stale inferred duplicates without collapsing legitimate role-playing or
composite joins.
"""

from __future__ import annotations

import json


def _norm(value: object) -> str:
    return str(value or "").strip().upper()


def relationship_column_pairs(relationship: dict) -> tuple[tuple[str, str], ...]:
    """Return the primary and composite join pairs in constraint order."""
    pairs: list[tuple[str, str]] = [
        (
            _norm(relationship.get("from_column")),
            _norm(relationship.get("to_column")),
        )
    ]
    raw_extra = relationship.get("join_conditions") or []
    if isinstance(raw_extra, str):
        try:
            raw_extra = json.loads(raw_extra)
        except (TypeError, ValueError, json.JSONDecodeError):
            raw_extra = []
    for condition in raw_extra:
        if not isinstance(condition, dict):
            continue
        pair = (
            _norm(condition.get("from_col")),
            _norm(condition.get("to_col")),
        )
        if pair != ("", ""):
            pairs.append(pair)
    return tuple(pair for pair in pairs if pair != ("", ""))


def physical_relationship_signature(relationship: dict) -> tuple:
    """Identity for an exact physical join, including role-playing endpoints."""
    source = _norm(relationship.get("from_entity"))
    target = _norm(relationship.get("to_entity"))
    pairs = tuple(sorted(relationship_column_pairs(relationship)))
    forward = (source, target, pairs)
    reverse = (
        target,
        source,
        tuple(sorted((right, left) for left, right in pairs)),
    )
    return min(forward, reverse)
