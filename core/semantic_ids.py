"""
Canonical semantic object identifiers.

Every compiler conflict, resolution-plan entry, and audit record must refer
to semantic objects (metrics, terms, entities, fields, joins, date roles) by
a CANONICAL ID, never by display name. Display names are mutable and can
collide — two metrics named "Revenue" with different formulas is itself one
of the conflicts Sprint 2 must detect, so the ID scheme has to keep such
objects distinguishable rather than accidentally merging them.

Design rule: each canonical ID is built from the actual DB uniqueness
invariant for that object type (verified against store/db.py, not assumed),
so the same real-world object mints the same ID on every compile, and two
different objects can never collide onto one ID.

    metric:<metric_registry.id>
        metric_registry has NO UNIQUE constraint on (account_id, name) —
        duplicate names are possible today, which is exactly the conflict
        class this module must keep distinguishable. Only the integer PK
        is guaranteed unique.

    term:<business_term.id>
        business_term has UNIQUE(account_id, term). The PK is still used
        (not the term text) so a rename doesn't orphan open conflicts tied
        to the same row.

    entity:<entity_name>
        entity_graph has UNIQUE(account_id, entity_name), and entity_name
        is already the identity every other table uses to reference an
        entity (entity_relationships.from_entity/to_entity,
        entity_properties.entity_name are all names, not the integer id).
        Matching that existing convention here — minting a PK-based id
        would create a second, inconsistent identity scheme.

    field:<table_fqn>.<column>
        Mirrors entity_properties' UNIQUE(account_id, entity_name,
        column_name) invariant, but keyed by the PHYSICAL table (schema.table
        from entity_graph.schema_name/table_name, or the semantic model's own
        qualified_name) rather than by entity_name. Two different admin
        surfaces describe the same physical column — the semantic model
        (schema-derived, keyed by fqn) and the entity graph
        (entity_properties, keyed by entity_name) — and they must resolve to
        the SAME field id or Sprint 2's "same field, different meaning"
        detectors can never tell they're looking at one column twice. See
        resolve_entity_table_fqn() for the entity_name -> table_fqn bridge.

    join:<entity_relationships.id>
        Graph join edges (the admin-managed relationships that steer SQL
        joins via core/graph_resolver.py). No natural-key alternative is
        reliable: relationship_key exists on the table but is not backed by
        a UNIQUE constraint and is not populated on every historical row.

    date_role:<table_fqn>.<column>
        One role-playing date assignment per physical column, matching how
        core/semantic_model.py's _date_roles() already always carries
        fact_table + fact_column on every entry.

Multi-object conflicts (e.g. "these two metrics collide") build their
conflict_key from a SORTED set of participant canonical ids via
conflict_key(), so discovery order never changes the key across compiles —
required for store.reconcile_semantic_conflicts() to recognize a conflict
as still-open (or resolved) rather than minting a duplicate every run.
"""

from __future__ import annotations

import re


def _norm(value) -> str:
    """Case/whitespace-insensitive normalization for identifier components.
    Table and column casing is inconsistent across this codebase (discovery,
    admin edits, and legacy data disagree) — normalizing here means the same
    physical object always mints the same id regardless of which path wrote
    the casing the caller happens to have on hand."""
    return re.sub(r"\s+", " ", str(value or "").strip()).upper()


def metric_id(metric_pk: int | str) -> str:
    return f"metric:{int(metric_pk)}"


def term_id(term_pk: int | str) -> str:
    return f"term:{int(term_pk)}"


def entity_id(entity_name: str) -> str:
    return f"entity:{_norm(entity_name)}"


def table_id(table_fqn: str) -> str:
    return f"table:{_norm(table_fqn)}"


def field_id(table_fqn: str, column: str) -> str:
    return f"field:{_norm(table_fqn)}.{_norm(column)}"


def join_id(relationship_pk: int | str) -> str:
    return f"join:{int(relationship_pk)}"


def date_role_id(table_fqn: str, column: str) -> str:
    return f"date_role:{_norm(table_fqn)}.{_norm(column)}"


_VALID_TYPES = {"metric", "term", "entity", "table", "field", "join", "date_role"}


def parse_semantic_id(canonical_id: str) -> tuple[str, str]:
    """Split "type:key" into (object_type, key).

    field/date_role keys legitimately contain their own ":"-free dotted
    component (table.column), so this only splits on the FIRST colon.
    Raises ValueError for anything that isn't a recognized, well-formed id —
    callers (conflict detectors, the resolution plan) should never silently
    accept a malformed reference to a semantic object.
    """
    object_type, sep, key = str(canonical_id or "").partition(":")
    if not sep or not key or object_type not in _VALID_TYPES:
        raise ValueError(f"Malformed canonical semantic id: {canonical_id!r}")
    return object_type, key


def resolve_entity_table_fqn(entity: dict) -> str:
    """Physical table fqn for a graph entity row, in the same 2-part
    "SCHEMA.TABLE" shape core/semantic_model.py's _qualified_name() produces
    for schema-derived tables — the shared shape is what lets field_id()
    mint the same id whether the caller started from the entity graph
    (entity_properties, keyed by entity_name) or the semantic model (keyed
    by fqn) for the same physical column."""
    schema = str((entity or {}).get("schema_name") or "").strip()
    table = str((entity or {}).get("table_name") or "").strip()
    return f"{schema}.{table}" if schema else table


def conflict_key(code: str, *object_ids: str) -> str:
    """Deterministic, order-independent conflict_key for one or more
    canonical semantic ids. Sorting the participants means "metric:3 vs
    metric:9" and "metric:9 vs metric:3" (the same underlying pair,
    discovered in a different order on two different compiles) always
    produce the identical key, so store.reconcile_semantic_conflicts()
    correctly matches it to the still-open row instead of opening a
    duplicate every run."""
    parts = sorted(str(oid) for oid in object_ids if oid)
    return f"{code}::" + "|".join(parts)
