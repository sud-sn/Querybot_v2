"""
core/join_coverage.py

Lossy-join caveat: when a live question's JOIN path uses an entity-graph
relationship that admin-time validation already found excludes a meaningful
fraction of rows (no matching dimension row), tell the user instead of
silently understating the total. The second "silent gap" scenario after
core/date_coverage.py's date-range coverage check -- same shape, different
cause: the answer looks complete but something got quietly excluded.

Unlike date coverage, this needs NO new database query at question time: the
relationship's orphan_rate is already persisted on the entity_relationships
row whenever an admin runs "Validate all" with the live-probe option
(core/relationship_validator.py), and the exact relationship row(s) a
question's join path used are already resolved by
core.graph_resolver.resolve_for_question (its "resolved_edges"/"edge_ids"
output) -- this module only has to read what already exists.

Known availability constraint, not a bug: orphan_rate is only ever populated
by the live-probe validation path (execute=True). The default schema-only
check never sets it (stays at the sentinel -1.0, never persisted). So this
caveat stays silent until an admin has run that probe at least once for a
given relationship -- a real limitation of what data exists to check
against, not a gap in this module's logic.
"""

from __future__ import annotations

import logging

log = logging.getLogger("querybot.join_coverage")

# Reuses the exact threshold core/relationship_validator.py's own admin-facing
# "warning" status already uses (line ~303: `orphan_rate > 5.0`) -- the same
# number the Entity Graph review UI already considers questionable, not a new
# one invented for this caveat.
_ORPHAN_RATE_WARNING_THRESHOLD = 5.0


def check_join_coverage(account_id: str, graph_edges: list[dict]) -> list[str]:
    """Build a caveat message for each already-resolved relationship edge
    whose persisted orphan_rate is above the warning threshold. Never raises:
    a missing relationship row, a malformed edge dict, or a store lookup
    failure just skips that edge silently.
    """
    if not graph_edges:
        return []

    import store

    messages: list[str] = []
    for edge in graph_edges:
        try:
            rel_id = int((edge or {}).get("id") or 0)
        except (TypeError, ValueError):
            continue
        if not rel_id:
            continue
        try:
            rel = store.get_relationship(account_id, rel_id)
        except Exception as exc:
            log.debug("Join-coverage lookup skipped for relationship %s: %s", rel_id, exc)
            continue
        if not rel:
            continue

        try:
            orphan_rate = float(rel.get("orphan_rate"))
        except (TypeError, ValueError):
            continue
        if orphan_rate < 0 or orphan_rate <= _ORPHAN_RATE_WARNING_THRESHOLD:
            continue

        from_entity = str(edge.get("from_entity") or rel.get("from_entity") or "the source table")
        to_entity = str(edge.get("to_entity") or rel.get("to_entity") or "the joined table")
        messages.append(
            f"The join from {from_entity} to {to_entity} excludes about "
            f"{orphan_rate:.0f}% of rows with no match — some data may not be counted."
        )

    return messages
