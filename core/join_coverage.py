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
from datetime import datetime

from core.i18n import plural as _plural, t as _t

log = logging.getLogger("querybot.join_coverage")

# Reuses the exact threshold core/relationship_validator.py's own admin-facing
# "warning" status already uses (line ~303: `orphan_rate > 5.0`) -- the same
# number the Entity Graph review UI already considers questionable, not a new
# one invented for this caveat.
_ORPHAN_RATE_WARNING_THRESHOLD = 5.0

# Past this, the measured percentage is reported as historical rather than as
# a present-tense fact about the join. A month of loads is enough for an
# orphan rate to have moved; quoting a precise figure from before them states
# more than the evidence supports.
_STALE_PROFILE_DAYS = 30


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

        from_entity = str(edge.get("from_entity") or rel.get("from_entity")
                          or _t("caveat.join.default_source"))
        to_entity = str(edge.get("to_entity") or rel.get("to_entity")
                        or _t("caveat.join.default_target"))
        messages.append(_coverage_message(
            from_entity, to_entity, orphan_rate,
            str(rel.get("last_profiled_at") or ""),
        ))

    return messages


def _measured_age_days(last_profiled_at: str) -> int | None:
    """Days since the orphan rate was measured, or None if never/unreadable."""
    raw = str(last_profiled_at or "").strip().replace("Z", "")
    if not raw:
        return None
    for parse in (
        lambda t: datetime.fromisoformat(t),
        lambda t: datetime.strptime(t[:19], "%Y-%m-%d %H:%M:%S"),
        lambda t: datetime.strptime(t[:10], "%Y-%m-%d"),
    ):
        try:
            return max(0, (datetime.now() - parse(raw)).days)
        except (ValueError, TypeError):
            continue
    return None


def _coverage_message(
    from_entity: str, to_entity: str, orphan_rate: float, last_profiled_at: str,
) -> str:
    """State the exclusion, and state when it was measured.

    The percentage comes from a profiling run at some past admin click. It was
    quoted to the user as a present-tense fact — "excludes about 23% of rows" —
    with no age check and no expiry, so a figure measured before a year of
    loads described a join it may no longer describe at all. The number is
    still the best evidence available; what was missing is that it is evidence
    from a point in time.
    """
    # Three whole sentences rather than a shared "The join from X to Y" stem
    # plus a clause: the stem is the subject of three different verbs in three
    # different tenses, and French does not let a subject phrase and its verb
    # be assembled from separate strings the way "{lead} excluded ..." does.
    age_days = _measured_age_days(last_profiled_at)
    fields = {
        "source": from_entity, "target": to_entity, "rate": f"{orphan_rate:.0f}",
    }
    if age_days is None:
        return _t("caveat.join.undated", **fields)
    when = (_t("caveat.join.today") if age_days == 0
            else _plural("caveat.join.days_ago", age_days))
    if age_days > _STALE_PROFILE_DAYS:
        return _t("caveat.join.stale", when=when, **fields)
    return _t("caveat.join.measured", when=when, **fields)
