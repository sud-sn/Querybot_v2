"""Gather the drafts for one workspace, stage them, and apply the accepted ones.

``core.model_drafts`` produces proposals from evidence and touches nothing.
This module is the half that talks to the store: it collects the evidence,
stages a draft as a row a human can accept or reject, and — only when somebody
does accept — writes the change.

The split is deliberate and load-bearing. Keeping the drafters pure is what
lets a test assert that no drafter can write, by monkeypatching every store
write path and calling all three. If gathering lived alongside them that test
would have nothing left to say.

Staging reuses ``graph_change_proposal`` rather than adding a fourth proposal
table. It already carries before/payload/confidence/generated_by/reason, is
already reviewed with an accept and a reject, and its accept path already does
the thing that matters most: it re-reads the target and **refuses when the
target changed since the proposal was made**. A proposal is a diff against a
state, and applying one against a different state is applying something nobody
reviewed.

Metric shapes are NOT staged. They are a case for authoring a measure, not a
change to apply — the accept button for one would have to invent SQL, and a
governed measure nobody wrote is worse than a missing one. They surface as a
read-only list beside the backlog.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger("querybot.draft_review")

# How many questions are read for the mined drafters.
MAX_QUESTIONS = 600

# The proposal target_kind these rows carry. graph_change_proposal's accept
# path branches on it, and 'entity'/'relationship' are taken.
TARGET_KIND = "property"

# Which draft kinds can be applied by accepting a row. A kind absent here is
# advisory and is reported without an accept button.
APPLICABLE_KINDS = ("date_role", "column_synonyms")


@dataclass(frozen=True)
class DraftSet:
    date_roles: tuple = ()
    column_vocabulary: tuple = ()
    metric_shapes: tuple = ()
    error: str = ""

    @property
    def applicable(self) -> tuple:
        return tuple(self.date_roles) + tuple(self.column_vocabulary)

    @property
    def total(self) -> int:
        return len(self.applicable) + len(self.metric_shapes)


def _entity_by_table(entities: list[dict]) -> dict[str, str]:
    """Every spelling of a table that might name it, mapped to its entity.

    The value index stores a fully-qualified ``SCHEMA.TABLE``; the schema file
    is keyed by FQN, ``schema.table`` and bare name alike; the graph holds the
    two halves separately. A lookup that knows only one of those spellings
    finds nothing and the whole drafter silently produces an empty report,
    which reads exactly like a workspace with no gaps.
    """
    out: dict[str, str] = {}
    for entity in entities or []:
        name = str(entity.get("entity_name") or "").strip()
        table = str(entity.get("table_name") or "").strip()
        schema = str(entity.get("schema_name") or "").strip()
        if not name or not table:
            continue
        for key in (table, f"{schema}.{table}" if schema else table):
            out.setdefault(key.upper(), name)
    return out


def _recent_questions(account_id: str, limit: int = MAX_QUESTIONS) -> list[str]:
    import store

    try:
        with store.get_db() as conn:
            return [str(row[0]) for row in conn.execute(
                "SELECT question FROM query_log WHERE account_id=? "
                "AND question IS NOT NULL AND question != '' "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (account_id, int(limit)),
            ).fetchall()]
    except Exception as exc:  # noqa: BLE001
        log.warning("draft_review: question log unavailable for %s: %s", account_id, exc)
        return []


def _metric_rows(account_id: str, limit: int = MAX_QUESTIONS) -> list[dict]:
    """query_log rows with whatever says a governed measure answered them.

    ``metric_id`` is the column when the log has one. Selected defensively:
    this table has grown columns over the product's life and a drafter that
    500s on an older deployment is a drafter nobody runs.
    """
    import store

    try:
        with store.get_db() as conn:
            columns = {row[1] for row in conn.execute(
                "PRAGMA table_info(query_log)").fetchall()}
            metric_col = "metric_id" if "metric_id" in columns else "NULL AS metric_id"
            return [dict(r) for r in conn.execute(
                f"SELECT question, {metric_col} FROM query_log WHERE account_id=? "
                "AND question IS NOT NULL AND question != '' "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (account_id, int(limit)),
            ).fetchall()]
    except Exception as exc:  # noqa: BLE001
        log.warning("draft_review: query log unavailable for %s: %s", account_id, exc)
        return []


def gather(account_id: str) -> DraftSet:
    """Every draft for one workspace. Never raises: this is a report."""
    import store

    from core.model_drafts import (
        column_vocabulary_drafts,
        date_role_drafts,
        metric_shape_drafts,
    )

    try:
        entities = store.list_entities(account_id)
        properties = store.list_all_entity_properties(account_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("draft_review: graph unavailable for %s: %s", account_id, exc)
        return DraftSet(error=str(exc)[:200])

    by_table = _entity_by_table(entities)

    # ── columns, for the date-role drafter ────────────────────────────────
    columns: list[dict] = []
    try:
        from core.schema import load_schema_columns
        state = store.get_client_state(account_id) or {}
        schema_columns = load_schema_columns(str(state.get("schema_dir") or ""))
        for entity in entities:
            table = str(entity.get("table_name") or "").strip()
            schema = str(entity.get("schema_name") or "").strip()
            found = (schema_columns.get(f"{schema}.{table}")
                     or schema_columns.get(table) or {})
            for column in found:
                columns.append({"entity": str(entity.get("entity_name") or ""),
                                "column": column})
    except Exception as exc:  # noqa: BLE001
        log.warning("draft_review: schema unavailable for %s: %s", account_id, exc)

    # ── indexed values, for the vocabulary drafter ────────────────────────
    indexed: list[dict] = []
    try:
        from core.value_index import sample_values_by_column
        for row in sample_values_by_column(account_id):
            entity = by_table.get(str(row.get("table_fqn") or "").upper())
            if not entity:
                # A value-indexed table with no entity is not a gap in
                # vocabulary, it is a gap in the graph, and model_readiness
                # already reports that one.
                continue
            indexed.append({"entity": entity, "column": row.get("column"),
                            "values": row.get("values") or []})
    except Exception as exc:  # noqa: BLE001
        log.warning("draft_review: value index unavailable for %s: %s", account_id, exc)

    questions = _recent_questions(account_id)

    return DraftSet(
        date_roles=tuple(date_role_drafts(columns, properties)),
        column_vocabulary=tuple(column_vocabulary_drafts(indexed, questions, properties)),
        metric_shapes=tuple(metric_shape_drafts(_metric_rows(account_id))),
    )


def target_id(draft) -> str:
    return f"{draft.entity}.{draft.column}"


def stage(account_id: str, drafts, *, generated_by: str = "model_drafts") -> dict:
    """Write the applicable drafts as pending proposals. Returns a summary.

    Idempotent by target: a draft whose target already has a pending proposal
    is skipped rather than duplicated, because drafting is a report an admin
    re-runs and a queue that grows a copy per run is a queue nobody finishes.

    A draft whose target was already REJECTED is skipped too. Rejection is an
    answer, and re-proposing it next Tuesday is the review queue arguing with
    the reviewer.
    """
    import store

    existing = {}
    for status in ("pending", "rejected"):
        for row in store.list_graph_change_proposals(account_id, status):
            if str(row.get("target_kind") or "") == TARGET_KIND:
                existing[str(row.get("target_id") or "")] = status

    staged, skipped = [], []
    for draft in drafts or []:
        if draft.kind not in APPLICABLE_KINDS:
            continue
        target = target_id(draft)
        if target in existing:
            skipped.append({"target": target, "because": existing[target]})
            continue
        proposal_id = store.create_graph_change_proposal(
            account_id,
            action=f"set_{draft.kind}",
            target_kind=TARGET_KIND,
            target_id=target,
            before=dict(draft.before or {}),
            payload=dict(draft.payload or {}),
            confidence_score=int(draft.confidence),
            generated_by=generated_by,
            reason=draft.reason,
        )
        existing[target] = "pending"
        staged.append({"id": proposal_id, "target": target, "kind": draft.kind})

    return {"staged": staged, "skipped": skipped,
            "staged_count": len(staged), "skipped_count": len(skipped)}


# ── Applying one ──────────────────────────────────────────────────────────────

# The fields a property proposal is allowed to touch. Anything else on the row
# -- confidence, provenance, who reviewed it -- is set by the accept, not
# carried in from a payload that was assembled by a drafter.
_APPLIABLE_FIELDS = ("role", "display_name", "synonyms")


def property_conflict(current: dict | None, before: dict | None) -> str:
    """Why this proposal must not be applied, or "" when it may be.

    A proposal is a diff against a state. If the row moved after the proposal
    was made -- an admin edited it, another proposal was accepted, a re-scan
    rewrote it -- then applying this one applies something nobody reviewed.

    A row that has since been CONFIRMED is refused whatever it says. That is
    the same rule the drafters follow, enforced a second time here because
    staging and accepting are separated by however long the queue sat.
    """
    if current is None:
        # Nothing there. A proposal that creates the row is fine; one that
        # claimed to change an existing row is not.
        if any(str((before or {}).get(f) or "") for f in _APPLIABLE_FIELDS):
            return "The column's saved meaning has been removed since this was proposed."
        return ""
    if str(current.get("status") or "confirmed") == "confirmed":
        return "This column has been confirmed by hand since this was proposed."
    for field_name in _APPLIABLE_FIELDS:
        if str(current.get(field_name) or "") != str((before or {}).get(field_name) or ""):
            return (f"The column's {field_name.replace('_', ' ')} changed after "
                    f"this was proposed. Review it again.")
    return ""


def split_target(target: str) -> tuple[str, str]:
    """``"Sales.ORDER_DATE"`` → ``("Sales", "ORDER_DATE")``.

    Split on the LAST dot: an entity name may contain one, a column name in
    this product may not.
    """
    text = str(target or "")
    if "." not in text:
        return text, ""
    entity, _, column = text.rpartition(".")
    return entity, column


def apply_property_proposal(account_id: str, proposal: dict) -> tuple[bool, str]:
    """Apply one accepted property proposal. Returns ``(applied, message)``.

    The write is a CONFIRMED row: a human has just looked at the diff and said
    yes, which is exactly what confirmed means. Writing it back as 'suggested'
    would leave it in the queue it was accepted out of.
    """
    import store

    entity, column = split_target(str(proposal.get("target_id") or ""))
    if not entity or not column:
        return False, "This proposal does not name a column."

    current = None
    for row in store.list_entity_properties(account_id, entity):
        if str(row.get("column_name") or "").upper() == column.upper():
            current = dict(row)
            break

    conflict = property_conflict(current, dict(proposal.get("before") or {}))
    if conflict:
        return False, conflict

    payload = dict(proposal.get("payload") or {})
    merged = {f: str(payload.get(f, (current or {}).get(f) or "") or "")
              for f in _APPLIABLE_FIELDS}
    store.save_entity_property(
        account_id=account_id,
        entity_name=entity,
        column_name=column,
        role=merged["role"] or "dimension",
        display_name=merged["display_name"],
        synonyms=merged["synonyms"],
        confidence_score=100,
        status="confirmed",
        generated_by=str(proposal.get("generated_by") or "model_drafts"),
        reason=str(proposal.get("reason") or ""),
    )
    return True, f"{entity}.{column} updated."
