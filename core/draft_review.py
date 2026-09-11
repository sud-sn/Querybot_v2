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
    warning = publish_column_terms(account_id, entity, column, merged["synonyms"])
    if merged["role"] == "date":
        date_warning = publish_date_role(
            account_id, entity, column, merged["display_name"],
            merged["synonyms"])
        warning = " ".join(part for part in (warning, date_warning) if part)
    return True, f"{entity}.{column} updated." + (f" {warning}" if warning else "")


def publish_date_role(account_id: str, entity: str, column: str,
                      display_name: str, synonyms: str) -> str:
    """Put an accepted date role where the RESOLVER reads it.

    Same shape as publish_column_terms, and the same reason. Accepting a
    date-role draft wrote entity_properties with role='date' and stopped there.
    core.contextual_dates.resolve_contextual_date_binding does not read
    entity_properties: it reads model.date_roles[].status == 'approved' out of
    _semantic_model.json, which is what the Date Roles admin screen patches.

    So the two surfaces disagreed. An admin who approved the role on the Date
    Roles screen got a governed default; an admin who accepted the identical
    proposal out of the drafts queue got a confirmed column property, no
    approved Date Role, and a measure that carried on asking "which date should
    I use?" on every period question. The queue said the work was done.

    Returns "" on success, or a sentence for the caller to show. It is not
    silent on failure: a draft accepted into nothing is worse than one that was
    never offered, because the admin has no reason to look again.
    """
    import store

    from core.semantic_model import patch_date_role

    try:
        state = store.get_client_state(account_id) or {}
        kb_dir = str(state.get("kb_dir") or "")
        if not kb_dir:
            return ("The date role was not approved in the semantic model: "
                    "this workspace has no KB directory configured.")

        # entity_properties is keyed on the ENTITY name; date roles are keyed on
        # the fact table. Resolve one to the other rather than assuming they
        # are spelled the same, because on most workspaces they are not.
        fact_table = ""
        for row in store.list_entities(account_id):
            if str(row.get("entity_name") or "").upper() != entity.upper():
                continue
            schema = str(row.get("schema_name") or "").strip()
            table = str(row.get("table_name") or "").strip()
            fact_table = f"{schema}.{table}" if schema else table
            break
        if not fact_table:
            return (f"The date role was not approved: {entity} does not map to "
                    "a table in the graph.")

        patched = patch_date_role(
            kb_dir=kb_dir,
            fact_table=fact_table,
            fact_column=column,
            name=display_name,
            synonyms=[value.strip() for value in str(synonyms or "").split(",")
                      if value.strip()],
            status="approved",
        )
        if not patched:
            return (f"The date role was not approved: {fact_table}.{column} is "
                    "not in the semantic model. Rebuild the knowledge base, "
                    "then approve it on the Date Roles screen.")
        return ""
    except Exception as exc:  # noqa: BLE001 — report it, never block the accept
        log.warning(
            "draft_review: date role for %s.%s accepted but not approved in the "
            "semantic model for %s: %s", entity, column, account_id, exc,
        )
        return ("The date role was accepted but could not be approved in the "
                "semantic model. Approve it on the Date Roles screen.")


def publish_column_terms(account_id: str, entity: str, column: str,
                         synonyms: str) -> str:
    """Put an accepted column's words where the RESOLVER reads them.

    entity_properties.synonyms reaches core.graph_resolver, which decides which
    TABLE a question is about. It does not reach direct_aliases, which is what
    decides which COLUMN a measure name resolves to -- that comes from
    table_description.column_synonyms (core.vocab_packs.vocab_for_account).

    So an admin could accept exactly the words a reader used for a measure and
    the question that produced the proposal would still fail to resolve it.
    The drafter's whole premise is "these are the words people used"; a word
    that only helps pick the table is not what it promised.

    Merged, never replaced: a mapping document or the setup page may have put
    other terms on this column and other columns on this table, and an accept
    of one proposal must not drop any of them.

    Returns "" on success, or a sentence for the admin when the terms were
    saved on the property but could not be published.
    """
    import store

    terms = [t.strip() for t in str(synonyms or "").split(",") if t.strip()]
    if not terms:
        return ""
    try:
        table = _table_for_entity(account_id, entity)
        if not table:
            return (f"{entity} is not mapped to a table, so the terms were "
                    f"saved but nothing will resolve to them yet.")

        stored = store.list_table_descriptions(account_id) or {}
        key = _stored_key(stored, table)
        entry = stored.get(key) or {}
        merged = dict(entry.get("column_synonym_map") or {})
        existing = list(merged.get(str(column).upper(), []))
        seen = {t.casefold() for t in existing}
        for term in terms:
            if term.casefold() not in seen:
                existing.append(term)
                seen.add(term.casefold())
        # Upper because that is how parse_column_synonyms keys the map that
        # was just read. The store raises it again on the way in, so this is
        # for the dict in hand rather than for what lands on disk.
        merged[str(column).upper()] = existing
        store.save_table_description(
            account_id, key,
            description=str(entry.get("description") or ""),
            synonyms=str(entry.get("synonyms") or ""),
            column_synonyms=merged,
            updated_by="model_drafts",
        )
        # The vocabulary cache key is built from file mtimes, so a term saved
        # to the DATABASE changes nothing it watches.
        from core.vocab_packs import forget_account_vocab
        forget_account_vocab(account_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not publish terms for %s.%s on %s: %s",
                    entity, column, account_id, exc)
        return "The terms were saved but could not be published to the resolver."
    return ""


def _table_for_entity(account_id: str, entity: str) -> str:
    import store

    for row in store.list_entities(account_id, active_only=False):
        if str(row.get("entity_name") or "") != entity:
            continue
        table = str(row.get("table_name") or "").strip()
        schema = str(row.get("schema_name") or "").strip()
        return f"{schema}.{table}" if (schema and table) else table
    return ""


def _stored_key(stored: dict, table: str) -> str:
    """The key this table's terms are already filed under, or the new one.

    Descriptions are keyed by whatever the admin selected, which may be bare
    where the graph is qualified. Writing a second key would leave two rows for
    one table, and the one the admin edits would not be the one this wrote.
    """
    bare = table.split(".")[-1].upper()
    for key in stored:
        candidate = str(key).upper()
        if candidate == table.upper() or candidate.split(".")[-1] == bare:
            return str(key)
    return table
