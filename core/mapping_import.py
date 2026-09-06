"""Turn a parsed mapping document into a plan, then apply the plan.

core.mapping_csv reads the file and knows nothing about a workspace. This
resolves what it read against one -- tables to entities, columns to a
verdict -- and reports what WOULD happen before anything happens.

The dry run is the feature. An import that half-succeeds leaves a graph
nobody can reason about, and one that refuses everything because row 14 is
wrong makes the admin find row 14 themselves. So every row is resolved and
checked first, the caller is shown created / updated / unchanged / rejected
with a reason per rejected row, and only then does anything get written.

Additive, never a replace. The JSON import this sits beside swaps the whole
graph out; a mapping document adds and amends what it names and leaves the
rest alone, which is what makes it safe to send somebody a file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger("querybot.mapping_import")

CREATE = "create"
UPDATE = "update"
UNCHANGED = "unchanged"
REJECT = "reject"


@dataclass
class PlannedRow:
    action: str
    line: int
    target: str
    detail: str = ""
    payload: dict = field(default_factory=dict)
    existing_id: int = 0

    def as_dict(self) -> dict:
        return {"action": self.action, "line": self.line, "target": self.target,
                "detail": self.detail}


@dataclass
class ImportPlan:
    rows: list[PlannedRow] = field(default_factory=list)
    problems: list = field(default_factory=list)
    fatal: str = ""

    def of(self, action: str) -> list[PlannedRow]:
        return [row for row in self.rows if row.action == action]

    def summary(self) -> dict:
        # `flagged` counts rows that WILL be written and then marked unverified
        # -- a column the discovered schema does not have. They are not a
        # fourth action, they are creates and updates carrying a caveat, and
        # the admin has to see the caveat before applying rather than find it
        # on the graph afterwards.
        return {
            "created": len(self.of(CREATE)),
            "updated": len(self.of(UPDATE)),
            "unchanged": len(self.of(UNCHANGED)),
            "flagged": sum(1 for row in self.rows
                           if row.action in (CREATE, UPDATE)
                           and row.payload.get("_flag")),
            "rejected": len(self.of(REJECT)),
            "parse_problems": len(self.problems),
            "fatal": self.fatal,
        }

    def as_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "rows": [row.as_dict() for row in self.rows],
            "problems": [p.as_dict() for p in self.problems],
        }


def _workspace(account_id: str):
    """Entities keyed by name, a table→entity index, and the schema."""
    import store

    from core.schema import load_schema_columns

    entities = {row["entity_name"]: dict(row)
                for row in store.list_entities(account_id, active_only=False)}

    by_table: dict[str, str] = {}
    for name, entity in entities.items():
        table = str(entity.get("table_name") or "").strip().upper()
        schema = str(entity.get("schema_name") or "").strip().upper()
        if not table:
            continue
        # Every spelling a mapping document might use: bare, schema-qualified,
        # and database-qualified. A file that says SALES.F_ORDER and a graph
        # that stored schema SALES table F_ORDER have to meet.
        by_table.setdefault(table, name)
        if schema:
            by_table.setdefault(f"{schema}.{table}", name)

    schema_columns = None
    try:
        state = store.get_client_state(account_id) or {}
        schema_columns = load_schema_columns(str(state.get("schema_dir") or "")) or None
    except Exception as exc:  # noqa: BLE001
        log.warning("Mapping import for %s could not load the schema: %s",
                    account_id, exc)
    return entities, by_table, schema_columns


def _entity_for(table: str, by_table: dict[str, str]) -> str:
    """The entity a mapping document's table name refers to.

    Tries the name as written, then progressively less qualified: a file may
    say DB.SCHEMA.TABLE where the graph holds SCHEMA.TABLE, or the reverse.
    """
    parts = [part for part in str(table or "").upper().split(".") if part]
    for size in range(len(parts), 0, -1):
        found = by_table.get(".".join(parts[-size:]))
        if found:
            return found
    return ""


def plan_joins(account_id: str, parsed) -> ImportPlan:
    """What importing these join rows would do. Writes nothing."""
    from core.join_governance import check_join
    from core.mapping_csv import group_composites

    import store

    plan = ImportPlan(problems=list(getattr(parsed, "problems", [])),
                      fatal=getattr(parsed, "fatal", ""))
    if plan.fatal:
        return plan

    entities, by_table, schema_columns = _workspace(account_id)
    existing = store.list_relationships(account_id, active_only=False)

    for row, pairs in group_composites(list(getattr(parsed, "rows", []))):
        target = f"{row.from_table}.{row.from_column} → {row.to_table}.{row.to_column}"
        from_entity = _entity_for(row.from_table, by_table)
        to_entity = _entity_for(row.to_table, by_table)
        unknown = [table for table, resolved in
                   ((row.from_table, from_entity), (row.to_table, to_entity))
                   if not resolved]
        if unknown:
            plan.rows.append(PlannedRow(
                REJECT, row.line, target,
                f"Not a table in this workspace: {', '.join(unknown)}. "
                f"Add it to the graph, or check the schema qualifier."))
            continue

        candidate = {
            "from_entity": from_entity, "to_entity": to_entity,
            "from_column": pairs[0]["from_col"], "to_column": pairs[0]["to_col"],
            "relationship_type": row.relationship,
            "join_conditions": pairs[1:],
        }
        verdict = check_join(candidate, entities, schema_columns)
        if verdict.refuses:
            plan.rows.append(PlannedRow(REJECT, row.line, target, verdict.reason))
            continue

        payload = dict(candidate,
                       join_type=row.join_type, label=row.label,
                       _flag=bool(verdict.flags))
        match = _existing_join(existing, from_entity, to_entity, pairs)
        if match is None:
            plan.rows.append(PlannedRow(
                CREATE, row.line, target,
                verdict.reason if verdict.flags else "", payload))
        elif _same_join(match, payload):
            plan.rows.append(PlannedRow(
                UNCHANGED, row.line, target,
                "Already in the graph exactly as written.", payload,
                int(match.get("id") or 0)))
        else:
            plan.rows.append(PlannedRow(
                UPDATE, row.line, target,
                _difference(match, payload), payload, int(match.get("id") or 0)))
    return plan


def _existing_join(existing: list[dict], from_entity: str, to_entity: str,
                   pairs: list[dict]) -> dict | None:
    """The stored join this row is about, matched on the pair it joins on.

    On the COLUMNS, not on the two entities: a fact can join a dimension more
    than once (an order date and a delivery date both point at the calendar),
    and matching on the pair of entities alone would make a mapping document
    overwrite one role-playing join with another.
    """
    from core.join_governance import join_pairs

    wanted = {(p["from_col"].upper(), p["to_col"].upper()) for p in pairs}
    for row in existing:
        if str(row.get("from_entity") or "") != from_entity:
            continue
        if str(row.get("to_entity") or "") != to_entity:
            continue
        stored = {(a.upper(), b.upper()) for a, b in join_pairs(dict(row))}
        if stored == wanted:
            return dict(row)
    return None


def _same_join(stored: dict, payload: dict) -> bool:
    return (
        str(stored.get("relationship_type") or "many_to_one")
        == str(payload.get("relationship_type") or "many_to_one")
        and str(stored.get("join_type") or "").upper()
        == str(payload.get("join_type") or "").upper()
        and str(stored.get("label") or "") == str(payload.get("label") or "")
    )


def _difference(stored: dict, payload: dict) -> str:
    changes = []
    for field_name, key in (("relationship", "relationship_type"),
                            ("join type", "join_type"), ("label", "label")):
        before = str(stored.get(key) or "")
        after = str(payload.get(key) or "")
        if field_name == "join type":
            before, after = before.upper(), after.upper()
        if before != after:
            changes.append(f"{field_name} {before or '(blank)'} → {after or '(blank)'}")
    return "; ".join(changes)


def apply_joins(account_id: str, plan: ImportPlan) -> dict:
    """Write the plan. Only rows the plan marked create or update."""
    import store

    written = 0
    for row in plan.rows:
        if row.action not in (CREATE, UPDATE):
            continue
        payload = row.payload
        rel_id = store.save_relationship(
            account_id=account_id,
            from_entity=payload["from_entity"], to_entity=payload["to_entity"],
            from_column=payload["from_column"], to_column=payload["to_column"],
            relationship_type=payload["relationship_type"],
            join_type=payload["join_type"], label=payload["label"],
            join_conditions=payload.get("join_conditions") or [],
            rel_id=row.existing_id or 0,
            generated_by="mapping_csv",
            reason=f"imported from a mapping document, line {row.line}",
        )
        if payload.get("_flag"):
            try:
                store.update_relationship_validation(account_id, int(rel_id), "broken")
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not flag imported join %s: %s", rel_id, exc)
        written += 1
    return dict(plan.summary(), applied=written)


# ── Column terms ──────────────────────────────────────────────────────────────

def plan_terms(account_id: str, parsed) -> ImportPlan:
    """What importing these column-term rows would do. Writes nothing."""
    import store

    plan = ImportPlan(problems=list(getattr(parsed, "problems", [])),
                      fatal=getattr(parsed, "fatal", ""))
    if plan.fatal:
        return plan

    _, by_table, schema_columns = _workspace(account_id)
    stored = store.list_table_descriptions(account_id) or {}
    stored_upper = {str(name).upper(): entry for name, entry in stored.items()}

    for row in getattr(parsed, "rows", []):
        target = f"{row.table}.{row.column}"
        # Terms are keyed on the TABLE as the admin selected it, which may not
        # be a graph entity at all -- tables are chosen long before the graph
        # exists. So an unknown table is not a rejection here; an unknown
        # COLUMN is worth saying something about.
        entry = (stored_upper.get(str(row.table).upper())
                 or stored.get(row.table) or {})
        # The stored map is upper-keyed by parse_column_synonyms, so the
        # file's spelling of the column has to be raised to match it.
        current = list((entry.get("column_synonym_map") or {}).get(
            str(row.column).upper(), []))

        detail = ""
        if schema_columns is not None and not _column_exists(
                row.table, row.column, schema_columns):
            detail = ("Not found in the discovered schema — the terms will be "
                      "saved but nothing will resolve to them.")

        if [t.casefold() for t in current] == [t.casefold() for t in row.terms]:
            plan.rows.append(PlannedRow(UNCHANGED, row.line, target,
                                        "Already exactly these terms."))
            continue
        action = UPDATE if current else CREATE
        plan.rows.append(PlannedRow(
            action, row.line, target,
            (detail + " " if detail else "")
            + (f"{', '.join(current)} → {', '.join(row.terms) or '(cleared)'}"),
            {"table": row.table, "column": row.column, "terms": list(row.terms)}))
    return plan


def _column_exists(table: str, column: str, schema_columns: dict) -> bool:
    bare = str(table or "").upper().split(".")[-1]
    wanted = str(column or "").upper()
    for key, columns in (schema_columns or {}).items():
        candidate = str(key).upper()
        if candidate == str(table).upper() or candidate.split(".")[-1] == bare:
            return wanted in {str(c).upper() for c in (columns or {})}
    return False


def apply_terms(account_id: str, plan: ImportPlan) -> dict:
    """Write the planned column terms, merged per table.

    Merged rather than replaced: a mapping document that names three columns
    of a table must not silently drop the terms an admin typed for its other
    twenty.
    """
    import store

    by_table: dict[str, dict[str, list[str]]] = {}
    for row in plan.rows:
        if row.action not in (CREATE, UPDATE):
            continue
        payload = row.payload
        by_table.setdefault(payload["table"], {})[
            str(payload["column"]).upper()] = payload["terms"]

    stored = store.list_table_descriptions(account_id) or {}
    written = 0
    for table, columns in by_table.items():
        entry = stored.get(table) or {}
        merged = dict(entry.get("column_synonym_map") or {})
        for column, terms in columns.items():
            # The pop keeps `merged` honest in memory; the store drops an
            # empty list on the way back out either way, so nothing downstream
            # can tell the two apart. Written explicitly because a dict that
            # says a column has no terms and a dict that omits the column are
            # not the same thing to read.
            if terms:
                merged[column] = terms
            else:
                merged.pop(column, None)
        store.save_table_description(
            account_id, table,
            description=str(entry.get("description") or ""),
            synonyms=str(entry.get("synonyms") or ""),
            column_synonyms=merged,
            updated_by="mapping_csv",
        )
        written += len(columns)

    try:
        from core.vocab_packs import forget_account_vocab
        forget_account_vocab(account_id)
    except Exception as exc:  # noqa: BLE001
        # The vocabulary cache is keyed on file mtimes, so a term saved to the
        # DATABASE changes nothing it watches. Without this the import is the
        # "I saved it and nothing happened" failure it exists to prevent.
        log.warning("Could not refresh the vocabulary for %s: %s", account_id, exc)
    return dict(plan.summary(), applied=written)
